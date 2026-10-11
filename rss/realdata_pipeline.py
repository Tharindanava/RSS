"""Configuration-driven, limb-level RSS processing for IMU CSV files."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import os
import time
from typing import Any, Callable

import numpy as np

from rss.axes import source_axes_from_direction_cosines
from rss.mutual_information import MI_UNIT
from rss.imu_input import load_and_validate_input
from rss.quaternion_series import quaternion_to_axis_angle_series
from rss.realdata_config import FileRunConfig, RealDataRuntimeConfig, load_file_run_configs, resolve_torch_device
from rss.sensor_layout import SEGMENT_SENSOR_MAPPING
from rss.exporters import (
    export_limb_quaternion_axis_angle_csv,
    export_limb_quaternion_csv,
    export_limb_angle_mode_csv,
    export_limb_axes_mode_csv,
    export_limb_mode_coefficients_csv,
    export_limb_mode_metrics_csv,
    export_limb_observed_coefficients_csv,
    export_limb_observed_signals_csv,
    export_loss_history_artifacts,
    export_ssa_axis_plots,
    export_ssa_component_metrics_csv,
    export_ssa_drift_correction_signals_csv,
    export_ssa_preprocessing_signals_csv,
    realdata_output_dir,
    ssa_plot_dir,
)
from rss.solver import RSSResult, fit_rss
from rss.ssa_preprocessing import SSAPreprocessResult, apply_ssa_preprocessing


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class RealDataFileRunOutputs:
    csv_path: str
    output_dir: str
    limb_observed_signals_csv: str | None
    limb_observed_coefficients_csv: str | None
    limb_angle_mode_csv: str | None
    limb_mode_coefficients_csv: str | None
    limb_axes_mode_csv: str | None
    limb_quaternion_csv: str | None
    limb_quaternion_axis_angle_csv: str | None
    limb_mode_metrics_csv: str | None
    loss_history_csv: str | None
    loss_history_plot: str | None
    processed_limb_names: tuple[str, ...]
    skipped_limb_names: tuple[str, ...]
    file_elapsed_seconds: float
    ssa_preprocessing_csv: str | None = None
    ssa_drift_correction_csv: str | None = None
    ssa_component_metrics_csv: str | None = None
    ssa_artifacts_dir: str | None = None


@dataclass(frozen=True)
class LimbDecompositionResult:
    rss_result: RSSResult
    reconstructed_angles: np.ndarray
    reconstructed_quaternions: np.ndarray
    reconstructed_coefficients: np.ndarray
    mode_axes: np.ndarray
    original_angle: np.ndarray
    original_axes: np.ndarray
    angle_reconstruction_rmse: float
    coefficient_reconstruction_rmse: float


def _emit(callback: ProgressCallback | None, **payload: Any) -> None:
    if callback is not None:
        callback(payload)


def _available_limbs(dataframe, *, csv_path: str, allow_partial: bool) -> tuple[dict[str, str], tuple[str, ...]]:
    available: dict[str, str] = {}
    skipped: list[str] = []
    missing: list[str] = []
    for limb_name, device_name in SEGMENT_SENSOR_MAPPING.items():
        columns = tuple(f"{device_name}_G{axis}" for axis in ("X", "Y", "Z"))
        absent = [column for column in columns if column not in dataframe.columns]
        if absent:
            skipped.append(limb_name)
            missing.extend(absent)
        else:
            available[limb_name] = device_name
    if skipped and not allow_partial:
        raise ValueError(f"Input CSV is missing mapped limb gyroscope columns. Missing: {', '.join(missing)}")
    if not available:
        raise ValueError(f"Input CSV has no usable mapped limbs in {csv_path}.")
    return available, tuple(skipped)


def _validate_parallel_devices(file_configs: list[FileRunConfig]) -> None:
    values = tuple(str(config.compute_device).strip().lower() for config in file_configs)
    if any(value == "cuda" or value.startswith("cuda:") for value in values):
        raise ValueError("Parallel file processing supports CPU devices only.")
    if "auto" in values:
        try:
            import torch

            if torch.cuda.is_available():
                raise ValueError("Use explicit compute_device='cpu' for parallel processing when CUDA is available.")
        except ImportError:
            pass


def _repair_nonfinite(values: np.ndarray, *, device_name: str, csv_path: str) -> tuple[np.ndarray, tuple[int, ...]]:
    array = np.asarray(values, dtype=float).reshape(-1, 3).copy()
    invalid = ~np.isfinite(array)
    repaired = tuple(int(value) for value in np.where(invalid.any(axis=1))[0])
    sample_index = np.arange(array.shape[0], dtype=float)
    for axis_index in range(3):
        column = array[:, axis_index]
        finite = np.isfinite(column)
        if finite.all():
            continue
        if not finite.any():
            raise ValueError(f"Cannot repair {device_name} gyro axis {axis_index} in {csv_path}: all values are non-finite.")
        if int(finite.sum()) == 1:
            column[~finite] = column[finite][0]
        else:
            column[~finite] = np.interp(sample_index[~finite], sample_index[finite], column[finite])
    return array, repaired


def _integrate_gyro(dataframe, device_name: str, *, sample_rate_hz: float, csv_path: str) -> tuple[np.ndarray, tuple[int, ...]]:
    gyro = dataframe[[f"{device_name}_GX", f"{device_name}_GY", f"{device_name}_GZ"]].to_numpy(dtype=float)
    gyro, repaired = _repair_nonfinite(gyro, device_name=device_name, csv_path=csv_path)
    return (np.cumsum(np.radians(gyro), axis=0) / float(sample_rate_hz)).T, repaired


def _rss_mode_order(source_angles: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(source_angles), axis=1))
    return np.argsort(rms)[::-1]


def _decompose(observations: np.ndarray, *, runtime: RealDataRuntimeConfig, device: str) -> LimbDecompositionResult:
    result = fit_rss(
        observations,
        replace(runtime.rss_config, device=device),
        sample_interval_seconds=1.0 / float(runtime.sample_rate_hz),
    )
    reconstructed_coefficients = result.reconstructed_observer_coefficients
    reconstructed_angles = result.reconstructed_observer_angles
    reconstructed_quaternions = result.composed_source_quaternions
    original_angle, original_axes = quaternion_to_axis_angle_series(reconstructed_quaternions)
    return LimbDecompositionResult(
        rss_result=result,
        reconstructed_angles=reconstructed_angles,
        reconstructed_quaternions=reconstructed_quaternions,
        reconstructed_coefficients=reconstructed_coefficients,
        mode_axes=source_axes_from_direction_cosines(result.recovery_matrix),
        original_angle=original_angle,
        original_axes=original_axes,
        angle_reconstruction_rmse=float(np.sqrt(np.mean((reconstructed_angles - observations) ** 2))),
        coefficient_reconstruction_rmse=float(
            np.sqrt(np.mean((reconstructed_coefficients - result.observer_coefficients) ** 2))
        ),
    )


def _metric_rows(
    limb_name: str,
    result: LimbDecompositionResult,
    mode_order: np.ndarray | None = None,
) -> list[dict[str, object]]:
    rss = result.rss_result
    source_angles = rss.source_angles if mode_order is None else rss.source_angles[mode_order]
    mode_axes = result.mode_axes if mode_order is None else result.mode_axes[mode_order]
    best_history = rss.loss_history[rss.best_epoch]
    applied_learning_rates = rss.applied_learning_rates
    warning_status = ";".join(rss.matrix_diagnostics.warnings)
    rows: list[dict[str, object]] = []
    for mode_index in range(3):
        signal = source_angles[mode_index]
        axis = mode_axes[mode_index]
        row = {
                "limb_name": limb_name,
                "mode_index": mode_index + 1,
                "rss_optimizer": rss.optimizer_name,
                "rss_loss_type": rss.config.loss_type,
                "rss_mi_estimator": rss.config.mi_estimator,
                "rss_mi_unit": MI_UNIT,
                "rss_gcmi_rank_method": rss.config.gcmi_rank_method,
                "rss_bspline_bins": rss.config.bspline_bins,
                "rss_bspline_order": rss.config.bspline_order,
                "rss_ksg_k": rss.config.ksg_k,
                "rss_ksg_standardize": rss.config.ksg_standardize,
                "rss_ksg_max_samples": rss.config.ksg_max_samples,
                "rss_ksg_subsample_mode": rss.config.ksg_subsample_mode,
                "rss_ksg_subsample_seed": rss.config.ksg_subsample_seed,
                "rss_sample_interval_seconds": rss.sample_interval_seconds,
                "rss_learning_rate_start": rss.config.learning_rate,
                "rss_learning_rate_end": rss.config.learning_rate_end,
                "rss_first_applied_learning_rate": (
                    applied_learning_rates[0] if applied_learning_rates else None
                ),
                "rss_terminal_applied_learning_rate": (
                    applied_learning_rates[-1] if applied_learning_rates else None
                ),
                "axis_x": axis[0],
                "axis_y": axis[1],
                "axis_z": axis[2],
                "mode_rms": float(np.sqrt(np.mean(signal * signal))),
                "mode_variance": float(np.var(signal)),
                "mode_mean": float(np.mean(signal)),
                "rss_loss_normalization_reference": rss.loss_normalization_reference,
                "rss_normalized_initial_loss": rss.normalized_initial_loss,
                "rss_normalized_best_loss": rss.normalized_best_loss,
                "rss_best_normalized_separation_loss": best_history["normalized_separation_loss"],
                "rss_best_normalized_orthogonality_loss": best_history["normalized_orthogonality_loss"],
                "rss_best_normalized_reconstruction_loss": best_history["normalized_reconstruction_loss"],
                "rss_normalized_terminal_loss": rss.normalized_terminal_loss,
                "rss_initial_loss": rss.initial_loss,
                "rss_best_loss": rss.best_loss,
                "rss_best_separation_loss": best_history["separation_loss"],
                "rss_best_orthogonality_loss": best_history["orthogonality_loss"],
                "rss_best_reconstruction_loss": best_history["reconstruction_loss"],
                "rss_terminal_loss": rss.terminal_loss,
                "rss_best_epoch": rss.best_epoch,
                "rss_stop_reason": rss.stop_reason,
                "reconstruction_rmse": result.angle_reconstruction_rmse,
                "coefficient_reconstruction_rmse": result.coefficient_reconstruction_rmse,
                "quaternion_reconstruction_mean_error_deg": float(np.degrees(rss.reconstruction_mean_error_rad)),
                "quaternion_reconstruction_max_error_deg": float(np.degrees(rss.reconstruction_max_error_rad)),
                "matrix_frobenius_norm": rss.matrix_diagnostics.frobenius_norm,
                "matrix_condition_number": rss.matrix_diagnostics.condition_number,
                "matrix_symmetry_error": rss.matrix_diagnostics.symmetry_error,
                "matrix_orthogonality_error": rss.matrix_diagnostics.orthogonality_error,
                "matrix_determinant": rss.matrix_diagnostics.determinant,
                "matrix_warning_status": warning_status,
            }
        if rss.config.adaptive_regularizer or rss.config.axis_flip_search:
            row.update(
                {
                    "rss_adaptive_regularizer": rss.config.adaptive_regularizer,
                    "rss_specialized_regularizer": rss.config.specialized_regularizer,
                    "rss_regularizer_start_weight": rss.config.regularizer_start_weight,
                    "rss_axis_flip_search": rss.config.axis_flip_search,
                    "rss_axis_flip_check_count": len(rss.axis_flip_events),
                    "rss_axis_flip_selected_patterns": ";".join(
                        event.selected_pattern for event in rss.axis_flip_events
                    ),
                    "rss_regularizer_min_loss_threshold": (
                        rss.config.regularizer_min_loss_threshold
                    ),
                    "rss_selection_loss_normalization_reference": (
                        rss.selection_loss_normalization_reference
                    ),
                    "rss_initial_selection_loss": rss.initial_selection_loss,
                    "rss_best_selection_loss": rss.best_selection_loss,
                    "rss_normalized_best_selection_loss": (
                        rss.normalized_best_selection_loss
                    ),
                    "rss_best_regularizer_loss": best_history.get("regularizer_loss"),
                    "rss_best_regularizer_weight": best_history.get(
                        "regularizer_weight"
                    ),
                    "rss_best_regularizer_phase": best_history.get(
                        "regularizer_phase"
                    ),
                    "rss_terminal_regularizer_loss": rss.loss_history[-1].get(
                        "regularizer_loss"
                    ),
                    "rss_terminal_regularizer_weight": rss.loss_history[-1].get(
                        "regularizer_weight"
                    ),
                    "rss_terminal_regularizer_phase": rss.loss_history[-1].get(
                        "regularizer_phase"
                    ),
                }
            )
        rows.append(row)
    return rows


def run_realdata_one_file(
    *,
    file_cfg: FileRunConfig,
    runtime_cfg: RealDataRuntimeConfig,
    progress_callback: ProgressCallback | None = None,
    file_index: int | None = None,
    file_total: int | None = None,
) -> RealDataFileRunOutputs:
    started = time.perf_counter()
    _emit(progress_callback, event_type="file_stage", file_index=file_index, file_total=file_total, csv_path=file_cfg.csv_path, stage_name="load_input")
    loaded = load_and_validate_input(
        file_cfg.csv_path,
        sample_rate_hz=runtime_cfg.sample_rate_hz,
        include_time_seconds=runtime_cfg.include_time_seconds,
        allow_partial_imu=runtime_cfg.allow_partial_limbs,
    )
    mapping, skipped = _available_limbs(loaded.dataframe, csv_path=file_cfg.csv_path, allow_partial=runtime_cfg.allow_partial_limbs)
    _emit(
        progress_callback,
        event_type="file_limb_availability",
        file_index=file_index,
        file_total=file_total,
        csv_path=file_cfg.csv_path,
        processed_limb_names=list(mapping),
        skipped_limb_names=list(skipped),
    )
    device = str(resolve_torch_device(file_cfg.compute_device))
    enabled = set(runtime_cfg.enabled_outputs)
    output_dir = realdata_output_dir(runtime_cfg.output_root, file_cfg.csv_path)
    ssa_directory = ssa_plot_dir(output_dir) if runtime_cfg.ssa_enabled and "ssa_plots" in enabled else None

    observed: dict[str, np.ndarray] = {}
    reconstructed: dict[str, np.ndarray] = {}
    observed_q: dict[str, np.ndarray] = {}
    observed_coeff: dict[str, np.ndarray] = {}
    reconstructed_q: dict[str, np.ndarray] = {}
    reconstructed_coeff: dict[str, np.ndarray] = {}
    sources: dict[str, np.ndarray] = {}
    source_q: dict[str, np.ndarray] = {}
    source_coeff: dict[str, np.ndarray] = {}
    mode_axes: dict[str, np.ndarray] = {}
    original_angles: dict[str, np.ndarray] = {}
    original_axes: dict[str, np.ndarray] = {}
    loss_histories: dict[str, tuple[dict[str, float | str | None], ...]] = {}
    metrics: list[dict[str, object]] = []
    ssa_results: dict[str, dict[str, SSAPreprocessResult]] = {}
    ssa_component_rows: list[dict[str, object]] = []

    for limb_index, (limb_name, device_name) in enumerate(mapping.items(), start=1):
        limb_observations, repaired = _integrate_gyro(
            loaded.dataframe,
            device_name,
            sample_rate_hz=runtime_cfg.sample_rate_hz,
            csv_path=file_cfg.csv_path,
        )
        _emit(
            progress_callback,
            event_type="file_limb_progress",
            file_index=file_index,
            file_total=file_total,
            csv_path=file_cfg.csv_path,
            limb_index=limb_index,
            limb_total=len(mapping),
            limb_name=limb_name,
            repaired_row_count=len(repaired),
        )
        if runtime_cfg.ssa_enabled:
            axis_results: dict[str, SSAPreprocessResult] = {}
            cleaned = np.zeros_like(limb_observations)
            for axis_index, axis_name in enumerate(("x", "y", "z")):
                axis_result = apply_ssa_preprocessing(
                    limb_observations[axis_index],
                    sample_rate_hz=runtime_cfg.sample_rate_hz,
                    requested_window_length=runtime_cfg.ssa_window_length,
                    requested_reconstruction_components=runtime_cfg.ssa_reconstruction_components,
                    drift_zero_crossing_limit=runtime_cfg.ssa_drift_zero_crossing_limit,
                    drift_turning_point_limit=runtime_cfg.ssa_drift_turning_point_limit,
                )
                cleaned[axis_index] = axis_result.cleaned_signal
                axis_results[axis_name] = axis_result
                for summary in axis_result.component_summaries:
                    ssa_component_rows.append(
                        {
                            "limb_name": limb_name,
                            "axis": axis_name,
                            "window_length": axis_result.window_length,
                            "component_index": summary.component_index,
                            "eigenvalue": summary.eigenvalue,
                            "variance_share_pct": summary.variance_share_pct,
                            "zero_crossings": summary.zero_crossings,
                            "turning_points": summary.turning_points,
                            "drift_zero_crossing_limit": runtime_cfg.ssa_drift_zero_crossing_limit,
                            "drift_turning_point_limit": runtime_cfg.ssa_drift_turning_point_limit,
                            "classification": summary.classification,
                            "kept_in_reconstruction": summary.kept_in_reconstruction,
                        }
                    )
                if ssa_directory is not None:
                    export_ssa_axis_plots(plot_dir=ssa_directory, limb_name=limb_name, axis_name=axis_name, result=axis_result)
            limb_observations = cleaned
            ssa_results[limb_name] = axis_results
        decomposed = _decompose(limb_observations, runtime=runtime_cfg, device=device)
        rss = decomposed.rss_result
        if runtime_cfg.rss_mode_order == "default":
            mode_order = None
        elif runtime_cfg.rss_mode_order == "amplitude":
            mode_order = _rss_mode_order(rss.source_angles)
        else:
            raise ValueError(f"Unknown RSS mode ordering: {runtime_cfg.rss_mode_order}")
        observed[limb_name] = rss.observer_angles
        reconstructed[limb_name] = decomposed.reconstructed_angles
        observed_q[limb_name] = rss.observer_quaternions
        observed_coeff[limb_name] = rss.observer_coefficients
        sources[limb_name] = rss.source_angles if mode_order is None else rss.source_angles[mode_order]
        source_q[limb_name] = rss.source_quaternions if mode_order is None else rss.source_quaternions[mode_order]
        source_coeff[limb_name] = rss.source_coefficients if mode_order is None else rss.source_coefficients[mode_order]
        mode_axes[limb_name] = decomposed.mode_axes if mode_order is None else decomposed.mode_axes[mode_order]
        original_angles[limb_name] = decomposed.original_angle
        original_axes[limb_name] = decomposed.original_axes
        loss_histories[limb_name] = rss.loss_history
        metrics.extend(_metric_rows(limb_name, decomposed, mode_order=mode_order))

    frame = loaded.frame_index
    times = loaded.time_seconds
    observed_csv = export_limb_observed_signals_csv(output_dir=output_dir, frame_index=frame, time_seconds=times, observed_by_limb=observed, reconstructed_by_limb=reconstructed) if "limb_observed_signals" in enabled else None
    observed_coeff_csv = export_limb_observed_coefficients_csv(output_dir=output_dir, frame_index=frame, time_seconds=times, observed_quaternions_by_limb=observed_q, observed_coefficients_by_limb=observed_coeff, reconstructed_quaternions_by_limb=reconstructed_q, reconstructed_coefficients_by_limb=reconstructed_coeff) if "limb_observed_coefficients" in enabled else None
    angle_csv = export_limb_angle_mode_csv(output_dir=output_dir, frame_index=frame, time_seconds=times, sources_by_limb=sources, original_angles_by_limb=original_angles) if "limb_angle_modes" in enabled else None
    coefficient_csv = export_limb_mode_coefficients_csv(output_dir=output_dir, frame_index=frame, time_seconds=times, source_coefficients_by_limb=source_coeff, source_quaternions_by_limb=source_q) if "limb_mode_coefficients" in enabled else None
    axes_csv = export_limb_axes_mode_csv(output_dir=output_dir, frame_index=frame, time_seconds=times, mode_axes_by_limb=mode_axes, original_axes_by_limb=original_axes) if "limb_axis_modes" in enabled else None
    quaternion_csv = export_limb_quaternion_csv(output_dir=output_dir, frame_index=frame, time_seconds=times, limb_quaternions=reconstructed_q) if "limb_quaternions" in enabled else None
    quaternion_axis_csv = export_limb_quaternion_axis_angle_csv(output_dir=output_dir, frame_index=frame, time_seconds=times, limb_quaternions=reconstructed_q) if "limb_quaternions" in enabled else None
    metrics_csv = export_limb_mode_metrics_csv(output_dir=output_dir, metric_rows=metrics) if "limb_mode_metrics" in enabled else None
    loss_csv, loss_plot = export_loss_history_artifacts(output_dir=output_dir, loss_history_by_limb=loss_histories) if "loss_history" in enabled else (None, None)
    ssa_pre_csv = export_ssa_preprocessing_signals_csv(output_dir=output_dir, frame_index=frame, time_seconds=times, ssa_by_limb_axis=ssa_results) if runtime_cfg.ssa_enabled and "ssa_preprocessing_signals" in enabled else None
    ssa_drift_csv = export_ssa_drift_correction_signals_csv(output_dir=output_dir, frame_index=frame, time_seconds=times, ssa_by_limb_axis=ssa_results) if runtime_cfg.ssa_enabled and "ssa_drift_correction_signals" in enabled else None
    ssa_metrics_csv = export_ssa_component_metrics_csv(output_dir=output_dir, component_rows=ssa_component_rows) if runtime_cfg.ssa_enabled and "ssa_component_metrics" in enabled else None
    return RealDataFileRunOutputs(
        csv_path=file_cfg.csv_path,
        output_dir=output_dir,
        limb_observed_signals_csv=observed_csv,
        limb_observed_coefficients_csv=observed_coeff_csv,
        limb_angle_mode_csv=angle_csv,
        limb_mode_coefficients_csv=coefficient_csv,
        limb_axes_mode_csv=axes_csv,
        limb_quaternion_csv=quaternion_csv,
        limb_quaternion_axis_angle_csv=quaternion_axis_csv,
        limb_mode_metrics_csv=metrics_csv,
        loss_history_csv=loss_csv,
        loss_history_plot=loss_plot,
        processed_limb_names=tuple(mapping),
        skipped_limb_names=skipped,
        file_elapsed_seconds=float(time.perf_counter() - started),
        ssa_preprocessing_csv=ssa_pre_csv,
        ssa_drift_correction_csv=ssa_drift_csv,
        ssa_component_metrics_csv=ssa_metrics_csv,
        ssa_artifacts_dir=ssa_directory,
    )


def _worker(file_cfg: FileRunConfig, runtime_cfg: RealDataRuntimeConfig) -> RealDataFileRunOutputs:
    return run_realdata_one_file(file_cfg=file_cfg, runtime_cfg=runtime_cfg)


def run_realdata_from_config(
    *,
    config_path: str,
    runtime_cfg: RealDataRuntimeConfig,
    progress_callback: ProgressCallback | None = None,
) -> list[RealDataFileRunOutputs]:
    files = load_file_run_configs(config_path)
    os.makedirs(runtime_cfg.output_root, exist_ok=True)
    if runtime_cfg.file_workers > 1:
        _validate_parallel_devices(files)
    _emit(progress_callback, event_type="run_start", file_total=len(files))
    if runtime_cfg.file_workers == 1:
        outputs = []
        for index, config in enumerate(files, start=1):
            _emit(progress_callback, event_type="file_start", file_index=index, file_total=len(files), csv_path=config.csv_path)
            try:
                output = run_realdata_one_file(file_cfg=config, runtime_cfg=runtime_cfg, progress_callback=progress_callback, file_index=index, file_total=len(files))
            except Exception as exc:
                _emit(progress_callback, event_type="run_error", file_index=index, file_total=len(files), csv_path=config.csv_path, message=str(exc))
                raise
            outputs.append(output)
            _emit(progress_callback, event_type="file_done", file_index=index, file_total=len(files), csv_path=config.csv_path, file_elapsed_seconds=output.file_elapsed_seconds)
        _emit(progress_callback, event_type="run_done", file_total=len(files))
        return outputs
    indexed: dict[int, RealDataFileRunOutputs] = {}
    with ProcessPoolExecutor(max_workers=runtime_cfg.file_workers) as pool:
        futures = {pool.submit(_worker, config, runtime_cfg): index for index, config in enumerate(files)}
        for index, config in enumerate(files, start=1):
            _emit(progress_callback, event_type="file_start", file_index=index, file_total=len(files), csv_path=config.csv_path)
        for future in as_completed(futures):
            zero_index = futures[future]
            try:
                indexed[zero_index] = future.result()
            except Exception as exc:
                _emit(progress_callback, event_type="run_error", file_index=zero_index + 1, file_total=len(files), csv_path=files[zero_index].csv_path, message=str(exc))
                for pending in futures:
                    pending.cancel()
                raise
            _emit(
                progress_callback,
                event_type="file_done",
                file_index=zero_index + 1,
                file_total=len(files),
                csv_path=files[zero_index].csv_path,
                file_elapsed_seconds=indexed[zero_index].file_elapsed_seconds,
            )
    outputs = [indexed[index] for index in range(len(files))]
    _emit(progress_callback, event_type="run_done", file_total=len(files))
    return outputs
