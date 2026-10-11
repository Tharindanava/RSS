"""Ground-truth evaluation for direct observer-signal BSS baselines."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd

from bss_baselines.core import BSSConfig, BSSFitResult, fit_bss
from rss.evaluation import (
    MatchResult,
    ValidationResult,
    build_metrics_frame,
    build_source_metrics_frame,
    match_estimated_sources,
)
from rss.model import (
    compose_source_quaternions,
    quaternion_error_angles,
    quaternions_to_angles,
    quaternions_to_integrated_body_angles,
    relative_reconstruction_quaternions,
)
from rss.simulation import ScenarioData, source_cross_correlation_matrix


@dataclass(frozen=True)
class BSSValidationResult:
    """One fitted BSS baseline aligned to simulation truth for evaluation."""

    scenario: ScenarioData
    fit_result: BSSFitResult
    match: MatchResult
    aligned_axes_observer_frame: np.ndarray
    oracle_scaled_sources: np.ndarray
    oracle_scales: np.ndarray
    oracle_source_metrics: dict[str, np.ndarray]
    original_source_cross_correlation_matrix: np.ndarray
    separated_source_cross_correlation_matrix: np.ndarray
    oracle_source_cross_correlation_matrix: np.ndarray
    physical_composed_quaternions: np.ndarray
    oracle_composed_quaternions: np.ndarray
    physical_reconstructed_observations: np.ndarray
    oracle_reconstructed_observations: np.ndarray
    metrics: dict[str, object]


def _cross_correlation_metrics(prefix: str, matrix: np.ndarray) -> dict[str, float]:
    values = np.asarray(matrix, dtype=float)
    pairs = {
        f"{first + 1}{second + 1}": abs(float(values[first, second]))
        for first in range(values.shape[0])
        for second in range(first + 1, values.shape[0])
    }
    pair_values = np.asarray(tuple(pairs.values()), dtype=float)
    metrics = {f"{prefix}_abs_cross_correlation_{key}": value for key, value in pairs.items()}
    metrics[f"{prefix}_mean_abs_cross_correlation"] = (
        float(np.mean(pair_values)) if pair_values.size else float("nan")
    )
    metrics[f"{prefix}_max_abs_cross_correlation"] = (
        float(np.max(pair_values)) if pair_values.size else float("nan")
    )
    return metrics


def _source_error_metrics(
    truth: np.ndarray,
    estimate: np.ndarray,
    active_source_indices: tuple[int, ...],
) -> dict[str, np.ndarray]:
    true_values = np.asarray(truth, dtype=float).reshape(3, -1)
    estimates = np.asarray(estimate, dtype=float).reshape(3, -1)
    active = np.asarray(active_source_indices, dtype=int)
    arrays = {
        name: np.full(3, np.nan, dtype=float)
        for name in (
            "mse",
            "rmse",
            "mae",
            "max_abs_error",
            "relative_rmse",
            "r2",
            "snr_db",
        )
    }
    errors = estimates[active] - true_values[active]
    arrays["mse"][active] = np.mean(np.square(errors), axis=1)
    arrays["rmse"][active] = np.sqrt(arrays["mse"][active])
    arrays["mae"][active] = np.mean(np.abs(errors), axis=1)
    arrays["max_abs_error"][active] = np.max(np.abs(errors), axis=1)
    floor = np.finfo(float).eps
    truth_rms = np.sqrt(np.mean(np.square(true_values[active]), axis=1))
    arrays["relative_rmse"][active] = arrays["rmse"][active] / np.maximum(truth_rms, floor)
    centered_truth = true_values[active] - np.mean(true_values[active], axis=1, keepdims=True)
    total_squares = np.sum(np.square(centered_truth), axis=1)
    residual_squares = np.sum(np.square(errors), axis=1)
    arrays["r2"][active] = 1.0 - residual_squares / np.maximum(total_squares, floor)
    signal_power = np.mean(np.square(true_values[active]), axis=1)
    arrays["snr_db"][active] = 10.0 * np.log10(
        np.maximum(signal_power, floor) / np.maximum(arrays["mse"][active], floor)
    )
    return arrays


def _oracle_scale_sources(
    truth: np.ndarray,
    estimate: np.ndarray,
    active_source_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    true_values = np.asarray(truth, dtype=float).reshape(3, -1)
    estimates = np.asarray(estimate, dtype=float).reshape(3, -1)
    scaled = estimates.copy()
    scales = np.ones(3, dtype=float)
    for source_index in active_source_indices:
        denominator = float(np.dot(estimates[source_index], estimates[source_index]))
        scales[source_index] = (
            0.0
            if denominator <= np.finfo(float).eps
            else float(np.dot(true_values[source_index], estimates[source_index]) / denominator)
        )
        scaled[source_index] = scales[source_index] * estimates[source_index]
    return scaled, scales, _source_error_metrics(true_values, scaled, active_source_indices)


def _quaternion_metrics(
    scenario: ScenarioData,
    source_angles: np.ndarray,
    axes_observer_frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    composed = compose_source_quaternions(source_angles, axes_observer_frame)
    relative = relative_reconstruction_quaternions(
        composed,
        scenario.physical_observer_quaternions,
    )
    identity = np.zeros_like(relative)
    identity[:, 0] = 1.0
    loss = float(
        np.mean(
            np.minimum(
                np.sum(np.square(relative - identity), axis=1),
                np.sum(np.square(relative + identity), axis=1),
            )
        )
    )
    errors = quaternion_error_angles(relative)
    if scenario.observation_model == "body_fixed_gyro":
        reconstructed = quaternions_to_integrated_body_angles(composed)
    else:
        reconstructed = quaternions_to_angles(composed)
    return composed, reconstructed, {
        "quaternion_reconstruction_loss": loss,
        "quaternion_reconstruction_mean_error_deg": float(np.degrees(np.mean(errors))),
        "quaternion_reconstruction_max_error_deg": float(np.degrees(np.max(errors))),
        "observation_reconstruction_rmse": float(
            np.sqrt(np.mean(np.square(reconstructed - scenario.observations)))
        ),
    }


def _mean_error_metrics(prefix: str, metrics: dict[str, np.ndarray]) -> dict[str, float]:
    suffix = f"_{prefix}" if prefix else ""
    return {
        f"mean_source_mse{suffix}": float(np.nanmean(metrics["mse"])),
        f"mean_source_rmse{suffix}": float(np.nanmean(metrics["rmse"])),
        f"mean_source_mae{suffix}": float(np.nanmean(metrics["mae"])),
        f"max_source_abs_error{suffix}": float(np.nanmax(metrics["max_abs_error"])),
        f"mean_source_relative_rmse{suffix}": float(np.nanmean(metrics["relative_rmse"])),
        f"mean_source_r2{suffix}": float(np.nanmean(metrics["r2"])),
        f"mean_source_snr_db{suffix}": float(np.nanmean(metrics["snr_db"])),
    }


def evaluate_bss_recovery(
    scenario: ScenarioData,
    fit_result: BSSFitResult,
) -> BSSValidationResult:
    """Evaluate one blind fit; truth is used only inside this evaluation stage."""

    match = match_estimated_sources(
        scenario.sources,
        fit_result.source_angles,
        fit_result.normalized_demixing_matrix,
        scenario.source_axes_global,
        scenario.observer_axes_global,
        scenario.active_source_indices,
    )
    permutation = np.asarray(match.permutation, dtype=int)
    aligned_axes_observer = (
        fit_result.normalized_demixing_matrix[permutation] * match.signs[:, None]
    )
    oracle_sources, oracle_scales, oracle_source_metrics = _oracle_scale_sources(
        scenario.sources,
        match.estimated_sources_matched,
        match.active_source_indices,
    )
    active = np.asarray(match.active_source_indices, dtype=int)
    original_correlation = source_cross_correlation_matrix(scenario.sources[active])
    separated_correlation = source_cross_correlation_matrix(
        match.estimated_sources_matched[active]
    )
    oracle_correlation = source_cross_correlation_matrix(oracle_sources[active])

    physical_quaternions, physical_observers, physical_quaternion_metrics = _quaternion_metrics(
        scenario,
        match.estimated_sources_matched,
        aligned_axes_observer,
    )
    oracle_quaternions, oracle_observers, oracle_quaternion_metrics = _quaternion_metrics(
        scenario,
        oracle_sources,
        aligned_axes_observer,
    )
    physical_source_metrics = {
        "mse": match.source_mses,
        "rmse": match.source_rmses,
        "mae": match.source_maes,
        "max_abs_error": match.source_max_abs_errors,
        "relative_rmse": match.source_relative_rmses,
        "r2": match.source_r2_scores,
        "snr_db": match.source_snr_db,
    }
    centered_covariance = np.cov(fit_result.centered_observations)
    separated_covariance = np.cov(fit_result.source_angles)
    centered_offdiag = centered_covariance - np.diag(np.diag(centered_covariance))
    separated_offdiag = separated_covariance - np.diag(np.diag(separated_covariance))
    aligned_matrix = aligned_axes_observer
    metrics: dict[str, object] = {
        "method_key": fit_result.method_key,
        "method": fit_result.method_label,
        "implementation": fit_result.implementation,
        "input_domain": "cumulative_observer_signals",
        "converged": bool(fit_result.converged),
        "iterations": int(fit_result.iterations),
        "runtime_seconds": float(fit_result.runtime_seconds),
        "warnings": " | ".join(fit_result.warnings),
        "active_source_count": float(len(match.active_source_indices)),
        "sparse_source_count": float(len(scenario.sparse_source_indices)),
        "mean_abs_source_correlation": match.mean_abs_correlation,
        **_mean_error_metrics("", physical_source_metrics),
        **_mean_error_metrics("oracle_scaled", oracle_source_metrics),
        "mean_axis_error_deg": float(np.degrees(np.nanmean(match.axis_errors_rad))),
        "linear_observation_reconstruction_rmse": float(
            np.sqrt(
                np.mean(
                    np.square(fit_result.reconstructed_observations - scenario.observations)
                )
            )
        ),
        **physical_quaternion_metrics,
        **{
            f"{name}_oracle_scaled": value
            for name, value in oracle_quaternion_metrics.items()
        },
        **_cross_correlation_metrics("original_source", original_correlation),
        **_cross_correlation_metrics("separated_source", separated_correlation),
        **_cross_correlation_metrics("oracle_scaled_source", oracle_correlation),
        "offdiag_covariance_energy_before": float(np.sum(np.square(centered_offdiag))),
        "offdiag_covariance_energy_after": float(np.sum(np.square(separated_offdiag))),
        "demixing_condition_number": float(np.linalg.cond(fit_result.demixing_matrix)),
        "normalized_demixing_orthogonality_error": float(
            np.linalg.norm(
                fit_result.normalized_demixing_matrix
                @ fit_result.normalized_demixing_matrix.T
                - np.eye(3),
                ord="fro",
            )
        ),
        "normalized_demixing_determinant": float(
            np.linalg.det(fit_result.normalized_demixing_matrix)
        ),
        "aligned_recovery_matrix_rmse": float(
            np.sqrt(np.mean(np.square(aligned_matrix - scenario.mixing_matrix)))
        ),
        "oracle_scales": oracle_scales.tolist(),
    }
    return BSSValidationResult(
        scenario=scenario,
        fit_result=fit_result,
        match=match,
        aligned_axes_observer_frame=aligned_axes_observer,
        oracle_scaled_sources=oracle_sources,
        oracle_scales=oracle_scales,
        oracle_source_metrics=oracle_source_metrics,
        original_source_cross_correlation_matrix=original_correlation,
        separated_source_cross_correlation_matrix=separated_correlation,
        oracle_source_cross_correlation_matrix=oracle_correlation,
        physical_composed_quaternions=physical_quaternions,
        oracle_composed_quaternions=oracle_quaternions,
        physical_reconstructed_observations=physical_observers,
        oracle_reconstructed_observations=oracle_observers,
        metrics=metrics,
    )


def run_bss_methods_for_scenario(
    scenario: ScenarioData,
    config: BSSConfig | None = None,
) -> list[BSSValidationResult]:
    """Fit every configured method independently to the same scenario observations."""

    active = config or BSSConfig()
    return [
        evaluate_bss_recovery(scenario, fit_bss(scenario.observations, method, active))
        for method in active.methods
    ]


def _scenario_row(scenario: ScenarioData) -> dict[str, object]:
    return {
        "family": scenario.family,
        "case_id": scenario.case_id,
        "seed": scenario.config.seed,
        "frame_count": scenario.config.T,
        "amplitude_tier": scenario.amplitude_tier,
        "stress_peak_angle_rad": scenario.stress_peak_angle_rad,
        "composition": scenario.composition,
        "replicate_index": scenario.replicate_index,
        "source_roles": "|".join(scenario.source_roles),
        "simulation_observation_model": scenario.observation_model,
    }


def build_bss_metrics_frame(results: Sequence[BSSValidationResult]) -> pd.DataFrame:
    """Build one method/case row for BSS comparisons and exports."""

    return pd.DataFrame([{**_scenario_row(result.scenario), **result.metrics} for result in results])


def build_bss_source_metrics_frame(results: Sequence[BSSValidationResult]) -> pd.DataFrame:
    """Build one row per active source, including physical and oracle-scaled errors."""

    rows: list[dict[str, object]] = []
    for result in results:
        scenario = result.scenario
        match = result.match
        oracle = result.oracle_source_metrics
        for source_index in match.active_source_indices:
            truth = scenario.sources[source_index]
            physical = match.estimated_sources_matched[source_index]
            scaled = result.oracle_scaled_sources[source_index]
            rows.append(
                {
                    **_scenario_row(scenario),
                    "method_key": result.fit_result.method_key,
                    "method": result.fit_result.method_label,
                    "implementation": result.fit_result.implementation,
                    "source_index": source_index + 1,
                    "source_role": scenario.source_roles[source_index],
                    "true_source_rms_rad": float(np.sqrt(np.mean(np.square(truth)))),
                    "estimated_source_rms_rad": float(np.sqrt(np.mean(np.square(physical)))),
                    "oracle_scaled_source_rms_rad": float(np.sqrt(np.mean(np.square(scaled)))),
                    "abs_source_correlation": float(match.correlations[source_index]),
                    "source_mse_rad2": float(match.source_mses[source_index]),
                    "source_rmse_rad": float(match.source_rmses[source_index]),
                    "source_mae_rad": float(match.source_maes[source_index]),
                    "source_max_abs_error_rad": float(match.source_max_abs_errors[source_index]),
                    "source_relative_rmse": float(match.source_relative_rmses[source_index]),
                    "source_r2": float(match.source_r2_scores[source_index]),
                    "source_snr_db": float(match.source_snr_db[source_index]),
                    "source_axis_error_deg": float(np.degrees(match.axis_errors_rad[source_index])),
                    "oracle_scale": float(result.oracle_scales[source_index]),
                    "source_mse_rad2_oracle_scaled": float(oracle["mse"][source_index]),
                    "source_rmse_rad_oracle_scaled": float(oracle["rmse"][source_index]),
                    "source_mae_rad_oracle_scaled": float(oracle["mae"][source_index]),
                    "source_max_abs_error_rad_oracle_scaled": float(
                        oracle["max_abs_error"][source_index]
                    ),
                    "source_relative_rmse_oracle_scaled": float(
                        oracle["relative_rmse"][source_index]
                    ),
                    "source_r2_oracle_scaled": float(oracle["r2"][source_index]),
                    "source_snr_db_oracle_scaled": float(oracle["snr_db"][source_index]),
                }
            )
    return pd.DataFrame(rows)


def _add_rss_oracle_case_metrics(frame: pd.DataFrame, results: Sequence[ValidationResult]) -> None:
    oracle_columns = (
        "mean_source_mse_oracle_scaled",
        "mean_source_rmse_oracle_scaled",
        "mean_source_mae_oracle_scaled",
        "max_source_abs_error_oracle_scaled",
        "mean_source_relative_rmse_oracle_scaled",
        "mean_source_r2_oracle_scaled",
        "mean_source_snr_db_oracle_scaled",
        "quaternion_reconstruction_loss_oracle_scaled",
        "quaternion_reconstruction_mean_error_deg_oracle_scaled",
        "quaternion_reconstruction_max_error_deg_oracle_scaled",
        "observation_reconstruction_rmse_oracle_scaled",
    )
    for name in oracle_columns:
        if name not in frame.columns:
            frame[name] = np.nan
    if "oracle_scales" not in frame.columns:
        frame["oracle_scales"] = pd.Series([None] * len(frame), dtype=object)
    for row_index, result in enumerate(results):
        scaled, scales, metrics = _oracle_scale_sources(
            result.scenario.sources,
            result.match.estimated_sources_matched,
            result.match.active_source_indices,
        )
        values = _mean_error_metrics("oracle_scaled", metrics)
        axes = result.rss_result.recovery_matrix[np.asarray(result.match.permutation)] * result.match.signs[:, None]
        _, _, quaternion_metrics = _quaternion_metrics(result.scenario, scaled, axes)
        values.update({f"{name}_oracle_scaled": value for name, value in quaternion_metrics.items()})
        values["oracle_scales"] = scales.tolist()
        for name, value in values.items():
            frame.at[row_index, name] = value


def build_method_comparison_frames(
    rss_results: Sequence[ValidationResult],
    bss_results: Sequence[BSSValidationResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return comparable case/source frames containing RSS and every BSS method."""

    bss_case = build_bss_metrics_frame(bss_results)
    bss_source = build_bss_source_metrics_frame(bss_results)
    case_frames: list[pd.DataFrame] = [bss_case] if not bss_case.empty else []
    source_frames: list[pd.DataFrame] = [bss_source] if not bss_source.empty else []
    if rss_results:
        rss_case = build_metrics_frame(list(rss_results)).copy()
        rss_case.insert(0, "method_key", "rss")
        rss_case.insert(1, "method", "RSS")
        rss_case.insert(2, "implementation", "rotational_source_separation")
        rss_case["converged"] = True
        rss_case["iterations"] = [len(result.rss_result.loss_history) for result in rss_results]
        rss_case["runtime_seconds"] = np.nan
        rss_case["linear_observation_reconstruction_rmse"] = np.nan
        _add_rss_oracle_case_metrics(rss_case, rss_results)
        case_frames.insert(0, rss_case)

        rss_source = build_source_metrics_frame(list(rss_results)).copy()
        rss_source.insert(0, "method_key", "rss")
        rss_source.insert(1, "method", "RSS")
        rss_source.insert(2, "implementation", "rotational_source_separation")
        rss_source["abs_source_correlation"] = rss_source["true_rss_abs_correlation"]
        oracle_rows: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        for result in rss_results:
            _, scales, metrics = _oracle_scale_sources(
                result.scenario.sources,
                result.match.estimated_sources_matched,
                result.match.active_source_indices,
            )
            oracle_rows[(result.scenario.case_id, result.scenario.config.seed)] = (scales, metrics)
        for row_index, row in rss_source.iterrows():
            scales, metrics = oracle_rows[(str(row["case_id"]), int(row["seed"]))]
            source_index = int(row["source_index"]) - 1
            rss_source.at[row_index, "oracle_scale"] = scales[source_index]
            for output_name, metric_name in (
                ("source_mse_rad2_oracle_scaled", "mse"),
                ("source_rmse_rad_oracle_scaled", "rmse"),
                ("source_mae_rad_oracle_scaled", "mae"),
                ("source_max_abs_error_rad_oracle_scaled", "max_abs_error"),
                ("source_relative_rmse_oracle_scaled", "relative_rmse"),
                ("source_r2_oracle_scaled", "r2"),
                ("source_snr_db_oracle_scaled", "snr_db"),
            ):
                rss_source.at[row_index, output_name] = metrics[metric_name][source_index]
        source_frames.insert(0, rss_source)

    case_comparison = pd.concat(case_frames, ignore_index=True, sort=False) if case_frames else pd.DataFrame()
    source_comparison = (
        pd.concat(source_frames, ignore_index=True, sort=False) if source_frames else pd.DataFrame()
    )
    return case_comparison, source_comparison
