"""Ground-truth evaluation for RSS simulations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np
import pandas as pd

from rss.axes import normalize_rows, source_axes_from_direction_cosines
from rss.model import (
    angles_to_quaternion_components,
    decode_scalar_and_vector,
    integrated_body_angles_to_quaternions,
    quaternion_error_angles,
    relative_reconstruction_quaternions,
)
from rss.mutual_information import MI_UNIT
from rss.simulation import ScenarioData, source_cross_correlation_matrix
from rss.solver import RSSConfig, RSSResult, fit_rss


@dataclass(frozen=True)
class MatchResult:
    permutation: tuple[int, int, int]
    signs: np.ndarray
    correlations: np.ndarray
    source_mses: np.ndarray
    source_rmses: np.ndarray
    source_maes: np.ndarray
    source_max_abs_errors: np.ndarray
    source_relative_rmses: np.ndarray
    source_r2_scores: np.ndarray
    source_snr_db: np.ndarray
    estimated_sources_matched: np.ndarray
    estimated_axes_matched: np.ndarray
    axis_errors_rad: np.ndarray
    active_source_indices: tuple[int, ...]
    unmatched_estimate_indices: tuple[int, ...]

    @property
    def mean_abs_correlation(self) -> float:
        return float(np.nanmean(np.abs(self.correlations)))


@dataclass(frozen=True)
class ValidationResult:
    scenario: ScenarioData
    rss_result: RSSResult
    match: MatchResult
    reconstructed_observations: np.ndarray
    reconstructed_coefficients: np.ndarray
    original_source_cross_correlation_matrix: np.ndarray
    separated_source_cross_correlation_matrix: np.ndarray
    metrics: dict[str, float | str]


def _corr(first: np.ndarray, second: np.ndarray) -> float:
    a = first - np.mean(first)
    b = second - np.mean(second)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator <= 1e-12 else float(np.dot(a, b) / denominator)


def _cross_correlation_metrics(prefix: str, matrix: np.ndarray) -> dict[str, float]:
    array = np.asarray(matrix, dtype=float)
    absolute_pairs = {
        f"{first + 1}{second + 1}": abs(float(array[first, second]))
        for first in range(array.shape[0])
        for second in range(first + 1, array.shape[0])
    }
    values = np.fromiter(absolute_pairs.values(), dtype=float)
    metrics = {
        f"{prefix}_abs_cross_correlation_{pair}": value
        for pair, value in absolute_pairs.items()
    }
    metrics[f"{prefix}_mean_abs_cross_correlation"] = (
        float(np.mean(values)) if values.size else float("nan")
    )
    metrics[f"{prefix}_max_abs_cross_correlation"] = (
        float(np.max(values)) if values.size else float("nan")
    )
    return metrics


def match_estimated_sources(
    true_sources: np.ndarray,
    estimated_sources: np.ndarray,
    estimated_matrix: np.ndarray,
    true_axes: np.ndarray,
    observer_axes: np.ndarray | None = None,
    active_source_indices: tuple[int, ...] | None = None,
) -> MatchResult:
    truth = np.asarray(true_sources, dtype=float).reshape(3, -1)
    estimate = np.asarray(estimated_sources, dtype=float).reshape(3, -1)
    active = tuple(range(3)) if active_source_indices is None else tuple(
        int(index) for index in active_source_indices
    )
    if not active or len(set(active)) != len(active) or any(index not in range(3) for index in active):
        raise ValueError("active_source_indices must contain one to three unique indices in [0, 2].")
    correlation_matrix = np.array([[_corr(truth[i], estimate[j]) for j in range(3)] for i in range(3)])
    active_assignment = max(
        permutations(range(3), len(active)),
        key=lambda candidate: sum(
            abs(correlation_matrix[source_index, estimate_index])
            for source_index, estimate_index in zip(active, candidate)
        ),
    )
    assignment_by_source: list[int | None] = [None, None, None]
    for source_index, estimate_index in zip(active, active_assignment):
        assignment_by_source[source_index] = int(estimate_index)
    remaining_estimates = [
        index for index in range(3) if index not in set(active_assignment)
    ]
    inactive_sources = [index for index in range(3) if index not in set(active)]
    for source_index, estimate_index in zip(inactive_sources, remaining_estimates):
        assignment_by_source[source_index] = int(estimate_index)
    assignment = tuple(int(index) for index in assignment_by_source)
    raw_corr = np.full(3, np.nan, dtype=float)
    for source_index in active:
        raw_corr[source_index] = correlation_matrix[source_index, assignment[source_index]]
    signs = np.ones(3, dtype=float)
    signs[np.asarray(active)] = np.where(raw_corr[np.asarray(active)] < 0.0, -1.0, 1.0)
    matched = estimate[np.asarray(assignment)] * signs[:, None]
    axes = (
        source_axes_from_direction_cosines(estimated_matrix, observer_axes)[np.asarray(assignment)]
        * signs[:, None]
    )
    truth_axes = normalize_rows(np.asarray(true_axes, dtype=float))
    axis_errors = np.full(3, np.nan, dtype=float)
    axis_errors[np.asarray(active)] = np.arccos(
        np.clip(
            np.abs(
                np.sum(
                    axes[np.asarray(active)] * truth_axes[np.asarray(active)],
                    axis=1,
                )
            ),
            -1.0,
            1.0,
        )
    )
    active_array = np.asarray(active)
    errors = matched[active_array] - truth[active_array]
    squared_errors = np.square(errors)
    mses = np.full(3, np.nan, dtype=float)
    mses[active_array] = np.mean(squared_errors, axis=1)
    rmses = np.full(3, np.nan, dtype=float)
    rmses[active_array] = np.sqrt(mses[active_array])
    maes = np.full(3, np.nan, dtype=float)
    maes[active_array] = np.mean(np.abs(errors), axis=1)
    max_abs_errors = np.full(3, np.nan, dtype=float)
    max_abs_errors[active_array] = np.max(np.abs(errors), axis=1)
    true_active = truth[active_array]
    numerical_floor = np.finfo(float).eps
    true_rms = np.sqrt(np.mean(np.square(true_active), axis=1))
    relative_rmses = np.full(3, np.nan, dtype=float)
    relative_rmses[active_array] = rmses[active_array] / np.maximum(
        true_rms, numerical_floor
    )
    centered_truth = true_active - np.mean(true_active, axis=1, keepdims=True)
    total_squares = np.sum(np.square(centered_truth), axis=1)
    residual_squares = np.sum(squared_errors, axis=1)
    r2_scores = np.full(3, np.nan, dtype=float)
    r2_scores[active_array] = 1.0 - residual_squares / np.maximum(
        total_squares, numerical_floor
    )
    signal_power = np.mean(np.square(true_active), axis=1)
    snr_db = np.full(3, np.nan, dtype=float)
    snr_db[active_array] = 10.0 * np.log10(
        np.maximum(signal_power, numerical_floor)
        / np.maximum(mses[active_array], numerical_floor)
    )
    return MatchResult(
        permutation=assignment,
        signs=signs,
        correlations=np.abs(raw_corr),
        source_mses=mses,
        source_rmses=rmses,
        source_maes=maes,
        source_max_abs_errors=max_abs_errors,
        source_relative_rmses=relative_rmses,
        source_r2_scores=r2_scores,
        source_snr_db=snr_db,
        estimated_sources_matched=matched,
        estimated_axes_matched=axes,
        axis_errors_rad=axis_errors,
        active_source_indices=active,
        unmatched_estimate_indices=tuple(remaining_estimates),
    )


def evaluate_recovery(scenario: ScenarioData, rss_result: RSSResult) -> ValidationResult:
    match = match_estimated_sources(
        scenario.sources,
        rss_result.source_angles,
        rss_result.recovery_matrix,
        scenario.source_axes_global,
        scenario.observer_axes_global,
        scenario.active_source_indices,
    )
    reconstructed_coefficients = rss_result.reconstructed_observer_coefficients
    reconstructed_observations = rss_result.reconstructed_observer_angles
    active_indices = np.asarray(match.active_source_indices)
    original_correlation = source_cross_correlation_matrix(scenario.sources[active_indices])
    separated_correlation = source_cross_correlation_matrix(
        match.estimated_sources_matched[active_indices]
    )
    best_history = rss_result.loss_history[rss_result.best_epoch]
    applied_learning_rates = rss_result.applied_learning_rates
    relative = rss_result.relative_reconstruction_quaternions
    identity = np.zeros_like(relative)
    identity[:, 0] = 1.0
    quaternion_reconstruction_loss = float(
        np.mean(
            np.minimum(
                np.sum((relative - identity) ** 2, axis=1),
                np.sum((relative + identity) ** 2, axis=1),
            )
        )
    )
    estimated_energy = np.mean(np.square(rss_result.source_angles), axis=1)
    total_estimated_energy = float(np.sum(estimated_energy))
    unmatched_output_energy_ratio = (
        float(np.sum(estimated_energy[np.asarray(match.unmatched_estimate_indices)]))
        / max(total_estimated_energy, 1e-12)
        if match.unmatched_estimate_indices
        else 0.0
    )
    aligned_recovery_matrix = (
        rss_result.recovery_matrix[np.asarray(match.permutation)] * match.signs[:, None]
    )
    if scenario.observation_model == "body_fixed_gyro":
        physical_from_channels = integrated_body_angles_to_quaternions(
            scenario.observations,
            eps=float(rss_result.config.quaternion_eps),
        )
    else:
        legacy_scalar, legacy_vector = angles_to_quaternion_components(
            scenario.observations
        )
        _, physical_from_channels = decode_scalar_and_vector(
            legacy_scalar,
            legacy_vector,
            eps=float(rss_result.config.quaternion_eps),
        )
    physical_consistency_angles = quaternion_error_angles(
        relative_reconstruction_quaternions(
            physical_from_channels,
            scenario.physical_observer_quaternions,
            eps=float(rss_result.config.quaternion_eps),
        ),
        eps=float(rss_result.config.quaternion_eps),
    )
    metrics = {
        "rss_optimizer": rss_result.optimizer_name,
        "rss_loss_type": rss_result.config.loss_type,
        "rss_mi_estimator": rss_result.config.mi_estimator,
        "rss_mi_unit": MI_UNIT,
        "rss_gcmi_rank_method": rss_result.config.gcmi_rank_method,
        "rss_bspline_bins": int(rss_result.config.bspline_bins),
        "rss_bspline_order": int(rss_result.config.bspline_order),
        "rss_ksg_k": int(rss_result.config.ksg_k),
        "rss_ksg_standardize": bool(rss_result.config.ksg_standardize),
        "rss_ksg_max_samples": rss_result.config.ksg_max_samples,
        "rss_ksg_subsample_mode": rss_result.config.ksg_subsample_mode,
        "rss_ksg_subsample_seed": int(rss_result.config.ksg_subsample_seed),
        "rss_sample_interval_seconds": rss_result.sample_interval_seconds,
        "matrix_parameterization": rss_result.config.matrix_parameterization,
        "observer_signal_model": rss_result.config.observer_signal_model,
        "simulation_observation_model": scenario.observation_model,
        "rss_learning_rate_start": float(rss_result.config.learning_rate),
        "rss_learning_rate_end": float(rss_result.config.learning_rate_end),
        "rss_first_applied_learning_rate": (
            float(applied_learning_rates[0]) if applied_learning_rates else None
        ),
        "rss_terminal_applied_learning_rate": (
            float(applied_learning_rates[-1]) if applied_learning_rates else None
        ),
        **_cross_correlation_metrics("original_source", original_correlation),
        **_cross_correlation_metrics("separated_source", separated_correlation),
        "active_source_count": float(len(match.active_source_indices)),
        "sparse_source_count": float(len(scenario.sparse_source_indices)),
        "mean_abs_source_correlation": match.mean_abs_correlation,
        "mean_source_mse": float(np.nanmean(match.source_mses)),
        "mean_source_rmse": float(np.nanmean(match.source_rmses)),
        "mean_source_mae": float(np.nanmean(match.source_maes)),
        "max_source_abs_error": float(np.nanmax(match.source_max_abs_errors)),
        "mean_source_relative_rmse": float(
            np.nanmean(match.source_relative_rmses)
        ),
        "mean_source_r2": float(np.nanmean(match.source_r2_scores)),
        "mean_source_snr_db": float(np.nanmean(match.source_snr_db)),
        "mean_axis_error_deg": float(np.degrees(np.nanmean(match.axis_errors_rad))),
        "unmatched_output_energy_ratio": unmatched_output_energy_ratio,
        "observation_reconstruction_rmse": float(np.sqrt(np.mean((reconstructed_observations - scenario.observations) ** 2))),
        "coefficient_reconstruction_rmse": float(np.sqrt(np.mean((reconstructed_coefficients - scenario.observation_coefficients) ** 2))),
        "quaternion_reconstruction_loss": quaternion_reconstruction_loss,
        "quaternion_reconstruction_mean_error_deg": float(np.degrees(rss_result.reconstruction_mean_error_rad)),
        "quaternion_reconstruction_max_error_deg": float(np.degrees(rss_result.reconstruction_max_error_rad)),
        "raw_recovery_matrix_rmse": float(
            np.sqrt(np.mean((rss_result.recovery_matrix - scenario.mixing_matrix) ** 2))
        ),
        "recovery_matrix_rmse": float(
            np.sqrt(np.mean((aligned_recovery_matrix - scenario.mixing_matrix) ** 2))
        ),
        "aligned_recovery_matrix_rmse": float(
            np.sqrt(np.mean((aligned_recovery_matrix - scenario.mixing_matrix) ** 2))
        ),
        "physical_quaternion_consistency_mean_error_deg": float(
            np.degrees(np.mean(physical_consistency_angles))
        ),
        "physical_quaternion_consistency_max_error_deg": float(
            np.degrees(np.max(physical_consistency_angles))
        ),
        "rss_loss_normalization_reference": float(rss_result.loss_normalization_reference),
        "rss_normalized_initial_loss": float(rss_result.normalized_initial_loss),
        "rss_normalized_best_loss": float(rss_result.normalized_best_loss),
        "rss_normalized_terminal_loss": float(rss_result.normalized_terminal_loss),
        "rss_initial_loss": float(rss_result.initial_loss),
        "rss_best_loss": float(rss_result.best_loss),
        "rss_best_separation_loss": float(best_history["separation_loss"]),
        "rss_best_normalized_separation_loss": float(
            best_history["normalized_separation_loss"]
        ),
        "rss_terminal_loss": float(rss_result.terminal_loss),
        "rss_best_reconstruction_loss": float(best_history["reconstruction_loss"]),
        "rss_best_normalized_reconstruction_loss": float(best_history["normalized_reconstruction_loss"]),
        "rss_best_epoch": float(rss_result.best_epoch),
        "matrix_condition_number": float(rss_result.matrix_diagnostics.condition_number),
        "matrix_orthogonality_error": float(rss_result.matrix_diagnostics.orthogonality_error),
        "matrix_determinant": float(rss_result.matrix_diagnostics.determinant),
    }
    if rss_result.config.adaptive_regularizer or rss_result.config.axis_flip_search:
        metrics.update(
            {
                "rss_adaptive_regularizer": bool(rss_result.config.adaptive_regularizer),
                "rss_specialized_regularizer": rss_result.config.specialized_regularizer or "",
                "rss_axis_flip_search": bool(rss_result.config.axis_flip_search),
                "rss_axis_flip_check_count": float(len(rss_result.axis_flip_events)),
                "rss_regularizer_min_loss_threshold": float(
                    rss_result.config.regularizer_min_loss_threshold
                ),
                "rss_best_regularizer_loss": float(
                    best_history.get("regularizer_loss", 0.0)
                ),
                "rss_best_regularizer_weight": float(
                    best_history.get("regularizer_weight", 0.0)
                ),
                "rss_best_regularizer_phase": str(
                    best_history.get("regularizer_phase", "disabled")
                ),
                "rss_terminal_regularizer_weight": float(
                    rss_result.loss_history[-1].get("regularizer_weight", 0.0)
                ),
                "rss_terminal_regularizer_phase": str(
                    rss_result.loss_history[-1].get(
                        "regularizer_phase", "disabled"
                    )
                ),
                "rss_selection_loss_normalization_reference": float(
                    rss_result.selection_loss_normalization_reference
                    if rss_result.selection_loss_normalization_reference is not None
                    else rss_result.loss_normalization_reference
                ),
                "rss_initial_selection_loss": float(
                    rss_result.initial_selection_loss
                    if rss_result.initial_selection_loss is not None
                    else rss_result.initial_loss
                ),
                "rss_best_selection_loss": float(
                    rss_result.best_selection_loss
                    if rss_result.best_selection_loss is not None
                    else rss_result.best_loss
                ),
                "rss_normalized_best_selection_loss": float(
                    rss_result.normalized_best_selection_loss
                    if rss_result.normalized_best_selection_loss is not None
                    else rss_result.normalized_best_loss
                ),
            }
        )
    return ValidationResult(
        scenario=scenario,
        rss_result=rss_result,
        match=match,
        reconstructed_observations=reconstructed_observations,
        reconstructed_coefficients=reconstructed_coefficients,
        original_source_cross_correlation_matrix=original_correlation,
        separated_source_cross_correlation_matrix=separated_correlation,
        metrics=metrics,
    )


def run_validation_for_scenario(scenario: ScenarioData, rss_config: RSSConfig) -> ValidationResult:
    times = np.asarray(scenario.time_seconds, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("Scenario timestamps must contain at least two samples.")
    intervals = np.diff(times)
    sample_interval = float(np.median(intervals))
    if not np.isfinite(intervals).all() or sample_interval <= 0.0 or not np.allclose(
        intervals, sample_interval, atol=max(1e-12, sample_interval * 1e-9), rtol=0.0
    ):
        raise ValueError("Scenario timestamps must be finite, increasing, and uniformly sampled.")
    return evaluate_recovery(
        scenario,
        fit_rss(
            scenario.observations,
            rss_config,
            sample_interval_seconds=sample_interval,
        ),
    )


def run_validation(sim_config: object, rss_config: RSSConfig) -> ValidationResult:
    from rss.simulation import generate_validation_scenario

    return run_validation_for_scenario(generate_validation_scenario(sim_config), rss_config)


def build_metrics_frame(results: list[ValidationResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        row: dict[str, object] = {
            "family": result.scenario.family,
            "case_id": result.scenario.case_id,
            "seed": result.scenario.config.seed,
            "frame_count": result.scenario.config.T,
            "amplitude_tier": result.scenario.amplitude_tier,
            "stress_peak_angle_rad": result.scenario.stress_peak_angle_rad,
            "composition": result.scenario.composition,
            "replicate_index": result.scenario.replicate_index,
            "source_roles": "|".join(result.scenario.source_roles),
            "simulation_observation_model": result.scenario.observation_model,
            "matrix_parameterization": result.rss_result.config.matrix_parameterization,
            "observer_signal_model": result.rss_result.config.observer_signal_model,
        }
        for index in range(3):
            ordinal = index + 1
            row[f"source_{ordinal}_role"] = result.scenario.source_roles[index]
            row[f"source_{ordinal}_requested_amplitude_rad"] = (
                result.scenario.requested_amplitudes_rad[index]
            )
            row[f"source_{ordinal}_realized_amplitude_rad"] = (
                result.scenario.realized_amplitudes_rad[index]
            )
            row[f"source_{ordinal}_requested_frequency_hz"] = (
                result.scenario.requested_frequencies_hz[index]
            )
            row[f"source_{ordinal}_realized_frequency_hz"] = (
                result.scenario.realized_frequencies_hz[index]
            )
            row[f"source_{ordinal}_sparse_duty_cycle"] = (
                result.scenario.sparse_duty_cycles[index]
            )
        active_amplitudes = [
            float(value)
            for value in result.scenario.requested_amplitudes_rad
            if value is not None
        ]
        active_frequencies = [
            float(value)
            for value in result.scenario.requested_frequencies_hz
            if value is not None
        ]
        row["mean_requested_amplitude_rad"] = (
            float(np.mean(active_amplitudes)) if active_amplitudes else float("nan")
        )
        row["mean_requested_frequency_hz"] = (
            float(np.mean(active_frequencies)) if active_frequencies else float("nan")
        )
        row.update(result.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def build_source_metrics_frame(results: list[ValidationResult]) -> pd.DataFrame:
    """Build one detailed metric row for every active matched source."""

    rows: list[dict[str, object]] = []
    for result in results:
        scenario = result.scenario
        match = result.match
        for source_index in match.active_source_indices:
            truth = scenario.sources[source_index]
            estimate = match.estimated_sources_matched[source_index]
            rows.append(
                {
                    "family": scenario.family,
                    "case_id": scenario.case_id,
                    "seed": scenario.config.seed,
                    "frame_count": scenario.config.T,
                    "amplitude_tier": scenario.amplitude_tier,
                    "source_index": source_index + 1,
                    "source_role": scenario.source_roles[source_index],
                    "requested_amplitude_rad": scenario.requested_amplitudes_rad[
                        source_index
                    ],
                    "realized_amplitude_rad": scenario.realized_amplitudes_rad[
                        source_index
                    ],
                    "requested_frequency_hz": scenario.requested_frequencies_hz[
                        source_index
                    ],
                    "realized_frequency_hz": scenario.realized_frequencies_hz[
                        source_index
                    ],
                    "true_source_rms_rad": float(
                        np.sqrt(np.mean(np.square(truth)))
                    ),
                    "rss_source_rms_rad": float(
                        np.sqrt(np.mean(np.square(estimate)))
                    ),
                    "true_rss_abs_correlation": float(
                        match.correlations[source_index]
                    ),
                    "source_mse_rad2": float(match.source_mses[source_index]),
                    "source_rmse_rad": float(match.source_rmses[source_index]),
                    "source_mae_rad": float(match.source_maes[source_index]),
                    "source_max_abs_error_rad": float(
                        match.source_max_abs_errors[source_index]
                    ),
                    "source_relative_rmse": float(
                        match.source_relative_rmses[source_index]
                    ),
                    "source_r2": float(match.source_r2_scores[source_index]),
                    "source_snr_db": float(match.source_snr_db[source_index]),
                    "source_axis_error_deg": float(
                        np.degrees(match.axis_errors_rad[source_index])
                    ),
                }
            )
    return pd.DataFrame(rows)


def describe_metric_distributions(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    metric_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Return tidy descriptive distributions for numeric metrics."""

    if not group_columns:
        raise ValueError("group_columns must contain at least one column.")
    missing = [
        column
        for column in (*group_columns, *metric_columns)
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing distribution columns: {missing}.")
    rows: list[dict[str, object]] = []
    grouped = frame.groupby(list(group_columns), sort=False, dropna=False)
    for group_key, group in grouped:
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        identifiers = dict(zip(group_columns, keys))
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            rows.append(
                {
                    **identifiers,
                    "metric": metric,
                    "count": int(values.size),
                    "mean": float(np.mean(values)),
                    "std": (
                        float(np.std(values, ddof=1))
                        if values.size > 1
                        else float("nan")
                    ),
                    "min": float(np.min(values)),
                    "p05": float(np.quantile(values, 0.05)),
                    "p25": float(np.quantile(values, 0.25)),
                    "median": float(np.median(values)),
                    "p75": float(np.quantile(values, 0.75)),
                    "p95": float(np.quantile(values, 0.95)),
                    "max": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def summarize_standard_suite(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [column for column in frame.select_dtypes(include=[np.number]).columns if column not in {"seed", "frame_count"}]
    return frame.groupby("family", as_index=False)[numeric].mean()


def summarize_stress_suite(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("stress_peak_angle_rad", kind="stable").reset_index(drop=True)


def summarize_formal_suite(frame: pd.DataFrame) -> pd.DataFrame:
    """Average formal metrics for each source composition and signal family."""

    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in {"seed", "frame_count", "replicate_index"}
    ]
    return (
        frame.groupby(["composition", "family"], as_index=False, sort=False)[numeric]
        .mean()
        .reset_index(drop=True)
    )
