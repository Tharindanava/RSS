"""Validation artifact writers for RSS."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rss.evaluation import ValidationResult
from rss.mutual_information import MI_UNIT
from rss.plotting import (
    plot_loss_history,
    plot_moving_rotation_geometry,
    plot_observed_signals,
    plot_observer_reconstruction,
    plot_source_axis_comparison,
    plot_source_comparison,
)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def validation_metrics_payload(result: ValidationResult) -> dict[str, object]:
    diagnostics = result.rss_result.matrix_diagnostics
    optional_values = lambda values: [
        None if value is None or not math.isfinite(float(value)) else float(value)
        for value in values
    ]
    payload = {
        "scenario": {
            "family": result.scenario.family,
            "case_id": result.scenario.case_id,
            "seed": result.scenario.config.seed,
            "frame_count": result.scenario.config.T,
            "amplitude_tier": result.scenario.amplitude_tier,
            "stress_peak_angle_rad": result.scenario.stress_peak_angle_rad,
            "composition": result.scenario.composition,
            "replicate_index": result.scenario.replicate_index,
            "source_roles": result.scenario.source_roles,
            "observation_model": result.scenario.observation_model,
            "requested_amplitudes_rad": optional_values(
                result.scenario.requested_amplitudes_rad
            ),
            "realized_amplitudes_rad": optional_values(
                result.scenario.realized_amplitudes_rad
            ),
            "requested_frequencies_hz": optional_values(
                result.scenario.requested_frequencies_hz
            ),
            "realized_frequencies_hz": optional_values(
                result.scenario.realized_frequencies_hz
            ),
            "sparse_duty_cycles": optional_values(
                result.scenario.sparse_duty_cycles
            ),
        },
        "metrics": _json_safe(result.metrics),
        "source_cross_correlations": {
            "limit": result.scenario.config.source_cross_correlation_limit,
            "original_source_matrix": result.original_source_cross_correlation_matrix.tolist(),
            "separated_source_matrix": result.separated_source_cross_correlation_matrix.tolist(),
        },
        "match": {
            "permutation": result.match.permutation,
            "active_source_indices": result.match.active_source_indices,
            "unmatched_estimate_indices": result.match.unmatched_estimate_indices,
            "signs": optional_values(result.match.signs),
            "correlations": optional_values(result.match.correlations),
            "source_mses": optional_values(result.match.source_mses),
            "source_rmses": optional_values(result.match.source_rmses),
            "source_maes": optional_values(result.match.source_maes),
            "source_max_abs_errors": optional_values(
                result.match.source_max_abs_errors
            ),
            "source_relative_rmses": optional_values(
                result.match.source_relative_rmses
            ),
            "source_r2_scores": optional_values(result.match.source_r2_scores),
            "source_snr_db": optional_values(result.match.source_snr_db),
            "axis_errors_rad": optional_values(result.match.axis_errors_rad),
        },
        "rss": {
            "optimizer": result.rss_result.optimizer_name,
            "loss_type": result.rss_result.config.loss_type,
            "sample_interval_seconds": result.rss_result.sample_interval_seconds,
            "separation_loss": {
                "weight": result.rss_result.config.separation_weight,
                "lags": result.rss_result.config.lags,
                "mi_estimator": result.rss_result.config.mi_estimator,
                "mi_unit": MI_UNIT,
                "gcmi_rank_method": result.rss_result.config.gcmi_rank_method,
                "bspline_bins": result.rss_result.config.bspline_bins,
                "bspline_order": result.rss_result.config.bspline_order,
                "ksg_k": result.rss_result.config.ksg_k,
                "ksg_standardize": result.rss_result.config.ksg_standardize,
                "ksg_max_samples": result.rss_result.config.ksg_max_samples,
                "ksg_subsample_mode": result.rss_result.config.ksg_subsample_mode,
                "ksg_subsample_seed": result.rss_result.config.ksg_subsample_seed,
                "mi_bins": result.rss_result.config.mi_bins,
                "mi_bandwidth": result.rss_result.config.mi_bandwidth,
                "action_inertias": result.rss_result.config.action_inertias,
                "action_stiffnesses": result.rss_result.config.action_stiffnesses,
            },
            "matrix_parameterization": result.rss_result.config.matrix_parameterization,
            "observer_signal_model": result.rss_result.config.observer_signal_model,
            "initial_recovery_matrix": result.rss_result.config.resolved_initial_matrix.tolist(),
            "learning_rate_schedule": {
                "type": "exponential",
                "configured_start": result.rss_result.config.learning_rate,
                "configured_end": result.rss_result.config.learning_rate_end,
                "first_applied": (
                    result.rss_result.applied_learning_rates[0]
                    if result.rss_result.applied_learning_rates
                    else None
                ),
                "terminal_applied": (
                    result.rss_result.applied_learning_rates[-1]
                    if result.rss_result.applied_learning_rates
                    else None
                ),
            },
            "recovery_matrix": result.rss_result.recovery_matrix.tolist(),
            "optimized_parameters": result.rss_result.optimized_parameters.tolist(),
            "optimized_weights": result.rss_result.optimized_weights.tolist(),
            "best_epoch": result.rss_result.best_epoch,
            "stop_reason": result.rss_result.stop_reason,
            "losses": {
                "normalization_reference": result.rss_result.loss_normalization_reference,
                "normalized_initial_loss": result.rss_result.normalized_initial_loss,
                "normalized_best_loss": result.rss_result.normalized_best_loss,
                "normalized_terminal_loss": result.rss_result.normalized_terminal_loss,
                "raw_initial_loss": result.rss_result.initial_loss,
                "raw_best_loss": result.rss_result.best_loss,
                "raw_terminal_loss": result.rss_result.terminal_loss,
            },
            "quaternion_reconstruction": {
                "mean_error_deg": result.metrics["quaternion_reconstruction_mean_error_deg"],
                "max_error_deg": result.metrics["quaternion_reconstruction_max_error_deg"],
                "full_quaternion_identity_loss": result.metrics["quaternion_reconstruction_loss"],
            },
            "matrix_diagnostics": {
                "frobenius_norm": diagnostics.frobenius_norm,
                "singular_values": diagnostics.singular_values,
                "condition_number": diagnostics.condition_number,
                "symmetry_error": diagnostics.symmetry_error,
                "orthogonality_error": diagnostics.orthogonality_error,
                "determinant": diagnostics.determinant,
                "warnings": diagnostics.warnings,
            },
        },
    }
    rss_result = result.rss_result
    if rss_result.config.adaptive_regularizer or rss_result.config.axis_flip_search:
        payload["rss"]["adaptive_regularizer"] = {
            "enabled": rss_result.config.adaptive_regularizer,
            "specialized_loss": rss_result.config.specialized_regularizer,
            "start_weight": rss_result.config.regularizer_start_weight,
            "plateau_patience": rss_result.config.regularizer_plateau_patience,
            "plateau_tolerance": rss_result.config.regularizer_plateau_tolerance,
            "ramp_epochs": rss_result.config.regularizer_ramp_epochs,
            "minimum_normalized_selection_loss": (
                rss_result.config.regularizer_min_loss_threshold
            ),
        }
        payload["rss"]["selection_losses"] = {
            "normalization_reference": rss_result.selection_loss_normalization_reference,
            "raw_initial": rss_result.initial_selection_loss,
            "raw_best": rss_result.best_selection_loss,
            "raw_terminal": rss_result.terminal_selection_loss,
            "normalized_initial": rss_result.normalized_initial_selection_loss,
            "normalized_best": rss_result.normalized_best_selection_loss,
            "normalized_terminal": rss_result.normalized_terminal_selection_loss,
        }
        payload["rss"]["axis_flip_search"] = {
            "enabled": rss_result.config.axis_flip_search,
            "interval": rss_result.config.axis_flip_interval,
            "trial_epochs": rss_result.config.axis_flip_trial_epochs,
            "events": [
                {
                    "check_epoch": event.check_epoch,
                    "completed_epoch": event.completed_epoch,
                    "selected_pattern": event.selected_pattern,
                    "candidates": [
                        {
                            "pattern": candidate.pattern,
                            "signs": candidate.signs,
                            "best_selection_loss": candidate.best_selection_loss,
                            "normalized_best_selection_loss": (
                                candidate.normalized_best_selection_loss
                            ),
                            "selected": candidate.selected,
                        }
                        for candidate in event.candidates
                    ],
                }
                for event in rss_result.axis_flip_events
            ],
        }
    return payload


def export_validation_case_artifacts(
    result: ValidationResult,
    output_dir: str | Path,
    *,
    include_plots: bool = True,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = validation_metrics_payload(result)
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    time = result.scenario.time_seconds
    pd.DataFrame(
        {"time_seconds": time, **{f"observer_{i + 1}": result.scenario.observations[i] for i in range(3)}}
    ).to_csv(output / "observed_signals.csv", index=False)
    pd.DataFrame(
        {
            "time_seconds": time,
            **{
                f"generated_source_{i + 1}": result.scenario.generated_sources[i]
                for i in range(3)
            },
            **{f"true_source_{i + 1}": result.scenario.sources[i] for i in range(3)},
            **{f"rss_source_{i + 1}": result.match.estimated_sources_matched[i] for i in range(3)},
        }
    ).to_csv(output / "source_comparison.csv", index=False)
    pd.DataFrame(
        {
            "time_seconds": time,
            **{
                f"observer_increment_{i + 1}": result.scenario.observer_rotation_increments[i]
                for i in range(3)
            },
            **{
                f"observer_rate_{i + 1}_rad_s": result.scenario.observer_angular_velocities[i]
                for i in range(3)
            },
        }
    ).to_csv(output / "observer_body_measurements.csv", index=False)
    pd.DataFrame(
        {
            "time_seconds": time,
            **{
                f"object_q_{name}": result.scenario.physical_object_quaternions[:, index]
                for index, name in enumerate(("w", "x", "y", "z"))
            },
            **{
                f"observer_q_{name}": result.scenario.physical_observer_quaternions[:, index]
                for index, name in enumerate(("w", "x", "y", "z"))
            },
        }
    ).to_csv(output / "physical_quaternions.csv", index=False)
    moving_axes: dict[str, object] = {"time_seconds": time}
    for triad_name, triad in (
        ("source", result.scenario.source_axes_global_time),
        ("observer", result.scenario.observer_axes_global_time),
    ):
        for axis in range(3):
            for coordinate, label in enumerate(("x", "y", "z")):
                moving_axes[f"{triad_name}_axis_{axis + 1}_{label}"] = triad[:, axis, coordinate]
    pd.DataFrame(moving_axes).to_csv(output / "moving_axes.csv", index=False)
    pd.DataFrame(result.rss_result.loss_history).to_csv(output / "loss_history.csv", index=False)
    if not include_plots:
        return payload
    for name, builder in (
        ("observed_signals.png", plot_observed_signals),
        ("source_comparison.png", plot_source_comparison),
        ("observer_reconstruction.png", plot_observer_reconstruction),
        ("axes_geometry.png", plot_source_axis_comparison),
        (
            "moving_axes_geometry.png",
            lambda validation: plot_moving_rotation_geometry(validation.scenario),
        ),
    ):
        figure = builder(result)
        figure.savefig(output / name, dpi=150, bbox_inches="tight")
        plt.close(figure)
    loss_figure = plot_loss_history(result.rss_result.loss_history)
    loss_figure.savefig(output / "loss_history.png", dpi=150, bbox_inches="tight")
    plt.close(loss_figure)
    return payload
