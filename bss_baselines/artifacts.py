"""Artifact writers for direct observer-signal BSS validation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bss_baselines.evaluation import BSSValidationResult
from bss_baselines.plotting import (
    plot_bss_observer_reconstruction,
    plot_bss_source_comparison,
)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def bss_metrics_payload(result: BSSValidationResult) -> dict[str, object]:
    """Build a JSON-safe scientific record for one method/case fit."""

    scenario = result.scenario
    fit = result.fit_result
    match = result.match
    return _json_safe(
        {
            "scenario": {
                "family": scenario.family,
                "case_id": scenario.case_id,
                "seed": scenario.config.seed,
                "frame_count": scenario.config.T,
                "amplitude_tier": scenario.amplitude_tier,
                "stress_peak_angle_rad": scenario.stress_peak_angle_rad,
                "composition": scenario.composition,
                "replicate_index": scenario.replicate_index,
                "source_roles": scenario.source_roles,
                "observation_model": scenario.observation_model,
            },
            "method": {
                "key": fit.method_key,
                "label": fit.method_label,
                "implementation": fit.implementation,
                "input_domain": "cumulative_observer_signals",
                "converged": fit.converged,
                "iterations": fit.iterations,
                "runtime_seconds": fit.runtime_seconds,
                "warnings": fit.warnings,
                "config": {
                    "jade_max_sweeps": fit.config.jade_max_sweeps,
                    "jade_tol": fit.config.jade_tol,
                    "fastica_algorithm": fit.config.fastica_algorithm,
                    "fastica_fun": fit.config.fastica_fun,
                    "fastica_max_iter": fit.config.fastica_max_iter,
                    "fastica_tol": fit.config.fastica_tol,
                    "sobi_lags": fit.config.sobi_lags,
                    "sobi_max_sweeps": fit.config.sobi_max_sweeps,
                    "sobi_tol": fit.config.sobi_tol,
                    "random_state": fit.config.random_state,
                    "eps": fit.config.eps,
                },
            },
            "metrics": result.metrics,
            "match": {
                "permutation": match.permutation,
                "signs": match.signs,
                "correlations": match.correlations,
                "physical_source_mses": match.source_mses,
                "physical_source_rmses": match.source_rmses,
                "physical_source_relative_rmses": match.source_relative_rmses,
                "axis_errors_rad": match.axis_errors_rad,
                "oracle_scales": result.oracle_scales,
                "oracle_source_metrics": result.oracle_source_metrics,
            },
            "matrices": {
                "demixing": fit.demixing_matrix,
                "normalized_demixing": fit.normalized_demixing_matrix,
                "aligned_axes_observer_frame": result.aligned_axes_observer_frame,
                "mixing_pseudoinverse": fit.mixing_matrix,
                "demixing_row_norms": fit.demixing_row_norms,
            },
            "source_cross_correlations": {
                "original": result.original_source_cross_correlation_matrix,
                "physical": result.separated_source_cross_correlation_matrix,
                "oracle_scaled": result.oracle_source_cross_correlation_matrix,
            },
            "diagnostics": fit.diagnostics,
        }
    )


def export_bss_case_artifacts(
    result: BSSValidationResult,
    output_dir: str | Path,
    *,
    include_plots: bool = True,
) -> dict[str, object]:
    """Write all numeric artifacts and optional diagnostic plots for one fit."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = bss_metrics_payload(result)
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    fit = result.fit_result
    pd.DataFrame(
        fit.demixing_matrix,
        index=[f"component_{index + 1}" for index in range(3)],
        columns=[f"observer_{index + 1}" for index in range(3)],
    ).rename_axis("component").to_csv(output / "demixing_matrix.csv")
    pd.DataFrame(
        {
            "component": np.arange(1, 4),
            "row_norm": fit.demixing_row_norms,
            **{
                f"normalized_observer_{axis + 1}": fit.normalized_demixing_matrix[:, axis]
                for axis in range(3)
            },
            **{
                f"matched_observer_axis_{axis + 1}": result.aligned_axes_observer_frame[:, axis]
                for axis in range(3)
            },
        }
    ).to_csv(output / "normalized_axes.csv", index=False)
    time = result.scenario.time_seconds
    pd.DataFrame(
        {
            "time_seconds": time,
            **{
                f"true_source_{index + 1}": result.scenario.sources[index]
                for index in range(3)
            },
            **{
                f"physical_source_{index + 1}": result.match.estimated_sources_matched[index]
                for index in range(3)
            },
            **{
                f"oracle_scaled_source_{index + 1}": result.oracle_scaled_sources[index]
                for index in range(3)
            },
        }
    ).to_csv(output / "source_comparison.csv", index=False)
    pd.DataFrame(
        {
            "time_seconds": time,
            **{
                f"observer_{index + 1}": result.scenario.observations[index]
                for index in range(3)
            },
            **{
                f"linear_reconstruction_{index + 1}": fit.reconstructed_observations[index]
                for index in range(3)
            },
            **{
                f"physical_reconstruction_{index + 1}": result.physical_reconstructed_observations[index]
                for index in range(3)
            },
            **{
                f"oracle_scaled_reconstruction_{index + 1}": result.oracle_reconstructed_observations[index]
                for index in range(3)
            },
        }
    ).to_csv(output / "observer_reconstruction.csv", index=False)
    if include_plots:
        for filename, builder in (
            ("source_comparison.png", plot_bss_source_comparison),
            ("observer_reconstruction.png", plot_bss_observer_reconstruction),
        ):
            figure = builder(result)
            figure.savefig(output / filename, dpi=150, bbox_inches="tight")
            plt.close(figure)
    return payload
