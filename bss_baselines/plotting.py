"""Matplotlib helpers for observer-signal BSS comparisons."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bss_baselines.evaluation import BSSValidationResult


def plot_bss_source_comparison(result: BSSValidationResult) -> plt.Figure:
    """Plot truth, physical BSS output, and oracle-scaled waveform output."""

    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    time = result.scenario.time_seconds
    for source_index, axis in enumerate(axes):
        axis.plot(time, result.scenario.sources[source_index], label="True source", linewidth=2.0)
        axis.plot(
            time,
            result.match.estimated_sources_matched[source_index],
            label=f"{result.fit_result.method_label} physical",
            alpha=0.85,
        )
        axis.plot(
            time,
            result.oracle_scaled_sources[source_index],
            label="Oracle-scaled waveform",
            linestyle="--",
            alpha=0.85,
        )
        axis.set_ylabel(f"Source {source_index + 1} (rad)")
        axis.grid(True, alpha=0.25)
    axes[0].legend(ncol=3, fontsize="small")
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(f"{result.fit_result.method_label}: direct observer-signal recovery")
    figure.tight_layout()
    return figure


def plot_bss_observer_reconstruction(result: BSSValidationResult) -> plt.Figure:
    """Plot observed channels with linear and physical source-composed reconstructions."""

    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    time = result.scenario.time_seconds
    for channel, axis in enumerate(axes):
        axis.plot(time, result.scenario.observations[channel], label="Observed", linewidth=2.0)
        axis.plot(
            time,
            result.fit_result.reconstructed_observations[channel],
            label="Linear inverse",
            linestyle=":",
        )
        axis.plot(
            time,
            result.physical_reconstructed_observations[channel],
            label="Physical composition",
            alpha=0.8,
        )
        axis.set_ylabel(f"Observer {channel + 1} (rad)")
        axis.grid(True, alpha=0.25)
    axes[0].legend(ncol=3, fontsize="small")
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(f"{result.fit_result.method_label}: observer reconstruction")
    figure.tight_layout()
    return figure


def plot_method_family_bars(
    frame: pd.DataFrame,
    metric_labels: dict[str, str],
) -> plt.Figure:
    """Plot family-mean metrics as grouped bars for every available method."""

    if frame.empty:
        raise ValueError("frame must contain at least one comparison row.")
    required = {"method", "family", *metric_labels}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing comparison columns: {sorted(missing)}.")
    methods = list(dict.fromkeys(frame["method"].astype(str)))
    families = list(dict.fromkeys(frame["family"].astype(str)))
    metric_names = list(metric_labels)
    column_count = min(3, len(metric_names))
    row_count = int(math.ceil(len(metric_names) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.5 * column_count, 4.3 * row_count),
        squeeze=False,
    )
    x = np.arange(len(families), dtype=float)
    width = 0.8 / max(len(methods), 1)
    for axis, metric in zip(axes.ravel(), metric_names):
        for method_index, method in enumerate(methods):
            method_frame = frame.loc[frame["method"] == method]
            values = [
                method_frame.loc[method_frame["family"] == family, metric].mean()
                for family in families
            ]
            offsets = x + (method_index - (len(methods) - 1) / 2.0) * width
            axis.bar(offsets, values, width=width, label=method)
        axis.set_xticks(x, families, rotation=25, ha="right")
        axis.set_title(metric_labels[metric])
        axis.grid(True, axis="y", alpha=0.25)
    for axis in axes.ravel()[len(metric_names) :]:
        axis.set_visible(False)
    axes.ravel()[0].legend(fontsize="small")
    figure.tight_layout()
    return figure


def plot_method_metric_curves(
    frame: pd.DataFrame,
    metric_labels: dict[str, str],
    *,
    x_column: str,
    x_label: str,
    aggregate: str | None = None,
    summary: pd.DataFrame | None = None,
    log_x: bool = False,
) -> plt.Figure:
    """Plot each method's metrics against one swept design variable.

    ``aggregate="mean"`` averages replicates at each x value, which is required whenever
    a design has more than one case per level. ``summary`` supplies ``ci_lower`` and
    ``ci_upper`` bands produced by the correlation sweep.
    """

    if frame.empty:
        raise ValueError("frame must contain at least one comparison row.")
    if aggregate not in {None, "mean", "median"}:
        raise ValueError("aggregate must be None, 'mean', or 'median'.")
    required = {"method", x_column, *metric_labels}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing comparison columns: {sorted(missing)}.")
    methods = list(dict.fromkeys(frame["method"].astype(str)))
    metric_names = list(metric_labels)
    column_count = min(3, len(metric_names))
    row_count = int(math.ceil(len(metric_names) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.5 * column_count, 4.3 * row_count),
        squeeze=False,
    )
    # A log axis cannot show an exact zero, so the zero level is drawn one decade below
    # the smallest positive level and labelled as zero.
    positive = frame.loc[frame[x_column] > 0.0, x_column]
    zero_position = float(positive.min()) / 10.0 if log_x and not positive.empty else 0.0

    def _positions(values: np.ndarray) -> np.ndarray:
        if not log_x:
            return values
        return np.where(values <= 0.0, zero_position, values)

    for axis, metric in zip(axes.ravel(), metric_names):
        for method in methods:
            subset = frame.loc[frame["method"].astype(str) == method]
            if aggregate is not None:
                grouped = subset.groupby(x_column, as_index=False, sort=True)[metric]
                subset = grouped.mean() if aggregate == "mean" else grouped.median()
            subset = subset.sort_values(x_column, kind="stable")
            x_positions = _positions(subset[x_column].to_numpy(dtype=float))
            axis.plot(x_positions, subset[metric].to_numpy(dtype=float), marker="o", label=method)
            if summary is not None and not summary.empty and {"ci_lower", "ci_upper", "metric"}.issubset(summary.columns):
                band = summary.loc[
                    (summary["method"].astype(str) == method) & (summary["metric"] == metric)
                ].sort_values(x_column, kind="stable")
                if not band.empty:
                    axis.fill_between(
                        _positions(band[x_column].to_numpy(dtype=float)),
                        band["ci_lower"].to_numpy(dtype=float),
                        band["ci_upper"].to_numpy(dtype=float),
                        alpha=0.15,
                    )
        if log_x:
            axis.set_xscale("log")
        axis.set_title(metric_labels[metric])
        axis.set_xlabel(x_label)
        axis.grid(True, alpha=0.25)
    for axis in axes.ravel()[len(metric_names) :]:
        axis.set_visible(False)
    axes.ravel()[0].legend(fontsize="small")
    figure.tight_layout()
    return figure


def plot_method_stress_curves(
    frame: pd.DataFrame,
    metric_labels: dict[str, str],
) -> plt.Figure:
    """Plot each method against the configured large-rotation stress angle."""

    return plot_method_metric_curves(
        frame,
        metric_labels,
        x_column="stress_peak_angle_rad",
        x_label="Configured peak rotation (rad)",
    )


def plot_method_correlation_curves(
    frame: pd.DataFrame,
    metric_labels: dict[str, str],
    *,
    summary: pd.DataFrame | None = None,
) -> plt.Figure:
    """Plot each method against the target inter-source correlation on a log axis."""

    return plot_method_metric_curves(
        frame,
        metric_labels,
        x_column="target_source_correlation",
        x_label="Target inter-source correlation (0 shown at left)",
        aggregate="mean",
        summary=summary,
        log_x=True,
    )


def plot_method_distribution_grid(
    frame: pd.DataFrame,
    metric_labels: dict[str, str],
    *,
    kind: str,
) -> plt.Figure:
    """Plot method-level boxplots or empirical CDFs for formal results."""

    if kind not in {"box", "ecdf"}:
        raise ValueError("kind must be 'box' or 'ecdf'.")
    if frame.empty:
        raise ValueError("frame must contain at least one comparison row.")
    required = {"method", *metric_labels}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing comparison columns: {sorted(missing)}.")
    methods = list(dict.fromkeys(frame["method"].astype(str)))
    metric_names = list(metric_labels)
    column_count = min(3, len(metric_names))
    row_count = int(math.ceil(len(metric_names) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.5 * column_count, 4.3 * row_count),
        squeeze=False,
    )
    for axis, metric in zip(axes.ravel(), metric_names):
        method_values = [
            frame.loc[frame["method"] == method, metric]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=float)
            for method in methods
        ]
        if kind == "box":
            try:
                axis.boxplot(method_values, tick_labels=methods, showfliers=True)
            except TypeError:  # Matplotlib < 3.9
                axis.boxplot(method_values, labels=methods, showfliers=True)
            axis.tick_params(axis="x", labelrotation=25)
        else:
            for method, values in zip(methods, method_values):
                ordered = np.sort(values)
                probabilities = np.arange(1, ordered.size + 1) / max(ordered.size, 1)
                axis.step(ordered, probabilities, where="post", label=method)
            axis.set_ylabel("Empirical cumulative probability")
        axis.set_title(metric_labels[metric])
        axis.grid(True, alpha=0.25)
    for axis in axes.ravel()[len(metric_names) :]:
        axis.set_visible(False)
    if kind == "ecdf":
        axes.ravel()[0].legend(fontsize="small")
    figure.tight_layout()
    return figure
