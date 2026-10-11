"""Small plotting helpers shared by the RSS notebook and exporters."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from rss.evaluation import ValidationResult
from rss.simulation import ScenarioData


def plot_rotation_geometry(scenario: ScenarioData):
    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    for index, vector in enumerate(scenario.observer_axes_global):
        axis.quiver(0, 0, 0, *vector, color="tab:blue", label="Observer axes" if index == 0 else None)
    for index, vector in enumerate(scenario.source_axes_global):
        axis.quiver(0, 0, 0, *vector, color="tab:orange", label="Source axes" if index == 0 else None)
    axis.set_title("RSS Source and Observer Geometry")
    axis.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
    axis.legend()
    return figure


def plot_moving_rotation_geometry(scenario: ScenarioData):
    """Show the attached source and observer triads at three trajectory frames."""

    sample_indices = (0, len(scenario.time_seconds) // 2, len(scenario.time_seconds) - 1)
    figure = plt.figure(figsize=(14, 4.5))
    for panel, sample in enumerate(sample_indices, start=1):
        axis = figure.add_subplot(1, 3, panel, projection="3d")
        for index, vector in enumerate(scenario.observer_axes_global_time[sample]):
            axis.quiver(
                0,
                0,
                0,
                *vector,
                color="tab:blue",
                label="Observer" if index == 0 else None,
            )
        for index, vector in enumerate(scenario.source_axes_global_time[sample]):
            axis.quiver(
                0,
                0,
                0,
                *vector,
                color="tab:orange",
                label="Source" if index == 0 else None,
            )
        axis.set_title(f"t = {scenario.time_seconds[sample]:.2f} s")
        axis.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        if panel == 1:
            axis.legend()
    figure.suptitle("Body-Fixed Axes Moving with the Virtual Object")
    figure.tight_layout()
    return figure


def plot_observed_signals(result: ValidationResult):
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for index in range(3):
        axes[0].plot(result.scenario.time_seconds, result.scenario.sources[index], label=f"Source {index + 1}")
        axes[1].plot(result.scenario.time_seconds, result.scenario.observations[index], label=f"Observer {index + 1}")
    axes[0].set_title("True Source Angles")
    axes[1].set_title("Cumulative Body-Fixed Observer Channels")
    axes[1].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.tight_layout()
    return figure


def plot_source_comparison(result: ValidationResult):
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for index, axis in enumerate(axes):
        axis.plot(result.scenario.time_seconds, result.scenario.sources[index], label="True", linewidth=1.4)
        axis.plot(result.scenario.time_seconds, result.match.estimated_sources_matched[index], label="RSS", alpha=0.85)
        role = result.scenario.source_roles[index]
        correlation = result.match.correlations[index]
        score = "not scored" if not np.isfinite(correlation) else f"|corr|={correlation:.3f}"
        axis.set_title(f"Source {index + 1} ({role}) | {score}")
        axis.grid(True, alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("Time (s)")
    figure.tight_layout()
    return figure


def plot_source_axis_comparison(result: ValidationResult):
    true_axes = result.scenario.source_axes_global
    estimated = result.match.estimated_axes_matched
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    coordinates = np.arange(3)
    for index, axis in enumerate(axes):
        axis.bar(coordinates - 0.18, true_axes[index], width=0.36, label="True")
        axis.bar(coordinates + 0.18, estimated[index], width=0.36, label="RSS")
        axis.set_xticks(coordinates, ("x", "y", "z"))
        axis.set_title(f"Axis {index + 1} ({result.scenario.source_roles[index]})")
        axis.grid(True, axis="y", alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    return figure


def plot_loss_history(loss_history: tuple[dict[str, float | str | None], ...]):
    epochs = np.asarray([row["epoch"] for row in loss_history])
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(epochs, [row["normalized_total_loss"] for row in loss_history], label="normalized_total_loss")
    if loss_history and "normalized_selection_loss" in loss_history[0]:
        axes[0].plot(
            epochs,
            [row["normalized_selection_loss"] for row in loss_history],
            label="normalized_selection_loss",
            linestyle="--",
        )
        threshold = float(loss_history[0].get("selection_stop_threshold", 0.0))
        if threshold > 0.0:
            axes[0].axhline(
                threshold,
                color="tab:red",
                linestyle=":",
                linewidth=0.9,
                label=f"selection threshold {threshold:g}",
            )
        for row in loss_history:
            if row.get("axis_flip_candidate_selection_losses") is not None:
                axes[0].axvline(
                    float(row["epoch"]),
                    color="tab:orange",
                    linestyle=":",
                    linewidth=0.8,
                    alpha=0.7,
                )
                axes[0].annotate(
                    f"flip: {row.get('axis_flip_pattern')}",
                    (float(row["epoch"]), float(row["normalized_selection_loss"])),
                    fontsize=7,
                    rotation=90,
                    va="bottom",
                )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="initial reference")
    axes[1].plot(
        epochs,
        [row["normalized_separation_loss"] for row in loss_history],
        label="normalized_separation_loss",
    )
    axes[1].plot(
        epochs,
        [row["normalized_orthogonality_loss"] for row in loss_history],
        label="normalized_orthogonality_loss",
    )
    axes[1].plot(
        epochs,
        [row["normalized_reconstruction_loss"] for row in loss_history],
        label="normalized_reconstruction_loss",
    )
    if loss_history and "normalized_regularizer_loss" in loss_history[0]:
        axes[1].plot(
            epochs,
            [row["normalized_regularizer_loss"] for row in loss_history],
            label="normalized_regularizer_loss",
        )
    axes[0].set_title("Normalized RSS Total Loss (Initial Objective = 1)")
    axes[1].set_title("Normalized Weighted RSS Loss Contributions")
    learning_rate_points = [
        (float(row["epoch"]), float(row["learning_rate"]))
        for row in loss_history
        if row.get("learning_rate") is not None
    ]
    if learning_rate_points:
        learning_rate_epochs, learning_rates = zip(*learning_rate_points)
        axes[2].semilogy(
            learning_rate_epochs,
            learning_rates,
            label="applied_learning_rate",
            color="tab:purple",
        )
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "No optimizer movement was applied", ha="center", va="center", transform=axes[2].transAxes)
    axes[2].set_title("Applied Exponential Learning Rate")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Epoch")
    axes[0].set_ylabel("Normalized loss")
    axes[1].set_ylabel("Normalized contribution")
    axes[2].set_ylabel("Learning rate (log scale)")
    if loss_history and "regularizer_weight" in loss_history[0]:
        regularizer_axis = axes[2].twinx()
        regularizer_axis.plot(
            epochs,
            [row["regularizer_weight"] for row in loss_history],
            label="regularizer_weight",
            color="tab:green",
            linestyle="--",
        )
        regularizer_axis.set_ylim(-0.05, 1.05)
        regularizer_axis.set_ylabel("Regularizer weight")
        regularizer_axis.legend(loc="lower right")
        previous_phase = None
        for row in loss_history:
            phase = row.get("regularizer_phase")
            if phase != previous_phase:
                regularizer_axis.annotate(
                    str(phase),
                    (float(row["epoch"]), float(row["regularizer_weight"])),
                    fontsize=7,
                    rotation=30,
                )
                previous_phase = phase
    for axis in axes:
        axis.grid(True, alpha=0.25)
        if axis.lines:
            axis.legend()
    figure.tight_layout()
    return figure


def plot_observer_reconstruction(result: ValidationResult):
    """Compare observed channels with the rotation recomposed from RSS sources."""

    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for index, axis in enumerate(axes):
        axis.plot(result.scenario.time_seconds, result.scenario.observations[index], label="Observed")
        axis.plot(
            result.scenario.time_seconds,
            result.reconstructed_observations[index],
            label="Source-composed",
            alpha=0.85,
        )
        axis.set_title(f"Observer channel {index + 1}")
        axis.grid(True, alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("Time (s)")
    figure.tight_layout()
    return figure
