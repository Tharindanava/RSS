"""Stable real-data CSV and diagnostic exporters for RSS."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from rss.quaternion_series import quaternion_to_axis_angle_series
from rss.ssa_preprocessing import SSAPreprocessResult


def resolve_realdata_output_dir(output_root: str, csv_path: str) -> str:
    input_path = Path(csv_path).expanduser().resolve()
    indices = [index for index, part in enumerate(input_path.parent.parts) if part.lower() == "sample_data"]
    if not indices:
        return os.path.join(output_root, Path(csv_path).stem)
    relative = input_path.parent.parts[indices[-1] + 1 :]
    return str(Path(output_root).expanduser().resolve().joinpath(*relative, input_path.stem))


def realdata_output_dir(output_root: str, csv_path: str) -> str:
    path = resolve_realdata_output_dir(output_root, csv_path)
    os.makedirs(path, exist_ok=True)
    return path


def _base(frame_index: np.ndarray, time_seconds: np.ndarray | None, *, mode: bool = False) -> dict[str, np.ndarray]:
    rows: dict[str, np.ndarray] = {"frame": np.asarray(frame_index, dtype=int)}
    if time_seconds is not None:
        rows["time" if mode else "time_seconds"] = np.asarray(time_seconds, dtype=float)
    return rows


def export_limb_quaternion_csv(
    *,
    output_dir: str,
    frame_index: np.ndarray,
    time_seconds: np.ndarray | None,
    limb_quaternions: dict[str, np.ndarray],
) -> str:
    rows = _base(frame_index, time_seconds)
    for limb in sorted(limb_quaternions):
        q = np.asarray(limb_quaternions[limb], dtype=float).reshape(-1, 4)
        rows[f"{limb}_w"] = q[:, 0]
        rows[f"{limb}_x"] = q[:, 1]
        rows[f"{limb}_y"] = q[:, 2]
        rows[f"{limb}_z"] = q[:, 3]
    out_path = os.path.join(output_dir, "limb_quaternion.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def export_limb_quaternion_axis_angle_csv(
    *,
    output_dir: str,
    frame_index: np.ndarray,
    time_seconds: np.ndarray | None,
    limb_quaternions: dict[str, np.ndarray],
) -> str:
    rows = _base(frame_index, time_seconds)
    for limb in sorted(limb_quaternions):
        q = np.asarray(limb_quaternions[limb], dtype=float).reshape(-1, 4)
        angle, axis = quaternion_to_axis_angle_series(q)
        rows[f"{limb}_angle"] = angle
        rows[f"{limb}_x"] = axis[:, 0]
        rows[f"{limb}_y"] = axis[:, 1]
        rows[f"{limb}_z"] = axis[:, 2]
    out_path = os.path.join(output_dir, "limb_quaternion_axis_angle.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def export_limb_observed_signals_csv(
    *, output_dir: str, frame_index: np.ndarray, time_seconds: np.ndarray | None,
    observed_by_limb: dict[str, np.ndarray], reconstructed_by_limb: dict[str, np.ndarray],
) -> str:
    rows = _base(frame_index, time_seconds)
    for limb in sorted(observed_by_limb):
        observed = np.asarray(observed_by_limb[limb]).reshape(3, -1)
        reconstructed = np.asarray(reconstructed_by_limb[limb]).reshape(3, -1)
        for index, axis in enumerate(("x", "y", "z")):
            rows[f"{limb}_observer_{axis}"] = observed[index]
        for index, axis in enumerate(("x", "y", "z")):
            rows[f"{limb}_reconstructed_{axis}"] = reconstructed[index]
    path = os.path.join(output_dir, "limb_observed_signals.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def export_limb_observed_coefficients_csv(
    *, output_dir: str, frame_index: np.ndarray, time_seconds: np.ndarray | None,
    observed_quaternions_by_limb: dict[str, np.ndarray], observed_coefficients_by_limb: dict[str, np.ndarray],
    reconstructed_quaternions_by_limb: dict[str, np.ndarray], reconstructed_coefficients_by_limb: dict[str, np.ndarray],
) -> str:
    rows = _base(frame_index, time_seconds)
    for limb in sorted(observed_coefficients_by_limb):
        oq = np.asarray(observed_quaternions_by_limb[limb]).reshape(-1, 4)
        rq = np.asarray(reconstructed_quaternions_by_limb[limb]).reshape(-1, 4)
        observed = np.asarray(observed_coefficients_by_limb[limb]).reshape(3, -1)
        reconstructed = np.asarray(reconstructed_coefficients_by_limb[limb]).reshape(3, -1)
        rows[f"{limb}_observer_w"] = oq[:, 0]
        for index, axis in enumerate(("x", "y", "z")):
            rows[f"{limb}_observer_{axis}"] = observed[index]
        rows[f"{limb}_reconstructed_w"] = rq[:, 0]
        for index, axis in enumerate(("x", "y", "z")):
            rows[f"{limb}_reconstructed_{axis}"] = reconstructed[index]
    path = os.path.join(output_dir, "limb_observed_coefficients.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def export_limb_angle_mode_csv(
    *, output_dir: str, frame_index: np.ndarray, time_seconds: np.ndarray | None,
    sources_by_limb: dict[str, np.ndarray], original_angles_by_limb: dict[str, np.ndarray],
) -> str:
    rows = _base(frame_index, time_seconds, mode=True)
    limbs = sorted(sources_by_limb)
    for limb in limbs:
        signals = np.asarray(sources_by_limb[limb]).reshape(3, -1)
        for index in range(3):
            rows[f"{limb}_mode_{index + 1}"] = signals[index]
    for limb in limbs:
        rows[f"{limb}_original_angle"] = np.asarray(original_angles_by_limb[limb]).reshape(-1)
    path = os.path.join(output_dir, "limb_angle_mode.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def export_limb_mode_coefficients_csv(
    *, output_dir: str, frame_index: np.ndarray, time_seconds: np.ndarray | None,
    source_coefficients_by_limb: dict[str, np.ndarray], source_quaternions_by_limb: dict[str, np.ndarray],
) -> str:
    rows = _base(frame_index, time_seconds, mode=True)
    limbs = sorted(source_coefficients_by_limb)
    for limb in limbs:
        signals = np.asarray(source_coefficients_by_limb[limb]).reshape(3, -1)
        for index in range(3):
            rows[f"{limb}_mode_{index + 1}"] = signals[index]
    for limb in limbs:
        rows[f"{limb}_quaternion_w"] = np.asarray(source_quaternions_by_limb[limb]).reshape(-1, 4)[:, 0]
    path = os.path.join(output_dir, "limb_mode_coefficients.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def export_limb_axes_mode_csv(
    *, output_dir: str, frame_index: np.ndarray, time_seconds: np.ndarray | None,
    mode_axes_by_limb: dict[str, np.ndarray], original_axes_by_limb: dict[str, np.ndarray],
) -> str:
    rows = _base(frame_index, time_seconds, mode=True)
    limbs = sorted(mode_axes_by_limb)
    for limb in limbs:
        axes = np.asarray(mode_axes_by_limb[limb]).reshape(3, 3)
        for mode in range(3):
            for coordinate, name in enumerate(("x", "y", "z")):
                rows[f"{limb}_mode_{mode + 1}_axis_{name}"] = np.full(len(frame_index), axes[mode, coordinate])
    for limb in limbs:
        axes = np.asarray(original_axes_by_limb[limb]).reshape(-1, 3)
        for coordinate, name in enumerate(("x", "y", "z")):
            rows[f"{limb}_original_axis_{name}"] = axes[:, coordinate]
    path = os.path.join(output_dir, "limb_axes_mode.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def export_limb_mode_metrics_csv(*, output_dir: str, metric_rows: list[dict[str, object]]) -> str:
    path = os.path.join(output_dir, "limb_mode_metrics.csv")
    frame = pd.DataFrame(metric_rows)
    if not frame.empty:
        frame = frame.sort_values(["limb_name", "mode_index"], kind="stable").reset_index(drop=True)
    frame.to_csv(path, index=False)
    return path


def export_ssa_preprocessing_signals_csv(
    *, output_dir: str, frame_index: np.ndarray, time_seconds: np.ndarray | None,
    ssa_by_limb_axis: dict[str, dict[str, SSAPreprocessResult]],
) -> str:
    rows = _base(frame_index, time_seconds)
    for limb in sorted(ssa_by_limb_axis):
        for axis in ("x", "y", "z"):
            result = ssa_by_limb_axis[limb][axis]
            rows[f"{limb}_observer_{axis}_raw"] = result.raw_signal
            rows[f"{limb}_observer_{axis}_cleaned"] = result.cleaned_signal
            rows[f"{limb}_observer_{axis}_removed"] = result.removed_signal
    path = os.path.join(output_dir, "ssa_preprocessing_signals.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def export_ssa_drift_correction_signals_csv(
    *, output_dir: str, frame_index: np.ndarray, time_seconds: np.ndarray | None,
    ssa_by_limb_axis: dict[str, dict[str, SSAPreprocessResult]],
) -> str:
    rows = _base(frame_index, time_seconds)
    for limb in sorted(ssa_by_limb_axis):
        for axis in ("x", "y", "z"):
            result = ssa_by_limb_axis[limb][axis]
            rows[f"{limb}_observer_{axis}_raw"] = result.raw_signal
            rows[f"{limb}_observer_{axis}_drift_corrected"] = result.drift_corrected_signal
            rows[f"{limb}_observer_{axis}_removed_drift"] = result.removed_drift_signal
    path = os.path.join(output_dir, "ssa_drift_correction_signals.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def export_ssa_component_metrics_csv(*, output_dir: str, component_rows: list[dict[str, object]]) -> str:
    path = os.path.join(output_dir, "ssa_component_metrics.csv")
    frame = pd.DataFrame(component_rows)
    if not frame.empty:
        frame = frame.sort_values(["limb_name", "axis", "component_index"], kind="stable")
    frame.to_csv(path, index=False)
    return path


def ssa_plot_dir(output_dir: str) -> str:
    path = os.path.join(output_dir, "ssa_plots")
    os.makedirs(path, exist_ok=True)
    return path


def export_ssa_axis_plots(*, plot_dir: str, limb_name: str, axis_name: str, result: SSAPreprocessResult) -> None:
    note_path = os.path.join(plot_dir, "ssa_plots_skipped.txt")
    try:
        if not os.environ.get("MPLCONFIGDIR"):
            os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib-rss")
            os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"drift": "tab:red", "noise": "tab:gray", "kept": "tab:blue"}
        figure, axis = plt.subplots(figsize=(12, 4.5))
        indices = np.arange(1, len(result.component_summaries) + 1)
        axis.bar(
            indices,
            [item.variance_share_pct for item in result.component_summaries],
            color=[colors[item.classification] for item in result.component_summaries],
        )
        axis.set_title(f"{limb_name} {axis_name.upper()} SSA Variance Share (L={result.window_length})")
        axis.set_xlabel("Component index")
        axis.set_ylabel("Variance share (%)")
        axis.grid(True, axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(os.path.join(plot_dir, f"{limb_name}_{axis_name}_variance_share.png"), dpi=150)
        plt.close(figure)

        selected = [
            index
            for index, item in enumerate(result.component_summaries)
            if item.classification in {"drift", "kept"}
        ]
        noise = sorted(
            (
                (-item.variance_share_pct, index)
                for index, item in enumerate(result.component_summaries)
                if item.classification == "noise"
            )
        )
        selected = sorted(set(selected + [index for _, index in noise[:10]]))
        page_size = 30
        for page, start in enumerate(range(0, len(selected), page_size), start=1):
            page_indices = selected[start : start + page_size]
            component_figure, component_axes = plt.subplots(5, 6, figsize=(18, 12), sharex=True)
            for component_axis in component_axes.reshape(-1):
                component_axis.set_visible(False)
            for slot, component_index in enumerate(page_indices):
                component_axis = component_axes.reshape(-1)[slot]
                component_axis.set_visible(True)
                summary = result.component_summaries[component_index]
                component_axis.plot(result.component_signals[component_index], color=colors[summary.classification], linewidth=0.9)
                component_axis.set_title(
                    f"C{summary.component_index} {summary.variance_share_pct:.2f}% {summary.classification}",
                    fontsize=8,
                )
                component_axis.grid(True, alpha=0.2)
            component_figure.suptitle(f"{limb_name} {axis_name.upper()} Filtered SSA Components")
            component_figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
            component_figure.savefig(
                os.path.join(plot_dir, f"{limb_name}_{axis_name}_components_page_{page:02d}.png"),
                dpi=130,
            )
            plt.close(component_figure)
    except Exception as exc:  # pragma: no cover
        with open(note_path, "w", encoding="utf-8") as handle:
            handle.write(f"Matplotlib could not render SSA plots.\nReason: {exc}\n")


def export_loss_history_artifacts(
    *, output_dir: str, loss_history_by_limb: dict[str, tuple[dict[str, float | str | None], ...]],
) -> tuple[str, str]:
    rows = [
        {
            "limb_name": limb,
            "epoch": int(item["epoch"]),
            "learning_rate": (
                None if item.get("learning_rate") is None else float(item["learning_rate"])
            ),
            "loss_type": str(item["loss_type"]),
            "normalized_total_loss": float(item["normalized_total_loss"]),
            "normalized_separation_loss": float(item["normalized_separation_loss"]),
            "normalized_orthogonality_loss": float(item["normalized_orthogonality_loss"]),
            "normalized_reconstruction_loss": float(item["normalized_reconstruction_loss"]),
            "total_loss": float(item["total_loss"]),
            "separation_loss": float(item["separation_loss"]),
            "orthogonality_loss": float(item["orthogonality_loss"]),
            "reconstruction_loss": float(item["reconstruction_loss"]),
            **(
                {
                    "selection_loss": float(item["selection_loss"]),
                    "normalized_selection_loss": float(
                        item["normalized_selection_loss"]
                    ),
                    "regularizer_type": item.get("regularizer_type"),
                    "regularizer_loss": float(item["regularizer_loss"]),
                    "regularizer_weight": float(item["regularizer_weight"]),
                    "regularizer_phase": item.get("regularizer_phase"),
                    "selection_stop_threshold": float(
                        item["selection_stop_threshold"]
                    ),
                    "normalized_regularizer_loss": float(
                        item["normalized_regularizer_loss"]
                    ),
                    "axis_flip_check_epoch": item.get("axis_flip_check_epoch"),
                    "axis_flip_pattern": item.get("axis_flip_pattern"),
                    "axis_flip_candidate_selection_losses": item.get(
                        "axis_flip_candidate_selection_losses"
                    ),
                    "axis_flip_candidate_normalized_selection_losses": item.get(
                        "axis_flip_candidate_normalized_selection_losses"
                    ),
                }
                if "selection_loss" in item
                else {}
            ),
        }
        for limb in sorted(loss_history_by_limb)
        for item in loss_history_by_limb[limb]
    ]
    frame = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "loss_history.csv")
    frame.to_csv(csv_path, index=False)
    plot_path = os.path.join(output_dir, "loss_history.png")
    note_path = os.path.join(output_dir, "loss_history_plot_skipped.txt")
    try:
        if not os.environ.get("MPLCONFIGDIR"):
            os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib-rss")
            os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        for limb, group in frame.groupby("limb_name", sort=True):
            axes[0].plot(group["epoch"], group["normalized_total_loss"], label=limb, linewidth=1.0)
            if "normalized_selection_loss" in group:
                axes[0].plot(
                    group["epoch"],
                    group["normalized_selection_loss"],
                    label=f"{limb} selection",
                    linewidth=0.9,
                    linestyle="--",
                )
                flip_rows = group.dropna(
                    subset=["axis_flip_candidate_selection_losses"]
                )
                if not flip_rows.empty:
                    axes[0].scatter(
                        flip_rows["epoch"],
                        flip_rows["normalized_selection_loss"],
                        marker="x",
                        s=28,
                        label=f"{limb} flip selection",
                    )
        axes[0].axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        if "selection_stop_threshold" in frame:
            thresholds = sorted(
                value
                for value in frame["selection_stop_threshold"].dropna().unique()
                if float(value) > 0.0
            )
            for threshold in thresholds:
                axes[0].axhline(
                    float(threshold),
                    color="tab:red",
                    linestyle=":",
                    linewidth=0.9,
                    label=f"selection threshold {float(threshold):g}",
                )
        component_columns = [
            "normalized_separation_loss",
            "normalized_orthogonality_loss",
            "normalized_reconstruction_loss",
        ]
        if "normalized_regularizer_loss" in frame:
            component_columns.insert(1, "normalized_regularizer_loss")
        components = frame.groupby("epoch")[component_columns].mean()
        for column in component_columns:
            axes[1].plot(components.index, components[column], label=column)
        learning_rate_plotted = False
        regularizer_axis = (
            axes[2].twinx() if "regularizer_weight" in frame else None
        )
        for limb, group in frame.groupby("limb_name", sort=True):
            if regularizer_axis is not None:
                regularizer_axis.plot(
                    group["epoch"],
                    group["regularizer_weight"],
                    label=f"{limb} regularizer weight",
                    linewidth=0.9,
                    linestyle="--",
                )
            applied = group.dropna(subset=["learning_rate"])
            if applied.empty:
                continue
            axes[2].semilogy(
                applied["epoch"],
                applied["learning_rate"],
                label=limb,
                linewidth=1.0,
            )
            learning_rate_plotted = True
        if not learning_rate_plotted:
            axes[2].text(0.5, 0.5, "No optimizer movement was applied", ha="center", va="center", transform=axes[2].transAxes)
        axes[0].set_title("Normalized RSS Total Loss by Limb (Initial Objective = 1)")
        axes[1].set_title("Mean Normalized Weighted RSS Loss Contributions")
        axes[2].set_title("Applied Exponential Learning Rate by Limb")
        axes[2].set_yscale("log")
        axes[2].set_xlabel("Epoch")
        axes[0].set_ylabel("Normalized loss")
        axes[1].set_ylabel("Normalized contribution")
        axes[2].set_ylabel("Learning rate (log scale)")
        if regularizer_axis is not None:
            regularizer_axis.set_ylabel("Regularizer weight")
            regularizer_axis.set_ylim(-0.05, 1.05)
            if regularizer_axis.lines:
                regularizer_axis.legend(fontsize=7, loc="lower right")
        for axis in axes:
            axis.grid(True, alpha=0.25)
            if axis.lines:
                axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(plot_path, dpi=160)
        plt.close(figure)
        return csv_path, plot_path
    except Exception as exc:  # pragma: no cover
        with open(note_path, "w", encoding="utf-8") as handle:
            handle.write(f"Matplotlib could not render the RSS loss plot.\nReason: {exc}\n")
        return csv_path, note_path
