"""Reproducible manuscript-oriented figures and figure manifests."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .progress import progress_note


CLUSTER_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#999999"]
NOISE_COLOR = "0.55"


def _scenario_labels(assignments: pd.DataFrame) -> pd.Series:
    terrain = assignments["terrain"].astype(str) if "terrain" in assignments else pd.Series("unknown", index=assignments.index)
    direction = assignments["direction"].astype(str) if "direction" in assignments else pd.Series("unknown", index=assignments.index)
    return terrain + "/" + direction


def plot_scenario_cluster_embedding(assignments: pd.DataFrame, *, title: str, noise_label: int = -1) -> plt.Figure:
    """2D spectral-embedding scatter: fill color marks experiment, outline marks cluster.

    Each point is filled by its terrain/direction experiment and ringed by an
    unfilled marker colored by its cluster, so experiment and cluster membership can
    be read independently off the same scatter instead of competing for one color
    channel. Points DBSCAN leaves unassigned (``cluster == noise_label``) get a gray
    "Noise" ring instead of a cluster color.
    """
    plotting_frame = assignments.copy()
    plotting_frame["scenario_label"] = _scenario_labels(assignments)
    scenarios = sorted(plotting_frame["scenario_label"].unique())
    scenario_cmap = plt.get_cmap("tab20" if len(scenarios) > 10 else "tab10")
    scenario_colors = {name: scenario_cmap(index % scenario_cmap.N) for index, name in enumerate(scenarios)}
    clusters = sorted(cluster for cluster in plotting_frame["cluster"].unique() if cluster != noise_label)
    cluster_colors = {cluster: CLUSTER_COLORS[index % len(CLUSTER_COLORS)] for index, cluster in enumerate(clusters)}
    has_noise = bool((plotting_frame["cluster"] == noise_label).any())

    figure, axis = plt.subplots(figsize=(8, 6.4))
    for row in plotting_frame.itertuples(index=False):
        axis.scatter(
            row.spectral_coordinate_1, row.spectral_coordinate_2,
            color=scenario_colors[row.scenario_label], marker="o", s=55, edgecolor="white", linewidth=0.5, zorder=3,
        )
        ring_color = NOISE_COLOR if row.cluster == noise_label else cluster_colors[row.cluster]
        axis.scatter(
            row.spectral_coordinate_1, row.spectral_coordinate_2,
            facecolor="none", edgecolor=ring_color, marker="o", s=170, linewidth=1.8, zorder=2,
        )
    scenario_handles = [Patch(facecolor=scenario_colors[name], edgecolor="none", label=name) for name in scenarios]
    cluster_handles = [
        Line2D(
            [0], [0], marker="o", markerfacecolor="none", markeredgecolor=cluster_colors[cluster],
            markeredgewidth=1.8, linestyle="", markersize=10, label=f"Cluster {cluster}",
        )
        for cluster in clusters
    ]
    if has_noise:
        cluster_handles.append(Line2D(
            [0], [0], marker="o", markerfacecolor="none", markeredgecolor=NOISE_COLOR,
            markeredgewidth=1.8, linestyle="", markersize=10, label="Noise",
        ))
    figure.subplots_adjust(right=0.66)
    figure.legend(
        handles=scenario_handles, title="Experiment (terrain/direction)",
        loc="upper left", bbox_to_anchor=(0.68, 0.92),
    )
    figure.legend(
        handles=cluster_handles, title="Cluster",
        loc="lower left", bbox_to_anchor=(0.68, 0.08), fontsize=8,
    )
    axis.set(xlabel="Spectral coordinate 1", ylabel="Spectral coordinate 2", title=title)
    axis.grid(alpha=0.18)
    return figure


def plot_cluster_embedding(assignments: pd.DataFrame, *, title: str) -> plt.Figure:
    """2D spectral-embedding scatter, colored by cluster, one marker per participant.

    Extracted from the frozen historical report's ``plot_clustering``
    (``report/build_report.py``) so the eigenfunction node-FPCA clustering pathway can
    reuse the same visual convention without depending on that module's hardcoded run
    identity / matrix-shape assertions (which stay pinned to their one historical run).
    """
    participants = sorted(assignments["participant_id"].astype(str).unique())
    marker_pool = ["o", "s", "^", "D", "P", "X", "v"]
    participant_markers = {participant: marker_pool[index % len(marker_pool)] for index, participant in enumerate(participants)}
    has_scenario = {"terrain", "direction", "footwear"}.issubset(assignments.columns)
    figure, axis = plt.subplots(figsize=(9, 6.8))
    for row in assignments.itertuples(index=False):
        color = CLUSTER_COLORS[int(row.cluster) % len(CLUSTER_COLORS)]
        marker = participant_markers[str(row.participant_id)]
        axis.scatter(
            row.spectral_coordinate_1, row.spectral_coordinate_2,
            color=color, marker=marker, s=85, edgecolor="white", linewidth=0.7,
        )
        label = f"{row.participant_id}\n{row.terrain}/{row.direction}; {row.footwear}" if has_scenario else str(row.participant_id)
        axis.annotate(label, (row.spectral_coordinate_1, row.spectral_coordinate_2), xytext=(4, 4), textcoords="offset points", fontsize=6)
    cluster_handles = [
        Patch(color=CLUSTER_COLORS[int(cluster) % len(CLUSTER_COLORS)], label=f"Cluster {cluster}")
        for cluster in sorted(assignments["cluster"].unique())
    ]
    participant_handles = [Line2D([0], [0], marker=participant_markers[p], color="0.35", linestyle="", label=p) for p in participants]
    figure.subplots_adjust(right=0.66)
    figure.legend(handles=cluster_handles, title="Solution", loc="upper left", bbox_to_anchor=(0.68, 0.92))
    figure.legend(handles=participant_handles, title="Participant", loc="lower left", bbox_to_anchor=(0.68, 0.08), fontsize=7)
    axis.set(xlabel="Spectral coordinate 1", ylabel="Spectral coordinate 2", title=title)
    axis.grid(alpha=0.18)
    return figure


def create_figures(run_dir: str | Path, *, statistics: pd.DataFrame | None = None, clusters: pd.DataFrame | None = None) -> pd.DataFrame:
    output = Path(run_dir) / "figures"
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    def save(figure: plt.Figure, name: str, source: str, *, suffix: str = "png") -> None:
        figure.tight_layout()
        path = output / f"{name}.{suffix}"
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{name}.", suffix=f".{suffix}", dir=output)
        os.close(descriptor)
        try:
            figure.savefig(temporary_name, dpi=180, bbox_inches="tight")
            os.replace(temporary_name, path)
        finally:
            plt.close(figure)
            Path(temporary_name).unlink(missing_ok=True)
        manifest.append({"figure": name, "path": str(path), "source": source, "signal_type": "rss_segment_mode1_proxy"})
        progress_note("figures", f"saved {name}.{suffix} from {source}")

    run_path = Path(run_dir)
    cohort_path = run_path / "cohort_manifest.csv"
    if cohort_path.exists():
        cohort = pd.read_csv(cohort_path)
        counts = cohort["terrain"].value_counts()
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.bar(counts.index.astype(str), counts.values)
        axis.set(xlabel="Terrain", ylabel="Source recordings", title="Frozen gait cohort")
        axis.tick_params(axis="x", rotation=35)
        save(figure, "cohort_by_terrain", "cohort_manifest.csv")

    step_path = run_path / "gait_steps.csv"
    if step_path.exists():
        steps = pd.read_csv(step_path)
        ankle_steps = steps[steps["node"].isin(["LA", "RA"])] if "node" in steps else steps
        if not ankle_steps.empty:
            counts = ankle_steps.groupby(["node_side", "accepted"]).size().unstack(fill_value=0)
            figure, axis = plt.subplots(figsize=(6, 4))
            counts.plot.bar(ax=axis)
            axis.set(xlabel="Side", ylabel="Candidate ankle-driven gait cycles", title="Limb-specific segmentation QC")
            save(figure, "segmentation_qc", "gait_steps.csv")

    trajectory_path = run_path / "normalized_step_trajectories.csv.gz"
    if trajectory_path.exists():
        trajectories = pd.read_csv(trajectory_path)
        if {"representation", "node", "percent_step", "value"}.issubset(trajectories):
            trajectories = trajectories[
                (trajectories["representation"] == "event_registered")
                & trajectories["node"].isin(["LH", "LK", "LA", "RH", "RK", "RA"])
            ]
        if not trajectories.empty:
            mean = trajectories.groupby(["node", "percent_step"])["value"].median().reset_index()
            figure, axis = plt.subplots(figsize=(8, 5))
            for node, group in mean.groupby("node"):
                axis.plot(group["percent_step"], group["value"], label=node)
            axis.axvline(60.0, color="0.5", linestyle="--", linewidth=1)
            axis.set(xlabel="Event-registered step (%)", ylabel="RSS Mode 1 proxy", title="Lower-limb normalized trajectories")
            axis.legend(ncol=3)
            save(figure, "lower_limb_trajectories", "normalized_step_trajectories.csv.gz")

    model_path = run_path / "mfpca" / "mfpca_model.npz"
    if model_path.exists():
        with np.load(model_path) as model:
            explained = model["explained_variance_ratio"]
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(np.arange(1, len(explained) + 1), np.cumsum(explained), marker="o")
        axis.axhline(0.95, color="0.5", linestyle="--")
        axis.set(xlabel="MFPCA component", ylabel="Cumulative FVE", ylim=(0, 1.02), title="MFPCA variance explained")
        save(figure, "mfpca_fve", "mfpca/mfpca_model.npz")

    te_path = run_path / "te_network_features.csv"
    if te_path.exists():
        te = pd.read_csv(te_path)
        block_columns = [
            column for column in ("te_lower_to_lower", "te_lower_to_upper", "te_upper_to_upper", "te_upper_to_lower")
            if column in te
        ]
        if block_columns and "terrain" in te:
            summary = te.groupby("terrain")[block_columns].median()
            figure, axis = plt.subplots(figsize=(9, 5))
            summary.plot.bar(ax=axis)
            axis.set(xlabel="Terrain", ylabel="Mean significant effective TE", title="Whole-body block transfer")
            axis.tick_params(axis="x", rotation=35)
            save(figure, "te_block_transfer", "te_network_features.csv")

    classic_path = run_path / "classic_trial_features.csv"
    if classic_path.exists():
        classic = pd.read_csv(classic_path)
        if {"participant_id", "terrain", "stride_duration_seconds"}.issubset(classic):
            linked = classic.groupby(["participant_id", "terrain"])["stride_duration_seconds"].median().unstack()
            if not linked.empty:
                figure, axis = plt.subplots(figsize=(8, 5))
                x = np.arange(len(linked.columns))
                for participant, values in linked.iterrows():
                    axis.plot(x, values, marker="o", alpha=0.7, label=str(participant))
                axis.set_xticks(x, linked.columns, rotation=35)
                axis.set(xlabel="Terrain", ylabel="Stride duration (s)", title="Participant-linked trial estimates")
                axis.legend(fontsize=7, ncol=2)
                save(figure, "participant_linked_stride", "classic_trial_features.csv")

    if statistics is not None and not statistics.empty:
        figure, axis = plt.subplots(figsize=(8, max(3, 0.35 * len(statistics))))
        y = np.arange(len(statistics))
        estimates = pd.to_numeric(statistics["estimate"], errors="coerce").to_numpy(float)
        ci_low = pd.to_numeric(statistics["ci_low"], errors="coerce").to_numpy(float)
        ci_high = pd.to_numeric(statistics["ci_high"], errors="coerce").to_numpy(float)
        finite_intervals = np.isfinite(ci_low) & np.isfinite(ci_high)
        axis.hlines(y[finite_intervals], ci_low[finite_intervals], ci_high[finite_intervals])
        finite_estimates = np.isfinite(estimates)
        axis.scatter(estimates[finite_estimates], y[finite_estimates], marker="o")
        axis.axvline(0.0, color="0.4", linewidth=1)
        axis.set_yticks(y, statistics["outcome"].astype(str) + ": " + statistics["condition"].astype(str))
        axis.set_xlabel("Within-participant trial effect (RSS proxy units)")
        save(figure, "condition_effects", "statistical_results.csv")

    if clusters is not None and not clusters.empty:
        counts = clusters.groupby(["analysis", "cluster"]).size() if "analysis" in clusters else clusters["cluster"].value_counts().sort_index()
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.bar([str(index) for index in counts.index], counts.values)
        axis.set(xlabel="Analysis / spectral cluster", ylabel="Trials", title="Cluster sizes")
        axis.tick_params(axis="x", rotation=35)
        save(figure, "cluster_sizes", "cluster_assignments.csv")

    for scope in ("all", "lower", "upper"):
        scope_name = f"cluster_assignments_{scope}.csv"
        scope_path = run_path / scope_name
        if not scope_path.exists():
            continue
        scope_assignments = pd.read_csv(scope_path)
        required_columns = {"spectral_coordinate_1", "spectral_coordinate_2", "cluster", "participant_id"}
        if scope_assignments.empty or not required_columns.issubset(scope_assignments.columns):
            continue
        figure = plot_cluster_embedding(
            scope_assignments, title=f"Eigenfunction spectral embedding ({scope} limbs)"
        )
        save(figure, f"cluster_spectral_embedding_{scope}", scope_name, suffix="pdf")

    return pd.DataFrame(manifest, columns=["figure", "path", "source", "signal_type"])
