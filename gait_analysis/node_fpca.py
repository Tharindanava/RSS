"""Per-node (single-channel) functional PCA, applied independently to every joint.

This fully replaces the old hand-built classic+MFPCA+TE blended clustering feature
vector: one independent FPCA is fit per node (all 12 joints, not just the legs),
against the ``node_step_trajectories.csv.gz`` table that Part A's
``normalize_node_trajectories`` produces. ``trajectory_tensor`` and
``hierarchical_step_pair_weights`` from ``mfpca.py`` are reused unchanged (a
singleton ``[node]`` list naturally yields the per-node, single-channel tensor);
``fit_mfpca`` is reused with its additive ``n_components`` override so each node
gets a fixed-rank fit instead of a variance-target search.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .mfpca import fit_mfpca, hierarchical_step_pair_weights, trajectory_tensor
from .models import MFPCAResult
from .progress import progress_note
from .utils import robust_zscore


def fit_all_node_fpca(
    trajectories: pd.DataFrame, config: AnalysisConfig
) -> dict[str, tuple[MFPCAResult, pd.DataFrame]]:
    """Fit one independent single-channel FPCA per configured node.

    Reads ``config.section("node_fpca")`` for the node list, representation and
    number of eigenfunctions ``K``. Nodes with fewer than two complete step
    windows (or whose single channel has zero variance) are skipped with a
    logged reason rather than raising. Each retained node's per-step/per-cycle
    metadata gains ``node_fpca__{node}__score_{1..K}`` columns, zero-padded when
    a low-rank node yields fewer than ``K`` real components.
    """
    settings = config.section("node_fpca")
    k = int(settings["n_eigenfunctions"])
    representation = str(settings["representation"])
    channel_weighting = str(settings.get("channel_weighting", "equal"))
    variance_epsilon = float(settings["variance_epsilon"])
    results: dict[str, tuple[MFPCAResult, pd.DataFrame]] = {}
    for node in settings["nodes"]:
        tensor, metadata = trajectory_tensor(trajectories, [node], representation)
        if len(tensor) < 2:
            progress_note(
                "node_fpca",
                f"skipping node={node}: fewer than two complete step windows ({len(tensor)})",
            )
            continue
        weights = hierarchical_step_pair_weights(metadata)
        ids = metadata["step_pair_id"].astype(str).tolist()
        try:
            result = fit_mfpca(
                tensor, weights, channel_weighting=channel_weighting,
                variance_epsilon=variance_epsilon, n_components=k, sample_ids=ids,
            )
        except ValueError as exc:
            progress_note("node_fpca", f"skipping node={node}: {exc}")
            continue
        n_real = result.n_components
        score_columns = {
            f"node_fpca__{node}__score_{index + 1}": (
                result.scores[:, index] if index < n_real else np.zeros(len(metadata))
            )
            for index in range(k)
        }
        scored_metadata = pd.concat((metadata.reset_index(drop=True), pd.DataFrame(score_columns)), axis=1)
        results[node] = (result, scored_metadata)
        progress_note(
            "node_fpca",
            f"fitted node={node} step_windows={len(tensor)} real_components={n_real}/{k}",
        )
    return results


def aggregate_node_trial_scores(
    node_results: dict[str, tuple[MFPCAResult, pd.DataFrame]], k: int
) -> pd.DataFrame:
    """Median-aggregate each node's per-step scores to one row per trial.

    Mirrors the aggregation convention in ``mfpca.py::run_mfpca`` (group by
    ``["participant_id", "trial_id"]``, take the median), then inner-merges the
    per-node trial frames together so the result has exactly one row per trial
    that has at least one accepted cycle for every node present in
    ``node_results``. A node absent from ``node_results`` (skipped during
    fitting) simply contributes no columns and does not constrain the merge.
    """
    trial_frames: list[pd.DataFrame] = []
    for node, (_, scored_metadata) in node_results.items():
        score_columns = [f"node_fpca__{node}__score_{index + 1}" for index in range(k)]
        trial = scored_metadata.groupby(["participant_id", "trial_id"], as_index=False)[score_columns].median()
        trial_frames.append(trial)
    if not trial_frames:
        return pd.DataFrame()
    merged = trial_frames[0]
    for frame in trial_frames[1:]:
        merged = merged.merge(frame, on=["participant_id", "trial_id"], how="inner")
    return merged


def node_score_matrix(
    trial_scores: pd.DataFrame,
    nodes: Sequence[str],
    k: int,
    node_weights: Mapping[str, float],
) -> tuple[np.ndarray, list[str]]:
    """Per-node robust-z-scored, weighted score matrix.

    Structurally mirrors ``phenotype.equal_family_matrix`` (robust z-score each
    block, then scale by ``sqrt(weight / normalizer)``), substituting "node" for
    "family": each node contributes exactly ``k`` score columns, scaled by
    ``sqrt(node_weight / len(nodes))`` so no single node dominates the aggregate
    distance just by being included alongside more or fewer peers.
    """
    blocks: list[np.ndarray] = []
    columns: list[str] = []
    for node in nodes:
        node_columns = [f"node_fpca__{node}__score_{index + 1}" for index in range(k)]
        node_columns = [column for column in node_columns if column in trial_scores]
        if not node_columns:
            continue
        scaled, _, _ = robust_zscore(trial_scores[node_columns])
        block = scaled.fillna(0.0).to_numpy(float)
        block *= np.sqrt(float(node_weights.get(node, 1.0)) / max(1, len(nodes)))
        blocks.append(block)
        columns.extend(node_columns)
    if not blocks:
        raise ValueError("No configured node FPCA feature columns are available")
    return np.column_stack(blocks), columns
