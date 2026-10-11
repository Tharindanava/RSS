"""Weighted multivariate functional PCA for normalized lower-limb proxies."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .models import MFPCAResult
from .progress import ProgressTracker, progress_note


def trajectory_tensor(
    trajectories: pd.DataFrame,
    nodes: Sequence[str],
    representation: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    selected = trajectories[trajectories["representation"] == representation]
    index_columns = ["participant_id", "trial_id", "recording_id", "step_pair_id"]
    wide = selected.pivot_table(index=index_columns, columns=["node", "percent_step"], values="value")
    desired = [(node, point) for node in nodes for point in sorted(selected["percent_step"].unique())]
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(desired, names=["node", "percent_step"])).dropna()
    # ``reset_index`` retains the pivoted column MultiIndex, turning identifiers
    # into tuples such as ("participant_id", "").  ``itertuples`` then renames
    # those invalid fields to ``_0``, ``_1``, ... and downstream weighting cannot
    # access ``row.participant_id``.  Build metadata directly from the row index
    # so the identifier columns remain ordinary strings.
    metadata = wide.index.to_frame(index=False).reindex(columns=index_columns)
    if wide.empty:
        return np.empty((0, len(nodes), 0)), metadata
    points = len(wide.columns) // len(nodes)
    return wide.to_numpy(float).reshape(len(wide), len(nodes), points), metadata


def hierarchical_step_pair_weights(metadata: pd.DataFrame) -> np.ndarray:
    """Equal participant, trial and matched-step-pair weight."""
    if metadata.empty:
        return np.empty(0)
    participant_counts = metadata.groupby("participant_id")["trial_id"].nunique()
    pair_counts = metadata.groupby(["participant_id", "trial_id"])["step_pair_id"].count()
    participant_total = float(metadata["participant_id"].nunique())
    weights = []
    for row in metadata.itertuples(index=False):
        weights.append(
            1.0 / participant_total
            / float(participant_counts.loc[row.participant_id])
            / float(pair_counts.loc[(row.participant_id, row.trial_id)])
        )
    values = np.asarray(weights, dtype=float)
    return values / values.sum()


def fit_mfpca(
    trajectories: np.ndarray,
    sample_weights: np.ndarray | None = None,
    *,
    channel_weighting: str = "inverse_integrated_variance",
    target_fve: float = 0.95,
    variance_epsilon: float = 1.0e-10,
    sample_ids: list[str] | None = None,
    n_components: int | None = None,
) -> MFPCAResult:
    """Fit MFPCA by weighted covariance/SVD under a functional inner product."""
    values = np.asarray(trajectories, dtype=float)
    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError("MFPCA needs at least two complete step-pair trajectories [pair, channel, point]")
    n_samples, n_channels, n_points = values.shape
    weights = np.full(n_samples, 1.0 / n_samples) if sample_weights is None else np.asarray(sample_weights, dtype=float)
    if weights.shape != (n_samples,) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("sample_weights must be non-negative and match the number of step pairs")
    weights = weights / weights.sum()
    mean = np.einsum("i,icp->cp", weights, values)
    centered = values - mean
    integrated_variance = np.einsum("i,icp->c", weights, centered * centered) / n_points
    if channel_weighting == "equal":
        channel_weights = np.ones(n_channels)
    elif channel_weighting == "inverse_integrated_variance":
        channel_weights = 1.0 / np.maximum(integrated_variance, variance_epsilon)
        channel_weights *= n_channels / channel_weights.sum()
    else:
        raise ValueError(f"Unknown channel weighting: {channel_weighting}")
    metric_scale = np.sqrt(np.repeat(channel_weights / n_points, n_points))
    flat = centered.reshape(n_samples, -1)
    weighted_design = np.sqrt(weights)[:, None] * flat * metric_scale[None, :]
    _, singular_values, right = np.linalg.svd(weighted_design, full_matrices=False)
    eigenvalues = singular_values ** 2
    positive = eigenvalues > variance_epsilon
    eigenvalues, right = eigenvalues[positive], right[positive]
    if not eigenvalues.size:
        raise ValueError("All functional channels have zero variance")
    explained = eigenvalues / eigenvalues.sum()
    if n_components is None:
        resolved_components = int(np.searchsorted(np.cumsum(explained), target_fve) + 1)
    else:
        resolved_components = max(1, min(int(n_components), len(eigenvalues)))
    n_components = resolved_components
    eigenfunctions_flat = right / metric_scale[None, :]
    scores = (flat * (metric_scale ** 2)[None, :]) @ eigenfunctions_flat.T
    reconstructed = scores[:, :n_components] @ eigenfunctions_flat[:n_components]
    error = np.sqrt(np.sum((flat - reconstructed) ** 2 * (metric_scale ** 2), axis=1))
    return MFPCAResult(
        mean, eigenfunctions_flat.reshape(-1, n_channels, n_points), eigenvalues,
        channel_weights, scores, explained, error, n_components,
        sample_ids or [str(index) for index in range(n_samples)],
        {"channel_weighting": channel_weighting, "target_fve": target_fve},
    )


def run_mfpca(trajectories: pd.DataFrame, config: AnalysisConfig) -> tuple[MFPCAResult, pd.DataFrame]:
    settings = config.section("mfpca")
    tensor, metadata = trajectory_tensor(trajectories, settings["nodes"], str(settings["representation"]))
    weights = hierarchical_step_pair_weights(metadata)
    ids = metadata["step_pair_id"].astype(str).tolist()
    result = fit_mfpca(
        tensor, weights, channel_weighting=str(settings["channel_weighting"]),
        target_fve=float(settings["target_fve"]), variance_epsilon=float(settings["variance_epsilon"]),
        sample_ids=ids,
    )
    score_columns = {f"mfpca_score_{index + 1}": result.scores[:, index] for index in range(result.n_components)}
    scores = pd.concat((metadata.reset_index(drop=True), pd.DataFrame(score_columns)), axis=1)
    context_columns = [
        column for column in ("step_pair_id", "left_step_id", "right_step_id", "terrain", "direction", "footwear", "raw_file", "rss_file")
        if column in trajectories
    ]
    if len(context_columns) > 1:
        scores = scores.merge(trajectories[context_columns].drop_duplicates("step_pair_id"), on="step_pair_id", how="left")
    scores["mfpca_reconstruction_error"] = result.reconstruction_error
    trial = scores.groupby(["participant_id", "trial_id"], as_index=False).median(numeric_only=True)
    for column in score_columns:
        mad = scores.groupby(["participant_id", "trial_id"])[column].apply(
            lambda values: 1.4826 * np.median(np.abs(values - np.median(values)))
        ).rename(f"{column}_scaled_mad").reset_index()
        trial = trial.merge(mad, on=["participant_id", "trial_id"], how="left")
    return result, scores.merge(trial, on=["participant_id", "trial_id"], how="left", suffixes=("", "_trial"))


def mfpca_stability_diagnostics(
    trajectories: pd.DataFrame,
    reference: MFPCAResult,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Quantify leave-one-participant-out and participant-bootstrap subspace stability."""
    settings = config.section("mfpca")
    tensor, metadata = trajectory_tensor(trajectories, settings["nodes"], str(settings["representation"]))
    if len(tensor) < 3:
        return pd.DataFrame()
    reference_flat = reference.eigenfunctions[: reference.n_components].reshape(reference.n_components, -1)
    metric = np.sqrt(np.repeat(reference.channel_weights / tensor.shape[2], tensor.shape[2]))
    reference_metric = reference_flat * metric[None, :]

    def fit_subset(indices: np.ndarray, subset_metadata: pd.DataFrame, kind: str, label: str) -> dict[str, float | str | int]:
        fitted = fit_mfpca(
            tensor[indices], hierarchical_step_pair_weights(subset_metadata),
            channel_weighting=str(settings["channel_weighting"]), target_fve=float(settings["target_fve"]),
            variance_epsilon=float(settings["variance_epsilon"]),
        )
        candidate = fitted.eigenfunctions[: reference.n_components].reshape(-1, tensor.shape[1] * tensor.shape[2])
        candidate *= metric[None, :]
        singular_values = np.linalg.svd(reference_metric @ candidate.T, compute_uv=False)
        return {
            "resample_type": kind, "resample": label, "n_step_pairs": len(indices),
            "n_components": fitted.n_components,
            "subspace_similarity": float(np.mean(np.clip(singular_values, 0.0, 1.0) ** 2)),
        }

    rows: list[dict[str, float | str | int]] = []
    participants = metadata["participant_id"].astype(str).unique()
    progress_note(
        "mfpca stability",
        f"step_pairs={len(tensor)} participants={len(participants)} components={reference.n_components}",
    )
    loo_tracker = ProgressTracker("mfpca leave-one-participant-out", len(participants))
    for index, participant in enumerate(participants, start=1):
        indices = np.flatnonzero(metadata["participant_id"].astype(str).to_numpy() != participant)
        if len(indices) >= 2:
            rows.append(fit_subset(indices, metadata.iloc[indices].reset_index(drop=True), "leave_one_participant_out", participant))
        loo_tracker.update(index, f"participant={participant}")
    rng = np.random.default_rng(int(config.values["random_seed"]))
    repetitions = int(settings["bootstrap_repetitions"])
    bootstrap_tracker = ProgressTracker(
        "mfpca participant bootstrap", repetitions, report_every=max(1, repetitions // 10)
    )
    for repetition in range(repetitions):
        sampled = rng.choice(participants, len(participants), replace=True)
        pieces, meta_pieces = [], []
        for draw, participant in enumerate(sampled):
            indices = np.flatnonzero(metadata["participant_id"].astype(str).to_numpy() == participant)
            pieces.extend(indices.tolist())
            piece = metadata.iloc[indices].copy()
            piece["participant_id"] = f"bootstrap_{draw}"
            meta_pieces.append(piece)
        indices = np.asarray(pieces, dtype=int)
        if len(indices) >= 2:
            rows.append(fit_subset(indices, pd.concat(meta_pieces, ignore_index=True), "participant_bootstrap", str(repetition)))
        bootstrap_tracker.update(repetition + 1, f"completed resample={repetition}")
    return pd.DataFrame(rows)
