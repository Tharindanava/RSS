"""Local-scale spectral clustering with participant-block stability selection."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from .config import AnalysisConfig
from .progress import ProgressTracker, progress_note


_METADATA_COLUMNS = (
    "trial_id", "participant_id", "terrain", "direction", "footwear", "phenotype_kind",
    "source_recordings", "raw_files", "rss_files",
)


def _assignment_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[[column for column in _METADATA_COLUMNS if column in frame]].copy()


def build_affinity(values: np.ndarray, neighbors: int = 5, method: str = "local_scale_knn_union") -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        raise ValueError("At least two trials are required for an affinity matrix")
    distances = cdist(values, values)
    np.fill_diagonal(distances, np.inf)
    k = min(max(1, int(neighbors)), n - 1)
    finite_distances = distances[np.isfinite(distances)]
    global_scale = float(np.median(finite_distances[finite_distances > 0])) if np.any(finite_distances > 0) else 1.0
    if method == "global_rbf":
        affinity = np.exp(-(distances ** 2) / max(2.0 * global_scale ** 2, 1.0e-12))
        affinity[~np.isfinite(distances)] = 0.0
        np.fill_diagonal(affinity, 0.0)
        return affinity
    if method != "local_scale_knn_union":
        raise ValueError(f"Unknown affinity method: {method}")
    order = np.argsort(distances, axis=1)[:, :k]
    local = np.take_along_axis(distances, order, axis=1)[:, -1]
    local = np.where(np.isfinite(local) & (local > 0), local, global_scale)
    affinity = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in order[i]:
            scale = max(local[i] * local[j], 1.0e-12)
            affinity[i, j] = np.exp(-(distances[i, j] ** 2) / scale)
    return np.maximum(affinity, affinity.T)


def laplacian_embedding(affinity: np.ndarray, dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    degree = affinity.sum(axis=1)
    inverse_sqrt = np.zeros_like(degree)
    inverse_sqrt[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    laplacian = np.eye(len(affinity)) - inverse_sqrt[:, None] * affinity * inverse_sqrt[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    order = np.argsort(eigenvalues)
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    embedding = eigenvectors[:, :dimensions]
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = np.divide(embedding, norms, out=np.zeros_like(embedding), where=norms > 0)
    return embedding, eigenvalues


def _cluster_once(values: np.ndarray, settings: dict[str, Any], k: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    affinity = build_affinity(values, int(settings["neighbors"]), str(settings["affinity"]))
    embedding, eigenvalues = laplacian_embedding(affinity, k)
    labels = KMeans(n_clusters=k, n_init=int(settings["kmeans_n_init"]), random_state=seed).fit_predict(embedding)
    return labels, embedding, eigenvalues


def cluster_phenotypes(
    frame: pd.DataFrame,
    config: AnalysisConfig,
    *,
    matrix_fn: Callable[[pd.DataFrame, Sequence[str]], tuple[np.ndarray, list[str]]],
    blocks: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select K by equal-rank eigengap, silhouette and participant bootstrap ARI.

    ``matrix_fn`` builds the clustering feature matrix from an arbitrary subset of
    ``blocks`` (e.g. node names for the eigenfunction pathway, or feature-family
    names for a hand-built phenotype); ``cluster_phenotypes`` itself is agnostic to
    what a "block" represents, so the same ablation-based stability diagnostics
    apply unchanged to either.
    """
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    values, feature_columns = matrix_fn(frame, blocks)
    settings = config.section("clustering")
    seed = int(config.values["random_seed"])
    participants = frame["participant_id"].astype(str).to_numpy()
    unique_participants = np.unique(participants)
    repetitions = int(config.values["profiles"][config.profile].get("bootstrap_repetitions", settings["bootstrap_repetitions"]))
    rng = np.random.default_rng(seed)
    candidates: list[dict[str, Any]] = []
    labels_by_k: dict[int, np.ndarray] = {}
    embeddings_by_k: dict[int, np.ndarray] = {}
    for k in map(int, settings["candidate_clusters"]):
        if k >= len(frame):
            progress_note("clustering", f"skipping k={k}: requires fewer clusters than trials={len(frame)}")
            continue
        progress_note(
            "clustering",
            f"evaluating k={k} trials={len(frame)} participant_bootstrap_repetitions={repetitions}",
        )
        labels, embedding, eigenvalues = _cluster_once(values, settings, k, seed + k)
        counts = np.bincount(labels, minlength=k)
        valid = bool(np.all(counts >= int(settings["min_cluster_trials"])) and np.all(counts / len(labels) >= float(settings["min_cluster_fraction"])))
        silhouette = float(silhouette_score(embedding, labels)) if len(np.unique(labels)) > 1 else -1.0
        eigengap = float(eigenvalues[k] - eigenvalues[k - 1]) if k < len(eigenvalues) else 0.0
        aris = []
        if len(unique_participants) > 1:
            bootstrap_tracker = ProgressTracker(
                f"clustering bootstrap k={k}", repetitions, report_every=max(1, repetitions // 10)
            )
            for repetition in range(repetitions):
                sampled = rng.choice(unique_participants, len(unique_participants), replace=True)
                selected = np.flatnonzero(np.isin(participants, np.unique(sampled)))
                if len(selected) > k and len(selected) != len(frame):
                    try:
                        boot_labels, _, _ = _cluster_once(values[selected], settings, k, int(rng.integers(0, 2**31 - 1)))
                        aris.append(adjusted_rand_score(labels[selected], boot_labels))
                    except ValueError:
                        pass
                bootstrap_tracker.update(repetition + 1, f"successful_resamples={len(aris)}")
        stability = float(np.mean(aris)) if aris else 0.0
        loo_aris = []
        for participant in unique_participants:
            selected = np.flatnonzero(participants != participant)
            if len(selected) <= k:
                continue
            try:
                loo_labels, _, _ = _cluster_once(values[selected], settings, k, seed + k)
                loo_aris.append(adjusted_rand_score(labels[selected], loo_labels))
            except ValueError:
                continue
        ablation_aris = []
        for omitted in blocks:
            try:
                reduced_values, _ = matrix_fn(frame, [block for block in blocks if block != omitted])
                reduced_labels, _, _ = _cluster_once(reduced_values, settings, k, seed + k)
                ablation_aris.append(adjusted_rand_score(labels, reduced_labels))
            except ValueError:
                continue
        labels_by_k[k] = labels
        embeddings_by_k[k] = embedding
        candidates.append({
            "k": k, "eigengap": eigengap, "silhouette": silhouette,
            "participant_block_ari": stability, "participant_block_ari_sd": float(np.std(aris)) if aris else np.nan,
            "bootstrap_successes": len(aris), "valid": valid,
            "leave_one_participant_out_ari": float(np.mean(loo_aris)) if loo_aris else np.nan,
            "feature_family_ablation_ari": float(np.mean(ablation_aris)) if ablation_aris else np.nan,
            "cluster_sizes": "|".join(map(str, sorted(counts.tolist()))),
        })
        progress_note(
            "clustering",
            f"completed k={k} valid={valid} silhouette={silhouette:.4f} eigengap={eigengap:.4f} "
            f"bootstrap_successes={len(aris)}",
        )
    metrics = pd.DataFrame(candidates)
    valid_metrics = metrics[metrics["valid"]].copy()
    if valid_metrics.empty:
        raise ValueError("No cluster count satisfies the predeclared minimum-size constraints")
    for metric in settings["selection_metrics"]:
        valid_metrics[f"{metric}_rank"] = valid_metrics[metric].rank(method="min", ascending=False)
    rank_columns = [f"{metric}_rank" for metric in settings["selection_metrics"]]
    valid_metrics["mean_rank"] = valid_metrics[rank_columns].mean(axis=1)
    selected_k = int(valid_metrics.sort_values(["mean_rank", "k"]).iloc[0]["k"])
    metrics = metrics.merge(valid_metrics[["k", *rank_columns, "mean_rank"]], on="k", how="left")
    metrics["selected"] = metrics["k"] == selected_k
    assignments = _assignment_metadata(frame)
    assignments["cluster"] = labels_by_k[selected_k]
    assignments["selected_k"] = selected_k
    assignments["feature_count"] = len(feature_columns)
    assignments["affinity"] = str(settings["affinity"])
    for dimension in range(selected_k):
        assignments[f"spectral_coordinate_{dimension + 1}"] = embeddings_by_k[selected_k][:, dimension]
    return assignments, metrics


def assignments_for_k(
    frame: pd.DataFrame,
    config: AnalysisConfig,
    *,
    matrix_fn: Callable[[pd.DataFrame, Sequence[str]], tuple[np.ndarray, list[str]]],
    blocks: Sequence[str],
    k: int,
) -> pd.DataFrame:
    """Rebuild cluster assignments for one specific candidate ``k`` (e.g. the runner-up).

    Reuses the same ``seed + k`` offset ``cluster_phenotypes`` uses when evaluating
    that candidate, so the result is bit-identical to what its sweep already scored.
    """
    values, feature_columns = matrix_fn(frame, blocks)
    settings = config.section("clustering")
    seed = int(config.values["random_seed"])
    labels, embedding, _ = _cluster_once(values, settings, int(k), seed + int(k))
    assignments = _assignment_metadata(frame)
    assignments["cluster"] = labels
    assignments["selected_k"] = int(k)
    assignments["feature_count"] = len(feature_columns)
    assignments["affinity"] = str(settings["affinity"])
    for dimension in range(int(k)):
        assignments[f"spectral_coordinate_{dimension + 1}"] = embedding[:, dimension]
    return assignments


def _dbscan_eps_candidates(embedding: np.ndarray, min_samples: int, percentiles: Sequence[float]) -> list[float]:
    """Plausible eps values from the min_samples-th nearest-neighbor distance distribution.

    Generalizes the standard DBSCAN k-distance/elbow heuristic into a candidate grid
    (percentiles of that distance across all points) so eps can be swept and ranked
    the same way ``candidate_clusters`` sweeps k for KMeans.
    """
    n = len(embedding)
    m = min(max(1, int(min_samples)), n - 1)
    distances = cdist(embedding, embedding)
    np.fill_diagonal(distances, np.inf)
    kth_distances = np.sort(distances, axis=1)[:, m - 1]
    kth_distances = kth_distances[np.isfinite(kth_distances) & (kth_distances > 0)]
    if len(kth_distances) == 0:
        return []
    candidates = sorted({round(float(np.percentile(kth_distances, p)), 10) for p in percentiles})
    return [value for value in candidates if value > 0]


def _cluster_once_dbscan(embedding: np.ndarray, eps: float, min_samples: int, metric: str) -> np.ndarray:
    return DBSCAN(eps=float(eps), min_samples=int(min_samples), metric=metric).fit_predict(embedding)


def cluster_phenotypes_dbscan(
    frame: pd.DataFrame,
    config: AnalysisConfig,
    *,
    matrix_fn: Callable[[pd.DataFrame, Sequence[str]], tuple[np.ndarray, list[str]]],
    blocks: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select DBSCAN's eps by equal-rank silhouette, participant bootstrap ARI and noise fraction.

    Mirrors ``cluster_phenotypes``'s sweep-and-rank selection, substituting a
    fixed-dimensionality spectral embedding + eps sweep for the per-k embedding + k
    sweep: DBSCAN has no analogue of eigengap, so ``clustering.dbscan.embedding_dimensions``
    is fixed rather than swept, and noise fraction (points DBSCAN leaves unassigned,
    ``cluster == -1``) stands in as an extra selection criterion instead.
    """
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    values, feature_columns = matrix_fn(frame, blocks)
    settings = config.section("clustering")
    dbscan_settings = settings["dbscan"]
    seed = int(config.values["random_seed"])
    participants = frame["participant_id"].astype(str).to_numpy()
    unique_participants = np.unique(participants)
    repetitions = int(config.values["profiles"][config.profile].get("bootstrap_repetitions", settings["bootstrap_repetitions"]))
    rng = np.random.default_rng(seed)
    min_samples = int(dbscan_settings["min_samples"])
    metric = str(dbscan_settings["metric"])
    affinity = build_affinity(values, int(settings["neighbors"]), str(settings["affinity"]))
    embedding, _ = laplacian_embedding(affinity, int(dbscan_settings["embedding_dimensions"]))
    eps_candidates = _dbscan_eps_candidates(embedding, min_samples, dbscan_settings["eps_candidate_percentiles"])
    candidates: list[dict[str, Any]] = []
    labels_by_eps: dict[float, np.ndarray] = {}
    for eps in eps_candidates:
        progress_note("clustering", f"evaluating dbscan eps={eps:.6f} min_samples={min_samples} trials={len(frame)}")
        labels = _cluster_once_dbscan(embedding, eps, min_samples, metric)
        non_noise = labels != -1
        cluster_ids = sorted(set(labels[non_noise].tolist()))
        n_clusters = len(cluster_ids)
        noise_fraction = float(np.mean(~non_noise))
        counts = np.array([int(np.sum(labels == cluster_id)) for cluster_id in cluster_ids])
        valid = bool(
            n_clusters >= 2
            and len(counts) > 0
            and np.all(counts >= int(dbscan_settings["min_cluster_trials"]))
            and np.all(counts / len(labels) >= float(dbscan_settings["min_cluster_fraction"]))
            and noise_fraction <= float(dbscan_settings["max_noise_fraction"])
        )
        if n_clusters > 1 and int(non_noise.sum()) > n_clusters:
            try:
                silhouette = float(silhouette_score(embedding[non_noise], labels[non_noise]))
            except ValueError:
                silhouette = -1.0
        else:
            silhouette = -1.0
        aris = []
        if len(unique_participants) > 1:
            bootstrap_tracker = ProgressTracker(
                f"clustering bootstrap dbscan eps={eps:.6f}", repetitions, report_every=max(1, repetitions // 10)
            )
            for repetition in range(repetitions):
                sampled = rng.choice(unique_participants, len(unique_participants), replace=True)
                selected = np.flatnonzero(np.isin(participants, np.unique(sampled)))
                if len(selected) > min_samples and len(selected) != len(frame):
                    boot_labels = _cluster_once_dbscan(embedding[selected], eps, min_samples, metric)
                    aris.append(adjusted_rand_score(labels[selected], boot_labels))
                bootstrap_tracker.update(repetition + 1, f"successful_resamples={len(aris)}")
        stability = float(np.mean(aris)) if aris else 0.0
        labels_by_eps[eps] = labels
        candidates.append({
            "eps": eps, "n_clusters": n_clusters, "noise_fraction": noise_fraction,
            "noise_quality": 1.0 - noise_fraction, "silhouette": silhouette,
            "participant_block_ari": stability, "participant_block_ari_sd": float(np.std(aris)) if aris else np.nan,
            "bootstrap_successes": len(aris), "valid": valid,
            "cluster_sizes": "|".join(map(str, sorted(counts.tolist()))) if len(counts) else "",
        })
        progress_note(
            "clustering",
            f"completed dbscan eps={eps:.6f} valid={valid} n_clusters={n_clusters} "
            f"noise_fraction={noise_fraction:.3f} silhouette={silhouette:.4f}",
        )
    metrics = pd.DataFrame(candidates)
    valid_metrics = metrics[metrics["valid"]].copy()
    if valid_metrics.empty:
        raise ValueError("No DBSCAN eps candidate satisfies the predeclared minimum-size/noise constraints")
    selection_metrics = dbscan_settings["selection_metrics"]
    for metric_name in selection_metrics:
        valid_metrics[f"{metric_name}_rank"] = valid_metrics[metric_name].rank(method="min", ascending=False)
    rank_columns = [f"{metric_name}_rank" for metric_name in selection_metrics]
    valid_metrics["mean_rank"] = valid_metrics[rank_columns].mean(axis=1)
    selected_eps = float(valid_metrics.sort_values(["mean_rank", "eps"]).iloc[0]["eps"])
    metrics = metrics.merge(valid_metrics[["eps", *rank_columns, "mean_rank"]], on="eps", how="left")
    metrics["selected"] = np.isclose(metrics["eps"], selected_eps)
    assignments = _assignment_metadata(frame)
    assignments["cluster"] = labels_by_eps[selected_eps]
    assignments["selected_eps"] = selected_eps
    assignments["min_samples"] = min_samples
    assignments["feature_count"] = len(feature_columns)
    assignments["affinity"] = str(settings["affinity"])
    for dimension in range(embedding.shape[1]):
        assignments[f"spectral_coordinate_{dimension + 1}"] = embedding[:, dimension]
    return assignments, metrics


def assignments_for_eps(
    frame: pd.DataFrame,
    config: AnalysisConfig,
    *,
    matrix_fn: Callable[[pd.DataFrame, Sequence[str]], tuple[np.ndarray, list[str]]],
    blocks: Sequence[str],
    eps: float,
) -> pd.DataFrame:
    """Rebuild DBSCAN cluster assignments for one specific candidate eps (e.g. the runner-up)."""
    values, feature_columns = matrix_fn(frame, blocks)
    settings = config.section("clustering")
    dbscan_settings = settings["dbscan"]
    min_samples = int(dbscan_settings["min_samples"])
    metric = str(dbscan_settings["metric"])
    affinity = build_affinity(values, int(settings["neighbors"]), str(settings["affinity"]))
    embedding, _ = laplacian_embedding(affinity, int(dbscan_settings["embedding_dimensions"]))
    labels = _cluster_once_dbscan(embedding, float(eps), min_samples, metric)
    assignments = _assignment_metadata(frame)
    assignments["cluster"] = labels
    assignments["selected_eps"] = float(eps)
    assignments["min_samples"] = min_samples
    assignments["feature_count"] = len(feature_columns)
    assignments["affinity"] = str(settings["affinity"])
    for dimension in range(embedding.shape[1]):
        assignments[f"spectral_coordinate_{dimension + 1}"] = embedding[:, dimension]
    return assignments


def interpret_clusters(assignments: pd.DataFrame, phenotypes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Describe cluster composition and feature signatures without supervised selection."""
    if assignments.empty or phenotypes.empty:
        return {"composition": pd.DataFrame(), "centroids": pd.DataFrame(), "associations": pd.DataFrame()}
    merged = assignments[["trial_id", "cluster"]].merge(phenotypes, on="trial_id", how="inner")
    composition_rows = []
    for metadata in ("participant_id", "terrain", "footwear"):
        if metadata not in merged:
            continue
        table = merged.groupby(["cluster", metadata]).size().rename("count").reset_index()
        table["fraction_within_cluster"] = table["count"] / table.groupby("cluster")["count"].transform("sum")
        table["metadata"] = metadata
        table["level"] = table[metadata].astype(str)
        composition_rows.append(table[["cluster", "metadata", "level", "count", "fraction_within_cluster"]])
    features = [column for column in merged if "__" in column]
    overall = merged[features].mean()
    centroids = merged.groupby("cluster")[features].mean().subtract(overall).stack().rename("contrast_from_overall").reset_index()
    centroids.columns = ["cluster", "feature", "contrast_from_overall"]
    associations = []
    for metadata in ("participant_id", "terrain", "footwear"):
        if metadata in merged:
            associations.append({
                "metadata": metadata,
                "adjusted_rand_index": adjusted_rand_score(merged[metadata].astype(str), merged["cluster"]),
                "normalized_mutual_information": normalized_mutual_info_score(merged[metadata].astype(str), merged["cluster"]),
            })
    return {
        "composition": pd.concat(composition_rows, ignore_index=True) if composition_rows else pd.DataFrame(),
        "centroids": centroids,
        "associations": pd.DataFrame(associations),
    }
