from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score

from gait_analysis.spectral_clustering import _cluster_once, build_affinity, cluster_phenotypes, laplacian_embedding


def test_nonconvex_graph_clusters(config) -> None:
    values, truth = make_moons(n_samples=120, noise=0.04, random_state=2)
    settings = config.section("clustering") | {"neighbors": 8}
    labels, _, _ = _cluster_once(values, settings, 2, 2)
    assert adjusted_rand_score(truth, labels) > 0.9


def test_duplicate_and_isolated_points_are_finite(config) -> None:
    values = np.array([[0, 0], [0, 0], [1, 1], [20, 20]], dtype=float)
    affinity = build_affinity(values, neighbors=2)
    embedding, eigenvalues = laplacian_embedding(affinity, 2)
    assert np.all(np.isfinite(affinity))
    assert np.all(np.isfinite(embedding))
    assert np.all(np.isfinite(eigenvalues))


def test_cluster_phenotypes_ablation_invokes_matrix_fn_once_per_block(config) -> None:
    """``cluster_phenotypes`` is decoupled from any specific feature-matrix builder:
    it must call the injected ``matrix_fn`` once up front with the full ``blocks``
    set, then exactly once more per omitted block in the leave-one-block-out
    ablation loop (generalizing the old hardcoded leave-one-family-out logic to an
    arbitrary block type, e.g. eigenfunction nodes instead of phenotype families)."""
    n_per_cluster = 6
    group = np.array([0] * n_per_cluster + [1] * n_per_cluster, dtype=float)
    frame = pd.DataFrame({
        "trial_id": [f"t{i}" for i in range(2 * n_per_cluster)],
        "participant_id": [f"p{i % 4}" for i in range(2 * n_per_cluster)],
        "group": group,
    })
    blocks = ["A", "B", "C"]
    calls: list[list[str]] = []

    def matrix_fn(sub_frame: pd.DataFrame, included_blocks) -> tuple[np.ndarray, list[str]]:
        included_blocks = list(included_blocks)
        calls.append(included_blocks)
        base = sub_frame["group"].to_numpy(float) * 5.0
        rng_local = np.random.default_rng(abs(hash(tuple(included_blocks))) % (2**32))
        columns = [base + rng_local.normal(scale=0.01, size=len(sub_frame)) for _ in included_blocks]
        if not columns:
            raise ValueError("no blocks remain")
        return np.column_stack(columns), [f"{block}__feat" for block in included_blocks]

    local_config = config.with_values({
        "clustering": {
            "candidate_clusters": [2], "neighbors": 3, "min_cluster_trials": 1,
            "min_cluster_fraction": 0.0, "bootstrap_repetitions": 2,
        },
        "profiles": {"development": {"bootstrap_repetitions": 2}},
    })

    assignments, metrics = cluster_phenotypes(frame, local_config, matrix_fn=matrix_fn, blocks=blocks)

    assert calls[0] == blocks
    ablation_calls = calls[1:]
    assert len(ablation_calls) == len(blocks)
    for omitted, call in zip(blocks, ablation_calls):
        assert omitted not in call
        assert set(call) == set(blocks) - {omitted}
    assert not assignments.empty
    assert set(metrics["k"]) == {2}

