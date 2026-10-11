from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gait_analysis.node_fpca import aggregate_node_trial_scores, fit_all_node_fpca, node_score_matrix
from gait_analysis.utils import robust_zscore


def _node_rows(node: str, n_windows: int, participants: tuple[str, ...], trials: tuple[str, ...]) -> list[dict]:
    percent_grid = np.linspace(0.0, 100.0, 11)
    rows: list[dict] = []
    for index in range(n_windows):
        participant = participants[index % len(participants)]
        trial = trials[index % len(trials)]
        amplitude = 1.0 + 0.3 * index
        secondary = 0.1 * index
        values = amplitude * np.sin(2 * np.pi * percent_grid / 100.0) + secondary * np.cos(4 * np.pi * percent_grid / 100.0)
        for percent, value in zip(percent_grid, values):
            rows.append({
                "participant_id": participant, "trial_id": trial, "recording_id": f"{node}_rec_{index}",
                "step_pair_id": f"{node}_{index:03d}", "representation": "event_registered",
                "node": node, "percent_step": percent, "value": float(value),
            })
    return rows


@pytest.fixture
def multi_node_config(config):
    return config.with_values({
        "node_fpca": {
            "nodes": ["LA", "LH", "RA"],
            "n_eigenfunctions": 2,
            "node_weights": {"LA": 1.0, "LH": 1.0, "RA": 1.0},
        },
        "clustering": {"analyses": {"all": ["LA", "LH", "RA"], "lower": ["LA", "LH", "RA"], "upper": []}},
    })


@pytest.fixture
def multi_node_trajectories() -> pd.DataFrame:
    participants = ("P1", "P2")
    trials = ("T1", "T2", "T3")
    rows = []
    rows += _node_rows("LA", 12, participants, trials)
    rows += _node_rows("LH", 9, participants, ("T1", "T2"))
    rows += _node_rows("RA", 1, participants, trials)  # too-short: must be skipped, not crash
    return pd.DataFrame(rows)


def test_fit_all_node_fpca_independent_nodes_and_graceful_skip(multi_node_config, multi_node_trajectories) -> None:
    results = fit_all_node_fpca(multi_node_trajectories, multi_node_config)

    # The too-short node (single window) is skipped, not a crash.
    assert set(results) == {"LA", "LH"}

    la_result, la_scores = results["LA"]
    lh_result, lh_scores = results["LH"]

    # Each node fit independently from its own windows, regardless of shared table.
    assert len(la_scores) == 12
    assert len(lh_scores) == 9
    assert set(la_scores["step_pair_id"]).isdisjoint(set(lh_scores["step_pair_id"]))

    for node, scores in (("LA", la_scores), ("LH", lh_scores)):
        expected_columns = {f"node_fpca__{node}__score_{index + 1}" for index in range(2)}
        assert expected_columns.issubset(scores.columns)

    assert la_result.n_components >= 1
    assert lh_result.n_components >= 1


def test_fit_all_node_fpca_zero_pads_low_rank_node(config) -> None:
    """A node whose true rank is 1 must still emit K=3 columns, zero-padded past its real rank."""
    local = config.with_values({
        "node_fpca": {"nodes": ["LA"], "n_eigenfunctions": 3, "node_weights": {"LA": 1.0}},
        "clustering": {"analyses": {"all": ["LA"], "lower": ["LA"], "upper": []}},
    })
    percent_grid = np.linspace(0.0, 100.0, 11)
    rows = []
    # Perfectly rank-1 data: every window is just a scalar multiple of the same shape.
    for index in range(6):
        amplitude = 1.0 + index
        values = amplitude * np.sin(2 * np.pi * percent_grid / 100.0)
        for percent, value in zip(percent_grid, values):
            rows.append({
                "participant_id": f"P{index % 2}", "trial_id": f"T{index % 3}", "recording_id": f"rec_{index}",
                "step_pair_id": f"LA_{index:03d}", "representation": "event_registered",
                "node": "LA", "percent_step": percent, "value": float(value),
            })
    trajectories = pd.DataFrame(rows)

    results = fit_all_node_fpca(trajectories, local)

    assert set(results) == {"LA"}
    result, scores = results["LA"]
    assert result.n_components == 1
    assert np.allclose(scores["node_fpca__LA__score_2"], 0.0)
    assert np.allclose(scores["node_fpca__LA__score_3"], 0.0)
    assert not np.allclose(scores["node_fpca__LA__score_1"], 0.0)


def test_aggregate_node_trial_scores_median_and_skip_handling(config) -> None:
    k = 2
    node_a_scores = pd.DataFrame({
        "participant_id": ["P1", "P1", "P1", "P2"],
        "trial_id": ["T1", "T1", "T1", "T1"],
        "node_fpca__A__score_1": [1.0, 3.0, 5.0, 10.0],
        "node_fpca__A__score_2": [0.0, 2.0, 4.0, -1.0],
    })
    node_b_scores = pd.DataFrame({
        "participant_id": ["P1", "P1", "P2"],
        "trial_id": ["T1", "T1", "T1"],
        "node_fpca__B__score_1": [7.0, 9.0, -2.0],
        "node_fpca__B__score_2": [1.0, 1.0, 1.0],
    })
    from gait_analysis.models import MFPCAResult

    def fake_result() -> MFPCAResult:
        return MFPCAResult(np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), 1)

    node_results = {"A": (fake_result(), node_a_scores), "B": (fake_result(), node_b_scores)}

    trial_scores = aggregate_node_trial_scores(node_results, k)

    assert set(trial_scores.columns) == {
        "participant_id", "trial_id", "node_fpca__A__score_1", "node_fpca__A__score_2",
        "node_fpca__B__score_1", "node_fpca__B__score_2",
    }
    # Inner merge on (participant_id, trial_id): only (P1, T1) and (P2, T1) survive.
    assert len(trial_scores) == 2
    p1_row = trial_scores[trial_scores["participant_id"] == "P1"].iloc[0]
    assert p1_row["node_fpca__A__score_1"] == pytest.approx(3.0)  # median(1, 3, 5)
    assert p1_row["node_fpca__B__score_1"] == pytest.approx(8.0)  # median(7, 9)

    # A node that was never fit (skipped upstream) simply contributes no columns.
    skipped_node_results = {"A": (fake_result(), node_a_scores)}
    only_a = aggregate_node_trial_scores(skipped_node_results, k)
    assert "node_fpca__B__score_1" not in only_a.columns
    assert set(only_a.columns) == {"participant_id", "trial_id", "node_fpca__A__score_1", "node_fpca__A__score_2"}


def test_node_score_matrix_shape_and_weighting() -> None:
    k = 2
    nodes = ["A", "B"]
    trial_scores = pd.DataFrame({
        "node_fpca__A__score_1": [1.0, 2.0, 3.0, 100.0, 5.0],
        "node_fpca__A__score_2": [5.0, 4.0, 3.0, 2.0, 1.0],
        "node_fpca__B__score_1": [10.0, 20.0, 30.0, 40.0, 50.0],
        "node_fpca__B__score_2": [-1.0, -2.0, -3.0, -4.0, -5.0],
    })
    node_weights = {"A": 1.0, "B": 2.0}

    values, columns = node_score_matrix(trial_scores, nodes, k, node_weights)

    assert values.shape == (5, len(nodes) * k)
    assert columns == [
        "node_fpca__A__score_1", "node_fpca__A__score_2",
        "node_fpca__B__score_1", "node_fpca__B__score_2",
    ]

    expected_a, _, _ = robust_zscore(trial_scores[["node_fpca__A__score_1", "node_fpca__A__score_2"]])
    expected_a = expected_a.fillna(0.0).to_numpy(float) * np.sqrt(1.0 / len(nodes))
    expected_b, _, _ = robust_zscore(trial_scores[["node_fpca__B__score_1", "node_fpca__B__score_2"]])
    expected_b = expected_b.fillna(0.0).to_numpy(float) * np.sqrt(2.0 / len(nodes))

    np.testing.assert_allclose(values[:, :2], expected_a)
    np.testing.assert_allclose(values[:, 2:], expected_b)


def test_node_score_matrix_skips_nodes_without_columns() -> None:
    trial_scores = pd.DataFrame({
        "node_fpca__A__score_1": [1.0, 2.0, 3.0],
        "node_fpca__A__score_2": [3.0, 2.0, 1.0],
    })
    values, columns = node_score_matrix(trial_scores, ["A", "MISSING"], 2, {"A": 1.0, "MISSING": 1.0})
    assert columns == ["node_fpca__A__score_1", "node_fpca__A__score_2"]
    assert values.shape == (3, 2)


def test_node_score_matrix_raises_when_no_nodes_available() -> None:
    trial_scores = pd.DataFrame({"trial_id": ["T1", "T2"]})
    with pytest.raises(ValueError):
        node_score_matrix(trial_scores, ["MISSING"], 2, {})
