from __future__ import annotations

import numpy as np

from gait_analysis.te_estimator import maximize_te_over_delays, quantile_discretize, valid_circular_shifts
from gait_analysis.te_network import estimate_te_network, form_te_windows
from gait_analysis.models import NodeStep


def test_delayed_directionality_and_lag_recovery() -> None:
    rng = np.random.default_rng(9)
    n, delay = 5000, 7
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.8 * x[t - 1] + rng.normal()
        y[t] = 0.5 * y[t - 1] + (0.8 * x[t - delay] if t >= delay else 0.0) + rng.normal(scale=0.7)
    xd, yd = quantile_discretize(x, 8), quantile_discretize(y, 8)
    forward, recovered, _ = maximize_te_over_delays(xd, yd, range(1, 16))
    reverse, _, _ = maximize_te_over_delays(yd, xd, range(1, 16))
    assert forward > reverse
    assert abs(recovered - delay) <= 1


def _node_step(node: str, node_side: str, driving_ankle_side: str, coordination: str, start: int, end: int, *, cycle_index: int, fs: float = 100.0, toe_off_fraction: float = 0.6) -> NodeStep:
    duration = (end - start) / fs
    toe_off = start + int(round((end - start) * toe_off_fraction))
    stance = (toe_off - start) / fs
    return NodeStep(
        recording_id="r", trial_id="t", step_id=f"{node}-{cycle_index}", node=node, node_side=node_side,
        driving_ankle_side=driving_ankle_side, coordination=coordination, cycle_id=f"{driving_ankle_side}:{cycle_index}",
        cycle_index=cycle_index, start_sample=start, end_sample=end, toe_off_sample=toe_off,
        toe_off_time_fraction=toe_off_fraction, duration_seconds=duration, stance_seconds=stance,
        swing_seconds=duration - stance, stance_fraction=toe_off_fraction, accepted=True,
    )


def _ankle_pair_steps(n_pairs: int = 17, *, step_samples: int = 100) -> list[NodeStep]:
    steps = []
    for i in range(n_pairs):
        steps.append(_node_step("LA", "left", "left", "ipsilateral", i * step_samples, (i + 1) * step_samples, cycle_index=i))
        steps.append(_node_step("RA", "right", "right", "ipsilateral", i * step_samples, (i + 1) * step_samples, cycle_index=i))
    return steps


def test_shift_exclusion_and_complete_eight_step_pair_windows() -> None:
    shifts = valid_circular_shifts(800, 30, 100.0, 0.1)
    assert np.all(np.minimum(shifts, 800 - shifts) > 30)
    assert not np.any(np.isclose(np.mod(shifts, 100), 0, atol=10))
    steps = _ankle_pair_steps(17)
    windows = form_te_windows(steps, 8)
    assert len(windows) == 2
    assert (windows[0].start_sample, windows[0].end_sample) == (0, 800)
    assert all(step.node in ("LA", "RA") for step in windows[0].reference_steps)


def test_form_te_windows_gives_every_node_local_boundaries_clipped_to_the_window() -> None:
    # LS (contralateral, driven by the right ankle) is offset half a stride from LA/RA.
    steps = _ankle_pair_steps(8, step_samples=100)
    offset = 50
    ls_steps = [_node_step("LS", "left", "right", "contralateral", i * 100 + offset, (i + 1) * 100 + offset, cycle_index=i) for i in range(9)]
    windows = form_te_windows(steps + ls_steps, 8)
    assert len(windows) == 1
    window = windows[0]
    assert set(window.local_boundaries) == {"LA", "RA", "LS"}
    assert window.local_boundaries["LA"][0] == (0, 100)
    # LS's boundaries are clipped to the window and offset from LA's by half a stride.
    assert window.local_boundaries["LS"][0][0] != window.local_boundaries["LA"][0][0]
    for node in ("LA", "RA", "LS"):
        for lo, hi in window.local_boundaries[node]:
            assert 0 <= lo < hi <= window.end_sample - window.start_sample


def _reference_steps(n: int = 8) -> list[NodeStep]:
    return [_node_step("LA", "left", "left", "ipsilateral", i * 100, (i + 1) * 100, cycle_index=i) for i in range(n)]


def _local_boundaries(nodes: list[str], n: int = 8) -> dict[str, list[tuple[int, int]]]:
    return {node: [(i * 100, (i + 1) * 100) for i in range(n)] for node in nodes}


def test_network_is_deterministic_fdr_corrected_and_uses_surrogate_max(config) -> None:
    rng = np.random.default_rng(22)
    nodes = config.section("transfer_entropy")["nodes"]
    signals = {node: rng.normal(size=801) for node in nodes}
    local = config.with_values({
        "transfer_entropy": {"delays_samples": [1, 3], "near_zero_shift_samples": 5},
        "profiles": {"development": {"te_surrogates": 3, "bootstrap_repetitions": 3}},
    })
    reference_steps = _reference_steps()
    boundaries = _local_boundaries(nodes)
    first = estimate_te_network(signals, 0, 800, reference_steps, boundaries, local, recording_id="r", trial_id="t", window_id="w")
    second = estimate_te_network(signals, 0, 800, reference_steps, boundaries, local, recording_id="r", trial_id="t", window_id="w")
    assert len(first.edge_table) == 132
    assert first.edge_table["q_value"].between(0, 1).all()
    assert (first.edge_table["null_mean_max_te"] >= 0).all()
    np.testing.assert_allclose(first.edge_table["effective_te"], second.edge_table["effective_te"])
    np.testing.assert_allclose(first.adjacency, second.adjacency)


def test_short_whole_body_window_is_rejected_without_padding(config) -> None:
    nodes = config.section("transfer_entropy")["nodes"]
    signals = {node: np.arange(20, dtype=float) for node in nodes}
    result = estimate_te_network(signals, 0, 19, _reference_steps(), _local_boundaries(nodes), config, recording_id="r", trial_id="t", window_id="short")
    assert result.status == "excluded"
    assert "short_window" in result.reasons


def test_alternative_surrogates_and_phase_conditioning_are_implemented(config) -> None:
    rng = np.random.default_rng(5)
    nodes = config.section("transfer_entropy")["nodes"]
    signals = {node: rng.normal(size=801) for node in nodes}
    boundaries = _local_boundaries(nodes)
    for method in ("cycle_block", "phase_randomized"):
        local = config.with_values({
            "transfer_entropy": {
                "delays_samples": [1, 2], "surrogate_method": method, "phase_conditioned": True,
                "near_zero_shift_samples": 5,
            },
            "profiles": {"development": {"te_surrogates": 2, "bootstrap_repetitions": 2}},
        })
        result = estimate_te_network(signals, 0, 800, _reference_steps(), boundaries, local, recording_id="r", trial_id="t", window_id=method)
        assert result.status == "ok"
        assert set(result.edge_table["surrogate_method"]) == {method}
        assert result.edge_table["phase_conditioned"].all()


def test_phase_conditioning_uses_per_node_local_boundaries_not_a_shared_list(config) -> None:
    # Two nodes with different local cycle boundaries in the same window should produce
    # a different target-conditioned phase axis, and therefore (generically) different
    # TE estimates into a downstream target, even though the raw signals are identical.
    rng = np.random.default_rng(3)
    nodes = config.section("transfer_entropy")["nodes"]
    signals = {node: rng.normal(size=801) for node in nodes}
    boundaries = dict(_local_boundaries(nodes))
    # Give one node (LS) boundaries offset by half a stride from the rest.
    boundaries["LS"] = [(i * 100 + 50, (i + 1) * 100 + 50) for i in range(7)]
    local = config.with_values({
        "transfer_entropy": {"delays_samples": [1, 2], "phase_conditioned": True, "near_zero_shift_samples": 5},
        "profiles": {"development": {"te_surrogates": 2, "bootstrap_repetitions": 2}},
    })
    result = estimate_te_network(signals, 0, 800, _reference_steps(), boundaries, local, recording_id="r", trial_id="t", window_id="phase")
    assert result.status == "ok"
    row_lh_to_ls = result.edge_table[(result.edge_table["source"] == "LH") & (result.edge_table["target"] == "LS")]
    row_lh_to_la = result.edge_table[(result.edge_table["source"] == "LH") & (result.edge_table["target"] == "LA")]
    assert not row_lh_to_ls.empty and not row_lh_to_la.empty
    assert row_lh_to_ls.iloc[0]["raw_te"] != row_lh_to_la.iloc[0]["raw_te"]
