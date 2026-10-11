from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gait_analysis.classical_features import (
    aggregate_trial_features, bounded_symmetry, compute_classic_features, compute_node_step_features,
    normalize_node_trajectories, normalize_step_trajectories,
)
from gait_analysis.models import NodeStep, SegmentationResult, TrialRecord


def _ankle_step(node: str, node_side: str, start: int, end: int, toe_off: int, *, fs: float = 100.0, cycle_index: int = 0, accepted: bool = True) -> NodeStep:
    duration = (end - start) / fs
    stance = (toe_off - start) / fs
    swing = (end - toe_off) / fs
    return NodeStep(
        recording_id="r", trial_id="t", step_id=f"r:{node}:{cycle_index:04d}", node=node, node_side=node_side,
        driving_ankle_side=node_side, coordination="ipsilateral", cycle_id=f"r:{node_side}:{cycle_index:04d}",
        cycle_index=cycle_index, start_sample=start, end_sample=end, toe_off_sample=toe_off,
        toe_off_time_fraction=stance / duration, duration_seconds=duration, stance_seconds=stance,
        swing_seconds=swing, stance_fraction=stance / duration, accepted=accepted,
    )


def test_hand_computed_timing_rom_and_symmetry(config) -> None:
    record = TrialRecord("r", "t", "p", Path("raw"), Path("rss"))
    steps = [
        _ankle_step("LA", "left", 0, 100, 60),
        _ankle_step("RA", "right", 50, 150, 110),
    ]
    result = SegmentationResult("r", "t", steps, 100.0, "ok")
    proxies = {node: np.linspace(0.0, float(index + 1), 151) for index, node in enumerate(("LH", "LK", "LA", "RH", "RK", "RA"))}
    frame = compute_classic_features(record, result, proxies, config)
    assert np.isclose(frame.iloc[0]["stride_duration_seconds"], 1.0)
    assert np.isclose(frame.iloc[0]["step_duration_seconds"], 0.5)
    assert np.isclose(frame.iloc[0]["stance_fraction"], 0.6)
    assert np.isclose(frame.iloc[0]["cadence_per_min"], 120.0)
    assert np.isclose(frame.iloc[0]["LH_proxy_rom"], 100 / 150)
    assert set(frame["side"]) == {"left", "right"}
    assert (frame["boundary_source"] == "hilbert_ankle_minima").all()
    assert "step_confidence" not in frame.columns
    assert np.isclose(bounded_symmetry(1.0, 2.0), 200.0 / 3.0)


def test_compute_classic_features_ignores_non_ankle_nodes(config) -> None:
    record = TrialRecord("r", "t", "p", Path("raw"), Path("rss"))
    steps = [
        _ankle_step("LA", "left", 0, 100, 60),
        _ankle_step("RA", "right", 50, 150, 110),
        _ankle_step("LS", "left", 25, 125, 85, cycle_index=1),  # contralateral upper-limb node; should be excluded
    ]
    result = SegmentationResult("r", "t", steps, 100.0, "ok")
    proxies = {node: np.linspace(0.0, float(index + 1), 151) for index, node in enumerate(("LH", "LK", "LA", "RH", "RK", "RA"))}
    frame = compute_classic_features(record, result, proxies, config)
    assert len(frame) == 2
    assert set(frame["side"]) == {"left", "right"}


def test_compute_node_step_features_covers_all_twelve_nodes(config) -> None:
    record = TrialRecord("r", "t", "p", Path("raw"), Path("rss"))
    nodes = ("LH", "LK", "LA", "RH", "RK", "RA", "LS", "LE", "LW", "RS", "RE", "RW")
    steps = [_ankle_step(node, "left" if index % 2 == 0 else "right", index * 10, index * 10 + 100, index * 10 + 60, cycle_index=index) for index, node in enumerate(nodes)]
    result = SegmentationResult("r", "t", steps, 100.0, "ok")
    proxies = {node: np.linspace(0.0, 10.0, 400) for node in nodes}
    frame = compute_node_step_features(record, result, proxies, config)
    assert set(frame["node"]) == set(nodes)
    assert len(frame) == len(nodes)
    assert (frame["boundary_source"] == "hilbert_ankle_minima").all()
    assert {"proxy_rom", "proxy_min", "proxy_max", "duration_seconds", "stance_fraction"}.issubset(frame.columns)


def test_compute_node_step_features_skips_unaccepted_cycles(config) -> None:
    record = TrialRecord("r", "t", "p", Path("raw"), Path("rss"))
    steps = [_ankle_step("LA", "left", 0, 100, 60), _ankle_step("LA", "left", 100, 200, 160, cycle_index=1, accepted=False)]
    result = SegmentationResult("r", "t", steps, 100.0, "ok")
    proxies = {"LA": np.linspace(0.0, 1.0, 300)}
    frame = compute_node_step_features(record, result, proxies, config)
    assert len(frame) == 1


def test_normalize_node_trajectories_matches_step_trajectory_schema_with_self_referential_pair_id(config) -> None:
    record = TrialRecord("r", "t", "p", Path("raw"), Path("rss"))
    steps = [
        _ankle_step("LA", "left", 0, 100, 60),
        _ankle_step("RA", "right", 50, 150, 110),
        _ankle_step("LS", "left", 25, 125, 85, cycle_index=1),
    ]
    result = SegmentationResult("r", "t", steps, 100.0, "ok")
    proxies = {node: np.linspace(0.0, 1.0, 200) for node in ("LH", "LK", "LA", "RH", "RK", "RA", "LS")}
    leg_trajectories = normalize_step_trajectories(record, result, proxies, config)
    node_trajectories = normalize_node_trajectories(record, result, proxies, config)
    assert set(leg_trajectories.columns) == set(node_trajectories.columns)
    assert set(node_trajectories["node"]) == {"LA", "RA", "LS"}
    # Every NodeStep's own step_id is used as its (self-referential) step_pair_id.
    ls_rows = node_trajectories[node_trajectories["node"] == "LS"]
    assert (ls_rows["step_pair_id"] == ls_rows["step_id"]).all()
    assert (ls_rows["step_id"] == "r:LS:0001").all()


def test_trial_aggregation_is_robust_across_phase_recordings(config) -> None:
    frame = pd.DataFrame({
        "recording_id": ["r1", "r2", "r2"], "trial_id": ["t"] * 3,
        "participant_id": ["p"] * 3, "terrain": ["flat"] * 3, "direction": ["level"] * 3,
        "footwear": ["0,0"] * 3, "stride_duration_seconds": [1.0, 1.1, 99.0],
    })
    trial = aggregate_trial_features(frame, config)
    assert len(trial) == 1
    assert trial.iloc[0]["n_recordings"] == 2
    assert trial.iloc[0]["stride_duration_seconds"] == 1.1
