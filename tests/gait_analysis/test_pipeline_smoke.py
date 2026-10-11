from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import gait_analysis.pipeline as pipeline
from gait_analysis.models import NodeStep, SegmentationResult
from gait_analysis.segmentation import LOWER_LIMB_NODES, NODE_ANATOMICAL_SIDE, driving_ankle_side


NODES = {
    "LH": "left_hip_to_knee", "LK": "left_knee_to_ankle", "LA": "left_ankle_to_foot",
    "RH": "right_hip_to_knee", "RK": "right_knee_to_ankle", "RA": "right_ankle_to_foot",
    "LS": "left_shoulder_to_elbow", "LE": "left_elbow_to_wrist", "LW": "left_wrist_to_hand",
    "RS": "right_shoulder_to_elbow", "RE": "right_elbow_to_wrist", "RW": "right_wrist_to_hand",
}


def _write_raw(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {"Timestamp": np.arange(1001)}
    for sensor in ("device10_IMU1", "device10_IMU3", "device8_IMU1", "device8_IMU3"):
        for kind in ("A", "G"):
            for axis in "XYZ":
                columns[f"{sensor}_{kind}{axis}"] = np.zeros(1001)
    pd.DataFrame(columns).to_csv(path, index=False)


def _write_rss(path: Path, amplitude: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(1001) / 100.0
    values = {"frame": np.arange(1001), "time": time, "segment_label": ["legacy"] * 1001, "cycle_index": [99] * 1001}
    for index, segment in enumerate(NODES.values()):
        values[f"{segment}_mode_1"] = amplitude * (1 + index / 20) * np.sin(2 * np.pi * time + index / 10)
    pd.DataFrame(values).to_csv(path, index=False)


def _segmented(record, _config):
    """Fabricate ankle-driven NodeStep cycles for all 12 nodes without running the real
    Hilbert-phase detector (that algorithm is exercised directly in test_segmentation.py);
    this only needs plausible, deterministic boundaries so downstream stage1/stage2/
    cluster/statistics stages have data to operate on."""
    steps: list[NodeStep] = []
    for ankle_side, offset in (("left", 0), ("right", 50)):
        cycles = []
        for index in range(9):
            start = index * 100 + offset
            if start + 100 > 1000:
                continue
            cycles.append((index, start, start + 100, start + 60))
        for node, anatomical_side in NODE_ANATOMICAL_SIDE.items():
            if driving_ankle_side(node) != ankle_side:
                continue
            coordination = "ipsilateral" if node in LOWER_LIMB_NODES else "contralateral"
            for cycle_index, start, end, toe_off in cycles:
                duration = (end - start) / 100.0
                stance = (toe_off - start) / 100.0
                steps.append(NodeStep(
                    recording_id=record.recording_id, trial_id=record.trial_id,
                    step_id=f"{record.recording_id}:{node}:{cycle_index:04d}", node=node,
                    node_side=anatomical_side, driving_ankle_side=ankle_side, coordination=coordination,
                    cycle_id=f"{record.recording_id}:{ankle_side}:{cycle_index:04d}", cycle_index=cycle_index,
                    start_sample=start, end_sample=end, toe_off_sample=toe_off,
                    toe_off_time_fraction=stance / duration, duration_seconds=duration, stance_seconds=stance,
                    swing_seconds=duration - stance, stance_fraction=stance / duration,
                    phase_rotation_per_stride=1.0, valley_prominence=1.0, accepted=True,
                ))
    return SegmentationResult(record.recording_id, record.trial_id, steps, 100.0, "ok")


def test_end_to_end_artifacts_cache_and_determinism(config, tmp_path: Path, monkeypatch) -> None:
    raw_root, rss_root, output_root = tmp_path / "raw", tmp_path / "rss", tmp_path / "out"
    for participant in ("P1", "P2"):
        for condition_index, terrain in enumerate(("Flat Terrain", "Stairs")):
            session = Path(participant) / terrain / "0,0_Walking" / f"session_{terrain}_Trial01"
            raw = raw_root / session / f"combined_log_wide_{participant}_{condition_index}.csv"
            _write_raw(raw)
            _write_rss(rss_root / session / raw.stem / "limb_angle_mode.csv", 1.0 + condition_index + (0.2 if participant == "P2" else 0.0))
    local = config.with_values({
        "paths": {"raw_root": str(raw_root), "rss_root": str(rss_root), "output_root": str(output_root), "reference_registry": str(tmp_path / "references.json")},
        "mfpca": {"bootstrap_repetitions": 1},
        "transfer_entropy": {"delays_samples": [1, 2], "surrogates_development": 1, "near_zero_shift_samples": 2},
        "profiles": {"development": {"te_surrogates": 1, "bootstrap_repetitions": 2}},
        "clustering": {"candidate_clusters": [2], "neighbors": 2, "min_cluster_trials": 1, "min_cluster_fraction": 0.0},
    })
    calls = {"count": 0}
    def counted(record, cfg):
        calls["count"] += 1
        return _segmented(record, cfg)
    monkeypatch.setattr(pipeline, "segment_recording", counted)
    first = pipeline.run_pipeline(local)
    first_calls = calls["count"]
    second = pipeline.run_pipeline(local)
    assert calls["count"] == first_calls
    assert first.run_dir == second.run_dir
    expected = [
        "cohort_manifest.csv", "gait_steps.csv", "classic_step_features.csv",
        "classic_trial_features.csv", "proxy_signal_qc.csv", "node_step_features.csv",
        "node_step_trajectories.csv.gz", "te_edges.csv.gz", "te_network_features.csv",
        "te_trial_features.csv",
        "node_fpca/node_fpca_step_scores.csv", "node_fpca/node_fpca_trial_scores.csv",
        "cluster_assignments_all.csv", "cluster_assignments_lower.csv", "cluster_assignments_upper.csv",
        "cluster_assignments.csv", "cluster_selection.csv",
        "statistical_results.csv", "figure_manifest.csv", "run_metadata.json", "errors.csv",
    ]
    assert all((first.run_dir / name).exists() for name in expected)
    metadata = (first.run_dir / "run_metadata.json").read_text(encoding="utf-8")
    assert "rss_segment_mode1_proxy" in metadata
    assert pd.read_csv(first.run_dir / "gait_steps.csv")["config_hash"].nunique() == 1
    steps = pd.read_csv(first.run_dir / "gait_steps.csv")
    assert set(steps["node"]) == set(NODES)
