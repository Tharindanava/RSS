from __future__ import annotations

from pathlib import Path

import pandas as pd

from gait_analysis.dashboard import clear_dashboard_cache, load_hybrid_segmentation


def _gait_steps_row(recording_id: str, node: str, node_side: str, start: int, end: int, toe_off: int, *, accepted: bool = True, reasons: str = "") -> dict:
    return {
        "recording_id": recording_id, "step_id": f"{recording_id}:{node}:0000", "node": node,
        "node_side": node_side, "driving_ankle_side": node_side, "coordination": "ipsilateral",
        "cycle_id": f"{recording_id}:{node_side}:0000", "cycle_index": 0,
        "start_sample": start, "end_sample": end, "toe_off_sample": toe_off,
        "toe_off_time_fraction": (toe_off - start) / max(end - start, 1),
        "duration_seconds": (end - start) / 100.0, "stance_seconds": (toe_off - start) / 100.0,
        "swing_seconds": (end - toe_off) / 100.0, "stance_fraction": (toe_off - start) / max(end - start, 1),
        "accepted": accepted, "reasons": reasons,
    }


def test_dashboard_uses_gait_steps_sidecar_and_never_a_legacy_event_file(tmp_path: Path) -> None:
    mode = tmp_path / "rss" / "combined_log_wide_1" / "limb_angle_mode.csv"
    mode.parent.mkdir(parents=True)
    pd.DataFrame({
        "left_hip_to_knee_mode_1": range(20),
        "left_hip_to_knee_mode_2": range(20),
        "left_hip_to_knee_mode_3": range(20),
        "segment_label": ["legacy"] * 20,
        "cycle_index": [99] * 20,
    }).to_csv(mode, index=False)
    run = tmp_path / "analysis" / "runs" / "run1"
    run.mkdir(parents=True)
    pd.DataFrame({"recording_id": ["rec1"], "rss_file": [str(mode)], "recording_identity": ["combined_log_wide_1"]}).to_csv(run / "manifest.csv", index=False)
    pd.DataFrame([
        _gait_steps_row("rec1", "LA", "left", 2, 15, 8),
        _gait_steps_row("rec1", "RA", "right", 5, 18, 11),
        _gait_steps_row("rec1", "LH", "left", 2, 15, 8),  # broadcast lower-limb node; not surfaced as a leg step
    ]).to_csv(run / "gait_steps.csv", index=False)
    # A stray legacy events file must never be read: only gait_steps.csv drives the adapter now.
    pd.DataFrame({"recording_id": ["rec1"], "side": ["left"], "event_type": ["IC"], "sample": [99]}).to_csv(run / "gait_events.csv", index=False)
    clear_dashboard_cache()
    result = load_hybrid_segmentation(mode, tmp_path / "analysis", 20)
    assert result["available"]
    assert result["step_segmentation_available"]
    assert result["segment_label"][2] == "L-IC"
    assert result["segment_label"][8] == "L-TO"
    assert result["step_index_by_side"]["left"][3] == 0
    assert 99 not in result["step_index_by_side"]["left"]
    assert {step["side"] for step in result["gait_steps"]} == {"left", "right"}
    assert all(event["primary_sensor"] == "hilbert_ankle_minima" for event in result["gait_events"])
    from HMA_Dashboard import signals

    signals.configure_analysis_results_dir(tmp_path / "analysis")
    displayed = signals.load_signal_trace(mode, "left_hip_to_knee", "mode_1", default_fs=100.0)
    assert displayed["hybrid_segmentation_available"]
    assert displayed["segment_label"][2] == "L-IC"
    assert 99 not in displayed["step_index"]


def test_dashboard_reports_unavailable_without_hybrid_result(tmp_path: Path) -> None:
    mode = tmp_path / "limb_angle_mode.csv"
    mode.write_text("left_hip_to_knee_mode_1\n0\n", encoding="utf-8")
    clear_dashboard_cache()
    result = load_hybrid_segmentation(mode, tmp_path / "empty", 1)
    assert not result["available"]
    assert result["message"] == "gait analysis artifacts unavailable"


def test_legacy_only_run_directory_without_gait_steps_is_unavailable(tmp_path: Path) -> None:
    mode = tmp_path / "rss" / "recording" / "limb_angle_mode.csv"
    mode.parent.mkdir(parents=True)
    pd.DataFrame({"left_hip_to_knee_mode_1": range(20)}).to_csv(mode, index=False)
    run = tmp_path / "analysis" / "runs" / "run1"
    run.mkdir(parents=True)
    pd.DataFrame({"recording_id": ["rec1"], "rss_file": [str(mode)]}).to_csv(run / "manifest.csv", index=False)
    # Only a legacy/irrelevant artifact is present (no gait_steps.csv); must not be
    # mistaken for segmentation data.
    pd.DataFrame({"recording_id": ["rec1"], "cycle_id": ["legacy"]}).to_csv(run / "gait_cycles.csv", index=False)
    clear_dashboard_cache()
    result = load_hybrid_segmentation(mode, tmp_path / "analysis", 20)
    assert not result["available"]


def test_rejected_ankle_step_is_excluded_from_overlays(tmp_path: Path) -> None:
    mode = tmp_path / "rss" / "recording2" / "limb_angle_mode.csv"
    mode.parent.mkdir(parents=True)
    pd.DataFrame({"left_hip_to_knee_mode_1": range(20)}).to_csv(mode, index=False)
    run = tmp_path / "analysis" / "runs" / "run1"
    run.mkdir(parents=True)
    pd.DataFrame({"recording_id": ["rec2"], "rss_file": [str(mode)]}).to_csv(run / "manifest.csv", index=False)
    pd.DataFrame([
        _gait_steps_row("rec2", "LA", "left", 2, 15, 8),
        _gait_steps_row("rec2", "LA", "left", 15, 30, 22, accepted=False, reasons="stride_duration_out_of_range"),
    ]).to_csv(run / "gait_steps.csv", index=False)
    clear_dashboard_cache()
    result = load_hybrid_segmentation(mode, tmp_path / "analysis", 30)
    assert result["available"]
    assert result["status"] == "ok"
    assert "1 accepted, 1 rejected" in result["message"]
    assert len(result["gait_steps"]) == 1
