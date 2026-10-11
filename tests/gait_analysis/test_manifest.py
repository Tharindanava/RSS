from __future__ import annotations

from pathlib import Path

import pandas as pd

from gait_analysis.metadata import build_manifest, canonical_header, resolve_sensor_columns


def _raw(path: Path, convention: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["Timestamp"]
    for sensor in ("device10_IMU1", "device10_IMU3", "device8_IMU1", "device8_IMU3"):
        for kind in ("A", "G"):
            for axis in "XYZ":
                if convention == 0:
                    columns.append(f"{sensor}_{kind}{axis}")
                elif convention == 1:
                    columns.append(f"{sensor}_{'Accelerometer' if kind == 'A' else 'Gyroscope'}_{axis.lower()}")
                else:
                    columns.append(f"{sensor.replace('device', 'device_').replace('IMU', 'IMU_')}_{kind}_{axis}")
    pd.DataFrame([[0.0] * len(columns), [1.0] * len(columns)], columns=columns).to_csv(path, index=False)


def test_header_conventions_resolve() -> None:
    for convention in range(3):
        names = []
        for kind in ("A", "G"):
            for axis in "XYZ":
                names.append(
                    f"device10_IMU1_{kind}{axis}" if convention == 0
                    else f"device10_IMU1_{'Accelerometer' if kind == 'A' else 'Gyroscope'}_{axis.lower()}"
                    if convention == 1 else f"device_10_IMU_1_{kind}_{axis}"
                )
        assert len(resolve_sensor_columns(names, "device10_IMU1")) == 6


def test_phase_files_share_trial_but_not_recording(config, tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    rss_root = tmp_path / "rss"
    session = Path("P01/Flat Terrain/0,0_Walking/session_Trial01")
    for phase in ("01", "02"):
        raw = raw_root / session / f"combined_log_wide_100_Phase_{phase}.csv"
        _raw(raw, int(phase) - 1)
        rss = rss_root / session / raw.stem / "limb_angle_mode.csv"
        rss.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"left_hip_to_knee_mode_1": [0.0, 1.0]}).to_csv(rss, index=False)
    local = config.with_values({"paths": {"raw_root": str(raw_root), "rss_root": str(rss_root), "output_root": str(tmp_path / "out")}})
    manifest = build_manifest(local)
    assert len(manifest) == 2
    assert manifest["recording_id"].nunique() == 2
    assert manifest["trial_id"].nunique() == 1
    assert set(manifest["phase"]) == {"phase_01", "phase_02"}
    assert set(manifest["terrain"]) == {"flat"}
    assert set(manifest["footwear"]) == {"0,0"}
    assert manifest["raw_checksum"].str.len().eq(64).all()


def test_missing_primary_is_retained_as_exclusion(config, tmp_path: Path) -> None:
    raw = tmp_path / "raw/P01/Sit_to_Stand/session/combined_log_wide_1.csv"
    raw.parent.mkdir(parents=True)
    pd.DataFrame({"Timestamp": [0, 1], "device10_IMU3_GX": [0, 1]}).to_csv(raw, index=False)
    local = config.with_values({"paths": {"raw_root": str(tmp_path / "raw"), "rss_root": str(tmp_path / "rss"), "output_root": str(tmp_path / "out")}})
    manifest = build_manifest(local)
    assert len(manifest) == 1
    assert not bool(manifest.iloc[0]["segmentation_eligible"])
    assert "non_walking_activity" in manifest.iloc[0]["exclusion_reasons"]
    assert "primary_event_sensor_missing" in manifest.iloc[0]["exclusion_reasons"]


def test_segmentation_eligibility_now_gates_on_ankle_proxy_not_gyro_sensors(config, tmp_path: Path) -> None:
    """A.7: segmentation no longer reads raw IMU, so full gyro with missing RSS ankle
    proxy columns flips eligible -> ineligible, and the reverse (incomplete gyro, full
    ankle proxy) stays eligible; gyro sensor_status remains informational only."""
    raw_root, rss_root = tmp_path / "raw", tmp_path / "rss"
    session = Path("P01/Flat Terrain/0,0_Walking/session_Trial01")
    proxy_nodes = config.section("proxy")["nodes"]

    healthy_gyro_missing_ankle = raw_root / session / "combined_log_wide_1.csv"
    _raw(healthy_gyro_missing_ankle)  # every gyro/accel channel present and finite
    rss_missing_ankle = rss_root / session / healthy_gyro_missing_ankle.stem / "limb_angle_mode.csv"
    rss_missing_ankle.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        f"{segment}_mode_1": [0.0, 1.0] for node, segment in proxy_nodes.items() if node not in {"LA", "RA"}
    }).to_csv(rss_missing_ankle, index=False)

    incomplete_gyro_full_ankle = raw_root / "P02/Flat Terrain/0,0_Walking/session_Trial01/combined_log_wide_2.csv"
    incomplete_gyro_full_ankle.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Timestamp": [0, 1], "device10_IMU3_GX": [0, 1]}).to_csv(incomplete_gyro_full_ankle, index=False)
    rss_full_ankle = rss_root / "P02/Flat Terrain/0,0_Walking/session_Trial01" / incomplete_gyro_full_ankle.stem / "limb_angle_mode.csv"
    rss_full_ankle.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({f"{segment}_mode_1": [0.0, 1.0] for segment in proxy_nodes.values()}).to_csv(rss_full_ankle, index=False)

    local = config.with_values({"paths": {"raw_root": str(raw_root), "rss_root": str(rss_root), "output_root": str(tmp_path / "out")}})
    manifest = build_manifest(local).set_index("participant_id")

    healthy_gyro_row = manifest.loc["P01"]
    assert not bool(healthy_gyro_row["segmentation_eligible"])
    assert "missing_ankle_proxy_nodes" in healthy_gyro_row["exclusion_reasons"]
    assert "primary_event_sensor_missing" not in healthy_gyro_row["exclusion_reasons"]

    incomplete_gyro_row = manifest.loc["P02"]
    assert bool(incomplete_gyro_row["segmentation_eligible"])
    assert "missing_ankle_proxy_nodes" not in incomplete_gyro_row["exclusion_reasons"]
    assert "primary_event_sensor_missing" in incomplete_gyro_row["exclusion_reasons"]

