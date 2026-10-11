"""Cohort manifest construction and recording-identity normalization."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import numpy as np

from .config import AnalysisConfig
from .models import TrialRecord
from .progress import ProgressTracker, progress_note
from .utils import hash_file


_PHASE_RE = re.compile(r"(?i)(?:[_ -]?phase[_ -]?)(0?1|0?2)")
_TRIAL_RE = re.compile(r"(?i)tr(?:ia|ai)l[_ -]?(\d+)")
_FOOTWEAR_RE = re.compile(r"(?<!\d)(\d+)\s*,\s*(\d+)(?!\d)")


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def normalized_recording_identity(path: str | Path) -> str:
    """Return a stable identity shared by a raw CSV and its RSS run directory."""
    value = Path(path)
    name = value.name
    if name in {"limb_angle_mode.csv", "joint_angle_mode.csv"}:
        value = value.parent
        name = value.name
    name = re.sub(r"(?i)\.csv$", "", name)
    return normalize_token(name)


def _parse_phase(name: str) -> str:
    match = _PHASE_RE.search(name)
    return f"phase_{int(match.group(1)):02d}" if match else "complete"


def _trial_stem(name: str) -> str:
    return _PHASE_RE.sub("", re.sub(r"(?i)\.csv$", "", name)).rstrip("_ -")


def _condition_metadata(relative: Path) -> dict[str, str]:
    text = " ".join(relative.parts)
    normalized = normalize_token(text)
    terrain = "unknown"
    for candidate, tokens in {
        "flat": ("flat",),
        "stairs": ("stair", "staircase"),
        "uneven": ("uneven", "rough"),
        "incline": ("incline", "slope"),
    }.items():
        if any(token in normalized for token in tokens):
            terrain = candidate
            break
    direction = "level"
    if re.search(r"(?i)(stair|slope|incline)[_ -]?up|\bup\b", text):
        direction = "up"
    elif re.search(r"(?i)(stair|slope|incline)[_ -]?down|\bdown\b", text):
        direction = "down"
    footwear_match = _FOOTWEAR_RE.search(text)
    footwear = f"{footwear_match.group(1)},{footwear_match.group(2)}" if footwear_match else "unknown"
    path_tokens = set(normalized.split("_"))
    if any(token in normalized for token in ("sit_to_stand", "sit_stand", "standing_up", "standingup")) or "sts" in path_tokens:
        activity = "sit_to_stand"
    elif any(token in normalized for token in ("walk", "gait")) or any(token.startswith("step") for token in path_tokens) or terrain != "unknown":
        activity = "walking"
    else:
        activity = "other"
    trial_match = _TRIAL_RE.search(text)
    return {
        "terrain": terrain,
        "direction": direction,
        "footwear": footwear,
        "activity": activity,
        "trial_number": trial_match.group(1) if trial_match else "unknown",
    }


def condition_metadata(path: str | Path) -> dict[str, str]:
    """Return normalized scenario metadata for a raw or RSS path."""
    return _condition_metadata(Path(path))


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle), [])


def canonical_header(column: str) -> str:
    """Normalize the three historical raw-IMU header conventions."""
    text = normalize_token(column).replace("accelerometer", "a").replace("accel", "a")
    text = text.replace("gyroscope", "g").replace("gyro", "g")
    text = re.sub(r"^device_?(\d+)", r"device\1", text)
    text = re.sub(r"imu_?(\d+)", r"imu\1", text)
    text = re.sub(r"_([ag])_?([xyz])$", r"_\1\2", text)
    return text


def resolve_sensor_columns(columns: Iterable[str], sensor: str) -> dict[str, str]:
    normalized = {canonical_header(column): column for column in columns}
    prefix = canonical_header(sensor)
    found: dict[str, str] = {}
    for kind in ("a", "g"):
        for axis in "xyz":
            candidates = (
                f"{prefix}_{kind}{axis}", f"{prefix}{kind}{axis}",
                f"{prefix}_{kind}_{axis}",
            )
            for candidate in candidates:
                if candidate in normalized:
                    found[f"{kind}{axis}"] = normalized[candidate]
                    break
    return found


def _rss_index(rss_root: Path, filename: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not rss_root.exists():
        return index
    for file_path in rss_root.rglob(filename):
        index[normalized_recording_identity(file_path)] = file_path.resolve()
    return index


def _sensor_availability(
    raw_file: Path, header: list[str], sensors: list[str], minimum_finite_fraction: float,
) -> tuple[dict[str, bool], dict[str, float], list[str], list[str]]:
    resolved = {sensor: resolve_sensor_columns(header, sensor) for sensor in sensors}
    missing = [sensor for sensor, columns in resolved.items() if len(columns) != 6]
    selected = sorted({column for columns in resolved.values() for column in columns.values()})
    finite_counts = {sensor: 0 for sensor in sensors}
    total_counts = {sensor: 0 for sensor in sensors}
    if selected:
        for chunk in pd.read_csv(raw_file, usecols=selected, chunksize=10000):
            for sensor, columns in resolved.items():
                if len(columns) != 6:
                    continue
                values = chunk[list(columns.values())].apply(pd.to_numeric, errors="coerce").to_numpy(float)
                finite_counts[sensor] += int(np.count_nonzero(np.isfinite(values)))
                total_counts[sensor] += int(values.size)
    fractions = {
        sensor: finite_counts[sensor] / total_counts[sensor] if total_counts[sensor] else 0.0
        for sensor in sensors
    }
    dropout = [
        sensor for sensor in sensors
        if sensor not in missing and fractions[sensor] < minimum_finite_fraction
    ]
    status = {
        sensor: sensor not in missing and sensor not in dropout for sensor in sensors
    }
    return status, fractions, missing, dropout


def build_manifest(config: AnalysisConfig, *, limit: int | None = None) -> pd.DataFrame:
    """Create one auditable row per raw CSV, including every exclusion."""
    raw_root = config.path("raw_root")
    rss_root = config.path("rss_root")
    settings = config.section("manifest")
    progress_note("manifest", f"discovering raw inputs under {raw_root}")
    raw_files = sorted(raw_root.glob(str(settings["raw_glob"])))
    if limit is not None:
        raw_files = raw_files[: int(limit)]
    progress_note("manifest", f"found {len(raw_files)} raw CSV file(s); indexing RSS inputs under {rss_root}")
    rss_by_identity = _rss_index(rss_root, str(settings["rss_filename"]))
    progress_note("manifest", f"indexed {len(rss_by_identity)} RSS recording(s)")
    sensor_config = config.section("event_sensors")
    required_sensors = [
        sensor_config[side][role]
        for side in ("left", "right")
        for role in ("primary", "fallback")
    ]
    rows: list[dict[str, Any]] = []
    tracker = ProgressTracker("manifest", len(raw_files))
    for index, raw_file in enumerate(raw_files, start=1):
        tracker.update(index - 1, f"processing {raw_file.relative_to(raw_root)}")
        relative = raw_file.relative_to(raw_root)
        participant = relative.parts[0] if relative.parts else "unknown"
        identity = normalized_recording_identity(raw_file)
        expected_rss = rss_root / relative.parent / raw_file.stem / str(settings["rss_filename"])
        rss_file = expected_rss.resolve() if expected_rss.exists() else rss_by_identity.get(identity)
        phase = _parse_phase(raw_file.name)
        condition = _condition_metadata(relative)
        header = _read_header(raw_file)
        with raw_file.open("rb") as handle:
            n_samples = max(0, sum(1 for _ in handle) - 1)
        sensor_status, sensor_fractions, missing_sensors, dropout_sensors = _sensor_availability(
            raw_file, header, required_sensors, float(settings["sensor_min_finite_fraction"])
        )
        proxy_nodes = config.section("proxy")["nodes"]
        rss_header = _read_header(rss_file) if rss_file else []
        missing_proxy_nodes = [
            node for node, segment in proxy_nodes.items() if f"{segment}_mode_1" not in rss_header
        ]
        ankle_nodes = {"LA", "RA"}
        lower_nodes = {"LH", "LK", "LA", "RH", "RK", "RA"}
        reasons: list[str] = []
        if condition["activity"] != "walking":
            reasons.append("non_walking_activity")
        if rss_file is None:
            reasons.append("rss_proxy_unavailable")
        # Gyro/event-sensor status below is retained for informational QC only: since
        # segmentation is now driven entirely by RSS ankle proxy angles (Hilbert-phase
        # ankle minima), it no longer reads raw IMU gyro/accel at all, so these reasons
        # are decoupled from segmentation_eligible (see missing_ankle_proxy_nodes below).
        if not sensor_status[required_sensors[0]] or not sensor_status[required_sensors[2]]:
            reasons.append("primary_event_sensor_missing")
        if not sensor_status[required_sensors[1]] or not sensor_status[required_sensors[3]]:
            reasons.append("fallback_event_sensor_missing")
        if dropout_sensors:
            reasons.append("event_sensor_dropout")
        for side, primary_index, fallback_index in (("left", 0, 1), ("right", 2, 3)):
            if not sensor_status[required_sensors[primary_index]] and not sensor_status[required_sensors[fallback_index]]:
                reasons.append(f"no_event_sensor_{side}")
        if any(node in ankle_nodes for node in missing_proxy_nodes):
            reasons.append("missing_ankle_proxy_nodes")
        if any(node in lower_nodes for node in missing_proxy_nodes):
            reasons.append("missing_stage1_proxy_nodes")
        if missing_proxy_nodes:
            reasons.append("missing_whole_body_proxy_nodes")
        segmentation_reasons = [
            reason for reason in reasons
            if reason in {"non_walking_activity", "rss_proxy_unavailable", "missing_ankle_proxy_nodes"}
        ]
        stage1_reasons = [
            reason for reason in reasons
            if reason in {"non_walking_activity", "rss_proxy_unavailable", "missing_stage1_proxy_nodes"}
        ]
        stage2_reasons = [
            reason for reason in reasons
            if reason in {"non_walking_activity", "rss_proxy_unavailable", "missing_whole_body_proxy_nodes"}
        ]
        trial_key = "/".join((*relative.parts[:-1], _trial_stem(raw_file.name)))
        recording_key = relative.as_posix()
        rows.append({
            "recording_id": "rec_" + hashlib.sha1(recording_key.encode()).hexdigest()[:16],
            "trial_id": "trial_" + hashlib.sha1(trial_key.encode()).hexdigest()[:16],
            "participant_id": participant,
            "raw_file": str(raw_file.resolve()),
            "rss_file": str(rss_file) if rss_file else "",
            "recording_identity": identity,
            "phase": phase,
            **condition,
            "heel_code": condition["footwear"],
            "sampling_hz": float(config.values["sampling"]["default_hz"]),
            "n_samples": n_samples,
            "duration_seconds": n_samples / float(config.values["sampling"]["default_hz"]),
            "raw_checksum": hash_file(raw_file),
            "rss_checksum": hash_file(rss_file) if rss_file else "",
            "raw_columns": len(header),
            "rss_columns": len(rss_header),
            "sensor_status_json": json.dumps(sensor_status, sort_keys=True),
            "sensor_finite_fraction_json": json.dumps(sensor_fractions, sort_keys=True),
            "missing_sensors": "|".join(missing_sensors),
            "dropout_sensors": "|".join(dropout_sensors),
            "missing_proxy_nodes": "|".join(missing_proxy_nodes),
            "segmentation_eligible": condition["activity"] == "walking" and rss_file is not None and not any(
                node in ankle_nodes for node in missing_proxy_nodes
            ),
            "stage1_eligible": condition["activity"] == "walking" and rss_file is not None and not any(node in lower_nodes for node in missing_proxy_nodes),
            "stage2_eligible": condition["activity"] == "walking" and rss_file is not None and not missing_proxy_nodes,
            "exclusion_reasons": "|".join(reasons),
            "segmentation_exclusion_reasons": "|".join(segmentation_reasons),
            "stage1_exclusion_reasons": "|".join(stage1_reasons),
            "stage2_exclusion_reasons": "|".join(stage2_reasons),
            "qc_status": "eligible" if not reasons else "review_or_excluded",
            "config_hash": config.config_hash,
        })
    tracker.update(len(raw_files), "manifest complete")
    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        progress_note(
            "manifest",
            "eligibility: segmentation=%d stage1=%d stage2=%d"
            % (
                int(manifest["segmentation_eligible"].sum()),
                int(manifest["stage1_eligible"].sum()),
                int(manifest["stage2_eligible"].sum()),
            ),
        )
    return manifest


def records_from_manifest(frame: pd.DataFrame) -> list[TrialRecord]:
    as_bool = lambda value: value if isinstance(value, (bool, np.bool_)) else str(value).strip().lower() in {"true", "1", "yes"}
    records: list[TrialRecord] = []
    for row in frame.to_dict("records"):
        reasons = tuple(filter(None, str(row.get("exclusion_reasons", "")).split("|")))
        records.append(TrialRecord(
            recording_id=str(row["recording_id"]),
            trial_id=str(row["trial_id"]),
            participant_id=str(row["participant_id"]),
            raw_file=Path(str(row["raw_file"])),
            rss_file=Path(str(row["rss_file"])) if row.get("rss_file") else None,
            activity=str(row.get("activity", "walking")),
            terrain=str(row.get("terrain", "unknown")),
            direction=str(row.get("direction", "unknown")),
            footwear=str(row.get("footwear", "unknown")),
            phase=str(row.get("phase", "complete")),
            sampling_hz=float(row.get("sampling_hz", 100.0)),
            raw_checksum=str(row.get("raw_checksum", "")),
            rss_checksum=str(row.get("rss_checksum", "")),
            segmentation_eligible=as_bool(row.get("segmentation_eligible", False)),
            stage1_eligible=as_bool(row.get("stage1_eligible", False)),
            stage2_eligible=as_bool(row.get("stage2_eligible", False)),
            exclusion_reasons=reasons,
            metadata={key: row.get(key) for key in ("trial_number", "recording_identity")},
        ))
    return records
