"""Readers for raw event sensors and explicitly named RSS Mode 1 proxies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .metadata import canonical_header, resolve_sensor_columns
from .models import TrialRecord


def _time_from_frame(frame: pd.DataFrame, default_hz: float) -> tuple[np.ndarray, float]:
    time_column = next((name for name in frame.columns if canonical_header(name) in {"time", "time_seconds"}), None)
    if time_column:
        time = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(float)
        diffs = np.diff(time[np.isfinite(time)])
        diffs = diffs[diffs > 0]
        if diffs.size:
            dt = float(np.median(diffs))
            if 1.0e-5 < dt < 0.5:
                return time - time[np.flatnonzero(np.isfinite(time))[0]], 1.0 / dt
    return np.arange(len(frame), dtype=float) / default_hz, default_hz


def load_raw_event_signals(record: TrialRecord, config: AnalysisConfig) -> dict[str, Any]:
    header = list(pd.read_csv(record.raw_file, nrows=0).columns)
    sensor_config = config.section("event_sensors")
    resolved: dict[str, dict[str, dict[str, str]]] = {}
    usecols: set[str] = set()
    for side in ("left", "right"):
        resolved[side] = {}
        for role in ("primary", "fallback"):
            sensor = str(sensor_config[side][role])
            columns = resolve_sensor_columns(header, sensor)
            resolved[side][role] = columns
            usecols.update(columns.values())
    time_columns = [column for column in header if canonical_header(column) in {"timestamp", "time", "time_seconds"}]
    usecols.update(time_columns[:1])
    raw = pd.read_csv(record.raw_file, usecols=list(usecols))
    time, fs = _time_from_frame(raw, record.sampling_hz)
    sensors: dict[str, Any] = {}
    for side in ("left", "right"):
        sensors[side] = {}
        for role in ("primary", "fallback"):
            columns = resolved[side][role]
            if len(columns) != 6:
                sensors[side][role] = None
                continue
            accel = np.column_stack([pd.to_numeric(raw[columns[f"a{axis}"]], errors="coerce") for axis in "xyz"])
            gyro = np.column_stack([pd.to_numeric(raw[columns[f"g{axis}"]], errors="coerce") for axis in "xyz"])
            sensors[side][role] = {
                "sensor": str(sensor_config[side][role]),
                "accel": accel.astype(float),
                "gyro": gyro.astype(float),
                "finite_fraction": float(np.mean(np.isfinite(np.column_stack((accel, gyro))))),
            }
    return {"time": time, "fs": fs, "sensors": sensors, "n_samples": len(raw)}


def _semantic_profile(config: AnalysisConfig) -> dict[str, Any]:
    configured = config.section("proxy").get("semantic_axis_profile")
    if not configured:
        return {}
    path = Path(str(configured))
    if not path.is_absolute():
        path = config.project_root / path
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("limbs", {})


def load_proxy_signals(record: TrialRecord, config: AnalysisConfig) -> dict[str, Any]:
    """Load stored Mode 1 signals; legacy semantic-axis signs require opt-in."""
    if record.rss_file is None:
        raise FileNotFoundError(f"No RSS proxy linked to {record.recording_id}")
    nodes = config.section("proxy")["nodes"]
    columns = [f"{segment}_mode_1" for segment in nodes.values()]
    header = list(pd.read_csv(record.rss_file, nrows=0).columns)
    missing = [column for column in columns if column not in header]
    if missing:
        raise ValueError(f"Missing RSS Mode 1 proxy columns: {', '.join(missing)}")
    time_columns = [column for column in ("time", "time_seconds", "frame") if column in header]
    frame = pd.read_csv(record.rss_file, usecols=time_columns + columns)
    time, fs = _time_from_frame(frame, record.sampling_hz)
    orientation_method = str(config.section("proxy").get("orientation_method", "stored"))
    legacy_semantic = orientation_method == "semantic_axis_legacy"
    axes_path = record.rss_file.with_name("limb_axes_mode.csv")
    axes = pd.read_csv(axes_path) if legacy_semantic and axes_path.exists() else pd.DataFrame()
    profile = _semantic_profile(config) if legacy_semantic else {}
    low_threshold = float(config.section("proxy")["low_alignment_threshold"])
    sign_floor = float(config.section("proxy")["sign_alignment_floor"])
    signals: dict[str, np.ndarray] = {}
    qc_rows: list[dict[str, Any]] = []
    for node, segment in nodes.items():
        axis_columns = [f"{segment}_mode_1_axis_{axis}" for axis in "xyz"]
        if all(column in axes for column in axis_columns):
            axis = np.nanmedian(axes[axis_columns].to_numpy(float), axis=0)
            norm = np.linalg.norm(axis)
            axis = axis / norm if np.isfinite(norm) and norm > 0 else np.full(3, np.nan)
        else:
            axis = np.full(3, np.nan)
        semantic = np.asarray(profile.get(segment, {}).get("semantic_axes", [[1.0, 0.0, 0.0]])[0], dtype=float)
        semantic /= np.linalg.norm(semantic) or 1.0
        dot = float(np.dot(axis, semantic)) if np.all(np.isfinite(axis)) else float("nan")
        if not legacy_semantic:
            sign, sign_basis = 1.0, "stored_csv"
        elif np.isfinite(dot) and abs(dot) >= sign_floor:
            sign = 1.0 if dot >= 0 else -1.0
            sign_basis = "semantic_axis"
        elif np.any(np.isfinite(axis)):
            dominant = axis[int(np.nanargmax(np.abs(axis)))]
            sign = 1.0 if dominant >= 0 else -1.0
            sign_basis = "dominant_axis_fallback"
        else:
            sign, sign_basis = 1.0, "axis_unavailable"
        values = pd.to_numeric(frame[f"{segment}_mode_1"], errors="coerce").to_numpy(float)
        signals[str(node)] = sign * values
        qc_rows.append({
            "recording_id": record.recording_id, "trial_id": record.trial_id,
            "node": node, "segment": segment, "signal_type": "rss_segment_mode1_proxy",
            "axis_x": axis[0], "axis_y": axis[1], "axis_z": axis[2],
            "semantic_alignment": dot, "absolute_axis_alignment": abs(dot) if np.isfinite(dot) else np.nan,
            "sign_multiplier": sign, "sign_basis": sign_basis,
            "low_alignment": not np.isfinite(dot) or abs(dot) < low_threshold,
            "finite_fraction": float(np.mean(np.isfinite(values))),
        })
    return {"time": time, "fs": fs, "signals": signals, "qc": pd.DataFrame(qc_rows), "n_samples": len(frame)}
