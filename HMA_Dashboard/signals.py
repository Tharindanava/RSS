"""Signal loading and compatibility helpers for the dashboard."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gait_analysis.dashboard import clear_dashboard_cache, load_hybrid_segmentation
from gait_analysis.references import side_from_group
from HMA_Dashboard import catalog


TRACE_TO_SUFFIX: dict[str, str] = {
    "original_angle": "original_angle",
    "mode_1": "mode_1",
    "mode_2": "mode_2",
    "mode_3": "mode_3",
}

_analysis_results_dir = Path(__file__).resolve().parents[1] / "outputs" / "gait_analysis"


def configure_analysis_results_dir(path: str | Path) -> None:
    """Select the read-only hybrid segmentation artifact root."""
    global _analysis_results_dir
    _analysis_results_dir = Path(path).expanduser().resolve()
    clear_caches()


def clear_caches() -> None:
    """Clear cached CSV parsing state."""
    _load_group_data_cached.cache_clear()
    _load_all_group_mode_data_cached.cache_clear()
    clear_dashboard_cache()


def trace_column_name(group_name: str, trace_name: str) -> str:
    """Return the CSV column name for a group/trace pair."""
    suffix = TRACE_TO_SUFFIX.get(trace_name)
    if suffix is None:
        raise ValueError(f"Unsupported trace name: {trace_name}")
    return f"{group_name}_{suffix}"


def load_group_data(mode_file_path: str | Path, group_name: str, default_fs: float | None = None) -> dict[str, Any]:
    """Load all traces for one signal group."""
    fs_key = "" if default_fs is None else f"{float(default_fs):.12g}"
    return _load_group_data_cached(str(Path(mode_file_path).resolve()), str(group_name), fs_key)


@lru_cache(maxsize=256)
def _load_group_data_cached(mode_file_path_str: str, group_name: str, fs_key: str) -> dict[str, Any]:
    path = Path(mode_file_path_str)
    schema = catalog.read_mode_file_schema(path)
    traces = list(schema["group_traces"].get(group_name, []))
    if not traces:
        raise ValueError(f"Signal group '{group_name}' was not found in {path}")

    usecols = [column for column in ("frame", "time", "time_seconds") if column in schema["columns"]]
    usecols.extend(trace_column_name(group_name, trace) for trace in traces)
    df = pd.read_csv(path, usecols=usecols)
    if df.empty:
        raise ValueError(f"Mode file has no rows: {path}")

    frame = df["frame"].to_numpy(dtype=float) if "frame" in df.columns else np.arange(len(df), dtype=float)
    sample_index = np.arange(len(df), dtype=float)
    default_fs = float(fs_key) if fs_key else None

    time = None
    fs = None
    x_label = "Sample Index"
    if "time" in df.columns:
        time_values = df["time"].to_numpy(dtype=float)
        if np.any(np.isfinite(time_values)):
            time = time_values
            fs = _infer_sample_rate(time_values)
            x_label = "Time (s)"
    elif "time_seconds" in df.columns:
        time_values = df["time_seconds"].to_numpy(dtype=float)
        if np.any(np.isfinite(time_values)):
            time = time_values
            fs = _infer_sample_rate(time_values)
            x_label = "Time (s)"

    if (
        time is not None
        and fs is not None
        and default_fs is not None
        and fs > float(default_fs) * 1000.0
    ):
        time = (frame - float(frame[0])) / float(default_fs)
        fs = float(default_fs)

    if time is None and default_fs is not None:
        time = (frame - float(frame[0])) / float(default_fs)
        fs = float(default_fs)
        x_label = "Time (s)"

    hybrid = load_hybrid_segmentation(path, _analysis_results_dir, len(df))
    side = side_from_group(group_name)
    events = [event for event in hybrid["gait_events"] if side is None or event.get("side") == side]
    step_index = hybrid["step_index_by_side"].get(side) if side else None
    steps = [step for step in hybrid["gait_steps"] if side is None or step.get("side") == side]
    segment_label = np.full(len(df), "", dtype=object)
    for event in events:
        sample = int(event["sample"])
        if event.get("accepted") and 0 <= sample < len(segment_label):
            label = f"{str(event['side'])[0].upper()}-{event['event_type']}"
            segment_label[sample] = label if not segment_label[sample] else f"{segment_label[sample]}|{label}"
    return {
        "source_path": str(path),
        "group_name": group_name,
        "group_label": catalog.pretty_signal_label(group_name),
        "frame": frame,
        "sample_index": sample_index,
        "time": time,
        "fs": fs,
        "x_label": x_label,
        # These arrays are derived exclusively from the hybrid sidecar tables.
        # Legacy columns embedded in RSS files are intentionally never read.
        "segment_label": segment_label,
        "step_index": step_index,
        "gait_events": events,
        "gait_steps": steps,
        "hybrid_segmentation_available": hybrid["available"],
        "hybrid_segmentation_status": hybrid["status"],
        "hybrid_segmentation_message": hybrid["message"],
        "traces": {
            trace: df[trace_column_name(group_name, trace)].to_numpy(dtype=float).reshape(-1)
            for trace in traces
        },
        "trace_names": traces,
        "row_count": int(len(df)),
    }


def load_signal_trace(mode_file_path: str | Path, group_name: str, trace_name: str, default_fs: float | None = None) -> dict[str, Any]:
    """Load one concrete trace from a mode file."""
    group_data = load_group_data(mode_file_path, group_name, default_fs=default_fs)
    if trace_name not in group_data["traces"]:
        raise ValueError(f"Trace '{trace_name}' is not available for group '{group_name}'")
    return {
        "mode_file_path": group_data["source_path"],
        "group_name": group_name,
        "group_label": group_data["group_label"],
        "trace_name": trace_name,
        "trace_label": catalog.pretty_trace_label(trace_name),
        "frame": group_data["frame"],
        "sample_index": group_data["sample_index"],
        "time": group_data["time"],
        "fs": group_data["fs"],
        "x_label": group_data["x_label"],
        "segment_label": group_data["segment_label"],
        "step_index": group_data["step_index"],
        "gait_events": group_data["gait_events"],
        "gait_steps": group_data["gait_steps"],
        "hybrid_segmentation_available": group_data["hybrid_segmentation_available"],
        "hybrid_segmentation_status": group_data["hybrid_segmentation_status"],
        "hybrid_segmentation_message": group_data["hybrid_segmentation_message"],
        "values": np.asarray(group_data["traces"][trace_name], dtype=float).reshape(-1),
        "row_count": group_data["row_count"],
    }


def load_all_group_mode_data(mode_file_path: str | Path, default_fs: float | None = None) -> dict[str, Any]:
    """Load all groups for overview download helpers."""
    fs_key = "" if default_fs is None else f"{float(default_fs):.12g}"
    return _load_all_group_mode_data_cached(str(Path(mode_file_path).resolve()), fs_key)


@lru_cache(maxsize=128)
def _load_all_group_mode_data_cached(mode_file_path_str: str, fs_key: str) -> dict[str, Any]:
    path = Path(mode_file_path_str)
    schema = catalog.read_mode_file_schema(path)
    usecols = [column for column in ("frame", "time", "time_seconds") if column in schema["columns"]]
    for group in schema["groups"]:
        for trace in schema["group_traces"][group]:
            usecols.append(trace_column_name(group, trace))
    df = pd.read_csv(path, usecols=usecols)
    if df.empty:
        raise ValueError(f"Mode file has no rows: {path}")

    default_fs = float(fs_key) if fs_key else None
    frame = df["frame"].to_numpy(dtype=float) if "frame" in df.columns else np.arange(len(df), dtype=float)
    sample_index = np.arange(len(df), dtype=float)
    time = None
    fs = None
    x_label = "Sample Index"
    if "time" in df.columns:
        time = df["time"].to_numpy(dtype=float)
        fs = _infer_sample_rate(time)
        x_label = "Time (s)"
    elif "time_seconds" in df.columns:
        time = df["time_seconds"].to_numpy(dtype=float)
        fs = _infer_sample_rate(time)
        x_label = "Time (s)"
    if (
        time is not None
        and fs is not None
        and default_fs is not None
        and fs > float(default_fs) * 1000.0
    ):
        time = (frame - float(frame[0])) / float(default_fs)
        fs = float(default_fs)
    if time is None and default_fs is not None:
        time = (frame - float(frame[0])) / float(default_fs)
        fs = float(default_fs)
        x_label = "Time (s)"

    return {
        "source_path": str(path),
        "data": df,
        "groups": list(schema["groups"]),
        "group_traces": {group: list(schema["group_traces"][group]) for group in schema["groups"]},
        "frame": frame,
        "sample_index": sample_index,
        "time": time,
        "fs": fs,
        "x_label": x_label,
    }


def can_pair_signals(signal_a: dict[str, Any] | None, signal_b: dict[str, Any] | None) -> bool:
    """Return whether two selected traces may be used for same-run pairwise analyses."""
    if signal_a is None or signal_b is None:
        return False
    return str(signal_a.get("mode_file_path")) == str(signal_b.get("mode_file_path"))


def _infer_sample_rate(time_values: np.ndarray) -> float | None:
    time_values = np.asarray(time_values, dtype=float).reshape(-1)
    valid = time_values[np.isfinite(time_values)]
    if valid.size < 2:
        return None
    diffs = np.diff(valid)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return None
    median_dt = float(np.median(diffs))
    if median_dt <= 0.0:
        return None
    return 1.0 / median_dt
