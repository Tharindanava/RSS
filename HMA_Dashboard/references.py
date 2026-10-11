"""Dashboard operations for recording, orientation, and step references."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from gait_analysis.references import (
    file_checksum, load_registry, median_interval_template, orientation_verdict, put_reference,
    reference_entry, reference_for, remove_reference, save_registry, side_from_group,
    standardize_template, template_correlation,
)
from HMA_Dashboard import signals


def registry_path(results_dir: str | Path) -> Path:
    return Path(results_dir).resolve() / "hma_dashboard_references.json"


def accepted_ic_intervals(events: list[dict[str, Any]], side: str) -> list[tuple[int, int]]:
    contacts = sorted({
        int(event["sample"]) for event in events
        if event.get("side") == side and event.get("event_type") == "IC" and bool(event.get("accepted"))
    })
    return [(start, end) for start, end in zip(contacts, contacts[1:]) if end > start]


def _scenario(record: dict[str, Any]) -> dict[str, str]:
    scenario = record.get("scenario") or {}
    return {name: str(scenario.get(name, "unknown")) for name in ("terrain", "direction", "footwear")}


def set_reference_recording(
    record: dict[str, Any], group_name: str, *, results_dir: str | Path, default_fs: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Capture all available same-side Mode 1 templates from one recording."""
    side = side_from_group(group_name)
    if side is None:
        raise ValueError("References can only be assigned to a left or right limb group")
    selected = signals.load_signal_trace(record["mode_file_path"], group_name, "mode_1", default_fs=default_fs)
    intervals = accepted_ic_intervals(selected.get("gait_events", []), side)
    if not intervals:
        raise ValueError("No consecutive accepted same-side IC intervals are available")
    templates: dict[str, np.ndarray] = {}
    for group in record.get("available_groups", []):
        if side_from_group(str(group)) != side:
            continue
        data = signals.load_signal_trace(record["mode_file_path"], str(group), "mode_1", default_fs=default_fs)
        templates[str(group)] = median_interval_template(data["values"], intervals)
    if not templates:
        raise ValueError(f"No {side}-side Mode 1 groups are available")
    scenario = _scenario(record)
    entry = reference_entry(
        **scenario, side=side, source_path=record["mode_file_path"], mode1_templates=templates,
        run_label=str(record.get("label", "")), project_root=Path(results_dir).resolve().parent,
    )
    path = registry_path(results_dir)
    updated = put_reference(load_registry(path), entry)
    save_registry(updated, path)
    return updated, entry


def clear_reference(record: dict[str, Any], group_name: str, *, results_dir: str | Path) -> dict[str, Any]:
    side = side_from_group(group_name)
    if side is None:
        raise ValueError("References can only be cleared for a left or right limb group")
    scenario = _scenario(record)
    path = registry_path(results_dir)
    updated = remove_reference(load_registry(path), **scenario, side=side)
    save_registry(updated, path)
    return updated


def check_orientation(
    record: dict[str, Any], group_name: str, *, results_dir: str | Path, default_fs: float | None = None,
) -> tuple[float, str]:
    side = side_from_group(group_name)
    if side is None:
        raise ValueError("Orientation checks require a left or right limb group")
    scenario = _scenario(record)
    entry = reference_for(load_registry(registry_path(results_dir)), **scenario, side=side)
    if entry is None:
        raise ValueError("No reference recording is set for this scenario and side")
    if reference_is_stale(entry):
        raise ValueError("The reference source checksum has changed; replace the reference before checking orientation")
    template = entry.get("mode1_templates", {}).get(group_name)
    if template is None:
        raise ValueError(f"The reference has no Mode 1 template for {group_name}")
    selected = signals.load_signal_trace(record["mode_file_path"], group_name, "mode_1", default_fs=default_fs)
    intervals = accepted_ic_intervals(selected.get("gait_events", []), side)
    if not intervals:
        raise ValueError("No consecutive accepted same-side IC intervals are available")
    current = median_interval_template(selected["values"], intervals)
    correlation = template_correlation(current, template)
    return correlation, orientation_verdict(correlation)


def reference_is_stale(entry: dict[str, Any]) -> bool:
    source = Path(str(entry.get("source", {}).get("path", "")))
    expected = str(entry.get("source", {}).get("sha256", ""))
    return not source.is_file() or not expected or file_checksum(source) != expected


def _filtered_valleys(values: np.ndarray, fs: float) -> np.ndarray:
    finite = np.isfinite(values)
    if finite.sum() < 2:
        raise ValueError("The selected Mode 1 signal has too few finite values")
    filled = np.interp(np.arange(values.size), np.flatnonzero(finite), values[finite])
    b, a = butter(4, [0.5 / (fs / 2.0), 6.0 / (fs / 2.0)], btype="bandpass")
    filtered = filtfilt(b, a, filled)
    return find_peaks(-filtered)[0]


def _snap_endpoint(valleys: np.ndarray, endpoint: int, radius: int) -> int:
    nearby = valleys[np.abs(valleys - endpoint) <= radius]
    if not nearby.size:
        raise ValueError(f"No signal valley occurs within the ±0.20 s snap window of sample {endpoint}")
    distances = np.abs(nearby - endpoint)
    return int(nearby[np.flatnonzero(distances == distances.min())[0]])


def set_reference_step(
    record: dict[str, Any], group_name: str, start_sample: int, end_sample: int, *,
    results_dir: str | Path, default_fs: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Snap a raw zoom range to valleys and store it as this side's driver step."""
    side = side_from_group(group_name)
    if side is None:
        raise ValueError("A reference step requires a left or right limb group")
    scenario = _scenario(record)
    path = registry_path(results_dir)
    registry = load_registry(path)
    entry = reference_for(registry, **scenario, side=side)
    if entry is None:
        raise ValueError("Set a reference recording for this scenario and side first")
    if reference_is_stale(entry):
        raise ValueError("The reference source checksum has changed; replace the reference before setting its step")
    if Path(entry["source"]["path"]).resolve() != Path(record["mode_file_path"]).resolve():
        raise ValueError("Set the reference step while viewing the protected reference recording")
    signal = signals.load_signal_trace(record["mode_file_path"], group_name, "mode_1", default_fs=default_fs)
    values = np.asarray(signal["values"], dtype=float)
    fs = float(signal.get("fs") or default_fs or 100.0)
    raw_start, raw_end = sorted((int(start_sample), int(end_sample)))
    valleys = _filtered_valleys(values, fs)
    radius = max(1, int(round(0.20 * fs)))
    start, end = _snap_endpoint(valleys, raw_start, radius), _snap_endpoint(valleys, raw_end, radius)
    if end <= start:
        raise ValueError("The snapped zoom endpoints do not enclose one valley-to-valley step")
    reference_step = {
        "driver_group": group_name,
        "start_sample": start,
        "end_sample": end,
        "duration_seconds": (end - start) / fs,
        "normalized_shape": standardize_template(values[start:end + 1]).tolist(),
        "start_snap_offset_seconds": (start - raw_start) / fs,
        "end_snap_offset_seconds": (end - raw_end) / fs,
        "boundary_source": "dashboard_zoom_valley_snap",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    replacement = deepcopy(entry)
    replacement["reference_step"] = reference_step
    updated = put_reference(registry, replacement)
    save_registry(updated, path)
    return updated, reference_step
