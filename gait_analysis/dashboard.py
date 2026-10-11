"""Read-only dashboard adapter for limb-specific gait artifacts."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metadata import normalized_recording_identity


def _candidate_manifests(results_root: Path) -> list[Path]:
    if results_root.is_file() and results_root.name == "manifest.csv":
        return [results_root]
    direct = results_root / "manifest.csv"
    candidates = ([direct] if direct.exists() else []) + list(results_root.glob("runs/*/manifest.csv"))
    return sorted(set(candidates), key=lambda path: path.stat().st_mtime, reverse=True)


def _boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _empty_result(message: str = "gait analysis artifacts unavailable") -> dict[str, Any]:
    return {
        "available": False, "step_segmentation_available": False, "status": "unavailable", "message": message,
        "segment_label": None, "step_index_by_side": {"left": None, "right": None},
        "gait_events": [], "gait_steps": [],
    }


@lru_cache(maxsize=256)
def _load_hybrid_cached(mode_file: str, results_root: str, row_count: int) -> dict[str, Any]:
    """Adapt ``gait_steps.csv`` (ankle-driven ``NodeStep`` rows, all 12 nodes) for the dashboard.

    Segmentation now produces only ``gait_steps.csv`` (no separate event file): each
    accepted ``LA``/``RA`` row carries ``start_sample``/``toe_off_sample``/``end_sample``
    for one ankle-driven gait cycle. IC/TO marker events shown by the dashboard overlay
    (and consumed by ``HMA_Dashboard``'s reference-authoring tools, which expect a
    ``gait_events``-shaped list) are derived in-memory from those two fields per step;
    no ``gait_events.csv`` file is written or read.
    """
    mode_path = Path(mode_file).resolve()
    identity = normalized_recording_identity(mode_path)
    for manifest_path in _candidate_manifests(Path(results_root).resolve()):
        run_dir = manifest_path.parent
        step_path = run_dir / "gait_steps.csv"
        if not step_path.exists():
            continue
        manifest = pd.read_csv(manifest_path)
        rss_match = manifest.get("rss_file", pd.Series("", index=manifest.index)).astype(str).map(
            lambda value: Path(value).resolve() == mode_path if value else False
        )
        identity_match = manifest.get("recording_identity", pd.Series("", index=manifest.index)).astype(str) == identity
        matches = manifest[rss_match | identity_match]
        if matches.empty:
            continue
        recording_id = str(matches.iloc[0]["recording_id"])
        steps = pd.read_csv(step_path)
        if not steps.empty:
            steps = steps[steps["recording_id"].astype(str) == recording_id].copy()
        leg_steps = steps[steps["node"].isin(["LA", "RA"])].copy() if not steps.empty and "node" in steps else pd.DataFrame()
        if leg_steps.empty:
            continue
        segment_label = np.full(int(row_count), "", dtype=object)
        event_records: list[dict[str, Any]] = []
        step_records: list[dict[str, Any]] = []
        step_index_by_side = {side: np.full(int(row_count), -1, dtype=int) for side in ("left", "right")}
        accepted_mask = _boolean_series(leg_steps["accepted"])
        for side in ("left", "right"):
            accepted_side = leg_steps[(leg_steps["node_side"].astype(str) == side) & accepted_mask].sort_values("start_sample")
            for index, step in enumerate(accepted_side.to_dict("records")):
                start, end = int(step["start_sample"]), int(step["end_sample"])
                toe_off = int(step["toe_off_sample"])
                step_index_by_side[side][max(0, start):min(row_count, end + 1)] = index
                step_records.append({
                    "step_id": str(step["step_id"]), "side": side, "start_sample": start,
                    "end_sample": end, "toe_off_sample": toe_off, "accepted": True,
                    "boundary_source": "hilbert_ankle_minima",
                    "reasons": "" if pd.isna(step.get("reasons")) else str(step.get("reasons", "")),
                })
                for event_type, sample in (("IC", start), ("TO", toe_off)):
                    accepted_event = True
                    if 0 <= sample < row_count:
                        label = f"{side[0].upper()}-{event_type}"
                        segment_label[sample] = label if not segment_label[sample] else f"{segment_label[sample]}|{label}"
                    event_records.append({
                        "sample": sample, "side": side, "event_type": event_type,
                        "accepted": accepted_event, "confidence": 1.0, "reasons": "",
                        "primary_sensor": "hilbert_ankle_minima",
                    })
        accepted_count = int(accepted_mask.sum())
        rejected_count = int(len(leg_steps) - accepted_count)
        reason_counts = Counter(
            reason for value in leg_steps.loc[~accepted_mask, "reasons"].dropna().astype(str)
            for reason in value.split("|") if reason
        )
        qc_suffix = f"; QC: {reason_counts.most_common(1)[0][0]}" if reason_counts else ""
        message = f"Step segmentation: {accepted_count} accepted, {rejected_count} rejected gait cycles{qc_suffix}"
        status = "ok" if accepted_count else "qc_failed"
        return {
            "available": True, "step_segmentation_available": True, "status": status,
            "message": message, "recording_id": recording_id, "run_dir": str(run_dir),
            "segment_label": segment_label, "step_index_by_side": step_index_by_side,
            "gait_events": event_records, "gait_steps": step_records,
        }
    return _empty_result()


def load_hybrid_segmentation(mode_file: str | Path, results_root: str | Path, row_count: int) -> dict[str, Any]:
    return _load_hybrid_cached(str(Path(mode_file).resolve()), str(Path(results_root).resolve()), int(row_count))


def clear_dashboard_cache() -> None:
    _load_hybrid_cached.cache_clear()
