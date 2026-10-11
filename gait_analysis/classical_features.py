"""Classic temporal features and per-node RSS step trajectories."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .models import NodeStep, SegmentationResult, TrialRecord
from .segmentation import pair_bilateral_steps
from .utils import scaled_mad


LOWER_NODES = {"left": ("LH", "LK", "LA"), "right": ("RH", "RK", "RA")}


def bounded_symmetry(left: float, right: float, epsilon: float = 1.0e-9) -> float:
    """Bounded bilateral symmetry index in percent, [-200, 200]."""
    denominator = abs(left) + abs(right)
    return float(200.0 * (right - left) / max(denominator, epsilon))


def _event_registered(values: np.ndarray, step: NodeStep, points: int, phase_threshold: float) -> np.ndarray:
    stance_points = int(round((points - 1) * phase_threshold)) + 1
    swing_points = points - stance_points + 1
    toe_off = int(np.clip(step.toe_off_sample, step.start_sample, step.end_sample))
    stance_x = np.linspace(step.start_sample, toe_off, stance_points)
    swing_x = np.linspace(toe_off, step.end_sample, swing_points)
    x = np.concatenate((stance_x, swing_x[1:]))
    source = np.arange(len(values), dtype=float)
    return np.interp(x, source, values)


def _timing_preserving(values: np.ndarray, step: NodeStep, points: int) -> np.ndarray:
    source = np.arange(len(values), dtype=float)
    x = np.linspace(step.start_sample, step.end_sample, points)
    return np.interp(x, source, values)


def normalize_step_trajectories(
    record: TrialRecord,
    segmentation: SegmentationResult,
    proxy_signals: dict[str, np.ndarray],
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Create complete six-channel lower-limb observations from matched ankle-driven steps.

    Feeds the existing, untouched lower-limb-only MFPCA pathway (``mfpca.py``); left
    unchanged in spirit from the pre-refactor implementation, just retargeted at
    ``NodeStep.node_side`` (the ``LA``/``RA`` ankle steps returned by
    ``pair_bilateral_steps``) instead of the old per-leg ``GaitStep.side``.
    """
    settings = config.section("segmentation")
    points = int(settings["normalized_points"])
    phase_threshold = float(settings["hilbert_phase_threshold"])
    rows: list[dict[str, float | int | str]] = []
    for step_pair_id, left_step, right_step in pair_bilateral_steps(segmentation.steps):
        for step in (left_step, right_step):
            if step.end_sample >= min(len(proxy_signals[node]) for node in LOWER_NODES[step.node_side]):
                continue
            for node in LOWER_NODES[step.node_side]:
                values = proxy_signals[node]
                for representation, trajectory in (
                    ("timing_preserving", _timing_preserving(values, step, points)),
                    ("event_registered", _event_registered(values, step, points, phase_threshold)),
                ):
                    rows.extend({
                        "recording_id": record.recording_id,
                        "trial_id": record.trial_id,
                        "participant_id": record.participant_id,
                        "raw_file": str(record.raw_file),
                        "rss_file": str(record.rss_file) if record.rss_file else "",
                        "terrain": record.terrain,
                        "direction": record.direction,
                        "footwear": record.footwear,
                        "step_pair_id": step_pair_id,
                        "left_step_id": left_step.step_id,
                        "right_step_id": right_step.step_id,
                        "node_side": step.node_side,
                        "step_id": step.step_id,
                        "representation": representation,
                        "node": node,
                        "percent_step": 100.0 * point / (points - 1),
                        "value": float(value),
                        "signal_type": "rss_segment_mode1_proxy",
                    } for point, value in enumerate(trajectory))
    return pd.DataFrame(rows)


def normalize_node_trajectories(
    record: TrialRecord,
    segmentation: SegmentationResult,
    proxy_signals: dict[str, np.ndarray],
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Per-node trajectory table covering all 12 nodes (not just bilaterally paired legs).

    Emits the identical column schema as ``normalize_step_trajectories`` so this table
    is a drop-in-compatible input to ``mfpca.trajectory_tensor``/
    ``hierarchical_step_pair_weights`` with zero changes to those functions. Since most
    upper-limb/contralateral cycles have no natural bilateral partner, ``step_pair_id``
    is populated with each ``NodeStep``'s own ``step_id`` (a value that is unique per
    row group, which is all ``trajectory_tensor`` requires); ``left_step_id``/
    ``right_step_id`` are likewise self-referential rather than a true bilateral pair.
    """
    settings = config.section("segmentation")
    points = int(settings["normalized_points"])
    phase_threshold = float(settings["hilbert_phase_threshold"])
    rows: list[dict[str, float | int | str]] = []
    for step in segmentation.steps:
        if not step.accepted:
            continue
        values = proxy_signals.get(step.node)
        if values is None or step.end_sample >= len(values):
            continue
        for representation, trajectory in (
            ("timing_preserving", _timing_preserving(values, step, points)),
            ("event_registered", _event_registered(values, step, points, phase_threshold)),
        ):
            rows.extend({
                "recording_id": record.recording_id,
                "trial_id": record.trial_id,
                "participant_id": record.participant_id,
                "raw_file": str(record.raw_file),
                "rss_file": str(record.rss_file) if record.rss_file else "",
                "terrain": record.terrain,
                "direction": record.direction,
                "footwear": record.footwear,
                "step_pair_id": step.step_id,
                "left_step_id": step.step_id,
                "right_step_id": step.step_id,
                "node_side": step.node_side,
                "step_id": step.step_id,
                "representation": representation,
                "node": step.node,
                "percent_step": 100.0 * point / (points - 1),
                "value": float(value),
                "signal_type": "rss_segment_mode1_proxy",
            } for point, value in enumerate(trajectory))
    return pd.DataFrame(rows)


def compute_classic_features(
    record: TrialRecord,
    segmentation: SegmentationResult,
    proxy_signals: dict[str, np.ndarray],
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Calculate temporal and proxy-ROM step features for the two ankle-driven legs."""
    fs = segmentation.sample_rate_hz
    epsilon = float(config.section("classic")["symmetry_denominator_epsilon"])
    accepted = [step for step in segmentation.steps if step.accepted and step.node in ("LA", "RA")]
    all_ics = sorted((step.start_sample, step.node_side) for step in accepted)
    pair_ids: dict[str, str] = {}
    for pair_id, left_step, right_step in pair_bilateral_steps(accepted):
        pair_ids[left_step.step_id] = pair_id
        pair_ids[right_step.step_id] = pair_id
    rows: list[dict[str, float | str]] = []
    for side_step in accepted:
        stride = side_step.duration_seconds
        stance = side_step.stance_seconds
        swing = side_step.swing_seconds
        opposite = [sample for sample, side in all_ics if side != side_step.node_side and side_step.start_sample < sample < side_step.end_sample]
        biomechanical_step_duration = (opposite[0] - side_step.start_sample) / fs if opposite else np.nan
        opposite_stances = [
            other for other in accepted
            if other.node_side != side_step.node_side and other.toe_off_sample > side_step.start_sample
            and other.start_sample < side_step.toe_off_sample
        ]
        double_support = sum(
            max(0, min(side_step.toe_off_sample, other.toe_off_sample) - max(side_step.start_sample, other.start_sample))
            for other in opposite_stances
        ) / fs
        row: dict[str, float | str] = {
            "recording_id": record.recording_id, "trial_id": record.trial_id,
            "participant_id": record.participant_id, "step_id": side_step.step_id,
            "step_pair_id": pair_ids.get(side_step.step_id, ""),
            "side": side_step.node_side, "terrain": record.terrain, "direction": record.direction,
            "footwear": record.footwear, "phase": record.phase,
            "raw_file": str(record.raw_file), "rss_file": str(record.rss_file) if record.rss_file else "",
            "stride_duration_seconds": stride, "step_duration_seconds": biomechanical_step_duration,
            "valley_to_valley_duration_seconds": side_step.duration_seconds,
            "stance_duration_seconds": stance, "swing_duration_seconds": swing,
            "stance_fraction": stance / stride, "swing_fraction": swing / stride,
            "cadence_per_min": 60.0 / biomechanical_step_duration if np.isfinite(biomechanical_step_duration) and biomechanical_step_duration > 0 else 120.0 / stride,
            "double_support_duration_seconds": double_support,
            "double_support_fraction": double_support / stride,
            "boundary_source": "hilbert_ankle_minima",
            "signal_type": "rss_segment_mode1_proxy",
        }
        for node in LOWER_NODES[side_step.node_side]:
            values = np.asarray(proxy_signals[node][side_step.start_sample:side_step.end_sample + 1], dtype=float)
            if values.size and np.any(np.isfinite(values)):
                minimum, maximum = float(np.nanmin(values)), float(np.nanmax(values))
                row[f"{node}_proxy_min"] = minimum
                row[f"{node}_proxy_max"] = maximum
                row[f"{node}_proxy_rom"] = maximum - minimum
                row[f"{node}_proxy_peak_timing_fraction"] = float(np.nanargmax(values) / max(1, values.size - 1))
                toe_offset = int(np.clip(side_step.toe_off_sample - side_step.start_sample, 1, max(1, values.size - 1)))
                stance_values = values[:toe_offset + 1]
                swing_values = values[toe_offset:]
                row[f"{node}_proxy_stance_rom"] = float(np.nanmax(stance_values) - np.nanmin(stance_values))
                row[f"{node}_proxy_swing_rom"] = float(np.nanmax(swing_values) - np.nanmin(swing_values))
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["step_index"] = frame.groupby(["recording_id", "side"]).cumcount()
    left = frame[frame["side"] == "left"]
    right = frame[frame["side"] == "right"]
    paired = left[left["step_pair_id"] != ""].merge(
        right[right["step_pair_id"] != ""], on=["recording_id", "step_pair_id"], suffixes=("_left", "_right")
    )
    if not paired.empty:
        symmetry = paired[["recording_id", "step_pair_id"]].copy()
        temporal = (
            "stride_duration_seconds", "step_duration_seconds", "stance_duration_seconds",
            "swing_duration_seconds", "stance_fraction", "swing_fraction", "cadence_per_min",
            "double_support_duration_seconds", "double_support_fraction",
        )
        for column in temporal:
            symmetry[f"{column}_symmetry_index"] = [
                bounded_symmetry(left_value, right_value, epsilon)
                for left_value, right_value in zip(paired[f"{column}_left"], paired[f"{column}_right"])
            ]
        symmetry["stride_symmetry_index"] = symmetry["stride_duration_seconds_symmetry_index"]
        for joint, left_node, right_node in (("hip", "LH", "RH"), ("knee", "LK", "RK"), ("ankle", "LA", "RA")):
            for suffix in ("rom", "stance_rom", "swing_rom"):
                symmetry[f"{joint}_proxy_{suffix}_symmetry_index"] = [
                    bounded_symmetry(left_value, right_value, epsilon)
                    for left_value, right_value in zip(
                        paired[f"{left_node}_proxy_{suffix}_left"], paired[f"{right_node}_proxy_{suffix}_right"]
                    )
                ]
        frame = frame.merge(symmetry, on=["recording_id", "step_pair_id"], how="left")
    return frame


def compute_node_step_features(
    record: TrialRecord,
    segmentation: SegmentationResult,
    proxy_signals: dict[str, np.ndarray],
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Long-format per-(node, cycle) timing and proxy-ROM features across all 12 nodes.

    Generalizes the per-node ROM computation in ``compute_classic_features`` (which is
    restricted to the two ankle-driven legs' ``LOWER_NODES``) to every node, one row per
    accepted ``NodeStep``.
    """
    rows: list[dict[str, float | str]] = []
    for step in segmentation.steps:
        if not step.accepted:
            continue
        row: dict[str, float | str] = {
            "recording_id": record.recording_id, "trial_id": record.trial_id,
            "participant_id": record.participant_id, "step_id": step.step_id,
            "cycle_id": step.cycle_id, "cycle_index": step.cycle_index,
            "node": step.node, "node_side": step.node_side,
            "driving_ankle_side": step.driving_ankle_side, "coordination": step.coordination,
            "terrain": record.terrain, "direction": record.direction, "footwear": record.footwear,
            "phase": record.phase,
            "raw_file": str(record.raw_file), "rss_file": str(record.rss_file) if record.rss_file else "",
            "duration_seconds": step.duration_seconds, "stance_seconds": step.stance_seconds,
            "swing_seconds": step.swing_seconds, "stance_fraction": step.stance_fraction,
            "swing_fraction": (1.0 - step.stance_fraction) if np.isfinite(step.stance_fraction) else np.nan,
            "toe_off_time_fraction": step.toe_off_time_fraction,
            "phase_rotation_per_stride": step.phase_rotation_per_stride,
            "boundary_source": "hilbert_ankle_minima",
            "signal_type": "rss_segment_mode1_proxy",
        }
        values_source = proxy_signals.get(step.node)
        if values_source is not None and step.end_sample < len(values_source):
            values = np.asarray(values_source[step.start_sample:step.end_sample + 1], dtype=float)
            if values.size and np.any(np.isfinite(values)):
                minimum, maximum = float(np.nanmin(values)), float(np.nanmax(values))
                row["proxy_min"] = minimum
                row["proxy_max"] = maximum
                row["proxy_rom"] = maximum - minimum
                row["proxy_peak_timing_fraction"] = float(np.nanargmax(values) / max(1, values.size - 1))
                toe_offset = int(np.clip(step.toe_off_sample - step.start_sample, 1, max(1, values.size - 1)))
                stance_values = values[:toe_offset + 1]
                swing_values = values[toe_offset:]
                row["proxy_stance_rom"] = float(np.nanmax(stance_values) - np.nanmin(stance_values))
                row["proxy_swing_rom"] = float(np.nanmax(swing_values) - np.nanmin(swing_values))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_trial_features(step_features: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Aggregate phase-recording steps to their shared trial using median/scaled MAD."""
    if step_features.empty:
        return pd.DataFrame()
    identifiers = ["trial_id", "participant_id", "terrain", "direction", "footwear"]
    excluded_numeric = {"step_index"}
    numeric = [column for column in step_features.select_dtypes(include=[np.number]).columns if column not in excluded_numeric]
    rows: list[dict[str, float | int | str]] = []
    scale = float(config.section("classic")["mad_scale"])
    for keys, group in step_features.groupby(identifiers, dropna=False):
        row: dict[str, float | int | str] = dict(zip(identifiers, keys))
        row["n_steps"] = int(len(group))
        row["n_recordings"] = int(group["recording_id"].nunique())
        row["source_recordings"] = "|".join(sorted(group["recording_id"].astype(str).unique()))
        row["raw_files"] = "|".join(sorted(group["raw_file"].astype(str).unique())) if "raw_file" in group else ""
        row["rss_files"] = "|".join(sorted(group["rss_file"].astype(str).unique())) if "rss_file" in group else ""
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce")
            row[column] = float(values.median())
            row[f"{column}_scaled_mad"] = scaled_mad(values, scale)
            if column in {"stride_duration_seconds", "step_duration_seconds", "stance_duration_seconds", "swing_duration_seconds"}:
                median = float(values.median())
                row[f"{column}_cv"] = float(100.0 * values.std(ddof=1) / median) if median and len(values.dropna()) > 1 else np.nan
                row[f"{column}_rcv"] = float(100.0 * scaled_mad(values, scale) / median) if median else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
