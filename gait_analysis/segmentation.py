"""Ankle-driven Hilbert-phase gait-cycle segmentation.

Every proxy node's gait cycles are derived from one of the two ankle proxy
signals (``LA``/``RA``): lower-limb nodes stay ipsilateral (the left ankle
drives ``LH``/``LK``/``LA``; the right ankle drives ``RH``/``RK``/``RA``), while
upper-limb nodes are driven by the *contralateral* ankle (the left ankle
drives ``RS``/``RE``/``RW``; the right ankle drives ``LS``/``LE``/``LW``),
matching natural contralateral arm-leg coupling during walking.

For each ankle independently: bandpass-filter the proxy angle trace, take its
Hilbert analytic-signal phase, find angle minima ("minimum stable points")
between which one gait cycle is defined, reject cycles with an out-of-range
duration or phase rotation, and locate toe-off as the sample where the
cycle's normalized Hilbert phase crosses a configurable threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, hilbert

from .config import AnalysisConfig
from .models import NodeStep, SegmentationResult, TrialRecord


NODE_ANATOMICAL_SIDE = {
    "LH": "left", "LK": "left", "LA": "left", "LS": "left", "LE": "left", "LW": "left",
    "RH": "right", "RK": "right", "RA": "right", "RS": "right", "RE": "right", "RW": "right",
}
LOWER_LIMB_NODES = frozenset({"LH", "LK", "LA", "RH", "RK", "RA"})
UPPER_LIMB_NODES = frozenset({"LS", "LE", "LW", "RS", "RE", "RW"})
ANKLE_PROXY_NODE = {"left": "LA", "right": "RA"}


def driving_ankle_side(node: str) -> str:
    """Return which ankle ("left"/"right") produces ``node``'s cycle boundaries.

    Lower-limb nodes are ipsilateral (driven by their own side's ankle); upper-limb
    nodes are driven by the *contralateral* ankle (left ankle -> right arm, right
    ankle -> left arm).
    """
    side = NODE_ANATOMICAL_SIDE[node]
    return side if node in LOWER_LIMB_NODES else ("right" if side == "left" else "left")


def _nodes_driven_by(ankle_side: str) -> tuple[str, ...]:
    return tuple(node for node in NODE_ANATOMICAL_SIDE if driving_ankle_side(node) == ankle_side)


def filter_walking_band(values: np.ndarray, fs: float, low: float = 0.5, high: float = 6.0, order: int = 4) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size < max(32, order * 6):
        raise ValueError("Signal is too short for zero-phase gait filtering")
    b, a = butter(order, [low / (fs / 2.0), high / (fs / 2.0)], btype="bandpass")
    if array.ndim == 1:
        finite = np.isfinite(array)
        fill = np.interp(np.arange(array.size), np.flatnonzero(finite), array[finite]) if finite.any() else np.zeros_like(array)
        return filtfilt(b, a, fill)
    return np.column_stack([filter_walking_band(array[:, index], fs, low, high, order) for index in range(array.shape[1])])


def walking_plane_gyro(gyro: np.ndarray, fs: float, settings: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Retained for ``report/build_report.py``'s frozen segmentation-example figure.

    No longer used by the ankle-minima detector below, which reads RSS proxy
    angle signals rather than raw gyroscope traces.
    """
    filtered = filter_walking_band(
        gyro, fs, float(settings["bandpass_hz"][0]), float(settings["bandpass_hz"][1]), int(settings["filter_order"])
    )
    centered = filtered - np.mean(filtered, axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    axis = right[0]
    trace = centered @ axis
    distance = max(1, int(float(settings.get("orientation_peak_distance_seconds", 0.25)) * fs))
    prominence = np.std(trace) * float(settings.get("orientation_min_prominence_sd", 0.25))
    positive, positive_props = find_peaks(trace, distance=distance, prominence=prominence)
    negative, negative_props = find_peaks(-trace, distance=distance, prominence=prominence)
    positive_score = np.median(positive_props.get("prominences", [0.0])) if positive.size else 0.0
    negative_score = np.median(negative_props.get("prominences", [0.0])) if negative.size else 0.0
    if negative_score > positive_score:
        trace, axis = -trace, -axis
    return trace, axis


@dataclass
class _AnkleCycle:
    cycle_index: int
    start_sample: int
    end_sample: int
    toe_off_sample: int
    toe_off_time_fraction: float
    duration_seconds: float
    stance_seconds: float
    swing_seconds: float
    stance_fraction: float
    phase_rotation_per_stride: float
    valley_prominence: float | None
    accepted: bool
    reasons: tuple[str, ...]


def _phase_threshold_offset(phase_segment: np.ndarray, threshold: float) -> float | None:
    """Sub-sample offset (0..len-1) within ``phase_segment`` where the phase,
    normalized to [0, 1] across the segment, first crosses ``threshold``."""
    span = phase_segment[-1] - phase_segment[0]
    if not np.isfinite(span) or span == 0:
        return None
    normalized = (phase_segment - phase_segment[0]) / span
    crossings = np.flatnonzero((normalized[:-1] < threshold) & (normalized[1:] >= threshold))
    if not crossings.size:
        return None
    index = int(crossings[0])
    lo, hi = normalized[index], normalized[index + 1]
    fraction = (threshold - lo) / (hi - lo) if hi != lo else 0.0
    return index + fraction


def _detect_ankle_cycles(values: np.ndarray, fs: float, settings: dict[str, Any]) -> list[_AnkleCycle]:
    """Detect minimum-to-minimum gait cycles on one ankle's proxy angle trace."""
    filtered = filter_walking_band(
        values, fs, float(settings["bandpass_hz"][0]), float(settings["bandpass_hz"][1]), int(settings["filter_order"]),
    )
    phase = np.unwrap(np.angle(hilbert(filtered)))
    stride_low, stride_high = map(float, settings["stride_seconds"])
    rotation_low, rotation_high = map(float, settings["phase_rotation_bounds"])
    threshold = float(settings["hilbert_phase_threshold"])
    distance = max(1, int(round(stride_low * fs)))
    median = float(np.median(filtered))
    mad = float(np.median(np.abs(filtered - median))) + 1.0e-12
    prominence = float(settings["valley_prominence_mad"]) * mad
    minima, properties = find_peaks(-filtered, distance=distance, prominence=prominence)
    prominences = properties.get("prominences", np.zeros(len(minima)))
    cycles: list[_AnkleCycle] = []
    for index in range(len(minima) - 1):
        start, end = int(minima[index]), int(minima[index + 1])
        duration = (end - start) / fs
        rotation = abs(float((phase[end] - phase[start]) / (2.0 * np.pi)))
        reasons: list[str] = []
        if not stride_low <= duration <= stride_high:
            reasons.append("stride_duration_out_of_range")
        if not rotation_low <= rotation <= rotation_high:
            reasons.append("phase_rotation_out_of_range")
        offset = _phase_threshold_offset(phase[start:end + 1], threshold)
        if offset is None:
            reasons.append("phase_threshold_not_crossed")
            offset = 0.5 * (end - start)
        toe_off_sample = int(np.clip(round(start + offset), start, end))
        toe_off_time_fraction = float(offset / (end - start)) if end > start else float("nan")
        stance_seconds = (toe_off_sample - start) / fs
        swing_seconds = (end - toe_off_sample) / fs
        cycles.append(_AnkleCycle(
            cycle_index=index, start_sample=start, end_sample=end, toe_off_sample=toe_off_sample,
            toe_off_time_fraction=toe_off_time_fraction, duration_seconds=duration,
            stance_seconds=stance_seconds, swing_seconds=swing_seconds,
            stance_fraction=(stance_seconds / duration) if duration > 0 else float("nan"),
            phase_rotation_per_stride=rotation,
            valley_prominence=float(prominences[index]) if index < len(prominences) else None,
            accepted=not reasons, reasons=tuple(reasons),
        ))
    return cycles


def _cycles_to_node_steps(
    cycles: list[_AnkleCycle], *, recording_id: str, trial_id: str, node: str, ankle_side: str,
) -> list[NodeStep]:
    node_side = NODE_ANATOMICAL_SIDE[node]
    coordination = "ipsilateral" if node in LOWER_LIMB_NODES else "contralateral"
    return [
        NodeStep(
            recording_id=recording_id,
            trial_id=trial_id,
            step_id=f"{recording_id}:{node}:{cycle.cycle_index:04d}",
            node=node,
            node_side=node_side,
            driving_ankle_side=ankle_side,
            coordination=coordination,
            cycle_id=f"{recording_id}:{ankle_side}:{cycle.cycle_index:04d}",
            cycle_index=cycle.cycle_index,
            start_sample=cycle.start_sample,
            end_sample=cycle.end_sample,
            toe_off_sample=cycle.toe_off_sample,
            toe_off_time_fraction=cycle.toe_off_time_fraction,
            duration_seconds=cycle.duration_seconds,
            stance_seconds=cycle.stance_seconds,
            swing_seconds=cycle.swing_seconds,
            stance_fraction=cycle.stance_fraction,
            phase_rotation_per_stride=cycle.phase_rotation_per_stride,
            valley_prominence=cycle.valley_prominence,
            accepted=cycle.accepted,
            reasons=cycle.reasons,
        )
        for cycle in cycles
    ]


def pair_bilateral_steps(steps: Iterable[NodeStep]) -> list[tuple[str, NodeStep, NodeStep]]:
    """Deterministic monotonic maximum-overlap matching of accepted ankle steps."""
    ankle_steps = [step for step in steps if step.node in ("LA", "RA") and step.accepted]
    left = sorted((step for step in ankle_steps if step.node_side == "left"), key=lambda step: (step.start_sample, step.end_sample, step.step_id))
    right = sorted((step for step in ankle_steps if step.node_side == "right"), key=lambda step: (step.start_sample, step.end_sample, step.step_id))
    rows, columns = len(left), len(right)
    scores = np.zeros((rows + 1, columns + 1), dtype=float)
    decisions = np.full((rows + 1, columns + 1), "", dtype=object)
    decisions[1:, 0], decisions[0, 1:] = "left", "right"
    priority = {"match": 2, "left": 1, "right": 0}
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            overlap = max(0, min(left[i - 1].end_sample, right[j - 1].end_sample) - max(left[i - 1].start_sample, right[j - 1].start_sample))
            options = [(scores[i - 1, j], "left"), (scores[i, j - 1], "right")]
            if overlap > 0:
                options.append((scores[i - 1, j - 1] + overlap, "match"))
            best = max(value for value, _ in options)
            scores[i, j], decisions[i, j] = max((item for item in options if item[0] == best), key=lambda item: priority[item[1]])
    matches: list[tuple[NodeStep, NodeStep]] = []
    i, j = rows, columns
    while i and j:
        if decisions[i, j] == "match":
            matches.append((left[i - 1], right[j - 1]))
            i, j = i - 1, j - 1
        elif decisions[i, j] == "left":
            i -= 1
        else:
            j -= 1
    matches.reverse()
    return [(f"{left_step.recording_id}:pair:{index:04d}", left_step, right_step) for index, (left_step, right_step) in enumerate(matches)]


def segment_signals(
    signals: dict[str, np.ndarray],
    fs: float,
    config: AnalysisConfig,
    *,
    recording_id: str = "synthetic",
    trial_id: str = "synthetic",
) -> SegmentationResult:
    """Detect ankle-driven gait cycles for both ankles and broadcast onto all 12 nodes."""
    settings = config.section("segmentation")
    minimum_filter_samples = max(32, int(settings["filter_order"]) * 6)
    diagnostics: dict[str, Any] = {}
    qc: list[str] = []
    steps: list[NodeStep] = []
    any_ankle_available = False
    for ankle_side, ankle_node in (("left", "LA"), ("right", "RA")):
        values = signals.get(ankle_node) if signals else None
        values = np.asarray(values, dtype=float) if values is not None else None
        if values is None or values.size < minimum_filter_samples or not np.any(np.isfinite(values)):
            qc.append(f"{ankle_side}:missing_ankle_proxy")
            diagnostics[f"{ankle_side}_cycle_count"] = 0
            continue
        any_ankle_available = True
        cycles = _detect_ankle_cycles(values, fs, settings)
        diagnostics[f"{ankle_side}_cycle_count"] = len(cycles)
        diagnostics[f"{ankle_side}_accepted_cycle_count"] = sum(cycle.accepted for cycle in cycles)
        for node in _nodes_driven_by(ankle_side):
            steps.extend(_cycles_to_node_steps(cycles, recording_id=recording_id, trial_id=trial_id, node=node, ankle_side=ankle_side))
    if not any_ankle_available:
        return SegmentationResult(recording_id, trial_id, [], fs, "excluded", ["missing_ankle_proxy_signals"], diagnostics)
    accepted = [step for step in steps if step.accepted]
    if not accepted:
        qc.append("no_accepted_cycles")
    diagnostics["accepted_step_count"] = len(accepted)
    diagnostics["matched_step_pair_count"] = len(pair_bilateral_steps(steps))
    return SegmentationResult(recording_id, trial_id, steps, fs, "ok" if accepted else "failed", qc, diagnostics)


def segment_recording(record: TrialRecord, config: AnalysisConfig) -> SegmentationResult:
    from .io import load_proxy_signals

    if not record.segmentation_eligible:
        return SegmentationResult(record.recording_id, record.trial_id, [], record.sampling_hz, "excluded", list(record.exclusion_reasons))
    proxy = load_proxy_signals(record, config)
    return segment_signals(
        proxy["signals"], float(proxy["fs"]), config, recording_id=record.recording_id, trial_id=record.trial_id,
    )
