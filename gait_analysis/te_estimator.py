"""Discrete transfer-entropy primitives shared by the toolkit and dashboard."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def quantile_discretize(values: np.ndarray, bins: int = 8) -> np.ndarray:
    """Map a continuous trace to deterministic marginal-quantile symbols."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if bins < 2:
        raise ValueError("bins must be at least two")
    if not np.all(np.isfinite(array)):
        raise ValueError("Transfer entropy does not accept non-finite samples")
    unique = np.unique(array)
    if unique.size <= bins:
        return np.searchsorted(unique, array).astype(np.int16)
    quantiles = np.quantile(array, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    edges = np.unique(quantiles)
    return np.searchsorted(edges, array, side="right").astype(np.int16)


def _conditional_mutual_information(y_future: np.ndarray, y_past: np.ndarray, x_past: np.ndarray, phase: np.ndarray | None, pseudocount: float) -> float:
    yf, yp, xp = map(lambda value: np.asarray(value, dtype=np.int64), (y_future, y_past, x_past))
    if not (len(yf) == len(yp) == len(xp)) or len(yf) == 0:
        raise ValueError("TE states must have equal non-zero lengths")
    z = np.zeros(len(yf), dtype=np.int64) if phase is None else np.asarray(phase, dtype=np.int64)
    by_future = int(yf.max(initial=0) + 1)
    by_past = int(yp.max(initial=0) + 1)
    bx = int(xp.max(initial=0) + 1)
    bz = int(z.max(initial=0) + 1)
    joint = np.zeros((by_future, by_past, bx, bz), dtype=float)
    np.add.at(joint, (yf, yp, xp, z), 1.0)
    if pseudocount > 0:
        joint += float(pseudocount)
    joint /= joint.sum()
    p_yp_xp_z = joint.sum(axis=0, keepdims=True)
    p_yf_yp_z = joint.sum(axis=2, keepdims=True)
    p_yp_z = joint.sum(axis=(0, 2), keepdims=True)
    valid = joint > 0
    ratio = np.divide(joint * p_yp_z, p_yp_xp_z * p_yf_yp_z, out=np.ones_like(joint), where=valid & (p_yp_xp_z > 0) & (p_yf_yp_z > 0))
    return float(np.sum(joint[valid] * np.log2(ratio[valid])))


def estimate_discrete_te(
    source: np.ndarray,
    target: np.ndarray,
    delay: int = 1,
    *,
    pseudocount: float = 1.0e-12,
    phase: np.ndarray | None = None,
    history_source: int = 1,
    history_target: int = 1,
) -> float:
    """Estimate discrete source-to-target TE at an explicit sample delay."""
    source = np.asarray(source).reshape(-1)
    target = np.asarray(target).reshape(-1)
    if source.size != target.size or delay < 1 or history_source < 1 or history_target < 1:
        raise ValueError("Source/target lengths or delay are invalid")
    start = max(int(history_target), int(delay) + int(history_source) - 1)
    if source.size <= start:
        raise ValueError("Signal is too short for the requested TE history")
    indices = np.arange(start, source.size)
    phase_state = None if phase is None else np.asarray(phase).reshape(-1)[indices]
    source_base = int(source.max(initial=0) + 1)
    target_base = int(target.max(initial=0) + 1)
    source_history = np.zeros(len(indices), dtype=np.int64)
    target_history = np.zeros(len(indices), dtype=np.int64)
    for history_index in range(history_source):
        source_history = source_history * source_base + source[indices - delay - history_index]
    for history_index in range(history_target):
        target_history = target_history * target_base + target[indices - 1 - history_index]
    return _conditional_mutual_information(
        target[indices], target_history, source_history, phase_state, pseudocount
    )


def maximize_te_over_delays(
    source: np.ndarray,
    target: np.ndarray,
    delays: Iterable[int],
    *,
    pseudocount: float = 1.0e-12,
    phase: np.ndarray | None = None,
    history_source: int = 1,
    history_target: int = 1,
) -> tuple[float, int, np.ndarray]:
    delays_array = np.asarray(list(delays), dtype=int)
    estimates = np.asarray([
        estimate_discrete_te(
            source, target, int(delay), pseudocount=pseudocount, phase=phase,
            history_source=history_source, history_target=history_target,
        )
        for delay in delays_array
    ])
    best = int(np.argmax(estimates))
    return float(estimates[best]), int(delays_array[best]), estimates


def valid_circular_shifts(
    length: int,
    near_zero: int,
    stride_samples: float,
    stride_multiple_fraction: float,
) -> np.ndarray:
    shifts = np.arange(1, length, dtype=int)
    circular_distance = np.minimum(shifts, length - shifts)
    valid = circular_distance > int(near_zero)
    if stride_samples > 0:
        remainder = np.mod(shifts, stride_samples)
        distance = np.minimum(remainder, stride_samples - remainder)
        valid &= distance > float(stride_multiple_fraction) * stride_samples
    return shifts[valid]
