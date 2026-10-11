"""Singular Spectrum Analysis drift removal and denoising for RSS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_DRIFT_ZERO_CROSSING_LIMIT = 2
DEFAULT_DRIFT_TURNING_POINT_LIMIT = 2


@dataclass(frozen=True)
class SSAComponentSummary:
    component_index: int
    eigenvalue: float
    variance_share_pct: float
    zero_crossings: int
    turning_points: int
    classification: str
    kept_in_reconstruction: bool


@dataclass(frozen=True)
class SSAPreprocessResult:
    raw_signal: np.ndarray
    drift_corrected_signal: np.ndarray
    removed_drift_signal: np.ndarray
    cleaned_signal: np.ndarray
    removed_signal: np.ndarray
    window_length: int
    component_signals: np.ndarray
    component_summaries: tuple[SSAComponentSummary, ...]
    drift_fallback_linear: bool


def _compact_sign(values: np.ndarray) -> np.ndarray:
    signs = np.sign(np.asarray(values, dtype=float).reshape(-1))
    return signs[signs != 0.0]


def _scale_tolerance(values: np.ndarray, eps: float) -> float:
    absolute = np.abs(np.asarray(values, dtype=float).reshape(-1))
    return float(eps) if absolute.size == 0 else max(float(eps), 0.01 * float(np.max(absolute)))


def _count_zero_crossings(values: np.ndarray, *, tolerance: float = 0.0) -> int:
    array = np.asarray(values, dtype=float).reshape(-1).copy()
    array[np.abs(array) <= float(tolerance)] = 0.0
    signs = _compact_sign(array)
    return 0 if signs.size < 2 else int(np.count_nonzero(signs[1:] != signs[:-1]))


def _count_turning_points(values: np.ndarray, *, tolerance: float = 0.0) -> int:
    differences = np.diff(np.asarray(values, dtype=float).reshape(-1))
    differences[np.abs(differences) <= float(tolerance)] = 0.0
    signs = _compact_sign(differences)
    return 0 if signs.size < 2 else int(np.count_nonzero(signs[1:] != signs[:-1]))


def _is_drift_component(
    *,
    zero_crossings: int,
    turning_points: int,
    zero_crossing_limit: int = DEFAULT_DRIFT_ZERO_CROSSING_LIMIT,
    turning_point_limit: int = DEFAULT_DRIFT_TURNING_POINT_LIMIT,
) -> bool:
    return zero_crossings < zero_crossing_limit or turning_points < turning_point_limit


def resolve_ssa_window_length(
    n_samples: int,
    *,
    sample_rate_hz: float,
    requested_window_length: int | None = None,
) -> int:
    if int(n_samples) < 4:
        raise ValueError("SSA preprocessing requires at least 4 samples.")
    requested = round(2.0 * sample_rate_hz) if requested_window_length is None else requested_window_length
    return int(min(max(2, int(requested)), int(n_samples) // 2))


def apply_ssa_preprocessing(
    signal: np.ndarray,
    *,
    sample_rate_hz: float,
    requested_window_length: int | None = None,
    requested_reconstruction_components: int | None = None,
    drift_zero_crossing_limit: int = DEFAULT_DRIFT_ZERO_CROSSING_LIMIT,
    drift_turning_point_limit: int = DEFAULT_DRIFT_TURNING_POINT_LIMIT,
    eps: float = 1e-12,
) -> SSAPreprocessResult:
    raw = np.asarray(signal, dtype=float).reshape(-1)
    if not np.isfinite(raw).all():
        raise ValueError("SSA preprocessing requires a finite input signal.")
    if requested_reconstruction_components is not None and requested_reconstruction_components < 1:
        raise ValueError("SSA reconstruction component count must be >= 1.")
    if drift_zero_crossing_limit < 1 or drift_turning_point_limit < 1:
        raise ValueError("SSA drift thresholds must be >= 1.")
    window = resolve_ssa_window_length(
        raw.size,
        sample_rate_hz=sample_rate_hz,
        requested_window_length=requested_window_length,
    )
    trajectory = np.asarray(np.lib.stride_tricks.sliding_window_view(raw, window).T, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(trajectory @ trajectory.T)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    counts = np.convolve(np.ones(window), np.ones(raw.size - window + 1))
    components: list[np.ndarray] = []
    crossings: list[int] = []
    turns: list[int] = []
    for index, eigenvalue in enumerate(eigenvalues):
        if eigenvalue <= eps:
            component = np.zeros_like(raw)
        else:
            left = eigenvectors[:, index]
            right = (trajectory.T @ left) / np.sqrt(eigenvalue)
            component = np.sqrt(eigenvalue) * np.convolve(left, right) / counts
        components.append(component)
        crossings.append(_count_zero_crossings(component, tolerance=_scale_tolerance(component, eps)))
        turns.append(_count_turning_points(component, tolerance=_scale_tolerance(np.diff(component), eps)))
    component_array = np.asarray(components)
    drift_mask = np.asarray(
        [
            _is_drift_component(
                zero_crossings=crossings[index],
                turning_points=turns[index],
                zero_crossing_limit=drift_zero_crossing_limit,
                turning_point_limit=drift_turning_point_limit,
            )
            for index in range(component_array.shape[0])
        ]
    )
    keep_mask = np.zeros(component_array.shape[0], dtype=bool)
    non_drift = np.flatnonzero(~drift_mask)
    keep_count = non_drift.size if requested_reconstruction_components is None else min(requested_reconstruction_components, non_drift.size)
    keep_mask[non_drift[:keep_count]] = True
    total = max(float(np.sum(eigenvalues)), float(eps))
    summaries = []
    for index in range(component_array.shape[0]):
        classification = "drift" if drift_mask[index] else ("kept" if keep_mask[index] else "noise")
        summaries.append(
            SSAComponentSummary(
                component_index=index + 1,
                eigenvalue=float(eigenvalues[index]),
                variance_share_pct=100.0 * float(eigenvalues[index]) / total,
                zero_crossings=crossings[index],
                turning_points=turns[index],
                classification=classification,
                kept_in_reconstruction=bool(keep_mask[index]),
            )
        )
    removed_drift = component_array[drift_mask].sum(axis=0) if np.any(drift_mask) else np.zeros_like(raw)
    drift_corrected = raw - removed_drift
    cleaned = drift_corrected.copy() if requested_reconstruction_components is None else (
        component_array[keep_mask].sum(axis=0) if np.any(keep_mask) else np.zeros_like(raw)
    )
    return SSAPreprocessResult(
        raw_signal=raw.copy(),
        drift_corrected_signal=drift_corrected,
        removed_drift_signal=removed_drift,
        cleaned_signal=cleaned,
        removed_signal=raw - cleaned,
        window_length=window,
        component_signals=component_array,
        component_summaries=tuple(summaries),
        drift_fallback_linear=False,
    )
