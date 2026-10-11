"""Non-Hilbert signal helpers shared by the completeness, phase-flip, and
segmentation modules. Kept separate from ``analytic.py`` so that module
stays a minimal, general-purpose Hilbert-transform core."""

from __future__ import annotations

import numpy as np
import pandas as pd


def infer_sampling_rate(
    time_seconds: "pd.Series | np.ndarray", *, default: float = 100.0
) -> float:
    """Infer the sampling rate (Hz) from a monotonically increasing time column."""
    values = np.asarray(time_seconds, dtype=float).reshape(-1)
    if values.size < 2:
        return float(default)
    diffs = np.diff(values)
    finite = diffs[np.isfinite(diffs) & (diffs > 0)]
    if finite.size == 0:
        return float(default)
    return float(1.0 / np.median(finite))


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Edge-padded centered moving average. ``window`` is forced odd and >= 1."""
    values = np.asarray(x, dtype=float).reshape(-1)
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window == 1 or values.size == 0:
        return values.copy()
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def find_sustained_runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Contiguous True-runs in ``mask`` at least ``min_len`` samples long.

    Returns inclusive ``(start, end)`` index pairs.
    """
    values = np.asarray(mask, dtype=bool).reshape(-1)
    min_len = max(1, int(min_len))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(values):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= min_len:
                runs.append((start, index - 1))
            start = None
    if start is not None and values.size - start >= min_len:
        runs.append((start, values.size - 1))
    return runs


def lagged_correlation(a: np.ndarray, b: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Pearson correlation of ``a(t)`` against ``b(t + lag)``, one value per lag.

    Each lag is scored on the **overlap only**, re-centred and re-scaled there.
    An earlier version of this repo divided a single full ``np.correlate`` by
    the array length, which biases every non-zero lag toward zero in
    proportion to the lag -- harmless when only comparing which of two
    extremes is larger, but wrong when the value *at* a particular lag is the
    measurement, as it is here.

    Returns ``0.0`` for any lag whose overlap is shorter than 16 samples or
    where either side is constant.
    """
    first = np.asarray(a, dtype=float).reshape(-1)
    second = np.asarray(b, dtype=float).reshape(-1)
    out = np.zeros(len(lags), dtype=float)
    for index, lag in enumerate(np.asarray(lags, dtype=int)):
        if lag >= 0:
            left, right = first[: first.size - lag], second[lag:]
        else:
            left, right = first[-lag:], second[: second.size + lag]
        size = min(left.size, right.size)
        if size < 16:
            continue
        left, right = left[:size], right[:size]
        if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
            continue
        left_std, right_std = float(np.std(left)), float(np.std(right))
        if left_std <= 0.0 or right_std <= 0.0:
            continue
        centred = (left - np.mean(left)) * (right - np.mean(right))
        out[index] = float(np.mean(centred) / (left_std * right_std))
    return out


def interpolate_small_gaps(x: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """Linearly interpolate interior NaN runs up to ``max_gap`` samples long.

    Longer gaps, and any gap touching either end of the array, are left as
    NaN. Never mutates the caller's array; used only to feed internal
    Hilbert-transform math, never written back into an output CSV.
    """
    values = np.asarray(x, dtype=float).copy()
    is_nan = ~np.isfinite(values)
    if not is_nan.any():
        return values

    n = values.size
    start: int | None = None
    for index in range(n):
        if is_nan[index] and start is None:
            start = index
        elif not is_nan[index] and start is not None:
            gap_len = index - start
            if gap_len <= max_gap and start > 0:
                values[start:index] = np.interp(
                    np.arange(start, index),
                    [start - 1, index],
                    [values[start - 1], values[index]],
                )
            start = None
    # A gap still open at the end of the array touches EOF and is left as NaN.
    return values
