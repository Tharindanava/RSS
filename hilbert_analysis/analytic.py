"""General-purpose Hilbert-transform / analytic-signal utilities.

This module is intentionally reusable beyond the gait-data cleaning task it
was originally written for: analytic signal, instantaneous phase, envelope,
instantaneous frequency, and Hilbert-phase cycle-boundary detection, all
operating on plain 1-D ``numpy`` arrays.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import detrend as _scipy_detrend
from scipy.signal import hilbert


def _prepare(x: np.ndarray, *, detrend: str | None) -> np.ndarray:
    values = np.asarray(x, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("Input signal must not be empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Input signal contains non-finite values. Callers must handle "
            "NaN/inf before calling into analytic_signal: a general-purpose "
            "transform must not silently change the output length by "
            "dropping samples."
        )
    if detrend is None:
        return values
    if detrend == "mean":
        return values - float(np.mean(values))
    if detrend == "linear":
        return np.asarray(_scipy_detrend(values, type="linear"), dtype=float)
    raise ValueError(f"Unsupported detrend option: {detrend!r}")


def analytic_signal(x: np.ndarray, *, detrend: str | None = "mean") -> np.ndarray:
    """Return the complex analytic signal via the Hilbert transform.

    ``detrend`` defaults to ``"mean"`` (subtract the mean before
    transforming), matching this repo's established convention
    (``HMA_Dashboard/plotting.py::_compute_hilbert``). Pass ``"linear"`` for
    a linear detrend, or ``None`` to skip detrending entirely.
    """
    prepared = _prepare(x, detrend=detrend)
    return np.asarray(hilbert(prepared))


def instantaneous_phase(
    x: np.ndarray, *, unwrap: bool = False, detrend: str | None = "mean"
) -> np.ndarray:
    """Instantaneous phase (radians, in ``(-pi, pi]`` unless unwrapped)."""
    phase = np.angle(analytic_signal(x, detrend=detrend))
    return np.unwrap(phase) if unwrap else phase


def envelope(x: np.ndarray, *, detrend: str | None = "mean") -> np.ndarray:
    """Instantaneous amplitude envelope of the analytic signal."""
    return np.abs(analytic_signal(x, detrend=detrend))


def instantaneous_frequency(
    x: np.ndarray, fs: float, *, detrend: str | None = "mean"
) -> np.ndarray:
    """Instantaneous frequency (Hz), length ``len(x) - 1``.

    Computed as the sample-rate-scaled derivative of unwrapped phase.
    Deliberately not padded back to the input length so callers cannot
    mistake it for a per-sample-aligned series without noticing the
    off-by-one.
    """
    if fs <= 0:
        raise ValueError("Sampling rate must be positive.")
    phase = instantaneous_phase(x, unwrap=True, detrend=detrend)
    return np.diff(phase) * fs / (2.0 * np.pi)


def cycle_start_indices(x: np.ndarray, *, detrend: str | None = "mean") -> list[int]:
    """Indices where instantaneous phase wraps from negative to non-negative.

    Package-local on purpose: the analysis packages in this repository
    (``hilbert_analysis``, ``limb_transfer_entropy``, ``rotating_poincare``) are
    self-contained and do not import one another.
    """
    phase = instantaneous_phase(x, detrend=detrend)
    return [
        index
        for index in range(1, phase.size)
        if phase[index - 1] < 0.0 <= phase[index]
    ]


def windowed_envelope(
    x: np.ndarray, window_samples: int, *, detrend: str | None = "linear"
) -> np.ndarray:
    """Mean Hilbert-envelope magnitude in each non-overlapping window.

    Computing the analytic signal over an entire long recording at once can
    leak substantial envelope energy far from a short burst, since the
    Hilbert kernel decays only as ``1/t`` rather than exponentially --
    unsuitable for detecting *where* a mostly-quiet signal is active.
    Splitting the signal into short windows and transforming each
    independently keeps the estimate local. ``detrend="linear"`` (the
    default here, unlike the other functions in this module) also removes
    within-window drift/ramp so a slowly settling but non-oscillating tail
    is not mistaken for activity.

    Returns one scalar per window (length ``ceil(len(x) / window_samples)``);
    the last window may be shorter than the rest. A window with fewer than
    4 finite samples repeats the previous window's value (or 0.0 for the
    first window) rather than raising.
    """
    values = np.asarray(x, dtype=float).reshape(-1)
    window_samples = max(2, int(window_samples))
    n = values.size
    n_windows = int(np.ceil(n / window_samples)) if n else 0
    out = np.empty(n_windows, dtype=float)
    for index in range(n_windows):
        start = index * window_samples
        end = min(n, start + window_samples)
        segment = values[start:end]
        if segment.size < 4 or not np.all(np.isfinite(segment)):
            out[index] = out[index - 1] if index > 0 else 0.0
            continue
        out[index] = float(np.mean(envelope(segment, detrend=detrend)))
    return out


def cycle_intervals(starts: list[int]) -> list[tuple[int, int]]:
    """Pair consecutive cycle-start indices into half-open ``[start, end)`` intervals."""
    return [
        (start, end)
        for start, end in zip(starts[:-1], starts[1:])
        if int(end) > int(start)
    ]


def knot_position(
    x: np.ndarray, *, radius_fraction: float = 0.9, detrend: str | None = "mean"
) -> float:
    """Where the analytic trajectory's inner tangle -- its "knot" -- sits on the signal axis.

    Plot the analytic signal as ``(x, Hilbert(x))`` and a periodic waveform
    traces a loop per cycle. An asymmetric waveform -- one large excursion per
    cycle and a low-amplitude remainder -- draws one wide loop plus a tight
    tangle of small loops where the trajectory passes close to the origin.
    That tangle is the knot, and its position along the signal axis says which
    way the waveform's large excursion points: a knot at positive signal values
    means the large excursion is negative.

    **Knot position does not indicate whether a signal is correctly oriented.**
    This was tried as a flip criterion and disproved by measurement: among
    left/right pairs that are correctly related (the same waveform half a
    stride apart), the knot sits on the positive side for 29, on the negative
    side for 48, and on opposite sides for 21. Orientation is a property of the
    *pair*, not of one signal -- see ``phase_flip``. This function is retained
    as a descriptive analytic-plane measurement only.

    The knot is taken as the median signal value over the samples whose
    analytic radius (the Hilbert envelope) is below ``radius_fraction`` of the
    median radius, normalised by the signal's 95th-percentile amplitude so the
    result is dimensionless and comparable across signals. Returns a value in
    roughly ``[-1, 1]``; ``0.0`` when the trajectory has no distinguishable
    inner region.

    A radius threshold *relative to the loop size* is deliberate: thresholding
    on a fixed sample quantile instead makes the estimate depend on how long
    the signal dwells near the origin, which varies with waveform shape. Even
    so, the knot is intrinsically harder to localise than the tail asymmetry
    it corroborates -- treat a small magnitude as "no reading", not as
    evidence of symmetry.
    """
    values = _prepare(np.asarray(x, dtype=float).reshape(-1), detrend=detrend)
    if values.size < 8:
        return 0.0
    radius = np.abs(hilbert(values))
    median_radius = float(np.median(radius))
    amplitude = float(np.percentile(np.abs(values), 95))
    if median_radius <= 0.0 or amplitude <= 0.0:
        return 0.0
    inner = values[radius <= radius_fraction * median_radius]
    if inner.size < 8:
        return 0.0
    return float(np.median(inner) / amplitude)
