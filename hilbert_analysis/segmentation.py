"""Problem 3: ``segment_label`` / ``cycle_index`` marker columns.

``segment_label`` boundaries reuse Algorithm 1's whole-body-envelope onset
and offset. ``cycle_index`` needs a genuinely oscillating (signed) reference
rather than a rectified envelope, so it is built from a combined whole-body
*phase* reference:

    phase_ref(t) = sum over lower-limb segments of z-scored mode_1 signals,
                   each sign-normalised *in memory* to its side's target

The sign normalisation here is a computation on a copy -- it does not
reflect, and does not require, any particular polarity on disk. It exists
only so the segments reinforce instead of cancelling: the RSS sign gauge is
random per file/limb/mode, so raw summation would cancel arbitrarily.

Two corrections relative to an earlier version of this module:

* It used ``*_original_angle``, which is a single-signed magnitude that never
  crosses zero -- a poor phase reference.
* It combined the sides so that the fundamental cancelled. Left and right
  trace the same waveform half a gait cycle apart in time, so summing them
  as-stored beats at *step* rate (two peaks per stride) and doubles the cycle
  count. Normalising one side against the other before summing recovers the
  stride-rate fundamental: measured ``std = 1.66`` for the reinforcing
  combination versus ``1.11`` for the cancelling one, and per-trial cycle
  counts dropped from ~29 (steps) to ~16 (strides).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import skew as _scipy_skew

from hilbert_analysis import signal_utils
from hilbert_analysis.analytic import cycle_intervals, cycle_start_indices
from hilbert_analysis.columns import LOWER_LIMB_SEGMENTS, PAIRED_LIMBS

SEGMENT_START_TRANSIENT = "start_transient"
SEGMENT_PERIODIC = "periodic"
SEGMENT_END_TRANSIENT = "end_transient"
NO_CYCLE = -1


@dataclass(frozen=True)
class SegmentationResult:
    segment_label: np.ndarray
    cycle_index: np.ndarray
    n_cycles: int


def _zscored(frame: pd.DataFrame, column: str) -> np.ndarray | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    filled = signal_utils.interpolate_small_gaps(values)
    if not np.all(np.isfinite(filled)):
        return None
    std = float(np.std(filled))
    if std <= 0.0:
        return None
    return (filled - float(np.mean(filled))) / std


def build_whole_body_phase_reference(
    frame: pd.DataFrame,
    *,
    segments: tuple[str, ...] = LOWER_LIMB_SEGMENTS,
    onset: int | None = None,
    offset: int | None = None,
) -> np.ndarray | None:
    """Combined signed whole-body reference for cycle-boundary detection.

    Sums the z-scored ``mode_1`` signals of the lower-limb segments after
    sign-normalising each one in memory (by its own skew sign) so that
    contributions reinforce instead of cancelling. Independent of whatever
    sign gauge the stored data happens to carry. Returns ``None`` if no
    usable column is present.
    """
    window = slice(onset, (offset + 1) if offset is not None else None)
    target_by_base: dict[str, int] = {}
    for _, left_base, right_base in PAIRED_LIMBS:
        target_by_base[left_base] = 1
        target_by_base[right_base] = -1

    stacked: list[np.ndarray] = []
    for segment in segments:
        target = target_by_base.get(segment)
        if target is None:
            continue
        z_scored = _zscored(frame, f"{segment}_mode_1")
        if z_scored is None:
            continue
        polarity = float(_scipy_skew(z_scored[window]))
        if not np.isfinite(polarity) or polarity == 0.0:
            continue
        # In-memory sign normalisation only (never written back): opposing
        # the two sides makes the half-cycle-offset contributions reinforce
        # at stride rate. Normalising every segment to the *same* sign would
        # instead make them interleave, doubling the apparent cycle rate.
        stacked.append(z_scored if np.sign(polarity) == target else -z_scored)
    if not stacked:
        return None
    return np.sum(np.vstack(stacked), axis=0)


STRIDE_FROM_WHOLE_BODY = "whole_body"
STRIDE_FROM_SIGNAL = "per_signal"
STRIDE_UNAVAILABLE = "unavailable"


def _period_from_reference(
    reference: np.ndarray, *, fs: float, min_stride_s: float, max_stride_s: float
) -> float | None:
    try:
        starts = cycle_start_indices(np.asarray(reference, dtype=float))
    except ValueError:
        return None
    if len(starts) < 3:
        return None
    period = float(np.median(np.diff(starts)))
    if not (min_stride_s * fs <= period <= max_stride_s * fs):
        return None
    return period


def stride_period_samples(
    frame: pd.DataFrame,
    *,
    onset: int | None,
    offset: int | None,
    fs: float,
    fallback_signal: np.ndarray | None = None,
    min_stride_s: float = 0.4,
    max_stride_s: float = 2.5,
    allow_full_record: bool = True,
) -> tuple[float | None, str]:
    """One stride period (in samples) for a whole recording, plus its source.

    Estimated from the combined whole-body phase reference, which is far more
    stable than any single signal: a per-signal autocorrelation disagrees with
    this estimate by more than 25% on 14% of signals and occasionally locks
    onto *twice* the stride. That failure is not benign here -- the half-stride
    lag is what the left/right polarity test is measured at, so a doubled
    period inverts the verdict.

    Falls back to cycle detection on ``fallback_signal`` when the whole-body
    reference is unusable. Returns ``(None, "unavailable")`` if neither works.
    """
    if onset is None or offset is None:
        # No periodic segment was located. The pair test still falls back to
        # the full record, so estimate the stride over the same span rather
        # than declining -- the sanity window below is what guards the result.
        if not allow_full_record:
            return None, STRIDE_UNAVAILABLE
        window = slice(None)
    else:
        window = slice(onset, offset + 1)

    reference = build_whole_body_phase_reference(frame, onset=onset, offset=offset)
    if reference is not None:
        period = _period_from_reference(
            np.asarray(reference, dtype=float)[window],
            fs=fs, min_stride_s=min_stride_s, max_stride_s=max_stride_s,
        )
        if period is not None:
            return period, STRIDE_FROM_WHOLE_BODY

    if fallback_signal is not None:
        values = np.asarray(fallback_signal, dtype=float)
        if np.all(np.isfinite(values)):
            period = _period_from_reference(
                values, fs=fs, min_stride_s=min_stride_s, max_stride_s=max_stride_s
            )
            if period is not None:
                return period, STRIDE_FROM_SIGNAL

    return None, STRIDE_UNAVAILABLE


def build_markers(
    n_rows: int,
    *,
    onset: int | None,
    offset: int | None,
    phase_reference: np.ndarray | None,
) -> SegmentationResult:
    segment_label = np.full(n_rows, SEGMENT_START_TRANSIENT, dtype=object)
    cycle_index = np.full(n_rows, NO_CYCLE, dtype=int)

    if onset is None or offset is None:
        return SegmentationResult(segment_label, cycle_index, 0)

    segment_label[onset : offset + 1] = SEGMENT_PERIODIC
    segment_label[offset + 1 :] = SEGMENT_END_TRANSIENT

    if phase_reference is None:
        return SegmentationResult(segment_label, cycle_index, 0)

    window = np.asarray(phase_reference, dtype=float)[onset : offset + 1]
    if window.size < 2 or not np.all(np.isfinite(window)):
        return SegmentationResult(segment_label, cycle_index, 0)

    try:
        starts = cycle_start_indices(window)
    except ValueError:
        return SegmentationResult(segment_label, cycle_index, 0)

    intervals = cycle_intervals(starts)
    for cycle_number, (start, end) in enumerate(intervals, start=1):
        cycle_index[onset + start : onset + end] = cycle_number

    return SegmentationResult(segment_label, cycle_index, len(intervals))
