"""Problem 2: correct inverted mode signals from the left/right pair relationship.

During gait the two limbs perform the same motion half a stride apart in
*time*. A correct pair therefore looks like one waveform and a delayed copy of
itself -- peaks and valleys arriving alternately, one side then the other --
and their phase portraits overlap. If exactly one side is inverted, that
alternation breaks: the delayed copy is negated rather than merely shifted.

The measurement is direct. Take the normalised cross-correlation between the
two sides and read it at a lag of **half a stride**:

* ``corr > 0`` -- the right side is a delayed *copy* of the left. Consistent.
* ``corr < 0`` -- the right side is a delayed *negated* copy. One of the two
  is inverted, and by convention the right side is the one negated.

This is a **relative** criterion, and deliberately so. Three earlier criteria
in this module all failed because they tried to judge a single signal on its
own: Hilbert-phase cycle-start timing (which conflates two 180-degree
contributions and inverted the verdict), canonicalising pairs to *opposite*
polarity, and requiring each signal's analytic-plane knot to sit at positive
values. The last was disproved by measurement -- among pairs that are already
consistent by the test above, the knot sits on the right for 29, on the left
for 48 and on opposite sides for 21. Knot position does not determine
correctness; the pair relationship does. ``analytic.knot_position`` survives as
a general-purpose utility but no longer takes part in any decision.

The pair test is also far more decisive than what it replaces: it settles ~98%
of ``mode_1`` pairs (median ``|corr|`` 0.87), including the arms, where the
per-signal asymmetry statistics decided only 29-50%.

Two things the accuracy of this test depends on:

* **The stride period**, which sets the lag being read. It comes once per
  recording from the whole-body reference (``segmentation.stride_period_samples``)
  rather than per signal, because a per-signal estimate sometimes locks onto
  twice the stride, which inverts the verdict.
* **Waveform asymmetry.** For a perfectly sinusoidal signal, inverting is
  indistinguishable from shifting by half a cycle, so the test degenerates.
  Real gait waveforms carry enough even-harmonic content to separate the two;
  where they do not, ``|corr|`` falls below the gate and the pair is reported
  ``low_confidence`` rather than guessed at.

When a side flips, its axis triplet ``{base}_mode_{k}_axis_{x,y,z}`` is negated
with it. The stored rotation is unchanged by this: ``*_original_angle`` is the
quaternion *composition* ``Rot(t3,u3) . Rot(t2,u2) . Rot(t1,u1)`` (see
``rss/model.py:442``), and ``Rot(t, u) == Rot(-t, -u)`` leaves each factor
identical. ``*_original_angle`` itself is a non-negative magnitude and is never
touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from hilbert_analysis import signal_utils
from hilbert_analysis.columns import PAIRED_LIMBS, axis_columns

WINDOW_PERIODIC = "periodic"
WINDOW_FULL_RECORD = "full_record"

SIDE_LEFT = "left"
SIDE_RIGHT = "right"

STATUS_CONSISTENT = "consistent"
STATUS_FLIPPED = "flipped"
STATUS_LOW_CONFIDENCE = "low_confidence"
STATUS_NO_STRIDE = "no_stride"
STATUS_DEGENERATE = "degenerate"


@dataclass(frozen=True)
class PhaseFlipConfig:
    #: Minimum |correlation| at the half-stride lag for a pair to be judged.
    min_abs_correlation: float = 0.30
    #: Lags are searched within this fraction of a stride either side of T/2.
    lag_search_fraction: float = 0.20
    min_stride_s: float = 0.4
    max_stride_s: float = 2.5
    min_peak_to_peak: float = 1e-3
    #: Modes judged and corrected. Each mode carries its own sign gauge, so a
    #: verdict on one says nothing about its siblings.
    modes: tuple[int, ...] = (1,)
    #: The side held fixed; the other is negated when a pair disagrees. The
    #: pair relationship cannot say *which* side is inverted, only that one is,
    #: so this is a convention rather than a measurement.
    reference_side: str = SIDE_LEFT
    allow_full_record_fallback: bool = True

    def validate(self) -> None:
        if not self.modes:
            raise ValueError("At least one mode must be selected.")
        if any(mode not in (1, 2, 3) for mode in self.modes):
            raise ValueError(f"Modes must be drawn from 1, 2, 3; got {self.modes}.")
        if self.reference_side not in (SIDE_LEFT, SIDE_RIGHT):
            raise ValueError("reference_side must be 'left' or 'right'.")
        if not 0.0 < self.lag_search_fraction < 0.5:
            raise ValueError("lag_search_fraction must lie in (0, 0.5).")


@dataclass(frozen=True)
class PairFlipResult:
    """One left/right pair at one mode: the verdict and the evidence for it."""

    limb: str
    mode: int
    left_column: str
    right_column: str
    correlation: float
    lag_samples: float
    #: ``lag / stride``; expected near 0.5 when the half-stride model holds.
    lag_fraction: float
    stride_samples: float
    stride_source: str
    status: str
    reason: str
    window_source: str = WINDOW_PERIODIC
    flipped_columns: tuple[str, ...] = ()
    flipped_axis_columns: tuple[str, ...] = ()
    secondary: dict[str, float] = field(default_factory=dict)

    @property
    def needs_flip(self) -> bool:
        return self.status == STATUS_FLIPPED


def half_stride_correlation(
    left: np.ndarray, right: np.ndarray, *, stride_samples: float, config: PhaseFlipConfig
) -> tuple[float, int]:
    """Strongest correlation near a half-stride lag, and the lag it occurred at.

    Both signs of lag are searched: which limb leads is arbitrary, and for a
    periodic signal ``corr(-T/2) ~= corr(+T/2)``, so this only adds robustness
    against a finite, non-stationary record.
    """
    half = int(round(stride_samples / 2.0))
    span = max(3, int(round(config.lag_search_fraction * stride_samples)))
    lags = np.concatenate([
        np.arange(half - span, half + span + 1),
        np.arange(-half - span, -half + span + 1),
    ])
    correlations = signal_utils.lagged_correlation(left, right, lags)
    if correlations.size == 0:
        return 0.0, half
    best = int(np.argmax(np.abs(correlations)))
    return float(correlations[best]), int(lags[best])


def evaluate_pair(
    left_values: np.ndarray,
    right_values: np.ndarray,
    *,
    limb: str,
    left_base: str,
    right_base: str,
    mode: int,
    stride_samples: float | None,
    stride_source: str,
    config: PhaseFlipConfig,
    window_source: str = WINDOW_PERIODIC,
) -> PairFlipResult:
    """Decide whether one left/right pair at one mode has an inverted side."""
    config.validate()
    left_column = f"{left_base}_mode_{mode}"
    right_column = f"{right_base}_mode_{mode}"

    def verdict(
        correlation: float, lag: float, status: str, reason: str,
        flipped: tuple[str, ...] = (), axes: tuple[str, ...] = (),
    ) -> PairFlipResult:
        stride = float(stride_samples) if stride_samples else float("nan")
        return PairFlipResult(
            limb=limb, mode=mode, left_column=left_column, right_column=right_column,
            correlation=correlation, lag_samples=lag,
            lag_fraction=(lag / stride) if stride_samples else float("nan"),
            stride_samples=stride, stride_source=stride_source,
            status=status, reason=reason, window_source=window_source,
            flipped_columns=flipped, flipped_axis_columns=axes,
        )

    left = signal_utils.interpolate_small_gaps(np.asarray(left_values, dtype=float))
    right = signal_utils.interpolate_small_gaps(np.asarray(right_values, dtype=float))
    if left.size < 16 or right.size < 16:
        return verdict(np.nan, np.nan, STATUS_DEGENERATE, "too_few_samples")
    if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
        return verdict(np.nan, np.nan, STATUS_DEGENERATE, "non_finite_samples")
    if np.ptp(left) < config.min_peak_to_peak or np.ptp(right) < config.min_peak_to_peak:
        return verdict(np.nan, np.nan, STATUS_DEGENERATE, "flat_signal")

    if not stride_samples:
        return verdict(np.nan, np.nan, STATUS_NO_STRIDE, "no_stride_period")

    correlation, lag = half_stride_correlation(
        left, right, stride_samples=stride_samples, config=config
    )
    if abs(correlation) < config.min_abs_correlation:
        return verdict(correlation, lag, STATUS_LOW_CONFIDENCE, "weak_correlation")
    if correlation > 0.0:
        return verdict(correlation, lag, STATUS_CONSISTENT, "delayed_copy")

    # Negated delayed copy: one side is inverted. Which one is not knowable
    # from the pair, so the configured reference side is held fixed.
    flipped_base = right_base if config.reference_side == SIDE_LEFT else left_base
    return verdict(
        correlation, lag, STATUS_FLIPPED, "delayed_negated_copy",
        flipped=(f"{flipped_base}_mode_{mode}",),
        axes=axis_columns(flipped_base, mode),
    )


def evaluate_frame(
    frame: pd.DataFrame,
    *,
    onset: int | None,
    offset: int | None,
    stride_samples: float | None,
    stride_source: str,
    config: PhaseFlipConfig,
    pairs: Sequence[tuple[str, str, str]] = PAIRED_LIMBS,
) -> list[PairFlipResult]:
    """Evaluate every paired limb present in ``frame``, for each selected mode."""
    config.validate()
    if onset is None or offset is None:
        if not config.allow_full_record_fallback:
            return []
        window, window_source = slice(None), WINDOW_FULL_RECORD
    else:
        window, window_source = slice(onset, offset + 1), WINDOW_PERIODIC

    results: list[PairFlipResult] = []
    for limb, left_base, right_base in pairs:
        for mode in config.modes:
            left_column = f"{left_base}_mode_{mode}"
            right_column = f"{right_base}_mode_{mode}"
            if left_column not in frame.columns or right_column not in frame.columns:
                continue
            left = pd.to_numeric(frame[left_column], errors="coerce").to_numpy(float)[window]
            right = pd.to_numeric(frame[right_column], errors="coerce").to_numpy(float)[window]
            results.append(
                evaluate_pair(
                    left, right, limb=limb, left_base=left_base, right_base=right_base,
                    mode=mode, stride_samples=stride_samples, stride_source=stride_source,
                    config=config, window_source=window_source,
                )
            )
    return results


def _flipped_columns(results: Iterable[PairFlipResult]) -> list[str]:
    return [column for result in results for column in result.flipped_columns]


def _flipped_axis_columns(results: Iterable[PairFlipResult]) -> list[str]:
    return [column for result in results for column in result.flipped_axis_columns]


def apply_flips(
    frame: pd.DataFrame, results: Sequence[PairFlipResult]
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Negate the mode angle of every side judged inverted.

    ``*_original_angle`` is never included: it is a non-negative magnitude
    carrying no sign information.
    """
    corrected = frame.copy()
    changed: list[str] = []
    for column in _flipped_columns(results):
        if column in corrected.columns:
            corrected[column] = -pd.to_numeric(corrected[column], errors="coerce")
            changed.append(column)
    return corrected, tuple(changed)


def apply_axis_flips(
    axes_frame: pd.DataFrame, results: Sequence[PairFlipResult]
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Negate the axis triplet of every flipped mode, keeping ``theta * u`` fixed."""
    corrected = axes_frame.copy()
    changed: list[str] = []
    for column in _flipped_axis_columns(results):
        if column in corrected.columns:
            corrected[column] = -pd.to_numeric(corrected[column], errors="coerce")
            changed.append(column)
    return corrected, tuple(changed)
