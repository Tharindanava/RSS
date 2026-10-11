"""Problem 1: detect recordings that end mid-motion instead of returning to
a quiet baseline, using a combined lower-limb short-time Hilbert envelope.

Whole-body activity is realized here as the *median*, across the 6
lower-limb paired-segment ``original_angle`` columns
(``columns.LOWER_LIMB_SEGMENTS``), of that column's own windowed envelope
expressed as a multiple of its own quiet-baseline level. Two choices were
forced by direct empirical testing against the labeled sample
recordings (not distributed; see ``tests/test_hilbert_analysis.py``):

* The envelope must be computed on short, non-overlapping windows
  (``analytic.windowed_envelope``), not the whole recording at once -- a
  single-shot Hilbert transform leaks energy far from a short burst.
* Combining across all 18 body segments (rather than just the 6 lower-limb
  ones) is numerically unstable: several upper-body/midline columns have
  near-zero motion during quiet stance, so their self-normalized activity
  ratio explodes from ordinary noise. Median combination (not mean) adds a
  further layer of robustness to any one column's outliers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hilbert_analysis import signal_utils
from hilbert_analysis.analytic import windowed_envelope
from hilbert_analysis.columns import LOWER_LIMB_SEGMENTS, original_angle_column

VERDICT_COMPLETE = "complete"
VERDICT_INCOMPLETE = "incomplete"
VERDICT_NO_PERIODIC_SEGMENT = "no_periodic_segment"


@dataclass(frozen=True)
class CompletenessConfig:
    threshold_multiplier: float = 3.0
    min_end_margin_s: float = 0.5
    chunk_window_s: float = 0.3
    baseline_window_s: float = 1.0
    # >= one full gait cycle: a shorter minimum lets brief pre-gait noise
    # blips (e.g. a single weight shift while standing) register as onset.
    min_sustained_s: float = 1.5
    min_baseline_floor: float = 1e-4


@dataclass(frozen=True)
class CompletenessResult:
    verdict: str
    onset_index: int | None
    offset_index: int | None
    onset_time_s: float | None
    offset_time_s: float | None
    end_margin_s: float | None
    whole_body_energy: np.ndarray
    chunk_window_samples: int


def compute_whole_body_energy(
    frame: pd.DataFrame,
    *,
    fs: float,
    config: CompletenessConfig,
) -> np.ndarray:
    """Combined lower-limb activity ratio, one value per non-overlapping
    ``chunk_window_s`` window (chunk resolution, not per-sample)."""
    chunk_window_samples = max(2, round(config.chunk_window_s * fs))
    baseline_chunks = max(1, round(config.baseline_window_s / config.chunk_window_s))

    ratios: list[np.ndarray] = []
    for segment in LOWER_LIMB_SEGMENTS:
        column = original_angle_column(segment)
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        filled = signal_utils.interpolate_small_gaps(values)
        if not np.all(np.isfinite(filled)):
            continue
        column_envelope = windowed_envelope(filled, chunk_window_samples, detrend="linear")
        baseline = max(
            float(np.median(column_envelope[:baseline_chunks])), config.min_baseline_floor
        )
        ratios.append(column_envelope / baseline)

    if not ratios:
        raise ValueError("No usable lower-limb original_angle columns to compute whole-body energy.")
    return np.median(np.vstack(ratios), axis=0)


def detect_completeness(
    frame: pd.DataFrame,
    *,
    fs: float,
    config: CompletenessConfig,
) -> CompletenessResult:
    n_rows = len(frame)
    chunk_window_samples = max(2, round(config.chunk_window_s * fs))
    energy = compute_whole_body_energy(frame, fs=fs, config=config)

    mask = energy > config.threshold_multiplier
    min_len_chunks = max(1, round(config.min_sustained_s / config.chunk_window_s))
    runs = signal_utils.find_sustained_runs(mask, min_len_chunks)
    if not runs:
        return CompletenessResult(
            VERDICT_NO_PERIODIC_SEGMENT, None, None, None, None, None, energy, chunk_window_samples
        )

    onset_chunk, offset_chunk = runs[0][0], runs[-1][1]
    onset_index = onset_chunk * chunk_window_samples
    offset_index = min(n_rows - 1, (offset_chunk + 1) * chunk_window_samples - 1)
    onset_time_s = onset_index / fs
    offset_time_s = offset_index / fs
    end_margin_s = (n_rows - 1 - offset_index) / fs
    verdict = VERDICT_INCOMPLETE if end_margin_s < config.min_end_margin_s else VERDICT_COMPLETE
    return CompletenessResult(
        verdict, onset_index, offset_index, onset_time_s, offset_time_s, end_margin_s,
        energy, chunk_window_samples,
    )
