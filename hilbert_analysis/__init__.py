"""General-purpose Hilbert-transform toolkit, plus a gait-data cleaning
pipeline built on it (truncated-recording detection, left/right sign-flip
correction, and segment/cycle marker labeling for RSS-pipeline outputs)."""

from hilbert_analysis.analytic import (
    analytic_signal,
    cycle_intervals,
    cycle_start_indices,
    envelope,
    instantaneous_frequency,
    instantaneous_phase,
    windowed_envelope,
)
from hilbert_analysis.pipeline import BatchConfig, BatchResult, run_batch

__all__ = [
    "BatchConfig",
    "BatchResult",
    "analytic_signal",
    "cycle_intervals",
    "cycle_start_indices",
    "envelope",
    "instantaneous_frequency",
    "instantaneous_phase",
    "run_batch",
    "windowed_envelope",
]
