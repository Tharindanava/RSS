"""Diagnostic overlays for spot-checking completeness verdicts before
trusting ``--apply`` on the full output tree."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from hilbert_analysis.completeness import CompletenessResult


def write_completeness_plot(
    completeness: CompletenessResult,
    *,
    fs: float,
    threshold_multiplier: float,
    title: str,
    output_path: Path,
) -> None:
    """Plot the whole-body activity ratio with its threshold/onset/offset markers.

    ``completeness.whole_body_energy`` is at chunk resolution (one value per
    ``completeness.chunk_window_samples``-sample window), not per-sample.
    """
    energy = completeness.whole_body_energy
    chunk_seconds = completeness.chunk_window_samples / fs
    time = np.arange(energy.size) * chunk_seconds

    figure, axis = plt.subplots(figsize=(11, 3.5))
    axis.plot(time, energy, color="#2563EB", linewidth=1.0, label="Whole-body activity ratio")
    axis.axhline(
        threshold_multiplier, color="#9CA3AF", linestyle=":", linewidth=1.0, label="Threshold"
    )
    if completeness.onset_time_s is not None:
        axis.axvline(completeness.onset_time_s, color="#16A34A", linestyle="--", label="Onset")
    if completeness.offset_time_s is not None:
        axis.axvline(completeness.offset_time_s, color="#DC2626", linestyle="--", label="Offset")
    axis.set_title(f"{title}\nverdict: {completeness.verdict}")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Activity ratio (multiples of quiet baseline)")
    axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
