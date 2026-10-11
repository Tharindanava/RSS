#!/usr/bin/env python3
"""Generate a small, fully synthetic RSS-style output tree for trying the analysis tools.

No human data is involved. For every body segment the script writes gait-like periodic
``mode_1/2/3`` angle signals in the same layout the RSS real-data pipeline produces::

    <output-root>/Human_Motion/<subject>/<terrain>/<experiment>/<stem>/limb_angle_mode.csv
                                                                          limb_axes_mode.csv

so that ``HMA_Dashboard``, ``hilbert_analysis``, ``rotating_poincare`` and
``limb_transfer_entropy`` can be exercised end to end.

The signal model is deliberately simple:

* every segment follows ``a1*sin(phi + p) + a2*sin(2*phi + q)`` with ``phi = 2*pi*t/stride``;
* the right side of a pair is the left waveform delayed by half a stride, which is exactly
  the "one waveform plus a delayed copy" structure the left/right pair test looks for;
* modes 2 and 3 are low-amplitude coloured noise;
* each mode's sign is flipped independently with probability ``--flip-probability`` (together
  with its axis vector), mimicking the free sign gauge of RSS outputs.

``*_original_angle`` is written as the norm of the three modes. In real RSS outputs it is the
angle of the composed quaternion rotation; the norm is only a stand-in with the same sign and
scale behaviour.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hilbert_analysis.columns import ALL_SEGMENTS, axis_columns, mode_columns, original_angle_column

SAMPLE_RATE_HZ = 100.0


def _coloured_noise(rng: np.random.Generator, n: int, alpha: float = 0.97) -> np.ndarray:
    white = rng.standard_normal(n)
    out = np.empty(n)
    out[0] = white[0]
    for i in range(1, n):
        out[i] = alpha * out[i - 1] + (1.0 - alpha) * white[i]
    return out / max(float(np.std(out)), 1e-12)


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    vector = rng.standard_normal(3)
    return vector / np.linalg.norm(vector)


def synthesize_recording(
    *,
    seconds: float,
    stride_seconds: float,
    flip_probability: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = int(round(seconds * SAMPLE_RATE_HZ))
    t = np.arange(n) / SAMPLE_RATE_HZ
    # Quiet lead-in and tail so the Hilbert completeness check sees a clean start and end.
    lead = int(1.5 * SAMPLE_RATE_HZ)
    envelope = np.zeros(n)
    envelope[lead : n - lead] = 1.0
    envelope = np.convolve(envelope, np.hanning(41) / np.hanning(41).sum(), mode="same")

    angle = {"frame": np.arange(n), "time": t}
    axes = {"frame": np.arange(n), "time": t}
    for segment in ALL_SEGMENTS:
        is_right = segment.startswith("right_") or "_right_" in segment
        # Shared parameters per limb so a left/right pair has the same waveform.
        limb_key = segment.replace("left_", "").replace("right_", "").replace("_left_", "_").replace("_right_", "_")
        limb_rng = np.random.default_rng(sum(map(ord, limb_key)))
        a1, a2 = limb_rng.uniform(0.15, 0.55), limb_rng.uniform(0.04, 0.20)
        p, q = limb_rng.uniform(0, 2 * np.pi), limb_rng.uniform(0, 2 * np.pi)
        delay = 0.5 * stride_seconds if is_right else 0.0
        phi = 2.0 * np.pi * (t - delay) / stride_seconds
        mode_1 = envelope * (a1 * np.sin(phi + p) + a2 * np.sin(2.0 * phi + q))
        modes = [
            mode_1 + 0.01 * rng.standard_normal(n),
            0.05 * envelope * _coloured_noise(rng, n),
            0.03 * envelope * _coloured_noise(rng, n),
        ]
        stacked = []
        for index, (column, signal) in enumerate(zip(mode_columns(segment), modes), start=1):
            sign = -1.0 if rng.random() < flip_probability else 1.0
            angle[column] = sign * signal
            axis = sign * _unit_vector(rng)
            for name, value in zip(axis_columns(segment, index), axis):
                axes[name] = np.full(n, value)
            stacked.append(signal)
        angle[original_angle_column(segment)] = np.linalg.norm(np.vstack(stacked), axis=0)
    ordered = ["frame", "time"] + [c for c in angle if c not in ("frame", "time")]
    return pd.DataFrame(angle)[ordered], pd.DataFrame(axes)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/rss/Demo_Synthetic"))
    parser.add_argument("--subject", default="Subject_01")
    parser.add_argument("--terrain", default="Flat terrain")
    parser.add_argument("--trials", type=int, default=2, help="Number of synthetic recordings to write.")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--stride-seconds", type=float, default=1.1)
    parser.add_argument("--flip-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    for trial in range(1, args.trials + 1):
        stamp = 1700000000 + 600 * trial
        experiment = f"{stamp}_01_01_2026_Flat_0,0_40Steps_Demo_Trial_{trial}"
        leaf = args.output_root / "Human_Motion" / args.subject / args.terrain / experiment / f"combined_log_wide_{stamp}"
        leaf.mkdir(parents=True, exist_ok=True)
        angles, axes = synthesize_recording(
            seconds=args.seconds,
            stride_seconds=args.stride_seconds * (1.0 + 0.03 * (trial - 1)),
            flip_probability=args.flip_probability,
            seed=args.seed + trial,
        )
        angles.to_csv(leaf / "limb_angle_mode.csv", index=False)
        axes.to_csv(leaf / "limb_axes_mode.csv", index=False)
        print(f"wrote {leaf}")


if __name__ == "__main__":
    main()
