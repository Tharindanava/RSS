"""Build per-trial merged CSVs (raw multi-IMU window + ground truth) for the QArm recording dates.

Each dated recording folder (e.g. ``session_2026-09-09``) holds seven batch
subfolders (``1``..``7``), each containing one ``combined_log_wide_*.csv`` with five
back-to-back movement regions. This script splits each batch into its five regions,
aligns each region against the matching ``Ground_Truth/results/test_N.csv`` (using the
same movement-detection and correlation-alignment approach as
``qarm_rss_experiments.practical_validation.reproduce_alignment_plans``), and writes
one merged ``Merged/test_N.csv`` per trial (35 per date).

The IMU node columns differ between the two dates (Node-1/3/4/7 on 2026-09-09 vs.
Node-2/6/7/10 on 2026-09-18), so the gyroscope-activity signal used for movement
detection is built generically from whichever ``OrbiX-Node-*_IMU1_G{X,Y,Z}`` columns
are present in a given batch file, rather than a fixed node list.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from qarm_rss_experiments.practical_validation import (
    _centered_moving_average,
    _measured_joint_speed,
    _movement_regions,
    _pearson,
)

SAMPLE_RATE_HZ = 100.0
DATA_ROOT = Path(__file__).resolve().parent.parent / "Sample_Data" / "QArm"
DATE_FOLDERS = ("session_2026-09-09", "session_2026-09-18")

_GYRO_COLUMN = re.compile(r"^(OrbiX-Node-\d+_IMU1)_G[XYZ]$")


def _gyro_activity(frame: pd.DataFrame) -> np.ndarray:
    """Max gyro-norm activity across whichever IMU nodes are present in ``frame``."""
    prefixes = sorted({match.group(1) for column in frame.columns if (match := _GYRO_COLUMN.match(column))})
    groups = []
    for prefix in prefixes:
        columns = [f"{prefix}_G{axis}" for axis in "XYZ"]
        if all(column in frame for column in columns):
            groups.append(np.linalg.norm(frame[columns].to_numpy(dtype=float), axis=1))
    if not groups:
        raise ValueError("No complete gyroscope triplets were found.")
    return np.max(np.stack(groups), axis=0)


def build_merged_trials(date_root: Path, sample_rate_hz: float = SAMPLE_RATE_HZ) -> list[dict]:
    truth_root = date_root / "Ground_Truth" / "results"
    merged_root = date_root / "Merged"
    merged_root.mkdir(parents=True, exist_ok=True)

    reports = []
    batch_dirs = sorted((p for p in date_root.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    for batch_dir in batch_dirs:
        batch = int(batch_dir.name)
        raw_paths = list(batch_dir.glob("combined_log_wide_*.csv"))
        if len(raw_paths) != 1:
            raise ValueError(f"Expected one combined IMU file in {batch_dir}; found {len(raw_paths)}.")
        raw = pd.read_csv(raw_paths[0])
        activity = _gyro_activity(raw)
        regions = _movement_regions(activity, sample_rate_hz)
        if len(regions) != 5:
            raise ValueError(f"Expected five movement regions in {batch_dir}; found {len(regions)}.")
        smooth_activity = _centered_moving_average(activity, round(0.21 * sample_rate_hz))

        for local_index, region in enumerate(regions, 1):
            number = (batch - 1) * 5 + local_index
            truth_path = truth_root / f"test_{number}.csv"
            truth = pd.read_csv(truth_path, float_precision="round_trip")

            speed = _centered_moving_average(_measured_joint_speed(truth), round(0.21 * sample_rate_hz))
            first = max(0, region[0] - round(1.5 * sample_rate_hz))
            last = min(len(raw) - len(truth), region[0] + round(1.5 * sample_rate_hz))
            candidates = [
                (_pearson(smooth_activity[start : start + len(truth)], speed), start)
                for start in range(first, last + 1)
            ]
            correlation, aligned = max(candidates)

            window = raw.iloc[aligned : aligned + len(truth)].reset_index(drop=True)
            merged = pd.concat([window, truth.reset_index(drop=True)], axis=1)
            merged_path = merged_root / f"test_{number}.csv"
            merged.to_csv(merged_path, index=False)

            reports.append({
                "date": date_root.name,
                "batch": batch,
                "test_number": number,
                "aligned_start": aligned,
                "coarse_start": region[0],
                "alignment_correlation": correlation,
                "row_count": len(merged),
                "merged_path": str(merged_path),
            })
    return reports


def main() -> None:
    all_reports = []
    for date_folder in DATE_FOLDERS:
        date_root = DATA_ROOT / date_folder
        reports = build_merged_trials(date_root)
        all_reports.extend(reports)
        low_quality = [r for r in reports if r["alignment_correlation"] < 0.85]
        print(f"{date_folder}: wrote {len(reports)} merged trials to {date_root / 'Merged'}")
        if low_quality:
            print(f"  WARNING: {len(low_quality)} trial(s) below 0.85 alignment correlation:")
            for r in low_quality:
                print(f"    test_{r['test_number']}: correlation={r['alignment_correlation']:.3f}")

    print(f"\nTotal merged trials written: {len(all_reports)}")
    correlations = [r["alignment_correlation"] for r in all_reports]
    print(f"Alignment correlation: min={min(correlations):.3f} mean={np.mean(correlations):.3f}")


if __name__ == "__main__":
    main()
