"""Split each batch's RSS-recovered sources into its five trials and pair them with ground truth.

Each ``batch_N_rss_sources.csv`` under ``outputs/rss/QArm_PSO_SO3_2000Epochs/<date>/``
holds one RSS fit over an entire combined recording (five back-to-back movement
regions separated by steady, near-zero-motion segments), row-aligned one-to-one
with the ``combined_log_wide_*.csv`` batch file it was fit from. This script
reuses the exact movement-detection and correlation-alignment plan that
``qarm_rss_experiments.build_merged_trials`` used to build ``Merged/test_N.csv``,
applies it to the recovered-source file instead of the raw IMU file, and writes
one ``matched/test_N.csv`` per trial pairing the recovered sources with the
corresponding ``Ground_Truth/results/test_N.csv`` row range.

The ``target_*`` columns are dropped (only measured joint angles are kept as
ground truth), and each ``source_N`` column is renamed to ``source_<joint>``
by one-to-one assignment against ``measured_base_rad`` / ``measured_elbow_rad``
/ ``measured_wrist_rad``, maximizing summed absolute Pearson correlation
(``scipy.optimize.linear_sum_assignment``, matching
``qarm_rss_experiments.practical_validation.match_and_score``). RSS mode signs
are an unresolved gauge freedom, so a renamed source is negated whenever its
correlation with the matched joint is negative (equivalent to checking whether
its peaks land on that joint's peaks or its troughs); only sign is corrected,
scale is left as RSS recovered it.

A batch with no recovered-source file yet (RSS has not been fit for it) is
skipped with a warning rather than raising, since not every batch has been
processed for every recording date.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from qarm_rss_experiments.build_merged_trials import _gyro_activity
from qarm_rss_experiments.practical_validation import (
    JOINT_NAMES,
    TRUTH_COLUMNS,
    _centered_moving_average,
    _measured_joint_speed,
    _movement_regions,
    _pearson,
)

SAMPLE_RATE_HZ = 100.0
DATA_ROOT = Path(__file__).resolve().parent.parent / "Sample_Data" / "QArm"
RSS_SOURCES_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "rss" / "QArm_PSO_SO3_2000Epochs"
DATE_FOLDERS = ("session_2026-09-09", "session_2026-09-18")
SOURCE_COLUMNS = ("source_1", "source_2", "source_3")


def _rename_and_unflip_sources(matched: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    """Drop target columns, rename each source to its best-matched joint, and correct sign flips."""

    truth_matrix = matched[list(TRUTH_COLUMNS)].to_numpy(dtype=float).T
    source_matrix = matched[list(SOURCE_COLUMNS)].to_numpy(dtype=float).T
    correlations = np.asarray([[_pearson(truth_matrix[i], source_matrix[j]) for j in range(3)] for i in range(3)])
    joint_indices, source_indices = linear_sum_assignment(-np.abs(correlations))

    rename_map = {SOURCE_COLUMNS[source_indices[k]]: f"source_{JOINT_NAMES[joint_indices[k]]}" for k in range(3)}
    matched = matched.drop(columns=[column for column in matched.columns if column.startswith("target_")])
    matched = matched.rename(columns=rename_map)

    matched_correlation: dict[str, float] = {}
    flipped_joints: list[str] = []
    for k in range(3):
        joint = JOINT_NAMES[joint_indices[k]]
        raw_correlation = float(correlations[joint_indices[k], source_indices[k]])
        column = f"source_{joint}"
        if raw_correlation < 0.0:
            matched[column] = -matched[column]
            flipped_joints.append(joint)
        matched_correlation[joint] = abs(raw_correlation)
    return matched, matched_correlation, flipped_joints


def build_matched_sources(
    date_root: Path,
    rss_sources_root: Path,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> list[dict]:
    truth_root = date_root / "Ground_Truth" / "results"
    matched_root = rss_sources_root / "matched"
    matched_root.mkdir(parents=True, exist_ok=True)

    reports = []
    batch_dirs = sorted((p for p in date_root.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    for batch_dir in batch_dirs:
        batch = int(batch_dir.name)
        sources_path = rss_sources_root / f"batch_{batch}_rss_sources.csv"
        if not sources_path.exists():
            print(f"  skipping batch {batch}: no recovered-source file at {sources_path}")
            continue

        raw_paths = list(batch_dir.glob("combined_log_wide_*.csv"))
        if len(raw_paths) != 1:
            raise ValueError(f"Expected one combined IMU file in {batch_dir}; found {len(raw_paths)}.")
        raw = pd.read_csv(raw_paths[0])
        recovered = pd.read_csv(sources_path)
        if len(recovered) != len(raw):
            raise ValueError(
                f"Batch {batch}: recovered-source row count {len(recovered)} does not match "
                f"raw row count {len(raw)}; they are no longer index-aligned."
            )

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

            window = recovered.iloc[aligned : aligned + len(truth)].reset_index(drop=True)
            window = window.rename(columns={"time_s": "batch_time_s"})
            matched = pd.concat([truth.reset_index(drop=True), window], axis=1)
            matched, joint_correlation, flipped_joints = _rename_and_unflip_sources(matched)
            matched_path = matched_root / f"test_{number}.csv"
            matched.to_csv(matched_path, index=False)

            reports.append({
                "date": date_root.name,
                "batch": batch,
                "test_number": number,
                "aligned_start": aligned,
                "coarse_start": region[0],
                "alignment_correlation": correlation,
                "source_joint_correlation": joint_correlation,
                "flipped_joints": flipped_joints,
                "row_count": len(matched),
                "matched_path": str(matched_path),
            })
    return reports


def main() -> None:
    all_reports = []
    for date_folder in DATE_FOLDERS:
        date_root = DATA_ROOT / date_folder
        rss_sources_root = RSS_SOURCES_ROOT / date_folder
        print(f"{date_folder}:")
        reports = build_matched_sources(date_root, rss_sources_root)
        all_reports.extend(reports)
        low_quality = [r for r in reports if r["alignment_correlation"] < 0.85]
        flipped = [r for r in reports if r["flipped_joints"]]
        print(f"  wrote {len(reports)} matched trials to {rss_sources_root / 'matched'}")
        if low_quality:
            print(f"  WARNING: {len(low_quality)} trial(s) below 0.85 alignment correlation:")
            for r in low_quality:
                print(f"    test_{r['test_number']}: correlation={r['alignment_correlation']:.3f}")
        if flipped:
            print(f"  unflipped {sum(len(r['flipped_joints']) for r in flipped)} source(s) across {len(flipped)} trial(s):")
            for r in flipped:
                print(f"    test_{r['test_number']}: {', '.join(r['flipped_joints'])}")

    print(f"\nTotal matched trials written: {len(all_reports)}")
    if all_reports:
        correlations = [r["alignment_correlation"] for r in all_reports]
        print(f"Alignment correlation: min={min(correlations):.3f} mean={np.mean(correlations):.3f}")
        total_flips = sum(len(r["flipped_joints"]) for r in all_reports)
        print(f"Total sign flips corrected: {total_flips} / {3 * len(all_reports)} source columns")


if __name__ == "__main__":
    main()
