"""Reprocess every usable QArm batch with the hyperparameter-search winner and score it.

``hyperparameter_search.py`` found that, relative to the original production
run (``rss-optimizer pso``, SSA preprocessing on, 2000 epochs, a custom
initial rotation), the two changes that reliably improved
measured/source correlation on the full 10-batch search sample were:

    * ``--rss-optimizer adam`` instead of ``pso`` (also ~13x cheaper per run)
    * ``--disable-ssa-preprocessing`` (SSA's aggressive smoothing/detrending
      was removing real joint-motion content the separation loss needs)

The epoch, learning-rate, and initial-matrix stages were swept on a smaller
4-batch subset (to bound wall-clock time) and did not hold up under full
10-batch validation, so this script deliberately uses the simpler, more
robustly-validated ``stage2_ssa_off`` settings (400 epochs, default learning
rate, identity start) rather than the search's nominal final config.

This reprocesses all 13 usable batches (not just the 10-batch search
sample), rebuilds ``matched/test_N.csv`` per trial with
``build_matched_rss_sources.build_matched_sources``, and reports the overall
mean/Fisher absolute correlation next to the existing production numbers so
the improvement is measured on the full dataset, not just the search sample.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from qarm_rss_experiments.build_matched_rss_sources import build_matched_sources
from qarm_rss_experiments.hyperparameter_search import (
    Config,
    DATA_ROOT,
    build_command,
    discover_batches,
)
import subprocess

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "outputs" / "rss" / "QArm_Adam_NoSSA_Optimized"
PRODUCTION_ROOT = REPO_ROOT / "outputs" / "rss" / "QArm_PSO_SO3_2000Epochs"

WINNING_CONFIG = Config(
    optimizer="adam",
    disable_ssa=True,
    epochs=400,
    lr_start=1e-1,
    lr_end=1e-3,
    initial_matrix=None,
)


def _score_matched_root(rss_sources_root: Path, label: str) -> None:
    joints = ("base", "elbow", "wrist")
    rows = []
    for date_dir in sorted(rss_sources_root.iterdir()):
        matched_dir = date_dir / "matched"
        if not matched_dir.is_dir():
            continue
        for path in sorted(matched_dir.glob("test_*.csv")):
            frame = pd.read_csv(path)
            for joint in joints:
                measured = frame[f"measured_{joint}_rad"].to_numpy(float)
                source = frame[f"source_{joint}"].to_numpy(float)
                m = measured - measured.mean()
                s = source - source.mean()
                denom = np.linalg.norm(m) * np.linalg.norm(s)
                r = abs(float(np.dot(m, s) / denom)) if denom > 0 else float("nan")
                rows.append({"joint": joint, "abs_r": r})
    frame = pd.DataFrame(rows)
    fisher = float(np.tanh(np.mean(np.arctanh(np.clip(frame["abs_r"], 0.0, 1.0 - 1e-12)))))
    print(f"\n{label}: n={len(frame)} (joint x trial pairs)")
    print(f"  overall mean|r|   = {frame['abs_r'].mean():.4f}")
    print(f"  overall fisher|r| = {fisher:.4f}")
    print(frame.groupby("joint")["abs_r"].mean().to_string())


def main() -> None:
    batches = discover_batches()
    print(f"Reprocessing {len(batches)} usable batches with the search-winning settings:")
    print(WINNING_CONFIG)

    for session in {b["session"] for b in batches}:
        (OUTPUT_ROOT / session).mkdir(parents=True, exist_ok=True)

    for batch in batches:
        output_path = OUTPUT_ROOT / batch["session"] / f"batch_{batch['batch']}_rss_sources.csv"
        started = time.perf_counter()
        cmd = build_command(batch, WINNING_CONFIG, output_path)
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FAILED {batch['session']} batch {batch['batch']}: {result.stderr[-2000:]}")
            continue
        print(f"  {batch['session']} batch {batch['batch']}: {time.perf_counter() - started:.1f}s")

    for session in sorted({b["session"] for b in batches}):
        date_root = DATA_ROOT / session
        rss_sources_root = OUTPUT_ROOT / session
        reports = build_matched_sources(date_root, rss_sources_root)
        print(f"{session}: wrote {len(reports)} matched trials")

    _score_matched_root(PRODUCTION_ROOT, "EXISTING PRODUCTION (pso, SSA on, 2000 epochs, custom init)")
    _score_matched_root(OUTPUT_ROOT, "SEARCH-OPTIMIZED (adam, SSA off, 400 epochs, default lr, identity init)")


if __name__ == "__main__":
    main()
