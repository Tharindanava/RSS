"""Reproduce the recommended QArm analysis end to end from the shipped recordings.

Steps (all outputs below ``outputs/rss/QArm_Adam_NoSSA_Piecewise``):

1. ``rss process-imu`` on every usable batch recording with the recommended recipe
   (Adam, SSA preprocessing off, 1.0 s bias window, 400 epochs, adaptive
   ``piecewise_sparse_innovation`` regularizer) -> ``<session>/batch_N_rss_sources.csv``.
2. Split every batch into its five trials and pair the recovered sources with the encoder
   ground truth (sign-corrected, one-to-one matched) -> ``<session>/matched/test_N.csv``.
3. Rank all matched trials by mean |Pearson r| across base/elbow/wrist, keep the best 30, fit
   PCA / JADE-ICA / FastICA / SOBI on exactly the same observation windows, and write the
   ranking table, comparison CSV, per-trial plots and PDF report
   (``best_results_report.py``).

Run from the repository root::

    python -m qarm_rss_experiments.reproduce_recommended_pipeline

RSS fits take a few minutes in total on a CPU; step 3 refits four baselines on 30 windows.
Use ``--skip-fit`` to reuse existing ``batch_N_rss_sources.csv`` files and ``--skip-report`` to stop
after the matched trials have been written.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from qarm_rss_experiments import best_results_report
from qarm_rss_experiments.build_matched_rss_sources import build_matched_sources
from qarm_rss_experiments.hyperparameter_search_round2 import Config2, build_command2
from qarm_rss_experiments.hyperparameter_search import DATA_ROOT, REPO_ROOT, _build_env, discover_batches
from qarm_rss_experiments.plot_matched_signals import RSS_SOURCES_ROOT

RECOMMENDED_CONFIG = Config2(
    optimizer="adam",
    disable_ssa=True,
    epochs=400,
    lr_start=1e-1,
    lr_end=1e-3,
    initial_matrix=None,
    bias_window_seconds=1.0,
)
ADAPTIVE_REGULARIZER_FLAGS = [
    "--rss-adaptive-regularizer",
    "--rss-specialized-regularizer", "piecewise_sparse_innovation",
    "--rss-regularizer-start-weight", "0.5",
    "--rss-regularizer-plateau-patience", "20",
    "--rss-regularizer-ramp-epochs", "20",
]


def _fit_batch(batch: dict, *, force: bool) -> tuple[dict, bool, str]:
    output_path = RSS_SOURCES_ROOT / batch["session"] / f"batch_{batch['batch']}_rss_sources.csv"
    if output_path.exists() and not force:
        return batch, True, "reused existing file"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_command2(batch, RECOMMENDED_CONFIG, output_path) + ADAPTIVE_REGULARIZER_FLAGS
    result = subprocess.run(command, cwd=REPO_ROOT, env=_build_env(), capture_output=True, text=True)
    if result.returncode != 0:
        return batch, False, result.stderr[-1500:]
    return batch, True, "fitted"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent `rss process-imu` processes.")
    parser.add_argument("--force", action="store_true", help="Refit batches even if their source file exists.")
    parser.add_argument("--skip-fit", action="store_true", help="Reuse existing batch_N_rss_sources.csv files.")
    parser.add_argument("--skip-report", action="store_true", help="Stop after writing matched/test_N.csv.")
    args = parser.parse_args(argv)

    batches = discover_batches()
    print(f"{len(batches)} usable batches under {DATA_ROOT}", flush=True)

    if not args.skip_fit:
        started = time.time()
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for batch, ok, message in pool.map(lambda b: _fit_batch(b, force=args.force), batches):
                label = f"{batch['session']} batch {batch['batch']}"
                print(f"  {label}: {message}" if ok else f"  {label}: FAILED\n{message}", flush=True)
                if not ok:
                    raise SystemExit(1)
        print(f"RSS fits finished in {time.time() - started:.0f} s", flush=True)

    for session in sorted({b["session"] for b in batches}):
        reports = build_matched_sources(DATA_ROOT / session, RSS_SOURCES_ROOT / session)
        print(f"{session}: wrote {len(reports)} matched trials", flush=True)

    # Every matched trial, best first: the paper's QArm column uses only the top 30 of these.
    ranking_path = RSS_SOURCES_ROOT / "all_trials_ranking.csv"
    ranked = best_results_report._rank_all_trials().drop(columns=["path"])
    ranked.to_csv(ranking_path, index=False)
    print(f"{len(ranked)} trials ranked; mean |r| over all of them = {ranked['mean_r'].mean():.4f} -> {ranking_path}", flush=True)

    if not args.skip_report:
        best_results_report.main()


if __name__ == "__main__":
    main()
