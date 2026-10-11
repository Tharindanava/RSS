"""Staged CLI hyperparameter search for ``rss process-imu`` on QArm recordings.

The original production run (PSO, SSA on, 2000 epochs, a fixed non-identity
initial rotation) fixed every ``rss process-imu`` setting
up front and produced disappointing measured/source correlation. This script
instead runs a *staged, greedy* search over a random 10-batch sample drawn
from the 13 usable batch recordings (7 from ``session_2026-09-09``,
6 from ``session_2026-09-18``; the 7th 2026-09-18 batch is missing its
Node-10 gyro columns and is excluded), in the exact order requested:

    1. optimizer (adam / sgd / pso)
    2. SSA preprocessing on vs off
    3. (only if SSA helped) SSA window length, then reconstruction components
    4. epoch budget
    5. learning rate (start/end)
    6. initial recovery matrix (identity / the production custom matrix /
       random SO(3) draws)

Each stage fixes every earlier winner and sweeps only its own parameter, so
this is coordinate-descent, not a full grid search. A run's "score" is the
plain mean of the abs-correlation between each measured joint (base, elbow,
wrist) and its matched, sign-corrected RSS source across every trial in the
evaluated batches -- the same one-to-one linear-sum-assignment matching used
by ``build_matched_rss_sources.py``.

Compute-cost note: a single production-scale batch fit (PSO, swarm 200,
2000 epochs) took ~40 minutes in the original run.log. Testing every stage at
that scale would take days, so stages 1-3 (the coarse, mostly-structural
choices: optimizer, SSA on/off, SSA window/components) run at a reduced
budget (400 epochs, PSO swarm 60) on the full 10-batch sample. Stages 4-6
(epochs, learning rate, initial matrix -- parameters whose effect only shows
up near convergence) restore the production PSO swarm size (200) but run on
a fixed 4-batch subset of the same sample to bound wall-clock time. A final
validation re-evaluates the fully-assembled winning configuration on all 10
batches. Every (config, batch) evaluation is content-addressed and cached
under ``outputs/rss_hparam_search/cache/``, so identical settings carried
forward between stages are never re-run.

Batch evaluations run concurrently (default 8 workers x 6 torch threads =
48 of this machine's 56 cores) since each is an independent subprocess call
to ``rss process-imu``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from qarm_rss_experiments.build_merged_trials import _gyro_activity
from qarm_rss_experiments.practical_validation import (
    JOINT_NAMES,
    TRUTH_COLUMNS,
    _centered_moving_average,
    _measured_joint_speed,
    _movement_regions,
    _pearson,
    haar_so3,
    match_and_score,
)

PYTHON_BIN = sys.executable
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "Sample_Data" / "QArm"
SEARCH_ROOT = REPO_ROOT / "outputs" / "rss_hparam_search"
CACHE_ROOT = SEARCH_ROOT / "cache"
RUNS_ROOT = SEARCH_ROOT / "runs"
LOG_PATH = SEARCH_ROOT / "search_log.csv"
NARRATIVE_LOG_PATH = SEARCH_ROOT / "search_progress.log"
SUMMARY_PATH = SEARCH_ROOT / "SEARCH_SUMMARY.md"
RECOMMENDED_COMMAND_PATH = SEARCH_ROOT / "recommended_command.sh"

DATE_FOLDERS = ("session_2026-09-09", "session_2026-09-18")
NODE_PREFIX = {
    "session_2026-09-09": "OrbiX-Node-4_IMU1",
    "session_2026-09-18": "OrbiX-Node-10_IMU1",
}
SAMPLE_RATE_HZ = 100.0
SAMPLE_SEED = 20260929

# The custom initial rotation used in the production nohup script.
ORIGINAL_INITIAL_MATRIX: tuple[float, ...] = (
    -0.4779489885, -0.5421200170, -0.6911372162,
    -0.5983844737, 0.7769586228, -0.1956305705,
    0.6430402679, 0.3200643460, -0.6957427889,
)

SEARCH_EPOCHS = 400  # stages 1-3 budget (coarse, structural comparisons)
SEARCH_PSO_SWARM = 60  # stages 1-3 PSO swarm size (reduced for speed)
PRODUCTION_PSO_SWARM = 200  # stages 4-6 PSO swarm size (matches production)
WORKERS = 8
THREADS_PER_PROC = 6
RUN_TIMEOUT_SECONDS = 3 * 3600


@dataclass(frozen=True)
class Config:
    optimizer: str = "adam"
    disable_ssa: bool = False
    ssa_window: int = 150
    ssa_components: int = 20
    epochs: int = SEARCH_EPOCHS
    lr_start: float = 1e-1
    lr_end: float = 1e-3
    initial_matrix: tuple[float, ...] | None = None  # None => identity start
    pso_swarm: int = SEARCH_PSO_SWARM
    pso_inertia: float = 0.9
    pso_cognitive: float = 1.0
    pso_social: float = 1.5
    pso_seed: int = 0


def _log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    NARRATIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NARRATIVE_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def discover_batches() -> list[dict]:
    batches = []
    for session in DATE_FOLDERS:
        session_root = DATA_ROOT / session
        prefix = NODE_PREFIX[session]
        required_columns = [f"{prefix}_G{axis}" for axis in "XYZ"]
        batch_dirs = sorted(
            (p for p in session_root.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name),
        )
        for batch_dir in batch_dirs:
            csv_paths = list(batch_dir.glob("combined_log_wide_*.csv"))
            if len(csv_paths) != 1:
                continue
            header = pd.read_csv(csv_paths[0], nrows=0).columns
            if not all(column in header for column in required_columns):
                _log(f"Excluding {session} batch {batch_dir.name}: missing {required_columns}")
                continue
            batches.append(
                {
                    "session": session,
                    "batch": int(batch_dir.name),
                    "csv": csv_paths[0],
                    "node_prefix": prefix,
                }
            )
    return batches


def build_command(batch: dict, config: Config, output_path: Path) -> list[str]:
    cmd = [
        PYTHON_BIN, "-u", "-m", "rss", "process-imu",
        "--csv", str(batch["csv"]),
        "--node-prefix", batch["node_prefix"],
        "--gyro-units", "deg",
        "--sample-rate-hz", "100",
        "--bias-window-seconds", "2.0",
        "--epochs", str(config.epochs),
        "--learning-rate-start", str(config.lr_start),
        "--learning-rate-end", str(config.lr_end),
        "--rss-optimizer", config.optimizer,
        "--rss-matrix-parameterization", "so3",
        "--rss-observer-signal-model", "integrated_body_rotation",
        "--rss-loss-type", "lagged_cross_correlation",
        "--rss-lags", "0",
        "--rss-orthogonality-weight", "0.0",
        "--rss-reconstruction-weight", "0.0",
        "--rss-convergence-tol", "0.0",
        "--rss-eps", "1e-24",
        "--rss-quaternion-eps", "1e-24",
        "--output", str(output_path),
    ]
    if config.optimizer == "pso":
        cmd += [
            "--rss-pso-swarm-size", str(config.pso_swarm),
            "--rss-pso-inertia", str(config.pso_inertia),
            "--rss-pso-cognitive", str(config.pso_cognitive),
            "--rss-pso-social", str(config.pso_social),
            "--rss-pso-seed", str(config.pso_seed),
        ]
    if config.disable_ssa:
        cmd += ["--disable-ssa-preprocessing"]
    else:
        cmd += [
            "--ssa-window-length", str(config.ssa_window),
            "--ssa-reconstruction-components", str(config.ssa_components),
        ]
    if config.initial_matrix is not None:
        cmd += ["--rss-initial-weight-matrix", *[str(v) for v in config.initial_matrix]]
    return cmd


def _build_env() -> dict:
    env = os.environ.copy()
    threads = str(THREADS_PER_PROC)
    env.update(
        {
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "NUMEXPR_NUM_THREADS": threads,
            "MPLCONFIGDIR": "/tmp/rss_hparam_mpl",
        }
    )
    return env


def evaluate_batch_output(batch: dict, sources_path: Path) -> list[dict]:
    raw = pd.read_csv(batch["csv"])
    recovered = pd.read_csv(sources_path)
    if len(recovered) != len(raw):
        raise ValueError(
            f"Row-count mismatch for {batch['session']} batch {batch['batch']}: "
            f"raw={len(raw)} recovered={len(recovered)}"
        )
    activity = _gyro_activity(raw)
    regions = _movement_regions(activity, SAMPLE_RATE_HZ)
    if len(regions) != 5:
        raise ValueError(
            f"Expected five movement regions for {batch['session']} batch {batch['batch']}; found {len(regions)}."
        )
    smooth_activity = _centered_moving_average(activity, round(0.21 * SAMPLE_RATE_HZ))
    truth_root = DATA_ROOT / batch["session"] / "Ground_Truth" / "results"

    rows: list[dict] = []
    for local_index, region in enumerate(regions, 1):
        number = (batch["batch"] - 1) * 5 + local_index
        truth = pd.read_csv(truth_root / f"test_{number}.csv", float_precision="round_trip")
        speed = _centered_moving_average(_measured_joint_speed(truth), round(0.21 * SAMPLE_RATE_HZ))
        first = max(0, region[0] - round(1.5 * SAMPLE_RATE_HZ))
        last = min(len(raw) - len(truth), region[0] + round(1.5 * SAMPLE_RATE_HZ))
        candidates = [
            (_pearson(smooth_activity[start : start + len(truth)], speed), start)
            for start in range(first, last + 1)
        ]
        _, aligned = max(candidates)
        window = recovered.iloc[aligned : aligned + len(truth)].reset_index(drop=True)
        truth_matrix = truth[list(TRUTH_COLUMNS)].to_numpy(dtype=float).T
        source_matrix = window[["source_1", "source_2", "source_3"]].to_numpy(dtype=float).T
        scored, _aligned = match_and_score(
            truth_matrix,
            source_matrix,
            test_number=number,
            batch_number=batch["batch"],
            method="search",
        )
        rows.extend(scored)
    return rows


def _config_cache_key(batch: dict, config: Config) -> str:
    payload = json.dumps(
        {"batch": f"{batch['session']}#{batch['batch']}", **asdict(config)},
        sort_keys=True,
        default=list,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def evaluate_one(batch: dict, config: Config) -> dict:
    key = _config_cache_key(batch, config)
    cache_file = CACHE_ROOT / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    output_path = RUNS_ROOT / f"{key}_sources.csv"
    started = time.perf_counter()
    try:
        cmd = build_command(batch, config, output_path)
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=_build_env(),
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-4000:])
        rows = evaluate_batch_output(batch, output_path)
    except Exception as exc:  # noqa: BLE001 - captured for the cache/log, not swallowed silently
        payload = {
            "ok": False,
            "batch": f"{batch['session']}#{batch['batch']}",
            "error": str(exc),
            "runtime_seconds": time.perf_counter() - started,
        }
        cache_file.write_text(json.dumps(payload))
        return payload

    payload = {
        "ok": True,
        "batch": f"{batch['session']}#{batch['batch']}",
        "runtime_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    cache_file.write_text(json.dumps(payload))
    return payload


def _append_search_log(summary: dict) -> None:
    is_new = not LOG_PATH.exists()
    flat = {
        "name": summary["name"],
        "n_batches": summary["n_batches"],
        "n_failed": summary["n_failed"],
        "mean_abs_correlation": summary["mean_abs_correlation"],
        "fisher_mean_abs_correlation": summary["fisher_mean_abs_correlation"],
        "base_abs_correlation": summary["per_joint"].get("base"),
        "elbow_abs_correlation": summary["per_joint"].get("elbow"),
        "wrist_abs_correlation": summary["per_joint"].get("wrist"),
        "total_runtime_seconds": summary["total_runtime_seconds"],
        **{f"cfg_{k}": v for k, v in summary["config"].items()},
    }
    frame = pd.DataFrame([flat])
    frame.to_csv(LOG_PATH, mode="a", header=is_new, index=False)


def evaluate_config(name: str, config: Config, batches: list[dict]) -> dict:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(lambda b: evaluate_one(b, config), batches))

    ok_results = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]
    for failure in failed:
        _log(f"  FAILED {name} on {failure.get('batch')}: {failure.get('error')}")

    all_rows = [row for r in ok_results for row in r["rows"]]
    abs_values = np.asarray([row["abs_correlation"] for row in all_rows], dtype=float)
    if abs_values.size == 0:
        mean_abs = float("nan")
        fisher_mean = float("nan")
        per_joint = {joint: float("nan") for joint in JOINT_NAMES}
    else:
        mean_abs = float(np.mean(abs_values))
        clipped = np.clip(abs_values, 0.0, 1.0 - 1e-12)
        fisher_mean = float(np.tanh(np.mean(np.arctanh(clipped))))
        per_joint = {
            joint: float(np.mean([row["abs_correlation"] for row in all_rows if row["joint"] == joint]))
            for joint in JOINT_NAMES
        }

    summary = {
        "name": name,
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(config).items()},
        "n_batches": len(batches),
        "n_failed": len(failed),
        "mean_abs_correlation": mean_abs,
        "fisher_mean_abs_correlation": fisher_mean,
        "per_joint": per_joint,
        "total_runtime_seconds": time.perf_counter() - started,
    }
    _append_search_log(summary)
    _log(
        f"  {name}: mean|r|={mean_abs:.4f} fisher|r|={fisher_mean:.4f} "
        f"(base={per_joint['base']:.3f} elbow={per_joint['elbow']:.3f} wrist={per_joint['wrist']:.3f}) "
        f"[{len(ok_results)}/{len(batches)} batches ok, {time.perf_counter() - started:.0f}s wall]"
    )
    return summary


def _log_stage_header(title: str) -> None:
    _log("")
    _log(f"===== {title} =====")


def _log_winner(stage: str, winner: dict) -> None:
    _log(f"  -> {stage} winner: {winner['name']} (mean|r|={winner['mean_abs_correlation']:.4f})")


def _matrix_name(candidates: dict[str, tuple[float, ...] | None], matrix) -> str:
    matrix_list = list(matrix) if matrix is not None else None
    for candidate_name, candidate_value in candidates.items():
        candidate_list = list(candidate_value) if candidate_value is not None else None
        if candidate_list == matrix_list:
            return candidate_name
    return "unknown"


def write_summary(stage_winners: list[tuple[str, dict]], final_config: Config, final_check: dict) -> None:
    lines = ["# QArm RSS hyperparameter search summary", ""]
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Stage winners")
    for stage_name, winner in stage_winners:
        lines.append(f"- **{stage_name}**: `{winner['name']}` -> mean|r| = {winner['mean_abs_correlation']:.4f}")
    lines.append("")
    lines.append("## Final recommended configuration")
    lines.append("```")
    lines.append(json.dumps(asdict(final_config), indent=2, default=list))
    lines.append("```")
    lines.append("")
    lines.append(
        f"Final validation on the full 10-batch sample: mean|r| = {final_check['mean_abs_correlation']:.4f}, "
        f"fisher|r| = {final_check['fisher_mean_abs_correlation']:.4f} "
        f"(base={final_check['per_joint']['base']:.3f}, elbow={final_check['per_joint']['elbow']:.3f}, "
        f"wrist={final_check['per_joint']['wrist']:.3f})"
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")

    cmd_lines = ["#!/usr/bin/env bash", "# Recommended rss process-imu settings from the staged hyperparameter search.", "set -euo pipefail", ""]
    example_batch = {"csv": "<csv>", "node_prefix": "<node-prefix>"}
    command = build_command(example_batch, final_config, Path("<output>.csv"))
    cmd_lines.append(" \\\n    ".join(command))
    RECOMMENDED_COMMAND_PATH.write_text("\n".join(cmd_lines) + "\n", encoding="utf-8")


def run_search(smoke_test: bool = False) -> None:
    SEARCH_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    all_batches = discover_batches()
    _log(f"Discovered {len(all_batches)} usable batches.")
    rng = np.random.default_rng(SAMPLE_SEED)
    sample_size = 3 if smoke_test else 10
    indices = sorted(rng.choice(len(all_batches), size=sample_size, replace=False).tolist())
    search_batches = [all_batches[i] for i in indices]
    final_batches = search_batches[: (2 if smoke_test else 4)]
    _log(f"Search sample ({len(search_batches)} batches): "
         + ", ".join(f"{b['session']}#{b['batch']}" for b in search_batches))
    _log(f"Reduced subset for expensive later stages ({len(final_batches)} batches): "
         + ", ".join(f"{b['session']}#{b['batch']}" for b in final_batches))

    search_epochs = 20 if smoke_test else SEARCH_EPOCHS
    baseline = Config(
        optimizer="adam",
        disable_ssa=False,
        ssa_window=150,
        ssa_components=20,
        epochs=search_epochs,
        lr_start=1e-1,
        lr_end=1e-3,
        initial_matrix=None,
        pso_swarm=SEARCH_PSO_SWARM,
    )

    stage_winners: list[tuple[str, dict]] = []

    # ---- Stage 1: optimizer ----
    _log_stage_header("Stage 1: optimizer")
    optimizers = ("adam", "sgd") if smoke_test else ("adam", "sgd", "pso")
    stage1 = [evaluate_config(f"stage1_optimizer_{opt}", replace(baseline, optimizer=opt), search_batches) for opt in optimizers]
    winner1 = max(stage1, key=lambda r: r["mean_abs_correlation"])
    stage_winners.append(("Stage 1 (optimizer)", winner1))
    _log_winner("Stage 1", winner1)
    baseline = replace(baseline, optimizer=winner1["config"]["optimizer"])

    # ---- Stage 2: SSA on/off ----
    _log_stage_header("Stage 2: SSA preprocessing on vs off")
    r_on = evaluate_config("stage2_ssa_on", replace(baseline, disable_ssa=False), search_batches)
    r_off = evaluate_config("stage2_ssa_off", replace(baseline, disable_ssa=True), search_batches)
    ssa_warranted = r_on["mean_abs_correlation"] > r_off["mean_abs_correlation"]
    winner2 = r_on if ssa_warranted else r_off
    stage_winners.append(("Stage 2 (SSA on/off)", winner2))
    _log_winner("Stage 2", winner2)
    _log(f"  SSA preprocessing warranted: {ssa_warranted}")
    baseline = replace(baseline, disable_ssa=not ssa_warranted)

    # ---- Stage 3: SSA window / components (only if warranted) ----
    if ssa_warranted:
        _log_stage_header("Stage 3a: SSA window length")
        window_candidates = [100, 150, 200] if smoke_test else [75, 100, 150, 200, 250]
        stage3a = [
            evaluate_config(f"stage3_window_{w}", replace(baseline, ssa_window=w), search_batches)
            for w in window_candidates
        ]
        winner3a = max(stage3a, key=lambda r: r["mean_abs_correlation"])
        stage_winners.append(("Stage 3a (SSA window)", winner3a))
        _log_winner("Stage 3a", winner3a)
        baseline = replace(baseline, ssa_window=winner3a["config"]["ssa_window"])

        _log_stage_header("Stage 3b: SSA reconstruction components")
        component_candidates = [15, 20, 25] if smoke_test else [10, 15, 20, 25, 30]
        stage3b = [
            evaluate_config(f"stage3_components_{c}", replace(baseline, ssa_components=c), search_batches)
            for c in component_candidates
        ]
        winner3b = max(stage3b, key=lambda r: r["mean_abs_correlation"])
        stage_winners.append(("Stage 3b (SSA components)", winner3b))
        _log_winner("Stage 3b", winner3b)
        baseline = replace(baseline, ssa_components=winner3b["config"]["ssa_components"])
    else:
        _log_stage_header("Stage 3: skipped (SSA did not help in Stage 2)")

    # Promote to production PSO swarm size for the remaining, per-run-expensive stages.
    if baseline.optimizer == "pso":
        baseline = replace(baseline, pso_swarm=PRODUCTION_PSO_SWARM)

    # ---- Stage 4: epochs ----
    _log_stage_header("Stage 4: epoch budget")
    epoch_candidates = [50, 100] if smoke_test else sorted({300, 700, 1500, 2000})
    stage4 = [
        evaluate_config(f"stage4_epochs_{e}", replace(baseline, epochs=e), final_batches)
        for e in epoch_candidates
    ]
    winner4 = max(stage4, key=lambda r: r["mean_abs_correlation"])
    stage_winners.append(("Stage 4 (epochs)", winner4))
    _log_winner("Stage 4", winner4)
    baseline = replace(baseline, epochs=winner4["config"]["epochs"])

    # ---- Stage 5: learning rate ----
    _log_stage_header("Stage 5: learning rate (start, end)")
    lr_candidates = (
        [(1e-1, 1e-3), (1e-1, 1e-4)]
        if smoke_test
        else [(1e-1, 1e-3), (2e-1, 1e-3), (1e-1, 1e-4), (5e-2, 1e-3)]
    )
    stage5 = [
        evaluate_config(f"stage5_lr_{start}_{end}", replace(baseline, lr_start=start, lr_end=end), final_batches)
        for start, end in lr_candidates
    ]
    winner5 = max(stage5, key=lambda r: r["mean_abs_correlation"])
    stage_winners.append(("Stage 5 (learning rate)", winner5))
    _log_winner("Stage 5", winner5)
    baseline = replace(baseline, lr_start=winner5["config"]["lr_start"], lr_end=winner5["config"]["lr_end"])

    # ---- Stage 6: initial recovery matrix ----
    _log_stage_header("Stage 6: initial recovery matrix")
    rng6 = np.random.default_rng(7)
    random_matrices = haar_so3(rng6, 2)
    init_candidates: dict[str, tuple[float, ...] | None] = {
        "identity": None,
        "original_custom": ORIGINAL_INITIAL_MATRIX,
        "random_1": tuple(float(v) for v in random_matrices[0].flatten()),
        "random_2": tuple(float(v) for v in random_matrices[1].flatten()),
    }
    if smoke_test:
        init_candidates = {"identity": None, "original_custom": ORIGINAL_INITIAL_MATRIX}
    stage6 = [
        evaluate_config(f"stage6_init_{name}", replace(baseline, initial_matrix=matrix), final_batches)
        for name, matrix in init_candidates.items()
    ]
    winner6 = max(stage6, key=lambda r: r["mean_abs_correlation"])
    stage_winners.append(("Stage 6 (initial matrix)", winner6))
    winner_matrix = winner6["config"]["initial_matrix"]
    winner_matrix_name = _matrix_name(init_candidates, winner_matrix)
    _log(f"  -> Stage 6 winner: {winner6['name']} ({winner_matrix_name}, mean|r|={winner6['mean_abs_correlation']:.4f})")
    baseline = replace(
        baseline,
        initial_matrix=tuple(winner_matrix) if winner_matrix is not None else None,
    )

    # ---- Final validation on the full search sample ----
    _log_stage_header("Final validation (full search sample)")
    final_check = evaluate_config("final_validation", baseline, search_batches)
    _log_winner("FINAL", final_check)

    write_summary(stage_winners, baseline, final_check)
    _log(f"Wrote summary to {SUMMARY_PATH}")
    _log(f"Wrote recommended command to {RECOMMENDED_COMMAND_PATH}")
    _log("Search complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast, trimmed-down version of every stage to validate the pipeline end-to-end.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_search(smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
