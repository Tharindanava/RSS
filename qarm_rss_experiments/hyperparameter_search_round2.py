"""Round 2: push the Adam + no-SSA winner from ``hyperparameter_search.py`` further.

Round 1 found optimizer=adam, SSA disabled, ~400 epochs, default learning
rate, identity start (mean|r|=0.765 across all 65 matched trials), and found
that the epoch/learning-rate/initial-matrix stages it explored did not move
the needle much. This round starts from that exact baseline and sweeps four
*new* levers that Round 1 never touched:

    1. ``--bias-window-seconds`` -- with SSA preprocessing disabled there is
       no detrending left in the pipeline, so the quality of the static gyro
       bias estimate (averaged over this many seconds of the stationary
       lead-in) directly controls how much the integrated angle drifts over
       the 30s trial.
    2. ``--rss-lags`` -- the separation loss decorrelates the three recovered
       sources at lag 0 by default. This is a *blind* loss (it never sees
       ground truth): forcing decorrelation at several lags too is the same
       idea as SOBI's joint diagonalization across lagged covariances, and
       is a materially stronger separation criterion for smooth, structured
       signals like joint motion (zero-lag decorrelation alone is a weak
       constraint for autocorrelated signals).
    3. ``--rss-axis-flip-search`` -- periodically compares identity against
       the three proper SO(3) double-axis flips during optimization, i.e. a
       cheap way to let Adam escape a bad basin without paying for a full
       PSO swarm.
    4. Multi-start Adam -- run several random SO(3) initial matrices per
       batch (via ``haar_so3``) and, for each batch, keep whichever run has
       the lowest *unsupervised* ``best_loss`` (never chosen by peeking at
       ground truth), then score only that winner. This approximates PSO's
       global search cheaply since Adam itself is inexpensive.

Each stage is evaluated on the same 10-batch search sample as round 1 (same
seed), coordinate-descent style, then the fully assembled winner is
reprocessed on all 13 usable batches for a final, comparable "mean|r| across
all 65 trials" number.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from qarm_rss_experiments.build_matched_rss_sources import build_matched_sources
from qarm_rss_experiments.hyperparameter_search import (
    Config,
    DATA_ROOT,
    REPO_ROOT,
    SAMPLE_SEED,
    JOINT_NAMES,
    _build_env,
    discover_batches,
    evaluate_batch_output,
)
from qarm_rss_experiments.practical_validation import haar_so3
import subprocess

PYTHON_BIN = sys.executable
ROUND2_ROOT = REPO_ROOT / "outputs" / "rss_hparam_search_round2"
CACHE_ROOT = ROUND2_ROOT / "cache"
RUNS_ROOT = ROUND2_ROOT / "runs"
LOG_PATH = ROUND2_ROOT / "search_log.csv"
NARRATIVE_LOG_PATH = ROUND2_ROOT / "search_progress.log"
SUMMARY_PATH = ROUND2_ROOT / "SEARCH_SUMMARY.md"
FINAL_OUTPUT_ROOT = REPO_ROOT / "outputs" / "rss" / "QArm_Adam_NoSSA_Round2"

RUN_TIMEOUT_SECONDS = 3 * 3600
WORKERS = 8
THREADS_PER_PROC = 6

# Round 1's winning baseline.
ROUND1_WINNER = Config(
    optimizer="adam",
    disable_ssa=True,
    epochs=400,
    lr_start=1e-1,
    lr_end=1e-3,
    initial_matrix=None,
)


@dataclass(frozen=True)
class Config2(Config):
    bias_window_seconds: float = 2.0
    lags: tuple[int, ...] = (0,)
    axis_flip_search: bool = False
    axis_flip_interval: int = 100
    axis_flip_trial_epochs: int = 10


def _log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    NARRATIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NARRATIVE_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def build_command2(batch: dict, config: Config2, output_path: Path) -> list[str]:
    cmd = [
        PYTHON_BIN, "-u", "-m", "rss", "process-imu",
        "--csv", str(batch["csv"]),
        "--node-prefix", batch["node_prefix"],
        "--gyro-units", "deg",
        "--sample-rate-hz", "100",
        "--bias-window-seconds", str(config.bias_window_seconds),
        "--epochs", str(config.epochs),
        "--learning-rate-start", str(config.lr_start),
        "--learning-rate-end", str(config.lr_end),
        "--rss-optimizer", config.optimizer,
        "--rss-matrix-parameterization", "so3",
        "--rss-observer-signal-model", "integrated_body_rotation",
        "--rss-loss-type", "lagged_cross_correlation",
        "--rss-lags", *[str(lag) for lag in config.lags],
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
    if config.axis_flip_search:
        cmd += [
            "--rss-axis-flip-search",
            "--rss-axis-flip-interval", str(config.axis_flip_interval),
            "--rss-axis-flip-trial-epochs", str(config.axis_flip_trial_epochs),
        ]
    return cmd


def _config_cache_key(batch: dict, config: Config2, tag: str = "") -> str:
    import hashlib
    from dataclasses import asdict

    payload = json.dumps(
        {"batch": f"{batch['session']}#{batch['batch']}", "tag": tag, **asdict(config)},
        sort_keys=True,
        default=list,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def run_one(batch: dict, config: Config2, tag: str = "") -> dict:
    """Run one process-imu call, returning ok/rows/best_loss, with disk caching."""
    key = _config_cache_key(batch, config, tag)
    cache_file = CACHE_ROOT / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    output_path = RUNS_ROOT / f"{key}_sources.csv"
    started = time.perf_counter()
    try:
        cmd = build_command2(batch, config, output_path)
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, env=_build_env(), capture_output=True, text=True, timeout=RUN_TIMEOUT_SECONDS
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-4000:])
        best_loss = json.loads(result.stdout)["best_loss"]
        rows = evaluate_batch_output(batch, output_path)
    except Exception as exc:  # noqa: BLE001
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
        "best_loss": best_loss,
        "runtime_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    cache_file.write_text(json.dumps(payload))
    return payload


def _summarize(name: str, config: Config2, results: list[dict], started: float) -> dict:
    ok_results = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]
    for failure in failed:
        _log(f"  FAILED {name} on {failure.get('batch')}: {failure.get('error')}")
    all_rows = [row for r in ok_results for row in r["rows"]]
    abs_values = np.asarray([row["abs_correlation"] for row in all_rows], dtype=float)
    if abs_values.size == 0:
        mean_abs, fisher_mean = float("nan"), float("nan")
        per_joint = {joint: float("nan") for joint in JOINT_NAMES}
    else:
        mean_abs = float(np.mean(abs_values))
        clipped = np.clip(abs_values, 0.0, 1.0 - 1e-12)
        fisher_mean = float(np.tanh(np.mean(np.arctanh(clipped))))
        per_joint = {
            joint: float(np.mean([row["abs_correlation"] for row in all_rows if row["joint"] == joint]))
            for joint in JOINT_NAMES
        }
    from dataclasses import asdict

    summary = {
        "name": name,
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(config).items()},
        "n_batches": len(results),
        "n_failed": len(failed),
        "mean_abs_correlation": mean_abs,
        "fisher_mean_abs_correlation": fisher_mean,
        "per_joint": per_joint,
        "total_runtime_seconds": time.perf_counter() - started,
    }
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
    pd.DataFrame([flat]).to_csv(LOG_PATH, mode="a", header=is_new, index=False)
    _log(
        f"  {name}: mean|r|={mean_abs:.4f} fisher|r|={fisher_mean:.4f} "
        f"(base={per_joint['base']:.3f} elbow={per_joint['elbow']:.3f} wrist={per_joint['wrist']:.3f}) "
        f"[{len(ok_results)}/{len(results)} ok, {time.perf_counter() - started:.0f}s]"
    )
    return summary


def evaluate_config2(name: str, config: Config2, batches: list[dict]) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(lambda b: run_one(b, config), batches))
    return _summarize(name, config, results, started)


def evaluate_multistart(name: str, config: Config2, candidate_matrices: list[tuple[float, ...] | None], batches: list[dict]) -> dict:
    """For each batch, run every candidate initial matrix and keep the lowest-best_loss run."""
    from concurrent.futures import ThreadPoolExecutor

    started = time.perf_counter()

    def _best_for_batch(batch: dict) -> dict:
        candidate_results = []
        for tag_index, matrix in enumerate(candidate_matrices):
            cfg = replace(config, initial_matrix=matrix)
            result = run_one(batch, cfg, tag=f"multistart_{tag_index}")
            candidate_results.append(result)
        ok_candidates = [r for r in candidate_results if r.get("ok")]
        if not ok_candidates:
            return candidate_results[0]
        return min(ok_candidates, key=lambda r: r["best_loss"])

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(_best_for_batch, batches))
    return _summarize(name, config, results, started)


def write_summary(stage_winners: list[tuple[str, dict]], final_config: Config2, final_check: dict) -> None:
    from dataclasses import asdict

    lines = ["# QArm RSS fine-tuning round 2 summary", "", f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
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
        f"Final validation (full 65-trial reprocessing): mean|r| = {final_check['mean_abs_correlation']:.4f}, "
        f"fisher|r| = {final_check['fisher_mean_abs_correlation']:.4f} "
        f"(base={final_check['per_joint']['base']:.3f}, elbow={final_check['per_joint']['elbow']:.3f}, "
        f"wrist={final_check['per_joint']['wrist']:.3f})"
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def reprocess_all_batches(config: Config2) -> dict:
    """Reprocess all 13 usable batches with `config` and score all 65 matched trials."""
    batches = discover_batches()
    for session in {b["session"] for b in batches}:
        (FINAL_OUTPUT_ROOT / session).mkdir(parents=True, exist_ok=True)

    for batch in batches:
        output_path = FINAL_OUTPUT_ROOT / batch["session"] / f"batch_{batch['batch']}_rss_sources.csv"
        cmd = build_command2(batch, config, output_path)
        result = subprocess.run(cmd, cwd=REPO_ROOT, env=_build_env(), capture_output=True, text=True)
        if result.returncode != 0:
            _log(f"FAILED final reprocess {batch['session']} batch {batch['batch']}: {result.stderr[-2000:]}")

    for session in sorted({b["session"] for b in batches}):
        reports = build_matched_sources(DATA_ROOT / session, FINAL_OUTPUT_ROOT / session)
        _log(f"{session}: wrote {len(reports)} matched trials")

    joints = ("base", "elbow", "wrist")
    rows = []
    for session in sorted({b["session"] for b in batches}):
        matched_dir = FINAL_OUTPUT_ROOT / session / "matched"
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
    per_joint = {joint: float(frame.loc[frame["joint"] == joint, "abs_r"].mean()) for joint in joints}
    return {
        "mean_abs_correlation": float(frame["abs_r"].mean()),
        "fisher_mean_abs_correlation": fisher,
        "per_joint": per_joint,
        "n": len(frame),
    }


def main() -> None:
    ROUND2_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    all_batches = discover_batches()
    rng = np.random.default_rng(SAMPLE_SEED)
    indices = sorted(rng.choice(len(all_batches), size=10, replace=False).tolist())
    search_batches = [all_batches[i] for i in indices]
    _log("Round 2: fine-tuning the Adam + no-SSA winner.")
    _log(f"Search sample (same as round 1, {len(search_batches)} batches): "
         + ", ".join(f"{b['session']}#{b['batch']}" for b in search_batches))

    baseline = Config2(
        optimizer=ROUND1_WINNER.optimizer,
        disable_ssa=ROUND1_WINNER.disable_ssa,
        epochs=ROUND1_WINNER.epochs,
        lr_start=ROUND1_WINNER.lr_start,
        lr_end=ROUND1_WINNER.lr_end,
        initial_matrix=ROUND1_WINNER.initial_matrix,
    )
    stage_winners: list[tuple[str, dict]] = []

    baseline_check = evaluate_config2("round1_baseline_recheck", baseline, search_batches)
    stage_winners.append(("Round 1 baseline (recheck)", baseline_check))

    # ---- Stage A: bias-window-seconds ----
    _log_stage = lambda title: (_log(""), _log(f"===== {title} ====="))
    _log_stage("Stage A: bias-window-seconds")
    candidates = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    stageA = [evaluate_config2(f"stageA_bias_{c}", replace(baseline, bias_window_seconds=c), search_batches) for c in candidates]
    winnerA = max(stageA, key=lambda r: r["mean_abs_correlation"])
    stage_winners.append(("Stage A (bias window)", winnerA))
    _log(f"  -> winner: {winnerA['name']} (mean|r|={winnerA['mean_abs_correlation']:.4f})")
    baseline = replace(baseline, bias_window_seconds=winnerA["config"]["bias_window_seconds"])

    # ---- Stage B: lags ----
    _log_stage("Stage B: rss-lags (multi-lag decorrelation)")
    lag_candidates = [(0,), (0, 1, 2), (0, 1, 2, 3, 5), (0, 1, 2, 3, 5, 8, 13), (0, 2, 4, 6, 8, 10)]
    stageB = [
        evaluate_config2(f"stageB_lags_{'_'.join(map(str, lags))}", replace(baseline, lags=lags), search_batches)
        for lags in lag_candidates
    ]
    winnerB = max(stageB, key=lambda r: r["mean_abs_correlation"])
    stage_winners.append(("Stage B (lags)", winnerB))
    _log(f"  -> winner: {winnerB['name']} (mean|r|={winnerB['mean_abs_correlation']:.4f})")
    baseline = replace(baseline, lags=tuple(winnerB["config"]["lags"]))

    # ---- Stage C: axis-flip-search ----
    _log_stage("Stage C: axis-flip-search on vs off")
    cfg_flip_on = replace(baseline, axis_flip_search=True, axis_flip_interval=100, axis_flip_trial_epochs=10)
    cfg_flip_off = replace(baseline, axis_flip_search=False)
    r_on = evaluate_config2("stageC_flip_on", cfg_flip_on, search_batches)
    r_off = evaluate_config2("stageC_flip_off", cfg_flip_off, search_batches)
    winnerC = r_on if r_on["mean_abs_correlation"] > r_off["mean_abs_correlation"] else r_off
    stage_winners.append(("Stage C (axis-flip)", winnerC))
    _log(f"  -> winner: {winnerC['name']} (mean|r|={winnerC['mean_abs_correlation']:.4f})")
    baseline = replace(baseline, axis_flip_search=winnerC["config"]["axis_flip_search"])

    # ---- Stage D: multi-start Adam ----
    _log_stage("Stage D: multi-start Adam (pick lowest unsupervised loss per batch)")
    rngD = np.random.default_rng(11)
    random_matrices = haar_so3(rngD, 5)
    candidate_matrices: list[tuple[float, ...] | None] = [None] + [
        tuple(float(v) for v in matrix.flatten()) for matrix in random_matrices
    ]
    winnerD = evaluate_multistart("stageD_multistart_6", baseline, candidate_matrices, search_batches)
    single_start = evaluate_config2("stageD_single_start", baseline, search_batches)
    stage_winners.append(("Stage D (multi-start, 6 candidates)", winnerD))
    stage_winners.append(("Stage D (single start, for comparison)", single_start))
    use_multistart = winnerD["mean_abs_correlation"] > single_start["mean_abs_correlation"]
    _log(
        f"  -> multi-start mean|r|={winnerD['mean_abs_correlation']:.4f} vs "
        f"single-start mean|r|={single_start['mean_abs_correlation']:.4f}; "
        f"using multi-start: {use_multistart}"
    )

    # ---- Final validation ----
    _log_stage("Final validation: full 13-batch / 65-trial reprocessing")
    if use_multistart:
        _log("  Multi-start selected -- reprocessing every batch with all 6 candidate starts and keeping the lowest-loss run.")
        all_batches_full = discover_batches()
        final_rows_by_batch = {}
        for batch in all_batches_full:
            candidate_results = [run_one(batch, replace(baseline, initial_matrix=m), tag=f"final_ms_{i}") for i, m in enumerate(candidate_matrices)]
            ok_candidates = [r for r in candidate_results if r.get("ok")]
            best = min(ok_candidates, key=lambda r: r["best_loss"]) if ok_candidates else candidate_results[0]
            final_rows_by_batch[f"{batch['session']}#{batch['batch']}"] = best
        all_rows = [row for r in final_rows_by_batch.values() if r.get("ok") for row in r["rows"]]
        abs_values = np.asarray([row["abs_correlation"] for row in all_rows], dtype=float)
        final_check = {
            "mean_abs_correlation": float(np.mean(abs_values)),
            "fisher_mean_abs_correlation": float(np.tanh(np.mean(np.arctanh(np.clip(abs_values, 0.0, 1.0 - 1e-12))))),
            "per_joint": {
                joint: float(np.mean([row["abs_correlation"] for row in all_rows if row["joint"] == joint]))
                for joint in JOINT_NAMES
            },
            "n": len(all_rows),
        }
        final_config = baseline
    else:
        final_config = baseline
        final_check = reprocess_all_batches(final_config)

    _log(
        f"  FINAL: mean|r|={final_check['mean_abs_correlation']:.4f} "
        f"fisher|r|={final_check['fisher_mean_abs_correlation']:.4f} "
        f"(base={final_check['per_joint']['base']:.3f} elbow={final_check['per_joint']['elbow']:.3f} "
        f"wrist={final_check['per_joint']['wrist']:.3f})"
    )

    write_summary(stage_winners, final_config, final_check)
    _log(f"Wrote summary to {SUMMARY_PATH}")
    _log("Round 2 complete.")


if __name__ == "__main__":
    main()
