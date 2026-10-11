"""Linear-grid correlation sensitivity sweep (rho = 0, 0.05, ..., 0.95).

Same scenarios/methods/configs as the notebook's log-grid study (equal seeds, 4 families,
10 replicates) but a linear rho grid.  rho is capped at 0.95 by CorrelationStudyConfig (rho=1
makes the three sources identical, so separation is undefined).  Writes only under
paper_figures/correlation_sensitivity_linear/.  Resumable: finished cases are skipped.
"""
import os
os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MPLCONFIGDIR="/tmp/rss_lin_mpl")
import json, sys, traceback, contextlib, io
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "notebooks"))
OUT = Path(__file__).resolve().parent / "correlation_sensitivity_linear"
CASES = OUT / "case_metrics"
GRID = tuple(round(0.05 * i, 2) for i in range(20))
NS = None


def init():
    import torch
    torch.set_num_threads(1)


def work(spec):
    dest = CASES / f"{spec.case_id}.json"
    if dest.exists():
        return spec.case_id, "skip"
    ns = NS
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            case = ns["generate_correlation_scenario"](spec, ns["batch_sim_config"], ns["batch_study_config"],
                                                       correlation_tolerance=1e-6)
            run = ns["run_specialized_validation_for_scenario"](case, ns["batch_rss_config"],
                                                                ns["DEFAULT_SPECIALIZED_LOSS_CONFIG"])
            bss = ns["run_bss_methods_for_scenario"](case, ns["bss_config"])
        rows = [("rss_specialized", run.validation.metrics)] + [(b.fit_result.method_key, b.metrics) for b in bss]
        payload = {"case_id": spec.case_id, "family": spec.family, "replicate": spec.replicate_index,
                   "rho": spec.target_source_correlation,
                   "methods": {k: {"mean_abs_source_correlation": float(m["mean_abs_source_correlation"]),
                                   "mean_source_rmse": float(m["mean_source_rmse"])} for k, m in rows}}
        dest.write_text(json.dumps(payload))
        return spec.case_id, "ok"
    except Exception:
        return spec.case_id, "FAIL " + traceback.format_exc()[-400:]


def main():
    CASES.mkdir(parents=True, exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    import run_specialized_correlation_resume as R
    global NS
    with contextlib.redirect_stdout(io.StringIO()):
        ns = NS = R.execute_notebook_setup()
    cfg = replace(ns["correlation_study_config"], target_correlations=GRID)
    specs = ns["create_correlation_case_specs"](cfg, ns["batch_study_config"])
    todo = [s for s in specs if not (CASES / f"{s.case_id}.json").exists()]
    print(f"{len(specs)} cases, {len(todo)} to run on {workers} workers", flush=True)
    done = 0
    with get_context("fork").Pool(workers, initializer=init) as pool:
        for cid, status in pool.imap_unordered(work, todo, chunksize=1):
            done += 1
            if status.startswith("FAIL"):
                print(cid, status, flush=True)
            if done % 20 == 0:
                print(f"{done}/{len(todo)}", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
