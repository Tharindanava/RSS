#!/usr/bin/env python3
"""Resume the specialized-loss correlation sweep without rerunning completed cases.

The specialized objectives are intentionally notebook-local.  To guarantee that a
resumed run uses the exact implementation and configuration that produced the
existing artifacts, this driver executes only the definition/configuration cells
from ``rss_validation_specialized_losses.ipynb`` and then processes missing case
IDs.  It refuses partial cases, because silently overwriting only part of a six-
method comparison would compromise the paired design.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import types

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "rss_validation_specialized_losses.ipynb"
ARTIFACT_ROOT = REPO_ROOT / "notebooks" / "artifacts" / "rss_validation_specialized_losses"
ORIGINAL_ROOT = REPO_ROOT / "notebooks" / "artifacts" / "rss_validation"
METHOD_PATHS = {
    "rss_corr_baseline": ARTIFACT_ROOT / "rss_corr_baseline" / "correlation_sensitivity" / "cases",
    "rss_specialized": ARTIFACT_ROOT / "rss_specialized" / "correlation_sensitivity" / "cases",
    "pca": ARTIFACT_ROOT / "bss_baselines" / "pca" / "correlation_sensitivity" / "cases",
    "jade_ica": ARTIFACT_ROOT / "bss_baselines" / "jade_ica" / "correlation_sensitivity" / "cases",
    "fastica": ARTIFACT_ROOT / "bss_baselines" / "fastica" / "correlation_sensitivity" / "cases",
    "sobi": ARTIFACT_ROOT / "bss_baselines" / "sobi" / "correlation_sensitivity" / "cases",
}
ORIGINAL_METHOD_PATHS = {
    "rss_corr_baseline": ORIGINAL_ROOT / "correlation_sensitivity" / "cases",
    "pca": ORIGINAL_ROOT / "bss_baselines" / "pca" / "correlation_sensitivity" / "cases",
    "jade_ica": ORIGINAL_ROOT / "bss_baselines" / "jade_ica" / "correlation_sensitivity" / "cases",
    "fastica": ORIGINAL_ROOT / "bss_baselines" / "fastica" / "correlation_sensitivity" / "cases",
    "sobi": ORIGINAL_ROOT / "bss_baselines" / "sobi" / "correlation_sensitivity" / "cases",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Audit and list missing cases only")
    parser.add_argument(
        "--max-cases", type=int, default=None,
        help="Process at most this many missing scenarios (useful for controlled batches)",
    )
    parser.add_argument("--shard-count", type=int, default=1, help="Number of non-overlapping missing-case shards")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard to process")
    return parser.parse_args()


def execute_notebook_setup() -> dict[str, object]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    module_name = "rss_specialized_resume_setup"
    module = types.ModuleType(module_name)
    module.__file__ = str(NOTEBOOK)
    sys.modules[module_name] = module
    namespace = module.__dict__
    # Imports/audits, specialized implementation, single-case configuration,
    # batch configuration, BSS helpers/configuration, correlation configuration.
    for index in (2, 4, 8, 15, 38, 50):
        source = "".join(notebook["cells"][index].get("source", []))
        exec(compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)
    return namespace


def case_state(case_id: str) -> tuple[bool, list[str]]:
    present = [key for key, root in METHOD_PATHS.items() if (root / case_id / "metrics.json").is_file()]
    if present and len(present) != len(METHOD_PATHS):
        missing = sorted(set(METHOD_PATHS) - set(present))
        raise RuntimeError(
            f"Partial case {case_id}: present={sorted(present)}, missing={missing}. "
            "Resolve the partial artifact explicitly before resuming."
        )
    return len(present) == len(METHOD_PATHS), present


def validate_reusable_original_case(case: object, case_id: str) -> None:
    """Prove that the original-run scenario matches before reusing fitted baselines."""
    source_path = ORIGINAL_METHOD_PATHS["rss_corr_baseline"] / case_id / "source_comparison.csv"
    observation_path = ORIGINAL_METHOD_PATHS["rss_corr_baseline"] / case_id / "observed_signals.csv"
    if not source_path.is_file() or not observation_path.is_file():
        raise RuntimeError(f"Missing original correlation artifacts for {case_id}")
    sources = pd.read_csv(source_path)[[f"true_source_{index}" for index in range(1, 4)]].to_numpy().T
    observations = pd.read_csv(observation_path)[[f"observer_{index}" for index in range(1, 4)]].to_numpy().T
    if not np.allclose(case.sources, sources, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"Original source waveforms do not match regenerated scenario {case_id}")
    if not np.allclose(case.observations, observations, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"Original observations do not match regenerated scenario {case_id}")
    for method_key, root in ORIGINAL_METHOD_PATHS.items():
        if not (root / case_id / "metrics.json").is_file():
            raise RuntimeError(f"Missing reusable {method_key} artifact for {case_id}")


def copy_original_baselines(case_id: str) -> None:
    for method_key, source_root in ORIGINAL_METHOD_PATHS.items():
        destination = METHOD_PATHS[method_key] / case_id
        if destination.exists():
            raise RuntimeError(f"Refusing to overwrite existing destination {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_root / case_id, destination)


def main() -> int:
    args = parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("Require shard-count >= 1 and 0 <= shard-index < shard-count")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/rss-report-mpl")
    ns = execute_notebook_setup()
    specs = ns["create_correlation_case_specs"](
        ns["correlation_study_config"], ns["batch_study_config"]
    )
    completed: list[object] = []
    missing: list[object] = []
    for spec in specs:
        (completed if case_state(spec.case_id)[0] else missing).append(spec)

    print(json.dumps({
        "expected_scenarios": len(specs),
        "completed_scenarios": len(completed),
        "missing_scenarios": len(missing),
        "first_missing": [item.case_id for item in missing[:5]],
    }, indent=2))
    if args.dry_run or not missing:
        return 0

    selected = [item for index, item in enumerate(missing) if index % args.shard_count == args.shard_index]
    if args.max_cases is not None:
        selected = selected[: args.max_cases]
    print(f"Processing shard {args.shard_index + 1}/{args.shard_count}: {len(selected)} scenarios")
    for index, spec in enumerate(selected, start=1):
        case = ns["generate_correlation_scenario"](
            spec,
            ns["batch_sim_config"],
            ns["batch_study_config"],
            correlation_tolerance=float(ns["correlation_study_config"].correlation_tolerance),
        )
        ns["validate_zero_mean_sources"](case)
        # Filtered-noise generation changed between notebook runs, so its outputs
        # are never reused.  All missing cases are currently chirp/piecewise; for
        # those families the scenario identity is verified numerically per case.
        reuse_original = spec.family != "filtered_noise"
        if reuse_original:
            validate_reusable_original_case(case, spec.case_id)
            specialized_run = ns["run_specialized_validation_for_scenario"](
                case, ns["batch_rss_config"], ns["DEFAULT_SPECIALIZED_LOSS_CONFIG"]
            )
            copy_original_baselines(spec.case_id)
        else:
            corr_run, specialized_run = ns["run_paired_rss_for_scenario"](
                case, ns["batch_rss_config"], ns["DEFAULT_SPECIALIZED_LOSS_CONFIG"]
            )
            bss_results = ns["run_bss_methods_for_scenario"](case, ns["bss_config"])
            ns["export_experiment_run"](
                corr_run, METHOD_PATHS["rss_corr_baseline"] / spec.case_id, include_plots=False
            )
            for result in bss_results:
                ns["export_bss_case_artifacts"](
                    result, METHOD_PATHS[result.fit_result.method_key] / spec.case_id, include_plots=False
                )
        ns["export_experiment_run"](
            specialized_run, METHOD_PATHS["rss_specialized"] / spec.case_id, include_plots=False
        )
        print(
            f"Resumed correlation case {index}/{len(selected)}: "
            f"{spec.case_id}, rho={spec.target_source_correlation:g}, "
            f"reused_original_baselines={reuse_original}",
            flush=True,
        )

    print(f"Completed {len(selected)} previously missing scenarios.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
