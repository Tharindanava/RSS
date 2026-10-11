#!/usr/bin/env python3
"""Run the paired 200-case RSS loss comparison and export performance plots.

The production defaults reproduce the formal experiment configuration from
``notebooks/rss_validation.ipynb``.  Results are checkpointed after every fit,
so repeating the same command safely resumes an interrupted run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Sequence
import uuid


# Configure headless plotting before importing rss, whose public API imports
# plotting helpers.  A writable cache also avoids Matplotlib warnings on
# compute nodes whose home directory is read-only.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
if "MPLCONFIGDIR" not in os.environ:
    cache_owner = getattr(os, "getuid", lambda: "unknown")()
    matplotlib_cache = Path(tempfile.gettempdir()) / f"rss-matplotlib-{cache_owner}"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import torch  # noqa: E402

from rss import (  # noqa: E402
    BatchStudyConfig,
    RSSConfig,
    SimulationConfig,
    build_metrics_frame,
    build_source_metrics_frame,
    create_standard_case_specs,
    export_validation_case_artifacts,
    generate_family_scenario,
    run_validation_for_scenario,
    validation_metrics_payload,
)


SCHEMA_VERSION = 1
METHOD_ORDER = (
    "zero_lag_cross_correlation",
    "lagged_cross_correlation",
    "mutual_information",
    "action",
)
METHOD_LABELS = {
    "zero_lag_cross_correlation": "Zero-lag correlation",
    "lagged_cross_correlation": "Lagged correlation",
    "mutual_information": "Mutual information",
    "action": "Action",
}
METHOD_SHORT_LABELS = {
    "zero_lag_cross_correlation": "Zero-lag",
    "lagged_cross_correlation": "Lagged",
    "mutual_information": "MI",
    "action": "Action",
}
METHOD_COLORS = {
    "zero_lag_cross_correlation": "#4c78a8",
    "lagged_cross_correlation": "#f58518",
    "mutual_information": "#54a24b",
    "action": "#e45756",
}
DEFAULT_LAGGED_LAGS = (0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)
DEFAULT_SEQUENCE_LENGTHS = (720, 840, 960, 1080)
DEFAULT_INITIAL_WEIGHT_MATRIX = (
    (-0.4779489885, -0.5421200170, -0.6911372162),
    (-0.5983844737, 0.7769586228, -0.1956305705),
    (0.6430402679, 0.3200643460, -0.6957427889),
)
SOURCE_ZERO_MEAN_ATOL = 1e-12


@dataclass(frozen=True)
class MetricSpec:
    label: str
    higher_is_better: bool
    logarithmic: bool = False


COMMON_METRICS: dict[str, MetricSpec] = {
    "mean_abs_source_correlation": MetricSpec("Mean absolute source correlation", True),
    "mean_source_rmse": MetricSpec("Mean source RMSE (rad)", False, True),
    "mean_source_relative_rmse": MetricSpec("Mean relative RMSE", False, True),
    "mean_source_snr_db": MetricSpec("Mean signal-to-error ratio (dB)", True),
    "mean_axis_error_deg": MetricSpec("Mean source-axis error (deg)", False, True),
    "recovery_matrix_rmse": MetricSpec("Recovery-matrix RMSE", False, True),
    "separated_source_mean_abs_cross_correlation": MetricSpec(
        "Residual mean absolute source correlation", False, True
    ),
}
FAILURE_COLUMNS = (
    "schema_version",
    "experiment_fingerprint",
    "status",
    "failed_at_utc",
    "loss_method",
    "family",
    "case_id",
    "seed",
    "elapsed_seconds",
    "error_type",
    "error_message",
    "error_file",
)


class ExperimentError(RuntimeError):
    """Base class for expected experiment setup and integrity failures."""


class ConfigurationMismatchError(ExperimentError):
    """Raised when an output directory belongs to a different experiment."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "unknown"
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_provenance() -> dict[str, Any]:
    root = REPO_ROOT
    source_paths = [Path(__file__).resolve(), *sorted((root / "rss").glob("*.py"))]
    hashes = {
        str(path.relative_to(root)): _sha256_file(path)
        for path in source_paths
        if path.is_file()
    }
    return {"source_sha256": hashes, "combined_sha256": _fingerprint(hashes)}


def _git_provenance() -> dict[str, Any]:
    root = REPO_ROOT

    def run_git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ("git", *arguments),
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    revision = run_git("rev-parse", "HEAD")
    status = run_git("status", "--short")
    return {
        "revision": revision,
        "dirty": bool(status) if status is not None else None,
        "status_entry_count": len(status.splitlines()) if status else 0,
    }


def _resolved_device(requested: str) -> str:
    normalized = str(requested).strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ExperimentError(f"CUDA device {requested!r} was requested but CUDA is unavailable.")
    return str(device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a resumable paired comparison of zero-lag correlation, lagged "
            "correlation, histogram mutual information, and action RSS losses."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/rss_formal_loss_comparison"),
        help="Experiment output directory (default: outputs/rss_formal_loss_comparison).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the resolved experiment without writing outputs.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed fit instead of continuing the cohort.")
    parser.add_argument(
        "--export-case-signals",
        action="store_true",
        help="Also export the full existing per-case signal artifacts (substantially increases storage).",
    )
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or a CUDA device such as cuda:0.")
    parser.add_argument("--cases-per-family", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=20000)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=list(DEFAULT_SEQUENCE_LENGTHS))
    parser.add_argument("--source-cross-correlation-limit", type=float, default=0.01)

    parser.add_argument("--epochs", type=int, default=2000, help="PSO swarm iterations per fit.")
    parser.add_argument("--swarm-size", type=int, default=200, help="Particles per PSO fit.")
    parser.add_argument("--pso-inertia", type=float, default=0.9)
    parser.add_argument("--pso-cognitive", type=float, default=1.0)
    parser.add_argument("--pso-social", type=float, default=1.5)
    parser.add_argument("--pso-seed", type=int, default=0)
    parser.add_argument("--learning-rate-start", type=float, default=1e-1)
    parser.add_argument("--learning-rate-end", type=float, default=1e-3)

    parser.add_argument("--lagged-lags", type=int, nargs="+", default=list(DEFAULT_LAGGED_LAGS))
    parser.add_argument("--mi-bins", type=int, default=16)
    parser.add_argument("--mi-bandwidth", type=float, default=0.2)
    parser.add_argument("--action-inertias", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("I1", "I2", "I3"))
    parser.add_argument("--action-stiffnesses", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("K1", "K2", "K3"))
    parser.add_argument(
        "--initial-weight-matrix",
        type=float,
        nargs=9,
        default=[value for row in DEFAULT_INITIAL_WEIGHT_MATRIX for value in row],
        metavar=("A11", "A12", "A13", "A21", "A22", "A23", "A31", "A32", "A33"),
    )
    parser.add_argument("--eps", type=float, default=1e-24)
    parser.add_argument("--quaternion-eps", type=float, default=1e-24)
    parser.add_argument("--plot-dpi", type=int, default=150)
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if not 1 <= int(args.cases_per_family) <= 101:
        raise ExperimentError("--cases-per-family must be between 1 and 101.")
    if int(args.base_seed) < 0:
        raise ExperimentError("--base-seed must be >= 0.")
    if not args.sequence_lengths or any(int(value) < 8 for value in args.sequence_lengths):
        raise ExperimentError("--sequence-lengths must contain integers >= 8.")
    if int(args.epochs) < 1 or int(args.swarm_size) < 1:
        raise ExperimentError("--epochs and --swarm-size must be >= 1.")
    if int(args.pso_seed) < 0:
        raise ExperimentError("--pso-seed must be >= 0.")
    rates = (float(args.learning_rate_start), float(args.learning_rate_end))
    if not np.isfinite(rates).all() or rates[0] <= 0 or rates[1] <= 0 or rates[1] > rates[0]:
        raise ExperimentError("Learning rates must be finite and satisfy 0 < end <= start.")
    if not args.lagged_lags or any(int(lag) < 0 for lag in args.lagged_lags):
        raise ExperimentError("--lagged-lags must contain nonnegative integers.")
    if len(set(int(lag) for lag in args.lagged_lags)) != len(args.lagged_lags):
        raise ExperimentError("--lagged-lags cannot contain duplicates.")
    if max(int(lag) for lag in args.lagged_lags) >= min(int(value) for value in args.sequence_lengths):
        raise ExperimentError("Every lag must be smaller than every configured sequence length.")
    if int(args.mi_bins) < 2 or not math.isfinite(float(args.mi_bandwidth)) or float(args.mi_bandwidth) <= 0:
        raise ExperimentError("MI bins must be >= 2 and MI bandwidth must be finite and positive.")
    for name, values in (("action inertias", args.action_inertias), ("action stiffnesses", args.action_stiffnesses)):
        if len(values) != 3 or not np.isfinite(values).all() or any(float(value) <= 0 for value in values):
            raise ExperimentError(f"{name} must contain three finite positive values.")
    if not math.isfinite(float(args.source_cross_correlation_limit)) or not 0 < float(args.source_cross_correlation_limit) < 1:
        raise ExperimentError("--source-cross-correlation-limit must lie strictly between 0 and 1.")
    if int(args.plot_dpi) < 20:
        raise ExperimentError("--plot-dpi must be >= 20.")
    _resolved_device(args.device)


def _initial_matrix(args: argparse.Namespace) -> tuple[tuple[float, float, float], ...]:
    values = tuple(float(value) for value in args.initial_weight_matrix)
    return tuple(tuple(values[3 * row + column] for column in range(3)) for row in range(3))


def _validate_zero_mean_sources(scenario: Any) -> None:
    """Fail fast unless every generated and recovery source has zero temporal mean."""

    for name in ("generated_sources", "relative_sources", "sources"):
        signals = np.asarray(getattr(scenario, name), dtype=float)
        if signals.ndim != 2 or signals.shape[0] != 3 or not np.isfinite(signals).all():
            raise ExperimentError(f"Scenario {scenario.case_id} has invalid {name} data.")
        means = np.mean(signals, axis=1)
        if not np.allclose(means, 0.0, rtol=0.0, atol=SOURCE_ZERO_MEAN_ATOL):
            raise ExperimentError(
                f"Scenario {scenario.case_id} has nonzero temporal means in {name}: "
                f"{means.tolist()}."
            )


def build_method_configs(args: argparse.Namespace) -> dict[str, RSSConfig]:
    """Resolve the four planned methods to concrete RSS configurations."""

    common: dict[str, Any] = {
        "separation_weight": 1.0,
        "optimizer": "pso",
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate_start),
        "learning_rate_end": float(args.learning_rate_end),
        "pso_swarm_size": int(args.swarm_size),
        "pso_inertia": float(args.pso_inertia),
        "pso_cognitive": float(args.pso_cognitive),
        "pso_social": float(args.pso_social),
        "pso_seed": int(args.pso_seed),
        "matrix_parameterization": "so3",
        "observer_signal_model": "integrated_body_rotation",
        "initial_weight_matrix": _initial_matrix(args),
        "orthogonality_weight": 0.0,
        "reconstruction_weight": 0.0,
        "convergence_tol": 0.0,
        "eps": float(args.eps),
        "quaternion_eps": float(args.quaternion_eps),
        "device": str(args.device),
        "mi_estimator": "histogram",
        "mi_bins": int(args.mi_bins),
        "mi_bandwidth": float(args.mi_bandwidth),
        "action_inertias": tuple(float(value) for value in args.action_inertias),
        "action_stiffnesses": tuple(float(value) for value in args.action_stiffnesses),
    }
    return {
        "zero_lag_cross_correlation": RSSConfig(
            **common, loss_type="lagged_cross_correlation", lags=(0,)
        ),
        "lagged_cross_correlation": RSSConfig(
            **common,
            loss_type="lagged_cross_correlation",
            lags=tuple(int(lag) for lag in args.lagged_lags),
        ),
        "mutual_information": RSSConfig(
            **common, loss_type="mutual_information", lags=(0,)
        ),
        "action": RSSConfig(**common, loss_type="action", lags=(0,)),
    }


def _build_cohort(args: argparse.Namespace):
    study = BatchStudyConfig(
        standard_cases_per_family=int(args.cases_per_family),
        batch_sequence_lengths=tuple(int(value) for value in args.sequence_lengths),
    )
    specs = create_standard_case_specs(
        study,
        base_seed=int(args.base_seed),
        case_id_prefix="formal_",
    )
    expected_count = 4 * int(args.cases_per_family)
    if len(specs) != expected_count:
        raise ExperimentError(f"Expected {expected_count} cases, received {len(specs)}.")
    if len({spec.case_id for spec in specs}) != expected_count:
        raise ExperimentError("Formal case IDs are not unique.")
    if len({spec.seed for spec in specs}) != expected_count:
        raise ExperimentError("Formal case seeds are not unique.")
    counts = pd.Series([spec.family for spec in specs]).value_counts().to_dict()
    expected_families = set(study.standard_families)
    if set(counts) != expected_families or any(
        int(counts[family]) != int(args.cases_per_family) for family in expected_families
    ):
        raise ExperimentError(f"Formal family balance is invalid: {counts}.")
    return study, specs


def _experiment_configuration(
    args: argparse.Namespace,
    method_configs: dict[str, RSSConfig],
    specs: Sequence[Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "cohort": {
            "design": "standard_three_dense_source_families",
            "source_centering": {
                "policy": "per_source_temporal_mean",
                "absolute_tolerance": SOURCE_ZERO_MEAN_ATOL,
            },
            "base_seed": int(args.base_seed),
            "cases_per_family": int(args.cases_per_family),
            "case_count": len(specs),
            "sequence_lengths": [int(value) for value in args.sequence_lengths],
            "source_cross_correlation_limit": float(args.source_cross_correlation_limit),
            "families": list(dict.fromkeys(spec.family for spec in specs)),
            "case_ids": [spec.case_id for spec in specs],
            "case_seeds": [int(spec.seed) for spec in specs],
        },
        "methods": {
            method: _json_safe(asdict(method_configs[method])) for method in METHOD_ORDER
        },
        "method_order": list(METHOD_ORDER),
        "export_case_signals": bool(args.export_case_signals),
        "plot_dpi": int(args.plot_dpi),
    }


def _build_manifest(
    args: argparse.Namespace,
    method_configs: dict[str, RSSConfig],
    specs: Sequence[Any],
) -> dict[str, Any]:
    implementation = _implementation_provenance()
    configuration = _experiment_configuration(args, method_configs, specs)
    identity = {
        "configuration": configuration,
        "implementation_combined_sha256": implementation["combined_sha256"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "experiment_fingerprint": _fingerprint(identity),
        "configuration": configuration,
        "implementation": implementation,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "matplotlib": matplotlib.__version__,
            "scipy": scipy.__version__,
            "requested_device": str(args.device),
            "resolved_device": _resolved_device(args.device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        },
        "git": _git_provenance(),
    }


def _ensure_manifest(output_root: Path, proposed: dict[str, Any]) -> dict[str, Any]:
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExperimentError(f"Cannot read existing manifest {manifest_path}: {exc}") from exc
        if existing.get("experiment_fingerprint") != proposed.get("experiment_fingerprint"):
            raise ConfigurationMismatchError(
                "The output directory contains a different experiment configuration or "
                "implementation. Choose a new --output-root; existing results were not changed."
            )
        return existing
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(manifest_path, proposed)
    return proposed


def _checkpoint_dir(output_root: Path, method: str, spec: Any) -> Path:
    return output_root / "checkpoints" / method / spec.family / spec.case_id


def _checkpoint_path(output_root: Path, method: str, spec: Any) -> Path:
    return _checkpoint_dir(output_root, method, spec) / "checkpoint.json"


def _load_checkpoint(path: Path, *, fingerprint: str, method: str, case_id: str) -> dict[str, Any]:
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"Invalid checkpoint {path}: {exc}") from exc
    expected = {
        "schema_version": SCHEMA_VERSION,
        "experiment_fingerprint": fingerprint,
        "loss_method": method,
        "case_id": case_id,
        "status": "complete",
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise ExperimentError(f"Checkpoint integrity failure in {path}: {mismatches}")
    if not isinstance(checkpoint.get("case_metrics"), dict) or not isinstance(
        checkpoint.get("source_metrics"), list
    ):
        raise ExperimentError(f"Checkpoint {path} is missing metric rows.")
    return checkpoint


def _completed_checkpoints(
    output_root: Path,
    fingerprint: str,
    specs: Sequence[Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    checkpoints: dict[tuple[str, str], dict[str, Any]] = {}
    for spec in specs:
        for method in METHOD_ORDER:
            path = _checkpoint_path(output_root, method, spec)
            if not path.exists():
                continue
            checkpoint = _load_checkpoint(
                path,
                fingerprint=fingerprint,
                method=method,
                case_id=spec.case_id,
            )
            checkpoints[(method, spec.case_id)] = checkpoint
    return checkpoints


def _frame_row(frame: pd.DataFrame) -> dict[str, Any]:
    if len(frame) != 1:
        raise ExperimentError(f"Expected one metric row, received {len(frame)}.")
    return _json_safe(frame.iloc[0].to_dict())


def _source_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_safe(row) for row in frame.to_dict(orient="records")]


def _write_fit_checkpoint(
    *,
    output_root: Path,
    fingerprint: str,
    method: str,
    spec: Any,
    result: Any,
    elapsed_seconds: float,
    export_case_signals: bool,
) -> dict[str, Any]:
    directory = _checkpoint_dir(output_root, method, spec)
    directory.mkdir(parents=True, exist_ok=True)
    case_metrics = _frame_row(build_metrics_frame([result]))
    source_metrics = _source_rows(build_source_metrics_frame([result]))
    loss_history = pd.DataFrame(result.rss_result.loss_history)
    _atomic_write_csv(directory / "loss_history.csv", loss_history)
    _atomic_write_json(directory / "validation_metrics.json", validation_metrics_payload(result))
    if export_case_signals:
        export_validation_case_artifacts(
            result,
            directory / "full_artifacts",
            include_plots=False,
        )
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "experiment_fingerprint": fingerprint,
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "loss_method": method,
        "loss_display_name": METHOD_LABELS[method],
        "family": spec.family,
        "case_id": spec.case_id,
        "seed": int(spec.seed),
        "elapsed_seconds": float(elapsed_seconds),
        "case_metrics": case_metrics,
        "source_metrics": source_metrics,
    }
    error_path = directory / "error.json"
    if error_path.exists():
        error_path.unlink()
    # The completion marker is deliberately written after every other artifact
    # and after clearing any failure from an earlier attempt.
    _atomic_write_json(directory / "checkpoint.json", checkpoint)
    return checkpoint


def _write_fit_error(
    *,
    output_root: Path,
    fingerprint: str,
    method: str,
    spec: Any,
    elapsed_seconds: float,
    error: BaseException,
) -> None:
    directory = _checkpoint_dir(output_root, method, spec)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_fingerprint": fingerprint,
        "status": "failed",
        "failed_at_utc": _utc_now(),
        "loss_method": method,
        "family": spec.family,
        "case_id": spec.case_id,
        "seed": int(spec.seed),
        "elapsed_seconds": float(elapsed_seconds),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }
    _atomic_write_json(directory / "error.json", payload)


def _read_failures(output_root: Path, fingerprint: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    checkpoint_root = output_root / "checkpoints"
    if checkpoint_root.exists():
        for path in sorted(checkpoint_root.glob("*/*/*/error.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                rows.append({"error_file": str(path), "error_type": "UnreadableErrorFile"})
                continue
            if payload.get("experiment_fingerprint") == fingerprint:
                payload.pop("traceback", None)
                payload["error_file"] = str(path)
                rows.append(payload)
    return pd.DataFrame(rows, columns=FAILURE_COLUMNS)


def _update_run_status(
    output_root: Path,
    *,
    expected_fit_count: int,
    completed_fit_count: int,
    failed_fit_count: int,
    started_at_utc: str,
    last_fit: dict[str, Any] | None = None,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "updated_at_utc": _utc_now(),
        "started_at_utc": started_at_utc,
        "expected_fit_count": int(expected_fit_count),
        "completed_fit_count": int(completed_fit_count),
        "failed_fit_count": int(failed_fit_count),
        "remaining_fit_count": int(max(expected_fit_count - completed_fit_count, 0)),
        "complete": bool(completed_fit_count == expected_fit_count and failed_fit_count == 0),
        "last_fit": last_fit,
    }
    _atomic_write_json(output_root / "run_status.json", payload)


def _descriptive_statistics(case_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [("overall", case_metrics)]
    scopes.extend((str(family), group) for family, group in case_metrics.groupby("family", sort=False))
    for scope, scoped in scopes:
        for method in METHOD_ORDER:
            method_frame = scoped.loc[scoped["loss_method"] == method]
            for metric in COMMON_METRICS:
                values = pd.to_numeric(method_frame[metric], errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if not values.size:
                    continue
                rows.append(
                    {
                        "scope": scope,
                        "loss_method": method,
                        "metric": metric,
                        "count": int(values.size),
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
                        "min": float(np.min(values)),
                        "p05": float(np.quantile(values, 0.05)),
                        "p25": float(np.quantile(values, 0.25)),
                        "median": float(np.median(values)),
                        "p75": float(np.quantile(values, 0.75)),
                        "p95": float(np.quantile(values, 0.95)),
                        "max": float(np.max(values)),
                    }
                )
    return pd.DataFrame(rows)


def _paired_comparisons(case_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    difference_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    baseline = "zero_lag_cross_correlation"
    for metric, metric_spec in COMMON_METRICS.items():
        pivot = case_metrics.pivot(index="case_id", columns="loss_method", values=metric)
        families = case_metrics.drop_duplicates("case_id").set_index("case_id")["family"]
        if baseline in pivot.columns:
            for method in METHOD_ORDER:
                if method == baseline or method not in pivot.columns:
                    continue
                paired = pivot[[baseline, method]].dropna()
                for case_id, values in paired.iterrows():
                    raw_difference = float(values[method] - values[baseline])
                    oriented = raw_difference if metric_spec.higher_is_better else -raw_difference
                    tolerance = 1e-12 + 1e-9 * max(
                        abs(float(values[method])), abs(float(values[baseline]))
                    )
                    outcome = "tie" if abs(oriented) <= tolerance else ("win" if oriented > 0 else "loss")
                    difference_rows.append(
                        {
                            "case_id": case_id,
                            "family": families.get(case_id),
                            "metric": metric,
                            "loss_method": method,
                            "baseline_method": baseline,
                            "method_value": float(values[method]),
                            "baseline_value": float(values[baseline]),
                            "raw_difference": raw_difference,
                            "oriented_improvement": oriented,
                            "outcome": outcome,
                        }
                    )
        for case_id, values in pivot.iterrows():
            finite = values.dropna()
            if finite.empty:
                continue
            ranks = finite.rank(
                method="average", ascending=not metric_spec.higher_is_better
            )
            best_value = float(finite.max() if metric_spec.higher_is_better else finite.min())
            for method, value in finite.items():
                rank_rows.append(
                    {
                        "case_id": case_id,
                        "family": families.get(case_id),
                        "metric": metric,
                        "loss_method": method,
                        "value": float(value),
                        "rank": float(ranks[method]),
                        "is_best": bool(
                            np.isclose(float(value), best_value, rtol=1e-9, atol=1e-12)
                        ),
                    }
                )
    differences = pd.DataFrame(difference_rows)
    ranks = pd.DataFrame(rank_rows)
    return differences, ranks


def _ranking_summary(ranks: pd.DataFrame) -> pd.DataFrame:
    if ranks.empty:
        return pd.DataFrame(
            columns=("metric", "loss_method", "case_count", "mean_rank", "median_rank", "best_count", "best_rate")
        )
    rows: list[dict[str, Any]] = []
    for (metric, method), group in ranks.groupby(["metric", "loss_method"], sort=False):
        rows.append(
            {
                "metric": metric,
                "loss_method": method,
                "case_count": int(len(group)),
                "mean_rank": float(group["rank"].mean()),
                "median_rank": float(group["rank"].median()),
                "best_count": int(group["is_best"].sum()),
                "best_rate": float(group["is_best"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _runtime_summary(case_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        values = case_metrics.loc[
            case_metrics["loss_method"] == method, "fit_elapsed_seconds"
        ].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        rows.append(
            {
                "loss_method": method,
                "count": int(values.size),
                "total_seconds": float(np.sum(values)),
                "mean_seconds": float(np.mean(values)),
                "median_seconds": float(np.median(values)),
                "p25_seconds": float(np.quantile(values, 0.25)),
                "p75_seconds": float(np.quantile(values, 0.75)),
                "min_seconds": float(np.min(values)),
                "max_seconds": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def _convergence_summary(
    output_root: Path,
    checkpoints: dict[tuple[str, str], dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        histories: list[pd.Series] = []
        for (checkpoint_method, _case_id), checkpoint in checkpoints.items():
            if checkpoint_method != method:
                continue
            path = (
                output_root
                / "checkpoints"
                / method
                / str(checkpoint["family"])
                / str(checkpoint["case_id"])
                / "loss_history.csv"
            )
            if not path.exists():
                continue
            history = pd.read_csv(path, usecols=["epoch", "normalized_total_loss"])
            histories.append(history.set_index("epoch")["normalized_total_loss"].rename(checkpoint["case_id"]))
        if not histories:
            continue
        values = pd.concat(histories, axis=1)
        for epoch, row in values.iterrows():
            finite = row.to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            if not finite.size:
                continue
            rows.append(
                {
                    "loss_method": method,
                    "epoch": int(epoch),
                    "case_count": int(finite.size),
                    "p25": float(np.quantile(finite, 0.25)),
                    "median": float(np.median(finite)),
                    "p75": float(np.quantile(finite, 0.75)),
                }
            )
    return pd.DataFrame(rows)


def _save_figure(figure: Any, plots_root: Path, stem: str, dpi: int) -> None:
    plots_root.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(plots_root / f"{stem}.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _method_values(frame: pd.DataFrame, metric: str, method: str) -> np.ndarray:
    values = pd.to_numeric(
        frame.loc[frame["loss_method"] == method, metric], errors="coerce"
    ).to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _boxplot(axis: Any, values: Sequence[np.ndarray], labels: Sequence[str]) -> None:
    try:
        artists = axis.boxplot(values, tick_labels=labels, showfliers=True, patch_artist=True)
    except TypeError:  # Matplotlib < 3.9
        artists = axis.boxplot(values, labels=labels, showfliers=True, patch_artist=True)
    for patch, method in zip(artists["boxes"], METHOD_ORDER):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.65)


def _plot_overall_boxplots(case_metrics: pd.DataFrame, plots_root: Path, dpi: int) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(17, 13), squeeze=False)
    for axis, (metric, spec) in zip(axes.ravel(), COMMON_METRICS.items()):
        values = [_method_values(case_metrics, metric, method) for method in METHOD_ORDER]
        _boxplot(axis, values, [METHOD_SHORT_LABELS[method] for method in METHOD_ORDER])
        if spec.logarithmic and all(array.size and np.all(array > 0) for array in values):
            axis.set_yscale("log")
        axis.set_title(spec.label)
        axis.tick_params(axis="x", labelrotation=20)
        axis.grid(True, alpha=0.25)
    for axis in axes.ravel()[len(COMMON_METRICS):]:
        axis.set_visible(False)
    figure.suptitle("Overall paired RSS performance across formal cases")
    figure.tight_layout()
    _save_figure(figure, plots_root, "overall_performance_boxplots", dpi)


def _plot_family_performance(case_metrics: pd.DataFrame, plots_root: Path, dpi: int) -> None:
    families = list(dict.fromkeys(case_metrics["family"].astype(str)))
    figure, axes = plt.subplots(
        len(COMMON_METRICS), len(families), figsize=(4.6 * len(families), 3.7 * len(COMMON_METRICS)), squeeze=False
    )
    for row, (metric, spec) in enumerate(COMMON_METRICS.items()):
        for column, family in enumerate(families):
            axis = axes[row, column]
            family_frame = case_metrics.loc[case_metrics["family"] == family]
            values = [_method_values(family_frame, metric, method) for method in METHOD_ORDER]
            _boxplot(axis, values, [METHOD_SHORT_LABELS[method] for method in METHOD_ORDER])
            if spec.logarithmic and all(array.size and np.all(array > 0) for array in values):
                axis.set_yscale("log")
            if row == 0:
                axis.set_title(family.replace("_", " ").title())
            if column == 0:
                axis.set_ylabel(spec.label)
            axis.tick_params(axis="x", labelrotation=28, labelsize="small")
            axis.grid(True, alpha=0.25)
    figure.suptitle("RSS performance by signal family")
    figure.tight_layout()
    _save_figure(figure, plots_root, "family_stratified_performance", dpi)


def _plot_ecdfs(case_metrics: pd.DataFrame, plots_root: Path, dpi: int) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(17, 13), squeeze=False)
    for axis, (metric, spec) in zip(axes.ravel(), COMMON_METRICS.items()):
        all_positive = True
        for method in METHOD_ORDER:
            values = np.sort(_method_values(case_metrics, metric, method))
            if not values.size:
                continue
            all_positive = all_positive and bool(np.all(values > 0))
            probabilities = np.arange(1, values.size + 1, dtype=float) / float(values.size)
            axis.step(
                values,
                probabilities,
                where="post",
                label=METHOD_SHORT_LABELS[method],
                color=METHOD_COLORS[method],
            )
        if spec.logarithmic and all_positive:
            axis.set_xscale("log")
        axis.set_title(spec.label)
        axis.set_ylabel("Empirical cumulative probability")
        axis.grid(True, alpha=0.25)
    for axis in axes.ravel()[len(COMMON_METRICS):]:
        axis.set_visible(False)
    axes.ravel()[0].legend(fontsize="small")
    figure.suptitle("Empirical distributions of common recovery metrics")
    figure.tight_layout()
    _save_figure(figure, plots_root, "performance_ecdfs", dpi)


def _plot_paired_differences(differences: pd.DataFrame, plots_root: Path, dpi: int) -> None:
    if differences.empty:
        return
    competitors = METHOD_ORDER[1:]
    figure, axes = plt.subplots(3, 3, figsize=(17, 13), squeeze=False)
    for axis, (metric, spec) in zip(axes.ravel(), COMMON_METRICS.items()):
        metric_frame = differences.loc[differences["metric"] == metric]
        values = [
            metric_frame.loc[
                metric_frame["loss_method"] == method, "oriented_improvement"
            ].to_numpy(dtype=float)
            for method in competitors
        ]
        try:
            artists = axis.boxplot(
                values,
                tick_labels=[METHOD_SHORT_LABELS[method] for method in competitors],
                showfliers=True,
                patch_artist=True,
            )
        except TypeError:
            artists = axis.boxplot(
                values,
                labels=[METHOD_SHORT_LABELS[method] for method in competitors],
                showfliers=True,
                patch_artist=True,
            )
        for patch, method in zip(artists["boxes"], competitors):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_alpha(0.65)
        axis.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
        axis.set_title(spec.label)
        axis.set_ylabel("Oriented difference vs zero-lag\n(positive is better)")
        axis.tick_params(axis="x", labelrotation=20)
        axis.grid(True, alpha=0.25)
    for axis in axes.ravel()[len(COMMON_METRICS):]:
        axis.set_visible(False)
    figure.suptitle("Paired case improvements relative to zero-lag correlation")
    figure.tight_layout()
    _save_figure(figure, plots_root, "paired_differences_vs_zero_lag", dpi)


def _annotated_heatmap(axis: Any, values: np.ndarray, title: str, fmt: str, cmap: str) -> None:
    image = axis.imshow(values, aspect="auto", cmap=cmap)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                axis.text(column, row, format(value, fmt), ha="center", va="center", fontsize="small")
    axis.set_title(title)
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _plot_ranks(ranking: pd.DataFrame, plots_root: Path, dpi: int) -> None:
    if ranking.empty:
        return
    metrics = list(COMMON_METRICS)
    mean_ranks = np.full((len(metrics), len(METHOD_ORDER)), np.nan)
    best_rates = np.full_like(mean_ranks, np.nan)
    for row, metric in enumerate(metrics):
        for column, method in enumerate(METHOD_ORDER):
            selected = ranking.loc[
                (ranking["metric"] == metric) & (ranking["loss_method"] == method)
            ]
            if selected.empty:
                continue
            mean_ranks[row, column] = float(selected.iloc[0]["mean_rank"])
            best_rates[row, column] = float(selected.iloc[0]["best_rate"])
    figure, axes = plt.subplots(1, 2, figsize=(15, 7))
    _annotated_heatmap(axes[0], mean_ranks, "Mean paired rank (lower is better)", ".2f", "viridis_r")
    _annotated_heatmap(axes[1], best_rates, "Best-case rate (higher is better)", ".1%", "YlGn")
    metric_labels = [COMMON_METRICS[metric].label for metric in metrics]
    method_labels = [METHOD_SHORT_LABELS[method] for method in METHOD_ORDER]
    for axis in axes:
        axis.set_xticks(range(len(method_labels)), method_labels, rotation=25, ha="right")
        axis.set_yticks(range(len(metric_labels)), metric_labels)
    figure.suptitle("Paired method ranks and win rates")
    figure.tight_layout()
    _save_figure(figure, plots_root, "method_ranks_and_win_rates", dpi)


def _plot_runtime(case_metrics: pd.DataFrame, plots_root: Path, dpi: int) -> None:
    values = [
        case_metrics.loc[
            case_metrics["loss_method"] == method, "fit_elapsed_seconds"
        ].to_numpy(dtype=float)
        for method in METHOD_ORDER
    ]
    figure, axis = plt.subplots(figsize=(10, 6))
    _boxplot(axis, values, [METHOD_SHORT_LABELS[method] for method in METHOD_ORDER])
    if all(array.size and np.all(array > 0) for array in values):
        axis.set_yscale("log")
    axis.set_ylabel("Fit runtime (seconds, log scale when positive)")
    axis.set_title("Per-case PSO runtime by separation loss")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, plots_root, "runtime_distributions", dpi)


def _plot_convergence(convergence: pd.DataFrame, plots_root: Path, dpi: int) -> None:
    if convergence.empty:
        return
    figure, axis = plt.subplots(figsize=(11, 7))
    for method in METHOD_ORDER:
        method_frame = convergence.loc[convergence["loss_method"] == method].sort_values("epoch")
        if method_frame.empty:
            continue
        epochs = method_frame["epoch"].to_numpy(dtype=float)
        median = method_frame["median"].to_numpy(dtype=float)
        p25 = method_frame["p25"].to_numpy(dtype=float)
        p75 = method_frame["p75"].to_numpy(dtype=float)
        axis.plot(epochs, median, label=METHOD_SHORT_LABELS[method], color=METHOD_COLORS[method])
        axis.fill_between(epochs, p25, p75, color=METHOD_COLORS[method], alpha=0.18)
    axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    axis.set_xlabel("PSO iteration")
    axis.set_ylabel("Normalized best-so-far objective")
    axis.set_title("Median within-method convergence with interquartile bands")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, plots_root, "normalized_convergence", dpi)


def _performance_summary_markdown(
    case_metrics: pd.DataFrame,
    failures: pd.DataFrame,
    ranking: pd.DataFrame,
) -> str:
    lines = [
        "# Formal RSS Loss Comparison",
        "",
        f"- Completed fits: {len(case_metrics)}",
        f"- Unique cases: {case_metrics['case_id'].nunique() if not case_metrics.empty else 0}",
        f"- Failed fits awaiting retry: {len(failures)}",
        "- Mutual-information estimator: soft histogram (16 bins by default)",
        "- Raw objectives are method-specific and are not used for cross-method ranking.",
        "",
        "## Overall medians",
        "",
        "| Method | Source correlation | Source RMSE | Relative RMSE | SNR (dB) | Axis error (deg) | Mean rank |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        selected = case_metrics.loc[case_metrics["loss_method"] == method]
        overall_ranks = ranking.loc[ranking["loss_method"] == method, "mean_rank"]
        values = [
            selected["mean_abs_source_correlation"].median() if not selected.empty else np.nan,
            selected["mean_source_rmse"].median() if not selected.empty else np.nan,
            selected["mean_source_relative_rmse"].median() if not selected.empty else np.nan,
            selected["mean_source_snr_db"].median() if not selected.empty else np.nan,
            selected["mean_axis_error_deg"].median() if not selected.empty else np.nan,
            overall_ranks.mean() if not overall_ranks.empty else np.nan,
        ]
        rendered = ["NA" if not np.isfinite(value) else f"{float(value):.6g}" for value in values]
        lines.append(f"| {METHOD_LABELS[method]} | " + " | ".join(rendered) + " |")
    lines.extend(
        [
            "",
            "Normalized convergence curves show relative progress from each method's own initial objective; they do not make objective magnitudes equivalent.",
            "",
        ]
    )
    return "\n".join(lines)


def export_summaries(
    output_root: Path,
    fingerprint: str,
    checkpoints: dict[tuple[str, str], dict[str, Any]],
    *,
    plot_dpi: int,
) -> dict[str, int]:
    summary_root = output_root / "summary"
    plots_root = output_root / "plots"
    case_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        method_checkpoints = [
            checkpoint
            for (checkpoint_method, _), checkpoint in checkpoints.items()
            if checkpoint_method == method
        ]
        method_checkpoints.sort(key=lambda item: (str(item["family"]), str(item["case_id"])))
        for checkpoint in method_checkpoints:
            case_row = dict(checkpoint["case_metrics"])
            case_row.update(
                {
                    "loss_method": method,
                    "loss_display_name": METHOD_LABELS[method],
                    "fit_elapsed_seconds": float(checkpoint["elapsed_seconds"]),
                }
            )
            case_rows.append(case_row)
            for row in checkpoint["source_metrics"]:
                source_row = dict(row)
                source_row.update(
                    {
                        "loss_method": method,
                        "loss_display_name": METHOD_LABELS[method],
                        "fit_elapsed_seconds": float(checkpoint["elapsed_seconds"]),
                    }
                )
                source_rows.append(source_row)
    case_metrics = pd.DataFrame(case_rows)
    source_metrics = pd.DataFrame(source_rows)
    failures = _read_failures(output_root, fingerprint)
    _atomic_write_csv(summary_root / "per_case_metrics.csv", case_metrics)
    _atomic_write_csv(summary_root / "per_source_metrics.csv", source_metrics)
    _atomic_write_csv(summary_root / "failures.csv", failures)
    if case_metrics.empty:
        return {"case_rows": 0, "source_rows": 0, "failures": len(failures)}

    descriptive = _descriptive_statistics(case_metrics)
    differences, ranks = _paired_comparisons(case_metrics)
    ranking = _ranking_summary(ranks)
    runtime = _runtime_summary(case_metrics)
    convergence = _convergence_summary(output_root, checkpoints)
    _atomic_write_csv(summary_root / "descriptive_statistics.csv", descriptive)
    _atomic_write_csv(summary_root / "paired_case_differences.csv", differences)
    _atomic_write_csv(summary_root / "paired_case_ranks.csv", ranks)
    _atomic_write_csv(summary_root / "method_ranking_summary.csv", ranking)
    _atomic_write_csv(summary_root / "runtime_summary.csv", runtime)
    _atomic_write_csv(summary_root / "convergence_summary.csv", convergence)
    _atomic_write_text(
        summary_root / "performance_summary.md",
        _performance_summary_markdown(case_metrics, failures, ranking),
    )

    _plot_overall_boxplots(case_metrics, plots_root, plot_dpi)
    _plot_family_performance(case_metrics, plots_root, plot_dpi)
    _plot_ecdfs(case_metrics, plots_root, plot_dpi)
    _plot_paired_differences(differences, plots_root, plot_dpi)
    _plot_ranks(ranking, plots_root, plot_dpi)
    _plot_runtime(case_metrics, plots_root, plot_dpi)
    _plot_convergence(convergence, plots_root, plot_dpi)
    return {
        "case_rows": len(case_metrics),
        "source_rows": len(source_metrics),
        "failures": len(failures),
    }


def _dry_run_report(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    specs: Sequence[Any],
) -> None:
    family_counts = {
        str(family): int(count)
        for family, count in pd.Series([spec.family for spec in specs]).value_counts(sort=False).items()
    }
    total_fits = len(specs) * len(METHOD_ORDER)
    print("Formal RSS loss comparison dry run")
    print(f"Output root: {Path(args.output_root).resolve()}")
    print(f"Experiment fingerprint: {manifest['experiment_fingerprint']}")
    print(f"Cases: {len(specs)} ({family_counts})")
    print(f"Methods: {', '.join(METHOD_ORDER)}")
    print(f"Total PSO fits: {total_fits}")
    print(f"PSO budget per fit: {args.epochs} iterations x {args.swarm_size} particles")
    print(f"Particle evaluations: {total_fits * int(args.epochs) * int(args.swarm_size):,}")
    print(f"Requested/resolved device: {args.device}/{manifest['environment']['resolved_device']}")
    print("Dry run complete; no experiment output was written.")


def run_experiment(args: argparse.Namespace) -> int:
    _validate_arguments(args)
    try:
        method_configs = build_method_configs(args)
    except ValueError as exc:
        raise ExperimentError(f"Invalid RSS configuration: {exc}") from exc
    study, specs = _build_cohort(args)
    proposed_manifest = _build_manifest(args, method_configs, specs)
    output_root = Path(args.output_root).resolve()
    if args.dry_run:
        if (output_root / "manifest.json").exists():
            _ensure_manifest(output_root, proposed_manifest)
        _dry_run_report(args, proposed_manifest, specs)
        return 0

    manifest = _ensure_manifest(output_root, proposed_manifest)
    fingerprint = str(manifest["experiment_fingerprint"])
    checkpoints = _completed_checkpoints(output_root, fingerprint, specs)
    total_fits = len(specs) * len(METHOD_ORDER)
    started_at_utc = _utc_now()
    durations = [
        float(checkpoint["elapsed_seconds"])
        for checkpoint in checkpoints.values()
        if math.isfinite(float(checkpoint.get("elapsed_seconds", float("nan"))))
    ]
    existing_failures = _read_failures(output_root, fingerprint)
    _update_run_status(
        output_root,
        expected_fit_count=total_fits,
        completed_fit_count=len(checkpoints),
        failed_fit_count=len(existing_failures),
        started_at_utc=started_at_utc,
    )
    _log(
        f"Starting/resuming {len(specs)} cases x {len(METHOD_ORDER)} losses = {total_fits} fits; "
        f"{len(checkpoints)} already complete; device={manifest['environment']['resolved_device']}."
    )

    stop_requested = False
    for case_index, spec in enumerate(specs, start=1):
        missing_methods = [
            method for method in METHOD_ORDER if (method, spec.case_id) not in checkpoints
        ]
        if not missing_methods:
            continue
        scenario = generate_family_scenario(
            spec,
            SimulationConfig(
                seed=int(spec.seed),
                T=int(spec.frame_count),
                source_cross_correlation_limit=float(args.source_cross_correlation_limit),
                observation_model="body_fixed_gyro",
            ),
            study,
        )
        _validate_zero_mean_sources(scenario)
        if scenario.source_roles != ("dense", "dense", "dense") or scenario.active_source_indices != (0, 1, 2):
            raise ExperimentError(f"Formal case {spec.case_id} is not a three-active-source scenario.")
        for method in missing_methods:
            overall_index = (case_index - 1) * len(METHOD_ORDER) + METHOD_ORDER.index(method) + 1
            remaining = total_fits - len(checkpoints)
            mean_duration = float(np.mean(durations)) if durations else None
            eta = mean_duration * remaining if mean_duration is not None else None
            _log(
                f"[{overall_index}/{total_fits}] fitting {spec.case_id} ({spec.family}) with {method}; "
                f"complete={len(checkpoints)}, ETA={_format_duration(eta)}."
            )
            fit_started = time.perf_counter()
            try:
                result = run_validation_for_scenario(scenario, method_configs[method])
                elapsed = time.perf_counter() - fit_started
                checkpoint = _write_fit_checkpoint(
                    output_root=output_root,
                    fingerprint=fingerprint,
                    method=method,
                    spec=spec,
                    result=result,
                    elapsed_seconds=elapsed,
                    export_case_signals=bool(args.export_case_signals),
                )
                checkpoints[(method, spec.case_id)] = checkpoint
                durations.append(elapsed)
                _log(
                    f"Completed {spec.case_id}/{method} in {_format_duration(elapsed)}; "
                    f"mean |corr|={result.metrics['mean_abs_source_correlation']:.6f}, "
                    f"RMSE={result.metrics['mean_source_rmse']:.6g}."
                )
            except KeyboardInterrupt:
                _log("Interrupted; the current incomplete fit will be retried on resume.")
                stop_requested = True
                break
            except Exception as exc:  # noqa: BLE001 - each long-running fit must be isolated
                elapsed = time.perf_counter() - fit_started
                _write_fit_error(
                    output_root=output_root,
                    fingerprint=fingerprint,
                    method=method,
                    spec=spec,
                    elapsed_seconds=elapsed,
                    error=exc,
                )
                _log(f"FAILED {spec.case_id}/{method} after {_format_duration(elapsed)}: {type(exc).__name__}: {exc}")
                if args.fail_fast:
                    stop_requested = True
            failures = _read_failures(output_root, fingerprint)
            _update_run_status(
                output_root,
                expected_fit_count=total_fits,
                completed_fit_count=len(checkpoints),
                failed_fit_count=len(failures),
                started_at_utc=started_at_utc,
                last_fit={"case_id": spec.case_id, "family": spec.family, "loss_method": method},
            )
            if stop_requested:
                break
        if stop_requested:
            break

    _log("Building combined summaries and headless performance plots.")
    summary_counts = export_summaries(
        output_root,
        fingerprint,
        checkpoints,
        plot_dpi=int(args.plot_dpi),
    )
    failures = _read_failures(output_root, fingerprint)
    is_complete = len(checkpoints) == total_fits and failures.empty
    _update_run_status(
        output_root,
        expected_fit_count=total_fits,
        completed_fit_count=len(checkpoints),
        failed_fit_count=len(failures),
        started_at_utc=started_at_utc,
    )
    _log(
        f"Summary complete: {summary_counts['case_rows']} case rows, "
        f"{summary_counts['source_rows']} source rows, {len(failures)} failures."
    )
    if is_complete:
        _log(f"Formal comparison completed successfully: {output_root}")
        return 0
    _log(
        "Run is incomplete. Repeat the identical command to retry failed or interrupted fits; "
        f"completed checkpoints: {len(checkpoints)}/{total_fits}."
    )
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_experiment(args)
    except ExperimentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
