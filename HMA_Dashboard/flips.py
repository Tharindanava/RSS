"""Transactional group-level RSS sign commits from the dashboard."""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gait_analysis.references import is_reference_source, load_registry
from hilbert_analysis.columns import axis_columns
from hilbert_analysis.discovery import SIBLING_MODE_FILES
from HMA_Dashboard import catalog, signals, te_matrix, transfer_entropy


FLIP_LOG_NAME = "hma_dashboard_manual_flips.csv"
BACKUP_ROOT_NAME = "hma_dashboard_flip_backup"
COMMITTABLE_TRACES: tuple[str, ...] = ("mode_1", "mode_2", "mode_3")
LOG_COLUMNS: tuple[str, ...] = (
    "timestamp_utc", "operation_id", "mode_file", "axes_file", "group_name", "trace_name",
    "angle_column", "axis_columns", "rows", "note",
)
ORIGINAL_ANGLE_REFUSAL = (
    "'Original Angle' is a single-signed magnitude and is never previewed or changed by a group flip."
)
_WRITE_LOCK = threading.Lock()
# Isolated so tests can inject a one-shot replacement failure while rollback
# continues to use the operating system primitive.
_replace = os.replace


@dataclass(frozen=True)
class FlipRequest:
    """One group-level flip; ``trace_name`` is retained for UI compatibility."""

    mode_file_path: Path
    group_name: str
    trace_name: str = "mode_1"


@dataclass(frozen=True)
class FlipOutcome:
    applied: bool
    message: str
    angle_column: str = ""
    angle_column_names: tuple[str, ...] = ()
    axis_column_names: tuple[str, ...] = ()
    mode_file: str = ""
    axes_file: str = ""
    backup_dir: str = ""
    recovery_dir: str = ""
    log_path: str = ""
    #: Matches ``operation_id`` in the leaf audit log; set on success only.
    operation_id: str = ""
    net_parity: int = 1
    rows: int = 0
    unhandled: dict[str, list[str]] = field(default_factory=dict)


def flip_key(mode_file_path: str | Path, group_name: str, trace_name: str | None = None) -> str:
    """Stable preview identity for a source file and anatomical group."""
    return f"{Path(mode_file_path).resolve()}|{group_name}"


def parse_flip_key(key: str) -> tuple[str, str]:
    return tuple(str(key).rsplit("|", 1))  # type: ignore[return-value]


def is_committable(trace_name: str) -> tuple[bool, str]:
    if trace_name == "original_angle":
        return False, ORIGINAL_ANGLE_REFUSAL
    if trace_name not in COMMITTABLE_TRACES:
        return False, f"Trace '{trace_name}' cannot be flipped on disk."
    return True, ""


def mode_index(trace_name: str) -> int:
    return int(str(trace_name).rsplit("_", 1)[1])


def axes_file_for(mode_file_path: str | Path) -> Path | None:
    path = Path(mode_file_path)
    for family, mode_name in catalog.MODE_FILENAMES.items():
        if path.name == mode_name:
            return path.with_name(catalog.AXES_FILENAMES[family])
    return None


def backup_root_for(results_dir: str | Path) -> Path:
    return Path(results_dir).resolve() / BACKUP_ROOT_NAME


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [column.strip() for column in next(csv.reader(handle), [])]


def unhandled_sign_columns(leaf: Path, group_name: str, mode: int) -> dict[str, list[str]]:
    prefix = f"{group_name}_mode_{mode}"
    known = set(axis_columns(group_name, mode))
    findings: dict[str, list[str]] = {}
    for sibling in SIBLING_MODE_FILES:
        sibling_path = leaf / sibling
        if not sibling_path.is_file():
            continue
        try:
            header = _read_header(sibling_path)
        except OSError:
            continue
        extra = [column for column in header if column.startswith(prefix) and column not in known]
        if extra:
            findings[sibling] = extra
    return findings


def read_flip_log(mode_file_path: str | Path) -> pd.DataFrame:
    log_path = Path(mode_file_path).parent / FLIP_LOG_NAME
    if not log_path.is_file():
        return pd.DataFrame(columns=list(LOG_COLUMNS))
    try:
        return pd.read_csv(log_path, encoding="utf-8-sig")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=list(LOG_COLUMNS))


def count_manual_flips(mode_file_path: str | Path, group_name: str, trace_name: str = "mode_1") -> int:
    log = read_flip_log(mode_file_path)
    if log.empty:
        return 0
    return int(len(log[(log["group_name"] == group_name) & (log["trace_name"] == trace_name)]))


def net_flip_parity(mode_file_path: str | Path, group_name: str, trace_name: str = "mode_1") -> int:
    return -1 if count_manual_flips(mode_file_path, group_name, trace_name) % 2 else 1


def _write_frame_temp(frame: pd.DataFrame, target: Path) -> Path:
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    frame.to_csv(temporary, index=False)
    return temporary


def _write_log_temp(log: pd.DataFrame, target: Path) -> Path:
    return _write_frame_temp(log.reindex(columns=LOG_COLUMNS), target)


def _restore_file(source: Path, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.rollback.{uuid.uuid4().hex}")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def _validate_source(mode_path: Path, results_root: Path) -> str | None:
    if catalog.is_backup_path(mode_path):
        return f"Refusing to change a backup path: {mode_path}"
    try:
        mode_path.relative_to(results_root)
    except ValueError:
        return f"Refusing to change a CSV outside the configured results directory: {mode_path}"
    return None


def commit_manual_flip(
    request: FlipRequest, *, results_dir: str | Path, backup_root: str | Path | None = None,
    reference_registry: str | Path | None = None, now: datetime | None = None,
) -> FlipOutcome:
    """Negate Modes 1-3 and their nine axis components as one transaction."""
    mode_path = Path(request.mode_file_path).resolve()
    results_root = Path(results_dir).resolve()
    refusal = _validate_source(mode_path, results_root)
    if refusal:
        return FlipOutcome(False, refusal, mode_file=str(mode_path))
    angle_names = tuple(signals.trace_column_name(request.group_name, trace) for trace in COMMITTABLE_TRACES)
    axis_names = tuple(column for mode in range(1, 4) for column in axis_columns(request.group_name, mode))
    with _WRITE_LOCK:
        if not mode_path.is_file():
            return FlipOutcome(False, f"Mode file no longer exists: {mode_path}", mode_file=str(mode_path))
        axes_path = axes_file_for(mode_path)
        if axes_path is None or not axes_path.is_file() or catalog.is_backup_path(axes_path):
            return FlipOutcome(False, "The matching non-backup axes CSV is missing; nothing was written.", mode_file=str(mode_path))
        registry_path = Path(reference_registry).resolve() if reference_registry else results_root / "hma_dashboard_references.json"
        try:
            if is_reference_source(load_registry(registry_path), mode_path):
                return FlipOutcome(False, "This recording is a protected reference. Replace or clear that reference before committing a flip.", mode_file=str(mode_path), axes_file=str(axes_path))
        except (OSError, ValueError) as error:
            return FlipOutcome(False, f"The reference registry could not be validated: {error}", mode_file=str(mode_path), axes_file=str(axes_path))
        mode_header, axes_header = _read_header(mode_path), _read_header(axes_path)
        missing_angles = [column for column in angle_names if column not in mode_header]
        missing_axes = [column for column in axis_names if column not in axes_header]
        if missing_angles or missing_axes:
            details = []
            if missing_angles:
                details.append("angles: " + ", ".join(missing_angles))
            if missing_axes:
                details.append("axes: " + ", ".join(missing_axes))
            return FlipOutcome(False, "Required group columns are missing (" + "; ".join(details) + "); nothing was written.", mode_file=str(mode_path), axes_file=str(axes_path))
        unhandled: dict[str, list[str]] = {}
        for mode in range(1, 4):
            for filename, columns in unhandled_sign_columns(mode_path.parent, request.group_name, mode).items():
                unhandled.setdefault(filename, []).extend(columns)
        if unhandled:
            detail = "; ".join(f"{name}: {', '.join(columns)}" for name, columns in unhandled.items())
            return FlipOutcome(False, f"Other exports contain unhandled sign columns ({detail}); nothing was written.", mode_file=str(mode_path), axes_file=str(axes_path), unhandled=unhandled)
        try:
            mode_frame = pd.read_csv(mode_path, encoding="utf-8-sig")
            axes_frame = pd.read_csv(axes_path, encoding="utf-8-sig")
            mode_before = mode_frame.loc[:, list(angle_names)].apply(pd.to_numeric, errors="raise").to_numpy(float)
            axes_before = axes_frame.loc[:, list(axis_names)].apply(pd.to_numeric, errors="raise").to_numpy(float)
        except (OSError, KeyError, ValueError) as error:
            return FlipOutcome(False, f"Source validation failed: {error}; nothing was written.", mode_file=str(mode_path), axes_file=str(axes_path))
        if len(mode_frame) != len(axes_frame):
            return FlipOutcome(False, "Mode and axes CSV row counts differ; nothing was written.", mode_file=str(mode_path), axes_file=str(axes_path))
        root = Path(backup_root).resolve() if backup_root else backup_root_for(results_root)
        relative = mode_path.parent.relative_to(results_root)
        backup_dir = root / relative
        operation_id = uuid.uuid4().hex
        recovery_dir = backup_dir / ".operations" / operation_id
        log_path = mode_path.parent / FLIP_LOG_NAME
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            recovery_dir.mkdir(parents=True, exist_ok=False)
            for source in (mode_path, axes_path):
                one_time = backup_dir / source.name
                if not one_time.exists():
                    shutil.copy2(source, one_time)
                shutil.copy2(source, recovery_dir / source.name)
            if log_path.exists():
                shutil.copy2(log_path, recovery_dir / log_path.name)
        except OSError as error:
            return FlipOutcome(False, f"Could not create recovery copies, so nothing was written: {error}", mode_file=str(mode_path), axes_file=str(axes_path), backup_dir=str(backup_dir))
        timestamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
        log = read_flip_log(mode_path)
        log_rows = [{
            "timestamp_utc": timestamp, "operation_id": operation_id, "mode_file": mode_path.name,
            "axes_file": axes_path.name, "group_name": request.group_name, "trace_name": trace,
            "angle_column": angle, "axis_columns": " ".join(axis_columns(request.group_name, mode)),
            "rows": len(mode_frame), "note": "manual dashboard three-mode group flip",
        } for mode, (trace, angle) in enumerate(zip(COMMITTABLE_TRACES, angle_names), start=1)]
        log_after = pd.concat((log.reindex(columns=LOG_COLUMNS), pd.DataFrame(log_rows)), ignore_index=True)
        mode_frame.loc[:, list(angle_names)] = -mode_before
        axes_frame.loc[:, list(axis_names)] = -axes_before
        temporary_files: list[Path] = []
        log_existed = log_path.exists()
        try:
            mode_temp = _write_frame_temp(mode_frame, mode_path)
            temporary_files.append(mode_temp)
            axes_temp = _write_frame_temp(axes_frame, axes_path)
            temporary_files.append(axes_temp)
            log_temp = _write_log_temp(log_after, log_path)
            temporary_files.append(log_temp)
            _replace(mode_temp, mode_path)
            _replace(axes_temp, axes_path)
            _replace(log_temp, log_path)
            verified_mode = pd.read_csv(mode_path, usecols=list(angle_names)).to_numpy(float)
            verified_axes = pd.read_csv(axes_path, usecols=list(axis_names)).to_numpy(float)
            if not np.allclose(verified_mode, -mode_before, equal_nan=True) or not np.allclose(verified_axes, -axes_before, equal_nan=True):
                raise OSError("post-write verification did not match the requested negation")
            verified_log = pd.read_csv(log_path)
            if operation_id not in set(verified_log.get("operation_id", [])):
                raise OSError("audit-log verification failed")
        except Exception as error:
            rollback_errors: list[str] = []
            for filename, target in ((mode_path.name, mode_path), (axes_path.name, axes_path)):
                try:
                    _restore_file(recovery_dir / filename, target)
                except OSError as rollback_error:
                    rollback_errors.append(str(rollback_error))
            try:
                if log_existed:
                    _restore_file(recovery_dir / log_path.name, log_path)
                else:
                    log_path.unlink(missing_ok=True)
            except OSError as rollback_error:
                rollback_errors.append(str(rollback_error))
            detail = f" Rollback errors: {'; '.join(rollback_errors)}" if rollback_errors else " The complete operation was rolled back."
            return FlipOutcome(False, f"Writing the flip failed: {error}.{detail}", mode_file=str(mode_path), axes_file=str(axes_path), backup_dir=str(backup_dir), recovery_dir=str(recovery_dir))
        finally:
            for temporary in temporary_files:
                temporary.unlink(missing_ok=True)
        clear_dependent_caches()
        catalog.scan_results_catalog(results_root)
        parity = net_flip_parity(mode_path, request.group_name, "mode_1")
    columns = ", ".join((*angle_names, *axis_names))
    return FlipOutcome(
        True, f"Saved group flip to source CSVs: {mode_path} and {axes_path}. Changed columns: {columns}",
        angle_column=angle_names[0], angle_column_names=angle_names, axis_column_names=axis_names,
        mode_file=str(mode_path), axes_file=str(axes_path), backup_dir=str(backup_dir),
        recovery_dir=str(recovery_dir), log_path=str(log_path), operation_id=operation_id,
        net_parity=parity, rows=len(mode_frame),
    )


def clear_dependent_caches() -> None:
    signals.clear_caches()
    catalog.clear_caches()
    transfer_entropy.clear_caches()
    te_matrix.clear_caches()


def describe_flip_state(mode_file_path: str | Path, group_name: str, trace_name: str, *, previewed: bool) -> str:
    if trace_name == "original_angle":
        return "Original Angle is unchanged by group previews and source commits"
    parts = ["Preview: Modes 1-3 inverted (not saved)" if previewed else "Preview: Modes 1-3 as stored"]
    flips = count_manual_flips(mode_file_path, group_name, trace_name)
    parts.append(f"on disk: {'inverted' if flips % 2 else 'stored parity'} after {flips} group flip(s)" if flips else "on disk: never manually flipped")
    return " · ".join(parts)
