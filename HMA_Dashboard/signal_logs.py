"""Append-only dashboard logs: faulty-signal reports and flip history.

Both files live at the root of the results directory, next to
``hma_dashboard_references.json``, so every report and flip decision for a
dataset can be read in one place. They are separate on purpose: a report says
"this signal is unusable", the flip history says "this is what was done to its
sign", and either is useful without the other.

``hma_dashboard_manual_flips.csv`` inside each run leaf stays the transactional
audit that ``flips`` reads for net parity and restores on rollback. The flip
history here is the dashboard-wide timeline: it also records previews, which
never touch a leaf, and links each commit to its leaf audit by ``operation_id``.
"""

from __future__ import annotations

import csv
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


FAULT_LOG_NAME = "hma_dashboard_faulty_signal_reports.csv"
FLIP_HISTORY_NAME = "hma_dashboard_flip_history.csv"

#: ``(code, label)``; the code is what lands in the log, so keep codes stable.
FAULT_REASONS: tuple[tuple[str, str], ...] = (
    ("flat_or_dropout", "Flat line / sensor dropout"),
    ("noise_or_spikes", "Noise or spike artefacts"),
    ("drift_or_offset", "Drift or offset"),
    ("bad_separation", "Poor source separation (modes mixed)"),
    ("implausible_waveform", "Implausible waveform / no clear steps"),
    ("other", "Other (describe in the note)"),
)
FAULT_REASON_LABELS: dict[str, str] = dict(FAULT_REASONS)
NOTE_MAX_CHARS = 500

FLIP_EVENTS: tuple[str, ...] = ("preview_on", "preview_off", "commit")

IDENTITY_COLUMNS: tuple[str, ...] = (
    "subject", "terrain", "activity", "experiment", "csv_file", "mode_file", "group_name", "trace_name",
)
FAULT_COLUMNS: tuple[str, ...] = (
    "timestamp_utc", "report_id", *IDENTITY_COLUMNS,
    "reason_code", "reason", "note", "previewed_flip", "committed_flips", "view_scope",
)
FLIP_HISTORY_COLUMNS: tuple[str, ...] = (
    "timestamp_utc", "event", *IDENTITY_COLUMNS, "view_scope", "operation_id", "detail",
)
_LOG_LOCK = threading.Lock()


@dataclass(frozen=True)
class FaultReportOutcome:
    ok: bool
    message: str
    report_id: str = ""
    log_path: str = ""
    #: Reports now on file for this signal group, including this one.
    report_count: int = 0


@dataclass(frozen=True)
class FlipEvent:
    """One flip-history row; ``record`` supplies the hierarchy columns."""

    event: str
    mode_file_path: str
    group_name: str
    trace_name: str = ""
    record: Mapping[str, Any] | None = None
    view_scope: str = ""
    operation_id: str = ""
    detail: str = ""


def fault_log_path(results_dir: str | Path) -> Path:
    return Path(results_dir).resolve() / FAULT_LOG_NAME


def flip_history_path(results_dir: str | Path) -> Path:
    return Path(results_dir).resolve() / FLIP_HISTORY_NAME


def _timestamp(now: datetime | None) -> str:
    return (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")


def _relative_mode_file(mode_file_path: str | Path, results_dir: str | Path) -> str:
    path = Path(mode_file_path).resolve()
    try:
        return path.relative_to(Path(results_dir).resolve()).as_posix()
    except ValueError:
        return str(path)


def signal_identity(
    record: Mapping[str, Any] | None,
    mode_file_path: str | Path,
    group_name: str,
    trace_name: str,
    results_dir: str | Path,
) -> dict[str, str]:
    """The columns that say which signal a row is about."""
    record = record or {}
    return {
        **{field: str(record.get(field, "")) for field in ("subject", "terrain", "activity", "experiment", "csv_file")},
        "mode_file": _relative_mode_file(mode_file_path, results_dir),
        "group_name": str(group_name),
        "trace_name": str(trace_name),
    }


def _header(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [column.strip() for column in next(csv.reader(handle), [])]


def _append_rows(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with _LOG_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = _header(path)
        if existing and existing != list(columns):
            # Appending under a different header would silently misalign every
            # later row, so refuse and let the caller tell the user.
            raise ValueError(f"{path.name} has an unexpected header; move it aside so a fresh log can start.")
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
            if not existing:
                writer.writeheader()
            writer.writerows(rows)


def _read_log(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=list(columns))
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=list(columns))


def read_fault_reports(results_dir: str | Path) -> pd.DataFrame:
    return _read_log(fault_log_path(results_dir), FAULT_COLUMNS)


def read_flip_history(results_dir: str | Path) -> pd.DataFrame:
    return _read_log(flip_history_path(results_dir), FLIP_HISTORY_COLUMNS)


def count_fault_reports(results_dir: str | Path, mode_file_path: str | Path, group_name: str) -> int:
    log = read_fault_reports(results_dir)
    if log.empty:
        return 0
    mode_file = _relative_mode_file(mode_file_path, results_dir)
    return int(((log["mode_file"] == mode_file) & (log["group_name"] == group_name)).sum())


def record_fault_report(
    results_dir: str | Path,
    *,
    record: Mapping[str, Any] | None,
    mode_file_path: str | Path,
    group_name: str,
    trace_name: str,
    reason: str | None,
    note: str | None,
    previewed_flip: bool = False,
    committed_flips: int = 0,
    view_scope: str = "",
    now: datetime | None = None,
) -> FaultReportOutcome:
    """Validate and append one faulty-signal report; nothing is written on refusal."""
    if reason not in FAULT_REASON_LABELS:
        return FaultReportOutcome(False, "Choose a reason for the report.")
    # One physical line per report keeps the log greppable.
    clean_note = re.sub(r"\s+", " ", str(note or "")).strip()
    if reason == "other" and not clean_note:
        return FaultReportOutcome(False, "Describe the problem in the note when the reason is 'Other'.")
    if len(clean_note) > NOTE_MAX_CHARS:
        return FaultReportOutcome(False, f"The note is {len(clean_note)} characters; the limit is {NOTE_MAX_CHARS}.")
    report_id = uuid.uuid4().hex
    row = {
        "timestamp_utc": _timestamp(now),
        "report_id": report_id,
        **signal_identity(record, mode_file_path, group_name, trace_name, results_dir),
        "reason_code": reason,
        "reason": FAULT_REASON_LABELS[reason],
        "note": clean_note,
        "previewed_flip": str(bool(previewed_flip)).lower(),
        "committed_flips": int(committed_flips),
        "view_scope": view_scope,
    }
    path = fault_log_path(results_dir)
    try:
        _append_rows(path, FAULT_COLUMNS, [row])
    except (OSError, ValueError) as error:
        return FaultReportOutcome(False, f"Could not save the report: {error}")
    count = count_fault_reports(results_dir, mode_file_path, group_name)
    return FaultReportOutcome(
        True, f"Logged faulty-signal report {report_id[:8]} to {path}.",
        report_id=report_id, log_path=str(path), report_count=count,
    )


def log_flip_events(
    results_dir: str | Path, events: Sequence[FlipEvent], *, now: datetime | None = None,
) -> str | None:
    """Append flip-history rows; returns an error message instead of raising.

    A flip decision must never be blocked by its own bookkeeping, so callers
    show the returned text next to their normal result.
    """
    if not events:
        return None
    unknown = sorted({event.event for event in events if event.event not in FLIP_EVENTS})
    if unknown:
        return f"Unknown flip event(s): {', '.join(unknown)}"
    stamp = _timestamp(now)
    rows = [{
        "timestamp_utc": stamp,
        "event": event.event,
        **signal_identity(event.record, event.mode_file_path, event.group_name, event.trace_name, results_dir),
        "view_scope": event.view_scope,
        "operation_id": event.operation_id,
        "detail": event.detail,
    } for event in events]
    try:
        _append_rows(flip_history_path(results_dir), FLIP_HISTORY_COLUMNS, rows)
    except (OSError, ValueError) as error:
        return str(error)
    return None
