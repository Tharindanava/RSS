"""Process-based RSS pipeline jobs for the dashboard.

One RSS run on one CSV takes 7-28 minutes (measured: 424-1681 s, median 619 s at
the production 2000-epoch preset), so it cannot run inside a Dash callback. With
no ``diskcache``/``celery`` available for Dash background callbacks, each job is
a detached ``subprocess.Popen`` of the same ``python -m rss run-config`` command
the batch shell script uses, and the dashboard polls it.

Job state is authoritative **on disk** (``job.json`` per job), not in module
globals, so a job survives a Werkzeug reloader restart and even the dashboard
being killed:

* ``start_new_session=True`` puts each child in its own session, so a Ctrl-C or
  a reloader restart in the dashboard's process group does not kill a 30-minute
  run -- and cancellation can signal the whole job group deliberately.
* ``close_fds=True`` (the default, never disable it) keeps the child from
  inheriting the listening socket, which would otherwise hold port 8050 open for
  the length of the job.
* Liveness for a job this process did not launch is checked by reading
  ``/proc/<pid>/cmdline`` and requiring the job's own config path in it, so pid
  reuse cannot make a dead job look alive.

The child also gets thread limits: ``rss run-config`` never calls
``torch.set_num_threads``, so on a 56-core machine two unbounded jobs would
starve the dashboard's own plotting and transfer-entropy callbacks.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rss.exporters import resolve_realdata_output_dir


PRESET_ID = "gait_rss_sgd_so3_cross"
PRESET_LABEL = "Gait RSS · SGD · SO(3) · lagged cross-correlation"

#: Every production-run flag that is *not* user-overridable, in the order the
#: original batch launcher used. Epochs and the SSA window length are appended per
#: job; `--quiet` must never appear here, since it suppresses the progress lines
#: read back by _parse_progress.
PRESET_FIXED_ARGS: tuple[str, ...] = (
    "--sample-rate-hz", "100.0",
    "--include-time-seconds",
    "--allow-partial-limbs",
    "--file-workers", "1",
    "--enable-output", "ssa_preprocessing_signals",
    "--enable-output", "ssa_drift_correction_signals",
    "--enable-output", "ssa_component_metrics",
    "--enable-output", "ssa_plots",
    "--ssa-reconstruction-components", "20",
    "--ssa-drift-zero-crossing-limit", "5",
    "--ssa-drift-turning-point-limit", "5",
    "--rss-optimizer", "sgd",
    "--learning-rate-start", "1e-1",
    "--learning-rate-end", "1e-3",
    "--rss-sgd-momentum", "0.9",
    "--rss-loss-type", "lagged_cross_correlation",
    "--rss-separation-weight", "1.0",
    "--rss-lags", "0",
    "--rss-matrix-parameterization", "so3",
    "--rss-observer-signal-model", "euler",
    "--rss-initial-weight-matrix",
    "-0.4779489885", "-0.5421200170", "-0.6911372162",
    "-0.5983844737", "0.7769586228", "-0.1956305705",
    "0.6430402679", "0.3200643460", "-0.6957427889",
    "--rss-orthogonality-weight", "0.0",
    "--rss-reconstruction-weight", "0.0",
    "--rss-convergence-tol", "0.0",
    "--rss-eps", "1e-24",
    "--rss-quaternion-eps", "1e-24",
    "--rss-mode-order", "amplitude",
)

DEFAULT_EPOCHS = 2000
DEFAULT_SSA_WINDOW_LENGTH = 200
DEFAULT_COMPUTE_DEVICE = "cpu"
COMPUTE_DEVICE_OPTIONS: tuple[str, ...] = ("cpu", "auto", "cuda", "cuda:0")
EPOCHS_RANGE = (1, 20000)
SSA_WINDOW_RANGE = (10, 500)
#: Matches the concurrency of the original batch launcher.
MAX_CONCURRENT_JOBS = 2
THREADS_PER_JOB = 8
DEFAULT_PYTHON_BIN = sys.executable
JOBS_DIR_PARTS: tuple[str, ...] = ("rss", "_dashboard_jobs")
#: Written into the output leaf so a dashboard run is never indistinguishable
#: from the batch results beside it.
RUN_PARAMS_NAME = "rss_run_params.json"
RECORD_NAME = "job.json"
CONFIG_NAME = "config.json"
LOG_NAME = "run.log"
TAIL_BYTES = 65536
LIMB_TOTAL = 18
ACTIVE_STATUSES = frozenset({"queued", "running"})
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
#: 619 s median / 2000 epochs, from 231 measured file_elapsed_seconds records.
SECONDS_PER_EPOCH_ALL_LIMBS = 0.31
FIXED_SETUP_SECONDS = 90.0
#: How long to wait after SIGTERM before escalating to SIGKILL.
CANCEL_ESCALATE_SECONDS = 10.0

_LOCK = threading.Lock()
#: Handles for jobs this process launched; empty after a reloader restart, which
#: is why every status decision can also be made from the record plus /proc.
_HANDLES: dict[str, subprocess.Popen] = {}

_LINE = re.compile(r"^\[RSS\s+(?P<elapsed>[0-9]+(?:\.[0-9]+)?)s\]\s+(?P<body>.+)$")
_LIMB = re.compile(r"^\[(?P<fi>\d+)/(?P<ft>\d+)\]\s+(?P<limb>[A-Za-z0-9_]+)\s+\((?P<li>\d+)/(?P<lt>\d+)\)$")
_FILE = re.compile(r"^\[(?P<fi>\d+)/(?P<ft>\d+)\]\s+(?P<event>file_start|file_done):\s+(?P<csv>.+)$")
_RUN = re.compile(r"^(?P<event>run_start|run_done):\s+(?P<ft>\d+) file\(s\)$")


def clear_caches() -> None:
    """No caches here; present so every dashboard module exposes the same hook."""
    return None


# --- parameter handling -------------------------------------------------------


def _clamp_int(value: Any, *, fallback: int, low: int, high: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = int(fallback)
    return max(low, min(high, parsed))


def clamp_epochs(value: Any) -> int:
    return _clamp_int(value, fallback=DEFAULT_EPOCHS, low=EPOCHS_RANGE[0], high=EPOCHS_RANGE[1])


def clamp_ssa_window(value: Any) -> int:
    return _clamp_int(
        value, fallback=DEFAULT_SSA_WINDOW_LENGTH, low=SSA_WINDOW_RANGE[0], high=SSA_WINDOW_RANGE[1]
    )


def normalize_compute_device(value: Any) -> str:
    """Validate a device string the way ``rss.realdata_config`` will."""
    text = str(value or "").strip().lower()
    if text in {"auto", "cpu", "cuda"}:
        return text
    if re.fullmatch(r"cuda:\d+", text):
        return text
    return DEFAULT_COMPUTE_DEVICE


def estimate_seconds(epochs: Any, *, limbs: int = LIMB_TOTAL) -> float:
    """Rough wall-clock estimate, scaled from the measured 2000-epoch runs."""
    scale = max(1, int(limbs)) / LIMB_TOTAL
    return FIXED_SETUP_SECONDS + SECONDS_PER_EPOCH_ALL_LIMBS * clamp_epochs(epochs) * scale


def format_duration(seconds: float | int | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m {total % 60:02d}s"
    return f"{total // 3600}h {(total % 3600) // 60:02d}m"


# --- paths --------------------------------------------------------------------


def jobs_dir_for(results_dir: str | Path) -> Path:
    """Where job records live. Not created here -- ``submit_job`` does that."""
    return Path(results_dir).resolve().joinpath(*JOBS_DIR_PARTS)


def output_leaf_for(output_root: str | Path, csv_path: str | Path) -> Path:
    """The leaf folder this input will produce under ``output_root``."""
    return Path(resolve_realdata_output_dir(str(output_root), str(csv_path)))


# --- records ------------------------------------------------------------------


@dataclass(frozen=True)
class JobRequest:
    csv_path: Path
    output_root: Path
    epochs: int
    ssa_window_length: int
    compute_device: str
    project_root: Path
    python_bin: str
    jobs_dir: Path
    csv_label: str = ""


@dataclass
class JobRecord:
    job_id: str
    status: str
    message: str
    csv_path: str
    csv_label: str
    output_root: str
    output_leaf: str
    epochs: int
    ssa_window_length: int
    compute_device: str
    preset_id: str
    argv: list[str]
    cwd: str
    job_dir: str
    log_path: str
    config_path: str
    pid: int | None = None
    exit_code: int | None = None
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    cancel_requested_at: float | None = None
    estimated_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JobRecord":
        fields = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in payload.items() if key in fields})


@dataclass(frozen=True)
class JobProgress:
    status: str
    stage: str = ""
    fraction: float = 0.0
    limb_name: str | None = None
    limb_index: int | None = None
    limb_total: int | None = None
    elapsed_seconds: float | None = None
    eta_seconds: float | None = None
    last_event: str = ""
    error_hint: str = ""
    log_tail: str = ""


def build_config_payload(request: JobRequest) -> list[dict[str, str]]:
    """The run-config JSON body. ``rss.realdata_config`` requires exactly these two keys."""
    return [
        {
            "csv_path": str(Path(request.csv_path).resolve()),
            "compute_device": normalize_compute_device(request.compute_device),
        }
    ]


def build_preset_argv(request: JobRequest, *, config_path: str | Path) -> list[str]:
    """The full ``python -m rss run-config`` command line for one job."""
    return [
        str(request.python_bin),
        "-u",
        "-m",
        "rss",
        "run-config",
        "--config",
        str(config_path),
        "--output-root",
        str(request.output_root),
        *PRESET_FIXED_ARGS,
        "--epochs",
        str(clamp_epochs(request.epochs)),
        "--ssa-window-length",
        str(clamp_ssa_window(request.ssa_window_length)),
    ]


def build_job_request(
    *,
    csv_path: str | Path,
    output_root: str | Path,
    results_dir: str | Path,
    project_root: str | Path,
    python_bin: str = DEFAULT_PYTHON_BIN,
    epochs: Any = DEFAULT_EPOCHS,
    ssa_window_length: Any = DEFAULT_SSA_WINDOW_LENGTH,
    compute_device: Any = DEFAULT_COMPUTE_DEVICE,
    csv_label: str = "",
) -> JobRequest:
    return JobRequest(
        csv_path=Path(csv_path).resolve(),
        output_root=Path(output_root).resolve(),
        epochs=clamp_epochs(epochs),
        ssa_window_length=clamp_ssa_window(ssa_window_length),
        compute_device=normalize_compute_device(compute_device),
        project_root=Path(project_root).resolve(),
        python_bin=str(python_bin),
        jobs_dir=jobs_dir_for(results_dir),
        csv_label=csv_label or Path(csv_path).name,
    )


def _record_path(jobs_dir: str | Path, job_id: str) -> Path:
    return Path(jobs_dir) / job_id / RECORD_NAME


def _write_record(record: JobRecord) -> None:
    path = _record_path(Path(record.job_dir).parent, record.job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _read_record(path: Path) -> JobRecord | None:
    try:
        return JobRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


def _new_job_id(csv_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", csv_path.stem)[:32]
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{stem}_{uuid.uuid4().hex[:6]}"


def _is_pid_alive(pid: int | None, *, config_path: str) -> bool:
    """Liveness that survives pid reuse, for jobs we did not launch."""
    if not pid or pid <= 0:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace")
    except OSError:
        return False
    return "rss" in cmdline and config_path in cmdline


def _child_env(request: JobRequest) -> dict[str, str]:
    env = os.environ.copy()
    mpl_dir = env.get("RSS_MPLCONFIGDIR") or os.path.join(tempfile.gettempdir(), "rss_mpl")
    try:
        os.makedirs(mpl_dir, exist_ok=True)
        env["MPLCONFIGDIR"] = mpl_dir
    except OSError:
        env.pop("MPLCONFIGDIR", None)
    # rss run-config never calls torch.set_num_threads, so cap the child here
    # or it will take every core and starve the dashboard's own callbacks.
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[variable] = str(THREADS_PER_JOB)
    env["PYTHONUNBUFFERED"] = "1"
    existing = env.get("PYTHONPATH", "")
    root = str(request.project_root)
    env["PYTHONPATH"] = f"{root}{os.pathsep}{existing}" if existing else root
    return env


def _write_run_params(record: JobRecord) -> None:
    """Stamp the run's parameters into the output leaf.

    The dashboard writes into the same run tree as the batch script, so without
    this a 200-epoch dashboard run would be indistinguishable from a 2000-epoch
    batch result sitting beside it.
    """
    try:
        leaf = Path(record.output_leaf)
        leaf.mkdir(parents=True, exist_ok=True)
        (leaf / RUN_PARAMS_NAME).write_text(
            json.dumps(
                {
                    "source": "HMA_Dashboard",
                    "job_id": record.job_id,
                    "preset_id": record.preset_id,
                    "preset_label": PRESET_LABEL,
                    "epochs": record.epochs,
                    "ssa_window_length": record.ssa_window_length,
                    "compute_device": record.compute_device,
                    "csv_path": record.csv_path,
                    "output_root": record.output_root,
                    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created_at)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        # Provenance is a nicety; never fail a run over it.
        return


# --- lifecycle ----------------------------------------------------------------


def submit_job(request: JobRequest) -> JobRecord:
    """Persist a job and start it if a slot is free, else leave it queued."""
    job_id = _new_job_id(request.csv_path)
    job_dir = Path(request.jobs_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    config_path = job_dir / CONFIG_NAME
    config_path.write_text(json.dumps(build_config_payload(request), indent=2), encoding="utf-8")

    record = JobRecord(
        job_id=job_id,
        status="queued",
        message="Queued.",
        csv_path=str(request.csv_path),
        csv_label=request.csv_label,
        output_root=str(request.output_root),
        output_leaf=str(output_leaf_for(request.output_root, request.csv_path)),
        epochs=request.epochs,
        ssa_window_length=request.ssa_window_length,
        compute_device=request.compute_device,
        preset_id=PRESET_ID,
        argv=build_preset_argv(request, config_path=config_path),
        cwd=str(request.project_root),
        job_dir=str(job_dir),
        log_path=str(job_dir / LOG_NAME),
        config_path=str(config_path),
        created_at=time.time(),
        estimated_seconds=estimate_seconds(request.epochs),
    )
    _write_record(record)
    _write_run_params(record)

    with _LOCK:
        if _running_count(request.jobs_dir) < MAX_CONCURRENT_JOBS:
            record = _launch(record)
    return record


def _running_count(jobs_dir: str | Path) -> int:
    """Live job count, read from disk so the cap survives a process restart."""
    count = 0
    for record in _iter_records(jobs_dir):
        if record.status != "running":
            continue
        handle = _HANDLES.get(record.job_id)
        if handle is not None:
            if handle.poll() is None:
                count += 1
        elif _is_pid_alive(record.pid, config_path=record.config_path):
            count += 1
    return count


def _launch(record: JobRecord) -> JobRecord:
    request_env = JobRequest(
        csv_path=Path(record.csv_path),
        output_root=Path(record.output_root),
        epochs=record.epochs,
        ssa_window_length=record.ssa_window_length,
        compute_device=record.compute_device,
        project_root=Path(record.cwd),
        python_bin=record.argv[0] if record.argv else DEFAULT_PYTHON_BIN,
        jobs_dir=Path(record.job_dir).parent,
    )
    try:
        log_handle = open(record.log_path, "ab", buffering=0)
    except OSError as error:
        record.status = "failed"
        record.message = f"Could not open the job log: {error}"
        record.finished_at = time.time()
        _write_record(record)
        return record

    try:
        process = subprocess.Popen(
            record.argv,
            cwd=record.cwd,
            env=_child_env(request_env),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
    except (OSError, ValueError) as error:
        record.status = "failed"
        record.message = f"Could not start the RSS process: {error}"
        record.finished_at = time.time()
        _write_record(record)
        return record
    finally:
        log_handle.close()

    _HANDLES[record.job_id] = process
    record.pid = process.pid
    record.status = "running"
    record.message = "Running."
    record.started_at = time.time()
    _write_record(record)
    return record


def _reap(record: JobRecord) -> JobRecord:
    """Update one running record's status; calling this also reaps zombies."""
    if record.status != "running":
        return record

    handle = _HANDLES.get(record.job_id)
    if handle is not None:
        exit_code = handle.poll()
        if exit_code is None:
            return _maybe_escalate_cancel(record)
    elif _is_pid_alive(record.pid, config_path=record.config_path):
        return _maybe_escalate_cancel(record)
    else:
        exit_code = None

    record.finished_at = time.time()
    record.exit_code = exit_code
    # Whole tail window, not the display slice: the completion marker sits above
    # the run's final JSON summary.
    tail = "\n".join(_read_tail_lines(Path(record.log_path)))
    if record.cancel_requested_at is not None:
        record.status = "cancelled"
        record.message = "Cancelled before completion."
    elif exit_code == 0:
        record.status = "succeeded"
        record.message = "Completed."
    elif exit_code is None:
        # Orphaned by a dashboard restart: judge it by what the log recorded.
        succeeded = "run_done:" in tail
        record.status = "succeeded" if succeeded else "failed"
        record.message = (
            "Completed (recovered after a dashboard restart)."
            if succeeded
            else "Process is gone and the log has no completion marker."
        )
    else:
        record.status = "failed"
        record.message = f"Exited with code {exit_code}. {_error_hint(tail)}".strip()
    _HANDLES.pop(record.job_id, None)
    _write_record(record)
    return record


def _maybe_escalate_cancel(record: JobRecord) -> JobRecord:
    if record.cancel_requested_at is None:
        return record
    if time.time() - record.cancel_requested_at < CANCEL_ESCALATE_SECONDS:
        return record
    _signal_job(record, signal.SIGKILL)
    return record


def _signal_job(record: JobRecord, which: int) -> bool:
    if not record.pid:
        return False
    try:
        # The whole session, since start_new_session made the child a leader.
        os.killpg(os.getpgid(record.pid), which)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def pump(jobs_dir: str | Path, *, max_concurrent: int = MAX_CONCURRENT_JOBS) -> list[JobRecord]:
    """Reap finished jobs and promote queued ones. Safe to call on every poll."""
    with _LOCK:
        records = [_reap(record) for record in _iter_records(jobs_dir)]
        running = _running_count(jobs_dir)
        for record in sorted(
            (record for record in records if record.status == "queued"),
            key=lambda item: item.created_at,
        ):
            if running >= max_concurrent:
                break
            _launch(record)
            running += 1
    return list_jobs(jobs_dir)


def _iter_records(jobs_dir: str | Path):
    root = Path(jobs_dir)
    if not root.is_dir():
        return []
    records = []
    for entry in sorted(root.iterdir()):
        record = _read_record(entry / RECORD_NAME) if entry.is_dir() else None
        if record is not None:
            records.append(record)
    return records


def list_jobs(jobs_dir: str | Path, *, limit: int = 25) -> list[JobRecord]:
    """Newest jobs first."""
    records = sorted(_iter_records(jobs_dir), key=lambda record: record.created_at, reverse=True)
    return records[: max(1, int(limit))]


def get_job(jobs_dir: str | Path, job_id: str | None) -> JobRecord | None:
    if not job_id:
        return None
    return _read_record(_record_path(jobs_dir, job_id))


def has_active_jobs(jobs_dir: str | Path) -> bool:
    return any(record.status in ACTIVE_STATUSES for record in _iter_records(jobs_dir))


def find_active_job_for(jobs_dir: str | Path, csv_path: str | Path, output_root: str | Path) -> JobRecord | None:
    """An already-queued/running job for the same input and destination."""
    target = str(Path(csv_path).resolve())
    root = str(Path(output_root).resolve())
    for record in _iter_records(jobs_dir):
        if record.status in ACTIVE_STATUSES and record.csv_path == target and record.output_root == root:
            return record
    return None


def request_cancel(jobs_dir: str | Path, job_id: str | None) -> JobRecord | None:
    """Ask one job to stop; ``pump`` escalates to SIGKILL if it lingers."""
    record = get_job(jobs_dir, job_id)
    if record is None:
        return None
    if record.status == "queued":
        record.status = "cancelled"
        record.message = "Cancelled before it started."
        record.finished_at = time.time()
        _write_record(record)
        return record
    if record.status != "running":
        return record
    record.cancel_requested_at = time.time()
    record.message = "Cancelling..."
    _write_record(record)
    _signal_job(record, signal.SIGTERM)
    return record


# --- progress -----------------------------------------------------------------


def _read_tail_lines(path: Path, *, tail_bytes: int = TAIL_BYTES) -> list[str]:
    """Every non-blank line in the last ``tail_bytes`` of the log."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - tail_bytes))
            data = handle.read()
    except OSError:
        return []
    text = data.decode("utf-8", "replace")
    return [line.rstrip() for line in text.splitlines() if line.strip()]


def _progress_lines(tail_lines: list[str], *, keep: int = 12) -> list[str]:
    """The last few ``[RSS ...]`` lines out of a tail.

    Needed because the run ends by printing a JSON summary to stdout (~5.9 KB /
    127 lines for a single-file job), which would otherwise push every progress
    line out of a plain last-N-lines view.
    """
    matches = [line for line in tail_lines if _LINE.match(line)]
    return matches[-max(1, int(keep)) :]


def _read_tail(path: Path, *, tail_bytes: int = TAIL_BYTES, tail_lines: int = 40) -> list[str]:
    """Last ``tail_lines`` lines, for display."""
    return _read_tail_lines(path, tail_bytes=tail_bytes)[-max(1, int(tail_lines)) :]


def _error_hint(tail: str) -> str:
    """`rss run-config` never prints a run_error line, so look for a traceback."""
    lines = [line for line in tail.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if "Traceback (most recent call last)" in line:
            return f"Last error: {lines[-1].strip()}" if index < len(lines) - 1 else "See the job log."
    for line in reversed(lines):
        if "Error" in line or "error:" in line:
            return f"Last error: {line.strip()}"
    return ""


def _parse_progress(lines: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "stage": "",
        "fraction": 0.0,
        "limb_name": None,
        "limb_index": None,
        "limb_total": None,
        "elapsed_seconds": None,
        "last_event": "",
        "done": False,
    }
    for line in lines:
        match = _LINE.match(line)
        if not match:
            continue
        parsed["elapsed_seconds"] = float(match.group("elapsed"))
        body = match.group("body").strip()
        parsed["last_event"] = body

        limb = _LIMB.match(body)
        if limb:
            index = int(limb.group("li"))
            total = int(limb.group("lt"))
            parsed["limb_name"] = limb.group("limb")
            parsed["limb_index"] = index
            parsed["limb_total"] = total
            # The limb line is printed *before* that limb runs, so index-1 limbs
            # are actually finished.
            parsed["fraction"] = max(0.0, min(1.0, (index - 1) / max(1, total)))
            parsed["stage"] = f"limb {index}/{total}: {limb.group('limb')}"
            continue

        file_event = _FILE.match(body)
        if file_event:
            if file_event.group("event") == "file_done":
                parsed["fraction"] = 1.0
                parsed["stage"] = "writing outputs"
                parsed["done"] = True
            else:
                parsed["stage"] = "loading input"
            continue

        run_event = _RUN.match(body)
        if run_event:
            parsed["stage"] = "starting" if run_event.group("event") == "run_start" else "finished"
            if run_event.group("event") == "run_done":
                parsed["fraction"] = 1.0
                parsed["done"] = True
    return parsed


def read_progress(record: JobRecord, *, tail_bytes: int = TAIL_BYTES, tail_lines: int = 40) -> JobProgress:
    """Progress for one job, derived from its log tail."""
    all_lines = _read_tail_lines(Path(record.log_path), tail_bytes=tail_bytes)
    progress_lines = _progress_lines(all_lines, keep=tail_lines)
    parsed = _parse_progress(progress_lines)
    display_lines = all_lines[-max(1, int(tail_lines)) :]
    if progress_lines and not any(_LINE.match(line) for line in display_lines):
        # A finished run ends with a JSON summary long enough to push the whole
        # progress narrative out of a plain last-N-lines view; show the run's
        # own story first, then whatever the log actually ends with.
        display_lines = [*progress_lines, "...", *display_lines[-8:]]
    tail_text = "\n".join(display_lines)

    if record.status in TERMINAL_STATUSES:
        fraction = 1.0 if record.status == "succeeded" else parsed["fraction"]
    else:
        fraction = parsed["fraction"]

    eta: float | None = None
    elapsed = parsed["elapsed_seconds"]
    if record.status == "running":
        index = parsed["limb_index"] or 0
        total = parsed["limb_total"] or LIMB_TOTAL
        if elapsed and index > 1:
            per_limb = elapsed / (index - 1)
            eta = max(0.0, per_limb * (total - (index - 1)))
        else:
            eta = max(0.0, record.estimated_seconds - (elapsed or 0.0))

    return JobProgress(
        status=record.status,
        stage=parsed["stage"] or record.status,
        fraction=float(fraction),
        limb_name=parsed["limb_name"],
        limb_index=parsed["limb_index"],
        limb_total=parsed["limb_total"],
        elapsed_seconds=elapsed,
        eta_seconds=eta,
        last_event=parsed["last_event"],
        error_hint=_error_hint(tail_text) if record.status == "failed" else "",
        log_tail=tail_text,
    )


def snapshot(jobs_dir: str | Path, *, limit: int = 25) -> list[dict[str, Any]]:
    """JSON-safe job rows for a ``dcc.Store``."""
    rows: list[dict[str, Any]] = []
    for record in list_jobs(jobs_dir, limit=limit):
        progress = read_progress(record)
        row = record.to_dict()
        row["progress"] = asdict(progress)
        # The tail can be large; the monitor re-reads it for the selected job.
        row["progress"].pop("log_tail", None)
        rows.append(row)
    return rows


def prune_terminal_jobs(jobs_dir: str | Path, *, older_than_seconds: float = 7 * 24 * 3600) -> int:
    """Delete finished job folders older than the cutoff. Returns the count."""
    import shutil

    now = time.time()
    removed = 0
    for record in _iter_records(jobs_dir):
        if record.status not in TERMINAL_STATUSES:
            continue
        finished = record.finished_at or record.created_at
        if now - finished < older_than_seconds:
            continue
        try:
            shutil.rmtree(record.job_dir)
            removed += 1
        except OSError:
            continue
    return removed
