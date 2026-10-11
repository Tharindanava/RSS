"""Discovery, validation, and ingest for the raw ``Sample_Data`` input tree.

The dashboard was a read-only viewer of ``outputs/`` and had no knowledge of the
inputs those outputs came from. This module adds that half: it enumerates the
raw IMU CSVs, works out which ones the RSS pipeline has already processed, and
accepts uploads of new ones.

The processed/unprocessed answer is exact rather than heuristic, and costs no
extra directory walk. ``rss.exporters.resolve_realdata_output_dir`` builds an
output leaf by mirroring every path component *after* the last ``Sample_Data``
component and appending the CSV stem, so inverting it means splitting each known
run directory at its last dataset component. The run directories come from
``catalog.scan_results_catalog``, which is already ``lru_cache``d.

Everything a browser sends is re-validated here. The destination folder is
checked against the discovered folder set rather than trusted from the dropdown,
new sub-path components are sanitized individually, and the joined path is
confirmed to still be inside ``Sample_Data`` *after* ``resolve()`` -- which also
catches an escape through a symlink.
"""

from __future__ import annotations

import base64
import binascii
import os
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from rss.imu_input import load_and_validate_input
from rss.sensor_layout import SEGMENT_SENSOR_MAPPING, required_imu_columns
from HMA_Dashboard import catalog


SAMPLE_DATA_DIR_NAME = "Sample_Data"
MAX_NEW_SUBPATH_DEPTH = 6
MAX_COMPONENT_LENGTH = 100
FORBIDDEN_COMPONENT_CHARS = '<>:"|?*\\'
RESERVED_COMPONENTS = frozenset({"", ".", ".."})
#: Only used to parse the upload for validation; the pipeline is told the real
#: rate at run time.
VALIDATION_SAMPLE_RATE_HZ = 100.0
DEFAULT_MAX_UPLOAD_BYTES = 32 * 1024 * 1024
_DRIVE_LETTER = re.compile(r"^[A-Za-z]:")
_DECODE_CHUNK_CHARS = 3 * 1024 * 1024 * 4 // 3


def clear_caches() -> None:
    """Clear cached raw-tree scans."""
    _scan_raw_catalog_cached.cache_clear()


# --- path safety --------------------------------------------------------------


def sanitize_component(text: str) -> str:
    """One folder or file name component, or raise with a user-facing reason."""
    candidate = unicodedata.normalize("NFC", str(text)).strip()
    if candidate in RESERVED_COMPONENTS:
        raise ValueError(f"'{text}' is not a usable folder name.")
    if "/" in candidate or "\\" in candidate:
        raise ValueError("Folder names cannot contain path separators.")
    if any(character in candidate for character in FORBIDDEN_COMPONENT_CHARS):
        raise ValueError(f"'{candidate}' contains one of these characters: {FORBIDDEN_COMPONENT_CHARS}")
    if any(ord(character) < 32 or ord(character) == 127 for character in candidate):
        raise ValueError("Folder names cannot contain control characters.")
    if candidate.startswith(".") or candidate.endswith("."):
        raise ValueError("Folder names cannot start or end with a dot.")
    if len(candidate) > MAX_COMPONENT_LENGTH:
        raise ValueError(f"Folder names are limited to {MAX_COMPONENT_LENGTH} characters.")
    return candidate


def sanitize_relative_subpath(text: str | None) -> tuple[str, ...]:
    """Split and sanitize a user-typed ``a/b/c`` sub-path (empty is allowed)."""
    raw = str(text or "").strip()
    if not raw:
        return ()
    if raw.startswith("/") or raw.startswith("\\") or _DRIVE_LETTER.match(raw):
        raise ValueError("Enter a relative sub-path, not an absolute path.")
    pieces = [piece for piece in re.split(r"[\\/]+", raw) if piece != ""]
    if len(pieces) > MAX_NEW_SUBPATH_DEPTH:
        raise ValueError(f"At most {MAX_NEW_SUBPATH_DEPTH} new folder levels can be created at once.")
    return tuple(sanitize_component(piece) for piece in pieces)


def sanitize_csv_filename(text: str) -> str:
    """A ``.csv`` file name component."""
    candidate = sanitize_component(text)
    if not candidate.lower().endswith(".csv"):
        raise ValueError("Raw IMU files must be .csv files.")
    return candidate


def resolve_destination(
    sample_data_dir: str | Path,
    *,
    parent_relative: str,
    new_subpath: str | None,
    filename: str,
    known_folders: Sequence[str] | None = None,
) -> Path:
    """Where an upload will land, validated against the real tree."""
    root = Path(sample_data_dir).resolve()
    parent_parts = sanitize_relative_subpath(parent_relative) if parent_relative else ()
    if known_folders is not None and parent_relative:
        # The dropdown value came from the browser, so confirm it against the
        # folders actually on disk instead of trusting it.
        if "/".join(parent_parts) not in set(known_folders):
            raise ValueError(f"'{parent_relative}' is not an existing folder under {SAMPLE_DATA_DIR_NAME}.")

    destination = root.joinpath(*parent_parts, *sanitize_relative_subpath(new_subpath))
    destination = destination / sanitize_csv_filename(filename)
    resolved = destination.resolve()
    # After resolve(), so a symlink pointing outside the tree is caught too.
    if not resolved.is_relative_to(root):
        raise ValueError(f"The destination must stay inside {SAMPLE_DATA_DIR_NAME}.")
    if resolved.exists() and resolved.is_dir():
        raise ValueError(f"'{resolved.name}' is an existing folder, not a file name.")
    return resolved


# --- validation ---------------------------------------------------------------


@dataclass(frozen=True)
class RawCsvValidation:
    ok: bool
    row_count: int = 0
    column_count: int = 0
    usable_limbs: tuple[str, ...] = ()
    skipped_limbs: tuple[str, ...] = ()
    missing_columns: tuple[str, ...] = ()
    has_timestamp: bool = False
    warnings: tuple[str, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class IngestResult:
    ok: bool
    message: str
    destination: str = ""
    created_dirs: tuple[str, ...] = ()
    replaced: bool = False
    validation: RawCsvValidation = field(default_factory=lambda: RawCsvValidation(ok=False))


def usable_limbs(columns: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Which of the 18 mapped limbs have a complete gyro triplet.

    Mirrors the availability rule in the private
    ``rss.realdata_pipeline._available_limbs`` (deliberately not imported); if
    that rule changes, this needs revisiting.
    """
    present = set(columns)
    available: list[str] = []
    skipped: list[str] = []
    for limb, device in SEGMENT_SENSOR_MAPPING.items():
        if all(f"{device}_G{axis}" in present for axis in ("X", "Y", "Z")):
            available.append(limb)
        else:
            skipped.append(limb)
    return tuple(available), tuple(skipped)


def validate_raw_csv(csv_path: str | Path) -> RawCsvValidation:
    """Parse one raw IMU CSV the way the pipeline will.

    Goes through the public loader so ``OrbiX-Node-N_*`` / ``Device_N_*``
    headers are normalized for free. Only a file with *no* usable limb is
    rejected: the RSS preset already runs with ``--allow-partial-limbs``, so a
    trial missing a device is still worth keeping.
    """
    path = Path(csv_path)
    try:
        loaded = load_and_validate_input(
            str(path),
            sample_rate_hz=VALIDATION_SAMPLE_RATE_HZ,
            include_time_seconds=True,
            allow_partial_imu=True,
        )
    except Exception as error:  # the loader raises a variety of error types
        return RawCsvValidation(ok=False, error=f"{type(error).__name__}: {error}")

    frame = loaded.dataframe
    columns = list(frame.columns)
    available, skipped = usable_limbs(columns)
    missing = tuple(column for column in required_imu_columns() if column not in set(columns))

    warnings: list[str] = []
    if missing:
        warnings.append(
            f"{len(missing)} of {len(required_imu_columns())} mapped IMU columns are absent, so "
            f"{len(skipped)} limb(s) will be skipped."
        )
    if not available:
        return RawCsvValidation(
            ok=False,
            row_count=int(len(frame)),
            column_count=len(columns),
            skipped_limbs=skipped,
            missing_columns=missing,
            error=(
                "No limb has a complete gyro triplet, so the RSS pipeline has nothing to "
                "process. Check that the header uses device{N}_IMU{M}_{AX..GZ} names."
            ),
        )

    return RawCsvValidation(
        ok=True,
        row_count=int(len(frame)),
        column_count=len(columns),
        usable_limbs=available,
        skipped_limbs=skipped,
        missing_columns=missing,
        has_timestamp="Timestamp" in columns,
        warnings=tuple(warnings),
    )


# --- upload -------------------------------------------------------------------


def decode_upload_contents(contents: str, *, max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES) -> bytes:
    """Decode a ``dcc.Upload`` data URL into bytes, refusing oversize payloads."""
    text = str(contents or "")
    if not text.startswith("data:") or "," not in text:
        raise ValueError("The upload payload was not a data URL.")
    header, payload = text.split(",", 1)
    if ";base64" not in header:
        raise ValueError("Only base64-encoded uploads are supported.")
    # 4 base64 characters per 3 bytes, so this bounds the decode before doing it.
    if len(payload) // 4 * 3 > max_bytes:
        raise ValueError(f"The file is larger than the {_format_bytes(max_bytes)} upload limit.")

    chunks: list[bytes] = []
    total = 0
    try:
        for start in range(0, len(payload), _DECODE_CHUNK_CHARS):
            chunk = base64.b64decode(payload[start : start + _DECODE_CHUNK_CHARS], validate=True)
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"The file is larger than the {_format_bytes(max_bytes)} upload limit.")
            chunks.append(chunk)
    except binascii.Error as error:
        raise ValueError(f"The upload payload was not valid base64: {error}") from error
    if total == 0:
        raise ValueError("The uploaded file was empty.")
    return b"".join(chunks)


def write_raw_csv(
    payload: bytes,
    destination: Path,
    *,
    sample_data_dir: str | Path,
    replace: bool,
) -> IngestResult:
    """Validate then publish one raw CSV atomically.

    Written to a temporary name in the *destination directory* (same filesystem,
    so the rename is atomic) and validated there, so a rejected or interrupted
    upload never appears as a real input file.
    """
    root = Path(sample_data_dir).resolve()
    resolved = Path(destination).resolve()
    if not resolved.is_relative_to(root):
        return IngestResult(ok=False, message=f"The destination must stay inside {SAMPLE_DATA_DIR_NAME}.")
    if resolved.exists() and not replace:
        return IngestResult(
            ok=False,
            message=(
                f"'{resolved.name}' already exists in that folder. Tick 'Replace an existing "
                "file' to overwrite it, or choose a different name."
            ),
            destination=str(resolved),
        )

    created: list[str] = []
    probe = resolved.parent
    while not probe.exists():
        created.append(str(probe))
        probe = probe.parent

    temporary = resolved.with_name(f"{resolved.name}.part-{uuid4().hex}")
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as handle:
            handle.write(payload)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        return IngestResult(ok=False, message=f"Could not write the file: {error}")

    validation = validate_raw_csv(temporary)
    if not validation.ok:
        temporary.unlink(missing_ok=True)
        return IngestResult(
            ok=False,
            message=f"The file was not stored because it failed validation. {validation.error}",
            validation=validation,
        )

    replaced = resolved.exists()
    try:
        os.replace(temporary, resolved)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        return IngestResult(ok=False, message=f"Could not publish the file: {error}", validation=validation)

    return IngestResult(
        ok=True,
        message=(
            f"Stored {resolved.name} ({validation.row_count} rows, "
            f"{len(validation.usable_limbs)}/{len(SEGMENT_SENSOR_MAPPING)} limbs usable)."
        ),
        destination=str(resolved),
        created_dirs=tuple(reversed(created)),
        replaced=replaced,
        validation=validation,
    )


# --- discovery ----------------------------------------------------------------


def output_key_for_csv(csv_path: str | Path) -> tuple[str, ...]:
    """The output-leaf suffix one raw CSV maps to.

    The exact inverse of ``rss.exporters.resolve_realdata_output_dir``: the
    components after the last ``Sample_Data`` component, plus the CSV stem.
    """
    path = Path(csv_path).expanduser().resolve()
    parent_parts = path.parent.parts
    indices = [index for index, part in enumerate(parent_parts) if part.lower() == "sample_data"]
    if not indices:
        return (path.stem,)
    return (*parent_parts[indices[-1] + 1 :], path.stem)


def scan_raw_catalog(sample_data_dir: str | Path, results_dir: str | Path) -> dict[str, Any]:
    """Scan the raw input tree and mark which files already have outputs."""
    return _scan_raw_catalog_cached(
        str(Path(sample_data_dir).resolve()), str(Path(results_dir).resolve())
    )


@lru_cache(maxsize=8)
def _scan_raw_catalog_cached(sample_data_dir_str: str, results_dir_str: str) -> dict[str, Any]:
    root = Path(sample_data_dir_str)
    if not root.is_dir():
        return {
            "sample_data_dir": sample_data_dir_str,
            "results_dir": results_dir_str,
            "exists": False,
            "datasets": [],
            "dataset_options": [],
            "folders": [],
            "folder_options": [],
            "files": [],
            "processed_count": 0,
            "unprocessed_count": 0,
            "status": f"No raw data directory at {sample_data_dir_str}.",
        }

    datasets = sorted(entry.name for entry in root.iterdir() if entry.is_dir())
    processed = _processed_output_keys(results_dir_str, set(datasets))

    # Real directories, not just the ones that happen to contain a CSV: a
    # freshly created (still empty) trial folder has to be selectable as an
    # upload destination too.
    folders: set[str] = set()
    for current, directory_names, _file_names in os.walk(root):
        for name in directory_names:
            folders.add(Path(current, name).relative_to(root).as_posix())

    files: list[dict[str, Any]] = []
    for csv_path in sorted(root.rglob("*.csv")):
        if not csv_path.is_file():
            continue
        relative = csv_path.relative_to(root)
        parts = relative.parts
        key = output_key_for_csv(csv_path)
        run_roots = processed.get(key, [])
        try:
            size = csv_path.stat().st_size
        except OSError:
            size = 0
        marker = "[done] " if run_roots else ""
        files.append(
            {
                "raw_id": relative.as_posix(),
                "abs_path": str(csv_path),
                "relative_path": relative.as_posix(),
                "dataset": parts[0] if len(parts) > 1 else "",
                "levels": list(parts[:-1]),
                "stem": csv_path.stem,
                "size_bytes": int(size),
                "size_label": _format_bytes(size),
                "output_key": "/".join(key),
                "processed": bool(run_roots),
                "processed_run_roots": list(run_roots),
                "label": f"{marker}{relative.as_posix()}  ({_format_bytes(size)})",
            }
        )

    processed_count = sum(1 for record in files if record["processed"])
    folder_list = sorted(folders)
    return {
        "sample_data_dir": sample_data_dir_str,
        "results_dir": results_dir_str,
        "exists": True,
        "datasets": datasets,
        "dataset_options": [{"label": name, "value": name} for name in datasets],
        "folders": folder_list,
        "folder_options": [
            {"label": f"{SAMPLE_DATA_DIR_NAME}/", "value": ""},
            *[{"label": name, "value": name} for name in folder_list],
        ],
        "files": files,
        "processed_count": processed_count,
        "unprocessed_count": len(files) - processed_count,
        "status": (
            f"{len(files)} raw CSV file(s) in {len(datasets)} dataset(s): "
            f"{processed_count} processed, {len(files) - processed_count} unprocessed."
        ),
    }


def _processed_output_keys(results_dir_str: str, dataset_names: set[str]) -> dict[tuple[str, ...], list[str]]:
    """Map each existing output leaf back to the raw CSV that produced it."""
    results_catalog = catalog.scan_results_catalog(results_dir_str)
    keys: dict[tuple[str, ...], list[str]] = {}
    for record in results_catalog.get("runs", []):
        parts = Path(record["run_dir"]).parts
        indices = [index for index, part in enumerate(parts) if part in dataset_names]
        if not indices:
            continue
        split = indices[-1]
        key = tuple(parts[split:])
        run_root = str(Path(*parts[:split])) if split else ""
        keys.setdefault(key, [])
        if run_root not in keys[key]:
            keys[key].append(run_root)
    return keys


def list_output_run_roots(results_dir: str | Path, sample_data_dir: str | Path) -> list[dict[str, str]]:
    """Existing RSS run roots, as dropdown options labelled relative to results."""
    results_root = Path(results_dir).resolve()
    datasets = {
        entry.name for entry in Path(sample_data_dir).resolve().iterdir() if entry.is_dir()
    } if Path(sample_data_dir).is_dir() else set()
    roots = {
        root
        for run_roots in _processed_output_keys(str(results_root), datasets).values()
        for root in run_roots
        if root
    }
    options: list[dict[str, str]] = []
    for root in sorted(roots):
        try:
            label = str(Path(root).relative_to(results_root))
        except ValueError:
            label = root
        options.append({"label": label, "value": root})
    return options


# --- accessors ----------------------------------------------------------------


def get_raw_file_record(raw_catalog: dict[str, Any], raw_id: str | None) -> dict[str, Any] | None:
    if not raw_id:
        return None
    for record in raw_catalog.get("files", []):
        if record["raw_id"] == raw_id:
            return record
    return None


def get_raw_file_record_by_path(raw_catalog: dict[str, Any], abs_path: str | Path | None) -> dict[str, Any] | None:
    """Look one raw file up by absolute path, e.g. straight after an upload."""
    if not abs_path:
        return None
    target = str(Path(abs_path).resolve())
    for record in raw_catalog.get("files", []):
        if record["abs_path"] == target:
            return record
    return None


def get_raw_file_options(
    raw_catalog: dict[str, Any],
    *,
    dataset: str | None = None,
    unprocessed_only: bool = True,
) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    for record in raw_catalog.get("files", []):
        if dataset and dataset != "all" and record["dataset"] != dataset:
            continue
        if unprocessed_only and record["processed"]:
            continue
        options.append({"label": record["label"], "value": record["raw_id"]})
    return options


def get_dataset_options(raw_catalog: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"label": "All datasets", "value": "all"},
        *raw_catalog.get("dataset_options", []),
    ]


def get_folder_options(raw_catalog: dict[str, Any]) -> list[dict[str, str]]:
    return list(raw_catalog.get("folder_options", []))


def get_raw_catalog_summary(raw_catalog: dict[str, Any]) -> str:
    return str(raw_catalog.get("status", ""))


def _format_bytes(size: float | int | None) -> str:
    if not size or size <= 0:
        return "0 B"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"
