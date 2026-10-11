"""Discovery helpers for RSS output folders."""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from gait_analysis.metadata import condition_metadata


MODE_FILENAMES: dict[str, str] = {
    "joint": "joint_angle_mode.csv",
    "limb": "limb_angle_mode.csv",
}
AXES_FILENAMES: dict[str, str] = {
    "joint": "joint_axes_mode.csv",
    "limb": "limb_axes_mode.csv",
}
METRICS_FILENAMES: dict[str, str] = {
    "joint": "joint_mode_metrics.csv",
    "limb": "limb_mode_metrics.csv",
}
FAMILY_LABELS: dict[str, str] = {
    "joint": "Joint outputs",
    "limb": "Limb outputs",
}
TRACE_ORDER: tuple[str, ...] = ("original_angle", "mode_1", "mode_2", "mode_3")
#: Folder that groups human-motion recordings: ``Human_Motion/<subject>/<terrain>/<experiment>/...``.
#: The levels after it become the subject / terrain / activity / experiment selectors.
HUMAN_MOTION_DIR = "Human_Motion"
GROUP_MODE_PATTERN = re.compile(r"^(?P<group>.+)_mode_(?P<mode>[123])$")
HIERARCHY_FIELDS: tuple[str, ...] = ("subject", "terrain", "activity", "experiment", "csv_file")
EXCLUDED_PATH_PARTS: frozenset[str] = frozenset({"hma_dashboard_flip_backup", "hilbert_analysis_backup"})


def is_backup_path(path: str | Path) -> bool:
    return any(part in EXCLUDED_PATH_PARTS for part in Path(path).parts)


def clear_caches() -> None:
    """Clear cached catalog and schema state."""
    _scan_results_catalog_cached.cache_clear()
    _read_mode_file_schema_cached.cache_clear()


def scan_results_catalog(results_dir: str | Path) -> dict[str, Any]:
    """Scan the results directory and return a serializable run catalog."""
    return _scan_results_catalog_cached(str(Path(results_dir).expanduser().resolve()))


@lru_cache(maxsize=8)
def _scan_results_catalog_cached(results_dir_str: str) -> dict[str, Any]:
    root = Path(results_dir_str)
    catalog: dict[str, Any] = {
        "results_dir": str(root),
        "exists": root.exists() and root.is_dir(),
        "runs": [],
        "run_map": {},
        "run_options": [],
        "family_counts": {"joint": 0, "limb": 0},
    }
    if not catalog["exists"]:
        catalog["status"] = f"Results directory not found: {root}"
        return catalog

    runs: list[dict[str, Any]] = []
    for family, filename in MODE_FILENAMES.items():
        for mode_file_path in sorted(root.rglob(filename), key=lambda path: str(path).lower()):
            if not mode_file_path.is_file() or is_backup_path(mode_file_path):
                continue
            runs.append(_build_run_record(root=root, mode_file_path=mode_file_path, family=family))

    runs.sort(key=_run_sort_key)
    catalog["runs"] = runs
    catalog["run_map"] = {record["run_id"]: record for record in runs}
    catalog["run_options"] = [_run_option(record) for record in runs]
    for record in runs:
        catalog["family_counts"][record["family"]] += 1

    total_runs = len(runs)
    joint_count = catalog["family_counts"]["joint"]
    limb_count = catalog["family_counts"]["limb"]
    catalog["status"] = f"Found {total_runs} dashboard runs ({joint_count} joint, {limb_count} limb)."
    return catalog


def _build_run_record(*, root: Path, mode_file_path: Path, family: str) -> dict[str, Any]:
    schema = read_mode_file_schema(mode_file_path)
    run_dir = mode_file_path.parent.resolve()
    relative_path = str(run_dir.relative_to(root))
    run_id = f"{family}:{run_dir}"
    axes_path = run_dir / AXES_FILENAMES[family]
    metrics_path = run_dir / METRICS_FILENAMES[family]
    hierarchy = _build_hierarchy(root=root, mode_file_path=mode_file_path, family=family)
    scenario = condition_metadata(mode_file_path)
    return {
        "run_id": run_id,
        "label": relative_path,
        "value": run_id,
        **hierarchy,
        "hierarchy": dict(hierarchy),
        "scenario": {name: scenario[name] for name in ("terrain", "direction", "footwear")},
        "family": family,
        "family_label": FAMILY_LABELS[family],
        "run_dir": str(run_dir),
        "relative_path": relative_path,
        "mode_file_path": str(mode_file_path.resolve()),
        "axes_file_path": str(axes_path.resolve()) if axes_path.exists() else "",
        "metrics_file_path": str(metrics_path.resolve()) if metrics_path.exists() else "",
        "available_groups": list(schema["groups"]),
        "group_labels": {group: pretty_signal_label(group) for group in schema["groups"]},
        "available_traces": list(schema["available_traces"]),
        "trace_options_by_group": {
            group: [{"label": pretty_trace_label(trace), "value": trace} for trace in schema["group_traces"][group]]
            for group in schema["groups"]
        },
        "group_traces": {group: list(schema["group_traces"][group]) for group in schema["groups"]},
        "has_time": bool(schema["has_time"]),
        "has_frame": bool(schema["has_frame"]),
        "mode_file_mtime_ns": int(mode_file_path.stat().st_mtime_ns),
    }


def _build_hierarchy(*, root: Path, mode_file_path: Path, family: str) -> dict[str, str]:
    relative_file_parts = mode_file_path.relative_to(root).parts
    try:
        group_index = relative_file_parts.index(HUMAN_MOTION_DIR)
    except ValueError:
        group_index = -1

    # Components after HUMAN_MOTION_DIR, including the mode-file name itself.
    # The full shape is subject/terrain/activity/experiment/csv_stem/<file> (6),
    # but some trials omit the activity level and sit directly under the terrain
    # (5). Accepting only 6 sent those runs to the generic fallback below, where
    # the subject read as the output-root name instead of the participant.
    tail = relative_file_parts[group_index + 1 :] if group_index >= 0 else ()
    if len(tail) >= 6:
        subject, terrain, activity, experiment, csv_stem = tail[0], tail[1], tail[2], tail[3], tail[4]
        return {
            "subject": subject,
            "terrain": terrain,
            "activity": activity,
            "experiment": experiment,
            "csv_file": f"{csv_stem} - {mode_file_path.name}",
        }
    if len(tail) == 5:
        subject, terrain, experiment, csv_stem = tail[0], tail[1], tail[2], tail[3]
        return {
            "subject": subject,
            "terrain": terrain,
            "activity": "Unspecified Activity",
            "experiment": experiment,
            "csv_file": f"{csv_stem} - {mode_file_path.name}",
        }

    relative_run_parts = mode_file_path.parent.relative_to(root).parts
    fallback = _fallback_hierarchy_parts(relative_run_parts)
    return {
        "subject": fallback[0],
        "terrain": fallback[1],
        "activity": fallback[2],
        "experiment": fallback[3],
        "csv_file": f"{fallback[4]} - {mode_file_path.name}",
    }


def _fallback_hierarchy_parts(relative_run_parts: tuple[str, ...]) -> tuple[str, str, str, str, str]:
    parts = list(relative_run_parts)
    if not parts:
        return ("Outputs", "Unspecified Terrain", "Unspecified Activity", "Unspecified Experiment", "run")
    subject = parts[0]
    terrain = parts[1] if len(parts) > 1 else "Unspecified Terrain"
    activity = parts[2] if len(parts) > 2 else "Unspecified Activity"
    experiment = " / ".join(parts[3:]) if len(parts) > 3 else parts[-1]
    csv_stem = parts[-1]
    return subject, terrain, activity, experiment, csv_stem


def _run_sort_key(record: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(record.get(field, "")).lower()
        for field in (*HIERARCHY_FIELDS, "family", "mode_file_path")
    )


def _run_option(record: dict[str, Any]) -> dict[str, str]:
    return {
        "label": str(record["csv_file"]),
        "value": str(record["run_id"]),
    }


def get_catalog_summary(catalog: dict[str, Any]) -> str:
    """Return a short status summary for the current catalog."""
    if catalog.get("run_options"):
        return str(catalog.get("status"))
    return str(catalog.get("status") or "No dashboard runs discovered.")


def get_run_options(
    catalog: dict[str, Any],
    *,
    subject: str | None = None,
    terrain: str | None = None,
    activity: str | None = None,
    experiment: str | None = None,
) -> list[dict[str, Any]]:
    """Return CSV-file dropdown options for discovered runs."""
    if subject is None and terrain is None and activity is None and experiment is None:
        return list(catalog.get("run_options", []))
    return [
        _run_option(record)
        for record in _filter_run_records(
            catalog,
            subject=subject,
            terrain=terrain,
            activity=activity,
            experiment=experiment,
        )
    ]


def list_run_records(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Return run records from the catalog."""
    return list(catalog.get("runs", []))


def get_run_record(catalog: dict[str, Any], run_id: str | None) -> dict[str, Any] | None:
    """Return one run record from the catalog."""
    if not run_id:
        return None
    return catalog.get("run_map", {}).get(run_id)


def get_first_run_id(catalog: dict[str, Any], family: str | None = None) -> str | None:
    """Return the first discovered run id, optionally filtered by family."""
    for record in list_run_records(catalog):
        if family is None or record["family"] == family:
            return str(record["run_id"])
    return None


def get_hierarchy_options(
    catalog: dict[str, Any],
    field: str,
    *,
    subject: str | None = None,
    terrain: str | None = None,
    activity: str | None = None,
    experiment: str | None = None,
) -> list[dict[str, str]]:
    """Return unique dropdown options for one hierarchy field."""
    if field not in HIERARCHY_FIELDS:
        raise ValueError(f"Unsupported hierarchy field: {field}")
    records = _filter_run_records(
        catalog,
        subject=subject if field != "subject" else None,
        terrain=terrain if field not in {"subject", "terrain"} else None,
        activity=activity if field not in {"subject", "terrain", "activity"} else None,
        experiment=experiment if field == "csv_file" else None,
    )
    values = sorted({str(record[field]) for record in records}, key=str.lower)
    return [{"label": value, "value": value} for value in values]


def get_default_hierarchy_value(
    catalog: dict[str, Any],
    field: str,
    run_id: str | None = None,
    *,
    subject: str | None = None,
    terrain: str | None = None,
    activity: str | None = None,
    experiment: str | None = None,
) -> str | None:
    """Return the hierarchy value from a current run when it fits the filters."""
    record = get_run_record(catalog, run_id)
    if (
        record is not None
        and _record_matches(
            record,
            subject=subject,
            terrain=terrain,
            activity=activity,
            experiment=experiment,
        )
    ):
        return str(record[field])
    options = get_hierarchy_options(
        catalog,
        field,
        subject=subject,
        terrain=terrain,
        activity=activity,
        experiment=experiment,
    )
    return str(options[0]["value"]) if options else None


def get_hierarchy_defaults(catalog: dict[str, Any], run_id: str | None) -> dict[str, str | None]:
    """Return hierarchy defaults for a selected run."""
    record = get_run_record(catalog, run_id)
    if record is not None:
        return {field: str(record[field]) for field in HIERARCHY_FIELDS}

    defaults: dict[str, str | None] = {}
    defaults["subject"] = get_default_hierarchy_value(catalog, "subject")
    defaults["terrain"] = get_default_hierarchy_value(catalog, "terrain", subject=defaults["subject"])
    defaults["activity"] = get_default_hierarchy_value(
        catalog,
        "activity",
        subject=defaults["subject"],
        terrain=defaults["terrain"],
    )
    defaults["experiment"] = get_default_hierarchy_value(
        catalog,
        "experiment",
        subject=defaults["subject"],
        terrain=defaults["terrain"],
        activity=defaults["activity"],
    )
    defaults["csv_file"] = get_default_hierarchy_value(
        catalog,
        "csv_file",
        subject=defaults["subject"],
        terrain=defaults["terrain"],
        activity=defaults["activity"],
        experiment=defaults["experiment"],
    )
    return defaults


def _filter_run_records(
    catalog: dict[str, Any],
    *,
    subject: str | None = None,
    terrain: str | None = None,
    activity: str | None = None,
    experiment: str | None = None,
) -> list[dict[str, Any]]:
    return [
        record
        for record in list_run_records(catalog)
        if _record_matches(
            record,
            subject=subject,
            terrain=terrain,
            activity=activity,
            experiment=experiment,
        )
    ]


def _record_matches(
    record: dict[str, Any],
    *,
    subject: str | None = None,
    terrain: str | None = None,
    activity: str | None = None,
    experiment: str | None = None,
) -> bool:
    filters = {
        "subject": subject,
        "terrain": terrain,
        "activity": activity,
        "experiment": experiment,
    }
    return all(value is None or str(record.get(field)) == str(value) for field, value in filters.items())


def get_group_options(catalog: dict[str, Any], run_id: str | None) -> list[dict[str, str]]:
    """Return dropdown options for signal groups within a run."""
    record = get_run_record(catalog, run_id)
    if record is None:
        return []
    return [
        {"label": record["group_labels"][group], "value": group}
        for group in record["available_groups"]
    ]


def get_default_group(catalog: dict[str, Any], run_id: str | None, *, slot: str | None = None) -> str | None:
    """Return a default group for a run."""
    record = get_run_record(catalog, run_id)
    if record is None or not record["available_groups"]:
        return None
    if slot == "b" and len(record["available_groups"]) > 1:
        return str(record["available_groups"][1])
    return str(record["available_groups"][0])


def get_trace_options(catalog: dict[str, Any], run_id: str | None, group: str | None) -> list[dict[str, str]]:
    """Return dropdown options for traces within a group."""
    record = get_run_record(catalog, run_id)
    if record is None or not group:
        return []
    options = record["trace_options_by_group"].get(group, [])
    return [dict(option) for option in options]


def get_default_trace(catalog: dict[str, Any], run_id: str | None, group: str | None, *, slot: str | None = None) -> str | None:
    """Return a default trace for a group."""
    record = get_run_record(catalog, run_id)
    if record is None or not group:
        return None
    traces = list(record["group_traces"].get(group, []))
    if not traces:
        return None
    if slot == "b" and "mode_1" in traces:
        return "mode_1"
    if "original_angle" in traces:
        return "original_angle"
    return str(traces[0])


def pretty_signal_label(name: str) -> str:
    """Convert an internal group name into a readable label."""
    text = str(name).replace("_", " ").strip()
    return " ".join(part.capitalize() for part in text.split())


def pretty_trace_label(trace_name: str) -> str:
    """Convert a trace key into a readable label."""
    if trace_name == "original_angle":
        return "Original Angle"
    return trace_name.replace("_", " ").title()


def read_mode_file_schema(mode_file_path: str | Path) -> dict[str, Any]:
    """Read the header schema for a mode file."""
    return _read_mode_file_schema_cached(str(Path(mode_file_path).resolve()))


@lru_cache(maxsize=256)
def _read_mode_file_schema_cached(mode_file_path_str: str) -> dict[str, Any]:
    path = Path(mode_file_path_str)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        columns = next(csv.reader(handle), [])

    group_modes: dict[str, set[str]] = {}
    for column in columns:
        match = GROUP_MODE_PATTERN.match(column)
        if not match:
            continue
        group_modes.setdefault(match.group("group"), set()).add(match.group("mode"))

    groups = sorted(group for group, modes in group_modes.items() if modes == {"1", "2", "3"})
    group_traces: dict[str, list[str]] = {}
    available_traces: set[str] = set()
    for group in groups:
        traces = ["mode_1", "mode_2", "mode_3"]
        if f"{group}_original_angle" in columns:
            traces = ["original_angle", *traces]
        group_traces[group] = traces
        available_traces.update(traces)

    return {
        "columns": columns,
        "groups": groups,
        "group_traces": group_traces,
        "available_traces": [trace for trace in TRACE_ORDER if trace in available_traces],
        "has_frame": "frame" in columns,
        "has_time": "time" in columns or "time_seconds" in columns,
        "has_markers": False,
    }
