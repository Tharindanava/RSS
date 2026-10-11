"""Atomic dashboard reference registry and signal-template utilities."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np

from .metadata import normalized_recording_identity


SCHEMA_VERSION = 1
TEMPLATE_POINTS = 101
ORIENTATION_THRESHOLD = 0.50


def _token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_") or "unknown"


def group_token(value: Any) -> str:
    """Public alias for the normalization used as ``mode1_templates`` dict keys."""
    return _token(value)


def scenario_fields(terrain: Any, direction: Any, footwear: Any, side: Any) -> dict[str, str]:
    terrain_token = _token(terrain)
    if "flat" in terrain_token:
        terrain_token = "flat"
    elif "stair" in terrain_token:
        terrain_token = "stairs"
    elif any(value in terrain_token for value in ("uneven", "rough")):
        terrain_token = "uneven"
    elif any(value in terrain_token for value in ("incline", "slope")):
        terrain_token = "incline"
    direction_token = _token(direction)
    direction_token = "up" if "up" in direction_token else "down" if "down" in direction_token else "level" if direction_token not in {"unknown", ""} else "unknown"
    return {
        "terrain": terrain_token,
        "direction": direction_token,
        "footwear": _token(footwear),
        "side": _token(side),
    }


def scenario_key(terrain: Any, direction: Any, footwear: Any, side: Any) -> str:
    fields = scenario_fields(terrain, direction, footwear, side)
    return "|".join(fields[name] for name in ("terrain", "direction", "footwear", "side"))


def side_from_group(group: str) -> str | None:
    value = _token(group)
    if value.startswith("left_"):
        return "left"
    if value.startswith("right_"):
        return "right"
    return None


def empty_registry() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "references": {}}


def load_registry(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return empty_registry()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("references"), dict):
        raise ValueError(f"Invalid reference registry: {source}")
    if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported reference registry schema: {payload.get('schema_version')}")
    return payload


def save_registry(registry: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = deepcopy(dict(registry))
    payload["schema_version"] = SCHEMA_VERSION
    payload.setdefault("references", {})
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def registry_checksum(path: str | Path) -> str:
    source = Path(path)
    if not source.exists():
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(source.read_bytes()).hexdigest()


def file_checksum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def reference_for(registry: Mapping[str, Any], terrain: Any, direction: Any, footwear: Any, side: Any) -> dict[str, Any] | None:
    value = registry.get("references", {}).get(scenario_key(terrain, direction, footwear, side))
    return deepcopy(value) if isinstance(value, dict) else None


def put_reference(registry: Mapping[str, Any], entry: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(registry))
    result.setdefault("schema_version", SCHEMA_VERSION)
    references = result.setdefault("references", {})
    scenario = entry["scenario"]
    key = scenario_key(scenario["terrain"], scenario["direction"], scenario["footwear"], entry["side"])
    references[key] = deepcopy(dict(entry))
    return result


def remove_reference(registry: Mapping[str, Any], terrain: Any, direction: Any, footwear: Any, side: Any) -> dict[str, Any]:
    result = deepcopy(dict(registry))
    result.setdefault("references", {}).pop(scenario_key(terrain, direction, footwear, side), None)
    return result


def reference_entry(
    *, terrain: Any, direction: Any, footwear: Any, side: str, source_path: str | Path,
    mode1_templates: Mapping[str, Iterable[float]], run_label: str = "", project_root: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(source_path).resolve()
    root = Path(project_root).resolve() if project_root is not None else None
    try:
        relative = str(source.relative_to(root)) if root else str(source)
    except ValueError:
        relative = str(source)
    fields = scenario_fields(terrain, direction, footwear, side)
    return {
        "scenario": {name: fields[name] for name in ("terrain", "direction", "footwear")},
        "side": fields["side"],
        "source": {
            "recording_identity": normalized_recording_identity(source),
            "path": str(source),
            "relative_path": relative,
            "sha256": file_checksum(source),
            "run_label": str(run_label),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode1_templates": {
            _token(group): standardize_template(template).tolist()
            for group, template in sorted(mode1_templates.items())
        },
        "reference_step": None,
    }


def is_reference_source(registry: Mapping[str, Any], source_path: str | Path) -> bool:
    target = Path(source_path).resolve()
    for entry in registry.get("references", {}).values():
        try:
            if Path(entry.get("source", {}).get("path", "")).resolve() == target:
                return True
        except (OSError, TypeError):
            continue
    return False


def standardize_template(values: Iterable[float], points: int = TEMPLATE_POINTS) -> np.ndarray:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    finite = np.isfinite(array)
    if finite.sum() < 2:
        raise ValueError("A reference template needs at least two finite samples")
    if not finite.all():
        array = np.interp(np.arange(array.size), np.flatnonzero(finite), array[finite])
    if array.size != points:
        array = np.interp(np.linspace(0.0, 1.0, points), np.linspace(0.0, 1.0, array.size), array)
    scale = float(np.std(array))
    if not np.isfinite(scale) or scale <= 1.0e-12:
        raise ValueError("A reference template must vary over the selected interval")
    return (array - float(np.mean(array))) / scale


def median_interval_template(values: Iterable[float], intervals: Iterable[tuple[int, int]], points: int = TEMPLATE_POINTS) -> np.ndarray:
    signal = np.asarray(list(values), dtype=float)
    templates = [
        standardize_template(signal[max(0, int(start)):min(len(signal), int(end)) + 1], points)
        for start, end in intervals if int(end) > int(start)
    ]
    if not templates:
        raise ValueError("No valid same-side intervals are available for a reference template")
    return standardize_template(np.median(np.vstack(templates), axis=0), points)


def template_correlation(first: Iterable[float], second: Iterable[float]) -> float:
    left, right = standardize_template(first), standardize_template(second)
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else float("nan")


def orientation_verdict(correlation: float, threshold: float = ORIENTATION_THRESHOLD) -> str:
    if not np.isfinite(correlation) or abs(float(correlation)) < float(threshold):
        return "undecided"
    return "consistent" if correlation >= 0.0 else "flip"
