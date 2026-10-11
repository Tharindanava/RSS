"""Small deterministic utilities shared by the analysis stages."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def hash_mapping(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def hash_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_dataframe(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if target.name.endswith(".csv.gz") else target.suffix
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=suffix, dir=target.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        compression = "gzip" if target.name.endswith(".gz") else None
        frame.to_csv(temp, index=False, compression=compression)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def atomic_write_json(value: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".json", dir=target.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def scaled_mad(values: Iterable[float], scale: float = 1.4826) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return float("nan")
    median = np.median(array)
    return float(scale * np.median(np.abs(array - median)))


def robust_zscore(frame: pd.DataFrame, center: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    numeric = frame.astype(float)
    centers = numeric.median(axis=0) if center is None else center.reindex(numeric.columns)
    scales = numeric.apply(lambda column: scaled_mad(column))
    fallback = numeric.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    scales = scales.where(np.isfinite(scales) & (scales > 0.0), fallback)
    return (numeric - centers) / scales, centers, scales


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    out = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not valid.size:
        return out
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out

