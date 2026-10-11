"""CSV loading and column resolution for rotating Poincare analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


TIME_COLUMN_CANDIDATES = ("time", "time_seconds", "Timestamp")
INDEX_COLUMN_NAMES = {"frame", "index", "sample", "sample_index"}


@dataclass(frozen=True)
class LoadedSignalFile:
    """A CSV dataframe with resolved time and signal columns."""

    path: Path
    dataframe: pd.DataFrame
    time_column: str | None
    time_values: np.ndarray | None
    signal_columns: tuple[str, ...]


def _case_insensitive_column(columns: pd.Index, requested: str) -> str | None:
    matches = [str(column) for column in columns if str(column).casefold() == requested.casefold()]
    if len(matches) > 1:
        raise ValueError(f"Column name {requested!r} is ambiguous under case-insensitive matching.")
    return matches[0] if matches else None


def _resolve_time_column(dataframe: pd.DataFrame, time_col: str | None) -> str | None:
    if time_col is not None:
        resolved = _case_insensitive_column(dataframe.columns, time_col)
        if resolved is None:
            raise ValueError(f"Requested time column {time_col!r} was not found.")
        return resolved
    for candidate in TIME_COLUMN_CANDIDATES:
        resolved = _case_insensitive_column(dataframe.columns, candidate)
        if resolved is not None:
            return resolved
    return None


def _parse_time_values(series: pd.Series, column_name: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    nonmissing = series.notna()
    if bool(numeric[nonmissing].notna().all()):
        return numeric.to_numpy(dtype=float)

    datetimes = pd.to_datetime(series, errors="coerce", utc=True)
    if bool(datetimes[nonmissing].notna().all()):
        valid_datetimes = datetimes.dropna()
        if valid_datetimes.empty:
            return np.full(len(series), np.nan, dtype=float)
        origin = valid_datetimes.iloc[0]
        return (datetimes - origin).dt.total_seconds().to_numpy(dtype=float)

    raise ValueError(
        f"Time column {column_name!r} must contain numeric values or parseable datetimes."
    )


def _resolve_signal_columns(
    dataframe: pd.DataFrame,
    time_column: str | None,
    signal_cols: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if signal_cols:
        resolved: list[str] = []
        for requested in signal_cols:
            column = _case_insensitive_column(dataframe.columns, requested)
            if column is None:
                raise ValueError(f"Requested signal column {requested!r} was not found.")
            if column == time_column:
                raise ValueError(f"Time column {column!r} cannot also be a signal column.")
            if column not in resolved:
                resolved.append(column)
        return tuple(resolved)

    inferred: list[str] = []
    for column in dataframe.columns:
        name = str(column)
        if name == time_column or name.casefold() in INDEX_COLUMN_NAMES:
            continue
        if is_numeric_dtype(dataframe[column]):
            inferred.append(name)
    if not inferred:
        raise ValueError(
            "No numeric signal columns were found after excluding time and index-like columns."
        )
    return tuple(inferred)


def load_signal_file(
    path: str | Path,
    time_col: str | None = None,
    signal_cols: list[str] | tuple[str, ...] | None = None,
) -> LoadedSignalFile:
    """Load a CSV and resolve its time and angle-signal columns."""

    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")
    if csv_path.suffix.casefold() != ".csv":
        raise ValueError(f"Only CSV input is supported: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    if dataframe.empty:
        raise ValueError(f"Input CSV is empty: {csv_path}")
    resolved_time = _resolve_time_column(dataframe, time_col)
    time_values = (
        _parse_time_values(dataframe[resolved_time], resolved_time)
        if resolved_time is not None
        else None
    )
    signals = _resolve_signal_columns(dataframe, resolved_time, signal_cols)
    return LoadedSignalFile(
        path=csv_path,
        dataframe=dataframe,
        time_column=resolved_time,
        time_values=time_values,
        signal_columns=signals,
    )
