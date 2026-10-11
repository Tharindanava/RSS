"""Loading and schema validation of wide-format multi-IMU CSV recordings."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import numpy as np
import pandas as pd

from rss.sensor_layout import required_imu_columns


@dataclass
class LoadedInput:
    csv_path: str
    dataframe: pd.DataFrame
    frame_index: np.ndarray
    time_seconds: np.ndarray | None


def _normalize_imu_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the supported header variants (``OrbiX-Node-<n>_...``, ``Device_<n>_...``) to ``device<n>_...``."""
    columns_by_normalized_name: dict[str, list[str]] = {}
    for column in df.columns:
        if isinstance(column, str):
            normalized_name = re.sub(
                r"^orbix-node-(\d+)",
                r"device\1",
                column,
                flags=re.IGNORECASE,
            )
            normalized_name = re.sub(
                r"^device_",
                "device",
                normalized_name,
                flags=re.IGNORECASE,
            ).casefold()
            columns_by_normalized_name.setdefault(normalized_name, []).append(column)

    rename: dict[str, str] = {}
    ambiguous: list[str] = []
    for canonical in required_imu_columns():
        matches = columns_by_normalized_name.get(canonical.casefold(), [])
        if len(matches) > 1:
            ambiguous.append(f"{canonical} ({', '.join(matches)})")
        elif len(matches) == 1 and matches[0] != canonical:
            rename[matches[0]] = canonical

    if ambiguous:
        preview = ", ".join(ambiguous[:3])
        if len(ambiguous) > 3:
            preview += f", ... ({len(ambiguous)} ambiguous)"
        raise ValueError(
            "Input CSV contains multiple header variants for required IMU columns. "
            f"Keep only one variant of each column: {preview}"
        )

    return df.rename(columns=rename) if rename else df


def _resolve_time_seconds(df: pd.DataFrame, *, sample_rate_hz: float, include_time_seconds: bool) -> np.ndarray | None:
    if not include_time_seconds:
        return None
    if "Timestamp" in df.columns:
        timestamp = df["Timestamp"]
        numeric_timestamp = pd.to_numeric(timestamp, errors="coerce")
        if numeric_timestamp.notna().all():
            sample_counter = numeric_timestamp.to_numpy(dtype=float)
            if sample_counter.size == 0:
                return sample_counter
            return (sample_counter - sample_counter[0]) / float(sample_rate_hz)

        datetime_timestamp = pd.to_datetime(timestamp, errors="coerce", utc=False)
        if datetime_timestamp.notna().all():
            elapsed = datetime_timestamp - datetime_timestamp.iloc[0]
            seconds = elapsed.dt.total_seconds().to_numpy(dtype=float)
            return seconds
    dt = 1.0 / float(sample_rate_hz)
    return np.arange(len(df), dtype=float) * dt


def load_and_validate_input(
    csv_path: str,
    *,
    sample_rate_hz: float,
    include_time_seconds: bool,
    allow_partial_imu: bool = False,
) -> LoadedInput:
    abs_path = os.path.abspath(csv_path)
    df = pd.read_csv(abs_path)
    df = _normalize_imu_column_names(df)
    if not allow_partial_imu:
        missing = [c for c in required_imu_columns() if c not in df.columns]
        if missing:
            first_missing = ", ".join(missing[:8])
            if len(missing) > 8:
                first_missing += f", ... ({len(missing)} missing)"
            raise ValueError(
                f"Input CSV is missing required IMU columns for the segment-to-sensor mapping. Missing: {first_missing}"
            )
    frame_index = np.arange(len(df), dtype=int)
    time_seconds = _resolve_time_seconds(
        df=df,
        sample_rate_hz=sample_rate_hz,
        include_time_seconds=include_time_seconds,
    )
    return LoadedInput(
        csv_path=abs_path,
        dataframe=df,
        frame_index=frame_index,
        time_seconds=time_seconds,
    )
