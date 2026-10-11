"""Transfer-entropy helpers for dashboard-only pairwise analysis."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gait_analysis.te_estimator import quantile_discretize


JOINT_ORDER = ("Hip", "Knee", "Ankle")


MIN_EMBEDDED_SAMPLES = 25
MATRIX_DEFAULTS: dict[str, Any] = {
    "bins": 8,
    "binning": "quantile",
    "te_mode": "legacy",
    "embedding_selection": "global",
    "history_grid": (1, 2, 3, 4, 5, 6, 8, 10),
    "tau_grid": (1, 2, 3, 5),
    "dmax_samples": 30,
    "n_surrogates": 100,
    "alpha": 0.05,
    "detrend_enabled": False,
    "smooth_window": None,
}


@dataclass(frozen=True)
class ContextSpec:
    """One lagged-context specification."""

    lag: int
    length: int
    stride: int


def clear_caches() -> None:
    """Clear cached matrix computations."""
    _compute_te_matrix_cached.cache_clear()


def build_te_samples(
    source: np.ndarray,
    target: np.ndarray,
    *,
    source_context: ContextSpec,
    receiver_context: ContextSpec,
) -> list[tuple[int, tuple[int, ...], tuple[int, ...]]]:
    """Build TE samples for generalized source and receiver histories."""
    source = np.asarray(source, dtype=int).reshape(-1)
    target = np.asarray(target, dtype=int).reshape(-1)
    if source.shape != target.shape:
        raise ValueError("Source and target must have the same length.")

    max_source_lag = source_context.lag + (source_context.length - 1) * source_context.stride
    max_receiver_lag = receiver_context.lag + (receiver_context.length - 1) * receiver_context.stride
    max_lag = max(max_source_lag, max_receiver_lag)
    if source.size <= max_lag:
        return []

    samples: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
    for index in range(max_lag, target.size):
        y_future = int(target[index])
        y_past = tuple(
            int(target[index - receiver_context.lag - offset * receiver_context.stride])
            for offset in range(receiver_context.length)
        )
        x_past = tuple(
            int(source[index - source_context.lag - offset * source_context.stride])
            for offset in range(source_context.length)
        )
        samples.append((y_future, y_past, x_past))
    return samples


def estimate_transfer_entropy_from_samples(
    samples: list[tuple[int, tuple[int, ...], tuple[int, ...]]]
) -> tuple[float, int]:
    """Estimate raw transfer entropy from discrete lagged samples."""
    total = len(samples)
    if total == 0:
        return math.nan, 0

    joint_counts: dict[tuple[int, tuple[int, ...], tuple[int, ...]], int] = {}
    ypast_x_counts: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
    yfuture_ypast_counts: dict[tuple[int, tuple[int, ...]], int] = {}
    ypast_counts: dict[tuple[int, ...], int] = {}

    for y_future, y_past, x_past in samples:
        joint_counts[(y_future, y_past, x_past)] = joint_counts.get((y_future, y_past, x_past), 0) + 1
        ypast_x_counts[(y_past, x_past)] = ypast_x_counts.get((y_past, x_past), 0) + 1
        yfuture_ypast_counts[(y_future, y_past)] = yfuture_ypast_counts.get((y_future, y_past), 0) + 1
        ypast_counts[y_past] = ypast_counts.get(y_past, 0) + 1

    te_bits = 0.0
    for (y_future, y_past, x_past), count in joint_counts.items():
        probability_joint = count / total
        probability_y_given_yx = count / ypast_x_counts[(y_past, x_past)]
        probability_y_given_y = yfuture_ypast_counts[(y_future, y_past)] / ypast_counts[y_past]
        te_bits += probability_joint * math.log2(probability_y_given_yx / probability_y_given_y)
    return te_bits, total


def compute_pairwise_transfer_entropy(
    source_values: np.ndarray,
    target_values: np.ndarray,
    *,
    source_context: ContextSpec,
    receiver_context: ContextSpec,
    bins: int,
) -> dict[str, Any]:
    """Compute dashboard pairwise transfer entropy for two continuous traces."""
    _validate_context(source_context)
    _validate_context(receiver_context)
    bins = max(2, int(bins))

    source_values = np.asarray(source_values, dtype=float).reshape(-1)
    target_values = np.asarray(target_values, dtype=float).reshape(-1)
    if source_values.shape != target_values.shape:
        raise ValueError("Source and target signals must have the same length.")

    mask = np.isfinite(source_values) & np.isfinite(target_values)
    paired_source = source_values[mask]
    paired_target = target_values[mask]
    valid_pairs = int(paired_source.size)
    if valid_pairs == 0:
        return {
            "available": False,
            "message": "No finite sample pairs were available for transfer entropy.",
            "te_bits": math.nan,
            "valid_pairs": 0,
            "n_samples_used": 0,
            "bins": bins,
            "source_context": _context_to_dict(source_context),
            "receiver_context": _context_to_dict(receiver_context),
        }

    discrete_source = normalize_and_discretize(paired_source, bins=bins)
    discrete_target = normalize_and_discretize(paired_target, bins=bins)

    samples = build_te_samples(
        discrete_source,
        discrete_target,
        source_context=source_context,
        receiver_context=receiver_context,
    )
    te_bits, n_samples_used = estimate_transfer_entropy_from_samples(samples)
    if n_samples_used < MIN_EMBEDDED_SAMPLES or math.isnan(te_bits):
        return {
            "available": False,
            "message": (
                f"Only {n_samples_used} embedded samples were available after context trimming; "
                f"at least {MIN_EMBEDDED_SAMPLES} are required."
            ),
            "te_bits": math.nan,
            "valid_pairs": valid_pairs,
            "n_samples_used": int(n_samples_used),
            "bins": bins,
            "source_context": _context_to_dict(source_context),
            "receiver_context": _context_to_dict(receiver_context),
        }

    return {
        "available": True,
        "message": "Transfer entropy computed successfully.",
        "te_bits": float(te_bits),
        "valid_pairs": valid_pairs,
        "n_samples_used": int(n_samples_used),
        "bins": bins,
        "source_context": _context_to_dict(source_context),
        "receiver_context": _context_to_dict(receiver_context),
    }


def compute_te_matrix_for_mode_file(mode_file_path: str | Path) -> dict[str, Any]:
    """Compute cached lower-limb raw-TE matrices for a mode file."""
    path = Path(mode_file_path).resolve()
    return _compute_te_matrix_cached(str(path), int(path.stat().st_mtime_ns))


@lru_cache(maxsize=32)
def _compute_te_matrix_cached(mode_file_path_str: str, mtime_ns: int) -> dict[str, Any]:
    del mtime_ns
    path = Path(mode_file_path_str)
    frame = pd.read_csv(path)
    side_results: list[dict[str, Any]] = []
    errors: list[str] = []
    side_groups = {
        "left": {
            "Hip": ("left_hip", "left_hip_to_knee"),
            "Knee": ("left_knee", "left_knee_to_ankle"),
            "Ankle": ("left_ankle", "left_ankle_to_foot"),
        },
        "right": {
            "Hip": ("right_hip", "right_hip_to_knee"),
            "Knee": ("right_knee", "right_knee_to_ankle"),
            "Ankle": ("right_ankle", "right_ankle_to_foot"),
        },
    }
    for side, groups in side_groups.items():
        try:
            selected: dict[str, str] = {}
            for joint, candidates in groups.items():
                column = next((f"{group}_mode_1" for group in candidates if f"{group}_mode_1" in frame), None)
                if column is not None:
                    selected[joint] = column
            if len(selected) != len(JOINT_ORDER):
                continue
            data = frame[[selected[joint] for joint in JOINT_ORDER]].apply(pd.to_numeric, errors="coerce")
            data = data[np.all(np.isfinite(data.to_numpy(float)), axis=1)]
            n_samples_aligned = len(data)
            if n_samples_aligned < MIN_EMBEDDED_SAMPLES + 1:
                raise ValueError("too few jointly finite samples")
            matrix = np.full((len(JOINT_ORDER), len(JOINT_ORDER)), np.nan, dtype=float)
            for row_index, source_joint in enumerate(JOINT_ORDER):
                for col_index, target_joint in enumerate(JOINT_ORDER):
                    if source_joint == target_joint:
                        continue
                    estimate = compute_pairwise_transfer_entropy(
                        data[selected[source_joint]].to_numpy(float), data[selected[target_joint]].to_numpy(float),
                        source_context=ContextSpec(1, 1, 1), receiver_context=ContextSpec(1, 1, 1),
                        bins=int(MATRIX_DEFAULTS["bins"]),
                    )
                    matrix[row_index, col_index] = float(estimate["te_bits"])
            side_results.append(
                {
                    "side": side,
                    "matrix": matrix.tolist(),
                    "global_embedding": {"Ly": 1, "tau": 1},
                    "n_samples_aligned": int(n_samples_aligned),
                }
            )
        except Exception as exc:
            errors.append(f"{side}: {exc}")

    if not side_results:
        message = "No complete lower-limb Hip/Knee/Ankle set was available for a TE matrix."
        if errors:
            message = f"{message} {errors[0]}"
        return {
            "available": False,
            "message": message,
            "results": [],
            "errors": errors,
            "defaults": dict(MATRIX_DEFAULTS),
        }

    return {
        "available": True,
        "message": "Lower-limb exploratory raw-TE matrix computed with the gait toolkit discretizer.",
        "results": side_results,
        "errors": errors,
        "defaults": dict(MATRIX_DEFAULTS),
    }


def matrix_results_to_rows(matrix_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a matrix result into row records for CSV export."""
    rows: list[dict[str, Any]] = []
    for side_record in matrix_result.get("results", []):
        matrix = np.asarray(side_record["matrix"], dtype=float)
        for row_index, source_joint in enumerate(JOINT_ORDER):
            for col_index, target_joint in enumerate(JOINT_ORDER):
                rows.append(
                    {
                        "side": side_record["side"],
                        "source_joint": source_joint,
                        "target_joint": target_joint,
                        "te_raw": float(matrix[row_index, col_index]) if np.isfinite(matrix[row_index, col_index]) else math.nan,
                        "n_samples_aligned": int(side_record["n_samples_aligned"]),
                        "selected_Ly": int(side_record["global_embedding"]["Ly"]),
                        "selected_tau": int(side_record["global_embedding"]["tau"]),
                    }
                )
    return rows


def _validate_context(context: ContextSpec) -> None:
    if context.lag <= 0 or context.length <= 0 or context.stride <= 0:
        raise ValueError("Transfer-entropy lag, length, and stride must all be positive integers.")


def _context_to_dict(context: ContextSpec) -> dict[str, int]:
    return {
        "lag": int(context.lag),
        "length": int(context.length),
        "stride": int(context.stride),
    }


def normalize_and_discretize(values: np.ndarray, *, bins: int) -> np.ndarray:
    """Quantile-discretize one finite signal using the analysis package primitive.

    Public so other dashboard modules can discretize exactly the way
    ``compute_pairwise_transfer_entropy`` does, instead of reaching for the
    private helpers and drifting out of step with it.
    """
    prepared = np.asarray(values, dtype=float).reshape(-1)
    return quantile_discretize(prepared, bins=max(2, int(bins)))


def _zscore_normalize(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std


def _uniform_discretize(values: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    unique_count = int(np.unique(values).size)
    if unique_count <= 1:
        return np.zeros(values.shape[0], dtype=int)
    effective_bins = max(2, min(int(bins), unique_count))
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value <= min_value:
        return np.zeros(values.shape[0], dtype=int)
    edges = np.linspace(min_value, max_value, effective_bins + 1)
    interior = edges[1:-1]
    return np.digitize(values, bins=interior, right=False).astype(int)
