"""Fixed-context discrete transfer-entropy calculations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
import pandas as pd


MIN_EMBEDDED_SAMPLES = 25
SEGMENT_ORDER: tuple[str, ...] = (
    "Pelvis-Hip",
    "Hip-Knee",
    "Knee-Ankle",
    "Ankle-Foot",
)
SIDE_COLUMNS: dict[str, dict[str, str]] = {
    "left": {
        "Pelvis-Hip": "pelvis_to_left_hip_mode_1",
        "Hip-Knee": "left_hip_to_knee_mode_1",
        "Knee-Ankle": "left_knee_to_ankle_mode_1",
        "Ankle-Foot": "left_ankle_to_foot_mode_1",
    },
    "right": {
        "Pelvis-Hip": "pelvis_to_right_hip_mode_1",
        "Hip-Knee": "right_hip_to_knee_mode_1",
        "Knee-Ankle": "right_knee_to_ankle_mode_1",
        "Ankle-Foot": "right_ankle_to_foot_mode_1",
    },
}


@dataclass(frozen=True)
class ContextSpec:
    """Shared source and receiver history specification."""

    lag: int = 1
    length: int = 1
    stride: int = 1

    def validate(self) -> None:
        for name, value in (
            ("lag", self.lag),
            ("context length", self.length),
            ("stride", self.stride),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
                raise ValueError(f"{name.capitalize()} must be a positive integer.")

    @property
    def max_lag(self) -> int:
        self.validate()
        return int(self.lag + (self.length - 1) * self.stride)


@dataclass(frozen=True)
class SideTransferEntropyResult:
    """One side's matrix and edge-level metadata."""

    side: str
    matrix: pd.DataFrame
    edges: pd.DataFrame
    n_input_rows: int
    n_samples_aligned: int
    effective_bins: dict[str, int]


def build_te_samples(
    source: np.ndarray,
    target: np.ndarray,
    context: ContextSpec,
) -> list[tuple[int, tuple[int, ...], tuple[int, ...]]]:
    """Build ``(future target, target history, source history)`` samples."""
    context.validate()
    source_values = np.asarray(source, dtype=int).reshape(-1)
    target_values = np.asarray(target, dtype=int).reshape(-1)
    if source_values.shape != target_values.shape:
        raise ValueError("Source and target must have the same length.")

    samples: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
    for index in range(context.max_lag, target_values.size):
        target_history = tuple(
            int(target_values[index - context.lag - offset * context.stride])
            for offset in range(context.length)
        )
        source_history = tuple(
            int(source_values[index - context.lag - offset * context.stride])
            for offset in range(context.length)
        )
        samples.append((int(target_values[index]), target_history, source_history))
    return samples


def estimate_transfer_entropy_from_samples(
    samples: list[tuple[int, tuple[int, ...], tuple[int, ...]]],
) -> float:
    """Estimate raw discrete transfer entropy in bits."""
    total = len(samples)
    if total == 0:
        return math.nan

    joint_counts = Counter(samples)
    target_source_history_counts = Counter(
        (target_history, source_history)
        for _, target_history, source_history in samples
    )
    future_target_history_counts = Counter(
        (future, target_history)
        for future, target_history, _ in samples
    )
    target_history_counts = Counter(target_history for _, target_history, _ in samples)

    transfer_entropy = 0.0
    for (future, target_history, source_history), count in joint_counts.items():
        probability_joint = count / total
        probability_with_source = count / target_source_history_counts[(target_history, source_history)]
        probability_without_source = (
            future_target_history_counts[(future, target_history)]
            / target_history_counts[target_history]
        )
        transfer_entropy += probability_joint * math.log2(
            probability_with_source / probability_without_source
        )
    return float(transfer_entropy)


def discretize_signal(
    values: np.ndarray,
    bins: int,
    method: str,
) -> tuple[np.ndarray, int]:
    """Discretize one finite signal and return codes plus occupied-bin count."""
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or int(bins) < 2:
        raise ValueError("Bins must be an integer greater than or equal to 2.")
    if method not in {"quantile", "uniform"}:
        raise ValueError("Binning method must be 'quantile' or 'uniform'.")

    numeric = np.asarray(values, dtype=float).reshape(-1)
    if numeric.size == 0:
        raise ValueError("Cannot discretize an empty signal.")
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Signal values must be finite before discretization.")
    if np.unique(numeric).size <= 1:
        return np.zeros(numeric.size, dtype=int), 1

    if method == "quantile":
        codes = pd.qcut(
            numeric,
            q=min(int(bins), int(np.unique(numeric).size)),
            labels=False,
            duplicates="drop",
        )
        discrete = np.asarray(codes, dtype=int)
    else:
        minimum = float(np.min(numeric))
        maximum = float(np.max(numeric))
        edges = np.linspace(minimum, maximum, int(bins) + 1)
        discrete = np.digitize(numeric, edges[1:-1], right=False).astype(int)

    return discrete, int(np.unique(discrete).size)


def compute_side_transfer_entropy(
    frame: pd.DataFrame,
    *,
    side: str,
    bins: int,
    binning: str,
    context: ContextSpec,
    min_embedded_samples: int = MIN_EMBEDDED_SAMPLES,
) -> SideTransferEntropyResult:
    """Compute all directed off-diagonal TE edges for one lower-limb side."""
    context.validate()
    if side not in SIDE_COLUMNS:
        raise ValueError(f"Side must be one of {tuple(SIDE_COLUMNS)}.")
    if min_embedded_samples <= 0:
        raise ValueError("Minimum embedded samples must be positive.")

    column_map = SIDE_COLUMNS[side]
    missing = [column for column in column_map.values() if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required {side} mode-1 columns: {missing}")

    numeric = pd.DataFrame(
        {
            segment: pd.to_numeric(frame[column], errors="coerce")
            for segment, column in column_map.items()
        }
    )
    finite_mask = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    aligned = numeric.loc[finite_mask, list(SEGMENT_ORDER)].reset_index(drop=True)
    n_samples_aligned = int(len(aligned))
    n_samples_used = n_samples_aligned - context.max_lag
    if n_samples_used < int(min_embedded_samples):
        raise ValueError(
            f"Only {n_samples_used} embedded samples remain after context trimming; "
            f"at least {min_embedded_samples} are required."
        )

    discrete: dict[str, np.ndarray] = {}
    effective_bins: dict[str, int] = {}
    for segment in SEGMENT_ORDER:
        discrete[segment], effective_bins[segment] = discretize_signal(
            aligned[segment].to_numpy(dtype=float),
            bins=bins,
            method=binning,
        )

    matrix_values = np.full((len(SEGMENT_ORDER), len(SEGMENT_ORDER)), np.nan, dtype=float)
    edge_rows: list[dict[str, object]] = []
    for source_index, source_segment in enumerate(SEGMENT_ORDER):
        for target_index, target_segment in enumerate(SEGMENT_ORDER):
            if source_segment == target_segment:
                continue
            samples = build_te_samples(
                discrete[source_segment],
                discrete[target_segment],
                context=context,
            )
            te_bits = estimate_transfer_entropy_from_samples(samples)
            matrix_values[source_index, target_index] = te_bits
            edge_rows.append(
                {
                    "side": side,
                    "source_segment": source_segment,
                    "source_column": column_map[source_segment],
                    "target_segment": target_segment,
                    "target_column": column_map[target_segment],
                    "te_bits": te_bits,
                    "n_input_rows": int(len(frame)),
                    "n_samples_aligned": n_samples_aligned,
                    "n_samples_used": len(samples),
                    "bins_requested": int(bins),
                    "source_effective_bins": effective_bins[source_segment],
                    "target_effective_bins": effective_bins[target_segment],
                    "binning": binning,
                    "lag": int(context.lag),
                    "context_length": int(context.length),
                    "stride": int(context.stride),
                    "max_context_lag": int(context.max_lag),
                }
            )

    matrix = pd.DataFrame(matrix_values, index=SEGMENT_ORDER, columns=SEGMENT_ORDER)
    matrix.index.name = "source_segment"
    return SideTransferEntropyResult(
        side=side,
        matrix=matrix,
        edges=pd.DataFrame.from_records(edge_rows),
        n_input_rows=int(len(frame)),
        n_samples_aligned=n_samples_aligned,
        effective_bins=effective_bins,
    )
