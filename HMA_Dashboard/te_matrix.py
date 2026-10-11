"""Directed transfer-entropy matrix over every limb-angle signal in one trial.

The dashboard already had a transfer-entropy matrix, but it is structurally a
3x3-per-body-side view: ``transfer_entropy.compute_te_matrix_for_mode_file``
delegates to the ``gait_analysis`` toolkit and indexes its results by the
fixed ``JOINT_ORDER = ("Hip", "Knee", "Ankle")``. No parameter widens that, so
this module computes the whole-body version from scratch, reusing the
dashboard-native estimator (``transfer_entropy.build_te_samples`` /
``estimate_transfer_entropy_from_samples`` /
``transfer_entropy.normalize_and_discretize``).

Two implementation details are load-bearing:

* **One joint finite mask across every selected segment**, the way
  ``limb_transfer_entropy.core.compute_side_transfer_entropy`` does per side.
  Masking each pair separately would give every cell a different sample count
  and make the heatmap's cells incomparable.
* **Discretize once per segment** (N calls), not once per pair (2N^2).

Cost on a real 18-segment leaf: ~1.8 s for 306 directed pairs, essentially
independent of bins and lag over the ranges tried (8-25 bins, lag 1-50). That is
fast per computation but it blocks the single Flask worker, so it is wired to an
explicit Compute button rather than to the cascading run dropdowns.

Dashboard defaults are 25 bins and a lag of 50 samples. Note that with 25 bins
the joint (future, past, source) state space is ~15.6k cells against ~4k samples,
so the plug-in estimate is strongly upward-biased: independently shuffled
segments (true TE = 0) read ~1.9 bits, above the real-data mean. Treat absolute
values as bias-dominated at these settings; the ordering of cells is the
informative part.
"""

from __future__ import annotations

import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from HMA_Dashboard import catalog
from HMA_Dashboard import signals
from HMA_Dashboard.transfer_entropy import (
    ContextSpec,
    build_te_samples,
    estimate_transfer_entropy_from_samples,
    normalize_and_discretize,
)


DEFAULT_BINS = 25
#: History lag in samples, applied to both source and receiver (length and
#: stride stay 1). 50 samples is 0.5 s at the dashboard's 100 Hz default.
DEFAULT_LAG = 50
DEFAULT_TRACE = "mode_1"
DEFAULT_CONTEXT = ContextSpec(lag=DEFAULT_LAG, length=1, stride=1)

#: The two regions the matrix is boxed into. The split is head-to-foot at the
#: pelvis: the neck, spine above it and the arms are "upper"; the pelvis
#: segments, the lower spine and the legs are "lower". Within a region the order
#: runs head to foot with left before right at each level, so mirrored segments
#: sit side by side and each region is one contiguous block.
UPPER_LIMB_SEGMENTS: tuple[str, ...] = (
    "neck_to_left_shoulder",
    "neck_to_right_shoulder",
    "mid_to_upper_spine",
    "left_shoulder_to_elbow",
    "right_shoulder_to_elbow",
    "left_elbow_to_wrist",
    "right_elbow_to_wrist",
    "left_wrist_to_hand",
    "right_wrist_to_hand",
)
LOWER_LIMB_SEGMENTS: tuple[str, ...] = (
    "pelvis_to_lower_spine",
    "pelvis_to_left_hip",
    "pelvis_to_right_hip",
    "left_hip_to_knee",
    "right_hip_to_knee",
    "left_knee_to_ankle",
    "right_knee_to_ankle",
    "left_ankle_to_foot",
    "right_ankle_to_foot",
)
HEAD_TO_FOOT_SEGMENTS: tuple[str, ...] = (*UPPER_LIMB_SEGMENTS, *LOWER_LIMB_SEGMENTS)
_REGION_BY_SEGMENT: dict[str, str] = {
    **{segment: "upper" for segment in UPPER_LIMB_SEGMENTS},
    **{segment: "lower" for segment in LOWER_LIMB_SEGMENTS},
}
#: A one-segment matrix has no off-diagonal cells to show.
MIN_SEGMENTS = 2
MIN_EMBEDDED_SAMPLES = 25


def clear_caches() -> None:
    """Clear cached all-segment matrices."""
    _compute_all_segment_te_matrix_cached.cache_clear()


def body_region(segment: str) -> str | None:
    """``"upper"`` or ``"lower"`` for a known segment, ``None`` for any other."""
    return _REGION_BY_SEGMENT.get(str(segment))


def resolve_matrix_segments(available_groups: Sequence[str]) -> tuple[str, ...]:
    """Order the present groups head to foot, keeping any unrecognised ones.

    ``HEAD_TO_FOOT_SEGMENTS`` puts every upper segment before every lower one;
    the sorted tail keeps joint-family files (``right_hip``, ``right_knee``, ...)
    working, since none of their group names appear in it.
    """
    present = list(dict.fromkeys(str(group) for group in available_groups))
    ordered = [segment for segment in HEAD_TO_FOOT_SEGMENTS if segment in present]
    remainder = sorted(group for group in present if group not in set(ordered))
    return tuple([*ordered, *remainder])


def compute_all_segment_te_matrix(
    mode_file_path: str | Path,
    *,
    segments: Sequence[str] | None = None,
    trace_name: str = DEFAULT_TRACE,
    bins: int = DEFAULT_BINS,
    source_context: ContextSpec = DEFAULT_CONTEXT,
    receiver_context: ContextSpec = DEFAULT_CONTEXT,
    default_fs: float | None = None,
) -> dict[str, Any]:
    """Directed N x N transfer entropy between every segment's ``trace_name``."""
    path = Path(mode_file_path).resolve()
    if segments is None:
        try:
            schema = catalog.read_mode_file_schema(path)
        except (OSError, ValueError) as error:
            return _unavailable(f"Could not read the mode file: {error}", trace_name, bins)
        segments = resolve_matrix_segments(schema["groups"])
    segment_tuple = tuple(str(segment) for segment in segments)
    if len(segment_tuple) < MIN_SEGMENTS:
        return _unavailable(
            "At least two signal groups are needed for a transfer-entropy matrix; "
            f"this file has {len(segment_tuple)}.",
            trace_name,
            bins,
        )

    try:
        # mtime read here, not taken from the catalog store: that snapshot is
        # stale after a manual flip rewrites the file.
        mtime_ns = path.stat().st_mtime_ns
    except OSError as error:
        return _unavailable(f"Could not stat the mode file: {error}", trace_name, bins)

    return _compute_all_segment_te_matrix_cached(
        str(path),
        mtime_ns,
        segment_tuple,
        str(trace_name),
        max(2, int(bins)),
        (source_context.lag, source_context.length, source_context.stride),
        (receiver_context.lag, receiver_context.length, receiver_context.stride),
        "" if default_fs is None else f"{float(default_fs):.12g}",
    )


@lru_cache(maxsize=16)
def _compute_all_segment_te_matrix_cached(
    mode_file_path_str: str,
    mtime_ns: int,
    segments: tuple[str, ...],
    trace_name: str,
    bins: int,
    source_key: tuple[int, int, int],
    receiver_key: tuple[int, int, int],
    fs_key: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    path = Path(mode_file_path_str)
    source_context = ContextSpec(*source_key)
    receiver_context = ContextSpec(*receiver_key)
    default_fs = float(fs_key) if fs_key else None

    try:
        group_data = signals.load_all_group_mode_data(path, default_fs=default_fs)
    except (OSError, ValueError) as error:
        return _unavailable(f"Could not load the mode file: {error}", trace_name, bins)

    columns: dict[str, np.ndarray] = {}
    skipped: list[dict[str, str]] = []
    for segment in segments:
        values = _segment_values(group_data, segment, trace_name)
        if values is None:
            skipped.append({"segment": segment, "reason": f"no '{trace_name}' trace"})
            continue
        columns[segment] = values

    usable = [segment for segment in segments if segment in columns]
    if len(usable) < MIN_SEGMENTS:
        return _unavailable(
            f"Fewer than two segments have a '{trace_name}' trace in this file.",
            trace_name,
            bins,
            skipped=skipped,
        )

    stacked = np.column_stack([columns[segment] for segment in usable])
    rows_total = int(stacked.shape[0])
    # One mask for every segment at once, so each cell sees the same samples.
    finite = np.isfinite(stacked).all(axis=1)
    aligned = stacked[finite]
    rows_finite = int(aligned.shape[0])
    embedded = rows_finite - _max_context_lag(source_context, receiver_context)
    if embedded < MIN_EMBEDDED_SAMPLES:
        return _unavailable(
            f"Only {max(0, embedded)} samples remain after context trimming; at least "
            f"{MIN_EMBEDDED_SAMPLES} are required.",
            trace_name,
            bins,
            skipped=skipped,
        )

    # Once per segment, not twice per pair.
    discrete = [normalize_and_discretize(aligned[:, index], bins=bins) for index in range(len(usable))]

    size = len(usable)
    matrix = np.full((size, size), np.nan, dtype=float)
    n_samples_used = 0
    for source_index in range(size):
        for target_index in range(size):
            if source_index == target_index:
                continue
            samples = build_te_samples(
                discrete[source_index],
                discrete[target_index],
                source_context=source_context,
                receiver_context=receiver_context,
            )
            te_bits, used = estimate_transfer_entropy_from_samples(samples)
            matrix[source_index, target_index] = te_bits
            n_samples_used = used

    return {
        "available": True,
        "message": "",
        "segments": tuple(usable),
        "labels": tuple(catalog.pretty_signal_label(segment) for segment in usable),
        "matrix": [[_json_float(value) for value in row] for row in matrix],
        "trace_name": trace_name,
        "bins": bins,
        "source_context": {
            "lag": source_context.lag,
            "length": source_context.length,
            "stride": source_context.stride,
        },
        "receiver_context": {
            "lag": receiver_context.lag,
            "length": receiver_context.length,
            "stride": receiver_context.stride,
        },
        "rows_total": rows_total,
        "rows_finite": rows_finite,
        "n_samples_used": int(n_samples_used),
        "skipped": skipped,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "mode_file_path": str(path),
    }


def matrix_rows(matrix_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten one matrix result into tidy source/target rows for export."""
    if not matrix_result.get("available"):
        return []
    segments = list(matrix_result["segments"])
    labels = list(matrix_result["labels"])
    matrix = matrix_result["matrix"]
    rows: list[dict[str, Any]] = []
    for source_index, source_segment in enumerate(segments):
        for target_index, target_segment in enumerate(segments):
            rows.append(
                {
                    "source_segment": source_segment,
                    "source_label": labels[source_index],
                    "target_segment": target_segment,
                    "target_label": labels[target_index],
                    "te_bits": matrix[source_index][target_index],
                    "trace_name": matrix_result["trace_name"],
                    "bins": matrix_result["bins"],
                    "source_lag": matrix_result["source_context"]["lag"],
                    "source_length": matrix_result["source_context"]["length"],
                    "source_stride": matrix_result["source_context"]["stride"],
                    "receiver_lag": matrix_result["receiver_context"]["lag"],
                    "receiver_length": matrix_result["receiver_context"]["length"],
                    "receiver_stride": matrix_result["receiver_context"]["stride"],
                    "rows_finite": matrix_result["rows_finite"],
                    "n_samples_used": matrix_result["n_samples_used"],
                    "mode_file": matrix_result["mode_file_path"],
                }
            )
    return rows


def _max_context_lag(source_context: ContextSpec, receiver_context: ContextSpec) -> int:
    """Rows ``build_te_samples`` consumes before it can emit its first sample."""
    return max(
        context.lag + (context.length - 1) * context.stride
        for context in (source_context, receiver_context)
    )


def _segment_values(
    group_data: dict[str, Any], segment: str, trace_name: str
) -> np.ndarray | None:
    """One segment's trace out of a ``load_all_group_mode_data`` payload."""
    if trace_name not in (group_data.get("group_traces") or {}).get(segment, []):
        return None
    frame = group_data["data"]
    column = signals.trace_column_name(segment, trace_name)
    if column not in frame.columns:
        return None
    return frame[column].to_numpy(dtype=float).reshape(-1)


def _json_float(value: float) -> float | None:
    """NaN is not valid JSON, and the diagonal is deliberately NaN."""
    return None if value is None or math.isnan(float(value)) else float(value)


def _unavailable(
    message: str,
    trace_name: str,
    bins: int,
    *,
    skipped: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "available": False,
        "message": message,
        "segments": (),
        "labels": (),
        "matrix": [],
        "trace_name": trace_name,
        "bins": int(bins),
        "source_context": {"lag": 1, "length": 1, "stride": 1},
        "receiver_context": {"lag": 1, "length": 1, "stride": 1},
        "rows_total": 0,
        "rows_finite": 0,
        "n_samples_used": 0,
        "skipped": skipped or [],
        "elapsed_seconds": 0.0,
        "mode_file_path": "",
    }
