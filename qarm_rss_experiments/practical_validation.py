"""Retrospective, ground-truth-audited RSS validation for the QArm recordings.

The module deliberately separates blind fitting from encoder-based evaluation.
Only the distal Node 10 (claw-mounted) gyroscope enters :func:`prepare_rss_input`
and ``rss.fit_rss``.  Encoder and auxiliary-IMU columns are consumed later by the
quality, geometry, and evaluation stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from itertools import groupby
from pathlib import Path
import time
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

from bss_baselines import (
    BSSConfig,
    METHOD_FASTICA,
    METHOD_JADE_ICA,
    METHOD_PCA,
    METHOD_SOBI,
    fit_bss,
)
from rss.model import decode_scalar_and_vector, integrated_body_angles_to_quaternions
from rss.solver import RSSConfig, fit_rss
from rss.ssa_preprocessing import apply_ssa_preprocessing


DISTAL_PREFIX = "OrbiX-Node-10_IMU1"
DISTAL_GYRO_COLUMNS = tuple(f"{DISTAL_PREFIX}_G{axis}" for axis in "XYZ")
TRUTH_COLUMNS = ("measured_base_rad", "measured_elbow_rad", "measured_wrist_rad")
TARGET_COLUMNS = ("target_base_rad", "target_elbow_rad", "target_wrist_rad")
JOINT_NAMES = ("base", "elbow", "wrist")
INITIAL_MATRIX = (
    (-0.4779489885, -0.5421200170, -0.6911372162),
    (-0.5983844737, 0.7769586228, -0.1956305705),
    (0.6430402679, 0.3200643460, -0.6957427889),
)
FIGURE_FORMATS = ("png", "pdf", "svg")
VELOCITY_SAVGOL_WINDOW = 101
VELOCITY_SAVGOL_POLYORDER = 3


@dataclass(frozen=True)
class ValidationConfig:
    data_root: str
    output_root: str
    sample_rate_hz: float = 100.0
    expected_samples: int = 3000
    expected_duration_seconds: float = 30.0
    minimum_alignment_correlation: float = 0.85
    stationary_seconds: float = 2.0
    stationary_gyro_threshold_deg_s: float = 2.0
    gyro_saturation_deg_s: float = 2000.0
    rss_epochs: int = 2000
    pso_swarm_size: int = 200
    null_draws: int = 1000
    stability_seeds: tuple[int, ...] = (1, 2, 3, 4)
    bootstrap_resamples: int = 10000
    random_seed: int = 0
    include_sensitivities: bool = True
    trial_numbers: tuple[int, ...] | None = None
    audit_only: bool = False


@dataclass(frozen=True)
class AlignmentPlan:
    test_number: int
    batch_number: int
    raw_path: str
    ground_truth_path: str
    merged_path: str
    coarse_start: int
    aligned_start: int
    row_count: int
    alignment_correlation: float


@dataclass(frozen=True)
class PreparedTrial:
    test_number: int
    batch_number: int
    frame: pd.DataFrame
    observations: np.ndarray
    gyro_bias_deg_s: np.ndarray
    quality: dict[str, object]
    geometry: dict[str, object]


def _finite_2d(values: np.ndarray, name: str, rows: int = 3) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] != rows:
        raise ValueError(f"{name} must have shape ({rows}, T).")
    if array.shape[1] < 8 or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain at least eight finite samples.")
    return array


def prepare_rss_input(
    imu_frame: pd.DataFrame,
    gyro_bias_deg_s: Sequence[float],
    *,
    sample_rate_hz: float = 100.0,
) -> np.ndarray:
    """Return cumulative body-rotation channels from Node 10 and nothing else."""

    missing = [column for column in DISTAL_GYRO_COLUMNS if column not in imu_frame]
    if missing:
        raise ValueError(f"Missing distal gyroscope columns: {missing}.")
    gyro = imu_frame.loc[:, DISTAL_GYRO_COLUMNS].to_numpy(dtype=float)
    bias = np.asarray(gyro_bias_deg_s, dtype=float)
    if bias.shape != (3,) or not np.isfinite(bias).all():
        raise ValueError("gyro_bias_deg_s must contain three finite values.")
    if not math.isfinite(float(sample_rate_hz)) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive and finite.")
    if not np.isfinite(gyro).all():
        raise ValueError("The distal gyroscope contains non-finite samples.")
    increments = np.radians(gyro - bias[None, :]) / float(sample_rate_hz)
    return np.cumsum(increments, axis=0).T


def _centered_moving_average(values: np.ndarray, width: int) -> np.ndarray:
    size = max(1, int(width))
    if size % 2 == 0:
        size += 1
    if size == 1:
        return np.asarray(values, dtype=float)
    return pd.Series(np.asarray(values, dtype=float)).rolling(size, center=True, min_periods=1).mean().to_numpy()


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float) - float(np.mean(left))
    b = np.asarray(right, dtype=float) - float(np.mean(right))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator <= np.finfo(float).eps else float(np.dot(a, b) / denominator)


def _gyro_activity(frame: pd.DataFrame) -> np.ndarray:
    groups: list[np.ndarray] = []
    for prefix in ("OrbiX-Node-2_IMU1", "OrbiX-Node-6_IMU1", DISTAL_PREFIX, "OrbiX-Node-7_IMU1"):
        columns = [f"{prefix}_G{axis}" for axis in "XYZ"]
        if all(column in frame for column in columns):
            groups.append(np.linalg.norm(frame[columns].to_numpy(dtype=float), axis=1))
    if not groups:
        raise ValueError("No complete gyroscope triplets were found.")
    return np.max(np.stack(groups), axis=0)


def _movement_regions(activity: np.ndarray, sample_rate_hz: float) -> list[tuple[int, int]]:
    active = np.asarray(activity) >= 2.0
    raw: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(active.size + 1):
        state = index < active.size and bool(active[index])
        if state and start is None:
            start = index
        elif not state and start is not None:
            raw.append((start, index))
            start = None
    merged: list[tuple[int, int]] = []
    max_gap = round(1.5 * sample_rate_hz)
    for region in raw:
        if merged and region[0] - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], region[1])
        else:
            merged.append(region)
    return [region for region in merged if region[1] - region[0] >= round(10.0 * sample_rate_hz)]


def _measured_joint_speed(frame: pd.DataFrame) -> np.ndarray:
    columns = [column for column in frame if column.startswith("measured_") and column.endswith("_rad")]
    if not columns:
        raise ValueError("No measured joint-position columns were found.")
    positions = frame[columns].to_numpy(dtype=float)
    return np.r_[0.0, np.sum(np.abs(np.diff(positions, axis=0)), axis=1)]


def reproduce_alignment_plans(data_root: str | Path, sample_rate_hz: float = 100.0) -> list[AlignmentPlan]:
    """Reproduce the checked-in splitter's movement detection and alignment."""

    root = Path(data_root)
    truth_root = root / "Ground_Truth" / "results"
    merged_root = root / "Merged"
    plans: list[AlignmentPlan] = []
    for batch_dir in sorted((path for path in root.iterdir() if path.is_dir() and path.name.isdigit()), key=lambda p: int(p.name)):
        batch = int(batch_dir.name)
        raw_paths = list(batch_dir.glob("combined_log_wide_*.csv"))
        if len(raw_paths) != 1:
            raise ValueError(f"Expected one combined IMU file in {batch_dir}; found {len(raw_paths)}.")
        raw = pd.read_csv(raw_paths[0])
        activity = _gyro_activity(raw)
        regions = _movement_regions(activity, sample_rate_hz)
        if len(regions) != 5:
            raise ValueError(f"Expected five movement regions in batch {batch}; found {len(regions)}.")
        smooth_activity = _centered_moving_average(activity, round(0.21 * sample_rate_hz))
        for local_index, region in enumerate(regions, 1):
            number = (batch - 1) * 5 + local_index
            truth_path = truth_root / f"test_{number}.csv"
            merged_path = merged_root / f"test_{number}.csv"
            truth = pd.read_csv(truth_path)
            speed = _centered_moving_average(_measured_joint_speed(truth), round(0.21 * sample_rate_hz))
            first = max(0, region[0] - round(1.5 * sample_rate_hz))
            last = min(len(raw) - len(truth), region[0] + round(1.5 * sample_rate_hz))
            candidates = [(_pearson(smooth_activity[start : start + len(truth)], speed), start) for start in range(first, last + 1)]
            correlation, aligned = max(candidates)
            plans.append(
                AlignmentPlan(number, batch, str(raw_paths[0]), str(truth_path), str(merged_path), region[0], aligned, len(truth), correlation)
            )
    return plans


def estimate_pretrial_bias(
    raw_frame: pd.DataFrame,
    aligned_start: int,
    *,
    sample_rate_hz: float = 100.0,
    stationary_seconds: float = 2.0,
    threshold_deg_s: float = 2.0,
) -> tuple[np.ndarray, tuple[int, int]]:
    gyro = raw_frame.loc[:, DISTAL_GYRO_COLUMNS].to_numpy(dtype=float)
    magnitude = np.linalg.norm(gyro, axis=1)
    minimum = max(1, round(float(stationary_seconds) * float(sample_rate_hz)))
    mask = np.isfinite(magnitude) & (magnitude < float(threshold_deg_s))
    runs: list[tuple[int, int]] = []
    for state, indices in groupby(range(max(0, int(aligned_start))), key=lambda index: bool(mask[index])):
        selected = list(indices)
        if state and len(selected) >= minimum:
            runs.append((selected[0], selected[-1] + 1))
    if not runs:
        raise ValueError("No valid pretrial stationary interval was found.")
    start, end = runs[-1]
    # The nearest fixed-duration portion prevents long idle recordings from receiving extra weight.
    start = max(start, end - minimum)
    return np.median(gyro[start:end], axis=0), (start, end)


def _rotation_x(angles: np.ndarray) -> np.ndarray:
    c, s, z, o = np.cos(angles), np.sin(angles), np.zeros_like(angles), np.ones_like(angles)
    return np.stack((o, z, z, z, c, -s, z, s, c), axis=-1).reshape(-1, 3, 3)


def _rotation_y(angles: np.ndarray) -> np.ndarray:
    c, s, z, o = np.cos(angles), np.sin(angles), np.zeros_like(angles), np.ones_like(angles)
    return np.stack((c, z, s, z, o, z, -s, z, c), axis=-1).reshape(-1, 3, 3)


def _rotation_z(angles: np.ndarray) -> np.ndarray:
    c, s, z, o = np.cos(angles), np.sin(angles), np.zeros_like(angles), np.ones_like(angles)
    return np.stack((c, -s, z, s, c, z, z, z, o), axis=-1).reshape(-1, 3, 3)


def project_to_so3(matrices: np.ndarray) -> np.ndarray:
    values = np.asarray(matrices, dtype=float)
    single = values.ndim == 2
    batch = values[None, :, :] if single else values
    if batch.ndim != 3 or batch.shape[1:] != (3, 3):
        raise ValueError("matrices must have shape (3, 3) or (T, 3, 3).")
    u, _, vt = np.linalg.svd(batch)
    result = u @ vt
    negative = np.linalg.det(result) < 0.0
    if np.any(negative):
        u = u.copy()
        u[negative, :, -1] *= -1.0
        result[negative] = u[negative] @ vt[negative]
    return result[0] if single else result


def qarm_source_axes_distal(frame: pd.DataFrame) -> np.ndarray:
    """Return sign-consistent base/elbow/wrist unit axes in the distal frame."""

    required = ("measured_base_rad", "measured_shoulder_rad", "measured_elbow_rad", "measured_wrist_rad")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing QArm kinematic columns: {missing}.")
    base, shoulder, elbow, wrist = (frame[column].to_numpy(dtype=float) for column in required)
    rz = _rotation_z(base)
    pitch = _rotation_y(shoulder + elbow)
    end_effector = rz @ pitch @ _rotation_x(wrist)
    count = len(frame)
    base_axis = np.tile((0.0, 0.0, 1.0), (count, 1))
    elbow_axis = rz @ np.array((0.0, 1.0, 0.0))
    # Axis signs are physically ambiguous.  This fixed sign makes the home triad proper.
    wrist_axis = -(rz @ pitch @ np.array((1.0, 0.0, 0.0)))
    world_axes = np.stack((base_axis, elbow_axis, wrist_axis), axis=2)
    return np.transpose(end_effector, (0, 2, 1)) @ world_axes


def geometry_metrics(frame: pd.DataFrame) -> dict[str, float]:
    axes = qarm_source_axes_distal(frame)
    gram = np.transpose(axes, (0, 2, 1)) @ axes
    defects = np.linalg.norm(gram - np.eye(3), axis=(1, 2))
    instantaneous = project_to_so3(axes)
    constant = project_to_so3(np.mean(instantaneous, axis=0))
    relative = np.transpose(constant) @ instantaneous
    traces = np.trace(relative, axis1=1, axis2=2)
    geodesic = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
    instantaneous_residual = np.linalg.norm(axes - instantaneous, axis=(1, 2))
    total_residual = np.linalg.norm(axes - constant, axis=(1, 2))
    result = {
        "orthogonality_defect_rms": float(np.sqrt(np.mean(defects**2))),
        "orthogonality_defect_median": float(np.median(defects)),
        "orthogonality_defect_p95": float(np.quantile(defects, 0.95)),
        "orthogonality_defect_max": float(np.max(defects)),
        "instantaneous_nonorthogonality_rms": float(np.sqrt(np.mean(instantaneous_residual**2))),
        "frame_variation_rms_rad": float(np.sqrt(np.mean(geodesic**2))),
        "frame_variation_rms_deg": float(np.degrees(np.sqrt(np.mean(geodesic**2)))),
        "total_constant_model_mismatch": float(np.sqrt(np.mean(total_residual**2))),
    }
    for row in range(3):
        for column in range(3):
            result[f"constant_rotation_{row + 1}{column + 1}"] = float(constant[row, column])
    return result


def source_correlation_metrics(frame: pd.DataFrame, sample_rate_hz: float = 100.0) -> dict[str, float]:
    positions = frame.loc[:, TRUTH_COLUMNS].to_numpy(dtype=float).T
    position_corr = np.corrcoef(positions)
    window = min(VELOCITY_SAVGOL_WINDOW, positions.shape[1] - (1 - positions.shape[1] % 2))
    window = max(5, window if window % 2 else window - 1)
    velocities = savgol_filter(positions, window_length=window, polyorder=VELOCITY_SAVGOL_POLYORDER, deriv=1, delta=1.0 / sample_rate_hz, axis=1)
    velocity_corr = np.corrcoef(velocities)
    upper = np.triu_indices(3, 1)
    ctheta = float(np.max(np.abs(position_corr[upper])))
    cvelocity = float(np.max(np.abs(velocity_corr[upper])))
    if ctheta < 0.2:
        stratum = "lower"
    elif ctheta < 0.5:
        stratum = "moderate"
    else:
        stratum = "high"
    return {"cmax_position": ctheta, "cmax_velocity": cvelocity, "correlation_stratum": stratum}


def match_and_score(
    truth: np.ndarray,
    estimated: np.ndarray,
    *,
    test_number: int,
    batch_number: int,
    method: str,
) -> tuple[list[dict[str, object]], np.ndarray]:
    true_values = _finite_2d(truth, "truth")
    estimates = _finite_2d(estimated, "estimated")
    if true_values.shape != estimates.shape:
        raise ValueError("truth and estimated sources must have equal shapes.")
    correlations = np.asarray([[_pearson(true_values[i], estimates[j]) for j in range(3)] for i in range(3)])
    true_indices, mode_indices = linear_sum_assignment(-np.abs(correlations))
    assignment = np.empty(3, dtype=int)
    assignment[true_indices] = mode_indices
    aligned = np.empty_like(true_values)
    rows: list[dict[str, object]] = []
    for joint_index, joint in enumerate(JOINT_NAMES):
        mode = int(assignment[joint_index])
        raw_correlation = float(correlations[joint_index, mode])
        sign = 1 if raw_correlation >= 0.0 else -1
        signed = sign * estimates[mode]
        offset = float(np.mean(true_values[joint_index] - signed))
        aligned[joint_index] = signed + offset
        error = aligned[joint_index] - true_values[joint_index]
        rmse = float(np.sqrt(np.mean(error**2)))
        mae = float(np.mean(np.abs(error)))
        standard_deviation = float(np.std(true_values[joint_index]))
        rows.append(
            {
                "test_number": test_number,
                "batch_number": batch_number,
                "method": method,
                "joint": joint,
                "mode_number": mode + 1,
                "raw_correlation": raw_correlation,
                "abs_correlation": abs(raw_correlation),
                "applied_sign": sign,
                "applied_offset_rad": offset,
                "amplitude_scale": 1.0,
                "rmse_rad": rmse,
                "rmse_deg": float(np.degrees(rmse)),
                "mae_rad": mae,
                "mae_deg": float(np.degrees(mae)),
                "nrmse_std": rmse / max(standard_deviation, np.finfo(float).eps),
                "measured_std_rad": standard_deviation,
                "measured_variance_rad2": float(np.var(true_values[joint_index])),
                "measured_range_rad": float(np.ptp(true_values[joint_index])),
            }
        )
    return rows, aligned


def aggregate_trial_rows(source_rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (test_number, batch_number, method), group in source_rows.groupby(["test_number", "batch_number", "method"], sort=True):
        values = np.clip(group["abs_correlation"].to_numpy(dtype=float), 0.0, 1.0 - 1e-12)
        records.append(
            {
                "test_number": int(test_number),
                "batch_number": int(batch_number),
                "method": method,
                "mean_abs_correlation_fisher": float(np.tanh(np.mean(np.arctanh(values)))),
                "mean_nrmse_std": float(group["nrmse_std"].mean()),
                "mean_rmse_rad": float(group["rmse_rad"].mean()),
                "mean_mae_rad": float(group["mae_rad"].mean()),
            }
        )
    return pd.DataFrame.from_records(records)


def haar_so3(rng: np.random.Generator, count: int) -> np.ndarray:
    gaussian = rng.normal(size=(int(count), 3, 3))
    q, r = np.linalg.qr(gaussian)
    signs = np.where(np.diagonal(r, axis1=1, axis2=2) < 0.0, -1.0, 1.0)
    q = q * signs[:, None, :]
    negative = np.linalg.det(q) < 0.0
    q[negative, :, -1] *= -1.0
    return q


def sources_from_rotation(observations: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    cumulative = _finite_2d(observations, "observations")
    observer_q = integrated_body_angles_to_quaternions(cumulative)
    sources, _ = decode_scalar_and_vector(observer_q[:, 0], np.asarray(matrix, dtype=float) @ observer_q[:, 1:].T)
    return sources


def frozen_rss_config(config: ValidationConfig, seed: int = 0) -> RSSConfig:
    return RSSConfig(
        loss_type="lagged_cross_correlation", separation_weight=1.0, lags=(0,),
        epochs=config.rss_epochs, optimizer="pso", learning_rate=1e-1,
        learning_rate_end=1e-3, pso_swarm_size=config.pso_swarm_size,
        pso_inertia=0.9, pso_cognitive=1.0, pso_social=1.5, pso_seed=seed,
        orthogonality_weight=0.0, reconstruction_weight=0.0, convergence_tol=0.0,
        eps=1e-24, quaternion_eps=1e-24, matrix_parameterization="so3",
        observer_signal_model="integrated_body_rotation", initial_weight_matrix=INITIAL_MATRIX,
    )


def legacy_rss_config(config: ValidationConfig) -> RSSConfig:
    return RSSConfig(
        loss_type="lagged_cross_correlation", separation_weight=1.0, lags=(0,),
        epochs=config.rss_epochs, optimizer="sgd", learning_rate=1e-1,
        learning_rate_end=1e-3, sgd_momentum=0.9, orthogonality_weight=0.0,
        reconstruction_weight=0.0, convergence_tol=0.0, eps=1e-24,
        quaternion_eps=1e-24, matrix_parameterization="so3", observer_signal_model="euler",
        initial_weight_matrix=INITIAL_MATRIX, adaptive_regularizer=True,
        specialized_regularizer="harmonic_spectral_overlap", regularizer_start_weight=0.1,
        regularizer_plateau_patience=50, regularizer_plateau_tolerance=1e-4,
        regularizer_ramp_epochs=25, regularizer_min_loss_threshold=0.0,
    )


def ssa_observations(observations: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    values = _finite_2d(observations, "observations")
    return np.stack(
        [
            apply_ssa_preprocessing(
                signal, sample_rate_hz=sample_rate_hz, requested_window_length=500,
                requested_reconstruction_components=25, drift_zero_crossing_limit=3,
                drift_turning_point_limit=3,
            ).cleaned_signal
            for signal in values
        ]
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_trial(plan: AlignmentPlan, config: ValidationConfig) -> PreparedTrial:
    raw = pd.read_csv(plan.raw_path)
    # round_trip avoids a bit-exactness mismatch against plot_practical_results.py's
    # independent re-read of this same file when it verifies the cached fit's input hash.
    frame = pd.read_csv(plan.merged_path, float_precision="round_trip")
    bias, stationary = estimate_pretrial_bias(
        raw, plan.aligned_start, sample_rate_hz=config.sample_rate_hz,
        stationary_seconds=config.stationary_seconds,
        threshold_deg_s=config.stationary_gyro_threshold_deg_s,
    )
    observations = prepare_rss_input(frame, bias, sample_rate_hz=config.sample_rate_hz)
    time_values = frame["time_s"].to_numpy(dtype=float)
    missing_count = int(frame.isna().sum().sum())
    gyro = frame.loc[:, DISTAL_GYRO_COLUMNS].to_numpy(dtype=float)
    saturation_count = int(np.count_nonzero(np.abs(gyro) >= config.gyro_saturation_deg_s))
    duration = float(time_values[-1] - time_values[0])
    sample_interval = float(np.median(np.diff(time_values)))
    quality_pass = (
        len(frame) == config.expected_samples
        and abs(duration - config.expected_duration_seconds) <= 0.05
        and plan.alignment_correlation >= config.minimum_alignment_correlation
        and missing_count == 0
        and saturation_count == 0
    )
    reason: list[str] = []
    if len(frame) != config.expected_samples or abs(duration - config.expected_duration_seconds) > 0.05:
        reason.append("incomplete_recording")
    if plan.alignment_correlation < config.minimum_alignment_correlation:
        reason.append("low_alignment_correlation")
    if missing_count:
        reason.append("missing_values")
    if saturation_count:
        reason.append("gyro_saturation")
    correlation = source_correlation_metrics(frame, config.sample_rate_hz)
    target = frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=float)
    truth = frame.loc[:, TRUTH_COLUMNS].to_numpy(dtype=float)
    tracking_rmse = np.sqrt(np.mean((target - truth) ** 2, axis=0))
    quality: dict[str, object] = {
        "test_number": plan.test_number, "batch_number": plan.batch_number,
        "raw_path": plan.raw_path, "ground_truth_path": plan.ground_truth_path,
        "merged_path": plan.merged_path, "raw_sha256": _sha256(plan.raw_path),
        "ground_truth_sha256": _sha256(plan.ground_truth_path), "merged_sha256": _sha256(plan.merged_path),
        "coarse_start": plan.coarse_start, "aligned_start": plan.aligned_start,
        "alignment_shift_samples": plan.aligned_start - plan.coarse_start,
        "alignment_correlation": plan.alignment_correlation, "row_count": len(frame),
        "duration_seconds": duration, "median_sample_interval_seconds": sample_interval,
        "effective_sample_rate_hz": 1.0 / sample_interval, "missing_value_count": missing_count,
        "distal_gyro_saturation_count": saturation_count, "stationary_start": stationary[0],
        "stationary_end": stationary[1], "stationary_rows": stationary[1] - stationary[0],
        "gyro_bias_x_deg_s": bias[0], "gyro_bias_y_deg_s": bias[1], "gyro_bias_z_deg_s": bias[2],
        "available_imu_prefixes": ";".join(sorted({column.rsplit("_", 1)[0] for column in frame if column.endswith(("_GX", "_GY", "_GZ"))})),
        "quality_pass": quality_pass, "exclusion_reasons": ";".join(reason),
        "ideal_low_correlation": correlation["cmax_position"] < 0.1,
        **correlation,
    }
    for index, joint in enumerate(JOINT_NAMES):
        quality[f"tracking_rmse_{joint}_rad"] = float(tracking_rmse[index])
        quality[f"tracking_nrmse_{joint}"] = float(tracking_rmse[index] / max(np.std(truth[:, index]), np.finfo(float).eps))
    quality["mean_tracking_nrmse"] = float(np.mean([quality[f"tracking_nrmse_{joint}"] for joint in JOINT_NAMES]))
    return PreparedTrial(plan.test_number, plan.batch_number, frame, observations, bias, quality, {"test_number": plan.test_number, "batch_number": plan.batch_number, **geometry_metrics(frame)})


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _fit_cache_path(output: Path, test_number: int, method: str) -> Path:
    return output / "fits" / f"test_{test_number}" / f"{method}.npz"


def _cached_fit(output: Path, trial: PreparedTrial, method: str, fit) -> tuple[np.ndarray, float, dict[str, object]]:
    path = _fit_cache_path(output, trial.test_number, method)
    input_hash = hashlib.sha256(np.ascontiguousarray(trial.observations).tobytes()).hexdigest()
    if path.is_file():
        cached = np.load(path, allow_pickle=False)
        if str(cached.get("input_hash", "")) == input_hash:
            return cached["sources"], float(cached["runtime_seconds"]), json.loads(str(cached["metadata_json"]))
    started = time.perf_counter()
    sources, metadata = fit()
    runtime = time.perf_counter() - started
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, sources=sources, runtime_seconds=runtime, metadata_json=json.dumps(_json_safe(metadata)), input_hash=input_hash)
    return sources, runtime, metadata


def _fit_methods(trial: PreparedTrial, config: ValidationConfig, output: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    truth = trial.frame.loc[:, TRUTH_COLUMNS].to_numpy(dtype=float).T
    all_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    stability_matrices: dict[int, np.ndarray] = {}
    aligned_signals: dict[str, np.ndarray] = {}

    def add(method: str, sources: np.ndarray, runtime: float, metadata: dict[str, object], *, stability_seed: int | None = None) -> None:
        rows, aligned = match_and_score(truth, sources, test_number=trial.test_number, batch_number=trial.batch_number, method=method)
        for row in rows:
            row.update({"runtime_seconds": runtime, "fit_metadata": json.dumps(_json_safe(metadata), sort_keys=True)})
        all_rows.extend(rows)
        aligned_signals[method] = aligned
        if stability_seed is not None:
            summary = aggregate_trial_rows(pd.DataFrame(rows)).iloc[0].to_dict()
            stability_rows.append({**summary, "pso_seed": stability_seed})
            if "recovery_matrix" in metadata:
                recovery = np.asarray(metadata["recovery_matrix"], dtype=float)
                matched = np.stack(
                    [
                        int(row["applied_sign"]) * recovery[int(row["mode_number"]) - 1]
                        for row in rows
                    ]
                )
                stability_matrices[stability_seed] = project_to_so3(matched)

    identity, runtime, metadata = _cached_fit(output, trial, "identity", lambda: (sources_from_rotation(trial.observations, np.eye(3)), {"matrix": np.eye(3).tolist()}))
    add("identity", identity, runtime, metadata)
    bss_methods = (METHOD_PCA, METHOD_JADE_ICA, METHOD_FASTICA, METHOD_SOBI)
    bss_config = BSSConfig(methods=bss_methods, random_state=0)
    for method in bss_methods:
        sources, runtime, metadata = _cached_fit(
            output, trial, method,
            lambda method=method: (
                (result := fit_bss(trial.observations, method, bss_config)).source_angles,
                {"converged": result.converged, "iterations": result.iterations, "warnings": result.warnings},
            ),
        )
        add(method, sources, runtime, metadata)
    sources, runtime, metadata = _cached_fit(
        output, trial, "rss",
        lambda: (
            (result := fit_rss(trial.observations, frozen_rss_config(config), sample_interval_seconds=1.0 / config.sample_rate_hz)).source_angles,
            {"best_loss": result.best_loss, "best_epoch": result.best_epoch, "recovery_matrix": result.recovery_matrix.tolist()},
        ),
    )
    add("rss", sources, runtime, metadata, stability_seed=0)
    for seed in config.stability_seeds:
        method = f"rss_seed_{seed}"
        sources, runtime, metadata = _cached_fit(
            output, trial, method,
            lambda seed=seed: (
                (result := fit_rss(trial.observations, frozen_rss_config(config, seed), sample_interval_seconds=1.0 / config.sample_rate_hz)).source_angles,
                {"best_loss": result.best_loss, "best_epoch": result.best_epoch, "recovery_matrix": result.recovery_matrix.tolist()},
            ),
        )
        add(method, sources, runtime, metadata, stability_seed=seed)
    if config.include_sensitivities:
        observations = ssa_observations(trial.observations, config.sample_rate_hz)
        sources, runtime, metadata = _cached_fit(
            output, trial, "legacy_euler_ssa_sgd",
            lambda: (
                (result := fit_rss(observations, legacy_rss_config(config), sample_interval_seconds=1.0 / config.sample_rate_hz)).source_angles,
                {"best_loss": result.best_loss, "best_epoch": result.best_epoch, "recovery_matrix": result.recovery_matrix.tolist()},
            ),
        )
        add("legacy_euler_ssa_sgd", sources, runtime, metadata)
    if 0 in stability_matrices:
        reference = stability_matrices[0]
        for row in stability_rows:
            matrix = stability_matrices.get(int(row["pso_seed"]))
            if matrix is None:
                row["recovery_matrix_geodesic_from_seed0_deg"] = float("nan")
                continue
            relative = reference.T @ matrix
            angle = math.acos(float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
            row["recovery_matrix_geodesic_from_seed0_deg"] = math.degrees(angle)
    return all_rows, stability_rows, aligned_signals


def _random_null(trial: PreparedTrial, config: ValidationConfig) -> list[dict[str, object]]:
    truth = trial.frame.loc[:, TRUTH_COLUMNS].to_numpy(dtype=float).T
    observer_q = integrated_body_angles_to_quaternions(trial.observations)
    scalar, vector = observer_q[:, 0], observer_q[:, 1:].T
    rng = np.random.default_rng(config.random_seed + 100003 * trial.test_number)
    rows: list[dict[str, object]] = []
    for draw, matrix in enumerate(haar_so3(rng, config.null_draws)):
        sources, _ = decode_scalar_and_vector(scalar, matrix @ vector)
        scored, _ = match_and_score(truth, sources, test_number=trial.test_number, batch_number=trial.batch_number, method="random_so3")
        summary = aggregate_trial_rows(pd.DataFrame(scored)).iloc[0]
        for row in scored:
            rows.append({"test_number": trial.test_number, "batch_number": trial.batch_number, "draw": draw, "level": "source", "joint": row["joint"], "abs_correlation": row["abs_correlation"], "nrmse_std": row["nrmse_std"]})
        rows.append({"test_number": trial.test_number, "batch_number": trial.batch_number, "draw": draw, "level": "trial", "joint": "all", "abs_correlation": summary["mean_abs_correlation_fisher"], "nrmse_std": summary["mean_nrmse_std"]})
    return rows


def _cluster_bootstrap(values: pd.DataFrame, column: str, *, resamples: int, seed: int, statistic: str = "mean") -> tuple[float, float, float]:
    clean = values[["batch_number", column]].dropna()
    batches = np.asarray(sorted(clean["batch_number"].unique()), dtype=int)
    if batches.size == 0:
        return float("nan"), float("nan"), float("nan")
    reducer = np.mean if statistic == "mean" else np.median
    point = float(reducer(clean[column]))
    rng = np.random.default_rng(seed)
    estimates = []
    grouped = {batch: clean.loc[clean["batch_number"] == batch, column].to_numpy(dtype=float) for batch in batches}
    for _ in range(resamples):
        selected = rng.choice(batches, size=len(batches), replace=True)
        estimates.append(float(reducer(np.concatenate([grouped[batch] for batch in selected]))))
    return point, float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _summary_statistics(source: pd.DataFrame, config: ValidationConfig) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (method, joint), group in source.groupby(["method", "joint"], sort=True):
        for metric in ("abs_correlation", "rmse_rad", "mae_rad", "nrmse_std"):
            values = group[metric].to_numpy(dtype=float)
            for statistic in ("mean", "median"):
                point, lower, upper = _cluster_bootstrap(group, metric, resamples=config.bootstrap_resamples, seed=config.random_seed, statistic=statistic)
                records.append({"method": method, "joint": joint, "metric": metric, "statistic": statistic, "value": point, "ci95_lower": lower, "ci95_upper": upper, "count": len(values), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "q25": float(np.quantile(values, 0.25)), "q75": float(np.quantile(values, 0.75))})
    return pd.DataFrame(records)


def _trial_summary_statistics(trials: pd.DataFrame, config: ValidationConfig) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for method, group in trials.groupby("method", sort=True):
        for metric in ("mean_abs_correlation_fisher", "mean_nrmse_std", "mean_rmse_rad", "mean_mae_rad"):
            values = group[metric].to_numpy(dtype=float)
            for statistic in ("mean", "median"):
                point, lower, upper = _cluster_bootstrap(group, metric, resamples=config.bootstrap_resamples, seed=config.random_seed, statistic=statistic)
                records.append({"method": method, "joint": "all", "metric": metric, "statistic": statistic, "value": point, "ci95_lower": lower, "ci95_upper": upper, "count": len(values), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "q25": float(np.quantile(values, 0.25)), "q75": float(np.quantile(values, 0.75))})
    return pd.DataFrame(records)


def _robustness_models(trials: pd.DataFrame, quality: pd.DataFrame, geometry: pd.DataFrame, config: ValidationConfig) -> pd.DataFrame:
    frame = trials.loc[trials.method == "rss"].merge(quality, on=["test_number", "batch_number"]).merge(geometry, on=["test_number", "batch_number"])
    predictors = ["total_constant_model_mismatch", "cmax_position", "mean_tracking_nrmse"]
    if len(frame) < len(predictors) + 2:
        return pd.DataFrame(
            [
                {
                    "outcome": outcome,
                    "term": "model",
                    "estimate": float("nan"),
                    "hc3_standard_error": float("nan"),
                    "hc3_ci95_lower": float("nan"),
                    "hc3_ci95_upper": float("nan"),
                    "batch_bootstrap_ci95_lower": float("nan"),
                    "batch_bootstrap_ci95_upper": float("nan"),
                    "interpretation": f"insufficient_data_n_{len(frame)}",
                }
                for outcome in ("mean_abs_correlation_fisher", "mean_nrmse_std")
            ]
        )
    xraw = frame[predictors].to_numpy(dtype=float)
    means, scales = xraw.mean(axis=0), xraw.std(axis=0, ddof=1)
    x = np.column_stack((np.ones(len(frame)), (xraw - means) / np.maximum(scales, np.finfo(float).eps)))
    inverse = np.linalg.pinv(x.T @ x)
    records: list[dict[str, object]] = []
    for outcome in ("mean_abs_correlation_fisher", "mean_nrmse_std"):
        y = frame[outcome].to_numpy(dtype=float)
        beta = inverse @ x.T @ y
        residual = y - x @ beta
        leverage = np.sum((x @ inverse) * x, axis=1)
        adjusted = residual / np.maximum(1.0 - leverage, 1e-12)
        covariance = inverse @ (x.T @ ((adjusted**2)[:, None] * x)) @ inverse
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        rng = np.random.default_rng(config.random_seed)
        batches = np.asarray(sorted(frame.batch_number.unique()))
        boot = []
        for _ in range(config.bootstrap_resamples):
            selected = rng.choice(batches, len(batches), replace=True)
            sampled = pd.concat([frame.loc[frame.batch_number == batch] for batch in selected], ignore_index=True)
            bxraw = sampled[predictors].to_numpy(dtype=float)
            bx = np.column_stack((np.ones(len(sampled)), (bxraw - means) / np.maximum(scales, np.finfo(float).eps)))
            boot.append(np.linalg.pinv(bx.T @ bx) @ bx.T @ sampled[outcome].to_numpy(dtype=float))
        boot_array = np.asarray(boot)
        for index, name in enumerate(("intercept", *predictors)):
            records.append({"outcome": outcome, "term": name, "estimate": beta[index], "hc3_standard_error": standard_errors[index], "hc3_ci95_lower": beta[index] - 1.96 * standard_errors[index], "hc3_ci95_upper": beta[index] + 1.96 * standard_errors[index], "batch_bootstrap_ci95_lower": float(np.quantile(boot_array[:, index], 0.025)), "batch_bootstrap_ci95_upper": float(np.quantile(boot_array[:, index], 0.975)), "interpretation": "exploratory_association_not_causal"})
    return pd.DataFrame(records)


def _paired_comparisons(source: pd.DataFrame, config: ValidationConfig) -> pd.DataFrame:
    rss = source.loc[source.method == "rss"]
    records: list[dict[str, object]] = []
    for method in ("identity", METHOD_PCA, METHOD_JADE_ICA, METHOD_FASTICA, METHOD_SOBI):
        comparison = source.loc[source.method == method]
        merged = rss.merge(comparison, on=["test_number", "batch_number", "joint"], suffixes=("_rss", "_comparator"))
        for joint, group in merged.groupby("joint", sort=True):
            for metric, direction in (("abs_correlation", 1.0), ("nrmse_std", -1.0)):
                differences = direction * (group[f"{metric}_rss"] - group[f"{metric}_comparator"])
                values = pd.DataFrame({"batch_number": group.batch_number, "improvement": differences})
                for statistic in ("mean", "median"):
                    point, lower, upper = _cluster_bootstrap(values, "improvement", resamples=config.bootstrap_resamples, seed=config.random_seed, statistic=statistic)
                    records.append({"comparator": method, "joint": joint, "metric": metric, "statistic": statistic, "rss_improvement": point, "ci95_lower": lower, "ci95_upper": upper, "positive_favors_rss": True, "trial_count": len(group)})
    return pd.DataFrame(records)


def _null_percentiles(source: pd.DataFrame, trial_metrics: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    observed_sources = source.loc[source.method == "rss"]
    null_sources = null.loc[null.level == "source"]
    for row in observed_sources.itertuples():
        group = null_sources.loc[(null_sources.test_number == row.test_number) & (null_sources.joint == row.joint)]
        records.append({"test_number": row.test_number, "batch_number": row.batch_number, "level": "source", "joint": row.joint, "correlation_percentile": 100.0 * float(np.mean(group.abs_correlation < row.abs_correlation)), "nrmse_superiority_percentile": 100.0 * float(np.mean(group.nrmse_std > row.nrmse_std))})
    observed_trials = trial_metrics.loc[trial_metrics.method == "rss"]
    null_trials = null.loc[null.level == "trial"]
    for row in observed_trials.itertuples():
        group = null_trials.loc[null_trials.test_number == row.test_number]
        records.append({"test_number": row.test_number, "batch_number": row.batch_number, "level": "trial", "joint": "all", "correlation_percentile": 100.0 * float(np.mean(group.abs_correlation < row.mean_abs_correlation_fisher)), "nrmse_superiority_percentile": 100.0 * float(np.mean(group.nrmse_std > row.mean_nrmse_std))})
    return pd.DataFrame(records)


def _evidence_table(paired: pd.DataFrame, percentiles: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for joint in JOINT_NAMES:
        corr = paired.loc[(paired.comparator == "identity") & (paired.joint == joint) & (paired.metric == "abs_correlation") & (paired.statistic == "mean")].iloc[0]
        nrmse = paired.loc[(paired.comparator == "identity") & (paired.joint == joint) & (paired.metric == "nrmse_std") & (paired.statistic == "mean")].iloc[0]
        null = percentiles.loc[(percentiles.level == "source") & (percentiles.joint == joint)]
        corr_percentile = float(null.correlation_percentile.median())
        nrmse_percentile = float(null.nrmse_superiority_percentile.median())
        passed = bool(corr.ci95_lower > 0.0 and nrmse.ci95_lower > 0.0 and corr_percentile >= 95.0 and nrmse_percentile >= 95.0)
        records.append({"joint": joint, "correlation_improvement_ci95_lower": corr.ci95_lower, "nrmse_improvement_ci95_lower": nrmse.ci95_lower, "median_correlation_null_percentile": corr_percentile, "median_nrmse_superiority_percentile": nrmse_percentile, "practical_separation_evidence": passed})
    return pd.DataFrame(records)


def _save_figure(figure: plt.Figure, root: Path, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for extension in FIGURE_FORMATS:
        figure.savefig(root / f"{name}.{extension}", dpi=200 if extension == "png" else None, bbox_inches="tight")
    plt.close(figure)


def _make_figures(output: Path, trials: list[PreparedTrial], source: pd.DataFrame, trial_metrics: pd.DataFrame, quality: pd.DataFrame, geometry: pd.DataFrame, null_percentiles: pd.DataFrame, stability: pd.DataFrame, aligned_by_test: dict[int, dict[str, np.ndarray]], excluded_aligned: dict[int, dict[str, np.ndarray]]) -> None:
    figure_root = output / "figures"
    eligible_trials = trial_metrics.loc[trial_metrics.method == "rss"].sort_values("mean_nrmse_std")
    representative = int(eligible_trials.iloc[len(eligible_trials) // 2].test_number)
    trial = next(item for item in trials if item.test_number == representative)
    truth = trial.frame.loc[:, TRUTH_COLUMNS].to_numpy(dtype=float).T
    estimate = aligned_by_test[representative]["rss"]
    times = np.arange(truth.shape[1]) / 100.0
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for index, (axis, joint) in enumerate(zip(axes, JOINT_NAMES)):
        axis.plot(times, truth[index], color="black", label="Measured", lw=1.4)
        axis.plot(times, estimate[index], color="#0072B2", label="RSS", lw=1.1)
        axis.set_ylabel(f"{joint.title()} (rad)"); axis.grid(alpha=0.25); axis.legend(frameon=False)
    axes[-1].set_xlabel("Time (s)"); fig.suptitle(f"Representative eligible trial {representative}")
    _save_figure(fig, figure_root, "representative_reconstruction")

    comparisons = source.loc[
        source.method.isin(
            ["rss", "identity", METHOD_PCA, METHOD_JADE_ICA, METHOD_FASTICA, METHOD_SOBI]
        )
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    method_order = ("identity", METHOD_PCA, METHOD_JADE_ICA, METHOD_FASTICA, METHOD_SOBI, "rss")
    colors = ("#999999", "#56B4E9", "#CC79A7", "#E69F00", "#009E73", "#0072B2")
    group_stride = len(method_order) + 1
    for position, (method, color) in enumerate(zip(method_order, colors)):
        group = comparisons.loc[comparisons.method == method]
        for axis, metric in zip(axes, ("abs_correlation", "nrmse_std")):
            artists = axis.boxplot(
                [group.loc[group.joint == joint, metric] for joint in JOINT_NAMES],
                positions=np.arange(3) * group_stride + position,
                widths=0.7,
                patch_artist=True,
            )
            for box in artists["boxes"]:
                box.set_facecolor(color)
                box.set_alpha(0.8)
    for axis, title in zip(axes, ("Absolute correlation", "Normalized RMSE")):
        axis.set_xticks(
            np.arange(3) * group_stride + (len(method_order) - 1) / 2.0,
            [name.title() for name in JOINT_NAMES],
        ); axis.set_title(title); axis.grid(alpha=0.2)
        axis.legend(
            [Patch(facecolor=color, alpha=0.8) for color in colors],
            ["Identity", "PCA", "ICA (JADE)", "ICA (FastICA)", "SOBI", "RSS"],
            frameon=False,
            fontsize=7,
        )
    _save_figure(fig, figure_root, "jointwise_comparator_distributions")

    merged = trial_metrics.loc[trial_metrics.method == "rss"].merge(quality, on=["test_number", "batch_number"]).merge(geometry, on=["test_number", "batch_number"])
    for x, color, name, xlabel in (
        ("total_constant_model_mismatch", "cmax_position", "performance_vs_geometry", "Total constant-model mismatch"),
        ("cmax_position", "total_constant_model_mismatch", "performance_vs_source_correlation", "Measured-source Cmax"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for axis, y, ylabel in zip(axes, ("mean_abs_correlation_fisher", "mean_nrmse_std"), ("Mean |correlation|", "Mean normalized RMSE")):
            points = axis.scatter(merged[x], merged[y], c=merged[color], cmap="viridis", edgecolor="black", linewidth=0.3)
            axis.set_xlabel(xlabel); axis.set_ylabel(ylabel); axis.grid(alpha=0.2); fig.colorbar(points, ax=axis, label=color)
        _save_figure(fig, figure_root, name)

    percentiles = null_percentiles.loc[null_percentiles.level == "trial"]
    fig, axis = plt.subplots(figsize=(10, 4.5)); axis.scatter(percentiles.test_number, percentiles.correlation_percentile, label="Correlation", marker="o"); axis.scatter(percentiles.test_number, percentiles.nrmse_superiority_percentile, label="NRMSE superiority", marker="x"); axis.axhline(95, color="red", ls="--"); axis.set(xlabel="Test", ylabel="Random-SO(3) percentile", ylim=(0, 101)); axis.legend(frameon=False); axis.grid(alpha=0.2)
    _save_figure(fig, figure_root, "random_rotation_null_percentiles")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5)); axes[0].scatter(quality.test_number, quality.mean_tracking_nrmse, c=quality.quality_pass.map({True: "#0072B2", False: "#D55E00"})); axes[0].set(xlabel="Test", ylabel="Mean tracking NRMSE"); axes[1].scatter(quality.alignment_correlation, quality.row_count, c=quality.quality_pass.map({True: "#0072B2", False: "#D55E00"})); axes[1].set(xlabel="Alignment correlation", ylabel="Rows"); [axis.grid(alpha=0.2) for axis in axes]
    _save_figure(fig, figure_root, "tracking_and_quality_diagnostics")
    if not stability.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5)); stability.boxplot(column="mean_abs_correlation_fisher", by="pso_seed", ax=axes[0]); stability.boxplot(column="mean_nrmse_std", by="pso_seed", ax=axes[1]); fig.suptitle(""); axes[0].set_title("Correlation stability"); axes[1].set_title("NRMSE stability")
        _save_figure(fig, figure_root, "pso_seed_stability")
    if excluded_aligned:
        number = sorted(excluded_aligned)[0]
        trial = next(item for item in trials if item.test_number == number)
        truth = trial.frame.loc[:, TRUTH_COLUMNS].to_numpy(dtype=float).T
        estimate = excluded_aligned[number]["rss"]
        times = np.arange(truth.shape[1]) / 100.0
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for index, (axis, joint) in enumerate(zip(axes, JOINT_NAMES)):
            axis.plot(times, truth[index], color="black", label="Measured", lw=1.3)
            axis.plot(times, estimate[index], color="#D55E00", label="RSS", lw=1.0)
            axis.set_ylabel(f"{joint.title()} (rad)"); axis.grid(alpha=0.2); axis.legend(frameon=False)
        axes[-1].set_xlabel("Time (s)"); fig.suptitle(f"Excluded acquisition-failure sensitivity: test {number}")
        _save_figure(fig, figure_root, "excluded_trial_sensitivity")


def _validation_report(output: Path, quality: pd.DataFrame, source: pd.DataFrame, evidence: pd.DataFrame) -> None:
    eligible = quality.loc[quality.quality_pass]
    excluded = quality.loc[~quality.quality_pass]
    rss = source.loc[source.method == "rss"]
    lines = [
        "# Retrospective QArm RSS validation", "",
        "This is a retrospective practical benchmark under correlated, configuration-dependent QArm motion. It is not an ideal orthogonal-source validation.", "",
        "## Analysis population", "",
        f"- Recorded trials: {len(quality)}", f"- Main quality-eligible trials: {len(eligible)}",
        f"- Excluded trials: {', '.join('test_' + str(value) for value in excluded.test_number) or 'none'}",
        f"- Trials satisfying the prespecified ideal criterion Cmax < 0.1: {int(quality.ideal_low_correlation.sum())} (the ideal analysis set is empty)",
        "- Hardware repeatability cannot be estimated because no trajectory was repeated; seed results measure algorithmic stability only.", "",
        "## Headline RSS results", "",
        "| Joint | Mean |r| | Median |r| | Mean RMSE (rad) | Mean NRMSE |", "|---|---:|---:|---:|---:|",
    ]
    for joint in JOINT_NAMES:
        group = rss.loc[rss.joint == joint]
        lines.append(f"| {joint.title()} | {group.abs_correlation.mean():.3f} | {group.abs_correlation.median():.3f} | {group.rmse_rad.mean():.3f} | {group.nrmse_std.mean():.3f} |")
    lines.extend(["", "## Prespecified evidence decision", "", "| Joint | Correlation improvement CI lower | NRMSE improvement CI lower | Median null percentile (r / NRMSE) | Evidence |", "|---|---:|---:|---:|---:|"])
    for row in evidence.itertuples():
        lines.append(f"| {row.joint.title()} | {row.correlation_improvement_ci95_lower:.3f} | {row.nrmse_improvement_ci95_lower:.3f} | {row.median_correlation_null_percentile:.1f} / {row.median_nrmse_superiority_percentile:.1f} | {'yes' if row.practical_separation_evidence else 'no'} |")
    lines.extend(["", "## Interpretation safeguards", "", "- Only Node 4 gyroscope channels entered RSS.", "- Encoder values were used after fitting for synchronization audit, geometry, matching, and scoring.", "- Source assignment corrected permutation, sign, and constant offset only; amplitude and lag were not fitted.", "- Comparator, null, robustness, confidence-interval, exclusion, and negative results are preserved in the accompanying CSV files.", "- No 12/6/6/3 grouping was created because the recorded protocol does not support it.", ""])
    (output / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_validation(config: ValidationConfig) -> dict[str, str]:
    output = Path(config.output_root)
    output.mkdir(parents=True, exist_ok=True)
    plans = reproduce_alignment_plans(config.data_root, config.sample_rate_hz)
    if config.trial_numbers is not None:
        selected = set(config.trial_numbers)
        plans = [plan for plan in plans if plan.test_number in selected]
    prepared = [prepare_trial(plan, config) for plan in plans]
    quality = pd.DataFrame([trial.quality for trial in prepared])
    geometry = pd.DataFrame([trial.geometry for trial in prepared])
    quality.to_csv(output / "data_quality_manifest.csv", index=False)
    geometry.to_csv(output / "trial_geometry_metrics.csv", index=False)
    manifest = {
        "config": asdict(config),
        "rss_config_seed_0": asdict(frozen_rss_config(config)),
        "distal_imu": DISTAL_PREFIX,
        "ground_truth_columns": TRUTH_COLUMNS,
        "kinematic_convention": {
            "chain": "Rz(base) Ry(shoulder) Ry(elbow) Rx(wrist)",
            "source_columns": ("base", "elbow", "wrist"),
            "axis_signs": (1, 1, -1),
            "reason": "fixed wrist sign makes the orthogonal home triad proper; source-axis sign is ambiguous",
        },
        "alignment_configuration": {
            "gyro_activity_threshold_deg_s": 2.0,
            "maximum_inactive_gap_seconds": 1.5,
            "minimum_movement_seconds": 10.0,
            "alignment_search_seconds": 1.5,
            "alignment_smoothing_seconds": 0.21,
        },
        "velocity_diagnostic": {
            "method": "Savitzky-Golay first derivative",
            "window_samples": VELOCITY_SAVGOL_WINDOW,
            "polynomial_order": VELOCITY_SAVGOL_POLYORDER,
        },
        "created_unix_time": time.time(),
    }
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("config") != _json_safe(manifest["config"]):
            raise ValueError(
                "The output directory contains cached fits from a different configuration; "
                "select a new --output-root."
            )
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    if config.audit_only:
        return {"data_quality_manifest": str(output / "data_quality_manifest.csv"), "trial_geometry_metrics": str(output / "trial_geometry_metrics.csv"), "run_manifest": str(output / "run_manifest.json")}

    # Quality/Cmax-based exclusion is disabled: every prepared trial is fit with the
    # full configuration and included in the headline results. quality_pass and
    # ideal_low_correlation are still computed and reported per trial for diagnostics.
    eligible = list(prepared)
    excluded: list[PreparedTrial] = []
    print(
        f"Audited {len(prepared)} trials: exclusion is disabled, all {len(eligible)} "
        f"will be fit ({int(quality.ideal_low_correlation.sum())} have Cmax < 0.1).",
        flush=True,
    )
    source_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    aligned_by_test: dict[int, dict[str, np.ndarray]] = {}
    for index, trial in enumerate(eligible, 1):
        print(f"Fitting eligible test {trial.test_number} ({index}/{len(eligible)})...", flush=True)
        rows, stability, aligned = _fit_methods(trial, config, output)
        source_rows.extend(rows); stability_rows.extend(stability); aligned_by_test[trial.test_number] = aligned
        null_rows.extend(_random_null(trial, config))
        pd.DataFrame(source_rows).to_csv(output / "comparator_metrics.csv", index=False)
        pd.DataFrame(stability_rows).to_csv(output / "algorithmic_stability.csv", index=False)
        pd.DataFrame(null_rows).to_csv(output / "random_rotation_null.csv", index=False)

    excluded_rows: list[dict[str, object]] = []
    excluded_aligned: dict[int, dict[str, np.ndarray]] = {}
    excluded_config = replace(config, stability_seeds=(), include_sensitivities=False)
    for trial in excluded:
        print(f"Fitting excluded test {trial.test_number} for sensitivity only...", flush=True)
        rows, _, aligned = _fit_methods(trial, excluded_config, output)
        excluded_rows.extend(rows)
        excluded_aligned[trial.test_number] = aligned
    pd.DataFrame(excluded_rows).to_csv(output / "excluded_trial_sensitivity.csv", index=False)

    comparator = pd.DataFrame(source_rows)
    primary_source = comparator.loc[comparator.method == "rss"].copy()
    all_trials = aggregate_trial_rows(comparator)
    primary_trials = all_trials.loc[all_trials.method == "rss"].copy()
    primary_trials = primary_trials.merge(
        quality.drop(columns=[column for column in ("quality_pass",) if column not in quality]),
        on=["test_number", "batch_number"],
        how="left",
    ).merge(geometry, on=["test_number", "batch_number"], how="left")
    primary_source.to_csv(output / "per_source_metrics.csv", index=False)
    primary_trials.to_csv(output / "per_trial_metrics.csv", index=False)
    comparator.to_csv(output / "comparator_metrics.csv", index=False)
    null = pd.DataFrame(null_rows); null.to_csv(output / "random_rotation_null.csv", index=False)
    stability = pd.DataFrame(stability_rows); stability.to_csv(output / "algorithmic_stability.csv", index=False)
    summary = pd.concat((_summary_statistics(comparator, config), _trial_summary_statistics(all_trials, config)), ignore_index=True)
    summary.to_csv(output / "summary_statistics.csv", index=False)
    paired = _paired_comparisons(comparator, config); paired.to_csv(output / "paired_comparisons.csv", index=False)
    percentiles = _null_percentiles(comparator, all_trials, null); percentiles.to_csv(output / "rss_null_percentiles.csv", index=False)
    evidence = _evidence_table(paired, percentiles); evidence.to_csv(output / "practical_separation_evidence.csv", index=False)
    models = _robustness_models(all_trials, quality, geometry.loc[geometry.test_number.isin(primary_trials.test_number)], config)
    models.to_csv(output / "robustness_models.csv", index=False)
    _make_figures(output, prepared, comparator, all_trials, quality, geometry, percentiles, stability, aligned_by_test, excluded_aligned)
    _validation_report(output, quality, primary_source, evidence)
    return {path.stem: str(path) for path in output.iterdir() if path.is_file()}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Sample_Data/QArm"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/qarm_rss_validation"))
    parser.add_argument("--rss-epochs", type=int, default=2000)
    parser.add_argument("--pso-swarm-size", type=int, default=200)
    parser.add_argument("--null-draws", type=int, default=1000)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--stability-seeds", type=int, nargs="*", default=[1, 2, 3, 4])
    parser.add_argument("--trials", type=int, nargs="*")
    parser.add_argument("--minimum-alignment-correlation", type=float, default=0.85)
    parser.add_argument("--skip-sensitivities", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rss_epochs < 1 or args.pso_swarm_size < 1 or args.null_draws < 1 or args.bootstrap_resamples < 1:
        raise SystemExit("Epochs, swarm size, null draws, and bootstrap resamples must be positive.")
    config = ValidationConfig(
        data_root=str(args.data_root.resolve()), output_root=str(args.output_root.resolve()),
        rss_epochs=args.rss_epochs, pso_swarm_size=args.pso_swarm_size,
        null_draws=args.null_draws, stability_seeds=tuple(args.stability_seeds),
        bootstrap_resamples=args.bootstrap_resamples,
        minimum_alignment_correlation=args.minimum_alignment_correlation,
        include_sensitivities=not args.skip_sensitivities,
        trial_numbers=None if args.trials is None else tuple(args.trials), audit_only=args.audit_only,
    )
    outputs = run_validation(config)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
