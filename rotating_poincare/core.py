"""Numerical core for the Rotating Poincare Section Transform (RPT)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.signal import savgol_filter


InterpolationMethod = Literal["pchip", "cubic"]
CrossingDirection = Literal["both", "positive", "negative"]


@dataclass(frozen=True)
class RPTConfig:
    """Configuration shared by the reusable transform and command-line runner."""

    normalize: bool = True
    derivative_smoothing: bool = True
    savgol_window: int = 21
    savgol_polyorder: int = 3
    interp_points: int = 5000
    interpolation_method: InterpolationMethod = "pchip"
    theta_min_deg: float = 0.0
    theta_max_deg: float = 360.0
    theta_step_deg: float = 1.0
    num_radial_bins: int = 80
    radial_min: float | None = None
    radial_max: float | None = None
    crossing_direction: CrossingDirection = "both"
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.savgol_window < 1:
            raise ValueError("savgol_window must be >= 1.")
        if self.savgol_polyorder < 0:
            raise ValueError("savgol_polyorder must be >= 0.")
        if self.interp_points < 2:
            raise ValueError("interp_points must be >= 2.")
        if self.interpolation_method not in {"pchip", "cubic"}:
            raise ValueError("interpolation_method must be 'pchip' or 'cubic'.")
        if not np.isfinite(self.theta_min_deg) or not np.isfinite(self.theta_max_deg):
            raise ValueError("theta bounds must be finite.")
        if self.theta_max_deg <= self.theta_min_deg:
            raise ValueError("theta_max_deg must be greater than theta_min_deg.")
        if not np.isfinite(self.theta_step_deg) or self.theta_step_deg <= 0.0:
            raise ValueError("theta_step_deg must be > 0.")
        if self.num_radial_bins < 1:
            raise ValueError("num_radial_bins must be >= 1.")
        if self.radial_min is not None and not np.isfinite(self.radial_min):
            raise ValueError("radial_min must be finite when supplied.")
        if self.radial_min is not None and self.radial_min < 0.0:
            raise ValueError("radial_min must be >= 0 for a positive-ray section.")
        if self.radial_max is not None and not np.isfinite(self.radial_max):
            raise ValueError("radial_max must be finite when supplied.")
        if (
            self.radial_min is not None
            and self.radial_max is not None
            and self.radial_max <= self.radial_min
        ):
            raise ValueError("radial_max must be greater than radial_min.")
        if self.crossing_direction not in {"both", "positive", "negative"}:
            raise ValueError("crossing_direction must be 'both', 'positive', or 'negative'.")
        if not np.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("eps must be a finite value > 0.")


@dataclass(frozen=True)
class NormalizationResult:
    """Centered/scaled phase points and the coordinate statistics used."""

    points: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RPTResult:
    """Arrays defining a complete Rotating Poincare Section Transform."""

    theta_values_deg: np.ndarray
    radial_bin_edges: np.ndarray
    radial_bin_centers: np.ndarray
    count_matrix: np.ndarray


def compute_derivative(
    x: np.ndarray,
    t: np.ndarray | None = None,
    fs: float | None = None,
    smooth: bool = True,
    savgol_window: int = 21,
    savgol_polyorder: int = 3,
    *,
    warnings: list[str] | None = None,
) -> np.ndarray:
    """Compute ``dx/dt``, optionally smoothing a copy of ``x`` first.

    The returned derivative always has the same length as ``x``. When a
    requested Savitzky-Golay window is too large, it is reduced to the largest
    usable odd window and the adjustment is reported through ``warnings``.
    """

    values = np.asarray(x, dtype=float).reshape(-1)
    if values.size < 2:
        raise ValueError("At least two samples are required to compute a derivative.")
    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite values.")
    messages = warnings if warnings is not None else []

    derivative_input = values
    if smooth:
        window = min(int(savgol_window), values.size if values.size % 2 == 1 else values.size - 1)
        if window % 2 == 0:
            window -= 1
        minimum_window = int(savgol_polyorder) + 1
        if minimum_window % 2 == 0:
            minimum_window += 1
        if window >= minimum_window and window >= 3:
            if window != int(savgol_window):
                messages.append(
                    f"Savitzky-Golay window reduced from {int(savgol_window)} to {window} "
                    "for the available samples."
                )
            derivative_input = savgol_filter(
                values,
                window_length=window,
                polyorder=int(savgol_polyorder),
                mode="interp",
            )
        else:
            messages.append(
                "Derivative smoothing was skipped because the signal is too short "
                "for the requested Savitzky-Golay polynomial order."
            )

    if t is not None:
        time = np.asarray(t, dtype=float).reshape(-1)
        if time.shape != values.shape:
            raise ValueError("t must have the same length as x.")
        if not np.all(np.isfinite(time)):
            raise ValueError("t must contain only finite values.")
        if np.any(np.diff(time) <= 0.0):
            raise ValueError("Time values must be strictly increasing.")
        return np.asarray(np.gradient(derivative_input, time), dtype=float)

    if fs is not None:
        sample_rate = float(fs)
        if not np.isfinite(sample_rate) or sample_rate <= 0.0:
            raise ValueError("fs must be a finite value > 0.")
        return np.asarray(np.gradient(derivative_input) * sample_rate, dtype=float)

    return np.asarray(np.gradient(derivative_input), dtype=float)


def make_phase_points(x: np.ndarray, dxdt: np.ndarray) -> np.ndarray:
    """Form the phase portrait points ``[x, dx/dt]``."""

    values = np.asarray(x, dtype=float).reshape(-1)
    derivative = np.asarray(dxdt, dtype=float).reshape(-1)
    if values.shape != derivative.shape:
        raise ValueError("x and dxdt must have the same length.")
    points = np.column_stack((values, derivative))
    if not np.all(np.isfinite(points)):
        raise ValueError("Phase points must contain only finite values.")
    return points


def center_and_normalize_phase_points(
    points: np.ndarray,
    normalize: bool = True,
    eps: float = 1e-12,
) -> NormalizationResult:
    """Center phase points and optionally map each coordinate to [-1, 1]."""

    phase = _validate_points(points)
    messages: list[str] = []

    if normalize:
        minimum = np.min(phase, axis=0)
        maximum = np.max(phase, axis=0)
        center = (minimum + maximum) / 2.0
        half_range = (maximum - minimum) / 2.0
        scale = np.ones(2, dtype=float)
        stable = half_range > float(eps)
        scale[stable] = half_range[stable]
        centered = phase - center
        for index, label in enumerate(("signal", "derivative")):
            if not stable[index]:
                messages.append(
                    f"The {label} coordinate has near-zero range; "
                    "it was mapped to zero."
                )
                centered[:, index] = 0.0
        centered[:, stable] = centered[:, stable] / scale[stable]
    else:
        center = np.mean(phase, axis=0)
        scale = np.ones(2, dtype=float)
        centered = phase - center

    return NormalizationResult(
        points=np.asarray(centered, dtype=float),
        mean=np.asarray(center, dtype=float),
        scale=scale,
        warnings=tuple(messages),
    )


def arc_length_parameterize(
    points: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unique normalized arc coordinates and corresponding phase points."""

    phase = _validate_points(points)
    if phase.shape[0] == 1:
        return np.array([0.0], dtype=float), phase.copy()

    segment_lengths = np.linalg.norm(np.diff(phase, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= float(eps):
        return np.array([0.0], dtype=float), phase[:1].copy()

    s = cumulative / total
    keep = np.concatenate(([True], np.diff(s) > float(eps)))
    return np.asarray(s[keep], dtype=float), np.asarray(phase[keep], dtype=float)


def interpolate_trajectory(
    points: np.ndarray,
    interp_points: int = 5000,
    method: InterpolationMethod = "pchip",
    eps: float = 1e-12,
) -> np.ndarray:
    """Arc-length parameterize and resample a phase trajectory."""

    if int(interp_points) < 2:
        raise ValueError("interp_points must be >= 2.")
    if method not in {"pchip", "cubic"}:
        raise ValueError("method must be 'pchip' or 'cubic'.")

    s, unique_points = arc_length_parameterize(points, eps=eps)
    if unique_points.shape[0] == 1:
        return np.repeat(unique_points, int(interp_points), axis=0)
    if method == "cubic" and unique_points.shape[0] < 4:
        raise ValueError("Cubic interpolation requires at least four unique arc-length points.")

    sample_s = np.linspace(0.0, 1.0, int(interp_points), dtype=float)
    interpolator_type = PchipInterpolator if method == "pchip" else CubicSpline
    x_interpolator = interpolator_type(s, unique_points[:, 0])
    y_interpolator = interpolator_type(s, unique_points[:, 1])
    interpolated = np.column_stack((x_interpolator(sample_s), y_interpolator(sample_s)))
    if not np.all(np.isfinite(interpolated)):
        raise ValueError("Interpolation produced non-finite phase points.")
    return np.asarray(interpolated, dtype=float)


def _crossings_for_section(
    points: np.ndarray,
    along: np.ndarray,
    normal: np.ndarray,
    direction: CrossingDirection,
    eps: float,
) -> np.ndarray:
    """Find nonnegative radial coordinates for one positive-ray section."""

    distances = points @ normal
    signs = np.zeros(distances.size, dtype=np.int8)
    signs[distances > eps] = 1
    signs[distances < -eps] = -1

    coordinates: list[np.ndarray] = []
    direction_values: list[np.ndarray] = []

    strict_indices = np.flatnonzero(signs[:-1] * signs[1:] == -1)
    if strict_indices.size:
        d0 = distances[strict_indices]
        d1 = distances[strict_indices + 1]
        alpha = -d0 / (d1 - d0)
        crossings = points[strict_indices] + alpha[:, None] * (
            points[strict_indices + 1] - points[strict_indices]
        )
        coordinates.append(crossings)
        direction_values.append(d1 - d0)

    zero_indices = np.flatnonzero(signs == 0)
    if zero_indices.size:
        split_at = np.flatnonzero(np.diff(zero_indices) > 1) + 1
        for group in np.split(zero_indices, split_at):
            start = int(group[0])
            end = int(group[-1])
            left = start - 1
            right = end + 1
            if left < 0 or right >= points.shape[0] or signs[left] * signs[right] != -1:
                continue
            d_left = float(distances[left])
            d_right = float(distances[right])
            alpha = -d_left / (d_right - d_left)
            crossing = points[left] + alpha * (points[right] - points[left])
            coordinates.append(crossing.reshape(1, 2))
            direction_values.append(np.array([d_right - d_left], dtype=float))

    if not coordinates:
        return np.empty(0, dtype=float)

    crossing_points = np.vstack(coordinates)
    crossing_directions = np.concatenate(direction_values)
    if direction == "positive":
        keep = crossing_directions > eps
    elif direction == "negative":
        keep = crossing_directions < -eps
    else:
        keep = np.ones(crossing_directions.size, dtype=bool)
    radial_coordinates = np.asarray(crossing_points[keep] @ along, dtype=float)
    on_positive_ray = radial_coordinates >= -float(eps)
    radial_coordinates = radial_coordinates[on_positive_ray]
    radial_coordinates[np.abs(radial_coordinates) <= float(eps)] = 0.0
    return radial_coordinates


def compute_rotating_poincare_transform(
    points_interp: np.ndarray,
    theta_min_deg: float = 0.0,
    theta_max_deg: float = 360.0,
    theta_step_deg: float = 1.0,
    num_radial_bins: int = 80,
    radial_min: float | None = None,
    radial_max: float | None = None,
    crossing_direction: CrossingDirection = "both",
    eps: float = 1e-12,
) -> RPTResult:
    """Compute positive-ray crossing histograms for an interpolated path."""

    points = _validate_points(points_interp)
    config = RPTConfig(
        theta_min_deg=theta_min_deg,
        theta_max_deg=theta_max_deg,
        theta_step_deg=theta_step_deg,
        num_radial_bins=num_radial_bins,
        radial_min=radial_min,
        radial_max=radial_max,
        crossing_direction=crossing_direction,
        eps=eps,
    )
    theta_values = np.arange(
        config.theta_min_deg,
        config.theta_max_deg,
        config.theta_step_deg,
        dtype=float,
    )
    if theta_values.size == 0:
        raise ValueError("The theta configuration produced an empty angle grid.")

    radius = float(np.max(np.linalg.norm(points, axis=1)))
    safe_radius = max(radius, config.eps)
    lower = float(config.radial_min) if config.radial_min is not None else 0.0
    upper = float(config.radial_max) if config.radial_max is not None else safe_radius
    if upper <= lower:
        raise ValueError("Resolved radial_max must be greater than radial_min.")
    edges = np.linspace(lower, upper, config.num_radial_bins + 1, dtype=float)

    counts = np.zeros((theta_values.size, config.num_radial_bins), dtype=np.int64)
    for theta_index, theta_deg in enumerate(theta_values):
        theta = np.deg2rad(theta_deg)
        along = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        normal = np.array([-np.sin(theta), np.cos(theta)], dtype=float)
        radial_crossings = _crossings_for_section(
            points,
            along,
            normal,
            config.crossing_direction,
            config.eps,
        )
        counts[theta_index], _ = np.histogram(radial_crossings, bins=edges)

    return RPTResult(
        theta_values_deg=theta_values,
        radial_bin_edges=edges,
        radial_bin_centers=0.5 * (edges[:-1] + edges[1:]),
        count_matrix=counts,
    )


def compute_rpt_features(
    count_matrix: np.ndarray,
    theta_values_deg: np.ndarray,
    radial_bin_centers: np.ndarray,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Compute section-wise crossing count, moments, entropy, and dominant bin."""

    counts = np.asarray(count_matrix)
    theta = np.asarray(theta_values_deg, dtype=float).reshape(-1)
    centers = np.asarray(radial_bin_centers, dtype=float).reshape(-1)
    if counts.ndim != 2:
        raise ValueError("count_matrix must be two-dimensional.")
    if counts.shape != (theta.size, centers.size):
        raise ValueError("count_matrix shape must match theta values and radial bin centers.")
    if np.any(counts < 0) or not np.all(np.isfinite(counts)):
        raise ValueError("count_matrix must contain finite nonnegative values.")

    counts_float = counts.astype(float)
    totals = np.sum(counts_float, axis=1)
    probabilities = np.divide(
        counts_float,
        totals[:, None],
        out=np.zeros_like(counts_float),
        where=totals[:, None] > 0.0,
    )
    means = probabilities @ centers
    variances = np.sum(probabilities * (centers[None, :] - means[:, None]) ** 2, axis=1)
    entropy = -np.sum(
        np.where(probabilities > 0.0, probabilities * np.log(probabilities + float(eps)), 0.0),
        axis=1,
    )
    has_crossings = totals > 0.0
    dominant_indices = np.full(theta.size, -1, dtype=int)
    dominant_indices[has_crossings] = np.argmax(counts[has_crossings], axis=1)
    dominant_r = np.full(theta.size, np.nan, dtype=float)
    dominant_r[has_crossings] = centers[dominant_indices[has_crossings]]

    return pd.DataFrame(
        {
            "theta_deg": theta,
            "total_crossings": totals.astype(np.int64),
            "mean_r": means,
            "var_r": variances,
            "entropy": entropy,
            "dominant_bin_index": dominant_indices,
            "dominant_r": dominant_r,
        }
    )


def _validate_points(points: np.ndarray) -> np.ndarray:
    phase = np.asarray(points, dtype=float)
    if phase.ndim != 2 or phase.shape[1] != 2:
        raise ValueError("points must have shape (n_samples, 2).")
    if phase.shape[0] < 1:
        raise ValueError("points must contain at least one sample.")
    if not np.all(np.isfinite(phase)):
        raise ValueError("points must contain only finite values.")
    return phase
