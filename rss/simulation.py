"""Deterministic synthetic scenarios for RSS validation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from rss.axes import normalize_rows
from rss.model import (
    angles_to_quaternion_components,
    change_quaternion_basis,
    compose_source_quaternions,
    decode_scalar_and_vector,
    quaternions_to_body_rotation_increments,
    rotate_axis_triad,
)


_SOURCE_ZERO_MEAN_ATOL = 1e-12


def orthonormality_error(axes: np.ndarray) -> float:
    """Return the Frobenius error of the row-axis Gram matrix from identity."""

    array = np.asarray(axes, dtype=float).reshape(3, 3)
    return float(np.linalg.norm(array @ array.T - np.eye(3), ord="fro"))


def source_cross_correlation_matrix(signals: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Return the mean-centered Pearson correlation matrix for one or more signals."""

    array = np.asarray(signals, dtype=float)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 2:
        raise ValueError("signals must have shape (N, T) with N >= 1 and T >= 2.")
    if not np.isfinite(array).all():
        raise ValueError("signals must contain only finite values.")
    centered = array - np.mean(array, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    if np.any(norms <= float(eps)):
        raise ValueError("Every signal must have nonzero variance.")
    correlation = (centered @ centered.T) / (norms[:, None] * norms[None, :])
    return np.clip(correlation, -1.0, 1.0)


def _symmetric_whitening(centered: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Return symmetrically (ZCA) whitened rows of an already mean-centered array."""

    covariance = (centered @ centered.T) / float(centered.shape[1])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    rank_tolerance = max(
        float(eps),
        float(np.max(eigenvalues)) * np.finfo(float).eps * max(covariance.shape),
    )
    if np.any(eigenvalues <= rank_tolerance):
        raise ValueError("Source signals must be linearly independent for symmetric whitening.")
    inverse_sqrt = eigenvectors @ np.diag(np.reciprocal(np.sqrt(eigenvalues))) @ eigenvectors.T
    return inverse_sqrt @ centered


def equicorrelation_matrix(count: int, correlation: float) -> np.ndarray:
    """Return the unit-diagonal matrix whose every off-diagonal entry is ``correlation``."""

    size = int(count)
    if size < 2:
        raise ValueError("count must be >= 2.")
    value = float(correlation)
    if not math.isfinite(value):
        raise ValueError("correlation must be finite.")
    # Eigenvalues are 1 + (size - 1) * rho once and 1 - rho with multiplicity size - 1,
    # so positive definiteness holds exactly on this open interval.
    lower = -1.0 / float(size - 1)
    if not lower < value < 1.0:
        raise ValueError(
            f"correlation must lie strictly between {lower:.6g} and 1 for a positive definite target."
        )
    return (1.0 - value) * np.eye(size, dtype=float) + value * np.ones((size, size), dtype=float)


def _correlation_matrix_square_root(target: np.ndarray) -> np.ndarray:
    """Return the symmetric positive definite square root of a correlation target."""

    symmetric = 0.5 * (target + target.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("target_correlation must be positive definite.")
    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T


def correlate_source_angles(
    sources: np.ndarray,
    target_correlation: float | np.ndarray,
    *,
    eps: float = 1e-12,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Recolor source rows to an exact target correlation, retaining means and peaks.

    This is the counterpart of :func:`_decorrelate_source_angles`: both symmetrically
    whiten the rows first, but this helper then applies the symmetric square root of
    the requested correlation matrix instead of stopping at the identity. Correlation
    is invariant to positive per-row scaling and to constant shifts, so the peak
    rescaling and mean restoration below preserve the realized correlation exactly.
    """

    array = np.asarray(sources, dtype=float)
    # The public helper provides shape, finite-value, and variance validation.
    source_cross_correlation_matrix(array, eps=eps)
    count = int(array.shape[0])
    if np.isscalar(target_correlation) or np.ndim(target_correlation) == 0:
        target = equicorrelation_matrix(count, float(target_correlation))
    else:
        target = np.asarray(target_correlation, dtype=float)
        if target.shape != (count, count):
            raise ValueError("target_correlation must be square with one entry per source row.")
        if not np.isfinite(target).all():
            raise ValueError("target_correlation must contain only finite values.")
        if not np.allclose(target, target.T, rtol=0.0, atol=1e-12):
            raise ValueError("target_correlation must be symmetric.")
        if not np.allclose(np.diag(target), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("target_correlation must have a unit diagonal.")
    means = np.mean(array, axis=1, keepdims=True)
    centered = array - means
    original_peaks = np.max(np.abs(centered), axis=1)
    whitened = _symmetric_whitening(centered, eps=eps)
    colored = _correlation_matrix_square_root(target) @ whitened
    colored_peaks = np.max(np.abs(colored), axis=1)
    if np.any(colored_peaks <= float(eps)):
        raise ValueError("Correlation coloring produced a zero-amplitude source signal.")
    result = colored * (original_peaks / colored_peaks)[:, None] + means
    if not np.isfinite(result).all():
        raise ValueError("Correlation coloring produced non-finite source signals.")
    realized = source_cross_correlation_matrix(result, eps=eps)
    upper = np.triu_indices(count, k=1)
    deviation = float(np.max(np.abs(realized[upper] - target[upper]))) if upper[0].size else 0.0
    if not deviation <= float(tolerance):
        raise RuntimeError(
            "Could not realize the requested source cross-correlation: "
            f"max_deviation={deviation:.6g}, tolerance={float(tolerance):.6g}."
        )
    return result


def _decorrelate_source_angles(
    sources: np.ndarray,
    *,
    cross_correlation_limit: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Symmetrically whiten source rows while retaining their means and peaks."""

    array = np.asarray(sources, dtype=float)
    # The public helper provides shape, finite-value, and variance validation.
    source_cross_correlation_matrix(array, eps=eps)
    means = np.mean(array, axis=1, keepdims=True)
    centered = array - means
    original_peaks = np.max(np.abs(centered), axis=1)
    whitened = _symmetric_whitening(centered, eps=eps)
    whitened_peaks = np.max(np.abs(whitened), axis=1)
    if np.any(whitened_peaks <= float(eps)):
        raise ValueError("Symmetric whitening produced a zero-amplitude source signal.")
    decorrelated = whitened * (original_peaks / whitened_peaks)[:, None] + means
    if not np.isfinite(decorrelated).all():
        raise ValueError("Symmetric whitening produced non-finite source signals.")
    correlation = source_cross_correlation_matrix(decorrelated, eps=eps)
    off_diagonal = correlation - np.diag(np.diag(correlation))
    maximum = float(np.max(np.abs(off_diagonal)))
    if not maximum < float(cross_correlation_limit):
        raise RuntimeError(
            "Could not enforce the source cross-correlation limit: "
            f"maximum={maximum:.6g}, limit={float(cross_correlation_limit):.6g}."
        )
    return decorrelated


def _center_source_angles(
    sources: np.ndarray,
    source_roles: tuple[str, str, str],
) -> np.ndarray:
    """Return source rows with zero temporal mean while preserving sparse support."""

    centered = np.asarray(sources, dtype=float).copy()
    if centered.ndim != 2 or centered.shape[0] != len(source_roles):
        raise ValueError("sources must contain one row per source role.")
    if not np.isfinite(centered).all():
        raise ValueError("sources must contain only finite values.")
    for index, role in enumerate(source_roles):
        if role == "inactive":
            centered[index] = 0.0
            continue
        if role == "sparse":
            support = centered[index] != 0.0
            if not np.any(support):
                raise ValueError("Active sparse sources must contain nonzero samples.")
            centered[index, support] -= (
                float(np.sum(centered[index])) / float(np.count_nonzero(support))
            )
        else:
            centered[index] -= float(np.mean(centered[index]))
    residual_means = np.mean(centered, axis=1)
    if not np.allclose(
        residual_means,
        0.0,
        rtol=0.0,
        atol=_SOURCE_ZERO_MEAN_ATOL,
    ):
        raise RuntimeError(
            "Could not enforce zero temporal means for the source signals: "
            f"means={residual_means.tolist()}."
        )
    return centered


def random_orthonormal_axes(count: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(int(count), int(count)))
    axes, _, _ = np.linalg.svd(matrix)
    for index in range(axes.shape[0]):
        pivot = int(np.argmax(np.abs(axes[index])))
        if axes[index, pivot] < 0.0:
            axes[index] *= -1.0
    if float(np.linalg.det(axes)) < 0.0:
        axes[-1] *= -1.0
    return axes


def symmetric_orthogonal_mixing_matrix(rng: np.random.Generator, count: int = 3) -> np.ndarray:
    direction = rng.normal(size=int(count))
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    return 2.0 * np.outer(direction, direction) - np.eye(int(count))


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 0
    T: int = 2000
    duration_seconds: float = 1.0
    base_frequencies_hz: tuple[float, float, float] = (2.0, 5.0, 11.0)
    modulation_frequencies_hz: tuple[float, float, float] = (0.5, 0.9, 1.3)
    modulation_depth: float = 0.15
    secondary_harmonic_scale: float = 0.18
    amplitude_jitter: float = 0.08
    source_cross_correlation_limit: float = 0.01
    observation_model: str = "body_fixed_gyro"

    def __post_init__(self) -> None:
        if int(self.T) < 8 or float(self.duration_seconds) <= 0.0:
            raise ValueError("T must be >= 8 and duration_seconds must be positive.")
        if len(self.base_frequencies_hz) != 3 or len(self.modulation_frequencies_hz) != 3:
            raise ValueError("Frequency tuples must contain three values.")
        limit = float(self.source_cross_correlation_limit)
        if not math.isfinite(limit) or not 0.0 < limit < 1.0:
            raise ValueError("source_cross_correlation_limit must be finite and strictly between 0 and 1.")
        if str(self.observation_model).strip().lower() not in {
            "body_fixed_gyro",
            "legacy_vector_mixing",
        }:
            raise ValueError(
                "observation_model must be 'body_fixed_gyro' or 'legacy_vector_mixing'."
            )


@dataclass(frozen=True)
class SuiteCaseSpec:
    family: str
    case_id: str
    seed: int
    frame_count: int
    dt: float
    amplitude_tier: str
    stress_peak_angle_rad: float | None = None

    def __post_init__(self) -> None:
        allowed = {"harmonic", "filtered_noise", "chirp_modulated", "piecewise_transient"}
        if self.family not in allowed:
            raise ValueError(f"Unsupported family {self.family!r}.")
        if int(self.frame_count) < 8 or float(self.dt) <= 0.0:
            raise ValueError("frame_count must be >= 8 and dt must be positive.")


@dataclass(frozen=True)
class BatchStudyConfig:
    standard_families: tuple[str, ...] = (
        "harmonic",
        "filtered_noise",
        "chirp_modulated",
        "piecewise_transient",
    )
    standard_cases_per_family: int = 5
    batch_sequence_lengths: tuple[int, ...] = (720, 840, 960, 1080)
    standard_amplitude_tiers: tuple[str, ...] = ("low", "medium", "high", "medium", "high")
    dt: float = 0.01
    low_amplitude_range_rad: tuple[float, float] = (0.22, 0.50)
    medium_amplitude_range_rad: tuple[float, float] = (0.50, 0.90)
    high_amplitude_range_rad: tuple[float, float] = (0.90, 1.30)
    stress_case_count: int = 8
    stress_min_peak_rad: float = 0.55
    stress_max_peak_rad: float = 2.65


FORMAL_SOURCE_COMPOSITIONS: tuple[tuple[str, tuple[str, str, str]], ...] = (
    ("single_rotation", ("dense", "inactive", "inactive")),
    ("dual_rotation", ("dense", "dense", "inactive")),
    ("triple_rotation", ("dense", "dense", "dense")),
    ("single_rotation_one_sparse", ("dense", "sparse", "inactive")),
    ("dual_rotation_one_sparse", ("dense", "dense", "sparse")),
    ("single_rotation_two_sparse", ("dense", "sparse", "sparse")),
)


@dataclass(frozen=True)
class FormalStudyConfig:
    """Controls for the formal dense/sparse RSS validation suite."""

    families: tuple[str, ...] = (
        "harmonic",
        "filtered_noise",
        "chirp_modulated",
        "piecewise_transient",
    )
    cases_per_family: int = 5
    batch_sequence_lengths: tuple[int, ...] = (720, 840, 960, 1080)
    dt: float = 0.01
    amplitude_range_rad: tuple[float, float] = (0.22, 1.30)
    frequency_range_hz: tuple[float, float] = (0.20, 4.00)
    sparse_duty_cycle_range: tuple[float, float] = (0.10, 0.20)
    base_seed: int = 9100

    def __post_init__(self) -> None:
        allowed = {"harmonic", "filtered_noise", "chirp_modulated", "piecewise_transient"}
        if not self.families or any(family not in allowed for family in self.families):
            raise ValueError("families must contain only supported signal families.")
        if int(self.cases_per_family) < 1:
            raise ValueError("cases_per_family must be >= 1.")
        if not self.batch_sequence_lengths or any(int(value) < 8 for value in self.batch_sequence_lengths):
            raise ValueError("batch_sequence_lengths must contain values >= 8.")
        if not math.isfinite(float(self.dt)) or float(self.dt) <= 0.0:
            raise ValueError("dt must be finite and positive.")
        for name, bounds in (
            ("amplitude_range_rad", self.amplitude_range_rad),
            ("frequency_range_hz", self.frequency_range_hz),
            ("sparse_duty_cycle_range", self.sparse_duty_cycle_range),
        ):
            values = np.asarray(bounds, dtype=float)
            if values.shape != (2,) or not np.isfinite(values).all() or values[0] >= values[1]:
                raise ValueError(f"{name} must contain finite increasing bounds.")
        if float(self.amplitude_range_rad[0]) <= 0.0:
            raise ValueError("amplitude_range_rad must be strictly positive.")
        if float(self.frequency_range_hz[0]) <= 0.0:
            raise ValueError("frequency_range_hz must be strictly positive.")
        if not 0.0 < float(self.sparse_duty_cycle_range[0]) < float(
            self.sparse_duty_cycle_range[1]
        ) < 1.0:
            raise ValueError("sparse_duty_cycle_range must lie strictly inside (0, 1).")


@dataclass(frozen=True)
class FormalCaseSpec:
    """Fully resolved, reproducible formal experiment case."""

    family: str
    composition: str
    case_id: str
    replicate_index: int
    seed: int
    frame_count: int
    dt: float
    source_roles: tuple[str, str, str]
    requested_amplitudes_rad: tuple[float | None, float | None, float | None]
    requested_frequencies_hz: tuple[float | None, float | None, float | None]
    requested_sparse_duty_cycles: tuple[float | None, float | None, float | None]

    def __post_init__(self) -> None:
        allowed_families = {"harmonic", "filtered_noise", "chirp_modulated", "piecewise_transient"}
        allowed_roles = {"dense", "sparse", "inactive"}
        if self.family not in allowed_families:
            raise ValueError(f"Unsupported family {self.family!r}.")
        if self.composition not in dict(FORMAL_SOURCE_COMPOSITIONS):
            raise ValueError(f"Unsupported formal composition {self.composition!r}.")
        if tuple(self.source_roles) != dict(FORMAL_SOURCE_COMPOSITIONS)[self.composition]:
            raise ValueError("source_roles do not match the named formal composition.")
        if int(self.replicate_index) < 1:
            raise ValueError("replicate_index must be >= 1.")
        if int(self.frame_count) < 8 or not math.isfinite(float(self.dt)) or float(self.dt) <= 0.0:
            raise ValueError("frame_count must be >= 8 and dt must be finite and positive.")
        if len(self.source_roles) != 3 or any(role not in allowed_roles for role in self.source_roles):
            raise ValueError("source_roles must contain three dense, sparse, or inactive values.")
        if any(
            len(values) != 3
            for values in (
                self.requested_amplitudes_rad,
                self.requested_frequencies_hz,
                self.requested_sparse_duty_cycles,
            )
        ):
            raise ValueError("Formal source parameter tuples must contain three values.")
        for index, role in enumerate(self.source_roles):
            amplitude = self.requested_amplitudes_rad[index]
            frequency = self.requested_frequencies_hz[index]
            duty_cycle = self.requested_sparse_duty_cycles[index]
            if role == "inactive":
                if amplitude is not None or frequency is not None or duty_cycle is not None:
                    raise ValueError("Inactive sources must not define amplitude, frequency, or duty cycle.")
                continue
            if amplitude is None or not math.isfinite(float(amplitude)) or float(amplitude) <= 0.0:
                raise ValueError("Active sources require a finite positive amplitude.")
            if frequency is None or not math.isfinite(float(frequency)) or float(frequency) <= 0.0:
                raise ValueError("Active sources require a finite positive frequency.")
            if role == "sparse":
                if duty_cycle is None or not 0.0 < float(duty_cycle) < 1.0:
                    raise ValueError("Sparse sources require a duty cycle in (0, 1).")
            elif duty_cycle is not None:
                raise ValueError("Dense sources must not define a sparse duty cycle.")


@dataclass(frozen=True)
class ScenarioData:
    config: SimulationConfig
    time_seconds: np.ndarray
    source_axes_global: np.ndarray
    observer_axes_global: np.ndarray
    source_axes_observer_frame: np.ndarray
    mixing_matrix: np.ndarray
    generated_sources: np.ndarray
    relative_sources: np.ndarray
    sources: np.ndarray
    observations: np.ndarray
    physical_object_quaternions: np.ndarray
    physical_observer_quaternions: np.ndarray
    observer_rotation_increments: np.ndarray
    observer_angular_velocities: np.ndarray
    source_axes_global_time: np.ndarray
    observer_axes_global_time: np.ndarray
    observation_model: str
    shared_quaternion_scalar: np.ndarray
    source_coefficients: np.ndarray
    observation_coefficients: np.ndarray
    family: str = "legacy"
    case_id: str = "legacy_single"
    amplitude_tier: str = "single"
    generator_label: str = "legacy"
    stress_peak_angle_rad: float | None = None
    composition: str = "triple_rotation"
    replicate_index: int | None = None
    source_roles: tuple[str, str, str] = ("dense", "dense", "dense")
    requested_amplitudes_rad: tuple[float | None, float | None, float | None] = (
        None,
        None,
        None,
    )
    realized_amplitudes_rad: tuple[float | None, float | None, float | None] = (
        None,
        None,
        None,
    )
    requested_frequencies_hz: tuple[float | None, float | None, float | None] = (
        None,
        None,
        None,
    )
    realized_frequencies_hz: tuple[float | None, float | None, float | None] = (
        None,
        None,
        None,
    )
    sparse_duty_cycles: tuple[float | None, float | None, float | None] = (
        None,
        None,
        None,
    )

    @property
    def active_source_indices(self) -> tuple[int, ...]:
        return tuple(index for index, role in enumerate(self.source_roles) if role != "inactive")

    @property
    def sparse_source_indices(self) -> tuple[int, ...]:
        return tuple(index for index, role in enumerate(self.source_roles) if role == "sparse")

    @property
    def raw_sources(self) -> np.ndarray:
        return self.generated_sources

    @property
    def recovery_sources(self) -> np.ndarray:
        return self.relative_sources

    @property
    def object_quaternions_global(self) -> np.ndarray:
        return self.physical_object_quaternions

    @property
    def observer_quaternions(self) -> np.ndarray:
        return self.physical_observer_quaternions

    @property
    def observer_increments(self) -> np.ndarray:
        return self.observer_rotation_increments

    @property
    def moving_source_axes(self) -> np.ndarray:
        return self.source_axes_global_time

    @property
    def moving_observer_axes(self) -> np.ndarray:
        return self.observer_axes_global_time


def _standardize(signal: np.ndarray) -> np.ndarray:
    centered = signal - float(np.mean(signal))
    peak = max(float(np.max(np.abs(centered))), 1e-12)
    return centered / peak


def _amplitude_range(tier: str, config: BatchStudyConfig) -> tuple[float, float]:
    mapping = {
        "low": config.low_amplitude_range_rad,
        "medium": config.medium_amplitude_range_rad,
        "high": config.high_amplitude_range_rad,
    }
    if tier not in mapping:
        raise ValueError(f"Unknown amplitude tier {tier!r}.")
    return mapping[tier]


def _legacy_sources(config: SimulationConfig, rng: np.random.Generator, time: np.ndarray) -> np.ndarray:
    rows = []
    for index, (base, modulation) in enumerate(zip(config.base_frequencies_hz, config.modulation_frequencies_hz)):
        amplitude = (0.55 + 0.12 * index) * (1.0 + rng.uniform(-config.amplitude_jitter, config.amplitude_jitter))
        phase = rng.uniform(-np.pi, np.pi)
        envelope = 1.0 + config.modulation_depth * np.sin(2.0 * np.pi * modulation * time + 0.5 * phase)
        fundamental = np.sin(2.0 * np.pi * base * time + phase)
        harmonic = config.secondary_harmonic_scale * np.sin(4.0 * np.pi * base * time - 0.3 * phase)
        rows.append(amplitude * envelope * (fundamental + harmonic))
    return np.stack(rows)


def _family_sources(
    family: str,
    time: np.ndarray,
    rng: np.random.Generator,
    amplitudes: np.ndarray,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for index in range(3):
        phase = rng.uniform(-np.pi, np.pi)
        if family == "harmonic":
            f1 = rng.uniform(0.2, 0.9)
            f2 = rng.uniform(1.0, 3.2)
            signal = np.sin(2.0 * np.pi * f1 * time + phase) + 0.3 * np.sin(2.0 * np.pi * f2 * time - phase)
        elif family == "filtered_noise":
            white = rng.normal(size=time.size)
            alpha = rng.uniform(0.88, 0.985)
            innovation_scale = math.sqrt(max(1.0 - alpha * alpha, 1e-12))
            signal = np.empty_like(white)
            # Start the AR(1) process in its stationary distribution.  Using an
            # unscaled white-noise sample here while scaling later innovations by
            # (1 - alpha) gives the first samples a much larger variance and leaves
            # a conspicuous filter-warm-up transient after centering.
            signal[0] = white[0]
            for sample in range(1, time.size):
                signal[sample] = (
                    alpha * signal[sample - 1]
                    + innovation_scale * white[sample]
                )
        elif family == "chirp_modulated":
            start = rng.uniform(0.1, 0.7)
            end = rng.uniform(1.1, 4.2)
            duration = max(float(time[-1] - time[0]), 1e-12)
            local = time - time[0]
            chirp_phase = 2.0 * np.pi * (start * local + 0.5 * (end - start) * local * local / duration)
            envelope = 0.75 + 0.25 * np.sin(2.0 * np.pi * rng.uniform(0.05, 0.30) * local + phase)
            signal = envelope * np.sin(chirp_phase + phase)
        else:
            knot_count = int(rng.integers(6, 11))
            knots = np.linspace(0, time.size - 1, knot_count, dtype=int)
            values = rng.uniform(-1.0, 1.0, knot_count)
            signal = np.interp(np.arange(time.size), knots, values)
            for _ in range(int(rng.integers(2, 6))):
                center = rng.uniform(time[0], time[-1])
                width = rng.uniform(0.02, 0.12) * max(time[-1] - time[0], 1e-12)
                signal += rng.uniform(-0.8, 0.8) * np.exp(-0.5 * ((time - center) / width) ** 2)
        rows.append(float(amplitudes[index]) * _standardize(signal))
    return np.stack(rows)


def _controlled_family_source(
    family: str,
    time: np.ndarray,
    rng: np.random.Generator,
    frequency_hz: float,
) -> np.ndarray:
    """Generate one standardized family signal with a controlled characteristic frequency."""

    frequency = float(frequency_hz)
    phase = rng.uniform(-np.pi, np.pi)
    duration = max(float(time[-1] - time[0]), 1e-12)
    local = time - time[0]
    dt = max(float(np.median(np.diff(time))), 1e-12)
    if family == "harmonic":
        signal = np.sin(2.0 * np.pi * frequency * local + phase)
        signal += 0.3 * np.sin(2.0 * np.pi * 2.5 * frequency * local - phase)
    elif family == "filtered_noise":
        white = rng.normal(size=time.size)
        alpha = float(np.exp(-2.0 * np.pi * frequency * dt))
        innovation_scale = math.sqrt(max(1.0 - alpha * alpha, 1e-12))
        signal = np.empty_like(white)
        signal[0] = white[0]
        for sample in range(1, time.size):
            signal[sample] = alpha * signal[sample - 1] + innovation_scale * white[sample]
    elif family == "chirp_modulated":
        start = 0.5 * frequency
        end = 1.5 * frequency
        chirp_phase = 2.0 * np.pi * (
            start * local + 0.5 * (end - start) * local * local / duration
        )
        envelope_frequency = max(0.05, min(0.30, 0.15 * frequency))
        envelope = 0.75 + 0.25 * np.sin(
            2.0 * np.pi * envelope_frequency * local + phase
        )
        signal = envelope * np.sin(chirp_phase + phase)
    elif family == "piecewise_transient":
        knot_count = max(4, min(time.size, int(round(2.0 * frequency * duration)) + 2))
        knots = np.linspace(0, time.size - 1, knot_count, dtype=int)
        values = rng.uniform(-1.0, 1.0, knot_count)
        signal = np.interp(np.arange(time.size), knots, values)
        transient_count = max(1, int(round(0.5 * frequency * duration)))
        transient_width = min(
            0.12 * duration,
            max(2.0 * dt, 0.15 / max(frequency, 1e-12)),
        )
        for _ in range(transient_count):
            center = rng.uniform(time[0], time[-1])
            signal += rng.uniform(-0.8, 0.8) * np.exp(
                -0.5 * ((time - center) / transient_width) ** 2
            )
    else:
        raise ValueError(f"Unsupported family {family!r}.")
    return _standardize(signal)


def _disjoint_sparse_windows(
    sample_count: int,
    sparse_indices: tuple[int, ...],
    duty_cycles: tuple[float, ...],
    rng: np.random.Generator,
) -> dict[int, list[tuple[int, int, np.ndarray]]]:
    """Allocate smooth, mutually disjoint burst windows for sparse sources."""

    if not sparse_indices:
        return {}
    burst_count = 3
    events: list[tuple[int, int]] = []
    for source_index, duty_cycle in zip(sparse_indices, duty_cycles):
        support = max(burst_count * 3, int(round(float(duty_cycle) * int(sample_count))))
        support = min(support, int(sample_count) - 1)
        base, remainder = divmod(support, burst_count)
        lengths = [base + (1 if index < remainder else 0) for index in range(burst_count)]
        events.extend((source_index, max(3, length)) for length in lengths)
    rng.shuffle(events)
    occupied = sum(length for _, length in events)
    gap_count = len(events) + 1
    minimum_gap = 1
    available = int(sample_count) - occupied - gap_count * minimum_gap
    if available < 0:
        raise ValueError("Sparse duty cycles leave insufficient space for disjoint smooth bursts.")
    random_gaps = rng.multinomial(available, np.full(gap_count, 1.0 / gap_count))
    gaps = random_gaps + minimum_gap
    windows: dict[int, list[tuple[int, int, np.ndarray]]] = {
        index: [] for index in sparse_indices
    }
    cursor = int(gaps[0])
    for event_index, (source_index, length) in enumerate(events):
        start = cursor
        stop = start + int(length)
        taper = np.hanning(int(length) + 2)[1:-1]
        windows[source_index].append((start, stop, taper))
        cursor = stop + int(gaps[event_index + 1])
    return windows


def _orthogonalize_formal_sources(
    sources: np.ndarray,
    source_roles: tuple[str, str, str],
    requested_amplitudes_rad: tuple[float | None, float | None, float | None],
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Keep sparse support fixed while making every active source pair uncorrelated."""

    result = np.asarray(sources, dtype=float).copy()
    active_indices = [index for index, role in enumerate(source_roles) if role != "inactive"]
    sparse_indices = [index for index, role in enumerate(source_roles) if role == "sparse"]
    dense_indices = [index for index, role in enumerate(source_roles) if role == "dense"]
    basis: list[np.ndarray] = []
    for index in sparse_indices:
        vector = result[index]
        norm = float(np.linalg.norm(vector))
        if norm <= eps:
            raise ValueError("Sparse source generation produced a zero-variance signal.")
        basis.append(vector / norm)
    for index in dense_indices:
        vector = result[index] - float(np.mean(result[index]))
        for direction in basis:
            vector = vector - float(np.dot(vector, direction)) * direction
        vector = vector - float(np.mean(vector))
        norm = float(np.linalg.norm(vector))
        if norm <= eps:
            raise ValueError("Dense source orthogonalization produced a zero-variance signal.")
        result[index] = vector
        basis.append(vector / norm)
    for index in active_indices:
        target = float(requested_amplitudes_rad[index])
        peak = float(np.max(np.abs(result[index])))
        if peak <= eps:
            raise ValueError("Formal source generation produced a zero-amplitude signal.")
        result[index] *= target / peak
    for index, role in enumerate(source_roles):
        if role == "inactive":
            result[index] = 0.0
    return result


def _dominant_frequency_hz(signal: np.ndarray, dt: float) -> float:
    centered = np.asarray(signal, dtype=float) - float(np.mean(signal))
    spectrum = np.abs(np.fft.rfft(centered))
    frequencies = np.fft.rfftfreq(centered.size, d=float(dt))
    if spectrum.size <= 1:
        return 0.0
    spectrum[0] = 0.0
    return float(frequencies[int(np.argmax(spectrum))])


def _build_scenario(
    *,
    config: SimulationConfig,
    time: np.ndarray,
    sources: np.ndarray,
    rng: np.random.Generator,
    family: str,
    case_id: str,
    amplitude_tier: str,
    generator_label: str,
    stress_peak_angle_rad: float | None,
    source_roles: tuple[str, str, str] = ("dense", "dense", "dense"),
    composition: str = "triple_rotation",
    replicate_index: int | None = None,
    requested_amplitudes_rad: tuple[float | None, float | None, float | None] = (
        None,
        None,
        None,
    ),
    requested_frequencies_hz: tuple[float | None, float | None, float | None] = (
        None,
        None,
        None,
    ),
    sparse_duty_cycles: tuple[float | None, float | None, float | None] = (
        None,
        None,
        None,
    ),
    decorrelate_sources: bool = True,
    enforce_cross_correlation_limit: bool = True,
) -> ScenarioData:
    active_indices = tuple(
        index for index, role in enumerate(source_roles) if role != "inactive"
    )
    if not active_indices:
        raise ValueError("At least one source must be active.")
    if decorrelate_sources:
        sources = _decorrelate_source_angles(
            sources,
            cross_correlation_limit=config.source_cross_correlation_limit,
        )
    else:
        sources = np.asarray(sources, dtype=float)
        for index, role in enumerate(source_roles):
            if role == "inactive" and np.any(sources[index] != 0.0):
                raise RuntimeError("Inactive formal sources must be exactly zero.")
    generated_sources = _center_source_angles(sources, source_roles)
    correlation = source_cross_correlation_matrix(
        generated_sources[np.asarray(active_indices)]
    )
    if enforce_cross_correlation_limit and len(active_indices) > 1:
        pairwise = np.abs(correlation[np.triu_indices(len(active_indices), k=1)])
        if float(np.max(pairwise)) >= float(config.source_cross_correlation_limit):
            raise RuntimeError(
                "Centered active sources exceed the cross-correlation limit: "
                f"maximum={float(np.max(pairwise)):.6g}, "
                f"limit={float(config.source_cross_correlation_limit):.6g}."
            )
    observation_model = str(config.observation_model).strip().lower()
    source_axes = random_orthonormal_axes(3, rng)
    if observation_model == "body_fixed_gyro":
        observer_axes = random_orthonormal_axes(3, rng)
        mixing = source_axes @ observer_axes.T
    else:
        mixing = symmetric_orthogonal_mixing_matrix(rng)
        observer_axes = mixing @ source_axes
    relative_sources = generated_sources.copy()
    tolerance = 1e-10
    if orthonormality_error(source_axes) > tolerance:
        raise RuntimeError("Generated source axes are not orthonormal.")
    if orthonormality_error(observer_axes) > tolerance:
        raise RuntimeError("Generated observer axes are not orthonormal.")
    if float(np.linalg.norm(mixing @ mixing.T - np.eye(3), ord="fro")) > tolerance:
        raise RuntimeError("Generated rotational mixing matrix is not orthogonal.")
    if float(np.linalg.det(source_axes)) <= 0.0 or float(np.linalg.det(observer_axes)) <= 0.0:
        raise RuntimeError("Generated source and observer axes must be right-handed.")
    if float(np.linalg.det(mixing)) <= 0.0:
        raise RuntimeError("Generated rotational mixing matrix must be proper orthogonal.")
    if observation_model == "body_fixed_gyro":
        physical_object_quaternions = compose_source_quaternions(
            relative_sources,
            source_axes,
        )
        physical_observer_quaternions = change_quaternion_basis(
            physical_object_quaternions,
            observer_axes,
        )
        observer_increments = quaternions_to_body_rotation_increments(
            physical_observer_quaternions,
        )
        observations = np.cumsum(observer_increments, axis=1)
        dt = float(time[1] - time[0])
        observer_angular_velocities = observer_increments / dt
        source_coefficients = source_axes @ physical_object_quaternions[:, 1:].T
        observation_coefficients = physical_observer_quaternions[:, 1:].T
        scalar = physical_observer_quaternions[:, 0]
    else:
        scalar, source_coefficients = angles_to_quaternion_components(relative_sources)
        observation_coefficients = mixing @ source_coefficients
        observations, physical_observer_quaternions = decode_scalar_and_vector(
            scalar,
            observation_coefficients,
        )
        _, physical_object_quaternions = decode_scalar_and_vector(
            scalar,
            source_coefficients,
        )
        observer_increments = np.empty_like(observations)
        observer_increments[:, 0] = observations[:, 0]
        observer_increments[:, 1:] = np.diff(observations, axis=1)
        observer_angular_velocities = observer_increments / float(time[1] - time[0])
    source_axes_time = rotate_axis_triad(source_axes, physical_object_quaternions)
    observer_axes_time = rotate_axis_triad(observer_axes, physical_object_quaternions)
    realized_amplitudes = tuple(
        None if role == "inactive" else float(np.max(np.abs(generated_sources[index])))
        for index, role in enumerate(source_roles)
    )
    realized_frequencies = tuple(
        None
        if role == "inactive"
        else _dominant_frequency_hz(generated_sources[index], float(time[1] - time[0]))
        for index, role in enumerate(source_roles)
    )
    return ScenarioData(
        config=config,
        time_seconds=time,
        source_axes_global=source_axes,
        observer_axes_global=observer_axes,
        source_axes_observer_frame=normalize_rows(mixing),
        mixing_matrix=mixing,
        generated_sources=generated_sources,
        relative_sources=relative_sources,
        sources=relative_sources,
        observations=observations,
        physical_object_quaternions=physical_object_quaternions,
        physical_observer_quaternions=physical_observer_quaternions,
        observer_rotation_increments=observer_increments,
        observer_angular_velocities=observer_angular_velocities,
        source_axes_global_time=source_axes_time,
        observer_axes_global_time=observer_axes_time,
        observation_model=observation_model,
        shared_quaternion_scalar=scalar,
        source_coefficients=source_coefficients,
        observation_coefficients=observation_coefficients,
        family=family,
        case_id=case_id,
        amplitude_tier=amplitude_tier,
        generator_label=generator_label,
        stress_peak_angle_rad=stress_peak_angle_rad,
        composition=composition,
        replicate_index=replicate_index,
        source_roles=source_roles,
        requested_amplitudes_rad=requested_amplitudes_rad,
        realized_amplitudes_rad=realized_amplitudes,
        requested_frequencies_hz=requested_frequencies_hz,
        realized_frequencies_hz=realized_frequencies,
        sparse_duty_cycles=sparse_duty_cycles,
    )


def generate_validation_scenario(config: SimulationConfig | None = None) -> ScenarioData:
    active = config or SimulationConfig()
    rng = np.random.default_rng(int(active.seed))
    time = np.linspace(0.0, float(active.duration_seconds), int(active.T), endpoint=False)
    return _build_scenario(
        config=active,
        time=time,
        sources=_legacy_sources(active, rng, time),
        rng=rng,
        family="legacy",
        case_id="legacy_single",
        amplitude_tier="single",
        generator_label="legacy_modulated_harmonic",
        stress_peak_angle_rad=None,
    )


def generate_family_scenario(
    spec: SuiteCaseSpec,
    base_config: SimulationConfig | None = None,
    study_config: BatchStudyConfig | None = None,
) -> ScenarioData:
    base = base_config or SimulationConfig()
    study = study_config or BatchStudyConfig()
    local = SimulationConfig(
        seed=spec.seed,
        T=spec.frame_count,
        duration_seconds=spec.frame_count * spec.dt,
        source_cross_correlation_limit=base.source_cross_correlation_limit,
        observation_model=base.observation_model,
    )
    rng = np.random.default_rng(int(spec.seed))
    time = np.arange(int(spec.frame_count), dtype=float) * float(spec.dt)
    low, high = _amplitude_range(spec.amplitude_tier, study)
    amplitudes = rng.uniform(low, high, size=3)
    if spec.stress_peak_angle_rad is not None:
        amplitudes *= float(spec.stress_peak_angle_rad) / max(float(np.max(amplitudes)), 1e-12)
    sources = _family_sources(spec.family, time, rng, amplitudes)
    if spec.stress_peak_angle_rad is not None:
        sources *= float(spec.stress_peak_angle_rad) / max(float(np.max(np.abs(sources))), 1e-12)
    return _build_scenario(
        config=local,
        time=time,
        sources=sources,
        rng=rng,
        family=spec.family,
        case_id=spec.case_id,
        amplitude_tier=spec.amplitude_tier,
        generator_label=spec.family,
        stress_peak_angle_rad=spec.stress_peak_angle_rad,
    )


def create_standard_case_specs(
    config: BatchStudyConfig | None = None,
    *,
    base_seed: int = 1000,
    case_id_prefix: str = "",
) -> list[SuiteCaseSpec]:
    """Create deterministic three-source family cases.

    ``base_seed`` and ``case_id_prefix`` allow a separate cohort to reuse the
    standard family design without overlapping the standard 20-case suite.
    """

    active = config or BatchStudyConfig()
    if int(active.standard_cases_per_family) > 101:
        raise ValueError(
            "standard_cases_per_family must be <= 101 to preserve unique family seeds."
        )
    specs: list[SuiteCaseSpec] = []
    for family_index, family in enumerate(active.standard_families):
        for case_index in range(int(active.standard_cases_per_family)):
            specs.append(
                SuiteCaseSpec(
                    family=family,
                    case_id=f"{case_id_prefix}{family}_{case_index + 1:02d}",
                    seed=int(base_seed) + 101 * family_index + case_index,
                    frame_count=active.batch_sequence_lengths[case_index % len(active.batch_sequence_lengths)],
                    dt=active.dt,
                    amplitude_tier=active.standard_amplitude_tiers[case_index % len(active.standard_amplitude_tiers)],
                )
            )
    return specs


def create_stress_case_specs(config: BatchStudyConfig | None = None) -> list[SuiteCaseSpec]:
    active = config or BatchStudyConfig()
    peaks = np.linspace(active.stress_min_peak_rad, active.stress_max_peak_rad, active.stress_case_count)
    return [
        SuiteCaseSpec(
            family="harmonic",
            case_id=f"harmonic_stress_{index + 1:02d}",
            seed=7300 + 31 * index,
            frame_count=active.batch_sequence_lengths[index % len(active.batch_sequence_lengths)],
            dt=active.dt,
            amplitude_tier="high",
            stress_peak_angle_rad=float(peak),
        )
        for index, peak in enumerate(peaks)
    ]


@dataclass(frozen=True)
class CorrelationStudyConfig:
    """Design of the inter-source correlation sensitivity sweep.

    Every correlation level within one ``(family, replicate)`` block reuses the same
    seed, so the raw waveforms, source axes, observer axes, and mixing matrix are
    identical across levels and the target correlation is the only factor that varies.
    """

    target_correlations: tuple[float, ...] = (
        0.0,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
        1e-1,
        3e-1,
        0.9,
    )
    families: tuple[str, ...] = (
        "harmonic",
        "filtered_noise",
        "chirp_modulated",
        "piecewise_transient",
    )
    replicates: int = 10
    dt: float = 0.01
    base_seed: int = 31000
    case_id_prefix: str = "corr_"
    maximum_abs_correlation: float = 0.95
    correlation_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        allowed = {"harmonic", "filtered_noise", "chirp_modulated", "piecewise_transient"}
        if not self.families:
            raise ValueError("families must contain at least one signal family.")
        unknown = set(self.families) - allowed
        if unknown:
            raise ValueError(f"Unsupported families: {sorted(unknown)}.")
        if len(set(self.families)) != len(self.families):
            raise ValueError("families must not contain duplicates.")
        if int(self.replicates) < 1:
            raise ValueError("replicates must be >= 1.")
        if not math.isfinite(float(self.dt)) or float(self.dt) <= 0.0:
            raise ValueError("dt must be finite and > 0.")
        limit = float(self.maximum_abs_correlation)
        if not math.isfinite(limit) or not 0.0 < limit < 1.0:
            raise ValueError("maximum_abs_correlation must be finite and strictly between 0 and 1.")
        tolerance = float(self.correlation_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("correlation_tolerance must be finite and > 0.")
        if not self.target_correlations:
            raise ValueError("target_correlations must contain at least one level.")
        values = tuple(float(value) for value in self.target_correlations)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("target_correlations must be finite.")
        if len(set(values)) != len(values):
            raise ValueError("target_correlations must not contain duplicates.")
        if any(abs(value) > limit for value in values):
            raise ValueError(
                f"target_correlations must satisfy |rho| <= maximum_abs_correlation ({limit:.6g})."
            )
        # Three equicorrelated sources stay positive definite only above -0.5.
        if any(value <= -0.5 for value in values):
            raise ValueError("target_correlations must be greater than -0.5 for three sources.")
        object.__setattr__(self, "target_correlations", values)
        object.__setattr__(self, "families", tuple(str(family) for family in self.families))


@dataclass(frozen=True)
class CorrelationCaseSpec:
    """One correlation-sensitivity case: a block, at one target correlation."""

    family: str
    case_id: str
    seed: int
    frame_count: int
    dt: float
    amplitude_tier: str
    target_source_correlation: float
    replicate_index: int
    block_id: str

    def __post_init__(self) -> None:
        allowed = {"harmonic", "filtered_noise", "chirp_modulated", "piecewise_transient"}
        if self.family not in allowed:
            raise ValueError(f"Unsupported family {self.family!r}.")
        if int(self.frame_count) < 8 or float(self.dt) <= 0.0:
            raise ValueError("frame_count must be >= 8 and dt must be positive.")
        if self.amplitude_tier not in {"low", "medium", "high"}:
            raise ValueError("amplitude_tier must be 'low', 'medium', or 'high'.")
        if int(self.replicate_index) < 1:
            raise ValueError("replicate_index must be >= 1.")
        value = float(self.target_source_correlation)
        if not math.isfinite(value) or not -0.5 < value < 1.0:
            raise ValueError("target_source_correlation must be finite and in (-0.5, 1).")


def _correlation_case_tag(correlation: float) -> str:
    """Return a filesystem-safe tag encoding one signed correlation level.

    Significant digits are kept rather than a fixed number of decimals, so levels far
    below one part per million still produce distinct case identifiers.
    """

    value = float(correlation)
    sign = "m" if value < 0.0 else "p"
    magnitude = f"{abs(value):.10g}".replace(".", "_").replace("-", "m").replace("+", "")
    return f"rho{sign}{magnitude}"


def create_correlation_case_specs(
    config: CorrelationStudyConfig | None = None,
    study_config: BatchStudyConfig | None = None,
) -> list[CorrelationCaseSpec]:
    """Create the paired correlation-sensitivity cases, blocked by family and replicate.

    Amplitude tier and sequence length are cycled over replicates so they stay balanced
    nuisance factors; because the analysis is within-block they do not inflate the
    correlation contrast.
    """

    active = config or CorrelationStudyConfig()
    study = study_config or BatchStudyConfig()
    tiers = ("low", "medium", "high")
    specs: list[CorrelationCaseSpec] = []
    for family_index, family in enumerate(active.families):
        for replicate in range(1, int(active.replicates) + 1):
            seed = int(active.base_seed) + 101 * family_index + (replicate - 1)
            block_id = f"{active.case_id_prefix}{family}_{replicate:02d}"
            frame_count = study.batch_sequence_lengths[
                (replicate - 1) % len(study.batch_sequence_lengths)
            ]
            amplitude_tier = tiers[(replicate - 1) % len(tiers)]
            for correlation in active.target_correlations:
                specs.append(
                    CorrelationCaseSpec(
                        family=family,
                        case_id=f"{block_id}_{_correlation_case_tag(correlation)}",
                        seed=seed,
                        frame_count=int(frame_count),
                        dt=float(active.dt),
                        amplitude_tier=amplitude_tier,
                        target_source_correlation=float(correlation),
                        replicate_index=replicate,
                        block_id=block_id,
                    )
                )
    identifiers = [spec.case_id for spec in specs]
    if len(set(identifiers)) != len(identifiers):
        # Case ids become artifact directory names and the join key used to attach the
        # target correlation back onto the metrics frames, so a collision would silently
        # mislabel results rather than fail.
        raise ValueError("Correlation case identifiers must be unique; correlation levels are too close.")
    return specs


def generate_correlation_scenario(
    spec: CorrelationCaseSpec,
    base_config: SimulationConfig | None = None,
    study_config: BatchStudyConfig | None = None,
    *,
    correlation_tolerance: float = 1e-9,
) -> ScenarioData:
    """Generate one three-source scenario whose sources have an exact target correlation.

    The usual ``source_cross_correlation_limit`` gate is deliberately bypassed here: this
    suite exists to study correlations above that limit, and the realized correlation is
    instead verified against the requested target.
    """

    base = base_config or SimulationConfig()
    study = study_config or BatchStudyConfig()
    local = SimulationConfig(
        seed=spec.seed,
        T=spec.frame_count,
        duration_seconds=spec.frame_count * spec.dt,
        source_cross_correlation_limit=base.source_cross_correlation_limit,
        observation_model=base.observation_model,
    )
    rng = np.random.default_rng(int(spec.seed))
    time = np.arange(int(spec.frame_count), dtype=float) * float(spec.dt)
    low, high = _amplitude_range(spec.amplitude_tier, study)
    amplitudes = rng.uniform(low, high, size=3)
    sources = _family_sources(spec.family, time, rng, amplitudes)
    sources = correlate_source_angles(
        sources,
        float(spec.target_source_correlation),
        tolerance=float(correlation_tolerance),
    )
    return _build_scenario(
        config=local,
        time=time,
        sources=sources,
        rng=rng,
        family=spec.family,
        case_id=spec.case_id,
        amplitude_tier=spec.amplitude_tier,
        generator_label=f"correlation_{spec.family}",
        stress_peak_angle_rad=None,
        replicate_index=spec.replicate_index,
        decorrelate_sources=False,
        enforce_cross_correlation_limit=False,
    )


def create_formal_case_specs(
    config: FormalStudyConfig | None = None,
) -> list[FormalCaseSpec]:
    """Create the deterministic 6-composition × 4-family × 5-case formal suite."""

    active = config or FormalStudyConfig()
    amplitude_low, amplitude_high = map(float, active.amplitude_range_rad)
    frequency_low, frequency_high = map(float, active.frequency_range_hz)
    amplitude_width = (amplitude_high - amplitude_low) / int(active.cases_per_family)
    frequency_width = (frequency_high - frequency_low) / int(active.cases_per_family)
    specs: list[FormalCaseSpec] = []
    for composition_index, (composition, roles) in enumerate(FORMAL_SOURCE_COMPOSITIONS):
        for family_index, family in enumerate(active.families):
            for case_index in range(int(active.cases_per_family)):
                seed = (
                    int(active.base_seed)
                    + 1009 * composition_index
                    + 101 * family_index
                    + case_index
                )
                rng = np.random.default_rng(seed)
                amplitude_bounds = (
                    amplitude_low + case_index * amplitude_width,
                    amplitude_low + (case_index + 1) * amplitude_width,
                )
                frequency_bounds = (
                    frequency_low + case_index * frequency_width,
                    frequency_low + (case_index + 1) * frequency_width,
                )
                amplitudes: list[float | None] = []
                frequencies: list[float | None] = []
                duty_cycles: list[float | None] = []
                for role in roles:
                    if role == "inactive":
                        amplitudes.append(None)
                        frequencies.append(None)
                        duty_cycles.append(None)
                        continue
                    amplitudes.append(float(rng.uniform(*amplitude_bounds)))
                    frequencies.append(float(rng.uniform(*frequency_bounds)))
                    duty_cycles.append(
                        float(rng.uniform(*active.sparse_duty_cycle_range))
                        if role == "sparse"
                        else None
                    )
                specs.append(
                    FormalCaseSpec(
                        family=family,
                        composition=composition,
                        case_id=f"{composition}_{family}_{case_index + 1:02d}",
                        replicate_index=case_index + 1,
                        seed=seed,
                        frame_count=int(
                            active.batch_sequence_lengths[
                                case_index % len(active.batch_sequence_lengths)
                            ]
                        ),
                        dt=float(active.dt),
                        source_roles=roles,
                        requested_amplitudes_rad=tuple(amplitudes),
                        requested_frequencies_hz=tuple(frequencies),
                        requested_sparse_duty_cycles=tuple(duty_cycles),
                    )
                )
    return specs


def generate_formal_scenario(
    spec: FormalCaseSpec,
    base_config: SimulationConfig | None = None,
) -> ScenarioData:
    """Generate a dense/sparse/inactive formal validation scenario."""

    base = base_config or SimulationConfig()
    local = SimulationConfig(
        seed=spec.seed,
        T=spec.frame_count,
        duration_seconds=spec.frame_count * spec.dt,
        source_cross_correlation_limit=base.source_cross_correlation_limit,
        observation_model=base.observation_model,
    )
    rng = np.random.default_rng(int(spec.seed))
    time = np.arange(int(spec.frame_count), dtype=float) * float(spec.dt)
    raw_sources = np.zeros((3, int(spec.frame_count)), dtype=float)
    for index, role in enumerate(spec.source_roles):
        if role == "inactive":
            continue
        raw_sources[index] = _controlled_family_source(
            spec.family,
            time,
            rng,
            float(spec.requested_frequencies_hz[index]),
        )
    sparse_indices = tuple(
        index for index, role in enumerate(spec.source_roles) if role == "sparse"
    )
    sparse_duties = tuple(
        float(spec.requested_sparse_duty_cycles[index]) for index in sparse_indices
    )
    sparse_windows = _disjoint_sparse_windows(
        int(spec.frame_count),
        sparse_indices,
        sparse_duties,
        rng,
    )
    realized_duties: list[float | None] = [None, None, None]
    for index in sparse_indices:
        sparse_signal = np.zeros(int(spec.frame_count), dtype=float)
        for start, stop, taper in sparse_windows[index]:
            segment = raw_sources[index, start:stop]
            weighted_mean = float(np.sum(taper * segment) / np.sum(taper))
            sparse_signal[start:stop] = taper * (segment - weighted_mean)
        raw_sources[index] = sparse_signal
        realized_duties[index] = float(np.count_nonzero(sparse_signal)) / float(
            spec.frame_count
        )
    sources = _orthogonalize_formal_sources(
        raw_sources,
        spec.source_roles,
        spec.requested_amplitudes_rad,
    )
    return _build_scenario(
        config=local,
        time=time,
        sources=sources,
        rng=rng,
        family=spec.family,
        case_id=spec.case_id,
        amplitude_tier=f"stratum_{spec.replicate_index:02d}",
        generator_label=f"formal_{spec.family}",
        stress_peak_angle_rad=None,
        source_roles=spec.source_roles,
        composition=spec.composition,
        replicate_index=spec.replicate_index,
        requested_amplitudes_rad=spec.requested_amplitudes_rad,
        requested_frequencies_hz=spec.requested_frequencies_hz,
        sparse_duty_cycles=tuple(realized_duties),
        decorrelate_sources=False,
    )
