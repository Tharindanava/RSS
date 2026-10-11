"""Build, persist, and validate the deterministic QArm RSS signal bank."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rss import (
    BatchStudyConfig,
    SimulationConfig,
    create_standard_case_specs,
    generate_family_scenario,
    generate_validation_scenario,
    source_cross_correlation_matrix,
)


SAMPLE_RATE_HZ = 100.0
DURATION_SECONDS = 60.0
SAMPLE_COUNT = int(SAMPLE_RATE_HZ * DURATION_SECONDS) + 1
DESIGN_SPEED_LIMIT_DEG_S = 19.5
MAX_ALLOWED_SPEED_DEG_S = 20.0
SOURCE_CROSS_CORRELATION_LIMIT = 0.01

JOINT_NAMES = ("shoulder", "elbow", "wrist")
CENTER_POSE_DEG = np.array([0.0, -10.0, 0.0], dtype=np.float64)
SAFE_POSITION_LIMITS_DEG = np.array(
    [[-75.0, 75.0], [-85.0, 65.0], [-150.0, 150.0]],
    dtype=np.float64,
)

PACKAGE_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = PACKAGE_DIRECTORY / "data"
DEFAULT_BANK_PATH = DATA_DIRECTORY / "qarm_rss_signals.npz"
DEFAULT_MANIFEST_PATH = DATA_DIRECTORY / "signal_manifest.csv"

EXPECTED_FAMILY_COUNTS = {
    "legacy": 17,
    "harmonic": 17,
    "piecewise_transient": 16,
}

MANIFEST_FIELDS = (
    "signal_number",
    "family",
    "family_case_number",
    "seed",
    "native_case_id",
    "generator_label",
    "amplitude_tier",
    "native_frame_count",
    "native_dt_seconds",
    "native_duration_seconds",
    "stretch_factor",
    "scale_factor",
    "source_max_abs_correlation",
    "shoulder_target_min_deg",
    "shoulder_target_max_deg",
    "elbow_target_min_deg",
    "elbow_target_max_deg",
    "wrist_target_min_deg",
    "wrist_target_max_deg",
    "shoulder_peak_velocity_deg_s",
    "elbow_peak_velocity_deg_s",
    "wrist_peak_velocity_deg_s",
)


@dataclass(frozen=True)
class _NativeCase:
    signal_number: int
    family: str
    family_case_number: int
    seed: int
    native_case_id: str
    generator_label: str
    amplitude_tier: str
    native_frame_count: int
    native_dt_seconds: float
    native_duration_seconds: float
    sources: np.ndarray


@dataclass(frozen=True)
class SelectedSignal:
    """One validated signal and its human-readable manifest metadata."""

    signal_number: int
    family: str
    family_case_number: int
    seed: int
    time_seconds: np.ndarray
    source_angles_rad: np.ndarray
    joint_targets_rad: np.ndarray
    metadata: Mapping[str, str]


def _native_cases() -> list[_NativeCase]:
    cases: list[_NativeCase] = []
    signal_number = 1

    for family_case_number, seed in enumerate(range(21000, 21017), start=1):
        config = SimulationConfig(
            seed=seed,
            T=2000,
            duration_seconds=1.0,
            source_cross_correlation_limit=SOURCE_CROSS_CORRELATION_LIMIT,
        )
        scenario = generate_validation_scenario(config)
        cases.append(
            _NativeCase(
                signal_number=signal_number,
                family="legacy",
                family_case_number=family_case_number,
                seed=seed,
                native_case_id=f"legacy_{family_case_number:02d}",
                generator_label=scenario.generator_label,
                amplitude_tier="single",
                native_frame_count=config.T,
                native_dt_seconds=config.duration_seconds / config.T,
                native_duration_seconds=config.duration_seconds,
                sources=np.asarray(scenario.generated_sources, dtype=np.float64),
            )
        )
        signal_number += 1

    study_config = BatchStudyConfig(standard_cases_per_family=50)
    formal_specs = create_standard_case_specs(
        study_config,
        base_seed=20000,
        case_id_prefix="formal_",
    )
    base_config = SimulationConfig(
        source_cross_correlation_limit=SOURCE_CROSS_CORRELATION_LIMIT,
    )
    family_case_counts = {"harmonic": 17, "piecewise_transient": 16}
    for family, count in family_case_counts.items():
        family_specs = [spec for spec in formal_specs if spec.family == family][:count]
        for family_case_number, spec in enumerate(family_specs, start=1):
            scenario = generate_family_scenario(spec, base_config, study_config)
            cases.append(
                _NativeCase(
                    signal_number=signal_number,
                    family=family,
                    family_case_number=family_case_number,
                    seed=spec.seed,
                    native_case_id=spec.case_id,
                    generator_label=scenario.generator_label,
                    amplitude_tier=spec.amplitude_tier,
                    native_frame_count=spec.frame_count,
                    native_dt_seconds=spec.dt,
                    native_duration_seconds=spec.frame_count * spec.dt,
                    sources=np.asarray(scenario.generated_sources, dtype=np.float64),
                )
            )
            signal_number += 1

    return cases


def _resample_and_scale(native: _NativeCase, time_seconds: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    native_phase = np.linspace(0.0, 1.0, native.sources.shape[1], dtype=np.float64)
    playback_phase = time_seconds / DURATION_SECONDS
    source_angles = np.stack(
        [np.interp(playback_phase, native_phase, row) for row in native.sources],
        axis=1,
    )
    source_angles -= np.mean(source_angles, axis=0, keepdims=True)

    correlation = source_cross_correlation_matrix(source_angles.T)
    pairwise = np.abs(correlation[np.triu_indices(len(JOINT_NAMES), k=1)])
    maximum_correlation = float(np.max(pairwise))
    if maximum_correlation >= SOURCE_CROSS_CORRELATION_LIMIT:
        raise RuntimeError(
            f"Signal {native.signal_number} exceeds the source correlation limit after resampling: "
            f"{maximum_correlation:.6g}."
        )

    center_rad = np.radians(CENTER_POSE_DEG)
    limits_rad = np.radians(SAFE_POSITION_LIMITS_DEG)
    scale_candidates = [1.0]
    for joint_index in range(len(JOINT_NAMES)):
        minimum = float(np.min(source_angles[:, joint_index]))
        maximum = float(np.max(source_angles[:, joint_index]))
        if minimum < 0.0:
            scale_candidates.append(
                float((limits_rad[joint_index, 0] - center_rad[joint_index]) / minimum)
            )
        if maximum > 0.0:
            scale_candidates.append(
                float((limits_rad[joint_index, 1] - center_rad[joint_index]) / maximum)
            )

    raw_velocity = np.diff(source_angles, axis=0) * SAMPLE_RATE_HZ
    raw_peak_velocity = float(np.max(np.abs(raw_velocity)))
    if raw_peak_velocity > 0.0:
        scale_candidates.append(math.radians(DESIGN_SPEED_LIMIT_DEG_S) / raw_peak_velocity)

    scale_factor = min(scale_candidates)
    if not math.isfinite(scale_factor) or scale_factor <= 0.0:
        raise RuntimeError(f"Signal {native.signal_number} produced an invalid scale factor.")
    scale_factor = min(1.0, float(scale_factor))

    source_angles *= scale_factor
    joint_targets = source_angles + center_rad
    return source_angles, joint_targets, scale_factor, maximum_correlation


def build_signal_bank() -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Generate the complete deterministic bank in memory."""

    native_cases = _native_cases()
    time_seconds = np.linspace(0.0, DURATION_SECONDS, SAMPLE_COUNT, dtype=np.float64)
    source_bank = np.empty((len(native_cases), SAMPLE_COUNT, len(JOINT_NAMES)), dtype=np.float64)
    target_bank = np.empty_like(source_bank)
    manifest_rows: list[dict[str, Any]] = []

    for bank_index, native in enumerate(native_cases):
        sources, targets, scale_factor, maximum_correlation = _resample_and_scale(native, time_seconds)
        source_bank[bank_index] = sources
        target_bank[bank_index] = targets
        targets_deg = np.degrees(targets)
        velocities_deg_s = np.degrees(np.diff(targets, axis=0) * SAMPLE_RATE_HZ)
        manifest_rows.append(
            {
                "signal_number": native.signal_number,
                "family": native.family,
                "family_case_number": native.family_case_number,
                "seed": native.seed,
                "native_case_id": native.native_case_id,
                "generator_label": native.generator_label,
                "amplitude_tier": native.amplitude_tier,
                "native_frame_count": native.native_frame_count,
                "native_dt_seconds": native.native_dt_seconds,
                "native_duration_seconds": native.native_duration_seconds,
                "stretch_factor": DURATION_SECONDS / native.native_duration_seconds,
                "scale_factor": scale_factor,
                "source_max_abs_correlation": maximum_correlation,
                "shoulder_target_min_deg": float(np.min(targets_deg[:, 0])),
                "shoulder_target_max_deg": float(np.max(targets_deg[:, 0])),
                "elbow_target_min_deg": float(np.min(targets_deg[:, 1])),
                "elbow_target_max_deg": float(np.max(targets_deg[:, 1])),
                "wrist_target_min_deg": float(np.min(targets_deg[:, 2])),
                "wrist_target_max_deg": float(np.max(targets_deg[:, 2])),
                "shoulder_peak_velocity_deg_s": float(np.max(np.abs(velocities_deg_s[:, 0]))),
                "elbow_peak_velocity_deg_s": float(np.max(np.abs(velocities_deg_s[:, 1]))),
                "wrist_peak_velocity_deg_s": float(np.max(np.abs(velocities_deg_s[:, 2]))),
            }
        )

    bank = {
        "time_seconds": time_seconds,
        "source_angles_rad": source_bank,
        "joint_targets_rad": target_bank,
        "signal_numbers": np.array([case.signal_number for case in native_cases], dtype=np.int32),
        "families": np.array([case.family for case in native_cases], dtype="<U32"),
        "family_case_numbers": np.array([case.family_case_number for case in native_cases], dtype=np.int16),
        "seeds": np.array([case.seed for case in native_cases], dtype=np.int64),
        "native_case_ids": np.array([case.native_case_id for case in native_cases], dtype="<U64"),
        "joint_names": np.array(JOINT_NAMES, dtype="<U16"),
        "sample_rate_hz": np.array(SAMPLE_RATE_HZ, dtype=np.float64),
        "duration_seconds": np.array(DURATION_SECONDS, dtype=np.float64),
        "center_pose_rad": np.radians(CENTER_POSE_DEG),
        "safe_position_limits_rad": np.radians(SAFE_POSITION_LIMITS_DEG),
        "design_speed_limit_rad_s": np.array(math.radians(DESIGN_SPEED_LIMIT_DEG_S), dtype=np.float64),
    }
    validate_signal_bank(bank)
    _validate_manifest(bank, manifest_rows)
    return bank, manifest_rows


def validate_signal_bank(bank: Mapping[str, np.ndarray]) -> None:
    """Reject malformed, unsafe, or inconsistent bank data."""

    required = {
        "time_seconds",
        "source_angles_rad",
        "joint_targets_rad",
        "signal_numbers",
        "families",
        "family_case_numbers",
        "seeds",
        "native_case_ids",
        "joint_names",
        "sample_rate_hz",
        "duration_seconds",
        "center_pose_rad",
        "safe_position_limits_rad",
        "design_speed_limit_rad_s",
    }
    missing = sorted(required - set(bank))
    if missing:
        raise ValueError(f"Signal bank is missing arrays: {missing}.")

    time_seconds = np.asarray(bank["time_seconds"], dtype=np.float64)
    sources = np.asarray(bank["source_angles_rad"], dtype=np.float64)
    targets = np.asarray(bank["joint_targets_rad"], dtype=np.float64)
    signal_numbers = np.asarray(bank["signal_numbers"], dtype=np.int64)
    families = np.asarray(bank["families"]).astype(str)
    seeds = np.asarray(bank["seeds"], dtype=np.int64)
    joint_names = tuple(np.asarray(bank["joint_names"]).astype(str).tolist())
    sample_rate = float(np.asarray(bank["sample_rate_hz"]).item())
    duration = float(np.asarray(bank["duration_seconds"]).item())
    center = np.asarray(bank["center_pose_rad"], dtype=np.float64)
    limits = np.asarray(bank["safe_position_limits_rad"], dtype=np.float64)

    expected_numbers = np.arange(1, 51, dtype=np.int64)
    if not np.array_equal(signal_numbers, expected_numbers):
        raise ValueError("Signal numbers must be the consecutive integers 1 through 50.")
    if len(np.unique(seeds)) != 50:
        raise ValueError("All signal seeds must be unique.")
    family_counts = {family: int(np.count_nonzero(families == family)) for family in np.unique(families)}
    if family_counts != EXPECTED_FAMILY_COUNTS:
        raise ValueError(f"Unexpected signal-family allocation: {family_counts}.")
    if joint_names != JOINT_NAMES:
        raise ValueError(f"Joint order must be {JOINT_NAMES}, received {joint_names}.")
    if not math.isclose(sample_rate, SAMPLE_RATE_HZ, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Sample rate must be {SAMPLE_RATE_HZ} Hz.")
    if not math.isclose(duration, DURATION_SECONDS, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Duration must be {DURATION_SECONDS} seconds.")
    if time_seconds.shape != (SAMPLE_COUNT,) or sources.shape != (50, SAMPLE_COUNT, 3) or targets.shape != sources.shape:
        raise ValueError("Signal-bank arrays have unexpected shapes.")
    expected_time = np.linspace(0.0, DURATION_SECONDS, SAMPLE_COUNT, dtype=np.float64)
    if not np.allclose(time_seconds, expected_time, rtol=0.0, atol=1e-12):
        raise ValueError("Signal-bank time values are not the required 100 Hz, 60-second grid.")
    if center.shape != (3,) or limits.shape != (3, 2):
        raise ValueError("Center pose or safe-position limits have an invalid shape.")
    if not all(np.isfinite(array).all() for array in (time_seconds, sources, targets, center, limits)):
        raise ValueError("Signal bank contains non-finite numeric values.")
    if not np.allclose(np.mean(sources, axis=1), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("Every practical source trajectory must have zero temporal mean.")
    if not np.allclose(targets, sources + center[None, None, :], rtol=0.0, atol=1e-12):
        raise ValueError("Joint targets do not equal source angles plus the center pose.")

    tolerance = 1e-12
    if np.any(targets < limits[None, :, 0] - tolerance) or np.any(targets > limits[None, :, 1] + tolerance):
        raise ValueError("At least one joint target exceeds the conservative position limits.")
    velocities = np.diff(targets, axis=1) * sample_rate
    if float(np.max(np.abs(velocities))) > math.radians(MAX_ALLOWED_SPEED_DEG_S) + tolerance:
        raise ValueError("At least one joint trajectory exceeds 20 deg/s.")
    for signal_index in range(sources.shape[0]):
        correlation = source_cross_correlation_matrix(sources[signal_index].T)
        pairwise = np.abs(correlation[np.triu_indices(3, k=1)])
        if float(np.max(pairwise)) >= SOURCE_CROSS_CORRELATION_LIMIT:
            raise ValueError(f"Signal {signal_index + 1} exceeds the source correlation limit.")


def _validate_manifest(bank: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 50:
        raise ValueError("The manifest must contain exactly 50 rows.")
    for index, row in enumerate(rows):
        number = int(row["signal_number"])
        if number != index + 1:
            raise ValueError("Manifest rows must be ordered by signal number.")
        if str(row["family"]) != str(np.asarray(bank["families"])[index]):
            raise ValueError(f"Manifest family mismatch for signal {number}.")
        if int(row["seed"]) != int(np.asarray(bank["seeds"])[index]):
            raise ValueError(f"Manifest seed mismatch for signal {number}.")


def save_signal_bank(
    bank_path: Path = DEFAULT_BANK_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> tuple[Path, Path]:
    """Regenerate and write the checked signal bank and manifest."""

    bank, rows = build_signal_bank()
    bank_path = Path(bank_path)
    manifest_path = Path(manifest_path)
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(bank_path, **bank)
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return bank_path, manifest_path


def load_signal_bank(bank_path: Path = DEFAULT_BANK_PATH) -> dict[str, np.ndarray]:
    bank_path = Path(bank_path)
    with np.load(bank_path, allow_pickle=False) as archive:
        bank = {name: archive[name] for name in archive.files}
    validate_signal_bank(bank)
    return bank


def load_manifest(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> list[dict[str, str]]:
    with Path(manifest_path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return rows


def load_selected_signal(
    signal_number: int,
    bank_path: Path = DEFAULT_BANK_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> SelectedSignal:
    """Load one signal after validating both persisted artifacts."""

    if isinstance(signal_number, bool) or not 1 <= int(signal_number) <= 50:
        raise ValueError("signal_number must be an integer from 1 through 50.")
    bank = load_signal_bank(bank_path)
    rows = load_manifest(manifest_path)
    _validate_manifest(bank, rows)
    index = int(signal_number) - 1
    row = rows[index]
    return SelectedSignal(
        signal_number=int(signal_number),
        family=str(bank["families"][index]),
        family_case_number=int(bank["family_case_numbers"][index]),
        seed=int(bank["seeds"][index]),
        time_seconds=np.array(bank["time_seconds"], copy=True),
        source_angles_rad=np.array(bank["source_angles_rad"][index], copy=True),
        joint_targets_rad=np.array(bank["joint_targets_rad"][index], copy=True),
        metadata=dict(row),
    )
