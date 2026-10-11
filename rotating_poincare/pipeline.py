"""End-to-end CSV analysis and artifact export for the RPT package."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rotating_poincare.core import (
    RPTConfig,
    RPTResult,
    center_and_normalize_phase_points,
    compute_derivative,
    compute_rotating_poincare_transform,
    compute_rpt_features,
    interpolate_trajectory,
    make_phase_points,
)
from rotating_poincare.io import LoadedSignalFile, load_signal_file
from rotating_poincare.plotting import (
    plot_phase_portrait,
    plot_rpt_features,
    plot_rpt_heatmap,
    plot_rpt_surface3d,
    save_plotly_surface_html,
)


MIN_VALID_SAMPLES = 5


@dataclass(frozen=True)
class SignalAnalysisArtifacts:
    """Paths and transform data produced for one signal column."""

    input_file: Path
    signal_column: str
    output_directory: Path
    transform: RPTResult
    features: pd.DataFrame
    warnings: tuple[str, ...]


def slugify(value: str) -> str:
    """Convert a column or file name into a portable artifact name."""

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._")
    return slug or "signal"


def _input_directory_names(paths: Iterable[Path]) -> dict[Path, str]:
    resolved = [path.resolve() for path in paths]
    stems: dict[str, list[Path]] = {}
    for path in resolved:
        stems.setdefault(slugify(path.stem), []).append(path)

    names: dict[Path, str] = {}
    for stem, members in stems.items():
        if len(members) == 1:
            names[members[0]] = stem
            continue
        for path in members:
            digest = hashlib.sha1(str(path).casefold().encode("utf-8")).hexdigest()[:8]
            names[path] = f"{stem}_{digest}"
    return names


def _clean_signal(
    loaded: LoadedSignalFile,
    signal_column: str,
    fs: float | None,
) -> tuple[np.ndarray, np.ndarray | None, str, list[str], int]:
    signal = pd.to_numeric(loaded.dataframe[signal_column], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(signal)
    if loaded.time_values is not None:
        mask &= np.isfinite(loaded.time_values)
    dropped = int(mask.size - np.count_nonzero(mask))
    clean_signal = signal[mask]
    if clean_signal.size < MIN_VALID_SAMPLES:
        raise ValueError(
            f"Signal {signal_column!r} in {loaded.path} has {clean_signal.size} valid samples; "
            f"at least {MIN_VALID_SAMPLES} are required."
        )

    messages: list[str] = []
    if dropped:
        messages.append(f"Dropped {dropped} row(s) with non-finite signal or time values.")

    if loaded.time_values is not None:
        clean_time = np.asarray(loaded.time_values[mask], dtype=float)
        if np.any(np.diff(clean_time) <= 0.0):
            raise ValueError(
                f"Time values for signal {signal_column!r} in {loaded.path} "
                "must be strictly increasing after invalid rows are removed."
            )
        return clean_signal, clean_time, f"time_column:{loaded.time_column}", messages, dropped

    if fs is not None:
        sample_rate = float(fs)
        if not np.isfinite(sample_rate) or sample_rate <= 0.0:
            raise ValueError("fs must be a finite value > 0.")
        full_time = np.arange(signal.size, dtype=float) / sample_rate
        return clean_signal, full_time[mask], "sampling_frequency", messages, dropped

    full_time = np.arange(signal.size, dtype=float)
    messages.append(
        "No time column or sampling frequency was available; unit sample spacing was used."
    )
    return clean_signal, full_time[mask], "unit_sample_spacing", messages, dropped


def _counts_dataframe(result: RPTResult) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {"theta_deg": result.theta_values_deg}
    for bin_index in range(result.count_matrix.shape[1]):
        data[f"bin_{bin_index:03d}"] = result.count_matrix[:, bin_index]
    return pd.DataFrame(data)


def _save_signal_outputs(
    *,
    loaded: LoadedSignalFile,
    signal_column: str,
    output_directory: Path,
    config: RPTConfig,
    phase_points: np.ndarray,
    interpolated_points: np.ndarray,
    transform: RPTResult,
    features: pd.DataFrame,
    mean: np.ndarray,
    scale: np.ndarray,
    time_source: str,
    valid_samples: int,
    dropped_rows: int,
    warnings: list[str],
    no_plots: bool,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    signal_slug = slugify(signal_column)
    _counts_dataframe(transform).to_csv(
        output_directory / f"{signal_slug}_rpt_counts.csv",
        index=False,
    )
    features.to_csv(output_directory / f"{signal_slug}_rpt_features.csv", index=False)

    metadata = {
        "input_file": str(loaded.path),
        "signal_column": signal_column,
        "time_column": loaded.time_column,
        "time_source": time_source,
        "valid_samples": int(valid_samples),
        "dropped_rows": int(dropped_rows),
        "normalize": config.normalize,
        "interp_points": config.interp_points,
        "interpolation_method": config.interpolation_method,
        "theta_min_deg": config.theta_min_deg,
        "theta_max_deg": config.theta_max_deg,
        "theta_step_deg": config.theta_step_deg,
        "theta_endpoint_included": False,
        "num_radial_bins": config.num_radial_bins,
        "radial_min": float(transform.radial_bin_edges[0]),
        "radial_max": float(transform.radial_bin_edges[-1]),
        "requested_radial_min": config.radial_min,
        "requested_radial_max": config.radial_max,
        "radial_bin_edges": transform.radial_bin_edges.tolist(),
        "radial_bin_centers": transform.radial_bin_centers.tolist(),
        "crossing_direction": config.crossing_direction,
        "derivative_method": "numpy.gradient",
        "derivative_smoothing": config.derivative_smoothing,
        "savgol_window": config.savgol_window,
        "savgol_polyorder": config.savgol_polyorder,
        "phase_preprocessing": (
            "centered_and_unit_range_scaled"
            if config.normalize
            else "centered_only"
        ),
        "phase_center": mean.tolist(),
        "phase_scale": scale.tolist(),
        "eps": config.eps,
        "warnings": warnings,
    }
    with (output_directory / f"{signal_slug}_rpt_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    if no_plots:
        return

    plot_specs = (
        (
            plot_phase_portrait(phase_points, interpolated_points, signal_column),
            f"{signal_slug}_phase_portrait.png",
        ),
        (plot_rpt_heatmap(transform, signal_column), f"{signal_slug}_rpt_heatmap.png"),
        (plot_rpt_surface3d(transform, signal_column), f"{signal_slug}_rpt_surface3d.png"),
        (plot_rpt_features(features, signal_column), f"{signal_slug}_rpt_features.png"),
    )
    for figure, filename in plot_specs:
        figure.savefig(output_directory / filename, dpi=180, bbox_inches="tight")
        plt.close(figure)
    save_plotly_surface_html(
        transform,
        signal_column,
        output_directory / f"{signal_slug}_rpt_surface3d.html",
    )


def analyze_loaded_file(
    loaded: LoadedSignalFile,
    output_directory: str | Path,
    config: RPTConfig,
    *,
    fs: float | None = None,
    no_plots: bool = False,
) -> list[SignalAnalysisArtifacts]:
    """Analyze every selected signal in one loaded CSV."""

    file_output = Path(output_directory)
    artifacts: list[SignalAnalysisArtifacts] = []
    used_slugs: dict[str, str] = {}
    for signal_column in loaded.signal_columns:
        signal_slug = slugify(signal_column)
        previous = used_slugs.get(signal_slug.casefold())
        if previous is not None and previous != signal_column:
            raise ValueError(
                f"Signal columns {previous!r} and {signal_column!r} map to the same "
                f"artifact name {signal_slug!r}."
            )
        used_slugs[signal_slug.casefold()] = signal_column

        signal, time, time_source, messages, dropped = _clean_signal(
            loaded,
            signal_column,
            fs,
        )
        derivative = compute_derivative(
            signal,
            t=time,
            smooth=config.derivative_smoothing,
            savgol_window=config.savgol_window,
            savgol_polyorder=config.savgol_polyorder,
            warnings=messages,
        )
        raw_phase = make_phase_points(signal, derivative)
        normalized = center_and_normalize_phase_points(
            raw_phase,
            normalize=config.normalize,
            eps=config.eps,
        )
        messages.extend(normalized.warnings)
        interpolated = interpolate_trajectory(
            normalized.points,
            interp_points=config.interp_points,
            method=config.interpolation_method,
            eps=config.eps,
        )
        if float(np.max(np.linalg.norm(interpolated - interpolated[0], axis=1))) <= config.eps:
            messages.append(
                "The centered phase trajectory has near-zero arc length; "
                "the transform contains no crossings."
            )
        transform = compute_rotating_poincare_transform(
            interpolated,
            theta_min_deg=config.theta_min_deg,
            theta_max_deg=config.theta_max_deg,
            theta_step_deg=config.theta_step_deg,
            num_radial_bins=config.num_radial_bins,
            radial_min=config.radial_min,
            radial_max=config.radial_max,
            crossing_direction=config.crossing_direction,
            eps=config.eps,
        )
        features = compute_rpt_features(
            transform.count_matrix,
            transform.theta_values_deg,
            transform.radial_bin_centers,
            eps=config.eps,
        )
        signal_output = file_output / signal_slug
        _save_signal_outputs(
            loaded=loaded,
            signal_column=signal_column,
            output_directory=signal_output,
            config=config,
            phase_points=normalized.points,
            interpolated_points=interpolated,
            transform=transform,
            features=features,
            mean=normalized.mean,
            scale=normalized.scale,
            time_source=time_source,
            valid_samples=signal.size,
            dropped_rows=dropped,
            warnings=messages,
            no_plots=no_plots,
        )
        artifacts.append(
            SignalAnalysisArtifacts(
                input_file=loaded.path,
                signal_column=signal_column,
                output_directory=signal_output,
                transform=transform,
                features=features,
                warnings=tuple(messages),
            )
        )
    return artifacts


def analyze_csv_files(
    input_paths: list[str | Path] | tuple[str | Path, ...],
    output_directory: str | Path,
    config: RPTConfig,
    *,
    time_col: str | None = None,
    signal_cols: list[str] | tuple[str, ...] | None = None,
    fs: float | None = None,
    no_plots: bool = False,
) -> list[SignalAnalysisArtifacts]:
    """Run batch RPT analysis over one or more CSV files."""

    if not input_paths:
        raise ValueError("At least one input CSV is required.")
    loaded_files = [
        load_signal_file(path, time_col=time_col, signal_cols=signal_cols)
        for path in input_paths
    ]
    names = _input_directory_names([loaded.path for loaded in loaded_files])
    output_root = Path(output_directory).expanduser().resolve()

    all_artifacts: list[SignalAnalysisArtifacts] = []
    for loaded in loaded_files:
        file_output = output_root / names[loaded.path]
        all_artifacts.extend(
            analyze_loaded_file(
                loaded,
                file_output,
                config,
                fs=fs,
                no_plots=no_plots,
            )
        )
    return all_artifacts


def generate_demo_inputs(output_directory: str | Path) -> tuple[Path, Path]:
    """Create deterministic clean-cycle and noisy-ellipse CSV inputs."""

    output_root = Path(output_directory).expanduser().resolve()
    input_directory = output_root / "input_data"
    input_directory.mkdir(parents=True, exist_ok=True)

    samples = 2400
    time = np.linspace(0.0, 16.0 * np.pi, samples, endpoint=False)
    clean = np.cos(time + 0.173)
    rng = np.random.default_rng(2026)
    noisy = (
        1.5 * np.cos(time + 0.173)
        + 0.16 * np.cos(2.7 * time + 0.4)
        + rng.normal(scale=0.10, size=samples)
    )
    clean_path = input_directory / "clean_limit_cycle.csv"
    noisy_path = input_directory / "noisy_ellipse.csv"
    pd.DataFrame({"time": time, "clean_cycle": clean}).to_csv(clean_path, index=False)
    pd.DataFrame({"time": time, "noisy_ellipse": noisy}).to_csv(noisy_path, index=False)
    return clean_path, noisy_path
