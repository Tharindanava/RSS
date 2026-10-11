"""Export per-test reconstruction and pre-RSS signal plots from cached fits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qarm_rss_experiments.practical_validation import (
    DISTAL_GYRO_COLUMNS,
    JOINT_NAMES,
    TRUTH_COLUMNS,
    match_and_score,
    prepare_rss_input,
    ssa_observations,
)


DEFAULT_FORMATS = ("png", "pdf", "svg")
AXIS_NAMES = ("x", "y", "z")


def _run_sample_rate_hz(root: Path) -> float:
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        return 100.0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return float(manifest.get("config", {}).get("sample_rate_hz", 100.0))


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def build_observed_signal_frame(
    frame: pd.DataFrame,
    gyro_bias_deg_s: Sequence[float],
    *,
    sample_rate_hz: float,
    include_ssa_filtered: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Reconstruct the exact headline RSS input and the legacy SSA sensitivity.

    The headline input is bias-corrected but otherwise unfiltered. SSA is
    applied to the integrated channels only for the labelled legacy
    sensitivity configuration.
    """

    raw_gyro = frame.loc[:, DISTAL_GYRO_COLUMNS].to_numpy(dtype=float)
    bias = np.asarray(gyro_bias_deg_s, dtype=float)
    if bias.shape != (3,) or not np.isfinite(bias).all():
        raise ValueError("gyro_bias_deg_s must contain three finite values.")
    corrected_deg_s = raw_gyro - bias[None, :]
    corrected_rad_s = np.radians(corrected_deg_s)
    observations = prepare_rss_input(frame, bias, sample_rate_hz=sample_rate_hz)
    filtered = (
        ssa_observations(observations, sample_rate_hz).T
        if include_ssa_filtered
        else np.full_like(observations.T, np.nan)
    )
    if "time_s" in frame:
        times = frame["time_s"].to_numpy(dtype=float)
    else:
        times = np.arange(len(frame), dtype=float) / sample_rate_hz

    exported = pd.DataFrame({"time_s": times})
    for index, axis_name in enumerate(AXIS_NAMES):
        exported[f"node4_raw_gyro_{axis_name}_deg_s"] = raw_gyro[:, index]
        exported[f"node4_bias_corrected_gyro_{axis_name}_deg_s"] = corrected_deg_s[:, index]
        exported[f"node4_bias_corrected_gyro_{axis_name}_rad_s"] = corrected_rad_s[:, index]
        exported[f"rss_observed_integrated_body_rotation_{axis_name}_rad"] = observations[index]
        if include_ssa_filtered:
            exported[f"legacy_ssa_filtered_integrated_{axis_name}_rad"] = filtered[:, index]
    return exported, observations


def _plot_observed_signals(
    signals: pd.DataFrame,
    *,
    test_number: int,
    status: str,
    include_ssa_filtered: bool,
) -> plt.Figure:
    times = signals["time_s"].to_numpy(dtype=float)
    colors = ("#0072B2", "#D55E00", "#009E73")
    figure, axes = plt.subplots(3, 2, figsize=(15.0, 9.0), sharex=True)
    for index, (axis_name, color) in enumerate(zip(AXIS_NAMES, colors)):
        rate_axis, integrated_axis = axes[index]
        rate_axis.plot(
            times,
            signals[f"node4_bias_corrected_gyro_{axis_name}_deg_s"],
            color=color,
            linewidth=1.1,
        )
        rate_axis.set_ylabel(f"{axis_name.upper()} (deg/s)")
        rate_axis.grid(True, alpha=0.22, linewidth=0.7)
        integrated_axis.plot(
            times,
            signals[f"rss_observed_integrated_body_rotation_{axis_name}_rad"],
            color=color,
            linewidth=1.35,
            label="Headline RSS input (unfiltered)",
        )
        if include_ssa_filtered:
            integrated_axis.plot(
                times,
                signals[f"legacy_ssa_filtered_integrated_{axis_name}_rad"],
                color="#202020",
                linewidth=1.0,
                alpha=0.85,
                linestyle="--",
                label="Legacy SSA-filtered sensitivity",
            )
        integrated_axis.set_ylabel(f"{axis_name.upper()} (rad)")
        integrated_axis.grid(True, alpha=0.22, linewidth=0.7)
    axes[0, 0].set_title("Node 4 bias-corrected gyro rate (before integration)", loc="left", fontsize=11)
    axes[0, 1].set_title("Integrated body-rotation observer channels", loc="left", fontsize=11)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[0, 1].legend(loc="upper right", frameon=False)
    figure.suptitle(
        f"QArm test {test_number}: signals prepared before RSS ({status})",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "The headline RSS fit used the bias-corrected, otherwise unfiltered integrated channels. "
        "SSA-filtered channels belong only to the legacy sensitivity analysis.",
        ha="center",
        fontsize=9,
        color="#4A4A4A",
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.955))
    return figure


def export_per_test_reconstruction_plots(
    results_root: str | Path,
    *,
    formats: Sequence[str] = DEFAULT_FORMATS,
    include_ssa_filtered: bool = True,
) -> list[Path]:
    """Export aligned results and reconstructed pre-RSS signals for every test."""

    root = Path(results_root)
    quality_path = root / "data_quality_manifest.csv"
    if not quality_path.is_file():
        raise FileNotFoundError(f"Missing completed-run quality manifest: {quality_path}")
    requested_formats = tuple(str(value).lower().lstrip(".") for value in formats)
    unsupported = sorted(set(requested_formats) - set(DEFAULT_FORMATS))
    if not requested_formats or unsupported:
        raise ValueError(f"formats must contain png, pdf, and/or svg; unsupported: {unsupported}")

    # The default C-engine float parser is not bit-exact, which breaks the
    # cache's exact-bytes input-hash check on the re-read gyro bias.
    quality = pd.read_csv(quality_path, float_precision="round_trip").sort_values("test_number")
    sample_rate_hz = _run_sample_rate_hz(root)
    output_root = root / "figures" / "per_test"
    output_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    index_rows: list[dict[str, object]] = []

    for quality_row in quality.itertuples(index=False):
        number = int(quality_row.test_number)
        merged_path = Path(str(quality_row.merged_path))
        fit_path = root / "fits" / f"test_{number}" / "rss.npz"
        if not merged_path.is_file():
            raise FileNotFoundError(f"Missing merged trial data for test {number}: {merged_path}")
        if not fit_path.is_file():
            raise FileNotFoundError(f"Missing completed RSS fit for test {number}: {fit_path}")

        frame = pd.read_csv(merged_path, float_precision="round_trip")
        with np.load(fit_path, allow_pickle=False) as cached:
            sources = np.asarray(cached["sources"], dtype=float)
            cached_input_hash = str(cached["input_hash"].item()) if "input_hash" in cached else ""
        truth = frame.loc[:, TRUTH_COLUMNS].to_numpy(dtype=float).T
        metrics, aligned = match_and_score(
            truth,
            sources,
            test_number=number,
            batch_number=int(quality_row.batch_number),
            method="rss",
        )
        if "time_s" in frame:
            times = frame["time_s"].to_numpy(dtype=float)
        else:
            times = np.arange(aligned.shape[1], dtype=float) / 100.0

        test_dir = output_root / f"test_{number:02d}"
        test_dir.mkdir(parents=True, exist_ok=True)
        signal_frame = pd.DataFrame({"time_s": times})
        for joint_index, joint in enumerate(JOINT_NAMES):
            signal_frame[f"measured_{joint}_rad"] = truth[joint_index]
            signal_frame[f"rss_{joint}_aligned_rad"] = aligned[joint_index]
            signal_frame[f"error_{joint}_rad"] = aligned[joint_index] - truth[joint_index]
        signal_path = test_dir / "source_separated_vs_ground_truth.csv"
        signal_frame.to_csv(signal_path, index=False)
        written.append(signal_path)

        eligible = bool(quality_row.quality_pass)
        status = "quality-eligible" if eligible else f"excluded: {quality_row.exclusion_reasons}"

        bias = np.asarray(
            [
                quality_row.gyro_bias_x_deg_s,
                quality_row.gyro_bias_y_deg_s,
                quality_row.gyro_bias_z_deg_s,
            ],
            dtype=float,
        )
        observed_frame, reconstructed_observations = build_observed_signal_frame(
            frame,
            bias,
            sample_rate_hz=sample_rate_hz,
            include_ssa_filtered=include_ssa_filtered,
        )
        reconstructed_hash = _array_sha256(reconstructed_observations)
        hash_verified = not cached_input_hash or reconstructed_hash == cached_input_hash
        if not hash_verified:
            raise ValueError(
                f"Reconstructed RSS input for test {number} does not match the cached fit "
                f"({reconstructed_hash} != {cached_input_hash})."
            )
        observed_path = test_dir / "rss_observed_signals.csv"
        observed_frame.to_csv(observed_path, index=False)
        written.append(observed_path)

        observed_figure = _plot_observed_signals(
            observed_frame,
            test_number=number,
            status=status,
            include_ssa_filtered=include_ssa_filtered,
        )
        observed_plot_paths: list[str] = []
        for extension in requested_formats:
            observed_plot_path = test_dir / f"test_{number:02d}_rss_observed_signals.{extension}"
            observed_figure.savefig(
                observed_plot_path,
                dpi=200 if extension == "png" else None,
                bbox_inches="tight",
            )
            written.append(observed_plot_path)
            observed_plot_paths.append(str(observed_plot_path.resolve()))
        plt.close(observed_figure)

        figure, axes = plt.subplots(3, 1, figsize=(13.0, 9.0), sharex=True)
        colors = ("#0072B2", "#D55E00", "#009E73")
        for joint_index, (axis, joint, color, metric) in enumerate(zip(axes, JOINT_NAMES, colors, metrics)):
            axis.plot(times, truth[joint_index], color="#202020", linewidth=1.65, label="Measured encoder")
            axis.plot(times, aligned[joint_index], color=color, linewidth=1.25, alpha=0.92, label=f"Aligned RSS mode {metric['mode_number']}")
            axis.set_title(
                f"{joint.title()} ↔ RSS mode {metric['mode_number']}   |   "
                f"|r| = {metric['abs_correlation']:.3f}   |   "
                f"RMSE = {metric['rmse_rad']:.3f} rad ({metric['rmse_deg']:.2f}°)",
                fontsize=11,
                loc="left",
            )
            axis.set_ylabel("Angle (rad)")
            axis.grid(True, alpha=0.22, linewidth=0.7)
            axis.legend(loc="upper right", frameon=False, ncol=2)
        axes[-1].set_xlabel("Time (s)")
        figure.suptitle(
            f"QArm test {number}: measured and RSS-separated sources ({status})",
            fontsize=14,
            fontweight="bold",
        )
        figure.text(
            0.5,
            0.012,
            "One-to-one matching uses measured encoders only after blind RSS fitting. "
            "Sign and constant offset are aligned; amplitude and time lag are not fitted.",
            ha="center",
            fontsize=9,
            color="#4A4A4A",
        )
        figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.955))

        plot_paths: list[str] = []
        for extension in requested_formats:
            plot_path = test_dir / f"test_{number:02d}_measured_vs_rss.{extension}"
            figure.savefig(plot_path, dpi=200 if extension == "png" else None, bbox_inches="tight")
            written.append(plot_path)
            plot_paths.append(str(plot_path.resolve()))
        plt.close(figure)
        index_rows.append(
            {
                "test_number": number,
                "quality_pass": eligible,
                "exclusion_reasons": "" if eligible else str(quality_row.exclusion_reasons),
                "signal_csv": str(signal_path.resolve()),
                "plot_files": ";".join(plot_paths),
                "observed_signal_csv": str(observed_path.resolve()),
                "observed_signal_plot_files": ";".join(observed_plot_paths),
                "rss_input_sha256": reconstructed_hash,
                "rss_input_hash_verified": hash_verified,
            }
        )
        print(f"Exported test {number} ({len(index_rows)}/{len(quality)}).", flush=True)

    index_path = output_root / "per_test_plot_index.csv"
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    written.append(index_path)
    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("outputs/qarm_rss_validation"))
    parser.add_argument("--formats", nargs="+", default=list(DEFAULT_FORMATS), choices=DEFAULT_FORMATS)
    parser.add_argument(
        "--skip-ssa-filtered",
        action="store_true",
        help="Omit the legacy SSA-filtered integrated channels.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    written = export_per_test_reconstruction_plots(
        args.results_root.resolve(),
        formats=args.formats,
        include_ssa_filtered=not args.skip_ssa_filtered,
    )
    print(f"Created {len(written)} files under {args.results_root / 'figures' / 'per_test'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
