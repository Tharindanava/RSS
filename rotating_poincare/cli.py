"""Command-line interface for the Rotating Poincare Section Transform."""

from __future__ import annotations

import argparse
from pathlib import Path

from rotating_poincare.core import RPTConfig
from rotating_poincare.pipeline import analyze_csv_files, generate_demo_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze angle-signal phase portraits with rotating Poincare "
            "section crossing histograms."
        )
    )
    parser.add_argument("--input", nargs="+", help="One or more input CSV files.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate and analyze deterministic clean-cycle and noisy-ellipse examples.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output root. Defaults to outputs/rotating_poincare for normal runs "
            "and results/rpt_demo with --demo."
        ),
    )
    parser.add_argument("--time-col", default=None, help="Time column name; auto-detected when omitted.")
    parser.add_argument(
        "--signal-cols",
        nargs="+",
        default=None,
        help="Signal columns to analyze; numeric non-time columns are inferred when omitted.",
    )
    parser.add_argument("--fs", type=float, default=None, help="Sampling frequency in Hz when no time column exists.")
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Map each phase-portrait coordinate to [-1, 1] before the transform (default: enabled).",
    )
    parser.add_argument(
        "--derivative-smoothing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply Savitzky-Golay smoothing before differentiation (default: enabled).",
    )
    parser.add_argument("--savgol-window", type=int, default=21)
    parser.add_argument("--savgol-polyorder", type=int, default=3)
    parser.add_argument("--interp-points", type=int, default=5000)
    parser.add_argument(
        "--interpolation-method",
        choices=["pchip", "cubic"],
        default="pchip",
    )
    parser.add_argument("--theta-min-deg", type=float, default=0.0)
    parser.add_argument("--theta-max-deg", type=float, default=360.0)
    parser.add_argument("--theta-step-deg", type=float, default=1.0)
    parser.add_argument("--num-radial-bins", type=int, default=80)
    parser.add_argument("--radial-min", type=float, default=None)
    parser.add_argument("--radial-max", type=float, default=None)
    parser.add_argument(
        "--crossing-direction",
        choices=["both", "positive", "negative"],
        default="both",
    )
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--no-plots", action="store_true", help="Write CSV/JSON artifacts only.")
    return parser


def config_from_args(args: argparse.Namespace) -> RPTConfig:
    return RPTConfig(
        normalize=bool(args.normalize),
        derivative_smoothing=bool(args.derivative_smoothing),
        savgol_window=int(args.savgol_window),
        savgol_polyorder=int(args.savgol_polyorder),
        interp_points=int(args.interp_points),
        interpolation_method=args.interpolation_method,
        theta_min_deg=float(args.theta_min_deg),
        theta_max_deg=float(args.theta_max_deg),
        theta_step_deg=float(args.theta_step_deg),
        num_radial_bins=int(args.num_radial_bins),
        radial_min=args.radial_min,
        radial_max=args.radial_max,
        crossing_direction=args.crossing_direction,
        eps=float(args.eps),
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.demo and args.input:
        parser.error("--demo and --input cannot be used together.")
    if not args.demo and not args.input:
        parser.error("provide --input FILE [FILE ...] or use --demo.")

    output_directory = Path(
        args.output_dir
        if args.output_dir is not None
        else ("results/rpt_demo" if args.demo else "outputs/rotating_poincare")
    )
    config = config_from_args(args)
    if args.demo:
        input_paths = list(generate_demo_inputs(output_directory))
        time_col = "time"
        signal_cols = None
    else:
        input_paths = list(args.input)
        time_col = args.time_col
        signal_cols = args.signal_cols

    artifacts = analyze_csv_files(
        input_paths,
        output_directory,
        config,
        time_col=time_col,
        signal_cols=signal_cols,
        fs=args.fs,
        no_plots=bool(args.no_plots),
    )
    print(f"Wrote {len(artifacts)} signal analysis result(s) to {output_directory.resolve()}")
    for artifact in artifacts:
        print(f"- {artifact.input_file.name} :: {artifact.signal_column} -> {artifact.output_directory}")
        for warning in artifact.warnings:
            print(f"  warning: {warning}")


if __name__ == "__main__":
    main()
