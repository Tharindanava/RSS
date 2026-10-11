"""Command-line interface for limb-wise transfer entropy."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from limb_transfer_entropy.core import ContextSpec
from limb_transfer_entropy.pipeline import BatchConfig, run_batch


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}.")
    return parsed


def bins_int(value: str) -> int:
    parsed = positive_int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("The number of bins must be at least 2.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively compute fixed-context raw transfer-entropy matrices "
            "for left and right lower-limb mode-1 signals."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs") / "rss",
        help="Root searched recursively for limb_angle_mode.csv files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / "limb_transfer_entropy",
        help="Separate root that receives the mirrored result tree.",
    )
    parser.add_argument(
        "--bins",
        type=bins_int,
        default=8,
        help="Requested number of discrete PDF bins. Default: 8.",
    )
    parser.add_argument(
        "--binning",
        choices=("quantile", "uniform"),
        default="quantile",
        help="Signal discretization method. Default: quantile.",
    )
    parser.add_argument(
        "--context-length",
        type=positive_int,
        default=1,
        help="Number of lagged samples in both source and receiver histories. Default: 1.",
    )
    parser.add_argument(
        "--lag",
        type=positive_int,
        default=1,
        help="Lag from the present sample to the newest context sample. Default: 1.",
    )
    parser.add_argument(
        "--stride",
        type=positive_int,
        default=1,
        help="Spacing between consecutive samples in each context. Default: 1.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console logging verbosity. Default: INFO.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    config = BatchConfig(
        input_root=args.input_root.resolve(),
        output_root=args.output_root.resolve(),
        bins=int(args.bins),
        binning=args.binning,
        context=ContextSpec(
            lag=int(args.lag),
            length=int(args.context_length),
            stride=int(args.stride),
        ),
    )
    try:
        result = run_batch(config)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        logging.getLogger("limb_transfer_entropy").error("%s", exc)
        return 2

    logging.getLogger("limb_transfer_entropy").info(
        "Files discovered: %s | successful sides: %s | failed sides: %s",
        result.files_discovered,
        result.sides_succeeded,
        result.sides_failed,
    )
    logging.getLogger("limb_transfer_entropy").info("Summary CSV: %s", result.summary_csv)
    logging.getLogger("limb_transfer_entropy").info("Errors CSV: %s", result.errors_csv)
    return 0 if result.sides_succeeded > 0 else 1
