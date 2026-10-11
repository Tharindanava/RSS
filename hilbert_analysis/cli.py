"""Command-line interface for the Hilbert-based gait-data cleaning batch."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hilbert_analysis.completeness import CompletenessConfig
from hilbert_analysis.phase_flip import PhaseFlipConfig
from hilbert_analysis.pipeline import (
    DEFAULT_BACKUP_ROOT,
    DEFAULT_INPUT_ROOT,
    REPORT_ROOT,
    BatchConfig,
    restore_from_source,
    run_batch,
)


def mode_list(value: str) -> tuple[int, ...]:
    try:
        modes = tuple(int(part) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected comma-separated modes, got {value!r}.") from exc
    if not modes or any(mode not in (1, 2, 3) for mode in modes):
        raise argparse.ArgumentTypeError(f"Modes must be drawn from 1, 2, 3; got {value!r}.")
    return modes


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    default_completeness = CompletenessConfig()
    default_flip = PhaseFlipConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Detect truncated recordings, correct inverted mode signals from the "
            "left/right pair relationship, and label transient/periodic/cycle "
            "structure across an RSS-pipeline output tree."
        )
    )
    parser.add_argument(
        "--input-root", type=Path, default=DEFAULT_INPUT_ROOT,
        help="Root searched recursively for limb_angle_mode.csv files.",
    )
    parser.add_argument(
        "--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT,
        help="Root that receives a one-time backup copy of each leaf folder before it is overwritten.",
    )
    parser.add_argument(
        "--report-root", type=Path, default=REPORT_ROOT,
        help="Root that receives the summary/detail/error report CSVs and diagnostic plots.",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually move incomplete folders and rewrite CSVs. Default is a dry-run report only.",
    )
    parser.add_argument(
        "--no-backup", dest="make_backup", action="store_false",
        help="Skip the pre-overwrite backup copy (not recommended).",
    )
    parser.add_argument(
        "--threshold-multiplier", type=float, default=default_completeness.threshold_multiplier,
        help="Whole-body envelope must exceed this multiple of the baseline to count as active.",
    )
    parser.add_argument(
        "--min-end-margin-s", type=float, default=default_completeness.min_end_margin_s,
        help="Minimum quiet time required after the last active run for a file to be 'complete'.",
    )
    parser.add_argument(
        "--chunk-window-s", type=float, default=default_completeness.chunk_window_s,
        help="Non-overlapping window length used for the short-time Hilbert envelope.",
    )
    parser.add_argument("--baseline-window-s", type=float, default=default_completeness.baseline_window_s)
    parser.add_argument("--min-sustained-s", type=float, default=default_completeness.min_sustained_s)
    parser.add_argument(
        "--min-correlation", type=float, default=default_flip.min_abs_correlation,
        help=(
            "Minimum |correlation| between left and right at the half-stride lag for a "
            "pair to be judged. Below this the pair is reported low_confidence and left "
            "alone. Default: 0.30."
        ),
    )
    parser.add_argument(
        "--lag-search-fraction", type=float, default=default_flip.lag_search_fraction,
        help="Lags are searched within this fraction of a stride either side of T/2.",
    )
    parser.add_argument(
        "--reference-side", choices=("left", "right"), default=default_flip.reference_side,
        help=(
            "The side held fixed. A disagreeing pair has exactly one inverted signal, but "
            "the pair relationship cannot say which, so the other side is the one negated."
        ),
    )
    parser.add_argument(
        "--min-peak-to-peak", type=float, default=default_flip.min_peak_to_peak,
        help="Pairs with a side flatter than this are treated as degenerate and left alone.",
    )
    parser.add_argument(
        "--modes", type=mode_list, default=",".join(str(m) for m in default_flip.modes),
        help="Comma-separated modes to correct, each judged independently. Default: 1.",
    )
    parser.add_argument(
        "--legacy-pair-flip", dest="flip_signals", action="store_true", default=False,
        help="Explicitly enable the legacy Hilbert left/right auto-flip writer.",
    )
    parser.add_argument(
        "--no-flip", dest="flip_signals", action="store_false", default=False,
        help="Deprecated compatibility flag; pair consistency is report-only by default.",
    )
    parser.add_argument(
        "--no-full-record-fallback", dest="allow_full_record_fallback", action="store_false",
        help="Skip flip detection entirely when no periodic segment was found.",
    )
    parser.add_argument(
        "--restore-source", type=Path, default=None,
        help=(
            "Restore pristine leaf folders from this root over --input-root instead of "
            "running the cleaning batch (a pristine copy of the tree, e.g. outputs/rss_pristine). "
            "Dry-run unless --apply is also given."
        ),
    )
    parser.add_argument(
        "--diagnostic-plots", action="store_true",
        help="Write one whole-body-envelope diagnostic PNG per processed file.",
    )
    parser.add_argument(
        "--diagnostic-limit", type=positive_int, default=None,
        help="Cap the number of diagnostic PNGs written (default: unlimited).",
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO",
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
        input_root=args.input_root,
        backup_root=args.backup_root,
        report_root=args.report_root,
        completeness=CompletenessConfig(
            threshold_multiplier=args.threshold_multiplier,
            min_end_margin_s=args.min_end_margin_s,
            chunk_window_s=args.chunk_window_s,
            baseline_window_s=args.baseline_window_s,
            min_sustained_s=args.min_sustained_s,
        ),
        phase_flip=PhaseFlipConfig(
            min_abs_correlation=args.min_correlation,
            lag_search_fraction=args.lag_search_fraction,
            min_peak_to_peak=args.min_peak_to_peak,
            modes=args.modes,
            reference_side=args.reference_side,
            allow_full_record_fallback=args.allow_full_record_fallback,
        ),
        apply=args.apply,
        flip_signals=args.flip_signals,
        make_backup=args.make_backup,
        diagnostic_plots=args.diagnostic_plots,
        diagnostic_limit=args.diagnostic_limit,
    )

    logger = logging.getLogger("hilbert_analysis")

    if args.restore_source is not None:
        try:
            restored = restore_from_source(
                source_root=args.restore_source,
                input_root=args.input_root,
                apply=args.apply,
            )
        except (FileNotFoundError, NotADirectoryError) as exc:
            logger.error("%s", exc)
            return 2
        logger.info(
            "Restore %s: %s leaves in source, %s matched, %s differing, %s restored.",
            "APPLIED" if args.apply else "dry run (pass --apply to write)",
            restored.leaves_in_source, restored.leaves_matched,
            restored.leaves_differing, restored.leaves_restored,
        )
        return 0

    try:
        result = run_batch(config)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        logger.error("%s", exc)
        return 2

    logger.info(
        "Files discovered: %s | processed: %s | incomplete: %s (moved: %s) | "
        "too short: %s | pairs flipped: %s",
        result.files_discovered, result.files_processed, result.files_incomplete,
        result.files_moved, result.files_too_short, result.pairs_flipped,
    )
    logger.info("Summary CSV: %s", result.summary_csv)
    logger.info("Phase-flip details CSV: %s", result.flip_details_csv)
    logger.info("Errors CSV: %s", result.errors_csv)
    return 0
