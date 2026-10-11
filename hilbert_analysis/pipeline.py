"""Recursive batch driver: completeness detection, left/right phase-flip
correction, and segment/cycle marker labeling over an RSS-pipeline output
tree. Mirrors the structure of ``limb_transfer_entropy/pipeline.py``
(dataclass config/result, per-file isolated errors, summary + error CSVs)."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from hilbert_analysis.completeness import (
    VERDICT_INCOMPLETE,
    CompletenessConfig,
    CompletenessResult,
    detect_completeness,
)
from hilbert_analysis.columns import PAIRED_LIMBS
from hilbert_analysis.discovery import (
    AXES_CSV_NAME,
    INCOMPLETE_SUBFOLDER_NAME,
    MODE_CSV_NAME,
    SIBLING_MODE_FILES,
    discover_leaf_folders,
)
from hilbert_analysis.phase_flip import (
    STATUS_CONSISTENT,
    STATUS_DEGENERATE,
    STATUS_FLIPPED,
    STATUS_LOW_CONFIDENCE,
    STATUS_NO_STRIDE,
    PhaseFlipConfig,
    apply_axis_flips,
    apply_flips,
    evaluate_frame,
)
from hilbert_analysis.plotting import write_completeness_plot
from hilbert_analysis.segmentation import (
    build_markers,
    build_whole_body_phase_reference,
    stride_period_samples,
)
from hilbert_analysis.signal_utils import infer_sampling_rate, interpolate_small_gaps

LOGGER = logging.getLogger("hilbert_analysis")

DEFAULT_INPUT_ROOT = Path("outputs") / "rss"
#: Folder that groups human-motion recordings: ``Human_Motion/<subject>/<terrain>/...``.
HUMAN_MOTION_DIR = "Human_Motion"
DEFAULT_BACKUP_ROOT = Path("outputs") / "hilbert_analysis_backup"
REPORT_ROOT = Path("outputs") / "hilbert_analysis"
MIN_RECORDING_SECONDS = 2.0

SUMMARY_COLUMNS: tuple[str, ...] = (
    "input_csv",
    "relative_leaf_folder",
    "subject",
    "terrain",
    "status",
    "completeness_verdict",
    "onset_time_s",
    "offset_time_s",
    "end_margin_s",
    "n_pairs_evaluated",
    "n_pairs_flipped",
    "n_pairs_consistent",
    "n_low_confidence",
    "n_no_stride",
    "n_degenerate",
    "stride_source",
    "stride_s",
    "cycle_count",
    "moved_to",
    "message",
)
FLIP_DETAIL_COLUMNS: tuple[str, ...] = (
    "input_csv",
    "relative_leaf_folder",
    "limb",
    "mode",
    "left_column",
    "right_column",
    "correlation",
    "lag_samples",
    "lag_fraction",
    "stride_samples",
    "stride_source",
    "status",
    "reason",
    "window_source",
    "flipped_columns",
    "flip_applied",
)
ERROR_COLUMNS: tuple[str, ...] = ("input_csv", "error_message")


@dataclass(frozen=True)
class BatchConfig:
    """Configuration for recursive Hilbert-based gait-data cleaning."""

    input_root: Path = DEFAULT_INPUT_ROOT
    backup_root: Path = DEFAULT_BACKUP_ROOT
    completeness: CompletenessConfig = field(default_factory=CompletenessConfig)
    phase_flip: PhaseFlipConfig = field(default_factory=PhaseFlipConfig)
    report_root: Path = REPORT_ROOT
    apply: bool = False
    #: When False, orientation is measured and reported but no signal is
    #: rewritten. Marker columns are still written under ``--apply``.
    flip_signals: bool = False
    make_backup: bool = True
    min_recording_seconds: float = MIN_RECORDING_SECONDS
    diagnostic_plots: bool = False
    diagnostic_limit: int | None = None

    def validate(self) -> None:
        if not self.input_root.exists():
            raise FileNotFoundError(f"Input root does not exist: {self.input_root}")
        if not self.input_root.is_dir():
            raise NotADirectoryError(f"Input root is not a directory: {self.input_root}")


@dataclass(frozen=True)
class BatchResult:
    """Paths and counts produced by a completed batch."""

    summary_csv: Path
    flip_details_csv: Path
    errors_csv: Path
    files_discovered: int
    files_processed: int
    files_incomplete: int
    files_moved: int
    files_too_short: int
    pairs_flipped: int


@dataclass
class _LeafOutcome:
    summary_row: dict[str, Any]
    flip_detail_rows: list[dict[str, Any]]
    n_pairs_flipped: int
    moved_to: str
    completeness: CompletenessResult | None
    fs: float


def _relative_subject_terrain(relative: Path, anchor: str = HUMAN_MOTION_DIR) -> tuple[str, str]:
    parts = relative.parts
    if anchor not in parts:
        return "", ""
    rest = parts[parts.index(anchor) + 1 :]
    subject = rest[0] if len(rest) > 0 else ""
    terrain = rest[1] if len(rest) > 1 else ""
    return subject, terrain


def _summary_row(**kwargs: Any) -> dict[str, Any]:
    return {column: kwargs.get(column, "") for column in SUMMARY_COLUMNS}


def _write_frame_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def _backup_leaf_folder(leaf_folder: Path, *, input_root: Path, backup_root: Path) -> None:
    relative = leaf_folder.relative_to(input_root)
    destination = backup_root.resolve() / relative
    if destination.exists():
        return  # already backed up on a prior --apply run
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(leaf_folder, destination)


def _propagate_markers_to_sibling(
    sibling_path: Path,
    *,
    frame_column_reference: "pd.Series[Any]",
    segment_label: np.ndarray,
    cycle_index: np.ndarray,
) -> None:
    sibling = pd.read_csv(sibling_path)
    aligned = (
        "frame" in sibling.columns
        and len(sibling) == len(frame_column_reference)
        and (sibling["frame"].to_numpy() == frame_column_reference.to_numpy()).all()
    )
    if not aligned:
        LOGGER.warning(
            "Skipped marker propagation for %s: 'frame' column missing or misaligned.",
            sibling_path,
        )
        return
    sibling["segment_label"] = segment_label
    sibling["cycle_index"] = cycle_index
    _write_frame_csv(sibling_path, sibling)


def _process_leaf_folder(mode_csv: Path, *, input_root: Path, config: BatchConfig) -> _LeafOutcome:
    leaf_folder = mode_csv.parent
    relative_folder = leaf_folder.relative_to(input_root)
    subject, terrain = _relative_subject_terrain(relative_folder)

    frame = pd.read_csv(mode_csv)
    n_rows = len(frame)
    fs = infer_sampling_rate(frame["time"]) if "time" in frame.columns else 100.0

    base = dict(
        input_csv=str(mode_csv),
        relative_leaf_folder=str(relative_folder),
        subject=subject,
        terrain=terrain,
    )

    if n_rows < round(config.min_recording_seconds * fs):
        row = _summary_row(**base, status="too_short", message="Fewer rows than min_recording_seconds.")
        return _LeafOutcome(row, [], 0, "", None, fs)

    completeness = detect_completeness(frame, fs=fs, config=config.completeness)

    if completeness.verdict == VERDICT_INCOMPLETE:
        moved_to = ""
        if config.apply:
            destination = input_root / INCOMPLETE_SUBFOLDER_NAME / relative_folder
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(leaf_folder), str(destination))
            moved_to = str(destination)
        row = _summary_row(
            **base,
            status="incomplete",
            completeness_verdict=completeness.verdict,
            onset_time_s=completeness.onset_time_s,
            offset_time_s=completeness.offset_time_s,
            end_margin_s=completeness.end_margin_s,
            moved_to=moved_to,
        )
        return _LeafOutcome(row, [], 0, moved_to, completeness, fs)

    onset = completeness.onset_index
    offset = completeness.offset_index

    # One stride estimate for the whole recording: the half-stride lag is what
    # the pair test is read at, so a per-signal estimate that locks onto twice
    # the stride would invert the verdict.
    first_pair = PAIRED_LIMBS[0][1] if PAIRED_LIMBS else None
    fallback_column = f"{first_pair}_mode_1" if first_pair else None
    fallback_signal = None
    if fallback_column and fallback_column in frame.columns:
        window = slice(onset, offset + 1) if onset is not None else slice(None)
        fallback_signal = interpolate_small_gaps(
            pd.to_numeric(frame[fallback_column], errors="coerce").to_numpy(float)[window]
        )
    stride_samples, stride_source = stride_period_samples(
        frame, onset=onset, offset=offset, fs=fs,
        fallback_signal=fallback_signal,
        min_stride_s=config.phase_flip.min_stride_s,
        max_stride_s=config.phase_flip.max_stride_s,
        allow_full_record=config.phase_flip.allow_full_record_fallback,
    )

    pair_results = evaluate_frame(
        frame, onset=onset, offset=offset, stride_samples=stride_samples,
        stride_source=stride_source, config=config.phase_flip,
    )

    n_pairs_flipped = sum(1 for r in pair_results if r.status == STATUS_FLIPPED)
    n_pairs_consistent = sum(1 for r in pair_results if r.status == STATUS_CONSISTENT)
    n_low_confidence = sum(1 for r in pair_results if r.status == STATUS_LOW_CONFIDENCE)
    n_no_stride = sum(1 for r in pair_results if r.status == STATUS_NO_STRIDE)
    n_degenerate = sum(1 for r in pair_results if r.status == STATUS_DEGENERATE)

    flip_detail_rows = [
        {
            "input_csv": str(mode_csv),
            "relative_leaf_folder": str(relative_folder),
            "limb": r.limb,
            "mode": r.mode,
            "left_column": r.left_column,
            "right_column": r.right_column,
            "correlation": r.correlation,
            "lag_samples": r.lag_samples,
            "lag_fraction": r.lag_fraction,
            "stride_samples": r.stride_samples,
            "stride_source": r.stride_source,
            "status": r.status,
            "reason": r.reason,
            "window_source": r.window_source,
            "flipped_columns": "|".join(r.flipped_columns),
            "flip_applied": bool(r.needs_flip and config.apply and config.flip_signals),
        }
        for r in pair_results
    ]

    to_flip = pair_results if config.flip_signals else []
    corrected, _ = apply_flips(frame, to_flip)
    phase_reference = build_whole_body_phase_reference(corrected, onset=onset, offset=offset)
    markers = build_markers(n_rows, onset=onset, offset=offset, phase_reference=phase_reference)
    cycle_count = markers.n_cycles

    if config.apply:
        if config.make_backup:
            _backup_leaf_folder(leaf_folder, input_root=input_root, backup_root=config.backup_root)

        corrected["segment_label"] = markers.segment_label
        corrected["cycle_index"] = markers.cycle_index
        _write_frame_csv(mode_csv, corrected)

        # Negate the matching axis vectors so each flipped mode's
        # (angle, axis) pair still describes the identical rotation.
        axes_path = leaf_folder / AXES_CSV_NAME
        if axes_path.exists() and to_flip and n_pairs_flipped:
            axes_frame = pd.read_csv(axes_path)
            flipped_axes, _ = apply_axis_flips(axes_frame, to_flip)
            _write_frame_csv(axes_path, flipped_axes)

        frame_reference = (
            corrected["frame"] if "frame" in corrected.columns else pd.Series(range(n_rows))
        )
        for sibling_name in SIBLING_MODE_FILES:
            sibling_path = leaf_folder / sibling_name
            if sibling_path.exists():
                _propagate_markers_to_sibling(
                    sibling_path,
                    frame_column_reference=frame_reference,
                    segment_label=markers.segment_label,
                    cycle_index=markers.cycle_index,
                )

    row = _summary_row(
        **base,
        status="processed",
        completeness_verdict=completeness.verdict,
        onset_time_s=completeness.onset_time_s,
        offset_time_s=completeness.offset_time_s,
        end_margin_s=completeness.end_margin_s,
        n_pairs_evaluated=len(pair_results),
        n_pairs_flipped=n_pairs_flipped,
        n_pairs_consistent=n_pairs_consistent,
        n_low_confidence=n_low_confidence,
        n_no_stride=n_no_stride,
        n_degenerate=n_degenerate,
        stride_source=stride_source,
        stride_s=(stride_samples / fs) if stride_samples else float("nan"),
        cycle_count=cycle_count,
    )
    return _LeafOutcome(row, flip_detail_rows, n_pairs_flipped, "", completeness, fs)


@dataclass(frozen=True)
class RestoreResult:
    """Outcome of restoring pristine leaf folders over the live tree."""

    leaves_in_source: int
    leaves_matched: int
    leaves_differing: int
    leaves_missing_in_live: int
    leaves_restored: int


def restore_from_source(
    *, source_root: Path, input_root: Path, apply: bool = False
) -> RestoreResult:
    """Copy pristine leaf folders from ``source_root`` over ``input_root``.

    Used to undo a previous ``--apply`` run before re-processing: it is the
    only way to recover ``*_original_angle`` columns, whose sign cannot be
    recovered from the data itself. Dry-run by default; reports how many
    leaves differ. Raises if a source leaf has no counterpart in the live
    tree, so a mismatched pair of roots fails loudly rather than scattering
    folders.
    """
    source_root = source_root.resolve()
    input_root = input_root.resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(f"Restore source does not exist: {source_root}")

    source_files = sorted(source_root.rglob(MODE_CSV_NAME))
    if not source_files:
        raise FileNotFoundError(f"No {MODE_CSV_NAME} files found beneath {source_root}.")

    matched = differing = missing = restored = 0
    for source_csv in source_files:
        relative = source_csv.parent.relative_to(source_root)
        live_folder = input_root / relative
        if not live_folder.is_dir():
            missing += 1
            LOGGER.warning("No live counterpart for %s", relative)
            continue
        matched += 1
        live_csv = live_folder / MODE_CSV_NAME
        if not live_csv.exists() or live_csv.read_bytes() != source_csv.read_bytes():
            differing += 1
        if apply:
            for item in source_csv.parent.iterdir():
                if item.is_file():
                    shutil.copy2(item, live_folder / item.name)
            restored += 1

    if missing:
        raise FileNotFoundError(
            f"{missing} source leaf folder(s) have no counterpart under {input_root}; "
            "check that --restore-source and --input-root refer to the same run."
        )

    LOGGER.info(
        "Restore %s: %s leaves in source, %s matched, %s differing, %s restored.",
        "applied" if apply else "dry run", len(source_files), matched, differing, restored,
    )
    return RestoreResult(len(source_files), matched, differing, missing, restored)


def _sanitize_for_filename(relative: Path) -> str:
    return str(relative).replace("/", "__").replace("\\", "__").replace(" ", "_").replace(",", "")


def run_batch(config: BatchConfig) -> BatchResult:
    """Process every ``limb_angle_mode.csv`` and keep individual file failures isolated."""
    config.validate()
    input_root = config.input_root.resolve()
    mode_files = discover_leaf_folders(input_root)
    if not mode_files:
        raise FileNotFoundError(f"No {MODE_CSV_NAME} files were found beneath {input_root}.")

    summary_rows: list[dict[str, Any]] = []
    flip_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    files_processed = 0
    files_incomplete = 0
    files_moved = 0
    files_too_short = 0
    pairs_flipped = 0
    diagnostics_written = 0

    LOGGER.info("Discovered %s %s files under %s", len(mode_files), MODE_CSV_NAME, input_root)
    for file_index, mode_csv in enumerate(mode_files, start=1):
        relative_display = mode_csv.parent.relative_to(input_root)
        LOGGER.info("[%s/%s] Processing %s", file_index, len(mode_files), relative_display)
        try:
            outcome = _process_leaf_folder(mode_csv, input_root=input_root, config=config)
        except Exception as exc:  # one bad file must never abort the batch
            error_rows.append({"input_csv": str(mode_csv), "error_message": str(exc)})
            LOGGER.error("%s: %s", mode_csv, exc)
            continue

        summary_rows.append(outcome.summary_row)
        flip_rows.extend(outcome.flip_detail_rows)
        pairs_flipped += outcome.n_pairs_flipped
        status = outcome.summary_row["status"]
        if status == "too_short":
            files_too_short += 1
        elif status == "incomplete":
            files_incomplete += 1
            if outcome.moved_to:
                files_moved += 1
        elif status == "processed":
            files_processed += 1

        if (
            config.diagnostic_plots
            and outcome.completeness is not None
            and (config.diagnostic_limit is None or diagnostics_written < config.diagnostic_limit)
        ):
            plot_path = (
                config.report_root / "diagnostics" / f"{_sanitize_for_filename(relative_display)}.png"
            )
            short_title = "/".join(relative_display.parts[-3:])
            write_completeness_plot(
                outcome.completeness,
                fs=outcome.fs,
                threshold_multiplier=config.completeness.threshold_multiplier,
                title=short_title,
                output_path=plot_path,
            )
            diagnostics_written += 1

    config.report_root.mkdir(parents=True, exist_ok=True)
    summary_csv = config.report_root / "processing_summary.csv"
    flip_details_csv = config.report_root / "phase_flip_details.csv"
    errors_csv = config.report_root / "processing_errors.csv"
    pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_csv(summary_csv, index=False)
    pd.DataFrame(flip_rows, columns=FLIP_DETAIL_COLUMNS).to_csv(flip_details_csv, index=False)
    pd.DataFrame(error_rows, columns=ERROR_COLUMNS).to_csv(errors_csv, index=False)

    LOGGER.info(
        "Complete: %s processed, %s incomplete (%s moved), %s too short, %s pairs flipped, %s errors.",
        files_processed, files_incomplete, files_moved, files_too_short, pairs_flipped, len(error_rows),
    )
    return BatchResult(
        summary_csv=summary_csv,
        flip_details_csv=flip_details_csv,
        errors_csv=errors_csv,
        files_discovered=len(mode_files),
        files_processed=files_processed,
        files_incomplete=files_incomplete,
        files_moved=files_moved,
        files_too_short=files_too_short,
        pairs_flipped=pairs_flipped,
    )
