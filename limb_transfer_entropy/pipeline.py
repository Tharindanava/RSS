"""Recursive batch processing and mirrored artifact export."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any
import warnings

import pandas as pd

from limb_transfer_entropy.core import (
    ContextSpec,
    MIN_EMBEDDED_SAMPLES,
    compute_side_transfer_entropy,
)
from limb_transfer_entropy.plotting import write_matrix_plot


LOGGER = logging.getLogger("limb_transfer_entropy")
SUMMARY_COLUMNS: tuple[str, ...] = (
    "input_csv",
    "relative_run_folder",
    "side",
    "status",
    "output_folder",
    "n_input_rows",
    "n_samples_aligned",
    "n_samples_used",
    "bins",
    "binning",
    "lag",
    "context_length",
    "stride",
    "message",
)
ERROR_COLUMNS: tuple[str, ...] = ("input_csv", "side", "error_message")


@dataclass(frozen=True)
class BatchConfig:
    """Configuration for recursive limb transfer-entropy analysis."""

    input_root: Path = Path("outputs") / "rss"
    output_root: Path = Path("outputs") / "limb_transfer_entropy"
    bins: int = 8
    binning: str = "quantile"
    context: ContextSpec = ContextSpec()
    min_embedded_samples: int = MIN_EMBEDDED_SAMPLES

    def validate(self) -> None:
        self.context.validate()
        if self.bins < 2:
            raise ValueError("Bins must be greater than or equal to 2.")
        if self.binning not in {"quantile", "uniform"}:
            raise ValueError("Binning must be 'quantile' or 'uniform'.")
        if self.min_embedded_samples <= 0:
            raise ValueError("Minimum embedded samples must be positive.")
        if not self.input_root.exists():
            raise FileNotFoundError(f"Input root does not exist: {self.input_root}")
        if not self.input_root.is_dir():
            raise NotADirectoryError(f"Input root is not a directory: {self.input_root}")


@dataclass(frozen=True)
class BatchResult:
    """Paths and counts produced by a completed batch."""

    summary_csv: Path
    errors_csv: Path
    files_discovered: int
    sides_succeeded: int
    sides_failed: int


def discover_mode_files(input_root: Path) -> list[Path]:
    """Find all exact ``limb_angle_mode.csv`` files recursively."""
    return sorted(
        path.resolve()
        for path in input_root.rglob("limb_angle_mode.csv")
        if path.is_file()
    )


def _output_folder_for(csv_path: Path, config: BatchConfig) -> Path:
    relative_parent = csv_path.parent.relative_to(config.input_root.resolve())
    return config.output_root.resolve() / relative_parent


def _summary_row(
    *,
    csv_path: Path,
    relative_folder: Path,
    side: str,
    status: str,
    output_folder: Path,
    config: BatchConfig,
    message: str,
    n_input_rows: int | None = None,
    n_samples_aligned: int | None = None,
    n_samples_used: int | None = None,
) -> dict[str, Any]:
    return {
        "input_csv": str(csv_path),
        "relative_run_folder": str(relative_folder),
        "side": side,
        "status": status,
        "output_folder": str(output_folder),
        "n_input_rows": n_input_rows,
        "n_samples_aligned": n_samples_aligned,
        "n_samples_used": n_samples_used,
        "bins": int(config.bins),
        "binning": config.binning,
        "lag": int(config.context.lag),
        "context_length": int(config.context.length),
        "stride": int(config.context.stride),
        "message": message,
    }


def _write_csv_without_cast_warning(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in cast")
        frame.to_csv(output_path, index=True)


def run_batch(config: BatchConfig) -> BatchResult:
    """Process every mode file and keep individual file/side failures isolated."""
    config.validate()
    input_root = config.input_root.resolve()
    output_root = config.output_root.resolve()
    mode_files = discover_mode_files(input_root)
    if not mode_files:
        raise FileNotFoundError(
            f"No limb_angle_mode.csv files were found beneath {input_root}."
        )

    summary_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, str]] = []
    sides_succeeded = 0
    sides_failed = 0

    LOGGER.info("Discovered %s limb_angle_mode.csv files under %s", len(mode_files), input_root)
    for file_index, csv_path in enumerate(mode_files, start=1):
        relative_folder = csv_path.parent.relative_to(input_root)
        output_folder = _output_folder_for(csv_path, config)
        LOGGER.info(
            "[%s/%s] Processing %s",
            file_index,
            len(mode_files),
            csv_path.relative_to(input_root),
        )
        try:
            frame = pd.read_csv(csv_path)
        except Exception as exc:
            message = f"Could not read CSV: {exc}"
            error_rows.append(
                {"input_csv": str(csv_path), "side": "", "error_message": message}
            )
            summary_rows.append(
                _summary_row(
                    csv_path=csv_path,
                    relative_folder=relative_folder,
                    side="",
                    status="failed",
                    output_folder=output_folder,
                    config=config,
                    message=message,
                )
            )
            sides_failed += 2
            LOGGER.error("%s", message)
            continue

        for side in ("left", "right"):
            try:
                result = compute_side_transfer_entropy(
                    frame,
                    side=side,
                    bins=config.bins,
                    binning=config.binning,
                    context=config.context,
                    min_embedded_samples=config.min_embedded_samples,
                )
                output_folder.mkdir(parents=True, exist_ok=True)
                matrix_csv = output_folder / f"{side}_te_matrix.csv"
                edges_csv = output_folder / f"{side}_te_edges.csv"
                plot_png = output_folder / f"{side}_te_matrix.png"
                _write_csv_without_cast_warning(result.matrix, matrix_csv)
                result.edges.to_csv(edges_csv, index=False)
                write_matrix_plot(
                    result.matrix,
                    side=side,
                    source_path=csv_path,
                    output_path=plot_png,
                )
                n_samples_used = result.n_samples_aligned - config.context.max_lag
                summary_rows.append(
                    _summary_row(
                        csv_path=csv_path,
                        relative_folder=relative_folder,
                        side=side,
                        status="success",
                        output_folder=output_folder,
                        config=config,
                        message="Transfer-entropy matrix written successfully.",
                        n_input_rows=result.n_input_rows,
                        n_samples_aligned=result.n_samples_aligned,
                        n_samples_used=n_samples_used,
                    )
                )
                sides_succeeded += 1
            except Exception as exc:
                message = str(exc)
                error_rows.append(
                    {
                        "input_csv": str(csv_path),
                        "side": side,
                        "error_message": message,
                    }
                )
                summary_rows.append(
                    _summary_row(
                        csv_path=csv_path,
                        relative_folder=relative_folder,
                        side=side,
                        status="failed",
                        output_folder=output_folder,
                        config=config,
                        message=message,
                        n_input_rows=int(len(frame)),
                    )
                )
                sides_failed += 1
                LOGGER.error("%s [%s]: %s", csv_path.relative_to(input_root), side, message)

    output_root.mkdir(parents=True, exist_ok=True)
    summary_csv = output_root / "processing_summary.csv"
    errors_csv = output_root / "processing_errors.csv"
    pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_csv(summary_csv, index=False)
    pd.DataFrame(error_rows, columns=ERROR_COLUMNS).to_csv(errors_csv, index=False)
    LOGGER.info(
        "Complete: %s side analyses succeeded; %s failed. Summary: %s",
        sides_succeeded,
        sides_failed,
        summary_csv,
    )
    return BatchResult(
        summary_csv=summary_csv,
        errors_csv=errors_csv,
        files_discovered=len(mode_files),
        sides_succeeded=sides_succeeded,
        sides_failed=sides_failed,
    )
