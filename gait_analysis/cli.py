"""Command-line interface: ``python -m gait_analysis <stage>``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Callable

from .config import AnalysisConfig
from .pipeline import (
    run_cluster, run_figures, run_manifest, run_pipeline, run_segmentation, run_stage1,
    run_stage2, run_statistics,
)
from .progress import format_duration, progress_note


COMMANDS: dict[str, Callable] = {
    "manifest": run_manifest,
    "segment": run_segmentation,
    "stage1": run_stage1,
    "stage2": run_stage2,
    "cluster": run_cluster,
    "statistics": run_statistics,
    "figures": run_figures,
    "all": run_pipeline,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gait_analysis", description="Two-stage gait analysis of RSS segment-mode signals")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", type=Path, default=None, help="Base YAML configuration (defaults to packaged study settings).")
    parser.add_argument("--sensitivity", type=Path, action="append", default=[], help="Validated YAML overlay; may be repeated.")
    parser.add_argument("--profile", choices=("development", "final"), default="development")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N manifest rows for smoke/development runs.")
    parser.add_argument("--force", action="store_true", help="Ignore matching derived caches.")
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO",
        help="Progress-log verbosity (default: INFO).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    started_at = time.monotonic()
    progress_note(
        "run",
        f"starting command={args.command} profile={args.profile} limit={args.limit} force={args.force}",
    )
    config = AnalysisConfig.from_yaml(args.config, overlays=args.sensitivity, profile=args.profile)
    progress_note(
        "run",
        f"config={config.source_path} config_hash={config.config_hash[:12]} output_root={config.path('output_root')}",
    )
    runner = COMMANDS[args.command]
    kwargs = {"limit": args.limit}
    if args.command != "manifest":
        kwargs["force"] = bool(args.force)
    try:
        artifacts = runner(config, **kwargs)
    except Exception:
        logging.getLogger("gait_analysis").exception(
            "[run] failed after %s", format_duration(time.monotonic() - started_at)
        )
        raise
    progress_note(
        "run",
        f"completed in {format_duration(time.monotonic() - started_at)} | output={artifacts.run_dir}",
    )
    print(artifacts.run_dir, flush=True)
