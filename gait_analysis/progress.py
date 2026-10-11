"""Lightweight, line-oriented progress reporting for long-running analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time


LOGGER = logging.getLogger("gait_analysis")


def format_duration(seconds: float) -> str:
    """Format a duration compactly for progress log lines."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def progress_note(stage: str, message: str) -> None:
    """Emit an immediately flushed note through the CLI logging handler."""
    LOGGER.info("[%s] %s", stage, message)


def progress_error(stage: str, message: str) -> None:
    """Make a failure visible while allowing failure-isolated stages to continue."""
    LOGGER.error("[%s] %s", stage, message)


@dataclass
class ProgressTracker:
    """Report determinate progress with elapsed time and a simple ETA."""

    stage: str
    total: int
    report_every: int = 1
    started_at: float = field(default_factory=time.monotonic)

    def update(self, current: int, message: str = "") -> None:
        total = max(0, int(self.total))
        current = max(0, int(current))
        interval = max(1, int(self.report_every))
        if current not in {0, 1, total} and current % interval != 0:
            return
        elapsed = time.monotonic() - self.started_at
        if total:
            bounded = min(current, total)
            percent = 100.0 * bounded / total
            rate = bounded / elapsed if bounded > 0 and elapsed > 0 else 0.0
            eta = (total - bounded) / rate if rate > 0 else 0.0
            timing = f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
            status = f"{bounded}/{total} ({percent:5.1f}%)"
        else:
            timing = f"elapsed={format_duration(elapsed)}"
            status = f"{current}/?"
        suffix = f" | {message}" if message else ""
        LOGGER.info("[%s] %s | %s%s", self.stage, status, timing, suffix)
