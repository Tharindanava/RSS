from __future__ import annotations

import logging
import time

from gait_analysis.progress import ProgressTracker, format_duration, progress_error, progress_note


def test_format_duration_is_compact() -> None:
    assert format_duration(9.6) == "10s"
    assert format_duration(65) == "1m05s"
    assert format_duration(3661) == "1h01m01s"


def test_progress_lines_include_stage_count_timing_and_messages(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="gait_analysis"):
        tracker = ProgressTracker("unit-stage", 10, started_at=time.monotonic() - 5.0)
        tracker.update(5, "working")
        progress_note("unit-stage", "note")
        progress_error("unit-stage", "failure")

    output = caplog.text
    assert "[unit-stage] 5/10" in output
    assert "50.0%" in output
    assert "elapsed=" in output
    assert "eta=" in output
    assert "working" in output
    assert "[unit-stage] note" in output
    assert "[unit-stage] failure" in output


def test_progress_tracker_throttles_intermediate_updates(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="gait_analysis"):
        tracker = ProgressTracker("throttled", 100, report_every=10)
        for current in range(1, 12):
            tracker.update(current)

    messages = [record.getMessage() for record in caplog.records if "[throttled]" in record.getMessage()]
    assert len(messages) == 2
    assert "1/100" in messages[0]
    assert "10/100" in messages[1]
