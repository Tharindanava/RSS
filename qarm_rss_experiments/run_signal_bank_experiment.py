"""Run one validated QArm RSS trajectory and record QArm feedback."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import time as _time
from typing import Callable, Protocol

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qarm_rss_experiments.signal_bank import (
    CENTER_POSE_DEG,
    JOINT_NAMES,
    MAX_ALLOWED_SPEED_DEG_S,
    SelectedSignal,
    load_selected_signal,
)


# Change this value to select any experiment from 1 through 50.
SIGNAL_NUMBER = 1

CARD_TYPE = "qarm_usb"
CARD_IDENTIFIER = "0"
POSITION_CHANNELS = np.array([1001, 1002, 1003], dtype=np.uint32)
VELOCITY_CHANNELS = np.array([3001, 3002, 3003], dtype=np.uint32)
PROFILE_VELOCITY_RAD_S = math.radians(MAX_ALLOWED_SPEED_DEG_S)
PROFILE_ACCELERATION_RAD_S2 = 1.0
POSITION_TOLERANCE_RAD = math.radians(1.0)
MOVE_TIMEOUT_SECONDS = 30.0
COUNTDOWN_SECONDS = 3
MAX_TIMING_LATENESS_SECONDS = 0.05
RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results"


class Clock(Protocol):
    def perf_counter(self) -> float: ...

    def time(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    perf_counter = staticmethod(_time.perf_counter)
    time = staticmethod(_time.time)
    sleep = staticmethod(_time.sleep)


class TimingDeadlineError(RuntimeError):
    """Raised when host scheduling is too late for safe playback."""


@dataclass(frozen=True)
class PlaybackSummary:
    sample_count: int
    maximum_lateness_seconds: float


LOG_FIELDS = (
    "signal_number",
    "family",
    "sample_index",
    "scheduled_time_s",
    "elapsed_time_s",
    "monotonic_time_s",
    "unix_time_s",
    "timing_jitter_s",
    "target_shoulder_rad",
    "target_elbow_rad",
    "target_wrist_rad",
    "target_shoulder_deg",
    "target_elbow_deg",
    "target_wrist_deg",
    "measured_shoulder_rad",
    "measured_elbow_rad",
    "measured_wrist_rad",
    "measured_shoulder_deg",
    "measured_elbow_deg",
    "measured_wrist_deg",
    "measured_shoulder_velocity_rad_s",
    "measured_elbow_velocity_rad_s",
    "measured_wrist_velocity_rad_s",
    "measured_shoulder_velocity_deg_s",
    "measured_elbow_velocity_deg_s",
    "measured_wrist_velocity_deg_s",
)


def _wait_until(clock: Clock, deadline: float) -> None:
    while True:
        remaining = deadline - clock.perf_counter()
        if remaining <= 0.0:
            return
        clock.sleep(remaining)


def _read_joint_values(card: object, channels: np.ndarray) -> np.ndarray:
    values = np.empty(len(channels), dtype=np.float64)
    card.read_other(channels, len(channels), values)
    if not np.isfinite(values).all():
        raise RuntimeError(f"QArm returned non-finite values for channels {channels.tolist()}.")
    return values


def command_pose_and_wait(
    card: object,
    target_rad: np.ndarray,
    *,
    clock: Clock,
    timeout_seconds: float = MOVE_TIMEOUT_SECONDS,
) -> np.ndarray:
    """Command a pose once and wait for the internal QArm profile to settle."""

    target = np.asarray(target_rad, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("target_rad must contain three finite joint positions.")
    card.write_other(POSITION_CHANNELS, len(POSITION_CHANNELS), target)
    deadline = clock.perf_counter() + float(timeout_seconds)
    while True:
        measured = _read_joint_values(card, POSITION_CHANNELS)
        if float(np.max(np.abs(measured - target))) <= POSITION_TOLERANCE_RAD:
            return measured
        if clock.perf_counter() >= deadline:
            raise TimeoutError("QArm did not reach the requested pose within the movement timeout.")
        clock.sleep(0.05)


def hold_current_position(card: object) -> bool:
    """Best-effort hold used after an interruption or playback failure."""

    try:
        measured = _read_joint_values(card, POSITION_CHANNELS)
        card.write_other(POSITION_CHANNELS, len(POSITION_CHANNELS), measured)
        return True
    except Exception:
        return False


def play_signal(
    card: object,
    signal: SelectedSignal,
    writer: csv.DictWriter,
    *,
    clock: Clock,
    flush: Callable[[], None] | None = None,
    maximum_lateness_seconds: float = MAX_TIMING_LATENESS_SECONDS,
) -> PlaybackSummary:
    """Play all samples in order and write synchronized QArm feedback rows."""

    start_monotonic = clock.perf_counter()
    maximum_lateness = 0.0
    for sample_index, scheduled_time in enumerate(signal.time_seconds):
        deadline = start_monotonic + float(scheduled_time)
        _wait_until(clock, deadline)
        monotonic_now = clock.perf_counter()
        elapsed = monotonic_now - start_monotonic
        timing_jitter = elapsed - float(scheduled_time)
        maximum_lateness = max(maximum_lateness, timing_jitter)
        if timing_jitter > float(maximum_lateness_seconds):
            raise TimingDeadlineError(
                f"Playback was {timing_jitter:.6f} s late at sample {sample_index}; "
                f"the allowed lateness is {maximum_lateness_seconds:.6f} s."
            )

        target = np.asarray(signal.joint_targets_rad[sample_index], dtype=np.float64)
        card.write_other(POSITION_CHANNELS, len(POSITION_CHANNELS), target)
        measured_position = _read_joint_values(card, POSITION_CHANNELS)
        measured_velocity = _read_joint_values(card, VELOCITY_CHANNELS)
        target_deg = np.degrees(target)
        measured_position_deg = np.degrees(measured_position)
        measured_velocity_deg = np.degrees(measured_velocity)
        row = {
            "signal_number": signal.signal_number,
            "family": signal.family,
            "sample_index": sample_index,
            "scheduled_time_s": f"{float(scheduled_time):.9f}",
            "elapsed_time_s": f"{elapsed:.9f}",
            "monotonic_time_s": f"{monotonic_now:.9f}",
            "unix_time_s": f"{clock.time():.9f}",
            "timing_jitter_s": f"{timing_jitter:.9f}",
        }
        for joint_index, joint_name in enumerate(JOINT_NAMES):
            row[f"target_{joint_name}_rad"] = f"{target[joint_index]:.12f}"
            row[f"target_{joint_name}_deg"] = f"{target_deg[joint_index]:.9f}"
            row[f"measured_{joint_name}_rad"] = f"{measured_position[joint_index]:.12f}"
            row[f"measured_{joint_name}_deg"] = f"{measured_position_deg[joint_index]:.9f}"
            row[f"measured_{joint_name}_velocity_rad_s"] = f"{measured_velocity[joint_index]:.12f}"
            row[f"measured_{joint_name}_velocity_deg_s"] = f"{measured_velocity_deg[joint_index]:.9f}"
        writer.writerow(row)
        if flush is not None and (sample_index + 1) % 100 == 0:
            flush()

    if flush is not None:
        flush()
    return PlaybackSummary(
        sample_count=len(signal.time_seconds),
        maximum_lateness_seconds=maximum_lateness,
    )


def _profile_options() -> str:
    options: list[str] = []
    for joint_index in (1, 2, 3):
        options.extend(
            (
                f"j{joint_index}_mode=0",
                f"j{joint_index}_profile_config=0",
                f"j{joint_index}_profile_velocity={PROFILE_VELOCITY_RAD_S:.15g}",
                f"j{joint_index}_profile_acceleration={PROFILE_ACCELERATION_RAD_S2:.15g}",
            )
        )
    return ",".join(options)


def _open_qarm() -> object:
    try:
        from quanser.hardware import HIL, MAX_STRING_LENGTH
    except ImportError as error:
        raise RuntimeError(
            "The Quanser Python API is required on the QArm control computer. "
            "Install it through the supported Quanser/QUARC environment."
        ) from error

    card = HIL()
    try:
        card.open(CARD_TYPE, CARD_IDENTIFIER)
        if not card.is_valid():
            raise RuntimeError("The opened QArm HIL card is invalid.")
        options = _profile_options()
        card.set_card_specific_options(options, MAX_STRING_LENGTH)
        return card
    except Exception:
        if card.is_valid():
            card.close()
        raise


def _print_preview(signal: SelectedSignal) -> None:
    metadata = signal.metadata
    print("\nSelected QArm RSS experiment")
    print(f"  Signal number: {signal.signal_number}")
    print(f"  Family: {signal.family} (family case {signal.family_case_number})")
    print(f"  Seed: {signal.seed}")
    print("  Duration/rate: 60 seconds at 100 Hz")
    for joint_name in JOINT_NAMES:
        minimum = float(metadata[f"{joint_name}_target_min_deg"])
        maximum = float(metadata[f"{joint_name}_target_max_deg"])
        peak_velocity = float(metadata[f"{joint_name}_peak_velocity_deg_s"])
        print(
            f"  {joint_name.capitalize():8s}: {minimum:8.3f} to {maximum:8.3f} deg, "
            f"peak {peak_velocity:6.3f} deg/s"
        )
    print("  Controlled joints: shoulder, elbow, wrist")
    print("  Yaw and gripper/claw are not commanded.")


def _countdown(clock: Clock) -> None:
    for remaining in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"Starting playback in {remaining}...")
        clock.sleep(1.0)


def _is_valid_card(card: object | None) -> bool:
    if card is None:
        return False
    try:
        return bool(card.is_valid())
    except Exception:
        return False


def main() -> int:
    try:
        signal = load_selected_signal(SIGNAL_NUMBER)
    except Exception as error:
        print(f"Could not load signal {SIGNAL_NUMBER}: {error}")
        return 2

    _print_preview(signal)
    expected_confirmation = f"RUN {signal.signal_number}"
    confirmation = input(f"\nType {expected_confirmation!r} to connect and move the QArm: ").strip()
    if confirmation != expected_confirmation:
        print("Confirmation did not match; no hardware connection was opened.")
        return 2

    clock = SystemClock()
    card: object | None = None
    completed_normally = False
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = RESULTS_DIRECTORY / f"signal_{signal.signal_number:02d}_{timestamp}.csv"

    try:
        print("Connecting to QArm...")
        card = _open_qarm()
        print("Moving to the first signal sample at the capped profile speed...")
        command_pose_and_wait(card, signal.joint_targets_rad[0], clock=clock)
        _countdown(clock)
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=LOG_FIELDS)
            writer.writeheader()
            stream.flush()
            summary = play_signal(card, signal, writer, clock=clock, flush=stream.flush)
        print(
            f"Playback completed: {summary.sample_count} samples, maximum lateness "
            f"{summary.maximum_lateness_seconds:.6f} s."
        )
        print("Returning to the center pose...")
        command_pose_and_wait(card, np.radians(CENTER_POSE_DEG), clock=clock)
        completed_normally = True
        print(f"Saved QArm feedback log: {output_path}")
        return 0
    except KeyboardInterrupt:
        print("\nExperiment interrupted. Holding the measured joint positions.")
        return 130
    except Exception as error:
        print(f"Experiment aborted: {error}")
        if output_path.exists():
            print(f"Partial feedback log: {output_path}")
        return 1
    finally:
        if _is_valid_card(card):
            if not completed_normally:
                hold_current_position(card)
            card.close()
            print("QArm connection closed.")


if __name__ == "__main__":
    raise SystemExit(main())
