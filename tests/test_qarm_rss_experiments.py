from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from qarm_rss_experiments.run_signal_bank_experiment import (
    POSITION_CHANNELS,
    VELOCITY_CHANNELS,
    TimingDeadlineError,
    _profile_options,
    command_pose_and_wait,
    hold_current_position,
    main,
    play_signal,
)
from qarm_rss_experiments.signal_bank import (
    DEFAULT_BANK_PATH,
    DEFAULT_MANIFEST_PATH,
    DESIGN_SPEED_LIMIT_DEG_S,
    DURATION_SECONDS,
    EXPECTED_FAMILY_COUNTS,
    JOINT_NAMES,
    SAMPLE_COUNT,
    SAMPLE_RATE_HZ,
    SOURCE_CROSS_CORRELATION_LIMIT,
    SelectedSignal,
    build_signal_bank,
    load_manifest,
    load_signal_bank,
    source_cross_correlation_matrix,
    validate_signal_bank,
)


class _FakeClock:
    def __init__(self, *, oversleep_seconds: float = 0.0) -> None:
        self.monotonic = 100.0
        self.unix = 1_800_000_000.0
        self.oversleep_seconds = float(oversleep_seconds)
        self.sleep_count = 0

    def perf_counter(self) -> float:
        return self.monotonic

    def time(self) -> float:
        return self.unix

    def sleep(self, seconds: float) -> None:
        delay = max(0.0, float(seconds))
        self.sleep_count += 1
        if self.sleep_count == 1:
            delay += self.oversleep_seconds
        self.monotonic += delay
        self.unix += delay


class _FakeCard:
    def __init__(self) -> None:
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.writes: list[tuple[np.ndarray, np.ndarray]] = []
        self.valid = True

    def is_valid(self) -> bool:
        return self.valid

    def close(self) -> None:
        self.valid = False

    def write_other(self, channels: np.ndarray, count: int, values: np.ndarray) -> None:
        channel_copy = np.array(channels[:count], copy=True)
        value_copy = np.array(values[:count], dtype=np.float64, copy=True)
        self.writes.append((channel_copy, value_copy))
        if np.array_equal(channel_copy, POSITION_CHANNELS):
            self.position[:] = value_copy

    def read_other(self, channels: np.ndarray, count: int, values: np.ndarray) -> None:
        selected = np.asarray(channels[:count])
        if np.array_equal(selected, POSITION_CHANNELS):
            values[:] = self.position
        elif np.array_equal(selected, VELOCITY_CHANNELS):
            values[:] = self.velocity
        else:
            raise AssertionError(f"Unexpected fake read channels: {selected.tolist()}")


class _CountingWriter:
    def __init__(self) -> None:
        self.count = 0
        self.first: dict[str, object] | None = None
        self.last: dict[str, object] | None = None

    def writerow(self, row: dict[str, object]) -> None:
        copied = dict(row)
        if self.first is None:
            self.first = copied
        self.last = copied
        self.count += 1


class QArmSignalBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generated_bank, cls.generated_manifest = build_signal_bank()

    def test_bank_has_required_allocation_timing_and_seed_schedule(self) -> None:
        bank = self.generated_bank
        self.assertEqual(bank["source_angles_rad"].shape, (50, SAMPLE_COUNT, 3))
        self.assertEqual(bank["joint_targets_rad"].shape, (50, SAMPLE_COUNT, 3))
        self.assertEqual(tuple(bank["joint_names"].tolist()), JOINT_NAMES)
        self.assertAlmostEqual(float(bank["time_seconds"][0]), 0.0)
        self.assertAlmostEqual(float(bank["time_seconds"][-1]), DURATION_SECONDS)
        self.assertTrue(np.allclose(np.diff(bank["time_seconds"]), 1.0 / SAMPLE_RATE_HZ))

        family_counts = {
            family: int(np.count_nonzero(bank["families"] == family))
            for family in np.unique(bank["families"])
        }
        self.assertEqual(family_counts, EXPECTED_FAMILY_COUNTS)
        self.assertTrue(np.array_equal(bank["seeds"][:17], np.arange(21000, 21017)))
        self.assertTrue(np.array_equal(bank["seeds"][17:34], np.arange(20000, 20017)))
        self.assertTrue(np.array_equal(bank["seeds"][34:], np.arange(20303, 20319)))
        self.assertEqual(len(np.unique(bank["seeds"])), 50)

    def test_every_signal_is_centered_independent_and_safe(self) -> None:
        bank = self.generated_bank
        sources = bank["source_angles_rad"]
        targets = bank["joint_targets_rad"]
        self.assertTrue(np.allclose(np.mean(sources, axis=1), 0.0, rtol=0.0, atol=1e-12))

        limits = bank["safe_position_limits_rad"]
        self.assertTrue(np.all(targets >= limits[None, :, 0] - 1e-12))
        self.assertTrue(np.all(targets <= limits[None, :, 1] + 1e-12))
        velocities = np.diff(targets, axis=1) * SAMPLE_RATE_HZ
        self.assertLessEqual(
            float(np.max(np.abs(velocities))),
            math.radians(DESIGN_SPEED_LIMIT_DEG_S) + 1e-12,
        )

        for signal in sources:
            correlation = source_cross_correlation_matrix(signal.T)
            pairwise = np.abs(correlation[np.triu_indices(3, k=1)])
            self.assertLess(float(np.max(pairwise)), SOURCE_CROSS_CORRELATION_LIMIT)

    def test_generation_is_deterministic(self) -> None:
        repeated_bank, repeated_manifest = build_signal_bank()
        for name, expected in self.generated_bank.items():
            self.assertTrue(np.array_equal(repeated_bank[name], expected), msg=name)
        self.assertEqual(repeated_manifest, self.generated_manifest)

    def test_checked_in_artifacts_match_regeneration(self) -> None:
        persisted_bank = load_signal_bank(DEFAULT_BANK_PATH)
        persisted_manifest = load_manifest(DEFAULT_MANIFEST_PATH)
        for name, expected in self.generated_bank.items():
            self.assertTrue(np.array_equal(persisted_bank[name], expected), msg=name)
        self.assertEqual(persisted_manifest[0]["signal_number"], "1")
        self.assertEqual(persisted_manifest[-1]["signal_number"], "50")
        self.assertEqual(persisted_manifest[0]["family"], "legacy")
        self.assertEqual(persisted_manifest[-1]["family"], "piecewise_transient")

    def test_validator_rejects_modified_target_contract(self) -> None:
        modified = {name: np.array(value, copy=True) for name, value in self.generated_bank.items()}
        modified["joint_targets_rad"][0, 10, 0] += 0.5
        with self.assertRaisesRegex(ValueError, "source angles plus the center pose"):
            validate_signal_bank(modified)


class QArmPlaybackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bank, manifest = build_signal_bank()
        index = 0
        cls.signal = SelectedSignal(
            signal_number=1,
            family=str(bank["families"][index]),
            family_case_number=int(bank["family_case_numbers"][index]),
            seed=int(bank["seeds"][index]),
            time_seconds=np.array(bank["time_seconds"], copy=True),
            source_angles_rad=np.array(bank["source_angles_rad"][index], copy=True),
            joint_targets_rad=np.array(bank["joint_targets_rad"][index], copy=True),
            metadata={key: str(value) for key, value in manifest[index].items()},
        )

    def test_fake_hil_receives_all_samples_in_order(self) -> None:
        card = _FakeCard()
        clock = _FakeClock()
        writer = _CountingWriter()
        flush_count = 0

        def flush() -> None:
            nonlocal flush_count
            flush_count += 1

        summary = play_signal(card, self.signal, writer, clock=clock, flush=flush)
        self.assertEqual(summary.sample_count, SAMPLE_COUNT)
        self.assertEqual(writer.count, SAMPLE_COUNT)
        self.assertEqual(len(card.writes), SAMPLE_COUNT)
        self.assertTrue(np.array_equal(card.writes[0][0], POSITION_CHANNELS))
        self.assertTrue(np.allclose(card.writes[0][1], self.signal.joint_targets_rad[0]))
        self.assertTrue(np.allclose(card.writes[-1][1], self.signal.joint_targets_rad[-1]))
        self.assertEqual(writer.first["sample_index"], 0)
        self.assertEqual(writer.last["sample_index"], SAMPLE_COUNT - 1)
        self.assertAlmostEqual(clock.monotonic, 100.0 + DURATION_SECONDS)
        self.assertGreaterEqual(flush_count, 61)

    def test_excessive_lateness_aborts_before_late_sample_is_written(self) -> None:
        card = _FakeCard()
        clock = _FakeClock(oversleep_seconds=0.1)
        writer = _CountingWriter()
        with self.assertRaises(TimingDeadlineError):
            play_signal(card, self.signal, writer, clock=clock, maximum_lateness_seconds=0.05)
        self.assertEqual(len(card.writes), 1)
        self.assertEqual(writer.count, 1)

    def test_preposition_and_interruption_hold_use_only_selected_joints(self) -> None:
        card = _FakeCard()
        clock = _FakeClock()
        target = self.signal.joint_targets_rad[0]
        measured = command_pose_and_wait(card, target, clock=clock)
        self.assertTrue(np.allclose(measured, target))
        self.assertTrue(hold_current_position(card))
        self.assertEqual(len(card.writes), 2)
        for channels, _ in card.writes:
            self.assertTrue(np.array_equal(channels, POSITION_CHANNELS))

    def test_profile_options_limit_only_shoulder_elbow_and_wrist(self) -> None:
        options = _profile_options()
        for joint_index in (1, 2, 3):
            self.assertIn(f"j{joint_index}_mode=0", options)
            self.assertIn(f"j{joint_index}_profile_velocity=", options)
            self.assertIn(f"j{joint_index}_profile_acceleration=", options)
        self.assertNotIn("j0_", options)
        self.assertNotIn("gripper", options)

    def test_main_interrupt_holds_position_and_closes_fake_hil(self) -> None:
        card = _FakeCard()
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("qarm_rss_experiments.run_signal_bank_experiment.load_selected_signal", return_value=self.signal),
                patch("builtins.input", return_value="RUN 1"),
                patch("qarm_rss_experiments.run_signal_bank_experiment._open_qarm", return_value=card),
                patch("qarm_rss_experiments.run_signal_bank_experiment._countdown"),
                patch("qarm_rss_experiments.run_signal_bank_experiment.play_signal", side_effect=KeyboardInterrupt),
                patch("qarm_rss_experiments.run_signal_bank_experiment.RESULTS_DIRECTORY", Path(directory)),
                patch("builtins.print"),
            ):
                exit_code = main()
        self.assertEqual(exit_code, 130)
        self.assertFalse(card.valid)
        self.assertGreaterEqual(len(card.writes), 2)
        self.assertTrue(np.array_equal(card.writes[-1][0], POSITION_CHANNELS))


if __name__ == "__main__":
    unittest.main()
