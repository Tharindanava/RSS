from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from hilbert_analysis.analytic import (
    analytic_signal,
    cycle_intervals,
    cycle_start_indices,
    envelope,
    instantaneous_frequency,
    instantaneous_phase,
    windowed_envelope,
)
from hilbert_analysis.columns import (
    LOWER_LIMB_SEGMENTS,
    PAIRED_LIMBS,
    mode_columns,
    original_angle_column,
)
from hilbert_analysis.completeness import (
    VERDICT_COMPLETE,
    VERDICT_INCOMPLETE,
    CompletenessConfig,
    detect_completeness,
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
    evaluate_pair,
)
from hilbert_analysis.pipeline import BatchConfig, run_batch
from hilbert_analysis.segmentation import (
    NO_CYCLE,
    SEGMENT_END_TRANSIENT,
    SEGMENT_PERIODIC,
    SEGMENT_START_TRANSIENT,
    build_markers,
    build_whole_body_phase_reference,
    stride_period_samples,
)
from hilbert_analysis.signal_utils import (
    infer_sampling_rate,
    interpolate_small_gaps,
    lagged_correlation,
)

SAMPLES_DIR = (
    Path(__file__).resolve().parents[1] / "Sample_Data" / "Labelled_Samples"
)
SAMPLE_B_INCOMPLETE = SAMPLES_DIR / "Sample_B_Flat_00_limb_angle_mode.csv"
SAMPLE_A_FLAT = SAMPLES_DIR / "Sample_A_Flat_00_limb_angle_mode.csv"
SAMPLE_A_ST_DOWN = SAMPLES_DIR / "Sample_A_St_Down_00_limb_angle_mode.csv"
SAMPLES_AVAILABLE = SAMPLE_B_INCOMPLETE.is_file() and SAMPLE_A_FLAT.is_file() and SAMPLE_A_ST_DOWN.is_file()


def make_sinusoid(freq_hz: float, fs: float, seconds: float, *, phase: float = 0.0) -> tuple[np.ndarray, float]:
    n = int(round(seconds * fs))
    t = np.arange(n) / fs
    return np.sin(2.0 * np.pi * freq_hz * t + phase), fs


class AnalyticCoreTests(unittest.TestCase):
    def test_envelope_is_constant_for_a_pure_sinusoid(self) -> None:
        signal, _ = make_sinusoid(2.0, 100.0, 5.0)
        env = envelope(signal, detrend=None)
        interior = env[20:-20]
        self.assertAlmostEqual(float(np.std(interior)), 0.0, delta=0.05)
        self.assertAlmostEqual(float(np.mean(interior)), 1.0, delta=0.05)

    def test_instantaneous_frequency_matches_known_frequency(self) -> None:
        signal, fs = make_sinusoid(3.0, 200.0, 4.0)
        freq = instantaneous_frequency(signal, fs, detrend=None)
        self.assertEqual(freq.size, signal.size - 1)
        self.assertAlmostEqual(float(np.median(freq[50:-50])), 3.0, delta=0.05)

    def test_instantaneous_phase_offset_matches_known_shift(self) -> None:
        reference, fs = make_sinusoid(2.0, 200.0, 4.0)
        shifted, _ = make_sinusoid(2.0, 200.0, 4.0, phase=-np.pi / 2)
        phase_ref = instantaneous_phase(reference, unwrap=True, detrend=None)
        phase_shifted = instantaneous_phase(shifted, unwrap=True, detrend=None)
        diff_deg = np.degrees(np.median((phase_ref - phase_shifted)[100:-100]))
        self.assertAlmostEqual(((diff_deg + 180) % 360) - 180, 90.0, delta=5.0)

    def test_analytic_signal_rejects_non_finite_input(self) -> None:
        signal = np.array([1.0, 2.0, np.nan, 4.0])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            analytic_signal(signal)

    def test_cycle_start_indices_count_matches_known_frequency(self) -> None:
        signal, fs = make_sinusoid(2.0, 200.0, 5.0)
        starts = cycle_start_indices(signal, detrend=None)
        # 5 seconds at 2 Hz is ~10 cycles; allow the usual +/-1 edge effect.
        self.assertIn(len(starts), range(8, 12))
        intervals = cycle_intervals(starts)
        self.assertEqual(len(intervals), len(starts) - 1)
        periods = [end - start for start, end in intervals]
        self.assertAlmostEqual(float(np.mean(periods)) / fs, 0.5, delta=0.03)

    def test_windowed_envelope_localizes_a_short_burst(self) -> None:
        fs = 100.0
        quiet = np.zeros(int(10 * fs))
        burst, _ = make_sinusoid(2.0, fs, 3.0)
        tail = np.zeros(int(10 * fs))
        signal = np.concatenate([quiet, burst, tail])
        env = windowed_envelope(signal, window_samples=30, detrend="linear")
        chunk_seconds = 30 / fs
        quiet_energy = env[: int(5 / chunk_seconds)]
        burst_energy = env[int(11 / chunk_seconds) : int(12 / chunk_seconds)]
        self.assertLess(float(np.max(quiet_energy)), 0.05)
        self.assertGreater(float(np.min(burst_energy)), 0.15)


class SignalUtilsTests(unittest.TestCase):
    def test_interpolate_small_gaps_fills_short_runs_only(self) -> None:
        values = np.array([1.0, 2.0, np.nan, np.nan, 5.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 12.0])
        filled = interpolate_small_gaps(values, max_gap=3)
        self.assertAlmostEqual(filled[2], 3.0)
        self.assertAlmostEqual(filled[3], 4.0)
        self.assertTrue(np.all(np.isnan(filled[5:11])))

    def test_infer_sampling_rate_from_regular_time_column(self) -> None:
        time = np.arange(500) * 0.01
        self.assertAlmostEqual(infer_sampling_rate(time), 100.0, delta=0.01)


class CompletenessTests(unittest.TestCase):
    def test_synthetic_flat_periodic_flat_signal_is_complete(self) -> None:
        fs = 100.0
        quiet = np.zeros(int(2 * fs))
        walking, _ = make_sinusoid(1.5, fs, 6.0)
        tail = np.zeros(int(3 * fs))
        signal = np.concatenate([quiet, walking * 0.3, tail])
        frame = _synthetic_lower_limb_frame(signal, fs)
        result = detect_completeness(frame, fs=fs, config=CompletenessConfig())
        self.assertEqual(result.verdict, VERDICT_COMPLETE)
        self.assertIsNotNone(result.onset_time_s)
        self.assertAlmostEqual(result.onset_time_s, 2.0, delta=0.5)

    def test_synthetic_signal_truncated_mid_motion_is_incomplete(self) -> None:
        fs = 100.0
        quiet = np.zeros(int(2 * fs))
        walking, _ = make_sinusoid(1.5, fs, 6.0)
        signal = np.concatenate([quiet, walking * 0.3])
        frame = _synthetic_lower_limb_frame(signal, fs)
        result = detect_completeness(frame, fs=fs, config=CompletenessConfig())
        self.assertEqual(result.verdict, VERDICT_INCOMPLETE)
        self.assertLess(result.end_margin_s, CompletenessConfig().min_end_margin_s)


def make_gait_waveform(fs: float, seconds: float, *, phase: float = 0.0) -> np.ndarray:
    """An asymmetric, gait-like waveform: one large excursion per cycle over a
    shallow remainder. ``phase=np.pi`` gives the half-stride-delayed partner
    limb, which is what a correct left/right pair looks like."""
    n = int(round(seconds * fs))
    theta = 2.0 * np.pi * 1.5 * (np.arange(n) / fs) + phase
    values = -np.exp(3.0 * np.sin(theta))
    return values - values.mean()


def make_leaf_frame(**columns: np.ndarray) -> pd.DataFrame:
    """Mode columns plus the untouchable ``original_angle`` for each segment."""
    data: dict[str, np.ndarray] = {}
    for base, series in columns.items():
        for mode in (1, 2, 3):
            data[f"{base}_mode_{mode}"] = series / mode
        data[f"{base}_original_angle"] = np.abs(series) + 1.0
    return pd.DataFrame(data)


class PairRelationshipTests(unittest.TestCase):
    """A correct pair is one waveform and a half-stride-delayed *copy* of it.
    A delayed *negated* copy means exactly one side is inverted."""

    fs = 100.0
    stride = 100.0 / 1.5          # samples per stride at 1.5 Hz

    def _evaluate(self, left: np.ndarray, right: np.ndarray, *, mode: int = 1, **overrides):
        return evaluate_pair(
            left, right, limb="ankle_to_foot",
            left_base="left_ankle_to_foot", right_base="right_ankle_to_foot",
            mode=mode, stride_samples=self.stride, stride_source="whole_body",
            config=PhaseFlipConfig(**overrides),
        )

    def test_delayed_copy_is_consistent(self) -> None:
        left = make_gait_waveform(self.fs, 10.0)
        right = make_gait_waveform(self.fs, 10.0, phase=np.pi)
        result = self._evaluate(left, right)
        self.assertEqual(result.status, STATUS_CONSISTENT)
        self.assertGreater(result.correlation, 0.0)
        self.assertEqual(result.flipped_columns, ())
        # The lag really is about half a stride.
        self.assertAlmostEqual(abs(result.lag_fraction), 0.5, delta=0.2)

    def test_delayed_negated_copy_flips_the_right_side(self) -> None:
        left = make_gait_waveform(self.fs, 10.0)
        right = -make_gait_waveform(self.fs, 10.0, phase=np.pi)
        result = self._evaluate(left, right)
        self.assertEqual(result.status, STATUS_FLIPPED)
        self.assertLess(result.correlation, 0.0)
        self.assertEqual(result.flipped_columns, ("right_ankle_to_foot_mode_1",))

        frame = make_leaf_frame(left_ankle_to_foot=left, right_ankle_to_foot=right)
        corrected, changed = apply_flips(frame, [result])
        self.assertEqual(changed, ("right_ankle_to_foot_mode_1",))
        np.testing.assert_allclose(corrected["right_ankle_to_foot_mode_1"].to_numpy(), -right)
        # Re-reading the corrected pair must now agree it is consistent.
        after = self._evaluate(
            corrected["left_ankle_to_foot_mode_1"].to_numpy(),
            corrected["right_ankle_to_foot_mode_1"].to_numpy(),
        )
        self.assertEqual(after.status, STATUS_CONSISTENT)

    def test_reference_side_is_never_the_one_flipped(self) -> None:
        left = make_gait_waveform(self.fs, 10.0)
        right = -make_gait_waveform(self.fs, 10.0, phase=np.pi)
        self.assertEqual(
            self._evaluate(left, right).flipped_columns, ("right_ankle_to_foot_mode_1",)
        )
        self.assertEqual(
            self._evaluate(left, right, reference_side="right").flipped_columns,
            ("left_ankle_to_foot_mode_1",),
        )

    def test_only_selected_modes_are_touched(self) -> None:
        """Modes carry independent sign gauges, and the default pass judges
        mode_1 only -- an inverted mode_2 pair must be left exactly as found."""
        good = make_gait_waveform(self.fs, 10.0)
        partner = make_gait_waveform(self.fs, 10.0, phase=np.pi)
        frame = pd.DataFrame({
            "left_ankle_to_foot_mode_1": good,
            "right_ankle_to_foot_mode_1": partner,
            "left_ankle_to_foot_mode_2": good,
            "right_ankle_to_foot_mode_2": -partner,      # inverted, but mode_2
        })
        results = evaluate_frame(
            frame, onset=0, offset=len(frame) - 1, stride_samples=self.stride,
            stride_source="whole_body", config=PhaseFlipConfig(modes=(1,)),
            pairs=(("ankle_to_foot", "left_ankle_to_foot", "right_ankle_to_foot"),),
        )
        self.assertEqual([r.mode for r in results], [1])
        _, changed = apply_flips(frame, results)
        self.assertEqual(changed, ())

    def test_original_angle_is_never_flipped(self) -> None:
        left = make_gait_waveform(self.fs, 10.0)
        right = -make_gait_waveform(self.fs, 10.0, phase=np.pi)
        frame = make_leaf_frame(left_ankle_to_foot=left, right_ankle_to_foot=right)
        result = self._evaluate(left, right)
        corrected, changed = apply_flips(frame, [result])
        self.assertTrue(all("original_angle" not in column for column in changed))
        np.testing.assert_allclose(
            corrected["right_ankle_to_foot_original_angle"].to_numpy(),
            frame["right_ankle_to_foot_original_angle"].to_numpy(),
        )

    def test_axis_triplet_negates_in_lockstep_preserving_the_rotation(self) -> None:
        left = make_gait_waveform(self.fs, 10.0)
        right = -make_gait_waveform(self.fs, 10.0, phase=np.pi)
        frame = make_leaf_frame(left_ankle_to_foot=left, right_ankle_to_foot=right)
        result = self._evaluate(left, right)

        axes = pd.DataFrame({col: np.full(len(frame), value) for col, value in (
            ("right_ankle_to_foot_mode_1_axis_x", 0.6),
            ("right_ankle_to_foot_mode_1_axis_y", 0.8),
            ("right_ankle_to_foot_mode_1_axis_z", 0.0),
            ("right_ankle_to_foot_mode_2_axis_x", 1.0),
            ("left_ankle_to_foot_mode_1_axis_x", 1.0),
        )})
        flipped_axes, changed = apply_axis_flips(axes, [result])
        np.testing.assert_allclose(flipped_axes["right_ankle_to_foot_mode_1_axis_x"], -0.6)
        np.testing.assert_allclose(flipped_axes["right_ankle_to_foot_mode_1_axis_y"], -0.8)
        # Neither the reference side nor a sibling mode is touched.
        np.testing.assert_allclose(flipped_axes["left_ankle_to_foot_mode_1_axis_x"], 1.0)
        np.testing.assert_allclose(flipped_axes["right_ankle_to_foot_mode_2_axis_x"], 1.0)
        self.assertNotIn("right_ankle_to_foot_mode_2_axis_x", changed)

        # theta * axis is invariant: (-theta) * (-axis) == theta * axis.
        corrected, _ = apply_flips(frame, [result])
        before = frame["right_ankle_to_foot_mode_1"].to_numpy() * 0.6
        after = corrected["right_ankle_to_foot_mode_1"].to_numpy() * -0.6
        np.testing.assert_allclose(before, after)

    def test_flat_side_is_degenerate(self) -> None:
        result = self._evaluate(make_gait_waveform(self.fs, 10.0), np.zeros(1000))
        self.assertEqual(result.status, STATUS_DEGENERATE)
        self.assertEqual(result.flipped_columns, ())

    def test_unrelated_signals_are_low_confidence(self) -> None:
        rng = np.random.default_rng(0)
        left = make_gait_waveform(self.fs, 10.0)
        right = rng.standard_normal(left.size)
        result = self._evaluate(left, right)
        self.assertEqual(result.status, STATUS_LOW_CONFIDENCE)
        self.assertEqual(result.flipped_columns, ())

    def test_missing_stride_period_is_reported_not_guessed(self) -> None:
        left = make_gait_waveform(self.fs, 10.0)
        right = -make_gait_waveform(self.fs, 10.0, phase=np.pi)
        result = evaluate_pair(
            left, right, limb="ankle_to_foot",
            left_base="left_ankle_to_foot", right_base="right_ankle_to_foot",
            mode=1, stride_samples=None, stride_source="unavailable",
            config=PhaseFlipConfig(),
        )
        self.assertEqual(result.status, STATUS_NO_STRIDE)
        self.assertEqual(result.flipped_columns, ())


class LaggedCorrelationTests(unittest.TestCase):
    def test_correlation_peaks_at_the_true_lag(self) -> None:
        """``lagged_correlation(a, b, lag)`` scores ``a(t)`` against
        ``b(t + lag)``, so a ``b`` delayed by 20 samples peaks at ``+20``."""
        fs = 100.0
        signal = make_gait_waveform(fs, 10.0)
        delayed = np.roll(signal, 20)
        lags = np.arange(-40, 41)
        correlations = lagged_correlation(signal, delayed, lags)
        self.assertAlmostEqual(float(lags[int(np.argmax(correlations))]), 20.0, delta=1.0)
        self.assertGreater(correlations.max(), 0.95)

    def test_negated_copy_correlates_negatively(self) -> None:
        signal = make_gait_waveform(100.0, 10.0)
        value = lagged_correlation(signal, -signal, np.array([0]))[0]
        self.assertAlmostEqual(value, -1.0, places=6)

    def test_short_overlap_and_constant_input_score_zero(self) -> None:
        signal = make_gait_waveform(100.0, 10.0)
        self.assertEqual(lagged_correlation(signal, signal, np.array([signal.size - 4]))[0], 0.0)
        self.assertEqual(lagged_correlation(signal, np.ones_like(signal), np.array([0]))[0], 0.0)


class SegmentationTests(unittest.TestCase):
    def test_build_markers_labels_transients_and_counts_cycles(self) -> None:
        fs = 100.0
        onset, offset = 200, 799  # 2s..8s
        n_rows = 1000
        walking, _ = make_sinusoid(1.5, fs, (offset - onset + 1) / fs)
        phase_reference = np.zeros(n_rows)
        phase_reference[onset : offset + 1] = walking

        markers = build_markers(n_rows, onset=onset, offset=offset, phase_reference=phase_reference)
        self.assertTrue(np.all(markers.segment_label[:onset] == SEGMENT_START_TRANSIENT))
        self.assertTrue(np.all(markers.segment_label[onset : offset + 1] == SEGMENT_PERIODIC))
        self.assertTrue(np.all(markers.segment_label[offset + 1 :] == SEGMENT_END_TRANSIENT))
        self.assertGreater(markers.n_cycles, 0)
        self.assertTrue(np.all(markers.cycle_index[:onset] == NO_CYCLE))
        self.assertTrue(np.any(markers.cycle_index[onset : offset + 1] > NO_CYCLE))

    def test_whole_body_phase_reference_preserves_the_gait_fundamental(self) -> None:
        """Left and right are half a cycle apart in TIME as well as opposite in
        polarity, so the reference must combine them without cancelling."""
        fs = 100.0
        left = make_gait_waveform(fs, 8.0)
        right = -make_gait_waveform(fs, 8.0, phase=np.pi)  # canonical: opposite polarity
        columns: dict[str, np.ndarray] = {}
        for segment in LOWER_LIMB_SEGMENTS:
            columns[f"{segment}_mode_1"] = left if segment.startswith("left") else right
        frame = pd.DataFrame(columns)

        reference = build_whole_body_phase_reference(frame)
        self.assertIsNotNone(reference)
        # The reference must retain the fundamental, i.e. one dominant cycle
        # per gait cycle: 8 s at 1.5 Hz is ~12 cycles.
        starts = cycle_start_indices(reference)
        self.assertIn(len(starts), range(10, 15))
        # A naive left-minus-right would double the frequency instead.
        naive_difference = left - right
        self.assertGreater(len(cycle_start_indices(naive_difference)), len(starts) + 4)


@unittest.skipUnless(SAMPLES_AVAILABLE, "Sample_Data/Labelled_Samples/ is not present in this checkout.")
class GoldenSampleTests(unittest.TestCase):
    """Pin the exact thresholds validated against the repo's own labeled
    sample files (see the hilbert_analysis plan file for the walkthrough)."""

    def test_sample_b_truncated_recording_is_incomplete(self) -> None:
        frame = pd.read_csv(SAMPLE_B_INCOMPLETE)
        fs = infer_sampling_rate(frame["time"])
        result = detect_completeness(frame, fs=fs, config=CompletenessConfig())
        self.assertEqual(result.verdict, VERDICT_INCOMPLETE)
        self.assertLess(result.end_margin_s, 0.1)

    def _pair_results(self, frame: pd.DataFrame) -> dict[str, object]:
        fs = infer_sampling_rate(frame["time"])
        completeness = detect_completeness(frame, fs=fs, config=CompletenessConfig())
        stride, source = stride_period_samples(
            frame, onset=completeness.onset_index, offset=completeness.offset_index, fs=fs
        )
        self.assertIsNotNone(stride, "sample file must yield a stride estimate")
        results = evaluate_frame(
            frame, onset=completeness.onset_index, offset=completeness.offset_index,
            stride_samples=stride, stride_source=source, config=PhaseFlipConfig(),
        )
        return {result.limb: result for result in results}

    def test_labeled_sample_pair_verdicts(self) -> None:
        """Five of the seven pairs in this sample are a delayed *negated* copy
        and need the right side flipped; two are already a delayed copy."""
        results = self._pair_results(pd.read_csv(SAMPLE_A_FLAT))
        expected = {
            "ankle_to_foot": -0.92, "hip_to_knee": -0.94, "knee_to_ankle": -0.92,
            "shoulder_to_elbow": -0.87, "wrist_to_hand": -0.87,
            "elbow_to_wrist": +0.89, "pelvis_to_hip": +0.81,
        }
        for limb, correlation in expected.items():
            result = results[limb]
            self.assertAlmostEqual(result.correlation, correlation, delta=0.05, msg=limb)
            self.assertEqual(
                result.status,
                STATUS_FLIPPED if correlation < 0 else STATUS_CONSISTENT, limb,
            )
        self.assertTrue(all(r.stride_source == "whole_body" for r in results.values()))

    def test_correcting_the_sample_makes_every_decided_pair_consistent(self) -> None:
        frame = pd.read_csv(SAMPLE_A_FLAT)
        results = self._pair_results(frame)
        self.assertTrue(any(r.needs_flip for r in results.values()))

        corrected, changed = apply_flips(frame, list(results.values()))
        self.assertTrue(all("original_angle" not in column for column in changed))
        self.assertTrue(all(column.startswith("right_") for column in changed))

        after = self._pair_results(corrected)
        self.assertEqual([limb for limb, r in after.items() if r.needs_flip], [])

    def test_negating_a_consistent_pair_is_detected_and_undone(self) -> None:
        frame = pd.read_csv(SAMPLE_A_FLAT)
        column = "right_elbow_to_wrist_mode_1"      # part of an already-consistent pair
        frame[column] = -frame[column]
        results = self._pair_results(frame)
        self.assertEqual(results["elbow_to_wrist"].status, STATUS_FLIPPED)

        restored, changed = apply_flips(frame, list(results.values()))
        self.assertIn(column, changed)
        np.testing.assert_allclose(
            restored[column].to_numpy(), pd.read_csv(SAMPLE_A_FLAT)[column].to_numpy()
        )

    def test_excluded_neck_pair_is_never_judged(self) -> None:
        """neck_to_left/right_shoulder are anatomically a pair but deliberately
        outside PAIRED_LIMBS, so the pass must not touch them."""
        results = self._pair_results(pd.read_csv(SAMPLE_A_FLAT))
        self.assertNotIn("neck_to_shoulder", results)
        self.assertTrue(all("neck_to" not in limb for limb in results))


class BatchPipelineTests(unittest.TestCase):
    def test_dry_run_writes_reports_but_touches_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "inputs"
            leaf = input_root / "Human_Motion" / "Subj" / "Terrain" / "trial_01"
            leaf.mkdir(parents=True)
            frame = _synthetic_leaf_frame()
            frame.to_csv(leaf / "limb_angle_mode.csv", index=False)
            original_bytes = (leaf / "limb_angle_mode.csv").read_bytes()

            config = BatchConfig(
                input_root=input_root, backup_root=root / "backup", report_root=root / "report", apply=False,
            )
            result = run_batch(config)

            self.assertEqual(result.files_discovered, 1)
            self.assertTrue(result.summary_csv.is_file())
            self.assertEqual((leaf / "limb_angle_mode.csv").read_bytes(), original_bytes)
            self.assertFalse((root / "backup").exists())

    def test_apply_rewrites_csv_and_keeps_a_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "inputs"
            leaf = input_root / "Human_Motion" / "Subj" / "Terrain" / "trial_01"
            leaf.mkdir(parents=True)
            frame = _synthetic_leaf_frame()
            frame.to_csv(leaf / "limb_angle_mode.csv", index=False)

            config = BatchConfig(
                input_root=input_root, backup_root=root / "backup", report_root=root / "report", apply=True,
            )
            result = run_batch(config)

            self.assertEqual(result.files_processed, 1)
            updated = pd.read_csv(leaf / "limb_angle_mode.csv")
            self.assertIn("segment_label", updated.columns)
            self.assertIn("cycle_index", updated.columns)
            self.assertTrue((root / "backup" / "Human_Motion" / "Subj" / "Terrain" / "trial_01" / "limb_angle_mode.csv").is_file())

    def test_incomplete_recording_is_moved_only_when_applying(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "inputs"
            leaf = input_root / "Human_Motion" / "Subj" / "Terrain" / "trial_cut"
            leaf.mkdir(parents=True)
            frame = _synthetic_leaf_frame(truncate=True)
            frame.to_csv(leaf / "limb_angle_mode.csv", index=False)

            dry_config = BatchConfig(
                input_root=input_root, backup_root=root / "backup", report_root=root / "report_dry", apply=False,
            )
            run_batch(dry_config)
            self.assertTrue(leaf.is_dir())

            apply_config = BatchConfig(
                input_root=input_root, backup_root=root / "backup", report_root=root / "report_apply", apply=True,
            )
            result = run_batch(apply_config)
            self.assertEqual(result.files_incomplete, 1)
            self.assertEqual(result.files_moved, 1)
            self.assertFalse(leaf.is_dir())
            moved = input_root / "_Incomplete_Signals" / "Human_Motion" / "Subj" / "Terrain" / "trial_cut"
            self.assertTrue((moved / "limb_angle_mode.csv").is_file())


def _synthetic_lower_limb_frame(signal: np.ndarray, fs: float) -> pd.DataFrame:
    from hilbert_analysis.columns import LOWER_LIMB_SEGMENTS, original_angle_column

    data: dict[str, np.ndarray] = {"time": np.arange(signal.size) / fs}
    for segment in LOWER_LIMB_SEGMENTS:
        data[original_angle_column(segment)] = signal
    return pd.DataFrame(data)


def _synthetic_leaf_frame(*, truncate: bool = False) -> pd.DataFrame:
    fs = 100.0
    quiet = np.zeros(int(2 * fs))
    walking, _ = make_sinusoid(1.5, fs, 6.0)
    tail = np.zeros(int(0.05 * fs)) if truncate else np.zeros(int(3 * fs))
    left_signal = np.concatenate([quiet, walking * 0.3, tail])
    right_signal = -left_signal
    n = left_signal.size

    data: dict[str, np.ndarray] = {
        "frame": np.arange(n),
        "time": np.arange(n) / fs,
    }
    for _, left_base, right_base in PAIRED_LIMBS:
        for base, series in ((left_base, left_signal), (right_base, right_signal)):
            for column in mode_columns(base):
                data[column] = series
            data[original_angle_column(base)] = series
    return pd.DataFrame(data)


if __name__ == "__main__":
    unittest.main()
