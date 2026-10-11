from __future__ import annotations

import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from qarm_rss_experiments.practical_validation import (
    DISTAL_GYRO_COLUMNS,
    AlignmentPlan,
    PreparedTrial,
    ValidationConfig,
    estimate_pretrial_bias,
    geometry_metrics,
    haar_so3,
    match_and_score,
    prepare_rss_input,
    project_to_so3,
    qarm_source_axes_distal,
    run_validation,
)
from qarm_rss_experiments.plot_practical_results import (
    build_observed_signal_frame,
    export_per_test_reconstruction_plots,
)
from rss.model import integrated_body_angles_to_quaternions


def _robot_frame(sample_count: int = 200) -> pd.DataFrame:
    time = np.arange(sample_count, dtype=float) / 100.0
    frame = pd.DataFrame(
        {
            "time_s": time,
            "measured_base_rad": 0.35 * np.sin(2.0 * np.pi * 0.7 * time),
            "measured_shoulder_rad": np.zeros(sample_count),
            "measured_elbow_rad": 0.25 * np.sin(2.0 * np.pi * 1.1 * time + 0.4),
            "measured_wrist_rad": 0.20 * np.sin(2.0 * np.pi * 1.7 * time + 1.0),
        }
    )
    for joint in ("base", "elbow", "wrist"):
        frame[f"target_{joint}_rad"] = frame[f"measured_{joint}_rad"]
    for index, column in enumerate(DISTAL_GYRO_COLUMNS):
        frame[column] = 8.0 * np.sin(2.0 * np.pi * (0.5 + 0.3 * index) * time + index)
    frame["OrbiX-Node-1_IMU1_GX"] = np.linspace(0.0, 1.0, sample_count)
    return frame


class QArmPracticalValidationUnitTests(unittest.TestCase):
    def test_input_uses_only_distal_gyro(self) -> None:
        frame = _robot_frame()
        first = prepare_rss_input(frame, (0.1, -0.2, 0.3))
        modified = frame.copy()
        for column in [name for name in modified if name.startswith(("measured_", "target_"))]:
            modified[column] = np.linspace(-1000.0, 1000.0, len(modified))
        modified["OrbiX-Node-1_IMU1_GX"] = 1e9
        second = prepare_rss_input(modified, (0.1, -0.2, 0.3))
        self.assertTrue(np.array_equal(first, second))

    def test_bias_estimation_uses_nearest_fixed_stationary_window(self) -> None:
        count = 500
        raw = pd.DataFrame({column: np.full(count, value) for column, value in zip(DISTAL_GYRO_COLUMNS, (0.25, -0.5, 0.75))})
        raw.loc[400:, DISTAL_GYRO_COLUMNS] = 20.0
        bias, interval = estimate_pretrial_bias(raw, 400, stationary_seconds=2.0)
        self.assertEqual(interval, (200, 400))
        self.assertTrue(np.allclose(bias, (0.25, -0.5, 0.75)))

    def test_body_increment_integration_is_quaternion_composed(self) -> None:
        frame = pd.DataFrame({column: np.zeros(101) for column in DISTAL_GYRO_COLUMNS})
        frame[DISTAL_GYRO_COLUMNS[2]] = 90.0
        cumulative = prepare_rss_input(frame, (0.0, 0.0, 0.0), sample_rate_hz=100.0)
        quaternions = integrated_body_angles_to_quaternions(cumulative)
        self.assertTrue(np.allclose(np.linalg.norm(quaternions, axis=1), 1.0))
        expected = math.radians(90.9)
        self.assertAlmostEqual(2.0 * math.atan2(quaternions[-1, 3], quaternions[-1, 0]), expected, places=9)

    def test_matching_handles_permutation_sign_offset_but_not_scale(self) -> None:
        time = np.linspace(0.0, 5.0, 500)
        truth = np.vstack((np.sin(time), np.cos(1.7 * time), np.sin(2.3 * time + 0.4)))
        estimate = np.vstack((-truth[2] + 3.0, truth[0] - 2.0, -truth[1] + 1.0))
        rows, aligned = match_and_score(truth, estimate, test_number=1, batch_number=1, method="test")
        self.assertTrue(np.allclose(aligned, truth))
        self.assertTrue(all(row["rmse_rad"] < 1e-12 for row in rows))
        scaled_rows, _ = match_and_score(truth, 1.5 * estimate, test_number=1, batch_number=1, method="test")
        self.assertTrue(all(row["rmse_rad"] > 0.1 for row in scaled_rows))
        self.assertTrue(all(row["amplitude_scale"] == 1.0 for row in scaled_rows))

    def test_so3_projection_and_haar_samples_are_proper(self) -> None:
        matrices = haar_so3(np.random.default_rng(7), 50)
        self.assertTrue(np.allclose(np.transpose(matrices, (0, 2, 1)) @ matrices, np.eye(3), atol=1e-12))
        self.assertTrue(np.allclose(np.linalg.det(matrices), 1.0, atol=1e-12))
        projected = project_to_so3(np.diag((1.2, 0.8, 1.1)))
        self.assertTrue(np.allclose(projected.T @ projected, np.eye(3), atol=1e-12))
        self.assertAlmostEqual(float(np.linalg.det(projected)), 1.0)

    def test_qarm_geometry_has_orthogonal_home_and_detects_mismatch(self) -> None:
        home = _robot_frame(20)
        for column in ("measured_base_rad", "measured_shoulder_rad", "measured_elbow_rad", "measured_wrist_rad"):
            home[column] = 0.0
        axes = qarm_source_axes_distal(home)
        self.assertTrue(np.allclose(np.transpose(axes, (0, 2, 1)) @ axes, np.eye(3), atol=1e-12))
        self.assertLess(geometry_metrics(home)["orthogonality_defect_rms"], 1e-12)
        mismatched = home.copy()
        mismatched["measured_elbow_rad"] = 0.6
        self.assertGreater(geometry_metrics(mismatched)["orthogonality_defect_rms"], 0.5)


class QArmPracticalValidationSmokeTests(unittest.TestCase):
    def test_completed_fit_can_export_per_test_plot_without_refitting(self) -> None:
        frame = _robot_frame(200)
        truth = frame.loc[:, ("measured_base_rad", "measured_elbow_rad", "measured_wrist_rad")].to_numpy(dtype=float).T
        sources = np.vstack((-truth[2] + 1.0, truth[0] - 2.0, -truth[1] + 0.5))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            merged_path = root / "test_1.csv"
            frame.to_csv(merged_path, index=False)
            pd.DataFrame(
                [
                    {
                        "test_number": 1,
                        "batch_number": 1,
                        "merged_path": str(merged_path),
                        "quality_pass": True,
                        "exclusion_reasons": "",
                        "gyro_bias_x_deg_s": 0.0,
                        "gyro_bias_y_deg_s": 0.0,
                        "gyro_bias_z_deg_s": 0.0,
                    }
                ]
            ).to_csv(root / "data_quality_manifest.csv", index=False)
            fit_dir = root / "fits" / "test_1"
            fit_dir.mkdir(parents=True)
            persisted_frame = pd.read_csv(merged_path, float_precision="round_trip")
            np.savez_compressed(
                fit_dir / "rss.npz",
                sources=sources,
                input_hash=hashlib.sha256(
                    np.ascontiguousarray(prepare_rss_input(persisted_frame, (0.0, 0.0, 0.0))).tobytes()
                ).hexdigest(),
            )

            written = export_per_test_reconstruction_plots(
                root,
                formats=("png",),
                include_ssa_filtered=False,
            )
            signal_path = root / "figures" / "per_test" / "test_01" / "source_separated_vs_ground_truth.csv"
            plot_path = root / "figures" / "per_test" / "test_01" / "test_01_measured_vs_rss.png"
            observed_path = root / "figures" / "per_test" / "test_01" / "rss_observed_signals.csv"
            observed_plot = root / "figures" / "per_test" / "test_01" / "test_01_rss_observed_signals.png"
            self.assertIn(signal_path, written)
            self.assertTrue(plot_path.is_file())
            self.assertTrue(observed_path.is_file())
            self.assertTrue(observed_plot.is_file())
            exported = pd.read_csv(signal_path)
            for joint in ("base", "elbow", "wrist"):
                self.assertTrue(np.allclose(exported[f"measured_{joint}_rad"], exported[f"rss_{joint}_aligned_rad"]))

    def test_export_survives_lossy_default_csv_float_parsing_of_bias(self) -> None:
        # pandas' default C-engine float parser is not bit-exact (e.g. it reads
        # back "-0.16749999999999998" as a different float64), which used to
        # break the cache's exact-bytes input-hash check on the re-read bias.
        bias = (-0.16749999999999998, -0.137, 0.335)
        frame = _robot_frame(200)
        observations = prepare_rss_input(frame, bias)
        sources = np.vstack((-observations[2], observations[0], -observations[1]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            merged_path = root / "test_1.csv"
            frame.to_csv(merged_path, index=False)
            pd.DataFrame(
                [
                    {
                        "test_number": 1,
                        "batch_number": 1,
                        "merged_path": str(merged_path),
                        "quality_pass": True,
                        "exclusion_reasons": "",
                        "gyro_bias_x_deg_s": bias[0],
                        "gyro_bias_y_deg_s": bias[1],
                        "gyro_bias_z_deg_s": bias[2],
                    }
                ]
            ).to_csv(root / "data_quality_manifest.csv", index=False)
            fit_dir = root / "fits" / "test_1"
            fit_dir.mkdir(parents=True)
            np.savez_compressed(
                fit_dir / "rss.npz",
                sources=sources,
                input_hash=hashlib.sha256(np.ascontiguousarray(observations).tobytes()).hexdigest(),
            )

            written = export_per_test_reconstruction_plots(root, formats=("png",), include_ssa_filtered=False)
            self.assertTrue(written)

    def test_observed_signal_export_reconstructs_exact_rss_input(self) -> None:
        frame = _robot_frame(200)
        bias = (0.1, -0.2, 0.3)
        exported, observations = build_observed_signal_frame(
            frame,
            bias,
            sample_rate_hz=100.0,
            include_ssa_filtered=False,
        )
        expected = prepare_rss_input(frame, bias, sample_rate_hz=100.0)
        self.assertTrue(np.array_equal(observations, expected))
        for index, axis in enumerate(("x", "y", "z")):
            self.assertTrue(
                np.array_equal(
                    exported[f"rss_observed_integrated_body_rotation_{axis}_rad"],
                    expected[index],
                )
            )

    def test_end_to_end_artifact_contract(self) -> None:
        frame = _robot_frame(200)
        observations = prepare_rss_input(frame, (0.0, 0.0, 0.0))
        quality = {
            "test_number": 1,
            "batch_number": 1,
            "quality_pass": True,
            "exclusion_reasons": "",
            "ideal_low_correlation": False,
            "cmax_position": 0.2,
            "cmax_velocity": 0.2,
            "correlation_stratum": "moderate",
            "mean_tracking_nrmse": 0.0,
            "alignment_correlation": 0.95,
            "row_count": 200,
        }
        geometry = {"test_number": 1, "batch_number": 1, **geometry_metrics(frame)}
        prepared = PreparedTrial(1, 1, frame, observations, np.zeros(3), quality, geometry)
        plan = AlignmentPlan(1, 1, "raw.csv", "truth.csv", "merged.csv", 0, 0, len(frame), 0.95)
        with tempfile.TemporaryDirectory() as directory:
            config = ValidationConfig(
                data_root=directory,
                output_root=str(Path(directory) / "output"),
                expected_samples=200,
                expected_duration_seconds=1.99,
                rss_epochs=1,
                pso_swarm_size=3,
                null_draws=3,
                stability_seeds=(),
                bootstrap_resamples=5,
                include_sensitivities=False,
            )
            with patch("qarm_rss_experiments.practical_validation.reproduce_alignment_plans", return_value=[plan]), patch(
                "qarm_rss_experiments.practical_validation.prepare_trial", return_value=prepared
            ):
                outputs = run_validation(config)
            required = {
                "data_quality_manifest",
                "per_trial_metrics",
                "per_source_metrics",
                "trial_geometry_metrics",
                "comparator_metrics",
                "random_rotation_null",
                "robustness_models",
                "algorithmic_stability",
                "validation_report",
            }
            self.assertTrue(required.issubset(outputs))
            report = Path(outputs["validation_report"]).read_text(encoding="utf-8")
            self.assertIn("ideal analysis set is empty", report)
            self.assertIn("No 12/6/6/3 grouping", report)


if __name__ == "__main__":
    unittest.main()
