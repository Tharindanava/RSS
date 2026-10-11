from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rotating_poincare.cli import main as cli_main
from rotating_poincare.core import (
    RPTConfig,
    arc_length_parameterize,
    center_and_normalize_phase_points,
    compute_derivative,
    compute_rotating_poincare_transform,
    compute_rpt_features,
    interpolate_trajectory,
)
from rotating_poincare.io import load_signal_file
from rotating_poincare.pipeline import analyze_csv_files
from rotating_poincare.plotting import plot_rpt_heatmap


class RotatingPoincareCoreTests(unittest.TestCase):
    def test_derivative_from_time_and_sampling_frequency(self) -> None:
        time = np.linspace(0.0, 2.0 * np.pi, 2000, endpoint=False)
        signal = np.sin(time)
        from_time = compute_derivative(signal, t=time, smooth=False)
        from_fs = compute_derivative(
            signal,
            fs=1.0 / float(time[1] - time[0]),
            smooth=False,
        )
        np.testing.assert_allclose(from_time[2:-2], np.cos(time)[2:-2], atol=2e-5)
        np.testing.assert_allclose(from_fs, from_time, atol=1e-10)

    def test_derivative_uses_unit_spacing_and_adapts_smoothing_window(self) -> None:
        warnings: list[str] = []
        derivative = compute_derivative(
            np.arange(7, dtype=float) ** 2,
            smooth=True,
            savgol_window=21,
            savgol_polyorder=3,
            warnings=warnings,
        )
        self.assertEqual(derivative.shape, (7,))
        self.assertTrue(np.all(np.isfinite(derivative)))
        self.assertTrue(any("reduced" in warning for warning in warnings))

    def test_arc_length_removes_duplicate_points(self) -> None:
        points = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        s, unique = arc_length_parameterize(points)
        np.testing.assert_allclose(s, [0.0, 0.5, 1.0])
        self.assertEqual(unique.shape, (3, 2))
        interpolated = interpolate_trajectory(points, interp_points=25, method="pchip")
        self.assertEqual(interpolated.shape, (25, 2))

    def test_center_normalize_constant_coordinates_safely(self) -> None:
        points = np.column_stack((np.ones(10), np.zeros(10)))
        result = center_and_normalize_phase_points(points, normalize=True)
        np.testing.assert_allclose(result.points, 0.0)
        np.testing.assert_allclose(result.scale, [1.0, 1.0])
        self.assertEqual(len(result.warnings), 2)

    def test_center_normalize_maps_asymmetric_axes_to_unit_range(self) -> None:
        points = np.array(
            [
                [10.0, -4.0],
                [12.0, -2.0],
                [20.0, 8.0],
            ]
        )
        result = center_and_normalize_phase_points(points, normalize=True)
        np.testing.assert_allclose(result.mean, [15.0, 2.0])
        np.testing.assert_allclose(result.scale, [5.0, 6.0])
        np.testing.assert_allclose(np.min(result.points, axis=0), [-1.0, -1.0])
        np.testing.assert_allclose(np.max(result.points, axis=0), [1.0, 1.0])

    def test_clean_circle_and_crossing_directions_are_angle_consistent(self) -> None:
        parameter = np.linspace(0.0, 12.0 * np.pi, 6000, endpoint=False) + 0.173
        circle = np.column_stack((np.cos(parameter), -np.sin(parameter)))
        both = compute_rotating_poincare_transform(
            circle,
            theta_step_deg=10.0,
            num_radial_bins=40,
            eps=1e-10,
        )
        positive = compute_rotating_poincare_transform(
            circle,
            theta_step_deg=10.0,
            num_radial_bins=40,
            crossing_direction="positive",
            eps=1e-10,
        )
        negative = compute_rotating_poincare_transform(
            circle,
            theta_step_deg=10.0,
            num_radial_bins=40,
            crossing_direction="negative",
            eps=1e-10,
        )
        self.assertEqual(both.count_matrix.shape[0], 36)
        np.testing.assert_array_equal(both.count_matrix.sum(axis=1), 6)
        np.testing.assert_array_equal(
            both.count_matrix,
            positive.count_matrix + negative.count_matrix,
        )
        np.testing.assert_array_equal(
            both.count_matrix[:18],
            both.count_matrix[18:],
        )
        self.assertTrue(
            np.all(
                (positive.count_matrix.sum(axis=1) == 0)
                | (negative.count_matrix.sum(axis=1) == 0)
            )
        )

    def test_near_zero_run_is_counted_once(self) -> None:
        points = np.array(
            [
                [-1.0, -1.0],
                [-0.5, 0.0],
                [0.5, 0.0],
                [1.0, 1.0],
            ]
        )
        result = compute_rotating_poincare_transform(
            points,
            theta_min_deg=0.0,
            theta_max_deg=1.0,
            theta_step_deg=1.0,
            num_radial_bins=10,
        )
        self.assertEqual(int(result.count_matrix.sum()), 1)

    def test_heatmap_includes_mean_radial_trajectory(self) -> None:
        parameter = np.linspace(0.0, 4.0 * np.pi, 1000, endpoint=False) + 0.173
        circle = np.column_stack((np.cos(parameter), np.sin(parameter)))
        result = compute_rotating_poincare_transform(
            circle,
            theta_step_deg=30.0,
            num_radial_bins=20,
        )
        figure = plot_rpt_heatmap(result, "circle")
        try:
            labels = [line.get_label() for line in figure.axes[0].lines]
            self.assertIn("Mean radial trajectory", labels)
        finally:
            plt.close(figure)

    def test_grid_radial_overrides_and_no_crossing_features(self) -> None:
        points = np.array([[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]])
        result = compute_rotating_poincare_transform(
            points,
            theta_min_deg=0.0,
            theta_max_deg=90.0,
            theta_step_deg=30.0,
            num_radial_bins=8,
            radial_min=0.5,
            radial_max=4.0,
        )
        np.testing.assert_allclose(result.theta_values_deg, [0.0, 30.0, 60.0])
        self.assertEqual(result.count_matrix.shape, (3, 8))
        self.assertEqual(float(result.radial_bin_edges[0]), 0.5)
        self.assertEqual(float(result.radial_bin_edges[-1]), 4.0)

        empty_features = compute_rpt_features(
            np.zeros((1, 8), dtype=int),
            np.array([0.0]),
            result.radial_bin_centers,
        )
        self.assertEqual(int(empty_features.loc[0, "dominant_bin_index"]), -1)
        self.assertTrue(np.isnan(empty_features.loc[0, "dominant_r"]))
        self.assertEqual(float(empty_features.loc[0, "entropy"]), 0.0)

    def test_negative_radial_min_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive-ray"):
            compute_rotating_poincare_transform(
                np.array([[0.0, -1.0], [0.0, 1.0]]),
                radial_min=-1.0,
            )

    def test_noisy_cycle_has_higher_median_entropy(self) -> None:
        rng = np.random.default_rng(2026)
        parameter = np.linspace(0.0, 24.0 * np.pi, 12000, endpoint=False) + 0.173
        clean = np.column_stack((np.cos(parameter), np.sin(parameter)))
        radial_noise = 1.0 + rng.normal(scale=0.09, size=parameter.size)
        noisy = np.column_stack(
            (
                radial_noise * np.cos(parameter),
                0.72 * radial_noise * np.sin(parameter)
                + rng.normal(scale=0.025, size=parameter.size),
            )
        )
        clean_result = compute_rotating_poincare_transform(
            clean,
            theta_step_deg=10.0,
            num_radial_bins=60,
            radial_min=0.0,
            radial_max=1.5,
        )
        noisy_result = compute_rotating_poincare_transform(
            noisy,
            theta_step_deg=10.0,
            num_radial_bins=60,
            radial_min=0.0,
            radial_max=1.5,
        )
        clean_features = compute_rpt_features(
            clean_result.count_matrix,
            clean_result.theta_values_deg,
            clean_result.radial_bin_centers,
        )
        noisy_features = compute_rpt_features(
            noisy_result.count_matrix,
            noisy_result.theta_values_deg,
            noisy_result.radial_bin_centers,
        )
        self.assertGreater(
            float(noisy_features["entropy"].median()),
            float(clean_features["entropy"].median()),
        )


class RotatingPoincareIntegrationTests(unittest.TestCase):
    def _write_csv(self, path: Path, *, samples: int = 180) -> None:
        time = np.linspace(0.0, 4.0 * np.pi, samples, endpoint=False)
        pd.DataFrame(
            {
                "frame": np.arange(samples),
                "time": time,
                "hip angle": np.cos(time),
                "knee_angle": np.sin(time),
            }
        ).to_csv(path, index=False)

    def test_loader_infers_signals_and_parses_datetime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "signals.csv"
            pd.DataFrame(
                {
                    "Timestamp": pd.date_range("2026-01-01", periods=8, freq="10ms"),
                    "frame": np.arange(8),
                    "angle": np.arange(8, dtype=float),
                    "label": ["x"] * 8,
                }
            ).to_csv(path, index=False)
            loaded = load_signal_file(path)
            self.assertEqual(loaded.time_column, "Timestamp")
            self.assertEqual(loaded.signal_columns, ("angle",))
            np.testing.assert_allclose(loaded.time_values, np.arange(8) * 0.01)

    def test_batch_cli_writes_namespaced_csv_and_metadata_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.csv"
            second = root / "second.csv"
            output = root / "out"
            self._write_csv(first)
            self._write_csv(second)
            cli_main(
                [
                    "--input",
                    str(first),
                    str(second),
                    "--output-dir",
                    str(output),
                    "--signal-cols",
                    "hip angle",
                    "--interp-points",
                    "300",
                    "--theta-step-deg",
                    "30",
                    "--num-radial-bins",
                    "12",
                    "--no-plots",
                ]
            )
            for stem in ("first", "second"):
                signal_dir = output / stem / "hip_angle"
                counts = signal_dir / "hip_angle_rpt_counts.csv"
                features = signal_dir / "hip_angle_rpt_features.csv"
                metadata = signal_dir / "hip_angle_rpt_metadata.json"
                self.assertTrue(counts.exists())
                self.assertTrue(features.exists())
                self.assertTrue(metadata.exists())
                self.assertEqual(pd.read_csv(counts).shape, (12, 13))
                payload = json.loads(metadata.read_text(encoding="utf-8"))
                self.assertEqual(payload["signal_column"], "hip angle")
                self.assertEqual(payload["time_source"], "time_column:time")
                self.assertEqual(len(payload["radial_bin_centers"]), 12)
                self.assertEqual(
                    payload["phase_preprocessing"],
                    "centered_and_unit_range_scaled",
                )
                self.assertTrue(payload["normalize"])
                self.assertEqual(len(payload["phase_center"]), 2)
                self.assertEqual(len(payload["phase_scale"]), 2)

    def test_unit_spacing_warning_and_constant_signal_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "constant.csv"
            pd.DataFrame({"angle": np.ones(20)}).to_csv(path, index=False)
            artifacts = analyze_csv_files(
                [path],
                root / "out",
                RPTConfig(interp_points=50, theta_step_deg=45.0, num_radial_bins=10),
                no_plots=True,
            )
            self.assertEqual(int(artifacts[0].transform.count_matrix.sum()), 0)
            metadata_path = artifacts[0].output_directory / "angle_rpt_metadata.json"
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(any("unit sample spacing" in item for item in payload["warnings"]))
            self.assertTrue(any("near-zero arc length" in item for item in payload["warnings"]))

    def test_pipeline_transform_normalizes_phase_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "offset.csv"
            time = np.linspace(0.0, 4.0 * np.pi, 240, endpoint=False)
            pd.DataFrame({"time": time, "angle": 10.0 + np.sin(time)}).to_csv(
                path,
                index=False,
            )
            artifacts = analyze_csv_files(
                [path],
                root / "out",
                RPTConfig(interp_points=300, theta_step_deg=30.0, num_radial_bins=12),
                no_plots=True,
            )
            transform_radius = float(artifacts[0].transform.radial_bin_edges[-1])
            self.assertGreater(transform_radius, 1.0)
            self.assertLessEqual(transform_radius, 2.0**0.5 + 1e-6)
            metadata_path = artifacts[0].output_directory / "angle_rpt_metadata.json"
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(float(payload["phase_center"][0]), 10.0, places=6)

    def test_invalid_time_and_short_signal_raise_clear_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad_time = root / "bad_time.csv"
            pd.DataFrame({"time": [0.0, 1.0, 1.0, 2.0, 3.0], "angle": range(5)}).to_csv(
                bad_time,
                index=False,
            )
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                analyze_csv_files(
                    [bad_time],
                    root / "out1",
                    RPTConfig(interp_points=20),
                    no_plots=True,
                )

            short = root / "short.csv"
            pd.DataFrame({"angle": [1.0, 2.0, 3.0, 4.0]}).to_csv(short, index=False)
            with self.assertRaisesRegex(ValueError, "at least 5"):
                analyze_csv_files(
                    [short],
                    root / "out2",
                    RPTConfig(interp_points=20),
                    no_plots=True,
                )

    def test_plot_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "plot.csv"
            self._write_csv(path, samples=100)
            artifacts = analyze_csv_files(
                [path],
                root / "out",
                RPTConfig(interp_points=120, theta_step_deg=45.0, num_radial_bins=10),
                signal_cols=["hip angle"],
                no_plots=False,
            )
            output = artifacts[0].output_directory
            expected = (
                "hip_angle_phase_portrait.png",
                "hip_angle_rpt_heatmap.png",
                "hip_angle_rpt_surface3d.png",
                "hip_angle_rpt_features.png",
                "hip_angle_rpt_surface3d.html",
            )
            for filename in expected:
                self.assertTrue((output / filename).exists(), filename)


if __name__ == "__main__":
    unittest.main()
