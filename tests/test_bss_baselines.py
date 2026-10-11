from __future__ import annotations

from itertools import permutations
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bss_baselines import (
    BSSConfig,
    DEFAULT_METHODS,
    build_bss_metrics_frame,
    build_bss_source_metrics_frame,
    build_method_comparison_frames,
    export_bss_case_artifacts,
    fit_bss,
    run_bss_methods_for_scenario,
)
from rss import (
    RSSConfig,
    SimulationConfig,
    generate_validation_scenario,
    run_validation_for_scenario,
)


def _best_mean_abs_correlation(truth: np.ndarray, estimate: np.ndarray) -> float:
    count = int(truth.shape[0])
    correlations = np.abs(np.corrcoef(truth, estimate)[:count, count:])
    return max(
        sum(correlations[index, permutation[index]] for index in range(count)) / count
        for permutation in permutations(range(count))
    )


class BSSCoreTests(unittest.TestCase):
    @staticmethod
    def _instantaneous_mixture(sample_count: int = 5000) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(41)
        sources = np.vstack(
            (
                rng.laplace(size=sample_count),
                rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=sample_count),
                rng.choice((-1.0, 1.0), size=sample_count),
            )
        )
        sources = (sources - sources.mean(axis=1, keepdims=True)) / sources.std(
            axis=1, keepdims=True
        )
        mixing = np.array(
            [[1.0, 0.3, 0.2], [0.2, 1.0, 0.4], [0.1, 0.3, 1.0]],
            dtype=float,
        )
        return sources, mixing @ sources

    def test_all_methods_reconstruct_and_use_unit_physical_axes(self) -> None:
        _, observations = self._instantaneous_mixture()
        config = BSSConfig(fastica_max_iter=1000)
        for method in DEFAULT_METHODS:
            result = fit_bss(observations, method, config)
            self.assertEqual(result.source_angles.shape, observations.shape)
            self.assertEqual(result.demixing_matrix.shape, (3, 3))
            self.assertTrue(np.isfinite(result.source_angles).all())
            self.assertTrue(
                np.allclose(
                    np.linalg.norm(result.normalized_demixing_matrix, axis=1),
                    1.0,
                    atol=1e-12,
                )
            )
            self.assertTrue(
                np.allclose(
                    result.source_angles,
                    result.raw_source_signals / result.demixing_row_norms[:, None],
                    atol=1e-12,
                )
            )
            self.assertLess(
                float(np.sqrt(np.mean((result.reconstructed_observations - observations) ** 2))),
                1e-10,
            )

    def test_jade_and_fastica_recover_known_non_gaussian_sources(self) -> None:
        sources, observations = self._instantaneous_mixture()
        config = BSSConfig(fastica_max_iter=1000, jade_tol=1e-10)
        for method in ("jade_ica", "fastica"):
            result = fit_bss(observations, method, config)
            self.assertTrue(result.converged)
            self.assertGreater(
                _best_mean_abs_correlation(sources, result.source_angles),
                0.97,
            )

    def test_iterative_methods_are_reproducible(self) -> None:
        _, observations = self._instantaneous_mixture(sample_count=1000)
        config = BSSConfig(fastica_max_iter=500, random_state=7)
        for method in ("jade_ica", "fastica", "sobi"):
            first = fit_bss(observations, method, config)
            second = fit_bss(observations, method, config)
            self.assertTrue(np.allclose(first.demixing_matrix, second.demixing_matrix))
            self.assertTrue(np.allclose(first.source_angles, second.source_angles))

    def test_invalid_and_rank_deficient_observations_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fit_bss(np.zeros((3, 7)), "pca")
        invalid = np.ones((3, 16), dtype=float)
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_bss(invalid, "pca")
        rank_deficient = np.vstack((np.arange(16.0), np.arange(16.0), np.arange(16.0)))
        with self.assertRaises(ValueError):
            fit_bss(rank_deficient, "jade_ica")

    @staticmethod
    def _colored_mixture(sample_count: int = 5000) -> tuple[np.ndarray, np.ndarray]:
        """AR(1) sources with distinct memory, mixed by the i.i.d. fixture's matrix.

        The rows are Gaussian, so fourth-order methods gain nothing here; the sources are
        told apart purely by their autocorrelation, which is what SOBI exploits.
        """

        rng = np.random.default_rng(17)
        sources = np.empty((3, sample_count), dtype=float)
        for row, coefficient in enumerate((0.95, 0.55, -0.80)):
            innovations = rng.standard_normal(sample_count)
            signal = np.zeros(sample_count, dtype=float)
            for index in range(1, sample_count):
                signal[index] = coefficient * signal[index - 1] + innovations[index]
            sources[row] = signal
        sources = (sources - sources.mean(axis=1, keepdims=True)) / sources.std(axis=1, keepdims=True)
        mixing = np.array(
            [[1.0, 0.3, 0.2], [0.2, 1.0, 0.4], [0.1, 0.3, 1.0]],
            dtype=float,
        )
        return sources, mixing @ sources

    def test_sobi_recovers_temporally_colored_sources(self) -> None:
        sources, observations = self._colored_mixture()
        result = fit_bss(observations, "sobi", BSSConfig(sobi_tol=1e-10))
        self.assertTrue(result.converged)
        self.assertEqual(result.implementation, "numpy_second_order_sobi")
        self.assertEqual(result.method_label, "SOBI")
        self.assertGreater(_best_mean_abs_correlation(sources, result.source_angles), 0.95)
        self.assertEqual(result.diagnostics["sobi_lags_used"], list(BSSConfig().sobi_lags))
        history = result.diagnostics["offdiag_energy_history"]
        self.assertLess(float(history[-1]), float(history[0]))

    def test_sobi_does_not_separate_temporally_white_sources(self) -> None:
        sources, observations = self._instantaneous_mixture()
        result = fit_bss(observations, "sobi", BSSConfig(sobi_tol=1e-10))
        # SOBI is a second-order method, so white sources carry no information it can use.
        self.assertLess(_best_mean_abs_correlation(sources, result.source_angles), 0.95)
        self.assertTrue(np.isfinite(result.source_angles).all())

    def test_sobi_separates_sources_with_distinct_autocorrelation(self) -> None:
        sources, observations = self._colored_mixture()
        result = fit_bss(
            observations,
            "sobi",
            BSSConfig(sobi_lags=(1, 2, 3, 5, 7, 10), sobi_max_sweeps=200, sobi_tol=1e-12),
        )
        self.assertGreater(_best_mean_abs_correlation(sources, result.source_angles), 0.95)

    def test_sobi_configuration_validation(self) -> None:
        for invalid in ({"sobi_lags": ()}, {"sobi_lags": (0, 2)}, {"sobi_lags": (1, 1)},
                        {"sobi_max_sweeps": 0}, {"sobi_tol": 0.0}):
            with self.assertRaises(ValueError):
                BSSConfig(**invalid)
        self.assertEqual(BSSConfig(methods=("sobi",)).methods, ("sobi",))

    def test_sobi_drops_lags_longer_than_the_record(self) -> None:
        _, observations = self._colored_mixture(sample_count=16)
        result = fit_bss(observations, "sobi", BSSConfig(sobi_lags=(1, 2, 20)))
        self.assertTrue(result.warnings)
        self.assertEqual(result.diagnostics["sobi_lags_used"], [1, 2])
        with self.assertRaises(ValueError):
            fit_bss(observations, "sobi", BSSConfig(sobi_lags=(20,)))

    def test_fastica_nonconvergence_is_recorded_without_discarding_output(self) -> None:
        _, observations = self._instantaneous_mixture(sample_count=500)
        result = fit_bss(
            observations,
            "fastica",
            BSSConfig(fastica_max_iter=1, fastica_tol=1e-12),
        )
        self.assertFalse(result.converged)
        self.assertTrue(result.warnings)
        self.assertTrue(np.isfinite(result.source_angles).all())


class BSSValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = generate_validation_scenario(SimulationConfig(seed=3, T=96))
        self.results = run_bss_methods_for_scenario(
            self.scenario,
            BSSConfig(fastica_max_iter=300),
        )

    def test_evaluation_reports_physical_and_oracle_metrics(self) -> None:
        self.assertEqual(len(self.results), len(DEFAULT_METHODS))
        metrics = build_bss_metrics_frame(self.results)
        sources = build_bss_source_metrics_frame(self.results)
        self.assertEqual(set(metrics["method_key"]), set(DEFAULT_METHODS))
        self.assertEqual(len(sources), 3 * len(DEFAULT_METHODS))
        required_case_columns = {
            "mean_abs_source_correlation",
            "mean_source_rmse",
            "mean_source_rmse_oracle_scaled",
            "mean_axis_error_deg",
            "linear_observation_reconstruction_rmse",
            "quaternion_reconstruction_mean_error_deg",
            "quaternion_reconstruction_mean_error_deg_oracle_scaled",
        }
        self.assertTrue(required_case_columns.issubset(metrics.columns))
        self.assertTrue(
            np.all(
                metrics["mean_source_rmse_oracle_scaled"].to_numpy(dtype=float)
                <= metrics["mean_source_rmse"].to_numpy(dtype=float) + 1e-12
            )
        )
        self.assertTrue(
            np.all(
                sources["source_rmse_rad_oracle_scaled"].to_numpy(dtype=float)
                <= sources["source_rmse_rad"].to_numpy(dtype=float) + 1e-12
            )
        )
        self.assertLess(float(metrics["linear_observation_reconstruction_rmse"].max()), 1e-10)

    def test_comparison_frames_include_rss_and_all_baselines(self) -> None:
        rss_result = run_validation_for_scenario(
            self.scenario,
            RSSConfig(
                epochs=1,
                optimizer="adam",
                observer_signal_model="integrated_body_rotation",
            ),
        )
        cases, sources = build_method_comparison_frames([rss_result], self.results)
        expected_methods = {"RSS", "PCA", "ICA (JADE)", "FastICA", "SOBI"}
        self.assertEqual(set(cases["method"]), expected_methods)
        self.assertEqual(set(sources["method"]), expected_methods)
        self.assertTrue(cases["mean_source_rmse_oracle_scaled"].notna().all())
        self.assertTrue(sources["source_rmse_rad_oracle_scaled"].notna().all())

    def test_case_artifact_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            payload = export_bss_case_artifacts(self.results[0], output, include_plots=True)
            expected = {
                "metrics.json",
                "demixing_matrix.csv",
                "normalized_axes.csv",
                "source_comparison.csv",
                "observer_reconstruction.csv",
                "source_comparison.png",
                "observer_reconstruction.png",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            saved = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["method"]["input_domain"], "cumulative_observer_signals")
            self.assertIn("oracle_source_metrics", saved["match"])
            self.assertIn("sobi_lags", saved["method"]["config"])
            self.assertEqual(payload["method"]["key"], "pca")


if __name__ == "__main__":
    unittest.main()
