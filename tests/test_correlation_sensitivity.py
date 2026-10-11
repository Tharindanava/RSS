from __future__ import annotations

import re
import unittest

import numpy as np

from bss_baselines import BSSConfig
from bss_baselines.correlation_sensitivity import (
    CRITERION_ABSOLUTE,
    CRITERION_RELATIVE_DEGRADATION,
    STATUS_ABOVE_GRID,
    STATUS_ATTAINED,
    STATUS_FLOOR_EXCEEDED,
    CorrelationSweepConfig,
    attach_correlation_columns,
    block_bootstrap_interval,
    estimate_correlation_thresholds,
    estimate_threshold_crossing,
    run_correlation_sensitivity_sweep,
)
from rss import (
    BatchStudyConfig,
    CorrelationStudyConfig,
    RSSConfig,
    SimulationConfig,
    correlate_source_angles,
    create_correlation_case_specs,
    create_standard_case_specs,
    equicorrelation_matrix,
    generate_correlation_scenario,
    generate_family_scenario,
    orthonormality_error,
    source_cross_correlation_matrix,
)
from rss.simulation import _decorrelate_source_angles


def _reference_sources(sample_count: int = 512) -> np.ndarray:
    time = np.arange(sample_count, dtype=float) * 0.01
    rng = np.random.default_rng(5)
    return np.vstack(
        (
            0.6 * np.sin(2.0 * np.pi * 2.0 * time) + 0.05 * rng.standard_normal(sample_count),
            0.4 * np.sin(2.0 * np.pi * 5.0 * time + 0.7) + 0.05 * rng.standard_normal(sample_count),
            0.5 * np.sin(2.0 * np.pi * 11.0 * time + 1.3) + 0.05 * rng.standard_normal(sample_count),
        )
    )


def _offdiagonals(signals: np.ndarray) -> np.ndarray:
    matrix = source_cross_correlation_matrix(signals)
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


class CorrelationColoringTests(unittest.TestCase):
    def test_coloring_hits_target_correlation(self) -> None:
        sources = _reference_sources()
        for correlation in (-0.4, -0.2, 0.0, 0.25, 0.5, 0.85):
            colored = correlate_source_angles(sources, correlation)
            deviation = float(np.max(np.abs(_offdiagonals(colored) - correlation)))
            self.assertLess(deviation, 1e-9)

    def test_coloring_preserves_means_and_peaks(self) -> None:
        sources = _reference_sources()
        colored = correlate_source_angles(sources, 0.5)
        self.assertTrue(
            np.allclose(np.mean(colored, axis=1), np.mean(sources, axis=1), rtol=0.0, atol=1e-12)
        )
        original_peaks = np.max(np.abs(sources - np.mean(sources, axis=1, keepdims=True)), axis=1)
        colored_peaks = np.max(np.abs(colored - np.mean(colored, axis=1, keepdims=True)), axis=1)
        self.assertTrue(np.allclose(colored_peaks, original_peaks, rtol=0.0, atol=1e-12))
        self.assertTrue(np.isfinite(colored).all())

    def test_coloring_preserves_zero_mean_inputs(self) -> None:
        sources = _reference_sources()
        centered = sources - np.mean(sources, axis=1, keepdims=True)
        colored = correlate_source_angles(centered, 0.3)
        self.assertTrue(np.allclose(np.mean(colored, axis=1), 0.0, rtol=0.0, atol=1e-12))

    def test_coloring_at_zero_matches_the_existing_decorrelation(self) -> None:
        sources = _reference_sources()
        colored = correlate_source_angles(sources, 0.0)
        whitened = _decorrelate_source_angles(sources, cross_correlation_limit=0.01)
        self.assertTrue(np.allclose(colored, whitened, rtol=0.0, atol=1e-12))

    def test_coloring_accepts_an_explicit_matrix_target(self) -> None:
        sources = _reference_sources()
        target = np.array([[1.0, 0.4, 0.0], [0.4, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        colored = correlate_source_angles(sources, target)
        realized = source_cross_correlation_matrix(colored)
        self.assertTrue(np.allclose(realized, target, rtol=0.0, atol=1e-9))

    def test_coloring_rejects_invalid_targets(self) -> None:
        sources = _reference_sources()
        for invalid in (1.0, -0.5, -0.7, float("nan")):
            with self.assertRaises(ValueError):
                correlate_source_angles(sources, invalid)
        with self.assertRaises(ValueError):
            correlate_source_angles(sources, np.array([[1.0, 0.4], [0.4, 1.0]]))
        with self.assertRaises(ValueError):
            correlate_source_angles(sources, np.array([[1.0, 0.4, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]]))

    def test_coloring_rejects_dependent_sources(self) -> None:
        base = _reference_sources()
        dependent = np.vstack((base[0], base[1], base[0] + base[1]))
        with self.assertRaises(ValueError):
            correlate_source_angles(dependent, 0.2)

    def test_equicorrelation_matrix_properties(self) -> None:
        self.assertTrue(np.allclose(equicorrelation_matrix(3, 0.0), np.eye(3)))
        matrix = equicorrelation_matrix(3, 0.6)
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.allclose(np.diag(matrix), 1.0))
        self.assertTrue(np.all(np.linalg.eigvalsh(matrix) > 0.0))
        for invalid in (1.0, -0.5, -0.9, float("inf")):
            with self.assertRaises(ValueError):
                equicorrelation_matrix(3, invalid)


class CorrelationScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.study = BatchStudyConfig()
        # A strict limit that the sweep must legitimately bypass.
        self.base = SimulationConfig(seed=0, T=720, source_cross_correlation_limit=0.01)

    def test_scenario_realizes_the_target_and_bypasses_the_limit(self) -> None:
        config = CorrelationStudyConfig(
            target_correlations=(0.6,), families=("harmonic",), replicates=1
        )
        spec = create_correlation_case_specs(config, self.study)[0]
        scenario = generate_correlation_scenario(spec, self.base, self.study)
        deviation = float(np.max(np.abs(_offdiagonals(scenario.sources) - 0.6)))
        self.assertLess(deviation, 1e-8)
        self.assertTrue(np.allclose(np.mean(scenario.sources, axis=1), 0.0, rtol=0.0, atol=1e-12))
        self.assertLess(orthonormality_error(scenario.mixing_matrix), 1e-10)
        self.assertGreater(float(np.linalg.det(scenario.mixing_matrix)), 0.0)
        self.assertEqual(scenario.replicate_index, spec.replicate_index)
        self.assertEqual(scenario.source_roles, ("dense", "dense", "dense"))

    def test_blocks_are_invariant_across_correlation_levels(self) -> None:
        config = CorrelationStudyConfig(
            target_correlations=(0.0, 0.01, 0.3, 0.9), families=("harmonic",), replicates=1
        )
        scenarios = [
            generate_correlation_scenario(spec, self.base, self.study)
            for spec in create_correlation_case_specs(config, self.study)
        ]
        reference = scenarios[0]
        for scenario in scenarios[1:]:
            # Coloring consumes no RNG, so the geometry must be bit-identical.
            self.assertTrue(np.array_equal(scenario.mixing_matrix, reference.mixing_matrix))
            self.assertTrue(np.array_equal(scenario.source_axes_global, reference.source_axes_global))
            self.assertTrue(np.array_equal(scenario.observer_axes_global, reference.observer_axes_global))

    def test_case_specs_are_deterministic_unique_and_path_safe(self) -> None:
        config = CorrelationStudyConfig(
            target_correlations=(0.0, 0.1), families=("harmonic", "filtered_noise"), replicates=3
        )
        specs = create_correlation_case_specs(config, self.study)
        self.assertEqual(len(specs), 2 * 2 * 3)
        self.assertEqual(len({spec.case_id for spec in specs}), len(specs))
        for spec in specs:
            self.assertRegex(spec.case_id, re.compile(r"^[A-Za-z0-9_]+$"))
        self.assertEqual(specs, create_correlation_case_specs(config, self.study))
        # Seeds are shared within a block and distinct across blocks.
        blocks = {spec.block_id: set() for spec in specs}
        for spec in specs:
            blocks[spec.block_id].add(spec.seed)
        self.assertTrue(all(len(seeds) == 1 for seeds in blocks.values()))
        self.assertEqual(len({spec.seed for spec in specs}), len(blocks))

    def test_case_ids_stay_distinct_at_very_fine_correlation_spacing(self) -> None:
        config = CorrelationStudyConfig(
            target_correlations=(0.0, 1e-7, 2e-7, 1e-6, 1e-3, -1e-7),
            families=("harmonic",),
            replicates=1,
        )
        specs = create_correlation_case_specs(config, self.study)
        identifiers = [spec.case_id for spec in specs]
        self.assertEqual(len(set(identifiers)), len(identifiers))
        for spec in specs:
            self.assertRegex(spec.case_id, re.compile(r"^[A-Za-z0-9_]+$"))

    def test_correlation_seeds_do_not_collide_with_existing_suites(self) -> None:
        specs = create_correlation_case_specs(CorrelationStudyConfig(), self.study)
        standard = {spec.seed for spec in create_standard_case_specs(self.study)}
        formal = {
            spec.seed
            for spec in create_standard_case_specs(self.study, base_seed=20000, case_id_prefix="formal_")
        }
        correlation = {spec.seed for spec in specs}
        self.assertFalse(correlation & standard)
        self.assertFalse(correlation & formal)

    def test_existing_generators_still_enforce_the_limit(self) -> None:
        spec = create_standard_case_specs(self.study)[0]
        scenario = generate_family_scenario(spec, self.base, self.study)
        pairwise = np.abs(_offdiagonals(scenario.sources))
        self.assertLess(float(np.max(pairwise)), scenario.config.source_cross_correlation_limit)

    def test_study_config_validation(self) -> None:
        for invalid in (
            {"target_correlations": ()},
            {"target_correlations": (0.0, 0.0)},
            {"target_correlations": (0.99,)},
            {"target_correlations": (-0.6,)},
            {"families": ()},
            {"families": ("unknown",)},
            {"replicates": 0},
            {"dt": 0.0},
        ):
            with self.assertRaises(ValueError):
                CorrelationStudyConfig(**invalid)


class ThresholdEstimatorTests(unittest.TestCase):
    def test_crossing_on_synthetic_monotone_data(self) -> None:
        x = np.linspace(0.0, 1.0, 11)
        crossing, status = estimate_threshold_crossing(x, 2.0 * x, 1.0)
        self.assertEqual(status, STATUS_ATTAINED)
        self.assertAlmostEqual(crossing, 0.5, places=12)

    def test_crossing_reports_floor_and_grid_limits(self) -> None:
        x = np.linspace(0.0, 1.0, 11)
        _, above = estimate_threshold_crossing(x, 2.0 * x, 5.0)
        self.assertEqual(above, STATUS_ABOVE_GRID)
        _, floor = estimate_threshold_crossing(x, 2.0 * x + 10.0, 1.0)
        self.assertEqual(floor, STATUS_FLOOR_EXCEEDED)

    def test_crossing_uses_logarithmic_interpolation_between_positive_levels(self) -> None:
        x = np.array([0.0, 1e-3, 1e-2])
        y = np.array([0.0, 0.0, 2.0])
        crossing, status = estimate_threshold_crossing(x, y, 1.0)
        self.assertEqual(status, STATUS_ATTAINED)
        # Halfway between the levels on a log axis, not on a linear one.
        self.assertAlmostEqual(crossing, 10.0 ** (-3.0 + 0.5), places=12)

    def test_crossing_returns_the_first_upcrossing(self) -> None:
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 2.0, 0.0, 2.0])
        crossing, status = estimate_threshold_crossing(x, y, 1.0)
        self.assertEqual(status, STATUS_ATTAINED)
        self.assertAlmostEqual(crossing, 0.5, places=12)

    def test_bootstrap_is_deterministic_and_brackets_the_point_estimate(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        first = block_bootstrap_interval(values, resamples=200, random_state=3)
        second = block_bootstrap_interval(values, resamples=200, random_state=3)
        self.assertEqual(first, second)
        point, lower, upper = first
        self.assertLessEqual(lower, point)
        self.assertLessEqual(point, upper)
        constant = block_bootstrap_interval(np.full(4, 2.5), resamples=50, random_state=0)
        self.assertEqual(constant, (2.5, 2.5, 2.5))
        self.assertTrue(all(np.isnan(value) for value in block_bootstrap_interval(np.array([np.nan]))))

    def test_threshold_frame_separates_absolute_and_degradation_criteria(self) -> None:
        import pandas as pd

        rows = []
        for block in range(4):
            for correlation in (0.0, 1e-3, 1e-2, 1e-1):
                rows.append(
                    {
                        "method": "RSS",
                        "block_id": f"b{block}",
                        "target_source_correlation": correlation,
                        "mean_axis_error_deg": 10.0 + 100.0 * correlation,
                    }
                )
        frame = pd.DataFrame(rows)
        thresholds = estimate_correlation_thresholds(
            frame,
            metric_columns=("mean_axis_error_deg",),
            absolute_thresholds=(0.01,),
            relative_degradations=(0.01,),
            resamples=100,
        )
        absolute = thresholds.loc[thresholds["criterion"] == CRITERION_ABSOLUTE].iloc[0]
        self.assertEqual(absolute["status"], STATUS_FLOOR_EXCEEDED)
        self.assertTrue(np.isnan(absolute["threshold_correlation"]))
        degradation = thresholds.loc[
            thresholds["criterion"] == CRITERION_RELATIVE_DEGRADATION
        ].iloc[0]
        self.assertEqual(degradation["status"], STATUS_ATTAINED)
        # A 1% rise above a floor of 10 deg needs 0.1 deg, i.e. rho = 1e-3.
        self.assertAlmostEqual(float(degradation["threshold_correlation"]), 1e-3, places=9)
        self.assertAlmostEqual(float(degradation["baseline_error_at_zero"]), 10.0, places=12)


class CorrelationSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.study = CorrelationStudyConfig(
            target_correlations=(0.0, 0.1, 0.6), families=("harmonic",), replicates=2
        )
        cls.sweep_config = CorrelationSweepConfig(
            study=cls.study, bootstrap_resamples=50, retain_case_results=True
        )
        cls.result = run_correlation_sensitivity_sweep(
            RSSConfig(epochs=1, optimizer="adam", observer_signal_model="integrated_body_rotation"),
            BSSConfig(fastica_max_iter=100),
            cls.sweep_config,
            base_config=SimulationConfig(seed=0, T=96, source_cross_correlation_limit=0.01),
            study_config=BatchStudyConfig(batch_sequence_lengths=(96,)),
        )

    def test_frame_has_one_row_per_method_correlation_and_replicate(self) -> None:
        frame = self.result.case_frame
        method_count = len(BSSConfig().methods) + 1
        self.assertEqual(len(frame), method_count * 3 * 2)
        self.assertEqual(len(self.result.source_frame), 3 * len(frame))
        self.assertIn("RSS", set(frame["method"]))
        self.assertEqual(sorted(set(frame["target_source_correlation"])), [0.0, 0.1, 0.6])
        sizes = set(frame.groupby(["method", "target_source_correlation"]).size())
        self.assertEqual(sizes, {2})
        self.assertFalse(frame["target_source_correlation"].isna().any())
        self.assertEqual(len(set(frame["block_id"])), 2)

    def test_attached_target_matches_the_realized_correlation(self) -> None:
        frame = self.result.case_frame
        # The recorded column is a mean of absolute pair correlations, so it is compared
        # against |target|; every pair of an equicorrelated target sits at exactly rho.
        deviation = (
            frame["target_source_correlation"].abs()
            - frame["original_source_mean_abs_cross_correlation"]
        ).abs().max()
        self.assertLess(float(deviation), 1e-6)

    def test_negative_correlation_levels_round_trip(self) -> None:
        config = CorrelationSweepConfig(
            study=CorrelationStudyConfig(
                target_correlations=(-0.4, 0.0, 0.4), families=("harmonic",), replicates=1
            ),
            include_rss=False,
            bootstrap_resamples=20,
            retain_case_results=True,
        )
        result = run_correlation_sensitivity_sweep(
            None,
            BSSConfig(fastica_max_iter=100),
            config,
            base_config=SimulationConfig(seed=0, T=96, source_cross_correlation_limit=0.01),
            study_config=BatchStudyConfig(batch_sequence_lengths=(96,)),
        )
        self.assertEqual(sorted(set(result.case_frame["target_source_correlation"])), [-0.4, 0.0, 0.4])
        deviation = (
            result.case_frame["target_source_correlation"].abs()
            - result.case_frame["original_source_mean_abs_cross_correlation"]
        ).abs().max()
        self.assertLess(float(deviation), 1e-6)
        for outcome in result.outcomes:
            realized = _offdiagonals(outcome.scenario.sources)
            self.assertLess(
                float(np.max(np.abs(realized - outcome.spec.target_source_correlation))), 1e-8
            )

    def test_attach_rejects_unmapped_case_ids(self) -> None:
        frame = self.result.case_frame.copy()
        frame.loc[frame.index[0], "case_id"] = "not_a_sweep_case"
        with self.assertRaises(ValueError):
            attach_correlation_columns(frame, self.result.outcomes)

    def test_summary_and_threshold_frames_are_populated(self) -> None:
        summary = self.result.case_summary
        expected = {
            "method",
            "target_source_correlation",
            "metric",
            "count",
            "mean",
            "median",
            "ci_lower",
            "ci_upper",
        }
        self.assertTrue(expected.issubset(summary.columns))
        self.assertTrue((summary["count"] == 2).all())
        thresholds = self.result.thresholds
        self.assertTrue({"method", "metric", "criterion", "status"}.issubset(thresholds.columns))
        self.assertEqual(
            set(thresholds["criterion"]), {CRITERION_ABSOLUTE, CRITERION_RELATIVE_DEGRADATION}
        )

    def test_paired_tests_compare_rss_against_every_baseline(self) -> None:
        paired = self.result.paired_tests
        self.assertFalse(paired.empty)
        self.assertEqual(set(paired["reference_method"]), {"RSS"})
        self.assertEqual(set(paired["method"]), {"PCA", "ICA (JADE)", "FastICA", "SOBI"})
        self.assertTrue((paired["block_count"] > 0).all())

    def test_sweep_without_rss_contains_only_baselines(self) -> None:
        config = CorrelationSweepConfig(
            study=CorrelationStudyConfig(
                target_correlations=(0.0, 0.3), families=("harmonic",), replicates=1
            ),
            include_rss=False,
            bootstrap_resamples=20,
            retain_case_results=True,
        )
        result = run_correlation_sensitivity_sweep(
            None,
            BSSConfig(fastica_max_iter=100),
            config,
            base_config=SimulationConfig(seed=0, T=96, source_cross_correlation_limit=0.01),
            study_config=BatchStudyConfig(batch_sequence_lengths=(96,)),
        )
        self.assertNotIn("RSS", set(result.case_frame["method"]))
        self.assertTrue(all(outcome.rss_result is None for outcome in result.outcomes))


if __name__ == "__main__":
    unittest.main()
