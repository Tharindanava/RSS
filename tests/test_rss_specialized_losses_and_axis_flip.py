from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch

from rss import (
    CHIRP_TIME_FREQUENCY_OVERLAP,
    FILTERED_MULTILAG_CROSS_CORRELATION,
    HARMONIC_SPECTRAL_OVERLAP,
    PIECEWISE_SPARSE_INNOVATION,
    RSSConfig,
    SimulationConfig,
    SpecializedLossConfig,
    compute_specialized_loss,
    fit_rss,
    generate_validation_scenario,
    run_validation,
)
from rss.cli import _rss_config, build_parser
from rss.realdata_config import build_realdata_runtime_config_from_args
from rss.solver import _AdaptiveRegularizerScheduler


class SpecializedLossTests(unittest.TestCase):
    def test_promoted_losses_preserve_notebook_ordering_and_guards(self) -> None:
        sample_rate = 100.0
        time = np.arange(512, dtype=float) / sample_rate
        distinct = np.stack(
            [np.sin(2.0 * np.pi * frequency * time) for frequency in (0.5, 1.5, 3.0)]
        )
        identical = np.repeat(distinct[:1], 3, axis=0)
        self.assertLess(
            compute_specialized_loss(
                distinct, sample_rate, HARMONIC_SPECTRAL_OVERLAP
            ),
            compute_specialized_loss(
                identical, sample_rate, HARMONIC_SPECTRAL_OVERLAP
            ),
        )
        chirps = np.stack(
            [
                np.sin(
                    2.0
                    * np.pi
                    * (start * time + 0.5 * slope * np.square(time))
                )
                for start, slope in ((0.2, 0.3), (0.8, 0.5), (1.6, 0.6))
            ]
        )
        identical_chirps = np.repeat(chirps[:1], 3, axis=0)
        self.assertLess(
            compute_specialized_loss(
                chirps, sample_rate, CHIRP_TIME_FREQUENCY_OVERLAP
            ),
            compute_specialized_loss(
                identical_chirps, sample_rate, CHIRP_TIME_FREQUENCY_OVERLAP
            ),
        )

        rng = np.random.default_rng(123)
        independent = rng.normal(size=(3, 512))
        delayed = independent.copy()
        delayed[1, 8:] = delayed[0, :-8]
        delayed[1, :8] = 0.0
        filtered_config = replace(SpecializedLossConfig(), filtered_max_lag=16)
        self.assertLess(
            compute_specialized_loss(
                independent,
                sample_rate,
                FILTERED_MULTILAG_CROSS_CORRELATION,
                filtered_config,
            ),
            compute_specialized_loss(
                delayed,
                sample_rate,
                FILTERED_MULTILAG_CROSS_CORRELATION,
                filtered_config,
            ),
        )

        separate_innovations = np.zeros((3, 256))
        shared_innovations = np.zeros((3, 256))
        for index, location in enumerate((40, 120, 200)):
            separate_innovations[index, location] = 1.0
        shared_innovations[:, 120] = 1.0
        separate = np.cumsum(np.cumsum(separate_innovations, axis=1), axis=1)
        shared = np.cumsum(np.cumsum(shared_innovations, axis=1), axis=1)
        self.assertLess(
            compute_specialized_loss(
                separate, sample_rate, PIECEWISE_SPARSE_INNOVATION
            ),
            compute_specialized_loss(
                shared, sample_rate, PIECEWISE_SPARSE_INNOVATION
            ),
        )
        self.assertEqual(
            compute_specialized_loss(
                np.ones((3, 32)), sample_rate, HARMONIC_SPECTRAL_OVERLAP
            ),
            SpecializedLossConfig().bad_loss,
        )
        nonfinite = distinct.copy()
        nonfinite[0, 10] = np.nan
        self.assertEqual(
            compute_specialized_loss(
                nonfinite, sample_rate, HARMONIC_SPECTRAL_OVERLAP
            ),
            SpecializedLossConfig().bad_loss,
        )
        with self.assertRaises(ValueError):
            compute_specialized_loss(
                np.ones((3, 3)), sample_rate, HARMONIC_SPECTRAL_OVERLAP
            )

    def test_specialized_losses_support_batches_and_autograd(self) -> None:
        generator = torch.Generator().manual_seed(7)
        values = torch.randn(
            (2, 3, 64), dtype=torch.float64, generator=generator, requires_grad=True
        )
        for loss_type in (
            HARMONIC_SPECTRAL_OVERLAP,
            FILTERED_MULTILAG_CROSS_CORRELATION,
            CHIRP_TIME_FREQUENCY_OVERLAP,
            PIECEWISE_SPARSE_INNOVATION,
        ):
            with self.subTest(loss_type=loss_type):
                loss = compute_specialized_loss(values, 100.0, loss_type)
                self.assertEqual(tuple(loss.shape), (2,))
                gradient = torch.autograd.grad(loss.sum(), values, retain_graph=True)[0]
                self.assertTrue(torch.isfinite(gradient).all())

                invalid = torch.ones(
                    (3, 64), dtype=torch.float64, requires_grad=True
                )
                invalid_loss = compute_specialized_loss(
                    invalid, 100.0, loss_type
                )
                invalid_gradient = torch.autograd.grad(
                    invalid_loss, invalid
                )[0]
                self.assertEqual(
                    float(invalid_loss.detach()), SpecializedLossConfig().bad_loss
                )
                self.assertTrue(torch.isfinite(invalid_gradient).all())


class EnhancedRSSConfigTests(unittest.TestCase):
    def test_cli_parses_all_enhancement_controls(self) -> None:
        args = build_parser().parse_args(
            [
                "validate",
                "--rss-axis-flip-search",
                "--rss-axis-flip-interval",
                "30",
                "--rss-axis-flip-trial-epochs",
                "4",
                "--rss-adaptive-regularizer",
                "--rss-specialized-regularizer",
                CHIRP_TIME_FREQUENCY_OVERLAP,
                "--rss-regularizer-start-weight",
                "0.25",
                "--rss-regularizer-plateau-patience",
                "7",
                "--rss-regularizer-plateau-tolerance",
                "0.002",
                "--rss-regularizer-ramp-epochs",
                "5",
                "--rss-regularizer-min-loss-threshold",
                "0.01",
            ]
        )
        config = _rss_config(args)
        self.assertTrue(config.axis_flip_search)
        self.assertEqual(config.axis_flip_interval, 30)
        self.assertEqual(config.axis_flip_trial_epochs, 4)
        self.assertTrue(config.adaptive_regularizer)
        self.assertEqual(config.specialized_regularizer, CHIRP_TIME_FREQUENCY_OVERLAP)
        self.assertEqual(config.regularizer_start_weight, 0.25)
        self.assertEqual(config.regularizer_plateau_patience, 7)
        self.assertEqual(config.regularizer_plateau_tolerance, 0.002)
        self.assertEqual(config.regularizer_ramp_epochs, 5)
        self.assertEqual(config.regularizer_min_loss_threshold, 0.01)

        real_args = build_parser().parse_args(
            [
                "run-config",
                "--config",
                "unused.json",
                "--rss-axis-flip-search",
                "--rss-adaptive-regularizer",
                "--rss-specialized-regularizer",
                FILTERED_MULTILAG_CROSS_CORRELATION,
            ]
        )
        real_config = build_realdata_runtime_config_from_args(real_args).rss_config
        self.assertTrue(real_config.axis_flip_search)
        self.assertTrue(real_config.adaptive_regularizer)
        self.assertEqual(
            real_config.specialized_regularizer,
            FILTERED_MULTILAG_CROSS_CORRELATION,
        )

    def test_invalid_feature_combinations_are_rejected(self) -> None:
        invalid = (
            {"adaptive_regularizer": True},
            {
                "adaptive_regularizer": True,
                "specialized_regularizer": HARMONIC_SPECTRAL_OVERLAP,
                "loss_type": "action",
            },
            {"axis_flip_search": True, "matrix_parameterization": "symmetric"},
            {
                "axis_flip_search": True,
                "axis_flip_interval": 10,
                "axis_flip_trial_epochs": 10,
            },
            {"regularizer_start_weight": 1.1},
            {"regularizer_plateau_patience": 0},
            {"regularizer_ramp_epochs": 0},
            {"regularizer_min_loss_threshold": -1.0},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                RSSConfig(**values)


class AdaptiveSchedulerTests(unittest.TestCase):
    def test_scheduler_completes_and_repeats_up_down_cycle(self) -> None:
        config = RSSConfig(
            adaptive_regularizer=True,
            specialized_regularizer=HARMONIC_SPECTRAL_OVERLAP,
            regularizer_start_weight=0.25,
            regularizer_plateau_patience=2,
            regularizer_plateau_tolerance=1e-4,
            regularizer_ramp_epochs=2,
        )
        scheduler = _AdaptiveRegularizerScheduler(
            config, base_loss=1.0, regularizer_loss=1.0
        )
        scheduler.update(base_loss=1.0, regularizer_loss=1.0)
        scheduler.update(base_loss=1.0, regularizer_loss=1.0)
        self.assertEqual(scheduler.phase, "ramp_up")
        scheduler.update(base_loss=1.0, regularizer_loss=1.0)
        self.assertAlmostEqual(scheduler.weight, 0.625)
        scheduler.update(base_loss=1.0, regularizer_loss=1.0)
        self.assertEqual((scheduler.phase, scheduler.weight), ("regularizer_monitor", 1.0))
        scheduler.update(base_loss=1.0, regularizer_loss=1.0)
        scheduler.update(base_loss=1.0, regularizer_loss=1.0)
        self.assertEqual(scheduler.phase, "ramp_down")
        scheduler.update(base_loss=1.0, regularizer_loss=1.0)
        scheduler.update(base_loss=1.0, regularizer_loss=1.0)
        self.assertEqual((scheduler.phase, scheduler.weight), ("base_monitor", 0.0))


class EnhancedRSSIntegrationTests(unittest.TestCase):
    def test_adaptive_regularizer_runs_with_all_optimizers(self) -> None:
        scenario = generate_validation_scenario(
            SimulationConfig(seed=44, T=48, duration_seconds=0.48)
        )
        for optimizer in ("adam", "sgd", "pso"):
            with self.subTest(optimizer=optimizer):
                result = fit_rss(
                    scenario.observations,
                    RSSConfig(
                        optimizer=optimizer,
                        epochs=3,
                        learning_rate=1e-3,
                        pso_swarm_size=4,
                        adaptive_regularizer=True,
                        specialized_regularizer=HARMONIC_SPECTRAL_OVERLAP,
                        observer_signal_model="integrated_body_rotation",
                    ),
                    sample_interval_seconds=0.01,
                )
                self.assertEqual(len(result.loss_history), 4)
                self.assertIsNotNone(result.best_selection_loss)
                self.assertIn("regularizer_loss", result.loss_history[0])
                self.assertTrue(np.isfinite(result.recovery_matrix).all())

    def test_normalized_selection_threshold_has_distinct_stop_reason(self) -> None:
        result = run_validation(
            SimulationConfig(seed=45, T=32, duration_seconds=0.32),
            RSSConfig(
                epochs=10,
                adaptive_regularizer=True,
                specialized_regularizer=HARMONIC_SPECTRAL_OVERLAP,
                regularizer_min_loss_threshold=1.0,
                observer_signal_model="integrated_body_rotation",
            ),
        ).rss_result
        self.assertEqual(result.stop_reason, "minimum_selection_loss")
        self.assertEqual(len(result.loss_history), 1)

    def test_combined_features_run_in_euler_mode_and_freeze_trial_schedule(self) -> None:
        sample_count = 48
        time = np.arange(sample_count, dtype=float) / 100.0
        observations = np.stack(
            (
                np.sin(2.0 * np.pi * 0.8 * time),
                np.sin(2.0 * np.pi * 1.7 * time + 0.2),
                np.sin(2.0 * np.pi * 2.9 * time + 0.4),
            )
        )
        for optimizer in ("adam", "sgd", "pso"):
            with self.subTest(optimizer=optimizer):
                result = fit_rss(
                    observations,
                    RSSConfig(
                        optimizer=optimizer,
                        epochs=6,
                        learning_rate=1e-3,
                        pso_swarm_size=4,
                        pso_seed=19,
                        observer_signal_model="euler",
                        axis_flip_search=True,
                        axis_flip_interval=3,
                        axis_flip_trial_epochs=2,
                        adaptive_regularizer=True,
                        specialized_regularizer=HARMONIC_SPECTRAL_OVERLAP,
                        regularizer_plateau_patience=1,
                        regularizer_plateau_tolerance=1e9,
                        regularizer_ramp_epochs=2,
                    ),
                    sample_interval_seconds=0.01,
                )
                self.assertEqual(len(result.loss_history), 7)
                self.assertEqual(len(result.axis_flip_events), 1)
                self.assertEqual(
                    result.loss_history[0]["selection_stop_threshold"], 0.0
                )
                trial_rows = result.loss_history[4:6]
                self.assertEqual(
                    len({row["regularizer_weight"] for row in trial_rows}), 1
                )
                self.assertEqual(
                    len({row["regularizer_phase"] for row in trial_rows}), 1
                )
                for row in result.loss_history:
                    expected_selection = (
                        row["separation_loss"]
                        + row["orthogonality_loss"]
                        * result.config.orthogonality_weight
                        + row["reconstruction_loss"]
                        * result.config.reconstruction_weight
                        + row["regularizer_loss"]
                    )
                    self.assertAlmostEqual(row["selection_loss"], expected_selection)

    def test_axis_flip_search_checks_four_proper_candidates_within_budget(self) -> None:
        for optimizer in ("adam", "sgd", "pso"):
            with self.subTest(optimizer=optimizer):
                config = RSSConfig(
                    optimizer=optimizer,
                    epochs=7,
                    learning_rate=1e-3,
                    pso_swarm_size=4,
                    pso_seed=11,
                    axis_flip_search=True,
                    axis_flip_interval=3,
                    axis_flip_trial_epochs=1,
                    observer_signal_model="integrated_body_rotation",
                )
                first = run_validation(
                    SimulationConfig(seed=46, T=40, duration_seconds=0.4), config
                ).rss_result
                second = run_validation(
                    SimulationConfig(seed=46, T=40, duration_seconds=0.4), config
                ).rss_result
                self.assertEqual(len(first.loss_history), config.epochs + 1)
                self.assertEqual(len(first.applied_learning_rates), config.epochs)
                self.assertEqual(len(first.axis_flip_events), 2)
                self.assertEqual(
                    [candidate.pattern for candidate in first.axis_flip_events[0].candidates],
                    ["identity", "xy", "xz", "yz"],
                )
                self.assertIn(
                    "identity=",
                    first.loss_history[4]["axis_flip_candidate_selection_losses"],
                )
                for event in first.axis_flip_events:
                    self.assertEqual(sum(candidate.selected for candidate in event.candidates), 1)
                    for candidate in event.candidates:
                        self.assertEqual(np.prod(candidate.signs), 1)
                self.assertAlmostEqual(np.linalg.det(first.recovery_matrix), 1.0, places=10)
                self.assertTrue(
                    np.array_equal(first.optimized_parameters, second.optimized_parameters)
                )


if __name__ == "__main__":
    unittest.main()
