from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from scipy.special import digamma

from rss import (
    BatchStudyConfig,
    FormalStudyConfig,
    RSSConfig,
    SimulationConfig,
    angles_to_quaternion_components,
    build_source_metrics_frame,
    compose_source_quaternions,
    compose_source_quaternions_torch,
    compute_rss_loss,
    create_formal_case_specs,
    create_standard_case_specs,
    create_stress_case_specs,
    decode_scalar_and_vector,
    describe_metric_distributions,
    estimate_mutual_information,
    export_validation_case_artifacts,
    fit_rss,
    generate_family_scenario,
    generate_formal_scenario,
    generate_validation_scenario,
    integrated_body_angles_to_quaternions,
    lagged_cross_correlation_loss,
    mutual_information_loss,
    orthonormality_error,
    quaternion_error_angles,
    quaternion_multiply,
    quaternion_reconstruction_loss,
    quaternion_reconstruction_loss_torch,
    relative_reconstruction_quaternions,
    quaternions_to_rotation_matrices,
    rotation_matrix_from_rotation_vector,
    rotation_vector_from_rotation_matrix,
    rotation_vectors_to_quaternions,
    rotation_vectors_to_quaternions_torch,
    rotational_action_loss,
    run_validation_for_scenario,
    select_ksg_subsample_indices,
    source_axes_from_direction_cosines,
    source_cross_correlation_matrix,
    symmetric_matrix_from_weights,
    symmetric_weights_from_matrix,
)
from rss.simulation import _decorrelate_source_angles
from rss.solver import (
    _compute_rss_loss_batch,
    _decode_scalar_and_vector_batch,
    _exponential_learning_rates,
    _pso_position_step,
    _rotation_matrices_from_rotation_vectors,
    _symmetric_matrices_from_weights,
    _torch_bspline_basis_memberships,
)
from rss.realdata_config import RealDataRuntimeConfig, build_realdata_runtime_config_from_args
from rss.cli import _rss_config, build_parser
from rss.exporters import export_limb_mode_metrics_csv, export_limb_observed_signals_csv, export_loss_history_artifacts
from rss.realdata_pipeline import run_realdata_from_config
from rss.plotting import plot_loss_history


class RSSCoreTests(unittest.TestCase):
    def test_quaternion_roundtrip_uses_half_angles(self) -> None:
        rng = np.random.default_rng(4)
        angles = rng.uniform(-0.8, 0.8, size=(3, 128))
        scalar, vector = angles_to_quaternion_components(angles)
        decoded, quaternions = decode_scalar_and_vector(scalar, vector)
        self.assertTrue(np.allclose(decoded, angles, atol=1e-10, rtol=1e-10))
        self.assertTrue(np.allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-12))

    def test_source_composed_quaternion_reconstruction_is_physical_and_differentiable(self) -> None:
        scenario = generate_validation_scenario(SimulationConfig(seed=3, T=64))
        _, observer_quaternions = decode_scalar_and_vector(
            scenario.shared_quaternion_scalar,
            scenario.observation_coefficients,
        )
        truth_loss, composed, relative = quaternion_reconstruction_loss(
            scenario.sources, scenario.mixing_matrix, observer_quaternions
        )
        self.assertGreater(float(np.linalg.det(scenario.mixing_matrix)), 0.0)
        self.assertGreater(float(np.linalg.det(scenario.source_axes_global)), 0.0)
        self.assertGreater(float(np.linalg.det(scenario.observer_axes_global)), 0.0)
        self.assertLess(truth_loss, 1e-24)
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        self.assertTrue(np.allclose(quaternion_multiply(relative, identity), relative, atol=1e-14))
        self.assertTrue(
            np.allclose(
                relative_reconstruction_quaternions(-composed, observer_quaternions),
                relative,
                atol=1e-14,
            )
        )

        reflection = np.eye(3) - 2.0 * np.outer(np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0), np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0))
        reflected = compose_source_quaternions(scenario.sources, reflection)
        reflected_error = relative_reconstruction_quaternions(reflected, observer_quaternions)
        self.assertLess(np.linalg.norm(reflection @ reflection.T - np.eye(3)), 1e-12)
        self.assertGreater(float(np.mean(np.sum(reflected_error[:, 1:] ** 2, axis=1))), 1e-4)

        source_tensor = torch.tensor(scenario.sources, dtype=torch.float64)
        observer_tensor = torch.tensor(observer_quaternions, dtype=torch.float64)
        _, true_terms = compute_rss_loss(
            source_tensor, (1,), recovery_matrix=torch.tensor(scenario.mixing_matrix, dtype=torch.float64),
            separation_weight=0.0, orthogonality_weight=0.0,
            reconstruction_weight=1.0, observer_quaternions=observer_tensor, eps=1e-12,
        )
        reflected_total, reflected_terms = compute_rss_loss(
            source_tensor, (1,), recovery_matrix=torch.tensor(reflection, dtype=torch.float64),
            separation_weight=0.0, orthogonality_weight=0.0,
            reconstruction_weight=1.0, observer_quaternions=observer_tensor, eps=1e-12,
        )
        self.assertLess(float(true_terms["reconstruction_loss"]), 1e-24)
        self.assertLess(float(reflected_terms["orthogonality_loss"]), 1e-24)
        self.assertGreater(float(reflected_total), 1e-4)

        angles = torch.tensor(scenario.sources, dtype=torch.float64)
        matrix = torch.tensor(scenario.mixing_matrix + 0.02 * np.eye(3), dtype=torch.float64, requires_grad=True)
        observer = torch.tensor(observer_quaternions, dtype=torch.float64)
        loss, torch_composed, torch_relative = quaternion_reconstruction_loss_torch(
            angles, matrix, observer
        )
        self.assertGreaterEqual(float(loss.detach()), 0.0)
        self.assertLessEqual(float(loss.detach()), 2.0 + 1e-12)
        antipodal_loss, _, _ = quaternion_reconstruction_loss_torch(
            angles, matrix, -observer
        )
        self.assertAlmostEqual(float(antipodal_loss.detach()), float(loss.detach()), places=12)
        loss.backward()
        self.assertIsNotNone(matrix.grad)
        self.assertTrue(torch.isfinite(matrix.grad).all())
        self.assertTrue(np.allclose(torch_composed.detach().numpy().shape, composed.shape))
        self.assertTrue(np.isfinite(torch_relative.detach().numpy()).all())

        mismatch_angles = np.zeros((3, 1), dtype=float)
        mismatch_observer = np.array([[np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]])
        numpy_loss, _, mismatch_relative = quaternion_reconstruction_loss(
            mismatch_angles, np.eye(3), mismatch_observer
        )
        old_vector_only_loss = float(np.mean(np.sum(mismatch_relative[:, 1:] ** 2, axis=1)))
        self.assertAlmostEqual(numpy_loss, 2.0 - np.sqrt(2.0), places=12)
        self.assertAlmostEqual(old_vector_only_loss, 0.5, places=12)
        self.assertNotAlmostEqual(numpy_loss, old_vector_only_loss, places=6)

    def test_six_weights_construct_exact_symmetric_matrix(self) -> None:
        matrix = symmetric_matrix_from_weights(np.arange(1.0, 7.0))
        expected = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
        self.assertTrue(np.array_equal(matrix, expected))
        self.assertTrue(np.array_equal(matrix, matrix.T))

    def test_so3_rotation_vector_numpy_torch_and_batch_paths_agree(self) -> None:
        vectors = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1e-10, -2e-10, 3e-10),
                (0.2, -0.4, 0.3),
                ((np.pi - 1e-8) / np.sqrt(3),) * 3,
            ),
            dtype=float,
        )
        numpy_matrices = np.stack(
            [rotation_matrix_from_rotation_vector(vector) for vector in vectors]
        )
        tensor = torch.tensor(vectors, dtype=torch.float64, requires_grad=True)
        torch_matrices = _rotation_matrices_from_rotation_vectors(tensor)
        self.assertTrue(
            np.allclose(torch_matrices.detach().numpy(), numpy_matrices, atol=1e-12)
        )
        for matrix in numpy_matrices:
            self.assertTrue(np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12))
            self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=12)
        torch_matrices.sum().backward()
        self.assertTrue(torch.isfinite(tensor.grad).all())
        numpy_quaternions = rotation_vectors_to_quaternions(vectors)
        torch_quaternions = rotation_vectors_to_quaternions_torch(
            torch.tensor(vectors, dtype=torch.float64)
        )
        self.assertTrue(
            np.allclose(torch_quaternions.numpy(), numpy_quaternions, atol=1e-12)
        )

    def test_manual_initial_weight_matrix_resolves_for_both_parameterizations(self) -> None:
        for rotation_vector in (
            np.array([0.0, 0.0, 0.0]),
            np.array([0.2, -0.4, 0.3]),
            np.array([np.pi, 0.0, 0.0]),
        ):
            initial_matrix = rotation_matrix_from_rotation_vector(rotation_vector)
            resolved_vector = rotation_vector_from_rotation_matrix(initial_matrix)
            self.assertTrue(
                np.allclose(
                    rotation_matrix_from_rotation_vector(resolved_vector),
                    initial_matrix,
                    atol=1e-10,
                )
            )
            config = RSSConfig(initial_weight_matrix=initial_matrix)
            self.assertTrue(np.allclose(config.resolved_initial_matrix, initial_matrix, atol=1e-10))
            self.assertIsInstance(config.initial_weight_matrix, tuple)

        symmetric_matrix = np.array(
            [[0.9, 0.1, -0.2], [0.1, 0.8, 0.05], [-0.2, 0.05, 1.1]]
        )
        packed = symmetric_weights_from_matrix(symmetric_matrix)
        self.assertTrue(np.allclose(symmetric_matrix_from_weights(packed), symmetric_matrix))
        symmetric_config = RSSConfig(
            matrix_parameterization="symmetric",
            initial_weight_matrix=symmetric_matrix,
        )
        self.assertTrue(np.allclose(symmetric_config.resolved_initial_matrix, symmetric_matrix))

    def test_body_fixed_observers_move_and_channels_reconstruct_orientation(self) -> None:
        scenario = generate_validation_scenario(SimulationConfig(seed=6, T=128))
        self.assertEqual(scenario.observation_model, "body_fixed_gyro")
        self.assertTrue(
            np.allclose(
                np.mean(scenario.generated_sources, axis=1),
                0.0,
                rtol=0.0,
                atol=1e-12,
            )
        )
        self.assertTrue(
            np.allclose(
                np.mean(scenario.relative_sources, axis=1),
                0.0,
                rtol=0.0,
                atol=1e-12,
            )
        )
        self.assertTrue(
            np.allclose(
                np.mean(scenario.sources, axis=1),
                0.0,
                rtol=0.0,
                atol=1e-12,
            )
        )
        self.assertTrue(
            np.allclose(
                scenario.mixing_matrix,
                scenario.source_axes_global @ scenario.observer_axes_global.T,
                atol=1e-12,
            )
        )
        reconstructed = integrated_body_angles_to_quaternions(scenario.observations)
        error = quaternion_error_angles(
            relative_reconstruction_quaternions(
                reconstructed,
                scenario.physical_observer_quaternions,
            )
        )
        self.assertLess(float(np.max(error)), 1e-12)
        rotations = quaternions_to_rotation_matrices(
            scenario.physical_object_quaternions
        )
        expected_observer_axes = np.einsum(
            "ij,tkj->tik",
            scenario.observer_axes_global,
            rotations,
        )
        expected_source_axes = np.einsum(
            "ij,tkj->tik",
            scenario.source_axes_global,
            rotations,
        )
        self.assertTrue(
            np.allclose(
                scenario.observer_axes_global_time,
                expected_observer_axes,
                atol=1e-12,
            )
        )
        self.assertTrue(
            np.allclose(
                scenario.source_axes_global_time,
                expected_source_axes,
                atol=1e-12,
            )
        )
        for axes in scenario.observer_axes_global_time:
            self.assertLessEqual(orthonormality_error(axes), 1e-12)
        for axes in scenario.source_axes_global_time:
            self.assertLessEqual(orthonormality_error(axes), 1e-12)
        self.assertGreater(
            float(
                np.max(
                    np.abs(
                        scenario.observer_axes_global_time[-1]
                        - scenario.observer_axes_global_time[0]
                    )
                )
            ),
            1e-4,
        )
        scalar, source_vector = angles_to_quaternion_components(scenario.sources)
        direct_euler, _ = decode_scalar_and_vector(
            scalar,
            scenario.mixing_matrix.T @ source_vector,
        )
        self.assertGreater(
            float(np.sqrt(np.mean((direct_euler - scenario.observations) ** 2))),
            1e-4,
        )

    def test_legacy_symmetric_euler_path_remains_selectable(self) -> None:
        scenario = generate_validation_scenario(
            SimulationConfig(
                seed=7,
                T=64,
                observation_model="legacy_vector_mixing",
            )
        )
        validation = run_validation_for_scenario(
            scenario,
            RSSConfig(
                lags=(1, 2),
                epochs=1,
                matrix_parameterization="symmetric",
                observer_signal_model="euler",
                convergence_tol=0.0,
            ),
        )
        result = validation.rss_result
        self.assertEqual(scenario.observation_model, "legacy_vector_mixing")
        self.assertTrue(
            np.allclose(
                np.mean(scenario.generated_sources, axis=1),
                0.0,
                rtol=0.0,
                atol=1e-12,
            )
        )
        self.assertTrue(
            np.allclose(
                np.mean(scenario.relative_sources, axis=1),
                0.0,
                rtol=0.0,
                atol=1e-12,
            )
        )
        self.assertTrue(
            np.allclose(
                np.mean(scenario.sources, axis=1),
                0.0,
                rtol=0.0,
                atol=1e-12,
            )
        )
        self.assertTrue(
            np.allclose(result.recovery_matrix, result.recovery_matrix.T, atol=1e-14)
        )
        self.assertEqual(result.optimized_parameters.shape, (6,))
        self.assertTrue(np.isfinite(result.reconstructed_observer_angles).all())
        self.assertLess(
            validation.metrics["physical_quaternion_consistency_max_error_deg"],
            1e-10,
        )

    def test_exponential_learning_rate_schedule_and_pso_position_step(self) -> None:
        rates = np.asarray(_exponential_learning_rates(1e-1, 1e-5, 5))
        self.assertAlmostEqual(rates[0], 1e-1, places=15)
        self.assertAlmostEqual(rates[-1], 1e-5, places=15)
        self.assertTrue(np.all(np.diff(rates) < 0.0))
        self.assertTrue(np.allclose(rates[1:] / rates[:-1], rates[1] / rates[0]))
        self.assertEqual(_exponential_learning_rates(0.2, 0.01, 1), (0.2,))
        self.assertEqual(_exponential_learning_rates(0.2, 0.01, 0), ())
        self.assertEqual(_exponential_learning_rates(0.03, 0.03, 3), (0.03, 0.03, 0.03))

        positions = torch.tensor([[0.0, 0.9]], dtype=torch.float64)
        velocities = torch.tensor([[2.0, 2.0]], dtype=torch.float64)
        updated_positions, updated_velocities = _pso_position_step(
            positions, velocities, 0.1, lower=-1.0, upper=1.0
        )
        self.assertTrue(torch.allclose(updated_positions, torch.tensor([[0.2, 1.0]], dtype=torch.float64)))
        self.assertTrue(torch.allclose(updated_velocities, torch.tensor([[2.0, 0.0]], dtype=torch.float64)))

    def test_lagged_loss_merges_zero_and_directed_lags_with_one_third_scaling(self) -> None:
        signals = torch.tensor(
            [[1.0, 0.0, 2.0, -1.0], [0.5, -0.5, 1.5, 2.0], [2.0, 1.0, -1.0, 0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        recovery = torch.tensor(
            [[1.0, 0.2, 0.0], [0.2, 0.9, 0.1], [0.0, 0.1, 1.1]],
            dtype=torch.float64,
            requires_grad=True,
        )
        total, terms = compute_rss_loss(
            signals,
            (0, 1),
            recovery_matrix=recovery,
            separation_weight=1.0,
            orthogonality_weight=1.0,
            eps=1e-12,
        )
        centered = signals.detach().numpy() - signals.detach().numpy().mean(axis=1, keepdims=True)
        c0 = centered @ centered.T / 4.0
        inv = 1.0 / np.sqrt(np.diag(c0))
        r0 = inv[:, None] * c0 * inv[None, :]
        c1 = centered[:, 1:] @ centered[:, :-1].T / 3.0
        r1 = inv[:, None] * c1 * inv[None, :]
        off0 = r0 - np.diag(np.diag(r0))
        off1 = r1 - np.diag(np.diag(r1))
        expected_separation = float(
            (np.sum(off0**2) / 3.0 + np.sum(off1**2) / 3.0) / 2.0
        )
        self.assertAlmostEqual(
            float(terms["separation_loss"].detach()), expected_separation, places=12
        )
        zero_lag = lagged_cross_correlation_loss(signals, (0,), eps=1e-12)
        self.assertAlmostEqual(float(zero_lag.detach()), float(np.sum(off0**2) / 3.0), places=12)
        recovery_array = recovery.detach().numpy()
        expected_orthogonality = np.sum((recovery_array @ recovery_array.T - np.eye(3)) ** 2) / 3.0
        self.assertAlmostEqual(
            float(terms["orthogonality_loss"].detach()),
            float(expected_orthogonality),
            places=12,
        )
        self.assertAlmostEqual(
            float(total.detach()),
            float(terms["separation_loss"].detach() + expected_orthogonality),
            places=12,
        )
        total.backward()
        self.assertTrue(torch.isfinite(signals.grad).all())
        self.assertIsNotNone(recovery.grad)
        self.assertTrue(torch.isfinite(recovery.grad).all())
        self.assertGreater(float(torch.linalg.norm(recovery.grad)), 0.0)

    def test_mutual_information_detects_dependence_and_has_finite_gradients(self) -> None:
        rng = np.random.default_rng(42)
        independent = torch.tensor(rng.normal(size=(3, 512)), dtype=torch.float64)
        dependent_values = rng.normal(size=(3, 512))
        dependent_values[1] = dependent_values[0] + 0.05 * rng.normal(size=512)
        dependent = torch.tensor(dependent_values, dtype=torch.float64, requires_grad=True)
        independent_loss = mutual_information_loss(
            independent, method="histogram", bins=16, bandwidth=0.5
        )
        dependent_loss = mutual_information_loss(
            dependent, method="histogram", bins=16, bandwidth=0.5
        )
        self.assertGreater(float(dependent_loss.detach()), float(independent_loss.detach()))
        self.assertGreaterEqual(float(independent_loss.detach()), 0.0)
        expected_dependent = np.mean(
            [
                estimate_mutual_information(
                    dependent_values[first],
                    dependent_values[second],
                    method="histogram",
                    bins=16,
                    bandwidth=0.5,
                )
                for first, second in ((0, 1), (0, 2), (1, 2))
            ]
        )
        self.assertAlmostEqual(float(dependent_loss.detach()), float(expected_dependent), places=12)
        dependent_loss.backward()
        self.assertIsNotNone(dependent.grad)
        self.assertTrue(torch.isfinite(dependent.grad).all())
        self.assertGreater(float(torch.linalg.vector_norm(dependent.grad)), 0.0)

    def test_ksg_matches_gaussian_truth_is_symmetric_and_orders_dependence(self) -> None:
        rng = np.random.default_rng(42)
        sample_count = 2000
        x = rng.normal(size=sample_count)
        innovation = rng.normal(size=sample_count)
        estimates = []
        for correlation in (0.0, 0.3, 0.6, 0.9):
            y = correlation * x + np.sqrt(1.0 - correlation**2) * innovation
            estimate = estimate_mutual_information(x, y, k=4)
            reverse = estimate_mutual_information(y, x, k=4)
            truth = -0.5 * np.log(1.0 - correlation**2) if correlation else 0.0
            self.assertAlmostEqual(estimate, reverse, places=12)
            self.assertLess(abs(estimate - truth), 0.06)
            estimates.append(estimate)
        self.assertLess(abs(estimates[0]), 0.06)
        self.assertTrue(np.all(np.diff(estimates) > 0.0))
        self.assertLess(estimates[0], 0.0)  # Raw finite-sample estimates are not clamped.

    def test_gcmi_matches_gaussian_truth_and_documents_nonlinear_behavior(self) -> None:
        rng = np.random.default_rng(55)
        sample_count = 3000
        x = rng.normal(size=sample_count)
        innovation = rng.normal(size=sample_count)
        estimates = []
        for correlation in (0.0, 0.3, 0.6, 0.9):
            y = correlation * x + np.sqrt(1.0 - correlation**2) * innovation
            estimate = estimate_mutual_information(x, y, method="gcmi")
            reverse = estimate_mutual_information(y, x, method="gcmi")
            truth = -0.5 * np.log(1.0 - correlation**2) if correlation else 0.0
            self.assertAlmostEqual(estimate, reverse, places=12)
            self.assertLess(abs(estimate - truth), 0.04)
            estimates.append(estimate)
        self.assertTrue(np.all(np.diff(estimates) > 0.0))

        nonlinear_x = rng.uniform(-1.0, 1.0, size=sample_count)
        monotonic = nonlinear_x**3 + 0.08 * rng.normal(size=sample_count)
        nonmonotonic = nonlinear_x**2 + 0.08 * rng.normal(size=sample_count)
        independent = rng.normal(size=sample_count)
        monotonic_gcmi = estimate_mutual_information(
            nonlinear_x, monotonic, method="gcmi"
        )
        independent_gcmi = estimate_mutual_information(
            nonlinear_x, independent, method="gcmi"
        )
        nonmonotonic_gcmi = estimate_mutual_information(
            nonlinear_x, nonmonotonic, method="gcmi"
        )
        nonmonotonic_bspline = estimate_mutual_information(
            nonlinear_x, nonmonotonic, method="bspline"
        )
        self.assertGreater(monotonic_gcmi, independent_gcmi + 0.2)
        self.assertGreater(nonmonotonic_bspline, 0.1)
        self.assertLess(nonmonotonic_gcmi, nonmonotonic_bspline * 0.1)
        self.assertEqual(
            estimate_mutual_information(np.ones(32), np.arange(32), method="gcmi"),
            0.0,
        )
        source_values = np.stack((x, 0.5 * x + innovation, innovation), axis=0)
        expected_pair_mean = np.mean(
            [
                estimate_mutual_information(
                    source_values[first], source_values[second], method="gcmi"
                )
                for first, second in ((0, 1), (0, 2), (1, 2))
            ]
        )
        actual_pair_mean = mutual_information_loss(
            torch.tensor(source_values, dtype=torch.float64), method="gcmi"
        )
        self.assertAlmostEqual(float(actual_pair_mean), float(expected_pair_mean), places=12)

    def test_bspline_is_symmetric_partitioned_batched_and_differentiable(self) -> None:
        coordinates = torch.tensor(
            [[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=torch.float64
        )
        memberships = _torch_bspline_basis_memberships(
            coordinates, bins=6, order=3, eps=1e-12
        )
        self.assertTrue(torch.all(memberships >= 0.0))
        self.assertTrue(
            torch.allclose(
                torch.sum(memberships, dim=-1),
                torch.ones_like(coordinates),
                atol=1e-14,
                rtol=0.0,
            )
        )
        self.assertEqual(float(memberships[0, 0, 0]), 1.0)
        self.assertEqual(float(memberships[0, -1, -1]), 1.0)

        rng = np.random.default_rng(81)
        independent = rng.normal(size=(3, 768))
        dependent = independent.copy()
        dependent[1] = dependent[0] ** 2 + 0.1 * rng.normal(size=768)
        independent_loss = mutual_information_loss(
            torch.tensor(independent, dtype=torch.float64), method="bspline"
        )
        signals = torch.tensor(dependent, dtype=torch.float64, requires_grad=True)
        dependent_loss = mutual_information_loss(signals, method="bspline")
        expected = np.mean(
            [
                estimate_mutual_information(
                    dependent[first], dependent[second], method="bspline"
                )
                for first, second in ((0, 1), (0, 2), (1, 2))
            ]
        )
        self.assertGreater(float(dependent_loss.detach()), float(independent_loss))
        self.assertAlmostEqual(float(dependent_loss.detach()), float(expected), places=12)
        self.assertAlmostEqual(
            estimate_mutual_information(dependent[0], dependent[1], method="bspline"),
            estimate_mutual_information(dependent[1], dependent[0], method="bspline"),
            places=12,
        )
        dependent_loss.backward()
        self.assertTrue(torch.isfinite(signals.grad).all())
        self.assertGreater(float(torch.linalg.vector_norm(signals.grad)), 0.0)

        batch = torch.stack((signals.detach(), 0.8 * signals.detach()), dim=0)
        batch_loss = mutual_information_loss(batch, method="bspline")
        separate = torch.stack(
            [mutual_information_loss(candidate, method="bspline") for candidate in batch]
        )
        self.assertTrue(torch.allclose(batch_loss, separate, atol=1e-14, rtol=0.0))
        self.assertEqual(
            estimate_mutual_information(np.ones(32), np.arange(32), method="bspline"),
            0.0,
        )

    def test_ksg_fixed_subsampling_is_shared_deterministic_and_optional(self) -> None:
        rng = np.random.default_rng(91)
        sample_count = 512
        x = rng.normal(size=sample_count)
        y = 0.6 * x + rng.normal(scale=0.5, size=sample_count)
        first_indices = select_ksg_subsample_indices(
            sample_count, max_samples=128, seed=7
        )
        second_indices = select_ksg_subsample_indices(
            sample_count, max_samples=128, seed=7
        )
        self.assertTrue(np.array_equal(first_indices, second_indices))
        self.assertEqual(first_indices.size, 128)
        self.assertTrue(np.all(np.diff(first_indices) > 0))

        full = estimate_mutual_information(x, y, method="ksg", k=4)
        uncapped = estimate_mutual_information(
            x, y, method="ksg", k=4, ksg_max_samples=sample_count + 1
        )
        self.assertEqual(full, uncapped)
        capped = estimate_mutual_information(
            x,
            y,
            method="ksg",
            k=4,
            ksg_max_samples=128,
            ksg_subsample_seed=7,
        )
        manual = estimate_mutual_information(
            x[first_indices], y[first_indices], method="ksg", k=4
        )
        self.assertEqual(capped, manual)
        self.assertEqual(
            capped,
            estimate_mutual_information(
                x,
                y,
                method="ksg",
                k=4,
                ksg_max_samples=128,
                ksg_subsample_seed=7,
            ),
        )

        sources = np.stack((x, y, rng.normal(size=sample_count)), axis=0)
        expected = np.mean(
            [
                estimate_mutual_information(
                    sources[first, first_indices],
                    sources[second, first_indices],
                    method="ksg",
                    k=4,
                )
                for first, second in ((0, 1), (0, 2), (1, 2))
            ]
        )
        actual = mutual_information_loss(
            torch.tensor(sources, dtype=torch.float64),
            method="ksg",
            k=4,
            ksg_max_samples=128,
            ksg_subsample_seed=7,
        )
        self.assertAlmostEqual(float(actual), float(expected), places=12)

    def test_ksg_is_multivariate_and_deterministic_with_repeated_values(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=(800, 2))
        y = x + 0.5 * rng.normal(size=(800, 2))
        dependent = estimate_mutual_information(x, y, k=4)
        independent = estimate_mutual_information(x, rng.normal(size=(800, 2)), k=4)
        self.assertGreater(dependent, independent)

        repeated_x = np.repeat(np.arange(10, dtype=float), 10)
        repeated_y = np.tile(np.arange(10, dtype=float), 10)
        first = estimate_mutual_information(repeated_x, repeated_y, k=4)
        second = estimate_mutual_information(repeated_x, repeated_y, k=4)
        reverse = estimate_mutual_information(repeated_y, repeated_x, k=4)
        self.assertEqual(first, second)
        self.assertEqual(first, reverse)

    def test_ksg_tree_counts_match_a_strict_brute_force_reference(self) -> None:
        rng = np.random.default_rng(17)
        x = rng.normal(size=(48, 2))
        y = rng.normal(size=(48, 1))
        k = 3
        estimate = estimate_mutual_information(x, y, k=k, standardize=True)

        x_standardized = (x - x.mean(axis=0)) / x.std(axis=0)
        y_standardized = (y - y.mean(axis=0)) / y.std(axis=0)
        joint = np.concatenate((x_standardized, y_standardized), axis=1)
        joint_distances = np.max(
            np.abs(joint[:, None, :] - joint[None, :, :]), axis=2
        )
        radii = np.sort(joint_distances, axis=1)[:, k]
        x_distances = np.max(
            np.abs(x_standardized[:, None, :] - x_standardized[None, :, :]), axis=2
        )
        y_distances = np.max(
            np.abs(y_standardized[:, None, :] - y_standardized[None, :, :]), axis=2
        )
        x_counts = np.sum(x_distances < radii[:, None], axis=1) - 1
        y_counts = np.sum(y_distances < radii[:, None], axis=1) - 1
        expected = float(
            digamma(k)
            + digamma(x.shape[0])
            - np.mean(digamma(x_counts + 1) + digamma(y_counts + 1))
        )
        self.assertAlmostEqual(estimate, expected, places=12)

    def test_ksg_rss_loss_preserves_the_existing_three_pair_mean(self) -> None:
        rng = np.random.default_rng(23)
        values = rng.normal(size=(3, 256))
        values[1] = 0.7 * values[0] + 0.3 * values[1]
        signals = torch.tensor(values, dtype=torch.float64)
        expected = np.mean(
            [
                estimate_mutual_information(values[first], values[second], k=4)
                for first, second in ((0, 1), (0, 2), (1, 2))
            ]
        )
        actual = mutual_information_loss(signals, method="ksg", k=4)
        self.assertAlmostEqual(float(actual), float(expected), places=12)

    def test_mutual_information_estimator_rejects_invalid_inputs(self) -> None:
        valid = np.arange(8, dtype=float)
        invalid_calls = (
            lambda: estimate_mutual_information(valid, valid[:-1]),
            lambda: estimate_mutual_information(valid, valid, k=0),
            lambda: estimate_mutual_information(valid, valid, k=2.5),
            lambda: estimate_mutual_information(valid, valid, k=len(valid)),
            lambda: estimate_mutual_information(valid, np.full(8, np.nan)),
            lambda: estimate_mutual_information(np.zeros((2, 2, 2)), valid),
            lambda: estimate_mutual_information(valid, valid, method="unknown"),
            lambda: estimate_mutual_information(valid, valid, standardize="yes"),
            lambda: estimate_mutual_information(
                valid, valid, method="gcmi", gcmi_rank_method="random"
            ),
            lambda: estimate_mutual_information(
                np.ones((8, 2)), np.ones((8, 2)), method="gcmi"
            ),
            lambda: estimate_mutual_information(
                valid, valid, method="bspline", bspline_bins=3, bspline_order=3
            ),
            lambda: estimate_mutual_information(
                np.ones((8, 2)), np.ones((8, 2)), method="bspline"
            ),
            lambda: estimate_mutual_information(
                valid, valid, method="ksg", k=4, ksg_max_samples=4
            ),
            lambda: estimate_mutual_information(
                np.ones((8, 2)), np.ones((8, 2)), method="histogram"
            ),
            lambda: select_ksg_subsample_indices(8, max_samples=0),
            lambda: select_ksg_subsample_indices(8, max_samples=4, mode="random_each_call"),
            lambda: select_ksg_subsample_indices(8, max_samples=4, seed=-1),
        )
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()

    def test_rotational_action_matches_linear_motion_and_wraps_angles(self) -> None:
        linear = torch.tensor(
            [[0.0, 0.1, 0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        expected_mode_action = 0.5 - 0.5 * np.mean(np.square((0.05, 0.15)))
        expected = expected_mode_action / 3.0
        action = rotational_action_loss(
            linear,
            sample_interval_seconds=0.1,
            inertias=(1.0, 1.0, 1.0),
            stiffnesses=(1.0, 1.0, 1.0),
            eps=1e-12,
        )
        self.assertAlmostEqual(float(action.detach()), expected, places=10)
        wrapped = torch.tensor(
            [
                [np.pi - 0.05, -np.pi + 0.05, -np.pi + 0.15],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        wrapped_action = rotational_action_loss(wrapped, sample_interval_seconds=0.1)
        self.assertAlmostEqual(float(wrapped_action), float(action.detach()), places=10)
        action.backward()
        self.assertTrue(torch.isfinite(linear.grad).all())

    def test_orthogonality_loss_is_zero_for_an_orthogonal_matrix(self) -> None:
        scenario = generate_validation_scenario(SimulationConfig(seed=17, T=32))
        signals = torch.tensor(scenario.sources, dtype=torch.float64)
        recovery = torch.tensor(scenario.mixing_matrix, dtype=torch.float64)
        total, terms = compute_rss_loss(
            signals,
            (1,),
            recovery_matrix=recovery,
            separation_weight=0.0,
            orthogonality_weight=1.0,
            eps=1e-12,
        )
        self.assertLess(float(terms["orthogonality_loss"]), 1e-24)
        self.assertLess(float(total), 1e-24)

    def test_fit_is_deterministic_so3_and_keeps_best_state(self) -> None:
        scenario = generate_validation_scenario(SimulationConfig(seed=2, T=96))
        config = RSSConfig(lags=(1, 2, 3), epochs=5, learning_rate=1e-3, convergence_tol=1.0)
        first = fit_rss(scenario.observations, config)
        second = fit_rss(scenario.observations, config)
        self.assertEqual(config.optimizer, "adam")
        self.assertEqual(first.optimizer_name, "adam")
        self.assertTrue(
            np.allclose(first.recovery_matrix @ first.recovery_matrix.T, np.eye(3), atol=1e-12)
        )
        self.assertAlmostEqual(float(np.linalg.det(first.recovery_matrix)), 1.0, places=12)
        self.assertTrue(np.allclose(first.recovery_matrix, second.recovery_matrix, atol=1e-14))
        self.assertAlmostEqual(first.best_loss, min(row["total_loss"] for row in first.loss_history))
        self.assertAlmostEqual(
            first.normalized_best_loss,
            min(row["normalized_total_loss"] for row in first.loss_history),
        )
        self.assertAlmostEqual(first.loss_normalization_reference, first.initial_loss)
        self.assertAlmostEqual(first.normalized_initial_loss, 1.0)
        self.assertEqual(first.loss_history[0]["epoch"], 0.0)
        self.assertIsNone(first.loss_history[0]["learning_rate"])
        expected_rates = _exponential_learning_rates(1e-3, 1e-5, 5)
        self.assertTrue(np.allclose(first.applied_learning_rates, expected_rates))
        self.assertTrue(
            np.allclose([row["learning_rate"] for row in first.loss_history[1:]], expected_rates)
        )
        self.assertEqual(first.stop_reason, "max_epochs")
        self.assertTrue(np.isfinite(first.source_angles).all())
        self.assertEqual(first.loss_history[0]["loss_type"], "lagged_cross_correlation")
        self.assertIn("separation_loss", first.loss_history[0])
        self.assertIn("normalized_separation_loss", first.loss_history[0])
        self.assertIn("orthogonality_loss", first.loss_history[0])
        self.assertIn("normalized_orthogonality_loss", first.loss_history[0])
        self.assertIn("reconstruction_loss", first.loss_history[0])
        self.assertIn("normalized_reconstruction_loss", first.loss_history[0])
        for row in first.loss_history:
            normalized_components = (
                row["normalized_separation_loss"]
                + row["normalized_orthogonality_loss"]
                + row["normalized_reconstruction_loss"]
            )
            self.assertAlmostEqual(row["normalized_total_loss"], normalized_components, places=12)
            self.assertAlmostEqual(
                row["normalized_total_loss"],
                row["total_loss"] / first.loss_normalization_reference,
                places=12,
            )
        expected_error = np.linalg.norm(first.recovery_matrix @ first.recovery_matrix.T - np.eye(3), ord="fro")
        self.assertAlmostEqual(first.matrix_diagnostics.orthogonality_error, float(expected_error), places=12)
        self.assertAlmostEqual(first.matrix_diagnostics.determinant, float(np.linalg.det(first.recovery_matrix)), places=12)

    def test_sgd_and_pso_optimizer_paths_are_finite_and_deterministic(self) -> None:
        scenario = generate_validation_scenario(SimulationConfig(seed=21, T=64))
        sgd = fit_rss(
            scenario.observations,
            RSSConfig(
                lags=(1, 2), epochs=3, optimizer="sgd", learning_rate=1e-3,
                sgd_momentum=0.9, convergence_tol=1.0,
            ),
        )
        self.assertEqual(sgd.optimizer_name, "sgd")
        self.assertEqual(sgd.stop_reason, "max_epochs")
        self.assertTrue(np.isfinite(sgd.optimized_weights).all())
        self.assertEqual(len(sgd.loss_history), 4)
        self.assertTrue(
            np.allclose(sgd.applied_learning_rates, _exponential_learning_rates(1e-3, 1e-5, 3))
        )

        pso_config = RSSConfig(
            lags=(1, 2), epochs=4, optimizer="pso", pso_swarm_size=6,
            pso_seed=13, convergence_tol=0.0,
        )
        first = fit_rss(scenario.observations, pso_config)
        second = fit_rss(scenario.observations, pso_config)
        self.assertEqual(first.optimizer_name, "pso")
        self.assertTrue(np.allclose(first.optimized_weights, second.optimized_weights, atol=0.0, rtol=0.0))
        lower, upper = pso_config.resolved_pso_bounds
        self.assertTrue(np.all(first.optimized_parameters >= lower))
        self.assertTrue(np.all(first.optimized_parameters <= upper))
        normalized = np.asarray([row["normalized_total_loss"] for row in first.loss_history])
        self.assertTrue(np.all(np.diff(normalized) <= 1e-14))
        self.assertAlmostEqual(first.normalized_initial_loss, 1.0)
        self.assertLessEqual(first.best_loss, first.initial_loss)
        self.assertIsNone(first.loss_history[0]["learning_rate"])
        self.assertIsNone(first.loss_history[1]["learning_rate"])
        expected_pso_rates = _exponential_learning_rates(1e-1, 1e-5, 3)
        self.assertTrue(np.allclose(first.applied_learning_rates, expected_pso_rates))
        self.assertTrue(
            np.allclose([row["learning_rate"] for row in first.loss_history[2:]], expected_pso_rates)
        )

    def test_optimizer_compatibility_for_primary_losses_and_mi_estimators(self) -> None:
        scenario = generate_validation_scenario(SimulationConfig(seed=121, T=48))
        for loss_type in ("lagged_cross_correlation", "action"):
            for optimizer in ("adam", "sgd", "pso"):
                with self.subTest(loss_type=loss_type, optimizer=optimizer):
                    config = RSSConfig(
                        loss_type=loss_type,
                        lags=(0, 1),
                        epochs=1,
                        optimizer=optimizer,
                        learning_rate=1e-3,
                        pso_swarm_size=3,
                    )
                    result = fit_rss(
                        scenario.observations,
                        config,
                        sample_interval_seconds=float(np.diff(scenario.time_seconds)[0]),
                    )
                    self.assertEqual(result.loss_history[0]["loss_type"], loss_type)
                    self.assertTrue(np.isfinite(result.best_loss))
                    self.assertTrue(np.isfinite(result.optimized_parameters).all())

        for method in ("bspline", "histogram"):
            for optimizer in ("adam", "sgd", "pso"):
                with self.subTest(
                    loss_type="mutual_information", method=method, optimizer=optimizer
                ):
                    result = fit_rss(
                        scenario.observations,
                        RSSConfig(
                            loss_type="mutual_information",
                            mi_estimator=method,
                            epochs=1,
                            optimizer=optimizer,
                            learning_rate=1e-3,
                            pso_swarm_size=3,
                        ),
                    )
                    self.assertTrue(np.isfinite(result.best_loss))

        for method in ("gcmi", "ksg"):
            for optimizer in ("adam", "sgd"):
                with self.subTest(
                    loss_type="mutual_information", method=method, optimizer=optimizer
                ):
                    with self.assertRaisesRegex(ValueError, "requires optimizer='pso'"):
                        RSSConfig(
                            loss_type="mutual_information",
                            mi_estimator=method,
                            optimizer=optimizer,
                        )

        for method in ("gcmi", "ksg"):
            config = RSSConfig(
                loss_type="mutual_information",
                mi_estimator=method,
                ksg_k=4,
                ksg_max_samples=32,
                optimizer="pso",
                epochs=1,
                pso_swarm_size=3,
                pso_seed=9,
            )
            first = fit_rss(scenario.observations, config)
            second = fit_rss(scenario.observations, config)
            self.assertTrue(np.isfinite(first.best_loss))
            self.assertTrue(np.array_equal(first.optimized_parameters, second.optimized_parameters))

    def test_action_timebase_uses_config_then_context_and_is_required_directly(self) -> None:
        scenario = generate_validation_scenario(
            SimulationConfig(seed=122, T=48, duration_seconds=0.48)
        )
        contextual = run_validation_for_scenario(
            scenario,
            RSSConfig(loss_type="action", epochs=1),
        ).rss_result
        self.assertAlmostEqual(contextual.sample_interval_seconds, 0.01)
        configured = run_validation_for_scenario(
            scenario,
            RSSConfig(loss_type="action", epochs=1, sample_interval_seconds=0.02),
        ).rss_result
        self.assertAlmostEqual(configured.sample_interval_seconds, 0.02)
        with self.assertRaisesRegex(ValueError, "sample_interval_seconds"):
            fit_rss(scenario.observations, RSSConfig(loss_type="action", epochs=1))

    def test_single_update_and_equal_rate_fits_report_applied_rates(self) -> None:
        scenario = generate_validation_scenario(SimulationConfig(seed=31, T=48))
        single = fit_rss(
            scenario.observations,
            RSSConfig(
                lags=(1,), epochs=1, learning_rate=0.02,
                learning_rate_end=1e-5, convergence_tol=0.0,
            ),
        )
        self.assertEqual(single.applied_learning_rates, (0.02,))
        fixed = fit_rss(
            scenario.observations,
            RSSConfig(
                lags=(1,), epochs=3, optimizer="sgd", learning_rate=0.003,
                learning_rate_end=0.003, convergence_tol=0.0,
            ),
        )
        self.assertEqual(fixed.applied_learning_rates, (0.003, 0.003, 0.003))

    def test_pso_uses_full_epoch_budget_and_batched_objective(self) -> None:
        scenario = generate_validation_scenario(SimulationConfig(seed=22, T=48))
        full_budget = fit_rss(
            scenario.observations,
            RSSConfig(
                lags=(1,), epochs=10, optimizer="pso", pso_swarm_size=1,
                pso_inertia=0.0, pso_cognitive=0.0, pso_social=0.0,
                pso_stall_patience=2, convergence_tol=1e-12,
            ),
        )
        expected_rates = _exponential_learning_rates(1e-1, 1e-5, 9)
        self.assertEqual(full_budget.stop_reason, "max_epochs")
        self.assertEqual(len(full_budget.loss_history), 11)
        self.assertTrue(
            np.allclose(full_budget.applied_learning_rates, expected_rates)
        )
        self.assertAlmostEqual(
            full_budget.applied_learning_rates[-1],
            full_budget.config.learning_rate_end,
        )

        scalar, vector = angles_to_quaternion_components(scenario.observations)
        weight_batch = torch.tensor(
            [
                RSSConfig(matrix_parameterization="symmetric").resolved_initial_parameters,
                (0.8, 0.1, -0.1, 0.9, 0.05, 0.7),
            ],
            dtype=torch.float64,
        )
        matrices = _symmetric_matrices_from_weights(weight_batch)
        scalar_tensor = torch.tensor(scalar, dtype=torch.float64)
        vector_tensor = torch.tensor(vector, dtype=torch.float64)
        _, observer_quaternions = decode_scalar_and_vector(scalar, vector)
        observer_tensor = torch.tensor(observer_quaternions, dtype=torch.float64)
        batched_angles = _decode_scalar_and_vector_batch(
            scalar_tensor,
            torch.einsum("pij,jt->pit", matrices, vector_tensor),
            eps=1e-12,
        )
        for loss_type in ("lagged_cross_correlation", "mutual_information", "action"):
            totals, terms = _compute_rss_loss_batch(
                batched_angles,
                (0, 1, 2),
                recovery_matrices=matrices,
                loss_type=loss_type,
                separation_weight=1.0,
                orthogonality_weight=1.0,
                reconstruction_weight=1.0,
                observer_quaternions=observer_tensor,
                sample_interval_seconds=0.01,
                eps=1e-12,
            )
            for index in range(2):
                single_total, single_terms = compute_rss_loss(
                    batched_angles[index],
                    (0, 1, 2),
                    recovery_matrix=matrices[index],
                    loss_type=loss_type,
                    separation_weight=1.0,
                    orthogonality_weight=1.0,
                    reconstruction_weight=1.0,
                    observer_quaternions=observer_tensor,
                    sample_interval_seconds=0.01,
                    eps=1e-12,
                )
                self.assertAlmostEqual(float(totals[index]), float(single_total), places=12)
                for name in terms:
                    self.assertAlmostEqual(
                        float(terms[name][index]), float(single_terms[name]), places=12
                    )

    def test_normalized_components_include_weights_and_zero_initial_loss_is_safe(self) -> None:
        scenario = generate_validation_scenario(SimulationConfig(seed=8, T=96))
        weighted = fit_rss(
            scenario.observations,
            RSSConfig(
                lags=(0, 1, 2), epochs=1, separation_weight=2.0,
                orthogonality_weight=4.0, reconstruction_weight=5.0, convergence_tol=0.0,
            ),
        )
        for row in weighted.loss_history:
            reference = weighted.loss_normalization_reference
            self.assertAlmostEqual(
                row["normalized_separation_loss"],
                2.0 * row["separation_loss"] / reference,
            )
            self.assertAlmostEqual(
                row["normalized_orthogonality_loss"],
                4.0 * row["orthogonality_loss"] / reference,
            )
            self.assertAlmostEqual(
                row["normalized_reconstruction_loss"],
                5.0 * row["reconstruction_loss"] / reference,
            )

        zero_initial = fit_rss(
            scenario.observations,
            RSSConfig(
                lags=(0,), epochs=1, separation_weight=0.0,
                matrix_parameterization="symmetric", orthogonality_weight=1.0,
                reconstruction_weight=0.0, convergence_tol=0.0, eps=1e-9,
            ),
        )
        self.assertLess(zero_initial.initial_loss, 1e-24)
        self.assertEqual(zero_initial.loss_normalization_reference, 1e-9)
        self.assertLess(zero_initial.normalized_initial_loss, 1e-14)
        self.assertTrue(
            all(
                np.isfinite(
                    [
                        row["normalized_total_loss"],
                        row["normalized_separation_loss"],
                        row["normalized_orthogonality_loss"],
                        row["normalized_reconstruction_loss"],
                    ]
                ).all()
                for row in zero_initial.loss_history
            )
        )

    def test_loss_plot_uses_normalized_total_and_weighted_components(self) -> None:
        scenario = generate_validation_scenario(SimulationConfig(seed=14, T=64))
        result = fit_rss(
            scenario.observations,
            RSSConfig(lags=(1, 2), epochs=1, convergence_tol=0.0),
        )
        figure = plot_loss_history(result.loss_history)
        self.assertIn("Normalized RSS Total Loss", figure.axes[0].get_title())
        self.assertIn("Normalized Weighted RSS Loss Contributions", figure.axes[1].get_title())
        total_labels = {line.get_label() for line in figure.axes[0].lines}
        component_labels = {line.get_label() for line in figure.axes[1].lines}
        self.assertIn("normalized_total_loss", total_labels)
        self.assertEqual(
            {
                "normalized_separation_loss",
                "normalized_orthogonality_loss",
                "normalized_reconstruction_loss",
            },
            component_labels,
        )
        self.assertIn("Applied Exponential Learning Rate", figure.axes[2].get_title())
        self.assertEqual(figure.axes[2].get_yscale(), "log")
        self.assertIn("applied_learning_rate", {line.get_label() for line in figure.axes[2].lines})
        import matplotlib.pyplot as plt

        plt.close(figure)

    def test_validation_suite_sizes_and_general_so3_truth(self) -> None:
        scenario = generate_validation_scenario(SimulationConfig(seed=9, T=32))
        self.assertFalse(np.allclose(scenario.mixing_matrix, scenario.mixing_matrix.T, atol=1e-6))
        self.assertTrue(np.allclose(scenario.mixing_matrix @ scenario.mixing_matrix.T, np.eye(3), atol=1e-12))
        self.assertGreater(float(np.linalg.det(scenario.mixing_matrix)), 0.0)
        self.assertGreater(float(np.linalg.det(scenario.source_axes_global)), 0.0)
        self.assertGreater(float(np.linalg.det(scenario.observer_axes_global)), 0.0)
        self.assertLessEqual(orthonormality_error(scenario.source_axes_global), 1e-12)
        self.assertLessEqual(orthonormality_error(scenario.observer_axes_global), 1e-12)
        self.assertEqual(len(create_standard_case_specs()), 20)
        self.assertEqual(len(create_stress_case_specs()), 8)

    def test_fresh_formal_family_cohort_has_200_three_source_cases(self) -> None:
        standard_specs = create_standard_case_specs()
        formal_config = BatchStudyConfig(
            standard_cases_per_family=50,
            batch_sequence_lengths=(96,),
        )
        formal_specs = create_standard_case_specs(
            formal_config,
            base_seed=20000,
            case_id_prefix="formal_",
        )
        self.assertEqual(len(formal_specs), 200)
        self.assertEqual(len({spec.case_id for spec in formal_specs}), 200)
        self.assertEqual(len({spec.seed for spec in formal_specs}), 200)
        self.assertFalse(
            {spec.seed for spec in standard_specs}
            & {spec.seed for spec in formal_specs}
        )
        family_counts = pd.Series([spec.family for spec in formal_specs]).value_counts()
        self.assertEqual(
            family_counts.to_dict(),
            {family: 50 for family in formal_config.standard_families},
        )
        base = SimulationConfig(T=96, source_cross_correlation_limit=0.01)
        for family in formal_config.standard_families:
            with self.subTest(family=family):
                spec = next(spec for spec in formal_specs if spec.family == family)
                scenario = generate_family_scenario(spec, base, formal_config)
                self.assertEqual(scenario.source_roles, ("dense", "dense", "dense"))
                self.assertEqual(scenario.active_source_indices, (0, 1, 2))
                self.assertEqual(scenario.observation_model, "body_fixed_gyro")
                correlation = source_cross_correlation_matrix(scenario.sources)
                pairwise = np.abs(correlation[np.triu_indices(3, k=1)])
                self.assertLess(
                    float(np.max(pairwise)),
                    scenario.config.source_cross_correlation_limit,
                )

    def test_formal_suite_has_all_compositions_families_and_stratified_cases(self) -> None:
        config = FormalStudyConfig()
        specs = create_formal_case_specs(config)
        self.assertEqual(len(specs), 120)
        self.assertEqual(len({spec.case_id for spec in specs}), 120)
        self.assertEqual(len({spec.seed for spec in specs}), 120)
        expected_compositions = {
            "single_rotation",
            "dual_rotation",
            "triple_rotation",
            "single_rotation_one_sparse",
            "dual_rotation_one_sparse",
            "single_rotation_two_sparse",
        }
        self.assertEqual({spec.composition for spec in specs}, expected_compositions)
        self.assertEqual({spec.family for spec in specs}, set(config.families))
        for composition in expected_compositions:
            for family in config.families:
                group = [
                    spec
                    for spec in specs
                    if spec.composition == composition and spec.family == family
                ]
                self.assertEqual([spec.replicate_index for spec in group], [1, 2, 3, 4, 5])
                for source_index in range(3):
                    amplitudes = [
                        spec.requested_amplitudes_rad[source_index]
                        for spec in group
                        if spec.source_roles[source_index] != "inactive"
                    ]
                    frequencies = [
                        spec.requested_frequencies_hz[source_index]
                        for spec in group
                        if spec.source_roles[source_index] != "inactive"
                    ]
                    if not amplitudes:
                        continue
                    self.assertEqual(amplitudes, sorted(amplitudes))
                    self.assertEqual(frequencies, sorted(frequencies))
                    self.assertTrue(
                        all(config.amplitude_range_rad[0] <= value <= config.amplitude_range_rad[1] for value in amplitudes)
                    )
                    self.assertTrue(
                        all(config.frequency_range_hz[0] <= value <= config.frequency_range_hz[1] for value in frequencies)
                    )
        self.assertEqual(specs, create_formal_case_specs(config))

    def test_formal_scenarios_preserve_roles_sparsity_and_active_correlation_limit(self) -> None:
        config = FormalStudyConfig(batch_sequence_lengths=(96,))
        specs = create_formal_case_specs(config)
        compositions = (
            "single_rotation",
            "dual_rotation",
            "triple_rotation",
            "single_rotation_one_sparse",
            "dual_rotation_one_sparse",
            "single_rotation_two_sparse",
        )
        for composition in compositions:
            with self.subTest(composition=composition):
                spec = next(spec for spec in specs if spec.composition == composition)
                scenario = generate_formal_scenario(spec, SimulationConfig(T=96))
                active = np.asarray(scenario.active_source_indices)
                self.assertTrue(
                    np.allclose(
                        np.mean(scenario.generated_sources, axis=1),
                        0.0,
                        rtol=0.0,
                        atol=1e-12,
                    )
                )
                self.assertTrue(
                    np.allclose(
                        np.mean(scenario.relative_sources, axis=1),
                        0.0,
                        rtol=0.0,
                        atol=1e-12,
                    )
                )
                self.assertTrue(
                    np.allclose(
                        np.mean(scenario.sources, axis=1),
                        0.0,
                        rtol=0.0,
                        atol=1e-12,
                    )
                )
                inactive = [
                    index for index, role in enumerate(scenario.source_roles) if role == "inactive"
                ]
                for index in inactive:
                    self.assertTrue(np.array_equal(scenario.sources[index], np.zeros(96)))
                    self.assertIsNone(scenario.realized_amplitudes_rad[index])
                    self.assertIsNone(scenario.realized_frequencies_hz[index])
                for index in scenario.active_source_indices:
                    self.assertAlmostEqual(
                        scenario.realized_amplitudes_rad[index],
                        spec.requested_amplitudes_rad[index],
                        places=12,
                    )
                for index in scenario.sparse_source_indices:
                    duty_cycle = scenario.sparse_duty_cycles[index]
                    self.assertGreaterEqual(duty_cycle, 0.08)
                    self.assertLessEqual(duty_cycle, 0.22)
                    self.assertGreater(np.count_nonzero(scenario.sources[index]), 0)
                    self.assertGreater(np.count_nonzero(scenario.sources[index] == 0.0), 0)
                correlation = source_cross_correlation_matrix(scenario.sources[active])
                if active.size > 1:
                    pairwise = np.abs(correlation[np.triu_indices(active.size, k=1)])
                    self.assertLess(
                        float(np.max(pairwise)),
                        scenario.config.source_cross_correlation_limit,
                    )
                _, observer_quaternions = decode_scalar_and_vector(
                    scenario.shared_quaternion_scalar,
                    scenario.observation_coefficients,
                )
                truth_loss, _, _ = quaternion_reconstruction_loss(
                    scenario.sources,
                    scenario.mixing_matrix,
                    observer_quaternions,
                )
                self.assertLess(truth_loss, 1e-22)

    def test_formal_evaluation_scores_only_active_sources(self) -> None:
        config = FormalStudyConfig(batch_sequence_lengths=(96,))
        specs = create_formal_case_specs(config)
        for composition, active_count in (
            ("single_rotation", 1),
            ("dual_rotation", 2),
            ("triple_rotation", 3),
        ):
            with self.subTest(composition=composition):
                spec = next(spec for spec in specs if spec.composition == composition)
                scenario = generate_formal_scenario(spec, SimulationConfig(T=96))
                result = run_validation_for_scenario(
                    scenario,
                    RSSConfig(lags=(1, 2), epochs=1, convergence_tol=0.0),
                )
                self.assertEqual(len(result.match.active_source_indices), active_count)
                self.assertEqual(len(result.match.unmatched_estimate_indices), 3 - active_count)
                inactive = [
                    index for index, role in enumerate(scenario.source_roles) if role == "inactive"
                ]
                self.assertTrue(np.isnan(result.match.source_rmses[inactive]).all())
                self.assertTrue(np.isnan(result.match.axis_errors_rad[inactive]).all())
                self.assertTrue(np.isnan(result.match.correlations[inactive]).all())
                self.assertTrue(
                    np.isfinite(result.match.source_rmses[np.asarray(scenario.active_source_indices)]).all()
                )
                self.assertEqual(result.metrics["active_source_count"], float(active_count))
                self.assertTrue(np.isfinite(result.metrics["mean_abs_source_correlation"]))
                self.assertGreaterEqual(result.metrics["unmatched_output_energy_ratio"], 0.0)
                self.assertLessEqual(result.metrics["unmatched_output_energy_ratio"], 1.0)

    def test_symmetric_source_whitening_preserves_means_peaks_and_limit(self) -> None:
        time = np.linspace(0.0, 1.0, 256, endpoint=False)
        raw = np.vstack(
            (
                0.4 + 0.7 * np.sin(2.0 * np.pi * time),
                -0.2 + 0.5 * np.sin(2.0 * np.pi * time) + 0.3 * np.cos(6.0 * np.pi * time),
                0.1 + 0.4 * np.sin(2.0 * np.pi * time) - 0.2 * np.cos(10.0 * np.pi * time),
            )
        )
        decorrelated = _decorrelate_source_angles(raw, cross_correlation_limit=0.01)
        self.assertTrue(np.isfinite(decorrelated).all())
        self.assertTrue(np.allclose(np.mean(decorrelated, axis=1), np.mean(raw, axis=1), atol=1e-12))
        raw_peaks = np.max(np.abs(raw - np.mean(raw, axis=1, keepdims=True)), axis=1)
        new_peaks = np.max(
            np.abs(decorrelated - np.mean(decorrelated, axis=1, keepdims=True)), axis=1
        )
        self.assertTrue(np.allclose(new_peaks, raw_peaks, atol=1e-12))
        matrix = source_cross_correlation_matrix(decorrelated)
        pairwise = np.abs(matrix[np.triu_indices(3, k=1)])
        self.assertLess(float(np.max(pairwise)), 0.01)

    def test_every_current_scenario_meets_source_cross_correlation_limit(self) -> None:
        study = BatchStudyConfig()
        base = SimulationConfig(seed=0, T=96, source_cross_correlation_limit=0.01)
        scenarios = [generate_validation_scenario(base)]
        standard_specs = create_standard_case_specs(study)
        stress_specs = create_stress_case_specs(study)
        scenarios.extend(generate_family_scenario(spec, base, study) for spec in standard_specs)
        scenarios.extend(generate_family_scenario(spec, base, study) for spec in stress_specs)
        self.assertEqual(len(standard_specs), 20)
        self.assertEqual(len(stress_specs), 8)
        self.assertEqual(len(scenarios), 29)
        for scenario in scenarios:
            self.assertTrue(
                np.allclose(
                    np.mean(scenario.generated_sources, axis=1),
                    0.0,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
            self.assertTrue(
                np.allclose(
                    np.mean(scenario.relative_sources, axis=1),
                    0.0,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
            self.assertTrue(
                np.allclose(
                    np.mean(scenario.sources, axis=1),
                    0.0,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
            matrix = source_cross_correlation_matrix(scenario.sources)
            pairwise = np.abs(matrix[np.triu_indices(3, k=1)])
            self.assertLess(float(np.max(pairwise)), scenario.config.source_cross_correlation_limit)
            self.assertGreater(float(np.linalg.det(scenario.mixing_matrix)), 0.0)
            _, observer_quaternions = decode_scalar_and_vector(
                scenario.shared_quaternion_scalar,
                scenario.observation_coefficients,
            )
            truth_loss, _, relative = quaternion_reconstruction_loss(
                scenario.sources, scenario.mixing_matrix, observer_quaternions
            )
            self.assertLess(truth_loss, 1e-22)
            self.assertTrue(np.all(relative[:, 0] >= 0.0))
        for scenario, spec in zip(scenarios[-8:], stress_specs):
            self.assertEqual(scenario.case_id, spec.case_id)
            self.assertAlmostEqual(
                float(np.max(np.abs(scenario.generated_sources))),
                spec.stress_peak_angle_rad,
                places=12,
            )

    def test_formal_filtered_noise_cohort_has_no_filter_startup_transient(self) -> None:
        study = BatchStudyConfig(standard_cases_per_family=50)
        base = SimulationConfig(seed=0, T=2000, source_cross_correlation_limit=0.01)
        specs = create_standard_case_specs(
            study,
            base_seed=20000,
            case_id_prefix="formal_",
        )
        filtered_specs = [spec for spec in specs if spec.family == "filtered_noise"]
        self.assertEqual(len(filtered_specs), 50)

        edge_rms_ratios: list[float] = []
        for spec in filtered_specs:
            scenario = generate_family_scenario(spec, base, study)
            edge_count = max(1, scenario.sources.shape[1] // 10)
            interior = scenario.sources[:, edge_count:-edge_count]
            interior_rms = np.sqrt(np.mean(np.square(interior), axis=1))
            leading_rms = np.sqrt(
                np.mean(np.square(scenario.sources[:, :edge_count]), axis=1)
            )
            trailing_rms = np.sqrt(
                np.mean(np.square(scenario.sources[:, -edge_count:]), axis=1)
            )
            edge_rms_ratios.extend((leading_rms / interior_rms).tolist())
            edge_rms_ratios.extend((trailing_rms / interior_rms).tolist())

        # A stationary AR(1) start keeps both edges comparable with the interior.
        # The former unscaled first sample made this ratio exceed six in this cohort.
        self.assertLess(max(edge_rms_ratios), 3.0)

    def test_validation_reports_and_exports_original_and_separated_cross_correlations(self) -> None:
        scenario = generate_validation_scenario(SimulationConfig(seed=5, T=96))
        result = run_validation_for_scenario(
            scenario,
            RSSConfig(lags=(1, 2), epochs=1, orthogonality_weight=0.0, convergence_tol=0.0),
        )
        expected_original = source_cross_correlation_matrix(scenario.sources)
        expected_separated = source_cross_correlation_matrix(result.match.estimated_sources_matched)
        self.assertTrue(np.allclose(result.original_source_cross_correlation_matrix, expected_original))
        self.assertTrue(np.allclose(result.separated_source_cross_correlation_matrix, expected_separated))
        direct_errors = result.match.estimated_sources_matched - scenario.sources
        direct_mses = np.mean(np.square(direct_errors), axis=1)
        direct_max_abs_errors = np.max(np.abs(direct_errors), axis=1)
        self.assertTrue(np.allclose(result.match.source_mses, direct_mses))
        self.assertTrue(
            np.allclose(result.match.source_max_abs_errors, direct_max_abs_errors)
        )
        self.assertAlmostEqual(
            result.metrics["mean_source_mse"], float(np.mean(direct_mses))
        )
        self.assertAlmostEqual(
            result.metrics["max_source_abs_error"],
            float(np.max(direct_max_abs_errors)),
        )
        self.assertEqual(result.metrics["rss_mi_estimator"], "ksg")
        self.assertEqual(result.metrics["rss_mi_unit"], "nats")
        self.assertEqual(result.metrics["rss_gcmi_rank_method"], "average")
        self.assertEqual(result.metrics["rss_bspline_bins"], 6)
        self.assertEqual(result.metrics["rss_bspline_order"], 3)
        self.assertEqual(result.metrics["rss_ksg_k"], 4.0)
        self.assertTrue(result.metrics["rss_ksg_standardize"])
        self.assertIsNone(result.metrics["rss_ksg_max_samples"])
        self.assertEqual(result.metrics["rss_ksg_subsample_mode"], "fixed_random")
        self.assertEqual(result.metrics["rss_ksg_subsample_seed"], 0)
        source_metrics = build_source_metrics_frame([result])
        self.assertEqual(len(source_metrics), 3)
        self.assertTrue(
            np.allclose(source_metrics["source_mse_rad2"], direct_mses)
        )
        self.assertTrue(
            np.allclose(
                source_metrics["source_max_abs_error_rad"],
                direct_max_abs_errors,
            )
        )
        relative = result.rss_result.relative_reconstruction_quaternions
        identity = np.zeros_like(relative)
        identity[:, 0] = 1.0
        direct_reconstruction_loss = float(
            np.mean(
                np.minimum(
                    np.sum((relative - identity) ** 2, axis=1),
                    np.sum((relative + identity) ** 2, axis=1),
                )
            )
        )
        self.assertAlmostEqual(result.metrics["quaternion_reconstruction_loss"], direct_reconstruction_loss)
        for prefix, matrix in (("original_source", expected_original), ("separated_source", expected_separated)):
            pairwise = np.abs(matrix[np.triu_indices(3, k=1)])
            self.assertAlmostEqual(result.metrics[f"{prefix}_abs_cross_correlation_12"], pairwise[0])
            self.assertAlmostEqual(result.metrics[f"{prefix}_abs_cross_correlation_13"], pairwise[1])
            self.assertAlmostEqual(result.metrics[f"{prefix}_abs_cross_correlation_23"], pairwise[2])
            self.assertAlmostEqual(result.metrics[f"{prefix}_mean_abs_cross_correlation"], np.mean(pairwise))
            self.assertAlmostEqual(result.metrics[f"{prefix}_max_abs_cross_correlation"], np.max(pairwise))
        with tempfile.TemporaryDirectory() as temporary:
            export_validation_case_artifacts(result, temporary)
            self.assertTrue((Path(temporary) / "observer_reconstruction.png").exists())
            payload = json.loads((Path(temporary) / "metrics.json").read_text(encoding="utf-8"))
            correlations = payload["source_cross_correlations"]
            self.assertEqual(correlations["limit"], scenario.config.source_cross_correlation_limit)
            self.assertTrue(np.allclose(correlations["original_source_matrix"], expected_original))
            self.assertTrue(np.allclose(correlations["separated_source_matrix"], expected_separated))
            separation_config = payload["rss"]["separation_loss"]
            self.assertEqual(separation_config["mi_estimator"], "ksg")
            self.assertEqual(separation_config["mi_unit"], "nats")
            self.assertEqual(separation_config["gcmi_rank_method"], "average")
            self.assertEqual(separation_config["bspline_bins"], 6)
            self.assertEqual(separation_config["bspline_order"], 3)
            self.assertEqual(separation_config["ksg_k"], 4)
            self.assertTrue(separation_config["ksg_standardize"])
            self.assertIsNone(separation_config["ksg_max_samples"])
            self.assertEqual(separation_config["ksg_subsample_mode"], "fixed_random")
            self.assertEqual(separation_config["ksg_subsample_seed"], 0)
            losses = payload["rss"]["losses"]
            learning_rates = payload["rss"]["learning_rate_schedule"]
            self.assertEqual(payload["rss"]["optimizer"], "adam")
            self.assertEqual(losses["normalization_reference"], result.rss_result.loss_normalization_reference)
            self.assertEqual(losses["normalized_best_loss"], result.rss_result.normalized_best_loss)
            self.assertEqual(losses["raw_best_loss"], result.rss_result.best_loss)
            self.assertEqual(learning_rates["type"], "exponential")
            self.assertEqual(learning_rates["configured_start"], result.rss_result.config.learning_rate)
            self.assertEqual(learning_rates["configured_end"], result.rss_result.config.learning_rate_end)
            self.assertEqual(learning_rates["first_applied"], result.rss_result.applied_learning_rates[0])
            self.assertEqual(learning_rates["terminal_applied"], result.rss_result.applied_learning_rates[-1])
            self.assertEqual(
                payload["rss"]["quaternion_reconstruction"]["full_quaternion_identity_loss"],
                direct_reconstruction_loss,
            )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            export_validation_case_artifacts(
                result,
                output,
                include_plots=False,
            )
            self.assertTrue((output / "metrics.json").exists())
            self.assertTrue((output / "source_comparison.csv").exists())
            self.assertFalse(any(output.glob("*.png")))
        self.assertEqual(
            result.metrics["rss_loss_normalization_reference"],
            result.rss_result.loss_normalization_reference,
        )
        self.assertEqual(result.metrics["rss_normalized_best_loss"], result.rss_result.normalized_best_loss)
        self.assertEqual(result.metrics["rss_optimizer"], "adam")

    def test_metric_distribution_summary_reports_all_requested_statistics(self) -> None:
        families = (
            "harmonic",
            "filtered_noise",
            "chirp_modulated",
            "piecewise_transient",
        )
        frame = pd.DataFrame(
            [
                {
                    "family": family,
                    "source_index": source_index,
                    "metric_a": float(case_index + source_index),
                }
                for family in families
                for case_index in range(50)
                for source_index in (1, 2, 3)
            ]
        )
        summary = describe_metric_distributions(
            frame,
            group_columns=("family",),
            metric_columns=("metric_a",),
        )
        self.assertEqual(set(summary["family"]), set(families))
        self.assertTrue((summary["count"] == 150).all())
        self.assertEqual(
            set(
                (
                    "count",
                    "mean",
                    "std",
                    "min",
                    "p05",
                    "p25",
                    "median",
                    "p75",
                    "p95",
                    "max",
                )
            )
            - set(summary.columns),
            set(),
        )

    def test_direction_cosines_need_observer_basis_to_recover_global_axes(self) -> None:
        scenario = generate_validation_scenario(SimulationConfig(seed=12, T=32))
        recovered_global = source_axes_from_direction_cosines(
            scenario.mixing_matrix,
            scenario.observer_axes_global,
        )
        self.assertTrue(np.allclose(recovered_global, scenario.source_axes_global, atol=1e-12))
        self.assertFalse(np.allclose(source_axes_from_direction_cosines(scenario.mixing_matrix), scenario.source_axes_global))

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RSSConfig(lags=(-1,))
        with self.assertRaises(ValueError):
            RSSConfig(lags=(0, 0))
        with self.assertRaises(ValueError):
            RSSConfig(matrix_parameterization="symmetric", initial_weights=(1.0, 2.0))
        with self.assertRaises(ValueError):
            RSSConfig(initial_weight_matrix=np.ones((2, 2)))
        with self.assertRaises(ValueError):
            RSSConfig(initial_weight_matrix=np.diag((1.0, 1.0, -1.0)))
        with self.assertRaises(ValueError):
            RSSConfig(
                matrix_parameterization="symmetric",
                initial_weight_matrix=np.array(
                    [[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                ),
            )
        with self.assertRaises(ValueError):
            RSSConfig(
                initial_rotation_vector=(0.1, 0.0, 0.0),
                initial_weight_matrix=np.eye(3),
            )
        with self.assertRaises(ValueError):
            RSSConfig(
                matrix_parameterization="symmetric",
                initial_weights=(1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
                initial_weight_matrix=np.eye(3),
            )
        with self.assertRaises(ValueError):
            RSSConfig(orthogonality_weight=-1.0)
        with self.assertRaises(ValueError):
            RSSConfig(orthogonality_weight=float("nan"))
        with self.assertRaises(ValueError):
            RSSConfig(reconstruction_weight=-1.0)
        with self.assertRaises(ValueError):
            SimulationConfig(source_cross_correlation_limit=0.0)
        with self.assertRaises(ValueError):
            SimulationConfig(source_cross_correlation_limit=1.0)
        with self.assertRaises(ValueError):
            source_cross_correlation_matrix(np.ones((3, 16)))
        with self.assertRaises(ValueError):
            RSSConfig(
                separation_weight=0.0,
                orthogonality_weight=0.0, reconstruction_weight=0.0,
            )
        for invalid in (
            {"optimizer": "unknown"},
            {"learning_rate": 0.0},
            {"learning_rate_end": 0.0},
            {"learning_rate": float("nan")},
            {"learning_rate_end": float("inf")},
            {"learning_rate": 1e-3, "learning_rate_end": 1e-2},
            {"sgd_momentum": 1.0},
            {"pso_swarm_size": 0},
            {"pso_inertia": -0.1},
            {"pso_lower_bound": 1.0, "pso_upper_bound": 1.0},
            {"pso_stall_patience": 0},
            {"loss_type": "unknown"},
            {"separation_weight": -1.0},
            {"mi_estimator": "unknown"},
            {"gcmi_rank_method": "random"},
            {"bspline_bins": 1},
            {"bspline_order": 0},
            {"bspline_bins": 3, "bspline_order": 3},
            {"ksg_k": 0},
            {"ksg_k": 2.5},
            {"ksg_standardize": "yes"},
            {"ksg_max_samples": 0},
            {"ksg_k": 4, "ksg_max_samples": 4},
            {"ksg_subsample_mode": "random_each_call"},
            {"ksg_subsample_seed": -1},
            {"mi_bins": 1},
            {"mi_bins": 2.5},
            {"mi_bandwidth": 0.0},
            {"action_inertias": (1.0, 0.0, 1.0)},
            {"action_stiffnesses": (1.0, float("nan"), 1.0)},
            {"sample_interval_seconds": 0.0},
            {
                "optimizer": "pso",
                "pso_lower_bound": -0.5,
                "pso_upper_bound": 0.5,
                "initial_rotation_vector": (1.0, 0.0, 0.0),
            },
            {"matrix_parameterization": "invalid"},
            {"observer_signal_model": "invalid"},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                RSSConfig(**invalid)
        with self.assertRaises(ValueError):
            fit_rss(np.zeros((2, 10)))
        with self.assertRaises(ValueError):
            fit_rss(np.full((3, 10), np.nan))
        with self.assertRaises(ValueError):
            fit_rss(np.zeros((3, 3)), RSSConfig(lags=(3,)))
        with self.assertRaisesRegex(ValueError, "ksg_k"):
            fit_rss(
                np.zeros((3, 4)),
                RSSConfig(
                    loss_type="mutual_information",
                    optimizer="pso",
                    ksg_k=4,
                ),
            )

    def test_cli_parses_and_propagates_optimizer_controls(self) -> None:
        args = build_parser().parse_args(
            [
                "validate", "--rss-orthogonality-weight", "0.375", "--epochs", "1",
                "--rss-optimizer", "pso", "--rss-pso-swarm-size", "12",
                "--rss-pso-seed", "7", "--rss-pso-lower-bound", "-2",
                "--rss-pso-upper-bound", "2",
                "--rss-reconstruction-weight", "0.75",
                "--rss-matrix-parameterization", "so3",
                "--rss-observer-signal-model", "integrated_body_rotation",
                "--rss-loss-type", "mutual_information",
                "--rss-separation-weight", "1.25",
                "--rss-lags", "0", "2",
                "--rss-mi-estimator", "ksg",
                "--rss-gcmi-rank-method", "min",
                "--rss-bspline-bins", "8",
                "--rss-bspline-order", "3",
                "--rss-ksg-k", "10",
                "--no-rss-ksg-standardize",
                "--rss-ksg-max-samples", "128",
                "--rss-ksg-subsample-mode", "fixed_random",
                "--rss-ksg-subsample-seed", "11",
                "--rss-mi-bins", "12",
                "--rss-mi-bandwidth", "0.3",
                "--rss-action-inertias", "1", "2", "3",
                "--rss-action-stiffnesses", "4", "5", "6",
                "--rss-initial-rotation-vector", "0.1", "-0.2", "0.3",
                "--learning-rate-start", "0.2", "--learning-rate-end", "0.002",
            ]
        )
        parsed = _rss_config(args)
        self.assertAlmostEqual(parsed.orthogonality_weight, 0.375)
        self.assertEqual(parsed.optimizer, "pso")
        self.assertEqual(parsed.pso_swarm_size, 12)
        self.assertEqual(parsed.pso_seed, 7)
        self.assertAlmostEqual(parsed.reconstruction_weight, 0.75)
        self.assertAlmostEqual(parsed.learning_rate, 0.2)
        self.assertAlmostEqual(parsed.learning_rate_end, 0.002)
        self.assertEqual(parsed.matrix_parameterization, "so3")
        self.assertEqual(parsed.observer_signal_model, "integrated_body_rotation")
        self.assertEqual(parsed.loss_type, "mutual_information")
        self.assertAlmostEqual(parsed.separation_weight, 1.25)
        self.assertEqual(parsed.lags, (0, 2))
        self.assertEqual(parsed.mi_estimator, "ksg")
        self.assertEqual(parsed.gcmi_rank_method, "min")
        self.assertEqual(parsed.bspline_bins, 8)
        self.assertEqual(parsed.bspline_order, 3)
        self.assertEqual(parsed.ksg_k, 10)
        self.assertFalse(parsed.ksg_standardize)
        self.assertEqual(parsed.ksg_max_samples, 128)
        self.assertEqual(parsed.ksg_subsample_mode, "fixed_random")
        self.assertEqual(parsed.ksg_subsample_seed, 11)
        self.assertEqual(parsed.mi_bins, 12)
        self.assertAlmostEqual(parsed.mi_bandwidth, 0.3)
        self.assertEqual(parsed.action_inertias, (1.0, 2.0, 3.0))
        self.assertEqual(parsed.action_stiffnesses, (4.0, 5.0, 6.0))
        self.assertEqual(parsed.initial_rotation_vector, (0.1, -0.2, 0.3))
        self.assertEqual(parsed.convergence_tol, 0.0)
        legacy_args = build_parser().parse_args(
            ["validate", "--learning-rate", "0.03", "--learning-rate-end", "0.003"]
        )
        legacy_config = _rss_config(legacy_args)
        self.assertAlmostEqual(legacy_config.learning_rate, 0.03)
        self.assertAlmostEqual(legacy_config.learning_rate_end, 0.003)
        self.assertEqual(legacy_config.loss_type, "lagged_cross_correlation")
        self.assertEqual(legacy_config.lags, (0,))
        self.assertEqual(legacy_config.mi_estimator, "ksg")
        self.assertEqual(legacy_config.gcmi_rank_method, "average")
        self.assertEqual(legacy_config.bspline_bins, 6)
        self.assertEqual(legacy_config.bspline_order, 3)
        self.assertEqual(legacy_config.ksg_k, 4)
        self.assertTrue(legacy_config.ksg_standardize)
        self.assertIsNone(legacy_config.ksg_max_samples)
        self.assertEqual(legacy_config.ksg_subsample_mode, "fixed_random")
        self.assertEqual(legacy_config.ksg_subsample_seed, 0)
        self.assertEqual(legacy_config.orthogonality_weight, 0.0)
        self.assertEqual(legacy_config.reconstruction_weight, 0.0)
        matrix_args = build_parser().parse_args(
            [
                "validate",
                "--rss-initial-weight-matrix",
                "0", "-1", "0",
                "1", "0", "0",
                "0", "0", "1",
            ]
        )
        matrix_config = _rss_config(matrix_args)
        self.assertTrue(
            np.allclose(
                matrix_config.resolved_initial_matrix,
                np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            )
        )
        realdata_args = build_parser().parse_args(
            [
                "run-config",
                "--config",
                "unused.json",
                "--output-root",
                "outputs/rss-test",
                "--rss-orthogonality-weight",
                "0.625",
                "--rss-optimizer",
                "sgd",
                "--rss-sgd-momentum",
                "0.8",
                "--rss-reconstruction-weight",
                "0.6",
                "--learning-rate-start",
                "0.04",
                "--learning-rate-end",
                "0.0004",
            ]
        )
        real_config = build_realdata_runtime_config_from_args(realdata_args).rss_config
        self.assertAlmostEqual(real_config.orthogonality_weight, 0.625)
        self.assertEqual(real_config.optimizer, "sgd")
        self.assertAlmostEqual(real_config.sgd_momentum, 0.8)
        self.assertAlmostEqual(real_config.reconstruction_weight, 0.6)
        self.assertAlmostEqual(real_config.learning_rate, 0.04)
        self.assertAlmostEqual(real_config.learning_rate_end, 0.0004)


class RSSRealDataTests(unittest.TestCase):
    def test_partial_limb_pipeline_writes_compatible_default_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "Sample_Data" / "Subject" / "Trial" / "recording.csv"
            csv_path.parent.mkdir(parents=True)
            frames = 24
            time = np.arange(frames) / 100.0
            pd.DataFrame(
                {
                    "device5_IMU3_GX": 10.0 * np.sin(2.0 * np.pi * time),
                    "device5_IMU3_GY": 8.0 * np.cos(2.0 * np.pi * 1.5 * time),
                    "device5_IMU3_GZ": 6.0 * np.sin(2.0 * np.pi * 2.0 * time),
                }
            ).to_csv(csv_path, index=False)
            config_path = root / "config.json"
            config_path.write_text(json.dumps([{"csv_path": str(csv_path), "compute_device": "cpu"}]), encoding="utf-8")
            runtime = RealDataRuntimeConfig(
                output_root=str(root / "outputs" / "rss"),
                sample_rate_hz=100.0,
                include_time_seconds=True,
                file_workers=1,
                rss_config=RSSConfig(epochs=1, convergence_tol=0.0),
                allow_partial_limbs=True,
                ssa_enabled=False,
            )
            output = run_realdata_from_config(config_path=str(config_path), runtime_cfg=runtime)[0]
            self.assertEqual(output.processed_limb_names, ("pelvis_to_lower_spine",))
            expected = {
                "limb_angle_mode.csv",
                "limb_axes_mode.csv",
                "limb_quaternion.csv",
                "limb_quaternion_axis_angle.csv",
            }
            self.assertEqual({path.name for path in Path(output.output_dir).iterdir()}, expected)
            angle_frame = pd.read_csv(output.limb_angle_mode_csv)
            self.assertEqual(
                list(angle_frame.columns),
                [
                    "frame",
                    "time",
                    "pelvis_to_lower_spine_mode_1",
                    "pelvis_to_lower_spine_mode_2",
                    "pelvis_to_lower_spine_mode_3",
                    "pelvis_to_lower_spine_original_angle",
                ],
            )

    def test_rss_mode_order_amplitude_sorts_exported_modes_by_rms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "Sample_Data" / "Subject" / "Trial" / "recording.csv"
            csv_path.parent.mkdir(parents=True)
            frames = 24
            time = np.arange(frames) / 100.0
            pd.DataFrame(
                {
                    "device5_IMU3_GX": 1.0 * np.sin(2.0 * np.pi * time),
                    "device5_IMU3_GY": 2.0 * np.cos(2.0 * np.pi * 2.0 * time),
                    "device5_IMU3_GZ": 3.0 * np.sin(2.0 * np.pi * 3.0 * time),
                }
            ).to_csv(csv_path, index=False)
            config_path = root / "config.json"
            config_path.write_text(json.dumps([{"csv_path": str(csv_path), "compute_device": "cpu"}]), encoding="utf-8")
            runtime = RealDataRuntimeConfig(
                output_root=str(root / "outputs" / "rss"),
                sample_rate_hz=100.0,
                include_time_seconds=True,
                file_workers=1,
                rss_config=RSSConfig(lags=(1, 2), epochs=1, convergence_tol=0.0),
                enabled_outputs=("limb_angle_modes",),
                allow_partial_limbs=True,
                ssa_enabled=False,
                rss_mode_order="amplitude",
            )
            output = run_realdata_from_config(config_path=str(config_path), runtime_cfg=runtime)[0]
            limb_name = "pelvis_to_lower_spine"
            frame = pd.read_csv(output.limb_angle_mode_csv)
            rms = [
                float(np.sqrt(np.mean(frame[f"{limb_name}_mode_{index + 1}"] ** 2)))
                for index in range(3)
            ]
            self.assertGreaterEqual(rms[0], rms[1])
            self.assertGreaterEqual(rms[1], rms[2])

    def test_realdata_metrics_include_normalized_and_raw_losses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "Sample_Data" / "Subject" / "Trial" / "recording.csv"
            csv_path.parent.mkdir(parents=True)
            frames = 24
            time = np.arange(frames) / 100.0
            pd.DataFrame(
                {
                    "device5_IMU3_GX": 10.0 * np.sin(2.0 * np.pi * time),
                    "device5_IMU3_GY": 8.0 * np.cos(2.0 * np.pi * 1.5 * time),
                    "device5_IMU3_GZ": 6.0 * np.sin(2.0 * np.pi * 2.0 * time),
                }
            ).to_csv(csv_path, index=False)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps([{"csv_path": str(csv_path), "compute_device": "cpu"}]),
                encoding="utf-8",
            )
            runtime = RealDataRuntimeConfig(
                output_root=str(root / "outputs" / "rss"),
                sample_rate_hz=100.0,
                include_time_seconds=True,
                file_workers=1,
                rss_config=RSSConfig(loss_type="action", epochs=1, convergence_tol=0.0),
                enabled_outputs=("limb_mode_metrics",),
                allow_partial_limbs=True,
                ssa_enabled=False,
            )
            output = run_realdata_from_config(config_path=str(config_path), runtime_cfg=runtime)[0]
            metrics = pd.read_csv(output.limb_mode_metrics_csv)
            expected = {
                "rss_loss_normalization_reference",
                "rss_sample_interval_seconds",
                "rss_normalized_initial_loss",
                "rss_normalized_best_loss",
                "rss_best_normalized_separation_loss",
                "rss_best_normalized_orthogonality_loss",
                "rss_best_normalized_reconstruction_loss",
                "rss_normalized_terminal_loss",
                "rss_initial_loss",
                "rss_best_loss",
                "rss_best_separation_loss",
                "rss_terminal_loss",
                "rss_best_reconstruction_loss",
                "quaternion_reconstruction_mean_error_deg",
                "quaternion_reconstruction_max_error_deg",
                "rss_learning_rate_start",
                "rss_learning_rate_end",
                "rss_first_applied_learning_rate",
                "rss_terminal_applied_learning_rate",
            }
            self.assertTrue(expected.issubset(metrics.columns))
            self.assertTrue(np.isfinite(metrics[list(expected)].to_numpy(dtype=float)).all())
            self.assertEqual(set(metrics["rss_loss_type"]), {"action"})
            self.assertEqual(set(metrics["rss_mi_estimator"]), {"ksg"})
            self.assertEqual(set(metrics["rss_mi_unit"]), {"nats"})
            self.assertEqual(set(metrics["rss_gcmi_rank_method"]), {"average"})
            self.assertEqual(set(metrics["rss_bspline_bins"]), {6})
            self.assertEqual(set(metrics["rss_bspline_order"]), {3})
            self.assertEqual(set(metrics["rss_ksg_k"]), {4})
            self.assertEqual(set(metrics["rss_ksg_standardize"]), {True})
            self.assertTrue(metrics["rss_ksg_max_samples"].isna().all())
            self.assertEqual(set(metrics["rss_ksg_subsample_mode"]), {"fixed_random"})
            self.assertEqual(set(metrics["rss_ksg_subsample_seed"]), {0})
            self.assertTrue(np.allclose(metrics["rss_sample_interval_seconds"], 0.01))

    def test_optional_rss_exports_use_compatible_and_native_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            frame = np.arange(4)
            signals = {"limb": np.arange(12, dtype=float).reshape(3, 4)}
            observed_path = export_limb_observed_signals_csv(
                output_dir=str(output),
                frame_index=frame,
                time_seconds=None,
                observed_by_limb=signals,
                reconstructed_by_limb=signals,
            )
            self.assertEqual(
                list(pd.read_csv(observed_path).columns),
                [
                    "frame", "limb_observer_x", "limb_observer_y", "limb_observer_z",
                    "limb_reconstructed_x", "limb_reconstructed_y", "limb_reconstructed_z",
                ],
            )
            loss_path, _ = export_loss_history_artifacts(
                output_dir=str(output),
                loss_history_by_limb={
                    "limb": ({
                        "epoch": 0.0,
                        "learning_rate": None,
                        "loss_type": "lagged_cross_correlation",
                        "normalized_total_loss": 1.0,
                        "normalized_separation_loss": 0.875,
                        "normalized_orthogonality_loss": 0.125,
                        "normalized_reconstruction_loss": 0.0,
                        "total_loss": 2.0,
                        "separation_loss": 1.75,
                        "orthogonality_loss": 0.25,
                        "reconstruction_loss": 0.0,
                    },)
                },
            )
            self.assertEqual(
                list(pd.read_csv(loss_path).columns),
                [
                    "limb_name", "epoch", "learning_rate", "loss_type",
                    "normalized_total_loss", "normalized_separation_loss",
                    "normalized_orthogonality_loss", "normalized_reconstruction_loss",
                    "total_loss", "separation_loss",
                    "orthogonality_loss", "reconstruction_loss",
                ],
            )
            metrics_path = export_limb_mode_metrics_csv(
                output_dir=str(output),
                metric_rows=[{"limb_name": "limb", "mode_index": 1, "rss_best_loss": 0.5}],
            )
            metrics_columns = set(pd.read_csv(metrics_path).columns)
            self.assertIn("rss_best_loss", metrics_columns)
            self.assertNotIn("sobi_final_offdiag_energy", metrics_columns)


class RSSNotebookTests(unittest.TestCase):
    def test_notebook_exposes_selectable_loss_controls(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "rss_validation.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertEqual(sources.count("    optimizer="), 2)
        self.assertIn("sgd_momentum=0.9", sources)
        self.assertIn("pso_swarm_size=", sources)
        self.assertNotIn("pso_stall_patience=", sources)
        self.assertEqual(sources.count("convergence_tol=0.0"), 2)
        self.assertEqual(
            sources.count("RSS always runs the complete epoch budget."),
            2,
        )
        self.assertEqual(sources.count("learning_rate=1e-1"), 2)
        self.assertEqual(sources.count("learning_rate_end=1e-3"), 2)
        self.assertIn("First applied learning rate", sources)
        self.assertIn("Terminal applied learning rate", sources)
        self.assertEqual(sources.count("reconstruction_weight=0.0"), 2)
        self.assertEqual(sources.count("separation_weight=1.0"), 2)
        self.assertIn("single_loss_type = 'lagged_cross_correlation'", sources)
        self.assertIn("batch_loss_type = 'lagged_cross_correlation'", sources)
        self.assertIn("single_lags = (0,1,2,3,5,7,11,13,17,19,23,29,31,37,41,43,47)", sources)
        self.assertIn("single_mi_estimator = 'gcmi'", sources)
        self.assertIn("single_gcmi_rank_method = 'average'", sources)
        self.assertIn("single_bspline_bins = 6", sources)
        self.assertIn("single_bspline_order = 3", sources)
        self.assertIn("single_ksg_k = 4", sources)
        self.assertIn("single_ksg_standardize = True", sources)
        self.assertIn("single_ksg_max_samples = None", sources)
        self.assertIn("single_ksg_subsample_mode = 'fixed_random'", sources)
        self.assertIn("single_ksg_subsample_seed = 0", sources)
        self.assertIn("batch_mi_estimator = 'histogram'", sources)
        self.assertIn("batch_gcmi_rank_method = 'average'", sources)
        self.assertIn("batch_bspline_bins = 6", sources)
        self.assertIn("batch_bspline_order = 3", sources)
        self.assertIn("batch_ksg_k = 4", sources)
        self.assertIn("batch_ksg_standardize = True", sources)
        self.assertIn("batch_ksg_max_samples = 1000", sources)
        self.assertIn("batch_ksg_subsample_mode = 'fixed_random'", sources)
        self.assertIn("batch_ksg_subsample_seed = 0", sources)
        self.assertIn("single_mi_bins = 16", sources)
        self.assertIn("single_mi_bandwidth = 0.2", sources)
        self.assertIn("single_action_inertias", sources)
        self.assertIn("single_action_stiffnesses", sources)
        self.assertIn("Primary separation loss", sources)
        self.assertIn("MI estimator", sources)
        self.assertIn("GCMI and KSG require PSO", sources)
        self.assertIn("B-spline and histogram MI support Adam, SGD, and PSO", sources)
        self.assertIn("KSG effective sample count", sources)
        self.assertIn("No automatic timing is performed", sources)
        self.assertIn("Applied sample interval", sources)
        self.assertIn("plot_observer_reconstruction", sources)
        self.assertIn("relative_reconstruction_quaternions", sources)
        self.assertIn("quaternion_reconstruction_loss", sources)
        self.assertIn("source_cross_correlation_limit = 0.01", sources)
        self.assertIn("source_cross_correlation_matrix", sources)
        self.assertIn("SOURCE_ZERO_MEAN_ATOL = 1e-12", sources)
        self.assertIn("FILTERED_NOISE_EDGE_RMS_RATIO_LIMIT = 3.0", sources)
        self.assertIn("def validate_zero_mean_sources(scenario):", sources)
        self.assertIn("all_active_sources_cross_time_axis", sources)
        self.assertIn("filtered_noise_max_edge_rms_ratio", sources)
        self.assertIn("Generated source temporal means", sources)
        self.assertIn("Recovery source temporal means", sources)
        self.assertEqual(sources.count("validate_zero_mean_sources(case)"), 6)
        self.assertIn("original_source_max_abs_cross_correlation", sources)
        self.assertIn("separated_source_max_abs_cross_correlation", sources)
        self.assertIn("loss_normalization_reference", sources)
        self.assertIn("normalized_initial_loss", sources)
        self.assertIn("rss_normalized_best_loss", sources)
        self.assertIn("matrix_diagnostics.orthogonality_error", sources)
        self.assertIn("matrix_diagnostics.determinant", sources)
        self.assertIn("def display_one_figure(figure):", sources)
        self.assertIn("plt.close(figure)", sources)
        self.assertNotIn("plt.show()", sources)
        self.assertNotIn("return figure", sources)
        self.assertIn("validation.scenario is not scenario", sources)
        self.assertIn("resolved_single_config", sources)
        self.assertIn("matrix_parameterization='so3'", sources)
        self.assertIn("observer_signal_model='integrated_body_rotation'", sources)
        self.assertIn("plot_moving_rotation_geometry", sources)
        self.assertIn("RUN_FORMAL_EXPERIMENT_SUITE = True", sources)
        self.assertIn("formal_cases_per_family = 50", sources)
        self.assertIn("formal_base_seed = 20000", sources)
        self.assertIn("case_id_prefix='formal_'", sources)
        self.assertIn("build_source_metrics_frame", sources)
        self.assertIn("describe_metric_distributions", sources)
        self.assertIn("include_plots=False", sources)
        self.assertIn("formal_experiments", sources)
        self.assertIn("per_source_metrics.csv", sources)
        self.assertIn("family_case_metric_distributions.csv", sources)
        self.assertIn("family_source_metric_distributions.csv", sources)
        self.assertIn("family_source_index_metric_distributions.csv", sources)
        self.assertIn("family_source_metric_boxplots.png", sources)
        self.assertIn("family_source_metric_ecdfs.png", sources)
        self.assertIn("from bss_baselines import (", sources)
        self.assertIn("BSSConfig", sources)
        self.assertIn("methods=('pca', 'jade_ica', 'fastica', 'sobi')", sources)
        self.assertIn("sobi_lags=(1, 2, 3, 5, 7, 10)", sources)
        self.assertIn("RUN_SINGLE_BSS_BASELINES = True", sources)
        self.assertIn("RUN_STANDARD_BSS_SUITE = True", sources)
        self.assertIn("RUN_STRESS_BSS_SUITE = True", sources)
        self.assertIn("RUN_FORMAL_BSS_SUITE = True", sources)
        self.assertIn("run_bss_methods_for_scenario", sources)
        self.assertIn("scenario.observations", sources)
        self.assertIn("BSS input domain: cumulative observer signals", sources)
        self.assertIn("mean_source_relative_rmse_oracle_scaled", sources)
        self.assertIn("linear_observation_reconstruction_rmse", sources)
        self.assertIn("bss_baselines", sources)
        self.assertIn("comparisons", sources)
        self.assertIn("method_family_average_metrics.csv", sources)
        self.assertIn("method_source_metric_boxplots.png", sources)
        self.assertIn("method_source_metric_ecdfs.png", sources)
        self.assertIn("include_plots=False", sources)
        self.assertNotIn("create_formal_case_specs", sources)
        self.assertNotIn("generate_formal_scenario", sources)
        self.assertIn("RUN_CORRELATION_SENSITIVITY_SWEEP = True", sources)
        self.assertIn("CorrelationStudyConfig", sources)
        self.assertIn("run_correlation_sensitivity_sweep", sources)
        self.assertIn("plot_method_correlation_curves", sources)
        self.assertIn("correlation_sensitivity", sources)
        self.assertIn("correlation_thresholds.csv", sources)
        self.assertIn("target_source_correlation", sources)
        self.assertIn("validate_zero_mean_sources(outcome.scenario)", sources)
        self.assertIn("Expected 200 audited formal cases", sources)
        self.assertNotIn("composition_family_average_metrics.csv", sources)
        self.assertNotIn("amplitude_frequency_trends.png", sources)
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), "rss_validation.ipynb", "exec")

    @unittest.skipUnless(
        importlib.util.find_spec("nbclient") is not None and importlib.util.find_spec("nbformat") is not None,
        "Notebook execution dependencies are unavailable.",
    )
    def test_notebook_executes_in_quick_mode(self) -> None:
        import nbformat
        from nbclient import NotebookClient

        notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "rss_validation.ipynb"
        notebook = nbformat.read(notebook_path, as_version=4)
        with tempfile.TemporaryDirectory() as temporary:
            artifact_path = Path(temporary) / "rss_validation"
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                cell.source = cell.source.replace("QUICK_MODE = False", "QUICK_MODE = True")
                cell.source = cell.source.replace("single_frame_count = 2000", "single_frame_count = 96")
                cell.source = cell.source.replace(
                    "batch_study_config = BatchStudyConfig()",
                    "batch_study_config = BatchStudyConfig(batch_sequence_lengths=(96,))",
                )
                if "BATCH_ARTIFACT_ROOT =" in cell.source:
                    lines = cell.source.splitlines()
                    lines[0] = f"BATCH_ARTIFACT_ROOT = Path({str(artifact_path)!r})"
                    cell.source = "\n".join(lines)
            NotebookClient(notebook, timeout=180, kernel_name="python3").execute(
                cwd=str(notebook_path.parents[1])
            )
            self.assertTrue((artifact_path / "standard_families" / "summary" / "per_case_metrics.csv").exists())
            self.assertTrue((artifact_path / "large_rotation_stress" / "summary" / "stress_metrics.csv").exists())
            self.assertTrue((artifact_path / "formal_experiments" / "summary" / "per_case_metrics.csv").exists())
            self.assertTrue(
                (artifact_path / "formal_experiments" / "summary" / "per_source_metrics.csv").exists()
            )
            self.assertTrue(
                (artifact_path / "formal_experiments" / "summary" / "family_case_metric_distributions.csv").exists()
            )
            self.assertTrue(
                (artifact_path / "formal_experiments" / "summary" / "family_source_metric_distributions.csv").exists()
            )
            self.assertTrue(
                (artifact_path / "formal_experiments" / "summary" / "family_source_index_metric_distributions.csv").exists()
            )
            bss_root = artifact_path / "bss_baselines"
            for method in ("pca", "jade_ica", "fastica", "sobi"):
                self.assertTrue(
                    (bss_root / method / "single_case" / "summary" / "per_case_metrics.csv").exists()
                )
                self.assertTrue(
                    (bss_root / method / "standard_families" / "summary" / "per_case_metrics.csv").exists()
                )
                self.assertTrue(
                    (bss_root / method / "large_rotation_stress" / "summary" / "per_case_metrics.csv").exists()
                )
                self.assertTrue(
                    (bss_root / method / "formal_experiments" / "summary" / "per_case_metrics.csv").exists()
                )
            self.assertTrue(
                (bss_root / "comparisons" / "standard_families" / "summary" / "method_family_average_metrics.csv").exists()
            )
            self.assertTrue(
                (bss_root / "comparisons" / "formal_experiments" / "summary" / "method_source_metric_boxplots.png").exists()
            )
            correlation_root = bss_root / "comparisons" / "correlation_sensitivity" / "summary"
            self.assertTrue((correlation_root / "per_case_metrics.csv").exists())
            self.assertTrue((correlation_root / "correlation_thresholds.csv").exists())
            self.assertTrue((correlation_root / "method_correlation_curves.png").exists())


if __name__ == "__main__":
    unittest.main()
