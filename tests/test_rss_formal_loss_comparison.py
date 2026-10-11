from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_formal_loss_comparison as comparison


class FormalLossComparisonTests(unittest.TestCase):
    def _args(self, output_root: Path, *extra: str):
        arguments = [
            "--output-root",
            str(output_root),
            "--cases-per-family",
            "1",
            "--sequence-lengths",
            "96",
            "--epochs",
            "2",
            "--swarm-size",
            "4",
            "--plot-dpi",
            "40",
            *extra,
        ]
        return comparison.build_parser().parse_args(arguments)

    def test_method_configs_match_the_four_planned_losses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._args(Path(temporary) / "comparison")
            configs = comparison.build_method_configs(args)

        self.assertEqual(tuple(configs), comparison.METHOD_ORDER)
        zero_lag = configs["zero_lag_cross_correlation"]
        self.assertEqual(zero_lag.loss_type, "lagged_cross_correlation")
        self.assertEqual(zero_lag.lags, (0,))

        lagged = configs["lagged_cross_correlation"]
        self.assertEqual(lagged.loss_type, "lagged_cross_correlation")
        self.assertEqual(lagged.lags, comparison.DEFAULT_LAGGED_LAGS)

        mutual_information = configs["mutual_information"]
        self.assertEqual(mutual_information.loss_type, "mutual_information")
        self.assertEqual(mutual_information.mi_estimator, "histogram")
        self.assertEqual(mutual_information.mi_bins, 16)
        self.assertAlmostEqual(mutual_information.mi_bandwidth, 0.2)

        action = configs["action"]
        self.assertEqual(action.loss_type, "action")
        self.assertEqual(action.action_inertias, (1.0, 1.0, 1.0))
        self.assertEqual(action.action_stiffnesses, (1.0, 1.0, 1.0))

        for config in configs.values():
            self.assertEqual(config.optimizer, "pso")
            self.assertEqual(config.matrix_parameterization, "so3")
            self.assertEqual(config.observer_signal_model, "integrated_body_rotation")
            self.assertEqual(config.orthogonality_weight, 0.0)
            self.assertEqual(config.reconstruction_weight, 0.0)

    def test_default_cohort_has_200_unique_balanced_cases(self) -> None:
        args = comparison.build_parser().parse_args([])
        comparison._validate_arguments(args)
        _, specs = comparison._build_cohort(args)
        self.assertEqual(len(specs), 200)
        self.assertEqual(len({spec.case_id for spec in specs}), 200)
        self.assertEqual(len({spec.seed for spec in specs}), 200)
        counts = pd.Series([spec.family for spec in specs]).value_counts().to_dict()
        self.assertEqual(
            counts,
            {
                "harmonic": 50,
                "filtered_noise": 50,
                "chirp_modulated": 50,
                "piecewise_transient": 50,
            },
        )

    def test_zero_mean_source_guard_accepts_centered_sources_and_rejects_offsets(self) -> None:
        centered = np.vstack(
            (
                np.linspace(-1.0, 1.0, 12),
                np.tile((-1.0, 1.0), 6),
                np.zeros(12),
            )
        )
        scenario = SimpleNamespace(
            case_id="centered",
            generated_sources=centered.copy(),
            relative_sources=centered.copy(),
            sources=centered.copy(),
        )
        comparison._validate_zero_mean_sources(scenario)

        scenario.sources[0] += 0.1
        with self.assertRaisesRegex(comparison.ExperimentError, "nonzero temporal means"):
            comparison._validate_zero_mean_sources(scenario)

    def test_dry_run_does_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "dry_run"
            with contextlib.redirect_stdout(io.StringIO()):
                status = comparison.main(
                    [
                        "--output-root",
                        str(output_root),
                        "--dry-run",
                        "--cases-per-family",
                        "1",
                        "--sequence-lengths",
                        "96",
                        "--epochs",
                        "2",
                        "--swarm-size",
                        "4",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertFalse(output_root.exists())

    def test_smoke_outputs_resume_and_configuration_protection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "comparison"
            command = [
                "--output-root",
                str(output_root),
                "--cases-per-family",
                "1",
                "--sequence-lengths",
                "96",
                "--epochs",
                "2",
                "--swarm-size",
                "4",
                "--plot-dpi",
                "40",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                first_status = comparison.main(command)
            self.assertEqual(first_status, 0)

            manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["configuration"]["cohort"]["case_count"], 4)
            self.assertEqual(
                manifest["configuration"]["cohort"]["source_centering"]["policy"],
                "per_source_temporal_mean",
            )
            self.assertAlmostEqual(
                manifest["configuration"]["cohort"]["source_centering"]["absolute_tolerance"],
                comparison.SOURCE_ZERO_MEAN_ATOL,
            )
            self.assertEqual(
                tuple(manifest["configuration"]["method_order"]), comparison.METHOD_ORDER
            )
            self.assertEqual(
                set(manifest["configuration"]["methods"]), set(comparison.METHOD_ORDER)
            )

            case_metrics = pd.read_csv(output_root / "summary" / "per_case_metrics.csv")
            source_metrics = pd.read_csv(output_root / "summary" / "per_source_metrics.csv")
            self.assertEqual(len(case_metrics), 16)
            self.assertEqual(len(source_metrics), 48)
            self.assertEqual(set(case_metrics["loss_method"]), set(comparison.METHOD_ORDER))
            self.assertTrue(case_metrics["mean_abs_source_correlation"].notna().all())
            self.assertTrue(case_metrics["mean_source_rmse"].notna().all())
            failures = pd.read_csv(output_root / "summary" / "failures.csv")
            self.assertTrue(failures.empty)

            ranking = pd.read_csv(output_root / "summary" / "method_ranking_summary.csv")
            best_counts = ranking.groupby("metric")["best_count"].sum()
            self.assertTrue((best_counts >= 4).all())

            expected_summary_files = (
                "descriptive_statistics.csv",
                "paired_case_differences.csv",
                "paired_case_ranks.csv",
                "method_ranking_summary.csv",
                "runtime_summary.csv",
                "convergence_summary.csv",
                "performance_summary.md",
            )
            for name in expected_summary_files:
                self.assertTrue((output_root / "summary" / name).is_file(), name)

            expected_plot_stems = (
                "overall_performance_boxplots",
                "family_stratified_performance",
                "performance_ecdfs",
                "paired_differences_vs_zero_lag",
                "method_ranks_and_win_rates",
                "runtime_distributions",
                "normalized_convergence",
            )
            for stem in expected_plot_stems:
                for suffix in ("png", "pdf"):
                    self.assertGreater((output_root / "plots" / f"{stem}.{suffix}").stat().st_size, 0)

            checkpoint_paths = sorted(output_root.glob("checkpoints/*/*/*/checkpoint.json"))
            self.assertEqual(len(checkpoint_paths), 16)
            modification_times = {path: path.stat().st_mtime_ns for path in checkpoint_paths}
            time.sleep(0.01)
            with contextlib.redirect_stdout(io.StringIO()):
                resumed_status = comparison.main(command)
            self.assertEqual(resumed_status, 0)
            self.assertEqual(
                modification_times,
                {path: path.stat().st_mtime_ns for path in checkpoint_paths},
            )

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                mismatch_status = comparison.main([*command, "--epochs", "3"])
            self.assertEqual(mismatch_status, 2)
            self.assertEqual(
                modification_times,
                {path: path.stat().st_mtime_ns for path in checkpoint_paths},
            )


if __name__ == "__main__":
    unittest.main()
