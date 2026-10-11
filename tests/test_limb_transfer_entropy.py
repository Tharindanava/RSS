from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from limb_transfer_entropy.cli import build_parser, main as cli_main
from limb_transfer_entropy.core import (
    ContextSpec,
    SEGMENT_ORDER,
    SIDE_COLUMNS,
    build_te_samples,
    compute_side_transfer_entropy,
    discretize_signal,
    estimate_transfer_entropy_from_samples,
)
from limb_transfer_entropy.pipeline import BatchConfig, run_batch


def make_mode_frame(rows: int = 240, *, include_right: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(20260706)
    time = np.arange(rows, dtype=float)
    source = rng.normal(size=rows)
    signals = {
        "Pelvis-Hip": source,
        "Hip-Knee": np.roll(source, 1) + 0.05 * rng.normal(size=rows),
        "Knee-Ankle": np.roll(source, 2) + 0.08 * rng.normal(size=rows),
        "Ankle-Foot": np.roll(source, 3) + 0.1 * rng.normal(size=rows),
    }
    data: dict[str, np.ndarray] = {
        "frame": np.arange(rows),
        "time": time / 100.0,
    }
    for segment, column in SIDE_COLUMNS["left"].items():
        data[column] = signals[segment]
    if include_right:
        for index, (segment, column) in enumerate(SIDE_COLUMNS["right"].items()):
            data[column] = signals[segment] + 0.01 * index
    return pd.DataFrame(data)


class TransferEntropyCoreTests(unittest.TestCase):
    def test_build_samples_uses_exact_lag_length_and_stride(self) -> None:
        source = np.arange(12)
        target = np.arange(100, 112)
        context = ContextSpec(lag=2, length=3, stride=2)

        samples = build_te_samples(source, target, context)

        self.assertEqual(context.max_lag, 6)
        self.assertEqual(len(samples), 6)
        self.assertEqual(samples[0], (106, (104, 102, 100), (4, 2, 0)))
        self.assertEqual(samples[-1], (111, (109, 107, 105), (9, 7, 5)))

    def test_source_driven_signal_has_stronger_forward_te(self) -> None:
        rng = np.random.default_rng(42)
        source = rng.integers(0, 4, size=5000)
        target = np.zeros_like(source)
        target[1:] = source[:-1]
        context = ContextSpec(lag=1, length=1, stride=1)

        forward = estimate_transfer_entropy_from_samples(
            build_te_samples(source, target, context)
        )
        reverse = estimate_transfer_entropy_from_samples(
            build_te_samples(target, source, context)
        )

        self.assertGreater(forward, 1.8)
        self.assertGreater(forward, reverse + 1.5)

    def test_quantile_uniform_duplicates_and_constant_signals(self) -> None:
        values = np.repeat(np.arange(4, dtype=float), 25)
        quantile, quantile_bins = discretize_signal(values, bins=8, method="quantile")
        uniform, uniform_bins = discretize_signal(values, bins=8, method="uniform")
        constant, constant_bins = discretize_signal(
            np.ones(20), bins=8, method="quantile"
        )

        self.assertEqual(quantile.shape, values.shape)
        self.assertLessEqual(quantile_bins, 4)
        self.assertEqual(uniform.shape, values.shape)
        self.assertEqual(uniform_bins, 4)
        np.testing.assert_array_equal(constant, np.zeros(20, dtype=int))
        self.assertEqual(constant_bins, 1)

    def test_invalid_context_and_discretization_arguments(self) -> None:
        for context in (
            ContextSpec(lag=0),
            ContextSpec(length=0),
            ContextSpec(stride=0),
        ):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                context.validate()
        with self.assertRaisesRegex(ValueError, "greater than or equal to 2"):
            discretize_signal(np.arange(10), bins=1, method="quantile")
        with self.assertRaisesRegex(ValueError, "quantile.*uniform"):
            discretize_signal(np.arange(10), bins=3, method="bad")

    def test_side_matrix_uses_common_finite_mask_and_has_blank_diagonal(self) -> None:
        frame = make_mode_frame(rows=100)
        frame.loc[3, SIDE_COLUMNS["left"]["Hip-Knee"]] = np.nan
        frame[SIDE_COLUMNS["left"]["Ankle-Foot"]] = frame[
            SIDE_COLUMNS["left"]["Ankle-Foot"]
        ].astype(object)
        frame.loc[7, SIDE_COLUMNS["left"]["Ankle-Foot"]] = "bad"
        result = compute_side_transfer_entropy(
            frame,
            side="left",
            bins=4,
            binning="quantile",
            context=ContextSpec(lag=2, length=2, stride=2),
        )

        self.assertEqual(result.n_samples_aligned, 98)
        self.assertEqual(result.matrix.shape, (4, 4))
        self.assertTrue(np.isnan(np.diag(result.matrix.to_numpy())).all())
        self.assertEqual(len(result.edges), 12)
        self.assertTrue((result.edges["n_samples_used"] == 94).all())
        self.assertEqual(result.matrix.index.tolist(), list(SEGMENT_ORDER))

    def test_insufficient_embedded_samples_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "embedded samples"):
            compute_side_transfer_entropy(
                make_mode_frame(rows=30),
                side="left",
                bins=4,
                binning="uniform",
                context=ContextSpec(lag=5, length=2, stride=1),
            )


class LimbTransferEntropyBatchTests(unittest.TestCase):
    def test_recursive_batch_mirrors_tree_and_writes_all_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "inputs"
            run_folder = input_root / "subject" / "condition" / "run_01"
            run_folder.mkdir(parents=True)
            make_mode_frame().to_csv(run_folder / "limb_angle_mode.csv", index=False)
            output_root = root / "outputs"

            result = run_batch(
                BatchConfig(
                    input_root=input_root,
                    output_root=output_root,
                    bins=5,
                    binning="quantile",
                    context=ContextSpec(lag=1, length=2, stride=1),
                )
            )

            mirrored = output_root / "subject" / "condition" / "run_01"
            self.assertEqual(result.files_discovered, 1)
            self.assertEqual(result.sides_succeeded, 2)
            self.assertEqual(result.sides_failed, 0)
            for side in ("left", "right"):
                self.assertTrue((mirrored / f"{side}_te_matrix.csv").is_file())
                self.assertTrue((mirrored / f"{side}_te_matrix.png").is_file())
                self.assertTrue((mirrored / f"{side}_te_edges.csv").is_file())
                matrix = pd.read_csv(
                    mirrored / f"{side}_te_matrix.csv",
                    index_col="source_segment",
                )
                self.assertEqual(matrix.shape, (4, 4))
                edges = pd.read_csv(mirrored / f"{side}_te_edges.csv")
                self.assertEqual(len(edges), 12)
                self.assertTrue((edges["bins_requested"] == 5).all())
                self.assertTrue((edges["context_length"] == 2).all())

            summary = pd.read_csv(result.summary_csv)
            errors = pd.read_csv(result.errors_csv)
            self.assertEqual(len(summary), 2)
            self.assertTrue((summary["status"] == "success").all())
            self.assertTrue(errors.empty)

    def test_missing_side_is_logged_without_losing_successful_side(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "inputs"
            run_folder = input_root / "subject" / "run"
            run_folder.mkdir(parents=True)
            make_mode_frame(include_right=False).to_csv(
                run_folder / "limb_angle_mode.csv",
                index=False,
            )
            output_root = root / "outputs"

            result = run_batch(
                BatchConfig(input_root=input_root, output_root=output_root)
            )

            self.assertEqual(result.sides_succeeded, 1)
            self.assertEqual(result.sides_failed, 1)
            self.assertTrue(
                (output_root / "subject" / "run" / "left_te_matrix.csv").is_file()
            )
            errors = pd.read_csv(result.errors_csv)
            self.assertEqual(errors["side"].tolist(), ["right"])
            self.assertIn("Missing required right", errors.loc[0, "error_message"])

    def test_cli_parser_defaults_and_temporary_smoke_run(self) -> None:
        defaults = build_parser().parse_args([])
        self.assertEqual(defaults.bins, 8)
        self.assertEqual(defaults.binning, "quantile")
        self.assertEqual(defaults.context_length, 1)
        self.assertEqual(defaults.lag, 1)
        self.assertEqual(defaults.stride, 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "inputs"
            input_root.mkdir()
            make_mode_frame().to_csv(input_root / "limb_angle_mode.csv", index=False)
            output_root = root / "results"

            exit_code = cli_main(
                [
                    "--input-root",
                    str(input_root),
                    "--output-root",
                    str(output_root),
                    "--bins",
                    "4",
                    "--binning",
                    "uniform",
                    "--context-length",
                    "2",
                    "--lag",
                    "2",
                    "--stride",
                    "2",
                    "--log-level",
                    "ERROR",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_root / "left_te_matrix.png").is_file())
            self.assertTrue((output_root / "processing_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
