from __future__ import annotations

import base64
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HMA_Dashboard.app import create_app
from HMA_Dashboard import app as dashboard_app
from HMA_Dashboard import callbacks as dashboard_callbacks
from HMA_Dashboard import catalog
from HMA_Dashboard import flips
from HMA_Dashboard import jobs
from HMA_Dashboard import layout
from HMA_Dashboard import plotting
from HMA_Dashboard import signals
from HMA_Dashboard import raw_catalog
from HMA_Dashboard import references as dashboard_references
from HMA_Dashboard import signal_logs
from HMA_Dashboard import te_matrix
from HMA_Dashboard import transfer_entropy
from rss.exporters import resolve_realdata_output_dir


def _write_mode_file(path: Path, *, groups: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = 48
    rows: dict[str, np.ndarray] = {
        "frame": np.arange(samples, dtype=int),
        "time": np.arange(samples, dtype=float) / 100.0,
    }
    for index, group in enumerate(groups):
        base = 0.2 * np.sin(np.linspace(0.0, 4.0 * np.pi, samples) + 0.3 * index)
        rows[f"{group}_mode_1"] = base
        rows[f"{group}_mode_2"] = 0.5 * base
        rows[f"{group}_mode_3"] = -0.25 * base
        rows[f"{group}_original_angle"] = base + 0.1
    pd.DataFrame(rows).to_csv(path, index=False)


class RmdDashboardTests(unittest.TestCase):
    def test_catalog_discovers_joint_and_limb_runs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "outputs"
            experiment = "1757660547_12_09_2025_Flat_0,0_40Steps_JRDC_Trial02"
            csv_stem = "combined_log_wide_1757660547_Phase_01"
            limb_run = (
                root
                / "limb_outputs"
                / "Default_Outputs"
                / "20260707_072314"
                / "Human_Motion"
                / "Subject_A"
                / "Flat Terrain"
                / "0,0_Walking"
                / experiment
                / csv_stem
            )
            joint_run = root / "Unlabelled_Joint_Results" / "0_0_Walking" / "run_beta"
            _write_mode_file(limb_run / "limb_angle_mode.csv", groups=["pelvis_to_right_hip", "right_hip_to_knee"])
            _write_mode_file(joint_run / "joint_angle_mode.csv", groups=["right_hip", "right_knee", "right_ankle"])
            _write_mode_file(joint_run / "joint_axes_mode.csv", groups=["right_hip", "right_knee", "right_ankle"])

            catalog.clear_caches()
            catalog_data = catalog.scan_results_catalog(root)

            self.assertTrue(catalog_data["exists"])
            self.assertEqual(catalog_data["family_counts"]["limb"], 1)
            self.assertEqual(catalog_data["family_counts"]["joint"], 1)
            self.assertEqual(len(catalog_data["run_options"]), 2)

            limb_run_id = catalog.get_first_run_id(catalog_data, family="limb")
            joint_run_id = catalog.get_first_run_id(catalog_data, family="joint")
            self.assertIsNotNone(limb_run_id)
            self.assertIsNotNone(joint_run_id)

            limb_record = catalog.get_run_record(catalog_data, limb_run_id)
            joint_record = catalog.get_run_record(catalog_data, joint_run_id)
            self.assertEqual(limb_record["available_groups"], ["pelvis_to_right_hip", "right_hip_to_knee"])
            self.assertEqual(joint_record["available_groups"], ["right_ankle", "right_hip", "right_knee"])
            self.assertTrue(joint_record["axes_file_path"].endswith("joint_axes_mode.csv"))
            self.assertEqual(limb_record["subject"], "Subject_A")
            self.assertEqual(limb_record["terrain"], "Flat Terrain")
            self.assertEqual(limb_record["activity"], "0,0_Walking")
            self.assertEqual(limb_record["experiment"], experiment)
            self.assertEqual(limb_record["csv_file"], f"{csv_stem} - limb_angle_mode.csv")
            self.assertEqual(
                catalog.get_hierarchy_options(catalog_data, "subject"),
                [
                    {"label": "Subject_A", "value": "Subject_A"},
                    {"label": "Unlabelled_Joint_Results", "value": "Unlabelled_Joint_Results"},
                ],
            )
            self.assertEqual(
                catalog.get_hierarchy_options(catalog_data, "terrain", subject="Subject_A"),
                [{"label": "Flat Terrain", "value": "Flat Terrain"}],
            )
            self.assertEqual(
                catalog.get_hierarchy_options(
                    catalog_data,
                    "activity",
                    subject="Subject_A",
                    terrain="Flat Terrain",
                ),
                [{"label": "0,0_Walking", "value": "0,0_Walking"}],
            )
            self.assertEqual(
                catalog.get_hierarchy_options(
                    catalog_data,
                    "experiment",
                    subject="Subject_A",
                    terrain="Flat Terrain",
                    activity="0,0_Walking",
                ),
                [{"label": experiment, "value": experiment}],
            )
            self.assertEqual(
                catalog.get_run_options(
                    catalog_data,
                    subject="Subject_A",
                    terrain="Flat Terrain",
                    activity="0,0_Walking",
                    experiment=experiment,
                ),
                [{"label": f"{csv_stem} - limb_angle_mode.csv", "value": limb_run_id}],
            )

    def test_catalog_excludes_all_nested_backup_runs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "outputs"
            _write_mode_file(root / "rss" / "real" / "limb_angle_mode.csv", groups=["left_hip_to_knee"])
            _write_mode_file(root / "hma_dashboard_flip_backup" / "nested" / "limb_angle_mode.csv", groups=["left_hip_to_knee"])
            _write_mode_file(root / "rss" / "hilbert_analysis_backup" / "nested" / "limb_angle_mode.csv", groups=["left_hip_to_knee"])
            catalog.clear_caches()
            discovered = catalog.scan_results_catalog(root)
            self.assertEqual(len(discovered["runs"]), 1)
            self.assertIn("/real", discovered["runs"][0]["run_dir"])

    def test_signal_schema_and_trace_loading(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            mode_file = Path(td) / "joint_angle_mode.csv"
            _write_mode_file(mode_file, groups=["right_hip", "right_knee", "right_ankle"])

            schema = catalog.read_mode_file_schema(mode_file)
            self.assertEqual(schema["groups"], ["right_ankle", "right_hip", "right_knee"])
            self.assertIn("original_angle", schema["group_traces"]["right_hip"])

            group_data = signals.load_group_data(mode_file, "right_hip")
            self.assertEqual(group_data["group_label"], "Right Hip")
            self.assertEqual(group_data["row_count"], 48)
            self.assertEqual(sorted(group_data["trace_names"]), ["mode_1", "mode_2", "mode_3", "original_angle"])

            trace_data = signals.load_signal_trace(mode_file, "right_hip", "mode_2")
            self.assertEqual(trace_data["trace_name"], "mode_2")
            self.assertEqual(trace_data["x_label"], "Time (s)")
            self.assertEqual(trace_data["values"].shape[0], 48)

    def test_signal_loading_repairs_legacy_nanosecond_time(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            mode_file = Path(td) / "limb_angle_mode.csv"
            _write_mode_file(mode_file, groups=["right_hip_to_knee"])
            frame = pd.read_csv(mode_file)
            frame["time"] = np.arange(len(frame), dtype=float) * 1.0e-9
            frame.to_csv(mode_file, index=False)

            signals.clear_caches()
            trace_data = signals.load_signal_trace(
                mode_file,
                "right_hip_to_knee",
                "mode_1",
                default_fs=100.0,
            )

            np.testing.assert_allclose(trace_data["time"][:4], [0.0, 0.01, 0.02, 0.03])
            self.assertEqual(trace_data["fs"], 100.0)

    def test_generalized_te_sample_builder_respects_independent_contexts(self) -> None:
        source = np.arange(10, dtype=int)
        target = np.arange(100, 110, dtype=int)
        samples = transfer_entropy.build_te_samples(
            source,
            target,
            source_context=transfer_entropy.ContextSpec(lag=2, length=2, stride=2),
            receiver_context=transfer_entropy.ContextSpec(lag=1, length=3, stride=1),
        )
        self.assertEqual(len(samples), 6)
        self.assertEqual(samples[0], (104, (103, 102, 101), (2, 0)))

    def test_pairwise_transfer_entropy_returns_positive_value_for_shifted_driver(self) -> None:
        rng = np.random.default_rng(123)
        source = rng.integers(0, 2, size=256).astype(float)
        target = np.roll(source, 1)
        target[0] = 0.0

        result = transfer_entropy.compute_pairwise_transfer_entropy(
            source,
            target,
            source_context=transfer_entropy.ContextSpec(lag=1, length=1, stride=1),
            receiver_context=transfer_entropy.ContextSpec(lag=1, length=1, stride=1),
            bins=2,
        )

        self.assertTrue(result["available"])
        self.assertGreater(result["te_bits"], 0.1)
        self.assertGreaterEqual(result["n_samples_used"], transfer_entropy.MIN_EMBEDDED_SAMPLES)

    def test_pairwise_gating_requires_same_mode_file(self) -> None:
        signal_a = {"mode_file_path": "C:/one/joint_angle_mode.csv"}
        signal_b = {"mode_file_path": "C:/two/joint_angle_mode.csv"}
        signal_c = {"mode_file_path": "C:/one/joint_angle_mode.csv"}

        self.assertFalse(signals.can_pair_signals(signal_a, signal_b))
        self.assertTrue(signals.can_pair_signals(signal_a, signal_c))

    def test_workbench_layout_uses_single_plot_dropdown(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "outputs"
            run_dir = root / "Unlabelled_Joint_Results" / "0_0_Walking" / "run_beta"
            _write_mode_file(run_dir / "joint_angle_mode.csv", groups=["right_hip", "right_knee", "right_ankle"])

            catalog.clear_caches()
            app = create_app(results_dir=root)
            component_ids = _collect_ids(app.layout)

            self.assertIn("workbench-plot-kind", component_ids)
            self.assertIn("workbench-single-plot", component_ids)
            self.assertIn("phase-unwrap-toggle", component_ids)
            self.assertIn("overview-phase-unwrap-toggle", component_ids)
            self.assertIn("overview-subject-dropdown", component_ids)
            self.assertIn("overview-terrain-dropdown", component_ids)
            self.assertIn("overview-activity-dropdown", component_ids)
            self.assertIn("overview-experiment-dropdown", component_ids)
            self.assertIn("overview-run-dropdown", component_ids)
            for te_id in (
                "overview-te-matrix",
                "overview-te-matrix-status",
                "overview-te-matrix-compute",
                "overview-te-bins",
                "overview-te-lag",
                "overview-te-show-values",
                "overview-te-matrix-download-button",
                "overview-te-matrix-download",
            ):
                self.assertIn(te_id, component_ids)
            for flip_id in (
                "flip-preview-store",
                "flip-pending-summary",
                "flip-commit-status",
                "flip-clear-button",
                *(layout.flip_status_id(scope) for scope in layout.FLIP_SCOPES),
            ):
                self.assertIn(flip_id, component_ids)
            for pipeline_id in (
                "ingest-upload",
                "ingest-parent-folder",
                "ingest-new-subpath",
                "ingest-filename-override",
                "ingest-overwrite-toggle",
                "ingest-path-preview",
                "ingest-status",
                "run-input-dropdown",
                "run-epochs",
                "run-ssa-window",
                "run-compute-device",
                "run-output-target",
                "run-overwrite-confirm",
                "run-start-button",
                "jobs-poll-interval",
                "jobs-panel",
                "jobs-log-selector",
                "jobs-log-tail",
                "raw-catalog-store",
                "jobs-state-store",
                "jobs-seen-store",
                "jobs-refresh-token",
            ):
                self.assertIn(pipeline_id, component_ids)
            self.assertNotIn("workbench-time-series", component_ids)
            self.assertNotIn("workbench-fft", component_ids)
            self.assertNotIn("workbench-pdf", component_ids)
            self.assertNotIn("workbench-cdf", component_ids)
            self.assertNotIn("workbench-phase-portrait", component_ids)
            self.assertNotIn("workbench-hilbert-analytic", component_ids)

    def test_workbench_single_plot_helpers_route_by_plot_kind(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            mode_file = Path(td) / "joint_angle_mode.csv"
            _write_mode_file(mode_file, groups=["right_hip", "right_knee"])

            signal_a = signals.load_signal_trace(mode_file, "right_hip", "mode_1")
            signal_b = signals.load_signal_trace(mode_file, "right_knee", "mode_2")
            signal_a["legend_name"] = "A"
            signal_b["legend_name"] = "B"
            signal_a["series_key"] = "a"
            signal_b["series_key"] = "b"

            time_figure = dashboard_callbacks._build_workbench_single_plot_figure("time-series", signal_a, signal_b, 12)
            fft_figure = dashboard_callbacks._build_workbench_single_plot_figure("fft", signal_a, signal_b, 12)
            pdf_figure = dashboard_callbacks._build_workbench_single_plot_figure("pdf", signal_a, signal_b, 12)
            cdf_figure = dashboard_callbacks._build_workbench_single_plot_figure("cdf", signal_a, signal_b, 12)
            phase_figure = dashboard_callbacks._build_workbench_single_plot_figure("phase-portrait", signal_a, signal_b, 12)
            hilbert_figure = dashboard_callbacks._build_workbench_single_plot_figure("hilbert-analytic", signal_a, signal_b, 12)
            inst_phase_figure = dashboard_callbacks._build_workbench_single_plot_figure(
                "instantaneous-phase", signal_a, signal_b, 12
            )
            inst_frequency_figure = dashboard_callbacks._build_workbench_single_plot_figure(
                "instantaneous-frequency", signal_a, signal_b, 12
            )
            poincare_signal_a = dict(signal_a)
            poincare_signal_b = dict(signal_b)
            poincare_signal_a["values"] = 100.0 + 1000.0 * signal_a["values"]
            poincare_signal_b["values"] = -50.0 + 0.001 * signal_b["values"]
            poincare_figure = dashboard_callbacks._build_workbench_single_plot_figure(
                "poincare-transform", poincare_signal_a, poincare_signal_b, 12
            )
            cdf_frame = dashboard_callbacks._build_workbench_single_plot_frame("cdf", signal_a, signal_b, 12)
            fft_frame = dashboard_callbacks._build_workbench_single_plot_frame("fft", signal_a, signal_b, 12)
            inst_phase_frame = dashboard_callbacks._build_workbench_single_plot_frame(
                "instantaneous-phase", signal_a, signal_b, 12
            )
            inst_frequency_frame = dashboard_callbacks._build_workbench_single_plot_frame(
                "instantaneous-frequency", signal_a, signal_b, 12
            )
            poincare_frame = dashboard_callbacks._build_workbench_single_plot_frame(
                "poincare-transform", signal_a, signal_b, 12
            )

            self.assertEqual(time_figure.layout.title.text, "Time Series")
            self.assertEqual(pdf_figure.layout.title.text, "PDF")
            self.assertEqual(inst_phase_figure.layout.title.text, "Instantaneous Phase")
            self.assertEqual(inst_frequency_figure.layout.title.text, "Instantaneous Frequency")
            self.assertEqual(
                [column for column in inst_phase_frame.columns],
                ["a_x", "a_instantaneous_phase", "b_x", "b_instantaneous_phase"],
            )
            self.assertEqual(
                [column for column in inst_frequency_frame.columns],
                [
                    "a_x",
                    "a_instantaneous_frequency_hz",
                    "b_x",
                    "b_instantaneous_frequency_hz",
                ],
            )
            for figure in (
                time_figure,
                fft_figure,
                pdf_figure,
                cdf_figure,
                phase_figure,
                hilbert_figure,
                inst_phase_figure,
                inst_frequency_figure,
            ):
                self.assertEqual(figure.layout.legend.itemclick, "toggle")
                self.assertEqual(figure.layout.legend.itemdoubleclick, "toggleothers")
                self.assertTrue(all(trace.showlegend for trace in figure.data))
                self.assertEqual(
                    [trace.legendgroup for trace in figure.data],
                    ["a", "b"],
                )
            self.assertEqual(poincare_figure.layout.title.text, "Positive-Ray Poincaré Transform")
            self.assertEqual(poincare_figure.layout.legend.itemclick, "toggle")
            self.assertEqual(poincare_figure.layout.legend.itemdoubleclick, "toggleothers")
            self.assertEqual(poincare_figure.layout.legend.groupclick, "togglegroup")
            self.assertEqual(poincare_figure.layout.plot_bgcolor, "#440154")
            self.assertEqual(float(poincare_figure.layout.title.x), 0.5)
            self.assertEqual(int(poincare_figure.layout.title.font.size), 24)
            self.assertFalse(poincare_figure.layout.xaxis.showgrid)
            self.assertFalse(poincare_figure.layout.yaxis.showgrid)
            self.assertTrue(poincare_figure.layout.xaxis.mirror)
            self.assertTrue(poincare_figure.layout.yaxis.mirror)
            self.assertEqual(len(poincare_figure.data), 4)
            self.assertEqual(
                [trace.legendgroup for trace in poincare_figure.data],
                ["a", "b", "a", "b"],
            )
            self.assertEqual(
                [trace.showlegend for trace in poincare_figure.data],
                [True, True, False, False],
            )
            self.assertNotEqual(
                poincare_figure.data[0].colorscale[-1][1],
                poincare_figure.data[1].colorscale[-1][1],
            )
            self.assertEqual(poincare_figure.data[0].colorscale[0][1], "#440154")
            self.assertEqual(
                poincare_figure.data[1].colorscale[0][1],
                "rgba(13,8,135,0.0)",
            )
            self.assertFalse(poincare_figure.data[0].zsmooth)
            self.assertFalse(poincare_figure.data[1].zsmooth)
            np.testing.assert_allclose(
                poincare_figure.data[0].x,
                poincare_figure.data[1].x,
            )
            for heatmap, mean_line in (
                (poincare_figure.data[0], poincare_figure.data[2]),
                (poincare_figure.data[1], poincare_figure.data[3]),
            ):
                counts = np.asarray(heatmap.z, dtype=float)
                radial_centers = np.asarray(heatmap.x, dtype=float)
                totals = counts.sum(axis=1)
                valid = totals > 0.0
                expected_mean = (counts[valid] @ radial_centers) / totals[valid]
                actual_mean = np.asarray(mean_line.x, dtype=float)
                np.testing.assert_allclose(actual_mean[valid], expected_mean)
                self.assertTrue(np.isnan(actual_mean[~valid]).all())
                np.testing.assert_allclose(
                    mean_line.y,
                    np.asarray(heatmap.y, dtype=float),
                )
                self.assertFalse(mean_line.connectgaps)
                self.assertEqual(mean_line.line.shape, "linear")
                self.assertEqual(float(mean_line.line.width), 3.5)
            self.assertLessEqual(float(np.max(poincare_figure.data[0].x)), 2.0**0.5 + 1e-6)
            self.assertIn("bin_center", cdf_frame.columns)
            self.assertTrue(any(column.endswith("_cdf") for column in cdf_frame.columns))
            self.assertTrue(any(column.endswith("_magnitude") for column in fft_frame.columns))
            self.assertEqual(
                list(poincare_frame.columns),
                [
                    "theta_deg",
                    "radial_coordinate",
                    "a_intersection_count",
                    "b_intersection_count",
                ],
            )
            self.assertEqual(len(poincare_frame), 360 * 80)

    def test_workbench_single_plot_falls_back_to_one_valid_signal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            mode_file = Path(td) / "joint_angle_mode.csv"
            _write_mode_file(mode_file, groups=["right_hip", "right_knee"])

            signal_b = signals.load_signal_trace(mode_file, "right_knee", "mode_2")
            signal_b["legend_name"] = "B"
            signal_b["series_key"] = "b"

            plot_signal_a, plot_signal_b = dashboard_callbacks._resolve_workbench_plot_signals(None, signal_b)
            figure = dashboard_callbacks._build_workbench_single_plot_figure("time-series", plot_signal_a, plot_signal_b, 12)
            frame = dashboard_callbacks._build_workbench_single_plot_frame("time-series", plot_signal_a, plot_signal_b, 12)

            self.assertIs(plot_signal_a, signal_b)
            self.assertIsNone(plot_signal_b)
            self.assertEqual(len(figure.data), 1)
            self.assertIn("b_signal_value", frame.columns)
            self.assertEqual(dashboard_callbacks._workbench_filename(plot_signal_a, plot_signal_b, "time-series"), "right_knee_time_series.csv")

    def test_refresh_run_dropdown_response_uses_updated_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "outputs"
            run_alpha = (
                root
                / "limb_outputs"
                / "Human_Motion"
                / "Subject_A"
                / "Flat Terrain"
                / "0,0_Walking"
                / "trial_alpha"
                / "combined_alpha"
            )
            run_beta = (
                root
                / "limb_outputs"
                / "Human_Motion"
                / "Subject_A"
                / "Flat Terrain"
                / "0,0_Walking"
                / "trial_beta"
                / "combined_beta"
            )
            _write_mode_file(run_alpha / "limb_angle_mode.csv", groups=["pelvis_to_right_hip"])

            catalog.clear_caches()
            initial_catalog = catalog.scan_results_catalog(root)
            initial_run = catalog.get_first_run_id(initial_catalog)

            _write_mode_file(run_beta / "limb_angle_mode.csv", groups=["right_hip_to_knee"])
            catalog.clear_caches()
            refreshed_catalog = catalog.scan_results_catalog(root)

            options, value = dashboard_callbacks._run_dropdown_response(refreshed_catalog, initial_run)

            self.assertEqual(len(options), 2)
            self.assertEqual(value, initial_run)
            self.assertTrue(any(str(option["value"]).endswith("combined_beta") for option in options))

            terrain_options, terrain_value = dashboard_callbacks._hierarchy_dropdown_response(
                refreshed_catalog,
                "terrain",
                None,
                current_run=initial_run,
                subject="Subject_A",
            )
            self.assertEqual(terrain_options, [{"label": "Flat Terrain", "value": "Flat Terrain"}])
            self.assertEqual(terrain_value, "Flat Terrain")

            filtered_options, filtered_value = dashboard_callbacks._run_dropdown_response(
                refreshed_catalog,
                initial_run,
                subject="Subject_A",
                terrain="Flat Terrain",
                activity="0,0_Walking",
                experiment="trial_alpha",
            )
            self.assertEqual(filtered_options, [{"label": "combined_alpha - limb_angle_mode.csv", "value": initial_run}])
            self.assertEqual(filtered_value, initial_run)

    def test_dashboard_registers_run_dropdown_refresh_callbacks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "outputs"
            run_dir = root / "limb_outputs" / "Subject_A" / "Flat_Walking" / "run_alpha"
            _write_mode_file(run_dir / "limb_angle_mode.csv", groups=["pelvis_to_right_hip"])

            app = create_app(results_dir=root)
            callback_keys = set(app.callback_map.keys())

            self.assertIn("..overview-subject-dropdown.options...overview-subject-dropdown.value..", callback_keys)
            self.assertIn("..overview-terrain-dropdown.options...overview-terrain-dropdown.value..", callback_keys)
            self.assertIn("..overview-activity-dropdown.options...overview-activity-dropdown.value..", callback_keys)
            self.assertIn("..overview-experiment-dropdown.options...overview-experiment-dropdown.value..", callback_keys)
            self.assertIn("..overview-run-dropdown.options...overview-run-dropdown.value..", callback_keys)
            self.assertIn('..{"slot":["MATCH"],"type":"workbench-subject"}.options...{"slot":["MATCH"],"type":"workbench-subject"}.value..', callback_keys)
            self.assertIn('..{"slot":["MATCH"],"type":"workbench-terrain"}.options...{"slot":["MATCH"],"type":"workbench-terrain"}.value..', callback_keys)
            self.assertIn('..{"slot":["MATCH"],"type":"workbench-activity"}.options...{"slot":["MATCH"],"type":"workbench-activity"}.value..', callback_keys)
            self.assertIn('..{"slot":["MATCH"],"type":"workbench-experiment"}.options...{"slot":["MATCH"],"type":"workbench-experiment"}.value..', callback_keys)
            self.assertIn('..{"slot":["MATCH"],"type":"workbench-run"}.options...{"slot":["MATCH"],"type":"workbench-run"}.value..', callback_keys)

    def test_windows_debug_run_options_disable_reloader(self) -> None:
        options = dashboard_app._build_run_options(debug=True, os_name="nt")

        self.assertTrue(options["debug"])
        self.assertFalse(options["use_reloader"])
        self.assertTrue(options["dev_tools_hot_reload"])

    def test_non_windows_debug_run_options_keep_default_reloader_behavior(self) -> None:
        options = dashboard_app._build_run_options(debug=True, os_name="posix")

        self.assertTrue(options["debug"])
        self.assertNotIn("use_reloader", options)
        self.assertNotIn("dev_tools_hot_reload", options)

    def test_dashboard_smoke_loads_current_outputs_tree(self) -> None:
        outputs_root = ROOT / "outputs"
        if not outputs_root.exists():
            self.skipTest("Current outputs tree is not available in this workspace.")

        catalog.clear_caches()
        signals.clear_caches()
        dashboard_catalog = catalog.scan_results_catalog(outputs_root)
        limb_run_id = catalog.get_first_run_id(dashboard_catalog, family="limb")
        joint_run_id = catalog.get_first_run_id(dashboard_catalog, family="joint")
        if limb_run_id is None or joint_run_id is None:
            self.skipTest("Current outputs tree does not contain both dashboard families.")

        app = create_app(results_dir=outputs_root)
        self.assertIsNotNone(app.layout)

        limb_record = catalog.get_run_record(dashboard_catalog, limb_run_id)
        joint_record = catalog.get_run_record(dashboard_catalog, joint_run_id)
        self.assertIsNotNone(limb_record)
        self.assertIsNotNone(joint_record)

        limb_group = limb_record["available_groups"][0]
        joint_group = joint_record["available_groups"][0]
        limb_signal = signals.load_signal_trace(limb_record["mode_file_path"], limb_group, "mode_1")
        joint_signal = signals.load_signal_trace(joint_record["mode_file_path"], joint_group, "mode_1")
        self.assertGreater(limb_signal["values"].shape[0], 0)
        self.assertGreater(joint_signal["values"].shape[0], 0)

    def test_read_only_mode_omits_the_pipeline_tab(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "outputs"
            run_dir = root / "Unlabelled_Joint_Results" / "0_0_Walking" / "run_beta"
            _write_mode_file(run_dir / "joint_angle_mode.csv", groups=["right_hip", "right_knee"])

            catalog.clear_caches()
            raw_catalog.clear_caches()
            app = create_app(results_dir=root, sample_data_dir=root, enable_pipeline_tab=False)
            component_ids = _collect_ids(app.layout)

            # The viewer still works; only the write/execute surface is gone.
            self.assertNotIn("ingest-upload", component_ids)
            self.assertNotIn("run-start-button", component_ids)
            self.assertIn("workbench-plot-kind", component_ids)

    def test_catalog_hierarchy_handles_a_missing_activity_level(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "outputs"
            base = (
                root
                / "rss"
                / "PSO_SO3_Run"
                / "20260806_122309"
                / "Human_Motion"
            )
            # Full depth: subject/terrain/activity/experiment/csv_stem.
            full = base / "Subject_A" / "Flat Terrain" / "0,0_Walking" / "trial_a" / "stem_a"
            # Activity omitted: the trial sits directly under the terrain.
            flat = base / "Subject_B" / "Rough Terrain" / "trial_b" / "stem_b"
            _write_mode_file(full / "limb_angle_mode.csv", groups=["right_hip_to_knee"])
            _write_mode_file(flat / "limb_angle_mode.csv", groups=["right_hip_to_knee"])

            catalog.clear_caches()
            dashboard_catalog = catalog.scan_results_catalog(root)
            by_subject = {record["subject"]: record for record in dashboard_catalog["runs"]}

            self.assertEqual(set(by_subject), {"Subject_A", "Subject_B"})
            self.assertEqual(by_subject["Subject_A"]["terrain"], "Flat Terrain")
            self.assertEqual(by_subject["Subject_A"]["activity"], "0,0_Walking")
            self.assertEqual(by_subject["Subject_A"]["experiment"], "trial_a")
            self.assertEqual(by_subject["Subject_A"]["csv_file"], "stem_a - limb_angle_mode.csv")

            self.assertEqual(by_subject["Subject_B"]["terrain"], "Rough Terrain")
            self.assertEqual(by_subject["Subject_B"]["activity"], "Unspecified Activity")
            self.assertEqual(by_subject["Subject_B"]["experiment"], "trial_b")
            self.assertEqual(by_subject["Subject_B"]["csv_file"], "stem_b - limb_angle_mode.csv")

    def test_plot_kind_options_include_instantaneous_views(self) -> None:
        values = [option["value"] for option in layout.PLOT_KIND_OPTIONS]
        self.assertEqual(len(values), 9)
        self.assertIn("instantaneous-phase", values)
        self.assertIn("instantaneous-frequency", values)

    def test_instantaneous_phase_is_wrapped_unless_unwrap_requested(self) -> None:
        signal = _synthetic_signal(cycles=6.0)

        wrapped = plotting.build_overlay_instantaneous_phase_figure(signal)
        wrapped_values = np.asarray(wrapped.data[0].y, dtype=float)
        self.assertEqual(wrapped_values.size, signal["values"].size)
        self.assertTrue(np.all(wrapped_values > -np.pi - 1e-9))
        self.assertTrue(np.all(wrapped_values <= np.pi + 1e-9))
        self.assertEqual(tuple(wrapped.layout.yaxis.ticktext), ("-π", "-π/2", "0", "π/2", "π"))

        unwrapped = plotting.build_overlay_instantaneous_phase_figure(signal, unwrap=True)
        unwrapped_values = np.asarray(unwrapped.data[0].y, dtype=float)
        self.assertGreater(unwrapped_values.max(), np.pi)
        self.assertEqual(unwrapped.layout.yaxis.title.text, "Unwrapped Phase (rad)")
        self.assertEqual(
            list(plotting.build_instantaneous_phase_frame(signal, unwrap=True).columns),
            ["a_x", "a_instantaneous_phase_unwrapped"],
        )

    def test_instantaneous_frequency_x_axis_drops_the_first_sample(self) -> None:
        signal = _synthetic_signal(cycles=6.0)

        figure = plotting.build_overlay_instantaneous_frequency_figure(signal)
        x_values = np.asarray(figure.data[0].x, dtype=float)
        y_values = np.asarray(figure.data[0].y, dtype=float)

        # np.diff of the unwrapped phase is a backward difference, so each value
        # belongs to the later sample of its pair.
        self.assertEqual(y_values.size, signal["values"].size - 1)
        self.assertEqual(x_values.size, y_values.size)
        self.assertAlmostEqual(float(x_values[0]), float(signal["time"][1]))
        self.assertEqual(figure.layout.yaxis.title.text, "Frequency (Hz)")
        # 6 cycles over 512 samples at 100 Hz -> 6 / 5.12 s
        self.assertAlmostEqual(float(np.median(y_values)), 6.0 / 5.12, places=2)

    def test_instantaneous_frequency_falls_back_to_cycles_per_sample_without_fs(self) -> None:
        signal = _synthetic_signal(cycles=6.0)
        signal["fs"] = None
        signal["time"] = None
        signal["x_label"] = "Sample Index"

        figure = plotting.build_overlay_instantaneous_frequency_figure(signal)

        self.assertEqual(figure.layout.title.text, "Instantaneous Frequency")
        self.assertEqual(figure.layout.yaxis.title.text, "Frequency (cycles/sample)")
        self.assertEqual(len(figure.layout.annotations), 1)
        self.assertIn("not in Hz", figure.layout.annotations[0].text)
        self.assertEqual(
            list(plotting.build_instantaneous_frequency_frame(signal).columns),
            ["a_x", "a_instantaneous_frequency_cycles_per_sample"],
        )
        self.assertAlmostEqual(
            float(np.median(np.asarray(figure.data[0].y, dtype=float))), 6.0 / 512.0, places=3
        )

    def test_instantaneous_views_survive_nan_gaps(self) -> None:
        small_gap = _synthetic_signal(cycles=6.0)
        small_gap["values"] = small_gap["values"].copy()
        small_gap["values"][100:103] = np.nan
        repaired = plotting.build_overlay_instantaneous_phase_figure(small_gap)
        # A short interior gap is interpolated, so no samples are lost.
        self.assertEqual(np.asarray(repaired.data[0].y).size, 512)
        self.assertEqual(repaired.layout.title.text, "Instantaneous Phase")

        long_gap = _synthetic_signal(cycles=6.0)
        long_gap["values"] = long_gap["values"].copy()
        long_gap["values"][100:120] = np.nan
        truncated = plotting.build_overlay_instantaneous_phase_figure(long_gap)
        # Too long to interpolate, so the longest finite run is used rather than
        # compacting the array and splicing across the gap.
        self.assertEqual(np.asarray(truncated.data[0].y).size, 512 - 120)
        self.assertEqual(truncated.layout.title.text, "Instantaneous Phase (392/512 samples)")

        all_nan = _synthetic_signal(cycles=6.0)
        all_nan["values"] = np.full(512, np.nan)
        empty = plotting.build_overlay_instantaneous_phase_figure(all_nan)
        self.assertEqual(len(empty.data), 0)

    def test_normalize_single_frame_columns_keeps_instantaneous_suffixes_distinct(self) -> None:
        signal = _synthetic_signal(cycles=6.0)

        frequency_columns = list(
            dashboard_callbacks._single_signal_export_frame(signal, "instantaneous-frequency", 12).columns
        )
        phase_columns = list(
            dashboard_callbacks._single_signal_export_frame(signal, "instantaneous-phase", 12).columns
        )
        unwrapped_columns = list(
            dashboard_callbacks._single_signal_export_frame(
                signal, "instantaneous-phase", 12, unwrap=True
            ).columns
        )
        fft_columns = list(dashboard_callbacks._single_signal_export_frame(signal, "fft", 12).columns)

        # The suffix chain is a sequential elif: the bare "frequency_hz" FFT-axis
        # rule must not swallow "<key>_instantaneous_frequency_hz".
        self.assertEqual(frequency_columns, ["x_value", "instantaneous_frequency_hz"])
        self.assertEqual(phase_columns, ["x_value", "instantaneous_phase"])
        self.assertEqual(unwrapped_columns, ["x_value", "instantaneous_phase_unwrapped"])
        self.assertEqual(fft_columns, ["frequency_hz", "magnitude"])


class ManualFlipTests(unittest.TestCase):
    """Manual sign flips are gauge-correct, group-level, and reversible."""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.results_dir = Path(self._temp.name) / "outputs"
        self.leaf = self.results_dir / "rss" / "run" / "stamp" / "stem"
        self.group = "left_shoulder_to_elbow"
        _write_mode_file(self.leaf / "limb_angle_mode.csv", groups=[self.group, "right_hip_to_knee"])
        _write_axes_file(self.leaf / "limb_axes_mode.csv", groups=[self.group, "right_hip_to_knee"])
        self.mode_file = self.leaf / "limb_angle_mode.csv"
        self.axes_file = self.leaf / "limb_axes_mode.csv"
        catalog.clear_caches()
        signals.clear_caches()
        self.addCleanup(self._temp.cleanup)

    def _commit(self, trace_name: str) -> flips.FlipOutcome:
        return flips.commit_manual_flip(
            flips.FlipRequest(self.mode_file, self.group, trace_name),
            results_dir=self.results_dir,
        )

    def test_flip_negates_all_three_angles_and_matching_axis_vectors(self) -> None:
        before_angle = pd.read_csv(self.mode_file)
        before_axes = pd.read_csv(self.axes_file)

        outcome = self._commit("mode_2")

        self.assertTrue(outcome.applied, outcome.message)
        after_angle = pd.read_csv(self.mode_file)
        after_axes = pd.read_csv(self.axes_file)
        for mode in (1, 2, 3):
            np.testing.assert_allclose(after_angle[f"{self.group}_mode_{mode}"], -before_angle[f"{self.group}_mode_{mode}"])
            for component in ("x", "y", "z"):
                column = f"{self.group}_mode_{mode}_axis_{component}"
                np.testing.assert_allclose(after_axes[column], -before_axes[column])

    def test_flip_keeps_the_rotation_unchanged(self) -> None:
        def rotation_vector() -> np.ndarray:
            angle = pd.read_csv(self.mode_file)[f"{self.group}_mode_2"].to_numpy(dtype=float)
            axis = pd.read_csv(self.axes_file)[
                [f"{self.group}_mode_2_axis_{component}" for component in "xyz"]
            ].to_numpy(dtype=float)
            return angle[:, None] * axis

        before = rotation_vector()
        self.assertTrue(self._commit("mode_2").applied)

        # Rot(theta, u) == Rot(-theta, -u), so theta * u must be untouched: the
        # flip is a pure sign-convention fix, not a change of rotation.
        np.testing.assert_allclose(rotation_vector(), before, atol=1e-12)

    def test_flip_leaves_original_angle_and_other_groups_untouched(self) -> None:
        before_angle = pd.read_csv(self.mode_file)
        before_axes = pd.read_csv(self.axes_file)

        self.assertTrue(self._commit("mode_2").applied)

        after_angle = pd.read_csv(self.mode_file)
        after_axes = pd.read_csv(self.axes_file)
        for column in (
            f"{self.group}_original_angle",
            "right_hip_to_knee_mode_1",
            "right_hip_to_knee_mode_2",
            "right_hip_to_knee_mode_3",
        ):
            np.testing.assert_allclose(after_angle[column], before_angle[column], rtol=1e-12)
        for column in ("right_hip_to_knee_mode_1_axis_x", "right_hip_to_knee_mode_3_axis_z"):
            np.testing.assert_allclose(after_axes[column], before_axes[column], rtol=1e-12)

    def test_original_angle_is_never_changed_by_a_group_commit(self) -> None:
        original = pd.read_csv(self.mode_file)[f"{self.group}_original_angle"].copy()
        self.assertTrue(self._commit("mode_1").applied)
        np.testing.assert_allclose(pd.read_csv(self.mode_file)[f"{self.group}_original_angle"], original)

    def test_flip_refuses_when_the_axes_file_is_missing(self) -> None:
        self.axes_file.unlink()
        original_bytes = self.mode_file.read_bytes()

        outcome = self._commit("mode_1")

        self.assertFalse(outcome.applied)
        self.assertIn("axes CSV is missing", outcome.message)
        self.assertEqual(self.mode_file.read_bytes(), original_bytes)

    def test_flip_refuses_a_protected_reference_source(self) -> None:
        from gait_analysis.references import empty_registry, put_reference, reference_entry, save_registry

        entry = reference_entry(
            terrain="flat", direction="level", footwear="unknown", side="left",
            source_path=self.mode_file, mode1_templates={self.group: np.sin(np.linspace(0, 2 * np.pi, 101))},
        )
        save_registry(put_reference(empty_registry(), entry), dashboard_references.registry_path(self.results_dir))
        outcome = self._commit("mode_1")
        self.assertFalse(outcome.applied)
        self.assertIn("protected reference", outcome.message)

    def test_flip_refuses_backup_targets(self) -> None:
        backup = self.results_dir / "hma_dashboard_flip_backup" / "nested"
        _write_mode_file(backup / "limb_angle_mode.csv", groups=[self.group])
        _write_axes_file(backup / "limb_axes_mode.csv", groups=[self.group])
        outcome = flips.commit_manual_flip(
            flips.FlipRequest(backup / "limb_angle_mode.csv", self.group), results_dir=self.results_dir,
        )
        self.assertFalse(outcome.applied)
        self.assertIn("backup path", outcome.message)

    def test_flip_refuses_when_a_sibling_export_carries_mode_columns(self) -> None:
        # A future row-aligned export gaining sign-bearing mode columns must
        # block the commit rather than be left half-flipped.
        sibling = pd.DataFrame({"frame": [0, 1], f"{self.group}_mode_2_w": [1.0, 1.0]})
        sibling.to_csv(self.leaf / "limb_quaternion.csv", index=False)
        original_bytes = self.mode_file.read_bytes()

        outcome = self._commit("mode_2")

        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.unhandled, {"limb_quaternion.csv": [f"{self.group}_mode_2_w"]})
        self.assertEqual(self.mode_file.read_bytes(), original_bytes)

    def test_flip_backs_up_only_the_two_rewritten_files_once(self) -> None:
        first = self._commit("mode_1")
        backup_dir = Path(first.backup_dir)
        self.assertEqual(
            sorted(path.name for path in backup_dir.iterdir()),
            [".operations", "limb_angle_mode.csv", "limb_axes_mode.csv"],
        )
        backed_up = (backup_dir / "limb_angle_mode.csv").read_bytes()

        self.assertTrue(self._commit("mode_1").applied)

        # Copy-once: the backup still holds the pre-first-flip content.
        self.assertEqual((backup_dir / "limb_angle_mode.csv").read_bytes(), backed_up)

    def test_two_flips_return_the_original_values(self) -> None:
        before = pd.read_csv(self.mode_file)[f"{self.group}_mode_3"].to_numpy(dtype=float)

        self.assertTrue(self._commit("mode_3").applied)
        self.assertEqual(flips.net_flip_parity(self.mode_file, self.group, "mode_3"), -1)
        self.assertTrue(self._commit("mode_3").applied)

        after = pd.read_csv(self.mode_file)[f"{self.group}_mode_3"].to_numpy(dtype=float)
        np.testing.assert_allclose(after, before)
        self.assertEqual(flips.net_flip_parity(self.mode_file, self.group, "mode_3"), 1)
        self.assertEqual(flips.count_manual_flips(self.mode_file, self.group, "mode_3"), 2)

    def test_flip_appends_an_audit_log(self) -> None:
        self.assertTrue(self._commit("mode_1").applied)

        log = flips.read_flip_log(self.mode_file)

        self.assertEqual(list(log.columns), list(flips.LOG_COLUMNS))
        self.assertEqual(len(log), 3)
        self.assertEqual(log["trace_name"].tolist(), ["mode_1", "mode_2", "mode_3"])
        self.assertEqual(log.iloc[0]["angle_column"], f"{self.group}_mode_1")
        self.assertEqual(
            log.iloc[0]["axis_columns"],
            " ".join(f"{self.group}_mode_1_axis_{component}" for component in "xyz"),
        )
        datetime.fromisoformat(str(log.iloc[0]["timestamp_utc"]))

    def test_flip_outcome_operation_id_matches_the_leaf_audit_log(self) -> None:
        outcome = self._commit("mode_1")

        self.assertTrue(outcome.applied, outcome.message)
        self.assertEqual(set(flips.read_flip_log(self.mode_file)["operation_id"]), {outcome.operation_id})

    def test_flip_rolls_back_both_csvs_and_log_on_replace_failure(self) -> None:
        before_mode, before_axes = self.mode_file.read_bytes(), self.axes_file.read_bytes()
        calls = {"count": 0}
        real_replace = flips.os.replace

        def fail_second(source, target):
            calls["count"] += 1
            if calls["count"] == 2:
                raise OSError("injected")
            return real_replace(source, target)

        with mock.patch.object(flips, "_replace", side_effect=fail_second):
            outcome = self._commit("mode_1")
        self.assertFalse(outcome.applied)
        self.assertEqual(self.mode_file.read_bytes(), before_mode)
        self.assertEqual(self.axes_file.read_bytes(), before_axes)
        self.assertFalse((self.leaf / flips.FLIP_LOG_NAME).exists())

    def test_flip_invalidates_the_signal_cache(self) -> None:
        # signals._load_group_data_cached has no mtime in its key, so without an
        # explicit clear the plots would serve pre-write values indefinitely.
        before = signals.load_signal_trace(self.mode_file, self.group, "mode_1")["values"].copy()

        self.assertTrue(self._commit("mode_1").applied)

        after = signals.load_signal_trace(self.mode_file, self.group, "mode_1")["values"]
        np.testing.assert_allclose(after, -before)

    def test_preview_flip_does_not_mutate_the_cached_array(self) -> None:
        catalog.clear_caches()
        dashboard_catalog = catalog.scan_results_catalog(self.results_dir)
        run_id = dashboard_catalog["runs"][0]["run_id"]
        config = {"results_dir": str(self.results_dir), "default_fs": 100.0}
        key = flips.flip_key(self.mode_file, self.group, "mode_1")
        group_data = signals.load_group_data(self.mode_file, self.group)
        cached_before = group_data["traces"]["mode_1"].copy()

        plain = dashboard_callbacks._load_selected_signal(
            dashboard_catalog, config, run_id, self.group, "mode_1", slot_label="A"
        )
        flipped = dashboard_callbacks._load_selected_signal(
            dashboard_catalog, config, run_id, self.group, "mode_1", slot_label="A", flip_state={key: True}
        )

        np.testing.assert_allclose(flipped["values"], -plain["values"])
        self.assertTrue(flipped["preview_flipped"])
        self.assertFalse(plain["preview_flipped"])
        self.assertIn("flipped preview", flipped["legend_name"])
        # signal["values"] is a view onto the lru_cached array, so the negation
        # must allocate rather than write in place.
        np.testing.assert_array_equal(group_data["traces"]["mode_1"], cached_before)

    def test_flip_scope_resolution_covers_both_tabs(self) -> None:
        catalog.clear_caches()
        dashboard_catalog = catalog.scan_results_catalog(self.results_dir)
        run_id = dashboard_catalog["runs"][0]["run_id"]
        workbench = {"a": (run_id, self.group, "mode_1"), "b": (run_id, self.group, "original_angle")}

        resolved = {
            scope: dashboard_callbacks._flip_selection(
                scope, dashboard_catalog, workbench=workbench, overview=(run_id, self.group)
            )
            for scope in (*layout.FLIP_SCOPES, "bogus")
        }

        self.assertEqual(resolved["workbench-a"][2], "mode_1")
        self.assertEqual(resolved["workbench-b"][2], "original_angle")
        self.assertEqual(resolved["overview-1"][2], "mode_1")
        self.assertEqual(resolved["overview-3"][2], "mode_3")
        self.assertIsNone(resolved["bogus"])
        # Every scope resolves to the same file, so one flip key is shared.
        self.assertEqual(resolved["workbench-a"], resolved["overview-1"])


class SignalLogTests(unittest.TestCase):
    """Faulty-signal reports and the flip history are separate append-only logs."""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.results_dir = Path(self._temp.name) / "outputs"
        self.leaf = self.results_dir / "rss" / "run" / "stamp" / "stem"
        self.group = "left_shoulder_to_elbow"
        _write_mode_file(self.leaf / "limb_angle_mode.csv", groups=[self.group])
        _write_axes_file(self.leaf / "limb_axes_mode.csv", groups=[self.group])
        self.mode_file = self.leaf / "limb_angle_mode.csv"
        catalog.clear_caches()
        signals.clear_caches()
        self.catalog = catalog.scan_results_catalog(self.results_dir)
        self.record = self.catalog["runs"][0]
        self.config = {"results_dir": str(self.results_dir), "default_fs": 100.0}

    def _report(self, reason="flat_or_dropout", note="", **overrides):
        arguments = dict(
            record=self.record, mode_file_path=self.mode_file, group_name=self.group, trace_name="mode_1",
            reason=reason, note=note,
        )
        arguments.update(overrides)
        return signal_logs.record_fault_report(self.results_dir, **arguments)

    def test_report_appends_one_row_identifying_the_signal(self) -> None:
        outcome = self._report(note="steps 3-5\nlook flat", previewed_flip=True, committed_flips=2, view_scope="overview-1")

        self.assertTrue(outcome.ok, outcome.message)
        log = signal_logs.read_fault_reports(self.results_dir)
        self.assertEqual(list(log.columns), list(signal_logs.FAULT_COLUMNS))
        row = log.iloc[0]
        self.assertEqual(row["report_id"], outcome.report_id)
        self.assertEqual(row["mode_file"], "rss/run/stamp/stem/limb_angle_mode.csv")
        self.assertEqual((row["group_name"], row["trace_name"]), (self.group, "mode_1"))
        self.assertEqual((row["subject"], row["csv_file"]), (self.record["subject"], self.record["csv_file"]))
        self.assertEqual((row["reason_code"], row["reason"]), ("flat_or_dropout", "Flat line / sensor dropout"))
        self.assertEqual(row["note"], "steps 3-5 look flat")
        self.assertEqual((row["previewed_flip"], row["committed_flips"], row["view_scope"]), ("true", "2", "overview-1"))
        datetime.fromisoformat(row["timestamp_utc"])

    def test_reports_append_under_a_single_header_and_are_counted_per_group(self) -> None:
        self.assertEqual(self._report().report_count, 1)
        self.assertEqual(self._report("noise_or_spikes").report_count, 2)

        text = signal_logs.fault_log_path(self.results_dir).read_text(encoding="utf-8")
        self.assertEqual(text.count("timestamp_utc"), 1)
        self.assertEqual(len(signal_logs.read_fault_reports(self.results_dir)), 2)
        self.assertEqual(signal_logs.count_fault_reports(self.results_dir, self.mode_file, self.group), 2)
        self.assertEqual(signal_logs.count_fault_reports(self.results_dir, self.mode_file, "right_hip_to_knee"), 0)

    def test_refused_reports_write_nothing(self) -> None:
        refusals = (
            self._report(reason=None),
            self._report(reason="not_a_reason"),
            self._report(reason="other", note="   "),
            self._report(note="x" * (signal_logs.NOTE_MAX_CHARS + 1)),
        )

        self.assertTrue(all(not outcome.ok for outcome in refusals))
        self.assertFalse(signal_logs.fault_log_path(self.results_dir).exists())

    def test_report_refuses_to_append_under_a_foreign_header(self) -> None:
        path = signal_logs.fault_log_path(self.results_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("some,other,columns\n1,2,3\n", encoding="utf-8")

        outcome = self._report()

        self.assertFalse(outcome.ok)
        self.assertIn("unexpected header", outcome.message)
        self.assertEqual(path.read_text(encoding="utf-8"), "some,other,columns\n1,2,3\n")

    def test_flip_history_is_a_separate_file_from_the_fault_log(self) -> None:
        self._report()
        events = [
            signal_logs.FlipEvent("preview_on", str(self.mode_file), self.group, "mode_1", self.record, "workbench-a", detail="Invert sign button"),
            signal_logs.FlipEvent("commit", str(self.mode_file), self.group, "mode_1 mode_2 mode_3", self.record, "overview-1", operation_id="abc"),
        ]

        self.assertIsNone(signal_logs.log_flip_events(self.results_dir, events))

        self.assertNotEqual(signal_logs.fault_log_path(self.results_dir), signal_logs.flip_history_path(self.results_dir))
        history = signal_logs.read_flip_history(self.results_dir)
        self.assertEqual(list(history.columns), list(signal_logs.FLIP_HISTORY_COLUMNS))
        self.assertEqual(history["event"].tolist(), ["preview_on", "commit"])
        self.assertEqual(history["operation_id"].tolist(), ["", "abc"])
        self.assertEqual(history["mode_file"].unique().tolist(), ["rss/run/stamp/stem/limb_angle_mode.csv"])
        self.assertEqual(len(signal_logs.read_fault_reports(self.results_dir)), 1)

    def test_flip_history_returns_an_error_instead_of_raising(self) -> None:
        bad = signal_logs.FlipEvent("sideways", str(self.mode_file), self.group)
        self.assertIn("sideways", signal_logs.log_flip_events(self.results_dir, [bad]))
        self.assertFalse(signal_logs.flip_history_path(self.results_dir).exists())

        # A path that cannot be a file (its parent is a regular file) must not raise.
        blocker = Path(self._temp.name) / "blocker"
        blocker.write_text("x", encoding="utf-8")
        good = signal_logs.FlipEvent("preview_on", str(self.mode_file), self.group)
        self.assertIsInstance(signal_logs.log_flip_events(blocker / "outputs", [good]), str)
        self.assertIsNone(signal_logs.log_flip_events(self.results_dir, []))

    def test_dashboard_files_stay_out_of_the_run_catalog(self) -> None:
        self._report()
        signal_logs.log_flip_events(
            self.results_dir, [signal_logs.FlipEvent("preview_on", str(self.mode_file), self.group)]
        )

        catalog.clear_caches()
        rescanned = catalog.scan_results_catalog(self.results_dir)

        self.assertEqual([run["run_id"] for run in rescanned["runs"]], [self.record["run_id"]])

    def test_file_fault_report_records_preview_and_committed_flip_state(self) -> None:
        target = {"scope": "workbench-a", "mode_file_path": str(self.mode_file), "group_name": self.group, "trace_name": "mode_1"}
        key = flips.flip_key(self.mode_file, self.group)
        self.assertTrue(flips.commit_manual_flip(
            flips.FlipRequest(self.mode_file, self.group), results_dir=self.results_dir,
        ).applied)
        catalog.clear_caches()
        current = catalog.scan_results_catalog(self.results_dir)

        ok, message = dashboard_callbacks._file_fault_report(target, "drift_or_offset", "", {key: True}, current, self.config)
        self.assertTrue(ok, message)
        # original_angle is never changed by a preview, so it must not claim one.
        ok, message = dashboard_callbacks._file_fault_report(
            {**target, "trace_name": "original_angle"}, "drift_or_offset", "", {key: True}, current, self.config
        )
        self.assertTrue(ok, message)

        log = signal_logs.read_fault_reports(self.results_dir)
        self.assertEqual(log["previewed_flip"].tolist(), ["true", "false"])
        self.assertEqual(log["committed_flips"].tolist(), ["1", "1"])

    def test_file_fault_report_refuses_without_a_target_or_catalog_entry(self) -> None:
        target = {"scope": "workbench-a", "mode_file_path": str(self.mode_file), "group_name": self.group, "trace_name": "mode_1"}

        self.assertFalse(dashboard_callbacks._file_fault_report(None, "other", "x", {}, self.catalog, self.config)[0])
        ok, message = dashboard_callbacks._file_fault_report(target, "other", "x", {}, {"runs": [], "run_map": {}}, self.config)
        self.assertFalse(ok)
        self.assertIn("no longer in the catalog", message)
        ok, message = dashboard_callbacks._file_fault_report(target, None, "", {}, self.catalog, self.config)
        self.assertFalse(ok)
        self.assertFalse(signal_logs.fault_log_path(self.results_dir).exists())

    def test_every_flip_scope_has_a_report_button_and_one_shared_dialog(self) -> None:
        for scope in layout.FLIP_SCOPES:
            ids = [getattr(component, "id", None) for component in layout._build_flip_controls(scope)._traverse()]
            self.assertIn({"type": "fault-report-button", "scope": scope}, ids)

        app = create_app(results_dir=self.results_dir)
        ids = [getattr(component, "id", None) for component in app.layout._traverse()]
        for dialog_id in ("fault-report-backdrop", "fault-report-reason", "fault-report-note",
                          "fault-report-submit", "fault-report-cancel", "fault-report-status"):
            self.assertIn(dialog_id, ids)
        callback_keys = " ".join(app.callback_map)
        self.assertIn("fault-report-target-store.data", callback_keys)
        self.assertIn("fault-report-error.children@", callback_keys)


class SegmentTeMatrixTests(unittest.TestCase):
    """The all-segment TE matrix is directed, jointly masked, and mtime-keyed."""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.mode_file = Path(self._temp.name) / "limb_angle_mode.csv"
        self.groups = ["left_hip_to_knee", "left_knee_to_ankle", "left_ankle_to_foot"]
        _write_te_mode_file(self.mode_file, groups=self.groups)
        catalog.clear_caches()
        signals.clear_caches()
        te_matrix.clear_caches()
        self.addCleanup(self._temp.cleanup)

    def test_matrix_is_square_with_a_nan_diagonal(self) -> None:
        result = te_matrix.compute_all_segment_te_matrix(self.mode_file)

        self.assertTrue(result["available"], result["message"])
        size = len(result["segments"])
        self.assertEqual(size, 3)
        self.assertEqual(len(result["matrix"]), size)
        for index, row in enumerate(result["matrix"]):
            self.assertEqual(len(row), size)
            # The diagonal is deliberately absent, and None keeps it JSON-safe.
            self.assertIsNone(row[index])
            for column, value in enumerate(row):
                if column != index:
                    self.assertIsNotNone(value)
                    self.assertTrue(math.isfinite(float(value)))

    def test_segments_are_ordered_head_to_foot_then_alphabetically(self) -> None:
        ordered = te_matrix.resolve_matrix_segments(
            ["zz_custom_group", "left_ankle_to_foot", "left_knee_to_ankle", "aa_other"]
        )

        # Head-to-foot order first (knee before ankle before foot), then anything
        # it does not know about, so joint-family files still work.
        self.assertEqual(
            ordered, ("left_knee_to_ankle", "left_ankle_to_foot", "aa_other", "zz_custom_group")
        )

    def test_head_to_foot_order_groups_upper_before_lower_with_pairs_adjacent(self) -> None:
        from hilbert_analysis.columns import ALL_SEGMENTS

        ordered = te_matrix.resolve_matrix_segments(reversed(ALL_SEGMENTS))

        self.assertEqual(sorted(ordered), sorted(ALL_SEGMENTS))
        self.assertEqual(len(ordered), 18)
        regions = [te_matrix.body_region(segment) for segment in ordered]
        self.assertEqual(regions, ["upper"] * 9 + ["lower"] * 9)
        self.assertEqual(ordered[0], "neck_to_left_shoulder")
        self.assertEqual(ordered[-1], "right_ankle_to_foot")
        # Left then right at every level, so mirrored segments sit side by side.
        for left, right in zip(ordered[3:9:2], ordered[4:9:2]):
            self.assertEqual(left.replace("left_", "right_"), right)
        for left, right in zip(ordered[12::2], ordered[13::2]):
            self.assertEqual(left.replace("left_", "right_"), right)
        # Head to foot within each limb: shoulder -> elbow -> wrist, hip -> knee -> ankle.
        for chain in (
            ("left_shoulder_to_elbow", "left_elbow_to_wrist", "left_wrist_to_hand"),
            ("right_hip_to_knee", "right_knee_to_ankle", "right_ankle_to_foot"),
        ):
            self.assertEqual([ordered.index(segment) for segment in chain], sorted(ordered.index(s) for s in chain))
        self.assertIsNone(te_matrix.body_region("right_hip"))

    def test_dashboard_defaults_are_25_bins_and_lag_50(self) -> None:
        self.assertEqual((te_matrix.DEFAULT_BINS, te_matrix.DEFAULT_LAG), (25, 50))

        result = te_matrix.compute_all_segment_te_matrix(self.mode_file)

        self.assertTrue(result["available"], result["message"])
        self.assertEqual(result["bins"], 25)
        self.assertEqual(result["source_context"], {"lag": 50, "length": 1, "stride": 1})
        self.assertEqual(result["receiver_context"], {"lag": 50, "length": 1, "stride": 1})

        with tempfile.TemporaryDirectory() as td:
            app = create_app(results_dir=Path(td) / "outputs")
            values = {
                getattr(component, "id", None): getattr(component, "value", None)
                for component in app.layout._traverse()
                if getattr(component, "id", None) in ("overview-te-bins", "overview-te-lag")
            }
        self.assertEqual(values, {"overview-te-bins": 25, "overview-te-lag": 50})

    def test_matrix_uses_one_joint_finite_mask(self) -> None:
        frame = pd.read_csv(self.mode_file)
        frame.loc[0:9, f"{self.groups[0]}_mode_1"] = np.nan
        frame.to_csv(self.mode_file, index=False)
        signals.clear_caches()
        te_matrix.clear_caches()

        result = te_matrix.compute_all_segment_te_matrix(self.mode_file)

        # NaNs in one segment drop those rows for every cell, so each estimate
        # sees an identical sample set and the cells stay comparable.
        self.assertTrue(result["available"], result["message"])
        self.assertEqual(result["rows_total"], len(frame))
        self.assertEqual(result["rows_finite"], len(frame) - 10)
        self.assertEqual(result["n_samples_used"], result["rows_finite"] - te_matrix.DEFAULT_LAG)

    def test_matrix_cache_keys_on_file_mtime(self) -> None:
        first = te_matrix.compute_all_segment_te_matrix(self.mode_file)
        first_value = first["matrix"][0][1]

        # Shuffled, not negated: transfer entropy is invariant under a sign
        # flip, so negation would not prove the cache had noticed the rewrite.
        frame = pd.read_csv(self.mode_file)
        rng = np.random.default_rng(0)
        column = f"{self.groups[0]}_mode_1"
        frame[column] = rng.permutation(frame[column].to_numpy())
        frame.to_csv(self.mode_file, index=False)
        signals.clear_caches()

        second = te_matrix.compute_all_segment_te_matrix(self.mode_file)

        self.assertNotAlmostEqual(first_value, second["matrix"][0][1], places=6)

    def test_matrix_is_invariant_under_a_sign_flip(self) -> None:
        before = te_matrix.compute_all_segment_te_matrix(self.mode_file)["matrix"]

        frame = pd.read_csv(self.mode_file)
        column = f"{self.groups[0]}_mode_1"
        frame[column] = -frame[column]
        frame.to_csv(self.mode_file, index=False)
        signals.clear_caches()
        te_matrix.clear_caches()

        after = te_matrix.compute_all_segment_te_matrix(self.mode_file)["matrix"]

        # This is why the TE matrix needs no flip awareness, only mtime-based
        # invalidation after a commit.
        np.testing.assert_allclose(
            np.array(before, dtype=float), np.array(after, dtype=float), rtol=1e-9
        )

    def test_matrix_requires_at_least_two_segments(self) -> None:
        single = Path(self._temp.name) / "single" / "limb_angle_mode.csv"
        _write_te_mode_file(single, groups=["left_hip_to_knee"])
        catalog.clear_caches()

        result = te_matrix.compute_all_segment_te_matrix(single)

        self.assertFalse(result["available"])
        self.assertIn("At least two signal groups", result["message"])

    def test_matrix_export_frame_is_tidy(self) -> None:
        result = te_matrix.compute_all_segment_te_matrix(self.mode_file)

        frame = plotting.build_all_segment_te_matrix_frame(result)

        size = len(result["segments"])
        self.assertEqual(len(frame), size * size)
        self.assertEqual(int(frame["te_bits"].isna().sum()), size)
        for column in ("source_segment", "source_label", "target_segment", "target_label", "te_bits"):
            self.assertIn(column, frame.columns)

    def test_matrix_figure_reads_like_a_table(self) -> None:
        result = te_matrix.compute_all_segment_te_matrix(self.mode_file)

        figure = plotting.build_all_segment_te_matrix_figure(result)

        self.assertEqual(len(figure.data), 1)
        self.assertEqual(figure.layout.yaxis.autorange, "reversed")
        self.assertEqual(figure.layout.xaxis.tickangle, -45)
        self.assertFalse(figure.data[0].hoverongaps)
        self.assertIsNone(figure.data[0].texttemplate)
        self.assertEqual(list(figure.data[0].x), list(figure.data[0].y))
        self.assertIn("Left Hip → Knee", list(figure.data[0].x))

        with_values = plotting.build_all_segment_te_matrix_figure(result, show_values=True)
        self.assertEqual(with_values.data[0].texttemplate, "%{z:.2f}")

    def _boxed_figure(self, groups: list[str]):
        path = Path(self._temp.name) / "boxes" / "limb_angle_mode.csv"
        _write_te_mode_file(path, groups=groups)
        catalog.clear_caches()
        signals.clear_caches()
        te_matrix.clear_caches()
        result = te_matrix.compute_all_segment_te_matrix(path)
        self.assertTrue(result["available"], result["message"])
        return result, plotting.build_all_segment_te_matrix_figure(result)

    def test_full_matrix_is_boxed_into_four_region_sections(self) -> None:
        from hilbert_analysis.columns import ALL_SEGMENTS

        result, figure = self._boxed_figure(list(ALL_SEGMENTS))

        self.assertEqual(
            [te_matrix.body_region(segment) for segment in result["segments"]], ["upper"] * 9 + ["lower"] * 9
        )
        boxes = {(shape.name, shape.x0, shape.x1, shape.y0, shape.y1) for shape in figure.layout.shapes}
        # Rows are sources, columns are targets; 9 upper then 9 lower, so the
        # boundary sits between index 8 and 9.
        self.assertEqual(
            boxes,
            {
                ("upper_to_upper", -0.5, 8.5, -0.5, 8.5),
                ("upper_to_lower", 8.5, 17.5, -0.5, 8.5),
                ("lower_to_upper", -0.5, 8.5, 8.5, 17.5),
                ("lower_to_lower", 8.5, 17.5, 8.5, 17.5),
            },
        )
        self.assertEqual(figure.layout.xaxis.type, "category")
        self.assertEqual(figure.layout.yaxis.type, "category")
        labels = sorted(annotation.text for annotation in figure.layout.annotations)
        self.assertEqual(
            labels,
            ["Source: Lower limbs", "Source: Upper limbs", "Target: Lower limbs", "Target: Upper limbs"],
        )
        # Head at the top of the (reversed) y axis, foot at the bottom.
        self.assertEqual(figure.data[0].y[0], "Neck → Left Shoulder")
        self.assertEqual(figure.data[0].y[-1], "Right Ankle → Foot")

    def test_region_boxes_follow_uneven_region_sizes(self) -> None:
        _, figure = self._boxed_figure(
            ["left_shoulder_to_elbow", "left_hip_to_knee", "left_knee_to_ankle", "left_ankle_to_foot"]
        )

        boxes = {shape.name: (shape.x0, shape.x1, shape.y0, shape.y1) for shape in figure.layout.shapes}

        self.assertEqual(boxes["upper_to_upper"], (-0.5, 0.5, -0.5, 0.5))
        self.assertEqual(boxes["lower_to_lower"], (0.5, 3.5, 0.5, 3.5))
        self.assertEqual(boxes["upper_to_lower"], (0.5, 3.5, -0.5, 0.5))
        self.assertEqual(boxes["lower_to_upper"], (-0.5, 0.5, 0.5, 3.5))

    def test_no_boxes_without_both_regions_or_for_unclassified_segments(self) -> None:
        for groups in (
            ["left_hip_to_knee", "left_knee_to_ankle", "left_ankle_to_foot"],  # lower only
            ["left_shoulder_to_elbow", "left_elbow_to_wrist"],  # upper only
            ["right_hip", "right_knee", "right_ankle"],  # joint family, no region
        ):
            with self.subTest(groups=groups):
                _, figure = self._boxed_figure(groups)
                self.assertEqual(len(figure.layout.shapes), 0)
                self.assertEqual(len(figure.layout.annotations), 0)

    def test_unavailable_matrix_renders_a_message_figure(self) -> None:
        result = te_matrix.compute_all_segment_te_matrix(Path(self._temp.name) / "missing.csv")

        figure = plotting.build_all_segment_te_matrix_figure(result)

        self.assertFalse(result["available"])
        self.assertEqual(len(figure.data), 0)
        self.assertEqual(figure.layout.title.text, "Segment TE Matrix")


class RawCatalogTests(unittest.TestCase):
    """Raw-tree discovery, path safety, and atomic ingest."""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.sample_data = self.root / "Sample_Data"
        self.results = self.root / "outputs"
        (self.sample_data / "Human_Motion").mkdir(parents=True)
        self.results.mkdir()
        catalog.clear_caches()
        raw_catalog.clear_caches()
        self.addCleanup(self._temp.cleanup)

    def test_sanitize_relative_subpath_rejects_traversal_and_absolute_paths(self) -> None:
        for bad in ("..", "a/../b", "/etc", "C:/x", "a\x00b", "a<b", "...", ".hidden", "trail."):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    raw_catalog.sanitize_relative_subpath(bad)
        with self.assertRaises(ValueError):
            raw_catalog.sanitize_relative_subpath("a/b/c/d/e/f/g")
        with self.assertRaises(ValueError):
            raw_catalog.sanitize_relative_subpath("x" * 200)

        self.assertEqual(raw_catalog.sanitize_relative_subpath(""), ())
        self.assertEqual(
            raw_catalog.sanitize_relative_subpath("NewSubject/Flat Terrain/0,0_Walking/Trial01"),
            ("NewSubject", "Flat Terrain", "0,0_Walking", "Trial01"),
        )
        # Backslashes are separators too, not literal name characters.
        self.assertEqual(raw_catalog.sanitize_relative_subpath("A\\B"), ("A", "B"))

    def test_resolve_destination_stays_inside_sample_data(self) -> None:
        destination = raw_catalog.resolve_destination(
            self.sample_data,
            parent_relative="Human_Motion",
            new_subpath="Subject/Trial",
            filename="combined_log_wide_1.csv",
            known_folders=["Human_Motion"],
        )
        self.assertTrue(destination.is_relative_to(self.sample_data))

        with self.assertRaises(ValueError):
            # The dropdown value is never trusted on its own.
            raw_catalog.resolve_destination(
                self.sample_data,
                parent_relative="Human_Motion/Nope",
                new_subpath="",
                filename="a.csv",
                known_folders=["Human_Motion"],
            )
        with self.assertRaises(ValueError):
            raw_catalog.resolve_destination(
                self.sample_data, parent_relative="", new_subpath="", filename="a.txt"
            )

    def test_resolve_destination_rejects_a_symlink_escape(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        (self.sample_data / "link").symlink_to(outside)
        raw_catalog.clear_caches()

        with self.assertRaises(ValueError):
            # resolve() follows the link, so the containment check still catches it.
            raw_catalog.resolve_destination(
                self.sample_data,
                parent_relative="link",
                new_subpath="",
                filename="a.csv",
                known_folders=["link"],
            )

    def test_folders_include_directories_without_csv_files(self) -> None:
        (self.sample_data / "Human_Motion" / "EmptySubject").mkdir()
        raw_catalog.clear_caches()

        scan = raw_catalog.scan_raw_catalog(self.sample_data, self.results)

        # A freshly created trial folder must be selectable as a destination
        # before it holds anything.
        self.assertIn("Human_Motion/EmptySubject", scan["folders"])

    def test_write_raw_csv_refuses_an_existing_file_without_replace(self) -> None:
        destination = self.sample_data / "Human_Motion" / "a.csv"
        payload = _raw_imu_bytes()

        first = raw_catalog.write_raw_csv(
            payload, destination, sample_data_dir=self.sample_data, replace=False
        )
        second = raw_catalog.write_raw_csv(
            payload, destination, sample_data_dir=self.sample_data, replace=False
        )
        third = raw_catalog.write_raw_csv(
            payload, destination, sample_data_dir=self.sample_data, replace=True
        )

        self.assertTrue(first.ok, first.message)
        self.assertEqual(len(first.validation.usable_limbs), 18)
        self.assertFalse(second.ok)
        self.assertIn("already exists", second.message)
        self.assertTrue(third.ok)
        self.assertTrue(third.replaced)

    def test_write_raw_csv_rejects_a_non_imu_payload_and_leaves_nothing_behind(self) -> None:
        destination = self.sample_data / "Human_Motion" / "junk.csv"

        result = raw_catalog.write_raw_csv(
            b"a,b\n1,2\n", destination, sample_data_dir=self.sample_data, replace=False
        )

        self.assertFalse(result.ok)
        self.assertFalse(destination.exists())
        # Validation happens on a temp file in the destination directory, so a
        # rejected upload must not leave a partial artifact either.
        self.assertEqual(list(destination.parent.glob("*.part-*")), [])

    def test_decode_upload_contents_enforces_the_size_limit(self) -> None:
        payload = _raw_imu_bytes()
        contents = "data:text/csv;base64," + base64.b64encode(payload).decode()

        self.assertEqual(raw_catalog.decode_upload_contents(contents), payload)
        with self.assertRaises(ValueError):
            raw_catalog.decode_upload_contents(contents, max_bytes=16)
        with self.assertRaises(ValueError):
            raw_catalog.decode_upload_contents("not-a-data-url")
        with self.assertRaises(ValueError):
            raw_catalog.decode_upload_contents("data:text/csv,plain")

    def test_processed_flag_inverts_the_output_path_resolver(self) -> None:
        raw_csv = (
            self.sample_data
            / "Human_Motion"
            / "Subject"
            / "Terrain"
            / "Activity"
            / "Trial"
            / "combined_log_wide_1.csv"
        )
        raw_csv.parent.mkdir(parents=True)
        raw_csv.write_bytes(_raw_imu_bytes())
        other = self.sample_data / "Human_Motion" / "Subject" / "combined_log_wide_2.csv"
        other.write_bytes(_raw_imu_bytes())

        run_root = self.results / "rss" / "Run" / "stamp"
        leaf = Path(resolve_realdata_output_dir(str(run_root), str(raw_csv)))
        _write_mode_file(leaf / "limb_angle_mode.csv", groups=["left_hip_to_knee"])
        catalog.clear_caches()
        raw_catalog.clear_caches()

        scan = raw_catalog.scan_raw_catalog(self.sample_data, self.results)
        by_id = {record["raw_id"]: record for record in scan["files"]}

        processed = by_id["Human_Motion/Subject/Terrain/Activity/Trial/combined_log_wide_1.csv"]
        unprocessed = by_id["Human_Motion/Subject/combined_log_wide_2.csv"]
        self.assertTrue(processed["processed"])
        self.assertFalse(unprocessed["processed"])
        self.assertEqual(scan["processed_count"], 1)
        self.assertEqual(scan["unprocessed_count"], 1)

        # The key really is the resolver's own suffix, for every file.
        for record in scan["files"]:
            resolved = Path(resolve_realdata_output_dir("/root", record["abs_path"]))
            self.assertEqual(
                record["output_key"], "/".join(resolved.relative_to("/root").parts)
            )

    def test_raw_file_options_filter_by_dataset_and_processed_state(self) -> None:
        for name in ("a", "b"):
            path = self.sample_data / "Human_Motion" / f"combined_log_wide_{name}.csv"
            path.write_bytes(_raw_imu_bytes())
        raw_catalog.clear_caches()
        scan = raw_catalog.scan_raw_catalog(self.sample_data, self.results)

        self.assertEqual(len(raw_catalog.get_raw_file_options(scan, unprocessed_only=True)), 2)
        self.assertEqual(
            len(raw_catalog.get_raw_file_options(scan, dataset="Weight_Lifting")), 0
        )
        self.assertIsNotNone(
            raw_catalog.get_raw_file_record_by_path(
                scan, self.sample_data / "Human_Motion" / "combined_log_wide_a.csv"
            )
        )


class PipelineJobTests(unittest.TestCase):
    """Job command construction, queueing, and progress parsing.

    Nothing here spawns an RSS process: ``_launch`` is stubbed, so the suite
    stays fast and never writes into a real output tree.
    """

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.results = self.root / "outputs"
        self.jobs_dir = jobs.jobs_dir_for(self.results)
        self.csv = self.root / "Sample_Data" / "Human_Motion" / "combined_log_wide_1.csv"
        self.csv.parent.mkdir(parents=True)
        self.csv.write_bytes(_raw_imu_bytes())
        self.addCleanup(self._temp.cleanup)

    def _request(self, **overrides):
        kwargs = dict(
            csv_path=self.csv,
            output_root=self.results / "rss" / "Run" / "stamp",
            results_dir=self.results,
            project_root=self.root,
            epochs=25,
            ssa_window_length=150,
            compute_device="cpu",
        )
        kwargs.update(overrides)
        return jobs.build_job_request(**kwargs)

    def test_preset_argv_matches_the_reference_invocation(self) -> None:
        argv = jobs.build_preset_argv(self._request(), config_path="/tmp/config.json")

        self.assertEqual(argv[1:5], ["-u", "-m", "rss", "run-config"])
        # --quiet would suppress the very progress lines the poller parses.
        self.assertNotIn("--quiet", argv)
        self.assertNotIn("--compute-device", argv)
        self.assertEqual(argv[argv.index("--file-workers") + 1], "1")
        self.assertEqual(argv[argv.index("--epochs") + 1], "25")
        self.assertEqual(argv[argv.index("--ssa-window-length") + 1], "150")
        for flag in jobs.PRESET_FIXED_ARGS:
            self.assertIn(flag, argv)

    def test_preset_argv_parses_with_the_real_rss_parser(self) -> None:
        from rss.cli import build_parser
        from rss.realdata_config import build_realdata_runtime_config_from_args

        argv = jobs.build_preset_argv(self._request(), config_path="/tmp/config.json")
        namespace = build_parser().parse_args(argv[4:])
        runtime = build_realdata_runtime_config_from_args(namespace)

        self.assertEqual(namespace.command, "run-config")
        self.assertEqual(namespace.epochs, 25)
        self.assertEqual(namespace.ssa_window_length, 150)
        self.assertEqual(namespace.rss_optimizer, "sgd")
        self.assertEqual(namespace.rss_matrix_parameterization, "so3")
        # nargs="+" options must not swallow the following flag, and the nine
        # negative weights must read as values rather than options.
        self.assertEqual(namespace.rss_lags, [0])
        self.assertEqual(len(namespace.rss_initial_weight_matrix), 9)
        self.assertEqual(runtime.rss_config.epochs, 25)

    def test_config_payload_round_trips_through_run_config(self) -> None:
        from rss.realdata_config import load_file_run_configs

        payload = jobs.build_config_payload(self._request())
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        loaded = load_file_run_configs(str(config_path))

        self.assertEqual(len(payload), 1)
        self.assertEqual(set(payload[0]), {"csv_path", "compute_device"})
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].compute_device, "cpu")

    def test_parameter_clamping_and_device_validation(self) -> None:
        self.assertEqual(jobs.clamp_epochs(0), jobs.EPOCHS_RANGE[0])
        self.assertEqual(jobs.clamp_epochs(10**9), jobs.EPOCHS_RANGE[1])
        self.assertEqual(jobs.clamp_epochs("nonsense"), jobs.DEFAULT_EPOCHS)
        self.assertEqual(jobs.clamp_ssa_window(1), jobs.SSA_WINDOW_RANGE[0])
        for good in ("cpu", "auto", "cuda", "cuda:0"):
            self.assertEqual(jobs.normalize_compute_device(good), good)
        for bad in ("cuda:x", "gpu", None, ""):
            self.assertEqual(jobs.normalize_compute_device(bad), jobs.DEFAULT_COMPUTE_DEVICE)

    def test_progress_parsing_matches_the_documented_log_format(self) -> None:
        lines = [
            "[RSS      0.0s] run_start: 1 file(s)",
            "[RSS      0.1s] [1/1] file_start: /x/a.csv",
            "[RSS     35.2s] [1/1] left_ankle_to_foot (3/18)",
        ]

        parsed = jobs._parse_progress(lines)

        self.assertEqual(parsed["limb_name"], "left_ankle_to_foot")
        self.assertEqual((parsed["limb_index"], parsed["limb_total"]), (3, 18))
        # The limb line prints before that limb runs, so two are finished.
        self.assertAlmostEqual(parsed["fraction"], 2 / 18)
        self.assertAlmostEqual(parsed["elapsed_seconds"], 35.2)
        self.assertFalse(parsed["done"])

        finished = jobs._parse_progress(
            lines + ["[RSS    531.5s] [1/1] file_done: /x/a.csv", "[RSS    531.6s] run_done: 1 file(s)"]
        )
        self.assertEqual(finished["fraction"], 1.0)
        self.assertTrue(finished["done"])

    def test_progress_survives_the_trailing_json_summary(self) -> None:
        log = self.root / "run.log"
        # A real run ends with a multi-line JSON summary on stdout, which would
        # push every progress line out of a plain last-N-lines view.
        log.write_text(
            "\n".join(
                [
                    "[RSS      0.1s] [1/1] file_start: /x/a.csv",
                    "[RSS     35.2s] [1/1] left_ankle_to_foot (3/18)",
                    "[RSS    531.6s] run_done: 1 file(s)",
                    *["{" + f'"line": {index}' + "}" for index in range(200)],
                ]
            ),
            encoding="utf-8",
        )
        record = self._stub_record(status="succeeded", log_path=log)

        progress = jobs.read_progress(record)

        self.assertEqual(progress.fraction, 1.0)
        self.assertIn("run_done", progress.log_tail)

    def _stub_record(self, **overrides) -> jobs.JobRecord:
        payload = dict(
            job_id="job-1",
            status="running",
            message="",
            csv_path=str(self.csv),
            csv_label="a.csv",
            output_root=str(self.results),
            output_leaf=str(self.results / "leaf"),
            epochs=25,
            ssa_window_length=150,
            compute_device="cpu",
            preset_id=jobs.PRESET_ID,
            argv=[],
            cwd=str(self.root),
            job_dir=str(self.jobs_dir / "job-1"),
            log_path=str(self.root / "run.log"),
            config_path=str(self.root / "config.json"),
            created_at=0.0,
        )
        payload.update({key: str(value) for key, value in overrides.items() if key == "log_path"})
        payload.update({key: value for key, value in overrides.items() if key != "log_path"})
        return jobs.JobRecord(**payload)

    def test_submit_queues_beyond_the_concurrency_cap(self) -> None:
        launched: list[str] = []

        def fake_launch(record: jobs.JobRecord) -> jobs.JobRecord:
            record.status = "running"
            record.pid = 10_000 + len(launched)
            launched.append(record.job_id)
            jobs._write_record(record)
            return record

        with mock.patch.object(jobs, "_launch", side_effect=fake_launch), mock.patch.object(
            jobs, "_is_pid_alive", return_value=True
        ):
            records = [jobs.submit_job(self._request()) for _ in range(3)]
            statuses = [record.status for record in records]
            self.assertEqual(statuses, ["running", "running", "queued"])

            # Once a slot frees up, the queued job is promoted by pump().
            with mock.patch.object(jobs, "_is_pid_alive", return_value=False):
                jobs.pump(self.jobs_dir)
        self.assertEqual(len(launched), 3)

    def test_cancelling_a_queued_job_never_launches_it(self) -> None:
        with mock.patch.object(jobs, "_launch", side_effect=AssertionError("must not launch")):
            with mock.patch.object(jobs, "_running_count", return_value=jobs.MAX_CONCURRENT_JOBS):
                record = jobs.submit_job(self._request())
            self.assertEqual(record.status, "queued")

            cancelled = jobs.request_cancel(self.jobs_dir, record.job_id)

        self.assertEqual(cancelled.status, "cancelled")

    def test_orphaned_job_status_comes_from_the_log_markers(self) -> None:
        for marker, expected in (("run_done: 1 file(s)", "succeeded"), ("still going", "failed")):
            with self.subTest(marker=marker):
                log = self.root / f"orphan_{expected}.log"
                log.write_text(f"[RSS 1.0s] {marker}\n", encoding="utf-8")
                record = self._stub_record(job_id=f"orphan-{expected}", log_path=log, pid=-1)
                Path(record.job_dir).mkdir(parents=True, exist_ok=True)
                jobs._write_record(record)

                with mock.patch.object(jobs, "_is_pid_alive", return_value=False):
                    reaped = jobs._reap(record)

                self.assertEqual(reaped.status, expected)

    def test_active_job_lookup_prevents_double_submission(self) -> None:
        with mock.patch.object(jobs, "_launch", side_effect=lambda record: record):
            jobs.submit_job(self._request())

        found = jobs.find_active_job_for(
            self.jobs_dir, self.csv, self.results / "rss" / "Run" / "stamp"
        )
        missing = jobs.find_active_job_for(self.jobs_dir, self.csv, self.results / "elsewhere")

        self.assertIsNotNone(found)
        self.assertIsNone(missing)

    def test_run_params_are_stamped_into_the_output_leaf(self) -> None:
        with mock.patch.object(jobs, "_launch", side_effect=lambda record: record):
            record = jobs.submit_job(self._request(epochs=25))

        params = json.loads((Path(record.output_leaf) / jobs.RUN_PARAMS_NAME).read_text())

        # Without this, a 25-epoch dashboard run would be indistinguishable from
        # a 2000-epoch batch result sitting in the same tree.
        self.assertEqual(params["epochs"], 25)
        self.assertEqual(params["ssa_window_length"], 150)
        self.assertEqual(params["source"], "HMA_Dashboard")


def _raw_imu_bytes(samples: int = 64) -> bytes:
    """A minimal but complete raw IMU CSV, covering all 18 mapped limbs."""
    from rss.sensor_layout import SEGMENT_SENSOR_MAPPING

    rows: dict[str, np.ndarray] = {"Timestamp": np.arange(1, samples + 1, dtype=int)}
    phase = np.linspace(0.0, 4.0 * np.pi, samples)
    for index, device in enumerate(sorted(set(SEGMENT_SENSOR_MAPPING.values()))):
        for axis_index, axis in enumerate(("AX", "AY", "AZ", "GX", "GY", "GZ")):
            rows[f"{device}_{axis}"] = np.sin(phase + 0.1 * index + 0.05 * axis_index)
    frame = pd.DataFrame(rows)
    return frame.to_csv(index=False).encode("utf-8")


def _write_te_mode_file(path: Path, *, groups: list[str], samples: int = 320) -> None:
    """Mode file with a lagged driver chain, so TE is non-zero and directed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    driver = np.sin(np.linspace(0.0, 24.0 * np.pi, samples)) + 0.05 * rng.standard_normal(samples)
    rows: dict[str, np.ndarray] = {
        "frame": np.arange(samples, dtype=int),
        "time": np.arange(samples, dtype=float) / 100.0,
    }
    for index, group in enumerate(groups):
        shifted = np.roll(driver, 3 * index) + 0.1 * rng.standard_normal(samples)
        rows[f"{group}_mode_1"] = shifted
        rows[f"{group}_mode_2"] = 0.5 * shifted
        rows[f"{group}_mode_3"] = -0.25 * shifted
        rows[f"{group}_original_angle"] = np.abs(shifted)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_axes_file(path: Path, *, groups: list[str]) -> None:
    """Row-aligned axes CSV matching ``_write_mode_file``'s shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = 48
    rows: dict[str, np.ndarray] = {
        "frame": np.arange(samples, dtype=int),
        "time": np.arange(samples, dtype=float) / 100.0,
    }
    for index, group in enumerate(groups):
        for mode in (1, 2, 3):
            offset = 0.1 * (index + mode)
            rows[f"{group}_mode_{mode}_axis_x"] = np.full(samples, 0.6 + offset)
            rows[f"{group}_mode_{mode}_axis_y"] = np.full(samples, -0.8 + offset)
            rows[f"{group}_mode_{mode}_axis_z"] = np.linspace(0.1, 0.9, samples) + offset
    pd.DataFrame(rows).to_csv(path, index=False)


def _synthetic_signal(*, cycles: float, samples: int = 512, fs: float = 100.0) -> dict:
    """A plotting-ready signal dict for a clean sinusoid."""
    index = np.arange(samples, dtype=float)
    return {
        "values": np.sin(2.0 * np.pi * cycles * index / samples),
        "time": index / fs,
        "sample_index": index,
        "fs": fs,
        "x_label": "Time (s)",
        "legend_name": "A",
        "series_key": "a",
    }


def _collect_ids(component) -> set[str]:
    ids: set[str] = set()
    if component is None:
        return ids
    component_id = getattr(component, "id", None)
    if isinstance(component_id, str):
        ids.add(component_id)
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            ids.update(_collect_ids(child))
    elif children is not None:
        ids.update(_collect_ids(children))
    return ids


if __name__ == "__main__":
    unittest.main()
