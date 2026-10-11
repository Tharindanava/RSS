"""Dash callbacks for the HMA Dashboard."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dash import ALL, Input, MATCH, Output, State, ctx, dcc, html, no_update

from HMA_Dashboard import catalog
from HMA_Dashboard import flips
from HMA_Dashboard import jobs
from HMA_Dashboard import layout
from HMA_Dashboard import plotting
from HMA_Dashboard import raw_catalog
from HMA_Dashboard import references as dashboard_references
from HMA_Dashboard import signal_logs
from HMA_Dashboard import signals
from HMA_Dashboard import te_matrix
from HMA_Dashboard import transfer_entropy


def register_callbacks(app) -> None:
    """Register all dashboard callbacks."""

    @app.callback(
        Output("catalog-store", "data"),
        Output("catalog-status", "children"),
        Input("refresh-button", "n_clicks"),
        State("dashboard-config-store", "data"),
        prevent_initial_call=True,
    )
    def refresh_catalog(_n_clicks: int, config: dict[str, Any]) -> tuple[dict[str, Any], str]:
        catalog.clear_caches()
        signals.clear_caches()
        transfer_entropy.clear_caches()
        te_matrix.clear_caches()
        refreshed_catalog = catalog.scan_results_catalog(config["results_dir"])
        return refreshed_catalog, catalog.get_catalog_summary(refreshed_catalog)

    @app.callback(
        Output({"type": "workbench-subject", "slot": MATCH}, "options"),
        Output({"type": "workbench-subject", "slot": MATCH}, "value"),
        Input("catalog-store", "data"),
        State({"type": "workbench-subject", "slot": MATCH}, "value"),
        State({"type": "workbench-run", "slot": MATCH}, "value"),
    )
    def update_workbench_subject_options(
        catalog_state: dict[str, Any],
        current_subject: str | None,
        current_run: str | None,
    ) -> tuple[list[dict[str, str]], str | None]:
        return _hierarchy_dropdown_response(
            catalog_state,
            "subject",
            current_subject,
            current_run=current_run,
        )

    @app.callback(
        Output({"type": "workbench-terrain", "slot": MATCH}, "options"),
        Output({"type": "workbench-terrain", "slot": MATCH}, "value"),
        Input({"type": "workbench-subject", "slot": MATCH}, "value"),
        Input("catalog-store", "data"),
        State({"type": "workbench-terrain", "slot": MATCH}, "value"),
        State({"type": "workbench-run", "slot": MATCH}, "value"),
    )
    def update_workbench_terrain_options(
        subject: str | None,
        catalog_state: dict[str, Any],
        current_terrain: str | None,
        current_run: str | None,
    ) -> tuple[list[dict[str, str]], str | None]:
        return _hierarchy_dropdown_response(
            catalog_state,
            "terrain",
            current_terrain,
            current_run=current_run,
            subject=subject,
        )

    @app.callback(
        Output({"type": "workbench-activity", "slot": MATCH}, "options"),
        Output({"type": "workbench-activity", "slot": MATCH}, "value"),
        Input({"type": "workbench-subject", "slot": MATCH}, "value"),
        Input({"type": "workbench-terrain", "slot": MATCH}, "value"),
        Input("catalog-store", "data"),
        State({"type": "workbench-activity", "slot": MATCH}, "value"),
        State({"type": "workbench-run", "slot": MATCH}, "value"),
    )
    def update_workbench_activity_options(
        subject: str | None,
        terrain: str | None,
        catalog_state: dict[str, Any],
        current_activity: str | None,
        current_run: str | None,
    ) -> tuple[list[dict[str, str]], str | None]:
        return _hierarchy_dropdown_response(
            catalog_state,
            "activity",
            current_activity,
            current_run=current_run,
            subject=subject,
            terrain=terrain,
        )

    @app.callback(
        Output({"type": "workbench-experiment", "slot": MATCH}, "options"),
        Output({"type": "workbench-experiment", "slot": MATCH}, "value"),
        Input({"type": "workbench-subject", "slot": MATCH}, "value"),
        Input({"type": "workbench-terrain", "slot": MATCH}, "value"),
        Input({"type": "workbench-activity", "slot": MATCH}, "value"),
        Input("catalog-store", "data"),
        State({"type": "workbench-experiment", "slot": MATCH}, "value"),
        State({"type": "workbench-run", "slot": MATCH}, "value"),
    )
    def update_workbench_experiment_options(
        subject: str | None,
        terrain: str | None,
        activity: str | None,
        catalog_state: dict[str, Any],
        current_experiment: str | None,
        current_run: str | None,
    ) -> tuple[list[dict[str, str]], str | None]:
        return _hierarchy_dropdown_response(
            catalog_state,
            "experiment",
            current_experiment,
            current_run=current_run,
            subject=subject,
            terrain=terrain,
            activity=activity,
        )

    @app.callback(
        Output({"type": "workbench-run", "slot": MATCH}, "options"),
        Output({"type": "workbench-run", "slot": MATCH}, "value"),
        Input({"type": "workbench-subject", "slot": MATCH}, "value"),
        Input({"type": "workbench-terrain", "slot": MATCH}, "value"),
        Input({"type": "workbench-activity", "slot": MATCH}, "value"),
        Input({"type": "workbench-experiment", "slot": MATCH}, "value"),
        Input("catalog-store", "data"),
        State({"type": "workbench-run", "slot": MATCH}, "value"),
    )
    def update_workbench_run_options(
        subject: str | None,
        terrain: str | None,
        activity: str | None,
        experiment: str | None,
        catalog_state: dict[str, Any],
        current_run: str | None,
    ) -> tuple[list[dict[str, str]], str | None]:
        return _run_dropdown_response(
            catalog_state,
            current_run,
            subject=subject,
            terrain=terrain,
            activity=activity,
            experiment=experiment,
        )

    @app.callback(
        Output({"type": "workbench-group", "slot": MATCH}, "options"),
        Output({"type": "workbench-group", "slot": MATCH}, "value"),
        Input({"type": "workbench-run", "slot": MATCH}, "value"),
        Input("catalog-store", "data"),
        State({"type": "workbench-group", "slot": MATCH}, "value"),
        State({"type": "workbench-group", "slot": MATCH}, "id"),
    )
    def update_workbench_group_options(
        run_id: str | None,
        catalog_state: dict[str, Any],
        current_group: str | None,
        dropdown_id: dict[str, str],
    ) -> tuple[list[dict[str, str]], str | None]:
        options = catalog.get_group_options(catalog_state, run_id)
        default_value = catalog.get_default_group(catalog_state, run_id, slot=dropdown_id["slot"])
        return options, _select_valid_dropdown_value(options, current_group, default_value)

    @app.callback(
        Output({"type": "workbench-trace", "slot": MATCH}, "options"),
        Output({"type": "workbench-trace", "slot": MATCH}, "value"),
        Input({"type": "workbench-run", "slot": MATCH}, "value"),
        Input({"type": "workbench-group", "slot": MATCH}, "value"),
        Input("catalog-store", "data"),
        State({"type": "workbench-trace", "slot": MATCH}, "value"),
        State({"type": "workbench-trace", "slot": MATCH}, "id"),
    )
    def update_workbench_trace_options(
        run_id: str | None,
        group_name: str | None,
        catalog_state: dict[str, Any],
        current_trace: str | None,
        dropdown_id: dict[str, str],
    ) -> tuple[list[dict[str, str]], str | None]:
        options = catalog.get_trace_options(catalog_state, run_id, group_name)
        default_value = catalog.get_default_trace(catalog_state, run_id, group_name, slot=dropdown_id["slot"])
        return options, _select_valid_dropdown_value(options, current_trace, default_value)

    @app.callback(
        Output({"type": "workbench-selection-metadata", "slot": MATCH}, "children"),
        Input({"type": "workbench-run", "slot": MATCH}, "value"),
        Input({"type": "workbench-group", "slot": MATCH}, "value"),
        Input({"type": "workbench-trace", "slot": MATCH}, "value"),
        Input("catalog-store", "data"),
        State({"type": "workbench-selection-metadata", "slot": MATCH}, "id"),
    )
    def update_workbench_metadata(
        run_id: str | None,
        group_name: str | None,
        trace_name: str | None,
        catalog_state: dict[str, Any],
        metadata_id: dict[str, str],
    ) -> str:
        slot_label = metadata_id["slot"].upper()
        record = catalog.get_run_record(catalog_state, run_id)
        if record is None:
            return f"Signal {slot_label}: select a run."
        group_label = record["group_labels"].get(group_name or "", "No signal group selected")
        trace_label = catalog.pretty_trace_label(trace_name) if trace_name else "No trace selected"
        summary = f"Signal {slot_label}: {record['label']} | {group_label} | {trace_label}"
        return f"{summary} · {_run_file_stats(record)}"

    @app.callback(
        Output("overview-subject-dropdown", "options"),
        Output("overview-subject-dropdown", "value"),
        Input("catalog-store", "data"),
        State("overview-subject-dropdown", "value"),
        State("overview-run-dropdown", "value"),
    )
    def update_overview_subject_options(
        catalog_state: dict[str, Any],
        current_subject: str | None,
        current_run: str | None,
    ) -> tuple[list[dict[str, str]], str | None]:
        return _hierarchy_dropdown_response(
            catalog_state,
            "subject",
            current_subject,
            current_run=current_run,
        )

    @app.callback(
        Output("overview-terrain-dropdown", "options"),
        Output("overview-terrain-dropdown", "value"),
        Input("overview-subject-dropdown", "value"),
        Input("catalog-store", "data"),
        State("overview-terrain-dropdown", "value"),
        State("overview-run-dropdown", "value"),
    )
    def update_overview_terrain_options(
        subject: str | None,
        catalog_state: dict[str, Any],
        current_terrain: str | None,
        current_run: str | None,
    ) -> tuple[list[dict[str, str]], str | None]:
        return _hierarchy_dropdown_response(
            catalog_state,
            "terrain",
            current_terrain,
            current_run=current_run,
            subject=subject,
        )

    @app.callback(
        Output("overview-activity-dropdown", "options"),
        Output("overview-activity-dropdown", "value"),
        Input("overview-subject-dropdown", "value"),
        Input("overview-terrain-dropdown", "value"),
        Input("catalog-store", "data"),
        State("overview-activity-dropdown", "value"),
        State("overview-run-dropdown", "value"),
    )
    def update_overview_activity_options(
        subject: str | None,
        terrain: str | None,
        catalog_state: dict[str, Any],
        current_activity: str | None,
        current_run: str | None,
    ) -> tuple[list[dict[str, str]], str | None]:
        return _hierarchy_dropdown_response(
            catalog_state,
            "activity",
            current_activity,
            current_run=current_run,
            subject=subject,
            terrain=terrain,
        )

    @app.callback(
        Output("overview-experiment-dropdown", "options"),
        Output("overview-experiment-dropdown", "value"),
        Input("overview-subject-dropdown", "value"),
        Input("overview-terrain-dropdown", "value"),
        Input("overview-activity-dropdown", "value"),
        Input("catalog-store", "data"),
        State("overview-experiment-dropdown", "value"),
        State("overview-run-dropdown", "value"),
    )
    def update_overview_experiment_options(
        subject: str | None,
        terrain: str | None,
        activity: str | None,
        catalog_state: dict[str, Any],
        current_experiment: str | None,
        current_run: str | None,
    ) -> tuple[list[dict[str, str]], str | None]:
        return _hierarchy_dropdown_response(
            catalog_state,
            "experiment",
            current_experiment,
            current_run=current_run,
            subject=subject,
            terrain=terrain,
            activity=activity,
        )

    @app.callback(
        Output("overview-run-dropdown", "options"),
        Output("overview-run-dropdown", "value"),
        Input("overview-subject-dropdown", "value"),
        Input("overview-terrain-dropdown", "value"),
        Input("overview-activity-dropdown", "value"),
        Input("overview-experiment-dropdown", "value"),
        Input("catalog-store", "data"),
        State("overview-run-dropdown", "value"),
    )
    def update_overview_run_options(
        subject: str | None,
        terrain: str | None,
        activity: str | None,
        experiment: str | None,
        catalog_state: dict[str, Any],
        current_run: str | None,
    ) -> tuple[list[dict[str, str]], str | None]:
        return _run_dropdown_response(
            catalog_state,
            current_run,
            subject=subject,
            terrain=terrain,
            activity=activity,
            experiment=experiment,
        )

    @app.callback(
        Output("overview-group-dropdown", "options"),
        Output("overview-group-dropdown", "value"),
        Input("overview-run-dropdown", "value"),
        Input("catalog-store", "data"),
        State("overview-group-dropdown", "value"),
    )
    def update_overview_group_options(
        run_id: str | None,
        catalog_state: dict[str, Any],
        current_group: str | None,
    ) -> tuple[list[dict[str, str]], str | None]:
        options = catalog.get_group_options(catalog_state, run_id)
        default_value = catalog.get_default_group(catalog_state, run_id)
        return options, _select_valid_dropdown_value(options, current_group, default_value)

    @app.callback(
        Output("overview-selection-metadata", "children"),
        Input("overview-run-dropdown", "value"),
        Input("overview-group-dropdown", "value"),
        Input("catalog-store", "data"),
    )
    def update_overview_metadata(
        run_id: str | None,
        group_name: str | None,
        catalog_state: dict[str, Any],
    ) -> str:
        record = catalog.get_run_record(catalog_state, run_id)
        if record is None:
            return "Overview: select a run."
        group_label = record["group_labels"].get(group_name or "", "No signal group selected")
        return f"Overview: {record['label']} | {group_label} · {_run_file_stats(record)}"

    @app.callback(
        Output("workbench-single-plot", "figure"),
        Output("workbench-joint-pdf", "figure"),
        Output("workbench-pairwise-te-summary", "children"),
        Output("workbench-markers-store", "data"),
        Input({"type": "workbench-run", "slot": "a"}, "value"),
        Input({"type": "workbench-group", "slot": "a"}, "value"),
        Input({"type": "workbench-trace", "slot": "a"}, "value"),
        Input({"type": "workbench-run", "slot": "b"}, "value"),
        Input({"type": "workbench-group", "slot": "b"}, "value"),
        Input({"type": "workbench-trace", "slot": "b"}, "value"),
        Input("workbench-plot-kind", "value"),
        Input("distribution-bins", "value"),
        Input("pairwise-bins", "value"),
        Input("source-lag", "value"),
        Input("source-length", "value"),
        Input("source-stride", "value"),
        Input("receiver-lag", "value"),
        Input("receiver-length", "value"),
        Input("receiver-stride", "value"),
        Input("phase-unwrap-toggle", "value"),
        Input("flip-preview-store", "data"),
        Input("workbench-selection-store", "data"),
        State("catalog-store", "data"),
        State("dashboard-config-store", "data"),
    )
    def update_workbench_outputs(
        run_a: str | None,
        group_a: str | None,
        trace_a: str | None,
        run_b: str | None,
        group_b: str | None,
        trace_b: str | None,
        workbench_plot_kind: str | None,
        distribution_bins: int | float | None,
        pairwise_bins: int | float | None,
        source_lag: int | float | None,
        source_length: int | float | None,
        source_stride: int | float | None,
        receiver_lag: int | float | None,
        receiver_length: int | float | None,
        receiver_stride: int | float | None,
        phase_unwrap: list[str] | None,
        flip_state: dict[str, Any] | None,
        selection: dict[str, Any] | None,
        catalog_state: dict[str, Any],
        config: dict[str, Any],
    ):
        signal_a = _load_selected_signal(
            catalog_state, config, run_a, group_a, trace_a, slot_label="A", flip_state=flip_state
        )
        signal_b = _load_selected_signal(
            catalog_state, config, run_b, group_b, trace_b, slot_label="B", flip_state=flip_state
        )
        plot_signal_a, plot_signal_b = _resolve_workbench_plot_signals(signal_a, signal_b)

        signal_key = f"{run_a}|{group_a}|{trace_a}"
        markers_payload = _build_markers_payload(plot_signal_a, signal_key=signal_key)
        window = _resolve_window(selection, signal_key)

        if plot_signal_a is None:
            message_figure = plotting.make_message_figure("Select two signals to begin.")
            return (
                message_figure,
                plotting.make_message_figure("Select two same-run signals for joint PDF."),
                html.Div("Select two signals to compute transfer entropy.", className="status-text"),
                markers_payload,
            )

        distribution_bins = _coerce_positive_int(distribution_bins, fallback=40, minimum=2)
        pairwise_bins = _coerce_positive_int(pairwise_bins, fallback=8, minimum=2)
        selected_plot_kind = str(workbench_plot_kind or "time-series")
        source_context = transfer_entropy.ContextSpec(
            lag=_coerce_positive_int(source_lag, fallback=1, minimum=1),
            length=_coerce_positive_int(source_length, fallback=1, minimum=1),
            stride=_coerce_positive_int(source_stride, fallback=1, minimum=1),
        )
        receiver_context = transfer_entropy.ContextSpec(
            lag=_coerce_positive_int(receiver_lag, fallback=1, minimum=1),
            length=_coerce_positive_int(receiver_length, fallback=1, minimum=1),
            stride=_coerce_positive_int(receiver_stride, fallback=1, minimum=1),
        )

        single_plot_figure = _build_workbench_single_plot_figure(
            selected_plot_kind,
            plot_signal_a,
            plot_signal_b,
            distribution_bins,
            unwrap=_is_unwrap_selected(phase_unwrap),
            window=window,
        )

        if signal_a is None or signal_b is None:
            missing_pair_message = "Select two signals to enable the joint PDF and transfer entropy results."
            return (
                single_plot_figure,
                plotting.make_message_figure(missing_pair_message, title="Joint PDF"),
                html.Div(missing_pair_message, className="status-text"),
                markers_payload,
            )

        if not signals.can_pair_signals(signal_a, signal_b):
            gated_message = "Pairwise dependence analyses are only enabled when both signals come from the same run file."
            return (
                single_plot_figure,
                plotting.make_message_figure(gated_message, title="Joint PDF"),
                html.Div(gated_message, className="status-text"),
                markers_payload,
            )

        joint_pdf_figure = plotting.build_joint_pdf_figure(signal_a, signal_b, bins=pairwise_bins)
        te_result = transfer_entropy.compute_pairwise_transfer_entropy(
            signal_a["values"],
            signal_b["values"],
            source_context=source_context,
            receiver_context=receiver_context,
            bins=pairwise_bins,
        )
        te_summary = _build_te_summary_children(te_result)
        return (
            single_plot_figure,
            joint_pdf_figure,
            te_summary,
            markers_payload,
        )

    @app.callback(
        Output("workbench-selection-store", "data"),
        Input("workbench-single-plot", "relayoutData"),
        State("workbench-selection-store", "data"),
        State("workbench-markers-store", "data"),
        prevent_initial_call=True,
    )
    def capture_workbench_selection(
        relayout_data: dict[str, Any] | None,
        current_selection: dict[str, Any] | None,
        markers_data: dict[str, Any] | None,
    ):
        return _snap_selection_from_relayout(relayout_data, current_selection, markers_data)

    @app.callback(
        Output("workbench-te-matrix", "figure"),
        Output("workbench-te-matrix-status", "children"),
        Input({"type": "workbench-run", "slot": "a"}, "value"),
        Input({"type": "workbench-group", "slot": "a"}, "value"),
        Input({"type": "workbench-trace", "slot": "a"}, "value"),
        Input({"type": "workbench-run", "slot": "b"}, "value"),
        Input({"type": "workbench-group", "slot": "b"}, "value"),
        Input({"type": "workbench-trace", "slot": "b"}, "value"),
        State("catalog-store", "data"),
        State("dashboard-config-store", "data"),
    )
    def update_workbench_te_matrix(
        run_a: str | None,
        group_a: str | None,
        trace_a: str | None,
        run_b: str | None,
        group_b: str | None,
        trace_b: str | None,
        catalog_state: dict[str, Any],
        config: dict[str, Any],
    ):
        # Deliberately not flip-aware: the plug-in transfer-entropy estimator is
        # invariant under a sign flip of the source and/or target (up to
        # bin-edge ties), so a preview flip cannot change this matrix.
        signal_a = _load_selected_signal(catalog_state, config, run_a, group_a, trace_a, slot_label="A")
        signal_b = _load_selected_signal(catalog_state, config, run_b, group_b, trace_b, slot_label="B")
        if signal_a is None or signal_b is None:
            message = "Select two signals to enable the lower-limb TE matrix."
            return plotting.make_message_figure(message, title="Lower-Limb TE Matrix"), message
        if not signals.can_pair_signals(signal_a, signal_b):
            message = "Pairwise dependence analyses are only enabled when both signals come from the same run file."
            return plotting.make_message_figure(message, title="Lower-Limb TE Matrix"), message

        matrix_result = transfer_entropy.compute_te_matrix_for_mode_file(signal_a["mode_file_path"])
        return plotting.build_te_matrix_figure(matrix_result), _build_matrix_status(matrix_result)

    @app.callback(
        Output({"type": "workbench-download-target", "kind": MATCH}, "data"),
        Input({"type": "workbench-download-button", "kind": MATCH}, "n_clicks"),
        State({"type": "workbench-download-button", "kind": MATCH}, "id"),
        State({"type": "workbench-run", "slot": "a"}, "value"),
        State({"type": "workbench-group", "slot": "a"}, "value"),
        State({"type": "workbench-trace", "slot": "a"}, "value"),
        State({"type": "workbench-run", "slot": "b"}, "value"),
        State({"type": "workbench-group", "slot": "b"}, "value"),
        State({"type": "workbench-trace", "slot": "b"}, "value"),
        State("workbench-plot-kind", "value"),
        State("distribution-bins", "value"),
        State("pairwise-bins", "value"),
        State("source-lag", "value"),
        State("source-length", "value"),
        State("source-stride", "value"),
        State("receiver-lag", "value"),
        State("receiver-length", "value"),
        State("receiver-stride", "value"),
        State("phase-unwrap-toggle", "value"),
        State("flip-preview-store", "data"),
        State("catalog-store", "data"),
        State("dashboard-config-store", "data"),
        prevent_initial_call=True,
    )
    def download_workbench_export(
        _n_clicks: int,
        button_id: dict[str, str],
        run_a: str | None,
        group_a: str | None,
        trace_a: str | None,
        run_b: str | None,
        group_b: str | None,
        trace_b: str | None,
        workbench_plot_kind: str | None,
        distribution_bins: int | float | None,
        pairwise_bins: int | float | None,
        source_lag: int | float | None,
        source_length: int | float | None,
        source_stride: int | float | None,
        receiver_lag: int | float | None,
        receiver_length: int | float | None,
        receiver_stride: int | float | None,
        phase_unwrap: list[str] | None,
        flip_state: dict[str, Any] | None,
        catalog_state: dict[str, Any],
        config: dict[str, Any],
    ):
        kind = button_id["kind"]
        signal_a = _load_selected_signal(
            catalog_state, config, run_a, group_a, trace_a, slot_label="A", flip_state=flip_state
        )
        signal_b = _load_selected_signal(
            catalog_state, config, run_b, group_b, trace_b, slot_label="B", flip_state=flip_state
        )
        plot_signal_a, plot_signal_b = _resolve_workbench_plot_signals(signal_a, signal_b)
        if kind == "single-plot" and plot_signal_a is None:
            return None
        if kind != "single-plot" and (signal_a is None or signal_b is None):
            return None

        distribution_bins = _coerce_positive_int(distribution_bins, fallback=40, minimum=2)
        pairwise_bins = _coerce_positive_int(pairwise_bins, fallback=8, minimum=2)
        selected_plot_kind = str(workbench_plot_kind or "time-series")
        source_context = transfer_entropy.ContextSpec(
            lag=_coerce_positive_int(source_lag, fallback=1, minimum=1),
            length=_coerce_positive_int(source_length, fallback=1, minimum=1),
            stride=_coerce_positive_int(source_stride, fallback=1, minimum=1),
        )
        receiver_context = transfer_entropy.ContextSpec(
            lag=_coerce_positive_int(receiver_lag, fallback=1, minimum=1),
            length=_coerce_positive_int(receiver_length, fallback=1, minimum=1),
            stride=_coerce_positive_int(receiver_stride, fallback=1, minimum=1),
        )

        if kind == "single-plot":
            data_frame = _build_workbench_single_plot_frame(
                selected_plot_kind,
                plot_signal_a,
                plot_signal_b,
                distribution_bins,
                unwrap=_is_unwrap_selected(phase_unwrap),
            )
            return dcc.send_data_frame(
                data_frame.to_csv,
                _workbench_filename(plot_signal_a, plot_signal_b, selected_plot_kind),
                index=False,
            )

        if not signals.can_pair_signals(signal_a, signal_b):
            return None

        if kind == "joint-pdf":
            data_frame = plotting.build_joint_pdf_frame(signal_a, signal_b, bins=pairwise_bins)
            return dcc.send_data_frame(data_frame.to_csv, _workbench_filename(signal_a, signal_b, kind), index=False)
        if kind == "pairwise-te-csv":
            te_result = transfer_entropy.compute_pairwise_transfer_entropy(
                signal_a["values"],
                signal_b["values"],
                source_context=source_context,
                receiver_context=receiver_context,
                bins=pairwise_bins,
            )
            data_frame = pd.DataFrame([_pairwise_te_record(signal_a, signal_b, te_result)])
            return dcc.send_data_frame(data_frame.to_csv, _workbench_filename(signal_a, signal_b, kind), index=False)
        if kind == "pairwise-te-json":
            te_result = transfer_entropy.compute_pairwise_transfer_entropy(
                signal_a["values"],
                signal_b["values"],
                source_context=source_context,
                receiver_context=receiver_context,
                bins=pairwise_bins,
            )
            payload = {
                "source_signal": _signal_export_metadata(signal_a),
                "target_signal": _signal_export_metadata(signal_b),
                "pairwise_transfer_entropy": _pairwise_te_record(signal_a, signal_b, te_result),
            }
            return dcc.send_string(json.dumps(payload, indent=2, sort_keys=True), _workbench_filename(signal_a, signal_b, kind).replace(".csv", ".json"))
        if kind == "te-matrix":
            matrix_result = transfer_entropy.compute_te_matrix_for_mode_file(signal_a["mode_file_path"])
            data_frame = plotting.build_te_matrix_frame(matrix_result)
            return dcc.send_data_frame(data_frame.to_csv, _workbench_filename(signal_a, signal_b, kind), index=False)
        return None

    @app.callback(
        Output({"type": "overview-mode-graph", "mode": MATCH}, "figure"),
        Output({"type": "overview-markers-store", "mode": MATCH}, "data"),
        Input("overview-run-dropdown", "value"),
        Input("overview-group-dropdown", "value"),
        Input("overview-distribution-bins", "value"),
        Input("overview-phase-unwrap-toggle", "value"),
        Input("flip-preview-store", "data"),
        Input({"type": "overview-plot-kind", "mode": MATCH}, "value"),
        Input({"type": "overview-selection-store", "mode": MATCH}, "data"),
        State({"type": "overview-mode-graph", "mode": MATCH}, "id"),
        State("catalog-store", "data"),
        State("dashboard-config-store", "data"),
    )
    def update_overview_mode_graph(
        run_id: str | None,
        group_name: str | None,
        distribution_bins: int | float | None,
        phase_unwrap: list[str] | None,
        flip_state: dict[str, Any] | None,
        plot_kind: str,
        selection: dict[str, Any] | None,
        graph_id: dict[str, int],
        catalog_state: dict[str, Any],
        config: dict[str, Any],
    ):
        mode = int(graph_id["mode"])
        signal = _load_selected_signal(
            catalog_state,
            config,
            run_id,
            group_name,
            f"mode_{mode}",
            slot_label=f"Mode {mode}",
            flip_state=flip_state,
        )
        signal_key = f"{run_id}|{group_name}|mode_{mode}"
        markers_payload = _build_markers_payload(signal, signal_key=signal_key)
        if signal is None:
            return plotting.make_message_figure("Select a run and signal group."), markers_payload
        bins = _coerce_positive_int(distribution_bins, fallback=40, minimum=2)
        window = _resolve_window(selection, signal_key)
        figure = _build_single_signal_figure(
            plot_kind, signal, bins, unwrap=_is_unwrap_selected(phase_unwrap), window=window
        )
        return figure, markers_payload

    @app.callback(
        Output({"type": "overview-selection-store", "mode": MATCH}, "data"),
        Input({"type": "overview-mode-graph", "mode": MATCH}, "relayoutData"),
        State({"type": "overview-selection-store", "mode": MATCH}, "data"),
        State({"type": "overview-markers-store", "mode": MATCH}, "data"),
        prevent_initial_call=True,
    )
    def capture_overview_selection(
        relayout_data: dict[str, Any] | None,
        current_selection: dict[str, Any] | None,
        markers_data: dict[str, Any] | None,
    ):
        return _snap_selection_from_relayout(relayout_data, current_selection, markers_data)

    @app.callback(
        Output("workbench-selection-store", "data", allow_duplicate=True),
        Output({"type": "overview-selection-store", "mode": ALL}, "data", allow_duplicate=True),
        Input("arrow-key-event", "data"),
        State("workbench-selection-store", "data"),
        State("workbench-markers-store", "data"),
        State({"type": "overview-selection-store", "mode": ALL}, "data"),
        State({"type": "overview-markers-store", "mode": ALL}, "data"),
        prevent_initial_call=True,
    )
    def page_selection_by_arrow_key(
        event: dict[str, Any] | None,
        workbench_selection: dict[str, Any] | None,
        workbench_markers: dict[str, Any] | None,
        overview_selections: list[dict[str, Any] | None],
        overview_markers: list[dict[str, Any] | None],
    ):
        overview_outputs: list[Any] = [no_update] * len(overview_selections)
        if not event:
            return no_update, overview_outputs
        scope = event.get("scope")
        direction = event.get("direction")
        if direction not in ("prev", "next"):
            return no_update, overview_outputs

        if scope == "workbench":
            return _page_selection(workbench_selection, workbench_markers, direction), overview_outputs

        if isinstance(scope, str) and scope.startswith("overview-"):
            try:
                mode = int(scope.split("-", 1)[1])
            except ValueError:
                return no_update, overview_outputs
            # Overview always has exactly modes 1, 2, 3 (layout.py:_build_overview_mode_card
            # is instantiated for (1, 2, 3)), so this index lines up with the ALL-matched lists.
            position = mode - 1
            if 0 <= position < len(overview_selections):
                overview_outputs[position] = _page_selection(
                    overview_selections[position], overview_markers[position], direction
                )
        return no_update, overview_outputs

    @app.callback(
        Output({"type": "overview-download-target", "mode": MATCH}, "data"),
        Input({"type": "overview-download-button", "mode": MATCH}, "n_clicks"),
        State({"type": "overview-download-button", "mode": MATCH}, "id"),
        State("overview-run-dropdown", "value"),
        State("overview-group-dropdown", "value"),
        State("overview-distribution-bins", "value"),
        State("overview-phase-unwrap-toggle", "value"),
        State("flip-preview-store", "data"),
        State({"type": "overview-plot-kind", "mode": MATCH}, "value"),
        State("catalog-store", "data"),
        State("dashboard-config-store", "data"),
        prevent_initial_call=True,
    )
    def download_overview_plot_csv(
        _n_clicks: int,
        button_id: dict[str, int],
        run_id: str | None,
        group_name: str | None,
        distribution_bins: int | float | None,
        phase_unwrap: list[str] | None,
        flip_state: dict[str, Any] | None,
        plot_kind: str,
        catalog_state: dict[str, Any],
        config: dict[str, Any],
    ):
        mode = int(button_id["mode"])
        signal = _load_selected_signal(
            catalog_state,
            config,
            run_id,
            group_name,
            f"mode_{mode}",
            slot_label=f"Mode {mode}",
            flip_state=flip_state,
        )
        if signal is None:
            return None
        bins = _coerce_positive_int(distribution_bins, fallback=40, minimum=2)
        data_frame = _single_signal_export_frame(
            signal, plot_kind, bins, unwrap=_is_unwrap_selected(phase_unwrap)
        )
        filename = f"{_safe_token(signal['run_label'])}_{_safe_token(signal['group_name'])}_mode_{mode}_{_safe_token(plot_kind)}.csv"
        return dcc.send_data_frame(data_frame.to_csv, filename, index=False)

    @app.callback(
        Output({"type": "overview-download-all-target", "mode": MATCH}, "data"),
        Input({"type": "overview-download-all-button", "mode": MATCH}, "n_clicks"),
        State({"type": "overview-download-all-button", "mode": MATCH}, "id"),
        State("overview-run-dropdown", "value"),
        State("overview-distribution-bins", "value"),
        State("overview-phase-unwrap-toggle", "value"),
        State("flip-preview-store", "data"),
        State({"type": "overview-plot-kind", "mode": MATCH}, "value"),
        State("catalog-store", "data"),
        State("dashboard-config-store", "data"),
        prevent_initial_call=True,
    )
    def download_overview_all_groups_csv(
        _n_clicks: int,
        button_id: dict[str, int],
        run_id: str | None,
        distribution_bins: int | float | None,
        phase_unwrap: list[str] | None,
        flip_state: dict[str, Any] | None,
        plot_kind: str,
        catalog_state: dict[str, Any],
        config: dict[str, Any],
    ):
        mode = int(button_id["mode"])
        record = catalog.get_run_record(catalog_state, run_id)
        if record is None:
            return None
        bins = _coerce_positive_int(distribution_bins, fallback=40, minimum=2)
        unwrap = _is_unwrap_selected(phase_unwrap)
        rows: list[pd.DataFrame] = []
        for group_name in record["available_groups"]:
            signal = _load_selected_signal(
                catalog_state,
                config,
                run_id,
                group_name,
                f"mode_{mode}",
                slot_label=f"Mode {mode}",
                flip_state=flip_state,
            )
            if signal is None:
                continue
            frame = _single_signal_export_frame(signal, plot_kind, bins, unwrap=unwrap)
            frame.insert(0, "group_name", signal["group_name"])
            frame.insert(1, "group_label", signal["group_label"])
            rows.append(frame)
        if not rows:
            return None
        data_frame = pd.concat(rows, ignore_index=True)
        filename = f"{_safe_token(record['label'])}_all_groups_mode_{mode}_{_safe_token(plot_kind)}.csv"
        return dcc.send_data_frame(data_frame.to_csv, filename, index=False)

    @app.callback(
        Output("overview-te-matrix", "figure"),
        Output("overview-te-matrix-status", "children"),
        Input("overview-te-matrix-compute", "n_clicks"),
        Input("overview-te-show-values", "value"),
        State("overview-run-dropdown", "value"),
        State("overview-te-bins", "value"),
        State("overview-te-lag", "value"),
        State("catalog-store", "data"),
        State("dashboard-config-store", "data"),
        prevent_initial_call=True,
    )
    def update_overview_te_matrix(
        _n_clicks: int,
        show_values: list[str] | None,
        run_id: str | None,
        bins: int | float | None,
        lag: int | float | None,
        catalog_state: dict[str, Any],
        config: dict[str, Any],
    ):
        record = catalog.get_run_record(catalog_state, run_id)
        if record is None:
            message = "Select a trial, then press Compute TE Matrix."
            return plotting.make_message_figure(message, title="Segment TE Matrix"), message

        context = transfer_entropy.ContextSpec(
            lag=_coerce_positive_int(lag, fallback=te_matrix.DEFAULT_LAG, minimum=1), length=1, stride=1
        )
        result = te_matrix.compute_all_segment_te_matrix(
            record["mode_file_path"],
            bins=_coerce_positive_int(bins, fallback=te_matrix.DEFAULT_BINS, minimum=2),
            source_context=context,
            receiver_context=context,
            default_fs=config.get("default_fs"),
        )
        figure = plotting.build_all_segment_te_matrix_figure(
            result, show_values="show" in (show_values or [])
        )
        return figure, _build_all_segment_matrix_status(result)

    @app.callback(
        Output("overview-te-matrix", "figure", allow_duplicate=True),
        Output("overview-te-matrix-status", "children", allow_duplicate=True),
        Input("overview-run-dropdown", "value"),
        prevent_initial_call=True,
    )
    def reset_overview_te_matrix(_run_id: str | None):
        # A stale 18x18 heatmap from another trial is easy to misread, so the
        # selection change clears it rather than leaving it on screen.
        message = "Selection changed - press Compute TE Matrix."
        return plotting.make_message_figure(message, title="Segment TE Matrix"), message

    @app.callback(
        Output("overview-te-matrix-download", "data"),
        Input("overview-te-matrix-download-button", "n_clicks"),
        State("overview-run-dropdown", "value"),
        State("overview-te-bins", "value"),
        State("overview-te-lag", "value"),
        State("catalog-store", "data"),
        State("dashboard-config-store", "data"),
        prevent_initial_call=True,
    )
    def download_overview_te_matrix(
        _n_clicks: int,
        run_id: str | None,
        bins: int | float | None,
        lag: int | float | None,
        catalog_state: dict[str, Any],
        config: dict[str, Any],
    ):
        record = catalog.get_run_record(catalog_state, run_id)
        if record is None:
            return None
        context = transfer_entropy.ContextSpec(
            lag=_coerce_positive_int(lag, fallback=te_matrix.DEFAULT_LAG, minimum=1), length=1, stride=1
        )
        result = te_matrix.compute_all_segment_te_matrix(
            record["mode_file_path"],
            bins=_coerce_positive_int(bins, fallback=te_matrix.DEFAULT_BINS, minimum=2),
            source_context=context,
            receiver_context=context,
            default_fs=config.get("default_fs"),
        )
        data_frame = plotting.build_all_segment_te_matrix_frame(result)
        if data_frame.empty:
            return None
        filename = f"{_safe_token(record['label'])}_segment_te_matrix.csv"
        return dcc.send_data_frame(data_frame.to_csv, filename, index=False)

    # --- manual sign flips: preview, then commit -------------------------------
    #
    # The buttons are write-only and the status text is derived from the store,
    # rather than a checklist that would need two-way syncing between five
    # scopes and one shared store.

    _FLIP_SELECTION_STATES = (
        State({"type": "workbench-run", "slot": "a"}, "value"),
        State({"type": "workbench-group", "slot": "a"}, "value"),
        State({"type": "workbench-trace", "slot": "a"}, "value"),
        State({"type": "workbench-run", "slot": "b"}, "value"),
        State({"type": "workbench-group", "slot": "b"}, "value"),
        State({"type": "workbench-trace", "slot": "b"}, "value"),
        State("overview-run-dropdown", "value"),
        State("overview-group-dropdown", "value"),
    )

    @app.callback(
        Output("flip-preview-store", "data"),
        Input({"type": "flip-preview-button", "scope": ALL}, "n_clicks"),
        *_FLIP_SELECTION_STATES,
        State("flip-preview-store", "data"),
        State("catalog-store", "data"),
        State("dashboard-config-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_flip_preview(
        _clicks: list[int | None],
        run_a: str | None,
        group_a: str | None,
        trace_a: str | None,
        run_b: str | None,
        group_b: str | None,
        trace_b: str | None,
        overview_run: str | None,
        overview_group: str | None,
        flip_state: dict[str, Any] | None,
        catalog_state: dict[str, Any],
        config: dict[str, Any],
    ):
        scope = _triggered_scope()
        selection = _flip_selection(
            scope,
            catalog_state,
            workbench={"a": (run_a, group_a, trace_a), "b": (run_b, group_b, trace_b)},
            overview=(overview_run, overview_group),
        )
        if selection is None:
            return no_update
        key = flips.flip_key(*selection)
        state = dict(flip_state or {})
        previewed = state.pop(key, None) is None
        if previewed:
            state[key] = True
        mode_file_path, group_name, trace_name = selection
        _log_flip_event(
            config, catalog_state, "preview_on" if previewed else "preview_off",
            mode_file_path, group_name, trace_name, view_scope=scope or "", detail="Invert sign button",
        )
        return state

    @app.callback(
        Output("flip-preview-store", "data", allow_duplicate=True),
        Output("flip-commit-status", "children"),
        Output("catalog-store", "data", allow_duplicate=True),
        Input({"type": "flip-commit-button", "scope": ALL}, "n_clicks"),
        *_FLIP_SELECTION_STATES,
        State("flip-preview-store", "data"),
        State("catalog-store", "data"),
        State("dashboard-config-store", "data"),
        prevent_initial_call=True,
    )
    def commit_flip_preview(
        _clicks: list[int | None],
        run_a: str | None,
        group_a: str | None,
        trace_a: str | None,
        run_b: str | None,
        group_b: str | None,
        trace_b: str | None,
        overview_run: str | None,
        overview_group: str | None,
        flip_state: dict[str, Any] | None,
        catalog_state: dict[str, Any],
        config: dict[str, Any],
    ):
        scope = _triggered_scope()
        selection = _flip_selection(
            scope,
            catalog_state,
            workbench={"a": (run_a, group_a, trace_a), "b": (run_b, group_b, trace_b)},
            overview=(overview_run, overview_group),
        )
        if selection is None:
            return no_update, html.Div("Select a signal first.", className="status-text"), no_update

        mode_file_path, group_name, trace_name = selection
        key = flips.flip_key(mode_file_path, group_name, trace_name)
        state = dict(flip_state or {})
        if not state.get(key):
            # Commit means "make the preview I am looking at permanent", so a
            # stray second click cannot silently flip the data back.
            return (
                no_update,
                html.Div(
                    "No preview flip to save for that selection. Press "
                    "'Invert sign (preview)' first.",
                    className="status-text",
                ),
                no_update,
            )

        outcome = flips.commit_manual_flip(
            flips.FlipRequest(Path(mode_file_path), group_name, trace_name),
            results_dir=config.get("results_dir", "outputs"),
            reference_registry=dashboard_references.registry_path(config.get("results_dir", "outputs")),
        )
        if not outcome.applied:
            return no_update, html.Div(outcome.message, className="status-text"), no_update

        # Drop the preview in the same response: the sign now lives on disk, and
        # leaving the toggle set would negate it a second time.
        state.pop(key, None)
        refreshed = catalog.scan_results_catalog(config.get("results_dir", "outputs"))
        message = outcome.message
        history_error = _log_flip_event(
            config, catalog_state, "commit", mode_file_path, group_name, " ".join(flips.COMMITTABLE_TRACES),
            view_scope=scope or "", operation_id=outcome.operation_id,
            detail="Modes 1-3 and axis vectors saved to the source CSVs",
        )
        if history_error:
            message += f" The flip itself was saved, but the flip history log could not be updated: {history_error}"
        return state, html.Div(message, className="status-text"), refreshed

    @app.callback(
        Output("flip-preview-store", "data", allow_duplicate=True),
        Input("flip-clear-button", "n_clicks"),
        State("flip-preview-store", "data"),
        State("catalog-store", "data"),
        State("dashboard-config-store", "data"),
        prevent_initial_call=True,
    )
    def clear_flip_previews(
        _n_clicks: int, flip_state: dict[str, Any] | None, catalog_state: dict[str, Any], config: dict[str, Any],
    ):
        for key in flip_state or {}:
            mode_file_path, group_name = flips.parse_flip_key(key)
            _log_flip_event(
                config, catalog_state, "preview_off", mode_file_path, group_name, "",
                detail="Clear Preview Flips button",
            )
        return {}

    # --- faulty-signal reports -------------------------------------------------
    #
    # One shared dialog, opened by the flip-row button of whichever of the five
    # scopes was pressed. Reports only append to the report log; they never touch
    # the run outputs, so nothing here needs the commit/rollback machinery.

    @app.callback(
        Output("fault-report-target-store", "data"),
        Output("fault-report-backdrop", "style"),
        Output("fault-report-summary", "children"),
        Output("fault-report-error", "children"),
        Output("fault-report-reason", "value"),
        Output("fault-report-note", "value"),
        Output("fault-report-status", "children"),
        Input({"type": "fault-report-button", "scope": ALL}, "n_clicks"),
        *_FLIP_SELECTION_STATES,
        State("catalog-store", "data"),
        prevent_initial_call=True,
    )
    def open_fault_report(
        _clicks: list[int | None],
        run_a: str | None,
        group_a: str | None,
        trace_a: str | None,
        run_b: str | None,
        group_b: str | None,
        trace_b: str | None,
        overview_run: str | None,
        overview_group: str | None,
        catalog_state: dict[str, Any],
    ):
        # Pattern-matched buttons also "fire" when a tab mounts them, with no
        # click behind it; only a real press may open the dialog.
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            return (no_update,) * 7
        scope = _triggered_scope()
        selection = _flip_selection(
            scope,
            catalog_state,
            workbench={"a": (run_a, group_a, trace_a), "b": (run_b, group_b, trace_b)},
            overview=(overview_run, overview_group),
        )
        record = _record_for_mode_file(catalog_state, selection[0]) if selection else None
        if selection is None or record is None:
            return (no_update,) * 6 + (html.Div("Select a signal before reporting it.", className="status-text"),)
        mode_file_path, group_name, trace_name = selection
        target = {"scope": scope, "mode_file_path": mode_file_path, "group_name": group_name, "trace_name": trace_name}
        group_label = record["group_labels"].get(group_name, group_name)
        summary = html.Div([
            html.Div(record["label"], className="status-line"),
            html.Div(f"{group_label} | {catalog.pretty_trace_label(trace_name)}", className="status-line"),
        ])
        return target, layout.FAULT_PANEL_SHOWN, summary, "", None, "", ""

    @app.callback(
        Output("fault-report-backdrop", "style", allow_duplicate=True),
        Output("fault-report-error", "children", allow_duplicate=True),
        Output("fault-report-status", "children", allow_duplicate=True),
        Input("fault-report-submit", "n_clicks"),
        Input("fault-report-cancel", "n_clicks"),
        State("fault-report-reason", "value"),
        State("fault-report-note", "value"),
        State("fault-report-target-store", "data"),
        State("flip-preview-store", "data"),
        State("catalog-store", "data"),
        State("dashboard-config-store", "data"),
        prevent_initial_call=True,
    )
    def submit_fault_report(
        _submit_clicks: int,
        _cancel_clicks: int,
        reason: str | None,
        note: str | None,
        target: dict[str, str] | None,
        flip_state: dict[str, Any] | None,
        catalog_state: dict[str, Any],
        config: dict[str, Any],
    ):
        if ctx.triggered_id == "fault-report-cancel":
            return layout.FAULT_PANEL_HIDDEN, "", no_update
        ok, message = _file_fault_report(target, reason, note, flip_state, catalog_state, config)
        if not ok:
            # Errors keep the dialog open so the typed note is not lost.
            return no_update, message, no_update
        return layout.FAULT_PANEL_HIDDEN, "", html.Div(message, className="status-text")

    @app.callback(
        Output("flip-pending-summary", "children"),
        *[Output(layout.flip_status_id(scope), "children") for scope in layout.FLIP_SCOPES],
        Input("flip-preview-store", "data"),
        *_FLIP_SELECTION_STATES,
        State("catalog-store", "data"),
    )
    def render_flip_status(
        flip_state: dict[str, Any] | None,
        run_a: str | None,
        group_a: str | None,
        trace_a: str | None,
        run_b: str | None,
        group_b: str | None,
        trace_b: str | None,
        overview_run: str | None,
        overview_group: str | None,
        catalog_state: dict[str, Any],
    ):
        state = flip_state or {}
        pending = len(state)
        summary = (
            "No unsaved sign flips."
            if not pending
            else f"{pending} unsaved sign-flip preview(s) - not written to disk yet."
        )
        workbench = {"a": (run_a, group_a, trace_a), "b": (run_b, group_b, trace_b)}
        statuses: list[str] = []
        for scope in layout.FLIP_SCOPES:
            selection = _flip_selection(
                scope, catalog_state, workbench=workbench, overview=(overview_run, overview_group)
            )
            if selection is None:
                statuses.append("No signal selected.")
                continue
            mode_file_path, group_name, trace_name = selection
            statuses.append(
                flips.describe_flip_state(
                    mode_file_path,
                    group_name,
                    trace_name,
                    previewed=bool(state.get(flips.flip_key(mode_file_path, group_name, trace_name))),
                )
            )
        return (summary, *statuses)

    @app.callback(
        Output("reference-registry-store", "data"),
        Output("flip-preview-store", "data", allow_duplicate=True),
        Output("reference-action-status", "children"),
        Input({"type": "reference-recording-button", "scope": ALL}, "n_clicks"),
        Input({"type": "orientation-check-button", "scope": ALL}, "n_clicks"),
        Input({"type": "reference-step-button", "scope": ALL}, "n_clicks"),
        Input({"type": "reference-clear-button", "scope": ALL}, "n_clicks"),
        *_FLIP_SELECTION_STATES,
        State("workbench-selection-store", "data"),
        State({"type": "overview-selection-store", "mode": ALL}, "data"),
        State("reference-registry-store", "data"),
        State("flip-preview-store", "data"),
        State("catalog-store", "data"),
        State("dashboard-config-store", "data"),
        prevent_initial_call=True,
    )
    def manage_references(
        _set_clicks, _check_clicks, _step_clicks, _clear_clicks,
        run_a, group_a, trace_a, run_b, group_b, trace_b, overview_run, overview_group,
        workbench_selection, overview_selections, registry_state, flip_state, catalog_state, config,
    ):
        triggered = ctx.triggered_id if isinstance(ctx.triggered_id, dict) else {}
        action, scope = triggered.get("type"), triggered.get("scope")
        selection = _flip_selection(
            scope, catalog_state,
            workbench={"a": (run_a, group_a, trace_a), "b": (run_b, group_b, trace_b)},
            overview=(overview_run, overview_group),
        )
        if selection is None:
            return no_update, no_update, html.Div("Select a limb signal first.", className="status-text")
        mode_file_path, group_name, _trace_name = selection
        record = _record_for_mode_file(catalog_state, mode_file_path)
        if record is None:
            return no_update, no_update, html.Div("The selected source is no longer in the catalog.", className="status-text")
        results_dir = config.get("results_dir", "outputs")
        default_fs = config.get("default_fs")
        state = dict(flip_state or {})
        try:
            if action == "reference-recording-button":
                updated, entry = dashboard_references.set_reference_recording(
                    record, group_name, results_dir=results_dir, default_fs=default_fs,
                )
                scenario = entry["scenario"]
                message = (
                    f"Reference recording set for {scenario['terrain']} / {scenario['direction']} / "
                    f"{scenario['footwear']} / {entry['side']}: {entry['source']['path']} "
                    f"({len(entry['mode1_templates'])} Mode 1 templates)."
                )
                return updated, state, html.Div(message, className="status-text")
            if action == "reference-clear-button":
                updated = dashboard_references.clear_reference(record, group_name, results_dir=results_dir)
                return updated, state, html.Div("Reference cleared for the selected scenario and side.", className="status-text")
            if action == "orientation-check-button":
                correlation, verdict = dashboard_references.check_orientation(
                    record, group_name, results_dir=results_dir, default_fs=default_fs,
                )
                key = flips.flip_key(mode_file_path, group_name)
                detail = f"Check Orientation r={correlation:.3f}"
                if verdict == "flip":
                    if key not in state:
                        _log_flip_event(
                            config, catalog_state, "preview_on", mode_file_path, group_name, "",
                            view_scope=scope or "", detail=detail,
                        )
                    state[key] = True
                    message = f"Orientation r={correlation:.3f}: flipped orientation; previewing Modes 1-3 for this group."
                elif verdict == "consistent":
                    if state.pop(key, None) is not None:
                        _log_flip_event(
                            config, catalog_state, "preview_off", mode_file_path, group_name, "",
                            view_scope=scope or "", detail=detail,
                        )
                    message = f"Orientation r={correlation:.3f}: consistent with the reference; no preview flip."
                else:
                    message = f"Orientation r={correlation:.3f}: undecided (|r| < 0.50); preview state was not changed."
                return registry_state, state, html.Div(message, className="status-text")
            if action == "reference-step-button":
                if scope == "workbench-a":
                    zoom = workbench_selection
                elif str(scope).startswith("overview-"):
                    mode = int(str(scope).split("-", 1)[1])
                    zoom = (overview_selections or [None, None, None])[mode - 1]
                else:
                    raise ValueError("Reference-step zoom selection is available from Signal A or an Overview mode plot")
                if not zoom:
                    raise ValueError("Zoom the raw time-series range before setting a reference step")
                start = zoom.get("raw_start_index", zoom.get("start_index"))
                end = zoom.get("raw_end_index", zoom.get("end_index"))
                if start is None or end is None:
                    raise ValueError("The current zoom range has no raw sample endpoints")
                updated, step = dashboard_references.set_reference_step(
                    record, group_name, int(start), int(end), results_dir=results_dir, default_fs=default_fs,
                )
                message = (
                    f"Reference step set for {group_name}: samples {step['start_sample']}–{step['end_sample']} "
                    f"({step['duration_seconds']:.3f} s), snapped to valleys."
                )
                return updated, state, html.Div(message, className="status-text")
        except (OSError, ValueError, KeyError) as error:
            return no_update, no_update, html.Div(str(error), className="status-text")
        return no_update, no_update, html.Div("Unknown reference action.", className="status-text")

    # --- raw-file ingest -------------------------------------------------------

    @app.callback(
        Output("ingest-parent-folder", "options"),
        Output("ingest-parent-folder", "value"),
        Input("raw-catalog-store", "data"),
        State("ingest-parent-folder", "value"),
    )
    def update_ingest_folder_options(raw_state: dict[str, Any], current: str | None):
        options = raw_catalog.get_folder_options(raw_state or {})
        return options, _select_valid_dropdown_value(options, current, "")

    @app.callback(
        Output("ingest-path-preview", "children"),
        Input("ingest-parent-folder", "value"),
        Input("ingest-new-subpath", "value"),
        Input("ingest-filename-override", "value"),
        Input("ingest-upload", "filename"),
        Input("ingest-overwrite-toggle", "value"),
        State("raw-catalog-store", "data"),
        State("dashboard-config-store", "data"),
    )
    def update_ingest_preview(
        parent: str | None,
        new_subpath: str | None,
        override: str | None,
        uploaded_name: str | None,
        overwrite: list[str] | None,
        raw_state: dict[str, Any],
        config: dict[str, Any],
    ):
        filename = (override or uploaded_name or "").strip()
        if not filename:
            return html.Div("Choose a file to see where it will be stored.", className="status-text")
        sample_data_dir = config.get("sample_data_dir", "Sample_Data")
        try:
            destination = raw_catalog.resolve_destination(
                sample_data_dir,
                parent_relative=parent or "",
                new_subpath=new_subpath,
                filename=filename,
                known_folders=(raw_state or {}).get("folders", []),
            )
        except ValueError as error:
            return html.Div(f"Cannot use that destination: {error}", className="status-text")

        lines = [html.Div(str(destination), className="status-line")]
        created = _folders_to_create(destination)
        if created:
            lines.append(html.Div(f"Creates {len(created)} new folder(s).", className="status-line"))
        if destination.exists():
            lines.append(
                html.Div(
                    "A file with that name already exists here"
                    + (" - it will be replaced." if "replace" in (overwrite or []) else "; tick Replace to overwrite."),
                    className="status-line",
                )
            )
        lines.append(html.Div(_hierarchy_hint(sample_data_dir, destination), className="status-line"))
        return html.Div(lines)

    @app.callback(
        Output("ingest-status", "children"),
        Output("raw-catalog-store", "data", allow_duplicate=True),
        Output("ingest-catalog-status", "children", allow_duplicate=True),
        Output("run-input-dropdown", "value", allow_duplicate=True),
        Output("ingest-upload", "contents"),
        Input("ingest-upload", "contents"),
        State("ingest-upload", "filename"),
        State("ingest-parent-folder", "value"),
        State("ingest-new-subpath", "value"),
        State("ingest-filename-override", "value"),
        State("ingest-overwrite-toggle", "value"),
        State("raw-catalog-store", "data"),
        State("dashboard-config-store", "data"),
        prevent_initial_call=True,
    )
    def handle_ingest_upload(
        contents: str | None,
        uploaded_name: str | None,
        parent: str | None,
        new_subpath: str | None,
        override: str | None,
        overwrite: list[str] | None,
        raw_state: dict[str, Any],
        config: dict[str, Any],
    ):
        if not contents:
            return no_update, no_update, no_update, no_update, no_update

        sample_data_dir = config.get("sample_data_dir", "Sample_Data")
        results_dir = config.get("results_dir", "outputs")
        max_bytes = int(config.get("max_upload_bytes", raw_catalog.DEFAULT_MAX_UPLOAD_BYTES))
        filename = (override or uploaded_name or "").strip()
        try:
            destination = raw_catalog.resolve_destination(
                sample_data_dir,
                parent_relative=parent or "",
                new_subpath=new_subpath,
                filename=filename,
                known_folders=(raw_state or {}).get("folders", []),
            )
            payload = raw_catalog.decode_upload_contents(contents, max_bytes=max_bytes)
        except ValueError as error:
            # Reset contents so the same file can be retried after a fix.
            return _status_div(str(error)), no_update, no_update, no_update, None

        result = raw_catalog.write_raw_csv(
            payload,
            destination,
            sample_data_dir=sample_data_dir,
            replace="replace" in (overwrite or []),
        )
        if not result.ok:
            return _status_div(result.message), no_update, no_update, no_update, None

        raw_catalog.clear_caches()
        refreshed = raw_catalog.scan_raw_catalog(sample_data_dir, results_dir)
        stored = raw_catalog.get_raw_file_record_by_path(refreshed, result.destination)
        return (
            _ingest_status_children(result),
            refreshed,
            raw_catalog.get_raw_catalog_summary(refreshed),
            stored["raw_id"] if stored else no_update,
            None,
        )

    @app.callback(
        Output("raw-catalog-store", "data", allow_duplicate=True),
        Output("ingest-catalog-status", "children", allow_duplicate=True),
        Input("ingest-refresh-button", "n_clicks"),
        State("dashboard-config-store", "data"),
        prevent_initial_call=True,
    )
    def refresh_raw_catalog(_n_clicks: int, config: dict[str, Any]):
        raw_catalog.clear_caches()
        refreshed = raw_catalog.scan_raw_catalog(
            config.get("sample_data_dir", "Sample_Data"), config.get("results_dir", "outputs")
        )
        return refreshed, raw_catalog.get_raw_catalog_summary(refreshed)

    # --- run launcher ----------------------------------------------------------

    @app.callback(
        Output("run-dataset-dropdown", "options"),
        Output("run-dataset-dropdown", "value"),
        Input("raw-catalog-store", "data"),
        State("run-dataset-dropdown", "value"),
    )
    def update_run_dataset_options(raw_state: dict[str, Any], current: str | None):
        options = raw_catalog.get_dataset_options(raw_state or {})
        return options, _select_valid_dropdown_value(options, current, "all")

    @app.callback(
        Output("run-input-dropdown", "options"),
        Output("run-input-dropdown", "value"),
        Input("raw-catalog-store", "data"),
        Input("run-dataset-dropdown", "value"),
        Input("run-file-filter", "value"),
        State("run-input-dropdown", "value"),
    )
    def update_run_input_options(
        raw_state: dict[str, Any],
        dataset: str | None,
        file_filter: str | None,
        current: str | None,
    ):
        options = raw_catalog.get_raw_file_options(
            raw_state or {},
            dataset=dataset,
            unprocessed_only=(file_filter or "unprocessed") == "unprocessed",
        )
        default = options[0]["value"] if options else None
        return options, _select_valid_dropdown_value(options, current, default)

    @app.callback(
        Output("run-output-target", "options"),
        Output("run-output-target", "value"),
        Input("catalog-store", "data"),
        State("run-output-target", "value"),
        State("dashboard-config-store", "data"),
    )
    def update_run_output_target_options(
        _catalog_state: dict[str, Any], current: str | None, config: dict[str, Any]
    ):
        options = [{"label": "Most recent existing run root", "value": layout.DEFAULT_OUTPUT_TARGET}]
        options.extend(
            raw_catalog.list_output_run_roots(
                config.get("results_dir", "outputs"), config.get("sample_data_dir", "Sample_Data")
            )
        )
        return options, _select_valid_dropdown_value(options, current, layout.DEFAULT_OUTPUT_TARGET)

    @app.callback(
        Output("run-input-metadata", "children"),
        Output("run-estimate", "children"),
        Input("run-input-dropdown", "value"),
        Input("run-epochs", "value"),
        Input("run-ssa-window", "value"),
        Input("run-compute-device", "value"),
        Input("run-output-target", "value"),
        State("raw-catalog-store", "data"),
        State("dashboard-config-store", "data"),
    )
    def update_run_metadata_and_estimate(
        raw_id: str | None,
        epochs: Any,
        ssa_window: Any,
        device: str | None,
        output_target: str | None,
        raw_state: dict[str, Any],
        config: dict[str, Any],
    ):
        record = raw_catalog.get_raw_file_record(raw_state or {}, raw_id)
        if record is None:
            return "Select a raw CSV to process.", ""

        output_root = _resolve_output_root(output_target, config)
        leaf = jobs.output_leaf_for(output_root, record["abs_path"])
        meta_lines = [
            html.Div(f"{record['relative_path']} ({record['size_label']})", className="status-line"),
            html.Div(f"Output leaf: {leaf}", className="status-line"),
        ]
        if record["processed"]:
            meta_lines.append(
                html.Div(
                    "This file already has outputs in "
                    f"{len(record['processed_run_roots'])} run root(s).",
                    className="status-line",
                )
            )
        if leaf.exists() and any(leaf.glob("limb_angle_mode.csv")):
            meta_lines.append(
                html.Div(
                    "Outputs already exist at that leaf - re-running overwrites them. "
                    "Tick the overwrite box below to allow it.",
                    className="status-line",
                )
            )
        meta_lines.append(html.Div(_hierarchy_hint(config.get("sample_data_dir", "Sample_Data"), Path(record["abs_path"])), className="status-line"))

        estimate = jobs.estimate_seconds(epochs)
        estimate_text = (
            f"Estimated runtime ~{jobs.format_duration(estimate)} "
            f"({jobs.clamp_epochs(epochs)} epochs, SSA window {jobs.clamp_ssa_window(ssa_window)}, "
            f"{jobs.normalize_compute_device(device)})."
        )
        return html.Div(meta_lines), estimate_text

    @app.callback(
        Output("run-status", "children"),
        Output("jobs-refresh-token", "data"),
        Input("run-start-button", "n_clicks"),
        State("run-input-dropdown", "value"),
        State("run-epochs", "value"),
        State("run-ssa-window", "value"),
        State("run-compute-device", "value"),
        State("run-output-target", "value"),
        State("run-overwrite-confirm", "value"),
        State("raw-catalog-store", "data"),
        State("dashboard-config-store", "data"),
        State("jobs-refresh-token", "data"),
        prevent_initial_call=True,
    )
    def start_rss_job(
        _n_clicks: int,
        raw_id: str | None,
        epochs: Any,
        ssa_window: Any,
        device: str | None,
        output_target: str | None,
        overwrite: list[str] | None,
        raw_state: dict[str, Any],
        config: dict[str, Any],
        token: int | None,
    ):
        record = raw_catalog.get_raw_file_record(raw_state or {}, raw_id)
        if record is None:
            return _status_div("Select a raw CSV first."), no_update

        results_dir = config.get("results_dir", "outputs")
        output_root = _resolve_output_root(output_target, config)
        jobs_dir = jobs.jobs_dir_for(results_dir)

        existing = jobs.find_active_job_for(jobs_dir, record["abs_path"], output_root)
        if existing is not None:
            return (
                _status_div(f"That file is already {existing.status} as job {existing.job_id}."),
                no_update,
            )

        leaf = jobs.output_leaf_for(output_root, record["abs_path"])
        if (leaf / "limb_angle_mode.csv").exists() and "overwrite" not in (overwrite or []):
            return (
                _status_div(
                    f"{leaf} already holds processed outputs. Tick 'Overwrite the existing "
                    "outputs for this file' to replace them."
                ),
                no_update,
            )

        request = jobs.build_job_request(
            csv_path=record["abs_path"],
            output_root=output_root,
            results_dir=results_dir,
            project_root=config.get("project_root", "."),
            python_bin=config.get("python_bin", jobs.DEFAULT_PYTHON_BIN),
            epochs=epochs,
            ssa_window_length=ssa_window,
            compute_device=device,
            csv_label=record["relative_path"],
        )
        job_record = jobs.submit_job(request)
        return (
            _status_div(
                f"Job {job_record.job_id} is {job_record.status}. "
                f"Estimated runtime ~{jobs.format_duration(job_record.estimated_seconds)}."
            ),
            int(token or 0) + 1,
        )

    # --- job monitoring --------------------------------------------------------

    @app.callback(
        Output("jobs-status", "children", allow_duplicate=True),
        Output("jobs-refresh-token", "data", allow_duplicate=True),
        Input("jobs-cancel-button", "n_clicks"),
        State("jobs-log-selector", "value"),
        State("dashboard-config-store", "data"),
        State("jobs-refresh-token", "data"),
        prevent_initial_call=True,
    )
    def cancel_rss_job(_n_clicks: int, job_id: str | None, config: dict[str, Any], token: int | None):
        if not job_id:
            return "Select a job in the Job Log dropdown first.", no_update
        record = jobs.request_cancel(jobs.jobs_dir_for(config.get("results_dir", "outputs")), job_id)
        if record is None:
            return f"Job {job_id} was not found.", no_update
        return f"Job {job_id}: {record.message}", int(token or 0) + 1

    @app.callback(
        Output("jobs-status", "children", allow_duplicate=True),
        Output("jobs-refresh-token", "data", allow_duplicate=True),
        Input("jobs-clear-button", "n_clicks"),
        State("dashboard-config-store", "data"),
        State("jobs-refresh-token", "data"),
        prevent_initial_call=True,
    )
    def clear_finished_jobs(_n_clicks: int, config: dict[str, Any], token: int | None):
        removed = jobs.prune_terminal_jobs(jobs.jobs_dir_for(config.get("results_dir", "outputs")))
        return f"Removed {removed} finished job record(s).", int(token or 0) + 1

    @app.callback(
        Output("jobs-refresh-token", "data", allow_duplicate=True),
        Input("jobs-refresh-button", "n_clicks"),
        State("jobs-refresh-token", "data"),
        prevent_initial_call=True,
    )
    def bump_jobs_refresh(_n_clicks: int, token: int | None):
        return int(token or 0) + 1

    @app.callback(
        Output("jobs-state-store", "data"),
        Output("jobs-seen-store", "data"),
        Output("jobs-poll-interval", "interval"),
        Output("catalog-store", "data", allow_duplicate=True),
        Output("catalog-status", "children", allow_duplicate=True),
        Output("raw-catalog-store", "data", allow_duplicate=True),
        Input("jobs-poll-interval", "n_intervals"),
        Input("jobs-refresh-token", "data"),
        State("dashboard-config-store", "data"),
        State("jobs-seen-store", "data"),
        prevent_initial_call=True,
    )
    def poll_jobs_state(
        _n_intervals: int,
        _token: int | None,
        config: dict[str, Any],
        seen: dict[str, Any] | None,
    ):
        """Reap and promote jobs, and refresh the catalogs when one finishes.

        Touches only top-level components, so it keeps running while the user is
        on another tab -- which is the whole point of the auto-refresh.
        """
        results_dir = config.get("results_dir", "outputs")
        jobs_dir = jobs.jobs_dir_for(results_dir)
        jobs.pump(jobs_dir, max_concurrent=int(config.get("max_concurrent_jobs", jobs.MAX_CONCURRENT_JOBS)))
        rows = jobs.snapshot(jobs_dir)

        previous = dict(seen or {})
        current = {row["job_id"]: row["status"] for row in rows}
        newly_finished = [
            job_id
            for job_id, status in current.items()
            if status in jobs.TERMINAL_STATUSES and previous.get(job_id) != status
        ]
        active = any(status in jobs.ACTIVE_STATUSES for status in current.values())
        interval = layout.JOBS_POLL_ACTIVE_MS if active else layout.JOBS_POLL_IDLE_MS

        if not newly_finished:
            return rows, current, interval, no_update, no_update, no_update

        # A run just landed: drop every cache that could hide it and rescan.
        catalog.clear_caches()
        signals.clear_caches()
        transfer_entropy.clear_caches()
        te_matrix.clear_caches()
        raw_catalog.clear_caches()
        refreshed_catalog = catalog.scan_results_catalog(results_dir)
        refreshed_raw = raw_catalog.scan_raw_catalog(
            config.get("sample_data_dir", "Sample_Data"), results_dir
        )
        return (
            rows,
            current,
            interval,
            refreshed_catalog,
            catalog.get_catalog_summary(refreshed_catalog),
            refreshed_raw,
        )

    @app.callback(
        Output("jobs-panel", "children"),
        Output("jobs-log-selector", "options"),
        Output("jobs-log-selector", "value"),
        Output("jobs-log-tail", "children"),
        Output("jobs-status", "children"),
        Input("jobs-state-store", "data"),
        Input("jobs-log-selector", "value"),
        State("dashboard-config-store", "data"),
    )
    def render_jobs_panel(
        rows: list[dict[str, Any]] | None,
        selected: str | None,
        config: dict[str, Any],
    ):
        rows = rows or []
        if not rows:
            return (
                html.Div("No pipeline jobs yet.", className="status-text"),
                [],
                None,
                "No job selected.",
                "No pipeline jobs yet.",
            )

        options = [
            {"label": f"{row['job_id']} - {row['status']}", "value": row["job_id"]} for row in rows
        ]
        value = _select_valid_dropdown_value(options, selected, options[0]["value"])
        record = jobs.get_job(jobs.jobs_dir_for(config.get("results_dir", "outputs")), value)
        tail = jobs.read_progress(record).log_tail if record is not None else ""

        counts: dict[str, int] = {}
        for row in rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        summary = " · ".join(f"{count} {status}" for status, count in sorted(counts.items()))
        return (
            [_job_card(row) for row in rows],
            options,
            value,
            tail or "No output yet.",
            summary,
        )


def _build_all_segment_matrix_status(result: dict[str, Any]) -> html.Div:
    """Status lines for the all-segment TE matrix.

    Not shared with ``_build_matrix_status``: that one reports the legacy
    gait-toolkit's ``history_grid``/``tau_grid``, which this estimator has no
    equivalent of.
    """
    if not result.get("available"):
        return html.Div(result.get("message") or "Matrix unavailable.", className="status-text")
    source = result["source_context"]
    lines = [
        f"{len(result['segments'])} segments x {len(result['segments'])} "
        f"({len(result['segments']) * (len(result['segments']) - 1)} directed pairs), "
        f"trace {result['trace_name']}",
        f"{result['rows_finite']} of {result['rows_total']} rows finite across all "
        f"segments; {result['n_samples_used']} embedded samples per cell",
        f"bins {result['bins']} - lag {source['lag']}, length {source['length']}, "
        f"stride {source['stride']} (length and stride fixed at 1)",
        f"computed in {result['elapsed_seconds']:.2f} s",
    ]
    if result.get("skipped"):
        skipped = ", ".join(f"{item['segment']} ({item['reason']})" for item in result["skipped"])
        lines.append(f"skipped: {skipped}")
    return html.Div([html.Div(line, className="status-line") for line in lines])


def _triggered_scope() -> str | None:
    """The ``scope`` key of the pattern-matched control that fired."""
    triggered = ctx.triggered_id
    if isinstance(triggered, dict):
        return triggered.get("scope")
    return None


def _flip_selection(
    scope: str | None,
    catalog_state: dict[str, Any],
    *,
    workbench: dict[str, tuple[str | None, str | None, str | None]],
    overview: tuple[str | None, str | None],
) -> tuple[str, str, str] | None:
    """Resolve one flip scope to ``(mode_file_path, group, trace)``.

    Scopes are the five places a signal can be selected; the flip itself is
    keyed on the resolved triple, so the same decision applies everywhere that
    signal is plotted.
    """
    if not scope:
        return None
    if scope.startswith("workbench-"):
        run_id, group_name, trace_name = workbench.get(scope.split("-", 1)[1], (None, None, None))
    elif scope.startswith("overview-"):
        run_id, group_name = overview
        trace_name = f"mode_{scope.split('-', 1)[1]}"
    else:
        return None
    record = catalog.get_run_record(catalog_state, run_id)
    if record is None or not group_name or not trace_name:
        return None
    return str(record["mode_file_path"]), str(group_name), str(trace_name)


def _record_for_mode_file(catalog_state: dict[str, Any], mode_file_path: str) -> dict[str, Any] | None:
    """The catalog record owning ``mode_file_path``, or None if it left the catalog."""
    return next(
        (item for item in catalog.list_run_records(catalog_state) if str(item.get("mode_file_path")) == str(mode_file_path)),
        None,
    )


def _file_fault_report(
    target: dict[str, str] | None,
    reason: str | None,
    note: str | None,
    flip_state: dict[str, Any] | None,
    catalog_state: dict[str, Any],
    config: dict[str, Any],
) -> tuple[bool, str]:
    """Log one faulty-signal report for the dialog's target; ``(ok, message)``."""
    if not target:
        return False, "No signal is selected for this report; close the dialog and try again."
    record = _record_for_mode_file(catalog_state, target["mode_file_path"])
    if record is None:
        return False, "That signal is no longer in the catalog; refresh the file list and try again."
    mode_file_path, group_name, trace_name = target["mode_file_path"], target["group_name"], target["trace_name"]
    outcome = signal_logs.record_fault_report(
        config.get("results_dir", "outputs"),
        record=record,
        mode_file_path=mode_file_path,
        group_name=group_name,
        trace_name=trace_name,
        reason=reason,
        note=note,
        # A preview only changes what the committable traces show.
        previewed_flip=trace_name in flips.COMMITTABLE_TRACES
        and bool((flip_state or {}).get(flips.flip_key(mode_file_path, group_name))),
        committed_flips=flips.count_manual_flips(mode_file_path, group_name, "mode_1"),
        view_scope=target.get("scope") or "",
    )
    if not outcome.ok:
        return False, outcome.message
    repeat = f" ({outcome.report_count} reports on file for this signal.)" if outcome.report_count > 1 else ""
    return True, outcome.message + repeat


def _log_flip_event(
    config: dict[str, Any],
    catalog_state: dict[str, Any],
    event: str,
    mode_file_path: str,
    group_name: str,
    trace_name: str,
    *,
    view_scope: str = "",
    operation_id: str = "",
    detail: str = "",
) -> str | None:
    """Append one row to the flip history; returns an error message, never raises.

    A failed log write must not undo or block the flip decision it describes, so
    the preview paths only warn and the commit path shows the message.
    """
    error = signal_logs.log_flip_events(
        config.get("results_dir", "outputs"),
        [signal_logs.FlipEvent(
            event=event, mode_file_path=mode_file_path, group_name=group_name, trace_name=trace_name,
            record=_record_for_mode_file(catalog_state, mode_file_path), view_scope=view_scope,
            operation_id=operation_id, detail=detail,
        )],
    )
    if error:
        logging.getLogger(__name__).warning("Flip history not updated: %s", error)
    return error


def _load_selected_signal(
    catalog_state: dict[str, Any],
    config: dict[str, Any],
    run_id: str | None,
    group_name: str | None,
    trace_name: str | None,
    *,
    slot_label: str,
    flip_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    record = catalog.get_run_record(catalog_state, run_id)
    if record is None or not group_name or not trace_name:
        return None
    try:
        signal = signals.load_signal_trace(
            record["mode_file_path"],
            group_name,
            trace_name,
            default_fs=config.get("default_fs"),
        )
    except Exception:
        return None
    series_key = _safe_token(f"{slot_label}_{group_name}_{trace_name}")
    signal["run_id"] = run_id
    signal["run_label"] = record["label"]
    signal["group_name"] = group_name
    signal["legend_name"] = f"{slot_label}: {signal['group_label']} | {signal['trace_label']}"
    signal["selection_label"] = f"{record['label']} | {signal['group_label']} | {signal['trace_label']}"
    signal["family"] = record["family"]
    signal["series_key"] = series_key

    key = flips.flip_key(record["mode_file_path"], group_name, trace_name)
    previewed = trace_name in flips.COMMITTABLE_TRACES and bool((flip_state or {}).get(key))
    signal["flip_key"] = key
    signal["preview_flipped"] = previewed
    if previewed:
        # Must allocate: signal["values"] is a view onto the array inside
        # signals._load_group_data_cached, so an in-place "values *= -1" would
        # poison that cache for every other consumer.
        signal["values"] = -signal["values"]
        signal["legend_name"] = f"{signal['legend_name']} (flipped preview)"
    return signal


def _compute_marker_indices(signal: dict[str, Any] | None) -> list[int]:
    """Sample indices worth snapping/paging to: transient and step boundaries.

    Always includes the first and last sample so a signal with no real markers
    still snaps/pages to something instead of having nowhere to go.
    """
    if signal is None:
        return []
    values = signal.get("values")
    if values is None:
        return []
    n = len(values)
    if n == 0:
        return []
    markers = {0, n - 1}
    for step in signal.get("gait_steps") or []:
        if bool(step.get("accepted", False)):
            for name in ("start_sample", "end_sample"):
                sample = int(step.get(name, -1))
                if 0 <= sample < n:
                    markers.add(sample)
    return sorted(markers)


def _build_markers_payload(signal: dict[str, Any] | None, *, signal_key: str) -> dict[str, Any] | None:
    """Snapshot of one signal's x-axis and marker indices, for window snapping/paging."""
    if signal is None:
        return None
    values = signal.get("values")
    if values is None or len(values) == 0:
        return None
    x_source = signal.get("time") if signal.get("time") is not None else signal.get("sample_index")
    if x_source is None:
        return None
    x_values = np.asarray(x_source, dtype=float).reshape(-1)
    return {
        "signal_key": signal_key,
        "x": x_values.tolist(),
        "markers": _compute_marker_indices(signal),
    }


def _resolve_window(selection: dict[str, Any] | None, signal_key: str) -> tuple[int, int] | None:
    """Read back a stored selection, ignoring one left over from a different signal."""
    if not selection or selection.get("signal_key") != signal_key:
        return None
    start = selection.get("start_index")
    end = selection.get("end_index")
    if start is None or end is None or end <= start:
        return None
    return (int(start), int(end))


def _nearest_index(x_values: np.ndarray, target: float) -> int:
    """Index of the value in a sorted array closest to ``target``."""
    pos = int(np.searchsorted(x_values, target))
    if pos <= 0:
        return 0
    if pos >= x_values.size:
        return x_values.size - 1
    before, after = pos - 1, pos
    return before if abs(x_values[before] - target) <= abs(x_values[after] - target) else after


def _nearest_marker(markers: list[int], index: int) -> int:
    """Snap a sample index to the nearest entry in a sorted marker list."""
    if not markers:
        return index
    markers_arr = np.asarray(markers)
    pos = int(np.searchsorted(markers_arr, index))
    if pos <= 0:
        return int(markers_arr[0])
    if pos >= markers_arr.size:
        return int(markers_arr[-1])
    before, after = markers_arr[pos - 1], markers_arr[pos]
    return int(before if abs(int(before) - index) <= abs(int(after) - index) else after)


def _snap_selection_from_relayout(
    relayout_data: dict[str, Any] | None,
    current_selection: dict[str, Any] | None,
    markers_data: dict[str, Any] | None,
) -> Any:
    """Convert a Plotly ``relayoutData`` zoom/rangeslider event into a marker-snapped window.

    Returns ``no_update`` when the event carries no usable x-range (e.g. a
    non-axis relayout) or the snapped result already matches the stored
    selection, so a figure redraw triggered by our own store write does not
    re-enter this callback.
    """
    if not relayout_data or not markers_data:
        return no_update

    x0 = relayout_data.get("xaxis.range[0]")
    x1 = relayout_data.get("xaxis.range[1]")
    if x0 is None or x1 is None:
        xrange = relayout_data.get("xaxis.range")
        if isinstance(xrange, (list, tuple)) and len(xrange) == 2:
            x0, x1 = xrange
    if x0 is None or x1 is None:
        if "xaxis.autorange" in relayout_data:
            return None if current_selection is not None else no_update
        return no_update

    x_values = np.asarray(markers_data.get("x") or [], dtype=float)
    markers = list(markers_data.get("markers") or [])
    if x_values.size == 0:
        return no_update

    raw_lo_idx = _nearest_index(x_values, float(min(x0, x1)))
    raw_hi_idx = _nearest_index(x_values, float(max(x0, x1)))
    lo_idx = _nearest_marker(markers, raw_lo_idx)
    hi_idx = _nearest_marker(markers, raw_hi_idx)
    if hi_idx <= lo_idx:
        hi_idx = min(x_values.size - 1, lo_idx + 1)

    signal_key = str(markers_data.get("signal_key"))
    if (
        current_selection
        and current_selection.get("signal_key") == signal_key
        and current_selection.get("start_index") == lo_idx
        and current_selection.get("end_index") == hi_idx
    ):
        return no_update
    return {
        "start_index": lo_idx, "end_index": hi_idx,
        "raw_start_index": raw_lo_idx, "raw_end_index": raw_hi_idx,
        "signal_key": signal_key,
    }


def _page_selection(
    current_selection: dict[str, Any] | None,
    markers_data: dict[str, Any] | None,
    direction: str,
) -> Any:
    """Shift both edges of the current window to the previous/next marker, together."""
    if not current_selection or not markers_data:
        return no_update
    markers = list(markers_data.get("markers") or [])
    if len(markers) < 2:
        return no_update
    markers_arr = np.asarray(markers)
    start = int(current_selection.get("start_index", markers[0]))
    end = int(current_selection.get("end_index", markers[-1]))

    start_pos = int(np.searchsorted(markers_arr, start))
    end_pos = int(np.searchsorted(markers_arr, end))
    step = 1 if direction == "next" else -1
    new_start_pos = max(0, min(markers_arr.size - 1, start_pos + step))
    new_end_pos = max(0, min(markers_arr.size - 1, end_pos + step))
    if new_end_pos <= new_start_pos:
        return no_update

    return {
        "start_index": int(markers_arr[new_start_pos]),
        "end_index": int(markers_arr[new_end_pos]),
        "signal_key": current_selection.get("signal_key", markers_data.get("signal_key")),
    }


def _resolve_workbench_plot_signals(
    signal_a: dict[str, Any] | None,
    signal_b: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if signal_a is not None:
        return signal_a, signal_b
    if signal_b is not None:
        return signal_b, None
    return None, None


def _hierarchy_dropdown_response(
    catalog_state: dict[str, Any],
    field: str,
    current_value: str | None,
    *,
    current_run: str | None = None,
    subject: str | None = None,
    terrain: str | None = None,
    activity: str | None = None,
    experiment: str | None = None,
) -> tuple[list[dict[str, str]], str | None]:
    options = catalog.get_hierarchy_options(
        catalog_state,
        field,
        subject=subject,
        terrain=terrain,
        activity=activity,
        experiment=experiment,
    )
    default_value = catalog.get_default_hierarchy_value(
        catalog_state,
        field,
        current_run,
        subject=subject,
        terrain=terrain,
        activity=activity,
        experiment=experiment,
    )
    return options, _select_valid_dropdown_value(options, current_value, default_value)


def _run_dropdown_response(
    catalog_state: dict[str, Any],
    current_run: str | None,
    *,
    subject: str | None = None,
    terrain: str | None = None,
    activity: str | None = None,
    experiment: str | None = None,
) -> tuple[list[dict[str, str]], str | None]:
    options = catalog.get_run_options(
        catalog_state,
        subject=subject,
        terrain=terrain,
        activity=activity,
        experiment=experiment,
    )
    default_value = str(options[0]["value"]) if options else catalog.get_first_run_id(catalog_state)
    return options, _select_valid_dropdown_value(options, current_run, default_value)


#: Plot kinds that get their underlying data sliced to the selected window.
#: Time series is deliberately excluded: it keeps showing the full recording
#: (with the rangeslider intact) and is only given an x-axis range instead, so
#: the small strip always reflects the whole signal.
_WINDOWED_PLOT_KINDS = frozenset(
    {
        "fft",
        "pdf",
        "cdf",
        "phase-portrait",
        "poincare-transform",
        "hilbert-analytic",
        "instantaneous-phase",
        "instantaneous-frequency",
    }
)


def _slice_signal_to_window(
    signal: dict[str, Any] | None, window: tuple[int, int] | None
) -> dict[str, Any] | None:
    """Restrict a signal dict's sample-aligned arrays to ``[start, end]`` inclusive."""
    if signal is None or window is None:
        return signal
    values = signal.get("values")
    if values is None:
        return signal
    n = len(values)
    if n == 0:
        return signal
    start, end = window
    start = max(0, min(int(start), n - 1))
    end = max(0, min(int(end), n - 1))
    if end <= start:
        end = min(n - 1, start + 1)
    sliced = dict(signal)
    for key in ("values", "time", "sample_index", "frame", "segment_label", "step_index"):
        value = signal.get(key)
        if value is not None:
            sliced[key] = value[start : end + 1]
    sliced["gait_events"] = [
        {**event, "sample": int(event["sample"]) - start}
        for event in signal.get("gait_events") or []
        if start <= int(event.get("sample", -1)) <= end
    ]
    sliced["gait_steps"] = [
        {
            **step,
            "start_sample": max(0, int(step["start_sample"]) - start),
            "end_sample": min(end - start, int(step["end_sample"]) - start),
        }
        for step in signal.get("gait_steps") or []
        if int(step.get("end_sample", -1)) >= start and int(step.get("start_sample", n)) <= end
    ]
    return sliced


def _apply_window_range(figure, signal: dict[str, Any] | None, window: tuple[int, int]) -> None:
    """Restrict a figure's visible x-range to the window without touching its data/rangeslider."""
    if signal is None:
        return
    x_source = signal.get("time") if signal.get("time") is not None else signal.get("sample_index")
    if x_source is None:
        return
    x_values = np.asarray(x_source, dtype=float).reshape(-1)
    if x_values.size == 0:
        return
    start, end = window
    start = max(0, min(int(start), x_values.size - 1))
    end = max(start, min(int(end), x_values.size - 1))
    figure.update_xaxes(range=[float(x_values[start]), float(x_values[end])])


def _build_workbench_single_plot_figure(
    plot_kind: str,
    signal_a: dict[str, Any],
    signal_b: dict[str, Any] | None,
    bins: int,
    *,
    unwrap: bool = False,
    window: tuple[int, int] | None = None,
):
    if plot_kind not in _WINDOWED_PLOT_KINDS:
        figure = plotting.build_overlay_time_series_figure(signal_a, signal_b)
        if window is not None:
            _apply_window_range(figure, signal_a, window)
        return figure

    windowed_a = _slice_signal_to_window(signal_a, window)
    windowed_b = _slice_signal_to_window(signal_b, window)
    if plot_kind == "fft":
        return plotting.build_overlay_fft_figure(windowed_a, windowed_b)
    if plot_kind == "pdf":
        return plotting.build_overlay_pdf_figure(windowed_a, windowed_b, bins=bins)
    if plot_kind == "cdf":
        return plotting.build_overlay_cdf_figure(windowed_a, windowed_b, bins=bins)
    if plot_kind == "phase-portrait":
        return plotting.build_overlay_phase_portrait_figure(windowed_a, windowed_b)
    if plot_kind == "poincare-transform":
        return plotting.build_overlay_poincare_transform_figure(windowed_a, windowed_b)
    if plot_kind == "hilbert-analytic":
        return plotting.build_overlay_hilbert_figure(windowed_a, windowed_b)
    if plot_kind == "instantaneous-phase":
        return plotting.build_overlay_instantaneous_phase_figure(windowed_a, windowed_b, unwrap=unwrap)
    return plotting.build_overlay_instantaneous_frequency_figure(windowed_a, windowed_b)


def _build_single_signal_figure(
    plot_kind: str,
    signal: dict[str, Any],
    bins: int,
    *,
    unwrap: bool = False,
    window: tuple[int, int] | None = None,
):
    if plot_kind not in _WINDOWED_PLOT_KINDS:
        figure = plotting.build_overlay_time_series_figure(signal)
        if window is not None:
            _apply_window_range(figure, signal, window)
        return figure

    windowed = _slice_signal_to_window(signal, window)
    if plot_kind == "fft":
        return plotting.build_overlay_fft_figure(windowed)
    if plot_kind == "pdf":
        return plotting.build_overlay_pdf_figure(windowed, bins=bins)
    if plot_kind == "cdf":
        return plotting.build_overlay_cdf_figure(windowed, bins=bins)
    if plot_kind == "phase-portrait":
        return plotting.build_overlay_phase_portrait_figure(windowed)
    if plot_kind == "poincare-transform":
        return plotting.build_overlay_poincare_transform_figure(windowed)
    if plot_kind == "hilbert-analytic":
        return plotting.build_overlay_hilbert_figure(windowed)
    if plot_kind == "instantaneous-phase":
        return plotting.build_overlay_instantaneous_phase_figure(windowed, unwrap=unwrap)
    return plotting.build_overlay_instantaneous_frequency_figure(windowed)


def _build_workbench_single_plot_frame(
    plot_kind: str,
    signal_a: dict[str, Any],
    signal_b: dict[str, Any] | None,
    bins: int,
    *,
    unwrap: bool = False,
) -> pd.DataFrame:
    if plot_kind == "fft":
        return plotting.build_fft_frame(signal_a, signal_b)
    if plot_kind == "pdf":
        return plotting.build_pdf_frame(signal_a, signal_b, bins=bins)
    if plot_kind == "cdf":
        return plotting.build_cdf_frame(signal_a, signal_b, bins=bins)
    if plot_kind == "phase-portrait":
        return plotting.build_phase_portrait_frame(signal_a, signal_b)
    if plot_kind == "poincare-transform":
        return plotting.build_poincare_transform_frame(signal_a, signal_b)
    if plot_kind == "hilbert-analytic":
        return plotting.build_hilbert_frame(signal_a, signal_b)
    if plot_kind == "instantaneous-phase":
        return plotting.build_instantaneous_phase_frame(signal_a, signal_b, unwrap=unwrap)
    if plot_kind == "instantaneous-frequency":
        return plotting.build_instantaneous_frequency_frame(signal_a, signal_b)
    return plotting.build_time_series_frame(signal_a, signal_b)


def _single_signal_export_frame(
    signal: dict[str, Any],
    plot_kind: str,
    bins: int,
    *,
    unwrap: bool = False,
) -> pd.DataFrame:
    if plot_kind == "fft":
        return _normalize_single_frame_columns(plotting.build_fft_frame(signal))
    if plot_kind == "pdf":
        return _normalize_single_frame_columns(plotting.build_pdf_frame(signal, bins=bins))
    if plot_kind == "cdf":
        return _normalize_single_frame_columns(plotting.build_cdf_frame(signal, bins=bins))
    if plot_kind == "phase-portrait":
        return _normalize_single_frame_columns(plotting.build_phase_portrait_frame(signal))
    if plot_kind == "poincare-transform":
        return _normalize_single_frame_columns(plotting.build_poincare_transform_frame(signal))
    if plot_kind == "hilbert-analytic":
        return _normalize_single_frame_columns(plotting.build_hilbert_frame(signal))
    if plot_kind == "instantaneous-phase":
        return _normalize_single_frame_columns(
            plotting.build_instantaneous_phase_frame(signal, unwrap=unwrap)
        )
    if plot_kind == "instantaneous-frequency":
        return _normalize_single_frame_columns(plotting.build_instantaneous_frequency_frame(signal))
    return _normalize_single_frame_columns(plotting.build_time_series_frame(signal))


def _normalize_single_frame_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed: dict[str, str] = {}
    for column in frame.columns:
        if column.endswith("_x"):
            renamed[column] = "x_value"
        # These four must precede the bare "frequency_hz" /
        # "frequency_cycles_per_sample" rules below: this chain is a sequential
        # elif, and those FFT-axis rules would otherwise swallow
        # "<key>_instantaneous_frequency_hz" and rename it to the FFT axis name.
        elif column.endswith("_instantaneous_phase_unwrapped"):
            renamed[column] = "instantaneous_phase_unwrapped"
        elif column.endswith("_instantaneous_phase"):
            renamed[column] = "instantaneous_phase"
        elif column.endswith("_instantaneous_frequency_cycles_per_sample"):
            renamed[column] = "instantaneous_frequency_cycles_per_sample"
        elif column.endswith("_instantaneous_frequency_hz"):
            renamed[column] = "instantaneous_frequency_hz"
        elif column.endswith("_signal_value"):
            renamed[column] = "signal_value"
        elif column.endswith("_magnitude"):
            renamed[column] = "magnitude"
        elif column.endswith("frequency_hz"):
            renamed[column] = "frequency_hz"
        elif column.endswith("frequency_cycles_per_sample"):
            renamed[column] = "frequency_cycles_per_sample"
        elif column.endswith("_density"):
            renamed[column] = "density"
        elif column.endswith("_cdf"):
            renamed[column] = "cdf"
        elif column.endswith("_count"):
            renamed[column] = "count"
        elif column.endswith("_derivative"):
            renamed[column] = "derivative"
        elif column.endswith("_hilbert_imag"):
            renamed[column] = "hilbert_imag"
    return frame.rename(columns=renamed)


def _build_te_summary_children(te_result: dict[str, Any]) -> html.Div:
    te_bits = te_result.get("te_bits")
    te_text = "Unavailable"
    if isinstance(te_bits, float) and math.isfinite(te_bits):
        te_text = f"{te_bits:.6f} bits"
    return html.Div(
        children=[
            html.Div(f"Value: {te_text}", className="status-line"),
            html.Div(str(te_result.get("message") or ""), className="status-line"),
            html.Div(f"Valid pairs: {int(te_result.get('valid_pairs', 0))}", className="status-line"),
            html.Div(f"Embedded samples: {int(te_result.get('n_samples_used', 0))}", className="status-line"),
            html.Div(f"Bins: {int(te_result.get('bins', 0))}", className="status-line"),
            html.Div(
                f"Source context: {te_result.get('source_context', {})}",
                className="status-line",
            ),
            html.Div(
                f"Receiver context: {te_result.get('receiver_context', {})}",
                className="status-line",
            ),
        ]
    )


def _build_matrix_status(matrix_result: dict[str, Any]) -> str:
    defaults = matrix_result.get("defaults", {})
    default_text = (
        f"Defaults: bins={defaults.get('bins')}, binning={defaults.get('binning')}, "
        f"history_grid={defaults.get('history_grid')}, tau_grid={defaults.get('tau_grid')}."
    )
    if matrix_result.get("available"):
        return f"{matrix_result.get('message')} {default_text}"
    return str(matrix_result.get("message") or default_text)


def _pairwise_te_record(signal_a: dict[str, Any], signal_b: dict[str, Any], te_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_run": signal_a["run_label"],
        "source_group": signal_a["group_name"],
        "source_trace": signal_a["trace_name"],
        "target_run": signal_b["run_label"],
        "target_group": signal_b["group_name"],
        "target_trace": signal_b["trace_name"],
        "available": bool(te_result.get("available")),
        "message": str(te_result.get("message") or ""),
        "te_bits": float(te_result["te_bits"]) if isinstance(te_result.get("te_bits"), float) and math.isfinite(te_result["te_bits"]) else math.nan,
        "valid_pairs": int(te_result.get("valid_pairs", 0)),
        "n_samples_used": int(te_result.get("n_samples_used", 0)),
        "bins": int(te_result.get("bins", 0)),
        "source_context": json.dumps(te_result.get("source_context", {}), sort_keys=True),
        "receiver_context": json.dumps(te_result.get("receiver_context", {}), sort_keys=True),
    }


def _signal_export_metadata(signal: dict[str, Any]) -> dict[str, Any]:
    return {
        "run": signal["run_label"],
        "group": signal["group_name"],
        "trace": signal["trace_name"],
    }


def _workbench_filename(signal_a: dict[str, Any], signal_b: dict[str, Any] | None, kind: str) -> str:
    left = _safe_token(signal_a["group_name"])
    if signal_b is None:
        return f"{left}_{_safe_token(kind)}.csv"
    right = _safe_token(signal_b["group_name"])
    return f"{left}_to_{right}_{_safe_token(kind)}.csv"


def _select_valid_dropdown_value(
    options: list[dict[str, Any]],
    current_value: str | None,
    default_value: str | None,
) -> str | None:
    values = [str(option["value"]) for option in options]
    if current_value is not None and str(current_value) in values:
        return str(current_value)
    if default_value is not None and str(default_value) in values:
        return str(default_value)
    return values[0] if values else None


def _format_bytes(size: float | int | None) -> str:
    if not size or size <= 0:
        return "unknown size"
    units = ("B", "KB", "MB", "GB")
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


def _run_file_stats(record: dict[str, Any]) -> str:
    """Cheap per-run stats for the metadata line.

    Deliberately ``stat()``-only: these callbacks fire on every selection
    change, and reading row count or sample rate would mean parsing a 5-16 MB
    CSV each time.
    """
    groups = len(record.get("available_groups") or [])
    traces = len(record.get("available_traces") or [])
    parts = [f"{groups} groups", f"{traces} traces"]
    try:
        stats = Path(record["mode_file_path"]).stat()
    except (OSError, KeyError, TypeError):
        parts.append("file stats unavailable")
        return " · ".join(parts)
    modified = datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M")
    parts.append(_format_bytes(stats.st_size))
    parts.append(f"modified {modified}")
    return " · ".join(parts)


def _is_unwrap_selected(toggle_value: list[str] | None) -> bool:
    """Read the phase-unwrap checklist, which reports a list of selected values."""
    return "unwrap" in (toggle_value or [])


def _coerce_positive_int(value: int | float | None, *, fallback: int, minimum: int) -> int:
    try:
        parsed = int(value) if value is not None else int(fallback)
    except (TypeError, ValueError):
        parsed = int(fallback)
    return max(int(minimum), parsed)


def _safe_token(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip())
    normalized = normalized.strip("_").lower()
    return normalized or "value"


def _status_div(message: str) -> html.Div:
    return html.Div(str(message), className="status-text")


def _folders_to_create(destination: Path) -> list[Path]:
    """Directories that do not exist yet on the way to ``destination``."""
    created: list[Path] = []
    probe = Path(destination).parent
    while not probe.exists() and probe != probe.parent:
        created.append(probe)
        probe = probe.parent
    return list(reversed(created))


def _hierarchy_hint(sample_data_dir: str | Path, csv_path: Path) -> str:
    """Warn when a destination will not produce browsable Subject/Terrain labels.

    ``catalog._build_hierarchy`` reads the dropdown labels from the components
    after ``Human_Motion``; a shallower layout still processes fine but shows
    up under generic labels.
    """
    try:
        relative = Path(csv_path).resolve().relative_to(Path(sample_data_dir).resolve())
    except (ValueError, OSError):
        return "Outside Sample_Data - outputs will use generic hierarchy labels."
    parts = relative.parts
    if not parts or parts[0] != catalog.HUMAN_MOTION_DIR:
        return (
            f"Under {parts[0] if parts else 'the root'}: outputs are browsable but use "
            "generic Subject/Terrain labels."
        )
    # Folder levels between "Human_Motion" and the file itself, so the two
    # known-good shapes are 4 (Subject/Terrain/Activity/Trial) and 3 (no Activity).
    depth = len(parts) - 2
    if depth >= 4:
        return "Full hierarchy: Subject / Terrain / Activity / Trial will all be labelled."
    if depth == 3:
        return "No activity level: Activity will show as 'Unspecified Activity'."
    return (
        f"Only {max(0, depth)} folder level(s) under {catalog.HUMAN_MOTION_DIR}; add "
        "Subject/Terrain/Activity/Trial for fully labelled outputs."
    )


def _resolve_output_root(output_target: str | None, config: dict[str, Any]) -> Path:
    """The run root a job writes into.

    The default is the newest existing run root, so a dashboard run lands beside
    the batch results instead of starting an isolated tree; ``rss_run_params.json``
    in each leaf records the parameters that produced it.
    """
    if output_target and output_target != layout.DEFAULT_OUTPUT_TARGET:
        return Path(output_target)
    results_dir = config.get("results_dir", "outputs")
    roots = raw_catalog.list_output_run_roots(
        results_dir, config.get("sample_data_dir", "Sample_Data")
    )
    if roots:
        return Path(roots[-1]["value"])
    return Path(results_dir) / "rss" / "Dashboard_Runs" / time.strftime("%Y%m%d_%H%M%S")


def _ingest_status_children(result: "raw_catalog.IngestResult") -> html.Div:
    lines = [html.Div(result.message, className="status-line")]
    validation = result.validation
    if validation.skipped_limbs:
        lines.append(
            html.Div(
                f"{len(validation.skipped_limbs)} limb(s) will be skipped: "
                + ", ".join(validation.skipped_limbs[:4])
                + ("..." if len(validation.skipped_limbs) > 4 else ""),
                className="status-line",
            )
        )
    for warning in validation.warnings:
        lines.append(html.Div(warning, className="status-line"))
    if result.created_dirs:
        lines.append(
            html.Div(f"Created {len(result.created_dirs)} new folder(s).", className="status-line")
        )
    if result.replaced:
        lines.append(html.Div("An existing file was replaced.", className="status-line"))
    lines.append(html.Div(result.destination, className="status-line"))
    return html.Div(lines)


def _status_pill(status: str) -> html.Span:
    return html.Span(status, className=f"status-pill status-pill-{status}")


def _progress_bar(fraction: float) -> html.Div:
    percent = max(0.0, min(1.0, float(fraction or 0.0))) * 100.0
    return html.Div(
        className="job-progress",
        children=html.Div(className="job-progress-fill", style={"width": f"{percent:.1f}%"}),
    )


def _job_card(row: dict[str, Any]) -> html.Div:
    progress = row.get("progress") or {}
    status = str(row.get("status", "unknown"))
    meta = [
        f"{jobs.clamp_epochs(row.get('epochs'))} epochs",
        f"SSA {row.get('ssa_window_length')}",
        str(row.get("compute_device", "")),
    ]
    if progress.get("elapsed_seconds"):
        meta.append(f"elapsed {jobs.format_duration(progress['elapsed_seconds'])}")
    if status == "running" and progress.get("eta_seconds") is not None:
        meta.append(f"eta {jobs.format_duration(progress['eta_seconds'])}")

    children = [
        html.Div(
            className="job-card-header",
            children=[
                html.Div(row.get("csv_label") or row.get("job_id", ""), className="job-title"),
                _status_pill(status),
            ],
        ),
        _progress_bar(progress.get("fraction", 0.0)),
        html.Div(str(progress.get("stage") or status), className="job-meta"),
        html.Div(" · ".join(part for part in meta if part), className="job-meta"),
    ]
    if row.get("message"):
        children.append(html.Div(str(row["message"]), className="job-meta"))
    if progress.get("error_hint"):
        children.append(html.Div(str(progress["error_hint"]), className="job-meta"))
    return html.Div(className="job-card", children=children)
