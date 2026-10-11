"""Layout builders for the HMA Dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dash import dcc, html

from HMA_Dashboard import catalog
from HMA_Dashboard import jobs
from HMA_Dashboard import plotting
from HMA_Dashboard import raw_catalog
from HMA_Dashboard import references as dashboard_references
from HMA_Dashboard import signal_logs
from HMA_Dashboard import te_matrix
from gait_analysis.references import load_registry


PLOT_KIND_OPTIONS = [
    {"label": "Time Series", "value": "time-series"},
    {"label": "FFT", "value": "fft"},
    {"label": "PDF", "value": "pdf"},
    {"label": "CDF", "value": "cdf"},
    {"label": "Phase Portrait", "value": "phase-portrait"},
    {"label": "Poincaré Transform", "value": "poincare-transform"},
    {"label": "Hilbert Analytic", "value": "hilbert-analytic"},
    {"label": "Inst. Phase", "value": "instantaneous-phase"},
    {"label": "Inst. Frequency", "value": "instantaneous-frequency"},
]
#: Poll cadence while a job is queued or running, and the idle heartbeat. The
#: interval is never disabled: a job started from another browser, or one still
#: queued, would otherwise stay invisible, and Popen.poll() would never run to
#: reap the finished child.
JOBS_POLL_ACTIVE_MS = 4000
JOBS_POLL_IDLE_MS = 30000
RUN_FILE_FILTER_OPTIONS = [
    {"label": "Unprocessed only", "value": "unprocessed"},
    {"label": "All raw files", "value": "all"},
]
COMPUTE_DEVICE_OPTIONS = [{"label": name, "value": name} for name in jobs.COMPUTE_DEVICE_OPTIONS]
FAULT_REASON_OPTIONS = [{"label": label, "value": code} for code, label in signal_logs.FAULT_REASONS]
#: The report dialog is one top-level overlay shared by every flip-row button, so
#: it is shown and hidden by swapping this style rather than by mounting it.
FAULT_PANEL_HIDDEN = {"display": "none"}
FAULT_PANEL_SHOWN = {"display": "flex"}
#: Sentinel for "use the most recent existing run root".
DEFAULT_OUTPUT_TARGET = "__default__"
GRAPH_CONFIG = {
    "displaylogo": False,
    "displayModeBar": True,
    "scrollZoom": True,
    "doubleClick": "reset+autosize",
    "responsive": True,
    # scale=2 makes the mode-bar PNG usable in a paper; lasso/box select do
    # nothing meaningful on these plots and only clutter the bar.
    "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "hma_dashboard_plot"},
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def build_layout(
    *,
    catalog_data: dict[str, Any],
    results_dir: Path,
    default_fs: float | None,
    raw_catalog_data: dict[str, Any] | None = None,
    run_settings: dict[str, Any] | None = None,
) -> html.Div:
    """Build the top-level dashboard layout."""
    raw_catalog_data = raw_catalog_data or {}
    settings = dict(run_settings or {})
    settings.setdefault("results_dir", str(results_dir))
    settings.setdefault("default_fs", default_fs)
    pipeline_enabled = bool(settings.get("pipeline_enabled", True))
    default_run = catalog.get_first_run_id(catalog_data)
    default_group_a = catalog.get_default_group(catalog_data, default_run, slot="a")
    default_group_b = catalog.get_default_group(catalog_data, default_run, slot="b")
    default_trace_a = catalog.get_default_trace(catalog_data, default_run, default_group_a, slot="a")
    default_trace_b = catalog.get_default_trace(catalog_data, default_run, default_group_b, slot="b")
    overview_run = default_run
    overview_defaults = catalog.get_hierarchy_defaults(catalog_data, overview_run)
    overview_group = catalog.get_default_group(catalog_data, overview_run)
    try:
        reference_registry = load_registry(dashboard_references.registry_path(results_dir))
    except (OSError, ValueError):
        reference_registry = {"schema_version": 1, "references": {}}

    return html.Div(
        className="app-shell",
        children=[
            dcc.Store(id="catalog-store", data=catalog_data),
            dcc.Store(id="dashboard-config-store", data=settings),
            # Outside dcc.Tabs on purpose: Dash only mounts the selected tab's
            # children, and a preview flip has to survive tab switches so the
            # same decision applies in the workbench and the overview alike.
            # "session" keeps it across an F5 mid-inspection but not across
            # browser restarts, where a stale invisible flip would mislead.
            dcc.Store(id="flip-preview-store", storage_type="session", data={}),
            dcc.Store(id="reference-registry-store", data=reference_registry),
            # Same reason these live out here: a selected window (and the arrow-key
            # nav that pages it) must survive a trip to the Overview tab and back.
            dcc.Store(id="workbench-selection-store", data=None),
            dcc.Store(id="workbench-markers-store", data=None),
            dcc.Store(id="arrow-key-event", data=None),
            # Same reason these live out here: the job poller has to keep
            # running while the user watches the Workbench, and Dash prunes any
            # callback whose components are not in the mounted tab.
            dcc.Store(id="raw-catalog-store", data=raw_catalog_data),
            dcc.Store(id="jobs-state-store", data=[]),
            dcc.Store(id="jobs-seen-store", data={}),
            dcc.Store(id="jobs-refresh-token", data=0),
            dcc.Interval(
                id="jobs-poll-interval",
                interval=JOBS_POLL_ACTIVE_MS,
                n_intervals=0,
                disabled=False,
                max_intervals=-1,
            ),
            html.Div(
                className="top-bar",
                children=[
                    html.Div(
                        children=[
                            html.H1("HMA Dashboard", className="page-title"),
                            html.Div(
                                id="catalog-status",
                                className="catalog-status",
                                children=catalog.get_catalog_summary(catalog_data),
                            ),
                            html.Div(id="flip-pending-summary", className="catalog-status"),
                            html.Div(id="flip-commit-status", className="matrix-status"),
                            html.Div(id="reference-action-status", className="matrix-status"),
                            html.Div(id="fault-report-status", className="matrix-status"),
                        ]
                    ),
                    html.Div(
                        className="top-bar-actions",
                        children=[
                            html.Button(
                                "Clear Preview Flips",
                                id="flip-clear-button",
                                className="download-button",
                                n_clicks=0,
                                title="Discard every unsaved sign-flip preview.",
                            ),
                            html.Button(
                                "Refresh File List", id="refresh-button", className="refresh-button", n_clicks=0
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Tabs(
                id="dashboard-tabs",
                value="workbench",
                className="dashboard-tabs",
                children=[
                    dcc.Tab(
                        label="Signal Workbench",
                        value="workbench",
                        className="dashboard-tab",
                        selected_className="dashboard-tab-selected",
                        children=[
                            html.Div(
                                className="workbench-shell",
                                children=[
                                    html.Div(
                                        className="workbench-control-grid",
                                        children=[
                                            _build_signal_selector_card(
                                                slot="a",
                                                title="Signal A",
                                                catalog_data=catalog_data,
                                                default_run=default_run,
                                                default_group=default_group_a,
                                                default_trace=default_trace_a,
                                            ),
                                            _build_signal_selector_card(
                                                slot="b",
                                                title="Signal B",
                                                catalog_data=catalog_data,
                                                default_run=default_run,
                                                default_group=default_group_b,
                                                default_trace=default_trace_b,
                                            ),
                                            _build_pairwise_settings_card(),
                                        ],
                                    ),
                                    html.Div(
                                        className="workbench-primary-plot",
                                        children=[
                                            _build_workbench_single_plot_card(),
                                        ],
                                    ),
                                    html.Div(
                                        className="chart-grid workbench-secondary-grid",
                                        children=[
                                            _build_graph_card("Joint PDF", "workbench-joint-pdf", "joint-pdf"),
                                            _build_pairwise_te_card(),
                                            _build_te_matrix_card(),
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Run Overview",
                        value="overview",
                        className="dashboard-tab",
                        selected_className="dashboard-tab-selected",
                        children=[
                            html.Div(
                                className="overview-shell",
                                children=[
                                    html.Div(
                                        className="controls-card overview-controls",
                                        children=[
                                            html.Div("Run Overview", className="section-title"),
                                            _labeled_dropdown(
                                                "Subject",
                                                "overview-subject-dropdown",
                                                catalog.get_hierarchy_options(catalog_data, "subject"),
                                                overview_defaults["subject"],
                                            ),
                                            _labeled_dropdown(
                                                "Terrain",
                                                "overview-terrain-dropdown",
                                                catalog.get_hierarchy_options(
                                                    catalog_data,
                                                    "terrain",
                                                    subject=overview_defaults["subject"],
                                                ),
                                                overview_defaults["terrain"],
                                            ),
                                            _labeled_dropdown(
                                                "Activity",
                                                "overview-activity-dropdown",
                                                catalog.get_hierarchy_options(
                                                    catalog_data,
                                                    "activity",
                                                    subject=overview_defaults["subject"],
                                                    terrain=overview_defaults["terrain"],
                                                ),
                                                overview_defaults["activity"],
                                            ),
                                            _labeled_dropdown(
                                                "Experiment",
                                                "overview-experiment-dropdown",
                                                catalog.get_hierarchy_options(
                                                    catalog_data,
                                                    "experiment",
                                                    subject=overview_defaults["subject"],
                                                    terrain=overview_defaults["terrain"],
                                                    activity=overview_defaults["activity"],
                                                ),
                                                overview_defaults["experiment"],
                                            ),
                                            _labeled_dropdown(
                                                "CSV File",
                                                "overview-run-dropdown",
                                                catalog.get_run_options(
                                                    catalog_data,
                                                    subject=overview_defaults["subject"],
                                                    terrain=overview_defaults["terrain"],
                                                    activity=overview_defaults["activity"],
                                                    experiment=overview_defaults["experiment"],
                                                ),
                                                overview_run,
                                            ),
                                            _labeled_dropdown(
                                                "Signal Group",
                                                "overview-group-dropdown",
                                                catalog.get_group_options(catalog_data, overview_run),
                                                overview_group,
                                            ),
                                            _labeled_number_input(
                                                "Distribution Bins",
                                                "overview-distribution-bins",
                                                value=40,
                                                minimum=2,
                                            ),
                                            _build_phase_unwrap_toggle("overview-phase-unwrap-toggle"),
                                            html.Div(
                                                "Click a trace in a plot legend to hide or show it; "
                                                "double-click to isolate it.",
                                                className="selection-metadata",
                                            ),
                                            html.Div(id="overview-selection-metadata", className="selection-metadata"),
                                        ],
                                    ),
                                    _build_overview_te_matrix_card(),
                                    html.Div(
                                        className="mode-stack",
                                        children=[_build_overview_mode_card(mode) for mode in (1, 2, 3)],
                                    ),
                                ],
                            )
                        ],
                    ),
                    *(
                        [
                            _build_pipeline_tab(
                                raw_catalog_data=raw_catalog_data,
                                catalog_data=catalog_data,
                                settings=settings,
                            )
                        ]
                        if pipeline_enabled
                        else []
                    ),
                ],
            ),
            _build_fault_report_dialog(),
        ],
    )


def _build_fault_report_dialog() -> html.Div:
    """Reason and note form for a faulty-signal report.

    Top level, outside ``dcc.Tabs``, for the reason the stores are: the buttons
    that open it live in five tab-scoped places, but there is only one form, and
    it has to survive a tab switch while the user is typing.
    """
    return html.Div(
        id="fault-report-backdrop",
        className="fault-report-backdrop",
        style=FAULT_PANEL_HIDDEN,
        children=[
            dcc.Store(id="fault-report-target-store", data=None),
            html.Div(
                className="fault-report-dialog",
                role="dialog",
                **{"aria-modal": "true", "aria-labelledby": "fault-report-title"},
                children=[
                    html.Div("Report faulty signal", id="fault-report-title", className="section-title"),
                    html.Div(id="fault-report-summary", className="selection-metadata"),
                    _labeled_dropdown(
                        "Reason",
                        "fault-report-reason",
                        FAULT_REASON_OPTIONS,
                        None,
                        help_text="Why this signal should not be trusted for gait analysis.",
                    ),
                    html.Div(
                        className="control-group",
                        children=[
                            _control_label("Note (required for 'Other')", "fault-report-note"),
                            dcc.Textarea(
                                id="fault-report-note",
                                className="number-input fault-report-note",
                                maxLength=signal_logs.NOTE_MAX_CHARS,
                                placeholder="Optional detail, e.g. the samples or steps affected.",
                                value="",
                            ),
                        ],
                    ),
                    html.Div(id="fault-report-error", className="status-line fault-report-error"),
                    html.Div(
                        className="fault-report-actions",
                        children=[
                            html.Button(
                                "Submit report", id="fault-report-submit", n_clicks=0,
                                className="download-button fault-report-button",
                            ),
                            html.Button(
                                "Cancel", id="fault-report-cancel", n_clicks=0, className="download-button",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def _build_signal_selector_card(
    *,
    slot: str,
    title: str,
    catalog_data: dict[str, Any],
    default_run: str | None,
    default_group: str | None,
    default_trace: str | None,
) -> html.Div:
    hierarchy_defaults = catalog.get_hierarchy_defaults(catalog_data, default_run)
    return html.Div(
        className="controls-card",
        children=[
            html.Div(title, className="section-title"),
            _labeled_dropdown(
                "Subject",
                {"type": "workbench-subject", "slot": slot},
                catalog.get_hierarchy_options(catalog_data, "subject"),
                hierarchy_defaults["subject"],
            ),
            _labeled_dropdown(
                "Terrain",
                {"type": "workbench-terrain", "slot": slot},
                catalog.get_hierarchy_options(
                    catalog_data,
                    "terrain",
                    subject=hierarchy_defaults["subject"],
                ),
                hierarchy_defaults["terrain"],
            ),
            _labeled_dropdown(
                "Activity",
                {"type": "workbench-activity", "slot": slot},
                catalog.get_hierarchy_options(
                    catalog_data,
                    "activity",
                    subject=hierarchy_defaults["subject"],
                    terrain=hierarchy_defaults["terrain"],
                ),
                hierarchy_defaults["activity"],
            ),
            _labeled_dropdown(
                "Experiment",
                {"type": "workbench-experiment", "slot": slot},
                catalog.get_hierarchy_options(
                    catalog_data,
                    "experiment",
                    subject=hierarchy_defaults["subject"],
                    terrain=hierarchy_defaults["terrain"],
                    activity=hierarchy_defaults["activity"],
                ),
                hierarchy_defaults["experiment"],
            ),
            _labeled_dropdown(
                "CSV File",
                {"type": "workbench-run", "slot": slot},
                catalog.get_run_options(
                    catalog_data,
                    subject=hierarchy_defaults["subject"],
                    terrain=hierarchy_defaults["terrain"],
                    activity=hierarchy_defaults["activity"],
                    experiment=hierarchy_defaults["experiment"],
                ),
                default_run,
            ),
            _labeled_dropdown(
                "Signal Group",
                {"type": "workbench-group", "slot": slot},
                catalog.get_group_options(catalog_data, default_run),
                default_group,
            ),
            _labeled_dropdown(
                "Trace",
                {"type": "workbench-trace", "slot": slot},
                catalog.get_trace_options(catalog_data, default_run, default_group),
                default_trace,
            ),
            html.Div(id={"type": "workbench-selection-metadata", "slot": slot}, className="selection-metadata"),
            _build_flip_controls(f"workbench-{slot}"),
        ],
    )


def _build_pairwise_settings_card() -> html.Div:
    return html.Div(
        className="controls-card",
        children=[
            html.Div("Pairwise Controls", className="section-title"),
            _labeled_number_input(
                "PDF / CDF Bins",
                "distribution-bins",
                value=40,
                minimum=2,
                help_text="Histogram bin count for the PDF and CDF plots.",
            ),
            _labeled_number_input(
                "Joint PDF / TE Bins",
                "pairwise-bins",
                value=8,
                minimum=2,
                help_text="Bin count per axis for the joint PDF and the pairwise "
                "transfer-entropy estimate.",
            ),
            _build_phase_unwrap_toggle("phase-unwrap-toggle"),
            # <details> keeps its children mounted, so the context Inputs below
            # keep firing while collapsed.
            html.Details(
                className="context-details",
                open=False,
                children=[
                    html.Summary("Context settings (lag / length / stride)", className="subsection-title"),
                    html.Div("Source Context", className="subsection-title"),
                    html.Div(
                        className="inline-number-grid",
                        children=[
                            _compact_number_input("Lag", "source-lag", value=1),
                            _compact_number_input("Length", "source-length", value=1),
                            _compact_number_input("Stride", "source-stride", value=1),
                        ],
                    ),
                    html.Div("Receiver Context", className="subsection-title"),
                    html.Div(
                        className="inline-number-grid",
                        children=[
                            _compact_number_input("Lag", "receiver-lag", value=1),
                            _compact_number_input("Length", "receiver-length", value=1),
                            _compact_number_input("Stride", "receiver-stride", value=1),
                        ],
                    ),
                ],
            ),
        ],
    )


def _build_graph_card(title: str, graph_id: str, download_kind: str) -> html.Div:
    return html.Div(
        className="chart-card",
        children=[
            html.Div(title, className="chart-title"),
            _loading_graph(
                id=graph_id,
                className="chart-graph",
                config=GRAPH_CONFIG,
                figure=plotting.make_message_figure(f"Loading {title.lower()}...", title=title),
                style={"height": "380px"},
            ),
            html.Div(
                className="download-row",
                children=[
                    html.Button(
                        "Download CSV",
                        id={"type": "workbench-download-button", "kind": download_kind},
                        n_clicks=0,
                        className="download-button",
                    ),
                    dcc.Download(id={"type": "workbench-download-target", "kind": download_kind}),
                ],
            ),
        ],
    )


def _build_workbench_single_plot_card() -> html.Div:
    return html.Div(
        className="chart-card chart-card-primary",
        children=[
            html.Div("Signal Plot", className="chart-title"),
            _labeled_dropdown("Plot Type", "workbench-plot-kind", PLOT_KIND_OPTIONS, "time-series"),
            html.Div(
                "Click Signal A or Signal B in the plot legend to hide or show it; "
                "double-click a signal to isolate it.",
                className="selection-metadata",
            ),
            html.Div(
                **{"data-selection-scope": "workbench"},
                children=[
                    _loading_graph(
                        id="workbench-single-plot",
                        className="chart-graph",
                        config=GRAPH_CONFIG,
                        figure=plotting.make_message_figure("Loading selected signals...", title="Signal Plot"),
                        style={"height": "640px"},
                    ),
                ],
            ),
            html.Div(
                className="download-row",
                children=[
                    html.Button(
                        "Download CSV",
                        id={"type": "workbench-download-button", "kind": "single-plot"},
                        n_clicks=0,
                        className="download-button",
                    ),
                    dcc.Download(id={"type": "workbench-download-target", "kind": "single-plot"}),
                ],
            ),
        ],
    )


def _build_pairwise_te_card() -> html.Div:
    return html.Div(
        className="chart-card",
        children=[
            html.Div("Pairwise Transfer Entropy", className="chart-title"),
            html.Div(
                id="workbench-pairwise-te-summary",
                className="te-summary",
                children=html.Div("Select two signals to compute transfer entropy.", className="status-text"),
            ),
            html.Div(
                className="download-row",
                children=[
                    html.Button(
                        "Download CSV",
                        id={"type": "workbench-download-button", "kind": "pairwise-te-csv"},
                        n_clicks=0,
                        className="download-button",
                    ),
                    html.Button(
                        "Download JSON",
                        id={"type": "workbench-download-button", "kind": "pairwise-te-json"},
                        n_clicks=0,
                        className="download-button",
                    ),
                    dcc.Download(id={"type": "workbench-download-target", "kind": "pairwise-te-csv"}),
                    dcc.Download(id={"type": "workbench-download-target", "kind": "pairwise-te-json"}),
                ],
            ),
        ],
    )


def _build_te_matrix_card() -> html.Div:
    return html.Div(
        className="chart-card chart-card-wide",
        children=[
            html.Div("Lower-Limb TE Matrix", className="chart-title"),
            html.Div(
                id="workbench-te-matrix-status",
                className="matrix-status",
                children="Select two same-run signals to enable the lower-limb TE matrix.",
            ),
            _loading_graph(
                id="workbench-te-matrix",
                className="chart-graph",
                config=GRAPH_CONFIG,
                figure=plotting.make_message_figure(
                    "Select two same-run signals to enable the lower-limb TE matrix.",
                    title="Lower-Limb TE Matrix",
                ),
                style={"height": "420px"},
            ),
            html.Div(
                className="download-row",
                children=[
                    html.Button(
                        "Download CSV",
                        id={"type": "workbench-download-button", "kind": "te-matrix"},
                        n_clicks=0,
                        className="download-button",
                    ),
                    dcc.Download(id={"type": "workbench-download-target", "kind": "te-matrix"}),
                ],
            ),
        ],
    )


def _build_overview_te_matrix_card() -> html.Div:
    """Whole-trial directed transfer entropy between every limb-angle signal.

    Behind an explicit Compute button, not the run dropdowns: the 18x18 matrix
    is 306 directed estimates (~1.5-3 s), and the overview selectors cascade, so
    an eager callback would stall the UI on every subject change.

    ``chart-card`` rather than ``chart-card-wide``: the latter relies on
    ``grid-column: 1 / -1`` and ``overview-shell`` is not a grid parent.
    """
    return html.Div(
        className="chart-card overview-te-card",
        children=[
            html.Div("Segment Transfer-Entropy Matrix", className="chart-title"),
            html.Div(
                "Directed transfer entropy between the mode-1 angle of every limb "
                "segment in the selected trial. Rows drive, columns receive.",
                className="selection-metadata",
            ),
            html.Div(
                className="inline-number-grid",
                children=[
                    _compact_number_input(
                        "Bins",
                        "overview-te-bins",
                        value=te_matrix.DEFAULT_BINS,
                        help_text="Uniform bins per signal after z-scoring.",
                    ),
                    _compact_number_input(
                        "Lag",
                        "overview-te-lag",
                        value=te_matrix.DEFAULT_LAG,
                        help_text="History lag in samples, applied to both the source "
                        "and the receiver. Length and stride are fixed at 1.",
                    ),
                ],
            ),
            dcc.Checklist(
                id="overview-te-show-values",
                options=[{"label": "Show values in cells", "value": "show"}],
                value=[],
                className="plot-kind-toggle",
                labelClassName="plot-kind-option",
            ),
            html.Div(
                className="download-row",
                children=[
                    html.Button(
                        "Compute TE Matrix",
                        id="overview-te-matrix-compute",
                        n_clicks=0,
                        className="refresh-button",
                    ),
                    html.Button(
                        "Download CSV",
                        id="overview-te-matrix-download-button",
                        n_clicks=0,
                        className="download-button",
                    ),
                    dcc.Download(id="overview-te-matrix-download"),
                ],
            ),
            html.Div(
                id="overview-te-matrix-status",
                className="matrix-status",
                children="Select a trial, then press Compute TE Matrix.",
            ),
            _loading_graph(
                id="overview-te-matrix",
                className="chart-graph",
                config=GRAPH_CONFIG,
                figure=plotting.make_message_figure(
                    "Select a trial, then press Compute TE Matrix.",
                    title="Segment TE Matrix",
                ),
                style={"height": "780px"},
            ),
        ],
    )


def _build_overview_mode_card(mode: int) -> html.Div:
    return html.Div(
        className="mode-card",
        children=[
            html.Div(f"Mode {mode}", className="mode-card-title"),
            dcc.Store(id={"type": "overview-selection-store", "mode": mode}, data=None),
            dcc.Store(id={"type": "overview-markers-store", "mode": mode}, data=None),
            dcc.RadioItems(
                id={"type": "overview-plot-kind", "mode": mode},
                options=PLOT_KIND_OPTIONS,
                value="time-series",
                inline=True,
                className="plot-kind-toggle",
                labelClassName="plot-kind-option",
            ),
            html.Div(
                **{"data-selection-scope": f"overview-{mode}"},
                children=[
                    _loading_graph(
                        id={"type": "overview-mode-graph", "mode": mode},
                        className="chart-graph",
                        config=GRAPH_CONFIG,
                        figure=plotting.make_message_figure(
                            "Select a run and signal group.", title=f"Mode {mode}"
                        ),
                        style={"height": "420px"},
                    ),
                ],
            ),
            html.Div(
                className="download-row",
                children=[
                    html.Button(
                        "Download CSV",
                        id={"type": "overview-download-button", "mode": mode},
                        n_clicks=0,
                        className="download-button",
                    ),
                    html.Button(
                        "Download All Groups CSV",
                        id={"type": "overview-download-all-button", "mode": mode},
                        n_clicks=0,
                        className="download-button",
                    ),
                    dcc.Download(id={"type": "overview-download-target", "mode": mode}),
                    dcc.Download(id={"type": "overview-download-all-target", "mode": mode}),
                ],
            ),
            _build_flip_controls(f"overview-{mode}"),
        ],
    )


#: One flip decision applies per (file, group), so these scopes name the
#: five places a signal can be selected, not five independent flip states.
FLIP_SCOPES: tuple[str, ...] = ("workbench-a", "workbench-b", "overview-1", "overview-2", "overview-3")


def flip_status_id(scope: str) -> str:
    return f"flip-status-{scope}"


def _build_flip_controls(scope: str) -> html.Div:
    """Preview / commit controls for the signal currently selected in ``scope``."""
    return html.Div(
        className="flip-row",
        children=[
            html.Button(
                "Invert sign (preview)",
                id={"type": "flip-preview-button", "scope": scope},
                n_clicks=0,
                className="download-button flip-preview-button",
                title="Negate this signal in every plot that shows it. Nothing is "
                "written to disk until you save.",
            ),
            html.Button(
                "Save flip to source CSVs",
                id={"type": "flip-commit-button", "scope": scope},
                n_clicks=0,
                className="download-button flip-commit-button",
                title="Write Modes 1-3 and their matching axis triplets into the real source CSVs as one transaction.",
            ),
            html.Button(
                "Set/Replace Reference Recording",
                id={"type": "reference-recording-button", "scope": scope},
                n_clicks=0,
                className="download-button",
            ),
            html.Button(
                "Check Orientation",
                id={"type": "orientation-check-button", "scope": scope},
                n_clicks=0,
                className="download-button",
            ),
            html.Button(
                "Set Reference Step",
                id={"type": "reference-step-button", "scope": scope},
                n_clicks=0,
                className="download-button",
                title="Use the raw zoomed range and snap both endpoints to nearby Mode 1 valleys.",
            ),
            html.Button(
                "Clear Reference",
                id={"type": "reference-clear-button", "scope": scope},
                n_clicks=0,
                className="download-button",
            ),
            html.Button(
                "Report faulty signal",
                id={"type": "fault-report-button", "scope": scope},
                n_clicks=0,
                className="download-button fault-report-button",
                title="Log this signal as faulty, with a reason, to the faulty-signal report file. "
                "Nothing in the run outputs changes.",
            ),
            html.Div(id=flip_status_id(scope), className="status-line flip-status"),
        ],
    )


def _loading_graph(**graph_kwargs: Any) -> dcc.Loading:
    """Wrap one graph in its own spinner overlay.

    Wrapped per graph rather than per tab so a slow card does not blank out its
    neighbours. ``overlay_style`` keeps the previous figure on screen, dimmed,
    and ``delay_show`` stops a sub-250 ms cached update from flashing a spinner.
    """
    return dcc.Loading(
        type="circle",
        delay_show=250,
        overlay_style={"visibility": "visible", "opacity": 0.55},
        children=dcc.Graph(**graph_kwargs),
    )


def _build_phase_unwrap_toggle(toggle_id: str) -> html.Div:
    """Unwrap switch for the instantaneous-phase plot.

    Lives outside ``PLOT_KIND_OPTIONS`` on purpose: wrapped and unwrapped phase
    are the same view under one option, exactly as ``distribution-bins`` is a
    kind-specific knob rather than four extra plot kinds.
    """
    return html.Div(
        className="control-group",
        title="Applies to the Inst. Phase plot only. Off shows the analytic signal's "
        "argument wrapped to (-pi, pi]; on shows the cumulative unwrapped phase.",
        children=[
            dcc.Checklist(
                id=toggle_id,
                options=[{"label": "Unwrap phase", "value": "unwrap"}],
                value=[],
                className="plot-kind-toggle",
                labelClassName="plot-kind-option",
            ),
        ],
    )


def _control_label(label: str, control_id: Any, *, extra_class: str = "") -> html.Label:
    """Label for one control, linked with ``htmlFor`` when the id allows it.

    Pattern-matching (dict) ids have no stable DOM id, so ``htmlFor`` is only
    set for plain string ids.
    """
    class_name = "control-label" + (f" {extra_class}" if extra_class else "")
    if isinstance(control_id, str):
        return html.Label(label, className=class_name, htmlFor=control_id)
    return html.Label(label, className=class_name)


def _labeled_dropdown(
    label: str,
    dropdown_id: Any,
    options: list[dict[str, Any]],
    value: str | None,
    *,
    help_text: str | None = None,
) -> html.Div:
    return html.Div(
        className="control-group",
        title=help_text,
        children=[
            _control_label(label, dropdown_id),
            dcc.Dropdown(id=dropdown_id, options=options, value=value, clearable=False),
        ],
    )


def _labeled_number_input(
    label: str,
    input_id: Any,
    *,
    value: int,
    minimum: int,
    help_text: str | None = None,
) -> html.Div:
    return html.Div(
        className="control-group",
        title=help_text,
        children=[
            _control_label(label, input_id),
            dcc.Input(
                id=input_id,
                type="number",
                min=minimum,
                step=1,
                value=value,
                debounce=True,
                className="number-input",
            ),
        ],
    )


def _compact_number_input(
    label: str,
    input_id: Any,
    *,
    value: int,
    help_text: str | None = None,
) -> html.Div:
    return html.Div(
        className="compact-number-input",
        title=help_text,
        children=[
            _control_label(label, input_id, extra_class="compact-label"),
            dcc.Input(
                id=input_id,
                type="number",
                min=1,
                step=1,
                value=value,
                debounce=True,
                className="number-input",
            ),
        ],
    )


def _build_pipeline_tab(
    *,
    raw_catalog_data: dict[str, Any],
    catalog_data: dict[str, Any],
    settings: dict[str, Any],
) -> dcc.Tab:
    """Ingest raw IMU files and run the RSS pipeline on them."""
    return dcc.Tab(
        label="Data & Runs",
        value="pipeline",
        className="dashboard-tab",
        selected_className="dashboard-tab-selected",
        children=[
            html.Div(
                className="pipeline-shell",
                children=[
                    _build_ingest_card(raw_catalog_data=raw_catalog_data, settings=settings),
                    _build_run_launcher_card(raw_catalog_data=raw_catalog_data, settings=settings),
                    _build_jobs_monitor_card(settings=settings),
                ],
            )
        ],
    )


def _build_ingest_card(*, raw_catalog_data: dict[str, Any], settings: dict[str, Any]) -> html.Div:
    """Upload a raw IMU CSV into the Sample_Data tree.

    The destination is a parent folder plus an optional relative sub-path rather
    than one dropdown per hierarchy level: the tree is four levels deep and the
    activity level is optional, so a fixed cascade would need a special case for
    every shape. This covers "drop it in an existing trial folder" and "create
    Subject/Terrain/Activity/Trial from scratch" with the same two controls.
    """
    max_upload_bytes = int(settings.get("max_upload_bytes", 32 * 1024 * 1024))
    return html.Div(
        className="controls-card ingest-card",
        children=[
            html.Div("Add Raw IMU File", className="section-title"),
            html.Div(
                id="ingest-catalog-status",
                className="catalog-status",
                children=raw_catalog.get_raw_catalog_summary(raw_catalog_data),
            ),
            dcc.Upload(
                id="ingest-upload",
                className="upload-drop",
                accept=".csv,text/csv",
                multiple=False,
                max_size=max_upload_bytes,
                children=html.Div(
                    [
                        "Drag a combined_log_wide_*.csv here, or ",
                        html.Span("browse", className="upload-link"),
                    ]
                ),
            ),
            _labeled_dropdown(
                "Destination Parent Folder",
                "ingest-parent-folder",
                raw_catalog.get_folder_options(raw_catalog_data),
                "",
                help_text="An existing folder under Sample_Data. Leave at the root to "
                "create a whole new branch below.",
            ),
            _labeled_text_input(
                "New Subfolders (optional)",
                "ingest-new-subpath",
                placeholder="Subject/Terrain/Activity/Trial",
                help_text="Relative path of folders to create inside the parent. "
                "Leave empty to upload straight into the parent folder.",
            ),
            _labeled_text_input(
                "Save As (optional)",
                "ingest-filename-override",
                placeholder="keep the uploaded file name",
                help_text="Must end in .csv.",
            ),
            dcc.Checklist(
                id="ingest-overwrite-toggle",
                options=[{"label": "Replace an existing file with the same name", "value": "replace"}],
                value=[],
                className="plot-kind-toggle",
                labelClassName="plot-kind-option",
            ),
            html.Div(id="ingest-path-preview", className="path-preview"),
            html.Div(
                className="inline-actions",
                children=[
                    html.Button(
                        "Rescan Sample_Data",
                        id="ingest-refresh-button",
                        className="download-button",
                        n_clicks=0,
                    )
                ],
            ),
            dcc.Loading(
                type="default",
                delay_show=250,
                children=html.Div(id="ingest-status", className="status-line"),
            ),
        ],
    )


def _build_run_launcher_card(*, raw_catalog_data: dict[str, Any], settings: dict[str, Any]) -> html.Div:
    """Start one RSS pipeline run on an unprocessed raw file."""
    file_options = raw_catalog.get_raw_file_options(raw_catalog_data, unprocessed_only=True)
    return html.Div(
        className="controls-card run-card",
        children=[
            html.Div("Run RSS Pipeline", className="section-title"),
            _labeled_dropdown(
                "Dataset",
                "run-dataset-dropdown",
                raw_catalog.get_dataset_options(raw_catalog_data),
                "all",
            ),
            dcc.RadioItems(
                id="run-file-filter",
                options=RUN_FILE_FILTER_OPTIONS,
                value="unprocessed",
                inline=True,
                className="plot-kind-toggle",
                labelClassName="plot-kind-option",
            ),
            _labeled_dropdown(
                "Raw CSV",
                "run-input-dropdown",
                file_options,
                file_options[0]["value"] if file_options else None,
            ),
            html.Div(id="run-input-metadata", className="selection-metadata"),
            html.Div("Preset Overrides", className="subsection-title"),
            _labeled_number_input(
                "Epochs",
                "run-epochs",
                value=jobs.DEFAULT_EPOCHS,
                minimum=jobs.EPOCHS_RANGE[0],
                help_text="2000 matches the published batch. Lower values finish sooner "
                "but are not comparable with it.",
            ),
            _labeled_number_input(
                "SSA Window Length",
                "run-ssa-window",
                value=jobs.DEFAULT_SSA_WINDOW_LENGTH,
                minimum=jobs.SSA_WINDOW_RANGE[0],
                help_text="SSA embedding window in samples; clamped to at most half the "
                "recording length by the pipeline.",
            ),
            _labeled_dropdown(
                "Compute Device",
                "run-compute-device",
                COMPUTE_DEVICE_OPTIONS,
                jobs.DEFAULT_COMPUTE_DEVICE,
                help_text="The reference batch ran on CPU; 'auto' picks CUDA when a GPU "
                "is visible, which changes the numerics.",
            ),
            _labeled_dropdown(
                "Output Run Root",
                "run-output-target",
                [{"label": "Most recent existing run root", "value": DEFAULT_OUTPUT_TARGET}],
                DEFAULT_OUTPUT_TARGET,
                help_text="Where the processed outputs are written.",
            ),
            html.Div(id="run-estimate", className="status-line"),
            dcc.Checklist(
                id="run-overwrite-confirm",
                options=[{"label": "Overwrite the existing outputs for this file", "value": "overwrite"}],
                value=[],
                className="plot-kind-toggle",
                labelClassName="plot-kind-option",
            ),
            html.Details(
                className="preset-details",
                open=False,
                children=[
                    html.Summary(jobs.PRESET_LABEL, className="subsection-title"),
                    html.Pre(" ".join(jobs.PRESET_FIXED_ARGS), className="preset-args"),
                ],
            ),
            html.Div(
                className="inline-actions",
                children=[
                    html.Button(
                        "Start RSS Run", id="run-start-button", className="refresh-button", n_clicks=0
                    )
                ],
            ),
            html.Div(id="run-status", className="status-line"),
        ],
    )


def _build_jobs_monitor_card(*, settings: dict[str, Any]) -> html.Div:
    """Live status, progress, and log tail for the RSS jobs."""
    return html.Div(
        className="chart-card jobs-card",
        children=[
            html.Div("Pipeline Jobs", className="chart-title"),
            html.Div(id="jobs-status", className="status-line"),
            html.Div(id="jobs-panel", className="job-grid"),
            _labeled_dropdown("Job Log", "jobs-log-selector", [], None),
            html.Div(
                className="inline-actions",
                children=[
                    html.Button(
                        "Cancel Selected Job",
                        id="jobs-cancel-button",
                        className="download-button danger-button",
                        n_clicks=0,
                    ),
                    html.Button(
                        "Refresh Jobs", id="jobs-refresh-button", className="download-button", n_clicks=0
                    ),
                    html.Button(
                        "Clear Finished (7d+)",
                        id="jobs-clear-button",
                        className="download-button",
                        n_clicks=0,
                    ),
                ],
            ),
            html.Pre(id="jobs-log-tail", className="job-log", children="No job selected."),
        ],
    )


def _labeled_text_input(
    label: str,
    input_id: Any,
    *,
    placeholder: str = "",
    value: str = "",
    help_text: str | None = None,
) -> html.Div:
    return html.Div(
        className="control-group",
        title=help_text,
        children=[
            _control_label(label, input_id),
            dcc.Input(
                id=input_id,
                type="text",
                value=value,
                placeholder=placeholder,
                debounce=True,
                className="number-input",
            ),
        ],
    )
