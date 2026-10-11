"""Plot construction helpers for the HMA Dashboard."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import hilbert

from hilbert_analysis.analytic import instantaneous_frequency, instantaneous_phase
from hilbert_analysis.signal_utils import find_sustained_runs, interpolate_small_gaps
from HMA_Dashboard import te_matrix
from HMA_Dashboard.transfer_entropy import JOINT_ORDER
from rotating_poincare.core import (
    RPTResult,
    center_and_normalize_phase_points,
    compute_derivative,
    compute_rotating_poincare_transform,
    compute_rpt_features,
    interpolate_trajectory,
    make_phase_points,
)


PLOT_TEMPLATE = "plotly_white"
SIGNAL_TOGGLE_LEGEND = {
    "orientation": "h",
    "x": 1.0,
    "xanchor": "right",
    "y": 1.02,
    "yanchor": "bottom",
    "itemclick": "toggle",
    "itemdoubleclick": "toggleothers",
    "groupclick": "togglegroup",
}


def make_message_figure(message: str, title: str = "Unavailable") -> go.Figure:
    """Return a simple placeholder figure with a centered message."""
    figure = go.Figure()
    figure.add_annotation(
        text=str(message),
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15},
    )
    figure.update_layout(
        template=PLOT_TEMPLATE,
        title=_title_dict(title),
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
    )
    return figure


def build_time_series_frame(signal_a: dict[str, Any], signal_b: dict[str, Any] | None = None) -> pd.DataFrame:
    """Build export data for time-series plotting."""
    return _merge_named_frames(
        [_single_signal_frame(signal, "signal_value") for signal in _present_signals(signal_a, signal_b)]
    )


def build_fft_frame(signal_a: dict[str, Any], signal_b: dict[str, Any] | None = None) -> pd.DataFrame:
    """Build export data for FFT plotting."""
    frames: list[pd.DataFrame] = []
    for signal in _present_signals(signal_a, signal_b):
        frequencies, magnitudes, x_label = _compute_fft(signal)
        frames.append(
            pd.DataFrame(
                {
                    f"{signal['series_key']}_{x_label}": frequencies,
                    f"{signal['series_key']}_magnitude": magnitudes,
                }
            )
        )
    return _merge_named_frames(frames)


def build_pdf_frame(signal_a: dict[str, Any], signal_b: dict[str, Any] | None = None, *, bins: int) -> pd.DataFrame:
    """Build export data for 1D PDF plotting."""
    densities = _compute_distribution_frames(signal_a, signal_b, bins=bins)
    return densities["pdf"]


def build_cdf_frame(signal_a: dict[str, Any], signal_b: dict[str, Any] | None = None, *, bins: int) -> pd.DataFrame:
    """Build export data for 1D CDF plotting."""
    densities = _compute_distribution_frames(signal_a, signal_b, bins=bins)
    return densities["cdf"]


def build_phase_portrait_frame(signal_a: dict[str, Any], signal_b: dict[str, Any] | None = None) -> pd.DataFrame:
    """Build export data for phase-portrait plotting."""
    frames: list[pd.DataFrame] = []
    for signal in _present_signals(signal_a, signal_b):
        signal_value, derivative, _valid = _compute_phase_portrait(signal)
        frames.append(
            pd.DataFrame(
                {
                    f"{signal['series_key']}_signal_value": signal_value,
                    f"{signal['series_key']}_derivative": derivative,
                }
            )
        )
    return _merge_named_frames(frames)


def build_hilbert_frame(signal_a: dict[str, Any], signal_b: dict[str, Any] | None = None) -> pd.DataFrame:
    """Build export data for Hilbert-analytic plotting."""
    frames: list[pd.DataFrame] = []
    for signal in _present_signals(signal_a, signal_b):
        signal_value, imaginary, _valid = _compute_hilbert(signal)
        frames.append(
            pd.DataFrame(
                {
                    f"{signal['series_key']}_signal_value": signal_value,
                    f"{signal['series_key']}_hilbert_imag": imaginary,
                }
            )
        )
    return _merge_named_frames(frames)


def build_instantaneous_phase_frame(
    signal_a: dict[str, Any],
    signal_b: dict[str, Any] | None = None,
    *,
    unwrap: bool = False,
) -> pd.DataFrame:
    """Build export data for instantaneous-phase plotting."""
    frames: list[pd.DataFrame] = []
    for signal in _present_signals(signal_a, signal_b):
        x_values, phase, field, _used, _total, _start = _compute_instantaneous_phase(signal, unwrap=unwrap)
        if x_values.size == 0:
            continue
        frames.append(
            pd.DataFrame(
                {
                    f"{signal['series_key']}_x": x_values,
                    f"{signal['series_key']}_{field}": phase,
                }
            )
        )
    return _merge_named_frames(frames)


def build_instantaneous_frequency_frame(
    signal_a: dict[str, Any],
    signal_b: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build export data for instantaneous-frequency plotting."""
    frames: list[pd.DataFrame] = []
    for signal in _present_signals(signal_a, signal_b):
        x_values, frequency, field, _unit, _used, _total, _start = _compute_instantaneous_frequency(signal)
        if x_values.size == 0:
            continue
        frames.append(
            pd.DataFrame(
                {
                    f"{signal['series_key']}_x": x_values,
                    f"{signal['series_key']}_{field}": frequency,
                }
            )
        )
    return _merge_named_frames(frames)


def build_poincare_transform_frame(
    signal_a: dict[str, Any],
    signal_b: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build flattened positive-ray Poincaré-transform data on a shared grid."""
    transforms = _compute_poincare_transforms(signal_a, signal_b)
    if not transforms:
        return pd.DataFrame(columns=["theta_deg", "radial_coordinate"])

    first_result = transforms[0][1]
    theta_grid, radial_grid = np.meshgrid(
        first_result.theta_values_deg,
        first_result.radial_bin_centers,
        indexing="ij",
    )
    data: dict[str, Any] = {
        "theta_deg": theta_grid.reshape(-1),
        "radial_coordinate": radial_grid.reshape(-1),
    }
    for signal, result in transforms:
        data[f"{signal['series_key']}_intersection_count"] = result.count_matrix.reshape(-1)
    return pd.DataFrame(data)


def build_joint_pdf_frame(signal_a: dict[str, Any], signal_b: dict[str, Any], *, bins: int) -> pd.DataFrame:
    """Build export data for 2D joint-PDF plotting."""
    x_values = np.asarray(signal_a["values"], dtype=float).reshape(-1)
    y_values = np.asarray(signal_b["values"], dtype=float).reshape(-1)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[mask]
    y_values = y_values[mask]
    if x_values.size == 0:
        return pd.DataFrame(columns=["x_bin_center", "y_bin_center", "density", "count"])

    x_edges = _build_bin_edges([x_values], bins=bins)
    y_edges = _build_bin_edges([y_values], bins=bins)
    counts, x_edges, y_edges = np.histogram2d(x_values, y_values, bins=(x_edges, y_edges), density=False)
    widths_x = np.diff(x_edges)
    widths_y = np.diff(y_edges)
    total = float(np.sum(counts))
    rows: list[dict[str, float]] = []
    for row_index in range(counts.shape[0]):
        for col_index in range(counts.shape[1]):
            width = float(widths_x[row_index] * widths_y[col_index])
            count = float(counts[row_index, col_index])
            density = 0.0 if total <= 0.0 or width <= 0.0 else count / (total * width)
            rows.append(
                {
                    "x_bin_center": float((x_edges[row_index] + x_edges[row_index + 1]) / 2.0),
                    "y_bin_center": float((y_edges[col_index] + y_edges[col_index + 1]) / 2.0),
                    "density": density,
                    "count": count,
                }
            )
    return pd.DataFrame(rows)


def build_te_matrix_frame(matrix_result: dict[str, Any]) -> pd.DataFrame:
    """Build export data for one or more TE matrices."""
    rows: list[dict[str, Any]] = []
    for side_record in matrix_result.get("results", []):
        matrix = np.asarray(side_record["matrix"], dtype=float)
        for row_index, source_joint in enumerate(JOINT_ORDER):
            for col_index, target_joint in enumerate(JOINT_ORDER):
                rows.append(
                    {
                        "side": side_record["side"],
                        "source_joint": source_joint,
                        "target_joint": target_joint,
                        "te_raw": float(matrix[row_index, col_index]) if np.isfinite(matrix[row_index, col_index]) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _signal_value_label(signal_a: dict[str, Any], signal_b: dict[str, Any] | None = None) -> str:
    signals = (signal_a,) if signal_b is None else (signal_a, signal_b)
    if any(str(signal.get("trace_name", "")).startswith("mode_") for signal in signals):
        return "RSS latent-mode proxy value"
    return "Angle"


def build_overlay_time_series_figure(signal_a: dict[str, Any], signal_b: dict[str, Any] | None = None) -> go.Figure:
    """Build an overlaid time-series figure."""
    figure = go.Figure()
    anchor: tuple[dict[str, Any], np.ndarray, np.ndarray] | None = None
    for signal in _present_signals(signal_a, signal_b):
        values = np.asarray(signal["values"], dtype=float).reshape(-1)
        x_values = np.asarray(signal["time"] if signal["time"] is not None else signal["sample_index"], dtype=float)
        valid = np.isfinite(values) & np.isfinite(x_values)
        figure.add_trace(
            go.Scatter(
                x=x_values[valid],
                y=values[valid],
                mode="lines",
                name=signal["legend_name"],
                legendgroup=signal["series_key"],
                showlegend=True,
                line={"width": 2},
            )
        )
        if anchor is None:
            anchor = (signal, x_values[valid], np.flatnonzero(valid))
    figure.update_layout(
        template=PLOT_TEMPLATE,
        title=_title_dict("Time Series"),
        xaxis_title=str(signal_a["x_label"]),
        yaxis_title=_signal_value_label(signal_a, signal_b),
        hovermode="x unified",
        dragmode="pan",
        legend=SIGNAL_TOGGLE_LEGEND,
        margin={"l": 60, "r": 20, "t": 75, "b": 60},
    )
    figure.update_xaxes(rangeslider={"visible": True})
    if anchor is not None:
        _add_marker_overlays(figure, *anchor)
    return figure


def build_overlay_fft_figure(signal_a: dict[str, Any], signal_b: dict[str, Any] | None = None) -> go.Figure:
    """Build an overlaid FFT figure."""
    figure = go.Figure()
    x_label = "Frequency"
    for signal in _present_signals(signal_a, signal_b):
        frequencies, magnitudes, x_label = _compute_fft(signal)
        if frequencies.size == 0:
            continue
        figure.add_trace(
            go.Scatter(
                x=frequencies,
                y=magnitudes,
                mode="lines",
                name=signal["legend_name"],
                legendgroup=signal["series_key"],
                showlegend=True,
                line={"width": 2},
            )
        )
    if not figure.data:
        return make_message_figure("Not enough samples for FFT.", title="FFT")
    figure.update_layout(
        template=PLOT_TEMPLATE,
        title=_title_dict("FFT"),
        xaxis_title=x_label,
        yaxis_title="Magnitude",
        hovermode="x unified",
        dragmode="pan",
        legend=SIGNAL_TOGGLE_LEGEND,
        margin={"l": 60, "r": 20, "t": 75, "b": 60},
    )
    return figure


def build_overlay_pdf_figure(signal_a: dict[str, Any], signal_b: dict[str, Any] | None = None, *, bins: int) -> go.Figure:
    """Build an overlaid PDF figure."""
    pdf_frame = build_pdf_frame(signal_a, signal_b, bins=bins)
    if pdf_frame.empty:
        return make_message_figure("No finite samples were available for PDF plotting.", title="PDF")
    figure = go.Figure()
    for signal in _present_signals(signal_a, signal_b):
        figure.add_trace(
            go.Scatter(
                x=pdf_frame["bin_center"],
                y=pdf_frame.get(f"{signal['series_key']}_density"),
                mode="lines+markers",
                name=signal["legend_name"],
                legendgroup=signal["series_key"],
                showlegend=True,
                line={"width": 2},
            )
        )
    figure.update_layout(
        template=PLOT_TEMPLATE,
        title=_title_dict("PDF"),
        xaxis_title=_signal_value_label(signal_a, signal_b),
        yaxis_title="Density",
        hovermode="x unified",
        dragmode="pan",
        legend=SIGNAL_TOGGLE_LEGEND,
        margin={"l": 60, "r": 20, "t": 75, "b": 60},
    )
    return figure


def build_overlay_cdf_figure(signal_a: dict[str, Any], signal_b: dict[str, Any] | None = None, *, bins: int) -> go.Figure:
    """Build an overlaid CDF figure."""
    cdf_frame = build_cdf_frame(signal_a, signal_b, bins=bins)
    if cdf_frame.empty:
        return make_message_figure("No finite samples were available for CDF plotting.", title="CDF")
    figure = go.Figure()
    for signal in _present_signals(signal_a, signal_b):
        figure.add_trace(
            go.Scatter(
                x=cdf_frame["bin_center"],
                y=cdf_frame.get(f"{signal['series_key']}_cdf"),
                mode="lines+markers",
                name=signal["legend_name"],
                legendgroup=signal["series_key"],
                showlegend=True,
                line={"width": 2},
            )
        )
    figure.update_layout(
        template=PLOT_TEMPLATE,
        title=_title_dict("CDF"),
        xaxis_title=_signal_value_label(signal_a, signal_b),
        yaxis_title="Cumulative Probability",
        hovermode="x unified",
        dragmode="pan",
        legend=SIGNAL_TOGGLE_LEGEND,
        margin={"l": 60, "r": 20, "t": 75, "b": 60},
    )
    return figure


def build_overlay_phase_portrait_figure(signal_a: dict[str, Any], signal_b: dict[str, Any] | None = None) -> go.Figure:
    """Build an overlaid phase-portrait figure."""
    figure = go.Figure()
    anchor: tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray] | None = None
    for signal in _present_signals(signal_a, signal_b):
        signal_value, derivative, valid = _compute_phase_portrait(signal)
        if signal_value.size == 0:
            continue
        figure.add_trace(
            go.Scatter(
                x=signal_value,
                y=derivative,
                mode="lines",
                name=signal["legend_name"],
                legendgroup=signal["series_key"],
                showlegend=True,
                line={"width": 2},
            )
        )
        if anchor is None:
            anchor = (signal, signal_value, derivative, valid)
    if not figure.data:
        return make_message_figure("Not enough samples for a phase portrait.", title="Phase Portrait")
    if anchor is not None:
        _add_marker_scatter_overlay(figure, *anchor)
    figure.update_layout(
        template=PLOT_TEMPLATE,
        title=_title_dict("Phase Portrait"),
        xaxis_title="Signal Value",
        yaxis_title="Derivative",
        hovermode="closest",
        dragmode="pan",
        legend=SIGNAL_TOGGLE_LEGEND,
        margin={"l": 60, "r": 20, "t": 75, "b": 60},
    )
    return figure


def build_overlay_hilbert_figure(signal_a: dict[str, Any], signal_b: dict[str, Any] | None = None) -> go.Figure:
    """Build an overlaid Hilbert-analytic figure."""
    figure = go.Figure()
    anchor: tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray] | None = None
    for signal in _present_signals(signal_a, signal_b):
        signal_value, imaginary, valid = _compute_hilbert(signal)
        if signal_value.size == 0:
            continue
        figure.add_trace(
            go.Scatter(
                x=signal_value,
                y=imaginary,
                mode="lines",
                name=signal["legend_name"],
                legendgroup=signal["series_key"],
                showlegend=True,
                line={"width": 2},
            )
        )
        if anchor is None:
            anchor = (signal, signal_value, imaginary, valid)
    if not figure.data:
        return make_message_figure("Not enough samples for a Hilbert analytic plot.", title="Hilbert Analytic")
    if anchor is not None:
        _add_marker_scatter_overlay(figure, *anchor)
    figure.update_layout(
        template=PLOT_TEMPLATE,
        title=_title_dict("Hilbert Analytic"),
        xaxis_title="Signal Value",
        yaxis_title="Hilbert Transform",
        hovermode="closest",
        dragmode="pan",
        legend=SIGNAL_TOGGLE_LEGEND,
        margin={"l": 60, "r": 20, "t": 75, "b": 60},
    )
    return figure


PHASE_TICK_VALUES = [-math.pi, -math.pi / 2.0, 0.0, math.pi / 2.0, math.pi]
PHASE_TICK_TEXT = ["-π", "-π/2", "0", "π/2", "π"]


def build_overlay_instantaneous_phase_figure(
    signal_a: dict[str, Any],
    signal_b: dict[str, Any] | None = None,
    *,
    unwrap: bool = False,
) -> go.Figure:
    """Build an overlaid instantaneous-phase figure (the analytic signal's argument)."""
    figure = go.Figure()
    coverage: list[tuple[int, int]] = []
    anchor: tuple[dict[str, Any], np.ndarray, np.ndarray] | None = None
    for signal in _present_signals(signal_a, signal_b):
        x_values, phase, _field, used, total, start = _compute_instantaneous_phase(signal, unwrap=unwrap)
        if x_values.size == 0:
            continue
        coverage.append((used, total))
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=phase,
                mode="lines",
                name=signal["legend_name"],
                legendgroup=signal["series_key"],
                showlegend=True,
                line={"width": 2},
            )
        )
        if anchor is None:
            anchor = (signal, x_values, np.arange(start, start + x_values.size))
    if not figure.data:
        return make_message_figure(
            "Not enough samples for an instantaneous-phase plot.",
            title="Instantaneous Phase",
        )
    figure.update_layout(
        template=PLOT_TEMPLATE,
        title=_title_dict("Instantaneous Phase" + _coverage_suffix(coverage)),
        xaxis_title=str(signal_a["x_label"]),
        yaxis_title="Unwrapped Phase (rad)" if unwrap else "Phase (rad)",
        hovermode="x unified",
        dragmode="pan",
        legend=SIGNAL_TOGGLE_LEGEND,
        margin={"l": 60, "r": 20, "t": 75, "b": 60},
    )
    if not unwrap:
        figure.update_yaxes(tickvals=PHASE_TICK_VALUES, ticktext=PHASE_TICK_TEXT)
    figure.update_xaxes(rangeslider={"visible": True})
    if anchor is not None:
        _add_marker_overlays(figure, *anchor)
    return figure


def build_overlay_instantaneous_frequency_figure(
    signal_a: dict[str, Any],
    signal_b: dict[str, Any] | None = None,
) -> go.Figure:
    """Build an overlaid instantaneous-frequency figure (d/dt of the unwrapped phase)."""
    figure = go.Figure()
    coverage: list[tuple[int, int]] = []
    unit_label = "Frequency (Hz)"
    per_sample = False
    anchor: tuple[dict[str, Any], np.ndarray, np.ndarray] | None = None
    for signal in _present_signals(signal_a, signal_b):
        x_values, frequency, field, unit, used, total, start = _compute_instantaneous_frequency(signal)
        if x_values.size == 0:
            continue
        coverage.append((used, total))
        unit_label = unit
        per_sample = per_sample or field.endswith("cycles_per_sample")
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=frequency,
                mode="lines",
                name=signal["legend_name"],
                legendgroup=signal["series_key"],
                showlegend=True,
                line={"width": 2},
            )
        )
        if anchor is None:
            # x_run[1:] was used for x_values, so the absolute sample index of
            # x_values[0] is start + 1, not start.
            anchor = (signal, x_values, np.arange(start + 1, start + 1 + x_values.size))
    if not figure.data:
        return make_message_figure(
            "Not enough samples for an instantaneous-frequency plot.",
            title="Instantaneous Frequency",
        )
    figure.update_layout(
        template=PLOT_TEMPLATE,
        title=_title_dict("Instantaneous Frequency" + _coverage_suffix(coverage)),
        xaxis_title=str(signal_a["x_label"]),
        yaxis_title=unit_label,
        hovermode="x unified",
        dragmode="pan",
        legend=SIGNAL_TOGGLE_LEGEND,
        margin={"l": 60, "r": 20, "t": 75, "b": 60},
    )
    if per_sample:
        figure.add_annotation(
            text="No usable sampling rate: frequency is per sample, not in Hz.",
            x=0.0,
            y=1.0,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            font={"size": 12},
        )
    figure.update_xaxes(rangeslider={"visible": True})
    if anchor is not None:
        _add_marker_overlays(figure, *anchor)
    return figure


def build_overlay_poincare_transform_figure(
    signal_a: dict[str, Any],
    signal_b: dict[str, Any] | None = None,
) -> go.Figure:
    """Overlay positive-ray Poincaré-transform heatmaps for selected signals."""
    transforms = _compute_poincare_transforms(signal_a, signal_b)
    if not transforms:
        return make_message_figure(
            "Not enough samples for a Poincaré transform.",
            title="Poincaré Transform",
        )

    color_scales = (
        "Viridis",
        [
            [0.0, "rgba(13,8,135,0.0)"],
            [1e-6, "rgba(13,8,135,0.55)"],
            [0.25, "rgba(126,3,168,0.65)"],
            [0.5, "rgba(204,71,120,0.72)"],
            [0.75, "rgba(248,149,64,0.82)"],
            [1.0, "rgba(240,249,33,0.95)"],
        ],
    )
    mean_line_colors = ("#22D3EE", "#FACC15")
    figure = go.Figure()
    for index, (signal, result) in enumerate(transforms):
        maximum_count = int(np.max(result.count_matrix)) if result.count_matrix.size else 0
        colorbar_x = 1.02 + 0.12 * index
        figure.add_trace(
            go.Heatmap(
                x=result.radial_bin_centers,
                y=result.theta_values_deg,
                z=result.count_matrix,
                zmin=0,
                zmax=max(1, maximum_count),
                name=signal["legend_name"],
                legendgroup=signal["series_key"],
                showlegend=True,
                colorscale=color_scales[index],
                zsmooth=False,
                xgap=0,
                ygap=0,
                hoverongaps=False,
                colorbar={
                    "title": {
                        "text": f"Signal {'A' if index == 0 else 'B'}<br>intersection count",
                        "font": {"size": 15, "color": "black"},
                    },
                    "x": colorbar_x,
                    "len": 0.88,
                    "thickness": 20,
                    "outlinecolor": "black",
                    "outlinewidth": 1,
                    "tickfont": {"size": 13, "color": "black"},
                },
                hovertemplate=(
                    "%{fullData.name}<br>"
                    "θ=%{y:.1f}°<br>"
                    "r=%{x:.5g}<br>"
                    "count=%{z}<extra></extra>"
                ),
            )
        )
    for index, (signal, result) in enumerate(transforms):
        features = compute_rpt_features(
            result.count_matrix,
            result.theta_values_deg,
            result.radial_bin_centers,
        )
        valid = features["total_crossings"].to_numpy(dtype=int) > 0
        mean_r = features["mean_r"].to_numpy(dtype=float)
        mean_r[~valid] = np.nan
        figure.add_trace(
            go.Scatter(
                x=mean_r,
                y=features["theta_deg"],
                mode="lines",
                name=f"{signal['legend_name']} mean trajectory",
                legendgroup=signal["series_key"],
                showlegend=False,
                line={
                    "color": mean_line_colors[index],
                    "width": 3.5,
                    "shape": "linear",
                },
                connectgaps=False,
                hovertemplate=(
                    "%{fullData.name}<br>"
                    "θ=%{y:.1f}°<br>"
                    "mean r=%{x:.5g}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        template="plotly_white",
        title={
            "text": "Positive-Ray Poincaré Transform",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 24, "color": "black"},
        },
        xaxis={
            "title": {"text": "Radial coordinate r", "font": {"size": 19, "color": "black"}},
            "tickfont": {"size": 15, "color": "black"},
            "showgrid": False,
            "zeroline": False,
            "showline": True,
            "mirror": True,
            "linecolor": "black",
            "linewidth": 1.5,
            "ticks": "outside",
            "ticklen": 6,
            "tickcolor": "black",
        },
        yaxis={
            "title": {
                "text": "Section angle θ (degrees)",
                "font": {"size": 19, "color": "black"},
            },
            "tickfont": {"size": 15, "color": "black"},
            "showgrid": False,
            "zeroline": False,
            "showline": True,
            "mirror": True,
            "linecolor": "black",
            "linewidth": 1.5,
            "ticks": "outside",
            "ticklen": 6,
            "tickcolor": "black",
        },
        paper_bgcolor="white",
        plot_bgcolor="#440154",
        font={"family": "Arial, sans-serif", "color": "black"},
        hovermode="closest",
        dragmode="pan",
        legend=SIGNAL_TOGGLE_LEGEND,
        margin={"l": 90, "r": 205 if len(transforms) > 1 else 125, "t": 95, "b": 85},
    )
    return figure


def build_joint_pdf_figure(signal_a: dict[str, Any], signal_b: dict[str, Any], *, bins: int) -> go.Figure:
    """Build a 2D joint-PDF heatmap."""
    joint_frame = build_joint_pdf_frame(signal_a, signal_b, bins=bins)
    if joint_frame.empty:
        return make_message_figure("No finite pairs were available for joint-PDF plotting.", title="Joint PDF")

    pivot = joint_frame.pivot(index="y_bin_center", columns="x_bin_center", values="density").sort_index()
    figure = go.Figure(
        data=[
            go.Heatmap(
                x=pivot.columns.to_numpy(dtype=float),
                y=pivot.index.to_numpy(dtype=float),
                z=pivot.to_numpy(dtype=float),
                colorbar={"title": "Density"},
                colorscale="Viridis",
            )
        ]
    )
    figure.update_layout(
        template=PLOT_TEMPLATE,
        title=_title_dict("Joint PDF"),
        xaxis_title=signal_a["legend_name"],
        yaxis_title=signal_b["legend_name"],
        margin={"l": 70, "r": 20, "t": 75, "b": 80},
    )
    return figure


def build_te_matrix_figure(matrix_result: dict[str, Any]) -> go.Figure:
    """Build a figure for one or more TE matrices."""
    results = list(matrix_result.get("results", []))
    if not results:
        return make_message_figure(str(matrix_result.get("message") or "No TE matrix available."), title="Lower-Limb TE Matrix")

    figure = make_subplots(
        rows=1,
        cols=len(results),
        subplot_titles=[record["side"] for record in results],
        horizontal_spacing=0.12,
    )
    zmin = min(
        (
            float(np.nanmin(np.asarray(record["matrix"], dtype=float)))
            for record in results
            if np.isfinite(np.asarray(record["matrix"], dtype=float)).any()
        ),
        default=0.0,
    )
    zmax = max(
        (
            float(np.nanmax(np.asarray(record["matrix"], dtype=float)))
            for record in results
            if np.isfinite(np.asarray(record["matrix"], dtype=float)).any()
        ),
        default=1.0,
    )
    for column_index, record in enumerate(results, start=1):
        matrix = np.asarray(record["matrix"], dtype=float)
        figure.add_trace(
            go.Heatmap(
                x=list(JOINT_ORDER),
                y=list(JOINT_ORDER),
                z=matrix,
                zmin=zmin,
                zmax=zmax,
                colorscale="RdYlGn",
                colorbar={"title": "Raw TE"} if column_index == len(results) else None,
                showscale=column_index == len(results),
            ),
            row=1,
            col=column_index,
        )
    figure.update_layout(
        template=PLOT_TEMPLATE,
        title=_title_dict("Lower-Limb TE Matrix"),
        margin={"l": 50, "r": 20, "t": 85, "b": 50},
    )
    return figure


def compact_segment_label(name: str) -> str:
    """``left_hip_to_knee`` -> ``Left Hip -> Knee``.

    About 20% narrower than the plain pretty label, which matters when 18 of
    them have to fit along both axes of a heatmap.
    """
    return str(name).replace("_to_", "_→_").replace("_", " ").strip().title()


def build_all_segment_te_matrix_figure(
    matrix_result: dict[str, Any],
    *,
    show_values: bool = False,
) -> go.Figure:
    """Directed transfer-entropy heatmap over every segment in one trial."""
    if not matrix_result.get("available"):
        return make_message_figure(
            matrix_result.get("message") or "Segment TE matrix is unavailable.",
            title="Segment TE Matrix",
        )

    segments = list(matrix_result["segments"])
    labels = [compact_segment_label(segment) for segment in segments]
    values = np.array(
        [[np.nan if cell is None else float(cell) for cell in row] for row in matrix_result["matrix"]],
        dtype=float,
    )
    finite = values[np.isfinite(values)]
    # Sequential, not diverging: transfer entropy is a non-negative magnitude,
    # so a red-to-green scale would imply a good/bad reading it does not have.
    heatmap = go.Heatmap(
        z=values,
        x=labels,
        y=labels,
        colorscale="Viridis",
        zmin=float(min(0.0, finite.min())) if finite.size else 0.0,
        zmax=float(finite.max()) if finite.size else 1.0,
        xgap=1,
        ygap=1,
        hoverongaps=False,
        colorbar={"title": {"text": "TE (bits)"}},
        hovertemplate="source %{y}<br>target %{x}<br>TE %{z:.4f} bits<extra></extra>",
    )
    if show_values:
        heatmap.update(texttemplate="%{z:.2f}", textfont={"size": 9})

    figure = go.Figure(data=[heatmap])
    figure.update_layout(
        template=PLOT_TEMPLATE,
        title=_title_dict(
            f"Segment TE Matrix ({matrix_result['trace_name']}, {len(segments)} segments)"
        ),
        # Explicit category axes: the region boxes below are placed by index,
        # which only means "the nth category" on a category axis.
        xaxis={"title": "Target (receiver)", "tickangle": -45, "side": "bottom", "type": "category"},
        # Reversed so the matrix reads like a table, top-left to bottom-right.
        yaxis={"title": "Source (driver)", "autorange": "reversed", "type": "category"},
        margin={"l": 190, "r": 160, "t": 125, "b": 190},
        # The NaN diagonal shows through as this colour, reading as
        # "not applicable" rather than as a zero-TE cell.
        plot_bgcolor="#f4f8fb",
        dragmode="pan",
    )
    _add_region_boxes(figure, segments)
    return figure


#: Display names for ``te_matrix.body_region`` values.
REGION_LABELS = {"upper": "Upper limbs", "lower": "Lower limbs"}
REGION_BOX_COLOR = "#1f2933"


def _region_runs(segments: Sequence[str]) -> list[tuple[str, int, int]]:
    """Contiguous ``(region, first, last)`` runs of classified segments.

    An unclassified segment (a joint-family group, say) ends the current run, so
    a box never spans one.
    """
    runs: list[list[Any]] = []
    for index, segment in enumerate(segments):
        region = te_matrix.body_region(segment)
        if region is None:
            continue
        if runs and runs[-1][0] == region and runs[-1][2] == index - 1:
            runs[-1][2] = index
        else:
            runs.append([region, index, index])
    return [(region, first, last) for region, first, last in runs]


def _add_region_boxes(figure: go.Figure, segments: Sequence[str]) -> None:
    """Box the matrix into upper/lower source-by-target sections.

    Rows are sources and columns are targets, so the four boxes are upper->upper,
    upper->lower, lower->upper and lower->lower. Nothing is drawn unless both
    regions are present: with one region there is no split to show.
    """
    runs = _region_runs(segments)
    if {region for region, _, _ in runs} != set(REGION_LABELS):
        return
    for source_region, row_first, row_last in runs:
        for target_region, column_first, column_last in runs:
            figure.add_shape(
                type="rect",
                xref="x", yref="y",
                x0=column_first - 0.5, x1=column_last + 0.5,
                y0=row_first - 0.5, y1=row_last + 0.5,
                line={"color": REGION_BOX_COLOR, "width": 2.5},
                fillcolor="rgba(0,0,0,0)",
                layer="above",
                name=f"{source_region}_to_{target_region}",
            )
    # Region names sit outside the plot area, where no cell is covered: targets
    # across the top (columns), sources down the right-hand side (rows).
    for region, first, last in runs:
        centre = (first + last) / 2
        figure.add_annotation(
            text=f"Target: {REGION_LABELS[region]}", x=centre, y=1.0, xref="x", yref="paper",
            yanchor="bottom", yshift=6, showarrow=False, font={"size": 12, "color": REGION_BOX_COLOR},
        )
        figure.add_annotation(
            text=f"Source: {REGION_LABELS[region]}", x=1.0, y=centre, xref="paper", yref="y",
            xanchor="left", xshift=8, textangle=90, showarrow=False,
            font={"size": 12, "color": REGION_BOX_COLOR},
        )


def build_all_segment_te_matrix_frame(matrix_result: dict[str, Any]) -> pd.DataFrame:
    """Tidy source/target rows for the segment TE matrix export."""
    rows = te_matrix.matrix_rows(matrix_result)
    if not rows:
        return pd.DataFrame(columns=["source_segment", "target_segment", "te_bits"])
    return pd.DataFrame(rows)


def _present_signals(signal_a: dict[str, Any], signal_b: dict[str, Any] | None) -> list[dict[str, Any]]:
    signals = [signal_a]
    if signal_b is not None:
        signals.append(signal_b)
    return signals


#: Above this many step-boundary ticks, skip drawing them individually (a long
#: recording can have hundreds of strides, and one add_vline per tick bloats
#: the figure JSON enough to slow down the browser).
MAX_STEP_MARKER_TICKS = 500


def _add_marker_overlays(
    figure: go.Figure,
    signal: dict[str, Any],
    x_values: np.ndarray,
    sample_indices: np.ndarray,
) -> None:
    """Draw gait-event lines and accepted-step shading on a time-indexed plot.

    ``sample_indices`` maps each position in ``x_values`` back to the full signal.
    """
    if "hybrid_segmentation_available" in signal and not bool(signal.get("hybrid_segmentation_available")):
        figure.add_annotation(
            text="hybrid segmentation unavailable", x=0.01, y=0.99, xref="paper", yref="paper",
            xanchor="left", yanchor="top", showarrow=False,
            font={"size": 11, "color": "#6b7280"}, bgcolor="rgba(255,255,255,0.8)",
        )
        return
    segment_label = signal.get("segment_label")
    step_index = signal.get("step_index")
    if segment_label is None and step_index is None:
        return
    if x_values.size == 0 or sample_indices.size == 0:
        return

    index_positions = {int(sample): position for position, sample in enumerate(sample_indices)}
    events = [event for event in (signal.get("gait_events") or []) if bool(event.get("accepted", False))]
    if len(events) <= MAX_STEP_MARKER_TICKS:
        for event in events:
            position = index_positions.get(int(event.get("sample", -1)))
            if position is None:
                continue
            event_type = str(event.get("event_type", ""))
            figure.add_vline(
                x=float(x_values[position]), line_width=1.2,
                line_dash={"IC": "solid", "TO": "dash", "MS": "dot"}.get(event_type, "dot"),
                line_color="#2563eb" if str(event.get("side")) == "left" else "#dc2626",
                opacity=0.65,
            )
    for step in signal.get("gait_steps") or []:
        start_position = index_positions.get(int(step.get("start_sample", -1)))
        end_position = index_positions.get(int(step.get("end_sample", -1)))
        if start_position is not None and end_position is not None:
            figure.add_vrect(
                x0=float(x_values[start_position]), x1=float(x_values[end_position]),
                fillcolor="rgba(37,99,235,0.025)" if step.get("side") == "left" else "rgba(220,38,38,0.025)",
                line_width=0, layer="below",
            )
    confidences = [float(event.get("confidence", np.nan)) for event in events]
    confidence_text = f" · median confidence {np.nanmedian(confidences):.2f}" if confidences else ""
    figure.add_annotation(
        text=(
            f"{signal.get('hybrid_segmentation_message', 'hybrid segmentation')}{confidence_text}"
            "<br>blue=left, red=right · IC solid, TO dashed, MS dotted"
        ),
        x=0.01, y=0.99, xref="paper", yref="paper", xanchor="left", yanchor="top",
        showarrow=False, font={"size": 11, "color": "#334155"}, bgcolor="rgba(255,255,255,0.8)",
    )


def _add_marker_scatter_overlay(
    figure: go.Figure,
    signal: dict[str, Any],
    x_values: np.ndarray,
    y_values: np.ndarray,
    valid: np.ndarray,
) -> None:
    """Scatter gait-event and step-boundary markers onto a value-indexed plot (phase portrait, Hilbert).

    ``valid`` is the boolean mask the caller applied to filter its raw sample
    array down to ``x_values``/``y_values``; it is used to project
    ``gait_events``/``step_index`` (full-length arrays/lists) into the same
    filtered, positionally-aligned index space.
    """
    if "hybrid_segmentation_available" in signal and not bool(signal.get("hybrid_segmentation_available")):
        return
    segment_label = signal.get("segment_label")
    step_index = signal.get("step_index")
    if segment_label is None and step_index is None:
        return
    if x_values.size == 0:
        return
    sample_indices = np.flatnonzero(valid)

    event_groups = {
        (side, event_type): []
        for side in ("left", "right") for event_type in ("IC", "TO", "MS")
    }
    positions = {int(sample): position for position, sample in enumerate(sample_indices)}
    for event in signal.get("gait_events") or []:
        if not bool(event.get("accepted", False)):
            continue
        position = positions.get(int(event.get("sample", -1)))
        key = (str(event.get("side")), str(event.get("event_type")))
        if position is not None and key in event_groups:
            event_groups[key].append(position)
    for (side, event_type), rows in event_groups.items():
        if rows:
            figure.add_trace(go.Scatter(
                x=x_values[rows], y=y_values[rows], mode="markers",
                marker={
                    "symbol": {"IC": "circle", "TO": "triangle-up", "MS": "diamond"}[event_type],
                    "size": 7, "color": "#2563eb" if side == "left" else "#dc2626",
                },
                name=f"{side} {event_type}", legendgroup=f"{signal['series_key']}_hybrid_events",
                showlegend=True,
            ))

    if step_index is not None:
        steps = np.asarray(step_index)[sample_indices]
        boundaries = np.flatnonzero(np.diff(steps) != 0) + 1
        if boundaries.size:
            figure.add_trace(
                go.Scatter(
                    x=x_values[boundaries],
                    y=y_values[boundaries],
                    mode="markers",
                    marker={"symbol": "x", "size": 7, "color": "rgba(0,0,0,0.75)"},
                    name=f"step boundary ({signal['legend_name']})",
                    legendgroup=f"{signal['series_key']}_markers",
                    showlegend=True,
                )
            )


def _single_signal_frame(signal: dict[str, Any], value_label: str) -> pd.DataFrame:
    values = np.asarray(signal["values"], dtype=float).reshape(-1)
    x_values = np.asarray(signal["time"] if signal["time"] is not None else signal["sample_index"], dtype=float)
    return pd.DataFrame(
        {
            f"{signal['series_key']}_x": x_values,
            f"{signal['series_key']}_{value_label}": values,
        }
    )


def _merge_named_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    max_rows = max(len(frame) for frame in frames)
    aligned = [frame.reindex(range(max_rows)) for frame in frames]
    return pd.concat(aligned, axis=1)


def _compute_fft(signal: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    values = np.asarray(signal["values"], dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float), "Frequency"
    demeaned = values - float(np.mean(values))
    window = np.hanning(demeaned.size) if demeaned.size >= 8 else np.ones(demeaned.size, dtype=float)
    spectrum = np.fft.rfft(demeaned * window)
    magnitudes = np.abs(spectrum)
    if magnitudes.size <= 1:
        return np.array([], dtype=float), np.array([], dtype=float), "Frequency"
    magnitudes = magnitudes[1:]
    if signal.get("fs") is not None:
        frequencies = np.fft.rfftfreq(demeaned.size, d=1.0 / float(signal["fs"]))[1:]
        return frequencies, magnitudes, "frequency_hz"
    frequencies = np.fft.rfftfreq(demeaned.size, d=1.0)[1:]
    return frequencies, magnitudes, "frequency_cycles_per_sample"


def _compute_distribution_frames(signal_a: dict[str, Any], signal_b: dict[str, Any] | None, *, bins: int) -> dict[str, pd.DataFrame]:
    signals = _present_signals(signal_a, signal_b)
    values_list = [np.asarray(signal["values"], dtype=float).reshape(-1) for signal in signals]
    finite_values_list = [values[np.isfinite(values)] for values in values_list]
    finite_values_list = [values for values in finite_values_list if values.size > 0]
    if not finite_values_list:
        empty = pd.DataFrame(columns=["bin_left", "bin_right", "bin_center"])
        return {"pdf": empty, "cdf": empty}

    edges = _build_bin_edges(finite_values_list, bins=bins)
    pdf_data: dict[str, Any] = {
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "bin_center": (edges[:-1] + edges[1:]) / 2.0,
    }
    cdf_data: dict[str, Any] = {
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "bin_center": (edges[:-1] + edges[1:]) / 2.0,
    }
    widths = np.diff(edges)
    for signal in signals:
        values = np.asarray(signal["values"], dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        counts, _ = np.histogram(values, bins=edges, density=False)
        total = float(np.sum(counts))
        density = np.zeros_like(widths, dtype=float)
        cdf = np.zeros_like(widths, dtype=float)
        if total > 0.0:
            density = counts.astype(float) / (total * widths)
            cdf = np.cumsum(counts.astype(float)) / total
        pdf_data[f"{signal['series_key']}_density"] = density
        pdf_data[f"{signal['series_key']}_count"] = counts.astype(float)
        cdf_data[f"{signal['series_key']}_cdf"] = cdf
        cdf_data[f"{signal['series_key']}_count"] = counts.astype(float)
    return {
        "pdf": pd.DataFrame(pdf_data),
        "cdf": pd.DataFrame(cdf_data),
    }


def _build_bin_edges(values_list: list[np.ndarray], *, bins: int) -> np.ndarray:
    merged = np.concatenate(values_list)
    minimum = float(np.min(merged))
    maximum = float(np.max(merged))
    effective_bins = max(2, int(bins))
    if math.isclose(minimum, maximum):
        delta = max(1e-6, abs(minimum) * 0.05 + 1e-6)
        return np.linspace(minimum - delta, maximum + delta, effective_bins + 1)
    return np.linspace(minimum, maximum, effective_bins + 1)


def _compute_phase_portrait(signal: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(centered_value, derivative, valid)``; ``valid`` lets callers project
    ``segment_label``/``step_index`` into the same filtered index space for markers."""
    values = np.asarray(signal["values"], dtype=float).reshape(-1)
    x_values = np.asarray(signal["time"] if signal["time"] is not None else signal["sample_index"], dtype=float)
    valid = np.isfinite(values) & np.isfinite(x_values)
    values = values[valid]
    x_values = x_values[valid]
    if values.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float), valid
    centered = values - float(np.mean(values))
    derivative = np.gradient(centered, x_values)
    return centered, derivative, valid


def _compute_poincare_transforms(
    signal_a: dict[str, Any],
    signal_b: dict[str, Any] | None,
) -> list[tuple[dict[str, Any], RPTResult]]:
    signals = _present_signals(signal_a, signal_b)
    trajectories: list[tuple[dict[str, Any], np.ndarray]] = []
    for signal in signals:
        values = np.asarray(signal["values"], dtype=float).reshape(-1)
        x_values = np.asarray(
            signal["time"] if signal["time"] is not None else signal["sample_index"],
            dtype=float,
        )
        valid = np.isfinite(values) & np.isfinite(x_values)
        values = values[valid]
        x_values = x_values[valid]
        if values.size < 2 or np.any(np.diff(x_values) <= 0.0):
            continue
        derivative = compute_derivative(values, t=x_values)
        raw_phase = make_phase_points(values, derivative)
        normalized_phase = center_and_normalize_phase_points(raw_phase, normalize=True)
        interpolated = interpolate_trajectory(normalized_phase.points)
        trajectories.append((signal, interpolated))

    if not trajectories:
        return []

    shared_radius = max(
        float(np.max(np.linalg.norm(points, axis=1)))
        for _, points in trajectories
    )
    shared_radius = max(shared_radius, 1e-12)
    return [
        (
            signal,
            compute_rotating_poincare_transform(
                points,
                radial_min=0.0,
                radial_max=shared_radius,
            ),
        )
        for signal, points in trajectories
    ]


def _compute_hilbert(signal: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(centered_value, hilbert_imag, valid)``; ``valid`` lets callers project
    ``segment_label``/``step_index`` into the same filtered index space for markers."""
    values = np.asarray(signal["values"], dtype=float).reshape(-1)
    valid = np.isfinite(values)
    values = values[valid]
    if values.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float), valid
    centered = values - float(np.mean(values))
    analytic = hilbert(centered)
    return centered, np.imag(analytic), valid


def _prepare_analytic_run(signal: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Longest contiguous finite ``(x, values)`` run, small interior gaps repaired.

    ``hilbert_analysis.analytic`` deliberately raises on non-finite input rather
    than dropping samples, so callers must decide what to do about gaps. Short
    interior runs are interpolated (``interpolate_small_gaps``, the same
    treatment ``phase_flip.evaluate_pair`` applies); anything left over is
    handled by *truncating to the longest finite run* rather than compacting the
    array. Compacting would splice across the gap, and because the Hilbert
    kernel decays only as ``1/t`` that discontinuity would smear across the
    entire record instead of staying local.

    Returns ``(x_run, values_run, samples_used, samples_total, start)``, where
    ``start`` is the run's offset into the original (untruncated) sample array,
    for callers that need to project marker columns into the same index space.
    """
    empty = np.array([], dtype=float)
    values = np.asarray(signal["values"], dtype=float).reshape(-1)
    x_source = signal["time"] if signal.get("time") is not None else signal["sample_index"]
    x_values = np.asarray(x_source, dtype=float).reshape(-1)
    total = int(min(values.size, x_values.size))
    if total == 0:
        return empty, empty, 0, 0, 0

    repaired = interpolate_small_gaps(values[:total])
    x_values = x_values[:total]
    finite = np.isfinite(repaired) & np.isfinite(x_values)
    runs = find_sustained_runs(finite, 1)
    if not runs:
        return empty, empty, 0, total, 0

    start, end = max(runs, key=lambda run: run[1] - run[0])
    return x_values[start : end + 1], repaired[start : end + 1], end + 1 - start, total, start


def _coverage_suffix(coverage: list[tuple[int, int]]) -> str:
    """Title suffix naming the sample coverage when any signal was truncated."""
    truncated = [(used, total) for used, total in coverage if total and used < total]
    if not truncated:
        return ""
    used, total = min(truncated, key=lambda item: item[0])
    return f" ({used}/{total} samples)"


def _compute_instantaneous_phase(
    signal: dict[str, Any], *, unwrap: bool
) -> tuple[np.ndarray, np.ndarray, str, int, int, int]:
    x_run, values_run, used, total, start = _prepare_analytic_run(signal)
    if values_run.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float), "instantaneous_phase", used, total, start
    phase = instantaneous_phase(values_run, unwrap=unwrap)
    field = "instantaneous_phase_unwrapped" if unwrap else "instantaneous_phase"
    return x_run, phase, field, used, total, start


def _compute_instantaneous_frequency(
    signal: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, str, str, int, int, int]:
    """Instantaneous frequency against ``x[1:]``.

    ``np.diff`` of the unwrapped phase is a backward difference, so each value
    belongs to the *later* sample of the pair; attributing it to ``x[1:]`` (rather
    than to bin midpoints) also keeps the export joinable on ``x_value`` with the
    time-series and phase exports.
    """
    empty = np.array([], dtype=float)
    x_run, values_run, used, total, start = _prepare_analytic_run(signal)
    if values_run.size < 3:
        return empty, empty, "instantaneous_frequency_hz", "Frequency (Hz)", used, total, start

    fs = signal.get("fs")
    usable_fs = fs is not None and np.isfinite(float(fs)) and float(fs) > 0.0
    if usable_fs:
        frequency = instantaneous_frequency(values_run, float(fs))
        return x_run[1:], frequency, "instantaneous_frequency_hz", "Frequency (Hz)", used, total, start

    phase = instantaneous_phase(values_run, unwrap=True)
    frequency = np.diff(phase) / (2.0 * math.pi)
    return (
        x_run[1:],
        frequency,
        "instantaneous_frequency_cycles_per_sample",
        "Frequency (cycles/sample)",
        used,
        total,
        start,
    )


def _title_dict(title_text: str) -> dict[str, Any]:
    return {
        "text": title_text,
        "x": 0.01,
        "xanchor": "left",
        "font": {"size": 16},
    }
