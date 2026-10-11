"""Scientific plotting helpers for rotating Poincare analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from rotating_poincare.core import RPTResult, compute_rpt_features


def plot_phase_portrait(
    phase_points: np.ndarray,
    interpolated_points: np.ndarray,
    signal_name: str,
) -> plt.Figure:
    """Plot centered/scaled samples and their arc-length interpolation."""

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot(
        interpolated_points[:, 0],
        interpolated_points[:, 1],
        color="tab:blue",
        linewidth=1.35,
        label="Interpolated trajectory",
    )
    ax.scatter(
        phase_points[:, 0],
        phase_points[:, 1],
        s=7,
        color="black",
        alpha=0.28,
        label="Phase samples",
    )
    ax.axhline(0.0, color="0.35", linewidth=0.7, alpha=0.5)
    ax.axvline(0.0, color="0.35", linewidth=0.7, alpha=0.5)
    ax.set_title(f"Phase Portrait: {signal_name}")
    ax.set_xlabel("Unit-range scaled signal")
    ax.set_ylabel("Unit-range scaled derivative")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_rpt_heatmap(result: RPTResult, signal_name: str) -> plt.Figure:
    """Plot section angle against radial crossing-bin count."""

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    image = ax.imshow(
        result.count_matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(
            float(result.radial_bin_edges[0]),
            float(result.radial_bin_edges[-1]),
            float(result.theta_values_deg[0]),
            float(result.theta_values_deg[-1]),
        ),
        cmap="viridis",
    )
    ax.set_title(f"Rotating Poincare Section Transform: {signal_name}")
    ax.set_xlabel("Radial coordinate r")
    ax.set_ylabel("Section angle theta (degrees)")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Intersection count")
    features = compute_rpt_features(
        result.count_matrix,
        result.theta_values_deg,
        result.radial_bin_centers,
    )
    valid = features["total_crossings"].to_numpy(dtype=int) > 0
    mean_r = features["mean_r"].to_numpy(dtype=float)
    mean_r[~valid] = np.nan
    mean_theta = features["theta_deg"].to_numpy(dtype=float)
    ax.plot(mean_r, mean_theta, color="white", linewidth=4.5, alpha=0.9)
    ax.plot(
        mean_r,
        mean_theta,
        color="black",
        linewidth=2.0,
        label="Mean radial trajectory",
    )
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_rpt_surface3d(result: RPTResult, signal_name: str) -> plt.Figure:
    """Create a Matplotlib 3D surface of the transform."""

    radial, theta = np.meshgrid(result.radial_bin_centers, result.theta_values_deg)
    fig = plt.figure(figsize=(10.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        radial,
        theta,
        result.count_matrix,
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
    )
    ax.set_title(f"Rotating Poincare Surface: {signal_name}")
    ax.set_xlabel("Radial coordinate r")
    ax.set_ylabel("Section angle theta (degrees)")
    ax.set_zlabel("Intersection count")
    fig.colorbar(surface, ax=ax, shrink=0.62, pad=0.1, label="Intersection count")
    fig.tight_layout()
    return fig


def plot_rpt_features(features: pd.DataFrame, signal_name: str) -> plt.Figure:
    """Plot section-wise crossing count, variance, and entropy."""

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=True)
    specifications = (
        ("total_crossings", "Total crossings", "tab:blue"),
        ("var_r", "Radial variance", "tab:orange"),
        ("entropy", "Radial entropy", "tab:green"),
    )
    for ax, (column, ylabel, color) in zip(axes, specifications, strict=True):
        ax.plot(features["theta_deg"], features[column], color=color, linewidth=1.5)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].set_title(f"RPT Section-wise Features: {signal_name}")
    axes[-1].set_xlabel("Section angle theta (degrees)")
    fig.tight_layout()
    return fig


def save_plotly_surface_html(
    result: RPTResult,
    signal_name: str,
    output_path: str | Path,
) -> None:
    """Write an interactive Plotly surface without loading remote assets."""

    figure = go.Figure(
        data=[
            go.Surface(
                x=result.radial_bin_centers,
                y=result.theta_values_deg,
                z=result.count_matrix,
                colorscale="Viridis",
                colorbar={"title": "Count"},
            )
        ]
    )
    figure.update_layout(
        title=f"Rotating Poincare Section Transform: {signal_name}",
        scene={
            "xaxis_title": "Radial coordinate r",
            "yaxis_title": "Section angle theta (degrees)",
            "zaxis_title": "Intersection count",
        },
    )
    figure.write_html(Path(output_path), include_plotlyjs=True, full_html=True)
