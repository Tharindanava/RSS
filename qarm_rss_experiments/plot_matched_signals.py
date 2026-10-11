"""Plot every matched RSS-vs-measured trial and bundle them into one PDF report.

Reads the ``matched/test_N.csv`` files written by
``qarm_rss_experiments.build_matched_rss_sources`` (65 trials across the two
recording dates), overlays the measured joint angle and its matched RSS
source for base, elbow, and wrist in one three-panel figure per trial, and
writes both per-trial PNGs and a combined PDF.

Points at ``QArm_Adam_NoSSA_Piecewise`` -- the hyperparameter search's
winning settings (adam, SSA preprocessing disabled, bias-window 1.0s, the
``piecewise_sparse_innovation`` adaptive regularizer; mean|r|=0.807 across
all 65 trials) -- which supersedes the original ``QArm_PSO_SO3_2000Epochs``
production run (mean|r|=0.638). See
``outputs/rss_hparam_search/`` and ``outputs/rss_hparam_search_round2/`` for
the full search that produced this recommendation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent.parent
RSS_SOURCES_ROOT = ROOT / "outputs" / "rss" / "QArm_Adam_NoSSA_Piecewise"
DATE_FOLDERS = ("session_2026-09-09", "session_2026-09-18")
JOINTS = ("base", "elbow", "wrist")
JOINT_COLORS = {"base": "#0072B2", "elbow": "#D55E00", "wrist": "#009E73"}
OUTPUT_DIR = RSS_SOURCES_ROOT / "matched_plots"
PDF_PATH = RSS_SOURCES_ROOT / "matched_signals_report.pdf"


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > np.finfo(float).eps else float("nan")


def _plot_trial(matched_path: Path, date_label: str, test_number: int) -> plt.Figure:
    df = pd.read_csv(matched_path)
    time = df["time_s"].to_numpy(float)
    figure, axes = plt.subplots(3, 1, figsize=(11.0, 8.4), sharex=True)
    for axis, joint in zip(axes, JOINTS):
        measured = df[f"measured_{joint}_rad"].to_numpy(float)
        source = df[f"source_{joint}"].to_numpy(float)
        r = _pearson(measured, source)
        axis.plot(time, measured, color="#202020", linewidth=1.6, label="Measured (ground truth)")
        axis.plot(time, source, color=JOINT_COLORS[joint], linewidth=1.2, alpha=0.9, label="RSS source")
        axis.set_title(f"{joint.title()}: r = {r:.3f}", loc="left", fontsize=11, fontweight="bold")
        axis.set_ylabel("Angle (rad)")
        axis.grid(True, alpha=0.2, linewidth=0.6)
        axis.legend(loc="upper right", frameon=False, ncol=2, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(f"{date_label} — Test {test_number}", fontsize=13, fontweight="bold")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return figure


def _collect_trials() -> list[tuple[str, int, Path]]:
    entries = []
    for date_folder in DATE_FOLDERS:
        matched_dir = RSS_SOURCES_ROOT / date_folder / "matched"
        paths = sorted(matched_dir.glob("test_*.csv"), key=lambda p: int(p.stem.split("_")[1]))
        for path in paths:
            entries.append((date_folder, int(path.stem.split("_")[1]), path))
    return entries


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = _collect_trials()
    if not entries:
        raise SystemExit(f"No matched trials found under {RSS_SOURCES_ROOT}")

    with PdfPages(PDF_PATH) as pdf:
        title_figure = plt.figure(figsize=(11.0, 8.4))
        title_figure.text(0.5, 0.60, "QArm RSS Matched Signals", ha="center", fontsize=24, fontweight="bold")
        title_figure.text(
            0.5, 0.52, f"{len(entries)} matched trials — base, elbow, wrist", ha="center", fontsize=13
        )
        title_figure.text(
            0.5,
            0.46,
            "Measured (ground truth) vs RSS-recovered source, overlaid per joint.\n"
            "Title of each panel gives the Pearson correlation r after sign correction.",
            ha="center",
            fontsize=10.5,
            color="#444444",
        )
        title_figure.text(
            0.5,
            0.38,
            "Settings: adam optimizer, SSA preprocessing disabled, bias-window 1.0s,\n"
            "piecewise_sparse_innovation adaptive regularizer (overall mean|r| = 0.807,\n"
            "vs. 0.638 for the original PSO + SSA + 2000-epoch production run).",
            ha="center",
            fontsize=9.5,
            color="#666666",
        )
        plt.axis("off")
        pdf.savefig(title_figure)
        plt.close(title_figure)

        for date_folder, test_number, path in entries:
            figure = _plot_trial(path, date_folder, test_number)
            png_name = f"{date_folder}_test_{test_number:02d}.png"
            figure.savefig(OUTPUT_DIR / png_name, dpi=150)
            pdf.savefig(figure)
            plt.close(figure)

    print(f"Wrote {len(entries)} per-trial plots to {OUTPUT_DIR}")
    print(f"Wrote combined PDF report to {PDF_PATH}")


if __name__ == "__main__":
    main()
