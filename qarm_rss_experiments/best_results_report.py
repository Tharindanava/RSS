"""Rank all 65 matched QArm trials and build a report of the 30 best.

Reuses the matched-trial plotting from ``plot_matched_signals.py`` (same
``QArm_Adam_NoSSA_Piecewise`` source, same three-panel overlay style),
but ranks every trial by its per-trial mean |Pearson r| across base, elbow,
and wrist, keeps the top 30, and writes a summary ranking table, box-plot
comparisons against PCA/JADE-ICA/FastICA/SOBI baselines (see
``best30_baseline_comparison.py``), and the 30 corresponding plots into one
PDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from qarm_rss_experiments.best30_baseline_comparison import (
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_ORDER,
    compute_comparison_table,
)
from qarm_rss_experiments.plot_matched_signals import (
    JOINTS,
    RSS_SOURCES_ROOT,
    _collect_trials,
    _pearson,
    _plot_trial,
)

TOP_N = 30
OUTPUT_DIR = RSS_SOURCES_ROOT / "best_30_plots"
PDF_PATH = RSS_SOURCES_ROOT / "best_30_signals_report.pdf"
TABLE_CSV_PATH = RSS_SOURCES_ROOT / "best_30_ranking.csv"
COMPARISON_CSV_PATH = RSS_SOURCES_ROOT / "best_30_baseline_comparison.csv"


def _score_trial(path: Path) -> dict:
    frame = pd.read_csv(path)
    row = {}
    for joint in JOINTS:
        measured = frame[f"measured_{joint}_rad"].to_numpy(float)
        source = frame[f"source_{joint}"].to_numpy(float)
        row[joint] = abs(_pearson(measured, source))
    row["mean_r"] = float(np.mean([row[j] for j in JOINTS]))
    return row


def _rank_all_trials() -> pd.DataFrame:
    entries = _collect_trials()
    rows = []
    for date_folder, test_number, path in entries:
        score = _score_trial(path)
        rows.append({"session": date_folder, "test_number": test_number, "path": path, **score})
    table = pd.DataFrame(rows)
    return table.sort_values("mean_r", ascending=False).reset_index(drop=True)


def _draw_ranking_table(top: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(11.0, 12.0))
    axis.axis("off")
    columns = ["Rank", "Session", "Test", "Base r", "Elbow r", "Wrist r", "Mean r"]
    cell_text = []
    for rank, row in enumerate(top.itertuples(index=False), 1):
        session_label = "2026-09-09" if row.session.endswith("2026-09-09") else "2026-09-18"
        cell_text.append(
            [
                str(rank),
                session_label,
                str(row.test_number),
                f"{row.base:.3f}",
                f"{row.elbow:.3f}",
                f"{row.wrist:.3f}",
                f"{row.mean_r:.3f}",
            ]
        )
    table = axis.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for col in range(len(columns)):
        table[0, col].set_text_props(fontweight="bold")
        table[0, col].set_facecolor("#E8E8E8")
    axis.set_title("Best 30 trials, ranked by mean |r| across base/elbow/wrist", fontsize=13, fontweight="bold", pad=14)
    figure.tight_layout()
    return figure


def _boxplot_data(results: pd.DataFrame, methods: tuple[str, ...], joint: str | None = None) -> list[np.ndarray]:
    subset = results if joint is None else results.loc[results["joint"] == joint]
    return [subset.loc[subset["method"] == method, "abs_correlation"].to_numpy(float) for method in methods]


def _style_boxes(bplot: dict, methods: tuple[str, ...]) -> None:
    for patch, method in zip(bplot["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.75)
    for median in bplot["medians"]:
        median.set_color("#202020")
        median.set_linewidth(1.6)


def _draw_pooled_boxplot(results: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9.5, 6.5))
    data = _boxplot_data(results, METHOD_ORDER)
    bplot = axis.boxplot(data, patch_artist=True, widths=0.55, showmeans=True, meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#202020", "markersize": 6})
    _style_boxes(bplot, METHOD_ORDER)
    means = [np.mean(d) for d in data]
    tick_labels = [f"{METHOD_LABELS[m]}\n(mean={mean:.3f})" for m, mean in zip(METHOD_ORDER, means)]
    axis.set_xticks(range(1, len(METHOD_ORDER) + 1), tick_labels)
    axis.set_ylabel("Absolute correlation |r| (base + elbow + wrist pooled)")
    axis.set_ylim(-0.05, 1.05)
    axis.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    axis.set_title(
        "RSS vs. classical BSS baselines on the 30 best trials\n"
        "(each fit on the identical bias-corrected, integrated observation window RSS used)",
        fontsize=12, fontweight="bold",
    )
    figure.tight_layout()
    return figure


def _draw_per_joint_boxplot(results: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 5.6), sharey=True)
    for axis, joint in zip(axes, JOINTS):
        data = _boxplot_data(results, METHOD_ORDER, joint=joint)
        bplot = axis.boxplot(data, patch_artist=True, widths=0.55, showmeans=True, meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#202020", "markersize": 5})
        _style_boxes(bplot, METHOD_ORDER)
        axis.set_xticks(range(1, len(METHOD_ORDER) + 1), [METHOD_LABELS[m] for m in METHOD_ORDER], rotation=20, ha="right")
        axis.set_title(joint.title(), fontsize=12, fontweight="bold")
        axis.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Absolute correlation |r|")
    axes[0].set_ylim(-0.05, 1.05)
    figure.suptitle("Per-joint breakdown: RSS vs. classical BSS baselines (best 30 trials)", fontsize=13, fontweight="bold")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return figure


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ranked = _rank_all_trials()
    top = ranked.iloc[:TOP_N].reset_index(drop=True)
    top.drop(columns=["path"]).to_csv(TABLE_CSV_PATH, index=False)

    overall_mean = float(ranked["mean_r"].mean())
    top_mean = float(top["mean_r"].mean())
    top_per_joint = {joint: float(top[joint].mean()) for joint in JOINTS}
    n_18 = int(top["session"].str.endswith("2026-09-18").sum())
    n_09 = len(top) - n_18

    comparison, failures = compute_comparison_table(top)
    comparison.to_csv(COMPARISON_CSV_PATH, index=False)
    if failures:
        print(f"{len(failures)} baseline-comparison failure(s):")
        for failure in failures:
            print(" ", failure)
    method_means = comparison.groupby("method")["abs_correlation"].mean().reindex(METHOD_ORDER)

    with PdfPages(PDF_PATH) as pdf:
        title_figure = plt.figure(figsize=(11.0, 8.4))
        title_figure.text(0.5, 0.62, "QArm RSS Best 30 Trials", ha="center", fontsize=24, fontweight="bold")
        title_figure.text(
            0.5, 0.54,
            f"Top {TOP_N} of {len(ranked)} matched trials by mean |Pearson r| across base/elbow/wrist "
            f"({n_09} from 2026-09-09, {n_18} from 2026-09-18)",
            ha="center", fontsize=12,
        )
        title_figure.text(
            0.5, 0.47,
            f"Best-30 mean|r| = {top_mean:.3f}  (base={top_per_joint['base']:.3f}, "
            f"elbow={top_per_joint['elbow']:.3f}, wrist={top_per_joint['wrist']:.3f})\n"
            f"vs. all-65 mean|r| = {overall_mean:.3f}",
            ha="center", fontsize=11, fontweight="bold",
        )
        title_figure.text(
            0.5, 0.39,
            "Settings: adam optimizer, SSA preprocessing disabled, bias-window 1.0s,\n"
            "piecewise_sparse_innovation adaptive regularizer.",
            ha="center", fontsize=9.5, color="#666666",
        )
        plt.axis("off")
        pdf.savefig(title_figure)
        plt.close(title_figure)

        table_figure = _draw_ranking_table(top)
        pdf.savefig(table_figure)
        plt.close(table_figure)

        pooled_figure = _draw_pooled_boxplot(comparison)
        pdf.savefig(pooled_figure)
        plt.close(pooled_figure)

        per_joint_figure = _draw_per_joint_boxplot(comparison)
        pdf.savefig(per_joint_figure)
        plt.close(per_joint_figure)

        for row in top.itertuples(index=False):
            figure = _plot_trial(row.path, row.session, row.test_number)
            png_name = f"{row.session}_test_{row.test_number:02d}.png"
            figure.savefig(OUTPUT_DIR / png_name, dpi=150)
            pdf.savefig(figure)
            plt.close(figure)

    print(f"Ranked all {len(ranked)} trials; kept the top {TOP_N}.")
    print(f"Best-30 mean|r| = {top_mean:.4f} vs all-65 mean|r| = {overall_mean:.4f}")
    print("Method comparison on the best 30 trials (mean |r|, pooled across joints):")
    for method in METHOD_ORDER:
        print(f"  {METHOD_LABELS[method]:12s} {method_means[method]:.4f}")
    print(f"Wrote ranking table to {TABLE_CSV_PATH}")
    print(f"Wrote baseline comparison table to {COMPARISON_CSV_PATH}")
    print(f"Wrote {len(top)} per-trial plots to {OUTPUT_DIR}")
    print(f"Wrote combined PDF report to {PDF_PATH}")


if __name__ == "__main__":
    main()
