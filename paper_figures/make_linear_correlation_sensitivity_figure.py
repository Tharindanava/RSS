"""Figures for the linear-grid correlation sensitivity sweep (no titles/axis titles/legends).

Panel order (2x2): harmonic, filtered noise, chirp modulated, piecewise transient.
Lines = mean over the 10 replicates; band = 95% t-interval over replicates.
"""
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rss_paper_fig_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent / "correlation_sensitivity_linear"
METHODS = [("rss_specialized", "#d62728"), ("pca", "#1f77b4"), ("jade_ica", "#2ca02c"),
           ("fastica", "#9467bd"), ("sobi", "#f5b21b")]
FAMS = ["harmonic", "filtered_noise", "chirp_modulated", "piecewise_transient"]
XT = [0, 0.2, 0.4, 0.6, 0.8, 0.95]

rows = []
for f in sorted((OUT / "case_metrics").glob("*.json")):
    p = json.loads(f.read_text())
    for m, v in p["methods"].items():
        rows.append({"family": p["family"], "replicate": p["replicate"], "rho": p["rho"], "method": m, **v})
df = pd.DataFrame(rows)
df.to_csv(OUT / "all_case_metrics.csv", index=False)
print(len(df) // 5, "cases loaded")


def agg(d, col):
    g = d.groupby("rho")[col]
    mean, sd, n = g.mean(), g.std(ddof=1), g.count()
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return mean, mean - h, mean + h


def style(ax, ylim):
    ax.set_xlim(-0.02, 0.97)
    ax.set_xticks(XT, ["0", "0.2", "0.4", "0.6", "0.8", "0.95"])
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="both", labelsize=14, width=1.2)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    ax.grid(True, alpha=0.25)


def line(ax, x, y, lo, hi, color, band):
    if band:
        ax.fill_between(x, lo, hi, color=color, alpha=band, lw=0)
    ax.plot(x, y, color=color, lw=2.6, marker="o", ms=4.5)


summary = {}
for col, tag, ylim in (("mean_abs_source_correlation", "correlation", (0.3, 1.0)), ("mean_source_rmse", "rmse", None)):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    for ax, fam in zip(axes.ravel(), FAMS):
        for key, color in METHODS:
            mean, lo, hi = agg(df[(df.family == fam) & (df.method == key)], col)
            line(ax, mean.index, mean, lo, hi, color, 0.06)
        style(ax, ylim)
    fig.tight_layout()
    fig.savefig(OUT / f"{tag}_sensitivity_by_family.png", dpi=300)
    fig.savefig(OUT / f"{tag}_sensitivity_by_family.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for key, color in METHODS:
        # per-family means first, then average families (equal weight)
        fm = df[df.method == key].groupby(["family", "rho"])[col].mean().groupby("rho").mean()
        line(ax, fm.index, fm, None, None, color, 0.0)
        summary[(tag, key)] = fm
    style(ax, ylim)
    fig.tight_layout()
    fig.savefig(OUT / f"{tag}_sensitivity_family_average.png", dpi=300)
    fig.savefig(OUT / f"{tag}_sensitivity_family_average.pdf")
    plt.close(fig)

for tag in ("correlation", "rmse"):
    t = pd.DataFrame({k: summary[(tag, k)] for k, _ in METHODS}).round(3)
    t.to_csv(OUT / f"{tag}_sensitivity_family_average_table.csv")
    print(tag); print(t.to_string())
