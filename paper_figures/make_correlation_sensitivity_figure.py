"""Correlation-sensitivity figures: metric vs imposed inter-source correlation rho.

No titles / axis titles / legends. Panel order (2x2): harmonic, filtered noise, chirp modulated,
piecewise transient. Lines = mean over 10 replicates, band = 95% CI. RSS = specialized loss.
"""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rss_paper_fig_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "correlation_sensitivity"
OUT.mkdir(exist_ok=True)
d = pd.read_csv(REPO / "notebooks/artifacts/rss_validation_specialized_losses/comparison_report/tables/"
                "correlation_sensitivity_summary.csv")
METHODS = [("rss_specialized", "#d62728"), ("pca", "#1f77b4"), ("jade_ica", "#2ca02c"),
           ("fastica", "#9467bd"), ("sobi", "#f5b21b")]
FAMS = ["harmonic", "filtered_noise", "chirp_modulated", "piecewise_transient"]
XT = [0, 1e-3, 1e-2, 1e-1, 0.9]
XL = ["0", "0.001", "0.01", "0.1", "0.9"]


def style(ax, ylim):
    ax.set_xscale("symlog", linthresh=1e-4, linscale=0.4)
    ax.set_xlim(-2e-5, 1.2)
    ax.set_xticks(XT, XL)
    ax.minorticks_off()
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="both", labelsize=14, width=1.2)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    ax.grid(True, alpha=0.25)


def line(ax, x, y, lo, hi, color, lw=2.6, band=0.15):
    ax.fill_between(x, lo, hi, color=color, alpha=band, lw=0)
    ax.plot(x, y, color=color, lw=lw, marker="o", ms=4.5)


for metric, tag, ylim in (("mean_abs_source_correlation", "correlation", (0.4, 1.0)),
                          ("mean_source_rmse", "rmse", None)):
    m = d[d.metric == metric]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    for ax, fam in zip(axes.ravel(), FAMS):
        for key, color in METHODS:
            s = m[(m.family == fam) & (m.method_key == key)].sort_values("target_source_correlation")
            line(ax, s.target_source_correlation, s["mean"], s.ci_lower, s.ci_upper, color, band=0.06)
        style(ax, ylim)
    fig.tight_layout()
    fig.savefig(OUT / f"{tag}_sensitivity_by_family.png", dpi=300)
    fig.savefig(OUT / f"{tag}_sensitivity_by_family.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for key, color in METHODS:
        s = m[m.method_key == key].groupby("target_source_correlation")[["mean", "ci_lower", "ci_upper"]].mean()
        line(ax, s.index, s["mean"], s.ci_lower, s.ci_upper, color, band=0.0)
    style(ax, ylim)
    fig.tight_layout()
    fig.savefig(OUT / f"{tag}_sensitivity_family_average.png", dpi=300)
    fig.savefig(OUT / f"{tag}_sensitivity_family_average.pdf")
    plt.close(fig)

piv = d[d.metric == "mean_abs_source_correlation"].groupby(["method_key", "target_source_correlation"])["mean"].mean().unstack(0)
piv.round(3).to_csv(OUT / "correlation_sensitivity_family_average_table.csv")
print(piv.round(3).to_string())
