"""Correlation comparison table + box plot: 4 simulation families and QArm.

Methods: RSS (specialized loss for simulations), PCA, ICA (JADE), FastICA, SOBI.
Metrics: mean |correlation| between matched separated and true sources per case, RMSE (rad), and
NRMSE = per-source RMSE / range (max - min) of the ground-truth source, averaged over the three sources
(simulation: 50 cases/family; QArm: mean over base/elbow/wrist per trial, 30 best trials).
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
OUT = Path(__file__).resolve().parent / "correlation_comparison"
OUT.mkdir(exist_ok=True)
SIM = (REPO / "notebooks/artifacts/rss_validation_specialized_losses/bss_baselines/comparisons/"
       "formal_experiments/summary/per_case_metrics.csv")
QARM = REPO / "outputs/rss/QArm_Adam_NoSSA_Piecewise/best_30_baseline_comparison.csv"

METHODS = [("RSS", "rss_specialized", "rss"), ("PCA", "pca", "pca"), ("ICA", "jade_ica", "jade_ica"),
           ("FastICA", "fastica", "fastica"), ("SOBI", "sobi", "sobi")]
GROUPS = [("Harmonic", "harmonic"), ("Filtered noise", "filtered_noise"),
          ("Chirp modulated", "chirp_modulated"), ("Piecewise transient", "piecewise_transient")]
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#f5b21b"]

QARM_RMSE = Path(__file__).resolve().parent / "correlation_comparison" / "qarm_best30_rmse.csv"
SIM_SRC = SIM.with_name("per_source_metrics.csv")
sim = pd.read_csv(SIM)
# NRMSE per source = RMSE / range (max-min) of the true source (read from the RSS case's source_comparison.csv)
src = pd.read_csv(SIM_SRC)
_ART = SIM.parents[4]


def _truth_range(case_id, family):
    t = pd.read_csv(_ART / "rss_specialized/formal_experiments/cases" / family / case_id / "source_comparison.csv")
    return (t[["true_source_1", "true_source_2", "true_source_3"]].max() - t[["true_source_1", "true_source_2", "true_source_3"]].min()).to_numpy()


_rng = {c: _truth_range(c, f) for c, f in src[["case_id", "family"]].drop_duplicates().itertuples(index=False)}
src["nrmse"] = src.source_rmse_rad / np.array([_rng[c][int(i) - 1 if src.source_index.min() == 1 else int(i)]
                                                for c, i in zip(src.case_id, src.source_index)])
sim = sim.merge(src.groupby(["case_id", "method_key"])["nrmse"].mean().rename("mean_source_nrmse").reset_index(),
                on=["case_id", "method_key"], how="left")
qall = pd.read_csv(QARM_RMSE)
group_labels = [g for g, _ in GROUPS] + ["QArm"]
METRICS = {
    "correlation": dict(sim_col="mean_abs_source_correlation", q_col="abs_correlation", ylim=(0.2, 1.05), best=np.argmax),
    "rmse": dict(sim_col="mean_source_rmse", q_col="rmse_rad", ylim=None, best=np.argmin),
    "nrmse": dict(sim_col="mean_source_nrmse", q_col="nrmse_range", ylim=None, best=np.argmin),
}


def build(metric):
    cfg = METRICS[metric]
    q = qall.groupby(["method", "session", "test_number"])[cfg["q_col"]].mean().reset_index()
    data = {}
    for g, fam in GROUPS:
        for m, skey, _ in METHODS:
            data[(g, m)] = sim[(sim.family == fam) & (sim.method_key == skey)][cfg["sim_col"]].to_numpy()
    for m, _, qkey in METHODS:
        data[("QArm", m)] = q[q.method == qkey][cfg["q_col"]].to_numpy()
    rows = []
    for g in group_labels:
        row = {"Dataset": g, "n": len(data[(g, "RSS")])}
        for m, _, _ in METHODS:
            x = data[(g, m)]
            row[m] = f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"
            row[m + "_mean"], row[m + "_std"] = x.mean(), x.std(ddof=1)
        rows.append(row)
    return data, pd.DataFrame(rows)


def write_tables(metric, table):
    suffix = "" if metric == "correlation" else "_rmse"
    table.to_csv(OUT / f"{metric}_table_full.csv", index=False)
    show = table[["Dataset", "n"] + [m for m, _, _ in METHODS]]
    show.to_csv(OUT / f"{metric}_table.csv", index=False)
    (OUT / f"{metric}_table.md").write_text(md(show), encoding="utf-8")
    best = METRICS[metric]["best"](table[[m + "_mean" for m, _, _ in METHODS]].to_numpy(), axis=1)
    tex = ["\\begin{tabular}{l" + "c" * (len(METHODS) + 1) + "}", "\\hline",
           "Dataset & $n$ & " + " & ".join(m for m, _, _ in METHODS) + " \\\\", "\\hline"]
    for i, r in table.iterrows():
        cells = []
        for j, (m, _, _) in enumerate(METHODS):
            t = f"{r[m + '_mean']:.3f} $\\pm$ {r[m + '_std']:.3f}"
            cells.append(f"\\textbf{{{t}}}" if j == best[i] else t)
        tex.append(f"{r['Dataset']} & {r['n']} & " + " & ".join(cells) + " \\\\")
    tex += ["\\hline", "\\end{tabular}"]
    (OUT / f"{metric}_table.tex").write_text("\n".join(tex), encoding="utf-8")
    return show


def md(df):
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
                     + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


def boxplot(metric, data):
    fig, ax = plt.subplots(figsize=(12, 5))
    width, gap = 0.15, 1.0
    centers = []
    for gi, g in enumerate(group_labels):
        c0 = gi * (5 * width + gap)
        centers.append(c0 + 2 * width)
        for mi, (m, _, _) in enumerate(METHODS):
            x = data[(g, m)]
            pos = c0 + mi * width
            bp = ax.boxplot(x, positions=[pos], widths=width * 0.8, patch_artist=True, showfliers=True,
                            medianprops=dict(color="black", lw=1.6),
                            flierprops=dict(marker="o", ms=3, mfc=COLORS[mi], mec=COLORS[mi], alpha=0.6))
            bp["boxes"][0].set(facecolor=COLORS[mi], alpha=0.55, edgecolor=COLORS[mi], lw=1.4)
            for k in ("whiskers", "caps"):
                for ln in bp[k]:
                    ln.set(color=COLORS[mi], lw=1.4)
            ax.errorbar(pos, x.mean(), yerr=x.std(ddof=1), fmt="D", color="black", ms=4.5, capsize=3, lw=1.4, zorder=5)
    ax.set_xticks(centers, group_labels)
    ylim = METRICS[metric]["ylim"]
    if ylim:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", labelsize=14, width=1.2)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / f"{metric}_boxplot.png", dpi=300)
    fig.savefig(OUT / f"{metric}_boxplot.pdf")
    plt.close(fig)


shows = {}
for metric in METRICS:
    data, table = build(metric)
    shows[metric] = write_tables(metric, table)
    boxplot(metric, data)
    print(metric); print(shows[metric].to_string(index=False))

combined = pd.concat([shows["correlation"].assign(Metric="Mean |r|"), shows["rmse"].assign(Metric="RMSE (rad)"),
                      shows["nrmse"].assign(Metric="NRMSE (max-normalised)")])
combined = combined[["Dataset", "Metric", "n"] + [m for m, _, _ in METHODS]].sort_values(
    "Dataset", key=lambda c: c.map({g: i for i, g in enumerate(group_labels)}), kind="stable")
combined.to_csv(OUT / "combined_table.csv", index=False)
(OUT / "combined_table.md").write_text(md(combined), encoding="utf-8")

# separate colour key (kept out of the figures themselves)
fig, ax = plt.subplots(figsize=(6, 0.5))
ax.axis("off")
for i, ((m, _, _), c) in enumerate(zip(METHODS, COLORS)):
    ax.add_patch(plt.Rectangle((i * 1.2, 0), 0.25, 0.6, color=c, alpha=0.7))
    ax.text(i * 1.2 + 0.32, 0.3, m, va="center", fontsize=11, fontweight="bold")
ax.set_xlim(0, 6); ax.set_ylim(0, 0.6)
fig.savefig(OUT / "boxplot_colour_key.png", dpi=200, bbox_inches="tight")
