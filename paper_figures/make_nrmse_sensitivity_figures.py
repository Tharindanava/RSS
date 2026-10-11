"""NRMSE (RMSE / range (max - min) of the ground-truth source) vs imposed inter-source correlation rho.

Companion to make_correlation_sensitivity_figure.py (log grid) and
make_linear_correlation_sensitivity_figure.py (linear grid); same style, no titles/legends.

Log grid   : exact. Per-source RMSE_i / range_i, averaged over the 3 sources, from per-case metrics.json.
Linear grid: only the mean RMSE over sources was stored, so NRMSE = mean RMSE / mean_i(range_i), with the
             ranges regenerated deterministically from the case specs (approximation; the log-grid check
             printed below quantifies the difference).
"""
import contextlib, io, json, os, sys
from dataclasses import replace
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rss_paper_fig_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
ART = REPO / "notebooks/artifacts/rss_validation_specialized_losses"
DIRS = {"rss_specialized": ART / "rss_specialized/correlation_sensitivity/cases"} | {
    k: ART / f"bss_baselines/{k}/correlation_sensitivity/cases" for k in ("pca", "jade_ica", "fastica", "sobi")}
METHODS = [("rss_specialized", "#d62728"), ("pca", "#1f77b4"), ("jade_ica", "#2ca02c"),
           ("fastica", "#9467bd"), ("sobi", "#f5b21b")]
FAMS = ["harmonic", "filtered_noise", "chirp_modulated", "piecewise_transient"]


def agg(d, col):
    g = d.groupby("rho")[col]
    mean, sd, n = g.mean(), g.std(ddof=1), g.count()
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return mean, mean - h, mean + h


def make(df, col, out, xt, xl, xlim, log, key_col="method"):
    out.mkdir(exist_ok=True)

    ytop = 1.05 * df.groupby(["family", "method", "rho"])[col].mean().max()

    def style(ax):
        if log:
            ax.set_xscale("symlog", linthresh=1e-4, linscale=0.4)
        ax.set_xlim(*xlim)
        ax.set_xticks(xt, xl)
        ax.minorticks_off()
        ax.set_ylim(0, ytop)
        ax.tick_params(axis="both", labelsize=14, width=1.2)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontweight("bold")
        ax.grid(True, alpha=0.25)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    for ax, fam in zip(axes.ravel(), FAMS):
        for key, color in METHODS:
            m, lo, hi = agg(df[(df.family == fam) & (df[key_col] == key)], col)
            ax.fill_between(m.index, lo, hi, color=color, alpha=0.06, lw=0)
            ax.plot(m.index, m, color=color, lw=2.6, marker="o", ms=4.5)
        style(ax)
    fig.tight_layout()
    fig.savefig(out / "nrmse_sensitivity_by_family.png", dpi=300)
    fig.savefig(out / "nrmse_sensitivity_by_family.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    tab = {}
    for key, color in METHODS:
        fm = df[df[key_col] == key].groupby(["family", "rho"])[col].mean().groupby("rho").mean()
        ax.plot(fm.index, fm, color=color, lw=2.6, marker="o", ms=4.5)
        tab[key] = fm
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "nrmse_sensitivity_family_average.png", dpi=300)
    fig.savefig(out / "nrmse_sensitivity_family_average.pdf")
    plt.close(fig)
    t = pd.DataFrame(tab).round(3)
    t.to_csv(out / "nrmse_sensitivity_family_average_table.csv")
    print(out.name); print(t.to_string())


# ---------------- log grid (exact) ----------------
rows = []
for case in sorted(p.name for p in DIRS["rss_specialized"].iterdir()):
    sc = json.loads((DIRS["rss_specialized"] / case / "metrics.json").read_text())["scenario"]
    tsrc = pd.read_csv(DIRS["rss_specialized"] / case / "source_comparison.csv")[["true_source_1", "true_source_2", "true_source_3"]]
    peaks = (tsrc.max() - tsrc.min()).to_numpy(float)  # ground-truth range per source
    rho = float(case.split("_rhop")[1].replace("_", ".") if "_rhop" in case else 0)
    rho = float("0." + case.split("rhop0_")[1]) if "rhop0_" in case else 0.0
    for key, _ in METHODS:
        m = json.loads((DIRS[key] / case / "metrics.json").read_text())
        rm = np.asarray(m["match"]["source_rmses" if key == "rss_specialized" else "physical_source_rmses"], float)
        rows.append({"family": sc["family"], "rho": rho, "method": key, "case": case,
                     "mean_rmse": rm.mean(), "nrmse": float(np.mean(rm / peaks)),
                     "nrmse_meanpeak": float(rm.mean() / peaks.mean()),
                     "mean_rmse_stored": m["metrics"]["mean_source_rmse"]})
log = pd.DataFrame(rows)
assert np.allclose(log.mean_rmse, log.mean_rmse_stored, atol=1e-9), "per-source RMSEs do not reproduce stored mean RMSE"
log.to_csv(HERE / "correlation_sensitivity/all_case_nrmse.csv", index=False)
rel = (log.nrmse_meanpeak / log.nrmse - 1).abs()
print(f"log grid: mean-peak approximation vs exact NRMSE, mean rel. diff {rel.mean():.3%}, max {rel.max():.3%}")
make(log, "nrmse", HERE / "correlation_sensitivity", [0, 1e-3, 1e-2, 1e-1, 0.9],
     ["0", "0.001", "0.01", "0.1", "0.9"], (-2e-5, 1.2), True)

# ---------------- linear grid (mean-peak normalisation) ----------------
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "notebooks"))
import run_specialized_correlation_resume as R
with contextlib.redirect_stdout(io.StringIO()):
    ns = R.execute_notebook_setup()
    cfg = replace(ns["correlation_study_config"], target_correlations=tuple(round(0.05 * i, 2) for i in range(20)))
    specs = ns["create_correlation_case_specs"](cfg, ns["batch_study_config"])
    peak = {}
    for s in specs:
        case = ns["generate_correlation_scenario"](s, ns["batch_sim_config"], ns["batch_study_config"],
                                                   correlation_tolerance=1e-6)
        peak[(s.family, s.replicate_index, s.target_source_correlation)] = float(
            np.mean(np.ptp(np.asarray(case.generated_sources), axis=1)))
lin = pd.read_csv(HERE / "correlation_sensitivity_linear/all_case_metrics.csv")
lin["mean_peak"] = [peak[(f, r, p)] for f, r, p in zip(lin.family, lin.replicate, lin.rho)]
lin["nrmse"] = lin.mean_source_rmse / lin.mean_peak
lin.to_csv(HERE / "correlation_sensitivity_linear/all_case_nrmse.csv", index=False)
make(lin, "nrmse", HERE / "correlation_sensitivity_linear", [0, 0.2, 0.4, 0.6, 0.8, 0.95],
     ["0", "0.2", "0.4", "0.6", "0.8", "0.95"], (-0.02, 0.97), False)
