"""Publication figures for the best Harmonic-family RSS case (case id via argv, default formal_harmonic_24).

Re-runs RSS-specialized on the saved scenario (seed 20023) using the notebook's
configuration, then writes label-free figures (no titles, axis titles or legends).
"""
from __future__ import annotations

import json
import os
import sys
import types
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rss_paper_fig_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
NOTEBOOK = REPO / "notebooks" / "rss_validation_specialized_losses.ipynb"
CASE_ID = sys.argv[1] if len(sys.argv) > 1 else "formal_harmonic_24"
SAVED = (REPO / "notebooks/artifacts/rss_validation_specialized_losses/rss_specialized"
         / f"formal_experiments/cases/harmonic/{CASE_ID}/source_comparison.csv")
OUT = Path(__file__).resolve().parent / f"harmonic_{CASE_ID}"

BLUE, YELLOW, RED = "#1f77b4", "#f5b21b", "#d62728"
SRC_COLORS = (BLUE, YELLOW, RED)
TICK = dict(labelsize=14, width=1.2)


def load_namespace() -> dict:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    module = types.ModuleType("harmonic_fig_setup")
    module.__file__ = str(NOTEBOOK)
    sys.modules["harmonic_fig_setup"] = module
    for index in (2, 4, 8, 15):
        src = "".join(nb["cells"][index].get("source", []))
        exec(compile(src, f"{NOTEBOOK.name}:cell-{index}", "exec"), module.__dict__)
    # batch config cell (contains batch_rss_config)
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "batch_rss_config = RSSConfig(" in src:
            exec(compile(src, "batch-config", "exec"), module.__dict__)
            break
    return module.__dict__


def style(ax) -> None:
    ax.tick_params(axis="both", **TICK)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(True, alpha=0.25)


def save(fig, name: str, tight: bool = True) -> None:
    kw = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(OUT / f"{name}.png", dpi=300, **kw)
    fig.savefig(OUT / f"{name}.pdf", **kw)
    plt.close(fig)


def stacked(t, rows, name, size=(6.4, 3.4)):
    fig, ax = plt.subplots(figsize=size)
    for row, color in zip(rows, SRC_COLORS):
        ax.plot(t, row, color=color, lw=1.8)
    ax.set_xlim(t[0], t[-1])
    style(ax)
    save(fig, name)


def main() -> None:
    from rss.simulation import BatchStudyConfig, create_standard_case_specs, generate_family_scenario

    ns = load_namespace()
    study = replace(BatchStudyConfig(), standard_cases_per_family=50)
    spec = next(s for s in create_standard_case_specs(study, base_seed=20000, case_id_prefix="formal_")
                if s.case_id == CASE_ID)
    scenario = generate_family_scenario(spec, ns["batch_sim_config"], study)
    run = ns["run_specialized_validation_for_scenario"](
        scenario, ns["batch_rss_config"], ns["DEFAULT_SPECIALIZED_LOSS_CONFIG"]
    )
    val = run.validation if hasattr(run, "validation") else run
    match = val.match
    est = np.asarray(match.estimated_sources_matched)
    saved = pd.read_csv(SAVED)[["rss_source_1", "rss_source_2", "rss_source_3"]].values.T
    print("max |rerun - saved| RSS source:", float(np.abs(est - saved).max()))
    print("mean |r|:", val.metrics["mean_abs_source_correlation"], "axis err deg:", val.metrics["mean_axis_error_deg"])

    t = np.asarray(scenario.time_seconds)
    OUT.mkdir(parents=True, exist_ok=True)
    stacked(t, scenario.sources, "01_source_signals")
    stacked(t, scenario.observations, "02_observer_signals")
    for i in range(3):
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.plot(t, scenario.sources[i], color=BLUE, lw=3.4)
        ax.plot(t, est[i], color=RED, lw=2.4, ls=(0, (3, 1.6)), zorder=3)
        ax.set_xlim(t[0], t[-1])
        style(ax)
        save(fig, f"03_overlay_source_{i + 1}")

    true_axes = np.asarray(scenario.source_axes_global)
    est_axes = np.asarray(match.estimated_axes_matched)
    fig = plt.figure(figsize=(5.2, 5.2))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], projection="3d")
    for i in range(3):
        ax.plot([0, true_axes[i, 0]], [0, true_axes[i, 1]], [0, true_axes[i, 2]], color=BLUE, lw=5.0)
        ax.plot([0, est_axes[i, 0]], [0, est_axes[i, 1]], [0, est_axes[i, 2]], color=RED, lw=3.2,
                ls=(0, (1.6, 1.2)), zorder=10)
        ax.scatter(*est_axes[i], color=RED, s=40, marker="x", zorder=10)
    ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.9, 0.9); ax.set_zlim(-0.9, 0.9)
    ax.set_box_aspect((1, 1, 1), zoom=0.9)
    ax.tick_params(axis="both", **TICK)
    ax.set_xticks([-0.5, 0, 0.5]); ax.set_yticks([-0.5, 0, 0.5]); ax.set_zticks([-0.5, 0, 0.5])
    ax.tick_params(axis="x", pad=-2); ax.tick_params(axis="y", pad=10); ax.tick_params(axis="z", pad=2)
    for lab in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        lab.set_fontweight("bold")
    save(fig, "04_axis_alignment", tight=False)
    np.savez(OUT / "figure_data.npz", time=t, sources=scenario.sources, observations=scenario.observations,
             rss_sources=est, true_axes=true_axes, rss_axes=est_axes)


if __name__ == "__main__":
    main()
