"""Bar-chart axis alignment figure from saved figure_data.npz (no titles/axis titles/legends)."""
import os, sys
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rss_paper_fig_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, RED = "#1f77b4", "#d62728"
root = Path(__file__).resolve().parent
cases = sys.argv[1:] or ["formal_harmonic_23", "formal_harmonic_24", "formal_harmonic_27"]
for case in cases:
    folder = root / f"harmonic_{case}"
    d = np.load(folder / "figure_data.npz")
    true, est = d["true_axes"], d["rss_axes"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w, gap = 0.38, 1.0
    ticks, labels, centers = [], [], []
    for a in range(3):
        base = a * (3 + gap)
        for c in range(3):
            x = base + c
            ax.bar(x - w / 2, true[a, c], w, color=BLUE, edgecolor=BLUE, lw=1.5)
            ax.bar(x + w / 2, est[a, c], w, facecolor=RED, alpha=0.25, edgecolor=RED, lw=2.4, ls=(0, (1.6, 1.2)))
            ax.bar(x + w / 2, est[a, c], w, facecolor="none", edgecolor=RED, lw=2.4, ls=(0, (1.6, 1.2)))
            ticks.append(x); labels.append("xyz"[c])
        centers.append(base + 1)
    ax.axhline(0, color="black", lw=1.2)
    ax.set_xticks(ticks, labels)
    lo = min(true.min(), est.min(), 0); hi = max(true.max(), est.max()); ax.set_ylim(lo - 0.08, hi + 0.08)
    ax.tick_params(axis="both", labelsize=14, width=1.2)
    sec = ax.secondary_xaxis("bottom")
    sec.set_xticks(centers, ["Axis 1", "Axis 2", "Axis 3"])
    sec.tick_params(length=0, pad=26, labelsize=14)
    for lab in ax.get_xticklabels() + ax.get_yticklabels() + sec.get_xticklabels():
        lab.set_fontweight("bold")
    sec.spines["bottom"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / "05_axis_alignment_bars.png", dpi=300)
    fig.savefig(folder / "05_axis_alignment_bars.pdf")
    plt.close(fig)
    print(case, "saved")
