"""Reference key for the correlation-sensitivity figures: method colours + 2x2 family layout."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rss_paper_fig_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent / "correlation_sensitivity_linear"
METHODS = [("RSS", "#d62728"), ("PCA", "#1f77b4"), ("ICA (JADE)", "#2ca02c"), ("FastICA", "#9467bd"), ("SOBI", "#f5b21b")]
FAMS = [["Harmonic", "Filtered noise"], ["Chirp modulated", "Piecewise transient"]]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.4), gridspec_kw={"width_ratios": [1, 1.25]})
a1.set_xlim(0, 4); a1.set_ylim(0, 5.4); a1.axis("off")
for i, (name, c) in enumerate(METHODS):
    y = 4.5 - i
    a1.plot([0.1, 1.0], [y, y], color=c, lw=3.2, marker="o", ms=5, markevery=[0, 1])
    a1.text(1.25, y, name, va="center", fontsize=15, fontweight="bold")

a2.set_xlim(0, 2); a2.set_ylim(0, 2); a2.axis("off"); a2.set_aspect("auto")
for r in range(2):
    for c in range(2):
        a2.add_patch(Rectangle((c + 0.04, 1 - r + 0.04), 0.92, 0.92, fill=False, ec="black", lw=1.6))
        a2.text(c + 0.5, 1 - r + 0.5, FAMS[r][c].replace(" ", "\n", 1) if len(FAMS[r][c]) > 12 else FAMS[r][c],
                ha="center", va="center", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "sensitivity_figure_key.png", dpi=300)
fig.savefig(OUT / "sensitivity_figure_key.pdf")
