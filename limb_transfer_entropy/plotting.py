"""Matrix plotting for limb transfer entropy."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_matrix_plot(
    matrix: pd.DataFrame,
    *,
    side: str,
    source_path: Path,
    output_path: Path,
) -> None:
    """Write an annotated source-row/target-column TE matrix plot."""
    values = matrix.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    finite = values[np.isfinite(values)]
    maximum = max(float(np.max(finite, initial=0.0)), 1e-12)

    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("#D1D5DB")
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(masked, cmap=cmap, vmin=0.0, vmax=maximum)
    labels = matrix.columns.astype(str).tolist()
    axis.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels=matrix.index.astype(str).tolist())
    axis.set_xlabel("Target segment")
    axis.set_ylabel("Source segment")
    axis.set_title(
        f"{side.capitalize()} Limb Raw Transfer Entropy\n{source_path.parent.name}"
    )

    threshold = maximum * 0.55
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                axis.text(
                    column,
                    row,
                    f"{value:.4f}",
                    ha="center",
                    va="center",
                    color="white" if value >= threshold else "black",
                    fontsize=8,
                )
            else:
                axis.text(column, row, "—", ha="center", va="center", color="#374151")

    figure.colorbar(image, ax=axis, label="Raw transfer entropy (bits)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
