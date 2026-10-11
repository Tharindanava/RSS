"""Independently validate the hyperparameter-search winner on a fresh random sample.

``hyperparameter_search.py`` tuned settings on a 10-batch sample that was
skewed toward ``session_2026-09-09`` (6 of its 7 valid batches vs.
only 4 of 6 valid ``session_2026-09-18`` batches). This script draws
a *new*, independent random sample of 20 matched trials -- deliberately
skewed the other way (12 from 2026-09-18, 8 from 2026-09-09) -- and compares the
existing production settings (``outputs/rss/QArm_PSO_SO3_2000Epochs``)
against the search-winning settings (``outputs/rss/QArm_Adam_NoSSA_Optimized``)
on that sample alone, using the already-computed matched CSVs from both
(no reprocessing needed).

Two things are checked: whether the aggregate improvement direction/size on
this fresh, 2026-09-18-heavy sample matches the full 65-trial result, and whether
it holds up trial-by-trial and not just on average. Three-way overlay plots
(measured ground truth vs. production source vs. optimized source, one
figure per trial with the three joints stacked) are written per trial and
bundled into one PDF for visual confirmation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_ROOT = REPO_ROOT / "outputs" / "rss" / "QArm_PSO_SO3_2000Epochs"
OPTIMIZED_ROOT = REPO_ROOT / "outputs" / "rss" / "QArm_Adam_NoSSA_Optimized"
VALIDATION_ROOT = REPO_ROOT / "outputs" / "rss_hparam_search" / "validation_sample"
PLOTS_DIR = VALIDATION_ROOT / "plots"
PDF_PATH = VALIDATION_ROOT / "validation_report.pdf"
TABLE_PATH = VALIDATION_ROOT / "validation_table.csv"

DATE_FOLDERS = ("session_2026-09-09", "session_2026-09-18")
JOINTS = ("base", "elbow", "wrist")
JOINT_COLORS = {"base": "#0072B2", "elbow": "#D55E00", "wrist": "#009E73"}
SAMPLE_SEED = 4271
N_FROM_SESSION_0918 = 12
N_FROM_SESSION_0909 = 8


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > np.finfo(float).eps else float("nan")


def _available_tests(root: Path, session: str) -> list[int]:
    matched_dir = root / session / "matched"
    return sorted(int(p.stem.split("_")[1]) for p in matched_dir.glob("test_*.csv"))


def draw_sample() -> list[tuple[str, int]]:
    rng = np.random.default_rng(SAMPLE_SEED)
    tests_18 = _available_tests(PRODUCTION_ROOT, "session_2026-09-18")
    tests_09 = _available_tests(PRODUCTION_ROOT, "session_2026-09-09")
    chosen_18 = rng.choice(tests_18, size=N_FROM_SESSION_0918, replace=False)
    chosen_09 = rng.choice(tests_09, size=N_FROM_SESSION_0909, replace=False)
    sample = [("session_2026-09-18", int(n)) for n in chosen_18] + [
        ("session_2026-09-09", int(n)) for n in chosen_09
    ]
    rng.shuffle(sample)
    return sample


def _joint_correlations(matched_path: Path) -> dict[str, tuple[float, np.ndarray, np.ndarray]]:
    frame = pd.read_csv(matched_path)
    time = frame["time_s"].to_numpy(float)
    result = {}
    for joint in JOINTS:
        measured = frame[f"measured_{joint}_rad"].to_numpy(float)
        source = frame[f"source_{joint}"].to_numpy(float)
        result[joint] = (_pearson(measured, source), measured, source)
    return time, result


def plot_comparison(session: str, test_number: int) -> tuple[plt.Figure, dict]:
    production_path = PRODUCTION_ROOT / session / "matched" / f"test_{test_number}.csv"
    optimized_path = OPTIMIZED_ROOT / session / "matched" / f"test_{test_number}.csv"
    time_p, production = _joint_correlations(production_path)
    _time_o, optimized = _joint_correlations(optimized_path)

    figure, axes = plt.subplots(3, 1, figsize=(11.0, 8.4), sharex=True)
    row = {"session": session, "test_number": test_number}
    for axis, joint in zip(axes, JOINTS):
        r_production, measured, source_production = production[joint]
        r_optimized, _measured2, source_optimized = optimized[joint]
        axis.plot(time_p, measured, color="#202020", linewidth=1.7, label="Measured (ground truth)")
        axis.plot(time_p, source_production, color="#B0B0B0", linewidth=1.1, linestyle="--", label="Production (PSO+SSA)")
        axis.plot(time_p, source_optimized, color=JOINT_COLORS[joint], linewidth=1.25, alpha=0.95, label="Optimized (Adam, no SSA)")
        axis.set_title(
            f"{joint.title()}: production r={r_production:.3f}  →  optimized r={r_optimized:.3f}"
            f"  (Δ={r_optimized - r_production:+.3f})",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
        )
        axis.set_ylabel("Angle (rad)")
        axis.grid(True, alpha=0.2, linewidth=0.6)
        axis.legend(loc="upper right", frameon=False, ncol=3, fontsize=7.5)
        row[f"{joint}_production_r"] = abs(r_production)
        row[f"{joint}_optimized_r"] = abs(r_optimized)
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(f"{session} — Test {test_number}: production vs. optimized settings", fontsize=13, fontweight="bold")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return figure, row


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sample = draw_sample()
    n_18 = sum(1 for session, _ in sample if session.endswith("2026-09-18"))
    n_09 = len(sample) - n_18
    print(f"Validation sample: {len(sample)} trials ({n_18} from 2026-09-18, {n_09} from 2026-09-09)")

    rows = []
    with PdfPages(PDF_PATH) as pdf:
        title_figure = plt.figure(figsize=(11.0, 8.4))
        title_figure.text(0.5, 0.62, "Independent Validation Sample", ha="center", fontsize=24, fontweight="bold")
        title_figure.text(
            0.5, 0.54,
            f"{len(sample)} trials ({n_18} from 2026-09-18, {n_09} from 2026-09-09) — not the hyperparameter-search's own tuning sample",
            ha="center", fontsize=12,
        )
        title_figure.text(
            0.5, 0.47,
            "Production (PSO, SSA on, 2000 epochs) vs. optimized (Adam, SSA off, 400 epochs) sources\noverlaid on measured ground truth.",
            ha="center", fontsize=10.5, color="#444444",
        )
        plt.axis("off")
        pdf.savefig(title_figure)
        plt.close(title_figure)

        for session, test_number in sample:
            figure, row = plot_comparison(session, test_number)
            png_name = f"{session}_test_{test_number:02d}.png"
            figure.savefig(PLOTS_DIR / png_name, dpi=150)
            pdf.savefig(figure)
            plt.close(figure)
            rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(TABLE_PATH, index=False)

    print(f"\nWrote {len(sample)} comparison plots to {PLOTS_DIR}")
    print(f"Wrote combined PDF to {PDF_PATH}")
    print(f"Wrote per-trial table to {TABLE_PATH}\n")

    production_cols = [f"{j}_production_r" for j in JOINTS]
    optimized_cols = [f"{j}_optimized_r" for j in JOINTS]
    production_values = table[production_cols].to_numpy(float).ravel()
    optimized_values = table[optimized_cols].to_numpy(float).ravel()
    print(f"On this {len(sample)}-trial validation sample ({len(production_values)} joint x trial pairs):")
    print(f"  production mean|r| = {production_values.mean():.4f}")
    print(f"  optimized  mean|r| = {optimized_values.mean():.4f}")
    print(f"  delta               = {optimized_values.mean() - production_values.mean():+.4f}")
    n_improved = int(np.sum(optimized_values > production_values))
    print(f"  optimized beat production on {n_improved}/{len(production_values)} joint-trial pairs")
    for joint in JOINTS:
        p = table[f"{joint}_production_r"].to_numpy(float)
        o = table[f"{joint}_optimized_r"].to_numpy(float)
        print(f"  {joint:6s}: production={p.mean():.4f}  optimized={o.mean():.4f}  delta={o.mean()-p.mean():+.4f}")

    for session in DATE_FOLDERS:
        subset = table.loc[table["session"] == session]
        if subset.empty:
            continue
        p = subset[production_cols].to_numpy(float).ravel()
        o = subset[optimized_cols].to_numpy(float).ravel()
        print(f"\n  {session} only (n={len(subset)} trials, {len(p)} joint-trial pairs):")
        print(f"    production mean|r| = {p.mean():.4f}")
        print(f"    optimized  mean|r| = {o.mean():.4f}")
        print(f"    delta               = {o.mean() - p.mean():+.4f}")


if __name__ == "__main__":
    main()
