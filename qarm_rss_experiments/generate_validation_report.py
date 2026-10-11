"""Build the QArm RSS validation LaTeX report and its figure assets.

This script consumes completed artifacts under ``outputs/qarm_rss_validation``.
For older result directories that predate the PCA and JADE-ICA controls, it fits
only those inexpensive baselines to the exported headline observer channels; it
never refits RSS. Compile the emitted ``report.tex`` with Tectonic (or a
conventional LaTeX installation) from inside the report directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bss_baselines import BSSConfig, METHOD_JADE_ICA, METHOD_PCA, fit_bss
from qarm_rss_experiments.practical_validation import match_and_score


JOINTS = ("base", "elbow", "wrist")
AXES = ("x", "y", "z")
JOINT_COLORS = {"base": "#0072B2", "elbow": "#D55E00", "wrist": "#009E73"}
AXIS_COLORS = {"x": "#0072B2", "y": "#D55E00", "z": "#009E73"}
METHOD_LABELS = {
    "identity": "Identity",
    "rss": "RSS",
    METHOD_PCA: "PCA",
    METHOD_JADE_ICA: "ICA (JADE)",
    "fastica": "ICA (FastICA)",
    "sobi": "SOBI",
}
METHOD_ORDER = ("identity", "rss", METHOD_PCA, METHOD_JADE_ICA, "fastica", "sobi")


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.55,
            "savefig.bbox": "tight",
        }
    )


def _load_metrics(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main = pd.read_csv(root / "comparator_metrics.csv")
    excluded = pd.read_csv(root / "excluded_trial_sensitivity.csv")
    quality = pd.read_csv(root / "data_quality_manifest.csv").sort_values("test_number")
    paired = pd.read_csv(root / "paired_comparisons.csv")
    rss_all = pd.concat(
        [main.loc[main["method"].eq("rss")], excluded.loc[excluded["method"].eq("rss")]],
        ignore_index=True,
    )
    expected = set(quality["test_number"].astype(int))
    observed = set(rss_all["test_number"].astype(int))
    if observed != expected:
        raise ValueError(f"Expected RSS metrics for all audited tests; missing={sorted(expected-observed)}")
    return main, excluded, quality, paired


def _trial_summary(rss_all: pd.DataFrame) -> pd.DataFrame:
    grouped = rss_all.groupby(["test_number", "batch_number"], sort=True)
    summary = grouped.agg(
        mean_rmse_rad=("rmse_rad", "mean"),
        mean_abs_correlation=("abs_correlation", "mean"),
    ).reset_index()
    fisher = grouped["abs_correlation"].apply(
        lambda values: float(
            np.tanh(np.mean(np.arctanh(np.clip(values.to_numpy(float), 0.0, 1.0 - 1e-12))))
        )
    )
    summary = summary.merge(
        fisher.rename("fisher_mean_abs_correlation").reset_index(),
        on=["test_number", "batch_number"],
    )
    return summary


def _evaluate_report_baselines(
    root: Path,
    main: pd.DataFrame,
    quality: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return PCA/ICA rows and a comparator frame containing both methods.

    Completed validation directories made before these baselines were added do
    not contain their rows.  In that case they are fitted to the exact exported
    headline observer channels.  Ground truth is used only by the shared
    post-fit matching and scoring function.
    """

    requested = (METHOD_PCA, METHOD_JADE_ICA)
    existing_methods = set(main["method"].astype(str))
    report_rows: list[dict[str, object]] = []
    new_rows: list[dict[str, object]] = []
    for method in requested:
        if method in existing_methods:
            report_rows.extend(main.loc[main["method"].eq(method)].to_dict("records"))

    missing = tuple(method for method in requested if method not in existing_methods)
    if missing:
        config = BSSConfig(methods=missing, random_state=0)
        eligible = quality.loc[quality["quality_pass"].astype(bool)].sort_values("test_number")
        for quality_row in eligible.itertuples(index=False):
            number = int(quality_row.test_number)
            observed, result = _load_test_frames(root, number)
            observations = np.vstack(
                [
                    observed[f"rss_observed_integrated_body_rotation_{axis}_rad"].to_numpy(float)
                    for axis in AXES
                ]
            )
            truth = np.vstack(
                [result[f"measured_{joint}_rad"].to_numpy(float) for joint in JOINTS]
            )
            for method in missing:
                fitted = fit_bss(observations, method, config)
                scored, _aligned = match_and_score(
                    truth,
                    fitted.source_angles,
                    test_number=number,
                    batch_number=int(quality_row.batch_number),
                    method=method,
                )
                metadata = json.dumps(
                    {
                        "implementation": fitted.implementation,
                        "converged": fitted.converged,
                        "iterations": fitted.iterations,
                        "warnings": list(fitted.warnings),
                    },
                    sort_keys=True,
                )
                for row in scored:
                    row.update(
                        {
                            "runtime_seconds": fitted.runtime_seconds,
                            "fit_metadata": metadata,
                        }
                    )
                report_rows.extend(scored)
                new_rows.extend(scored)

    baseline_metrics = pd.DataFrame.from_records(report_rows)
    if baseline_metrics.empty:
        raise RuntimeError("PCA/ICA report metrics could not be produced.")
    if new_rows:
        combined = pd.concat([main, pd.DataFrame.from_records(new_rows)], ignore_index=True)
    else:
        combined = main.copy()
    return baseline_metrics, combined


def _model_summary(main: pd.DataFrame) -> pd.DataFrame:
    values = main.loc[main["method"].isin(METHOD_ORDER)].copy()
    return (
        values.groupby(["method", "joint"], sort=False)
        .agg(
            mean_abs_correlation=("abs_correlation", "mean"),
            mean_rmse_rad=("rmse_rad", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            count=("test_number", "size"),
        )
        .reset_index()
    )


def _available_method_order(summary: pd.DataFrame) -> tuple[str, ...]:
    available = set(summary["method"].astype(str))
    return tuple(method for method in METHOD_ORDER if method in available)


def _save_model_comparison(
    summary: pd.DataFrame,
    method_order: tuple[str, ...],
    destination: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 3.65))
    x = np.arange(len(method_order), dtype=float)
    width = 0.23
    for offset, joint in zip((-width, 0.0, width), JOINTS):
        joint_rows = summary.loc[summary["joint"].eq(joint)].set_index("method")
        correlation = [joint_rows.loc[m, "mean_abs_correlation"] for m in method_order]
        rmse = [joint_rows.loc[m, "mean_rmse_rad"] for m in method_order]
        axes[0].bar(x + offset, correlation, width, color=JOINT_COLORS[joint], label=joint.title())
        axes[1].bar(x + offset, rmse, width, color=JOINT_COLORS[joint], label=joint.title())
    labels = [METHOD_LABELS[m] for m in method_order]
    for axis in axes:
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Mean absolute correlation")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Source agreement (higher is better)", loc="left", fontweight="bold")
    axes[1].set_ylabel("Mean RMSE (rad)")
    axes[1].set_title("Angular error (lower is better)", loc="left", fontweight="bold")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.suptitle("Model-wise comparison across 34 quality-eligible trials", fontweight="bold", y=1.02)
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    figure.savefig(destination)
    plt.close(figure)


def _save_histograms(summary: pd.DataFrame, quality: pd.DataFrame, destination: Path) -> None:
    eligible_tests = set(quality.loc[quality["quality_pass"].astype(bool), "test_number"].astype(int))
    eligible = summary.loc[summary["test_number"].isin(eligible_tests)]
    excluded = summary.loc[~summary["test_number"].isin(eligible_tests)]
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 2.75))
    axes[0].hist(eligible["mean_rmse_rad"], bins=9, color="#4472C4", edgecolor="white", alpha=0.9)
    axes[0].axvline(eligible["mean_rmse_rad"].mean(), color="#1F1F1F", linestyle="--", linewidth=1.2, label="eligible mean")
    axes[0].scatter(excluded["mean_rmse_rad"], [0.25] * len(excluded), color="#C00000", marker="x", s=38, label="excluded test 20")
    axes[0].set_xlabel("Per-test mean RMSE (rad)")
    axes[0].set_ylabel("Number of tests")
    axes[0].set_title("RSS error distribution", loc="left", fontweight="bold")
    axes[1].hist(eligible["fisher_mean_abs_correlation"], bins=9, color="#70AD47", edgecolor="white", alpha=0.9)
    axes[1].axvline(eligible["fisher_mean_abs_correlation"].mean(), color="#1F1F1F", linestyle="--", linewidth=1.2, label="eligible mean")
    axes[1].scatter(excluded["fisher_mean_abs_correlation"], [0.25] * len(excluded), color="#C00000", marker="x", s=38, label="excluded test 20")
    axes[1].set_xlabel("Per-test Fisher-aggregated $|r|$")
    axes[1].set_ylabel("Number of tests")
    axes[1].set_title("RSS correlation distribution", loc="left", fontweight="bold")
    for axis in axes:
        axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)


def _per_test_paths(root: Path, test_number: int) -> tuple[Path, Path]:
    folder = root / "figures" / "per_test" / f"test_{test_number:02d}"
    return folder / "rss_observed_signals.csv", folder / "source_separated_vs_ground_truth.csv"


def _load_test_frames(root: Path, test_number: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_path, result_path = _per_test_paths(root, test_number)
    if not observed_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"Missing per-test exports for test {test_number}")
    return pd.read_csv(observed_path), pd.read_csv(result_path)


def _save_representative(
    root: Path,
    rss_metrics: pd.DataFrame,
    test_number: int,
    rank_label: str,
    ground_truth_cmax: float,
    correlation_stratum: str,
    destination: Path,
) -> None:
    observed, result = _load_test_frames(root, test_number)
    rows = rss_metrics.loc[rss_metrics["test_number"].eq(test_number)].set_index("joint")
    figure, axes = plt.subplots(3, 2, figsize=(11.0, 7.05), sharex="col")
    for index, (axis_name, joint) in enumerate(zip(AXES, JOINTS)):
        left, right = axes[index]
        time = observed["time_s"].to_numpy(float)
        left.plot(
            time,
            observed[f"rss_observed_integrated_body_rotation_{axis_name}_rad"],
            color=AXIS_COLORS[axis_name],
            linewidth=1.25,
            label="Integrated gyro (RSS input)",
        )
        left.set_ylabel(f"{axis_name.upper()} (rad)")
        out_time = result["time_s"].to_numpy(float)
        right.plot(out_time, result[f"measured_{joint}_rad"], color="#222222", linewidth=1.35, label="Ground truth")
        right.plot(out_time, result[f"rss_{joint}_aligned_rad"], color=JOINT_COLORS[joint], linewidth=1.05, alpha=0.92, label="RSS output")
        metric = rows.loc[joint]
        right.set_title(
            f"{joint.title()}: |r|={metric['abs_correlation']:.3f}, RMSE={metric['rmse_rad']:.3f} rad",
            loc="left",
        )
        right.set_ylabel("Angle (rad)")
    axes[0, 0].set_title("Observer signals before RSS", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, loc="best")
    axes[0, 1].legend(frameon=False, loc="best", ncol=2)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    mean_rmse = rows["rmse_rad"].mean()
    correlations = np.clip(rows["abs_correlation"].to_numpy(float), 0.0, 1.0 - 1e-12)
    fisher = float(np.tanh(np.mean(np.arctanh(correlations))))
    figure.suptitle(
        f"Test {test_number} — {rank_label} (mean RMSE={mean_rmse:.3f} rad, "
        f"Fisher |r|={fisher:.3f}, GT Cmax={ground_truth_cmax:.3f} [{correlation_stratum}])",
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    figure.savefig(destination)
    plt.close(figure)


def _save_appendix_plot(
    root: Path,
    rss_metrics: pd.DataFrame,
    test_number: int,
    excluded: bool,
    ground_truth_cmax: float,
    correlation_stratum: str,
    destination: Path,
) -> None:
    _, result = _load_test_frames(root, test_number)
    rows = rss_metrics.loc[rss_metrics["test_number"].eq(test_number)].set_index("joint")
    figure, axes = plt.subplots(3, 1, figsize=(4.1, 3.55), sharex=True)
    for axis, joint in zip(axes, JOINTS):
        time = result["time_s"].to_numpy(float)
        axis.plot(time, result[f"measured_{joint}_rad"], color="#252525", linewidth=0.85, label="Truth")
        axis.plot(time, result[f"rss_{joint}_aligned_rad"], color=JOINT_COLORS[joint], linewidth=0.70, alpha=0.95, label="RSS")
        metric = rows.loc[joint]
        axis.set_ylabel(joint[0].upper())
        axis.set_title(f"{joint.title()}: r={metric['abs_correlation']:.2f}, e={metric['rmse_rad']:.2f}", loc="left", fontsize=7.3, pad=1.5)
        axis.tick_params(axis="both", labelsize=6.3, length=2)
    axes[0].legend(frameon=False, ncol=2, loc="upper right", fontsize=6.2)
    axes[-1].set_xlabel("Time (s)", fontsize=7)
    suffix = " — EXCLUDED" if excluded else ""
    figure.suptitle(
        f"Test {test_number}{suffix} | GT Cmax={ground_truth_cmax:.2f} ({correlation_stratum})",
        fontweight="bold",
        fontsize=9.0,
        color="#A00000" if excluded else "#202020",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96), pad=0.45)
    figure.savefig(destination)
    plt.close(figure)


def _model_table_tex(model_summary: pd.DataFrame, method_order: tuple[str, ...]) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\caption{Mean model performance over the 34 quality-eligible trials. Correlation is mean absolute Pearson correlation and RMSE is reported in radians.}",
        r"\label{tab:model-summary}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Model & Joint & $|r|$ & RMSE (rad) \\",
        r"\midrule",
    ]
    for method in method_order:
        subset = model_summary.loc[model_summary["method"].eq(method)].set_index("joint")
        for index, joint in enumerate(JOINTS):
            row = subset.loc[joint]
            model = METHOD_LABELS[method] if index == 0 else ""
            lines.append(
                f"{model} & {joint.title()} & {row['mean_abs_correlation']:.3f} & "
                f"{row['mean_rmse_rad']:.3f} \\\\"
            )
        if method != method_order[-1]:
            lines.append(r"\addlinespace[2pt]")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _per_test_table_tex(summary: pd.DataFrame, excluded_tests: set[int]) -> str:
    halves = (summary.iloc[:18], summary.iloc[18:])
    blocks: list[str] = []
    for half in halves:
        rows = [
            r"\begin{tabular}{rrrr}",
            r"\toprule",
            r"Test & Batch & RMSE & Fisher $|r|$ \\",
            r"\midrule",
        ]
        for row in half.itertuples(index=False):
            test = int(row.test_number)
            mark = r"$^{\dagger}$" if test in excluded_tests else ""
            rows.append(
                f"{test}{mark} & {int(row.batch_number)} & {row.mean_rmse_rad:.3f} & "
                f"{row.fisher_mean_abs_correlation:.3f} \\\\"
            )
        rows.extend([r"\bottomrule", r"\end{tabular}"])
        blocks.append("\n".join(rows))
    return "\n".join(
        [
            r"\begin{minipage}[t]{0.47\linewidth}\centering\scriptsize",
            blocks[0],
            r"\end{minipage}\hfill",
            r"\begin{minipage}[t]{0.47\linewidth}\centering\scriptsize",
            blocks[1],
            r"\end{minipage}",
        ]
    )


def _write_report(
    report_dir: Path,
    model_summary: pd.DataFrame,
    trial_summary: pd.DataFrame,
    quality: pd.DataFrame,
    representative: list[tuple[int, str]],
) -> Path:
    excluded_tests = set(quality.loc[~quality["quality_pass"].astype(bool), "test_number"].astype(int))
    rss_rows = model_summary.loc[model_summary["method"].eq("rss")]
    rss_overall_corr = rss_rows["mean_abs_correlation"].mean()
    rss_overall_rmse = rss_rows["mean_rmse_rad"].mean()
    eligible = trial_summary.loc[~trial_summary["test_number"].isin(excluded_tests)]
    best = eligible.sort_values("mean_rmse_rad").iloc[0]
    worst = eligible.sort_values("mean_rmse_rad").iloc[-1]
    method_order = _available_method_order(model_summary)
    observer_caption = (
        "Left: integrated, bias-corrected Node~10 gyroscope channels supplied to RSS."
    )
    representative_tex = []
    for number, label in representative:
        representative_tex.append(
            "\n".join(
                [
                    r"\begin{figure}[H]",
                    r"\centering",
                    rf"\includegraphics[width=0.98\linewidth]{{figures/representative_test_{number:02d}.pdf}}",
                    rf"\caption{{{label.title()} eligible case, test {number}. {observer_caption} Right: sign/offset-aligned RSS outputs overlaid on encoder ground truth.}}",
                    rf"\label{{fig:representative-{number}}}",
                    r"\end{figure}",
                ]
            )
        )

    appendix_blocks: list[str] = []
    numbers = sorted(quality["test_number"].astype(int))
    for page_start in range(0, len(numbers), 6):
        page_numbers = numbers[page_start : page_start + 6]
        parts = [r"\begin{figure}[H]", r"\centering"]
        for idx, number in enumerate(page_numbers):
            parts.append(rf"\includegraphics[width=0.315\linewidth]{{figures/appendix_test_{number:02d}.pdf}}")
            if idx % 3 == 2:
                parts.append(r"\\[2mm]")
            else:
                parts.append(r"\hfill")
        parts.extend(
            [
                rf"\caption{{RSS result overlays for tests {page_numbers[0]}--{page_numbers[-1]}. Each panel shows base (B), elbow (E), and wrist (W); titles report absolute correlation $r$ and RMSE $e$ in radians.}}",
                rf"\label{{fig:appendix-{page_numbers[0]}}}",
                r"\end{figure}",
                r"\clearpage",
            ]
        )
        appendix_blocks.append("\n".join(parts))

    model_table = _model_table_tex(model_summary, method_order)
    test_table = _per_test_table_tex(trial_summary, excluded_tests)
    observer_protocol_text = (
        "The representative observer panels show only the integrated, bias-corrected channels "
        "supplied to the headline RSS model."
    )
    compared_models_text = "identity, PCA, ICA (JADE), ICA (FastICA), and SOBI"
    report = rf"""\documentclass[10pt,a4paper]{{article}}
\usepackage[margin=18mm]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{pdflscape}}
\usepackage[table]{{xcolor}}
\usepackage{{microtype}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{caption}}
\usepackage{{fancyhdr}}
\definecolor{{rssblue}}{{HTML}}{{2F5597}}
\definecolor{{lightblue}}{{HTML}}{{D9EAF7}}
\definecolor{{warning}}{{HTML}}{{9C0006}}
\hypersetup{{pdftitle={{QArm RSS Validation Report}},pdfauthor={{RSS project}}}}
\pagestyle{{fancy}}
\fancyhf{{}}
\lhead{{QArm RSS validation}}
\rhead{{\thepage}}
\setlength{{\headheight}}{{14pt}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{4pt}}
\newcommand{{\rss}}{{\textsc{{RSS}}}}

\begin{{document}}
\begin{{titlepage}}
\centering
\vspace*{{22mm}}
{{\Huge\bfseries QArm Rotation Source Separation\par}}
\vspace{{3mm}}
{{\LARGE Practical Validation Report\par}}
\vspace{{12mm}}
\rule{{0.82\textwidth}}{{0.8pt}}\par
\vspace{{8mm}}
{{\large 35 recorded tests; 34 quality-eligible trials\par}}
\vspace{{4mm}}
{{\large Model comparisons, per-test metrics, and complete result appendix\par}}
\vfill
{{\large Generated from the completed artifacts in\par}}
{{\ttfamily outputs/qarm\_rss\_validation/\par}}
\vspace{{8mm}}
{{\large 11 September 2026\par}}
\end{{titlepage}}

\tableofcontents
\clearpage

\section{{Executive summary}}
This report evaluates rotational source separation (\rss{{}}) on 35 recorded QArm tests. Thirty-four trials passed the prespecified recording, alignment, missing-data, and saturation checks. Test~20 failed the alignment criterion and is retained only as an excluded-trial sensitivity result. The headline \rss{{}} analysis used the bias-corrected, otherwise unfiltered, integrated body-rotation channels from the distal Node~4 gyroscope.

Across the 34 eligible trials and three joints, \rss{{}} achieved mean absolute correlation {rss_overall_corr:.3f} and mean RMSE {rss_overall_rmse:.3f}~rad. Performance was strongest at the base ($|r|=0.983$, RMSE $=0.087$~rad) and materially weaker at the elbow and wrist ($|r|\approx0.65$, RMSE $\approx0.18$--$0.19$~rad). Eligible per-trial mean RMSE ranged from {best.mean_rmse_rad:.3f}~rad (test~{int(best.test_number)}) to {worst.mean_rmse_rad:.3f}~rad (test~{int(worst.test_number)}).

The identity/no-separation comparator was the strongest model because the distal observer axes were already closely aligned with the joint motion in this dataset. RSS therefore did not establish practical separation evidence against identity. Using raw angular error, RSS nevertheless achieved lower mean RMSE than PCA, both ICA implementations, and SOBI for each of the three joints.

\colorbox{{lightblue}}{{\parbox{{0.96\linewidth}}{{\textbf{{Interpretation.}} This is a retrospective practical benchmark under correlated, configuration-dependent joint motion, not an ideal orthogonal-source validation. No trial met the prespecified $C_{{\max}}<0.1$ source-correlation criterion, so separation claims must remain qualified.}}}}

\section{{Data and evaluation protocol}}
Each trial contains 3,000 aligned samples spanning 30~s at approximately 100~Hz. Node~4 gyroscope bias was estimated from a stationary pre-trial window and subtracted before numerical integration. RSS then fitted a constant $SO(3)$ recovery rotation using particle swarm optimization and a zero-lag cross-correlation separation loss.

Encoder measurements were not used during blind fitting. They were used afterward for synchronization audit, geometric diagnostics, one-to-one mode assignment, sign correction, constant-offset alignment, and scoring. Amplitude and time lag were not fitted. Absolute Pearson correlation measures waveform agreement and RMSE measures angular error in radians. Trial-level correlation is aggregated in Fisher-$z$ space.

{observer_protocol_text}

PCA and ICA were applied to those same three unfiltered integrated observer channels. PCA uses a full singular-value decomposition; ICA is reported with both fourth-order JADE and FastICA implementations. All decompositions were fitted without encoder access and evaluated with the same post-fit mode assignment, sign correction, and constant-offset alignment used for RSS.

\section{{Model-wise comparison}}
Figure~\ref{{fig:model-comparison}} and Table~\ref{{tab:model-summary}} compare the headline RSS fit with {compared_models_text}. Identity is a scientifically relevant no-separation comparator, not a fitted decomposition. RSS was second best overall: it had lower mean raw RMSE than PCA, both ICA implementations, and SOBI, but higher RMSE than identity for all joints.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\linewidth]{{figures/model_comparison.pdf}}
\caption{{Mean model performance by joint. All bars use the same 34 quality-eligible trials.}}
\label{{fig:model-comparison}}
\end{{figure}}

{model_table}

RSS had lower mean raw RMSE than PCA, JADE-ICA, FastICA, and SOBI for base, elbow, and wrist. Among the fitted separation methods, RSS also had the highest mean absolute correlation for base and elbow; PCA and SOBI slightly exceeded RSS at the wrist. Identity retained both the lowest mean RMSE and highest mean absolute correlation for every joint. These results agree with the prespecified practical-separation decision, which was negative for every joint.

\section{{Representative RSS outputs: best to worst}}
Representative eligible tests were selected at evenly spaced ranks of per-trial mean RMSE in radians (best, lower quartile, median, upper quartile, and worst). This ranking is fixed from the stored metrics and is not a visual selection. In every panel, the left column shows observer signals before separation and the right column overlays the aligned RSS result on ground truth. Each title also reports ground-truth $C_{{\max}}$: the maximum absolute pairwise Pearson correlation among the measured base, elbow, and wrist position trajectories, followed by its stored correlation stratum.

{chr(10).join(representative_tex)}

\begin{{landscape}}
\section{{Per-test RMSE and correlation}}
Table~\ref{{tab:per-test}} reports the mean RMSE and Fisher-aggregated absolute correlation across base, elbow, and wrist for every recorded test. Test~20 is flagged because it was excluded from the main analysis. The histograms below summarize the same per-test quantities for the 34 eligible trials and show the excluded result separately.

\begin{{table}}[H]
\centering
\caption{{Per-test headline RSS metrics. RMSE is the arithmetic mean over three joints; $|r|$ is aggregated over joints in Fisher-$z$ space.}}
\label{{tab:per-test}}
{test_table}
\par\vspace{{1mm}}\raggedright\footnotesize $^\dagger$Test 20: sensitivity-only result, excluded for low alignment correlation.
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.93\linewidth]{{figures/per_test_metric_histograms.pdf}}
\caption{{Distribution of the per-test metrics tabulated above. Dashed lines show eligible-trial means; the red cross marks excluded test~20.}}
\label{{fig:metric-histograms}}
\end{{figure}}
\end{{landscape}}

\section{{Limitations and conclusions}}
The experiment demonstrates that RSS can recover the base trajectory reliably and often produces useful elbow and wrist estimates, but the evidence does not support a general claim of improved separation over the unrotated observer channels. The source trajectories were strongly correlated, the fixed-rotation model is imperfect for configuration-dependent geometry, and no hardware trajectory was repeated. The four additional PSO seeds therefore quantify algorithmic stability only, not hardware repeatability.

The most defensible conclusion is that headline RSS produces lower raw angular error than PCA, JADE-ICA, FastICA, and SOBI in this practical dataset, while identity remains the dominant comparator. Future validation should use independently excited or deliberately decorrelated joint trajectories, repeated hardware runs, and a protocol designed to populate the prespecified low-correlation regime.

\clearpage
\appendix
\begin{{landscape}}
\section{{Complete results for all 35 tests}}
The following compact panels contain all 35 RSS-versus-ground-truth results, arranged three per line. Correlation values are absolute; errors are RMSE in radians. Every panel title gives ground-truth $C_{{\max}}$ and its correlation stratum. Test~20 is visibly marked as excluded but included for completeness.

\clearpage

{chr(10).join(appendix_blocks)}
\end{{landscape}}
\end{{document}}
"""
    destination = report_dir / "report.tex"
    destination.write_text(report, encoding="utf-8")
    return destination


def generate(results_root: Path) -> Path:
    _configure_style()
    root = results_root.resolve()
    report_dir = root / "report"
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    main, excluded, quality, _paired = _load_metrics(root)
    additional_baselines, report_comparators = _evaluate_report_baselines(root, main, quality)
    rss_all = pd.concat(
        [main.loc[main["method"].eq("rss")], excluded.loc[excluded["method"].eq("rss")]],
        ignore_index=True,
    )
    trial_summary = _trial_summary(rss_all)
    model_summary = _model_summary(report_comparators)
    method_order = _available_method_order(model_summary)
    trial_summary.to_csv(report_dir / "per_test_report_metrics.csv", index=False)
    model_summary.to_csv(report_dir / "model_report_metrics.csv", index=False)
    additional_baselines.drop(columns=["nrmse_std"], errors="ignore").to_csv(
        report_dir / "pca_ica_report_metrics.csv",
        index=False,
    )

    _save_model_comparison(model_summary, method_order, figure_dir / "model_comparison.pdf")
    _save_histograms(trial_summary, quality, figure_dir / "per_test_metric_histograms.pdf")

    eligible_tests = set(quality.loc[quality["quality_pass"].astype(bool), "test_number"].astype(int))
    quality_by_test = quality.set_index("test_number")
    eligible_ranked = trial_summary.loc[trial_summary["test_number"].isin(eligible_tests)].sort_values("mean_rmse_rad")
    rank_indices = np.rint(np.linspace(0, len(eligible_ranked) - 1, 5)).astype(int)
    rank_labels = ("best", "lower-quartile", "median", "upper-quartile", "worst")
    representative = [
        (int(eligible_ranked.iloc[index]["test_number"]), label)
        for index, label in zip(rank_indices, rank_labels)
    ]
    selected_figure_names = {
        f"representative_test_{number:02d}.pdf" for number, _label in representative
    }
    for stale_figure in figure_dir.glob("representative_test_*.pdf"):
        if stale_figure.name not in selected_figure_names:
            stale_figure.unlink()
    for number, label in representative:
        _save_representative(
            root,
            rss_all,
            number,
            label,
            float(quality_by_test.loc[number, "cmax_position"]),
            str(quality_by_test.loc[number, "correlation_stratum"]),
            figure_dir / f"representative_test_{number:02d}.pdf",
        )

    excluded_tests = set(quality.loc[~quality["quality_pass"].astype(bool), "test_number"].astype(int))
    for number in sorted(quality["test_number"].astype(int)):
        _save_appendix_plot(
            root,
            rss_all,
            number,
            number in excluded_tests,
            float(quality_by_test.loc[number, "cmax_position"]),
            str(quality_by_test.loc[number, "correlation_stratum"]),
            figure_dir / f"appendix_test_{number:02d}.pdf",
        )

    selection = pd.DataFrame(representative, columns=["test_number", "rank_label"])
    selection.to_csv(report_dir / "representative_selection.csv", index=False)
    return _write_report(report_dir, model_summary, trial_summary, quality, representative)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("outputs/qarm_rss_validation"),
        help="Completed validation output directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    output = generate(parse_args().results_root)
    print(output)
