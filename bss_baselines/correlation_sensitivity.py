"""Inter-source correlation sensitivity sweep for RSS and the observer-signal baselines.

RSS minimizes the zero-lag cross-correlation between separated sources, so source
independence is one of its modelling assumptions rather than an incidental property of
the simulator. This module sweeps a prescribed inter-source correlation, runs RSS and
every configured baseline on identical observations, and estimates how far the
correlation can rise before a method's accuracy degrades past a stated criterion.

The design is paired: all correlation levels inside one ``(family, replicate)`` block
share a seed, so the waveforms, source axes, observer axes, and mixing matrix are
identical across levels and the correlation is the only factor that varies. Every
bootstrap therefore resamples whole blocks, never individual rows.

This module lives in ``bss_baselines`` because it drives both ``rss.evaluation`` and
``bss_baselines.evaluation``; ``rss`` does not depend on ``bss_baselines``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd

from bss_baselines.core import BSSConfig
from bss_baselines.evaluation import (
    BSSValidationResult,
    build_method_comparison_frames,
    run_bss_methods_for_scenario,
)
from rss.evaluation import ValidationResult, describe_metric_distributions, run_validation_for_scenario
from rss.simulation import (
    BatchStudyConfig,
    CorrelationCaseSpec,
    CorrelationStudyConfig,
    ScenarioData,
    SimulationConfig,
    create_correlation_case_specs,
    generate_correlation_scenario,
    source_cross_correlation_matrix,
)
from rss.solver import RSSConfig


CORRELATION_COLUMN = "target_source_correlation"
BLOCK_COLUMN = "block_id"

# Axis error leads because it is the endpoint that stays stable under the optimizer's
# run-to-run branch selection, and because it does not depend on the correlation-based
# source matching that becomes ambiguous once the true sources are themselves correlated.
DEFAULT_CORRELATION_CASE_METRICS: tuple[str, ...] = (
    "mean_axis_error_deg",
    "mean_source_relative_rmse_oracle_scaled",
    "mean_source_relative_rmse",
    "mean_abs_source_correlation",
    "quaternion_reconstruction_mean_error_deg",
)

CRITERION_ABSOLUTE = "absolute"
CRITERION_RELATIVE_DEGRADATION = "relative_degradation"

STATUS_ATTAINED = "attained"
STATUS_FLOOR_EXCEEDED = "not attained at rho=0"
STATUS_ABOVE_GRID = "above grid max"
STATUS_INSUFFICIENT = "insufficient data"


@dataclass(frozen=True)
class CorrelationSweepConfig:
    """Controls for the correlation sensitivity sweep and its statistics."""

    study: CorrelationStudyConfig = field(default_factory=CorrelationStudyConfig)
    include_rss: bool = True
    case_metric_columns: tuple[str, ...] = DEFAULT_CORRELATION_CASE_METRICS
    primary_metric: str = "mean_axis_error_deg"
    absolute_thresholds: tuple[float, ...] = (0.01,)
    relative_degradations: tuple[float, ...] = (0.01,)
    bootstrap_resamples: int = 2000
    bootstrap_confidence: float = 0.95
    bootstrap_seed: int = 0
    retain_case_results: bool = False

    def __post_init__(self) -> None:
        if not self.case_metric_columns:
            raise ValueError("case_metric_columns must contain at least one metric.")
        if len(set(self.case_metric_columns)) != len(self.case_metric_columns):
            raise ValueError("case_metric_columns must not contain duplicates.")
        if self.primary_metric not in self.case_metric_columns:
            raise ValueError("primary_metric must appear in case_metric_columns.")
        if int(self.bootstrap_resamples) < 1:
            raise ValueError("bootstrap_resamples must be >= 1.")
        confidence = float(self.bootstrap_confidence)
        if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("bootstrap_confidence must be finite and strictly between 0 and 1.")
        for name, values in (
            ("absolute_thresholds", self.absolute_thresholds),
            ("relative_degradations", self.relative_degradations),
        ):
            if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in values):
                raise ValueError(f"{name} entries must be finite and > 0.")


@dataclass(frozen=True)
class CorrelationCaseOutcome:
    """One generated scenario together with every method fitted to it."""

    spec: CorrelationCaseSpec
    scenario: ScenarioData
    realized_mean_abs_source_correlation: float
    realized_max_abs_source_correlation: float
    rss_result: ValidationResult | None
    bss_results: tuple[BSSValidationResult, ...]


@dataclass(frozen=True)
class CorrelationSweepResult:
    """Frames and threshold estimates produced by one sweep."""

    config: CorrelationSweepConfig
    specs: tuple[CorrelationCaseSpec, ...]
    case_frame: pd.DataFrame
    source_frame: pd.DataFrame
    case_summary: pd.DataFrame
    thresholds: pd.DataFrame
    paired_tests: pd.DataFrame
    outcomes: tuple[CorrelationCaseOutcome, ...] = ()


def run_correlation_case(
    spec: CorrelationCaseSpec,
    rss_config: RSSConfig | None,
    bss_config: BSSConfig | None = None,
    *,
    base_config: SimulationConfig | None = None,
    study_config: BatchStudyConfig | None = None,
    correlation_tolerance: float = 1e-9,
) -> CorrelationCaseOutcome:
    """Generate one correlation case and fit RSS (optional) plus every baseline to it."""

    scenario = generate_correlation_scenario(
        spec,
        base_config,
        study_config,
        correlation_tolerance=float(correlation_tolerance),
    )
    active = np.asarray(scenario.active_source_indices, dtype=int)
    correlation = source_cross_correlation_matrix(scenario.sources[active])
    upper = np.triu_indices(len(active), k=1)
    pairwise = np.abs(correlation[upper]) if upper[0].size else np.zeros(0, dtype=float)
    rss_result = (
        run_validation_for_scenario(scenario, rss_config) if rss_config is not None else None
    )
    bss_results = tuple(run_bss_methods_for_scenario(scenario, bss_config))
    return CorrelationCaseOutcome(
        spec=spec,
        scenario=scenario,
        realized_mean_abs_source_correlation=float(np.mean(pairwise)) if pairwise.size else float("nan"),
        realized_max_abs_source_correlation=float(np.max(pairwise)) if pairwise.size else float("nan"),
        rss_result=rss_result,
        bss_results=bss_results,
    )


def attach_correlation_columns(
    frame: pd.DataFrame,
    outcomes: Sequence[CorrelationCaseOutcome],
) -> pd.DataFrame:
    """Merge the design columns of a sweep onto a metrics frame keyed by ``case_id``.

    The target correlation is attached here rather than carried on ``ScenarioData`` so
    that the schema of every pre-existing suite is untouched.
    """

    if frame.empty:
        return frame.copy()
    if "case_id" not in frame.columns:
        raise ValueError("frame must contain a 'case_id' column.")
    design = {
        outcome.spec.case_id: {
            CORRELATION_COLUMN: float(outcome.spec.target_source_correlation),
            BLOCK_COLUMN: outcome.spec.block_id,
            "realized_mean_abs_source_correlation": outcome.realized_mean_abs_source_correlation,
            "realized_max_abs_source_correlation": outcome.realized_max_abs_source_correlation,
        }
        for outcome in outcomes
    }
    identifiers = frame["case_id"].astype(str)
    unmapped = sorted(set(identifiers) - set(design))
    if unmapped:
        raise ValueError(f"No correlation design entry for case ids: {unmapped}.")
    attached = frame.copy()
    for column in (
        CORRELATION_COLUMN,
        BLOCK_COLUMN,
        "realized_mean_abs_source_correlation",
        "realized_max_abs_source_correlation",
    ):
        attached[column] = [design[identifier][column] for identifier in identifiers]
    return attached


def block_bootstrap_interval(
    values: np.ndarray,
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    random_state: int = 0,
    statistic: str = "mean",
) -> tuple[float, float, float]:
    """Return ``(point, lower, upper)`` from a percentile bootstrap over independent units.

    Callers must pass one value per block; blocks are the independent unit of this design
    because every correlation level inside a block shares its scenario.
    """

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan"), float("nan")
    if statistic not in {"mean", "geometric_mean", "median"}:
        raise ValueError("statistic must be 'mean', 'geometric_mean', or 'median'.")

    def _apply(sample: np.ndarray) -> float:
        if statistic == "mean":
            return float(np.mean(sample))
        if statistic == "median":
            return float(np.median(sample))
        positive = sample[sample > 0.0]
        if positive.size == 0:
            return float("nan")
        return float(np.exp(np.mean(np.log(positive))))

    point = _apply(array)
    if array.size == 1:
        return point, point, point
    rng = np.random.default_rng(int(random_state))
    draws = rng.integers(0, array.size, size=(int(resamples), array.size))
    estimates = np.array([_apply(array[row]) for row in draws], dtype=float)
    estimates = estimates[np.isfinite(estimates)]
    if estimates.size == 0:
        return point, float("nan"), float("nan")
    alpha = (1.0 - float(confidence)) / 2.0
    lower = float(np.quantile(estimates, alpha))
    upper = float(np.quantile(estimates, 1.0 - alpha))
    return point, lower, upper


def summarize_correlation_sensitivity(
    frame: pd.DataFrame,
    *,
    metric_columns: Sequence[str],
    group_columns: Sequence[str] = ("method",),
    x_column: str = CORRELATION_COLUMN,
    block_column: str = BLOCK_COLUMN,
    resamples: int = 2000,
    confidence: float = 0.95,
    random_state: int = 0,
) -> pd.DataFrame:
    """Describe each metric per method and correlation level, with block-bootstrap bounds.

    Descriptive statistics come from :func:`rss.evaluation.describe_metric_distributions`
    so this suite reports them exactly the way every other suite does; only the
    confidence bounds are added here.
    """

    if frame.empty:
        return pd.DataFrame()
    grouping = (*group_columns, x_column)
    summary = describe_metric_distributions(
        frame,
        group_columns=tuple(grouping),
        metric_columns=tuple(metric_columns),
    )
    if summary.empty:
        return summary
    bounds: list[dict[str, object]] = []
    for index, (keys, group) in enumerate(frame.groupby(list(grouping), sort=False, dropna=False)):
        identifiers = dict(zip(grouping, keys if isinstance(keys, tuple) else (keys,)))
        for metric in metric_columns:
            # One value per block keeps the bootstrap unit independent.
            if block_column in group.columns:
                per_block = group.groupby(block_column, sort=False)[metric].mean().to_numpy(dtype=float)
            else:
                per_block = group[metric].to_numpy(dtype=float)
            point, lower, upper = block_bootstrap_interval(
                per_block,
                resamples=int(resamples),
                confidence=float(confidence),
                random_state=int(random_state) + index,
            )
            bounds.append(
                {
                    **identifiers,
                    "metric": metric,
                    "block_count": int(np.sum(np.isfinite(per_block))),
                    "block_mean": point,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    if not bounds:
        return summary
    return summary.merge(pd.DataFrame(bounds), on=[*grouping, "metric"], how="left")


def estimate_threshold_crossing(
    x_values: np.ndarray,
    y_values: np.ndarray,
    threshold: float,
) -> tuple[float, str]:
    """Return the first upcrossing of ``threshold`` and its status.

    The threshold is defined as ``sup{x0 : y(x) < threshold for all x <= x0}`` — the
    largest correlation that is still safe *throughout*, which is the operationally
    correct reading and stays well defined when the response is not monotone.
    Interpolation is logarithmic when the bracketing points are both positive, matching
    the log-spaced design grid.
    """

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x_values and y_values must have the same shape.")
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size == 0:
        return float("nan"), STATUS_INSUFFICIENT
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    limit = float(threshold)
    if not math.isfinite(limit):
        return float("nan"), STATUS_INSUFFICIENT
    if y[0] >= limit:
        return float("nan"), STATUS_FLOOR_EXCEEDED
    crossings = np.nonzero(y >= limit)[0]
    if crossings.size == 0:
        return float(x[-1]), STATUS_ABOVE_GRID
    index = int(crossings[0])
    x_low, x_high = float(x[index - 1]), float(x[index])
    y_low, y_high = float(y[index - 1]), float(y[index])
    if y_high == y_low:
        return x_low, STATUS_ATTAINED
    fraction = (limit - y_low) / (y_high - y_low)
    if x_low > 0.0 and x_high > 0.0:
        crossing = 10.0 ** (
            math.log10(x_low) + fraction * (math.log10(x_high) - math.log10(x_low))
        )
    else:
        crossing = x_low + fraction * (x_high - x_low)
    return float(crossing), STATUS_ATTAINED


def _block_threshold(
    block: pd.DataFrame,
    metric: str,
    criterion: str,
    value: float,
    x_column: str,
) -> tuple[float, str, float]:
    ordered = block.sort_values(x_column, kind="stable")
    x = ordered[x_column].to_numpy(dtype=float)
    y = ordered[metric].to_numpy(dtype=float)
    baseline_mask = np.isclose(x, 0.0)
    baseline = float(y[baseline_mask][0]) if baseline_mask.any() and np.isfinite(y[baseline_mask][0]) else float("nan")
    if criterion == CRITERION_ABSOLUTE:
        limit = float(value)
    elif criterion == CRITERION_RELATIVE_DEGRADATION:
        # A zero baseline makes a proportional criterion degenerate rather than strict,
        # so it is reported as unusable instead of as an immediately exceeded floor.
        if not math.isfinite(baseline) or baseline <= 0.0:
            return float("nan"), STATUS_INSUFFICIENT, baseline
        limit = baseline * (1.0 + float(value))
    else:
        raise ValueError(f"Unknown criterion {criterion!r}.")
    crossing, status = estimate_threshold_crossing(x, y, limit)
    return crossing, status, baseline


def estimate_correlation_thresholds(
    frame: pd.DataFrame,
    *,
    metric_columns: Sequence[str],
    absolute_thresholds: Sequence[float] = (0.01,),
    relative_degradations: Sequence[float] = (0.01,),
    group_columns: Sequence[str] = ("method",),
    x_column: str = CORRELATION_COLUMN,
    block_column: str = BLOCK_COLUMN,
    resamples: int = 2000,
    confidence: float = 0.95,
    random_state: int = 0,
) -> pd.DataFrame:
    """Estimate, per method and metric, the largest correlation meeting each criterion.

    Each block yields its own crossing; those are summarized by a geometric mean with a
    block bootstrap interval, plus the 5th percentile as a design-to value. Blocks whose
    error already exceeds the criterion at zero correlation are reported as such rather
    than folded into the estimate.
    """

    if frame.empty:
        return pd.DataFrame()
    required = {*group_columns, x_column, block_column, *metric_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing threshold columns: {missing}.")
    criteria: list[tuple[str, float]] = [
        (CRITERION_ABSOLUTE, float(value)) for value in absolute_thresholds
    ]
    criteria.extend((CRITERION_RELATIVE_DEGRADATION, float(value)) for value in relative_degradations)

    rows: list[dict[str, object]] = []
    seed_offset = 0
    for keys, group in frame.groupby(list(group_columns), sort=False, dropna=False):
        identifiers = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        for metric in metric_columns:
            for criterion, value in criteria:
                crossings: list[float] = []
                statuses: list[str] = []
                baselines: list[float] = []
                for _, block in group.groupby(block_column, sort=False):
                    crossing, status, baseline = _block_threshold(
                        block, metric, criterion, value, x_column
                    )
                    statuses.append(status)
                    baselines.append(baseline)
                    if status in {STATUS_ATTAINED, STATUS_ABOVE_GRID}:
                        crossings.append(crossing)
                seed_offset += 1
                usable = np.asarray(crossings, dtype=float)
                usable = usable[np.isfinite(usable)]
                positive = usable[usable > 0.0]
                if positive.size:
                    point, lower, upper = block_bootstrap_interval(
                        positive,
                        resamples=int(resamples),
                        confidence=float(confidence),
                        random_state=int(random_state) + seed_offset,
                        statistic="geometric_mean",
                    )
                    percentile_05 = float(np.quantile(positive, 0.05))
                else:
                    point = lower = upper = percentile_05 = float("nan")
                block_total = len(statuses)
                attained = sum(1 for status in statuses if status == STATUS_ATTAINED)
                floor_exceeded = sum(1 for status in statuses if status == STATUS_FLOOR_EXCEEDED)
                above_grid = sum(1 for status in statuses if status == STATUS_ABOVE_GRID)
                if attained:
                    overall = STATUS_ATTAINED
                elif above_grid:
                    overall = STATUS_ABOVE_GRID
                elif floor_exceeded:
                    overall = STATUS_FLOOR_EXCEEDED
                else:
                    overall = STATUS_INSUFFICIENT
                finite_baselines = [item for item in baselines if math.isfinite(item)]
                rows.append(
                    {
                        **identifiers,
                        "metric": metric,
                        "criterion": criterion,
                        "criterion_value": float(value),
                        "threshold_correlation": point,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "p05_correlation": percentile_05,
                        "status": overall,
                        "block_count": block_total,
                        "blocks_attained": attained,
                        "blocks_above_grid": above_grid,
                        "blocks_floor_exceeded": floor_exceeded,
                        "baseline_error_at_zero": (
                            float(np.mean(finite_baselines)) if finite_baselines else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _wilcoxon_signed_rank(differences: np.ndarray) -> tuple[float, float, float, float]:
    """Return ``(statistic, p_value, hodges_lehmann, rank_biserial)`` for paired data."""

    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    nonzero = values[values != 0.0]
    if nonzero.size == 0:
        return float("nan"), float("nan"), 0.0 if values.size else float("nan"), float("nan")
    try:
        from scipy.stats import wilcoxon

        statistic, p_value = wilcoxon(nonzero, alternative="two-sided")
        statistic, p_value = float(statistic), float(p_value)
    except Exception:  # pragma: no cover - scipy is a declared dependency
        statistic, p_value = float("nan"), float("nan")
    ranks = np.argsort(np.argsort(np.abs(nonzero))) + 1.0
    positive = float(np.sum(ranks[nonzero > 0.0]))
    negative = float(np.sum(ranks[nonzero < 0.0]))
    total = positive + negative
    rank_biserial = (positive - negative) / total if total > 0.0 else float("nan")
    walsh = (nonzero[:, None] + nonzero[None, :]) / 2.0
    hodges_lehmann = float(np.median(walsh[np.triu_indices(nonzero.size, k=0)]))
    return statistic, p_value, hodges_lehmann, rank_biserial


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm step-down adjusted p-values, preserving input order."""

    values = [(index, float(value)) for index, value in enumerate(p_values)]
    finite = [item for item in values if math.isfinite(item[1])]
    adjusted = [float("nan")] * len(values)
    if not finite:
        return adjusted
    finite.sort(key=lambda item: item[1])
    count = len(finite)
    running = 0.0
    for position, (index, value) in enumerate(finite):
        candidate = (count - position) * value
        running = max(running, min(candidate, 1.0))
        adjusted[index] = running
    return adjusted


def paired_method_comparisons(
    frame: pd.DataFrame,
    *,
    metric: str,
    reference_method: str = "RSS",
    x_column: str = CORRELATION_COLUMN,
    block_column: str = BLOCK_COLUMN,
    confirmatory_correlations: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Compare the reference method against every other method at each correlation level.

    Contrasts are per-block log ratios, which is the natural paired statistic for a
    multiplicative error response. Holm correction is applied across the confirmatory
    family only; every other level is reported as exploratory with unadjusted p-values.
    """

    if frame.empty:
        return pd.DataFrame()
    required = {"method", metric, x_column, block_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing comparison columns: {missing}.")
    methods = [name for name in dict.fromkeys(frame["method"].astype(str)) if name != reference_method]
    if not methods:
        return pd.DataFrame()
    confirmatory = (
        None if confirmatory_correlations is None else [float(value) for value in confirmatory_correlations]
    )
    rows: list[dict[str, object]] = []
    for correlation, level in frame.groupby(x_column, sort=True):
        reference = (
            level.loc[level["method"].astype(str) == reference_method]
            .groupby(block_column, sort=False)[metric]
            .mean()
        )
        for method in methods:
            other = (
                level.loc[level["method"].astype(str) == method]
                .groupby(block_column, sort=False)[metric]
                .mean()
            )
            shared = reference.index.intersection(other.index)
            if shared.empty:
                continue
            first = reference.loc[shared].to_numpy(dtype=float)
            second = other.loc[shared].to_numpy(dtype=float)
            valid = np.isfinite(first) & np.isfinite(second) & (first > 0.0) & (second > 0.0)
            if not valid.any():
                continue
            differences = np.log10(first[valid]) - np.log10(second[valid])
            statistic, p_value, hodges_lehmann, rank_biserial = _wilcoxon_signed_rank(differences)
            is_confirmatory = (
                True
                if confirmatory is None
                else any(math.isclose(float(correlation), value, rel_tol=1e-9, abs_tol=1e-12) for value in confirmatory)
            )
            rows.append(
                {
                    x_column: float(correlation),
                    "metric": metric,
                    "reference_method": reference_method,
                    "method": method,
                    "block_count": int(valid.sum()),
                    "median_log10_ratio": float(np.median(differences)),
                    "hodges_lehmann_log10_ratio": hodges_lehmann,
                    "error_ratio": float(10.0**hodges_lehmann),
                    "reference_win_rate": float(np.mean(differences < 0.0)),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "rank_biserial": rank_biserial,
                    "confirmatory": bool(is_confirmatory),
                }
            )
    if not rows:
        return pd.DataFrame()
    comparisons = pd.DataFrame(rows)
    comparisons["p_value_holm"] = float("nan")
    confirmatory_mask = comparisons["confirmatory"].to_numpy(dtype=bool)
    if confirmatory_mask.any():
        adjusted = _holm_adjust(comparisons.loc[confirmatory_mask, "p_value"].tolist())
        comparisons.loc[confirmatory_mask, "p_value_holm"] = adjusted
    return comparisons


def run_correlation_sensitivity_sweep(
    rss_config: RSSConfig | None,
    bss_config: BSSConfig | None = None,
    sweep_config: CorrelationSweepConfig | None = None,
    *,
    base_config: SimulationConfig | None = None,
    study_config: BatchStudyConfig | None = None,
    case_callback: Callable[[CorrelationCaseOutcome], None] | None = None,
) -> CorrelationSweepResult:
    """Run the full paired correlation sweep and summarize it."""

    active = sweep_config or CorrelationSweepConfig()
    study = study_config or BatchStudyConfig()
    specs = create_correlation_case_specs(active.study, study)
    effective_rss_config = rss_config if active.include_rss else None

    outcomes: list[CorrelationCaseOutcome] = []
    rss_results: list[ValidationResult] = []
    bss_results: list[BSSValidationResult] = []
    for spec in specs:
        outcome = run_correlation_case(
            spec,
            effective_rss_config,
            bss_config,
            base_config=base_config,
            study_config=study,
            correlation_tolerance=float(active.study.correlation_tolerance),
        )
        outcomes.append(outcome)
        if outcome.rss_result is not None:
            rss_results.append(outcome.rss_result)
        bss_results.extend(outcome.bss_results)
        if case_callback is not None:
            case_callback(outcome)

    # One call keeps the RSS oracle-scaled columns and row ordering deterministic.
    case_frame, source_frame = build_method_comparison_frames(rss_results, bss_results)
    case_frame = attach_correlation_columns(case_frame, outcomes)
    source_frame = attach_correlation_columns(source_frame, outcomes)

    available = [column for column in active.case_metric_columns if column in case_frame.columns]
    case_summary = summarize_correlation_sensitivity(
        case_frame,
        metric_columns=available,
        resamples=int(active.bootstrap_resamples),
        confidence=float(active.bootstrap_confidence),
        random_state=int(active.bootstrap_seed),
    )
    thresholds = estimate_correlation_thresholds(
        case_frame,
        metric_columns=available,
        absolute_thresholds=active.absolute_thresholds,
        relative_degradations=active.relative_degradations,
        resamples=int(active.bootstrap_resamples),
        confidence=float(active.bootstrap_confidence),
        random_state=int(active.bootstrap_seed),
    )
    paired_tests = (
        paired_method_comparisons(case_frame, metric=active.primary_metric)
        if active.include_rss and active.primary_metric in case_frame.columns
        else pd.DataFrame()
    )
    return CorrelationSweepResult(
        config=active,
        specs=tuple(specs),
        case_frame=case_frame,
        source_frame=source_frame,
        case_summary=case_summary,
        thresholds=thresholds,
        paired_tests=paired_tests,
        outcomes=tuple(outcomes) if active.retain_case_results else (),
    )
