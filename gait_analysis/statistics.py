"""Trial-unit condition estimates, mixed effects, bootstrap CIs and LOO checks."""

from __future__ import annotations

import warnings
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .progress import ProgressTracker, progress_note
from .utils import benjamini_hochberg


def scenario_eligible_trials(
    trial_features: pd.DataFrame,
    *,
    scenario_columns: Sequence[str] = ("terrain", "direction", "footwear"),
    min_subjects_per_scenario: int = 2,
) -> pd.DataFrame:
    """Drop trial rows belonging to a (terrain, direction, footwear) scenario with too few distinct subjects."""
    scenario_columns = list(scenario_columns)
    if trial_features.empty or any(column not in trial_features for column in scenario_columns):
        return trial_features
    counts = trial_features.groupby(scenario_columns)["participant_id"].nunique()
    eligible = counts[counts >= min_subjects_per_scenario].index
    key = trial_features.set_index(scenario_columns).index
    return trial_features[key.isin(eligible)].reset_index(drop=True)


def balance_trials_per_subject(
    trial_frame: pd.DataFrame,
    group_cols: Sequence[str] = ("terrain", "direction", "footwear"),
    *,
    seed: int = 0,
) -> pd.DataFrame:
    """Subsample every participant's trials within each scenario down to that scenario's minimum per-subject trial count."""
    group_cols = list(group_cols)
    if trial_frame.empty or any(column not in trial_frame for column in group_cols):
        return trial_frame
    frame = trial_frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(frame), dtype=bool)
    for _, group in frame.groupby(group_cols, sort=True):
        participant_positions: dict[str, list[int]] = {}
        for participant_id in sorted(group["participant_id"].astype(str).unique()):
            participant_mask = group["participant_id"].astype(str) == participant_id
            ordering = group.loc[participant_mask, "trial_id"].astype(str).sort_values()
            participant_positions[participant_id] = ordering.index.tolist()
        minimum_count = min(len(positions) for positions in participant_positions.values())
        for participant_id in sorted(participant_positions):
            positions = participant_positions[participant_id]
            chosen = rng.choice(len(positions), size=minimum_count, replace=False)
            keep[[positions[index] for index in sorted(chosen)]] = True
    return frame[keep].reset_index(drop=True)


def _participant_bootstrap_delta(
    frame: pd.DataFrame, outcome: str, condition_column: str, reference: str, condition: str,
    repetitions: int, rng: np.random.Generator,
) -> tuple[float, float]:
    participants = frame["participant_id"].unique()
    estimates = []
    tracker = ProgressTracker(
        f"statistics bootstrap {condition_column}:{outcome}:{condition}",
        repetitions,
        report_every=max(1, repetitions // 10),
    )
    for repetition in range(repetitions):
        sampled = rng.choice(participants, len(participants), replace=True)
        pieces = [frame[frame["participant_id"] == participant].assign(_bootstrap_id=index) for index, participant in enumerate(sampled)]
        boot = pd.concat(pieces, ignore_index=True)
        means = boot.groupby(["_bootstrap_id", condition_column])[outcome].median().unstack()
        if reference in means and condition in means:
            estimates.append(float((means[condition] - means[reference]).median()))
        tracker.update(repetition + 1, f"valid_estimates={len(estimates)}")
    return tuple(np.quantile(estimates, [0.025, 0.975])) if estimates else (np.nan, np.nan)


def run_condition_statistics(
    trial_features: pd.DataFrame,
    config: AnalysisConfig,
    *,
    outcome_columns: Iterable[str] | None = None,
    condition_column: str = "terrain",
) -> pd.DataFrame:
    """Estimate within-participant condition effects at the trial unit."""
    if trial_features.empty or condition_column not in trial_features:
        return pd.DataFrame()
    settings = config.section("statistics")
    if outcome_columns is None:
        configured = sum((list(values) for values in settings["primary_outcomes"].values()), [])
        outcome_columns = [column for column in configured if column in trial_features]
    outcome_columns = list(outcome_columns)
    conditions = sorted(trial_features[condition_column].dropna().astype(str).unique())
    reference = str(config.section("phenotype")["reference_terrain"]) if condition_column == "terrain" else conditions[0]
    if reference not in conditions and conditions:
        reference = conditions[0]
    repetitions = int(config.values["profiles"][config.profile].get("bootstrap_repetitions", settings["bootstrap_repetitions"]))
    rng = np.random.default_rng(int(config.values["random_seed"]))
    rows = []
    progress_note(
        "statistics",
        f"condition_family={condition_column} outcomes={len(outcome_columns)} conditions={len(conditions)} "
        f"bootstrap_repetitions={repetitions}",
    )
    outcome_tracker = ProgressTracker(f"statistics outcomes {condition_column}", len(outcome_columns))
    for outcome_index, outcome in enumerate(outcome_columns, start=1):
        progress_note("statistics", f"starting condition_family={condition_column} outcome={outcome}")
        data = trial_features[["participant_id", condition_column, outcome]].dropna().copy()
        data[condition_column] = data[condition_column].astype(str)
        participant_condition = data.groupby(["participant_id", condition_column])[outcome].median().unstack()
        mixed_coefficients: dict[str, float] = {}
        mixed_p_values: dict[str, float] = {}
        mixed_model_name = "participant_paired_robust"
        try:
            import statsmodels.api as sm

            other_conditions = [condition for condition in conditions if condition != reference]
            design = np.column_stack((
                np.ones(len(data)),
                *[(data[condition_column].to_numpy() == condition).astype(float) for condition in other_conditions],
            ))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = sm.MixedLM(
                    data[outcome].to_numpy(float), design, groups=data["participant_id"].astype(str).to_numpy()
                ).fit(reml=False, method="lbfgs", disp=False)
            if not bool(getattr(fitted, "converged", False)) or not np.all(np.isfinite(fitted.params)):
                raise ValueError("mixed model did not converge to finite parameters")
            mixed_coefficients = {condition: float(fitted.params[index + 1]) for index, condition in enumerate(other_conditions)}
            mixed_p_values = {condition: float(fitted.pvalues[index + 1]) for index, condition in enumerate(other_conditions)}
            mixed_model_name = "mixedlm_participant_random_intercept"
        except Exception:
            pass
        for condition in conditions:
            if condition == reference or reference not in participant_condition or condition not in participant_condition:
                continue
            paired = (participant_condition[condition] - participant_condition[reference]).dropna()
            paired_median = float(paired.median()) if len(paired) else np.nan
            estimate = mixed_coefficients.get(condition, paired_median)
            ci_low, ci_high = _participant_bootstrap_delta(data, outcome, condition_column, reference, condition, repetitions, rng)
            # A participant-label sign-flip test is an exact paired null for modest cohorts.
            if condition in mixed_p_values:
                p_value = mixed_p_values[condition]
            elif len(paired):
                draws = max(999, repetitions)
                null = np.asarray([np.mean(paired.to_numpy() * rng.choice([-1.0, 1.0], len(paired))) for _ in range(draws)])
                p_value = float((1 + np.count_nonzero(np.abs(null) >= abs(float(paired.mean())))) / (draws + 1))
            else:
                p_value = np.nan
            loo = [float(paired.drop(index).median()) for index in paired.index] if len(paired) > 1 else []
            rows.append({
                "outcome": outcome, "condition_family": condition_column, "reference": reference,
                "condition": condition, "estimate": estimate, "ci_low": ci_low, "ci_high": ci_high,
                "p_value": p_value, "participants": len(paired),
                "paired_median_effect": paired_median,
                "participant_direction_agreement": float(np.mean(np.sign(paired) == np.sign(estimate))) if len(paired) else np.nan,
                "effect_direction": "increase" if estimate > 0 else "decrease" if estimate < 0 else "no_change",
                "loo_min": min(loo) if loo else np.nan, "loo_max": max(loo) if loo else np.nan,
                "loo_direction_stable": bool(loo and all(np.sign(value) == np.sign(estimate) for value in loo)),
                "unit": "trial", "model": mixed_model_name,
            })
        outcome_tracker.update(outcome_index, f"outcome={outcome}")
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_value"] = result.groupby("condition_family")["p_value"].transform(benjamini_hochberg)
        result["fdr_significant"] = result["q_value"] <= float(settings["fdr_q"])
    return result
