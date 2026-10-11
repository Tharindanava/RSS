"""Trial-level phenotype construction and participant-reference adaptation scores."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .utils import robust_zscore


IDENTIFIERS = [
    "trial_id", "participant_id", "terrain", "direction", "footwear",
    "source_recordings", "raw_files", "rss_files",
]


def _prefix_features(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    identifiers = [column for column in IDENTIFIERS if column in frame.columns]
    numeric = frame.select_dtypes(include=[np.number]).columns
    return frame[identifiers + list(numeric)].rename(columns={column: f"{family}__{column}" for column in numeric})


def build_trial_phenotypes(
    classic_trial: pd.DataFrame,
    mfpca_trial: pd.DataFrame,
    te_window_features: pd.DataFrame,
    config: AnalysisConfig,
) -> dict[str, pd.DataFrame]:
    """Return absolute, terrain-adaptation and footwear-adaptation phenotypes."""
    def empty_results(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {
            "absolute": frame.copy(),
            "terrain_adaptation": frame.copy(),
            "footwear_adaptation": frame.copy(),
        }

    selection = config.section("phenotype")["feature_selection"]
    classic_identifiers = [column for column in IDENTIFIERS if column in classic_trial]
    classic_columns = [column for column in selection["classic"] if column in classic_trial]
    classic = _prefix_features(classic_trial[classic_identifiers + classic_columns], "classic")
    if mfpca_trial.empty:
        mfpca = pd.DataFrame(columns=["trial_id", "participant_id"])
    else:
        mfpca_numeric = [
            column for column in mfpca_trial
            if any(column.startswith(prefix) for prefix in selection["mfpca_prefixes"])
        ]
        mfpca = mfpca_trial.groupby(["trial_id", "participant_id"], as_index=False)[mfpca_numeric].median()
        mfpca = _prefix_features(mfpca, "mfpca")
    if te_window_features.empty:
        te = pd.DataFrame(columns=["trial_id"])
    else:
        te_numeric = [
            column for column in te_window_features
            if any(column.startswith(prefix) for prefix in selection["te_prefixes"])
            and not any(column.endswith(suffix) for suffix in selection.get("exclude_suffixes", []))
        ]
        te = te_window_features.groupby("trial_id", as_index=False)[te_numeric].median()
        te = _prefix_features(te, "te")
    excluded = set(config.section("phenotype").get("exclude_families", []))
    phenotype = classic.copy()
    if "mfpca" not in excluded:
        merge_keys = [
            column for column in ("trial_id", "participant_id")
            if column in phenotype and column in mfpca
        ]
        if not merge_keys:
            return empty_results(phenotype)
        phenotype = phenotype.merge(mfpca, on=merge_keys, how="inner")
    if "te" not in excluded:
        if "trial_id" not in phenotype or "trial_id" not in te:
            return empty_results(phenotype)
        phenotype = phenotype.merge(te, on="trial_id", how="inner")
    phenotype = phenotype[[column for column in phenotype if not any(column.startswith(f"{family}__") for family in excluded)]]
    features = [column for column in phenotype if "__" in column]
    if bool(config.section("phenotype").get("complete_case", True)):
        phenotype = phenotype.dropna(subset=features).reset_index(drop=True)
    if phenotype.empty:
        return empty_results(phenotype)
    reference = config.section("phenotype")
    baseline_rows = phenotype[
        (phenotype["terrain"].astype(str).str.lower() == str(reference["reference_terrain"]).lower())
        & (phenotype["footwear"].astype(str) == str(reference["reference_footwear"]))
    ]
    participant_baseline = baseline_rows.groupby("participant_id")[features].median()
    global_baseline = baseline_rows[features].median() if not baseline_rows.empty else phenotype[features].median()
    baseline = phenotype[["participant_id"]].join(participant_baseline, on="participant_id")[features].fillna(global_baseline)
    delta = phenotype.copy()
    delta[features] = phenotype[features].to_numpy(float) - baseline.to_numpy(float)
    scaled, centers, scales = robust_zscore(phenotype[features])
    absolute = phenotype.copy()
    absolute[features] = scaled
    for frame, kind in ((absolute, "absolute"), (delta, "adaptation")):
        frame["phenotype_kind"] = kind
        frame["signal_type"] = "rss_segment_mode1_proxy"
    absolute.attrs.update({"centers": centers.to_dict(), "scales": scales.to_dict()})
    terrain_delta = delta[
        (delta["footwear"].astype(str) == str(reference["reference_footwear"]))
        & (delta["terrain"].astype(str).str.lower() != str(reference["reference_terrain"]).lower())
    ].copy()
    footwear_delta = delta[
        (delta["terrain"].astype(str).str.lower() == str(reference["reference_terrain"]).lower())
        & (delta["footwear"].astype(str) != str(reference["reference_footwear"]))
    ].copy()
    return {"absolute": absolute, "terrain_adaptation": terrain_delta, "footwear_adaptation": footwear_delta}


def equal_family_matrix(frame: pd.DataFrame, family_weights: Mapping[str, float]) -> tuple[np.ndarray, list[str]]:
    """Create a phenotype matrix with equal aggregate distance weight per family."""
    blocks: list[np.ndarray] = []
    columns: list[str] = []
    for family, weight in family_weights.items():
        family_columns = [column for column in frame if column.startswith(f"{family}__")]
        if not family_columns:
            continue
        scaled, _, _ = robust_zscore(frame[family_columns])
        block = scaled.fillna(0.0).to_numpy(float)
        block *= np.sqrt(float(weight) / max(1, block.shape[1]))
        blocks.append(block)
        columns.extend(family_columns)
    if not blocks:
        raise ValueError("No configured phenotype feature families are available")
    return np.column_stack(blocks), columns
