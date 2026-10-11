from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gait_analysis.statistics import balance_trials_per_subject, scenario_eligible_trials


def _trial_frame(rows: list[tuple[str, str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["trial_id", "participant_id", "terrain", "direction", "footwear"])


def test_scenario_eligible_trials_drops_single_subject_scenario() -> None:
    frame = _trial_frame([
        ("t1", "A", "flat", "level", "0,0"),
        ("t2", "B", "flat", "level", "0,0"),
        ("t3", "A", "flat", "level", "1,4"),
        ("t4", "A", "flat", "level", "1,4"),
    ])
    result = scenario_eligible_trials(frame, min_subjects_per_scenario=2)
    assert set(result["trial_id"]) == {"t1", "t2"}
    assert not (result["footwear"] == "1,4").any()


def test_scenario_eligible_trials_respects_configurable_threshold() -> None:
    frame = _trial_frame([
        ("t1", "A", "flat", "level", "0,0"),
        ("t2", "B", "flat", "level", "0,0"),
        ("t3", "C", "flat", "level", "0,0"),
    ])
    assert len(scenario_eligible_trials(frame, min_subjects_per_scenario=2)) == 3
    assert scenario_eligible_trials(frame, min_subjects_per_scenario=4).empty


def test_scenario_eligible_trials_handles_empty_and_missing_columns() -> None:
    empty = pd.DataFrame(columns=["trial_id", "participant_id", "terrain", "direction", "footwear"])
    assert scenario_eligible_trials(empty).empty
    no_scenario_columns = pd.DataFrame({"trial_id": ["t1"], "participant_id": ["A"]})
    pd.testing.assert_frame_equal(scenario_eligible_trials(no_scenario_columns), no_scenario_columns)


def test_balance_trials_per_subject_caps_to_scenario_minimum() -> None:
    rows = [(f"t{i}", "A", "flat", "level", "0,0") for i in range(5)]
    rows += [(f"u{i}", "B", "flat", "level", "0,0") for i in range(2)]
    frame = _trial_frame(rows)
    balanced = balance_trials_per_subject(frame, seed=0)
    counts = balanced.groupby("participant_id")["trial_id"].nunique()
    assert set(counts.unique()) == {2}
    assert set(balanced.loc[balanced["participant_id"] == "B", "trial_id"]) == {"u0", "u1"}
    assert set(balanced.loc[balanced["participant_id"] == "A", "trial_id"]).issubset({f"t{i}" for i in range(5)})


def test_balance_trials_per_subject_is_deterministic_for_a_fixed_seed() -> None:
    rows = [(f"t{i}", "A", "flat", "level", "0,0") for i in range(6)]
    rows += [(f"u{i}", "B", "flat", "level", "0,0") for i in range(2)]
    frame = _trial_frame(rows)
    first = balance_trials_per_subject(frame, seed=7)
    second = balance_trials_per_subject(frame, seed=7)
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))


def test_balance_trials_per_subject_differs_across_seeds_but_stays_valid() -> None:
    rows = [(f"t{i}", "A", "flat", "level", "0,0") for i in range(10)]
    rows += [(f"u{i}", "B", "flat", "level", "0,0") for i in range(2)]
    frame = _trial_frame(rows)
    samples = [set(balance_trials_per_subject(frame, seed=seed).loc[lambda f: f["participant_id"] == "A", "trial_id"]) for seed in range(6)]
    for sample in samples:
        assert len(sample) == 2
    assert len(set(map(frozenset, samples))) > 1


def test_balance_trials_per_subject_handles_empty_frame() -> None:
    empty = pd.DataFrame(columns=["trial_id", "participant_id", "terrain", "direction", "footwear"])
    assert balance_trials_per_subject(empty).empty


def test_scenario_filter_then_balance_matches_pipeline_ordering() -> None:
    """Mirrors pipeline.run_statistics: filter single-subject scenarios first, then balance."""
    rows = [(f"t{i}", "A", "flat", "level", "0,0") for i in range(3)]
    rows += [(f"u{i}", "B", "flat", "level", "0,0") for i in range(1)]
    rows += [("s0", "C", "flat", "level", "1,4")]  # single-subject scenario
    frame = _trial_frame(rows)
    filtered = scenario_eligible_trials(frame, min_subjects_per_scenario=2)
    assert "s0" not in set(filtered["trial_id"])
    balanced = balance_trials_per_subject(filtered, seed=0)
    assert set(balanced.groupby("participant_id")["trial_id"].nunique().unique()) == {1}
