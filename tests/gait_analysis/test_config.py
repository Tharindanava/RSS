from __future__ import annotations

import pytest

from gait_analysis import AnalysisConfig


def test_default_config_validates_the_new_ankle_minima_segmentation_schema() -> None:
    config = AnalysisConfig.from_yaml()
    segment = config.section("segmentation")
    assert set(segment) == {
        "filter_order", "bandpass_hz", "stride_seconds", "normalized_points",
        "hilbert_phase_threshold", "valley_prominence_mad", "phase_rotation_bounds",
    }
    assert 0.0 < segment["hilbert_phase_threshold"] < 1.0


def test_validate_rejects_out_of_range_hilbert_phase_threshold(config) -> None:
    with pytest.raises(ValueError, match="hilbert_phase_threshold"):
        config.with_values({"segmentation": {"hilbert_phase_threshold": 1.5}})
    with pytest.raises(ValueError, match="hilbert_phase_threshold"):
        config.with_values({"segmentation": {"hilbert_phase_threshold": 0.0}})


def test_validate_rejects_invalid_phase_rotation_bounds(config) -> None:
    with pytest.raises(ValueError, match="phase_rotation_bounds"):
        config.with_values({"segmentation": {"phase_rotation_bounds": [1.5, 0.5]}})
    with pytest.raises(ValueError, match="phase_rotation_bounds"):
        config.with_values({"segmentation": {"phase_rotation_bounds": [-0.1, 1.0]}})


def test_validate_no_longer_requires_confidence_weights() -> None:
    config = AnalysisConfig.from_yaml()
    assert "confidence_weights" not in config.section("segmentation")
    assert "use_hilbert_phase" not in config.section("segmentation")
    assert "stance_fraction" not in config.section("segmentation")


def test_default_config_has_scenario_filter_section() -> None:
    config = AnalysisConfig.from_yaml()
    scenario_filter = config.section("scenario_filter")
    assert scenario_filter["min_subjects_per_scenario"] >= 1
    assert config.section("statistics")["balance_trials_per_subject"] is True


def test_validate_rejects_non_positive_min_subjects_per_scenario(config) -> None:
    with pytest.raises(ValueError, match="min_subjects_per_scenario"):
        config.with_values({"scenario_filter": {"min_subjects_per_scenario": 0}})
