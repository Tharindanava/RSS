from __future__ import annotations

from pathlib import Path

import pytest

from gait_analysis import AnalysisConfig


@pytest.fixture
def config() -> AnalysisConfig:
    return AnalysisConfig.from_yaml().with_values({
        "mfpca": {"bootstrap_repetitions": 2},
        "clustering": {"bootstrap_repetitions": 3, "candidate_clusters": [2, 3]},
        "statistics": {"bootstrap_repetitions": 20},
        "profiles": {"development": {"te_surrogates": 3, "bootstrap_repetitions": 3}},
    })

