from __future__ import annotations

import numpy as np
import pandas as pd

from gait_analysis.visualization import create_figures


def test_condition_effects_allows_estimate_outside_interval(tmp_path) -> None:
    statistics = pd.DataFrame({
        "outcome": ["stride_duration", "cadence"],
        "condition": ["uneven", "incline"],
        "estimate": [0.09, 14.3],
        "ci_low": [0.00, np.nan],
        "ci_high": [0.08, np.nan],
    })

    manifest = create_figures(tmp_path, statistics=statistics)

    figure = manifest.loc[manifest["figure"] == "condition_effects"]
    assert len(figure) == 1
    assert (tmp_path / "figures" / "condition_effects.png").is_file()
