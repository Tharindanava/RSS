"""Compare PCA, JADE-ICA, FastICA, and SOBI against RSS on the 30 best trials.

For each of the 30 trials picked by ``best_results_report.py``, reconstructs
the exact bias-corrected, integrated 3-channel gyro "observations" window
that ``rss process-imu`` was fit on (bias-window 1.0s, the same
movement-region alignment ``build_matched_rss_sources.py`` uses to slice a
batch into its five trials), fits each baseline with
``bss_baselines.fit_bss`` on that same window, and scores every method with
the same one-to-one linear-sum-assignment matching used throughout this
project (``practical_validation.match_and_score``). RSS's own scores are
read straight from ``best_30_ranking.csv`` rather than refit, since they are
already the exact same window's result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bss_baselines import (
    BSSConfig,
    METHOD_FASTICA,
    METHOD_JADE_ICA,
    METHOD_PCA,
    METHOD_SOBI,
    fit_bss,
)
from qarm_rss_experiments.build_merged_trials import _gyro_activity
from qarm_rss_experiments.hyperparameter_search import DATA_ROOT, NODE_PREFIX, SAMPLE_RATE_HZ
from qarm_rss_experiments.plot_matched_signals import JOINTS
from qarm_rss_experiments.practical_validation import (
    TRUTH_COLUMNS,
    _centered_moving_average,
    _measured_joint_speed,
    _movement_regions,
    _pearson,
    match_and_score,
)

BIAS_WINDOW_SECONDS = 1.0
BASELINE_METHODS = (METHOD_PCA, METHOD_JADE_ICA, METHOD_FASTICA, METHOD_SOBI)
METHOD_ORDER = ("rss", METHOD_PCA, METHOD_JADE_ICA, METHOD_FASTICA, METHOD_SOBI)
METHOD_LABELS = {
    "rss": "RSS",
    METHOD_PCA: "PCA",
    METHOD_JADE_ICA: "ICA (JADE)",
    METHOD_FASTICA: "FastICA",
    METHOD_SOBI: "SOBI",
}
# Okabe-Ito colorblind-safe palette.
METHOD_COLORS = {
    "rss": "#0072B2",
    METHOD_PCA: "#E69F00",
    METHOD_JADE_ICA: "#009E73",
    METHOD_FASTICA: "#CC79A7",
    METHOD_SOBI: "#D55E00",
}


def _reconstruct_batch_observations(session: str, batch_number: int) -> tuple[np.ndarray, pd.DataFrame]:
    batch_dir = DATA_ROOT / session / str(batch_number)
    csv_paths = list(batch_dir.glob("combined_log_wide_*.csv"))
    if len(csv_paths) != 1:
        raise ValueError(f"Expected one combined log csv in {batch_dir}; found {len(csv_paths)}.")
    raw = pd.read_csv(csv_paths[0])
    prefix = NODE_PREFIX[session]
    columns = [f"{prefix}_G{axis}" for axis in "XYZ"]
    gyro = np.radians(raw.loc[:, columns].to_numpy(dtype=float))
    bias_count = min(gyro.shape[0], max(1, round(BIAS_WINDOW_SECONDS * SAMPLE_RATE_HZ)))
    bias = np.mean(gyro[:bias_count], axis=0)
    increments = (gyro - bias[None, :]) / SAMPLE_RATE_HZ
    observations = np.cumsum(increments, axis=0).T
    return observations, raw


def _trial_observation_window(session: str, test_number: int) -> tuple[np.ndarray, pd.DataFrame, int]:
    batch_number = (test_number - 1) // 5 + 1
    local_index = (test_number - 1) % 5 + 1
    observations, raw = _reconstruct_batch_observations(session, batch_number)

    activity = _gyro_activity(raw)
    regions = _movement_regions(activity, SAMPLE_RATE_HZ)
    if len(regions) != 5:
        raise ValueError(f"Expected five movement regions for {session} batch {batch_number}; found {len(regions)}.")
    smooth_activity = _centered_moving_average(activity, round(0.21 * SAMPLE_RATE_HZ))
    region = regions[local_index - 1]

    truth_path = DATA_ROOT / session / "Ground_Truth" / "results" / f"test_{test_number}.csv"
    truth = pd.read_csv(truth_path, float_precision="round_trip")
    speed = _centered_moving_average(_measured_joint_speed(truth), round(0.21 * SAMPLE_RATE_HZ))
    first = max(0, region[0] - round(1.5 * SAMPLE_RATE_HZ))
    last = min(observations.shape[1] - len(truth), region[0] + round(1.5 * SAMPLE_RATE_HZ))
    candidates = [
        (_pearson(smooth_activity[start : start + len(truth)], speed), start)
        for start in range(first, last + 1)
    ]
    _, aligned = max(candidates)
    window = observations[:, aligned : aligned + len(truth)]
    return window, truth, batch_number


def compute_comparison_table(top: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int, str]]]:
    """Return long-form (session, test_number, method, joint, abs_correlation) rows for RSS + baselines."""

    config = BSSConfig(methods=BASELINE_METHODS, random_state=0)
    rows: list[dict] = []
    failures: list[tuple[str, int, str]] = []

    for row in top.itertuples(index=False):
        for joint in JOINTS:
            rows.append(
                {
                    "session": row.session,
                    "test_number": int(row.test_number),
                    "method": "rss",
                    "joint": joint,
                    "abs_correlation": float(getattr(row, joint)),
                }
            )

    for row in top.itertuples(index=False):
        session = row.session
        test_number = int(row.test_number)
        try:
            window, truth, batch_number = _trial_observation_window(session, test_number)
        except Exception as exc:  # noqa: BLE001
            failures.append((session, test_number, f"alignment: {exc}"))
            continue
        truth_matrix = truth[list(TRUTH_COLUMNS)].to_numpy(dtype=float).T
        for method in BASELINE_METHODS:
            try:
                fitted = fit_bss(window, method, config)
                scored, _aligned = match_and_score(
                    truth_matrix,
                    fitted.source_angles,
                    test_number=test_number,
                    batch_number=batch_number,
                    method=method,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append((session, test_number, f"{method}: {exc}"))
                continue
            for item in scored:
                rows.append(
                    {
                        "session": session,
                        "test_number": test_number,
                        "method": method,
                        "joint": item["joint"],
                        "abs_correlation": item["abs_correlation"],
                    }
                )

    return pd.DataFrame(rows), failures
