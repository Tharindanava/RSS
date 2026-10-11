"""Per-joint RMSE (rad) for RSS + PCA/JADE-ICA/FastICA/SOBI on the QArm best-30 trials.

Same scoring as the correlation comparison (``practical_validation.match_and_score``:
one-to-one matching, sign + mean-offset alignment, no amplitude scaling). RSS is scored
from its saved matched sources; baselines are refit exactly as in best30_baseline_comparison.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from bss_baselines import BSSConfig, fit_bss
from qarm_rss_experiments.best30_baseline_comparison import BASELINE_METHODS, _trial_observation_window
from qarm_rss_experiments.plot_matched_signals import RSS_SOURCES_ROOT
from qarm_rss_experiments.practical_validation import TRUTH_COLUMNS, match_and_score

ranking = pd.read_csv(RSS_SOURCES_ROOT / "best_30_ranking.csv")
config = BSSConfig(methods=BASELINE_METHODS, random_state=0)
rows = []
for r in ranking.itertuples(index=False):
    session, test = r.session, int(r.test_number)
    matched = pd.read_csv(RSS_SOURCES_ROOT / session / "matched" / f"test_{test}.csv")
    truth_rss = matched[list(TRUTH_COLUMNS)].to_numpy(float).T
    est_rss = matched[["source_base", "source_elbow", "source_wrist"]].to_numpy(float).T
    window, truth, batch = _trial_observation_window(session, test)
    jobs = [("rss", truth_rss, est_rss)]
    truth_m = truth[list(TRUTH_COLUMNS)].to_numpy(float).T
    for m in BASELINE_METHODS:
        jobs.append((m, truth_m, fit_bss(window, m, config).source_angles))
    for m, tr, es in jobs:
        scored, _ = match_and_score(tr, es, test_number=test, batch_number=batch, method=m)
        for k, it in enumerate(scored):
            rows.append({"truth_range_rad": float(np.ptp(tr[k])), "session": session, "test_number": test, "method": m, "joint": it["joint"],
                         "abs_correlation": it["abs_correlation"], "rmse_rad": it["rmse_rad"]})
            rows[-1]["nrmse_range"] = it["rmse_rad"] / rows[-1]["truth_range_rad"]
out = pd.DataFrame(rows)
dest = Path(__file__).resolve().parent / "correlation_comparison" / "qarm_best30_rmse.csv"
dest.parent.mkdir(exist_ok=True)
out.to_csv(dest, index=False)
print(out.groupby("method")[["abs_correlation", "rmse_rad"]].mean())
