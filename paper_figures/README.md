# Paper figures and tables

Scripts that produce the tables and figures of the RSS letter from stored experiment results, together with the
generated outputs. Every script is run from anywhere (`python paper_figures/<script>.py`); paths are resolved
relative to the repository root.

| Script | Produces | Reads |
| --- | --- | --- |
| `make_comparison_table_boxplot.py` | Table I (mean abs. correlation, RMSE, NRMSE; four simulated families + QArm) and box plots → `correlation_comparison/` | per-case metrics of the formal 200-case study (`notebooks/artifacts/rss_validation_specialized_losses/…`), `outputs/rss/QArm_Adam_NoSSA_Piecewise/best_30_baseline_comparison.csv`, `correlation_comparison/qarm_best30_rmse.csv` |
| `compute_qarm_best30_rmse.py` | per-joint RMSE of RSS and the four baselines on the QArm best-30 trials → `correlation_comparison/qarm_best30_rmse.csv` | `outputs/rss/QArm_Adam_NoSSA_Piecewise/` (run `python -m qarm_rss_experiments.reproduce_recommended_pipeline` first); refits the baselines |
| `run_linear_correlation_sweep.py` | linear-grid correlation-sensitivity sweep, ρ = 0, 0.05, …, 0.95 (800 cases, resumable, hours) → `correlation_sensitivity_linear/case_metrics/` | – (re-simulates and refits) |
| `make_linear_correlation_sensitivity_figure.py` | the correlation-sensitivity figure and tables → `correlation_sensitivity_linear/` | `correlation_sensitivity_linear/case_metrics/*.json` (shipped) |
| `make_correlation_sensitivity_figure.py` | log-grid version of the same study → `correlation_sensitivity/` | `…/rss_validation_specialized_losses/comparison_report/tables/correlation_sensitivity_summary.csv` |
| `make_nrmse_sensitivity_figures.py` | NRMSE vs ρ (log and linear grids) | per-case `source_comparison.csv` of the log-grid study + the linear sweep |
| `make_sensitivity_key.py` | colour/layout key for the sensitivity figures | – |
| `make_harmonic_figures.py` | label-free source / observer / overlay figures for one harmonic case (re-fits RSS on the saved scenario) → `harmonic_formal_harmonic_<n>/` | `…/rss_specialized/formal_experiments/cases/harmonic/<case>/source_comparison.csv` |
| `make_axis_bar_figures.py` | axis-alignment bar chart | `harmonic_formal_harmonic_<n>/figure_data.npz` (shipped) |

`run_linear_correlation_sweep.py` starts its worker pool with the `fork` method (Linux/macOS only) and defaults to many workers; check its
constants before launching it on a small machine. The shipped sub-folders are the outputs those scripts produced for the paper. The inputs marked
`notebooks/artifacts/…` are written by `notebooks/rss_validation_specialized_losses.ipynb` (about 4.9 GB, many hours of
compute) and are **not** distributed; regenerate them with the notebook before re-running the scripts that read them.
The QArm inputs under `outputs/` regenerate in about two minutes (see `qarm_rss_experiments/README.md`).

## What the table's RSS columns are

* **Simulated families** — the RSS column is the family-matched *specialized-loss* variant (`rss_specialized`, method key
  of the specialized-loss notebook): harmonic → `harmonic_spectral_overlap`, filtered noise →
  `filtered_multilag_cross_correlation`, chirp-modulated → `chirp_time_frequency_overlap`, piecewise transient →
  `piecewise_sparse_innovation`. The plain zero-lag cross-correlation runs of `rss_validation.ipynb` are the
  `rss_corr_baseline` arm of the same notebook study and are not what the table reports.
* **QArm** — the best 30 of 65 trials by RSS's own score (see `qarm_rss_experiments/README.md`).
* **Baselines** — PCA, JADE-ICA (`ICA`), FastICA and SOBI from `bss_baselines`, fitted on the same observation windows.
