# Notebooks

| Notebook | What it does | Compute |
| --- | --- | --- |
| `rss_validation.ipynb` | Main validation workflow: single synthetic case, the standard 20-case family suite, the 8-case large-rotation stress suite, the formal 200-case four-family study (50 cases per family), the matching PCA / JADE-ICA / FastICA / SOBI comparisons (`bss_baselines`), the inter-source **correlation-sensitivity** study (ρ sweep) and the source-centering/edge-transient audit. | Single case: seconds. Formal study and sensitivity sweep: hours (set `QUICK_MODE = True` for a smoke run). |
| `rss_validation_specialized_losses.ipynb` | The same study with the four source-structure-aware losses (harmonic spectral overlap, filtered multi-lag cross-correlation, chirp time-frequency overlap, piecewise sparse innovation) next to the zero-lag baseline: loss landscapes, paired comparisons, hyper-parameter sensitivity, comparison report. Its `rss_specialized` results feed the paper's Table I. | Many hours. |
| `paper_plots.ipynb` | Reads the formal-study result tables of the plain zero-lag run (`artifacts/rss_validation/formal_experiments`, plus the zero-mean loss-comparison outputs) and draws the PDF/CDF, box and example-case figures. It writes to `…/formal_experiments/paper_plots`; [`paper_plots/`](paper_plots) is a shipped copy of the figures from the original run. | Seconds, once the formal results exist. |

Standalone script:

* `run_specialized_correlation_resume.py` — resumes the specialized-loss correlation sweep without rerunning completed
  cases; it executes only the definition/configuration cells of `rss_validation_specialized_losses.ipynb` so a resumed run
  uses exactly the implementation that produced the existing artifacts.

The notebooks locate the repository root themselves (run them with the repository, or `notebooks/`, as working
directory) and write everything below `notebooks/artifacts/` (git-ignored). Results are stored in the notebooks as
last executed; they are reproducible because every scenario is generated from fixed seeds
(formal study base seed 20000, single case seed 0). `tests/test_rss.py` checks the structure of
`rss_validation.ipynb` and executes the whole notebook end to end in `QUICK_MODE`.

Note that the batch configuration in the notebooks uses PSO (200 particles, 2000 iterations) from a fixed non-identity
initial rotation matrix — the paper text describes 80 particles and 1000 iterations; check the cells before quoting settings.
