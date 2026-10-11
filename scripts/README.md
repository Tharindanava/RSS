# scripts

* `make_demo_outputs.py` — writes a small, fully **synthetic** RSS-style output tree (`limb_angle_mode.csv`,
  `limb_axes_mode.csv`, gait-like periodic modes with randomly flipped signs) so that `HMA_Dashboard`,
  `hilbert_analysis`, `rotating_poincare` and `limb_transfer_entropy` can be tried without human data.

  ```bash
  python scripts/make_demo_outputs.py --output-root outputs/rss/Demo_Synthetic
  ```

* `run_formal_loss_comparison.py` — the paired 200-case RSS loss comparison of `notebooks/rss_validation.ipynb`, as a
  resumable command-line job (results are checkpointed after every fit, so repeating the same command resumes an
  interrupted run). `python scripts/run_formal_loss_comparison.py --help` lists the options; outputs go to
  `outputs/rss_formal_loss_comparison/` by default.
