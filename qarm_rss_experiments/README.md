# QArm practical validation

Practical validation of RSS with a rigidly mounted IMU on a Quanser QArm. This folder holds

1. the **recorded validation** — code and recordings behind the QArm column of the paper
   (`Sample_Data/QArm/`, one-command reproduction below);
2. a **hardware experiment bank** — 50 deterministic joint trajectories and a guarded QArm runner, for new
   recordings (IMU acquisition is separate; see the end of this file).

## 1. Recorded validation

### Protocol

Each test is a 30 s, 100 Hz trajectory commanded to three QArm joints — **base, elbow and wrist**; the shoulder
is locked and the gripper is held at a fixed opening. Each joint follows its own random multisine (2–5 sinusoids,
0.05–0.12 Hz); test *N* uses seed `20000 + N`, and the exact sine parameters are stored with the recording
(`test_N.json`). The hardware runner is [`run_multisine_protocol.py`](run_multisine_protocol.py) (requires the Quanser API). **The two sessions were not generated identically:**

| | 2026-09-18 session | 2026-09-09 session |
| --- | --- | --- |
| joint signals | mean-centred and symmetrically whitened together (target \|ρ\| ≈ 0), then scaled | not whitened (no `decorrelation` entry in the JSON) |
| elbow | kept below 5° (design amplitude 4.5°) so base and wrist axes stay roughly orthogonal | amplitude ±30° |
| speed cap | 45 °/s, enforced and recorded | not recorded |
| pairwise \|ρ\| of the three commanded joint signals | ≈ 0 | median of the per-test maximum 0.60, up to 0.95 |
| pairwise \|ρ\| of the *measured* joint angles (per-test maximum) | median 0.12 (max 0.27) | median 0.59 (max 0.94) |

So the 2026-09-09 recordings do **not** satisfy the "mutually uncorrelated sources" condition that the RSS objective assumes
(and that the letter states was kept below 0.1 for all input signals); of the best-30 trials used in the paper only one
has every pairwise \|ρ\| of the measured joints below 0.1. Test 20 of 2026-09-09 is 26 s long rather than 30 s.

A recording batch holds **five tests back to back** in one multi-IMU log
(`combined_log_wide_<epoch>.csv`, 100 Hz). The encoder log of every test (commanded and measured joint angles,
`test_N.csv`) is stored separately as ground truth.

| Session | Batches | Tests | Wrist/gripper IMU | Used |
| --- | --- | --- | --- | --- |
| `session_2026-09-09` | 1–7 | 1–35 | `OrbiX-Node-4_IMU1` | 35 trials |
| `session_2026-09-18` | 1–6 (and 7) | 1–30 (and 31–35) | `OrbiX-Node-10_IMU1` (batch 7: Node 7) | 30 trials |

`setup.txt` in each session folder records which node sat on the wrist, forearm and shoulder. Only the wrist
node's gyroscope is used by the separation; the other nodes are quality-control data.

> **Batch 7 of the 2026-09-18 session** has no Node-10 columns — `setup.txt` says the wrist IMU was Node 7 for that batch
> — so the analysis (`discover_batches`) excludes it. It looks recoverable by mapping that batch to
> `OrbiX-Node-7_IMU1`; that has not been done, so tests 31–35 of 2026-09-18 are not in any reported number.

### Data layout

```text
Sample_Data/QArm/<session>/
    setup.txt                                  which IMU node was mounted where
    1/ ... 7/combined_log_wide_<epoch>.csv     raw multi-IMU log, five tests per batch
    Ground_Truth/results/test_N.csv            encoder log: time_s, target_*_rad, measured_*_rad (base/shoulder/elbow/wrist/gripper)
    Ground_Truth/results/test_N.json           generator parameters of the commanded trajectory
```

### Reproduce the paper's QArm numbers

From the repository root (about two minutes on a CPU):

```bash
python -m qarm_rss_experiments.reproduce_recommended_pipeline
```

This (1) runs `rss process-imu` on each of the 13 usable batches with the recommended recipe, (2) splits every batch
into its five tests and pairs the recovered sources with the encoder ground truth, and (3) ranks all trials, keeps the
best 30, refits PCA / JADE-ICA / FastICA / SOBI on exactly the same observation windows and writes tables, plots and a
PDF. Everything lands in `outputs/rss/QArm_Adam_NoSSA_Piecewise/`. The recipe is

```text
--rss-optimizer adam --disable-ssa-preprocessing --bias-window-seconds 1.0 --epochs 400
--rss-adaptive-regularizer --rss-specialized-regularizer piecewise_sparse_innovation
--rss-regularizer-start-weight 0.5 --rss-regularizer-plateau-patience 20 --rss-regularizer-ramp-epochs 20
```

with the default SO(3) parameterisation, zero-lag cross-correlation and an identity start. The exact command line is
`hyperparameter_search_round2.build_command2` plus the regularizer flags above: learning rate 0.1 → 1e-3, `--rss-eps 1e-24`,
`--rss-quaternion-eps 1e-24`, no orthogonality/reconstruction terms and no convergence stop.
The separation sees **only the wrist IMU's gyroscope**: the first 1 s (sensor stationary) gives a static bias, the
bias-corrected rates are integrated to three body-rotation channels, and RSS is fitted to those. The encoder data are
used only afterwards, to match each recovered source one-to-one (Hungarian assignment on |Pearson r|, sign-corrected)
to a measured joint and to score it.

[`reference_results/`](reference_results) holds the three CSVs a run produces
(`all_trials_ranking.csv`, `best_30_ranking.csv`, `best_30_baseline_comparison.csv`); a fresh run reproduces them exactly.

### Results — read the selection

| | trials | RSS mean abs. correlation |
| --- | ---: | ---: |
| all matched trials | 65 | **0.807** (base 0.843, elbow 0.758, wrist 0.822) |
| 2026-09-09 session (Node 4) | 35 | 0.942 |
| 2026-09-18 session (Node 10) | 30 | 0.651 |
| best 30 trials by RSS's own score | 30 | **0.961** |

PCA / JADE-ICA / FastICA / SOBI reach 0.656 / 0.666 / 0.656 / 0.653 on the best 30. **The paper's QArm column is
this best-30 subset**: the 30 trials are chosen by RSS's own mean |r| against the encoder ground truth
(`best_results_report.py`), 29 of them from the 2026-09-09 session, and the baselines are then scored on the same 30
windows. The selection favours RSS by construction. Report the all-65 figures alongside the best-30 figures, or
describe the selection explicitly, when quoting these numbers. The large gap between sessions (0.942 vs 0.651) is confounded: the sessions differ in wrist IMU (Node 4 vs Node 10),
in mounting, and in trajectory design (see the protocol table), so it cannot be attributed to any one of them.

How the recipe was chosen (staged search over optimiser, SSA, epochs, learning rate and initial matrix, then a second
round on bias window, lags, axis-flip search and multi-start) is kept in `hyperparameter_search.py`,
`hyperparameter_search_round2.py`, `reprocess_optimized_settings.py` and `validate_optimized_settings.py`. These are provenance,
not part of the reproduction: the last two compare against an earlier PSO + SSA + 2000-epoch run
(`outputs/rss/QArm_PSO_SO3_2000Epochs`) that is not shipped. The search found Adam to beat PSO and
SGD at equal epoch budgets (and to be about 13× cheaper than the original 200-particle, 2000-iteration PSO run), SSA
smoothing to hurt on this data, and the `piecewise_sparse_innovation` regularizer (together with a 1.0 s instead of
2.0 s bias window) to give the one robust further gain: mean |r| 0.765 → 0.807 over all 65 trials.

### Other scripts

| Script | Purpose |
| --- | --- |
| `build_merged_trials.py` | Split each batch into five tests and write `<session>/Merged/test_N.csv` (raw IMU window + ground truth). |
| `build_matched_rss_sources.py` | Pair recovered sources with ground truth → `matched/test_N.csv`. |
| `plot_matched_signals.py`, `best_results_report.py`, `best30_baseline_comparison.py` | Per-trial overlays, best-30 ranking, baseline comparison. |
| `practical_validation.py` | Original heavy audit/validation harness (PSO, null draws, bootstrap; `python -m qarm_rss_experiments.practical_validation`); also the shared helper library of the other scripts. |
| `plot_practical_results.py`, `generate_validation_report.py` | Plots and LaTeX report for that harness. |

`practical_validation` is hard-wired to the `OrbiX-Node-10_IMU1` gyro columns, so it only works on the 2026-09-18 session; build
the merged trials first:

```bash
python -m qarm_rss_experiments.build_merged_trials
python -m qarm_rss_experiments.practical_validation \
  --data-root "Sample_Data/QArm/session_2026-09-18" \
  --trials $(seq 1 30) --audit-only --output-root outputs/qarm_rss_validation_audit
```

`--audit-only` reproduces trial alignment, stationary-bias windows, quality eligibility and QArm geometry without
fitting any separator. Without it the default is the expensive publication run (2000 PSO iterations, 200 particles,
1000 random-SO(3) null draws per trial, 10 000 bootstrap resamples).

The motion limitations of the QArm itself matter for interpretation: it has a roll-pitch-pitch-roll layout, so
shoulder and elbow are parallel pitch joints. With the shoulder locked and the elbow near zero, base, elbow and wrist
are approximately orthogonal axes, but the setup is a practical test, not an ideal three-independent-axis one.

## 2. Hardware experiment bank (new recordings)

50 deterministic, one-minute joint-space experiments for a gripper-mounted IMU, generated from the same signal
generators as `notebooks/rss_validation.ipynb`. Shoulder, elbow and wrist are commanded; yaw and the gripper are
never written. Shoulder and elbow are parallel pitch joints, so a single gripper IMU cannot separate their
contributions uniquely: these experiments exercise RSS under a rank-deficient practical geometry and are not an
identifiable three-source ground-truth test.

- `data/qarm_rss_signals.npz`: compressed numeric trajectories and metadata; `data/signal_manifest.csv`: one row per experiment.
- `signal_bank.py`: deterministic generation, loading and safety validation; `generate_signal_bank.py`: regeneration entry point.
- `run_signal_bank_experiment.py`: guarded QArm hardware runner and feedback logger (requires the Quanser Python API, `quanser.hardware`).

| Signal numbers | Family | Count |
| --- | --- | ---: |
| 1-17 | Legacy modulated harmonic | 17 |
| 18-34 | Harmonic | 17 |
| 35-50 | Piecewise transient | 16 |

Each experiment has three zero-mean source angles and their absolute joint targets, 6 001 samples from 0 to 60 s
inclusive at 100 Hz. Regeneration is deterministic:

```bash
python -m qarm_rss_experiments.generate_signal_bank
```

### Safety design

The target centre pose is shoulder 0°, elbow −10°, wrist 0°. Every generated case is validated against shoulder
−75…+75°, elbow −85…+65°, wrist −150…+150°, a per-joint bank velocity of at most 19.5 °/s, and the QArm profile limits
(20 °/s, 1.0 rad/s²). The runner repeats all validation before connecting to hardware. This does not replace normal
laboratory safety: clear the arm workspace, secure the IMU and its cable, stay near the emergency stop and run signal 1
first while observing the robot.

### Running an experiment

1. Edit `SIGNAL_NUMBER` near the top of `run_signal_bank_experiment.py`.
2. On the QArm control computer: `python -m qarm_rss_experiments.run_signal_bank_experiment`.
3. Review the printed family, seed, ranges and peak speeds; the arm is not connected until you type the exact
   confirmation, e.g. `RUN 1`.

The runner moves to the initial sample with the QArm's capped position profile, counts down, then sends every sample
against absolute `time.perf_counter()` deadlines and aborts if the host is more than 50 ms late. Logs are written under
`qarm_rss_experiments/results/` (scheduled/elapsed/Unix timestamps, commanded and measured joint angles and
velocities). IMU acquisition is intentionally separate: align the QArm log and the IMU recorder with the Unix
timestamps and keep the IMU's own high-resolution timestamps for the RSS analysis. The offline tests
(`tests/test_qarm_rss_experiments.py`, `tests/test_qarm_practical_validation.py`) do not import the Quanser API.
