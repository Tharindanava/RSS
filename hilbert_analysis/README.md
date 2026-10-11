# Hilbert Analysis

A general-purpose Hilbert-transform / analytic-signal toolkit (`hilbert_analysis.analytic`), plus a batch pipeline built on it that cleans up three known data-quality problems in the RSS pipeline's `limb_angle_mode.csv` outputs:

1. **Truncated recordings** — files that stop mid-motion instead of returning to a quiet baseline are detected from a combined lower-limb short-time Hilbert envelope and, when `--apply` is used, moved into an `_Incomplete_Signals/` subfolder.
2. **Inverted mode signals** — a correct left/right pair is one waveform and a half-stride-delayed *copy* of it, so the sign of their cross-correlation at that lag says whether one side is inverted. When it is, the right side is negated together with its axis vector. Pairs too weakly correlated to judge are reported rather than guessed at.
3. **Segment/cycle labeling** — every processed CSV gets two new columns, `segment_label` (`start_transient` / `periodic` / `end_transient`) and `cycle_index` (1-based per detected gait cycle, `-1` outside the periodic region).

This does not modify `rss`, `HMA_Dashboard`, or `gait_analysis`.

## The reusable core: `hilbert_analysis.analytic`

```python
from hilbert_analysis import analytic_signal, envelope, instantaneous_phase, instantaneous_frequency

phase = instantaneous_phase(signal, unwrap=True)
env = envelope(signal)
freq_hz = instantaneous_frequency(signal, fs=100.0)
```

All four functions take a plain 1-D array and mean-detrend by default (`detrend="mean"`; also accepts `"linear"` or `None`). They raise `ValueError` on non-finite input rather than silently dropping samples, since a general-purpose transform must not desynchronize its output length from a caller's row-aligned data. `cycle_start_indices`/`cycle_intervals` find Hilbert-phase wrap-around crossings — useful for any periodic-motion cycle segmentation, not just this cleaning task.

## Run

From the repository root:

```bash
python -m hilbert_analysis \
  --input-root "outputs/rss" \
  --apply
```

Omitting `--apply` runs a dry-run: every file is analyzed and the report CSVs are written, but nothing on disk is moved or rewritten. This is safe to re-run repeatedly while tuning thresholds. Use `python -m hilbert_analysis --help` for the full option list.

To undo a previous run before re-processing, pass `--restore-source` with a pristine copy of the tree (also dry-run unless `--apply` is given):

```bash
python -m hilbert_analysis \
  --input-root "outputs/rss/Human_Motion" \
  --restore-source "outputs/rss_pristine/Human_Motion" --apply
```

This is the only way to undo an orientation pass wholesale — though re-running the pass is idempotent, since an already-oriented signal is reported `already_correct` and left alone.

Add `--diagnostic-plots [--diagnostic-limit N]` to also write one whole-body-envelope PNG per processed file (with onset/offset markers) under `outputs/hilbert_analysis/diagnostics/`, for spot-checking verdicts before trusting `--apply` on the full tree.

## Output

- **In place**: `--apply` adds the two marker columns to `limb_angle_mode.csv` and its row-aligned siblings (`limb_axes_mode.csv`, `limb_quaternion.csv`, `limb_quaternion_axis_angle.csv`, `ssa_drift_correction_signals.csv`, `ssa_preprocessing_signals.csv`), and moves incomplete leaf folders into `_Incomplete_Signals/` under the input root. `ssa_component_metrics.csv` is untouched — it is a per-component summary table, not per-frame.
- **Backup**: unless `--no-backup` is passed, each leaf folder is copied once to `outputs/hilbert_analysis_backup/<same relative path>/` before its first rewrite, since the source data comes from an expensive 2000-epoch PSO run.
- **Reports**, always written to `outputs/hilbert_analysis/` (dry-run or `--apply`):
  - `processing_summary.csv` — one row per leaf folder: status, completeness verdict + onset/offset/end-margin, pairs evaluated/flipped/consistent/low-confidence, stride estimate, cycle count.
  - `phase_flip_details.csv` — one row per leaf folder × paired limb × mode: the correlation and the lag it was read at, `lag_fraction` (should sit near 0.5), the stride estimate and its source, the status and reason, and whether a flip was actually applied.
  - `processing_errors.csv` — hard failures; one bad file never stops the batch.

## Algorithms

**Completeness** (`hilbert_analysis.completeness`): a single-shot Hilbert transform over an entire ~10-40 s recording leaks non-negligible envelope energy far from a short burst (the Hilbert kernel decays only as `1/t`, not exponentially), so the envelope is instead computed on short, non-overlapping windows (`analytic.windowed_envelope`, default 0.3 s, linearly detrended per window to avoid mistaking a settling drift for activity). Each of the 6 lower-limb paired-segment `original_angle` columns (`columns.LOWER_LIMB_SEGMENTS`) is expressed as a ratio to its own first-second baseline, then combined by **median** (not mean) across columns for robustness — upper-body/midline columns turned out to have near-zero baseline motion during quiet stance, which made a naive mean numerically unstable (division by near-zero). A sustained run (≥1.5 s, roughly one gait cycle, to reject brief pre-walking noise blips) above 3× baseline defines onset/offset; less than 0.5 s of quiet time after the last active run classifies the file as `incomplete`.

**Left/right pair correction** (`hilbert_analysis.phase_flip`). During gait the two limbs perform the same motion half a stride apart in *time*. A correct pair therefore looks like one waveform and a delayed copy of itself — peaks and valleys arriving alternately, one side then the other. If exactly one side is inverted, the delayed copy is *negated* rather than merely shifted, and that alternation breaks.

The measurement is the normalised cross-correlation between the two sides, read at a lag of **half a stride**:

| | meaning | action |
|---|---|---|
| `corr > 0` | right is a delayed **copy** of left | `consistent` — nothing to do |
| `corr < 0` | right is a delayed **negated** copy | `flipped` — negate the right side |
| `\|corr\| < --min-correlation` | too weakly related to judge | `low_confidence` — left alone |

The criterion is **relative**, and deliberately so. The pair relationship says that one of the two signals is inverted but not *which*, so `--reference-side` (default `left`) fixes which side is held and which is negated. That also means each pair's overall polarity is left as the RSS optimiser produced it — the pass makes pairs internally consistent, not globally canonical.

Two things the accuracy depends on. **The stride period** sets the lag being read, and comes once per recording from the whole-body reference (`segmentation.stride_period_samples`) rather than per signal — a per-signal estimate disagrees by more than 25% on 14% of signals and occasionally locks onto *twice* the stride, which inverts the verdict. **Waveform asymmetry** is what makes inversion distinguishable from a half-cycle shift at all; for a perfect sinusoid the two are identical. Real gait waveforms carry enough even-harmonic content, and where they do not `|corr|` falls below the gate.

The pair-consistency calculation remains available for legacy comparison, but rewriting is disabled by default. Pass `--legacy-pair-flip` explicitly to enable the old left/right writer. Dashboard reference-guided orientation and stored CSV signs are the normal analysis path.

`neck_to_left_shoulder` / `neck_to_right_shoulder` are anatomically a pair but excluded from `PAIRED_LIMBS`: the test is decisive on them (96%) yet judges 81% inconsistent versus ~50% everywhere else, which suggests the two clavicle segments are defined in mirrored directions and the "same waveform" expectation does not hold there.

When a side flips, its `{base}_mode_{k}_axis_{x,y,z}` vector in `limb_axes_mode.csv` is negated with it. The stored rotation is untouched by this: `*_original_angle` is the quaternion *composition* `Rot(θ₃,û₃) ∘ Rot(θ₂,û₂) ∘ Rot(θ₁,û₁)`, and `Rot(θ, û) = Rot(−θ, −û)` leaves each factor identical. `*_original_angle` is itself a non-negative magnitude and is **never** flipped.

**Three criteria that were wrong before this one**, recorded so they are not retried: Hilbert-phase cycle-start *timing* offsets (conflates two 180° contributions and inverts the verdict); canonicalising pairs to *opposite* polarity; and requiring each signal's analytic-plane **knot** to sit at positive values. The last was disproved by measurement — among pairs that are correctly related, the knot sits on the positive side for 29, the negative side for 48, and opposite sides for 21. `analytic.knot_position` remains available as a descriptive utility but takes no part in any decision.

**Segmentation** (`hilbert_analysis.segmentation`): `segment_label` boundaries come directly from the completeness onset/offset. `cycle_index` is detected via Hilbert-phase cycle starts on a combined whole-body phase reference — the sum of the z-scored lower-limb `mode_1` signals, each sign-normalised **in memory** (never written back) so the segments reinforce instead of cancelling under the random RSS gauge. Two details matter: `*_original_angle` never crosses zero and makes a poor phase reference, and because the sides are half a gait cycle apart in time, summing them as-stored beats at *step* rate and doubles the cycle count, while opposing the sides recovers the stride-rate fundamental — measured `std = 1.66` vs `1.11`, and per-trial counts of ~16 strides rather than ~29 steps. `cycle_index` therefore numbers **strides**, not steps.

All thresholds are CLI-configurable; see `hilbert_analysis/cli.py` or `--help`.
