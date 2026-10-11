# HMA Dashboard

An interactive [Dash](https://dash.plotly.com/) application for inspecting RSS output folders, running
the classic signal analyses on any recovered mode, checking and fixing the free sign gauge of RSS modes,
and launching the RSS pipeline on new raw IMU recordings.

It sits on top of the analysis packages in this repository: `hilbert_analysis` (analytic signal, phase,
frequency), `rotating_poincare` (Poincaré transform), the transfer-entropy estimators from
`gait_analysis`, and `rss` (pipeline jobs).

## Quick start

From the repository root, generate synthetic demo outputs (no human data is shipped) and open the dashboard on them:

```bash
python scripts/make_demo_outputs.py --output-root outputs/rss/Demo_Synthetic
python -m HMA_Dashboard --results-dir outputs --host 127.0.0.1 --port 8050 --default-fs 100.0
```

then browse to <http://127.0.0.1:8050>. With your own data, point `--results-dir` at the folder that
contains your RSS output leaves (any folder holding `limb_angle_mode.csv` or `joint_angle_mode.csv`).

Useful options:

- `--smoke-test`: scan the output and raw trees, build the layout, load one run from each available output family, and exit
- `--debug`: run the Dash server in debug mode
- `--sample-data-dir`: root of the raw IMU input tree (default `Sample_Data`). That folder also holds the QArm recordings, which the dashboard lists as unusable raw files; pass `--sample-data-dir Sample_Data/Human_Motion` to show only human-motion recordings
- `--gait-analysis-results`: root of `gait_analysis` runs (default `outputs/gait_analysis`); optional — step markers are simply unavailable without it
- `--read-only`: hide the **Data & Runs** tab (no uploads, no pipeline jobs); flip saves and the report/flip logs still write
- `--python-bin`, `--max-upload-mb`, `--max-concurrent-jobs`: pipeline-job settings (`--python-bin` defaults to the interpreter running the dashboard)

## Expected data layout

The dashboard discovers every `limb_angle_mode.csv` (limb family, RSS) and `joint_angle_mode.csv` (joint family)
below `--results-dir`. When the path contains a `Human_Motion` component, the levels after it are read as

```text
Human_Motion/<subject>/<terrain>/<experiment>/<csv-stem>/limb_angle_mode.csv
```

(an extra `<activity>` level between terrain and experiment is also accepted) and drive the subject / terrain /
activity / experiment / file selectors. Other layouts still load, with generic labels. The raw tree under
`Sample_Data` mirrors the same hierarchy.

## Features

- a signal workbench for time-series, **FFT**, PDF, CDF, **phase portrait**, Poincaré transform, **Hilbert** analytic signal, instantaneous phase, instantaneous frequency, joint PDF, and configurable pairwise **transfer entropy**
- hierarchical file selection by subject, terrain, activity, experiment, and CSV file
- a lower-limb raw-TE matrix view for same-run joint files with complete Hip/Knee/Ankle sets
- a run-overview tab with Mode 1/2/3 browsing, CSV export, and an all-segment transfer-entropy matrix
- reference-guided Mode 1 orientation checks per scenario/side, with group-level Modes 1–3 previews
- transactional **Save flip to source CSVs**, which negates the selected group's three modes and nine matching axis columns while preserving `original_angle`
- zoom-based valley-to-valley reference-step selection for each scenario/side
- a **Data & Runs** tab for uploading raw IMU CSVs and running the RSS pipeline

## Instantaneous phase and frequency

Both plot kinds come from `hilbert_analysis.analytic`. *Inst. Phase* is the argument
of the analytic signal, wrapped to `(-π, π]`; tick the **Unwrap phase** box to see the
cumulative phase instead. *Inst. Frequency* is the derivative of the unwrapped phase,
scaled to Hz by the file's sampling rate and plotted against `x[1:]` (a backward
difference belongs to the later sample of its pair). When a file has no usable time
column the frequency is reported in cycles/sample and the figure says so, rather than
mislabelling a per-sample rate as Hz.

## Segment transfer-entropy matrix

The Run Overview tab's **Compute TE Matrix** gives the directed transfer entropy
between the Mode 1 angle of every segment in the selected trial (rows drive, columns
receive). It defaults to **25 bins** and a history **lag of 50 samples** (length and
stride fixed at 1); both are editable, and `te_matrix.DEFAULT_BINS` /
`te_matrix.DEFAULT_LAG` are the single source for the inputs and their fallbacks.

Segments run head to foot, with left before right at each level so mirrored segments
sit side by side. The matrix is boxed into four sections — upper→upper, upper→lower,
lower→upper and lower→lower (source row region → target column region) — with the
region names in the margins. "Upper" is the neck, the spine above the pelvis and the
arms; "lower" is the pelvis, the lower spine and the legs (`te_matrix.UPPER_LIMB_SEGMENTS`
/ `LOWER_LIMB_SEGMENTS`). Files with only one region, or with joint-family groups,
are shown unboxed.

> **Caution:** at 25 bins the joint state space (~15.6k cells) is large against a few
> thousand samples, so the plug-in estimate is strongly upward-biased. On a 4.3k-sample
> trial with every segment independently shuffled (true TE = 0) the mean cell reads
> ~1.9 bits at 25 bins, versus ~0.07 bits at 8 bins, and above the real-data mean at
> 25 bins / lag 50. Read the relative pattern, not the absolute bits, at these settings.

## Manual sign flips

The RSS real-data pipeline does not canonicalise mode signs — each mode's sign is whatever the optimiser
converged to (modes keep the optimiser's order unless `--rss-mode-order amplitude` is given; the dashboard's pipeline
preset does). `hilbert_analysis` can test left/right pairs for inconsistency; by default it only reports, and it
deliberately leaves pairs it cannot judge confidently alone — mostly upper limbs, whose walking waveforms are not
asymmetric enough to decide from. **Invert sign (preview)** negates the selected group's signals (Modes 1–3) in every
plot that shows them, keyed by `(file, group)` so the workbench and the overview always agree. **Save flip to source
CSVs** then makes it permanent in one transaction: it negates `{group}_mode_1..3` in `limb_angle_mode.csv` *and* the nine
matching axis components in `limb_axes_mode.csv`, so `Rot(θ, û) == Rot(-θ, -û)` leaves the stored rotation identical and
only the sign convention changes.

Commit is only allowed for a group that is currently previewed, so a stray second
click cannot silently flip the data back. Before the first rewrite in a leaf, the two
affected CSVs are copied to `<results-dir>/hma_dashboard_flip_backup/<relative path>/` (`outputs/…` with the default
`--results-dir`), and every flip is appended to `hma_dashboard_manual_flips.csv` in the leaf itself.
`*_original_angle` is a single-signed magnitude and is never written.

## Faulty-signal reports and flip history

**Report faulty signal** (in every flip row) opens one shared dialog for the signal
currently selected in that panel: pick a reason, add a note (required for *Other*), and
submit. A report only appends a row to a log; it never touches the run outputs. The
dashboard keeps two separate append-only CSV logs at the root of the results directory:

| File                                        | One row per          | Key columns                                                                                                                                                                                                                       |
| ------------------------------------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hma_dashboard_faulty_signal_reports.csv` | faulty-signal report | `timestamp_utc`, `report_id`, subject/terrain/activity/experiment/`csv_file`, `mode_file`, `group_name`, `trace_name`, `reason_code`, `reason`, `note`, `previewed_flip`, `committed_flips`, `view_scope` |
| `hma_dashboard_flip_history.csv`          | flip action          | `timestamp_utc`, `event` (`preview_on` / `preview_off` / `commit`), the same signal columns, `view_scope`, `operation_id`, `detail`                                                                               |

`previewed_flip` and `committed_flips` record the sign state the signal was in when it
was reported, so a report can be read against the flips applied to it. The flip history
covers every dashboard flip action — **Invert sign (preview)**, **Check Orientation**
and **Clear Preview Flips** as well as saved commits — and a `commit` row carries the
same `operation_id` as the matching rows in that leaf's `hma_dashboard_manual_flips.csv`,
which stays the transactional audit used for net parity and rollback. Both logs refuse
to append under a header they do not recognise rather than misalign later rows. The
reason list is `signal_logs.FAULT_REASONS`; keep the codes stable once logged.

> **Note:** like the flip commit, these writes are not blocked by `--read-only`, which
> only hides the Data & Runs tab.

## Data & Runs

Uploading validates the CSV with the pipeline's own loader before storing anything,
writes it through a temporary file in the destination directory so nothing partial is
ever published, and refuses to overwrite an existing file unless **Replace** is ticked.
Destination paths are re-validated server-side and confined to `Sample_Data`.

Starting a run launches `python -m rss run-config` as a detached subprocess using the
production flag set defined in `HMA_Dashboard/jobs.py` (SGD, SO(3), zero-lag cross-correlation,
SSA preprocessing), with **epochs**, **SSA window length**, and **compute device** overridable. Expect roughly 7–28 minutes per file at the
2000-epoch preset (measured 424–1681 s, median 619 s); at most two jobs run at once and
the rest queue. Each run stamps `rss_run_params.json` into its output leaf, so a
dashboard run is never indistinguishable from a batch result beside it. Job state lives
on disk under `<results-dir>/rss/_dashboard_jobs/<job_id>/`, so runs survive a dashboard
restart, and the file list refreshes automatically when a job finishes.

> **Note:** this tab lets anyone who can reach the server write files and start
> compute, and the Dash dev server has no authentication. Bind to `127.0.0.1` (the
> default) and use SSH port-forwarding for remote access, or start it with
> `--read-only`.
