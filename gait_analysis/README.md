# gait_analysis

Gait analysis of the signals RSS recovers from body-worn IMUs. It turns a set of recordings (raw IMU logs plus their
RSS `limb_angle_mode.csv` outputs) into gait events and steps, classic temporal gait features, functional-PCA
shape descriptors, transfer-entropy networks, trial phenotypes, clusters and statistics. The method is written up in
[`docs/Gait_Two_Stage_Analysis_Methodology.pdf`](../docs/Gait_Two_Stage_Analysis_Methodology.pdf).

The package is also what [`HMA_Dashboard`](../HMA_Dashboard) builds on: its reference registry, condition metadata,
transfer-entropy discretisation and gait-step overlays come from here (`references.py`, `metadata.py`,
`te_estimator.py`, `dashboard.py`).

> **Data.** No recordings are distributed with this repository. The package expects raw recordings under
> `Sample_Data/Human_Motion/<subject>/<terrain>/…` and the matching RSS outputs under `outputs/rss/Human_Motion/…`
> (see [`Sample_Data/README.md`](../Sample_Data/README.md)); change `paths` in `configs/default.yaml` to use other
> locations. `scripts/make_demo_outputs.py` writes synthetic RSS-style outputs for trying the dashboard.

## Scientific guardrail

Every stored Mode 1 signal is treated as an `rss_segment_mode1_proxy`: a latent segment-motion proxy, not a validated
anatomical joint angle. Analysis uses the orientation (sign) stored in the source CSV and never reorders modes. An
optional sign canonicalisation from per-segment semantic axes exists for comparison
(`proxy.orientation_method: semantic_axis_legacy`, needs a profile JSON that is not included). The toolkit does not
estimate stride length, walking speed, absolute posture or causal physiology. Recordings that were split at a removed turn
(`Phase_01`, `Phase_02`, …) stay separate straight-walking legs that share a `trial_id`; the trial — not the gait step or a
transfer-entropy window — is the inferential and clustering unit.

## Modules

| Module | Purpose |
| --- | --- |
| `metadata.py` | recording manifest, normalised recording identity and condition metadata (terrain, direction, footwear, activity) |
| `io.py` | readers for the raw event sensors and for the stored RSS Mode 1 proxies |
| `segmentation.py` | ankle-driven Hilbert-phase gait-cycle segmentation (events and valley-to-valley steps) |
| `classical_features.py` | classic temporal gait features (timing, range of motion, variability, bilateral symmetry) and per-step normalised trajectories |
| `mfpca.py`, `node_fpca.py` | weighted multivariate and per-node functional PCA of normalised step trajectories |
| `te_estimator.py`, `te_network.py` | discrete transfer-entropy primitives and whole-body transfer-entropy window networks |
| `phenotype.py`, `spectral_clustering.py`, `statistics.py` | trial phenotypes, spectral clustering with participant-block stability, mixed-effects estimates |
| `visualization.py` | manuscript-style figures and their manifest |
| `references.py` | dashboard reference registry (one reference recording per terrain/direction/footwear/side) and signal templates |
| `dashboard.py` | read-only adapter that feeds gait events and steps to the dashboard |
| `pipeline.py`, `cli.py`, `config.py`, `models.py`, `progress.py`, `utils.py` | resumable stage orchestration, command line, validated YAML configuration, data contracts |

## Running the stages

From the repository root (needs the `statsmodels`, `scikit-learn` and `PyYAML` entries of `requirements.txt`):

```bash
python -m gait_analysis manifest   --profile development
python -m gait_analysis segment    --profile development --limit 3
python -m gait_analysis stage1     --profile development --limit 3     # classic features + MFPCA
python -m gait_analysis stage2     --profile development --limit 3     # transfer-entropy networks
python -m gait_analysis cluster    --profile development
python -m gait_analysis statistics --profile development
python -m gait_analysis figures    --profile development
```

`python -m gait_analysis all` runs the whole chain; use it for a final run once the development checks and sensitivity
choices are settled. `--force` bypasses derived caches, `--config` selects another YAML file and `--sensitivity` applies
validated YAML overlays (repeatable):

```bash
python -m gait_analysis stage2 \
  --sensitivity gait_analysis/configs/sensitivity/te_bins_10.yaml \
  --sensitivity gait_analysis/configs/sensitivity/te_cycle_block.yaml
```

The CLI prints timestamped, line-buffered progress (manifest discovery, each recording, cache hits, MFPCA resampling,
transfer-entropy windows, clustering bootstraps, statistics, figures, isolated failures). For a long run in the background:

```bash
mkdir -p outputs/gait_analysis
PYTHONUNBUFFERED=1 nohup python -u -m gait_analysis all --profile final \
  > outputs/gait_analysis/full_final.log 2>&1 &
tail -n 50 -F outputs/gait_analysis/full_final.log
```

## Stages and artifacts

Runs are written to `outputs/gait_analysis/runs/<profile>-<manifest-hash>-<config-hash>-<reference-hash>`. Per-recording
caches live under `outputs/gait_analysis/cache` and are keyed by source checksums plus the complete configuration and
reference-registry hashes. Tables use atomic replacement, failures are appended to `errors.csv`, and raw or RSS inputs are
never edited.

- `manifest.csv`: every raw source, normalised condition fields, phase/trial identity, checksums, sensor availability, stage flags, and structured exclusions.
- `gait_events.csv`, `gait_steps.csv`: independently detected left/right events and accepted/rejected valley-to-valley steps, boundary provenance, confidence, snap offsets, reference similarity, and QC.
- `proxy_signal_qc.csv`, `normalized_step_trajectories.csv.gz`: stored-orientation QC and 101-point trajectories normalised on each side's own boundaries, combined by `step_pair_id`.
- `classic_step_features.csv`, `classic_trial_features.csv`: temporal, range-of-motion/extrema/peak-timing, variability, bilateral raw values and bounded symmetry summaries. No spatial estimates.
- `mfpca/`: model arrays, phase grid, weighted eigenfunctions, `mfpca_step_pair_scores.csv`, reconstruction error, trial summaries and stability diagnostics.
- `te_edges.csv.gz`, `te_network_features.csv`: all 132 directed edges per complete eight-step-pair window plus predeclared node/block/network reductions.
- `phenotype_absolute.csv`, `phenotype_delta.csv`: complete-case, family-balanced trial phenotypes.
- `clusters.csv`, `cluster_selection.csv`, `cluster_stability.csv`: absolute and separate matched terrain/footwear spectral solutions, coordinates, selection criteria and robustness.
- `statistical_results.csv`, `figure_manifest.csv`, `run_metadata.json`: mixed-effects and paired estimates, participant-first uncertainty and leave-one-out checks, traceable figures, hashes and guardrails.

## Configuration and sensitivities

All thresholds, event sensors, node mappings, seeds, primary outcomes and selection rules are in
[`configs/default.yaml`](configs/default.yaml). [`configs/sensitivity/`](configs/sensitivity) holds declared reruns:
foot/shank and gyro-only segmentation, registration/weighting MFPCA variants, transfer-entropy bins/history/null/common-phase
variants, phenotype family ablations and a global-RBF affinity. Use these overlays instead of editing code.

## Use by the dashboard

`HMA_Dashboard` accepts `--gait-analysis-results` (default `outputs/gait_analysis`). It matches a displayed RSS recording
through the manifest and reads markers only from `gait_events.csv` and `gait_steps.csv`. Event overlays can appear when
steps are unavailable; step overlays require a pipeline run.

The shared `outputs/hma_dashboard_references.json` registry stores one reference recording per terrain/direction/footwear/side,
including same-side Mode 1 templates and an optional reference step. Its checksum is part of every run identity, cache key,
output table and provenance record.

## Tests

```bash
python -m pytest tests/gait_analysis
```

The suite covers manifests and header variants, phase grouping, hybrid event rules, classic features, weighted MFPCA,
delayed transfer entropy, network FDR, spectral clustering, cache/determinism, and strict dashboard sidecar lookup.
`tests/gait_analysis/fixtures/annotation_import_template.csv` defines a future expert-review import format; it is a schema
fixture, not ground truth.
