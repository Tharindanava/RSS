# Rotation Source Separation (RSS)

This package implements Rotation Source Separation with a general
three-parameter SO(3) recovery matrix by default. A six-parameter symmetric
matrix remains available as the explicit legacy mode. RSS is independent of
the SOBI initializer and selects one primary separation loss in the recovered
angle domain: lagged cross-correlation, KSG/legacy soft-histogram mutual
information, or torsional rotational action. Lagged cross-correlation is the
default with `lags=(0,)`, where lag `0` is the former zero-lag correlation
term. Quaternion reconstruction remains available as an optional constraint. The
soft matrix orthogonality constraint is used only by the legacy symmetric mode;
SO(3) recovery is proper orthogonal by construction. The reconstruction composes
`qS3 ⊗ qS2 ⊗ qS1` about the recovered source axes, forms
`qS^-1 ⊗ qR`, and minimizes its closest-sign squared four-component distance
to the identity quaternion. This is `min(||qDelta-I||^2, ||qDelta+I||^2)`, so
antipodal quaternions represent the same physical rotation and the loss lies
in `[0, 2]`. The
optimized matrix is interpreted as a direction-cosine matrix: its entries are
source/observer-axis dot products, not global physical-axis coordinates.

## Package map

| Module | Purpose |
| --- | --- |
| `model.py`, `quaternion_series.py`, `axes.py` | quaternion composition/basis change of the RSS model, vectorised quaternion series helpers, direction-cosine → source-axis helpers |
| `solver.py` | the fit: SO(3)/symmetric recovery matrix, losses, Adam / SGD / particle-swarm optimisers, adaptive regularizer, axis-flip search (`fit_rss`, `RSSConfig`) |
| `mutual_information.py`, `specialized_losses.py` | MI estimators (GCMI, B-spline, KSG, soft histogram) and the four signal-family losses |
| `simulation.py`, `evaluation.py`, `artifacts.py`, `plotting.py` | deterministic synthetic scenarios, ground-truth evaluation (matching, errors, correlations), artifact writers and plots |
| `ssa_preprocessing.py` | singular-spectrum drift removal and denoising for recorded signals |
| `sensor_layout.py`, `imu_input.py` | body-segment ↔ IMU-channel mapping and loading/validation of wide-format multi-IMU CSV files |
| `realdata_config.py`, `realdata_pipeline.py`, `exporters.py` | processing of **recorded** IMU data ("real data", as opposed to simulation): JSON run config, per-limb pipeline, CSV/plot exporters |
| `cli.py`, `__main__.py` | `python -m rss {validate,run-config,process-imu}` |

The recovery matrix can be initialized directly through `RSSConfig`. In the
default `so3` mode, the supplied matrix must be a proper orthogonal rotation
matrix; in legacy `symmetric` mode, it must be symmetric. Leaving it as `None`
retains identity initialization:

```python
initial_matrix = (
    (0.0, -1.0, 0.0),
    (1.0,  0.0, 0.0),
    (0.0,  0.0, 1.0),
)
config = RSSConfig(matrix_parameterization="so3", initial_weight_matrix=initial_matrix)
```

The CLI equivalent is `--rss-initial-weight-matrix`, followed by the nine
row-major entries. `initial_rotation_vector` and the legacy six-value
`initial_weights` remain supported when no matrix is supplied.

Run one synthetic validation:

```powershell
python -m rss validate --output-root outputs/rss_validation --epochs 500
```

Adam is the default optimizer. Select momentum SGD or particle swarm
optimization with `--rss-optimizer sgd` or `--rss-optimizer pso`. Every
optimizer uses an endpoint-inclusive exponential learning-rate schedule from
`--learning-rate-start` (legacy alias: `--learning-rate`) to
`--learning-rate-end`, defaulting to `1e-1` and `1e-5`. Adam and SGD apply it
as their native learning rate; PSO multiplies each position step by it while
retaining its velocity equation. PSO interprets `--epochs` as swarm iterations
and uses its own `--rss-pso-*` controls. The defaults are 40 particles, inertia
`0.7298`, cognitive/social coefficients `1.49618`, SO(3) rotation-vector
bounds `[-pi, pi]`, seed `0`, and 20 stalled iterations before convergence.
Legacy symmetric mode uses bounds `[-1, 1]`. Run `python -m rss validate -h`
or `python -m rss run-config -h` for every optimizer option.

Process all files listed in an existing configuration:

```powershell
python -m rss run-config `
  --config config.example.json `
  --output-root outputs\rss `
  --sample-rate-hz 100 `
  --include-time-seconds `
  --rss-optimizer adam `
  --learning-rate-start 1e-1 `
  --learning-rate-end 1e-5 `
  --rss-loss-type lagged_cross_correlation `
  --rss-lags 0 `
  --epochs 500
```

Process a single rigidly-mounted IMU's gyroscope channels directly — for a
rig with only one sensor, such as a robot-arm end effector, rather than the
multi-limb wide-CSV schema `run-config` expects:

```bash
python -m rss process-imu \
  --csv path/to/combined_log_wide_<timestamp>.csv \
  --node-prefix OrbiX-Node-10_IMU1 \
  --sample-rate-hz 100 \
  --bias-window-seconds 2.0 \
  --rss-optimizer adam \
  --epochs 500
```

`--node-prefix` expands to `<prefix>_GX/_GY/_GZ`; use `--gyro-columns GX GY GZ`
for other column-naming conventions. The command averages the first
`--bias-window-seconds` of gyro data as a static bias (the sensor must start
stationary), integrates the bias-corrected gyro to body-rotation angle
channels, applies SSA preprocessing by default (disable with
`--disable-ssa-preprocessing`), and fits RSS with
`--rss-observer-signal-model` defaulted to `integrated_body_rotation`. It
writes recovered `source_1/2/3` angle channels to `--output` (default:
`<csv stem>_rss_sources.csv` next to `--csv`) and prints the recovery matrix
and fit diagnostics as JSON. It shares every `--rss-*` solver flag with
`validate` and `run-config`. A rank-deficient rig (for example, two parallel
robot-arm joints observed by one IMU) is not an identifiable three-source
separation; RSS still fits the best available SO(3) recovery, but per-source
labels in that case describe combined rather than individually-identifiable
motion.

All MI backends return natural-log units (nats) and preserve the existing mean
over source pairs `(0, 1)`, `(0, 2)`, and `(1, 2)`. Select a backend with
`--rss-mi-estimator {gcmi,bspline,ksg,histogram}`.

GCMI applies marginal ranks, an inverse-normal transform, and the scalar
Gaussian-copula expression `-0.5 log(1-rho^2)`. Average ranks are used by
default and can be changed with `--rss-gcmi-rank-method`. GCMI is a fast
Gaussian-copula contrast rather than a replacement mathematically identical
to KSG: it can underestimate symmetric non-monotonic dependence. Ranking is
non-differentiable, so GCMI requires PSO.

B-spline MI assigns every sample soft membership in a clamped B-spline basis,
then forms marginal and joint probabilities from those memberships. Its
defaults are six bins and mathematical spline order three (quadratic basis),
configured with `--rss-bspline-bins` and `--rss-bspline-order`. The Torch
implementation is vectorized over PSO particles and remains differentiable for
Adam and SGD.

KSG-1 follows Kraskov, Stögbauer, and Grassberger (2004), estimator `I^(1)`,
Eq. (8). It uses Chebyshev distance, strict marginal neighbour counts, and raw
estimates that can be slightly negative at finite sample sizes. The fixed
neighbour count defaults to `k=4`; KSG standardizes local variable copies by
default, and repeated coordinates receive deterministic conditional jitter.
Classical KSG requires PSO.

Full-data KSG remains the default. Set `--rss-ksg-max-samples 1000` (or another
positive value) to select one deterministic paired subset at the start of an
RSS run and reuse it for every candidate and all three source pairs. Configure
that subset with `--rss-ksg-subsample-seed`; use
`--rss-ksg-max-samples none` to explicitly select every observation.

The `histogram` backend retains the original differentiable Gaussian
soft-histogram estimator unchanged. Its defaults are 16 bins over standardized
coordinates and bandwidth `0.5`, configured with `--rss-mi-bins` and
`--rss-mi-bandwidth`. Examples:

```bash
python -m rss validate --rss-loss-type mutual_information \
  --rss-mi-estimator gcmi --rss-optimizer pso
python -m rss validate --rss-loss-type mutual_information \
  --rss-mi-estimator bspline --rss-bspline-bins 6 --rss-bspline-order 3 \
  --rss-optimizer adam
python -m rss validate --rss-loss-type mutual_information \
  --rss-mi-estimator ksg --rss-ksg-k 4 --rss-ksg-max-samples 1000 \
  --rss-ksg-subsample-seed 0 --rss-optimizer pso
python -m rss validate --rss-loss-type mutual_information \
  --rss-mi-estimator histogram --rss-optimizer adam
```

```python
gcmi = RSSConfig(
    loss_type="mutual_information", mi_estimator="gcmi", optimizer="pso",
)
bspline = RSSConfig(
    loss_type="mutual_information", mi_estimator="bspline",
    bspline_bins=6, bspline_order=3, optimizer="adam",
)
capped_ksg = RSSConfig(
    loss_type="mutual_information", mi_estimator="ksg",
    ksg_k=4, ksg_standardize=True, ksg_max_samples=1000,
    ksg_subsample_mode="fixed_random", ksg_subsample_seed=0, optimizer="pso",
)
legacy = RSSConfig(
    loss_type="mutual_information", mi_estimator="histogram",
    mi_bins=16, mi_bandwidth=0.5, optimizer="adam",
)
```

For estimator-only validation, `rss.estimate_mutual_information(x, y, ...)`
accepts scalar arrays for every backend and multivariate `(N, d)` variables for
KSG. The paired sample counts must match. `rss.select_ksg_subsample_indices`
exposes the exact deterministic selection used for capped KSG experiments.

Select `--rss-loss-type action` for the mean magnitude of
the three per-mode time-averaged torsional actions. Configure its moments of
inertia and stiffnesses with `--rss-action-inertias I1 I2 I3` and
`--rss-action-stiffnesses K1 K2 K3`. Simulations use their timestamps and real
data uses `--sample-rate-hz` for the action timebase. Direct `fit_rss` calls can
set `RSSConfig.sample_interval_seconds`.

## Adaptive specialized regularization

RSS can keep lagged cross-correlation as its base objective while adaptively
adding one of the four signal-family losses developed in
`notebooks/rss_validation_specialized_losses.ipynb`:

- `harmonic_spectral_overlap`
- `filtered_multilag_cross_correlation`
- `chirp_time_frequency_overlap`
- `piecewise_sparse_innovation`

The feature is disabled by default. Enable it and choose the loss explicitly:

```bash
python -m rss validate \
  --rss-observer-signal-model integrated_body_rotation \
  --rss-adaptive-regularizer \
  --rss-specialized-regularizer harmonic_spectral_overlap \
  --rss-regularizer-start-weight 0.1 \
  --rss-regularizer-plateau-patience 20 \
  --rss-regularizer-plateau-tolerance 1e-4 \
  --rss-regularizer-ramp-epochs 20
```

The configured `--rss-lags` remain active. After a base-loss plateau, the
specialized weight ramps to `1`; after a specialized-loss plateau it ramps to
`0`, and the cycle repeats. Best-state comparisons use a stable full objective
whose specialized weight is always `1`. Set
`--rss-regularizer-min-loss-threshold` to a positive normalized full-objective
value for early stopping; its default `0` preserves the full epoch budget.
All four losses need a known sample interval, which validation and
`run-config` supply automatically.

Formula-specific controls keep their notebook defaults and are intentionally
programmatic rather than CLI options. For example:

```python
from rss import RSSConfig, SpecializedLossConfig

config = RSSConfig(
    adaptive_regularizer=True,
    specialized_regularizer="filtered_multilag_cross_correlation",
    specialized_loss_config=SpecializedLossConfig(filtered_max_lag=50),
)
```

Adaptive regularization supports Euler and integrated-body-rotation inputs and
all three optimizers. It requires lagged cross-correlation, a selected
specialized loss, at least four samples, and a valid sample interval.

## Periodic proper-axis flip search

Use `--rss-axis-flip-search` to compare the current SO(3) solution with the
three proper double-axis flips `xy`, `xz`, and `yz` every
`--rss-axis-flip-interval` accepted epochs. Each candidate receives
`--rss-axis-flip-trial-epochs`; the trial window counts once against
`--epochs`, although evaluating four branches increases runtime. For example:

```bash
python -m rss validate \
  --rss-observer-signal-model integrated_body_rotation \
  --rss-axis-flip-search \
  --rss-axis-flip-interval 100 \
  --rss-axis-flip-trial-epochs 10
```

Flip search requires `matrix_parameterization=so3`, a trial length smaller
than the interval, and bounds covering the complete rotation-vector range
`[-pi, pi]`. Single-axis flips are intentionally excluded because they produce
an improper matrix with determinant `-1`. As with all unsupervised
sign-symmetric separation losses, this branch search explores optimizer basins
but cannot determine a ground-truth axis sign without an external reference.
The bounded paired implementation report is available in
[`docs/adaptive_axis_flip_report.md`](../docs/adaptive_axis_flip_report.md).

The default real-data outputs retain the established schemas:

- `limb_angle_mode.csv`
- `limb_axes_mode.csv`
- `limb_quaternion.csv`
- `limb_quaternion_axis_angle.csv`

Use repeatable `--enable-output` and `--disable-output` options for additional
observed-signal, coefficient, RSS diagnostic, and SSA outputs. The optional
RSS loss history contains normalized total and weighted-component
contributions alongside the selected `loss_type`, applied `learning_rate`, and
raw `total_loss`, `separation_loss`, `orthogonality_loss`, and
`reconstruction_loss`. Enhanced runs also add the specialized and stable
selection losses, current regularizer weight/phase, normalized stop threshold,
and flip-candidate diagnostics. Each fit uses the fixed reference
`max(raw_total_loss_at_epoch_0, eps)`, so the normalized initial objective is
normally `1.0`; every optimizer uses normalized values for best-state
selection, convergence, and loss plots. Adam and momentum SGD use PyTorch
automatic differentiation. PSO is gradient-free and its history plots the
global-best-so-far objective. The raw orthogonality term is defined as
`||A A^T - I||_F^2 / 3`. Physical source axes can only be expressed in a
global or anatomical frame when the corresponding observer-axis basis is
available; otherwise exported axes are expressed in the sensor XYZ frame.
The default observer channels are Euler-angle signals produced by SSA-cleaned
real-data inputs. RSS encodes those channels into quaternions and solves for
source mixing. Use `--rss-observer-signal-model integrated_body_rotation`
only when the input channels are componentwise integrated body-frame rotation
increments, such as the synthetic body-fixed gyro simulation path.
Select `--rss-observer-signal-model euler --rss-matrix-parameterization symmetric`
to reproduce the legacy Euler/vector-mixing interpretation.

See `notebooks/rss_validation.ipynb` for the single-case, 20-case family, and
eight-case large-rotation validation workflows. Synthetic source angles are
symmetrically whitened and then centered to zero temporal mean before
quaternion mixing, while retaining their centered peak amplitudes, so every
absolute zero-lag Pearson cross-correlation is below
`SimulationConfig.source_cross_correlation_limit` (default `0.01`).
Validation metrics and JSON artifacts report the full ground-truth and matched
separated-source correlation diagnostics together with quaternion
reconstruction loss and angular error. Simulation source and observer bases
are sampled independently as proper right-handed bases. The general
direction-cosine truth matrix has determinant `+1`. Both triads are attached to
the virtual object and rotate with its composed quaternion; body-frame
increments provide the synthetic gyroscope measurements. The zero-mean source
traces provide both amplitude metadata and recovery ground truth. Their first
sample may be nonzero, so the first body-frame increment can encode a
nonidentity initial orientation. Integrated observer channels remain physically
derived from the source rotations and are not independently centered.
