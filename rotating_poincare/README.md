# Rotating Poincaré Section Transform

This package analyzes a scalar limb or joint-angle signal through its phase
portrait and a family of Poincaré sections. It is a standalone analysis tool:
it reads CSV files and does not change the `rss` pipeline or its output schemas.

## Definition

For an angle signal `x(t)`, the phase trajectory is

```text
p(t) = [x(t), dx(t)/dt].
```

The trajectory is mapped coordinate by coordinate to `[-1, 1]`,
parameterized by cumulative arc length, and interpolated at high resolution.
For every section angle `theta` in `[0, 360)` the transform uses the positive
ray from the origin with along-section vector

```text
e(theta) = [cos(theta), sin(theta)]
```

and normal

```text
n(theta) = [-sin(theta), cos(theta)].
```

Consecutive trajectory points cross the supporting line when their signed
distances `n(theta) dot p` change sign. Each crossing is linearly interpolated
and projected onto `e(theta)`. Only crossings with nonnegative radial
coordinate `r >= 0` belong to the positive ray and are accumulated in its
radial histogram. Stacking those histograms produces the RPT count matrix:

```text
rows    = section angles
columns = radial bins
values  = intersection counts
```

## Run

From the repository root in PowerShell:

```powershell
python -m rotating_poincare `
  --input outputs\rss\your_run\limb_angle_mode.csv `
  --output-dir outputs\rotating_poincare `
  --time-col time `
  --signal-cols left_hip_to_knee_original_angle left_knee_to_ankle_original_angle `
  --interp-points 5000 `
  --theta-step-deg 1 `
  --num-radial-bins 80 `
  --crossing-direction both
```

One invocation can process multiple files:

```powershell
python -m rotating_poincare `
  --input path\run_1.csv path\run_2.csv `
  --output-dir outputs\rotating_poincare
```

When `--signal-cols` is omitted, every numeric column other than the resolved
time column and index-like columns such as `frame` is analyzed. Time columns
named `time`, `time_seconds`, or `Timestamp` are detected automatically.
Without a time column, pass `--fs 100`; if neither source is available, unit
sample spacing is used and recorded as a warning.

Unit-range normalization and derivative smoothing are enabled by default.
Disable them with `--no-normalize` and `--no-derivative-smoothing`. Use
`python -m rotating_poincare --help` for all interpolation, angle-grid, radial,
and plotting options.

## Demo

Generate deterministic clean-limit-cycle and noisy-ellipse examples, then
analyze both:

```powershell
python -m rotating_poincare --demo
```

The default demo directory is `results\rpt_demo`. Use `--output-dir` to change
it, or add `--no-plots` for CSV/JSON-only validation.

## Outputs

Each signal is written under:

```text
<output-dir>/<input-stem>/<signal-name>/
```

The directory contains:

- `<signal>_rpt_counts.csv`: theta plus the radial-bin count matrix
- `<signal>_rpt_features.csv`: crossing count, mean, variance, entropy, and
  dominant radial bin for each section
- `<signal>_rpt_metadata.json`: resolved configuration, radial coordinates,
  preprocessing statistics, time source, and warnings
- `<signal>_phase_portrait.png`
- `<signal>_rpt_heatmap.png`: crossing counts with the section-wise mean radial
  trajectory overlaid
- `<signal>_rpt_surface3d.png`
- `<signal>_rpt_surface3d.html`: self-contained interactive Plotly surface
- `<signal>_rpt_features.png`

Use `--no-plots` to omit the PNG and HTML artifacts.

## Interpretation

Concentrated, repeatable ridges in the heatmap indicate that crossings recur at
similar radial positions, as expected for stable limit-cycle-like behavior.
The overlaid mean trajectory is the count-weighted mean radial-bin coordinate
at each section angle with at least one crossing.
Broader or fragmented distributions indicate variable or irregular phase
trajectories. Section-wise radial variance and entropy quantify that spread;
`total_crossings` also reveals angles with unusually few or many intersections.

The transform is descriptive rather than a proof of periodicity. Filtering,
sampling quality, trajectory duration, and the chosen normalization all affect
the result and should remain consistent when signals are compared.
