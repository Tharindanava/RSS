# Limb-Wise Transfer Entropy

This standalone package reads existing `limb_angle_mode.csv` artifacts and
computes fixed-context raw transfer entropy between four lower-limb angle
signals. It does not modify `rss`, `HMA_Dashboard`, or
`gait_analysis`.

Each CSV is analyzed independently for the left and right sides using mode 1
from:

1. pelvis to hip,
2. hip to knee,
3. knee to ankle, and
4. ankle to foot.

Cross-limb edges are not calculated.

## Run

From the repository root in PowerShell:

```powershell
python -m limb_transfer_entropy `
  --input-root outputs\rss `
  --output-root outputs\limb_transfer_entropy `
  --bins 8 `
  --binning quantile `
  --context-length 2 `
  --lag 1 `
  --stride 1
```

The input and output arguments shown above are also their defaults. The
analysis always creates matrix CSVs, long-form edge CSVs, and matrix PNGs.
Progress is printed as each discovered file is processed.

Use `python -m limb_transfer_entropy --help` for the complete CLI.

## Context definition

For a source signal `X`, receiver signal `Y`, present sample `t`, context
length `L`, lag `d`, and stride `s`, both histories are

```text
X context = (X[t-d], X[t-d-s], ..., X[t-d-(L-1)s])
Y context = (Y[t-d], Y[t-d-s], ..., Y[t-d-(L-1)s])
```

The largest history offset is therefore:

```text
max context lag = lag + (context length - 1) * stride
```

The discrete plug-in estimator reports raw transfer entropy in bits:

```text
TE(X -> Y) =
    sum p(y_t, y_past, x_past)
        log2[p(y_t | y_past, x_past) / p(y_t | y_past)]
```

There is no delay scan, surrogate correction, significance test, or
cross-file concatenation.

## Binning

`--bins` is the requested number of discrete states and must be at least 2.
The default `--binning quantile` uses approximately equal-occupancy bins,
matching the default in the gait transfer-entropy workflow. Duplicate
quantile boundaries are removed, so the effective number of bins can be
smaller than requested. Constant signals become one discrete state.

Use `--binning uniform` for equal-width bins between each signal's minimum and
maximum. The long-form CSV records both the requested bin count and the
effective occupied-bin count for the source and target.

## Output tree

For an input such as:

```text
outputs/rss/Subject_01/condition/run/limb_angle_mode.csv
```

the package writes:

```text
outputs/limb_transfer_entropy/Subject_01/condition/run/
    left_te_matrix.csv
    left_te_matrix.png
    left_te_edges.csv
    right_te_matrix.csv
    right_te_matrix.png
    right_te_edges.csv
```

Matrix rows are sources and columns are targets. The diagonal is blank and
masked because self-transfer is not evaluated.

The output root also contains:

- `processing_summary.csv`: one status row per input side.
- `processing_errors.csv`: missing-column, short-signal, and read failures.

One bad file or side does not stop the remaining batch.

## Input handling

All four required columns for a side are converted to numeric values. Rows
where any of those four values is non-finite are removed with one common mask,
so every matrix edge uses the same aligned samples. At least 25 embedded
samples must remain after context trimming.
