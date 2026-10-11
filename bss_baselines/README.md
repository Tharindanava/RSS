# Direct observer-signal BSS baselines

This package provides PCA, JADE-ICA, scikit-learn FastICA, and SOBI controls for
`notebooks/rss_validation.ipynb`. Each fit receives only the three cumulative
observer channels. Simulation truth is used later for permutation/sign
matching, physical error metrics, and the explicitly labelled oracle-scaled
waveform metrics.

JADE uses whitening, fourth-order cumulant eigenmatrices, and orthogonal Jacobi
joint diagonalization. FastICA uses the parallel `logcosh` algorithm by default.
SOBI whitens the observer channels, forms symmetrized lag covariances at
`sobi_lags`, and jointly diagonalizes them with the same orthogonal Jacobi sweep
JADE uses. All demixing rows are normalized to unit axes and the corresponding
separated signals are divided by the same row norms before physical evaluation.

SOBI is a *second-order* method: it separates sources by their differing
autocorrelation structure, so unlike the fourth-order JADE and FastICA controls
it cannot separate temporally white sources, and it reports an identifiability
warning when the per-source lag profiles are nearly identical. Lags at or beyond
the record length are dropped with a recorded warning rather than rejected, and
the lags actually used are stored in the fit diagnostics.

This SOBI is an independent, NumPy-only implementation (no PyTorch or Matplotlib import chain).
