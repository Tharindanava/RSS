# Adaptive regularization and axis-flip paired report

This is a bounded implementation-validation run, not an efficacy claim. The
baseline and enabled member of every pair used the same input, seed, SGD
optimizer, 60 accepted epochs, and learning-rate schedule (`0.1` to `0.001`).
The enabled runs used a flip interval of 20, four trial epochs, regularizer
start weight `0.1`, plateau patience 8, tolerance `1e-4`, and eight ramp
epochs. Trial branches count once against the 60-epoch budget.

## Primary result: one-to-one matched absolute correlation

The QArm evaluation covers all 35 merged tests and Node 4. Inputs use the
same practical preprocessing settings as the notebook workflow: 100 Hz,
SSA window 100, 25 reconstruction components, and drift thresholds 3/3. The
enabled run uses `piecewise_sparse_innovation`.

| QArm cohort statistic | Baseline | Enabled |
| --- | ---: | ---: |
| Mean matched absolute correlation | 0.5908 | 0.6152 |
| Median matched absolute correlation | 0.5846 | 0.5927 |
| Tests improved | - | 22 / 35 |

The simulation smoke uses one deterministic 480-frame medium-amplitude case
per standard family and the corresponding family-specific regularizer.

| Simulation family | Baseline correlation | Enabled correlation | Delta |
| --- | ---: | ---: | ---: |
| Harmonic | 0.9336 | 0.9019 | -0.0317 |
| Filtered noise | 0.9447 | 0.8228 | -0.1218 |
| Chirp/modulated | 0.9494 | 0.8327 | -0.1166 |
| Piecewise/transient | 0.7890 | 0.9271 | +0.1381 |
| Mean | 0.9042 | 0.8711 | -0.0330 |

The mixed simulation outcome is important: these controls expose useful basin
search and regularization behavior, but the short-run schedule is not a
generally superior default. Both features therefore remain disabled by
default and should be tuned by signal family.

## Secondary results

RMSE is calculated after the one-to-one permutation, sign alignment, and a
constant offset correction. Amplitude is not rescaled.

| Dataset | Baseline mean RMSE (rad) | Enabled mean RMSE (rad) | Delta |
| --- | ---: | ---: | ---: |
| QArm, 35 tests | 0.2613 | 0.2532 | -0.0080 |
| Four simulation families | 0.1069 | 0.1462 | +0.0393 |

QArm aligned RMSE improved on 18 of 35 tests. Simulation axis recovery
was also mixed:

| Simulation family | Baseline axis error (deg) | Enabled axis error (deg) | Delta |
| --- | ---: | ---: | ---: |
| Harmonic | 5.998 | 14.607 | +8.609 |
| Filtered noise | 13.630 | 29.304 | +15.675 |
| Chirp/modulated | 11.530 | 26.264 | +14.734 |
| Piecewise/transient | 31.506 | 16.970 | -14.536 |
| Mean | 15.666 | 21.786 | +6.121 |

## QArm per-test pairs

| Test | Correlation baseline | Correlation enabled | RMSE baseline | RMSE enabled |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6863 | 0.6766 | 0.2095 | 0.2237 |
| 2 | 0.4690 | 0.5719 | 0.4362 | 0.4070 |
| 3 | 0.5908 | 0.5854 | 0.2656 | 0.2660 |
| 4 | 0.6552 | 0.6553 | 0.2076 | 0.2075 |
| 5 | 0.6680 | 0.6962 | 0.2897 | 0.2803 |
| 6 | 0.6482 | 0.6425 | 0.1336 | 0.1340 |
| 7 | 0.5619 | 0.5893 | 0.2759 | 0.2895 |
| 8 | 0.5310 | 0.5277 | 0.2009 | 0.2015 |
| 9 | 0.7204 | 0.7863 | 0.2691 | 0.2464 |
| 10 | 0.6345 | 0.5839 | 0.1977 | 0.2506 |
| 11 | 0.7084 | 0.7118 | 0.1176 | 0.1143 |
| 12 | 0.5158 | 0.4989 | 0.2951 | 0.2993 |
| 13 | 0.7459 | 0.7160 | 0.2219 | 0.2253 |
| 14 | 0.5139 | 0.5795 | 0.4338 | 0.3873 |
| 15 | 0.4898 | 0.5265 | 0.2634 | 0.2514 |
| 16 | 0.4717 | 0.4961 | 0.4267 | 0.4090 |
| 17 | 0.5766 | 0.6466 | 0.3270 | 0.2965 |
| 18 | 0.5500 | 0.5473 | 0.2008 | 0.2007 |
| 19 | 0.5150 | 0.5582 | 0.3010 | 0.2907 |
| 20 | 0.5956 | 0.4846 | 0.2972 | 0.3371 |
| 21 | 0.3686 | 0.5345 | 0.2525 | 0.2199 |
| 22 | 0.7105 | 0.7179 | 0.2056 | 0.2006 |
| 23 | 0.5846 | 0.5792 | 0.2856 | 0.2936 |
| 24 | 0.6270 | 0.6396 | 0.2623 | 0.2879 |
| 25 | 0.7243 | 0.7288 | 0.1373 | 0.1375 |
| 26 | 0.7287 | 0.7605 | 0.1975 | 0.1950 |
| 27 | 0.5309 | 0.6948 | 0.4112 | 0.2945 |
| 28 | 0.5645 | 0.5686 | 0.1889 | 0.2062 |
| 29 | 0.7656 | 0.7633 | 0.2033 | 0.2040 |
| 30 | 0.3724 | 0.4211 | 0.3254 | 0.3070 |
| 31 | 0.7603 | 0.7387 | 0.1882 | 0.1945 |
| 32 | 0.5668 | 0.6145 | 0.2810 | 0.1699 |
| 33 | 0.4266 | 0.5927 | 0.2889 | 0.2897 |
| 34 | 0.6371 | 0.6393 | 0.2198 | 0.2182 |
| 35 | 0.4632 | 0.4593 | 0.3264 | 0.3265 |

True sign remains unidentifiable from the unsupervised sign-symmetric losses.
The QArm geometry is also partly rank-deficient, so neither this report
nor the optimizer assumes that all three joint sources are uniquely
recoverable.
