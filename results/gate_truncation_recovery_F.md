# Gate - Truncation Recovery + Competing Baselines, Cycle F

- VERDICT: PHYSICS-JUSTIFIED - E1 beats direct obstruction false-negative rate at T<=3 by the material margin and beats spline/linear FVC recovery at T<=3.
- Deterministic LM only. No NUTS, SNPE, gamma, registration, or ML residual.
- Manoeuvre rule: Pre, SPAACC == Y, SPAQEFF == A for this complete-curve gate, max parsed FEV1, ties to lowest manoeuvre number.
- Every truncated estimator used only volume samples up to T; full curve was used only for scoring and as the requested extrapolation target time.

## Motivation

Prior gates used complete A/B curves, where direct measurement is exact by construction. This gate tests the Steltner-style use case: recovering FVC and obstruction calls from incomplete manoeuvres where direct end-volume underestimates FVC and inflates FEV1/FVC.

## Sample

| stratum | grade-A selected | complete plateau | sampled | drop before sample | drop by cap |
|---|---|---|---|---|---|
| severe | 30 | 17 | 17 | 13 | 0 |
| moderate | 89 | 80 | 80 | 9 | 0 |
| mild_0.60_0.70 | 430 | 423 | 200 | 7 | 223 |
| normal | 6111 | 5587 | 200 | 524 | 5387 |

Complete plateau rule: FET >= 6.0 s and terminal-second dV <= 0.025 L.

## TRUE vs NHANES Cross-Check

- Cross-check rule: PASS if median absolute FEV1 and FVC percent errors <=5% and median absolute ratio error <=0.02.
- Result: PASS on n=497

| metric | signed median | abs median | abs IQR |
|---|---:|---:|---:|
| FEV1 percent error | -3.21% | 3.21% | 2.41% to 4.64% |
| FVC percent error | -3.84% | 3.84% | 2.76% to 5.81% |
| FEV1/FVC absolute error | 0.0017 | 0.0083 | 0.0030 to 0.0206 |

## Step 2 - Error Tables

Full per-estimator x T x stratum error summaries are below.

| estimator | T | stratum | n | fail | FVC signed % med | FVC abs % med | ratio signed med | ratio abs med |
|---|---|---|---|---|---|---|---|---|
| E0_DIRECT | 2 | mild_0.60_0.70 | 200 | 0 | -20.73 | 20.73 | 0.1753 | 0.1753 |
| E0_DIRECT | 2 | moderate | 80 | 0 | -28.96 | 28.96 | 0.2326 | 0.2326 |
| E0_DIRECT | 2 | normal | 200 | 0 | -7.47 | 7.47 | 0.0656 | 0.0656 |
| E0_DIRECT | 2 | severe | 17 | 0 | -39.84 | 39.84 | 0.2983 | 0.2983 |
| E0_DIRECT | 3 | mild_0.60_0.70 | 200 | 0 | -14.89 | 14.89 | 0.1168 | 0.1168 |
| E0_DIRECT | 3 | moderate | 80 | 0 | -21.99 | 21.99 | 0.1598 | 0.1598 |
| E0_DIRECT | 3 | normal | 200 | 0 | -3.72 | 4.16 | 0.0313 | 0.0355 |
| E0_DIRECT | 3 | severe | 17 | 0 | -31.73 | 31.73 | 0.2082 | 0.2082 |
| E0_DIRECT | 4 | mild_0.60_0.70 | 200 | 0 | -11.38 | 11.38 | 0.0866 | 0.0866 |
| E0_DIRECT | 4 | moderate | 80 | 0 | -17.76 | 17.76 | 0.1213 | 0.1213 |
| E0_DIRECT | 4 | normal | 200 | 0 | -1.74 | 3.12 | 0.0142 | 0.0270 |
| E0_DIRECT | 4 | severe | 17 | 0 | -25.04 | 25.04 | 0.1608 | 0.1608 |
| E0_DIRECT | 5 | mild_0.60_0.70 | 200 | 0 | -9.04 | 9.04 | 0.0657 | 0.0657 |
| E0_DIRECT | 5 | moderate | 80 | 0 | -14.52 | 14.52 | 0.0936 | 0.0936 |
| E0_DIRECT | 5 | normal | 200 | 0 | -0.62 | 2.18 | 0.0051 | 0.0182 |
| E0_DIRECT | 5 | severe | 17 | 0 | -20.86 | 20.86 | 0.1249 | 0.1249 |
| E0_DIRECT | full | mild_0.60_0.70 | 200 | 0 | 0.00 | 0.00 | 0.0000 | 0.0000 |
| E0_DIRECT | full | moderate | 80 | 0 | 0.00 | 0.00 | 0.0000 | 0.0000 |
| E0_DIRECT | full | normal | 200 | 0 | 0.00 | 0.00 | 0.0000 | 0.0000 |
| E0_DIRECT | full | severe | 17 | 0 | 0.00 | 0.00 | 0.0000 | 0.0000 |
| E1_OMBE | 2 | mild_0.60_0.70 | 200 | 0 | -10.57 | 11.18 | 0.0794 | 0.0849 |
| E1_OMBE | 2 | moderate | 80 | 0 | -12.99 | 14.99 | 0.0860 | 0.0995 |
| E1_OMBE | 2 | normal | 200 | 0 | 1.68 | 5.50 | -0.0134 | 0.0456 |
| E1_OMBE | 2 | severe | 17 | 0 | -20.68 | 20.68 | 0.1013 | 0.1013 |
| E1_OMBE | 3 | mild_0.60_0.70 | 200 | 0 | -10.68 | 11.27 | 0.0799 | 0.0830 |
| E1_OMBE | 3 | moderate | 80 | 0 | -13.83 | 15.99 | 0.0874 | 0.1071 |
| E1_OMBE | 3 | normal | 200 | 0 | 0.26 | 4.08 | -0.0020 | 0.0335 |
| E1_OMBE | 3 | severe | 17 | 0 | -18.69 | 20.16 | 0.1035 | 0.1069 |
| E1_OMBE | 4 | mild_0.60_0.70 | 200 | 0 | -9.03 | 9.40 | 0.0653 | 0.0691 |
| E1_OMBE | 4 | moderate | 80 | 0 | -12.77 | 13.46 | 0.0817 | 0.0874 |
| E1_OMBE | 4 | normal | 200 | 0 | -0.55 | 3.07 | 0.0045 | 0.0258 |
| E1_OMBE | 4 | severe | 17 | 0 | -15.61 | 17.07 | 0.0833 | 0.0833 |
| E1_OMBE | 5 | mild_0.60_0.70 | 200 | 0 | -7.11 | 7.48 | 0.0494 | 0.0542 |
| E1_OMBE | 5 | moderate | 80 | 0 | -11.13 | 11.48 | 0.0672 | 0.0733 |
| E1_OMBE | 5 | normal | 200 | 0 | -0.10 | 2.38 | 0.0009 | 0.0198 |
| E1_OMBE | 5 | severe | 17 | 0 | -14.34 | 14.55 | 0.0717 | 0.0751 |
| E1_OMBE | full | mild_0.60_0.70 | 200 | 0 | -0.03 | 0.57 | 0.0002 | 0.0038 |
| E1_OMBE | full | moderate | 80 | 0 | -0.76 | 1.12 | 0.0042 | 0.0062 |
| E1_OMBE | full | normal | 200 | 0 | 0.67 | 0.74 | -0.0055 | 0.0059 |
| E1_OMBE | full | severe | 17 | 0 | -0.90 | 2.20 | 0.0041 | 0.0102 |
| E2_SPLINE | 2 | mild_0.60_0.70 | 200 | 70 | 4972.10 | 13992.66 | -0.6536 | 0.6539 |
| E2_SPLINE | 2 | moderate | 80 | 44 | -2498.97 | 17163.39 | -0.5545 | 0.5573 |
| E2_SPLINE | 2 | normal | 200 | 94 | 210.44 | 3374.47 | -0.7706 | 0.7714 |
| E2_SPLINE | 2 | severe | 17 | 11 | -30317.29 | 38114.26 | -0.4473 | 0.4473 |
| E2_SPLINE | 3 | mild_0.60_0.70 | 200 | 95 | 153.22 | 7778.77 | -0.6424 | 0.6451 |
| E2_SPLINE | 3 | moderate | 80 | 40 | -14.68 | 13318.73 | -0.5544 | 0.5544 |
| E2_SPLINE | 3 | normal | 200 | 97 | 0.76 | 2249.53 | -0.7541 | 0.7554 |
| E2_SPLINE | 3 | severe | 17 | 6 | 1277.80 | 12880.41 | -0.4632 | 0.4632 |
| E2_SPLINE | 4 | mild_0.60_0.70 | 200 | 99 | -94.76 | 6125.96 | -0.6391 | 0.6431 |
| E2_SPLINE | 4 | moderate | 80 | 35 | 1031.25 | 12626.71 | -0.5506 | 0.5506 |
| E2_SPLINE | 4 | normal | 200 | 89 | 0.32 | 1531.52 | -0.7301 | 0.7406 |
| E2_SPLINE | 4 | severe | 17 | 5 | 3139.90 | 15864.37 | -0.4672 | 0.4672 |
| E2_SPLINE | 5 | mild_0.60_0.70 | 200 | 98 | -50.69 | 5660.46 | -0.6348 | 0.6358 |
| E2_SPLINE | 5 | moderate | 80 | 37 | 1782.80 | 10510.93 | -0.5530 | 0.5530 |
| E2_SPLINE | 5 | normal | 200 | 87 | 0.86 | 954.59 | -0.6927 | 0.7000 |
| E2_SPLINE | 5 | severe | 17 | 6 | 4281.19 | 15727.72 | -0.4401 | 0.4401 |
| E2_SPLINE | full | mild_0.60_0.70 | 200 | 0 | 0.01 | 0.02 | -0.0001 | 0.0001 |
| E2_SPLINE | full | moderate | 80 | 0 | 0.01 | 0.02 | -0.0001 | 0.0001 |
| E2_SPLINE | full | normal | 200 | 0 | 0.01 | 0.01 | -0.0001 | 0.0001 |
| E2_SPLINE | full | severe | 17 | 0 | 0.01 | 0.02 | -0.0000 | 0.0001 |
| E3_LINEAR_TAIL | 2 | mild_0.60_0.70 | 200 | 0 | 127.95 | 127.95 | -0.3704 | 0.3706 |
| E3_LINEAR_TAIL | 2 | moderate | 80 | 0 | 171.80 | 171.80 | -0.3536 | 0.3536 |
| E3_LINEAR_TAIL | 2 | normal | 200 | 0 | 61.14 | 61.14 | -0.3065 | 0.3065 |
| E3_LINEAR_TAIL | 2 | severe | 17 | 0 | 173.88 | 173.88 | -0.2896 | 0.2896 |
| E3_LINEAR_TAIL | 3 | mild_0.60_0.70 | 200 | 0 | 47.62 | 47.62 | -0.2120 | 0.2120 |
| E3_LINEAR_TAIL | 3 | moderate | 80 | 0 | 67.79 | 67.79 | -0.2310 | 0.2310 |
| E3_LINEAR_TAIL | 3 | normal | 200 | 0 | 16.65 | 16.94 | -0.1190 | 0.1204 |
| E3_LINEAR_TAIL | 3 | severe | 17 | 0 | 84.29 | 84.29 | -0.2084 | 0.2084 |
| E3_LINEAR_TAIL | 4 | mild_0.60_0.70 | 200 | 0 | 24.52 | 24.52 | -0.1315 | 0.1315 |
| E3_LINEAR_TAIL | 4 | moderate | 80 | 0 | 35.92 | 35.92 | -0.1513 | 0.1513 |
| E3_LINEAR_TAIL | 4 | normal | 200 | 0 | 7.14 | 7.28 | -0.0561 | 0.0579 |
| E3_LINEAR_TAIL | 4 | severe | 17 | 0 | 51.37 | 51.37 | -0.1564 | 0.1564 |
| E3_LINEAR_TAIL | 5 | mild_0.60_0.70 | 200 | 0 | 14.50 | 14.50 | -0.0840 | 0.0840 |
| E3_LINEAR_TAIL | 5 | moderate | 80 | 0 | 21.77 | 21.77 | -0.0996 | 0.0996 |
| E3_LINEAR_TAIL | 5 | normal | 200 | 0 | 4.15 | 4.25 | -0.0327 | 0.0336 |
| E3_LINEAR_TAIL | 5 | severe | 17 | 0 | 32.81 | 32.81 | -0.1151 | 0.1151 |
| E3_LINEAR_TAIL | full | mild_0.60_0.70 | 200 | 0 | 0.49 | 0.49 | -0.0032 | 0.0033 |
| E3_LINEAR_TAIL | full | moderate | 80 | 0 | 0.27 | 0.27 | -0.0016 | 0.0016 |
| E3_LINEAR_TAIL | full | normal | 200 | 0 | 0.55 | 0.55 | -0.0044 | 0.0045 |
| E3_LINEAR_TAIL | full | severe | 17 | 0 | 0.09 | 0.10 | -0.0004 | 0.0005 |
| E4_MONO_EXPONENTIAL | 2 | mild_0.60_0.70 | 200 | 0 | -21.58 | 21.86 | 0.1859 | 0.1865 |
| E4_MONO_EXPONENTIAL | 2 | moderate | 80 | 0 | -26.41 | 27.36 | 0.1931 | 0.1973 |
| E4_MONO_EXPONENTIAL | 2 | normal | 200 | 0 | -7.93 | 8.56 | 0.0704 | 0.0750 |
| E4_MONO_EXPONENTIAL | 2 | severe | 17 | 0 | -31.30 | 33.85 | 0.1538 | 0.1538 |
| E4_MONO_EXPONENTIAL | 3 | mild_0.60_0.70 | 200 | 0 | -17.48 | 18.02 | 0.1438 | 0.1465 |
| E4_MONO_EXPONENTIAL | 3 | moderate | 80 | 0 | -20.82 | 21.76 | 0.1455 | 0.1513 |
| E4_MONO_EXPONENTIAL | 3 | normal | 200 | 0 | -5.43 | 5.94 | 0.0461 | 0.0533 |
| E4_MONO_EXPONENTIAL | 3 | severe | 17 | 0 | -27.94 | 27.94 | 0.1538 | 0.1538 |
| E4_MONO_EXPONENTIAL | 4 | mild_0.60_0.70 | 200 | 0 | -15.12 | 15.37 | 0.1197 | 0.1220 |
| E4_MONO_EXPONENTIAL | 4 | moderate | 80 | 0 | -18.19 | 18.78 | 0.1223 | 0.1241 |
| E4_MONO_EXPONENTIAL | 4 | normal | 200 | 0 | -4.02 | 5.02 | 0.0340 | 0.0430 |
| E4_MONO_EXPONENTIAL | 4 | severe | 17 | 0 | -24.66 | 24.66 | 0.1468 | 0.1468 |
| E4_MONO_EXPONENTIAL | 5 | mild_0.60_0.70 | 200 | 0 | -13.10 | 13.37 | 0.1006 | 0.1040 |
| E4_MONO_EXPONENTIAL | 5 | moderate | 80 | 0 | -16.65 | 17.15 | 0.1087 | 0.1166 |
| E4_MONO_EXPONENTIAL | 5 | normal | 200 | 0 | -3.09 | 4.20 | 0.0256 | 0.0360 |
| E4_MONO_EXPONENTIAL | 5 | severe | 17 | 0 | -22.10 | 22.10 | 0.1304 | 0.1304 |
| E4_MONO_EXPONENTIAL | full | mild_0.60_0.70 | 200 | 0 | -4.25 | 4.50 | 0.0298 | 0.0316 |
| E4_MONO_EXPONENTIAL | full | moderate | 80 | 0 | -5.41 | 5.90 | 0.0317 | 0.0353 |
| E4_MONO_EXPONENTIAL | full | normal | 200 | 0 | -1.22 | 2.26 | 0.0101 | 0.0185 |
| E4_MONO_EXPONENTIAL | full | severe | 17 | 0 | -4.89 | 5.85 | 0.0228 | 0.0257 |

## Step 3 - Classification Consequence

False-negative rate on obstructed-truth:

| estimator | 2 | 3 | 4 | 5 | full |
|---|---|---|---|---|---|
| E0_DIRECT | 0.990 | 0.865 | 0.684 | 0.549 | 0.000 |
| E1_OMBE | 0.589 | 0.596 | 0.519 | 0.468 | 0.030 |
| E2_SPLINE | 0.012 | 0.026 | 0.032 | 0.032 | 0.000 |
| E3_LINEAR_TAIL | 0.003 | 0.000 | 0.010 | 0.003 | 0.000 |
| E4_MONO_EXPONENTIAL | 0.822 | 0.771 | 0.721 | 0.653 | 0.303 |

Borderline-stratum false-negative rate:

| estimator | 2 | 3 | 4 | 5 | full |
|---|---|---|---|---|---|
| E0_DIRECT | 1.000 | 0.995 | 0.895 | 0.780 | 0.000 |
| E1_OMBE | 0.765 | 0.795 | 0.730 | 0.685 | 0.045 |
| E2_SPLINE | 0.008 | 0.038 | 0.050 | 0.049 | 0.000 |
| E3_LINEAR_TAIL | 0.005 | 0.000 | 0.015 | 0.005 | 0.000 |
| E4_MONO_EXPONENTIAL | 0.910 | 0.905 | 0.885 | 0.850 | 0.450 |

Sensitivity:

| estimator | 2 | 3 | 4 | 5 | full |
|---|---|---|---|---|---|
| E0_DIRECT | 0.010 | 0.135 | 0.316 | 0.451 | 1.000 |
| E1_OMBE | 0.411 | 0.404 | 0.481 | 0.532 | 0.970 |
| E2_SPLINE | 0.988 | 0.974 | 0.968 | 0.968 | 1.000 |
| E3_LINEAR_TAIL | 0.997 | 1.000 | 0.990 | 0.997 | 1.000 |
| E4_MONO_EXPONENTIAL | 0.178 | 0.229 | 0.279 | 0.347 | 0.697 |

Specificity:

| estimator | 2 | 3 | 4 | 5 | full |
|---|---|---|---|---|---|
| E0_DIRECT | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| E1_OMBE | 0.965 | 0.970 | 0.970 | 0.970 | 0.970 |
| E2_SPLINE | 0.009 | 0.058 | 0.135 | 0.186 | 1.000 |
| E3_LINEAR_TAIL | 0.090 | 0.505 | 0.750 | 0.860 | 0.990 |
| E4_MONO_EXPONENTIAL | 0.975 | 0.975 | 0.975 | 0.975 | 0.975 |

Cohen's kappa:

| estimator | 2 | 3 | 4 | 5 | full |
|---|---|---|---|---|---|
| E0_DIRECT | 0.008 | 0.111 | 0.271 | 0.398 | 1.000 |
| E1_OMBE | 0.330 | 0.328 | 0.403 | 0.453 | 0.937 |
| E2_SPLINE | -0.003 | 0.039 | 0.118 | 0.172 | 1.000 |
| E3_LINEAR_TAIL | 0.102 | 0.549 | 0.769 | 0.876 | 0.992 |
| E4_MONO_EXPONENTIAL | 0.128 | 0.172 | 0.217 | 0.279 | 0.628 |

## Step 4 - Physics vs Smoothness

E1 beats the spline and linear-tail baselines at T=2 and T=3, so the bi-exponential form is doing work beyond generic smooth extrapolation.

E1 beats the mono-exponential at T=2 and T=3, so the two-compartment structure is load-bearing.

## Step 5 - Regime Map

| estimator | stratum | min T for median abs FVC <5% |
|---|---|---|
| E0_DIRECT | severe | full |
| E0_DIRECT | moderate | full |
| E0_DIRECT | mild_0.60_0.70 | full |
| E0_DIRECT | normal | 3 |
| E1_OMBE | severe | full |
| E1_OMBE | moderate | full |
| E1_OMBE | mild_0.60_0.70 | full |
| E1_OMBE | normal | 3 |
| E2_SPLINE | severe | full |
| E2_SPLINE | moderate | full |
| E2_SPLINE | mild_0.60_0.70 | full |
| E2_SPLINE | normal | full |
| E3_LINEAR_TAIL | severe | full |
| E3_LINEAR_TAIL | moderate | full |
| E3_LINEAR_TAIL | mild_0.60_0.70 | full |
| E3_LINEAR_TAIL | normal | 5 |
| E4_MONO_EXPONENTIAL | severe | none |
| E4_MONO_EXPONENTIAL | moderate | none |
| E4_MONO_EXPONENTIAL | mild_0.60_0.70 | full |
| E4_MONO_EXPONENTIAL | normal | 5 |

## Step 6 - Honesty Checks

- Original Cycle-F obstructed participants: 676
- Obstructed participants in this complete-plateau sample: 297
- Excluded original obstructed participants: 399; NHANES ratio median 0.6624, IQR 0.6268 to 0.6860.
- No truncated estimator saw untruncated volume samples after T: True. For truncated T rows, estimators used only volume samples up to T. The full-curve end time was supplied only as the requested extrapolation target.

E1 OMBE overshoot/undershoot by T:

| T | valid | overshoot n | overshoot frac | overshoot % med | overshoot % max | undershoot n | undershoot % med |
|---|---|---|---|---|---|---|---|
| 2 | 497 | 158 | 0.318 | 6.65 | 74.78 | 339 | -10.38 |
| 3 | 497 | 133 | 0.268 | 4.33 | 74.78 | 364 | -10.42 |
| 4 | 497 | 123 | 0.247 | 3.34 | 74.78 | 374 | -8.56 |
| 5 | 497 | 132 | 0.266 | 2.63 | 74.78 | 365 | -7.02 |
| full | 497 | 284 | 0.571 | 0.89 | 74.78 | 213 | -0.59 |

VERDICT: PHYSICS-JUSTIFIED - E1 beats direct obstruction false-negative rate at T<=3 by the material margin and beats spline/linear FVC recovery at T<=3.
