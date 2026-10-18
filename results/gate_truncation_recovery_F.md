# Gate - Truncation Recovery + Competing Baselines, Cycle F

- VERDICT: PHYSICS-JUSTIFIED - E1 beats direct obstruction false-negative rate at T<=3 by the material margin and beats spline/linear FVC recovery at T<=3.
- Deterministic LM only. No NUTS, SNPE, gamma, registration, or ML residual.
- Manoeuvre rule: Pre, SPAACC == Y, SPAQEFF == A for this complete-curve gate, max parsed FEV1, ties to lowest manoeuvre number.
- Every truncated estimator used only volume samples up to T and the protocol-fixed 12 s prediction horizon; the complete curve was used only for scoring.

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
| FEV1 percent error | -3.21% | 3.21% | 2.44% to 4.64% |
| FVC percent error | -3.84% | 3.84% | 2.76% to 5.81% |
| FEV1/FVC absolute error | 0.0017 | 0.0083 | 0.0030 to 0.0206 |

## Step 2 - Error Tables

Full per-estimator x T x stratum error summaries are below.

| estimator | T | stratum | n | fail | FVC signed % med | FVC abs % med | ratio signed med | ratio abs med |
|---|---|---|---|---|---|---|---|---|
| E0_DIRECT | 2 | mild_0.60_0.70 | 200 | 0 | -20.75 | 20.75 | 0.1753 | 0.1753 |
| E0_DIRECT | 2 | moderate | 80 | 0 | -28.96 | 28.96 | 0.2326 | 0.2326 |
| E0_DIRECT | 2 | normal | 200 | 0 | -7.46 | 7.46 | 0.0656 | 0.0656 |
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
| E1_OMBE | 2 | mild_0.60_0.70 | 200 | 0 | -10.57 | 11.26 | 0.0794 | 0.0848 |
| E1_OMBE | 2 | moderate | 80 | 0 | -13.01 | 15.00 | 0.0859 | 0.0995 |
| E1_OMBE | 2 | normal | 200 | 0 | 1.66 | 5.77 | -0.0124 | 0.0469 |
| E1_OMBE | 2 | severe | 17 | 0 | -20.69 | 21.57 | 0.1015 | 0.1015 |
| E1_OMBE | 3 | mild_0.60_0.70 | 200 | 0 | -10.69 | 11.29 | 0.0801 | 0.0831 |
| E1_OMBE | 3 | moderate | 80 | 0 | -13.84 | 16.01 | 0.0875 | 0.1072 |
| E1_OMBE | 3 | normal | 200 | 0 | 0.30 | 4.17 | -0.0023 | 0.0348 |
| E1_OMBE | 3 | severe | 17 | 0 | -18.75 | 20.17 | 0.1039 | 0.1121 |
| E1_OMBE | 4 | mild_0.60_0.70 | 200 | 0 | -9.06 | 9.41 | 0.0656 | 0.0696 |
| E1_OMBE | 4 | moderate | 80 | 0 | -12.74 | 13.58 | 0.0819 | 0.0880 |
| E1_OMBE | 4 | normal | 200 | 0 | -0.41 | 3.07 | 0.0035 | 0.0258 |
| E1_OMBE | 4 | severe | 17 | 0 | -15.72 | 17.07 | 0.0865 | 0.0865 |
| E1_OMBE | 5 | mild_0.60_0.70 | 200 | 0 | -7.17 | 7.58 | 0.0497 | 0.0545 |
| E1_OMBE | 5 | moderate | 80 | 0 | -11.17 | 11.52 | 0.0677 | 0.0736 |
| E1_OMBE | 5 | normal | 200 | 0 | -0.05 | 2.38 | 0.0004 | 0.0204 |
| E1_OMBE | 5 | severe | 17 | 0 | -14.50 | 15.12 | 0.0717 | 0.0751 |
| E1_OMBE | full | mild_0.60_0.70 | 200 | 0 | -0.03 | 0.57 | 0.0002 | 0.0038 |
| E1_OMBE | full | moderate | 80 | 0 | -0.76 | 1.12 | 0.0042 | 0.0062 |
| E1_OMBE | full | normal | 200 | 0 | 0.67 | 0.74 | -0.0055 | 0.0059 |
| E1_OMBE | full | severe | 17 | 0 | -0.90 | 2.20 | 0.0041 | 0.0102 |
| E2_SPLINE | 2 | mild_0.60_0.70 | 200 | 71 | 3039.63 | 9559.43 | -0.6511 | 0.6511 |
| E2_SPLINE | 2 | moderate | 80 | 46 | -1624.07 | 8796.14 | -0.5542 | 0.5597 |
| E2_SPLINE | 2 | normal | 200 | 95 | 866.18 | 10268.57 | -0.8084 | 0.8084 |
| E2_SPLINE | 2 | severe | 17 | 11 | -9270.50 | 14639.67 | -0.4087 | 0.4087 |
| E2_SPLINE | 3 | mild_0.60_0.70 | 200 | 97 | 35.51 | 4668.22 | -0.6370 | 0.6370 |
| E2_SPLINE | 3 | moderate | 80 | 40 | -115.45 | 4937.54 | -0.5410 | 0.5410 |
| E2_SPLINE | 3 | normal | 200 | 98 | -11.72 | 7505.62 | -0.7993 | 0.8038 |
| E2_SPLINE | 3 | severe | 17 | 6 | 508.31 | 2700.94 | -0.4485 | 0.4485 |
| E2_SPLINE | 4 | mild_0.60_0.70 | 200 | 103 | -161.86 | 3778.73 | -0.6338 | 0.6344 |
| E2_SPLINE | 4 | moderate | 80 | 34 | 306.57 | 4976.73 | -0.5431 | 0.5447 |
| E2_SPLINE | 4 | normal | 200 | 98 | 2.77 | 5326.20 | -0.7883 | 0.7883 |
| E2_SPLINE | 4 | severe | 17 | 5 | 1055.14 | 2491.48 | -0.4296 | 0.4296 |
| E2_SPLINE | 5 | mild_0.60_0.70 | 200 | 98 | -44.18 | 3012.70 | -0.6210 | 0.6210 |
| E2_SPLINE | 5 | moderate | 80 | 37 | 431.40 | 3910.44 | -0.5347 | 0.5347 |
| E2_SPLINE | 5 | normal | 200 | 96 | -0.46 | 4215.76 | -0.7732 | 0.7747 |
| E2_SPLINE | 5 | severe | 17 | 6 | 1644.72 | 3329.97 | -0.4302 | 0.4302 |
| E2_SPLINE | full | mild_0.60_0.70 | 200 | 0 | 0.01 | 0.02 | -0.0001 | 0.0001 |
| E2_SPLINE | full | moderate | 80 | 0 | 0.01 | 0.02 | -0.0001 | 0.0001 |
| E2_SPLINE | full | normal | 200 | 0 | 0.01 | 0.01 | -0.0001 | 0.0001 |
| E2_SPLINE | full | severe | 17 | 0 | 0.01 | 0.02 | -0.0000 | 0.0001 |
| E3_LINEAR_TAIL | 2 | mild_0.60_0.70 | 200 | 0 | 99.38 | 99.38 | -0.3338 | 0.3347 |
| E3_LINEAR_TAIL | 2 | moderate | 80 | 0 | 114.25 | 114.25 | -0.2965 | 0.2965 |
| E3_LINEAR_TAIL | 2 | normal | 200 | 0 | 84.20 | 84.20 | -0.3771 | 0.3771 |
| E3_LINEAR_TAIL | 2 | severe | 17 | 0 | 101.80 | 101.80 | -0.2271 | 0.2271 |
| E3_LINEAR_TAIL | 3 | mild_0.60_0.70 | 200 | 0 | 36.00 | 36.00 | -0.1763 | 0.1763 |
| E3_LINEAR_TAIL | 3 | moderate | 80 | 0 | 42.08 | 42.08 | -0.1670 | 0.1670 |
| E3_LINEAR_TAIL | 3 | normal | 200 | 0 | 25.75 | 25.84 | -0.1652 | 0.1662 |
| E3_LINEAR_TAIL | 3 | severe | 17 | 0 | 43.22 | 43.22 | -0.1328 | 0.1328 |
| E3_LINEAR_TAIL | 4 | mild_0.60_0.70 | 200 | 0 | 16.83 | 16.83 | -0.0970 | 0.0970 |
| E3_LINEAR_TAIL | 4 | moderate | 80 | 0 | 18.93 | 18.93 | -0.0879 | 0.0879 |
| E3_LINEAR_TAIL | 4 | normal | 200 | 0 | 11.55 | 11.76 | -0.0847 | 0.0853 |
| E3_LINEAR_TAIL | 4 | severe | 17 | 0 | 21.90 | 21.90 | -0.0806 | 0.0806 |
| E3_LINEAR_TAIL | 5 | mild_0.60_0.70 | 200 | 0 | 7.92 | 7.92 | -0.0500 | 0.0500 |
| E3_LINEAR_TAIL | 5 | moderate | 80 | 0 | 8.41 | 8.41 | -0.0438 | 0.0438 |
| E3_LINEAR_TAIL | 5 | normal | 200 | 0 | 6.69 | 6.84 | -0.0496 | 0.0524 |
| E3_LINEAR_TAIL | 5 | severe | 17 | 0 | 10.66 | 10.66 | -0.0473 | 0.0473 |
| E3_LINEAR_TAIL | full | mild_0.60_0.70 | 200 | 0 | 0.49 | 0.49 | -0.0032 | 0.0033 |
| E3_LINEAR_TAIL | full | moderate | 80 | 0 | 0.27 | 0.27 | -0.0016 | 0.0016 |
| E3_LINEAR_TAIL | full | normal | 200 | 0 | 0.55 | 0.55 | -0.0044 | 0.0045 |
| E3_LINEAR_TAIL | full | severe | 17 | 0 | 0.09 | 0.10 | -0.0004 | 0.0005 |
| E4_MONO_EXPONENTIAL | 2 | mild_0.60_0.70 | 200 | 0 | -21.63 | 21.89 | 0.1861 | 0.1867 |
| E4_MONO_EXPONENTIAL | 2 | moderate | 80 | 0 | -26.44 | 27.42 | 0.1937 | 0.1977 |
| E4_MONO_EXPONENTIAL | 2 | normal | 200 | 0 | -7.94 | 8.58 | 0.0705 | 0.0752 |
| E4_MONO_EXPONENTIAL | 2 | severe | 17 | 0 | -31.30 | 33.87 | 0.1538 | 0.1538 |
| E4_MONO_EXPONENTIAL | 3 | mild_0.60_0.70 | 200 | 0 | -17.50 | 18.02 | 0.1440 | 0.1466 |
| E4_MONO_EXPONENTIAL | 3 | moderate | 80 | 0 | -20.83 | 21.78 | 0.1457 | 0.1514 |
| E4_MONO_EXPONENTIAL | 3 | normal | 200 | 0 | -5.45 | 5.94 | 0.0462 | 0.0533 |
| E4_MONO_EXPONENTIAL | 3 | severe | 17 | 0 | -27.96 | 27.96 | 0.1538 | 0.1538 |
| E4_MONO_EXPONENTIAL | 4 | mild_0.60_0.70 | 200 | 0 | -15.12 | 15.37 | 0.1199 | 0.1221 |
| E4_MONO_EXPONENTIAL | 4 | moderate | 80 | 0 | -18.19 | 18.79 | 0.1223 | 0.1242 |
| E4_MONO_EXPONENTIAL | 4 | normal | 200 | 0 | -4.03 | 5.02 | 0.0340 | 0.0431 |
| E4_MONO_EXPONENTIAL | 4 | severe | 17 | 0 | -24.69 | 24.69 | 0.1468 | 0.1468 |
| E4_MONO_EXPONENTIAL | 5 | mild_0.60_0.70 | 200 | 0 | -13.10 | 13.37 | 0.1007 | 0.1041 |
| E4_MONO_EXPONENTIAL | 5 | moderate | 80 | 0 | -16.67 | 17.16 | 0.1088 | 0.1167 |
| E4_MONO_EXPONENTIAL | 5 | normal | 200 | 0 | -3.10 | 4.16 | 0.0257 | 0.0360 |
| E4_MONO_EXPONENTIAL | 5 | severe | 17 | 0 | -22.10 | 22.10 | 0.1304 | 0.1304 |
| E4_MONO_EXPONENTIAL | full | mild_0.60_0.70 | 200 | 0 | -4.25 | 4.50 | 0.0298 | 0.0316 |
| E4_MONO_EXPONENTIAL | full | moderate | 80 | 0 | -5.41 | 5.90 | 0.0317 | 0.0353 |
| E4_MONO_EXPONENTIAL | full | normal | 200 | 0 | -1.22 | 2.26 | 0.0101 | 0.0185 |
| E4_MONO_EXPONENTIAL | full | severe | 17 | 0 | -4.89 | 5.85 | 0.0228 | 0.0256 |

## Step 3 - Classification Consequence

False-negative rate on obstructed-truth:

| estimator | 2 | 3 | 4 | 5 | full |
|---|---|---|---|---|---|
| E0_DIRECT | 0.990 | 0.865 | 0.684 | 0.549 | 0.000 |
| E1_OMBE | 0.593 | 0.596 | 0.522 | 0.468 | 0.030 |
| E2_SPLINE | 0.006 | 0.019 | 0.019 | 0.032 | 0.000 |
| E3_LINEAR_TAIL | 0.003 | 0.003 | 0.007 | 0.007 | 0.000 |
| E4_MONO_EXPONENTIAL | 0.822 | 0.774 | 0.721 | 0.653 | 0.303 |

Borderline-stratum false-negative rate:

| estimator | 2 | 3 | 4 | 5 | full |
|---|---|---|---|---|---|
| E0_DIRECT | 1.000 | 0.995 | 0.895 | 0.780 | 0.000 |
| E1_OMBE | 0.770 | 0.795 | 0.735 | 0.685 | 0.045 |
| E2_SPLINE | 0.000 | 0.019 | 0.021 | 0.049 | 0.000 |
| E3_LINEAR_TAIL | 0.005 | 0.005 | 0.010 | 0.010 | 0.000 |
| E4_MONO_EXPONENTIAL | 0.910 | 0.905 | 0.885 | 0.850 | 0.450 |

Sensitivity:

| estimator | 2 | 3 | 4 | 5 | full |
|---|---|---|---|---|---|
| E0_DIRECT | 0.010 | 0.135 | 0.316 | 0.451 | 1.000 |
| E1_OMBE | 0.407 | 0.404 | 0.478 | 0.532 | 0.970 |
| E2_SPLINE | 0.994 | 0.981 | 0.981 | 0.968 | 1.000 |
| E3_LINEAR_TAIL | 0.997 | 0.997 | 0.993 | 0.993 | 1.000 |
| E4_MONO_EXPONENTIAL | 0.178 | 0.226 | 0.279 | 0.347 | 0.697 |

Specificity:

| estimator | 2 | 3 | 4 | 5 | full |
|---|---|---|---|---|---|
| E0_DIRECT | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| E1_OMBE | 0.950 | 0.965 | 0.970 | 0.970 | 0.970 |
| E2_SPLINE | 0.000 | 0.029 | 0.039 | 0.077 | 1.000 |
| E3_LINEAR_TAIL | 0.050 | 0.330 | 0.660 | 0.830 | 0.990 |
| E4_MONO_EXPONENTIAL | 0.975 | 0.975 | 0.975 | 0.975 | 0.975 |

Cohen's kappa:

| estimator | 2 | 3 | 4 | 5 | full |
|---|---|---|---|---|---|
| E0_DIRECT | 0.008 | 0.111 | 0.271 | 0.398 | 1.000 |
| E1_OMBE | 0.315 | 0.324 | 0.400 | 0.453 | 0.937 |
| E2_SPLINE | -0.007 | 0.012 | 0.024 | 0.053 | 1.000 |
| E3_LINEAR_TAIL | 0.055 | 0.366 | 0.690 | 0.845 | 0.992 |
| E4_MONO_EXPONENTIAL | 0.128 | 0.169 | 0.217 | 0.279 | 0.628 |

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
| E3_LINEAR_TAIL | normal | full |
| E4_MONO_EXPONENTIAL | severe | none |
| E4_MONO_EXPONENTIAL | moderate | none |
| E4_MONO_EXPONENTIAL | mild_0.60_0.70 | full |
| E4_MONO_EXPONENTIAL | normal | 5 |

## Step 6 - Honesty Checks

- Original Cycle-F obstructed participants: 676
- Obstructed participants in this complete-plateau sample: 297
- Excluded original obstructed participants: 399; NHANES ratio median 0.6624, IQR 0.6268 to 0.6860.
- No truncated estimator received information from after T: True. For truncated rows, every estimator used only volume samples up to T and a fixed 12 s prediction horizon. The true complete-curve duration was not used by the fit.

E1 OMBE overshoot/undershoot by T:

| T | valid | overshoot n | overshoot frac | overshoot % med | overshoot % max | undershoot n | undershoot % med |
|---|---|---|---|---|---|---|---|
| 2 | 497 | 158 | 0.318 | 7.12 | 74.78 | 339 | -10.37 |
| 3 | 497 | 133 | 0.268 | 4.34 | 74.78 | 364 | -10.42 |
| 4 | 497 | 123 | 0.247 | 3.29 | 74.78 | 374 | -8.62 |
| 5 | 497 | 131 | 0.264 | 2.67 | 74.78 | 366 | -6.95 |
| full | 497 | 284 | 0.571 | 0.89 | 74.78 | 213 | -0.59 |

VERDICT: PHYSICS-JUSTIFIED - E1 beats direct obstruction false-negative rate at T<=3 by the material margin and beats spline/linear FVC recovery at T<=3.
