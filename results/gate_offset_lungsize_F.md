# Gate - Constant-Offset Correction + Lung-Size Interaction, Cycle F

- Deterministic LM only. No NUTS, SNPE, gamma, registration, or ML residual.
- Clinical threshold 0.70 and GLI LLN were not moved.
- Corrections were fitted on TRAIN only; all classification headline metrics below are TEST.
- Same split helper/seed as the Cycle F pooling gates: Uses the same stratified_half_split helper and seed 20260708 used by the Cycle F pooling/hybrid gates.

## Verdict

OFFSET: constant-offset correction PARTIALLY CLOSES the model-observed gap. Best K variant: K2; NHANES P3 kappa 0.731; held-out P3 kappa 0.583; bias removal degraded kappa: False.
HONEST-ENDPOINT: on held-out-blow, the best estimator is observed; distinguishable from observed by bootstrap CI: False.
GLI: lung-size-interaction REFUTED. If refuted, GLI degradation remains unexplained after the four tested hypotheses.
DEPLOYABLE: use observed for complete curves and H1 for truncated curves.

## Part 1 - Offset Characterization

Signed rows are model-minus-observed in mL. Absolute-magnitude rows are |model-minus-observed| in mL.

| metric | type | stratum | n | median mL | IQR low | IQR high |
|---|---|---|---|---|---|---|
| fvc | signed | all | 11968 | 28.99 | 1.62 | 106.38 |
| fvc | signed | severe | 38 | -25.46 | -75.65 | 50.46 |
| fvc | signed | moderate | 150 | -19.68 | -70.95 | 15.23 |
| fvc | signed | mild_0.60_0.70 | 862 | 4.86 | -18.01 | 47.31 |
| fvc | signed | normal | 10918 | 31.93 | 3.69 | 114.37 |
| fvc | absolute_magnitude | all | 11968 | 35.04 | 12.49 | 119.63 |
| fvc | absolute_magnitude | severe | 38 | 73.01 | 32.25 | 166.24 |
| fvc | absolute_magnitude | moderate | 150 | 43.28 | 19.24 | 96.24 |
| fvc | absolute_magnitude | mild_0.60_0.70 | 862 | 25.61 | 11.46 | 65.17 |
| fvc | absolute_magnitude | normal | 10918 | 36.02 | 12.44 | 126.02 |
| fev1 | signed | all | 11968 | 4.70 | -17.42 | 24.60 |
| fev1 | signed | severe | 38 | 22.89 | -61.42 | 35.05 |
| fev1 | signed | moderate | 150 | 37.11 | 19.04 | 57.95 |
| fev1 | signed | mild_0.60_0.70 | 862 | 35.42 | 8.25 | 61.92 |
| fev1 | signed | normal | 10918 | 3.02 | -19.49 | 20.73 |
| fev1 | absolute_magnitude | all | 11968 | 21.77 | 8.90 | 48.50 |
| fev1 | absolute_magnitude | severe | 38 | 36.80 | 24.50 | 62.75 |
| fev1 | absolute_magnitude | moderate | 150 | 44.48 | 29.00 | 62.67 |
| fev1 | absolute_magnitude | mild_0.60_0.70 | 862 | 41.43 | 20.68 | 67.93 |
| fev1 | absolute_magnitude | normal | 10918 | 20.32 | 8.41 | 45.60 |

### Offset Regressions

| metric | type | predictor | n | slope | R2 | p |
|---|---|---|---|---|---|---|
| fvc | signed | observed_fvc_mL | 11968 | -0.07818 | 0.2182 | 0.0000 |
| fvc | signed | FET_s | 11968 | -21.46118 | 0.1171 | 0.0000 |
| fvc | signed | tau_s | 11968 | 47.26181 | 0.1197 | 0.0000 |
| fvc | signed | age_years | 11968 | -2.09696 | 0.0491 | 0.0000 |
| fvc | absolute_magnitude | observed_fvc_mL | 11968 | -0.04584 | 0.0834 | 0.0000 |
| fvc | absolute_magnitude | FET_s | 11968 | -15.89093 | 0.0714 | 0.0000 |
| fvc | absolute_magnitude | tau_s | 11968 | 41.78824 | 0.1041 | 0.0000 |
| fvc | absolute_magnitude | age_years | 11968 | -1.97195 | 0.0483 | 0.0000 |
| fev1 | signed | observed_fvc_mL | 11968 | 0.01265 | 0.0306 | 0.0000 |
| fev1 | signed | FET_s | 11968 | 6.66546 | 0.0604 | 0.0000 |
| fev1 | signed | tau_s | 11968 | -2.67442 | 0.0021 | 0.0000 |
| fev1 | signed | age_years | 11968 | 0.95216 | 0.0542 | 0.0000 |
| fev1 | absolute_magnitude | observed_fvc_mL | 11968 | -0.01066 | 0.0290 | 0.0000 |
| fev1 | absolute_magnitude | FET_s | 11968 | -2.98244 | 0.0162 | 0.0000 |
| fev1 | absolute_magnitude | tau_s | 11968 | 11.23907 | 0.0484 | 0.0000 |
| fev1 | absolute_magnitude | age_years | 11968 | -0.47753 | 0.0182 | 0.0000 |

## Part 2 - Constant-Offset Correction

### Fitted Corrections

| variant | form | n | c L | c0 L | c1 | R2 |
|---|---|---|---|---|---|---|
| K1 | FVC_model + c | 11968 | -0.02899 | NA | NA | NA |
| K2 | FVC_model + c0 + c1*FET | 11968 | NA | -0.28082 | 0.02146 | 0.1171 |
| K3 | FVC_model + c0 + c1*tau_s | 11968 | NA | 0.03088 | -0.04726 | 0.1197 |

### Multiplicative Baseline

| n | a | b | R2 |
|---|---|---|---|
| 3326 | 0.846388 | 1.112146 | 0.9535 |

### FVC Residuals

| estimator | split | stratum | n | median mL | IQR low | IQR high | IQR crosses 0 |
|---|---|---|---|---|---|---|---|
| model | test | all | 11766 | 27.31 | 1.73 | 99.71 | False |
| model | test | severe | 41 | -77.86 | -103.06 | 1.28 | True |
| model | test | moderate | 148 | -18.74 | -51.89 | 14.97 | True |
| model | test | mild_0.60_0.70 | 829 | -2.57 | -20.87 | 29.46 | True |
| model | test | normal | 10748 | 30.59 | 4.19 | 111.29 | False |
| model | train | all | 11968 | 28.99 | 1.62 | 106.38 | False |
| model | train | severe | 38 | -25.46 | -75.65 | 50.46 | True |
| model | train | moderate | 150 | -19.68 | -70.95 | 15.23 | True |
| model | train | mild_0.60_0.70 | 862 | 4.86 | -18.01 | 47.31 | True |
| model | train | normal | 10918 | 31.93 | 3.69 | 114.37 | False |
| K1 | test | all | 11766 | -1.69 | -27.27 | 70.71 | True |
| K1 | test | severe | 41 | -106.85 | -132.06 | -27.71 | False |
| K1 | test | moderate | 148 | -47.73 | -80.89 | -14.02 | False |
| K1 | test | mild_0.60_0.70 | 829 | -31.56 | -49.86 | 0.47 | True |
| K1 | test | normal | 10748 | 1.59 | -24.81 | 82.29 | True |
| K1 | train | all | 11968 | -0.00 | -27.37 | 77.39 | True |
| K1 | train | severe | 38 | -54.45 | -104.64 | 21.47 | True |
| K1 | train | moderate | 150 | -48.67 | -99.95 | -13.77 | False |
| K1 | train | mild_0.60_0.70 | 862 | -24.13 | -47.00 | 18.31 | True |
| K1 | train | normal | 10918 | 2.94 | -25.30 | 85.38 | True |
| K2 | test | all | 11766 | -38.71 | -85.86 | 45.46 | True |
| K2 | test | severe | 41 | 30.70 | -30.49 | 66.70 | True |
| K2 | test | moderate | 148 | 35.84 | 0.96 | 73.32 | False |
| K2 | test | mild_0.60_0.70 | 829 | 20.27 | -26.11 | 67.08 | True |
| K2 | test | normal | 10748 | -44.61 | -88.50 | 36.60 | True |
| K2 | train | all | 11968 | -40.78 | -86.57 | 46.05 | True |
| K2 | train | severe | 38 | 76.87 | 32.20 | 122.69 | False |
| K2 | train | moderate | 150 | 40.41 | -10.92 | 69.64 | True |
| K2 | train | mild_0.60_0.70 | 862 | 26.12 | -30.32 | 79.96 | True |
| K2 | train | normal | 10918 | -46.80 | -89.23 | 36.34 | True |
| K3 | test | all | 11766 | -23.76 | -91.05 | 50.76 | True |
| K3 | test | severe | 41 | -283.28 | -308.49 | -173.49 | False |
| K3 | test | moderate | 148 | -198.52 | -253.83 | -134.41 | False |
| K3 | test | mild_0.60_0.70 | 829 | -131.79 | -196.47 | -73.36 | False |
| K3 | test | normal | 10748 | -14.91 | -78.02 | 60.04 | True |
| K3 | train | all | 11968 | -20.45 | -86.96 | 53.70 | True |
| K3 | train | severe | 38 | -227.69 | -281.07 | -101.23 | False |
| K3 | train | moderate | 150 | -195.16 | -272.84 | -110.27 | False |
| K3 | train | mild_0.60_0.70 | 862 | -121.64 | -183.69 | -28.29 | False |
| K3 | train | normal | 10918 | -13.76 | -75.76 | 61.06 | True |
| model_mult | test | all | 11766 | -37.45 | -110.58 | 90.64 | True |
| model_mult | test | severe | 41 | -186.72 | -212.64 | -134.68 | False |
| model_mult | test | moderate | 148 | -139.78 | -178.18 | -49.98 | False |
| model_mult | test | mild_0.60_0.70 | 829 | -66.93 | -127.23 | 28.23 | True |
| model_mult | test | normal | 10748 | -32.58 | -107.35 | 98.97 | True |
| model_mult | train | all | 11968 | -28.87 | -109.84 | 98.37 | True |
| model_mult | train | severe | 38 | -161.80 | -193.26 | 129.47 | True |
| model_mult | train | moderate | 150 | -131.79 | -186.09 | -13.27 | False |
| model_mult | train | mild_0.60_0.70 | 862 | -40.86 | -129.31 | 79.30 | True |
| model_mult | train | normal | 10918 | -27.13 | -106.51 | 100.25 | True |
| H1 | test | all | 11766 | 0.00 | 0.00 | 0.00 | True |
| H1 | test | severe | 41 | 0.00 | 0.00 | 0.00 | True |
| H1 | test | moderate | 148 | 0.00 | 0.00 | 0.00 | True |
| H1 | test | mild_0.60_0.70 | 829 | 0.00 | 0.00 | 0.00 | True |
| H1 | test | normal | 10748 | 0.00 | 0.00 | 0.00 | True |
| H1 | train | all | 11968 | 0.00 | 0.00 | 0.00 | True |
| H1 | train | severe | 38 | 0.00 | 0.00 | 0.00 | True |
| H1 | train | moderate | 150 | 0.00 | 0.00 | 0.00 | True |
| H1 | train | mild_0.60_0.70 | 862 | 0.00 | 0.00 | 0.00 | True |
| H1 | train | normal | 10918 | 0.00 | 0.00 | 0.00 | True |

### Ratio Residuals on TEST

| estimator | stratum | n | median | IQR low | IQR high | IQR crosses 0 |
|---|---|---|---|---|---|---|
| model | severe | 41 | 0.0180 | 0.0085 | 0.0257 | False |
| model | moderate | 148 | 0.0161 | 0.0072 | 0.0231 | False |
| model | mild_0.60_0.70 | 829 | 0.0101 | 0.0010 | 0.0178 | False |
| model | normal | 10748 | -0.0075 | -0.0288 | 0.0019 | True |
| K1 | severe | 41 | 0.0222 | 0.0131 | 0.0297 | False |
| K1 | moderate | 148 | 0.0214 | 0.0117 | 0.0287 | False |
| K1 | mild_0.60_0.70 | 829 | 0.0155 | 0.0065 | 0.0233 | False |
| K1 | normal | 10748 | -0.0005 | -0.0223 | 0.0092 | True |
| K2 | severe | 41 | 0.0057 | 0.0006 | 0.0102 | False |
| K2 | moderate | 148 | 0.0059 | -0.0017 | 0.0152 | True |
| K2 | mild_0.60_0.70 | 829 | 0.0076 | -0.0006 | 0.0154 | True |
| K2 | normal | 10748 | 0.0116 | -0.0100 | 0.0241 | True |
| K3 | severe | 41 | 0.0483 | 0.0396 | 0.0569 | False |
| K3 | moderate | 148 | 0.0515 | 0.0345 | 0.0635 | False |
| K3 | mild_0.60_0.70 | 829 | 0.0356 | 0.0221 | 0.0506 | False |
| K3 | normal | 10748 | 0.0039 | -0.0170 | 0.0223 | True |
| model_mult | severe | 41 | 0.0359 | 0.0318 | 0.0438 | False |
| model_mult | moderate | 148 | 0.0378 | 0.0165 | 0.0512 | False |
| model_mult | mild_0.60_0.70 | 829 | 0.0223 | 0.0045 | 0.0401 | False |
| model_mult | normal | 10748 | 0.0080 | -0.0217 | 0.0323 | True |
| H1 | severe | 41 | 0.0081 | 0.0037 | 0.0128 | False |
| H1 | moderate | 148 | 0.0119 | 0.0060 | 0.0162 | False |
| H1 | mild_0.60_0.70 | 829 | 0.0109 | 0.0052 | 0.0159 | False |
| H1 | normal | 10748 | 0.0011 | -0.0055 | 0.0062 | True |

### TEST Classification, NHANES Target

| estimator | method | definition | n | sens | spec | kappa | borderline FN |
|---|---|---|---|---|---|---|---|
| model | P1 | fixed_0.70 | 3329 | 0.737 | 0.948 | 0.611 | 0.318 |
| model | P1 | GLI_LLN | 3317 | 0.748 | 0.933 | 0.585 | 0.589 |
| model | P2 | fixed_0.70 | 3329 | 0.807 | 0.918 | 0.565 | 0.236 |
| model | P2 | GLI_LLN | 3317 | 0.855 | 0.905 | 0.575 | 0.498 |
| model | P3 | fixed_0.70 | 3329 | 0.820 | 0.952 | 0.680 | 0.219 |
| model | P3 | GLI_LLN | 3317 | 0.828 | 0.936 | 0.645 | 0.531 |
| K1 | P1 | fixed_0.70 | 3329 | 0.643 | 0.952 | 0.562 | 0.434 |
| K1 | P1 | GLI_LLN | 3317 | 0.695 | 0.936 | 0.562 | 0.614 |
| K1 | P2 | fixed_0.70 | 3329 | 0.743 | 0.926 | 0.547 | 0.314 |
| K1 | P2 | GLI_LLN | 3317 | 0.812 | 0.913 | 0.570 | 0.535 |
| K1 | P3 | fixed_0.70 | 3329 | 0.733 | 0.958 | 0.645 | 0.326 |
| K1 | P3 | GLI_LLN | 3317 | 0.788 | 0.940 | 0.632 | 0.564 |
| K2 | P1 | fixed_0.70 | 3329 | 0.743 | 0.960 | 0.659 | 0.310 |
| K2 | P1 | GLI_LLN | 3317 | 0.655 | 0.949 | 0.573 | 0.564 |
| K2 | P2 | fixed_0.70 | 3329 | 0.817 | 0.941 | 0.639 | 0.223 |
| K2 | P2 | GLI_LLN | 3317 | 0.760 | 0.926 | 0.575 | 0.473 |
| K2 | P3 | fixed_0.70 | 3329 | 0.857 | 0.960 | 0.731 | 0.174 |
| K2 | P3 | GLI_LLN | 3317 | 0.742 | 0.949 | 0.632 | 0.465 |
| K3 | P1 | fixed_0.70 | 3329 | 0.450 | 0.966 | 0.458 | 0.669 |
| K3 | P1 | GLI_LLN | 3317 | 0.548 | 0.953 | 0.505 | 0.739 |
| K3 | P2 | fixed_0.70 | 3329 | 0.563 | 0.951 | 0.502 | 0.537 |
| K3 | P2 | GLI_LLN | 3317 | 0.674 | 0.931 | 0.532 | 0.676 |
| K3 | P3 | fixed_0.70 | 3329 | 0.500 | 0.969 | 0.512 | 0.616 |
| K3 | P3 | GLI_LLN | 3317 | 0.622 | 0.958 | 0.577 | 0.714 |
| model_mult | P1 | fixed_0.70 | 3329 | 0.613 | 0.960 | 0.568 | 0.467 |
| model_mult | P1 | GLI_LLN | 3317 | 0.597 | 0.949 | 0.531 | 0.656 |
| model_mult | P2 | fixed_0.70 | 3329 | 0.680 | 0.946 | 0.569 | 0.393 |
| model_mult | P2 | GLI_LLN | 3317 | 0.702 | 0.925 | 0.533 | 0.581 |
| model_mult | P3 | fixed_0.70 | 3329 | 0.663 | 0.962 | 0.614 | 0.413 |
| model_mult | P3 | GLI_LLN | 3317 | 0.658 | 0.953 | 0.586 | 0.618 |
| H1 | P1 | fixed_0.70 | 3329 | 0.707 | 0.987 | 0.750 | 0.355 |
| H1 | P1 | GLI_LLN | 3317 | 0.729 | 0.973 | 0.709 | 0.564 |
| H1 | P2 | fixed_0.70 | 3329 | 0.780 | 0.973 | 0.734 | 0.269 |
| H1 | P2 | GLI_LLN | 3317 | 0.825 | 0.949 | 0.685 | 0.502 |
| H1 | P3 | fixed_0.70 | 3329 | 0.817 | 0.985 | 0.814 | 0.219 |
| H1 | P3 | GLI_LLN | 3317 | 0.812 | 0.971 | 0.758 | 0.519 |
| observed | P1 | fixed_0.70 | 3329 | 0.807 | 0.996 | 0.861 | 0.231 |
| observed | P1 | GLI_LLN | 3317 | 0.803 | 0.994 | 0.849 | 0.485 |
| observed | P2 | fixed_0.70 | 3329 | 0.903 | 0.980 | 0.844 | 0.116 |
| observed | P2 | GLI_LLN | 3317 | 0.883 | 0.972 | 0.804 | 0.398 |
| observed | P3 | fixed_0.70 | 3329 | 0.927 | 0.992 | 0.918 | 0.087 |
| observed | P3 | GLI_LLN | 3317 | 0.911 | 0.988 | 0.890 | 0.390 |

### Classification Flips vs Uncorrected Model

| corrected | definition | n | normal->obs | obs->normal | sens delta | spec delta | kappa delta |
|---|---|---|---|---|---|---|---|
| K1 | fixed_0.70 | 3329 | 0 | 42 | -0.087 | 0.005 | -0.036 |
| K1 | GLI_LLN | 3317 | 0 | 25 | -0.040 | 0.004 | -0.012 |
| K2 | fixed_0.70 | 3329 | 33 | 45 | 0.037 | 0.008 | 0.051 |
| K2 | GLI_LLN | 3317 | 25 | 92 | -0.086 | 0.013 | -0.013 |
| K3 | fixed_0.70 | 3329 | 0 | 146 | -0.320 | 0.017 | -0.168 |
| K3 | GLI_LLN | 3317 | 0 | 132 | -0.206 | 0.022 | -0.067 |
| model_mult | fixed_0.70 | 3329 | 24 | 101 | -0.157 | 0.010 | -0.067 |
| model_mult | GLI_LLN | 3317 | 17 | 121 | -0.169 | 0.016 | -0.058 |

## Part 3 - Non-Circular Held-Out-Blow Endpoint

| n | test n | nonbest frac | rule |
|---|---|---|---|
| 6655 | 3328 | 0.989 | Within each participant, choose one random non-best manoeuvre when available; best means max observed FEV1 or max observed FVC. Predict held-out obstruction from remaining manoeuvres. Seed=20260785. |

| estimator | method | n | sens | spec | kappa | borderline FN |
|---|---|---|---|---|---|---|
| model | P2 | 3328 | 0.676 | 0.924 | 0.543 | 0.377 |
| model | P3 | 3328 | 0.633 | 0.942 | 0.559 | 0.404 |
| K1 | P2 | 3328 | 0.638 | 0.931 | 0.533 | 0.418 |
| K1 | P3 | 3328 | 0.594 | 0.948 | 0.547 | 0.455 |
| K2 | P2 | 3328 | 0.653 | 0.940 | 0.567 | 0.401 |
| K2 | P3 | 3328 | 0.635 | 0.950 | 0.583 | 0.401 |
| K3 | P2 | 3328 | 0.513 | 0.951 | 0.489 | 0.569 |
| K3 | P3 | 3328 | 0.457 | 0.963 | 0.475 | 0.626 |
| model_mult | P2 | 3328 | 0.582 | 0.944 | 0.525 | 0.481 |
| model_mult | P3 | 3328 | 0.538 | 0.955 | 0.523 | 0.519 |
| H1 | P2 | 3328 | 0.661 | 0.976 | 0.683 | 0.394 |
| H1 | P3 | 3328 | 0.630 | 0.983 | 0.684 | 0.397 |
| observed | P2 | 3328 | 0.686 | 0.979 | 0.714 | 0.347 |
| observed | P3 | 3328 | 0.648 | 0.983 | 0.700 | 0.364 |

The held-out-blow endpoint is the fair estimator-quality test because the target is another raw-curve readout withheld from pooling, not an NHANES summary value that is itself an observed-curve readout.

### Endpoint Kappa Gaps vs Observed

| estimator | NHANES kappa | NHANES gap | heldout kappa | heldout gap |
|---|---|---|---|---|
| model | 0.680 | 0.237 | 0.559 | 0.141 |
| K1 | 0.645 | 0.273 | 0.547 | 0.153 |
| K2 | 0.731 | 0.187 | 0.583 | 0.118 |
| K3 | 0.512 | 0.406 | 0.475 | 0.225 |
| model_mult | 0.614 | 0.304 | 0.523 | 0.178 |
| H1 | 0.814 | 0.104 | 0.684 | 0.016 |
| observed | 0.918 | 0.000 | 0.700 | 0.000 |

### Held-Out Bootstrap 95% CI

| estimator | n | kappa | CI low | CI high | B |
|---|---|---|---|---|---|
| K2 | 3328 | 0.583 | 0.539 | 0.626 | 1000 |
| H1 | 3328 | 0.684 | 0.641 | 0.723 | 1000 |
| observed | 3328 | 0.700 | 0.659 | 0.738 | 1000 |

## Part 4 - GLI as Lung-Size Interaction

| FVC quartile | low L | high L | n | model ratio err med | IQR low | IQR high |
|---|---|---|---|---|---|---|
| Q1_smallest | 0.831 | 2.758 | 830 | -0.0078 | -0.0701 | 0.0028 |
| Q2 | 2.760 | 3.504 | 829 | 0.0012 | -0.0066 | 0.0068 |
| Q3 | 3.505 | 4.289 | 829 | 0.0016 | -0.0047 | 0.0088 |
| Q4_largest | 4.291 | 7.331 | 829 | 0.0036 | -0.0033 | 0.0102 |

| FVC quartile | n | LLN med | IQR low | IQR high |
|---|---|---|---|---|
| Q1_smallest | 830 | 0.7654 | 0.6860 | 0.7875 |
| Q2 | 829 | 0.7237 | 0.6885 | 0.7659 |
| Q3 | 829 | 0.7230 | 0.6905 | 0.7535 |
| Q4_largest | 829 | 0.7191 | 0.6958 | 0.7428 |

| predictor | n | r | slope | p |
|---|---|---|---|---|
| observed_FVC_l | 3317 | -0.198 | -0.00869 | 0.0000 |
| age_years | 3317 | -0.950 | -0.00226 | 0.0000 |
| height_cm | 3317 | -0.505 | -0.00166 | 0.0000 |

### GLI False Negatives by FVC Quartile

| FVC quartile | estimator | n | sens | spec | FN obs | borderline FN |
|---|---|---|---|---|---|---|
| Q1_smallest | model | 830 | 0.909 | 0.772 | 0.091 | 0.176 |
| Q1_smallest | H1 | 830 | 0.864 | 0.908 | 0.136 | 0.176 |
| Q1_smallest | observed | 830 | 0.894 | 0.983 | 0.106 | 0.059 |
| Q1_smallest | K2 | 830 | 0.758 | 0.819 | 0.242 | 0.118 |
| Q2 | model | 829 | 0.890 | 0.991 | 0.110 | 0.171 |
| Q2 | H1 | 829 | 0.841 | 0.993 | 0.159 | 0.229 |
| Q2 | observed | 829 | 0.939 | 0.991 | 0.061 | 0.057 |
| Q2 | K2 | 829 | 0.756 | 0.992 | 0.244 | 0.171 |
| Q3 | model | 829 | 0.792 | 0.995 | 0.208 | 0.206 |
| Q3 | H1 | 829 | 0.753 | 0.996 | 0.247 | 0.265 |
| Q3 | observed | 829 | 0.935 | 0.989 | 0.065 | 0.059 |
| Q3 | K2 | 829 | 0.727 | 0.993 | 0.273 | 0.147 |
| Q4_largest | model | 829 | 0.750 | 0.992 | 0.250 | 0.286 |
| Q4_largest | H1 | 829 | 0.800 | 0.989 | 0.200 | 0.179 |
| Q4_largest | observed | 829 | 0.880 | 0.989 | 0.120 | 0.036 |
| Q4_largest | K2 | 829 | 0.730 | 0.996 | 0.270 | 0.179 |

### GLI After Best K Correction

| estimator | method | n | sens | spec | kappa | borderline FN |
|---|---|---|---|---|---|---|
| model | P3 | 3317 | 0.828 | 0.936 | 0.645 | 0.531 |
| K2 | P3 | 3317 | 0.742 | 0.949 | 0.632 | 0.465 |

## Part 5 - H2 and Truncation Fallback

H2 = model FEV1 / K2-corrected model FVC.

### H2 Complete-Curve Classification

| method | definition | n | sens | spec | kappa | borderline FN |
|---|---|---|---|---|---|---|
| P1 | fixed_0.70 | 3329 | 0.743 | 0.960 | 0.659 | 0.310 |
| P1 | GLI_LLN | 3317 | 0.655 | 0.949 | 0.573 | 0.564 |
| P2 | fixed_0.70 | 3329 | 0.817 | 0.941 | 0.639 | 0.223 |
| P2 | GLI_LLN | 3317 | 0.760 | 0.926 | 0.575 | 0.473 |
| P3 | fixed_0.70 | 3329 | 0.857 | 0.960 | 0.731 | 0.174 |
| P3 | GLI_LLN | 3317 | 0.742 | 0.949 | 0.632 | 0.465 |

### H2 Held-Out-Blow Classification

| method | n | sens | spec | kappa | borderline FN |
|---|---|---|---|---|---|
| P2 | 3328 | 0.653 | 0.940 | 0.567 | 0.401 |
| P3 | 3328 | 0.635 | 0.950 | 0.583 | 0.401 |

### Truncation Classification

| estimator | T | n | sens | spec | kappa | obstructed FN |
|---|---|---|---|---|---|---|
| E0_DIRECT | 2 | 497 | 0.010 | 1.000 | 0.008 | 0.990 |
| E0_DIRECT | 3 | 497 | 0.135 | 1.000 | 0.111 | 0.865 |
| E0_DIRECT | 4 | 497 | 0.316 | 1.000 | 0.271 | 0.684 |
| E0_DIRECT | 5 | 497 | 0.451 | 1.000 | 0.398 | 0.549 |
| E1_OMBE | 2 | 497 | 0.411 | 0.965 | 0.330 | 0.589 |
| E1_OMBE | 3 | 497 | 0.404 | 0.970 | 0.328 | 0.596 |
| E1_OMBE | 4 | 497 | 0.481 | 0.970 | 0.403 | 0.519 |
| E1_OMBE | 5 | 497 | 0.532 | 0.970 | 0.453 | 0.468 |
| H1 | 2 | 497 | 0.421 | 0.965 | 0.340 | 0.579 |
| H1 | 3 | 497 | 0.421 | 0.970 | 0.344 | 0.579 |
| H1 | 4 | 497 | 0.505 | 0.970 | 0.426 | 0.495 |
| H1 | 5 | 497 | 0.559 | 0.970 | 0.480 | 0.441 |
| H2 | 2 | 497 | 0.222 | 0.985 | 0.175 | 0.778 |
| H2 | 3 | 497 | 0.209 | 0.985 | 0.163 | 0.791 |
| H2 | 4 | 497 | 0.259 | 0.985 | 0.207 | 0.741 |
| H2 | 5 | 497 | 0.333 | 0.980 | 0.270 | 0.667 |

### Truncation Offset Refit Drift

| T | train n | refit c L | complete K1 c L | drift L |
|---|---|---|---|---|
| 2 | 242 | 0.16607 | -0.02899 | 0.19507 |
| 3 | 242 | 0.20240 | -0.02899 | 0.23139 |
| 4 | 242 | 0.17895 | -0.02899 | 0.20794 |
| 5 | 242 | 0.14009 | -0.02899 | 0.16908 |

## Output Files

- `results/gate_offset_lungsize_F.md`
- `results/gate_offset_lungsize_F.json`
- `results/gate_offset_lungsize_F_rows.csv`
