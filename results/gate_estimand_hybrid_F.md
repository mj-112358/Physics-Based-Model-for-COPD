# Gate - Estimand Reconciliation + Hybrid Readout, Cycle F

- CIRCULARITY: observed_P3 IS NOT a reconstruction of the NHANES reference algorithm.
- MULTIPLICATIVE: sign flip REMOVED (a=0.846388, b=1.112146); overfit=False; kappa gap closed: False.
- HYBRID: H1 does not dominate both (H1 complete P3 kappa 0.814 vs observed 0.918; T=2 FN H1 0.579, E0 0.990, E1 0.589).
- GLI: unexplained.
- Deterministic LM only. No NUTS, SNPE, gamma, registration, or ML residual.

## Part 1 - Is observed_P3 Circular?

### Available Field Definitions

- SPXNSTAT: Spirometry First Test Status
- SPXNFVC: Baseline FVC (mL)
- SPXNFEV1: Baseline FEV 1 (mL)
- SPXNQEFF: Baseline Effort Quality Attribute
- SPXNQFVC: Baseline FVC Quality Attribute
- SPXNQFV1: Baseline FEV1 Quality Attribute
- SPXBSTAT: Spirometry Second Test Status
- SPXBFVC: 2nd Test FVC (mL)
- SPXBFEV1: 2nd Test FEV 1 (mL)
- SPXBQEFF: 2nd Test Effort Quality Attribute
- SPXBQFVC: 2nd Test FVC Quality Attribute
- SPXBQFV1: 2nd Test FEV1 Quality Attribute

- SPXB fields labelled as best fields by metadata: False
- SPAFACT field definition: BTPS Factor. The raw-curve readouts in this gate apply the per-manoeuvre BTPS factor through the existing parser.

### observed_P3 vs Summary Fields

| prefix | reference | n | FEV1 abs mL med | FVC abs mL med | ratio abs med | both <=25mL |
|---|---|---|---|---|---|---|
| SPXN | baseline | 3329 | 87.69 | 128.62 | 0.0062 | 0.003 |
| SPXB | second_test | 241 | 259.31 | 135.74 | 0.0487 | 0.004 |

### Held-Out Manoeuvre Generalization

| estimator | method | n | sens | spec | kappa |
|---|---|---|---|---|---|
| model | P2 | 3328 | 0.676 | 0.924 | 0.543 |
| model | P3 | 3328 | 0.633 | 0.942 | 0.559 |
| observed | P2 | 3328 | 0.686 | 0.979 | 0.714 |
| observed | P3 | 3328 | 0.648 | 0.983 | 0.700 |
| H1 | P2 | 3328 | 0.661 | 0.976 | 0.683 |
| H1 | P3 | 3328 | 0.630 | 0.983 | 0.684 |

Held-out P3 kappa gap: model 0.559, observed 0.700, H1 0.684. Bootstrap CIs overlap: observed 0.700 (95% CI 0.656-0.738), H1 0.684 (95% CI 0.640-0.722); paired observed-minus-H1 delta 0.016 (95% CI -0.006 to 0.038), so indistinguishable/overlap is supported.

Bootstrap details: patient bootstrap over held-out test participants, 2000 resamples, seed 20260710; endpoint reconstruction seed 20260785 reproduced the stored point kappas exactly.

Metadata/holdout note: SPXB is labelled as second test, not best-of, so the SPXB comparison is not evidence that observed_P3 reconstructs a best-field algorithm.

## Part 2 - Multiplicative FVC Correction

| predictor | n | slope | intercept | R2 | p |
|---|---|---|---|---|---|
| log_observed_fvc | 3326 | 0.85733 | 0.20051 | 0.953 | 0.000 |

| n | a | b | R2 | p |
|---|---|---|---|---|
| 3326 | 0.846388 | 1.112146 | 0.953 | 0.000 |

### FVC Residuals Before Correction

| split | stratum | n | median % | IQR low | IQR high | IQR crosses 0 |
|---|---|---|---|---|---|---|
| test | mild_0.60_0.70 | 242 | -0.184 | -0.620 | 0.650 | True |
| test | moderate | 45 | -0.961 | -2.078 | 0.232 | True |
| test | normal | 3029 | 0.755 | 0.088 | 2.237 | False |
| test | severe | 13 | -2.589 | -3.037 | -1.887 | False |
| test | obstructed_all | 300 | -0.255 | -0.894 | 0.545 | True |
| train | mild_0.60_0.70 | 241 | -0.027 | -0.583 | 0.721 | True |
| train | moderate | 44 | -0.713 | -2.693 | -0.051 | False |
| train | normal | 3029 | 0.771 | 0.081 | 2.297 | False |
| train | severe | 12 | -1.464 | -2.783 | 0.648 | True |
| train | obstructed_all | 297 | -0.095 | -0.802 | 0.630 | True |

### FVC Residuals After Multiplicative Correction

| split | stratum | n | median % | IQR low | IQR high | IQR crosses 0 |
|---|---|---|---|---|---|---|
| test | mild_0.60_0.70 | 242 | -1.634 | -4.041 | 0.692 | True |
| test | moderate | 45 | -4.311 | -6.049 | -1.353 | False |
| test | normal | 3029 | -1.036 | -3.671 | 1.912 | True |
| test | severe | 13 | -5.929 | -6.551 | -4.694 | False |
| test | obstructed_all | 300 | -2.221 | -4.839 | 0.081 | True |
| train | mild_0.60_0.70 | 241 | -1.574 | -4.230 | 1.502 | True |
| train | moderate | 44 | -4.126 | -6.593 | -1.180 | False |
| train | normal | 3029 | -0.881 | -3.559 | 1.855 | True |
| train | severe | 12 | -6.251 | -7.158 | 1.130 | True |
| train | obstructed_all | 297 | -2.013 | -4.988 | 1.266 | True |

TRAIN-to-TEST stability flag: overfit=False. Aggregate obstructed and normal residuals did not materially worsen on TEST under the predeclared rule.
Severity-specific residuals can still fail to bracket zero even when the aggregate obstructed-vs-normal sign flip is removed.

### Classification Side-by-Side

| estimator | method | definition | n | sens | spec | kappa | borderline FN |
|---|---|---|---|---|---|---|---|
| model | P1 | fixed_0.70 | 3329 | 0.737 | 0.948 | 0.611 | 0.318 |
| model | P1 | GLI_LLN | 3317 | 0.748 | 0.933 | 0.585 | 0.589 |
| model | P2 | fixed_0.70 | 3329 | 0.807 | 0.918 | 0.565 | 0.236 |
| model | P2 | GLI_LLN | 3317 | 0.855 | 0.905 | 0.575 | 0.498 |
| model | P3 | fixed_0.70 | 3329 | 0.820 | 0.952 | 0.680 | 0.219 |
| model | P3 | GLI_LLN | 3317 | 0.828 | 0.936 | 0.645 | 0.531 |
| model_mult | P1 | fixed_0.70 | 3329 | 0.613 | 0.960 | 0.568 | 0.467 |
| model_mult | P1 | GLI_LLN | 3317 | 0.597 | 0.949 | 0.531 | 0.656 |
| model_mult | P2 | fixed_0.70 | 3329 | 0.680 | 0.946 | 0.569 | 0.393 |
| model_mult | P2 | GLI_LLN | 3317 | 0.702 | 0.925 | 0.533 | 0.581 |
| model_mult | P3 | fixed_0.70 | 3329 | 0.663 | 0.962 | 0.614 | 0.413 |
| model_mult | P3 | GLI_LLN | 3317 | 0.658 | 0.953 | 0.586 | 0.618 |
| observed | P1 | fixed_0.70 | 3329 | 0.807 | 0.996 | 0.861 | 0.231 |
| observed | P1 | GLI_LLN | 3317 | 0.803 | 0.994 | 0.849 | 0.485 |
| observed | P2 | fixed_0.70 | 3329 | 0.903 | 0.980 | 0.844 | 0.116 |
| observed | P2 | GLI_LLN | 3317 | 0.883 | 0.972 | 0.804 | 0.398 |
| observed | P3 | fixed_0.70 | 3329 | 0.927 | 0.992 | 0.918 | 0.087 |
| observed | P3 | GLI_LLN | 3317 | 0.911 | 0.988 | 0.890 | 0.390 |
| H1 | P1 | fixed_0.70 | 3329 | 0.707 | 0.987 | 0.750 | 0.355 |
| H1 | P1 | GLI_LLN | 3317 | 0.729 | 0.973 | 0.709 | 0.564 |
| H1 | P2 | fixed_0.70 | 3329 | 0.780 | 0.973 | 0.734 | 0.269 |
| H1 | P2 | GLI_LLN | 3317 | 0.825 | 0.949 | 0.685 | 0.502 |
| H1 | P3 | fixed_0.70 | 3329 | 0.817 | 0.985 | 0.814 | 0.219 |
| H1 | P3 | GLI_LLN | 3317 | 0.812 | 0.971 | 0.758 | 0.519 |

## Part 3 - H1 Hybrid Readout

H1 = model FEV1 / observed segmented-BTPS FVC on complete curves.

### H1 Signed Ratio Error

| stratum | n | median | IQR low | IQR high | IQR crosses 0 |
|---|---|---|---|---|---|
| mild_0.60_0.70 | 242 | 0.0107 | 0.0046 | 0.0173 | False |
| moderate | 45 | 0.0103 | 0.0006 | 0.0178 | False |
| normal | 3029 | 0.0048 | -0.0035 | 0.0148 | True |
| severe | 13 | 0.0100 | 0.0016 | 0.0151 | False |

### H1 Graceful Degradation Under Truncation

| source | T | n | sens | spec | kappa | obstructed FN |
|---|---|---|---|---|---|---|
| E0_observed_direct | 2 | 497 | 0.010 | 1.000 | 0.008 | 0.990 |
| E0_observed_direct | 3 | 497 | 0.135 | 1.000 | 0.111 | 0.865 |
| E0_observed_direct | 4 | 497 | 0.316 | 1.000 | 0.271 | 0.684 |
| E0_observed_direct | 5 | 497 | 0.451 | 1.000 | 0.398 | 0.549 |
| E1_existing_model | 2 | 497 | 0.411 | 0.965 | 0.330 | 0.589 |
| E1_existing_model | 3 | 497 | 0.404 | 0.970 | 0.328 | 0.596 |
| E1_existing_model | 4 | 497 | 0.481 | 0.970 | 0.403 | 0.519 |
| E1_existing_model | 5 | 497 | 0.532 | 0.970 | 0.453 | 0.468 |
| H1_fallback_model_fvc | 2 | 497 | 0.421 | 0.965 | 0.340 | 0.579 |
| H1_fallback_model_fvc | 3 | 497 | 0.421 | 0.970 | 0.344 | 0.579 |
| H1_fallback_model_fvc | 4 | 497 | 0.505 | 0.970 | 0.426 | 0.495 |
| H1_fallback_model_fvc | 5 | 497 | 0.559 | 0.970 | 0.480 | 0.441 |

### Repeatability

| estimator | n | FEV1 SD | FVC SD | ratio SD | ATS repeatable |
|---|---|---|---|---|---|
| model | 6655 | 0.0863 | 0.0745 | 0.0143 | 0.876 |
| observed | 6655 | 0.0830 | 0.0930 | 0.0185 | 0.844 |
| H1 | 6655 | 0.0863 | 0.0930 | 0.0178 | 0.835 |

## Part 4 - GLI Reference-Equation Coverage

- clean-mapping excluded fraction: 0.383

| subgroup | estimator | method | n | sens | spec | kappa | borderline FN |
|---|---|---|---|---|---|---|---|
| clean_mapping | model | P1 | 2045 | 0.743 | 0.937 | 0.604 | 0.637 |
| clean_mapping | model | P2 | 2045 | 0.850 | 0.912 | 0.601 | 0.553 |
| clean_mapping | model | P3 | 2045 | 0.822 | 0.941 | 0.667 | 0.584 |
| clean_mapping | observed | P1 | 2045 | 0.818 | 0.996 | 0.871 | 0.526 |
| clean_mapping | observed | P2 | 2045 | 0.893 | 0.974 | 0.826 | 0.453 |
| clean_mapping | observed | P3 | 2045 | 0.925 | 0.992 | 0.919 | 0.442 |
| clean_mapping | H1 | P1 | 2045 | 0.743 | 0.976 | 0.736 | 0.595 |
| clean_mapping | H1 | P2 | 2045 | 0.832 | 0.955 | 0.717 | 0.542 |
| clean_mapping | H1 | P3 | 2045 | 0.832 | 0.975 | 0.790 | 0.553 |
| other_mixed_or_fallback | model | P1 | 1272 | 0.757 | 0.927 | 0.553 | 0.412 |
| other_mixed_or_fallback | model | P2 | 1272 | 0.865 | 0.896 | 0.531 | 0.294 |
| other_mixed_or_fallback | model | P3 | 1272 | 0.838 | 0.929 | 0.606 | 0.333 |
| other_mixed_or_fallback | observed | P1 | 1272 | 0.775 | 0.990 | 0.807 | 0.333 |
| other_mixed_or_fallback | observed | P2 | 1272 | 0.865 | 0.968 | 0.764 | 0.196 |
| other_mixed_or_fallback | observed | P3 | 1272 | 0.883 | 0.982 | 0.838 | 0.196 |
| other_mixed_or_fallback | H1 | P1 | 1272 | 0.703 | 0.968 | 0.660 | 0.451 |
| other_mixed_or_fallback | H1 | P2 | 1272 | 0.811 | 0.941 | 0.628 | 0.353 |
| other_mixed_or_fallback | H1 | P3 | 1272 | 0.775 | 0.966 | 0.698 | 0.392 |

## Part 5 - GP Discrepancy Diagnostic

GP length scale is curve-scale: median length/FET 0.643; amplitude-FVC correlation r=0.004, slope=0.0075 mL per mL.

| diagnostic | n | r | slope | p | length/FET med | amp mL med |
|---|---|---|---|---|---|---|
| length_vs_fet | 143 | 0.733 | 0.49269 | 0.000 | 0.643 | NA |
| amplitude_vs_fvc | 143 | 0.004 | 0.00747 | 0.962 | NA | 419.99 |

## Verdict

CIRCULARITY: observed_P3 IS NOT a reconstruction of the NHANES reference algorithm; held-out manoeuvre rows above report the non-circular kappa gap.
MULTIPLICATIVE: sign flip REMOVED with a=0.846388, b=1.112146; overfit=False; kappa gap closed: False.
HYBRID: H1 kappa 0.814 vs observed_P3 0.918; T=2 obstructed FN H1 0.579, E0 0.990, E1 0.589.
GLI: unexplained.
