# Gate - Truncation Hardening, Cycle F

> Evidence status: historical and provisional. This report predates the
> fixed-horizon correction. The primary corrected result is
> `gate_truncation_recovery_F.md`.

## Part 0 - Ground-Truth Dump

This section was read before writing any new hardening output.

### Clean Truncation Sample And Endpoint

| item | value | source |
|---|---|---|
| endpoint | Every truncated estimator used only volume samples up to T; full curve was used only for scoring and as the requested extrapolation target time. | results/gate_truncation_recovery_F.md:6 |
| total clean-cut n | 497 | results/gate_truncation_recovery_F.md:26 |
| obstructed denominator | 297 = 17 severe + 80 moderate + 200 mild_0.60_0.70 | results/gate_truncation_recovery_F.md:16-18 |
| specificity denominator | 200 normal | results/gate_truncation_recovery_F.md:19 |
| plateau rule | FET >= 6.0 s and terminal-second dV <= 0.025 L. | results/gate_truncation_recovery_F.md:21 |
| competitor tuple | ("E0_DIRECT", "E1_OMBE", "E2_SPLINE", "E3_LINEAR_TAIL", "E4_MONO_EXPONENTIAL") | predictx/gate_truncation_recovery_f.py:24 |
| competitor definitions | spline, linear tail, mono-exponential are implemented; no GP is in the estimator tuple. | predictx/gate_truncation_recovery_f.py:736-738 |

### Existing Clean-Cut Classification Rates

| estimator | T | n | tp | tn | fp | fn | fit failures | obstructed FN | specificity | source |
|---|---|---|---|---|---|---|---|---|---|---|
| E0_DIRECT | 2 | 497 | 3 | 200 | 0 | 294 | 0 | 0.989899 | 1.000000 | results/gate_truncation_recovery_F.json:classification_summary |
| E0_DIRECT | 3 | 497 | 40 | 200 | 0 | 257 | 0 | 0.865320 | 1.000000 | results/gate_truncation_recovery_F.json:classification_summary |
| E0_DIRECT | 4 | 497 | 94 | 200 | 0 | 203 | 0 | 0.683502 | 1.000000 | results/gate_truncation_recovery_F.json:classification_summary |
| E0_DIRECT | 5 | 497 | 134 | 200 | 0 | 163 | 0 | 0.548822 | 1.000000 | results/gate_truncation_recovery_F.json:classification_summary |
| E1_OMBE | 2 | 497 | 122 | 193 | 7 | 175 | 0 | 0.589226 | 0.965000 | results/gate_truncation_recovery_F.json:classification_summary |
| E1_OMBE | 3 | 497 | 120 | 194 | 6 | 177 | 0 | 0.595960 | 0.970000 | results/gate_truncation_recovery_F.json:classification_summary |
| E1_OMBE | 4 | 497 | 143 | 194 | 6 | 154 | 0 | 0.518519 | 0.970000 | results/gate_truncation_recovery_F.json:classification_summary |
| E1_OMBE | 5 | 497 | 158 | 194 | 6 | 139 | 0 | 0.468013 | 0.970000 | results/gate_truncation_recovery_F.json:classification_summary |
| E2_SPLINE | 2 | 278 | 170 | 1 | 105 | 2 | 219 | 0.011628 | 0.009434 | results/gate_truncation_recovery_F.json:classification_summary |
| E2_SPLINE | 3 | 259 | 152 | 6 | 97 | 4 | 238 | 0.025641 | 0.058252 | results/gate_truncation_recovery_F.json:classification_summary |
| E2_SPLINE | 4 | 269 | 153 | 15 | 96 | 5 | 228 | 0.031646 | 0.135135 | results/gate_truncation_recovery_F.json:classification_summary |
| E2_SPLINE | 5 | 269 | 151 | 21 | 92 | 5 | 228 | 0.032051 | 0.185841 | results/gate_truncation_recovery_F.json:classification_summary |
| E3_LINEAR_TAIL | 2 | 497 | 296 | 18 | 182 | 1 | 0 | 0.003367 | 0.090000 | results/gate_truncation_recovery_F.json:classification_summary |
| E3_LINEAR_TAIL | 3 | 497 | 297 | 101 | 99 | 0 | 0 | 0.000000 | 0.505000 | results/gate_truncation_recovery_F.json:classification_summary |
| E3_LINEAR_TAIL | 4 | 497 | 294 | 150 | 50 | 3 | 0 | 0.010101 | 0.750000 | results/gate_truncation_recovery_F.json:classification_summary |
| E3_LINEAR_TAIL | 5 | 497 | 296 | 172 | 28 | 1 | 0 | 0.003367 | 0.860000 | results/gate_truncation_recovery_F.json:classification_summary |
| E4_MONO_EXPONENTIAL | 2 | 497 | 53 | 195 | 5 | 244 | 0 | 0.821549 | 0.975000 | results/gate_truncation_recovery_F.json:classification_summary |
| E4_MONO_EXPONENTIAL | 3 | 497 | 68 | 195 | 5 | 229 | 0 | 0.771044 | 0.975000 | results/gate_truncation_recovery_F.json:classification_summary |
| E4_MONO_EXPONENTIAL | 4 | 497 | 83 | 195 | 5 | 214 | 0 | 0.720539 | 0.975000 | results/gate_truncation_recovery_F.json:classification_summary |
| E4_MONO_EXPONENTIAL | 5 | 497 | 103 | 195 | 5 | 194 | 0 | 0.653199 | 0.975000 | results/gate_truncation_recovery_F.json:classification_summary |
| H1 | 2 | 497 | 125 | 193 | 7 | 172 | 0 | 0.579125 | 0.965000 | results/gate_offset_lungsize_F_rows.csv:part5_truncation_classification |
| H1 | 3 | 497 | 125 | 194 | 6 | 172 | 0 | 0.579125 | 0.970000 | results/gate_offset_lungsize_F_rows.csv:part5_truncation_classification |
| H1 | 4 | 497 | 150 | 194 | 6 | 147 | 0 | 0.494949 | 0.970000 | results/gate_offset_lungsize_F_rows.csv:part5_truncation_classification |
| H1 | 5 | 497 | 166 | 194 | 6 | 131 | 0 | 0.441077 | 0.970000 | results/gate_offset_lungsize_F_rows.csv:part5_truncation_classification |

### Other Current Claims

| item | value | source |
|---|---|---|
| fixed 0.70 confusion | NHANES normal 287/13; NHANES obstructed 2/112 | results/gate_clinical_classification_F.md:34-37 |
| fixed 0.70 metrics | sensitivity 0.982; specificity 0.957; Cohen's kappa 0.912 | results/gate_clinical_classification_F.md:41-44 |
| GLI LLN metrics | sensitivity 0.966; specificity 0.939; Cohen's kappa 0.874 | results/gate_clinical_classification_F.md:58-61 |
| severe demoted to normal | 0/31 (0.000) | results/gate_clinical_classification_F.md:73-74 |
| held-out H1/observed CI | H1 3328 0.684 0.641 0.723 1000; observed 3328 0.700 0.659 0.738 1000 | results/gate_offset_lungsize_F.md:270-276 |
| complete curve margin | model P3 kappa 0.680 vs observed P3 0.918 | results/gate_model_vs_observed_F.md:117 |
| repeatability verdict | VERDICT: UNSTABLE; borderline model unanimous fraction 0.698; model-vs-observed ATS repeatability margin 3.2 percentage points. | results/gate_reproducibility_volume_F.md:57 |
| FVC SD | observed 0.0931; model 0.0745 | results/gate_reproducibility_volume_F.md:27 |
| ratio ICC | observed 0.793; model 0.778; n=6677 | results/gate_reproducibility_volume_F.md:30-31 |
| ATS repeatability | observed 0.843; model 0.875; delta 3.2 percentage points | results/gate_reproducibility_volume_F.md:47-50 |
| Gate 2 parse | PASS; cycle F historical; E and G analyzed separately in the three-cycle extension; n 7079; FEV1 2.91%; FVC 3.53% | results/gate2_F_parse.md:3-5,37-39 |
| Gate 3 adequacy | FAIL; cycle F historical; E and G analyzed separately in the three-cycle extension; 0/31851 (0.0% of all curves; 0.0% of NRSS-valid curves) | results/gate3_F_ombe_rerun.md:3-5,26 |

## Protocol Decisions

- Realistic quality flag rule: `SPAQEFF in {C,D} OR SPAACC != Y OR SPAPLAT != Y`.
- Used actual same-patient lower-quality/incomplete Pre curves where a clean complete-plateau target exists.
- Selected the shortest usable flagged curve per patient to avoid repeated correlated rows and stress early termination.
- The field-realistic endpoint is the extracted forced-expiration duration of that flagged curve. If it exceeds the clean target end, the analysis clips at the clean target end so no estimator sees beyond the target endpoint.
- Hardening metrics are conservative: failed or nonfinite fits are treated as non-obstructed calls for false-negative and specificity calculations, and fit-failure counts are reported.
- No NUTS, TSNPE, sampler, gamma-onset, registration, or ML residual code is used.

## Part 1 - Realistic Truncation

| clean target patients | flagged Pre rows | matched candidate rows | matched patients | usable candidate rows | selected patients |
|---|---|---|---|---|---|
| 497 | 10286 | 593 | 286 | 582 | 286 |

Selected endpoint bucket counts: `{'field_2_3': 42, 'field_3_4': 42, 'field_4_5': 20, 'field_gt_5': 119, 'field_le_2': 63}`.

### Hardening Classification Summary

| dataset | endpoint | estimator | all n | valid n | fit/nonfinite n | obstructed n | normal n | obstructed FN | specificity | kappa |
|---|---|---|---|---|---|---|---|---|---|---|
| clean_cut | 2 | E0_DIRECT | 497 | 497 | 0 | 297 | 200 | 0.990 | 1.000 | 0.008 |
| clean_cut | 2 | E1_OMBE | 497 | 497 | 0 | 297 | 200 | 0.589 | 0.965 | 0.330 |
| clean_cut | 2 | E2_SPLINE | 497 | 278 | 219 | 297 | 200 | 0.428 | 0.475 | 0.047 |
| clean_cut | 2 | E3_LINEAR_TAIL | 497 | 497 | 0 | 297 | 200 | 0.003 | 0.090 | 0.102 |
| clean_cut | 2 | E4_MONO_EXPONENTIAL | 497 | 497 | 0 | 297 | 200 | 0.822 | 0.975 | 0.128 |
| clean_cut | 2 | H1 | 497 | 497 | 0 | 297 | 200 | 0.579 | 0.965 | 0.340 |
| clean_cut | 3 | E0_DIRECT | 497 | 497 | 0 | 297 | 200 | 0.865 | 1.000 | 0.111 |
| clean_cut | 3 | E1_OMBE | 497 | 497 | 0 | 297 | 200 | 0.596 | 0.970 | 0.328 |
| clean_cut | 3 | E2_SPLINE | 497 | 259 | 238 | 297 | 200 | 0.488 | 0.515 | 0.026 |
| clean_cut | 3 | E3_LINEAR_TAIL | 497 | 497 | 0 | 297 | 200 | 0.000 | 0.505 | 0.549 |
| clean_cut | 3 | E4_MONO_EXPONENTIAL | 497 | 497 | 0 | 297 | 200 | 0.771 | 0.975 | 0.172 |
| clean_cut | 3 | H1 | 497 | 497 | 0 | 297 | 200 | 0.579 | 0.970 | 0.344 |
| clean_cut | 4 | E0_DIRECT | 497 | 497 | 0 | 297 | 200 | 0.684 | 1.000 | 0.271 |
| clean_cut | 4 | E1_OMBE | 497 | 497 | 0 | 297 | 200 | 0.519 | 0.970 | 0.403 |
| clean_cut | 4 | E2_SPLINE | 497 | 269 | 228 | 297 | 200 | 0.485 | 0.520 | 0.034 |
| clean_cut | 4 | E3_LINEAR_TAIL | 497 | 497 | 0 | 297 | 200 | 0.010 | 0.750 | 0.769 |
| clean_cut | 4 | E4_MONO_EXPONENTIAL | 497 | 497 | 0 | 297 | 200 | 0.721 | 0.975 | 0.217 |
| clean_cut | 4 | H1 | 497 | 497 | 0 | 297 | 200 | 0.495 | 0.970 | 0.426 |
| clean_cut | 5 | E0_DIRECT | 497 | 497 | 0 | 297 | 200 | 0.549 | 1.000 | 0.398 |
| clean_cut | 5 | E1_OMBE | 497 | 497 | 0 | 297 | 200 | 0.468 | 0.970 | 0.453 |
| clean_cut | 5 | E2_SPLINE | 497 | 269 | 228 | 297 | 200 | 0.492 | 0.540 | 0.046 |
| clean_cut | 5 | E3_LINEAR_TAIL | 497 | 497 | 0 | 297 | 200 | 0.003 | 0.860 | 0.876 |
| clean_cut | 5 | E4_MONO_EXPONENTIAL | 497 | 497 | 0 | 297 | 200 | 0.653 | 0.975 | 0.279 |
| clean_cut | 5 | H1 | 497 | 497 | 0 | 297 | 200 | 0.441 | 0.970 | 0.480 |
| field_realistic | field_2_3 | E0_DIRECT | 42 | 42 | 0 | 26 | 16 | 0.769 | 1.000 | 0.186 |
| field_realistic | field_2_3 | E1_OMBE | 42 | 42 | 0 | 26 | 16 | 0.038 | 0.438 | 0.441 |
| field_realistic | field_2_3 | E2_SPLINE | 42 | 8 | 34 | 26 | 16 | 0.808 | 0.812 | 0.004 |
| field_realistic | field_2_3 | E3_LINEAR_TAIL | 42 | 42 | 0 | 26 | 16 | 0.038 | 0.250 | 0.244 |
| field_realistic | field_2_3 | E4_MONO_EXPONENTIAL | 42 | 42 | 0 | 26 | 16 | 0.423 | 0.625 | 0.190 |
| field_realistic | field_2_3 | H1 | 42 | 42 | 0 | 26 | 16 | 0.038 | 0.438 | 0.441 |
| field_realistic | field_3_4 | E0_DIRECT | 42 | 42 | 0 | 21 | 21 | 0.667 | 1.000 | 0.333 |
| field_realistic | field_3_4 | E1_OMBE | 42 | 42 | 0 | 21 | 21 | 0.048 | 1.000 | 0.952 |
| field_realistic | field_3_4 | E2_SPLINE | 42 | 7 | 35 | 21 | 21 | 0.857 | 0.810 | -0.048 |
| field_realistic | field_3_4 | E3_LINEAR_TAIL | 42 | 42 | 0 | 21 | 21 | 0.095 | 0.714 | 0.619 |
| field_realistic | field_3_4 | E4_MONO_EXPONENTIAL | 42 | 42 | 0 | 21 | 21 | 0.476 | 1.000 | 0.524 |
| field_realistic | field_3_4 | H1 | 42 | 42 | 0 | 21 | 21 | 0.048 | 1.000 | 0.952 |
| field_realistic | field_4_5 | E0_DIRECT | 20 | 20 | 0 | 13 | 7 | 0.385 | 1.000 | 0.528 |
| field_realistic | field_4_5 | E1_OMBE | 20 | 20 | 0 | 13 | 7 | 0.154 | 0.857 | 0.681 |
| field_realistic | field_4_5 | E2_SPLINE | 20 | 3 | 17 | 13 | 7 | 0.769 | 1.000 | 0.174 |
| field_realistic | field_4_5 | E3_LINEAR_TAIL | 20 | 20 | 0 | 13 | 7 | 0.231 | 0.857 | 0.588 |
| field_realistic | field_4_5 | E4_MONO_EXPONENTIAL | 20 | 20 | 0 | 13 | 7 | 0.385 | 1.000 | 0.528 |
| field_realistic | field_4_5 | H1 | 20 | 20 | 0 | 13 | 7 | 0.154 | 1.000 | 0.794 |
| field_realistic | field_all | E0_DIRECT | 286 | 286 | 0 | 178 | 108 | 0.427 | 0.981 | 0.489 |
| field_realistic | field_all | E1_OMBE | 286 | 286 | 0 | 178 | 108 | 0.067 | 0.657 | 0.618 |
| field_realistic | field_all | E2_SPLINE | 286 | 104 | 182 | 178 | 108 | 0.629 | 0.852 | 0.190 |
| field_realistic | field_all | E3_LINEAR_TAIL | 286 | 286 | 0 | 178 | 108 | 0.056 | 0.509 | 0.492 |
| field_realistic | field_all | E4_MONO_EXPONENTIAL | 286 | 286 | 0 | 178 | 108 | 0.275 | 0.796 | 0.497 |
| field_realistic | field_all | H1 | 286 | 286 | 0 | 178 | 108 | 0.073 | 0.694 | 0.645 |
| field_realistic | field_gt_5 | E0_DIRECT | 119 | 119 | 0 | 90 | 29 | 0.111 | 0.966 | 0.773 |
| field_realistic | field_gt_5 | E1_OMBE | 119 | 119 | 0 | 90 | 29 | 0.067 | 0.897 | 0.802 |
| field_realistic | field_gt_5 | E2_SPLINE | 119 | 70 | 49 | 90 | 29 | 0.511 | 0.862 | 0.235 |
| field_realistic | field_gt_5 | E3_LINEAR_TAIL | 119 | 119 | 0 | 90 | 29 | 0.033 | 0.931 | 0.887 |
| field_realistic | field_gt_5 | E4_MONO_EXPONENTIAL | 119 | 119 | 0 | 90 | 29 | 0.189 | 0.966 | 0.654 |
| field_realistic | field_gt_5 | H1 | 119 | 119 | 0 | 90 | 29 | 0.078 | 0.966 | 0.830 |
| field_realistic | field_le_2 | E0_DIRECT | 63 | 63 | 0 | 28 | 35 | 0.964 | 0.971 | 0.008 |
| field_realistic | field_le_2 | E1_OMBE | 63 | 63 | 0 | 28 | 35 | 0.071 | 0.314 | 0.225 |
| field_realistic | field_le_2 | E2_SPLINE | 63 | 16 | 47 | 28 | 35 | 0.607 | 0.857 | 0.261 |
| field_realistic | field_le_2 | E3_LINEAR_TAIL | 63 | 63 | 0 | 28 | 35 | 0.036 | 0.086 | 0.045 |
| field_realistic | field_le_2 | E4_MONO_EXPONENTIAL | 63 | 63 | 0 | 28 | 35 | 0.214 | 0.571 | 0.346 |
| field_realistic | field_le_2 | H1 | 63 | 63 | 0 | 28 | 35 | 0.071 | 0.343 | 0.252 |

## Part 2 - Bootstrap Confidence Intervals

| dataset | endpoint | quantity | estimator | comparison | patients | obstructed | point | CI low | CI high | excludes zero |
|---|---|---|---|---|---|---|---|---|---|---|
| clean_cut | 2 | fn_rate | E0_DIRECT |  | 497 | 297 | 0.990 | 0.976 | 1.000 |  |
| clean_cut | 2 | fn_rate | E1_OMBE |  | 497 | 297 | 0.589 | 0.534 | 0.645 |  |
| clean_cut | 2 | fn_rate | H1 |  | 497 | 297 | 0.579 | 0.524 | 0.635 |  |
| clean_cut | 2 | fn_rate_difference |  | DIRECT_minus_OMBE | 497 | 297 | 0.400 | 0.344 | 0.456 | True |
| clean_cut | 2 | fn_rate_difference |  | DIRECT_minus_H1 | 497 | 297 | 0.410 | 0.355 | 0.465 | True |
| clean_cut | 3 | fn_rate | E0_DIRECT |  | 497 | 297 | 0.865 | 0.826 | 0.904 |  |
| clean_cut | 3 | fn_rate | E1_OMBE |  | 497 | 297 | 0.596 | 0.541 | 0.649 |  |
| clean_cut | 3 | fn_rate | H1 |  | 497 | 297 | 0.579 | 0.523 | 0.633 |  |
| clean_cut | 3 | fn_rate_difference |  | DIRECT_minus_OMBE | 497 | 297 | 0.270 | 0.221 | 0.318 | True |
| clean_cut | 3 | fn_rate_difference |  | DIRECT_minus_H1 | 497 | 297 | 0.287 | 0.238 | 0.338 | True |
| clean_cut | 4 | fn_rate | E0_DIRECT |  | 497 | 297 | 0.684 | 0.632 | 0.738 |  |
| clean_cut | 4 | fn_rate | E1_OMBE |  | 497 | 297 | 0.519 | 0.465 | 0.577 |  |
| clean_cut | 4 | fn_rate | H1 |  | 497 | 297 | 0.495 | 0.441 | 0.552 |  |
| clean_cut | 4 | fn_rate_difference |  | DIRECT_minus_OMBE | 497 | 297 | 0.165 | 0.125 | 0.209 | True |
| clean_cut | 4 | fn_rate_difference |  | DIRECT_minus_H1 | 497 | 297 | 0.189 | 0.144 | 0.234 | True |
| clean_cut | 5 | fn_rate | E0_DIRECT |  | 497 | 297 | 0.549 | 0.490 | 0.605 |  |
| clean_cut | 5 | fn_rate | E1_OMBE |  | 497 | 297 | 0.468 | 0.412 | 0.523 |  |
| clean_cut | 5 | fn_rate | H1 |  | 497 | 297 | 0.441 | 0.382 | 0.498 |  |
| clean_cut | 5 | fn_rate_difference |  | DIRECT_minus_OMBE | 497 | 297 | 0.080 | 0.050 | 0.115 | True |
| clean_cut | 5 | fn_rate_difference |  | DIRECT_minus_H1 | 497 | 297 | 0.107 | 0.073 | 0.146 | True |
| field_realistic | field_all | fn_rate | E0_DIRECT |  | 286 | 178 | 0.427 | 0.353 | 0.503 |  |
| field_realistic | field_all | fn_rate | E1_OMBE |  | 286 | 178 | 0.067 | 0.033 | 0.108 |  |
| field_realistic | field_all | fn_rate | H1 |  | 286 | 178 | 0.073 | 0.038 | 0.116 |  |
| field_realistic | field_all | fn_rate_difference |  | DIRECT_minus_OMBE | 286 | 178 | 0.359 | 0.290 | 0.432 | True |
| field_realistic | field_all | fn_rate_difference |  | DIRECT_minus_H1 | 286 | 178 | 0.354 | 0.283 | 0.429 | True |
| field_realistic | field_le_2 | fn_rate | E0_DIRECT |  | 63 | 28 | 0.964 | 0.879 | 1.000 |  |
| field_realistic | field_le_2 | fn_rate | E1_OMBE |  | 63 | 28 | 0.071 | 0.000 | 0.182 |  |
| field_realistic | field_le_2 | fn_rate | H1 |  | 63 | 28 | 0.071 | 0.000 | 0.182 |  |
| field_realistic | field_le_2 | fn_rate_difference |  | DIRECT_minus_OMBE | 63 | 28 | 0.894 | 0.769 | 1.000 | True |
| field_realistic | field_le_2 | fn_rate_difference |  | DIRECT_minus_H1 | 63 | 28 | 0.894 | 0.769 | 1.000 | True |
| field_realistic | field_2_3 | fn_rate | E0_DIRECT |  | 42 | 26 | 0.769 | 0.591 | 0.923 |  |
| field_realistic | field_2_3 | fn_rate | E1_OMBE |  | 42 | 26 | 0.038 | 0.000 | 0.125 |  |
| field_realistic | field_2_3 | fn_rate | H1 |  | 42 | 26 | 0.038 | 0.000 | 0.125 |  |
| field_realistic | field_2_3 | fn_rate_difference |  | DIRECT_minus_OMBE | 42 | 26 | 0.734 | 0.545 | 0.900 | True |
| field_realistic | field_2_3 | fn_rate_difference |  | DIRECT_minus_H1 | 42 | 26 | 0.734 | 0.545 | 0.900 | True |
| field_realistic | field_3_4 | fn_rate | E0_DIRECT |  | 42 | 21 | 0.667 | 0.450 | 0.864 |  |
| field_realistic | field_3_4 | fn_rate | E1_OMBE |  | 42 | 21 | 0.048 | 0.000 | 0.160 |  |
| field_realistic | field_3_4 | fn_rate | H1 |  | 42 | 21 | 0.048 | 0.000 | 0.160 |  |
| field_realistic | field_3_4 | fn_rate_difference |  | DIRECT_minus_OMBE | 42 | 21 | 0.617 | 0.400 | 0.826 | True |
| field_realistic | field_3_4 | fn_rate_difference |  | DIRECT_minus_H1 | 42 | 21 | 0.617 | 0.400 | 0.826 | True |
| field_realistic | field_4_5 | fn_rate | E0_DIRECT |  | 20 | 13 | 0.385 | 0.111 | 0.667 |  |
| field_realistic | field_4_5 | fn_rate | E1_OMBE |  | 20 | 13 | 0.154 | 0.000 | 0.385 |  |
| field_realistic | field_4_5 | fn_rate | H1 |  | 20 | 13 | 0.154 | 0.000 | 0.385 |  |
| field_realistic | field_4_5 | fn_rate_difference |  | DIRECT_minus_OMBE | 20 | 13 | 0.229 | 0.000 | 0.500 | False |
| field_realistic | field_4_5 | fn_rate_difference |  | DIRECT_minus_H1 | 20 | 13 | 0.229 | 0.000 | 0.500 | False |
| field_realistic | field_gt_5 | fn_rate | E0_DIRECT |  | 119 | 90 | 0.111 | 0.048 | 0.178 |  |
| field_realistic | field_gt_5 | fn_rate | E1_OMBE |  | 119 | 90 | 0.067 | 0.012 | 0.122 |  |
| field_realistic | field_gt_5 | fn_rate | H1 |  | 119 | 90 | 0.078 | 0.023 | 0.136 |  |
| field_realistic | field_gt_5 | fn_rate_difference |  | DIRECT_minus_OMBE | 119 | 90 | 0.044 | 0.011 | 0.089 | True |
| field_realistic | field_gt_5 | fn_rate_difference |  | DIRECT_minus_H1 | 119 | 90 | 0.033 | -0.011 | 0.081 | False |

## Part 3 - Recovered-Curve Failure-Mode Audit

| dataset | endpoint | method | group | n | true ratio median | severe+moderate n | severe+moderate frac | <0.50 n | 0.50-0.60 n | 0.60-0.65 n | 0.65-0.70 n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_cut | 2 | E1 | recovered | 119 | 0.585 | 72 | 0.605 | 14 | 58 | 24 | 23 |
| clean_cut | 2 | E1 | not_recovered | 175 | 0.664 | 22 | 0.126 | 0 | 22 | 36 | 117 |
| clean_cut | 2 | H1 | recovered | 122 | 0.585 | 75 | 0.615 | 14 | 61 | 24 | 23 |
| clean_cut | 2 | H1 | not_recovered | 172 | 0.666 | 19 | 0.110 | 0 | 19 | 36 | 117 |
| clean_cut | 3 | E1 | recovered | 80 | 0.601 | 40 | 0.500 | 3 | 37 | 21 | 19 |
| clean_cut | 3 | E1 | not_recovered | 177 | 0.667 | 18 | 0.102 | 0 | 18 | 38 | 121 |
| clean_cut | 3 | H1 | recovered | 85 | 0.597 | 44 | 0.518 | 3 | 41 | 22 | 19 |
| clean_cut | 3 | H1 | not_recovered | 172 | 0.669 | 14 | 0.081 | 0 | 14 | 37 | 121 |
| clean_cut | 4 | E1 | recovered | 49 | 0.628 | 16 | 0.327 | 0 | 16 | 17 | 16 |
| clean_cut | 4 | E1 | not_recovered | 154 | 0.672 | 8 | 0.052 | 0 | 8 | 30 | 116 |
| clean_cut | 4 | H1 | recovered | 56 | 0.628 | 19 | 0.339 | 0 | 19 | 20 | 17 |
| clean_cut | 4 | H1 | not_recovered | 147 | 0.674 | 5 | 0.034 | 0 | 5 | 27 | 115 |
| clean_cut | 5 | E1 | recovered | 25 | 0.642 | 5 | 0.200 | 0 | 5 | 12 | 8 |
| clean_cut | 5 | E1 | not_recovered | 138 | 0.679 | 2 | 0.014 | 0 | 2 | 22 | 114 |
| clean_cut | 5 | H1 | recovered | 33 | 0.647 | 5 | 0.152 | 0 | 5 | 16 | 12 |
| clean_cut | 5 | H1 | not_recovered | 130 | 0.679 | 2 | 0.015 | 0 | 2 | 18 | 110 |
| field_realistic | field_2_3 | E1 | recovered | 19 | 0.675 | 4 | 0.211 | 1 | 3 | 4 | 11 |
| field_realistic | field_2_3 | E1 | not_recovered | 1 | 0.663 | 0 | 0.000 | 0 | 0 | 0 | 1 |
| field_realistic | field_2_3 | H1 | recovered | 19 | 0.675 | 4 | 0.211 | 1 | 3 | 4 | 11 |
| field_realistic | field_2_3 | H1 | not_recovered | 1 | 0.663 | 0 | 0.000 | 0 | 0 | 0 | 1 |
| field_realistic | field_3_4 | E1 | recovered | 13 | 0.661 | 3 | 0.231 | 0 | 3 | 3 | 7 |
| field_realistic | field_3_4 | E1 | not_recovered | 1 | 0.585 | 1 | 1.000 | 0 | 1 | 0 | 0 |
| field_realistic | field_3_4 | H1 | recovered | 13 | 0.661 | 3 | 0.231 | 0 | 3 | 3 | 7 |
| field_realistic | field_3_4 | H1 | not_recovered | 1 | 0.585 | 1 | 1.000 | 0 | 1 | 0 | 0 |
| field_realistic | field_4_5 | E1 | recovered | 3 | 0.658 | 0 | 0.000 | 0 | 0 | 1 | 2 |
| field_realistic | field_4_5 | E1 | not_recovered | 2 | 0.606 | 1 | 0.500 | 0 | 1 | 1 | 0 |
| field_realistic | field_4_5 | H1 | recovered | 3 | 0.658 | 0 | 0.000 | 0 | 0 | 1 | 2 |
| field_realistic | field_4_5 | H1 | not_recovered | 2 | 0.606 | 1 | 0.500 | 0 | 1 | 1 | 0 |
| field_realistic | field_all | E1 | recovered | 64 | 0.661 | 12 | 0.188 | 1 | 11 | 15 | 37 |
| field_realistic | field_all | E1 | not_recovered | 12 | 0.671 | 2 | 0.167 | 0 | 2 | 1 | 9 |
| field_realistic | field_all | H1 | recovered | 64 | 0.661 | 12 | 0.188 | 1 | 11 | 15 | 37 |
| field_realistic | field_all | H1 | not_recovered | 12 | 0.671 | 2 | 0.167 | 0 | 2 | 1 | 9 |
| field_realistic | field_gt_5 | E1 | recovered | 4 | 0.679 | 0 | 0.000 | 0 | 0 | 0 | 4 |
| field_realistic | field_gt_5 | E1 | not_recovered | 6 | 0.689 | 0 | 0.000 | 0 | 0 | 0 | 6 |
| field_realistic | field_gt_5 | H1 | recovered | 4 | 0.679 | 0 | 0.000 | 0 | 0 | 0 | 4 |
| field_realistic | field_gt_5 | H1 | not_recovered | 6 | 0.689 | 0 | 0.000 | 0 | 0 | 0 | 6 |
| field_realistic | field_le_2 | E1 | recovered | 25 | 0.656 | 5 | 0.200 | 0 | 5 | 7 | 13 |
| field_realistic | field_le_2 | E1 | not_recovered | 2 | 0.670 | 0 | 0.000 | 0 | 0 | 0 | 2 |
| field_realistic | field_le_2 | H1 | recovered | 25 | 0.656 | 5 | 0.200 | 0 | 5 | 7 | 13 |
| field_realistic | field_le_2 | H1 | not_recovered | 2 | 0.670 | 0 | 0.000 | 0 | 0 | 0 | 2 |

## Part 4 - Limitation Ledger

| limitation | source |
|---|---|
| Cycle F historical gate; E and G analyzed separately in the three-cycle extension. | results/gate2_F_parse.md:3-5; results/gate3_F_ombe_rerun.md:3-5 |
| Gate 3 flow-domain adequacy is 0/31851; this is a genuine flow-shape adequacy failure. | results/gate3_F_ombe_rerun.md:26,62 |
| The flow-domain failure does not propagate to the selected volume-domain screening decision in the clinical gate. | results/gate_clinical_classification_F.md:3-5 |
| Repeatability remains UNSTABLE on ratio ICC: observed 0.793 vs model 0.778. | results/gate_reproducibility_volume_F.md:3,30-31,57 |
| Complete curves: observed beats model; H1 is for truncated use, not a complete-curve replacement. | results/gate_model_vs_observed_F.md:117; results/gate_offset_lungsize_F.md:13 |
| GLI degradation is unexplained; lung-size interaction was refuted and age tracks GLI LLN with r=-0.950. | results/gate_offset_lungsize_F.md:12,294-297 |
| Posterior inference is not usable for decisioning on this path: posterior calibration is NOT-CALIBRATED and S9 initialization is insufficient. | results/gate_posterior_calibration_repair_F.md:3-5; results/gate_s9_modeinit_F.md:11-18 |

## Verdict

VERDICT: TRUNCATION-QUALIFIED - recovery survives field-realistic truncation, but at least one robustness condition is weaker or denominator-limited.

- field_realistic DIRECT FN: 0.426966
- field_realistic E1 FN: 0.067416
- field_realistic H1 FN: 0.073034
- clean T<=3 difference CIs exclude zero: True
- field_all difference CIs exclude zero: True
- field recovery concentrates in severe/moderate: False
