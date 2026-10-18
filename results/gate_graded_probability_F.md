# Gate - Leakage-Free Calibrated Obstruction Classifier + Graded Probability, Cycle F

VERDICT: GRADED-UNCALIBRATED - Reweighted TEST ECE exceeds the conservative 0.10 calibration warning threshold.

## Protocol Decisions

- TEST split is scored once; all model fitting and zone boundary selection use TRAIN only.
- The requested true 0.60-0.70 band is one-class-only and cannot support AUC; the reported boundary AUC uses true FEV1/FVC in [0.65, 0.75), matching the prior around-threshold check.
- CLEAN candidate features are limited to the task-specified shape/fit-quality/repeatability fields plus the expected ratio-proxy fields needed for audit/quarantine.
- fit_alpha was expected to survive from prior notes, but its TRAIN Spearman correlation to observed ratio is above 0.60, so the leakage audit excludes it from CLEAN.
- gli_lln_ratio measured just below 0.60 but is excluded because the task explicitly identifies it as an expected ratio/threshold proxy that must not enter CLEAN.
- Median imputation values are learned on TRAIN only and applied to TEST.
- The empirical target prevalence is computed from results/merged_summary.parquet, then TRAIN constructed prevalence is used to form class reweighting factors.
- Zone boundaries maximize TRAIN confident coverage subject to severe FN=0 among confident normal and reweighted PPV/NPV >=0.90.
- Reweighted ECE > 0.10 is treated as uncalibrated for the verdict.
- Forced p>=0.5 deferred-case accuracy is reported only as a diagnostic; it is not retained as recovery because the TRAIN-frozen zones fail TEST safety/calibration floors.
- The RATIO-AWARE model is diagnostic only and is never used for graded output.

## Part 0 - Feature Assembly + Leakage Audit

- Banned columns present in source frame: fev1_fvc_ratio, fev1_mL, fvc_mL, parsed_fev1_l, parsed_fvc_l, parsed_ratio, truth_obstructed_fixed, truth_severity, workflow_obstructed_action, workflow_predicted_obstructed_when_final.
- Quarantined candidates: model_ratio, corrected_ratio_c1, corrected_ratio_c2, gli_lln_ratio, fet_s, duration_s, fit_alpha.
- CLEAN fitted features: volume_r2, volume_nrss_norm, fit_tau_f, fit_tau_s, fit_tau_o, dV_last_L, maneuver_count, model_ratio_sd, model_fev1_sd_l, model_fvc_sd_l, model_classification_majority_fraction.
- Leakage guard tripped: False.

Top audit rows by max absolute TRAIN correlation to observed ratio:

| feature | partition | pearson | spearman | max abs corr | train missing | test missing |
|---|---|---|---|---|---|---|
| model_ratio | QUARANTINED | 0.849 | 0.885 | 0.885 | 0.000 | 0.000 |
| corrected_ratio_c1 | QUARANTINED | 0.849 | 0.885 | 0.885 | 0.000 | 0.000 |
| corrected_ratio_c2 | QUARANTINED | 0.849 | 0.885 | 0.885 | 0.000 | 0.000 |
| fet_s | QUARANTINED | -0.704 | -0.704 | 0.704 | 0.000 | 0.000 |
| duration_s | QUARANTINED | -0.704 | -0.704 | 0.704 | 0.000 | 0.000 |
| fit_alpha | QUARANTINED | 0.560 | 0.631 | 0.631 | 0.000 | 0.000 |
| gli_lln_ratio | QUARANTINED | 0.598 | 0.590 | 0.598 | 0.005 | 0.002 |
| fit_tau_s | CLEAN | -0.543 | -0.533 | 0.543 | 0.000 | 0.000 |
| dV_last_L | CLEAN | -0.165 | -0.333 | 0.333 | 0.000 | 0.000 |
| fit_tau_f | CLEAN | -0.283 | -0.312 | 0.312 | 0.000 | 0.000 |
| fit_tau_o | CLEAN | 0.045 | 0.248 | 0.248 | 0.000 | 0.000 |
| model_fvc_sd_l | CLEAN | -0.085 | -0.142 | 0.142 | 0.032 | 0.041 |
| volume_r2 | CLEAN | -0.065 | 0.105 | 0.105 | 0.000 | 0.000 |
| volume_nrss_norm | CLEAN | 0.093 | -0.082 | 0.093 | 0.002 | 0.004 |
| model_fev1_sd_l | CLEAN | 0.085 | 0.049 | 0.085 | 0.032 | 0.041 |
| model_ratio_sd | CLEAN | 0.047 | -0.016 | 0.047 | 0.032 | 0.041 |
| maneuver_count | CLEAN | 0.030 | 0.038 | 0.038 | 0.032 | 0.041 |
| model_classification_majority_fraction | CLEAN | -0.003 | -0.032 | 0.032 | 0.032 | 0.041 |

Missingness/imputation for CLEAN features:

| feature | train missing | test missing | train median | >20% missing |
|---|---|---|---|---|
| volume_r2 | 0.000 | 0.000 | 0.998 | False |
| volume_nrss_norm | 0.002 | 0.004 | 1742.704 | False |
| fit_tau_f | 0.000 | 0.000 | 0.317 | False |
| fit_tau_s | 0.000 | 0.000 | 3.081 | False |
| fit_tau_o | 0.000 | 0.000 | 0.030 | False |
| dV_last_L | 0.000 | 0.000 | -0.008 | False |
| maneuver_count | 0.032 | 0.041 | 3.000 | False |
| model_ratio_sd | 0.032 | 0.041 | 0.015 | False |
| model_fev1_sd_l | 0.032 | 0.041 | 0.086 | False |
| model_fvc_sd_l | 0.032 | 0.041 | 0.078 | False |
| model_classification_majority_fraction | 0.032 | 0.041 | 1.000 | False |

## Part 1 - Calibrated CLEAN Classifier

| model | whole AUC | whole CI low | whole CI high | boundary AUC | boundary CI low | boundary CI high | boundary n |
|---|---|---|---|---|---|---|---|
| CLEAN GBT | 0.887 | 0.859 | 0.915 | 0.592 | 0.486 | 0.705 | 190 |
| CLEAN logistic | 0.841 | 0.808 | 0.874 | 0.592 | 0.472 | 0.699 | 190 |
| RATIO-AWARE reference | 0.986 | 0.978 | 0.993 | 0.919 | 0.873 | 0.957 | 190 |

- Boundary band definition: true fev1_fvc_ratio in [0.65, 0.75); true [0.60, 0.70) is one-class-only and AUC-undefined.
- Forced p>=0.5 accuracy on strict workflow-deferred TEST cases: 0.792 (244/308).
- Forced p>=0.5 accuracy on operational residual-deferred TEST cases: 0.739 (139/188).
- RATIO-AWARE is diagnostic only; it includes the same ratio measurement being questioned at the boundary and is not used for the graded output.

Permutation feature importance for CLEAN GBT:

| feature | mean AUC drop | sd |
|---|---|---|
| fit_tau_s | 0.209 | 0.018 |
| fit_tau_f | 0.022 | 0.005 |
| volume_nrss_norm | 0.022 | 0.005 |
| model_classification_majority_fraction | 0.022 | 0.004 |
| dV_last_L | 0.012 | 0.005 |
| fit_tau_o | 0.009 | 0.003 |
| model_fvc_sd_l | 0.008 | 0.003 |
| model_ratio_sd | 0.002 | 0.002 |
| volume_r2 | 0.001 | 0.002 |
| model_fev1_sd_l | 0.000 | 0.003 |

## Part 2 - Natural-Prevalence Reweighting

- Prevalence path: empirical_full_unenriched_cohort.
- Target prevalence: 0.096 (1932/20050 valid ratios).
- Constructed TRAIN prevalence used for weights: 0.555.

| scope | n | ECE | Brier |
|---|---|---|---|
| TEST raw | 563 | 0.040 | 0.133 |
| TEST reweighted | 563 | 0.226 | 0.150 |
| indeterminate raw | 563 | 0.040 | 0.133 |
| indeterminate reweighted | 563 | 0.226 | 0.150 |

Prevalence sensitivity sweep for the TRAIN-frozen candidate boundaries before rejection:

| target prev | AUC | PPV | NPV | ECE | Brier |
|---|---|---|---|---|---|
| 0.050 | 0.887 | 0.227 | 0.984 | 0.250 | 0.152 |
| 0.100 | 0.887 | 0.383 | 0.967 | 0.224 | 0.150 |
| 0.150 | 0.887 | 0.496 | 0.948 | 0.198 | 0.148 |
| 0.200 | 0.887 | 0.583 | 0.928 | 0.171 | 0.146 |

## Part 3 - Graded Three-Zone Output Policy

- TRAIN-frozen boundaries: p_low=0.661, p_high=0.665; status=FEASIBLE.
- Candidate zone rejected on TEST: True.
- Candidate rejection reasons: candidate confident-normal zone has severe false negatives on TEST; candidate confident-obstructed PPV is below the reweighted 0.90 floor on TEST.

TRAIN-frozen candidate zones before rejection:

| scope | confident n | conf coverage | kappa | PPV rw | NPV rw | severe conf | severe FN | severe FN CP high |
|---|---|---|---|---|---|---|---|---|
| candidate zones | 561 | 0.996 | 0.589 | 0.373 | 0.968 | 16 | 2 | 0.383 |

Retained graded output after enforcing the safety/floor guard:

| zone | n | fraction |
|---|---|---|
| INDETERMINATE_GRADED | 563 | 1.000 |

| scope | confident n | conf coverage | accuracy | sens | spec | kappa | PPV rw | NPV rw | severe conf | severe FN | severe FN CP high |
|---|---|---|---|---|---|---|---|---|---|---|---|
| graded confident zones | 0 | 0.000 | NA | NA | NA | NA | NA | NA | 0 | 0 | NA |

- GRADED coverage is 100% because every TEST patient receives p_obs and a zone. Retained confident coverage is 0.000; the remaining 1.000 receive probability-only indeterminate output.

## Part 4 - Confident Coverage vs Deterministic Baseline

| scope | coverage | kappa | severe FN | severe FN CP high |
|---|---|---|---|---|
| strict deterministic final-call | 0.453 | 0.842 | 0 | 0.842 |
| operational recovery baseline | 0.666 | 0.923 | 0 | 0.459 |
| graded confident zones | 0.000 | NA | 0 | NA |

## Part 5 - Orthogonal Demographic Lift Test

| feature | available | deferred | fraction |
|---|---|---|---|
| RIDAGEYR | 188 | 188 | 1.000 |
| BMXHT | 188 | 188 | 1.000 |
| RIAGENDR | 188 | 188 | 1.000 |
| RIDRETH1 | 188 | 188 | 1.000 |
| ever_smoker | 188 | 188 | 1.000 |

| model | AUC | boundary AUC | boundary n |
|---|---|---|---|
| clean | 0.887 | 0.592 | 190 |
| clean+demographics | 0.908 | 0.585 | 190 |
| demographics-only deferred | 0.734 | NA | NA |

- Clean boundary AUC: 0.592.
- Clean+demographics boundary AUC: 0.585.
- Demographics-only AUC on operational deferred TEST cases: 0.734.
- Demographic lift verdict: DEMOGRAPHICS-NEUTRAL.

## Part 6 - Knife-Edge vs Recoverable Decomposition

| bucket | n | fraction deferred | graded confident | graded fraction |
|---|---|---|---|---|
| true_abs_lt_0.02 | 84 | 0.447 | 0 | 0.000 |
| true_abs_0.02_to_0.05 | 49 | 0.261 | 0 | 0.000 |
| true_abs_ge_0.05 | 55 | 0.293 | 0 | 0.000 |

Knife-edge cases are not counted as a model failure; the fixed 0.70 threshold cuts through a continuum under measurement noise.

VERDICT: GRADED-UNCALIBRATED - Reweighted TEST ECE exceeds the conservative 0.10 calibration warning threshold.
