# Gate - Coverage-Accuracy-Safety Frontier for the Graded Obstruction Classifier, Cycle F

VERDICT: FRONTIER-UNSAFE - The TRAIN-selected CLEAN point fails at least one held-out TEST safety constraint.

## Protocol Decisions

- The same CLEAN p_adj was deterministically reconstructed because the prior gate did not persist TRAIN probabilities; execution halted unless TEST p_adj matched the saved participants CSV.
- Symmetric probability bands are centered at p=0.5; p<=p_low is confident normal and p>=p_high is confident obstructed.
- The TRAIN target coverage search measures coverage after applying the ratio<0.60 normal-call safety guard, because that is the deployable operating coverage.
- The operating point is selected only from TRAIN weighted PPV, NPV, severe-FN, coverage, and kappa. TEST metrics are reported once and never used to choose the point.
- All accuracy, kappa, sensitivity, specificity, PPV, and NPV values in the frontier use natural-prevalence class weights.
- The ratio<0.60 guard only removes confident normal calls; it is reported separately as a safety guard and not used to add confidence.
- The frontier target grid is fixed at 0.45 to 1.00 in approximately 0.02 increments before looking at TEST performance.
- The fit_alpha arm is secondary because fit_alpha was previously quarantined by ratio correlation, despite being mechanistically independent of the label computation.

## Constraint Fixes

- Severe safety guard: a case with ratio < 0.60 is never allowed to enter `CONFIDENT_NORMAL`; it is forced to `INDETERMINATE_GRADED` instead.
- This is a monotone safety guard, not a classifier feature: it only removes normal confidence and cannot add obstructed confidence.
- New asymmetric floors at natural prevalence: NPV >= 0.95 for `CONFIDENT_NORMAL`, PPV >= 0.70 for `CONFIDENT_OBSTRUCTED`, severe FN = 0.
- Old audit floor retained for comparison: PPV >= NPV >= 0.90.

## Part 1 - CLEAN Frontier Shape

- Deterministic operational baseline overlay: coverage 0.666, kappa 0.923, severe FN 0.
- Best CLEAN kappa on the swept frontier is only 0.674 at coverage 0.430; the deterministic baseline kappa remains far higher.
- CLEAN PPV is non-monotone at very low coverage; it passes 0.70 over coverage 0.359 to 0.936.
- CLEAN NPV is the limiting floor: its passing range ends at coverage 0.746, and the first swept point below 0.95 is 0.783.
- CLEAN kappa is below 0.80 throughout this swept grid; below-0.80 starts at coverage 0.359.
- The severe guard first forces TEST referrals at coverage 0.359.
- Severe FN is zero at every swept frontier point across both arms: True.

Selected rows from the CLEAN frontier:

| target | coverage | kappa | PPV | NPV | severe FN | knife-edge frac | new pass | old pass |
|---|---|---|---|---|---|---|---|---|
| 0.450 | 0.359 | 0.653 | 0.712 | 0.990 | 0 | 0.168 | True | False |
| 0.550 | 0.501 | 0.602 | 0.794 | 0.980 | 0 | 0.255 | True | False |
| 0.650 | 0.663 | 0.435 | 0.801 | 0.962 | 0 | 0.346 | True | False |
| 0.670 | 0.686 | 0.435 | 0.810 | 0.960 | 0 | 0.358 | True | False |
| 0.750 | 0.746 | 0.409 | 0.822 | 0.954 | 0 | 0.393 | True | False |
| 0.850 | 0.831 | 0.410 | 0.749 | 0.946 | 0 | 0.429 | False | False |
| 0.870 | 0.860 | 0.398 | 0.752 | 0.944 | 0 | 0.440 | False | False |
| 0.950 | 0.936 | 0.403 | 0.712 | 0.938 | 0 | 0.467 | False | False |
| 1.000 | 0.947 | 0.401 | 0.663 | 0.938 | 0 | 0.469 | False | False |

## Part 2 - Train-Selected Candidates and Held-Out Evaluation

| constraint | target | coverage | kappa | PPV | NPV | severe FN | p_low | p_high |
|---|---|---|---|---|---|---|---|---|
| CLEAN train-selected, old symmetric 0.90/0.90 | 0.970 | 0.947 | 0.401 | 0.663 | 0.938 | 0 | 0.500 | 0.500 |
| CLEAN train-selected, new asymmetric | 0.870 | 0.860 | 0.398 | 0.752 | 0.944 | 0 | 0.370 | 0.630 |
| fit_alpha train-selected, old symmetric 0.90/0.90 | 1.000 | 0.991 | 0.743 | 0.825 | 0.970 | 0 | 0.500 | 0.500 |
| fit_alpha train-selected, new asymmetric | 1.000 | 0.991 | 0.743 | 0.825 | 0.970 | 0 | 0.500 | 0.500 |

- CLEAN train-selected candidate coverage on TEST: 0.860 vs operational baseline 0.666; delta 0.194.
- Count accounting: 154 operational deferrals become confident, 45 operational-finalized cases become indeterminate, net delta 109 cases.
- CLEAN train-selected candidate severe FN on TEST: 0 with CP upper bound 0.369.
- CLEAN train-selected candidate PPV/NPV on TEST: 0.752/0.944; held-out safety pass: False.
- Source of gained confident calls from operational deferrals:

| bucket | gained n |
|---|---|
| KNIFE_EDGE | 74 |
| NEAR | 40 |
| RECOVERABLE | 40 |

## Part 3 - Held-Out Bucket Performance at the CLEAN Train-Selected Candidate

| bucket | true distance | n | conf finalized | fraction | accuracy | severe FN |
|---|---|---|---|---|---|---|
| KNIFE_EDGE | <0.02 | 84 | 74 | 0.881 | 0.573 | 0 |
| NEAR | 0.02-0.05 | 49 | 40 | 0.816 | 0.681 | 0 |
| RECOVERABLE | >=0.05 | 55 | 40 | 0.727 | 0.877 | 0 |

- The expected bucket pattern did not fully hold: `RECOVERABLE` calls are accurate, but the probability-only CLEAN frontier also finalizes many `KNIFE_EDGE` cases with poor bucket accuracy. That is why this is a coverage/safety frontier result, not a claim that the knife-edge was solved.

## Part 4 - fit_alpha Frontier

- fit_alpha whole AUC: 0.970; boundary AUC: 0.812.
- CLEAN whole AUC: 0.886; boundary AUC: 0.591.
- fit_alpha best kappa: 0.979 at coverage 0.456.
- fit_alpha PPV never falls below 0.70 on this grid; new-asymmetric passing coverage reaches 0.991.
- fit_alpha is reported as secondary because the prior audit found Spearman correlation with the ratio around 0.63; strict CLEAN remains the conservative headline arm.

## Interpretation

- The CLEAN candidate selected on TRAIN fails the held-out TEST safety floors and cannot be used as an operating point.
- The secondary fit_alpha candidate passes the held-out floors, but it remains exploratory because fit_alpha was quarantined by its correlation with the target ratio.
- fit_alpha count accounting: 184 operational deferrals become confident, 1 operational-finalized cases become indeterminate, net delta 183 cases.
- KNIFE_EDGE cases that remain indeterminate should be treated as correctly deferred; this frontier does not prove the knife-edge itself is solved.

VERDICT: FRONTIER-UNSAFE - The TRAIN-selected CLEAN point fails at least one held-out TEST safety constraint.
