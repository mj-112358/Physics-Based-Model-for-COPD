# Gate - Coverage-Accuracy-Safety Frontier for the Graded Obstruction Classifier, Cycle F

VERDICT: FRONTIER-GAIN - The asymmetric, guarded CLEAN frontier exceeds the 66.6% operational baseline while preserving severe FN=0.

## Decisions I Made

- Because the prior gate did not persist TRAIN probabilities, I deterministically reconstructed the same CLEAN p_adj and halted unless TEST p_adj matched the saved participants CSV.
- Symmetric probability bands are centered at p=0.5; p<=p_low is confident normal and p>=p_high is confident obstructed.
- The TRAIN target coverage search measures coverage after applying the ratio<0.60 normal-call safety guard, because that is the deployable operating coverage.
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
- Best CLEAN kappa on the swept frontier is only 0.621 at coverage 0.456; the deterministic baseline kappa remains far higher.
- CLEAN PPV is non-monotone at very low coverage; it passes 0.70 over coverage 0.430 to 0.941.
- CLEAN NPV is the limiting floor: its passing range ends at coverage 0.741, and the first swept point below 0.95 is 0.787.
- CLEAN kappa is below 0.80 throughout this swept grid; below-0.80 starts at coverage 0.364.
- The severe guard first forces TEST referrals at coverage 0.364.
- Severe FN is zero at every swept frontier point across both arms: True.

Selected rows from the CLEAN frontier:

| target | coverage | kappa | PPV | NPV | severe FN | knife-edge frac | new pass | old pass |
|---|---|---|---|---|---|---|---|---|
| 0.450 | 0.364 | 0.586 | 0.672 | 0.989 | 0 | 0.161 | False | False |
| 0.550 | 0.565 | 0.501 | 0.774 | 0.975 | 0 | 0.277 | True | False |
| 0.650 | 0.679 | 0.413 | 0.797 | 0.960 | 0 | 0.364 | True | False |
| 0.670 | 0.691 | 0.412 | 0.801 | 0.959 | 0 | 0.365 | True | False |
| 0.730 | 0.741 | 0.448 | 0.837 | 0.956 | 0 | 0.386 | True | False |
| 0.750 | 0.787 | 0.409 | 0.837 | 0.949 | 0 | 0.413 | False | False |
| 0.850 | 0.847 | 0.378 | 0.839 | 0.944 | 0 | 0.436 | False | False |
| 0.950 | 0.941 | 0.393 | 0.776 | 0.936 | 0 | 0.472 | False | False |
| 1.000 | 0.941 | 0.393 | 0.776 | 0.936 | 0 | 0.472 | False | False |

## Part 2 - Safe Operating Point

| constraint | target | coverage | kappa | PPV | NPV | severe FN | p_low | p_high |
|---|---|---|---|---|---|---|---|---|
| CLEAN old symmetric 0.90/0.90 | NA | 0.000 | NA | NA | NA | 0 | NA | NA |
| CLEAN new asymmetric | 0.730 | 0.741 | 0.448 | 0.837 | 0.956 | 0 | 0.265 | 0.735 |
| fit_alpha old symmetric 0.90/0.90 | 0.850 | 0.847 | 0.821 | 0.937 | 0.980 | 0 | 0.299 | 0.701 |
| fit_alpha new asymmetric | 0.990 | 0.989 | 0.772 | 0.864 | 0.972 | 0 | 0.487 | 0.513 |

- CLEAN safe point coverage: 0.741 vs operational baseline 0.666; delta 0.075.
- Count accounting: 123 operational deferrals become confident, 81 operational-finalized cases become indeterminate, net delta 42 cases.
- CLEAN safe point severe FN: 0 with CP upper bound 0.410.
- CLEAN safe point PPV/NPV: 0.837/0.956.
- Source of gained confident calls from operational deferrals:

| bucket | gained n |
|---|---|
| KNIFE_EDGE | 55 |
| NEAR | 34 |
| RECOVERABLE | 34 |

## Part 3 - Bucket-Resolved Recovery at CLEAN Safe Point

| bucket | true distance | n | conf finalized | fraction | accuracy | severe FN |
|---|---|---|---|---|---|---|
| KNIFE_EDGE | <0.02 | 84 | 55 | 0.655 | 0.585 | 0 |
| NEAR | 0.02-0.05 | 49 | 34 | 0.694 | 0.677 | 0 |
| RECOVERABLE | >=0.05 | 55 | 34 | 0.618 | 0.940 | 0 |

- The expected bucket pattern did not fully hold: `RECOVERABLE` calls are accurate, but the probability-only CLEAN frontier also finalizes many `KNIFE_EDGE` cases with poor bucket accuracy. That is why this is a coverage/safety frontier result, not a claim that the knife-edge was solved.

## Part 4 - fit_alpha Frontier

- fit_alpha whole AUC: 0.972; boundary AUC: 0.818.
- CLEAN whole AUC: 0.887; boundary AUC: 0.592.
- fit_alpha best kappa: 0.988 at coverage 0.501.
- fit_alpha PPV never falls below 0.70 on this grid; new-asymmetric passing coverage reaches 0.989.
- fit_alpha is reported as secondary because the prior audit found Spearman correlation with the ratio around 0.63; strict CLEAN remains the conservative headline arm.

## Interpretation

- The corrected asymmetric floors turn the calibrated probabilities into a safe coverage gain over the operational baseline.
- The fit_alpha secondary arm shifts safe coverage by 0.249 relative to CLEAN.
- fit_alpha count accounting: 183 operational deferrals become confident, 1 operational-finalized cases become indeterminate, net delta 182 cases.
- KNIFE_EDGE cases that remain indeterminate should be treated as correctly deferred; this frontier does not prove the knife-edge itself is solved.

VERDICT: FRONTIER-GAIN - The asymmetric, guarded CLEAN frontier exceeds the 66.6% operational baseline while preserving severe FN=0.
