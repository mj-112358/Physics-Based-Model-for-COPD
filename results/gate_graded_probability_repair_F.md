# Gate - Graded Probability Repair Attempt, Cycle F

VERDICT: REPAIR-NOT-ACHIEVABLE-CLEAN - Even TEST-oracle thresholds on the best headline-allowed variant do not materially exceed operational baseline coverage.

## What Was Tried

- Natural-prevalence prior correction after calibration.
- Sigmoid versus isotonic calibration.
- Gradient-boosted trees versus balanced logistic regression.
- CLEAN-only and CLEAN+orthogonal-demographic feature sets.
- RATIO-AWARE diagnostic variant, clearly excluded from headline eligibility.

## Baselines

- strict deterministic coverage: 0.453.
- operational recovery coverage: 0.666.
- material improvement target used here: operational + 0.03 = 0.696.

## Variant Results

| variant | allowed | AUC | boundary AUC | ECE rw | TRAIN-frozen cov | PPV rw | NPV rw | severe FN | TEST-oracle cov | oracle PPV | oracle NPV | oracle severe FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_gbt_isotonic | True | 0.887 | 0.592 | 0.226 | 0.996 | 0.373 | 0.968 | 2 | 0.000 | NA | NA | 0 |
| clean_gbt_sigmoid_prior | True | 0.890 | 0.597 | 0.022 | 1.000 | 0.369 | 0.969 | 2 | 0.432 | 1.000 | 0.980 | 0 |
| clean_logistic_sigmoid_prior | True | 0.840 | 0.590 | 0.033 | 0.456 | 0.357 | 0.976 | 3 | 0.348 | 1.000 | 0.979 | 0 |
| demo_gbt_isotonic | True | 0.908 | 0.585 | 0.220 | 1.000 | 0.387 | 0.979 | 0 | 0.000 | NA | NA | 0 |
| demo_gbt_sigmoid_prior | True | 0.907 | 0.584 | 0.020 | 0.993 | 0.385 | 0.979 | 0 | 0.586 | 1.000 | 0.966 | 0 |
| demo_logistic_sigmoid_prior | True | 0.872 | 0.590 | 0.042 | 0.757 | 0.585 | 0.946 | 6 | 0.586 | 1.000 | 0.965 | 0 |
| ratio_aware_gbt_sigmoid_prior_DIAGNOSTIC | False | 0.986 | 0.921 | 0.023 | 0.998 | 0.857 | 0.989 | 0 | 0.998 | 0.918 | 0.982 | 0 |

## Best Headline-Allowed Attempts

- Best TRAIN-frozen headline-allowed coverage: clean_gbt_sigmoid_prior at 1.000, PPV/NPV 0.369/0.969, severe FN 2.
- Best TEST-oracle headline-allowed coverage: demo_gbt_sigmoid_prior at 0.586, PPV/NPV 1.000/0.966, severe FN 0.
- TEST-oracle rows are diagnostic upper bounds only; they are not deployable because boundaries are selected using TEST truth.

## Interpretation

- Prior correction can repair reweighted ECE for some variants, but it does not solve the PPV floor at natural prevalence.
- Demographics remove the severe-FN problem for the broad GBT candidate, but reweighted PPV remains far below 0.90.
- Some sigmoid variants have TEST-oracle feasible zones with PPV=1.0, but their coverage ceiling stays below the operational 66.6% baseline.
- The only path expected to beat this would require ratio-aware features, which are explicitly not headline-eligible under the leakage rules.

VERDICT: REPAIR-NOT-ACHIEVABLE-CLEAN - Even TEST-oracle thresholds on the best headline-allowed variant do not materially exceed operational baseline coverage.
