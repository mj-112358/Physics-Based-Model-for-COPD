# Gate - Severe-Truncation Diagnosis, Cycle F

> Evidence status: historical and provisional. This report depends on the
> legacy truncation-hardening artifact and must be regenerated under the
> fixed-horizon protocol before its estimates are cited.

## Protocol Decisions

- Severity uses complete-curve true FEV1/FVC: severe <0.50, moderate 0.50-0.60, mild 0.60-0.70, normal >=0.70.
- Field-realistic incomplete maneuver rule is inherited from the hardening gate: SPAQEFF in {C,D} OR SPAACC != Y OR SPAPLAT != Y.
- The paired severe cohort is defined before looking at recovery: complete plateau severe target plus a separate same-patient flagged/incomplete Pre maneuver.
- Because the paired severe cohort is under 20, Part 1 stops at the count finding and does not run a severe-specific recovery benchmark.
- Deterministic only; no NUTS, TSNPE, sampler, gamma-onset, registration, or ML residual work.

## Part 0 - Count The Severe Population First

### Clean-Cut Set

| n total | severe | moderate | mild | normal | obstructed |
|---|---|---|---|---|---|
| 497 | 17 | 80 | 200 | 200 | 297 |

Clean-cut obstructed subset:

| n total | severe | moderate | mild | normal | obstructed |
|---|---|---|---|---|---|
| 297 | 17 | 80 | 200 | 0 | 297 |

### Field-Realistic Set

| n total | severe | moderate | mild | normal | obstructed |
|---|---|---|---|---|---|
| 286 | 13 | 55 | 110 | 108 | 178 |

Field-realistic counts by endpoint bucket:

| endpoint bucket | n total | severe | moderate | mild | normal | obstructed |
|---|---|---|---|---|---|---|
| field_2_3 | 42 | 1 | 6 | 19 | 16 | 26 |
| field_3_4 | 42 | 1 | 6 | 14 | 21 | 21 |
| field_4_5 | 20 | 0 | 4 | 9 | 7 | 13 |
| field_gt_5 | 119 | 11 | 33 | 46 | 29 | 90 |
| field_le_2 | 63 | 0 | 6 | 22 | 35 | 28 |

Field-realistic true-severe and effective end <=3s count: 1.

No severity conclusion can be drawn from the field-realistic severe+short cell; n < 10, so the 18.8% recovered-case severe+moderate figure is denominator-limited, not a method finding.

## Part 1 - The Correct Severe Cohort

Paired severe cohort size: 13.

Cycle F does not contain enough severe patients with paired incomplete maneuvers to power a severe-recovery claim; the count is the finding.

The cohort rows are written to `results/gate_severe_truncation_diagnosis_F_cohort.csv`.

## Part 2 - Severity-Resolved Recovery Curve

| estimator | T | severity | n | FN n | FN rate | n>=10 |
|---|---|---|---|---|---|---|
| E1_OMBE | 2 | severe | 17 | 0 | 0.000 | True |
| E1_OMBE | 2 | moderate | 80 | 22 | 0.275 | True |
| E1_OMBE | 2 | mild_0.60_0.70 | 200 | 153 | 0.765 | True |
| E1_OMBE | 3 | severe | 17 | 0 | 0.000 | True |
| E1_OMBE | 3 | moderate | 80 | 18 | 0.225 | True |
| E1_OMBE | 3 | mild_0.60_0.70 | 200 | 159 | 0.795 | True |
| E1_OMBE | 4 | severe | 17 | 0 | 0.000 | True |
| E1_OMBE | 4 | moderate | 80 | 8 | 0.100 | True |
| E1_OMBE | 4 | mild_0.60_0.70 | 200 | 146 | 0.730 | True |
| E1_OMBE | 5 | severe | 17 | 0 | 0.000 | True |
| E1_OMBE | 5 | moderate | 80 | 2 | 0.025 | True |
| E1_OMBE | 5 | mild_0.60_0.70 | 200 | 137 | 0.685 | True |
| H1 | 2 | severe | 17 | 0 | 0.000 | True |
| H1 | 2 | moderate | 80 | 19 | 0.237 | True |
| H1 | 2 | mild_0.60_0.70 | 200 | 153 | 0.765 | True |
| H1 | 3 | severe | 17 | 0 | 0.000 | True |
| H1 | 3 | moderate | 80 | 14 | 0.175 | True |
| H1 | 3 | mild_0.60_0.70 | 200 | 158 | 0.790 | True |
| H1 | 4 | severe | 17 | 0 | 0.000 | True |
| H1 | 4 | moderate | 80 | 5 | 0.062 | True |
| H1 | 4 | mild_0.60_0.70 | 200 | 142 | 0.710 | True |
| H1 | 5 | severe | 17 | 0 | 0.000 | True |
| H1 | 5 | moderate | 80 | 2 | 0.025 | True |
| H1 | 5 | mild_0.60_0.70 | 200 | 129 | 0.645 | True |

Clean-cut severe-band recovery is adequate by the task threshold: E1 and H1 severe FN is 0.000 at T=2,3,4,5 with n=17.

## Part 3 - Honest Verdict

VERDICT: DENOMINATOR-LIMITED

field severe+short n < 10 and clean-cut severe-band E1/H1 FN is not high where severe n is adequate.

Claim becomes: "recovery holds across severity where severe cases are present; the field-realistic severe+short population is too small to assess."

- field severe+short n: 1
- paired severe cohort n: 13
- max clean severe FN rate across E1/H1/T: 0.000

### Paired Severe Cohort Preview

| SEQN | true ratio | field end s | bucket | SPAQEFF | SPAACC | SPAPLAT |
|---|---|---|---|---|---|---|
| 57217 | 0.456 | 2.938 | field_2_3 | C | Y | N |
| 55915 | 0.485 | 3.885 | field_3_4 | D | N | N |
| 61609 | 0.450 | 5.744 | field_gt_5 | D | N | N |
| 59073 | 0.389 | 8.854 | field_gt_5 | D | N | Y |
| 51995 | 0.272 | 9.747 | field_gt_5 | C | Y | N |
| 60345 | 0.500 | 13.765 | field_gt_5 | C | Y | N |
| 54834 | 0.371 | 15.439 | field_gt_5 | C | Y | N |
| 56952 | 0.367 | 17.492 | field_gt_5 | C | Y | N |
| 51893 | 0.460 | 17.761 | field_gt_5 | C | Y | N |
| 59547 | 0.467 | 18.542 | field_gt_5 | C | Y | N |
| 60124 | 0.492 | 18.692 | field_gt_5 | C | Y | N |
| 58714 | 0.469 | 19.974 | field_gt_5 | C | Y | N |
| 60119 | 0.499 | 19.985 | field_gt_5 | C | Y | N |
