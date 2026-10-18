# GLI LLN Diagnosis

Phase 1 only. No pipeline fix or recount was run after this diagnosis.

## Raw rspiro Reference Values

`rspiro` version: 0.5

| person | description | pred_GLI FEV1 (L) | pred_GLI FVC (L) | pred_GLI FEV1/FVC (ratio) | LLN_GLI FEV1 (L) | LLN_GLI FVC (L) | LLN_GLI FEV1/FVC (ratio) |
|---|---|---:|---:|---:|---:|---:|---:|
| A | 65yo male, 175cm, Caucasian | 3.282708 | 4.278042 | 0.769153 | 2.424725 | 3.219681 | 0.642040 |
| B | 25yo male, 175cm, Caucasian | 4.465684 | 5.326230 | 0.843754 | 3.599421 | 4.308388 | 0.728305 |
| C | 65yo female, 160cm, Caucasian | 2.276736 | 2.904094 | 0.788920 | 1.690699 | 2.161483 | 0.662560 |

## Assert Checks

| check | result | evidence |
|---|---|---|
| 65yo male FEV1/FVC ratio LLN should be below 0.70 and approximately 0.65-0.68 | FAIL on the narrow range, PASS on below 0.70 | value = 0.642040 |
| 25yo male ratio LLN should be higher than the 65yo male ratio LLN | PASS | 25yo = 0.728305; 65yo = 0.642040 |
| FEV1 volume LLN is distinct from FEV1/FVC ratio LLN | PASS | A FEV1 LLN = 2.424725 L; A ratio LLN = 0.642040 |

## Required Diagnosis Statements

(a) The FEV1/FVC ratio lower bound is `LLN_GLI(..., param = "FEV1FVC")`; it returns a unitless ratio, e.g. 0.642040 for Person A.

(b) The current `obstruction.py` comparison is `fev1_fvc_ratio < gli_lln_ratio`, and `gli_lln_ratio` is produced in `gli_reference.R` from `LLN_GLI(..., param = "FEV1FVC")`.

(c) No wrong-parameter, inverted-comparison, or unit-error mismatch is proven by Phase 1. The observed age direction follows the raw GLI values: the young adult ratio LLN is higher than the older adult ratio LLN, so LLN-vs-fixed is expected to add some younger cases above 0.70 and remove some older cases below 0.70.

## Stop Point

Hard stop after Phase 1. Do not run Phase 2 or Phase 3 from this diagnosis without an explicit decision on whether the expected age-split gate should be changed.
