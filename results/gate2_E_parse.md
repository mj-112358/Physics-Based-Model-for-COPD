# Gate 2 - Cycle E Parse + FEV1 Recovery

- status: PASS
- reason: Gate 2 parser recovery criterion passed under max/best-test comparison.
- cycle: E
- parser callable: predictx.gate23_f.derive_curve
- raw source: <PROJECT_ROOT>/NHANES_SpiroSBI/2007-2008/spxraw_e.sas7bdat
- raw source note: using extracted SAS7BDAT from new nested root: <PROJECT_ROOT>/NHANES_SpiroSBI/2007-2008/spxraw_e.sas7bdat
- raw metadata rows: 34857
- raw loaded rows: 34857
- required columns present: SEQN, SPATTYPE, SPAMANU, SPAFACT, SPAPLAT, SPAACC, SPAQEFF, SPXPTS, SPXRAW
- parse errors: 26

## Row Counts

- raw_rows: 34857
- pre_rows: 32353
- acceptable_pre_rows: 29574
- A_pre_rows: 16391
- acceptable_A_pre_rows: 16391
- participants_with_truth: 6319
- participants_first_acceptable: 6319
- participants_first_acceptable_A: 5813

## Ground-Truth Comparisons

| comparison set | n | median signed FEV1 err | median abs FEV1 err | median signed FVC err | median abs FVC err | median abs ratio err | BEV flag frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| all acceptable A Pre curves | 16348 | -4.99% | 4.99% | -6.07% | 6.07% | NA | 0.013 (209/16348; CI 0.011-0.015) |
| first acceptable A Pre per participant | 5813 | -5.86% | 5.87% | -6.69% | 6.71% | NA | 0.018 (104/5813; CI 0.014-0.021) |
| first acceptable Pre per participant | 6319 | -8.65% | 8.66% | -9.34% | 9.39% | NA | 0.075 (472/6319; CI 0.069-0.081) |
| max acceptable A Pre per participant | 5813 | -3.29% | 3.29% | -4.05% | 4.06% | NA | 0.031 (181/5813; CI 0.027-0.036) |
| max acceptable Pre per participant | 6319 | -3.13% | 3.14% | -3.66% | 3.72% | NA | 0.164 (1038/6319; CI 0.156-0.173) |

## Criterion

Criterion set: max acceptable (`SPAACC=='Y'`) pre-bronchodilator value per participant, matching the observed `SPX_E` best-test/composite behavior.
- criterion n: 6319
- median absolute FEV1 error: 3.14%
- median absolute FVC error: 3.72%
- pass threshold: both <= 5%
- diagnosis if failed: none; Gate 2 passed
- residual note: the remaining one-direction low bias is reported as composite/selection mismatch plus possible mild back-extrapolation clipping; no scaling correction was applied.

Parsing formula used exactly: comma-split SPXRAW to integer volume increments; `flow_Lps=incr*0.1*SPAFACT`; `volume_mL=cumsum(incr)*SPAFACT`; no differentiation.
