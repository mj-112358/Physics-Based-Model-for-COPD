# Gate 2 - Cycle G Parse + FEV1 Recovery

- status: PASS
- reason: Gate 2 parser recovery criterion passed under max/best-test comparison.
- cycle: G
- parser callable: predictx.gate23_f.derive_curve
- raw source: <PROJECT_ROOT>/NHANES_SpiroSBI/2011-2012/spxraw_g.sas7bdat
- raw source note: using extracted SAS7BDAT from new nested root: <PROJECT_ROOT>/NHANES_SpiroSBI/2011-2012/spxraw_g.sas7bdat
- raw metadata rows: 37661
- raw loaded rows: 37661
- required columns present: SEQN, SPATTYPE, SPAMANU, SPAFACT, SPAPLAT, SPAACC, SPAQEFF, SPXPTS, SPXRAW
- parse errors: 60

## Row Counts

- raw_rows: 37661
- pre_rows: 35642
- acceptable_pre_rows: 32979
- A_pre_rows: 18395
- acceptable_A_pre_rows: 18395
- participants_with_truth: 6652
- participants_first_acceptable: 6652
- participants_first_acceptable_A: 6228

## Ground-Truth Comparisons

| comparison set | n | median signed FEV1 err | median abs FEV1 err | median signed FVC err | median abs FVC err | median abs ratio err | BEV flag frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| all acceptable A Pre curves | 18383 | -5.00% | 5.00% | -6.30% | 6.30% | 2.00% | 0.017 (306/18383; CI 0.015-0.018) |
| first acceptable A Pre per participant | 6228 | -6.06% | 6.06% | -7.03% | 7.03% | 2.16% | 0.025 (154/6228; CI 0.021-0.029) |
| first acceptable Pre per participant | 6652 | -9.38% | 9.40% | -10.10% | 10.14% | 3.64% | 0.084 (562/6652; CI 0.078-0.091) |
| max acceptable A Pre per participant | 6228 | -3.08% | 3.08% | -3.97% | 3.98% | 2.94% | 0.044 (272/6228; CI 0.039-0.049) |
| max acceptable Pre per participant | 6652 | -2.91% | 2.95% | -3.56% | 3.61% | 5.18% | 0.185 (1229/6652; CI 0.175-0.194) |

## Criterion

Criterion set: max acceptable (`SPAACC=='Y'`) pre-bronchodilator value per participant, matching the observed `SPX_G` best-test/composite behavior.
- criterion n: 6652
- median absolute FEV1 error: 2.95%
- median absolute FVC error: 3.61%
- pass threshold: both <= 5%
- diagnosis if failed: none; Gate 2 passed
- residual note: the remaining one-direction low bias is reported as composite/selection mismatch plus possible mild back-extrapolation clipping; no scaling correction was applied.

Parsing formula used exactly: comma-split SPXRAW to integer volume increments; `flow_Lps=incr*0.1*SPAFACT`; `volume_mL=cumsum(incr)*SPAFACT`; no differentiation.
