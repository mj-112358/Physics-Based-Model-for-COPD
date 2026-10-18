# Gate 2 - Cycle F Parse + FEV1 Recovery

- status: PASS
- reason: Gate 2 parser recovery criterion passed under max/best-test comparison.
- cycle: F
- parser callable: predictx.gate23_f.derive_curve
- raw source: <PROJECT_ROOT>/NHANES_SpiroSBI/_extracted/spxraw_f.sas7bdat
- raw source note: extracted <PROJECT_ROOT>/NHANES_SpiroSBI/2009-2010/SPXRAW_F.zip -> <PROJECT_ROOT>/NHANES_SpiroSBI/_extracted/spxraw_f.sas7bdat
- raw metadata rows: 36421
- raw loaded rows: 36421
- required columns present: SEQN, SPATTYPE, SPAMANU, SPAFACT, SPAPLAT, SPAACC, SPAQEFF, SPXPTS, SPXRAW
- parse errors: 43

## Row Counts

- raw_rows: 36421
- pre_rows: 34413
- acceptable_pre_rows: 31881
- A_pre_rows: 18676
- acceptable_A_pre_rows: 18676
- participants_with_truth: 7079
- participants_first_acceptable: 7079
- participants_first_acceptable_A: 6638

## Ground-Truth Comparisons

| comparison set | n | median signed FEV1 err | median abs FEV1 err | median signed FVC err | median abs FVC err | median abs ratio err | BEV flag frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| all acceptable A Pre curves | 18617 | -4.73% | 4.73% | -5.88% | 5.89% | 1.81% | 0.014 (260/18617; CI 0.012-0.016) |
| first acceptable A Pre per participant | 6638 | -5.62% | 5.62% | -6.53% | 6.54% | 1.97% | 0.017 (111/6638; CI 0.014-0.020) |
| first acceptable Pre per participant | 7079 | -8.30% | 8.32% | -8.79% | 8.87% | 2.97% | 0.070 (495/7079; CI 0.064-0.076) |
| max acceptable A Pre per participant | 6638 | -3.02% | 3.02% | -3.82% | 3.82% | 2.57% | 0.034 (223/6638; CI 0.029-0.038) |
| max acceptable Pre per participant | 7079 | -2.89% | 2.91% | -3.49% | 3.53% | 4.34% | 0.151 (1071/7079; CI 0.143-0.160) |

## Criterion

Criterion set: max acceptable (`SPAACC=='Y'`) pre-bronchodilator value per participant, matching the observed `SPX_F` best-test/composite behavior.
- criterion n: 7079
- median absolute FEV1 error: 2.91%
- median absolute FVC error: 3.53%
- pass threshold: both <= 5%
- diagnosis if failed: none; Gate 2 passed
- residual note: the remaining one-direction low bias is reported as composite/selection mismatch plus possible mild back-extrapolation clipping; no scaling correction was applied.

Parsing formula used exactly: comma-split SPXRAW to integer volume increments; `flow_Lps=incr*0.1*SPAFACT`; `volume_mL=cumsum(incr)*SPAFACT`; no differentiation.
