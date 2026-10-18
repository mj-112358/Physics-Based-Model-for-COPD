# Ground-Truth Reconciliation + Truncation Sanity + Coverage/Reproducibility Audit, Cycle F

- Deterministic only: no NUTS, SNPE, gamma onset, registration, or ML residual.
- Random seed: 20260708
- Raw source: <PROJECT_ROOT>/NHANES_SpiroSBI/2009-2010/spxraw_f.sas7bdat

## Verdicts

- Part A: TRUTH-CONFIRMED - truncation TRUE fields match the ATS observed-curve definition and the headline survives.
- Part B: NOISE - Paired bootstrap CI for E1 T=3 minus T=2 false-negative rate includes zero.
- Part C: The system produces a confident call for 88.4% of severe+moderate obstructed participants and recommends repeat/FLAG for 11.6%, with residual confident false-negative rate 1.6% over the full severe+moderate denominator.
- Part D: UNSTABLE - all-manoeuvre OMBE ratio ICC 0.721, borderline unanimous fraction 0.607.

## Part A - Code Paths

| gate/path | t0 | FEV1 | FVC | grid |
|---|---|---|---|---|
| clinical observed parser | ATS back-extrapolated time-zero from gate23_f.backextrap_t0 | np.interp(min(1.0, vol_t[-1]), observed vol_t, observed vol_l) | observed segmented end-volume vol_l[-1] | native 100 Hz samples after detected start/end, with inserted t=0 if needed |
| clinical reported model | same observed ATS back-extrapolated t0 before fitting | np.interp(min(1.0, t[-1]), fitted OMBE volume, native observed grid) | fitted OMBE volume at observed end-of-test, not asymptote beyond the grid | native observed volume grid |
| truncation TRUE | same ATS back-extrapolated time-zero | np.interp(1.0, full observed vol_t, observed vol_l) | observed segmented end-volume full_y[-1] | native observed volume grid; truncated estimators see samples only up to T |

## Part A - Common-Curve Comparison

- Common curves available: 78
- Common curves reported in row CSV: 30

The per-curve table is in `results/gate_reconciliation_audit_F_rows.csv` with `row_type == part_a_common_curve`.

## Part A - Gap Attribution

| comparison | n | median abs | IQR |
|---|---:|---:|---:|
| clinical model FEV1 vs NHANES, obstructed | 114 | 1.376% | 0.615 to 3.315% |
| clinical observed FEV1 vs NHANES, obstructed | 114 | 2.762% | 1.951 to 4.549% |
| truncation observed FEV1 vs NHANES, sampled | 497 | 3.214% | 2.407 to 4.639% |
| clinical model FVC vs NHANES, obstructed | 114 | 5.779% | 3.500 to 8.162% |
| clinical observed FVC vs NHANES, obstructed | 114 | 3.661% | 2.189 to 5.455% |
| truncation observed FVC vs NHANES, sampled | 497 | 3.841% | 2.758 to 5.809% |

- Dominant attribution: definition: clinical headline is fitted-model vs NHANES, while truncation TRUE is observed-curve vs NHANES.
- Nearest-sample versus interpolation FEV1 difference on common curves: median 0.289 mL.
- Max recomputed TRUE delta: FEV1 0.000000 mL, FVC 0.000000 mL, ratio 0.00000000.

### Headline Recompute

| estimator | T | old FN | recomputed FN | delta |
|---|---|---|---|---|
| E0_DIRECT | 2 | 0.990 | 0.990 | 0.0000 |
| E1_OMBE | 2 | 0.589 | 0.589 | 0.0000 |

## Part B - Non-Monotonicity

### Obstructed Sample Counts

| truth stratum | n |
|---|---|
| mild_0.60_0.70 | 200 |
| moderate | 80 |
| severe | 17 |

### E1 Obstructed False-Negative Bootstrap

| T | n | FN rate | 95% CI low | 95% CI high |
|---|---|---|---|---|
| 2 | 297 | 0.589 | 0.532 | 0.646 |
| 3 | 297 | 0.596 | 0.542 | 0.653 |
| 4 | 297 | 0.519 | 0.465 | 0.576 |
| 5 | 297 | 0.468 | 0.414 | 0.525 |

- Paired T=3 minus T=2 FN difference: 0.0067 (95% CI -0.0236 to 0.0404).

### E1 Bound Counts

| T | parameter | fit success n | low | high | bound frac |
|---|---|---|---|---|---|
| 2 | tau_f | 497 | 147 | 1 | 0.298 |
| 2 | tau_s | 497 | 214 | 62 | 0.555 |
| 2 | alpha | 497 | 45 | 36 | 0.163 |
| 2 | fvc_l | 497 | 53 | 7 | 0.121 |
| 2 | tau_o | 497 | 421 | 2 | 0.851 |
| 3 | tau_f | 497 | 126 | 3 | 0.260 |
| 3 | tau_s | 497 | 176 | 45 | 0.445 |
| 3 | alpha | 497 | 34 | 41 | 0.151 |
| 3 | fvc_l | 497 | 50 | 4 | 0.109 |
| 3 | tau_o | 497 | 419 | 3 | 0.849 |
| 4 | tau_f | 497 | 93 | 3 | 0.193 |
| 4 | tau_s | 497 | 109 | 29 | 0.278 |
| 4 | alpha | 497 | 22 | 42 | 0.129 |
| 4 | fvc_l | 497 | 46 | 5 | 0.103 |
| 4 | tau_o | 497 | 421 | 5 | 0.857 |
| 5 | tau_f | 497 | 63 | 3 | 0.133 |
| 5 | tau_s | 497 | 73 | 26 | 0.199 |
| 5 | alpha | 497 | 16 | 45 | 0.123 |
| 5 | fvc_l | 497 | 41 | 5 | 0.093 |
| 5 | tau_o | 497 | 424 | 6 | 0.865 |

### E1 FVC Error Trend

| T | n | signed median % | abs median % |
|---|---|---|---|
| 2 | 297 | -11.259 | 12.297 |
| 3 | 297 | -11.456 | 11.953 |
| 4 | 297 | -9.915 | 10.663 |
| 5 | 297 | -8.088 | 8.652 |
| full | 297 | -0.160 | 0.707 |

## Part C - Drops as FLAG

- Clinical severe+moderate denominator: 129
- Clinical gate drops classified: 15
- Drop taxonomy counts: {'FIT-QUALITY': 15}
- Parse bugs fixed: 0
- Policy rows available inside the severe+moderate denominator: 114; policy rows outside that denominator: 11; policy-missing counted as FLAG: 15
- Full-denominator policy outcome counts: {'OBSTRUCTED': 112, 'NORMAL': 2, 'FLAG': 15}
- The system produces a confident call for 88.4% of severe+moderate obstructed participants and recommends repeat/FLAG for 11.6%, with residual confident false-negative rate 1.6% over the full severe+moderate denominator.

Per-participant drop taxonomy rows are in the row CSV with `row_type == part_c_clinical_drop_taxonomy`.

## Part D - Cross-Manoeuvre Reproducibility

- Limitation: Part D reuses the existing all-manoeuvre deterministic flow-domain OMBE fit table. It is not a fresh all-manoeuvre volume-domain refit.
- Participants with >=2 acceptable A/B manoeuvres: 6677
- Manoeuvre-count distribution: {'2': 269, '3': 3959, '4': 1475, '5': 553, '6': 266, '7': 107, '8': 48}
- Model fit manoeuvre rows used: 23809

| metric | median within-participant SD | IQR |
|---|---:|---:|
| model FEV1 (L) | 0.0828 | 0.0483 to 0.1425 |
| model FVC (L) | 0.0875 | 0.0440 to 0.1450 |
| model FEV1/FVC | 0.0188 | 0.0104 to 0.0335 |

- ICC for model ratio: 0.721
- Classification stability, all: unanimous 0.853, split 0.147.
- Classification stability, borderline 0.60-0.70: unanimous 0.607, split 0.393.

### ATS Repeatability

| source | n | repeatable n | repeatable fraction | FEV1 top-2 diff median L | FVC top-2 diff median L |
|---|---:|---:|---:|---:|---:|
| observed curves | 6677 | 5630 | 0.843 | 0.0441 | 0.0534 |
| OMBE model estimates | 6677 | 5373 | 0.805 | 0.0479 | 0.0523 |

The model estimates are not more repeatable than the observed raw manoeuvre metrics under the ATS top-two rule.
