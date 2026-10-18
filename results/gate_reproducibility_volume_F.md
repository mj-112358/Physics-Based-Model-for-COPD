# Gate - Volume-Domain Cross-Manoeuvre Reproducibility, Cycle F

- VERDICT: UNSTABLE
- Deterministic LM only. No NUTS, SNPE, gamma onset, registration, or ML residual.
- Random seed: 20260708
- Fit path used: `predictx.gate3_diagnosis.fit_ombe_volume_lm`
- Volume-domain assertion: volume-domain fit path confirmed: gate_clinical_classification_f.fit_ombe_volume_lm is gate3_diagnosis.fit_ombe_volume_lm

## D1. Eligible Manoeuvres

- participants with >=2 acceptable A/B manoeuvres: 6696
- acceptable A/B manoeuvre count distribution: {'2': 271, '3': 3947, '4': 1491, '5': 560, '6': 269, '7': 109, '8': 49}
- candidate manoeuvre rows fit-attempted: 23916

## D2. Fresh Volume-Domain Fits

- fit rows total: 23916
- successful finite volume-domain fit rows: 23821
- failed/nonfinite fit rows: 95
- participants with >=2 successful volume-domain fits: 6677

## D3. Within-Participant Variability

| metric | observed median SD | observed IQR | model median SD | model IQR |
|---|---:|---:|---:|---:|
| FEV1 (L) | 0.0831 | 0.0479 to 0.1417 | 0.0865 | 0.0492 to 0.1477 |
| FVC (L) | 0.0931 | 0.0553 to 0.1514 | 0.0745 | 0.0369 to 0.1280 |
| FEV1/FVC | 0.0185 | 0.0104 to 0.0324 | 0.0144 | 0.0079 to 0.0289 |

- Observed ratio ICC(2,1): 0.793 on n=6677 participants.
- Model ratio ICC(2,1): 0.778 on n=6677 participants.
- ICC note: ICC(2,1) computed on the first two acceptable A/B manoeuvres per participant after sorting by SPAMANU.

## D4. Fixed-0.70 Classification Stability

| stratum | source | n | unanimous fraction | split fraction |
|---|---|---:|---:|---:|
| all | observed | 6677 | 0.884 | 0.116 |
| all | model | 6677 | 0.869 | 0.131 |
| borderline 0.60-0.70 | observed | 483 | 0.667 | 0.333 |
| borderline 0.60-0.70 | model | 483 | 0.698 | 0.302 |

## D5. ATS Repeatability

| source | n | repeatable n | repeatable fraction | FEV1 top-2 diff median L | FVC top-2 diff median L |
|---|---:|---:|---:|---:|---:|
| observed curve values | 6677 | 5630 | 0.843 | 0.0441 | 0.0534 |
| volume-domain model estimates | 6677 | 5841 | 0.875 | 0.0455 | 0.0406 |

Model estimates are more ATS-repeatable than observed curves by 3.2 percentage points.

## Output Rows

- Per-manoeuvre rows contain observed and volume-domain model FEV1/FVC metrics.
- Participant summary rows contain within-participant SD and classification-stability flags.

VERDICT: UNSTABLE; borderline model unanimous fraction 0.698; model-vs-observed ATS repeatability margin 3.2 percentage points.
