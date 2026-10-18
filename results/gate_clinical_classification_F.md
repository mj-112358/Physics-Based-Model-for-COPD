# Gate - Clinical Classification Accuracy, Cycle F

- VERDICT: ALIVE - obstruction sensitivity clears 0.90 and severe demotion-to-normal is <=0.05; onset failure does not propagate to the screening decision.
- model tested: plain bi-exponential OMBE fit in the volume domain only
- excluded methods: gamma onset, time registration, ML residual correction
- random seed: 20260708
- raw source: <PROJECT_ROOT>/NHANES_SpiroSBI/2009-2010/spxraw_f.sas7bdat
- raw loaded rows: 36421
- source mapping: `merged_summary.parquet` canonical fields resolved from `results/column_map.json`; cycle-F source variables are `SPXNFEV1` for FEV1 and `SPXNFVC` for FVC.

## Manoeuvre Selection

Use Pre curves with `SPAACC == 'Y'` and raw manoeuvre grade `SPAQEFF in {'A','B'}`; parse each curve; select the per-participant manoeuvre with maximum parsed FEV1.

Conservative tie rule: if derived FEV1 ties, the lowest manoeuvre number is selected.

## Sample Construction

| quantity | n |
|---|---:|
| Tier-2 cycle-F severe truth participants (<0.50) | 38 |
| Tier-2 cycle-F moderate truth participants (0.50-0.60) | 91 |
| severe with usable selected A/B raw curve | 31 |
| moderate with usable selected A/B raw curve | 83 |
| severe dropped for missing/unusable raw curve | 7 |
| moderate dropped for missing/unusable raw curve | 8 |
| normal candidates with ratio >0.70 and usable selected A/B raw curve | 6214 |
| normal sampled | 300 |
| participants fit attempted | 414 |
| participants fit valid | 414 |

## A. Fixed 0.70 Classification

|                   |   model normal |   model obstructed |
|:------------------|---------------:|-------------------:|
| NHANES normal     |            287 |                 13 |
| NHANES obstructed |              2 |                112 |

| metric | value |
|---|---:|
| sensitivity | 0.982 |
| specificity | 0.957 |
| overall agreement | 0.964 |
| Cohen's kappa | 0.912 |

## B. GLI LLN Classification

- valid GLI rows: 412/414
- excluded fraction: 0.005

|                   |   model normal |   model obstructed |
|:------------------|---------------:|-------------------:|
| NHANES normal     |            276 |                 18 |
| NHANES obstructed |              4 |                114 |

| metric | value |
|---|---:|
| sensitivity | 0.966 |
| specificity | 0.939 |
| overall agreement | 0.947 |
| Cohen's kappa | 0.874 |

## C. Severity Confusion

Truth rows use the requested strata. Model columns include `mild_0.60_0.70` explicitly because fitted ratios can land in the gap between moderate and normal.

| truth_severity   |   severe |   moderate |   mild_0.60_0.70 |   normal |
|:-----------------|---------:|-----------:|-----------------:|---------:|
| severe           |       24 |          7 |                0 |        0 |
| moderate         |        5 |         46 |               30 |        2 |
| normal           |        0 |          7 |                6 |      287 |

- severe cases demoted to moderate-or-higher: 7/31 (0.226)
- severe cases demoted to normal: 0/31 (0.000)

## D. Obstructed-Set Diagnostic Error Split

| metric | median | IQR |
|---|---:|---:|
| FEV1 abs percent error | 1.38% | 0.62% to 3.31% |
| FVC abs percent error | 5.78% | 3.50% to 8.16% |
| FEV1/FVC absolute error | 0.0219 | 0.0119 to 0.0383 |

## E. Signed Ratio Error

- median signed model-minus-NHANES FEV1/FVC error on obstructed set: 0.0197
- signed-error IQR: 0.0073 to 0.0327
- consistent-bias rule used for MARGINAL: abs(median) > 0.02, IQR width <= 0.04, and IQR does not cross zero.
- consistent bias detected: False

## Gate Logic

- ALIVE requires fixed-threshold obstruction sensitivity >= 0.90 and severe-case demotion-to-normal rate <= 0.05.
- MARGINAL requires sensitivity 0.80-0.90 or a consistent signed ratio bias.
- DEAD requires sensitivity < 0.80 with symmetric/non-correctable ratio error.

VERDICT: ALIVE - obstruction sensitivity clears 0.90 and severe demotion-to-normal is <=0.05; onset failure does not propagate to the screening decision.
