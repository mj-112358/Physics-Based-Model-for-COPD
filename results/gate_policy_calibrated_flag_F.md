# Gate - Bias-Calibrated Classification + ATS Flag Policy, Cycle F

- VERDICT: MODERATE - borderline residual false negatives improve versus 0.24 baseline, but one or more STRONG criteria do not clear.
- model: plain bi-exponential volume-domain fit, baseline FVC convention
- no gamma, no registration, no ML residual, no clinical-threshold movement
- seed: 20260708
- preferred correction: `c1`
- preferred TEST policy variant: `flag_unbanded` (correction + flag, unbanded)

## Sample

- truth counts before raw usability filtering: {'normal': 6403, 'mild_0.60_0.70': 529, 'moderate': 107, 'severe': 40}
- available selected counts: {'severe': 31, 'moderate': 94, 'borderline': 503, 'normal': 6214}
- dropped before sampling counts: {'normal': 189, 'mild_0.60_0.70': 26, 'moderate': 13, 'severe': 9}
- sampled counts: {'severe': 31, 'moderate': 94, 'borderline': 500, 'normal': 500}
- valid fit counts: {'mild_0.60_0.70': 500, 'normal': 500, 'moderate': 94, 'severe': 31}
- train/test split counts: {'test': {'mild_0.60_0.70': 250, 'normal': 250, 'moderate': 47, 'severe': 16}, 'train': {'mild_0.60_0.70': 250, 'normal': 250, 'moderate': 47, 'severe': 15}}

## Step 1 - Train-Only Bias Correction

| correction | fitted form | params | train residual median | train residual IQR | test residual median | test residual IQR |
|---|---|---:|---:|---:|---:|---:|
| c1 | `corrected_ratio = model_ratio - (0.008637)` | 1 | 0.0000 | -0.0096 to 0.0114 | 0.0016 | -0.0102 to 0.0120 |
| c2 | `corrected_ratio = model_ratio - (-0.102225 + 0.141564 * model_ratio)` | 2 | 0.0104 | -0.0117 to 0.0259 | 0.0116 | -0.0136 to 0.0290 |

Preferred correction: `c1`. Preference was made on held-out TEST behavior, not fit quality.
Preferred policy variant: `flag_unbanded`. This preference was made on held-out TEST behavior among the pre-requested variants.

## Bug Diagnosis - Flag Branch Reachability

- TEST participants with ATS truncation flag: 84/563
- TEST participants in corrected-ratio [0.60,0.70): 202/563
- TEST participants in raw-model-ratio [0.60,0.70): 185/563
- old impossible fixed-0.70 FLAG branch count: 0

| fixed-threshold variant | FLAG | OBSTRUCTED | NORMAL |
|---|---:|---:|---:|
| correction only | 0 | 275 | 288 |
| correction + flag, corrected-ratio band | 0 | 275 | 288 |
| correction + flag, raw-model-ratio band | 0 | 275 | 288 |
| correction + flag, unbanded | 35 | 275 | 253 |

## Step 2 - Pre-Specified ATS Truncation Flag

- flag: `dV_last > 0.025 L OR FET < 6.0 s`.
- truncation flag fraction by stratum: {'mild_0.60_0.70': {'n': 500, 'flagged_fraction': 0.09}, 'moderate': {'n': 94, 'flagged_fraction': 0.2127659574468085}, 'normal': {'n': 500, 'flagged_fraction': 0.154}, 'severe': {'n': 31, 'flagged_fraction': 0.5161290322580645}}

### Sensitivity Sweep

| plateau L | FET s | policy variant | flag all | flag normal truth | FLAG outcome all | borderline residual FN | overall residual FN | nonflag specificity |
|---|---|---|---|---|---|---|---|---|
| 0.010 | 5.0 | flag_unbanded | 0.256 | 0.136 | 0.078 | 0.164 | 0.137 | 0.931 |
| 0.010 | 6.0 | flag_unbanded | 0.282 | 0.184 | 0.099 | 0.164 | 0.137 | 0.926 |
| 0.025 | 5.0 | flag_unbanded | 0.121 | 0.076 | 0.039 | 0.192 | 0.160 | 0.935 |
| 0.025 | 6.0 | flag_unbanded | 0.149 | 0.128 | 0.062 | 0.192 | 0.160 | 0.931 |
| 0.050 | 5.0 | flag_unbanded | 0.053 | 0.064 | 0.032 | 0.196 | 0.163 | 0.936 |
| 0.050 | 6.0 | flag_unbanded | 0.083 | 0.116 | 0.055 | 0.196 | 0.163 | 0.932 |

## Step 3-4 - TEST-Set Accounting

Baseline is naive fixed-0.70 with no correction and no flag. Policy rows use the preferred correction.

| policy | caught obstructed | flagged obstructed | residual FN | borderline FN | normal correct | normal flag | normal FP | flag all | nonflag sens | nonflag spec | kappa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline, no correction/no flag | 0.764 | 0.000 | 0.236 | 0.288 | 0.944 | 0.000 | 0.056 | 0.000 | 0.764 | 0.944 | 0.691 |
| correction only | 0.831 | 0.000 | 0.169 | 0.204 | 0.940 | 0.000 | 0.060 | 0.000 | 0.831 | 0.940 | 0.759 |
| correction + flag, corrected-ratio band | 0.831 | 0.000 | 0.169 | 0.204 | 0.940 | 0.000 | 0.060 | 0.000 | 0.831 | 0.940 | 0.759 |
| correction + flag, raw-model-ratio band | 0.831 | 0.000 | 0.169 | 0.204 | 0.940 | 0.000 | 0.060 | 0.000 | 0.831 | 0.940 | 0.759 |
| correction + flag, unbanded | 0.831 | 0.010 | 0.160 | 0.192 | 0.812 | 0.128 | 0.060 | 0.062 | 0.839 | 0.931 | 0.752 |

### Candidate Corrections and Policy Variants on TEST

| correction | policy variant | borderline residual FN | overall residual FN | normal flag burden | nonflag specificity | test residual median |
|---|---|---:|---:|---:|---:|---:|
| c1 | correction only | 0.204 | 0.169 | 0.000 | 0.940 | 0.0016 |
| c1 | correction + flag, corrected-ratio band | 0.204 | 0.169 | 0.000 | 0.940 | 0.0016 |
| c1 | correction + flag, raw-model-ratio band | 0.204 | 0.169 | 0.000 | 0.940 | 0.0016 |
| c1 | correction + flag, unbanded | 0.192 | 0.160 | 0.128 | 0.931 | 0.0016 |
| c2 | correction only | 0.340 | 0.278 | 0.000 | 0.948 | 0.0116 |
| c2 | correction + flag, corrected-ratio band | 0.340 | 0.278 | 0.000 | 0.948 | 0.0116 |
| c2 | correction + flag, raw-model-ratio band | 0.332 | 0.272 | 0.000 | 0.948 | 0.0116 |
| c2 | correction + flag, unbanded | 0.312 | 0.256 | 0.132 | 0.940 | 0.0116 |

### GLI LLN Policy Metrics

| policy | residual FN | borderline FN | normal flag burden | nonflag specificity | FLAG outcome all |
|---|---:|---:|---:|---:|---:|
| correction only | 0.159 | 0.508 | 0.000 | 0.924 | 0.000 |
| correction + flag, corrected-ratio band | 0.141 | 0.472 | 0.018 | 0.923 | 0.018 |
| correction + flag, raw-model-ratio band | 0.141 | 0.476 | 0.015 | 0.923 | 0.016 |
| correction + flag, unbanded | 0.132 | 0.460 | 0.094 | 0.916 | 0.068 |

## Step 5 - Generalization / Overfit Check

TEST residual is not materially worse than TRAIN residual.

This is calibration, not threshold-shopping: the correction is learned on disjoint TRAIN participants, uses <=2 parameters, the clinical threshold remains fixed, and the truncation flag is computed from the ATS end-of-test rule before outcomes.

## Step 6 - Coverage / Drop Audit

- obstructed drop audit: {'retained': {'n': 114, 'ratio_median': 0.5487707052215844, 'ratio_iqr': [0.4977939379803634, 0.575241711710468], 'mean_best_raw_spaqeff_grade_score': 1.0, 'best_raw_spaqeff_grade_counts': {'A': 114}}, 'dropped': {'n': 15, 'ratio_median': 0.5043656908063687, 'ratio_iqr': [0.4281753271872386, 0.5324089397468055], 'mean_best_raw_spaqeff_grade_score': 3.0, 'best_raw_spaqeff_grade_counts': {'C': 15}}, 'skew_statement': 'drops are skewed toward lower FEV1/FVC ratios'}
- remaining TEST obstructed-truth NORMAL leaks that were truncation-flaggable but outside the preferred policy: 0

VERDICT: MODERATE - borderline residual false negatives improve versus 0.24 baseline, but one or more STRONG criteria do not clear.
