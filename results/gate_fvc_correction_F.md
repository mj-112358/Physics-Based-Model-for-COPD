# Gate - FVC Correction + Borderline Stress Test, Cycle F

- VERDICT: PARTIAL - bias reduced but signed ratio median is not within [-0.01,+0.01]; borderline false negatives worsened
- model tested: plain bi-exponential OMBE in the volume domain
- excluded methods: gamma onset, time registration, ML residual correction
- random seed: 20260708

## Step 1 - Signed-Error Diagnosis

| group | n | signed FEV1 % median | IQR | signed FVC % median | IQR | signed ratio median | IQR |
|---|---:|---:|---:|---:|---:|---:|---:|
| obstructed | 114 | -1.16% | -3.31% to 0.06% | -5.18% | -7.93% to -2.73% | 0.0197 | 0.0073 to 0.0327 |
| normal | 300 | -3.03% | -4.59% to -2.04% | -3.20% | -4.36% to -2.36% | 0.0020 | -0.0079 to 0.0128 |
| pooled | 414 | -2.74% | -4.47% to -1.58% | -3.54% | -5.26% to -2.44% | 0.0051 | -0.0049 to 0.0178 |

Mechanism verdict: FVC-underestimate mechanism confirmed: signed FVC error is consistently negative and larger than signed FEV1 error.

## Step 2 - FVC Convention Comparison

- baseline: prior gate convention, fitted model FVC/end-volume.
- variant_a_model_raw_eot: fitted model volume at raw end-of-test. Raw EOT is available for all selected manoeuvres, so no 6 s fallback was used.
- variant_b_observed_raw_eot: observed selected-manoeuvre end-expiratory raw volume is used as FVC; the model still supplies FEV1.

| scope | convention | n | signed ratio median | IQR | IQR brackets zero |
|---|---|---:|---:|---:|---:|
| obstructed | baseline | 114 | 0.0197 | 0.0073 to 0.0327 | False |
| obstructed | variant_a_model_raw_eot | 114 | 0.0197 | 0.0073 to 0.0327 | False |
| obstructed | variant_b_observed_raw_eot | 114 | 0.0108 | 0.0018 to 0.0200 | False |
| normal | baseline | 300 | 0.0020 | -0.0079 to 0.0128 | True |
| normal | variant_a_model_raw_eot | 300 | 0.0020 | -0.0079 to 0.0128 | True |
| normal | variant_b_observed_raw_eot | 300 | 0.0098 | -0.0003 to 0.0267 | True |

Preferred convention: `variant_b_observed_raw_eot` because its obstructed-set signed ratio median is closest to zero (0.0108) with IQR 0.0018 to 0.0200.

## Step 3 - Classification Retest Under Preferred FVC

| definition | convention | sensitivity | specificity | agreement | kappa | TP | TN | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed 0.70 | baseline | 0.982 | 0.957 | 0.964 | 0.912 | 112 | 287 | 13 | 2 |
| fixed 0.70 | preferred | 0.991 | 0.993 | 0.993 | 0.982 | 113 | 298 | 2 | 1 |
| GLI LLN | baseline | 0.966 | 0.939 | 0.947 | 0.874 | 114 | 276 | 18 | 4 |
| GLI LLN | preferred | 0.975 | 0.976 | 0.976 | 0.941 | 115 | 287 | 7 | 3 |

Specificity guard: PASS: preferred fixed-threshold specificity remains >=0.90

### Fixed 0.70 Confusion Matrices

Baseline:

|                   |   model normal |   model obstructed |
|:------------------|---------------:|-------------------:|
| NHANES normal     |            287 |                 13 |
| NHANES obstructed |              2 |                112 |

Preferred:

|                   |   model normal |   model obstructed |
|:------------------|---------------:|-------------------:|
| NHANES normal     |            298 |                  2 |
| NHANES obstructed |              1 |                113 |

## Step 4 - Borderline-Band Stress Test

- borderline available with usable selected A/B raw curves: 503
- borderline sampled: 150
- fit-valid sampled rows: 150
- available-but-not-sampled count: 353

| convention | n | correct obstructed fraction | pushed-to-normal fraction | false negatives | signed ratio median | IQR |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 150 | 0.760 | 0.240 | 36 | 0.0138 | 0.0058 to 0.0246 |
| preferred | 150 | 0.700 | 0.300 | 45 | 0.0135 | 0.0069 to 0.0267 |

## Step 5 - Drop-Bias Audit

Retained obstructed participants: n=114, NHANES ratio median=0.5488, IQR=0.4978 to 0.5752, mean best raw SPAQEFF grade score=1.00.

Dropped obstructed participants: n=15, NHANES ratio median=0.5044, IQR=0.4282 to 0.5324, mean best raw SPAQEFF grade score=3.00. drops are skewed toward lower FEV1/FVC ratios.

VERDICT: PARTIAL - bias reduced but signed ratio median is not within [-0.01,+0.01]; borderline false negatives worsened
