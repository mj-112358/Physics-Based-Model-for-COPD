# Gate - Model vs Observed Reconciliation, Cycle F

- VERDICT: GENUINE-MODEL-DEFICIENCY (model P3 kappa 0.680 vs observed P3 0.918 under segmented_btps)
- c1: RETRACTED (TRAIN fitted value 0.000231)
- MODEL-CONTRIBUTION: truncation recovery, ATS repeatability
- Deterministic LM only. No NUTS, SNPE, gamma, registration, or ML residual.
- Readout swap note: For max_cumulative_btps and final_record_btps, model FVC uses the same observed endpoint delta added to the deterministic LM segmented FVC. OMBE parameters are not refit.

## Part 1 - Stratum Sign Flip

| stratum | metric | n | median % | IQR low | IQR high | IQR crosses 0 |
|---|---|---|---|---|---|---|
| mild_0.60_0.70 | FVC | 242 | -0.184 | -0.620 | 0.650 | True |
| mild_0.60_0.70 | FEV1 | 242 | 1.737 | 0.971 | 2.339 | False |
| moderate | FVC | 45 | -0.961 | -2.078 | 0.232 | True |
| moderate | FEV1 | 45 | 2.069 | 0.878 | 2.569 | False |
| normal | FVC | 3029 | 0.755 | 0.088 | 2.237 | False |
| normal | FEV1 | 3029 | 0.151 | -0.533 | 0.747 | True |
| severe | FVC | 13 | -2.589 | -3.037 | -1.887 | False |
| severe | FEV1 | 13 | 1.711 | 0.594 | 2.750 | False |

### Bias Regressions

| predictor | n | slope | intercept | R2 | p |
|---|---|---|---|---|---|
| observed_ratio_P3 | 3329 | -0.24150 | 0.17672 | 0.109 | 0.000 |
| tau_s_median | 3329 | -0.01300 | 0.01189 | 0.078 | 0.000 |
| FET_max | 3329 | 0.00727 | -0.09511 | 0.136 | 0.000 |

Best bias predictor by R2: FET_max.

## Part 2 - FVC Convention Sweep

Preferred convention: `segmented_btps`. No convention had every per-stratum model-vs-observed FVC IQR crossing zero; selected highest TEST model P3 fixed-0.70 kappa.

### Convention Choice

| convention | model P3 kappa | sens | spec | borderline FN | all IQRs cross 0 |
|---|---|---|---|---|---|
| segmented_btps | 0.680 | 0.820 | 0.952 | 0.219 | False |
| segmented_no_btps | 0.680 | 0.820 | 0.952 | 0.219 | False |
| max_cumulative_btps | 0.658 | 0.897 | 0.933 | 0.124 | False |
| final_record_btps | 0.520 | 0.517 | 0.967 | 0.595 | False |

### Classification by Convention

| convention | source | method | definition | n | sens | spec | kappa | signed err | borderline FN |
|---|---|---|---|---|---|---|---|---|---|
| segmented_btps | model | single_P1 | fixed_0.70 | 3329 | 0.737 | 0.948 | 0.611 | 0.0169 | 0.318 |
| segmented_btps | model | single_P1 | GLI_LLN | 3317 | 0.748 | 0.933 | 0.585 | 0.0117 | 0.589 |
| segmented_btps | model | pooled_P3 | fixed_0.70 | 3329 | 0.820 | 0.952 | 0.680 | 0.0126 | 0.219 |
| segmented_btps | model | pooled_P3 | GLI_LLN | 3317 | 0.828 | 0.936 | 0.645 | 0.0077 | 0.531 |
| segmented_btps | observed | single_P1 | fixed_0.70 | 3329 | 0.807 | 0.996 | 0.861 | 0.0036 | 0.231 |
| segmented_btps | observed | single_P1 | GLI_LLN | 3317 | 0.803 | 0.994 | 0.849 | 0.0037 | 0.485 |
| segmented_btps | observed | pooled_P3 | fixed_0.70 | 3329 | 0.927 | 0.992 | 0.918 | -0.0006 | 0.087 |
| segmented_btps | observed | pooled_P3 | GLI_LLN | 3317 | 0.911 | 0.988 | 0.890 | 0.0002 | 0.390 |
| segmented_no_btps | model | single_P1 | fixed_0.70 | 3329 | 0.737 | 0.948 | 0.611 | 0.0169 | 0.318 |
| segmented_no_btps | model | single_P1 | GLI_LLN | 3317 | 0.748 | 0.933 | 0.585 | 0.0117 | 0.589 |
| segmented_no_btps | model | pooled_P3 | fixed_0.70 | 3329 | 0.820 | 0.952 | 0.680 | 0.0126 | 0.219 |
| segmented_no_btps | model | pooled_P3 | GLI_LLN | 3317 | 0.828 | 0.936 | 0.645 | 0.0077 | 0.531 |
| segmented_no_btps | observed | single_P1 | fixed_0.70 | 3329 | 0.807 | 0.996 | 0.861 | 0.0036 | 0.231 |
| segmented_no_btps | observed | single_P1 | GLI_LLN | 3317 | 0.803 | 0.994 | 0.849 | 0.0037 | 0.485 |
| segmented_no_btps | observed | pooled_P3 | fixed_0.70 | 3329 | 0.927 | 0.992 | 0.918 | -0.0006 | 0.087 |
| segmented_no_btps | observed | pooled_P3 | GLI_LLN | 3317 | 0.911 | 0.988 | 0.890 | 0.0002 | 0.390 |
| max_cumulative_btps | model | single_P1 | fixed_0.70 | 3329 | 0.797 | 0.934 | 0.603 | 0.0107 | 0.244 |
| max_cumulative_btps | model | single_P1 | GLI_LLN | 3317 | 0.834 | 0.918 | 0.597 | 0.0056 | 0.519 |
| max_cumulative_btps | model | pooled_P3 | fixed_0.70 | 3329 | 0.897 | 0.933 | 0.658 | 0.0057 | 0.124 |
| max_cumulative_btps | model | pooled_P3 | GLI_LLN | 3317 | 0.908 | 0.917 | 0.634 | 0.0011 | 0.461 |
| max_cumulative_btps | observed | single_P1 | fixed_0.70 | 3329 | 0.883 | 0.989 | 0.877 | -0.0021 | 0.136 |
| max_cumulative_btps | observed | single_P1 | GLI_LLN | 3317 | 0.917 | 0.986 | 0.886 | -0.0024 | 0.402 |
| max_cumulative_btps | observed | pooled_P3 | fixed_0.70 | 3329 | 0.987 | 0.984 | 0.911 | -0.0045 | 0.012 |
| max_cumulative_btps | observed | pooled_P3 | GLI_LLN | 3317 | 0.991 | 0.980 | 0.902 | -0.0038 | 0.344 |
| final_record_btps | model | single_P1 | fixed_0.70 | 3329 | 0.433 | 0.966 | 0.443 | 0.0519 | 0.682 |
| final_record_btps | model | single_P1 | GLI_LLN | 3317 | 0.385 | 0.954 | 0.370 | 0.0460 | 0.793 |
| final_record_btps | model | pooled_P3 | fixed_0.70 | 3329 | 0.517 | 0.967 | 0.520 | 0.0436 | 0.595 |
| final_record_btps | model | pooled_P3 | GLI_LLN | 3317 | 0.468 | 0.957 | 0.453 | 0.0393 | 0.730 |
| final_record_btps | observed | single_P1 | fixed_0.70 | 3329 | 0.493 | 0.999 | 0.634 | 0.0395 | 0.616 |
| final_record_btps | observed | single_P1 | GLI_LLN | 3317 | 0.385 | 0.999 | 0.527 | 0.0405 | 0.734 |
| final_record_btps | observed | pooled_P3 | fixed_0.70 | 3329 | 0.553 | 0.997 | 0.679 | 0.0325 | 0.550 |
| final_record_btps | observed | pooled_P3 | GLI_LLN | 3317 | 0.458 | 0.998 | 0.595 | 0.0348 | 0.664 |

### P3 Model-vs-Observed FVC Error by Stratum

| convention | stratum | n | median % | IQR low | IQR high | IQR crosses 0 |
|---|---|---|---|---|---|---|
| segmented_btps | mild_0.60_0.70 | 242 | -0.184 | -0.620 | 0.650 | True |
| segmented_btps | moderate | 45 | -0.961 | -2.078 | 0.232 | True |
| segmented_btps | normal | 3029 | 0.755 | 0.088 | 2.237 | False |
| segmented_btps | severe | 13 | -2.589 | -3.037 | -1.887 | False |
| segmented_no_btps | mild_0.60_0.70 | 242 | -0.162 | -0.612 | 0.650 | True |
| segmented_no_btps | moderate | 45 | -0.961 | -2.078 | 0.232 | True |
| segmented_no_btps | normal | 3029 | 0.754 | 0.087 | 2.235 | False |
| segmented_no_btps | severe | 13 | -2.589 | -3.037 | -1.887 | False |
| max_cumulative_btps | mild_0.60_0.70 | 242 | -0.056 | -0.541 | 0.977 | True |
| max_cumulative_btps | moderate | 45 | -0.817 | -1.777 | 0.248 | True |
| max_cumulative_btps | normal | 3029 | 1.165 | 0.214 | 3.554 | False |
| max_cumulative_btps | severe | 13 | -2.588 | -3.037 | -1.886 | False |
| final_record_btps | mild_0.60_0.70 | 242 | -0.088 | -0.632 | 1.017 | True |
| final_record_btps | moderate | 45 | -0.947 | -2.149 | 0.279 | True |
| final_record_btps | normal | 3029 | 1.241 | 0.228 | 3.759 | False |
| final_record_btps | severe | 13 | -2.844 | -3.037 | -2.001 | False |

## Part 3 - c1 Retraction Check

- preferred convention c1 fitted on TRAIN: 0.000231
- c1 status: RETRACTED

| method | correction | n | sens | spec | kappa | borderline FN |
|---|---|---|---|---|---|---|
| single_P1 | no_c1 | 3329 | 0.737 | 0.948 | 0.611 | 0.318 |
| single_P1 | train_c1 | 3329 | 0.737 | 0.948 | 0.611 | 0.318 |
| pooled_P3 | no_c1 | 3329 | 0.820 | 0.952 | 0.680 | 0.219 |
| pooled_P3 | train_c1 | 3329 | 0.820 | 0.952 | 0.680 | 0.219 |

## Part 4 - Where the Model Wins

On complete curves, the model loses to pooled direct measurement under segmented_btps: model P3 kappa 0.680 vs observed P3 0.918.

On truncated curves, the model beats direct measurement at T=2 by obstructed false-negative rate.

### Truncation Recovery

| source | T | n | sens | spec | kappa | obstructed FN |
|---|---|---|---|---|---|---|
| OBSERVED_DIRECT | 2 | 497 | 0.010 | 1.000 | 0.008 | 0.990 |
| OBSERVED_DIRECT | 3 | 497 | 0.135 | 1.000 | 0.111 | 0.865 |
| OBSERVED_DIRECT | 4 | 497 | 0.316 | 1.000 | 0.271 | 0.684 |
| OBSERVED_DIRECT | 5 | 497 | 0.451 | 1.000 | 0.398 | 0.549 |
| MODEL_OMBE | 2 | 497 | 0.411 | 0.965 | 0.330 | 0.589 |
| MODEL_OMBE | 3 | 497 | 0.404 | 0.970 | 0.328 | 0.596 |
| MODEL_OMBE | 4 | 497 | 0.481 | 0.970 | 0.403 | 0.519 |
| MODEL_OMBE | 5 | 497 | 0.532 | 0.970 | 0.453 | 0.468 |

### Repeatability

| source | n | FEV1 SD | FVC SD | ratio SD | ATS repeatable |
|---|---|---|---|---|---|
| model | 6655 | 0.0863 | 0.0745 | 0.0143 | 0.876 |
| observed | 6655 | 0.0830 | 0.0930 | 0.0185 | 0.844 |

### Matched Normal-Truth Flag Burden

| source | method | train target | spread threshold | test normal flag | test borderline flag | borderline confident FN |
|---|---|---|---|---|---|---|
| model | P3 | 0.120 | 0.1199 | 0.108 | 0.058 | 0.219 |
| observed | P3 | 0.120 | 0.1172 | 0.105 | 0.074 | 0.085 |

## Part 5 - GLI Residual Diagnosis

### GLI FN by |LLN - 0.70|

| correction | |LLN-0.70| | n | GLI obstructed n | FN | distance med |
|---|---|---|---|---|---|
| no_c1 | 0.00-0.02 | 878 | 90 | 0.233 | 0.010 |
| no_c1 | 0.02-0.04 | 698 | 65 | 0.169 | 0.030 |
| no_c1 | 0.04-0.08 | 1202 | 126 | 0.143 | 0.059 |
| no_c1 | >=0.08 | 539 | 44 | 0.136 | 0.094 |
| train_c1 | 0.00-0.02 | 878 | 90 | 0.233 | 0.010 |
| train_c1 | 0.02-0.04 | 698 | 65 | 0.169 | 0.030 |
| train_c1 | 0.04-0.08 | 1202 | 126 | 0.143 | 0.059 |
| train_c1 | >=0.08 | 539 | 44 | 0.136 | 0.094 |

### GLI Preferred Classification

| correction | n | sens | spec | kappa | borderline FN |
|---|---|---|---|---|---|
| no_c1 | 3317 | 0.828 | 0.936 | 0.645 | 0.531 |
| train_c1 | 3317 | 0.828 | 0.936 | 0.644 | 0.531 |

## Verdict

CONVENTION: GENUINE-MODEL-DEFICIENCY - model P3 kappa 0.680, observed P3 kappa 0.918.
c1: RETRACTED - fitted value 0.000231.
MODEL-CONTRIBUTION: truncation recovery, ATS repeatability.
