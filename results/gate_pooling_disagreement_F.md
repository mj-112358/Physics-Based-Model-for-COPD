# Gate - Multi-Manoeuvre Pooling + Disagreement Flag, Cycle F

- VERDICT: BORDERLINE-FLAG-USEFUL
- spread threshold learned on TRAIN: 0.0987 (TRAIN flag rate 0.150)
- train/test counts: {'train': {'normal': 3029, 'mild_0.60_0.70': 241, 'moderate': 44, 'severe': 12}, 'test': {'normal': 3029, 'mild_0.60_0.70': 242, 'moderate': 45, 'severe': 13}}

## B1. Pooling Metrics on TEST

| source | method | n | sens | spec | kappa | signed err | borderline FN |
|---|---|---|---|---|---|---|---|
| model | P1_best_fev1 | 3329 | 0.737 | 0.948 | 0.611 | 0.0029 | 0.318 |
| model | P2_median_ratio | 3329 | 0.807 | 0.918 | 0.565 | -0.0028 | 0.236 |
| model | P3_ATS_largest | 3329 | 0.820 | 0.952 | 0.680 | 0.0005 | 0.219 |
| observed | P1_best_fev1 | 3329 | 0.807 | 0.996 | 0.861 | 0.0077 | 0.231 |
| observed | P2_median_ratio | 3329 | 0.903 | 0.980 | 0.844 | 0.0028 | 0.116 |
| observed | P3_ATS_largest | 3329 | 0.927 | 0.992 | 0.918 | 0.0031 | 0.087 |

## B1. Leave-One-Out Pooling Stability

| source | method | n | median LOO SD | IQR |
|---|---|---|---|---|
| model | P1 | 3188 | 0.0056 | [0.002503008232587778, 0.011315874613852035] |
| model | P2 | 3188 | 0.0054 | [0.00281932172631046, 0.0108588917250198] |
| model | P3 | 3188 | 0.0055 | [0.002581529340145023, 0.010684734152651877] |
| observed | P1 | 3188 | 0.0079 | [0.0036381253069652695, 0.015920534583365394] |
| observed | P2 | 3188 | 0.0070 | [0.0037239136972637295, 0.013004850808899949] |
| observed | P3 | 3188 | 0.0072 | [0.0035342281331199868, 0.013658896350269732] |

## B2. Label-Independent Flag Rates

| flag | stratum | n | flag rate |
|---|---|---|---|
| DISAGREE | mild_0.60_0.70 | 242 | 0.322 |
| DISAGREE | moderate | 45 | 0.022 |
| DISAGREE | normal | 3029 | 0.120 |
| DISAGREE | severe | 13 | 0.000 |
| DISAGREE | ALL | 3329 | 0.133 |
| SPREAD | mild_0.60_0.70 | 242 | 0.087 |
| SPREAD | moderate | 45 | 0.133 |
| SPREAD | normal | 3029 | 0.139 |
| SPREAD | severe | 13 | 0.000 |
| SPREAD | ALL | 3329 | 0.135 |
| ATS_TRUNCATION_EXISTING_POLICY | mild_0.60_0.70 | 250 | 0.096 |
| ATS_TRUNCATION_EXISTING_POLICY | moderate | 47 | 0.213 |
| ATS_TRUNCATION_EXISTING_POLICY | normal | 250 | 0.164 |
| ATS_TRUNCATION_EXISTING_POLICY | severe | 16 | 0.562 |

## B3. Three-Outcome Policy Metrics

| policy | flag | n | obs->OBSTRUCTED | obs->FLAG | obs->NORMAL | borderline FN | normal flag | conf sens | conf spec | kappa |
|---|---|---|---|---|---|---|---|---|---|---|
| model_P2 | DISAGREE | 3329 | 0.670 | 0.263 | 0.067 | 0.079 | 0.120 | 0.910 | 0.949 | 0.693 |
| model_P2 | SPREAD | 3329 | 0.727 | 0.090 | 0.183 | 0.223 | 0.139 | 0.799 | 0.957 | 0.690 |
| model_P3 | DISAGREE | 3329 | 0.670 | 0.263 | 0.067 | 0.079 | 0.120 | 0.910 | 0.949 | 0.693 |
| model_P3 | SPREAD | 3329 | 0.740 | 0.090 | 0.170 | 0.207 | 0.139 | 0.813 | 0.967 | 0.736 |
| observed_P2 | DISAGREE | 3329 | 0.693 | 0.263 | 0.043 | 0.050 | 0.120 | 0.941 | 0.996 | 0.941 |
| observed_P2 | SPREAD | 3329 | 0.817 | 0.090 | 0.093 | 0.112 | 0.139 | 0.897 | 0.989 | 0.887 |
| observed_P3 | DISAGREE | 3329 | 0.700 | 0.263 | 0.037 | 0.041 | 0.120 | 0.950 | 0.997 | 0.951 |
| observed_P3 | SPREAD | 3329 | 0.840 | 0.090 | 0.070 | 0.083 | 0.139 | 0.923 | 0.995 | 0.930 |
| current_single_c1_ATS | ATS_TRUNCATION | 563 | 0.831 | 0.010 | 0.160 | 0.192 | 0.128 | 0.839 | 0.931 | 0.752 |

## B4. Irreducibility Bound

{'row_type': 'part_b_b4_irreducibility', 'borderline_n': 242, 'observed_p2_borderline_fn_rate_no_flag': 0.11570247933884298, 'observed_p2_borderline_confident_fn_after_observed_disagree_flag': 0.01652892561983471, 'label_aware_perfect_flag_bound': 0.0}
