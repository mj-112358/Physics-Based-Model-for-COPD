# Gate - Likelihood Repair, Cycle F

- VERDICT: PARTIAL
- best variant: L4_ar1_time_hetero
- message: Coverage improved but did not clear; next step is a Kennedy-O'Hagan discrepancy term over expiratory time, not further noise-only tweaks.
- posterior classification unblocked: False

## Sample

- TEST subset n: 150
- subset counts: {'mild_0.60_0.70': 67, 'normal': 67, 'moderate': 12, 'severe': 4}
- nested prior-gate 100 present: 100
- NUTS settings: {'warmup': 250, 'draws': 250, 'chains': 2, 'grid_stride': 5, 'nominal_hz': 20.0, 'converged_rule': {'max_rhat_lte': 1.1, 'min_bulk_ess_gte': 50.0}}
- widened priors: {'tau_f': {'low': 0.01, 'high': 1.0}, 'tau_s': {'low': 1.0, 'high': 20.0}, 'alpha': {'low': 0.001, 'high': 0.999}, 'fvc_l': {'low': 0.25, 'high': 12.0}, 'tau_o': {'low': 0.0005, 'high': 5.0}}

## Step 0 - Residual Structure

- diagnosis: True
- residual summary: {'all': {'n': 1125, 'lag1_autocorr': {'n': 1125, 'median': 0.9858146905899048, 'iqr': [0.98117995262146, 0.9887285828590392]}, 'durbin_watson': {'n': 1125, 'median': 1.2413706826919224e-05, 'iqr': [7.058959454298019e-06, 2.1295029000611976e-05]}, 'residual_rms_ml': {'n': 1125, 'median': 1475.110107421875, 'iqr': [668.8220825195312, 2415.387451171875]}}, 'obstructed_truth': {'n': 625, 'lag1_autocorr': {'n': 625, 'median': 0.9879142045974731, 'iqr': [0.9863407611846924, 0.9899507761001588]}, 'durbin_watson': {'n': 625, 'median': 7.723232556600124e-06, 'iqr': [5.689740646630526e-06, 1.4677041690447368e-05]}, 'residual_rms_ml': {'n': 625, 'median': 1476.4951171875, 'iqr': [650.69970703125, 2479.6396484375]}}, 'normal_truth': {'n': 500, 'lag1_autocorr': {'n': 500, 'median': 0.9805408120155334, 'iqr': [0.9774883389472961, 0.983284130692482]}, 'durbin_watson': {'n': 500, 'median': 1.704536498436937e-05, 'iqr': [1.2663558209169423e-05, 2.6974404136126395e-05]}, 'residual_rms_ml': {'n': 500, 'median': 1464.520751953125, 'iqr': [685.8675537109375, 2340.1768798828125]}}, 'assumed_fixed_sigma_ml': 50.0, 'iid_gaussian_falsified': True}

## Step 1 - Likelihood Variants

| variant | n converged | excluded | cov50 | cov80 | cov95 | PIT p | PIT mean | ratio sd median | sigma mL median | rho median | calibrated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| L0_baseline | 143 | 7 | 0.091 | 0.161 | 0.231 | 2.29e-121 | 0.217 | 0.0032 | 50.0 | nan | False |
| L1_learned_sigma | 147 | 3 | 0.034 | 0.048 | 0.082 | 3.46e-168 | 0.192 | 0.0017 | 23.8 | nan | False |
| L2_sigma_ar1 | 146 | 4 | 0.144 | 0.315 | 0.411 | 1.56e-63 | 0.305 | 0.0079 | 6.3 | 0.986 | False |
| L3_student_t | 142 | 8 | 0.049 | 0.099 | 0.176 | 1.31e-99 | 0.379 | 0.0019 | 7.7 | nan | False |
| L4_ar1_time_hetero | 143 | 7 | 0.154 | 0.308 | 0.406 | 1.85e-65 | 0.300 | 0.0080 | 6.1 | 0.986 | False |

## Nested 100 Comparison Set

- nested summaries: {'L0_baseline': {'nested_only': True, 'n_attempted': 100, 'n_fit_errors': 0, 'n_converged': 96, 'n_nonconverged_excluded': 4, 'coverage': {'0.5': 0.08333333333333333, '0.8': 0.16666666666666666, '0.95': 0.2604166666666667}, 'pit_counts': [66, 3, 3, 3, 0, 2, 1, 2, 2, 14], 'pit_chisquare_p': 8.028828979269002e-77, 'pit_p_threshold': 0.01, 'pit_mean': 0.21806250000000002, 'gross_skew': True, 'calibrated': False, 'ratio_sd_median': 0.0033640690380707, 'sigma_ml_median': 50.0, 'rho_median': nan, 'nu_median': nan, 'beta_time_median': nan, 'max_rhat_median': 1.0181932475688846, 'max_rhat_max': 1.0501761132032197, 'min_bulk_ess_median': 114.90868994182674, 'min_bulk_ess_min': 57.00740935681472}, 'L1_learned_sigma': {'nested_only': True, 'n_attempted': 100, 'n_fit_errors': 0, 'n_converged': 99, 'n_nonconverged_excluded': 1, 'coverage': {'0.5': 0.010101010101010102, '0.8': 0.020202020202020204, '0.95': 0.0707070707070707}, 'pit_counts': [80, 0, 0, 1, 0, 0, 0, 0, 1, 17], 'pit_chisquare_p': 1.9343134927686545e-118, 'pit_p_threshold': 0.01, 'pit_mean': 0.1827070707070707, 'gross_skew': True, 'calibrated': False, 'ratio_sd_median': 0.0017223889008164, 'sigma_ml_median': 23.158056661486626, 'rho_median': nan, 'nu_median': nan, 'beta_time_median': nan, 'max_rhat_median': 1.0164865176603777, 'max_rhat_max': 1.082092963715391, 'min_bulk_ess_median': 135.06404405339686, 'min_bulk_ess_min': 55.72117061657199}, 'L2_sigma_ar1': {'nested_only': True, 'n_attempted': 100, 'n_fit_errors': 0, 'n_converged': 98, 'n_nonconverged_excluded': 2, 'coverage': {'0.5': 0.1836734693877551, '0.8': 0.35714285714285715, '0.95': 0.47959183673469385}, 'pit_counts': [47, 9, 3, 1, 6, 2, 5, 5, 4, 16], 'pit_chisquare_p': 1.0796096623267166e-32, 'pit_p_threshold': 0.01, 'pit_mean': 0.335530612244898, 'gross_skew': True, 'calibrated': False, 'ratio_sd_median': 0.009116213768720599, 'sigma_ml_median': 6.33165892213583, 'rho_median': 0.9844590723514557, 'nu_median': nan, 'beta_time_median': nan, 'max_rhat_median': 1.012902396782813, 'max_rhat_max': 1.0819635438274344, 'min_bulk_ess_median': 147.87845967764252, 'min_bulk_ess_min': 52.2867803111778}, 'L3_student_t': {'nested_only': True, 'n_attempted': 100, 'n_fit_errors': 0, 'n_converged': 94, 'n_nonconverged_excluded': 6, 'coverage': {'0.5': 0.0425531914893617, '0.8': 0.09574468085106383, '0.95': 0.18085106382978725}, 'pit_counts': [55, 1, 1, 0, 0, 1, 2, 1, 3, 30], 'pit_chisquare_p': 1.0765240292282777e-64, 'pit_p_threshold': 0.01, 'pit_mean': 0.37825531914893623, 'gross_skew': True, 'calibrated': False, 'ratio_sd_median': 0.00190325628500425, 'sigma_ml_median': 6.67661358602345, 'rho_median': nan, 'nu_median': 2.0662447214126587, 'beta_time_median': nan, 'max_rhat_median': 1.0128239482107917, 'max_rhat_max': 1.0776062163235904, 'min_bulk_ess_median': 136.6283990238744, 'min_bulk_ess_min': 52.413049526415946}, 'L4_ar1_time_hetero': {'nested_only': True, 'n_attempted': 100, 'n_fit_errors': 0, 'n_converged': 95, 'n_nonconverged_excluded': 5, 'coverage': {'0.5': 0.18947368421052632, '0.8': 0.3263157894736842, '0.95': 0.47368421052631576}, 'pit_counts': [48, 7, 3, 1, 6, 1, 6, 3, 4, 16], 'pit_chisquare_p': 2.542637432422911e-36, 'pit_p_threshold': 0.01, 'pit_mean': 0.32741052631578943, 'gross_skew': True, 'calibrated': False, 'ratio_sd_median': 0.008786323480308, 'sigma_ml_median': 6.179039366543293, 'rho_median': 0.9850220680236816, 'nu_median': nan, 'beta_time_median': 0.0116053689271211, 'max_rhat_median': 1.014237870817346, 'max_rhat_max': 1.067138205849944, 'min_bulk_ess_median': 154.4402658505082, 'min_bulk_ess_min': 57.361059944255935}}

## Step 2 - Mechanism Confirmation

- mechanism table: [{'variant': 'L0_baseline', 'sigma_ml_median': 50.0, 'rho_median': nan, 'ratio_sd_median': 0.0032493278849869, 'coverage_50': 0.09090909090909091}, {'variant': 'L1_learned_sigma', 'sigma_ml_median': 23.763248696923256, 'rho_median': nan, 'ratio_sd_median': 0.0016973002348095, 'coverage_50': 0.034013605442176874}, {'variant': 'L2_sigma_ar1', 'sigma_ml_median': 6.29977323114872, 'rho_median': 0.9860157370567322, 'ratio_sd_median': 0.00786310387775295, 'coverage_50': 0.14383561643835616}, {'variant': 'L3_student_t', 'sigma_ml_median': 7.653875043615699, 'rho_median': nan, 'ratio_sd_median': 0.0019393088878132, 'coverage_50': 0.04929577464788732}, {'variant': 'L4_ar1_time_hetero', 'sigma_ml_median': 6.060911808162928, 'rho_median': 0.9862574934959412, 'ratio_sd_median': 0.0080462424084544, 'coverage_50': 0.15384615384615385}]
- pattern holds: True

## NUTS Diagnostics

- diagnostics by variant: {'L0_baseline': {'max_rhat_median': 1.017984870915549, 'max_rhat_max': 1.0534132936573162, 'min_bulk_ess_median': 116.21570800742924, 'min_bulk_ess_min': 51.56474222139716, 'n_nonconverged_excluded': 7}, 'L1_learned_sigma': {'max_rhat_median': 1.0152320394990435, 'max_rhat_max': 1.082092963715391, 'min_bulk_ess_median': 132.1039776173835, 'min_bulk_ess_min': 55.72117061657199, 'n_nonconverged_excluded': 3}, 'L2_sigma_ar1': {'max_rhat_median': 1.0132039234120138, 'max_rhat_max': 1.0819635438274344, 'min_bulk_ess_median': 150.2020591220804, 'min_bulk_ess_min': 52.2867803111778, 'n_nonconverged_excluded': 4}, 'L3_student_t': {'max_rhat_median': 1.0122233596435368, 'max_rhat_max': 1.0941031155883278, 'min_bulk_ess_median': 143.14214151283, 'min_bulk_ess_min': 52.413049526415946, 'n_nonconverged_excluded': 8}, 'L4_ar1_time_hetero': {'max_rhat_median': 1.0138138028550874, 'max_rhat_max': 1.0964629955372736, 'min_bulk_ess_median': 154.35733457831728, 'min_bulk_ess_min': 57.361059944255935, 'n_nonconverged_excluded': 7}}

## Step 3 - Verdict

Coverage improved but did not clear; next step is a Kennedy-O'Hagan discrepancy term over expiratory time, not further noise-only tweaks.
