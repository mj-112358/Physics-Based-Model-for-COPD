# Gate - GLI/Age Mechanism, Cycle F

## Protocol Decisions

- Seed recorded as 20260710; Part 0 is deterministic on existing Cycle F clinical/offset artifacts and does not resample.
- Primary Part 0 sample is participant-level P3, all valid clinical/offset participants with age, true FEV1/FVC, and true FVC.
- Train/test participant rows are reported only as sensitivity checks using the existing clinical/offset split machinery.
- The requested signed error is model-minus-observed FEV1/FVC ratio error. Positive means the OMBE-derived ratio is higher than the parsed observed ratio.
- Parts 1-3 are not run if Part 0 fails the explicit |age r| > 0.70 guard.

## Part 0 - Establish The Signal

### Ratio Error Signal

| level | scope | primary | predictor | n | r | R^2 | slope | slope units | p |
|---|---|---|---|---|---|---|---|---|---|
| participant_P3 | all | True | age_years | 6655 | 0.3185 | 0.1014 | 0.01048 | ratio error per decade | 9.200e-157 |
| participant_P3 | all | True | true_fvc_l | 6655 | 0.4159 | 0.1730 | 0.02404 | ratio error per L | 9.710e-277 |
| participant_P3 | train | False | age_years | 3326 | 0.3218 | 0.1035 | 0.01116 | ratio error per decade | 5.627e-81 |
| participant_P3 | train | False | true_fvc_l | 3326 | 0.4385 | 0.1923 | 0.02641 | ratio error per L | 2.253e-156 |
| participant_P3 | test | False | age_years | 3329 | 0.3147 | 0.0990 | 0.00976 | ratio error per decade | 2.004e-77 |
| participant_P3 | test | False | true_fvc_l | 3329 | 0.3914 | 0.1532 | 0.02157 | ratio error per L | 2.800e-122 |
| maneuver | all | False | age_years | 23743 | 0.2726 | 0.0743 | 0.01155 | ratio error per decade | 0.000e+00 |
| maneuver | all | False | true_fvc_l | 23743 | 0.4550 | 0.2070 | 0.03359 | ratio error per L | 0.000e+00 |

### Explained-Variance Contrast

| outcome | level | scope | primary | age R^2 | true FVC R^2 | age/FVC R^2 | FVC/age R^2 |
|---|---|---|---|---|---|---|---|
| model_minus_observed_fev1_fvc_ratio_error | participant_P3 | all | True | 0.1014 | 0.1730 | 0.586 | 1.705 |
| model_minus_observed_fev1_fvc_ratio_error | participant_P3 | train | False | 0.1035 | 0.1923 | 0.538 | 1.857 |
| model_minus_observed_fev1_fvc_ratio_error | participant_P3 | test | False | 0.0990 | 0.1532 | 0.647 | 1.547 |
| model_minus_observed_fev1_fvc_ratio_error | maneuver | all | False | 0.0743 | 0.2070 | 0.359 | 2.786 |
| gli_lln_ratio_context | participant_P3 | all | False | 0.9040 | 0.0378 | 23.938 | 0.042 |
| gli_lln_ratio_context | participant_P3 | train | False | 0.9051 | 0.0376 | 24.100 | 0.041 |
| gli_lln_ratio_context | participant_P3 | test | False | 0.9028 | 0.0381 | 23.694 | 0.042 |

### GLI LLN Context

| scope | predictor | n | r | R^2 | slope | slope units | p |
|---|---|---|---|---|---|---|---|
| all | age_years | 6627 | -0.9508 | 0.9040 | -0.02258 | GLI LLN ratio per decade | 0.000e+00 |
| all | true_fvc_l | 6627 | -0.1943 | 0.0378 | -0.00810 | GLI LLN ratio per L | 2.095e-57 |
| train | age_years | 3310 | -0.9514 | 0.9051 | -0.02258 | GLI LLN ratio per decade | 0.000e+00 |
| train | true_fvc_l | 3310 | -0.1938 | 0.0376 | -0.00797 | GLI LLN ratio per L | 2.263e-29 |
| test | age_years | 3317 | -0.9502 | 0.9028 | -0.02259 | GLI LLN ratio per decade | 0.000e+00 |
| test | true_fvc_l | 3317 | -0.1952 | 0.0381 | -0.00825 | GLI LLN ratio per L | 7.649e-30 |

The original GLI LLN age signal does reproduce: all-participant GLI LLN vs age r=-0.9508, R^2=0.9040; GLI LLN vs true FVC r=-0.1943, R^2=0.0378; age/FVC explained-variance ratio=23.938.

But the required forward-model error premise does not reproduce: primary ratio error vs age r=0.3185, R^2=0.1014, slope=0.01048 ratio error per decade; ratio error vs true FVC r=0.4159, R^2=0.1730, slope=0.02404 ratio error per L. The age/FVC R^2 ratio is 0.586, not 23x.

HALT: signed model-minus-observed FEV1/FVC ratio error vs age did not reproduce at |r| > 0.70.

## Parts 1-3 - Not Run

Part 1 parameter attribution, Part 2 residual-shape stratification, and Part 3 tri-exponential falsification were not run because the task required Part 0 to reproduce the age signal before explaining it.

Running tau_s attribution or a tri-exponential diagnostic after this guard fails would explain the wrong target: the strong age signal is present in GLI LLN itself, not in the current model-minus-observed ratio error outcome.

## Part 4 - Verdict

VERDICT: MECHANISM-REFUTED

Gate status: HALTED-PREMISE-NOT-STABLE.

The tau_s age-drift hypothesis is not supported because the required age-dependent forward-model ratio-error signal is absent. GLI degradation remains unexplained after the five tested mechanisms, with the added clarification that the strong age dependence is carried by the GLI threshold, not by current OMBE model-minus-observed ratio error.
