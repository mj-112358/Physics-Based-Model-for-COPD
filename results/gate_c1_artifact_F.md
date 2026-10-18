# Gate - c1 Artifact Diagnosis, Cycle F

- VERDICT: MODEL-FAITHFUL
- MODEL-FAITHFUL: model tracks observed FVC closely and both observed and model FVC undershoot NHANES; c1 corrects a raw-curve-versus-NHANES definitional gap, not a bi-exponential tail defect.
- c1 bias from policy gate: 0.008637

## A1. Signed Percent Error vs NHANES

| set | source | metric | n | median % | IQR low | IQR high |
|---|---|---|---|---|---|---|
| obstructed | observed | FEV1 | 114 | -2.762 | -4.549 | -1.951 |
| obstructed | observed | FVC | 114 | -3.661 | -5.455 | -2.189 |
| obstructed | observed | ratio | 114 | 0.102 | -0.805 | 2.312 |
| obstructed | model | FEV1 | 114 | -1.158 | -3.315 | 0.059 |
| obstructed | model | FVC | 114 | -5.178 | -7.929 | -2.731 |
| obstructed | model | ratio | 114 | 3.634 | 1.341 | 5.911 |
| normal | observed | FEV1 | 300 | -2.989 | -3.994 | -2.315 |
| normal | observed | FVC | 300 | -4.345 | -6.245 | -3.120 |
| normal | observed | ratio | 300 | 1.183 | 0.052 | 2.974 |
| normal | model | FEV1 | 300 | -3.033 | -4.594 | -2.036 |
| normal | model | FVC | 300 | -3.198 | -4.359 | -2.356 |
| normal | model | ratio | 300 | 0.245 | -0.945 | 1.595 |

## A2. Model FVC vs Observed FVC

| set | n | median % | IQR low | IQR high |
|---|---|---|---|---|
| obstructed | 114 | -0.916 | -2.435 | 0.175 |
| normal | 300 | 0.900 | 0.134 | 2.482 |

Retract the FVC-underestimate mechanism story from gate_fvc_correction_F if using c1; the observed curve itself undershoots NHANES.

## A3. Available NHANES Fields and EOT/BTPS Variants

- candidate value fields: ['SPAFACT', 'SPXNSTAT', 'SPXNCMT', 'SPXNFVC', 'SPXNEV', 'SPXNFEV5', 'SPXNFEV7', 'SPXNFEV1', 'SPXNFEV3', 'SPXNFEV6', 'SPXNPEF', 'SPXNF257', 'SPXNFET', 'SPXNQFVC', 'SPXNQFV1', 'SPXNQEFF', 'SPXBFVC', 'SPXBFEV5', 'SPXBFEV7', 'SPXBFEV1', 'SPXBFEV3', 'SPXBFEV6', 'SPXBPEF', 'SPXBFET', 'SPXBQFVC']

| variant | n | FVC median % | IQR low | IQR high | ratio median % |
|---|---|---|---|---|---|
| segmented_btps | 414 | -4.168 | -6.186 | -2.936 | 0.915 |
| segmented_no_btps | 414 | -11.860 | -13.763 | -10.458 | 0.915 |
| max_cumulative_btps | 414 | -2.909 | -4.257 | -2.104 | -0.327 |
| final_record_btps | 414 | -9.125 | -11.785 | -7.545 | 6.589 |

## A4. Classification Without c1 vs c1

| definition | ratio | n | sens | spec | kappa | obs signed ratio err | borderline FN |
|---|---|---|---|---|---|---|---|
| fixed_0.70 | uncorrected | 563 | 0.764 | 0.944 | 0.691 | 0.0163 | 0.288 |
| GLI_LLN | uncorrected | 562 | 0.786 | 0.936 | 0.737 | 0.0160 | 0.552 |
| fixed_0.70 | c1_corrected | 563 | 0.831 | 0.940 | 0.759 | 0.0077 | 0.204 |
| GLI_LLN | c1_corrected | 562 | 0.841 | 0.924 | 0.771 | 0.0073 | 0.508 |
