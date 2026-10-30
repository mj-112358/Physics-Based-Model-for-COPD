# Manuscript Baseline Comparison

All results use the existing Cycle F TRAIN/TEST split. The final policy remains an internal policy-development result because Cycle F informed policy comparison.

| Comparator | n | Coverage | Sensitivity | Specificity | Kappa |
|---|---:|---:|---:|---:|---:|
| QC-first direct measurement | 479 | 0.851 | 0.811 | 0.995 | 0.785 |
| Direct-ratio margin at matched coverage | 464 | 0.824 | 0.905 | 1.000 | 0.909 |
| FEV1/FEV6 at 0.73 | 563 | 1.000 | 0.732 | 0.988 | 0.697 |
| Prefix-summary ridge, all cases | 563 | 1.000 | 0.965 | 0.936 | 0.903 |
| Prefix-summary ridge at matched coverage | 464 | 0.824 | 0.992 | 0.990 | 0.983 |
| Prefix-summary ridge at fixed specificity | 563 | 1.000 | 0.850 | 0.984 | 0.820 |
| Proposed selective policy | 464 | 0.824 | 0.963 | 0.977 | 0.940 |

The ridge baseline used alpha=0.0001, selected on TRAIN participants only. Its all-case FVC mean absolute error was 85.1 mL.

The quality-control-first comparator defers every ATS-truncated manoeuvre before classification. The matched-coverage comparators rank cases by absolute distance from the 0.70 decision boundary and retain exactly the final policy's number of classified cases.

The committed repository does not contain the Cycle E or G participant-level workflow features or the full natural-pair row set. An untouched-cycle policy evaluation and a natural-pair FEV1/FEV6 analysis therefore cannot be reproduced from this checkout.
