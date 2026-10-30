# Physics-Informed Selective Reliability Evaluation

The reliability rule was developed with nested participant-separated cross-fitting on Cycle F TRAIN only. The operating coverage and all model parameters were then fixed for the Cycle E and G comparison.

| Cycle | Method | Coverage | Sensitivity | Specificity | Kappa | FN | FP |
|---|---|---:|---:|---:|---:|---:|---:|
| E | ridge_reliability | 0.750 | 0.978 | 0.993 | 0.933 | 6 | 30 |
| E | physics_reliability | 0.750 | 0.960 | 0.997 | 0.952 | 11 | 14 |
| E | ridge_margin | 0.750 | 0.957 | 0.993 | 0.907 | 9 | 30 |
| E | direct_margin | 0.750 | 0.949 | 0.998 | 0.944 | 9 | 10 |
| G | ridge_reliability | 0.750 | 0.961 | 0.995 | 0.928 | 9 | 23 |
| G | physics_reliability | 0.750 | 0.961 | 0.998 | 0.959 | 9 | 9 |
| G | ridge_margin | 0.750 | 0.961 | 0.995 | 0.917 | 7 | 23 |
| G | direct_margin | 0.750 | 0.965 | 0.998 | 0.953 | 5 | 8 |

Overall audited success rule: **PASS**.
