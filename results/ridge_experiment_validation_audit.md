# Ridge Experiment Validation Audit

## Verdict

The Cycle F ridge fit has a participant-disjoint training/test split, and ridge regularization, imputation, scaling, and coefficients are estimated without test labels. The experiment nevertheless is not a clean censored-FVC baseline because `parsed_fvc_l` is the terminal volume of the full selected same-session manoeuvre.

The terminal-volume feature correlates 0.9934 with released summary FVC and has an absolute error of 185.7 mL. This is target-proxy leakage relative to the missing-endpoint estimand, not participant record leakage.

## Reproducibility Checks

| Check | Result |
|---|---:|
| participant separated split | True |
| one row per participant | True |
| reported test is final policy participant set | True |
| truth matches final policy | True |
| preprocessing fitted on training rows | True |
| ridge alpha selected by training folds | True |
| test labels absent from alpha selection | True |
| bootstrap resamples participant rows | True |
| bootstrap refits ridge | False |

## Comparability

The existing matched-coverage numbers compare decisions on the same participant universe, but they do not isolate estimator value under equal information. A frozen cross-cycle study must report both the archived ridge and an endpoint-censored ridge using the same observation window as the physics estimator.

## Bootstrap

The 2,000-replicate percentile bootstrap is participant-level because each table has one row per participant. It conditions on the fitted ridge model and does not include training-sample uncertainty or paired confidence intervals for method differences.
