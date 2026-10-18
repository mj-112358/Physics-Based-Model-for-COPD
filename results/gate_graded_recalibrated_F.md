# Gate - Prevalence-Corrected Recalibration of the Graded Obstruction Classifier, Cycle F

VERDICT: GRADED-VIABLE-NEUTRAL - ECE is restored, but TRAIN-frozen confident zones do not safely exceed the 66.6% operational baseline.

## Protocol Decisions

- TEST is scored once after TRAIN-only model fitting and TRAIN-only zone selection.
- Natural-prevalence calibration metrics use class reweighting on TEST because the TEST split is constructed/enriched.
- The post-hoc prior-corrected CLEAN GBT is the primary arm; a weighted-retrained GBT is confirmatory and can be selected only if its ECE is cleaner.
- If the post-hoc prior correction exceeded ECE 0.15, the only fallback allowed was TRAIN-only isotonic recalibration after prior correction.
- Operational deterministic residual deferrals are used for the non-boundary bucket analysis because the requested head-to-head baseline is the 66.6% operational policy.
- A TEST candidate that violates severe FN=0 or PPV/NPV >=0.90 is reported and then retained as probability-only indeterminate output.
- fit_alpha is included only in the secondary physics arm requested here; it remains outside CLEAN because the prior leakage audit quarantined it by ratio correlation.

## Part 1 - Prior Correction

- TRAIN constructed prevalence pi_train: 0.555.
- Natural target prevalence pi_target: 0.096.
- Prior-correction multipliers: a=0.1736, b=2.0314.
- Prior gate reweighted ECE: 0.226.
- Selected probability source for zones: posthoc_prior_corrected.

| source | weighting | n | ECE | Brier |
|---|---|---|---|---|
| raw_train_prevalence | unweighted_constructed_test | 563 | 0.022 | 0.133 |
| uncorrected_train_prevalence | reweighted_natural | 563 | 0.235 | 0.158 |
| posthoc_prior_corrected | reweighted_natural | 563 | 0.013 | 0.064 |
| weighted_retrained | reweighted_natural | 563 | 0.025 | 0.071 |
| fallback_train_only_isotonic_after_prior | not_run | 563 | NA | NA |

- Post-hoc prior-correction status: RESTORED.
- Confirmatory weighted-retrained status: RESTORED.
- Confirmatory ECE difference: 0.012 (AGREES_WITHIN_0.03_ECE).
- The per-TEST-patient corrected probability `p_adj` is in the participants CSV.

Probability distributions:

| probability | mean | p05 | median | p95 | min | max |
|---|---|---|---|---|---|---|
| raw_train_prevalence_test | 0.552 | 0.000 | 0.613 | 0.999 | 0.000 | 1.000 |
| posthoc_prior_corrected_p_adj_test | 0.247 | 0.000 | 0.119 | 0.984 | 0.000 | 1.000 |
| weighted_retrained_test | 0.210 | 0.004 | 0.176 | 0.596 | 0.002 | 0.854 |
| selected_test | 0.247 | 0.000 | 0.119 | 0.984 | 0.000 | 1.000 |

## Part 2 - Frozen Zones on Corrected Probabilities

- TRAIN-frozen boundaries: p_low=0.1194, p_high=0.1250, status=FEASIBLE.
- Candidate rejected on TEST: True.
- Candidate rejection reasons: candidate confident-obstructed PPV is below the reweighted 0.90 floor on TEST.

Candidate zones before enforcing the TEST safety/floor guard:

| scope | confident n | coverage | accuracy | sens | spec | kappa | PPV rw | NPV rw | severe conf | severe FN | severe FN CP high |
|---|---|---|---|---|---|---|---|---|---|---|---|
| candidate | 558 | 0.991 | 0.799 | 0.765 | 0.843 | 0.599 | 0.342 | 0.971 | 16 | 0 | 0.206 |

Retained zones after guard enforcement:

| zone | n | fraction |
|---|---|---|
| INDETERMINATE_GRADED | 563 | 1.000 |

| scope | confident n | coverage | accuracy | sens | spec | kappa | PPV rw | NPV rw | severe conf | severe FN | severe FN CP high |
|---|---|---|---|---|---|---|---|---|---|---|---|
| retained | 0 | 0.000 | NA | NA | NA | NA | NA | NA | 0 | 0 | NA |

## Part 3 - Non-Boundary Deferral Recovery

Operational deterministic deferrals only:

| bucket | true abs distance | n | fraction | retained confident | retained acc | retained severe FN | candidate confident | candidate acc | candidate severe FN |
|---|---|---|---|---|---|---|---|---|---|
| KNIFE_EDGE | <0.02 | 84 | 0.447 | 0 | NA | 0 | 82 | 0.683 | 0 |
| NEAR | 0.02-0.05 | 49 | 0.261 | 0 | NA | 0 | 49 | 0.633 | 0 |
| RECOVERABLE | >=0.05 | 55 | 0.293 | 0 | NA | 0 | 54 | 0.667 | 0 |

## Part 4 - Head-to-Head vs Deterministic Baseline

| scope | coverage | kappa | sens | spec | severe FN | severe FN CP high |
|---|---|---|---|---|---|---|
| strict deterministic | 0.453 | 0.842 | 0.847 | 0.974 | 0 | 0.842 |
| operational deterministic | 0.666 | 0.923 | 0.943 | 0.977 | 0 | 0.459 |
| graded retained confident | 0.000 | NA | NA | NA | 0 | NA |
| graded rejected candidate | 0.991 | 0.599 | 0.765 | 0.843 | 0 | 0.206 |

- Operational baseline coverage: 0.666.
- Retained graded confident coverage: 0.000.
- Candidate graded confident coverage before rejection: 0.991.

## Part 5 - Physics-Inclusive Arm

- Arm: CLEAN + fit_alpha; mechanistically independent of label computation despite ratio correlation.

| model | whole AUC | boundary AUC | candidate cov | retained cov | rejected |
|---|---|---|---|---|---|
| CLEAN prior-corrected | 0.886 | 0.591 | 0.991 | 0.000 | True |
| CLEAN + fit_alpha | 0.970 | 0.812 | 0.995 | 0.000 | True |

- Whole AUC: 0.970.
- Boundary AUC: 0.812 (n=190).
- Natural-prevalence ECE: 0.016.
- Candidate confident coverage: 0.995.
- Candidate rejected: True; reasons: candidate confident-obstructed PPV is below the reweighted 0.90 floor on TEST.
- Retained coverage if the same guard is enforced: 0.000.

## Interpretation

- Label-shift prior correction changed ECE from 0.226 to 0.013.
- That restores probability calibration at natural prevalence.
- The corrected TRAIN-frozen zones still fail the TEST guard if they are allowed to emit broad confident calls.
- Therefore the deployable retained graded output is probability-calibrated but does not add safe confident coverage over the 66.6% operational deterministic baseline.
- The secondary fit_alpha arm improves ranking, including the boundary AUC, but still fails the natural-prevalence PPV floor and therefore does not produce deployable confident coverage.

VERDICT: GRADED-VIABLE-NEUTRAL - ECE is restored, but TRAIN-frozen confident zones do not safely exceed the 66.6% operational baseline.
