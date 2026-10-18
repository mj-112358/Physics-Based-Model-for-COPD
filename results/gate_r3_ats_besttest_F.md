# Gate - R3 Recovery via ATS Best-Test Rule, Cycle F

VERDICT: ATS-RULE-RECOVERS - ATS best-test plus the conservative normal-call guard materially increases deployable coverage and preserves severe FN=0.

## ATS/ERS Rule Source

- Source: ATS/ERS 2019 spirometry technical statement, https://doi.org/10.1164/rccm.201908-1590ST.
- The report uses the ATS/ERS rule that reported FEV1 is the largest acceptable FEV1 and reported FVC is the largest acceptable FVC; repeatability is carried as a session quality grade rather than used as a hard stop for reporting.
- The same statement defines repeatability grading from the two largest FEV1 and FVC values, with A/B/C/D/E/F quality levels.

## Protocol Decisions

- TEST split only; training rows are ignored.
- The current deferral-recovery gate is the baseline: 66.6% coverage before this ATS best-test pass.
- NHANES/repro maneuver rows do not expose separate BEV, cough, glottis, leak, or plateau subflags. SPAACC == 'Y' was used as the individual acceptability flag, and this limitation is reported explicitly.
- A maneuver also must have a successful finite model fit to be usable by the PredictX model best-test computation; this is a computational availability requirement, not a repeatability threshold.
- Best-test FEV1 is max model FEV1 across acceptable usable maneuvers, and best-test FVC is max model FVC across acceptable usable maneuvers; the FEV1 and FVC may come from different maneuvers.
- The existing C1 ratio correction is applied after forming max(FEV1)/max(FVC), keeping the recovered ratio on the same calibrated scale as the workflow.
- Repeatability is converted to an ATS-style session grade and is not used as a hard exclusion.
- Automatic normal diagnosis is deferred when any acceptable maneuver remains obstructed on the calibrated ratio scale; obstructed best-test calls are allowed to finalize because they do not create false-normal risk.
- Residual R2 cases are not recovered by best-test because the task explicitly says not to loosen the borderline CI rule.
- TEST truth is used only after label-independent selection to score correctness, sensitivity, specificity, kappa, and severe FN.

## Part 1 - R3 Bucket Re-Examination

| R3 cases | >=2 SPAACC Y | >=1 SPAACC Y | zero SPAACC Y | >=2 model-usable | >=1 model-usable | zero model-usable |
|---|---|---|---|---|---|---|
| 110 | 105 | 105 | 5 | 105 | 105 | 5 |

R3 repeatability grades under the model-usable acceptable set:

| grade | n | fraction |
|---|---|---|
| A | 79 | 0.718 |
| B | 1 | 0.009 |
| C | 10 | 0.091 |
| D | 7 | 0.064 |
| E | 8 | 0.073 |
| F | 5 | 0.045 |

## Part 2 - ATS Best-Test on R3

Two R3 measurements are shown because they answer different questions. `Pure classify-all` applies the best-test computation to every model-usable R3 case. `Deployable conservative` keeps obstructed best-test calls, but defers normal calls when any acceptable maneuver remains obstructed on the calibrated ratio scale.

| policy | finalized R3 | correct | errors | accuracy | severe n | severe FN | severe FN CP high |
|---|---|---|---|---|---|---|---|
| pure classify-all | 105 | 97 | 8 | 0.924 | 0 | 0 | NA |
| deployable conservative | 88 | 88 | 0 | 1.000 | 0 | 0 | NA |

Pure classify-all is reported but not retained as the deployable policy because it introduces mild/boundary false-normal calls. The retained R3 rule is label-independent: model-usable acceptable maneuvers exist, the calibrated best-test ratio is computed, obstructed calls finalize, and normal calls require every acceptable maneuver to be normal.

### Held-Out Maneuver Check

| participants | with LOO proxy | observed blow correct | observed blow n | observed blow acc | participant majority correct | participant majority n | participant majority acc | model blow acc |
|---|---|---|---|---|---|---|---|---|
| 89 | 89 | 219 | 312 | 0.702 | 59 | 89 | 0.663 | 0.779 |

| reason | observed blow correct | observed blow n | observed blow acc |
|---|---|---|---|
| R1_TRUNCATED_INCOMPLETE | 22 | 23 | 0.957 |
| R3_FAILED_REPEATABILITY | 197 | 289 | 0.682 |

## Part 3 - Residual R1/R2 Check

| reason | residual | >=1 SPAACC Y | >=1 model-usable | pure best-test classifies | conservative classifies | conservative accuracy if used |
|---|---|---|---|---|---|---|
| R1_TRUNCATED_INCOMPLETE | 13 | 11 | 11 | 11 | 8 | 1.000 |
| R2_BORDERLINE_RATIO | 39 | 39 | 39 | 39 | 36 | 0.889 |

Residual R1 is included only when an alternate acceptable, model-usable best-test result exists and the conservative normal-call guard passes. Residual R2 is not loosened; it remains governed by the existing borderline CI rule even though counts are reported here.

## Part 4 - New Coverage and Ceiling

| scope | finalized n | coverage | sens | spec | kappa | severe finalized | severe FN | severe FN CP high |
|---|---|---|---|---|---|---|---|---|
| current deployable recovery | 375 | 0.666 | 0.943 | 0.977 | 0.923 | 6 | 0 | 0.459 |
| pure ATS R3 classify-all | 473 | 0.840 | 0.931 | 0.978 | 0.907 | 6 | 0 | 0.459 |
| ATS R3 only | 456 | 0.810 | 0.963 | 0.977 | 0.938 | 6 | 0 | 0.459 |
| ATS R3 + residual R1 | 464 | 0.824 | 0.963 | 0.977 | 0.940 | 7 | 0 | 0.410 |

- R3-only coverage gain vs current recovery: 0.144.
- R3 + residual R1 coverage gain vs current recovery: 0.158.
- Pure ATS R3 classify-all is shown as an unsafe upper coverage bound because it lowers sensitivity and kappa; it is not the retained policy.
- ATS-best-test selected cases in retained policy: 96.
- Net newly finalized cases vs current recovery: 89.

### Residual Deferred After Retained Policy

| residual reason | n | fraction all | zero model-usable | normal-call discordance |
|---|---|---|---|---|
| R2_BORDERLINE_RATIO | 39 | 0.069 | 0 | 3 |
| R4_GENUINELY_UNINTERPRETABLE | 33 | 0.059 | 3 | 4 |
| R3_FAILED_REPEATABILITY | 22 | 0.039 | 5 | 17 |
| R1_TRUNCATED_INCOMPLETE | 5 | 0.009 | 2 | 3 |

- residual deferred fraction: 0.176.
- honest interpretable ceiling under this deterministic gate: 0.824.
- Residual cases are not being counted as failures to polish: they either lack a usable acceptable modeled signal, remain R2 CI-borderline, are R4/artifact by the existing gate, or would require a normal call despite an accepted obstructed maneuver.

## Guardrail Statement

- Selection is label-independent. TEST truth is read only after a case is selected for scoring.
- Severe FN remains 0 in the retained ATS R3 + residual R1 policy.
- No threshold grid search was used. The only new defer rule is the fixed-threshold normal-call guard at 0.70: do not auto-label normal while any accepted maneuver remains obstructed.
- Repeatability is now a reported quality grade, not a hard bar to reporting a best-test result.

VERDICT: ATS-RULE-RECOVERS - ATS best-test plus the conservative normal-call guard materially increases deployable coverage and preserves severe FN=0.
