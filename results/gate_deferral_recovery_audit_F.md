# Gate - Deferral Recovery Label-Leakage Audit, Cycle F

VERDICT: DEPLOYABLE-AS-REPORTED - selection is label-independent in the current recovery runner; coverage and accuracy are scored after deployable selection.

## Part 1 - Selection Rule

| bucket | candidate code path | candidate keep rule | candidate label-independent | final keep path | truth usage |
|---|---|---|---|---|---|
| R1 | attempt_recovery -> r1_candidate (predictx/gate_deferral_recovery_f.py:138-150) | volume_r2 >= R1_GOOD_FIT_R2_MIN and corrected_ratio_c1 <0.60 => obstructed; corrected_ratio_c1 >=0.80 => normal | True | apply_recovery label-independent keep rule (predictx/gate_deferral_recovery_f.py:218-259) | current report keeps every non-empty candidate_decision; truth_label(row) is used only after selection for candidate_correct scoring |
| R2 | attempt_recovery -> r2_candidate (predictx/gate_deferral_recovery_f.py:153-169) | bootstrap median CI must sit on one side of 0.70; below-side median <0.67, above-side median >=0.73 | True | apply_recovery label-independent keep rule (predictx/gate_deferral_recovery_f.py:218-259) | current report keeps every non-empty candidate_decision; truth_label(row) is used only after selection for candidate_correct scoring |
| R3 | attempt_recovery -> r3_candidate (predictx/gate_deferral_recovery_f.py:172-204) | good-fit maneuvers volume_r2 >=0.95; cluster width <=0.020; cluster majority; cluster high <0.65 or low >=0.75 | True | apply_recovery label-independent keep rule (predictx/gate_deferral_recovery_f.py:218-259) | current report keeps every non-empty candidate_decision; truth_label(row) is used only after selection for candidate_correct scoring |

R1/R2/R3 candidate generation is deployable: it uses OMBE fit quality, corrected ratio distance, bootstrap CI, and maneuver consistency.

Current keep rule: a candidate is recovered when `candidate_decision` is non-empty. TEST truth is used afterward to populate `candidate_correct` and score the selected cases.

## Part 2 - Deployable Recompute

Deployable keep rule used here: `deferred_baseline and candidate_decision != ""`. The true label is evaluated only after that selection.

| bucket | deferred | deployable finalized | correct | wrong | accuracy | severe finalized | severe FN | severe FN CP high |
|---|---|---|---|---|---|---|---|---|
| R1_TRUNCATED_INCOMPLETE | 38 | 25 | 25 | 0 | 1.000 | 4 | 0 | 0.602 |
| R2_BORDERLINE_RATIO | 127 | 88 | 88 | 0 | 1.000 | 0 | 0 | NA |
| R3_FAILED_REPEATABILITY | 110 | 7 | 7 | 0 | 1.000 | 0 | 0 | NA |
| R4_GENUINELY_UNINTERPRETABLE | 33 | 0 | 0 | 0 | NA | 0 | 0 | NA |

## Overall Metrics

| scope | finalized n | coverage | sens | spec | kappa | severe finalized | severe FN | severe FN CP high |
|---|---|---|---|---|---|---|---|---|
| baseline workflow | 255 | 0.453 | 0.847 | 0.974 | 0.842 | 2 | 0 | 0.842 |
| current recovery runner | 375 | 0.666 | 0.943 | 0.977 | 0.923 | 6 | 0 | 0.459 |
| deployable label-independent recovery | 375 | 0.666 | 0.943 | 0.977 | 0.923 | 6 | 0 | 0.459 |

- deployable recovered cases: 120
- deployable recovered accuracy: 1.000
- current-runner recovered cases: 120
- current-runner recovered accuracy: 1.000

## Part 3 - Honest Restatement

- Label leakage confirmed in current recovery runner: False.
- The recovery runner and the independent deployable recompute both select on the signal-only candidate rules first, then score those cases against TEST truth.
- In this deterministic TEST run, 120 recovered candidates are selected and 0 are wrong.
- The deployable label-independent result is 0.666 coverage with recovered-case accuracy 1.000.
- Severe FN in the deployable finalized set is 0/6 with CP upper 0.459.

VERDICT: DEPLOYABLE-AS-REPORTED - selection is label-independent in the current recovery runner; coverage and accuracy are scored after deployable selection.
