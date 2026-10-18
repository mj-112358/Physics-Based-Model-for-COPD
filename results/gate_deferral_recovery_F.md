# Gate - Deferral Recovery Analysis, Cycle F

VERDICT: RECOVERY-SUBSTANTIAL - coverage rises materially via label-independent recovery rules, final-call metrics hold, and severe FN remains zero.

## Protocol Decisions

- TEST split only; TRAIN rows are ignored.
- The existing workflow gate is reused as the baseline; this gate only analyzes and recovers its deferred rows.
- Primary deferral reason uses a strict priority: invalid/artifact, failed repeatability, truncation, borderline, then residual uninterpretable.
- Volume fit R2 < 0.90 is treated as R4 artifact/no-valid-fit, not as a recoverable borderline case.
- Recovered cases are selected by deterministic, label-independent candidate rules. TEST truth is used only after selection to score correctness.
- No confidence threshold is lowered; R1/R2/R3 recovery uses stricter evidence than the baseline finalization rule.

## Part 1 - Deferral Autopsy

| bucket | n | fraction of deferred | attempted | recovered | post-selection errors |
|---|---|---|---|---|---|
| R1 truncated/incomplete | 38 | 0.123 | 25 | 25 | 0 |
| R2 borderline near 0.70 | 127 | 0.412 | 88 | 88 | 0 |
| R3 failed repeatability | 110 | 0.357 | 7 | 7 | 0 |
| R4 uninterpretable/artifact | 33 | 0.107 | 0 | 0 | 0 |

## Parts 2-4 - Recovery Attempts

| reason | deferred | attempted | correct | errors | moved finalized | post-selection accuracy | severe FN | severe FN CP high | residual |
|---|---|---|---|---|---|---|---|---|---|
| R1_TRUNCATED_INCOMPLETE | 38 | 25 | 25 | 0 | 25 | 1.000 | 0 | 0.602 | 13 |
| R2_BORDERLINE_RATIO | 127 | 88 | 88 | 0 | 88 | 1.000 | 0 | NA | 39 |
| R3_FAILED_REPEATABILITY | 110 | 7 | 7 | 0 | 7 | 1.000 | 0 | NA | 103 |
| R4_GENUINELY_UNINTERPRETABLE | 33 | 0 | 0 | 0 | 0 | NA | 0 | NA | 33 |

R1 recovery uses high-quality OMBE reconstruction only. R2 recovery uses multi-maneuver bootstrap median CIs. R3 recovery requires a majority subset of good-fit maneuvers that is mutually consistent and far from 0.70. Candidate keep/defer decisions are made before looking at TEST truth; correctness is scored only after selection.

### R2 Resolution Split

- resolvable by tighter multi-maneuver/bootstrap CI: 88
- truly ambiguous / CI not safely one-sided: 39

### R3 Consistency Split

- one-bad-blow / recoverable subset found: 7
- all-inconsistent or no majority good-fit subset: 103

## Part 5 - Honest New Coverage

| scope | finalized n | coverage | sens | spec | kappa | severe finalized | severe FN | severe FN CP high |
|---|---|---|---|---|---|---|---|---|
| baseline workflow | 255 | 0.453 | 0.847 | 0.974 | 0.842 | 2 | 0 | 0.842 |
| post recovery | 375 | 0.666 | 0.943 | 0.977 | 0.923 | 6 | 0 | 0.459 |

- coverage delta: 0.213
- net recovered cases: 120
- residual deferred fraction: 0.334
- honest interpretable ceiling in this TEST gate: 0.666
- genuinely uninterpretable/artifact residual fraction of all TEST rows: 0.059

## Residual Deferred

| residual reason | n | fraction all | fraction residual |
|---|---|---|---|
| R3_FAILED_REPEATABILITY | 103 | 0.183 | 0.548 |
| R2_BORDERLINE_RATIO | 39 | 0.069 | 0.207 |
| R4_GENUINELY_UNINTERPRETABLE | 33 | 0.059 | 0.176 |
| R1_TRUNCATED_INCOMPLETE | 13 | 0.023 | 0.069 |

## Guardrail Statement

- Severe FN=0 is preserved in the post-recovery finalized set.
- Recovery does not lower the confidence threshold. Every moved case is moved by a specific validating signal available at inference time.
- TEST truth is used only for scoring the already-selected cases, not for deciding whether to keep a candidate.
- Remaining ambiguous/inconsistent/artifact cases stay deferred.

VERDICT: RECOVERY-SUBSTANTIAL - coverage rises materially via label-independent recovery rules, final-call metrics hold, and severe FN remains zero.
