# Gate - Deterministic Workflow Policy, Cycle F

VERDICT: WORKFLOW-RECOVERED - held-out severe accepted-normal FN is zero, borderline accepted-normal FN is <=0.10, and normal non-accept burden is <=0.35.

## Protocol Decisions

- Cycle F is the first workflow target because the existing calibrated policy artifacts are strongest there.
- The headline evaluation uses held-out TEST rows from the existing policy gate; all rows are reported secondarily.
- No NUTS, SNPE, posterior draws, refits, threshold movement, or dashboard code is used.
- Only ACCEPT_NORMAL is counted as a false-negative workflow failure for obstructed truth; repeat/referral/unusable are safe non-final actions.
- Predicted severe-range corrected ratio <0.50 is routed to referral instead of accepted as a routine obstructed call.
- Borderline corrected ratios in [0.60,0.72), raw/corrected threshold disagreement, and ATS-truncated normal calls are routed to repeat.
- Repeated-maneuver instability near the decision boundary routes to referral/unstable rather than a confident diagnosis.
- The stopping audit is explicitly a deterministic stability proxy, not Bayesian expected information gain.

## Inputs

| artifact | rows | role |
|---|---|---|
| results/gate_policy_calibrated_flag_F_rows.csv | 1125 | bias-corrected selected-maneuver policy sample |
| results/gate_reproducibility_volume_F_rows.csv | 30593 | cross-maneuver repeatability and stopping proxy |

## Workflow Rule

| priority | condition | output |
|---:|---|---|
| 1 | missing/nonfinite fit or ratio | UNUSABLE |
| 2 | corrected ratio <0.50 | REFER_SEVERE_OR_UNSTABLE |
| 3 | near threshold and repeated-maneuver instability or ratio SD >=0.030 | REFER_SEVERE_OR_UNSTABLE |
| 4 | ATS truncation flag and corrected ratio <0.70 | TRUNCATED_RECOVERED |
| 5 | ATS truncation flag with non-obstructed corrected ratio | BORDERLINE_REPEAT |
| 6 | raw/corrected ratio crosses 0.70, or corrected ratio in [0.60,0.72) | BORDERLINE_REPEAT |
| 7 | corrected ratio <0.60 | ACCEPT_OBSTRUCTED |
| 8 | otherwise | ACCEPT_NORMAL |

## Held-Out TEST Workflow Rates

| metric | rate | num | den | exact 95% CI |
|---|---|---|---|---|
| accepted_normal_fn_all_obstructed | 0.029 | 9 | 313 | 0.013-0.054 |
| accepted_normal_fn_severe | 0.000 | 0 | 16 | 0.000-0.206 |
| accepted_normal_fn_moderate | 0.021 | 1 | 47 | 0.001-0.113 |
| accepted_normal_fn_borderline | 0.032 | 8 | 250 | 0.014-0.062 |
| normal_non_accept_burden | 0.236 | 59 | 250 | 0.185-0.294 |
| final_diagnostic_call_rate | 0.453 | 255 | 563 | 0.411-0.495 |
| repeat_referral_unusable_rate | 0.547 | 308 | 563 | 0.505-0.589 |

Held-out bootstrap checks:

| metric | point | boot low | boot high | n | B |
|---|---|---|---|---|---|
| accepted_normal_fn_all_obstructed | 0.029 | 0.013 | 0.048 | 313 | 2000 |
| accepted_normal_fn_borderline | 0.032 | 0.012 | 0.056 | 250 | 2000 |
| normal_non_accept_burden | 0.236 | 0.184 | 0.288 | 250 | 2000 |
| final_diagnostic_call_rate | 0.453 | 0.412 | 0.494 | 563 | 2000 |

## Held-Out Decision Distribution

| decision | n | fraction |
|---|---|---|
| ACCEPT_NORMAL | 200 | 0.355 |
| ACCEPT_OBSTRUCTED | 29 | 0.052 |
| BORDERLINE_REPEAT | 165 | 0.293 |
| TRUNCATED_RECOVERED | 26 | 0.046 |
| REFER_SEVERE_OR_UNSTABLE | 143 | 0.254 |
| UNUSABLE | 0 | 0.000 |

## Final Diagnostic Calls Only

This excludes `BORDERLINE_REPEAT`, `REFER_SEVERE_OR_UNSTABLE`, and `UNUSABLE`; those are safe workflow actions, not final normal/obstructed diagnoses.

| n | TP | TN | FP | FN | sens | spec | kappa |
|---|---|---|---|---|---|---|---|
| 255 | 50 | 191 | 5 | 9 | 0.847 | 0.974 | 0.842 |

## All Policy Rows

| metric | rate | num | den | exact 95% CI |
|---|---|---|---|---|
| accepted_normal_fn_all_obstructed | 0.022 | 14 | 625 | 0.012-0.037 |
| accepted_normal_fn_severe | 0.000 | 0 | 31 | 0.000-0.112 |
| accepted_normal_fn_moderate | 0.021 | 2 | 94 | 0.003-0.075 |
| accepted_normal_fn_borderline | 0.024 | 12 | 500 | 0.012-0.042 |
| normal_non_accept_burden | 0.230 | 115 | 500 | 0.194-0.269 |
| final_diagnostic_call_rate | 0.450 | 506 | 1125 | 0.420-0.479 |
| repeat_referral_unusable_rate | 0.550 | 619 | 1125 | 0.521-0.580 |

## Deterministic Stopping Proxy

This is not Bayesian EIG. It is a deterministic value-of-additional-maneuver proxy using first/second/third maneuver stability from existing volume-domain fits.

| participants | >=3 maneuvers | mean stop k | mean stop k, >=3 | saved vs 3 |
|---|---|---|---|---|
| 6664 | 6389 | 1.82 | 1.83 | 1.17 |

| metric | rate | num | den | CI low | CI high |
|---|---|---|---|---|---|
| stopping_accepted_normal_fn_all_obstructed | 0.027 | 16 | 597 | 0.015 | 0.043 |
| stopping_accepted_normal_fn_severe | 0.000 | 0 | 25 | 0.000 | 0.137 |
| stopping_normal_non_accept_burden | 0.178 | 1079 | 6067 | 0.168 | 0.188 |

Stopping decision counts: {'ACCEPT_NORMAL': 5004, 'BORDERLINE_REPEAT': 1206, 'REFER_SEVERE_OR_UNSTABLE': 291, 'ACCEPT_OBSTRUCTED': 162, 'UNUSABLE': 1}

## Proposal Recovery Statement

- Bayesian posterior uncertainty remains blocked; this gate makes no posterior claim.
- The triage component is now explicit as a deterministic policy over validated OMBE, ATS truncation, and repeatability signals.
- The stopping component is attemptable and now has a deterministic proxy audit; it should be framed as value-of-additional-maneuver stability, not full Bayesian expected information gain.

VERDICT: WORKFLOW-RECOVERED - held-out severe accepted-normal FN is zero, borderline accepted-normal FN is <=0.10, and normal non-accept burden is <=0.35.
