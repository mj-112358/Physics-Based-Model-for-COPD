# Gate - Sampler Convergence Pilot, Cycle F

## Protocol Decisions

- S1-S4 were carried forward as standing facts and were not rerun.
- S7 was run first, then S6. Both run all 30 pilot curves; there is no population-fraction early stopping.
- Exact cached S6 curves from the previous run are reused only when curve id, sampler settings, Cell A construction, and seed match.
- Divergence is judged by transition fraction: total divergent post-warmup draws divided by total post-warmup draws.

## Verdict: GEOMETRY-INTRACTABLE-CONFIRMED

- S_min: None
- Stop reason: Dense mass did not resolve the convergence failure at either budget.

## Part 1 - Pilot Ladder

| rung | n | chains | warmup | draws | target | dense | cov50 | 50 lo | 50 hi | cov80 | 80 lo | 80 hi | cov95 | 95 lo | 95 hi | PIT KS | PIT p | median Rhat | Rhat>1.01 | min ESS | median ESS | ESS<400 | div frac | any div curve | sec/curve | converged | coverage nominal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S7 | 30 | 4 | 1000 | 1000 | 0.95 | True | 0.600 | 0.333 | 0.667 | 0.800 | 0.633 | 0.933 | 1.000 | 0.867 | 1.000 | 0.159 | 0.390 | 1.010 | 0.500 | 4.0 | 351.0 | 0.567 | 0.000525 | 0.400 | 12.48 | False | True |
| S6 | 30 | 4 | 2500 | 2500 | 0.99 | True | 0.567 | 0.333 | 0.667 | 0.867 | 0.633 | 0.933 | 0.900 | 0.867 | 1.000 | 0.259 | 0.029 | 218.786 | 0.633 | 2.0 | 8.0 | 0.633 | 0.000000 | 0.000 | 60.54 | False | True |


S7 exact per-curve pass under the conjunctive rule `max_rhat <= 1.01 AND min_bulk_ess >= 400 AND divergence_count == 0`: 7/30 = 0.233. The previous approximate 45% reading is not the conjunctive pass rate; it matches the marginal ESS>=400 fraction (13/30 = 0.433).

## Part 2 - Condition Number Reconciliation

Gate 1 kappa is the condition number of the Fisher information matrix J^T J / sigma^2 for OMBE flow on the transformed phi scale, using sigma=0.10 L/s. The convergence-gate 172,668 number is the condition number of the empirical posterior covariance of S1 NUTS draws in phi under the volume likelihood with fixed 50 mL sigma and real FET grids. They are different quantities; the Fisher-information kappa belongs in the paper as the sloppiness measure. The 50-row table uses the first 50 Cell A theta from the wide, stratified Cell A prior, so its Fisher median is not expected to equal Gate 1's original 1000-draw narrow-prior median of 1990.

| curve | severity | Fisher kappa | S1 posterior cov kappa |
|---|---|---|---|
| 0 | mild_0.60_0.70 | 966397.3 | 14540.6 |
| 1 | mild_0.60_0.70 | 223086.5 | 18753.5 |
| 2 | mild_0.60_0.70 | 3027845.0 | 511652.8 |
| 3 | mild_0.60_0.70 | 577993.7 | 130664.1 |
| 4 | mild_0.60_0.70 | 9678.4 | 75279.8 |
| 5 | mild_0.60_0.70 | 14655.2 | 210210.1 |
| 6 | mild_0.60_0.70 | 19036.3 | 310033.0 |
| 7 | mild_0.60_0.70 | 1424.6 | 264651.9 |
| 8 | mild_0.60_0.70 | 96363.5 | 3931670.8 |
| 9 | mild_0.60_0.70 | 4820.2 | 42778.2 |
| 10 | mild_0.60_0.70 | 3659.8 | 62631.7 |
| 11 | mild_0.60_0.70 | 78931.8 | 106914.2 |
| 12 | mild_0.60_0.70 | 3946057.4 | 547331.9 |
| 13 | mild_0.60_0.70 | 83980.3 | 695299.1 |
| 14 | mild_0.60_0.70 | 962652.9 | 51837.8 |
| 15 | mild_0.60_0.70 | 8904.8 | 500935.9 |
| 16 | mild_0.60_0.70 | 92430.5 | 1455143.5 |
| 17 | mild_0.60_0.70 | 52756.7 | 161293.7 |
| 18 | mild_0.60_0.70 | 2555071.2 | 3190375.5 |
| 19 | mild_0.60_0.70 | 650253.3 | 168089.9 |
| 20 | mild_0.60_0.70 | 15439.4 | 174026.5 |
| 21 | mild_0.60_0.70 | 110606.8 | 476542.8 |
| 22 | mild_0.60_0.70 | 265450.0 | 130294.9 |
| 23 | mild_0.60_0.70 | 145537.5 | 107289.7 |
| 24 | mild_0.60_0.70 | 11679.8 | 60504.3 |
| 25 | mild_0.60_0.70 | 31193.2 | 122084.1 |
| 26 | mild_0.60_0.70 | 7623.0 | 14330.2 |
| 27 | mild_0.60_0.70 | 24710.1 | 194886.7 |
| 28 | mild_0.60_0.70 | 359641.8 | 133234.1 |
| 29 | mild_0.60_0.70 | 738403.3 | 1767127.9 |
| 30 | mild_0.60_0.70 | 8695.0 | 43039.1 |
| 31 | mild_0.60_0.70 | 34453.3 | 242479.5 |
| 32 | mild_0.60_0.70 | 2077941.9 | 436244.9 |
| 33 | mild_0.60_0.70 | 2385.5 | 31625.3 |
| 34 | mild_0.60_0.70 | 1019.9 | 214303.9 |
| 35 | mild_0.60_0.70 | 43620.5 | 69535.7 |
| 36 | mild_0.60_0.70 | 3267.9 | 137562.1 |
| 37 | mild_0.60_0.70 | 60090.5 | 1771025.1 |
| 38 | mild_0.60_0.70 | 126953.2 | 370220.8 |
| 39 | mild_0.60_0.70 | 27793.3 | 397051.4 |
| 40 | mild_0.60_0.70 | 3495.5 | 345897.6 |
| 41 | mild_0.60_0.70 | 7128.8 | 137607.4 |
| 42 | mild_0.60_0.70 | 172790.6 | 38916.2 |
| 43 | mild_0.60_0.70 | 88790.5 | 32289.4 |
| 44 | mild_0.60_0.70 | 66526.4 | 509732.2 |
| 45 | mild_0.60_0.70 | 96347.9 | 1849199.3 |
| 46 | mild_0.60_0.70 | 13225.9 | 12985.2 |
| 47 | mild_0.60_0.70 | 103046.7 | 435658.7 |
| 48 | mild_0.60_0.70 | 336620.4 | 2209.4 |
| 49 | mild_0.60_0.70 | 653423.0 | 204377.2 |

## Part 3 - Corrected Audit

UNAFFECTED:

- predictx/gate_clinical_classification_f.py
- predictx/gate_estimand_hybrid_f.py
- predictx/gate_four_part_fix_f.py
- predictx/gate_fvc_correction_f.py
- predictx/gate_model_vs_observed_f.py
- predictx/gate_offset_lungsize_f.py
- predictx/gate_policy_calibrated_flag_f.py
- predictx/gate_posterior_method_f.py
- predictx/gate_reclassification_fixed_integration_f.py
- predictx/gate_reconciliation_audit_f.py
- predictx/gate_reproducibility_volume_f.py
- predictx/gate_truncation_recovery_f.py
- predictx/gate_two_path_reconciliation_f.py

INVALIDATED:

- none

NUTS-DEPENDENT PENDING S_MIN:

- predictx/gate5_coverage.py
- predictx/gate5_sbc.py
- predictx/gate_koh_discrepancy_f.py
- predictx/gate_likelihood_repair_f.py
- predictx/gate_misspecification_isolation_f.py
- predictx/gate_posterior_calibration_repair_f.py
- predictx/gate_posterior_width_f.py
- predictx/gate_sampler_convergence_f.py
- predictx/gate_sampler_pilot_f.py
- predictx/synthetic_gates.py
- predictx/tsnpe.py

| gate file | result file | NUTS | MCMC | chains | warmup | draws | deterministic header | status | reason |
|---|---|---|---|---|---|---|---|---|---|
| predictx/gate5_coverage.py |  | True | True | NA | NA | NA | False | NUTS_DEPENDENT_PENDING_S_MIN | NUTS is instantiated; S_min not established |
| predictx/gate5_sbc.py |  | True | True | NA | NA | NA | False | NUTS_DEPENDENT_PENDING_S_MIN | NUTS is instantiated; S_min not established |
| predictx/gate_clinical_classification_f.py | results/gate_clinical_classification_F.json | False | False | NA | NA | NA | False | UNAFFECTED | no NUTS constructor call in AST |
| predictx/gate_estimand_hybrid_f.py | results/gate_estimand_hybrid_F.json | False | False | NA | NA | NA | True | UNAFFECTED | no NUTS constructor call in AST |
| predictx/gate_four_part_fix_f.py |  | False | False | NA | NA | NA | False | UNAFFECTED | no NUTS constructor call in AST |
| predictx/gate_fvc_correction_f.py | results/gate_fvc_correction_F.json | False | False | NA | NA | NA | False | UNAFFECTED | no NUTS constructor call in AST |
| predictx/gate_koh_discrepancy_f.py | results/gate_koh_discrepancy_F.json | True | True | NA | NA | NA | False | NUTS_DEPENDENT_PENDING_S_MIN | NUTS is instantiated; S_min not established |
| predictx/gate_likelihood_repair_f.py | results/gate_likelihood_repair_F.json | True | True | 2 | 250 | 250 | False | NUTS_DEPENDENT_PENDING_S_MIN | NUTS is instantiated; S_min not established |
| predictx/gate_misspecification_isolation_f.py | results/gate_misspecification_isolation_F.json | True | True | NA | NA | NA | False | NUTS_DEPENDENT_PENDING_S_MIN | NUTS is instantiated; S_min not established |
| predictx/gate_model_vs_observed_f.py | results/gate_model_vs_observed_F.json | False | False | NA | NA | NA | True | UNAFFECTED | no NUTS constructor call in AST |
| predictx/gate_offset_lungsize_f.py | results/gate_offset_lungsize_F.json | False | False | NA | NA | NA | True | UNAFFECTED | no NUTS constructor call in AST |
| predictx/gate_policy_calibrated_flag_f.py | results/gate_policy_calibrated_flag_F.json | False | False | NA | NA | NA | False | UNAFFECTED | no NUTS constructor call in AST |
| predictx/gate_posterior_calibration_repair_f.py | results/gate_posterior_calibration_repair_F.json | True | True | 2 | 300 | 300 | False | NUTS_DEPENDENT_PENDING_S_MIN | NUTS is instantiated; S_min not established |
| predictx/gate_posterior_method_f.py | results/gate_posterior_method_F.json | False | False | NA | NA | NA | False | UNAFFECTED | no NUTS constructor call in AST |
| predictx/gate_posterior_width_f.py | results/gate_posterior_width_F.json | True | True | NA | NA | NA | False | NUTS_DEPENDENT_PENDING_S_MIN | NUTS is instantiated; S_min not established |
| predictx/gate_reclassification_fixed_integration_f.py | results/gate_reclassification_fixed_integration_F.json | False | False | NA | NA | NA | False | UNAFFECTED | no NUTS constructor call in AST |
| predictx/gate_reconciliation_audit_f.py | results/gate_reconciliation_audit_F.json | False | False | NA | NA | NA | False | UNAFFECTED | no NUTS constructor call in AST |
| predictx/gate_reproducibility_volume_f.py | results/gate_reproducibility_volume_F.json | False | False | NA | NA | NA | True | UNAFFECTED | no NUTS constructor call in AST |
| predictx/gate_sampler_convergence_f.py | results/gate_sampler_convergence_F.json | True | True | NA | NA | NA | False | NUTS_DEPENDENT_PENDING_S_MIN | NUTS is instantiated; S_min not established |
| predictx/gate_sampler_pilot_f.py | results/gate_sampler_pilot_F.json | True | False | NA | NA | NA | False | NUTS_DEPENDENT_PENDING_S_MIN | NUTS is instantiated; S_min not established |
| predictx/gate_truncation_recovery_f.py | results/gate_truncation_recovery_F.json | False | False | NA | NA | NA | True | UNAFFECTED | no NUTS constructor call in AST |
| predictx/gate_two_path_reconciliation_f.py | results/gate_two_path_reconciliation_F.json | False | False | NA | NA | NA | False | UNAFFECTED | no NUTS constructor call in AST |
| predictx/synthetic_gates.py |  | True | True | 4 | 2500 | 2500 | False | NUTS_DEPENDENT_PENDING_S_MIN | NUTS is instantiated; S_min not established |
| predictx/tsnpe.py |  | True | True | NA | NA | NA | False | NUTS_DEPENDENT_PENDING_S_MIN | NUTS is instantiated; S_min not established |
