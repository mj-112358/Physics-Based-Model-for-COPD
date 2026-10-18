# Gate - Sampler Convergence Establishment, Cycle F

## Protocol Decisions

- Cell A construction is reused from `gate_misspecification_isolation_f.py` with seed 20260710 and target counts 67/67/12/4.
- The ladder varies sampler settings only: chains, warmup, draws, target_accept, dense_mass.
- Coverage intervals are exact binomial Monte Carlo 95% intervals using the attempted row count; complete rungs use n=150 and early-stopped rungs use their partial n.
- tau_f < tau_s is enforced by prior support: tau_f in [0.01, 1.0], tau_s in [1.0, 20.0]. There is no rejection step and no rejection rate.

## Verdict: GEOMETRY-INTRACTABLE

- S_min: None
- Gate A halt retracted: False

## Part 1 - Ladder

| rung | n | complete | chains | warmup | draws | target accept | dense | cov50 | 50 lo | 50 hi | cov80 | cov95 | median Rhat | Rhat>1.01 | Rhat>1.05 | min ESS | median ESS | ESS<400 | div | any div | ratio sd med | sec/curve | converged | nominal | early stop reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 150 | True | 2 | 250 | 250 | NA | False | 0.413 | 0.420 | 0.580 | 0.787 | 0.953 | 1.031 | 0.833 | 0.247 | 2.0 | 75.3 | 1.000 | 251 | 0.207 | 0.0030 | 2.81 | False | False |  |
| S2 | 150 | True | 4 | 500 | 500 | NA | False | 0.473 | 0.420 | 0.580 | 0.780 | 0.953 | 1.015 | 0.640 | 0.193 | 8.0 | 290.2 | 0.780 | 1846 | 0.267 | 0.0029 | 6.50 | False | True |  |
| S3 | 150 | True | 4 | 1000 | 1000 | NA | False | 0.460 | 0.420 | 0.580 | 0.767 | 0.973 | 1.007 | 0.373 | 0.153 | 8.0 | 592.4 | 0.280 | 3695 | 0.293 | 0.0029 | 9.83 | False | True |  |
| S4 | 150 | True | 4 | 2500 | 2500 | NA | False | 0.447 | 0.420 | 0.580 | 0.760 | 0.973 | 1.003 | 0.247 | 0.180 | 8.0 | 1433.8 | 0.213 | 14398 | 0.313 | 0.0029 | 19.41 | False | True |  |
| S5 | 16 | False | 4 | 2500 | 2500 | 0.95 | False | 0.625 | 0.250 | 0.750 | 0.875 | 1.000 | 1.003 | 0.312 | 0.312 | 4.0 | 1543.5 | 0.312 | 62 | 0.125 | 0.0047 | 41.46 | False | False | 5 curves with Rhat>1.01 exceeds allowed 1; 5 curves with ESS<400 exceeds allowed 1 |
| S6 | 4 | False | 4 | 2500 | 2500 | 0.99 | True | 0.000 | 0.000 | 1.000 | 0.750 | 1.000 | 1.015 | 0.500 | 0.250 | 4.0 | 704.4 | 0.500 | 0 | 0.000 | 0.0079 | 44.46 | False | False | 2 curves with Rhat>1.01 exceeds allowed 1; 2 curves with ESS<400 exceeds allowed 1 |

## Part 2 - Divergence Geometry

- baseline condition number median/IQR and S_min condition number are in the ladder table. Mechanism note: S1 posterior condition number median 172667.6; Rhat>1.01 fraction 0.833; divergent transitions 251. The tau_f/tau_s ordering is by disjoint prior support, not rejection.
- divergent transition coordinates are cached under `results/_cache_sampler_convergence/*_divergent_draws.csv` for scatter plotting.

## Part 3 - NUTS Re-Audit

| gate file | result file | chains | warmup | draws | target | dense | below S_min? | headline |
|---|---|---|---|---|---|---|---|---|
| predictx/gate5_coverage.py |  | 1 | NA | NA | NA | False | S_min_not_established |  |
| predictx/gate5_sbc.py |  | 1 | NA | NA | NA | False | S_min_not_established |  |
| predictx/gate_estimand_hybrid_f.py |  | NA | NA | NA | NA | False | S_min_not_established |  |
| predictx/gate_koh_discrepancy_f.py |  | NA | NA | NA | NA | False | S_min_not_established |  |
| predictx/gate_likelihood_repair_f.py |  | 2 | 250 | 250 | NA | False | S_min_not_established |  |
| predictx/gate_misspecification_isolation_f.py |  | NA | NA | NA | NA | False | S_min_not_established |  |
| predictx/gate_model_vs_observed_f.py |  | NA | NA | NA | NA | False | S_min_not_established |  |
| predictx/gate_offset_lungsize_f.py |  | NA | NA | NA | NA | False | S_min_not_established |  |
| predictx/gate_posterior_calibration_repair_f.py |  | 2 | 300 | 300 | NA | False | S_min_not_established |  |
| predictx/gate_posterior_method_f.py |  | NA | NA | NA | NA | False | S_min_not_established |  |
| predictx/gate_posterior_width_f.py |  | NA | NA | NA | NA | False | S_min_not_established |  |
| predictx/gate_reconciliation_audit_f.py |  | NA | NA | NA | NA | False | S_min_not_established |  |
| predictx/gate_reproducibility_volume_f.py |  | NA | NA | NA | NA | False | S_min_not_established |  |
| predictx/gate_sampler_convergence_f.py |  | NA | NA | NA | NA | False | S_min_not_established |  |
| predictx/gate_truncation_recovery_f.py |  | NA | NA | NA | NA | False | S_min_not_established |  |
| predictx/gate_two_path_reconciliation_f.py |  | NA | NA | NA | NA | False | S_min_not_established |  |
| predictx/synthetic_gates.py |  | 4 | 2500 | 2500 | NA | False | S_min_not_established |  |
| predictx/synthetic_stack.py |  | NA | NA | NA | NA | False | S_min_not_established |  |
| predictx/tsnpe.py |  | NA | NA | NA | NA | False | S_min_not_established |  |

Rows marked below S_min are not interpretable as posterior quantities and must be recomputed. This gate does not recompute them.

## Deterministic Results Unaffected

| gate file | exists | touches NUTS | unaffected |
|---|---|---|---|
| predictx/gate_clinical_classification_f.py | True | False | True |
| predictx/gate_fvc_correction_f.py | True | False | True |
| predictx/gate_offset_lungsize_f.py | True | True | False |
| predictx/gate_truncation_recovery_f.py | True | True | False |
| predictx/gate_reproducibility_volume_f.py | True | True | False |
| predictx/gate_estimand_hybrid_f.py | True | True | False |
| predictx/gate23_f.py | True | False | True |
| predictx/gate3_f_rerun.py | True | False | True |
| predictx/gate_model_vs_observed_f.py | True | True | False |

## Part 4 - Cost Projection

- source: last_rung_run
- seconds per curve: 44.46
- Cell A/B/C x five likelihoods, 2250 fits: 27.79 hours
- dose response, 900 fits: 11.12 hours
- likelihood repair, 750 fits: 9.26 hours
