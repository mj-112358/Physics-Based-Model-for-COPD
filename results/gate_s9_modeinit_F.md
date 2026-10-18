# Gate - S9 Mode-Initialized Micro-Pilot, Cycle F

## Protocol Decisions

- S9 reuses the exact S8 20-curve selection and sampler settings.
- The only sampler change is per-chain initialization from a Gaussian ball around the stored deterministic LM fit.
- LM fits are loaded from `results/gate_policy_calibrated_flag_F_rows.csv` using each synthetic row's `template_raw_index`; no LM refit is performed.
- Initialization sd is 5% of each wide-prior natural-parameter range and values are clipped to prior support.
- Per-curve convergence requires R-hat <= 1.01, ESS >= 400, and zero divergent transitions.

## Verdict: INIT-INSUFFICIENT

- realized stratum counts: {'mild_0.60_0.70': 9, 'normal': 9, 'moderate': 1, 'severe': 1}
- LM fallback count: 0
- overall pass: 7 / 20 (0.350)
- S8 passed on same curves: 9 / 20; S9 delta: -2
- S8 observed failures now passing: 3 / 11
- frozen chains total S8 -> S9: 8 -> 17
- kappa < 50000 pass: 3 / 10 (0.300)
- kappa >= 50000 pass: 4 / 10 (0.400)

## Curves Sorted By Fisher Kappa

| curve | severity | Fisher kappa | R-hat | ESS | div | sec | frozen | init | pass |
|---|---|---|---|---|---|---|---|---|---|
| 141 | moderate | 630.8 | 1.000 | 8000.0 | 0 | 16.50 | 0 | lm_ball | True |
| 34 | mild_0.60_0.70 | 1019.9 | 623.186 | 8.0 | 66 | 67.77 | 1 | lm_ball | False |
| 149 | severe | 4538.4 | 1.001 | 1704.1 | 3 | 10.31 | 0 | lm_ball | False |
| 114 | normal | 6857.7 | 1.002 | 1445.9 | 2 | 11.02 | 0 | lm_ball | False |
| 106 | normal | 6950.8 | 1.026 | 645.2 | 5 | 20.78 | 0 | lm_ball | False |
| 78 | normal | 11346.1 | 1.003 | 961.5 | 0 | 12.67 | 0 | lm_ball | True |
| 59 | mild_0.60_0.70 | 27170.6 | 1.002 | 1442.2 | 0 | 10.28 | 0 | lm_ball | True |
| 57 | mild_0.60_0.70 | 29500.3 | 568.286 | 4.0 | 2 | 93.18 | 2 | lm_ball | False |
| 25 | mild_0.60_0.70 | 31193.2 | 16865.198 | 2.0 | 0 | 184.15 | 2 | lm_ball | False |
| 66 | mild_0.60_0.70 | 35610.8 | 401.288 | 4.0 | 0 | 117.53 | 2 | lm_ball | False |
| 43 | mild_0.60_0.70 | 88790.5 | 1073.386 | 2.7 | 0 | 136.94 | 3 | lm_ball | False |
| 72 | normal | 92407.6 | 1.002 | 588.0 | 0 | 30.69 | 0 | lm_ball | True |
| 120 | normal | 92800.8 | 1.001 | 1331.9 | 0 | 16.15 | 0 | lm_ball | True |
| 8 | mild_0.60_0.70 | 96363.5 | 442.551 | 8.0 | 0 | 46.84 | 1 | lm_ball | False |
| 111 | normal | 154435.3 | 405.771 | 4.0 | 0 | 106.01 | 2 | lm_ball | False |
| 22 | mild_0.60_0.70 | 265450.0 | 1.005 | 610.6 | 0 | 20.91 | 0 | lm_ball | True |
| 109 | normal | 449200.6 | 1.008 | 525.6 | 20 | 15.53 | 0 | lm_ball | False |
| 100 | normal | 1397887.9 | 1.004 | 1195.7 | 1 | 47.56 | 0 | lm_ball | False |
| 18 | mild_0.60_0.70 | 2555071.2 | 3836.904 | 2.0 | 0 | 139.67 | 4 | lm_ball | False |
| 112 | normal | 7579683.3 | 1.006 | 760.1 | 0 | 15.20 | 0 | lm_ball | True |

## Critical S8 vs S9 Comparison

| curve | severity | S8 pass | S9 pass | S8 frozen | S9 frozen | declared S8 fail | observed S8 fail |
|---|---|---|---|---|---|---|---|
| 8 | mild_0.60_0.70 | True | False | 0 | 1 | False | False |
| 18 | mild_0.60_0.70 | False | False | 1 | 4 | True | True |
| 22 | mild_0.60_0.70 | True | True | 0 | 0 | False | False |
| 25 | mild_0.60_0.70 | True | False | 0 | 2 | False | False |
| 34 | mild_0.60_0.70 | True | False | 0 | 1 | False | False |
| 43 | mild_0.60_0.70 | True | False | 0 | 3 | False | False |
| 57 | mild_0.60_0.70 | False | False | 0 | 2 | True | True |
| 59 | mild_0.60_0.70 | True | True | 0 | 0 | False | False |
| 66 | mild_0.60_0.70 | True | False | 0 | 2 | False | False |
| 72 | normal | True | True | 0 | 0 | False | False |
| 78 | normal | True | True | 0 | 0 | False | False |
| 100 | normal | False | False | 0 | 0 | False | True |
| 106 | normal | False | False | 1 | 0 | True | True |
| 109 | normal | False | False | 0 | 0 | True | True |
| 111 | normal | False | False | 0 | 2 | True | True |
| 112 | normal | False | True | 0 | 0 | True | True |
| 114 | normal | False | False | 1 | 0 | True | True |
| 120 | normal | False | True | 2 | 0 | True | True |
| 141 | moderate | False | True | 3 | 0 | True | True |
| 149 | severe | False | False | 0 | 0 | True | True |

## Coverage

| sample | n | cov50 | 50 lo | 50 hi | cov80 | 80 lo | 80 hi | cov95 | 95 lo | 95 hi |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 20 | 0.450 | 0.300 | 0.700 | 0.800 | 0.600 | 0.950 | 0.950 | 0.850 | 1.000 |
| converged_only | 7 | 0.143 | 0.143 | 0.857 | 0.714 | 0.429 | 1.000 | 1.000 | 0.714 | 1.000 |

## Severity Pass Fractions

| severity | n | passed | fraction | conclude? |
|---|---|---|---|---|
| mild_0.60_0.70 | 9 | 2 | 0.222 | True |
| moderate | 1 | 1 | 1.000 | False |
| normal | 9 | 4 | 0.444 | True |
| severe | 1 | 0 | 0.000 | False |

Trace dump for still-failing curves: `results/gate_s9_modeinit_F_traces.csv`.
