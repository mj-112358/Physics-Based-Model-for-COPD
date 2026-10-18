# Gate - S8 Micro-Pilot, Cycle F

## Protocol Decisions

- S8 is identical to S7 except warmup/draws are 2000/2000 instead of 1000/1000.
- The 20 curves are sampled across Cell A stratum blocks at target proportions 9 mild / 9 normal / 1 moderate / 1 severe.
- Per-curve convergence requires R-hat <= 1.01, ESS >= 400, and zero divergent transitions.
- Coverage is reported at n=20 and again after excluding non-converged curves; n=1 severity strata are reported but not interpreted.

## Verdict: BUDGET-INSUFFICIENT

- realized stratum counts: {'mild_0.60_0.70': 9, 'normal': 9, 'moderate': 1, 'severe': 1}
- overall pass: 9 / 20 (0.450)
- kappa < 50000 pass: 5 / 10 (0.500)
- kappa >= 50000 pass: 4 / 10 (0.400)
- best kappa split: {'cutoff': 92800.7557741458, 'correct': 13, 'low_fail_exceptions': 5, 'high_pass_exceptions': 2}
- wall-clock seconds per curve median/IQR: 14.26 / [10.321554301772267, 27.97414502117317]
- failure mechanism: Mixed frozen-chain non-mixing: 0.500 of worst-curve chains have adapted step size <1e-5, 0.500 hit max treedepth on nearly all transitions, and per-chain parameter means are widely separated.

## Curves Sorted By Fisher Kappa

| curve | severity | Fisher kappa | R-hat | ESS | div | div frac | sec | pass |
|---|---|---|---|---|---|---|---|---|
| 141 | moderate | 630.8 | 1177.554 | 2.7 | 0 | 0.000000 | 140.04 | False |
| 34 | mild_0.60_0.70 | 1019.9 | 1.002 | 1381.9 | 0 | 0.000000 | 9.08 | True |
| 149 | severe | 4538.4 | 1.002 | 1653.6 | 3 | 0.000375 | 10.29 | False |
| 114 | normal | 6857.7 | 103.296 | 8.0 | 0 | 0.000000 | 43.93 | False |
| 106 | normal | 6950.8 | 303.737 | 8.0 | 20 | 0.002500 | 39.74 | False |
| 78 | normal | 11346.1 | 1.004 | 990.9 | 0 | 0.000000 | 10.33 | True |
| 59 | mild_0.60_0.70 | 27170.6 | 1.001 | 1522.6 | 0 | 0.000000 | 7.79 | True |
| 57 | mild_0.60_0.70 | 29500.3 | 1.017 | 632.7 | 16 | 0.002000 | 11.41 | False |
| 25 | mild_0.60_0.70 | 31193.2 | 1.001 | 4536.4 | 0 | 0.000000 | 5.90 | True |
| 66 | mild_0.60_0.70 | 35610.8 | 1.001 | 1114.8 | 0 | 0.000000 | 21.36 | True |
| 43 | mild_0.60_0.70 | 88790.5 | 1.002 | 1643.6 | 0 | 0.000000 | 7.18 | True |
| 72 | normal | 92407.6 | 1.003 | 567.8 | 0 | 0.000000 | 24.05 | True |
| 120 | normal | 92800.8 | 735.352 | 4.0 | 0 | 0.000000 | 75.49 | False |
| 8 | mild_0.60_0.70 | 96363.5 | 1.002 | 798.8 | 0 | 0.000000 | 18.58 | True |
| 111 | normal | 154435.3 | 1.026 | 414.1 | 5 | 0.000625 | 15.41 | False |
| 22 | mild_0.60_0.70 | 265450.0 | 1.004 | 641.0 | 0 | 0.000000 | 18.72 | True |
| 109 | normal | 449200.6 | 1.016 | 983.7 | 4 | 0.000500 | 12.59 | False |
| 100 | normal | 1397887.9 | 1.012 | 954.5 | 6 | 0.000750 | 10.99 | False |
| 18 | mild_0.60_0.70 | 2555071.2 | 606.232 | 8.0 | 0 | 0.000000 | 52.73 | False |
| 112 | normal | 7579683.3 | 1.013 | 735.6 | 5 | 0.000625 | 13.11 | False |

## Coverage

| sample | n | cov50 | 50 lo | 50 hi | cov80 | 80 lo | 80 hi | cov95 | 95 lo | 95 hi |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 20 | 0.500 | 0.300 | 0.700 | 0.900 | 0.600 | 0.950 | 1.000 | 0.850 | 1.000 |
| converged_only | 9 | 0.444 | 0.222 | 0.778 | 0.778 | 0.556 | 1.000 | 1.000 | 0.778 | 1.000 |

## Severity Pass Fractions

| severity | n | passed | fraction | conclude? |
|---|---|---|---|---|
| mild_0.60_0.70 | 9 | 7 | 0.778 | True |
| moderate | 1 | 0 | 0.000 | False |
| normal | 9 | 2 | 0.222 | True |
| severe | 1 | 0 | 0.000 | False |

## Worst-Rhat Failure Instrumentation

- worst curve ids: [141, 120, 18]
- worst-curve chains with adapted step size <1e-5: 0.500
- worst-curve chains hitting max treedepth on nearly all transitions: 0.500
- `results/gate_s8_micropilot_F_traces.csv` contains curve, chain, param, mean, sd, rhat_contrib, adapted step size, mean accept probability, divergences, and tree-depth hit fraction.

| curve | chain | step size | accept | div | max depth frac |
|---|---|---|---|---|---|
| 18 | 0 | 0.020461 | 0.948 | 0 | 0.000 |
| 18 | 1 | 0.016159 | 0.940 | 0 | 0.000 |
| 18 | 2 | 0.000000 | 0.953 | 0 | 1.000 |
| 18 | 3 | 0.025027 | 0.954 | 0 | 0.000 |
| 120 | 0 | 0.000000 | 0.964 | 0 | 1.000 |
| 120 | 1 | 0.074491 | 0.962 | 0 | 0.000 |
| 120 | 2 | 0.000000 | 0.947 | 0 | 1.000 |
| 120 | 3 | 0.056627 | 0.977 | 0 | 0.000 |
| 141 | 0 | 0.000000 | 0.954 | 0 | 1.000 |
| 141 | 1 | 0.505039 | 0.948 | 0 | 0.000 |
| 141 | 2 | 0.000000 | 0.959 | 0 | 1.000 |
| 141 | 3 | 0.000000 | 0.949 | 0 | 1.000 |
