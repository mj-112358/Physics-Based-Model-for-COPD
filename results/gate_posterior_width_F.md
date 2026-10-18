# Gate - Posterior Width Thinning + Longer-Memory Error, Cycle F

- VERDICT: IMPROVED-NOT-CLEARED
- best variant: L4_THIN_k20
- message: Best variant L4_THIN_k20 improves 50% coverage but does not clear all calibration criteria.
- posterior classification unblocked: False
- k=1 is the existing gate_likelihood_repair_F control grid: raw curve sampled every 5 native 100 Hz points, i.e. 20 Hz. Larger k thins that grid further.
- GP Matern-3/2 uses a dense covariance on at most 80 evenly-spaced control-grid points per curve for tractability.
- NUTS settings: warmup=250, draws=250, chains=2, convergence R-hat <= 1.1, ESS >= 50.0.

## Variant Summary

| variant | n converged | excluded | points | cov50 | cov80 | cov95 | PIT p | PIT mean | ratio SD med | sigma mL | rho | corr len samples | corr len s | calibrated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| L4_THIN_k1 | 150 | 0 | 222 | 0.140 | 0.280 | 0.407 | 2.43e-73 | 0.297 | 0.0097 | 6.9 | 0.988 | 85.6 | 4.28 | False |
| L4_THIN_k10 | 150 | 0 | 23 | 0.113 | 0.253 | 0.427 | 1.21e-139 | 0.116 | 0.0648 | 175.7 | 0.167 | 0.6 | 0.28 | False |
| L4_THIN_k20 | 148 | 2 | 12 | 0.405 | 0.736 | 0.932 | 2.1e-08 | 0.567 | 0.1824 | 420.4 | 0.365 | 1.0 | 0.99 | False |
| L4_THIN_k5 | 150 | 0 | 45 | 0.073 | 0.153 | 0.233 | 1.37e-187 | 0.080 | 0.0508 | 73.1 | 0.209 | 0.6 | 0.16 | False |
| L4_THIN_k50 | 87 | 63 | 6 | 0.046 | 0.230 | 0.885 | 1.12e-58 | 0.883 | 0.1824 | 652.7 | 0.399 | 1.1 | 2.72 | False |
| M1_AR2 | 150 | 0 | 222 | 0.047 | 0.113 | 0.247 | 1.55e-171 | 0.857 | 0.0092 | 3.3 | nan | nan | nan | False |
| M2_GP_MATERN32 | 150 | 0 | 80 | 0.100 | 0.147 | 0.267 | 4.36e-85 | 0.423 | 0.0653 | 18.3 | nan | nan | nan | False |

- thinning monotone 50% coverage pattern: False

## Longer-Memory Parameters

| variant | rho1 | rho2 | GP amp mL | GP length s | GP length samples | runtime s |
|---|---:|---:|---:|---:|---:|---:|
| M1_AR2 | 1.515 | -0.531 | nan | nan | nan | 462.8 |
| M2_GP_MATERN32 | nan | nan | 420.0 | 5.94 | 50.4 | 2372.1 |

## Row-Level Output

Each row contains one NUTS fit for one participant/curve/variant, with R-hat, ESS, PIT, coverage indicators, posterior ratio quantiles, and likelihood-process parameters.
