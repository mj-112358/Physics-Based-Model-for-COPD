# Gate 1 - Sloppiness

Synthetic-only OMBE forward model:

`q(t) = [alpha exp(-t/tau_f)/tau_f + (1-alpha) exp(-t/tau_s)/tau_s] * [1 - exp(-t/tau_o)]`

`F(t) = FVC * q(t) / integral(q)` on a 12 s, 100 Hz grid.

- status: PASS
- interpretation: sloppiness confirmed, Bayesian inference justified
- flow noise sigma: 0.1 L/s
- midrange integral: 4.000000 L
- midrange FVC: 4.000000 L
- volume relative error: 0.000006%

## Kappa

- n prior samples: 1000
- median: 1990.24
- IQR: 851.833 - 5670.02
- min: 129.096
- max: 8.10925e+07

## Full Histogram

| log10(kappa) bin | count |
|---|---:|
| 2.11 to 2.40 | 30 |
| 2.40 to 2.69 | 100 |
| 2.69 to 2.98 | 163 |
| 2.98 to 3.27 | 185 |
| 3.27 to 3.56 | 177 |
| 3.56 to 3.85 | 131 |
| 3.85 to 4.14 | 68 |
| 4.14 to 4.43 | 45 |
| 4.43 to 4.72 | 26 |
| 4.72 to 5.01 | 19 |
| 5.01 to 5.30 | 19 |
| 5.30 to 5.59 | 14 |
| 5.59 to 5.88 | 5 |
| 5.88 to 6.17 | 6 |
| 6.17 to 6.46 | 4 |
| 6.46 to 6.75 | 2 |
| 6.75 to 7.04 | 2 |
| 7.04 to 7.33 | 1 |
| 7.33 to 7.62 | 1 |
| 7.62 to 7.91 | 2 |

## Median Absolute Eigenvector Loadings

### Stiffest Direction

| parameter | loading |
|---|---:|
| `log_tau_f` | 0.2042 |
| `log_tau_s` | 0.1504 |
| `logit_alpha` | 0.1823 |
| `log_fvc` | 0.9204 |
| `log_tau_o` | 0.1302 |

### Sloppiest Direction

| parameter | loading |
|---|---:|
| `log_tau_f` | 0.2945 |
| `log_tau_s` | 0.3057 |
| `logit_alpha` | 0.5201 |
| `log_fvc` | 0.0116 |
| `log_tau_o` | 0.5397 |
