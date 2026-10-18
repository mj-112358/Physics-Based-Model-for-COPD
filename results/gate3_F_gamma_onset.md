# OMBE-Gamma Onset Gate

- status: FAIL
- reason: OMBE-gamma fails identifiability and does not clear the 200-curve gamma-helps gate.
- scope: cycle F 200-curve A-grade non-obstructed sample unless Step 3 passes.
- model: `gamma(t)=(t/tau_o)^k exp(-t/tau_o)` times the bi-exponential emptying term, normalized so integral flow equals FVC.

## Step 1 - Volume Conservation

- midrange integral: 4.000000 L
- midrange FVC: 4.000000 L
- relative error: 0.000000%

## Step 2 - Identifiability

- Fisher samples: 1000
- kappa median: 1.78e+09
- kappa IQR: 6.24e+08 to 7.09e+09
- kappa p95: 1.72e+11
- kappa >1e4 fraction: 1.000
- kappa >1e6 fraction: 1.000
- identifiability flag: True (median kappa exceeds 1e4)

Median stiff-direction loadings:

| parameter | loading |
|---|---:|
| `tau_f` | 0.140 |
| `tau_s` | 0.018 |
| `alpha` | 0.041 |
| `fvc_l` | 0.368 |
| `tau_o` | 0.726 |
| `k` | 0.507 |

Median sloppy-direction loadings:

| parameter | loading |
|---|---:|
| `tau_f` | 0.117 |
| `tau_s` | 0.755 |
| `alpha` | 0.641 |
| `fvc_l` | 0.000 |
| `tau_o` | 0.054 |
| `k` | 0.000 |

## Step 3 - 200-Curve Flow-Domain Fit

| metric | bi-exponential baseline | registered baseline | OMBE-gamma |
|---|---:|---:|---:|
| dipole fraction | 0.845 | 0.535 | 0.960 |
| flow NRSS_norm median | 19.572 | 6.951 | 46.675 |
| all-three-within-5% | 0.840 | 0.680 | 0.065 |
| FEV1/FVC within 0.02 | 0.675 | 0.435 | 0.120 |

- flow adequate fraction: 0.000
- FEV1 median abs error: 6.10%
- FVC median abs error: 15.71%
- FEV1/FVC median absolute error: 0.1115
- gamma k median: 1.000
- gamma k IQR: 1.000 to 1.000

Parameter-at-bound counts:

| parameter | lower | upper | any | fraction any |
|---|---:|---:|---:|---:|
| `tau_o` | 0 | 51 | 51 | 0.255 |
| `tau_s` | 1 | 52 | 53 | 0.265 |
| `k` | 197 | 0 | 197 | 0.985 |

- Step 3 gamma-helps gate: False
- Step 3 gate reason: dipoles<30%=False; tau_o/tau_s/k bound-piling<50 each=False; ratio-within-0.02 materially above baseline=False; all-within-5% not below baseline=False

## Verdict

OMBE-gamma is not adopted. The augmentation adds complexity without clearing the onset, clinical, and identifiability gates.
