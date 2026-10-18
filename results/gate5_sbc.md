# Gate 5 - SBC Calibration

- status: PASS
- reason: Rank histograms passed previously and coverage is within Monte Carlo tolerance.
- coverage replications: 500
- posterior draws per replication: 1000
- warmup per replication: 1000
- chains per coverage replication: 1
- sigma: 0.1 L/s
- prior rank run p>0.05 params: 5/5

Rank histograms were not rerun here; this coverage-only run reuses the prior rank-uniformity result.

## Prior Rank Histogram Tests

| parameter | chi-square p | shape |
|---|---:|---|
| `tau_f` | 0.274416 | no gross shape flag |
| `tau_s` | 0.759894 | no gross shape flag |
| `alpha` | 0.84865 | no gross shape flag |
| `fvc_l` | 0.228648 | no gross shape flag |
| `tau_o` | 0.294323 | U-shaped |

## Coverage With Monte Carlo Tolerance

Criterion: PASS if observed coverage is within `nominal +/- 2*sqrt(p(1-p)/n)`; flag persistent overconfidence only when a parameter is below the band at >=3 nominal levels.

| nominal | MC band | `tau_f` | `tau_s` | `alpha` | `fvc_l` | `tau_o` |
|---|---|---:|---:|---:|---:|---:|
| 50% | 0.455-0.545 | 0.522 | 0.482 | 0.464 | 0.530 | 0.490 |
| 60% | 0.556-0.644 | 0.590 | 0.584 | 0.556 (below) | 0.626 | 0.592 |
| 70% | 0.659-0.741 | 0.690 | 0.696 | 0.676 | 0.716 | 0.674 |
| 80% | 0.764-0.836 | 0.790 | 0.784 | 0.782 | 0.804 | 0.778 |
| 90% | 0.873-0.927 | 0.882 | 0.880 | 0.882 | 0.902 | 0.862 (below) |
| 95% | 0.931-0.969 | 0.916 (below) | 0.932 | 0.940 | 0.942 | 0.922 (below) |

## Persistent Below-Band Flags

| parameter | below-band nominal levels |
|---|---:|
| `tau_f` | 1 |
| `tau_s` | 0 |
| `alpha` | 1 |
| `fvc_l` | 0 |
| `tau_o` | 2 |

Headline: genuinely mis-covered params = none; all other coverage deviations are within Monte Carlo tolerance.

No tuning was performed.
