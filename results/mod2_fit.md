# Module 2 - OMBE Least-Squares Fitter

- status: PASS
- synthetic curves: 200
- normalized RSS threshold: 2.76
- threshold rationale: normalized RSS is mean squared residual divided by known synthetic noise variance; 2.76 is used as a fixed Steltner-like adequacy ceiling, not tuned to this run.
- clean curves adequate: 200/200 (100.0%)
- median normalized RSS: 0.995
- median R^2: 0.9767
- corrupted curves flagged inadequate: 49/50 (98.0%)

## Parameter Recovery

| parameter | median abs relative error | IQR |
|---|---:|---|
| `tau_f` | 2.66% | 1.22% to 5.16% |
| `tau_s` | 3.50% | 1.53% to 6.97% |
| `alpha` | 1.97% | 0.75% to 4.08% |
| `fvc_l` | 0.50% | 0.22% to 0.89% |
| `tau_o` | 4.23% | 1.62% to 8.52% |

Gate 1 identified `log_fvc` as the stiffest direction and `tau_o`, `alpha`, and time constants as sloppy directions. This report should be read in that identifiability context.
