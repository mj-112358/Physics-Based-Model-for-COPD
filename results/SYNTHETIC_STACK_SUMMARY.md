# Synthetic Stack Summary

- Module 1 segmentation: PASS
- Module 2 least-squares fitter: PASS
- Module 3 TSNPE: PASS
  - reason: TSNPE agreed with converged NUTS on >=4/6 synthetic metrics with no catastrophic divergence.
  - NUTS warmup/draws/chains: 2500/2500/4
  - NUTS converged: 181/200
  - metric pass count: 4/6
  - held-out curves evaluated: 181/200
  - FEV1 diff pass fraction: 0.901
  - FEV1/FVC diff pass fraction: 0.972
  - TSNPE 90% coverage: 0.888
  - worst agreement parameter: tau_f

## Boundary

This validates that TSNPE approximates NUTS on curves drawn from the OMBE
generative model. It does NOT establish that either engine fits real NHANES
curves - that is Gate 3, which requires the raw-curve file and is outstanding.
Modules 2 and 3 confirm the inference machinery is internally correct, not that
OMBE describes real physiology.
