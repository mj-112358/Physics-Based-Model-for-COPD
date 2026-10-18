# Module 3 - TSNPE Training + NUTS Agreement

- status: PASS
- reason: TSNPE agreed with converged NUTS on >=4/6 synthetic metrics with no catastrophic divergence.

## Boundary

This validates that TSNPE approximates NUTS on curves drawn from the OMBE
generative model. It does NOT establish that either engine fits real NHANES
curves - that is Gate 3, which requires the raw-curve file and is outstanding.
Modules 2 and 3 confirm the inference machinery is internally correct, not that
OMBE describes real physiology.

## Environment

- sbi: 0.26.1
- torch: 2.8.0
- nflows: 0.14
- zuko: 1.6.0
- numpyro: 0.19.0
- jax: 0.6.2

## Training

- sbi estimator: sbi SNPE_C posterior_nn(model='nsf')
- flow: rational-quadratic spline NSF
- coupling/spline transforms: 5
- embedding: 1D-CNN to 64-dimensional summary
- training rounds: 10
- simulations per round: 2000
- total simulation budget: 20000
- proposal note: prior simulations in appended rounds; no observation-specific proposal
- final training loss: -6.44875
- final validation loss: -6.37921
- best validation loss: -6.58802

## Prior vs Converged-NUTS Baseline

| metric | prior 500/500 single-chain NUTS | current converged-NUTS run |
|---|---:|---:|
| NUTS warmup/draws/chains | 500/500/1 | 2500/2500/4 |
| NUTS convergence gate | not checked | 181/200 converged |
| metric pass count | 2/6 | 4/6 |
| FEV1 median diff | 209 mL | 18.68 mL |
| FEV1/FVC median diff | 0.05008 | 0.004072 |
| std-ratio pass fraction | 0.021 | 0.218 |
| CI-overlap pass fraction | 0.017 | 0.156 |
| TSNPE 90% coverage | 0.894 | 0.888 |
| worst parameter | tau_f | tau_f |

## Initial Converged-NUTS Run

- run label: converged_10k
- NUTS warmup/draws/chains: 2500/2500/4
- NUTS converged: 181/200
- metric pass count: 3/6
- worst parameter: tau_f
- FEV1/FVC median diff: 0.0063
- std-ratio pass fraction: 0.182
- CI-overlap pass fraction: 0.143

## NUTS Convergence Gate

- run label: budget_20k
- held-out synthetic curves: 200
- NUTS warmup/draws/chains: 2500/2500/4
- NUTS chain method: sequential
- R-hat threshold: < 1.01
- ESS threshold: > 400.0
- NUTS attempted: 200/200
- NUTS converged: 181/200
- NUTS failed: 19/200 (9.5%)
- convergence gate passed: True
- max R-hat observed: 23.8542
- min ESS observed: 2.00362

## Agreement Metrics

- held-out curves with converged NUTS used for agreement: 181
- TSNPE posterior samples per curve: 1000
- metric pass count: 4/6
- catastrophic divergence: False

| metric | value | pass |
|---|---:|---|
| FEV1 mean diff <50 mL | 0.901 (median 18.7 mL) | True |
| FEV1/FVC mean diff <0.02 | 0.972 (median 0.0041) | True |
| posterior std ratio 0.7-1.5 | 0.218 | False |
| 90% CI overlap >70% | 0.156 | False |
| SBC rank chi-square p>0.05 | 4/5 params | True |
| TSNPE 90% coverage >=85% | 0.888 | True |

CI overlap is measured as intersection length divided by union length for each parameter's 90% credible interval, then aggregated across curve-parameter pairs.

## Per-Parameter Agreement

| parameter | std ratio median | std ratio pass frac | CI overlap median | CI pass frac | TSNPE 90% coverage | SBC p | median abs mean diff / truth |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tau_f` | 5.803 | 0.044 | 0.159 | 0.044 | 0.878 | 0.831137 | 0.123 |
| `tau_s` | 3.574 | 0.033 | 0.257 | 0.028 | 0.845 | 0.0617623 | 0.176 |
| `alpha` | 5.373 | 0.044 | 0.169 | 0.044 | 0.906 | 0.165325 | 0.117 |
| `fvc_l` | 1.196 | 0.867 | 0.730 | 0.586 | 0.890 | 0.0177001 | 0.004 |
| `tau_o` | 3.594 | 0.099 | 0.237 | 0.077 | 0.923 | 0.343556 | 0.282 |

FEV1/FVC median-diff bias flag (>0.02): False.
Worst agreement parameter: `tau_f`.
tau_f persists as worst against converged baseline: True.
tau_o worst as expected from sloppy-direction diagnostics: False.
If tau_f persists as worst, the CNN summary network losing fast-rise information is the likely cause; this run diagnoses that but does not change the architecture.
Flag: tau_o was not the worst parameter, which is inconsistent with the Gate 1 / Gate 5 / Module 2 pattern.
