# Gate Degradation Robustness

> Evidence status: historical and provisional. This report depends on legacy
> truncation artifacts and has not been regenerated under the fixed-horizon
> protocol. Its verdict is not part of the current supported evidence.

VERDICT: ROBUST-TO-BUDGET-DEVICE

## Data Safety And Guards

- Clean-copy location: in-memory Python CurveRecord/SevereRecord lists built from migrated E/F/G resolver; no scratch curve files written
- NHANES source safety: no writes are made to `NHANES_SpiroSBI/`; only `results/` outputs are written.
- Forward-model guard at start: max JAX-vs-NumPy volume diff 3.5527136788e-12 mL.
- Forward-model guard at end: max JAX-vs-NumPy volume diff 2.6645352591e-12 mL.

## Physical Basis

Citations used for the device-class basis: ATS/ERS 2019 spirometry technical standard https://doi.org/10.1164/rccm.201908-1590ST, portable AioCare spirometry work https://arxiv.org/abs/1903.03588, and low-cost paper/acoustic spirometry work https://arxiv.org/abs/2408.09389. The levels below are stress-test budgets, not a claim that every cited device has exactly these errors.

| operator | realistic budget level | basis |
|---|---|---|
| D1 additive flow noise | 2% of PEF | portable and acoustic/low-cost devices motivate explicit noise-floor stress; ATS/ERS accuracy standards define the research-grade reference expectation. |
| D2 coarser sampling | 30 Hz | NHANES waveforms are 100 Hz; 30 Hz tests a lower ADC/sample-rate budget device while preserving anti-alias filtering. |
| D3 gain/calibration error | +/-2%; combined uses +2% | ATS/ERS calibration expectations make volume scale error a clinically relevant stressor; the sweep tests both signs through +/-10%. |
| D4 baseline drift | small, 0.5% FVC net drift | zero instability in low-cost flow sensing produces slow baseline wander; this is modeled as low-frequency net drift over the maneuver. |

## Baseline Lock

The clean in-memory copy was required to match the current three-cycle pooled result before degradation was accepted.

| T | DIRECT FN | OMBE FN | DIRECT-OMBE | CI low | CI high | excludes 0 |
|---|---|---|---|---|---|---|
| 2 | 0.983 (877/892; CI 0.974-0.991) | 0.576 (514/892; CI 0.544-0.609) | 0.407 | 0.373 | 0.438 | True |
| 3 | 0.851 (759/892; CI 0.825-0.873) | 0.580 (517/892; CI 0.547-0.612) | 0.271 | 0.242 | 0.302 | True |

Severe field baseline:

| estimator | FN | CP upper 95 | fit failures |
|---|---|---|---|
| E0_DIRECT | 0.152 (7/46; CI 0.043-0.261) | NA | 0 |
| E1_OMBE | 0.000 (0/46; CI 0.000-0.077) | 0.077 | 0 |

## Full Single-Degradation Sweep

### D1_additive_noise

| level | T | DIRECT FN | OMBE FN | diff | CI low | CI high | excludes 0 | Gate2 FEV1 abs % | Gate2 FVC abs % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5% PEF | 2 | 0.983 (877/892; CI 0.975-0.991) | 0.574 (512/892; CI 0.543-0.607) | 0.409 | 0.376 | 0.438 | True | 0.149 | 0.657 | HOLDS |
| 0.5% PEF | 3 | 0.848 (756/892; CI 0.824-0.872) | 0.580 (517/892; CI 0.548-0.612) | 0.268 | 0.240 | 0.296 | True | 0.149 | 0.657 | HOLDS |
| 1.0% PEF | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.566 (505/892; CI 0.534-0.598) | 0.417 | 0.385 | 0.451 | True | 0.209 | 0.851 | HOLDS |
| 1.0% PEF | 3 | 0.851 (759/892; CI 0.827-0.874) | 0.578 (516/892; CI 0.544-0.612) | 0.272 | 0.244 | 0.303 | True | 0.209 | 0.851 | HOLDS |
| 2.0% PEF | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.577 (515/892; CI 0.545-0.609) | 0.406 | 0.374 | 0.437 | True | 0.390 | 1.261 | HOLDS |
| 2.0% PEF | 3 | 0.850 (758/892; CI 0.827-0.873) | 0.571 (509/892; CI 0.539-0.603) | 0.279 | 0.250 | 0.308 | True | 0.390 | 1.261 | HOLDS |
| 5.0% PEF | 2 | 0.984 (878/892; CI 0.975-0.992) | 0.559 (499/892; CI 0.526-0.593) | 0.425 | 0.392 | 0.457 | True | 0.899 | 2.582 | HOLDS |
| 5.0% PEF | 3 | 0.850 (758/892; CI 0.826-0.872) | 0.531 (474/892; CI 0.498-0.565) | 0.318 | 0.287 | 0.349 | True | 0.899 | 2.582 | HOLDS |
| 10.0% PEF | 2 | 0.984 (878/892; CI 0.975-0.992) | 0.467 (417/892; CI 0.435-0.500) | 0.517 | 0.484 | 0.549 | True | 1.946 | 4.713 | HOLDS |
| 10.0% PEF | 3 | 0.825 (736/892; CI 0.799-0.849) | 0.462 (412/892; CI 0.428-0.494) | 0.363 | 0.332 | 0.393 | True | 1.946 | 4.713 | HOLDS |

### D2_coarser_sampling

| level | T | DIRECT FN | OMBE FN | diff | CI low | CI high | excludes 0 | Gate2 FEV1 abs % | Gate2 FVC abs % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 Hz | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.576 (514/892; CI 0.543-0.608) | 0.407 | 0.377 | 0.439 | True | 0.000 | 0.000 | HOLDS |
| 100 Hz | 3 | 0.851 (759/892; CI 0.827-0.876) | 0.580 (517/892; CI 0.547-0.613) | 0.271 | 0.242 | 0.300 | True | 0.000 | 0.000 | HOLDS |
| 50 Hz | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.577 (515/892; CI 0.544-0.610) | 0.406 | 0.374 | 0.438 | True | 0.272 | 0.976 | HOLDS |
| 50 Hz | 3 | 0.849 (757/892; CI 0.826-0.871) | 0.576 (514/892; CI 0.545-0.608) | 0.272 | 0.243 | 0.303 | True | 0.272 | 0.976 | HOLDS |
| 30 Hz | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.580 (517/892; CI 0.548-0.613) | 0.404 | 0.372 | 0.437 | True | 0.271 | 0.945 | HOLDS |
| 30 Hz | 3 | 0.848 (756/892; CI 0.825-0.870) | 0.576 (514/892; CI 0.544-0.608) | 0.271 | 0.241 | 0.300 | True | 0.271 | 0.945 | HOLDS |
| 20 Hz | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.584 (521/892; CI 0.552-0.615) | 0.399 | 0.369 | 0.434 | True | 0.282 | 0.914 | HOLDS |
| 20 Hz | 3 | 0.844 (753/892; CI 0.820-0.869) | 0.573 (511/892; CI 0.540-0.604) | 0.271 | 0.243 | 0.300 | True | 0.282 | 0.914 | HOLDS |
| 10 Hz | 2 | 0.981 (875/892; CI 0.972-0.990) | 0.577 (515/892; CI 0.545-0.611) | 0.404 | 0.372 | 0.437 | True | 1.053 | 0.609 | HOLDS |
| 10 Hz | 3 | 0.837 (747/892; CI 0.812-0.862) | 0.549 (490/892; CI 0.517-0.580) | 0.288 | 0.258 | 0.317 | True | 1.053 | 0.609 | HOLDS |

### D3_gain_error

| level | T | DIRECT FN | OMBE FN | diff | CI low | CI high | excludes 0 | Gate2 FEV1 abs % | Gate2 FVC abs % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| -1% | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.573 (511/892; CI 0.539-0.607) | 0.410 | 0.378 | 0.442 | True | 1.000 | 1.000 | HOLDS |
| -1% | 3 | 0.851 (759/892; CI 0.827-0.874) | 0.578 (516/892; CI 0.548-0.611) | 0.272 | 0.241 | 0.302 | True | 1.000 | 1.000 | HOLDS |
| +1% | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.580 (517/892; CI 0.545-0.612) | 0.404 | 0.372 | 0.436 | True | 1.000 | 1.000 | HOLDS |
| +1% | 3 | 0.851 (759/892; CI 0.827-0.874) | 0.583 (520/892; CI 0.552-0.617) | 0.268 | 0.237 | 0.298 | True | 1.000 | 1.000 | HOLDS |
| -2% | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.568 (507/892; CI 0.535-0.602) | 0.415 | 0.383 | 0.446 | True | 2.000 | 2.000 | HOLDS |
| -2% | 3 | 0.851 (759/892; CI 0.828-0.874) | 0.575 (513/892; CI 0.544-0.607) | 0.276 | 0.248 | 0.307 | True | 2.000 | 2.000 | HOLDS |
| +2% | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.582 (519/892; CI 0.550-0.612) | 0.401 | 0.370 | 0.435 | True | 2.000 | 2.000 | HOLDS |
| +2% | 3 | 0.851 (759/892; CI 0.826-0.873) | 0.584 (521/892; CI 0.553-0.615) | 0.267 | 0.240 | 0.297 | True | 2.000 | 2.000 | HOLDS |
| -5% | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.554 (494/892; CI 0.519-0.586) | 0.429 | 0.398 | 0.461 | True | 5.000 | 5.000 | HOLDS |
| -5% | 3 | 0.851 (759/892; CI 0.826-0.874) | 0.557 (497/892; CI 0.526-0.590) | 0.294 | 0.265 | 0.323 | True | 5.000 | 5.000 | HOLDS |
| +5% | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.594 (530/892; CI 0.562-0.626) | 0.389 | 0.357 | 0.422 | True | 5.000 | 5.000 | HOLDS |
| +5% | 3 | 0.851 (759/892; CI 0.826-0.873) | 0.596 (532/892; CI 0.564-0.630) | 0.254 | 0.224 | 0.285 | True | 5.000 | 5.000 | HOLDS |
| -10% | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.536 (478/892; CI 0.504-0.570) | 0.447 | 0.414 | 0.479 | True | 10.000 | 10.000 | HOLDS |
| -10% | 3 | 0.851 (759/892; CI 0.828-0.874) | 0.540 (482/892; CI 0.509-0.572) | 0.311 | 0.281 | 0.341 | True | 10.000 | 10.000 | HOLDS |
| +10% | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.610 (544/892; CI 0.577-0.641) | 0.373 | 0.342 | 0.406 | True | 10.000 | 10.000 | HOLDS |
| +10% | 3 | 0.851 (759/892; CI 0.827-0.873) | 0.610 (544/892; CI 0.578-0.642) | 0.241 | 0.212 | 0.269 | True | 10.000 | 10.000 | HOLDS |

### D4_baseline_drift

| level | T | DIRECT FN | OMBE FN | diff | CI low | CI high | excludes 0 | Gate2 FEV1 abs % | Gate2 FVC abs % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| small (0.5% FVC net drift) | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.577 (515/892; CI 0.545-0.612) | 0.406 | 0.373 | 0.437 | True | 0.137 | 0.718 | HOLDS |
| small (0.5% FVC net drift) | 3 | 0.853 (761/892; CI 0.828-0.876) | 0.578 (516/892; CI 0.545-0.611) | 0.275 | 0.243 | 0.304 | True | 0.137 | 0.718 | HOLDS |
| medium (1.0% FVC net drift) | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.576 (514/892; CI 0.543-0.609) | 0.407 | 0.377 | 0.439 | True | 0.142 | 1.120 | HOLDS |
| medium (1.0% FVC net drift) | 3 | 0.850 (758/892; CI 0.826-0.873) | 0.576 (514/892; CI 0.543-0.609) | 0.274 | 0.246 | 0.304 | True | 0.142 | 1.120 | HOLDS |
| large (2.0% FVC net drift) | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.574 (512/892; CI 0.541-0.607) | 0.409 | 0.378 | 0.441 | True | 0.153 | 2.075 | HOLDS |
| large (2.0% FVC net drift) | 3 | 0.851 (759/892; CI 0.827-0.873) | 0.573 (511/892; CI 0.539-0.604) | 0.278 | 0.249 | 0.308 | True | 0.153 | 2.075 | HOLDS |

## Breaking Points

| operator | breaking point | breaks | note |
|---|---|---|---|
| D1_additive_noise | not reached in sweep | False | all swept levels retained positive DIRECT-minus-OMBE CI exclusion at T=2 and T=3 |
| D2_coarser_sampling | not reached in sweep | False | all swept levels retained positive DIRECT-minus-OMBE CI exclusion at T=2 and T=3 |
| D3_gain_error | not reached in sweep | False | all swept levels retained positive DIRECT-minus-OMBE CI exclusion at T=2 and T=3 |
| D4_baseline_drift | not reached in sweep | False | all swept levels retained positive DIRECT-minus-OMBE CI exclusion at T=2 and T=3 |

## Combined Realistic Budget Profile

Profile: 2% PEF noise + 30 Hz + +2% gain + small drift.

| T | DIRECT FN | OMBE FN | DIRECT-OMBE | CI low | CI high | excludes 0 | delta OMBE FN |
|---|---|---|---|---|---|---|---|
| 2 | 0.982 (876/892; CI 0.972-0.990) | 0.583 (520/892; CI 0.549-0.614) | 0.399 | 0.369 | 0.430 | True | 0.007 |
| 3 | 0.849 (757/892; CI 0.824-0.871) | 0.578 (516/892; CI 0.547-0.611) | 0.270 | 0.242 | 0.302 | True | -0.001 |

Combined severe field recovery:

| estimator | FN | CP upper 95 | fit failures |
|---|---|---|---|
| E0_DIRECT | 0.130 (6/46; CI 0.043-0.239) | NA | 0 |
| E1_OMBE | 0.000 (0/46; CI 0.000-0.077) | 0.077 | 0 |

COPD-positive external-validity stratum under the combined profile:

| T | n | DIRECT FN | OMBE FN | DIRECT-OMBE | CI low | CI high | excludes 0 |
|---|---|---|---|---|---|---|---|
| 2 | 126 | 0.960 (121/126; CI 0.921-0.992) | 0.437 (55/126; CI 0.349-0.524) | 0.524 | 0.429 | 0.611 | True |
| 3 | 126 | 0.722 (91/126; CI 0.643-0.802) | 0.437 (55/126; CI 0.349-0.524) | 0.286 | 0.206 | 0.365 | True |

## Sensitivity

| operator | worst level | baseline min margin | worst margin | margin drop |
|---|---|---|---|---|
| D1_additive_noise | 0.5% PEF | 0.271 | 0.268 | 0.003 |
| D2_coarser_sampling | 100 Hz | 0.271 | 0.271 | 0.000 |
| D3_gain_error | +10% | 0.271 | 0.241 | 0.030 |
| D4_baseline_drift | medium (1.0% FVC net drift) | 0.271 | 0.274 | -0.002 |

- Most sensitive: D3_gain_error at +10%.
- Least sensitive: D4_baseline_drift at medium (1.0% FVC net drift).

## Protocol Decisions

- Gate 2 degradation is measured against the clean parsed curve copy, not NHANES summary truth, because the task defines clean parsed curves as the ground truth for degradation.
- Only obstructed clean-copy curves are re-fit for the truncation FN sweep because the requested headline is obstructed FN and DIRECT-minus-OMBE FN advantage; Gate 2 measurement error is still computed on all clean curves.
- The existing SciPy bounded-TRF OMBE loop fit is called unchanged via fit_ombe_truncated; no NUTS and no vectorized solver path is used.
- Gaussian flow noise is expressed as a fraction of each curve's clean peak expiratory flow and uses deterministic per-curve seeds.
- Coarser sampling is simulated by scipy.signal.resample_poly anti-alias/resampling on flow, followed by interpolation back to the original time grid so the downstream solver interface stays unchanged.
- Pure 100 Hz sampling and pure gain profiles bypass flow reconstruction; gain-only profiles multiply the copied volume curve directly as specified.
- Volume is forced monotone after flow degradation by cumulative integration plus maximum-accumulate; this keeps degraded copied curves physically usable for the existing volume-fit solver.
- Baseline drift is modeled as a deterministic-sign, low-frequency half-sine flow offset whose net integrated volume equals the requested FVC fraction.
- The combined budget profile uses +2% gain as one representative calibration sign; the D3 independent sweep reports both signs through +/-10%.
- Severe field recovery is evaluated only for the combined profile and clean baseline, matching Part 3; single-degradation sweeps focus on pooled clean T=2/T=3 headlines.

## Breaking-Point Confirmation

This confirmation pass pushed the prior no-break result past realistic budget-device levels. It used the same in-memory copied E/F/G curves and the same deterministic SciPy OMBE loop fit; NHANES source files were not written.

### Extreme Breaking Points

| operator | breaking level | breaks | note |
|---|---|---|---|
| D1_additive_noise | 300% PEF | True | first level where T=2/T=3 advantage did not both exclude zero |
| D2_coarser_sampling | not reached | False | worst retained margin 0.271 at 100 Hz T=3 |
| D3_gain_error | +150% | True | first level where T=2/T=3 advantage did not both exclude zero |
| D4_baseline_drift | not reached | False | worst retained margin 0.265 at 4x prior large (8% FVC net drift) T=3 |
| D3_differential_gain | -60% early-vs-late | True | first level where T=2/T=3 advantage did not both exclude zero |

### Extreme Sweep Rows

| operator | level | T | DIRECT FN | OMBE FN | diff | CI low | CI high | excludes 0 | Gate2 ratio abs % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| D1_additive_noise | 15% PEF | 2 | 0.985 (879/892; CI 0.976-0.993) | 0.446 (398/892; CI 0.414-0.478) | 0.539 | 0.508 | 0.570 | True | 6.486 | HOLDS |
| D1_additive_noise | 15% PEF | 3 | 0.799 (713/892; CI 0.774-0.825) | 0.404 (360/892; CI 0.372-0.437) | 0.396 | 0.361 | 0.427 | True | 6.486 | HOLDS |
| D1_additive_noise | 25% PEF | 2 | 0.961 (857/892; CI 0.947-0.974) | 0.369 (329/892; CI 0.339-0.401) | 0.592 | 0.561 | 0.626 | True | 9.570 | HOLDS |
| D1_additive_noise | 25% PEF | 3 | 0.742 (662/892; CI 0.713-0.771) | 0.346 (309/892; CI 0.315-0.379) | 0.396 | 0.363 | 0.427 | True | 9.570 | HOLDS |
| D1_additive_noise | 40% PEF | 2 | 0.922 (822/892; CI 0.902-0.939) | 0.360 (321/892; CI 0.330-0.392) | 0.562 | 0.529 | 0.594 | True | 13.982 | HOLDS |
| D1_additive_noise | 40% PEF | 3 | 0.660 (589/892; CI 0.629-0.692) | 0.339 (302/892; CI 0.307-0.369) | 0.322 | 0.289 | 0.354 | True | 13.982 | HOLDS |
| D1_additive_noise | 60% PEF | 2 | 0.860 (767/892; CI 0.836-0.881) | 0.337 (301/892; CI 0.308-0.369) | 0.522 | 0.489 | 0.554 | True | 18.573 | HOLDS |
| D1_additive_noise | 60% PEF | 3 | 0.578 (516/892; CI 0.546-0.611) | 0.311 (277/892; CI 0.281-0.342) | 0.268 | 0.238 | 0.298 | True | 18.573 | HOLDS |
| D1_additive_noise | 100% PEF | 2 | 0.738 (658/892; CI 0.709-0.766) | 0.385 (343/892; CI 0.351-0.416) | 0.353 | 0.320 | 0.386 | True | 26.900 | HOLDS |
| D1_additive_noise | 100% PEF | 3 | 0.518 (462/892; CI 0.484-0.548) | 0.340 (303/892; CI 0.308-0.369) | 0.178 | 0.149 | 0.206 | True | 26.900 | HOLDS |
| D1_additive_noise | 150% PEF | 2 | 0.688 (614/892; CI 0.658-0.719) | 0.415 (370/892; CI 0.382-0.447) | 0.274 | 0.242 | 0.307 | True | 32.854 | HOLDS |
| D1_additive_noise | 150% PEF | 3 | 0.491 (438/892; CI 0.457-0.524) | 0.395 (352/892; CI 0.362-0.427) | 0.096 | 0.068 | 0.123 | True | 32.854 | HOLDS |
| D1_additive_noise | 200% PEF | 2 | 0.661 (590/892; CI 0.631-0.692) | 0.441 (393/892; CI 0.409-0.474) | 0.221 | 0.191 | 0.251 | True | 37.560 | HOLDS |
| D1_additive_noise | 200% PEF | 3 | 0.484 (432/892; CI 0.450-0.516) | 0.411 (367/892; CI 0.380-0.444) | 0.073 | 0.044 | 0.103 | True | 37.560 | HOLDS |
| D1_additive_noise | 300% PEF | 2 | 0.638 (569/892; CI 0.609-0.672) | 0.519 (463/892; CI 0.487-0.554) | 0.119 | 0.090 | 0.148 | True | 44.120 | HOLDS |
| D1_additive_noise | 300% PEF | 3 | 0.475 (424/892; CI 0.442-0.510) | 0.455 (406/892; CI 0.420-0.488) | 0.020 | -0.009 | 0.050 | False | 44.120 | BREAKS |
| D2_coarser_sampling | 8 Hz | 2 | 0.981 (875/892; CI 0.972-0.990) | 0.566 (505/892; CI 0.533-0.599) | 0.415 | 0.382 | 0.448 | True | 1.588 | HOLDS |
| D2_coarser_sampling | 8 Hz | 3 | 0.830 (740/892; CI 0.804-0.854) | 0.527 (470/892; CI 0.496-0.561) | 0.303 | 0.272 | 0.334 | True | 1.588 | HOLDS |
| D2_coarser_sampling | 5 Hz | 2 | 0.980 (874/892; CI 0.970-0.989) | 0.474 (423/892; CI 0.443-0.509) | 0.506 | 0.474 | 0.538 | True | 2.434 | HOLDS |
| D2_coarser_sampling | 5 Hz | 3 | 0.806 (719/892; CI 0.778-0.831) | 0.474 (423/892; CI 0.443-0.508) | 0.332 | 0.302 | 0.364 | True | 2.434 | HOLDS |
| D2_coarser_sampling | 3 Hz | 2 | 0.976 (871/892; CI 0.966-0.985) | 0.180 (161/892; CI 0.156-0.206) | 0.796 | 0.770 | 0.822 | True | 3.682 | HOLDS |
| D2_coarser_sampling | 3 Hz | 3 | 0.776 (692/892; CI 0.749-0.804) | 0.249 (222/892; CI 0.221-0.277) | 0.527 | 0.494 | 0.561 | True | 3.682 | HOLDS |
| D2_coarser_sampling | 2 Hz | 2 | 0.975 (870/892; CI 0.964-0.985) | 0.165 (147/892; CI 0.140-0.189) | 0.811 | 0.785 | 0.835 | True | 3.881 | HOLDS |
| D2_coarser_sampling | 2 Hz | 3 | 0.768 (685/892; CI 0.740-0.795) | 0.093 (83/892; CI 0.074-0.112) | 0.675 | 0.646 | 0.705 | True | 3.881 | HOLDS |
| D2_coarser_sampling | 1 Hz | 2 | 0.906 (808/892; CI 0.886-0.924) | 0.219 (195/892; CI 0.191-0.244) | 0.687 | 0.656 | 0.717 | True | 9.881 | HOLDS |
| D2_coarser_sampling | 1 Hz | 3 | 0.545 (486/892; CI 0.513-0.578) | 0.114 (102/892; CI 0.093-0.136) | 0.430 | 0.398 | 0.463 | True | 9.881 | HOLDS |
| D3_gain_error | -15% | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.509 (454/892; CI 0.476-0.544) | 0.474 | 0.439 | 0.508 | True | 0.000 | HOLDS |
| D3_gain_error | -15% | 3 | 0.851 (759/892; CI 0.827-0.874) | 0.512 (457/892; CI 0.478-0.544) | 0.339 | 0.308 | 0.371 | True | 0.000 | HOLDS |
| D3_gain_error | +15% | 2 | 0.983 (877/892; CI 0.974-0.990) | 0.620 (553/892; CI 0.590-0.652) | 0.363 | 0.334 | 0.395 | True | 0.000 | HOLDS |
| D3_gain_error | +15% | 3 | 0.851 (759/892; CI 0.827-0.873) | 0.618 (551/892; CI 0.586-0.648) | 0.233 | 0.203 | 0.262 | True | 0.000 | HOLDS |
| D3_gain_error | -25% | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.450 (401/892; CI 0.416-0.483) | 0.534 | 0.502 | 0.566 | True | 0.000 | HOLDS |
| D3_gain_error | -25% | 3 | 0.851 (759/892; CI 0.827-0.874) | 0.453 (404/892; CI 0.419-0.485) | 0.398 | 0.367 | 0.430 | True | 0.000 | HOLDS |
| D3_gain_error | +25% | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.638 (569/892; CI 0.607-0.668) | 0.345 | 0.315 | 0.374 | True | 0.000 | HOLDS |
| D3_gain_error | +25% | 3 | 0.851 (759/892; CI 0.827-0.873) | 0.637 (568/892; CI 0.604-0.667) | 0.214 | 0.187 | 0.243 | True | 0.000 | HOLDS |
| D3_gain_error | -40% | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.290 (259/892; CI 0.260-0.318) | 0.693 | 0.663 | 0.723 | True | 0.000 | HOLDS |
| D3_gain_error | -40% | 3 | 0.851 (759/892; CI 0.827-0.876) | 0.299 (267/892; CI 0.269-0.330) | 0.552 | 0.519 | 0.585 | True | 0.000 | HOLDS |
| D3_gain_error | +40% | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.656 (585/892; CI 0.624-0.687) | 0.327 | 0.296 | 0.359 | True | 0.000 | HOLDS |
| D3_gain_error | +40% | 3 | 0.851 (759/892; CI 0.827-0.873) | 0.651 (581/892; CI 0.621-0.682) | 0.200 | 0.172 | 0.229 | True | 0.000 | HOLDS |
| D3_gain_error | +80% | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.733 (654/892; CI 0.702-0.760) | 0.250 | 0.222 | 0.279 | True | 0.000 | HOLDS |
| D3_gain_error | +80% | 3 | 0.851 (759/892; CI 0.828-0.873) | 0.721 (643/892; CI 0.692-0.751) | 0.130 | 0.102 | 0.159 | True | 0.000 | HOLDS |
| D3_gain_error | +150% | 2 | 0.983 (877/892; CI 0.975-0.991) | 0.848 (756/892; CI 0.825-0.870) | 0.136 | 0.113 | 0.159 | True | 0.000 | HOLDS |
| D3_gain_error | +150% | 3 | 0.851 (759/892; CI 0.828-0.873) | 0.839 (748/892; CI 0.814-0.862) | 0.012 | -0.013 | 0.039 | False | 0.000 | BREAKS |
| D3_gain_error | +200% | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.893 (797/892; CI 0.873-0.913) | 0.090 | 0.070 | 0.112 | True | 0.000 | HOLDS |
| D3_gain_error | +200% | 3 | 0.851 (759/892; CI 0.827-0.873) | 0.883 (788/892; CI 0.861-0.904) | -0.033 | -0.057 | -0.009 | True | 0.000 | BREAKS |
| D4_baseline_drift | 2x prior large (4% FVC net drift) | 2 | 0.982 (876/892; CI 0.973-0.990) | 0.576 (514/892; CI 0.545-0.607) | 0.406 | 0.373 | 0.437 | True | 3.925 | HOLDS |
| D4_baseline_drift | 2x prior large (4% FVC net drift) | 3 | 0.844 (753/892; CI 0.821-0.868) | 0.575 (513/892; CI 0.540-0.607) | 0.269 | 0.239 | 0.299 | True | 3.925 | HOLDS |
| D4_baseline_drift | 4x prior large (8% FVC net drift) | 2 | 0.983 (877/892; CI 0.974-0.991) | 0.574 (512/892; CI 0.541-0.607) | 0.409 | 0.377 | 0.442 | True | 7.626 | HOLDS |
| D4_baseline_drift | 4x prior large (8% FVC net drift) | 3 | 0.840 (749/892; CI 0.815-0.863) | 0.575 (513/892; CI 0.544-0.608) | 0.265 | 0.237 | 0.294 | True | 7.626 | HOLDS |
| D4_baseline_drift | 8x prior large (16% FVC net drift) | 2 | 0.982 (876/892; CI 0.972-0.990) | 0.581 (518/892; CI 0.548-0.612) | 0.401 | 0.368 | 0.433 | True | 13.847 | HOLDS |
| D4_baseline_drift | 8x prior large (16% FVC net drift) | 3 | 0.837 (747/892; CI 0.814-0.860) | 0.557 (497/892; CI 0.525-0.590) | 0.280 | 0.251 | 0.311 | True | 13.847 | HOLDS |
| D4_baseline_drift | 16x prior large (32% FVC net drift) | 2 | 0.987 (880/892; CI 0.978-0.993) | 0.567 (506/892; CI 0.537-0.600) | 0.419 | 0.389 | 0.453 | True | 23.685 | HOLDS |
| D4_baseline_drift | 16x prior large (32% FVC net drift) | 3 | 0.817 (729/892; CI 0.790-0.842) | 0.541 (483/892; CI 0.510-0.575) | 0.276 | 0.248 | 0.305 | True | 23.685 | HOLDS |
| D4_baseline_drift | 32x prior large (64% FVC net drift) | 2 | 0.981 (875/892; CI 0.972-0.989) | 0.536 (478/892; CI 0.506-0.568) | 0.445 | 0.411 | 0.475 | True | 37.425 | HOLDS |
| D4_baseline_drift | 32x prior large (64% FVC net drift) | 3 | 0.751 (670/892; CI 0.723-0.778) | 0.444 (396/892; CI 0.413-0.476) | 0.307 | 0.277 | 0.340 | True | 37.425 | HOLDS |
| D3_differential_gain | -5% early-vs-late | 2 | 0.974 (869/892; CI 0.963-0.984) | 0.436 (389/892; CI 0.404-0.470) | 0.538 | 0.507 | 0.570 | True | 4.114 | HOLDS |
| D3_differential_gain | -5% early-vs-late | 3 | 0.765 (682/892; CI 0.738-0.792) | 0.444 (396/892; CI 0.410-0.476) | 0.321 | 0.289 | 0.352 | True | 4.114 | HOLDS |
| D3_differential_gain | +5% early-vs-late | 2 | 0.991 (884/892; CI 0.984-0.997) | 0.692 (617/892; CI 0.661-0.722) | 0.299 | 0.270 | 0.332 | True | 2.538 | HOLDS |
| D3_differential_gain | +5% early-vs-late | 3 | 0.902 (805/892; CI 0.885-0.922) | 0.678 (605/892; CI 0.650-0.709) | 0.224 | 0.197 | 0.252 | True | 2.538 | HOLDS |
| D3_differential_gain | -10% early-vs-late | 2 | 0.955 (852/892; CI 0.942-0.967) | 0.252 (225/892; CI 0.222-0.280) | 0.703 | 0.672 | 0.733 | True | 7.542 | HOLDS |
| D3_differential_gain | -10% early-vs-late | 3 | 0.672 (599/892; CI 0.640-0.702) | 0.285 (254/892; CI 0.257-0.315) | 0.387 | 0.356 | 0.418 | True | 7.542 | HOLDS |
| D3_differential_gain | +10% early-vs-late | 2 | 0.997 (889/892; CI 0.992-1.000) | 0.772 (689/892; CI 0.744-0.799) | 0.224 | 0.197 | 0.253 | True | 5.341 | HOLDS |
| D3_differential_gain | +10% early-vs-late | 3 | 0.944 (842/892; CI 0.928-0.959) | 0.769 (686/892; CI 0.741-0.797) | 0.175 | 0.152 | 0.198 | True | 5.341 | HOLDS |
| D3_differential_gain | -15% early-vs-late | 2 | 0.914 (815/892; CI 0.896-0.932) | 0.121 (108/892; CI 0.100-0.142) | 0.793 | 0.767 | 0.818 | True | 10.981 | HOLDS |
| D3_differential_gain | -15% early-vs-late | 3 | 0.528 (471/892; CI 0.497-0.559) | 0.130 (116/892; CI 0.109-0.154) | 0.398 | 0.367 | 0.430 | True | 10.981 | HOLDS |
| D3_differential_gain | +15% early-vs-late | 2 | 0.998 (890/892; CI 0.994-1.000) | 0.833 (743/892; CI 0.807-0.858) | 0.165 | 0.140 | 0.189 | True | 8.151 | HOLDS |
| D3_differential_gain | +15% early-vs-late | 3 | 0.965 (861/892; CI 0.953-0.978) | 0.818 (730/892; CI 0.794-0.844) | 0.147 | 0.123 | 0.172 | True | 8.151 | HOLDS |
| D3_differential_gain | -25% early-vs-late | 2 | 0.740 (660/892; CI 0.710-0.768) | 0.006 (5/892; CI 0.001-0.011) | 0.734 | 0.704 | 0.761 | True | 18.180 | HOLDS |
| D3_differential_gain | -25% early-vs-late | 3 | 0.222 (198/892; CI 0.195-0.250) | 0.006 (5/892; CI 0.001-0.011) | 0.216 | 0.189 | 0.243 | True | 18.180 | HOLDS |
| D3_differential_gain | +25% early-vs-late | 2 | 0.999 (891/892; CI 0.997-1.000) | 0.911 (813/892; CI 0.892-0.930) | 0.087 | 0.068 | 0.107 | True | 13.552 | HOLDS |
| D3_differential_gain | +25% early-vs-late | 3 | 0.989 (882/892; CI 0.982-0.996) | 0.888 (792/892; CI 0.867-0.908) | 0.101 | 0.082 | 0.121 | True | 13.552 | HOLDS |
| D3_differential_gain | -40% early-vs-late | 2 | 0.284 (253/892; CI 0.254-0.314) | 0.000 (0/892; CI 0.000-0.000) | 0.284 | 0.254 | 0.314 | True | 30.292 | HOLDS |
| D3_differential_gain | -40% early-vs-late | 3 | 0.004 (4/892; CI 0.001-0.009) | 0.000 (0/892; CI 0.000-0.000) | 0.004 | 0.001 | 0.009 | True | 30.292 | HOLDS |
| D3_differential_gain | +40% early-vs-late | 2 | 0.999 (891/892; CI 0.997-1.000) | 0.950 (847/892; CI 0.935-0.964) | 0.049 | 0.036 | 0.064 | True | 20.976 | HOLDS |
| D3_differential_gain | +40% early-vs-late | 3 | 0.998 (890/892; CI 0.994-1.000) | 0.932 (831/892; CI 0.915-0.948) | 0.066 | 0.050 | 0.083 | True | 20.976 | HOLDS |
| D3_differential_gain | -60% early-vs-late | 2 | 0.006 (5/892; CI 0.001-0.011) | 0.000 (0/892; CI 0.000-0.000) | 0.006 | 0.001 | 0.011 | True | 49.163 | HOLDS |
| D3_differential_gain | -60% early-vs-late | 3 | 0.002 (2/892; CI 0.000-0.006) | 0.000 (0/892; CI 0.000-0.000) | 0.002 | 0.000 | 0.006 | False | 49.163 | BREAKS |
| D3_differential_gain | +60% early-vs-late | 2 | 1.000 (892/892; CI 1.000-1.000) | 0.970 (865/892; CI 0.959-0.981) | 0.030 | 0.019 | 0.043 | True | 29.997 | HOLDS |
| D3_differential_gain | +60% early-vs-late | 3 | 1.000 (892/892; CI 1.000-1.000) | 0.970 (865/892; CI 0.959-0.980) | 0.030 | 0.019 | 0.041 | True | 29.997 | HOLDS |
| D3_differential_gain | +80% early-vs-late | 2 | 1.000 (892/892; CI 1.000-1.000) | 0.982 (876/892; CI 0.973-0.990) | 0.018 | 0.009 | 0.027 | True | 37.962 | HOLDS |
| D3_differential_gain | +80% early-vs-late | 3 | 1.000 (892/892; CI 1.000-1.000) | 0.982 (876/892; CI 0.973-0.990) | 0.018 | 0.010 | 0.027 | True | 37.962 | HOLDS |
| D3_differential_gain | +95% early-vs-late | 2 | 1.000 (892/892; CI 1.000-1.000) | 0.990 (883/892; CI 0.983-0.996) | 0.010 | 0.004 | 0.017 | True | 43.378 | HOLDS |
| D3_differential_gain | +95% early-vs-late | 3 | 1.000 (892/892; CI 1.000-1.000) | 0.990 (883/892; CI 0.983-0.996) | 0.010 | 0.004 | 0.017 | True | 43.378 | HOLDS |
| D3_differential_gain | +99% early-vs-late | 2 | 1.000 (892/892; CI 1.000-1.000) | 0.990 (883/892; CI 0.983-0.996) | 0.010 | 0.004 | 0.017 | True | 44.762 | HOLDS |
| D3_differential_gain | +99% early-vs-late | 3 | 1.000 (892/892; CI 1.000-1.000) | 0.990 (883/892; CI 0.983-0.996) | 0.010 | 0.004 | 0.017 | True | 44.762 | HOLDS |

### Operator Perturbation Checks

| operator | worst level | T | worst margin | max ratio abs % | max FEV1 abs % | max FVC abs % | DIRECT FN range | OMBE FN range |
|---|---|---|---|---|---|---|---|---|
| D1_additive_noise | 300% PEF | 3 | 0.020 | 44.120 | 48.786 | 125.476 | 0.475-0.985 | 0.311-0.580 |
| D2_coarser_sampling | 100 Hz | 3 | 0.271 | 9.881 | 8.820 | 3.046 | 0.545-0.983 | 0.093-0.584 |
| D3_gain_error | +200% | 3 | -0.033 | 0.000 | 200.000 | 200.000 | 0.851-0.983 | 0.290-0.893 |
| D4_baseline_drift | 4x prior large (8% FVC net drift) | 3 | 0.265 | 37.425 | 1.601 | 63.907 | 0.751-0.987 | 0.444-0.581 |
| D3_differential_gain | -60% early-vs-late | 3 | 0.002 | 49.163 | 98.583 | 37.213 | 0.002-1.000 | 0.000-0.990 |

### D3 Gain-Model Clarification

- Proportional D3 gain multiplies the copied volume curve by one scalar before FEV1 and FVC are read. DIRECT and Gate-2 FEV1/FVC ratios therefore cancel the gain exactly; this is immunity by construction for ratio-based diagnosis, not evidence that proportional calibration error perturbs the ratio.
- Proportional gain can still perturb OMBE near solver bounds because the bounded fit estimates FVC on an absolute liter scale, but the full-curve ratio diagnostic remains exactly canceled.
- Differential gain was added as an explicit ratio-threatening variant: positive levels over-read pre-1s FEV1-weighted flow and under-read the post-1s FVC-tail flow; negative levels do the reverse. Its confirmed breaking point is -60% early-vs-late.
- For the FN-threatening positive differential-gain sign, no break was reached through +99% early-vs-late; the first statistical CI failure is the opposite sign, where DIRECT nearly stops missing obstruction and the DIRECT-minus-OMBE advantage becomes too small to distinguish.

### Honest Restatement

- Realistic-profile claim remains unchanged: ROBUST-TO-BUDGET-DEVICE under the original 2% noise + 30 Hz + +2% proportional gain + small drift profile.
- Corrected gain framing: ratio-based diagnosis is inherently immune to proportional gain, while the differential-gain stress test measures the non-proportional calibration error that can actually move FEV1/FVC.
- D2 and D4 still showed no breaking point after the failure-seeking extension, but this was not because the operators were inert: D2 reached 1 Hz with 9.881% median ratio error, and D4 reached 64% FVC net drift with 37.425% median ratio error.
