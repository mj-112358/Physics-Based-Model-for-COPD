# Gate - Deterministic Fit Optimization Pass, Cycle F

## Protocol Decisions

- This is a guarded optimization gate, not a production rewrite. No production estimator is changed unless every guard accepts the new configuration.
- Baseline headlines are locked by reading existing result files before any refit.
- The forward model is guarded by JAX-vs-NumPy volume agreement in mL with x64 enabled.
- The current production volume fit is SciPy bounded TRF least squares; JAX is used for a batched residual equivalence test, not as an unreviewed replacement optimizer.
- Because the batched-fit equivalence guard fails, Parts 3-5 do not ship robust loss, onset changes, or multi-start headline recomputation.

## Part 0 - Baseline Lock

| headline | metric | estimator | endpoint | n | denom | baseline | unit |
|---|---|---|---|---|---|---|---|
| Gate 2 parser | median_abs_fev1_error_pct |  |  | 7079 | NA | 2.908327 | percent |
| Gate 2 parser | median_abs_fvc_error_pct |  |  | 7079 | NA | 3.527326 | percent |
| Gate 2 parser | median_signed_fev1_error_pct |  |  | 7079 | NA | -2.885889 | percent |
| Gate 2 parser | median_signed_fvc_error_pct |  |  | 7079 | NA | -3.489326 | percent |
| Clinical fixed-0.70 | sensitivity |  |  | 414 | 114 | 0.982456 | rate |
| Clinical fixed-0.70 | specificity |  |  | 414 | 300 | 0.956667 | rate |
| Clinical fixed-0.70 | cohens_kappa |  |  | 414 | NA | 0.911847 | kappa |
| Truncation clean-cut | obstructed_false_negative_rate | E1_OMBE | 2 | 497 | 297 | 0.589226 | rate |
| Truncation clean-cut | obstructed_false_negative_rate | H1 | 2 | 497 | 297 | 0.579125 | rate |
| Truncation clean-cut | obstructed_false_negative_rate | E1_OMBE | 3 | 497 | 297 | 0.595960 | rate |
| Truncation clean-cut | obstructed_false_negative_rate | H1 | 3 | 497 | 297 | 0.579125 | rate |
| Truncation clean-cut | obstructed_false_negative_rate | E1_OMBE | 4 | 497 | 297 | 0.518519 | rate |
| Truncation clean-cut | obstructed_false_negative_rate | H1 | 4 | 497 | 297 | 0.494949 | rate |
| Truncation clean-cut | obstructed_false_negative_rate | E1_OMBE | 5 | 497 | 297 | 0.468013 | rate |
| Truncation clean-cut | obstructed_false_negative_rate | H1 | 5 | 497 | 297 | 0.441077 | rate |
| Truncation field-realistic | obstructed_false_negative_rate | E0_DIRECT | field_all | 286 | 178 | 0.426966 | rate |
| Truncation field-realistic | obstructed_false_negative_rate | E1_OMBE | field_all | 286 | 178 | 0.067416 | rate |
| Truncation field-realistic | obstructed_false_negative_rate | H1 | field_all | 286 | 178 | 0.073034 | rate |
| Severe-band clean recovery | severe_false_negative_rate | E1_OMBE | 2 | 17 | 17 | 0.000000 | rate |
| Severe-band clean recovery | severe_false_negative_rate | E1_OMBE | 3 | 17 | 17 | 0.000000 | rate |
| Severe-band clean recovery | severe_false_negative_rate | E1_OMBE | 4 | 17 | 17 | 0.000000 | rate |
| Severe-band clean recovery | severe_false_negative_rate | E1_OMBE | 5 | 17 | 17 | 0.000000 | rate |
| Severe-band clean recovery | severe_false_negative_rate | H1 | 2 | 17 | 17 | 0.000000 | rate |
| Severe-band clean recovery | severe_false_negative_rate | H1 | 3 | 17 | 17 | 0.000000 | rate |
| Severe-band clean recovery | severe_false_negative_rate | H1 | 4 | 17 | 17 | 0.000000 | rate |
| Severe-band clean recovery | severe_false_negative_rate | H1 | 5 | 17 | 17 | 0.000000 | rate |

## Forward-Model Guard

| phase | param sets | grid n | max abs volume mL | tolerance mL | passed |
|---|---|---|---|---|---|
| start | 24 | 1201 | 0.000000000004 | 0.000000001000 | True |
| end | 24 | 1201 | 0.000000000003 | 0.000000001000 | True |

## Part 1 - Vectorized Fit Enabling Step

Loop refit completed on 200 curves in 1.124s (1.124s per 200 curves).

The stacked sparse batched solver pilot used 20 curves. It took 4.196s (linear projection 41.962s per 200), with max relative difference 0.00678442 against the current loop for (tau_f, tau_s, alpha, FVC).

Guard result: FAIL; tolerance is < 1e-06.

20-curve pilot already exceeded the <1e-6 relative-parameter equivalence guard; running the 200-curve stacked solver would be more expensive and still unshippable.

## Part 2 - Multi-Start Landscape Diagnostic

This diagnostic was run on the 200-curve audit set, but it is not allowed to change headline gates because Part 1 failed.

Raw non-default-best rate: 170 / 200 = 0.850.

Material local-minimum rate, requiring relative SSE improvement > 1e-06: 5 / 200 = 0.025.

| severity | n | raw non-default | raw rate | material non-default | material rate | material median rel SSE improvement |
|---|---|---|---|---|---|---|
| severe | 50 | 46 | 0.920 | 0 | 0.000 | 0.000000 |
| moderate | 50 | 43 | 0.860 | 0 | 0.000 | 0.000000 |
| mild_0.60_0.70 | 50 | 47 | 0.940 | 0 | 0.000 | 0.000000 |
| normal | 50 | 34 | 0.680 | 5 | 0.100 | 0.129542 |

## Parts 3-5 - Guarded Out

Robust loss, onset/back-extrapolation replacement, and combined final configuration were not propagated because the enabling vectorized-fit guard failed. This prevents accidental regression in already-passing headlines, especially severe-band FN = 0.000.

Accepted configuration: multi-start off; robust loss off; onset replacement off.

## Final Headline Deltas

| headline | metric | estimator | endpoint | baseline | final | delta | statement |
|---|---|---|---|---|---|---|---|
| Gate 2 parser | median_abs_fev1_error_pct |  |  | 2.908327 | 2.908327 | +0.000000 | UNCHANGED |
| Gate 2 parser | median_abs_fvc_error_pct |  |  | 3.527326 | 3.527326 | +0.000000 | UNCHANGED |
| Gate 2 parser | median_signed_fev1_error_pct |  |  | -2.885889 | -2.885889 | +0.000000 | UNCHANGED |
| Gate 2 parser | median_signed_fvc_error_pct |  |  | -3.489326 | -3.489326 | +0.000000 | UNCHANGED |
| Clinical fixed-0.70 | sensitivity |  |  | 0.982456 | 0.982456 | +0.000000 | UNCHANGED |
| Clinical fixed-0.70 | specificity |  |  | 0.956667 | 0.956667 | +0.000000 | UNCHANGED |
| Clinical fixed-0.70 | cohens_kappa |  |  | 0.911847 | 0.911847 | +0.000000 | UNCHANGED |
| Truncation clean-cut | obstructed_false_negative_rate | E1_OMBE | 2 | 0.589226 | 0.589226 | +0.000000 | UNCHANGED |
| Truncation clean-cut | obstructed_false_negative_rate | H1 | 2 | 0.579125 | 0.579125 | +0.000000 | UNCHANGED |
| Truncation clean-cut | obstructed_false_negative_rate | E1_OMBE | 3 | 0.595960 | 0.595960 | +0.000000 | UNCHANGED |
| Truncation clean-cut | obstructed_false_negative_rate | H1 | 3 | 0.579125 | 0.579125 | +0.000000 | UNCHANGED |
| Truncation clean-cut | obstructed_false_negative_rate | E1_OMBE | 4 | 0.518519 | 0.518519 | +0.000000 | UNCHANGED |
| Truncation clean-cut | obstructed_false_negative_rate | H1 | 4 | 0.494949 | 0.494949 | +0.000000 | UNCHANGED |
| Truncation clean-cut | obstructed_false_negative_rate | E1_OMBE | 5 | 0.468013 | 0.468013 | +0.000000 | UNCHANGED |
| Truncation clean-cut | obstructed_false_negative_rate | H1 | 5 | 0.441077 | 0.441077 | +0.000000 | UNCHANGED |
| Truncation field-realistic | obstructed_false_negative_rate | E0_DIRECT | field_all | 0.426966 | 0.426966 | +0.000000 | UNCHANGED |
| Truncation field-realistic | obstructed_false_negative_rate | E1_OMBE | field_all | 0.067416 | 0.067416 | +0.000000 | UNCHANGED |
| Truncation field-realistic | obstructed_false_negative_rate | H1 | field_all | 0.073034 | 0.073034 | +0.000000 | UNCHANGED |
| Severe-band clean recovery | severe_false_negative_rate | E1_OMBE | 2 | 0.000000 | 0.000000 | +0.000000 | UNCHANGED |
| Severe-band clean recovery | severe_false_negative_rate | E1_OMBE | 3 | 0.000000 | 0.000000 | +0.000000 | UNCHANGED |
| Severe-band clean recovery | severe_false_negative_rate | E1_OMBE | 4 | 0.000000 | 0.000000 | +0.000000 | UNCHANGED |
| Severe-band clean recovery | severe_false_negative_rate | E1_OMBE | 5 | 0.000000 | 0.000000 | +0.000000 | UNCHANGED |
| Severe-band clean recovery | severe_false_negative_rate | H1 | 2 | 0.000000 | 0.000000 | +0.000000 | UNCHANGED |
| Severe-band clean recovery | severe_false_negative_rate | H1 | 3 | 0.000000 | 0.000000 | +0.000000 | UNCHANGED |
| Severe-band clean recovery | severe_false_negative_rate | H1 | 4 | 0.000000 | 0.000000 | +0.000000 | UNCHANGED |
| Severe-band clean recovery | severe_false_negative_rate | H1 | 5 | 0.000000 | 0.000000 | +0.000000 | UNCHANGED |

VERDICT: OPTIMIZATION-NEUTRAL

No headline changed. Raw non-default-start wins were common, but material SSE improvements were rare, so the baseline deterministic fit appears to be at its practical ceiling under this guarded pass.
