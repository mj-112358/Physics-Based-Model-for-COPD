# Gate - Onset / Back-Extrapolation Bias Test, Cycle F

## Protocol Decisions

- This gate is standalone from the fit-optimization vectorization path.
- No solver code is changed; no vectorization is used.
- Part 2 tests the literal volume-axis tangent requested here: t0 = t_PEF - V_PEF / PEF.
- The propagation scope is guarded: clinical/truncation recomputation only occurs if Gate 2 signed bias moves toward zero and absolute error does not increase.
- BEV compliance decision scope is `SPAACC == Y` acceptable Pre curves, because that is the Gate 2 criterion pool.

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

## Part 1 - Current Onset

Current code path:

- Gate 2: `predictx/gate23_f.py::derive_curve` -> `predictx.segment.detect_forced_exhale_segment` -> `predictx.segment.backextrap_t0` -> `predictx.segment.derive_fev1_fvc`.
- Clinical/truncation fit parsing: `predictx/gate_clinical_classification_f.py::extract_curve_for_fit` imports the same detection/back-extrapolation functions through `gate23_f`.
- Current t0 formula: `time_s[pef_idx] - (volume_ml[pef_idx] - volume_ml[onset_idx]) / (flow_lps[pef_idx] * 1000)`.

| scope | n | BEV mL med | BEV IQR low | BEV IQR high | BEV/FVC med | ATS violation | ATS compliant |
|---|---|---|---|---|---|---|---|
| all_parsed_pre | 34369 | 89.69 | 65.70 | 114.69 | 0.0274 | 0.0861 | 0.9139 |
| gate2_acceptable_pre | 31851 | 88.40 | 65.47 | 110.72 | 0.0269 | 0.0433 | 0.9567 |
| clinical_acceptable_ab_pre | 24119 | 90.18 | 69.02 | 111.93 | 0.0262 | 0.0350 | 0.9650 |
| grade_a_acceptable_pre | 18673 | 89.10 | 69.55 | 108.46 | 0.0253 | 0.0141 | 0.9859 |

Decision: Gate 2 acceptable Pre curves are ATS-compliant at 0.9567. The >95% stop rule is therefore met.

## Part 2 - Volume-Axis ATS Tangent Diagnostic

| onset | n | FEV1 signed % | FEV1 abs % | FVC signed % | FVC abs % | ratio abs % |
|---|---|---|---|---|---|---|
| current baseline | 7079 | -2.8859 | 2.9083 | -3.4893 | 3.5273 | 4.3383 |
| volume-axis tangent | 7079 | -2.9052 | 2.9260 | -3.5069 | 3.5441 | 4.3341 |

Bootstrap deltas are volume-axis minus current baseline on the same participant set:

| metric | n | baseline | axis | delta | CI low | CI high |
|---|---|---|---|---|---|---|
| fev1_signed_pct | 7079 | -2.885889 | -2.905242 | -0.019353 | -0.029490 | -0.011254 |
| fev1_abs_pct | 7079 | 2.908327 | 2.926004 | 0.017677 | 0.010333 | 0.028463 |
| fvc_signed_pct | 7079 | -3.489326 | -3.506902 | -0.017575 | -0.026382 | -0.006245 |
| fvc_abs_pct | 7079 | 3.527326 | 3.544068 | 0.016742 | 0.006450 | 0.030514 |
| ratio_abs_pct | 7079 | 4.338320 | 4.334118 | -0.004203 | -0.010654 | 0.007589 |

Part 2 pass: False. The candidate moves neither FEV1 nor FVC signed bias toward zero and slightly increases absolute percent error, so it is not an acceptable onset replacement.

## Part 3 - Propagation

Part 3 was not run. The Gate 2 acceptable-curve BEV stop rule fired and the volume-axis tangent candidate did not pass the signed-bias/absolute-error guard, so clinical/truncation recomputation would only risk regressing already-passing headlines.

## Part 4 - Verdict

VERDICT: ONSET-NEUTRAL

Current acceptable Pre curves are already >95% ATS-compliant by BEV, and the literal volume-axis tangent does not reduce the known Gate 2 signed low bias. The retained baseline onset leaves all headlines unchanged; the remaining Gate 2 bias is best characterized as composite/selection mismatch rather than onset-driven.
