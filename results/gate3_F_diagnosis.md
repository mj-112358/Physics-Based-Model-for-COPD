# Gate 3 Diagnosis - Flow Peak Fatality Check

- status: FIXABLE_TIMING_SUSPECT
- reason: Dipole peak residuals dominate, supporting onset/phase misregistration as the leading fixable cause.
- sample: 200 A-grade non-obstructed curves selected with fixed seed 20260707
- eligible A-grade non-obstructed curves without prior fit error: 16807
- valid analyzed curves: 200/200

## Check 1 - Dipole vs Undershoot

- residual convention: actual flow minus fitted flow.
- dipole fraction: 0.845
- single-signed fraction: 0.010
- single-signed undershoot fraction: 0.000
- single-signed overshoot fraction: 0.010

| peak class | n | fraction |
|---|---:|---:|
| dipole | 169 | 0.845 |
| mixed | 29 | 0.145 |
| single_signed_overshoot | 2 | 0.010 |

## Check 2 - Fit Domain and Clinical Integrals

- current fitter domain: predictx/fit.py fits the FLOW-time curve: fit_ombe_lm(t, flow_lps) minimizes np_ombe_flow(params, t) - flow_lps.
- volume refit: OMBE parameters were re-fit to parsed volume-time curves for the same 200 curves.
- volume NRSS valid curves: 200
- volume NRSS_norm median: 1278.271
- volume NRSS_norm IQR: 570.531 to 2617.728
- volume chi-square threshold median: 1.079
- volume adequate by corrected NRSS: 0/200 (0.0%)
- volume R^2 median: 0.99797

Clinical metrics from volume-fit OMBE vs parsed same-curve volume truth:
- all FEV1, FVC, and ratio within 5%: 168/200 (84.0%)
- FEV1/FVC ratio within 5%: 171/200 (85.5%)
- FEV1/FVC ratio within 0.02 absolute: 135/200 (67.5%)
- median abs FEV1 error: 0.70%
- median abs FVC error: 1.01%
- median abs FEV1/FVC percent error: 1.49%
- median abs FEV1/FVC absolute error: 0.0114

## Check 3 - Steltner Comparability

Unresolved: local repo search found only internal mentions of Steltner/2.76, and web search did not locate a primary source for whether the benchmark used flow-time or volume-time residuals. Do not claim comparability until the paper is supplied or verified.

## Verdict

- Dipole-dominant means the leading cause is fit registration/timing, not settled physical inadequacy.
- Single-signed with clinical metrics within 5% means the flow-peak miss is clinically non-fatal for FEV1/FVC outputs.
- Single-signed with clinical metrics failing means OMBE needs an augmented onset model.

Observed verdict: FIXABLE_TIMING_SUSPECT - Dipole peak residuals dominate, supporting onset/phase misregistration as the leading fixable cause.
