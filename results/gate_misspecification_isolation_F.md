# Gate - Misspecification Isolation, Cycle F

## Protocol Decisions

- Used the checked-in likelihood-repair sampler settings from `predictx/gate_likelihood_repair_f.py`: 2 chains, 250 warmup, 250 draws, stride 5. The initial specification states 4 chains x 2500 warmup/draws, but the reusable repo configuration does not.
- Cell A uses IID sigma 50.0 mL so L0 is exactly the data-generating likelihood. The initial specification also states 28 mL; that conflicts with the mandatory exact Cell A/L0 halt check. Cell B and dose-response use AR(1) marginal RMS 28.0 mL.
- The only arm-specific branch is data loading/generation. The same NUTS model, priors, seed policy, diagnostics, and coverage code are used for every generated curve.
- Real Cell C reuses the existing likelihood-repair TEST subset rows from `gate_likelihood_repair_F_rows.csv`; this is the same repo code path and preserves the requested SEQN list.
- Real-arm coverage targets observed curve FEV1/FVC, not known theta. Synthetic-arm coverage targets known theta. This asymmetry is not hidden.

## Verdict: IMPLEMENTATION-FAULT

- guard passed: True (max difference 1.599e-11 mL)
- Cell A/L0 nominal at all levels: False
- AR(1) innovation sd: nan mL; algebra: NOT-RUN
- halt reason: Mandatory Cell A/L0 nominal-coverage check failed; remaining cells, dose response, severity stratification, and figures were not run.
- prior-coverage diagnostics: {'tau_f': 1.0, 'tau_s': 1.0, 'alpha': 1.0, 'fvc_l': 1.0, 'tau_o': 1.0, 'note': 'Cell A theta were sampled directly from WIDE_PRIOR_RANGES; stratification rejection only selected by generated ratio, not by leaving prior support.'}

## 2x2 Coverage/PIT Summary

| cell | variant | n | cov50 | 50 CI lo | 50 CI hi | cov80 | cov95 | KS | KS p | ratio sd med | median Rhat | min ESS | div | Rhat>1.01 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cell_A_synth_iid | L0_baseline | 150 | 0.413 | 0.420 | 0.580 | 0.787 | 0.953 | 0.065 | 0.523 | 0.0030 | 1.031 | 2.0 | 251 | 0.833 |

## Decomposition

- implementation share at 50%: nan
- noise-model recovered share at 50%: nan
- model residual share at 50%: nan
- interaction term at 50%: nan
- arithmetic: NOT-RUN

## Dose Response

| eps | cov50 | cov80 | cov95 | KS | abs bias med |
|---|---|---|---|---|---|

- eps*: NOT-RUN
- identifiable: NOT-RUN; reason: NOT-RUN

## Obstruction Stratification: Real Cell C / L2

| stratum | n | cov50 | CI lo | CI hi | conclude? |
|---|---|---|---|---|---|

Severe stratum has n=4; no conclusion is drawn from it.

## Output Files

- `results/gate_misspecification_isolation_F.md`
- `results/gate_misspecification_isolation_F.json`
- `results/gate_misspecification_isolation_F_rows.csv`
- `results/gate_misspecification_isolation_F_summary.csv`
- `results/gate_misspecification_isolation_F_dose.csv`
- `results/gate_misspecification_isolation_F_severity.csv`
- figures: NOT-RUN because Cell A/L0 triggered the mandatory halt.
