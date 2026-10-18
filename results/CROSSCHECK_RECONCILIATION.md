# Crosscheck Reconciliation Audit

- Scope: deterministic reconciliation of the six issues from the prior crosscheck.
- No NUTS, Bayesian reruns, or science-gate reruns were performed.
- The only resampling was the requested deterministic patient bootstrap for held-out kappa CIs: 2000 resamples, seed 20260710.
- F raw/SPX/ZIP files at the legacy F location and `NHANES_SpiroSBI/2009-2010` were byte-identical by `cmp`, so current non-stale provenance strings were relabeled to the new nested path.

## Resolution

| issue | wrong | correct value | changed | file:line |
|---|---|---|---|---|
| 1. Gate 5 SBC md/json conflict | `gate5_sbc.json` carried stale `FAIL` from the older 200-replication SBC JSON; `gate5_sbc.md` and the newer coverage JSON were already PASS. | PASS. Gate criterion is persistent below-band only when a parameter is below the 2-SE MC band at >=3 nominal levels. Current 500-replication coverage has max below-band count 2 (`tau_o`), below threshold 3; persistent params = none. | Rewrote `gate5_sbc.json` status/reason/criterion/current coverage while retaining prior rank histograms. Stale file was `gate5_sbc.json`; markdown was correct. | results/gate5_sbc.json:2; results/gate5_sbc.json:5; results/gate5_sbc.json:23 |
| 2. Severe pooled 0/46 degenerate CI | Severe pooled OMBE field-incomplete FN was shown as `0/46; CI 0.000-0.000`. | Exact two-sided Clopper-Pearson 95% upper bound for 0/46 = 0.077061827; rule-of-three upper = 0.065217391. | Updated severe pooled field-incomplete zero-event intervals in markdown and JSON. E4 mono-exponential had the same 0/46 degeneracy and was corrected with the same bound. | results/gate_severe_cohort_pooled_EFG.md:22; results/gate_severe_cohort_pooled_EFG.md:28; results/gate_three_cycle_EFG.json:2916; results/gate_three_cycle_EFG.json:2924 |
| 3. Held-out H1 vs observed kappa CIs missing | Point kappas existed, but no bootstrap intervals were stored, so overlap could not be verified from the files. | Observed P3 kappa 0.700138, 95% CI 0.656123-0.738010. H1 P3 kappa 0.683768, 95% CI 0.640189-0.722241. Paired observed-minus-H1 delta 0.016370, 95% CI -0.006159 to 0.038451. | Rebuilt the existing held-out endpoint deterministically, reproduced stored point kappas exactly, added bootstrap CIs to md/json/rows. CIs overlap and paired delta CI includes zero, so the indistinguishable/overlap claim is supported. | results/gate_estimand_hybrid_F.md:47; results/gate_estimand_hybrid_F.md:49; results/gate_estimand_hybrid_F.json:53; results/gate_estimand_hybrid_F.json:567; results/gate_estimand_hybrid_F_rows.csv:2234 |
| 4. S7 pass fraction not explicit | The informal `~45% pass` reading was not the requested conjunctive pass rule. | S7 exact pass is 7/30 = 0.233333 under `max_rhat <= 1.01 AND min_bulk_ess >= 400 AND divergence_count == 0`. The ~0.43 value is only marginal ESS>=400 (13/30), not pass. | Added explicit conjunctive numerator/denominator/fraction to S7 markdown and JSON. | results/gate_sampler_pilot_F.md:23; results/gate_sampler_pilot_F.json:54; results/gate_sampler_pilot_F.json:64 |
| 5. Stale F-only / E-G-blocked wording | Historical F-only wording implied E and G were unavailable. | Historical F gates now say E and G are analyzed separately in the three-cycle extension. Numbers/verdicts were not changed. | Replaced descriptive language only in the flagged result files. | results/gate3_F_ombe.md:5; results/gate3_F_ombe_initial_nrss276.json:4; results/gate3_F_ombe_initial_nrss276.md:5; results/gate3_F_ombe_rerun.md:5; results/gate_truncation_hardening_F.json:409; results/gate_truncation_hardening_F.md:62; results/migration_EFG.json:290; results/migration_regression_F.md:47 |
| 6. Stale data provenance | Non-stale F result artifacts still cited the legacy F raw path. | Current provenance is `<PROJECT_ROOT>/NHANES_SpiroSBI/2009-2010/spxraw_f.sas7bdat` with note `using extracted SAS7BDAT from new nested root`. | Updated current artifacts only. Did not relabel artifacts explicitly marked regenerate by migration stale audit. | results/gate2_F_scaling_diagnostic.json:2; results/gate2_F_scaling_diagnostic.md:5; results/gate3_F_diagnosis.json:4; results/gate3_F_ombe.json:4; results/gate_estimand_hybrid_F.json:560; results/gate_model_vs_observed_F.json:1414; results/gate_offset_lungsize_F.json:103; results/gate_onset_bias_F.json:3; results/gate_reproducibility_volume_F.json:4 |

## Issue 1 Detail

- Current gate statistic: max persistent below-band count = 2.
- Threshold: >=3 nominal levels below the Monte Carlo band for the same parameter.
- Implied verdict: PASS.
- Divergence cause: `gate5_sbc.json` mtime was earlier than the coverage-only result and markdown; it preserved the older 200-replication overconfidence wording, while `gate5_coverage.json` and `gate5_sbc.md` recorded the newer 500-replication MC-band criterion.

## Issue 2 Detail

- Exact two-sided Clopper-Pearson 95% upper bound for 0 successes in 46: 0.077061827.
- Rule-of-three upper bound: 0.065217391.
- Quotation-safe severe OMBE claim: `0/46, 95% CI 0 to 0.077`.

## Issue 3 Detail

| estimator | method | n | kappa | 95% CI | bootstrap n | seed |
|---|---|---:|---:|---|---:|---:|
| observed | P3 | 3328 | 0.700138 | 0.656123-0.738010 | 2000 | 20260710 |
| H1 | P3 | 3328 | 0.683768 | 0.640189-0.722241 | 2000 | 20260710 |

- Observed-minus-H1 paired delta: 0.016370, 95% CI -0.006159 to 0.038451.
- CI overlap: yes.
- Paired delta CI includes zero: yes.
- Claim status: supported after adding the bootstrap fields.

## Issue 5 Detail

Additional stale-language edits not all shown in the compact table:

- `results/gate_truncation_hardening_F.json:414` updated Gate 3 descriptive value.
- `results/gate_truncation_hardening_F.json:2907` updated limitation text.
- `results/gate_truncation_hardening_F.md:63` updated Gate 3 row.
- `results/gate_truncation_hardening_F.md:251` updated limitation row.
- `results/migration_EFG.json:484`, `results/migration_EFG.json:494`, and `results/migration_EFG.json:503` updated source-audit text.
- `results/migration_regression_F.md:71`, `results/migration_regression_F.md:72`, and `results/migration_regression_F.md:73` updated source-audit text.

## Issue 6 Detail

Relabeled current/non-stale provenance:

- `results/gate3_F_gamma_onset.json:92`
- `results/gate3_F_ombe_rerun.json:4`
- `results/gate3_F_registered_fix.json:4`
- `results/gate_estimand_hybrid_F.json:560`
- `results/gate_model_vs_observed_F.json:1414`
- `results/gate_offset_lungsize_F.json:103`
- `results/gate_posterior_width_F.json:304`
- `results/gate_reconciliation_audit_F.json:3`
- `results/gate_reconciliation_audit_F.md:5`
- `results/gate_reproducibility_volume_F.json:4`
- `results/gate_two_path_reconciliation_F.json:2`

Flagged and intentionally not relabeled:

- `results/gate_truncation_recovery_F.json:40-41`
- `results/gate_truncation_hardening_F.json:419-420`
- `results/gate_severe_truncation_diagnosis_F.json:3-4`

Reason: these are explicitly classified for regeneration in `results/migration_stale_audit.md:17-22` and `results/migration_EFG.json:661-690`.

## Verification

- No stale F-only/E-G-blocked text remains in live result files after the edits; the report no longer carries the unresolved old strings.
- Legacy raw provenance remains only in the three files intentionally flagged for regeneration.
- Gate5 SBC, severe CI, held-out bootstrap CI, and S7 pass fraction were re-read from the edited artifacts after writing.

## Final Verdict

RECONCILED-CLEAN
