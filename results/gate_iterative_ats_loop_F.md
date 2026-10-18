# Gate - Iterative ATS/ERS Maneuver Loop, Cycle F

VERDICT: LOOP-NEUTRAL - Repeats did not materially improve deployable coverage; unresolved cases remain censored, borderline, or discordant around 0.70.

## ATS/ERS Source

- Source: ATS/ERS 2019 Standardization of Spirometry, https://doi.org/10.1164/rccm.201908-1590ST.
- Protocol constants used here: at least 3 acceptable maneuvers, repeatability of the two largest FEV1 within 150 mL and the two largest FVC within 150 mL, maximum 8 attempts.

## Protocol Decisions

- TEST split only; TRAIN rows are ignored.
- The single-pass baseline is the current deployable recovery gate: post_recovery_finalized from gate_deferral_recovery_f.apply_recovery.
- The loop uses raw PRE attempt order from SPXRAW, but uses only existing OMBE-fitted maneuver rows for FEV1/FVC/ratio values; no solver is rerun.
- POST/Bron rows are audited as availability only and are never pooled into PRE repeats.
- A maneuver is model-usable only when it is SPAACC == Y and has a successful finite existing OMBE fit. Raw SPAACC counts are also retained in the output.
- At each k, prefix_stats receives only part.iloc[:k]. Later maneuvers are used only after termination for held-out proxy scoring.
- The ATS stop criterion is >=3 model-usable acceptable PRE maneuvers plus top-two FEV1 and FVC repeatability within 150 mL.
- Before ATS criteria are met, the loop keeps requesting PRE blows unless the 8-attempt cap is hit, the participant runs out of recorded PRE blows, or the best-test ratio is severe enough to refer.
- When ATS criteria are met, the same baseline finalization bands are retained: obstructed only below 0.60, normal only at or above 0.72, near-threshold results are referred rather than forced.
- At the ATS 8-cap, the best-test ratio is emitted with grade unless it is severe-range or the conservative normal-call guard blocks a normal call.
- The conservative normal-call guard is enforced at every terminal normal call: no ACCEPT_NORMAL if any model-usable acceptable prefix maneuver has corrected ratio <0.70.
- CENSORED means NHANES recorded no additional PRE maneuver; censored rows are never counted as finalized in the honest floor.
- Held-out accuracy uses only fitted PRE maneuvers after the stopping attempt. NHANES truth is used only for sens/spec/kappa and severe-FN safety scoring.

## Part 0 - Blow Availability and Censoring

- Raw source: `<PROJECT_ROOT>/NHANES_SpiroSBI/2009-2010/spxraw_f.sas7bdat`.
- Raw source note: using extracted SAS7BDAT from new nested root: <PROJECT_ROOT>/NHANES_SpiroSBI/2009-2010/spxraw_f.sas7bdat.
- TEST participants: 563.
- PRE median maneuvers: 4.0; PRE fraction with >=3 maneuvers: 0.996 (561/563).
- POST/Bron audit only: 153/563 TEST participants have POST/Bron rows. These are never pooled into the loop.

PRE maneuver count distribution:

| PRE count | n | fraction |
|---|---|---|
| 1 | 0 | 0.000 |
| 2 | 2 | 0.004 |
| 3 | 164 | 0.291 |
| 4 | 149 | 0.265 |
| 5 | 97 | 0.172 |
| 6 | 54 | 0.096 |
| 7 | 30 | 0.053 |
| 8+ | 67 | 0.119 |

POST/Bron maneuver count distribution (availability audit only):

| POST/Bron count | n | fraction |
|---|---|---|
| 0 | 410 | 0.728 |
| 1 | 0 | 0.000 |
| 2 | 0 | 0.000 |
| 3 | 76 | 0.135 |
| 4 | 38 | 0.067 |
| 5 | 22 | 0.039 |
| 6 | 6 | 0.011 |
| 7 | 6 | 0.011 |
| 8+ | 5 | 0.009 |

Censoring problem: NHANES records the maneuvers the technician performed and NHANES retained. The simulation cannot invent later PRE blows, so each session is censored at its recorded PRE count. Censored sessions are counted as not finalized in the honest floor.

## Part 1 - Causal Iterative Loop

The enforcing code path is `simulate_participant()` -> `prefix = part.iloc[:k].copy()` -> `prefix_stats(prefix)` -> `policy_decision(stats, ...)`. The decision function receives only the prefix `1..k`; later maneuvers are only used after termination for held-out scoring. `n_available` is used only to decide CENSORED after the prefix decision cannot stop.

| termination | decision | n | fraction all |
|---|---|---|---|
| ATS_CAP | ACCEPT_NORMAL | 5 | 0.009 |
| ATS_CAP | ACCEPT_OBSTRUCTED | 14 | 0.025 |
| ATS_CAP | REFER | 18 | 0.032 |
| CENSORED | CENSORED | 67 | 0.119 |
| FINALIZE | ACCEPT_NORMAL | 184 | 0.327 |
| FINALIZE | ACCEPT_OBSTRUCTED | 25 | 0.044 |
| REFER | REFER | 250 | 0.444 |

## Part 2 - Post-Repeat Coverage

Operational post-repeat coverage retains the current single-pass finalized cases and applies the causal repeat loop only to cases that the single-pass policy deferred. The strict from-scratch loop audit is also shown because Part 1 simulates every participant from k=1, but it is not the quoted operational post-repeat coverage.

| scope | finalized | denominator | coverage |
|---|---|---|---|
| single-pass baseline | 375 | 563 | 0.666 |
| post-repeat, non-censored deferred denominator | 391 | 532 | 0.735 |
| post-repeat honest floor | 391 | 563 | 0.694 |
| strict from-scratch loop audit | 228 | 563 | 0.405 |

- Coverage delta vs single-pass baseline: 0.028.
- Strict from-scratch loop audit delta vs single-pass baseline: -0.261. This audit is not the quoted operational post-repeat coverage because already-finalized cases would not be forced back into repeat/referral.
- Mean / median blows requested: 4.15 / 4.0.
- ATS 8-cap fraction: 0.066 (37/563).
- Censored fraction: 0.119 (67/563).
- Censored fraction among single-pass deferred cases: 0.165 (31/188).

Held-out maneuver proxy accuracy for newly resolved single-pass deferred cases:

| finalized participants | with held-out proxy | observed correct | observed n | observed blow acc | participant majority correct | participant majority n | participant majority acc |
|---|---|---|---|---|---|---|---|
| 16 | 4 | 5 | 7 | 0.714 | 3 | 4 | 0.750 |

Held-out maneuver proxy accuracy for all strict from-scratch loop finalizations:

| finalized participants | with held-out proxy | observed correct | observed n | observed blow acc | participant majority correct | participant majority n | participant majority acc |
|---|---|---|---|---|---|---|---|
| 228 | 63 | 86 | 92 | 0.935 | 59 | 63 | 0.937 |

Truth-derived diagnostic metrics (reported because the task requests sens/spec/kappa; not quoted as held-out accuracy):

| scope | finalized n | coverage | sens | spec | kappa | severe finalized | severe FN | severe FN CP high |
|---|---|---|---|---|---|---|---|---|
| single-pass baseline | 375 | 0.666 | 0.943 | 0.977 | 0.923 | 6 | 0 | 0.459 |
| post-repeat policy | 391 | 0.694 | 0.941 | 0.977 | 0.922 | 6 | 0 | 0.459 |
| strict from-scratch loop audit | 228 | 0.405 | 0.857 | 0.984 | 0.865 | 0 | 0 | NA |

## Part 3 - Does Repeating Help?

| single-pass deferred | resolved by loop | 0 addl | 1 addl | 2 addl | 3+ addl | never resolved |
|---|---|---|---|---|---|---|
| 188 | 16 | 0 | 0 | 2 | 14 | 172 |

Never-resolved deferred cases by existing R1-R4 taxonomy:

| reason | n | fraction |
|---|---|---|
| R3_FAILED_REPEATABILITY | 94 | 0.547 |
| R2_BORDERLINE_RATIO | 38 | 0.221 |
| R4_GENUINELY_UNINTERPRETABLE | 33 | 0.192 |
| R1_TRUNCATED_INCOMPLETE | 7 | 0.041 |

Repeated ratio behavior:

| scope | cases | with ratios | median spread | straddles 0.70 n | straddles 0.70 frac | median margin |
|---|---|---|---|---|---|---|
| resolved deferred | 16 | 16 | 0.056 | 3 | 0.188 | 0.063 |
| never-resolved deferred | 172 | 162 | 0.038 | 72 | 0.444 | 0.021 |

KEY QUESTION: repeated blows often keep landing on both sides of 0.70 in the never-resolved set. More blows do not reliably resolve those cases; this supports the biological-ambiguity interpretation.

## Part 4 - Safety Under the Loop

| path | finalized | severe finalized | severe FN | severe FN CP high |
|---|---|---|---|---|
| LOOP_FINALIZE | 209 | 0 | 0 | NA |
| LOOP_ATS_CAP | 19 | 0 | 0 | NA |
| STRICT_LOOP_OVERALL | 228 | 0 | 0 | NA |
| POST_REPEAT_POLICY | 391 | 6 | 0 | 0.459 |

- Normal-call guard violations across terminal normal calls: 0.
- ATS session grade is emitted for every finalized row in the participant CSV and summarized below.

| grade | n | fraction |
|---|---|---|
| A | 209 | 0.917 |
| B | 3 | 0.013 |
| C | 3 | 0.013 |
| D | 6 | 0.026 |
| E | 7 | 0.031 |

## Guardrail Statement

- The loop is PRE-only. POST/Bron rows are counted only in the availability audit.
- Deferred participants that become CENSORED in the repeat loop are not counted as newly finalized in the operational honest floor; the strict from-scratch audit also counts every CENSORED loop row as not finalized.
- Held-out accuracy uses only maneuvers after the stopping attempt, where such maneuvers exist.
- The NHANES fixed-ratio truth is used for sens/spec/kappa and severe-FN scoring, not for selection and not as the quoted accuracy.
- Severe FN=0 and the normal-call guard are required before this loop can be considered safe.

VERDICT: LOOP-NEUTRAL - Repeats did not materially improve deployable coverage; unresolved cases remain censored, borderline, or discordant around 0.70.
