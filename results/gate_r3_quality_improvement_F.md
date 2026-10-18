# Gate - R3 Recovery Quality Improvement, Cycle F

VERDICT: R3-UNRELIABLE - fit_weighted_r2 had the best restricted held-out accuracy (0.747) but stayed below the 0.90 bar; expanded R3 recovery is not retained.

## Protocol Decisions

- TEST split only; TRAIN rows are ignored.
- R3 truth accuracy is not quoted as an R3 quality metric; R3 method quality is judged by held-out maneuver proxy only.
- Held-out blow labels use observed held-out FEV1/FVC < 0.70, matching the fixed-threshold gate.
- The restricted held-out proxy keeps only held-out blows with SPAACC == Y and volume-fit R2 >= 0.95.
- FET is represented by maneuver duration_s because the reproducibility rows do not carry a separate FET field.
- A short/truncated held-out blow is proxied as duration_s < 6.0 seconds.
- Fit-filtered bars are predeclared at volume R2 >= 0.95 and >= 0.98; no search over bars is performed.
- Fit-weighted estimates use clipped nonnegative volume R2 weights, weight = clip(R2, 0, 1).
- The conservative normal-call guard is retained: do not auto-label normal while the selected candidate pool contains an obstructed maneuver.
- Expanded R3 recovery must reach restricted held-out blow accuracy >= 0.90 to be retained.

## Part 1 - Why Current R3 Held-Out Accuracy Is Low

| held-out group | n | R2 median | R2 Q1 | R2 Q3 | SPAACC Y rate | FET median | FET Q1 | FET Q3 | FET<6s rate | |ratio-0.70| median |
|---|---|---|---|---|---|---|---|---|---|---|
| agreed_with_besttest_call | 197 | 0.997 | 0.995 | 0.999 | 1.000 | 13.605 | 11.111 | 15.758 | 0.010 | 0.035 |
| disagreed_with_besttest_call | 92 | 0.998 | 0.995 | 0.999 | 1.000 | 10.893 | 8.843 | 13.588 | 0.065 | 0.018 |

Finding: Disagreeing held-out blows are not systematically lower quality; their median volume R2 is slightly higher than agreeing blows and they are near/above the 0.70 boundary.

## Part 2 - Restricted Held-Out Proxy

| scope | participants | blow correct | blow n | blow acc | participant majority correct | participant n | participant majority acc |
|---|---|---|---|---|---|---|---|
| original unrestricted | 81 | 197 | 289 | 0.682 | 52 | 81 | 0.642 |
| restricted good blows | 81 | 188 | 276 | 0.681 | 52 | 81 | 0.642 |

The restricted proxy does not rescue current broad R3 recovery; high-quality held-out blows still disagree often.
Note: isolating R3-only gives participant-majority accuracy 0.642; the earlier 0.663 headline included the residual R1 recoveries in the same newly finalized set.

## Part 3 - ATS Grade Stratification

| grade | participants | blow correct | blow n | restricted blow acc | participant majority acc |
|---|---|---|---|---|---|
| A | 62 | 145 | 210 | 0.690 | 0.645 |
| C | 8 | 16 | 28 | 0.571 | 0.500 |
| D | 5 | 12 | 18 | 0.667 | 0.600 |
| E | 6 | 15 | 20 | 0.750 | 0.833 |

Grade finding: Grade A is already below 0.90; reliability is not rescued by ATS grade.

## Part 4 - Fit-Quality-Arbitrated Best-Test

| variant | R3 finalized | net new vs current | coverage contribution | restricted blow correct | restricted blow n | restricted blow acc | participant majority acc | severe n | severe FN | severe FN CP high |
|---|---|---|---|---|---|---|---|---|---|---|
| naive_ats_besttest | 88 | 81 | 0.156 | 211 | 300 | 0.703 | 0.670 | 0 | 0 | NA |
| fit_filtered_r2_0_95 | 89 | 82 | 0.158 | 211 | 300 | 0.703 | 0.667 | 0 | 0 | NA |
| fit_filtered_r2_0_98 | 89 | 82 | 0.158 | 210 | 300 | 0.700 | 0.667 | 0 | 0 | NA |
| fit_weighted_r2 | 85 | 78 | 0.151 | 210 | 281 | 0.747 | 0.726 | 0 | 0 | NA |

Best restricted held-out variant: fit_weighted_r2 at blow accuracy 0.747.
That is a modest improvement over naive ATS best-test, but it is still far below the predeclared 0.90 retention bar and far below R1/R2 recovery.

## Part 5 - Final Tiered Policy

| tier | selected participants | with held-out proxy | coverage contribution | held-out blow correct | held-out blow n | held-out blow acc | participant majority acc | reason |
|---|---|---|---|---|---|---|---|---|
| tier1_r1_ombe_plus_r2_clear_ci | 113 | 110 | 0.201 | 361 | 366 | 0.986 | 0.991 | None |
| legacy_strict_r3_cluster_retained | 7 | 7 | 0.012 | 23 | 24 | 0.958 | 1.000 | None |
| tier2_expanded_r3_fit_arbitrated | 0 | NA | 0.000 | NA | NA | NA | NA | not retained because best restricted held-out accuracy is below the predeclared 0.90 bar |

| scope | finalized n | coverage | sens | spec | kappa | severe finalized | severe FN | severe FN CP high |
|---|---|---|---|---|---|---|---|---|
| retained final policy | 375 | 0.666 | 0.943 | 0.977 | 0.923 | 6 | 0 | 0.459 |

Final policy statement: Expanded ATS/physics R3 recovery is not retained; keep the current 66.6% high-confidence policy including the legacy strict R3 cluster.

## Guardrail Statement

- Label-independent selection is preserved; TEST truth is used only for severe-FN and final-set metric scoring.
- R3 quality claims use held-out maneuver proxy, not circular participant truth accuracy.
- Severe FN remains 0 in the retained final policy.
- Expanded broad R3 recovery is deferred because the held-out proxy says it is not yet defensible.

VERDICT: R3-UNRELIABLE - fit_weighted_r2 had the best restricted held-out accuracy (0.747) but stayed below the 0.90 bar; expanded R3 recovery is not retained.
