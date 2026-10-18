# Three-Cycle Replication EFG

> Evidence status: historical and provisional. This report predates the fixed
> 12-second prediction horizon and has not been regenerated. Do not cite its
> truncation estimates as confirmatory evidence.

## Protocol Decisions

- F Gate 2 and clinical rows are reused from the migration-validated cache; F truncation and severe rows are regenerated through this script.
- Clean truncation uses the existing F rule: Pre, SPAACC == Y, SPAQEFF == A, complete plateau FET >= 6.0 s and terminal-second dV <= 0.025 L, max parsed FEV1, tie to lowest SPAMANU.
- Clean per-cycle truncation keeps cycles separate for replication and uses the prior 200-per-stratum cap.
- Complete-plateau cohort counts are stratified after selecting the best complete maneuver; any upward shift relative to the all-A selected maneuver is reported as complete_selection_severity_shift_n rather than a negative drop count.
- Pooled severe uses global keys of the form cycle:SEQN and selects the shortest usable separate flagged incomplete Pre maneuver per severe target patient.
- H1 is recomputed deterministically from the same OMBE fit at each truncation time; no stale F H1 cache is read.
- External validity uses SMQ020/SMQ040 for smoking, MCQ160G/MCQ160K for self-reported obstructive disease, and RIDAGEYR for age bands.

## Global Guards

| phase | param sets | grid n | max volume diff mL | passed |
|---|---|---|---|---|
| start | 24 | 1201 | 0.000000000004 | True |
| end | 24 | 1201 | 0.000000000003 | True |

- Parser identity: predictx.gate23_f.derive_curve served cycles E, F, and G; no per-cycle parser branch was used.

## Part 0 - Cohort Accounting

| cycle | raw rows | pre | acceptable pre | A pre | acceptable A pre | truth participants | first acceptable | first acceptable A |
|---|---|---|---|---|---|---|---|---|
| F | 36421 | 34413 | 31881 | 18676 | 18676 | 7079 | 7079 | 6638 |
| E | 34857 | 32353 | 29574 | 16391 | 16391 | 6319 | 6319 | 5813 |
| G | 37661 | 35642 | 32979 | 18395 | 18395 | 6652 | 6652 | 6228 |

## Part 1 - Parser Smoke And Curve Length

| cycle | raw rows | parse errors | usable selected A/B | SPXPTS median | SPXPTS p25 | SPXPTS p75 | SPXPTS finding |
|---|---|---|---|---|---|---|---|
| F | 36421 | 45 | 6905 | 853.0 | 597.0 | 1108.0 | unchanged |
| E | 34857 | 26 | 6107 | 866.0 | 593.0 | 1132.0 | unchanged |
| G | 37661 | 60 | 6447 | 832.0 | 548.0 | 1053.0 | unchanged |

## Part 2 - Gate 2 Per Cycle

| cycle | n | abs FEV1 % | abs FVC % | signed FEV1 % | signed FVC % | abs ratio % | BEV flag frac | PASS |
|---|---|---|---|---|---|---|---|---|
| E | 6319 | 3.143 | 3.716 | -3.133 | -3.657 | NA | 0.164 (1038/6319; CI 0.156-0.173) | True |
| F | 7079 | 2.908 | 3.527 | -2.886 | -3.489 | 4.338 | 0.151 (1066/7079; CI 0.142-0.159) | True |
| G | 6652 | 2.948 | 3.611 | -2.909 | -3.556 | 5.178 | 0.185 (1229/6652; CI 0.175-0.194) | True |

The F accuracy result reproduces on two independent NHANES waves.

## Part 3 - Truncation Recovery Per Cycle

| cycle | T | DIRECT FN | OMBE FN | H1 FN | DIRECT-OMBE | CI low | CI high | excludes 0 |
|---|---|---|---|---|---|---|---|---|
| E | 2 | 0.990 (298/301; CI 0.977-1.000) | 0.611 (184/301; CI 0.555-0.664) | 0.608 (183/301; CI 0.555-0.661) | 0.379 | 0.326 | 0.435 | True |
| E | 3 | 0.887 (267/301; CI 0.850-0.920) | 0.615 (185/301; CI 0.561-0.668) | 0.598 (180/301; CI 0.545-0.654) | 0.272 | 0.223 | 0.322 | True |
| E | 4 | 0.724 (218/301; CI 0.674-0.774) | 0.542 (163/301; CI 0.485-0.598) | 0.522 (157/301; CI 0.468-0.578) | 0.183 | 0.140 | 0.226 | True |
| E | 5 | 0.571 (172/301; CI 0.515-0.628) | 0.458 (138/301; CI 0.402-0.515) | 0.449 (135/301; CI 0.395-0.502) | 0.113 | 0.076 | 0.153 | True |
| F | 2 | 0.987 (293/297; CI 0.973-0.997) | 0.593 (176/297; CI 0.539-0.650) | 0.586 (174/297; CI 0.532-0.643) | 0.394 | 0.340 | 0.448 | True |
| F | 3 | 0.848 (252/297; CI 0.808-0.889) | 0.630 (187/297; CI 0.576-0.680) | 0.616 (183/297; CI 0.562-0.670) | 0.219 | 0.172 | 0.266 | True |
| F | 4 | 0.684 (203/297; CI 0.630-0.737) | 0.559 (166/297; CI 0.502-0.613) | 0.542 (161/297; CI 0.481-0.596) | 0.125 | 0.084 | 0.165 | True |
| F | 5 | 0.556 (165/297; CI 0.502-0.609) | 0.492 (146/297; CI 0.434-0.546) | 0.481 (143/297; CI 0.428-0.535) | 0.064 | 0.034 | 0.094 | True |
| G | 2 | 0.973 (286/294; CI 0.952-0.990) | 0.524 (154/294; CI 0.466-0.582) | 0.507 (149/294; CI 0.452-0.561) | 0.449 | 0.391 | 0.510 | True |
| G | 3 | 0.816 (240/294; CI 0.772-0.861) | 0.493 (145/294; CI 0.435-0.551) | 0.483 (142/294; CI 0.425-0.541) | 0.323 | 0.272 | 0.378 | True |
| G | 4 | 0.650 (191/294; CI 0.592-0.704) | 0.469 (138/294; CI 0.415-0.527) | 0.459 (135/294; CI 0.401-0.517) | 0.180 | 0.136 | 0.224 | True |
| G | 5 | 0.524 (154/294; CI 0.466-0.578) | 0.391 (115/294; CI 0.333-0.449) | 0.384 (113/294; CI 0.323-0.442) | 0.133 | 0.095 | 0.173 | True |

REPLICATION: REPLICATES-3-CYCLE

The direct-vs-OMBE recovery gap reproduces across E, F, and G at T=2 and T=3 with paired bootstrap CIs excluding zero.
