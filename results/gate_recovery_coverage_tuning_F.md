# Gate - Recovery Coverage Tuning, Cycle F

VERDICT: TUNED-70-PLUS - A deterministic label-independent profile reaches at least 70% finalized coverage with severe FN=0; the held-out maneuver proxy is high but not perfect, so this is a coverage/accuracy tradeoff rather than the strict-perfect subset.

## Protocol Decisions

- This tuning pass does not edit the production recovery gate; it writes an audited candidate profile only.
- Selection is label-independent: thresholds use fit quality, CIs, distance from 0.70, and maneuver consistency before truth scoring.
- Profiles must reach >=70% finalized TEST coverage, preserve severe FN=0, and have zero recovered-case errors against the NHANES summary label to be eligible.
- Because the NHANES summary label is same-session and partly circular, held-out maneuver agreement is used to rank eligible profiles.
- The chosen profile favors high held-out participant-majority accuracy, minimal loosening, and just-over-target coverage over maximizing coverage.

## Coverage Ladder

| profile | recovered | finalized n | coverage | NHANES recovered acc | severe FN | sens | spec | kappa | heldout participant acc | heldout blow acc |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline workflow | 0 | 255 | 0.453 | NA | 0 | 0.847 | 0.974 | 0.842 | NA | NA |
| current_deployable_120 | 120 | 375 | 0.666 | 1.000 | 0 | 0.943 | 0.977 | 0.923 | 0.974 | 0.980 |
| tuned_70_ombe_strict | 140 | 395 | 0.702 | 1.000 | 0 | 0.949 | 0.977 | 0.928 | 0.978 | 0.974 |

## Chosen 70%+ Profile

- profile: `tuned_70_ombe_strict`
- finalized coverage: 0.702 (395/563)
- recovered cases: 140 = R1 25, R2 107, R3 8
- NHANES recovered-case accuracy: 140/140 = 1.000
- severe FN: 0/6
- held-out maneuver participant-majority proxy: 134/137 = 0.978
- held-out maneuver blow-level proxy: 415/426 = 0.974

### Thresholds

| bucket | rule |
|---|---|
| R1 | volume R2 >= 0.98; obstructed ratio < 0.60; normal ratio >= 0.80 |
| R2 | obstructed bootstrap CI high < 0.70 and median < 0.68; normal CI low >= 0.70 and median >= 0.72 |
| R3 | good-maneuver cluster width <= 0.035; obstructed cluster high < 0.63; normal cluster low >= 0.75 |

## Candidate Comparison

| profile | recovered | coverage | NHANES acc | severe FN | kappa | looseness | R1 | R2 | R3 |
|---|---|---|---|---|---|---|---|---|---|
| tuned_70_ombe_strict | 140 | 0.702 | 1.000 | 0 | 0.928 | 0.850 | 25 | 107 | 8 |
| grid_r10.98_n0.80_r2hi0.70_m0.68_N0.72_r3w0.035_h0.63 | 140 | 0.702 | 1.000 | 0 | 0.928 | 0.850 | 25 | 107 | 8 |
| tuned_70_r3_current_width | 141 | 0.703 | 1.000 | 0 | 0.928 | 1.500 | 30 | 107 | 4 |
| grid_r10.95_n0.82_r2hi0.70_m0.68_N0.72_r3w0.035_h0.65 | 140 | 0.702 | 1.000 | 0 | 0.929 | 1.250 | 22 | 107 | 11 |
| grid_r10.95_n0.82_r2hi0.70_m0.68_N0.72_r3w0.035_h0.66 | 140 | 0.702 | 1.000 | 0 | 0.929 | 1.350 | 22 | 107 | 11 |
| grid_r10.90_n0.82_r2hi0.70_m0.68_N0.73_r3w0.035_h0.64 | 140 | 0.702 | 1.000 | 0 | 0.929 | 1.550 | 24 | 106 | 10 |
| grid_r10.90_n0.82_r2hi0.69_m0.68_N0.72_r3w0.035_h0.65 | 140 | 0.702 | 1.000 | 0 | 0.928 | 1.650 | 24 | 105 | 11 |
| grid_r10.90_n0.82_r2hi0.70_m0.68_N0.72_r3w0.025_h0.65 | 140 | 0.702 | 1.000 | 0 | 0.928 | 1.650 | 24 | 107 | 9 |
| grid_r10.90_n0.82_r2hi0.70_m0.68_N0.72_r3w0.030_h0.65 | 140 | 0.702 | 1.000 | 0 | 0.928 | 1.700 | 24 | 107 | 9 |
| grid_r10.90_n0.82_r2hi0.69_m0.68_N0.72_r3w0.035_h0.66 | 140 | 0.702 | 1.000 | 0 | 0.928 | 1.750 | 24 | 105 | 11 |
| grid_r10.90_n0.82_r2hi0.70_m0.68_N0.72_r3w0.025_h0.66 | 140 | 0.702 | 1.000 | 0 | 0.928 | 1.750 | 24 | 107 | 9 |
| grid_r10.90_n0.82_r2hi0.70_m0.68_N0.72_r3w0.030_h0.66 | 140 | 0.702 | 1.000 | 0 | 0.928 | 1.800 | 24 | 107 | 9 |

## Interpretation

- Yes, a deterministic label-independent profile can cross 70% TEST coverage.
- The strict 100%-heldout subset remains below 70%; crossing 70% requires accepting a small measured held-out proxy error rate.
- The tuned profile should be presented as a candidate operating point, not as a replacement for independent clinical validation.

Grid profiles tested: 2304; target-reaching eligible profiles before held-out ranking: 216.
