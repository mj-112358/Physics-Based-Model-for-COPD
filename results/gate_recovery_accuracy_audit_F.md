# Gate - Recovery Perfect-Accuracy Sanity Audit, Cycle F

VERDICT: ACCURACY-CIRCULAR - NHANES truth and recovery share same-session spirometry inputs; R2/R3 include the nearest truth-component maneuvers, loosened criteria still score 1.000 against NHANES truth, and the held-out-maneuver proxy is below 1.000.

## Part 1 - Independence Check

- Truth label: truth_label(row) = OBSTRUCTED if fev1_fvc_ratio < 0.70 else NORMAL.
- Truth ratio: merged_summary.parquet computes fev1_fvc_ratio = fev1_mL / fvc_mL.
- Cycle-F source fields: `SPXNFEV1` and `SPXNFVC`.
- Caveat: Cycle-F fev1_mL/fvc_mL are NHANES SPX summary best-test/composite columns, not an independent clinical adjudication.
- Recovery inputs: R1/R2/R3 recovery uses OMBE/corrected ratios fitted from NHANES raw maneuvers from the same spirometry session.

| bucket | recovered | overlap assessed | missing maneuver rows | input maneuvers median | FEV1-nearest in input | FVC-nearest in input | ratio-nearest in input | selected is FEV1-nearest | selected is FVC-nearest | nearest ratio delta med |
|---|---|---|---|---|---|---|---|---|---|---|
| R1_TRUNCATED_INCOMPLETE | 25 | 22 | 3 | 1.0 | 1.000 | 0.682 | 0.545 | 1.000 | 0.682 | 0.0054 |
| R2_BORDERLINE_RATIO | 88 | 88 | 0 | 3.0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.523 | 0.0035 |
| R3_FAILED_REPEATABILITY | 7 | 7 | 0 | 3.0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.571 | 0.0073 |

Interpretation: R2/R3 recovery uses the same-session maneuver pool that contains the maneuvers nearest to the NHANES summary FEV1, FVC, and ratio components. The 1.000 score against NHANES truth is therefore not a fully independent clinical-label accuracy.

## Part 2 - Margin Analysis

| group | n | median abs margin | IQR low | IQR high | min | p10 | p90 | <0.02 | >=0.05 |
|---|---|---|---|---|---|---|---|---|---|
| recovered | 120 | 0.0646 | 0.0476 | 0.0920 | 0.0131 | 0.0328 | 0.1793 | 0.017 | 0.733 |
| residual_deferred | 188 | 0.0253 | 0.0099 | 0.0590 | 0.0001 | 0.0038 | 0.1706 | 0.447 | 0.293 |

Recovered cases are the easy tail: their median true-ratio margin is much larger than the residual deferred set, and only a small fraction sit within 0.02 of the threshold.

### Margins By Bucket

| bucket | group | n | median | IQR low | IQR high | <0.02 | >=0.05 |
|---|---|---|---|---|---|---|---|
| R1_TRUNCATED_INCOMPLETE | recovered | 25 | 0.1790 | 0.1399 | 0.2429 | 0.000 | 1.000 |
| R1_TRUNCATED_INCOMPLETE | residual | 13 | 0.1519 | 0.0847 | 0.2487 | 0.077 | 0.846 |
| R2_BORDERLINE_RATIO | recovered | 88 | 0.0566 | 0.0426 | 0.0743 | 0.023 | 0.659 |
| R2_BORDERLINE_RATIO | residual | 39 | 0.0150 | 0.0083 | 0.0203 | 0.744 | 0.000 |
| R3_FAILED_REPEATABILITY | recovered | 7 | 0.0761 | 0.0499 | 0.0780 | 0.000 | 0.714 |
| R3_FAILED_REPEATABILITY | residual | 103 | 0.0199 | 0.0081 | 0.0372 | 0.505 | 0.184 |
| R4_GENUINELY_UNINTERPRETABLE | residual | 33 | 0.1178 | 0.0503 | 0.2127 | 0.061 | 0.758 |

## Part 3 - Stress The Criteria

Current code uses R1 volume R2 >= 0.98. The requested stress analysis names 0.90 -> 0.85, so both R1 >=0.90 and R1 >=0.85 variants are reported.

| variant | finalized | NHANES accuracy | wrong | severe finalized | severe FN | R1 | R2 | R3 |
|---|---|---|---|---|---|---|---|---|
| strict_current | 120 | 1.000 | 0 | 4 | 0 | 25 | 88 | 7 |
| R1_volume_R2_min_0p90 | 125 | 1.000 | 0 | 6 | 0 | 30 | 88 | 7 |
| R1_volume_R2_min_0p85 | 125 | 1.000 | 0 | 6 | 0 | 30 | 88 | 7 |
| R2_median_band_0p68_0p72 | 139 | 1.000 | 0 | 4 | 0 | 25 | 107 | 7 |
| R3_cluster_width_0p030 | 122 | 1.000 | 0 | 4 | 0 | 25 | 88 | 9 |
| all_loosened | 146 | 1.000 | 0 | 6 | 0 | 30 | 107 | 9 |

Accuracy remains 1.000 under all requested loosened criteria. That does not prove the criteria are magically perfect; combined with the overlap table, it is evidence that NHANES truth is not independent of the same-session maneuver information.

## De-Circularized Same-Session Proxy

Because this dataset does not contain an external independent clinical adjudication label, a fully corrected independent clinical-label accuracy is not identifiable from these files. As the closest available check, recovered decisions were scored against held-out raw maneuvers from the same session: for R2/R3, the candidate was regenerated after leaving one maneuver out; for R1, the selected-maneuver decision was scored against other available maneuvers.

| bucket | heldout blows | blow correct | blow accuracy | participants | all-correct participants | all-correct fraction |
|---|---|---|---|---|---|---|
| R1_TRUNCATED_INCOMPLETE | 47 | 46 | 0.979 | 22 | 21 | 0.955 |
| R2_BORDERLINE_RATIO | 294 | 289 | 0.983 | 88 | 84 | 0.955 |
| R3_FAILED_REPEATABILITY | 17 | 16 | 0.941 | 7 | 6 | 0.857 |

- held-out blow accuracy: 0.980 (351/358)
- held-out participant majority accuracy: 0.974 (114/117)
- participant all-heldout-correct fraction: 0.949 (111/117)
- recovered participants without a held-out proxy: 3

## Strict Rescue Profile

A stricter deployable subset was evaluated using only recovery confidence signals. This does not rescue the full 120-case claim, but it does find a smaller profile whose held-out-maneuver proxy is perfect.

| profile | selected | dropped from 120 | coverage if used | NHANES accuracy | heldout blow accuracy | participant all-correct | without proxy | verdict if used |
|---|---|---|---|---|---|---|---|---|
| strict_rescue_85 | 85 | 35 | 0.604 | 1.000 | 1.000 | 1.000 | 3 | ACCURACY-GENUINE-STRICT-SUBSET |

- bucket counts: {'R2_BORDERLINE_RATIO': 61, 'R1_TRUNCATED_INCOMPLETE': 20, 'R3_FAILED_REPEATABILITY': 4}
- strict thresholds: R1 normal requires corrected ratio >= 0.82; R2 requires CI high < 0.67 and median < 0.66; R3 requires cluster high < 0.63.

| bucket | heldout blows | blow correct | blow accuracy | participants | all-correct participants | all-correct fraction |
|---|---|---|---|---|---|---|
| R1_TRUNCATED_INCOMPLETE | 38 | 38 | 1.000 | 17 | 17 | 1.000 |
| R2_BORDERLINE_RATIO | 197 | 197 | 1.000 | 61 | 61 | 1.000 |
| R3_FAILED_REPEATABILITY | 10 | 10 | 1.000 | 4 | 4 | 1.000 |

## Part 4 - Honest Verdict

- The 120/120 NHANES-truth accuracy is inflated by same-session circularity; it should not be described as independent clinical-label accuracy.
- The recovery rules are still conservative: they recover cases farther from 0.70 and leave the hard borderline/inconsistent cases deferred.
- The strict rescue profile is the honest path toward `ACCURACY-GENUINE-STRICT`: it lowers recovered cases from 120 to 85 and is perfect on all available held-out-maneuver proxy checks.
- The claimable statement is: recovery selected an easy, same-session-consistent tail with 1.000 agreement to NHANES summary truth, but the de-circularized held-out-maneuver proxy is below perfect.

VERDICT: ACCURACY-CIRCULAR - NHANES truth and recovery share same-session spirometry inputs; R2/R3 include the nearest truth-component maneuvers, loosened criteria still score 1.000 against NHANES truth, and the held-out-maneuver proxy is below 1.000.
