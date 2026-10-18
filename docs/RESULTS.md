# Results and evidence status

## Evidence summary

| Question | Result | Status |
|---|---|---|
| Can the raw curves be reconstructed consistently? | Participant-level median absolute FEV1 and FVC errors are approximately 3 to 4 percent across cycles E, F, and G. | Supported as an agreement result |
| Does the five-parameter model fit complete real curves? | 0 of 31,851 cycle-F curves passed the corrected noise-normalized adequacy gate. | Refuted for the current model form |
| Does the model improve complete-curve classification? | In the broader reconciliation sample, pooled model kappa was 0.680 versus 0.918 for pooled direct measurement. | No advantage on complete curves |
| Is the model repeatable across maneuvers? | Model ratio ICC was 0.778 versus 0.793 for observed ratios. Borderline classification unanimity was 0.698. | Unstable near the threshold |
| Are Bayesian posterior claims established? | NUTS convergence remained inadequate across the tested ladder; several posterior calibration reports failed. | Not established |
| Does mechanistic extrapolation help after truncation? | In the corrected Cycle F experiment, the model reduced FVC error and false negatives at 2 to 3 seconds relative to direct truncation and the registered baselines. | Supported internally for the selected sample, not externally validated |
| Does a graded classifier add safe coverage? | Recalibration improved ECE, but the CLEAN point selected on training data failed the held-out NPV floor, 0.944 observed versus 0.950 required. | No confirmatory gain established |

## Parser result

The reconstructed values agree reasonably with NHANES summary fields in all three cycles. The cross-cycle report includes 6,319 participants in cycle E, 7,079 in cycle F, and 6,652 in cycle G for the relevant participant-level comparisons.

This agreement is important because it verifies decoding, BTPS adjustment, and basic curve processing. It does not prove diagnostic accuracy because the comparator is derived from the same spirometry session.

## Complete-curve model result

The decisive negative result is the real-curve adequacy gate. The median normalized residual statistic was 19.572, while the median adequacy threshold was 1.085. Residuals were concentrated around the rise to peak flow. This indicates structural mismatch, not merely optimizer failure or random noise.

The complete-curve reconciliation analysis reaches the same practical conclusion. Direct pooled measurements outperform the model on the primary fixed-threshold classification comparison.

## Corrected truncation result

The primary Cycle F truncation analysis was regenerated with a fixed 12-second prediction horizon. Every estimator received only measurements through the 2, 3, 4, or 5 second cut point. On the 497-participant stratified sample, the mechanistic model reduced pooled median absolute FVC error from 17.38 to 10.01 percent at 2 seconds and from 12.29 to 9.49 percent at 3 seconds. The obstruction false-negative rate fell from 0.990 to 0.593 at 2 seconds and from 0.865 to 0.596 at 3 seconds.

These relative gains met the registered internal criterion, including comparisons with the linear-tail, smoothing-spline, and one-compartment baselines. They do not establish clinical adequacy. Model sensitivity was only 0.407 at 2 seconds and 0.404 at 3 seconds. The sample was enriched and restricted to complete grade-A maneuvers, and its truth values came from the same spirometry session. The spline comparator was numerically unstable over long extrapolation horizons, so superiority over that comparator is not strong evidence by itself.

Earlier truncation artifacts that used the complete maneuver duration remain historical and must not be combined with the corrected result. The three-cycle and degradation analyses still require fixed-horizon regeneration.

## Defensible conclusion

The repository demonstrates a careful sequence of falsification, repair, and audit steps. Its strongest positive deliverables are the raw-curve processing framework and an internally replicated indication that constrained extrapolation can improve recovery from early termination. The current physiological model should still be presented as a hypothesis-generating component because its complete-curve form is inadequate, its short-curve sensitivity is low, and its incomplete-curve benefit has not been tested on an independent clinical cohort.
