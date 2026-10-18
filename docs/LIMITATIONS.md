# Limitations and likely reviewer concerns

This document lists the issues most likely to be raised in academic review. They are stated directly so that negative findings are not hidden behind downstream performance summaries.

## Critical issues

### The outcome is not a COPD diagnosis

Most labels are based on FEV1/FVC from the same NHANES spirometry session. COPD requires clinical context and confirmation of persistent post-bronchodilator obstruction. The repository can support claims about spirometric obstruction or curve reconstruction, not COPD diagnosis.

### The current mechanistic form fails real-curve adequacy

The complete real-curve gate found no adequate fits under its corrected residual criterion. Classification metrics do not erase this structural failure. Any physiological interpretation of fitted parameters must therefore be treated cautiously.

### Historical truncation results used unavailable future information

The original truncation protocol supplied the complete maneuver duration as the extrapolation target. The model normalization also depended on that duration. This was information leakage. The primary Cycle F truncation analysis has now been regenerated with a fixed 12-second horizon and explicit tests that exclude post-cutoff information. Older three-cycle, degradation, and robustness artifacts remain provisional until they are regenerated under the same protocol.

### The positive truncation result has low absolute sensitivity

The corrected model improves substantially on direct use of a 2 to 3 second endpoint, but its sensitivity is only about 40 percent at those cut points. The result supports relative recovery value, not readiness for screening or diagnosis.

### Reference labels are partly circular

NHANES summary FEV1 and FVC values are composites of maneuvers from the same session used by the recovery rules. Agreement can therefore be high even without independent clinical validity. The repository's own recovery audit demonstrates this circularity.

## Major issues

### No independent external cohort

Cycles E, F, and G are separate NHANES waves, which supports temporal replication of parsing behavior. They are not independent clinical cohorts with separately adjudicated disease labels or device diversity.

### Complex survey design is not fully used

NHANES recommends examination weights and appropriate variance estimation for population inference. Most downstream model and classification reports use unweighted or class-reweighted samples rather than the full survey design. Population prevalence and nationally representative performance must not be inferred from them.

### Extensive sequential experimentation

The repository contains many gates, repairs, alternative thresholds, and follow-up analyses. This creates substantial researcher degrees of freedom. Later analyses should be considered exploratory unless the complete protocol and operating thresholds were frozen before evaluation.

### Some headline samples are constructed or enriched

The 414-person clinical classification experiment includes all available moderate and severe participants meeting its rules plus a sample of 300 normal participants. Its sensitivity and specificity are useful internal checks, but the sample construction is not a natural clinical prevalence cohort.

### Borderline decisions are unstable

Repeated-maneuver analysis shows instability near FEV1/FVC 0.70. Deferral is a reasonable safety response, but it lowers effective coverage and does not solve the underlying uncertainty.

### Posterior inference is not reliable on the real-data geometry

The NUTS ladder did not establish a converged minimum configuration, and later posterior calibration reports failed or remained partial. Posterior widths and probability claims from affected gates should not be treated as validated.

## Software and reproducibility issues addressed in this revision

- The package and all internal references now use the PredictX namespace.
- Packaging metadata and dependency groups are provided.
- Raw interval timestamps use one convention.
- Truncation estimators use a fixed horizon.
- Frontier operating-point selection uses training data only.
- Historical absolute workstation paths have been replaced with project-relative placeholders.
- Current and historical result status are separated in the top-level documentation.

## Work required before a strong confirmatory paper

1. Replicate the corrected truncation analysis in cycles E and G and regenerate all dependent robustness reports.
2. Freeze a single primary estimand and analysis plan before examining the final test cohort.
3. Validate against post-bronchodilator spirometry and an independent clinical endpoint.
4. Evaluate an external cohort collected on different devices and sites.
5. Apply appropriate NHANES survey methods for population-level claims.
6. Replace or extend the current flow model to address the peak-flow residual structure.
7. Establish parameter identifiability and posterior convergence before interpreting latent physiological parameters.
