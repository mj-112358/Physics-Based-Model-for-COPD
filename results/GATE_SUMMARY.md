# Evidence summary

This file replaces the obsolete synthetic-only gate summary. See `docs/RESULTS.md` for interpretation and `docs/LIMITATIONS.md` for reviewer-facing limitations.

## Supported

- Raw-curve parsing and BTPS reconstruction reproduce NHANES summary measurements with approximately 3 to 4 percent median absolute error across cycles E, F, and G.
- Synthetic segmentation, deterministic optimization, and simulation-based inference machinery pass their internal synthetic checks.
- The corrected Cycle F truncation analysis uses a fixed 12-second horizon and excludes post-cutoff information. At 2 to 3 seconds, the two-compartment model improves FVC recovery and obstruction false-negative rates relative to direct truncation and the registered extrapolation baselines.

## Negative or unresolved

- The current five-parameter model fails the complete real-curve adequacy criterion.
- Complete-curve model classification is inferior to pooled direct measurement in the broader reconciliation analysis.
- Borderline repeatability is unstable.
- Real-data posterior convergence and calibration are not established.
- Older three-cycle and robustness truncation findings remain provisional because their legacy protocol used the true complete-curve end time.
- The CLEAN graded probability model has not demonstrated a confirmatory safe-coverage gain on the held-out test set.

## Interpretation boundary

These analyses concern reconstruction of spirometric measurements and classification of airflow obstruction. They do not establish a clinical COPD diagnosis.
