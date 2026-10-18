# Project overview

## Aim

This PredictX extension asks whether a low-dimensional physiological model can recover useful spirometric measurements from raw forced-expiration recordings, especially when the recording is incomplete. The target quantities are FEV1, FVC, and their ratio. The clinical motivation is that premature termination tends to underestimate FVC, which can inflate FEV1/FVC and conceal airflow obstruction.

The work is best understood as a model-based signal-recovery study. It is not a validated COPD diagnostic system.

## Motivation

Conventional spirometry uses directly measured FEV1 and FVC after quality control and repeated maneuvers. This is appropriate when a technically acceptable plateau is present. It becomes less reliable when expiration ends before FVC is reached.

The proposed approach represents expiratory flow as an onset-modulated mixture of fast and slow exponential compartments. The model imposes a smooth, monotone approach to a finite total expired volume. In principle, the early portion of a curve can then constrain its later volume.

## Analysis pipeline

1. Parse the 100 Hz NHANES raw volume increments.
2. Apply the BTPS correction factor supplied by NHANES.
3. Identify the forced-expiration segment and estimate time zero by back-extrapolation.
4. Reconstruct observed volume, FEV1, FVC, and FEV1/FVC.
5. Fit the mechanistic curve in flow or volume space.
6. Compare model-derived measurements with direct measurements, GLI reference limits, and NHANES summary fields.
7. Evaluate complete curves, synthetically truncated curves, repeated maneuvers, and deterministic triage policies.

## How the approach differs from conventional methods

| Approach | Information used | Main strength | Main weakness |
|---|---|---|---|
| Direct measurement | Observed volume through the recorded endpoint | Simple, standard, and accurate for complete maneuvers | Underestimates FVC after early termination |
| Spline or linear extrapolation | Local smoothness or tail trend | Flexible and easy to fit | Limited physiological structure and unstable long-range extrapolation |
| Machine-learning classifier | Engineered features or the full signal | Can learn complex predictive patterns | Requires representative labels and can obscure failure mechanisms |
| Mechanistic model in this repository | A constrained two-timescale expiratory model | Interpretable parameters and a finite-volume constraint | The specified shape does not adequately fit complete real curves |

## Main conclusion

The parser and evaluation framework are useful. The current mechanistic form is not an adequate general model of complete NHANES spirometry curves. In the corrected Cycle F fixed-horizon experiment, it nevertheless improved FVC recovery and reduced false negatives after truncation at 2 to 3 seconds. That result supports further study of constrained extrapolation, but low absolute sensitivity, same-session reference values, and the absence of external validation prevent a clinical performance claim.
