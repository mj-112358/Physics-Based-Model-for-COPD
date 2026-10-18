# Methods

## Data source

The analyses use public NHANES spirometry data from cycles E, F, and G, corresponding to 2007-2008, 2009-2010, and 2011-2012. Raw curve files contain comma-delimited changes in expired volume at 0.01-second intervals. Each interval is multiplied by the record-specific BTPS factor before cumulative volume is computed.

Raw data are not committed to this repository. The official NHANES documentation states that the raw release includes both acceptable and poor-quality maneuvers and that `SPAACC` indicates individual-curve acceptability.

## Curve reconstruction

For an increment sequence \(d_i\), BTPS factor \(b\), and sample interval \(\Delta t=0.01\) s:

\[
V_k = b\sum_{i=1}^{k} d_i,
\qquad
Q_k = \frac{b d_k}{\Delta t}.
\]

The first cumulative volume is assigned to 0.01 s because it represents the first completed sampling interval. Forced-expiration onset is estimated from smoothed flow, followed by back-extrapolation around peak expiratory flow.

## Mechanistic model

The unnormalized expiratory profile is

\[
q(t)=\left[\alpha\frac{e^{-t/\tau_f}}{\tau_f}+(1-\alpha)\frac{e^{-t/\tau_s}}{\tau_s}\right]
\left(1-e^{-t/\tau_o}\right).
\]

Flow is normalized to a finite volume parameter:

\[
Q(t)=\mathrm{FVC}\frac{q(t)}{\int_0^H q(u)\,du},
\]

where \(H\) is the prespecified modeling horizon. The parameters are fast and slow time constants \(\tau_f\) and \(\tau_s\), mixing weight \(\alpha\), total volume FVC, and onset time constant \(\tau_o\).

The deterministic implementation uses bounded nonlinear least squares. Bayesian and neural posterior experiments are retained as research records but are not part of the supported clinical evidence because convergence and calibration criteria were not established on real curves.

## Evaluation settings

### Parser agreement

Participant-level reconstructed values are compared with NHANES summary fields. Because NHANES summary values and raw maneuvers come from the same session, this is an agreement check rather than independent clinical validation.

### Complete-curve adequacy

Residual variance is estimated per curve from deviations around a Savitzky-Golay smooth. A curve passes only when its normalized residual sum of squares is below the 95th-percentile chi-square threshold after accounting for five fitted parameters.

### Truncation evaluation

Complete acceptable maneuvers are cut at 2, 3, 4, and 5 seconds. Estimators receive only observations through the cut time. The corrected protocol uses a fixed 12-second prediction horizon for every truncated curve. It does not use the true complete-curve duration.

Comparators are direct end-volume, cubic smoothing spline extrapolation, linear tail extrapolation, and a single-exponential model with the same onset term.

### Obstruction labels

Analyses report either a fixed FEV1/FVC threshold of 0.70 or a GLI lower limit of normal. These are physiological classifications. They are not equivalent to a clinical COPD diagnosis.

## Statistical principles

- All splits are performed at participant level.
- Random seeds are recorded in result artifacts.
- Confidence intervals are reported for binomial rates and selected paired comparisons.
- Exploratory threshold searches are identified as exploratory.
- A held-out test set must not be used to choose an operating point.
- NHANES survey weights are required for population-level prevalence claims. Most committed gate reports do not apply the full complex-survey design, so their metrics are sample-level only.
