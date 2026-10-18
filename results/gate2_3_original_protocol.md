# Original Real-Curve Evaluation Protocol

This file records the prespecified decision criteria used before the raw NHANES
curve analysis was run. It is retained to distinguish planned criteria from
post hoc interpretation.

## Parser validation

Raw comma-separated samples were interpreted as 10 ms volume increments.
Flow was calculated as `increment * 0.1 * SPAFACT`, and cumulative volume as
`cumsum(increment) * SPAFACT`. The parser was required to recover both FEV1 and
FVC with a median absolute percentage error no greater than 5 percent on
acceptable pre-bronchodilator efforts.

## Real-curve model adequacy

The physics-based model was fitted to acceptable pre-bronchodilator curves.
Its normalized residual sum of squares was compared with the prespecified
noise-based threshold. The planned adequacy criterion was a pass fraction of
at least 85 percent. The threshold was not to be relaxed after inspection of
the real data.

Results were to be reported overall, by effort-quality grade, and by fixed
ratio obstruction status. A failure was to be treated as evidence that the
two-compartment model required structural augmentation before downstream
inference could be considered established.

## Interpretation

The parser criterion was met, but the model-adequacy criterion was not. The
observed pass fraction was 0 of 31,851 Cycle F curves under the corrected
noise-normalized criterion. This negative result is therefore part of the
primary scientific conclusion, not an implementation detail to be hidden or
retuned away.
