# Gate 2+3 - Real-Curve Verdict Prompt

Status: STAGED ONLY. Do not run until real `data/SPXRAW_E`, `data/SPXRAW_F`, and `data/SPXRAW_G` raw-curve files exist and pass size/header checks.

Current blocker observed July 7, 2026:

- `data/SPXRAW_E.rds`: 44 bytes, stub
- `data/SPXRAW_F.rds`: 44 bytes, stub
- `data/SPXRAW_G.rds`: 44 bytes, stub
- `data/SPXRAW_G.xpt`: 20 KB, corrupt/error-page-sized stub
- `data/SPXRAW_G.csv`: 3 bytes, empty/stub

Synthetic work is complete enough for now. Module 1 segmentation, Module 2 fitting, and Module 3 synthetic TSNPE/NUTS agreement are not substitutes for the real-curve verdict. Gate 3 remains the feasibility test.

```text
# GATE 2+3 - REAL-CURVE VERDICT. Run the instant data/SPXRAW_{E,F,G} exist.
# This is the project's feasibility verdict: does OMBE fit REAL NHANES curves?
# Reuse predictx/{ombe,segment,fit}.py (validated on synthetic). Report ACTUAL
# numbers. Never tune to pass. A FAIL here is the most important result the
# project can produce - it means the physics model needs augmentation.

## PRECONDITION
Assert data/SPXRAW_E.rds/.xpt, _F, _G exist and are tens-of-MB real files
(not the 20KB HTML stubs). If absent or stub-sized, STOP: "raw curves not
present." Do not fabricate.

## GATE 2 - Parse + FEV1 recovery vs ground truth
1. Parse SPXRAW: comma-split -> int array (len==SPXPTS). Integers are volume
   increments (mL/0.01s). flow_Lps=incr*0.1*SPAFACT; volume_mL=cumsum*SPAFACT.
   Do NOT differentiate.
2. Apply the VALIDATED segmenter (predictx/segment.py) + back-extrapolated t0.
3. Per participant: derived FEV1/FVC on acceptable Pre curves. Compare
   per-participant MAX vs merged_summary SPXNFEV1/SPXNFVC (first-test; report
   both max and first-acceptable comparisons - expect small positive bias on max).
4. Save figures/gate2_real_single.png (1 curve flow+volume) + gate2_grid.png (20).
CRITERION: on effort_quality=='A' Pre curves, median |%err| <=5% for FEV1 AND FVC.
FAIL -> DIAGNOSE: constant multiplicative=unit/BTPS; constant additive=t0;
scattered=segmentation. STOP, state which. This is the parser reality-check the
synthetic Module 1 could not provide.

## GATE 3 - OMBE fit fraction on REAL curves. THE VERDICT.
Only if Gate 2 passes.
1. LM-fit OMBE (predictx/fit.py) to each acceptable Pre curve. Report
   normalized-RSS and R^2 distributions.
2. Fraction with adequate fit (same threshold justified vs Steltner ~2.76 used
   on synthetic - state it, do NOT re-tune it to make real data pass).
3. CRITERION: OMBE fits adequately on >=85% of REAL curves.
   FAIL (<85%) -> STOP. Characterize failing curves: obstructed? truncated?
   high-noise? cough-contaminated (cross-ref SPAQEFF=='D')? This decides whether
   the physics model needs augmentation BEFORE any inference is built on it.
4. Report fit fraction stratified by SPAQEFF grade (A/B/C/D) - expect A>D.
   And by obstruction (ratio<0.70 vs >=0.70): does OMBE fit obstructed lungs as
   well as healthy? This is the clinically critical subgroup and the one most
   likely to break a two-compartment model.
OUTPUT: results/gate3_ombe_real.md, figures/gate3_fit_examples.png (good+bad real)

## DISCIPLINE
- Synthetic thresholds carry over UNCHANGED. Do not loosen to pass real data -
  that would be the exact circular/tuning failure avoided all along.
- The obstructed-subgroup fit fraction is the number that matters most: a
  two-compartment model may fit healthy lungs and fail obstructed ones, which
  would directly undercut the screening thesis. Report it prominently.
```
