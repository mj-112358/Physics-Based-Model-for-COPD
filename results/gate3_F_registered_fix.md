# Gate 3 Registered Onset Fix

- status: SAMPLE_NOT_FIXED
- reason: registered sample did not clear the fixed criteria
- method: jointly optimized a bounded sub-sample onset time shift with the five OMBE parameters.
- shift convention: positive shift delays the model relative to data; prediction at data time t is unshifted OMBE at t-shift.
- max shift bound: +/-0.08 s.
- criterion change discipline: flow-onset weakness remains documented; clinical Gate 3 uses volume-derived outputs because FEV1, FVC, and FEV1/FVC are volume integrals.

## Step 2 - 200-Curve A-Grade Non-Obstructed Sample

- baseline dipole fraction: 0.845
- registered dipole fraction: 0.535
- registered flow NRSS_norm median: 6.951
- registered flow adequate fraction: 0.000
- registered shift median: -1.78 samples
- registered shift IQR: -2.38 to -1.35 samples

| clinical metric | baseline | registered | delta points |
|---|---:|---:|---:|
| all FEV1/FVC/FVC within 5% | 0.840 | 0.680 | -16.0 |
| FEV1/FVC within 5% | 0.855 | 0.745 | -11.0 |
| FEV1/FVC within 0.02 absolute | 0.675 | 0.435 | -24.0 |

- fixed criteria: dipoles<30%=False; all-three-within-5%>90%=False; FEV1/FVC within 0.02 improved by >=5 points=False (delta=-24.0 points)

## Step 3 - Full Cycle F Registered Gate

- NOT RUN: sample did not clear the fixed criteria, so the script did not spend a full 31,851-curve rerun.

## Step 4 - Residual Clinical Failures

- sample curves failing FEV1/FVC within 0.02: 113
- sample failed median registered shift: -1.77 samples
