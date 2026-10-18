# Gate 3 Re-Run - Cycle F Corrected NRSS + Residual Shape

- status: FAIL
- reason: Corrected noise-normalized NRSS still fails the 85% adequacy bar.
- scope: F historical gate; E and G analyzed separately in the three-cycle extension. Obstructed-subgroup numbers are cycle-F-specific.
- correction: prior `NRSS<=2.76` used a synthetic fixed sigma and is not comparable to real curves with unknown per-curve noise.
- discipline: this is a noise-normalization correctness fix, not a softer metric chosen after seeing the prior fail.

## Per-Curve Noise

- sigma_hat estimator: MAD of flow minus Savitzky-Golay smooth, divided by 0.6745.
- sigma_hat valid curves: 31525/31851
- sigma_hat median: 0.0510 L/s
- sigma_hat IQR: 0.0423 to 0.0603 L/s
- sigma_hat p05-p95: 0.0223 to 0.0803 L/s

## Properly Normalized NRSS

- formula: `NRSS_norm = RSS / (dof * sigma_hat^2)`, where `dof = n_fit_samples - 5`.
- threshold: curve-specific `chi2.ppf(0.95, dof) / dof`; under a correct fit this normalized chi-square is approximately 1.
- NRSS valid curves: 31299/31851
- fit-error curves: 326
- NRSS_norm median: 19.572
- NRSS_norm IQR: 11.457 to 36.195
- threshold median: 1.085
- adequate by corrected NRSS: 0/31851 (0.0% of all curves; 0.0% of NRSS-valid curves)

## Residual Shape

- shape-valid curves: 31060/31851
- peak-concentrated curves: 29702/31060 (95.6% of shape-valid curves)
- median onset-to-PEF RMS / total RMS: 7.235
- median PEF-to-1s RMS / total RMS: 1.220
- median tail >1s RMS / total RMS: 0.495

## Stratified by SPAQEFF

| SPAQEFF | n | NRSS-valid | adequate corrected NRSS | adequate fraction | peak-valid | peak-concentrated | peak fraction | onset/total | PEF-1s/total | tail/total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 18673 | 18514 | 0 | 0.000 | 18642 | 18586 | 0.997 | 8.515 | 1.329 | 0.464 |
| B | 5446 | 5317 | 0 | 0.000 | 5314 | 5138 | 0.967 | 5.672 | 1.349 | 0.515 |
| C | 7732 | 7468 | 0 | 0.000 | 7104 | 5978 | 0.841 | 3.646 | 0.915 | 0.627 |

## Stratified by Obstruction - F Only Provisional Pending E+G

| obstruction group | n | NRSS-valid | adequate corrected NRSS | adequate fraction | peak-valid | peak-concentrated | peak fraction | onset/total | PEF-1s/total | tail/total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| non_obstructed_derived_ratio_ge_0.70 | 28023 | 27472 | 0 | 0.000 | 27290 | 26104 | 0.957 | 7.099 | 1.174 | 0.487 |
| obstructed_derived_ratio_lt_0.70 | 3828 | 3827 | 0 | 0.000 | 3770 | 3598 | 0.954 | 8.570 | 1.632 | 0.560 |

## Key Inversion Check

- A-grade corrected NRSS adequate fraction: 0.000
- B-grade corrected NRSS adequate fraction: 0.000
- non-obstructed corrected NRSS adequate fraction: 0.000
- obstructed corrected NRSS adequate fraction: 0.000
- A-grade non-obstructed peak-concentrated fraction: 0.997
- inversion resolution: A-grade non-obstructed curves are peak-concentrated, supporting a fast-rise/onset limitation.

## Verdict Logic

Corrected NRSS still fails under per-curve noise normalization, so this is a genuine model-adequacy problem under the specified gate.
