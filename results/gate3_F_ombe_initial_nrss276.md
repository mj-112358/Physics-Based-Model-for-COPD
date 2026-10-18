# Gate 3 - Cycle F OMBE Fit on Real Curves

- status: FAIL
- reason: Gate 3 real-curve OMBE fit fraction failed.
- scope: F historical gate; E and G analyzed separately in the three-cycle extension. Obstructed-subgroup numbers are cycle-F-specific.
- adequate threshold NRSS: 2.76
- threshold note: unchanged from synthetic Module 2/Steltner-like threshold; not tuned on real data.
- fit curves: 31851
- adequate fits: 8041/31851 (25.2%)
- NRSS median: 4.578
- NRSS IQR: 2.729 to 8.484
- R^2 median: 0.9614
- R^2 IQR: 0.9410 to 0.9719

## Fit Fraction by SPAQEFF

| SPAQEFF | n | adequate | adequate fraction |
|---|---:|---:|---:|
| A | 18673 | 5014 | 0.269 |
| B | 5446 | 2361 | 0.434 |
| C | 7732 | 666 | 0.086 |

## Obstructed Fit Fraction - F Only Provisional Pending E+G

All obstructed-subgroup numbers here use derived per-curve FEV1/FVC and are cycle F historical; E and G are analyzed separately in the three-cycle extension.

| obstruction group | n | adequate | adequate fraction |
|---|---:|---:|---:|
| non_obstructed_derived_ratio_ge_0.70 | 28023 | 6010 | 0.214 |
| obstructed_derived_ratio_lt_0.70 | 3828 | 2031 | 0.531 |

## Obstructed Severity - F Only Provisional Pending E+G

| derived FEV1/FVC severity bin | n | adequate | adequate fraction |
|---|---:|---:|---:|
| 0.50-0.60 | 755 | 436 | 0.577 |
| 0.60-0.70 | 2653 | 1351 | 0.509 |
| <0.50 | 420 | 244 | 0.581 |

## Failure Characterization

- failed curves: 23810
- failed obstructed curves: 1797 (cycle F historical; E and G analyzed separately in the three-cycle extension)
- failed truncated <6s curves: 8373
- failed SPAQEFF=='D' curves: 0
- fit-error curves: 326
