# Track A Merge Report

Join strategy: DEMO is the base frame; BMX and SPX are left-joined on `SEQN`.

| cycle | DEMO rows | BMX rows | SPX rows | merged rows | BMX matched | DEMO without BMX | SPX matched | DEMO without SPX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E | 10149 | 9762 | 7743 | 10149 | 9762 | 387 | 7743 | 2406 |
| F | 10537 | 10253 | 8195 | 10537 | 10253 | 284 | 8195 | 2342 |
| G | 9756 | 9338 | 7495 | 9756 | 9338 | 418 | 7495 | 2261 |

## Missingness in Merged Output

| column | missing_pct |
|---|---:|
| `pef_mL_s` | 34.14 |
| `fev1_mL` | 34.14 |
| `fvc_mL` | 34.14 |
| `fev1_fvc_ratio` | 34.14 |
| `fev6_mL` | 34.14 |
| `effort_quality` | 33.43 |
| `bronchodilator_selected` | 33.28 |
| `acceptable_curve_count` | 33.27 |
| `fvc_quality` | 33.27 |
| `fev1_quality` | 33.27 |
| `spirometry_status` | 23.02 |
| `height_cm` | 11.61 |
| `weight_kg` | 4.62 |
| `SDMVPSU` | 0.00 |
| `WTMEC2YR` | 0.00 |
| `SEQN` | 0.00 |
| `cycle` | 0.00 |
| `race_ethnicity` | 0.00 |
| `sex` | 0.00 |
| `age_years` | 0.00 |
| `SDMVSTRA` | 0.00 |
