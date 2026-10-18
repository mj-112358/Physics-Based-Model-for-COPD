# GLI Mapping Notes

- rspiro version: 0.5
- GLI functions called: `pred_GLI(age, height, gender, ethnicity, param=c("FEV1","FVC","FEV1FVC"))`; `LLN_GLI(..., param="FEV1FVC")`.
- rspiro documentation states height is in meters; `height_cm` was divided by 100.
- Validation: rspiro documented `pred_GLI` examples for Caucasian women aged 20, 30, and 70 at height 1.70 m matched within 2%.

## NHANES to GLI Ethnicity Mapping

| NHANES code | Interpretation | GLI code | Fallback |
|---:|---|---:|---|
| 1 | Mexican-American | 5 Other/mixed | yes |
| 2 | Other Hispanic | 5 Other/mixed | yes |
| 3 | Non-Hispanic White | 1 Caucasian | no |
| 4 | Non-Hispanic Black | 2 African-American | no |
| 5 | Other Race including Multi-Racial (RIDRETH1) | 5 Other/mixed | yes |
| 6 | Non-Hispanic Asian (RIDRETH3; no NE/SE split) | 5 Other/mixed | yes |
| 7 | Other Race including Multi-Racial (RIDRETH3) | 5 Other/mixed | yes |

- Input rows using fallback GLI Other/mixed mapping: 10430
- Output rows using fallback GLI Other/mixed mapping: 10033
- Input rows dropped because rspiro returned non-finite GLI values: 860
- Finite GLI rows dropped by predicted FEV1 sanity range 0.5-6 L: 1
