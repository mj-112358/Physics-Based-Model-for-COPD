# External Validity EFG

EXTERNAL VALIDITY: CLINICALLY-GENERALIZES

## Variables Used

- smoking: SMQ020 + SMQ040
- self_reported_obstructive_disease: MCQ160G emphysema OR MCQ160K chronic bronchitis
- age: DEMO.RIDAGEYR
- ethnicity_common_key: DEMO.RIDRETH1
- body_measurements_loaded: BMXHT, BMXWT, BMXBMI

## Covariate Notes

- Cycle G has RIDRETH3, but RIDRETH1 was used as the common ethnicity key.

## Stratified DIRECT vs OMBE Recovery

| stratification | stratum | T | n | DIRECT FN | OMBE FN | DIRECT-OMBE | CI low | CI high | excludes 0 | UNDERPOWERED n<30 |
|---|---|---|---|---|---|---|---|---|---|---|
| smoking_status | current | 2 | 320 | 0.981 (314/320; CI 0.966-0.994) | 0.562 (180/320; CI 0.506-0.616) | 0.419 | 0.363 | 0.472 | True | False |
| smoking_status | current | 3 | 320 | 0.819 (262/320; CI 0.775-0.863) | 0.522 (167/320; CI 0.466-0.578) | 0.297 | 0.247 | 0.347 | True | False |
| smoking_status | former | 2 | 276 | 0.993 (274/276; CI 0.982-1.000) | 0.605 (167/276; CI 0.551-0.663) | 0.388 | 0.330 | 0.446 | True | False |
| smoking_status | former | 3 | 276 | 0.884 (244/276; CI 0.844-0.920) | 0.630 (174/276; CI 0.576-0.685) | 0.254 | 0.203 | 0.304 | True | False |
| smoking_status | never | 2 | 242 | 0.988 (239/242; CI 0.971-1.000) | 0.649 (157/242; CI 0.587-0.707) | 0.339 | 0.277 | 0.397 | True | False |
| smoking_status | never | 3 | 242 | 0.921 (223/242; CI 0.884-0.955) | 0.686 (166/242; CI 0.628-0.744) | 0.236 | 0.182 | 0.289 | True | False |
| smoking_status | unknown | 2 | 54 | 0.926 (50/54; CI 0.852-0.981) | 0.185 (10/54; CI 0.093-0.296) | 0.741 | 0.630 | 0.852 | True | False |
| smoking_status | unknown | 3 | 54 | 0.556 (30/54; CI 0.426-0.685) | 0.185 (10/54; CI 0.093-0.296) | 0.370 | 0.241 | 0.500 | True | False |
| self_reported_obstructive_disease | no | 2 | 706 | 0.992 (700/706; CI 0.984-0.997) | 0.627 (443/706; CI 0.592-0.662) | 0.364 | 0.329 | 0.398 | True | False |
| self_reported_obstructive_disease | no | 3 | 706 | 0.894 (631/706; CI 0.870-0.916) | 0.640 (452/706; CI 0.606-0.676) | 0.254 | 0.222 | 0.285 | True | False |
| self_reported_obstructive_disease | unknown | 2 | 60 | 0.933 (56/60; CI 0.867-0.983) | 0.233 (14/60; CI 0.133-0.350) | 0.700 | 0.583 | 0.800 | True | False |
| self_reported_obstructive_disease | unknown | 3 | 60 | 0.600 (36/60; CI 0.483-0.717) | 0.233 (14/60; CI 0.133-0.350) | 0.367 | 0.250 | 0.483 | True | False |
| self_reported_obstructive_disease | yes | 2 | 126 | 0.960 (121/126; CI 0.921-0.992) | 0.452 (57/126; CI 0.365-0.540) | 0.508 | 0.421 | 0.595 | True | False |
| self_reported_obstructive_disease | yes | 3 | 126 | 0.730 (92/126; CI 0.643-0.802) | 0.405 (51/126; CI 0.317-0.492) | 0.325 | 0.246 | 0.405 | True | False |
| age_band | 40-59 | 2 | 258 | 0.996 (257/258; CI 0.988-1.000) | 0.647 (167/258; CI 0.585-0.705) | 0.349 | 0.287 | 0.411 | True | False |
| age_band | 40-59 | 3 | 258 | 0.849 (219/258; CI 0.806-0.895) | 0.636 (164/258; CI 0.578-0.694) | 0.213 | 0.163 | 0.264 | True | False |
| age_band | 60-79 | 2 | 472 | 0.987 (466/472; CI 0.977-0.996) | 0.572 (270/472; CI 0.528-0.614) | 0.415 | 0.371 | 0.460 | True | False |
| age_band | 60-79 | 3 | 472 | 0.898 (424/472; CI 0.871-0.924) | 0.583 (275/472; CI 0.536-0.627) | 0.316 | 0.275 | 0.356 | True | False |
| age_band | <40 | 2 | 162 | 0.951 (154/162; CI 0.914-0.981) | 0.475 (77/162; CI 0.401-0.556) | 0.475 | 0.401 | 0.556 | True | False |
| age_band | <40 | 3 | 162 | 0.716 (116/162; CI 0.648-0.784) | 0.481 (78/162; CI 0.401-0.562) | 0.235 | 0.173 | 0.302 | True | False |
