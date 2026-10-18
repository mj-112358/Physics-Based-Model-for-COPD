# NHANES Track A Schema Dump

## Cycle E

### DEMO_E.xpt

- n_rows: 10149
- n_columns: 43
- SEQN present: yes

| column_name | dtype | SAS label | n_non_null | missing_pct | example_values |
|---|---:|---|---:|---:|---|
| `SEQN` | `float64` | Respondent sequence number | 10149 | 0.00 | `[41475.0, 41476.0, 41477.0]` |
| `SDDSRVYR` | `float64` | Data Release Number | 10149 | 0.00 | `[5.0, 5.0, 5.0]` |
| `RIDSTATR` | `float64` | Interview/Examination Status | 10149 | 0.00 | `[2.0, 2.0, 2.0]` |
| `RIDEXMON` | `float64` | Six month time period | 9762 | 3.81 | `[2.0, 1.0, 2.0]` |
| `RIAGENDR` | `float64` | Gender | 10149 | 0.00 | `[2.0, 2.0, 1.0]` |
| `RIDAGEYR` | `float64` | Age at Screening Adjudicated - Recode | 10149 | 0.00 | `[62.0, 6.0, 71.0]` |
| `RIDAGEMN` | `float64` | Age in Months at Screening - Recode | 9710 | 4.33 | `[751.0, 81.0, 859.0]` |
| `RIDAGEEX` | `float64` | Age in Months at Exam - Recode | 9367 | 7.71 | `[752.0, 82.0, 860.0]` |
| `RIDRETH1` | `float64` | Race/Ethnicity - Recode | 10149 | 0.00 | `[5.0, 5.0, 3.0]` |
| `DMQMILIT` | `float64` | Served in the US Armed Forces | 6371 | 37.23 | `[2.0, 1.0, 2.0]` |
| `DMDBORN2` | `float64` | Country of Birth - Recode | 10149 | 0.00 | `[1.0, 1.0, 1.0]` |
| `DMDCITZN` | `float64` | Citizenship Status | 10143 | 0.06 | `[1.0, 1.0, 1.0]` |
| `DMDYRSUS` | `float64` | Length of time in US | 1710 | 83.15 | `[6.0, 7.0, 4.0]` |
| `DMDEDUC3` | `float64` | Education Level - Children/Youth 6-19 | 2500 | 75.37 | `[5.397605346934028e-79, 5.397605346934028e-79, 4.0]` |
| `DMDEDUC2` | `float64` | Education Level - Adults 20+ | 5935 | 41.52 | `[3.0, 3.0, 1.0]` |
| `DMDSCHOL` | `float64` | Now attending school? | 2270 | 77.63 | `[1.0, 1.0, 1.0]` |
| `DMDMARTL` | `float64` | Marital Status | 5935 | 41.52 | `[1.0, 1.0, 1.0]` |
| `DMDHHSIZ` | `float64` | Total number of people in the Household | 10149 | 0.00 | `[2.0, 6.0, 2.0]` |
| `DMDFMSIZ` | `float64` | Total number of people in the Family | 10149 | 0.00 | `[2.0, 6.0, 2.0]` |
| `INDHHIN2` | `float64` | Annual Household Income | 10072 | 0.76 | `[6.0, 15.0, 5.0]` |
| `INDFMIN2` | `float64` | Annual Family Income | 10055 | 0.93 | `[6.0, 15.0, 5.0]` |
| `INDFMPIR` | `float64` | Ratio of family income to poverty | 9255 | 8.81 | `[1.83, 5.0, 1.5]` |
| `RIDEXPRG` | `float64` | Pregnancy Status at Exam - Recode | 1208 | 88.10 | `[2.0, 2.0, 2.0]` |
| `DMDHRGND` | `float64` | HH Ref Person Gender | 10149 | 0.00 | `[1.0, 1.0, 1.0]` |
| `DMDHRAGE` | `float64` | HH Ref Person Age | 10149 | 0.00 | `[64.0, 51.0, 71.0]` |
| `DMDHRBR2` | `float64` | HH Ref Person Country of Birth | 9881 | 2.64 | `[1.0, 1.0, 1.0]` |
| `DMDHREDU` | `float64` | HH Ref Person Education Level | 9883 | 2.62 | `[4.0, 5.0, 3.0]` |
| `DMDHRMAR` | `float64` | HH Ref Person Marital Status | 9808 | 3.36 | `[1.0, 1.0, 1.0]` |
| `DMDHSEDU` | `float64` | HH Ref Person's Spouse Education Level | 5411 | 46.68 | `[3.0, 5.0, 4.0]` |
| `SIALANG` | `float64` | Language of SP Interview | 10148 | 0.01 | `[1.0, 1.0, 1.0]` |
| `SIAPROXY` | `float64` | Proxy used in SP Interview? | 10148 | 0.01 | `[2.0, 1.0, 2.0]` |
| `SIAINTRP` | `float64` | Interpreter used in SP Interview? | 10148 | 0.01 | `[2.0, 2.0, 2.0]` |
| `FIALANG` | `float64` | Language of Family Interview | 10021 | 1.26 | `[1.0, 1.0, 1.0]` |
| `FIAPROXY` | `float64` | Proxy used in Family Interview? | 10021 | 1.26 | `[2.0, 2.0, 2.0]` |
| `FIAINTRP` | `float64` | Interpreter used in Family Interview? | 10021 | 1.26 | `[2.0, 2.0, 2.0]` |
| `MIALANG` | `float64` | Language of MEC Interview | 7146 | 29.59 | `[1.0, 1.0, 2.0]` |
| `MIAPROXY` | `float64` | Proxy used in MEC Interview? | 7146 | 29.59 | `[2.0, 2.0, 2.0]` |
| `MIAINTRP` | `float64` | Interpreter used in MEC Interview? | 7146 | 29.59 | `[2.0, 2.0, 2.0]` |
| `AIALANG` | `float64` | Language of ACASI Interview | 7734 | 23.80 | `[1.0, 1.0, 2.0]` |
| `WTINT2YR` | `float64` | Full Sample 2 Year Interview Weight | 10149 | 0.00 | `[59356.356426, 35057.218405, 9935.266183]` |
| `WTMEC2YR` | `float64` | Full Sample 2 Year MEC Exam Weight | 10149 | 0.00 | `[60045.772497, 35353.21044, 10074.150074]` |
| `SDMVPSU` | `float64` | Masked Variance Pseudo-PSU | 10149 | 0.00 | `[1.0, 1.0, 1.0]` |
| `SDMVSTRA` | `float64` | Masked Variance Pseudo-Stratum | 10149 | 0.00 | `[60.0, 70.0, 67.0]` |

### BMX_E.xpt

- n_rows: 9762
- n_columns: 23
- SEQN present: yes

| column_name | dtype | SAS label | n_non_null | missing_pct | example_values |
|---|---:|---|---:|---:|---|
| `SEQN` | `float64` | Respondent sequence number | 9762 | 0.00 | `[41475.0, 41476.0, 41477.0]` |
| `BMDSTATS` | `float64` | Body Measures Component Status Code | 9762 | 0.00 | `[3.0, 1.0, 1.0]` |
| `BMXWT` | `float64` | Weight (kg) | 9631 | 1.34 | `[138.9, 22.0, 83.9]` |
| `BMIWT` | `float64` | Weight Comment | 339 | 96.53 | `[3.0, 4.0, 3.0]` |
| `BMXRECUM` | `float64` | Recumbent Length (cm) | 1194 | 87.77 | `[77.4, 72.7, 90.0]` |
| `BMIRECUM` | `float64` | Recumbent Length Comment | 18 | 99.82 | `[1.0, 1.0, 1.0]` |
| `BMXHEAD` | `float64` | Head Circumference (cm) | 260 | 97.34 | `[41.8, 39.0, 40.8]` |
| `BMIHEAD` | `float64` | Head Circumference Comment | 0 | 100.00 | `[]` |
| `BMXHT` | `float64` | Standing Height (cm) | 8873 | 9.11 | `[154.7, 120.4, 167.1]` |
| `BMIHT` | `float64` | Standing Height Comment | 309 | 96.83 | `[3.0, 3.0, 3.0]` |
| `BMXBMI` | `float64` | Body Mass Index (kg/m**2) | 8861 | 9.23 | `[58.04, 15.18, 30.05]` |
| `BMXLEG` | `float64` | Upper Leg Length (cm) | 7304 | 25.18 | `[34.2, 32.4, 33.3]` |
| `BMILEG` | `float64` | Upper Leg Length Comment | 324 | 96.68 | `[1.0, 1.0, 1.0]` |
| `BMXARML` | `float64` | Upper Arm Length (cm) | 9276 | 4.98 | `[37.6, 25.1, 38.2]` |
| `BMIARML` | `float64` | Upper Arm Length Comment | 299 | 96.94 | `[1.0, 1.0, 1.0]` |
| `BMXARMC` | `float64` | Arm Circumference (cm) | 9281 | 4.93 | `[45.2, 17.1, 34.1]` |
| `BMIARMC` | `float64` | Arm Circumference Comment | 302 | 96.91 | `[1.0, 1.0, 1.0]` |
| `BMXWAIST` | `float64` | Waist Circumference (cm) | 8538 | 12.54 | `[156.3, 52.7, 109.5]` |
| `BMIWAIST` | `float64` | Waist Circumference Comment | 369 | 96.22 | `[1.0, 1.0, 1.0]` |
| `BMXTRI` | `float64` | Triceps Skinfold (mm) | 8944 | 8.38 | `[7.6, 15.6, 13.0]` |
| `BMITRI` | `float64` | Triceps Skinfold Comment | 641 | 93.43 | `[2.0, 1.0, 1.0]` |
| `BMXSUB` | `float64` | Subscapular Skinfold (mm) | 8213 | 15.87 | `[4.7, 25.8, 9.8]` |
| `BMISUB` | `float64` | Subscapular Skinfold Comment | 1372 | 85.95 | `[1.0, 1.0, 1.0]` |

### SPX_E.xpt

- n_rows: 7743
- n_columns: 51
- SEQN present: yes

| column_name | dtype | SAS label | n_non_null | missing_pct | example_values |
|---|---:|---|---:|---:|---|
| `SEQN` | `float64` | Respondent sequence number | 7743 | 0.00 | `[41475.0, 41476.0, 41477.0]` |
| `ENQ010` | `float64` | Breathing problem require oxygen? | 7160 | 7.53 | `[2.0, 2.0, 2.0]` |
| `ENQ020` | `float64` | Problem taking deep breath? | 7096 | 8.36 | `[2.0, 2.0, 2.0]` |
| `SPQ010` | `float64` | Have a current painful ear infection? | 1807 | 76.66 | `[2.0, 2.0, 2.0]` |
| `SPQ020` | `float64` | Have you/Has SP ever had eye surgery? | 6950 | 10.24 | `[2.0, 2.0, 2.0]` |
| `SPQ030` | `float64` | Eye surgery in the last 3 months? | 589 | 92.39 | `[2.0, 2.0, 2.0]` |
| `SPQ040` | `float64` | Ever had open chest/abdominal surgery? | 6914 | 10.71 | `[1.0, 2.0, 2.0]` |
| `SPQ050` | `float64` | Chest/abdominal surgery last 3 months? | 1112 | 85.64 | `[2.0, 2.0, 2.0]` |
| `SPQ060` | `float64` | Tuberculosis in last year? | 6876 | 11.20 | `[2.0, 2.0, 2.0]` |
| `SPQ070A` | `float64` | Ever told had an aneurysm? | 35 | 99.55 | `[1.0, 1.0, 1.0]` |
| `SPQ070B` | `float64` | Ever told had a collapsed lung? | 68 | 99.12 | `[2.0, 2.0, 2.0]` |
| `SPQ070C` | `float64` | Ever told had a detached retina? | 44 | 99.43 | `[3.0, 3.0, 3.0]` |
| `SPQ070D` | `float64` | Ever told had a stroke? | 133 | 98.28 | `[4.0, 4.0, 4.0]` |
| `SPQ070E` | `float64` | Ever told had a heart attack? | 140 | 98.19 | `[5.0, 5.0, 5.0]` |
| `SPQ080` | `float64` | Stroke in the last 3 months? | 120 | 98.45 | `[2.0, 2.0, 2.0]` |
| `SPQ090` | `float64` | Heart attack in last 3 months? | 123 | 98.41 | `[2.0, 2.0, 2.0]` |
| `SPQ100` | `float64` | Coughed up blood past month? | 6712 | 13.32 | `[2.0, 2.0, 2.0]` |
| `ENQ100` | `float64` | Had respiratory illness? | 6650 | 14.12 | `[1.0, 1.0, 2.0]` |
| `SPXNSTAT` | `float64` | Spirometry First Test Status | 7743 | 0.00 | `[1.0, 1.0, 2.0]` |
| `SPXNCMT` | `float64` | Spirometry First Test Comment | 933 | 87.95 | `[4.0, 56.0, 99.0]` |
| `SPXNFVC` | `float64` | Baseline FVC (mL) | 6319 | 18.39 | `[2473.0, 1742.0, 3545.0]` |
| `SPXNEV` | `float64` | Baseline Extrapolated Volume (mL) | 6318 | 18.40 | `[94.0, 50.0, 100.0]` |
| `SPXNFEV5` | `float64` | Baseline FEV 0.5 (mL) | 6319 | 18.39 | `[1681.0, 1282.0, 2629.0]` |
| `SPXNFEV7` | `float64` | Baseline FEV 0.75 (mL) | 6319 | 18.39 | `[1894.0, 1466.0, 2860.0]` |
| `SPXNFEV1` | `float64` | Baseline FEV 1 (mL) | 6319 | 18.39 | `[2025.0, 1551.0, 2957.0]` |
| `SPXNFEV3` | `float64` | Baseline FEV 3 (mL) | 6319 | 18.39 | `[2299.0, 1698.0, 3267.0]` |
| `SPXNFEV6` | `float64` | Baseline FEV 6 (mL) | 6319 | 18.39 | `[2429.0, 1742.0, 3444.0]` |
| `SPXNPEF` | `float64` | Baseline PEF (mL/s) | 6319 | 18.39 | `[5755.0, 3635.0, 8932.0]` |
| `SPXNF257` | `float64` | Baseline FEF 25-75% (mL/s) | 6319 | 18.39 | `[2119.0, 2122.0, 4081.0]` |
| `SPXNFET` | `float64` | Baseline Forced Expiratory Time (s) | 6318 | 18.40 | `[8.7, 5.4, 10.5]` |
| `SPXNQFVC` | `object` | Baseline FVC Quality Attribute | 7743 | 0.00 | `["A", "A", "F"]` |
| `SPXNQFV1` | `object` | Baseline FEV1 Quality Attribute | 7743 | 0.00 | `["A", "A", "F"]` |
| `SPDNACC` | `float64` | Baseline Number of Acceptable Curves | 6444 | 16.78 | `[3.0, 8.0, 5.397605346934028e-79]` |
| `SPXNQEFF` | `object` | Baseline Effort Quality Attribute | 7743 | 0.00 | `["A", "A", "A"]` |
| `SPDBRONC` | `float64` | Selected for Bronchodilator | 6444 | 16.78 | `[2.0, 2.0, 2.0]` |
| `SPXBSTAT` | `float64` | Spirometry Second Test Status | 1001 | 87.07 | `[1.0, 4.0, 3.0]` |
| `SPXBCMT` | `float64` | Spirometry Second Test Comment | 187 | 97.58 | `[2.0, 99.0, 2.0]` |
| `SPXBFVC` | `float64` | 2nd Test FVC (mL) | 611 | 92.11 | `[2752.0, 4387.0, 4231.0]` |
| `SPXBEV` | `float64` | 2nd Test Extrapolated Volume (mL) | 611 | 92.11 | `[77.0, 68.0, 78.0]` |
| `SPXBFEV5` | `float64` | 2nd Test FEV 0.5 (mL) | 611 | 92.11 | `[2031.0, 2072.0, 2620.0]` |
| `SPXBFEV7` | `float64` | 2nd Test FEV 0.75 (mL) | 611 | 92.11 | `[2348.0, 2455.0, 3048.0]` |
| `SPXBFEV1` | `float64` | 2nd Test FEV 1 (mL) | 611 | 92.11 | `[2506.0, 2704.0, 3305.0]` |
| `SPXBFEV3` | `float64` | 2nd Test FEV 3 (mL) | 611 | 92.11 | `[2694.0, 3455.0, 3901.0]` |
| `SPXBFEV6` | `float64` | 2nd Test FEV 6 (mL) | 611 | 92.11 | `[2752.0, 3848.0, 4108.0]` |
| `SPXBPEF` | `float64` | 2nd Test PEF (mL/s) | 611 | 92.11 | `[6036.0, 7743.0, 9520.0]` |
| `SPXBF257` | `float64` | 2nd Test FEF 25-75% (mL/s) | 611 | 92.11 | `[3359.0, 1079.0, 2820.0]` |
| `SPXBFET` | `float64` | 2nd Test Forced Expiratory Time (s) | 611 | 92.11 | `[8.0, 20.2, 11.1]` |
| `SPXBQFVC` | `object` | 2nd Test FVC Quality Attribute | 7743 | 0.00 | `["", "", ""]` |
| `SPXBQFV1` | `object` | 2nd Test FEV1 Quality Attribute | 7743 | 0.00 | `["", "", ""]` |
| `SPDBACC` | `float64` | 2nd Test Number of Acceptable Curves | 611 | 92.11 | `[3.0, 5.0, 3.0]` |
| `SPXBQEFF` | `object` | 2nd Test Effort Quality Attribute | 7743 | 0.00 | `["", "", ""]` |

## Cycle F

### DEMO_F.xpt

- n_rows: 10537
- n_columns: 43
- SEQN present: yes

| column_name | dtype | SAS label | n_non_null | missing_pct | example_values |
|---|---:|---|---:|---:|---|
| `SEQN` | `float64` | Respondent sequence number | 10537 | 0.00 | `[51624.0, 51625.0, 51626.0]` |
| `SDDSRVYR` | `float64` | Data Release Number | 10537 | 0.00 | `[6.0, 6.0, 6.0]` |
| `RIDSTATR` | `float64` | Interview/Examination Status | 10537 | 0.00 | `[2.0, 2.0, 2.0]` |
| `RIDEXMON` | `float64` | Six month time period | 10253 | 2.70 | `[1.0, 2.0, 1.0]` |
| `RIAGENDR` | `float64` | Gender | 10537 | 0.00 | `[1.0, 1.0, 1.0]` |
| `RIDAGEYR` | `float64` | Age at Screening Adjudicated - Recode | 10537 | 0.00 | `[34.0, 4.0, 16.0]` |
| `RIDAGEMN` | `float64` | Age in Months at Screening - Recode | 10112 | 4.03 | `[409.0, 49.0, 202.0]` |
| `RIDAGEEX` | `float64` | Age in Months at Exam - Recode | 9855 | 6.47 | `[410.0, 50.0, 202.0]` |
| `RIDRETH1` | `float64` | Race/Ethnicity - Recode | 10537 | 0.00 | `[3.0, 5.0, 4.0]` |
| `DMQMILIT` | `float64` | Served in the US Armed Forces | 6723 | 36.20 | `[2.0, 2.0, 2.0]` |
| `DMDBORN2` | `float64` | Country of Birth - Recode | 10537 | 0.00 | `[1.0, 1.0, 1.0]` |
| `DMDCITZN` | `float64` | Citizenship Status | 10530 | 0.07 | `[1.0, 1.0, 1.0]` |
| `DMDYRSUS` | `float64` | Length of time in US | 1982 | 81.19 | `[3.0, 7.0, 9.0]` |
| `DMDEDUC3` | `float64` | Education Level - Children/Youth 6-19 | 2595 | 75.37 | `[8.0, 4.0, 3.0]` |
| `DMDEDUC2` | `float64` | Education Level - Adults 20+ | 6218 | 40.99 | `[3.0, 3.0, 2.0]` |
| `DMDSCHOL` | `float64` | Now attending school? | 2353 | 77.67 | `[1.0, 1.0, 1.0]` |
| `DMDMARTL` | `float64` | Marital Status | 6218 | 40.99 | `[1.0, 2.0, 1.0]` |
| `DMDHHSIZ` | `float64` | Total number of people in the Household | 10537 | 0.00 | `[4.0, 4.0, 4.0]` |
| `DMDFMSIZ` | `float64` | Total number of people in the Family | 10537 | 0.00 | `[4.0, 4.0, 4.0]` |
| `INDHHIN2` | `float64` | Annual Household Income | 10450 | 0.83 | `[6.0, 5.0, 8.0]` |
| `INDFMIN2` | `float64` | Annual Family Income | 10465 | 0.68 | `[6.0, 5.0, 8.0]` |
| `INDFMPIR` | `float64` | Ratio of family income to poverty | 9541 | 9.45 | `[1.36, 1.07, 2.27]` |
| `RIDEXPRG` | `float64` | Pregnancy Status at Exam - Recode | 1405 | 86.67 | `[3.0, 2.0, 2.0]` |
| `DMDHRGND` | `float64` | HH Ref Person Gender | 10537 | 0.00 | `[1.0, 1.0, 1.0]` |
| `DMDHRAGE` | `float64` | HH Ref Person Age | 10537 | 0.00 | `[34.0, 38.0, 47.0]` |
| `DMDHRBR2` | `float64` | HH Ref Person Country of Birth | 10260 | 2.63 | `[1.0, 5.0, 1.0]` |
| `DMDHREDU` | `float64` | HH Ref Person Education Level | 10260 | 2.63 | `[3.0, 5.0, 1.0]` |
| `DMDHRMAR` | `float64` | HH Ref Person Marital Status | 10412 | 1.19 | `[1.0, 1.0, 77.0]` |
| `DMDHSEDU` | `float64` | HH Ref Person's Spouse Education Level | 5473 | 48.06 | `[3.0, 5.0, 5.0]` |
| `SIALANG` | `float64` | Language of SP Interview | 10537 | 0.00 | `[1.0, 1.0, 1.0]` |
| `SIAPROXY` | `float64` | Proxy used in SP Interview? | 10534 | 0.03 | `[2.0, 1.0, 2.0]` |
| `SIAINTRP` | `float64` | Interpreter used in SP Interview? | 10537 | 0.00 | `[2.0, 2.0, 2.0]` |
| `FIALANG` | `float64` | Language of Family Interview | 10469 | 0.65 | `[1.0, 1.0, 1.0]` |
| `FIAPROXY` | `float64` | Proxy used in Family Interview? | 10469 | 0.65 | `[2.0, 2.0, 2.0]` |
| `FIAINTRP` | `float64` | Interpreter used in Family Interview? | 10469 | 0.65 | `[2.0, 2.0, 2.0]` |
| `MIALANG` | `float64` | Language of MEC Interview | 7359 | 30.16 | `[1.0, 1.0, 1.0]` |
| `MIAPROXY` | `float64` | Proxy used in MEC Interview? | 7359 | 30.16 | `[2.0, 2.0, 2.0]` |
| `MIAINTRP` | `float64` | Interpreter used in MEC Interview? | 7358 | 30.17 | `[2.0, 2.0, 2.0]` |
| `AIALANG` | `float64` | Language of ACASI Interview | 5488 | 47.92 | `[1.0, 1.0, 1.0]` |
| `WTINT2YR` | `float64` | Full Sample 2 Year Interview Weight | 10537 | 0.00 | `[80100.543512, 53901.104285, 13953.078343]` |
| `WTMEC2YR` | `float64` | Full Sample 2 Year MEC Exam Weight | 10537 | 0.00 | `[81528.772006, 56995.035425, 14509.27886]` |
| `SDMVPSU` | `float64` | Masked Variance Pseudo-PSU | 10537 | 0.00 | `[1.0, 2.0, 1.0]` |
| `SDMVSTRA` | `float64` | Masked Variance Pseudo-Stratum | 10537 | 0.00 | `[83.0, 79.0, 84.0]` |

### BMX_F.xpt

- n_rows: 10253
- n_columns: 23
- SEQN present: yes

| column_name | dtype | SAS label | n_non_null | missing_pct | example_values |
|---|---:|---|---:|---:|---|
| `SEQN` | `float64` | Respondent sequence number | 10253 | 0.00 | `[51624.0, 51625.0, 51626.0]` |
| `BMDSTATS` | `float64` | Body Measures Component Status Code | 10253 | 0.00 | `[1.0, 1.0, 1.0]` |
| `BMXWT` | `float64` | Weight (kg) | 10162 | 0.89 | `[87.4, 17.0, 72.3]` |
| `BMIWT` | `float64` | Weight Comment | 380 | 96.29 | `[3.0, 3.0, 3.0]` |
| `BMXRECUM` | `float64` | Recumbent Length (cm) | 1206 | 88.24 | `[75.7, 72.6, 79.5]` |
| `BMIRECUM` | `float64` | Recumbent Length Comment | 15 | 99.85 | `[1.0, 1.0, 1.0]` |
| `BMXHEAD` | `float64` | Head Circumference (cm) | 238 | 97.68 | `[42.7, 42.8, 38.8]` |
| `BMIHEAD` | `float64` | Head Circumference Comment | 0 | 100.00 | `[]` |
| `BMXHT` | `float64` | Standing Height (cm) | 9420 | 8.12 | `[164.7, 105.4, 181.3]` |
| `BMIHT` | `float64` | Standing Height Comment | 303 | 97.04 | `[3.0, 3.0, 3.0]` |
| `BMXBMI` | `float64` | Body Mass Index (kg/m**2) | 9412 | 8.20 | `[32.22, 15.3, 22.0]` |
| `BMXLEG` | `float64` | Upper Leg Length (cm) | 7755 | 24.36 | `[41.5, 42.0, 37.0]` |
| `BMILEG` | `float64` | Upper Leg Length Comment | 359 | 96.50 | `[1.0, 1.0, 1.0]` |
| `BMXARML` | `float64` | Upper Arm Length (cm) | 9708 | 5.32 | `[40.0, 19.3, 39.5]` |
| `BMIARML` | `float64` | Upper Arm Length Comment | 406 | 96.04 | `[1.0, 1.0, 1.0]` |
| `BMXARMC` | `float64` | Arm Circumference (cm) | 9699 | 5.40 | `[36.4, 15.9, 26.6]` |
| `BMIARMC` | `float64` | Arm Circumference Comment | 416 | 95.94 | `[1.0, 1.0, 1.0]` |
| `BMXWAIST` | `float64` | Waist Circumference (cm) | 8973 | 12.48 | `[100.4, 49.0, 74.7]` |
| `BMIWAIST` | `float64` | Waist Circumference Comment | 479 | 95.33 | `[1.0, 1.0, 1.0]` |
| `BMXTRI` | `float64` | Triceps Skinfold (mm) | 9358 | 8.73 | `[16.4, 6.6, 10.2]` |
| `BMITRI` | `float64` | Triceps Skinfold Comment | 760 | 92.59 | `[1.0, 1.0, 1.0]` |
| `BMXSUB` | `float64` | Subscapular Skinfold (mm) | 8776 | 14.41 | `[24.9, 5.2, 10.5]` |
| `BMISUB` | `float64` | Subscapular Skinfold Comment | 1341 | 86.92 | `[1.0, 1.0, 1.0]` |

### SPX_F.xpt

- n_rows: 8195
- n_columns: 51
- SEQN present: yes

| column_name | dtype | SAS label | n_non_null | missing_pct | example_values |
|---|---:|---|---:|---:|---|
| `SEQN` | `float64` | Respondent sequence number | 8195 | 0.00 | `[51624.0, 51626.0, 51627.0]` |
| `ENQ010` | `float64` | Breathing problem require oxygen? | 7879 | 3.86 | `[2.0, 2.0, 2.0]` |
| `ENQ020` | `float64` | Problem taking deep breath? | 7831 | 4.44 | `[2.0, 2.0, 2.0]` |
| `SPQ010` | `float64` | Have a current painful ear infection? | 1868 | 77.21 | `[2.0, 2.0, 2.0]` |
| `SPQ020` | `float64` | Have you/Has SP ever had eye surgery? | 7685 | 6.22 | `[2.0, 2.0, 2.0]` |
| `SPQ030` | `float64` | Eye surgery in the last 3 months? | 676 | 91.75 | `[2.0, 2.0, 2.0]` |
| `SPQ040` | `float64` | Ever had open chest/abdominal surgery? | 7642 | 6.75 | `[1.0, 2.0, 2.0]` |
| `SPQ050` | `float64` | Chest/abdominal surgery last 3 months? | 1251 | 84.73 | `[1.0, 2.0, 2.0]` |
| `SPQ060` | `float64` | Tuberculosis in last year? | 7605 | 7.20 | `[2.0, 2.0, 2.0]` |
| `SPQ070A` | `float64` | Ever told had an aneurysm? | 47 | 99.43 | `[1.0, 1.0, 99.0]` |
| `SPQ070B` | `float64` | Ever told had a collapsed lung? | 63 | 99.23 | `[2.0, 2.0, 2.0]` |
| `SPQ070C` | `float64` | Ever told had a detached retina? | 50 | 99.39 | `[3.0, 3.0, 3.0]` |
| `SPQ070D` | `float64` | Ever told had a stroke? | 122 | 98.51 | `[4.0, 4.0, 4.0]` |
| `SPQ070E` | `float64` | Ever told had a heart attack? | 144 | 98.24 | `[5.0, 5.0, 5.0]` |
| `SPQ080` | `float64` | Stroke in the last 3 months? | 101 | 98.77 | `[2.0, 2.0, 1.0]` |
| `SPQ090` | `float64` | Heart attack in last 3 months? | 132 | 98.39 | `[2.0, 2.0, 2.0]` |
| `SPQ100` | `float64` | Coughed up blood past month? | 7424 | 9.41 | `[2.0, 2.0, 2.0]` |
| `ENQ100` | `float64` | Had respiratory illness? | 7416 | 9.51 | `[2.0, 1.0, 2.0]` |
| `SPXNSTAT` | `float64` | Spirometry First Test Status | 8195 | 0.00 | `[4.0, 1.0, 1.0]` |
| `SPXNCMT` | `float64` | Spirometry First Test Comment | 569 | 93.06 | `[56.0, 122.0, 84.0]` |
| `SPXNFVC` | `float64` | Baseline FVC (mL) | 7079 | 13.62 | `[5207.0, 2636.0, 2345.0]` |
| `SPXNEV` | `float64` | Baseline Extrapolated Volume (mL) | 7079 | 13.62 | `[145.0, 47.0, 47.0]` |
| `SPXNFEV5` | `float64` | Baseline FEV 0.5 (mL) | 7079 | 13.62 | `[3718.0, 1567.0, 1592.0]` |
| `SPXNFEV7` | `float64` | Baseline FEV 0.75 (mL) | 7079 | 13.62 | `[4350.0, 1850.0, 1762.0]` |
| `SPXNFEV1` | `float64` | Baseline FEV 1 (mL) | 7079 | 13.62 | `[4731.0, 2037.0, 1846.0]` |
| `SPXNFEV3` | `float64` | Baseline FEV 3 (mL) | 7079 | 13.62 | `[5171.0, 2532.0, 2088.0]` |
| `SPXNFEV6` | `float64` | Baseline FEV 6 (mL) | 7079 | 13.62 | `[5183.0, 2623.0, 2222.0]` |
| `SPXNPEF` | `float64` | Baseline PEF (mL/s) | 7079 | 13.62 | `[11399.0, 5502.0, 7455.0]` |
| `SPXNF257` | `float64` | Baseline FEF 25-75% (mL/s) | 7079 | 13.62 | `[5895.0, 1680.0, 1764.0]` |
| `SPXNFET` | `float64` | Baseline Forced Expiratory Time (s) | 7079 | 13.62 | `[8.8, 7.9, 12.5]` |
| `SPXNQFVC` | `object` | Baseline FVC Quality Attribute | 8195 | 0.00 | `["", "A", "B"]` |
| `SPXNQFV1` | `object` | Baseline FEV1 Quality Attribute | 8195 | 0.00 | `["", "B", "A"]` |
| `SPDNACC` | `float64` | Baseline Number of Acceptable Curves | 7173 | 12.47 | `[7.0, 7.0, 3.0]` |
| `SPXNQEFF` | `object` | Baseline Effort Quality Attribute | 8195 | 0.00 | `["", "A", "A"]` |
| `SPDBRONC` | `float64` | Selected for Bronchodilator | 7173 | 12.47 | `[2.0, 2.0, 2.0]` |
| `SPXBSTAT` | `float64` | Spirometry Second Test Status | 926 | 88.70 | `[1.0, 4.0, 4.0]` |
| `SPXBCMT` | `float64` | Spirometry Second Test Comment | 191 | 97.67 | `[99.0, 56.0, 3.0]` |
| `SPXBFVC` | `float64` | 2nd Test FVC (mL) | 496 | 93.95 | `[4554.0, 3773.0, 3550.0]` |
| `SPXBEV` | `float64` | 2nd Test Extrapolated Volume (mL) | 496 | 93.95 | `[46.0, 36.0, 121.0]` |
| `SPXBFEV5` | `float64` | 2nd Test FEV 0.5 (mL) | 496 | 93.95 | `[2696.0, 2104.0, 1785.0]` |
| `SPXBFEV7` | `float64` | 2nd Test FEV 0.75 (mL) | 496 | 93.95 | `[3030.0, 2415.0, 2109.0]` |
| `SPXBFEV1` | `float64` | 2nd Test FEV 1 (mL) | 496 | 93.95 | `[3249.0, 2620.0, 2326.0]` |
| `SPXBFEV3` | `float64` | 2nd Test FEV 3 (mL) | 496 | 93.95 | `[3853.0, 3218.0, 2947.0]` |
| `SPXBFEV6` | `float64` | 2nd Test FEV 6 (mL) | 496 | 93.95 | `[4174.0, 3484.0, 3202.0]` |
| `SPXBPEF` | `float64` | 2nd Test PEF (mL/s) | 496 | 93.95 | `[9239.0, 7744.0, 6979.0]` |
| `SPXBF257` | `float64` | 2nd Test FEF 25-75% (mL/s) | 496 | 93.95 | `[1959.0, 1547.0, 1133.0]` |
| `SPXBFET` | `float64` | 2nd Test Forced Expiratory Time (s) | 496 | 93.95 | `[15.9, 16.1, 17.0]` |
| `SPXBQFVC` | `object` | 2nd Test FVC Quality Attribute | 8195 | 0.00 | `["", "", ""]` |
| `SPXBQFV1` | `object` | 2nd Test FEV1 Quality Attribute | 8195 | 0.00 | `["", "", ""]` |
| `SPDBACC` | `float64` | 2nd Test Number of Acceptable Curves | 496 | 93.95 | `[8.0, 5.0, 3.0]` |
| `SPXBQEFF` | `object` | 2nd Test Effort Quality Attribute | 8195 | 0.00 | `["", "", ""]` |

## Cycle G

### DEMO_G.xpt

- n_rows: 9756
- n_columns: 48
- SEQN present: yes

| column_name | dtype | SAS label | n_non_null | missing_pct | example_values |
|---|---:|---|---:|---:|---|
| `SEQN` | `float64` | Respondent sequence number | 9756 | 0.00 | `[62161.0, 62162.0, 62163.0]` |
| `SDDSRVYR` | `float64` | Data release cycle | 9756 | 0.00 | `[7.0, 7.0, 7.0]` |
| `RIDSTATR` | `float64` | Interview/Examination status | 9756 | 0.00 | `[2.0, 2.0, 2.0]` |
| `RIAGENDR` | `float64` | Gender | 9756 | 0.00 | `[1.0, 2.0, 1.0]` |
| `RIDAGEYR` | `float64` | Age in years at screening | 9756 | 0.00 | `[22.0, 3.0, 14.0]` |
| `RIDAGEMN` | `float64` | Age in months at screening - 0 to 24 mos | 626 | 93.58 | `[11.0, 17.0, 5.397605346934028e-79]` |
| `RIDRETH1` | `float64` | Race/Hispanic origin | 9756 | 0.00 | `[3.0, 1.0, 5.0]` |
| `RIDRETH3` | `float64` | Race/Hispanic origin w/ NH Asian | 9756 | 0.00 | `[3.0, 1.0, 6.0]` |
| `RIDEXMON` | `float64` | Six month time period | 9338 | 4.28 | `[2.0, 1.0, 2.0]` |
| `RIDEXAGY` | `float64` | Age in years at exam - 2 to 19 years | 3418 | 64.97 | `[3.0, 14.0, 14.0]` |
| `RIDEXAGM` | `float64` | Age in months at exam - 0 to 19 years | 4009 | 58.91 | `[41.0, 177.0, 179.0]` |
| `DMQMILIZ` | `float64` | Served active duty in US Armed Forces | 6007 | 38.43 | `[2.0, 1.0, 2.0]` |
| `DMQADFC` | `float64` | Served in a foreign country | 551 | 94.35 | `[2.0, 1.0, 1.0]` |
| `DMDBORN4` | `float64` | Country of birth | 9756 | 0.00 | `[1.0, 1.0, 1.0]` |
| `DMDCITZN` | `float64` | Citizenship status | 9751 | 0.05 | `[1.0, 1.0, 1.0]` |
| `DMDYRSUS` | `float64` | Length of time in US | 2073 | 78.75 | `[3.0, 4.0, 4.0]` |
| `DMDEDUC3` | `float64` | Education level - Children/Youth 6-19 | 2599 | 73.36 | `[8.0, 7.0, 3.0]` |
| `DMDEDUC2` | `float64` | Education level - Adults 20+ | 5560 | 43.01 | `[3.0, 4.0, 3.0]` |
| `DMDMARTL` | `float64` | Marital status | 5560 | 43.01 | `[5.0, 1.0, 5.0]` |
| `RIDEXPRG` | `float64` | Pregnancy status at exam | 1208 | 87.62 | `[2.0, 2.0, 2.0]` |
| `SIALANG` | `float64` | Language of SP Interview | 9756 | 0.00 | `[1.0, 1.0, 1.0]` |
| `SIAPROXY` | `float64` | Proxy used in SP Interview? | 9750 | 0.06 | `[1.0, 1.0, 1.0]` |
| `SIAINTRP` | `float64` | Interpreter used in SP Interview? | 9756 | 0.00 | `[2.0, 2.0, 2.0]` |
| `FIALANG` | `float64` | Language of Family Interview | 9651 | 1.08 | `[1.0, 1.0, 1.0]` |
| `FIAPROXY` | `float64` | Proxy used in Family Interview? | 9651 | 1.08 | `[2.0, 2.0, 2.0]` |
| `FIAINTRP` | `float64` | Interpreter used in Family Interview? | 9651 | 1.08 | `[2.0, 2.0, 2.0]` |
| `MIALANG` | `float64` | Language of MEC Interview | 6713 | 31.19 | `[1.0, 1.0, 1.0]` |
| `MIAPROXY` | `float64` | Proxy used in MEC Interview? | 6713 | 31.19 | `[2.0, 2.0, 2.0]` |
| `MIAINTRP` | `float64` | Interpreter used in MEC Interview? | 6713 | 31.19 | `[2.0, 2.0, 2.0]` |
| `AIALANGA` | `float64` | Language of ACASI Interview | 5754 | 41.02 | `[1.0, 1.0, 1.0]` |
| `WTINT2YR` | `float64` | Full sample 2 year interview weight | 9756 | 0.00 | `[102641.406474, 15457.736897, 7397.684828]` |
| `WTMEC2YR` | `float64` | Full sample 2 year MEC exam weight | 9756 | 0.00 | `[104236.582554, 16116.35401, 7869.485117]` |
| `SDMVPSU` | `float64` | Masked variance pseudo-PSU | 9756 | 0.00 | `[1.0, 3.0, 3.0]` |
| `SDMVSTRA` | `float64` | Masked variance pseudo-stratum | 9756 | 0.00 | `[91.0, 92.0, 90.0]` |
| `INDHHIN2` | `float64` | Annual household income | 9675 | 0.83 | `[14.0, 4.0, 15.0]` |
| `INDFMIN2` | `float64` | Annual family income | 9705 | 0.52 | `[14.0, 4.0, 15.0]` |
| `INDFMPIR` | `float64` | Ratio of family income to poverty | 8916 | 8.61 | `[3.15, 0.6, 4.07]` |
| `DMDHHSIZ` | `float64` | Total number of people in the Household | 9756 | 0.00 | `[5.0, 6.0, 5.0]` |
| `DMDFMSIZ` | `float64` | Total number of people in the Family | 9756 | 0.00 | `[5.0, 6.0, 5.0]` |
| `DMDHHSZA` | `float64` | # of children 5 years or younger in HH | 9756 | 0.00 | `[5.397605346934028e-79, 2.0, 5.397605346934028e-79]` |
| `DMDHHSZB` | `float64` | # of children 6-17 years old in HH | 9756 | 0.00 | `[1.0, 2.0, 2.0]` |
| `DMDHHSZE` | `float64` | # of adults 60 years or older in HH | 9756 | 0.00 | `[5.397605346934028e-79, 5.397605346934028e-79, 1.0]` |
| `DMDHRGND` | `float64` | HH ref person's gender | 9756 | 0.00 | `[2.0, 2.0, 1.0]` |
| `DMDHRAGE` | `float64` | HH ref person's age in years | 9756 | 0.00 | `[50.0, 24.0, 42.0]` |
| `DMDHRBR4` | `float64` | HH ref person's country of birth | 9391 | 3.74 | `[1.0, 1.0, 1.0]` |
| `DMDHREDU` | `float64` | HH ref person's education level | 9394 | 3.71 | `[5.0, 3.0, 5.0]` |
| `DMDHRMAR` | `float64` | HH ref person's marital status | 9620 | 1.39 | `[1.0, 6.0, 1.0]` |
| `DMDHSEDU` | `float64` | HH ref person's spouse's education level | 4875 | 50.03 | `[5.0, 4.0, 4.0]` |

### BMX_G.xpt

- n_rows: 9338
- n_columns: 26
- SEQN present: yes

| column_name | dtype | SAS label | n_non_null | missing_pct | example_values |
|---|---:|---|---:|---:|---|
| `SEQN` | `float64` | Respondent sequence number | 9338 | 0.00 | `[62161.0, 62162.0, 62163.0]` |
| `BMDSTATS` | `float64` | Body Measures Component Status Code | 9338 | 0.00 | `[1.0, 1.0, 1.0]` |
| `BMXWT` | `float64` | Weight (kg) | 9243 | 1.02 | `[69.2, 12.7, 49.4]` |
| `BMIWT` | `float64` | Weight Comment | 379 | 95.94 | `[3.0, 3.0, 4.0]` |
| `BMXRECUM` | `float64` | Recumbent Length (cm) | 1079 | 88.45 | `[95.7, 79.5, 92.5]` |
| `BMIRECUM` | `float64` | Recumbent Length Comment | 31 | 99.67 | `[1.0, 1.0, 1.0]` |
| `BMXHEAD` | `float64` | Head Circumference (cm) | 236 | 97.47 | `[39.4, 44.4, 42.9]` |
| `BMIHEAD` | `float64` | Head Circumference Comment | 0 | 100.00 | `[]` |
| `BMXHT` | `float64` | Standing Height (cm) | 8615 | 7.74 | `[172.3, 94.7, 168.9]` |
| `BMIHT` | `float64` | Standing Height Comment | 268 | 97.13 | `[3.0, 1.0, 1.0]` |
| `BMXBMI` | `float64` | Body Mass Index (kg/m**2) | 8602 | 7.88 | `[23.3, 14.2, 17.3]` |
| `BMDBMIC` | `float64` | BMI Category - Children/Adolescents | 3355 | 64.07 | `[2.0, 2.0, 3.0]` |
| `BMXLEG` | `float64` | Upper Leg Length (cm) | 6955 | 25.52 | `[40.2, 40.3, 40.5]` |
| `BMILEG` | `float64` | Upper Leg Length Comment | 354 | 96.21 | `[1.0, 1.0, 1.0]` |
| `BMXARML` | `float64` | Upper Arm Length (cm) | 8826 | 5.48 | `[35.0, 18.5, 36.3]` |
| `BMIARML` | `float64` | Upper Arm Length Comment | 369 | 96.05 | `[1.0, 1.0, 1.0]` |
| `BMXARMC` | `float64` | Arm Circumference (cm) | 8826 | 5.48 | `[32.5, 16.6, 22.0]` |
| `BMIARMC` | `float64` | Arm Circumference Comment | 373 | 96.01 | `[1.0, 1.0, 1.0]` |
| `BMXWAIST` | `float64` | Waist Circumference (cm) | 8204 | 12.14 | `[81.0, 45.4, 64.6]` |
| `BMIWAIST` | `float64` | Waist Circumference Comment | 456 | 95.12 | `[1.0, 1.0, 1.0]` |
| `BMXSAD1` | `float64` | Sagittal Abdominal Diameter 1st (cm) | 6795 | 27.23 | `[17.7, 15.6, 18.3]` |
| `BMXSAD2` | `float64` | Sagittal Abdominal Diameter 2nd (cm) | 6795 | 27.23 | `[17.9, 15.5, 18.5]` |
| `BMXSAD3` | `float64` | Sagittal Abdominal Diameter 3rd (cm) | 398 | 95.74 | `[24.0, 18.4, 24.7]` |
| `BMXSAD4` | `float64` | Sagittal Abdominal Diameter 4th (cm) | 398 | 95.74 | `[23.9, 18.6, 24.7]` |
| `BMDAVSAD` | `float64` | Average Sagittal Abdominal Diameter (cm) | 6795 | 27.23 | `[17.8, 15.6, 18.4]` |
| `BMDSADCM` | `float64` | Sagittal Abdominal Diameter Comment | 485 | 94.81 | `[4.0, 1.0, 1.0]` |

### SPX_G.xpt

- n_rows: 7495
- n_columns: 51
- SEQN present: yes

| column_name | dtype | SAS label | n_non_null | missing_pct | example_values |
|---|---:|---|---:|---:|---|
| `SEQN` | `float64` | Respondent sequence number | 7495 | 0.00 | `[62161.0, 62163.0, 62164.0]` |
| `ENQ010` | `float64` | Breathing problem require oxygen? | 7193 | 4.03 | `[2.0, 2.0, 2.0]` |
| `ENQ020` | `float64` | Problem taking deep breath? | 7152 | 4.58 | `[2.0, 2.0, 2.0]` |
| `SPQ010` | `float64` | Have a current painful ear infection? | 1903 | 74.61 | `[2.0, 2.0, 2.0]` |
| `SPQ020` | `float64` | Have you/Has SP ever had eye surgery? | 7083 | 5.50 | `[2.0, 2.0, 1.0]` |
| `SPQ030` | `float64` | Eye surgery in the last 3 months? | 578 | 92.29 | `[2.0, 2.0, 2.0]` |
| `SPQ040` | `float64` | Ever had open chest/abdominal surgery? | 7057 | 5.84 | `[2.0, 2.0, 2.0]` |
| `SPQ050` | `float64` | Chest/abdominal surgery last 3 months? | 1044 | 86.07 | `[2.0, 2.0, 2.0]` |
| `SPQ060` | `float64` | Tuberculosis in last year? | 7025 | 6.27 | `[2.0, 2.0, 2.0]` |
| `SPQ070A` | `float64` | Ever told had an aneurysm? | 37 | 99.51 | `[1.0, 1.0, 1.0]` |
| `SPQ070B` | `float64` | Ever told had a collapsed lung? | 39 | 99.48 | `[2.0, 2.0, 2.0]` |
| `SPQ070C` | `float64` | Ever told had a detached retina? | 38 | 99.49 | `[3.0, 3.0, 3.0]` |
| `SPQ070D` | `float64` | Ever told had a stroke? | 103 | 98.63 | `[4.0, 4.0, 4.0]` |
| `SPQ070E` | `float64` | Ever told had a heart attack? | 110 | 98.53 | `[5.0, 5.0, 5.0]` |
| `SPQ080` | `float64` | Stroke in the last 3 months? | 95 | 98.73 | `[2.0, 2.0, 2.0]` |
| `SPQ090` | `float64` | Heart attack in last 3 months? | 94 | 98.75 | `[2.0, 2.0, 2.0]` |
| `SPQ100` | `float64` | Coughed up blood past month? | 6897 | 7.98 | `[2.0, 2.0, 2.0]` |
| `ENQ100` | `float64` | Had respiratory illness? | 6832 | 8.85 | `[2.0, 2.0, 2.0]` |
| `SPXNSTAT` | `float64` | Spirometry First Test Status | 7495 | 0.00 | `[1.0, 1.0, 1.0]` |
| `SPXNCMT` | `float64` | Spirometry First Test Comment | 532 | 92.90 | `[7.0, 72.0, 84.0]` |
| `SPXNFVC` | `float64` | Baseline FVC (mL) | 6652 | 11.25 | `[4536.0, 3667.0, 4566.0]` |
| `SPXNEV` | `float64` | Baseline Extrapolated Volume (mL) | 6650 | 11.27 | `[101.0, 69.0, 62.0]` |
| `SPXNFEV5` | `float64` | Baseline FEV 0.5 (mL) | 6652 | 11.25 | `[3333.0, 2788.0, 2691.0]` |
| `SPXNFEV7` | `float64` | Baseline FEV 0.75 (mL) | 6652 | 11.25 | `[4005.0, 3322.0, 3125.0]` |
| `SPXNFEV1` | `float64` | Baseline FEV 1 (mL) | 6652 | 11.25 | `[4265.0, 3563.0, 3397.0]` |
| `SPXNFEV3` | `float64` | Baseline FEV 3 (mL) | 6652 | 11.25 | `[4514.0, 3646.0, 4167.0]` |
| `SPXNFEV6` | `float64` | Baseline FEV 6 (mL) | 6652 | 11.25 | `[4493.0, 3663.0, 4374.0]` |
| `SPXNPEF` | `float64` | Baseline PEF (mL/s) | 6652 | 11.25 | `[10421.0, 8549.0, 9378.0]` |
| `SPXNF257` | `float64` | Baseline FEF 25-75% (mL/s) | 6652 | 11.25 | `[5693.0, 4892.0, 2558.0]` |
| `SPXNFET` | `float64` | Baseline Forced Expiratory Time (s) | 6652 | 11.25 | `[7.6, 7.7, 11.6]` |
| `SPXNQFVC` | `object` | Baseline FVC Quality Attribute | 7495 | 0.00 | `["C", "B", "A"]` |
| `SPXNQFV1` | `object` | Baseline FEV1 Quality Attribute | 7495 | 0.00 | `["B", "A", "A"]` |
| `SPDNACC` | `float64` | Baseline Number of Acceptable Curves | 6696 | 10.66 | `[4.0, 4.0, 3.0]` |
| `SPXNQEFF` | `object` | Baseline Effort Quality Attribute | 7495 | 0.00 | `["A", "A", "A"]` |
| `SPDBRONC` | `float64` | Selected for Bronchodilator | 6695 | 10.67 | `[2.0, 2.0, 2.0]` |
| `SPXBSTAT` | `float64` | Spirometry Second Test Status | 842 | 88.77 | `[1.0, 3.0, 1.0]` |
| `SPXBCMT` | `float64` | Spirometry Second Test Comment | 204 | 97.28 | `[99.0, 2.0, 99.0]` |
| `SPXBFVC` | `float64` | 2nd Test FVC (mL) | 457 | 93.90 | `[2466.0, 4487.0, 1875.0]` |
| `SPXBEV` | `float64` | 2nd Test Extrapolated Volume (mL) | 457 | 93.90 | `[15.0, 60.0, 37.0]` |
| `SPXBFEV5` | `float64` | 2nd Test FEV 0.5 (mL) | 457 | 93.90 | `[1427.0, 2494.0, 1165.0]` |
| `SPXBFEV7` | `float64` | 2nd Test FEV 0.75 (mL) | 457 | 93.90 | `[1634.0, 3028.0, 1394.0]` |
| `SPXBFEV1` | `float64` | 2nd Test FEV 1 (mL) | 457 | 93.90 | `[1751.0, 3377.0, 1536.0]` |
| `SPXBFEV3` | `float64` | 2nd Test FEV 3 (mL) | 457 | 93.90 | `[2101.0, 4217.0, 1807.0]` |
| `SPXBFEV6` | `float64` | 2nd Test FEV 6 (mL) | 457 | 93.90 | `[2275.0, 4423.0, 1871.0]` |
| `SPXBPEF` | `float64` | 2nd Test PEF (mL/s) | 457 | 93.90 | `[4821.0, 8514.0, 3372.0]` |
| `SPXBF257` | `float64` | 2nd Test FEF 25-75% (mL/s) | 457 | 93.90 | `[1101.0, 2655.0, 1584.0]` |
| `SPXBFET` | `float64` | 2nd Test Forced Expiratory Time (s) | 457 | 93.90 | `[13.4, 9.3, 5.9]` |
| `SPXBQFVC` | `object` | 2nd Test FVC Quality Attribute | 7495 | 0.00 | `["", "", ""]` |
| `SPXBQFV1` | `object` | 2nd Test FEV1 Quality Attribute | 7495 | 0.00 | `["", "", ""]` |
| `SPDBACC` | `float64` | 2nd Test Number of Acceptable Curves | 457 | 93.90 | `[8.0, 4.0, 5.0]` |
| `SPXBQEFF` | `object` | 2nd Test Effort Quality Attribute | 7495 | 0.00 | `["", "", ""]` |

## Cross-Cycle Drift Checks

### DEMO

- common columns: 37
- columns absent in cycle E: `AIALANGA, DMDBORN4, DMDHHSZA, DMDHHSZB, DMDHHSZE, DMDHRBR4, DMQADFC, DMQMILIZ, RIDEXAGM, RIDEXAGY, RIDRETH3`
- columns absent in cycle F: `AIALANGA, DMDBORN4, DMDHHSZA, DMDHHSZB, DMDHHSZE, DMDHRBR4, DMQADFC, DMQMILIZ, RIDEXAGM, RIDEXAGY, RIDRETH3`
- columns absent in cycle G: `AIALANG, DMDBORN2, DMDHRBR2, DMDSCHOL, DMQMILIT, RIDAGEEX`
- columns unique to cycle G: `AIALANGA, DMDBORN4, DMDHHSZA, DMDHHSZB, DMDHHSZE, DMDHRBR4, DMQADFC, DMQMILIZ, RIDEXAGM, RIDEXAGY, RIDRETH3`
- label drift:
  - `DMDCITZN`: E='Citizenship Status'; F='Citizenship Status'; G='Citizenship status'
  - `DMDEDUC2`: E='Education Level - Adults 20+'; F='Education Level - Adults 20+'; G='Education level - Adults 20+'
  - `DMDEDUC3`: E='Education Level - Children/Youth 6-19'; F='Education Level - Children/Youth 6-19'; G='Education level - Children/Youth 6-19'
  - `DMDHRAGE`: E='HH Ref Person Age'; F='HH Ref Person Age'; G="HH ref person's age in years"
  - `DMDHREDU`: E='HH Ref Person Education Level'; F='HH Ref Person Education Level'; G="HH ref person's education level"
  - `DMDHRGND`: E='HH Ref Person Gender'; F='HH Ref Person Gender'; G="HH ref person's gender"
  - `DMDHRMAR`: E='HH Ref Person Marital Status'; F='HH Ref Person Marital Status'; G="HH ref person's marital status"
  - `DMDHSEDU`: E="HH Ref Person's Spouse Education Level"; F="HH Ref Person's Spouse Education Level"; G="HH ref person's spouse's education level"
  - `DMDMARTL`: E='Marital Status'; F='Marital Status'; G='Marital status'
  - `INDFMIN2`: E='Annual Family Income'; F='Annual Family Income'; G='Annual family income'
  - `INDHHIN2`: E='Annual Household Income'; F='Annual Household Income'; G='Annual household income'
  - `RIDAGEMN`: E='Age in Months at Screening - Recode'; F='Age in Months at Screening - Recode'; G='Age in months at screening - 0 to 24 mos'
  - `RIDAGEYR`: E='Age at Screening Adjudicated - Recode'; F='Age at Screening Adjudicated - Recode'; G='Age in years at screening'
  - `RIDEXPRG`: E='Pregnancy Status at Exam - Recode'; F='Pregnancy Status at Exam - Recode'; G='Pregnancy status at exam'
  - `RIDRETH1`: E='Race/Ethnicity - Recode'; F='Race/Ethnicity - Recode'; G='Race/Hispanic origin'
  - `RIDSTATR`: E='Interview/Examination Status'; F='Interview/Examination Status'; G='Interview/Examination status'
  - `SDDSRVYR`: E='Data Release Number'; F='Data Release Number'; G='Data release cycle'
  - `SDMVPSU`: E='Masked Variance Pseudo-PSU'; F='Masked Variance Pseudo-PSU'; G='Masked variance pseudo-PSU'
  - `SDMVSTRA`: E='Masked Variance Pseudo-Stratum'; F='Masked Variance Pseudo-Stratum'; G='Masked variance pseudo-stratum'
  - `WTINT2YR`: E='Full Sample 2 Year Interview Weight'; F='Full Sample 2 Year Interview Weight'; G='Full sample 2 year interview weight'
  - `WTMEC2YR`: E='Full Sample 2 Year MEC Exam Weight'; F='Full Sample 2 Year MEC Exam Weight'; G='Full sample 2 year MEC exam weight'
- low-cardinality coding drift:
  - `DMDCITZN`: E=['1.0', '2.0', '7.0', '9.0']; F=['1.0', '2.0', '7.0', '9.0']; G=['1.0', '2.0', '7.0']
  - `DMDEDUC3`: E=['1.0', '10.0', '11.0', '12.0', '13.0', '14.0', '15.0', '2.0', '3.0', '4.0', '5.0', '5.397605346934028e-79', '55.0', '6.0', '66.0', '7.0', '8.0', '9.0']; F=['1.0', '10.0', '11.0', '12.0', '13.0', '14.0', '15.0', '2.0', '3.0', '4.0', '5.0', '5.397605346934028e-79', '55.0', '6.0', '66.0', '7.0', '8.0', '9.0', '99.0']; G=['1.0', '10.0', '11.0', '12.0', '13.0', '14.0', '15.0', '2.0', '3.0', '4.0', '5.0', '5.397605346934028e-79', '55.0', '6.0', '66.0', '7.0', '8.0', '9.0']
  - `DMDHSEDU`: E=['1.0', '2.0', '3.0', '4.0', '5.0', '7.0', '9.0']; F=['1.0', '2.0', '3.0', '4.0', '5.0', '7.0', '9.0']; G=['1.0', '2.0', '3.0', '4.0', '5.0', '9.0']
  - `DMDMARTL`: E=['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '77.0']; F=['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '77.0', '99.0']; G=['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '77.0', '99.0']
  - `SDDSRVYR`: E=['5.0']; F=['6.0']; G=['7.0']
  - `SDMVPSU`: E=['1.0', '2.0']; F=['1.0', '2.0', '3.0']; G=['1.0', '2.0', '3.0']
  - `SDMVSTRA`: E=['59.0', '60.0', '61.0', '62.0', '63.0', '64.0', '65.0', '66.0', '67.0', '68.0', '69.0', '70.0', '71.0', '72.0', '73.0', '74.0']; F=['75.0', '76.0', '77.0', '78.0', '79.0', '80.0', '81.0', '82.0', '83.0', '84.0', '85.0', '86.0', '87.0', '88.0', '89.0']; G=['100.0', '101.0', '102.0', '103.0', '90.0', '91.0', '92.0', '93.0', '94.0', '95.0', '96.0', '97.0', '98.0', '99.0']

### BMX

- common columns: 19
- columns absent in cycle E: `BMDAVSAD, BMDBMIC, BMDSADCM, BMXSAD1, BMXSAD2, BMXSAD3, BMXSAD4`
- columns absent in cycle F: `BMDAVSAD, BMDBMIC, BMDSADCM, BMXSAD1, BMXSAD2, BMXSAD3, BMXSAD4`
- columns absent in cycle G: `BMISUB, BMITRI, BMXSUB, BMXTRI`
- columns unique to cycle G: `BMDAVSAD, BMDBMIC, BMDSADCM, BMXSAD1, BMXSAD2, BMXSAD3, BMXSAD4`
- label drift: none detected for common columns
- low-cardinality coding drift:
  - `BMIRECUM`: E=['1.0', '3.0']; F=['1.0', '3.0']; G=['1.0']

### SPX

- common columns: 51
- label drift: none detected for common columns
- low-cardinality coding drift:
  - `ENQ010`: E=['1.0', '2.0', '9.0']; F=['1.0', '2.0', '7.0', '9.0']; G=['1.0', '2.0']
  - `SPDBACC`: E=['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0']; F=['2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0']; G=['2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0']
  - `SPDNACC`: E=['1.0', '10.0', '2.0', '3.0', '4.0', '5.0', '5.397605346934028e-79', '6.0', '7.0', '8.0', '9.0']; F=['1.0', '2.0', '3.0', '4.0', '5.0', '5.397605346934028e-79', '6.0', '7.0', '8.0', '9.0']; G=['1.0', '10.0', '12.0', '2.0', '3.0', '4.0', '5.0', '5.397605346934028e-79', '6.0', '7.0', '8.0', '9.0']
  - `SPQ030`: E=['1.0', '2.0', '9.0']; F=['1.0', '2.0', '9.0']; G=['1.0', '2.0']
  - `SPQ050`: E=['1.0', '2.0', '9.0']; F=['1.0', '2.0', '9.0']; G=['1.0', '2.0']
  - `SPQ060`: E=['1.0', '2.0', '9.0']; F=['1.0', '2.0', '7.0', '9.0']; G=['1.0', '2.0', '9.0']
  - `SPQ070B`: E=['2.0']; F=['2.0', '99.0']; G=['2.0']
  - `SPXBCMT`: E=['14.0', '2.0', '3.0', '4.0', '5.0', '84.0', '99.0']; F=['2.0', '27.0', '3.0', '5.0', '56.0', '7.0', '84.0', '99.0']; G=['2.0', '27.0', '3.0', '4.0', '5.0', '51.0', '7.0', '99.0']
  - `SPXBQEFF`: E=['', 'A', 'C']; F=['', 'A', 'B', 'C']; G=['', 'A', 'B', 'C']
  - `SPXBQFV1`: E=['', 'A', 'B', 'C', 'D']; F=['', 'A', 'B', 'C', 'D']; G=['', 'A', 'B', 'C']
  - `SPXNCMT`: E=['14.0', '2.0', '27.0', '3.0', '4.0', '5.0', '56.0', '6.0', '7.0', '84.0', '99.0']; F=['122.0', '14.0', '2.0', '3.0', '4.0', '5.0', '51.0', '56.0', '6.0', '62.0', '7.0', '84.0', '99.0']; G=['122.0', '14.0', '2.0', '3.0', '4.0', '5.0', '51.0', '56.0', '6.0', '7.0', '72.0', '84.0', '99.0']

## Resolved Column Map

```json
{
  "participant_id": {
    "E": "SEQN",
    "F": "SEQN",
    "G": "SEQN"
  },
  "age_years": {
    "E": "RIDAGEYR",
    "F": "RIDAGEYR",
    "G": "RIDAGEYR"
  },
  "sex": {
    "E": "RIAGENDR",
    "F": "RIAGENDR",
    "G": "RIAGENDR"
  },
  "race_ethnicity": {
    "E": "RIDRETH1",
    "F": "RIDRETH1",
    "G": "RIDRETH3"
  },
  "survey_weight_mec_2yr": {
    "E": "WTMEC2YR",
    "F": "WTMEC2YR",
    "G": "WTMEC2YR"
  },
  "survey_psu": {
    "E": "SDMVPSU",
    "F": "SDMVPSU",
    "G": "SDMVPSU"
  },
  "survey_strata": {
    "E": "SDMVSTRA",
    "F": "SDMVSTRA",
    "G": "SDMVSTRA"
  },
  "height_cm": {
    "E": "BMXHT",
    "F": "BMXHT",
    "G": "BMXHT"
  },
  "weight_kg": {
    "E": "BMXWT",
    "F": "BMXWT",
    "G": "BMXWT"
  },
  "fev1_mL": {
    "E": "SPXNFEV1",
    "F": "SPXNFEV1",
    "G": "SPXNFEV1"
  },
  "fvc_mL": {
    "E": "SPXNFVC",
    "F": "SPXNFVC",
    "G": "SPXNFVC"
  },
  "fev1_fvc_ratio": {
    "E": null,
    "F": null,
    "G": null
  },
  "pef_mL_s": {
    "E": "SPXNPEF",
    "F": "SPXNPEF",
    "G": "SPXNPEF"
  },
  "fev6_mL": {
    "E": "SPXNFEV6",
    "F": "SPXNFEV6",
    "G": "SPXNFEV6"
  },
  "spirometry_status": {
    "E": "SPXNSTAT",
    "F": "SPXNSTAT",
    "G": "SPXNSTAT"
  },
  "fvc_quality": {
    "E": "SPXNQFVC",
    "F": "SPXNQFVC",
    "G": "SPXNQFVC"
  },
  "fev1_quality": {
    "E": "SPXNQFV1",
    "F": "SPXNQFV1",
    "G": "SPXNQFV1"
  },
  "acceptable_curve_count": {
    "E": "SPDNACC",
    "F": "SPDNACC",
    "G": "SPDNACC"
  },
  "effort_quality": {
    "E": "SPXNQEFF",
    "F": "SPXNQEFF",
    "G": "SPXNQEFF"
  },
  "bronchodilator_selected": {
    "E": "SPDBRONC",
    "F": "SPDBRONC",
    "G": "SPDBRONC"
  },
  "raw_curve": {
    "E": null,
    "F": null,
    "G": null
  },
  "n_points": {
    "E": null,
    "F": null,
    "G": null
  },
  "btps_factor": {
    "E": null,
    "F": null,
    "G": null
  },
  "test_type": {
    "E": null,
    "F": null,
    "G": null
  },
  "maneuver_num": {
    "E": null,
    "F": null,
    "G": null
  },
  "acceptable": {
    "E": null,
    "F": null,
    "G": null
  },
  "position": {
    "E": null,
    "F": null,
    "G": null
  },
  "plateau": {
    "E": null,
    "F": null,
    "G": null
  },
  "raw_effort_grade": {
    "E": null,
    "F": null,
    "G": null
  }
}
```

## Resolution Warnings

- Cycle E `raw_curve` unresolved in SPXRAW (optional=True).
- Cycle F `raw_curve` unresolved in SPXRAW (optional=True).
- Cycle G `raw_curve` unresolved in SPXRAW (optional=True).
- Cycle E `n_points` unresolved in SPXRAW (optional=True).
- Cycle F `n_points` unresolved in SPXRAW (optional=True).
- Cycle G `n_points` unresolved in SPXRAW (optional=True).
- Cycle E `btps_factor` unresolved in SPXRAW (optional=True).
- Cycle F `btps_factor` unresolved in SPXRAW (optional=True).
- Cycle G `btps_factor` unresolved in SPXRAW (optional=True).
- Cycle E `test_type` unresolved in SPXRAW (optional=True).
- Cycle F `test_type` unresolved in SPXRAW (optional=True).
- Cycle G `test_type` unresolved in SPXRAW (optional=True).
- Cycle E `maneuver_num` unresolved in SPXRAW (optional=True).
- Cycle F `maneuver_num` unresolved in SPXRAW (optional=True).
- Cycle G `maneuver_num` unresolved in SPXRAW (optional=True).
- Cycle E `acceptable` unresolved in SPXRAW (optional=True).
- Cycle F `acceptable` unresolved in SPXRAW (optional=True).
- Cycle G `acceptable` unresolved in SPXRAW (optional=True).
- Cycle E `position` unresolved in SPXRAW (optional=True).
- Cycle F `position` unresolved in SPXRAW (optional=True).
- Cycle G `position` unresolved in SPXRAW (optional=True).
- Cycle E `plateau` unresolved in SPXRAW (optional=True).
- Cycle F `plateau` unresolved in SPXRAW (optional=True).
- Cycle G `plateau` unresolved in SPXRAW (optional=True).
- Cycle E `raw_effort_grade` unresolved in SPXRAW (optional=True).
- Cycle F `raw_effort_grade` unresolved in SPXRAW (optional=True).
- Cycle G `raw_effort_grade` unresolved in SPXRAW (optional=True).
