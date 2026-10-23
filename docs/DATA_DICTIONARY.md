# Data dictionary and provenance

## Source datasets

The research uses the United States National Health and Nutrition Examination Survey (NHANES) publicly released spirometry data, cycles E (2007–2008), F (2009–2010) and G (2011–2012). Official CDC dataset pages define sampling, survey design, protocol details and the release-specific schema. Raw data are not redistributed in this repository.

| Source | Example files | Format | Study relationship |
| --- | --- | --- | --- |
| Spirometry summary | `SPX_E.xpt`, `SPX_F.xpt`, `SPX_G.xpt` | SAS transport / XPT | Participant-level published measurements and exam quality |
| Raw spirometry | `spxraw_e.sas7bdat`, `spxraw_f.sas7bdat`, `spxraw_g.sas7bdat` | SAS dataset | Curve-level sampled volume increments and maneuver metadata |
| Demographics | `DEMO_[E/F/G].xpt` | XPT | Participant demographic and NHANES design variables |
| Body measures | `BMX_[E/F/G].xpt` | XPT | Height and related measurements |
| Supplemental files | Cycle-specific smoking and medical-condition releases | XPT | Covariates used by selected analysis modules |

## Key raw and reconstructed fields

Field names follow the mappings encoded in `predictx/schema.py`. Source names may vary across NHANES cycles; consult source documentation and generated `results/column_map.json`.

| Logical variable | NHANES field | Definition | Units / conventions |
| --- | --- | --- | --- |
| Participant ID | `SEQN` | NHANES participant identifier | Identifier |
| Age | `RIDAGEYR` | Age at screening | Years |
| Sex code | `RIAGENDR` | Released demographic code | Coded; use NHANES codebook |
| Height | `BMXHT` | Standing height | cm |
| Examination weight | `WTMEC2YR` | Two-year MEC survey sampling weight | Survey weight |
| Survey PSU | `SDMVPSU` | Masked variance pseudo-PSU | Survey design |
| Survey stratum | `SDMVSTRA` | Masked variance pseudo-stratum | Survey design |
| FEV1 | `SPXNFEV1` | Released forced-expiratory volume in the first second | Refer to source codebook / internal conversion |
| FVC | `SPXNFVC` | Released forced vital capacity | Refer to source codebook / internal conversion |
| Peak flow | `SPXNPEF` | Released peak expiratory flow | Refer to source codebook |
| FEV6 | `SPXNFEV6` | Released six-second expired volume | Refer to source codebook |
| Raw increments | `SPXRAW` | Encoded sample-wise change in expired volume | Source-native increments; see parser |
| Curve sample count | `SPXPTS` | Reported number of sample points | Count |
| BTPS factor | `SPAFACT` | Correction from ambient to body temperature and pressure, saturated | Multiplicative |
| Maneuver number | `SPAMANU` | Individual effort identifier | Integer / coded |
| Acceptability | `SPAACC` | Individual curve acceptability | Source-coded flag |
| Onset-corrected curve | Derived | Cumulative BTPS-adjusted volume referenced to forced-expiration onset | Volume versus seconds |
| FEV1/FVC | Derived | Ratio between first-second volume and forced vital capacity | Dimensionless |

## Sampling and processing

At 100 Hz (Δt = 0.01 s), the volume series is reconstructed by accumulating BTPS-scaled increments, and discrete flow is obtained by dividing each corrected increment by Δt. The first increment represents the interval ending at 0.01 s. A registered onset is estimated from a smoothed flow trace with back-extrapolation near peak flow. Derived metrics, quality gates and normalization conventions are specified in [Methods](METHODS.md).

## Files, versions, and quality

- `results/*.json`: machine-readable snapshots containing experiment parameters and numerical summaries.
- `results/*.csv`: individual analysis rows and diagnostic outputs; variable sets vary by experiment.
- `results/*.md`: human-readable experiment interpretations and audit records.
- `figures/*.png`: publication-style and exploratory visualizations.
- `results/column_map.json`: detected mapping between logical fields and the loaded source-file schemas.

**Missing data:** original NHANES files contain release-defined missingness and categorical codes. Each experiment applies its own eligibility and complete-case rules; do not interpret absent output rows as negative clinical findings.

**Quality flags:** use `SPAACC` and related effort fields according to their NHANES codebooks. Historical gate verdicts are software-generated protocol findings; cross-reference [Results](RESULTS.md) before treating any as the current primary result.

**Dates and updates:** cycle years describe *data collection*, not code publication. The Git history is authoritative for code changes. Reproduced outputs should document the precise commit, collection cycles, generation date in ISO 8601, dependency versions, random seed and input checksums.

## Data access, rights, and references

Use [CDC NHANES Examination Data](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&Cycle=) and the corresponding [raw curve documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/SPXRAW_F.htm) for authoritative field descriptions, quality codes, usage information and official citations. Source files are not bundled. No independent repository data license or DOI is stated.

See [Reproducibility](REPRODUCIBILITY.md) for file-placement instructions and command-level workflows.
