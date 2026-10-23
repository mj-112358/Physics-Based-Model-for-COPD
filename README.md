# Physics-Based Modeling of Forced-Expiration Spirometry

**An interpretable computational framework for spirometric signal reconstruction, constrained expiratory-flow modeling, and incomplete-maneuver analysis.**

[Methods](docs/METHODS.md) · [Research findings](docs/RESULTS.md) · [Data dictionary](docs/DATA_DICTIONARY.md) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Study scope](docs/LIMITATIONS.md)

## 1. General information

| Field | Description |
| --- | --- |
| Research title | Physics-Based Modeling of Forced-Expiration Spirometry |
| Repository | [mj-112358/Physics-Based-Model-for-COPD](https://github.com/mj-112358/Physics-Based-Model-for-COPD) |
| Research domain | Respiratory physiology; computational modeling; biomedical signal processing |
| Primary measurement | Forced-expiratory volume–time and flow–time curves |
| Data source | CDC/NCHS National Health and Nutrition Examination Survey (NHANES) |
| Study cycles | 2007–2008 (E), 2009–2010 (F), 2011–2012 (G) |
| Geographic coverage | United States; NHANES study population |
| Repository language | English; Python and supplemental R |
| Keywords | spirometry, forced expiration, FEV1, FVC, airflow obstruction, mechanistic models, constrained extrapolation, NHANES |
| Maintainer and correspondence | Repository owner via [GitHub issues](https://github.com/mj-112358/Physics-Based-Model-for-COPD/issues) |
| Version identification | Git commit SHA; the Python distribution currently declares version `0.1.0` |

**Study focus.** Premature termination of a forced-expiration maneuver can yield an incomplete forced vital capacity (FVC) measurement and alter the ratio of forced expiratory volume in one second (FEV1) to FVC. This project studies the recovery of respiratory-volume measurements from sampled expiratory traces using a compact, physiologically constrained flow model. The workflow brings together signal decoding, respiratory-feature estimation, mechanistic fitting, benchmark comparisons, and protocol-controlled truncation experiments.

## 2. Research overview

### Scientific objective

To characterize whether a low-dimensional, physically constrained representation of expiratory flow can recover FEV1, FVC and FEV1/FVC from observed spirometric recordings, with specific emphasis on the information available before an expiration is complete.

The research separates three analytical questions:

1. **Signal reconstruction:** Can released NHANES sampled curves be decoded into consistent respiratory measurements?
2. **Mechanistic representation:** How well does a two-timescale expiratory-flow model represent measured curves?
3. **Incomplete-maneuver recovery:** Under a fixed extrapolation horizon, how do model-derived estimates compare with directly observed truncated volumes and conventional extrapolation methods?

### Computational workflow

```text
NHANES SPX + SPXRAW
        |
        v
Input integrity, schema mapping and BTPS correction
        |
        v
100 Hz volume and flow reconstruction
        |
        v
Expiration-onset registration and spirometric features
        |
        +-------------------+
        |                   |
        v                   v
Direct measurements    Mechanistic flow fitting
        |                   |
        +---------+---------+
                  |
                  v
Participant-level evaluation and benchmark comparisons
                  |
                  v
Quality, sensitivity, calibration and truncation analyses
                  |
                  v
Versioned numerical results, figures and audits
```

## 3. Mathematical formulation

Let `t` denote elapsed time from forced-expiration onset. The model combines two exponential flow components with an onset term:

```math
q(t)=\left[\alpha\frac{e^{-t/\tau_f}}{\tau_f}+(1-\alpha)\frac{e^{-t/\tau_s}}{\tau_s}\right](1-e^{-t/\tau_o}).
```

For prespecified prediction horizon `H`, the volume-normalized flow is:

```math
Q(t)=\mathrm{FVC}\,\frac{q(t)}{\int_0^H q(u)\,du},\qquad V(t)=\int_0^t Q(u)\,du.
```

Here `τ_f` and `τ_s` are characteristic expiratory time constants, `α` is a mixing coefficient, and `τ_o` controls flow onset. Model fitting uses bounded nonlinear least squares in the deterministic pipeline; supplementary experiments investigate inferential variants. Detailed equations, sampling conventions, quality criteria, cohort construction and evaluation design are recorded in [Methods](docs/METHODS.md).

## 4. Data and file overview

This repository is organized into reusable scientific software, study documentation, published result snapshots, and visualization assets.

| Location | Contents | Relationship to analysis |
| --- | --- | --- |
| `predictx/io.py`, `schema.py`, `dataset_resolver.py` | Public dataset handling, validation and field resolution | Input and schema layer |
| `predictx/raw.py`, `merge.py`, `gli.py` | Curve reconstruction, linked tabular data and reference processing | Measurement layer |
| `predictx/ombe.py`, `ombe_gamma.py`, `fit.py` | Respiratory flow formulations and fitting | Model layer |
| `predictx/obstruction.py` | Spirometric threshold analysis | Outcome layer |
| `predictx/gate*.py` | Registered and exploratory analytical experiments and audits | Experiment layer; scripts preserve historical protocol variants |
| `predictx/run.py` | Main dataset-preparation workflow | Orchestration |
| `tests/` | Automated test suite | Software verification |
| `results/` | JSON/CSV reports and analytical summaries | Historical analysis artifacts; see evidence-status documentation |
| `figures/` | PNG plots | Visualization of study outputs |
| `docs/` | Methods, results, provenance, data dictionary and scope | Research documentation |
| `.github/workflows/quality.yml` | Automated quality checks | Continuous integration |

For file-level provenance and variable definitions, consult [Data dictionary](docs/DATA_DICTIONARY.md). Raw NHANES datasets are sourced separately; they are not included with this source distribution. Filenames and original study periods are maintained as released by CDC. Refer to commit history for software update dates; generated reports should record their own execution timestamp and code revision.

## 5. Principal research findings

### Raw-curve reconstruction

Across three NHANES cycles, participant-level reconstructed FEV1 and FVC show median absolute agreement errors of approximately **3–4%** relative to released summary measurements. Relevant cycle-level participant counts are **6,319 (E), 7,079 (F), and 6,652 (G)**. This is an internal measurement-reconstruction comparison against same-session fields.

### Fixed-horizon recovery from truncated expirations

In a corrected Cycle F evaluation on **497 selected participants**, curves were truncated at prespecified times while every extrapolation method used a **common 12-second horizon** and only information available up to the cut point.

| Cut point | Direct endpoint median absolute FVC error | Mechanistic estimate | Direct false-negative rate | Mechanistic estimate |
| --- | ---: | ---: | ---: | ---: |
| 2 s | 17.38% | **10.01%** | 0.990 | **0.593** |
| 3 s | 12.29% | **9.49%** | 0.865 | **0.596** |

The comparisons characterize relative recovery performance in a selected internal sample, rather than clinical diagnostic accuracy. Model sensitivity at these cut points was 0.407 and 0.404, respectively.

### Model adequacy and research interpretation

Independent complete-curve goodness-of-fit analysis found **0 of 31,851 Cycle F curves** meeting the prespecified noise-normalized adequacy criterion. The study therefore differentiates observed recovery gains in its truncation experiment from adequacy of the current parametric shape on general complete curves. All quantitative findings, including confirmatory versus exploratory status, are consolidated in [Research findings](docs/RESULTS.md); methodological qualifications are documented in [Study scope](docs/LIMITATIONS.md).

## 6. Software environment

**Requirements:** Python 3.10+, access to public NHANES spirometry data for full replication, and optional R support for GLI reference generation.

```bash
git clone https://github.com/mj-112358/Physics-Based-Model-for-COPD.git
cd Physics-Based-Model-for-COPD
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For supplementary simulation-based inference:

```bash
python -m pip install -e ".[inference,dev]"
```

Base dependencies and compatible version ranges are specified in [pyproject.toml](pyproject.toml); the optional GLI workflow uses R and `rspiro` 0.5. Pin an environment-specific lockfile for exact computational replication.

## 7. Data access and directory layout

Obtain source data directly from the [CDC NHANES examination database](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&Cycle=) and the [SPXRAW cycle-F documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/SPXRAW_F.htm). The expected analysis input layout is:

```text
NHANES_SpiroSBI/
├── 2007-2008/
│   ├── SPX_E.xpt
│   └── spxraw_e.sas7bdat
├── 2009-2010/
│   ├── SPX_F.xpt
│   └── spxraw_f.sas7bdat
└── 2011-2012/
    ├── SPX_G.xpt
    └── spxraw_g.sas7bdat
```

Additional modules use demographic, examination, smoking and medical-condition releases. Their precise requirements are described in [Reproducibility](docs/REPRODUCIBILITY.md) and within the relevant experiment modules.

## 8. Reproducing the analysis

Start with the software tests and parser / mechanistic-analysis workflows:

```bash
python -m pytest
python -m compileall -q predictx tests
python -m predictx.gate23_f --root . --cycle F
python -m predictx.gate_truncation_recovery_f --root .
```

The real-curve adequacy command encodes its scientific adequacy verdict as a process exit status; an unsuccessful adequacy gate is an analysis outcome and should be interpreted alongside its report, rather than as proof of a broken software installation. Complete input setup, supplementary cycle replication, validation checks and artifact handling are in [Reproducibility](docs/REPRODUCIBILITY.md).

## 9. Methodological and data-specific information

- **Sampling:** raw spirometry volume increments at 0.01 s intervals (100 Hz).
- **Transformations:** BTPS adjustment, cumulative volume reconstruction, smoothed-flow identification of expiration onset, and derivation of FEV1/FVC.
- **Primary units:** FEV1 and FVC in liters or milliliters according to the report; time in seconds; dimensionless FEV1/FVC.
- **Identifiers:** NHANES participant identifier `SEQN`; curve-level maneuver identifiers vary by input file.
- **Quality metadata:** acceptability and effort-grade fields accompany the raw-curve release.
- **Missing values:** analysis modules handle absent measurements according to their input filters; no single imputation method is asserted across all experiments.
- **Quality assurance:** participant-level splitting, prespecified truncation horizons, regression checks and method-comparison audits.
- **Observational framework:** comparisons are based on released NHANES measurements and selected spirometry-derived labels; results are not independently adjudicated COPD diagnoses.
- **Detailed variable definitions:** [Data dictionary](docs/DATA_DICTIONARY.md).

## 10. Sharing, citation and research use

The source is publicly accessible through this repository. NHANES data remain subject to the CDC's source-specific documentation and usage terms. **No repository-wide software license is declared** in the current codebase; do not infer permission for redistribution or relicensing beyond applicable law. No publication DOI or published article citation is supplied in repository metadata.

Suggested *software citation* (replace revision and access date with those actually used):

> Physics-Based Modeling of Forced-Expiration Spirometry. Source code, GitHub repository `mj-112358/Physics-Based-Model-for-COPD`, commit `<SHA>`, accessed `<YYYY-MM-DD>`. https://github.com/mj-112358/Physics-Based-Model-for-COPD

When publishing results, also cite the relevant CDC/NCHS NHANES source datasets, study periods and documentation. Research correspondence and proposed corrections may be submitted through repository issues.

## 11. Research interpretation

This repository is a computational research study of spirometric signals and airflow-obstruction surrogates. It is not a clinical device or a substitute for professionally administered spirometry or post-bronchodilator clinical assessment. The documentation distinguishes prespecified evaluations, subsequent protocol corrections, historical exploratory artifacts and the independent evidence required for clinical translation.
