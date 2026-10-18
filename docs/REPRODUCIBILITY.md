# Reproducibility

## Environment

Create an isolated Python environment and install the repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Install the optional inference stack only for the NUTS and simulation-based inference modules:

```bash
python -m pip install -e '.[inference,dev]'
```

The GLI reference-generation step also requires R and `rspiro` 0.5.

## Public data layout

Download the official SPX and SPXRAW files for the three NHANES cycles. Extract each raw SAS file into this structure:

```text
NHANES_SpiroSBI/
  2007-2008/
    SPX_E.xpt
    spxraw_e.sas7bdat
  2009-2010/
    SPX_F.xpt
    spxraw_f.sas7bdat
  2011-2012/
    SPX_G.xpt
    spxraw_g.sas7bdat
```

Additional analyses require the DEMO, BMX, smoking, and medical-condition files referenced by the relevant module. Raw data directories and archives are excluded by `.gitignore`.

Official documentation:

- [NHANES 2009-2010 raw spirometry curves](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/SPXRAW_F.htm)
- [NHANES examination data index](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&Cycle=)

## Minimum verification

```bash
python -m pytest
python -m compileall -q predictx tests
```

## Parser and real-curve gate

```bash
python -m predictx.gate23_f --root . --cycle F
```

The command is expected to return a nonzero exit status when the real-curve adequacy gate fails. A nonzero status in this case records a scientific result, not necessarily a software error.

## Corrected truncation analysis

```bash
python -m predictx.gate_truncation_recovery_f --root .
```

Before accepting regenerated results, verify that the JSON output reports a 12-second prediction horizon and that no truncated estimator received information after its cut time.

## Three-cycle replication

```bash
python -m predictx.gate_three_cycle_efg --root .
```

The run is computationally intensive. Retain the emitted seed, cohort counts, and complete software environment with any manuscript result.

## Result provenance

Committed result files are snapshots, not automatically trusted ground truth. The evidence status in `docs/RESULTS.md` takes precedence over legacy verdict labels inside historical artifacts. Regenerated files should include the commit identifier, package versions, input file hashes, and execution timestamp before use in a formal submission.
