# Cycle-Parametric Migration Regression, Cycle F

- VERDICT: MIGRATION-CLEAN
- seed: 20260710
- elapsed seconds: 331.32

## Part 0a - Cached F Regression Targets

| metric | expected | cached before migration | delta | pass |
|---|---:|---:|---:|---|
| gate2.med_abs_FEV1 | 2.9083271328 | 2.9083271328 | 0 | True |
| gate2.med_abs_FVC | 3.5273255814 | 3.5273255814 | 0 | True |
| gate2.signed_FEV1 | -2.88588914146 | -2.88588914146 | 0 | True |
| gate2.signed_FVC | -3.48932643396 | -3.48932643396 | 0 | True |
| gate2.parse_errors | 45 | 45 | 0 | True |
| gate2.participants_with_truth | 7079 | 7079 | 0 | True |
| clinical.fixed.sensitivity | 0.982456140351 | 0.982456140351 | 0 | True |
| clinical.fixed.specificity | 0.956666666667 | 0.956666666667 | 0 | True |
| clinical.fixed.kappa | 0.911847372455 | 0.911847372455 | 0 | True |
| clinical.fixed.n | 414 | 414 | 0 | True |

## Part 0b - Migrated Checklist

- predictx/gate23_f.py: raw resolver formerly hardcoded data/ and DATASET?/SPXRAW_F paths
- predictx/gate23_f.py: summary truth formerly hardcoded data/SPX_F.xpt
- predictx/gate23_f.py: Gate 2 and Gate 3 output filenames now include the cycle token
- predictx/gate_clinical_classification_f.py: load_truth now filters df[df['cycle'] == cycle]
- predictx/gate_clinical_classification_f.py: column_map source fields now use .get(cycle)
- predictx/gate_clinical_classification_f.py: raw resolver and output filenames now include the cycle token
- predictx/gate_estimand_hybrid_f.py: read_spx_summary now resolves SPX_{cycle}.xpt
- predictx/gate_four_part_fix_f.py: Part A SPX summary probe now resolves SPX_{cycle}.xpt

## Current Source Audit After Migration

| path | line | patterns | text |
|---|---|---|---|
| predictx/dataset_resolver.py | 119 | DATASET? | root / "DATASET?" / f"{file_type}_{cycle}.xpt", |
| predictx/dataset_resolver.py | 188 | DATASET? | root / "DATASET?" / f"spxraw_{cycle.lower()}.sas7bdat", |
| predictx/dataset_resolver.py | 189 | DATASET? | root / "DATASET?" / f"SPXRAW_{cycle}.sas7bdat", |
| predictx/dataset_resolver.py | 206 | DATASET? | root / "DATASET?" / f"SPXRAW_{cycle}.zip", |
| predictx/gate3_diagnosis.py | 211 | gate3_F | fits_path = root / "results" / "gate3_F_curve_fits.csv" |
| predictx/gate3_diagnosis.py | 390 | gate3_F | fig.savefig(figures_dir / "gate3_F_diagnosis_examples.png", dpi=170, bbox_inches="tight") |
| predictx/gate3_diagnosis.py | 456 | gate3_F | (results_dir / "gate3_F_diagnosis.md").write_text("\n".join(lines), encoding="utf-8") |
| predictx/gate3_diagnosis.py | 457 | gate3_F | (results_dir / "gate3_F_diagnosis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8") |
| predictx/gate3_diagnosis.py | 458 | gate3_F | rows.to_csv(results_dir / "gate3_F_diagnosis_sample.csv", index=False) |
| predictx/gate3_f_rerun.py | 256 | gate3_F | fig.savefig(figures_dir / "gate3_F_residual_shape.png", dpi=170, bbox_inches="tight") |
| predictx/gate3_f_rerun.py | 266 | cycle F historical, E/G extension | "- scope: F historical gate; E and G analyzed separately in the three-cycle extension. Obstructed-subgroup numbers are cycle-F-specific.", |
| predictx/gate3_f_rerun.py | 344 | gate3_F | (results_dir / "gate3_F_ombe_rerun.md").write_text("\n".join(lines), encoding="utf-8") |
| predictx/gate3_f_rerun.py | 345 | gate3_F | (results_dir / "gate3_F_ombe_rerun.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8") |
| predictx/gate3_f_rerun.py | 346 | gate3_F | rows.to_csv(results_dir / "gate3_F_residual_shape.csv", index=False) |
| predictx/gate3_f_rerun.py | 352 | gate3_F | fits_path = results_dir / "gate3_F_curve_fits.csv" |
| predictx/gate3_gamma_onset.py | 207 | gate3_F | fits = pd.read_csv(root / "results" / "gate3_F_curve_fits.csv") |
| predictx/gate3_gamma_onset.py | 320 | gate3_F | fig.savefig(figures_dir / "gate3_F_gamma_onset_examples.png", dpi=170, bbox_inches="tight") |
| predictx/gate3_gamma_onset.py | 396 | gate3_F | (results_dir / "gate3_F_gamma_onset.md").write_text("\n".join(lines), encoding="utf-8") |
| predictx/gate3_gamma_onset.py | 397 | gate3_F | (results_dir / "gate3_F_gamma_onset.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8") |
| predictx/gate3_gamma_onset.py | 398 | gate3_F | rows.to_csv(results_dir / "gate3_F_gamma_onset_sample.csv", index=False) |
| predictx/gate3_registration_fix.py | 353 | gate3_F | (results_dir / "gate3_F_registered_fix.md").write_text("\n".join(lines), encoding="utf-8") |
| predictx/gate3_registration_fix.py | 354 | gate3_F | (results_dir / "gate3_F_registered_fix.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8") |
| predictx/gate3_registration_fix.py | 355 | gate3_F | sample_rows.to_csv(results_dir / "gate3_F_registered_sample.csv", index=False) |
| predictx/gate3_registration_fix.py | 357 | gate3_F | full_rows.to_csv(results_dir / "gate3_F_registered_full.csv", index=False) |
| predictx/gate3_registration_fix.py | 361 | gate3_F | fits = pd.read_csv(root / "results" / "gate3_F_curve_fits.csv") |
| predictx/gate3_registration_fix.py | 386 | gate3_F | draw_registered_examples(sample_rows, sample_plots, root / "figures", "gate3_F_registered_sample_examples.png") |
| predictx/gate3_registration_fix.py | 404 | gate3_F | draw_registered_examples(full_rows, full_plots, root / "figures", "gate3_F_registered_full_examples.png") |
| predictx/gate_fit_optimization_f.py | 108 | gate2_F | gate2 = read_json(root / "results" / "gate2_F_parse.json") |
| predictx/gate_fit_optimization_f.py | 118 | gate2_F | source="results/gate2_F_parse.json:criterion", |
| predictx/gate_fit_optimization_f.py | 126 | gate2_F | source="results/gate2_F_parse.json:criterion", |
| predictx/gate_fit_optimization_f.py | 134 | gate2_F | source="results/gate2_F_parse.json:criterion", |
| predictx/gate_fit_optimization_f.py | 142 | gate2_F | source="results/gate2_F_parse.json:criterion", |
| predictx/gate_reconciliation_audit_f.py | 121 | gate3_F | "gate3_fits": root / "results" / "gate3_F_curve_fits.csv", |
| predictx/gate_reconciliation_audit_f.py | 807 | gate3_F | "fit_source": "results/gate3_F_curve_fits.csv flow-domain deterministic OMBE fit", |
| predictx/gate_truncation_hardening_f.py | 161 | cycle F historical, E/G extension, gate2_F | {"item": "Gate 2 parse", "value": "PASS; cycle F historical; E and G analyzed separately in the three-cycle extension; n 7079; FEV1 2.91%; FVC 3.53%", "source": source_ref("results/gate2_F_parse.md", "3-5,37-39")}, |
| predictx/gate_truncation_hardening_f.py | 162 | cycle F historical, E/G extension, gate3_F | {"item": "Gate 3 adequacy", "value": "FAIL; cycle F historical; E and G analyzed separately in the three-cycle extension; 0/31851 (0.0% of all curves; 0.0% of NRSS-valid curves)", "source": source_ref("results/gate3_F_ombe_rerun.md", "3-5,26")}, |
| predictx/gate_truncation_hardening_f.py | 729 | cycle F historical, E/G extension | "limitation": "Cycle F historical gate; E and G analyzed separately in the three-cycle extension.", |
| predictx/gate_truncation_hardening_f.py | 730 | gate2_F, gate3_F | "source": "results/gate2_F_parse.md:3-5; results/gate3_F_ombe_rerun.md:3-5", |
| predictx/gate_truncation_hardening_f.py | 734 | gate3_F | "source": "results/gate3_F_ombe_rerun.md:26,62", |
| predictx/synthetic_gates.py | 23 | SPXRAW_F | RAW_RDS = ("SPXRAW_E.rds", "SPXRAW_F.rds", "SPXRAW_G.rds") |
| predictx/synthetic_gates.py | 24 | SPXRAW_F | RAW_XPT = ("SPXRAW_E.xpt", "SPXRAW_F.xpt", "SPXRAW_G.xpt") |

## Part 0c - New Dataset Layout

| cycle | year | has_extracted_sas7bdat | has_raw_zip | has_spx_xpt |
|---|---|---|---|---|
| E | 2007-2008 | True | True | True |
| F | 2009-2010 | True | True | True |
| G | 2011-2012 | True | True | True |

## Resolver Check

| cycle | year | raw_path | raw_source | raw_row_count | spx_path | spx_source |
|---|---|---|---|---|---|---|
| E | 2007-2008 | <PROJECT_ROOT>/NHANES_SpiroSBI/2007-2008/spxraw_e.sas7bdat | new-extracted | 34857 | <PROJECT_ROOT>/NHANES_SpiroSBI/2007-2008/SPX_E.xpt | new-nested |
| F | 2009-2010 | <PROJECT_ROOT>/NHANES_SpiroSBI/2009-2010/spxraw_f.sas7bdat | new-extracted | 36421 | <PROJECT_ROOT>/NHANES_SpiroSBI/2009-2010/SPX_F.xpt | new-nested |
| G | 2011-2012 | <PROJECT_ROOT>/NHANES_SpiroSBI/2011-2012/spxraw_g.sas7bdat | new-extracted | 37661 | <PROJECT_ROOT>/NHANES_SpiroSBI/2011-2012/SPX_G.xpt | new-nested |

## Part 5 - F Regression Guard

| metric | old | new(F) | delta | PASS |
|---|---:|---:|---:|---|
| gate2.med_abs_FEV1 | 2.9083271328 | 2.9083271328 | 0 | True |
| gate2.med_abs_FVC | 3.5273255814 | 3.5273255814 | 0 | True |
| gate2.signed_FEV1 | -2.88588914146 | -2.88588914146 | 0 | True |
| gate2.signed_FVC | -3.48932643396 | -3.48932643396 | 0 | True |
| gate2.parse_errors | 45 | 45 | 0 | True |
| gate2.participants_with_truth | 7079 | 7079 | 0 | True |
| clinical.fixed.sensitivity | 0.982456140351 | 0.982456140351 | 0 | True |
| clinical.fixed.specificity | 0.956666666667 | 0.956666666667 | 0 | True |
| clinical.fixed.kappa | 0.911847372455 | 0.911847372455 | 0 | True |
| clinical.fixed.n | 414 | 414 | 0 | True |

## Protocol Decisions

- Implemented a dedicated dataset_resolver.py instead of broadening io.data_path because raw SAS7BDAT/ZIP integrity checks are specific to the validation gates.
- Kept default cycle=F on migrated functions and CLIs so old invocations continue to target F.
- Used the new NHANES_SpiroSBI nested root whenever present; legacy data/ and DATASET?/ fallback is available only when the new root is absent.
- For E/G smoke, parsed curves and counted truth participants but did not run Gate 2 scoring, Gate 3 fitting, truncation, or clinical scoring.
- Classified truncation and severe cached F artifacts for regeneration because they predate resolver provenance and embed F-only/old-path evidence.
- Left a non-F guard in gate_four_part_fix_f.py because that gate still consumes F-named cached clinical/policy rows outside the registered SPX path-only scope.
