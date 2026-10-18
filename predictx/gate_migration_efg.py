from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .dataset_resolver import CYCLE_TO_YEAR, RAW_ROW_COUNTS, resolve_dataset_path, resolver_check
from .gate23_f import derive_all, read_ground_truth, read_raw, run_gate2
from .gate_clinical_classification_f import run as run_clinical

SEED = 20260710

EXPECTED_F = {
    "gate2.med_abs_FEV1": 2.908327132804675,
    "gate2.med_abs_FVC": 3.5273255813953415,
    "gate2.signed_FEV1": -2.885889141458292,
    "gate2.signed_FVC": -3.489326433958947,
    "gate2.parse_errors": 45,
    "gate2.participants_with_truth": 7079,
    "clinical.fixed.sensitivity": 0.9824561403508771,
    "clinical.fixed.specificity": 0.9566666666666667,
    "clinical.fixed.kappa": 0.9118473724554977,
    "clinical.fixed.n": 414,
}

MIGRATED_CHECKLIST = [
    "predictx/gate23_f.py: raw resolver formerly hardcoded data/ and DATASET?/SPXRAW_F paths",
    "predictx/gate23_f.py: summary truth formerly hardcoded data/SPX_F.xpt",
    "predictx/gate23_f.py: Gate 2 and Gate 3 output filenames now include the cycle token",
    "predictx/gate_clinical_classification_f.py: load_truth now filters df[df['cycle'] == cycle]",
    "predictx/gate_clinical_classification_f.py: column_map source fields now use .get(cycle)",
    "predictx/gate_clinical_classification_f.py: raw resolver and output filenames now include the cycle token",
    "predictx/gate_estimand_hybrid_f.py: read_spx_summary now resolves SPX_{cycle}.xpt",
    "predictx/gate_four_part_fix_f.py: Part A SPX summary probe now resolves SPX_{cycle}.xpt",
]

STALE_ARTIFACTS = [
    ("clinical", "results/gate_clinical_classification_F.json"),
    ("clinical", "results/gate_clinical_classification_F.md"),
    ("clinical", "results/gate_clinical_classification_F_rows.csv"),
    ("truncation", "results/gate_truncation_recovery_F.json"),
    ("truncation", "results/gate_truncation_recovery_F.md"),
    ("truncation", "results/gate_truncation_hardening_F.json"),
    ("truncation", "results/gate_truncation_hardening_F.md"),
    ("severe cohort", "results/gate_severe_truncation_diagnosis_F.json"),
    ("severe cohort", "results/gate_severe_truncation_diagnosis_F.md"),
]

PROTOCOL_DECISIONS = [
    "Implemented a dedicated dataset_resolver.py instead of broadening io.data_path because raw SAS7BDAT/ZIP integrity checks are specific to the validation gates.",
    "Kept default cycle=F on migrated functions and CLIs so old invocations continue to target F.",
    "Used the new NHANES_SpiroSBI nested root whenever present; legacy data/ and DATASET?/ fallback is available only when the new root is absent.",
    "For E/G smoke, parsed curves and counted truth participants but did not run Gate 2 scoring, Gate 3 fitting, truncation, or clinical scoring.",
    "Classified truncation and severe cached F artifacts for regeneration because they predate resolver provenance and embed F-only/old-path evidence.",
    "Left a non-F guard in gate_four_part_fix_f.py because that gate still consumes F-named cached clinical/policy rows outside the registered SPX path-only scope.",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cached_f_metrics(root: Path) -> dict[str, float | int]:
    gate2 = read_json(root / "results" / "gate2_F_parse.json")
    clinical = read_json(root / "results" / "gate_clinical_classification_F.json")
    criterion = gate2["criterion"]
    fixed = clinical["fixed"]
    return {
        "gate2.med_abs_FEV1": float(criterion["fev1_median_abs_pct_error"]),
        "gate2.med_abs_FVC": float(criterion["fvc_median_abs_pct_error"]),
        "gate2.signed_FEV1": float(criterion["fev1_median_signed_pct_error"]),
        "gate2.signed_FVC": float(criterion["fvc_median_signed_pct_error"]),
        "gate2.parse_errors": int(gate2["parse_error_count"]),
        "gate2.participants_with_truth": int(gate2["counts"]["participants_with_truth"]),
        "clinical.fixed.sensitivity": float(fixed["sensitivity"]),
        "clinical.fixed.specificity": float(fixed["specificity"]),
        "clinical.fixed.kappa": float(fixed["cohens_kappa"]),
        "clinical.fixed.n": int(fixed["n"]),
    }


def gate2_metrics(gate2: dict[str, Any]) -> dict[str, float | int]:
    criterion = gate2["criterion"]
    return {
        "gate2.med_abs_FEV1": float(criterion["fev1_median_abs_pct_error"]),
        "gate2.med_abs_FVC": float(criterion["fvc_median_abs_pct_error"]),
        "gate2.signed_FEV1": float(criterion["fev1_median_signed_pct_error"]),
        "gate2.signed_FVC": float(criterion["fvc_median_signed_pct_error"]),
        "gate2.parse_errors": int(gate2["parse_error_count"]),
        "gate2.participants_with_truth": int(gate2["counts"]["participants_with_truth"]),
    }


def clinical_metrics(clinical: dict[str, Any]) -> dict[str, float | int]:
    fixed = clinical["fixed"]
    return {
        "clinical.fixed.sensitivity": float(fixed["sensitivity"]),
        "clinical.fixed.specificity": float(fixed["specificity"]),
        "clinical.fixed.kappa": float(fixed["cohens_kappa"]),
        "clinical.fixed.n": int(fixed["n"]),
    }


def compare_metrics(old: dict[str, float | int], new: dict[str, float | int]) -> list[dict[str, Any]]:
    rows = []
    for metric, old_value in old.items():
        new_value = new[metric]
        delta = float(new_value) - float(old_value)
        if isinstance(old_value, int):
            passed = int(new_value) == old_value
        else:
            passed = abs(delta) <= 1e-6
        rows.append(
            {
                "metric": metric,
                "old": old_value,
                "new": new_value,
                "delta": delta,
                "pass": bool(passed),
            }
        )
    return rows


def source_audit(root: Path) -> list[dict[str, Any]]:
    patterns = (
        "DATASET?",
        "SPXRAW_F",
        "spxraw_f.sas7bdat",
        "SPX_F.xpt",
        'cycle == "F"',
        ".get(\"F\")",
        "F only",
        "E and G",
        "gate2_F",
        "gate3_F",
    )
    rows = []
    for base in (root / "predictx", root / "tests"):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(root)
            if str(rel) == "predictx/gate_migration_efg.py":
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                hits = [pat for pat in patterns if pat in line]
                if hits:
                    rows.append(
                        {
                            "path": str(rel),
                            "line": number,
                            "patterns": hits,
                            "text": line.strip(),
                        }
                    )
    return rows


def dataset_listing(root: Path) -> list[dict[str, Any]]:
    data_root = root / "NHANES_SpiroSBI"
    rows = []
    for cycle, year in CYCLE_TO_YEAR.items():
        year_dir = data_root / year
        files = sorted(p.name for p in year_dir.iterdir() if p.is_file()) if year_dir.exists() else []
        rows.append(
            {
                "cycle": cycle,
                "year": year,
                "folder": str(year_dir),
                "has_extracted_sas7bdat": any(name.lower() == f"spxraw_{cycle.lower()}.sas7bdat" for name in files),
                "has_raw_zip": f"SPXRAW_{cycle}.zip" in files,
                "has_spx_xpt": f"SPX_{cycle}.xpt" in files,
                "files": files,
            }
        )
    return rows


def smoke_parse_cycle(root: Path, cycle: str) -> dict[str, Any]:
    raw_resolved = resolve_dataset_path(root, "SPXRAW", cycle)
    spx_resolved = resolve_dataset_path(root, "SPX", cycle)
    raw, raw_info = read_raw(raw_resolved.path, cycle)
    derived = derive_all(raw)
    truth = read_ground_truth(root, cycle)
    participants_with_truth = int(truth[["SPXNFEV1", "SPXNFVC"]].dropna().shape[0])
    return {
        "cycle": cycle,
        "raw_path": str(raw_resolved.path),
        "raw_source": raw_resolved.source,
        "raw_note": raw_resolved.note,
        "spx_path": str(spx_resolved.path),
        "spx_source": spx_resolved.source,
        "metadata_rows": int(raw_info["metadata_rows"]),
        "loaded_rows": int(raw_info["loaded_rows"]),
        "expected_raw_rows": RAW_ROW_COUNTS[cycle],
        "parse_errors": int(derived.attrs.get("parse_error_count", 0)),
        "participants_with_truth": participants_with_truth,
        "distinct_from_f": None,
    }


def classify_stale_artifacts(root: Path, regression_passed: bool) -> dict[str, list[dict[str, Any]]]:
    reuse: list[dict[str, Any]] = []
    regenerate: list[dict[str, Any]] = []
    for group, rel_path in STALE_ARTIFACTS:
        path = root / rel_path
        exists = path.exists()
        if group == "clinical" and regression_passed:
            reuse.append(
                {
                    "file": rel_path,
                    "exists": exists,
                    "reason": "regenerated by the migrated cycle-parametric clinical path during Part 5",
                }
            )
        else:
            regenerate.append(
                {
                    "file": rel_path,
                    "exists": exists,
                    "reason": "predates this resolver migration and/or embeds F-only raw-source provenance; regenerate before three-cycle analysis",
                }
            )
    return {"reuse_as_is": reuse, "regenerate": regenerate}


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    out = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---" for _ in columns]) + "|"]
    for row in rows:
        out.append("| " + " | ".join(format_md(row.get(col, "")) for col in columns) + " |")
    return out


def format_md(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.12g}"
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)
    return str(value)


def write_regression_report(root: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Cycle-Parametric Migration Regression, Cycle F",
        "",
        f"- VERDICT: {result['verdict']}",
        f"- seed: {SEED}",
        f"- elapsed seconds: {result['elapsed_s']:.2f}",
        "",
        "## Part 0a - Cached F Regression Targets",
        "",
        "| metric | expected | cached before migration | delta | pass |",
        "|---|---:|---:|---:|---|",
    ]
    for row in result["expected_checks"]:
        lines.append(
            f"| {row['metric']} | {format_md(row['expected'])} | {format_md(row['cached'])} | {format_md(row['delta'])} | {row['pass']} |"
        )
    lines.extend(["", "## Part 0b - Migrated Checklist", ""])
    lines.extend(f"- {item}" for item in MIGRATED_CHECKLIST)
    lines.extend(["", "## Current Source Audit After Migration", ""])
    if result["source_audit"]:
        lines.extend(md_table(result["source_audit"], ["path", "line", "patterns", "text"]))
    else:
        lines.append("No hardcoded F references found in predictx/tests.")
    lines.extend(["", "## Part 0c - New Dataset Layout", ""])
    layout_rows = [
        {
            "cycle": row["cycle"],
            "year": row["year"],
            "has_extracted_sas7bdat": row["has_extracted_sas7bdat"],
            "has_raw_zip": row["has_raw_zip"],
            "has_spx_xpt": row["has_spx_xpt"],
        }
        for row in result["dataset_listing"]
    ]
    lines.extend(md_table(layout_rows, ["cycle", "year", "has_extracted_sas7bdat", "has_raw_zip", "has_spx_xpt"]))
    lines.extend(["", "## Resolver Check", ""])
    lines.extend(md_table(result["resolver_check"], ["cycle", "year", "raw_path", "raw_source", "raw_row_count", "spx_path", "spx_source"]))
    lines.extend(["", "## Part 5 - F Regression Guard", ""])
    lines.extend(["| metric | old | new(F) | delta | PASS |", "|---|---:|---:|---:|---|"])
    for row in result["regression_rows"]:
        lines.append(
            f"| {row['metric']} | {format_md(row['old'])} | {format_md(row['new'])} | {format_md(row['delta'])} | {row['pass']} |"
        )
    lines.extend(["", "## Protocol Decisions", ""])
    lines.extend(f"- {item}" for item in result["protocol_decisions"])
    (root / "results" / "migration_regression_F.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_smoke_report(root: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Migration Smoke Test, Cycles E/G",
        "",
        f"- VERDICT: {result['verdict']}",
        f"- status: {result['smoke_status']}",
        "",
    ]
    if result["smoke_rows"]:
        lines.extend(md_table(result["smoke_rows"], ["cycle", "raw_source", "loaded_rows", "expected_raw_rows", "parse_errors", "participants_with_truth", "distinct_from_f"]))
    else:
        lines.append("Smoke test not run because the F regression guard failed.")
    (root / "results" / "migration_smoke_EG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stale_report(root: Path, result: dict[str, Any]) -> None:
    stale = result["stale_audit"]
    lines = [
        "# Migration Stale Artifact Audit",
        "",
        f"- VERDICT: {result['verdict']}",
        "",
        "## REUSE-AS-IS",
        "",
    ]
    if stale["reuse_as_is"]:
        lines.extend(md_table(stale["reuse_as_is"], ["file", "exists", "reason"]))
    else:
        lines.append("None.")
    lines.extend(["", "## REGENERATE", ""])
    if stale["regenerate"]:
        lines.extend(md_table(stale["regenerate"], ["file", "exists", "reason"]))
    else:
        lines.append("None.")
    (root / "results" / "migration_stale_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def run(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    cached_before = cached_f_metrics(root)
    expected_checks = []
    for metric, expected in EXPECTED_F.items():
        cached = cached_before[metric]
        delta = float(cached) - float(expected)
        passed = int(cached) == expected if isinstance(expected, int) else abs(delta) <= 1e-6
        expected_checks.append(
            {
                "metric": metric,
                "expected": expected,
                "cached": cached,
                "delta": delta,
                "pass": bool(passed),
            }
        )

    dataset_rows = dataset_listing(root)
    resolver_rows = resolver_check(root)

    gate2_f, _raw, _derived, _truth = run_gate2(root, "F")
    clinical_f = run_clinical(root, "F")
    new_metrics = {}
    new_metrics.update(gate2_metrics(gate2_f))
    new_metrics.update(clinical_metrics(clinical_f))
    regression_rows = compare_metrics(cached_before, new_metrics)
    regression_passed = all(row["pass"] for row in regression_rows)

    smoke_rows: list[dict[str, Any]] = []
    smoke_status = "NOT-RUN"
    verdict = "MIGRATION-BLOCKED"
    if regression_passed:
        f_smoke = {
            "cycle": "F",
            "loaded_rows": int(gate2_f["raw_info"]["loaded_rows"]),
            "parse_errors": int(gate2_f["parse_error_count"]),
            "participants_with_truth": int(gate2_f["counts"]["participants_with_truth"]),
        }
        smoke_status = "PASS"
        for cycle in ("E", "G"):
            row = smoke_parse_cycle(root, cycle)
            row["distinct_from_f"] = (
                row["loaded_rows"],
                row["parse_errors"],
                row["participants_with_truth"],
            ) != (
                f_smoke["loaded_rows"],
                f_smoke["parse_errors"],
                f_smoke["participants_with_truth"],
            )
            if not row["distinct_from_f"]:
                smoke_status = "FAIL"
            smoke_rows.append(row)
        verdict = "MIGRATION-CLEAN" if smoke_status == "PASS" else "MIGRATION-CYCLE-QUIRK"

    source_rows = source_audit(root)
    stale = classify_stale_artifacts(root, regression_passed)
    result = {
        "verdict": verdict,
        "seed": SEED,
        "elapsed_s": float(time.perf_counter() - started),
        "cached_f_before_migration": cached_before,
        "expected_checks": expected_checks,
        "dataset_listing": dataset_rows,
        "resolver_check": resolver_rows,
        "migrated_checklist": MIGRATED_CHECKLIST,
        "source_audit": source_rows,
        "regression_rows": regression_rows,
        "regression_passed": regression_passed,
        "smoke_status": smoke_status,
        "smoke_rows": smoke_rows,
        "stale_audit": stale,
        "protocol_decisions": PROTOCOL_DECISIONS,
    }
    write_regression_report(root, result)
    write_smoke_report(root, result)
    write_stale_report(root, result)
    (root / "results" / "migration_EFG.json").write_text(json.dumps(to_jsonable(result), indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--reports-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.reports_only:
        result = read_json(root / "results" / "migration_EFG.json")
        result["source_audit"] = source_audit(root)
        result["protocol_decisions"] = PROTOCOL_DECISIONS
        write_regression_report(root, result)
        write_smoke_report(root, result)
        write_stale_report(root, result)
        (root / "results" / "migration_EFG.json").write_text(json.dumps(to_jsonable(result), indent=2) + "\n", encoding="utf-8")
    else:
        result = run(root)
    print(f"VERDICT: {result['verdict']}")
    for row in result["regression_rows"]:
        print(f"{row['metric']}: old={row['old']} new={row['new']} delta={row['delta']} pass={row['pass']}")
    if result["smoke_rows"]:
        for row in result["smoke_rows"]:
            print(
                f"cycle {row['cycle']}: rows={row['loaded_rows']} parse_errors={row['parse_errors']} "
                f"participants={row['participants_with_truth']} distinct_from_f={row['distinct_from_f']}"
            )
    return 0 if result["verdict"] == "MIGRATION-CLEAN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
