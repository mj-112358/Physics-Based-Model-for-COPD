from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from . import gate_clinical_classification_f, gate_fvc_correction_f, gate_policy_calibrated_flag_f
from .gate_clinical_classification_f import q
from .ombe import midrange_phi, ombe_flow, ombe_volume

GATES = {
    "clinical": "gate_clinical_classification_F",
    "fvc": "gate_fvc_correction_F",
    "policy": "gate_policy_calibrated_flag_F",
}


def clean_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    return obj


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(clean_json(payload), indent=2) + "\n", encoding="utf-8")


def fmt_num(value: Any, digits: int = 4) -> str:
    try:
        value_f = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(value_f):
        return "NA"
    return f"{value_f:.{digits}f}"


def fmt_delta(new: Any, old: Any, digits: int = 4) -> str:
    try:
        new_f = float(new)
        old_f = float(old)
    except Exception:
        return "NA"
    if not np.isfinite(new_f) or not np.isfinite(old_f):
        return "NA"
    return f"{new_f - old_f:+.{digits}f}"


def md_table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    out = ["| " + " | ".join(label for _, label, _ in columns) + " |"]
    out.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        cells = []
        for key, _, kind in columns:
            value = row.get(key)
            if kind == "int":
                cells.append(str(int(value)) if pd.notna(value) else "NA")
            elif kind.startswith("float"):
                digits = int(kind.replace("float", "") or "4")
                cells.append(fmt_num(value, digits))
            else:
                cells.append(str(value))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def signed_summary(values: pd.Series | np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)) if arr.size else np.nan,
        "iqr": q(arr),
    }


def obstruction_error_summary(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {}
    obstructed = rows[rows["sample_group"] == "obstructed"].copy() if "sample_group" in rows else rows.copy()
    needed = ["model_fev1_mL", "model_fvc_mL", "model_ratio", "fev1_mL", "fvc_mL", "fev1_fvc_ratio"]
    valid = obstructed.dropna(subset=needed).copy()
    valid = valid[(valid["fev1_mL"] > 0) & (valid["fvc_mL"] > 0)]
    fev1_pct = 100.0 * (valid["model_fev1_mL"] - valid["fev1_mL"]) / valid["fev1_mL"]
    fvc_pct = 100.0 * (valid["model_fvc_mL"] - valid["fvc_mL"]) / valid["fvc_mL"]
    ratio = valid["model_ratio"] - valid["fev1_fvc_ratio"]
    return {
        "n": int(len(valid)),
        "fev1_abs_pct": signed_summary(np.abs(fev1_pct)),
        "fev1_signed_pct": signed_summary(fev1_pct),
        "fvc_abs_pct": signed_summary(np.abs(fvc_pct)),
        "fvc_signed_pct": signed_summary(fvc_pct),
        "ratio_abs": signed_summary(np.abs(ratio)),
        "ratio_signed": signed_summary(ratio),
    }


def severity_confusion(rows: pd.DataFrame) -> dict:
    if rows.empty or "truth_severity" not in rows or "model_severity" not in rows:
        return {"table": {}, "severe_demoted_normal_n": 0, "severe_demoted_normal_rate": np.nan}
    truth_order = ["severe", "moderate", "mild_0.60_0.70", "normal"]
    model_order = ["severe", "moderate", "mild_0.60_0.70", "normal"]
    tab = (
        pd.crosstab(rows["truth_severity"], rows["model_severity"], dropna=False)
        .reindex(index=truth_order, columns=model_order, fill_value=0)
    )
    severe = rows[rows["truth_severity"] == "severe"]
    demoted_normal = int((severe["model_severity"] == "normal").sum()) if len(severe) else 0
    return {
        "table": {str(idx): {str(col): int(tab.loc[idx, col]) for col in tab.columns} for idx in tab.index},
        "severe_demoted_normal_n": demoted_normal,
        "severe_demoted_normal_rate": float(demoted_normal / len(severe)) if len(severe) else np.nan,
    }


def clinical_summary(result: dict, rows: pd.DataFrame) -> dict:
    return {
        "verdict": result.get("verdict"),
        "fixed": result.get("fixed", {}),
        "lln": result.get("lln", {}),
        "errors_obstructed_recomputed": obstruction_error_summary(rows),
        "severity": severity_confusion(rows),
        "reported_errors_obstructed": result.get("errors_obstructed", {}),
    }


def fvc_summary(result: dict) -> dict:
    return {
        "verdict": result.get("verdict"),
        "preferred_variant": result.get("preferred_variant"),
        "preferred_signed_ratio_median": result.get("preferred_signed_ratio_median"),
        "preferred_signed_ratio_iqr": result.get("preferred_signed_ratio_iqr"),
        "signed_error_summaries": result.get("signed_error_summaries", []),
        "convention_summaries": result.get("convention_summaries", []),
        "classification": result.get("classification", {}),
        "borderline": result.get("borderline", {}),
    }


def policy_extract(result: dict, correction: str, variant: str) -> dict:
    metrics = result.get("candidate_metrics", {}).get(correction, {}).get("fixed", {}).get(variant, {})
    nonflag = metrics.get("confident_nonflagged", {})
    return {
        "residual_fn": metrics.get("obstructed_truth", {}).get("normal_fraction"),
        "borderline_fn": metrics.get("borderline_residual_false_negative_rate"),
        "normal_truth_flag_burden": metrics.get("flag_burden_normal_truth"),
        "nonflag_sensitivity": nonflag.get("sensitivity"),
        "nonflag_specificity": nonflag.get("specificity"),
        "nonflag_kappa": nonflag.get("cohens_kappa"),
    }


def policy_summary(result: dict) -> dict:
    baseline = result.get("baseline_test_metrics", {})
    base_nonflag = baseline.get("confident_nonflagged", {})
    out = {
        "verdict": result.get("verdict"),
        "preferred_correction": result.get("preferred_correction"),
        "preferred_policy_variant": result.get("preferred_policy_variant"),
        "c1": result.get("corrections", {}).get("c1", {}),
        "c2": result.get("corrections", {}).get("c2", {}),
        "baseline": {
            "residual_fn": baseline.get("obstructed_truth", {}).get("normal_fraction"),
            "borderline_fn": baseline.get("borderline_residual_false_negative_rate"),
            "normal_truth_flag_burden": baseline.get("flag_burden_normal_truth"),
            "nonflag_sensitivity": base_nonflag.get("sensitivity"),
            "nonflag_specificity": base_nonflag.get("specificity"),
            "nonflag_kappa": base_nonflag.get("cohens_kappa"),
        },
        "c1_correction_only": policy_extract(result, "c1", "correction_only"),
        "c1_flag_unbanded": policy_extract(result, "c1", "flag_unbanded"),
        "c2_correction_only": policy_extract(result, "c2", "correction_only"),
        "c2_flag_unbanded": policy_extract(result, "c2", "flag_unbanded"),
    }
    return out


def module_audit() -> list[dict]:
    return [
        {
            "module_function": "predictx/ombe.py::ombe_volume",
            "path": "shared JAX OMBE volume",
            "status": "fixed",
            "evidence": "uses jnp.diff(t) for trapezoid interval widths",
        },
        {
            "module_function": "predictx/ombe_gamma.py::ombe_gamma_volume",
            "path": "JAX OMBE-gamma volume, out of scope for this gate",
            "status": "fixed",
            "evidence": "uses jnp.diff(t) for trapezoid interval widths",
        },
        {
            "module_function": "predictx/gate3_diagnosis.py::np_ombe_volume",
            "path": "NumPy OMBE volume used by clinical/FVC/policy point gates",
            "status": "already-correct",
            "evidence": "calls cumulative_trapezoid(...), which uses np.diff(x)",
        },
        {
            "module_function": "predictx/gate_clinical_classification_f.py::fit_selected_rows",
            "path": "clinical classification point estimates",
            "status": "already-correct-through-import",
            "evidence": "imports fit_ombe_volume_lm from gate3_diagnosis.py",
        },
        {
            "module_function": "predictx/gate_fvc_correction_f.py::run",
            "path": "FVC convention gate",
            "status": "already-correct-through-clinical-rows",
            "evidence": "uses gate_clinical_classification_F_rows.csv and fit_selected_rows",
        },
        {
            "module_function": "predictx/gate_policy_calibrated_flag_f.py::run",
            "path": "bias correction + ATS flag policy",
            "status": "already-correct-through-import",
            "evidence": "imports fit_selected_rows from gate_clinical_classification_f.py",
        },
        {
            "module_function": "predictx/gate_likelihood_repair_f.py and posterior gates",
            "path": "JAX likelihood/posterior paths",
            "status": "fixed-by-shared-ombe",
            "evidence": "these import predictx.ombe.ombe_volume, now nonuniform-grid safe",
        },
    ]


def nonuniform_integration_assertion() -> dict:
    phi = midrange_phi()
    t = jnp.asarray([0.0, 0.003, 0.013, 0.023, 0.050, 0.100, 0.200], dtype=jnp.float32)
    flow = np.asarray(ombe_flow(phi, t), dtype=float)
    expected = np.concatenate([[0.0], np.cumsum((flow[:-1] + flow[1:]) * 0.5 * np.diff(np.asarray(t)))])
    constant_dt = np.concatenate([[0.0], np.cumsum((flow[:-1] + flow[1:]) * 0.5 * float(t[1] - t[0]))])
    got = np.asarray(ombe_volume(phi, t), dtype=float)
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-6)
    assert not np.allclose(got[-1], constant_dt[-1], rtol=1e-2, atol=1e-4)
    return {
        "status": "PASS",
        "grid_s": [float(x) for x in np.asarray(t)],
        "correct_final_volume_l": float(got[-1]),
        "constant_dt_final_volume_l": float(constant_dt[-1]),
    }


def snapshot_outputs(root: Path) -> dict:
    results = root / "results"
    out = {}
    for key, stem in GATES.items():
        out[key] = {
            "json": load_json(results / f"{stem}.json"),
            "rows": load_rows(results / f"{stem}_rows.csv"),
        }
    return out


def rerun_gates(root: Path) -> dict:
    clinical = gate_clinical_classification_f.run(root)
    fvc = gate_fvc_correction_f.run(root)
    policy = gate_policy_calibrated_flag_f.run(root)
    return {"clinical": clinical, "fvc": fvc, "policy": policy}


def clinical_metric_rows(old: dict, new: dict) -> list[dict]:
    rows = []
    for definition in ["fixed", "lln"]:
        old_m = old.get(definition, {})
        new_m = new.get(definition, {})
        for metric in ["n", "tp", "tn", "fp", "fn", "sensitivity", "specificity", "agreement", "cohens_kappa"]:
            rows.append(
                {
                    "definition": definition,
                    "metric": metric,
                    "old": old_m.get(metric),
                    "new": new_m.get(metric),
                    "delta": (new_m.get(metric) - old_m.get(metric)) if isinstance(new_m.get(metric), (int, float)) and isinstance(old_m.get(metric), (int, float)) else np.nan,
                }
            )
    return rows


def error_rows(old: dict, new: dict) -> list[dict]:
    rows = []
    for metric in ["fev1_abs_pct", "fev1_signed_pct", "fvc_abs_pct", "fvc_signed_pct", "ratio_abs", "ratio_signed"]:
        old_m = old.get(metric, {})
        new_m = new.get(metric, {})
        rows.append(
            {
                "metric": metric,
                "old_median": old_m.get("median"),
                "new_median": new_m.get("median"),
                "delta_median": (new_m.get("median") - old_m.get("median")) if isinstance(new_m.get("median"), (int, float)) and isinstance(old_m.get("median"), (int, float)) else np.nan,
                "old_iqr": old_m.get("iqr"),
                "new_iqr": new_m.get("iqr"),
            }
        )
    return rows


def policy_metric_rows(old: dict, new: dict) -> list[dict]:
    rows = []
    for block in ["baseline", "c1_correction_only", "c1_flag_unbanded", "c2_correction_only", "c2_flag_unbanded"]:
        old_b = old.get(block, {})
        new_b = new.get(block, {})
        for metric in [
            "residual_fn",
            "borderline_fn",
            "normal_truth_flag_burden",
            "nonflag_sensitivity",
            "nonflag_specificity",
            "nonflag_kappa",
        ]:
            rows.append(
                {
                    "block": block,
                    "metric": metric,
                    "old": old_b.get(metric),
                    "new": new_b.get(metric),
                    "delta": (new_b.get(metric) - old_b.get(metric)) if isinstance(new_b.get(metric), (int, float)) and isinstance(old_b.get(metric), (int, float)) else np.nan,
                }
            )
    return rows


def fvc_obstructed_summary(fvc: dict) -> dict:
    for row in fvc.get("signed_error_summaries", []):
        if row.get("group") == "obstructed":
            return row
    return {}


def convention_lookup(fvc: dict, scope: str = "obstructed") -> dict:
    out = {}
    for row in fvc.get("convention_summaries", []):
        if row.get("scope") == scope:
            out[row.get("variant")] = row
    return out


def build_rows_csv(old_snap: dict, new_snap: dict) -> pd.DataFrame:
    frames = []
    for gate in GATES:
        for label, snap in [("old_snapshot", old_snap), ("new_rerun", new_snap)]:
            df = snap[gate]["rows"].copy()
            if df.empty:
                continue
            df.insert(0, "run_label", label)
            df.insert(0, "source_gate", gate)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def audit_markdown(audit: list[dict]) -> str:
    return md_table(
        audit,
        [
            ("module_function", "module/function", "str"),
            ("path", "use", "str"),
            ("status", "status", "str"),
            ("evidence", "evidence", "str"),
        ],
    )


def metric_markdown(rows: list[dict], dimensions: tuple[str, str]) -> str:
    dim_a, dim_b = dimensions
    out = [f"| {dim_a} | {dim_b} | old | new | delta |", "|---|---|---:|---:|---:|"]
    for row in rows:
        out.append(
            f"| {row[dim_a]} | {row[dim_b]} | {fmt_num(row['old'])} | {fmt_num(row['new'])} | {fmt_delta(row['new'], row['old'])} |"
        )
    return "\n".join(out)


def error_markdown(rows: list[dict]) -> str:
    out = ["| metric | old median | new median | delta | old IQR | new IQR |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        out.append(
            f"| {row['metric']} | {fmt_num(row['old_median'])} | {fmt_num(row['new_median'])} | "
            f"{fmt_delta(row['new_median'], row['old_median'])} | {row['old_iqr']} | {row['new_iqr']} |"
        )
    return "\n".join(out)


def convention_markdown(old_fvc: dict, new_fvc: dict) -> str:
    old_c = convention_lookup(old_fvc)
    new_c = convention_lookup(new_fvc)
    variants = sorted(set(old_c) | set(new_c))
    out = ["| convention | old signed ratio median | new signed ratio median | delta | old IQR | new IQR |", "|---|---:|---:|---:|---:|---:|"]
    for variant in variants:
        old = old_c.get(variant, {})
        new = new_c.get(variant, {})
        out.append(
            f"| {variant} | {fmt_num(old.get('signed_ratio_median'))} | {fmt_num(new.get('signed_ratio_median'))} | "
            f"{fmt_delta(new.get('signed_ratio_median'), old.get('signed_ratio_median'))} | "
            f"{old.get('signed_ratio_iqr')} | {new.get('signed_ratio_iqr')} |"
        )
    return "\n".join(out)


def reconciliation(old_clinical: dict, new_clinical: dict, old_fvc: dict, new_fvc: dict, old_policy: dict, new_policy: dict) -> dict:
    old_fixed = old_clinical.get("fixed", {})
    new_fixed = new_clinical.get("fixed", {})
    old_ratio = old_clinical.get("errors_obstructed_recomputed", {}).get("ratio_signed", {}).get("median")
    new_ratio = new_clinical.get("errors_obstructed_recomputed", {}).get("ratio_signed", {}).get("median")
    old_fvc_obs = fvc_obstructed_summary(old_fvc)
    new_fvc_obs = fvc_obstructed_summary(new_fvc)
    old_border = old_policy.get("baseline", {}).get("borderline_fn")
    new_border = new_policy.get("baseline", {}).get("borderline_fn")
    return {
        "kappa_sensitivity_specificity": (
            f"Fixed-threshold sensitivity/specificity/kappa moved "
            f"{fmt_delta(new_fixed.get('sensitivity'), old_fixed.get('sensitivity'), 6)}, "
            f"{fmt_delta(new_fixed.get('specificity'), old_fixed.get('specificity'), 6)}, "
            f"{fmt_delta(new_fixed.get('cohens_kappa'), old_fixed.get('cohens_kappa'), 6)} respectively."
        ),
        "ratio_bias": (
            f"Signed obstructed-set ratio median changed from {fmt_num(old_ratio)} to {fmt_num(new_ratio)} "
            f"(delta {fmt_delta(new_ratio, old_ratio)})."
        ),
        "fvc_mechanism": (
            f"Obstructed signed FVC percent median changed from {fmt_num(old_fvc_obs.get('signed_fvc_pct_median'), 2)}% "
            f"to {fmt_num(new_fvc_obs.get('signed_fvc_pct_median'), 2)}%."
        ),
        "borderline_leak": (
            f"Baseline TEST borderline residual false-negative rate changed from {fmt_num(old_border, 3)} "
            f"to {fmt_num(new_border, 3)}."
        ),
        "retractions": (
            "No point-estimate conclusion is retracted from this rerun because the point-estimate gates were already using the "
            "NumPy nonuniform-grid volume fitter. The review premise that clinical/FVC/policy used the broken JAX dt path is not supported by the code audit. "
            "Retract only the narrower implementation statement that the shared JAX OMBE volume helper was safe on nonuniform real-curve grids before this patch."
        ),
    }


def make_report(result: dict) -> str:
    old_clinical = result["old"]["clinical"]
    new_clinical = result["new"]["clinical"]
    old_fvc = result["old"]["fvc"]
    new_fvc = result["new"]["fvc"]
    old_policy = result["old"]["policy"]
    new_policy = result["new"]["policy"]
    rec = result["reconciliation"]
    lines = [
        "# Gate - Reclassification After Fixed Integration, Cycle F",
        "",
        f"- VERDICT: {result['verdict']} - {result['verdict_reason']}",
        "- Scope: clinical classification, FVC convention, and ATS flag policy gates only.",
        "- No gamma, registration, ML residual, or threshold movement was introduced.",
        "",
        "## Step 0 - Integration Audit",
        "",
        audit_markdown(result["module_audit"]),
        "",
        f"Nonuniform-grid regression assertion: {result['nonuniform_assertion']['status']}. "
        f"Correct final volume={result['nonuniform_assertion']['correct_final_volume_l']:.6f} L; "
        f"constant-dt final volume would be {result['nonuniform_assertion']['constant_dt_final_volume_l']:.6f} L.",
        "",
        "Important reconciliation: the shared JAX volume helper was broken before this patch, but the Cycle-F point-estimate gates audited here already used `gate3_diagnosis.np_ombe_volume`, which integrates with `np.diff(t)`. Therefore old-vs-new point-estimate changes are expected to be zero or numerical noise.",
        "",
        "## Step 1 - Clinical Classification Old vs New",
        "",
        metric_markdown(result["comparisons"]["clinical_metrics"], ("definition", "metric")),
        "",
        "### Obstructed-Set Errors",
        "",
        error_markdown(result["comparisons"]["clinical_errors"]),
        "",
        "### Severity Confusion",
        "",
        "Old severity table:",
        "",
        "```json",
        json.dumps(clean_json(old_clinical["severity"]["table"]), indent=2),
        "```",
        "",
        "New severity table:",
        "",
        "```json",
        json.dumps(clean_json(new_clinical["severity"]["table"]), indent=2),
        "```",
        "",
        f"Severe demotion-to-normal old={fmt_num(old_clinical['severity']['severe_demoted_normal_rate'], 3)}, new={fmt_num(new_clinical['severity']['severe_demoted_normal_rate'], 3)}, delta={fmt_delta(new_clinical['severity']['severe_demoted_normal_rate'], old_clinical['severity']['severe_demoted_normal_rate'], 3)}.",
        "",
        "## Step 2 - FVC Convention Old vs New",
        "",
        "| field | old | new |",
        "|---|---:|---:|",
        f"| verdict | {old_fvc.get('verdict')} | {new_fvc.get('verdict')} |",
        f"| preferred variant | {old_fvc.get('preferred_variant')} | {new_fvc.get('preferred_variant')} |",
        f"| preferred signed ratio median | {fmt_num(old_fvc.get('preferred_signed_ratio_median'))} | {fmt_num(new_fvc.get('preferred_signed_ratio_median'))} |",
        "",
        "### Obstructed Signed Error Mechanism",
        "",
        "```json",
        json.dumps(clean_json({"old": fvc_obstructed_summary(old_fvc), "new": fvc_obstructed_summary(new_fvc)}), indent=2),
        "```",
        "",
        "### FVC Convention Comparison",
        "",
        convention_markdown(old_fvc, new_fvc),
        "",
        "## Step 3 - Bias Correction + Flag Policy Old vs New",
        "",
        "| correction parameter | old | new | delta |",
        "|---|---:|---:|---:|",
        f"| c1 constant bias | {fmt_num(old_policy.get('c1', {}).get('bias'), 6)} | {fmt_num(new_policy.get('c1', {}).get('bias'), 6)} | {fmt_delta(new_policy.get('c1', {}).get('bias'), old_policy.get('c1', {}).get('bias'), 6)} |",
        f"| c2 intercept | {fmt_num(old_policy.get('c2', {}).get('intercept'), 6)} | {fmt_num(new_policy.get('c2', {}).get('intercept'), 6)} | {fmt_delta(new_policy.get('c2', {}).get('intercept'), old_policy.get('c2', {}).get('intercept'), 6)} |",
        f"| c2 slope | {fmt_num(old_policy.get('c2', {}).get('slope'), 6)} | {fmt_num(new_policy.get('c2', {}).get('slope'), 6)} | {fmt_delta(new_policy.get('c2', {}).get('slope'), old_policy.get('c2', {}).get('slope'), 6)} |",
        "",
        metric_markdown(result["comparisons"]["policy_metrics"], ("block", "metric")),
        "",
        "## Step 4 - Reconciliation",
        "",
        f"a. {rec['kappa_sensitivity_specificity']}",
        "",
        f"b. {rec['ratio_bias']}",
        "",
        f"c. {rec['fvc_mechanism']}",
        "",
        f"d. {rec['borderline_leak']}",
        "",
        f"e. {rec['retractions']}",
        "",
        "## Step 5 - Integrity Note",
        "",
        result["integrity_note"],
        "",
        f"HEADLINE: fixed sensitivity {fmt_num(new_clinical['fixed'].get('sensitivity'), 3)}, fixed specificity {fmt_num(new_clinical['fixed'].get('specificity'), 3)}, fixed kappa {fmt_num(new_clinical['fixed'].get('cohens_kappa'), 3)}, signed ratio median {fmt_num(new_clinical['errors_obstructed_recomputed']['ratio_signed'].get('median'), 4)}, policy baseline borderline FN {fmt_num(new_policy['baseline'].get('borderline_fn'), 3)}.",
        "",
    ]
    return "\n".join(lines)


def run(root: Path) -> dict:
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    old_snap = snapshot_outputs(root)

    assertion = nonuniform_integration_assertion()
    rerun_gates(root)
    new_snap = snapshot_outputs(root)

    old = {
        "clinical": clinical_summary(old_snap["clinical"]["json"], old_snap["clinical"]["rows"]),
        "fvc": fvc_summary(old_snap["fvc"]["json"]),
        "policy": policy_summary(old_snap["policy"]["json"]),
    }
    new = {
        "clinical": clinical_summary(new_snap["clinical"]["json"], new_snap["clinical"]["rows"]),
        "fvc": fvc_summary(new_snap["fvc"]["json"]),
        "policy": policy_summary(new_snap["policy"]["json"]),
    }
    comparisons = {
        "clinical_metrics": clinical_metric_rows(old["clinical"], new["clinical"]),
        "clinical_errors": error_rows(
            old["clinical"].get("errors_obstructed_recomputed", {}),
            new["clinical"].get("errors_obstructed_recomputed", {}),
        ),
        "policy_metrics": policy_metric_rows(old["policy"], new["policy"]),
    }
    rec = reconciliation(old["clinical"], new["clinical"], old["fvc"], new["fvc"], old["policy"], new["policy"])
    c1_old = old["policy"].get("c1", {}).get("bias")
    c1_new = new["policy"].get("c1", {}).get("bias")
    ratio_old = old["clinical"].get("errors_obstructed_recomputed", {}).get("ratio_signed", {}).get("median")
    ratio_new = new["clinical"].get("errors_obstructed_recomputed", {}).get("ratio_signed", {}).get("median")
    materially_changed = any(
        abs(float(v)) > 1e-9
        for v in [
            (new["clinical"].get("fixed", {}).get("sensitivity", np.nan) - old["clinical"].get("fixed", {}).get("sensitivity", np.nan)),
            (new["clinical"].get("fixed", {}).get("specificity", np.nan) - old["clinical"].get("fixed", {}).get("specificity", np.nan)),
            (new["clinical"].get("fixed", {}).get("cohens_kappa", np.nan) - old["clinical"].get("fixed", {}).get("cohens_kappa", np.nan)),
            (ratio_new - ratio_old),
            (c1_new - c1_old),
        ]
        if np.isfinite(float(v))
    )
    verdict = "UNCHANGED-POINT-GATES" if not materially_changed else "POINT-GATES-CHANGED"
    verdict_reason = (
        "clinical/FVC/policy rerun values are unchanged because these gates already used the nonuniform NumPy volume path"
        if not materially_changed
        else "one or more point-gate metrics changed after the shared integration fix"
    )
    integrity_note = (
        "The c1 correction remains numerically identical in this rerun, so its empirical effect is unchanged. However, it should not be justified as a repair for the shared JAX constant-dt bug in these point-estimate gates; that bug did not feed this path. The defensible justification remains the observed held-out signed-ratio bias and ATS truncation behavior in the point-estimate gate itself."
        if not materially_changed
        else "The c1 correction must be interpreted from the corrected rerun only; prior point-estimate values should be treated as stale."
    )
    result = {
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "module_audit": module_audit(),
        "nonuniform_assertion": assertion,
        "old": old,
        "new": new,
        "comparisons": comparisons,
        "reconciliation": rec,
        "integrity_note": integrity_note,
    }
    rows_csv = build_rows_csv(old_snap, new_snap)
    rows_csv.to_csv(results_dir / "gate_reclassification_fixed_integration_F_rows.csv", index=False)
    write_json(results_dir / "gate_reclassification_fixed_integration_F.json", result)
    (results_dir / "gate_reclassification_fixed_integration_F.md").write_text(make_report(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    print(f"Reclassification fixed integration gate: {result['verdict']} - {result['verdict_reason']}")
    print(
        "Fixed clinical sensitivity/specificity/kappa: "
        f"{result['new']['clinical']['fixed']['sensitivity']:.3f}/"
        f"{result['new']['clinical']['fixed']['specificity']:.3f}/"
        f"{result['new']['clinical']['fixed']['cohens_kappa']:.3f}"
    )
    print(f"Signed ratio median: {result['new']['clinical']['errors_obstructed_recomputed']['ratio_signed']['median']:.4f}")
    print(f"Policy baseline borderline FN: {result['new']['policy']['baseline']['borderline_fn']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
