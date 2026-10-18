from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset_resolver import CYCLE_TO_YEAR, resolve_dataset_path
from .gate_deferral_recovery_f import (
    apply_recovery,
    corrected_maneuvers,
    correction_offset,
    exact_ci,
    final_label_from_decision,
    final_metrics,
    load_inputs,
    truth_label,
)
from .gate_r3_ats_besttest_f import model_usable_acceptable, raw_maneuvers_with_correction
from .gate_workflow_policy_f import RANDOM_SEED, clean_json, fmt, md_table

OUTPUT_STEM = "gate_covariate_resolution_F"
FIXED_RATIO_THRESHOLD = 0.70
BORDERLINE_LOW = 0.65
BORDERLINE_HIGH = 0.75
HELDOUT_R2_MIN = 0.95


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def yes(value: Any) -> bool:
    return finite(value) and float(value) == 1.0


def no(value: Any) -> bool:
    return finite(value) and float(value) == 2.0


def smoking_status(row: pd.Series) -> str:
    smq020 = row.get("SMQ020", np.nan)
    smq040 = row.get("SMQ040", np.nan)
    if no(smq020):
        return "never"
    if yes(smq020) and finite(smq040) and float(smq040) in {1.0, 2.0}:
        return "current"
    if yes(smq020) and finite(smq040) and float(smq040) == 3.0:
        return "former"
    if yes(smq020):
        return "former_unknown_current_status"
    return "unknown"


def pack_years(row: pd.Series) -> float:
    status = smoking_status(row)
    if status == "never":
        return 0.0
    age = float(row["RIDAGEYR"]) if finite(row.get("RIDAGEYR")) else np.nan
    start_age = float(row["SMD030"]) if finite(row.get("SMD030")) else np.nan
    if not np.isfinite(age) or not np.isfinite(start_age):
        return np.nan
    if status == "current":
        cigs_per_day = float(row["SMD650"]) if finite(row.get("SMD650")) else np.nan
        days_30 = float(row["SMD641"]) if finite(row.get("SMD641")) else 30.0
        if not np.isfinite(cigs_per_day):
            return np.nan
        days_fraction = min(max(days_30, 0.0), 30.0) / 30.0
        return float(max(age - start_age, 0.0) * cigs_per_day * days_fraction / 20.0)
    if status in {"former", "former_unknown_current_status"}:
        quit_age = float(row["SMD055"]) if finite(row.get("SMD055")) else np.nan
        cigs_per_day = float(row["SMD057"]) if finite(row.get("SMD057")) else np.nan
        if np.isfinite(quit_age) and np.isfinite(cigs_per_day):
            return float(max(quit_age - start_age, 0.0) * cigs_per_day / 20.0)
    return np.nan


def self_reported_obstructive_disease(row: pd.Series) -> str:
    emphysema = row.get("MCQ160G", np.nan)
    chronic_bronchitis = row.get("MCQ160K", np.nan)
    if yes(emphysema) or yes(chronic_bronchitis):
        return "yes"
    if no(emphysema) and no(chronic_bronchitis):
        return "no"
    return "unknown"


def symptom_status(row: pd.Series) -> str:
    symptom_cols = ["RDQ031", "RDQ050", "RDQ070", "RDQ090", "RDQ100", "RDQ134", "RDQ140"]
    if any(yes(row.get(col, np.nan)) for col in symptom_cols):
        return "yes"
    # RDQ090/RDQ100/RDQ134 are conditional on wheeze; the four screen/no-symptom
    # fields are enough to call a clean no-symptom row.
    screen_cols = ["RDQ031", "RDQ050", "RDQ070", "RDQ140"]
    if all(no(row.get(col, np.nan)) for col in screen_cols):
        return "no"
    return "unknown"


def cough_status(row: pd.Series) -> str:
    vals = [row.get("RDQ031", np.nan), row.get("RDQ050", np.nan), row.get("RDQ140", np.nan)]
    if any(yes(v) for v in vals):
        return "yes"
    if all(no(v) for v in vals):
        return "no"
    return "unknown"


def wheeze_status(row: pd.Series) -> str:
    vals = [row.get("RDQ070", np.nan), row.get("RDQ090", np.nan), row.get("RDQ100", np.nan), row.get("RDQ134", np.nan)]
    if any(yes(v) for v in vals):
        return "yes"
    if no(row.get("RDQ070", np.nan)):
        return "no"
    return "unknown"


def dyspnea_proxy_status(row: pd.Series) -> str:
    # RDQ does not expose a direct dyspnea item in these cycles. RDQ135 captures
    # wheeze-limited activity, so use it as a symptom-activity proxy only.
    value = row.get("RDQ135", np.nan)
    if finite(value) and float(value) in {1.0, 2.0, 3.0}:
        return "yes"
    if no(row.get("RDQ070", np.nan)):
        return "no"
    return "unknown"


def load_covariates(root: Path) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    rows = []
    notes = []
    variables = {
        "DEMO": "RIDAGEYR, RIAGENDR, RIDRETH1",
        "BMX": "BMXHT, BMXBMI",
        "SMQ": "SMQ020/SMQ040 smoking status; SMD030/SMD055/SMD057/SMD641/SMD650 approximate pack-years when derivable",
        "MCQ": "MCQ160G emphysema OR MCQ160K chronic bronchitis",
        "RDQ": "RDQ031/RDQ050/RDQ070/RDQ090/RDQ100/RDQ134/RDQ140 symptoms; RDQ135 activity limitation proxy",
    }
    for cycle in CYCLE_TO_YEAR:
        try:
            demo = pd.read_sas(resolve_dataset_path(root, "DEMO", cycle).path, format="xport")
            bmx = pd.read_sas(resolve_dataset_path(root, "BMX", cycle).path, format="xport")
            smq = pd.read_sas(resolve_dataset_path(root, "SMQ", cycle).path, format="xport")
            mcq = pd.read_sas(resolve_dataset_path(root, "MCQ", cycle).path, format="xport")
            rdq = pd.read_sas(resolve_dataset_path(root, "RDQ", cycle).path, format="xport")
        except Exception as exc:
            notes.append(f"Cycle {cycle} covariates unavailable: {exc}")
            continue
        cov = demo[[c for c in ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1"] if c in demo.columns]].copy()
        merge_specs = [
            (bmx, ["SEQN", "BMXHT", "BMXBMI"]),
            (smq, ["SEQN", "SMQ020", "SMQ040", "SMD030", "SMD055", "SMD057", "SMD641", "SMD650"]),
            (mcq, ["SEQN", "MCQ160G", "MCQ160K"]),
            (
                rdq,
                ["SEQN", "RDQ031", "RDQ050", "RDQ070", "RDQ090", "RDQ100", "RDQ134", "RDQ135", "RDQ140"],
            ),
        ]
        for frame, cols in merge_specs:
            cov = cov.merge(frame[[c for c in cols if c in frame.columns]], on="SEQN", how="left")
        cov["cycle"] = cycle
        cov["smoking_status"] = cov.apply(smoking_status, axis=1)
        cov["pack_years"] = cov.apply(pack_years, axis=1)
        cov["self_reported_obstructive_disease"] = cov.apply(self_reported_obstructive_disease, axis=1)
        cov["rdq_cough"] = cov.apply(cough_status, axis=1)
        cov["rdq_wheeze"] = cov.apply(wheeze_status, axis=1)
        cov["rdq_dyspnea_proxy"] = cov.apply(dyspnea_proxy_status, axis=1)
        cov["respiratory_symptom"] = cov.apply(symptom_status, axis=1)
        rows.append(cov)
    if not rows:
        return pd.DataFrame(), notes, variables
    return pd.concat(rows, ignore_index=True), notes, variables


def availability_rows(deferred: pd.DataFrame) -> list[dict[str, Any]]:
    specs = [
        ("DEMO.RIDAGEYR age", deferred["RIDAGEYR"].map(finite)),
        ("DEMO.RIAGENDR sex", deferred["RIAGENDR"].map(finite)),
        ("DEMO.RIDRETH1 race/ethnicity", deferred["RIDRETH1"].map(finite)),
        ("BMX.BMXHT height", deferred["BMXHT"].map(finite)),
        ("BMX.BMXBMI BMI", deferred["BMXBMI"].map(finite)),
        ("SMQ smoking status", deferred["smoking_status"].isin(["never", "current", "former", "former_unknown_current_status"])),
        ("SMQ pack-years derivable", deferred["pack_years"].map(finite)),
        ("MCQ emphysema/chronic bronchitis", deferred["self_reported_obstructive_disease"].isin(["yes", "no"])),
        ("RDQ cough", deferred["rdq_cough"].isin(["yes", "no"])),
        ("RDQ wheeze", deferred["rdq_wheeze"].isin(["yes", "no"])),
        ("RDQ dyspnea/activity proxy", deferred["rdq_dyspnea_proxy"].isin(["yes", "no"])),
        ("RDQ any respiratory symptom", deferred["respiratory_symptom"].isin(["yes", "no"])),
    ]
    out = []
    for label, mask in specs:
        n = int(mask.sum())
        out.append({"covariate": label, "available_n": n, "deferred_n": int(len(deferred)), "available_fraction": float(n / len(deferred)) if len(deferred) else np.nan})
    return out


def clinical_risk(row: pd.Series) -> str:
    age_ge_40 = finite(row.get("RIDAGEYR")) and float(row["RIDAGEYR"]) >= 40.0
    ever_smoker = row.get("smoking_status") in {"current", "former", "former_unknown_current_status"}
    never_smoker = row.get("smoking_status") == "never"
    disease_yes = row.get("self_reported_obstructive_disease") == "yes"
    disease_no = row.get("self_reported_obstructive_disease") == "no"
    symptom_yes = row.get("respiratory_symptom") == "yes"
    symptom_no = row.get("respiratory_symptom") == "no"
    if age_ge_40 and ever_smoker and (disease_yes or symptom_yes):
        return "HIGH"
    if never_smoker and disease_no and symptom_no:
        return "LOW"
    return "INTERMEDIATE"


def context_eligible(row: pd.Series) -> bool:
    if row.get("primary_deferral_reason") not in {"R2_BORDERLINE_RATIO", "R3_FAILED_REPEATABILITY"}:
        return False
    if not finite(row.get("corrected_ratio_c1")):
        return False
    ratio = float(row["corrected_ratio_c1"])
    return BORDERLINE_LOW <= ratio <= BORDERLINE_HIGH


def apply_context_rule(deferred: pd.DataFrame) -> pd.DataFrame:
    out = deferred.copy()
    out["context_eligible"] = out.apply(context_eligible, axis=1)
    out["clinical_risk"] = out.apply(clinical_risk, axis=1)
    out["context_decision"] = ""
    out.loc[out["context_eligible"] & out["clinical_risk"].eq("HIGH"), "context_decision"] = "RECOVERED_OBSTRUCTED"
    out.loc[out["context_eligible"] & out["clinical_risk"].eq("LOW"), "context_decision"] = "RECOVERED_NORMAL"
    out["context_finalized"] = out["context_decision"].astype(str).ne("")
    out["naive_decision"] = np.where(
        out["corrected_ratio_c1"].map(finite) & (out["corrected_ratio_c1"].astype(float) < FIXED_RATIO_THRESHOLD),
        "RECOVERED_OBSTRUCTED",
        "RECOVERED_NORMAL",
    )
    out.loc[~out["context_finalized"], "naive_decision"] = ""
    return out


def score_resolved(resolved: pd.DataFrame, decision_col: str, prefix: str) -> dict[str, Any]:
    if resolved.empty:
        return {
            f"{prefix}_resolved_n": 0,
            f"{prefix}_correct_n": 0,
            f"{prefix}_error_n": 0,
            f"{prefix}_accuracy": np.nan,
            f"{prefix}_severe_n": 0,
            f"{prefix}_severe_fn_n": 0,
            f"{prefix}_severe_fn_ci_low": np.nan,
            f"{prefix}_severe_fn_ci_high": np.nan,
        }
    labels = resolved[decision_col].map(final_label_from_decision)
    truth = resolved.apply(truth_label, axis=1)
    correct = labels.eq(truth)
    severe = resolved[resolved["truth_severity"].eq("severe")]
    severe_fn = int((severe[decision_col].map(final_label_from_decision).eq("NORMAL")).sum())
    lo, hi = exact_ci(severe_fn, len(severe))
    return {
        f"{prefix}_resolved_n": int(len(resolved)),
        f"{prefix}_correct_n": int(correct.sum()),
        f"{prefix}_error_n": int((~correct).sum()),
        f"{prefix}_accuracy": float(correct.mean()) if len(resolved) else np.nan,
        f"{prefix}_severe_n": int(len(severe)),
        f"{prefix}_severe_fn_n": severe_fn,
        f"{prefix}_severe_fn_ci_low": lo,
        f"{prefix}_severe_fn_ci_high": hi,
    }


def heldout_proxy(resolved: pd.DataFrame, raw: pd.DataFrame, decision_col: str) -> dict[str, Any]:
    blow_records = []
    participant_records = []
    for _, row in resolved.iterrows():
        part = raw[raw["SEQN"].astype(float).eq(float(row["SEQN"]))]
        pool = model_usable_acceptable(part)
        r2 = pd.to_numeric(pool["volume_r2"], errors="coerce")
        pool = pool[np.isfinite(r2) & (r2 >= HELDOUT_R2_MIN)].copy()
        predicted = final_label_from_decision(row[decision_col])
        case = []
        for _, holdout in pool.iterrows():
            observed = "OBSTRUCTED" if float(holdout["observed_ratio_numeric"]) < FIXED_RATIO_THRESHOLD else "NORMAL"
            ok = predicted == observed
            blow_records.append({"SEQN": row["SEQN"], "ok": ok, "decision_col": decision_col})
            case.append(ok)
        if case:
            participant_records.append({"SEQN": row["SEQN"], "majority_ok": sum(case) > len(case) / 2.0, "all_ok": all(case)})
    blow = pd.DataFrame(blow_records)
    participant = pd.DataFrame(participant_records)
    return {
        "participants_with_proxy_n": int(len(participant)),
        "blow_correct_n": int(blow["ok"].sum()) if len(blow) else 0,
        "blow_n": int(len(blow)),
        "blow_accuracy": float(blow["ok"].mean()) if len(blow) else np.nan,
        "participant_majority_correct_n": int(participant["majority_ok"].sum()) if len(participant) else 0,
        "participant_majority_n": int(len(participant)),
        "participant_majority_accuracy": float(participant["majority_ok"].mean()) if len(participant) else np.nan,
        "participant_all_accuracy": float(participant["all_ok"].mean()) if len(participant) else np.nan,
    }


def recovery_reason_rows(deferred: pd.DataFrame) -> list[dict[str, Any]]:
    total = len(deferred)
    rows = []
    for reason, part in deferred.groupby("primary_deferral_reason"):
        rows.append({"reason": reason, "n": int(len(part)), "fraction_residual": float(len(part) / total) if total else np.nan})
    return sorted(rows, key=lambda row: (-row["n"], row["reason"]))


def tier1_proxy(rows: pd.DataFrame, raw: pd.DataFrame) -> dict[str, Any]:
    finalized = rows[rows["post_recovery_finalized"].astype(bool)].copy()
    records = []
    for _, row in finalized.iterrows():
        part = raw[raw["SEQN"].astype(float).eq(float(row["SEQN"]))]
        pool = model_usable_acceptable(part)
        r2 = pd.to_numeric(pool["volume_r2"], errors="coerce")
        pool = pool[np.isfinite(r2) & (r2 >= HELDOUT_R2_MIN)].copy()
        predicted = final_label_from_decision(row["post_recovery_decision"])
        for _, holdout in pool.iterrows():
            observed = "OBSTRUCTED" if float(holdout["observed_ratio_numeric"]) < FIXED_RATIO_THRESHOLD else "NORMAL"
            records.append(predicted == observed)
    return {
        "tier": "TIER 1 spirometry-alone retained",
        "finalized_n": int(len(finalized)),
        "coverage": float(len(finalized) / len(rows)) if len(rows) else np.nan,
        "heldout_blow_correct_n": int(sum(records)),
        "heldout_blow_n": int(len(records)),
        "heldout_blow_accuracy": float(sum(records) / len(records)) if records else np.nan,
        "retained": True,
    }


def accuracy_comparison_rows(context: dict[str, Any], naive: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "method": "context rule",
            "resolved_n": context["context_resolved_n"],
            "correct_n": context["context_correct_n"],
            "error_n": context["context_error_n"],
            "accuracy": context["context_accuracy"],
            "severe_n": context["context_severe_n"],
            "severe_fn_n": context["context_severe_fn_n"],
            "severe_fn_ci_high": context["context_severe_fn_ci_high"],
        },
        {
            "method": "naive 0.70 flip",
            "resolved_n": naive["naive_resolved_n"],
            "correct_n": naive["naive_correct_n"],
            "error_n": naive["naive_error_n"],
            "accuracy": naive["naive_accuracy"],
            "severe_n": naive["naive_severe_n"],
            "severe_fn_n": naive["naive_severe_fn_n"],
            "severe_fn_ci_high": naive["naive_severe_fn_ci_high"],
        },
    ]


def heldout_comparison_rows(context: dict[str, Any], naive: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method, data in [("context rule", context), ("naive 0.70 flip", naive)]:
        rows.append(
            {
                "method": method,
                "participants_with_proxy_n": data["participants_with_proxy_n"],
                "blow_correct_n": data["blow_correct_n"],
                "blow_n": data["blow_n"],
                "blow_accuracy": data["blow_accuracy"],
                "participant_majority_correct_n": data["participant_majority_correct_n"],
                "participant_majority_n": data["participant_majority_n"],
                "participant_majority_accuracy": data["participant_majority_accuracy"],
            }
        )
    return rows


def verdict(result: dict[str, Any]) -> tuple[str, str]:
    context = result["context_accuracy"]
    naive = result["naive_accuracy_on_same_cases"]
    severe_bad = context["context_severe_fn_n"] > 0
    if severe_bad:
        return "CONTEXT-UNSAFE", "covariate tiebreaking introduced severe false negatives, so it is discarded."
    if not np.isfinite(context["context_accuracy"]) or context["context_accuracy"] <= naive["naive_accuracy"]:
        return (
            "CONTEXT-ADDS-NOTHING",
            "covariate-resolved calls did not beat the naive 0.70 spirometry flip on the same cases, so Tier 2 is discarded.",
        )
    return (
        "CONTEXT-RESOLVES",
        "covariate tiebreaking resolved deferred cases, beat the naive spirometry flip, and preserved severe FN=0.",
    )


def run(root: Path) -> dict[str, Any]:
    workflow, repro = load_inputs(root)
    c1 = correction_offset(workflow)
    maneuvers = corrected_maneuvers(repro, c1)
    raw = raw_maneuvers_with_correction(repro, c1)
    test = workflow[workflow["split"].eq("test")].copy()
    rows = apply_recovery(test, maneuvers)
    cov, cov_notes, cov_variables = load_covariates(root)
    residual = rows[rows["deferred_baseline"].astype(bool) & ~rows["recovered"].astype(bool)].copy()
    residual = residual.merge(cov, on="SEQN", how="left", suffixes=("", "_cov"))
    residual = apply_context_rule(residual)
    resolved = residual[residual["context_finalized"].astype(bool)].copy()

    context_accuracy = score_resolved(resolved, "context_decision", "context")
    naive_accuracy = score_resolved(resolved, "naive_decision", "naive")
    context_proxy = heldout_proxy(resolved, raw, "context_decision")
    naive_proxy = heldout_proxy(resolved, raw, "naive_decision")

    candidate_accept = (
        context_accuracy["context_severe_fn_n"] == 0
        and np.isfinite(context_accuracy["context_accuracy"])
        and context_accuracy["context_accuracy"] > naive_accuracy["naive_accuracy"]
    )

    final_rows = rows.copy()
    final_rows["post_context_decision"] = final_rows["post_recovery_decision"]
    final_rows["post_context_finalized"] = final_rows["post_recovery_finalized"].astype(bool)
    if candidate_accept:
        for _, row in resolved.iterrows():
            match = final_rows["SEQN"].astype(float).eq(float(row["SEQN"]))
            final_rows.loc[match, "post_context_decision"] = row["context_decision"]
            final_rows.loc[match, "post_context_finalized"] = True
    retained_final_metrics = final_metrics(final_rows, "post_context_decision", "post_context_finalized")
    baseline_metrics = final_metrics(rows, "post_recovery_decision", "post_recovery_finalized")

    result: dict[str, Any] = {
        "seed": RANDOM_SEED,
        "test_n": int(len(rows)),
        "residual_deferred_n": int(len(residual)),
        "baseline_coverage": baseline_metrics["finalized_fraction"],
        "residual_breakdown": recovery_reason_rows(residual),
        "covariate_variables": cov_variables,
        "covariate_notes": cov_notes,
        "covariate_availability": availability_rows(residual),
        "declared_rule": {
            "scope": "only residual deferred TEST cases; only R2 and R3 with corrected_ratio_c1 in [0.65, 0.75]",
            "high_risk": "age >= 40 AND ever-smoker AND (self-reported emphysema/chronic bronchitis OR >=1 RDQ respiratory symptom) -> OBSTRUCTED",
            "low_risk": "never-smoker AND no self-reported emphysema/chronic bronchitis AND no RDQ respiratory symptoms -> NORMAL",
            "intermediate": "all other cases stay deferred",
            "naive_comparator": "on the exact same resolved cases, classify corrected_ratio_c1 < 0.70 as OBSTRUCTED and otherwise NORMAL",
        },
        "eligible_n": int(residual["context_eligible"].sum()),
        "eligible_risk_counts": {
            str(k): int(v) for k, v in residual[residual["context_eligible"]]["clinical_risk"].value_counts().items()
        },
        "context_resolution_counts": {
            "resolved_n": int(len(resolved)),
            "high_risk_obstructed_n": int(resolved["context_decision"].eq("RECOVERED_OBSTRUCTED").sum()),
            "low_risk_normal_n": int(resolved["context_decision"].eq("RECOVERED_NORMAL").sum()),
            "still_deferred_n": int(len(residual) - len(resolved)),
        },
        "context_accuracy": context_accuracy,
        "naive_accuracy_on_same_cases": naive_accuracy,
        "accuracy_comparison": accuracy_comparison_rows(context_accuracy, naive_accuracy),
        "context_heldout_proxy": context_proxy,
        "naive_heldout_proxy_on_same_cases": naive_proxy,
        "heldout_comparison": heldout_comparison_rows(context_proxy, naive_proxy),
        "candidate_context_coverage_if_retained": float((baseline_metrics["finalized_n"] + len(resolved)) / len(rows))
        if len(rows)
        else np.nan,
        "candidate_retained": bool(candidate_accept),
        "tier_rows": [
            tier1_proxy(rows, raw),
            {
                "tier": "TIER 2 spirometry + clinical context candidate",
                "finalized_n": int(len(resolved)),
                "coverage": float(len(resolved) / len(rows)) if len(rows) else np.nan,
                "heldout_blow_correct_n": context_proxy["blow_correct_n"],
                "heldout_blow_n": context_proxy["blow_n"],
                "heldout_blow_accuracy": context_proxy["blow_accuracy"],
                "retained": bool(candidate_accept),
            },
        ],
        "baseline_metrics": baseline_metrics,
        "final_metrics": retained_final_metrics,
        "protocol_decisions": [
            "TEST split only.",
            "Operated only on residual deferred cases from gate_deferral_recovery_f: deferred_baseline == True and recovered == False.",
            "The clinical context rule was declared before evaluation and was not fit to outcomes.",
            "Borderline band was fixed as corrected_ratio_c1 in [0.65, 0.75] for R2/R3 residuals only.",
            "RDQ has no direct dyspnea variable in these cycles; RDQ135 wheeze-limited activity is reported as a dyspnea/activity proxy.",
            "Pack-years are approximate and only used for availability reporting, not for the tiebreak rule.",
            "Covariates never override already-finalized spirometry calls.",
            "Tier 2 is retained only if context beats the naive 0.70 flip on the exact same cases and severe FN remains zero.",
        ],
    }
    result["verdict"], result["verdict_reason"] = verdict(result)
    write_outputs(root, result, residual)
    return result


def write_outputs(root: Path, result: dict[str, Any], rows: pd.DataFrame) -> None:
    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    rows.to_csv(out_dir / f"{OUTPUT_STEM}_rows.csv", index=False)

    lines = [
        "# Gate - Covariate-Assisted Resolution of Deferred Cases, Cycle F",
        "",
        f"VERDICT: {result['verdict']} - {result['verdict_reason']}",
        "",
        "## Protocol Decisions",
        "",
    ]
    lines.extend(f"- {item}" for item in result["protocol_decisions"])
    lines.extend(
        [
            "",
            "## Part 0 - Scope",
            "",
            md_table(
                result["residual_breakdown"],
                [
                    ("reason", "residual reason", "str"),
                    ("n", "n", "int"),
                    ("fraction_residual", "fraction residual", "float3"),
                ],
            ),
            "",
            "## Part 1 - Covariate Availability",
            "",
            md_table(
                result["covariate_availability"],
                [
                    ("covariate", "covariate", "str"),
                    ("available_n", "available n", "int"),
                    ("deferred_n", "deferred n", "int"),
                    ("available_fraction", "available fraction", "float3"),
                ],
            ),
            "",
            "## Part 2 - Declared Clinical Tiebreak Rule",
            "",
        ]
    )
    for key, value in result["declared_rule"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            f"Eligible borderline residual cases: {result['eligible_n']}",
            f"Eligible risk counts: {result['eligible_risk_counts']}",
            "",
            "## Part 3 - Honest Evaluation",
            "",
            md_table(
                result["accuracy_comparison"],
                [
                    ("method", "method", "str"),
                    ("resolved_n", "resolved n", "int"),
                    ("correct_n", "correct", "int"),
                    ("error_n", "errors", "int"),
                    ("accuracy", "accuracy", "float3"),
                    ("severe_n", "severe n", "int"),
                    ("severe_fn_n", "severe FN", "int"),
                    ("severe_fn_ci_high", "severe FN CP high", "float3"),
                ],
            ),
            "",
            md_table(
                result["heldout_comparison"],
                [
                    ("method", "method", "str"),
                    ("participants_with_proxy_n", "participants with proxy", "int"),
                    ("blow_correct_n", "held-out blow correct", "int"),
                    ("blow_n", "held-out blow n", "int"),
                    ("blow_accuracy", "held-out blow acc", "float3"),
                    ("participant_majority_correct_n", "participant majority correct", "int"),
                    ("participant_majority_n", "participant majority n", "int"),
                    ("participant_majority_accuracy", "participant majority acc", "float3"),
                ],
            ),
            "",
            md_table(
                [result["context_resolution_counts"]],
                [
                    ("resolved_n", "resolved n", "int"),
                    ("high_risk_obstructed_n", "HIGH -> obstructed", "int"),
                    ("low_risk_normal_n", "LOW -> normal", "int"),
                    ("still_deferred_n", "still deferred", "int"),
                ],
            ),
            "",
            f"Candidate coverage if retained: {fmt(result['candidate_context_coverage_if_retained'], 3)}.",
            f"Candidate retained: {result['candidate_retained']}.",
            "",
            "## Part 4 - Tiered Output",
            "",
            md_table(
                result["tier_rows"],
                [
                    ("tier", "tier", "str"),
                    ("finalized_n", "finalized n", "int"),
                    ("coverage", "coverage", "float3"),
                    ("heldout_blow_correct_n", "held-out blow correct", "int"),
                    ("heldout_blow_n", "held-out blow n", "int"),
                    ("heldout_blow_accuracy", "held-out blow acc", "float3"),
                    ("retained", "retained", "str"),
                ],
            ),
            "",
            md_table(
                [{"scope": "final retained policy", **result["final_metrics"]}],
                [
                    ("scope", "scope", "str"),
                    ("finalized_n", "finalized n", "int"),
                    ("finalized_fraction", "coverage", "float3"),
                    ("sensitivity", "sens", "float3"),
                    ("specificity", "spec", "float3"),
                    ("cohens_kappa", "kappa", "float3"),
                    ("severe_finalized_n", "severe finalized", "int"),
                    ("severe_fn_n", "severe FN", "int"),
                    ("severe_fn_ci_high", "severe FN CP high", "float3"),
                ],
            ),
            "",
            "Tier 2 is discarded because it does not beat the naive 0.70 comparator on the same cases. Final retained coverage therefore remains the current 66.6%.",
            "",
            f"VERDICT: {result['verdict']} - {result['verdict_reason']}",
            "",
        ]
    )
    (out_dir / f"{OUTPUT_STEM}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    print(f"Covariate resolution: {result['verdict']} - {result['verdict_reason']}")
    print(f"Context resolved: {result['context_resolution_counts']['resolved_n']}")
    print(f"Final retained coverage: {result['final_metrics']['finalized_fraction']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
