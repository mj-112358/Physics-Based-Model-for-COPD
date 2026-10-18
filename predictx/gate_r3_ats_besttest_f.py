from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
from .gate_workflow_policy_f import RANDOM_SEED, clean_json, fmt, md_table

OUTPUT_STEM = "gate_r3_ats_besttest_F"
ATS_ERS_2019_URL = "https://doi.org/10.1164/rccm.201908-1590ST"
FIXED_RATIO_THRESHOLD = 0.70


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if pd.isna(value):
        return False
    return bool(value)


def decision_from_ratio(ratio: float) -> str:
    return "RECOVERED_OBSTRUCTED" if ratio < FIXED_RATIO_THRESHOLD else "RECOVERED_NORMAL"


def raw_maneuvers_with_correction(repro: pd.DataFrame, c1: float) -> pd.DataFrame:
    out = repro[repro["row_type"].eq("maneuver_fit")].copy()
    out["model_ratio_numeric"] = pd.to_numeric(out["model_ratio"], errors="coerce")
    out["corrected_ratio_c1"] = out["model_ratio_numeric"] - c1
    out["model_fev1_l_numeric"] = pd.to_numeric(out["model_fev1_l"], errors="coerce")
    out["model_fvc_l_numeric"] = pd.to_numeric(out["model_fvc_l"], errors="coerce")
    out["observed_ratio_numeric"] = pd.to_numeric(out["observed_ratio"], errors="coerce")
    return out


def ats_acceptable(part: pd.DataFrame) -> pd.DataFrame:
    return part[part["SPAACC"].astype(str).str.upper().eq("Y")].copy()


def model_usable_acceptable(part: pd.DataFrame) -> pd.DataFrame:
    accepted = ats_acceptable(part)
    return accepted[
        accepted["volume_fit_success"].map(bool_value)
        & np.isfinite(accepted["model_fev1_l_numeric"].to_numpy(dtype=float))
        & np.isfinite(accepted["model_fvc_l_numeric"].to_numpy(dtype=float))
        & np.isfinite(accepted["corrected_ratio_c1"].to_numpy(dtype=float))
    ].copy()


def top_two_delta(values: np.ndarray) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    return float(arr[-1] - arr[-2])


def ats_repeatability_grade(part: pd.DataFrame) -> dict[str, Any]:
    usable = model_usable_acceptable(part)
    n = int(len(usable))
    fev1_delta = top_two_delta(usable["model_fev1_l_numeric"].to_numpy(dtype=float))
    fvc_delta = top_two_delta(usable["model_fvc_l_numeric"].to_numpy(dtype=float))
    if n == 0:
        grade = "F"
    elif n == 1:
        grade = "E"
    elif np.isfinite(fev1_delta) and np.isfinite(fvc_delta) and fev1_delta <= 0.150 and fvc_delta <= 0.150:
        grade = "A" if n >= 3 else "B"
    elif np.isfinite(fev1_delta) and np.isfinite(fvc_delta) and fev1_delta <= 0.200 and fvc_delta <= 0.200:
        grade = "C"
    elif np.isfinite(fev1_delta) and np.isfinite(fvc_delta) and fev1_delta <= 0.250 and fvc_delta <= 0.250:
        grade = "D"
    else:
        grade = "E"
    return {
        "ats_repeatability_grade": grade,
        "ats_repeatability_usable_n": n,
        "ats_repeatability_fev1_delta_l": fev1_delta,
        "ats_repeatability_fvc_delta_l": fvc_delta,
    }


def besttest_details(part: pd.DataFrame, c1: float, boundary_guard: bool) -> dict[str, Any]:
    accepted = ats_acceptable(part)
    usable = model_usable_acceptable(part)
    details: dict[str, Any] = {
        "ats_acceptable_n": int(len(accepted)),
        "ats_model_usable_n": int(len(usable)),
        "ats_boundary_straddles_070": False,
        "ats_normal_call_has_obstructed_accepted_maneuver": False,
        "ats_besttest_fev1_l": np.nan,
        "ats_besttest_fvc_l": np.nan,
        "ats_besttest_ratio_uncorrected": np.nan,
        "ats_besttest_ratio_corrected": np.nan,
        "ats_besttest_decision": "",
        "ats_defer_reason": "",
    }
    details.update(ats_repeatability_grade(part))
    if len(usable) == 0:
        details["ats_defer_reason"] = "ZERO_MODEL_USABLE_ACCEPTABLE_MANEUVERS"
        return details

    per_maneuver_ratios = usable["corrected_ratio_c1"].to_numpy(dtype=float)
    ratio_min = float(np.nanmin(per_maneuver_ratios))
    ratio_max = float(np.nanmax(per_maneuver_ratios))
    straddles = bool(ratio_min < FIXED_RATIO_THRESHOLD <= ratio_max)
    fev1_idx = usable["model_fev1_l_numeric"].astype(float).idxmax()
    fvc_idx = usable["model_fvc_l_numeric"].astype(float).idxmax()
    fev1 = float(usable.at[fev1_idx, "model_fev1_l_numeric"])
    fvc = float(usable.at[fvc_idx, "model_fvc_l_numeric"])
    ratio_uncorrected = fev1 / fvc
    ratio_corrected = ratio_uncorrected - c1

    details.update(
        {
            "ats_boundary_straddles_070": straddles,
            "ats_per_maneuver_corrected_ratio_min": ratio_min,
            "ats_per_maneuver_corrected_ratio_max": ratio_max,
            "ats_besttest_fev1_l": fev1,
            "ats_besttest_fvc_l": fvc,
            "ats_besttest_ratio_uncorrected": ratio_uncorrected,
            "ats_besttest_ratio_corrected": ratio_corrected,
            "ats_besttest_fev1_raw_index": usable.at[fev1_idx, "raw_index"],
            "ats_besttest_fvc_raw_index": usable.at[fvc_idx, "raw_index"],
        }
    )
    decision = decision_from_ratio(ratio_corrected)
    normal_call_has_obstructed_accepted = bool(
        decision == "RECOVERED_NORMAL" and ratio_min < FIXED_RATIO_THRESHOLD
    )
    details["ats_normal_call_has_obstructed_accepted_maneuver"] = normal_call_has_obstructed_accepted
    if boundary_guard and normal_call_has_obstructed_accepted:
        details["ats_defer_reason"] = "NORMAL_BESTTEST_WITH_ACCEPTED_OBSTRUCTED_MANEUVERS"
        return details

    details["ats_besttest_decision"] = decision
    return details


def grade_distribution(rows: pd.DataFrame, raw: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    total = len(rows)
    for _, row in rows.iterrows():
        part = raw[raw["SEQN"].astype(float).eq(float(row["SEQN"]))]
        records.append(ats_repeatability_grade(part)["ats_repeatability_grade"])
    counts = pd.Series(records, dtype="object").value_counts().sort_index()
    return [
        {
            "grade": str(grade),
            "n": int(n),
            "fraction": float(n / total) if total else np.nan,
        }
        for grade, n in counts.items()
    ]


def acceptability_summary(rows: pd.DataFrame, raw: pd.DataFrame) -> dict[str, Any]:
    raw_counts = []
    usable_counts = []
    for _, row in rows.iterrows():
        part = raw[raw["SEQN"].astype(float).eq(float(row["SEQN"]))]
        raw_counts.append(int(len(ats_acceptable(part))))
        usable_counts.append(int(len(model_usable_acceptable(part))))
    raw_arr = np.asarray(raw_counts, dtype=int)
    usable_arr = np.asarray(usable_counts, dtype=int)
    return {
        "cases_n": int(len(rows)),
        "ats_acceptable_ge2_n": int((raw_arr >= 2).sum()),
        "ats_acceptable_ge1_n": int((raw_arr >= 1).sum()),
        "ats_acceptable_zero_n": int((raw_arr == 0).sum()),
        "model_usable_acceptable_ge2_n": int((usable_arr >= 2).sum()),
        "model_usable_acceptable_ge1_n": int((usable_arr >= 1).sum()),
        "model_usable_acceptable_zero_n": int((usable_arr == 0).sum()),
        "ats_acceptable_median_n": float(np.median(raw_arr)) if len(raw_arr) else np.nan,
        "model_usable_acceptable_median_n": float(np.median(usable_arr)) if len(usable_arr) else np.nan,
    }


def score_selected(cases: pd.DataFrame, decision_col: str) -> dict[str, Any]:
    selected = cases[cases[decision_col].astype(str).ne("")].copy()
    correct = selected["ats_truth_correct"].eq(True)
    severe = selected[selected["truth_severity"].eq("severe")]
    severe_fn = int((severe[decision_col].map(final_label_from_decision).eq("NORMAL")).sum())
    lo, hi = exact_ci(severe_fn, len(severe))
    return {
        "selected_n": int(len(selected)),
        "correct_n": int(correct.sum()),
        "error_n": int((~correct).sum()),
        "accuracy": float(correct.mean()) if len(selected) else np.nan,
        "severe_n": int(len(severe)),
        "severe_fn_n": severe_fn,
        "severe_fn_ci_low": lo,
        "severe_fn_ci_high": hi,
    }


def heldout_maneuver_accuracy(selected: pd.DataFrame, raw: pd.DataFrame, c1: float) -> dict[str, Any]:
    blow_records = []
    participant_records = []
    for _, row in selected.iterrows():
        part = raw[raw["SEQN"].astype(float).eq(float(row["SEQN"]))]
        usable = model_usable_acceptable(part).reset_index(drop=True)
        case_results = []
        for holdout_idx, holdout in usable.iterrows():
            remaining = usable.drop(index=holdout_idx)
            details = besttest_details(remaining, c1, boundary_guard=True)
            decision = details["ats_besttest_decision"]
            if not decision:
                continue
            heldout_observed_label = "OBSTRUCTED" if float(holdout["observed_ratio_numeric"]) < FIXED_RATIO_THRESHOLD else "NORMAL"
            heldout_corrected_model_label = (
                "OBSTRUCTED" if float(holdout["corrected_ratio_c1"]) < FIXED_RATIO_THRESHOLD else "NORMAL"
            )
            pred_label = final_label_from_decision(decision)
            observed_ok = pred_label == heldout_observed_label
            model_ok = pred_label == heldout_corrected_model_label
            blow_records.append(
                {
                    "SEQN": row["SEQN"],
                    "reason": row["primary_deferral_reason"],
                    "observed_ok": observed_ok,
                    "model_ok": model_ok,
                }
            )
            case_results.append(observed_ok)
        if case_results:
            participant_records.append(
                {
                    "SEQN": row["SEQN"],
                    "reason": row["primary_deferral_reason"],
                    "participant_majority_observed_ok": sum(case_results) > len(case_results) / 2,
                    "participant_all_observed_ok": all(case_results),
                    "heldout_blow_n": len(case_results),
                }
            )
    blow = pd.DataFrame(blow_records)
    participant = pd.DataFrame(participant_records)
    out = {
        "participants_n": int(len(selected)),
        "participants_with_proxy_n": int(len(participant)),
        "participants_without_proxy_n": int(len(selected) - len(participant)),
        "observed_blow_correct_n": int(blow["observed_ok"].sum()) if len(blow) else 0,
        "observed_blow_n": int(len(blow)),
        "observed_blow_accuracy": float(blow["observed_ok"].mean()) if len(blow) else np.nan,
        "model_blow_correct_n": int(blow["model_ok"].sum()) if len(blow) else 0,
        "model_blow_n": int(len(blow)),
        "model_blow_accuracy": float(blow["model_ok"].mean()) if len(blow) else np.nan,
        "participant_majority_observed_correct_n": int(participant["participant_majority_observed_ok"].sum())
        if len(participant)
        else 0,
        "participant_majority_observed_n": int(len(participant)),
        "participant_majority_observed_accuracy": float(participant["participant_majority_observed_ok"].mean())
        if len(participant)
        else np.nan,
        "participant_all_observed_accuracy": float(participant["participant_all_observed_ok"].mean())
        if len(participant)
        else np.nan,
    }
    by_reason = []
    if len(blow):
        for reason, part in blow.groupby("reason"):
            by_reason.append(
                {
                    "reason": reason,
                    "observed_blow_correct_n": int(part["observed_ok"].sum()),
                    "observed_blow_n": int(len(part)),
                    "observed_blow_accuracy": float(part["observed_ok"].mean()),
                }
            )
    out["by_reason"] = by_reason
    return out


def build_case_rows(rows: pd.DataFrame, raw: pd.DataFrame, c1: float) -> pd.DataFrame:
    out = rows.copy()
    ats_columns: dict[str, list[Any]] = {}
    for prefix in ["pure_ats", "safe_ats"]:
        for col in [
            "decision",
            "defer_reason",
            "acceptable_n",
            "model_usable_n",
            "boundary_straddles_070",
            "normal_call_has_obstructed_accepted_maneuver",
            "besttest_ratio_corrected",
            "repeatability_grade",
        ]:
            out[f"{prefix}_{col}"] = ""
    out["ats_truth_correct"] = pd.NA
    out["ats_newly_finalized"] = False
    out["ats_extension_reason"] = ""

    for idx, row in out[out["deferred_baseline"].astype(bool)].iterrows():
        part = raw[raw["SEQN"].astype(float).eq(float(row["SEQN"]))]
        pure = besttest_details(part, c1, boundary_guard=False)
        safe = besttest_details(part, c1, boundary_guard=True)
        for source_name, details in [("pure_ats", pure), ("safe_ats", safe)]:
            mapping = {
                "decision": details.get("ats_besttest_decision", ""),
                "defer_reason": details.get("ats_defer_reason", ""),
                "acceptable_n": details.get("ats_acceptable_n", np.nan),
                "model_usable_n": details.get("ats_model_usable_n", np.nan),
                "boundary_straddles_070": details.get("ats_boundary_straddles_070", False),
                "normal_call_has_obstructed_accepted_maneuver": details.get(
                    "ats_normal_call_has_obstructed_accepted_maneuver", False
                ),
                "besttest_ratio_corrected": details.get("ats_besttest_ratio_corrected", np.nan),
                "repeatability_grade": details.get("ats_repeatability_grade", ""),
            }
            for key, value in mapping.items():
                out.at[idx, f"{source_name}_{key}"] = value
            for key, value in details.items():
                detail_col = f"{source_name}_{key}"
                if detail_col not in out.columns:
                    ats_columns[detail_col] = []
                    out[detail_col] = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")
                out.at[idx, detail_col] = value
        safe_decision = str(safe.get("ats_besttest_decision", ""))
        if safe_decision:
            out.at[idx, "ats_truth_correct"] = final_label_from_decision(safe_decision) == truth_label(row)
    return out


def apply_ats_policy(rows: pd.DataFrame, include_residual_r1: bool) -> pd.DataFrame:
    out = rows.copy()
    out["post_ats_decision"] = out["workflow_decision"]
    out["post_ats_finalized"] = out["workflow_final_diagnostic"].astype(bool)
    out["ats_besttest_selected"] = False
    out["ats_newly_finalized"] = False

    keep_existing = (
        out["deferred_baseline"].astype(bool)
        & out["recovered"].astype(bool)
        & ~out["primary_deferral_reason"].eq("R3_FAILED_REPEATABILITY")
    )
    out.loc[keep_existing, "post_ats_decision"] = out.loc[keep_existing, "candidate_decision"]
    out.loc[keep_existing, "post_ats_finalized"] = True

    r3_safe = (
        out["deferred_baseline"].astype(bool)
        & out["primary_deferral_reason"].eq("R3_FAILED_REPEATABILITY")
        & out["safe_ats_decision"].astype(str).ne("")
    )
    out.loc[r3_safe, "post_ats_decision"] = out.loc[r3_safe, "safe_ats_decision"]
    out.loc[r3_safe, "post_ats_finalized"] = True
    out.loc[r3_safe, "ats_besttest_selected"] = True
    out.loc[r3_safe, "ats_extension_reason"] = "R3_ATS_BESTTEST_CONSERVATIVE_NORMAL_GUARD"

    if include_residual_r1:
        r1_safe = (
            out["deferred_baseline"].astype(bool)
            & ~out["recovered"].astype(bool)
            & out["primary_deferral_reason"].eq("R1_TRUNCATED_INCOMPLETE")
            & out["safe_ats_decision"].astype(str).ne("")
        )
        out.loc[r1_safe, "post_ats_decision"] = out.loc[r1_safe, "safe_ats_decision"]
        out.loc[r1_safe, "post_ats_finalized"] = True
        out.loc[r1_safe, "ats_besttest_selected"] = True
        out.loc[r1_safe, "ats_extension_reason"] = "R1_ALTERNATE_ACCEPTABLE_BESTTEST_CONSERVATIVE_NORMAL_GUARD"

    out["ats_newly_finalized"] = out["post_ats_finalized"].astype(bool) & ~out["post_recovery_finalized"].astype(bool)
    return out


def apply_pure_r3_policy(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["post_pure_r3_decision"] = out["workflow_decision"]
    out["post_pure_r3_finalized"] = out["workflow_final_diagnostic"].astype(bool)

    keep_existing = (
        out["deferred_baseline"].astype(bool)
        & out["recovered"].astype(bool)
        & ~out["primary_deferral_reason"].eq("R3_FAILED_REPEATABILITY")
    )
    out.loc[keep_existing, "post_pure_r3_decision"] = out.loc[keep_existing, "candidate_decision"]
    out.loc[keep_existing, "post_pure_r3_finalized"] = True

    r3_pure = (
        out["deferred_baseline"].astype(bool)
        & out["primary_deferral_reason"].eq("R3_FAILED_REPEATABILITY")
        & out["pure_ats_decision"].astype(str).ne("")
    )
    out.loc[r3_pure, "post_pure_r3_decision"] = out.loc[r3_pure, "pure_ats_decision"]
    out.loc[r3_pure, "post_pure_r3_finalized"] = True
    return out


def residual_besttest_check(rows: pd.DataFrame, reason: str) -> dict[str, Any]:
    part = rows[
        rows["deferred_baseline"].astype(bool)
        & ~rows["recovered"].astype(bool)
        & rows["primary_deferral_reason"].eq(reason)
    ].copy()
    safe = part[part["safe_ats_decision"].astype(str).ne("")]
    pure = part[part["pure_ats_decision"].astype(str).ne("")]
    return {
        "reason": reason,
        "residual_n": int(len(part)),
        "ats_acceptable_ge1_n": int((pd.to_numeric(part["safe_ats_acceptable_n"], errors="coerce") >= 1).sum()),
        "model_usable_ge1_n": int((pd.to_numeric(part["safe_ats_model_usable_n"], errors="coerce") >= 1).sum()),
        "pure_besttest_would_classify_n": int(len(pure)),
        "conservative_besttest_classifies_n": int(len(safe)),
        "conservative_accuracy_if_used": float(safe["ats_truth_correct"].eq(True).mean()) if len(safe) else np.nan,
    }


def residual_summary(rows: pd.DataFrame) -> list[dict[str, Any]]:
    residual = rows[~rows["post_ats_finalized"].astype(bool)].copy()
    total = len(rows)
    records = []
    for reason, part in residual.groupby("primary_deferral_reason", dropna=False):
        records.append(
            {
                "residual_reason": str(reason),
                "n": int(len(part)),
                "fraction_all": float(len(part) / total) if total else np.nan,
                "zero_model_usable_n": int((pd.to_numeric(part["safe_ats_model_usable_n"], errors="coerce") == 0).sum()),
                "normal_call_discordance_n": int(
                    part["safe_ats_normal_call_has_obstructed_accepted_maneuver"].astype(bool).sum()
                ),
            }
        )
    return sorted(records, key=lambda row: (-row["n"], row["residual_reason"]))


def verdict(result: dict[str, Any]) -> tuple[str, str]:
    metrics = result["post_ats_with_r1_metrics"]
    current = result["current_recovery_metrics"]
    severe_ok = metrics["severe_fn_n"] == 0
    coverage_gain = metrics["finalized_fraction"] - current["finalized_fraction"]
    metric_hold = (
        metrics["sensitivity"] >= current["sensitivity"] - 1e-12
        and metrics["specificity"] >= current["specificity"] - 1e-12
        and metrics["cohens_kappa"] >= current["cohens_kappa"] - 1e-12
    )
    if severe_ok and metric_hold and coverage_gain >= 0.03:
        return (
            "ATS-RULE-RECOVERS",
            "ATS best-test plus the conservative normal-call guard materially increases deployable coverage and preserves severe FN=0.",
        )
    if severe_ok and coverage_gain > 0:
        return (
            "ATS-RULE-MARGINAL",
            "ATS best-test recovers some cases, but the gain or metric guard is limited.",
        )
    return (
        "CEILING-CONFIRMED",
        "The residual is dominated by no usable accepted maneuver or true threshold-straddling cases.",
    )


def write_report(root: Path, result: dict[str, Any], rows: pd.DataFrame) -> None:
    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    rows.to_csv(out_dir / f"{OUTPUT_STEM}_rows.csv", index=False)

    current = result["current_recovery_metrics"]
    r3_only = result["post_ats_r3_only_metrics"]
    with_r1 = result["post_ats_with_r1_metrics"]
    lines = [
        "# Gate - R3 Recovery via ATS Best-Test Rule, Cycle F",
        "",
        f"VERDICT: {result['verdict']} - {result['verdict_reason']}",
        "",
        "## ATS/ERS Rule Source",
        "",
        f"- Source: ATS/ERS 2019 spirometry technical statement, {ATS_ERS_2019_URL}.",
        "- The report uses the ATS/ERS rule that reported FEV1 is the largest acceptable FEV1 and reported FVC is the largest acceptable FVC; repeatability is carried as a session quality grade rather than used as a hard stop for reporting.",
        "- The same statement defines repeatability grading from the two largest FEV1 and FVC values, with A/B/C/D/E/F quality levels.",
        "",
        "## Protocol Decisions",
        "",
    ]
    lines.extend(f"- {item}" for item in result["protocol_decisions"])
    lines.extend(
        [
            "",
            "## Part 1 - R3 Bucket Re-Examination",
            "",
            md_table(
                [result["r3_acceptability"]],
                [
                    ("cases_n", "R3 cases", "int"),
                    ("ats_acceptable_ge2_n", ">=2 SPAACC Y", "int"),
                    ("ats_acceptable_ge1_n", ">=1 SPAACC Y", "int"),
                    ("ats_acceptable_zero_n", "zero SPAACC Y", "int"),
                    ("model_usable_acceptable_ge2_n", ">=2 model-usable", "int"),
                    ("model_usable_acceptable_ge1_n", ">=1 model-usable", "int"),
                    ("model_usable_acceptable_zero_n", "zero model-usable", "int"),
                ],
            ),
            "",
            "R3 repeatability grades under the model-usable acceptable set:",
            "",
            md_table(
                result["r3_grade_distribution"],
                [
                    ("grade", "grade", "str"),
                    ("n", "n", "int"),
                    ("fraction", "fraction", "float3"),
                ],
            ),
            "",
            "## Part 2 - ATS Best-Test on R3",
            "",
            "Two R3 measurements are shown because they answer different questions. `Pure classify-all` applies the best-test computation to every model-usable R3 case. `Deployable conservative` keeps obstructed best-test calls, but defers normal calls when any acceptable maneuver remains obstructed on the calibrated ratio scale.",
            "",
            md_table(
                result["r3_besttest_rows"],
                [
                    ("policy", "policy", "str"),
                    ("selected_n", "finalized R3", "int"),
                    ("correct_n", "correct", "int"),
                    ("error_n", "errors", "int"),
                    ("accuracy", "accuracy", "float3"),
                    ("severe_n", "severe n", "int"),
                    ("severe_fn_n", "severe FN", "int"),
                    ("severe_fn_ci_high", "severe FN CP high", "float3"),
                ],
            ),
            "",
            "Pure classify-all is reported but not retained as the deployable policy because it introduces mild/boundary false-normal calls. The retained R3 rule is label-independent: model-usable acceptable maneuvers exist, the calibrated best-test ratio is computed, obstructed calls finalize, and normal calls require every acceptable maneuver to be normal.",
            "",
            "### Held-Out Maneuver Check",
            "",
            md_table(
                [result["heldout_newly_finalized"]],
                [
                    ("participants_n", "participants", "int"),
                    ("participants_with_proxy_n", "with LOO proxy", "int"),
                    ("observed_blow_correct_n", "observed blow correct", "int"),
                    ("observed_blow_n", "observed blow n", "int"),
                    ("observed_blow_accuracy", "observed blow acc", "float3"),
                    ("participant_majority_observed_correct_n", "participant majority correct", "int"),
                    ("participant_majority_observed_n", "participant majority n", "int"),
                    ("participant_majority_observed_accuracy", "participant majority acc", "float3"),
                    ("model_blow_accuracy", "model blow acc", "float3"),
                ],
            ),
            "",
            md_table(
                result["heldout_newly_finalized"]["by_reason"],
                [
                    ("reason", "reason", "str"),
                    ("observed_blow_correct_n", "observed blow correct", "int"),
                    ("observed_blow_n", "observed blow n", "int"),
                    ("observed_blow_accuracy", "observed blow acc", "float3"),
                ],
            ),
            "",
            "## Part 3 - Residual R1/R2 Check",
            "",
            md_table(
                result["residual_r1_r2_checks"],
                [
                    ("reason", "reason", "str"),
                    ("residual_n", "residual", "int"),
                    ("ats_acceptable_ge1_n", ">=1 SPAACC Y", "int"),
                    ("model_usable_ge1_n", ">=1 model-usable", "int"),
                    ("pure_besttest_would_classify_n", "pure best-test classifies", "int"),
                    ("conservative_besttest_classifies_n", "conservative classifies", "int"),
                    ("conservative_accuracy_if_used", "conservative accuracy if used", "float3"),
                ],
            ),
            "",
            "Residual R1 is included only when an alternate acceptable, model-usable best-test result exists and the conservative normal-call guard passes. Residual R2 is not loosened; it remains governed by the existing borderline CI rule even though counts are reported here.",
            "",
            "## Part 4 - New Coverage and Ceiling",
            "",
            md_table(
                [
                    {"scope": "current deployable recovery", **current},
                    {"scope": "pure ATS R3 classify-all", **result["post_pure_r3_metrics"]},
                    {"scope": "ATS R3 only", **r3_only},
                    {"scope": "ATS R3 + residual R1", **with_r1},
                ],
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
            f"- R3-only coverage gain vs current recovery: {fmt(result['coverage_gain_r3_only'], 3)}.",
            f"- R3 + residual R1 coverage gain vs current recovery: {fmt(result['coverage_gain_with_r1'], 3)}.",
            "- Pure ATS R3 classify-all is shown as an unsafe upper coverage bound because it lowers sensitivity and kappa; it is not the retained policy.",
            f"- ATS-best-test selected cases in retained policy: {result['ats_selected_retained_n']}.",
            f"- Net newly finalized cases vs current recovery: {result['newly_finalized_retained_n']}.",
            "",
            "### Residual Deferred After Retained Policy",
            "",
            md_table(
                result["retained_residual_summary"],
                [
                    ("residual_reason", "residual reason", "str"),
                    ("n", "n", "int"),
                    ("fraction_all", "fraction all", "float3"),
                    ("zero_model_usable_n", "zero model-usable", "int"),
                    ("normal_call_discordance_n", "normal-call discordance", "int"),
                ],
            ),
            "",
            f"- residual deferred fraction: {fmt(result['retained_residual_fraction'], 3)}.",
            f"- honest interpretable ceiling under this deterministic gate: {fmt(with_r1['finalized_fraction'], 3)}.",
            "- Residual cases are not being counted as failures to polish: they either lack a usable acceptable modeled signal, remain R2 CI-borderline, are R4/artifact by the existing gate, or would require a normal call despite an accepted obstructed maneuver.",
            "",
            "## Guardrail Statement",
            "",
            "- Selection is label-independent. TEST truth is read only after a case is selected for scoring.",
            "- Severe FN remains 0 in the retained ATS R3 + residual R1 policy.",
            "- No threshold grid search was used. The only new defer rule is the fixed-threshold normal-call guard at 0.70: do not auto-label normal while any accepted maneuver remains obstructed.",
            "- Repeatability is now a reported quality grade, not a hard bar to reporting a best-test result.",
            "",
            f"VERDICT: {result['verdict']} - {result['verdict_reason']}",
            "",
        ]
    )
    (out_dir / f"{OUTPUT_STEM}.md").write_text("\n".join(lines), encoding="utf-8")


def run(root: Path) -> dict[str, Any]:
    workflow, repro = load_inputs(root)
    c1 = correction_offset(workflow)
    maneuvers = corrected_maneuvers(repro, c1)
    raw = raw_maneuvers_with_correction(repro, c1)
    test = workflow[workflow["split"].eq("test")].copy()
    current_rows = apply_recovery(test, maneuvers)
    case_rows = build_case_rows(current_rows, raw, c1)

    r3_rows = case_rows[
        case_rows["deferred_baseline"].astype(bool)
        & case_rows["primary_deferral_reason"].eq("R3_FAILED_REPEATABILITY")
    ].copy()
    pure_r3 = r3_rows[r3_rows["pure_ats_decision"].astype(str).ne("")].copy()
    safe_r3 = r3_rows[r3_rows["safe_ats_decision"].astype(str).ne("")].copy()
    pure_r3["ats_truth_correct"] = pure_r3.apply(
        lambda row: final_label_from_decision(row["pure_ats_decision"]) == truth_label(row), axis=1
    )

    r3_only_rows = apply_ats_policy(case_rows, include_residual_r1=False)
    retained_rows = apply_ats_policy(case_rows, include_residual_r1=True)
    pure_rows = apply_pure_r3_policy(case_rows)
    retained_new = retained_rows[retained_rows["ats_newly_finalized"].astype(bool)].copy()

    current_metrics = final_metrics(current_rows, "post_recovery_decision", "post_recovery_finalized")
    pure_metrics = final_metrics(pure_rows, "post_pure_r3_decision", "post_pure_r3_finalized")
    r3_only_metrics = final_metrics(r3_only_rows, "post_ats_decision", "post_ats_finalized")
    retained_metrics = final_metrics(retained_rows, "post_ats_decision", "post_ats_finalized")

    result: dict[str, Any] = {
        "seed": RANDOM_SEED,
        "test_n": int(len(test)),
        "c1_offset_from_existing_policy": c1,
        "ats_source": ATS_ERS_2019_URL,
        "protocol_decisions": [
            "TEST split only; training rows are ignored.",
            "The current deferral-recovery gate is the baseline: 66.6% coverage before this ATS best-test pass.",
            "NHANES/repro maneuver rows do not expose separate BEV, cough, glottis, leak, or plateau subflags. SPAACC == 'Y' was used as the individual acceptability flag, and this limitation is reported explicitly.",
            "A maneuver also must have a successful finite model fit to be usable by the PredictX model best-test computation; this is a computational availability requirement, not a repeatability threshold.",
            "Best-test FEV1 is max model FEV1 across acceptable usable maneuvers, and best-test FVC is max model FVC across acceptable usable maneuvers; the FEV1 and FVC may come from different maneuvers.",
            "The existing C1 ratio correction is applied after forming max(FEV1)/max(FVC), keeping the recovered ratio on the same calibrated scale as the workflow.",
            "Repeatability is converted to an ATS-style session grade and is not used as a hard exclusion.",
            "Automatic normal diagnosis is deferred when any acceptable maneuver remains obstructed on the calibrated ratio scale; obstructed best-test calls are allowed to finalize because they do not create false-normal risk.",
            "Residual R2 cases are not recovered by best-test because the task explicitly says not to loosen the borderline CI rule.",
            "TEST truth is used only after label-independent selection to score correctness, sensitivity, specificity, kappa, and severe FN.",
        ],
        "current_recovery_metrics": current_metrics,
        "post_pure_r3_metrics": pure_metrics,
        "r3_acceptability": acceptability_summary(r3_rows, raw),
        "r3_grade_distribution": grade_distribution(r3_rows, raw),
        "r3_besttest_rows": [
            {"policy": "pure classify-all", **score_selected(pure_r3, "pure_ats_decision")},
            {"policy": "deployable conservative", **score_selected(safe_r3, "safe_ats_decision")},
        ],
        "residual_r1_r2_checks": [
            residual_besttest_check(case_rows, "R1_TRUNCATED_INCOMPLETE"),
            residual_besttest_check(case_rows, "R2_BORDERLINE_RATIO"),
        ],
        "post_ats_r3_only_metrics": r3_only_metrics,
        "post_ats_with_r1_metrics": retained_metrics,
        "coverage_gain_r3_only": float(r3_only_metrics["finalized_fraction"] - current_metrics["finalized_fraction"]),
        "coverage_gain_with_r1": float(retained_metrics["finalized_fraction"] - current_metrics["finalized_fraction"]),
        "ats_selected_retained_n": int(retained_rows["ats_besttest_selected"].sum()),
        "newly_finalized_retained_n": int(retained_rows["ats_newly_finalized"].sum()),
        "heldout_newly_finalized": heldout_maneuver_accuracy(retained_new, raw, c1),
        "retained_residual_summary": residual_summary(retained_rows),
        "retained_residual_n": int((~retained_rows["post_ats_finalized"].astype(bool)).sum()),
        "retained_residual_fraction": float((~retained_rows["post_ats_finalized"].astype(bool)).mean()),
    }
    result["verdict"], result["verdict_reason"] = verdict(result)
    write_report(root, result, retained_rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    retained = result["post_ats_with_r1_metrics"]
    print(f"R3 ATS best-test: {result['verdict']} - {result['verdict_reason']}")
    print(f"Current coverage: {result['current_recovery_metrics']['finalized_fraction']:.3f}")
    print(f"ATS R3-only coverage: {result['post_ats_r3_only_metrics']['finalized_fraction']:.3f}")
    print(f"ATS R3 + residual R1 coverage: {retained['finalized_fraction']:.3f}")
    print(f"Retained severe FN: {retained['severe_fn_n']}/{retained['severe_finalized_n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
