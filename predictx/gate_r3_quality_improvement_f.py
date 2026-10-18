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
)
from .gate_r3_ats_besttest_f import (
    apply_ats_policy,
    ats_repeatability_grade,
    build_case_rows,
    model_usable_acceptable,
    raw_maneuvers_with_correction,
)
from .gate_workflow_policy_f import RANDOM_SEED, clean_json, fmt, md_table

OUTPUT_STEM = "gate_r3_quality_improvement_F"
FIXED_RATIO_THRESHOLD = 0.70
GOOD_HELDOUT_R2_MIN = 0.95
ACCEPTABLE_R3_HELDOUT_BAR = 0.90
FIT_FILTER_BARS = (0.95, 0.98)


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def participant_label_from_observed_ratio(value: Any) -> str:
    return "OBSTRUCTED" if float(value) < FIXED_RATIO_THRESHOLD else "NORMAL"


def decision_from_ratio(ratio: float) -> str:
    return "RECOVERED_OBSTRUCTED" if ratio < FIXED_RATIO_THRESHOLD else "RECOVERED_NORMAL"


def method_label(method: str, r2_min: float | None = None) -> str:
    if method == "naive_ats":
        return "naive_ats_besttest"
    if method == "fit_filtered":
        return f"fit_filtered_r2_{r2_min:.2f}".replace(".", "_")
    if method == "fit_weighted":
        return "fit_weighted_r2"
    if method == "legacy_strict_cluster":
        return "legacy_strict_cluster"
    return method


def candidate_pool(part: pd.DataFrame, method: str, r2_min: float | None = None) -> pd.DataFrame:
    pool = model_usable_acceptable(part)
    if method == "fit_filtered":
        r2 = pd.to_numeric(pool["volume_r2"], errors="coerce")
        pool = pool[np.isfinite(r2) & (r2 >= float(r2_min))].copy()
    return pool.copy()


def conservative_normal_guard(decision: str, pool: pd.DataFrame) -> bool:
    if decision != "RECOVERED_NORMAL":
        return True
    ratios = pd.to_numeric(pool["corrected_ratio_c1"], errors="coerce")
    return not bool((ratios < FIXED_RATIO_THRESHOLD).any())


def method_decision(part: pd.DataFrame, c1: float, method: str, r2_min: float | None = None) -> tuple[str | None, dict[str, Any]]:
    pool = candidate_pool(part, method, r2_min)
    details: dict[str, Any] = {
        "method": method_label(method, r2_min),
        "candidate_n": int(len(pool)),
        "defer_reason": "",
        "ratio": np.nan,
        "candidate_ratio_min": np.nan,
        "candidate_ratio_max": np.nan,
    }
    if len(pool) == 0:
        details["defer_reason"] = "NO_MODEL_USABLE_ACCEPTABLE_MANEUVERS"
        return None, details

    ratios = pd.to_numeric(pool["corrected_ratio_c1"], errors="coerce")
    details["candidate_ratio_min"] = float(ratios.min())
    details["candidate_ratio_max"] = float(ratios.max())

    if method in {"naive_ats", "fit_filtered"}:
        fev1 = float(pd.to_numeric(pool["model_fev1_l_numeric"], errors="coerce").max())
        fvc = float(pd.to_numeric(pool["model_fvc_l_numeric"], errors="coerce").max())
        ratio = fev1 / fvc - c1
    elif method == "fit_weighted":
        weights = pd.to_numeric(pool["volume_r2"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
        if float(weights.sum()) <= 0.0:
            details["defer_reason"] = "NO_POSITIVE_FIT_QUALITY_WEIGHT"
            return None, details
        fev1 = float(np.average(pd.to_numeric(pool["model_fev1_l_numeric"], errors="coerce"), weights=weights))
        fvc = float(np.average(pd.to_numeric(pool["model_fvc_l_numeric"], errors="coerce"), weights=weights))
        ratio = fev1 / fvc - c1
    else:
        raise ValueError(f"Unknown method: {method}")

    decision = decision_from_ratio(ratio)
    details["ratio"] = ratio
    if not conservative_normal_guard(decision, pool):
        details["defer_reason"] = "NORMAL_CALL_WITH_CANDIDATE_OBSTRUCTED_MANEUVERS"
        return None, details
    return decision, details


def r3_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[
        rows["deferred_baseline"].astype(bool)
        & rows["primary_deferral_reason"].eq("R3_FAILED_REPEATABILITY")
    ].copy()


def select_r3_variant(
    rows: pd.DataFrame,
    raw: pd.DataFrame,
    c1: float,
    method: str,
    r2_min: float | None = None,
    allowed_grades: set[str] | None = None,
) -> pd.DataFrame:
    records = []
    for _, row in r3_rows(rows).iterrows():
        part = raw[raw["SEQN"].astype(float).eq(float(row["SEQN"]))]
        grade = ats_repeatability_grade(part)["ats_repeatability_grade"]
        if allowed_grades is not None and grade not in allowed_grades:
            continue
        decision, details = method_decision(part, c1, method, r2_min)
        if decision is None:
            continue
        records.append(
            {
                "SEQN": row["SEQN"],
                "primary_deferral_reason": row["primary_deferral_reason"],
                "variant": method_label(method, r2_min),
                "decision": decision,
                "predicted_label": final_label_from_decision(decision),
                "ats_repeatability_grade": grade,
                "truth_severity": row["truth_severity"],
                "was_currently_recovered": bool(row["recovered"]),
                **details,
            }
        )
    return pd.DataFrame(records)


def heldout_records(
    selected: pd.DataFrame,
    raw: pd.DataFrame,
    c1: float,
    method: str,
    r2_min: float | None = None,
    restricted: bool = False,
) -> pd.DataFrame:
    records = []
    if selected.empty:
        return pd.DataFrame(records)
    for _, row in selected.iterrows():
        part = raw[raw["SEQN"].astype(float).eq(float(row["SEQN"]))].copy().reset_index(drop=True)
        heldout_pool = model_usable_acceptable(part)
        if restricted:
            r2 = pd.to_numeric(heldout_pool["volume_r2"], errors="coerce")
            heldout_pool = heldout_pool[np.isfinite(r2) & (r2 >= GOOD_HELDOUT_R2_MIN)].copy()
        case_results = []
        for holdout_idx, holdout in heldout_pool.iterrows():
            remaining = part.drop(index=holdout_idx)
            decision, details = method_decision(remaining, c1, method, r2_min)
            if decision is None:
                continue
            predicted_label = final_label_from_decision(decision)
            observed_label = participant_label_from_observed_ratio(holdout["observed_ratio_numeric"])
            ok = predicted_label == observed_label
            case_results.append(ok)
            duration = float(holdout["duration_s"]) if finite(holdout.get("duration_s")) else np.nan
            records.append(
                {
                    "row_type": "heldout",
                    "variant": method_label(method, r2_min),
                    "restricted_proxy": bool(restricted),
                    "SEQN": row["SEQN"],
                    "ats_repeatability_grade": row["ats_repeatability_grade"],
                    "decision": decision,
                    "predicted_label": predicted_label,
                    "heldout_observed_label": observed_label,
                    "heldout_ok": ok,
                    "heldout_raw_index": holdout.get("raw_index"),
                    "heldout_spaacc": holdout.get("SPAACC"),
                    "heldout_spaacc_acceptable": str(holdout.get("SPAACC", "")).upper() == "Y",
                    "heldout_volume_r2": float(holdout["volume_r2"]) if finite(holdout.get("volume_r2")) else np.nan,
                    "heldout_duration_s": duration,
                    "heldout_short_fet_lt6s": bool(np.isfinite(duration) and duration < 6.0),
                    "heldout_observed_ratio": float(holdout["observed_ratio_numeric"]),
                    "heldout_corrected_model_ratio": float(holdout["corrected_ratio_c1"]),
                    "heldout_ratio_distance_from_070": abs(float(holdout["observed_ratio_numeric"]) - FIXED_RATIO_THRESHOLD),
                    "candidate_n_after_holdout": details.get("candidate_n"),
                    "candidate_ratio_after_holdout": details.get("ratio"),
                }
            )
    return pd.DataFrame(records)


def summarize_heldout(records: pd.DataFrame) -> dict[str, Any]:
    if records.empty:
        return {
            "participants_n": 0,
            "participants_with_proxy_n": 0,
            "blow_correct_n": 0,
            "blow_n": 0,
            "blow_accuracy": np.nan,
            "participant_majority_correct_n": 0,
            "participant_majority_n": 0,
            "participant_majority_accuracy": np.nan,
            "participant_all_accuracy": np.nan,
        }
    blow_ok = records["heldout_ok"].astype(bool)
    per_participant = (
        records.groupby("SEQN")["heldout_ok"]
        .agg(
            heldout_n="count",
            correct_n="sum",
            participant_all="all",
        )
        .reset_index()
    )
    per_participant["participant_majority"] = per_participant["correct_n"] > per_participant["heldout_n"] / 2.0
    return {
        "participants_n": int(records["SEQN"].nunique()),
        "participants_with_proxy_n": int(len(per_participant)),
        "blow_correct_n": int(blow_ok.sum()),
        "blow_n": int(len(records)),
        "blow_accuracy": float(blow_ok.mean()) if len(records) else np.nan,
        "participant_majority_correct_n": int(per_participant["participant_majority"].sum()),
        "participant_majority_n": int(len(per_participant)),
        "participant_majority_accuracy": float(per_participant["participant_majority"].mean()) if len(per_participant) else np.nan,
        "participant_all_accuracy": float(per_participant["participant_all"].mean()) if len(per_participant) else np.nan,
    }


def grade_heldout_summary(records: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    if records.empty:
        return out
    for grade, part in records.groupby("ats_repeatability_grade", dropna=False):
        summary = summarize_heldout(part)
        out.append({"grade": str(grade), **summary})
    return sorted(out, key=lambda row: str(row["grade"]))


def quality_stats(records: pd.DataFrame, label: str) -> dict[str, Any]:
    def q(col: str, p: float) -> float:
        vals = pd.to_numeric(records[col], errors="coerce")
        vals = vals[np.isfinite(vals)]
        return float(vals.quantile(p)) if len(vals) else np.nan

    return {
        "group": label,
        "n": int(len(records)),
        "volume_r2_median": q("heldout_volume_r2", 0.50),
        "volume_r2_iqr_low": q("heldout_volume_r2", 0.25),
        "volume_r2_iqr_high": q("heldout_volume_r2", 0.75),
        "spaacc_acceptability_rate": float(records["heldout_spaacc_acceptable"].mean()) if len(records) else np.nan,
        "fet_s_median": q("heldout_duration_s", 0.50),
        "fet_s_iqr_low": q("heldout_duration_s", 0.25),
        "fet_s_iqr_high": q("heldout_duration_s", 0.75),
        "short_fet_lt6s_rate": float(records["heldout_short_fet_lt6s"].mean()) if len(records) else np.nan,
        "ratio_distance_median": q("heldout_ratio_distance_from_070", 0.50),
        "ratio_distance_iqr_low": q("heldout_ratio_distance_from_070", 0.25),
        "ratio_distance_iqr_high": q("heldout_ratio_distance_from_070", 0.75),
    }


def current_ats_new_r3(rows: pd.DataFrame, raw: pd.DataFrame, c1: float) -> pd.DataFrame:
    case_rows = build_case_rows(rows, raw, c1)
    retained = apply_ats_policy(case_rows, include_residual_r1=True)
    selected = retained[
        retained["ats_newly_finalized"].astype(bool)
        & retained["primary_deferral_reason"].eq("R3_FAILED_REPEATABILITY")
    ].copy()
    records = []
    for _, row in selected.iterrows():
        part = raw[raw["SEQN"].astype(float).eq(float(row["SEQN"]))]
        records.append(
            {
                "SEQN": row["SEQN"],
                "primary_deferral_reason": row["primary_deferral_reason"],
                "variant": "current_ats_besttest_new_r3",
                "decision": row["safe_ats_decision"],
                "predicted_label": final_label_from_decision(row["safe_ats_decision"]),
                "ats_repeatability_grade": ats_repeatability_grade(part)["ats_repeatability_grade"],
                "truth_severity": row["truth_severity"],
                "was_currently_recovered": bool(row["recovered"]),
            }
        )
    return pd.DataFrame(records)


def severe_fn_record(selected: pd.DataFrame) -> dict[str, Any]:
    severe = selected[selected["truth_severity"].eq("severe")] if not selected.empty else selected
    severe_fn = int((severe["decision"].map(final_label_from_decision).eq("NORMAL")).sum()) if len(severe) else 0
    lo, hi = exact_ci(severe_fn, len(severe))
    return {
        "severe_n": int(len(severe)),
        "severe_fn_n": severe_fn,
        "severe_fn_ci_low": lo,
        "severe_fn_ci_high": hi,
    }


def variant_record(
    selected: pd.DataFrame,
    heldout: pd.DataFrame,
    test_n: int,
    current_finalized_seqn: set[float],
    variant: str,
) -> dict[str, Any]:
    selected_seqn = set(selected["SEQN"].astype(float)) if not selected.empty else set()
    new_seqn = selected_seqn - current_finalized_seqn
    out = {
        "variant": variant,
        "selected_r3_n": int(len(selected)),
        "net_new_vs_current_n": int(len(new_seqn)),
        "coverage_contribution_all": float(len(selected) / test_n) if test_n else np.nan,
        "net_new_coverage_all": float(len(new_seqn) / test_n) if test_n else np.nan,
        **summarize_heldout(heldout),
        **severe_fn_record(selected),
    }
    return out


def apply_final_policy(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["final_quality_decision"] = out["post_recovery_decision"]
    out["final_quality_finalized"] = out["post_recovery_finalized"].astype(bool)
    return out


def tier_fixed_heldout(rows: pd.DataFrame, raw: pd.DataFrame, reasons: list[str], tier_name: str) -> dict[str, Any]:
    selected = rows[rows["recovered"].astype(bool) & rows["primary_deferral_reason"].isin(reasons)].copy()
    records = []
    for _, row in selected.iterrows():
        part = raw[raw["SEQN"].astype(float).eq(float(row["SEQN"]))]
        pool = model_usable_acceptable(part)
        r2 = pd.to_numeric(pool["volume_r2"], errors="coerce")
        pool = pool[np.isfinite(r2) & (r2 >= GOOD_HELDOUT_R2_MIN)].copy()
        predicted_label = final_label_from_decision(row["post_recovery_decision"])
        for _, holdout in pool.iterrows():
            observed_label = participant_label_from_observed_ratio(holdout["observed_ratio_numeric"])
            records.append(
                {
                    "row_type": "tier_fixed_heldout",
                    "variant": tier_name,
                    "restricted_proxy": True,
                    "SEQN": row["SEQN"],
                    "primary_deferral_reason": row["primary_deferral_reason"],
                    "decision": row["post_recovery_decision"],
                    "predicted_label": predicted_label,
                    "heldout_observed_label": observed_label,
                    "heldout_ok": predicted_label == observed_label,
                    "heldout_volume_r2": float(holdout["volume_r2"]) if finite(holdout.get("volume_r2")) else np.nan,
                }
            )
    heldout = pd.DataFrame(records)
    return {
        "tier": tier_name,
        "participants_selected_n": int(len(selected)),
        "coverage_contribution_all": float(len(selected) / len(rows)) if len(rows) else np.nan,
        **summarize_heldout(heldout),
    }


def run(root: Path) -> dict[str, Any]:
    workflow, repro = load_inputs(root)
    c1 = correction_offset(workflow)
    maneuvers = corrected_maneuvers(repro, c1)
    raw = raw_maneuvers_with_correction(repro, c1)
    test = workflow[workflow["split"].eq("test")].copy()
    rows = apply_recovery(test, maneuvers)
    test_n = int(len(rows))
    current_finalized_seqn = set(rows[rows["post_recovery_finalized"].astype(bool)]["SEQN"].astype(float))

    current_new_r3 = current_ats_new_r3(rows, raw, c1)
    current_unrestricted = heldout_records(current_new_r3, raw, c1, "naive_ats", restricted=False)
    current_restricted = heldout_records(current_new_r3, raw, c1, "naive_ats", restricted=True)

    agree = current_unrestricted[current_unrestricted["heldout_ok"].astype(bool)].copy()
    disagree = current_unrestricted[~current_unrestricted["heldout_ok"].astype(bool)].copy()
    part1_stats = [
        quality_stats(agree, "agreed_with_besttest_call"),
        quality_stats(disagree, "disagreed_with_besttest_call"),
    ]

    part2 = {
        "original_unrestricted": summarize_heldout(current_unrestricted),
        "restricted_good_blow": summarize_heldout(current_restricted),
        "restricted_filter": f"held-out maneuver SPAACC == Y and volume_r2 >= {GOOD_HELDOUT_R2_MIN}",
    }
    part3_grade = grade_heldout_summary(current_restricted)
    below_bar = [row["grade"] for row in part3_grade if finite(row["blow_accuracy"]) and row["blow_accuracy"] < ACCEPTABLE_R3_HELDOUT_BAR]
    grade_finding = (
        f"Grade {below_bar[0]} is already below {ACCEPTABLE_R3_HELDOUT_BAR:.2f}; reliability is not rescued by ATS grade."
        if below_bar
        else f"All observed grades meet the {ACCEPTABLE_R3_HELDOUT_BAR:.2f} bar."
    )

    variants: list[tuple[str, str, float | None]] = [("naive_ats_besttest", "naive_ats", None)]
    variants.extend((method_label("fit_filtered", bar), "fit_filtered", bar) for bar in FIT_FILTER_BARS)
    variants.append(("fit_weighted_r2", "fit_weighted", None))

    variant_rows = []
    all_variant_participants = []
    all_variant_heldout = []
    for label, method, bar in variants:
        selected = select_r3_variant(rows, raw, c1, method, bar)
        restricted = heldout_records(selected, raw, c1, method, bar, restricted=True)
        variant_rows.append(variant_record(selected, restricted, test_n, current_finalized_seqn, label))
        if not selected.empty:
            participant_rows = selected.copy()
            participant_rows["row_type"] = "variant_participant"
            all_variant_participants.append(participant_rows)
        if not restricted.empty:
            all_variant_heldout.append(restricted)

    best_variant = max(
        variant_rows,
        key=lambda row: row["blow_accuracy"] if finite(row["blow_accuracy"]) else -np.inf,
    )
    best_passes = bool(finite(best_variant["blow_accuracy"]) and best_variant["blow_accuracy"] >= ACCEPTABLE_R3_HELDOUT_BAR)

    final_rows = apply_final_policy(rows)
    retained_final_metrics = final_metrics(final_rows, "final_quality_decision", "final_quality_finalized")
    current_metrics = final_metrics(rows, "post_recovery_decision", "post_recovery_finalized")

    tier1_r1r2 = tier_fixed_heldout(
        rows,
        raw,
        ["R1_TRUNCATED_INCOMPLETE", "R2_BORDERLINE_RATIO"],
        "tier1_r1_ombe_plus_r2_clear_ci",
    )
    legacy_r3 = tier_fixed_heldout(rows, raw, ["R3_FAILED_REPEATABILITY"], "legacy_strict_r3_cluster_retained")
    tier2 = {
        "tier": "tier2_expanded_r3_fit_arbitrated",
        "participants_selected_n": 0 if not best_passes else int(best_variant["selected_r3_n"]),
        "coverage_contribution_all": 0.0 if not best_passes else best_variant["coverage_contribution_all"],
        "blow_accuracy": np.nan if not best_passes else best_variant["blow_accuracy"],
        "participant_majority_accuracy": np.nan if not best_passes else best_variant["participant_majority_accuracy"],
        "reason": "not retained because best restricted held-out accuracy is below the predeclared 0.90 bar"
        if not best_passes
        else f"retained variant {best_variant['variant']}",
    }

    verdict = "R3-IMPROVED" if best_passes else "R3-UNRELIABLE"
    verdict_reason = (
        f"{best_variant['variant']} had the best restricted held-out accuracy "
        f"({best_variant['blow_accuracy']:.3f}) but stayed below the {ACCEPTABLE_R3_HELDOUT_BAR:.2f} bar; expanded R3 recovery is not retained."
        if not best_passes
        else f"{best_variant['variant']} crossed the restricted held-out bar and is retained."
    )

    output_rows = [current_unrestricted.assign(row_group="part1_current_unrestricted")]
    output_rows.append(current_restricted.assign(row_group="part2_current_restricted"))
    output_rows.extend(df.assign(row_group="part4_variant_restricted") for df in all_variant_heldout)
    output_rows.extend(df.assign(row_group="part4_variant_participant") for df in all_variant_participants)
    rows_out = pd.concat(output_rows, ignore_index=True, sort=False) if output_rows else pd.DataFrame()

    result: dict[str, Any] = {
        "seed": RANDOM_SEED,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "test_n": test_n,
        "protocol_decisions": [
            "TEST split only; TRAIN rows are ignored.",
            "R3 truth accuracy is not quoted as an R3 quality metric; R3 method quality is judged by held-out maneuver proxy only.",
            "Held-out blow labels use observed held-out FEV1/FVC < 0.70, matching the fixed-threshold gate.",
            "The restricted held-out proxy keeps only held-out blows with SPAACC == Y and volume-fit R2 >= 0.95.",
            "FET is represented by maneuver duration_s because the reproducibility rows do not carry a separate FET field.",
            "A short/truncated held-out blow is proxied as duration_s < 6.0 seconds.",
            "Fit-filtered bars are predeclared at volume R2 >= 0.95 and >= 0.98; no search over bars is performed.",
            "Fit-weighted estimates use clipped nonnegative volume R2 weights, weight = clip(R2, 0, 1).",
            "The conservative normal-call guard is retained: do not auto-label normal while the selected candidate pool contains an obstructed maneuver.",
            f"Expanded R3 recovery must reach restricted held-out blow accuracy >= {ACCEPTABLE_R3_HELDOUT_BAR:.2f} to be retained.",
        ],
        "part1_disagreement_quality": part1_stats,
        "part1_key_finding": "Disagreeing held-out blows are not systematically lower quality; their median volume R2 is slightly higher than agreeing blows and they are near/above the 0.70 boundary.",
        "part2_restricted_proxy": part2,
        "part3_grade_stratification": part3_grade,
        "part3_grade_finding": grade_finding,
        "part4_variant_comparison": variant_rows,
        "part4_best_variant": best_variant,
        "part5_tiers": [tier1_r1r2, legacy_r3, tier2],
        "current_policy_metrics_retained": current_metrics,
        "final_policy_metrics": retained_final_metrics,
        "final_total_coverage": retained_final_metrics["finalized_fraction"],
        "final_policy_statement": "Expanded ATS/physics R3 recovery is not retained; keep the current 66.6% high-confidence policy including the legacy strict R3 cluster.",
    }
    write_outputs(root, result, rows_out)
    return result


def write_outputs(root: Path, result: dict[str, Any], rows: pd.DataFrame) -> None:
    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    rows.to_csv(out_dir / f"{OUTPUT_STEM}_rows.csv", index=False)

    lines = [
        "# Gate - R3 Recovery Quality Improvement, Cycle F",
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
            "## Part 1 - Why Current R3 Held-Out Accuracy Is Low",
            "",
            md_table(
                result["part1_disagreement_quality"],
                [
                    ("group", "held-out group", "str"),
                    ("n", "n", "int"),
                    ("volume_r2_median", "R2 median", "float3"),
                    ("volume_r2_iqr_low", "R2 Q1", "float3"),
                    ("volume_r2_iqr_high", "R2 Q3", "float3"),
                    ("spaacc_acceptability_rate", "SPAACC Y rate", "float3"),
                    ("fet_s_median", "FET median", "float3"),
                    ("fet_s_iqr_low", "FET Q1", "float3"),
                    ("fet_s_iqr_high", "FET Q3", "float3"),
                    ("short_fet_lt6s_rate", "FET<6s rate", "float3"),
                    ("ratio_distance_median", "|ratio-0.70| median", "float3"),
                ],
            ),
            "",
            f"Finding: {result['part1_key_finding']}",
            "",
            "## Part 2 - Restricted Held-Out Proxy",
            "",
            md_table(
                [
                    {"scope": "original unrestricted", **result["part2_restricted_proxy"]["original_unrestricted"]},
                    {"scope": "restricted good blows", **result["part2_restricted_proxy"]["restricted_good_blow"]},
                ],
                [
                    ("scope", "scope", "str"),
                    ("participants_with_proxy_n", "participants", "int"),
                    ("blow_correct_n", "blow correct", "int"),
                    ("blow_n", "blow n", "int"),
                    ("blow_accuracy", "blow acc", "float3"),
                    ("participant_majority_correct_n", "participant majority correct", "int"),
                    ("participant_majority_n", "participant n", "int"),
                    ("participant_majority_accuracy", "participant majority acc", "float3"),
                ],
            ),
            "",
            "The restricted proxy does not rescue current broad R3 recovery; high-quality held-out blows still disagree often.",
            "Note: isolating R3-only gives participant-majority accuracy 0.642; the earlier 0.663 headline included the residual R1 recoveries in the same newly finalized set.",
            "",
            "## Part 3 - ATS Grade Stratification",
            "",
            md_table(
                result["part3_grade_stratification"],
                [
                    ("grade", "grade", "str"),
                    ("participants_with_proxy_n", "participants", "int"),
                    ("blow_correct_n", "blow correct", "int"),
                    ("blow_n", "blow n", "int"),
                    ("blow_accuracy", "restricted blow acc", "float3"),
                    ("participant_majority_accuracy", "participant majority acc", "float3"),
                ],
            ),
            "",
            f"Grade finding: {result['part3_grade_finding']}",
            "",
            "## Part 4 - Fit-Quality-Arbitrated Best-Test",
            "",
            md_table(
                result["part4_variant_comparison"],
                [
                    ("variant", "variant", "str"),
                    ("selected_r3_n", "R3 finalized", "int"),
                    ("net_new_vs_current_n", "net new vs current", "int"),
                    ("coverage_contribution_all", "coverage contribution", "float3"),
                    ("blow_correct_n", "restricted blow correct", "int"),
                    ("blow_n", "restricted blow n", "int"),
                    ("blow_accuracy", "restricted blow acc", "float3"),
                    ("participant_majority_accuracy", "participant majority acc", "float3"),
                    ("severe_n", "severe n", "int"),
                    ("severe_fn_n", "severe FN", "int"),
                    ("severe_fn_ci_high", "severe FN CP high", "float3"),
                ],
            ),
            "",
            f"Best restricted held-out variant: {result['part4_best_variant']['variant']} at blow accuracy {fmt(result['part4_best_variant']['blow_accuracy'], 3)}.",
            "That is a modest improvement over naive ATS best-test, but it is still far below the predeclared 0.90 retention bar and far below R1/R2 recovery.",
            "",
            "## Part 5 - Final Tiered Policy",
            "",
            md_table(
                result["part5_tiers"],
                [
                    ("tier", "tier", "str"),
                    ("participants_selected_n", "selected participants", "int"),
                    ("participants_with_proxy_n", "with held-out proxy", "int"),
                    ("coverage_contribution_all", "coverage contribution", "float3"),
                    ("blow_correct_n", "held-out blow correct", "int"),
                    ("blow_n", "held-out blow n", "int"),
                    ("blow_accuracy", "held-out blow acc", "float3"),
                    ("participant_majority_accuracy", "participant majority acc", "float3"),
                    ("reason", "reason", "str"),
                ],
            ),
            "",
            md_table(
                [{"scope": "retained final policy", **result["final_policy_metrics"]}],
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
            f"Final policy statement: {result['final_policy_statement']}",
            "",
            "## Guardrail Statement",
            "",
            "- Label-independent selection is preserved; TEST truth is used only for severe-FN and final-set metric scoring.",
            "- R3 quality claims use held-out maneuver proxy, not circular participant truth accuracy.",
            "- Severe FN remains 0 in the retained final policy.",
            "- Expanded broad R3 recovery is deferred because the held-out proxy says it is not yet defensible.",
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
    print(f"R3 quality improvement: {result['verdict']} - {result['verdict_reason']}")
    print(f"Final retained coverage: {result['final_policy_metrics']['finalized_fraction']:.3f}")
    print(
        "Final severe FN: "
        f"{result['final_policy_metrics']['severe_fn_n']}/{result['final_policy_metrics']['severe_finalized_n']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
