from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .gate23_f import find_or_unzip_raw, read_raw
from .gate_deferral_recovery_f import (
    apply_recovery,
    corrected_maneuvers,
    correction_offset,
    exact_ci,
    final_label_from_decision,
    final_metrics,
    load_inputs,
)
from .gate_workflow_policy_f import RANDOM_SEED, clean_json, fmt, md_table

OUTPUT_STEM = "gate_iterative_ats_loop_F"
ATS_ERS_2019_URL = "https://doi.org/10.1164/rccm.201908-1590ST"
ATS_REPEATABILITY_L = 0.150
ATS_MAX_ATTEMPTS = 8
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


def clean_grade(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    if pd.isna(value):
        return ""
    return str(value).strip()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def top_two_delta(values: np.ndarray) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    return float(arr[-1] - arr[-2])


def session_grade(usable: pd.DataFrame) -> str:
    n = int(len(usable))
    if n == 0:
        return "F"
    if n == 1:
        return "E"
    fev1_delta = top_two_delta(usable["model_fev1_l_numeric"].to_numpy(dtype=float))
    fvc_delta = top_two_delta(usable["model_fvc_l_numeric"].to_numpy(dtype=float))
    if np.isfinite(fev1_delta) and np.isfinite(fvc_delta):
        if fev1_delta <= ATS_REPEATABILITY_L and fvc_delta <= ATS_REPEATABILITY_L:
            return "A" if n >= 3 else "B"
        if fev1_delta <= 0.200 and fvc_delta <= 0.200:
            return "C"
        if fev1_delta <= 0.250 and fvc_delta <= 0.250:
            return "D"
    return "E"


def count_distribution(counts: pd.Series, include_zero: bool) -> list[dict[str, Any]]:
    buckets = [0, 1, 2, 3, 4, 5, 6, 7] if include_zero else [1, 2, 3, 4, 5, 6, 7]
    total = int(len(counts))
    rows: list[dict[str, Any]] = []
    for bucket in buckets:
        n = int((counts == bucket).sum())
        rows.append({"bucket": str(bucket), "n": n, "fraction": float(n / total) if total else np.nan})
    n8 = int((counts >= 8).sum())
    rows.append({"bucket": "8+", "n": n8, "fraction": float(n8 / total) if total else np.nan})
    return rows


def availability_audit(raw: pd.DataFrame, test_seqns: pd.Series) -> dict[str, Any]:
    seqns = pd.Index(test_seqns.astype(float).unique())
    raw2 = raw.copy()
    raw2["SEQN"] = raw2["SEQN"].astype(float)
    raw2["SPATTYPE_clean"] = raw2["SPATTYPE"].map(clean_grade)
    test_raw = raw2[raw2["SEQN"].isin(seqns)].copy()
    pre = test_raw[test_raw["SPATTYPE_clean"].eq("Pre")]
    post = test_raw[test_raw["SPATTYPE_clean"].str.lower().isin({"bron", "post"})]
    pre_counts = pre.groupby("SEQN").size().reindex(seqns, fill_value=0).astype(int)
    post_counts = post.groupby("SEQN").size().reindex(seqns, fill_value=0).astype(int)
    return {
        "test_n": int(len(seqns)),
        "pre_distribution": count_distribution(pre_counts, include_zero=bool((pre_counts == 0).any())),
        "post_distribution": count_distribution(post_counts, include_zero=True),
        "pre_median": float(pre_counts.median()) if len(pre_counts) else np.nan,
        "post_median": float(post_counts.median()) if len(post_counts) else np.nan,
        "pre_ge3_n": int((pre_counts >= 3).sum()),
        "pre_ge3_fraction": float((pre_counts >= 3).mean()) if len(pre_counts) else np.nan,
        "post_any_n": int((post_counts > 0).sum()),
        "post_any_fraction": float((post_counts > 0).mean()) if len(post_counts) else np.nan,
        "raw_pre_rows": int(len(pre)),
        "raw_post_rows": int(len(post)),
        "post_type_counts": post["SPATTYPE_clean"].value_counts(dropna=False).to_dict(),
    }


def build_attempt_table(raw: pd.DataFrame, repro: pd.DataFrame, c1_offset: float, test_seqns: pd.Series) -> pd.DataFrame:
    seqns = pd.Index(test_seqns.astype(float).unique())
    raw2 = raw.copy()
    raw2["raw_index"] = raw2.index.astype(int)
    raw2["SEQN"] = raw2["SEQN"].astype(float)
    raw2["SPATTYPE_clean"] = raw2["SPATTYPE"].map(clean_grade)
    raw2["SPAACC_clean"] = raw2["SPAACC"].map(clean_grade)
    raw2["SPAQEFF_clean"] = raw2["SPAQEFF"].map(clean_grade)
    pre = raw2[raw2["SEQN"].isin(seqns) & raw2["SPATTYPE_clean"].eq("Pre")].copy()
    pre["SPAMANU_numeric"] = numeric(pre["SPAMANU"])
    pre = pre.sort_values(["SEQN", "SPAMANU_numeric", "raw_index"]).copy()
    pre["attempt_ordinal"] = pre.groupby("SEQN").cumcount() + 1

    fits = repro[repro["row_type"].eq("maneuver_fit")].copy()
    fits["raw_index"] = numeric(fits["raw_index"]).astype("Int64")
    fits["model_ratio_numeric"] = numeric(fits["model_ratio"])
    fits["model_fev1_l_numeric"] = numeric(fits["model_fev1_l"])
    fits["model_fvc_l_numeric"] = numeric(fits["model_fvc_l"])
    fits["observed_ratio_numeric"] = numeric(fits["observed_ratio"])
    fits["volume_r2_numeric"] = numeric(fits["volume_r2"])
    fits["corrected_ratio_c1"] = fits["model_ratio_numeric"] - c1_offset
    fit_cols = [
        "raw_index",
        "model_ratio_numeric",
        "corrected_ratio_c1",
        "model_fev1_l_numeric",
        "model_fvc_l_numeric",
        "observed_ratio_numeric",
        "volume_r2_numeric",
        "volume_fit_success",
        "fit_error",
    ]
    out = pre.merge(fits[fit_cols], on="raw_index", how="left")
    out["raw_acceptable"] = out["SPAACC_clean"].str.upper().eq("Y")
    out["model_usable"] = (
        out["raw_acceptable"]
        & out["volume_fit_success"].map(bool_value)
        & np.isfinite(out["model_fev1_l_numeric"].to_numpy(dtype=float))
        & np.isfinite(out["model_fvc_l_numeric"].to_numpy(dtype=float))
        & np.isfinite(out["corrected_ratio_c1"].to_numpy(dtype=float))
    )
    return out


def prefix_stats(prefix: pd.DataFrame) -> dict[str, Any]:
    usable = prefix[prefix["model_usable"].astype(bool)].copy()
    raw_acceptable_n = int(prefix["raw_acceptable"].astype(bool).sum())
    usable_n = int(len(usable))
    fev1_delta = top_two_delta(usable["model_fev1_l_numeric"].to_numpy(dtype=float))
    fvc_delta = top_two_delta(usable["model_fvc_l_numeric"].to_numpy(dtype=float))
    repeatability_met = bool(
        usable_n >= 3
        and np.isfinite(fev1_delta)
        and np.isfinite(fvc_delta)
        and fev1_delta <= ATS_REPEATABILITY_L
        and fvc_delta <= ATS_REPEATABILITY_L
    )
    ratios = usable["corrected_ratio_c1"].to_numpy(dtype=float)
    if usable_n:
        fev1 = float(np.nanmax(usable["model_fev1_l_numeric"].to_numpy(dtype=float)))
        fvc = float(np.nanmax(usable["model_fvc_l_numeric"].to_numpy(dtype=float)))
        ratio_uncorrected = fev1 / fvc if fvc > 0 else np.nan
        ratio = float(ratio_uncorrected - float(prefix["c1_offset"].iloc[0])) if np.isfinite(ratio_uncorrected) else np.nan
        ratio_min = float(np.nanmin(ratios))
        ratio_max = float(np.nanmax(ratios))
    else:
        fev1 = np.nan
        fvc = np.nan
        ratio_uncorrected = np.nan
        ratio = np.nan
        ratio_min = np.nan
        ratio_max = np.nan
    return {
        "raw_acceptable_n": raw_acceptable_n,
        "usable_acceptable_n": usable_n,
        "fev1_delta_l": fev1_delta,
        "fvc_delta_l": fvc_delta,
        "ats_repeatability_met": repeatability_met,
        "ats_stop_criteria_met": bool(usable_n >= 3 and repeatability_met),
        "best_fev1_l": fev1,
        "best_fvc_l": fvc,
        "best_ratio_uncorrected": ratio_uncorrected,
        "best_ratio_corrected": ratio,
        "prefix_ratio_min": ratio_min,
        "prefix_ratio_max": ratio_max,
        "prefix_straddles_070": bool(np.isfinite(ratio_min) and np.isfinite(ratio_max) and ratio_min < 0.70 <= ratio_max),
        "ats_session_grade": session_grade(usable),
    }


def policy_decision(stats: dict[str, Any], terminal: bool = False, cap: bool = False) -> tuple[str, str, bool]:
    ratio = float(stats["best_ratio_corrected"]) if finite(stats.get("best_ratio_corrected")) else np.nan
    ratio_min = float(stats["prefix_ratio_min"]) if finite(stats.get("prefix_ratio_min")) else np.nan
    if not np.isfinite(ratio):
        return ("REFER" if terminal or cap else "REPEAT"), "no model-usable acceptable maneuver", False
    if ratio < 0.50:
        return "REFER", "severe-range best-test ratio", False

    normal_guard_blocks = bool(ratio >= FIXED_RATIO_THRESHOLD and np.isfinite(ratio_min) and ratio_min < FIXED_RATIO_THRESHOLD)
    if cap:
        if normal_guard_blocks:
            return "REFER", "ATS cap reached but normal-call guard blocks a normal label", False
        if ratio >= FIXED_RATIO_THRESHOLD:
            return "ACCEPT_NORMAL", "ATS cap best-test normal with guard satisfied", False
        return "ACCEPT_OBSTRUCTED", "ATS cap best-test obstructed", False

    if ratio < 0.60:
        return "ACCEPT_OBSTRUCTED", "baseline finalization: clear obstructed best-test ratio", False
    if ratio >= 0.72:
        if normal_guard_blocks:
            return ("REFER" if terminal else "REPEAT"), "normal-call guard blocks a normal label", False
        return "ACCEPT_NORMAL", "baseline finalization: clear normal best-test ratio with guard satisfied", False
    return ("REFER" if terminal else "REPEAT"), "baseline finalization: near-threshold best-test ratio remains deferred", False


def simulate_participant(row: pd.Series, attempts: pd.DataFrame, c1_offset: float) -> dict[str, Any]:
    seqn = float(row["SEQN"])
    part = attempts[attempts["SEQN"].astype(float).eq(seqn)].sort_values(["attempt_ordinal", "raw_index"]).copy()
    part["c1_offset"] = c1_offset
    n_available = int(len(part))
    trace: list[dict[str, Any]] = []
    if n_available == 0:
        return {
            "SEQN": seqn,
            "blows_available_pre": 0,
            "blows_requested": 0,
            "loop_decision": "CENSORED",
            "loop_finalized": False,
            "termination_type": "CENSORED",
            "termination_reason": "no recorded PRE maneuvers",
            "ats_session_grade": "F",
            "final_ratio": np.nan,
            "final_fev1_l": np.nan,
            "final_fvc_l": np.nan,
            "usable_acceptable_n": 0,
            "raw_acceptable_n": 0,
            "fev1_delta_l": np.nan,
            "fvc_delta_l": np.nan,
            "normal_guard_violation": False,
            "attempt_trace_json": json.dumps(trace),
        }

    final_stats: dict[str, Any] | None = None
    for k in range(1, min(n_available, ATS_MAX_ATTEMPTS) + 1):
        prefix = part.iloc[:k].copy()
        stats = prefix_stats(prefix)
        final_stats = stats
        ats_met = bool(stats["ats_stop_criteria_met"])
        action = "REPEAT"
        reason = "ATS stop criteria not yet met"
        terminal_type = ""
        terminal = False

        if ats_met:
            action, reason, _ = policy_decision(stats, terminal=True, cap=False)
            terminal = True
            terminal_type = "FINALIZE" if action in {"ACCEPT_NORMAL", "ACCEPT_OBSTRUCTED"} else "REFER"
        elif k == ATS_MAX_ATTEMPTS:
            action, reason, _ = policy_decision(stats, terminal=True, cap=True)
            terminal = True
            terminal_type = "ATS_CAP"
        elif finite(stats.get("best_ratio_corrected")) and float(stats["best_ratio_corrected"]) < 0.50:
            action = "REFER"
            reason = "severe-range best-test ratio before ATS stop criteria"
            terminal = True
            terminal_type = "REFER"
        elif k >= n_available:
            action = "CENSORED"
            reason = "no more recorded PRE maneuvers before ATS stop criteria were met"
            terminal = True
            terminal_type = "CENSORED"

        normal_guard_violation = bool(
            action == "ACCEPT_NORMAL"
            and np.isfinite(stats["prefix_ratio_min"])
            and float(stats["prefix_ratio_min"]) < FIXED_RATIO_THRESHOLD
        )
        trace.append(
            {
                "k": int(k),
                "raw_acceptable_n": int(stats["raw_acceptable_n"]),
                "usable_acceptable_n": int(stats["usable_acceptable_n"]),
                "ats_repeatability_met": bool(stats["ats_repeatability_met"]),
                "ats_stop_criteria_met": ats_met,
                "best_ratio_corrected": stats["best_ratio_corrected"],
                "prefix_ratio_min": stats["prefix_ratio_min"],
                "prefix_ratio_max": stats["prefix_ratio_max"],
                "ats_session_grade": stats["ats_session_grade"],
                "action": action,
                "reason": reason,
            }
        )
        if terminal:
            decision = action if action in {"ACCEPT_NORMAL", "ACCEPT_OBSTRUCTED"} else action
            return {
                "SEQN": seqn,
                "blows_available_pre": n_available,
                "blows_requested": int(k),
                "loop_decision": decision,
                "loop_finalized": decision in {"ACCEPT_NORMAL", "ACCEPT_OBSTRUCTED"},
                "termination_type": terminal_type,
                "termination_reason": reason,
                "ats_session_grade": stats["ats_session_grade"],
                "final_ratio": stats["best_ratio_corrected"],
                "final_fev1_l": stats["best_fev1_l"],
                "final_fvc_l": stats["best_fvc_l"],
                "usable_acceptable_n": int(stats["usable_acceptable_n"]),
                "raw_acceptable_n": int(stats["raw_acceptable_n"]),
                "fev1_delta_l": stats["fev1_delta_l"],
                "fvc_delta_l": stats["fvc_delta_l"],
                "prefix_ratio_min": stats["prefix_ratio_min"],
                "prefix_ratio_max": stats["prefix_ratio_max"],
                "prefix_straddles_070": stats["prefix_straddles_070"],
                "normal_guard_violation": normal_guard_violation,
                "attempt_trace_json": json.dumps(clean_json(trace)),
            }

    stats = final_stats or prefix_stats(part.iloc[:0].copy())
    return {
        "SEQN": seqn,
        "blows_available_pre": n_available,
        "blows_requested": int(min(n_available, ATS_MAX_ATTEMPTS)),
        "loop_decision": "CENSORED",
        "loop_finalized": False,
        "termination_type": "CENSORED",
        "termination_reason": "loop exhausted without terminal branch",
        "ats_session_grade": stats.get("ats_session_grade", "F"),
        "final_ratio": stats.get("best_ratio_corrected", np.nan),
        "final_fev1_l": stats.get("best_fev1_l", np.nan),
        "final_fvc_l": stats.get("best_fvc_l", np.nan),
        "usable_acceptable_n": int(stats.get("usable_acceptable_n", 0)),
        "raw_acceptable_n": int(stats.get("raw_acceptable_n", 0)),
        "fev1_delta_l": stats.get("fev1_delta_l", np.nan),
        "fvc_delta_l": stats.get("fvc_delta_l", np.nan),
        "prefix_ratio_min": stats.get("prefix_ratio_min", np.nan),
        "prefix_ratio_max": stats.get("prefix_ratio_max", np.nan),
        "prefix_straddles_070": stats.get("prefix_straddles_070", False),
        "normal_guard_violation": False,
        "attempt_trace_json": json.dumps(clean_json(trace)),
    }


def run_loop(current_rows: pd.DataFrame, attempts: pd.DataFrame, c1_offset: float) -> pd.DataFrame:
    loop_records = [simulate_participant(row, attempts, c1_offset) for _, row in current_rows.iterrows()]
    loop = pd.DataFrame(loop_records)
    out = current_rows.merge(loop, on="SEQN", how="left")
    return out


def add_post_repeat_columns(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    baseline_final = out["post_recovery_finalized"].astype(bool)
    loop_resolved_deferred = ~baseline_final & out["loop_finalized"].astype(bool)
    out["post_repeat_decision"] = out["post_recovery_decision"]
    out["post_repeat_finalized"] = baseline_final
    out.loc[loop_resolved_deferred, "post_repeat_decision"] = out.loc[loop_resolved_deferred, "loop_decision"]
    out.loc[loop_resolved_deferred, "post_repeat_finalized"] = True
    out["post_repeat_added_by_loop"] = loop_resolved_deferred
    return out


def loop_coverage_metrics(rows: pd.DataFrame, baseline_fraction: float) -> dict[str, Any]:
    strict_finalized = rows["loop_finalized"].astype(bool)
    strict_non_censored = ~rows["termination_type"].eq("CENSORED")
    baseline_final = rows["post_recovery_finalized"].astype(bool)
    deferred = ~baseline_final
    deferred_non_censored = deferred & strict_non_censored
    post_repeat_finalized = rows["post_repeat_finalized"].astype(bool)
    post_repeat_non_censored_den = int(baseline_final.sum() + deferred_non_censored.sum())
    post_repeat_non_censored_num = int(baseline_final.sum() + (deferred & strict_non_censored & strict_finalized).sum())
    return {
        "single_pass_baseline_coverage": baseline_fraction,
        "post_repeat_finalized_n": int(post_repeat_finalized.sum()),
        "post_repeat_total_n": int(len(rows)),
        "post_repeat_honest_floor_coverage": float(post_repeat_finalized.mean()) if len(rows) else np.nan,
        "post_repeat_non_censored_n": post_repeat_non_censored_den,
        "post_repeat_non_censored_finalized_n": post_repeat_non_censored_num,
        "post_repeat_non_censored_coverage": float(post_repeat_non_censored_num / post_repeat_non_censored_den)
        if post_repeat_non_censored_den
        else np.nan,
        "coverage_delta_vs_single_pass": float(post_repeat_finalized.mean() - baseline_fraction) if len(rows) else np.nan,
        "strict_from_scratch_loop_finalized_n": int(strict_finalized.sum()),
        "strict_from_scratch_loop_total_n": int(len(rows)),
        "strict_from_scratch_loop_honest_floor_coverage": float(strict_finalized.mean()) if len(rows) else np.nan,
        "strict_from_scratch_loop_non_censored_n": int(strict_non_censored.sum()),
        "strict_from_scratch_loop_non_censored_finalized_n": int((strict_finalized & strict_non_censored).sum()),
        "strict_from_scratch_loop_non_censored_coverage": float(
            (strict_finalized & strict_non_censored).sum() / strict_non_censored.sum()
        )
        if int(strict_non_censored.sum())
        else np.nan,
        "strict_from_scratch_loop_delta_vs_single_pass": float(strict_finalized.mean() - baseline_fraction)
        if len(rows)
        else np.nan,
        "mean_blows_requested": float(rows["blows_requested"].mean()) if len(rows) else np.nan,
        "median_blows_requested": float(rows["blows_requested"].median()) if len(rows) else np.nan,
        "ats_cap_n": int(rows["termination_type"].eq("ATS_CAP").sum()),
        "ats_cap_fraction": float(rows["termination_type"].eq("ATS_CAP").mean()) if len(rows) else np.nan,
        "censored_n": int(rows["termination_type"].eq("CENSORED").sum()),
        "censored_fraction": float(rows["termination_type"].eq("CENSORED").mean()) if len(rows) else np.nan,
        "deferred_censored_n": int((deferred & rows["termination_type"].eq("CENSORED")).sum()),
        "deferred_censored_fraction": float((deferred & rows["termination_type"].eq("CENSORED")).sum() / deferred.sum())
        if int(deferred.sum())
        else np.nan,
    }


def severe_by_path(rows: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    scopes = [
        (
            "LOOP_FINALIZE",
            rows[rows["termination_type"].eq("FINALIZE") & rows["loop_finalized"].astype(bool)],
            "loop_decision",
        ),
        (
            "LOOP_ATS_CAP",
            rows[rows["termination_type"].eq("ATS_CAP") & rows["loop_finalized"].astype(bool)],
            "loop_decision",
        ),
        ("STRICT_LOOP_OVERALL", rows[rows["loop_finalized"].astype(bool)], "loop_decision"),
        ("POST_REPEAT_POLICY", rows[rows["post_repeat_finalized"].astype(bool)], "post_repeat_decision"),
    ]
    for name, part, decision_col in scopes:
        severe = part[part["truth_severity"].eq("severe")].copy()
        severe_fn = int((severe[decision_col].map(final_label_from_decision).eq("NORMAL")).sum())
        lo, hi = exact_ci(severe_fn, int(len(severe)))
        records.append(
            {
                "path": name,
                "finalized_n": int(len(part)),
                "severe_finalized_n": int(len(severe)),
                "severe_fn_n": severe_fn,
                "severe_fn_rate": float(severe_fn / len(severe)) if len(severe) else np.nan,
                "severe_fn_ci_low": lo,
                "severe_fn_ci_high": hi,
            }
        )
    return records


def heldout_maneuver_accuracy(rows: pd.DataFrame, attempts: pd.DataFrame) -> dict[str, Any]:
    blow_records: list[dict[str, Any]] = []
    participant_records: list[dict[str, Any]] = []
    selected = rows[rows["loop_finalized"].astype(bool)].copy()
    for _, row in selected.iterrows():
        seqn = float(row["SEQN"])
        pred_label = final_label_from_decision(str(row["loop_decision"]))
        consumed = int(row["blows_requested"])
        part = attempts[
            attempts["SEQN"].astype(float).eq(seqn)
            & attempts["model_usable"].astype(bool)
            & (attempts["attempt_ordinal"].astype(int) > consumed)
        ].copy()
        case_obs: list[bool] = []
        case_model: list[bool] = []
        for _, holdout in part.iterrows():
            observed_ratio = float(holdout["observed_ratio_numeric"]) if finite(holdout["observed_ratio_numeric"]) else np.nan
            model_ratio = float(holdout["corrected_ratio_c1"]) if finite(holdout["corrected_ratio_c1"]) else np.nan
            if not pred_label or not np.isfinite(observed_ratio):
                continue
            observed_label = "OBSTRUCTED" if observed_ratio < FIXED_RATIO_THRESHOLD else "NORMAL"
            model_label = "OBSTRUCTED" if model_ratio < FIXED_RATIO_THRESHOLD else "NORMAL" if np.isfinite(model_ratio) else ""
            obs_ok = bool(pred_label == observed_label)
            model_ok = bool(model_label and pred_label == model_label)
            blow_records.append(
                {
                    "SEQN": seqn,
                    "observed_ok": obs_ok,
                    "model_ok": model_ok,
                    "termination_type": row["termination_type"],
                }
            )
            case_obs.append(obs_ok)
            case_model.append(model_ok)
        if case_obs:
            participant_records.append(
                {
                    "SEQN": seqn,
                    "participant_majority_observed_ok": sum(case_obs) > len(case_obs) / 2,
                    "participant_all_observed_ok": all(case_obs),
                    "participant_majority_model_ok": sum(case_model) > len(case_model) / 2,
                    "heldout_blow_n": len(case_obs),
                }
            )
    blow = pd.DataFrame(blow_records)
    participant = pd.DataFrame(participant_records)
    return {
        "participants_finalized_n": int(len(selected)),
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


def resolved_deferred_summary(rows: pd.DataFrame) -> dict[str, Any]:
    deferred = rows[~rows["post_recovery_finalized"].astype(bool)].copy()
    resolved = deferred[deferred["loop_finalized"].astype(bool)].copy()
    additional = (resolved["blows_requested"].astype(int) - 1).clip(lower=0)
    never = deferred[~deferred["loop_finalized"].astype(bool)].copy()
    by_reason = []
    for reason, part in never.groupby("primary_deferral_reason", dropna=False):
        by_reason.append(
            {
                "reason": str(reason),
                "n": int(len(part)),
                "fraction_never_resolved": float(len(part) / len(never)) if len(never) else np.nan,
            }
        )
    by_reason = sorted(by_reason, key=lambda item: (-item["n"], item["reason"]))
    return {
        "single_pass_deferred_n": int(len(deferred)),
        "resolved_n": int(len(resolved)),
        "resolved_by_0_additional_blow_n": int((additional == 0).sum()),
        "resolved_by_1_additional_blow_n": int((additional == 1).sum()),
        "resolved_by_2_additional_blows_n": int((additional == 2).sum()),
        "resolved_by_3plus_additional_blows_n": int((additional >= 3).sum()),
        "never_resolved_n": int(len(never)),
        "never_resolved_by_reason": by_reason,
    }


def ratio_spread_for_cases(cases: pd.DataFrame, attempts: pd.DataFrame) -> dict[str, Any]:
    records = []
    for _, row in cases.iterrows():
        seqn = float(row["SEQN"])
        cap = min(int(row["blows_available_pre"]), ATS_MAX_ATTEMPTS)
        vals = attempts[
            attempts["SEQN"].astype(float).eq(seqn)
            & attempts["model_usable"].astype(bool)
            & (attempts["attempt_ordinal"].astype(int) <= cap)
        ]["corrected_ratio_c1"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            records.append({"SEQN": seqn, "ratio_n": 0, "spread": np.nan, "straddles": False, "median_margin": np.nan})
            continue
        records.append(
            {
                "SEQN": seqn,
                "ratio_n": int(len(vals)),
                "spread": float(np.max(vals) - np.min(vals)) if len(vals) > 1 else 0.0,
                "straddles": bool(np.min(vals) < FIXED_RATIO_THRESHOLD <= np.max(vals)),
                "median_margin": float(abs(np.median(vals) - FIXED_RATIO_THRESHOLD)),
            }
        )
    df = pd.DataFrame(records)
    if df.empty:
        return {
            "cases_n": 0,
            "with_ratio_n": 0,
            "median_spread": np.nan,
            "iqr_spread": [np.nan, np.nan],
            "straddles_070_n": 0,
            "straddles_070_fraction": np.nan,
            "median_margin_to_070": np.nan,
        }
    with_ratio = df[df["ratio_n"] > 0].copy()
    spread = with_ratio["spread"].to_numpy(dtype=float)
    return {
        "cases_n": int(len(df)),
        "with_ratio_n": int(len(with_ratio)),
        "median_spread": float(np.nanmedian(spread)) if len(spread) else np.nan,
        "iqr_spread": [float(x) for x in np.nanpercentile(spread, [25, 75])] if len(spread) else [np.nan, np.nan],
        "straddles_070_n": int(with_ratio["straddles"].sum()) if len(with_ratio) else 0,
        "straddles_070_fraction": float(with_ratio["straddles"].mean()) if len(with_ratio) else np.nan,
        "median_margin_to_070": float(np.nanmedian(with_ratio["median_margin"].to_numpy(dtype=float)))
        if len(with_ratio)
        else np.nan,
    }


def grade_distribution(rows: pd.DataFrame) -> list[dict[str, Any]]:
    final = rows[rows["loop_finalized"].astype(bool)].copy()
    total = int(len(final))
    counts = final["ats_session_grade"].value_counts(dropna=False).sort_index()
    return [
        {"grade": str(grade), "n": int(n), "fraction": float(n / total) if total else np.nan}
        for grade, n in counts.items()
    ]


def decision_counts(rows: pd.DataFrame) -> list[dict[str, Any]]:
    total = int(len(rows))
    counts = rows.groupby(["termination_type", "loop_decision"], dropna=False).size().reset_index(name="n")
    out = []
    for item in counts.to_dict("records"):
        out.append(
            {
                "termination_type": str(item["termination_type"]),
                "loop_decision": str(item["loop_decision"]),
                "n": int(item["n"]),
                "fraction": float(item["n"] / total) if total else np.nan,
            }
        )
    return out


def verdict(result: dict[str, Any]) -> tuple[str, str]:
    coverage = result["coverage_metrics"]
    post_repeat_metrics = result["post_repeat_truth_metrics"]
    heldout = result["heldout_newly_resolved_accuracy"]
    severe_bad = post_repeat_metrics["severe_fn_n"] > 0 or result["normal_guard_violation_n"] > 0
    heldout_bad = (
        np.isfinite(heldout.get("observed_blow_accuracy", np.nan))
        and heldout.get("observed_blow_n", 0) >= 30
        and heldout["observed_blow_accuracy"] < 0.90
    )
    if severe_bad or heldout_bad:
        return (
            "LOOP-UNSAFE",
            "The loop produced a safety or held-out-agreement regression and should not replace the retained policy.",
        )
    if (
        coverage["deferred_censored_fraction"] >= 0.50
        and coverage["post_repeat_honest_floor_coverage"] <= coverage["single_pass_baseline_coverage"]
    ):
        return (
            "LOOP-CENSORED",
            "NHANES has too few recorded PRE maneuvers to estimate the full 8-attempt protocol as a coverage improvement; censored cases are not counted as finalized.",
        )
    if coverage["coverage_delta_vs_single_pass"] >= 0.03:
        return (
            "LOOP-IMPROVES-COVERAGE",
            "The honest floor coverage materially exceeds the single-pass baseline while preserving severe FN=0.",
        )
    return (
        "LOOP-NEUTRAL",
        "Repeats did not materially improve deployable coverage; unresolved cases remain censored, borderline, or discordant around 0.70.",
    )


def write_report(root: Path, result: dict[str, Any], rows: pd.DataFrame) -> None:
    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    rows.to_csv(out_dir / f"{OUTPUT_STEM}_participants.csv", index=False)

    baseline = result["baseline_metrics"]
    coverage = result["coverage_metrics"]
    loop_metrics = result["loop_truth_metrics"]
    post_repeat_metrics = result["post_repeat_truth_metrics"]
    resolved = result["deferred_resolution"]
    spread_never = result["never_resolved_ratio_spread"]
    spread_resolved = result["resolved_ratio_spread"]
    lines = [
        "# Gate - Iterative ATS/ERS Maneuver Loop, Cycle F",
        "",
        f"VERDICT: {result['verdict']} - {result['verdict_reason']}",
        "",
        "## ATS/ERS Source",
        "",
        f"- Source: ATS/ERS 2019 Standardization of Spirometry, {ATS_ERS_2019_URL}.",
        f"- Protocol constants used here: at least 3 acceptable maneuvers, repeatability of the two largest FEV1 within {int(ATS_REPEATABILITY_L * 1000)} mL and the two largest FVC within {int(ATS_REPEATABILITY_L * 1000)} mL, maximum {ATS_MAX_ATTEMPTS} attempts.",
        "",
        "## Protocol Decisions",
        "",
    ]
    lines.extend(f"- {item}" for item in result["protocol_decisions"])
    lines.extend(
        [
            "",
            "## Part 0 - Blow Availability and Censoring",
            "",
            f"- Raw source: `{result['raw_source']}`.",
            f"- Raw source note: {result['raw_source_note']}.",
            f"- TEST participants: {result['test_n']}.",
            f"- PRE median maneuvers: {fmt(result['availability']['pre_median'], 1)}; PRE fraction with >=3 maneuvers: {fmt(result['availability']['pre_ge3_fraction'], 3)} ({result['availability']['pre_ge3_n']}/{result['test_n']}).",
            f"- POST/Bron audit only: {result['availability']['post_any_n']}/{result['test_n']} TEST participants have POST/Bron rows. These are never pooled into the loop.",
            "",
            "PRE maneuver count distribution:",
            "",
            md_table(
                result["availability"]["pre_distribution"],
                [("bucket", "PRE count", "str"), ("n", "n", "int"), ("fraction", "fraction", "float3")],
            ),
            "",
            "POST/Bron maneuver count distribution (availability audit only):",
            "",
            md_table(
                result["availability"]["post_distribution"],
                [("bucket", "POST/Bron count", "str"), ("n", "n", "int"), ("fraction", "fraction", "float3")],
            ),
            "",
            "Censoring problem: NHANES records the maneuvers the technician performed and NHANES retained. The simulation cannot invent later PRE blows, so each session is censored at its recorded PRE count. Censored sessions are counted as not finalized in the honest floor.",
            "",
            "## Part 1 - Causal Iterative Loop",
            "",
            "The enforcing code path is `simulate_participant()` -> `prefix = part.iloc[:k].copy()` -> `prefix_stats(prefix)` -> `policy_decision(stats, ...)`. The decision function receives only the prefix `1..k`; later maneuvers are only used after termination for held-out scoring. `n_available` is used only to decide CENSORED after the prefix decision cannot stop.",
            "",
            md_table(
                result["loop_decision_counts"],
                [
                    ("termination_type", "termination", "str"),
                    ("loop_decision", "decision", "str"),
                    ("n", "n", "int"),
                    ("fraction", "fraction all", "float3"),
                ],
            ),
            "",
            "## Part 2 - Post-Repeat Coverage",
            "",
            "Operational post-repeat coverage retains the current single-pass finalized cases and applies the causal repeat loop only to cases that the single-pass policy deferred. The strict from-scratch loop audit is also shown because Part 1 simulates every participant from k=1, but it is not the quoted operational post-repeat coverage.",
            "",
            md_table(
                [
                    {
                        "scope": "single-pass baseline",
                        "finalized_n": baseline["finalized_n"],
                        "denominator": result["test_n"],
                        "coverage": coverage["single_pass_baseline_coverage"],
                    },
                    {
                        "scope": "post-repeat, non-censored deferred denominator",
                        "finalized_n": coverage["post_repeat_non_censored_finalized_n"],
                        "denominator": coverage["post_repeat_non_censored_n"],
                        "coverage": coverage["post_repeat_non_censored_coverage"],
                    },
                    {
                        "scope": "post-repeat honest floor",
                        "finalized_n": coverage["post_repeat_finalized_n"],
                        "denominator": coverage["post_repeat_total_n"],
                        "coverage": coverage["post_repeat_honest_floor_coverage"],
                    },
                    {
                        "scope": "strict from-scratch loop audit",
                        "finalized_n": coverage["strict_from_scratch_loop_finalized_n"],
                        "denominator": coverage["strict_from_scratch_loop_total_n"],
                        "coverage": coverage["strict_from_scratch_loop_honest_floor_coverage"],
                    },
                ],
                [
                    ("scope", "scope", "str"),
                    ("finalized_n", "finalized", "int"),
                    ("denominator", "denominator", "int"),
                    ("coverage", "coverage", "float3"),
                ],
            ),
            "",
            f"- Coverage delta vs single-pass baseline: {fmt(coverage['coverage_delta_vs_single_pass'], 3)}.",
            f"- Strict from-scratch loop audit delta vs single-pass baseline: {fmt(coverage['strict_from_scratch_loop_delta_vs_single_pass'], 3)}. This audit is not the quoted operational post-repeat coverage because already-finalized cases would not be forced back into repeat/referral.",
            f"- Mean / median blows requested: {fmt(coverage['mean_blows_requested'], 2)} / {fmt(coverage['median_blows_requested'], 1)}.",
            f"- ATS 8-cap fraction: {fmt(coverage['ats_cap_fraction'], 3)} ({coverage['ats_cap_n']}/{result['test_n']}).",
            f"- Censored fraction: {fmt(coverage['censored_fraction'], 3)} ({coverage['censored_n']}/{result['test_n']}).",
            f"- Censored fraction among single-pass deferred cases: {fmt(coverage['deferred_censored_fraction'], 3)} ({coverage['deferred_censored_n']}/{resolved['single_pass_deferred_n']}).",
            "",
            "Held-out maneuver proxy accuracy for newly resolved single-pass deferred cases:",
            "",
            md_table(
                [result["heldout_newly_resolved_accuracy"]],
                [
                    ("participants_finalized_n", "finalized participants", "int"),
                    ("participants_with_proxy_n", "with held-out proxy", "int"),
                    ("observed_blow_correct_n", "observed correct", "int"),
                    ("observed_blow_n", "observed n", "int"),
                    ("observed_blow_accuracy", "observed blow acc", "float3"),
                    ("participant_majority_observed_correct_n", "participant majority correct", "int"),
                    ("participant_majority_observed_n", "participant majority n", "int"),
                    ("participant_majority_observed_accuracy", "participant majority acc", "float3"),
                ],
            ),
            "",
            "Held-out maneuver proxy accuracy for all strict from-scratch loop finalizations:",
            "",
            md_table(
                [result["heldout_maneuver_accuracy"]],
                [
                    ("participants_finalized_n", "finalized participants", "int"),
                    ("participants_with_proxy_n", "with held-out proxy", "int"),
                    ("observed_blow_correct_n", "observed correct", "int"),
                    ("observed_blow_n", "observed n", "int"),
                    ("observed_blow_accuracy", "observed blow acc", "float3"),
                    ("participant_majority_observed_correct_n", "participant majority correct", "int"),
                    ("participant_majority_observed_n", "participant majority n", "int"),
                    ("participant_majority_observed_accuracy", "participant majority acc", "float3"),
                ],
            ),
            "",
            "Truth-derived diagnostic metrics (reported because the task requests sens/spec/kappa; not quoted as held-out accuracy):",
            "",
            md_table(
                [
                    {"scope": "single-pass baseline", **baseline},
                    {"scope": "post-repeat policy", **post_repeat_metrics},
                    {"scope": "strict from-scratch loop audit", **loop_metrics},
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
            "## Part 3 - Does Repeating Help?",
            "",
            md_table(
                [resolved],
                [
                    ("single_pass_deferred_n", "single-pass deferred", "int"),
                    ("resolved_n", "resolved by loop", "int"),
                    ("resolved_by_0_additional_blow_n", "0 addl", "int"),
                    ("resolved_by_1_additional_blow_n", "1 addl", "int"),
                    ("resolved_by_2_additional_blows_n", "2 addl", "int"),
                    ("resolved_by_3plus_additional_blows_n", "3+ addl", "int"),
                    ("never_resolved_n", "never resolved", "int"),
                ],
            ),
            "",
            "Never-resolved deferred cases by existing R1-R4 taxonomy:",
            "",
            md_table(
                resolved["never_resolved_by_reason"],
                [("reason", "reason", "str"), ("n", "n", "int"), ("fraction_never_resolved", "fraction", "float3")],
            ),
            "",
            "Repeated ratio behavior:",
            "",
            md_table(
                [
                    {"scope": "resolved deferred", **spread_resolved},
                    {"scope": "never-resolved deferred", **spread_never},
                ],
                [
                    ("scope", "scope", "str"),
                    ("cases_n", "cases", "int"),
                    ("with_ratio_n", "with ratios", "int"),
                    ("median_spread", "median spread", "float3"),
                    ("straddles_070_n", "straddles 0.70 n", "int"),
                    ("straddles_070_fraction", "straddles 0.70 frac", "float3"),
                    ("median_margin_to_070", "median margin", "float3"),
                ],
            ),
            "",
            result["repeat_help_statement"],
            "",
            "## Part 4 - Safety Under the Loop",
            "",
            md_table(
                result["severe_by_path"],
                [
                    ("path", "path", "str"),
                    ("finalized_n", "finalized", "int"),
                    ("severe_finalized_n", "severe finalized", "int"),
                    ("severe_fn_n", "severe FN", "int"),
                    ("severe_fn_ci_high", "severe FN CP high", "float3"),
                ],
            ),
            "",
            f"- Normal-call guard violations across terminal normal calls: {result['normal_guard_violation_n']}.",
            "- ATS session grade is emitted for every finalized row in the participant CSV and summarized below.",
            "",
            md_table(result["finalized_grade_distribution"], [("grade", "grade", "str"), ("n", "n", "int"), ("fraction", "fraction", "float3")]),
            "",
            "## Guardrail Statement",
            "",
            "- The loop is PRE-only. POST/Bron rows are counted only in the availability audit.",
            "- Deferred participants that become CENSORED in the repeat loop are not counted as newly finalized in the operational honest floor; the strict from-scratch audit also counts every CENSORED loop row as not finalized.",
            "- Held-out accuracy uses only maneuvers after the stopping attempt, where such maneuvers exist.",
            "- The NHANES fixed-ratio truth is used for sens/spec/kappa and severe-FN scoring, not for selection and not as the quoted accuracy.",
            "- Severe FN=0 and the normal-call guard are required before this loop can be considered safe.",
            "",
            f"VERDICT: {result['verdict']} - {result['verdict_reason']}",
            "",
        ]
    )
    (out_dir / f"{OUTPUT_STEM}.md").write_text("\n".join(lines), encoding="utf-8")


def run(root: Path) -> dict[str, Any]:
    workflow, repro = load_inputs(root)
    c1 = correction_offset(workflow)
    test = workflow[workflow["split"].eq("test")].copy()
    current_rows = apply_recovery(test, corrected_maneuvers(repro, c1))
    raw_path, raw_note = find_or_unzip_raw(root, "F")
    raw, raw_info = read_raw(raw_path, "F")
    availability = availability_audit(raw, test["SEQN"])
    attempts = build_attempt_table(raw, repro, c1, test["SEQN"])
    rows = add_post_repeat_columns(run_loop(current_rows, attempts, c1))
    baseline_metrics = final_metrics(current_rows, "post_recovery_decision", "post_recovery_finalized")
    post_repeat_truth_metrics = final_metrics(rows, "post_repeat_decision", "post_repeat_finalized")
    loop_truth_metrics = final_metrics(rows, "loop_decision", "loop_finalized")
    coverage_metrics = loop_coverage_metrics(rows, baseline_metrics["finalized_fraction"])
    heldout = heldout_maneuver_accuracy(rows, attempts)
    heldout_newly_resolved = heldout_maneuver_accuracy(rows[rows["post_repeat_added_by_loop"].astype(bool)].copy(), attempts)
    severe_paths = severe_by_path(rows)
    deferred_summary = resolved_deferred_summary(rows)
    deferred = rows[~rows["post_recovery_finalized"].astype(bool)].copy()
    resolved_deferred = deferred[deferred["loop_finalized"].astype(bool)].copy()
    never_deferred = deferred[~deferred["loop_finalized"].astype(bool)].copy()
    spread_resolved = ratio_spread_for_cases(resolved_deferred, attempts)
    spread_never = ratio_spread_for_cases(never_deferred, attempts)
    normal_guard_violation_n = int(rows["normal_guard_violation"].astype(bool).sum())
    if spread_never["straddles_070_fraction"] and spread_never["straddles_070_fraction"] >= 0.25:
        repeat_help_statement = (
            "KEY QUESTION: repeated blows often keep landing on both sides of 0.70 in the never-resolved set. "
            "More blows do not reliably resolve those cases; this supports the biological-ambiguity interpretation."
        )
    elif deferred_summary["resolved_n"] > 0:
        repeat_help_statement = (
            "KEY QUESTION: repeats resolve a subset of deferred cases, but the remaining cases are dominated by censoring, "
            "near-threshold spread, or no usable acceptable fitted maneuvers."
        )
    else:
        repeat_help_statement = (
            "KEY QUESTION: repeats do not resolve the deferred set under the causal PRE-only simulation; the result is neutral or censored rather than a coverage gain."
        )

    result: dict[str, Any] = {
        "seed": RANDOM_SEED,
        "test_n": int(len(test)),
        "raw_source": str(raw_path),
        "raw_source_note": raw_note,
        "raw_loaded_rows": int(raw_info["loaded_rows"]),
        "ats_source": ATS_ERS_2019_URL,
        "ats_repeatability_l": ATS_REPEATABILITY_L,
        "ats_max_attempts": ATS_MAX_ATTEMPTS,
        "c1_offset_from_existing_policy": c1,
        "protocol_decisions": [
            "TEST split only; TRAIN rows are ignored.",
            "The single-pass baseline is the current deployable recovery gate: post_recovery_finalized from gate_deferral_recovery_f.apply_recovery.",
            "The loop uses raw PRE attempt order from SPXRAW, but uses only existing OMBE-fitted maneuver rows for FEV1/FVC/ratio values; no solver is rerun.",
            "POST/Bron rows are audited as availability only and are never pooled into PRE repeats.",
            "A maneuver is model-usable only when it is SPAACC == Y and has a successful finite existing OMBE fit. Raw SPAACC counts are also retained in the output.",
            "At each k, prefix_stats receives only part.iloc[:k]. Later maneuvers are used only after termination for held-out proxy scoring.",
            "The ATS stop criterion is >=3 model-usable acceptable PRE maneuvers plus top-two FEV1 and FVC repeatability within 150 mL.",
            "Before ATS criteria are met, the loop keeps requesting PRE blows unless the 8-attempt cap is hit, the participant runs out of recorded PRE blows, or the best-test ratio is severe enough to refer.",
            "When ATS criteria are met, the same baseline finalization bands are retained: obstructed only below 0.60, normal only at or above 0.72, near-threshold results are referred rather than forced.",
            "At the ATS 8-cap, the best-test ratio is emitted with grade unless it is severe-range or the conservative normal-call guard blocks a normal call.",
            "The conservative normal-call guard is enforced at every terminal normal call: no ACCEPT_NORMAL if any model-usable acceptable prefix maneuver has corrected ratio <0.70.",
            "CENSORED means NHANES recorded no additional PRE maneuver; censored rows are never counted as finalized in the honest floor.",
            "Held-out accuracy uses only fitted PRE maneuvers after the stopping attempt. NHANES truth is used only for sens/spec/kappa and severe-FN safety scoring.",
        ],
        "availability": availability,
        "baseline_metrics": baseline_metrics,
        "coverage_metrics": coverage_metrics,
        "post_repeat_truth_metrics": post_repeat_truth_metrics,
        "loop_truth_metrics": loop_truth_metrics,
        "heldout_maneuver_accuracy": heldout,
        "heldout_newly_resolved_accuracy": heldout_newly_resolved,
        "loop_decision_counts": decision_counts(rows),
        "deferred_resolution": deferred_summary,
        "resolved_ratio_spread": spread_resolved,
        "never_resolved_ratio_spread": spread_never,
        "repeat_help_statement": repeat_help_statement,
        "severe_by_path": severe_paths,
        "normal_guard_violation_n": normal_guard_violation_n,
        "finalized_grade_distribution": grade_distribution(rows),
    }
    result["verdict"], result["verdict_reason"] = verdict(result)
    write_report(root, result, rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    coverage = result["coverage_metrics"]
    post_repeat = result["post_repeat_truth_metrics"]
    print(f"Iterative ATS loop: {result['verdict']} - {result['verdict_reason']}")
    print(f"Single-pass baseline coverage: {coverage['single_pass_baseline_coverage']:.3f}")
    print(f"Post-repeat honest floor coverage: {coverage['post_repeat_honest_floor_coverage']:.3f}")
    print(f"Post-repeat non-censored coverage: {coverage['post_repeat_non_censored_coverage']:.3f}")
    print(f"Strict from-scratch loop coverage: {coverage['strict_from_scratch_loop_honest_floor_coverage']:.3f}")
    print(f"Censored fraction: {coverage['censored_fraction']:.3f}")
    print(f"Post-repeat severe FN: {post_repeat['severe_fn_n']}/{post_repeat['severe_finalized_n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
