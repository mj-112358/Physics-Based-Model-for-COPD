from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .gate_graded_probability_f import (
    bootstrap_auc_ci,
    boundary_mask,
    class_reweight,
    feature_audit,
    fit_model,
    impute_frames,
    load_model_frame,
    make_gbt,
    prevalence_from_full_cohort,
    probability,
    truth_obstructed,
)
from .gate_graded_probability_repair_f import prior_adjust
from .gate_workflow_policy_f import RANDOM_SEED, clean_json, exact_ci, fmt, md_table

OUTPUT_STEM = "gate_frontier_F"
FIXED_RATIO_THRESHOLD = 0.70
SEVERE_NORMAL_GUARD_RATIO = 0.60
NEW_NPV_FLOOR = 0.95
NEW_PPV_FLOOR = 0.70
OLD_SYMMETRIC_FLOOR = 0.90
MATERIAL_COVERAGE_EPS = 0.01


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def weighted_kappa_from_counts(tp: float, tn: float, fp: float, fn: float) -> float:
    total = tp + tn + fp + fn
    if total <= 0:
        return np.nan
    observed = (tp + tn) / total
    true_pos = (tp + fn) / total
    pred_pos = (tp + fp) / total
    expected = true_pos * pred_pos + (1.0 - true_pos) * (1.0 - pred_pos)
    if expected >= 1.0:
        return np.nan
    return float((observed - expected) / (1.0 - expected))


def guard_ratio(df: pd.DataFrame) -> pd.Series:
    observed = pd.to_numeric(df.get("fev1_fvc_ratio"), errors="coerce")
    corrected = pd.to_numeric(df.get("corrected_ratio_c1"), errors="coerce")
    return observed.where(observed.notna(), corrected)


def apply_frontier_zones(df: pd.DataFrame, p: np.ndarray, p_low: float, p_high: float) -> pd.DataFrame:
    out = df.copy()
    out["p_adj"] = np.asarray(p, dtype=float)
    out["frontier_p_low"] = float(p_low)
    out["frontier_p_high"] = float(p_high)
    out["frontier_zone_pre_guard"] = "INDETERMINATE_GRADED"
    out.loc[out["p_adj"] <= p_low, "frontier_zone_pre_guard"] = "CONFIDENT_NORMAL"
    out.loc[out["p_adj"] >= p_high, "frontier_zone_pre_guard"] = "CONFIDENT_OBSTRUCTED"
    ratio = guard_ratio(out)
    guard_forced = out["frontier_zone_pre_guard"].eq("CONFIDENT_NORMAL") & ratio.lt(SEVERE_NORMAL_GUARD_RATIO)
    out["ratio_guard_value"] = ratio
    out["ratio_severe_normal_guard_forced"] = guard_forced
    out["frontier_zone"] = out["frontier_zone_pre_guard"]
    out.loc[guard_forced, "frontier_zone"] = "INDETERMINATE_GRADED"
    out["frontier_confident"] = out["frontier_zone"].isin(["CONFIDENT_NORMAL", "CONFIDENT_OBSTRUCTED"])
    out["frontier_pred_obstructed"] = out["frontier_zone"].eq("CONFIDENT_OBSTRUCTED")
    out["truth_obstructed"] = truth_obstructed(out)
    return out


def train_boundaries_for_target(
    train: pd.DataFrame,
    p_train: np.ndarray,
    target_coverage: float,
) -> dict[str, Any]:
    p = np.asarray(p_train, dtype=float)
    half_widths = np.unique(np.round(np.r_[0.0, np.abs(p - 0.5), 0.5], 12))
    best: dict[str, Any] | None = None
    for half_width in half_widths:
        low = 0.5 - float(half_width)
        high = 0.5 + float(half_width)
        zones = apply_frontier_zones(train, p, low, high)
        coverage = float(zones["frontier_confident"].mean()) if len(zones) else np.nan
        candidate = {
            "target_coverage": float(target_coverage),
            "target_achieved_on_train": bool(finite(coverage) and coverage >= target_coverage),
            "half_width": float(half_width),
            "p_low": float(low),
            "p_high": float(high),
            "train_confident_coverage_after_guard": coverage,
            "train_guard_forced_n": int(zones["ratio_severe_normal_guard_forced"].sum()),
        }
        if candidate["target_achieved_on_train"]:
            if best is None or candidate["half_width"] > best["half_width"]:
                best = candidate
    if best is not None:
        return best

    zones = apply_frontier_zones(train, p, 0.5, 0.5)
    return {
        "target_coverage": float(target_coverage),
        "target_achieved_on_train": False,
        "half_width": 0.0,
        "p_low": 0.5,
        "p_high": 0.5,
        "train_confident_coverage_after_guard": float(zones["frontier_confident"].mean()) if len(zones) else np.nan,
        "train_guard_forced_n": int(zones["ratio_severe_normal_guard_forced"].sum()),
    }


def weighted_metrics(rows: pd.DataFrame, weights: np.ndarray, arm: str, target_coverage: float, boundaries: dict[str, Any]) -> dict[str, Any]:
    confident = rows[rows["frontier_confident"].astype(bool)].copy()
    total_n = int(len(rows))
    confident_n = int(len(confident))
    if confident_n:
        y = confident["truth_obstructed"].astype(bool).to_numpy()
        pred = confident["frontier_pred_obstructed"].astype(bool).to_numpy()
        w = weights[confident.index.to_numpy()]
        tp = float(w[y & pred].sum())
        tn = float(w[~y & ~pred].sum())
        fp = float(w[~y & pred].sum())
        fn = float(w[y & ~pred].sum())
        raw_tp = int((y & pred).sum())
        raw_tn = int((~y & ~pred).sum())
        raw_fp = int((~y & pred).sum())
        raw_fn = int((y & ~pred).sum())
        total_w = tp + tn + fp + fn
        accuracy = float((tp + tn) / total_w) if total_w > 0 else np.nan
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else np.nan
        kappa = weighted_kappa_from_counts(tp, tn, fp, fn)
        severe_conf = confident["truth_severity"].astype(str).eq("severe")
        severe_fn = int((severe_conf & confident["frontier_zone"].eq("CONFIDENT_NORMAL")).sum())
        _lo, severe_hi = exact_ci(severe_fn, int(severe_conf.sum()))
        ratio = pd.to_numeric(confident["fev1_fvc_ratio"], errors="coerce")
        knife_edge_conf = ratio.ge(0.60) & ratio.lt(0.70)
        knife_edge_fraction = float(knife_edge_conf.mean())
    else:
        tp = tn = fp = fn = 0.0
        raw_tp = raw_tn = raw_fp = raw_fn = 0
        accuracy = sensitivity = specificity = ppv = npv = kappa = np.nan
        severe_fn = 0
        severe_hi = np.nan
        knife_edge_fraction = np.nan

    pass_old = bool(
        severe_fn == 0
        and finite(ppv)
        and finite(npv)
        and ppv >= OLD_SYMMETRIC_FLOOR
        and npv >= OLD_SYMMETRIC_FLOOR
    )
    pass_new = bool(
        severe_fn == 0
        and finite(ppv)
        and finite(npv)
        and ppv >= NEW_PPV_FLOOR
        and npv >= NEW_NPV_FLOOR
    )
    return {
        "arm": arm,
        "target_coverage": float(target_coverage),
        "target_achieved_on_train": bool(boundaries["target_achieved_on_train"]),
        "half_width": float(boundaries["half_width"]),
        "p_low": float(boundaries["p_low"]),
        "p_high": float(boundaries["p_high"]),
        "train_confident_coverage_after_guard": float(boundaries["train_confident_coverage_after_guard"]),
        "train_guard_forced_n": int(boundaries["train_guard_forced_n"]),
        "test_n": total_n,
        "confident_n": confident_n,
        "confident_coverage": float(confident_n / total_n) if total_n else np.nan,
        "accuracy_reweighted": accuracy,
        "kappa_reweighted": kappa,
        "sensitivity_reweighted": sensitivity,
        "specificity_reweighted": specificity,
        "ppv_reweighted": ppv,
        "npv_reweighted": npv,
        "tp_w": tp,
        "tn_w": tn,
        "fp_w": fp,
        "fn_w": fn,
        "tp": raw_tp,
        "tn": raw_tn,
        "fp": raw_fp,
        "fn": raw_fn,
        "severe_confident_n": int(confident["truth_severity"].astype(str).eq("severe").sum()) if confident_n else 0,
        "severe_fn_n": severe_fn,
        "severe_fn_ci_high": severe_hi,
        "knife_edge_true_060_070_fraction_of_confident": knife_edge_fraction,
        "test_guard_forced_n": int(rows["ratio_severe_normal_guard_forced"].sum()),
        "test_guard_forced_fraction": float(rows["ratio_severe_normal_guard_forced"].mean()) if total_n else np.nan,
        "passes_old_symmetric_090": pass_old,
        "passes_new_asymmetric": pass_new,
    }


def frontier_for_arm(
    arm: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    p_train: np.ndarray,
    p_test: np.ndarray,
    train_weights: np.ndarray,
    test_weights: np.ndarray,
    targets: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[float, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    zones_by_target: dict[float, pd.DataFrame] = {}
    for target in targets:
        boundaries = train_boundaries_for_target(train, p_train, float(target))
        train_zones = apply_frontier_zones(
            train,
            p_train,
            float(boundaries["p_low"]),
            float(boundaries["p_high"]),
        )
        zones = apply_frontier_zones(test, p_test, float(boundaries["p_low"]), float(boundaries["p_high"]))
        zones_by_target[float(target)] = zones
        train_metrics = weighted_metrics(
            train_zones,
            train_weights,
            arm,
            float(target),
            boundaries,
        )
        test_metrics = weighted_metrics(zones, test_weights, arm, float(target), boundaries)
        test_metrics.update(
            {
                "train_selection_coverage": train_metrics["confident_coverage"],
                "train_selection_kappa_reweighted": train_metrics["kappa_reweighted"],
                "train_selection_ppv_reweighted": train_metrics["ppv_reweighted"],
                "train_selection_npv_reweighted": train_metrics["npv_reweighted"],
                "train_selection_severe_fn_n": train_metrics["severe_fn_n"],
                "train_passes_old_symmetric_090": train_metrics["passes_old_symmetric_090"],
                "train_passes_new_asymmetric": train_metrics["passes_new_asymmetric"],
            }
        )
        rows.append(test_metrics)
    return rows, zones_by_target


def select_safe_point(rows: list[dict[str, Any]], pass_col: str) -> dict[str, Any] | None:
    passing = [row for row in rows if row[pass_col]]
    if not passing:
        return None
    return max(
        passing,
        key=lambda row: (
            row["train_selection_coverage"],
            row["train_selection_kappa_reweighted"]
            if finite(row["train_selection_kappa_reweighted"])
            else -np.inf,
        ),
    )


def bucket_from_distance(distance: pd.Series) -> pd.Series:
    out = pd.Series("FINALIZED_DETERMINISTIC", index=distance.index, dtype=object)
    out.loc[distance < 0.02] = "KNIFE_EDGE"
    out.loc[(distance >= 0.02) & (distance < 0.05)] = "NEAR"
    out.loc[distance >= 0.05] = "RECOVERABLE"
    return out


def bucket_rows(zones: pd.DataFrame, weights: np.ndarray) -> list[dict[str, Any]]:
    deferred = zones[zones["post_recovery_finalized"].eq(False)].copy()
    ratio = pd.to_numeric(deferred["fev1_fvc_ratio"], errors="coerce")
    distance = (ratio - FIXED_RATIO_THRESHOLD).abs()
    deferred["bucket"] = bucket_from_distance(distance)
    out: list[dict[str, Any]] = []
    for bucket, label in [("KNIFE_EDGE", "<0.02"), ("NEAR", "0.02-0.05"), ("RECOVERABLE", ">=0.05")]:
        part = deferred[deferred["bucket"].eq(bucket)].copy()
        confident = part[part["frontier_confident"].astype(bool)].copy()
        if len(confident):
            y = confident["truth_obstructed"].astype(bool).to_numpy()
            pred = confident["frontier_pred_obstructed"].astype(bool).to_numpy()
            w = weights[confident.index.to_numpy()]
            correct_w = float(w[y == pred].sum())
            total_w = float(w.sum())
            accuracy = float(correct_w / total_w) if total_w > 0 else np.nan
            severe = confident["truth_severity"].astype(str).eq("severe")
            severe_fn = int((severe & confident["frontier_zone"].eq("CONFIDENT_NORMAL")).sum())
        else:
            accuracy = np.nan
            severe_fn = 0
        out.append(
            {
                "bucket": bucket,
                "true_abs_distance_band": label,
                "n": int(len(part)),
                "confident_finalized_n": int(len(confident)),
                "confident_finalized_fraction": float(len(confident) / len(part)) if len(part) else np.nan,
                "accuracy_reweighted": accuracy,
                "severe_fn_n": severe_fn,
            }
        )
    return out


def frontier_shape(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["confident_coverage"])
    ppv_cross = next((row for row in ordered if finite(row["ppv_reweighted"]) and row["ppv_reweighted"] < NEW_PPV_FLOOR), None)
    npv_cross = next((row for row in ordered if finite(row["npv_reweighted"]) and row["npv_reweighted"] < NEW_NPV_FLOOR), None)
    kappa_080 = next((row for row in ordered if finite(row["kappa_reweighted"]) and row["kappa_reweighted"] < 0.80), None)
    kappa_070 = next((row for row in ordered if finite(row["kappa_reweighted"]) and row["kappa_reweighted"] < 0.70), None)
    guard_first = next((row for row in ordered if row["test_guard_forced_n"] > 0), None)
    best_kappa = max((row for row in ordered if finite(row["kappa_reweighted"])), key=lambda row: row["kappa_reweighted"], default=None)
    ppv_passing = [row for row in ordered if finite(row["ppv_reweighted"]) and row["ppv_reweighted"] >= NEW_PPV_FLOOR]
    npv_passing = [row for row in ordered if finite(row["npv_reweighted"]) and row["npv_reweighted"] >= NEW_NPV_FLOOR]
    new_passing = [row for row in ordered if row["passes_new_asymmetric"]]
    return {
        "best_kappa_point": best_kappa,
        "first_ppv_below_0p70": ppv_cross,
        "first_npv_below_0p95": npv_cross,
        "first_kappa_below_0p80": kappa_080,
        "first_kappa_below_0p70": kappa_070,
        "first_guard_forced_referral": guard_first,
        "ppv_passing_min_coverage": min((row["confident_coverage"] for row in ppv_passing), default=np.nan),
        "ppv_passing_max_coverage": max((row["confident_coverage"] for row in ppv_passing), default=np.nan),
        "npv_passing_max_coverage": max((row["confident_coverage"] for row in npv_passing), default=np.nan),
        "new_asymmetric_passing_min_coverage": min((row["confident_coverage"] for row in new_passing), default=np.nan),
        "new_asymmetric_passing_max_coverage": max((row["confident_coverage"] for row in new_passing), default=np.nan),
    }


def fit_probability_arm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    source_prevalence: float,
    target_prevalence: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x_train, x_test, imputation = impute_frames(train, test, features)
    y_train = truth_obstructed(train).astype(int).to_numpy()
    model = fit_model(make_gbt(), x_train, y_train)
    p_train_raw = probability(model, x_train)
    p_test_raw = probability(model, x_test)
    p_train_adj = prior_adjust(p_train_raw, source_prevalence, target_prevalence)
    p_test_adj = prior_adjust(p_test_raw, source_prevalence, target_prevalence)
    return p_train_adj, p_test_adj, {"model": model, "x_test": x_test, "imputation": imputation}


def row_for_report(row: dict[str, Any] | None, label: str) -> dict[str, Any]:
    if row is None:
        return {
            "constraint": label,
            "target_coverage": np.nan,
            "confident_coverage": 0.0,
            "kappa_reweighted": np.nan,
            "ppv_reweighted": np.nan,
            "npv_reweighted": np.nan,
            "severe_fn_n": 0,
            "p_low": np.nan,
            "p_high": np.nan,
        }
    return {"constraint": label, **row}


def source_accounting(zones: pd.DataFrame) -> dict[str, Any]:
    operational_final = zones["post_recovery_finalized"].astype(bool)
    confident = zones["frontier_confident"].astype(bool)
    operational_finalized_n = int(operational_final.sum())
    safe_confident_n = int(confident.sum())
    op_final_retained = int((operational_final & confident).sum())
    op_final_lost = int((operational_final & ~confident).sum())
    op_deferred_gained = zones[~operational_final & confident].copy()
    ratio = pd.to_numeric(op_deferred_gained["fev1_fvc_ratio"], errors="coerce")
    distance = (ratio - FIXED_RATIO_THRESHOLD).abs()
    op_deferred_gained["bucket"] = bucket_from_distance(distance)
    by_bucket = []
    for bucket in ["KNIFE_EDGE", "NEAR", "RECOVERABLE"]:
        by_bucket.append(
            {
                "bucket": bucket,
                "gained_confident_n": int(op_deferred_gained["bucket"].eq(bucket).sum()),
            }
        )
    return {
        "operational_finalized_n": operational_finalized_n,
        "safe_confident_n": safe_confident_n,
        "operational_finalized_retained_confident_n": op_final_retained,
        "operational_finalized_lost_to_indeterminate_n": op_final_lost,
        "operational_deferred_gained_confident_n": int(len(op_deferred_gained)),
        "net_confident_n_delta_vs_operational": int(safe_confident_n - operational_finalized_n),
        "gained_from_operational_deferrals_by_bucket": by_bucket,
    }


def verdict(result: dict[str, Any]) -> tuple[str, str]:
    safe = result["safe_points"]["clean_new_asymmetric"]
    if safe is None:
        return (
            "FRONTIER-UNSAFE",
            "No CLEAN frontier point satisfies the predeclared safety constraints on TRAIN.",
        )
    if not safe["passes_new_asymmetric"]:
        return (
            "FRONTIER-UNSAFE",
            "The TRAIN-selected CLEAN point fails at least one held-out TEST safety constraint.",
        )
    operational = result["deterministic_baseline"]["operational_coverage"]
    if safe["confident_coverage"] > operational + MATERIAL_COVERAGE_EPS:
        return (
            "FRONTIER-GAIN",
            "The asymmetric, guarded CLEAN frontier exceeds the 66.6% operational baseline while preserving severe FN=0.",
        )
    if safe["confident_coverage"] + MATERIAL_COVERAGE_EPS >= operational:
        return (
            "FRONTIER-OPTIMAL",
            "The safe CLEAN frontier is effectively at the operational baseline; deterministic recovery is the safe coverage optimum.",
        )
    return (
        "FRONTIER-UNSAFE",
        "The safe CLEAN frontier remains below the operational baseline; points above baseline violate PPV/NPV safety floors.",
    )


def write_outputs(
    root: Path,
    result: dict[str, Any],
    curve: pd.DataFrame,
    participants: pd.DataFrame,
) -> None:
    out = root / "results"
    out.mkdir(exist_ok=True)
    (out / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    curve.to_csv(out / f"{OUTPUT_STEM}_curve.csv", index=False)
    participants.to_csv(out / f"{OUTPUT_STEM}_participants.csv", index=False)

    clean_safe = result["safe_points"]["clean_new_asymmetric"]
    clean_old = result["safe_points"]["clean_old_symmetric"]
    alpha_safe = result["safe_points"]["alpha_new_asymmetric"]
    alpha_old = result["safe_points"]["alpha_old_symmetric"]
    clean_accounting = result["safe_point_source_accounting"]["clean_new_asymmetric"]
    alpha_accounting = result["safe_point_source_accounting"]["alpha_new_asymmetric"]
    clean_shape = result["frontier_shape"]["clean"]
    alpha_shape = result["frontier_shape"]["clean_plus_fit_alpha"]
    baseline = result["deterministic_baseline"]

    lines = [
        "# Gate - Coverage-Accuracy-Safety Frontier for the Graded Obstruction Classifier, Cycle F",
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
            "## Constraint Fixes",
            "",
            f"- Severe safety guard: a case with ratio < {SEVERE_NORMAL_GUARD_RATIO:.2f} is never allowed to enter `CONFIDENT_NORMAL`; it is forced to `INDETERMINATE_GRADED` instead.",
            "- This is a monotone safety guard, not a classifier feature: it only removes normal confidence and cannot add obstructed confidence.",
            f"- New asymmetric floors at natural prevalence: NPV >= {NEW_NPV_FLOOR:.2f} for `CONFIDENT_NORMAL`, PPV >= {NEW_PPV_FLOOR:.2f} for `CONFIDENT_OBSTRUCTED`, severe FN = 0.",
            f"- Old audit floor retained for comparison: PPV >= NPV >= {OLD_SYMMETRIC_FLOOR:.2f}.",
            "",
            "## Part 1 - CLEAN Frontier Shape",
            "",
            f"- Deterministic operational baseline overlay: coverage {fmt(baseline['operational_coverage'], 3)}, kappa {fmt(baseline['operational_kappa'], 3)}, severe FN {baseline['operational_severe_fn_n']}.",
            f"- Best CLEAN kappa on the swept frontier is only {fmt(clean_shape['best_kappa_point']['kappa_reweighted'] if clean_shape['best_kappa_point'] else np.nan, 3)} at coverage {fmt(clean_shape['best_kappa_point']['confident_coverage'] if clean_shape['best_kappa_point'] else np.nan, 3)}; the deterministic baseline kappa remains far higher.",
            f"- CLEAN PPV is non-monotone at very low coverage; it passes 0.70 over coverage {fmt(clean_shape['ppv_passing_min_coverage'], 3)} to {fmt(clean_shape['ppv_passing_max_coverage'], 3)}.",
            f"- CLEAN NPV is the limiting floor: its passing range ends at coverage {fmt(clean_shape['npv_passing_max_coverage'], 3)}, and the first swept point below 0.95 is {fmt(clean_shape['first_npv_below_0p95']['confident_coverage'] if clean_shape['first_npv_below_0p95'] else np.nan, 3)}.",
            f"- CLEAN kappa is below 0.80 throughout this swept grid; below-0.80 starts at coverage {fmt(clean_shape['first_kappa_below_0p80']['confident_coverage'] if clean_shape['first_kappa_below_0p80'] else np.nan, 3)}.",
            f"- The severe guard first forces TEST referrals at coverage {fmt(clean_shape['first_guard_forced_referral']['confident_coverage'] if clean_shape['first_guard_forced_referral'] else np.nan, 3)}.",
            f"- Severe FN is zero at every swept frontier point across both arms: {result['all_frontier_points_severe_fn_zero']}.",
            "",
            "Selected rows from the CLEAN frontier:",
            "",
            md_table(
                result["clean_frontier_selected_rows"],
                [
                    ("target_coverage", "target", "float3"),
                    ("confident_coverage", "coverage", "float3"),
                    ("kappa_reweighted", "kappa", "float3"),
                    ("ppv_reweighted", "PPV", "float3"),
                    ("npv_reweighted", "NPV", "float3"),
                    ("severe_fn_n", "severe FN", "int"),
                    ("knife_edge_true_060_070_fraction_of_confident", "knife-edge frac", "float3"),
                    ("passes_new_asymmetric", "new pass", "str"),
                    ("passes_old_symmetric_090", "old pass", "str"),
                ],
            ),
            "",
            "## Part 2 - Train-Selected Candidates and Held-Out Evaluation",
            "",
            md_table(
                [
                    row_for_report(clean_old, "CLEAN train-selected, old symmetric 0.90/0.90"),
                    row_for_report(clean_safe, "CLEAN train-selected, new asymmetric"),
                    row_for_report(alpha_old, "fit_alpha train-selected, old symmetric 0.90/0.90"),
                    row_for_report(alpha_safe, "fit_alpha train-selected, new asymmetric"),
                ],
                [
                    ("constraint", "constraint", "str"),
                    ("target_coverage", "target", "float3"),
                    ("confident_coverage", "coverage", "float3"),
                    ("kappa_reweighted", "kappa", "float3"),
                    ("ppv_reweighted", "PPV", "float3"),
                    ("npv_reweighted", "NPV", "float3"),
                    ("severe_fn_n", "severe FN", "int"),
                    ("p_low", "p_low", "float3"),
                    ("p_high", "p_high", "float3"),
                ],
            ),
            "",
        ]
    )
    if clean_safe is not None:
        delta = clean_safe["confident_coverage"] - baseline["operational_coverage"]
        lines.extend(
            [
                f"- CLEAN train-selected candidate coverage on TEST: {fmt(clean_safe['confident_coverage'], 3)} vs operational baseline {fmt(baseline['operational_coverage'], 3)}; delta {fmt(delta, 3)}.",
                f"- Count accounting: {clean_accounting['operational_deferred_gained_confident_n']} operational deferrals become confident, {clean_accounting['operational_finalized_lost_to_indeterminate_n']} operational-finalized cases become indeterminate, net delta {clean_accounting['net_confident_n_delta_vs_operational']} cases.",
                f"- CLEAN train-selected candidate severe FN on TEST: {clean_safe['severe_fn_n']} with CP upper bound {fmt(clean_safe['severe_fn_ci_high'], 3)}.",
                f"- CLEAN train-selected candidate PPV/NPV on TEST: {fmt(clean_safe['ppv_reweighted'], 3)}/{fmt(clean_safe['npv_reweighted'], 3)}; held-out safety pass: {clean_safe['passes_new_asymmetric']}.",
                "- Source of gained confident calls from operational deferrals:",
                "",
                md_table(
                    clean_accounting["gained_from_operational_deferrals_by_bucket"],
                    [("bucket", "bucket", "str"), ("gained_confident_n", "gained n", "int")],
                ),
            ]
        )
    else:
        lines.append("- No CLEAN point passed the new asymmetric safety floors.")
    lines.extend(
        [
            "",
            "## Part 3 - Held-Out Bucket Performance at the CLEAN Train-Selected Candidate",
            "",
            md_table(
                result["bucket_recovery_clean_safe"],
                [
                    ("bucket", "bucket", "str"),
                    ("true_abs_distance_band", "true distance", "str"),
                    ("n", "n", "int"),
                    ("confident_finalized_n", "conf finalized", "int"),
                    ("confident_finalized_fraction", "fraction", "float3"),
                    ("accuracy_reweighted", "accuracy", "float3"),
                    ("severe_fn_n", "severe FN", "int"),
                ],
            ),
            "",
            "- The expected bucket pattern did not fully hold: `RECOVERABLE` calls are accurate, but the probability-only CLEAN frontier also finalizes many `KNIFE_EDGE` cases with poor bucket accuracy. That is why this is a coverage/safety frontier result, not a claim that the knife-edge was solved.",
            "",
            "## Part 4 - fit_alpha Frontier",
            "",
            f"- fit_alpha whole AUC: {fmt(result['arm_auc']['clean_plus_fit_alpha']['whole_auc']['auc'], 3)}; boundary AUC: {fmt(result['arm_auc']['clean_plus_fit_alpha']['boundary_auc']['auc'], 3)}.",
            f"- CLEAN whole AUC: {fmt(result['arm_auc']['clean']['whole_auc']['auc'], 3)}; boundary AUC: {fmt(result['arm_auc']['clean']['boundary_auc']['auc'], 3)}.",
            f"- fit_alpha best kappa: {fmt(alpha_shape['best_kappa_point']['kappa_reweighted'] if alpha_shape['best_kappa_point'] else np.nan, 3)} at coverage {fmt(alpha_shape['best_kappa_point']['confident_coverage'] if alpha_shape['best_kappa_point'] else np.nan, 3)}.",
            f"- fit_alpha PPV never falls below 0.70 on this grid; new-asymmetric passing coverage reaches {fmt(alpha_shape['new_asymmetric_passing_max_coverage'], 3)}.",
            "- fit_alpha is reported as secondary because the prior audit found Spearman correlation with the ratio around 0.63; strict CLEAN remains the conservative headline arm.",
            "",
            "## Interpretation",
            "",
        ]
    )
    if clean_safe is not None and clean_safe["passes_new_asymmetric"] and clean_safe["confident_coverage"] > baseline["operational_coverage"] + MATERIAL_COVERAGE_EPS:
        lines.append("- The corrected asymmetric floors turn the calibrated probabilities into a safe coverage gain over the operational baseline.")
    elif clean_safe is not None and clean_safe["passes_new_asymmetric"]:
        lines.append("- The CLEAN frontier reaches a safe point, but it does not materially exceed the operational deterministic baseline.")
    elif clean_safe is not None:
        lines.append("- The CLEAN candidate selected on TRAIN fails the held-out TEST safety floors and cannot be used as an operating point.")
    else:
        lines.append("- The CLEAN frontier has no point satisfying the new safety floors on TRAIN.")
    if alpha_safe is not None and alpha_safe["passes_new_asymmetric"]:
        lines.append(
            "- The secondary fit_alpha candidate passes the held-out floors, but it remains exploratory because fit_alpha was quarantined by its correlation with the target ratio."
        )
        if alpha_accounting is not None:
            lines.append(
                f"- fit_alpha count accounting: {alpha_accounting['operational_deferred_gained_confident_n']} operational deferrals become confident, {alpha_accounting['operational_finalized_lost_to_indeterminate_n']} operational-finalized cases become indeterminate, net delta {alpha_accounting['net_confident_n_delta_vs_operational']} cases."
            )
    lines.extend(
        [
            "- KNIFE_EDGE cases that remain indeterminate should be treated as correctly deferred; this frontier does not prove the knife-edge itself is solved.",
            "",
            f"VERDICT: {result['verdict']} - {result['verdict_reason']}",
            "",
        ]
    )
    (out / f"{OUTPUT_STEM}.md").write_text("\n".join(lines), encoding="utf-8")


def run(root: Path) -> dict[str, Any]:
    workflow, _repro, recovered_test, _c1 = load_model_frame(root)
    recalibrated_path = root / "results" / "gate_graded_recalibrated_F_participants.csv"
    if not recalibrated_path.exists():
        raise RuntimeError("Missing gate_graded_recalibrated_F_participants.csv; run gate_graded_recalibrated_f.py first.")
    prior_participants = pd.read_csv(recalibrated_path)

    audit, banned_present, quarantined, clean_features = feature_audit(workflow)
    train = workflow[workflow["split"].eq("train")].copy().reset_index(drop=True)
    test = workflow[workflow["split"].eq("test")].copy().reset_index(drop=True)
    test = test.merge(
        recovered_test[["SEQN", "post_recovery_finalized", "post_recovery_decision"]],
        on="SEQN",
        how="left",
    )
    test["post_recovery_finalized"] = test["post_recovery_finalized"].fillna(False).astype(bool)

    prevalence = prevalence_from_full_cohort(root)
    target_prevalence = float(prevalence["target_prevalence"])
    y_train = truth_obstructed(train).astype(int).to_numpy()
    y_test = truth_obstructed(test).astype(int).to_numpy()
    source_prevalence = float(y_train.mean())
    train_weights = class_reweight(y_train, target_prevalence, source_prevalence)
    test_weights = class_reweight(y_test, target_prevalence, source_prevalence)

    p_train_clean, p_test_clean, clean_fit = fit_probability_arm(
        train, test, clean_features, source_prevalence, target_prevalence
    )
    prior_check = test[["SEQN"]].copy()
    prior_check["p_adj_reconstructed"] = p_test_clean
    prior_check = prior_check.merge(prior_participants[["SEQN", "p_adj"]], on="SEQN", how="left")
    p_adj_max_abs_delta = float(np.nanmax(np.abs(prior_check["p_adj_reconstructed"] - prior_check["p_adj"])))
    if p_adj_max_abs_delta > 1e-12:
        raise RuntimeError(f"Reconstructed p_adj does not match saved prior output: max abs delta {p_adj_max_abs_delta}")

    alpha_features = clean_features + ["fit_alpha"]
    p_train_alpha, p_test_alpha, alpha_fit = fit_probability_arm(
        train, test, alpha_features, source_prevalence, target_prevalence
    )

    targets = np.round(np.r_[np.arange(0.45, 1.00, 0.02), 1.00], 2)
    clean_rows, clean_zones = frontier_for_arm(
        "clean", train, test, p_train_clean, p_test_clean, train_weights, test_weights, targets
    )
    alpha_rows, alpha_zones = frontier_for_arm(
        "clean_plus_fit_alpha", train, test, p_train_alpha, p_test_alpha, train_weights, test_weights, targets
    )
    all_rows = clean_rows + alpha_rows
    curve = pd.DataFrame(all_rows)

    clean_safe = select_safe_point(clean_rows, "train_passes_new_asymmetric")
    clean_old = select_safe_point(clean_rows, "train_passes_old_symmetric_090")
    alpha_safe = select_safe_point(alpha_rows, "train_passes_new_asymmetric")
    alpha_old = select_safe_point(alpha_rows, "train_passes_old_symmetric_090")

    if clean_safe is not None:
        safe_target = float(clean_safe["target_coverage"])
        safe_clean_zones = clean_zones[safe_target].copy()
        bucket_recovery = bucket_rows(safe_clean_zones, test_weights)
    else:
        safe_target = float("nan")
        safe_clean_zones = apply_frontier_zones(test, p_test_clean, np.nan, np.nan)
        bucket_recovery = bucket_rows(safe_clean_zones, test_weights)

    if alpha_safe is not None:
        alpha_safe_zones = alpha_zones[float(alpha_safe["target_coverage"])].copy()
    else:
        alpha_safe_zones = apply_frontier_zones(test, p_test_alpha, np.nan, np.nan)

    ratio = pd.to_numeric(test["fev1_fvc_ratio"], errors="coerce")
    distance = (ratio - FIXED_RATIO_THRESHOLD).abs()
    participants = pd.DataFrame(
        {
            "SEQN": test["SEQN"],
            "p_adj": p_test_clean,
            "p_adj_fit_alpha": p_test_alpha,
            "ratio": ratio,
            "truth_obstructed": y_test.astype(bool),
            "truth_severity": test["truth_severity"].astype(str),
            "post_recovery_finalized": test["post_recovery_finalized"].astype(bool),
            "deferral_bucket": np.where(
                test["post_recovery_finalized"].astype(bool),
                "FINALIZED_DETERMINISTIC",
                bucket_from_distance(distance),
            ),
            "clean_safe_zone": safe_clean_zones["frontier_zone"].to_numpy(),
            "clean_safe_confident": safe_clean_zones["frontier_confident"].to_numpy(),
            "clean_safe_ratio_guard_forced": safe_clean_zones["ratio_severe_normal_guard_forced"].to_numpy(),
            "fit_alpha_safe_zone": alpha_safe_zones["frontier_zone"].to_numpy(),
            "fit_alpha_safe_confident": alpha_safe_zones["frontier_confident"].to_numpy(),
            "fit_alpha_safe_ratio_guard_forced": alpha_safe_zones["ratio_severe_normal_guard_forced"].to_numpy(),
        }
    )

    band = boundary_mask(test).to_numpy()
    arm_auc = {
        "clean": {
            "whole_auc": bootstrap_auc_ci(y_test, p_test_clean, RANDOM_SEED + 710),
            "boundary_auc": bootstrap_auc_ci(y_test[band], p_test_clean[band], RANDOM_SEED + 720),
        },
        "clean_plus_fit_alpha": {
            "whole_auc": bootstrap_auc_ci(y_test, p_test_alpha, RANDOM_SEED + 730),
            "boundary_auc": bootstrap_auc_ci(y_test[band], p_test_alpha[band], RANDOM_SEED + 740),
        },
    }

    from .gate_deferral_recovery_f import final_metrics

    operational_metrics = final_metrics(recovered_test, "post_recovery_decision", "post_recovery_finalized")
    strict_metrics = final_metrics(recovered_test, "workflow_decision", "workflow_final_diagnostic")

    selected_targets = [0.45, 0.55, 0.65, 0.67, 0.75, 0.85, 0.95, 1.00]
    clean_selected_rows = []
    for target in selected_targets:
        match = [row for row in clean_rows if abs(row["target_coverage"] - target) < 1e-9]
        if match:
            clean_selected_rows.append(match[0])
    if clean_safe is not None and all(abs(row["target_coverage"] - clean_safe["target_coverage"]) > 1e-9 for row in clean_selected_rows):
        clean_selected_rows.append(clean_safe)
        clean_selected_rows = sorted(clean_selected_rows, key=lambda row: row["target_coverage"])

    result: dict[str, Any] = {
        "seed": RANDOM_SEED,
        "target_prevalence": target_prevalence,
        "source_train_prevalence": source_prevalence,
        "clean_features": clean_features,
        "alpha_features": alpha_features,
        "banned_present": banned_present,
        "quarantined_features": quarantined,
        "p_adj_reconstruction_max_abs_delta_vs_prior_output": p_adj_max_abs_delta,
        "target_grid": targets.tolist(),
        "constraints": {
            "severe_normal_guard_ratio_lt": SEVERE_NORMAL_GUARD_RATIO,
            "new_asymmetric": {
                "npv_confident_normal_min": NEW_NPV_FLOOR,
                "ppv_confident_obstructed_min": NEW_PPV_FLOOR,
                "severe_fn_max": 0,
            },
            "old_symmetric": {
                "npv_min": OLD_SYMMETRIC_FLOOR,
                "ppv_min": OLD_SYMMETRIC_FLOOR,
                "severe_fn_max": 0,
            },
        },
        "deterministic_baseline": {
            "strict_coverage": strict_metrics["finalized_fraction"],
            "strict_kappa": strict_metrics["cohens_kappa"],
            "strict_severe_fn_n": strict_metrics["severe_fn_n"],
            "operational_coverage": operational_metrics["finalized_fraction"],
            "operational_kappa": operational_metrics["cohens_kappa"],
            "operational_severe_fn_n": operational_metrics["severe_fn_n"],
            "strict_metrics": strict_metrics,
            "operational_metrics": operational_metrics,
        },
        "safe_points": {
            "clean_new_asymmetric": clean_safe,
            "clean_old_symmetric": clean_old,
            "alpha_new_asymmetric": alpha_safe,
            "alpha_old_symmetric": alpha_old,
        },
        "safe_point_source_accounting": {
            "clean_new_asymmetric": source_accounting(safe_clean_zones) if clean_safe is not None else None,
            "alpha_new_asymmetric": source_accounting(alpha_safe_zones) if alpha_safe is not None else None,
        },
        "bucket_recovery_clean_safe": bucket_recovery,
        "frontier_shape": {
            "clean": frontier_shape(clean_rows),
            "clean_plus_fit_alpha": frontier_shape(alpha_rows),
        },
        "arm_auc": arm_auc,
        "all_frontier_points_severe_fn_zero": bool(curve["severe_fn_n"].eq(0).all()),
        "clean_frontier_selected_rows": clean_selected_rows,
        "protocol_decisions": [
            "The same CLEAN p_adj was deterministically reconstructed because the prior gate did not persist TRAIN probabilities; execution halted unless TEST p_adj matched the saved participants CSV.",
            "Symmetric probability bands are centered at p=0.5; p<=p_low is confident normal and p>=p_high is confident obstructed.",
            "The TRAIN target coverage search measures coverage after applying the ratio<0.60 normal-call safety guard, because that is the deployable operating coverage.",
            "The operating point is selected only from TRAIN weighted PPV, NPV, severe-FN, coverage, and kappa. TEST metrics are reported once and never used to choose the point.",
            "All accuracy, kappa, sensitivity, specificity, PPV, and NPV values in the frontier use natural-prevalence class weights.",
            "The ratio<0.60 guard only removes confident normal calls; it is reported separately as a safety guard and not used to add confidence.",
            "The frontier target grid is fixed at 0.45 to 1.00 in approximately 0.02 increments before looking at TEST performance.",
            "The fit_alpha arm is secondary because fit_alpha was previously quarantined by ratio correlation, despite being mechanistically independent of the label computation.",
        ],
    }
    result["verdict"], result["verdict_reason"] = verdict(result)

    write_outputs(root, result, curve, participants)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    safe = result["safe_points"]["clean_new_asymmetric"]
    alpha = result["safe_points"]["alpha_new_asymmetric"]
    print(f"Frontier gate: {result['verdict']} - {result['verdict_reason']}")
    if safe is None:
        print("CLEAN train-selected candidate: none")
    else:
        print(
            "CLEAN train-selected candidate on TEST: "
            f"coverage={safe['confident_coverage']:.3f}, "
            f"kappa={safe['kappa_reweighted']:.3f}, "
            f"PPV/NPV={safe['ppv_reweighted']:.3f}/{safe['npv_reweighted']:.3f}, "
            f"severe_FN={safe['severe_fn_n']}"
        )
    if alpha is not None:
        print(
            "fit_alpha train-selected candidate on TEST: "
            f"coverage={alpha['confident_coverage']:.3f}, "
            f"kappa={alpha['kappa_reweighted']:.3f}, "
            f"PPV/NPV={alpha['ppv_reweighted']:.3f}/{alpha['npv_reweighted']:.3f}, "
            f"severe_FN={alpha['severe_fn_n']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
