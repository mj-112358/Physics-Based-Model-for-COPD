from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .dataset_resolver import resolve_dataset_path
from .gate3_diagnosis import clinical_metrics_from_volume, np_ombe_volume, savgol_sigma
from .gate23_f import HZ, backextrap_t0, detect_forced_exhale_segment, find_or_unzip_raw, parse_increments, read_raw
from .gate_clinical_classification_f import (
    RANDOM_SEED,
    confusion_binary,
    decode_grade,
    extract_curve_for_fit,
    load_column_map,
    load_truth,
    q,
    severity_category,
)
from .gate_policy_calibrated_flag_f import (
    stratified_half_split,
    three_outcome_metrics,
)
from .gate_reconciliation_audit_f import clean_json, md_table, median_iqr

PART_A_STEM = "gate_c1_artifact_F"
PART_B_STEM = "gate_pooling_disagreement_F"
PART_C_STEM = "gate_recovery_gli_F"
BOUNDS_LO = np.array([0.15, 1.00, 0.25, 2.00, 0.03], dtype=float)
BOUNDS_HI = np.array([0.80, 5.00, 0.85, 6.00, 0.40], dtype=float)
PARAM_NAMES = ("tau_f", "tau_s", "alpha", "fvc_l", "tau_o")


def pct_error(est: pd.Series | np.ndarray, truth: pd.Series | np.ndarray) -> np.ndarray:
    est = np.asarray(est, dtype=float)
    truth = np.asarray(truth, dtype=float)
    out = np.full_like(est, np.nan, dtype=float)
    ok = np.isfinite(est) & np.isfinite(truth) & (truth != 0)
    out[ok] = 100.0 * (est[ok] - truth[ok]) / truth[ok]
    return out


def signed_summary(values: pd.Series | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return {
        "n": int(len(arr)),
        "median": float(np.median(arr)) if len(arr) else np.nan,
        "iqr": q(arr) if len(arr) else [np.nan, np.nan],
        "iqr_crosses_zero": bool(len(arr) and np.percentile(arr, 25) <= 0.0 <= np.percentile(arr, 75)),
    }


def load_gli_reference(root: Path) -> pd.DataFrame:
    path = root / "results" / "gli_reference.csv"
    if not path.exists():
        return pd.DataFrame(columns=["SEQN", "gli_lln_ratio"])
    return pd.read_csv(path)


def raw_by_index(raw: pd.DataFrame) -> dict[int, Any]:
    return {int(idx): row for idx, row in raw.iterrows()}


def curve_variant_metrics(row: Any) -> dict[str, float]:
    incr = parse_increments(row.SPXRAW)
    fact = float(row.SPAFACT)
    time = np.arange(len(incr), dtype=float) / HZ

    def build(scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        flow = incr * 0.1 * scale
        volume_ml = np.cumsum(incr) * scale
        onset_idx, pef_idx, end_idx = detect_forced_exhale_segment(flow, hz=HZ)
        t0_s = backextrap_t0(time, flow, volume_ml, onset_idx, pef_idx)
        return flow, volume_ml, np.array([t0_s], dtype=float), end_idx

    _flow, vol_ml, t0_arr, end_idx = build(fact)
    t0_s = float(t0_arr[0])
    v0_ml = float(np.interp(t0_s, time, vol_ml))
    fev1_btps_mL = float(np.interp(t0_s + 1.0, time, vol_ml) - v0_ml)
    segmented_fvc_btps_mL = float(vol_ml[end_idx] - v0_ml)
    max_fvc_btps_mL = float(np.nanmax(vol_ml[time >= max(t0_s, 0.0)]) - v0_ml)
    final_fvc_btps_mL = float(vol_ml[-1] - v0_ml)

    _flow0, vol0_ml, t0_no_arr, end_no = build(1.0)
    t0_no = float(t0_no_arr[0])
    v0_no = float(np.interp(t0_no, time, vol0_ml))
    fev1_no_mL = float(np.interp(t0_no + 1.0, time, vol0_ml) - v0_no)
    fvc_no_mL = float(vol0_ml[end_no] - v0_no)
    return {
        "spafact": fact,
        "segmented_btps_fev1_mL": fev1_btps_mL,
        "segmented_btps_fvc_mL": segmented_fvc_btps_mL,
        "segmented_btps_ratio": fev1_btps_mL / segmented_fvc_btps_mL if segmented_fvc_btps_mL > 0 else np.nan,
        "segmented_no_btps_fev1_mL": fev1_no_mL,
        "segmented_no_btps_fvc_mL": fvc_no_mL,
        "segmented_no_btps_ratio": fev1_no_mL / fvc_no_mL if fvc_no_mL > 0 else np.nan,
        "max_cumulative_btps_fvc_mL": max_fvc_btps_mL,
        "max_cumulative_btps_ratio": fev1_btps_mL / max_fvc_btps_mL if max_fvc_btps_mL > 0 else np.nan,
        "final_record_btps_fvc_mL": final_fvc_btps_mL,
        "final_record_btps_ratio": fev1_btps_mL / final_fvc_btps_mL if final_fvc_btps_mL > 0 else np.nan,
    }


def summarize_clinical_errors(rows: pd.DataFrame, group_name: str) -> list[dict[str, Any]]:
    out = []
    specs = [
        ("observed", "FEV1", rows["parsed_fev1_l"] * 1000.0, rows["fev1_mL"]),
        ("observed", "FVC", rows["parsed_fvc_l"] * 1000.0, rows["fvc_mL"]),
        ("observed", "ratio", rows["parsed_ratio"], rows["fev1_fvc_ratio"]),
        ("model", "FEV1", rows["model_fev1_mL"], rows["fev1_mL"]),
        ("model", "FVC", rows["model_fvc_mL"], rows["fvc_mL"]),
        ("model", "ratio", rows["model_ratio"], rows["fev1_fvc_ratio"]),
    ]
    for source, metric, est, truth in specs:
        summary = signed_summary(pct_error(est, truth))
        out.append(
            {
                "row_type": "part_a_a1_error_summary",
                "set": group_name,
                "source": source,
                "metric": metric,
                "signed_pct_median": summary["median"],
                "signed_pct_iqr_low": summary["iqr"][0],
                "signed_pct_iqr_high": summary["iqr"][1],
                "iqr_crosses_zero": summary["iqr_crosses_zero"],
                "n": summary["n"],
            }
        )
    return out


def fixed_confusion(df: pd.DataFrame, ratio_col: str, threshold_col: str | None = None) -> dict[str, Any]:
    if threshold_col is None:
        truth = df["fev1_fvc_ratio"] < 0.70
        pred = df[ratio_col] < 0.70
    else:
        valid = df[threshold_col].notna() & np.isfinite(df[threshold_col])
        df = df[valid].copy()
        truth = df["fev1_fvc_ratio"] < df[threshold_col]
        pred = df[ratio_col] < df[threshold_col]
    out = confusion_binary(truth, pred)
    obs = df[truth].copy()
    out["obstructed_signed_ratio_error_median"] = float(np.nanmedian(df.loc[truth, ratio_col] - df.loc[truth, "fev1_fvc_ratio"])) if len(obs) else np.nan
    borderline = df[(df["fev1_fvc_ratio"] >= 0.60) & (df["fev1_fvc_ratio"] < 0.70)]
    if len(borderline):
        threshold = 0.70 if threshold_col is None else borderline[threshold_col]
        out["borderline_false_negative_rate"] = float((borderline[ratio_col] >= threshold).mean())
        out["borderline_n"] = int(len(borderline))
    else:
        out["borderline_false_negative_rate"] = np.nan
        out["borderline_n"] = 0
    return out


def run_part_a(root: Path, raw: pd.DataFrame, truth: pd.DataFrame, cycle: str = "F") -> dict[str, Any]:
    results_dir = root / "results"
    clinical = pd.read_csv(results_dir / "gate_clinical_classification_F_rows.csv")
    policy = pd.read_csv(results_dir / "gate_policy_calibrated_flag_F_rows.csv")
    policy = policy[policy["split"] == "test"].copy()
    with (results_dir / "gate_policy_calibrated_flag_F.json").open() as f:
        policy_json = json.load(f)
    c1 = float(policy_json["corrections"]["c1"]["bias"])

    row_records: list[dict[str, Any]] = []
    for group_name, group in [
        ("obstructed", clinical[clinical["sample_group"] == "obstructed"].copy()),
        ("normal", clinical[clinical["sample_group"] == "normal"].copy()),
    ]:
        row_records.extend(summarize_clinical_errors(group, group_name))

    clinical["model_fvc_vs_observed_fvc_pct_error"] = pct_error(clinical["model_fvc_mL"], clinical["parsed_fvc_l"] * 1000.0)
    model_vs_observed = []
    for group_name, group in [
        ("obstructed", clinical[clinical["sample_group"] == "obstructed"].copy()),
        ("normal", clinical[clinical["sample_group"] == "normal"].copy()),
    ]:
        summary = signed_summary(group["model_fvc_vs_observed_fvc_pct_error"])
        item = {
            "row_type": "part_a_a2_model_vs_observed",
            "set": group_name,
            "metric": "model_fvc_vs_observed_fvc_pct_error",
            "signed_pct_median": summary["median"],
            "signed_pct_iqr_low": summary["iqr"][0],
            "signed_pct_iqr_high": summary["iqr"][1],
            "iqr_crosses_zero": summary["iqr_crosses_zero"],
            "n": summary["n"],
        }
        model_vs_observed.append(item)
        row_records.append(item)

    obstructed_error = next(
        r
        for r in row_records
        if r["row_type"] == "part_a_a1_error_summary"
        and r["set"] == "obstructed"
        and r["source"] == "observed"
        and r["metric"] == "FVC"
    )
    model_observed_error = next(r for r in model_vs_observed if r["set"] == "obstructed")
    model_fvc_error = next(
        r
        for r in row_records
        if r["row_type"] == "part_a_a1_error_summary"
        and r["set"] == "obstructed"
        and r["source"] == "model"
        and r["metric"] == "FVC"
    )
    model_faithful = (
        abs(float(model_observed_error["signed_pct_median"])) < 2.0
        and float(obstructed_error["signed_pct_median"]) < -3.0
        and float(obstructed_error["signed_pct_iqr_high"]) < 0.0
        and float(model_fvc_error["signed_pct_median"]) < -3.0
        and float(model_fvc_error["signed_pct_iqr_high"]) < 0.0
    )
    mechanism = "MODEL-FAITHFUL" if model_faithful else "MODEL-BIASED"
    mechanism_sentence = (
        "MODEL-FAITHFUL: model tracks observed FVC closely and both observed and model FVC undershoot NHANES; c1 corrects a raw-curve-versus-NHANES definitional gap, not a bi-exponential tail defect."
        if model_faithful
        else "MODEL-BIASED: model undershoots the observed curve materially; the tail-undershoot mechanism remains plausible."
    )

    raw_cols = list(raw.columns)
    spx_path = resolve_dataset_path(root, "SPX", cycle).path
    spx_cols = list(pd.read_sas(spx_path, format="xport").columns) if spx_path.exists() else []
    value_fields = [c for c in raw_cols + spx_cols if any(key in c.upper() for key in ["FEV", "FVC", "PEF", "FET", "EOT", "BTPS", "SPAF", "SPXN"])]

    raw_map = raw_by_index(raw)
    variant_rows = []
    for row in clinical.itertuples(index=False):
        try:
            variants = curve_variant_metrics(raw_map[int(row.raw_index)])
        except Exception as exc:
            variants = {"variant_error": str(exc)}
        base = {
            "row_type": "part_a_a3_curve_variant",
            "SEQN": float(row.SEQN),
            "raw_index": int(row.raw_index),
            "set": row.sample_group,
            "nhanes_fev1_mL": float(row.fev1_mL),
            "nhanes_fvc_mL": float(row.fvc_mL),
            "nhanes_ratio": float(row.fev1_fvc_ratio),
        }
        base.update(variants)
        variant_rows.append(base)
    row_records.extend(variant_rows)
    variant_df = pd.DataFrame(variant_rows)
    variant_summary = []
    for name in ["segmented_btps", "segmented_no_btps", "max_cumulative_btps", "final_record_btps"]:
        fvc_col = f"{name}_fvc_mL"
        ratio_col = f"{name}_ratio"
        if fvc_col not in variant_df:
            continue
        summary = signed_summary(pct_error(variant_df[fvc_col], variant_df["nhanes_fvc_mL"]))
        ratio_summary = signed_summary(pct_error(variant_df[ratio_col], variant_df["nhanes_ratio"])) if ratio_col in variant_df else {}
        item = {
            "row_type": "part_a_a3_variant_summary",
            "variant": name,
            "fvc_signed_pct_median": summary["median"],
            "fvc_signed_pct_iqr_low": summary["iqr"][0],
            "fvc_signed_pct_iqr_high": summary["iqr"][1],
            "ratio_signed_pct_median": ratio_summary.get("median", np.nan),
            "n": summary["n"],
        }
        variant_summary.append(item)
        row_records.append(item)

    policy = policy.copy()
    policy["ratio_uncorrected"] = policy["model_ratio"]
    policy["ratio_c1"] = policy["model_ratio"] - c1
    metrics = []
    for label, ratio_col in [("uncorrected", "ratio_uncorrected"), ("c1_corrected", "ratio_c1")]:
        fixed = fixed_confusion(policy, ratio_col)
        fixed.update({"row_type": "part_a_a4_classification", "definition": "fixed_0.70", "ratio_variant": label})
        metrics.append(fixed)
        if "gli_lln_ratio" in policy.columns:
            lln = fixed_confusion(policy, ratio_col, "gli_lln_ratio")
            lln.update({"row_type": "part_a_a4_classification", "definition": "GLI_LLN", "ratio_variant": label})
            metrics.append(lln)
    row_records.extend(metrics)

    verdict = mechanism
    result = {
        "verdict": verdict,
        "mechanism_sentence": mechanism_sentence,
        "c1_bias_from_gate_policy": c1,
        "row_counts": {
            "clinical_obstructed": int((clinical["sample_group"] == "obstructed").sum()),
            "clinical_normal": int((clinical["sample_group"] == "normal").sum()),
            "policy_test": int(len(policy)),
        },
        "a1_error_summaries": [r for r in row_records if r["row_type"] == "part_a_a1_error_summary"],
        "a2_model_vs_observed": model_vs_observed,
        "a3_fields": {
            "raw_columns": raw_cols,
            "spx_f_columns": spx_cols,
            "candidate_value_fields": value_fields,
        },
        "a3_variant_summary": variant_summary,
        "a4_classification": metrics,
        "retraction_statement": (
            "Retract the FVC-underestimate mechanism story from gate_fvc_correction_F if using c1; the observed curve itself undershoots NHANES."
            if model_faithful
            else "Do not retract the FVC tail-undershoot mechanism from gate_fvc_correction_F based on this gate."
        ),
    }
    write_part_a(root, result, pd.DataFrame(row_records))
    return result


def conf_metrics(df: pd.DataFrame, ratio_col: str, flag_col: str | None = None) -> dict[str, Any]:
    flag = pd.Series(False, index=df.index) if flag_col is None else df[flag_col].astype(bool)
    outcome = pd.Series("NORMAL", index=df.index, dtype=object)
    outcome[df[ratio_col] < 0.70] = "OBSTRUCTED"
    outcome[flag] = "FLAG"
    return three_outcome_metrics(df, outcome)


def split_participants(df: pd.DataFrame) -> pd.DataFrame:
    return stratified_half_split(df, "truth_severity")


def pooled_rows(volume_rows: pd.DataFrame, truth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    maneuvers = volume_rows[volume_rows["row_type"] == "maneuver_fit"].copy()
    maneuvers["volume_fit_success"] = maneuvers["volume_fit_success"].map({True: True, False: False, "True": True, "False": False}).fillna(False)
    maneuvers = maneuvers[
        maneuvers["volume_fit_success"]
        & maneuvers["fit_error"].fillna("").eq("")
        & maneuvers[["model_ratio", "model_fev1_l", "model_fvc_l", "observed_ratio", "observed_fev1_l", "observed_fvc_l"]].apply(np.isfinite).all(axis=1)
    ].copy()
    counts = maneuvers.groupby("SEQN").size()
    seqn = set(counts[counts >= 2].index.astype(float))
    maneuvers = maneuvers[maneuvers["SEQN"].astype(float).isin(seqn)].copy()
    maneuvers = maneuvers.merge(truth[["SEQN", "fev1_fvc_ratio"]], on="SEQN", how="left", suffixes=("", "_truth"))
    if "fev1_fvc_ratio_truth" in maneuvers:
        maneuvers["fev1_fvc_ratio"] = maneuvers["fev1_fvc_ratio"].fillna(maneuvers["fev1_fvc_ratio_truth"])
    maneuvers["truth_severity"] = maneuvers["fev1_fvc_ratio"].map(severity_category)
    rows = []
    for seq, group in maneuvers.groupby("SEQN"):
        group = group.sort_values(["SPAMANU", "raw_index"]).copy()
        best = group.sort_values(["observed_fev1_l", "SPAMANU"], ascending=[False, True]).iloc[0]
        spread = float(group["model_ratio"].max() - group["model_ratio"].min())
        observed_spread = float(group["observed_ratio"].max() - group["observed_ratio"].min())
        model_calls = group["model_ratio"] < 0.70
        observed_calls = group["observed_ratio"] < 0.70
        rows.append(
            {
                "SEQN": float(seq),
                "truth_severity": group["truth_severity"].iloc[0],
                "fev1_fvc_ratio": float(group["fev1_fvc_ratio"].iloc[0]),
                "n_maneuvers": int(len(group)),
                "model_ratio_sd": float(group["model_ratio"].std(ddof=1)),
                "observed_ratio_sd": float(group["observed_ratio"].std(ddof=1)),
                "model_spread": spread,
                "observed_spread": observed_spread,
                "model_disagree": bool(model_calls.nunique() > 1),
                "observed_disagree": bool(observed_calls.nunique() > 1),
                "p1_model_ratio": float(best["model_ratio"]),
                "p2_model_ratio": float(group["model_ratio"].median()),
                "p3_model_ratio": float(group["model_fev1_l"].max() / group["model_fvc_l"].max()),
                "p1_observed_ratio": float(best["observed_ratio"]),
                "p2_observed_ratio": float(group["observed_ratio"].median()),
                "p3_observed_ratio": float(group["observed_fev1_l"].max() / group["observed_fvc_l"].max()),
            }
        )
    return pd.DataFrame(rows), maneuvers


def leave_one_out_sd(group: pd.DataFrame, source: str, method: str) -> float:
    if len(group) < 3:
        return np.nan
    vals = []
    for idx in group.index:
        sub = group.drop(index=idx)
        if method == "P1":
            best = sub.sort_values([f"{source}_fev1_l", "SPAMANU"], ascending=[False, True]).iloc[0]
            vals.append(float(best[f"{source}_ratio"]))
        elif method == "P2":
            vals.append(float(sub[f"{source}_ratio"].median()))
        elif method == "P3":
            vals.append(float(sub[f"{source}_fev1_l"].max() / sub[f"{source}_fvc_l"].max()))
    return float(np.std(vals, ddof=1)) if len(vals) >= 2 else np.nan


def run_part_b(root: Path, truth: pd.DataFrame) -> dict[str, Any]:
    results_dir = root / "results"
    volume_rows = pd.read_csv(results_dir / "gate_reproducibility_volume_F_rows.csv", low_memory=False)
    policy_rows = pd.read_csv(results_dir / "gate_policy_calibrated_flag_F_rows.csv")
    participants, maneuvers = pooled_rows(volume_rows, truth)
    participants = participants[participants["fev1_fvc_ratio"].notna()].copy()
    split = split_participants(participants)
    train = split[split["split"] == "train"].copy()
    test = split[split["split"] == "test"].copy()
    threshold = float(np.nanquantile(train["model_spread"], 0.85))
    train["model_spread_flag"] = train["model_spread"] >= threshold
    test["model_spread_flag"] = test["model_spread"] >= threshold
    train["model_disagree_flag"] = train["model_disagree"]
    test["model_disagree_flag"] = test["model_disagree"]

    rows_out: list[dict[str, Any]] = []
    metrics = []
    for method, col in [("P1_best_fev1", "p1_model_ratio"), ("P2_median_ratio", "p2_model_ratio"), ("P3_ATS_largest", "p3_model_ratio")]:
        m = confusion_binary(test["fev1_fvc_ratio"] < 0.70, test[col] < 0.70)
        err = signed_summary(test[col] - test["fev1_fvc_ratio"])
        borderline = test[test["truth_severity"] == "mild_0.60_0.70"]
        item = {
            "row_type": "part_b_b1_pooling_metric",
            "source": "model",
            "method": method,
            **m,
            "signed_ratio_error_median": err["median"],
            "signed_ratio_error_iqr_low": err["iqr"][0],
            "signed_ratio_error_iqr_high": err["iqr"][1],
            "borderline_confident_false_negative_rate": float((borderline[col] >= 0.70).mean()) if len(borderline) else np.nan,
            "borderline_n": int(len(borderline)),
        }
        metrics.append(item)
        rows_out.append(item)
    for method, col in [("P1_best_fev1", "p1_observed_ratio"), ("P2_median_ratio", "p2_observed_ratio"), ("P3_ATS_largest", "p3_observed_ratio")]:
        m = confusion_binary(test["fev1_fvc_ratio"] < 0.70, test[col] < 0.70)
        err = signed_summary(test[col] - test["fev1_fvc_ratio"])
        borderline = test[test["truth_severity"] == "mild_0.60_0.70"]
        item = {
            "row_type": "part_b_b1_pooling_metric",
            "source": "observed",
            "method": method,
            **m,
            "signed_ratio_error_median": err["median"],
            "signed_ratio_error_iqr_low": err["iqr"][0],
            "signed_ratio_error_iqr_high": err["iqr"][1],
            "borderline_confident_false_negative_rate": float((borderline[col] >= 0.70).mean()) if len(borderline) else np.nan,
            "borderline_n": int(len(borderline)),
        }
        metrics.append(item)
        rows_out.append(item)

    loo_rows = []
    for seq, group in maneuvers[maneuvers["SEQN"].isin(test["SEQN"])].groupby("SEQN"):
        row = {"row_type": "part_b_leave_one_out_sd", "SEQN": float(seq)}
        for source in ["model", "observed"]:
            for method in ["P1", "P2", "P3"]:
                row[f"{source}_{method}_loo_sd"] = leave_one_out_sd(group, source, method)
        loo_rows.append(row)
    rows_out.extend(loo_rows)
    loo_df = pd.DataFrame(loo_rows)
    loo_summary = []
    for source in ["model", "observed"]:
        for method in ["P1", "P2", "P3"]:
            item = {"row_type": "part_b_loo_summary", "source": source, "method": method, **median_iqr(loo_df[f"{source}_{method}_loo_sd"])}
            loo_summary.append(item)
            rows_out.append(item)

    flag_rows = []
    for flag_name, flag_col in [("DISAGREE", "model_disagree_flag"), ("SPREAD", "model_spread_flag")]:
        for stratum, group in test.groupby("truth_severity"):
            flag_rows.append(
                {
                    "row_type": "part_b_b2_flag_rate",
                    "flag": flag_name,
                    "truth_severity": stratum,
                    "n": int(len(group)),
                    "flag_rate": float(group[flag_col].mean()) if len(group) else np.nan,
                }
            )
        flag_rows.append(
            {
                "row_type": "part_b_b2_flag_rate",
                "flag": flag_name,
                "truth_severity": "ALL",
                "n": int(len(test)),
                "flag_rate": float(test[flag_col].mean()) if len(test) else np.nan,
            }
        )
    policy_test = policy_rows[policy_rows["split"] == "test"].copy()
    if "truncated_ats" in policy_test:
        for stratum, group in policy_test.groupby("truth_severity"):
            flag_rows.append(
                {
                    "row_type": "part_b_b2_flag_rate",
                    "flag": "ATS_TRUNCATION_EXISTING_POLICY",
                    "truth_severity": stratum,
                    "n": int(len(group)),
                    "flag_rate": float(group["truncated_ats"].mean()) if len(group) else np.nan,
                }
            )
    rows_out.extend(flag_rows)

    policy_metrics = []
    for source, ratio_col in [
        ("model_P2", "p2_model_ratio"),
        ("model_P3", "p3_model_ratio"),
        ("observed_P2", "p2_observed_ratio"),
        ("observed_P3", "p3_observed_ratio"),
    ]:
        for flag_name, flag_col in [("DISAGREE", "model_disagree_flag"), ("SPREAD", "model_spread_flag")]:
            work = test.copy()
            met = conf_metrics(work, ratio_col, flag_col)
            item = {"row_type": "part_b_b3_policy_metric", "source_policy": source, "flag": flag_name, **flatten_policy_metric(met)}
            policy_metrics.append(item)
            rows_out.append(item)

    current = policy_test[policy_test["policy_outcome_fixed_preferred"].notna()].copy()
    current_metric = three_outcome_metrics(current, current["policy_outcome_fixed_preferred"]) if len(current) else {}
    current_flat = {"row_type": "part_b_b3_policy_metric", "source_policy": "current_single_c1_ATS", "flag": "ATS_TRUNCATION", **flatten_policy_metric(current_metric)}
    policy_metrics.append(current_flat)
    rows_out.append(current_flat)

    borderline_mask = test["truth_severity"] == "mild_0.60_0.70"
    borderline_n = int(borderline_mask.sum())
    observed_false_negative = (test["p2_observed_ratio"] >= 0.70) & borderline_mask
    perfect_flag_residual = 0.0
    observed_disagree_residual = float((observed_false_negative & (~test["observed_disagree"])).sum() / max(borderline_n, 1))
    irreducibility = {
        "row_type": "part_b_b4_irreducibility",
        "borderline_n": borderline_n,
        "observed_p2_borderline_fn_rate_no_flag": float(observed_false_negative.sum() / borderline_n) if borderline_n else np.nan,
        "observed_p2_borderline_confident_fn_after_observed_disagree_flag": observed_disagree_residual,
        "label_aware_perfect_flag_bound": perfect_flag_residual,
    }
    rows_out.append(irreducibility)

    disagree_borderline = next(r for r in flag_rows if r["flag"] == "DISAGREE" and r["truth_severity"] == "mild_0.60_0.70")
    disagree_normal = next(r for r in flag_rows if r["flag"] == "DISAGREE" and r["truth_severity"] == "normal")
    verdict = "BORDERLINE-FLAG-USEFUL" if disagree_borderline["flag_rate"] > disagree_normal["flag_rate"] else "NO-BORDERLINE-ENRICHMENT"
    result = {
        "verdict": verdict,
        "train_test_counts": {
            "train": train["truth_severity"].value_counts().to_dict(),
            "test": test["truth_severity"].value_counts().to_dict(),
        },
        "spread_threshold_train_85pct": threshold,
        "spread_train_flag_rate": float(train["model_spread_flag"].mean()),
        "pooling_metrics": metrics,
        "leave_one_out_sd_summary": loo_summary,
        "flag_rates": flag_rows,
        "policy_metrics": policy_metrics,
        "irreducibility": irreducibility,
    }
    write_part_b(root, result, pd.DataFrame(rows_out))
    return result


def flatten_policy_metric(m: dict[str, Any]) -> dict[str, Any]:
    if not m:
        return {}
    return {
        "n": m.get("n"),
        "obstructed_truth_obstructed_fraction": m["obstructed_truth"]["obstructed_fraction"],
        "obstructed_truth_flagged_fraction": m["obstructed_truth"]["flagged_fraction"],
        "obstructed_truth_normal_fraction": m["obstructed_truth"]["normal_fraction"],
        "borderline_residual_false_negative_rate": m["borderline_residual_false_negative_rate"],
        "borderline_flagged_fraction": m["borderline_flagged_fraction"],
        "normal_truth_flagged_fraction": m["normal_truth"]["flagged_fraction"],
        "confident_sensitivity": m["confident_nonflagged"]["sensitivity"],
        "confident_specificity": m["confident_nonflagged"]["specificity"],
        "confident_kappa": m["confident_nonflagged"]["cohens_kappa"],
    }


def multi_start_fit(curve: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    t = np.asarray(curve["vol_t"], dtype=float)
    y = np.asarray(curve["vol_l"], dtype=float)
    sigma = savgol_sigma(y)
    scale = sigma if np.isfinite(sigma) and sigma > 0 else max(float(np.std(y)), 0.01)
    starts = [np.array([0.35, 2.50, 0.60, np.clip(y[-1], 2.0, 6.0), 0.12], dtype=float)]
    # Physics-informed rough start: fast time from early rise, slow time from terminal fraction.
    fvc0 = float(np.clip(max(y[-1], np.nanmax(y)), 2.0, 6.0))
    starts.append(np.array([0.25, 3.0, 0.55, fvc0, 0.08], dtype=float))
    for _ in range(8):
        starts.append(rng.uniform(BOUNDS_LO, BOUNDS_HI))

    def residual(x: np.ndarray) -> np.ndarray:
        return (np_ombe_volume(x, t) - y) / scale

    best = None
    for x0 in starts:
        try:
            opt = least_squares(residual, x0=x0, bounds=(BOUNDS_LO, BOUNDS_HI), method="trf", jac="3-point", max_nfev=1200)
            rss = float(np.sum(residual(opt.x) ** 2))
            if best is None or rss < best["rss_scaled"]:
                pred = np_ombe_volume(opt.x, t)
                fev1, fvc, ratio = clinical_metrics_from_volume(t, pred)
                best = {
                    "success": bool(opt.success),
                    "rss_scaled": rss,
                    "params": opt.x,
                    "model_fev1_l": fev1,
                    "model_fvc_l": fvc,
                    "model_ratio": ratio,
                    "nfev": int(opt.nfev),
                    "message": str(opt.message),
                }
        except Exception as exc:
            if best is None:
                best = {"success": False, "rss_scaled": np.inf, "message": str(exc), "params": np.full(5, np.nan)}
    if best is None:
        return {"success": False, "message": "no starts attempted"}
    plausible = (
        best["success"]
        and np.isfinite(best["model_ratio"])
        and 0.1 <= best["model_ratio"] <= 1.0
        and np.isfinite(best["model_fvc_l"])
        and 0.5 * curve["parsed_fvc_l"] <= best["model_fvc_l"] <= 1.5 * curve["parsed_fvc_l"]
        and np.all(np.isfinite(best["params"]))
    )
    best["plausible"] = bool(plausible)
    return best


def run_part_c(root: Path, raw: pd.DataFrame, truth: pd.DataFrame) -> dict[str, Any]:
    results_dir = root / "results"
    clinical = pd.read_csv(results_dir / "gate_clinical_classification_F_rows.csv")
    policy = pd.read_csv(results_dir / "gate_policy_calibrated_flag_F_rows.csv")
    gli = load_gli_reference(root)
    tier2 = truth[
        truth["fev1_fvc_ratio"].notna()
        & truth["effort_quality"].isin(["A", "B"])
        & truth["fev1_quality"].isin(["A", "B"])
    ].copy()
    modsev = tier2[tier2["fev1_fvc_ratio"] < 0.60].copy()
    retained = set(clinical[clinical["sample_group"] == "obstructed"]["SEQN"].astype(float))
    dropped = modsev[~modsev["SEQN"].astype(float).isin(retained)].copy()
    rng = np.random.default_rng(RANDOM_SEED + 303)
    rows_out: list[dict[str, Any]] = []
    recovered = []
    for truth_row in dropped.itertuples(index=False):
        seqn = float(truth_row.SEQN)
        raw_part = raw[(raw["SEQN"].astype(float) == seqn) & (raw["SPATTYPE"].map(decode_grade) == "Pre")].copy()
        raw_part = raw_part[raw_part["SPAACC"].map(decode_grade) == "Y"].copy()
        attempts = []
        for row in raw_part.itertuples(index=True):
            try:
                curve = extract_curve_for_fit(row)
                fit = multi_start_fit(curve, rng)
                attempts.append((fit.get("rss_scaled", np.inf), row.Index, curve, fit, row))
            except Exception as exc:
                rows_out.append({"row_type": "part_c_c1_recovery_attempt", "SEQN": seqn, "raw_index": int(row.Index), "fit_error": str(exc), "plausible": False})
        if attempts:
            attempts = sorted(attempts, key=lambda x: x[0])
            _rss, raw_index, curve, fit, raw_row = attempts[0]
            plausible = bool(fit.get("plausible", False))
            item = {
                "row_type": "part_c_c1_recovery_attempt",
                "SEQN": seqn,
                "raw_index": int(raw_index),
                "SPAQEFF": decode_grade(raw_row.SPAQEFF),
                "truth_ratio": float(truth_row.fev1_fvc_ratio),
                "truth_severity": severity_category(float(truth_row.fev1_fvc_ratio)),
                "observed_ratio": float(curve["parsed_ratio"]),
                "observed_fvc_l": float(curve["parsed_fvc_l"]),
                "model_ratio": float(fit.get("model_ratio", np.nan)),
                "model_fvc_l": float(fit.get("model_fvc_l", np.nan)),
                "model_obstructed_fixed": bool(fit.get("model_ratio", np.nan) < 0.70) if np.isfinite(fit.get("model_ratio", np.nan)) else False,
                "plausible": plausible,
                "fit_success": bool(fit.get("success", False)),
                "fit_error": "" if fit.get("success", False) else fit.get("message", ""),
            }
            rows_out.append(item)
            if plausible:
                recovered.append(item)
        else:
            rows_out.append({"row_type": "part_c_c1_recovery_attempt", "SEQN": seqn, "fit_error": "no acceptable Pre SPAACC==Y raw rows", "plausible": False})

    old_policy_available = policy[policy["truth_severity"].isin(["severe", "moderate"])].drop_duplicates("SEQN")
    old_policy_available = old_policy_available[old_policy_available["SEQN"].astype(float).isin(set(modsev["SEQN"].astype(float)))]
    old_flag_n = int(len(modsev) - len(old_policy_available))
    recovered_n = len(recovered)
    new_flag_n = max(old_flag_n - recovered_n, 0)
    recovered_obstructed = int(sum(r["model_obstructed_fixed"] for r in recovered))
    recovered_normal = int(recovered_n - recovered_obstructed)
    old_obstructed = int((old_policy_available["corrected_ratio_c1"] < 0.70).sum()) if "corrected_ratio_c1" in old_policy_available else 0
    old_normal = int((old_policy_available["corrected_ratio_c1"] >= 0.70).sum()) if "corrected_ratio_c1" in old_policy_available else 0
    denom = int(len(modsev))
    coverage = {
        "denominator": denom,
        "old_confident_rate": float((old_obstructed + old_normal) / denom),
        "old_flag_rate": float(old_flag_n / denom),
        "new_confident_rate": float((old_obstructed + old_normal + recovered_n) / denom),
        "new_flag_rate": float(new_flag_n / denom),
        "new_residual_confident_fn_rate": float((old_normal + recovered_normal) / denom),
        "recovered_n": recovered_n,
        "recovered_obstructed_n": recovered_obstructed,
        "recovered_normal_n": recovered_normal,
    }
    coverage["sentence"] = (
        f"After multi-start recovery, the system produces a confident call for {100*coverage['new_confident_rate']:.1f}% "
        f"of severe+moderate obstructed participants and recommends repeat/FLAG for {100*coverage['new_flag_rate']:.1f}%, "
        f"with residual confident false-negative rate {100*coverage['new_residual_confident_fn_rate']:.1f}%."
    )

    policy_test = policy[policy["split"] == "test"].copy()
    policy_test = policy_test.merge(gli[["SEQN", "gli_lln_ratio"]], on="SEQN", how="left", suffixes=("", "_gli_ref"))
    if "gli_lln_ratio_gli_ref" in policy_test:
        policy_test["gli_lln_ratio"] = policy_test["gli_lln_ratio"].fillna(policy_test["gli_lln_ratio_gli_ref"])
    gli_valid = policy_test[policy_test["gli_lln_ratio"].notna() & np.isfinite(policy_test["gli_lln_ratio"])].copy()
    gli_obs = gli_valid[gli_valid["fev1_fvc_ratio"] < gli_valid["gli_lln_ratio"]].copy()
    fixed_obs = policy_test[policy_test["fev1_fvc_ratio"] < 0.70].copy()
    gli_near = (gli_obs["fev1_fvc_ratio"] >= gli_obs["gli_lln_ratio"] - 0.05) & (gli_obs["fev1_fvc_ratio"] <= gli_obs["gli_lln_ratio"])
    fixed_near = (fixed_obs["fev1_fvc_ratio"] >= 0.65) & (fixed_obs["fev1_fvc_ratio"] <= 0.70)
    gli_diag = {
        "row_type": "part_c_c2_gli_diagnosis",
        "test_n": int(len(policy_test)),
        "gli_valid_n": int(len(gli_valid)),
        "gli_lln_median": float(np.nanmedian(gli_valid["gli_lln_ratio"])) if len(gli_valid) else np.nan,
        "gli_lln_iqr_low": q(gli_valid["gli_lln_ratio"])[0] if len(gli_valid) else np.nan,
        "gli_lln_iqr_high": q(gli_valid["gli_lln_ratio"])[1] if len(gli_valid) else np.nan,
        "gli_obstructed_n": int(len(gli_obs)),
        "gli_obstructed_near_threshold_fraction": float(gli_near.mean()) if len(gli_obs) else np.nan,
        "fixed_obstructed_n": int(len(fixed_obs)),
        "fixed_obstructed_near_threshold_fraction": float(fixed_near.mean()) if len(fixed_obs) else np.nan,
    }
    gli_diag["explains_gap"] = bool(gli_diag["gli_obstructed_near_threshold_fraction"] > gli_diag["fixed_obstructed_near_threshold_fraction"] + 0.10)
    rows_out.append(gli_diag)
    verdict = "RECOVERY-HELPS" if recovered_n else "NO-RECOVERY"
    result = {
        "verdict": verdict,
        "c1_dropped_n": int(len(dropped)),
        "recovered_plausible_n": recovered_n,
        "coverage": coverage,
        "gli_diagnosis": gli_diag,
    }
    write_part_c(root, result, pd.DataFrame(rows_out))
    return result


def write_part_a(root: Path, result: dict[str, Any], rows: pd.DataFrame) -> None:
    out = root / "results"
    lines = [
        "# Gate - c1 Artifact Diagnosis, Cycle F",
        "",
        f"- VERDICT: {result['verdict']}",
        f"- {result['mechanism_sentence']}",
        f"- c1 bias from policy gate: {result['c1_bias_from_gate_policy']:.6f}",
        "",
        "## A1. Signed Percent Error vs NHANES",
        "",
        md_table(result["a1_error_summaries"], [("set", "set", "str"), ("source", "source", "str"), ("metric", "metric", "str"), ("n", "n", "int"), ("signed_pct_median", "median %", "float3"), ("signed_pct_iqr_low", "IQR low", "float3"), ("signed_pct_iqr_high", "IQR high", "float3")]),
        "",
        "## A2. Model FVC vs Observed FVC",
        "",
        md_table(result["a2_model_vs_observed"], [("set", "set", "str"), ("n", "n", "int"), ("signed_pct_median", "median %", "float3"), ("signed_pct_iqr_low", "IQR low", "float3"), ("signed_pct_iqr_high", "IQR high", "float3")]),
        "",
        result["retraction_statement"],
        "",
        "## A3. Available NHANES Fields and EOT/BTPS Variants",
        "",
        f"- candidate value fields: {result['a3_fields']['candidate_value_fields']}",
        "",
        md_table(result["a3_variant_summary"], [("variant", "variant", "str"), ("n", "n", "int"), ("fvc_signed_pct_median", "FVC median %", "float3"), ("fvc_signed_pct_iqr_low", "IQR low", "float3"), ("fvc_signed_pct_iqr_high", "IQR high", "float3"), ("ratio_signed_pct_median", "ratio median %", "float3")]),
        "",
        "## A4. Classification Without c1 vs c1",
        "",
        md_table(result["a4_classification"], [("definition", "definition", "str"), ("ratio_variant", "ratio", "str"), ("n", "n", "int"), ("sensitivity", "sens", "float3"), ("specificity", "spec", "float3"), ("cohens_kappa", "kappa", "float3"), ("obstructed_signed_ratio_error_median", "obs signed ratio err", "float4"), ("borderline_false_negative_rate", "borderline FN", "float3")]),
        "",
    ]
    (out / f"{PART_A_STEM}.md").write_text("\n".join(lines), encoding="utf-8")
    (out / f"{PART_A_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    rows.to_csv(out / f"{PART_A_STEM}_rows.csv", index=False)


def write_part_b(root: Path, result: dict[str, Any], rows: pd.DataFrame) -> None:
    out = root / "results"
    lines = [
        "# Gate - Multi-Manoeuvre Pooling + Disagreement Flag, Cycle F",
        "",
        f"- VERDICT: {result['verdict']}",
        f"- spread threshold learned on TRAIN: {result['spread_threshold_train_85pct']:.4f} (TRAIN flag rate {result['spread_train_flag_rate']:.3f})",
        f"- train/test counts: {result['train_test_counts']}",
        "",
        "## B1. Pooling Metrics on TEST",
        "",
        md_table(result["pooling_metrics"], [("source", "source", "str"), ("method", "method", "str"), ("n", "n", "int"), ("sensitivity", "sens", "float3"), ("specificity", "spec", "float3"), ("cohens_kappa", "kappa", "float3"), ("signed_ratio_error_median", "signed err", "float4"), ("borderline_confident_false_negative_rate", "borderline FN", "float3")]),
        "",
        "## B1. Leave-One-Out Pooling Stability",
        "",
        md_table(result["leave_one_out_sd_summary"], [("source", "source", "str"), ("method", "method", "str"), ("n", "n", "int"), ("median", "median LOO SD", "float4"), ("iqr", "IQR", "str")]),
        "",
        "## B2. Label-Independent Flag Rates",
        "",
        md_table(result["flag_rates"], [("flag", "flag", "str"), ("truth_severity", "stratum", "str"), ("n", "n", "int"), ("flag_rate", "flag rate", "float3")]),
        "",
        "## B3. Three-Outcome Policy Metrics",
        "",
        md_table(result["policy_metrics"], [("source_policy", "policy", "str"), ("flag", "flag", "str"), ("n", "n", "int"), ("obstructed_truth_obstructed_fraction", "obs->OBSTRUCTED", "float3"), ("obstructed_truth_flagged_fraction", "obs->FLAG", "float3"), ("obstructed_truth_normal_fraction", "obs->NORMAL", "float3"), ("borderline_residual_false_negative_rate", "borderline FN", "float3"), ("normal_truth_flagged_fraction", "normal flag", "float3"), ("confident_sensitivity", "conf sens", "float3"), ("confident_specificity", "conf spec", "float3"), ("confident_kappa", "kappa", "float3")]),
        "",
        "## B4. Irreducibility Bound",
        "",
        f"{result['irreducibility']}",
        "",
    ]
    (out / f"{PART_B_STEM}.md").write_text("\n".join(lines), encoding="utf-8")
    (out / f"{PART_B_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    rows.to_csv(out / f"{PART_B_STEM}_rows.csv", index=False)


def write_part_c(root: Path, result: dict[str, Any], rows: pd.DataFrame) -> None:
    out = root / "results"
    lines = [
        "# Gate - Fit-Failure Recovery + GLI Diagnosis, Cycle F",
        "",
        f"- VERDICT: {result['verdict']}",
        f"- dropped severe+moderate participants retried: {result['c1_dropped_n']}",
        f"- recovered plausible fits: {result['recovered_plausible_n']}",
        f"- {result['coverage']['sentence']}",
        "",
        "## C1. Multi-Start Recovery",
        "",
        "Per-participant recovery attempts are in the row CSV.",
        "",
        "## C2. GLI LLN Diagnosis",
        "",
        f"- GLI LLN median: {result['gli_diagnosis']['gli_lln_median']:.4f}",
        f"- GLI LLN IQR: {result['gli_diagnosis']['gli_lln_iqr_low']:.4f} to {result['gli_diagnosis']['gli_lln_iqr_high']:.4f}",
        f"- GLI-obstructed near-threshold fraction: {result['gli_diagnosis']['gli_obstructed_near_threshold_fraction']:.3f}",
        f"- fixed-0.70 obstructed near-threshold fraction: {result['gli_diagnosis']['fixed_obstructed_near_threshold_fraction']:.3f}",
        f"- near-threshold explanation holds: {result['gli_diagnosis']['explains_gap']}",
        "",
    ]
    (out / f"{PART_C_STEM}.md").write_text("\n".join(lines), encoding="utf-8")
    (out / f"{PART_C_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    rows.to_csv(out / f"{PART_C_STEM}_rows.csv", index=False)


def run(root: Path, cycle: str = "F") -> dict[str, Any]:
    cycle = cycle.upper()
    if cycle != "F":
        raise RuntimeError(
            "gate_four_part_fix_f.py still consumes F-named cached clinical/policy row files; "
            "non-F cycles must migrate those cached inputs before this gate can run without mixed-cycle leakage."
        )
    start = time.perf_counter()
    column_map = load_column_map(root)
    truth = load_truth(root, column_map, cycle)
    raw_path, raw_note = find_or_unzip_raw(root, cycle)
    raw, raw_info = read_raw(raw_path, cycle)
    print(f"Four-part deterministic gates raw source: {raw_path}")
    part_a = run_part_a(root, raw, truth, cycle)
    part_b = run_part_b(root, truth)
    part_c = run_part_c(root, raw, truth)
    return {
        "raw_source": str(raw_path),
        "raw_source_note": raw_note,
        "raw_loaded_rows": int(raw_info["loaded_rows"]),
        "part_a": part_a["verdict"],
        "part_b": part_b["verdict"],
        "part_c": part_c["verdict"],
        "wall_clock_s": float(time.perf_counter() - start),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--cycle", default="F", choices=("E", "F", "G"))
    args = parser.parse_args()
    result = run(Path(args.root).resolve(), args.cycle.upper())
    print(f"Part A: {result['part_a']}")
    print(f"Part B: {result['part_b']}")
    print(f"Part C: {result['part_c']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
