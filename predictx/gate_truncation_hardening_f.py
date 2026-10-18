from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache" / "xdg"))

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .gate3_diagnosis import np_ombe_volume, savgol_sigma
from .gate23_f import find_or_unzip_raw, read_raw
from .gate_clinical_classification_f import decode_grade, extract_curve_for_fit
from .gate_truncation_recovery_f import (
    ESTIMATORS as BASE_ESTIMATORS,
)
from .gate_truncation_recovery_f import (
    PREDICTION_HORIZON_S,
    TRUNCATION_TIMES,
    clean_json,
    confusion_binary,
    fit_linear_tail,
    fit_spline_extrapolate,
    insert_time_point,
    mono_exp_volume,
    pct_error,
)

OUTPUT_STEM = "gate_truncation_hardening_F"
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20260710
HEADLINE_ESTIMATORS = ("E0_DIRECT", "E1_OMBE", "H1")
ALL_ESTIMATORS = BASE_ESTIMATORS + ("H1",)
REALISTIC_FLAG_RULE = "SPAQEFF in {C,D} OR SPAACC != Y OR SPAPLAT != Y"
REALISTIC_BUCKETS = (
    ("field_le_2", 0.0, 2.0),
    ("field_2_3", 2.0, 3.0),
    ("field_3_4", 3.0, 4.0),
    ("field_4_5", 4.0, 5.0),
    ("field_gt_5", 5.0, np.inf),
)


def norm_t_label(value: Any) -> str:
    if pd.isna(value):
        return ""
    try:
        f = float(value)
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    return str(value)


def coerce_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def source_ref(path: str, lines: str) -> str:
    return f"{path}:{lines}"


def endpoint_bucket(seconds: float) -> str:
    for label, lo, hi in REALISTIC_BUCKETS:
        if seconds > lo and seconds <= hi:
            return label
    return "field_unbucketed"


def severity_band(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "unknown"
    if ratio < 0.50:
        return "<0.50"
    if ratio < 0.60:
        return "0.50-0.60"
    if ratio < 0.65:
        return "0.60-0.65"
    if ratio < 0.70:
        return "0.65-0.70"
    return ">=0.70"


def line_value_dump(root: Path) -> dict[str, Any]:
    trunc = json.loads((root / "results" / "gate_truncation_recovery_F.json").read_text(encoding="utf-8"))
    trunc_cls = [
        {
            "source": source_ref("results/gate_truncation_recovery_F.json", "classification_summary"),
            "estimator": row["estimator"],
            "T_label": norm_t_label(row["T_label"]),
            "n": row["n"],
            "tp": row["tp"],
            "tn": row["tn"],
            "fp": row["fp"],
            "fn": row["fn"],
            "fit_failure_n": row["fit_failure_n"],
            "false_negative_rate_obstructed": row["false_negative_rate_obstructed"],
            "specificity": row["specificity"],
        }
        for row in trunc["classification_summary"]
        if norm_t_label(row["T_label"]) in {"2", "3", "4", "5"}
    ]

    offset_rows = pd.read_csv(root / "results" / "gate_offset_lungsize_F_rows.csv")
    h1 = offset_rows[
        (offset_rows["row_type"].eq("part5_truncation_classification"))
        & (offset_rows["estimator"].eq("H1"))
        & (offset_rows["T_label"].isin([2.0, 3.0, 4.0, 5.0]))
    ].copy()
    h1_cls = [
        {
            "source": source_ref("results/gate_offset_lungsize_F_rows.csv", "part5_truncation_classification"),
            "estimator": "H1",
            "T_label": norm_t_label(row.T_label),
            "n": int(row.n),
            "tp": int(row.tp),
            "tn": int(row.tn),
            "fp": int(row.fp),
            "fn": int(row.fn),
            "fit_failure_n": 0,
            "false_negative_rate_obstructed": float(row.obstructed_false_negative_rate),
            "specificity": float(row.specificity),
        }
        for row in h1.itertuples(index=False)
    ]

    return {
        "clean_classification_existing": trunc_cls,
        "h1_classification_existing": h1_cls,
        "sample_sources": [
            {"item": "endpoint", "value": f"Every truncated estimator used only volume samples up to T and a fixed {PREDICTION_HORIZON_S:.0f} s prediction horizon; the full curve was used only for scoring.", "source": source_ref("results/gate_truncation_recovery_F.md", "6")},
            {"item": "total clean-cut n", "value": "497", "source": source_ref("results/gate_truncation_recovery_F.md", "26")},
            {"item": "obstructed denominator", "value": "297 = 17 severe + 80 moderate + 200 mild_0.60_0.70", "source": source_ref("results/gate_truncation_recovery_F.md", "16-18")},
            {"item": "specificity denominator", "value": "200 normal", "source": source_ref("results/gate_truncation_recovery_F.md", "19")},
            {"item": "plateau rule", "value": "FET >= 6.0 s and terminal-second dV <= 0.025 L.", "source": source_ref("results/gate_truncation_recovery_F.md", "21")},
            {"item": "competitor tuple", "value": '("E0_DIRECT", "E1_OMBE", "E2_SPLINE", "E3_LINEAR_TAIL", "E4_MONO_EXPONENTIAL")', "source": source_ref("predictx/gate_truncation_recovery_f.py", "24")},
            {"item": "competitor definitions", "value": "spline, linear tail, mono-exponential are implemented; no GP is in the estimator tuple.", "source": source_ref("predictx/gate_truncation_recovery_f.py", "736-738")},
        ],
        "other_sources": [
            {"item": "fixed 0.70 confusion", "value": "NHANES normal 287/13; NHANES obstructed 2/112", "source": source_ref("results/gate_clinical_classification_F.md", "34-37")},
            {"item": "fixed 0.70 metrics", "value": "sensitivity 0.982; specificity 0.957; Cohen's kappa 0.912", "source": source_ref("results/gate_clinical_classification_F.md", "41-44")},
            {"item": "GLI LLN metrics", "value": "sensitivity 0.966; specificity 0.939; Cohen's kappa 0.874", "source": source_ref("results/gate_clinical_classification_F.md", "58-61")},
            {"item": "severe demoted to normal", "value": "0/31 (0.000)", "source": source_ref("results/gate_clinical_classification_F.md", "73-74")},
            {"item": "held-out H1/observed CI", "value": "H1 3328 0.684 0.641 0.723 1000; observed 3328 0.700 0.659 0.738 1000", "source": source_ref("results/gate_offset_lungsize_F.md", "270-276")},
            {"item": "complete curve margin", "value": "model P3 kappa 0.680 vs observed P3 0.918", "source": source_ref("results/gate_model_vs_observed_F.md", "117")},
            {"item": "repeatability verdict", "value": "VERDICT: UNSTABLE; borderline model unanimous fraction 0.698; model-vs-observed ATS repeatability margin 3.2 percentage points.", "source": source_ref("results/gate_reproducibility_volume_F.md", "57")},
            {"item": "FVC SD", "value": "observed 0.0931; model 0.0745", "source": source_ref("results/gate_reproducibility_volume_F.md", "27")},
            {"item": "ratio ICC", "value": "observed 0.793; model 0.778; n=6677", "source": source_ref("results/gate_reproducibility_volume_F.md", "30-31")},
            {"item": "ATS repeatability", "value": "observed 0.843; model 0.875; delta 3.2 percentage points", "source": source_ref("results/gate_reproducibility_volume_F.md", "47-50")},
            {"item": "Gate 2 parse", "value": "PASS; F only; E and G are BLOCKED; n 7079; FEV1 2.91%; FVC 3.53%", "source": source_ref("results/gate2_F_parse.md", "3-5,37-39")},
            {"item": "Gate 3 adequacy", "value": "FAIL; F only; E and G are BLOCKED; 0/31851 (0.0% of all curves; 0.0% of NRSS-valid curves)", "source": source_ref("results/gate3_F_ombe_rerun.md", "3-5,26")},
        ],
    }


def current_clean_detail(root: Path) -> pd.DataFrame:
    trunc = pd.read_csv(root / "results" / "gate_truncation_recovery_F_rows.csv")
    trunc["T_label"] = trunc["T_label"].map(norm_t_label)
    clean = trunc[trunc["T_label"].isin({"2", "3", "4", "5"})].copy()
    clean["dataset"] = "clean_cut"
    clean["endpoint_bucket"] = "clean_T" + clean["T_label"].astype(str)
    clean["field_endpoint_s"] = clean["T_s"].astype(float)
    clean["used_stop_s"] = clean["T_s"].astype(float)
    clean["source_curve_type"] = "same_complete_curve_fixed_cut"

    h1_rows = pd.read_csv(root / "results" / "gate_estimand_hybrid_F_rows.csv")
    h1 = h1_rows[h1_rows["row_type"].eq("part3_truncation_h1_detail")].copy()
    h1["T_label"] = h1["T_label"].map(norm_t_label)
    h1 = h1[h1["T_label"].isin({"2", "3", "4", "5"})].copy()
    h1["estimator"] = "H1"
    h1["dataset"] = "clean_cut"
    h1["endpoint_bucket"] = "clean_T" + h1["T_label"].astype(str)
    h1["field_endpoint_s"] = h1["T_label"].astype(float)
    h1["used_stop_s"] = h1["T_label"].astype(float)
    h1["source_curve_type"] = "same_complete_curve_fixed_cut"
    h1["fit_success"] = h1["fit_success"].map(coerce_bool)
    h1["truth_obstructed_fixed"] = h1["truth_obstructed_fixed"].map(coerce_bool)
    h1["est_obstructed_fixed"] = h1["est_obstructed_fixed"].map(coerce_bool)

    for col in clean.columns:
        if col not in h1.columns:
            h1[col] = np.nan
    for col in h1.columns:
        if col not in clean.columns:
            clean[col] = np.nan
    detail = pd.concat([clean[h1.columns], h1], ignore_index=True)
    detail["fit_success"] = detail["fit_success"].map(coerce_bool)
    detail["truth_obstructed_fixed"] = detail["truth_obstructed_fixed"].map(coerce_bool)
    detail["est_obstructed_fixed"] = detail["est_obstructed_fixed"].map(coerce_bool)
    return detail


def fit_ombe_to_endpoint(t_fit: np.ndarray, y_fit: np.ndarray, target_end_s: float) -> dict[str, Any]:
    t_fit = np.asarray(t_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    if len(t_fit) < 5 or not np.isfinite(t_fit[-1]) or t_fit[-1] <= 0:
        return {"success": False, "failure": "too few usable time points", "params": np.full(5, np.nan), "fvc_l": np.nan, "fev1_l": np.nan}
    target_end_s = float(max(target_end_s, 1.0))
    grid = np.unique(np.concatenate([t_fit, [1.0, target_end_s]]))
    grid.sort()
    fit_idx = np.where(grid <= t_fit[-1] + 1e-10)[0]
    y_obs = np.interp(grid[fit_idx], t_fit, y_fit)
    obs_end = float(y_obs[-1])
    x0 = np.array([0.35, 2.50, 0.60, np.clip(obs_end, 2.0, 6.0), 0.12], dtype=float)
    lo = np.array([0.15, 1.00, 0.25, 2.00, 0.03], dtype=float)
    hi = np.array([0.80, 5.00, 0.85, 6.00, 0.40], dtype=float)
    sigma = savgol_sigma(y_obs)
    scale = sigma if np.isfinite(sigma) and sigma > 0 else max(float(np.std(y_obs)), 0.01)

    def residual(x: np.ndarray) -> np.ndarray:
        return (np_ombe_volume(x, grid)[fit_idx] - y_obs) / scale

    try:
        opt = least_squares(residual, x0=x0, bounds=(lo, hi), method="trf", jac="3-point", max_nfev=600)
        pred = np_ombe_volume(opt.x, grid)
        fvc_l = float(np.interp(target_end_s, grid, pred))
        fev1_l = float(np.interp(1.0, grid, pred))
        success = bool(opt.success and np.isfinite(fvc_l) and fvc_l > 0 and np.isfinite(fev1_l))
        return {"success": success, "failure": "" if success else str(opt.message), "params": opt.x, "fvc_l": fvc_l, "fev1_l": fev1_l, "nfev": int(opt.nfev)}
    except Exception as exc:
        return {"success": False, "failure": str(exc), "params": np.full(5, np.nan), "fvc_l": np.nan, "fev1_l": np.nan}


def fit_mono_to_endpoint(t_fit: np.ndarray, y_fit: np.ndarray, target_end_s: float) -> dict[str, Any]:
    t_fit = np.asarray(t_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    if len(t_fit) < 5 or not np.isfinite(t_fit[-1]) or t_fit[-1] <= 0:
        return {"success": False, "failure": "too few usable time points", "fvc_l": np.nan}
    target_end_s = float(max(target_end_s, 1.0))
    grid = np.unique(np.concatenate([t_fit, [target_end_s]]))
    grid.sort()
    fit_idx = np.where(grid <= t_fit[-1] + 1e-10)[0]
    y_obs = np.interp(grid[fit_idx], t_fit, y_fit)
    obs_end = float(y_obs[-1])
    x0 = np.array([1.50, np.clip(obs_end, 2.0, 6.0), 0.12], dtype=float)
    lo = np.array([0.15, 2.00, 0.03], dtype=float)
    hi = np.array([5.00, 6.00, 0.40], dtype=float)
    sigma = savgol_sigma(y_obs)
    scale = sigma if np.isfinite(sigma) and sigma > 0 else max(float(np.std(y_obs)), 0.01)

    def residual(x: np.ndarray) -> np.ndarray:
        return (mono_exp_volume(x, grid)[fit_idx] - y_obs) / scale

    try:
        opt = least_squares(residual, x0=x0, bounds=(lo, hi), method="trf", jac="3-point", max_nfev=600)
        pred = mono_exp_volume(opt.x, grid)
        fvc_l = float(np.interp(target_end_s, grid, pred))
        success = bool(opt.success and np.isfinite(fvc_l) and fvc_l > 0)
        return {"success": success, "failure": "" if success else str(opt.message), "fvc_l": fvc_l, "nfev": int(opt.nfev)}
    except Exception as exc:
        return {"success": False, "failure": str(exc), "fvc_l": np.nan}


def realistic_candidates(root: Path, raw: pd.DataFrame, clean_detail: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    target = clean_detail[
        (clean_detail["dataset"].eq("clean_cut"))
        & (clean_detail["T_label"].eq("2"))
        & (clean_detail["estimator"].eq("E0_DIRECT"))
    ].copy()
    target = target.drop_duplicates("SEQN").copy()
    target_cols = [
        "SEQN",
        "raw_index",
        "SPAMANU",
        "truth_severity",
        "true_fev1_l",
        "true_fvc_l",
        "true_ratio",
        "nhanes_fev1_mL",
        "nhanes_fvc_mL",
        "nhanes_ratio",
        "full_end_s",
    ]
    target = target[target_cols].rename(columns={"raw_index": "target_raw_index", "SPAMANU": "target_SPAMANU"})

    raw2 = raw.copy()
    raw2["raw_index"] = raw2.index.astype(int)
    raw2["SPATTYPE_dec"] = raw2["SPATTYPE"].map(decode_grade)
    raw2["SPAACC_dec"] = raw2["SPAACC"].map(decode_grade)
    raw2["SPAQEFF_dec"] = raw2["SPAQEFF"].map(decode_grade)
    raw2["SPAPLAT_dec"] = raw2["SPAPLAT"].map(decode_grade)
    raw2 = raw2[raw2["SPATTYPE_dec"].eq("Pre")].copy()
    raw2["field_realistic_flag"] = (
        raw2["SPAQEFF_dec"].isin(["C", "D"])
        | raw2["SPAACC_dec"].ne("Y")
        | raw2["SPAPLAT_dec"].ne("Y")
    )
    flagged = raw2[raw2["field_realistic_flag"]].copy()
    joined = flagged.merge(target, on="SEQN", how="inner")
    joined = joined[joined["raw_index"].astype(int).ne(joined["target_raw_index"].astype(int))].copy()

    raw_by_index = {int(idx): row for idx, row in raw.iterrows()}
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(joined.itertuples(index=False), start=1):
        base = row._asdict()
        try:
            curve = extract_curve_for_fit(raw_by_index[int(row.raw_index)])
            t = np.asarray(curve["vol_t"], dtype=float)
            y = np.asarray(curve["vol_l"], dtype=float)
            if len(t) < 5 or float(t[-1]) <= 0 or not np.isfinite(float(y[-1])) or float(y[-1]) <= 0:
                raise ValueError("unusable extracted curve")
            duration = float(t[-1])
            v_start = float(np.interp(max(0.0, duration - 1.0), t, y))
            base.update(
                {
                    "field_parse_error": "",
                    "field_duration_s": duration,
                    "field_endpoint_bucket": endpoint_bucket(duration),
                    "field_end_volume_l": float(y[-1]),
                    "field_fev1_l": float(np.interp(1.0, t, y)),
                    "field_dV_last_L": float(y[-1] - v_start),
                    "field_n_points": int(len(t)),
                }
            )
        except Exception as exc:
            base.update(
                {
                    "field_parse_error": str(exc),
                    "field_duration_s": np.nan,
                    "field_endpoint_bucket": "parse_failed",
                    "field_end_volume_l": np.nan,
                    "field_fev1_l": np.nan,
                    "field_dV_last_L": np.nan,
                    "field_n_points": 0,
                }
            )
        rows.append(base)
        if i % 100 == 0 or i == len(joined):
            print(f"Realistic candidate parse progress: {i}/{len(joined)}")

    parsed = pd.DataFrame(rows)
    usable = parsed[parsed["field_parse_error"].fillna("").eq("")].copy()
    selected = (
        usable.sort_values(["SEQN", "field_duration_s", "SPAMANU", "raw_index"], ascending=[True, True, True, True])
        .groupby("SEQN", as_index=False)
        .first()
    )
    meta = {
        "flag_rule": REALISTIC_FLAG_RULE,
        "target_clean_sample_n": int(target["SEQN"].nunique()),
        "flagged_raw_pre_rows_n": int(len(flagged)),
        "matched_flagged_candidate_rows_n": int(len(joined)),
        "matched_patients_with_any_flagged_curve_n": int(joined["SEQN"].nunique()),
        "usable_flagged_candidate_rows_n": int(len(usable)),
        "selected_one_per_patient_n": int(len(selected)),
        "selected_endpoint_bucket_counts": {str(k): int(v) for k, v in selected["field_endpoint_bucket"].value_counts().sort_index().items()} if len(selected) else {},
        "decision": "Selected the shortest usable field-realistic flagged Pre curve per clean-target patient to avoid repeated correlated rows and to stress early termination.",
    }
    return selected, meta


def estimate_realistic_curve(row: Any, raw_by_index: dict[int, Any]) -> list[dict[str, Any]]:
    raw_row = raw_by_index[int(row.raw_index)]
    curve = extract_curve_for_fit(raw_row)
    field_t = np.asarray(curve["vol_t"], dtype=float)
    field_y = np.asarray(curve["vol_l"], dtype=float)
    target_end = PREDICTION_HORIZON_S
    field_endpoint = float(field_t[-1])
    used_stop = min(field_endpoint, target_end)
    t_fit, y_fit = insert_time_point(field_t, field_y, used_stop)
    observed_fev1 = float(np.interp(1.0, t_fit, y_fit))
    direct_fvc = float(y_fit[-1])
    ombe = fit_ombe_to_endpoint(t_fit, y_fit, target_end)
    mono = fit_mono_to_endpoint(t_fit, y_fit, target_end)
    spline = fit_spline_extrapolate(t_fit, y_fit, target_end)
    linear = fit_linear_tail(t_fit, y_fit, target_end)
    fits = {
        "E0_DIRECT": {"success": np.isfinite(direct_fvc) and direct_fvc > 0, "fvc_l": direct_fvc, "failure": ""},
        "E1_OMBE": ombe,
        "E2_SPLINE": spline,
        "E3_LINEAR_TAIL": linear,
        "E4_MONO_EXPONENTIAL": mono,
    }
    rows = []
    for estimator, fit in fits.items():
        est_fvc = float(fit.get("fvc_l", np.nan)) if np.isfinite(fit.get("fvc_l", np.nan)) else np.nan
        success = bool(fit.get("success", False) and np.isfinite(est_fvc) and est_fvc > 0)
        est_ratio = observed_fev1 / est_fvc if success else np.nan
        rows.append(
            {
                "dataset": "field_realistic",
                "source_curve_type": "same_patient_flagged_curve",
                "SEQN": float(row.SEQN),
                "raw_index": int(row.raw_index),
                "target_raw_index": int(row.target_raw_index),
                "SPAMANU": float(row.SPAMANU),
                "target_SPAMANU": float(row.target_SPAMANU),
                "SPAQEFF": str(row.SPAQEFF_dec),
                "SPAACC": str(row.SPAACC_dec),
                "SPAPLAT": str(row.SPAPLAT_dec),
                "truth_severity": row.truth_severity,
                "T_label": "field_all",
                "endpoint_bucket": row.field_endpoint_bucket,
                "field_endpoint_s": field_endpoint,
                "used_stop_s": used_stop,
                "prediction_horizon_s": target_end,
                "estimator": estimator,
                "fit_success": success,
                "fit_failure": "" if success else str(fit.get("failure", "nonfinite/nonpositive estimate")),
                "true_fev1_l": float(row.true_fev1_l),
                "true_fvc_l": float(row.true_fvc_l),
                "true_ratio": float(row.true_ratio),
                "nhanes_fev1_mL": float(row.nhanes_fev1_mL),
                "nhanes_fvc_mL": float(row.nhanes_fvc_mL),
                "nhanes_ratio": float(row.nhanes_ratio),
                "observed_truncated_fev1_l": observed_fev1,
                "observed_truncated_end_l": direct_fvc,
                "est_fvc_l": est_fvc,
                "est_ratio": est_ratio,
                "model_fev1_l": np.nan,
                "truth_obstructed_fixed": bool(float(row.true_ratio) < 0.70),
                "est_obstructed_fixed": bool(est_ratio < 0.70) if np.isfinite(est_ratio) else False,
                "fvc_signed_pct_error": pct_error(est_fvc, float(row.true_fvc_l)),
                "fvc_abs_pct_error": abs(pct_error(est_fvc, float(row.true_fvc_l))) if np.isfinite(pct_error(est_fvc, float(row.true_fvc_l))) else np.nan,
                "ratio_signed_error": float(est_ratio - float(row.true_ratio)) if np.isfinite(est_ratio) else np.nan,
                "ratio_abs_error": float(abs(est_ratio - float(row.true_ratio))) if np.isfinite(est_ratio) else np.nan,
            }
        )

    h1_ratio = np.nan
    if ombe.get("success", False) and np.isfinite(ombe.get("fev1_l", np.nan)) and np.isfinite(ombe.get("fvc_l", np.nan)) and float(ombe["fvc_l"]) > 0:
        h1_ratio = float(ombe["fev1_l"]) / float(ombe["fvc_l"])
    h1_success = bool(np.isfinite(h1_ratio))
    rows.append(
        {
            **{k: rows[0][k] for k in rows[0].keys()},
            "estimator": "H1",
            "fit_success": h1_success,
            "fit_failure": "" if h1_success else str(ombe.get("failure", "nonfinite H1 ratio")),
            "est_fvc_l": float(ombe.get("fvc_l", np.nan)) if np.isfinite(ombe.get("fvc_l", np.nan)) else np.nan,
            "model_fev1_l": float(ombe.get("fev1_l", np.nan)) if np.isfinite(ombe.get("fev1_l", np.nan)) else np.nan,
            "est_ratio": h1_ratio,
            "est_obstructed_fixed": bool(h1_ratio < 0.70) if np.isfinite(h1_ratio) else False,
            "fvc_signed_pct_error": pct_error(float(ombe.get("fvc_l", np.nan)), float(row.true_fvc_l)) if np.isfinite(ombe.get("fvc_l", np.nan)) else np.nan,
            "fvc_abs_pct_error": abs(pct_error(float(ombe.get("fvc_l", np.nan)), float(row.true_fvc_l))) if np.isfinite(ombe.get("fvc_l", np.nan)) else np.nan,
            "ratio_signed_error": float(h1_ratio - float(row.true_ratio)) if np.isfinite(h1_ratio) else np.nan,
            "ratio_abs_error": float(abs(h1_ratio - float(row.true_ratio))) if np.isfinite(h1_ratio) else np.nan,
        }
    )
    return rows


def run_realistic_estimators(raw: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    raw_by_index = {int(idx): row for idx, row in raw.iterrows()}
    rows: list[dict[str, Any]] = []
    total = len(selected)
    for i, row in enumerate(selected.itertuples(index=False), start=1):
        rows.extend(estimate_realistic_curve(row, raw_by_index))
        if i % 25 == 0 or i == total:
            print(f"Field-realistic estimator progress: {i}/{total} participants")
    detail = pd.DataFrame(rows)
    if not detail.empty:
        bucket_rows = detail.copy()
        bucket_rows["T_label"] = bucket_rows["endpoint_bucket"]
        detail = pd.concat([detail, bucket_rows], ignore_index=True)
    return detail


def conservative_metrics(part: pd.DataFrame) -> dict[str, Any]:
    all_rows = part.drop_duplicates(["SEQN", "estimator"]).copy()
    truth = all_rows["truth_obstructed_fixed"].map(coerce_bool)
    valid = all_rows["fit_success"].map(coerce_bool) & np.isfinite(all_rows["est_ratio"].astype(float))
    pred = all_rows["est_obstructed_fixed"].map(coerce_bool) & valid
    metrics = confusion_binary(truth, pred)
    metrics.update(
        {
            "all_n": int(len(all_rows)),
            "valid_n": int(valid.sum()),
            "fit_failure_or_nonfinite_n": int((~valid).sum()),
            "obstructed_denominator_n": int(truth.sum()),
            "normal_denominator_n": int((~truth).sum()),
            "false_negative_rate_obstructed": metrics["fn"] / (metrics["tp"] + metrics["fn"]) if (metrics["tp"] + metrics["fn"]) else np.nan,
        }
    )
    return metrics


def classification_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, t_label, estimator), part in detail.groupby(["dataset", "T_label", "estimator"], dropna=False):
        metrics = conservative_metrics(part)
        rows.append({"dataset": dataset, "T_label": str(t_label), "estimator": estimator, **metrics})
    return pd.DataFrame(rows).sort_values(["dataset", "T_label", "estimator"]).reset_index(drop=True)


def bootstrap_group(detail: pd.DataFrame, dataset: str, t_label: str, rng: np.random.Generator) -> list[dict[str, Any]]:
    group = detail[(detail["dataset"].eq(dataset)) & (detail["T_label"].astype(str).eq(str(t_label))) & (detail["estimator"].isin(HEADLINE_ESTIMATORS))].copy()
    if group.empty:
        return []
    wide_rows = []
    for seq, part in group.groupby("SEQN"):
        row = {
            "SEQN": float(seq),
            "truth_obstructed_fixed": bool(part["truth_obstructed_fixed"].iloc[0]),
        }
        for est in HEADLINE_ESTIMATORS:
            est_part = part[part["estimator"].eq(est)]
            if len(est_part) == 0:
                row[f"{est}_pred"] = False
                row[f"{est}_valid"] = False
            else:
                item = est_part.iloc[0]
                valid = coerce_bool(item["fit_success"]) and np.isfinite(float(item["est_ratio"]))
                row[f"{est}_pred"] = bool(item["est_obstructed_fixed"]) if valid else False
                row[f"{est}_valid"] = valid
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows)
    seqs = wide["SEQN"].to_numpy(float)
    n = len(wide)
    out: list[dict[str, Any]] = []

    def fn_rate(df: pd.DataFrame, est: str) -> float:
        obs = df[df["truth_obstructed_fixed"]].copy()
        return float((~obs[f"{est}_pred"].astype(bool)).mean()) if len(obs) else np.nan

    point = {est: fn_rate(wide, est) for est in HEADLINE_ESTIMATORS}
    draws = {est: [] for est in HEADLINE_ESTIMATORS}
    diffs = {"DIRECT_minus_OMBE": [], "DIRECT_minus_H1": []}
    for _ in range(BOOTSTRAP_B):
        sampled = rng.choice(seqs, size=n, replace=True)
        boot = pd.DataFrame({"SEQN": sampled}).merge(wide, on="SEQN", how="left")
        rates = {est: fn_rate(boot, est) for est in HEADLINE_ESTIMATORS}
        for est in HEADLINE_ESTIMATORS:
            draws[est].append(rates[est])
        diffs["DIRECT_minus_OMBE"].append(rates["E0_DIRECT"] - rates["E1_OMBE"])
        diffs["DIRECT_minus_H1"].append(rates["E0_DIRECT"] - rates["H1"])

    obstructed_n = int(wide["truth_obstructed_fixed"].sum())
    for est in HEADLINE_ESTIMATORS:
        arr = np.asarray(draws[est], dtype=float)
        out.append(
            {
                "dataset": dataset,
                "T_label": t_label,
                "quantity": "fn_rate",
                "estimator": est,
                "comparison": "",
                "n_patients": n,
                "n_obstructed": obstructed_n,
                "point": point[est],
                "ci_low": float(np.nanpercentile(arr, 2.5)),
                "ci_high": float(np.nanpercentile(arr, 97.5)),
                "bootstrap_n": BOOTSTRAP_B,
                "ci_excludes_zero": "",
            }
        )
    for comp, arr_values in diffs.items():
        arr = np.asarray(arr_values, dtype=float)
        out.append(
            {
                "dataset": dataset,
                "T_label": t_label,
                "quantity": "fn_rate_difference",
                "estimator": "",
                "comparison": comp,
                "n_patients": n,
                "n_obstructed": obstructed_n,
                "point": float(np.nanmean(arr)),
                "ci_low": float(np.nanpercentile(arr, 2.5)),
                "ci_high": float(np.nanpercentile(arr, 97.5)),
                "bootstrap_n": BOOTSTRAP_B,
                "ci_excludes_zero": bool(float(np.nanpercentile(arr, 2.5)) > 0 or float(np.nanpercentile(arr, 97.5)) < 0),
            }
        )
    return out


def bootstrap_headlines(detail: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    labels = (
        [("clean_cut", str(int(t))) for t in TRUNCATION_TIMES]
        + [("field_realistic", "field_all")]
        + [("field_realistic", label) for label, _, _ in REALISTIC_BUCKETS]
    )
    for dataset, label in labels:
        rows.extend(bootstrap_group(detail, dataset, label, rng))
    return pd.DataFrame(rows)


def recovered_curve_audit(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (dataset, t_label, seq), group in detail[detail["estimator"].isin(HEADLINE_ESTIMATORS)].groupby(["dataset", "T_label", "SEQN"], dropna=False):
        direct = group[group["estimator"].eq("E0_DIRECT")]
        e1 = group[group["estimator"].eq("E1_OMBE")]
        h1 = group[group["estimator"].eq("H1")]
        if len(direct) == 0 or len(e1) == 0 or len(h1) == 0:
            continue
        d = direct.iloc[0]
        e = e1.iloc[0]
        h = h1.iloc[0]
        truth_obstructed = coerce_bool(d["truth_obstructed_fixed"])
        direct_valid = coerce_bool(d["fit_success"]) and np.isfinite(float(d["est_ratio"]))
        direct_call = bool(d["est_obstructed_fixed"]) if direct_valid else False
        e1_valid = coerce_bool(e["fit_success"]) and np.isfinite(float(e["est_ratio"]))
        h1_valid = coerce_bool(h["fit_success"]) and np.isfinite(float(h["est_ratio"]))
        e1_call = bool(e["est_obstructed_fixed"]) if e1_valid else False
        h1_call = bool(h["est_obstructed_fixed"]) if h1_valid else False
        direct_missed = truth_obstructed and not direct_call
        rows.append(
            {
                "dataset": dataset,
                "T_label": t_label,
                "SEQN": float(seq),
                "truth_severity": d["truth_severity"],
                "true_ratio": float(d["true_ratio"]),
                "ratio_band": severity_band(float(d["true_ratio"])),
                "truth_obstructed_fixed": truth_obstructed,
                "direct_missed": direct_missed,
                "recovered_by_E1": bool(direct_missed and e1_call),
                "recovered_by_H1": bool(direct_missed and h1_call),
                "not_recovered_by_E1": bool(direct_missed and not e1_call),
                "not_recovered_by_H1": bool(direct_missed and not h1_call),
                "direct_ratio": float(d["est_ratio"]) if np.isfinite(float(d["est_ratio"])) else np.nan,
                "E1_ratio": float(e["est_ratio"]) if np.isfinite(float(e["est_ratio"])) else np.nan,
                "H1_ratio": float(h["est_ratio"]) if np.isfinite(float(h["est_ratio"])) else np.nan,
                "field_endpoint_s": float(d.get("field_endpoint_s", np.nan)) if pd.notna(d.get("field_endpoint_s", np.nan)) else np.nan,
                "endpoint_bucket": d.get("endpoint_bucket", ""),
                "raw_index": int(d["raw_index"]) if pd.notna(d.get("raw_index", np.nan)) else np.nan,
                "target_raw_index": int(d["target_raw_index"]) if pd.notna(d.get("target_raw_index", np.nan)) else np.nan,
            }
        )
    audit = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for (dataset, t_label), part in audit[audit["direct_missed"]].groupby(["dataset", "T_label"], dropna=False):
        for method, recovered_col, failed_col in [
            ("E1", "recovered_by_E1", "not_recovered_by_E1"),
            ("H1", "recovered_by_H1", "not_recovered_by_H1"),
        ]:
            for group_name, subset in [("recovered", part[part[recovered_col]]), ("not_recovered", part[part[failed_col]])]:
                severe_mod = subset["truth_severity"].isin(["severe", "moderate"]).sum() if len(subset) else 0
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "T_label": t_label,
                        "method": method,
                        "group": group_name,
                        "n": int(len(subset)),
                        "true_ratio_median": float(np.nanmedian(subset["true_ratio"])) if len(subset) else np.nan,
                        "severe_moderate_n": int(severe_mod),
                        "severe_moderate_fraction": float(severe_mod / len(subset)) if len(subset) else np.nan,
                        "band_<0.50_n": int((subset["ratio_band"] == "<0.50").sum()) if len(subset) else 0,
                        "band_0.50_0.60_n": int((subset["ratio_band"] == "0.50-0.60").sum()) if len(subset) else 0,
                        "band_0.60_0.65_n": int((subset["ratio_band"] == "0.60-0.65").sum()) if len(subset) else 0,
                        "band_0.65_0.70_n": int((subset["ratio_band"] == "0.65-0.70").sum()) if len(subset) else 0,
                    }
                )
    return audit, pd.DataFrame(summary_rows)


def md_table(rows: pd.DataFrame | list[dict[str, Any]], columns: list[tuple[str, str, str]]) -> str:
    df = pd.DataFrame(rows)
    out = ["| " + " | ".join(label for _, label, _ in columns) + " |"]
    out.append("|" + "|".join("---" for _ in columns) + "|")
    for _, row in df.iterrows():
        cells = []
        for key, _, fmt in columns:
            value = row.get(key, np.nan)
            if fmt == "int":
                cells.append(str(int(value)) if pd.notna(value) else "NA")
            elif fmt == "bool":
                cells.append(str(bool(value)) if pd.notna(value) else "")
            elif fmt.startswith("float"):
                digits = int(fmt.replace("float", "") or "3")
                try:
                    v = float(value)
                    cells.append(f"{v:.{digits}f}" if np.isfinite(v) else "NA")
                except Exception:
                    cells.append(str(value))
            else:
                cells.append(str(value))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def part0_markdown(dump: dict[str, Any]) -> list[str]:
    clean_rows = pd.DataFrame(dump["clean_classification_existing"])
    h1_rows = pd.DataFrame(dump["h1_classification_existing"])
    lines = [
        "## Part 0 - Ground-Truth Dump",
        "",
        "This section was read before writing any new hardening output.",
        "",
        "### Clean Truncation Sample And Endpoint",
        "",
        md_table(dump["sample_sources"], [("item", "item", "str"), ("value", "value", "str"), ("source", "source", "str")]),
        "",
        "### Existing Clean-Cut Classification Rates",
        "",
        md_table(
            pd.concat([clean_rows, h1_rows], ignore_index=True).sort_values(["estimator", "T_label"]),
            [
                ("estimator", "estimator", "str"),
                ("T_label", "T", "str"),
                ("n", "n", "int"),
                ("tp", "tp", "int"),
                ("tn", "tn", "int"),
                ("fp", "fp", "int"),
                ("fn", "fn", "int"),
                ("fit_failure_n", "fit failures", "int"),
                ("false_negative_rate_obstructed", "obstructed FN", "float6"),
                ("specificity", "specificity", "float6"),
                ("source", "source", "str"),
            ],
        ),
        "",
        "### Other Current Claims",
        "",
        md_table(dump["other_sources"], [("item", "item", "str"), ("value", "value", "str"), ("source", "source", "str")]),
        "",
    ]
    return lines


def limitation_ledger() -> list[dict[str, str]]:
    return [
        {
            "limitation": "Cycle F only; E and G are blocked.",
            "source": "results/gate2_F_parse.md:3-5; results/gate3_F_ombe_rerun.md:3-5",
        },
        {
            "limitation": "Gate 3 flow-domain adequacy is 0/31851; this is a genuine flow-shape adequacy failure.",
            "source": "results/gate3_F_ombe_rerun.md:26,62",
        },
        {
            "limitation": "The flow-domain failure does not propagate to the selected volume-domain screening decision in the clinical gate.",
            "source": "results/gate_clinical_classification_F.md:3-5",
        },
        {
            "limitation": "Repeatability remains UNSTABLE on ratio ICC: observed 0.793 vs model 0.778.",
            "source": "results/gate_reproducibility_volume_F.md:3,30-31,57",
        },
        {
            "limitation": "Complete curves: observed beats model; H1 is for truncated use, not a complete-curve replacement.",
            "source": "results/gate_model_vs_observed_F.md:117; results/gate_offset_lungsize_F.md:13",
        },
        {
            "limitation": "GLI degradation is unexplained; lung-size interaction was refuted and age tracks GLI LLN with r=-0.950.",
            "source": "results/gate_offset_lungsize_F.md:12,294-297",
        },
        {
            "limitation": "Posterior inference is not usable for decisioning on this path: posterior calibration is NOT-CALIBRATED and S9 initialization is insufficient.",
            "source": "results/gate_posterior_calibration_repair_F.md:3-5; results/gate_s9_modeinit_F.md:11-18",
        },
    ]


def choose_verdict(cls: pd.DataFrame, boot: pd.DataFrame, recovery_summary: pd.DataFrame) -> dict[str, Any]:
    def fn(dataset: str, t_label: str, estimator: str) -> float:
        part = cls[(cls["dataset"].eq(dataset)) & (cls["T_label"].astype(str).eq(str(t_label))) & (cls["estimator"].eq(estimator))]
        return float(part.iloc[0]["false_negative_rate_obstructed"]) if len(part) else np.nan

    field_direct = fn("field_realistic", "field_all", "E0_DIRECT")
    field_e1 = fn("field_realistic", "field_all", "E1_OMBE")
    field_h1 = fn("field_realistic", "field_all", "H1")
    survives = np.isfinite(field_direct) and np.isfinite(field_e1) and np.isfinite(field_h1) and field_e1 < field_direct and field_h1 < field_direct

    clean_ci = boot[
        (boot["dataset"].eq("clean_cut"))
        & (boot["T_label"].isin(["2", "3"]))
        & (boot["quantity"].eq("fn_rate_difference"))
        & (boot["comparison"].isin(["DIRECT_minus_OMBE", "DIRECT_minus_H1"]))
    ]
    clean_ci_ok = bool(len(clean_ci) == 4 and clean_ci["ci_low"].astype(float).gt(0).all())
    field_ci = boot[
        (boot["dataset"].eq("field_realistic"))
        & (boot["T_label"].eq("field_all"))
        & (boot["quantity"].eq("fn_rate_difference"))
        & (boot["comparison"].isin(["DIRECT_minus_OMBE", "DIRECT_minus_H1"]))
    ]
    field_ci_ok = bool(len(field_ci) == 2 and field_ci["ci_low"].astype(float).gt(0).all())
    rec = recovery_summary[
        (recovery_summary["dataset"].eq("field_realistic"))
        & (recovery_summary["T_label"].eq("field_all"))
        & (recovery_summary["group"].eq("recovered"))
        & (recovery_summary["method"].isin(["E1", "H1"]))
    ]
    recovery_concentrates = bool(len(rec) and rec["severe_moderate_fraction"].fillna(0).max() >= 0.50)

    if survives and clean_ci_ok and field_ci_ok and recovery_concentrates:
        verdict = "TRUNCATION-ROBUST"
        reason = "recovery survives field-realistic truncation, clean and field CIs exclude zero, and recovered cases include real obstruction."
    elif survives:
        verdict = "TRUNCATION-QUALIFIED"
        reason = "recovery survives field-realistic truncation, but at least one robustness condition is weaker or denominator-limited."
    else:
        verdict = "TRUNCATION-FRAGILE"
        reason = "recovery does not survive field-realistic truncation; the headline must be restated."
    return {
        "verdict": verdict,
        "reason": reason,
        "field_direct_fn": field_direct,
        "field_e1_fn": field_e1,
        "field_h1_fn": field_h1,
        "clean_T2_T3_difference_cis_exclude_zero": clean_ci_ok,
        "field_all_difference_cis_exclude_zero": field_ci_ok,
        "field_recovery_concentrates_in_severe_moderate": recovery_concentrates,
    }


def make_report(result: dict[str, Any], cls: pd.DataFrame, boot: pd.DataFrame, recovery_summary: pd.DataFrame) -> str:
    lines = ["# Gate - Truncation Hardening, Cycle F", ""]
    lines.extend(part0_markdown(result["part0_dump"]))
    lines.extend(
        [
            "## Protocol Decisions",
            "",
            f"- Realistic quality flag rule: `{REALISTIC_FLAG_RULE}`.",
            "- Used actual same-patient lower-quality/incomplete Pre curves where a clean complete-plateau target exists.",
            "- Selected the shortest usable flagged curve per patient to avoid repeated correlated rows and stress early termination.",
            "- The field-realistic endpoint is the extracted forced-expiration duration of that flagged curve. If it exceeds the clean target end, the analysis clips at the clean target end so no estimator sees beyond the target endpoint.",
            "- Hardening metrics are conservative: failed or nonfinite fits are treated as non-obstructed calls for false-negative and specificity calculations, and fit-failure counts are reported.",
            "- No NUTS, TSNPE, sampler, gamma-onset, registration, or ML residual code is used.",
            "",
            "## Part 1 - Realistic Truncation",
            "",
            md_table(pd.DataFrame([result["realistic_meta"]]), [
                ("target_clean_sample_n", "clean target patients", "int"),
                ("flagged_raw_pre_rows_n", "flagged Pre rows", "int"),
                ("matched_flagged_candidate_rows_n", "matched candidate rows", "int"),
                ("matched_patients_with_any_flagged_curve_n", "matched patients", "int"),
                ("usable_flagged_candidate_rows_n", "usable candidate rows", "int"),
                ("selected_one_per_patient_n", "selected patients", "int"),
            ]),
            "",
            f"Selected endpoint bucket counts: `{result['realistic_meta']['selected_endpoint_bucket_counts']}`.",
            "",
            "### Hardening Classification Summary",
            "",
            md_table(
                cls[cls["T_label"].isin(["2", "3", "4", "5", "field_all", "field_le_2", "field_2_3", "field_3_4", "field_4_5", "field_gt_5"])],
                [
                    ("dataset", "dataset", "str"),
                    ("T_label", "endpoint", "str"),
                    ("estimator", "estimator", "str"),
                    ("all_n", "all n", "int"),
                    ("valid_n", "valid n", "int"),
                    ("fit_failure_or_nonfinite_n", "fit/nonfinite n", "int"),
                    ("obstructed_denominator_n", "obstructed n", "int"),
                    ("normal_denominator_n", "normal n", "int"),
                    ("false_negative_rate_obstructed", "obstructed FN", "float3"),
                    ("specificity", "specificity", "float3"),
                    ("cohens_kappa", "kappa", "float3"),
                ],
            ),
            "",
            "## Part 2 - Bootstrap Confidence Intervals",
            "",
            md_table(
                boot,
                [
                    ("dataset", "dataset", "str"),
                    ("T_label", "endpoint", "str"),
                    ("quantity", "quantity", "str"),
                    ("estimator", "estimator", "str"),
                    ("comparison", "comparison", "str"),
                    ("n_patients", "patients", "int"),
                    ("n_obstructed", "obstructed", "int"),
                    ("point", "point", "float3"),
                    ("ci_low", "CI low", "float3"),
                    ("ci_high", "CI high", "float3"),
                    ("ci_excludes_zero", "excludes zero", "str"),
                ],
            ),
            "",
            "## Part 3 - Recovered-Curve Failure-Mode Audit",
            "",
            md_table(
                recovery_summary,
                [
                    ("dataset", "dataset", "str"),
                    ("T_label", "endpoint", "str"),
                    ("method", "method", "str"),
                    ("group", "group", "str"),
                    ("n", "n", "int"),
                    ("true_ratio_median", "true ratio median", "float3"),
                    ("severe_moderate_n", "severe+moderate n", "int"),
                    ("severe_moderate_fraction", "severe+moderate frac", "float3"),
                    ("band_<0.50_n", "<0.50 n", "int"),
                    ("band_0.50_0.60_n", "0.50-0.60 n", "int"),
                    ("band_0.60_0.65_n", "0.60-0.65 n", "int"),
                    ("band_0.65_0.70_n", "0.65-0.70 n", "int"),
                ],
            ),
            "",
            "## Part 4 - Limitation Ledger",
            "",
            md_table(limitation_ledger(), [("limitation", "limitation", "str"), ("source", "source", "str")]),
            "",
            "## Verdict",
            "",
            f"VERDICT: {result['verdict']['verdict']} - {result['verdict']['reason']}",
            "",
            f"- field_realistic DIRECT FN: {result['verdict']['field_direct_fn']:.6f}",
            f"- field_realistic E1 FN: {result['verdict']['field_e1_fn']:.6f}",
            f"- field_realistic H1 FN: {result['verdict']['field_h1_fn']:.6f}",
            f"- clean T<=3 difference CIs exclude zero: {result['verdict']['clean_T2_T3_difference_cis_exclude_zero']}",
            f"- field_all difference CIs exclude zero: {result['verdict']['field_all_difference_cis_exclude_zero']}",
            f"- field recovery concentrates in severe/moderate: {result['verdict']['field_recovery_concentrates_in_severe_moderate']}",
            "",
        ]
    )
    return "\n".join(lines)


def run(root: Path) -> dict[str, Any]:
    out_dir = root / "results"
    part0 = line_value_dump(root)
    clean = current_clean_detail(root)
    raw_path, raw_note = find_or_unzip_raw(root)
    raw, raw_info = read_raw(raw_path)
    selected, realistic_meta = realistic_candidates(root, raw, clean)
    realistic = run_realistic_estimators(raw, selected)
    detail = pd.concat([clean, realistic], ignore_index=True, sort=False)
    cls = classification_summary(detail)
    boot = bootstrap_headlines(detail)
    recovered, recovery_summary = recovered_curve_audit(detail)
    verdict = choose_verdict(cls, boot, recovery_summary)

    result = {
        "part0_dump": part0,
        "raw_source": str(raw_path),
        "raw_source_note": raw_note,
        "raw_loaded_rows": int(raw_info["loaded_rows"]),
        "realistic_meta": realistic_meta,
        "classification_summary": cls.to_dict("records"),
        "bootstrap_summary": boot.to_dict("records"),
        "recovery_summary": recovery_summary.to_dict("records"),
        "limitation_ledger": limitation_ledger(),
        "verdict": verdict,
    }
    out_dir.mkdir(exist_ok=True)
    boot.to_csv(out_dir / f"{OUTPUT_STEM}_bootstrap.csv", index=False)
    recovered.to_csv(out_dir / f"{OUTPUT_STEM}_recovered_curves.csv", index=False)
    (out_dir / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    (out_dir / f"{OUTPUT_STEM}.md").write_text(make_report(result, cls, boot, recovery_summary), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    print(f"Truncation hardening gate: {result['verdict']['verdict']} - {result['verdict']['reason']}")
    print(f"Field realistic FN: direct={result['verdict']['field_direct_fn']:.3f}, E1={result['verdict']['field_e1_fn']:.3f}, H1={result['verdict']['field_h1_fn']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
