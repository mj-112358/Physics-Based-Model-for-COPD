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
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares

from .gate3_diagnosis import np_ombe_volume, savgol_sigma
from .gate23_f import find_or_unzip_raw, read_raw
from .gate_clinical_classification_f import (
    decode_grade,
    extract_curve_for_fit,
    load_column_map,
    load_truth,
    severity_category,
)

RANDOM_SEED = 20260708
TRUNCATION_TIMES = (2.0, 3.0, 4.0, 5.0)
ESTIMATORS = ("E0_DIRECT", "E1_OMBE", "E2_SPLINE", "E3_LINEAR_TAIL", "E4_MONO_EXPONENTIAL")
STRATA = ("severe", "moderate", "mild_0.60_0.70", "normal")
SAMPLE_MAX_PER_STRATUM = 200
PLATEAU_L = 0.025
MIN_FET_S = 6.0
MATERIAL_FN_MARGIN = 0.05
PREDICTION_HORIZON_S = 12.0


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if np.isfinite(out) else None
    return value


def q(values: pd.Series | np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return [float("nan"), float("nan")]
    return [float(x) for x in np.percentile(arr, [25, 75])]


def pct_error(est: float, truth: float) -> float:
    return float(100.0 * (est - truth) / truth) if np.isfinite(est) and np.isfinite(truth) and truth > 0 else np.nan


def confusion_binary(truth: pd.Series, pred: pd.Series) -> dict[str, Any]:
    truth = truth.astype(bool)
    pred = pred.astype(bool)
    tp = int((truth & pred).sum())
    tn = int((~truth & ~pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    n = tp + tn + fp + fn
    agreement = (tp + tn) / n if n else np.nan
    truth_pos = (tp + fn) / n if n else np.nan
    truth_neg = (tn + fp) / n if n else np.nan
    pred_pos = (tp + fp) / n if n else np.nan
    pred_neg = (tn + fn) / n if n else np.nan
    pe = truth_pos * pred_pos + truth_neg * pred_neg if n else np.nan
    kappa = (agreement - pe) / (1.0 - pe) if n and abs(1.0 - pe) > 1e-12 else np.nan
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "agreement": agreement,
        "cohens_kappa": kappa,
        "false_negative_rate_obstructed": fn / (tp + fn) if (tp + fn) else np.nan,
    }


def insert_time_point(t: np.ndarray, y: np.ndarray, stop_s: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if stop_s >= t[-1]:
        return t.copy(), y.copy()
    mask = t < stop_s
    out_t = t[mask]
    out_y = y[mask]
    y_stop = float(np.interp(stop_s, t, y))
    if len(out_t) == 0 or abs(out_t[-1] - stop_s) > 1e-9:
        out_t = np.concatenate([out_t, [float(stop_s)]])
        out_y = np.concatenate([out_y, [y_stop]])
    return out_t, out_y


def model_grid_with_stop(
    observed_t: np.ndarray,
    stop_s: float,
    prediction_horizon_s: float = PREDICTION_HORIZON_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a fixed-horizon model grid without using the true curve end time.

    A deployable truncation method knows the observed samples through ``stop_s``
    but does not know when the unobserved maneuver would have ended. The model
    grid therefore combines the observed sampling interval with a horizon fixed
    before evaluation. This prevents the complete curve duration from entering
    either model normalization or extrapolation.
    """
    t = np.asarray(observed_t, dtype=float)
    if t.size < 2:
        raise ValueError("At least two observed time points are required")
    if stop_s <= 0.0:
        raise ValueError("stop_s must be positive")
    horizon = max(float(prediction_horizon_s), float(stop_s))
    dt = round(float(np.nanmedian(np.diff(t))), 12)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Observed time grid must be strictly increasing")
    n_steps = int(np.floor(horizon / dt + 1e-10))
    grid = np.arange(n_steps + 1, dtype=float) * dt
    grid = np.unique(np.round(np.r_[grid, float(stop_s), horizon], 12))
    fit_idx = np.where(grid <= stop_s + 1e-10)[0]
    return grid, fit_idx


def mono_exp_volume(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    tau, fvc_l, tau_o = params
    t = np.asarray(t, dtype=float)
    base = np.exp(-t / tau) / tau
    rise = 1.0 - np.exp(-t / tau_o)
    q = np.maximum(base * rise, 1e-12)
    area = np.trapezoid(q, t)
    flow = fvc_l * q / area
    increments = (flow[:-1] + flow[1:]) * 0.5 * np.diff(t)
    return np.concatenate([[0.0], np.cumsum(increments)])


def fit_ombe_truncated(
    full_t: np.ndarray,
    full_y: np.ndarray,
    stop_s: float,
    prediction_horizon_s: float = PREDICTION_HORIZON_S,
) -> dict[str, Any]:
    grid, fit_idx = model_grid_with_stop(full_t, stop_s, prediction_horizon_s)
    y_fit = np.interp(grid[fit_idx], full_t, full_y)
    obs_end = float(y_fit[-1])
    x0 = np.array([0.35, 2.50, 0.60, np.clip(obs_end, 2.0, 6.0), 0.12], dtype=float)
    lo = np.array([0.15, 1.00, 0.25, 2.00, 0.03], dtype=float)
    hi = np.array([0.80, 5.00, 0.85, 6.00, 0.40], dtype=float)
    sigma = savgol_sigma(y_fit)
    scale = sigma if np.isfinite(sigma) and sigma > 0 else max(float(np.std(y_fit)), 0.01)

    def residual(x: np.ndarray) -> np.ndarray:
        pred = np_ombe_volume(x, grid)[fit_idx]
        return (pred - y_fit) / scale

    opt = least_squares(residual, x0=x0, bounds=(lo, hi), method="trf", jac="3-point", max_nfev=600)
    pred_full = np_ombe_volume(opt.x, grid)
    return {
        "success": bool(opt.success),
        "fvc_l": float(pred_full[-1]),
        "nfev": int(opt.nfev),
        "params": opt.x,
        "failure": "" if opt.success else str(opt.message),
    }


def fit_mono_truncated(
    full_t: np.ndarray,
    full_y: np.ndarray,
    stop_s: float,
    prediction_horizon_s: float = PREDICTION_HORIZON_S,
) -> dict[str, Any]:
    grid, fit_idx = model_grid_with_stop(full_t, stop_s, prediction_horizon_s)
    y_fit = np.interp(grid[fit_idx], full_t, full_y)
    obs_end = float(y_fit[-1])
    x0 = np.array([1.50, np.clip(obs_end, 2.0, 6.0), 0.12], dtype=float)
    lo = np.array([0.15, 2.00, 0.03], dtype=float)
    hi = np.array([5.00, 6.00, 0.40], dtype=float)
    sigma = savgol_sigma(y_fit)
    scale = sigma if np.isfinite(sigma) and sigma > 0 else max(float(np.std(y_fit)), 0.01)

    def residual(x: np.ndarray) -> np.ndarray:
        pred = mono_exp_volume(x, grid)[fit_idx]
        return (pred - y_fit) / scale

    opt = least_squares(residual, x0=x0, bounds=(lo, hi), method="trf", jac="3-point", max_nfev=600)
    pred_full = mono_exp_volume(opt.x, grid)
    return {
        "success": bool(opt.success),
        "fvc_l": float(pred_full[-1]),
        "nfev": int(opt.nfev),
        "params": opt.x,
        "failure": "" if opt.success else str(opt.message),
    }


def fit_spline_extrapolate(t_fit: np.ndarray, y_fit: np.ndarray, full_end_s: float) -> dict[str, Any]:
    try:
        if len(t_fit) < 5:
            raise ValueError("too few points for cubic smoothing spline")
        sigma = savgol_sigma(y_fit)
        scale = sigma if np.isfinite(sigma) and sigma > 0 else max(float(np.std(np.diff(y_fit))), 0.005)
        spline = UnivariateSpline(t_fit, y_fit, k=3, s=len(t_fit) * scale**2)
        return {"success": True, "fvc_l": float(spline(full_end_s)), "failure": ""}
    except Exception as exc:
        return {"success": False, "fvc_l": np.nan, "failure": str(exc)}


def fit_linear_tail(t_fit: np.ndarray, y_fit: np.ndarray, full_end_s: float) -> dict[str, Any]:
    try:
        tail_mask = t_fit >= max(0.0, float(t_fit[-1]) - 1.0)
        if int(tail_mask.sum()) < 2:
            raise ValueError("too few tail points for linear extrapolation")
        slope, intercept = np.polyfit(t_fit[tail_mask], y_fit[tail_mask], 1)
        return {"success": True, "fvc_l": float(slope * full_end_s + intercept), "failure": ""}
    except Exception as exc:
        return {"success": False, "fvc_l": np.nan, "failure": str(exc)}


def participant_candidates(raw: pd.DataFrame, truth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = raw[
        (raw["SPATTYPE"].map(decode_grade) == "Pre")
        & (raw["SPAACC"].map(decode_grade) == "Y")
        & (raw["SPAQEFF"].map(decode_grade) == "A")
    ].copy()
    rows = []
    for idx, row in enumerate(filtered.itertuples(index=True), start=1):
        base = {
            "raw_index": int(row.Index),
            "SEQN": float(row.SEQN),
            "SPAMANU": float(row.SPAMANU),
            "SPAQEFF": decode_grade(row.SPAQEFF),
            "SPAACC": decode_grade(row.SPAACC),
            "parse_error": "",
        }
        try:
            curve = extract_curve_for_fit(row)
            t = np.asarray(curve["vol_t"], dtype=float)
            y = np.asarray(curve["vol_l"], dtype=float)
            fet = float(t[-1])
            v_start = float(np.interp(max(0.0, fet - 1.0), t, y))
            d_last = float(y[-1] - v_start)
            base.update(
                {
                    "parsed_fev1_l": float(curve["parsed_fev1_l"]),
                    "true_fvc_l": float(curve["parsed_fvc_l"]),
                    "true_ratio": float(curve["parsed_ratio"]),
                    "fet_s": fet,
                    "dV_last_L": d_last,
                    "complete_plateau": bool(fet >= MIN_FET_S and d_last <= PLATEAU_L),
                }
            )
        except Exception as exc:
            base["parse_error"] = str(exc)
        rows.append(base)
        if idx % 5000 == 0 or idx == len(filtered):
            print(f"Truncation gate candidate parse progress: {idx}/{len(filtered)} A-grade candidates")
    candidates = pd.DataFrame(rows)
    good = candidates[candidates["parse_error"].fillna("") == ""].copy()
    good["truth_severity"] = good["true_ratio"].map(severity_category)
    selected_a = (
        good.sort_values(["SEQN", "parsed_fev1_l", "SPAMANU"], ascending=[True, False, True])
        .groupby("SEQN", as_index=False)
        .first()
    )
    complete = good[good["complete_plateau"]].copy()
    selected_complete = (
        complete.sort_values(["SEQN", "parsed_fev1_l", "SPAMANU"], ascending=[True, False, True])
        .groupby("SEQN", as_index=False)
        .first()
    )
    selected_complete = selected_complete.merge(
        truth[["SEQN", "fev1_mL", "fvc_mL", "fev1_fvc_ratio"]],
        on="SEQN",
        how="left",
    )
    selected_complete = selected_complete.dropna(subset=["fev1_mL", "fvc_mL", "fev1_fvc_ratio"]).copy()
    return selected_a, selected_complete


def sample_by_stratum(selected_a: pd.DataFrame, selected_complete: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    sampled_parts = []
    sample_table = []
    all_counts = selected_a["truth_severity"].value_counts().to_dict()
    complete_counts = selected_complete["truth_severity"].value_counts().to_dict()
    rng = RANDOM_SEED
    for stratum in STRATA:
        part = selected_complete[selected_complete["truth_severity"] == stratum].copy()
        n_sample = min(SAMPLE_MAX_PER_STRATUM, len(part))
        sampled = part.sample(n=n_sample, random_state=rng) if n_sample else part
        sampled_parts.append(sampled)
        sample_table.append(
            {
                "stratum": stratum,
                "selected_grade_A_n": int(all_counts.get(stratum, 0)),
                "complete_plateau_n": int(complete_counts.get(stratum, 0)),
                "sampled_n": int(n_sample),
                "dropped_by_plateau_or_missing_nhanes": int(all_counts.get(stratum, 0) - complete_counts.get(stratum, 0)),
                "dropped_by_sampling_cap": int(max(0, complete_counts.get(stratum, 0) - n_sample)),
            }
        )
        rng += 101
    sampled_all = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame()
    return sampled_all, {"sample_table": sample_table}


def crosscheck(sample: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    fev1_err = 100.0 * (sample["parsed_fev1_l"] * 1000.0 - sample["fev1_mL"]) / sample["fev1_mL"]
    fvc_err = 100.0 * (sample["true_fvc_l"] * 1000.0 - sample["fvc_mL"]) / sample["fvc_mL"]
    ratio_err = sample["true_ratio"] - sample["fev1_fvc_ratio"]
    out["fev1_abs_pct"] = {"median": float(np.nanmedian(np.abs(fev1_err))), "iqr": q(np.abs(fev1_err)), "signed_median": float(np.nanmedian(fev1_err))}
    out["fvc_abs_pct"] = {"median": float(np.nanmedian(np.abs(fvc_err))), "iqr": q(np.abs(fvc_err)), "signed_median": float(np.nanmedian(fvc_err))}
    out["ratio_abs"] = {"median": float(np.nanmedian(np.abs(ratio_err))), "iqr": q(np.abs(ratio_err)), "signed_median": float(np.nanmedian(ratio_err))}
    out["pass_rule"] = "PASS if median absolute FEV1 and FVC percent errors <=5% and median absolute ratio error <=0.02."
    out["passed"] = bool(out["fev1_abs_pct"]["median"] <= 5.0 and out["fvc_abs_pct"]["median"] <= 5.0 and out["ratio_abs"]["median"] <= 0.02)
    out["n"] = int(len(sample))
    return out


def estimate_one_curve(curve_row: Any, raw_row: Any) -> list[dict[str, Any]]:
    curve = extract_curve_for_fit(raw_row)
    full_t = np.asarray(curve["vol_t"], dtype=float)
    full_y = np.asarray(curve["vol_l"], dtype=float)
    true_fev1_l = float(np.interp(1.0, full_t, full_y))
    true_fvc_l = float(full_y[-1])
    true_ratio = true_fev1_l / true_fvc_l if true_fvc_l > 0 else np.nan
    full_end = float(full_t[-1])
    rows = []
    stop_specs: list[tuple[str, float]] = [(f"{t:g}", t) for t in TRUNCATION_TIMES] + [("full", full_end)]

    for t_label, stop_s in stop_specs:
        t_fit, y_fit = insert_time_point(full_t, full_y, stop_s)
        observed_fev1_l = float(np.interp(1.0, t_fit, y_fit))
        prediction_horizon = full_end if t_label == "full" else PREDICTION_HORIZON_S
        fits = {
            "E0_DIRECT": {"success": True, "fvc_l": float(y_fit[-1]), "failure": ""},
            "E1_OMBE": fit_ombe_truncated(full_t, full_y, stop_s, prediction_horizon),
            "E2_SPLINE": fit_spline_extrapolate(t_fit, y_fit, prediction_horizon),
            "E3_LINEAR_TAIL": fit_linear_tail(t_fit, y_fit, prediction_horizon),
            "E4_MONO_EXPONENTIAL": fit_mono_truncated(full_t, full_y, stop_s, prediction_horizon),
        }
        for estimator, fit in fits.items():
            est_fvc_l = float(fit["fvc_l"]) if np.isfinite(fit.get("fvc_l", np.nan)) else np.nan
            success = bool(fit.get("success", False) and np.isfinite(est_fvc_l) and est_fvc_l > 0)
            est_ratio = observed_fev1_l / est_fvc_l if success and est_fvc_l > 0 else np.nan
            rows.append(
                {
                    "SEQN": float(curve_row.SEQN),
                    "raw_index": int(curve_row.raw_index),
                    "SPAMANU": float(curve_row.SPAMANU),
                    "truth_severity": curve_row.truth_severity,
                    "T_label": t_label,
                    "T_s": float(stop_s),
                    "estimator": estimator,
                    "fit_success": success,
                    "fit_failure": "" if success else fit.get("failure", "nonfinite or nonpositive FVC estimate"),
                    "true_fev1_l": true_fev1_l,
                    "true_fvc_l": true_fvc_l,
                    "true_ratio": true_ratio,
                    "nhanes_fev1_mL": float(curve_row.fev1_mL),
                    "nhanes_fvc_mL": float(curve_row.fvc_mL),
                    "nhanes_ratio": float(curve_row.fev1_fvc_ratio),
                    "observed_truncated_fev1_l": observed_fev1_l,
                    "observed_truncated_end_l": float(y_fit[-1]),
                    "est_fvc_l": est_fvc_l,
                    "est_ratio": est_ratio,
                    "fvc_signed_pct_error": pct_error(est_fvc_l, true_fvc_l),
                    "fvc_abs_pct_error": abs(pct_error(est_fvc_l, true_fvc_l)) if np.isfinite(pct_error(est_fvc_l, true_fvc_l)) else np.nan,
                    "ratio_signed_error": float(est_ratio - true_ratio) if np.isfinite(est_ratio) and np.isfinite(true_ratio) else np.nan,
                    "ratio_abs_error": float(abs(est_ratio - true_ratio)) if np.isfinite(est_ratio) and np.isfinite(true_ratio) else np.nan,
                    "truth_obstructed_fixed": bool(true_ratio < 0.70) if np.isfinite(true_ratio) else False,
                    "est_obstructed_fixed": bool(est_ratio < 0.70) if np.isfinite(est_ratio) else False,
                    "full_end_s": full_end,
                    "prediction_horizon_s": float(prediction_horizon),
                    "fit_n": int(len(t_fit)),
                    "no_untruncated_information_after_T_used": bool(
                        t_label == "full"
                        or (
                            t_fit[-1] <= stop_s + 1e-9
                            and abs(prediction_horizon - PREDICTION_HORIZON_S) <= 1e-9
                        )
                    ),
                }
            )
    return rows


def run_estimators(sample: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    raw_by_index = {int(idx): row for idx, row in raw.iterrows()}
    rows = []
    total = len(sample)
    for i, row in enumerate(sample.itertuples(index=False), start=1):
        rows.extend(estimate_one_curve(row, raw_by_index[int(row.raw_index)]))
        if i % 25 == 0 or i == total:
            print(f"Truncation recovery estimator progress: {i}/{total} participants")
    return pd.DataFrame(rows)


def error_summary(rows: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for (est, t_label, stratum), part in rows.groupby(["estimator", "T_label", "truth_severity"], dropna=False):
        out.append(
            {
                "estimator": est,
                "T_label": t_label,
                "truth_severity": stratum,
                "n": int(len(part)),
                "fit_failure_n": int((~part["fit_success"]).sum()),
                "fvc_signed_pct_median": float(np.nanmedian(part["fvc_signed_pct_error"])),
                "fvc_signed_pct_iqr": q(part["fvc_signed_pct_error"]),
                "fvc_abs_pct_median": float(np.nanmedian(part["fvc_abs_pct_error"])),
                "fvc_abs_pct_iqr": q(part["fvc_abs_pct_error"]),
                "ratio_signed_median": float(np.nanmedian(part["ratio_signed_error"])),
                "ratio_signed_iqr": q(part["ratio_signed_error"]),
                "ratio_abs_median": float(np.nanmedian(part["ratio_abs_error"])),
                "ratio_abs_iqr": q(part["ratio_abs_error"]),
            }
        )
    return sorted(out, key=lambda r: (r["estimator"], str(r["T_label"]), str(r["truth_severity"])))


def classification_summary(rows: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for (est, t_label), part in rows.groupby(["estimator", "T_label"], dropna=False):
        valid = part[part["fit_success"] & part["est_ratio"].notna()].copy()
        metrics = confusion_binary(valid["truth_obstructed_fixed"], valid["est_obstructed_fixed"]) if len(valid) else {
            "n": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "sensitivity": np.nan,
            "specificity": np.nan,
            "agreement": np.nan,
            "cohens_kappa": np.nan,
            "false_negative_rate_obstructed": np.nan,
        }
        borderline = valid[valid["truth_severity"] == "mild_0.60_0.70"].copy()
        borderline_fn = float(((borderline["truth_obstructed_fixed"]) & (~borderline["est_obstructed_fixed"])).mean()) if len(borderline) else np.nan
        metrics.update(
            {
                "estimator": est,
                "T_label": t_label,
                "fit_failure_n": int((~part["fit_success"]).sum()),
                "borderline_n": int(len(borderline)),
                "borderline_false_negative_rate": borderline_fn,
            }
        )
        out.append(metrics)
    return sorted(out, key=lambda r: (r["estimator"], str(r["T_label"])))


def regime_map(error_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(error_rows)
    out = []
    order = ["2", "3", "4", "5", "full"]
    for est in ESTIMATORS:
        for stratum in STRATA:
            part = df[(df["estimator"] == est) & (df["truth_severity"] == stratum)].copy()
            min_t = "none"
            for label in order:
                m = part[part["T_label"] == label]["fvc_abs_pct_median"]
                if len(m) and np.isfinite(float(m.iloc[0])) and float(m.iloc[0]) < 5.0:
                    min_t = label
                    break
            out.append({"estimator": est, "truth_severity": stratum, "minimum_T_for_median_abs_FVC_error_lt_5pct": min_t})
    return out


def honesty_checks(rows: pd.DataFrame, truth: pd.DataFrame, sample: pd.DataFrame) -> dict[str, Any]:
    original_obstructed = truth[truth["fev1_fvc_ratio"].notna() & (truth["fev1_fvc_ratio"] < 0.70)].copy()
    sample_obstructed_seqn = set(sample[sample["true_ratio"] < 0.70]["SEQN"])
    excluded = original_obstructed[~original_obstructed["SEQN"].isin(sample_obstructed_seqn)].copy()
    e1 = rows[rows["estimator"] == "E1_OMBE"].copy()
    overshoot = e1[e1["fit_success"] & (e1["est_fvc_l"] > e1["true_fvc_l"])].copy()
    undershoot = e1[e1["fit_success"] & (e1["est_fvc_l"] < e1["true_fvc_l"])].copy()
    overshoot["overshoot_pct"] = 100.0 * (overshoot["est_fvc_l"] - overshoot["true_fvc_l"]) / overshoot["true_fvc_l"]
    undershoot["undershoot_pct"] = 100.0 * (undershoot["est_fvc_l"] - undershoot["true_fvc_l"]) / undershoot["true_fvc_l"]
    by_t = []
    for t_label, part in e1.groupby("T_label"):
        valid = part[part["fit_success"]].copy()
        ov = valid[valid["est_fvc_l"] > valid["true_fvc_l"]].copy()
        un = valid[valid["est_fvc_l"] < valid["true_fvc_l"]].copy()
        ov_pct = 100.0 * (ov["est_fvc_l"] - ov["true_fvc_l"]) / ov["true_fvc_l"] if len(ov) else pd.Series(dtype=float)
        un_pct = 100.0 * (un["est_fvc_l"] - un["true_fvc_l"]) / un["true_fvc_l"] if len(un) else pd.Series(dtype=float)
        by_t.append(
            {
                "T_label": t_label,
                "valid_n": int(len(valid)),
                "overshoot_n": int(len(ov)),
                "overshoot_fraction": float(len(ov) / len(valid)) if len(valid) else np.nan,
                "overshoot_pct_median": float(np.nanmedian(ov_pct)) if len(ov_pct) else np.nan,
                "overshoot_pct_max": float(np.nanmax(ov_pct)) if len(ov_pct) else np.nan,
                "undershoot_n": int(len(un)),
                "undershoot_pct_median": float(np.nanmedian(un_pct)) if len(un_pct) else np.nan,
            }
        )
    return {
        "original_cycle_f_obstructed_n": int(len(original_obstructed)),
        "sample_obstructed_n": int(len(sample_obstructed_seqn)),
        "excluded_original_obstructed_n": int(len(excluded)),
        "excluded_original_obstructed_ratio_median": float(np.nanmedian(excluded["fev1_fvc_ratio"])) if len(excluded) else np.nan,
        "excluded_original_obstructed_ratio_iqr": q(excluded["fev1_fvc_ratio"]) if len(excluded) else [np.nan, np.nan],
        "e1_overshoot_by_T": by_t,
        "no_estimator_saw_untruncated_information_after_T": bool(
            rows[rows["T_label"] != "full"]["no_untruncated_information_after_T_used"].all()
        ),
        "prediction_horizon_s": PREDICTION_HORIZON_S,
        "note": "For truncated rows, every estimator used only volume samples up to T and a fixed 12 s prediction horizon. The true complete-curve duration was not used by the fit.",
    }


def comparison_verdict(error_rows: list[dict[str, Any]], cls_rows: list[dict[str, Any]]) -> dict[str, Any]:
    e = pd.DataFrame(error_rows)
    c = pd.DataFrame(cls_rows)
    pooled = rows_to_pooled_errors(e)
    cls = c.set_index(["estimator", "T_label"]).to_dict("index")

    def fvc_abs(est: str, t_label: str) -> float:
        return float(pooled.get((est, t_label), np.nan))

    def fn(est: str, t_label: str) -> float:
        return float(cls.get((est, t_label), {}).get("false_negative_rate_obstructed", np.nan))

    e1_beats_e0_fn = all(np.isfinite(fn("E1_OMBE", t)) and np.isfinite(fn("E0_DIRECT", t)) and (fn("E0_DIRECT", t) - fn("E1_OMBE", t) >= MATERIAL_FN_MARGIN) for t in ["2", "3"])
    e1_beats_spline = all(np.isfinite(fvc_abs("E1_OMBE", t)) and np.isfinite(fvc_abs("E2_SPLINE", t)) and fvc_abs("E1_OMBE", t) < fvc_abs("E2_SPLINE", t) for t in ["2", "3"])
    e1_beats_linear = all(np.isfinite(fvc_abs("E1_OMBE", t)) and np.isfinite(fvc_abs("E3_LINEAR_TAIL", t)) and fvc_abs("E1_OMBE", t) < fvc_abs("E3_LINEAR_TAIL", t) for t in ["2", "3"])
    e1_beats_mono = all(np.isfinite(fvc_abs("E1_OMBE", t)) and np.isfinite(fvc_abs("E4_MONO_EXPONENTIAL", t)) and fvc_abs("E1_OMBE", t) < fvc_abs("E4_MONO_EXPONENTIAL", t) for t in ["2", "3"])
    e1_beats_e0_any = all(np.isfinite(fn("E1_OMBE", t)) and np.isfinite(fn("E0_DIRECT", t)) and fn("E1_OMBE", t) < fn("E0_DIRECT", t) for t in ["2", "3"])
    if e1_beats_e0_fn and e1_beats_spline and e1_beats_linear:
        verdict = "PHYSICS-JUSTIFIED"
        reason = "E1 beats direct obstruction false-negative rate at T<=3 by the material margin and beats spline/linear FVC recovery at T<=3."
    elif e1_beats_e0_any and not e1_beats_spline:
        verdict = "SMOOTHNESS-ONLY"
        reason = "E1 improves over direct measurement, but spline matches or beats it on FVC recovery at T<=3."
    elif not e1_beats_e0_any:
        verdict = "NOT-JUSTIFIED"
        reason = "E1 does not beat direct measurement on obstructed false-negative rate at T<=3."
    else:
        verdict = "NOT-JUSTIFIED"
        reason = "E1 does not satisfy the full physics-justified criterion under the predeclared comparisons."
    return {
        "verdict": verdict,
        "reason": reason,
        "material_false_negative_margin": MATERIAL_FN_MARGIN,
        "pooled_fvc_abs_pct_medians": {f"{k[0]}|{k[1]}": v for k, v in pooled.items()},
        "e1_beats_e0_fn_material_T2_T3": e1_beats_e0_fn,
        "e1_beats_spline_fvc_T2_T3": e1_beats_spline,
        "e1_beats_linear_fvc_T2_T3": e1_beats_linear,
        "e1_beats_mono_fvc_T2_T3": e1_beats_mono,
        "physics_vs_smoothness_sentence": (
            "E1 beats the spline and linear-tail baselines at T=2 and T=3, so the bi-exponential form is doing work beyond generic smooth extrapolation."
            if e1_beats_spline and e1_beats_linear
            else "E1 does not beat both spline and linear-tail baselines at T=2 and T=3, so the gate does not support a specific physics advantage over smooth extrapolation."
        ),
        "biexponential_vs_mono_sentence": (
            "E1 beats the mono-exponential at T=2 and T=3, so the two-compartment structure is load-bearing."
            if e1_beats_mono
            else "E1 does not beat the mono-exponential at T=2 and T=3, so this gate does not show that the second compartment is load-bearing."
        ),
    }


def rows_to_pooled_errors(error_rows: pd.DataFrame) -> dict[tuple[str, str], float]:
    out = {}
    for (est, t_label), part in error_rows.groupby(["estimator", "T_label"]):
        values = part["fvc_abs_pct_median"].to_numpy(float)
        weights = part["n"].to_numpy(float)
        finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        out[(est, t_label)] = float(np.average(values[finite], weights=weights[finite])) if finite.any() else np.nan
    return out


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str, str]]) -> str:
    out = ["| " + " | ".join(label for _, label, _ in columns) + " |"]
    out.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        cells = []
        for key, _, fmt in columns:
            value = row.get(key)
            if fmt == "int":
                cells.append(str(int(value)) if pd.notna(value) else "NA")
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


def matrix_table(cls_rows: list[dict[str, Any]], metric: str) -> str:
    df = pd.DataFrame(cls_rows)
    order_t = ["2", "3", "4", "5", "full"]
    lines = ["| estimator | " + " | ".join(order_t) + " |", "|" + "|".join(["---"] * (len(order_t) + 1)) + "|"]
    for est in ESTIMATORS:
        cells = [est]
        for t in order_t:
            part = df[(df["estimator"] == est) & (df["T_label"] == t)]
            cells.append(f"{float(part.iloc[0][metric]):.3f}" if len(part) and np.isfinite(float(part.iloc[0][metric])) else "NA")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_report(result: dict[str, Any]) -> str:
    sample_rows = result["sample"]["sample_table"]
    cross = result["truth_crosscheck"]
    err = result["error_summary"]
    cls = result["classification_summary"]
    honesty = result["honesty_checks"]
    verdict = result["verdict"]
    lines = [
        "# Gate - Truncation Recovery + Competing Baselines, Cycle F",
        "",
        f"- VERDICT: {verdict['verdict']} - {verdict['reason']}",
        "- Deterministic LM only. No NUTS, SNPE, gamma, registration, or ML residual.",
        "- Manoeuvre rule: Pre, SPAACC == Y, SPAQEFF == A for this complete-curve gate, max parsed FEV1, ties to lowest manoeuvre number.",
        f"- Every truncated estimator used only volume samples up to T and the protocol-fixed {PREDICTION_HORIZON_S:.0f} s prediction horizon; the complete curve was used only for scoring.",
        "",
        "## Motivation",
        "",
        "Prior gates used complete A/B curves, where direct measurement is exact by construction. This gate tests the Steltner-style use case: recovering FVC and obstruction calls from incomplete manoeuvres where direct end-volume underestimates FVC and inflates FEV1/FVC.",
        "",
        "## Sample",
        "",
        md_table(sample_rows, [("stratum", "stratum", "str"), ("selected_grade_A_n", "grade-A selected", "int"), ("complete_plateau_n", "complete plateau", "int"), ("sampled_n", "sampled", "int"), ("dropped_by_plateau_or_missing_nhanes", "drop before sample", "int"), ("dropped_by_sampling_cap", "drop by cap", "int")]),
        "",
        "Complete plateau rule: FET >= 6.0 s and terminal-second dV <= 0.025 L.",
        "",
        "## TRUE vs NHANES Cross-Check",
        "",
        f"- Cross-check rule: {cross['pass_rule']}",
        f"- Result: {'PASS' if cross['passed'] else 'FAIL'} on n={cross['n']}",
        "",
        "| metric | signed median | abs median | abs IQR |",
        "|---|---:|---:|---:|",
        f"| FEV1 percent error | {cross['fev1_abs_pct']['signed_median']:.2f}% | {cross['fev1_abs_pct']['median']:.2f}% | {cross['fev1_abs_pct']['iqr'][0]:.2f}% to {cross['fev1_abs_pct']['iqr'][1]:.2f}% |",
        f"| FVC percent error | {cross['fvc_abs_pct']['signed_median']:.2f}% | {cross['fvc_abs_pct']['median']:.2f}% | {cross['fvc_abs_pct']['iqr'][0]:.2f}% to {cross['fvc_abs_pct']['iqr'][1]:.2f}% |",
        f"| FEV1/FVC absolute error | {cross['ratio_abs']['signed_median']:.4f} | {cross['ratio_abs']['median']:.4f} | {cross['ratio_abs']['iqr'][0]:.4f} to {cross['ratio_abs']['iqr'][1]:.4f} |",
        "",
    ]
    if not cross["passed"]:
        lines.extend(["Cross-check failed, so estimator scoring was halted.", ""])
        return "\n".join(lines)

    lines.extend(
        [
            "## Step 2 - Error Tables",
            "",
            "Full per-estimator x T x stratum error summaries are below.",
            "",
            md_table(
                err,
                [
                    ("estimator", "estimator", "str"),
                    ("T_label", "T", "str"),
                    ("truth_severity", "stratum", "str"),
                    ("n", "n", "int"),
                    ("fit_failure_n", "fail", "int"),
                    ("fvc_signed_pct_median", "FVC signed % med", "float2"),
                    ("fvc_abs_pct_median", "FVC abs % med", "float2"),
                    ("ratio_signed_median", "ratio signed med", "float4"),
                    ("ratio_abs_median", "ratio abs med", "float4"),
                ],
            ),
            "",
            "## Step 3 - Classification Consequence",
            "",
            "False-negative rate on obstructed-truth:",
            "",
            matrix_table(cls, "false_negative_rate_obstructed"),
            "",
            "Borderline-stratum false-negative rate:",
            "",
            matrix_table(cls, "borderline_false_negative_rate"),
            "",
            "Sensitivity:",
            "",
            matrix_table(cls, "sensitivity"),
            "",
            "Specificity:",
            "",
            matrix_table(cls, "specificity"),
            "",
            "Cohen's kappa:",
            "",
            matrix_table(cls, "cohens_kappa"),
            "",
            "## Step 4 - Physics vs Smoothness",
            "",
            verdict["physics_vs_smoothness_sentence"],
            "",
            verdict["biexponential_vs_mono_sentence"],
            "",
            "## Step 5 - Regime Map",
            "",
            md_table(result["regime_map"], [("estimator", "estimator", "str"), ("truth_severity", "stratum", "str"), ("minimum_T_for_median_abs_FVC_error_lt_5pct", "min T for median abs FVC <5%", "str")]),
            "",
            "## Step 6 - Honesty Checks",
            "",
            f"- Original Cycle-F obstructed participants: {honesty['original_cycle_f_obstructed_n']}",
            f"- Obstructed participants in this complete-plateau sample: {honesty['sample_obstructed_n']}",
            f"- Excluded original obstructed participants: {honesty['excluded_original_obstructed_n']}; NHANES ratio median {honesty['excluded_original_obstructed_ratio_median']:.4f}, IQR {honesty['excluded_original_obstructed_ratio_iqr'][0]:.4f} to {honesty['excluded_original_obstructed_ratio_iqr'][1]:.4f}.",
            f"- No truncated estimator received information from after T: {honesty['no_estimator_saw_untruncated_information_after_T']}. {honesty['note']}",
            "",
            "E1 OMBE overshoot/undershoot by T:",
            "",
            md_table(honesty["e1_overshoot_by_T"], [("T_label", "T", "str"), ("valid_n", "valid", "int"), ("overshoot_n", "overshoot n", "int"), ("overshoot_fraction", "overshoot frac", "float3"), ("overshoot_pct_median", "overshoot % med", "float2"), ("overshoot_pct_max", "overshoot % max", "float2"), ("undershoot_n", "undershoot n", "int"), ("undershoot_pct_median", "undershoot % med", "float2")]),
            "",
            f"VERDICT: {verdict['verdict']} - {verdict['reason']}",
            "",
        ]
    )
    return "\n".join(lines)


def run(root: Path) -> dict[str, Any]:
    column_map = load_column_map(root)
    truth = load_truth(root, column_map)
    raw_path, raw_note = find_or_unzip_raw(root)
    raw, raw_info = read_raw(raw_path)
    selected_a, selected_complete = participant_candidates(raw, truth)
    sample, sample_meta = sample_by_stratum(selected_a, selected_complete)
    cross = crosscheck(sample)
    if not cross["passed"]:
        result = {
            "verdict": {"verdict": "HALTED", "reason": "TRUE_* vs NHANES cross-check failed."},
            "raw_source": str(raw_path),
            "raw_source_note": raw_note,
            "raw_loaded_rows": int(raw_info["loaded_rows"]),
            "sample": sample_meta,
            "truth_crosscheck": cross,
        }
        out = root / "results"
        out.mkdir(exist_ok=True)
        (out / "gate_truncation_recovery_F.md").write_text(make_report(result), encoding="utf-8")
        (out / "gate_truncation_recovery_F.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
        pd.DataFrame().to_csv(out / "gate_truncation_recovery_F_rows.csv", index=False)
        return result
    rows = run_estimators(sample, raw)
    err = error_summary(rows)
    cls = classification_summary(rows)
    reg = regime_map(err)
    honest = honesty_checks(rows, truth, sample)
    verdict = comparison_verdict(err, cls)
    result = {
        "verdict": verdict,
        "raw_source": str(raw_path),
        "raw_source_note": raw_note,
        "raw_loaded_rows": int(raw_info["loaded_rows"]),
        "sample": sample_meta,
        "truth_crosscheck": cross,
        "error_summary": err,
        "classification_summary": cls,
        "regime_map": reg,
        "honesty_checks": honest,
        "estimator_definitions": {
            "E0_DIRECT": "FVC is observed truncated end-volume.",
            "E1_OMBE": f"Two-compartment OMBE fit to volume samples up to T; FVC is model volume at the fixed {PREDICTION_HORIZON_S:.0f} s horizon.",
            "E2_SPLINE": f"Cubic smoothing spline fit to samples up to T; FVC is evaluated at the fixed {PREDICTION_HORIZON_S:.0f} s horizon.",
            "E3_LINEAR_TAIL": f"Line fit to the final 1.0 s before T; FVC is evaluated at the fixed {PREDICTION_HORIZON_S:.0f} s horizon.",
            "E4_MONO_EXPONENTIAL": f"Single exponential with onset rise fit like E1; FVC is model volume at the fixed {PREDICTION_HORIZON_S:.0f} s horizon.",
        },
    }
    out = root / "results"
    out.mkdir(exist_ok=True)
    rows.to_csv(out / "gate_truncation_recovery_F_rows.csv", index=False)
    (out / "gate_truncation_recovery_F.md").write_text(make_report(result), encoding="utf-8")
    (out / "gate_truncation_recovery_F.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    print(f"Truncation recovery gate: {result['verdict']['verdict']} - {result['verdict']['reason']}")
    if result["verdict"]["verdict"] != "HALTED":
        cls = pd.DataFrame(result["classification_summary"])
        for t in ["2", "3"]:
            e0 = cls[(cls["estimator"] == "E0_DIRECT") & (cls["T_label"] == t)]["false_negative_rate_obstructed"].iloc[0]
            e1 = cls[(cls["estimator"] == "E1_OMBE") & (cls["T_label"] == t)]["false_negative_rate_obstructed"].iloc[0]
            print(f"T={t}s obstructed FN: E0={e0:.3f}, E1={e1:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
