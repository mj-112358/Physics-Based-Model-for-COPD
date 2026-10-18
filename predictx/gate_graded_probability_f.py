from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .gate_covariate_resolution_f import load_covariates
from .gate_deferral_recovery_f import (
    apply_recovery,
    corrected_maneuvers,
    correction_offset,
    exact_ci,
    final_metrics,
)
from .gate_workflow_policy_f import (
    RANDOM_SEED,
    add_repeatability,
    add_workflow_columns,
    clean_json,
    fmt,
    load_inputs,
    md_table,
)

OUTPUT_STEM = "gate_graded_probability_F"
FIXED_RATIO_THRESHOLD = 0.70
CORR_QUARANTINE_THRESHOLD = 0.60
CALIBRATION_ECE_WARN = 0.10
PPV_NPV_FLOOR = 0.90
BOOTSTRAP_N = 1000
PERMUTATION_N = 20

BANNED_EXACT = {
    "fev1_fvc_ratio",
    "fev1_mL",
    "fvc_mL",
    "parsed_ratio",
    "parsed_fev1_l",
    "parsed_fvc_l",
    "truth_severity",
}

EXPECTED_PROXY_EXCLUDE = {
    "gli_lln_ratio": "standing task identifies GLI LLN as a ratio/threshold proxy; measured max |corr| is within rounding of 0.60",
}

CANDIDATE_FEATURES = [
    "model_ratio",
    "corrected_ratio_c1",
    "corrected_ratio_c2",
    "gli_lln_ratio",
    "fet_s",
    "duration_s",
    "volume_r2",
    "volume_nrss_norm",
    "fit_tau_f",
    "fit_tau_s",
    "fit_alpha",
    "fit_tau_o",
    "dV_last_L",
    "maneuver_count",
    "model_ratio_sd",
    "model_fev1_sd_l",
    "model_fvc_sd_l",
    "model_classification_majority_fraction",
]

RATIO_AWARE_FEATURES = ["model_ratio", "corrected_ratio_c1"]
DEMOGRAPHIC_BASE_FEATURES = ["RIDAGEYR", "BMXHT", "sex_male", "ever_smoker"]


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def truth_obstructed(df: pd.DataFrame) -> pd.Series:
    return numeric(df["fev1_fvc_ratio"]) < FIXED_RATIO_THRESHOLD


def auc_safe(y: np.ndarray, p: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(p, dtype=float)
    mask = np.isfinite(p_arr)
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)[mask]
    else:
        w = None
    y_arr = y_arr[mask]
    p_arr = p_arr[mask]
    if len(np.unique(y_arr)) < 2:
        return np.nan
    return float(roc_auc_score(y_arr, p_arr, sample_weight=w))


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, seed: int, sample_weight: np.ndarray | None = None) -> dict[str, Any]:
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(p, dtype=float)
    valid = np.isfinite(p_arr)
    y_arr = y_arr[valid]
    p_arr = p_arr[valid]
    weights = np.asarray(sample_weight, dtype=float)[valid] if sample_weight is not None else None
    point = auc_safe(y_arr, p_arr, weights)
    if len(y_arr) < 10 or len(np.unique(y_arr)) < 2:
        return {"auc": point, "ci_low": np.nan, "ci_high": np.nan, "bootstrap_n": 0}
    rng = np.random.default_rng(seed)
    draws = []
    idx = np.arange(len(y_arr))
    for _ in range(BOOTSTRAP_N):
        draw = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_arr[draw])) < 2:
            continue
        w_draw = weights[draw] if weights is not None else None
        draws.append(auc_safe(y_arr[draw], p_arr[draw], w_draw))
    if not draws:
        return {"auc": point, "ci_low": np.nan, "ci_high": np.nan, "bootstrap_n": 0}
    lo, hi = np.nanpercentile(np.asarray(draws, dtype=float), [2.5, 97.5])
    return {"auc": point, "ci_low": float(lo), "ci_high": float(hi), "bootstrap_n": int(len(draws))}


def class_reweight(y: np.ndarray, target_prevalence: float, constructed_prevalence: float) -> np.ndarray:
    y_arr = np.asarray(y, dtype=int)
    pos = max(min(float(constructed_prevalence), 1.0 - 1e-9), 1e-9)
    target = max(min(float(target_prevalence), 1.0 - 1e-9), 1e-9)
    weights = np.where(y_arr == 1, target / pos, (1.0 - target) / (1.0 - pos))
    return weights.astype(float)


def weighted_rate(mask: np.ndarray, weights: np.ndarray | None = None) -> float:
    arr = np.asarray(mask, dtype=bool)
    if weights is None:
        return float(arr.mean()) if len(arr) else np.nan
    w = np.asarray(weights, dtype=float)
    denom = float(np.sum(w))
    return float(np.sum(w[arr]) / denom) if denom > 0 else np.nan


def ppv_npv(y: np.ndarray, pred: np.ndarray, weights: np.ndarray | None = None) -> dict[str, Any]:
    yb = np.asarray(y, dtype=bool)
    pb = np.asarray(pred, dtype=bool)
    if weights is None:
        w = np.ones(len(yb), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    tp = float(np.sum(w[yb & pb]))
    tn = float(np.sum(w[~yb & ~pb]))
    fp = float(np.sum(w[~yb & pb]))
    fn = float(np.sum(w[yb & ~pb]))
    return {
        "ppv": float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan,
        "npv": float(tn / (tn + fn)) if (tn + fn) > 0 else np.nan,
        "tp_w": tp,
        "tn_w": tn,
        "fp_w": fp,
        "fn_w": fn,
    }


def kappa_from_calls(y: np.ndarray, pred: np.ndarray) -> float:
    y_arr = np.asarray(y, dtype=int)
    pred_arr = np.asarray(pred, dtype=int)
    if len(y_arr) == 0 or len(np.unique(pred_arr)) < 1:
        return np.nan
    return float(cohen_kappa_score(y_arr, pred_arr))


def load_model_frame(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    policy, repro = load_inputs(root)
    workflow = add_workflow_columns(add_repeatability(policy, repro))
    c1 = correction_offset(workflow)
    maneuvers = corrected_maneuvers(repro, c1)
    recovered = apply_recovery(workflow[workflow["split"].eq("test")].copy(), maneuvers)
    return workflow, repro, recovered, c1


def participant_summary(repro: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "SEQN",
        "maneuver_count",
        "model_ratio_sd",
        "model_fev1_sd_l",
        "model_fvc_sd_l",
        "model_classification_majority_fraction",
    ]
    return repro[repro["row_type"].eq("participant_summary")][cols].drop_duplicates("SEQN").copy()


def correlation_pair(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    x_num = numeric(x)
    y_num = numeric(y)
    valid = x_num.notna() & y_num.notna()
    if int(valid.sum()) < 3:
        return np.nan, np.nan, int(valid.sum())
    pear = float(pearsonr(x_num[valid], y_num[valid]).statistic)
    spear = float(spearmanr(x_num[valid], y_num[valid]).statistic)
    return pear, spear, int(valid.sum())


def feature_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    train = df[df["split"].eq("train")].copy()
    test = df[df["split"].eq("test")].copy()
    rows = []
    clean: list[str] = []
    quarantined: list[str] = []
    banned_present = sorted(
        [
            col
            for col in df.columns
            if col in BANNED_EXACT or col.endswith("_obstructed") or "_obstructed_" in col or col.startswith("observed_obstructed")
        ]
    )
    for feature in CANDIDATE_FEATURES:
        if feature not in df.columns:
            rows.append(
                {
                    "feature": feature,
                    "partition": "MISSING",
                    "pearson_corr_train_ratio": np.nan,
                    "spearman_corr_train_ratio": np.nan,
                    "max_abs_corr": np.nan,
                    "corr_n": 0,
                    "missing_train_fraction": np.nan,
                    "missing_test_fraction": np.nan,
                    "rationale": "candidate not present in merged frame",
                }
            )
            continue
        pear, spear, n = correlation_pair(train[feature], train["fev1_fvc_ratio"])
        max_abs = float(np.nanmax(np.abs([pear, spear]))) if np.isfinite(pear) or np.isfinite(spear) else np.nan
        is_banned = feature in BANNED_EXACT or feature.endswith("_obstructed") or "_obstructed_" in feature
        corr_quarantine = bool(np.isfinite(max_abs) and max_abs > CORR_QUARANTINE_THRESHOLD)
        expected_proxy = feature in EXPECTED_PROXY_EXCLUDE
        if is_banned:
            partition = "BANNED"
            rationale = "explicit label component or observed-obstruction label"
        elif corr_quarantine:
            partition = "QUARANTINED"
            rationale = f"TRAIN max |corr(feature, observed ratio)| = {max_abs:.3f} > {CORR_QUARANTINE_THRESHOLD:.2f}"
        elif expected_proxy:
            partition = "QUARANTINED"
            rationale = EXPECTED_PROXY_EXCLUDE[feature]
        else:
            partition = "CLEAN"
            rationale = "passes banned-name and TRAIN correlation audit"
            clean.append(feature)
        if partition == "QUARANTINED":
            quarantined.append(feature)
        rows.append(
            {
                "feature": feature,
                "partition": partition,
                "pearson_corr_train_ratio": pear,
                "spearman_corr_train_ratio": spear,
                "max_abs_corr": max_abs,
                "corr_n": n,
                "missing_train_fraction": float(train[feature].isna().mean()),
                "missing_test_fraction": float(test[feature].isna().mean()),
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows), banned_present, quarantined, clean


def impute_frames(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    medians: dict[str, float] = {}
    train_out = pd.DataFrame(index=train.index)
    test_out = pd.DataFrame(index=test.index)
    missing_rows = []
    for feature in features:
        tr = numeric(train[feature])
        te = numeric(test[feature])
        median = float(tr.median()) if tr.notna().any() else 0.0
        medians[feature] = median
        train_out[feature] = tr.fillna(median)
        test_out[feature] = te.fillna(median)
        missing_rows.append(
            {
                "feature": feature,
                "train_missing_fraction": float(tr.isna().mean()),
                "test_missing_fraction": float(te.isna().mean()),
                "train_median_imputation": median,
                "flag_gt_20pct_missing": bool(tr.isna().mean() > 0.20 or te.isna().mean() > 0.20),
            }
        )
    return train_out, test_out, {"medians": medians, "missingness": missing_rows}


def make_gbt() -> CalibratedClassifierCV:
    base = GradientBoostingClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.8,
        random_state=RANDOM_SEED,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    return CalibratedClassifierCV(estimator=base, method="isotonic", cv=cv)


def make_logistic() -> CalibratedClassifierCV:
    base = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, solver="lbfgs", random_state=RANDOM_SEED),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    return CalibratedClassifierCV(estimator=base, method="isotonic", cv=cv)


def fit_model(model: CalibratedClassifierCV, x_train: pd.DataFrame, y_train: np.ndarray) -> CalibratedClassifierCV:
    fitted = clone(model)
    fitted.fit(x_train, y_train)
    return fitted


def probability(model: CalibratedClassifierCV, x: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x)[:, 1].astype(float)


def boundary_mask(df: pd.DataFrame) -> pd.Series:
    ratio = numeric(df["fev1_fvc_ratio"])
    return ratio.ge(0.65) & ratio.lt(0.75)


def model_auc_report(
    name: str,
    model: CalibratedClassifierCV,
    x_test: pd.DataFrame,
    test: pd.DataFrame,
    weights: np.ndarray,
    seed_offset: int,
) -> dict[str, Any]:
    y = truth_obstructed(test).astype(int).to_numpy()
    p = probability(model, x_test)
    band = boundary_mask(test).to_numpy()
    whole = bootstrap_auc_ci(y, p, RANDOM_SEED + seed_offset)
    whole_rw = bootstrap_auc_ci(y, p, RANDOM_SEED + seed_offset + 100, sample_weight=weights)
    boundary = bootstrap_auc_ci(y[band], p[band], RANDOM_SEED + seed_offset + 200)
    boundary_rw = bootstrap_auc_ci(y[band], p[band], RANDOM_SEED + seed_offset + 300, sample_weight=weights[band])
    return {
        "model": name,
        "test_auc": whole,
        "test_auc_reweighted": whole_rw,
        "boundary_definition": "true fev1_fvc_ratio in [0.65, 0.75); true [0.60, 0.70) is one-class-only and AUC-undefined",
        "boundary_n": int(band.sum()),
        "boundary_positive_n": int(y[band].sum()),
        "boundary_negative_n": int((1 - y[band]).sum()),
        "boundary_auc": boundary,
        "boundary_auc_reweighted": boundary_rw,
    }


def reliability_bins(
    y: np.ndarray,
    p: np.ndarray,
    weights: np.ndarray | None,
    scope: str,
    weighting: str,
    target_prevalence: float,
    n_bins: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(p, dtype=float)
    valid = np.isfinite(p_arr)
    y_arr = y_arr[valid]
    p_arr = p_arr[valid]
    if weights is None:
        w = np.ones(len(y_arr), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)[valid]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    total_w = float(np.sum(w))
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p_arr >= lo) & (p_arr <= hi)
        else:
            mask = (p_arr >= lo) & (p_arr < hi)
        n = int(mask.sum())
        w_sum = float(np.sum(w[mask]))
        mean_p = float(np.average(p_arr[mask], weights=w[mask])) if w_sum > 0 else np.nan
        frac_pos = float(np.average(y_arr[mask], weights=w[mask])) if w_sum > 0 else np.nan
        abs_gap = abs(mean_p - frac_pos) if np.isfinite(mean_p) and np.isfinite(frac_pos) else np.nan
        if np.isfinite(abs_gap) and total_w > 0:
            ece += (w_sum / total_w) * abs_gap
        rows.append(
            {
                "scope": scope,
                "weighting": weighting,
                "target_prevalence": target_prevalence,
                "bin": i,
                "bin_low": lo,
                "bin_high": hi,
                "n": n,
                "weight_sum": w_sum,
                "mean_p_obs": mean_p,
                "observed_fraction": frac_pos,
                "abs_gap": abs_gap,
            }
        )
    brier = float(np.average((p_arr - y_arr) ** 2, weights=w)) if len(y_arr) and float(np.sum(w)) > 0 else np.nan
    return rows, {"scope": scope, "weighting": weighting, "ece": float(ece), "brier": brier, "n": int(len(y_arr))}


def choose_boundaries(
    train: pd.DataFrame,
    p_train: np.ndarray,
    train_weights: np.ndarray,
) -> dict[str, Any]:
    y = truth_obstructed(train).astype(int).to_numpy()
    severe = train["truth_severity"].astype(str).eq("severe").to_numpy()
    probs = np.unique(np.round(np.asarray(p_train, dtype=float), 12))
    lows = np.r_[0.0, probs]
    highs = np.r_[probs, 1.0]
    best: dict[str, Any] | None = None
    for low in lows:
        for high in highs:
            if not low < high:
                continue
            normal = p_train <= low
            obstructed = p_train >= high
            if int(normal.sum()) == 0 or int(obstructed.sum()) == 0:
                continue
            severe_fn = int((normal & severe).sum())
            if severe_fn > 0:
                continue
            pred = np.full(len(y), -1, dtype=int)
            pred[normal] = 0
            pred[obstructed] = 1
            ppvnpv = ppv_npv(y[pred != -1], pred[pred != -1], train_weights[pred != -1])
            if not np.isfinite(ppvnpv["ppv"]) or not np.isfinite(ppvnpv["npv"]):
                continue
            if ppvnpv["ppv"] < PPV_NPV_FLOOR or ppvnpv["npv"] < PPV_NPV_FLOOR:
                continue
            coverage = float((normal | obstructed).mean())
            candidate = {
                "p_low": float(low),
                "p_high": float(high),
                "train_confident_coverage": coverage,
                "train_confident_n": int((normal | obstructed).sum()),
                "train_confident_normal_n": int(normal.sum()),
                "train_confident_obstructed_n": int(obstructed.sum()),
                "train_ppv_reweighted": ppvnpv["ppv"],
                "train_npv_reweighted": ppvnpv["npv"],
                "train_severe_fn_confident_normal": severe_fn,
            }
            key = (
                candidate["train_confident_coverage"],
                min(candidate["train_ppv_reweighted"], candidate["train_npv_reweighted"]),
                candidate["p_low"] - candidate["p_high"],
            )
            if best is None or key > best["_key"]:
                candidate["_key"] = key
                best = candidate
    if best is None:
        return {
            "p_low": 0.0,
            "p_high": 1.0,
            "train_confident_coverage": 0.0,
            "train_confident_n": 0,
            "train_confident_normal_n": 0,
            "train_confident_obstructed_n": 0,
            "train_ppv_reweighted": np.nan,
            "train_npv_reweighted": np.nan,
            "train_severe_fn_confident_normal": 0,
            "boundary_status": "NO_FEASIBLE_TWO_SIDED_BOUNDARIES",
        }
    best.pop("_key", None)
    best["boundary_status"] = "FEASIBLE"
    return best


def apply_zones(df: pd.DataFrame, p: np.ndarray, boundaries: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out["p_obs"] = p
    low = float(boundaries["p_low"])
    high = float(boundaries["p_high"])
    out["graded_zone"] = "INDETERMINATE_GRADED"
    out.loc[out["p_obs"] <= low, "graded_zone"] = "CONFIDENT_NORMAL"
    out.loc[out["p_obs"] >= high, "graded_zone"] = "CONFIDENT_OBSTRUCTED"
    out["graded_confident"] = out["graded_zone"].isin(["CONFIDENT_NORMAL", "CONFIDENT_OBSTRUCTED"])
    out["graded_pred_obstructed"] = out["graded_zone"].eq("CONFIDENT_OBSTRUCTED")
    out["truth_obstructed"] = truth_obstructed(out)
    return out


def all_indeterminate_zones(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["p_obs"] = p
    out["graded_zone"] = "INDETERMINATE_GRADED"
    out["graded_confident"] = False
    out["graded_pred_obstructed"] = False
    out["truth_obstructed"] = truth_obstructed(out)
    return out


def zone_metrics(rows: pd.DataFrame, weights: np.ndarray) -> dict[str, Any]:
    zone_counts = []
    total = int(len(rows))
    for zone, part in rows.groupby("graded_zone", dropna=False):
        zone_counts.append({"zone": str(zone), "n": int(len(part)), "fraction": float(len(part) / total) if total else np.nan})
    confident = rows[rows["graded_confident"].astype(bool)].copy()
    y = confident["truth_obstructed"].astype(int).to_numpy()
    pred = confident["graded_pred_obstructed"].astype(int).to_numpy()
    w = weights[confident.index.to_numpy()] if len(confident) else np.array([], dtype=float)
    if len(confident):
        tp = int(((y == 1) & (pred == 1)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        ppvnpv_raw = ppv_npv(y, pred, None)
        ppvnpv_rw = ppv_npv(y, pred, w)
        severe_conf = confident[confident["truth_severity"].eq("severe")]
        severe_fn = int((severe_conf["graded_zone"].eq("CONFIDENT_NORMAL")).sum())
        lo, hi = exact_ci(severe_fn, int(len(severe_conf)))
    else:
        tp = tn = fp = fn = severe_fn = 0
        ppvnpv_raw = {"ppv": np.nan, "npv": np.nan}
        ppvnpv_rw = {"ppv": np.nan, "npv": np.nan}
        lo = hi = np.nan
    return {
        "zone_counts": sorted(zone_counts, key=lambda item: item["zone"]),
        "confident_n": int(len(confident)),
        "confident_coverage": float(len(confident) / len(rows)) if len(rows) else np.nan,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float((tp + tn) / len(confident)) if len(confident) else np.nan,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "cohens_kappa": kappa_from_calls(y, pred) if len(confident) else np.nan,
        "ppv_raw": ppvnpv_raw["ppv"],
        "npv_raw": ppvnpv_raw["npv"],
        "ppv_reweighted": ppvnpv_rw["ppv"],
        "npv_reweighted": ppvnpv_rw["npv"],
        "severe_confident_n": int(len(severe_conf)) if len(confident) else 0,
        "severe_fn_n": severe_fn,
        "severe_fn_ci_low": lo,
        "severe_fn_ci_high": hi,
    }


def prevalence_from_full_cohort(root: Path) -> dict[str, Any]:
    path = root / "results" / "merged_summary.parquet"
    if path.exists():
        full = pd.read_parquet(path)
        valid = full[numeric(full["fev1_fvc_ratio"]).notna()].copy()
        prevalence = float((numeric(valid["fev1_fvc_ratio"]) < FIXED_RATIO_THRESHOLD).mean()) if len(valid) else np.nan
        return {
            "path": "empirical_full_unenriched_cohort",
            "source": str(path),
            "valid_ratio_n": int(len(valid)),
            "obstructed_n": int((numeric(valid["fev1_fvc_ratio"]) < FIXED_RATIO_THRESHOLD).sum()),
            "target_prevalence": prevalence,
            "sensitivity_prevalences": [0.05, 0.10, 0.15, 0.20],
        }
    return {
        "path": "fallback_reference_sensitivity_only",
        "source": "merged_summary.parquet not found; using requested sensitivity sweep only",
        "valid_ratio_n": 0,
        "obstructed_n": 0,
        "target_prevalence": 0.10,
        "sensitivity_prevalences": [0.05, 0.10, 0.15, 0.20],
    }


def permutation_rows(model: CalibratedClassifierCV, x_test: pd.DataFrame, y_test: np.ndarray) -> list[dict[str, Any]]:
    if len(np.unique(y_test)) < 2:
        return []
    result = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=PERMUTATION_N,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    rows = []
    for feature, mean, sd in zip(x_test.columns, result.importances_mean, result.importances_std):
        rows.append({"feature": feature, "importance_mean_auc_drop": float(mean), "importance_sd": float(sd)})
    return sorted(rows, key=lambda item: item["importance_mean_auc_drop"], reverse=True)


def add_demographics(root: Path, df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    cov, notes, _variables = load_covariates(root)
    if cov.empty:
        out = df.copy()
        for col in DEMOGRAPHIC_BASE_FEATURES:
            out[col] = np.nan
        return out, [], notes
    keep = ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "BMXHT", "smoking_status"]
    cov2 = cov[[c for c in keep if c in cov.columns]].drop_duplicates("SEQN").copy()
    cov2["sex_male"] = np.where(numeric(cov2.get("RIAGENDR", pd.Series(index=cov2.index))) == 1.0, 1.0, 0.0)
    cov2["ever_smoker"] = np.where(cov2.get("smoking_status", "").isin(["current", "former", "former_unknown_current_status"]), 1.0, 0.0)
    race = pd.get_dummies(cov2.get("RIDRETH1"), prefix="race", dummy_na=True).astype(float)
    cov2 = pd.concat([cov2, race], axis=1)
    out = df.merge(cov2.drop(columns=["smoking_status"], errors="ignore"), on="SEQN", how="left")
    deferred_test = out[out["split"].eq("test") & out.get("post_recovery_finalized", False).eq(False)] if "post_recovery_finalized" in out else out[out["split"].eq("test")]
    availability = []
    for feature in ["RIDAGEYR", "BMXHT", "RIAGENDR", "RIDRETH1", "ever_smoker"]:
        if feature not in deferred_test.columns:
            availability.append({"feature": feature, "available_n": 0, "deferred_test_n": int(len(deferred_test)), "available_fraction": 0.0})
        else:
            available = numeric(deferred_test[feature]).notna()
            availability.append(
                {
                    "feature": feature,
                    "available_n": int(available.sum()),
                    "deferred_test_n": int(len(deferred_test)),
                    "available_fraction": float(available.mean()) if len(available) else np.nan,
                }
            )
    return out, availability, notes


def forced_deferred_accuracy(rows: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, mask in [
        ("strict", rows["split"].eq("test") & rows["workflow_final_diagnostic"].eq(False)),
        ("operational", rows["split"].eq("test") & rows["post_recovery_finalized"].eq(False)),
    ]:
        deferred = rows[mask].copy()
        if deferred.empty:
            out[f"{label}_deferred_n"] = 0
            out[f"{label}_forced_accuracy_at_0p5"] = np.nan
            out[f"{label}_correct_n"] = 0
            continue
        pred = deferred["p_obs"].astype(float) >= 0.5
        truth = deferred["truth_obstructed"].astype(bool)
        out[f"{label}_deferred_n"] = int(len(deferred))
        out[f"{label}_forced_accuracy_at_0p5"] = float(pred.eq(truth).mean())
        out[f"{label}_correct_n"] = int(pred.eq(truth).sum())
    return {
        **out,
    }


def knife_edge_decomposition(rows: pd.DataFrame) -> list[dict[str, Any]]:
    deferred = rows[rows["split"].eq("test") & rows["post_recovery_finalized"].eq(False)].copy()
    dist = (numeric(deferred["fev1_fvc_ratio"]) - FIXED_RATIO_THRESHOLD).abs()
    specs = [
        ("true_abs_lt_0.02", dist < 0.02),
        ("true_abs_0.02_to_0.05", (dist >= 0.02) & (dist < 0.05)),
        ("true_abs_ge_0.05", dist >= 0.05),
    ]
    out = []
    for label, mask in specs:
        part = deferred[mask].copy()
        recovered = int(part["graded_confident"].sum()) if "graded_confident" in part else 0
        out.append(
            {
                "bucket": label,
                "n": int(len(part)),
                "fraction_deferred": float(len(part) / len(deferred)) if len(deferred) else np.nan,
                "graded_confident_recovered_n": recovered,
                "graded_confident_recovered_fraction": float(recovered / len(part)) if len(part) else np.nan,
            }
        )
    return out


def sensitivity_metrics(y: np.ndarray, p: np.ndarray, p_low: float, p_high: float, constructed_prev: float, prevalences: list[float]) -> list[dict[str, Any]]:
    rows = []
    pred_zone = np.full(len(y), -1, dtype=int)
    pred_zone[p <= p_low] = 0
    pred_zone[p >= p_high] = 1
    conf = pred_zone != -1
    for prev in prevalences:
        w = class_reweight(y, prev, constructed_prev)
        rel_rows, rel = reliability_bins(y, p, w, "test_all", f"reweighted_prev_{prev:.2f}", prev)
        ppvnpv = ppv_npv(y[conf], pred_zone[conf], w[conf]) if conf.any() else {"ppv": np.nan, "npv": np.nan}
        rows.append(
            {
                "target_prevalence": prev,
                "auc_reweighted": auc_safe(y, p, w),
                "ppv_reweighted": ppvnpv["ppv"],
                "npv_reweighted": ppvnpv["npv"],
                "ece_reweighted": rel["ece"],
                "brier_reweighted": rel["brier"],
                "calibration_bins": rel_rows,
            }
        )
    return rows


def verdict(result: dict[str, Any]) -> tuple[str, str]:
    if result["leakage_guard_tripped"]:
        return "LEAKAGE-GUARD-TRIPPED", "A banned or quarantined feature reached the clean fitted feature list; downstream clean claims are invalid."
    zone = result["graded_zone_metrics"]
    calib = result["calibration_summary"]["test_all_reweighted"]
    if np.isfinite(calib["ece"]) and calib["ece"] > CALIBRATION_ECE_WARN:
        return "GRADED-UNCALIBRATED", "Reweighted TEST ECE exceeds the conservative 0.10 calibration warning threshold."
    if result["candidate_zone_rejected"]:
        return "GRADED-NEUTRAL", "TRAIN-frozen candidate zones failed TEST safety or PPV/NPV floors, so retained output is probability-only indeterminate."
    operational = result["deterministic_baseline"]["operational_recovery"]["finalized_fraction"]
    if (
        zone["confident_coverage"] >= operational + 0.03
        and zone["ppv_reweighted"] >= PPV_NPV_FLOOR
        and zone["npv_reweighted"] >= PPV_NPV_FLOOR
        and zone["severe_fn_n"] == 0
    ):
        return "GRADED-IMPROVES-COVERAGE", "Confident graded coverage materially exceeds the operational deterministic baseline at the frozen PPV/NPV floors."
    return "GRADED-NEUTRAL", "The graded arm's value is calibrated probability output for indeterminate cases rather than a material confident-coverage gain."


def write_outputs(
    root: Path,
    result: dict[str, Any],
    audit: pd.DataFrame,
    calibration_rows: list[dict[str, Any]],
    participants: pd.DataFrame,
) -> None:
    out = root / "results"
    out.mkdir(exist_ok=True)
    (out / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    audit.to_csv(out / f"{OUTPUT_STEM}_features_audit.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(out / f"{OUTPUT_STEM}_calibration.csv", index=False)
    participants.to_csv(out / f"{OUTPUT_STEM}_participants.csv", index=False)

    baseline = result["deterministic_baseline"]
    clean_auc = result["clean_model_auc"]
    ratio_auc = result["ratio_aware_auc"]
    zone = result["graded_zone_metrics"]
    lines = [
        "# Gate - Leakage-Free Calibrated Obstruction Classifier + Graded Probability, Cycle F",
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
            "## Part 0 - Feature Assembly + Leakage Audit",
            "",
            f"- Banned columns present in source frame: {', '.join(result['banned_present'])}.",
            f"- Quarantined candidates: {', '.join(result['quarantined_features']) if result['quarantined_features'] else 'none'}.",
            f"- CLEAN fitted features: {', '.join(result['clean_features'])}.",
            f"- Leakage guard tripped: {result['leakage_guard_tripped']}.",
            "",
            "Top audit rows by max absolute TRAIN correlation to observed ratio:",
            "",
            md_table(
                result["feature_audit_top_corr"],
                [
                    ("feature", "feature", "str"),
                    ("partition", "partition", "str"),
                    ("pearson_corr_train_ratio", "pearson", "float3"),
                    ("spearman_corr_train_ratio", "spearman", "float3"),
                    ("max_abs_corr", "max abs corr", "float3"),
                    ("missing_train_fraction", "train missing", "float3"),
                    ("missing_test_fraction", "test missing", "float3"),
                ],
            ),
            "",
            "Missingness/imputation for CLEAN features:",
            "",
            md_table(
                result["clean_missingness"],
                [
                    ("feature", "feature", "str"),
                    ("train_missing_fraction", "train missing", "float3"),
                    ("test_missing_fraction", "test missing", "float3"),
                    ("train_median_imputation", "train median", "float3"),
                    ("flag_gt_20pct_missing", ">20% missing", "str"),
                ],
            ),
            "",
            "## Part 1 - Calibrated CLEAN Classifier",
            "",
            md_table(
                [
                    {
                        "model": "CLEAN GBT",
                        "whole_auc": clean_auc["test_auc"]["auc"],
                        "whole_ci_low": clean_auc["test_auc"]["ci_low"],
                        "whole_ci_high": clean_auc["test_auc"]["ci_high"],
                        "boundary_auc": clean_auc["boundary_auc"]["auc"],
                        "boundary_ci_low": clean_auc["boundary_auc"]["ci_low"],
                        "boundary_ci_high": clean_auc["boundary_auc"]["ci_high"],
                        "boundary_n": clean_auc["boundary_n"],
                    },
                    {
                        "model": "CLEAN logistic",
                        "whole_auc": result["logistic_auc"]["test_auc"]["auc"],
                        "whole_ci_low": result["logistic_auc"]["test_auc"]["ci_low"],
                        "whole_ci_high": result["logistic_auc"]["test_auc"]["ci_high"],
                        "boundary_auc": result["logistic_auc"]["boundary_auc"]["auc"],
                        "boundary_ci_low": result["logistic_auc"]["boundary_auc"]["ci_low"],
                        "boundary_ci_high": result["logistic_auc"]["boundary_auc"]["ci_high"],
                        "boundary_n": result["logistic_auc"]["boundary_n"],
                    },
                    {
                        "model": "RATIO-AWARE reference",
                        "whole_auc": ratio_auc["test_auc"]["auc"],
                        "whole_ci_low": ratio_auc["test_auc"]["ci_low"],
                        "whole_ci_high": ratio_auc["test_auc"]["ci_high"],
                        "boundary_auc": ratio_auc["boundary_auc"]["auc"],
                        "boundary_ci_low": ratio_auc["boundary_auc"]["ci_low"],
                        "boundary_ci_high": ratio_auc["boundary_auc"]["ci_high"],
                        "boundary_n": ratio_auc["boundary_n"],
                    },
                ],
                [
                    ("model", "model", "str"),
                    ("whole_auc", "whole AUC", "float3"),
                    ("whole_ci_low", "whole CI low", "float3"),
                    ("whole_ci_high", "whole CI high", "float3"),
                    ("boundary_auc", "boundary AUC", "float3"),
                    ("boundary_ci_low", "boundary CI low", "float3"),
                    ("boundary_ci_high", "boundary CI high", "float3"),
                    ("boundary_n", "boundary n", "int"),
                ],
            ),
            "",
            f"- Boundary band definition: {clean_auc['boundary_definition']}.",
            f"- Forced p>=0.5 accuracy on strict workflow-deferred TEST cases: {fmt(result['clean_deferred_forced_accuracy']['strict_forced_accuracy_at_0p5'], 3)} ({result['clean_deferred_forced_accuracy'].get('strict_correct_n', 0)}/{result['clean_deferred_forced_accuracy']['strict_deferred_n']}).",
            f"- Forced p>=0.5 accuracy on operational residual-deferred TEST cases: {fmt(result['clean_deferred_forced_accuracy']['operational_forced_accuracy_at_0p5'], 3)} ({result['clean_deferred_forced_accuracy'].get('operational_correct_n', 0)}/{result['clean_deferred_forced_accuracy']['operational_deferred_n']}).",
            "- RATIO-AWARE is diagnostic only; it includes the same ratio measurement being questioned at the boundary and is not used for the graded output.",
            "",
            "Permutation feature importance for CLEAN GBT:",
            "",
            md_table(
                result["permutation_importance"][:10],
                [
                    ("feature", "feature", "str"),
                    ("importance_mean_auc_drop", "mean AUC drop", "float3"),
                    ("importance_sd", "sd", "float3"),
                ],
            ),
            "",
            "## Part 2 - Natural-Prevalence Reweighting",
            "",
            f"- Prevalence path: {result['prevalence']['path']}.",
            f"- Target prevalence: {fmt(result['prevalence']['target_prevalence'], 3)} ({result['prevalence']['obstructed_n']}/{result['prevalence']['valid_ratio_n']} valid ratios).",
            f"- Constructed TRAIN prevalence used for weights: {fmt(result['constructed_train_prevalence'], 3)}.",
            "",
            md_table(
                [
                    {**result["calibration_summary"]["test_all_raw"], "scope": "TEST raw"},
                    {**result["calibration_summary"]["test_all_reweighted"], "scope": "TEST reweighted"},
                    {**result["calibration_summary"]["indeterminate_raw"], "scope": "indeterminate raw"},
                    {**result["calibration_summary"]["indeterminate_reweighted"], "scope": "indeterminate reweighted"},
                ],
                [
                    ("scope", "scope", "str"),
                    ("n", "n", "int"),
                    ("ece", "ECE", "float3"),
                    ("brier", "Brier", "float3"),
                ],
            ),
            "",
            "Prevalence sensitivity sweep for the TRAIN-frozen candidate boundaries before rejection:",
            "",
            md_table(
                result["prevalence_sensitivity"],
                [
                    ("target_prevalence", "target prev", "float3"),
                    ("auc_reweighted", "AUC", "float3"),
                    ("ppv_reweighted", "PPV", "float3"),
                    ("npv_reweighted", "NPV", "float3"),
                    ("ece_reweighted", "ECE", "float3"),
                    ("brier_reweighted", "Brier", "float3"),
                ],
            ),
            "",
            "## Part 3 - Graded Three-Zone Output Policy",
            "",
            f"- TRAIN-frozen boundaries: p_low={fmt(result['zone_boundaries']['p_low'], 3)}, p_high={fmt(result['zone_boundaries']['p_high'], 3)}; status={result['zone_boundaries']['boundary_status']}.",
            f"- Candidate zone rejected on TEST: {result['candidate_zone_rejected']}.",
            f"- Candidate rejection reasons: {'; '.join(result['candidate_zone_reject_reasons']) if result['candidate_zone_reject_reasons'] else 'none'}.",
            "",
            "TRAIN-frozen candidate zones before rejection:",
            "",
            md_table(
                [{"scope": "candidate zones", **result["candidate_zone_metrics"]}],
                [
                    ("scope", "scope", "str"),
                    ("confident_n", "confident n", "int"),
                    ("confident_coverage", "conf coverage", "float3"),
                    ("cohens_kappa", "kappa", "float3"),
                    ("ppv_reweighted", "PPV rw", "float3"),
                    ("npv_reweighted", "NPV rw", "float3"),
                    ("severe_confident_n", "severe conf", "int"),
                    ("severe_fn_n", "severe FN", "int"),
                    ("severe_fn_ci_high", "severe FN CP high", "float3"),
                ],
            ),
            "",
            "Retained graded output after enforcing the safety/floor guard:",
            "",
            md_table(
                zone["zone_counts"],
                [("zone", "zone", "str"), ("n", "n", "int"), ("fraction", "fraction", "float3")],
            ),
            "",
            md_table(
                [{"scope": "graded confident zones", **zone}],
                [
                    ("scope", "scope", "str"),
                    ("confident_n", "confident n", "int"),
                    ("confident_coverage", "conf coverage", "float3"),
                    ("accuracy", "accuracy", "float3"),
                    ("sensitivity", "sens", "float3"),
                    ("specificity", "spec", "float3"),
                    ("cohens_kappa", "kappa", "float3"),
                    ("ppv_reweighted", "PPV rw", "float3"),
                    ("npv_reweighted", "NPV rw", "float3"),
                    ("severe_confident_n", "severe conf", "int"),
                    ("severe_fn_n", "severe FN", "int"),
                    ("severe_fn_ci_high", "severe FN CP high", "float3"),
                ],
            ),
            "",
            f"- GRADED coverage is 100% because every TEST patient receives p_obs and a zone. Retained confident coverage is {fmt(zone['confident_coverage'], 3)}; the remaining {fmt(1.0 - zone['confident_coverage'], 3)} receive probability-only indeterminate output.",
            "",
            "## Part 4 - Confident Coverage vs Deterministic Baseline",
            "",
            md_table(
                [
                    {"scope": "strict deterministic final-call", **baseline["strict_workflow"]},
                    {"scope": "operational recovery baseline", **baseline["operational_recovery"]},
                    {"scope": "graded confident zones", "finalized_fraction": zone["confident_coverage"], "cohens_kappa": zone["cohens_kappa"], "severe_fn_n": zone["severe_fn_n"], "severe_fn_ci_high": zone["severe_fn_ci_high"]},
                ],
                [
                    ("scope", "scope", "str"),
                    ("finalized_fraction", "coverage", "float3"),
                    ("cohens_kappa", "kappa", "float3"),
                    ("severe_fn_n", "severe FN", "int"),
                    ("severe_fn_ci_high", "severe FN CP high", "float3"),
                ],
            ),
            "",
            "## Part 5 - Orthogonal Demographic Lift Test",
            "",
            md_table(
                result["demographic_availability_deferred_test"],
                [("feature", "feature", "str"), ("available_n", "available", "int"), ("deferred_test_n", "deferred", "int"), ("available_fraction", "fraction", "float3")],
            ),
            "",
            md_table(
                [
                    {
                        "model": "clean",
                        "whole_auc": result["clean_model_auc"]["test_auc"]["auc"],
                        "boundary_auc": result["clean_model_auc"]["boundary_auc"]["auc"],
                        "boundary_n": result["clean_model_auc"]["boundary_n"],
                    },
                    {
                        "model": "clean+demographics",
                        "whole_auc": result["demographic_model_auc"]["test_auc"]["auc"],
                        "boundary_auc": result["demographic_model_auc"]["boundary_auc"]["auc"],
                        "boundary_n": result["demographic_model_auc"]["boundary_n"],
                    },
                    {
                        "model": "demographics-only deferred",
                        "whole_auc": result["demographics_only_deferred_auc"],
                        "boundary_auc": np.nan,
                        "boundary_n": np.nan,
                    },
                ],
                [
                    ("model", "model", "str"),
                    ("whole_auc", "AUC", "float3"),
                    ("boundary_auc", "boundary AUC", "float3"),
                    ("boundary_n", "boundary n", "int"),
                ],
            ),
            "",
            f"- Clean boundary AUC: {fmt(result['clean_model_auc']['boundary_auc']['auc'], 3)}.",
            f"- Clean+demographics boundary AUC: {fmt(result['demographic_model_auc']['boundary_auc']['auc'], 3)}.",
            f"- Demographics-only AUC on operational deferred TEST cases: {fmt(result['demographics_only_deferred_auc'], 3)}.",
            f"- Demographic lift verdict: {result['demographic_lift_verdict']}.",
            "",
            "## Part 6 - Knife-Edge vs Recoverable Decomposition",
            "",
            md_table(
                result["knife_edge_decomposition"],
                [
                    ("bucket", "bucket", "str"),
                    ("n", "n", "int"),
                    ("fraction_deferred", "fraction deferred", "float3"),
                    ("graded_confident_recovered_n", "graded confident", "int"),
                    ("graded_confident_recovered_fraction", "graded fraction", "float3"),
                ],
            ),
            "",
            "Knife-edge cases are not counted as a model failure; the fixed 0.70 threshold cuts through a continuum under measurement noise.",
            "",
            f"VERDICT: {result['verdict']} - {result['verdict_reason']}",
            "",
        ]
    )
    (out / f"{OUTPUT_STEM}.md").write_text("\n".join(lines), encoding="utf-8")


def run(root: Path) -> dict[str, Any]:
    workflow, repro, recovered_test, _c1 = load_model_frame(root)
    prevalence = prevalence_from_full_cohort(root)
    train = workflow[workflow["split"].eq("train")].copy().reset_index(drop=True)
    test = workflow[workflow["split"].eq("test")].copy().reset_index(drop=True)
    test = test.merge(
        recovered_test[["SEQN", "post_recovery_finalized", "post_recovery_decision"]],
        on="SEQN",
        how="left",
    )

    audit, banned_present, quarantined, clean_features = feature_audit(workflow)
    leakage_guard_tripped = bool(set(clean_features) & (set(quarantined) | set(banned_present)))
    if leakage_guard_tripped:
        result = {
            "verdict": "LEAKAGE-GUARD-TRIPPED",
            "verdict_reason": "A quarantined or banned feature reached CLEAN features.",
            "leakage_guard_tripped": True,
            "banned_present": banned_present,
            "quarantined_features": quarantined,
            "clean_features": clean_features,
        }
        write_outputs(root, result, audit, [], pd.DataFrame())
        return result

    x_train, x_test, imputation = impute_frames(train, test, clean_features)
    y_train = truth_obstructed(train).astype(int).to_numpy()
    y_test = truth_obstructed(test).astype(int).to_numpy()
    constructed_train_prevalence = float(y_train.mean())
    target_prevalence = float(prevalence["target_prevalence"])
    train_weights = class_reweight(y_train, target_prevalence, constructed_train_prevalence)
    test_weights = class_reweight(y_test, target_prevalence, constructed_train_prevalence)

    clean_model = fit_model(make_gbt(), x_train, y_train)
    logistic_model = fit_model(make_logistic(), x_train, y_train)
    clean_p_train = probability(clean_model, x_train)
    clean_p_test = probability(clean_model, x_test)
    logistic_auc = model_auc_report("clean_logistic", logistic_model, x_test, test, test_weights, 10)
    clean_auc = model_auc_report("clean_gbt", clean_model, x_test, test, test_weights, 20)

    ratio_features = clean_features + [feature for feature in RATIO_AWARE_FEATURES if feature in workflow.columns]
    xr_train, xr_test, _ratio_imp = impute_frames(train, test, ratio_features)
    ratio_model = fit_model(make_gbt(), xr_train, y_train)
    ratio_auc = model_auc_report("ratio_aware_gbt", ratio_model, xr_test, test, test_weights, 30)

    boundaries = choose_boundaries(train, clean_p_train, train_weights)
    candidate_zones = apply_zones(test, clean_p_test, boundaries)
    candidate_zone = zone_metrics(candidate_zones, test_weights)
    reject_reasons = []
    if candidate_zone["severe_fn_n"] > 0:
        reject_reasons.append("candidate confident-normal zone has severe false negatives on TEST")
    if not np.isfinite(candidate_zone["ppv_reweighted"]) or candidate_zone["ppv_reweighted"] < PPV_NPV_FLOOR:
        reject_reasons.append("candidate confident-obstructed PPV is below the reweighted 0.90 floor on TEST")
    if not np.isfinite(candidate_zone["npv_reweighted"]) or candidate_zone["npv_reweighted"] < PPV_NPV_FLOOR:
        reject_reasons.append("candidate confident-normal NPV is below the reweighted 0.90 floor on TEST")
    candidate_rejected = bool(reject_reasons)
    if candidate_rejected:
        test_zones = all_indeterminate_zones(test, clean_p_test)
        test_zones["candidate_graded_zone"] = candidate_zones["graded_zone"].to_numpy()
        test_zones["candidate_graded_confident"] = candidate_zones["graded_confident"].to_numpy()
        test_zones["candidate_rejected_reason"] = "; ".join(reject_reasons)
    else:
        test_zones = candidate_zones.copy()
        test_zones["candidate_graded_zone"] = candidate_zones["graded_zone"].to_numpy()
        test_zones["candidate_graded_confident"] = candidate_zones["graded_confident"].to_numpy()
        test_zones["candidate_rejected_reason"] = ""
    zone = zone_metrics(test_zones, test_weights)
    clean_deferred = forced_deferred_accuracy(test_zones)
    permutation = permutation_rows(clean_model, x_test, y_test)

    calibration_rows: list[dict[str, Any]] = []
    whole_raw_rows, whole_raw = reliability_bins(y_test, clean_p_test, None, "test_all", "raw", target_prevalence)
    whole_rw_rows, whole_rw = reliability_bins(y_test, clean_p_test, test_weights, "test_all", "reweighted", target_prevalence)
    calibration_rows.extend(whole_raw_rows)
    calibration_rows.extend(whole_rw_rows)
    ind_mask = test_zones["graded_zone"].eq("INDETERMINATE_GRADED").to_numpy()
    ind_raw_rows, ind_raw = reliability_bins(y_test[ind_mask], clean_p_test[ind_mask], None, "indeterminate", "raw", target_prevalence)
    ind_rw_rows, ind_rw = reliability_bins(y_test[ind_mask], clean_p_test[ind_mask], test_weights[ind_mask], "indeterminate", "reweighted", target_prevalence)
    calibration_rows.extend(ind_raw_rows)
    calibration_rows.extend(ind_rw_rows)

    sensitivity = sensitivity_metrics(
        y_test,
        clean_p_test,
        float(boundaries["p_low"]),
        float(boundaries["p_high"]),
        constructed_train_prevalence,
        prevalence["sensitivity_prevalences"],
    )
    sensitivity_flat = []
    for item in sensitivity:
        calibration_rows.extend(item.pop("calibration_bins"))
        sensitivity_flat.append(item)

    # Demographic arm.
    demo_frame, demo_availability, demo_notes = add_demographics(root, workflow.merge(test[["SEQN", "post_recovery_finalized"]], on="SEQN", how="left"))
    demo_train = demo_frame[demo_frame["split"].eq("train")].copy().reset_index(drop=True)
    demo_test = demo_frame[demo_frame["split"].eq("test")].copy().reset_index(drop=True)
    demo_features = clean_features + [col for col in demo_frame.columns if col in DEMOGRAPHIC_BASE_FEATURES or col.startswith("race_")]
    xd_train, xd_test, _demo_imp = impute_frames(demo_train, demo_test, demo_features)
    demo_model = fit_model(make_gbt(), xd_train, y_train)
    demo_auc = model_auc_report("clean_plus_demographics", demo_model, xd_test, test, test_weights, 40)
    demo_p_train = probability(demo_model, xd_train)
    demo_boundaries = choose_boundaries(demo_train, demo_p_train, train_weights)
    demo_p_test = probability(demo_model, xd_test)
    demo_zones = apply_zones(test, demo_p_test, demo_boundaries)
    demo_zone = zone_metrics(demo_zones, test_weights)

    demo_only_features = [col for col in demo_frame.columns if col in DEMOGRAPHIC_BASE_FEATURES or col.startswith("race_")]
    xdo_train, xdo_test, _ = impute_frames(demo_train, demo_test, demo_only_features)
    demo_only_model = fit_model(make_gbt(), xdo_train, y_train)
    demo_only_p = probability(demo_only_model, xdo_test)
    deferred_mask = test["post_recovery_finalized"].eq(False).to_numpy()
    demo_deferred_auc = auc_safe(y_test[deferred_mask], demo_only_p[deferred_mask])
    clean_boundary_auc = clean_auc["boundary_auc"]["auc"]
    demo_boundary_auc = demo_auc["boundary_auc"]["auc"]
    if np.isfinite(clean_boundary_auc) and np.isfinite(demo_boundary_auc) and demo_boundary_auc >= clean_boundary_auc + 0.05 and demo_zone["severe_fn_n"] == 0:
        demo_verdict = "DEMOGRAPHICS-LIFT"
    else:
        demo_verdict = "DEMOGRAPHICS-NEUTRAL"

    strict_metrics = final_metrics(recovered_test, "workflow_decision", "workflow_final_diagnostic")
    operational_metrics = final_metrics(recovered_test, "post_recovery_decision", "post_recovery_finalized")
    participants = test_zones[
        [
            "SEQN",
            "p_obs",
            "graded_zone",
            "graded_confident",
            "candidate_graded_zone",
            "candidate_graded_confident",
            "candidate_rejected_reason",
            "truth_obstructed",
            "fev1_fvc_ratio",
            "truth_severity",
            "split",
            "workflow_final_diagnostic",
            "post_recovery_finalized",
            "post_recovery_decision",
        ]
    ].copy()

    result: dict[str, Any] = {
        "seed": RANDOM_SEED,
        "leakage_guard_tripped": False,
        "banned_present": banned_present,
        "quarantined_features": quarantined,
        "clean_features": clean_features,
        "feature_audit_top_corr": audit.sort_values("max_abs_corr", ascending=False).head(18).to_dict("records"),
        "clean_missingness": imputation["missingness"],
        "prevalence": prevalence,
        "constructed_train_prevalence": constructed_train_prevalence,
        "clean_model_auc": clean_auc,
        "logistic_auc": logistic_auc,
        "ratio_aware_auc": ratio_auc,
        "permutation_importance": permutation,
        "zone_boundaries": boundaries,
        "candidate_zone_metrics": candidate_zone,
        "candidate_zone_rejected": candidate_rejected,
        "candidate_zone_reject_reasons": reject_reasons,
        "graded_zone_metrics": zone,
        "calibration_summary": {
            "test_all_raw": whole_raw,
            "test_all_reweighted": whole_rw,
            "indeterminate_raw": ind_raw,
            "indeterminate_reweighted": ind_rw,
        },
        "prevalence_sensitivity": sensitivity_flat,
        "deterministic_baseline": {
            "strict_workflow": strict_metrics,
            "operational_recovery": operational_metrics,
        },
        "clean_deferred_forced_accuracy": clean_deferred,
        "demographic_availability_deferred_test": demo_availability,
        "demographic_notes": demo_notes,
        "demographic_model_auc": demo_auc,
        "demographic_zone_boundaries": demo_boundaries,
        "demographic_zone_metrics": demo_zone,
        "demographics_only_deferred_auc": demo_deferred_auc,
        "demographic_lift_verdict": demo_verdict,
        "knife_edge_decomposition": knife_edge_decomposition(test_zones),
        "protocol_decisions": [
            "TEST split is scored once; all model fitting and zone boundary selection use TRAIN only.",
            "The requested true 0.60-0.70 band is one-class-only and cannot support AUC; the reported boundary AUC uses true FEV1/FVC in [0.65, 0.75), matching the prior around-threshold check.",
            "CLEAN candidate features are limited to the task-specified shape/fit-quality/repeatability fields plus the expected ratio-proxy fields needed for audit/quarantine.",
            "fit_alpha was expected to survive from prior notes, but its TRAIN Spearman correlation to observed ratio is above 0.60, so the leakage audit excludes it from CLEAN.",
            "gli_lln_ratio measured just below 0.60 but is excluded because the task explicitly identifies it as an expected ratio/threshold proxy that must not enter CLEAN.",
            "Median imputation values are learned on TRAIN only and applied to TEST.",
            "The empirical target prevalence is computed from results/merged_summary.parquet, then TRAIN constructed prevalence is used to form class reweighting factors.",
            "Zone boundaries maximize TRAIN confident coverage subject to severe FN=0 among confident normal and reweighted PPV/NPV >=0.90.",
            f"Reweighted ECE > {CALIBRATION_ECE_WARN:.2f} is treated as uncalibrated for the verdict.",
            "Forced p>=0.5 deferred-case accuracy is reported only as a diagnostic; it is not retained as recovery because the TRAIN-frozen zones fail TEST safety/calibration floors.",
            "The RATIO-AWARE model is diagnostic only and is never used for graded output.",
        ],
    }
    result["verdict"], result["verdict_reason"] = verdict(result)
    write_outputs(root, result, audit, calibration_rows, participants)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    zone = result["graded_zone_metrics"]
    print(f"Graded probability gate: {result['verdict']} - {result['verdict_reason']}")
    print(f"CLEAN features: {', '.join(result['clean_features'])}")
    print(f"CLEAN TEST AUC: {result['clean_model_auc']['test_auc']['auc']:.3f}")
    print(f"CLEAN boundary AUC: {result['clean_model_auc']['boundary_auc']['auc']:.3f}")
    print(f"Confident coverage: {zone['confident_coverage']:.3f}")
    print(f"Severe FN: {zone['severe_fn_n']}/{zone['severe_confident_n']}")
    print(f"Reweighted PPV/NPV: {zone['ppv_reweighted']:.3f}/{zone['npv_reweighted']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
