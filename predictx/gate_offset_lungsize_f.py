from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr

from .gate3_diagnosis import np_ombe_volume
from .gate23_f import find_or_unzip_raw, read_raw
from .gate_clinical_classification_f import (
    confusion_binary,
    extract_curve_for_fit,
    load_column_map,
    load_gli,
    load_truth,
    severity_category,
)
from .gate_estimand_hybrid_f import finite_pair, fit_multiplicative
from .gate_model_vs_observed_f import fixed_metrics, signed_summary
from .gate_policy_calibrated_flag_f import stratified_half_split
from .gate_truncation_recovery_f import fit_ombe_truncated, model_grid_with_stop

OUTPUT_STEM = "gate_offset_lungsize_F"
RANDOM_SEED = 20260708
METHODS = ("P1", "P2", "P3")
ESTIMATORS_BASE = ("model", "K1", "K2", "K3", "model_mult", "H1", "observed")
STRATA = ("severe", "moderate", "mild_0.60_0.70", "normal")
BOOTSTRAP_N = 1000


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if np.isfinite(out) else None
    return value


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str, str]]) -> str:
    out = ["| " + " | ".join(label for _, label, _ in columns) + " |"]
    out.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        cells: list[str] = []
        for key, _, fmt in columns:
            value = row.get(key)
            if fmt == "int":
                try:
                    cells.append(str(int(value)) if pd.notna(value) else "NA")
                except Exception:
                    cells.append("NA")
            elif fmt.startswith("float"):
                digits = int(fmt.replace("float", "") or "3")
                try:
                    v = float(value)
                    cells.append(f"{v:.{digits}f}" if np.isfinite(v) else "NA")
                except Exception:
                    cells.append("NA")
            elif fmt == "bool":
                cells.append(str(bool(value)) if pd.notna(value) else "NA")
            else:
                cells.append("" if value is None else str(value))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def q(values: Any) -> list[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return [np.nan, np.nan]
    return [float(x) for x in np.percentile(arr, [25, 75])]


def finite_linreg(y: Any, x: Any) -> dict[str, Any]:
    x_ok, y_ok = finite_pair(x, y)
    if len(x_ok) < 3:
        return {"n": int(len(x_ok)), "slope": np.nan, "intercept": np.nan, "r2": np.nan, "p_value": np.nan}
    fit = linregress(x_ok, y_ok)
    return {
        "n": int(len(x_ok)),
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "r2": float(fit.rvalue**2),
        "p_value": float(fit.pvalue),
    }


def safe_corr(x: Any, y: Any) -> dict[str, Any]:
    x_ok, y_ok = finite_pair(x, y)
    if len(x_ok) < 3:
        return {"n": int(len(x_ok)), "pearson_r": np.nan, "p_value": np.nan, "slope": np.nan, "intercept": np.nan}
    r, p = pearsonr(x_ok, y_ok)
    fit = linregress(x_ok, y_ok)
    return {"n": int(len(x_ok)), "pearson_r": float(r), "p_value": float(p), "slope": float(fit.slope), "intercept": float(fit.intercept)}


def load_manoeuvres_fast(root: Path, truth: pd.DataFrame) -> pd.DataFrame:
    path = root / "results" / "gate_reproducibility_volume_F_rows.csv"
    if not path.exists():
        raise RuntimeError(f"Missing volume reproducibility rows: {path}")
    rows = pd.read_csv(path, low_memory=False)
    rows = rows[rows["row_type"].eq("maneuver_fit")].copy()
    rows["volume_fit_success"] = rows["volume_fit_success"].map(
        {True: True, False: False, "True": True, "False": False, "true": True, "false": False}
    ).fillna(False)
    needed = ["model_fev1_l", "model_fvc_l", "observed_fev1_l", "observed_fvc_l", "fit_tau_s", "duration_s"]
    rows = rows[
        rows["volume_fit_success"]
        & rows["fit_error"].fillna("").eq("")
        & rows[needed].apply(np.isfinite).all(axis=1)
        & (rows["model_fvc_l"] > 0)
        & (rows["observed_fvc_l"] > 0)
    ].copy()
    rows = rows.merge(
        truth[["SEQN", "fev1_mL", "fvc_mL", "fev1_fvc_ratio", "age_years", "height_cm"]],
        on="SEQN",
        how="left",
        suffixes=("", "_truth"),
    )
    if "fev1_fvc_ratio_truth" in rows.columns:
        rows["fev1_fvc_ratio"] = rows["fev1_fvc_ratio"].fillna(rows["fev1_fvc_ratio_truth"])
    rows = rows[rows["fev1_fvc_ratio"].notna()].copy()
    rows["truth_severity"] = rows["fev1_fvc_ratio"].map(severity_category)
    rows["model_fev1_l_est"] = rows["model_fev1_l"]
    rows["model_fvc_l_est"] = rows["model_fvc_l"]
    rows["model_ratio_est"] = rows["model_ratio"]
    rows["observed_fev1_l_est"] = rows["observed_fev1_l"]
    rows["observed_fvc_l_est"] = rows["observed_fvc_l"]
    rows["observed_ratio_est"] = rows["observed_ratio"]
    rows["fvc_residual_l"] = rows["observed_fvc_l_est"] - rows["model_fvc_l_est"]
    rows["fev1_residual_l"] = rows["observed_fev1_l_est"] - rows["model_fev1_l_est"]
    return rows


def base_pool(maneuvers: pd.DataFrame, estimators: tuple[str, ...]) -> pd.DataFrame:
    needed: list[str] = []
    for estimator in estimators:
        needed.extend([f"{estimator}_fev1_l_est", f"{estimator}_fvc_l_est", f"{estimator}_ratio_est"])
    m = maneuvers[maneuvers[needed].apply(np.isfinite).all(axis=1)].copy()
    counts = m.groupby("SEQN").size()
    keep = set(counts[counts >= 2].index.astype(float))
    m = m[m["SEQN"].astype(float).isin(keep)].copy()
    rows: list[dict[str, Any]] = []
    for seq, group in m.groupby("SEQN"):
        group = group.sort_values(["SPAMANU", "raw_index"]).copy()
        best = group.sort_values(["observed_fev1_l_est", "SPAMANU"], ascending=[False, True]).iloc[0]
        row: dict[str, Any] = {
            "SEQN": float(seq),
            "truth_severity": group["truth_severity"].iloc[0],
            "fev1_fvc_ratio": float(group["fev1_fvc_ratio"].iloc[0]),
            "n_maneuvers": int(len(group)),
            "fit_tau_s_median": float(group["fit_tau_s"].median()),
            "fet_s_max": float(group["duration_s"].max()),
            "age_years": float(group["age_years"].iloc[0]) if np.isfinite(float(group["age_years"].iloc[0])) else np.nan,
            "height_cm": float(group["height_cm"].iloc[0]) if np.isfinite(float(group["height_cm"].iloc[0])) else np.nan,
        }
        for estimator in estimators:
            calls = group[f"{estimator}_ratio_est"] < 0.70
            row[f"{estimator}_spread"] = float(group[f"{estimator}_ratio_est"].max() - group[f"{estimator}_ratio_est"].min())
            row[f"{estimator}_disagree"] = bool(calls.nunique() > 1)
            row[f"P1_{estimator}_fev1_l"] = float(best[f"{estimator}_fev1_l_est"])
            row[f"P1_{estimator}_fvc_l"] = float(best[f"{estimator}_fvc_l_est"])
            row[f"P1_{estimator}_ratio"] = float(best[f"{estimator}_ratio_est"])
            row[f"P2_{estimator}_ratio"] = float(group[f"{estimator}_ratio_est"].median())
            row[f"P3_{estimator}_fev1_l"] = float(group[f"{estimator}_fev1_l_est"].max())
            row[f"P3_{estimator}_fvc_l"] = float(group[f"{estimator}_fvc_l_est"].max())
            row[f"P3_{estimator}_ratio"] = row[f"P3_{estimator}_fev1_l"] / row[f"P3_{estimator}_fvc_l"]
        rows.append(row)
    return pd.DataFrame(rows)


def attach_split_and_gli(participants: pd.DataFrame, gli: pd.DataFrame) -> pd.DataFrame:
    split = stratified_half_split(participants, "truth_severity")
    if not gli.empty:
        split = split.merge(gli, on="SEQN", how="left")
    else:
        split["gli_lln_ratio"] = np.nan
        split["gli_ethnicity_mapped"] = np.nan
        split["gli_mapping_fallback"] = np.nan
    return split


def fit_offsets(train_m: pd.DataFrame) -> dict[str, dict[str, Any]]:
    residual = train_m["observed_fvc_l_est"].to_numpy(float) - train_m["model_fvc_l_est"].to_numpy(float)
    fet = train_m["duration_s"].to_numpy(float)
    tau_s = train_m["fit_tau_s"].to_numpy(float)
    ok = np.isfinite(residual)
    k1_c = float(np.nanmedian(residual[ok])) if ok.any() else np.nan
    k2 = finite_linreg(residual, fet)
    k3 = finite_linreg(residual, tau_s)
    return {
        "K1": {"variant": "K1", "form": "FVC_model + c", "c": k1_c, "n": int(ok.sum())},
        "K2": {"variant": "K2", "form": "FVC_model + c0 + c1*FET", "c0": k2["intercept"], "c1": k2["slope"], **k2},
        "K3": {"variant": "K3", "form": "FVC_model + c0 + c1*tau_s", "c0": k3["intercept"], "c1": k3["slope"], **k3},
    }


def add_corrections(m: pd.DataFrame, offsets: dict[str, dict[str, Any]], mult: dict[str, float]) -> pd.DataFrame:
    out = m.copy()
    out["K1_fvc_l_est"] = out["model_fvc_l_est"] + offsets["K1"]["c"]
    out["K2_fvc_l_est"] = out["model_fvc_l_est"] + offsets["K2"]["c0"] + offsets["K2"]["c1"] * out["duration_s"]
    out["K3_fvc_l_est"] = out["model_fvc_l_est"] + offsets["K3"]["c0"] + offsets["K3"]["c1"] * out["fit_tau_s"]
    for k in ("K1", "K2", "K3"):
        out[f"{k}_fev1_l_est"] = out["model_fev1_l_est"]
        out.loc[out[f"{k}_fvc_l_est"] <= 0, f"{k}_fvc_l_est"] = np.nan
        out[f"{k}_ratio_est"] = out[f"{k}_fev1_l_est"] / out[f"{k}_fvc_l_est"]
    out["model_mult_fev1_l_est"] = out["model_fev1_l_est"]
    out["model_mult_fvc_l_est"] = mult["a"] * np.power(out["model_fvc_l_est"], mult["b"])
    out["model_mult_ratio_est"] = out["model_mult_fev1_l_est"] / out["model_mult_fvc_l_est"]
    out["H1_fev1_l_est"] = out["model_fev1_l_est"]
    out["H1_fvc_l_est"] = out["observed_fvc_l_est"]
    out["H1_ratio_est"] = out["H1_fev1_l_est"] / out["H1_fvc_l_est"]
    return out


def part1_rows(train_m: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train = train_m.copy()
    train["fvc_model_minus_observed_ml"] = 1000.0 * (train["model_fvc_l_est"] - train["observed_fvc_l_est"])
    train["fev1_model_minus_observed_ml"] = 1000.0 * (train["model_fev1_l_est"] - train["observed_fev1_l_est"])
    train["abs_fvc_model_minus_observed_ml"] = np.abs(train["fvc_model_minus_observed_ml"])
    train["abs_fev1_model_minus_observed_ml"] = np.abs(train["fev1_model_minus_observed_ml"])
    for metric in ("fvc", "fev1"):
        for signedness, col in [
            ("signed", f"{metric}_model_minus_observed_ml"),
            ("absolute_magnitude", f"abs_{metric}_model_minus_observed_ml"),
        ]:
            rows.append({"row_type": "part1_offset_summary", "metric": metric, "signedness": signedness, "truth_severity": "all", **signed_summary(train[col])})
            for stratum in STRATA:
                part = train[train["truth_severity"].eq(stratum)]
                rows.append({"row_type": "part1_offset_summary", "metric": metric, "signedness": signedness, "truth_severity": stratum, **signed_summary(part[col])})
    predictors = [
        ("observed_fvc_mL", train["observed_fvc_l_est"] * 1000.0),
        ("FET_s", train["duration_s"]),
        ("tau_s", train["fit_tau_s"]),
        ("age_years", train["age_years"]),
    ]
    for metric in ("fvc", "fev1"):
        for signedness, col in [
            ("signed", f"{metric}_model_minus_observed_ml"),
            ("absolute_magnitude", f"abs_{metric}_model_minus_observed_ml"),
        ]:
            for predictor, x in predictors:
                rows.append(
                    {
                        "row_type": "part1_offset_regression",
                        "metric": metric,
                        "signedness": signedness,
                        "predictor": predictor,
                        **finite_linreg(train[col], x),
                    }
                )
    return rows


def residual_rows(m: pd.DataFrame, split_by_seqn: pd.DataFrame, estimators: tuple[str, ...]) -> list[dict[str, Any]]:
    if "split" not in m.columns:
        m = m.merge(split_by_seqn[["SEQN", "split"]], on="SEQN", how="left")
    else:
        m = m.copy()
    rows: list[dict[str, Any]] = []
    for estimator in estimators:
        if estimator == "observed":
            continue
        work = m.copy()
        work["fvc_resid_ml"] = 1000.0 * (work[f"{estimator}_fvc_l_est"] - work["observed_fvc_l_est"])
        work["ratio_resid"] = work[f"{estimator}_ratio_est"] - work["observed_ratio_est"]
        for split_name, split_part in work.groupby("split"):
            rows.append({"row_type": "part2_fvc_residual", "estimator": estimator, "split": split_name, "truth_severity": "all", **signed_summary(split_part["fvc_resid_ml"])})
            rows.append({"row_type": "part2_fvc_abs_residual", "estimator": estimator, "split": split_name, "truth_severity": "all", **signed_summary(np.abs(split_part["fvc_resid_ml"]))})
            for stratum in STRATA:
                part = split_part[split_part["truth_severity"].eq(stratum)]
                rows.append({"row_type": "part2_fvc_residual", "estimator": estimator, "split": split_name, "truth_severity": stratum, **signed_summary(part["fvc_resid_ml"])})
                rows.append({"row_type": "part2_ratio_residual", "estimator": estimator, "split": split_name, "truth_severity": stratum, **signed_summary(part["ratio_resid"])})
    return rows


def classification_rows(split: pd.DataFrame, estimators: tuple[str, ...]) -> list[dict[str, Any]]:
    test = split[split["split"].eq("test")].copy()
    rows: list[dict[str, Any]] = []
    for estimator in estimators:
        for method in METHODS:
            ratio_col = f"{method}_{estimator}_ratio"
            for definition, threshold_col in [("fixed_0.70", None), ("GLI_LLN", "gli_lln_ratio")]:
                metrics = fixed_metrics(test, ratio_col, threshold_col=threshold_col)
                rows.append({"row_type": "part2_classification", "endpoint": "NHANES_target", "estimator": estimator, "method": method, "definition": definition, **metrics})
    return rows


def flip_rows(split: pd.DataFrame, baseline: str, corrected: str, definition: str = "fixed_0.70") -> dict[str, Any]:
    test = split[split["split"].eq("test")].copy()
    threshold = 0.70 if definition == "fixed_0.70" else test["gli_lln_ratio"]
    valid = test[f"P3_{baseline}_ratio"].notna() & test[f"P3_{corrected}_ratio"].notna() & test["fev1_fvc_ratio"].notna()
    if definition != "fixed_0.70":
        valid &= test["gli_lln_ratio"].notna()
    work = test[valid].copy()
    base_call = work[f"P3_{baseline}_ratio"] < threshold.loc[work.index] if hasattr(threshold, "loc") else work[f"P3_{baseline}_ratio"] < threshold
    corr_call = work[f"P3_{corrected}_ratio"] < threshold.loc[work.index] if hasattr(threshold, "loc") else work[f"P3_{corrected}_ratio"] < threshold
    truth = work["fev1_fvc_ratio"] < (threshold.loc[work.index] if hasattr(threshold, "loc") else threshold)
    base_m = confusion_binary(truth, base_call)
    corr_m = confusion_binary(truth, corr_call)
    return {
        "row_type": "part2_classification_flip",
        "baseline": baseline,
        "corrected": corrected,
        "definition": definition,
        "n": int(len(work)),
        "normal_to_obstructed": int((~base_call & corr_call).sum()),
        "obstructed_to_normal": int((base_call & ~corr_call).sum()),
        "sensitivity_delta": float(corr_m["sensitivity"] - base_m["sensitivity"]),
        "specificity_delta": float(corr_m["specificity"] - base_m["specificity"]),
        "kappa_delta": float(corr_m["cohens_kappa"] - base_m["cohens_kappa"]),
    }


def heldout_split_rows(m: pd.DataFrame, estimators: tuple[str, ...]) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 77)
    rows: list[dict[str, Any]] = []
    for seq, group in m.groupby("SEQN"):
        if len(group) < 2:
            continue
        group = group.sort_values(["SPAMANU", "raw_index"]).copy()
        best_fev1 = group["observed_fev1_l_est"].max()
        best_fvc = group["observed_fvc_l_est"].max()
        nonbest = group[(group["observed_fev1_l_est"] < best_fev1) & (group["observed_fvc_l_est"] < best_fvc)]
        pool = nonbest if len(nonbest) else group
        held_idx = int(rng.choice(pool.index.to_numpy()))
        held = group.loc[held_idx]
        rem = group.drop(index=held_idx)
        if len(rem) == 0:
            continue
        out: dict[str, Any] = {
            "SEQN": float(seq),
            "fev1_fvc_ratio": float(held["observed_ratio_est"]),
            "truth_severity": severity_category(float(held["observed_ratio_est"])),
            "heldout_raw_index": int(held["raw_index"]),
            "heldout_was_nonbest": bool(held_idx in set(nonbest.index)),
        }
        for estimator in estimators:
            out[f"remaining_P3_{estimator}_ratio"] = float(rem[f"{estimator}_fev1_l_est"].max() / rem[f"{estimator}_fvc_l_est"].max())
            out[f"remaining_P2_{estimator}_ratio"] = float(rem[f"{estimator}_ratio_est"].median())
        rows.append(out)
    heldout = pd.DataFrame(rows)
    return stratified_half_split(heldout, "truth_severity")


def heldout_metrics_rows(heldout_split: pd.DataFrame, estimators: tuple[str, ...]) -> list[dict[str, Any]]:
    test = heldout_split[heldout_split["split"].eq("test")].copy()
    rows: list[dict[str, Any]] = []
    for estimator in estimators:
        for method in ("P2", "P3"):
            ratio_col = f"remaining_{method}_{estimator}_ratio"
            metrics = fixed_metrics(test, ratio_col, threshold_col=None)
            rows.append({"row_type": "part3_heldout_classification", "endpoint": "heldout_blow", "estimator": estimator, "method": method, "definition": "fixed_0.70", **metrics})
    rows.append(
        {
            "row_type": "part3_heldout_metadata",
            "n": int(len(heldout_split)),
            "test_n": int(len(test)),
            "nonbest_heldout_fraction": float(heldout_split["heldout_was_nonbest"].mean()) if len(heldout_split) else np.nan,
            "selection_rule": "Within each participant, choose one random non-best manoeuvre when available; best means max observed FEV1 or max observed FVC. Predict held-out obstruction from remaining manoeuvres. Seed=20260785.",
        }
    )
    return rows


def bootstrap_kappa_ci(df: pd.DataFrame, ratio_col: str, n_boot: int = BOOTSTRAP_N) -> dict[str, Any]:
    work = df[df[ratio_col].notna() & df["fev1_fvc_ratio"].notna()].copy()
    if len(work) == 0:
        return {"n": 0, "kappa": np.nan, "ci_low": np.nan, "ci_high": np.nan, "bootstrap_n": n_boot}
    truth = (work["fev1_fvc_ratio"].to_numpy(float) < 0.70)
    pred = (work[ratio_col].to_numpy(float) < 0.70)
    point = confusion_binary(pd.Series(truth), pd.Series(pred))["cohens_kappa"]
    rng = np.random.default_rng(RANDOM_SEED + 333)
    vals: list[float] = []
    n = len(work)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(float(confusion_binary(pd.Series(truth[idx]), pd.Series(pred[idx]))["cohens_kappa"]))
    return {
        "n": int(n),
        "kappa": float(point),
        "ci_low": float(np.nanpercentile(vals, 2.5)),
        "ci_high": float(np.nanpercentile(vals, 97.5)),
        "bootstrap_n": int(n_boot),
    }


def gap_rows(nhanes_rows: list[dict[str, Any]], heldout_rows: list[dict[str, Any]], estimators: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    nhanes = pd.DataFrame(nhanes_rows)
    held = pd.DataFrame(heldout_rows)
    obs_n = nhanes[
        nhanes["row_type"].eq("part2_classification")
        & nhanes["endpoint"].eq("NHANES_target")
        & nhanes["definition"].eq("fixed_0.70")
        & nhanes["method"].eq("P3")
        & nhanes["estimator"].eq("observed")
    ]["cohens_kappa"].iloc[0]
    obs_h = held[
        held["row_type"].eq("part3_heldout_classification")
        & held["method"].eq("P3")
        & held["estimator"].eq("observed")
    ]["cohens_kappa"].iloc[0]
    for estimator in estimators:
        n_part = nhanes[
            nhanes["row_type"].eq("part2_classification")
            & nhanes["endpoint"].eq("NHANES_target")
            & nhanes["definition"].eq("fixed_0.70")
            & nhanes["method"].eq("P3")
            & nhanes["estimator"].eq(estimator)
        ]
        h_part = held[
            held["row_type"].eq("part3_heldout_classification")
            & held["method"].eq("P3")
            & held["estimator"].eq(estimator)
        ]
        rows.append(
            {
                "row_type": "part3_endpoint_gap",
                "estimator": estimator,
                "nhanes_p3_kappa": float(n_part["cohens_kappa"].iloc[0]) if len(n_part) else np.nan,
                "nhanes_gap_to_observed": float(obs_n - n_part["cohens_kappa"].iloc[0]) if len(n_part) else np.nan,
                "heldout_p3_kappa": float(h_part["cohens_kappa"].iloc[0]) if len(h_part) else np.nan,
                "heldout_gap_to_observed": float(obs_h - h_part["cohens_kappa"].iloc[0]) if len(h_part) else np.nan,
            }
        )
    return rows


def gli_lungsize_rows(split: pd.DataFrame, best_k: str) -> list[dict[str, Any]]:
    test = split[split["split"].eq("test") & split["gli_lln_ratio"].notna()].copy()
    test = test[test["P3_observed_fvc_l"].notna()].copy()
    rows: list[dict[str, Any]] = []
    if len(test) == 0:
        return [{"row_type": "part4_gli_lungsize", "error": "no valid GLI rows"}]
    test["fvc_quartile"] = pd.qcut(test["P3_observed_fvc_l"], q=4, labels=["Q1_smallest", "Q2", "Q3", "Q4_largest"], duplicates="drop")
    for quartile, part in test.groupby("fvc_quartile", observed=False):
        ratio_err = part["P3_model_ratio"] - part["fev1_fvc_ratio"]
        rows.append(
            {
                "row_type": "part4_model_ratio_error_by_fvc_quartile",
                "fvc_quartile": str(quartile),
                "fvc_l_low": float(part["P3_observed_fvc_l"].min()),
                "fvc_l_high": float(part["P3_observed_fvc_l"].max()),
                **signed_summary(ratio_err),
            }
        )
        rows.append(
            {
                "row_type": "part4_gli_lln_by_fvc_quartile",
                "fvc_quartile": str(quartile),
                **signed_summary(part["gli_lln_ratio"]),
            }
        )
        for estimator in ("model", "H1", "observed", best_k):
            truth = part["fev1_fvc_ratio"] < part["gli_lln_ratio"]
            pred = part[f"P3_{estimator}_ratio"] < part["gli_lln_ratio"]
            metrics = confusion_binary(truth, pred)
            fn_rate = metrics["fn"] / (metrics["tp"] + metrics["fn"]) if (metrics["tp"] + metrics["fn"]) else np.nan
            borderline = part[(part["fev1_fvc_ratio"] >= 0.60) & (part["fev1_fvc_ratio"] < 0.70)].copy()
            borderline_truth = borderline["fev1_fvc_ratio"] < borderline["gli_lln_ratio"]
            borderline_pred = borderline[f"P3_{estimator}_ratio"] < borderline["gli_lln_ratio"]
            rows.append(
                {
                    "row_type": "part4_gli_fn_by_fvc_quartile",
                    "fvc_quartile": str(quartile),
                    "estimator": estimator,
                    **metrics,
                    "false_negative_rate_obstructed": fn_rate,
                    "borderline_n": int(len(borderline)),
                    "borderline_gli_obstructed_n": int(borderline_truth.sum()),
                    "borderline_false_negative_rate": float((borderline_truth & ~borderline_pred).sum() / borderline_truth.sum()) if int(borderline_truth.sum()) else np.nan,
                }
            )
    for predictor, col in [("observed_FVC_l", test["P3_observed_fvc_l"]), ("age_years", test["age_years"]), ("height_cm", test["height_cm"])]:
        rows.append({"row_type": "part4_gli_lln_correlation", "predictor": predictor, **safe_corr(col, test["gli_lln_ratio"])})
    for estimator in ("model", best_k):
        for method in ("P3",):
            metrics = fixed_metrics(test, f"{method}_{estimator}_ratio", threshold_col="gli_lln_ratio")
            rows.append({"row_type": "part4_gli_after_correction", "estimator": estimator, "method": method, **metrics})
    return rows


def truncation_rows(root: Path, raw: pd.DataFrame, split_by_seqn: pd.DataFrame, best_k: str, offsets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    existing_path = root / "results" / "gate_truncation_recovery_F_rows.csv"
    if not existing_path.exists():
        return [{"row_type": "part5_truncation", "error": f"missing {existing_path}"}]
    existing = pd.read_csv(existing_path, low_memory=False)
    base = existing[existing["T_label"].astype(str).isin(["2", "3", "4", "5"])].copy()
    base = base.merge(split_by_seqn[["SEQN", "split"]], on="SEQN", how="left")
    rows: list[dict[str, Any]] = []
    for estimator in ("E0_DIRECT", "E1_OMBE"):
        part = base[base["estimator"].eq(estimator)].copy()
        for t_label, group in part.groupby("T_label"):
            valid = group[group["fit_success"].astype(bool) & group["est_ratio"].notna()].copy()
            metrics = confusion_binary(valid["truth_obstructed_fixed"], valid["est_obstructed_fixed"])
            rows.append(
                {
                    "row_type": "part5_truncation_classification",
                    "estimator": estimator,
                    "T_label": str(t_label),
                    **metrics,
                    "obstructed_false_negative_rate": metrics["fn"] / (metrics["tp"] + metrics["fn"]) if (metrics["tp"] + metrics["fn"]) else np.nan,
                    "source": "existing_gate_truncation_recovery_F_rows",
                }
            )
    e1 = base[base["estimator"].eq("E1_OMBE") & base["fit_success"].astype(bool)].copy()
    for t_label, group in e1.groupby("T_label"):
        train = group[group["split"].eq("train")].copy()
        train["offset_l"] = train["true_fvc_l"] - train["est_fvc_l"]
        c_refit = float(np.nanmedian(train["offset_l"])) if len(train) else np.nan
        rows.append(
            {
                "row_type": "part5_truncation_offset_refit",
                "T_label": str(t_label),
                "train_n": int(len(train)),
                "refit_constant_offset_l": c_refit,
                "complete_K1_offset_l": offsets["K1"]["c"],
                "drift_from_complete_l": float(c_refit - offsets["K1"]["c"]) if np.isfinite(c_refit) else np.nan,
            }
        )

    sample = base[["SEQN", "raw_index", "SPAMANU", "truth_severity", "T_label", "true_ratio", "truth_obstructed_fixed"]].drop_duplicates(["raw_index", "T_label"])
    raw_by_index = {int(idx): row for idx, row in raw.iterrows()}
    detail: list[dict[str, Any]] = []
    total = len(sample)
    for i, row in enumerate(sample.itertuples(index=False), start=1):
        t_label = str(row.T_label)
        stop_s = float(t_label)
        try:
            curve = extract_curve_for_fit(raw_by_index[int(row.raw_index)])
            t = np.asarray(curve["vol_t"], dtype=float)
            y = np.asarray(curve["vol_l"], dtype=float)
            fit = fit_ombe_truncated(t, y, stop_s)
            grid, _fit_idx = model_grid_with_stop(t, stop_s)
            pred = np_ombe_volume(np.asarray(fit["params"], dtype=float), grid)
            model_fev1 = float(np.interp(1.0, grid, pred))
            model_fvc = float(pred[-1])
            tau_s = float(np.asarray(fit["params"], dtype=float)[1])
            success = bool(fit["success"] and np.isfinite(model_fev1) and np.isfinite(model_fvc) and model_fvc > 0)
            failure = "" if success else str(fit.get("failure", "nonfinite H ratio"))
        except Exception as exc:
            model_fev1 = np.nan
            model_fvc = np.nan
            tau_s = np.nan
            success = False
            failure = str(exc)
        h1_ratio = model_fev1 / model_fvc if success and model_fvc > 0 else np.nan
        if best_k == "K1":
            h2_fvc = model_fvc + offsets["K1"]["c"]
            h2_note = "complete K1 constant offset applied to truncated model FVC"
        elif best_k == "K2":
            h2_fvc = model_fvc + offsets["K2"]["c0"] + offsets["K2"]["c1"] * stop_s
            h2_note = "complete K2 FET coefficient applied with available truncation time T"
        else:
            h2_fvc = model_fvc + offsets["K3"]["c0"] + offsets["K3"]["c1"] * tau_s
            h2_note = "complete K3 tau_s coefficient applied with truncated-fit tau_s"
        h2_ratio = model_fev1 / h2_fvc if success and np.isfinite(h2_fvc) and h2_fvc > 0 else np.nan
        for estimator, ratio in [("H1", h1_ratio), ("H2", h2_ratio)]:
            detail.append(
                {
                    "row_type": "part5_truncation_detail",
                    "SEQN": float(row.SEQN),
                    "raw_index": int(row.raw_index),
                    "SPAMANU": float(row.SPAMANU),
                    "truth_severity": row.truth_severity,
                    "T_label": t_label,
                    "estimator": estimator,
                    "fit_success": success and np.isfinite(ratio),
                    "fit_failure": failure,
                    "true_ratio": float(row.true_ratio),
                    "truth_obstructed_fixed": bool(row.truth_obstructed_fixed),
                    "est_ratio": float(ratio) if np.isfinite(ratio) else np.nan,
                    "est_obstructed_fixed": bool(ratio < 0.70) if np.isfinite(ratio) else False,
                    "model_fev1_l": float(model_fev1) if np.isfinite(model_fev1) else np.nan,
                    "model_fvc_l": float(model_fvc) if np.isfinite(model_fvc) else np.nan,
                    "h2_fvc_l": float(h2_fvc) if np.isfinite(h2_fvc) else np.nan,
                    "best_k_for_h2": best_k,
                    "h2_application_note": h2_note,
                }
            )
        if i % 100 == 0 or i == total:
            print(f"Offset/H2 truncation refit progress: {i}/{total}")
    d = pd.DataFrame(detail)
    for estimator in ("H1", "H2"):
        part = d[d["estimator"].eq(estimator)].copy()
        for t_label, group in part.groupby("T_label"):
            valid = group[group["fit_success"].astype(bool) & group["est_ratio"].notna()].copy()
            metrics = confusion_binary(valid["truth_obstructed_fixed"], valid["est_obstructed_fixed"])
            rows.append(
                {
                    "row_type": "part5_truncation_classification",
                    "estimator": estimator,
                    "T_label": str(t_label),
                    **metrics,
                    "obstructed_false_negative_rate": metrics["fn"] / (metrics["tp"] + metrics["fn"]) if (metrics["tp"] + metrics["fn"]) else np.nan,
                    "source": "deterministic_refit_in_gate_offset_lungsize_F",
                }
            )
    return rows + detail


def choose_best_k(class_rows: list[dict[str, Any]]) -> str:
    df = pd.DataFrame(class_rows)
    part = df[
        df["row_type"].eq("part2_classification")
        & df["endpoint"].eq("NHANES_target")
        & df["definition"].eq("fixed_0.70")
        & df["method"].eq("P3")
        & df["estimator"].isin(["K1", "K2", "K3"])
    ].copy()
    part = part.sort_values(["cohens_kappa", "sensitivity", "specificity"], ascending=[False, False, False])
    return str(part.iloc[0]["estimator"])


def add_h2_to_split_and_heldout(split: pd.DataFrame, heldout: pd.DataFrame, best_k: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = split.copy()
    h = heldout.copy()
    for method in METHODS:
        for suffix in ("fev1_l", "fvc_l", "ratio"):
            src = f"{method}_{best_k}_{suffix}"
            dst = f"{method}_H2_{suffix}"
            if src in out.columns:
                out[dst] = out[src]
    for method in ("P2", "P3"):
        h[f"remaining_{method}_H2_ratio"] = h[f"remaining_{method}_{best_k}_ratio"]
    return out, h


def run(root: Path) -> dict[str, Any]:
    start = time.perf_counter()
    results_dir = root / "results"
    column_map = load_column_map(root)
    truth = load_truth(root, column_map)
    gli = load_gli(root)
    maneuvers = load_manoeuvres_fast(root, truth)
    participants_base = base_pool(maneuvers, ("model", "observed"))
    split_base = attach_split_and_gli(participants_base, gli)
    split_by_seqn = split_base[["SEQN", "split"]].copy()
    maneuvers = maneuvers.merge(split_by_seqn, on="SEQN", how="inner")
    train_m = maneuvers[maneuvers["split"].eq("train")].copy()

    offsets = fit_offsets(train_m)
    mult = fit_multiplicative(split_base[split_base["split"].eq("train")].copy())
    maneuvers_corr = add_corrections(maneuvers, offsets, mult)
    estimators = ESTIMATORS_BASE
    participants = base_pool(maneuvers_corr, estimators)
    split = split_by_seqn.merge(participants, on="SEQN", how="left").merge(gli, on="SEQN", how="left")
    rows: list[dict[str, Any]] = []
    rows.extend(part1_rows(train_m))
    for item in offsets.values():
        rows.append({"row_type": "part2_offset_fit", **item})
    rows.append({"row_type": "part2_multiplicative_fit", **mult})
    rows.extend(residual_rows(maneuvers_corr, split_by_seqn, ("model", "K1", "K2", "K3", "model_mult", "H1")))
    class_rows = classification_rows(split, estimators)
    rows.extend(class_rows)
    for corrected in ("K1", "K2", "K3", "model_mult"):
        rows.append(flip_rows(split, "model", corrected, "fixed_0.70"))
        rows.append(flip_rows(split, "model", corrected, "GLI_LLN"))
    best_k = choose_best_k(class_rows)

    heldout = heldout_split_rows(maneuvers_corr, estimators)
    heldout_rows = heldout_metrics_rows(heldout, estimators)
    rows.extend(heldout_rows)
    rows.extend(gap_rows(class_rows, heldout_rows, estimators))
    heldout_test = heldout[heldout["split"].eq("test")].copy()
    for estimator, ratio_col in [
        (best_k, f"remaining_P3_{best_k}_ratio"),
        ("H1", "remaining_P3_H1_ratio"),
        ("observed", "remaining_P3_observed_ratio"),
    ]:
        rows.append({"row_type": "part3_heldout_kappa_bootstrap", "estimator": estimator, "method": "P3", **bootstrap_kappa_ci(heldout_test, ratio_col)})

    split, heldout = add_h2_to_split_and_heldout(split, heldout, best_k)
    rows.extend(gli_lungsize_rows(split, best_k))
    h2_class_rows = classification_rows(split, ("H2",))
    for row in h2_class_rows:
        row["row_type"] = "part5_h2_classification"
        row["endpoint"] = "NHANES_target"
        row["best_k_for_h2"] = best_k
    rows.extend(h2_class_rows)
    h2_held_rows = heldout_metrics_rows(heldout, ("H2",))
    for row in h2_held_rows:
        if row["row_type"] == "part3_heldout_classification":
            row["row_type"] = "part5_h2_heldout_classification"
            row["best_k_for_h2"] = best_k
        else:
            continue
        rows.append(row)

    raw_path, raw_note = find_or_unzip_raw(root)
    raw, raw_info = read_raw(raw_path)
    rows.extend(truncation_rows(root, raw, split_by_seqn, best_k, offsets))

    result = summarize_result(rows, split, heldout, best_k, offsets, mult, raw_path, raw_note, raw_info, start)
    rows_df = pd.DataFrame(rows)
    write_outputs(results_dir, result, rows_df)
    return result


def metric_lookup(rows: list[dict[str, Any]], row_type: str, **filters: Any) -> dict[str, Any]:
    for row in rows:
        if row.get("row_type") != row_type:
            continue
        if all(row.get(k) == v for k, v in filters.items()):
            return row
    return {}


def summarize_result(
    rows: list[dict[str, Any]],
    split: pd.DataFrame,
    heldout: pd.DataFrame,
    best_k: str,
    offsets: dict[str, dict[str, Any]],
    mult: dict[str, float],
    raw_path: Path,
    raw_note: str,
    raw_info: dict[str, Any],
    start: float,
) -> dict[str, Any]:
    model = metric_lookup(rows, "part2_classification", estimator="model", method="P3", definition="fixed_0.70")
    best = metric_lookup(rows, "part2_classification", estimator=best_k, method="P3", definition="fixed_0.70")
    observed = metric_lookup(rows, "part2_classification", estimator="observed", method="P3", definition="fixed_0.70")
    mult_row = metric_lookup(rows, "part2_classification", estimator="model_mult", method="P3", definition="fixed_0.70")
    held_best = metric_lookup(rows, "part3_heldout_classification", estimator=best_k, method="P3", definition="fixed_0.70")
    held_obs = metric_lookup(rows, "part3_heldout_classification", estimator="observed", method="P3", definition="fixed_0.70")
    h1_held = metric_lookup(rows, "part3_heldout_classification", estimator="H1", method="P3", definition="fixed_0.70")
    h2_held = metric_lookup(rows, "part5_h2_heldout_classification", estimator="H2", method="P3", definition="fixed_0.70")
    boot_best = metric_lookup(rows, "part3_heldout_kappa_bootstrap", estimator=best_k, method="P3")
    boot_obs = metric_lookup(rows, "part3_heldout_kappa_bootstrap", estimator="observed", method="P3")
    boot_h1 = metric_lookup(rows, "part3_heldout_kappa_bootstrap", estimator="H1", method="P3")
    if best.get("cohens_kappa", np.nan) >= observed.get("cohens_kappa", np.nan) - 0.03 and held_best.get("cohens_kappa", np.nan) >= held_obs.get("cohens_kappa", np.nan) - 0.03:
        offset_verdict = "CLOSES"
    elif best.get("cohens_kappa", -np.inf) > model.get("cohens_kappa", np.nan) or held_best.get("cohens_kappa", -np.inf) > metric_lookup(rows, "part3_heldout_classification", estimator="model", method="P3", definition="fixed_0.70").get("cohens_kappa", np.nan):
        offset_verdict = "PARTIALLY CLOSES"
    else:
        offset_verdict = "DOES NOT CLOSE"
    bias_flip = metric_lookup(rows, "part2_classification_flip", corrected=best_k, definition="fixed_0.70")
    bias_degraded = bool(bias_flip and bias_flip.get("kappa_delta", 0.0) < 0)
    honest_best = pd.DataFrame([r for r in rows if r.get("row_type") == "part3_heldout_classification" and r.get("method") == "P3"])
    honest_name = str(honest_best.sort_values("cohens_kappa", ascending=False).iloc[0]["estimator"]) if len(honest_best) else "NA"
    if honest_name == "observed":
        distinguishable = False
    else:
        honest_boot = boot_h1 if honest_name == "H1" else boot_best if honest_name == best_k else {}
        distinguishable = bool(
            np.isfinite(honest_boot.get("ci_low", np.nan))
            and np.isfinite(boot_obs.get("ci_low", np.nan))
            and not (honest_boot["ci_low"] <= boot_obs["ci_high"] and boot_obs["ci_low"] <= honest_boot["ci_high"])
        )
    gli_q = pd.DataFrame([r for r in rows if r.get("row_type") == "part4_gli_fn_by_fvc_quartile"])
    gli_corr = pd.DataFrame([r for r in rows if r.get("row_type") == "part4_gli_after_correction"])
    lungsize_confirmed = False
    if len(gli_q):
        model_q = gli_q[gli_q["estimator"].eq("model")].copy()
        obs_q = gli_q[gli_q["estimator"].eq("observed")].copy()
        if len(model_q) and len(obs_q):
            q_order = ["Q1_smallest", "Q2", "Q3", "Q4_largest"]
            model_q = model_q.set_index("fvc_quartile").reindex(q_order)
            obs_q = obs_q.set_index("fvc_quartile").reindex(q_order)
            lungsize_confirmed = bool(
                np.isfinite(model_q["false_negative_rate_obstructed"].iloc[0])
                and model_q["false_negative_rate_obstructed"].iloc[0] >= model_q["false_negative_rate_obstructed"].iloc[-1] + 0.05
                and not (
                    np.isfinite(obs_q["false_negative_rate_obstructed"].iloc[0])
                    and obs_q["false_negative_rate_obstructed"].iloc[0] >= obs_q["false_negative_rate_obstructed"].iloc[-1] + 0.05
                )
            )
    if len(gli_corr) >= 2:
        model_gli = gli_corr[gli_corr["estimator"].eq("model")].iloc[0]
        best_gli = gli_corr[gli_corr["estimator"].eq(best_k)].iloc[0]
        gli_fn_drop = float((1.0 - model_gli["sensitivity"]) - (1.0 - best_gli["sensitivity"]))
    else:
        gli_fn_drop = np.nan
    gli_verdict = "CONFIRMED" if lungsize_confirmed and np.isfinite(gli_fn_drop) and gli_fn_drop > 0.03 else "REFUTED"
    trunc = pd.DataFrame([r for r in rows if r.get("row_type") == "part5_truncation_classification"])
    complete_estim = "observed" if observed.get("cohens_kappa", -np.inf) >= max(best.get("cohens_kappa", -np.inf), metric_lookup(rows, "part5_h2_classification", estimator="H2", method="P3", definition="fixed_0.70").get("cohens_kappa", -np.inf)) else best_k
    if len(trunc):
        trunc_valid = trunc[trunc["estimator"].isin(["E0_DIRECT", "E1_OMBE", "H1", "H2"])].copy()
        trunc_summary = trunc_valid.groupby("estimator", dropna=False)["cohens_kappa"].mean().sort_values(ascending=False)
        truncated_estim = str(trunc_summary.index[0]) if len(trunc_summary) else "E1_OMBE"
    else:
        truncated_estim = "E1_OMBE"
    return {
        "verdict": {
            "offset": offset_verdict,
            "honest_endpoint_best": honest_name,
            "honest_endpoint_best_distinguishable_from_observed": distinguishable,
            "gli": gli_verdict,
            "deployable_complete": complete_estim,
            "deployable_truncated": truncated_estim,
        },
        "headline": {
            "best_k": best_k,
            "model_nhanes_p3_kappa": model.get("cohens_kappa"),
            "best_k_nhanes_p3_kappa": best.get("cohens_kappa"),
            "observed_nhanes_p3_kappa": observed.get("cohens_kappa"),
            "model_mult_nhanes_p3_kappa": mult_row.get("cohens_kappa"),
            "best_k_heldout_p3_kappa": held_best.get("cohens_kappa"),
            "observed_heldout_p3_kappa": held_obs.get("cohens_kappa"),
            "h1_heldout_p3_kappa": h1_held.get("cohens_kappa"),
            "h2_heldout_p3_kappa": h2_held.get("cohens_kappa"),
            "bias_removal_degraded_kappa": bias_degraded,
            "best_k_flip_row": bias_flip,
            "heldout_bootstrap_best_k": boot_best,
            "heldout_bootstrap_h1": boot_h1,
            "heldout_bootstrap_observed": boot_obs,
            "gli_fn_drop_after_best_k": gli_fn_drop,
        },
        "offset_fits": offsets,
        "multiplicative_fit": mult,
        "inputs": {
            "raw_source": str(raw_path),
            "raw_note": raw_note,
            "raw_loaded_rows": int(raw_info.get("loaded_rows", 0)),
            "participants_n": int(len(split)),
            "test_n": int((split["split"] == "test").sum()),
            "heldout_n": int(len(heldout)),
            "wall_clock_s": float(time.perf_counter() - start),
            "split_note": "Uses the same stratified_half_split helper and seed 20260708 used by the Cycle F pooling/hybrid gates.",
        },
    }


def write_outputs(results_dir: Path, result: dict[str, Any], rows: pd.DataFrame) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(results_dir / f"{OUTPUT_STEM}_rows.csv", index=False)
    (results_dir / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    lines = make_report(result, rows)
    (results_dir / f"{OUTPUT_STEM}.md").write_text(lines, encoding="utf-8")


def rows_by_type(rows: pd.DataFrame, row_type: str) -> list[dict[str, Any]]:
    return rows[rows["row_type"].eq(row_type)].to_dict("records") if "row_type" in rows.columns else []


def make_report(result: dict[str, Any], rows: pd.DataFrame) -> str:
    h = result["headline"]
    lines = [
        "# Gate - Constant-Offset Correction + Lung-Size Interaction, Cycle F",
        "",
        "- Deterministic LM only. No NUTS, SNPE, gamma, registration, or ML residual.",
        "- Clinical threshold 0.70 and GLI LLN were not moved.",
        "- Corrections were fitted on TRAIN only; all classification headline metrics below are TEST.",
        f"- Same split helper/seed as the Cycle F pooling gates: {result['inputs']['split_note']}",
        "",
        "## Verdict",
        "",
        f"OFFSET: constant-offset correction {result['verdict']['offset']} the model-observed gap. Best K variant: {h['best_k']}; NHANES P3 kappa {h['best_k_nhanes_p3_kappa']:.3f}; held-out P3 kappa {h['best_k_heldout_p3_kappa']:.3f}; bias removal degraded kappa: {h['bias_removal_degraded_kappa']}.",
        f"HONEST-ENDPOINT: on held-out-blow, the best estimator is {result['verdict']['honest_endpoint_best']}; distinguishable from observed by bootstrap CI: {result['verdict']['honest_endpoint_best_distinguishable_from_observed']}.",
        f"GLI: lung-size-interaction {result['verdict']['gli']}. If refuted, GLI degradation remains unexplained after the four tested hypotheses.",
        f"DEPLOYABLE: use {result['verdict']['deployable_complete']} for complete curves and {result['verdict']['deployable_truncated']} for truncated curves.",
        "",
        "## Part 1 - Offset Characterization",
        "",
        "Signed rows are model-minus-observed in mL. Absolute-magnitude rows are |model-minus-observed| in mL.",
        "",
        md_table(rows_by_type(rows, "part1_offset_summary"), [("metric", "metric", "str"), ("signedness", "type", "str"), ("truth_severity", "stratum", "str"), ("n", "n", "int"), ("median", "median mL", "float2"), ("iqr_low", "IQR low", "float2"), ("iqr_high", "IQR high", "float2")]),
        "",
        "### Offset Regressions",
        "",
        md_table(rows_by_type(rows, "part1_offset_regression"), [("metric", "metric", "str"), ("signedness", "type", "str"), ("predictor", "predictor", "str"), ("n", "n", "int"), ("slope", "slope", "float5"), ("r2", "R2", "float4"), ("p_value", "p", "float4")]),
        "",
        "## Part 2 - Constant-Offset Correction",
        "",
        "### Fitted Corrections",
        "",
        md_table(rows_by_type(rows, "part2_offset_fit"), [("variant", "variant", "str"), ("form", "form", "str"), ("n", "n", "int"), ("c", "c L", "float5"), ("c0", "c0 L", "float5"), ("c1", "c1", "float5"), ("r2", "R2", "float4")]),
        "",
        "### Multiplicative Baseline",
        "",
        md_table(rows_by_type(rows, "part2_multiplicative_fit"), [("n", "n", "int"), ("a", "a", "float6"), ("b", "b", "float6"), ("r2", "R2", "float4")]),
        "",
        "### FVC Residuals",
        "",
        md_table(rows_by_type(rows, "part2_fvc_residual"), [("estimator", "estimator", "str"), ("split", "split", "str"), ("truth_severity", "stratum", "str"), ("n", "n", "int"), ("median", "median mL", "float2"), ("iqr_low", "IQR low", "float2"), ("iqr_high", "IQR high", "float2"), ("iqr_crosses_zero", "IQR crosses 0", "bool")]),
        "",
        "### Ratio Residuals on TEST",
        "",
        md_table([r for r in rows_by_type(rows, "part2_ratio_residual") if r.get("split") == "test"], [("estimator", "estimator", "str"), ("truth_severity", "stratum", "str"), ("n", "n", "int"), ("median", "median", "float4"), ("iqr_low", "IQR low", "float4"), ("iqr_high", "IQR high", "float4"), ("iqr_crosses_zero", "IQR crosses 0", "bool")]),
        "",
        "### TEST Classification, NHANES Target",
        "",
        md_table(rows_by_type(rows, "part2_classification"), [("estimator", "estimator", "str"), ("method", "method", "str"), ("definition", "definition", "str"), ("n", "n", "int"), ("sensitivity", "sens", "float3"), ("specificity", "spec", "float3"), ("cohens_kappa", "kappa", "float3"), ("borderline_confident_false_negative_rate", "borderline FN", "float3")]),
        "",
        "### Classification Flips vs Uncorrected Model",
        "",
        md_table(rows_by_type(rows, "part2_classification_flip"), [("corrected", "corrected", "str"), ("definition", "definition", "str"), ("n", "n", "int"), ("normal_to_obstructed", "normal->obs", "int"), ("obstructed_to_normal", "obs->normal", "int"), ("sensitivity_delta", "sens delta", "float3"), ("specificity_delta", "spec delta", "float3"), ("kappa_delta", "kappa delta", "float3")]),
        "",
        "## Part 3 - Non-Circular Held-Out-Blow Endpoint",
        "",
        md_table(rows_by_type(rows, "part3_heldout_metadata"), [("n", "n", "int"), ("test_n", "test n", "int"), ("nonbest_heldout_fraction", "nonbest frac", "float3"), ("selection_rule", "rule", "str")]),
        "",
        md_table(rows_by_type(rows, "part3_heldout_classification"), [("estimator", "estimator", "str"), ("method", "method", "str"), ("n", "n", "int"), ("sensitivity", "sens", "float3"), ("specificity", "spec", "float3"), ("cohens_kappa", "kappa", "float3"), ("borderline_confident_false_negative_rate", "borderline FN", "float3")]),
        "",
        "The held-out-blow endpoint is the fair estimator-quality test because the target is another raw-curve readout withheld from pooling, not an NHANES summary value that is itself an observed-curve readout.",
        "",
        "### Endpoint Kappa Gaps vs Observed",
        "",
        md_table(rows_by_type(rows, "part3_endpoint_gap"), [("estimator", "estimator", "str"), ("nhanes_p3_kappa", "NHANES kappa", "float3"), ("nhanes_gap_to_observed", "NHANES gap", "float3"), ("heldout_p3_kappa", "heldout kappa", "float3"), ("heldout_gap_to_observed", "heldout gap", "float3")]),
        "",
        "### Held-Out Bootstrap 95% CI",
        "",
        md_table(rows_by_type(rows, "part3_heldout_kappa_bootstrap"), [("estimator", "estimator", "str"), ("n", "n", "int"), ("kappa", "kappa", "float3"), ("ci_low", "CI low", "float3"), ("ci_high", "CI high", "float3"), ("bootstrap_n", "B", "int")]),
        "",
        "## Part 4 - GLI as Lung-Size Interaction",
        "",
        md_table(rows_by_type(rows, "part4_model_ratio_error_by_fvc_quartile"), [("fvc_quartile", "FVC quartile", "str"), ("fvc_l_low", "low L", "float3"), ("fvc_l_high", "high L", "float3"), ("n", "n", "int"), ("median", "model ratio err med", "float4"), ("iqr_low", "IQR low", "float4"), ("iqr_high", "IQR high", "float4")]),
        "",
        md_table(rows_by_type(rows, "part4_gli_lln_by_fvc_quartile"), [("fvc_quartile", "FVC quartile", "str"), ("n", "n", "int"), ("median", "LLN med", "float4"), ("iqr_low", "IQR low", "float4"), ("iqr_high", "IQR high", "float4")]),
        "",
        md_table(rows_by_type(rows, "part4_gli_lln_correlation"), [("predictor", "predictor", "str"), ("n", "n", "int"), ("pearson_r", "r", "float3"), ("slope", "slope", "float5"), ("p_value", "p", "float4")]),
        "",
        "### GLI False Negatives by FVC Quartile",
        "",
        md_table(rows_by_type(rows, "part4_gli_fn_by_fvc_quartile"), [("fvc_quartile", "FVC quartile", "str"), ("estimator", "estimator", "str"), ("n", "n", "int"), ("sensitivity", "sens", "float3"), ("specificity", "spec", "float3"), ("false_negative_rate_obstructed", "FN obs", "float3"), ("borderline_false_negative_rate", "borderline FN", "float3")]),
        "",
        "### GLI After Best K Correction",
        "",
        md_table(rows_by_type(rows, "part4_gli_after_correction"), [("estimator", "estimator", "str"), ("method", "method", "str"), ("n", "n", "int"), ("sensitivity", "sens", "float3"), ("specificity", "spec", "float3"), ("cohens_kappa", "kappa", "float3"), ("borderline_confident_false_negative_rate", "borderline FN", "float3")]),
        "",
        "## Part 5 - H2 and Truncation Fallback",
        "",
        f"H2 = model FEV1 / {h['best_k']}-corrected model FVC.",
        "",
        "### H2 Complete-Curve Classification",
        "",
        md_table(rows_by_type(rows, "part5_h2_classification"), [("method", "method", "str"), ("definition", "definition", "str"), ("n", "n", "int"), ("sensitivity", "sens", "float3"), ("specificity", "spec", "float3"), ("cohens_kappa", "kappa", "float3"), ("borderline_confident_false_negative_rate", "borderline FN", "float3")]),
        "",
        "### H2 Held-Out-Blow Classification",
        "",
        md_table(rows_by_type(rows, "part5_h2_heldout_classification"), [("method", "method", "str"), ("n", "n", "int"), ("sensitivity", "sens", "float3"), ("specificity", "spec", "float3"), ("cohens_kappa", "kappa", "float3"), ("borderline_confident_false_negative_rate", "borderline FN", "float3")]),
        "",
        "### Truncation Classification",
        "",
        md_table(rows_by_type(rows, "part5_truncation_classification"), [("estimator", "estimator", "str"), ("T_label", "T", "str"), ("n", "n", "int"), ("sensitivity", "sens", "float3"), ("specificity", "spec", "float3"), ("cohens_kappa", "kappa", "float3"), ("obstructed_false_negative_rate", "obstructed FN", "float3")]),
        "",
        "### Truncation Offset Refit Drift",
        "",
        md_table(rows_by_type(rows, "part5_truncation_offset_refit"), [("T_label", "T", "str"), ("train_n", "train n", "int"), ("refit_constant_offset_l", "refit c L", "float5"), ("complete_K1_offset_l", "complete K1 c L", "float5"), ("drift_from_complete_l", "drift L", "float5")]),
        "",
        "## Output Files",
        "",
        f"- `results/{OUTPUT_STEM}.md`",
        f"- `results/{OUTPUT_STEM}.json`",
        f"- `results/{OUTPUT_STEM}_rows.csv`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    h = result["headline"]
    print(f"OFFSET: {result['verdict']['offset']} best={h['best_k']} NHANES_kappa={h['best_k_nhanes_p3_kappa']:.3f} heldout_kappa={h['best_k_heldout_p3_kappa']:.3f}")
    print(f"HONEST-ENDPOINT: best={result['verdict']['honest_endpoint_best']} distinguishable_from_observed={result['verdict']['honest_endpoint_best_distinguishable_from_observed']}")
    print(f"GLI: {result['verdict']['gli']} fn_drop_after_bestK={h['gli_fn_drop_after_best_k']:.3f}")
    print(f"DEPLOYABLE: complete={result['verdict']['deployable_complete']} truncated={result['verdict']['deployable_truncated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
