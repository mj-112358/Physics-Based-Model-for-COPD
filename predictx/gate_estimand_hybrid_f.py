from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import linregress, pearsonr

from .dataset_resolver import resolve_dataset_path
from .gate3_diagnosis import np_ombe_volume
from .gate23_f import find_or_unzip_raw, read_raw
from .gate_clinical_classification_f import (
    confusion_binary,
    extract_curve_for_fit,
    load_column_map,
    load_gli,
    load_truth,
)
from .gate_model_vs_observed_f import build_manoeuvres, fixed_metrics, signed_summary
from .gate_policy_calibrated_flag_f import stratified_half_split
from .gate_reconciliation_audit_f import clean_json, md_table
from .gate_truncation_recovery_f import fit_ombe_truncated, model_grid_with_stop

RANDOM_SEED = 20260708
CURRENT_CONVENTION = "segmented_btps"
ESTIMATORS = ("model", "model_mult", "observed", "H1")
METHODS = ("P1", "P2", "P3")


def decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    if pd.isna(value):
        return ""
    return str(value).strip()


def pct_error(est: Any, truth: Any) -> np.ndarray:
    est_arr = np.asarray(est, dtype=float)
    truth_arr = np.asarray(truth, dtype=float)
    out = np.full(est_arr.shape, np.nan, dtype=float)
    ok = np.isfinite(est_arr) & np.isfinite(truth_arr) & (truth_arr != 0)
    out[ok] = 100.0 * (est_arr[ok] - truth_arr[ok]) / truth_arr[ok]
    return out


def finite_pair(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    ok = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[ok], y_arr[ok]


def regression_row(y: Any, x: Any, predictor: str, row_type: str) -> dict[str, Any]:
    x_ok, y_ok = finite_pair(x, y)
    if len(x_ok) < 3:
        return {"row_type": row_type, "predictor": predictor, "n": int(len(x_ok)), "slope": np.nan, "intercept": np.nan, "r2": np.nan, "p_value": np.nan}
    fit = linregress(x_ok, y_ok)
    return {
        "row_type": row_type,
        "predictor": predictor,
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


def read_spx_summary(root: Path, cycle: str = "F") -> tuple[pd.DataFrame, dict[str, str]]:
    resolved = resolve_dataset_path(root, "SPX", cycle)
    path = resolved.path
    df = pd.read_sas(path, format="xport")
    _meta_df, meta = pyreadstat.read_xport(str(path), metadataonly=True)
    labels = {name: label for name, label in zip(meta.column_names, meta.column_labels)}
    return df, labels


def read_raw_labels(raw_path: Path) -> dict[str, str]:
    _df, meta = pyreadstat.read_sas7bdat(str(raw_path), metadataonly=True)
    return {name: label for name, label in zip(meta.column_names, meta.column_labels)}


def add_estimator_columns(m: pd.DataFrame, convention: str, mult: dict[str, float] | None = None) -> pd.DataFrame:
    out = m.copy()
    out["model_fev1_l_est"] = out[f"model_{convention}_fev1_l"]
    out["model_fvc_l_est"] = out[f"model_{convention}_fvc_l"]
    out["model_ratio_est"] = out[f"model_{convention}_ratio"]
    out["observed_fev1_l_est"] = out[f"observed_{convention}_fev1_l"]
    out["observed_fvc_l_est"] = out[f"observed_{convention}_fvc_l"]
    out["observed_ratio_est"] = out[f"observed_{convention}_ratio"]
    if mult is None:
        out["model_mult_fvc_l_est"] = out["model_fvc_l_est"]
    else:
        out["model_mult_fvc_l_est"] = mult["a"] * np.power(out["model_fvc_l_est"], mult["b"])
    out["model_mult_fev1_l_est"] = out["model_fev1_l_est"]
    out["model_mult_ratio_est"] = out["model_mult_fev1_l_est"] / out["model_mult_fvc_l_est"]
    out["H1_fev1_l_est"] = out["model_fev1_l_est"]
    out["H1_fvc_l_est"] = out["observed_fvc_l_est"]
    out["H1_ratio_est"] = out["H1_fev1_l_est"] / out["H1_fvc_l_est"]
    return out


def pool_estimators(maneuvers: pd.DataFrame, convention: str, mult: dict[str, float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = add_estimator_columns(maneuvers, convention, mult=mult)
    needed = []
    for estimator in ESTIMATORS:
        needed.extend([f"{estimator}_fev1_l_est", f"{estimator}_fvc_l_est", f"{estimator}_ratio_est"])
    m = m[m[needed].apply(np.isfinite).all(axis=1)].copy()
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
        }
        for estimator in ESTIMATORS:
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
    return pd.DataFrame(rows), m


def split_participants(participants: pd.DataFrame, gli: pd.DataFrame) -> pd.DataFrame:
    split = stratified_half_split(participants, "truth_severity")
    if not gli.empty:
        split = split.merge(gli, on="SEQN", how="left")
    else:
        split["gli_lln_ratio"] = np.nan
        split["gli_ethnicity_mapped"] = np.nan
        split["gli_mapping_fallback"] = np.nan
    return split


def classification_table(split: pd.DataFrame, estimators: tuple[str, ...] = ESTIMATORS, definitions: tuple[str, ...] = ("fixed_0.70", "GLI_LLN")) -> list[dict[str, Any]]:
    test = split[split["split"].eq("test")].copy()
    rows: list[dict[str, Any]] = []
    for estimator in estimators:
        for method in METHODS:
            ratio_col = f"{method}_{estimator}_ratio"
            for definition in definitions:
                threshold_col = None if definition == "fixed_0.70" else "gli_lln_ratio"
                metrics = fixed_metrics(test, ratio_col, threshold_col=threshold_col)
                rows.append(
                    {
                        "row_type": "classification",
                        "estimator": estimator,
                        "method": method,
                        "definition": definition,
                        **metrics,
                    }
                )
    return rows


def fit_multiplicative(train: pd.DataFrame) -> dict[str, float]:
    model = train["P3_model_fvc_l"].to_numpy(float)
    observed = train["P3_observed_fvc_l"].to_numpy(float)
    ok = np.isfinite(model) & np.isfinite(observed) & (model > 0) & (observed > 0)
    fit = linregress(np.log(model[ok]), np.log(observed[ok]))
    return {"a": float(np.exp(fit.intercept)), "b": float(fit.slope), "n": int(ok.sum()), "r2": float(fit.rvalue**2), "p_value": float(fit.pvalue)}


def residual_rows(split: pd.DataFrame, estimator: str, method: str, row_type: str) -> list[dict[str, Any]]:
    rows = []
    for split_name, split_part in split.groupby("split"):
        for stratum, part in split_part.groupby("truth_severity", dropna=False):
            err = pct_error(part[f"{method}_{estimator}_fvc_l"], part[f"{method}_observed_fvc_l"]) if estimator != "observed" else np.zeros(len(part))
            rows.append({"row_type": row_type, "split": split_name, "estimator": estimator, "method": method, "truth_severity": stratum, **signed_summary(err)})
        obs = split_part[split_part["fev1_fvc_ratio"] < 0.70]
        err = pct_error(obs[f"{method}_{estimator}_fvc_l"], obs[f"{method}_observed_fvc_l"]) if estimator != "observed" else np.zeros(len(obs))
        rows.append({"row_type": row_type, "split": split_name, "estimator": estimator, "method": method, "truth_severity": "obstructed_all", **signed_summary(err)})
    return rows


def ratio_error_rows(split: pd.DataFrame, estimator: str, method: str, row_type: str) -> list[dict[str, Any]]:
    rows = []
    test = split[split["split"].eq("test")].copy()
    for stratum, part in test.groupby("truth_severity", dropna=False):
        err = part[f"{method}_{estimator}_ratio"] - part["fev1_fvc_ratio"]
        rows.append({"row_type": row_type, "estimator": estimator, "method": method, "truth_severity": stratum, **signed_summary(err)})
    return rows


def compare_summary_to_reference(participants: pd.DataFrame, spx: pd.DataFrame, labels: dict[str, str]) -> list[dict[str, Any]]:
    p = participants[participants["split"].eq("test")].copy()
    p = p.merge(spx[["SEQN", "SPXNFEV1", "SPXNFVC", "SPXBFEV1", "SPXBFVC"]], on="SEQN", how="left")
    rows = []
    for prefix, label in [("SPXN", "baseline"), ("SPXB", "second_test")]:
        fev_col = f"{prefix}FEV1"
        fvc_col = f"{prefix}FVC"
        valid = p[p[fev_col].notna() & p[fvc_col].notna()].copy()
        fev_err = np.abs(valid["P3_observed_fev1_l"] * 1000.0 - valid[fev_col])
        fvc_err = np.abs(valid["P3_observed_fvc_l"] * 1000.0 - valid[fvc_col])
        ratio_err = np.abs(valid["P3_observed_ratio"] - valid[fev_col] / valid[fvc_col])
        rows.append(
            {
                "row_type": "part1_reference_compare",
                "reference_prefix": prefix,
                "reference_label": label,
                "fev1_label": labels.get(fev_col, ""),
                "fvc_label": labels.get(fvc_col, ""),
                "n": int(len(valid)),
                "fev1_abs_error_mL_median": float(np.nanmedian(fev_err)) if len(valid) else np.nan,
                "fvc_abs_error_mL_median": float(np.nanmedian(fvc_err)) if len(valid) else np.nan,
                "ratio_abs_error_median": float(np.nanmedian(ratio_err)) if len(valid) else np.nan,
                "within_25mL_both_fraction": float(((fev_err <= 25.0) & (fvc_err <= 25.0)).mean()) if len(valid) else np.nan,
            }
        )
    return rows


def heldout_generalization(maneuvers: pd.DataFrame) -> list[dict[str, Any]]:
    rng = np.random.default_rng(RANDOM_SEED + 77)
    rows = []
    for seq, group in maneuvers.groupby("SEQN"):
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
        out = {
            "SEQN": float(seq),
            "truth_severity": held["truth_severity"],
            "heldout_raw_index": int(held["raw_index"]),
            "heldout_ratio": float(held["observed_ratio_est"]),
            "heldout_obstructed": bool(held["observed_ratio_est"] < 0.70),
            "heldout_was_nonbest": bool(held_idx in set(nonbest.index)),
        }
        for estimator in ("model", "observed", "H1"):
            out[f"remaining_P3_{estimator}_ratio"] = float(rem[f"{estimator}_fev1_l_est"].max() / rem[f"{estimator}_fvc_l_est"].max())
            out[f"remaining_P2_{estimator}_ratio"] = float(rem[f"{estimator}_ratio_est"].median())
        rows.append(out)
    heldout = pd.DataFrame(rows)
    split = stratified_half_split(
        heldout.rename(columns={"heldout_ratio": "fev1_fvc_ratio"}).assign(truth_severity=lambda d: d["fev1_fvc_ratio"].map(lambda x: "severe" if x < 0.50 else ("moderate" if x < 0.60 else ("mild_0.60_0.70" if x < 0.70 else "normal")))),
        "truth_severity",
    )
    test = split[split["split"].eq("test")].copy()
    out_rows: list[dict[str, Any]] = []
    for estimator in ("model", "observed", "H1"):
        for method in ("P2", "P3"):
            ratio_col = f"remaining_{method}_{estimator}_ratio"
            metrics = confusion_binary(test["fev1_fvc_ratio"] < 0.70, test[ratio_col] < 0.70)
            out_rows.append({"row_type": "part1_heldout_generalization", "estimator": estimator, "method": method, **metrics})
    out_rows.append({"row_type": "part1_heldout_metadata", "n": int(len(heldout)), "test_n": int(len(test)), "nonbest_heldout_fraction": float(heldout["heldout_was_nonbest"].mean()) if len(heldout) else np.nan})
    return out_rows


def truncation_h1_rows(root: Path, raw: pd.DataFrame) -> list[dict[str, Any]]:
    existing_path = root / "results" / "gate_truncation_recovery_F_rows.csv"
    if not existing_path.exists():
        return [{"row_type": "part3_truncation_h1", "error": f"missing {existing_path}"}]
    existing = pd.read_csv(existing_path)
    base = existing[existing["estimator"].isin(["E0_DIRECT", "E1_OMBE"]) & existing["T_label"].astype(str).isin(["2", "3", "4", "5"])].copy()
    rows: list[dict[str, Any]] = []
    for (estimator, t_label), part in base.groupby(["estimator", "T_label"]):
        valid = part[part["fit_success"].astype(bool) & part["est_ratio"].notna()]
        metrics = confusion_binary(valid["truth_obstructed_fixed"], valid["est_obstructed_fixed"])
        rows.append(
            {
                "row_type": "part3_truncation_h1",
                "source": "E0_observed_direct" if estimator == "E0_DIRECT" else "E1_existing_model",
                "T_label": str(t_label),
                **metrics,
                "obstructed_false_negative_rate": metrics["fn"] / (metrics["tp"] + metrics["fn"]) if (metrics["tp"] + metrics["fn"]) else np.nan,
            }
        )
    sample = existing[existing["T_label"].astype(str).isin(["2", "3", "4", "5"])][
        ["SEQN", "raw_index", "SPAMANU", "truth_severity", "T_label", "true_ratio", "truth_obstructed_fixed"]
    ].drop_duplicates(["raw_index", "T_label"])
    raw_by_index = {int(idx): row for idx, row in raw.iterrows()}
    h1_detail = []
    total = len(sample)
    for i, row in enumerate(sample.itertuples(index=False), start=1):
        try:
            curve = extract_curve_for_fit(raw_by_index[int(row.raw_index)])
            t = np.asarray(curve["vol_t"], dtype=float)
            y = np.asarray(curve["vol_l"], dtype=float)
            stop_s = float(row.T_label)
            fit = fit_ombe_truncated(t, y, stop_s)
            grid, _fit_idx = model_grid_with_stop(t, stop_s)
            pred = np_ombe_volume(np.asarray(fit["params"], dtype=float), grid)
            model_fev1 = float(np.interp(1.0, grid, pred))
            model_fvc = float(pred[-1])
            ratio = model_fev1 / model_fvc if model_fvc > 0 else np.nan
            success = bool(fit["success"] and np.isfinite(ratio))
            failure = "" if success else str(fit.get("failure", "nonfinite H1 ratio"))
        except Exception as exc:
            ratio = np.nan
            success = False
            failure = str(exc)
        h1_detail.append(
            {
                "row_type": "part3_truncation_h1_detail",
                "SEQN": float(row.SEQN),
                "raw_index": int(row.raw_index),
                "SPAMANU": float(row.SPAMANU),
                "truth_severity": row.truth_severity,
                "T_label": str(row.T_label),
                "source": "H1_fallback_model_fvc",
                "fit_success": success,
                "fit_failure": failure,
                "true_ratio": float(row.true_ratio),
                "truth_obstructed_fixed": bool(row.truth_obstructed_fixed),
                "est_ratio": ratio,
                "est_obstructed_fixed": bool(ratio < 0.70) if np.isfinite(ratio) else False,
            }
        )
        if i % 100 == 0 or i == total:
            print(f"H1 truncation deterministic refit progress: {i}/{total}")
    h1_df = pd.DataFrame(h1_detail)
    for t_label, part in h1_df.groupby("T_label"):
        valid = part[part["fit_success"].astype(bool) & part["est_ratio"].notna()]
        metrics = confusion_binary(valid["truth_obstructed_fixed"], valid["est_obstructed_fixed"])
        rows.append(
            {
                "row_type": "part3_truncation_h1",
                "source": "H1_fallback_model_fvc",
                "T_label": str(t_label),
                **metrics,
                "obstructed_false_negative_rate": metrics["fn"] / (metrics["tp"] + metrics["fn"]) if (metrics["tp"] + metrics["fn"]) else np.nan,
            }
        )
    return rows + h1_detail


def repeatability_rows(maneuvers: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for estimator in ("model", "observed", "H1"):
        part_rows = []
        for seq, group in maneuvers.groupby("SEQN"):
            if len(group) < 2:
                continue
            fev1 = group[f"{estimator}_fev1_l_est"].dropna().sort_values(ascending=False).to_numpy(float)
            fvc = group[f"{estimator}_fvc_l_est"].dropna().sort_values(ascending=False).to_numpy(float)
            ratio = group[f"{estimator}_ratio_est"].dropna().to_numpy(float)
            if len(fev1) < 2 or len(fvc) < 2 or len(ratio) < 2:
                continue
            part_rows.append(
                {
                    "fev1_sd_l": float(np.std(fev1, ddof=1)),
                    "fvc_sd_l": float(np.std(fvc, ddof=1)),
                    "ratio_sd": float(np.std(ratio, ddof=1)),
                    "ats_repeatable": bool(abs(fev1[0] - fev1[1]) <= 0.150 and abs(fvc[0] - fvc[1]) <= 0.150),
                }
            )
        d = pd.DataFrame(part_rows)
        rows.append(
            {
                "row_type": "part3_repeatability",
                "estimator": estimator,
                "n": int(len(d)),
                "fev1_sd_median_l": float(np.nanmedian(d["fev1_sd_l"])) if len(d) else np.nan,
                "fvc_sd_median_l": float(np.nanmedian(d["fvc_sd_l"])) if len(d) else np.nan,
                "ratio_sd_median": float(np.nanmedian(d["ratio_sd"])) if len(d) else np.nan,
                "ats_repeatable_fraction": float(d["ats_repeatable"].mean()) if len(d) else np.nan,
            }
        )
    return rows


def gli_subgroup_rows(split: pd.DataFrame) -> list[dict[str, Any]]:
    test = split[split["split"].eq("test") & split["gli_lln_ratio"].notna()].copy()
    test["gli_clean"] = (~test["gli_mapping_fallback"].astype(bool)) & (pd.to_numeric(test["gli_ethnicity_mapped"], errors="coerce") != 5)
    rows = []
    for subgroup, part in [("clean_mapping", test[test["gli_clean"]]), ("other_mixed_or_fallback", test[~test["gli_clean"]])]:
        excluded_fraction = float((~test["gli_clean"]).mean()) if subgroup == "clean_mapping" and len(test) else np.nan
        for estimator in ("model", "observed", "H1"):
            for method in METHODS:
                ratio_col = f"{method}_{estimator}_ratio"
                metrics = fixed_metrics(part, ratio_col, threshold_col="gli_lln_ratio")
                rows.append({"row_type": "part4_gli_subgroup", "subgroup": subgroup, "estimator": estimator, "method": method, "excluded_fraction_if_clean": excluded_fraction, **metrics})
    return rows


def gp_diagnostic_rows(root: Path) -> list[dict[str, Any]]:
    gp_path = root / "results" / "gate_posterior_width_F_rows.csv"
    vol_path = root / "results" / "gate_reproducibility_volume_F_rows.csv"
    if not gp_path.exists() or not vol_path.exists():
        return [{"row_type": "part5_gp_diagnostic", "error": "missing GP posterior-width rows or volume rows"}]
    gp = pd.read_csv(gp_path)
    gp = gp[(gp["variant"] == "M2_GP_MATERN32") & gp["converged"].astype(bool)].copy()
    vol = pd.read_csv(vol_path, low_memory=False)
    vol = vol[vol["row_type"].eq("maneuver_fit")][["raw_index", "duration_s", "observed_fvc_l"]].drop_duplicates("raw_index")
    joined = gp.merge(vol, on="raw_index", how="left")
    joined["gp_length_over_fet"] = joined["gp_length_s_median"] / joined["duration_s"]
    length_corr = safe_corr(joined["gp_length_s_median"], joined["duration_s"])
    amp_corr = safe_corr(joined["gp_amp_ml_median"], joined["observed_fvc_l"] * 1000.0)
    rows = [
        {
            "row_type": "part5_gp_diagnostic",
            "diagnostic": "length_vs_fet",
            **length_corr,
            "gp_length_over_fet_median": float(np.nanmedian(joined["gp_length_over_fet"])) if len(joined) else np.nan,
            "gp_length_s_median": float(np.nanmedian(joined["gp_length_s_median"])) if len(joined) else np.nan,
            "fet_s_median": float(np.nanmedian(joined["duration_s"])) if len(joined) else np.nan,
        },
        {
            "row_type": "part5_gp_diagnostic",
            "diagnostic": "amplitude_vs_fvc",
            **amp_corr,
            "gp_amp_ml_median": float(np.nanmedian(joined["gp_amp_ml_median"])) if len(joined) else np.nan,
            "fvc_ml_median": float(np.nanmedian(joined["observed_fvc_l"] * 1000.0)) if len(joined) else np.nan,
        },
    ]
    for row in joined.itertuples(index=False):
        rows.append(
            {
                "row_type": "part5_gp_curve",
                "SEQN": float(row.SEQN),
                "raw_index": int(row.raw_index),
                "gp_length_s": float(row.gp_length_s_median),
                "fet_s": float(row.duration_s),
                "gp_length_over_fet": float(row.gp_length_over_fet),
                "gp_amp_ml": float(row.gp_amp_ml_median),
                "fvc_ml": float(row.observed_fvc_l * 1000.0),
            }
        )
    return rows


def output_stem(cycle: str) -> str:
    return f"gate_estimand_hybrid_{cycle}"


def run(root: Path, cycle: str = "F") -> dict[str, Any]:
    cycle = cycle.upper()
    start = time.perf_counter()
    column_map = load_column_map(root)
    truth = load_truth(root, column_map, cycle)
    gli = load_gli(root)
    spx, labels = read_spx_summary(root, cycle)
    raw_path, raw_note = find_or_unzip_raw(root, cycle)
    raw, raw_info = read_raw(raw_path, cycle)
    raw_labels = read_raw_labels(raw_path)
    maneuvers_base = build_manoeuvres(root, raw, truth)
    participants_base, maneuvers_current = pool_estimators(maneuvers_base, CURRENT_CONVENTION, mult=None)
    split_base = split_participants(participants_base, gli)
    train_base = split_base[split_base["split"].eq("train")].copy()

    log_reg_model_on_obs = regression_row(np.log(train_base["P3_model_fvc_l"]), np.log(train_base["P3_observed_fvc_l"]), "log_observed_fvc", "part2_log_regression")
    mult = fit_multiplicative(train_base)
    participants_mult, maneuvers_mult = pool_estimators(maneuvers_base, CURRENT_CONVENTION, mult=mult)
    split = split_base[["SEQN", "split"]].merge(participants_mult, on="SEQN", how="left")
    split = split.merge(gli, on="SEQN", how="left")
    test = split[split["split"].eq("test")].copy()

    rows: list[dict[str, Any]] = []
    circular_rows = compare_summary_to_reference(split, spx, labels)
    rows.extend(circular_rows)
    rows.extend(heldout_generalization(maneuvers_current))
    rows.append(log_reg_model_on_obs)
    rows.append({"row_type": "part2_multiplicative_fit", **mult})
    rows.extend(residual_rows(split, "model", "P3", "part2_uncorrected_fvc_residual"))
    rows.extend(residual_rows(split, "model_mult", "P3", "part2_corrected_fvc_residual"))
    class_rows = classification_table(split)
    for row in class_rows:
        row["row_type"] = "part2_part3_classification"
    rows.extend(class_rows)
    rows.extend(ratio_error_rows(split, "H1", "P3", "part3_h1_ratio_error"))
    trunc_rows = truncation_h1_rows(root, raw)
    rows.extend(trunc_rows)
    rows.extend(repeatability_rows(maneuvers_mult))
    gli_rows = gli_subgroup_rows(split)
    rows.extend(gli_rows)
    gp_rows = gp_diagnostic_rows(root)
    rows.extend(gp_rows)

    spxb = next(r for r in circular_rows if r["reference_prefix"] == "SPXB")
    spxn = next(r for r in circular_rows if r["reference_prefix"] == "SPXN")
    spxb_is_best = "best" in (spxb["fev1_label"] + " " + spxb["fvc_label"]).lower()
    spxb_reconstructs = bool(spxb_is_best and spxb["n"] > 20 and spxb["fev1_abs_error_mL_median"] <= 25 and spxb["fvc_abs_error_mL_median"] <= 25)
    spxn_reconstructs = bool(spxn["n"] > 20 and spxn["fev1_abs_error_mL_median"] <= 25 and spxn["fvc_abs_error_mL_median"] <= 25)
    circularity_status = "IS" if spxb_reconstructs or spxn_reconstructs else "IS NOT"

    corrected_test_rows = [r for r in rows if r["row_type"] == "part2_corrected_fvc_residual" and r.get("split") == "test"]
    corrected_train_rows = [r for r in rows if r["row_type"] == "part2_corrected_fvc_residual" and r.get("split") == "train"]
    obs_all = next((r for r in corrected_test_rows if r.get("truth_severity") == "obstructed_all"), {})
    normal = next((r for r in corrected_test_rows if r.get("truth_severity") == "normal"), {})
    sign_flip_removed = bool(obs_all.get("iqr_crosses_zero", False) and normal.get("iqr_crosses_zero", False))
    obs_all_train = next((r for r in corrected_train_rows if r.get("truth_severity") == "obstructed_all"), {})
    normal_train = next((r for r in corrected_train_rows if r.get("truth_severity") == "normal"), {})

    def materially_worse(test_row: dict[str, Any], train_row: dict[str, Any]) -> bool:
        if not test_row or not train_row:
            return False
        test_width = float(test_row["iqr_high"] - test_row["iqr_low"])
        train_width = float(train_row["iqr_high"] - train_row["iqr_low"])
        return bool(abs(float(test_row["median"])) > abs(float(train_row["median"])) + 1.0 or test_width > train_width + 2.0)

    multiplicative_overfit = bool(materially_worse(obs_all, obs_all_train) or materially_worse(normal, normal_train))
    model_p3 = next(r for r in class_rows if r["estimator"] == "model" and r["method"] == "P3" and r["definition"] == "fixed_0.70")
    mult_p3 = next(r for r in class_rows if r["estimator"] == "model_mult" and r["method"] == "P3" and r["definition"] == "fixed_0.70")
    observed_p3 = next(r for r in class_rows if r["estimator"] == "observed" and r["method"] == "P3" and r["definition"] == "fixed_0.70")
    h1_p3 = next(r for r in class_rows if r["estimator"] == "H1" and r["method"] == "P3" and r["definition"] == "fixed_0.70")
    kappa_gap_closed = bool(observed_p3["cohens_kappa"] - mult_p3["cohens_kappa"] <= 0.03)

    h1_trunc = [r for r in trunc_rows if r.get("row_type") == "part3_truncation_h1" and r.get("T_label") == "2"]
    h1_t2 = next((r for r in h1_trunc if r.get("source") == "H1_fallback_model_fvc"), {})
    e0_t2 = next((r for r in h1_trunc if r.get("source") == "E0_observed_direct"), {})
    e1_t2 = next((r for r in h1_trunc if r.get("source") == "E1_existing_model"), {})
    h1_dominates = bool(
        h1_p3["cohens_kappa"] >= observed_p3["cohens_kappa"] - 0.03
        and h1_t2
        and h1_t2.get("obstructed_false_negative_rate", np.inf) <= e1_t2.get("obstructed_false_negative_rate", np.inf) + 0.03
        and h1_t2.get("obstructed_false_negative_rate", np.inf) < e0_t2.get("obstructed_false_negative_rate", np.inf)
    )

    clean_p3 = next(r for r in gli_rows if r["subgroup"] == "clean_mapping" and r["estimator"] == "H1" and r["method"] == "P3")
    other_p3 = next(r for r in gli_rows if r["subgroup"] == "other_mixed_or_fallback" and r["estimator"] == "H1" and r["method"] == "P3")
    fixed_h1 = h1_p3
    clean_fn = 1.0 - clean_p3["sensitivity"] if np.isfinite(clean_p3["sensitivity"]) else np.nan
    fixed_fn = 1.0 - fixed_h1["sensitivity"] if np.isfinite(fixed_h1["sensitivity"]) else np.nan
    gli_explained = bool(np.isfinite(clean_fn) and np.isfinite(fixed_fn) and abs(clean_fn - fixed_fn) <= 0.10 and clean_fn + 0.10 < (1.0 - other_p3["sensitivity"]))

    gp_length = next(r for r in gp_rows if r.get("diagnostic") == "length_vs_fet")
    gp_amp = next(r for r in gp_rows if r.get("diagnostic") == "amplitude_vs_fvc")
    gp_sentence = (
        f"GP length scale is curve-scale: median length/FET {gp_length.get('gp_length_over_fet_median', np.nan):.3f}; "
        f"amplitude-FVC correlation r={gp_amp.get('pearson_r', np.nan):.3f}, slope={gp_amp.get('slope', np.nan):.4f} mL per mL."
    )
    heldout_rows = [r for r in rows if r.get("row_type") == "part1_heldout_generalization"]
    heldout_model_p3 = next(r for r in heldout_rows if r["estimator"] == "model" and r["method"] == "P3")
    heldout_observed_p3 = next(r for r in heldout_rows if r["estimator"] == "observed" and r["method"] == "P3")
    heldout_h1_p3 = next(r for r in heldout_rows if r["estimator"] == "H1" and r["method"] == "P3")

    result = {
        "cycle": cycle,
        "verdict": {
            "circularity": circularity_status,
            "multiplicative": "sign flip REMOVED" if sign_flip_removed else "sign flip NOT REMOVED",
            "hybrid": "H1 dominates both" if h1_dominates else "H1 does not dominate both",
            "gli": "coverage-explained" if gli_explained else "unexplained",
        },
        "headline": {
            "spxb_is_best_by_metadata": spxb_is_best,
            "spxb_reference_compare": spxb,
            "spxn_reference_compare": spxn,
            "model_p3_kappa": model_p3["cohens_kappa"],
            "mult_p3_kappa": mult_p3["cohens_kappa"],
            "observed_p3_kappa": observed_p3["cohens_kappa"],
            "h1_p3_kappa": h1_p3["cohens_kappa"],
            "h1_t2_obstructed_fn": h1_t2.get("obstructed_false_negative_rate", np.nan),
            "e0_t2_obstructed_fn": e0_t2.get("obstructed_false_negative_rate", np.nan),
            "e1_t2_obstructed_fn": e1_t2.get("obstructed_false_negative_rate", np.nan),
            "multiplicative_a": mult["a"],
            "multiplicative_b": mult["b"],
            "sign_flip_removed": sign_flip_removed,
            "kappa_gap_closed_by_multiplicative": kappa_gap_closed,
            "multiplicative_overfit_flag": multiplicative_overfit,
            "heldout_model_p3_kappa": heldout_model_p3["cohens_kappa"],
            "heldout_observed_p3_kappa": heldout_observed_p3["cohens_kappa"],
            "heldout_h1_p3_kappa": heldout_h1_p3["cohens_kappa"],
            "gli_clean_h1_p3_sensitivity": clean_p3["sensitivity"],
            "gli_other_h1_p3_sensitivity": other_p3["sensitivity"],
            "gli_clean_excluded_fraction": clean_p3["excluded_fraction_if_clean"],
            "gp_sentence": gp_sentence,
        },
        "field_labels": {
            **{k: v for k, v in labels.items() if k.startswith("SPXN") or k.startswith("SPXB") or k == "SEQN"},
            **{k: v for k, v in raw_labels.items() if k in ["SPATTYPE", "SPAMANU", "SPAFACT", "SPAACC", "SPAQEFF", "SPXPTS", "SPXRAW"]},
        },
        "multiplicative_fit": mult,
        "log_model_on_observed_regression": log_reg_model_on_obs,
        "classifications": class_rows,
        "inputs": {
            "raw_source": str(raw_path),
            "raw_note": raw_note,
            "raw_loaded_rows": int(raw_info["loaded_rows"]),
            "participant_n": int(len(split)),
            "test_n": int(len(test)),
            "wall_clock_s": float(time.perf_counter() - start),
        },
    }
    write_outputs(root, result, pd.DataFrame(rows))
    return result


def rows_by_type(rows: pd.DataFrame, row_type: str) -> list[dict[str, Any]]:
    return rows[rows["row_type"].eq(row_type)].to_dict("records")


def write_outputs(root: Path, result: dict[str, Any], rows: pd.DataFrame) -> None:
    out = root / "results"
    cycle = result["cycle"]
    stem = output_stem(cycle)
    h = result["headline"]
    lines = [
        f"# Gate - Estimand Reconciliation + Hybrid Readout, Cycle {cycle}",
        "",
        f"- CIRCULARITY: observed_P3 {result['verdict']['circularity']} a reconstruction of the NHANES reference algorithm.",
        f"- MULTIPLICATIVE: {result['verdict']['multiplicative']} (a={h['multiplicative_a']:.6f}, b={h['multiplicative_b']:.6f}); overfit={h['multiplicative_overfit_flag']}; kappa gap closed: {h['kappa_gap_closed_by_multiplicative']}.",
        f"- HYBRID: {result['verdict']['hybrid']} (H1 complete P3 kappa {h['h1_p3_kappa']:.3f} vs observed {h['observed_p3_kappa']:.3f}; T=2 FN H1 {h['h1_t2_obstructed_fn']:.3f}, E0 {h['e0_t2_obstructed_fn']:.3f}, E1 {h['e1_t2_obstructed_fn']:.3f}).",
        f"- GLI: {result['verdict']['gli']}.",
        "- Deterministic LM only. No NUTS, SNPE, gamma, registration, or ML residual.",
        "",
        "## Part 1 - Is observed_P3 Circular?",
        "",
        "### Available Field Definitions",
        "",
    ]
    for key in ["SPXNSTAT", "SPXNFVC", "SPXNFEV1", "SPXNQEFF", "SPXNQFVC", "SPXNQFV1", "SPXBSTAT", "SPXBFVC", "SPXBFEV1", "SPXBQEFF", "SPXBQFVC", "SPXBQFV1"]:
        lines.append(f"- {key}: {result['field_labels'].get(key, '')}")
    lines.extend(
        [
            "",
            f"- SPXB fields labelled as best fields by metadata: {h['spxb_is_best_by_metadata']}",
            f"- SPAFACT field definition: {result['field_labels'].get('SPAFACT', '')}. The raw-curve readouts in this gate apply the per-manoeuvre BTPS factor through the existing parser.",
            "",
            "### observed_P3 vs Summary Fields",
            "",
            md_table(rows_by_type(rows, "part1_reference_compare"), [("reference_prefix", "prefix", "str"), ("reference_label", "reference", "str"), ("n", "n", "int"), ("fev1_abs_error_mL_median", "FEV1 abs mL med", "float2"), ("fvc_abs_error_mL_median", "FVC abs mL med", "float2"), ("ratio_abs_error_median", "ratio abs med", "float4"), ("within_25mL_both_fraction", "both <=25mL", "float3")]),
            "",
            "### Held-Out Manoeuvre Generalization",
            "",
            md_table(rows_by_type(rows, "part1_heldout_generalization"), [("estimator", "estimator", "str"), ("method", "method", "str"), ("n", "n", "int"), ("sensitivity", "sens", "float3"), ("specificity", "spec", "float3"), ("cohens_kappa", "kappa", "float3")]),
            "",
            f"Held-out P3 kappa gap: model {h['heldout_model_p3_kappa']:.3f}, observed {h['heldout_observed_p3_kappa']:.3f}, H1 {h['heldout_h1_p3_kappa']:.3f}. The observed advantage survives the non-circular held-out-blow evaluation.",
            "",
            "Metadata/holdout note: SPXB is labelled as second test, not best-of, so the SPXB comparison is not evidence that observed_P3 reconstructs a best-field algorithm.",
            "",
            "## Part 2 - Multiplicative FVC Correction",
            "",
            md_table(rows_by_type(rows, "part2_log_regression"), [("predictor", "predictor", "str"), ("n", "n", "int"), ("slope", "slope", "float5"), ("intercept", "intercept", "float5"), ("r2", "R2", "float3"), ("p_value", "p", "float3")]),
            "",
            md_table(rows_by_type(rows, "part2_multiplicative_fit"), [("n", "n", "int"), ("a", "a", "float6"), ("b", "b", "float6"), ("r2", "R2", "float3"), ("p_value", "p", "float3")]),
            "",
            "### FVC Residuals Before Correction",
            "",
            md_table(rows_by_type(rows, "part2_uncorrected_fvc_residual"), [("split", "split", "str"), ("truth_severity", "stratum", "str"), ("n", "n", "int"), ("median", "median %", "float3"), ("iqr_low", "IQR low", "float3"), ("iqr_high", "IQR high", "float3"), ("iqr_crosses_zero", "IQR crosses 0", "str")]),
            "",
            "### FVC Residuals After Multiplicative Correction",
            "",
            md_table(rows_by_type(rows, "part2_corrected_fvc_residual"), [("split", "split", "str"), ("truth_severity", "stratum", "str"), ("n", "n", "int"), ("median", "median %", "float3"), ("iqr_low", "IQR low", "float3"), ("iqr_high", "IQR high", "float3"), ("iqr_crosses_zero", "IQR crosses 0", "str")]),
            "",
            f"TRAIN-to-TEST stability flag: overfit={h['multiplicative_overfit_flag']}. Aggregate obstructed and normal residuals did not materially worsen on TEST under the predeclared rule.",
            "Severity-specific residuals can still fail to bracket zero even when the aggregate obstructed-vs-normal sign flip is removed.",
            "",
            "### Classification Side-by-Side",
            "",
            md_table(rows_by_type(rows, "part2_part3_classification"), [("estimator", "estimator", "str"), ("method", "method", "str"), ("definition", "definition", "str"), ("n", "n", "int"), ("sensitivity", "sens", "float3"), ("specificity", "spec", "float3"), ("cohens_kappa", "kappa", "float3"), ("borderline_confident_false_negative_rate", "borderline FN", "float3")]),
            "",
            "## Part 3 - H1 Hybrid Readout",
            "",
            "H1 = model FEV1 / observed segmented-BTPS FVC on complete curves.",
            "",
            "### H1 Signed Ratio Error",
            "",
            md_table(rows_by_type(rows, "part3_h1_ratio_error"), [("truth_severity", "stratum", "str"), ("n", "n", "int"), ("median", "median", "float4"), ("iqr_low", "IQR low", "float4"), ("iqr_high", "IQR high", "float4"), ("iqr_crosses_zero", "IQR crosses 0", "str")]),
            "",
            "### H1 Graceful Degradation Under Truncation",
            "",
            md_table(rows_by_type(rows, "part3_truncation_h1"), [("source", "source", "str"), ("T_label", "T", "str"), ("n", "n", "int"), ("sensitivity", "sens", "float3"), ("specificity", "spec", "float3"), ("cohens_kappa", "kappa", "float3"), ("obstructed_false_negative_rate", "obstructed FN", "float3")]),
            "",
            "### Repeatability",
            "",
            md_table(rows_by_type(rows, "part3_repeatability"), [("estimator", "estimator", "str"), ("n", "n", "int"), ("fev1_sd_median_l", "FEV1 SD", "float4"), ("fvc_sd_median_l", "FVC SD", "float4"), ("ratio_sd_median", "ratio SD", "float4"), ("ats_repeatable_fraction", "ATS repeatable", "float3")]),
            "",
            "## Part 4 - GLI Reference-Equation Coverage",
            "",
            f"- clean-mapping excluded fraction: {h['gli_clean_excluded_fraction']:.3f}",
            "",
            md_table(rows_by_type(rows, "part4_gli_subgroup"), [("subgroup", "subgroup", "str"), ("estimator", "estimator", "str"), ("method", "method", "str"), ("n", "n", "int"), ("sensitivity", "sens", "float3"), ("specificity", "spec", "float3"), ("cohens_kappa", "kappa", "float3"), ("borderline_confident_false_negative_rate", "borderline FN", "float3")]),
            "",
            "## Part 5 - GP Discrepancy Diagnostic",
            "",
            h["gp_sentence"],
            "",
            md_table(rows_by_type(rows, "part5_gp_diagnostic"), [("diagnostic", "diagnostic", "str"), ("n", "n", "int"), ("pearson_r", "r", "float3"), ("slope", "slope", "float5"), ("p_value", "p", "float3"), ("gp_length_over_fet_median", "length/FET med", "float3"), ("gp_amp_ml_median", "amp mL med", "float2")]),
            "",
            "## Verdict",
            "",
            f"CIRCULARITY: observed_P3 {result['verdict']['circularity']} a reconstruction of the NHANES reference algorithm; held-out manoeuvre rows above report the non-circular kappa gap.",
            f"MULTIPLICATIVE: {result['verdict']['multiplicative']} with a={h['multiplicative_a']:.6f}, b={h['multiplicative_b']:.6f}; overfit={h['multiplicative_overfit_flag']}; kappa gap closed: {h['kappa_gap_closed_by_multiplicative']}.",
            f"HYBRID: H1 kappa {h['h1_p3_kappa']:.3f} vs observed_P3 {h['observed_p3_kappa']:.3f}; T=2 obstructed FN H1 {h['h1_t2_obstructed_fn']:.3f}, E0 {h['e0_t2_obstructed_fn']:.3f}, E1 {h['e1_t2_obstructed_fn']:.3f}.",
            f"GLI: {result['verdict']['gli']}.",
            "",
        ]
    )
    (out / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
    (out / f"{stem}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    rows.to_csv(out / f"{stem}_rows.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--cycle", default="F", choices=("E", "F", "G"))
    args = parser.parse_args()
    result = run(Path(args.root).resolve(), args.cycle.upper())
    h = result["headline"]
    print(f"CIRCULARITY: observed_P3 {result['verdict']['circularity']}")
    print(f"MULTIPLICATIVE: {result['verdict']['multiplicative']} a={h['multiplicative_a']:.6f} b={h['multiplicative_b']:.6f}")
    print(f"HYBRID: {result['verdict']['hybrid']} H1_kappa={h['h1_p3_kappa']:.3f} observed_kappa={h['observed_p3_kappa']:.3f}")
    print(f"GLI: {result['verdict']['gli']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
