from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress

from .gate23_f import find_or_unzip_raw, read_raw
from .gate_clinical_classification_f import (
    confusion_binary,
    load_column_map,
    load_gli,
    load_truth,
    severity_category,
)
from .gate_four_part_fix_f import curve_variant_metrics
from .gate_policy_calibrated_flag_f import stratified_half_split
from .gate_reconciliation_audit_f import clean_json, md_table

OUTPUT_STEM = "gate_model_vs_observed_F"
CONVENTIONS = ("segmented_btps", "segmented_no_btps", "max_cumulative_btps", "final_record_btps")
SOURCE_LABELS = {"model": "MODEL", "observed": "OBSERVED"}


def finite_mask(*arrays: Any) -> np.ndarray:
    mask = None
    for arr in arrays:
        cur = np.isfinite(np.asarray(arr, dtype=float))
        mask = cur if mask is None else (mask & cur)
    return mask if mask is not None else np.array([], dtype=bool)


def pct_error(est: Any, truth: Any) -> np.ndarray:
    est_arr = np.asarray(est, dtype=float)
    truth_arr = np.asarray(truth, dtype=float)
    out = np.full(est_arr.shape, np.nan, dtype=float)
    ok = np.isfinite(est_arr) & np.isfinite(truth_arr) & (truth_arr != 0)
    out[ok] = 100.0 * (est_arr[ok] - truth_arr[ok]) / truth_arr[ok]
    return out


def signed_summary(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"n": 0, "median": np.nan, "iqr_low": np.nan, "iqr_high": np.nan, "iqr_crosses_zero": False}
    lo, hi = np.percentile(arr, [25, 75])
    return {
        "n": int(len(arr)),
        "median": float(np.median(arr)),
        "iqr_low": float(lo),
        "iqr_high": float(hi),
        "iqr_crosses_zero": bool(lo <= 0.0 <= hi),
    }


def linear_fit(y: pd.Series, x: pd.Series, label: str) -> dict[str, Any]:
    ok = finite_mask(x, y)
    if int(ok.sum()) < 3:
        return {"predictor": label, "n": int(ok.sum()), "slope": np.nan, "intercept": np.nan, "r2": np.nan, "p_value": np.nan}
    fit = linregress(np.asarray(x, dtype=float)[ok], np.asarray(y, dtype=float)[ok])
    return {
        "predictor": label,
        "n": int(ok.sum()),
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "r2": float(fit.rvalue**2),
        "p_value": float(fit.pvalue),
    }


def fixed_metrics(df: pd.DataFrame, ratio_col: str, *, threshold_col: str | None = None) -> dict[str, Any]:
    work = df.copy()
    if threshold_col is None:
        valid = work[ratio_col].notna() & work["fev1_fvc_ratio"].notna()
        work = work[valid].copy()
        truth = work["fev1_fvc_ratio"] < 0.70
        pred = work[ratio_col] < 0.70
    else:
        valid = work[ratio_col].notna() & work["fev1_fvc_ratio"].notna() & work[threshold_col].notna()
        work = work[valid].copy()
        truth = work["fev1_fvc_ratio"] < work[threshold_col]
        pred = work[ratio_col] < work[threshold_col]
    out = confusion_binary(truth, pred)
    obs = work[truth].copy()
    out["obstructed_signed_ratio_error_median"] = float(np.nanmedian(obs[ratio_col] - obs["fev1_fvc_ratio"])) if len(obs) else np.nan
    borderline = work[(work["fev1_fvc_ratio"] >= 0.60) & (work["fev1_fvc_ratio"] < 0.70)].copy()
    if len(borderline):
        threshold = 0.70 if threshold_col is None else borderline[threshold_col]
        out["borderline_confident_false_negative_rate"] = float((borderline[ratio_col] >= threshold).mean())
        out["borderline_n"] = int(len(borderline))
    else:
        out["borderline_confident_false_negative_rate"] = np.nan
        out["borderline_n"] = 0
    return out


def raw_variant_cache(raw: pd.DataFrame, raw_indices: np.ndarray) -> pd.DataFrame:
    raw_by_index = {int(idx): row for idx, row in raw.iterrows()}
    rows = []
    raw_indices = np.unique(np.asarray(raw_indices, dtype=int))
    total = len(raw_indices)
    for i, raw_index in enumerate(raw_indices, start=1):
        base = {"raw_index": int(raw_index), "variant_error": ""}
        try:
            base.update(curve_variant_metrics(raw_by_index[int(raw_index)]))
        except Exception as exc:
            base["variant_error"] = str(exc)
        rows.append(base)
        if i % 5000 == 0 or i == total:
            print(f"Model-vs-observed convention parse progress: {i}/{total} raw rows")
    return pd.DataFrame(rows)


def build_manoeuvres(root: Path, raw: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    path = root / "results" / "gate_reproducibility_volume_F_rows.csv"
    if not path.exists():
        raise RuntimeError(f"Missing volume reproducibility rows: {path}")
    rows = pd.read_csv(path, low_memory=False)
    rows = rows[rows["row_type"].eq("maneuver_fit")].copy()
    rows["volume_fit_success"] = rows["volume_fit_success"].map(
        {True: True, False: False, "True": True, "False": False, "true": True, "false": False}
    ).fillna(False)
    rows = rows[
        rows["volume_fit_success"]
        & rows["fit_error"].fillna("").eq("")
        & rows[["model_fev1_l", "model_fvc_l", "observed_fev1_l", "observed_fvc_l", "fit_tau_s"]].apply(np.isfinite).all(axis=1)
    ].copy()
    variants = raw_variant_cache(raw, rows["raw_index"].to_numpy(dtype=int))
    rows = rows.merge(variants, on="raw_index", how="left")
    rows = rows.merge(
        truth[["SEQN", "fev1_mL", "fvc_mL", "fev1_fvc_ratio"]],
        on="SEQN",
        how="left",
        suffixes=("", "_truth"),
    )
    if "fev1_fvc_ratio_truth" in rows:
        rows["fev1_fvc_ratio"] = rows["fev1_fvc_ratio"].fillna(rows["fev1_fvc_ratio_truth"])
    rows = rows[rows["fev1_fvc_ratio"].notna()].copy()
    rows["truth_severity"] = rows["fev1_fvc_ratio"].map(severity_category)

    # Observed convention columns are direct raw-curve readouts.
    rows["observed_segmented_btps_fev1_l"] = rows["observed_fev1_l"]
    rows["observed_segmented_btps_fvc_l"] = rows["observed_fvc_l"]
    rows["observed_segmented_btps_ratio"] = rows["observed_ratio"]
    rows["observed_segmented_no_btps_fev1_l"] = rows["segmented_no_btps_fev1_mL"] / 1000.0
    rows["observed_segmented_no_btps_fvc_l"] = rows["segmented_no_btps_fvc_mL"] / 1000.0
    rows["observed_segmented_no_btps_ratio"] = rows["segmented_no_btps_ratio"]
    rows["observed_max_cumulative_btps_fev1_l"] = rows["segmented_btps_fev1_mL"] / 1000.0
    rows["observed_max_cumulative_btps_fvc_l"] = rows["max_cumulative_btps_fvc_mL"] / 1000.0
    rows["observed_max_cumulative_btps_ratio"] = rows["observed_max_cumulative_btps_fev1_l"] / rows["observed_max_cumulative_btps_fvc_l"]
    rows["observed_final_record_btps_fev1_l"] = rows["segmented_btps_fev1_mL"] / 1000.0
    rows["observed_final_record_btps_fvc_l"] = rows["final_record_btps_fvc_mL"] / 1000.0
    rows["observed_final_record_btps_ratio"] = rows["observed_final_record_btps_fev1_l"] / rows["observed_final_record_btps_fvc_l"]

    # Model convention columns are readout swaps on the same deterministic LM fit.
    # For FVC-only convention changes, the observed endpoint delta is applied to
    # the fitted FVC; no OMBE parameter is refit.
    rows["model_segmented_btps_fev1_l"] = rows["model_fev1_l"]
    rows["model_segmented_btps_fvc_l"] = rows["model_fvc_l"]
    rows["model_segmented_btps_ratio"] = rows["model_ratio"]
    scale = rows["spafact"].where(rows["spafact"] > 0, np.nan)
    rows["model_segmented_no_btps_fev1_l"] = rows["model_fev1_l"] / scale
    rows["model_segmented_no_btps_fvc_l"] = rows["model_fvc_l"] / scale
    rows["model_segmented_no_btps_ratio"] = rows["model_segmented_no_btps_fev1_l"] / rows["model_segmented_no_btps_fvc_l"]
    for convention in ("max_cumulative_btps", "final_record_btps"):
        delta = rows[f"observed_{convention}_fvc_l"] - rows["observed_segmented_btps_fvc_l"]
        rows[f"model_{convention}_fev1_l"] = rows["model_fev1_l"]
        rows[f"model_{convention}_fvc_l"] = rows["model_fvc_l"] + delta
        rows.loc[rows[f"model_{convention}_fvc_l"] <= 0, f"model_{convention}_fvc_l"] = np.nan
        rows[f"model_{convention}_ratio"] = rows[f"model_{convention}_fev1_l"] / rows[f"model_{convention}_fvc_l"]
    return rows


def pool_participants(maneuvers: pd.DataFrame, convention: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    needed = [
        f"model_{convention}_fev1_l",
        f"model_{convention}_fvc_l",
        f"model_{convention}_ratio",
        f"observed_{convention}_fev1_l",
        f"observed_{convention}_fvc_l",
        f"observed_{convention}_ratio",
    ]
    m = maneuvers[maneuvers[needed].apply(np.isfinite).all(axis=1)].copy()
    counts = m.groupby("SEQN").size()
    keep = set(counts[counts >= 2].index.astype(float))
    m = m[m["SEQN"].astype(float).isin(keep)].copy()
    rows = []
    for seq, group in m.groupby("SEQN"):
        group = group.sort_values(["SPAMANU", "raw_index"]).copy()
        best = group.sort_values([f"observed_{convention}_fev1_l", "SPAMANU"], ascending=[False, True]).iloc[0]
        model_calls = group[f"model_{convention}_ratio"] < 0.70
        observed_calls = group[f"observed_{convention}_ratio"] < 0.70
        rows.append(
            {
                "SEQN": float(seq),
                "truth_severity": group["truth_severity"].iloc[0],
                "fev1_fvc_ratio": float(group["fev1_fvc_ratio"].iloc[0]),
                "n_maneuvers": int(len(group)),
                "fit_tau_s_median": float(group["fit_tau_s"].median()),
                "fet_s_max": float(group["duration_s"].max()),
                "p1_model_fev1_l": float(best[f"model_{convention}_fev1_l"]),
                "p1_model_fvc_l": float(best[f"model_{convention}_fvc_l"]),
                "p1_model_ratio": float(best[f"model_{convention}_ratio"]),
                "p1_observed_fev1_l": float(best[f"observed_{convention}_fev1_l"]),
                "p1_observed_fvc_l": float(best[f"observed_{convention}_fvc_l"]),
                "p1_observed_ratio": float(best[f"observed_{convention}_ratio"]),
                "p3_model_fev1_l": float(group[f"model_{convention}_fev1_l"].max()),
                "p3_model_fvc_l": float(group[f"model_{convention}_fvc_l"].max()),
                "p3_observed_fev1_l": float(group[f"observed_{convention}_fev1_l"].max()),
                "p3_observed_fvc_l": float(group[f"observed_{convention}_fvc_l"].max()),
                "model_spread": float(group[f"model_{convention}_ratio"].max() - group[f"model_{convention}_ratio"].min()),
                "observed_spread": float(group[f"observed_{convention}_ratio"].max() - group[f"observed_{convention}_ratio"].min()),
                "model_disagree": bool(model_calls.nunique() > 1),
                "observed_disagree": bool(observed_calls.nunique() > 1),
            }
        )
    participants = pd.DataFrame(rows)
    participants["p3_model_ratio"] = participants["p3_model_fev1_l"] / participants["p3_model_fvc_l"]
    participants["p3_observed_ratio"] = participants["p3_observed_fev1_l"] / participants["p3_observed_fvc_l"]
    return participants, m


def attach_split_and_gli(participants: pd.DataFrame, gli: pd.DataFrame) -> pd.DataFrame:
    split = stratified_half_split(participants, "truth_severity")
    if not gli.empty:
        split = split.merge(gli[["SEQN", "gli_lln_ratio"]], on="SEQN", how="left")
    else:
        split["gli_lln_ratio"] = np.nan
    return split


def classification_rows(split: pd.DataFrame, convention: str, source: str, method: str, ratio_col: str) -> list[dict[str, Any]]:
    rows = []
    test = split[split["split"].eq("test")].copy()
    for definition, threshold_col in [("fixed_0.70", None), ("GLI_LLN", "gli_lln_ratio")]:
        metrics = fixed_metrics(test, ratio_col, threshold_col=threshold_col)
        rows.append(
            {
                "row_type": "part2_classification",
                "convention": convention,
                "source": source,
                "method": method,
                "definition": definition,
                **metrics,
            }
        )
    return rows


def per_stratum_fvc_error_rows(split: pd.DataFrame, convention: str, method: str = "p3") -> list[dict[str, Any]]:
    rows = []
    test = split[split["split"].eq("test")].copy()
    for stratum, part in test.groupby("truth_severity", dropna=False):
        summary = signed_summary(pct_error(part[f"{method}_model_fvc_l"], part[f"{method}_observed_fvc_l"]))
        rows.append(
            {
                "row_type": "part2_model_vs_observed_fvc_by_stratum",
                "convention": convention,
                "method": method.upper(),
                "truth_severity": stratum,
                **summary,
            }
        )
    return rows


def fit_c1(train: pd.DataFrame, ratio_col: str) -> float:
    e = train[ratio_col] - train["fev1_fvc_ratio"]
    return float(np.nanmedian(e))


def repeatability_rows(maneuvers: pd.DataFrame, convention: str) -> list[dict[str, Any]]:
    rows = []
    for source in ("model", "observed"):
        part_rows = []
        for seq, group in maneuvers.groupby("SEQN"):
            if len(group) < 2:
                continue
            fev1 = group[f"{source}_{convention}_fev1_l"].dropna().sort_values(ascending=False).to_numpy(float)
            fvc = group[f"{source}_{convention}_fvc_l"].dropna().sort_values(ascending=False).to_numpy(float)
            ratio = group[f"{source}_{convention}_ratio"].dropna().to_numpy(float)
            if len(fev1) < 2 or len(fvc) < 2 or len(ratio) < 2:
                continue
            part_rows.append(
                {
                    "SEQN": float(seq),
                    "fev1_sd_l": float(np.std(fev1, ddof=1)),
                    "fvc_sd_l": float(np.std(fvc, ddof=1)),
                    "ratio_sd": float(np.std(ratio, ddof=1)),
                    "fev1_top2_diff_l": float(abs(fev1[0] - fev1[1])),
                    "fvc_top2_diff_l": float(abs(fvc[0] - fvc[1])),
                    "ats_repeatable": bool(abs(fev1[0] - fev1[1]) <= 0.150 and abs(fvc[0] - fvc[1]) <= 0.150),
                }
            )
        df = pd.DataFrame(part_rows)
        rows.append(
            {
                "row_type": "part4_repeatability",
                "convention": convention,
                "source": source,
                "n": int(len(df)),
                "fev1_sd_median_l": float(np.nanmedian(df["fev1_sd_l"])) if len(df) else np.nan,
                "fvc_sd_median_l": float(np.nanmedian(df["fvc_sd_l"])) if len(df) else np.nan,
                "ratio_sd_median": float(np.nanmedian(df["ratio_sd"])) if len(df) else np.nan,
                "ats_repeatable_fraction": float(df["ats_repeatable"].mean()) if len(df) else np.nan,
                "fev1_top2_diff_median_l": float(np.nanmedian(df["fev1_top2_diff_l"])) if len(df) else np.nan,
                "fvc_top2_diff_median_l": float(np.nanmedian(df["fvc_top2_diff_l"])) if len(df) else np.nan,
            }
        )
    return rows


def truncation_rows(root: Path, preferred: str) -> list[dict[str, Any]]:
    path = root / "results" / "gate_truncation_recovery_F_rows.csv"
    if not path.exists():
        return [{"row_type": "part4_truncation", "error": f"missing {path}"}]
    rows = pd.read_csv(path)
    out = []
    for estimator, source in [("E0_DIRECT", "OBSERVED_DIRECT"), ("E1_OMBE", "MODEL_OMBE")]:
        for t_label in ["2", "3", "4", "5"]:
            part = rows[(rows["estimator"].eq(estimator)) & (rows["T_label"].astype(str).eq(t_label))].copy()
            valid = part[part["fit_success"].astype(bool) & part["est_ratio"].notna()].copy()
            metrics = confusion_binary(valid["truth_obstructed_fixed"], valid["est_obstructed_fixed"]) if len(valid) else {}
            out.append(
                {
                    "row_type": "part4_truncation",
                    "preferred_convention": preferred,
                    "note": "Uses existing truncation gate's withheld-after-T scoring target; preferred complete-curve convention is reported for the surrounding classification gate, not refit into truncated targets.",
                    "source": source,
                    "estimator": estimator,
                    "T_label": t_label,
                    **metrics,
                    "obstructed_false_negative_rate": metrics.get("fn", np.nan) / (metrics.get("tp", 0) + metrics.get("fn", 0)) if metrics else np.nan,
                }
            )
    return out


def matched_flag_rows(split: pd.DataFrame, target_normal_burden: float = 0.12) -> list[dict[str, Any]]:
    rows = []
    train = split[split["split"].eq("train")].copy()
    test = split[split["split"].eq("test")].copy()
    normal_train = train[train["fev1_fvc_ratio"] >= 0.70].copy()
    for source in ("model", "observed"):
        spread_col = f"{source}_spread"
        threshold = float(np.nanquantile(normal_train[spread_col], 1.0 - target_normal_burden)) if len(normal_train) else np.nan
        work = test.copy()
        work["flag"] = work[spread_col] >= threshold
        for label, ratio_col in [("P3", f"p3_{source}_ratio")]:
            normal = work[work["fev1_fvc_ratio"] >= 0.70]
            borderline = work[work["truth_severity"].eq("mild_0.60_0.70")]
            confident_borderline = borderline[~borderline["flag"]]
            fn = float((confident_borderline[ratio_col] >= 0.70).mean()) if len(confident_borderline) else np.nan
            rows.append(
                {
                    "row_type": "part4_matched_flag_burden",
                    "source": source,
                    "method": label,
                    "target_train_normal_flag_burden": target_normal_burden,
                    "spread_threshold_train": threshold,
                    "test_normal_flag_burden": float(normal["flag"].mean()) if len(normal) else np.nan,
                    "test_borderline_flag_burden": float(borderline["flag"].mean()) if len(borderline) else np.nan,
                    "test_borderline_confident_fn": fn,
                    "test_borderline_confident_n": int(len(confident_borderline)),
                }
            )
    return rows


def gli_distance_rows(split: pd.DataFrame, ratio_col: str, c1: float) -> list[dict[str, Any]]:
    test = split[split["split"].eq("test")].copy()
    test = test[test["gli_lln_ratio"].notna()].copy()
    test["abs_lln_minus_070"] = (test["gli_lln_ratio"] - 0.70).abs()
    bins = [0.0, 0.02, 0.04, 0.08, np.inf]
    labels = ["0.00-0.02", "0.02-0.04", "0.04-0.08", ">=0.08"]
    test["lln_distance_bin"] = pd.cut(test["abs_lln_minus_070"], bins=bins, labels=labels, include_lowest=True, right=False)
    rows = []
    for correction_name, correction in [("no_c1", 0.0), ("train_c1", c1)]:
        pred_ratio = test[ratio_col] - correction
        truth_obs = test["fev1_fvc_ratio"] < test["gli_lln_ratio"]
        pred_obs = pred_ratio < test["gli_lln_ratio"]
        for bin_label, part in test.groupby("lln_distance_bin", dropna=False):
            idx = part.index
            obs = truth_obs.loc[idx]
            pred = pred_obs.loc[idx]
            obstructed = part[obs].copy()
            rows.append(
                {
                    "row_type": "part5_gli_distance_bin",
                    "correction": correction_name,
                    "lln_distance_bin": str(bin_label),
                    "n": int(len(part)),
                    "gli_obstructed_n": int(obs.sum()),
                    "gli_false_negative_rate": float(((obs) & (~pred)).sum() / obs.sum()) if obs.sum() else np.nan,
                    "gli_borderline_false_negative_rate": float((pred_ratio.loc[obstructed.index] >= obstructed["gli_lln_ratio"]).mean()) if len(obstructed) else np.nan,
                    "lln_distance_median": float(np.nanmedian(part["abs_lln_minus_070"])) if len(part) else np.nan,
                }
            )
    return rows


def run(root: Path) -> dict[str, Any]:
    start = time.perf_counter()
    column_map = load_column_map(root)
    truth = load_truth(root, column_map)
    gli = load_gli(root)
    raw_path, raw_note = find_or_unzip_raw(root)
    raw, raw_info = read_raw(raw_path)
    maneuvers = build_manoeuvres(root, raw, truth)

    row_records: list[dict[str, Any]] = []
    convention_results: dict[str, Any] = {}
    splits: dict[str, pd.DataFrame] = {}
    maneuver_by_convention: dict[str, pd.DataFrame] = {}

    for convention in CONVENTIONS:
        participants, m_conv = pool_participants(maneuvers, convention)
        split = attach_split_and_gli(participants, gli)
        splits[convention] = split
        maneuver_by_convention[convention] = m_conv
        part_rows: list[dict[str, Any]] = []
        for source in ("model", "observed"):
            for method, ratio_col in [("single_P1", f"p1_{source}_ratio"), ("pooled_P3", f"p3_{source}_ratio")]:
                part_rows.extend(classification_rows(split, convention, source, method, ratio_col))
        part_rows.extend(per_stratum_fvc_error_rows(split, convention, "p3"))
        row_records.extend(part_rows)
        convention_results[convention] = {
            "n_participants": int(len(split)),
            "test_counts": split[split["split"].eq("test")]["truth_severity"].value_counts().to_dict(),
            "classification": [r for r in part_rows if r["row_type"] == "part2_classification"],
            "fvc_error_by_stratum": [r for r in part_rows if r["row_type"] == "part2_model_vs_observed_fvc_by_stratum"],
        }

    current = splits["segmented_btps"]
    current_test = current[current["split"].eq("test")].copy()
    part1_rows = []
    for stratum, part in current_test.groupby("truth_severity", dropna=False):
        for metric in ("fvc", "fev1"):
            err = pct_error(part[f"p3_model_{metric}_l"], part[f"p3_observed_{metric}_l"])
            item = {
                "row_type": "part1_stratum_sign_flip",
                "truth_severity": stratum,
                "metric": metric.upper(),
                **signed_summary(err),
            }
            part1_rows.append(item)
    current_test["p3_model_minus_observed_ratio"] = current_test["p3_model_ratio"] - current_test["p3_observed_ratio"]
    reg_rows = [
        {"row_type": "part1_bias_regression", **linear_fit(current_test["p3_model_minus_observed_ratio"], current_test["p3_observed_ratio"], "observed_ratio_P3")},
        {"row_type": "part1_bias_regression", **linear_fit(current_test["p3_model_minus_observed_ratio"], current_test["fit_tau_s_median"], "tau_s_median")},
        {"row_type": "part1_bias_regression", **linear_fit(current_test["p3_model_minus_observed_ratio"], current_test["fet_s_max"], "FET_max")},
    ]
    row_records.extend(part1_rows)
    row_records.extend(reg_rows)
    best_predictor = max(reg_rows, key=lambda r: r.get("r2", -np.inf) if np.isfinite(r.get("r2", np.nan)) else -np.inf)

    # Preferred convention: require all four stratum FVC IQRs to cross zero; then maximize fixed P3 model kappa.
    convention_choice_rows = []
    for convention, result in convention_results.items():
        fixed_p3 = [
            r
            for r in result["classification"]
            if r["source"] == "model" and r["method"] == "pooled_P3" and r["definition"] == "fixed_0.70"
        ][0]
        fvc_rows = result["fvc_error_by_stratum"]
        all_iqr_cross = bool(fvc_rows and all(r["iqr_crosses_zero"] for r in fvc_rows))
        convention_choice_rows.append(
            {
                "row_type": "part2_convention_choice",
                "convention": convention,
                "model_p3_fixed_kappa": fixed_p3["cohens_kappa"],
                "model_p3_fixed_sensitivity": fixed_p3["sensitivity"],
                "model_p3_fixed_specificity": fixed_p3["specificity"],
                "model_p3_borderline_fn": fixed_p3["borderline_confident_false_negative_rate"],
                "all_stratum_fvc_iqrs_cross_zero": all_iqr_cross,
            }
        )
    row_records.extend(convention_choice_rows)
    eligible = [r for r in convention_choice_rows if r["all_stratum_fvc_iqrs_cross_zero"]]
    preferred_row = max(eligible if eligible else convention_choice_rows, key=lambda r: r["model_p3_fixed_kappa"])
    preferred = preferred_row["convention"]
    if not eligible:
        preferred_reason = "No convention had every per-stratum model-vs-observed FVC IQR crossing zero; selected highest TEST model P3 fixed-0.70 kappa."
    else:
        preferred_reason = "Selected highest TEST model P3 fixed-0.70 kappa among conventions whose per-stratum FVC IQRs all cross zero."

    preferred_split = splits[preferred]
    train = preferred_split[preferred_split["split"].eq("train")].copy()
    test = preferred_split[preferred_split["split"].eq("test")].copy()
    c1 = fit_c1(train, "p3_model_ratio")
    part3_rows = []
    for method, base_col in [("single_P1", "p1_model_ratio"), ("pooled_P3", "p3_model_ratio")]:
        for label, corr in [("no_c1", 0.0), ("train_c1", c1)]:
            work = test.copy()
            col = f"{base_col}_{label}"
            work[col] = work[base_col] - corr
            metrics = fixed_metrics(work, col)
            part3_rows.append({"row_type": "part3_c1_classification", "convention": preferred, "method": method, "correction": label, "c1_value": c1, **metrics})
    row_records.extend(part3_rows)
    c1_status = "RETRACTED" if abs(c1) < 0.003 else "RETAINED"

    trunc_rows = truncation_rows(root, preferred)
    repeat_rows = repeatability_rows(maneuver_by_convention[preferred], preferred)
    flag_rows = matched_flag_rows(preferred_split)
    row_records.extend(trunc_rows)
    row_records.extend(repeat_rows)
    row_records.extend(flag_rows)

    gli_rows = gli_distance_rows(preferred_split, "p3_model_ratio", c1)
    row_records.extend(gli_rows)
    gli_no_c1 = fixed_metrics(test, "p3_model_ratio", threshold_col="gli_lln_ratio")
    test_c1 = test.copy()
    test_c1["p3_model_ratio_c1"] = test_c1["p3_model_ratio"] - c1
    gli_with_c1 = fixed_metrics(test_c1, "p3_model_ratio_c1", threshold_col="gli_lln_ratio")
    row_records.append({"row_type": "part5_gli_preferred_classification", "correction": "no_c1", "convention": preferred, **gli_no_c1})
    row_records.append({"row_type": "part5_gli_preferred_classification", "correction": "train_c1", "convention": preferred, **gli_with_c1})

    pref_model_p3 = [
        r
        for r in convention_results[preferred]["classification"]
        if r["source"] == "model" and r["method"] == "pooled_P3" and r["definition"] == "fixed_0.70"
    ][0]
    pref_obs_p3 = [
        r
        for r in convention_results[preferred]["classification"]
        if r["source"] == "observed" and r["method"] == "pooled_P3" and r["definition"] == "fixed_0.70"
    ][0]
    gap = pref_obs_p3["cohens_kappa"] - pref_model_p3["cohens_kappa"]
    if gap <= 0.03:
        verdict1 = "CONVENTION-FIXES-GAP"
    elif pref_model_p3["cohens_kappa"] > 0.680 + 0.03:
        verdict1 = "CONVENTION-HELPS-PARTIALLY"
    else:
        verdict1 = "GENUINE-MODEL-DEFICIENCY"
    trunc_df = pd.DataFrame(trunc_rows)
    t2_model = trunc_df[(trunc_df["source"] == "MODEL_OMBE") & (trunc_df["T_label"] == "2")]
    t2_obs = trunc_df[(trunc_df["source"] == "OBSERVED_DIRECT") & (trunc_df["T_label"] == "2")]
    trunc_win = bool(
        len(t2_model)
        and len(t2_obs)
        and float(t2_model.iloc[0]["obstructed_false_negative_rate"]) < float(t2_obs.iloc[0]["obstructed_false_negative_rate"])
    )
    complete_sentence = (
        f"On complete curves, the model loses to pooled direct measurement under {preferred}: model P3 kappa {pref_model_p3['cohens_kappa']:.3f} vs observed P3 {pref_obs_p3['cohens_kappa']:.3f}."
        if pref_model_p3["cohens_kappa"] + 0.03 < pref_obs_p3["cohens_kappa"]
        else f"On complete curves, the model matches/beats pooled direct measurement under {preferred}: model P3 kappa {pref_model_p3['cohens_kappa']:.3f} vs observed P3 {pref_obs_p3['cohens_kappa']:.3f}."
    )
    trunc_sentence = (
        "On truncated curves, the model beats direct measurement at T=2 by obstructed false-negative rate."
        if trunc_win
        else "On truncated curves, the model does not beat direct measurement at T=2 by obstructed false-negative rate."
    )
    contribution = []
    if trunc_win:
        contribution.append("truncation recovery")
    rep_model = next((r for r in repeat_rows if r["source"] == "model"), {})
    rep_obs = next((r for r in repeat_rows if r["source"] == "observed"), {})
    if rep_model and rep_obs and rep_model.get("ats_repeatable_fraction", 0) > rep_obs.get("ats_repeatable_fraction", 0):
        contribution.append("ATS repeatability")
    if not contribution:
        contribution.append("none found")

    result = {
        "verdict": verdict1,
        "c1_status": c1_status,
        "model_contribution": contribution,
        "headline": {
            "preferred_convention": preferred,
            "preferred_reason": preferred_reason,
            "model_p3_fixed_kappa": pref_model_p3["cohens_kappa"],
            "observed_p3_fixed_kappa": pref_obs_p3["cohens_kappa"],
            "model_p3_borderline_fn": pref_model_p3["borderline_confident_false_negative_rate"],
            "observed_p3_borderline_fn": pref_obs_p3["borderline_confident_false_negative_rate"],
            "c1_value_train": c1,
            "complete_curve_sentence": complete_sentence,
            "truncated_curve_sentence": trunc_sentence,
        },
        "part1": {
            "stratum_error_rows": part1_rows,
            "regressions": reg_rows,
            "best_bias_predictor": best_predictor,
        },
        "part2": {
            "conventions": convention_results,
            "choice_rows": convention_choice_rows,
        },
        "part3": {
            "c1_value": c1,
            "c1_status": c1_status,
            "classification": part3_rows,
        },
        "part4": {
            "truncation": trunc_rows,
            "repeatability": repeat_rows,
            "matched_flag_burden": flag_rows,
            "complete_curve_sentence": complete_sentence,
            "truncated_curve_sentence": trunc_sentence,
        },
        "part5": {
            "gli_distance_bins": gli_rows,
            "gli_no_c1": gli_no_c1,
            "gli_with_c1": gli_with_c1,
        },
        "inputs": {
            "raw_source": str(raw_path),
            "raw_source_note": raw_note,
            "raw_loaded_rows": int(raw_info["loaded_rows"]),
            "maneuver_rows_used": int(len(maneuvers)),
            "wall_clock_s": float(time.perf_counter() - start),
            "readout_swap_note": "For max_cumulative_btps and final_record_btps, model FVC uses the same observed endpoint delta added to the deterministic LM segmented FVC. OMBE parameters are not refit.",
        },
    }
    write_outputs(root, result, pd.DataFrame(row_records))
    return result


def write_outputs(root: Path, result: dict[str, Any], rows: pd.DataFrame) -> None:
    out = root / "results"
    h = result["headline"]
    lines = [
        "# Gate - Model vs Observed Reconciliation, Cycle F",
        "",
        f"- VERDICT: {result['verdict']} (model P3 kappa {h['model_p3_fixed_kappa']:.3f} vs observed P3 {h['observed_p3_fixed_kappa']:.3f} under {h['preferred_convention']})",
        f"- c1: {result['c1_status']} (TRAIN fitted value {h['c1_value_train']:.6f})",
        f"- MODEL-CONTRIBUTION: {', '.join(result['model_contribution'])}",
        "- Deterministic LM only. No NUTS, SNPE, gamma, registration, or ML residual.",
        f"- Readout swap note: {result['inputs']['readout_swap_note']}",
        "",
        "## Part 1 - Stratum Sign Flip",
        "",
        md_table(result["part1"]["stratum_error_rows"], [("truth_severity", "stratum", "str"), ("metric", "metric", "str"), ("n", "n", "int"), ("median", "median %", "float3"), ("iqr_low", "IQR low", "float3"), ("iqr_high", "IQR high", "float3"), ("iqr_crosses_zero", "IQR crosses 0", "str")]),
        "",
        "### Bias Regressions",
        "",
        md_table(result["part1"]["regressions"], [("predictor", "predictor", "str"), ("n", "n", "int"), ("slope", "slope", "float5"), ("intercept", "intercept", "float5"), ("r2", "R2", "float3"), ("p_value", "p", "float3")]),
        "",
        f"Best bias predictor by R2: {result['part1']['best_bias_predictor']['predictor']}.",
        "",
        "## Part 2 - FVC Convention Sweep",
        "",
        f"Preferred convention: `{h['preferred_convention']}`. {h['preferred_reason']}",
        "",
        "### Convention Choice",
        "",
        md_table(result["part2"]["choice_rows"], [("convention", "convention", "str"), ("model_p3_fixed_kappa", "model P3 kappa", "float3"), ("model_p3_fixed_sensitivity", "sens", "float3"), ("model_p3_fixed_specificity", "spec", "float3"), ("model_p3_borderline_fn", "borderline FN", "float3"), ("all_stratum_fvc_iqrs_cross_zero", "all IQRs cross 0", "str")]),
        "",
        "### Classification by Convention",
        "",
        md_table(
            [r for c in result["part2"]["conventions"].values() for r in c["classification"]],
            [("convention", "convention", "str"), ("source", "source", "str"), ("method", "method", "str"), ("definition", "definition", "str"), ("n", "n", "int"), ("sensitivity", "sens", "float3"), ("specificity", "spec", "float3"), ("cohens_kappa", "kappa", "float3"), ("obstructed_signed_ratio_error_median", "signed err", "float4"), ("borderline_confident_false_negative_rate", "borderline FN", "float3")],
        ),
        "",
        "### P3 Model-vs-Observed FVC Error by Stratum",
        "",
        md_table(
            [r for c in result["part2"]["conventions"].values() for r in c["fvc_error_by_stratum"]],
            [("convention", "convention", "str"), ("truth_severity", "stratum", "str"), ("n", "n", "int"), ("median", "median %", "float3"), ("iqr_low", "IQR low", "float3"), ("iqr_high", "IQR high", "float3"), ("iqr_crosses_zero", "IQR crosses 0", "str")],
        ),
        "",
        "## Part 3 - c1 Retraction Check",
        "",
        f"- preferred convention c1 fitted on TRAIN: {h['c1_value_train']:.6f}",
        f"- c1 status: {result['c1_status']}",
        "",
        md_table(result["part3"]["classification"], [("method", "method", "str"), ("correction", "correction", "str"), ("n", "n", "int"), ("sensitivity", "sens", "float3"), ("specificity", "spec", "float3"), ("cohens_kappa", "kappa", "float3"), ("borderline_confident_false_negative_rate", "borderline FN", "float3")]),
        "",
        "## Part 4 - Where the Model Wins",
        "",
        h["complete_curve_sentence"],
        "",
        h["truncated_curve_sentence"],
        "",
        "### Truncation Recovery",
        "",
        md_table(result["part4"]["truncation"], [("source", "source", "str"), ("T_label", "T", "str"), ("n", "n", "int"), ("sensitivity", "sens", "float3"), ("specificity", "spec", "float3"), ("cohens_kappa", "kappa", "float3"), ("obstructed_false_negative_rate", "obstructed FN", "float3")]),
        "",
        "### Repeatability",
        "",
        md_table(result["part4"]["repeatability"], [("source", "source", "str"), ("n", "n", "int"), ("fev1_sd_median_l", "FEV1 SD", "float4"), ("fvc_sd_median_l", "FVC SD", "float4"), ("ratio_sd_median", "ratio SD", "float4"), ("ats_repeatable_fraction", "ATS repeatable", "float3")]),
        "",
        "### Matched Normal-Truth Flag Burden",
        "",
        md_table(result["part4"]["matched_flag_burden"], [("source", "source", "str"), ("method", "method", "str"), ("target_train_normal_flag_burden", "train target", "float3"), ("spread_threshold_train", "spread threshold", "float4"), ("test_normal_flag_burden", "test normal flag", "float3"), ("test_borderline_flag_burden", "test borderline flag", "float3"), ("test_borderline_confident_fn", "borderline confident FN", "float3")]),
        "",
        "## Part 5 - GLI Residual Diagnosis",
        "",
        "### GLI FN by |LLN - 0.70|",
        "",
        md_table(result["part5"]["gli_distance_bins"], [("correction", "correction", "str"), ("lln_distance_bin", "|LLN-0.70|", "str"), ("n", "n", "int"), ("gli_obstructed_n", "GLI obstructed n", "int"), ("gli_false_negative_rate", "FN", "float3"), ("lln_distance_median", "distance med", "float3")]),
        "",
        "### GLI Preferred Classification",
        "",
        md_table(
            [
                {"correction": "no_c1", **result["part5"]["gli_no_c1"]},
                {"correction": "train_c1", **result["part5"]["gli_with_c1"]},
            ],
            [("correction", "correction", "str"), ("n", "n", "int"), ("sensitivity", "sens", "float3"), ("specificity", "spec", "float3"), ("cohens_kappa", "kappa", "float3"), ("borderline_confident_false_negative_rate", "borderline FN", "float3")],
        ),
        "",
        "## Verdict",
        "",
        f"CONVENTION: {result['verdict']} - model P3 kappa {h['model_p3_fixed_kappa']:.3f}, observed P3 kappa {h['observed_p3_fixed_kappa']:.3f}.",
        f"c1: {result['c1_status']} - fitted value {h['c1_value_train']:.6f}.",
        f"MODEL-CONTRIBUTION: {', '.join(result['model_contribution'])}.",
        "",
    ]
    (out / f"{OUTPUT_STEM}.md").write_text("\n".join(lines), encoding="utf-8")
    (out / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    rows.to_csv(out / f"{OUTPUT_STEM}_rows.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    h = result["headline"]
    print(f"Convention verdict: {result['verdict']} model_kappa={h['model_p3_fixed_kappa']:.3f} observed_kappa={h['observed_p3_fixed_kappa']:.3f}")
    print(f"c1: {result['c1_status']} value={h['c1_value_train']:.6f}")
    print(f"MODEL-CONTRIBUTION: {', '.join(result['model_contribution'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
