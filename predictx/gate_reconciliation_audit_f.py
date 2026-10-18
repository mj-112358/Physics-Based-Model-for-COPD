from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .fit import np_ombe_flow
from .gate23_f import find_or_unzip_raw, read_raw
from .gate_clinical_classification_f import (
    RANDOM_SEED,
    build_usable_raw_candidates,
    decode_grade,
    extract_curve_for_fit,
    load_column_map,
    load_truth,
    severity_category,
)
from .gate_truncation_recovery_f import (
    TRUNCATION_TIMES,
    classification_summary,
    fit_ombe_truncated,
)

OUTPUT_STEM = "gate_reconciliation_audit_F"
MATERIAL_FN_DELTA = 0.02
BOUND_TOL = 1e-4
BOOTSTRAP_N = 2000


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
    if pd.isna(value) if not isinstance(value, (str, bytes, list, tuple, dict)) else False:
        return None
    return value


def q(values: pd.Series | np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return [float("nan"), float("nan")]
    return [float(x) for x in np.percentile(arr, [25, 75])]


def median_iqr(values: pd.Series | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return {
        "n": int(len(arr)),
        "median": float(np.median(arr)) if len(arr) else np.nan,
        "iqr": q(arr),
    }


def pct_error(est: pd.Series | np.ndarray, truth: pd.Series | np.ndarray) -> np.ndarray:
    est_arr = np.asarray(est, dtype=float)
    truth_arr = np.asarray(truth, dtype=float)
    out = np.full_like(est_arr, np.nan, dtype=float)
    ok = np.isfinite(est_arr) & np.isfinite(truth_arr) & (truth_arr != 0)
    out[ok] = 100.0 * (est_arr[ok] - truth_arr[ok]) / truth_arr[ok]
    return out


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.map({True: True, False: False, "True": True, "False": False, "true": True, "false": False}).fillna(False)


def fmt(value: Any, digits: int = 3) -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{digits}f}"


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str, str]]) -> str:
    lines = ["| " + " | ".join(label for _, label, _ in columns) + " |"]
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        cells = []
        for key, _, kind in columns:
            value = row.get(key)
            if kind == "int":
                cells.append(str(int(value)) if pd.notna(value) else "NA")
            elif kind.startswith("float"):
                digits = int(kind.replace("float", "") or "3")
                cells.append(fmt(value, digits))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_prior_outputs(root: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "clinical": root / "results" / "gate_clinical_classification_F_rows.csv",
        "truncation": root / "results" / "gate_truncation_recovery_F_rows.csv",
        "policy": root / "results" / "gate_policy_calibrated_flag_F_rows.csv",
        "gate3_fits": root / "results" / "gate3_F_curve_fits.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required prior result files: {missing}")
    return {name: pd.read_csv(path) for name, path in paths.items()}


def raw_index_map(raw: pd.DataFrame) -> dict[int, Any]:
    return {int(idx): row for idx, row in raw.iterrows()}


def curve_observed_metrics(raw_by_index: dict[int, Any], raw_index: int) -> dict[str, float]:
    curve = extract_curve_for_fit(raw_by_index[int(raw_index)])
    target = min(1.0, float(curve["vol_t"][-1]))
    nearest_idx = int(np.argmin(np.abs(curve["vol_t"] - target)))
    nearest_fev1_l = float(curve["vol_l"][nearest_idx])
    interp_fev1_l = float(np.interp(target, curve["vol_t"], curve["vol_l"]))
    return {
        "observed_fev1_l": interp_fev1_l,
        "observed_fvc_l": float(curve["parsed_fvc_l"]),
        "observed_ratio": float(curve["parsed_ratio"]),
        "nearest_sample_fev1_l": nearest_fev1_l,
        "nearest_minus_interp_fev1_mL": float(1000.0 * (nearest_fev1_l - interp_fev1_l)),
        "duration_s": float(curve["duration_s"]),
        "t0_s": float(curve["t0_s"]),
    }


def summarize_abs_pct(est_mL: pd.Series, truth_mL: pd.Series) -> dict[str, Any]:
    err = pct_error(est_mL, truth_mL)
    return {
        "signed": median_iqr(err),
        "absolute": median_iqr(np.abs(err)),
    }


def summarize_abs_ratio(est: pd.Series, truth: pd.Series) -> dict[str, Any]:
    diff = np.asarray(est, dtype=float) - np.asarray(truth, dtype=float)
    return {
        "signed": median_iqr(diff),
        "absolute": median_iqr(np.abs(diff)),
    }


def code_path_table() -> list[dict[str, str]]:
    return [
        {
            "gate": "clinical observed parser",
            "code_path": "gate_clinical_classification_f.extract_curve_for_fit",
            "t0": "ATS back-extrapolated time-zero from gate23_f.backextrap_t0",
            "fev1": "np.interp(min(1.0, vol_t[-1]), observed vol_t, observed vol_l)",
            "fvc": "observed segmented end-volume vol_l[-1]",
            "grid": "native 100 Hz samples after detected start/end, with inserted t=0 if needed",
        },
        {
            "gate": "clinical reported model",
            "code_path": "gate_clinical_classification_f.fit_selected_rows -> gate3_diagnosis.fit_ombe_volume_lm -> clinical_metrics_from_volume",
            "t0": "same observed ATS back-extrapolated t0 before fitting",
            "fev1": "np.interp(min(1.0, t[-1]), fitted OMBE volume, native observed grid)",
            "fvc": "fitted OMBE volume at observed end-of-test, not asymptote beyond the grid",
            "grid": "native observed volume grid",
        },
        {
            "gate": "truncation TRUE",
            "code_path": "gate_truncation_recovery_f.estimate_one_curve -> extract_curve_for_fit",
            "t0": "same ATS back-extrapolated time-zero",
            "fev1": "np.interp(1.0, full observed vol_t, observed vol_l)",
            "fvc": "observed segmented end-volume full_y[-1]",
            "grid": "native observed volume grid; truncated estimators see samples only up to T",
        },
    ]


def part_a_reconciliation(
    raw_by_index: dict[int, Any],
    clinical: pd.DataFrame,
    truncation: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    row_records: list[dict[str, Any]] = []
    clinical = clinical.copy()
    truncation = truncation.copy()
    clinical["raw_index"] = clinical["raw_index"].astype(int)
    truncation["raw_index"] = truncation["raw_index"].astype(int)

    trunc_full_e0 = truncation[(truncation["estimator"] == "E0_DIRECT") & (truncation["T_label"].astype(str) == "full")].copy()
    clinical_by_raw = clinical.drop_duplicates("raw_index").set_index("raw_index")
    trunc_by_raw = trunc_full_e0.drop_duplicates("raw_index").set_index("raw_index")
    common = sorted(set(clinical_by_raw.index).intersection(set(trunc_by_raw.index)))
    common_df = pd.DataFrame({"raw_index": common})
    if len(common_df) > 30:
        common_df = common_df.sample(n=30, random_state=RANDOM_SEED).sort_values("raw_index")

    per_curve_rows = []
    for raw_index in common_df["raw_index"].tolist():
        c = clinical_by_raw.loc[int(raw_index)]
        t = trunc_by_raw.loc[int(raw_index)]
        observed = curve_observed_metrics(raw_by_index, int(raw_index))
        out = {
            "row_type": "part_a_common_curve",
            "raw_index": int(raw_index),
            "SEQN": float(c["SEQN"]),
            "SPAMANU": float(c["SPAMANU"]),
            "nhanes_fev1_mL": float(c["fev1_mL"]),
            "nhanes_fvc_mL": float(c["fvc_mL"]),
            "nhanes_ratio": float(c["fev1_fvc_ratio"]),
            "observed_fev1_mL": observed["observed_fev1_l"] * 1000.0,
            "observed_fvc_mL": observed["observed_fvc_l"] * 1000.0,
            "observed_ratio": observed["observed_ratio"],
            "clinical_model_fev1_mL": float(c["model_fev1_mL"]),
            "clinical_model_fvc_mL": float(c["model_fvc_mL"]),
            "clinical_model_ratio": float(c["model_ratio"]),
            "trunc_true_fev1_mL": float(t["true_fev1_l"]) * 1000.0,
            "trunc_true_fvc_mL": float(t["true_fvc_l"]) * 1000.0,
            "trunc_true_ratio": float(t["true_ratio"]),
            "clinical_model_minus_observed_fev1_mL": float(c["model_fev1_mL"] - observed["observed_fev1_l"] * 1000.0),
            "clinical_model_minus_observed_fvc_mL": float(c["model_fvc_mL"] - observed["observed_fvc_l"] * 1000.0),
            "clinical_model_minus_observed_ratio": float(c["model_ratio"] - observed["observed_ratio"]),
            "trunc_true_minus_observed_fev1_mL": float(t["true_fev1_l"] * 1000.0 - observed["observed_fev1_l"] * 1000.0),
            "trunc_true_minus_observed_fvc_mL": float(t["true_fvc_l"] * 1000.0 - observed["observed_fvc_l"] * 1000.0),
            "trunc_true_minus_observed_ratio": float(t["true_ratio"] - observed["observed_ratio"]),
            "nearest_minus_interp_fev1_mL": observed["nearest_minus_interp_fev1_mL"],
        }
        per_curve_rows.append(out)
        row_records.append(out)

    clinical_obstructed = clinical[clinical["sample_group"] == "obstructed"].copy()
    clinical_obstructed = clinical_obstructed.dropna(
        subset=["fev1_mL", "fvc_mL", "fev1_fvc_ratio", "model_fev1_mL", "model_fvc_mL", "model_ratio", "parsed_fev1_l", "parsed_fvc_l", "parsed_ratio"]
    )

    trunc_cross = trunc_full_e0.dropna(subset=["nhanes_fev1_mL", "nhanes_fvc_mL", "nhanes_ratio", "true_fev1_l", "true_fvc_l", "true_ratio"]).copy()
    same_clinical = clinical_obstructed.copy()
    shared_seqn = set(same_clinical["SEQN"].astype(float))
    trunc_same_seqn = trunc_cross[trunc_cross["SEQN"].astype(float).isin(shared_seqn)].copy()

    attribution = {
        "clinical_model_vs_nhanes_obstructed_fev1_abs_pct": summarize_abs_pct(clinical_obstructed["model_fev1_mL"], clinical_obstructed["fev1_mL"]),
        "clinical_observed_vs_nhanes_obstructed_fev1_abs_pct": summarize_abs_pct(clinical_obstructed["parsed_fev1_l"] * 1000.0, clinical_obstructed["fev1_mL"]),
        "clinical_model_vs_nhanes_obstructed_fvc_abs_pct": summarize_abs_pct(clinical_obstructed["model_fvc_mL"], clinical_obstructed["fvc_mL"]),
        "clinical_observed_vs_nhanes_obstructed_fvc_abs_pct": summarize_abs_pct(clinical_obstructed["parsed_fvc_l"] * 1000.0, clinical_obstructed["fvc_mL"]),
        "clinical_model_vs_nhanes_obstructed_ratio_abs": summarize_abs_ratio(clinical_obstructed["model_ratio"], clinical_obstructed["fev1_fvc_ratio"]),
        "clinical_observed_vs_nhanes_obstructed_ratio_abs": summarize_abs_ratio(clinical_obstructed["parsed_ratio"], clinical_obstructed["fev1_fvc_ratio"]),
        "trunc_observed_vs_nhanes_sample_fev1_abs_pct": summarize_abs_pct(trunc_cross["true_fev1_l"] * 1000.0, trunc_cross["nhanes_fev1_mL"]),
        "trunc_observed_vs_nhanes_sample_fvc_abs_pct": summarize_abs_pct(trunc_cross["true_fvc_l"] * 1000.0, trunc_cross["nhanes_fvc_mL"]),
        "trunc_observed_vs_nhanes_sample_ratio_abs": summarize_abs_ratio(trunc_cross["true_ratio"], trunc_cross["nhanes_ratio"]),
        "trunc_observed_vs_nhanes_clinical_overlap_fev1_abs_pct": summarize_abs_pct(trunc_same_seqn["true_fev1_l"] * 1000.0, trunc_same_seqn["nhanes_fev1_mL"]) if len(trunc_same_seqn) else {},
        "nearest_vs_interpolation_fev1_mL_common30": median_iqr([r["nearest_minus_interp_fev1_mL"] for r in per_curve_rows]),
    }

    # Recompute the truncation TRUE fields from the observed ATS definition for every sampled raw curve.
    recomputed_rows = []
    for raw_index in sorted(trunc_cross["raw_index"].unique()):
        observed = curve_observed_metrics(raw_by_index, int(raw_index))
        recomputed_rows.append(
            {
                "raw_index": int(raw_index),
                "recalc_true_fev1_l": observed["observed_fev1_l"],
                "recalc_true_fvc_l": observed["observed_fvc_l"],
                "recalc_true_ratio": observed["observed_ratio"],
            }
        )
    recalc = pd.DataFrame(recomputed_rows)
    corrected = truncation.merge(recalc, on="raw_index", how="left")
    corrected["old_truth_obstructed_fixed"] = bool_series(corrected["truth_obstructed_fixed"])
    corrected["truth_obstructed_fixed"] = corrected["recalc_true_ratio"] < 0.70
    corrected["true_fev1_l"] = corrected["recalc_true_fev1_l"]
    corrected["true_fvc_l"] = corrected["recalc_true_fvc_l"]
    corrected["true_ratio"] = corrected["recalc_true_ratio"]

    old_cls = pd.DataFrame(classification_summary(truncation))
    new_cls = pd.DataFrame(classification_summary(corrected))
    old_key = old_cls.set_index(["estimator", "T_label"]).to_dict("index")
    new_key = new_cls.set_index(["estimator", "T_label"]).to_dict("index")
    headline_rows = []
    halt = False
    for estimator, t_label in [("E0_DIRECT", "2"), ("E1_OMBE", "2")]:
        old_fn = float(old_key[(estimator, t_label)]["false_negative_rate_obstructed"])
        new_fn = float(new_key[(estimator, t_label)]["false_negative_rate_obstructed"])
        delta = new_fn - old_fn
        if abs(delta) > MATERIAL_FN_DELTA:
            halt = True
        headline_rows.append(
            {
                "row_type": "part_a_headline_recalc",
                "estimator": estimator,
                "T_label": t_label,
                "old_false_negative_rate_obstructed": old_fn,
                "recomputed_false_negative_rate_obstructed": new_fn,
                "delta": delta,
                "material_delta_threshold": MATERIAL_FN_DELTA,
            }
        )
    row_records.extend(headline_rows)

    recalc_diff = corrected[corrected["T_label"].astype(str) == "full"].copy()
    recalc_diff["true_fev1_l_delta"] = recalc_diff["recalc_true_fev1_l"] - recalc_diff["true_fev1_l"]
    recalc_diff["true_fvc_l_delta"] = recalc_diff["recalc_true_fvc_l"] - recalc_diff["true_fvc_l"]
    recalc_diff["true_ratio_delta"] = recalc_diff["recalc_true_ratio"] - recalc_diff["true_ratio"]

    clinical_model_fev1 = attribution["clinical_model_vs_nhanes_obstructed_fev1_abs_pct"]["absolute"]["median"]
    clinical_observed_fev1 = attribution["clinical_observed_vs_nhanes_obstructed_fev1_abs_pct"]["absolute"]["median"]
    trunc_fev1 = attribution["trunc_observed_vs_nhanes_sample_fev1_abs_pct"]["absolute"]["median"]
    model_gap = abs(clinical_model_fev1 - clinical_observed_fev1)
    sample_gap = abs(clinical_observed_fev1 - trunc_fev1)
    if model_gap >= sample_gap:
        dominant = "definition: clinical headline is fitted-model vs NHANES, while truncation TRUE is observed-curve vs NHANES"
    else:
        dominant = "sample: truncation sample is grade-A complete-plateau and stratum-capped, not the clinical obstructed sample"

    result = {
        "verdict": "TRUTH-CORRECTED" if halt else "TRUTH-CONFIRMED",
        "halt": halt,
        "material_false_negative_delta_threshold": MATERIAL_FN_DELTA,
        "code_paths": code_path_table(),
        "common_curve_count_total": int(len(common)),
        "common_curve_count_reported": int(len(per_curve_rows)),
        "attribution": attribution,
        "dominant_gap_attribution": dominant,
        "clinical_obstructed_n": int(len(clinical_obstructed)),
        "truncation_crosscheck_sample_n": int(len(trunc_cross)),
        "truncation_overlap_with_clinical_obstructed_n": int(len(trunc_same_seqn)),
        "headline_recalc": headline_rows,
        "max_recomputed_truth_abs_delta": {
            "fev1_l": float(np.nanmax(np.abs(recalc_diff["true_fev1_l_delta"]))) if len(recalc_diff) else np.nan,
            "fvc_l": float(np.nanmax(np.abs(recalc_diff["true_fvc_l_delta"]))) if len(recalc_diff) else np.nan,
            "ratio": float(np.nanmax(np.abs(recalc_diff["true_ratio_delta"]))) if len(recalc_diff) else np.nan,
        },
    }
    return result, row_records, halt


def bootstrap_fn_rates(truncation: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(RANDOM_SEED)
    e1 = truncation[(truncation["estimator"] == "E1_OMBE") & bool_series(truncation["truth_obstructed_fixed"]) & bool_series(truncation["fit_success"])].copy()
    e1["T_label"] = e1["T_label"].astype(str)
    e1["fn"] = ~bool_series(e1["est_obstructed_fixed"])
    participants = sorted(e1["SEQN"].dropna().astype(float).unique())
    rows = []
    for t_label in ["2", "3", "4", "5"]:
        part = e1[e1["T_label"] == t_label].drop_duplicates("SEQN")
        by_seqn = part.set_index("SEQN")["fn"].astype(float)
        values = by_seqn.reindex(participants).dropna().to_numpy(float)
        boots = []
        for _ in range(BOOTSTRAP_N):
            sample = rng.choice(values, size=len(values), replace=True)
            boots.append(float(np.mean(sample)))
        rows.append(
            {
                "row_type": "part_b_bootstrap_fn",
                "T_label": t_label,
                "n_obstructed": int(len(values)),
                "fn_rate": float(np.mean(values)),
                "bootstrap_ci_low": float(np.percentile(boots, 2.5)),
                "bootstrap_ci_high": float(np.percentile(boots, 97.5)),
                "bootstrap_n": BOOTSTRAP_N,
            }
        )

    pivot = e1[e1["T_label"].isin(["2", "3"])].pivot_table(index="SEQN", columns="T_label", values="fn", aggfunc="first").dropna()
    diff_values = pivot["3"].astype(float).to_numpy() - pivot["2"].astype(float).to_numpy()
    diff_boots = []
    for _ in range(BOOTSTRAP_N):
        sample = rng.choice(diff_values, size=len(diff_values), replace=True)
        diff_boots.append(float(np.mean(sample)))
    diff = {
        "n_paired": int(len(diff_values)),
        "mean_difference_T3_minus_T2": float(np.mean(diff_values)),
        "ci_low": float(np.percentile(diff_boots, 2.5)),
        "ci_high": float(np.percentile(diff_boots, 97.5)),
        "distinguishable_from_zero": bool(np.percentile(diff_boots, 2.5) > 0 or np.percentile(diff_boots, 97.5) < 0),
    }
    rows.append({"row_type": "part_b_bootstrap_pairdiff", **diff})
    return rows, diff


def refit_bound_counts(raw_by_index: dict[int, Any], truncation: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample = truncation.drop_duplicates("raw_index")[["SEQN", "raw_index", "SPAMANU", "truth_severity"]].copy()
    param_names = ["tau_f", "tau_s", "alpha", "fvc_l", "tau_o"]
    lo = np.array([0.15, 1.00, 0.25, 2.00, 0.03], dtype=float)
    hi = np.array([0.80, 5.00, 0.85, 6.00, 0.40], dtype=float)
    fit_rows: list[dict[str, Any]] = []
    total = len(sample) * len(TRUNCATION_TIMES)
    done = 0
    for row in sample.itertuples(index=False):
        try:
            curve = extract_curve_for_fit(raw_by_index[int(row.raw_index)])
            full_t = np.asarray(curve["vol_t"], dtype=float)
            full_y = np.asarray(curve["vol_l"], dtype=float)
        except Exception as exc:
            for t_s in TRUNCATION_TIMES:
                fit_rows.append(
                    {
                        "row_type": "part_b_bound_fit",
                        "SEQN": float(row.SEQN),
                        "raw_index": int(row.raw_index),
                        "SPAMANU": float(row.SPAMANU),
                        "truth_severity": row.truth_severity,
                        "T_label": f"{t_s:g}",
                        "fit_success": False,
                        "fit_failure": str(exc),
                    }
                )
            continue
        for t_s in TRUNCATION_TIMES:
            done += 1
            base = {
                "row_type": "part_b_bound_fit",
                "SEQN": float(row.SEQN),
                "raw_index": int(row.raw_index),
                "SPAMANU": float(row.SPAMANU),
                "truth_severity": row.truth_severity,
                "T_label": f"{t_s:g}",
            }
            try:
                fit = fit_ombe_truncated(full_t, full_y, float(t_s))
                params = np.asarray(fit.get("params", np.repeat(np.nan, len(param_names))), dtype=float)
                base["fit_success"] = bool(fit.get("success", False))
                base["fit_failure"] = str(fit.get("failure", ""))
                for name, value, lower, upper in zip(param_names, params, lo, hi):
                    base[f"{name}"] = float(value)
                    base[f"{name}_low_bound"] = bool(np.isfinite(value) and value <= lower + BOUND_TOL)
                    base[f"{name}_high_bound"] = bool(np.isfinite(value) and value >= upper - BOUND_TOL)
            except Exception as exc:
                base["fit_success"] = False
                base["fit_failure"] = str(exc)
            fit_rows.append(base)
            if done % 250 == 0 or done == total:
                print(f"Reconciliation audit bound-fit progress: {done}/{total}")

    fit_df = pd.DataFrame(fit_rows)
    summary_rows = []
    for t_label, group in fit_df.groupby("T_label", dropna=False):
        fit_success_n = int(group["fit_success"].fillna(False).sum()) if "fit_success" in group else 0
        for name in param_names:
            low_col = f"{name}_low_bound"
            high_col = f"{name}_high_bound"
            low_n = int(group.get(low_col, pd.Series(False, index=group.index)).fillna(False).sum())
            high_n = int(group.get(high_col, pd.Series(False, index=group.index)).fillna(False).sum())
            either_n = int((group.get(low_col, pd.Series(False, index=group.index)).fillna(False) | group.get(high_col, pd.Series(False, index=group.index)).fillna(False)).sum())
            summary_rows.append(
                {
                    "row_type": "part_b_bound_summary",
                    "T_label": str(t_label),
                    "parameter": name,
                    "fit_success_n": fit_success_n,
                    "low_bound_n": low_n,
                    "high_bound_n": high_n,
                    "either_bound_n": either_n,
                    "either_bound_fraction": float(either_n / len(group)) if len(group) else np.nan,
                }
            )
    return fit_rows, summary_rows


def part_b_nonmonotonicity(raw_by_index: dict[int, Any], truncation: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row_records: list[dict[str, Any]] = []
    e1_obs = truncation[(truncation["estimator"] == "E1_OMBE") & bool_series(truncation["truth_obstructed_fixed"])].copy()
    e1_obs["T_label"] = e1_obs["T_label"].astype(str)
    stratum_counts = (
        e1_obs[e1_obs["T_label"] == "2"]
        .groupby("truth_severity", dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("truth_severity")
    )
    stratum_rows = [{"row_type": "part_b_obstructed_stratum_count", **row._asdict()} for row in stratum_counts.itertuples(index=False)]
    row_records.extend(stratum_rows)

    boot_rows, diff = bootstrap_fn_rates(truncation)
    row_records.extend(boot_rows)

    bound_fit_rows, bound_summary = refit_bound_counts(raw_by_index, truncation)
    row_records.extend(bound_fit_rows)
    row_records.extend(bound_summary)

    trend_rows = []
    for t_label in ["2", "3", "4", "5", "full"]:
        part = e1_obs[e1_obs["T_label"] == t_label].copy()
        trend_rows.append(
            {
                "row_type": "part_b_fvc_signed_error_trend",
                "T_label": t_label,
                "n": int(len(part)),
                "fvc_signed_pct_median": float(np.nanmedian(part["fvc_signed_pct_error"])) if len(part) else np.nan,
                "fvc_signed_pct_iqr": q(part["fvc_signed_pct_error"]),
                "fvc_abs_pct_median": float(np.nanmedian(part["fvc_abs_pct_error"])) if len(part) else np.nan,
            }
        )
    row_records.extend(trend_rows)

    bound_df = pd.DataFrame(bound_summary)
    bound_limited = False
    bound_reason = "No tau_s/fvc_l bound-pile pattern meeting the predeclared audit rule."
    for param in ["tau_s", "fvc_l"]:
        t2 = bound_df[(bound_df["parameter"] == param) & (bound_df["T_label"] == "2")]
        t3 = bound_df[(bound_df["parameter"] == param) & (bound_df["T_label"] == "3")]
        if len(t2) and len(t3):
            f2 = float(t2.iloc[0]["either_bound_fraction"])
            f3 = float(t3.iloc[0]["either_bound_fraction"])
            if f2 >= 0.25 and (f2 - f3) >= 0.15:
                bound_limited = True
                bound_reason = f"{param} bound fraction is {f2:.3f} at T=2 and {f3:.3f} at T=3."

    trend_vals = [row["fvc_abs_pct_median"] for row in trend_rows if row["T_label"] in ["2", "3", "4", "5"]]
    fvc_abs_strictly_shrinks = bool(all(trend_vals[i + 1] <= trend_vals[i] for i in range(len(trend_vals) - 1)))
    if not diff["distinguishable_from_zero"]:
        verdict = "NOISE"
        reason = "Paired bootstrap CI for E1 T=3 minus T=2 false-negative rate includes zero."
    elif bound_limited:
        verdict = "BOUND-LIMITED"
        reason = bound_reason
    else:
        verdict = "REAL-EFFECT"
        reason = "T=3 worse-than-T=2 false-negative rate is distinguishable and not explained by the bound-count rule."

    result = {
        "verdict": verdict,
        "reason": reason,
        "obstructed_stratum_counts": stratum_rows,
        "bootstrap_rows": boot_rows,
        "paired_difference_T3_minus_T2": diff,
        "bound_summary": bound_summary,
        "fvc_signed_error_trend": trend_rows,
        "fvc_abs_error_strictly_shrinks_T2_to_T5": fvc_abs_strictly_shrinks,
        "bound_limited": bound_limited,
        "bound_reason": bound_reason,
    }
    return result, row_records


def classify_drop(raw: pd.DataFrame, seqn: float) -> dict[str, Any]:
    rows = raw[raw["SEQN"].astype(float) == float(seqn)].copy()
    pre = rows[rows["SPATTYPE"].map(decode_grade) == "Pre"].copy()
    acc_ab = pre[(pre["SPAACC"].map(decode_grade) == "Y") & (pre["SPAQEFF"].map(decode_grade).isin(["A", "B"]))].copy()
    best_grade = ""
    if len(pre):
        grades = [decode_grade(x) for x in pre["SPAQEFF"].tolist()]
        order = {"A": 1, "B": 2, "C": 3, "D": 4}
        best_grade = sorted(grades, key=lambda g: order.get(g, 99))[0]
    base = {
        "row_type": "part_c_clinical_drop_taxonomy",
        "SEQN": float(seqn),
        "raw_pre_rows": int(len(pre)),
        "acceptable_ab_pre_rows": int(len(acc_ab)),
        "best_pre_spaqeff": best_grade,
    }
    if len(acc_ab) == 0:
        base.update(
            {
                "failure_class": "FIT-QUALITY",
                "failure_detail": "no Pre curve with SPAACC == Y and SPAQEFF in {A,B}; curve quality gate failure",
                "parse_bug_fixed": False,
                "became_usable": False,
            }
        )
        return base
    parse_ok = 0
    parse_errors = []
    for row in acc_ab.itertuples(index=True):
        try:
            extract_curve_for_fit(row)
            parse_ok += 1
        except Exception as exc:
            parse_errors.append(str(exc))
    if parse_ok == 0:
        base.update(
            {
                "failure_class": "PARSE",
                "failure_detail": "; ".join(sorted(set(parse_errors)))[:300],
                "parse_bug_fixed": False,
                "became_usable": False,
            }
        )
        return base
    base.update(
        {
            "failure_class": "FIT-NONCONVERGENCE",
            "failure_detail": "acceptable A/B raw curve exists and parses; absent from clinical selected set for non-parse reason",
            "parse_bug_fixed": False,
            "became_usable": False,
        }
    )
    return base


def policy_outcome(row: pd.Series) -> str:
    ratio = row.get("corrected_ratio_c1", np.nan)
    if np.isfinite(float(ratio)) and float(ratio) < 0.70:
        return "OBSTRUCTED"
    if bool(row.get("truncated_ats", False)):
        return "FLAG"
    return "NORMAL"


def part_c_coverage(raw: pd.DataFrame, truth: pd.DataFrame, clinical: pd.DataFrame, policy: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row_records: list[dict[str, Any]] = []
    tier2 = truth[
        truth["fev1_fvc_ratio"].notna()
        & truth["effort_quality"].isin(["A", "B"])
        & truth["fev1_quality"].isin(["A", "B"])
    ].copy()
    modsev_truth = tier2[tier2["fev1_fvc_ratio"] < 0.60].copy()
    clinical_obstructed_seqn = set(clinical[clinical["sample_group"] == "obstructed"]["SEQN"].astype(float))
    clinical_drops = modsev_truth[~modsev_truth["SEQN"].astype(float).isin(clinical_obstructed_seqn)].copy()
    drop_rows = []
    for seqn in sorted(clinical_drops["SEQN"].astype(float).tolist()):
        item = classify_drop(raw, seqn)
        item["nhanes_ratio"] = float(clinical_drops.loc[clinical_drops["SEQN"].astype(float) == seqn, "fev1_fvc_ratio"].iloc[0])
        item["truth_severity"] = severity_category(item["nhanes_ratio"])
        drop_rows.append(item)
    row_records.extend(drop_rows)

    all_seqn = set(modsev_truth["SEQN"].astype(float))
    policy_modsev_all = policy[policy["truth_severity"].isin(["severe", "moderate"])].copy()
    policy_modsev_all = policy_modsev_all.drop_duplicates("SEQN").copy()
    policy_extra_seqn = set(policy_modsev_all["SEQN"].astype(float)) - all_seqn
    policy_modsev = policy_modsev_all[policy_modsev_all["SEQN"].astype(float).isin(all_seqn)].copy()
    policy_modsev["full_denominator_outcome"] = policy_modsev.apply(policy_outcome, axis=1)
    for row in policy_modsev.itertuples(index=False):
        row_records.append(
            {
                "row_type": "part_c_policy_usable_obstructed",
                "SEQN": float(row.SEQN),
                "truth_severity": row.truth_severity,
                "corrected_ratio_c1": float(row.corrected_ratio_c1),
                "truncated_ats": bool(row.truncated_ats),
                "full_denominator_outcome": row.full_denominator_outcome,
            }
        )

    policy_seqn = set(policy_modsev["SEQN"].astype(float))
    policy_missing_seqn = sorted(all_seqn - policy_seqn)
    for seqn in policy_missing_seqn:
        row_records.append(
            {
                "row_type": "part_c_policy_missing_obstructed_flag",
                "SEQN": float(seqn),
                "truth_severity": severity_category(float(modsev_truth.loc[modsev_truth["SEQN"].astype(float) == seqn, "fev1_fvc_ratio"].iloc[0])),
                "full_denominator_outcome": "FLAG",
                "reason": "not present in gate_policy_calibrated_flag_F rows; counted as repeat/FLAG",
            }
        )

    outcome_counts = policy_modsev["full_denominator_outcome"].value_counts().to_dict()
    flag_missing_n = len(policy_missing_seqn)
    flag_n = int(outcome_counts.get("FLAG", 0) + flag_missing_n)
    obstructed_n = int(outcome_counts.get("OBSTRUCTED", 0))
    normal_n = int(outcome_counts.get("NORMAL", 0))
    denom = int(len(modsev_truth))
    confident_n = obstructed_n + normal_n
    result = {
        "clinical_drop_denominator_n": int(len(modsev_truth)),
        "clinical_drop_n": int(len(clinical_drops)),
        "clinical_drop_taxonomy_counts": pd.DataFrame(drop_rows)["failure_class"].value_counts().to_dict() if drop_rows else {},
        "parse_bug_fixed_n": int(sum(bool(r.get("parse_bug_fixed")) for r in drop_rows)),
        "policy_available_modsev_n": int(len(policy_modsev)),
        "policy_extra_modsev_outside_denominator_n": int(len(policy_extra_seqn)),
        "policy_missing_modsev_counted_flag_n": int(flag_missing_n),
        "policy_denominator_all_modsev_n": denom,
        "policy_outcome_counts_full_denominator": {
            "OBSTRUCTED": obstructed_n,
            "NORMAL": normal_n,
            "FLAG": flag_n,
        },
        "confident_call_rate": float(confident_n / denom) if denom else np.nan,
        "flag_rate": float(flag_n / denom) if denom else np.nan,
        "residual_confident_false_negative_rate": float(normal_n / denom) if denom else np.nan,
    }
    result["coverage_sentence"] = (
        f"The system produces a confident call for {100.0 * result['confident_call_rate']:.1f}% "
        f"of severe+moderate obstructed participants and recommends repeat/FLAG for "
        f"{100.0 * result['flag_rate']:.1f}%, with residual confident false-negative rate "
        f"{100.0 * result['residual_confident_false_negative_rate']:.1f}% over the full severe+moderate denominator."
    )
    return result, row_records


def volume_from_flow_params(params: np.ndarray, duration_s: float) -> tuple[float, float, float]:
    if not np.isfinite(duration_s) or duration_s <= 0:
        return np.nan, np.nan, np.nan
    step = 0.01
    t = np.arange(0.0, float(duration_s) + step, step)
    if t[-1] > duration_s:
        t[-1] = float(duration_s)
    if len(t) < 3:
        return np.nan, np.nan, np.nan
    flow = np_ombe_flow(params, t)
    increments = (flow[:-1] + flow[1:]) * 0.5 * np.diff(t)
    volume = np.concatenate([[0.0], np.cumsum(increments)])
    fev1 = float(np.interp(min(1.0, t[-1]), t, volume))
    fvc = float(volume[-1])
    return fev1, fvc, fev1 / fvc if fvc > 0 else np.nan


def icc_oneway(data: pd.DataFrame, subject_col: str, value_col: str) -> float:
    part = data[[subject_col, value_col]].dropna().copy()
    groups = [g[value_col].to_numpy(float) for _, g in part.groupby(subject_col) if len(g) >= 2]
    if len(groups) < 2:
        return np.nan
    n_i = np.array([len(g) for g in groups], dtype=float)
    n_total = float(n_i.sum())
    grand = float(np.sum([g.sum() for g in groups]) / n_total)
    ss_between = float(np.sum([len(g) * (float(np.mean(g)) - grand) ** 2 for g in groups]))
    ss_within = float(np.sum([np.sum((g - float(np.mean(g))) ** 2) for g in groups]))
    m = float(len(groups))
    ms_between = ss_between / (m - 1.0) if m > 1 else np.nan
    ms_within = ss_within / (n_total - m) if n_total > m else np.nan
    k0 = (n_total - float(np.sum(n_i**2)) / n_total) / (m - 1.0) if m > 1 else np.nan
    denom = ms_between + (k0 - 1.0) * ms_within
    return float((ms_between - ms_within) / denom) if np.isfinite(denom) and abs(denom) > 1e-12 else np.nan


def repeatability_summary(df: pd.DataFrame, prefix: str) -> dict[str, Any]:
    rows = []
    for seqn, group in df.groupby("SEQN"):
        if len(group) < 2:
            continue
        fev1 = np.sort(group[f"{prefix}_fev1_l"].to_numpy(float))[::-1]
        fvc = np.sort(group[f"{prefix}_fvc_l"].to_numpy(float))[::-1]
        rows.append(
            {
                "SEQN": float(seqn),
                f"{prefix}_fev1_top2_diff_l": float(abs(fev1[0] - fev1[1])),
                f"{prefix}_fvc_top2_diff_l": float(abs(fvc[0] - fvc[1])),
                f"{prefix}_ats_repeatable": bool(abs(fev1[0] - fev1[1]) <= 0.150 and abs(fvc[0] - fvc[1]) <= 0.150),
            }
        )
    out = pd.DataFrame(rows)
    return {
        "n": int(len(out)),
        "repeatable_n": int(out[f"{prefix}_ats_repeatable"].sum()) if len(out) else 0,
        "repeatable_fraction": float(out[f"{prefix}_ats_repeatable"].mean()) if len(out) else np.nan,
        "fev1_top2_diff_l": median_iqr(out[f"{prefix}_fev1_top2_diff_l"]) if len(out) else {},
        "fvc_top2_diff_l": median_iqr(out[f"{prefix}_fvc_top2_diff_l"]) if len(out) else {},
    }


def part_d_reproducibility(truth: pd.DataFrame, candidates: pd.DataFrame, gate3_fits: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row_records: list[dict[str, Any]] = []
    usable = candidates[candidates["parse_error"].fillna("") == ""].copy()
    counts = usable.groupby("SEQN").size()
    eligible_seqn = counts[counts >= 2].index
    multi = usable[usable["SEQN"].isin(eligible_seqn)].copy()
    fit_cols = ["SEQN", "SPAMANU", "fit_success", "duration_s", "fit_tau_f", "fit_tau_s", "fit_alpha", "fit_fvc_l", "fit_tau_o"]
    fits = gate3_fits[fit_cols].copy()
    fits["fit_success"] = bool_series(fits["fit_success"])
    merged = multi.merge(fits, on=["SEQN", "SPAMANU"], how="left", suffixes=("_observed", "_fit"))
    merged = merged[merged["fit_success"].fillna(False)].copy()
    model_metrics = []
    for row in merged.itertuples(index=False):
        params = np.array([row.fit_tau_f, row.fit_tau_s, row.fit_alpha, row.fit_fvc_l, row.fit_tau_o], dtype=float)
        fev1, fvc, ratio = volume_from_flow_params(params, float(row.duration_s_fit))
        model_metrics.append((fev1, fvc, ratio))
    if model_metrics:
        mm = np.asarray(model_metrics, dtype=float)
        merged["model_fev1_l"] = mm[:, 0]
        merged["model_fvc_l"] = mm[:, 1]
        merged["model_ratio"] = mm[:, 2]
    else:
        merged["model_fev1_l"] = np.nan
        merged["model_fvc_l"] = np.nan
        merged["model_ratio"] = np.nan
    merged["observed_fev1_l"] = merged["parsed_fev1_l"]
    merged["observed_fvc_l"] = merged["parsed_fvc_l"]
    merged["observed_ratio"] = merged["parsed_ratio"]
    merged = merged.merge(truth[["SEQN", "fev1_fvc_ratio"]], on="SEQN", how="left")
    merged["truth_severity"] = merged["fev1_fvc_ratio"].map(severity_category)
    merged["model_obstructed"] = merged["model_ratio"] < 0.70

    for row in merged.itertuples(index=False):
        row_records.append(
            {
                "row_type": "part_d_manoeuvre",
                "SEQN": float(row.SEQN),
                "raw_index": int(row.raw_index),
                "SPAMANU": float(row.SPAMANU),
                "SPAQEFF": row.SPAQEFF,
                "truth_severity": row.truth_severity,
                "observed_fev1_l": float(row.observed_fev1_l),
                "observed_fvc_l": float(row.observed_fvc_l),
                "observed_ratio": float(row.observed_ratio),
                "model_fev1_l": float(row.model_fev1_l),
                "model_fvc_l": float(row.model_fvc_l),
                "model_ratio": float(row.model_ratio),
                "model_obstructed_fixed": bool(row.model_obstructed),
                "fit_source": "results/gate3_F_curve_fits.csv flow-domain deterministic OMBE fit",
            }
        )

    per_participant_rows = []
    for seqn, group in merged.groupby("SEQN"):
        if len(group) < 2:
            continue
        calls = group["model_obstructed"].astype(bool)
        majority_fraction = float(max(calls.mean(), 1.0 - calls.mean()))
        per_participant_rows.append(
            {
                "row_type": "part_d_participant_summary",
                "SEQN": float(seqn),
                "truth_severity": group["truth_severity"].iloc[0],
                "maneuvre_count": int(len(group)),
                "model_ratio_sd": float(group["model_ratio"].std(ddof=1)),
                "model_fev1_sd_l": float(group["model_fev1_l"].std(ddof=1)),
                "model_fvc_sd_l": float(group["model_fvc_l"].std(ddof=1)),
                "classification_unanimous": bool(calls.nunique() == 1),
                "classification_majority_fraction": majority_fraction,
            }
        )
    row_records.extend(per_participant_rows)
    pp = pd.DataFrame(per_participant_rows)

    def stability(part: pd.DataFrame) -> dict[str, Any]:
        if part.empty:
            return {"n": 0, "unanimous_fraction": np.nan, "split_fraction": np.nan, "majority_fraction": {}}
        unanimous = part["classification_unanimous"].astype(bool)
        return {
            "n": int(len(part)),
            "unanimous_fraction": float(unanimous.mean()),
            "split_fraction": float((~unanimous).mean()),
            "majority_fraction": median_iqr(part["classification_majority_fraction"]),
        }

    manoeuvre_counts = counts[counts >= 2].value_counts().sort_index()
    count_distribution = {str(int(k)): int(v) for k, v in manoeuvre_counts.items()}
    within_sd = {
        "model_fev1_l": median_iqr(pp["model_fev1_sd_l"]) if len(pp) else {},
        "model_fvc_l": median_iqr(pp["model_fvc_sd_l"]) if len(pp) else {},
        "model_ratio": median_iqr(pp["model_ratio_sd"]) if len(pp) else {},
    }
    icc_ratio = icc_oneway(merged, "SEQN", "model_ratio")
    borderline_pp = pp[pp["truth_severity"] == "mild_0.60_0.70"].copy() if len(pp) else pd.DataFrame()
    observed_repeat = repeatability_summary(merged, "observed")
    model_repeat = repeatability_summary(merged, "model")
    result = {
        "verdict": "REPRODUCIBLE" if np.isfinite(icc_ratio) and icc_ratio >= 0.75 and stability(pp)["unanimous_fraction"] >= 0.90 else "UNSTABLE",
        "note": "Part D reuses the existing all-manoeuvre deterministic flow-domain OMBE fit table. It is not a fresh all-manoeuvre volume-domain refit.",
        "participants_with_at_least_2_acceptable_ab_manoeuvres": int(len(eligible_seqn)),
        "manoeuvre_count_distribution": count_distribution,
        "model_fit_manoeuvre_rows_used": int(len(merged)),
        "within_participant_sd": within_sd,
        "icc_ratio": icc_ratio,
        "classification_stability_all": stability(pp),
        "classification_stability_borderline_0.60_0.70": stability(borderline_pp),
        "ats_repeatability_observed": observed_repeat,
        "ats_repeatability_model": model_repeat,
        "model_more_repeatable_than_raw": bool(
            np.isfinite(model_repeat.get("repeatable_fraction", np.nan))
            and np.isfinite(observed_repeat.get("repeatable_fraction", np.nan))
            and model_repeat["repeatable_fraction"] > observed_repeat["repeatable_fraction"]
        ),
    }
    return result, row_records


def make_report(result: dict[str, Any]) -> str:
    a = result["part_a"]
    b = result.get("part_b", {})
    c = result.get("part_c", {})
    d = result.get("part_d", {})
    lines = [
        "# Ground-Truth Reconciliation + Truncation Sanity + Coverage/Reproducibility Audit, Cycle F",
        "",
        "- Deterministic only: no NUTS, SNPE, gamma onset, registration, or ML residual.",
        f"- Random seed: {RANDOM_SEED}",
        f"- Raw source: {result['raw_source']}",
        "",
        "## Verdicts",
        "",
        f"- Part A: {a['verdict']} - {'halted because the truncation headline moved materially' if a['halt'] else 'truncation TRUE fields match the ATS observed-curve definition and the headline survives.'}",
    ]
    if b:
        lines.append(f"- Part B: {b['verdict']} - {b['reason']}")
    if c:
        lines.append(f"- Part C: {c['coverage_sentence']}")
    if d:
        lines.append(
            f"- Part D: {d['verdict']} - all-manoeuvre OMBE ratio ICC {fmt(d['icc_ratio'], 3)}, borderline unanimous fraction {fmt(d['classification_stability_borderline_0.60_0.70']['unanimous_fraction'], 3)}."
        )

    lines.extend(
        [
            "",
            "## Part A - Code Paths",
            "",
            md_table(
                a["code_paths"],
                [
                    ("gate", "gate/path", "str"),
                    ("t0", "t0", "str"),
                    ("fev1", "FEV1", "str"),
                    ("fvc", "FVC", "str"),
                    ("grid", "grid", "str"),
                ],
            ),
            "",
            "## Part A - Common-Curve Comparison",
            "",
            f"- Common curves available: {a['common_curve_count_total']}",
            f"- Common curves reported in row CSV: {a['common_curve_count_reported']}",
            "",
            "The per-curve table is in `results/gate_reconciliation_audit_F_rows.csv` with `row_type == part_a_common_curve`.",
            "",
            "## Part A - Gap Attribution",
            "",
            "| comparison | n | median abs | IQR |",
            "|---|---:|---:|---:|",
        ]
    )
    attr_rows = [
        ("clinical model FEV1 vs NHANES, obstructed", a["attribution"]["clinical_model_vs_nhanes_obstructed_fev1_abs_pct"]["absolute"], "%"),
        ("clinical observed FEV1 vs NHANES, obstructed", a["attribution"]["clinical_observed_vs_nhanes_obstructed_fev1_abs_pct"]["absolute"], "%"),
        ("truncation observed FEV1 vs NHANES, sampled", a["attribution"]["trunc_observed_vs_nhanes_sample_fev1_abs_pct"]["absolute"], "%"),
        ("clinical model FVC vs NHANES, obstructed", a["attribution"]["clinical_model_vs_nhanes_obstructed_fvc_abs_pct"]["absolute"], "%"),
        ("clinical observed FVC vs NHANES, obstructed", a["attribution"]["clinical_observed_vs_nhanes_obstructed_fvc_abs_pct"]["absolute"], "%"),
        ("truncation observed FVC vs NHANES, sampled", a["attribution"]["trunc_observed_vs_nhanes_sample_fvc_abs_pct"]["absolute"], "%"),
    ]
    for label, item, suffix in attr_rows:
        lines.append(f"| {label} | {item['n']} | {fmt(item['median'], 3)}{suffix} | {fmt(item['iqr'][0], 3)} to {fmt(item['iqr'][1], 3)}{suffix} |")
    lines.extend(
        [
            "",
            f"- Dominant attribution: {a['dominant_gap_attribution']}.",
            f"- Nearest-sample versus interpolation FEV1 difference on common curves: median {fmt(a['attribution']['nearest_vs_interpolation_fev1_mL_common30']['median'], 3)} mL.",
            f"- Max recomputed TRUE delta: FEV1 {fmt(a['max_recomputed_truth_abs_delta']['fev1_l'] * 1000.0, 6)} mL, FVC {fmt(a['max_recomputed_truth_abs_delta']['fvc_l'] * 1000.0, 6)} mL, ratio {fmt(a['max_recomputed_truth_abs_delta']['ratio'], 8)}.",
            "",
            "### Headline Recompute",
            "",
            md_table(
                a["headline_recalc"],
                [
                    ("estimator", "estimator", "str"),
                    ("T_label", "T", "str"),
                    ("old_false_negative_rate_obstructed", "old FN", "float3"),
                    ("recomputed_false_negative_rate_obstructed", "recomputed FN", "float3"),
                    ("delta", "delta", "float4"),
                ],
            ),
            "",
        ]
    )

    if a["halt"]:
        lines.extend(
            [
                "Part A materially changed the truncation headline, so Parts B-D were not run.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "## Part B - Non-Monotonicity",
            "",
            "### Obstructed Sample Counts",
            "",
            md_table(b["obstructed_stratum_counts"], [("truth_severity", "truth stratum", "str"), ("n", "n", "int")]),
            "",
            "### E1 Obstructed False-Negative Bootstrap",
            "",
            md_table(
                [r for r in b["bootstrap_rows"] if r["row_type"] == "part_b_bootstrap_fn"],
                [
                    ("T_label", "T", "str"),
                    ("n_obstructed", "n", "int"),
                    ("fn_rate", "FN rate", "float3"),
                    ("bootstrap_ci_low", "95% CI low", "float3"),
                    ("bootstrap_ci_high", "95% CI high", "float3"),
                ],
            ),
            "",
            f"- Paired T=3 minus T=2 FN difference: {fmt(b['paired_difference_T3_minus_T2']['mean_difference_T3_minus_T2'], 4)} "
            f"(95% CI {fmt(b['paired_difference_T3_minus_T2']['ci_low'], 4)} to {fmt(b['paired_difference_T3_minus_T2']['ci_high'], 4)}).",
            "",
            "### E1 Bound Counts",
            "",
            md_table(
                b["bound_summary"],
                [
                    ("T_label", "T", "str"),
                    ("parameter", "parameter", "str"),
                    ("fit_success_n", "fit success n", "int"),
                    ("low_bound_n", "low", "int"),
                    ("high_bound_n", "high", "int"),
                    ("either_bound_fraction", "bound frac", "float3"),
                ],
            ),
            "",
            "### E1 FVC Error Trend",
            "",
            md_table(
                b["fvc_signed_error_trend"],
                [
                    ("T_label", "T", "str"),
                    ("n", "n", "int"),
                    ("fvc_signed_pct_median", "signed median %", "float3"),
                    ("fvc_abs_pct_median", "abs median %", "float3"),
                ],
            ),
            "",
        ]
    )

    lines.extend(
        [
            "## Part C - Drops as FLAG",
            "",
            f"- Clinical severe+moderate denominator: {c['clinical_drop_denominator_n']}",
            f"- Clinical gate drops classified: {c['clinical_drop_n']}",
            f"- Drop taxonomy counts: {c['clinical_drop_taxonomy_counts']}",
            f"- Parse bugs fixed: {c['parse_bug_fixed_n']}",
            f"- Policy rows available inside the severe+moderate denominator: {c['policy_available_modsev_n']}; policy rows outside that denominator: {c['policy_extra_modsev_outside_denominator_n']}; policy-missing counted as FLAG: {c['policy_missing_modsev_counted_flag_n']}",
            f"- Full-denominator policy outcome counts: {c['policy_outcome_counts_full_denominator']}",
            f"- {c['coverage_sentence']}",
            "",
            "Per-participant drop taxonomy rows are in the row CSV with `row_type == part_c_clinical_drop_taxonomy`.",
            "",
        ]
    )

    lines.extend(
        [
            "## Part D - Cross-Manoeuvre Reproducibility",
            "",
            f"- Limitation: {d['note']}",
            f"- Participants with >=2 acceptable A/B manoeuvres: {d['participants_with_at_least_2_acceptable_ab_manoeuvres']}",
            f"- Manoeuvre-count distribution: {d['manoeuvre_count_distribution']}",
            f"- Model fit manoeuvre rows used: {d['model_fit_manoeuvre_rows_used']}",
            "",
            "| metric | median within-participant SD | IQR |",
            "|---|---:|---:|",
            f"| model FEV1 (L) | {fmt(d['within_participant_sd']['model_fev1_l']['median'], 4)} | {fmt(d['within_participant_sd']['model_fev1_l']['iqr'][0], 4)} to {fmt(d['within_participant_sd']['model_fev1_l']['iqr'][1], 4)} |",
            f"| model FVC (L) | {fmt(d['within_participant_sd']['model_fvc_l']['median'], 4)} | {fmt(d['within_participant_sd']['model_fvc_l']['iqr'][0], 4)} to {fmt(d['within_participant_sd']['model_fvc_l']['iqr'][1], 4)} |",
            f"| model FEV1/FVC | {fmt(d['within_participant_sd']['model_ratio']['median'], 4)} | {fmt(d['within_participant_sd']['model_ratio']['iqr'][0], 4)} to {fmt(d['within_participant_sd']['model_ratio']['iqr'][1], 4)} |",
            "",
            f"- ICC for model ratio: {fmt(d['icc_ratio'], 3)}",
            f"- Classification stability, all: unanimous {fmt(d['classification_stability_all']['unanimous_fraction'], 3)}, split {fmt(d['classification_stability_all']['split_fraction'], 3)}.",
            f"- Classification stability, borderline 0.60-0.70: unanimous {fmt(d['classification_stability_borderline_0.60_0.70']['unanimous_fraction'], 3)}, split {fmt(d['classification_stability_borderline_0.60_0.70']['split_fraction'], 3)}.",
            "",
            "### ATS Repeatability",
            "",
            "| source | n | repeatable n | repeatable fraction | FEV1 top-2 diff median L | FVC top-2 diff median L |",
            "|---|---:|---:|---:|---:|---:|",
            f"| observed curves | {d['ats_repeatability_observed']['n']} | {d['ats_repeatability_observed']['repeatable_n']} | {fmt(d['ats_repeatability_observed']['repeatable_fraction'], 3)} | {fmt(d['ats_repeatability_observed']['fev1_top2_diff_l']['median'], 4)} | {fmt(d['ats_repeatability_observed']['fvc_top2_diff_l']['median'], 4)} |",
            f"| OMBE model estimates | {d['ats_repeatability_model']['n']} | {d['ats_repeatability_model']['repeatable_n']} | {fmt(d['ats_repeatability_model']['repeatable_fraction'], 3)} | {fmt(d['ats_repeatability_model']['fev1_top2_diff_l']['median'], 4)} | {fmt(d['ats_repeatability_model']['fvc_top2_diff_l']['median'], 4)} |",
            "",
        ]
    )
    if d["model_more_repeatable_than_raw"]:
        lines.append("The model estimates are more repeatable than the observed raw manoeuvre metrics under the ATS top-two rule.")
    else:
        lines.append("The model estimates are not more repeatable than the observed raw manoeuvre metrics under the ATS top-two rule.")
    lines.append("")
    return "\n".join(lines)


def run(root: Path) -> dict[str, Any]:
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    outputs = load_prior_outputs(root)
    column_map = load_column_map(root)
    truth = load_truth(root, column_map)

    raw_path, raw_note = find_or_unzip_raw(root)
    raw, raw_info = read_raw(raw_path)
    raw_by_index = raw_index_map(raw)

    all_rows: list[dict[str, Any]] = []
    part_a, a_rows, halt = part_a_reconciliation(raw_by_index, outputs["clinical"], outputs["truncation"])
    all_rows.extend(a_rows)

    result: dict[str, Any] = {
        "random_seed": RANDOM_SEED,
        "raw_source": str(raw_path),
        "raw_source_note": raw_note,
        "raw_loaded_rows": int(raw_info["loaded_rows"]),
        "part_a": part_a,
    }
    if halt:
        report = make_report(result)
        (results_dir / f"{OUTPUT_STEM}.md").write_text(report, encoding="utf-8")
        (results_dir / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
        pd.DataFrame(all_rows).to_csv(results_dir / f"{OUTPUT_STEM}_rows.csv", index=False)
        return result

    part_b, b_rows = part_b_nonmonotonicity(raw_by_index, outputs["truncation"])
    all_rows.extend(b_rows)
    result["part_b"] = part_b

    part_c, c_rows = part_c_coverage(raw, truth, outputs["clinical"], outputs["policy"])
    all_rows.extend(c_rows)
    result["part_c"] = part_c

    print("Reconciliation audit building A/B candidate table for Part D.")
    candidates = build_usable_raw_candidates(raw)
    part_d, d_rows = part_d_reproducibility(truth, candidates, outputs["gate3_fits"])
    all_rows.extend(d_rows)
    result["part_d"] = part_d

    report = make_report(result)
    (results_dir / f"{OUTPUT_STEM}.md").write_text(report, encoding="utf-8")
    (results_dir / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(all_rows).to_csv(results_dir / f"{OUTPUT_STEM}_rows.csv", index=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    print(f"Part A: {result['part_a']['verdict']}")
    if "part_b" in result:
        print(f"Part B: {result['part_b']['verdict']} - {result['part_b']['reason']}")
    if "part_c" in result:
        print(f"Part C: {result['part_c']['coverage_sentence']}")
    if "part_d" in result:
        print(f"Part D: {result['part_d']['verdict']} - ICC ratio {result['part_d']['icc_ratio']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
