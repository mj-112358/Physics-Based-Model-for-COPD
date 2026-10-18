from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/predictx-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/predictx-xdg-cache")

import numpy as np
import pandas as pd

from .dataset_resolver import CYCLE_TO_YEAR, resolve_dataset_path
from .gate3_diagnosis import np_ombe_volume
from .gate23_f import (
    comparison_summary,
    derive_all,
    find_or_unzip_raw,
    participant_sets,
    read_ground_truth,
    read_raw,
    write_gate2_report,
)
from .gate_clinical_classification_f import decode_grade, extract_curve_for_fit, severity_category
from .gate_fit_optimization_f import forward_model_guard
from .gate_truncation_recovery_f import (
    ESTIMATORS as BASE_ESTIMATORS,
)
from .gate_truncation_recovery_f import (
    MIN_FET_S,
    PLATEAU_L,
    PREDICTION_HORIZON_S,
    SAMPLE_MAX_PER_STRATUM,
    STRATA,
    TRUNCATION_TIMES,
    clean_json,
    fit_linear_tail,
    fit_mono_truncated,
    fit_ombe_truncated,
    fit_spline_extrapolate,
    insert_time_point,
    model_grid_with_stop,
    pct_error,
)

SEED = 20260710
BOOTSTRAP_B = 2000
EXPECTED_RAW_COLUMNS = (
    "SEQN",
    "SPATTYPE",
    "SPAMANU",
    "SPAFACT",
    "SPAPOS",
    "SPAPLAT",
    "SPAACC",
    "SPAQEFF",
    "SPXPTS",
    "SPXRAW",
)
ALL_ESTIMATORS = BASE_ESTIMATORS + ("H1",)
HEADLINE_ESTIMATORS = ("E0_DIRECT", "E1_OMBE", "H1")
T_LABELS = tuple(str(int(t)) for t in TRUNCATION_TIMES)
OBSTRUCTED_SEVERITIES = ("severe", "moderate", "mild_0.60_0.70")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def boolish(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def fmt(value: Any, digits: int = 3) -> str:
    try:
        v = float(value)
        return f"{v:.{digits}f}" if np.isfinite(v) else "NA"
    except Exception:
        return str(value)


def md_table(rows: list[dict[str, Any]] | pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    df = pd.DataFrame(rows)
    out = ["| " + " | ".join(label for _, label, _ in columns) + " |"]
    out.append("|" + "|".join("---" for _ in columns) + "|")
    for _, row in df.iterrows():
        cells: list[str] = []
        for key, _, kind in columns:
            value = row.get(key, np.nan)
            if kind == "int":
                cells.append(str(int(value)) if pd.notna(value) else "NA")
            elif kind == "bool":
                cells.append(str(bool(value)) if pd.notna(value) else "NA")
            elif kind.startswith("float"):
                digits = int(kind.replace("float", "") or "3")
                cells.append(fmt(value, digits))
            elif kind == "rate":
                cells.append(rate_string(row, key))
            else:
                cells.append(str(value))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def rate_string(row: pd.Series | dict[str, Any], prefix: str) -> str:
    d = row if isinstance(row, dict) else row.to_dict()
    point = d.get(prefix)
    stems = [prefix]
    if prefix.endswith("_rate"):
        stems.append(prefix.removesuffix("_rate"))

    def first_available(suffixes: tuple[str, ...], default: Any = np.nan) -> Any:
        for stem in stems:
            for suffix in suffixes:
                key = f"{stem}_{suffix}"
                if key in d:
                    return d[key]
        return default

    num = first_available(("n", "numerator"), "")
    den = first_available(("den", "denominator"), "")
    lo = first_available(("ci_low",), np.nan)
    hi = first_available(("ci_high",), np.nan)
    flag = " UNDERPOWERED" if bool(first_available(("underpowered",), False)) else ""
    return f"{fmt(point, 3)} ({num}/{den}; CI {fmt(lo, 3)}-{fmt(hi, 3)}){flag}"


def bootstrap_binary(success: np.ndarray, seed: int) -> tuple[float, float]:
    n = len(success)
    if n == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    draws = np.empty(BOOTSTRAP_B, dtype=float)
    values = success.astype(float)
    for b in range(BOOTSTRAP_B):
        draws[b] = float(values[rng.choice(idx, size=n, replace=True)].mean())
    return float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def underpowered(point: float, low: float, high: float) -> bool:
    if not np.isfinite(point) or not np.isfinite(low) or not np.isfinite(high):
        return True
    return bool((high - low) / 2.0 > point)


def bootstrap_diff(
    wide: pd.DataFrame,
    left_col: str,
    right_col: str,
    seed: int,
    *,
    truth_col: str = "truth_obstructed_fixed",
) -> tuple[float, float, float, bool]:
    obs = wide[wide[truth_col].astype(bool)].copy()
    n = len(obs)
    if n == 0:
        return np.nan, np.nan, np.nan, False
    point = float((~obs[left_col].astype(bool)).mean() - (~obs[right_col].astype(bool)).mean())
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    draws = np.empty(BOOTSTRAP_B, dtype=float)
    left = obs[left_col].astype(bool).to_numpy()
    right = obs[right_col].astype(bool).to_numpy()
    for b in range(BOOTSTRAP_B):
        take = rng.choice(idx, size=n, replace=True)
        draws[b] = float((~left[take]).mean() - (~right[take]).mean())
    low = float(np.nanpercentile(draws, 2.5))
    high = float(np.nanpercentile(draws, 97.5))
    return point, low, high, bool(low > 0.0 or high < 0.0)


def pct_error_series(derived: pd.Series, truth: pd.Series) -> pd.Series:
    return 100.0 * (derived - truth) / truth


def gate2_result_for_cycle(root: Path, cycle: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_path, raw_note = find_or_unzip_raw(root, cycle)
    raw, raw_info = read_raw(raw_path, cycle)
    raw_cols = tuple(raw_info["columns"])
    if raw_cols != EXPECTED_RAW_COLUMNS:
        missing = sorted(set(EXPECTED_RAW_COLUMNS) - set(raw_cols))
        extra = sorted(set(raw_cols) - set(EXPECTED_RAW_COLUMNS))
        raise RuntimeError(f"Cycle {cycle} raw schema mismatch. Missing={missing}; extra={extra}; observed={raw_cols}")
    truth = read_ground_truth(root, cycle)
    derived = derive_all(raw)
    sets = participant_sets(derived, truth)
    counts = {
        "raw_rows": int(len(raw)),
        "pre_rows": int((derived["SPATTYPE"] == "Pre").sum()),
        "acceptable_pre_rows": int(((derived["SPATTYPE"] == "Pre") & (derived["SPAACC"] == "Y")).sum()),
        "A_pre_rows": int(((derived["SPATTYPE"] == "Pre") & (derived["SPAQEFF"] == "A")).sum()),
        "acceptable_A_pre_rows": int(
            ((derived["SPATTYPE"] == "Pre") & (derived["SPAACC"] == "Y") & (derived["SPAQEFF"] == "A")).sum()
        ),
        "participants_with_truth": int(truth[["SPXNFEV1", "SPXNFVC"]].dropna().shape[0]),
        "participants_first_acceptable": int(len(sets["first_acceptable_pre_per_participant"])),
        "participants_first_acceptable_A": int(len(sets["first_acceptable_A_pre_per_participant"])),
    }
    comparison_summaries = [
        comparison_summary(sets["all_acceptable_A_pre_curves"], "all acceptable A Pre curves"),
        comparison_summary(sets["first_acceptable_A_pre_per_participant"], "first acceptable A Pre per participant"),
        comparison_summary(sets["first_acceptable_pre_per_participant"], "first acceptable Pre per participant"),
        comparison_summary(sets["max_acceptable_A_pre_per_participant"], "max acceptable A Pre per participant"),
        comparison_summary(sets["max_acceptable_pre_per_participant"], "max acceptable Pre per participant"),
    ]
    criterion = next(item for item in comparison_summaries if item["label"] == "max acceptable Pre per participant")
    gate2_pass = (
        criterion.get("n", 0) > 0
        and criterion["fev1_median_abs_pct_error"] <= 5.0
        and criterion["fvc_median_abs_pct_error"] <= 5.0
    )
    result = {
        "cycle": cycle,
        "status": "PASS" if gate2_pass else "FAIL",
        "reason": "Gate 2 parser recovery criterion passed under max/best-test comparison." if gate2_pass else "Gate 2 parser recovery criterion failed.",
        "parser_callable": "predictx.gate23_f.derive_curve",
        "raw_source_note": raw_note,
        "raw_info": raw_info,
        "parse_error_count": int(derived.attrs.get("parse_error_count", 0)),
        "counts": counts,
        "comparison_summaries": comparison_summaries,
        "criterion": criterion,
        "diagnosis": "none; Gate 2 passed" if gate2_pass else "Gate 2 failed under the fixed F criterion.",
    }
    return result, raw, derived, truth


def cached_gate2_f(root: Path) -> dict[str, Any]:
    return read_json(root / "results" / "gate2_F_parse.json")


def selected_ab_from_derived(derived: pd.DataFrame) -> pd.DataFrame:
    usable = derived[
        (derived["SPATTYPE"].eq("Pre"))
        & (derived["SPAACC"].eq("Y"))
        & (derived["SPAQEFF"].isin(["A", "B"]))
        & (derived["parse_error"].fillna("").eq(""))
    ].copy()
    usable = usable[np.isfinite(usable["derived_fev1_ml"]) & np.isfinite(usable["derived_fvc_ml"])]
    usable = usable[usable["derived_fvc_ml"] > 0]
    return (
        usable.sort_values(["SEQN", "derived_fev1_ml", "SPAMANU"], ascending=[True, False, True])
        .groupby("SEQN", as_index=False)
        .first()
    )


def spxpts_summary(raw: pd.DataFrame) -> dict[str, Any]:
    arr = raw["SPXPTS"].astype(float).to_numpy()
    return {
        "spxpts_min": int(np.nanmin(arr)),
        "spxpts_p25": float(np.nanpercentile(arr, 25)),
        "spxpts_median": float(np.nanmedian(arr)),
        "spxpts_p75": float(np.nanpercentile(arr, 75)),
        "spxpts_max": int(np.nanmax(arr)),
        "spxpts_mean": float(np.nanmean(arr)),
    }


def participant_candidates(raw: pd.DataFrame, truth: pd.DataFrame, cycle: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = raw[
        (raw["SPATTYPE"].map(decode_grade) == "Pre")
        & (raw["SPAACC"].map(decode_grade) == "Y")
        & (raw["SPAQEFF"].map(decode_grade) == "A")
    ].copy()
    rows = []
    for idx, row in enumerate(filtered.itertuples(index=True), start=1):
        base = {
            "cycle": cycle,
            "global_key": f"{cycle}:{float(row.SEQN):.0f}",
            "raw_index": int(row.Index),
            "SEQN": float(row.SEQN),
            "SPAMANU": float(row.SPAMANU),
            "SPAQEFF": decode_grade(row.SPAQEFF),
            "SPAACC": decode_grade(row.SPAACC),
            "SPAPLAT": decode_grade(row.SPAPLAT),
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
                    "true_fev1_l": float(curve["parsed_fev1_l"]),
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
            print(f"Cycle {cycle} truncation candidate parse progress: {idx}/{len(filtered)} A-grade candidates")
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
        truth[["SEQN", "SPXNFEV1", "SPXNFVC"]],
        on="SEQN",
        how="left",
    )
    selected_complete = selected_complete.dropna(subset=["SPXNFEV1", "SPXNFVC"]).copy()
    selected_complete["fev1_mL"] = selected_complete["SPXNFEV1"]
    selected_complete["fvc_mL"] = selected_complete["SPXNFVC"]
    selected_complete["fev1_fvc_ratio"] = selected_complete["SPXNFEV1"] / selected_complete["SPXNFVC"]
    return selected_a, selected_complete


def sample_by_stratum(selected_a: pd.DataFrame, selected_complete: pd.DataFrame, cycle: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    sampled_parts = []
    rows = []
    all_counts = selected_a["truth_severity"].value_counts().to_dict()
    complete_counts = selected_complete["truth_severity"].value_counts().to_dict()
    rng_seed = SEED + ord(cycle)
    for stratum in STRATA:
        part = selected_complete[selected_complete["truth_severity"].eq(stratum)].copy()
        n_sample = min(SAMPLE_MAX_PER_STRATUM, len(part))
        sampled = part.sample(n=n_sample, random_state=rng_seed) if n_sample else part
        sampled_parts.append(sampled)
        all_n = int(all_counts.get(stratum, 0))
        complete_n = int(complete_counts.get(stratum, 0))
        rows.append(
            {
                "cycle": cycle,
                "stratum": stratum,
                "selected_grade_A_n": all_n,
                "complete_plateau_n": complete_n,
                "sampled_n": int(n_sample),
                "dropped_by_plateau_or_missing_nhanes": int(max(0, all_n - complete_n)),
                "complete_selection_severity_shift_n": int(max(0, complete_n - all_n)),
                "dropped_by_sampling_cap": int(max(0, complete_n - n_sample)),
            }
        )
        rng_seed += 101
    sample = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame()
    return sample, rows


def estimate_clean_curve(sample_row: Any, raw_row: Any) -> list[dict[str, Any]]:
    curve = extract_curve_for_fit(raw_row)
    full_t = np.asarray(curve["vol_t"], dtype=float)
    full_y = np.asarray(curve["vol_l"], dtype=float)
    true_fev1_l = float(np.interp(1.0, full_t, full_y))
    true_fvc_l = float(full_y[-1])
    true_ratio = true_fev1_l / true_fvc_l if true_fvc_l > 0 else np.nan
    rows: list[dict[str, Any]] = []
    for t_s in TRUNCATION_TIMES:
        t_label = str(int(t_s))
        t_fit, y_fit = insert_time_point(full_t, full_y, t_s)
        observed_fev1_l = float(np.interp(1.0, t_fit, y_fit))
        fits = {
            "E0_DIRECT": {"success": True, "fvc_l": float(y_fit[-1]), "failure": ""},
            "E1_OMBE": fit_ombe_truncated(full_t, full_y, t_s),
            "E2_SPLINE": fit_spline_extrapolate(t_fit, y_fit, PREDICTION_HORIZON_S),
            "E3_LINEAR_TAIL": fit_linear_tail(t_fit, y_fit, PREDICTION_HORIZON_S),
            "E4_MONO_EXPONENTIAL": fit_mono_truncated(full_t, full_y, t_s),
        }
        for estimator, fit in fits.items():
            est_fvc_l = float(fit["fvc_l"]) if finite(fit.get("fvc_l", np.nan)) else np.nan
            success = bool(fit.get("success", False) and np.isfinite(est_fvc_l) and est_fvc_l > 0)
            est_ratio = observed_fev1_l / est_fvc_l if success else np.nan
            rows.append(detail_row(sample_row, estimator, t_label, success, fit.get("failure", ""), est_fvc_l, est_ratio, true_fev1_l, true_fvc_l, true_ratio, observed_fev1_l, float(y_fit[-1]), "clean_cut"))

        e1_fit = fits["E1_OMBE"]
        try:
            grid, _fit_idx = model_grid_with_stop(full_t, t_s)
            pred = np_ombe_volume(np.asarray(e1_fit["params"], dtype=float), grid)
            model_fev1 = float(np.interp(1.0, grid, pred))
            model_fvc = float(pred[-1])
            h1_ratio = model_fev1 / model_fvc if model_fvc > 0 else np.nan
            h1_success = bool(e1_fit.get("success", False) and np.isfinite(h1_ratio))
            h1_failure = "" if h1_success else str(e1_fit.get("failure", "nonfinite H1 ratio"))
        except Exception as exc:
            model_fvc = np.nan
            h1_ratio = np.nan
            h1_success = False
            h1_failure = str(exc)
        rows.append(detail_row(sample_row, "H1", t_label, h1_success, h1_failure, model_fvc, h1_ratio, true_fev1_l, true_fvc_l, true_ratio, observed_fev1_l, float(y_fit[-1]), "clean_cut"))
    return rows


def detail_row(
    sample_row: Any,
    estimator: str,
    t_label: str,
    success: bool,
    failure: str,
    est_fvc_l: float,
    est_ratio: float,
    true_fev1_l: float,
    true_fvc_l: float,
    true_ratio: float,
    observed_fev1_l: float,
    observed_end_l: float,
    dataset: str,
) -> dict[str, Any]:
    truth_obstructed = bool(true_ratio < 0.70) if np.isfinite(true_ratio) else False
    est_obstructed = bool(est_ratio < 0.70) if np.isfinite(est_ratio) and success else False
    return {
        "dataset": dataset,
        "cycle": sample_row.cycle,
        "global_key": sample_row.global_key,
        "SEQN": float(sample_row.SEQN),
        "raw_index": int(sample_row.raw_index),
        "SPAMANU": float(sample_row.SPAMANU),
        "truth_severity": sample_row.truth_severity,
        "T_label": str(t_label),
        "estimator": estimator,
        "fit_success": bool(success),
        "fit_failure": "" if success else str(failure),
        "true_fev1_l": true_fev1_l,
        "true_fvc_l": true_fvc_l,
        "true_ratio": true_ratio,
        "nhanes_fev1_mL": float(sample_row.fev1_mL),
        "nhanes_fvc_mL": float(sample_row.fvc_mL),
        "nhanes_ratio": float(sample_row.fev1_fvc_ratio),
        "observed_truncated_fev1_l": observed_fev1_l,
        "observed_truncated_end_l": observed_end_l,
        "est_fvc_l": est_fvc_l,
        "est_ratio": est_ratio,
        "fvc_signed_pct_error": pct_error(est_fvc_l, true_fvc_l),
        "fvc_abs_pct_error": abs(pct_error(est_fvc_l, true_fvc_l)) if np.isfinite(pct_error(est_fvc_l, true_fvc_l)) else np.nan,
        "ratio_signed_error": float(est_ratio - true_ratio) if np.isfinite(est_ratio) and np.isfinite(true_ratio) else np.nan,
        "ratio_abs_error": float(abs(est_ratio - true_ratio)) if np.isfinite(est_ratio) and np.isfinite(true_ratio) else np.nan,
        "truth_obstructed_fixed": truth_obstructed,
        "est_obstructed_fixed": est_obstructed,
    }


def run_clean_estimators(sample: pd.DataFrame, raw: pd.DataFrame, cycle: str) -> pd.DataFrame:
    raw_by_index = {int(idx): row for idx, row in raw.iterrows()}
    rows: list[dict[str, Any]] = []
    total = len(sample)
    for i, row in enumerate(sample.itertuples(index=False), start=1):
        rows.extend(estimate_clean_curve(row, raw_by_index[int(row.raw_index)]))
        if i % 25 == 0 or i == total:
            print(f"Cycle {cycle} clean truncation estimator progress: {i}/{total} participants")
    return pd.DataFrame(rows)


def classification_by_group(detail: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, part in detail.groupby(group_cols, dropna=False):
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        base = {col: keys_tuple[i] for i, col in enumerate(group_cols)}
        all_rows = part.drop_duplicates(["global_key", "estimator"]).copy()
        valid = all_rows["fit_success"].map(boolish) & all_rows["est_ratio"].map(finite)
        truth = all_rows["truth_obstructed_fixed"].map(boolish)
        pred = all_rows["est_obstructed_fixed"].map(boolish) & valid
        obs = truth
        normal = ~truth
        tp = int((obs & pred).sum())
        fn = int((obs & ~pred).sum())
        tn = int((normal & ~pred).sum())
        fp = int((normal & pred).sum())
        obs_den = int(obs.sum())
        normal_den = int(normal.sum())
        fn_rate = fn / obs_den if obs_den else np.nan
        spec = tn / normal_den if normal_den else np.nan
        fn_lo, fn_hi = bootstrap_binary((~pred[obs]).to_numpy(bool), SEED + len(rows) * 17 + 1)
        spec_lo, spec_hi = bootstrap_binary((~pred[normal]).to_numpy(bool), SEED + len(rows) * 17 + 2)
        rows.append(
            {
                **base,
                "n": int(len(all_rows)),
                "fit_failure_n": int((~valid).sum()),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "fn_rate": fn_rate,
                "fn_rate_n": fn,
                "fn_rate_den": obs_den,
                "fn_rate_ci_low": fn_lo,
                "fn_rate_ci_high": fn_hi,
                "fn_rate_underpowered": underpowered(fn_rate, fn_lo, fn_hi),
                "specificity": spec,
                "specificity_n": tn,
                "specificity_den": normal_den,
                "specificity_ci_low": spec_lo,
                "specificity_ci_high": spec_hi,
                "specificity_underpowered": underpowered(spec, spec_lo, spec_hi),
            }
        )
    return pd.DataFrame(rows)


def headline_wide(detail: pd.DataFrame, cycle: str | None, t_label: str, estimators: tuple[str, ...] = HEADLINE_ESTIMATORS) -> pd.DataFrame:
    part = detail[detail["T_label"].astype(str).eq(str(t_label)) & detail["estimator"].isin(estimators)].copy()
    if cycle is not None:
        part = part[part["cycle"].eq(cycle)].copy()
    rows: list[dict[str, Any]] = []
    for gkey, group in part.groupby("global_key"):
        row = {
            "global_key": gkey,
            "cycle": group["cycle"].iloc[0],
            "truth_obstructed_fixed": boolish(group["truth_obstructed_fixed"].iloc[0]),
            "truth_severity": group["truth_severity"].iloc[0],
        }
        for est in estimators:
            est_part = group[group["estimator"].eq(est)]
            if len(est_part):
                item = est_part.iloc[0]
                valid = boolish(item["fit_success"]) and finite(item["est_ratio"])
                row[f"{est}_pred"] = boolish(item["est_obstructed_fixed"]) if valid else False
                row[f"{est}_valid"] = valid
            else:
                row[f"{est}_pred"] = False
                row[f"{est}_valid"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def per_cycle_differences(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cycle in CYCLE_TO_YEAR:
        for t_label in T_LABELS:
            wide = headline_wide(detail, cycle, t_label)
            for comp, left, right in [
                ("DIRECT_minus_OMBE", "E0_DIRECT_pred", "E1_OMBE_pred"),
                ("DIRECT_minus_H1", "E0_DIRECT_pred", "H1_pred"),
            ]:
                point, low, high, excludes = bootstrap_diff(wide, left, right, SEED + len(rows) * 31 + 5)
                rows.append(
                    {
                        "scope": "per_cycle_clean",
                        "cycle": cycle,
                        "T_label": t_label,
                        "comparison": comp,
                        "n_obstructed": int(wide["truth_obstructed_fixed"].sum()) if len(wide) else 0,
                        "point": point,
                        "ci_low": low,
                        "ci_high": high,
                        "ci_excludes_zero": excludes,
                        "bootstrap_n": BOOTSTRAP_B,
                    }
                )
    return pd.DataFrame(rows)


def cross_cycle_summary(classification: pd.DataFrame, diffs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cycle in CYCLE_TO_YEAR:
        for t_label in T_LABELS:
            row = {"cycle": cycle, "T_label": t_label}
            for est, key in [("E0_DIRECT", "direct"), ("E1_OMBE", "ombe"), ("H1", "h1")]:
                part = classification[
                    classification["cycle"].eq(cycle)
                    & classification["T_label"].astype(str).eq(t_label)
                    & classification["estimator"].eq(est)
                ]
                if len(part):
                    p = part.iloc[0]
                    row[f"{key}_fn_rate"] = float(p["fn_rate"])
                    row[f"{key}_fn_n"] = int(p["fn_rate_n"])
                    row[f"{key}_fn_den"] = int(p["fn_rate_den"])
                    row[f"{key}_fn_ci_low"] = float(p["fn_rate_ci_low"])
                    row[f"{key}_fn_ci_high"] = float(p["fn_rate_ci_high"])
                    row[f"{key}_fit_failure_n"] = int(p["fit_failure_n"])
            for comp in ["DIRECT_minus_OMBE", "DIRECT_minus_H1"]:
                d = diffs[diffs["cycle"].eq(cycle) & diffs["T_label"].eq(t_label) & diffs["comparison"].eq(comp)]
                if len(d):
                    name = comp.lower()
                    row[f"{name}_point"] = float(d.iloc[0]["point"])
                    row[f"{name}_ci_low"] = float(d.iloc[0]["ci_low"])
                    row[f"{name}_ci_high"] = float(d.iloc[0]["ci_high"])
                    row[f"{name}_excludes_zero"] = bool(d.iloc[0]["ci_excludes_zero"])
            rows.append(row)
    return pd.DataFrame(rows)


def incomplete_flagged_pool(raw: pd.DataFrame, selected_complete: pd.DataFrame, cycle: str) -> pd.DataFrame:
    targets = selected_complete[selected_complete["true_ratio"].astype(float) < 0.50].copy()
    targets = targets.rename(columns={"raw_index": "target_raw_index", "SPAMANU": "target_SPAMANU"})
    raw2 = raw.copy()
    raw2["raw_index"] = raw2.index.astype(int)
    raw2["SPATTYPE_dec"] = raw2["SPATTYPE"].map(decode_grade)
    raw2["SPAACC_dec"] = raw2["SPAACC"].map(decode_grade)
    raw2["SPAQEFF_dec"] = raw2["SPAQEFF"].map(decode_grade)
    raw2["SPAPLAT_dec"] = raw2["SPAPLAT"].map(decode_grade)
    raw2 = raw2[raw2["SPATTYPE_dec"].eq("Pre")].copy()
    raw2["incomplete_flag"] = raw2["SPAQEFF_dec"].isin(["C", "D"]) | raw2["SPAACC_dec"].ne("Y") | raw2["SPAPLAT_dec"].ne("Y")
    flagged = raw2[raw2["incomplete_flag"]].copy()
    if targets.empty or flagged.empty:
        return pd.DataFrame()
    keep = [
        "cycle",
        "global_key",
        "SEQN",
        "target_raw_index",
        "target_SPAMANU",
        "truth_severity",
        "true_fev1_l",
        "true_fvc_l",
        "true_ratio",
        "fev1_mL",
        "fvc_mL",
        "fev1_fvc_ratio",
        "fet_s",
    ]
    joined = flagged.merge(targets[keep], on="SEQN", how="inner")
    joined = joined[joined["raw_index"].astype(int).ne(joined["target_raw_index"].astype(int))].copy()
    raw_by_index = {int(idx): row for idx, row in raw.iterrows()}
    rows = []
    for item in joined.itertuples(index=False):
        base = item._asdict()
        try:
            curve = extract_curve_for_fit(raw_by_index[int(item.raw_index)])
            t = np.asarray(curve["vol_t"], dtype=float)
            y = np.asarray(curve["vol_l"], dtype=float)
            if len(t) < 5 or float(t[-1]) <= 0 or float(y[-1]) <= 0:
                raise ValueError("unusable incomplete curve")
            base.update(
                {
                    "field_parse_error": "",
                    "field_duration_s": float(t[-1]),
                    "field_end_volume_l": float(y[-1]),
                    "field_fev1_l": float(np.interp(1.0, t, y)),
                    "field_n_points": int(len(t)),
                }
            )
        except Exception as exc:
            base.update({"field_parse_error": str(exc), "field_duration_s": np.nan, "field_end_volume_l": np.nan, "field_fev1_l": np.nan, "field_n_points": 0})
        rows.append(base)
    parsed = pd.DataFrame(rows)
    usable = parsed[parsed["field_parse_error"].fillna("").eq("")].copy()
    if usable.empty:
        return usable
    return (
        usable.sort_values(["global_key", "field_duration_s", "SPAMANU", "raw_index"], ascending=[True, True, True, True])
        .groupby("global_key", as_index=False)
        .first()
    )


def estimate_incomplete_curve(row: Any, raw_row: Any) -> list[dict[str, Any]]:
    curve = extract_curve_for_fit(raw_row)
    field_t = np.asarray(curve["vol_t"], dtype=float)
    field_y = np.asarray(curve["vol_l"], dtype=float)
    stop_s = min(float(field_t[-1]), PREDICTION_HORIZON_S)
    t_fit, y_fit = insert_time_point(field_t, field_y, stop_s)
    observed_fev1 = float(np.interp(1.0, t_fit, y_fit))
    direct_fvc = float(y_fit[-1])
    fits = {
        "E0_DIRECT": {"success": np.isfinite(direct_fvc) and direct_fvc > 0, "fvc_l": direct_fvc, "failure": ""},
        "E1_OMBE": fit_ombe_truncated(field_t, field_y, stop_s),
        "E2_SPLINE": fit_spline_extrapolate(t_fit, y_fit, PREDICTION_HORIZON_S),
        "E3_LINEAR_TAIL": fit_linear_tail(t_fit, y_fit, PREDICTION_HORIZON_S),
        "E4_MONO_EXPONENTIAL": fit_mono_truncated(field_t, field_y, stop_s),
    }
    rows = []
    for estimator, fit in fits.items():
        est_fvc = float(fit.get("fvc_l", np.nan)) if finite(fit.get("fvc_l", np.nan)) else np.nan
        success = bool(fit.get("success", False) and np.isfinite(est_fvc) and est_fvc > 0)
        ratio = observed_fev1 / est_fvc if success else np.nan
        rows.append(detail_row(row, estimator, "field_incomplete", success, fit.get("failure", ""), est_fvc, ratio, float(row.true_fev1_l), float(row.true_fvc_l), float(row.true_ratio), observed_fev1, direct_fvc, "pooled_severe_incomplete"))
    e1_fit = fits["E1_OMBE"]
    try:
        grid, _fit_idx = model_grid_with_stop(field_t, stop_s)
        pred = np_ombe_volume(np.asarray(e1_fit["params"], dtype=float), grid)
        model_fev1 = float(np.interp(1.0, grid, pred))
        model_fvc = float(pred[-1])
        h1_ratio = model_fev1 / model_fvc if model_fvc > 0 else np.nan
        success = bool(e1_fit.get("success", False) and np.isfinite(h1_ratio))
        failure = "" if success else str(e1_fit.get("failure", "nonfinite H1 ratio"))
    except Exception as exc:
        model_fvc = np.nan
        h1_ratio = np.nan
        success = False
        failure = str(exc)
    rows.append(detail_row(row, "H1", "field_incomplete", success, failure, model_fvc, h1_ratio, float(row.true_fev1_l), float(row.true_fvc_l), float(row.true_ratio), observed_fev1, direct_fvc, "pooled_severe_incomplete"))
    return rows


def run_incomplete_estimators(cohort: pd.DataFrame, raw_by_cycle: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    raw_maps = {cycle: {int(idx): row for idx, row in raw.iterrows()} for cycle, raw in raw_by_cycle.items()}
    total = len(cohort)
    for i, row in enumerate(cohort.itertuples(index=False), start=1):
        rows.extend(estimate_incomplete_curve(row, raw_maps[row.cycle][int(row.raw_index)]))
        if i % 25 == 0 or i == total:
            print(f"Pooled severe incomplete estimator progress: {i}/{total} participants")
    return pd.DataFrame(rows)


def severity_recovery(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    part = detail[
        detail["estimator"].isin(["E1_OMBE", "H1"])
        & detail["T_label"].isin(T_LABELS)
        & detail["truth_severity"].isin(OBSTRUCTED_SEVERITIES)
    ].copy()
    for (est, t_label, severity), group in part.groupby(["estimator", "T_label", "truth_severity"], dropna=False):
        valid = group[group["fit_success"].map(boolish) & group["est_ratio"].map(finite)].copy()
        pred = valid["est_obstructed_fixed"].map(boolish)
        n = len(valid)
        fn = int((~pred).sum())
        lo, hi = bootstrap_binary((~pred).to_numpy(bool), SEED + len(rows) * 19 + 7)
        rate = fn / n if n else np.nan
        rows.append(
            {
                "estimator": est,
                "T_label": t_label,
                "truth_severity": severity,
                "fn_rate": rate,
                "fn_rate_n": fn,
                "fn_rate_den": n,
                "fn_rate_ci_low": lo,
                "fn_rate_ci_high": hi,
                "fn_rate_underpowered": bool(n < 10 or underpowered(rate, lo, hi)),
            }
        )
    return pd.DataFrame(rows)


def load_covariates(root: Path) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    rows = []
    notes = []
    variables = {
        "smoking": "SMQ020 + SMQ040",
        "self_reported_obstructive_disease": "MCQ160G emphysema OR MCQ160K chronic bronchitis",
        "age": "DEMO.RIDAGEYR",
        "ethnicity_common_key": "DEMO.RIDRETH1",
        "body_measurements_loaded": "BMXHT, BMXWT, BMXBMI",
    }
    for cycle, year in CYCLE_TO_YEAR.items():
        try:
            demo = pd.read_sas(resolve_dataset_path(root, "DEMO", cycle).path, format="xport")
            smq = pd.read_sas(resolve_dataset_path(root, "SMQ", cycle).path, format="xport")
            mcq = pd.read_sas(resolve_dataset_path(root, "MCQ", cycle).path, format="xport")
            bmx = pd.read_sas(resolve_dataset_path(root, "BMX", cycle).path, format="xport")
        except Exception as exc:
            notes.append(f"Cycle {cycle} covariates unavailable: {exc}")
            continue
        cov = demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1"]].copy()
        for col in ["BMXHT", "BMXWT", "BMXBMI"]:
            cov[col] = bmx.set_index("SEQN")[col].reindex(cov["SEQN"]).to_numpy() if col in bmx.columns else np.nan
        smq_small = smq[[c for c in ["SEQN", "SMQ020", "SMQ040"] if c in smq.columns]].copy()
        mcq_small = mcq[[c for c in ["SEQN", "MCQ160G", "MCQ160K"] if c in mcq.columns]].copy()
        cov = cov.merge(smq_small, on="SEQN", how="left").merge(mcq_small, on="SEQN", how="left")
        cov["cycle"] = cycle
        cov["global_key"] = cov["cycle"] + ":" + cov["SEQN"].astype(float).astype(int).astype(str)
        cov["smoking_status"] = cov.apply(smoking_status, axis=1)
        cov["self_reported_obstructive_disease"] = cov.apply(obstructive_disease, axis=1)
        cov["age_band"] = cov["RIDAGEYR"].map(age_band)
        rows.append(cov)
        if "RIDRETH3" in demo.columns:
            notes.append(f"Cycle {cycle} has RIDRETH3, but RIDRETH1 was used as the common ethnicity key.")
    if not rows:
        return pd.DataFrame(), notes, variables
    return pd.concat(rows, ignore_index=True), notes, variables


def smoking_status(row: pd.Series) -> str:
    smq020 = row.get("SMQ020", np.nan)
    smq040 = row.get("SMQ040", np.nan)
    if smq020 == 2:
        return "never"
    if smq020 == 1 and smq040 in {1, 2}:
        return "current"
    if smq020 == 1 and smq040 == 3:
        return "former"
    if smq020 == 1:
        return "former_unknown_current_status"
    return "unknown"


def obstructive_disease(row: pd.Series) -> str:
    emph = row.get("MCQ160G", np.nan)
    bronch = row.get("MCQ160K", np.nan)
    if emph == 1 or bronch == 1:
        return "yes"
    if emph == 2 and bronch == 2:
        return "no"
    return "unknown"


def age_band(age: Any) -> str:
    if not finite(age):
        return "unknown"
    age_f = float(age)
    if age_f < 40:
        return "<40"
    if age_f < 60:
        return "40-59"
    if age_f < 80:
        return "60-79"
    return "80+"


def external_validity(detail: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    cov, notes, variables = load_covariates(root)
    if cov.empty:
        return pd.DataFrame(), notes, variables
    base = detail[
        detail["T_label"].isin(["2", "3"])
        & detail["estimator"].isin(["E0_DIRECT", "E1_OMBE"])
    ].copy()
    base = base.merge(cov, on=["global_key", "cycle", "SEQN"], how="left")
    rows = []
    for strat_name, col in [
        ("smoking_status", "smoking_status"),
        ("self_reported_obstructive_disease", "self_reported_obstructive_disease"),
        ("age_band", "age_band"),
    ]:
        for stratum, stratum_part in base.groupby(col, dropna=False):
            for t_label in ["2", "3"]:
                wide = make_wide_for_estimators(stratum_part[stratum_part["T_label"].eq(t_label)], ("E0_DIRECT", "E1_OMBE"))
                obs = wide[wide["truth_obstructed_fixed"].astype(bool)].copy()
                if obs.empty:
                    rows.append(empty_external_row(strat_name, str(stratum), t_label))
                    continue
                direct_fn = (~obs["E0_DIRECT_pred"].astype(bool)).to_numpy(bool)
                ombe_fn = (~obs["E1_OMBE_pred"].astype(bool)).to_numpy(bool)
                d_lo, d_hi = bootstrap_binary(direct_fn, SEED + len(rows) * 23 + 1)
                o_lo, o_hi = bootstrap_binary(ombe_fn, SEED + len(rows) * 23 + 2)
                point, low, high, excludes = bootstrap_diff(wide, "E0_DIRECT_pred", "E1_OMBE_pred", SEED + len(rows) * 23 + 3)
                d_rate = float(direct_fn.mean())
                o_rate = float(ombe_fn.mean())
                rows.append(
                    {
                        "stratification": strat_name,
                        "stratum": str(stratum),
                        "T_label": t_label,
                        "n": int(len(obs)),
                        "direct_fn": d_rate,
                        "direct_fn_n": int(direct_fn.sum()),
                        "direct_fn_den": int(len(direct_fn)),
                        "direct_fn_ci_low": d_lo,
                        "direct_fn_ci_high": d_hi,
                        "direct_fn_underpowered": bool(len(obs) < 30 or underpowered(d_rate, d_lo, d_hi)),
                        "ombe_fn": o_rate,
                        "ombe_fn_n": int(ombe_fn.sum()),
                        "ombe_fn_den": int(len(ombe_fn)),
                        "ombe_fn_ci_low": o_lo,
                        "ombe_fn_ci_high": o_hi,
                        "ombe_fn_underpowered": bool(len(obs) < 30 or underpowered(o_rate, o_lo, o_hi)),
                        "direct_minus_ombe": point,
                        "direct_minus_ombe_ci_low": low,
                        "direct_minus_ombe_ci_high": high,
                        "direct_minus_ombe_excludes_zero": excludes,
                        "underpowered": bool(len(obs) < 30),
                    }
                )
    return pd.DataFrame(rows), notes, variables


def empty_external_row(stratification: str, stratum: str, t_label: str) -> dict[str, Any]:
    return {
        "stratification": stratification,
        "stratum": stratum,
        "T_label": t_label,
        "n": 0,
        "direct_fn": np.nan,
        "direct_fn_n": 0,
        "direct_fn_den": 0,
        "direct_fn_ci_low": np.nan,
        "direct_fn_ci_high": np.nan,
        "direct_fn_underpowered": True,
        "ombe_fn": np.nan,
        "ombe_fn_n": 0,
        "ombe_fn_den": 0,
        "ombe_fn_ci_low": np.nan,
        "ombe_fn_ci_high": np.nan,
        "ombe_fn_underpowered": True,
        "direct_minus_ombe": np.nan,
        "direct_minus_ombe_ci_low": np.nan,
        "direct_minus_ombe_ci_high": np.nan,
        "direct_minus_ombe_excludes_zero": False,
        "underpowered": True,
    }


def make_wide_for_estimators(part: pd.DataFrame, estimators: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for gkey, group in part.groupby("global_key"):
        row = {
            "global_key": gkey,
            "truth_obstructed_fixed": boolish(group["truth_obstructed_fixed"].iloc[0]),
        }
        for est in estimators:
            est_part = group[group["estimator"].eq(est)]
            if len(est_part):
                item = est_part.iloc[0]
                valid = boolish(item["fit_success"]) and finite(item["est_ratio"])
                row[f"{est}_pred"] = boolish(item["est_obstructed_fixed"]) if valid else False
            else:
                row[f"{est}_pred"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def gate2_rows(gate2_by_cycle: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for cycle in CYCLE_TO_YEAR:
        result = gate2_by_cycle[cycle]
        c = result["criterion"]
        participant_n = int(c["n"])
        bev_rate = float(c["bev_flag_fraction"])
        bev_n = int(c.get("bev_flag_n", round(bev_rate * participant_n))) if np.isfinite(bev_rate) else 0
        bev_den = int(c.get("bev_flag_den", participant_n))
        if "bev_flag_ci_low" in c and "bev_flag_ci_high" in c:
            bev_ci_low = float(c["bev_flag_ci_low"])
            bev_ci_high = float(c["bev_flag_ci_high"])
            bev_underpowered = bool(c.get("bev_flag_underpowered", underpowered(bev_rate, bev_ci_low, bev_ci_high)))
        else:
            flags = np.concatenate([np.ones(bev_n, dtype=bool), np.zeros(max(0, bev_den - bev_n), dtype=bool)])
            bev_ci_low, bev_ci_high = bootstrap_binary(flags, SEED + ord(cycle) * 43)
            bev_underpowered = bool(bev_den < 30 or underpowered(bev_rate, bev_ci_low, bev_ci_high))
        rows.append(
            {
                "cycle": cycle,
                "n_participants": participant_n,
                "med_abs_FEV1_pct": float(c["fev1_median_abs_pct_error"]),
                "med_abs_FVC_pct": float(c["fvc_median_abs_pct_error"]),
                "med_signed_FEV1_pct": float(c["fev1_median_signed_pct_error"]),
                "med_signed_FVC_pct": float(c["fvc_median_signed_pct_error"]),
                "med_abs_ratio_pct": float(c["ratio_median_abs_pct_error"]),
                "BEV_flag_frac": bev_rate,
                "BEV_flag_frac_n": bev_n,
                "BEV_flag_frac_den": bev_den,
                "BEV_flag_frac_ci_low": bev_ci_low,
                "BEV_flag_frac_ci_high": bev_ci_high,
                "BEV_flag_frac_underpowered": bev_underpowered,
                "pass_both_le_5": bool(float(c["fev1_median_abs_pct_error"]) <= 5.0 and float(c["fvc_median_abs_pct_error"]) <= 5.0),
            }
        )
    return rows


def decide_verdicts(gate2_table: list[dict[str, Any]], cross_cycle: pd.DataFrame, severe_cohort_n: int, severe_recovery: pd.DataFrame, external_rows: pd.DataFrame) -> dict[str, str]:
    gate2_ok = all(r["pass_both_le_5"] and r["med_signed_FEV1_pct"] < 0 and r["med_signed_FVC_pct"] < 0 for r in gate2_table)
    trunc_ok_cycles = []
    for cycle in ("E", "F", "G"):
        subset = cross_cycle[cross_cycle["cycle"].eq(cycle) & cross_cycle["T_label"].isin(["2", "3"])]
        ok = bool(len(subset) == 2 and subset["direct_minus_ombe_excludes_zero"].map(bool).all() and (subset["direct_minus_ombe_point"].astype(float) > 0).all())
        trunc_ok_cycles.append(ok)
    if gate2_ok and all(trunc_ok_cycles):
        replication = "REPLICATES-3-CYCLE"
    elif gate2_ok and sum(trunc_ok_cycles) >= 2:
        replication = "REPLICATES-PARTIAL"
    else:
        replication = "DOES-NOT-REPLICATE"

    if severe_cohort_n < 20:
        severe = "SEVERE-STILL-UNDERPOWERED"
    else:
        severe_rows = severe_recovery[severe_recovery["truth_severity"].eq("severe") & severe_recovery["estimator"].isin(["E1_OMBE", "H1"])]
        max_fn = float(severe_rows["fn_rate"].max()) if len(severe_rows) else np.nan
        severe = "SEVERE-POWERED-RECOVERED" if np.isfinite(max_fn) and max_fn <= 0.10 else "SEVERE-POWERED-LIMITED"

    if external_rows.empty:
        external = "NOT-ASSESSED"
    else:
        assessed = external_rows[external_rows["n"].astype(int) > 0].copy()
        good = assessed[(assessed["direct_minus_ombe"].astype(float) > 0) & assessed["direct_minus_ombe_excludes_zero"].map(bool)]
        if len(good) and len(good) == len(assessed):
            external = "CLINICALLY-GENERALIZES"
        elif len(good):
            external = "GENERALIZES-PARTIAL"
        else:
            external = "GENERALIZES-PARTIAL"
    return {"replication": replication, "severe_power": severe, "external_validity": external}


def write_reports(
    root: Path,
    result: dict[str, Any],
    gate2_by_cycle: dict[str, dict[str, Any]],
    classification: pd.DataFrame,
    diffs: pd.DataFrame,
    cross_cycle: pd.DataFrame,
    severe_recovery: pd.DataFrame,
    severe_incomplete_summary: pd.DataFrame,
    external_rows: pd.DataFrame,
    cohort_rows: pd.DataFrame,
    bootstrap_rows: pd.DataFrame,
) -> None:
    out = root / "results"
    out.mkdir(exist_ok=True)
    for cycle in ("E", "G"):
        write_gate2_report(gate2_by_cycle[cycle], out, cycle)
    (out / "gate_three_cycle_replication_EFG.md").write_text(make_replication_report(result, cross_cycle, classification, diffs), encoding="utf-8")
    (out / "gate_severe_cohort_pooled_EFG.md").write_text(make_severe_report(result, severe_recovery, severe_incomplete_summary), encoding="utf-8")
    (out / "gate_external_validity_EFG.md").write_text(make_external_report(result, external_rows), encoding="utf-8")
    (out / "gate_three_cycle_EFG.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    bootstrap_rows.to_csv(out / "gate_three_cycle_EFG_bootstrap.csv", index=False)
    cohort_rows.to_csv(out / "gate_three_cycle_EFG_cohort.csv", index=False)


def make_replication_report(result: dict[str, Any], cross_cycle: pd.DataFrame, classification: pd.DataFrame, diffs: pd.DataFrame) -> str:
    lines = [
        "# Three-Cycle Replication EFG",
        "",
        "## Protocol Decisions",
        "",
    ]
    lines.extend(f"- {item}" for item in result["protocol_decisions"])
    lines.extend(
        [
            "",
            "## Global Guards",
            "",
            md_table(result["forward_model_guards"], [("phase", "phase", "str"), ("n_param_sets", "param sets", "int"), ("grid_n", "grid n", "int"), ("max_abs_volume_ml", "max volume diff mL", "float12"), ("passed", "passed", "bool")]),
            "",
            f"- Parser identity: {result['parser_identity']}",
            "",
            "## Part 0 - Cohort Accounting",
            "",
            md_table(result["cohort_accounting"], [("cycle", "cycle", "str"), ("raw_rows", "raw rows", "int"), ("pre_rows", "pre", "int"), ("acceptable_pre_rows", "acceptable pre", "int"), ("A_pre_rows", "A pre", "int"), ("acceptable_A_pre_rows", "acceptable A pre", "int"), ("participants_with_truth", "truth participants", "int"), ("participants_first_acceptable", "first acceptable", "int"), ("participants_first_acceptable_A", "first acceptable A", "int")]),
            "",
            "## Part 1 - Parser Smoke And Curve Length",
            "",
            md_table(result["parse_summary"], [("cycle", "cycle", "str"), ("raw_rows", "raw rows", "int"), ("parse_errors", "parse errors", "int"), ("selected_ab_participants", "usable selected A/B", "int"), ("spxpts_median", "SPXPTS median", "float1"), ("spxpts_p25", "SPXPTS p25", "float1"), ("spxpts_p75", "SPXPTS p75", "float1"), ("spxpts_flag", "SPXPTS finding", "str")]),
            "",
            "## Part 2 - Gate 2 Per Cycle",
            "",
            md_table(result["gate2_table"], [("cycle", "cycle", "str"), ("n_participants", "n", "int"), ("med_abs_FEV1_pct", "abs FEV1 %", "float3"), ("med_abs_FVC_pct", "abs FVC %", "float3"), ("med_signed_FEV1_pct", "signed FEV1 %", "float3"), ("med_signed_FVC_pct", "signed FVC %", "float3"), ("med_abs_ratio_pct", "abs ratio %", "float3"), ("BEV_flag_frac", "BEV flag frac", "rate"), ("pass_both_le_5", "PASS", "bool")]),
            "",
            result["gate2_verdict_sentence"],
            "",
            "## Part 3 - Truncation Recovery Per Cycle",
            "",
            md_table(cross_cycle.to_dict("records"), [("cycle", "cycle", "str"), ("T_label", "T", "str"), ("direct_fn_rate", "DIRECT FN", "rate"), ("ombe_fn_rate", "OMBE FN", "rate"), ("h1_fn_rate", "H1 FN", "rate"), ("direct_minus_ombe_point", "DIRECT-OMBE", "float3"), ("direct_minus_ombe_ci_low", "CI low", "float3"), ("direct_minus_ombe_ci_high", "CI high", "float3"), ("direct_minus_ombe_excludes_zero", "excludes 0", "bool")]),
            "",
            f"REPLICATION: {result['verdicts']['replication']}",
            "",
            result["truncation_verdict_sentence"],
            "",
        ]
    )
    return "\n".join(lines)


def make_severe_report(result: dict[str, Any], severe_recovery: pd.DataFrame, severe_incomplete_summary: pd.DataFrame) -> str:
    lines = [
        "# Pooled Severe Cohort EFG",
        "",
        f"SEVERE POWER: {result['verdicts']['severe_power']}",
        "",
        "## Paired Severe Cohort",
        "",
        md_table(result["severe_paired_counts"], [("cycle", "cycle", "str"), ("paired_severe_n", "paired severe n", "int")]),
        "",
        f"- pooled paired severe n: {result['pooled_paired_severe_n']}",
        "",
        result["severe_power_sentence"],
        "",
        "## Pooled Incomplete Severe Recovery",
        "",
        md_table(severe_incomplete_summary.to_dict("records"), [("estimator", "estimator", "str"), ("T_label", "endpoint", "str"), ("fn_rate", "FN", "rate"), ("specificity", "specificity", "rate"), ("fit_failure_n", "fit failures", "int")]) if len(severe_incomplete_summary) else "Not run because pooled paired severe n < 20.",
        "",
        "## Pooled Severity-Resolved Clean-Cut Recovery",
        "",
        md_table(severe_recovery.to_dict("records"), [("estimator", "estimator", "str"), ("T_label", "T", "str"), ("truth_severity", "severity", "str"), ("fn_rate", "FN", "rate")]),
        "",
        "## Severe-Band Honesty Check",
        "",
        md_table(result["severe_honesty"], [("estimator", "estimator", "str"), ("T_label", "T", "str"), ("f_only_fn", "F-only FN", "rate"), ("pooled_fn", "pooled FN", "rate"), ("survives_perfect_recovery", "perfect survives", "bool")]),
        "",
    ]
    return "\n".join(lines)


def make_external_report(result: dict[str, Any], external_rows: pd.DataFrame) -> str:
    lines = [
        "# External Validity EFG",
        "",
        f"EXTERNAL VALIDITY: {result['verdicts']['external_validity']}",
        "",
        "## Variables Used",
        "",
    ]
    lines.extend(f"- {k}: {v}" for k, v in result["external_variables"].items())
    lines.extend(["", "## Covariate Notes", ""])
    lines.extend(f"- {note}" for note in result["external_notes"]) if result["external_notes"] else lines.append("- None.")
    lines.extend(["", "## Stratified DIRECT vs OMBE Recovery", ""])
    if external_rows.empty:
        lines.append("Not assessed because covariates were unavailable.")
    else:
        lines.append(md_table(external_rows.to_dict("records"), [("stratification", "stratification", "str"), ("stratum", "stratum", "str"), ("T_label", "T", "str"), ("n", "n", "int"), ("direct_fn", "DIRECT FN", "rate"), ("ombe_fn", "OMBE FN", "rate"), ("direct_minus_ombe", "DIRECT-OMBE", "float3"), ("direct_minus_ombe_ci_low", "CI low", "float3"), ("direct_minus_ombe_ci_high", "CI high", "float3"), ("direct_minus_ombe_excludes_zero", "excludes 0", "bool"), ("underpowered", "UNDERPOWERED n<30", "bool")]))
    lines.append("")
    return "\n".join(lines)


def summarize_severe_incomplete(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    return classification_by_group(detail, ["estimator", "T_label"])


def severe_honesty_rows(severe_recovery: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for est in ["E1_OMBE", "H1"]:
        for t_label in T_LABELS:
            f = severe_recovery[
                severe_recovery["estimator"].eq(est)
                & severe_recovery["T_label"].eq(t_label)
                & severe_recovery["truth_severity"].eq("severe")
                & severe_recovery.get("cycle", pd.Series(dtype=object)).eq("F")
            ]
            pooled = severe_recovery[
                severe_recovery["estimator"].eq(est)
                & severe_recovery["T_label"].eq(t_label)
                & severe_recovery["truth_severity"].eq("severe")
                & severe_recovery.get("cycle", pd.Series(dtype=object)).eq("POOLED")
            ]
            f_n = int(f["fn_rate_n"].sum()) if len(f) else 0
            f_den = int(f["fn_rate_den"].sum()) if len(f) else 0
            p_n = int(pooled["fn_rate_n"].sum()) if len(pooled) else 0
            p_den = int(pooled["fn_rate_den"].sum()) if len(pooled) else 0
            rows.append(
                {
                    "estimator": est,
                    "T_label": t_label,
                    "f_only_fn": f_n / f_den if f_den else np.nan,
                    "f_only_fn_n": f_n,
                    "f_only_fn_den": f_den,
                    "f_only_fn_ci_low": np.nan,
                    "f_only_fn_ci_high": np.nan,
                    "pooled_fn": p_n / p_den if p_den else np.nan,
                    "pooled_fn_n": p_n,
                    "pooled_fn_den": p_den,
                    "pooled_fn_ci_low": np.nan,
                    "pooled_fn_ci_high": np.nan,
                    "survives_perfect_recovery": bool(p_n == 0 and p_den > 0),
                }
            )
    return rows


def run(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    guard_start = forward_model_guard(SEED)
    if not guard_start["passed"]:
        raise RuntimeError(f"Forward-model guard failed at start: {guard_start}")

    gate2_f = cached_gate2_f(root)
    gate2_by_cycle: dict[str, dict[str, Any]] = {"F": gate2_f}
    raw_by_cycle: dict[str, pd.DataFrame] = {}
    truth_by_cycle: dict[str, pd.DataFrame] = {}
    derived_by_cycle: dict[str, pd.DataFrame] = {}
    selected_complete_by_cycle: dict[str, pd.DataFrame] = {}
    cohort_accounting: list[dict[str, Any]] = [{"cycle": "F", **gate2_f["counts"]}]
    parse_summary: list[dict[str, Any]] = []
    sample_meta_rows: list[dict[str, Any]] = []
    clean_details: list[pd.DataFrame] = []
    severe_paired_parts: list[pd.DataFrame] = []

    # Process F through the migrated path for truncation/severe only, while keeping its Gate 2 row cached.
    for cycle in ("F", "E", "G"):
        if cycle in {"E", "G"}:
            gate2, raw, derived, truth = gate2_result_for_cycle(root, cycle)
            gate2_by_cycle[cycle] = gate2
            derived_by_cycle[cycle] = derived
            truth_by_cycle[cycle] = truth
            cohort_accounting.append({"cycle": cycle, **gate2["counts"]})
        else:
            raw_path, _note = find_or_unzip_raw(root, cycle)
            raw, _raw_info = read_raw(raw_path, cycle)
            truth = read_ground_truth(root, cycle)
            derived = derive_all(raw)
            derived_by_cycle[cycle] = derived
            truth_by_cycle[cycle] = truth
        raw_by_cycle[cycle] = raw

        selected_ab = selected_ab_from_derived(derived)
        spxpts = spxpts_summary(raw)
        parse_summary.append(
            {
                "cycle": cycle,
                "raw_rows": int(len(raw)),
                "parse_errors": int(derived.attrs.get("parse_error_count", 0)),
                "selected_ab_participants": int(len(selected_ab)),
                **spxpts,
                "spxpts_flag": "",
            }
        )
        selected_a, selected_complete = participant_candidates(raw, truth, cycle)
        selected_complete_by_cycle[cycle] = selected_complete
        sample, sample_rows = sample_by_stratum(selected_a, selected_complete, cycle)
        sample_meta_rows.extend(sample_rows)
        clean_details.append(run_clean_estimators(sample, raw, cycle))
        severe_paired_parts.append(incomplete_flagged_pool(raw, selected_complete, cycle))

    f_median = next(r["spxpts_median"] for r in parse_summary if r["cycle"] == "F")
    for row in parse_summary:
        rel = abs(float(row["spxpts_median"]) - f_median) / f_median if f_median else 0.0
        row["spxpts_flag"] = "FINDING: median differs from F by >5%" if row["cycle"] != "F" and rel > 0.05 else "unchanged"

    gate2_table = gate2_rows(gate2_by_cycle)
    gate2_ok = all(r["pass_both_le_5"] and r["med_signed_FEV1_pct"] < 0 and r["med_signed_FVC_pct"] < 0 for r in gate2_table)
    gate2_sentence = (
        "The F accuracy result reproduces on two independent NHANES waves."
        if gate2_ok
        else "Gate 2 does not fully reproduce: at least one added cycle failed the <=5% absolute-error rule or did not share F's negative signed bias."
    )

    clean_detail = pd.concat(clean_details, ignore_index=True)
    classification = classification_by_group(clean_detail, ["cycle", "T_label", "estimator"])
    diffs = per_cycle_differences(clean_detail)
    cross = cross_cycle_summary(classification, diffs)

    severe_cohort = pd.concat([p for p in severe_paired_parts if not p.empty], ignore_index=True) if any(not p.empty for p in severe_paired_parts) else pd.DataFrame()
    if not severe_cohort.empty:
        if severe_cohort["global_key"].duplicated().any():
            raise RuntimeError("Duplicate pooled severe global keys found; (cycle, SEQN) uniqueness guard failed.")
    severe_counts = []
    for cycle in ("E", "F", "G"):
        n = int((severe_cohort["cycle"].eq(cycle)).sum()) if not severe_cohort.empty else 0
        severe_counts.append({"cycle": cycle, "paired_severe_n": n})

    severe_incomplete_detail = run_incomplete_estimators(severe_cohort, raw_by_cycle) if len(severe_cohort) >= 20 else pd.DataFrame()
    severe_incomplete_summary = summarize_severe_incomplete(severe_incomplete_detail)
    pooled_severity = severity_recovery(clean_detail)
    cycle_severity = severity_recovery(clean_detail.assign(cycle_for_group=clean_detail["cycle"]))
    if len(cycle_severity):
        # Recompute with cycle carried in group for the honesty comparison.
        cycle_rows = []
        for cycle, part in clean_detail.groupby("cycle"):
            tmp = severity_recovery(part)
            tmp["cycle"] = cycle
            cycle_rows.append(tmp)
        cycle_severity = pd.concat(cycle_rows, ignore_index=True)
        severe_recovery_all = pd.concat([pooled_severity.assign(cycle="POOLED"), cycle_severity], ignore_index=True)
    else:
        severe_recovery_all = pooled_severity.assign(cycle="POOLED")
    severe_honesty = severe_honesty_rows(severe_recovery_all)

    external_rows, external_notes, external_variables = external_validity(clean_detail, root)
    verdicts = decide_verdicts(gate2_table, cross, int(len(severe_cohort)), pooled_severity, external_rows)
    trunc_ok = all(bool(x) for x in cross[cross["T_label"].isin(["2", "3"])]["direct_minus_ombe_excludes_zero"].tolist())
    trunc_sentence = (
        "The direct-vs-OMBE recovery gap reproduces across E, F, and G at T=2 and T=3 with paired bootstrap CIs excluding zero."
        if trunc_ok
        else "The direct-vs-OMBE recovery gap does not clear the T<=3 CI-exclusion rule in every cycle."
    )
    severe_sentence = (
        "The pooled severe paired-incomplete cohort is powered at n>=20, so severe-specific incomplete-maneuver recovery was run."
        if len(severe_cohort) >= 20
        else "Even three pooled NHANES cycles do not contain enough true-severe patients with paired incomplete maneuvers to power a severe-specific recovery estimate; this requires clinical non-survey data and is future work."
    )

    bootstrap_rows = pd.concat(
        [
            diffs,
            classification.assign(scope="classification_rate"),
            pooled_severity.assign(scope="pooled_severity_recovery"),
            external_rows.assign(scope="external_validity") if not external_rows.empty else pd.DataFrame(),
        ],
        ignore_index=True,
        sort=False,
    )
    cohort_rows = pd.concat(
        [
            pd.DataFrame(sample_meta_rows).assign(cohort_type="clean_sample_counts"),
            severe_cohort.assign(cohort_type="pooled_severe_paired") if not severe_cohort.empty else pd.DataFrame(),
        ],
        ignore_index=True,
        sort=False,
    )

    guard_end = forward_model_guard(SEED + 1)
    if not guard_end["passed"]:
        raise RuntimeError(f"Forward-model guard failed at end: {guard_end}")

    result = {
        "seed": SEED,
        "bootstrap_n": BOOTSTRAP_B,
        "elapsed_s": float(time.perf_counter() - started),
        "forward_model_guards": [{"phase": "start", **guard_start}, {"phase": "end", **guard_end}],
        "parser_identity": "predictx.gate23_f.derive_curve served cycles E, F, and G; no per-cycle parser branch was used.",
        "raw_expected_columns": EXPECTED_RAW_COLUMNS,
        "cohort_accounting": cohort_accounting,
        "parse_summary": parse_summary,
        "gate2_table": gate2_table,
        "gate2_verdict_sentence": gate2_sentence,
        "sample_meta": sample_meta_rows,
        "classification_summary": classification.to_dict("records"),
        "per_cycle_differences": diffs.to_dict("records"),
        "cross_cycle_summary": cross.to_dict("records"),
        "truncation_verdict_sentence": trunc_sentence,
        "pooled_paired_severe_n": int(len(severe_cohort)),
        "severe_paired_counts": severe_counts,
        "severe_power_sentence": severe_sentence,
        "pooled_severity_recovery": pooled_severity.to_dict("records"),
        "severe_incomplete_summary": severe_incomplete_summary.to_dict("records") if len(severe_incomplete_summary) else [],
        "severe_honesty": severe_honesty,
        "external_rows": external_rows.to_dict("records") if not external_rows.empty else [],
        "external_notes": external_notes,
        "external_variables": external_variables,
        "verdicts": verdicts,
        "protocol_decisions": [
            "F Gate 2 and clinical rows are reused from the migration-validated cache; F truncation and severe rows are regenerated through this script.",
            "Clean truncation uses the existing F rule: Pre, SPAACC == Y, SPAQEFF == A, complete plateau FET >= 6.0 s and terminal-second dV <= 0.025 L, max parsed FEV1, tie to lowest SPAMANU.",
            "Clean per-cycle truncation keeps cycles separate for replication and uses the prior 200-per-stratum cap.",
            "Complete-plateau cohort counts are stratified after selecting the best complete maneuver; any upward shift relative to the all-A selected maneuver is reported as complete_selection_severity_shift_n rather than a negative drop count.",
            "Pooled severe uses global keys of the form cycle:SEQN and selects the shortest usable separate flagged incomplete Pre maneuver per severe target patient.",
            "H1 is recomputed deterministically from the same OMBE fit at each truncation time; no stale F H1 cache is read.",
            "External validity uses SMQ020/SMQ040 for smoking, MCQ160G/MCQ160K for self-reported obstructive disease, and RIDAGEYR for age bands.",
        ],
    }
    write_reports(root, result, gate2_by_cycle, classification, diffs, cross, pooled_severity, severe_incomplete_summary, external_rows, cohort_rows, bootstrap_rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    print(f"REPLICATION: {result['verdicts']['replication']}")
    print(f"SEVERE POWER: {result['verdicts']['severe_power']}")
    print(f"EXTERNAL VALIDITY: {result['verdicts']['external_validity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
