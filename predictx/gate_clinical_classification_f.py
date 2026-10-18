from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .gate3_diagnosis import clinical_metrics_from_volume, fit_ombe_volume_lm
from .gate23_f import (
    HZ,
    backextrap_t0,
    detect_forced_exhale_segment,
    find_or_unzip_raw,
    parse_increments,
    read_raw,
)

RANDOM_SEED = 20260708
NORMAL_SAMPLE_MAX = 300


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def q(values: pd.Series | np.ndarray, probs: tuple[float, float] = (25, 75)) -> list[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan"), float("nan")]
    return [float(x) for x in np.percentile(arr, probs)]


def decode_grade(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_column_map(root: Path) -> dict:
    path = root / "results" / "column_map.json"
    if not path.exists():
        raise RuntimeError(f"Missing column map: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_truth(root: Path, column_map: dict, cycle: str = "F") -> pd.DataFrame:
    cycle = cycle.upper()
    path = root / "results" / "merged_summary.parquet"
    if not path.exists():
        raise RuntimeError(f"Missing merged summary parquet: {path}")
    df = pd.read_parquet(path)
    required = ["SEQN", "cycle", "fev1_mL", "fvc_mL", "fev1_fvc_ratio", "effort_quality", "fev1_quality"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"merged_summary.parquet missing required canonical columns: {missing}")
    out = df[df["cycle"] == cycle].copy()
    out["effort_quality"] = out["effort_quality"].map(decode_grade)
    out["fev1_quality"] = out["fev1_quality"].map(decode_grade)
    out["source_actual_fev1"] = column_map["fields"].get("fev1_mL", {}).get(cycle)
    out["source_actual_fvc"] = column_map["fields"].get("fvc_mL", {}).get(cycle)
    return out


def load_gli(root: Path) -> pd.DataFrame:
    path = root / "results" / "gli_reference.csv"
    if not path.exists():
        return pd.DataFrame(columns=["SEQN", "gli_lln_ratio"])
    df = pd.read_csv(path)
    if "SEQN" not in df.columns or "gli_lln_ratio" not in df.columns:
        raise RuntimeError(f"gli_reference.csv lacks SEQN/gli_lln_ratio: {path}")
    return df[["SEQN", "gli_lln_ratio", "gli_ethnicity_mapped", "gli_mapping_fallback"]].copy()


def extract_curve_for_fit(row: Any) -> dict:
    incr = parse_increments(row.SPXRAW)
    expected = int(row.SPXPTS)
    if len(incr) != expected:
        raise ValueError(f"SPXRAW length {len(incr)} != SPXPTS {expected}")
    fact = float(row.SPAFACT)
    flow = incr * 0.1 * fact
    volume_ml = np.cumsum(incr) * fact
    time = np.arange(len(incr), dtype=float) / HZ
    onset_idx, pef_idx, end_idx = detect_forced_exhale_segment(flow, hz=HZ)
    t0_s = backextrap_t0(time, flow, volume_ml, onset_idx, pef_idx)
    start_idx = max(0, int(np.searchsorted(time, max(t0_s, 0.0), side="left")))
    end_idx = max(end_idx, start_idx + 20)
    end_idx = min(end_idx, len(flow) - 1)

    v0_ml = float(np.interp(t0_s, time, volume_ml))
    vol_t = time[start_idx : end_idx + 1] - t0_s
    vol_l = (volume_ml[start_idx : end_idx + 1] - v0_ml) / 1000.0
    if vol_t[0] > 1e-9:
        vol_t = np.concatenate([[0.0], vol_t])
        vol_l = np.concatenate([[0.0], vol_l])
    else:
        vol_l = vol_l - vol_l[0]
        vol_t = vol_t - vol_t[0]
    if len(vol_t) < 30 or vol_t[-1] <= 0:
        raise ValueError("curve segment too short for volume fit")
    parsed_fev1_l = float(np.interp(min(1.0, vol_t[-1]), vol_t, vol_l))
    parsed_fvc_l = float(vol_l[-1])
    return {
        "vol_t": vol_t,
        "vol_l": vol_l,
        "parsed_fev1_l": parsed_fev1_l,
        "parsed_fvc_l": parsed_fvc_l,
        "parsed_ratio": parsed_fev1_l / parsed_fvc_l if parsed_fvc_l > 0 else np.nan,
        "t0_s": float(t0_s),
        "duration_s": float(vol_t[-1]),
    }


def build_usable_raw_candidates(raw: pd.DataFrame) -> pd.DataFrame:
    filtered = raw[
        (raw["SPATTYPE"].map(decode_grade) == "Pre")
        & (raw["SPAACC"].map(decode_grade) == "Y")
        & (raw["SPAQEFF"].map(decode_grade).isin(["A", "B"]))
    ].copy()
    rows: list[dict] = []
    for idx, row in enumerate(filtered.itertuples(index=True)):
        base = {
            "raw_index": int(row.Index),
            "SEQN": float(row.SEQN),
            "SPAMANU": float(row.SPAMANU),
            "SPAQEFF": decode_grade(row.SPAQEFF),
            "SPAACC": decode_grade(row.SPAACC),
            "SPXPTS": int(row.SPXPTS),
            "parse_error": "",
        }
        try:
            curve = extract_curve_for_fit(row)
            base.update(
                {
                    "parsed_fev1_l": curve["parsed_fev1_l"],
                    "parsed_fvc_l": curve["parsed_fvc_l"],
                    "parsed_ratio": curve["parsed_ratio"],
                    "t0_s": curve["t0_s"],
                    "duration_s": curve["duration_s"],
                }
            )
        except Exception as exc:
            base["parse_error"] = str(exc)
        rows.append(base)
        if (idx + 1) % 5000 == 0 or idx + 1 == len(filtered):
            print(f"Clinical classification raw parse progress: {idx + 1}/{len(filtered)} candidate rows")
    return pd.DataFrame(rows)


def select_best_manoeuvre(candidates: pd.DataFrame) -> pd.DataFrame:
    usable = candidates[candidates["parse_error"].fillna("") == ""].copy()
    usable = usable[np.isfinite(usable["parsed_fev1_l"]) & np.isfinite(usable["parsed_fvc_l"])]
    usable = usable[usable["parsed_fvc_l"] > 0]
    usable = usable.sort_values(["SEQN", "parsed_fev1_l", "SPAMANU"], ascending=[True, False, True])
    return usable.groupby("SEQN", as_index=False).first()


def severity_category(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "missing"
    if ratio < 0.50:
        return "severe"
    if ratio < 0.60:
        return "moderate"
    if ratio < 0.70:
        return "mild_0.60_0.70"
    return "normal"


def fit_selected_rows(raw: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    raw_by_index = {int(idx): row for idx, row in raw.iterrows()}
    rows: list[dict] = []
    total = len(selected)
    for i, row in enumerate(selected.itertuples(index=False), start=1):
        base = row._asdict()
        try:
            raw_row = raw_by_index[int(row.raw_index)]
            curve = extract_curve_for_fit(raw_row)
            fit = fit_ombe_volume_lm(curve["vol_t"], curve["vol_l"])
            model_fev1_l, model_fvc_l, model_ratio = clinical_metrics_from_volume(curve["vol_t"], fit["pred_volume_l"])
            base.update(
                {
                    "fit_error": "",
                    "model_fev1_mL": model_fev1_l * 1000.0,
                    "model_fvc_mL": model_fvc_l * 1000.0,
                    "model_ratio": model_ratio,
                    "volume_fit_success": fit["success"],
                    "volume_r2": fit["volume_r2"],
                    "volume_nrss_norm": fit["volume_nrss_norm"],
                    "fit_tau_f": float(fit["params"][0]),
                    "fit_tau_s": float(fit["params"][1]),
                    "fit_alpha": float(fit["params"][2]),
                    "fit_fvc_l": float(fit["params"][3]),
                    "fit_tau_o": float(fit["params"][4]),
                }
            )
        except Exception as exc:
            base["fit_error"] = str(exc)
        rows.append(base)
        if i % 50 == 0 or i == total:
            print(f"Clinical classification volume-fit progress: {i}/{total} participants")
    return pd.DataFrame(rows)


def confusion_binary(truth: pd.Series, pred: pd.Series) -> dict:
    truth = truth.astype(bool)
    pred = pred.astype(bool)
    tp = int((truth & pred).sum())
    tn = int((~truth & ~pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    n = tp + tn + fp + fn
    po = (tp + tn) / n if n else np.nan
    truth_pos = (tp + fn) / n if n else np.nan
    truth_neg = (tn + fp) / n if n else np.nan
    pred_pos = (tp + fp) / n if n else np.nan
    pred_neg = (tn + fn) / n if n else np.nan
    pe = truth_pos * pred_pos + truth_neg * pred_neg if n else np.nan
    kappa = (po - pe) / (1.0 - pe) if n and abs(1.0 - pe) > 1e-12 else np.nan
    return {
        "n": int(n),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "agreement": po,
        "cohens_kappa": kappa,
    }


def confusion_table(df: pd.DataFrame, truth_col: str, pred_col: str) -> pd.DataFrame:
    labels = [False, True]
    table = pd.crosstab(df[truth_col], df[pred_col], dropna=False).reindex(index=labels, columns=labels, fill_value=0)
    table.index = ["NHANES normal", "NHANES obstructed"]
    table.columns = ["model normal", "model obstructed"]
    return table


def summarize_errors(obstructed: pd.DataFrame) -> dict:
    valid = obstructed.dropna(subset=["model_fev1_mL", "model_fvc_mL", "model_ratio", "fev1_mL", "fvc_mL", "fev1_fvc_ratio"]).copy()
    valid = valid[(valid["fev1_mL"] > 0) & (valid["fvc_mL"] > 0)]
    fev1_abs_pct = np.abs(100.0 * (valid["model_fev1_mL"] - valid["fev1_mL"]) / valid["fev1_mL"])
    fvc_abs_pct = np.abs(100.0 * (valid["model_fvc_mL"] - valid["fvc_mL"]) / valid["fvc_mL"])
    ratio_abs = np.abs(valid["model_ratio"] - valid["fev1_fvc_ratio"])
    signed = valid["model_ratio"] - valid["fev1_fvc_ratio"]
    signed_iqr = q(signed)
    return {
        "n": int(len(valid)),
        "fev1_abs_pct_median": float(np.nanmedian(fev1_abs_pct)) if len(valid) else np.nan,
        "fev1_abs_pct_iqr": q(fev1_abs_pct),
        "fvc_abs_pct_median": float(np.nanmedian(fvc_abs_pct)) if len(valid) else np.nan,
        "fvc_abs_pct_iqr": q(fvc_abs_pct),
        "ratio_abs_median": float(np.nanmedian(ratio_abs)) if len(valid) else np.nan,
        "ratio_abs_iqr": q(ratio_abs),
        "ratio_signed_median": float(np.nanmedian(signed)) if len(valid) else np.nan,
        "ratio_signed_iqr": signed_iqr,
        "ratio_signed_iqr_width": float(signed_iqr[1] - signed_iqr[0]) if np.all(np.isfinite(signed_iqr)) else np.nan,
    }


def make_report(result: dict, rows: pd.DataFrame, fixed_table: pd.DataFrame, lln_table: pd.DataFrame, severity_table: pd.DataFrame) -> str:
    cycle = result["cycle"]
    e = result["errors_obstructed"]
    lines = [
        f"# Gate - Clinical Classification Accuracy, Cycle {cycle}",
        "",
        f"- VERDICT: {result['verdict']} - {result['verdict_reason']}",
        "- model tested: plain bi-exponential OMBE fit in the volume domain only",
        "- excluded methods: gamma onset, time registration, ML residual correction",
        f"- random seed: {result['random_seed']}",
        f"- raw source: {result['raw_source']}",
        f"- raw loaded rows: {result['raw_loaded_rows']}",
        f"- source mapping: `merged_summary.parquet` canonical fields resolved from `results/column_map.json`; cycle-{cycle} source variables are "
        f"`{result['source_actual_fev1']}` for FEV1 and `{result['source_actual_fvc']}` for FVC.",
        "",
        "## Manoeuvre Selection",
        "",
        result["selection_rule"],
        "",
        "Conservative tie rule: if derived FEV1 ties, the lowest manoeuvre number is selected.",
        "",
        "## Sample Construction",
        "",
        "| quantity | n |",
        "|---|---:|",
        f"| Tier-2 cycle-{cycle} severe truth participants (<0.50) | {result['truth_severe_n']} |",
        f"| Tier-2 cycle-{cycle} moderate truth participants (0.50-0.60) | {result['truth_moderate_n']} |",
        f"| severe with usable selected A/B raw curve | {result['selected_severe_n']} |",
        f"| moderate with usable selected A/B raw curve | {result['selected_moderate_n']} |",
        f"| severe dropped for missing/unusable raw curve | {result['dropped_severe_n']} |",
        f"| moderate dropped for missing/unusable raw curve | {result['dropped_moderate_n']} |",
        f"| normal candidates with ratio >0.70 and usable selected A/B raw curve | {result['normal_available_n']} |",
        f"| normal sampled | {result['normal_sample_n']} |",
        f"| participants fit attempted | {result['fit_attempted_n']} |",
        f"| participants fit valid | {result['fit_valid_n']} |",
        "",
        "## A. Fixed 0.70 Classification",
        "",
        fixed_table.to_markdown(),
        "",
        "| metric | value |",
        "|---|---:|",
        f"| sensitivity | {result['fixed']['sensitivity']:.3f} |",
        f"| specificity | {result['fixed']['specificity']:.3f} |",
        f"| overall agreement | {result['fixed']['agreement']:.3f} |",
        f"| Cohen's kappa | {result['fixed']['cohens_kappa']:.3f} |",
        "",
        "## B. GLI LLN Classification",
        "",
        f"- valid GLI rows: {result['lln_valid_n']}/{result['fit_valid_n']}",
        f"- excluded fraction: {result['lln_excluded_fraction']:.3f}",
        "",
        lln_table.to_markdown() if result["lln_valid_n"] else "No valid GLI LLN rows.",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| sensitivity | {result['lln']['sensitivity']:.3f} |",
        f"| specificity | {result['lln']['specificity']:.3f} |",
        f"| overall agreement | {result['lln']['agreement']:.3f} |",
        f"| Cohen's kappa | {result['lln']['cohens_kappa']:.3f} |",
        "",
        "## C. Severity Confusion",
        "",
        "Truth rows use the requested strata. Model columns include `mild_0.60_0.70` explicitly because fitted ratios can land in the gap between moderate and normal.",
        "",
        severity_table.to_markdown(),
        "",
        f"- severe cases demoted to moderate-or-higher: {result['severe_demoted_moderate_or_higher_n']}/{result['severe_valid_n']} ({result['severe_demoted_moderate_or_higher_rate']:.3f})",
        f"- severe cases demoted to normal: {result['severe_demoted_normal_n']}/{result['severe_valid_n']} ({result['severe_demoted_normal_rate']:.3f})",
        "",
        "## D. Obstructed-Set Diagnostic Error Split",
        "",
        "| metric | median | IQR |",
        "|---|---:|---:|",
        f"| FEV1 abs percent error | {e['fev1_abs_pct_median']:.2f}% | {e['fev1_abs_pct_iqr'][0]:.2f}% to {e['fev1_abs_pct_iqr'][1]:.2f}% |",
        f"| FVC abs percent error | {e['fvc_abs_pct_median']:.2f}% | {e['fvc_abs_pct_iqr'][0]:.2f}% to {e['fvc_abs_pct_iqr'][1]:.2f}% |",
        f"| FEV1/FVC absolute error | {e['ratio_abs_median']:.4f} | {e['ratio_abs_iqr'][0]:.4f} to {e['ratio_abs_iqr'][1]:.4f} |",
        "",
        "## E. Signed Ratio Error",
        "",
        f"- median signed model-minus-NHANES FEV1/FVC error on obstructed set: {e['ratio_signed_median']:.4f}",
        f"- signed-error IQR: {e['ratio_signed_iqr'][0]:.4f} to {e['ratio_signed_iqr'][1]:.4f}",
        "- consistent-bias rule used for MARGINAL: abs(median) > 0.02, IQR width <= 0.04, and IQR does not cross zero.",
        f"- consistent bias detected: {result['consistent_bias']}",
        "",
        "## Gate Logic",
        "",
        "- ALIVE requires fixed-threshold obstruction sensitivity >= 0.90 and severe-case demotion-to-normal rate <= 0.05.",
        "- MARGINAL requires sensitivity 0.80-0.90 or a consistent signed ratio bias.",
        "- DEAD requires sensitivity < 0.80 with symmetric/non-correctable ratio error.",
        "",
        f"VERDICT: {result['verdict']} - {result['verdict_reason']}",
        "",
    ]
    return "\n".join(lines)


def run(root: Path, cycle: str = "F") -> dict:
    cycle = cycle.upper()
    results_dir = root / "results"
    column_map = load_column_map(root)
    truth = load_truth(root, column_map, cycle)
    gli = load_gli(root)

    raw_path, raw_note = find_or_unzip_raw(root, cycle)
    raw, raw_info = read_raw(raw_path, cycle)
    candidates = build_usable_raw_candidates(raw)
    selected = select_best_manoeuvre(candidates)

    tier2 = truth[
        truth["fev1_fvc_ratio"].notna()
        & truth["effort_quality"].isin(["A", "B"])
        & truth["fev1_quality"].isin(["A", "B"])
    ].copy()
    severe_truth = tier2[tier2["fev1_fvc_ratio"] < 0.50].copy()
    moderate_truth = tier2[(tier2["fev1_fvc_ratio"] >= 0.50) & (tier2["fev1_fvc_ratio"] < 0.60)].copy()
    obstructed_truth = pd.concat([severe_truth, moderate_truth], ignore_index=True)

    selected_truth = selected.merge(
        truth[["SEQN", "fev1_mL", "fvc_mL", "fev1_fvc_ratio", "effort_quality", "fev1_quality", "source_actual_fev1", "source_actual_fvc"]],
        on="SEQN",
        how="left",
    )

    obstructed_selected = selected_truth.merge(
        obstructed_truth[["SEQN"]].assign(sample_group="obstructed"),
        on="SEQN",
        how="inner",
    )
    obstructed_selected["truth_severity"] = obstructed_selected["fev1_fvc_ratio"].map(severity_category)

    normal_candidates = selected_truth[
        selected_truth["fev1_fvc_ratio"].notna() & (selected_truth["fev1_fvc_ratio"] > 0.70)
    ].copy()
    normal_sample_n = min(NORMAL_SAMPLE_MAX, len(normal_candidates))
    normal_selected = normal_candidates.sample(n=normal_sample_n, random_state=RANDOM_SEED).copy() if normal_sample_n else normal_candidates
    normal_selected["sample_group"] = "normal"
    normal_selected["truth_severity"] = "normal"

    selected_for_fit = pd.concat([obstructed_selected, normal_selected], ignore_index=True)
    fit_rows = fit_selected_rows(raw, selected_for_fit)
    fit_rows = fit_rows.merge(gli, on="SEQN", how="left")
    valid = fit_rows[fit_rows["fit_error"].fillna("") == ""].copy()
    valid = valid.dropna(subset=["model_ratio", "fev1_fvc_ratio"])
    valid["truth_obstructed_fixed"] = valid["fev1_fvc_ratio"] < 0.70
    valid["model_obstructed_fixed"] = valid["model_ratio"] < 0.70
    valid["truth_severity"] = valid["fev1_fvc_ratio"].map(severity_category)
    valid["model_severity"] = valid["model_ratio"].map(severity_category)

    fixed = confusion_binary(valid["truth_obstructed_fixed"], valid["model_obstructed_fixed"])
    fixed_table = confusion_table(valid, "truth_obstructed_fixed", "model_obstructed_fixed")

    lln_valid = valid[valid["gli_lln_ratio"].map(finite)].copy()
    if len(lln_valid):
        lln_valid["truth_obstructed_lln"] = lln_valid["fev1_fvc_ratio"] < lln_valid["gli_lln_ratio"]
        lln_valid["model_obstructed_lln"] = lln_valid["model_ratio"] < lln_valid["gli_lln_ratio"]
        lln = confusion_binary(lln_valid["truth_obstructed_lln"], lln_valid["model_obstructed_lln"])
        lln_table = confusion_table(lln_valid, "truth_obstructed_lln", "model_obstructed_lln")
    else:
        lln = {"n": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0, "sensitivity": np.nan, "specificity": np.nan, "agreement": np.nan, "cohens_kappa": np.nan}
        lln_table = pd.DataFrame()

    severity_order = ["severe", "moderate", "normal"]
    model_order = ["severe", "moderate", "mild_0.60_0.70", "normal"]
    severity_table = (
        pd.crosstab(valid["truth_severity"], valid["model_severity"], dropna=False)
        .reindex(index=severity_order, columns=model_order, fill_value=0)
    )

    obstructed_valid = valid[valid["sample_group"] == "obstructed"].copy()
    severe_valid = valid[valid["truth_severity"] == "severe"].copy()
    severe_n = len(severe_valid)
    severe_demoted_moderate_or_higher = int((severe_valid["model_ratio"] >= 0.50).sum()) if severe_n else 0
    severe_demoted_normal = int((severe_valid["model_ratio"] >= 0.70).sum()) if severe_n else 0
    errors = summarize_errors(obstructed_valid)
    signed_iqr = errors["ratio_signed_iqr"]
    iqr_width = errors["ratio_signed_iqr_width"]
    iqr_crosses_zero = bool(signed_iqr[0] <= 0.0 <= signed_iqr[1]) if np.all(np.isfinite(signed_iqr)) else True
    consistent_bias = bool(
        abs(errors["ratio_signed_median"]) > 0.02
        and np.isfinite(iqr_width)
        and iqr_width <= 0.04
        and not iqr_crosses_zero
    )

    sensitivity = fixed["sensitivity"]
    severe_demoted_normal_rate = severe_demoted_normal / severe_n if severe_n else np.nan
    if sensitivity >= 0.90 and severe_demoted_normal_rate <= 0.05:
        verdict = "ALIVE"
        reason = "obstruction sensitivity clears 0.90 and severe demotion-to-normal is <=0.05; onset failure does not propagate to the screening decision."
    elif (0.80 <= sensitivity < 0.90) or consistent_bias:
        verdict = "MARGINAL"
        reason = "classification is below ALIVE but within the marginal band or shows a consistent signed ratio bias."
    else:
        verdict = "DEAD"
        reason = "fixed-threshold sensitivity is <0.80 without a narrow one-sided ratio bias."

    result = {
        "cycle": cycle,
        "verdict": verdict,
        "verdict_reason": reason,
        "random_seed": RANDOM_SEED,
        "selection_rule": "Use Pre curves with `SPAACC == 'Y'` and raw manoeuvre grade `SPAQEFF in {'A','B'}`; parse each curve; select the per-participant manoeuvre with maximum parsed FEV1.",
        "raw_source": str(raw_path),
        "raw_source_note": raw_note,
        "raw_loaded_rows": int(raw_info["loaded_rows"]),
        "raw_candidate_rows_pre_acc_ab": int(len(candidates)),
        "raw_candidate_parse_errors": int((candidates["parse_error"].fillna("") != "").sum()),
        "selected_participants_with_usable_raw": int(len(selected)),
        "source_actual_fev1": column_map["fields"].get("fev1_mL", {}).get(cycle),
        "source_actual_fvc": column_map["fields"].get("fvc_mL", {}).get(cycle),
        "truth_severe_n": int(len(severe_truth)),
        "truth_moderate_n": int(len(moderate_truth)),
        "selected_severe_n": int((obstructed_selected["truth_severity"] == "severe").sum()),
        "selected_moderate_n": int((obstructed_selected["truth_severity"] == "moderate").sum()),
        "dropped_severe_n": int(len(severe_truth) - (obstructed_selected["truth_severity"] == "severe").sum()),
        "dropped_moderate_n": int(len(moderate_truth) - (obstructed_selected["truth_severity"] == "moderate").sum()),
        "normal_available_n": int(len(normal_candidates)),
        "normal_sample_n": int(len(normal_selected)),
        "fit_attempted_n": int(len(fit_rows)),
        "fit_valid_n": int(len(valid)),
        "fit_error_n": int(len(fit_rows) - len(valid)),
        "fixed": fixed,
        "lln": lln,
        "lln_valid_n": int(len(lln_valid)),
        "lln_excluded_fraction": float(1.0 - len(lln_valid) / len(valid)) if len(valid) else np.nan,
        "severe_valid_n": int(severe_n),
        "severe_demoted_moderate_or_higher_n": int(severe_demoted_moderate_or_higher),
        "severe_demoted_moderate_or_higher_rate": float(severe_demoted_moderate_or_higher / severe_n) if severe_n else np.nan,
        "severe_demoted_normal_n": int(severe_demoted_normal),
        "severe_demoted_normal_rate": float(severe_demoted_normal_rate),
        "errors_obstructed": errors,
        "consistent_bias": consistent_bias,
    }

    results_dir.mkdir(exist_ok=True)
    report = make_report(result, valid, fixed_table, lln_table, severity_table)
    (results_dir / f"gate_clinical_classification_{cycle}.md").write_text(report, encoding="utf-8")
    (results_dir / f"gate_clinical_classification_{cycle}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    valid.to_csv(results_dir / f"gate_clinical_classification_{cycle}_rows.csv", index=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--cycle", default="F", choices=("E", "F", "G"))
    args = parser.parse_args()
    result = run(Path(args.root).resolve(), args.cycle.upper())
    print(f"Clinical classification gate: {result['verdict']} - {result['verdict_reason']}")
    print(f"Fixed sensitivity={result['fixed']['sensitivity']:.3f}, specificity={result['fixed']['specificity']:.3f}")
    print(f"Severe demotion-to-normal={result['severe_demoted_normal_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
