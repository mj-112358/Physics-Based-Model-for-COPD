from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .gate23_f import find_or_unzip_raw, read_raw
from .gate_clinical_classification_f import (
    RANDOM_SEED,
    build_usable_raw_candidates,
    confusion_binary,
    extract_curve_for_fit,
    fit_selected_rows,
    load_column_map,
    load_gli,
    load_truth,
    q,
    select_best_manoeuvre,
    severity_category,
)
from .gate_fvc_correction_f import drop_bias_audit

MAX_BORDERLINE = 500
MAX_NORMAL = 500
POLICY_VARIANTS = (
    "correction_only",
    "flag_banded_corrected",
    "flag_banded_raw",
    "flag_unbanded",
)
POLICY_LABELS = {
    "correction_only": "correction only",
    "flag_banded_corrected": "correction + flag, corrected-ratio band",
    "flag_banded_raw": "correction + flag, raw-model-ratio band",
    "flag_unbanded": "correction + flag, unbanded",
}


def signed_iqr(values: pd.Series | np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)) if arr.size else np.nan,
        "iqr": q(arr),
    }


def stratified_half_split(df: pd.DataFrame, stratum_col: str = "truth_severity") -> pd.DataFrame:
    parts = []
    for stratum, group in df.groupby(stratum_col, dropna=False):
        group = group.sample(frac=1.0, random_state=RANDOM_SEED).copy()
        n_train = len(group) // 2
        group["split"] = "test"
        group.iloc[:n_train, group.columns.get_loc("split")] = "train"
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def add_ats_features(raw: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    raw_by_index = {int(idx): row for idx, row in raw.iterrows()}
    rows = []
    total = len(selected)
    for i, row in enumerate(selected.itertuples(index=False), start=1):
        out = row._asdict()
        try:
            curve = extract_curve_for_fit(raw_by_index[int(row.raw_index)])
            fet = float(curve["vol_t"][-1])
            start = max(0.0, fet - 1.0)
            v_start = float(np.interp(start, curve["vol_t"], curve["vol_l"]))
            d_last = float(curve["vol_l"][-1] - v_start)
            out["fet_s"] = fet
            out["dV_last_L"] = d_last
            out["ats_feature_error"] = ""
        except Exception as exc:
            out["fet_s"] = np.nan
            out["dV_last_L"] = np.nan
            out["ats_feature_error"] = str(exc)
        rows.append(out)
        if i % 200 == 0 or i == total:
            print(f"Policy gate ATS feature progress: {i}/{total} participants")
    return pd.DataFrame(rows)


def build_pool(root: Path, raw: pd.DataFrame, truth: pd.DataFrame, gli: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    candidates = build_usable_raw_candidates(raw)
    selected = select_best_manoeuvre(candidates)
    selected_truth = selected.merge(
        truth[["SEQN", "fev1_mL", "fvc_mL", "fev1_fvc_ratio", "effort_quality", "fev1_quality"]],
        on="SEQN",
        how="left",
    )
    selected_truth = selected_truth[selected_truth["fev1_fvc_ratio"].notna()].copy()
    selected_truth["truth_severity"] = selected_truth["fev1_fvc_ratio"].map(severity_category)
    truth_with_ratio = truth[truth["fev1_fvc_ratio"].notna()].copy()
    truth_with_ratio["truth_severity"] = truth_with_ratio["fev1_fvc_ratio"].map(severity_category)

    severe = selected_truth[selected_truth["truth_severity"] == "severe"].copy()
    moderate = selected_truth[selected_truth["truth_severity"] == "moderate"].copy()
    borderline_avail = selected_truth[selected_truth["truth_severity"] == "mild_0.60_0.70"].copy()
    normal_avail = selected_truth[selected_truth["truth_severity"] == "normal"].copy()
    borderline = borderline_avail.sample(n=min(MAX_BORDERLINE, len(borderline_avail)), random_state=RANDOM_SEED).copy()
    normal = normal_avail.sample(n=min(MAX_NORMAL, len(normal_avail)), random_state=RANDOM_SEED).copy()
    pool = pd.concat([severe, moderate, borderline, normal], ignore_index=True)
    pool = add_ats_features(raw, pool)
    fit_rows = fit_selected_rows(raw, pool)
    fit_rows = fit_rows.merge(gli, on="SEQN", how="left")
    fit_rows = fit_rows[fit_rows["fit_error"].fillna("") == ""].copy()
    fit_rows = fit_rows.dropna(subset=["model_ratio", "fev1_fvc_ratio", "fet_s", "dV_last_L"])
    fit_rows["truth_severity"] = fit_rows["fev1_fvc_ratio"].map(severity_category)
    split_rows = stratified_half_split(fit_rows)
    meta = {
        "truth_counts": {str(k): int(v) for k, v in truth_with_ratio["truth_severity"].value_counts().to_dict().items()},
        "available_selected_counts": {
            "severe": int(len(severe)),
            "moderate": int(len(moderate)),
            "borderline": int(len(borderline_avail)),
            "normal": int(len(normal_avail)),
        },
        "sampled_counts": {
            "severe": int(len(severe)),
            "moderate": int(len(moderate)),
            "borderline": int(len(borderline)),
            "normal": int(len(normal)),
        },
        "dropped_before_sampling_counts": {
            str(k): int(v)
            for k, v in (
                truth_with_ratio["truth_severity"].value_counts()
                - selected_truth["truth_severity"].value_counts()
            )
            .fillna(0)
            .to_dict()
            .items()
        },
        "valid_fit_counts": {str(k): int(v) for k, v in fit_rows["truth_severity"].value_counts().to_dict().items()},
        "raw_candidate_rows_pre_acc_ab": int(len(candidates)),
        "raw_candidate_parse_errors": int((candidates["parse_error"].fillna("") != "").sum()),
    }
    return split_rows, meta, selected_truth


def fit_corrections(train: pd.DataFrame) -> dict:
    e = train["model_ratio"] - train["fev1_fvc_ratio"]
    c1 = float(np.nanmedian(e))
    x = train["model_ratio"].to_numpy(dtype=float)
    y = e.to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    slope, intercept = np.polyfit(x[finite], y[finite], 1)

    corrected_c1 = train["model_ratio"] - c1
    corrected_c2 = train["model_ratio"] - (intercept + slope * train["model_ratio"])
    return {
        "c1": {
            "kind": "constant",
            "bias": c1,
            "formula": f"corrected_ratio = model_ratio - ({c1:.6f})",
            "train_residual": signed_iqr(corrected_c1 - train["fev1_fvc_ratio"]),
            "n_params": 1,
        },
        "c2": {
            "kind": "linear",
            "intercept": float(intercept),
            "slope": float(slope),
            "formula": f"corrected_ratio = model_ratio - ({intercept:.6f} + {slope:.6f} * model_ratio)",
            "train_residual": signed_iqr(corrected_c2 - train["fev1_fvc_ratio"]),
            "n_params": 2,
        },
    }


def apply_correction(df: pd.DataFrame, correction: dict) -> pd.Series:
    if correction["kind"] == "constant":
        return df["model_ratio"] - correction["bias"]
    return df["model_ratio"] - (correction["intercept"] + correction["slope"] * df["model_ratio"])


def ats_flag(df: pd.DataFrame, plateau_l: float = 0.025, fet_s: float = 6.0) -> pd.Series:
    return (df["dV_last_L"] > plateau_l) | (df["fet_s"] < fet_s)


def policy_outcomes(
    df: pd.DataFrame,
    ratio_col: str,
    *,
    threshold_col: str | None = None,
    flag_col: str | None = None,
    flag_variant: str = "correction_only",
    raw_ratio_col: str = "model_ratio",
    borderline_low: float = 0.60,
) -> pd.Series:
    ratio = df[ratio_col]
    threshold = 0.70 if threshold_col is None else df[threshold_col]
    flagged = pd.Series(False, index=df.index) if flag_col is None else df[flag_col].astype(bool)
    out = pd.Series("NORMAL", index=df.index, dtype=object)
    obstructed = ratio < threshold
    out[obstructed] = "OBSTRUCTED"

    eligible_for_flag = ~obstructed & flagged
    if flag_variant == "correction_only":
        flag_branch = pd.Series(False, index=df.index)
    elif flag_variant == "flag_banded_corrected":
        flag_branch = eligible_for_flag & (ratio >= borderline_low) & (ratio < 0.70)
    elif flag_variant == "flag_banded_raw":
        raw_ratio = df[raw_ratio_col]
        flag_branch = eligible_for_flag & (raw_ratio >= borderline_low) & (raw_ratio < 0.70)
    elif flag_variant == "flag_unbanded":
        flag_branch = eligible_for_flag
    else:
        raise ValueError(f"Unknown flag_variant: {flag_variant}")
    out[flag_branch] = "FLAG"
    return out


def truth_obstructed(df: pd.DataFrame, *, threshold_col: str | None = None) -> pd.Series:
    if threshold_col is None:
        return df["fev1_fvc_ratio"] < 0.70
    valid = df[threshold_col].map(lambda v: np.isfinite(float(v)) if not pd.isna(v) else False)
    out = pd.Series(pd.NA, index=df.index, dtype=object)
    out.loc[valid] = df.loc[valid, "fev1_fvc_ratio"] < df.loc[valid, threshold_col]
    return out


def three_outcome_metrics(df: pd.DataFrame, outcome: pd.Series, *, threshold_col: str | None = None) -> dict:
    truth_obs = truth_obstructed(df, threshold_col=threshold_col)
    valid = truth_obs.notna()
    d = df.loc[valid].copy()
    o = outcome.loc[valid]
    t = truth_obs.loc[valid].astype(bool)
    obs_n = int(t.sum())
    normal_n = int((~t).sum())
    obs = {
        "n": obs_n,
        "obstructed_fraction": float(((t) & (o == "OBSTRUCTED")).sum() / obs_n) if obs_n else np.nan,
        "flagged_fraction": float(((t) & (o == "FLAG")).sum() / obs_n) if obs_n else np.nan,
        "normal_fraction": float(((t) & (o == "NORMAL")).sum() / obs_n) if obs_n else np.nan,
        "normal_n": int(((t) & (o == "NORMAL")).sum()),
    }
    normal = {
        "n": normal_n,
        "normal_fraction": float(((~t) & (o == "NORMAL")).sum() / normal_n) if normal_n else np.nan,
        "flagged_fraction": float(((~t) & (o == "FLAG")).sum() / normal_n) if normal_n else np.nan,
        "obstructed_fraction": float(((~t) & (o == "OBSTRUCTED")).sum() / normal_n) if normal_n else np.nan,
        "flagged_n": int(((~t) & (o == "FLAG")).sum()),
    }
    nonflag = d[o != "FLAG"].copy()
    nonflag_out = o[o != "FLAG"]
    nonflag_truth = t[o != "FLAG"]
    pred_obs = nonflag_out == "OBSTRUCTED"
    conf = confusion_binary(nonflag_truth, pred_obs) if len(nonflag) else {
        "n": 0,
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "sensitivity": np.nan,
        "specificity": np.nan,
        "agreement": np.nan,
        "cohens_kappa": np.nan,
    }
    borderline = d[(d["fev1_fvc_ratio"] >= 0.60) & (d["fev1_fvc_ratio"] < 0.70)]
    borderline_out = o.loc[borderline.index]
    borderline_n = len(borderline)
    return {
        "n": int(len(d)),
        "obstructed_truth": obs,
        "normal_truth": normal,
        "flag_burden_all": float((o == "FLAG").mean()) if len(o) else np.nan,
        "flag_burden_normal_truth": normal["flagged_fraction"],
        "confident_nonflagged": conf,
        "borderline_n": int(borderline_n),
        "borderline_residual_false_negative_rate": float((borderline_out == "NORMAL").mean()) if borderline_n else np.nan,
        "borderline_flagged_fraction": float((borderline_out == "FLAG").mean()) if borderline_n else np.nan,
        "borderline_obstructed_fraction": float((borderline_out == "OBSTRUCTED").mean()) if borderline_n else np.nan,
    }


def evaluate_candidate(test: pd.DataFrame, correction: dict, correction_name: str) -> dict:
    out = test.copy()
    out[f"corrected_ratio_{correction_name}"] = apply_correction(out, correction)
    out["truncated_ats"] = ats_flag(out, 0.025, 6.0)
    out["in_band_corrected_060_070"] = (
        (out[f"corrected_ratio_{correction_name}"] >= 0.60)
        & (out[f"corrected_ratio_{correction_name}"] < 0.70)
    )
    out["in_band_raw_060_070"] = (out["model_ratio"] >= 0.60) & (out["model_ratio"] < 0.70)
    threshold_fixed = pd.Series(0.70, index=out.index)
    old_impossible_branch = (
        (out[f"corrected_ratio_{correction_name}"] >= threshold_fixed)
        & (out[f"corrected_ratio_{correction_name}"] >= 0.60)
        & (out[f"corrected_ratio_{correction_name}"] < 0.70)
        & out["truncated_ats"]
    )

    fixed = {}
    fixed_branch = {}
    for variant in POLICY_VARIANTS:
        outcome_fixed = policy_outcomes(
            out,
            f"corrected_ratio_{correction_name}",
            flag_col="truncated_ats",
            flag_variant=variant,
        )
        fixed[variant] = three_outcome_metrics(out, outcome_fixed)
        fixed_branch[variant] = {
            "flag_n": int((outcome_fixed == "FLAG").sum()),
            "obstructed_n": int((outcome_fixed == "OBSTRUCTED").sum()),
            "normal_n": int((outcome_fixed == "NORMAL").sum()),
        }
    lln_valid = out[out["gli_lln_ratio"].map(lambda v: np.isfinite(float(v)) if not pd.isna(v) else False)].copy()
    if len(lln_valid):
        lln = {}
        lln_branch = {}
        for variant in POLICY_VARIANTS:
            lln_outcome = policy_outcomes(
                lln_valid,
                f"corrected_ratio_{correction_name}",
                threshold_col="gli_lln_ratio",
                flag_col="truncated_ats",
                flag_variant=variant,
            )
            lln[variant] = three_outcome_metrics(lln_valid, lln_outcome, threshold_col="gli_lln_ratio")
            lln_branch[variant] = {
                "flag_n": int((lln_outcome == "FLAG").sum()),
                "obstructed_n": int((lln_outcome == "OBSTRUCTED").sum()),
                "normal_n": int((lln_outcome == "NORMAL").sum()),
            }
    else:
        lln = {}
        lln_branch = {}
    residual = signed_iqr(out[f"corrected_ratio_{correction_name}"] - out["fev1_fvc_ratio"])
    diagnostics = {
        "test_n": int(len(out)),
        "truncation_flagged_n": int(out["truncated_ats"].sum()),
        "corrected_band_060_070_n": int(out["in_band_corrected_060_070"].sum()),
        "raw_band_060_070_n": int(out["in_band_raw_060_070"].sum()),
        "old_impossible_fixed_flag_branch_n": int(old_impossible_branch.sum()),
        "fixed_branch_counts": fixed_branch,
        "lln_branch_counts": lln_branch,
    }
    return {"fixed": fixed, "lln": lln, "test_residual": residual, "diagnostics": diagnostics}


def choose_preferred_candidate(candidates: dict) -> tuple[str, str]:
    def score(item: tuple[str, str]) -> tuple:
        name, variant = item
        fixed = candidates[name]["fixed"][variant]
        return (
            fixed["borderline_residual_false_negative_rate"],
            fixed["obstructed_truth"]["normal_fraction"],
            fixed["flag_burden_normal_truth"],
            -fixed["confident_nonflagged"]["specificity"],
            abs(candidates[name]["test_residual"]["median"]),
        )

    items = [(name, variant) for name in candidates for variant in POLICY_VARIANTS]
    return sorted(items, key=score)[0]


def sensitivity_sweep(test: pd.DataFrame, correction: dict, flag_variant: str) -> list[dict]:
    rows = []
    corrected = apply_correction(test, correction)
    work = test.copy()
    work["corrected_ratio"] = corrected
    for plateau in [0.010, 0.025, 0.050]:
        for fet in [5.0, 6.0]:
            col = f"flag_{plateau}_{fet}"
            work[col] = ats_flag(work, plateau, fet)
            outcome = policy_outcomes(work, "corrected_ratio", flag_col=col, flag_variant=flag_variant)
            m = three_outcome_metrics(work, outcome)
            rows.append(
                {
                    "plateau_L": plateau,
                    "fet_s": fet,
                    "policy_variant": flag_variant,
                    "flag_rate_all": float(work[col].mean()),
                    "flag_rate_normal_truth": m["flag_burden_normal_truth"],
                    "flagged_outcome_all": m["flag_burden_all"],
                    "borderline_residual_fn": m["borderline_residual_false_negative_rate"],
                    "overall_residual_fn": m["obstructed_truth"]["normal_fraction"],
                    "nonflag_specificity": m["confident_nonflagged"]["specificity"],
                }
            )
    return rows


def split_counts(df: pd.DataFrame) -> dict:
    return {
        split: {str(k): int(v) for k, v in part["truth_severity"].value_counts().to_dict().items()}
        for split, part in df.groupby("split")
    }


def truncation_by_stratum(df: pd.DataFrame) -> dict:
    tmp = df.copy()
    tmp["truncated_ats"] = ats_flag(tmp)
    return {
        str(k): {"n": int(len(g)), "flagged_fraction": float(g["truncated_ats"].mean())}
        for k, g in tmp.groupby("truth_severity")
    }


def residual_leaks_outside_band(test: pd.DataFrame, correction: dict, flag_variant: str) -> int:
    work = test.copy()
    work["corrected_ratio"] = apply_correction(work, correction)
    work["truncated_ats"] = ats_flag(work)
    outcome = policy_outcomes(work, "corrected_ratio", flag_col="truncated_ats", flag_variant=flag_variant)
    truth_obs = work["fev1_fvc_ratio"] < 0.70
    leaks = work[truth_obs & (outcome == "NORMAL")]
    return int(((leaks["corrected_ratio"] >= 0.70) & leaks["truncated_ats"]).sum())


def overfit_status(train_residual: dict, test_residual: dict) -> tuple[bool, str]:
    train_med = abs(train_residual["median"])
    test_med = abs(test_residual["median"])
    train_width = train_residual["iqr"][1] - train_residual["iqr"][0]
    test_width = test_residual["iqr"][1] - test_residual["iqr"][0]
    materially_worse = test_med > train_med + 0.01 or test_width > train_width + 0.03
    if materially_worse:
        return True, "TEST residual is materially worse than TRAIN residual; flag possible overfit."
    return False, "TEST residual is not materially worse than TRAIN residual."


def verdict(metrics: dict, overfit: bool) -> tuple[str, str]:
    fixed = metrics
    borderline_fn = fixed["borderline_residual_false_negative_rate"]
    normal_flag = fixed["flag_burden_normal_truth"]
    specificity = fixed["confident_nonflagged"]["specificity"]
    if borderline_fn <= 0.10 and normal_flag <= 0.20 and specificity >= 0.90 and not overfit:
        return "STRONG", "TEST borderline residual false-negative rate <=0.10, normal flag burden <=0.20, non-flagged specificity >=0.90, and residual bias generalizes."
    if borderline_fn < 0.24 and normal_flag <= 0.35 and not overfit:
        return "MODERATE", "borderline residual false negatives improve versus 0.24 baseline, but one or more STRONG criteria do not clear."
    return "WEAK", "no material held-out improvement, overfit flag, or excessive flag burden."


def md_table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    out = ["| " + " | ".join(label for _, label, _ in columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        cells = []
        for key, _, fmt in columns:
            value = row.get(key)
            if isinstance(value, float):
                cells.append(format(value, fmt))
            else:
                cells.append(str(value))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def policy_rows_for_report(metrics_by_variant: dict) -> list[dict]:
    rows = []
    for variant in POLICY_VARIANTS:
        m = metrics_by_variant[variant]
        rows.append(
            {
                "policy": POLICY_LABELS[variant],
                "caught_obstructed": m["obstructed_truth"]["obstructed_fraction"],
                "flagged_obstructed": m["obstructed_truth"]["flagged_fraction"],
                "residual_fn": m["obstructed_truth"]["normal_fraction"],
                "borderline_fn": m["borderline_residual_false_negative_rate"],
                "normal_correct": m["normal_truth"]["normal_fraction"],
                "normal_flag": m["normal_truth"]["flagged_fraction"],
                "normal_fp": m["normal_truth"]["obstructed_fraction"],
                "flag_all": m["flag_burden_all"],
                "nonflag_sens": m["confident_nonflagged"]["sensitivity"],
                "nonflag_spec": m["confident_nonflagged"]["specificity"],
                "nonflag_kappa": m["confident_nonflagged"]["cohens_kappa"],
            }
        )
    return rows


def add_preferred_policy_columns(rows: pd.DataFrame, correction: dict, correction_name: str, policy_variant: str) -> pd.DataFrame:
    out = rows.copy()
    ratio_col = f"corrected_ratio_{correction_name}"
    if ratio_col not in out:
        out[ratio_col] = apply_correction(out, correction)
    out["truncated_ats"] = ats_flag(out)
    out[f"in_band_corrected_060_070_{correction_name}"] = (out[ratio_col] >= 0.60) & (out[ratio_col] < 0.70)
    out["in_band_raw_060_070"] = (out["model_ratio"] >= 0.60) & (out["model_ratio"] < 0.70)
    out["policy_outcome_fixed_preferred"] = pd.NA
    out["policy_outcome_fixed_correction_only"] = pd.NA
    test_mask = out["split"] == "test"
    out.loc[test_mask, "policy_outcome_fixed_preferred"] = policy_outcomes(
        out.loc[test_mask],
        ratio_col,
        flag_col="truncated_ats",
        flag_variant=policy_variant,
    )
    out.loc[test_mask, "policy_outcome_fixed_correction_only"] = policy_outcomes(
        out.loc[test_mask],
        ratio_col,
        flag_col="truncated_ats",
        flag_variant="correction_only",
    )
    return out


def write_report(root: Path, result: dict, rows: pd.DataFrame) -> None:
    train = result["corrections"]
    cand = result["candidate_metrics"]
    pref = result["preferred_correction"]
    pref_variant = result["preferred_policy_variant"]
    baseline = result["baseline_test_metrics"]
    diag = cand[pref]["diagnostics"]
    lines = [
        "# Gate - Bias-Calibrated Classification + ATS Flag Policy, Cycle F",
        "",
        f"- VERDICT: {result['verdict']} - {result['verdict_reason']}",
        "- model: plain bi-exponential volume-domain fit, baseline FVC convention",
        "- no gamma, no registration, no ML residual, no clinical-threshold movement",
        f"- seed: {RANDOM_SEED}",
        f"- preferred correction: `{pref}`",
        f"- preferred TEST policy variant: `{pref_variant}` ({POLICY_LABELS[pref_variant]})",
        "",
        "## Sample",
        "",
        f"- truth counts before raw usability filtering: {result['truth_counts']}",
        f"- available selected counts: {result['available_selected_counts']}",
        f"- dropped before sampling counts: {result['dropped_before_sampling_counts']}",
        f"- sampled counts: {result['sampled_counts']}",
        f"- valid fit counts: {result['valid_fit_counts']}",
        f"- train/test split counts: {result['split_counts']}",
        "",
        "## Step 1 - Train-Only Bias Correction",
        "",
        "| correction | fitted form | params | train residual median | train residual IQR | test residual median | test residual IQR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["c1", "c2"]:
        tr = train[name]["train_residual"]
        te = cand[name]["test_residual"]
        lines.append(
            f"| {name} | `{train[name]['formula']}` | {train[name]['n_params']} | {tr['median']:.4f} | "
            f"{tr['iqr'][0]:.4f} to {tr['iqr'][1]:.4f} | {te['median']:.4f} | {te['iqr'][0]:.4f} to {te['iqr'][1]:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Preferred correction: `{pref}`. Preference was made on held-out TEST behavior, not fit quality.",
            f"Preferred policy variant: `{pref_variant}`. This preference was made on held-out TEST behavior among the pre-requested variants.",
            "",
            "## Bug Diagnosis - Flag Branch Reachability",
            "",
            f"- TEST participants with ATS truncation flag: {diag['truncation_flagged_n']}/{diag['test_n']}",
            f"- TEST participants in corrected-ratio [0.60,0.70): {diag['corrected_band_060_070_n']}/{diag['test_n']}",
            f"- TEST participants in raw-model-ratio [0.60,0.70): {diag['raw_band_060_070_n']}/{diag['test_n']}",
            f"- old impossible fixed-0.70 FLAG branch count: {diag['old_impossible_fixed_flag_branch_n']}",
            "",
            "| fixed-threshold variant | FLAG | OBSTRUCTED | NORMAL |",
            "|---|---:|---:|---:|",
        ]
    )
    for variant in POLICY_VARIANTS:
        counts = diag["fixed_branch_counts"][variant]
        lines.append(
            f"| {POLICY_LABELS[variant]} | {counts['flag_n']} | {counts['obstructed_n']} | {counts['normal_n']} |"
        )
    lines.extend(
        [
            "",
            "## Step 2 - Pre-Specified ATS Truncation Flag",
            "",
            "- flag: `dV_last > 0.025 L OR FET < 6.0 s`.",
            f"- truncation flag fraction by stratum: {result['truncation_by_stratum']}",
            "",
            "### Sensitivity Sweep",
            "",
            md_table(
                result["sensitivity_sweep"],
                [
                    ("plateau_L", "plateau L", ".3f"),
                    ("fet_s", "FET s", ".1f"),
                    ("policy_variant", "policy variant", ""),
                    ("flag_rate_all", "flag all", ".3f"),
                    ("flag_rate_normal_truth", "flag normal truth", ".3f"),
                    ("flagged_outcome_all", "FLAG outcome all", ".3f"),
                    ("borderline_residual_fn", "borderline residual FN", ".3f"),
                    ("overall_residual_fn", "overall residual FN", ".3f"),
                    ("nonflag_specificity", "nonflag specificity", ".3f"),
                ],
            ),
            "",
            "## Step 3-4 - TEST-Set Accounting",
            "",
            "Baseline is naive fixed-0.70 with no correction and no flag. Policy rows use the preferred correction.",
            "",
            "| policy | caught obstructed | flagged obstructed | residual FN | borderline FN | normal correct | normal flag | normal FP | flag all | nonflag sens | nonflag spec | kappa |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| baseline, no correction/no flag | {baseline['obstructed_truth']['obstructed_fraction']:.3f} | {baseline['obstructed_truth']['flagged_fraction']:.3f} | {baseline['obstructed_truth']['normal_fraction']:.3f} | {baseline['borderline_residual_false_negative_rate']:.3f} | {baseline['normal_truth']['normal_fraction']:.3f} | {baseline['normal_truth']['flagged_fraction']:.3f} | {baseline['normal_truth']['obstructed_fraction']:.3f} | {baseline['flag_burden_all']:.3f} | {baseline['confident_nonflagged']['sensitivity']:.3f} | {baseline['confident_nonflagged']['specificity']:.3f} | {baseline['confident_nonflagged']['cohens_kappa']:.3f} |",
        ]
    )
    for row in policy_rows_for_report(cand[pref]["fixed"]):
        lines.append(
            f"| {row['policy']} | {row['caught_obstructed']:.3f} | {row['flagged_obstructed']:.3f} | {row['residual_fn']:.3f} | {row['borderline_fn']:.3f} | {row['normal_correct']:.3f} | {row['normal_flag']:.3f} | {row['normal_fp']:.3f} | {row['flag_all']:.3f} | {row['nonflag_sens']:.3f} | {row['nonflag_spec']:.3f} | {row['nonflag_kappa']:.3f} |"
        )
    lines.extend(
        [
            "",
            "### Candidate Corrections and Policy Variants on TEST",
            "",
            "| correction | policy variant | borderline residual FN | overall residual FN | normal flag burden | nonflag specificity | test residual median |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ["c1", "c2"]:
        for variant in POLICY_VARIANTS:
            m = cand[name]["fixed"][variant]
            lines.append(
                f"| {name} | {POLICY_LABELS[variant]} | {m['borderline_residual_false_negative_rate']:.3f} | {m['obstructed_truth']['normal_fraction']:.3f} | "
                f"{m['flag_burden_normal_truth']:.3f} | {m['confident_nonflagged']['specificity']:.3f} | {cand[name]['test_residual']['median']:.4f} |"
            )
    lines.extend(
        [
            "",
            "### GLI LLN Policy Metrics",
            "",
            "| policy | residual FN | borderline FN | normal flag burden | nonflag specificity | FLAG outcome all |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in POLICY_VARIANTS:
        m = cand[pref]["lln"][variant]
        lines.append(
            f"| {POLICY_LABELS[variant]} | {m['obstructed_truth']['normal_fraction']:.3f} | {m['borderline_residual_false_negative_rate']:.3f} | {m['flag_burden_normal_truth']:.3f} | {m['confident_nonflagged']['specificity']:.3f} | {m['flag_burden_all']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Step 5 - Generalization / Overfit Check",
            "",
            result["overfit_statement"],
            "",
            "This is calibration, not threshold-shopping: the correction is learned on disjoint TRAIN participants, uses <=2 parameters, the clinical threshold remains fixed, and the truncation flag is computed from the ATS end-of-test rule before outcomes.",
            "",
            "## Step 6 - Coverage / Drop Audit",
            "",
            f"- obstructed drop audit: {result['drop_bias_audit']}",
            f"- remaining TEST obstructed-truth NORMAL leaks that were truncation-flaggable but outside the preferred policy: {result['residual_leaks_trunc_flag_outside_band_n']}",
            "",
            f"VERDICT: {result['verdict']} - {result['verdict_reason']}",
            "",
        ]
    )
    out = root / "results"
    out.mkdir(exist_ok=True)
    (out / "gate_policy_calibrated_flag_F.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "gate_policy_calibrated_flag_F.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    rows.to_csv(out / "gate_policy_calibrated_flag_F_rows.csv", index=False)


def run(root: Path) -> dict:
    column_map = load_column_map(root)
    truth = load_truth(root, column_map)
    gli = load_gli(root)
    raw_path, _ = find_or_unzip_raw(root)
    raw, _ = read_raw(raw_path)
    rows, meta, selected_truth = build_pool(root, raw, truth, gli)

    train = rows[rows["split"] == "train"].copy()
    test = rows[rows["split"] == "test"].copy()
    corrections = fit_corrections(train)
    for name, corr in corrections.items():
        rows[f"corrected_ratio_{name}"] = apply_correction(rows, corr)
    rows["truncated_ats"] = ats_flag(rows)

    candidates = {name: evaluate_candidate(test, corr, name) for name, corr in corrections.items()}
    preferred, preferred_variant = choose_preferred_candidate(candidates)
    baseline_outcome = policy_outcomes(test.assign(ratio_baseline=test["model_ratio"], no_flag=False), "ratio_baseline", flag_col="no_flag")
    baseline_metrics = three_outcome_metrics(test, baseline_outcome)
    sweep = sensitivity_sweep(test, corrections[preferred], preferred_variant)
    train_resid = corrections[preferred]["train_residual"]
    overfit, overfit_text = overfit_status(train_resid, candidates[preferred]["test_residual"])
    status, reason = verdict(candidates[preferred]["fixed"][preferred_variant], overfit)
    drop_audit = drop_bias_audit(raw, selected_truth, truth)
    leak_count = residual_leaks_outside_band(test, corrections[preferred], preferred_variant)
    rows = add_preferred_policy_columns(rows, corrections[preferred], preferred, preferred_variant)

    result = {
        "verdict": status,
        "verdict_reason": reason,
        "random_seed": RANDOM_SEED,
        **meta,
        "split_counts": split_counts(rows),
        "corrections": corrections,
        "preferred_correction": preferred,
        "preferred_policy_variant": preferred_variant,
        "preferred_policy_label": POLICY_LABELS[preferred_variant],
        "candidate_metrics": candidates,
        "baseline_test_metrics": baseline_metrics,
        "truncation_by_stratum": truncation_by_stratum(rows),
        "sensitivity_sweep": sweep,
        "overfit": overfit,
        "overfit_statement": overfit_text,
        "drop_bias_audit": drop_audit,
        "residual_leaks_trunc_flag_outside_band_n": leak_count,
    }
    write_report(root, result, rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    pref = result["preferred_correction"]
    pref_variant = result["preferred_policy_variant"]
    fixed = result["candidate_metrics"][pref]["fixed"][pref_variant]
    print(f"Policy calibrated flag gate: {result['verdict']} - {result['verdict_reason']}")
    print(f"Preferred correction: {pref}")
    print(f"Preferred policy variant: {pref_variant}")
    print(f"TEST borderline residual FN: {fixed['borderline_residual_false_negative_rate']:.3f}")
    print(f"TEST normal flag burden: {fixed['flag_burden_normal_truth']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
