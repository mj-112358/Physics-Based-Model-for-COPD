from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import gate_clinical_classification_f as clinical_gate
from .gate3_diagnosis import clinical_metrics_from_volume, fit_ombe_volume_lm
from .gate23_f import find_or_unzip_raw, read_raw
from .gate_clinical_classification_f import (
    RANDOM_SEED,
    build_usable_raw_candidates,
    extract_curve_for_fit,
    load_column_map,
    load_truth,
    severity_category,
)
from .gate_reconciliation_audit_f import clean_json, fmt, median_iqr

OUTPUT_STEM = "gate_reproducibility_volume_F"
CHECKPOINT_NAME = "_gate_reproducibility_volume_F_fits_checkpoint.csv"
PROGRESS_EVERY = 250


def assert_volume_domain_path() -> None:
    if clinical_gate.fit_ombe_volume_lm is not fit_ombe_volume_lm:
        raise AssertionError("gate_clinical_classification_f.py is not using the imported fit_ombe_volume_lm callable")
    if fit_ombe_volume_lm.__module__ != "predictx.gate3_diagnosis":
        raise AssertionError(f"Unexpected volume fitter module: {fit_ombe_volume_lm.__module__}")


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.map({True: True, False: False, "True": True, "False": False, "true": True, "false": False}).fillna(False)


def raw_index_map(raw: pd.DataFrame) -> dict[int, Any]:
    return {int(idx): row for idx, row in raw.iterrows()}


def fit_one_volume(raw_by_index: dict[int, Any], candidate: Any) -> dict[str, Any]:
    base = {
        "row_type": "maneuver_fit",
        "SEQN": float(candidate.SEQN),
        "raw_index": int(candidate.raw_index),
        "SPAMANU": float(candidate.SPAMANU),
        "SPAQEFF": str(candidate.SPAQEFF),
        "SPAACC": str(candidate.SPAACC),
        "SPXPTS": int(candidate.SPXPTS),
        "parse_error": str(candidate.parse_error) if pd.notna(candidate.parse_error) else "",
        "observed_fev1_l": float(candidate.parsed_fev1_l) if pd.notna(candidate.parsed_fev1_l) else np.nan,
        "observed_fvc_l": float(candidate.parsed_fvc_l) if pd.notna(candidate.parsed_fvc_l) else np.nan,
        "observed_ratio": float(candidate.parsed_ratio) if pd.notna(candidate.parsed_ratio) else np.nan,
        "t0_s": float(candidate.t0_s) if pd.notna(candidate.t0_s) else np.nan,
        "duration_s": float(candidate.duration_s) if pd.notna(candidate.duration_s) else np.nan,
        "fit_domain": "volume",
        "fit_path": "gate3_diagnosis.fit_ombe_volume_lm",
    }
    if base["parse_error"]:
        base.update(
            {
                "fit_error": f"parse_error: {base['parse_error']}",
                "volume_fit_success": False,
                "model_fev1_l": np.nan,
                "model_fvc_l": np.nan,
                "model_ratio": np.nan,
                "volume_r2": np.nan,
                "volume_nrss_norm": np.nan,
                "fit_tau_f": np.nan,
                "fit_tau_s": np.nan,
                "fit_alpha": np.nan,
                "fit_fvc_l": np.nan,
                "fit_tau_o": np.nan,
            }
        )
        return base
    try:
        curve = extract_curve_for_fit(raw_by_index[int(candidate.raw_index)])
        fit = fit_ombe_volume_lm(curve["vol_t"], curve["vol_l"])
        model_fev1_l, model_fvc_l, model_ratio = clinical_metrics_from_volume(curve["vol_t"], fit["pred_volume_l"])
        base.update(
            {
                "fit_error": "",
                "volume_fit_success": bool(fit["success"]),
                "model_fev1_l": float(model_fev1_l),
                "model_fvc_l": float(model_fvc_l),
                "model_ratio": float(model_ratio),
                "volume_r2": float(fit["volume_r2"]),
                "volume_nrss_norm": float(fit["volume_nrss_norm"]),
                "fit_tau_f": float(fit["params"][0]),
                "fit_tau_s": float(fit["params"][1]),
                "fit_alpha": float(fit["params"][2]),
                "fit_fvc_l": float(fit["params"][3]),
                "fit_tau_o": float(fit["params"][4]),
            }
        )
    except Exception as exc:
        base.update(
            {
                "fit_error": str(exc),
                "volume_fit_success": False,
                "model_fev1_l": np.nan,
                "model_fvc_l": np.nan,
                "model_ratio": np.nan,
                "volume_r2": np.nan,
                "volume_nrss_norm": np.nan,
                "fit_tau_f": np.nan,
                "fit_tau_s": np.nan,
                "fit_alpha": np.nan,
                "fit_fvc_l": np.nan,
                "fit_tau_o": np.nan,
            }
        )
    return base


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", index=False, header=not path.exists())


def run_volume_fits(raw: pd.DataFrame, candidates: pd.DataFrame, results_dir: Path, resume: bool) -> pd.DataFrame:
    checkpoint = results_dir / CHECKPOINT_NAME
    raw_by_index = raw_index_map(raw)
    counts = candidates.groupby("SEQN").size()
    eligible_seqn = set(counts[counts >= 2].index.astype(float))
    fit_targets = candidates[candidates["SEQN"].astype(float).isin(eligible_seqn)].copy()
    fit_targets = fit_targets.sort_values(["SEQN", "SPAMANU", "raw_index"]).reset_index(drop=True)

    done_keys: set[tuple[int, float, float]] = set()
    if resume and checkpoint.exists():
        prior = pd.read_csv(checkpoint, usecols=["raw_index", "SEQN", "SPAMANU"])
        done_keys = set((int(r.raw_index), float(r.SEQN), float(r.SPAMANU)) for r in prior.itertuples(index=False))
        print(f"Volume reproducibility checkpoint found: {len(done_keys)} completed manoeuvre fits.")
    elif checkpoint.exists():
        checkpoint.unlink()

    batch: list[dict[str, Any]] = []
    total = len(fit_targets)
    completed_this_run = 0
    for i, candidate in enumerate(fit_targets.itertuples(index=False), start=1):
        key = (int(candidate.raw_index), float(candidate.SEQN), float(candidate.SPAMANU))
        if key in done_keys:
            continue
        batch.append(fit_one_volume(raw_by_index, candidate))
        completed_this_run += 1
        if len(batch) >= PROGRESS_EVERY:
            append_rows(checkpoint, batch)
            batch = []
            done_total = len(done_keys) + completed_this_run
            print(f"Volume reproducibility fit progress: {done_total}/{total} manoeuvres")
    append_rows(checkpoint, batch)
    if not checkpoint.exists():
        raise RuntimeError("No checkpoint rows were written; cannot summarize reproducibility gate.")
    return pd.read_csv(checkpoint)


def icc_2_1_first_two(df: pd.DataFrame, value_col: str) -> dict[str, Any]:
    rows = []
    for seqn, group in df.dropna(subset=[value_col]).groupby("SEQN"):
        group = group.sort_values(["SPAMANU", "raw_index"])
        if len(group) < 2:
            continue
        vals = group[value_col].iloc[:2].to_numpy(float)
        if np.all(np.isfinite(vals)):
            rows.append(vals)
    matrix = np.asarray(rows, dtype=float)
    if matrix.shape[0] < 2:
        return {"n_participants": int(matrix.shape[0]), "k": int(matrix.shape[1]) if matrix.ndim == 2 else 0, "icc_2_1": np.nan}
    n, k = matrix.shape
    grand = float(matrix.mean())
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    ss_rows = float(k * np.sum((row_means - grand) ** 2))
    ss_cols = float(n * np.sum((col_means - grand) ** 2))
    ss_total = float(np.sum((matrix - grand) ** 2))
    ss_err = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + (k * (ms_cols - ms_err) / n)
    icc = (ms_rows - ms_err) / denom if np.isfinite(denom) and abs(denom) > 1e-12 else np.nan
    return {
        "n_participants": int(n),
        "k": int(k),
        "icc_2_1": float(icc) if np.isfinite(icc) else np.nan,
        "note": "ICC(2,1) computed on the first two acceptable A/B manoeuvres per participant after sorting by SPAMANU.",
    }


def participant_sd_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seqn, group in df.groupby("SEQN"):
        if len(group) < 2:
            continue
        truth_severity = group["truth_severity"].dropna().iloc[0] if group["truth_severity"].notna().any() else "missing"
        row = {
            "row_type": "participant_summary",
            "SEQN": float(seqn),
            "truth_severity": truth_severity,
            "maneuver_count": int(len(group)),
        }
        for prefix in ["observed", "model"]:
            row[f"{prefix}_fev1_sd_l"] = float(group[f"{prefix}_fev1_l"].std(ddof=1))
            row[f"{prefix}_fvc_sd_l"] = float(group[f"{prefix}_fvc_l"].std(ddof=1))
            row[f"{prefix}_ratio_sd"] = float(group[f"{prefix}_ratio"].std(ddof=1))
            calls = (group[f"{prefix}_ratio"] < 0.70).astype(bool)
            row[f"{prefix}_classification_unanimous"] = bool(calls.nunique() == 1)
            row[f"{prefix}_classification_majority_fraction"] = float(max(calls.mean(), 1.0 - calls.mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_sd(pp: pd.DataFrame, prefix: str) -> dict[str, Any]:
    return {
        "fev1_l": median_iqr(pp[f"{prefix}_fev1_sd_l"]) if len(pp) else {},
        "fvc_l": median_iqr(pp[f"{prefix}_fvc_sd_l"]) if len(pp) else {},
        "ratio": median_iqr(pp[f"{prefix}_ratio_sd"]) if len(pp) else {},
    }


def summarize_stability(pp: pd.DataFrame, prefix: str) -> dict[str, Any]:
    if pp.empty:
        return {"n": 0, "unanimous_fraction": np.nan, "split_fraction": np.nan, "majority_fraction": {}}
    unanimous = pp[f"{prefix}_classification_unanimous"].astype(bool)
    return {
        "n": int(len(pp)),
        "unanimous_fraction": float(unanimous.mean()),
        "split_fraction": float((~unanimous).mean()),
        "majority_fraction": median_iqr(pp[f"{prefix}_classification_majority_fraction"]),
    }


def ats_repeatability(df: pd.DataFrame, prefix: str) -> dict[str, Any]:
    rows = []
    for seqn, group in df.groupby("SEQN"):
        if len(group) < 2:
            continue
        fev1 = np.sort(group[f"{prefix}_fev1_l"].to_numpy(float))[::-1]
        fvc = np.sort(group[f"{prefix}_fvc_l"].to_numpy(float))[::-1]
        rows.append(
            {
                "SEQN": float(seqn),
                "fev1_top2_diff_l": float(abs(fev1[0] - fev1[1])),
                "fvc_top2_diff_l": float(abs(fvc[0] - fvc[1])),
                "ats_repeatable": bool(abs(fev1[0] - fev1[1]) <= 0.150 and abs(fvc[0] - fvc[1]) <= 0.150),
            }
        )
    out = pd.DataFrame(rows)
    return {
        "n": int(len(out)),
        "repeatable_n": int(out["ats_repeatable"].sum()) if len(out) else 0,
        "repeatable_fraction": float(out["ats_repeatable"].mean()) if len(out) else np.nan,
        "fev1_top2_diff_l": median_iqr(out["fev1_top2_diff_l"]) if len(out) else {},
        "fvc_top2_diff_l": median_iqr(out["fvc_top2_diff_l"]) if len(out) else {},
    }


def summarize(
    root: Path,
    raw: pd.DataFrame,
    candidates: pd.DataFrame,
    fit_rows: pd.DataFrame,
    truth: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    counts_all = candidates.groupby("SEQN").size()
    eligible_seqn = set(counts_all[counts_all >= 2].index.astype(float))
    count_distribution = {str(int(k)): int(v) for k, v in counts_all[counts_all >= 2].value_counts().sort_index().items()}

    fit_rows = fit_rows.copy()
    fit_rows["volume_fit_success"] = bool_series(fit_rows["volume_fit_success"])
    fit_rows["fit_error"] = fit_rows["fit_error"].fillna("")
    fit_rows = fit_rows.merge(truth[["SEQN", "fev1_fvc_ratio"]], on="SEQN", how="left")
    fit_rows["truth_severity"] = fit_rows["fev1_fvc_ratio"].map(severity_category)
    fit_rows["observed_obstructed_fixed"] = fit_rows["observed_ratio"] < 0.70
    fit_rows["model_obstructed_fixed"] = fit_rows["model_ratio"] < 0.70

    valid_model = fit_rows[
        fit_rows["volume_fit_success"]
        & fit_rows["fit_error"].eq("")
        & fit_rows[["observed_fev1_l", "observed_fvc_l", "observed_ratio", "model_fev1_l", "model_fvc_l", "model_ratio"]].apply(np.isfinite).all(axis=1)
    ].copy()
    model_counts = valid_model.groupby("SEQN").size()
    model_common_seqn = set(model_counts[model_counts >= 2].index.astype(float))
    common = valid_model[valid_model["SEQN"].astype(float).isin(model_common_seqn)].copy()
    pp = participant_sd_rows(common)

    borderline_pp = pp[pp["truth_severity"] == "mild_0.60_0.70"].copy() if len(pp) else pd.DataFrame()
    observed_ats = ats_repeatability(common, "observed")
    model_ats = ats_repeatability(common, "model")
    observed_icc = icc_2_1_first_two(common, "observed_ratio")
    model_icc = icc_2_1_first_two(common, "model_ratio")
    observed_stability_all = summarize_stability(pp, "observed")
    model_stability_all = summarize_stability(pp, "model")
    observed_stability_borderline = summarize_stability(borderline_pp, "observed")
    model_stability_borderline = summarize_stability(borderline_pp, "model")

    model_more_repeatable = bool(
        np.isfinite(model_ats.get("repeatable_fraction", np.nan))
        and np.isfinite(observed_ats.get("repeatable_fraction", np.nan))
        and model_ats["repeatable_fraction"] > observed_ats["repeatable_fraction"]
    )
    ats_margin = (
        float(model_ats["repeatable_fraction"] - observed_ats["repeatable_fraction"])
        if np.isfinite(model_ats.get("repeatable_fraction", np.nan)) and np.isfinite(observed_ats.get("repeatable_fraction", np.nan))
        else np.nan
    )
    verdict = (
        "REPRODUCIBLE"
        if np.isfinite(model_icc["icc_2_1"])
        and model_icc["icc_2_1"] >= 0.75
        and model_stability_all["unanimous_fraction"] >= 0.90
        else "UNSTABLE"
    )
    result = {
        "verdict": verdict,
        "random_seed": RANDOM_SEED,
        "raw_source": str(root),
        "fit_domain_assertion": "volume-domain fit path confirmed: gate_clinical_classification_f.fit_ombe_volume_lm is gate3_diagnosis.fit_ombe_volume_lm",
        "fit_path_used": "predictx.gate3_diagnosis.fit_ombe_volume_lm",
        "same_fitter_as_gate_clinical_classification_f": True,
        "participants_with_at_least_2_acceptable_ab_manoeuvres": int(len(eligible_seqn)),
        "acceptable_ab_manoeuvre_count_distribution": count_distribution,
        "candidate_manoeuvre_rows": int(len(candidates[candidates["SEQN"].astype(float).isin(eligible_seqn)])),
        "fit_rows_total": int(len(fit_rows)),
        "fit_success_rows": int(valid_model.shape[0]),
        "fit_failure_rows": int(len(fit_rows) - valid_model.shape[0]),
        "participants_with_at_least_2_successful_volume_fits": int(len(model_common_seqn)),
        "fit_failure_counts": fit_rows.loc[~(fit_rows["volume_fit_success"] & fit_rows["fit_error"].eq("")), "fit_error"].replace("", "optimizer_success_false").value_counts().head(20).to_dict(),
        "within_participant_sd_common_successful_fits": {
            "observed": summarize_sd(pp, "observed"),
            "model": summarize_sd(pp, "model"),
        },
        "icc_2_1_ratio_first_two": {
            "observed": observed_icc,
            "model": model_icc,
        },
        "classification_stability_fixed_070": {
            "all": {
                "observed": observed_stability_all,
                "model": model_stability_all,
            },
            "borderline_0.60_0.70": {
                "observed": observed_stability_borderline,
                "model": model_stability_borderline,
            },
        },
        "ats_repeatability_common_successful_fits": {
            "observed": observed_ats,
            "model": model_ats,
            "model_minus_observed_fraction": ats_margin,
            "model_more_repeatable_than_observed": model_more_repeatable,
        },
    }
    per_manoeuvre = fit_rows.copy()
    per_manoeuvre["row_type"] = "maneuver_fit"
    all_rows = pd.concat([per_manoeuvre, pp], ignore_index=True, sort=False)
    return result, all_rows


def make_report(result: dict[str, Any]) -> str:
    sd = result["within_participant_sd_common_successful_fits"]
    icc = result["icc_2_1_ratio_first_two"]
    stab = result["classification_stability_fixed_070"]
    ats = result["ats_repeatability_common_successful_fits"]
    model_margin = ats["model_minus_observed_fraction"]
    repeat_sentence = (
        f"Model estimates are more ATS-repeatable than observed curves by {100.0 * model_margin:.1f} percentage points."
        if ats["model_more_repeatable_than_observed"]
        else f"Model estimates are less ATS-repeatable than observed curves by {100.0 * abs(model_margin):.1f} percentage points."
    )
    lines = [
        "# Gate - Volume-Domain Cross-Manoeuvre Reproducibility, Cycle F",
        "",
        f"- VERDICT: {result['verdict']}",
        "- Deterministic LM only. No NUTS, SNPE, gamma onset, registration, or ML residual.",
        f"- Random seed: {result['random_seed']}",
        f"- Fit path used: `{result['fit_path_used']}`",
        f"- Volume-domain assertion: {result['fit_domain_assertion']}",
        "",
        "## D1. Eligible Manoeuvres",
        "",
        f"- participants with >=2 acceptable A/B manoeuvres: {result['participants_with_at_least_2_acceptable_ab_manoeuvres']}",
        f"- acceptable A/B manoeuvre count distribution: {result['acceptable_ab_manoeuvre_count_distribution']}",
        f"- candidate manoeuvre rows fit-attempted: {result['candidate_manoeuvre_rows']}",
        "",
        "## D2. Fresh Volume-Domain Fits",
        "",
        f"- fit rows total: {result['fit_rows_total']}",
        f"- successful finite volume-domain fit rows: {result['fit_success_rows']}",
        f"- failed/nonfinite fit rows: {result['fit_failure_rows']}",
        f"- participants with >=2 successful volume-domain fits: {result['participants_with_at_least_2_successful_volume_fits']}",
        "",
        "## D3. Within-Participant Variability",
        "",
        "| metric | observed median SD | observed IQR | model median SD | model IQR |",
        "|---|---:|---:|---:|---:|",
        f"| FEV1 (L) | {fmt(sd['observed']['fev1_l']['median'], 4)} | {fmt(sd['observed']['fev1_l']['iqr'][0], 4)} to {fmt(sd['observed']['fev1_l']['iqr'][1], 4)} | {fmt(sd['model']['fev1_l']['median'], 4)} | {fmt(sd['model']['fev1_l']['iqr'][0], 4)} to {fmt(sd['model']['fev1_l']['iqr'][1], 4)} |",
        f"| FVC (L) | {fmt(sd['observed']['fvc_l']['median'], 4)} | {fmt(sd['observed']['fvc_l']['iqr'][0], 4)} to {fmt(sd['observed']['fvc_l']['iqr'][1], 4)} | {fmt(sd['model']['fvc_l']['median'], 4)} | {fmt(sd['model']['fvc_l']['iqr'][0], 4)} to {fmt(sd['model']['fvc_l']['iqr'][1], 4)} |",
        f"| FEV1/FVC | {fmt(sd['observed']['ratio']['median'], 4)} | {fmt(sd['observed']['ratio']['iqr'][0], 4)} to {fmt(sd['observed']['ratio']['iqr'][1], 4)} | {fmt(sd['model']['ratio']['median'], 4)} | {fmt(sd['model']['ratio']['iqr'][0], 4)} to {fmt(sd['model']['ratio']['iqr'][1], 4)} |",
        "",
        f"- Observed ratio ICC(2,1): {fmt(icc['observed']['icc_2_1'], 3)} on n={icc['observed']['n_participants']} participants.",
        f"- Model ratio ICC(2,1): {fmt(icc['model']['icc_2_1'], 3)} on n={icc['model']['n_participants']} participants.",
        f"- ICC note: {icc['model']['note']}",
        "",
        "## D4. Fixed-0.70 Classification Stability",
        "",
        "| stratum | source | n | unanimous fraction | split fraction |",
        "|---|---|---:|---:|---:|",
        f"| all | observed | {stab['all']['observed']['n']} | {fmt(stab['all']['observed']['unanimous_fraction'], 3)} | {fmt(stab['all']['observed']['split_fraction'], 3)} |",
        f"| all | model | {stab['all']['model']['n']} | {fmt(stab['all']['model']['unanimous_fraction'], 3)} | {fmt(stab['all']['model']['split_fraction'], 3)} |",
        f"| borderline 0.60-0.70 | observed | {stab['borderline_0.60_0.70']['observed']['n']} | {fmt(stab['borderline_0.60_0.70']['observed']['unanimous_fraction'], 3)} | {fmt(stab['borderline_0.60_0.70']['observed']['split_fraction'], 3)} |",
        f"| borderline 0.60-0.70 | model | {stab['borderline_0.60_0.70']['model']['n']} | {fmt(stab['borderline_0.60_0.70']['model']['unanimous_fraction'], 3)} | {fmt(stab['borderline_0.60_0.70']['model']['split_fraction'], 3)} |",
        "",
        "## D5. ATS Repeatability",
        "",
        "| source | n | repeatable n | repeatable fraction | FEV1 top-2 diff median L | FVC top-2 diff median L |",
        "|---|---:|---:|---:|---:|---:|",
        f"| observed curve values | {ats['observed']['n']} | {ats['observed']['repeatable_n']} | {fmt(ats['observed']['repeatable_fraction'], 3)} | {fmt(ats['observed']['fev1_top2_diff_l']['median'], 4)} | {fmt(ats['observed']['fvc_top2_diff_l']['median'], 4)} |",
        f"| volume-domain model estimates | {ats['model']['n']} | {ats['model']['repeatable_n']} | {fmt(ats['model']['repeatable_fraction'], 3)} | {fmt(ats['model']['fev1_top2_diff_l']['median'], 4)} | {fmt(ats['model']['fvc_top2_diff_l']['median'], 4)} |",
        "",
        repeat_sentence,
        "",
        "## Output Rows",
        "",
        "- Per-manoeuvre rows contain observed and volume-domain model FEV1/FVC metrics.",
        "- Participant summary rows contain within-participant SD and classification-stability flags.",
        "",
        f"VERDICT: {result['verdict']}; borderline model unanimous fraction {fmt(stab['borderline_0.60_0.70']['model']['unanimous_fraction'], 3)}; model-vs-observed ATS repeatability margin {fmt(100.0 * model_margin, 1)} percentage points.",
        "",
    ]
    return "\n".join(lines)


def run(root: Path, resume: bool = True) -> dict[str, Any]:
    assert_volume_domain_path()
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    column_map = load_column_map(root)
    truth = load_truth(root, column_map)
    raw_path, raw_note = find_or_unzip_raw(root)
    raw, raw_info = read_raw(raw_path)
    print(f"Volume reproducibility raw source: {raw_path}")
    candidates = build_usable_raw_candidates(raw)
    fit_rows = run_volume_fits(raw, candidates, results_dir, resume=resume)
    result, all_rows = summarize(raw_path, raw, candidates, fit_rows, truth)
    result["raw_source"] = str(raw_path)
    result["raw_source_note"] = raw_note
    result["raw_loaded_rows"] = int(raw_info["loaded_rows"])
    report = make_report(result)
    (results_dir / f"{OUTPUT_STEM}.md").write_text(report, encoding="utf-8")
    (results_dir / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    all_rows.to_csv(results_dir / f"{OUTPUT_STEM}_rows.csv", index=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result = run(Path(args.root).resolve(), resume=not args.no_resume)
    ats = result["ats_repeatability_common_successful_fits"]
    stab = result["classification_stability_fixed_070"]
    print(f"Volume reproducibility gate: {result['verdict']}")
    print(f"Model ratio ICC(2,1): {result['icc_2_1_ratio_first_two']['model']['icc_2_1']:.3f}")
    print(f"Borderline model unanimous fraction: {stab['borderline_0.60_0.70']['model']['unanimous_fraction']:.3f}")
    print(f"ATS model-observed repeatability margin: {100.0 * ats['model_minus_observed_fraction']:.1f} percentage points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
