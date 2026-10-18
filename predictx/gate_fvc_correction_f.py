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
    confusion_table,
    decode_grade,
    finite,
    fit_selected_rows,
    load_column_map,
    load_gli,
    load_truth,
    q,
    select_best_manoeuvre,
)

BORDERLINE_SAMPLE_MAX = 150
GRADE_SCORE = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "F": 5.0}


def signed_summary(df: pd.DataFrame, ratio_col: str = "model_ratio", group_label: str = "all") -> dict:
    valid = df.dropna(subset=["model_fev1_mL", "model_fvc_mL", ratio_col, "fev1_mL", "fvc_mL", "fev1_fvc_ratio"]).copy()
    valid = valid[(valid["fev1_mL"] > 0) & (valid["fvc_mL"] > 0)]
    fev1 = 100.0 * (valid["model_fev1_mL"] - valid["fev1_mL"]) / valid["fev1_mL"]
    fvc = 100.0 * (valid["model_fvc_mL"] - valid["fvc_mL"]) / valid["fvc_mL"]
    ratio = valid[ratio_col] - valid["fev1_fvc_ratio"]
    return {
        "group": group_label,
        "n": int(len(valid)),
        "signed_fev1_pct_median": float(np.nanmedian(fev1)) if len(valid) else np.nan,
        "signed_fev1_pct_iqr": q(fev1),
        "signed_fvc_pct_median": float(np.nanmedian(fvc)) if len(valid) else np.nan,
        "signed_fvc_pct_iqr": q(fvc),
        "signed_ratio_median": float(np.nanmedian(ratio)) if len(valid) else np.nan,
        "signed_ratio_iqr": q(ratio),
    }


def mechanism_confirmed(obstructed_summary: dict) -> tuple[bool, str]:
    fvc_iqr = obstructed_summary["signed_fvc_pct_iqr"]
    fev1_med = abs(obstructed_summary["signed_fev1_pct_median"])
    fvc_med = obstructed_summary["signed_fvc_pct_median"]
    fvc_negative = fvc_med < 0 and fvc_iqr[1] < 0
    fev1_smaller = fev1_med < 0.5 * abs(fvc_med)
    if fvc_negative and fev1_smaller:
        return True, "FVC-underestimate mechanism confirmed: signed FVC error is consistently negative and larger than signed FEV1 error."
    return (
        False,
        "mechanism not confirmed - ratio bias is not an FVC-convention artifact",
    )


def add_fvc_conventions(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["ratio_baseline"] = out["model_ratio"]
    out["fvc_baseline_mL"] = out["model_fvc_mL"]

    # Variant A: raw end-of-test time is available for every selected manoeuvre in this dataset.
    # With the existing grid-normalized OMBE volume fit, evaluating at raw EOT reproduces the
    # fitted model end-volume convention. A fixed 6 s fallback is therefore not used here.
    out["fvc_model_raw_eot_mL"] = out["model_fvc_mL"]
    out["ratio_model_raw_eot"] = out["model_fev1_mL"] / out["fvc_model_raw_eot_mL"]

    # Variant B: use the observed selected-manoeuvre end-expiratory volume as FVC.
    out["fvc_observed_raw_eot_mL"] = out["parsed_fvc_l"] * 1000.0
    out["ratio_observed_raw_eot"] = out["model_fev1_mL"] / out["fvc_observed_raw_eot_mL"]
    return out


def convention_summary(df: pd.DataFrame, scope: str) -> list[dict]:
    variants = [
        ("baseline", "ratio_baseline"),
        ("variant_a_model_raw_eot", "ratio_model_raw_eot"),
        ("variant_b_observed_raw_eot", "ratio_observed_raw_eot"),
    ]
    rows = []
    for label, col in variants:
        err = df[col] - df["fev1_fvc_ratio"]
        iqr = q(err)
        rows.append(
            {
                "scope": scope,
                "variant": label,
                "n": int(err.notna().sum()),
                "signed_ratio_median": float(np.nanmedian(err)),
                "signed_ratio_iqr": iqr,
                "signed_ratio_iqr_width": float(iqr[1] - iqr[0]) if np.all(np.isfinite(iqr)) else np.nan,
                "iqr_brackets_zero": bool(iqr[0] <= 0.0 <= iqr[1]) if np.all(np.isfinite(iqr)) else False,
            }
        )
    return rows


def choose_preferred(conventions: list[dict]) -> dict:
    candidates = [row for row in conventions if row["scope"] == "obstructed"]
    return sorted(
        candidates,
        key=lambda row: (
            abs(row["signed_ratio_median"]),
            0 if row["iqr_brackets_zero"] else 1,
            row["signed_ratio_iqr_width"],
        ),
    )[0]


def classification_metrics(df: pd.DataFrame, ratio_col: str, threshold_col: str | None = None) -> tuple[dict, pd.DataFrame]:
    valid = df.dropna(subset=[ratio_col, "fev1_fvc_ratio"]).copy()
    if threshold_col is None:
        valid["truth_obstructed"] = valid["fev1_fvc_ratio"] < 0.70
        valid["model_obstructed"] = valid[ratio_col] < 0.70
    else:
        valid = valid[valid[threshold_col].map(finite)].copy()
        valid["truth_obstructed"] = valid["fev1_fvc_ratio"] < valid[threshold_col]
        valid["model_obstructed"] = valid[ratio_col] < valid[threshold_col]
    metrics = confusion_binary(valid["truth_obstructed"], valid["model_obstructed"])
    table = confusion_table(valid, "truth_obstructed", "model_obstructed")
    return metrics, table


def classification_side_by_side(rows: pd.DataFrame, preferred_ratio_col: str) -> dict:
    fixed_base, fixed_base_table = classification_metrics(rows, "ratio_baseline")
    fixed_pref, fixed_pref_table = classification_metrics(rows, preferred_ratio_col)
    lln_base, lln_base_table = classification_metrics(rows, "ratio_baseline", "gli_lln_ratio")
    lln_pref, lln_pref_table = classification_metrics(rows, preferred_ratio_col, "gli_lln_ratio")
    return {
        "fixed_baseline": fixed_base,
        "fixed_preferred": fixed_pref,
        "lln_baseline": lln_base,
        "lln_preferred": lln_pref,
        "fixed_baseline_table": fixed_base_table,
        "fixed_preferred_table": fixed_pref_table,
        "lln_baseline_table": lln_base_table,
        "lln_preferred_table": lln_pref_table,
    }


def prepare_selected_truth(root: Path, raw: pd.DataFrame, truth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = build_usable_raw_candidates(raw)
    selected = select_best_manoeuvre(candidates)
    selected_truth = selected.merge(
        truth[["SEQN", "fev1_mL", "fvc_mL", "fev1_fvc_ratio", "effort_quality", "fev1_quality"]],
        on="SEQN",
        how="left",
    )
    return candidates, selected_truth


def build_borderline_rows(raw: pd.DataFrame, selected_truth: pd.DataFrame, gli: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    borderline_available = selected_truth[
        selected_truth["fev1_fvc_ratio"].notna()
        & (selected_truth["fev1_fvc_ratio"] >= 0.60)
        & (selected_truth["fev1_fvc_ratio"] < 0.70)
    ].copy()
    n_available = len(borderline_available)
    n_sample = min(BORDERLINE_SAMPLE_MAX, n_available)
    sample = (
        borderline_available.sample(n=n_sample, random_state=RANDOM_SEED).copy()
        if n_sample
        else borderline_available.copy()
    )
    sample["sample_group"] = "borderline"
    sample["truth_severity"] = "mild_0.60_0.70"
    fit_rows = fit_selected_rows(raw, sample)
    fit_rows = fit_rows.merge(gli, on="SEQN", how="left")
    fit_rows = add_fvc_conventions(fit_rows[fit_rows["fit_error"].fillna("") == ""].copy())
    meta = {
        "borderline_available_n": int(n_available),
        "borderline_sample_n": int(n_sample),
        "borderline_fit_valid_n": int(len(fit_rows)),
        "borderline_drop_n": int(n_available - n_sample),
    }
    return fit_rows, meta


def borderline_metrics(rows: pd.DataFrame, ratio_col: str) -> dict:
    valid = rows.dropna(subset=[ratio_col, "fev1_fvc_ratio"]).copy()
    truth_obstructed = valid["fev1_fvc_ratio"] < 0.70
    model_obstructed = valid[ratio_col] < 0.70
    signed = valid[ratio_col] - valid["fev1_fvc_ratio"]
    return {
        "n": int(len(valid)),
        "correctly_classified_obstructed_fraction": float((truth_obstructed & model_obstructed).sum() / truth_obstructed.sum()) if truth_obstructed.sum() else np.nan,
        "pushed_to_normal_fraction": float((truth_obstructed & ~model_obstructed).sum() / truth_obstructed.sum()) if truth_obstructed.sum() else np.nan,
        "false_negative_n": int((truth_obstructed & ~model_obstructed).sum()),
        "signed_ratio_median": float(np.nanmedian(signed)) if len(valid) else np.nan,
        "signed_ratio_iqr": q(signed),
    }


def participant_best_grade(raw: pd.DataFrame) -> pd.DataFrame:
    pre = raw[raw["SPATTYPE"].map(decode_grade) == "Pre"].copy()
    pre["grade"] = pre["SPAQEFF"].map(decode_grade)
    pre["grade_score"] = pre["grade"].map(GRADE_SCORE)
    best = (
        pre.dropna(subset=["grade_score"])
        .sort_values(["SEQN", "grade_score", "SPAMANU"], ascending=[True, True, True])
        .groupby("SEQN", as_index=False)
        .first()[["SEQN", "grade", "grade_score"]]
    )
    return best


def drop_bias_audit(raw: pd.DataFrame, selected_truth: pd.DataFrame, truth: pd.DataFrame) -> dict:
    tier2 = truth[
        truth["fev1_fvc_ratio"].notna()
        & truth["effort_quality"].isin(["A", "B"])
        & truth["fev1_quality"].isin(["A", "B"])
        & (truth["fev1_fvc_ratio"] < 0.60)
    ].copy()
    retained_seqn = set(selected_truth[selected_truth["SEQN"].isin(tier2["SEQN"])]["SEQN"])
    audit = tier2.copy()
    audit["retained"] = audit["SEQN"].isin(retained_seqn)
    grades = participant_best_grade(raw)
    audit = audit.merge(grades, on="SEQN", how="left")

    def summarize(group: pd.DataFrame) -> dict:
        ratio = group["fev1_fvc_ratio"].dropna()
        grades_present = group["grade_score"].dropna()
        return {
            "n": int(len(group)),
            "ratio_median": float(np.nanmedian(ratio)) if len(ratio) else np.nan,
            "ratio_iqr": q(ratio),
            "mean_best_raw_spaqeff_grade_score": float(np.nanmean(grades_present)) if len(grades_present) else np.nan,
            "best_raw_spaqeff_grade_counts": {str(k): int(v) for k, v in group["grade"].value_counts(dropna=False).to_dict().items()},
        }

    retained = summarize(audit[audit["retained"]])
    dropped = summarize(audit[~audit["retained"]])
    skew = "not evaluable"
    if np.isfinite(retained["ratio_median"]) and np.isfinite(dropped["ratio_median"]):
        if dropped["ratio_median"] < retained["ratio_median"]:
            skew = "drops are skewed toward lower FEV1/FVC ratios"
        elif dropped["ratio_median"] > retained["ratio_median"]:
            skew = "drops are skewed toward higher FEV1/FVC ratios"
        else:
            skew = "dropped and retained medians are equal"
    return {"retained": retained, "dropped": dropped, "skew_statement": skew}


def table_signed(summaries: list[dict]) -> str:
    rows = ["| group | n | signed FEV1 % median | IQR | signed FVC % median | IQR | signed ratio median | IQR |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        rows.append(
            f"| {row['group']} | {row['n']} | {row['signed_fev1_pct_median']:.2f}% | "
            f"{row['signed_fev1_pct_iqr'][0]:.2f}% to {row['signed_fev1_pct_iqr'][1]:.2f}% | "
            f"{row['signed_fvc_pct_median']:.2f}% | {row['signed_fvc_pct_iqr'][0]:.2f}% to {row['signed_fvc_pct_iqr'][1]:.2f}% | "
            f"{row['signed_ratio_median']:.4f} | {row['signed_ratio_iqr'][0]:.4f} to {row['signed_ratio_iqr'][1]:.4f} |"
        )
    return "\n".join(rows)


def table_conventions(rows: list[dict]) -> str:
    out = ["| scope | convention | n | signed ratio median | IQR | IQR brackets zero |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        out.append(
            f"| {row['scope']} | {row['variant']} | {row['n']} | {row['signed_ratio_median']:.4f} | "
            f"{row['signed_ratio_iqr'][0]:.4f} to {row['signed_ratio_iqr'][1]:.4f} | {row['iqr_brackets_zero']} |"
        )
    return "\n".join(out)


def table_classification(side: dict) -> str:
    rows = ["| definition | convention | sensitivity | specificity | agreement | kappa | TP | TN | FP | FN |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for definition, baseline_key, preferred_key in [
        ("fixed 0.70", "fixed_baseline", "fixed_preferred"),
        ("GLI LLN", "lln_baseline", "lln_preferred"),
    ]:
        for label, key in [("baseline", baseline_key), ("preferred", preferred_key)]:
            m = side[key]
            rows.append(
                f"| {definition} | {label} | {m['sensitivity']:.3f} | {m['specificity']:.3f} | {m['agreement']:.3f} | "
                f"{m['cohens_kappa']:.3f} | {m['tp']} | {m['tn']} | {m['fp']} | {m['fn']} |"
            )
    return "\n".join(rows)


def table_borderline(metrics: dict) -> str:
    rows = ["| convention | n | correct obstructed fraction | pushed-to-normal fraction | false negatives | signed ratio median | IQR |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label in ["baseline", "preferred"]:
        m = metrics[label]
        rows.append(
            f"| {label} | {m['n']} | {m['correctly_classified_obstructed_fraction']:.3f} | "
            f"{m['pushed_to_normal_fraction']:.3f} | {m['false_negative_n']} | {m['signed_ratio_median']:.4f} | "
            f"{m['signed_ratio_iqr'][0]:.4f} to {m['signed_ratio_iqr'][1]:.4f} |"
        )
    return "\n".join(rows)


def write_outputs(
    root: Path,
    result: dict,
    rows: pd.DataFrame,
    signed: list[dict],
    conventions: list[dict],
    side: dict,
    borderline: dict,
    drop_audit: dict,
) -> None:
    results_dir = root / "results"
    lines = [
        "# Gate - FVC Correction + Borderline Stress Test, Cycle F",
        "",
        f"- VERDICT: {result['verdict']} - {result['verdict_reason']}",
        "- model tested: plain bi-exponential OMBE in the volume domain",
        "- excluded methods: gamma onset, time registration, ML residual correction",
        f"- random seed: {RANDOM_SEED}",
        "",
        "## Step 1 - Signed-Error Diagnosis",
        "",
        table_signed(signed),
        "",
        f"Mechanism verdict: {result['mechanism_verdict']}",
        "",
        "## Step 2 - FVC Convention Comparison",
        "",
        "- baseline: prior gate convention, fitted model FVC/end-volume.",
        "- variant_a_model_raw_eot: fitted model volume at raw end-of-test. Raw EOT is available for all selected manoeuvres, so no 6 s fallback was used.",
        "- variant_b_observed_raw_eot: observed selected-manoeuvre end-expiratory raw volume is used as FVC; the model still supplies FEV1.",
        "",
        table_conventions(conventions),
        "",
        f"Preferred convention: `{result['preferred_variant']}` because its obstructed-set signed ratio median is closest to zero "
        f"({result['preferred_signed_ratio_median']:.4f}) with IQR {result['preferred_signed_ratio_iqr'][0]:.4f} to {result['preferred_signed_ratio_iqr'][1]:.4f}.",
        "",
        "## Step 3 - Classification Retest Under Preferred FVC",
        "",
        table_classification(side),
        "",
        f"Specificity guard: {result['specificity_guard']}",
        "",
        "### Fixed 0.70 Confusion Matrices",
        "",
        "Baseline:",
        "",
        side["fixed_baseline_table"].to_markdown(),
        "",
        "Preferred:",
        "",
        side["fixed_preferred_table"].to_markdown(),
        "",
        "## Step 4 - Borderline-Band Stress Test",
        "",
        f"- borderline available with usable selected A/B raw curves: {borderline['meta']['borderline_available_n']}",
        f"- borderline sampled: {borderline['meta']['borderline_sample_n']}",
        f"- fit-valid sampled rows: {borderline['meta']['borderline_fit_valid_n']}",
        f"- available-but-not-sampled count: {borderline['meta']['borderline_drop_n']}",
        "",
        table_borderline(borderline["metrics"]),
        "",
        "## Step 5 - Drop-Bias Audit",
        "",
        f"Retained obstructed participants: n={drop_audit['retained']['n']}, NHANES ratio median={drop_audit['retained']['ratio_median']:.4f}, "
        f"IQR={drop_audit['retained']['ratio_iqr'][0]:.4f} to {drop_audit['retained']['ratio_iqr'][1]:.4f}, "
        f"mean best raw SPAQEFF grade score={drop_audit['retained']['mean_best_raw_spaqeff_grade_score']:.2f}.",
        "",
        f"Dropped obstructed participants: n={drop_audit['dropped']['n']}, NHANES ratio median={drop_audit['dropped']['ratio_median']:.4f}, "
        f"IQR={drop_audit['dropped']['ratio_iqr'][0]:.4f} to {drop_audit['dropped']['ratio_iqr'][1]:.4f}, "
        f"mean best raw SPAQEFF grade score={drop_audit['dropped']['mean_best_raw_spaqeff_grade_score']:.2f}. "
        f"{drop_audit['skew_statement']}.",
        "",
        f"VERDICT: {result['verdict']} - {result['verdict_reason']}",
        "",
    ]
    (results_dir / "gate_fvc_correction_F.md").write_text("\n".join(lines), encoding="utf-8")
    (results_dir / "gate_fvc_correction_F.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    rows.to_csv(results_dir / "gate_fvc_correction_F_rows.csv", index=False)


def write_mechanism_halt(root: Path, result: dict, rows: pd.DataFrame, signed: list[dict]) -> None:
    results_dir = root / "results"
    lines = [
        "# Gate - FVC Correction + Borderline Stress Test, Cycle F",
        "",
        f"- VERDICT: {result['verdict']} - {result['verdict_reason']}",
        "- model tested: plain bi-exponential OMBE in the volume domain",
        "- excluded methods: gamma onset, time registration, ML residual correction",
        f"- random seed: {RANDOM_SEED}",
        "",
        "## Step 1 - Signed-Error Diagnosis",
        "",
        table_signed(signed),
        "",
        f"Mechanism verdict: {result['mechanism_verdict']}",
        "",
        "Step 1 did not confirm the FVC-underestimate mechanism, so Steps 2-5 were not run.",
        "",
        f"VERDICT: {result['verdict']} - {result['verdict_reason']}",
        "",
    ]
    (results_dir / "gate_fvc_correction_F.md").write_text("\n".join(lines), encoding="utf-8")
    (results_dir / "gate_fvc_correction_F.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    rows.to_csv(results_dir / "gate_fvc_correction_F_rows.csv", index=False)


def run(root: Path) -> dict:
    column_map = load_column_map(root)
    truth = load_truth(root, column_map)
    gli = load_gli(root)
    raw_path, _ = find_or_unzip_raw(root)
    raw, _ = read_raw(raw_path)

    prior_path = root / "results" / "gate_clinical_classification_F_rows.csv"
    if not prior_path.exists():
        raise RuntimeError(f"Missing prior clinical gate rows: {prior_path}")
    prior = pd.read_csv(prior_path)
    prior = add_fvc_conventions(prior)

    obstructed = prior[prior["sample_group"] == "obstructed"].copy()
    normal = prior[prior["sample_group"] == "normal"].copy()
    signed = [
        signed_summary(obstructed, group_label="obstructed"),
        signed_summary(normal, group_label="normal"),
        signed_summary(prior, group_label="pooled"),
    ]
    confirmed, mechanism_message = mechanism_confirmed(signed[0])
    if not confirmed:
        result = {
            "verdict": "MECHANISM-NOT-FVC",
            "verdict_reason": mechanism_message,
            "mechanism_verdict": mechanism_message,
            "random_seed": RANDOM_SEED,
        }
        write_mechanism_halt(root, result, prior, signed)
        return result

    conventions = convention_summary(obstructed, "obstructed") + convention_summary(normal, "normal")
    preferred = choose_preferred(conventions)
    preferred_ratio_col = {
        "baseline": "ratio_baseline",
        "variant_a_model_raw_eot": "ratio_model_raw_eot",
        "variant_b_observed_raw_eot": "ratio_observed_raw_eot",
    }[preferred["variant"]]

    side = classification_side_by_side(prior, preferred_ratio_col)
    specificity_guard = (
        "PASS: preferred fixed-threshold specificity remains >=0.90"
        if side["fixed_preferred"]["specificity"] >= 0.90
        else "FLAG: preferred fixed-threshold specificity dropped below 0.90"
    )

    _, selected_truth = prepare_selected_truth(root, raw, truth)
    borderline_rows, borderline_meta = build_borderline_rows(raw, selected_truth, gli)
    borderline_metrics_out = {
        "baseline": borderline_metrics(borderline_rows, "ratio_baseline"),
        "preferred": borderline_metrics(borderline_rows, preferred_ratio_col),
    }

    drop_audit = drop_bias_audit(raw, selected_truth, truth)
    baseline_borderline_fn = borderline_metrics_out["baseline"]["pushed_to_normal_fraction"]
    preferred_borderline_fn = borderline_metrics_out["preferred"]["pushed_to_normal_fraction"]
    bias_corrected = (
        -0.01 <= preferred["signed_ratio_median"] <= 0.01
        and side["fixed_preferred"]["specificity"] >= 0.90
        and preferred_borderline_fn <= baseline_borderline_fn
    )
    if bias_corrected:
        verdict = "BIAS-CORRECTED"
        reason = "preferred convention drives signed ratio median into [-0.01,+0.01], preserves specificity >=0.90, and does not worsen borderline false negatives."
    else:
        verdict = "PARTIAL"
        reasons = []
        if not (-0.01 <= preferred["signed_ratio_median"] <= 0.01):
            reasons.append("bias reduced but signed ratio median is not within [-0.01,+0.01]")
        if side["fixed_preferred"]["specificity"] < 0.90:
            reasons.append("specificity dropped below 0.90")
        if preferred_borderline_fn > baseline_borderline_fn:
            reasons.append("borderline false negatives worsened")
        reason = "; ".join(reasons) if reasons else "bias reduced but one or more correction criteria did not clear."

    result = {
        "verdict": verdict,
        "verdict_reason": reason,
        "mechanism_verdict": mechanism_message,
        "random_seed": RANDOM_SEED,
        "preferred_variant": preferred["variant"],
        "preferred_ratio_col": preferred_ratio_col,
        "preferred_signed_ratio_median": preferred["signed_ratio_median"],
        "preferred_signed_ratio_iqr": preferred["signed_ratio_iqr"],
        "signed_error_summaries": signed,
        "convention_summaries": conventions,
        "classification": {
            key: value
            for key, value in side.items()
            if not isinstance(value, pd.DataFrame)
        },
        "specificity_guard": specificity_guard,
        "borderline": {"meta": borderline_meta, "metrics": borderline_metrics_out},
        "drop_bias_audit": drop_audit,
    }
    combined_rows = pd.concat([prior.assign(analysis_sample="prior_obstructed_normal"), borderline_rows.assign(analysis_sample="borderline")], ignore_index=True)
    write_outputs(root, result, combined_rows, signed, conventions, side, result["borderline"], drop_audit)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    print(f"FVC correction gate: {result['verdict']} - {result['verdict_reason']}")
    if "preferred_variant" in result:
        print(f"Preferred convention: {result['preferred_variant']}")
        print(f"Preferred signed ratio median: {result['preferred_signed_ratio_median']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
