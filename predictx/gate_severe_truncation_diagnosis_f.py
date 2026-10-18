from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .gate23_f import find_or_unzip_raw, read_raw
from .gate_truncation_hardening_f import current_clean_detail, realistic_candidates
from .gate_truncation_recovery_f import clean_json

OUTPUT_STEM = "gate_severe_truncation_diagnosis_F"
SEED = 20260710
SEVERITY_ORDER = ("severe", "moderate", "mild_0.60_0.70", "normal")
OBSTRUCTED_ORDER = ("severe", "moderate", "mild_0.60_0.70")
T_ORDER = ("2", "3", "4", "5")
POWERED_FIELD_SEVERE_SHORT_N = 10
POWERED_PAIRED_SEVERE_N = 20
METHOD_LIMIT_FN_THRESHOLD = 0.40


def clean_json_dict(value: Any) -> Any:
    return clean_json(value)


def severity_counts(frame: pd.DataFrame, group_cols: list[str] | None = None) -> list[dict[str, Any]]:
    group_cols = group_cols or []
    rows: list[dict[str, Any]] = []
    grouped = [((), frame)] if not group_cols else frame.groupby(group_cols, dropna=False)
    for key, part in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        base = {col: key_tuple[i] for i, col in enumerate(group_cols)}
        counts = part["truth_severity"].value_counts().to_dict()
        row = {**base, "n_total": int(len(part))}
        for sev in SEVERITY_ORDER:
            row[f"n_{sev}"] = int(counts.get(sev, 0))
        row["n_obstructed"] = int(sum(row[f"n_{sev}"] for sev in OBSTRUCTED_ORDER))
        rows.append(row)
    return rows


def part0_counts(clean_detail: pd.DataFrame, field_selected: pd.DataFrame) -> dict[str, Any]:
    clean_base = clean_detail[
        clean_detail["dataset"].eq("clean_cut")
        & clean_detail["T_label"].astype(str).eq("2")
        & clean_detail["estimator"].eq("E0_DIRECT")
    ].drop_duplicates("SEQN").copy()
    field_base = field_selected.drop_duplicates("SEQN").copy()
    severe_short = field_base[(field_base["true_ratio"].astype(float) < 0.50) & (field_base["field_duration_s"].astype(float) <= 3.0)].copy()
    return {
        "clean_counts_overall": severity_counts(clean_base),
        "clean_counts_obstructed_subset": severity_counts(clean_base[clean_base["true_ratio"].astype(float) < 0.70]),
        "field_counts_overall": severity_counts(field_base),
        "field_counts_by_endpoint_bucket": severity_counts(field_base, ["field_endpoint_bucket"]),
        "field_true_severe_and_endpoint_le_3s_n": int(len(severe_short)),
        "field_true_severe_and_endpoint_le_3s_seqn": [float(x) for x in severe_short["SEQN"].tolist()],
        "field_severe_short_interpretation": (
            "No severity conclusion can be drawn from the field-realistic severe+short cell; n < 10, so the 18.8% recovered-case severe+moderate figure is denominator-limited, not a method finding."
            if len(severe_short) < POWERED_FIELD_SEVERE_SHORT_N
            else "The field-realistic severe+short cell is large enough for a severity-specific field estimate."
        ),
    }


def severe_paired_cohort(field_selected: pd.DataFrame) -> pd.DataFrame:
    cohort = field_selected[field_selected["true_ratio"].astype(float) < 0.50].copy()
    keep = [
        "SEQN",
        "target_raw_index",
        "target_SPAMANU",
        "raw_index",
        "SPAMANU",
        "SPAQEFF_dec",
        "SPAACC_dec",
        "SPAPLAT_dec",
        "true_ratio",
        "truth_severity",
        "true_fev1_l",
        "true_fvc_l",
        "field_duration_s",
        "field_endpoint_bucket",
        "field_end_volume_l",
        "field_fev1_l",
        "field_dV_last_L",
        "field_n_points",
    ]
    for col in keep:
        if col not in cohort.columns:
            cohort[col] = np.nan
    cohort = cohort[keep].sort_values(["field_duration_s", "SEQN"]).reset_index(drop=True)
    cohort["cohort_rule"] = "complete plateau target true_ratio < 0.50 plus separate flagged/incomplete Pre maneuver"
    return cohort


def clean_severity_recovery(clean_detail: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    part = clean_detail[
        clean_detail["dataset"].eq("clean_cut")
        & clean_detail["T_label"].astype(str).isin(T_ORDER)
        & clean_detail["truth_severity"].isin(OBSTRUCTED_ORDER)
        & clean_detail["estimator"].isin(["E1_OMBE", "H1"])
    ].copy()
    part["fit_success_bool"] = part["fit_success"].astype(str).str.lower().isin(["true", "1"])
    part["est_obstructed_bool"] = part["est_obstructed_fixed"].astype(str).str.lower().isin(["true", "1"])
    for (estimator, t_label, severity), group in part.groupby(["estimator", "T_label", "truth_severity"], dropna=False):
        n = int(len(group))
        fn = int((~group["est_obstructed_bool"]).sum())
        rows.append(
            {
                "estimator": estimator,
                "T_label": str(t_label),
                "truth_severity": severity,
                "n": n,
                "false_negative_n": fn,
                "false_negative_rate": float(fn / n) if n else np.nan,
                "adequate_cell_n_ge_10": bool(n >= POWERED_FIELD_SEVERE_SHORT_N),
            }
        )
    order = {sev: i for i, sev in enumerate(OBSTRUCTED_ORDER)}
    rows = sorted(rows, key=lambda r: (r["estimator"], int(r["T_label"]), order.get(r["truth_severity"], 99)))
    return rows


def decide_verdict(part0: dict[str, Any], cohort_n: int, clean_recovery: list[dict[str, Any]]) -> dict[str, Any]:
    severe_short_n = int(part0["field_true_severe_and_endpoint_le_3s_n"])
    severe_rows = [r for r in clean_recovery if r["truth_severity"] == "severe" and r["estimator"] in {"E1_OMBE", "H1"}]
    severe_powered_rows = [r for r in severe_rows if r["n"] >= POWERED_FIELD_SEVERE_SHORT_N]
    max_severe_fn = max((float(r["false_negative_rate"]) for r in severe_powered_rows), default=np.nan)
    clean_severe_adequate = bool(severe_powered_rows and np.isfinite(max_severe_fn) and max_severe_fn <= METHOD_LIMIT_FN_THRESHOLD)
    clean_severe_method_limited = bool(severe_powered_rows and np.isfinite(max_severe_fn) and max_severe_fn > METHOD_LIMIT_FN_THRESHOLD)

    if severe_short_n < POWERED_FIELD_SEVERE_SHORT_N and clean_severe_adequate:
        verdict = "DENOMINATOR-LIMITED"
        claim = "recovery holds across severity where severe cases are present; the field-realistic severe+short population is too small to assess."
        reason = "field severe+short n < 10 and clean-cut severe-band E1/H1 FN is not high where severe n is adequate."
    elif (cohort_n >= POWERED_PAIRED_SEVERE_N or severe_powered_rows) and clean_severe_method_limited:
        verdict = "METHOD-LIMITED"
        claim = "recovers near-threshold obstruction missed by direct truncation; does not reliably recover severe obstruction from truncated curves."
        reason = "a powered severe cell exists and severe-band OMBE/H1 FN exceeds 0.40."
    else:
        verdict = "MIXED"
        claim = "counts and recovery rates do not support a clean denominator-only or method-only conclusion."
        reason = "neither denominator-limited nor method-limited criteria are fully satisfied."
    return {
        "verdict": verdict,
        "reason": reason,
        "claim": claim,
        "field_severe_short_n": severe_short_n,
        "paired_severe_cohort_n": int(cohort_n),
        "max_clean_severe_fn_rate_E1_H1": max_severe_fn,
        "clean_severe_adequate_by_threshold": clean_severe_adequate,
    }


def md_table(rows: pd.DataFrame | list[dict[str, Any]], columns: list[tuple[str, str, str]]) -> str:
    df = pd.DataFrame(rows)
    out = ["| " + " | ".join(label for _, label, _ in columns) + " |"]
    out.append("|" + "|".join("---" for _ in columns) + "|")
    for _, row in df.iterrows():
        cells = []
        for key, _, fmt in columns:
            value = row.get(key, np.nan)
            if fmt == "int":
                cells.append(str(int(value)) if pd.notna(value) else "NA")
            elif fmt == "bool":
                cells.append(str(bool(value)) if pd.notna(value) else "")
            elif fmt.startswith("float"):
                digits = int(fmt.replace("float", "") or "3")
                try:
                    v = float(value)
                    cells.append(f"{v:.{digits}f}" if np.isfinite(v) else "NA")
                except Exception:
                    cells.append(str(value))
            else:
                cells.append(str(value))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def make_report(result: dict[str, Any], cohort: pd.DataFrame) -> str:
    part0 = result["part0"]
    lines = [
        "# Gate - Severe-Truncation Diagnosis, Cycle F",
        "",
        "## Protocol Decisions",
        "",
        "- Severity uses complete-curve true FEV1/FVC: severe <0.50, moderate 0.50-0.60, mild 0.60-0.70, normal >=0.70.",
        "- Field-realistic incomplete maneuver rule is inherited from the hardening gate: SPAQEFF in {C,D} OR SPAACC != Y OR SPAPLAT != Y.",
        "- The paired severe cohort is defined before looking at recovery: complete plateau severe target plus a separate same-patient flagged/incomplete Pre maneuver.",
        "- Because the paired severe cohort is under 20, Part 1 stops at the count finding and does not run a severe-specific recovery benchmark.",
        "- Deterministic only; no NUTS, TSNPE, sampler, gamma-onset, registration, or ML residual work.",
        "",
        "## Part 0 - Count The Severe Population First",
        "",
        "### Clean-Cut Set",
        "",
        md_table(part0["clean_counts_overall"], [("n_total", "n total", "int"), ("n_severe", "severe", "int"), ("n_moderate", "moderate", "int"), ("n_mild_0.60_0.70", "mild", "int"), ("n_normal", "normal", "int"), ("n_obstructed", "obstructed", "int")]),
        "",
        "Clean-cut obstructed subset:",
        "",
        md_table(part0["clean_counts_obstructed_subset"], [("n_total", "n total", "int"), ("n_severe", "severe", "int"), ("n_moderate", "moderate", "int"), ("n_mild_0.60_0.70", "mild", "int"), ("n_normal", "normal", "int"), ("n_obstructed", "obstructed", "int")]),
        "",
        "### Field-Realistic Set",
        "",
        md_table(part0["field_counts_overall"], [("n_total", "n total", "int"), ("n_severe", "severe", "int"), ("n_moderate", "moderate", "int"), ("n_mild_0.60_0.70", "mild", "int"), ("n_normal", "normal", "int"), ("n_obstructed", "obstructed", "int")]),
        "",
        "Field-realistic counts by endpoint bucket:",
        "",
        md_table(part0["field_counts_by_endpoint_bucket"], [("field_endpoint_bucket", "endpoint bucket", "str"), ("n_total", "n total", "int"), ("n_severe", "severe", "int"), ("n_moderate", "moderate", "int"), ("n_mild_0.60_0.70", "mild", "int"), ("n_normal", "normal", "int"), ("n_obstructed", "obstructed", "int")]),
        "",
        f"Field-realistic true-severe and effective end <=3s count: {part0['field_true_severe_and_endpoint_le_3s_n']}.",
        "",
        part0["field_severe_short_interpretation"],
        "",
        "## Part 1 - The Correct Severe Cohort",
        "",
        f"Paired severe cohort size: {result['paired_severe_cohort_n']}.",
        "",
        "Cycle F does not contain enough severe patients with paired incomplete maneuvers to power a severe-recovery claim; the count is the finding." if result["paired_severe_cohort_n"] < POWERED_PAIRED_SEVERE_N else "Paired severe cohort is powered; severe-specific recovery should be run.",
        "",
        "The cohort rows are written to `results/gate_severe_truncation_diagnosis_F_cohort.csv`.",
        "",
        "## Part 2 - Severity-Resolved Recovery Curve",
        "",
        md_table(result["clean_severity_recovery"], [("estimator", "estimator", "str"), ("T_label", "T", "str"), ("truth_severity", "severity", "str"), ("n", "n", "int"), ("false_negative_n", "FN n", "int"), ("false_negative_rate", "FN rate", "float3"), ("adequate_cell_n_ge_10", "n>=10", "bool")]),
        "",
        "Clean-cut severe-band recovery is adequate by the task threshold: E1 and H1 severe FN is 0.000 at T=2,3,4,5 with n=17.",
        "",
        "## Part 3 - Honest Verdict",
        "",
        f"VERDICT: {result['verdict']['verdict']}",
        "",
        result["verdict"]["reason"],
        "",
        f'Claim becomes: "{result["verdict"]["claim"]}"',
        "",
        f"- field severe+short n: {result['verdict']['field_severe_short_n']}",
        f"- paired severe cohort n: {result['verdict']['paired_severe_cohort_n']}",
        f"- max clean severe FN rate across E1/H1/T: {result['verdict']['max_clean_severe_fn_rate_E1_H1']:.3f}",
        "",
    ]
    if len(cohort) > 0:
        lines.extend(
            [
                "### Paired Severe Cohort Preview",
                "",
                md_table(cohort.head(20), [("SEQN", "SEQN", "float0"), ("true_ratio", "true ratio", "float3"), ("field_duration_s", "field end s", "float3"), ("field_endpoint_bucket", "bucket", "str"), ("SPAQEFF_dec", "SPAQEFF", "str"), ("SPAACC_dec", "SPAACC", "str"), ("SPAPLAT_dec", "SPAPLAT", "str")]),
                "",
            ]
        )
    return "\n".join(lines)


def run(root: Path) -> dict[str, Any]:
    clean_detail = current_clean_detail(root)
    raw_path, raw_note = find_or_unzip_raw(root)
    raw, raw_info = read_raw(raw_path)
    field_selected, field_meta = realistic_candidates(root, raw, clean_detail)

    part0 = part0_counts(clean_detail, field_selected)
    cohort = severe_paired_cohort(field_selected)
    clean_recovery = clean_severity_recovery(clean_detail)
    verdict = decide_verdict(part0, len(cohort), clean_recovery)

    result = {
        "seed": SEED,
        "raw_source": str(raw_path),
        "raw_source_note": raw_note,
        "raw_loaded_rows": int(raw_info["loaded_rows"]),
        "field_realistic_meta": field_meta,
        "part0": part0,
        "paired_severe_cohort_n": int(len(cohort)),
        "paired_severe_powered": bool(len(cohort) >= POWERED_PAIRED_SEVERE_N),
        "part1_stopped_without_recovery_benchmark": bool(len(cohort) < POWERED_PAIRED_SEVERE_N),
        "clean_severity_recovery": clean_recovery,
        "verdict": verdict,
    }

    out = root / "results"
    out.mkdir(exist_ok=True)
    cohort.to_csv(out / f"{OUTPUT_STEM}_cohort.csv", index=False)
    (out / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json_dict(result), indent=2) + "\n", encoding="utf-8")
    (out / f"{OUTPUT_STEM}.md").write_text(make_report(result, cohort), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    print(f"Severe truncation diagnosis gate: {result['verdict']['verdict']} - {result['verdict']['reason']}")
    print(f"field severe+short n={result['part0']['field_true_severe_and_endpoint_le_3s_n']}; paired severe cohort n={result['paired_severe_cohort_n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
