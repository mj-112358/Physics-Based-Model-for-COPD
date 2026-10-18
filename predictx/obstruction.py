from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

QUALITY_VALUES = {"A", "B"}


def run_obstruction_analysis(results_dir: Path) -> dict[str, dict[str, int]]:
    merged = pd.read_parquet(results_dir / "merged_summary.parquet")
    _assert_quality_strings(merged)

    gli = pd.read_csv(results_dir / "gli_reference.csv")
    joined = merged.merge(gli, on="SEQN", how="left", indicator="gli_join")

    join_log = {
        "merged_rows": int(len(merged)),
        "gli_reference_rows": int(len(gli)),
        "matched_rows": int((joined["gli_join"] == "both").sum()),
        "unmatched_rows": int((joined["gli_join"] == "left_only").sum()),
    }
    print(
        "GLI SEQN join: "
        f"matched={join_log['matched_rows']}, unmatched={join_log['unmatched_rows']}"
    )

    logs: list[tuple[str, int, int]] = []
    ratio_set = _filter_with_log(
        joined,
        joined["fev1_fvc_ratio"].notna(),
        "Restrict to fev1_fvc_ratio.notna()",
        logs,
    )
    quality_set = _filter_with_log(
        ratio_set,
        ratio_set["effort_quality"].isin(QUALITY_VALUES) & ratio_set["fev1_quality"].isin(QUALITY_VALUES),
        "Quality filter: effort_quality in {A,B} AND fev1_quality in {A,B}",
        logs,
    )
    usable = _filter_with_log(
        quality_set,
        quality_set["gli_lln_ratio"].notna(),
        "Require matched finite GLI LLN reference",
        logs,
    )

    ratio_join_log = {
        "ratio_rows": int(len(ratio_set)),
        "ratio_matched_gli": int(ratio_set["gli_lln_ratio"].notna().sum()),
        "ratio_unmatched_gli": int(ratio_set["gli_lln_ratio"].isna().sum()),
        "quality_rows": int(len(quality_set)),
        "quality_matched_gli": int(quality_set["gli_lln_ratio"].notna().sum()),
        "quality_unmatched_gli": int(quality_set["gli_lln_ratio"].isna().sum()),
    }

    fixed_mask = usable["fev1_fvc_ratio"] < 0.70
    lln_mask = usable["fev1_fvc_ratio"] < usable["gli_lln_ratio"]
    counts = {
        "fixed_0_70": _counts(usable, fixed_mask),
        "lln": _counts(usable, lln_mask),
    }
    adds_removes = _adds_removes_by_age(usable, fixed_mask, lln_mask)
    pattern_flag = _pattern_flag(adds_removes)

    write_report(
        results_dir / "obstruction_composition.md",
        merged,
        logs,
        join_log,
        ratio_join_log,
        counts,
        adds_removes,
        pattern_flag,
        _read_meta(results_dir),
        _read_gli_input_log(results_dir),
    )
    return counts


def write_report(
    output_path: Path,
    merged: pd.DataFrame,
    logs: list[tuple[str, int, int]],
    join_log: dict[str, int],
    ratio_join_log: dict[str, int],
    counts: dict[str, dict[str, int]],
    adds_removes: dict[str, dict[str, int]],
    pattern_flag: str,
    meta: dict[str, str],
    gli_input_log: dict[str, int],
) -> None:
    fixed = counts["fixed_0_70"]
    lln = counts["lln"]
    lines = [
        "# Obstruction Composition",
        "",
        "## Phase 0 GLI Input Export",
        "",
        f"- merged rows: {gli_input_log['input_rows']}",
        f"- exported rows: {gli_input_log['exported_rows']}",
        f"- dropped for missing age/height/sex: {gli_input_log['dropped_missing_age_height_or_sex']}",
        "",
        "## Phase 1 Assertion",
        "",
        "Post-fix `effort_quality.value_counts(dropna=False)`:",
        "",
        "```text",
        merged["effort_quality"].value_counts(dropna=False).to_string(),
        "```",
        "",
        "## GLI Reference",
        "",
        f"- rspiro version: {meta.get('rspiro_version', 'unknown')}",
        f"- exact GLI functions called: {meta.get('gli_functions', 'unknown')}",
        f"- GLI input rows: {meta.get('input_rows', 'unknown')}",
        f"- GLI reference rows written: {meta.get('output_rows', 'unknown')}",
        f"- fallback input rows mapped to GLI Other/mixed: {meta.get('fallback_input_rows', 'unknown')}",
        f"- fallback output rows mapped to GLI Other/mixed: {meta.get('fallback_output_rows', 'unknown')}",
        f"- rows dropped because rspiro returned non-finite GLI values: {meta.get('dropped_nonfinite_gli_rows', 'unknown')}",
        f"- finite rows dropped by predicted FEV1 sanity range 0.5-6 L: {meta.get('dropped_pred_fev1_sanity_rows', 'unknown')}",
        "",
        "See `results/gli_mapping_notes.md` for mapping comments and validation notes.",
        "",
        "## SEQN Join Log",
        "",
        "| join scope | matched | unmatched |",
        "|---|---:|---:|",
        f"| merged -> GLI reference | {join_log['matched_rows']} | {join_log['unmatched_rows']} |",
        f"| ratio-not-null rows -> GLI reference | {ratio_join_log['ratio_matched_gli']} | {ratio_join_log['ratio_unmatched_gli']} |",
        f"| quality-filtered rows -> GLI reference | {ratio_join_log['quality_matched_gli']} | {ratio_join_log['quality_unmatched_gli']} |",
        "",
        "## Filter Log",
        "",
        "| filter | before | after | dropped |",
        "|---|---:|---:|---:|",
    ]
    for name, before, after in logs:
        lines.append(f"| {name} | {before} | {after} | {before - after} |")

    lines.extend(
        [
            "",
            "Repeatability note: `acceptable_curve_count >= 3` is not used here; true repeatability requires raw curves and is deferred.",
            "",
            "## Obstruction Composition",
            "",
            "| definition | n_total | n_obstructed | n_0_60_to_0_70 | n_0_50_to_0_60 | n_lt_0_50 | n_obstructed_gte_0_70 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            _count_row("Fixed <0.70", fixed),
            _count_row("LLN", lln),
            "",
            "## LLN Adds/Removes Versus Fixed <0.70",
            "",
            "| age band | LLN adds vs fixed | LLN removes vs fixed |",
            "|---|---:|---:|",
        ]
    )
    for age_band in ("<50", ">=50"):
        item = adds_removes[age_band]
        lines.append(f"| {age_band} | {item['adds']} | {item['removes']} |")
    lines.extend(
        [
            "",
            pattern_flag,
            "",
            "## Decision Inputs",
            "",
            "| definition | usable moderate 0.50-0.60 | usable severe <0.50 | combined moderate+severe |",
            "|---|---:|---:|---:|",
            f"| Fixed <0.70 | {fixed['n_0_50_to_0_60']} | {fixed['n_lt_0_50']} | {fixed['n_0_50_to_0_60'] + fixed['n_lt_0_50']} |",
            f"| LLN | {lln['n_0_50_to_0_60']} | {lln['n_lt_0_50']} | {lln['n_0_50_to_0_60'] + lln['n_lt_0_50']} |",
            "",
            f"Severe n under LLN = {lln['n_lt_0_50']}. Combined moderate+severe = {lln['n_0_50_to_0_60'] + lln['n_lt_0_50']}. Below ~300 combined means collapse to a single 'moderate-or-worse' stratum for coverage claims.",
            "",
            f"Headline: fixed severe = {fixed['n_lt_0_50']}; fixed moderate+severe = {fixed['n_0_50_to_0_60'] + fixed['n_lt_0_50']}; LLN severe = {lln['n_lt_0_50']}; LLN moderate+severe = {lln['n_0_50_to_0_60'] + lln['n_lt_0_50']}.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _assert_quality_strings(df: pd.DataFrame) -> None:
    value_counts = df["effort_quality"].value_counts(dropna=False)
    print("Post-fix effort_quality value_counts(dropna=False):")
    print(value_counts.to_string())
    non_null = df["effort_quality"].dropna()
    has_bytes = non_null.map(lambda value: isinstance(value, bytes)).any()
    if has_bytes:
        raise RuntimeError("Phase 1 assertion failed: effort_quality still contains bytes.")


def _filter_with_log(
    df: pd.DataFrame, mask: pd.Series, name: str, logs: list[tuple[str, int, int]]
) -> pd.DataFrame:
    before = int(len(df))
    after_df = df.loc[mask].copy()
    after = int(len(after_df))
    logs.append((name, before, after))
    print(f"{name}: before={before}, after={after}, dropped={before - after}")
    return after_df


def _counts(df: pd.DataFrame, obstructed_mask: pd.Series) -> dict[str, int]:
    obstructed = df.loc[obstructed_mask]
    obstructed_ratio = obstructed["fev1_fvc_ratio"]
    return {
        "n_total": int(len(df)),
        "n_obstructed": int(obstructed_mask.sum()),
        "n_0_60_to_0_70": int(((obstructed_ratio >= 0.60) & (obstructed_ratio < 0.70)).sum()),
        "n_0_50_to_0_60": int(((obstructed_ratio >= 0.50) & (obstructed_ratio < 0.60)).sum()),
        "n_lt_0_50": int((obstructed_ratio < 0.50).sum()),
        "n_obstructed_gte_0_70": int((obstructed_ratio >= 0.70).sum()),
    }


def _adds_removes_by_age(df: pd.DataFrame, fixed_mask: pd.Series, lln_mask: pd.Series) -> dict[str, dict[str, int]]:
    added = lln_mask & ~fixed_mask
    removed = fixed_mask & ~lln_mask
    return {
        "<50": {
            "adds": int((added & (df["age_years"] < 50)).sum()),
            "removes": int((removed & (df["age_years"] < 50)).sum()),
        },
        ">=50": {
            "adds": int((added & (df["age_years"] >= 50)).sum()),
            "removes": int((removed & (df["age_years"] >= 50)).sum()),
        },
    }


def _pattern_flag(adds_removes: dict[str, dict[str, int]]) -> str:
    adds_older = adds_removes[">=50"]["adds"]
    adds_younger = adds_removes["<50"]["adds"]
    removes_younger = adds_removes["<50"]["removes"]
    removes_older = adds_removes[">=50"]["removes"]
    if adds_older >= adds_younger and removes_younger >= removes_older and adds_older > 0:
        return "Pattern flag: LLN adds are concentrated in >=50 and removals are concentrated in <50."
    return "Pattern flag: expected age-band add/remove pattern did NOT appear; GLI mapping or units may be wrong."


def _count_row(label: str, item: dict[str, int]) -> str:
    return (
        f"| {label} | {item['n_total']} | {item['n_obstructed']} | "
        f"{item['n_0_60_to_0_70']} | {item['n_0_50_to_0_60']} | "
        f"{item['n_lt_0_50']} | {item['n_obstructed_gte_0_70']} |"
    )


def _read_meta(results_dir: Path) -> dict[str, str]:
    path = results_dir / "gli_reference_meta.csv"
    if not path.exists():
        return {}
    meta = pd.read_csv(path)
    return dict(zip(meta["key"].astype(str), meta["value"].astype(str)))


def _read_gli_input_log(results_dir: Path) -> dict[str, int]:
    path = results_dir / "gli_input_log.json"
    if not path.exists():
        return {"input_rows": 0, "exported_rows": 0, "dropped_missing_age_height_or_sex": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    run_obstruction_analysis(Path.cwd() / "results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
