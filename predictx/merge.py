from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import CYCLES, data_path, read_xpt

OUTPUT_FIELDS = {
    "participant_id": "SEQN",
    "age_years": "age_years",
    "sex": "sex",
    "race_ethnicity": "race_ethnicity",
    "height_cm": "height_cm",
    "weight_kg": "weight_kg",
    "fev1_mL": "fev1_mL",
    "fvc_mL": "fvc_mL",
    "pef_mL_s": "pef_mL_s",
    "fev6_mL": "fev6_mL",
    "spirometry_status": "spirometry_status",
    "fvc_quality": "fvc_quality",
    "fev1_quality": "fev1_quality",
    "acceptable_curve_count": "acceptable_curve_count",
    "effort_quality": "effort_quality",
    "bronchodilator_selected": "bronchodilator_selected",
    "survey_weight_mec_2yr": "WTMEC2YR",
    "survey_psu": "SDMVPSU",
    "survey_strata": "SDMVSTRA",
}

QUALITY_COLUMNS = ("fvc_quality", "fev1_quality", "effort_quality")


def run_merge(data_dir: Path, results_dir: Path, column_map: dict | None = None) -> pd.DataFrame:
    if column_map is None:
        column_map = json.loads((results_dir / "column_map.json").read_text(encoding="utf-8"))

    frames: list[pd.DataFrame] = []
    report: list[dict[str, object]] = []
    for cycle in CYCLES:
        demo = read_xpt(data_path(data_dir, "DEMO", cycle)).frame
        bmx = read_xpt(data_path(data_dir, "BMX", cycle)).frame
        spx = read_xpt(data_path(data_dir, "SPX", cycle)).frame

        seqn = _field(column_map, "participant_id", cycle)
        demo_selected = _select_fields(demo, column_map, cycle, "DEMO")
        bmx_selected = _select_fields(bmx, column_map, cycle, "BMX")
        spx_selected = _select_fields(spx, column_map, cycle, "SPX")

        merged = demo_selected.merge(bmx_selected, on=seqn, how="left", indicator="bmx_join")
        merged = merged.merge(spx_selected, on=seqn, how="left", indicator="spx_join")
        merged.insert(1, "cycle", cycle)
        merged = _rename_outputs(merged, column_map, cycle)
        merged["fev1_fvc_ratio"] = np.where(
            merged["fvc_mL"].notna() & (merged["fvc_mL"] != 0),
            merged["fev1_mL"] / merged["fvc_mL"],
            np.nan,
        )

        report.append(
            {
                "cycle": cycle,
                "demo_rows": int(len(demo)),
                "bmx_rows": int(len(bmx)),
                "spx_rows": int(len(spx)),
                "merged_rows_left_join": int(len(merged)),
                "bmx_matched": int((merged["bmx_join"] == "both").sum()),
                "bmx_missing_from_demo_base": int((merged["bmx_join"] == "left_only").sum()),
                "spx_matched": int((merged["spx_join"] == "both").sum()),
                "spx_missing_from_demo_base": int((merged["spx_join"] == "left_only").sum()),
            }
        )
        frames.append(merged.drop(columns=["bmx_join", "spx_join"]))

    combined = pd.concat(frames, ignore_index=True)
    ordered = [
        "SEQN",
        "cycle",
        "age_years",
        "sex",
        "race_ethnicity",
        "height_cm",
        "weight_kg",
        "fev1_mL",
        "fvc_mL",
        "fev1_fvc_ratio",
        "pef_mL_s",
        "fev6_mL",
        "spirometry_status",
        "fvc_quality",
        "fev1_quality",
        "acceptable_curve_count",
        "effort_quality",
        "bronchodilator_selected",
        "WTMEC2YR",
        "SDMVPSU",
        "SDMVSTRA",
    ]
    combined = combined[ordered]
    combined = normalize_string_columns(combined)
    combined.to_parquet(results_dir / "merged_summary.parquet", index=False)
    write_merge_report(combined, report, results_dir / "merge_report.md")
    return combined


def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].map(_decode_cell)
    for col in QUALITY_COLUMNS:
        if col in out.columns:
            out[col] = out[col].replace("", pd.NA)
    return out


def write_merge_report(combined: pd.DataFrame, report: list[dict[str, object]], output_path: Path) -> None:
    lines = [
        "# Track A Merge Report",
        "",
        "Join strategy: DEMO is the base frame; BMX and SPX are left-joined on `SEQN`.",
        "",
        "| cycle | DEMO rows | BMX rows | SPX rows | merged rows | BMX matched | DEMO without BMX | SPX matched | DEMO without SPX |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report:
        lines.append(
            f"| {item['cycle']} | {item['demo_rows']} | {item['bmx_rows']} | {item['spx_rows']} | "
            f"{item['merged_rows_left_join']} | {item['bmx_matched']} | {item['bmx_missing_from_demo_base']} | "
            f"{item['spx_matched']} | {item['spx_missing_from_demo_base']} |"
        )
    lines.extend(["", "## Missingness in Merged Output", ""])
    missing = combined.isna().mean().mul(100).sort_values(ascending=False)
    lines.extend(["| column | missing_pct |", "|---|---:|"])
    for col, pct in missing.items():
        lines.append(f"| `{col}` | {pct:.2f} |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _select_fields(df: pd.DataFrame, column_map: dict, cycle: str, source_file: str) -> pd.DataFrame:
    seqn = _field(column_map, "participant_id", cycle)
    selected = {seqn}
    for logical, meta in column_map["field_sources"].items():
        if meta["source_file"] != source_file:
            continue
        actual = column_map["fields"][logical][cycle]
        if actual is not None:
            selected.add(actual)
    return df.loc[:, [col for col in df.columns if col in selected]].copy()


def _rename_outputs(df: pd.DataFrame, column_map: dict, cycle: str) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for logical, output in OUTPUT_FIELDS.items():
        actual = column_map["fields"][logical][cycle]
        if actual is not None and actual in df.columns:
            rename[actual] = output
    out = df.rename(columns=rename)
    for output in OUTPUT_FIELDS.values():
        if output not in out.columns:
            out[output] = np.nan
    return out


def _field(column_map: dict, logical: str, cycle: str) -> str:
    actual = column_map["fields"][logical][cycle]
    if actual is None:
        raise KeyError(f"Required field {logical!r} is unresolved for cycle {cycle}")
    return actual


def _decode_cell(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return value
