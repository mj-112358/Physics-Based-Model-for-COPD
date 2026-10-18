from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .io import CYCLES, TRACK_A_FILES, XptData, data_path, example_values, read_xpt


@dataclass(frozen=True)
class FieldSpec:
    source_file: str
    exact: tuple[str, ...]
    patterns: tuple[str, ...] = ()
    optional: bool = False
    computed: bool = False


FIELD_SPECS: dict[str, FieldSpec] = {
    "participant_id": FieldSpec("ANY", ("SEQN",)),
    "age_years": FieldSpec("DEMO", ("RIDAGEYR",)),
    "sex": FieldSpec("DEMO", ("RIAGENDR",)),
    "race_ethnicity": FieldSpec("DEMO", ("RIDRETH3", "RIDRETH1")),
    "survey_weight_mec_2yr": FieldSpec("DEMO", ("WTMEC2YR",)),
    "survey_psu": FieldSpec("DEMO", ("SDMVPSU",)),
    "survey_strata": FieldSpec("DEMO", ("SDMVSTRA",)),
    "height_cm": FieldSpec("BMX", ("BMXHT",)),
    "weight_kg": FieldSpec("BMX", ("BMXWT",)),
    "fev1_mL": FieldSpec("SPX", ("SPXNFEV1",), (r"^SPXN.*FEV1$",)),
    "fvc_mL": FieldSpec("SPX", ("SPXNFVC",), (r"^SPXN.*FVC$",)),
    "fev1_fvc_ratio": FieldSpec("SPX", (), optional=True, computed=True),
    "pef_mL_s": FieldSpec("SPX", ("SPXNPEF",), (r"^SPXN.*PEF$",)),
    "fev6_mL": FieldSpec("SPX", ("SPXNFEV6",), (r"^SPXN.*FEV6$",)),
    "spirometry_status": FieldSpec("SPX", ("SPXNSTAT",)),
    "fvc_quality": FieldSpec("SPX", ("SPXNQFVC",)),
    "fev1_quality": FieldSpec("SPX", ("SPXNQFV1",)),
    "acceptable_curve_count": FieldSpec("SPX", ("SPDNACC",)),
    "effort_quality": FieldSpec("SPX", ("SPXNQEFF",)),
    "bronchodilator_selected": FieldSpec("SPX", ("SPDBRONC",), optional=True),
    "raw_curve": FieldSpec("SPXRAW", ("SPXRAW",), optional=True),
    "n_points": FieldSpec("SPXRAW", ("SPXPTS",), optional=True),
    "btps_factor": FieldSpec("SPXRAW", ("SPAFACT",), optional=True),
    "test_type": FieldSpec("SPXRAW", ("SPATTYPE",), optional=True),
    "maneuver_num": FieldSpec("SPXRAW", ("SPAMANU",), optional=True),
    "acceptable": FieldSpec("SPXRAW", ("SPAACC",), optional=True),
    "position": FieldSpec("SPXRAW", ("SPAPOS",), optional=True),
    "plateau": FieldSpec("SPXRAW", ("SPAPLAT",), optional=True),
    "raw_effort_grade": FieldSpec("SPXRAW", ("SPAQEFF",), optional=True),
}


def run_schema_discovery(data_dir: Path, results_dir: Path) -> tuple[dict[str, dict[str, XptData]], dict]:
    loaded: dict[str, dict[str, XptData]] = {}
    for cycle in CYCLES:
        loaded[cycle] = {}
        for file_type in TRACK_A_FILES:
            loaded[cycle][file_type] = read_xpt(data_path(data_dir, file_type, cycle))

    column_map = build_column_map(loaded)
    write_schema_dump(loaded, column_map, results_dir / "schema_dump.md")
    (results_dir / "column_map.json").write_text(
        json.dumps(column_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return loaded, column_map


def build_column_map(loaded: dict[str, dict[str, XptData]]) -> dict:
    fields: dict[str, dict[str, str | None]] = {}
    warnings: list[dict[str, object]] = []
    candidates_seen: dict[str, dict[str, list[str]]] = {}

    for logical, spec in FIELD_SPECS.items():
        fields[logical] = {}
        candidates_seen[logical] = {}
        for cycle in CYCLES:
            available = _available_columns(loaded, cycle, spec.source_file)
            resolved = _resolve_column(available, spec)
            fields[logical][cycle] = resolved
            candidates_seen[logical][cycle] = available
            if resolved is None and not spec.computed:
                warnings.append(
                    {
                        "cycle": cycle,
                        "field": logical,
                        "source_file": spec.source_file,
                        "expected": list(spec.exact),
                        "patterns": list(spec.patterns),
                        "candidate_columns": available,
                        "optional": spec.optional,
                    }
                )

    return {
        "version": 1,
        "cycles": list(CYCLES),
        "field_sources": {
            name: {
                "source_file": spec.source_file,
                "optional": spec.optional,
                "computed": spec.computed,
            }
            for name, spec in FIELD_SPECS.items()
        },
        "fields": fields,
        "warnings": warnings,
    }


def write_schema_dump(
    loaded: dict[str, dict[str, XptData]], column_map: dict, output_path: Path
) -> None:
    lines: list[str] = ["# NHANES Track A Schema Dump", ""]
    for cycle in CYCLES:
        lines.extend([f"## Cycle {cycle}", ""])
        for file_type in TRACK_A_FILES:
            xpt = loaded[cycle][file_type]
            df = xpt.frame
            lines.extend(
                [
                    f"### {xpt.path.name}",
                    "",
                    f"- n_rows: {len(df)}",
                    f"- n_columns: {len(df.columns)}",
                    f"- SEQN present: {'yes' if 'SEQN' in df.columns else 'no'}",
                    "",
                    "| column_name | dtype | SAS label | n_non_null | missing_pct | example_values |",
                    "|---|---:|---|---:|---:|---|",
                ]
            )
            for col in df.columns:
                non_null = int(df[col].notna().sum())
                missing_pct = float(df[col].isna().mean() * 100.0)
                examples = json.dumps(example_values(df[col]), ensure_ascii=False)
                label = _md(xpt.labels.get(col, ""))
                lines.append(
                    f"| `{col}` | `{df[col].dtype}` | {label} | {non_null} | {missing_pct:.2f} | `{_md(examples)}` |"
                )
            lines.append("")

    lines.extend(_drift_report(loaded))
    lines.extend(["## Resolved Column Map", "", "```json", json.dumps(column_map["fields"], indent=2), "```", ""])
    warnings = column_map.get("warnings", [])
    lines.extend(["## Resolution Warnings", ""])
    if warnings:
        for warning in warnings:
            lines.append(
                f"- Cycle {warning['cycle']} `{warning['field']}` unresolved in {warning['source_file']} "
                f"(optional={warning['optional']})."
            )
    else:
        lines.append("- None.")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _drift_report(loaded: dict[str, dict[str, XptData]]) -> list[str]:
    lines: list[str] = ["## Cross-Cycle Drift Checks", ""]
    for file_type in TRACK_A_FILES:
        lines.extend([f"### {file_type}", ""])
        colsets = {
            cycle: set(loaded[cycle][file_type].frame.columns)
            for cycle in CYCLES
        }
        all_cols = set().union(*colsets.values())
        common_cols = set.intersection(*colsets.values())
        lines.append(f"- common columns: {len(common_cols)}")
        for cycle in CYCLES:
            missing = sorted(all_cols - colsets[cycle])
            extra = sorted(colsets[cycle] - set().union(*(colsets[c] for c in CYCLES if c != cycle)))
            if missing:
                lines.append(f"- columns absent in cycle {cycle}: `{', '.join(missing)}`")
            if extra:
                lines.append(f"- columns unique to cycle {cycle}: `{', '.join(extra)}`")

        label_drift = _label_drift(loaded, file_type, common_cols)
        if label_drift:
            lines.append("- label drift:")
            lines.extend(f"  - `{col}`: {desc}" for col, desc in label_drift)
        else:
            lines.append("- label drift: none detected for common columns")

        coding_drift = _coding_drift(loaded, file_type, common_cols)
        if coding_drift:
            lines.append("- low-cardinality coding drift:")
            lines.extend(f"  - `{col}`: {desc}" for col, desc in coding_drift)
        else:
            lines.append("- low-cardinality coding drift: none detected")
        lines.append("")
    return lines


def _label_drift(
    loaded: dict[str, dict[str, XptData]], file_type: str, columns: Iterable[str]
) -> list[tuple[str, str]]:
    drift = []
    for col in sorted(columns):
        labels = {cycle: loaded[cycle][file_type].labels.get(col, "") for cycle in CYCLES}
        if len(set(labels.values())) > 1:
            drift.append((col, "; ".join(f"{c}={labels[c]!r}" for c in CYCLES)))
    return drift


def _coding_drift(
    loaded: dict[str, dict[str, XptData]], file_type: str, columns: Iterable[str]
) -> list[tuple[str, str]]:
    drift = []
    for col in sorted(columns):
        value_sets: dict[str, tuple[str, ...]] = {}
        eligible = True
        for cycle in CYCLES:
            values = loaded[cycle][file_type].frame[col].dropna().unique()
            if len(values) > 20:
                eligible = False
                break
            value_sets[cycle] = tuple(sorted(map(_stable_value, values)))
        if eligible and len(set(value_sets.values())) > 1:
            drift.append((col, "; ".join(f"{c}={list(value_sets[c])}" for c in CYCLES)))
    return drift[:100]


def _available_columns(loaded: dict[str, dict[str, XptData]], cycle: str, source_file: str) -> list[str]:
    if source_file == "ANY":
        columns: list[str] = []
        for file_type in TRACK_A_FILES:
            columns.extend(list(loaded[cycle][file_type].frame.columns))
        return sorted(set(columns))
    if source_file not in loaded.get(cycle, {}):
        return []
    return list(loaded[cycle][source_file].frame.columns)


def _resolve_column(columns: list[str], spec: FieldSpec) -> str | None:
    upper_to_actual = {col.upper(): col for col in columns}
    for candidate in spec.exact:
        actual = upper_to_actual.get(candidate.upper())
        if actual is not None:
            return actual
    for pattern in spec.patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        matches = sorted(col for col in columns if regex.match(col))
        if matches:
            return matches[0]
    return None


def _md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _stable_value(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        value = value.item()
    return repr(value)
