from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import CYCLES, XptValidationError, data_path, read_xpt, validate_xpt

RAW_REQUIRED_FIELDS = (
    "participant_id",
    "raw_curve",
    "n_points",
    "btps_factor",
    "test_type",
    "maneuver_num",
    "acceptable",
)
RAW_METADATA_FIELDS = ("position", "plateau", "raw_effort_grade")
SAMPLE_HZ = 100


def interval_end_times(n_points: int, hz: int = SAMPLE_HZ) -> np.ndarray:
    """Return timestamps for interval-valued raw volume increments."""
    if n_points < 0:
        raise ValueError("n_points must be nonnegative")
    if hz <= 0:
        raise ValueError("hz must be positive")
    return (np.arange(n_points, dtype=float) + 1.0) / float(hz)


def gate_track_b(data_dir: Path, results_dir: Path) -> dict[str, object]:
    awaited: list[str] = []
    problems: list[str] = []
    for cycle in CYCLES:
        path = data_path(data_dir, "SPXRAW", cycle)
        try:
            validate_xpt(path)
        except XptValidationError as exc:
            awaited.append(str(path))
            problems.append(str(exc))

    if awaited:
        skipped = {
            "track": "raw-curve",
            "status": "GATED",
            "reason": "SPXRAW inputs are absent or invalid; raw-curve parsing skipped without fabricating data.",
            "awaited_files": awaited,
            "problems": problems,
        }
        (results_dir / "trackB_skipped.json").write_text(
            json.dumps(skipped, indent=2) + "\n",
            encoding="utf-8",
        )
        return skipped

    return {"track": "raw-curve", "status": "READY", "awaited_files": []}


def parse_spxraw(path: Path, column_map: dict, cycle: str | None = None) -> pd.DataFrame:
    xpt = read_xpt(path)
    if cycle is None:
        cycle = _infer_cycle(path)
    return parse_spxraw_frame(xpt.frame, column_map, cycle)


def parse_spxraw_frame(frame: pd.DataFrame, column_map: dict, cycle: str) -> pd.DataFrame:
    columns = {field: _raw_field(column_map, field, cycle) for field in RAW_REQUIRED_FIELDS}
    metadata = {field: _raw_field(column_map, field, cycle, required=False) for field in RAW_METADATA_FIELDS}
    records: list[pd.DataFrame] = []

    for row_number, row in frame.iterrows():
        raw_value = row[columns["raw_curve"]]
        if pd.isna(raw_value):
            continue
        deltas = _parse_delta_string(raw_value)
        n_points = int(row[columns["n_points"]])
        if len(deltas) != n_points:
            raise ValueError(
                f"SPXRAW point count mismatch at row {row_number}: SPXPTS={n_points}, parsed={len(deltas)}"
            )
        btps = float(row[columns["btps_factor"]])
        flow_interval_ml = deltas * btps
        volume_ml = np.cumsum(deltas) * btps
        out = pd.DataFrame(
            {
                "SEQN": row[columns["participant_id"]],
                "cycle": cycle,
                "maneuver_num": row[columns["maneuver_num"]],
                "test_type": row[columns["test_type"]],
                "acceptable": row[columns["acceptable"]],
                "point_index": np.arange(n_points, dtype=int),
                "time_s": interval_end_times(n_points),
                "flow_interval_mL": flow_interval_ml,
                "flow_mL_s": flow_interval_ml * 100.0,
                "volume_mL": volume_ml,
                "btps_factor": btps,
            }
        )
        for logical, actual in metadata.items():
            if actual is not None:
                out[logical] = row[actual]
        records.append(out)

    if not records:
        return pd.DataFrame(
            columns=[
                "SEQN",
                "cycle",
                "maneuver_num",
                "test_type",
                "acceptable",
                "point_index",
                "time_s",
                "flow_interval_mL",
                "flow_mL_s",
                "volume_mL",
                "btps_factor",
                *RAW_METADATA_FIELDS,
            ]
        )
    return pd.concat(records, ignore_index=True)


def _parse_delta_string(value: object) -> np.ndarray:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    return np.array([int(part) for part in parts], dtype=float)


def _raw_field(column_map: dict, logical: str, cycle: str, *, required: bool = True) -> str | None:
    actual = column_map["fields"].get(logical, {}).get(cycle)
    if actual is None and required:
        raise KeyError(f"Raw field {logical!r} is unresolved for cycle {cycle}")
    return actual


def _infer_cycle(path: Path) -> str:
    stem = path.stem.upper()
    for cycle in CYCLES:
        if stem.endswith(f"_{cycle}"):
            return cycle
    raise ValueError(f"Could not infer NHANES cycle from {path}")
