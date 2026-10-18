from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.io.sas.sas_xport import XportReader

CYCLES = ("E", "F", "G")
TRACK_A_FILES = ("DEMO", "BMX", "SPX")
TRACK_B_FILES = ("SPXRAW",)
XPORT_HEADER = b"HEADER RECORD*******LIBRARY"
MIN_XPT_BYTES = 100_000


class XptValidationError(RuntimeError):
    """Raised when an expected XPORT input is absent or corrupt."""


@dataclass(frozen=True)
class XptData:
    path: Path
    frame: pd.DataFrame
    labels: dict[str, str]


def data_path(data_dir: Path, file_type: str, cycle: str) -> Path:
    return data_dir / f"{file_type}_{cycle}.xpt"


def validate_xpt(path: Path, *, min_bytes: int = MIN_XPT_BYTES) -> None:
    if not path.exists():
        raise XptValidationError(f"missing: {path}")
    size = path.stat().st_size
    if size < min_bytes:
        raise XptValidationError(f"too small ({size} bytes): {path}")
    with path.open("rb") as fh:
        header = fh.read(len(XPORT_HEADER))
    if header != XPORT_HEADER:
        raise XptValidationError(f"invalid XPORT header: {path}")


def read_xpt(path: Path) -> XptData:
    validate_xpt(path)
    reader = XportReader(path)
    try:
        labels = {
            _decode_field(field["name"]): _decode_field(field.get("label", b""))
            for field in reader.fields
        }
    finally:
        reader.close()
    frame = pd.read_sas(path, format="xport")
    return XptData(path=path, frame=frame, labels=labels)


def example_values(series: pd.Series, limit: int = 3) -> list[Any]:
    values = []
    for value in series.dropna().head(limit).tolist():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        values.append(value)
    return values


def _decode_field(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()
