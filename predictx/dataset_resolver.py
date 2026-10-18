from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import pyreadstat

from .io import validate_xpt

CYCLE_TO_YEAR = {
    "E": "2007-2008",
    "F": "2009-2010",
    "G": "2011-2012",
}

RAW_ROW_COUNTS = {
    "E": 34_857,
    "F": 36_421,
    "G": 37_661,
}


@dataclass(frozen=True)
class ResolvedDatasetPath:
    cycle: str
    file_type: str
    path: Path
    source: str
    note: str
    row_count: int | None = None

    def json_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["path"] = str(self.path)
        return out


def normalize_cycle(cycle: str) -> str:
    value = cycle.upper().strip()
    if value not in CYCLE_TO_YEAR:
        raise ValueError(f"Unsupported NHANES cycle {cycle!r}; expected one of {sorted(CYCLE_TO_YEAR)}")
    return value


def resolve_dataset_path(root: Path, file_type: str, cycle: str = "F") -> ResolvedDatasetPath:
    cycle = normalize_cycle(cycle)
    file_type = file_type.upper().strip()
    if file_type == "SPXRAW":
        return resolve_raw_path(root, cycle)
    return resolve_xpt_path(root, file_type, cycle)


def resolve_raw_path(root: Path, cycle: str = "F") -> ResolvedDatasetPath:
    cycle = normalize_cycle(cycle)
    tried: list[Path] = []
    new_root = root / "NHANES_SpiroSBI"
    if new_root.exists():
        year_dir = new_root / CYCLE_TO_YEAR[cycle]
        candidates = [
            year_dir / f"spxraw_{cycle.lower()}.sas7bdat",
            year_dir / f"SPXRAW_{cycle}.sas7bdat",
        ]
        for path in candidates:
            tried.append(path)
            if path.exists():
                row_count = validate_raw_sas(path, cycle)
                return ResolvedDatasetPath(
                    cycle=cycle,
                    file_type="SPXRAW",
                    path=path,
                    source="new-extracted",
                    note=f"using extracted SAS7BDAT from new nested root: {path}",
                    row_count=row_count,
                )

        zip_path = year_dir / f"SPXRAW_{cycle}.zip"
        tried.append(zip_path)
        if zip_path.exists():
            cache_path = _extract_raw_zip_once(new_root, zip_path, cycle)
            row_count = validate_raw_sas(cache_path, cycle)
            return ResolvedDatasetPath(
                cycle=cycle,
                file_type="SPXRAW",
                path=cache_path,
                source="new-zip-cache",
                note=f"extracted {zip_path} -> {cache_path}",
                row_count=row_count,
            )
        raise FileNotFoundError(
            f"Missing SPXRAW_{cycle} under {year_dir}; tried: {', '.join(str(p) for p in tried)}"
        )

    return _resolve_legacy_raw(root, cycle)


def resolve_xpt_path(root: Path, file_type: str, cycle: str = "F") -> ResolvedDatasetPath:
    cycle = normalize_cycle(cycle)
    file_type = file_type.upper().strip()
    new_root = root / "NHANES_SpiroSBI"
    if new_root.exists():
        path = new_root / CYCLE_TO_YEAR[cycle] / f"{file_type}_{cycle}.xpt"
        if not path.exists():
            raise FileNotFoundError(f"Missing {file_type}_{cycle}.xpt under new nested root: {path}")
        validate_xpt(path)
        return ResolvedDatasetPath(
            cycle=cycle,
            file_type=file_type,
            path=path,
            source="new-nested",
            note=f"using {file_type}_{cycle}.xpt from new nested root: {path}",
        )

    candidates = [
        root / "data" / f"{file_type}_{cycle}.xpt",
        root / "DATASET?" / f"{file_type}_{cycle}.xpt",
    ]
    for path in candidates:
        if path.exists():
            validate_xpt(path)
            return ResolvedDatasetPath(
                cycle=cycle,
                file_type=file_type,
                path=path,
                source="legacy",
                note=f"using legacy {file_type}_{cycle}.xpt path because NHANES_SpiroSBI is absent: {path}",
            )
    raise FileNotFoundError(
        f"Missing {file_type}_{cycle}.xpt and NHANES_SpiroSBI is absent; tried: {', '.join(str(p) for p in candidates)}"
    )


def validate_raw_sas(path: Path, cycle: str) -> int:
    cycle = normalize_cycle(cycle)
    if not path.exists():
        raise FileNotFoundError(f"Missing raw SAS file for cycle {cycle}: {path}")
    with path.open("rb") as fh:
        header = fh.read(1024)
    lowered = header.lstrip().lower()
    if lowered.startswith(b"<!doctype html") or b"<html" in lowered:
        raise RuntimeError(f"Raw file for cycle {cycle} is HTML, not SAS7BDAT: {path}")
    if b"SAS FILE" not in header:
        raise RuntimeError(f"Raw file for cycle {cycle} does not contain SAS7BDAT header magic: {path}")
    _df, meta = pyreadstat.read_sas7bdat(str(path), metadataonly=True)
    row_count = int(meta.number_rows)
    expected = RAW_ROW_COUNTS[cycle]
    if row_count != expected:
        raise RuntimeError(
            f"Raw row count mismatch for cycle {cycle}: expected {expected}, observed {row_count} at {path}"
        )
    return row_count


def _extract_raw_zip_once(new_root: Path, zip_path: Path, cycle: str) -> Path:
    cache_dir = new_root / "_extracted"
    cache_dir.mkdir(exist_ok=True)
    output_path = cache_dir / f"spxraw_{cycle.lower()}.sas7bdat"
    if output_path.exists():
        validate_raw_sas(output_path, cycle)
        return output_path
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            name
            for name in zf.namelist()
            if name.lower().endswith(".sas7bdat") and f"spxraw_{cycle.lower()}" in name.lower()
        ]
        if not members:
            members = [name for name in zf.namelist() if name.lower().endswith(".sas7bdat")]
        if not members:
            raise RuntimeError(f"No .sas7bdat member found in {zip_path}")
        with zf.open(members[0]) as src, output_path.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    return output_path


def _resolve_legacy_raw(root: Path, cycle: str) -> ResolvedDatasetPath:
    tried: list[Path] = []
    candidates = [
        root / "data" / f"spxraw_{cycle.lower()}.sas7bdat",
        root / "data" / f"SPXRAW_{cycle}.sas7bdat",
        root / "DATASET?" / f"spxraw_{cycle.lower()}.sas7bdat",
        root / "DATASET?" / f"SPXRAW_{cycle}.sas7bdat",
    ]
    for path in candidates:
        tried.append(path)
        if path.exists():
            row_count = validate_raw_sas(path, cycle)
            return ResolvedDatasetPath(
                cycle=cycle,
                file_type="SPXRAW",
                path=path,
                source="legacy-extracted",
                note=f"using legacy SAS7BDAT because NHANES_SpiroSBI is absent: {path}",
                row_count=row_count,
            )

    zip_candidates = [
        root / "data" / f"SPXRAW_{cycle}.zip",
        root / "DATASET?" / f"SPXRAW_{cycle}.zip",
    ]
    for zip_path in zip_candidates:
        tried.append(zip_path)
        if not zip_path.exists():
            continue
        output_path = root / "data" / f"spxraw_{cycle.lower()}.sas7bdat"
        output_path.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            members = [name for name in zf.namelist() if name.lower().endswith(".sas7bdat")]
            if not members:
                raise RuntimeError(f"No .sas7bdat member found in {zip_path}")
            with zf.open(members[0]) as src, output_path.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
        row_count = validate_raw_sas(output_path, cycle)
        return ResolvedDatasetPath(
            cycle=cycle,
            file_type="SPXRAW",
            path=output_path,
            source="legacy-zip-cache",
            note=f"extracted legacy {zip_path} -> {output_path}",
            row_count=row_count,
        )

    raise FileNotFoundError(
        f"Missing real SPXRAW_{cycle} SAS7BDAT or ZIP; tried: {', '.join(str(p) for p in tried)}"
    )


def resolver_check(root: Path) -> list[dict[str, object]]:
    rows = []
    for cycle in CYCLE_TO_YEAR:
        raw = resolve_raw_path(root, cycle)
        spx = resolve_xpt_path(root, "SPX", cycle)
        rows.append(
            {
                "cycle": cycle,
                "year": CYCLE_TO_YEAR[cycle],
                "raw_path": str(raw.path),
                "raw_source": raw.source,
                "raw_note": raw.note,
                "raw_row_count": raw.row_count,
                "spx_path": str(spx.path),
                "spx_source": spx.source,
                "spx_note": spx.note,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    rows = resolver_check(Path(args.root).resolve())
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
