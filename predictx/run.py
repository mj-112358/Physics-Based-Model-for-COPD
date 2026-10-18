from __future__ import annotations

from pathlib import Path

from .gli import export_gli_input, run_gli_reference
from .io import CYCLES, TRACK_A_FILES, XptValidationError, data_path, validate_xpt
from .merge import run_merge
from .obstruction import run_obstruction_analysis
from .raw import gate_track_b, parse_spxraw
from .schema import run_schema_discovery


def main() -> int:
    root = Path.cwd()
    data_dir = root / "data"
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    track_a_problems = _validate_track_a(data_dir)
    if track_a_problems:
        for problem in track_a_problems:
            print(problem)
        print("Track A = GATED")
        return 1

    _, column_map = run_schema_discovery(data_dir, results_dir)
    run_merge(data_dir, results_dir, column_map)
    export_gli_input(results_dir)
    run_gli_reference(root)
    obstruction_counts = run_obstruction_analysis(results_dir)

    track_b = gate_track_b(data_dir, results_dir)
    if track_b["status"] == "READY":
        raw_frames = [
            parse_spxraw(data_path(data_dir, "SPXRAW", cycle), column_map, cycle)
            for cycle in CYCLES
        ]
        raw = __import__("pandas").concat(raw_frames, ignore_index=True)
        raw.to_parquet(results_dir / "raw_curves.parquet", index=False)
        track_b_summary = "Track B = RAN"
    else:
        track_b_summary = "Track B = GATED awaiting " + ", ".join(track_b["awaited_files"])

    print("Final gate summary:")
    print("Track A = RAN")
    print(track_b_summary)
    fixed = obstruction_counts["fixed_0_70"]
    lln = obstruction_counts["lln"]
    print(
        "Headline: "
        f"fixed severe = {fixed['n_lt_0_50']}; "
        f"fixed moderate+severe = {fixed['n_0_50_to_0_60'] + fixed['n_lt_0_50']}; "
        f"LLN severe = {lln['n_lt_0_50']}; "
        f"LLN moderate+severe = {lln['n_0_50_to_0_60'] + lln['n_lt_0_50']}."
    )
    return 0


def _validate_track_a(data_dir: Path) -> list[str]:
    problems: list[str] = []
    for cycle in CYCLES:
        for file_type in TRACK_A_FILES:
            path = data_path(data_dir, file_type, cycle)
            try:
                validate_xpt(path)
            except XptValidationError as exc:
                problems.append(str(exc))
    return problems


if __name__ == "__main__":
    raise SystemExit(main())
