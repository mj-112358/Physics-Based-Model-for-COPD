"""Orchestrate the NHANES spirometry preparation and analysis workflow.

This module provides an explicit command-line interface for the core dataset
pipeline. Scientific experiment entry points are maintained separately in the
`predictx.gate_*` modules to preserve individual protocol configurations.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from .gli import export_gli_input, run_gli_reference
from .io import CYCLES, TRACK_A_FILES, XptValidationError, data_path, validate_xpt
from .merge import run_merge
from .obstruction import run_obstruction_analysis
from .raw import gate_track_b, parse_spxraw
from .schema import run_schema_discovery

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Define the public command-line interface for the core workflow."""
    parser = argparse.ArgumentParser(
        description="Prepare NHANES spirometry records and generate analysis outputs.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Working directory containing the data/ and results/ directories.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed execution logging.",
    )
    return parser


def run_pipeline(root: Path) -> int:
    """Execute the research data pipeline relative to *root*.

    Returns
    -------
    int
        Zero for a completed core run; one if essential input validation fails.
    """
    root = root.expanduser().resolve()
    data_dir = root / "data"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    input_errors = validate_summary_inputs(data_dir)
    if input_errors:
        for message in input_errors:
            LOGGER.error("Input validation: %s", message)
        LOGGER.error("Core pipeline not executed: required NHANES files unavailable")
        return 1

    LOGGER.info("Inspecting dataset schemas for cycles %s", ", ".join(CYCLES))
    _, column_map = run_schema_discovery(data_dir, results_dir)

    LOGGER.info("Merging participant-level spirometry and auxiliary measurements")
    run_merge(data_dir, results_dir, column_map)

    LOGGER.info("Preparing GLI reference inputs and spirometric measurements")
    export_gli_input(results_dir)
    run_gli_reference(root)
    obstruction_counts = run_obstruction_analysis(results_dir)

    raw_status = gate_track_b(data_dir, results_dir)
    if raw_status["status"] == "READY":
        LOGGER.info("Reconstructing raw expiratory measurements")
        frames = [
            parse_spxraw(data_path(data_dir, "SPXRAW", cycle), column_map, cycle)
            for cycle in CYCLES
        ]
        pd.concat(frames, ignore_index=True).to_parquet(
            results_dir / "raw_curves.parquet",
            index=False,
        )
        LOGGER.info("Raw-curve reconstruction complete")
    else:
        missing = ", ".join(raw_status["awaited_files"])
        LOGGER.info("Raw-curve reconstruction deferred; additional inputs: %s", missing)

    fixed = obstruction_counts["fixed_0_70"]
    lln = obstruction_counts["lln"]
    LOGGER.info(
        "Spirometric summary | fixed severe=%s | fixed moderate+severe=%s | "
        "LLN severe=%s | LLN moderate+severe=%s",
        fixed["n_lt_0_50"],
        fixed["n_0_50_to_0_60"] + fixed["n_lt_0_50"],
        lln["n_lt_0_50"],
        lln["n_0_50_to_0_60"] + lln["n_lt_0_50"],
    )
    LOGGER.info("Pipeline outputs written to %s", results_dir)
    return 0


def validate_summary_inputs(data_dir: Path) -> list[str]:
    """Collect integrity errors for required per-cycle XPORT files."""
    errors: list[str] = []
    for cycle in CYCLES:
        for file_type in TRACK_A_FILES:
            source = data_path(data_dir, file_type, cycle)
            try:
                validate_xpt(source)
            except XptValidationError as exc:
                errors.append(str(exc))
    return errors


def _validate_track_a(data_dir: Path) -> list[str]:
    """Backward-compatible alias for existing research scripts."""
    return validate_summary_inputs(data_dir)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line options and execute the preparation workflow."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    return run_pipeline(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
