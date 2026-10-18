from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

GLI_INPUT_COLUMNS = ["SEQN", "age_years", "sex", "height_cm", "race_ethnicity"]


def export_gli_input(results_dir: Path) -> dict[str, int]:
    df = pd.read_parquet(results_dir / "merged_summary.parquet")
    before = int(len(df))
    gli_input = df.loc[:, GLI_INPUT_COLUMNS].dropna(subset=["age_years", "sex", "height_cm"]).copy()
    after = int(len(gli_input))
    gli_input.to_csv(results_dir / "_gli_input.csv", index=False)
    log = {
        "input_rows": before,
        "exported_rows": after,
        "dropped_missing_age_height_or_sex": before - after,
    }
    (results_dir / "gli_input_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    print(
        "GLI input export: "
        f"before={before}, after={after}, dropped_missing_demographics={before - after}"
    )
    return log


def run_gli_reference(root: Path) -> None:
    subprocess.run(["Rscript", str(root / "predictx" / "gli_reference.R")], cwd=root, check=True)
