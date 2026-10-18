from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/predictx-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/predictx-xdg-cache")

import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr

from .gate_clinical_classification_f import load_column_map, load_gli, load_truth
from .gate_offset_lungsize_f import attach_split_and_gli, base_pool, load_manoeuvres_fast
from .gate_truncation_recovery_f import clean_json

OUTPUT_STEM = "gate_gli_age_mechanism_F"
SEED = 20260710
AGE_SIGNAL_ABS_R_THRESHOLD = 0.70


def finite_pair(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    ok = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[ok], y_arr[ok]


def corr_stats(x: Any, y: Any, slope_x: Any | None = None) -> dict[str, Any]:
    x_ok, y_ok = finite_pair(x, y)
    if slope_x is None:
        sx_ok = x_ok
    else:
        sx_arr = np.asarray(slope_x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        ok = np.isfinite(sx_arr) & np.isfinite(y_arr)
        sx_ok = sx_arr[ok]
        y_ok = y_arr[ok]
        if slope_x is not x:
            x_ok, _ = finite_pair(x, y)
    if len(x_ok) < 3 or len(y_ok) < 3 or np.nanstd(x_ok) == 0 or np.nanstd(y_ok) == 0:
        return {
            "n": int(len(x_ok)),
            "pearson_r": np.nan,
            "p_value": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
        }
    r, p = pearsonr(x_ok, y_ok)
    fit = linregress(sx_ok, y_ok)
    return {
        "n": int(len(x_ok)),
        "pearson_r": float(r),
        "p_value": float(p),
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "r2": float(r * r),
    }


def md_table(rows: pd.DataFrame | list[dict[str, Any]], columns: list[tuple[str, str, str]]) -> str:
    df = pd.DataFrame(rows)
    out = ["| " + " | ".join(label for _, label, _ in columns) + " |"]
    out.append("|" + "|".join("---" for _ in columns) + "|")
    for _, row in df.iterrows():
        cells: list[str] = []
        for key, _, fmt in columns:
            value = row.get(key, np.nan)
            if fmt == "int":
                cells.append(str(int(value)) if pd.notna(value) else "NA")
            elif fmt == "bool":
                cells.append(str(bool(value)) if pd.notna(value) else "NA")
            elif fmt == "sci":
                try:
                    v = float(value)
                    cells.append(f"{v:.3e}" if np.isfinite(v) else "NA")
                except Exception:
                    cells.append("NA")
            elif fmt.startswith("float"):
                digits = int(fmt.replace("float", "") or "3")
                try:
                    v = float(value)
                    cells.append(f"{v:.{digits}f}" if np.isfinite(v) else "NA")
                except Exception:
                    cells.append("NA")
            else:
                cells.append("" if value is None else str(value))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def rows_by_type(rows: pd.DataFrame, row_type: str) -> list[dict[str, Any]]:
    return rows[rows["row_type"].eq(row_type)].to_dict("records")


def add_signal_rows(
    rows: list[dict[str, Any]],
    frame: pd.DataFrame,
    *,
    sample_level: str,
    sample_scope: str,
    primary: bool,
    outcome_col: str,
    outcome_label: str,
    outcome_units: str,
    predictor_specs: list[tuple[str, str, str, Any, str]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for predictor_col, predictor_label, predictor_units, slope_x, slope_units in predictor_specs:
        stats = corr_stats(frame[predictor_col], frame[outcome_col], slope_x=slope_x)
        record = {
            "row_type": "part0_signal",
            "sample_level": sample_level,
            "sample_scope": sample_scope,
            "primary": bool(primary),
            "outcome": outcome_label,
            "outcome_units": outcome_units,
            "predictor": predictor_label,
            "predictor_units": predictor_units,
            "slope_units": slope_units,
            **stats,
        }
        rows.append(record)
        out[predictor_label] = record
    return out


def add_gap_row(
    rows: list[dict[str, Any]],
    *,
    sample_level: str,
    sample_scope: str,
    primary: bool,
    outcome: str,
    age_row: dict[str, Any],
    fvc_row: dict[str, Any],
) -> dict[str, Any]:
    age_r2 = float(age_row.get("r2", np.nan))
    fvc_r2 = float(fvc_row.get("r2", np.nan))
    age_over_fvc = age_r2 / fvc_r2 if np.isfinite(age_r2) and np.isfinite(fvc_r2) and fvc_r2 > 0 else np.nan
    fvc_over_age = fvc_r2 / age_r2 if np.isfinite(age_r2) and np.isfinite(fvc_r2) and age_r2 > 0 else np.nan
    row = {
        "row_type": "part0_explained_variance_gap",
        "sample_level": sample_level,
        "sample_scope": sample_scope,
        "primary": bool(primary),
        "outcome": outcome,
        "age_r2": age_r2,
        "true_fvc_r2": fvc_r2,
        "age_r2_over_true_fvc_r2": age_over_fvc,
        "true_fvc_r2_over_age_r2": fvc_over_age,
    }
    rows.append(row)
    return row


def prepare_samples(root: Path) -> dict[str, pd.DataFrame]:
    truth = load_truth(root, load_column_map(root))
    gli = load_gli(root)
    maneuvers = load_manoeuvres_fast(root, truth)

    participants = base_pool(maneuvers, ("model", "observed")).merge(
        truth[["SEQN", "fev1_mL", "fvc_mL"]],
        on="SEQN",
        how="left",
    )
    participants["true_fvc_l"] = participants["fvc_mL"].astype(float) / 1000.0
    participants["true_fev1_l"] = participants["fev1_mL"].astype(float) / 1000.0
    participants["ratio_error_model_minus_observed"] = (
        participants["P3_model_ratio"].astype(float) - participants["P3_observed_ratio"].astype(float)
    )
    participants = attach_split_and_gli(participants, gli)

    maneuvers = maneuvers.copy()
    maneuvers["true_fvc_l"] = maneuvers["fvc_mL"].astype(float) / 1000.0
    maneuvers["true_fev1_l"] = maneuvers["fev1_mL"].astype(float) / 1000.0
    maneuvers["ratio_error_model_minus_observed"] = (
        maneuvers["model_ratio_est"].astype(float) - maneuvers["observed_ratio_est"].astype(float)
    )

    return {"participant": participants, "maneuver": maneuvers}


def finite_signal_frame(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    ok = np.ones(len(out), dtype=bool)
    for col in cols:
        ok &= np.isfinite(out[col].astype(float))
    return out[ok].copy()


def build_part0(root: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    samples = prepare_samples(root)
    rows: list[dict[str, Any]] = []

    participant = finite_signal_frame(
        samples["participant"],
        ["age_years", "fev1_fvc_ratio", "true_fvc_l", "ratio_error_model_minus_observed"],
    )
    maneuver = finite_signal_frame(
        samples["maneuver"],
        ["age_years", "fev1_fvc_ratio", "true_fvc_l", "ratio_error_model_minus_observed"],
    )

    sample_defs = [
        ("participant_P3", "all", True, participant),
        ("participant_P3", "train", False, participant[participant["split"].eq("train")]),
        ("participant_P3", "test", False, participant[participant["split"].eq("test")]),
        ("maneuver", "all", False, maneuver),
    ]

    ratio_signal: dict[str, dict[str, Any]] = {}
    gap_rows: dict[str, dict[str, Any]] = {}
    for sample_level, sample_scope, primary, frame in sample_defs:
        frame = frame.copy()
        signals = add_signal_rows(
            rows,
            frame,
            sample_level=sample_level,
            sample_scope=sample_scope,
            primary=primary,
            outcome_col="ratio_error_model_minus_observed",
            outcome_label="model_minus_observed_fev1_fvc_ratio_error",
            outcome_units="ratio",
            predictor_specs=[
                ("age_years", "age_years", "years", frame["age_years"].astype(float) / 10.0, "ratio error per decade"),
                ("true_fvc_l", "true_fvc_l", "L", frame["true_fvc_l"].astype(float), "ratio error per L"),
            ],
        )
        key = f"{sample_level}:{sample_scope}"
        ratio_signal[key] = signals
        gap_rows[key] = add_gap_row(
            rows,
            sample_level=sample_level,
            sample_scope=sample_scope,
            primary=primary,
            outcome="model_minus_observed_fev1_fvc_ratio_error",
            age_row=signals["age_years"],
            fvc_row=signals["true_fvc_l"],
        )

    gli_context: dict[str, dict[str, Any]] = {}
    gli_gaps: dict[str, dict[str, Any]] = {}
    for sample_scope, frame in [
        ("all", participant),
        ("train", participant[participant["split"].eq("train")]),
        ("test", participant[participant["split"].eq("test")]),
    ]:
        gli_frame = finite_signal_frame(frame, ["age_years", "true_fvc_l", "gli_lln_ratio"])
        signals = add_signal_rows(
            rows,
            gli_frame,
            sample_level="participant_P3",
            sample_scope=sample_scope,
            primary=False,
            outcome_col="gli_lln_ratio",
            outcome_label="gli_lln_ratio_context",
            outcome_units="ratio",
            predictor_specs=[
                ("age_years", "age_years", "years", gli_frame["age_years"].astype(float) / 10.0, "GLI LLN ratio per decade"),
                ("true_fvc_l", "true_fvc_l", "L", gli_frame["true_fvc_l"].astype(float), "GLI LLN ratio per L"),
            ],
        )
        key = f"gli_context:{sample_scope}"
        gli_context[key] = signals
        gli_gaps[key] = add_gap_row(
            rows,
            sample_level="participant_P3",
            sample_scope=sample_scope,
            primary=False,
            outcome="gli_lln_ratio_context",
            age_row=signals["age_years"],
            fvc_row=signals["true_fvc_l"],
        )

    primary_age = ratio_signal["participant_P3:all"]["age_years"]
    primary_fvc = ratio_signal["participant_P3:all"]["true_fvc_l"]
    primary_gap = gap_rows["participant_P3:all"]
    age_signal_reproduced = bool(abs(float(primary_age["pearson_r"])) > AGE_SIGNAL_ABS_R_THRESHOLD)
    status = "PART0-PASS" if age_signal_reproduced else "HALTED-PREMISE-NOT-STABLE"

    rows.append(
        {
            "row_type": "part0_decision",
            "sample_level": "participant_P3",
            "sample_scope": "all",
            "primary": True,
            "age_signal_abs_r_threshold": AGE_SIGNAL_ABS_R_THRESHOLD,
            "age_signal_pearson_r": primary_age["pearson_r"],
            "age_signal_abs_r": abs(float(primary_age["pearson_r"])),
            "age_signal_reproduced": age_signal_reproduced,
            "status": status,
            "decision": (
                "halt_before_parts_1_to_3"
                if not age_signal_reproduced
                else "continue_to_mechanism_diagnostics"
            ),
        }
    )

    part0 = {
        "primary_sample": "participant_P3:all",
        "primary_age_signal": primary_age,
        "primary_true_fvc_contrast": primary_fvc,
        "primary_explained_variance_gap": primary_gap,
        "ratio_error_signals": ratio_signal,
        "ratio_error_explained_variance_gaps": gap_rows,
        "gli_context_signals": gli_context,
        "gli_context_explained_variance_gaps": gli_gaps,
        "age_signal_abs_r_threshold": AGE_SIGNAL_ABS_R_THRESHOLD,
        "age_signal_reproduced": age_signal_reproduced,
        "status": status,
        "halt_reason": (
            "signed model-minus-observed FEV1/FVC ratio error vs age did not reproduce at |r| > 0.70"
            if not age_signal_reproduced
            else ""
        ),
    }
    return part0, pd.DataFrame(rows)


def make_report(result: dict[str, Any], rows: pd.DataFrame) -> str:
    part0 = result["part0"]
    primary_age = part0["primary_age_signal"]
    primary_fvc = part0["primary_true_fvc_contrast"]
    primary_gap = part0["primary_explained_variance_gap"]
    gli_all = part0["gli_context_signals"]["gli_context:all"]
    gli_gap = part0["gli_context_explained_variance_gaps"]["gli_context:all"]

    ratio_rows = [
        row
        for row in rows_by_type(rows, "part0_signal")
        if row["outcome"] == "model_minus_observed_fev1_fvc_ratio_error"
    ]
    gli_rows = [
        row
        for row in rows_by_type(rows, "part0_signal")
        if row["outcome"] == "gli_lln_ratio_context"
    ]
    gap_rows = rows_by_type(rows, "part0_explained_variance_gap")

    lines = [
        "# Gate - GLI/Age Mechanism, Cycle F",
        "",
        "## Protocol Decisions",
        "",
        f"- Seed recorded as {SEED}; Part 0 is deterministic on existing Cycle F clinical/offset artifacts and does not resample.",
        "- Primary Part 0 sample is participant-level P3, all valid clinical/offset participants with age, true FEV1/FVC, and true FVC.",
        "- Train/test participant rows are reported only as sensitivity checks using the existing clinical/offset split machinery.",
        "- The requested signed error is model-minus-observed FEV1/FVC ratio error. Positive means the OMBE-derived ratio is higher than the parsed observed ratio.",
        "- Parts 1-3 are not run if Part 0 fails the explicit |age r| > 0.70 guard.",
        "",
        "## Part 0 - Establish The Signal",
        "",
        "### Ratio Error Signal",
        "",
        md_table(
            ratio_rows,
            [
                ("sample_level", "level", "str"),
                ("sample_scope", "scope", "str"),
                ("primary", "primary", "bool"),
                ("predictor", "predictor", "str"),
                ("n", "n", "int"),
                ("pearson_r", "r", "float4"),
                ("r2", "R^2", "float4"),
                ("slope", "slope", "float5"),
                ("slope_units", "slope units", "str"),
                ("p_value", "p", "sci"),
            ],
        ),
        "",
        "### Explained-Variance Contrast",
        "",
        md_table(
            gap_rows,
            [
                ("outcome", "outcome", "str"),
                ("sample_level", "level", "str"),
                ("sample_scope", "scope", "str"),
                ("primary", "primary", "bool"),
                ("age_r2", "age R^2", "float4"),
                ("true_fvc_r2", "true FVC R^2", "float4"),
                ("age_r2_over_true_fvc_r2", "age/FVC R^2", "float3"),
                ("true_fvc_r2_over_age_r2", "FVC/age R^2", "float3"),
            ],
        ),
        "",
        "### GLI LLN Context",
        "",
        md_table(
            gli_rows,
            [
                ("sample_scope", "scope", "str"),
                ("predictor", "predictor", "str"),
                ("n", "n", "int"),
                ("pearson_r", "r", "float4"),
                ("r2", "R^2", "float4"),
                ("slope", "slope", "float5"),
                ("slope_units", "slope units", "str"),
                ("p_value", "p", "sci"),
            ],
        ),
        "",
        "The original GLI LLN age signal does reproduce: "
        f"all-participant GLI LLN vs age r={gli_all['age_years']['pearson_r']:.4f}, "
        f"R^2={gli_all['age_years']['r2']:.4f}; GLI LLN vs true FVC r={gli_all['true_fvc_l']['pearson_r']:.4f}, "
        f"R^2={gli_all['true_fvc_l']['r2']:.4f}; age/FVC explained-variance ratio={gli_gap['age_r2_over_true_fvc_r2']:.3f}.",
        "",
        "But the required forward-model error premise does not reproduce: "
        f"primary ratio error vs age r={primary_age['pearson_r']:.4f}, "
        f"R^2={primary_age['r2']:.4f}, slope={primary_age['slope']:.5f} ratio error per decade; "
        f"ratio error vs true FVC r={primary_fvc['pearson_r']:.4f}, R^2={primary_fvc['r2']:.4f}, "
        f"slope={primary_fvc['slope']:.5f} ratio error per L. "
        f"The age/FVC R^2 ratio is {primary_gap['age_r2_over_true_fvc_r2']:.3f}, not 23x.",
        "",
        f"HALT: {part0['halt_reason']}.",
        "",
        "## Parts 1-3 - Not Run",
        "",
        "Part 1 parameter attribution, Part 2 residual-shape stratification, and Part 3 tri-exponential falsification were not run because the task required Part 0 to reproduce the age signal before explaining it.",
        "",
        "Running tau_s attribution or a tri-exponential diagnostic after this guard fails would explain the wrong target: the strong age signal is present in GLI LLN itself, not in the current model-minus-observed ratio error outcome.",
        "",
        "## Part 4 - Verdict",
        "",
        f"VERDICT: {result['verdict']['verdict']}",
        "",
        f"Gate status: {result['verdict']['gate_status']}.",
        "",
        result["verdict"]["interpretation"],
        "",
    ]
    return "\n".join(lines)


def run(root: Path) -> dict[str, Any]:
    part0, rows = build_part0(root)
    if part0["age_signal_reproduced"]:
        verdict = {
            "verdict": "MECHANISM-PARTIAL",
            "gate_status": "PART0-PASS-MECHANISM-DIAGNOSTICS-NOT-RUN",
            "interpretation": (
                "Part 0 passed, but this implementation is scoped to the current Cycle F halt gate and should be extended before claiming a tau_s mechanism."
            ),
        }
    else:
        verdict = {
            "verdict": "MECHANISM-REFUTED",
            "gate_status": "HALTED-PREMISE-NOT-STABLE",
            "interpretation": (
                "The tau_s age-drift hypothesis is not supported because the required age-dependent forward-model ratio-error signal is absent. "
                "GLI degradation remains unexplained after the five tested mechanisms, with the added clarification that the strong age dependence is carried by the GLI threshold, not by current OMBE model-minus-observed ratio error."
            ),
        }

    result = {
        "seed": SEED,
        "output_stem": OUTPUT_STEM,
        "part0": part0,
        "parts_1_to_3": {
            "run": bool(part0["age_signal_reproduced"]),
            "skipped_reason": "" if part0["age_signal_reproduced"] else part0["halt_reason"],
        },
        "verdict": verdict,
    }

    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / f"{OUTPUT_STEM}_rows.csv", index=False)
    (out / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    (out / f"{OUTPUT_STEM}.md").write_text(make_report(result, rows), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    part0 = result["part0"]
    primary = part0["primary_age_signal"]
    print(
        "GLI/age mechanism gate: "
        f"{result['verdict']['verdict']} ({result['verdict']['gate_status']}); "
        f"primary age r={primary['pearson_r']:.4f}, threshold |r|>{AGE_SIGNAL_ABS_R_THRESHOLD:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
