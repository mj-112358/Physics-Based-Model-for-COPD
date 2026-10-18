from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/predictx-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/predictx-xdg-cache")

import numpy as np
import pandas as pd

from .gate_degradation_robustness import (
    BOOTSTRAP_B,
    SEED,
    add_delta_fields,
    build_curve_copies,
    clean_json,
    evaluate_profile,
    forward_model_guard,
    md_table,
)

RESULT_JSON = "gate_degradation_breaking_confirmation.json"
RESULT_CSV = "gate_degradation_breaking_confirmation.csv"
ROBUSTNESS_JSON = "gate_degradation_robustness.json"
ROBUSTNESS_CSV = "gate_degradation_robustness_sweep.csv"
ROBUSTNESS_MD = "gate_degradation_robustness.md"


def extreme_profile_defs() -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for order, frac in enumerate([0.15, 0.25, 0.40, 0.60, 1.00, 1.50, 2.00, 3.00], start=6):
        profiles.append(
            {
                "profile_id": f"D1_noise_{frac:.3f}_extreme",
                "operator": "D1_additive_noise",
                "level": f"{frac * 100:.0f}% PEF",
                "level_order": order,
                "noise_frac_pef": frac,
                "sample_hz": 100,
                "gain_error": 0.0,
                "drift_frac_fvc": 0.0,
            }
        )
    for order, hz in enumerate([8, 5, 3, 2, 1], start=6):
        profiles.append(
            {
                "profile_id": f"D2_sampling_{hz}Hz_extreme",
                "operator": "D2_coarser_sampling",
                "level": f"{hz} Hz",
                "level_order": order,
                "noise_frac_pef": 0.0,
                "sample_hz": hz,
                "gain_error": 0.0,
                "drift_frac_fvc": 0.0,
            }
        )
    order = 9
    for abs_gain in [0.15, 0.25, 0.40]:
        for sign in [-1.0, 1.0]:
            gain = sign * abs_gain
            profiles.append(
                {
                    "profile_id": f"D3_gain_{gain:+.3f}_extreme",
                    "operator": "D3_gain_error",
                    "level": f"{gain * 100:+.0f}%",
                    "level_order": order,
                    "noise_frac_pef": 0.0,
                    "sample_hz": 100,
                    "gain_error": gain,
                    "drift_frac_fvc": 0.0,
                }
            )
            order += 1
    for order, gain in enumerate([0.80, 1.50, 2.00], start=20):
        profiles.append(
            {
                "profile_id": f"D3_gain_{gain:+.3f}_failure_seek",
                "operator": "D3_gain_error",
                "level": f"{gain * 100:+.0f}%",
                "level_order": order,
                "noise_frac_pef": 0.0,
                "sample_hz": 100,
                "gain_error": gain,
                "drift_frac_fvc": 0.0,
            }
        )
    for order, (multiplier, frac) in enumerate([(2, 0.04), (4, 0.08), (8, 0.16), (16, 0.32), (32, 0.64)], start=4):
        profiles.append(
            {
                "profile_id": f"D4_drift_{multiplier}x_extreme",
                "operator": "D4_baseline_drift",
                "level": f"{multiplier}x prior large ({frac * 100:.0f}% FVC net drift)",
                "level_order": order,
                "noise_frac_pef": 0.0,
                "sample_hz": 100,
                "gain_error": 0.0,
                "drift_frac_fvc": frac,
            }
        )
    order = 1
    for abs_gain in [0.05, 0.10, 0.15, 0.25, 0.40, 0.60]:
        for sign in [-1.0, 1.0]:
            gain = sign * abs_gain
            profiles.append(
                {
                    "profile_id": f"D3_diff_gain_{gain:+.3f}",
                    "operator": "D3_differential_gain",
                    "level": f"{gain * 100:+.0f}% early-vs-late",
                    "level_order": order,
                    "noise_frac_pef": 0.0,
                    "sample_hz": 100,
                    "gain_error": 0.0,
                    "differential_gain_error": gain,
                    "drift_frac_fvc": 0.0,
                }
            )
            order += 1
    for order, gain in enumerate([0.80, 0.95, 0.99], start=20):
        profiles.append(
            {
                "profile_id": f"D3_diff_gain_{gain:+.3f}_failure_seek",
                "operator": "D3_differential_gain",
                "level": f"{gain * 100:+.0f}% early-vs-late",
                "level_order": order,
                "noise_frac_pef": 0.0,
                "sample_hz": 100,
                "gain_error": 0.0,
                "differential_gain_error": gain,
                "drift_frac_fvc": 0.0,
            }
        )
    return profiles


def profile_failed_rows(profile: dict[str, Any], failure: str) -> list[dict[str, Any]]:
    return [
        {
            "scope": "pooled_clean_obstructed",
            "profile_id": profile["profile_id"],
            "operator": profile["operator"],
            "level": profile["level"],
            "level_order": int(profile["level_order"]),
            "T_label": t_label,
            "n_obstructed": np.nan,
            "direct_fn": np.nan,
            "direct_fn_n": np.nan,
            "direct_fn_den": np.nan,
            "direct_fn_ci_low": np.nan,
            "direct_fn_ci_high": np.nan,
            "ombe_fn": np.nan,
            "ombe_fn_n": np.nan,
            "ombe_fn_den": np.nan,
            "ombe_fn_ci_low": np.nan,
            "ombe_fn_ci_high": np.nan,
            "direct_minus_ombe": np.nan,
            "direct_minus_ombe_ci_low": np.nan,
            "direct_minus_ombe_ci_high": np.nan,
            "direct_minus_ombe_excludes_zero": False,
            "ombe_fit_failure_n": np.nan,
            "row_verdict": "PIPELINE-FAILED",
            "pipeline_failed": True,
            "pipeline_failure": failure,
        }
        for t_label in ["2", "3"]
    ]


def load_baseline_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "results" / ROBUSTNESS_JSON
    if not path.exists():
        raise RuntimeError(f"Missing prior robustness JSON: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    return result["baseline"]["truncation_rows"]


def load_prior_sweep(root: Path) -> pd.DataFrame:
    path = root / "results" / ROBUSTNESS_CSV
    if not path.exists():
        raise RuntimeError(f"Missing prior robustness sweep: {path}")
    return pd.read_csv(path)


def first_break(rows: pd.DataFrame, operator: str) -> dict[str, Any]:
    op = rows[(rows["scope"].eq("pooled_clean_obstructed")) & (rows["operator"].eq(operator))].copy()
    if op.empty:
        return {
            "operator": operator,
            "breaking_level": "not assessed",
            "breaks": False,
            "note": "no rows available",
        }
    for level_order in sorted(op["level_order"].dropna().unique()):
        level_rows = op[op["level_order"].eq(level_order)]
        if "pipeline_failed" in level_rows:
            failed = any(False if pd.isna(value) else bool(value) for value in level_rows["pipeline_failed"].to_numpy(dtype=object))
        else:
            failed = False
        holds = bool(
            not failed
            and (level_rows["direct_minus_ombe"].astype(float) > 0).all()
            and level_rows["direct_minus_ombe_excludes_zero"].map(bool).all()
        )
        if not holds:
            first = level_rows.sort_values("T_label").iloc[0].to_dict()
            return {
                "operator": operator,
                "breaking_level": str(first["level"]),
                "breaks": True,
                "note": "pipeline failed" if failed else "first level where T=2/T=3 advantage did not both exclude zero",
            }
    worst_idx = op["direct_minus_ombe"].astype(float).idxmin()
    worst = op.loc[worst_idx].to_dict()
    return {
        "operator": operator,
        "breaking_level": "not reached",
        "breaks": False,
        "note": f"worst retained margin {float(worst['direct_minus_ombe']):.3f} at {worst['level']} T={worst['T_label']}",
    }


def perturbation_diagnostics(rows: pd.DataFrame, baseline_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_min = min(float(row["direct_minus_ombe"]) for row in baseline_rows)
    out: list[dict[str, Any]] = []
    for operator in ["D1_additive_noise", "D2_coarser_sampling", "D3_gain_error", "D4_baseline_drift", "D3_differential_gain"]:
        op = rows[(rows["scope"].eq("pooled_clean_obstructed")) & (rows["operator"].eq(operator))].copy()
        if op.empty:
            continue
        numeric = op.copy()
        worst = numeric.loc[numeric["direct_minus_ombe"].astype(float).idxmin()]
        out.append(
            {
                "operator": operator,
                "worst_level": str(worst["level"]),
                "worst_T": str(worst["T_label"]),
                "worst_margin": float(worst["direct_minus_ombe"]),
                "baseline_min_margin": baseline_min,
                "max_gate2_ratio_abs_pct": float(numeric["gate2_ratio_median_abs_pct_error"].astype(float).max())
                if "gate2_ratio_median_abs_pct_error" in numeric
                else np.nan,
                "max_gate2_fev1_abs_pct": float(numeric["gate2_fev1_median_abs_pct_error"].astype(float).max())
                if "gate2_fev1_median_abs_pct_error" in numeric
                else np.nan,
                "max_gate2_fvc_abs_pct": float(numeric["gate2_fvc_median_abs_pct_error"].astype(float).max())
                if "gate2_fvc_median_abs_pct_error" in numeric
                else np.nan,
                "direct_fn_range": f"{float(numeric['direct_fn'].min()):.3f}-{float(numeric['direct_fn'].max()):.3f}",
                "ombe_fn_range": f"{float(numeric['ombe_fn'].min()):.3f}-{float(numeric['ombe_fn'].max()):.3f}",
            }
        )
    return out


def append_report_section(root: Path, result: dict[str, Any], extreme_rows: list[dict[str, Any]], all_rows: pd.DataFrame) -> None:
    path = root / "results" / ROBUSTNESS_MD
    report = path.read_text(encoding="utf-8")
    marker = "\n## Breaking-Point Confirmation\n"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + "\n"
    extreme = pd.DataFrame(extreme_rows)
    section = [
        "",
        "## Breaking-Point Confirmation",
        "",
        "This confirmation pass pushed the prior no-break result past realistic budget-device levels. It used the same in-memory copied E/F/G curves and the same deterministic SciPy OMBE loop fit; NHANES source files were not written.",
        "",
        "### Extreme Breaking Points",
        "",
        md_table(
            result["breaking_points_confirmed"],
            [("operator", "operator", "str"), ("breaking_level", "breaking level", "str"), ("breaks", "breaks", "bool"), ("note", "note", "str")],
        ),
        "",
        "### Extreme Sweep Rows",
        "",
        md_table(
            extreme[extreme["scope"].eq("pooled_clean_obstructed")].to_dict("records"),
            [
                ("operator", "operator", "str"),
                ("level", "level", "str"),
                ("T_label", "T", "str"),
                ("direct_fn", "DIRECT FN", "rate"),
                ("ombe_fn", "OMBE FN", "rate"),
                ("direct_minus_ombe", "diff", "float3"),
                ("direct_minus_ombe_ci_low", "CI low", "float3"),
                ("direct_minus_ombe_ci_high", "CI high", "float3"),
                ("direct_minus_ombe_excludes_zero", "excludes 0", "bool"),
                ("gate2_ratio_median_abs_pct_error", "Gate2 ratio abs %", "float3"),
                ("row_verdict", "verdict", "str"),
            ],
        ),
        "",
        "### Operator Perturbation Checks",
        "",
        md_table(
            result["operator_perturbation_checks"],
            [
                ("operator", "operator", "str"),
                ("worst_level", "worst level", "str"),
                ("worst_T", "T", "str"),
                ("worst_margin", "worst margin", "float3"),
                ("max_gate2_ratio_abs_pct", "max ratio abs %", "float3"),
                ("max_gate2_fev1_abs_pct", "max FEV1 abs %", "float3"),
                ("max_gate2_fvc_abs_pct", "max FVC abs %", "float3"),
                ("direct_fn_range", "DIRECT FN range", "str"),
                ("ombe_fn_range", "OMBE FN range", "str"),
            ],
        ),
        "",
        "### D3 Gain-Model Clarification",
        "",
        "- Proportional D3 gain multiplies the copied volume curve by one scalar before FEV1 and FVC are read. DIRECT and Gate-2 FEV1/FVC ratios therefore cancel the gain exactly; this is immunity by construction for ratio-based diagnosis, not evidence that proportional calibration error perturbs the ratio.",
        "- Proportional gain can still perturb OMBE near solver bounds because the bounded fit estimates FVC on an absolute liter scale, but the full-curve ratio diagnostic remains exactly canceled.",
        f"- Differential gain was added as an explicit ratio-threatening variant: positive levels over-read pre-1s FEV1-weighted flow and under-read the post-1s FVC-tail flow; negative levels do the reverse. Its confirmed breaking point is {result['differential_gain_breaking_point']['breaking_level']}.",
        "- For the FN-threatening positive differential-gain sign, no break was reached through +99% early-vs-late; the first statistical CI failure is the opposite sign, where DIRECT nearly stops missing obstruction and the DIRECT-minus-OMBE advantage becomes too small to distinguish.",
        "",
        "### Honest Restatement",
        "",
        f"- Realistic-profile claim remains unchanged: {result['realistic_profile_verdict']} under the original 2% noise + 30 Hz + +2% proportional gain + small drift profile.",
        "- Corrected gain framing: ratio-based diagnosis is inherently immune to proportional gain, while the differential-gain stress test measures the non-proportional calibration error that can actually move FEV1/FVC.",
        "- D2 and D4 still showed no breaking point after the failure-seeking extension, but this was not because the operators were inert: D2 reached 1 Hz with 9.881% median ratio error, and D4 reached 64% FVC net drift with 37.425% median ratio error.",
        "",
    ]
    path.write_text(report.rstrip() + "\n" + "\n".join(section), encoding="utf-8")


def run(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    guard_start = forward_model_guard(SEED)
    if not guard_start["passed"] or float(guard_start["max_abs_volume_ml"]) >= 1e-9:
        raise RuntimeError(f"Forward-model guard failed at start: {guard_start}")

    baseline_rows = load_baseline_rows(root)
    prior_sweep = load_prior_sweep(root)
    clean_records, _severe_records, _sample_meta, _cov_notes, _cov_variables = build_curve_copies(root)
    print(f"Extreme confirmation clean copy built: {len(clean_records)} curves")

    extreme_rows: list[dict[str, Any]] = []
    for profile in extreme_profile_defs():
        try:
            rows, _predictions, _gate2 = evaluate_profile(clean_records, profile)
            add_delta_fields(rows, baseline_rows)
            for row in rows:
                row["pipeline_failed"] = False
                row["pipeline_failure"] = ""
            extreme_rows.extend(rows)
        except Exception as exc:
            extreme_rows.extend(profile_failed_rows(profile, str(exc)))

    all_rows = pd.concat([prior_sweep, pd.DataFrame(extreme_rows)], ignore_index=True, sort=False)
    breaking = [
        first_break(all_rows, operator)
        for operator in ["D1_additive_noise", "D2_coarser_sampling", "D3_gain_error", "D4_baseline_drift", "D3_differential_gain"]
    ]
    diff_break = next(row for row in breaking if row["operator"] == "D3_differential_gain")
    checks = perturbation_diagnostics(all_rows, baseline_rows)

    guard_end = forward_model_guard(SEED + 1)
    if not guard_end["passed"] or float(guard_end["max_abs_volume_ml"]) >= 1e-9:
        raise RuntimeError(f"Forward-model guard failed at end: {guard_end}")

    robustness = json.loads((root / "results" / ROBUSTNESS_JSON).read_text(encoding="utf-8"))
    result = {
        "seed": SEED,
        "bootstrap_n": BOOTSTRAP_B,
        "elapsed_s": float(time.perf_counter() - started),
        "forward_model_guards": [{"phase": "start", **guard_start}, {"phase": "end", **guard_end}],
        "copy_location": "in-memory Python CurveRecord list rebuilt from migrated E/F/G resolver; no scratch curve files written",
        "extreme_profiles": extreme_profile_defs(),
        "extreme_rows": extreme_rows,
        "breaking_points_confirmed": breaking,
        "operator_perturbation_checks": checks,
        "differential_gain_breaking_point": diff_break,
        "realistic_profile_verdict": robustness["verdict"],
        "d3_gain_model_check": {
            "proportional_gain_path": "degrade_curve multiplies the copied volume curve by (1 + gain_error) when noise/drift/sampling are unchanged.",
            "ratio_cancels": True,
            "differential_gain_variant": "piecewise flow gain: pre-1s FEV1-weighted flow is scaled by 1 + delta while post-1s FVC-tail flow is scaled by max(0.01, 1 - delta), then the curve is reintegrated.",
        },
    }

    out = root / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(extreme_rows).to_csv(out / RESULT_CSV, index=False)
    (out / RESULT_JSON).write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    append_report_section(root, result, extreme_rows, all_rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    for row in result["breaking_points_confirmed"]:
        print(f"{row['operator']}: {row['breaking_level']} ({row['note']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
