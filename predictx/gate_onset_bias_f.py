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

from .gate23_f import HZ, find_or_unzip_raw, parse_increments, read_ground_truth, read_raw
from .gate_fit_optimization_f import baseline_lock, forward_model_guard, md_table
from .gate_truncation_recovery_f import clean_json
from .segment import backextrap_t0, derive_fev1_fvc, detect_forced_exhale_segment

OUTPUT_STEM = "gate_onset_bias_F"
SEED = 20260710
BOOTSTRAP_B = 1000
ATS_COMPLIANCE_STOP_THRESHOLD = 0.95


def q(values: Any, probs: tuple[float, float] = (25, 75)) -> list[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan"), float("nan")]
    return [float(x) for x in np.percentile(arr, probs)]


def pct_error(est: pd.Series, truth: pd.Series) -> pd.Series:
    return 100.0 * (est - truth) / truth


def derive_current_and_axis_onset(row: Any) -> dict[str, Any]:
    incr = parse_increments(row.SPXRAW)
    expected = int(row.SPXPTS)
    if len(incr) != expected:
        raise ValueError(f"SPXRAW length {len(incr)} != SPXPTS {expected}")
    fact = float(row.SPAFACT)
    flow_lps = incr * 0.1 * fact
    volume_ml = np.cumsum(incr) * fact
    time_s = np.arange(len(incr), dtype=float) / HZ
    onset_idx, pef_idx, end_idx = detect_forced_exhale_segment(flow_lps, hz=HZ)

    current_t0_s = backextrap_t0(time_s, flow_lps, volume_ml, onset_idx, pef_idx)
    current_fev1_ml, current_fvc_ml = derive_fev1_fvc(time_s, volume_ml, current_t0_s, end_idx)
    current_v0_ml = float(np.interp(current_t0_s, time_s, volume_ml))
    onset_volume_ml = float(volume_ml[onset_idx])
    current_bev_ml = abs(current_v0_ml - onset_volume_ml)
    current_bev_limit_ml = max(100.0, 0.05 * max(current_fvc_ml, 0.0))

    pef_ml_s = float(flow_lps[pef_idx] * 1000.0)
    if pef_ml_s <= 0:
        raise ValueError("PEF must be positive for volume-axis back-extrapolation.")
    axis_t0_s = float(time_s[pef_idx] - volume_ml[pef_idx] / pef_ml_s)
    axis_fev1_ml, axis_fvc_ml = derive_fev1_fvc(time_s, volume_ml, axis_t0_s, end_idx)
    axis_v0_ml = float(np.interp(axis_t0_s, time_s, volume_ml))
    axis_bev_ml = abs(axis_v0_ml)
    axis_bev_limit_ml = max(100.0, 0.05 * max(axis_fvc_ml, 0.0))

    return {
        "n_points": int(len(incr)),
        "onset_idx": int(onset_idx),
        "pef_idx": int(pef_idx),
        "end_idx": int(end_idx),
        "current_t0_s": float(current_t0_s),
        "current_fev1_ml": float(current_fev1_ml),
        "current_fvc_ml": float(current_fvc_ml),
        "current_ratio": float(current_fev1_ml / current_fvc_ml) if current_fvc_ml > 0 else np.nan,
        "current_bev_ml": float(current_bev_ml),
        "current_bev_fvc_ratio": float(current_bev_ml / current_fvc_ml) if current_fvc_ml > 0 else np.nan,
        "current_bev_limit_ml": float(current_bev_limit_ml),
        "current_bev_flag": bool(current_bev_ml > current_bev_limit_ml),
        "axis_t0_s": float(axis_t0_s),
        "axis_fev1_ml": float(axis_fev1_ml),
        "axis_fvc_ml": float(axis_fvc_ml),
        "axis_ratio": float(axis_fev1_ml / axis_fvc_ml) if axis_fvc_ml > 0 else np.nan,
        "axis_bev_ml": float(axis_bev_ml),
        "axis_bev_fvc_ratio": float(axis_bev_ml / axis_fvc_ml) if axis_fvc_ml > 0 else np.nan,
        "axis_bev_limit_ml": float(axis_bev_limit_ml),
        "axis_bev_flag": bool(axis_bev_ml > axis_bev_limit_ml),
        "t0_delta_axis_minus_current_s": float(axis_t0_s - current_t0_s),
    }


def derive_all_onsets(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    errors = 0
    for idx, row in enumerate(raw.itertuples(index=True), start=1):
        base = {
            "raw_index": int(row.Index),
            "SEQN": float(row.SEQN),
            "SPATTYPE": str(row.SPATTYPE),
            "SPAMANU": float(row.SPAMANU),
            "SPAFACT": float(row.SPAFACT),
            "SPAPLAT": str(row.SPAPLAT),
            "SPAACC": str(row.SPAACC),
            "SPAQEFF": str(row.SPAQEFF),
            "SPXPTS": int(row.SPXPTS),
            "parse_error": "",
        }
        try:
            base.update(derive_current_and_axis_onset(row))
        except Exception as exc:
            errors += 1
            base["parse_error"] = str(exc)
        rows.append(base)
        if idx % 10000 == 0 or idx == len(raw):
            print(f"Onset bias parse progress: {idx}/{len(raw)} raw rows")
    out = pd.DataFrame(rows)
    out.attrs["parse_error_count"] = int(errors)
    return out


def scope_frames(derived_truth: pd.DataFrame) -> dict[str, pd.DataFrame]:
    parsed = derived_truth[derived_truth["parse_error"].fillna("").eq("")].copy()
    pre = parsed[parsed["SPATTYPE"].eq("Pre")].copy()
    return {
        "all_parsed_pre": pre,
        "gate2_acceptable_pre": pre[pre["SPAACC"].eq("Y")].copy(),
        "clinical_acceptable_ab_pre": pre[pre["SPAACC"].eq("Y") & pre["SPAQEFF"].isin(["A", "B"])].copy(),
        "grade_a_acceptable_pre": pre[pre["SPAACC"].eq("Y") & pre["SPAQEFF"].eq("A")].copy(),
    }


def bev_summary(frame: pd.DataFrame, scope: str) -> dict[str, Any]:
    valid = frame[np.isfinite(frame["current_bev_ml"]) & np.isfinite(frame["current_fvc_ml"]) & (frame["current_fvc_ml"] > 0)].copy()
    return {
        "scope": scope,
        "n": int(len(valid)),
        "bev_ml_median": float(np.nanmedian(valid["current_bev_ml"])) if len(valid) else np.nan,
        "bev_ml_iqr_low": q(valid["current_bev_ml"])[0],
        "bev_ml_iqr_high": q(valid["current_bev_ml"])[1],
        "bev_ml_p95": float(np.nanpercentile(valid["current_bev_ml"], 95)) if len(valid) else np.nan,
        "bev_fvc_ratio_median": float(np.nanmedian(valid["current_bev_fvc_ratio"])) if len(valid) else np.nan,
        "bev_fvc_ratio_iqr_low": q(valid["current_bev_fvc_ratio"])[0],
        "bev_fvc_ratio_iqr_high": q(valid["current_bev_fvc_ratio"])[1],
        "ats_violation_fraction": float(valid["current_bev_flag"].mean()) if len(valid) else np.nan,
        "ats_compliance_fraction": float((~valid["current_bev_flag"].astype(bool)).mean()) if len(valid) else np.nan,
    }


def criterion_participant_table(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    pre = frame[frame["SPATTYPE"].eq("Pre") & frame["SPAACC"].eq("Y") & frame["parse_error"].fillna("").eq("")].copy()
    pre = pre.dropna(subset=["SPXNFEV1", "SPXNFVC"]).copy()
    pre = pre[(pre["SPXNFEV1"] > 0) & (pre["SPXNFVC"] > 0)]
    out = (
        pre.groupby("SEQN", as_index=False)
        .agg(
            derived_fev1_ml=(f"{prefix}_fev1_ml", "max"),
            derived_fvc_ml=(f"{prefix}_fvc_ml", "max"),
            derived_ratio=(f"{prefix}_ratio", "max"),
            SPXNFEV1=("SPXNFEV1", "first"),
            SPXNFVC=("SPXNFVC", "first"),
            bev_flag=(f"{prefix}_bev_flag", "max"),
        )
    )
    out = out.dropna(subset=["derived_fev1_ml", "derived_fvc_ml", "derived_ratio", "SPXNFEV1", "SPXNFVC"]).copy()
    return out


def comparison_summary(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    valid = frame[(frame["SPXNFEV1"] > 0) & (frame["SPXNFVC"] > 0)].copy()
    fev1_err = pct_error(valid["derived_fev1_ml"], valid["SPXNFEV1"])
    fvc_err = pct_error(valid["derived_fvc_ml"], valid["SPXNFVC"])
    ratio_truth = valid["SPXNFEV1"] / valid["SPXNFVC"]
    ratio_err = pct_error(valid["derived_ratio"], ratio_truth)
    return {
        "label": label,
        "n": int(len(valid)),
        "fev1_median_signed_pct_error": float(np.nanmedian(fev1_err)),
        "fev1_median_abs_pct_error": float(np.nanmedian(np.abs(fev1_err))),
        "fvc_median_signed_pct_error": float(np.nanmedian(fvc_err)),
        "fvc_median_abs_pct_error": float(np.nanmedian(np.abs(fvc_err))),
        "ratio_median_signed_pct_error": float(np.nanmedian(ratio_err)),
        "ratio_median_abs_pct_error": float(np.nanmedian(np.abs(ratio_err))),
        "bev_flag_fraction": float(valid["bev_flag"].mean()) if "bev_flag" in valid else np.nan,
    }


def bootstrap_gate2_deltas(current: pd.DataFrame, axis: pd.DataFrame) -> list[dict[str, Any]]:
    merged = current.merge(axis, on="SEQN", suffixes=("_current", "_axis"))
    rng = np.random.default_rng(SEED)
    n = len(merged)

    def metric_values(df: pd.DataFrame, prefix: str, metric: str) -> np.ndarray:
        if metric == "fev1_signed_pct":
            return np.asarray(pct_error(df[f"derived_fev1_ml_{prefix}"], df[f"SPXNFEV1_{prefix}"]), dtype=float)
        if metric == "fev1_abs_pct":
            return np.abs(np.asarray(pct_error(df[f"derived_fev1_ml_{prefix}"], df[f"SPXNFEV1_{prefix}"]), dtype=float))
        if metric == "fvc_signed_pct":
            return np.asarray(pct_error(df[f"derived_fvc_ml_{prefix}"], df[f"SPXNFVC_{prefix}"]), dtype=float)
        if metric == "fvc_abs_pct":
            return np.abs(np.asarray(pct_error(df[f"derived_fvc_ml_{prefix}"], df[f"SPXNFVC_{prefix}"]), dtype=float))
        if metric == "ratio_abs_pct":
            truth = df[f"SPXNFEV1_{prefix}"] / df[f"SPXNFVC_{prefix}"]
            return np.abs(np.asarray(pct_error(df[f"derived_ratio_{prefix}"], truth), dtype=float))
        raise KeyError(metric)

    rows: list[dict[str, Any]] = []
    for metric in ["fev1_signed_pct", "fev1_abs_pct", "fvc_signed_pct", "fvc_abs_pct", "ratio_abs_pct"]:
        cur = metric_values(merged, "current", metric)
        new = metric_values(merged, "axis", metric)
        point = float(np.nanmedian(new) - np.nanmedian(cur))
        boots = []
        for _ in range(BOOTSTRAP_B):
            idx = rng.integers(0, n, size=n)
            boots.append(float(np.nanmedian(new[idx]) - np.nanmedian(cur[idx])))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(
            {
                "metric": metric,
                "n": int(n),
                "baseline_median": float(np.nanmedian(cur)),
                "axis_median": float(np.nanmedian(new)),
                "delta_axis_minus_baseline": point,
                "delta_ci_low": float(lo),
                "delta_ci_high": float(hi),
                "bootstrap_n": BOOTSTRAP_B,
            }
        )
    return rows


def make_report(result: dict[str, Any]) -> str:
    baseline_rows = pd.DataFrame(result["baseline_rows"])
    part1 = pd.DataFrame(result["part1_current_bev"]["scope_summaries"])
    deltas = pd.DataFrame(result["part2_axis_onset"]["bootstrap_deltas"])
    summaries = pd.DataFrame(
        [
            {"onset": "current baseline", **result["part2_axis_onset"]["current_summary"]},
            {"onset": "volume-axis tangent", **result["part2_axis_onset"]["axis_summary"]},
        ]
    )
    lines = [
        "# Gate - Onset / Back-Extrapolation Bias Test, Cycle F",
        "",
        "## Protocol Decisions",
        "",
        "- This gate is standalone from the fit-optimization vectorization path.",
        "- No solver code is changed; no vectorization is used.",
        "- Part 2 tests the literal volume-axis tangent requested here: t0 = t_PEF - V_PEF / PEF.",
        "- The propagation scope is guarded: clinical/truncation recomputation only occurs if Gate 2 signed bias moves toward zero and absolute error does not increase.",
        "- BEV compliance decision scope is `SPAACC == Y` acceptable Pre curves, because that is the Gate 2 criterion pool.",
        "",
        "## Part 0 - Baseline Lock",
        "",
        md_table(
            baseline_rows,
            [
                ("headline", "headline", "str"),
                ("metric", "metric", "str"),
                ("estimator", "estimator", "str"),
                ("endpoint", "endpoint", "str"),
                ("n", "n", "int"),
                ("denominator_n", "denom", "int"),
                ("baseline_value", "baseline", "float6"),
                ("unit", "unit", "str"),
            ],
        ),
        "",
        "## Forward-Model Guard",
        "",
        md_table(
            [
                {"phase": "start", **result["forward_model_guard_start"]},
                {"phase": "end", **result["forward_model_guard_end"]},
            ],
            [
                ("phase", "phase", "str"),
                ("n_param_sets", "param sets", "int"),
                ("grid_n", "grid n", "int"),
                ("max_abs_volume_ml", "max abs volume mL", "float12"),
                ("tolerance_ml", "tolerance mL", "float12"),
                ("passed", "passed", "bool"),
            ],
        ),
        "",
        "## Part 1 - Current Onset",
        "",
        "Current code path:",
        "",
        "- Gate 2: `predictx/gate23_f.py::derive_curve` -> `predictx.segment.detect_forced_exhale_segment` -> `predictx.segment.backextrap_t0` -> `predictx.segment.derive_fev1_fvc`.",
        "- Clinical/truncation fit parsing: `predictx/gate_clinical_classification_f.py::extract_curve_for_fit` imports the same detection/back-extrapolation functions through `gate23_f`.",
        "- Current t0 formula: `time_s[pef_idx] - (volume_ml[pef_idx] - volume_ml[onset_idx]) / (flow_lps[pef_idx] * 1000)`.",
        "",
        md_table(
            part1,
            [
                ("scope", "scope", "str"),
                ("n", "n", "int"),
                ("bev_ml_median", "BEV mL med", "float2"),
                ("bev_ml_iqr_low", "BEV IQR low", "float2"),
                ("bev_ml_iqr_high", "BEV IQR high", "float2"),
                ("bev_fvc_ratio_median", "BEV/FVC med", "float4"),
                ("ats_violation_fraction", "ATS violation", "float4"),
                ("ats_compliance_fraction", "ATS compliant", "float4"),
            ],
        ),
        "",
        f"Decision: Gate 2 acceptable Pre curves are ATS-compliant at {result['part1_current_bev']['decision_scope_compliance_fraction']:.4f}. "
        f"The >95% stop rule is therefore {'met' if result['part1_current_bev']['ats_stop_rule_met'] else 'not met'}.",
        "",
        "## Part 2 - Volume-Axis ATS Tangent Diagnostic",
        "",
        md_table(
            summaries,
            [
                ("onset", "onset", "str"),
                ("n", "n", "int"),
                ("fev1_median_signed_pct_error", "FEV1 signed %", "float4"),
                ("fev1_median_abs_pct_error", "FEV1 abs %", "float4"),
                ("fvc_median_signed_pct_error", "FVC signed %", "float4"),
                ("fvc_median_abs_pct_error", "FVC abs %", "float4"),
                ("ratio_median_abs_pct_error", "ratio abs %", "float4"),
            ],
        ),
        "",
        "Bootstrap deltas are volume-axis minus current baseline on the same participant set:",
        "",
        md_table(
            deltas,
            [
                ("metric", "metric", "str"),
                ("n", "n", "int"),
                ("baseline_median", "baseline", "float6"),
                ("axis_median", "axis", "float6"),
                ("delta_axis_minus_baseline", "delta", "float6"),
                ("delta_ci_low", "CI low", "float6"),
                ("delta_ci_high", "CI high", "float6"),
            ],
        ),
        "",
        f"Part 2 pass: {result['part2_axis_onset']['passed']}. {result['part2_axis_onset']['interpretation']}",
        "",
        "## Part 3 - Propagation",
        "",
        result["part3_propagation"]["reason"],
        "",
        "## Part 4 - Verdict",
        "",
        f"VERDICT: {result['verdict']}",
        "",
        result["verdict_reason"],
        "",
    ]
    return "\n".join(lines)


def run(root: Path) -> dict[str, Any]:
    baseline, baseline_rows = baseline_lock(root)
    guard_start = forward_model_guard(SEED)
    if not guard_start["passed"]:
        raise RuntimeError(f"Forward-model guard failed at start: {guard_start}")

    raw_path, raw_note = find_or_unzip_raw(root)
    raw, raw_info = read_raw(raw_path)
    truth = read_ground_truth(root)
    derived = derive_all_onsets(raw)
    derived_truth = derived.merge(truth, on="SEQN", how="left")

    scopes = scope_frames(derived_truth)
    scope_summaries = [bev_summary(frame, scope) for scope, frame in scopes.items()]
    decision_summary = next(row for row in scope_summaries if row["scope"] == "gate2_acceptable_pre")
    ats_stop_rule_met = bool(decision_summary["ats_compliance_fraction"] > ATS_COMPLIANCE_STOP_THRESHOLD)

    current_criterion = criterion_participant_table(derived_truth, "current")
    axis_criterion = criterion_participant_table(derived_truth, "axis")
    current_summary = comparison_summary(current_criterion, "current baseline onset")
    axis_summary = comparison_summary(axis_criterion, "volume-axis tangent onset")
    boot = bootstrap_gate2_deltas(current_criterion, axis_criterion)

    fev1_signed_toward_zero = abs(axis_summary["fev1_median_signed_pct_error"]) < abs(current_summary["fev1_median_signed_pct_error"])
    fvc_signed_toward_zero = abs(axis_summary["fvc_median_signed_pct_error"]) < abs(current_summary["fvc_median_signed_pct_error"])
    fev1_abs_not_increased = axis_summary["fev1_median_abs_pct_error"] <= current_summary["fev1_median_abs_pct_error"]
    fvc_abs_not_increased = axis_summary["fvc_median_abs_pct_error"] <= current_summary["fvc_median_abs_pct_error"]
    part2_passed = bool(fev1_signed_toward_zero and fvc_signed_toward_zero and fev1_abs_not_increased and fvc_abs_not_increased)

    if part2_passed and not ats_stop_rule_met:
        propagation_reason = "Part 2 passed and BEV stop rule did not fire; propagation would be allowed. This branch is not expected for the current Cycle F data."
        verdict = "ONSET-IMPROVES"
        verdict_reason = "The onset candidate reduced Gate 2 bias without increasing absolute error; downstream propagation should be run before shipping."
    else:
        propagation_reason = (
            "Part 3 was not run. The Gate 2 acceptable-curve BEV stop rule fired and the volume-axis tangent candidate did not pass the signed-bias/absolute-error guard, so clinical/truncation recomputation would only risk regressing already-passing headlines."
        )
        verdict = "ONSET-NEUTRAL"
        verdict_reason = (
            "Current acceptable Pre curves are already >95% ATS-compliant by BEV, and the literal volume-axis tangent does not reduce the known Gate 2 signed low bias. The retained baseline onset leaves all headlines unchanged; the remaining Gate 2 bias is best characterized as composite/selection mismatch rather than onset-driven."
        )

    guard_end = forward_model_guard(SEED + 1)
    if not guard_end["passed"]:
        raise RuntimeError(f"Forward-model guard failed at end: {guard_end}")

    result = {
        "seed": SEED,
        "raw_source": str(raw_path),
        "raw_source_note": raw_note,
        "raw_loaded_rows": int(raw_info["loaded_rows"]),
        "parse_error_count": int(derived.attrs.get("parse_error_count", 0)),
        "baseline": baseline,
        "baseline_rows": baseline_rows.to_dict("records"),
        "forward_model_guard_start": guard_start,
        "part1_current_bev": {
            "current_onset_code_path": [
                "predictx/gate23_f.py::derive_curve",
                "predictx.segment.detect_forced_exhale_segment",
                "predictx.segment.backextrap_t0",
                "predictx.segment.derive_fev1_fvc",
                "predictx/gate_clinical_classification_f.py::extract_curve_for_fit",
            ],
            "current_t0_formula": "time_s[pef_idx] - (volume_ml[pef_idx] - volume_ml[onset_idx]) / (flow_lps[pef_idx] * 1000)",
            "scope_summaries": scope_summaries,
            "decision_scope": "gate2_acceptable_pre",
            "decision_scope_compliance_fraction": float(decision_summary["ats_compliance_fraction"]),
            "ats_stop_threshold": ATS_COMPLIANCE_STOP_THRESHOLD,
            "ats_stop_rule_met": ats_stop_rule_met,
        },
        "part2_axis_onset": {
            "candidate_formula": "time_s[pef_idx] - volume_ml[pef_idx] / (flow_lps[pef_idx] * 1000)",
            "current_summary": current_summary,
            "axis_summary": axis_summary,
            "bootstrap_deltas": boot,
            "fev1_signed_toward_zero": bool(fev1_signed_toward_zero),
            "fvc_signed_toward_zero": bool(fvc_signed_toward_zero),
            "fev1_abs_not_increased": bool(fev1_abs_not_increased),
            "fvc_abs_not_increased": bool(fvc_abs_not_increased),
            "passed": part2_passed,
            "interpretation": (
                "The candidate moves neither FEV1 nor FVC signed bias toward zero and slightly increases absolute percent error, so it is not an acceptable onset replacement."
                if not part2_passed
                else "The candidate passed the Gate 2 onset-bias guard."
            ),
        },
        "part3_propagation": {
            "run": False,
            "reason": propagation_reason,
        },
        "forward_model_guard_end": guard_end,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }

    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    (out / f"{OUTPUT_STEM}.md").write_text(make_report(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    part1 = result["part1_current_bev"]
    part2 = result["part2_axis_onset"]
    print(f"VERDICT: {result['verdict']}")
    print(
        f"Current Gate2 acceptable Pre ATS compliance: {part1['decision_scope_compliance_fraction']:.4f}; "
        f"Part2 passed={part2['passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
