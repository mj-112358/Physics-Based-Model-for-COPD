from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/predictx-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/predictx-xdg-cache")

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from .dataset_resolver import CYCLE_TO_YEAR
from .gate23_f import find_or_unzip_raw, read_ground_truth, read_raw
from .gate_clinical_classification_f import extract_curve_for_fit
from .gate_fit_optimization_f import forward_model_guard
from .gate_three_cycle_efg import (
    BOOTSTRAP_B,
    bootstrap_binary,
    clean_json,
    finite,
    incomplete_flagged_pool,
    load_covariates,
    md_table,
    participant_candidates,
    sample_by_stratum,
)
from .gate_truncation_recovery_f import fit_ombe_truncated, insert_time_point

SEED = 20260710
SOURCE_HZ = 100
TRUNCATION_TIMES = (2.0, 3.0)
OBSTRUCTION_THRESHOLD = 0.70
RESULT_JSON = "gate_degradation_robustness.json"
RESULT_MD = "gate_degradation_robustness.md"
RESULT_SWEEP = "gate_degradation_robustness_sweep.csv"


@dataclass(frozen=True)
class CurveRecord:
    cycle: str
    global_key: str
    seqn: float
    raw_index: int
    truth_severity: str
    true_fev1_l: float
    true_fvc_l: float
    true_ratio: float
    t: np.ndarray
    y: np.ndarray
    smoking_status: str = "unknown"
    self_reported_obstructive_disease: str = "unknown"
    age_band: str = "unknown"


@dataclass(frozen=True)
class SevereRecord:
    cycle: str
    global_key: str
    seqn: float
    raw_index: int
    target_fet_s: float
    true_fev1_l: float
    true_fvc_l: float
    true_ratio: float
    t: np.ndarray
    y: np.ndarray


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in (SEED, *parts)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def pct_error(est: float, truth: float) -> float:
    return float(100.0 * (est - truth) / truth) if np.isfinite(est) and np.isfinite(truth) and truth > 0 else np.nan


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float, float, bool]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan, False
    point = float(arr.mean())
    rng = np.random.default_rng(seed)
    idx = np.arange(arr.size)
    draws = np.empty(BOOTSTRAP_B, dtype=float)
    for b in range(BOOTSTRAP_B):
        draws[b] = float(arr[rng.choice(idx, size=arr.size, replace=True)].mean())
    low = float(np.nanpercentile(draws, 2.5))
    high = float(np.nanpercentile(draws, 97.5))
    return point, low, high, bool(low > 0.0 or high < 0.0)


def cp_upper_zero(n: int, alpha: float = 0.05) -> float:
    if n <= 0:
        return np.nan
    return float(1.0 - (alpha / 2.0) ** (1.0 / n))


def summarize_rate(flags: np.ndarray, seed: int, *, severe: bool = False) -> dict[str, Any]:
    arr = np.asarray(flags, dtype=bool)
    n = int(arr.size)
    k = int(arr.sum())
    rate = k / n if n else np.nan
    if severe and n and k == 0:
        low = 0.0
        high = cp_upper_zero(n)
        method = f"exact two-sided Clopper-Pearson 95% upper bound for 0/{n}"
    else:
        low, high = bootstrap_binary(arr, seed)
        method = "paired nonparametric bootstrap"
    return {
        "rate": rate,
        "n": k,
        "den": n,
        "ci_low": low,
        "ci_high": high,
        "ci_method": method,
        "cp_upper_95": cp_upper_zero(n) if severe and n and k == 0 else np.nan,
    }


def flow_from_volume(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    edge_order = 2 if len(t) > 2 else 1
    flow = np.gradient(y, t, edge_order=edge_order)
    return np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)


def volume_from_flow(t: np.ndarray, flow: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    flow = np.asarray(flow, dtype=float)
    flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
    increments = 0.5 * (flow[:-1] + flow[1:]) * np.diff(t)
    y = np.concatenate([[0.0], np.cumsum(increments)])
    y = np.maximum.accumulate(y)
    y = y - y[0]
    return y


def resample_flow_antialias(t: np.ndarray, flow: np.ndarray, target_hz: int) -> np.ndarray:
    if target_hz >= SOURCE_HZ:
        return flow.copy()
    gcd = math.gcd(SOURCE_HZ, int(target_hz))
    up = int(target_hz // gcd)
    down = int(SOURCE_HZ // gcd)
    low = resample_poly(flow, up=up, down=down, padtype="line")
    low_t = np.linspace(float(t[0]), float(t[-1]), num=len(low))
    return np.interp(t, low_t, low)


def degrade_curve(record_key: str, t: np.ndarray, y: np.ndarray, profile: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    out_t = np.asarray(t, dtype=float).copy()
    clean_y = np.asarray(y, dtype=float).copy()
    noise_frac = float(profile.get("noise_frac_pef", 0.0) or 0.0)
    drift_frac = float(profile.get("drift_frac_fvc", 0.0) or 0.0)
    sample_hz = int(profile.get("sample_hz", SOURCE_HZ) or SOURCE_HZ)
    gain = float(profile.get("gain_error", 0.0) or 0.0)
    differential_gain = float(profile.get("differential_gain_error", 0.0) or 0.0)
    if noise_frac == 0.0 and drift_frac == 0.0 and sample_hz >= SOURCE_HZ and differential_gain == 0.0:
        out_y = clean_y.copy()
        if gain:
            out_y = out_y * (1.0 + gain)
        return out_t, out_y

    flow = flow_from_volume(out_t, clean_y)
    pef = max(float(np.nanmax(flow)), 1e-9)
    rng = np.random.default_rng(stable_seed(profile["profile_id"], record_key))

    if noise_frac > 0:
        flow = flow + rng.normal(loc=0.0, scale=noise_frac * pef, size=flow.shape)

    if differential_gain != 0.0 and out_t[-1] > out_t[0]:
        early_gain = max(0.01, 1.0 + differential_gain)
        late_gain = max(0.01, 1.0 - differential_gain)
        flow_gain = np.where(out_t <= 1.0, early_gain, late_gain)
        flow = flow * flow_gain

    if drift_frac > 0 and out_t[-1] > out_t[0]:
        sign = -1.0 if rng.random() < 0.5 else 1.0
        duration = max(float(out_t[-1] - out_t[0]), 1e-6)
        area = 2.0 * duration / math.pi
        amp_l = drift_frac * max(float(clean_y[-1]), 1e-9)
        drift_flow_amp = amp_l / area
        phase_t = (out_t - out_t[0]) / duration
        flow = flow + sign * drift_flow_amp * np.sin(np.pi * phase_t)

    flow = resample_flow_antialias(out_t, flow, sample_hz)
    out_y = volume_from_flow(out_t, flow)

    if gain:
        out_y = out_y * (1.0 + gain)
    return out_t, out_y


def profile_defs() -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = [
        {
            "profile_id": "baseline",
            "operator": "baseline",
            "level": "clean",
            "level_order": 0,
            "noise_frac_pef": 0.0,
            "sample_hz": SOURCE_HZ,
            "gain_error": 0.0,
            "drift_frac_fvc": 0.0,
        }
    ]
    for order, frac in enumerate([0.005, 0.01, 0.02, 0.05, 0.10], start=1):
        profiles.append(
            {
                "profile_id": f"D1_noise_{frac:.3f}",
                "operator": "D1_additive_noise",
                "level": f"{frac * 100:.1f}% PEF",
                "level_order": order,
                "noise_frac_pef": frac,
                "sample_hz": SOURCE_HZ,
                "gain_error": 0.0,
                "drift_frac_fvc": 0.0,
            }
        )
    for order, hz in enumerate([100, 50, 30, 20, 10], start=1):
        profiles.append(
            {
                "profile_id": f"D2_sampling_{hz}Hz",
                "operator": "D2_coarser_sampling",
                "level": f"{hz} Hz",
                "level_order": order,
                "noise_frac_pef": 0.0,
                "sample_hz": hz,
                "gain_error": 0.0,
                "drift_frac_fvc": 0.0,
            }
        )
    gain_order = 1
    for abs_gain in [0.01, 0.02, 0.05, 0.10]:
        for sign in [-1.0, 1.0]:
            gain = sign * abs_gain
            profiles.append(
                {
                    "profile_id": f"D3_gain_{gain:+.3f}",
                    "operator": "D3_gain_error",
                    "level": f"{gain * 100:+.0f}%",
                    "level_order": gain_order,
                    "noise_frac_pef": 0.0,
                    "sample_hz": SOURCE_HZ,
                    "gain_error": gain,
                    "drift_frac_fvc": 0.0,
                }
            )
            gain_order += 1
    for order, (label, frac) in enumerate([("small", 0.005), ("medium", 0.01), ("large", 0.02)], start=1):
        profiles.append(
            {
                "profile_id": f"D4_drift_{label}",
                "operator": "D4_baseline_drift",
                "level": f"{label} ({frac * 100:.1f}% FVC net drift)",
                "level_order": order,
                "noise_frac_pef": 0.0,
                "sample_hz": SOURCE_HZ,
                "gain_error": 0.0,
                "drift_frac_fvc": frac,
            }
        )
    profiles.append(
        {
            "profile_id": "combined_budget_realistic",
            "operator": "combined_realistic_budget",
            "level": "2% PEF noise + 30 Hz + +2% gain + small drift",
            "level_order": 1,
            "noise_frac_pef": 0.02,
            "sample_hz": 30,
            "gain_error": 0.02,
            "drift_frac_fvc": 0.005,
        }
    )
    return profiles


def build_covariate_lookup(root: Path) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, str]]:
    cov, notes, variables = load_covariates(root)
    if cov.empty:
        return {}, notes, variables
    fields = ["smoking_status", "self_reported_obstructive_disease", "age_band"]
    return cov.set_index("global_key")[fields].to_dict("index"), notes, variables


def build_curve_copies(root: Path) -> tuple[list[CurveRecord], list[SevereRecord], list[dict[str, Any]], list[str], dict[str, str]]:
    cov_lookup, cov_notes, cov_variables = build_covariate_lookup(root)
    clean_records: list[CurveRecord] = []
    severe_records: list[SevereRecord] = []
    sample_meta: list[dict[str, Any]] = []
    raw_by_cycle: dict[str, pd.DataFrame] = {}
    selected_complete_by_cycle: dict[str, pd.DataFrame] = {}

    for cycle in CYCLE_TO_YEAR:
        raw_path, _note = find_or_unzip_raw(root, cycle)
        raw, _raw_info = read_raw(raw_path, cycle)
        truth = read_ground_truth(root, cycle)
        raw_by_cycle[cycle] = raw
        selected_a, selected_complete = participant_candidates(raw, truth, cycle)
        selected_complete_by_cycle[cycle] = selected_complete
        sample, sample_rows = sample_by_stratum(selected_a, selected_complete, cycle)
        sample_meta.extend(sample_rows)
        raw_by_index = {int(idx): row for idx, row in raw.iterrows()}
        total = len(sample)
        for i, sample_row in enumerate(sample.itertuples(index=False), start=1):
            curve = extract_curve_for_fit(raw_by_index[int(sample_row.raw_index)])
            t = np.asarray(curve["vol_t"], dtype=float).copy()
            y = np.asarray(curve["vol_l"], dtype=float).copy()
            true_fev1_l = float(np.interp(1.0, t, y))
            true_fvc_l = float(y[-1])
            true_ratio = true_fev1_l / true_fvc_l if true_fvc_l > 0 else np.nan
            cov = cov_lookup.get(str(sample_row.global_key), {})
            clean_records.append(
                CurveRecord(
                    cycle=str(sample_row.cycle),
                    global_key=str(sample_row.global_key),
                    seqn=float(sample_row.SEQN),
                    raw_index=int(sample_row.raw_index),
                    truth_severity=str(sample_row.truth_severity),
                    true_fev1_l=true_fev1_l,
                    true_fvc_l=true_fvc_l,
                    true_ratio=true_ratio,
                    t=t,
                    y=y,
                    smoking_status=str(cov.get("smoking_status", "unknown")),
                    self_reported_obstructive_disease=str(cov.get("self_reported_obstructive_disease", "unknown")),
                    age_band=str(cov.get("age_band", "unknown")),
                )
            )
            if i % 100 == 0 or i == total:
                print(f"Cycle {cycle} clean-copy parse progress: {i}/{total}")

    for cycle, raw in raw_by_cycle.items():
        selected_complete = selected_complete_by_cycle[cycle]
        severe_cohort = incomplete_flagged_pool(raw, selected_complete, cycle)
        if severe_cohort.empty:
            continue
        raw_by_index = {int(idx): row for idx, row in raw.iterrows()}
        for row in severe_cohort.itertuples(index=False):
            curve = extract_curve_for_fit(raw_by_index[int(row.raw_index)])
            severe_records.append(
                SevereRecord(
                    cycle=str(row.cycle),
                    global_key=str(row.global_key),
                    seqn=float(row.SEQN),
                    raw_index=int(row.raw_index),
                    target_fet_s=float(row.fet_s),
                    true_fev1_l=float(row.true_fev1_l),
                    true_fvc_l=float(row.true_fvc_l),
                    true_ratio=float(row.true_ratio),
                    t=np.asarray(curve["vol_t"], dtype=float).copy(),
                    y=np.asarray(curve["vol_l"], dtype=float).copy(),
                )
            )
    return clean_records, severe_records, sample_meta, cov_notes, cov_variables


def gate2_accuracy(records: list[CurveRecord], degraded: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    for rec in records:
        t, y = degraded[rec.global_key]
        fev1_l = float(np.interp(min(1.0, float(t[-1])), t, y))
        fvc_l = float(y[-1])
        ratio = fev1_l / fvc_l if fvc_l > 0 else np.nan
        rows.append(
            {
                "fev1_signed_pct_error": pct_error(fev1_l, rec.true_fev1_l),
                "fev1_abs_pct_error": abs(pct_error(fev1_l, rec.true_fev1_l)),
                "fvc_signed_pct_error": pct_error(fvc_l, rec.true_fvc_l),
                "fvc_abs_pct_error": abs(pct_error(fvc_l, rec.true_fvc_l)),
                "ratio_signed_pct_error": pct_error(ratio, rec.true_ratio),
                "ratio_abs_pct_error": abs(pct_error(ratio, rec.true_ratio)),
            }
        )
    df = pd.DataFrame(rows)
    return {
        "gate2_clean_copy_n": int(len(df)),
        "gate2_fev1_median_signed_pct_error": float(df["fev1_signed_pct_error"].median()),
        "gate2_fev1_median_abs_pct_error": float(df["fev1_abs_pct_error"].median()),
        "gate2_fvc_median_signed_pct_error": float(df["fvc_signed_pct_error"].median()),
        "gate2_fvc_median_abs_pct_error": float(df["fvc_abs_pct_error"].median()),
        "gate2_ratio_median_signed_pct_error": float(df["ratio_signed_pct_error"].median()),
        "gate2_ratio_median_abs_pct_error": float(df["ratio_abs_pct_error"].median()),
        "gate2_pass_both_abs_le_5": bool(
            float(df["fev1_abs_pct_error"].median()) <= 5.0 and float(df["fvc_abs_pct_error"].median()) <= 5.0
        ),
    }


def evaluate_profile(records: list[CurveRecord], profile: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    degraded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for rec in records:
        degraded[rec.global_key] = degrade_curve(rec.global_key, rec.t, rec.y, profile)
    gate2 = gate2_accuracy(records, degraded)

    prediction_rows: list[dict[str, Any]] = []
    obstructed = [rec for rec in records if np.isfinite(rec.true_ratio) and rec.true_ratio < OBSTRUCTION_THRESHOLD]
    total = len(obstructed)
    for i, rec in enumerate(obstructed, start=1):
        t, y = degraded[rec.global_key]
        for stop_s in TRUNCATION_TIMES:
            t_fit, y_fit = insert_time_point(t, y, stop_s)
            observed_fev1_l = float(np.interp(min(1.0, float(t_fit[-1])), t_fit, y_fit))
            direct_fvc_l = float(y_fit[-1])
            direct_ratio = observed_fev1_l / direct_fvc_l if direct_fvc_l > 0 else np.nan
            direct_pred = bool(np.isfinite(direct_ratio) and direct_ratio < OBSTRUCTION_THRESHOLD)
            fit = fit_ombe_truncated(t, y, stop_s)
            ombe_fvc_l = float(fit.get("fvc_l", np.nan)) if finite(fit.get("fvc_l", np.nan)) else np.nan
            ombe_success = bool(fit.get("success", False) and np.isfinite(ombe_fvc_l) and ombe_fvc_l > 0)
            ombe_ratio = observed_fev1_l / ombe_fvc_l if ombe_success else np.nan
            ombe_pred = bool(np.isfinite(ombe_ratio) and ombe_ratio < OBSTRUCTION_THRESHOLD and ombe_success)
            prediction_rows.append(
                {
                    "profile_id": profile["profile_id"],
                    "operator": profile["operator"],
                    "level": profile["level"],
                    "cycle": rec.cycle,
                    "global_key": rec.global_key,
                    "SEQN": rec.seqn,
                    "T_label": str(int(stop_s)),
                    "truth_obstructed_fixed": True,
                    "truth_severity": rec.truth_severity,
                    "self_reported_obstructive_disease": rec.self_reported_obstructive_disease,
                    "smoking_status": rec.smoking_status,
                    "age_band": rec.age_band,
                    "direct_pred": direct_pred,
                    "ombe_pred": ombe_pred,
                    "direct_fn": not direct_pred,
                    "ombe_fn": not ombe_pred,
                    "ombe_fit_success": ombe_success,
                    "ombe_fit_failure": "" if ombe_success else str(fit.get("failure", "nonfinite OMBE fit")),
                }
            )
        if i % 100 == 0 or i == total:
            print(f"{profile['profile_id']} OMBE progress: {i}/{total} obstructed clean curves")

    summary_rows = summarize_prediction_rows(prediction_rows, gate2, profile)
    return summary_rows, prediction_rows, gate2


def summarize_prediction_rows(
    prediction_rows: list[dict[str, Any]],
    gate2: dict[str, Any],
    profile: dict[str, Any],
    *,
    scope: str = "pooled_clean_obstructed",
) -> list[dict[str, Any]]:
    df = pd.DataFrame(prediction_rows)
    rows: list[dict[str, Any]] = []
    for t_label in ["2", "3"]:
        part = df[df["T_label"].astype(str).eq(t_label)].copy()
        direct_fn = part["direct_fn"].to_numpy(bool)
        ombe_fn = part["ombe_fn"].to_numpy(bool)
        direct = summarize_rate(direct_fn, stable_seed(profile["profile_id"], t_label, "direct"))
        ombe = summarize_rate(ombe_fn, stable_seed(profile["profile_id"], t_label, "ombe"))
        diff_point, diff_low, diff_high, excludes = bootstrap_mean(
            direct_fn.astype(float) - ombe_fn.astype(float),
            stable_seed(profile["profile_id"], t_label, "diff"),
        )
        rows.append(
            {
                "scope": scope,
                "profile_id": profile["profile_id"],
                "operator": profile["operator"],
                "level": profile["level"],
                "level_order": int(profile["level_order"]),
                "T_label": t_label,
                "n_obstructed": int(len(part)),
                "direct_fn": direct["rate"],
                "direct_fn_n": direct["n"],
                "direct_fn_den": direct["den"],
                "direct_fn_ci_low": direct["ci_low"],
                "direct_fn_ci_high": direct["ci_high"],
                "ombe_fn": ombe["rate"],
                "ombe_fn_n": ombe["n"],
                "ombe_fn_den": ombe["den"],
                "ombe_fn_ci_low": ombe["ci_low"],
                "ombe_fn_ci_high": ombe["ci_high"],
                "direct_minus_ombe": diff_point,
                "direct_minus_ombe_ci_low": diff_low,
                "direct_minus_ombe_ci_high": diff_high,
                "direct_minus_ombe_excludes_zero": excludes,
                "ombe_fit_failure_n": int((~part["ombe_fit_success"].astype(bool)).sum()),
                "row_verdict": "HOLDS" if excludes and np.isfinite(diff_point) and diff_point > 0 else "BREAKS",
                **gate2,
            }
        )
    return rows


def evaluate_severe(records: list[SevereRecord], profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        t, y = degrade_curve(f"{rec.global_key}:field", rec.t, rec.y, profile)
        stop_s = float(t[-1])
        t_fit, y_fit = insert_time_point(t, y, stop_s)
        observed_fev1_l = float(np.interp(min(1.0, float(t_fit[-1])), t_fit, y_fit))
        direct_fvc_l = float(y_fit[-1])
        direct_ratio = observed_fev1_l / direct_fvc_l if direct_fvc_l > 0 else np.nan
        direct_pred = bool(np.isfinite(direct_ratio) and direct_ratio < OBSTRUCTION_THRESHOLD)
        fit = fit_ombe_truncated(t, y, stop_s)
        ombe_fvc_l = float(fit.get("fvc_l", np.nan)) if finite(fit.get("fvc_l", np.nan)) else np.nan
        ombe_success = bool(fit.get("success", False) and np.isfinite(ombe_fvc_l) and ombe_fvc_l > 0)
        ombe_ratio = observed_fev1_l / ombe_fvc_l if ombe_success else np.nan
        ombe_pred = bool(np.isfinite(ombe_ratio) and ombe_ratio < OBSTRUCTION_THRESHOLD and ombe_success)
        rows.append(
            {
                "profile_id": profile["profile_id"],
                "operator": profile["operator"],
                "level": profile["level"],
                "global_key": rec.global_key,
                "T_label": "field_incomplete",
                "direct_pred": direct_pred,
                "ombe_pred": ombe_pred,
                "direct_fn": not direct_pred,
                "ombe_fn": not ombe_pred,
                "ombe_fit_success": ombe_success,
            }
        )
    df = pd.DataFrame(rows)
    out = []
    for estimator, col in [("E0_DIRECT", "direct_fn"), ("E1_OMBE", "ombe_fn")]:
        flags = df[col].to_numpy(bool)
        rate = summarize_rate(flags, stable_seed(profile["profile_id"], "severe", estimator), severe=True)
        out.append(
            {
                "profile_id": profile["profile_id"],
                "operator": profile["operator"],
                "level": profile["level"],
                "estimator": estimator,
                "T_label": "field_incomplete",
                "fn_rate": rate["rate"],
                "fn_rate_n": rate["n"],
                "fn_rate_den": rate["den"],
                "fn_rate_ci_low": rate["ci_low"],
                "fn_rate_ci_high": rate["ci_high"],
                "fn_rate_ci_method": rate["ci_method"],
                "fn_rate_cp_upper_95": rate["cp_upper_95"],
                "ombe_fit_failure_n": int((~df["ombe_fit_success"].astype(bool)).sum()) if estimator == "E1_OMBE" else 0,
            }
        )
    return out


def summarize_external(prediction_rows: list[dict[str, Any]], profile: dict[str, Any]) -> list[dict[str, Any]]:
    df = pd.DataFrame(prediction_rows)
    part = df[df["self_reported_obstructive_disease"].eq("yes")].copy()
    rows: list[dict[str, Any]] = []
    gate2_blank = {
        "gate2_clean_copy_n": np.nan,
        "gate2_fev1_median_signed_pct_error": np.nan,
        "gate2_fev1_median_abs_pct_error": np.nan,
        "gate2_fvc_median_signed_pct_error": np.nan,
        "gate2_fvc_median_abs_pct_error": np.nan,
        "gate2_ratio_median_signed_pct_error": np.nan,
        "gate2_ratio_median_abs_pct_error": np.nan,
        "gate2_pass_both_abs_le_5": False,
    }
    for t_label in ["2", "3"]:
        sub = part[part["T_label"].astype(str).eq(t_label)]
        if sub.empty:
            continue
        direct_fn = sub["direct_fn"].to_numpy(bool)
        ombe_fn = sub["ombe_fn"].to_numpy(bool)
        direct = summarize_rate(direct_fn, stable_seed(profile["profile_id"], "external", t_label, "direct"))
        ombe = summarize_rate(ombe_fn, stable_seed(profile["profile_id"], "external", t_label, "ombe"))
        diff_point, diff_low, diff_high, excludes = bootstrap_mean(
            direct_fn.astype(float) - ombe_fn.astype(float),
            stable_seed(profile["profile_id"], "external", t_label, "diff"),
        )
        rows.append(
            {
                "scope": "external_self_reported_obstructive_disease_yes",
                "profile_id": profile["profile_id"],
                "operator": profile["operator"],
                "level": profile["level"],
                "level_order": int(profile["level_order"]),
                "T_label": t_label,
                "n_obstructed": int(len(sub)),
                "direct_fn": direct["rate"],
                "direct_fn_n": direct["n"],
                "direct_fn_den": direct["den"],
                "direct_fn_ci_low": direct["ci_low"],
                "direct_fn_ci_high": direct["ci_high"],
                "ombe_fn": ombe["rate"],
                "ombe_fn_n": ombe["n"],
                "ombe_fn_den": ombe["den"],
                "ombe_fn_ci_low": ombe["ci_low"],
                "ombe_fn_ci_high": ombe["ci_high"],
                "direct_minus_ombe": diff_point,
                "direct_minus_ombe_ci_low": diff_low,
                "direct_minus_ombe_ci_high": diff_high,
                "direct_minus_ombe_excludes_zero": excludes,
                "ombe_fit_failure_n": int((~sub["ombe_fit_success"].astype(bool)).sum()),
                "row_verdict": "HOLDS" if excludes and np.isfinite(diff_point) and diff_point > 0 else "BREAKS",
                **gate2_blank,
            }
        )
    return rows


def expected_baseline_from_current(root: Path) -> dict[str, Any]:
    path = root / "results" / "gate_three_cycle_EFG.json"
    if not path.exists():
        raise RuntimeError(f"Missing current three-cycle result for baseline lock: {path}")
    current = json.loads(path.read_text(encoding="utf-8"))
    expected: dict[str, Any] = {"T": {}, "severe": {}}
    cross = pd.DataFrame(current["cross_cycle_summary"])
    for t_label in ["2", "3"]:
        part = cross[cross["T_label"].astype(str).eq(t_label)]
        expected["T"][t_label] = {
            "direct_fn_n": int(part["direct_fn_n"].sum()),
            "direct_fn_den": int(part["direct_fn_den"].sum()),
            "ombe_fn_n": int(part["ombe_fn_n"].sum()),
            "ombe_fn_den": int(part["ombe_fn_den"].sum()),
        }
    severe = pd.DataFrame(current.get("severe_incomplete_summary", []))
    e1 = severe[(severe["estimator"].eq("E1_OMBE")) & (severe["T_label"].eq("field_incomplete"))]
    if e1.empty:
        raise RuntimeError("Current three-cycle result lacks E1_OMBE severe field baseline.")
    expected["severe"] = {
        "ombe_fn_n": int(e1.iloc[0]["fn_rate_n"]),
        "ombe_fn_den": int(e1.iloc[0]["fn_rate_den"]),
    }
    return expected


def assert_baseline_matches(root: Path, baseline_rows: list[dict[str, Any]], severe_rows: list[dict[str, Any]]) -> None:
    expected = expected_baseline_from_current(root)
    by_t = {str(row["T_label"]): row for row in baseline_rows}
    mismatches = []
    for t_label, exp in expected["T"].items():
        row = by_t[t_label]
        for key in ["direct_fn_n", "direct_fn_den", "ombe_fn_n", "ombe_fn_den"]:
            if int(row[key]) != int(exp[key]):
                mismatches.append(f"T={t_label} {key}: got {int(row[key])}, expected {int(exp[key])}")
    e1 = [row for row in severe_rows if row["estimator"] == "E1_OMBE"]
    if not e1:
        mismatches.append("missing severe E1_OMBE baseline")
    else:
        row = e1[0]
        if int(row["fn_rate_n"]) != int(expected["severe"]["ombe_fn_n"]) or int(row["fn_rate_den"]) != int(expected["severe"]["ombe_fn_den"]):
            mismatches.append(
                f"severe E1_OMBE: got {int(row['fn_rate_n'])}/{int(row['fn_rate_den'])}, "
                f"expected {expected['severe']['ombe_fn_n']}/{expected['severe']['ombe_fn_den']}"
            )
    if mismatches:
        raise RuntimeError("Clean-copy baseline diverged from current three-cycle results; halting. " + "; ".join(mismatches))


def breaking_points(sweep_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(sweep_rows)
    rows: list[dict[str, Any]] = []
    for operator in ["D1_additive_noise", "D2_coarser_sampling", "D3_gain_error", "D4_baseline_drift"]:
        op = df[(df["scope"].eq("pooled_clean_obstructed")) & (df["operator"].eq(operator))].copy()
        break_row: dict[str, Any] | None = None
        for level_order in sorted(op["level_order"].unique()):
            part = op[op["level_order"].eq(level_order)]
            holds = bool((part["direct_minus_ombe"].astype(float) > 0).all() and part["direct_minus_ombe_excludes_zero"].map(bool).all())
            if not holds:
                first = part.sort_values("T_label").iloc[0].to_dict()
                break_row = {
                    "operator": operator,
                    "breaking_level": str(first["level"]),
                    "breaking_level_order": int(level_order),
                    "breaks": True,
                    "note": "first level where T=2/T=3 OMBE advantage did not both exclude zero",
                }
                break
        if break_row is None:
            rows.append(
                {
                    "operator": operator,
                    "breaking_level": "not reached in sweep",
                    "breaking_level_order": None,
                    "breaks": False,
                    "note": "all swept levels retained positive DIRECT-minus-OMBE CI exclusion at T=2 and T=3",
                }
            )
        else:
            rows.append(break_row)
    return rows


def sensitivity_summary(sweep_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(sweep_rows)
    baseline_min = min(float(row["direct_minus_ombe"]) for row in baseline_rows)
    rows = []
    for operator in ["D1_additive_noise", "D2_coarser_sampling", "D3_gain_error", "D4_baseline_drift"]:
        op = df[(df["scope"].eq("pooled_clean_obstructed")) & (df["operator"].eq(operator))].copy()
        worst = float(op["direct_minus_ombe"].min())
        rows.append(
            {
                "operator": operator,
                "baseline_min_margin": baseline_min,
                "worst_margin": worst,
                "margin_drop": baseline_min - worst,
                "worst_level": str(op.loc[op["direct_minus_ombe"].astype(float).idxmin(), "level"]),
            }
        )
    ranked = sorted(rows, key=lambda row: row["margin_drop"], reverse=True)
    return {"rows": rows, "most_sensitive": ranked[0], "least_sensitive": ranked[-1]}


def decide_verdict(combined_rows: list[dict[str, Any]], break_rows: list[dict[str, Any]]) -> str:
    combined_holds = all(
        bool(row["direct_minus_ombe_excludes_zero"]) and float(row["direct_minus_ombe"]) > 0
        for row in combined_rows
        if row["scope"] == "pooled_clean_obstructed"
    )
    if not combined_holds:
        return "NOT-ROBUST"
    any_break = any(bool(row["breaks"]) for row in break_rows)
    return "ROBUST-WITH-LIMITS" if any_break else "ROBUST-TO-BUDGET-DEVICE"


def add_delta_fields(rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> None:
    baseline_by_t = {row["T_label"]: row for row in baseline_rows}
    for row in rows:
        if row.get("scope") != "pooled_clean_obstructed":
            continue
        base = baseline_by_t.get(row["T_label"])
        if not base:
            continue
        row["delta_direct_fn"] = float(row["direct_fn"]) - float(base["direct_fn"])
        row["delta_ombe_fn"] = float(row["ombe_fn"]) - float(base["ombe_fn"])
        row["delta_direct_minus_ombe"] = float(row["direct_minus_ombe"]) - float(base["direct_minus_ombe"])


def model_basis_rows() -> list[dict[str, Any]]:
    return [
        {
            "operator": "D1 additive flow noise",
            "realistic_level": "2% of PEF",
            "basis": "portable and acoustic/low-cost devices motivate explicit noise-floor stress; ATS/ERS accuracy standards define the research-grade reference expectation.",
        },
        {
            "operator": "D2 coarser sampling",
            "realistic_level": "30 Hz",
            "basis": "NHANES waveforms are 100 Hz; 30 Hz tests a lower ADC/sample-rate budget device while preserving anti-alias filtering.",
        },
        {
            "operator": "D3 gain/calibration error",
            "realistic_level": "+/-2%; combined uses +2%",
            "basis": "ATS/ERS calibration expectations make volume scale error a clinically relevant stressor; the sweep tests both signs through +/-10%.",
        },
        {
            "operator": "D4 baseline drift",
            "realistic_level": "small, 0.5% FVC net drift",
            "basis": "zero instability in low-cost flow sensing produces slow baseline wander; this is modeled as low-frequency net drift over the maneuver.",
        },
    ]


def write_report(root: Path, result: dict[str, Any], sweep_rows: list[dict[str, Any]]) -> None:
    out = root / "results"
    baseline_rows = result["baseline"]["truncation_rows"]
    combined_rows = result["combined"]["truncation_rows"]
    severe_combined = result["combined"]["severe_rows"]
    external_combined = result["combined"]["external_rows"]
    lines = [
        "# Gate Degradation Robustness",
        "",
        f"VERDICT: {result['verdict']}",
        "",
        "## Data Safety And Guards",
        "",
        f"- Clean-copy location: {result['copy_location']}",
        "- NHANES source safety: no writes are made to `NHANES_SpiroSBI/`; only `results/` outputs are written.",
        f"- Forward-model guard at start: max JAX-vs-NumPy volume diff {result['forward_model_guards'][0]['max_abs_volume_ml']:.12g} mL.",
        f"- Forward-model guard at end: max JAX-vs-NumPy volume diff {result['forward_model_guards'][1]['max_abs_volume_ml']:.12g} mL.",
        "",
        "## Physical Basis",
        "",
        "Citations used for the device-class basis: ATS/ERS 2019 spirometry technical standard "
        "https://doi.org/10.1164/rccm.201908-1590ST, portable AioCare spirometry work "
        "https://arxiv.org/abs/1903.03588, and low-cost paper/acoustic spirometry work "
        "https://arxiv.org/abs/2408.09389. The levels below are stress-test budgets, not a claim that every cited device has exactly these errors.",
        "",
        md_table(model_basis_rows(), [("operator", "operator", "str"), ("realistic_level", "realistic budget level", "str"), ("basis", "basis", "str")]),
        "",
        "## Baseline Lock",
        "",
        "The clean in-memory copy was required to match the current three-cycle pooled result before degradation was accepted.",
        "",
        md_table(
            baseline_rows,
            [
                ("T_label", "T", "str"),
                ("direct_fn", "DIRECT FN", "rate"),
                ("ombe_fn", "OMBE FN", "rate"),
                ("direct_minus_ombe", "DIRECT-OMBE", "float3"),
                ("direct_minus_ombe_ci_low", "CI low", "float3"),
                ("direct_minus_ombe_ci_high", "CI high", "float3"),
                ("direct_minus_ombe_excludes_zero", "excludes 0", "bool"),
            ],
        ),
        "",
        "Severe field baseline:",
        "",
        md_table(
            result["baseline"]["severe_rows"],
            [
                ("estimator", "estimator", "str"),
                ("fn_rate", "FN", "rate"),
                ("fn_rate_cp_upper_95", "CP upper 95", "float3"),
                ("ombe_fit_failure_n", "fit failures", "int"),
            ],
        ),
        "",
        "## Full Single-Degradation Sweep",
        "",
    ]
    for operator in ["D1_additive_noise", "D2_coarser_sampling", "D3_gain_error", "D4_baseline_drift"]:
        op_rows = [
            row
            for row in sweep_rows
            if row["scope"] == "pooled_clean_obstructed" and row["operator"] == operator
        ]
        lines.extend(
            [
                f"### {operator}",
                "",
                md_table(
                    op_rows,
                    [
                        ("level", "level", "str"),
                        ("T_label", "T", "str"),
                        ("direct_fn", "DIRECT FN", "rate"),
                        ("ombe_fn", "OMBE FN", "rate"),
                        ("direct_minus_ombe", "diff", "float3"),
                        ("direct_minus_ombe_ci_low", "CI low", "float3"),
                        ("direct_minus_ombe_ci_high", "CI high", "float3"),
                        ("direct_minus_ombe_excludes_zero", "excludes 0", "bool"),
                        ("gate2_fev1_median_abs_pct_error", "Gate2 FEV1 abs %", "float3"),
                        ("gate2_fvc_median_abs_pct_error", "Gate2 FVC abs %", "float3"),
                        ("row_verdict", "verdict", "str"),
                    ],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Breaking Points",
            "",
            md_table(
                result["breaking_points"],
                [("operator", "operator", "str"), ("breaking_level", "breaking point", "str"), ("breaks", "breaks", "bool"), ("note", "note", "str")],
            ),
            "",
            "## Combined Realistic Budget Profile",
            "",
            f"Profile: {result['combined']['profile']['level']}.",
            "",
            md_table(
                combined_rows,
                [
                    ("T_label", "T", "str"),
                    ("direct_fn", "DIRECT FN", "rate"),
                    ("ombe_fn", "OMBE FN", "rate"),
                    ("direct_minus_ombe", "DIRECT-OMBE", "float3"),
                    ("direct_minus_ombe_ci_low", "CI low", "float3"),
                    ("direct_minus_ombe_ci_high", "CI high", "float3"),
                    ("direct_minus_ombe_excludes_zero", "excludes 0", "bool"),
                    ("delta_ombe_fn", "delta OMBE FN", "float3"),
                ],
            ),
            "",
            "Combined severe field recovery:",
            "",
            md_table(
                severe_combined,
                [
                    ("estimator", "estimator", "str"),
                    ("fn_rate", "FN", "rate"),
                    ("fn_rate_cp_upper_95", "CP upper 95", "float3"),
                    ("ombe_fit_failure_n", "fit failures", "int"),
                ],
            ),
            "",
            "COPD-positive external-validity stratum under the combined profile:",
            "",
            md_table(
                external_combined,
                [
                    ("T_label", "T", "str"),
                    ("n_obstructed", "n", "int"),
                    ("direct_fn", "DIRECT FN", "rate"),
                    ("ombe_fn", "OMBE FN", "rate"),
                    ("direct_minus_ombe", "DIRECT-OMBE", "float3"),
                    ("direct_minus_ombe_ci_low", "CI low", "float3"),
                    ("direct_minus_ombe_ci_high", "CI high", "float3"),
                    ("direct_minus_ombe_excludes_zero", "excludes 0", "bool"),
                ],
            ),
            "",
            "## Sensitivity",
            "",
            md_table(
                result["sensitivity"]["rows"],
                [
                    ("operator", "operator", "str"),
                    ("worst_level", "worst level", "str"),
                    ("baseline_min_margin", "baseline min margin", "float3"),
                    ("worst_margin", "worst margin", "float3"),
                    ("margin_drop", "margin drop", "float3"),
                ],
            ),
            "",
            f"- Most sensitive: {result['sensitivity']['most_sensitive']['operator']} at {result['sensitivity']['most_sensitive']['worst_level']}.",
            f"- Least sensitive: {result['sensitivity']['least_sensitive']['operator']} at {result['sensitivity']['least_sensitive']['worst_level']}.",
            "",
            "## Protocol Decisions",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in result["protocol_decisions"])
    lines.append("")
    (out / RESULT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    guard_start = forward_model_guard(SEED)
    if not guard_start["passed"] or float(guard_start["max_abs_volume_ml"]) >= 1e-9:
        raise RuntimeError(f"Forward-model guard failed at start: {guard_start}")

    profiles = profile_defs()
    clean_records, severe_records, sample_meta, cov_notes, cov_variables = build_curve_copies(root)
    copy_location = "in-memory Python CurveRecord/SevereRecord lists built from migrated E/F/G resolver; no scratch curve files written"
    print(f"Clean copy built: {len(clean_records)} clean curves; {len(severe_records)} severe field curves")

    baseline_profile = profiles[0]
    baseline_rows, baseline_predictions, baseline_gate2 = evaluate_profile(clean_records, baseline_profile)
    baseline_severe = evaluate_severe(severe_records, baseline_profile)
    assert_baseline_matches(root, baseline_rows, baseline_severe)

    sweep_rows: list[dict[str, Any]] = []
    all_predictions: dict[str, list[dict[str, Any]]] = {"baseline": baseline_predictions}
    sweep_rows.extend(baseline_rows)

    combined_profile = next(profile for profile in profiles if profile["profile_id"] == "combined_budget_realistic")
    combined_rows: list[dict[str, Any]] = []
    combined_predictions: list[dict[str, Any]] = []
    combined_gate2: dict[str, Any] = {}

    for profile in profiles[1:]:
        rows, predictions, gate2 = evaluate_profile(clean_records, profile)
        add_delta_fields(rows, baseline_rows)
        sweep_rows.extend(rows)
        if profile["profile_id"] in {"combined_budget_realistic"}:
            combined_rows = rows
            combined_predictions = predictions
            combined_gate2 = gate2
        if profile["profile_id"] in {"baseline", "combined_budget_realistic"}:
            all_predictions[profile["profile_id"]] = predictions

    add_delta_fields(baseline_rows, baseline_rows)
    add_delta_fields(combined_rows, baseline_rows)
    combined_severe = evaluate_severe(severe_records, combined_profile)
    external_baseline = summarize_external(baseline_predictions, baseline_profile)
    external_combined = summarize_external(combined_predictions, combined_profile)
    sweep_rows.extend(external_baseline)
    sweep_rows.extend(external_combined)

    break_rows = breaking_points(sweep_rows)
    sensitivity = sensitivity_summary(sweep_rows, baseline_rows)
    verdict = decide_verdict(combined_rows, break_rows)

    guard_end = forward_model_guard(SEED + 1)
    if not guard_end["passed"] or float(guard_end["max_abs_volume_ml"]) >= 1e-9:
        raise RuntimeError(f"Forward-model guard failed at end: {guard_end}")

    result = {
        "seed": SEED,
        "bootstrap_n": BOOTSTRAP_B,
        "elapsed_s": float(time.perf_counter() - started),
        "forward_model_guards": [{"phase": "start", **guard_start}, {"phase": "end", **guard_end}],
        "copy_location": copy_location,
        "data_safety": "NHANES_SpiroSBI is read-only for this gate; degradation is applied only to copied parsed arrays.",
        "cohort": {
            "clean_curve_n": int(len(clean_records)),
            "clean_obstructed_n": int(sum(rec.true_ratio < OBSTRUCTION_THRESHOLD for rec in clean_records)),
            "severe_field_curve_n": int(len(severe_records)),
            "sample_meta": sample_meta,
        },
        "degradation_profiles": profiles,
        "model_basis": model_basis_rows(),
        "baseline": {
            "profile": baseline_profile,
            "gate2_accuracy": baseline_gate2,
            "truncation_rows": baseline_rows,
            "severe_rows": baseline_severe,
            "external_rows": external_baseline,
        },
        "combined": {
            "profile": combined_profile,
            "gate2_accuracy": combined_gate2,
            "truncation_rows": combined_rows,
            "severe_rows": combined_severe,
            "external_rows": external_combined,
        },
        "sweep_rows": sweep_rows,
        "breaking_points": break_rows,
        "sensitivity": sensitivity,
        "covariate_notes": cov_notes,
        "covariate_variables": cov_variables,
        "verdict": verdict,
        "protocol_decisions": [
            "Gate 2 degradation is measured against the clean parsed curve copy, not NHANES summary truth, because the task defines clean parsed curves as the ground truth for degradation.",
            "Only obstructed clean-copy curves are re-fit for the truncation FN sweep because the requested headline is obstructed FN and DIRECT-minus-OMBE FN advantage; Gate 2 measurement error is still computed on all clean curves.",
            "The existing SciPy bounded-TRF OMBE loop fit is called unchanged via fit_ombe_truncated; no NUTS and no vectorized solver path is used.",
            "Gaussian flow noise is expressed as a fraction of each curve's clean peak expiratory flow and uses deterministic per-curve seeds.",
            "Coarser sampling is simulated by scipy.signal.resample_poly anti-alias/resampling on flow, followed by interpolation back to the original time grid so the downstream solver interface stays unchanged.",
            "Pure 100 Hz sampling and pure gain profiles bypass flow reconstruction; gain-only profiles multiply the copied volume curve directly as specified.",
            "Volume is forced monotone after flow degradation by cumulative integration plus maximum-accumulate; this keeps degraded copied curves physically usable for the existing volume-fit solver.",
            "Baseline drift is modeled as a deterministic-sign, low-frequency half-sine flow offset whose net integrated volume equals the requested FVC fraction.",
            "The combined budget profile uses +2% gain as one representative calibration sign; the D3 independent sweep reports both signs through +/-10%.",
            "Severe field recovery is evaluated only for the combined profile and clean baseline, matching Part 3; single-degradation sweeps focus on pooled clean T=2/T=3 headlines.",
        ],
    }

    out = root / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(sweep_rows).to_csv(out / RESULT_SWEEP, index=False)
    (out / RESULT_JSON).write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    write_report(root, result, sweep_rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    print(f"VERDICT: {result['verdict']}")
    for row in result["breaking_points"]:
        print(f"{row['operator']}: {row['breaking_level']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
