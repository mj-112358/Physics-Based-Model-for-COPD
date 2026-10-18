from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .gate3_diagnosis import clinical_metrics_from_volume, np_ombe_volume, savgol_sigma
from .gate23_f import find_or_unzip_raw, read_raw
from .gate_clinical_classification_f import extract_curve_for_fit
from .gate_likelihood_repair_f import NUTS_GRID_STRIDE
from .ombe import OmbeGrid, ombe_volume, phi_from_natural
from .tsnpe import features_from_phi

PARAM_NAMES = ("tau_f", "tau_s", "alpha", "fvc_l", "tau_o")
BOUNDS_LO = np.array([0.15, 1.00, 0.25, 2.00, 0.03], dtype=float)
BOUNDS_HI = np.array([0.80, 5.00, 0.85, 6.00, 0.40], dtype=float)
EXPECTED_MEDIAN_GAP = 0.0197 - 0.0019


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if np.isfinite(out) else None
    return value


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(clean_json(payload), indent=2) + "\n", encoding="utf-8")


def q(values: pd.Series | np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return [float("nan"), float("nan")]
    return [float(x) for x in np.percentile(arr, [25, 75])]


def phi_from_natural_np(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    alpha = np.clip(theta[2], 1e-8, 1.0 - 1e-8)
    return np.array(
        [
            np.log(theta[0]),
            np.log(theta[1]),
            np.log(alpha / (1.0 - alpha)),
            np.log(theta[3]),
            np.log(theta[4]),
        ],
        dtype=float,
    )


def jax_volume_from_natural(theta: np.ndarray, t: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    phi = phi_from_natural(
        jnp.asarray(theta[0], dtype=jnp.float64),
        jnp.asarray(theta[1], dtype=jnp.float64),
        jnp.asarray(np.clip(theta[2], 1e-8, 1.0 - 1e-8), dtype=jnp.float64),
        jnp.asarray(theta[3], dtype=jnp.float64),
        jnp.asarray(theta[4], dtype=jnp.float64),
    )
    return np.asarray(ombe_volume(phi, jnp.asarray(t, dtype=jnp.float64)), dtype=float)


def metrics_from_volume(t: np.ndarray, volume_l: np.ndarray) -> dict[str, float]:
    fev1_l, fvc_l, ratio = clinical_metrics_from_volume(np.asarray(t, dtype=float), np.asarray(volume_l, dtype=float))
    return {"fev1_l": fev1_l, "fvc_l": fvc_l, "ratio": ratio}


def likelihood_tsnpe_readout(theta: np.ndarray, t_full: np.ndarray) -> dict[str, float]:
    phi = phi_from_natural_np(theta)[None, :].astype(np.float32)
    features = features_from_phi(phi, t_full.astype(np.float32))
    fev1_l = float(features["fev1_l"][0])
    ratio = float(features["ratio"][0])
    fvc_l = fev1_l / ratio if ratio > 0 else np.nan
    return {"fev1_l": fev1_l, "fvc_l": fvc_l, "ratio": ratio}


def point_readout(theta: np.ndarray, t: np.ndarray) -> dict[str, float]:
    return metrics_from_volume(t, jax_volume_from_natural(theta, t))


def koh_readout(theta: np.ndarray, t: np.ndarray) -> dict[str, float]:
    t_down = np.asarray(t, dtype=float)[::NUTS_GRID_STRIDE]
    return metrics_from_volume(t_down, jax_volume_from_natural(theta, t_down))


def theta_from_row(row: Any) -> np.ndarray:
    return np.array(
        [
            float(row.fit_tau_f),
            float(row.fit_tau_s),
            float(row.fit_alpha),
            float(row.fit_fvc_l),
            float(row.fit_tau_o),
        ],
        dtype=float,
    )


def select_sample(root: Path, n: int = 20) -> pd.DataFrame:
    path = root / "results" / "gate_clinical_classification_F_rows.csv"
    if not path.exists():
        raise RuntimeError(f"Missing clinical rows: {path}")
    rows = pd.read_csv(path)
    needed = {
        "raw_index",
        "sample_group",
        "model_ratio",
        "fev1_fvc_ratio",
        "fit_tau_f",
        "fit_tau_s",
        "fit_alpha",
        "fit_fvc_l",
        "fit_tau_o",
        "fit_error",
    }
    missing = sorted(needed - set(rows.columns))
    if missing:
        raise RuntimeError(f"Clinical rows missing required columns: {missing}")
    valid = rows[
        (rows["sample_group"] == "obstructed")
        & (rows["fit_error"].fillna("") == "")
        & rows["model_ratio"].notna()
        & rows["fev1_fvc_ratio"].notna()
    ].copy()
    valid["numpy_signed_ratio_error"] = valid["model_ratio"] - valid["fev1_fvc_ratio"]
    valid["abs_numpy_signed_ratio_error"] = valid["numpy_signed_ratio_error"].abs()
    return valid.sort_values("abs_numpy_signed_ratio_error", ascending=False).head(n).reset_index(drop=True)


def fit_jax_lm(t: np.ndarray, volume_l: np.ndarray, x0_override: np.ndarray | None = None) -> dict[str, Any]:
    fvc_est = float(np.clip(volume_l[-1], 2.0, 6.0))
    x0 = np.array([0.35, 2.50, 0.60, fvc_est, 0.12], dtype=float) if x0_override is None else np.asarray(x0_override, dtype=float)
    sigma_l = savgol_sigma(volume_l)
    scale = sigma_l if np.isfinite(sigma_l) and sigma_l > 0 else 1.0

    def residual(x: np.ndarray) -> np.ndarray:
        return (jax_volume_from_natural(x, t) - volume_l) / scale

    opt = least_squares(residual, x0=x0, bounds=(BOUNDS_LO, BOUNDS_HI), method="trf", jac="3-point", max_nfev=1000)
    pred = jax_volume_from_natural(opt.x, t)
    fev1_l, fvc_l, ratio = clinical_metrics_from_volume(t, pred)
    return {
        "theta": opt.x,
        "success": bool(opt.success),
        "cost": float(opt.cost),
        "nfev": int(opt.nfev),
        "model_fev1_l": float(fev1_l),
        "model_fvc_l": float(fvc_l),
        "model_ratio": float(ratio),
        "bound_hits": {
            name: bool(abs(opt.x[i] - BOUNDS_LO[i]) <= 1e-5 or abs(opt.x[i] - BOUNDS_HI[i]) <= 1e-5)
            for i, name in enumerate(PARAM_NAMES)
        },
    }


def summarize_numeric(rows: list[dict], fields: list[str]) -> dict[str, Any]:
    out = {}
    df = pd.DataFrame(rows)
    for field in fields:
        if field not in df:
            continue
        arr = pd.to_numeric(df[field], errors="coerce").to_numpy(float)
        arr = arr[np.isfinite(arr)]
        out[field] = {
            "n": int(len(arr)),
            "median": float(np.median(arr)) if len(arr) else np.nan,
            "iqr": q(arr),
            "max": float(np.max(arr)) if len(arr) else np.nan,
        }
    return out


def per_curve_analysis(sample: pd.DataFrame, raw: pd.DataFrame) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    t_full = np.asarray(OmbeGrid().t, dtype=np.float32)
    raw_by_index = {int(idx): row for idx, row in raw.iterrows()}
    step1_rows: list[dict] = []
    step2_rows: list[dict] = []
    step3_rows: list[dict] = []
    step4_rows: list[dict] = []

    for i, row in enumerate(sample.itertuples(index=False), start=1):
        raw_idx = int(row.raw_index)
        curve = extract_curve_for_fit(raw_by_index[raw_idx])
        t = np.asarray(curve["vol_t"], dtype=float)
        y = np.asarray(curve["vol_l"], dtype=float)
        theta_np = theta_from_row(row)

        vol_np = np_ombe_volume(theta_np, t)
        vol_jax = jax_volume_from_natural(theta_np, t)
        np_m = metrics_from_volume(t, vol_np)
        jax_m = metrics_from_volume(t, vol_jax)
        diff = np.asarray(vol_jax - vol_np, dtype=float)
        max_i = int(np.nanargmax(np.abs(diff)))
        step1_rows.append(
            {
                "SEQN": float(row.SEQN),
                "raw_index": raw_idx,
                "n_points": int(len(t)),
                "duration_s": float(t[-1]),
                "truth_ratio": float(row.fev1_fvc_ratio),
                "numpy_model_ratio": float(row.model_ratio),
                "numpy_signed_ratio_error": float(row.numpy_signed_ratio_error),
                "max_abs_volume_diff_ml": float(np.nanmax(np.abs(diff)) * 1000.0),
                "max_diff_time_s": float(t[max_i]),
                "max_signed_volume_diff_ml": float(diff[max_i] * 1000.0),
                "fev1_diff_ml_jax_minus_np": float((jax_m["fev1_l"] - np_m["fev1_l"]) * 1000.0),
                "fvc_diff_ml_jax_minus_np": float((jax_m["fvc_l"] - np_m["fvc_l"]) * 1000.0),
                "ratio_diff_jax_minus_np": float(jax_m["ratio"] - np_m["ratio"]),
            }
        )

        point = jax_m
        likelihood = likelihood_tsnpe_readout(theta_np, t_full)
        koh = koh_readout(theta_np, t)
        step2_rows.append(
            {
                "SEQN": float(row.SEQN),
                "raw_index": raw_idx,
                "truth_ratio": float(row.fev1_fvc_ratio),
                "point_fev1_l": point["fev1_l"],
                "point_fvc_l": point["fvc_l"],
                "point_ratio": point["ratio"],
                "likelihood_tsnpe_fev1_l": likelihood["fev1_l"],
                "likelihood_tsnpe_fvc_l": likelihood["fvc_l"],
                "likelihood_tsnpe_ratio": likelihood["ratio"],
                "likelihood_ratio_diff_vs_point": float(likelihood["ratio"] - point["ratio"]),
                "likelihood_fev1_diff_ml_vs_point": float((likelihood["fev1_l"] - point["fev1_l"]) * 1000.0),
                "likelihood_fvc_diff_ml_vs_point": float((likelihood["fvc_l"] - point["fvc_l"]) * 1000.0),
                "koh_downsampled_fev1_l": koh["fev1_l"],
                "koh_downsampled_fvc_l": koh["fvc_l"],
                "koh_downsampled_ratio": koh["ratio"],
                "koh_ratio_diff_vs_point": float(koh["ratio"] - point["ratio"]),
                "koh_fev1_diff_ml_vs_point": float((koh["fev1_l"] - point["fev1_l"]) * 1000.0),
                "koh_fvc_diff_ml_vs_point": float((koh["fvc_l"] - point["fvc_l"]) * 1000.0),
            }
        )

        t_stride = t[::NUTS_GRID_STRIDE]
        y_stride = y[::NUTS_GRID_STRIDE]
        fit_jax = fit_jax_lm(t, y)
        fit_jax_stride = fit_jax_lm(t_stride, y_stride)
        fit_jax_stride_from_np = fit_jax_lm(t_stride, y_stride, x0_override=theta_np)
        theta_jax = np.asarray(fit_jax["theta"], dtype=float)
        theta_stride = np.asarray(fit_jax_stride["theta"], dtype=float)
        theta_stride_from_np = np.asarray(fit_jax_stride_from_np["theta"], dtype=float)
        point_jax_fit = point_readout(theta_jax, t)
        stride_jax_fit_native_readout = point_readout(theta_stride, t)
        stride_from_np_native_readout = point_readout(theta_stride_from_np, t)
        theta_diff = theta_jax - theta_np
        theta_rel = 100.0 * theta_diff / np.where(np.abs(theta_np) > 1e-12, theta_np, np.nan)
        theta_stride_diff = theta_stride - theta_np
        theta_stride_rel = 100.0 * theta_stride_diff / np.where(np.abs(theta_np) > 1e-12, theta_np, np.nan)
        theta_stride_from_np_diff = theta_stride_from_np - theta_np
        theta_stride_from_np_rel = 100.0 * theta_stride_from_np_diff / np.where(np.abs(theta_np) > 1e-12, theta_np, np.nan)
        step3 = {
            "SEQN": float(row.SEQN),
            "raw_index": raw_idx,
            "jax_native_lm_success": bool(fit_jax["success"]),
            "jax_native_lm_nfev": int(fit_jax["nfev"]),
            "jax_stride_lm_success": bool(fit_jax_stride["success"]),
            "jax_stride_lm_nfev": int(fit_jax_stride["nfev"]),
            "jax_stride_from_numpy_lm_success": bool(fit_jax_stride_from_np["success"]),
            "jax_stride_from_numpy_lm_nfev": int(fit_jax_stride_from_np["nfev"]),
            "numpy_ratio_from_saved_theta": point["ratio"],
            "jax_native_lm_ratio": point_jax_fit["ratio"],
            "jax_stride_lm_ratio_native_readout": stride_jax_fit_native_readout["ratio"],
            "jax_stride_from_numpy_lm_ratio_native_readout": stride_from_np_native_readout["ratio"],
            "ratio_diff_jax_native_lm_minus_numpy_lm": float(point_jax_fit["ratio"] - point["ratio"]),
            "ratio_diff_jax_stride_lm_minus_numpy_lm": float(stride_jax_fit_native_readout["ratio"] - point["ratio"]),
            "ratio_diff_jax_stride_from_numpy_lm_minus_numpy_lm": float(stride_from_np_native_readout["ratio"] - point["ratio"]),
            "max_abs_native_param_rel_diff_pct": float(np.nanmax(np.abs(theta_rel))),
            "max_abs_stride_param_rel_diff_pct": float(np.nanmax(np.abs(theta_stride_rel))),
            "max_abs_stride_from_numpy_param_rel_diff_pct": float(np.nanmax(np.abs(theta_stride_from_np_rel))),
        }
        for j, name in enumerate(PARAM_NAMES):
            step3[f"numpy_lm_{name}"] = float(theta_np[j])
            step3[f"jax_native_lm_{name}"] = float(theta_jax[j])
            step3[f"jax_stride_lm_{name}"] = float(theta_stride[j])
            step3[f"jax_stride_from_numpy_lm_{name}"] = float(theta_stride_from_np[j])
            step3[f"native_{name}_rel_diff_pct"] = float(theta_rel[j])
            step3[f"stride_{name}_rel_diff_pct"] = float(theta_stride_rel[j])
            step3[f"stride_from_numpy_{name}_rel_diff_pct"] = float(theta_stride_from_np_rel[j])
            step3[f"numpy_lm_{name}_at_bound"] = bool(abs(theta_np[j] - BOUNDS_LO[j]) <= 1e-5 or abs(theta_np[j] - BOUNDS_HI[j]) <= 1e-5)
            step3[f"jax_native_lm_{name}_at_bound"] = bool(fit_jax["bound_hits"][name])
            step3[f"jax_stride_lm_{name}_at_bound"] = bool(fit_jax_stride["bound_hits"][name])
            step3[f"jax_stride_from_numpy_lm_{name}_at_bound"] = bool(fit_jax_stride_from_np["bound_hits"][name])
        step3_rows.append(step3)

        stride_np = metrics_from_volume(t_stride, np_ombe_volume(theta_np, t_stride))
        step4_rows.append(
            {
                "SEQN": float(row.SEQN),
                "raw_index": raw_idx,
                "native_n": int(len(t)),
                "stride_n": int(len(t_stride)),
                "native_dt_median_s": float(np.median(np.diff(t))),
                "stride_dt_median_s": float(np.median(np.diff(t_stride))),
                "native_ratio": np_m["ratio"],
                "stride_ratio": stride_np["ratio"],
                "ratio_diff_stride_minus_native": float(stride_np["ratio"] - np_m["ratio"]),
                "abs_stride_ratio_diff_fraction_of_0p0178_gap": float(abs(stride_np["ratio"] - np_m["ratio"]) / EXPECTED_MEDIAN_GAP),
                "native_fev1_l": np_m["fev1_l"],
                "stride_fev1_l": stride_np["fev1_l"],
                "fev1_diff_ml_stride_minus_native": float((stride_np["fev1_l"] - np_m["fev1_l"]) * 1000.0),
                "native_fvc_l": np_m["fvc_l"],
                "stride_fvc_l": stride_np["fvc_l"],
                "fvc_diff_ml_stride_minus_native": float((stride_np["fvc_l"] - np_m["fvc_l"]) * 1000.0),
            }
        )
        print(f"Two-path reconciliation progress: {i}/{len(sample)} raw_index={raw_idx}")
    return step1_rows, step2_rows, step3_rows, step4_rows


def gate_findings(step1: list[dict], step2: list[dict], step3: list[dict], step4: list[dict]) -> dict[str, Any]:
    s1 = pd.DataFrame(step1)
    s2 = pd.DataFrame(step2)
    s3 = pd.DataFrame(step3)
    s4 = pd.DataFrame(step4)
    integrator_bad = bool((s1["max_abs_volume_diff_ml"] > 5.0).any())
    likelihood_readout_bad = bool((s2["likelihood_ratio_diff_vs_point"].abs() > 0.002).any())
    koh_readout_bad = bool((s2["koh_ratio_diff_vs_point"].abs() > 0.002).any())
    native_fit_diff_bad = bool((s3["max_abs_native_param_rel_diff_pct"] > 2.0).any())
    stride_fit_diff_bad = bool((s3["max_abs_stride_param_rel_diff_pct"] > 2.0).any())
    stride_from_np_fit_diff_bad = bool((s3["max_abs_stride_from_numpy_param_rel_diff_pct"] > 2.0).any())
    stride_material = bool((s4["ratio_diff_stride_minus_native"].abs().median() > 0.002))

    if integrator_bad:
        verdict = "JAX-WRONG"
        responsible = "integrator mismatch on identical theta/time grid"
    elif stride_fit_diff_bad or stride_from_np_fit_diff_bad:
        verdict = "BOTH-CORRECT-DIFFERENT-READOUT"
        responsible = "posterior path uses a downsampled fitting grid/objective; deterministic stride-grid LM moves theta and ratios relative to native-grid point LM"
    elif likelihood_readout_bad or koh_readout_bad:
        verdict = "BOTH-CORRECT-DIFFERENT-READOUT"
        if likelihood_readout_bad and not koh_readout_bad:
            responsible = "likelihood/TSNPE posterior ratio readout uses a different grid/FVC convention than the point gate"
        elif koh_readout_bad and not likelihood_readout_bad:
            responsible = "KOH downsampled-grid readout differs from the point gate"
        else:
            responsible = "posterior readout conventions differ from the point gate; likelihood/TSNPE readout and KOH downsampled readout both move ratios"
    elif native_fit_diff_bad:
        verdict = "UNRESOLVED"
        responsible = "native-grid deterministic JAX least-squares did not reproduce saved NumPy LM theta despite integrator equivalence"
    elif stride_material:
        verdict = "BOTH-CORRECT-DIFFERENT-READOUT"
        responsible = "grid_stride=5 alone materially changes the point ratio"
    else:
        verdict = "UNRESOLVED"
        responsible = "Steps 1-4 did not explain the reported 0.0197 vs 0.0019 gap"

    point_bias = float(s2["point_ratio"].sub(s2["truth_ratio"]).median())
    likelihood_bias = float(s2["likelihood_tsnpe_ratio"].sub(s2["truth_ratio"]).median())
    koh_bias = float(s2["koh_downsampled_ratio"].sub(s2["truth_ratio"]).median())
    stride_bias = float(s4["stride_ratio"].sub(s2["truth_ratio"]).median())
    c1_answer = (
        "The c1 correction is a convention/readout patch for the point-estimate clinical convention, not evidence that the OMBE integrator itself is wrong."
        if verdict == "BOTH-CORRECT-DIFFERENT-READOUT"
        else "c1 interpretation depends on the unresolved or failed diagnostic above."
    )
    tail_answer = (
        "The tail-undershoot mechanism survives only for the point-gate native-grid clinical FVC convention; it is not supported as a universal OMBE/JAX posterior mechanism."
        if verdict == "BOTH-CORRECT-DIFFERENT-READOUT"
        else "tail-undershoot mechanism is not adjudicated because the primary diagnostic did not isolate a readout convention."
    )
    return {
        "verdict": verdict,
        "responsible_difference": responsible,
        "gates": {
            "step1_integrator_mismatch_gt_5ml": integrator_bad,
            "step2_likelihood_readout_ratio_diff_gt_0p002": likelihood_readout_bad,
            "step2_koh_readout_ratio_diff_gt_0p002": koh_readout_bad,
            "step3_native_theta_diff_gt_2pct": native_fit_diff_bad,
            "step3_stride_theta_diff_gt_2pct": stride_fit_diff_bad,
            "step3_stride_from_numpy_theta_diff_gt_2pct": stride_from_np_fit_diff_bad,
            "step4_stride_median_diff_gt_0p002": stride_material,
        },
        "bias_medians_on_20_curve_sample": {
            "point_native_signed_ratio": point_bias,
            "likelihood_tsnpe_signed_ratio": likelihood_bias,
            "koh_downsampled_signed_ratio": koh_bias,
            "numpy_stride_signed_ratio": stride_bias,
        },
        "direct_answers": {
            "is_c1_real_physical_correction_or_patch": c1_answer,
            "does_tail_undershoot_survive": tail_answer,
        },
    }


def fmt(value: Any, digits: int = 4) -> str:
    try:
        x = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def table(rows: list[dict], columns: list[tuple[str, str, int]]) -> str:
    out = ["| " + " | ".join(label for _, label, _ in columns) + " |"]
    out.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        cells = []
        for key, _, digits in columns:
            value = row.get(key)
            if isinstance(value, (int, np.integer)) or digits < 0:
                cells.append(str(value))
            elif isinstance(value, (float, np.floating)):
                cells.append(fmt(value, digits))
            else:
                cells.append(str(value))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def make_report(result: dict) -> str:
    s1 = result["step1_forward_equivalence"]
    s2 = result["step2_readout_equivalence"]
    s3 = result["step3_fit_objective_equivalence"]
    s4 = result["step4_grid_stride_check"]
    findings = result["findings"]
    lines = [
        "# Gate - Two-Path Reconciliation, Cycle F",
        "",
        f"- VERDICT: {findings['verdict']}",
        f"- responsible difference: {findings['responsible_difference']}",
        "- no NUTS, no sampling, no downstream classification rerun",
        "- sample: 20 obstructed Cycle-F selected manoeuvres with largest absolute signed ratio error under the NumPy point path",
        "",
        "## Sample",
        "",
        table(
            result["sample_rows"],
            [
                ("SEQN", "SEQN", 0),
                ("raw_index", "raw_index", -1),
                ("truth_ratio", "truth ratio", 4),
                ("numpy_model_ratio", "NumPy model ratio", 4),
                ("numpy_signed_ratio_error", "NumPy signed error", 4),
            ],
        ),
        "",
        "## Step 1 - Forward-Evaluation Equivalence",
        "",
        "Fixed theta is the saved NumPy LM theta from `gate_clinical_classification_F_rows.csv`. NumPy path is `gate3_diagnosis.py::np_ombe_volume(theta, t)`. JAX path is `ombe.py::ombe_volume(phi, t)` after the `jnp.diff(t)` fix.",
        "",
        table(
            s1,
            [
                ("raw_index", "raw_index", -1),
                ("n_points", "n", -1),
                ("duration_s", "duration s", 3),
                ("max_abs_volume_diff_ml", "max |dV| mL", 3),
                ("max_diff_time_s", "max diff t", 3),
                ("fev1_diff_ml_jax_minus_np", "dFEV1 mL", 3),
                ("fvc_diff_ml_jax_minus_np", "dFVC mL", 3),
                ("ratio_diff_jax_minus_np", "dRatio", 6),
            ],
        ),
        "",
        f"Gate result: max volume difference >5 mL on any curve = {findings['gates']['step1_integrator_mismatch_gt_5ml']}.",
        "",
        "## Step 2 - Readout Equivalence",
        "",
        "Point convention: `extract_curve_for_fit` applies segmentation/back-extrapolated t0, builds `vol_t`, then `clinical_metrics_from_volume` reads FEV1 by `np.interp(min(1.0, t[-1]), t, volume)`, FVC as `volume[-1]`, and ratio as FEV1/FVC.",
        "",
        "Likelihood/TSNPE convention: likelihood gates evaluate posterior ratios on `OmbeGrid().t` through `gate_posterior_calibration_repair_f.ratio_samples_from_phi -> tsnpe.features_from_phi`, which uses a uniform-grid feature readout and returns FVC as the natural `fvc_l` parameter.",
        "",
        "KOH convention: KOH fits/readouts use each curve's `vol_t[::grid_stride]` with `grid_stride=5`; ratio is FEV1 by interpolation on that downsampled grid divided by predicted volume at that grid's final sample.",
        "",
        table(
            s2,
            [
                ("raw_index", "raw_index", -1),
                ("truth_ratio", "truth", 4),
                ("point_ratio", "point ratio", 4),
                ("likelihood_tsnpe_ratio", "likelihood ratio", 4),
                ("likelihood_ratio_diff_vs_point", "likelihood dRatio", 4),
                ("koh_downsampled_ratio", "KOH ratio", 4),
                ("koh_ratio_diff_vs_point", "KOH dRatio", 4),
                ("likelihood_fev1_diff_ml_vs_point", "likelihood dFEV1 mL", 2),
                ("likelihood_fvc_diff_ml_vs_point", "likelihood dFVC mL", 2),
            ],
        ),
        "",
        f"Gate result: likelihood ratio differs by >0.002 on any curve = {findings['gates']['step2_likelihood_readout_ratio_diff_gt_0p002']}; KOH ratio differs by >0.002 on any curve = {findings['gates']['step2_koh_readout_ratio_diff_gt_0p002']}.",
        "",
        "## Step 3 - Fit-Objective Equivalence",
        "",
        "JAX LM refit uses the same natural-parameter bounds and the same deterministic least-squares objective as the NumPy LM, but evaluates volume through corrected `ombe_volume`.",
        "",
        table(
            s3,
            [
                ("raw_index", "raw_index", -1),
                ("jax_native_lm_success", "native success", -1),
                ("max_abs_native_param_rel_diff_pct", "native max |d%|", 4),
                ("ratio_diff_jax_native_lm_minus_numpy_lm", "native dRatio", 6),
                ("jax_stride_lm_success", "stride success", -1),
                ("max_abs_stride_param_rel_diff_pct", "stride max |d%|", 4),
                ("ratio_diff_jax_stride_lm_minus_numpy_lm", "stride dRatio", 6),
                ("max_abs_stride_from_numpy_param_rel_diff_pct", "stride-from-NumPy max |d%|", 4),
                ("ratio_diff_jax_stride_from_numpy_lm_minus_numpy_lm", "stride-from-NumPy dRatio", 6),
                ("numpy_lm_tau_o_at_bound", "NumPy tau_o bound", -1),
                ("jax_native_lm_tau_o_at_bound", "native tau_o bound", -1),
                ("jax_stride_lm_tau_o_at_bound", "stride tau_o bound", -1),
            ],
        ),
        "",
        "### Step 3 Per-Parameter Relative Differences",
        "",
        table(
            s3,
            [
                ("raw_index", "raw_index", -1),
                ("native_tau_f_rel_diff_pct", "native tau_f d%", 2),
                ("native_tau_s_rel_diff_pct", "native tau_s d%", 2),
                ("native_alpha_rel_diff_pct", "native alpha d%", 2),
                ("native_fvc_l_rel_diff_pct", "native fvc d%", 2),
                ("native_tau_o_rel_diff_pct", "native tau_o d%", 2),
                ("stride_from_numpy_tau_f_rel_diff_pct", "stride tau_f d%", 2),
                ("stride_from_numpy_tau_s_rel_diff_pct", "stride tau_s d%", 2),
                ("stride_from_numpy_alpha_rel_diff_pct", "stride alpha d%", 2),
                ("stride_from_numpy_fvc_l_rel_diff_pct", "stride fvc d%", 2),
                ("stride_from_numpy_tau_o_rel_diff_pct", "stride tau_o d%", 2),
            ],
        ),
        "",
        f"Gate result: native theta differs by >2% on any curve = {findings['gates']['step3_native_theta_diff_gt_2pct']}; stride theta differs by >2% on any curve = {findings['gates']['step3_stride_theta_diff_gt_2pct']}; stride initialized from NumPy theta differs by >2% = {findings['gates']['step3_stride_from_numpy_theta_diff_gt_2pct']}.",
        "",
        "## Step 4 - Grid-Stride Check",
        "",
        table(
            s4,
            [
                ("raw_index", "raw_index", -1),
                ("native_n", "native n", -1),
                ("stride_n", "stride n", -1),
                ("native_ratio", "native ratio", 4),
                ("stride_ratio", "stride ratio", 4),
                ("ratio_diff_stride_minus_native", "stride dRatio", 5),
                ("abs_stride_ratio_diff_fraction_of_0p0178_gap", "fraction of 0.0178 gap", 3),
                ("fev1_diff_ml_stride_minus_native", "dFEV1 mL", 2),
                ("fvc_diff_ml_stride_minus_native", "dFVC mL", 2),
            ],
        ),
        "",
        f"Gate result: median absolute stride-only ratio difference >0.002 = {findings['gates']['step4_stride_median_diff_gt_0p002']}.",
        "",
        "## Summary Statistics",
        "",
        "```json",
        json.dumps(clean_json(result["summaries"]), indent=2),
        "```",
        "",
        "## Step 5 - Verdict and Consequence",
        "",
        f"Verdict: **{findings['verdict']}**.",
        "",
        findings["responsible_difference"],
        "",
        f"Bias medians on the 20-curve stress sample: {findings['bias_medians_on_20_curve_sample']}",
        "",
        f"Direct answer - c1: {findings['direct_answers']['is_c1_real_physical_correction_or_patch']}",
        "",
        f"Direct answer - tail undershoot: {findings['direct_answers']['does_tail_undershoot_survive']}",
        "",
    ]
    return "\n".join(lines)


def run(root: Path) -> dict:
    sample = select_sample(root, n=20)
    raw_path, raw_note = find_or_unzip_raw(root)
    raw, raw_info = read_raw(raw_path)
    step1, step2, step3, step4 = per_curve_analysis(sample, raw)
    findings = gate_findings(step1, step2, step3, step4)
    sample_rows = [
        {
            "SEQN": float(r.SEQN),
            "raw_index": int(r.raw_index),
            "truth_ratio": float(r.fev1_fvc_ratio),
            "numpy_model_ratio": float(r.model_ratio),
            "numpy_signed_ratio_error": float(r.numpy_signed_ratio_error),
        }
        for r in sample.itertuples(index=False)
    ]
    result = {
        "raw_source": str(raw_path),
        "raw_source_note": raw_note,
        "raw_loaded_rows": int(raw_info["loaded_rows"]),
        "sample_rows": sample_rows,
        "step1_forward_equivalence": step1,
        "step2_readout_equivalence": step2,
        "step3_fit_objective_equivalence": step3,
        "step4_grid_stride_check": step4,
        "summaries": {
            "step1": summarize_numeric(step1, ["max_abs_volume_diff_ml", "fev1_diff_ml_jax_minus_np", "fvc_diff_ml_jax_minus_np", "ratio_diff_jax_minus_np"]),
            "step2": summarize_numeric(
                step2,
                [
                    "likelihood_ratio_diff_vs_point",
                    "koh_ratio_diff_vs_point",
                    "likelihood_fev1_diff_ml_vs_point",
                    "likelihood_fvc_diff_ml_vs_point",
                    "koh_fev1_diff_ml_vs_point",
                    "koh_fvc_diff_ml_vs_point",
                ],
            ),
            "step3": summarize_numeric(
                step3,
                [
                    "max_abs_native_param_rel_diff_pct",
                    "ratio_diff_jax_native_lm_minus_numpy_lm",
                    "max_abs_stride_param_rel_diff_pct",
                    "ratio_diff_jax_stride_lm_minus_numpy_lm",
                    "max_abs_stride_from_numpy_param_rel_diff_pct",
                    "ratio_diff_jax_stride_from_numpy_lm_minus_numpy_lm",
                ],
            ),
            "step4": summarize_numeric(step4, ["ratio_diff_stride_minus_native", "abs_stride_ratio_diff_fraction_of_0p0178_gap", "fev1_diff_ml_stride_minus_native", "fvc_diff_ml_stride_minus_native"]),
        },
        "findings": findings,
    }
    out = root / "results"
    out.mkdir(exist_ok=True)
    write_json(out / "gate_two_path_reconciliation_F.json", result)
    (out / "gate_two_path_reconciliation_F.md").write_text(make_report(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    findings = result["findings"]
    print(f"Two-path reconciliation: {findings['verdict']} - {findings['responsible_difference']}")
    print(f"Bias medians: {findings['bias_medians_on_20_curve_sample']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
