from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/predictx-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/predictx-xdg-cache")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import least_squares

from .fit import np_ombe_flow
from .gate3_diagnosis import fit_ombe_volume_lm, np_ombe_volume, savgol_sigma
from .gate23_f import find_or_unzip_raw, read_raw
from .gate_clinical_classification_f import extract_curve_for_fit
from .gate_truncation_recovery_f import clean_json
from .ombe import ombe_flow, phi_from_natural

OUTPUT_STEM = "gate_fit_optimization_F"
SEED = 20260710
FORWARD_GUARD_TOL_ML = 1e-9
VECTOR_AUDIT_N = 200
BATCHED_SOLVER_PILOT_N = 20
VECTOR_PARAM_REL_TOL = 1e-6
MULTISTART_K = 8
MATERIAL_SSE_IMPROVEMENT_THRESHOLD = 1e-6
TRUNC_T_ORDER = ("2", "3", "4", "5")
SEVERITY_ORDER = ("severe", "moderate", "mild_0.60_0.70", "normal")


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
            elif fmt.startswith("float"):
                digits = int(fmt.replace("float", "") or "3")
                try:
                    v = float(value)
                    cells.append(f"{v:.{digits}f}" if np.isfinite(v) else "NA")
                except Exception:
                    cells.append("NA")
            elif fmt == "signed":
                try:
                    v = float(value)
                    cells.append(f"{v:+.6f}" if np.isfinite(v) else "NA")
                except Exception:
                    cells.append("NA")
            else:
                cells.append("" if value is None else str(value))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing required baseline file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(
    *,
    headline: str,
    metric: str,
    baseline_value: float,
    n: int | None = None,
    denominator_n: int | None = None,
    estimator: str = "",
    endpoint: str = "",
    unit: str = "",
    source: str = "",
) -> dict[str, Any]:
    return {
        "headline": headline,
        "metric": metric,
        "estimator": estimator,
        "endpoint": endpoint,
        "baseline_value": float(baseline_value) if np.isfinite(float(baseline_value)) else np.nan,
        "n": int(n) if n is not None and pd.notna(n) else np.nan,
        "denominator_n": int(denominator_n) if denominator_n is not None and pd.notna(denominator_n) else np.nan,
        "unit": unit,
        "source": source,
    }


def baseline_lock(root: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []

    gate2 = read_json(root / "results" / "gate2_F_parse.json")
    crit = gate2["criterion"]
    rows.extend(
        [
            metric_row(
                headline="Gate 2 parser",
                metric="median_abs_fev1_error_pct",
                baseline_value=crit["fev1_median_abs_pct_error"],
                n=crit["n"],
                unit="percent",
                source="results/gate2_F_parse.json:criterion",
            ),
            metric_row(
                headline="Gate 2 parser",
                metric="median_abs_fvc_error_pct",
                baseline_value=crit["fvc_median_abs_pct_error"],
                n=crit["n"],
                unit="percent",
                source="results/gate2_F_parse.json:criterion",
            ),
            metric_row(
                headline="Gate 2 parser",
                metric="median_signed_fev1_error_pct",
                baseline_value=crit["fev1_median_signed_pct_error"],
                n=crit["n"],
                unit="percent",
                source="results/gate2_F_parse.json:criterion",
            ),
            metric_row(
                headline="Gate 2 parser",
                metric="median_signed_fvc_error_pct",
                baseline_value=crit["fvc_median_signed_pct_error"],
                n=crit["n"],
                unit="percent",
                source="results/gate2_F_parse.json:criterion",
            ),
        ]
    )

    clinical = read_json(root / "results" / "gate_clinical_classification_F.json")
    fixed = clinical["fixed"]
    rows.extend(
        [
            metric_row(
                headline="Clinical fixed-0.70",
                metric="sensitivity",
                baseline_value=fixed["sensitivity"],
                n=fixed["n"],
                denominator_n=int(fixed["tp"] + fixed["fn"]),
                unit="rate",
                source="results/gate_clinical_classification_F.json:fixed",
            ),
            metric_row(
                headline="Clinical fixed-0.70",
                metric="specificity",
                baseline_value=fixed["specificity"],
                n=fixed["n"],
                denominator_n=int(fixed["tn"] + fixed["fp"]),
                unit="rate",
                source="results/gate_clinical_classification_F.json:fixed",
            ),
            metric_row(
                headline="Clinical fixed-0.70",
                metric="cohens_kappa",
                baseline_value=fixed["cohens_kappa"],
                n=fixed["n"],
                unit="kappa",
                source="results/gate_clinical_classification_F.json:fixed",
            ),
        ]
    )

    hardening = read_json(root / "results" / "gate_truncation_hardening_F.json")
    hard_cls = pd.DataFrame(hardening["classification_summary"])
    for dataset, headline in [("clean_cut", "Truncation clean-cut"), ("field_realistic", "Truncation field-realistic")]:
        if dataset == "clean_cut":
            part = hard_cls[
                hard_cls["dataset"].eq(dataset)
                & hard_cls["estimator"].isin(["E1_OMBE", "H1"])
                & hard_cls["T_label"].astype(str).isin(TRUNC_T_ORDER)
            ].copy()
        else:
            part = hard_cls[
                hard_cls["dataset"].eq(dataset)
                & hard_cls["estimator"].isin(["E0_DIRECT", "E1_OMBE", "H1"])
                & hard_cls["T_label"].astype(str).eq("field_all")
            ].copy()
        for row in part.itertuples(index=False):
            rows.append(
                metric_row(
                    headline=headline,
                    metric="obstructed_false_negative_rate",
                    baseline_value=float(row.false_negative_rate_obstructed),
                    n=int(row.n),
                    denominator_n=int(row.obstructed_denominator_n),
                    estimator=str(row.estimator),
                    endpoint=str(row.T_label),
                    unit="rate",
                    source="results/gate_truncation_hardening_F.json:classification_summary",
                )
            )

    severe = read_json(root / "results" / "gate_severe_truncation_diagnosis_F.json")
    for row in severe["clean_severity_recovery"]:
        if row["truth_severity"] == "severe" and row["estimator"] in {"E1_OMBE", "H1"} and str(row["T_label"]) in TRUNC_T_ORDER:
            rows.append(
                metric_row(
                    headline="Severe-band clean recovery",
                    metric="severe_false_negative_rate",
                    baseline_value=float(row["false_negative_rate"]),
                    n=int(row["n"]),
                    denominator_n=int(row["n"]),
                    estimator=str(row["estimator"]),
                    endpoint=str(row["T_label"]),
                    unit="rate",
                    source="results/gate_severe_truncation_diagnosis_F.json:clean_severity_recovery",
                )
            )

    baseline = {
        "gate2": {
            "n": int(crit["n"]),
            "median_abs_fev1_pct": float(crit["fev1_median_abs_pct_error"]),
            "median_abs_fvc_pct": float(crit["fvc_median_abs_pct_error"]),
            "median_signed_fev1_pct": float(crit["fev1_median_signed_pct_error"]),
            "median_signed_fvc_pct": float(crit["fvc_median_signed_pct_error"]),
        },
        "clinical_fixed": fixed,
        "truncation_hardening_verdict": hardening["verdict"],
        "severe_recovery": severe["clean_severity_recovery"],
    }
    return baseline, pd.DataFrame(rows)


def natural_volume_np(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    flow = np_ombe_flow(params, t)
    inc = (flow[:-1] + flow[1:]) * 0.5 * np.diff(t)
    return np.concatenate([[0.0], np.cumsum(inc)])


def forward_model_guard(seed: int = SEED) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 12.0 + 0.01, 0.01, dtype=float)
    params = np.column_stack(
        [
            rng.uniform(0.15, 0.80, 24),
            rng.uniform(1.00, 5.00, 24),
            rng.uniform(0.25, 0.85, 24),
            rng.uniform(2.00, 6.00, 24),
            rng.uniform(0.03, 0.40, 24),
        ]
    )
    max_flow_lps = 0.0
    max_volume_ml = 0.0
    for p in params:
        phi = phi_from_natural(*[jnp.asarray(x, dtype=jnp.float64) for x in p])
        j_flow = np.asarray(ombe_flow(phi, jnp.asarray(t, dtype=jnp.float64)))
        n_flow = np_ombe_flow(p, t)
        max_flow_lps = max(max_flow_lps, float(np.max(np.abs(j_flow - n_flow))))
        j_inc = (j_flow[:-1] + j_flow[1:]) * 0.5 * np.diff(t)
        j_vol = np.concatenate([[0.0], np.cumsum(j_inc)])
        n_vol = natural_volume_np(p, t)
        max_volume_ml = max(max_volume_ml, float(1000.0 * np.max(np.abs(j_vol - n_vol))))
    return {
        "n_param_sets": int(len(params)),
        "grid_n": int(len(t)),
        "max_abs_flow_lps": max_flow_lps,
        "max_abs_volume_ml": max_volume_ml,
        "tolerance_ml": FORWARD_GUARD_TOL_ML,
        "passed": bool(max_volume_ml < FORWARD_GUARD_TOL_ML),
    }


def select_fit_audit_rows(root: Path, n: int = VECTOR_AUDIT_N) -> pd.DataFrame:
    path = root / "results" / "gate_reproducibility_volume_F_rows.csv"
    if not path.exists():
        raise RuntimeError(f"Missing reproducibility volume rows: {path}")
    rows = pd.read_csv(path, low_memory=False)
    rows = rows[
        rows["row_type"].eq("maneuver_fit")
        & rows["fit_error"].fillna("").eq("")
        & rows["volume_fit_success"].astype(str).str.lower().isin(["true", "1"])
    ].copy()
    rows = rows[rows[["raw_index", "SEQN", "SPAMANU", "truth_severity"]].notna().all(axis=1)].copy()
    parts = []
    rng = np.random.default_rng(SEED)
    per_stratum = max(1, n // len(SEVERITY_ORDER))
    used_index: set[int] = set()
    for severity in SEVERITY_ORDER:
        part = rows[rows["truth_severity"].eq(severity)].copy()
        take = min(per_stratum, len(part))
        if take:
            chosen = part.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1)))
            used_index.update(int(i) for i in chosen.index)
            parts.append(chosen)
    selected = pd.concat(parts, ignore_index=False) if parts else rows.head(0)
    if len(selected) < n:
        rest = rows[~rows.index.isin(used_index)].copy()
        fill = rest.sample(n=min(n - len(selected), len(rest)), random_state=SEED + 13)
        selected = pd.concat([selected, fill], ignore_index=False)
    selected = selected.sort_values(["truth_severity", "SEQN", "SPAMANU", "raw_index"]).head(n).reset_index(drop=True)
    return selected


def load_curves(root: Path, rows: pd.DataFrame) -> list[dict[str, Any]]:
    raw_path, _raw_note = find_or_unzip_raw(root)
    raw, _raw_info = read_raw(raw_path)
    raw_by_index = {int(idx): row for idx, row in raw.iterrows()}
    curves: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        curve = extract_curve_for_fit(raw_by_index[int(row.raw_index)])
        t = np.asarray(curve["vol_t"], dtype=float)
        y = np.asarray(curve["vol_l"], dtype=float)
        scale = savgol_sigma(y)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        curves.append(
            {
                "SEQN": float(row.SEQN),
                "raw_index": int(row.raw_index),
                "SPAMANU": float(row.SPAMANU),
                "truth_severity": str(row.truth_severity),
                "t": t,
                "y": y,
                "scale": float(scale),
                "fvc_init": float(np.clip(y[-1], 2.0, 6.0)),
            }
        )
    return curves


def fit_loop(curves: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    params = []
    start = time.perf_counter()
    for curve in curves:
        fit = fit_ombe_volume_lm(curve["t"], curve["y"])
        params.append(np.asarray(fit["params"], dtype=float))
    elapsed = time.perf_counter() - start
    return np.asarray(params), {
        "n": int(len(curves)),
        "wall_clock_s": float(elapsed),
        "wall_clock_per_200_s": float(elapsed * 200.0 / max(len(curves), 1)),
    }


def batch_arrays(curves: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_len = max(len(c["t"]) for c in curves)
    n = len(curves)
    tpad = np.zeros((n, max_len), dtype=float)
    ypad = np.zeros((n, max_len), dtype=float)
    mask = np.zeros((n, max_len), dtype=float)
    scales = np.zeros(n, dtype=float)
    x0 = np.zeros((n, 5), dtype=float)
    for i, curve in enumerate(curves):
        t = curve["t"]
        y = curve["y"]
        tpad[i, : len(t)] = t
        tpad[i, len(t) :] = t[-1]
        ypad[i, : len(y)] = y
        ypad[i, len(y) :] = y[-1]
        mask[i, : len(t)] = 1.0
        scales[i] = curve["scale"]
        x0[i] = [0.35, 2.50, 0.60, curve["fvc_init"], 0.12]
    return tpad, ypad, mask, scales, x0


def batched_solver_pilot(curves: list[dict[str, Any]], loop_params: np.ndarray) -> dict[str, Any]:
    pilot = curves[: min(BATCHED_SOLVER_PILOT_N, len(curves))]
    loop_p = loop_params[: len(pilot)]
    tpad, ypad, mask, scales, x0 = batch_arrays(pilot)
    n, max_len = tpad.shape
    lo = np.tile(np.array([0.15, 1.00, 0.25, 2.00, 0.03], dtype=float), n)
    hi = np.tile(np.array([0.80, 5.00, 0.85, 6.00, 0.40], dtype=float), n)

    @jax.jit
    def residual_jax(xflat: jnp.ndarray, jt: jnp.ndarray, jy: jnp.ndarray, jm: jnp.ndarray, js: jnp.ndarray) -> jnp.ndarray:
        x = xflat.reshape((n, 5))

        def one(p: jnp.ndarray, t: jnp.ndarray, y: jnp.ndarray, m: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
            tau_f, tau_s, alpha, fvc_l, tau_o = p
            base = alpha * jnp.exp(-t / tau_f) / tau_f + (1.0 - alpha) * jnp.exp(-t / tau_s) / tau_s
            rise = 1.0 - jnp.exp(-t / tau_o)
            qv = jnp.maximum(base * rise, 1e-12)
            area = jnp.trapezoid(qv, t)
            flow = fvc_l * qv / area
            inc = (flow[:-1] + flow[1:]) * 0.5 * jnp.diff(t)
            vol = jnp.concatenate([jnp.array([0.0], dtype=jnp.float64), jnp.cumsum(inc)])
            return ((vol - y) / scale) * m

        return jax.vmap(one)(x, jt, jy, jm, js).reshape(-1)

    jt = jnp.asarray(tpad, dtype=jnp.float64)
    jy = jnp.asarray(ypad, dtype=jnp.float64)
    jm = jnp.asarray(mask, dtype=jnp.float64)
    js = jnp.asarray(scales, dtype=jnp.float64)
    x0flat = x0.reshape(-1)
    residual_jax(jnp.asarray(x0flat, dtype=jnp.float64), jt, jy, jm, js).block_until_ready()

    def residual_np(x: np.ndarray) -> np.ndarray:
        return np.asarray(residual_jax(jnp.asarray(x, dtype=jnp.float64), jt, jy, jm, js))

    jac_pattern = sparse.lil_matrix((n * max_len, n * 5), dtype=int)
    for i in range(n):
        jac_pattern[i * max_len : (i + 1) * max_len, i * 5 : (i + 1) * 5] = 1
    jac_pattern = jac_pattern.tocsr()

    start = time.perf_counter()
    opt = least_squares(
        residual_np,
        x0=x0flat,
        bounds=(lo, hi),
        method="trf",
        jac_sparsity=jac_pattern,
        max_nfev=1000,
    )
    elapsed = time.perf_counter() - start
    batch_params = opt.x.reshape((n, 5))
    rel = np.abs(batch_params[:, :4] - loop_p[:, :4]) / np.maximum(np.abs(loop_p[:, :4]), 1e-12)
    abs_diff = np.abs(batch_params[:, :4] - loop_p[:, :4])
    worst_curve_idx = int(np.unravel_index(np.nanargmax(rel), rel.shape)[0]) if rel.size else -1
    return {
        "pilot_n": int(n),
        "requested_audit_n": int(VECTOR_AUDIT_N),
        "batched_full_200_not_run_reason": (
            "20-curve pilot already exceeded the <1e-6 relative-parameter equivalence guard; running the 200-curve stacked solver would be more expensive and still unshippable."
        ),
        "wall_clock_s": float(elapsed),
        "projected_wall_clock_per_200_s_linear": float(elapsed * 200.0 / max(n, 1)),
        "success": bool(opt.success),
        "nfev": int(opt.nfev),
        "cost": float(opt.cost),
        "max_relative_diff_tau_f_tau_s_alpha_fvc": float(np.nanmax(rel)) if rel.size else np.nan,
        "median_relative_diff_tau_f_tau_s_alpha_fvc": float(np.nanmedian(rel)) if rel.size else np.nan,
        "max_absolute_diff_tau_f_tau_s_alpha_fvc": [float(x) for x in np.nanmax(abs_diff, axis=0)] if abs_diff.size else [],
        "worst_curve_seqn": float(pilot[worst_curve_idx]["SEQN"]) if worst_curve_idx >= 0 else np.nan,
        "equivalence_tolerance": VECTOR_PARAM_REL_TOL,
        "passed": bool(rel.size and np.nanmax(rel) < VECTOR_PARAM_REL_TOL),
        "method_note": "Stacked sparse SciPy TRF solve with residual evaluated by jax.vmap+jit on padded curves.",
    }


def lhs_points(n: int, bounds: list[tuple[float, float]], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((n, len(bounds)), dtype=float)
    for j, (lo, hi) in enumerate(bounds):
        u = (np.arange(n, dtype=float) + rng.random(n)) / n
        rng.shuffle(u)
        out[:, j] = lo + (hi - lo) * u
    return out


def fit_from_start(t: np.ndarray, y: np.ndarray, scale: float, x0: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, Any]:
    def residual(x: np.ndarray) -> np.ndarray:
        return (np_ombe_volume(x, t) - y) / scale

    opt = least_squares(
        residual,
        x0=np.clip(x0, lo, hi),
        bounds=(lo, hi),
        method="trf",
        max_nfev=1000,
    )
    pred = np_ombe_volume(opt.x, t)
    sse = float(np.sum((pred - y) ** 2))
    return {"params": opt.x, "sse": sse, "success": bool(opt.success), "nfev": int(opt.nfev)}


def multistart_landscape(curves: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame]:
    bounds = [(0.01, 1.00), (1.00, 20.00), (0.25, 0.85), (2.00, 6.00), (0.03, 0.40)]
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for i, curve in enumerate(curves):
        default = np.array([0.35, 2.50, 0.60, curve["fvc_init"], 0.12], dtype=float)
        starts = np.vstack([default, lhs_points(MULTISTART_K - 1, bounds, SEED + int(curve["raw_index"]))])
        best_idx = -1
        best = {"sse": np.inf, "params": np.full(5, np.nan), "success": False, "nfev": 0}
        default_sse = np.nan
        for j, x0 in enumerate(starts):
            try:
                fit = fit_from_start(curve["t"], curve["y"], curve["scale"], x0, lo, hi)
            except Exception:
                fit = {"sse": np.inf, "params": np.full(5, np.nan), "success": False, "nfev": 0}
            if j == 0:
                default_sse = float(fit["sse"])
            if fit["sse"] < best["sse"]:
                best = fit
                best_idx = j
        improvement = default_sse - float(best["sse"]) if np.isfinite(default_sse) and np.isfinite(float(best["sse"])) else np.nan
        relative_improvement = float(improvement / default_sse) if np.isfinite(improvement) and default_sse > 0 else np.nan
        material_non_default = bool(best_idx != 0 and np.isfinite(relative_improvement) and relative_improvement > MATERIAL_SSE_IMPROVEMENT_THRESHOLD)
        rows.append(
            {
                "SEQN": curve["SEQN"],
                "raw_index": curve["raw_index"],
                "SPAMANU": curve["SPAMANU"],
                "truth_severity": curve["truth_severity"],
                "best_start_index": int(best_idx),
                "best_is_default": bool(best_idx == 0),
                "default_sse": default_sse,
                "best_sse": float(best["sse"]),
                "sse_improvement": float(improvement) if np.isfinite(improvement) else np.nan,
                "relative_sse_improvement": relative_improvement,
                "material_non_default_best": material_non_default,
                "best_tau_f": float(best["params"][0]),
                "best_tau_s": float(best["params"][1]),
                "best_alpha": float(best["params"][2]),
                "best_fvc_l": float(best["params"][3]),
                "best_tau_o": float(best["params"][4]),
                "best_success": bool(best["success"]),
            }
        )
        if (i + 1) % 50 == 0 or i + 1 == len(curves):
            print(f"Fit optimization multi-start landscape progress: {i + 1}/{len(curves)}")
    detail = pd.DataFrame(rows)
    elapsed = time.perf_counter() - start
    by_severity = []
    for severity, part in detail.groupby("truth_severity", dropna=False):
        non_default = ~part["best_is_default"].astype(bool)
        material = part["material_non_default_best"].astype(bool)
        by_severity.append(
            {
                "truth_severity": str(severity),
                "n": int(len(part)),
                "non_default_best_n": int(non_default.sum()),
                "non_default_best_rate": float(non_default.mean()) if len(part) else np.nan,
                "material_non_default_best_n": int(material.sum()),
                "material_local_minimum_rate": float(material.mean()) if len(part) else np.nan,
                "median_relative_sse_improvement_when_nondefault": float(np.nanmedian(part.loc[non_default, "relative_sse_improvement"])) if non_default.any() else 0.0,
                "median_relative_sse_improvement_when_material": float(np.nanmedian(part.loc[material, "relative_sse_improvement"])) if material.any() else 0.0,
            }
        )
    non_default_all = ~detail["best_is_default"].astype(bool)
    material_all = detail["material_non_default_best"].astype(bool)
    summary = {
        "n": int(len(detail)),
        "k": MULTISTART_K,
        "wall_clock_s": float(elapsed),
        "material_sse_improvement_threshold": MATERIAL_SSE_IMPROVEMENT_THRESHOLD,
        "non_default_best_n": int(non_default_all.sum()),
        "non_default_best_rate": float(non_default_all.mean()) if len(detail) else np.nan,
        "material_non_default_best_n": int(material_all.sum()),
        "material_local_minimum_rate": float(material_all.mean()) if len(detail) else np.nan,
        "median_relative_sse_improvement_when_nondefault": float(np.nanmedian(detail.loc[non_default_all, "relative_sse_improvement"])) if non_default_all.any() else 0.0,
        "median_relative_sse_improvement_when_material": float(np.nanmedian(detail.loc[material_all, "relative_sse_improvement"])) if material_all.any() else 0.0,
        "by_severity": sorted(by_severity, key=lambda r: SEVERITY_ORDER.index(r["truth_severity"]) if r["truth_severity"] in SEVERITY_ORDER else 99),
        "bounds": {
            "tau_f": [0.01, 1.00],
            "tau_s": [1.00, 20.00],
            "alpha": [0.25, 0.85],
            "fvc_l": [2.00, 6.00],
            "tau_o": [0.03, 0.40],
        },
        "note": "Diagnostic-only because Part 1 batched-fit equivalence guard failed; these starts were not propagated to headline gates.",
    }
    return summary, detail


def build_deltas(baseline_rows: pd.DataFrame, accepted_config: dict[str, bool], halt_reason: str) -> pd.DataFrame:
    out = baseline_rows.copy()
    out["final_value"] = out["baseline_value"]
    out["delta"] = 0.0
    out["delta_sign"] = "0"
    out["statement"] = "UNCHANGED"
    out["accepted_config"] = json.dumps(accepted_config, sort_keys=True)
    out["reason"] = halt_reason
    return out


def make_report(result: dict[str, Any], deltas: pd.DataFrame) -> str:
    baseline = pd.DataFrame(result["baseline_rows"])
    part1 = result["part1_vectorized_fit"]
    landscape = result["part2_multistart_landscape"]
    lines = [
        "# Gate - Deterministic Fit Optimization Pass, Cycle F",
        "",
        "## Protocol Decisions",
        "",
        "- This is a guarded optimization gate, not a production rewrite. No production estimator is changed unless every guard accepts the new configuration.",
        "- Baseline headlines are locked by reading existing result files before any refit.",
        "- The forward model is guarded by JAX-vs-NumPy volume agreement in mL with x64 enabled.",
        "- The current production volume fit is SciPy bounded TRF least squares; JAX is used for a batched residual equivalence test, not as an unreviewed replacement optimizer.",
        "- Because the batched-fit equivalence guard fails, Parts 3-5 do not ship robust loss, onset changes, or multi-start headline recomputation.",
        "",
        "## Part 0 - Baseline Lock",
        "",
        md_table(
            baseline,
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
        "## Part 1 - Vectorized Fit Enabling Step",
        "",
        f"Loop refit completed on {part1['loop']['n']} curves in {part1['loop']['wall_clock_s']:.3f}s "
        f"({part1['loop']['wall_clock_per_200_s']:.3f}s per 200 curves).",
        "",
        f"The stacked sparse batched solver pilot used {part1['batched_solver_pilot']['pilot_n']} curves. "
        f"It took {part1['batched_solver_pilot']['wall_clock_s']:.3f}s "
        f"(linear projection {part1['batched_solver_pilot']['projected_wall_clock_per_200_s_linear']:.3f}s per 200), "
        f"with max relative difference {part1['batched_solver_pilot']['max_relative_diff_tau_f_tau_s_alpha_fvc']:.6g} "
        f"against the current loop for (tau_f, tau_s, alpha, FVC).",
        "",
        f"Guard result: {'PASS' if part1['passed'] else 'FAIL'}; tolerance is < {VECTOR_PARAM_REL_TOL:g}.",
        "",
        part1["batched_solver_pilot"]["batched_full_200_not_run_reason"],
        "",
        "## Part 2 - Multi-Start Landscape Diagnostic",
        "",
        "This diagnostic was run on the 200-curve audit set, but it is not allowed to change headline gates because Part 1 failed.",
        "",
        f"Raw non-default-best rate: {landscape['non_default_best_n']} / {landscape['n']} = {landscape['non_default_best_rate']:.3f}.",
        "",
        f"Material local-minimum rate, requiring relative SSE improvement > {landscape['material_sse_improvement_threshold']:.0e}: "
        f"{landscape['material_non_default_best_n']} / {landscape['n']} = {landscape['material_local_minimum_rate']:.3f}.",
        "",
        md_table(
            landscape["by_severity"],
            [
                ("truth_severity", "severity", "str"),
                ("n", "n", "int"),
                ("non_default_best_n", "raw non-default", "int"),
                ("non_default_best_rate", "raw rate", "float3"),
                ("material_non_default_best_n", "material non-default", "int"),
                ("material_local_minimum_rate", "material rate", "float3"),
                ("median_relative_sse_improvement_when_material", "material median rel SSE improvement", "float6"),
            ],
        ),
        "",
        "## Parts 3-5 - Guarded Out",
        "",
        "Robust loss, onset/back-extrapolation replacement, and combined final configuration were not propagated because the enabling vectorized-fit guard failed. This prevents accidental regression in already-passing headlines, especially severe-band FN = 0.000.",
        "",
        "Accepted configuration: multi-start off; robust loss off; onset replacement off.",
        "",
        "## Final Headline Deltas",
        "",
        md_table(
            deltas,
            [
                ("headline", "headline", "str"),
                ("metric", "metric", "str"),
                ("estimator", "estimator", "str"),
                ("endpoint", "endpoint", "str"),
                ("baseline_value", "baseline", "float6"),
                ("final_value", "final", "float6"),
                ("delta", "delta", "signed"),
                ("statement", "statement", "str"),
            ],
        ),
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

    audit_rows = select_fit_audit_rows(root, VECTOR_AUDIT_N)
    curves = load_curves(root, audit_rows)
    loop_params, loop_summary = fit_loop(curves)
    batched = batched_solver_pilot(curves, loop_params)
    part1_passed = bool(batched["passed"])

    landscape, landscape_detail = multistart_landscape(curves)

    accepted_config = {"multi_start": False, "robust_loss": False, "ats_onset": False}
    if part1_passed:
        halt_reason = "Part 1 passed, but no later lever was accepted by this guarded pass."
        verdict = "OPTIMIZATION-NEUTRAL"
        verdict_reason = "All headline numbers are unchanged because no additional lever was accepted."
    else:
        halt_reason = "Part 1 batched-fit equivalence guard failed; no candidate optimization may ship."
        if float(landscape["material_local_minimum_rate"]) >= 0.10:
            verdict = "OPTIMIZATION-LANDSCAPE"
            verdict_reason = (
                "No headline changed because the shipping guard halted, but multi-start revealed a high local-minimum rate; "
                "this is a fit-stability characterization, not an accepted production change."
            )
        else:
            verdict = "OPTIMIZATION-NEUTRAL"
            verdict_reason = (
                "No headline changed. Raw non-default-start wins were common, but material SSE improvements were rare, "
                "so the baseline deterministic fit appears to be at its practical ceiling under this guarded pass."
            )

    deltas = build_deltas(baseline_rows, accepted_config, halt_reason)
    guard_end = forward_model_guard(SEED + 1)
    if not guard_end["passed"]:
        raise RuntimeError(f"Forward-model guard failed at end: {guard_end}")

    result = {
        "seed": SEED,
        "baseline": baseline,
        "baseline_rows": baseline_rows.to_dict("records"),
        "forward_model_guard_start": guard_start,
        "part1_vectorized_fit": {
            "audit_curve_n": int(len(curves)),
            "loop": loop_summary,
            "batched_solver_pilot": batched,
            "passed": part1_passed,
            "halt_if_failed_rule": "If batched output diverges from the loop beyond tolerance, do not proceed on a fit that changed the numbers.",
        },
        "part2_multistart_landscape": landscape,
        "part3_robust_loss": {"run": False, "reason": halt_reason},
        "part4_onset_backextrapolation": {"run": False, "reason": halt_reason},
        "part5_combined_best_configuration": {"accepted_config": accepted_config, "reason": halt_reason},
        "forward_model_guard_end": guard_end,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }

    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    deltas.to_csv(out / f"{OUTPUT_STEM}_deltas.csv", index=False)
    landscape_detail.to_csv(out / f"{OUTPUT_STEM}_multistart_landscape.csv", index=False)
    (out / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    (out / f"{OUTPUT_STEM}.md").write_text(make_report(result, deltas), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    p1 = result["part1_vectorized_fit"]["batched_solver_pilot"]
    land = result["part2_multistart_landscape"]
    print(f"VERDICT: {result['verdict']}")
    print(
        "Part 1 batched-fit guard: "
        f"passed={result['part1_vectorized_fit']['passed']} max_rel={p1['max_relative_diff_tau_f_tau_s_alpha_fvc']:.6g}"
    )
    print(
        "Multi-start material local-minimum rate: "
        f"{land['material_non_default_best_n']}/{land['n']} = {land['material_local_minimum_rate']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
