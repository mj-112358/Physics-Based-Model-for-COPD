from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache" / "xdg"))

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS
from scipy.interpolate import BSpline
from scipy.stats import chisquare

from .gate_clinical_classification_f import RANDOM_SEED, extract_curve_for_fit
from .gate_likelihood_repair_f import (
    CONVERGED_ESS_MIN,
    CONVERGED_RHAT_MAX,
    COVERAGE_LEVELS,
    FIXED_SIGMA_L,
    NUTS_CHAINS,
    NUTS_DRAWS,
    NUTS_GRID_STRIDE,
    NUTS_WARMUP,
    PARAM_NAMES,
    PIT_BINS,
    PIT_P_THRESHOLD,
    WIDE_PRIOR_RANGES,
    build_subsets,
    load_rows_and_raw,
)
from .gate_posterior_calibration_repair_f import bulk_ess, natural_ranges_for_report, split_rhat, to_jsonable
from .ombe import ombe_volume, phi_from_natural

KOH_KNOTS = (2, 4, 6, 8)
REPORT_TIMES_S = (0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
EXAMPLE_RAW_INDICES = (26434, 28367, 11209)
VARIANT_SEED_OFFSETS = {
    "L0_fixed": 110_000,
    "L4_ar1_time_hetero": 120_000,
    "KOH_K2": 130_000,
    "KOH_K4": 140_000,
    "KOH_K6": 150_000,
    "KOH_K8": 160_000,
}


def configure_caches(root: Path) -> None:
    cache = root / ".cache"
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))


def q(values: np.ndarray | pd.Series) -> list[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return [float("nan"), float("nan")]
    return [float(x) for x in np.percentile(arr, [25, 75])]


def jax_volume_nonuniform(phi: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    tau_f = jnp.exp(phi[0])
    tau_s = jnp.exp(phi[1])
    alpha = jax.nn.sigmoid(phi[2])
    fvc_l = jnp.exp(phi[3])
    tau_o = jnp.exp(phi[4])
    base = alpha * jnp.exp(-t / tau_f) / tau_f + (1.0 - alpha) * jnp.exp(-t / tau_s) / tau_s
    rise = 1.0 - jnp.exp(-t / tau_o)
    shape = jnp.maximum(base * rise, 1e-12)
    area = jnp.trapezoid(shape, t)
    flow = fvc_l * shape / area
    increments = (flow[:-1] + flow[1:]) * 0.5 * jnp.diff(t)
    return jnp.concatenate([jnp.array([0.0]), jnp.cumsum(increments)])


def numpy_volume_nonuniform(phi: np.ndarray, t: np.ndarray) -> np.ndarray:
    phi = np.atleast_2d(np.asarray(phi, dtype=float))
    t = np.asarray(t, dtype=float)
    tau_f = np.exp(phi[:, 0])[:, None]
    tau_s = np.exp(phi[:, 1])[:, None]
    alpha = (1.0 / (1.0 + np.exp(-phi[:, 2])))[:, None]
    fvc_l = np.exp(phi[:, 3])[:, None]
    tau_o = np.exp(phi[:, 4])[:, None]
    tt = t[None, :]
    base = alpha * np.exp(-tt / tau_f) / tau_f + (1.0 - alpha) * np.exp(-tt / tau_s) / tau_s
    rise = 1.0 - np.exp(-tt / tau_o)
    shape = np.maximum(base * rise, 1e-12)
    area = np.trapezoid(shape, t, axis=1)[:, None]
    flow = fvc_l * shape / area
    increments = (flow[:, :-1] + flow[:, 1:]) * 0.5 * np.diff(t)[None, :]
    return np.concatenate([np.zeros((len(phi), 1)), np.cumsum(increments, axis=1)], axis=1)


def phi_from_row(row: Any) -> jnp.ndarray:
    alpha = min(max(float(row.fit_alpha), 1e-6), 1.0 - 1e-6)
    return phi_from_natural(
        jnp.asarray(float(row.fit_tau_f)),
        jnp.asarray(float(row.fit_tau_s)),
        jnp.asarray(alpha),
        jnp.asarray(float(row.fit_fvc_l)),
        jnp.asarray(float(row.fit_tau_o)),
    )


def bspline_basis(t: np.ndarray, n_internal_knots: int) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    degree = 3
    lo = float(t[0])
    hi = float(t[-1])
    if hi <= lo:
        raise ValueError("time grid must be increasing")
    internal = np.linspace(lo, hi, n_internal_knots + 2)[1:-1]
    knots = np.concatenate([np.repeat(lo, degree + 1), internal, np.repeat(hi, degree + 1)])
    n_basis = len(knots) - degree - 1
    basis = []
    eye = np.eye(n_basis)
    for i in range(n_basis):
        values = BSpline(knots, eye[i], degree, extrapolate=False)(t)
        basis.append(np.nan_to_num(values, nan=0.0))
    mat = np.column_stack(basis).astype(np.float32)
    return mat - mat[0:1, :]


def fit_row_phi_np(row: Any) -> np.ndarray:
    alpha = min(max(float(row.fit_alpha), 1e-6), 1.0 - 1e-6)
    return np.array(
        [
            np.log(float(row.fit_tau_f)),
            np.log(float(row.fit_tau_s)),
            np.log(alpha / (1.0 - alpha)),
            np.log(float(row.fit_fvc_l)),
            np.log(float(row.fit_tau_o)),
        ],
        dtype=np.float32,
    )


def residual_arrays(raw_row: Any, fit_row: Any) -> dict[str, np.ndarray]:
    curve = extract_curve_for_fit(raw_row)
    t = np.asarray(curve["vol_t"], dtype=np.float32)
    y = np.asarray(curve["vol_l"], dtype=np.float32)
    phi_jax = phi_from_row(fit_row)
    old = np.asarray(ombe_volume(phi_jax, jnp.asarray(t)))
    fixed = np.asarray(jax_volume_nonuniform(phi_jax, jnp.asarray(t)))
    return {
        "t_s": t,
        "observed_l": y,
        "fitted_old_l": old,
        "fitted_fixed_l": fixed,
        "residual_old_l": y - old,
        "residual_fixed_l": y - fixed,
    }


def residual_stats_from_resid(resid_l: np.ndarray) -> dict[str, float]:
    resid_ml = np.asarray(resid_l, dtype=float) * 1000.0
    resid_ml = resid_ml[np.isfinite(resid_ml)]
    if len(resid_ml) < 3:
        return {"rms_ml": np.nan, "lag1_autocorr": np.nan, "durbin_watson": np.nan}
    centered = resid_ml - np.mean(resid_ml)
    denom = float(np.dot(centered, centered))
    lag1 = float(np.dot(centered[:-1], centered[1:]) / denom) if denom > 0.0 else np.nan
    dw_denom = float(np.dot(resid_ml, resid_ml))
    dw = float(np.sum(np.diff(resid_ml) ** 2) / dw_denom) if dw_denom > 0.0 else np.nan
    return {"rms_ml": float(np.sqrt(np.mean(resid_ml**2))), "lag1_autocorr": lag1, "durbin_watson": dw}


def step0_residual_diagnosis(rows: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_by_index = {int(idx): row for idx, row in raw.iterrows()}
    out = []
    examples = []
    usable = rows[(rows["fit_error"].fillna("") == "") & (rows["volume_fit_success"].fillna(False).astype(bool))].copy()
    for i, row in enumerate(usable.itertuples(index=False), start=1):
        raw_idx = int(row.raw_index)
        base = {
            "SEQN": row.SEQN,
            "raw_index": raw_idx,
            "split": row.split,
            "truth_severity": row.truth_severity,
            "fev1_fvc_ratio": float(row.fev1_fvc_ratio),
            "truth_obstructed": bool(float(row.fev1_fvc_ratio) < 0.70),
        }
        try:
            arr = residual_arrays(raw_by_index[raw_idx], row)
            old = residual_stats_from_resid(arr["residual_old_l"])
            fixed = residual_stats_from_resid(arr["residual_fixed_l"])
            base.update(
                {
                    "old_rms_ml": old["rms_ml"],
                    "old_lag1_autocorr": old["lag1_autocorr"],
                    "old_durbin_watson": old["durbin_watson"],
                    "fixed_rms_ml": fixed["rms_ml"],
                    "fixed_lag1_autocorr": fixed["lag1_autocorr"],
                    "fixed_durbin_watson": fixed["durbin_watson"],
                    "residual_error": "",
                }
            )
            if raw_idx in EXAMPLE_RAW_INDICES:
                examples.append(example_block(raw_idx, arr))
        except Exception as exc:
            base["residual_error"] = str(exc)
        out.append(base)
        if i % 250 == 0 or i == len(usable):
            print(f"KOH Step 0 residual progress: {i}/{len(usable)}")
    residual_rows = pd.DataFrame(out)
    good = residual_rows[residual_rows["residual_error"].fillna("") == ""].copy()
    before = summarize_residual_block(good, "old")
    after = summarize_residual_block(good, "fixed")
    scale_ml = after["all"]["rms_ml"]["median"]
    reconciled = bool(5.0 <= scale_ml <= 250.0)
    result = {
        "examples": examples,
        "before": before,
        "after": after,
        "reconciled_residual_scale_ml": float(scale_ml),
        "delta_prior_scale_l": float(scale_ml / 1000.0),
        "delta_prior_scale_rule": "Gaussian spline-coefficient SD equals the median fixed residual RMS across the fitted Cycle-F pool.",
        "unit_alignment": "observed, fitted, residual arrays are all litres; reported RMS is converted to mL.",
        "time_grid_alignment": "fixed model integrates OMBE volume using np.diff(t), preserving the fractional first interval after back-extrapolated t0.",
        "baseline_alignment": "delta and OMBE volume both start at 0 L; spline basis is centered as B(t)-B(0) so delta(0)=0.",
        "residual_scale_reconciled": reconciled,
    }
    return residual_rows, result


def example_block(raw_idx: int, arr: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "raw_index": raw_idx,
        "units": {"time": "seconds", "volume": "litres", "residual": "litres"},
        "first10": {
            "time_s": np.round(arr["t_s"][:10], 6).tolist(),
            "observed_l": np.round(arr["observed_l"][:10], 6).tolist(),
            "fitted_old_l": np.round(arr["fitted_old_l"][:10], 6).tolist(),
            "fitted_fixed_l": np.round(arr["fitted_fixed_l"][:10], 6).tolist(),
            "residual_old_l": np.round(arr["residual_old_l"][:10], 6).tolist(),
            "residual_fixed_l": np.round(arr["residual_fixed_l"][:10], 6).tolist(),
        },
        "last10": {
            "time_s": np.round(arr["t_s"][-10:], 6).tolist(),
            "observed_l": np.round(arr["observed_l"][-10:], 6).tolist(),
            "fitted_old_l": np.round(arr["fitted_old_l"][-10:], 6).tolist(),
            "fitted_fixed_l": np.round(arr["fitted_fixed_l"][-10:], 6).tolist(),
            "residual_old_l": np.round(arr["residual_old_l"][-10:], 6).tolist(),
            "residual_fixed_l": np.round(arr["residual_fixed_l"][-10:], 6).tolist(),
        },
    }


def summarize_residual_block(df: pd.DataFrame, prefix: str) -> dict[str, Any]:
    def metric(series: pd.Series) -> dict[str, Any]:
        arr = np.asarray(series, dtype=float)
        arr = arr[np.isfinite(arr)]
        return {"n": int(len(arr)), "median": float(np.median(arr)) if len(arr) else np.nan, "iqr": q(arr)}

    def block(part: pd.DataFrame) -> dict[str, Any]:
        return {
            "n": int(len(part)),
            "rms_ml": metric(part[f"{prefix}_rms_ml"]),
            "lag1_autocorr": metric(part[f"{prefix}_lag1_autocorr"]),
            "durbin_watson": metric(part[f"{prefix}_durbin_watson"]),
        }

    return {
        "all": block(df),
        "obstructed_truth": block(df[df["fev1_fvc_ratio"] < 0.70]),
        "normal_truth": block(df[df["fev1_fvc_ratio"] >= 0.70]),
    }


def phi_from_samples(samples: dict[str, np.ndarray]) -> np.ndarray:
    tau_f = np.ravel(samples["tau_f"])
    tau_s = np.ravel(samples["tau_s"])
    alpha = np.clip(np.ravel(samples["alpha"]), 1e-6, 1.0 - 1e-6)
    fvc_l = np.ravel(samples["fvc_l"])
    tau_o = np.ravel(samples["tau_o"])
    return np.column_stack([np.log(tau_f), np.log(tau_s), np.log(alpha / (1.0 - alpha)), np.log(fvc_l), np.log(tau_o)]).astype(np.float32)


def koh_model(t: jnp.ndarray, basis: jnp.ndarray | None, y: jnp.ndarray, variant: str, delta_scale_l: float) -> None:
    tau_f = numpyro.sample("tau_f", dist.Uniform(*WIDE_PRIOR_RANGES["tau_f"]))
    tau_s = numpyro.sample("tau_s", dist.Uniform(*WIDE_PRIOR_RANGES["tau_s"]))
    alpha = numpyro.sample("alpha", dist.Uniform(*WIDE_PRIOR_RANGES["alpha"]))
    fvc_l = numpyro.sample("fvc_l", dist.Uniform(*WIDE_PRIOR_RANGES["fvc_l"]))
    tau_o = numpyro.sample("tau_o", dist.Uniform(*WIDE_PRIOR_RANGES["tau_o"]))
    phi = phi_from_natural(tau_f, tau_s, alpha, fvc_l, tau_o)
    mu = jax_volume_nonuniform(phi, t)
    if variant.startswith("KOH_"):
        if basis is None:
            raise ValueError("KOH variant requires spline basis")
        n_basis = basis.shape[1]
        coef = numpyro.sample("delta_coef", dist.Normal(0.0, delta_scale_l).expand([n_basis]).to_event(1))
        mu = mu + basis @ coef
    resid = y - mu
    if variant == "L0_fixed":
        numpyro.sample("y", dist.Normal(mu, FIXED_SIGMA_L), obs=y)
    elif variant == "L4_ar1_time_hetero":
        sigma = numpyro.sample("sigma_l", dist.HalfNormal(0.50))
        rho = numpyro.sample("rho", dist.Uniform(0.0, 0.99))
        beta_time = numpyro.sample("beta_time", dist.HalfNormal(2.0))
        scaled_t = t / jnp.maximum(t[-1], 1e-3)
        sigma_t = sigma * (1.0 + beta_time * scaled_t)
        init_sd = sigma_t[0] / jnp.sqrt(jnp.maximum(1.0 - rho**2, 1e-4))
        numpyro.sample("e0", dist.Normal(0.0, init_sd), obs=resid[0])
        numpyro.sample("e_ar_hetero", dist.Normal(rho * resid[:-1], sigma_t[1:]), obs=resid[1:])
    elif variant.startswith("KOH_"):
        sigma = numpyro.sample("sigma_l", dist.HalfNormal(0.50))
        rho = numpyro.sample("rho", dist.Uniform(0.0, 0.99))
        init_sd = sigma / jnp.sqrt(jnp.maximum(1.0 - rho**2, 1e-4))
        numpyro.sample("e0", dist.Normal(0.0, init_sd), obs=resid[0])
        numpyro.sample("e_ar", dist.Normal(rho * resid[:-1], sigma), obs=resid[1:])
    else:
        raise ValueError(f"unknown variant: {variant}")


def diagnostics_from_mcmc(mcmc: MCMC) -> dict[str, Any]:
    samples = mcmc.get_samples(group_by_chain=True)
    rhat_values = {}
    ess_values = {}
    for name, arr in samples.items():
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 2:
            dims = [(name, arr)]
        else:
            flat = arr.reshape(arr.shape[0], arr.shape[1], -1)
            dims = [(f"{name}[{i}]", flat[:, :, i]) for i in range(flat.shape[2])]
        for label, values in dims:
            rhat_values[label] = split_rhat(values)
            ess_values[label] = bulk_ess(values)
    rhat_arr = np.asarray(list(rhat_values.values()), dtype=float)
    ess_arr = np.asarray(list(ess_values.values()), dtype=float)
    max_rhat = float(np.nanmax(rhat_arr)) if np.isfinite(rhat_arr).any() else np.nan
    min_ess = float(np.nanmin(ess_arr)) if np.isfinite(ess_arr).any() else np.nan
    return {
        "max_rhat": max_rhat,
        "min_bulk_ess": min_ess,
        "rhat_by_site": rhat_values,
        "bulk_ess_by_site": ess_values,
        "converged": bool(max_rhat <= CONVERGED_RHAT_MAX and min_ess >= CONVERGED_ESS_MIN),
    }


def posterior_ratio_and_delta(
    flat_samples: dict[str, np.ndarray],
    t: np.ndarray,
    basis: np.ndarray | None,
    variant: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    phi = phi_from_samples(flat_samples)
    volume = numpy_volume_nonuniform(phi, t)
    delta = None
    if variant.startswith("KOH_"):
        coef = np.asarray(flat_samples["delta_coef"], dtype=float)
        if coef.ndim == 1:
            coef = coef[:, None]
        delta = coef @ basis.T
        pred = volume + delta
    else:
        pred = volume
    fev1 = np.array([np.interp(min(1.0, t[-1]), t, row) for row in pred])
    fvc = pred[:, -1]
    ratio = fev1 / np.maximum(fvc, 1e-8)
    return ratio[np.isfinite(ratio)], delta


def posterior_extra_summary(flat: dict[str, np.ndarray], variant: str) -> dict[str, float]:
    out = {}
    if variant == "L0_fixed":
        out["sigma_ml_median"] = FIXED_SIGMA_L * 1000.0
    if "sigma_l" in flat:
        out["sigma_ml_median"] = float(np.median(flat["sigma_l"]) * 1000.0)
    if "rho" in flat:
        out["rho_median"] = float(np.median(flat["rho"]))
    if "beta_time" in flat:
        out["beta_time_median"] = float(np.median(flat["beta_time"]))
    return out


def param_sds(flat: dict[str, np.ndarray]) -> dict[str, float]:
    return {f"{name}_sd": float(np.std(np.ravel(flat[name]), ddof=1)) for name in PARAM_NAMES if name in flat}


def theta_delta_correlations(flat: dict[str, np.ndarray]) -> dict[str, float]:
    if "delta_coef" not in flat:
        return {f"{name}_max_abs_corr_delta": np.nan for name in PARAM_NAMES}
    coef = np.asarray(flat["delta_coef"], dtype=float)
    if coef.ndim == 1:
        coef = coef[:, None]
    out = {}
    for name in PARAM_NAMES:
        x = np.ravel(flat[name])
        corrs = []
        for j in range(coef.shape[1]):
            y = coef[:, j]
            if np.std(x) > 0 and np.std(y) > 0:
                corrs.append(abs(float(np.corrcoef(x, y)[0, 1])))
        out[f"{name}_max_abs_corr_delta"] = float(np.max(corrs)) if corrs else np.nan
    return out


def run_variant(variant: str, subset: pd.DataFrame, raw: pd.DataFrame, delta_scale_l: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_by_index = {int(idx): row for idx, row in raw.iterrows()}
    rows_out = []
    start = time.perf_counter()
    knot_count = int(variant.split("_K")[-1]) if variant.startswith("KOH_K") else None
    for i, row in enumerate(subset.itertuples(index=False), start=1):
        raw_idx = int(row.raw_index)
        base = {
            "variant": variant,
            "SEQN": row.SEQN,
            "raw_index": raw_idx,
            "truth_severity": row.truth_severity,
            "truth_ratio": float(row.fev1_fvc_ratio),
            "truth_obstructed": bool(float(row.fev1_fvc_ratio) < 0.70),
            "nested_prior_100": bool(row.nested_prior_100),
        }
        try:
            curve = extract_curve_for_fit(raw_by_index[raw_idx])
            t = np.asarray(curve["vol_t"], dtype=np.float32)[::NUTS_GRID_STRIDE]
            y = np.asarray(curve["vol_l"], dtype=np.float32)[::NUTS_GRID_STRIDE]
            if len(t) < 20:
                raise ValueError("too few downsampled volume points")
            basis = bspline_basis(t, knot_count) if knot_count is not None else None
            kernel = NUTS(lambda obs=None: koh_model(jnp.asarray(t), None if basis is None else jnp.asarray(basis), obs, variant, delta_scale_l))
            mcmc = MCMC(
                kernel,
                num_warmup=NUTS_WARMUP,
                num_samples=NUTS_DRAWS,
                num_chains=NUTS_CHAINS,
                chain_method="sequential",
                progress_bar=False,
            )
            mcmc.run(jax.random.PRNGKey(RANDOM_SEED + raw_idx + VARIANT_SEED_OFFSETS[variant]), obs=jnp.asarray(y))
            diag = diagnostics_from_mcmc(mcmc)
            samples_chain = mcmc.get_samples(group_by_chain=True)
            flat = {name: np.asarray(value).reshape((-1,) + np.asarray(value).shape[2:]) for name, value in samples_chain.items()}
            ratio, delta = posterior_ratio_and_delta(flat, t, basis, variant)
            pit = float(np.mean(ratio < float(row.fev1_fvc_ratio))) if len(ratio) else np.nan
            base.update(
                {
                    "fit_error": "",
                    "n_ratio_draws": int(len(ratio)),
                    "ratio_mean": float(np.mean(ratio)) if len(ratio) else np.nan,
                    "ratio_sd": float(np.std(ratio, ddof=1)) if len(ratio) > 1 else np.nan,
                    "ratio_q025": float(np.quantile(ratio, 0.025)) if len(ratio) else np.nan,
                    "ratio_q10": float(np.quantile(ratio, 0.10)) if len(ratio) else np.nan,
                    "ratio_q25": float(np.quantile(ratio, 0.25)) if len(ratio) else np.nan,
                    "ratio_q50": float(np.quantile(ratio, 0.50)) if len(ratio) else np.nan,
                    "ratio_q75": float(np.quantile(ratio, 0.75)) if len(ratio) else np.nan,
                    "ratio_q90": float(np.quantile(ratio, 0.90)) if len(ratio) else np.nan,
                    "ratio_q975": float(np.quantile(ratio, 0.975)) if len(ratio) else np.nan,
                    "signed_ratio_median_error": float(np.median(ratio) - float(row.fev1_fvc_ratio)) if len(ratio) else np.nan,
                    "pit": pit,
                    "n_delta_basis": int(basis.shape[1]) if basis is not None else 0,
                    **posterior_extra_summary(flat, variant),
                    **param_sds(flat),
                    **theta_delta_correlations(flat),
                    **diag,
                }
            )
            for level in COVERAGE_LEVELS:
                lo = (1.0 - level) / 2.0
                hi = 1.0 - lo
                qlo, qhi = np.quantile(ratio, [lo, hi]) if len(ratio) else (np.nan, np.nan)
                base[f"covered_{int(level * 100)}"] = bool(qlo <= float(row.fev1_fvc_ratio) <= qhi)
            if delta is not None:
                mean_delta = np.mean(delta, axis=0)
                for rt in REPORT_TIMES_S:
                    base[f"delta_mean_{rt:g}s_ml"] = (
                        float(np.interp(rt, t, mean_delta) * 1000.0) if rt <= float(t[-1]) else np.nan
                    )
        except Exception as exc:
            base.update({"fit_error": str(exc), "converged": False})
        rows_out.append(base)
        if i % 10 == 0 or i == len(subset):
            print(f"{variant}: NUTS progress {i}/{len(subset)}")
    rows = pd.DataFrame(rows_out)
    summary = summarize_variant(rows)
    summary["wall_clock_s"] = float(time.perf_counter() - start)
    return rows, summary


def summarize_variant(rows: pd.DataFrame) -> dict[str, Any]:
    ok = rows[(rows["fit_error"].fillna("") == "") & (rows["converged"].fillna(False).astype(bool))].copy()
    p = ok["pit"].to_numpy(float)
    counts, _ = np.histogram(p[np.isfinite(p)], bins=np.linspace(0.0, 1.0, PIT_BINS + 1))
    pit_p = float(chisquare(counts).pvalue) if len(ok) else np.nan
    coverage = {}
    for level in COVERAGE_LEVELS:
        col = f"covered_{int(level * 100)}"
        coverage[str(level)] = float(ok[col].mean()) if len(ok) and col in ok else np.nan
    gross_skew = bool(
        len(ok) == 0
        or pit_p < PIT_P_THRESHOLD
        or counts.max() > 2.5 * max(len(ok) / PIT_BINS, 1e-12)
        or abs(float(np.nanmean(p)) - 0.5) > 0.10
    )
    return {
        "n_attempted": int(len(rows)),
        "n_fit_errors": int((rows["fit_error"].fillna("") != "").sum()),
        "n_converged": int(len(ok)),
        "n_nonconverged_excluded": int(len(rows) - len(ok)),
        "coverage": coverage,
        "pit_counts": counts.astype(int).tolist(),
        "pit_chisquare_p": pit_p,
        "pit_p_threshold": PIT_P_THRESHOLD,
        "pit_mean": float(np.nanmean(p)) if len(ok) else np.nan,
        "gross_skew": gross_skew,
        "calibrated": bool(
            len(ok) > 0 and all(abs(coverage[str(level)] - level) <= 0.07 for level in COVERAGE_LEVELS) and not gross_skew
        ),
        "ratio_sd_median": float(np.nanmedian(ok["ratio_sd"])) if len(ok) else np.nan,
        "signed_ratio_median_error": float(np.nanmedian(ok["signed_ratio_median_error"])) if len(ok) else np.nan,
        "sigma_ml_median": float(np.nanmedian(ok["sigma_ml_median"])) if len(ok) and "sigma_ml_median" in ok else np.nan,
        "rho_median": float(np.nanmedian(ok["rho_median"])) if len(ok) and "rho_median" in ok else np.nan,
        "beta_time_median": float(np.nanmedian(ok["beta_time_median"])) if len(ok) and "beta_time_median" in ok else np.nan,
        "max_rhat_median": float(np.nanmedian(ok["max_rhat"])) if len(ok) else np.nan,
        "max_rhat_max": float(np.nanmax(ok["max_rhat"])) if len(ok) else np.nan,
        "min_bulk_ess_median": float(np.nanmedian(ok["min_bulk_ess"])) if len(ok) else np.nan,
        "min_bulk_ess_min": float(np.nanmin(ok["min_bulk_ess"])) if len(ok) else np.nan,
    }


def delta_shape_table(rows: pd.DataFrame, variant: str = "KOH_K4") -> list[dict[str, Any]]:
    part = rows[(rows["variant"] == variant) & (rows["fit_error"].fillna("") == "") & (rows["converged"].fillna(False).astype(bool))]
    out = []
    for rt in REPORT_TIMES_S:
        col = f"delta_mean_{rt:g}s_ml"
        if col not in part:
            continue
        all_values = part[col]
        obs_values = part[part["truth_obstructed"]][col]
        normal_values = part[~part["truth_obstructed"]][col]
        out.append(
            {
                "time_s": float(rt),
                "all_mean_delta_ml": float(np.nanmean(all_values)),
                "obstructed_mean_delta_ml": float(np.nanmean(obs_values)),
                "normal_mean_delta_ml": float(np.nanmean(normal_values)),
                "n_all": int(all_values.notna().sum()),
                "n_obstructed": int(obs_values.notna().sum()),
                "n_normal": int(normal_values.notna().sum()),
            }
        )
    tail = [r for r in out if r["time_s"] >= 4.0 and r["n_all"] > 0]
    early = [r for r in out if r["time_s"] <= 1.0 and r["n_all"] > 0]
    return out, {
        "tail_positive": bool(np.nanmean([r["all_mean_delta_ml"] for r in tail]) > 0.0) if tail else False,
        "early_near_zero_abs_mean_ml": float(np.nanmean(np.abs([r["all_mean_delta_ml"] for r in early]))) if early else np.nan,
        "obstructed_tail_larger_than_normal": bool(
            np.nanmean([r["obstructed_mean_delta_ml"] for r in tail])
            > np.nanmean([r["normal_mean_delta_ml"] for r in tail])
        )
        if tail
        else False,
    }


def identifiability_table(rows: pd.DataFrame) -> list[dict[str, Any]]:
    l4 = rows[(rows["variant"] == "L4_ar1_time_hetero") & (rows["fit_error"].fillna("") == "") & (rows["converged"].fillna(False).astype(bool))]
    k4 = rows[(rows["variant"] == "KOH_K4") & (rows["fit_error"].fillna("") == "") & (rows["converged"].fillna(False).astype(bool))]
    out = []
    for name in PARAM_NAMES:
        l4_sd = float(np.nanmedian(l4[f"{name}_sd"])) if f"{name}_sd" in l4 else np.nan
        k4_sd = float(np.nanmedian(k4[f"{name}_sd"])) if f"{name}_sd" in k4 else np.nan
        corr_col = f"{name}_max_abs_corr_delta"
        out.append(
            {
                "parameter": name,
                "l4_sd_median": l4_sd,
                "koh_k4_sd_median": k4_sd,
                "sd_ratio_koh_over_l4": float(k4_sd / l4_sd) if np.isfinite(k4_sd) and np.isfinite(l4_sd) and l4_sd > 0 else np.nan,
                "koh_k4_max_abs_theta_delta_corr_median": float(np.nanmedian(k4[corr_col])) if corr_col in k4 else np.nan,
            }
        )
    return out


def mechanism_summary(summaries: dict[str, Any]) -> dict[str, Any]:
    l4 = summaries.get("L4_ar1_time_hetero", {})
    k4 = summaries.get("KOH_K4", {})
    k6 = summaries.get("KOH_K6", {})
    best = max(summaries.items(), key=lambda kv: kv[1].get("coverage", {}).get("0.5", -np.inf))
    width_moves = bool(
        np.isfinite(k4.get("ratio_sd_median", np.nan))
        and np.isfinite(l4.get("ratio_sd_median", np.nan))
        and k4["ratio_sd_median"] >= 0.8 * l4["ratio_sd_median"]
    )
    loc_moves = bool(
        abs(k4.get("signed_ratio_median_error", np.nan)) < abs(l4.get("signed_ratio_median_error", np.nan))
        or abs(k6.get("signed_ratio_median_error", np.nan)) < abs(l4.get("signed_ratio_median_error", np.nan))
    )
    pit_moves = bool(
        abs(k4.get("pit_mean", np.nan) - 0.5) < abs(l4.get("pit_mean", np.nan) - 0.5)
        or abs(k6.get("pit_mean", np.nan) - 0.5) < abs(l4.get("pit_mean", np.nan) - 0.5)
    )
    return {
        "best_coverage_50_variant": best[0],
        "width_moves_as_predicted": width_moves,
        "location_moves_toward_zero": loc_moves,
        "pit_mean_moves_toward_0_5": pit_moves,
        "both_width_and_location_move": bool(width_moves and loc_moves),
    }


def verdict_from_summaries(summaries: dict[str, Any], mechanism: dict[str, Any]) -> dict[str, str]:
    calibrated = [name for name, item in summaries.items() if name.startswith("KOH_") and item["calibrated"]]
    if calibrated:
        best = max(calibrated, key=lambda name: summaries[name]["coverage"]["0.5"])
        return {
            "verdict": "CALIBRATED",
            "best_variant": best,
            "message": "KOH posterior calibration cleared; posterior classification is unblocked and the next gate is posterior-mass classification on the full TEST set.",
        }
    l4 = summaries["L4_ar1_time_hetero"]
    best_koh_name, best_koh = max(
        ((k, v) for k, v in summaries.items() if k.startswith("KOH_")),
        key=lambda kv: abs(kv[1]["pit_mean"] - 0.5) * -1,
    )
    coverage_improves = best_koh["coverage"]["0.5"] >= l4["coverage"]["0.5"] + 0.05
    pit_improves = abs(best_koh["pit_mean"] - 0.5) < abs(l4["pit_mean"] - 0.5)
    if coverage_improves and pit_improves:
        return {
            "verdict": "PARTIAL",
            "best_variant": best_koh_name,
            "message": "KOH improved coverage and PIT location but did not clear; remaining defect is reported in the calibration table.",
        }
    return {
        "verdict": "NO-IMPROVEMENT",
        "best_variant": best_koh_name,
        "message": "KOH did not beat L4; additive time-only discrepancy is insufficient, so the next route is RoPE using NHANES ratio calibration.",
    }


def write_outputs(root: Path, result: dict[str, Any], rows: pd.DataFrame) -> None:
    out = root / "results"
    out.mkdir(exist_ok=True)
    lines = [
        "# Gate - Kennedy-O'Hagan Discrepancy, Cycle F",
        "",
        f"- VERDICT: {result['verdict']['verdict']}",
        f"- best variant: {result['verdict']['best_variant']}",
        f"- message: {result['verdict']['message']}",
        f"- posterior classification unblocked: {result['posterior_classification_unblocked']}",
        "",
        "## Step 0 - Residual Bug Fix",
        "",
        f"- residual scale reconciled: {result['step0']['residual_scale_reconciled']}",
        f"- reconciled residual scale: {result['step0']['reconciled_residual_scale_ml']:.2f} mL",
        f"- delta prior scale: {result['step0']['delta_prior_scale_l']:.6f} L",
        f"- bug explanation: {result['step0']['time_grid_alignment']}",
        f"- before summary: {result['step0']['before']}",
        f"- after summary: {result['step0']['after']}",
        f"- example curves: {result['step0']['examples']}",
        "",
        "## Step 1 - KOH Specification",
        "",
        f"- theta priors: {result['theta_priors']}",
        f"- KOH specification: {result['koh_spec']}",
        "",
        "## Step 2 - Knot Sensitivity / Step 3 - Calibration",
        "",
        "| variant | n conv | excluded | cov50 | cov80 | cov95 | PIT p | PIT mean | ratio sd med | signed ratio med err | sigma mL | rho | calibrated |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, item in result["variant_summaries"].items():
        lines.append(
            f"| {name} | {item['n_converged']} | {item['n_nonconverged_excluded']} | "
            f"{item['coverage']['0.5']:.3f} | {item['coverage']['0.8']:.3f} | {item['coverage']['0.95']:.3f} | "
            f"{item['pit_chisquare_p']:.3g} | {item['pit_mean']:.3f} | {item['ratio_sd_median']:.4f} | "
            f"{item['signed_ratio_median_error']:.4f} | {item.get('sigma_ml_median', np.nan):.1f} | "
            f"{item.get('rho_median', np.nan):.3f} | {item['calibrated']} |"
        )
    lines.extend(
        [
            "",
            f"- mechanism summary: {result['mechanism']}",
            "",
            "## Step 4 - Delta Shape",
            "",
            f"- delta shape finding: {result['delta_shape_finding']}",
            f"- delta table: {result['delta_shape_table']}",
            "",
            "## Step 5 - Identifiability Cost",
            "",
            f"- identifiability table: {result['identifiability_table']}",
            f"- interpretation: {result['identifiability_interpretation']}",
            "",
            "## NUTS Diagnostics",
            "",
            f"- diagnostics: {result['nuts_diagnostics']}",
            "",
            "## Verdict",
            "",
            result["verdict"]["message"],
            "",
        ]
    )
    (out / "gate_koh_discrepancy_F.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "gate_koh_discrepancy_F.json").write_text(json.dumps(to_jsonable(result), indent=2) + "\n", encoding="utf-8")
    rows.to_csv(out / "gate_koh_discrepancy_F_rows.csv", index=False)


def run(root: Path) -> dict[str, Any]:
    configure_caches(root)
    policy_rows, raw = load_rows_and_raw(root)
    residual_rows, step0 = step0_residual_diagnosis(policy_rows, raw)
    if not step0["residual_scale_reconciled"]:
        result = {
            "verdict": {
                "verdict": "HALTED",
                "best_variant": "none",
                "message": "Residual scale could not be reconciled after fixing nonuniform-grid integration; KOH was not run.",
            },
            "posterior_classification_unblocked": False,
            "step0": step0,
        }
        write_outputs(root, result, residual_rows)
        return result

    subset = build_subsets(policy_rows)
    variants = ["L0_fixed", "L4_ar1_time_hetero"] + [f"KOH_K{k}" for k in KOH_KNOTS]
    all_rows = []
    summaries = {}
    for variant in variants:
        part, summary = run_variant(variant, subset, raw, step0["delta_prior_scale_l"])
        all_rows.append(part)
        summaries[variant] = summary
    variant_rows = pd.concat(all_rows, ignore_index=True)
    residual_rows = residual_rows.copy()
    residual_rows["row_type"] = "residual_diagnosis"
    variant_rows["row_type"] = "nuts_variant"
    rows_out = pd.concat([residual_rows, variant_rows], ignore_index=True, sort=False)
    delta_table, delta_finding = delta_shape_table(variant_rows, "KOH_K4")
    ident_table = identifiability_table(variant_rows)
    sd_ratios = [row["sd_ratio_koh_over_l4"] for row in ident_table if np.isfinite(row["sd_ratio_koh_over_l4"])]
    ident_interp = (
        "KOH increases posterior parameter uncertainty versus L4; this is the expected theta-delta tradeoff."
        if sd_ratios and np.nanmedian(sd_ratios) > 1.10
        else "KOH did not materially increase median theta SD versus L4 in this run."
    )
    mechanism = mechanism_summary(summaries)
    verdict = verdict_from_summaries(summaries, mechanism)
    result = {
        "verdict": verdict,
        "posterior_classification_unblocked": bool(verdict["verdict"] == "CALIBRATED"),
        "sample": {
            "n_subset": int(len(subset)),
            "subset_counts": subset["truth_severity"].value_counts().to_dict(),
            "nested_100_count": int(subset["nested_prior_100"].sum()),
            "nested_100_counts": subset[subset["nested_prior_100"]]["truth_severity"].value_counts().to_dict(),
        },
        "nuts_settings": {
            "warmup": NUTS_WARMUP,
            "draws": NUTS_DRAWS,
            "chains": NUTS_CHAINS,
            "grid_stride": NUTS_GRID_STRIDE,
            "nominal_hz": float(100 / NUTS_GRID_STRIDE),
            "converged_rule": {"max_rhat_lte": CONVERGED_RHAT_MAX, "min_bulk_ess_gte": CONVERGED_ESS_MIN},
        },
        "theta_priors": natural_ranges_for_report(WIDE_PRIOR_RANGES),
        "koh_spec": {
            "delta_form": "cubic B-spline basis over each curve's expiratory window",
            "knot_sensitivity_internal_knots": list(KOH_KNOTS),
            "primary_k": 4,
            "delta_prior": f"delta coefficients ~ Normal(0, {step0['delta_prior_scale_l']:.6f} L)",
            "delta_zero_constraint": "basis matrix is centered as B(t)-B(0), so delta(0)=0 for every coefficient vector",
            "noise_model": "AR(1) innovation model with learned sigma and rho, matching L2 from the likelihood gate",
        },
        "step0": step0,
        "variant_summaries": summaries,
        "mechanism": mechanism,
        "delta_shape_table": delta_table,
        "delta_shape_finding": delta_finding,
        "identifiability_table": ident_table,
        "identifiability_interpretation": ident_interp,
        "nuts_diagnostics": {
            name: {
                "max_rhat_median": item.get("max_rhat_median"),
                "max_rhat_max": item.get("max_rhat_max"),
                "min_bulk_ess_median": item.get("min_bulk_ess_median"),
                "min_bulk_ess_min": item.get("min_bulk_ess_min"),
                "n_nonconverged_excluded": item.get("n_nonconverged_excluded"),
            }
            for name, item in summaries.items()
        },
    }
    write_outputs(root, result, rows_out)
    return result


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    print(f"KOH discrepancy gate: {result['verdict']['verdict']} - {result['verdict']['message']}")
    print(f"Best variant: {result['verdict']['best_variant']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
