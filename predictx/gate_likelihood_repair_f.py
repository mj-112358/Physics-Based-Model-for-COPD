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
from scipy.stats import chisquare

from .gate23_f import find_or_unzip_raw, read_raw
from .gate_clinical_classification_f import RANDOM_SEED, extract_curve_for_fit
from .gate_posterior_calibration_repair_f import (
    PIT_BINS,
    PIT_P_THRESHOLD,
    WIDE_PRIOR_RANGES,
    bulk_ess,
    natural_ranges_for_report,
    ratio_samples_from_phi,
    split_rhat,
    to_jsonable,
)
from .ombe import OmbeGrid, ombe_volume, phi_from_natural

COVERAGE_LEVELS = (0.50, 0.80, 0.95)
FIXED_SIGMA_L = 0.050
SUBSET_N = 150
NESTED_N = 100
NUTS_WARMUP = 250
NUTS_DRAWS = 250
NUTS_CHAINS = 2
NUTS_GRID_STRIDE = 5
CONVERGED_RHAT_MAX = 1.10
CONVERGED_ESS_MIN = 50.0
PARAM_NAMES = ("tau_f", "tau_s", "alpha", "fvc_l", "tau_o")
VARIANTS = ("L0_baseline", "L1_learned_sigma", "L2_sigma_ar1", "L3_student_t")
VARIANT_SEED_OFFSETS = {
    "L0_baseline": 10_000,
    "L1_learned_sigma": 20_000,
    "L2_sigma_ar1": 30_000,
    "L3_student_t": 40_000,
    "L4_ar1_time_hetero": 50_000,
}


def configure_caches(root: Path) -> None:
    cache = root / ".cache"
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))


def q(values: list[float] | np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return [float("nan"), float("nan")]
    return [float(x) for x in np.percentile(arr, [25, 75])]


def phi_from_row(row: Any) -> jnp.ndarray:
    alpha = float(row.fit_alpha)
    alpha = min(max(alpha, 1e-6), 1.0 - 1e-6)
    return phi_from_natural(
        jnp.asarray(float(row.fit_tau_f)),
        jnp.asarray(float(row.fit_tau_s)),
        jnp.asarray(alpha),
        jnp.asarray(float(row.fit_fvc_l)),
        jnp.asarray(float(row.fit_tau_o)),
    )


def residual_metrics_for_row(raw_row: Any, fit_row: Any) -> dict[str, float]:
    curve = extract_curve_for_fit(raw_row)
    t = np.asarray(curve["vol_t"], dtype=np.float32)
    y = np.asarray(curve["vol_l"], dtype=np.float32)
    pred = np.asarray(ombe_volume(phi_from_row(fit_row), jnp.asarray(t)))
    resid_l = y - pred
    resid_ml = resid_l * 1000.0
    finite = np.isfinite(resid_ml)
    resid_ml = resid_ml[finite]
    if len(resid_ml) < 3:
        raise ValueError("too few residual points")
    centered = resid_ml - np.mean(resid_ml)
    denom = float(np.dot(centered, centered))
    lag1 = float(np.dot(centered[:-1], centered[1:]) / denom) if denom > 0 else np.nan
    dw_denom = float(np.dot(resid_ml, resid_ml))
    durbin_watson = float(np.sum(np.diff(resid_ml) ** 2) / dw_denom) if dw_denom > 0 else np.nan
    rms_ml = float(np.sqrt(np.mean(resid_ml**2)))
    return {
        "lag1_autocorr": lag1,
        "durbin_watson": durbin_watson,
        "residual_rms_ml": rms_ml,
        "n_points": int(len(resid_ml)),
    }


def summarize_metric(values: pd.Series | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return {
        "n": int(len(arr)),
        "median": float(np.median(arr)) if len(arr) else np.nan,
        "iqr": q(arr),
    }


def summarize_residuals(rows: pd.DataFrame) -> dict[str, Any]:
    def block(df: pd.DataFrame) -> dict[str, Any]:
        return {
            "n": int(len(df)),
            "lag1_autocorr": summarize_metric(df["lag1_autocorr"]),
            "durbin_watson": summarize_metric(df["durbin_watson"]),
            "residual_rms_ml": summarize_metric(df["residual_rms_ml"]),
        }

    out = {
        "all": block(rows),
        "obstructed_truth": block(rows[rows["fev1_fvc_ratio"] < 0.70]),
        "normal_truth": block(rows[rows["fev1_fvc_ratio"] >= 0.70]),
        "assumed_fixed_sigma_ml": FIXED_SIGMA_L * 1000.0,
    }
    out["iid_gaussian_falsified"] = bool(
        out["all"]["lag1_autocorr"]["median"] > 0.5
        and out["all"]["residual_rms_ml"]["median"] > FIXED_SIGMA_L * 1000.0 * 1.25
    )
    return out


def load_rows_and_raw(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = root / "results" / "gate_policy_calibrated_flag_F_rows.csv"
    if not policy.exists():
        raise RuntimeError(f"Missing required policy rows: {policy}")
    rows = pd.read_csv(policy)
    needed = {"raw_index", "split", "truth_severity", "fev1_fvc_ratio", "fit_tau_f", "fit_tau_s", "fit_alpha", "fit_fvc_l", "fit_tau_o"}
    missing = sorted(needed - set(rows.columns))
    if missing:
        raise RuntimeError(f"Policy rows missing required columns: {missing}")
    raw_path, _ = find_or_unzip_raw(root)
    raw, _ = read_raw(raw_path)
    return rows, raw


def build_subsets(rows: pd.DataFrame) -> pd.DataFrame:
    test = rows[(rows["split"] == "test") & rows["fev1_fvc_ratio"].notna()].copy()
    nested = test.sample(n=min(NESTED_N, len(test)), random_state=RANDOM_SEED + 44_000).copy()
    nested_indices = set(nested.index)
    targets = proportional_targets(test["truth_severity"].value_counts().to_dict(), SUBSET_N)
    chosen = nested.copy()
    rng_seed = RANDOM_SEED + 55_000
    for stratum, target in targets.items():
        have = int((chosen["truth_severity"] == stratum).sum())
        need = max(0, target - have)
        if need == 0:
            continue
        pool = test[(test["truth_severity"] == stratum) & (~test.index.isin(chosen.index))]
        if len(pool) == 0:
            continue
        chosen = pd.concat([chosen, pool.sample(n=min(need, len(pool)), random_state=rng_seed + len(chosen))])
    if len(chosen) < SUBSET_N:
        pool = test[~test.index.isin(chosen.index)]
        chosen = pd.concat([chosen, pool.sample(n=min(SUBSET_N - len(chosen), len(pool)), random_state=rng_seed + 99)])
    if len(chosen) > SUBSET_N:
        extras = chosen[~chosen.index.isin(nested_indices)]
        keep_extra = extras.sample(n=max(0, SUBSET_N - len(nested)), random_state=rng_seed + 199)
        chosen = pd.concat([nested, keep_extra])
    chosen = chosen.copy()
    chosen["nested_prior_100"] = chosen.index.isin(nested_indices)
    chosen["likelihood_subset_150"] = True
    return chosen.sort_values(["truth_severity", "SEQN", "raw_index"]).reset_index(drop=False).rename(columns={"index": "source_row_index"})


def proportional_targets(counts: dict[str, int], total: int) -> dict[str, int]:
    order = ["severe", "moderate", "mild_0.60_0.70", "normal"]
    denom = sum(counts.get(k, 0) for k in order)
    raw = {k: total * counts.get(k, 0) / denom for k in order}
    floors = {k: int(np.floor(v)) for k, v in raw.items()}
    remainder = total - sum(floors.values())
    frac_order = sorted(order, key=lambda k: raw[k] - floors[k], reverse=True)
    for k in frac_order[:remainder]:
        floors[k] += 1
    return floors


def ombe_likelihood_model(t: jnp.ndarray, y: jnp.ndarray, variant: str) -> None:
    tau_f = numpyro.sample("tau_f", dist.Uniform(*WIDE_PRIOR_RANGES["tau_f"]))
    tau_s = numpyro.sample("tau_s", dist.Uniform(*WIDE_PRIOR_RANGES["tau_s"]))
    alpha = numpyro.sample("alpha", dist.Uniform(*WIDE_PRIOR_RANGES["alpha"]))
    fvc_l = numpyro.sample("fvc_l", dist.Uniform(*WIDE_PRIOR_RANGES["fvc_l"]))
    tau_o = numpyro.sample("tau_o", dist.Uniform(*WIDE_PRIOR_RANGES["tau_o"]))
    phi = phi_from_natural(tau_f, tau_s, alpha, fvc_l, tau_o)
    mu = ombe_volume(phi, t)
    resid = y - mu
    if variant == "L0_baseline":
        numpyro.sample("y", dist.Normal(mu, FIXED_SIGMA_L), obs=y)
    elif variant == "L1_learned_sigma":
        sigma = numpyro.sample("sigma_l", dist.HalfNormal(0.50))
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
    elif variant == "L2_sigma_ar1":
        sigma = numpyro.sample("sigma_l", dist.HalfNormal(0.50))
        rho = numpyro.sample("rho", dist.Uniform(0.0, 0.99))
        init_sd = sigma / jnp.sqrt(jnp.maximum(1.0 - rho**2, 1e-4))
        numpyro.sample("e0", dist.Normal(0.0, init_sd), obs=resid[0])
        numpyro.sample("e_ar", dist.Normal(rho * resid[:-1], sigma), obs=resid[1:])
    elif variant == "L3_student_t":
        sigma = numpyro.sample("sigma_l", dist.HalfNormal(0.50))
        nu = numpyro.sample("nu_minus_2", dist.Exponential(1.0 / 10.0)) + 2.0
        numpyro.sample("y", dist.StudentT(nu, mu, sigma), obs=y)
    elif variant == "L4_ar1_time_hetero":
        sigma = numpyro.sample("sigma_l", dist.HalfNormal(0.50))
        rho = numpyro.sample("rho", dist.Uniform(0.0, 0.99))
        beta_time = numpyro.sample("beta_time", dist.HalfNormal(2.0))
        scaled_t = t / jnp.maximum(t[-1], 1e-3)
        sigma_t = sigma * (1.0 + beta_time * scaled_t)
        init_sd = sigma_t[0] / jnp.sqrt(jnp.maximum(1.0 - rho**2, 1e-4))
        numpyro.sample("e0", dist.Normal(0.0, init_sd), obs=resid[0])
        numpyro.sample("e_ar_hetero", dist.Normal(rho * resid[:-1], sigma_t[1:]), obs=resid[1:])
    else:
        raise ValueError(f"Unknown likelihood variant: {variant}")


def samples_to_phi(samples: dict[str, np.ndarray]) -> np.ndarray:
    tau_f = np.ravel(samples["tau_f"])
    tau_s = np.ravel(samples["tau_s"])
    alpha = np.clip(np.ravel(samples["alpha"]), 1e-6, 1.0 - 1e-6)
    fvc_l = np.ravel(samples["fvc_l"])
    tau_o = np.ravel(samples["tau_o"])
    return np.column_stack([np.log(tau_f), np.log(tau_s), np.log(alpha / (1.0 - alpha)), np.log(fvc_l), np.log(tau_o)]).astype(np.float32)


def mcmc_diagnostics(mcmc: MCMC) -> dict[str, Any]:
    samples = mcmc.get_samples(group_by_chain=True)
    rhat_by_param = {}
    ess_by_param = {}
    for name in PARAM_NAMES:
        arr = np.asarray(samples[name], dtype=float)
        rhat_by_param[name] = split_rhat(arr)
        ess_by_param[name] = bulk_ess(arr)
    rhat_vals = np.array(list(rhat_by_param.values()), dtype=float)
    ess_vals = np.array(list(ess_by_param.values()), dtype=float)
    max_rhat = float(np.nanmax(rhat_vals)) if np.isfinite(rhat_vals).any() else np.nan
    min_ess = float(np.nanmin(ess_vals)) if np.isfinite(ess_vals).any() else np.nan
    return {
        "max_rhat": max_rhat,
        "min_bulk_ess": min_ess,
        "rhat_by_param": rhat_by_param,
        "bulk_ess_by_param": ess_by_param,
        "converged": bool(max_rhat <= CONVERGED_RHAT_MAX and min_ess >= CONVERGED_ESS_MIN),
    }


def posterior_extra_summary(samples: dict[str, np.ndarray], variant: str) -> dict[str, float]:
    out = {}
    if "sigma_l" in samples:
        out["sigma_l_median"] = float(np.median(np.ravel(samples["sigma_l"])))
        out["sigma_ml_median"] = out["sigma_l_median"] * 1000.0
    if "rho" in samples:
        out["rho_median"] = float(np.median(np.ravel(samples["rho"])))
    if "nu_minus_2" in samples:
        out["nu_median"] = float(np.median(np.ravel(samples["nu_minus_2"]) + 2.0))
    if "beta_time" in samples:
        out["beta_time_median"] = float(np.median(np.ravel(samples["beta_time"])))
    if variant == "L0_baseline":
        out["sigma_l_median"] = FIXED_SIGMA_L
        out["sigma_ml_median"] = FIXED_SIGMA_L * 1000.0
    return out


def run_variant(variant: str, subset: pd.DataFrame, raw: pd.DataFrame, t_full: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_by_index = {int(idx): row for idx, row in raw.iterrows()}
    rows_out = []
    start = time.perf_counter()
    for i, row in enumerate(subset.itertuples(index=False), start=1):
        raw_idx = int(row.raw_index)
        base = {
            "variant": variant,
            "SEQN": row.SEQN,
            "raw_index": raw_idx,
            "truth_severity": row.truth_severity,
            "truth_ratio": float(row.fev1_fvc_ratio),
            "nested_prior_100": bool(row.nested_prior_100),
            "likelihood_subset_150": bool(row.likelihood_subset_150),
        }
        try:
            curve = extract_curve_for_fit(raw_by_index[raw_idx])
            t_obs = np.asarray(curve["vol_t"], dtype=np.float32)[::NUTS_GRID_STRIDE]
            y_obs = np.asarray(curve["vol_l"], dtype=np.float32)[::NUTS_GRID_STRIDE]
            if len(t_obs) < 20:
                raise ValueError("too few downsampled volume points")
            kernel = NUTS(lambda y=None: ombe_likelihood_model(jnp.asarray(t_obs), y, variant))
            mcmc = MCMC(
                kernel,
                num_warmup=NUTS_WARMUP,
                num_samples=NUTS_DRAWS,
                num_chains=NUTS_CHAINS,
                chain_method="sequential",
                progress_bar=False,
            )
            mcmc.run(jax.random.PRNGKey(RANDOM_SEED + raw_idx + VARIANT_SEED_OFFSETS[variant]), y=jnp.asarray(y_obs))
            samples_chain = mcmc.get_samples(group_by_chain=True)
            diag = mcmc_diagnostics(mcmc)
            flat = {name: np.asarray(samples_chain[name]).reshape(-1) for name in samples_chain}
            phi = samples_to_phi(flat)
            ratio = ratio_samples_from_phi(phi, t_full)
            ratio = ratio[np.isfinite(ratio)]
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
                    "pit": pit,
                    **diag,
                    **posterior_extra_summary(flat, variant),
                }
            )
            for level in COVERAGE_LEVELS:
                lo = (1.0 - level) / 2.0
                hi = 1.0 - lo
                qlo, qhi = np.quantile(ratio, [lo, hi]) if len(ratio) else (np.nan, np.nan)
                base[f"covered_{int(level * 100)}"] = bool(qlo <= float(row.fev1_fvc_ratio) <= qhi)
        except Exception as exc:
            base.update({"fit_error": str(exc), "converged": False})
        rows_out.append(base)
        if i % 10 == 0 or i == len(subset):
            print(f"{variant}: NUTS progress {i}/{len(subset)}")
    rows = pd.DataFrame(rows_out)
    summary = summarize_variant(rows, nested_only=False)
    summary["nested_100"] = summarize_variant(rows[rows["nested_prior_100"]], nested_only=True)
    summary["wall_clock_s"] = float(time.perf_counter() - start)
    return rows, summary


def summarize_variant(rows: pd.DataFrame, nested_only: bool) -> dict[str, Any]:
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
        "nested_only": nested_only,
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
            len(ok) > 0
            and all(abs(coverage[str(level)] - level) <= 0.07 for level in COVERAGE_LEVELS)
            and not gross_skew
        ),
        "ratio_sd_median": float(np.nanmedian(ok["ratio_sd"])) if len(ok) else np.nan,
        "sigma_ml_median": float(np.nanmedian(ok["sigma_ml_median"])) if len(ok) and "sigma_ml_median" in ok else np.nan,
        "rho_median": float(np.nanmedian(ok["rho_median"])) if len(ok) and "rho_median" in ok else np.nan,
        "nu_median": float(np.nanmedian(ok["nu_median"])) if len(ok) and "nu_median" in ok else np.nan,
        "beta_time_median": float(np.nanmedian(ok["beta_time_median"])) if len(ok) and "beta_time_median" in ok else np.nan,
        "max_rhat_median": float(np.nanmedian(ok["max_rhat"])) if len(ok) else np.nan,
        "max_rhat_max": float(np.nanmax(ok["max_rhat"])) if len(ok) else np.nan,
        "min_bulk_ess_median": float(np.nanmedian(ok["min_bulk_ess"])) if len(ok) else np.nan,
        "min_bulk_ess_min": float(np.nanmin(ok["min_bulk_ess"])) if len(ok) else np.nan,
    }


def residual_diagnosis(rows: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_by_index = {int(idx): row for idx, row in raw.iterrows()}
    out = []
    usable = rows[(rows["fit_error"].fillna("") == "") & (rows["volume_fit_success"].fillna(False).astype(bool))].copy()
    for i, row in enumerate(usable.itertuples(index=False), start=1):
        base = {
            "SEQN": row.SEQN,
            "raw_index": int(row.raw_index),
            "split": row.split,
            "truth_severity": row.truth_severity,
            "fev1_fvc_ratio": float(row.fev1_fvc_ratio),
            "truth_obstructed": bool(float(row.fev1_fvc_ratio) < 0.70),
        }
        try:
            base.update(residual_metrics_for_row(raw_by_index[int(row.raw_index)], row))
            base["residual_error"] = ""
        except Exception as exc:
            base["residual_error"] = str(exc)
        out.append(base)
        if i % 250 == 0 or i == len(usable):
            print(f"Residual diagnosis progress: {i}/{len(usable)}")
    residual_rows = pd.DataFrame(out)
    good = residual_rows[residual_rows["residual_error"].fillna("") == ""].copy()
    return residual_rows, summarize_residuals(good)


def should_run_l4(summaries: dict[str, Any]) -> bool:
    l0 = summaries["L0_baseline"]
    l2 = summaries["L2_sigma_ar1"]
    if l2["calibrated"]:
        return False
    return bool(l2["coverage"]["0.5"] >= l0["coverage"]["0.5"] + 0.05)


def mechanism_confirmation(summaries: dict[str, Any]) -> dict[str, Any]:
    table = []
    for name, summary in summaries.items():
        table.append(
            {
                "variant": name,
                "sigma_ml_median": summary.get("sigma_ml_median"),
                "rho_median": summary.get("rho_median"),
                "ratio_sd_median": summary.get("ratio_sd_median"),
                "coverage_50": summary["coverage"].get("0.5"),
            }
        )
    ordered = [row for row in table if row["variant"] in summaries]
    l0 = summaries.get("L0_baseline", {})
    best = max(summaries.items(), key=lambda kv: kv[1]["coverage"].get("0.5", -np.inf))
    sd_rises = any(
        np.isfinite(s.get("ratio_sd_median", np.nan))
        and np.isfinite(l0.get("ratio_sd_median", np.nan))
        and s["ratio_sd_median"] > l0["ratio_sd_median"]
        for s in summaries.values()
    )
    cov_rises = any(
        s["coverage"].get("0.5", np.nan) > l0.get("coverage", {}).get("0.5", np.nan) + 0.05
        for s in summaries.values()
    )
    return {
        "table": ordered,
        "best_50_coverage_variant": best[0],
        "posterior_sd_rises_vs_l0": bool(sd_rises),
        "coverage_50_materially_improves_vs_l0": bool(cov_rises),
        "pattern_holds": bool(sd_rises and cov_rises),
    }


def verdict_from_summaries(summaries: dict[str, Any], mechanism: dict[str, Any]) -> dict[str, str]:
    calibrated = [name for name, summary in summaries.items() if summary["calibrated"]]
    if calibrated:
        best = max(calibrated, key=lambda name: summaries[name]["coverage"]["0.5"])
        return {
            "verdict": "CALIBRATED",
            "best_variant": best,
            "message": "Posterior classification is unblocked; next gate is posterior-mass classification on the full TEST set.",
        }
    l0_50 = summaries["L0_baseline"]["coverage"]["0.5"]
    best_name, best_summary = max(summaries.items(), key=lambda kv: kv[1]["coverage"].get("0.5", -np.inf))
    if best_summary["coverage"]["0.5"] >= l0_50 + 0.05:
        return {
            "verdict": "PARTIAL",
            "best_variant": best_name,
            "message": "Coverage improved but did not clear; next step is a Kennedy-O'Hagan discrepancy term over expiratory time, not further noise-only tweaks.",
        }
    return {
        "verdict": "NO-IMPROVEMENT",
        "best_variant": best_name,
        "message": "No likelihood variant materially improved calibration; miscalibration is structural/model-discrepancy dominated.",
    }


def write_outputs(root: Path, result: dict[str, Any], rows: pd.DataFrame) -> None:
    out = root / "results"
    out.mkdir(exist_ok=True)
    lines = [
        "# Gate - Likelihood Repair, Cycle F",
        "",
        f"- VERDICT: {result['verdict']['verdict']}",
        f"- best variant: {result['verdict']['best_variant']}",
        f"- message: {result['verdict']['message']}",
        f"- posterior classification unblocked: {result['posterior_classification_unblocked']}",
        "",
        "## Sample",
        "",
        f"- TEST subset n: {result['sample']['n_subset']}",
        f"- subset counts: {result['sample']['subset_counts']}",
        f"- nested prior-gate 100 present: {result['sample']['nested_100_count']}",
        f"- NUTS settings: {result['nuts_settings']}",
        f"- widened priors: {result['widened_priors']}",
        "",
        "## Step 0 - Residual Structure",
        "",
        f"- diagnosis: {result['residual_diagnosis']['iid_gaussian_falsified']}",
        f"- residual summary: {result['residual_diagnosis']}",
        "",
        "## Step 1 - Likelihood Variants",
        "",
        "| variant | n converged | excluded | cov50 | cov80 | cov95 | PIT p | PIT mean | ratio sd median | sigma mL median | rho median | calibrated |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, summary in result["variant_summaries"].items():
        lines.append(
            f"| {name} | {summary['n_converged']} | {summary['n_nonconverged_excluded']} | "
            f"{summary['coverage']['0.5']:.3f} | {summary['coverage']['0.8']:.3f} | {summary['coverage']['0.95']:.3f} | "
            f"{summary['pit_chisquare_p']:.3g} | {summary['pit_mean']:.3f} | {summary['ratio_sd_median']:.4f} | "
            f"{summary.get('sigma_ml_median', np.nan):.1f} | {summary.get('rho_median', np.nan):.3f} | {summary['calibrated']} |"
        )
    lines.extend(
        [
            "",
            "## Nested 100 Comparison Set",
            "",
            f"- nested summaries: {result['nested_100_summaries']}",
            "",
            "## Step 2 - Mechanism Confirmation",
            "",
            f"- mechanism table: {result['mechanism_confirmation']['table']}",
            f"- pattern holds: {result['mechanism_confirmation']['pattern_holds']}",
            "",
            "## NUTS Diagnostics",
            "",
            f"- diagnostics by variant: {result['nuts_diagnostics']}",
            "",
            "## Step 3 - Verdict",
            "",
            result["verdict"]["message"],
            "",
        ]
    )
    (out / "gate_likelihood_repair_F.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "gate_likelihood_repair_F.json").write_text(json.dumps(to_jsonable(result), indent=2) + "\n", encoding="utf-8")
    rows.to_csv(out / "gate_likelihood_repair_F_rows.csv", index=False)


def rebuild_from_existing_rows(root: Path) -> dict[str, Any]:
    path = root / "results" / "gate_likelihood_repair_F_rows.csv"
    if not path.exists():
        raise RuntimeError(f"Missing existing likelihood repair rows: {path}")
    rows_out = pd.read_csv(path)
    residual_rows = rows_out[rows_out["row_type"] == "residual_diagnosis"].copy()
    variant_rows = rows_out[rows_out["row_type"] == "nuts_variant"].copy()
    residual_summary = summarize_residuals(residual_rows[residual_rows["residual_error"].fillna("") == ""])
    summaries = {}
    for variant, group in variant_rows.groupby("variant", dropna=False):
        summaries[str(variant)] = summarize_variant(group, nested_only=False)
        summaries[str(variant)]["nested_100"] = summarize_variant(group[group["nested_prior_100"].fillna(False).astype(bool)], nested_only=True)
    mechanism = mechanism_confirmation(summaries)
    verdict = verdict_from_summaries(summaries, mechanism)
    first_variant = variant_rows[variant_rows["variant"] == sorted(summaries.keys())[0]].copy()
    result = {
        "verdict": verdict,
        "posterior_classification_unblocked": bool(verdict["verdict"] == "CALIBRATED"),
        "sample": {
            "n_subset": int(len(first_variant)),
            "subset_counts": first_variant["truth_severity"].value_counts().to_dict(),
            "nested_100_count": int(first_variant["nested_prior_100"].fillna(False).astype(bool).sum()),
            "nested_100_counts": first_variant[first_variant["nested_prior_100"].fillna(False).astype(bool)]["truth_severity"].value_counts().to_dict(),
            "target_subset_n": SUBSET_N,
        },
        "nuts_settings": {
            "warmup": NUTS_WARMUP,
            "draws": NUTS_DRAWS,
            "chains": NUTS_CHAINS,
            "grid_stride": NUTS_GRID_STRIDE,
            "nominal_hz": float(100 / NUTS_GRID_STRIDE),
            "converged_rule": {"max_rhat_lte": CONVERGED_RHAT_MAX, "min_bulk_ess_gte": CONVERGED_ESS_MIN},
        },
        "widened_priors": natural_ranges_for_report(WIDE_PRIOR_RANGES),
        "residual_diagnosis": residual_summary,
        "variant_summaries": summaries,
        "nested_100_summaries": {name: summary.get("nested_100", {}) for name, summary in summaries.items()},
        "mechanism_confirmation": mechanism,
        "nuts_diagnostics": {
            name: {
                "max_rhat_median": summary.get("max_rhat_median"),
                "max_rhat_max": summary.get("max_rhat_max"),
                "min_bulk_ess_median": summary.get("min_bulk_ess_median"),
                "min_bulk_ess_min": summary.get("min_bulk_ess_min"),
                "n_nonconverged_excluded": summary.get("n_nonconverged_excluded"),
            }
            for name, summary in summaries.items()
        },
    }
    write_outputs(root, result, rows_out)
    return result


def run(root: Path) -> dict[str, Any]:
    configure_caches(root)
    rows, raw = load_rows_and_raw(root)
    residual_rows, residual_summary = residual_diagnosis(rows, raw)
    if not residual_summary["iid_gaussian_falsified"]:
        result = {
            "verdict": {"verdict": "HALTED", "best_variant": "none", "message": "Residual structure did not falsify the i.i.d. Gaussian diagnosis; likelihood repair not run."},
            "posterior_classification_unblocked": False,
            "residual_diagnosis": residual_summary,
        }
        write_outputs(root, result, residual_rows)
        return result

    subset = build_subsets(rows)
    t_full = np.asarray(OmbeGrid().t, dtype=np.float32)
    variant_rows = []
    summaries: dict[str, Any] = {}
    run_order = list(VARIANTS)
    for variant in run_order:
        part, summary = run_variant(variant, subset, raw, t_full)
        variant_rows.append(part)
        summaries[variant] = summary
    if should_run_l4(summaries):
        part, summary = run_variant("L4_ar1_time_hetero", subset, raw, t_full)
        variant_rows.append(part)
        summaries["L4_ar1_time_hetero"] = summary
    else:
        summaries["L4_ar1_time_hetero"] = {
            "status": "NOT-RUN",
            "reason": "L2 did not materially improve 50% coverage or already cleared; L4 condition was not met.",
            "coverage": {"0.5": np.nan, "0.8": np.nan, "0.95": np.nan},
            "calibrated": False,
            "n_converged": 0,
            "n_nonconverged_excluded": 0,
            "pit_chisquare_p": np.nan,
            "pit_mean": np.nan,
            "ratio_sd_median": np.nan,
            "sigma_ml_median": np.nan,
            "rho_median": np.nan,
            "nested_100": {},
        }
    all_variant_rows = pd.concat(variant_rows, ignore_index=True)
    residual_subset = residual_rows.merge(subset[["raw_index", "nested_prior_100", "likelihood_subset_150"]], on="raw_index", how="left")
    residual_subset["row_type"] = "residual_diagnosis"
    all_variant_rows["row_type"] = "nuts_variant"
    rows_out = pd.concat([residual_subset, all_variant_rows], ignore_index=True, sort=False)
    mechanism = mechanism_confirmation({k: v for k, v in summaries.items() if v.get("status") != "NOT-RUN"})
    verdict = verdict_from_summaries({k: v for k, v in summaries.items() if v.get("status") != "NOT-RUN"}, mechanism)
    result = {
        "verdict": verdict,
        "posterior_classification_unblocked": bool(verdict["verdict"] == "CALIBRATED"),
        "sample": {
            "n_subset": int(len(subset)),
            "subset_counts": subset["truth_severity"].value_counts().to_dict(),
            "nested_100_count": int(subset["nested_prior_100"].sum()),
            "nested_100_counts": subset[subset["nested_prior_100"]]["truth_severity"].value_counts().to_dict(),
            "target_subset_n": SUBSET_N,
        },
        "nuts_settings": {
            "warmup": NUTS_WARMUP,
            "draws": NUTS_DRAWS,
            "chains": NUTS_CHAINS,
            "grid_stride": NUTS_GRID_STRIDE,
            "nominal_hz": float(100 / NUTS_GRID_STRIDE),
            "converged_rule": {"max_rhat_lte": CONVERGED_RHAT_MAX, "min_bulk_ess_gte": CONVERGED_ESS_MIN},
        },
        "widened_priors": natural_ranges_for_report(WIDE_PRIOR_RANGES),
        "residual_diagnosis": residual_summary,
        "variant_summaries": summaries,
        "nested_100_summaries": {name: summary.get("nested_100", {}) for name, summary in summaries.items()},
        "mechanism_confirmation": mechanism,
        "nuts_diagnostics": {
            name: {
                "max_rhat_median": summary.get("max_rhat_median"),
                "max_rhat_max": summary.get("max_rhat_max"),
                "min_bulk_ess_median": summary.get("min_bulk_ess_median"),
                "min_bulk_ess_min": summary.get("min_bulk_ess_min"),
                "n_nonconverged_excluded": summary.get("n_nonconverged_excluded"),
            }
            for name, summary in summaries.items()
        },
    }
    write_outputs(root, result, rows_out)
    return result


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--reuse-existing-rows", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result = rebuild_from_existing_rows(root) if args.reuse_existing_rows else run(root)
    print(f"Likelihood repair gate: {result['verdict']['verdict']} - {result['verdict']['message']}")
    print(f"Best variant: {result['verdict']['best_variant']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
