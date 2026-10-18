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
from .gate_likelihood_repair_f import (
    CONVERGED_ESS_MIN,
    CONVERGED_RHAT_MAX,
    COVERAGE_LEVELS,
    NUTS_CHAINS,
    NUTS_DRAWS,
    NUTS_GRID_STRIDE,
    NUTS_WARMUP,
    WIDE_PRIOR_RANGES,
    build_subsets,
    mcmc_diagnostics,
    natural_ranges_for_report,
    samples_to_phi,
)
from .gate_posterior_calibration_repair_f import PIT_BINS, PIT_P_THRESHOLD, ratio_samples_from_phi, to_jsonable
from .ombe import OmbeGrid, ombe_volume, phi_from_natural

OUTPUT_STEM = "gate_posterior_width_F"
CHECKPOINT_NAME = "_gate_posterior_width_F_rows_checkpoint.csv"
THIN_KS = (1, 5, 10, 20, 50)
MAX_GP_POINTS = 80
VARIANT_SEED_OFFSETS = {
    "L4_THIN_k1": 61_000,
    "L4_THIN_k5": 62_000,
    "L4_THIN_k10": 63_000,
    "L4_THIN_k20": 64_000,
    "L4_THIN_k50": 65_000,
    "M1_AR2": 66_000,
    "M2_GP_MATERN32": 67_000,
}


def configure_caches(root: Path) -> None:
    cache = root / ".cache"
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))


def load_policy_and_subset(root: Path) -> pd.DataFrame:
    existing = root / "results" / "gate_likelihood_repair_F_rows.csv"
    if existing.exists():
        rows = pd.read_csv(existing)
        subset = rows[rows["row_type"].eq("nuts_variant") & rows["variant"].eq("L4_ar1_time_hetero")].copy()
        if len(subset):
            keep = [
                "SEQN",
                "raw_index",
                "truth_severity",
                "truth_ratio",
                "nested_prior_100",
                "likelihood_subset_150",
            ]
            out = subset[keep].drop_duplicates("raw_index").rename(columns={"truth_ratio": "fev1_fvc_ratio"}).copy()
            out["source_row_index"] = np.arange(len(out))
            return out.sort_values(["truth_severity", "SEQN", "raw_index"]).reset_index(drop=True)
    policy = root / "results" / "gate_policy_calibrated_flag_F_rows.csv"
    if not policy.exists():
        raise RuntimeError(f"Missing policy rows: {policy}")
    return build_subsets(pd.read_csv(policy))


def phi_prior() -> tuple[jnp.ndarray, jnp.ndarray]:
    lo = jnp.asarray(
        [
            WIDE_PRIOR_RANGES["tau_f"][0],
            WIDE_PRIOR_RANGES["tau_s"][0],
            WIDE_PRIOR_RANGES["alpha"][0],
            WIDE_PRIOR_RANGES["fvc_l"][0],
            WIDE_PRIOR_RANGES["tau_o"][0],
        ]
    )
    hi = jnp.asarray(
        [
            WIDE_PRIOR_RANGES["tau_f"][1],
            WIDE_PRIOR_RANGES["tau_s"][1],
            WIDE_PRIOR_RANGES["alpha"][1],
            WIDE_PRIOR_RANGES["fvc_l"][1],
            WIDE_PRIOR_RANGES["tau_o"][1],
        ]
    )
    return lo, hi


def sample_ombe_params() -> jnp.ndarray:
    tau_f = numpyro.sample("tau_f", dist.Uniform(*WIDE_PRIOR_RANGES["tau_f"]))
    tau_s = numpyro.sample("tau_s", dist.Uniform(*WIDE_PRIOR_RANGES["tau_s"]))
    alpha = numpyro.sample("alpha", dist.Uniform(*WIDE_PRIOR_RANGES["alpha"]))
    fvc_l = numpyro.sample("fvc_l", dist.Uniform(*WIDE_PRIOR_RANGES["fvc_l"]))
    tau_o = numpyro.sample("tau_o", dist.Uniform(*WIDE_PRIOR_RANGES["tau_o"]))
    return phi_from_natural(tau_f, tau_s, alpha, fvc_l, tau_o)


def model_l4_ar1_time_hetero(t: jnp.ndarray, y: jnp.ndarray) -> None:
    phi = sample_ombe_params()
    mu = ombe_volume(phi, t)
    resid = y - mu
    sigma = numpyro.sample("sigma_l", dist.HalfNormal(0.50))
    rho = numpyro.sample("rho", dist.Uniform(0.0, 0.995))
    beta_time = numpyro.sample("beta_time", dist.HalfNormal(2.0))
    scaled_t = t / jnp.maximum(t[-1], 1e-3)
    sigma_t = sigma * (1.0 + beta_time * scaled_t)
    init_sd = sigma_t[0] / jnp.sqrt(jnp.maximum(1.0 - rho**2, 1e-4))
    numpyro.sample("e0", dist.Normal(0.0, init_sd), obs=resid[0])
    numpyro.sample("e_ar_hetero", dist.Normal(rho * resid[:-1], sigma_t[1:]), obs=resid[1:])


def model_ar2(t: jnp.ndarray, y: jnp.ndarray) -> None:
    phi = sample_ombe_params()
    mu = ombe_volume(phi, t)
    resid = y - mu
    sigma = numpyro.sample("sigma_l", dist.HalfNormal(0.50))
    pacf1 = numpyro.sample("pacf1", dist.Uniform(-0.99, 0.99))
    pacf2 = numpyro.sample("pacf2", dist.Uniform(-0.99, 0.99))
    rho1 = numpyro.deterministic("rho1", pacf1 * (1.0 - pacf2))
    rho2 = numpyro.deterministic("rho2", pacf2)
    init_sd = sigma / jnp.sqrt(jnp.maximum(1.0 - pacf1**2, 1e-4))
    numpyro.sample("e0", dist.Normal(0.0, init_sd), obs=resid[0])
    numpyro.sample("e1", dist.Normal(rho1 * resid[0], init_sd), obs=resid[1])
    mean = rho1 * resid[1:-1] + rho2 * resid[:-2]
    numpyro.sample("e_ar2", dist.Normal(mean, sigma), obs=resid[2:])


def matern32_cov(t: jnp.ndarray, amplitude: jnp.ndarray, length_s: jnp.ndarray, jitter: float = 1e-5) -> jnp.ndarray:
    d = jnp.abs(t[:, None] - t[None, :])
    r = jnp.sqrt(3.0) * d / jnp.maximum(length_s, 1e-4)
    cov = amplitude**2 * (1.0 + r) * jnp.exp(-r)
    return cov + (jitter**2) * jnp.eye(t.shape[0])


def model_gp_matern32(t: jnp.ndarray, y: jnp.ndarray) -> None:
    phi = sample_ombe_params()
    mu = ombe_volume(phi, t)
    amp = numpyro.sample("gp_amp_l", dist.HalfNormal(0.50))
    length_s = numpyro.sample("gp_length_s", dist.LogNormal(jnp.log(2.0), 1.0))
    sigma_l = numpyro.sample("sigma_l", dist.HalfNormal(0.05))
    cov = matern32_cov(t, amp, length_s) + sigma_l**2 * jnp.eye(t.shape[0])
    numpyro.sample("y", dist.MultivariateNormal(mu, covariance_matrix=cov), obs=y)


def curve_arrays(raw_row: Any, k: int, *, gp: bool = False, min_points: int = 8) -> tuple[np.ndarray, np.ndarray]:
    curve = extract_curve_for_fit(raw_row)
    stride = NUTS_GRID_STRIDE * int(k)
    t = np.asarray(curve["vol_t"], dtype=np.float32)[::stride]
    y = np.asarray(curve["vol_l"], dtype=np.float32)[::stride]
    if gp and len(t) > MAX_GP_POINTS:
        idx = np.unique(np.linspace(0, len(t) - 1, MAX_GP_POINTS).round().astype(int))
        t = t[idx]
        y = y[idx]
    if len(t) < min_points:
        raise ValueError(f"too few likelihood points after thinning: {len(t)}")
    return t, y


def extra_summary(samples: dict[str, np.ndarray], variant: str, t: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    if "sigma_l" in samples:
        out["sigma_l_median"] = float(np.median(np.ravel(samples["sigma_l"])))
        out["sigma_ml_median"] = 1000.0 * out["sigma_l_median"]
    if "rho" in samples:
        rho = float(np.median(np.ravel(samples["rho"])))
        out["rho_median"] = rho
        out["corr_length_samples"] = float(-1.0 / np.log(max(rho, 1e-6))) if 0 < rho < 1 else np.nan
        out["corr_length_s"] = out["corr_length_samples"] * float(np.median(np.diff(t))) if len(t) > 1 else np.nan
    if "rho1" in samples:
        out["rho1_median"] = float(np.median(np.ravel(samples["rho1"])))
    if "rho2" in samples:
        out["rho2_median"] = float(np.median(np.ravel(samples["rho2"])))
    if "gp_amp_l" in samples:
        out["gp_amp_ml_median"] = 1000.0 * float(np.median(np.ravel(samples["gp_amp_l"])))
    if "gp_length_s" in samples:
        length = float(np.median(np.ravel(samples["gp_length_s"])))
        out["gp_length_s_median"] = length
        out["gp_length_samples_median"] = length / float(np.median(np.diff(t))) if len(t) > 1 else np.nan
    if "beta_time" in samples:
        out["beta_time_median"] = float(np.median(np.ravel(samples["beta_time"])))
    return out


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        all_rows = pd.concat([old, new], ignore_index=True, sort=False)
        all_rows.to_csv(path, index=False)
    else:
        new.to_csv(path, index=False)


def run_one(variant: str, subset_row: Any, raw_by_index: dict[int, Any], t_full: np.ndarray) -> dict[str, Any]:
    raw_idx = int(subset_row.raw_index)
    base = {
        "row_type": "part_d_nuts_variant",
        "variant": variant,
        "SEQN": float(subset_row.SEQN),
        "raw_index": raw_idx,
        "truth_severity": subset_row.truth_severity,
        "truth_ratio": float(subset_row.fev1_fvc_ratio),
        "nested_prior_100": bool(getattr(subset_row, "nested_prior_100", False)),
        "likelihood_subset_150": True,
    }
    try:
        if variant.startswith("L4_THIN_k"):
            k = int(variant.split("k", 1)[1])
            t_obs, y_obs = curve_arrays(raw_by_index[raw_idx], k, min_points=5)
            model = model_l4_ar1_time_hetero
        elif variant == "M1_AR2":
            t_obs, y_obs = curve_arrays(raw_by_index[raw_idx], 1, min_points=8)
            model = model_ar2
        elif variant == "M2_GP_MATERN32":
            t_obs, y_obs = curve_arrays(raw_by_index[raw_idx], 1, gp=True, min_points=8)
            model = model_gp_matern32
        else:
            raise ValueError(f"unknown variant: {variant}")
        start = time.perf_counter()
        kernel = NUTS(lambda y=None: model(jnp.asarray(t_obs), y))
        mcmc = MCMC(
            kernel,
            num_warmup=NUTS_WARMUP,
            num_samples=NUTS_DRAWS,
            num_chains=NUTS_CHAINS,
            chain_method="sequential",
            progress_bar=False,
        )
        seed = RANDOM_SEED + raw_idx + VARIANT_SEED_OFFSETS[variant]
        mcmc.run(jax.random.PRNGKey(seed), y=jnp.asarray(y_obs))
        samples_chain = mcmc.get_samples(group_by_chain=True)
        diag = mcmc_diagnostics(mcmc)
        flat = {name: np.asarray(samples_chain[name]).reshape(-1) for name in samples_chain}
        phi = samples_to_phi(flat)
        ratio = ratio_samples_from_phi(phi, t_full)
        ratio = ratio[np.isfinite(ratio)]
        pit = float(np.mean(ratio < float(subset_row.fev1_fvc_ratio))) if len(ratio) else np.nan
        base.update(
            {
                "fit_error": "",
                "n_likelihood_points": int(len(t_obs)),
                "likelihood_dt_s_median": float(np.median(np.diff(t_obs))) if len(t_obs) > 1 else np.nan,
                "wall_clock_s": float(time.perf_counter() - start),
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
                **extra_summary(flat, variant, t_obs),
            }
        )
        for level in COVERAGE_LEVELS:
            lo = (1.0 - level) / 2.0
            hi = 1.0 - lo
            qlo, qhi = np.quantile(ratio, [lo, hi]) if len(ratio) else (np.nan, np.nan)
            base[f"covered_{int(level * 100)}"] = bool(qlo <= float(subset_row.fev1_fvc_ratio) <= qhi)
    except Exception as exc:
        base.update({"fit_error": str(exc), "converged": False})
    return base


def summarize_variant(rows: pd.DataFrame) -> dict[str, Any]:
    ok = rows[(rows["fit_error"].fillna("") == "") & rows["converged"].fillna(False).astype(bool)].copy()
    numeric_cols = [
        "pit",
        "n_likelihood_points",
        "ratio_sd",
        "sigma_ml_median",
        "rho_median",
        "rho1_median",
        "rho2_median",
        "corr_length_samples",
        "corr_length_s",
        "gp_amp_ml_median",
        "gp_length_s_median",
        "gp_length_samples_median",
        "max_rhat",
        "min_bulk_ess",
        "wall_clock_s",
    ] + [f"covered_{int(level * 100)}" for level in COVERAGE_LEVELS]
    for col in numeric_cols:
        if col in ok:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")

    def median_col(col: str) -> float:
        if not len(ok) or col not in ok:
            return np.nan
        arr = ok[col].to_numpy(float)
        if not np.isfinite(arr).any():
            return np.nan
        return float(np.nanmedian(arr))

    p = ok["pit"].to_numpy(float)
    counts, _ = np.histogram(p[np.isfinite(p)], bins=np.linspace(0.0, 1.0, PIT_BINS + 1))
    pit_p = float(chisquare(counts).pvalue) if len(ok) else np.nan
    coverage = {}
    for level in COVERAGE_LEVELS:
        col = f"covered_{int(level * 100)}"
        coverage[str(level)] = float(ok[col].mean()) if len(ok) and col in ok else np.nan
    calibrated = bool(
        len(ok)
        and all(abs(coverage[str(level)] - level) <= 0.07 for level in COVERAGE_LEVELS)
        and pit_p >= PIT_P_THRESHOLD
    )
    return {
        "n_attempted": int(len(rows)),
        "n_converged": int(len(ok)),
        "n_fit_errors": int((rows["fit_error"].fillna("") != "").sum()),
        "n_nonconverged_excluded": int(len(rows) - len(ok)),
        "n_likelihood_points_median": float(np.nanmedian(ok["n_likelihood_points"])) if len(ok) else np.nan,
        "coverage": coverage,
        "pit_counts": counts.astype(int).tolist(),
        "pit_chisquare_p": pit_p,
        "pit_mean": float(np.nanmean(p)) if len(ok) else np.nan,
        "calibrated": calibrated,
        "ratio_sd_median": median_col("ratio_sd"),
        "sigma_ml_median": median_col("sigma_ml_median"),
        "rho_median": median_col("rho_median"),
        "rho1_median": median_col("rho1_median"),
        "rho2_median": median_col("rho2_median"),
        "corr_length_samples_median": median_col("corr_length_samples"),
        "corr_length_s_median": median_col("corr_length_s"),
        "gp_amp_ml_median": median_col("gp_amp_ml_median"),
        "gp_length_s_median": median_col("gp_length_s_median"),
        "gp_length_samples_median": median_col("gp_length_samples_median"),
        "max_rhat_median": median_col("max_rhat"),
        "max_rhat_max": float(np.nanmax(ok["max_rhat"])) if len(ok) and np.isfinite(ok["max_rhat"]).any() else np.nan,
        "min_bulk_ess_median": median_col("min_bulk_ess"),
        "min_bulk_ess_min": float(np.nanmin(ok["min_bulk_ess"])) if len(ok) and np.isfinite(ok["min_bulk_ess"]).any() else np.nan,
        "wall_clock_s_sum": float(np.nansum(ok["wall_clock_s"])) if len(ok) else np.nan,
    }


def summarize_all(rows: pd.DataFrame) -> dict[str, Any]:
    summaries = {variant: summarize_variant(part) for variant, part in rows.groupby("variant")}
    calibrated = [name for name, summary in summaries.items() if summary["calibrated"]]
    if calibrated:
        verdict = "CALIBRATED"
        best = calibrated[0]
        message = f"Posterior classification and straddle-abstention are unblocked with {best}."
    else:
        best_name, best_summary = max(summaries.items(), key=lambda kv: kv[1]["coverage"].get("0.5", -np.inf))
        if best_summary["coverage"].get("0.5", 0.0) > summaries.get("L4_THIN_k1", {}).get("coverage", {}).get("0.5", 0.0) + 0.05:
            verdict = "IMPROVED-NOT-CLEARED"
            message = f"Best variant {best_name} improves 50% coverage but does not clear all calibration criteria."
        else:
            verdict = "NO-IMPROVEMENT"
            message = "Thinning and longer-memory variants did not materially improve calibration."
        best = best_name
    thin = [summaries.get(f"L4_THIN_k{k}", {}) for k in THIN_KS]
    cov50 = [s.get("coverage", {}).get("0.5", np.nan) for s in thin]
    monotone_coverage50 = bool(all(np.isnan(cov50[i]) or np.isnan(cov50[i + 1]) or cov50[i + 1] >= cov50[i] for i in range(len(cov50) - 1)))
    return {
        "verdict": verdict,
        "best_variant": best,
        "message": message,
        "variant_summaries": summaries,
        "thinning_monotone_coverage50": monotone_coverage50,
        "posterior_classification_unblocked": bool(verdict == "CALIBRATED"),
    }


def write_outputs(root: Path, result: dict[str, Any], rows: pd.DataFrame) -> None:
    out = root / "results"
    lines = [
        "# Gate - Posterior Width Thinning + Longer-Memory Error, Cycle F",
        "",
        f"- VERDICT: {result['verdict']}",
        f"- best variant: {result['best_variant']}",
        f"- message: {result['message']}",
        f"- posterior classification unblocked: {result['posterior_classification_unblocked']}",
        "- k=1 is the existing gate_likelihood_repair_F control grid: raw curve sampled every 5 native 100 Hz points, i.e. 20 Hz. Larger k thins that grid further.",
        f"- GP Matern-3/2 uses a dense covariance on at most {MAX_GP_POINTS} evenly-spaced control-grid points per curve for tractability.",
        f"- NUTS settings: warmup={NUTS_WARMUP}, draws={NUTS_DRAWS}, chains={NUTS_CHAINS}, convergence R-hat <= {CONVERGED_RHAT_MAX}, ESS >= {CONVERGED_ESS_MIN}.",
        "",
        "## Variant Summary",
        "",
        "| variant | n converged | excluded | points | cov50 | cov80 | cov95 | PIT p | PIT mean | ratio SD med | sigma mL | rho | corr len samples | corr len s | calibrated |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, summary in result["variant_summaries"].items():
        lines.append(
            f"| {name} | {summary['n_converged']} | {summary['n_nonconverged_excluded']} | {summary['n_likelihood_points_median']:.0f} | "
            f"{summary['coverage']['0.5']:.3f} | {summary['coverage']['0.8']:.3f} | {summary['coverage']['0.95']:.3f} | "
            f"{summary['pit_chisquare_p']:.3g} | {summary['pit_mean']:.3f} | {summary['ratio_sd_median']:.4f} | "
            f"{summary.get('sigma_ml_median', np.nan):.1f} | {summary.get('rho_median', np.nan):.3f} | "
            f"{summary.get('corr_length_samples_median', np.nan):.1f} | {summary.get('corr_length_s_median', np.nan):.2f} | {summary['calibrated']} |"
        )
    lines.extend(
        [
            "",
            f"- thinning monotone 50% coverage pattern: {result['thinning_monotone_coverage50']}",
            "",
            "## Longer-Memory Parameters",
            "",
            "| variant | rho1 | rho2 | GP amp mL | GP length s | GP length samples | runtime s |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, summary in result["variant_summaries"].items():
        if name == "M1_AR2" or name.startswith("M2_GP"):
            lines.append(
                f"| {name} | {summary.get('rho1_median', np.nan):.3f} | {summary.get('rho2_median', np.nan):.3f} | "
                f"{summary.get('gp_amp_ml_median', np.nan):.1f} | {summary.get('gp_length_s_median', np.nan):.2f} | "
                f"{summary.get('gp_length_samples_median', np.nan):.1f} | {summary.get('wall_clock_s_sum', np.nan):.1f} |"
            )
    lines.extend(
        [
            "",
            "## Row-Level Output",
            "",
            "Each row contains one NUTS fit for one participant/curve/variant, with R-hat, ESS, PIT, coverage indicators, posterior ratio quantiles, and likelihood-process parameters.",
            "",
        ]
    )
    (out / f"{OUTPUT_STEM}.md").write_text("\n".join(lines), encoding="utf-8")
    (out / f"{OUTPUT_STEM}.json").write_text(json.dumps(to_jsonable(result), indent=2) + "\n", encoding="utf-8")
    rows.to_csv(out / f"{OUTPUT_STEM}_rows.csv", index=False)


def run(root: Path, variants: list[str] | None = None, resume: bool = True) -> dict[str, Any]:
    configure_caches(root)
    out = root / "results"
    out.mkdir(exist_ok=True)
    checkpoint = out / CHECKPOINT_NAME
    subset = load_policy_and_subset(root)
    if variants is None:
        variants = [f"L4_THIN_k{k}" for k in THIN_KS] + ["M1_AR2", "M2_GP_MATERN32"]
    raw_path, _ = find_or_unzip_raw(root)
    raw, _ = read_raw(raw_path)
    raw_map = {int(idx): row for idx, row in raw.iterrows()}
    t_full = np.asarray(OmbeGrid().t, dtype=np.float32)
    done: set[tuple[str, int]] = set()
    if resume and checkpoint.exists():
        prior = pd.read_csv(checkpoint, usecols=["variant", "raw_index"])
        prior = prior[prior["variant"].isin(variants)]
        done = set((str(r.variant), int(r.raw_index)) for r in prior.itertuples(index=False))
        print(f"Posterior width checkpoint: {len(done)} completed fits for requested variants.")
    elif checkpoint.exists():
        checkpoint.unlink()
    batch = []
    total = len(subset) * len(variants)
    already_done = len(done)
    completed = 0
    for variant in variants:
        for i, row in enumerate(subset.itertuples(index=False), start=1):
            key = (variant, int(row.raw_index))
            if key in done:
                continue
            batch.append(run_one(variant, row, raw_map, t_full))
            completed += 1
            if len(batch) >= 5:
                append_rows(checkpoint, batch)
                batch = []
            if completed % 10 == 0:
                print(f"Posterior width NUTS progress: {already_done + completed}/{total} ({variant})")
    append_rows(checkpoint, batch)
    rows = pd.read_csv(checkpoint)
    result = summarize_all(rows)
    result.update(
        {
            "random_seed": RANDOM_SEED,
            "raw_source": str(raw_path),
            "subset_n": int(len(subset)),
            "subset_counts": subset["truth_severity"].value_counts().to_dict(),
            "variants_requested": variants,
            "widened_priors": natural_ranges_for_report(WIDE_PRIOR_RANGES),
            "nuts_settings": {
                "warmup": NUTS_WARMUP,
                "draws": NUTS_DRAWS,
                "chains": NUTS_CHAINS,
                "base_grid_stride": NUTS_GRID_STRIDE,
                "base_grid_hz": float(100 / NUTS_GRID_STRIDE),
                "converged_rule": {"max_rhat_lte": CONVERGED_RHAT_MAX, "min_bulk_ess_gte": CONVERGED_ESS_MIN},
            },
        }
    )
    write_outputs(root, result, rows)
    return result


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result = run(Path(args.root).resolve(), variants=args.variants, resume=not args.no_resume)
    print(f"Posterior width gate: {result['verdict']} - {result['message']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
