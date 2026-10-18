from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache" / "xdg"))

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
from numpyro.infer import MCMC, NUTS
from scipy.stats import binom, kstest

from .gate_likelihood_repair_f import (
    NUTS_CHAINS as REPO_NUTS_CHAINS,
)
from .gate_likelihood_repair_f import (
    NUTS_DRAWS as REPO_NUTS_DRAWS,
)
from .gate_likelihood_repair_f import (
    NUTS_GRID_STRIDE,
    VARIANT_SEED_OFFSETS,
    mcmc_diagnostics,
    ombe_likelihood_model,
    samples_to_phi,
)
from .gate_likelihood_repair_f import (
    NUTS_WARMUP as REPO_NUTS_WARMUP,
)
from .gate_misspecification_isolation_f import (
    COVERAGE_LEVELS,
    SEED,
    TARGET_COUNTS,
    clean_json,
    load_real_subset,
    md_table,
    real_templates,
    synthetic_dataset,
)
from .gate_posterior_calibration_repair_f import ratio_samples_from_phi
from .ombe import OmbeGrid

OUTPUT_STEM = "gate_sampler_convergence_F"
CACHE_VERSION = "v2_gateA_seed"
VARIANT = "L0_baseline"

RUNG_SPECS: list[dict[str, Any]] = [
    {"rung": "S1", "chains": 2, "warmup": 250, "draws": 250, "target_accept": None, "dense_mass": False},
    {"rung": "S2", "chains": 4, "warmup": 500, "draws": 500, "target_accept": None, "dense_mass": False},
    {"rung": "S3", "chains": 4, "warmup": 1000, "draws": 1000, "target_accept": None, "dense_mass": False},
    {"rung": "S4", "chains": 4, "warmup": 2500, "draws": 2500, "target_accept": None, "dense_mass": False},
    {"rung": "S5", "chains": 4, "warmup": 2500, "draws": 2500, "target_accept": 0.95, "dense_mass": False},
    {"rung": "S6", "chains": 4, "warmup": 2500, "draws": 2500, "target_accept": 0.99, "dense_mass": True},
]


def configure(root: Path, chains: int) -> None:
    cache = root / ".cache"
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))
    numpyro.set_host_device_count(max(chains, 1))


def binomial_interval(n: int, nominal: float) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    return float(binom.ppf(0.025, n, nominal) / n), float(binom.ppf(0.975, n, nominal) / n)


def cache_dir(root: Path) -> Path:
    path = root / "results" / "_cache_sampler_convergence"
    path.mkdir(parents=True, exist_ok=True)
    return path


def rung_cache_path(root: Path, rung: str) -> Path:
    return cache_dir(root) / f"{CACHE_VERSION}_{rung}_cell_A_L0_rows.csv"


def div_cache_path(root: Path, rung: str) -> Path:
    return cache_dir(root) / f"{CACHE_VERSION}_{rung}_divergent_draws.csv"


def cond_cache_path(root: Path, rung: str) -> Path:
    return cache_dir(root) / f"{CACHE_VERSION}_{rung}_condition_numbers.csv"


def convergence_failure_counts(rows: pd.DataFrame) -> dict[str, int]:
    if not len(rows):
        return {"divergence_curve_failures": 0, "rhat_curve_failures": 0, "ess_curve_failures": 0}
    ok = rows[rows.get("fit_error", "").fillna("").eq("")].copy()
    if not len(ok):
        return {"divergence_curve_failures": 0, "rhat_curve_failures": 0, "ess_curve_failures": 0}
    return {
        "divergence_curve_failures": int((ok["divergence_count"] > 0).sum()) if "divergence_count" in ok else 0,
        "rhat_curve_failures": int((ok["max_rhat"] > 1.01).sum()) if "max_rhat" in ok else 0,
        "ess_curve_failures": int((ok["min_bulk_ess"] < 400.0).sum()) if "min_bulk_ess" in ok else 0,
    }


def convergence_impossible_reason(rows: pd.DataFrame, n_total: int) -> str:
    counts = convergence_failure_counts(rows)
    max_divergence_failures = int(np.floor(0.05 * n_total))
    max_rhat_failures = int(np.floor(0.01 * n_total))
    max_ess_failures = int(np.floor(0.01 * n_total))
    reasons = []
    if counts["divergence_curve_failures"] > max_divergence_failures:
        reasons.append(f"{counts['divergence_curve_failures']} curves with divergences exceeds allowed {max_divergence_failures}")
    if counts["rhat_curve_failures"] > max_rhat_failures:
        reasons.append(f"{counts['rhat_curve_failures']} curves with Rhat>1.01 exceeds allowed {max_rhat_failures}")
    if counts["ess_curve_failures"] > max_ess_failures:
        reasons.append(f"{counts['ess_curve_failures']} curves with ESS<400 exceeds allowed {max_ess_failures}")
    return "; ".join(reasons)


def phi_condition_number(phi: np.ndarray) -> float:
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 2 or phi.shape[0] < 10:
        return np.nan
    cov = np.cov(phi.T)
    eig = np.linalg.eigvalsh(cov)
    eig = eig[np.isfinite(eig) & (eig > 1e-14)]
    if len(eig) == 0:
        return np.nan
    return float(np.max(eig) / np.min(eig))


def natural_from_phi_samples(phi: np.ndarray) -> pd.DataFrame:
    tau_f = np.exp(phi[:, 0])
    tau_s = np.exp(phi[:, 1])
    alpha = 1.0 / (1.0 + np.exp(-phi[:, 2]))
    fvc_l = np.exp(phi[:, 3])
    tau_o = np.exp(phi[:, 4])
    return pd.DataFrame({"tau_f": tau_f, "tau_s": tau_s, "alpha": alpha, "fvc_l": fvc_l, "tau_o": tau_o})


def run_one_curve(spec: dict[str, Any], data_row: Any, arrays: dict[int, dict[str, np.ndarray]], t_full: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    curve_id = int(data_row.curve_id)
    arr = arrays[curve_id]
    t_obs = arr["t"]
    y_obs = arr["y"]
    target_accept = spec["target_accept"]
    nuts_kwargs: dict[str, Any] = {}
    if target_accept is not None:
        nuts_kwargs["target_accept_prob"] = float(target_accept)
    if spec["dense_mass"]:
        nuts_kwargs["dense_mass"] = True
    base: dict[str, Any] = {
        "row_type": "ladder_curve",
        "rung": spec["rung"],
        "variant": VARIANT,
        "curve_id": curve_id,
        "truth_severity": data_row.truth_severity,
        "truth_ratio": float(data_row.truth_ratio),
        "chains": int(spec["chains"]),
        "warmup": int(spec["warmup"]),
        "draws": int(spec["draws"]),
        "target_accept": np.nan if target_accept is None else float(target_accept),
        "dense_mass": bool(spec["dense_mass"]),
        "fit_error": "",
    }
    start = time.perf_counter()
    div_rows = pd.DataFrame()
    cond_row: dict[str, Any] = {"rung": spec["rung"], "curve_id": curve_id, "posterior_condition_number": np.nan}
    try:
        kernel = NUTS(lambda y=None: ombe_likelihood_model(jnp.asarray(t_obs), y, VARIANT), **nuts_kwargs)
        mcmc = MCMC(
            kernel,
            num_warmup=int(spec["warmup"]),
            num_samples=int(spec["draws"]),
            num_chains=int(spec["chains"]),
            chain_method="sequential",
            progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(SEED + curve_id + VARIANT_SEED_OFFSETS[VARIANT]), y=jnp.asarray(y_obs), extra_fields=("diverging",))
        samples_chain = mcmc.get_samples(group_by_chain=True)
        diag = mcmc_diagnostics(mcmc)
        flat = {name: np.asarray(samples_chain[name]).reshape(-1) for name in samples_chain}
        phi = samples_to_phi(flat)
        ratio = ratio_samples_from_phi(phi, t_full)
        ratio = ratio[np.isfinite(ratio)]
        extra = mcmc.get_extra_fields(group_by_chain=True)
        diverging = np.asarray(extra.get("diverging", np.zeros((int(spec["chains"]), int(spec["draws"])), dtype=bool))).astype(bool)
        divergence_count = int(diverging.sum())
        phi_chain = samples_to_phi({name: np.asarray(samples_chain[name]) for name in samples_chain}).reshape(int(spec["chains"]), int(spec["draws"]), 5)
        if divergence_count:
            div_phi = phi_chain[diverging]
            div_nat = natural_from_phi_samples(div_phi)
            div_nat.insert(0, "curve_id", curve_id)
            div_nat.insert(0, "rung", spec["rung"])
            div_rows = div_nat
        cond_row["posterior_condition_number"] = phi_condition_number(phi)
        base.update(
            {
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
                "pit": float(np.mean(ratio < float(data_row.truth_ratio))) if len(ratio) else np.nan,
                "divergence_count": divergence_count,
                "wall_clock_s": float(time.perf_counter() - start),
                **diag,
            }
        )
        for level in COVERAGE_LEVELS:
            lo = (1.0 - level) / 2.0
            hi = 1.0 - lo
            qlo, qhi = np.quantile(ratio, [lo, hi]) if len(ratio) else (np.nan, np.nan)
            base[f"covered_{int(level * 100)}"] = bool(qlo <= float(data_row.truth_ratio) <= qhi)
    except Exception as exc:
        base.update({"fit_error": str(exc), "converged": False, "divergence_count": np.nan, "wall_clock_s": float(time.perf_counter() - start)})
    return base, div_rows, cond_row


def run_rung(root: Path, spec: dict[str, Any], data: pd.DataFrame, arrays: dict[int, dict[str, np.ndarray]], t_full: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows_path = rung_cache_path(root, spec["rung"])
    div_path = div_cache_path(root, spec["rung"])
    cond_path = cond_cache_path(root, spec["rung"])
    if rows_path.exists():
        rows = pd.read_csv(rows_path, low_memory=False)
        if len(rows) == len(data):
            div = pd.read_csv(div_path, low_memory=False) if div_path.exists() and div_path.stat().st_size > 1 else pd.DataFrame()
            cond = pd.read_csv(cond_path, low_memory=False) if cond_path.exists() and cond_path.stat().st_size > 1 else pd.DataFrame()
            print(f"Using cached {spec['rung']}: {rows_path}")
            return rows, div, cond
    configure(root, int(spec["chains"]))
    done = pd.read_csv(rows_path, low_memory=False) if rows_path.exists() and rows_path.stat().st_size > 1 else pd.DataFrame()
    done_ids = set(done["curve_id"].astype(int)) if len(done) and "curve_id" in done else set()
    rows_accum = done.to_dict("records") if len(done) else []
    div_accum = pd.read_csv(div_path, low_memory=False) if div_path.exists() and div_path.stat().st_size > 1 else pd.DataFrame()
    cond_accum = pd.read_csv(cond_path, low_memory=False) if cond_path.exists() and cond_path.stat().st_size > 1 else pd.DataFrame()
    cond_records = cond_accum.to_dict("records") if len(cond_accum) else []
    reason = convergence_impossible_reason(pd.DataFrame(rows_accum), len(data)) if len(rows_accum) else ""
    if reason:
        print(f"{spec['rung']}: stopping early; convergence impossible after {len(rows_accum)}/{len(data)} curves ({reason})")
        rows = pd.DataFrame(rows_accum).sort_values("curve_id").reset_index(drop=True)
        cond = pd.DataFrame(cond_records).sort_values("curve_id").reset_index(drop=True)
        return rows, div_accum, cond
    for i, row in enumerate(data.itertuples(index=False), start=1):
        if int(row.curve_id) in done_ids:
            continue
        out, div_rows, cond_row = run_one_curve(spec, row, arrays, t_full)
        rows_accum.append(out)
        cond_records.append(cond_row)
        pd.DataFrame(rows_accum).to_csv(rows_path, index=False)
        pd.DataFrame(cond_records).to_csv(cond_path, index=False)
        if len(div_rows):
            div_accum = pd.concat([div_accum, div_rows], ignore_index=True)
            div_accum.to_csv(div_path, index=False)
        elif not div_path.exists():
            pd.DataFrame().to_csv(div_path, index=False)
        completed = len(rows_accum)
        if completed % 10 == 0 or completed == len(data):
            print(f"{spec['rung']}: NUTS progress {completed}/{len(data)}")
        reason = convergence_impossible_reason(pd.DataFrame(rows_accum), len(data))
        if reason:
            print(f"{spec['rung']}: stopping early; convergence impossible after {completed}/{len(data)} curves ({reason})")
            break
    rows = pd.DataFrame(rows_accum).sort_values("curve_id").reset_index(drop=True)
    cond = pd.DataFrame(cond_records).sort_values("curve_id").reset_index(drop=True)
    div = div_accum
    return rows, div, cond


def summarize_rung(rows: pd.DataFrame, cond: pd.DataFrame, n_planned: int) -> dict[str, Any]:
    ok = rows[rows["fit_error"].fillna("").eq("")].copy()
    pits = ok["pit"].to_numpy(float)
    pits = pits[np.isfinite(pits)]
    ks_stat, ks_p = kstest(pits, "uniform") if len(pits) else (np.nan, np.nan)
    complete = bool(len(rows) == n_planned)
    impossible_reason = "" if complete else convergence_impossible_reason(rows, n_planned)
    failure_counts = convergence_failure_counts(rows)
    out: dict[str, Any] = {
        "rung": str(rows["rung"].iloc[0]) if len(rows) else "",
        "n_planned": int(n_planned),
        "n_attempted": int(len(rows)),
        "n_ok": int(len(ok)),
        "complete": complete,
        "early_stop_reason": impossible_reason,
        **failure_counts,
        "chains": int(rows["chains"].iloc[0]) if len(rows) else np.nan,
        "warmup": int(rows["warmup"].iloc[0]) if len(rows) else np.nan,
        "draws": int(rows["draws"].iloc[0]) if len(rows) else np.nan,
        "target_accept": float(rows["target_accept"].iloc[0]) if len(rows) and pd.notna(rows["target_accept"].iloc[0]) else np.nan,
        "dense_mass": bool(rows["dense_mass"].iloc[0]) if len(rows) else False,
        "pit_ks_statistic": float(ks_stat),
        "pit_ks_p_value": float(ks_p),
        "median_rhat": float(np.nanmedian(ok["max_rhat"])) if len(ok) else np.nan,
        "rhat_gt_1p01_fraction": float((ok["max_rhat"] > 1.01).mean()) if len(ok) else np.nan,
        "rhat_gt_1p05_fraction": float((ok["max_rhat"] > 1.05).mean()) if len(ok) else np.nan,
        "min_ess": float(np.nanmin(ok["min_bulk_ess"])) if len(ok) else np.nan,
        "median_ess": float(np.nanmedian(ok["min_bulk_ess"])) if len(ok) else np.nan,
        "ess_lt_400_fraction": float((ok["min_bulk_ess"] < 400.0).mean()) if len(ok) else np.nan,
        "total_divergences": int(np.nansum(ok["divergence_count"])) if len(ok) else 0,
        "any_divergence_fraction": float((ok["divergence_count"] > 0).mean()) if len(ok) else np.nan,
        "posterior_ratio_sd_median": float(np.nanmedian(ok["ratio_sd"])) if len(ok) else np.nan,
        "wall_clock_s_per_curve": float(np.nanmean(ok["wall_clock_s"])) if len(ok) else np.nan,
        "condition_number_median": float(np.nanmedian(cond["posterior_condition_number"])) if len(cond) else np.nan,
        "condition_number_iqr_low": float(np.nanpercentile(cond["posterior_condition_number"], 25)) if len(cond) else np.nan,
        "condition_number_iqr_high": float(np.nanpercentile(cond["posterior_condition_number"], 75)) if len(cond) else np.nan,
    }
    for level in COVERAGE_LEVELS:
        col = f"covered_{int(level * 100)}"
        n = int(ok[col].notna().sum()) if col in ok else 0
        empirical = float(ok[col].mean()) if n else np.nan
        lo, hi = binomial_interval(n, level)
        out[f"coverage_{int(level * 100)}"] = empirical
        out[f"coverage_{int(level * 100)}_ci_low"] = lo
        out[f"coverage_{int(level * 100)}_ci_high"] = hi
        out[f"coverage_{int(level * 100)}_nominal"] = bool(np.isfinite(empirical) and lo <= empirical <= hi)
    out["converged"] = bool(
        complete
        and
        np.isfinite(out["any_divergence_fraction"])
        and out["any_divergence_fraction"] <= 0.05
        and out["rhat_gt_1p01_fraction"] <= 0.01
        and out["ess_lt_400_fraction"] <= 0.01
    )
    out["nominal_all"] = bool(complete and all(out[f"coverage_{int(level * 100)}_nominal"] for level in COVERAGE_LEVELS))
    return out


def audit_nuts_files(root: Path, s_min: dict[str, Any] | None) -> pd.DataFrame:
    rows = []
    patterns = ("NUTS", "MCMC", "numpyro", "SNPE", "sbi", "NUTS_WARMUP", "NUTS_DRAWS", "NUTS_CHAINS")
    for file in sorted((root / "predictx").glob("*.py")):
        text = file.read_text(encoding="utf-8", errors="ignore")
        if not any(p in text for p in patterns):
            continue
        warmup = extract_int(text, r"NUTS_WARMUP\s*=\s*(\d+)") or extract_int(text, r"num_warmup\s*=\s*(\d+)")
        draws = extract_int(text, r"NUTS_DRAWS\s*=\s*(\d+)") or extract_int(text, r"num_samples\s*=\s*(\d+)")
        chains = extract_int(text, r"NUTS_CHAINS\s*=\s*(\d+)") or extract_int(text, r"num_chains\s*=\s*(\d+)")
        target_accept = extract_float(text, r"target_accept_prob\s*=\s*([0-9.]+)")
        dense_mass = bool(re.search(r"dense_mass\s*=\s*True", text))
        result_file = infer_result_file(root, file.name)
        headline = extract_headline(result_file)
        below = below_s_min(chains, warmup, draws, target_accept, dense_mass, s_min)
        rows.append(
            {
                "gate_file": str(file.relative_to(root)),
                "result_file": str(result_file.relative_to(root)) if result_file and result_file.exists() else "",
                "chains": chains,
                "warmup": warmup,
                "draws": draws,
                "target_accept": target_accept,
                "dense_mass": dense_mass,
                "below_S_min": below,
                "headline_NUTS_number_reported": headline,
            }
        )
    return pd.DataFrame(rows)


def extract_int(text: str, pattern: str) -> int | None:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def extract_float(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def infer_result_file(root: Path, code_name: str) -> Path | None:
    stem = code_name.removesuffix(".py")
    suffix = stem.removeprefix("gate_")
    candidates = [
        root / "results" / f"{suffix}_F.json",
        root / "results" / f"gate_{suffix}_F.json",
        root / "results" / f"{stem}_F.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def extract_headline(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "verdict" in data:
                return str(data["verdict"])[:500]
            if "headline" in data:
                return str(data["headline"])[:500]
    except Exception:
        pass
    return path.read_text(encoding="utf-8", errors="ignore")[:500]


def below_s_min(chains: int | None, warmup: int | None, draws: int | None, target_accept: float | None, dense_mass: bool, s_min: dict[str, Any] | None) -> Any:
    if s_min is None:
        return "S_min_not_established"
    if chains is None or warmup is None or draws is None:
        return "unknown"
    if chains < int(s_min["chains"]) or warmup < int(s_min["warmup"]) or draws < int(s_min["draws"]):
        return True
    min_ta = s_min.get("target_accept")
    if pd.notna(min_ta) and (target_accept is None or float(target_accept) < float(min_ta)):
        return True
    if bool(s_min.get("dense_mass")) and not dense_mass:
        return True
    return False


def deterministic_unaffected_rows(root: Path) -> pd.DataFrame:
    names = [
        "gate_clinical_classification_f.py",
        "gate_fvc_correction_f.py",
        "gate_offset_lungsize_f.py",
        "gate_truncation_recovery_f.py",
        "gate_reproducibility_volume_f.py",
        "gate_estimand_hybrid_f.py",
        "gate23_f.py",
        "gate3_f_rerun.py",
        "gate_model_vs_observed_f.py",
    ]
    rows = []
    for name in names:
        path = root / "predictx" / name
        if not path.exists():
            rows.append({"gate_file": f"predictx/{name}", "exists": False, "touches_nuts": np.nan, "unaffected": False})
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        touches = any(p in text for p in ("NUTS", "MCMC", "numpyro"))
        rows.append({"gate_file": str(path.relative_to(root)), "exists": True, "touches_nuts": touches, "unaffected": not touches})
    return pd.DataFrame(rows)


def cost_projection(ladder: pd.DataFrame, s_min: dict[str, Any] | None) -> dict[str, Any]:
    if s_min is None:
        ref = ladder.iloc[-1].to_dict() if len(ladder) else {}
        source = "last_rung_run"
    else:
        ref = s_min
        source = "S_min"
    sec_per_curve = float(ref.get("wall_clock_s_per_curve", np.nan))
    return {
        "source": source,
        "sec_per_curve": sec_per_curve,
        "misspec_2x2_2250_fits_hours": sec_per_curve * 2250.0 / 3600.0 if np.isfinite(sec_per_curve) else np.nan,
        "dose_900_fits_hours": sec_per_curve * 900.0 / 3600.0 if np.isfinite(sec_per_curve) else np.nan,
        "likelihood_repair_750_fits_hours": sec_per_curve * 750.0 / 3600.0 if np.isfinite(sec_per_curve) else np.nan,
    }


def write_outputs(root: Path, result: dict[str, Any], ladder: pd.DataFrame, audit: pd.DataFrame, unaffected: pd.DataFrame) -> None:
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    ladder.to_csv(out / f"{OUTPUT_STEM}_ladder.csv", index=False)
    audit.to_csv(out / f"{OUTPUT_STEM}_audit.csv", index=False)
    unaffected.to_csv(out / f"{OUTPUT_STEM}_deterministic_unaffected.csv", index=False)
    (out / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Gate - Sampler Convergence Establishment, Cycle F",
        "",
        "## Protocol Decisions",
        "",
        "- Cell A construction is reused from `gate_misspecification_isolation_f.py` with seed 20260710 and target counts 67/67/12/4.",
        "- The ladder varies sampler settings only: chains, warmup, draws, target_accept, dense_mass.",
        "- Coverage intervals are exact binomial Monte Carlo 95% intervals using the attempted row count; complete rungs use n=150 and early-stopped rungs use their partial n.",
        "- tau_f < tau_s is enforced by prior support: tau_f in [0.01, 1.0], tau_s in [1.0, 20.0]. There is no rejection step and no rejection rate.",
        "",
        f"## Verdict: {result['verdict']}",
        "",
        f"- S_min: {result.get('S_min', 'NOT-ESTABLISHED')}",
        f"- Gate A halt retracted: {result.get('gate_A_halt_retracted', False)}",
        "",
        "## Part 1 - Ladder",
        "",
        md_table(ladder.to_dict("records"), [("rung", "rung", "str"), ("n_attempted", "n", "int"), ("complete", "complete", "str"), ("chains", "chains", "int"), ("warmup", "warmup", "int"), ("draws", "draws", "int"), ("target_accept", "target accept", "float2"), ("dense_mass", "dense", "str"), ("coverage_50", "cov50", "float3"), ("coverage_50_ci_low", "50 lo", "float3"), ("coverage_50_ci_high", "50 hi", "float3"), ("coverage_80", "cov80", "float3"), ("coverage_95", "cov95", "float3"), ("median_rhat", "median Rhat", "float3"), ("rhat_gt_1p01_fraction", "Rhat>1.01", "float3"), ("rhat_gt_1p05_fraction", "Rhat>1.05", "float3"), ("min_ess", "min ESS", "float1"), ("median_ess", "median ESS", "float1"), ("ess_lt_400_fraction", "ESS<400", "float3"), ("total_divergences", "div", "int"), ("any_divergence_fraction", "any div", "float3"), ("posterior_ratio_sd_median", "ratio sd med", "float4"), ("wall_clock_s_per_curve", "sec/curve", "float2"), ("converged", "converged", "str"), ("nominal_all", "nominal", "str"), ("early_stop_reason", "early stop reason", "str")]),
        "",
        "## Part 2 - Divergence Geometry",
        "",
        f"- baseline condition number median/IQR and S_min condition number are in the ladder table. Mechanism note: {result.get('divergence_geometry_note', '')}",
        "- divergent transition coordinates are cached under `results/_cache_sampler_convergence/*_divergent_draws.csv` for scatter plotting.",
        "",
        "## Part 3 - NUTS Re-Audit",
        "",
        md_table(audit.to_dict("records"), [("gate_file", "gate file", "str"), ("result_file", "result file", "str"), ("chains", "chains", "int"), ("warmup", "warmup", "int"), ("draws", "draws", "int"), ("target_accept", "target", "float2"), ("dense_mass", "dense", "str"), ("below_S_min", "below S_min?", "str"), ("headline_NUTS_number_reported", "headline", "str")]),
        "",
        "Rows marked below S_min are not interpretable as posterior quantities and must be recomputed. This gate does not recompute them.",
        "",
        "## Deterministic Results Unaffected",
        "",
        md_table(unaffected.to_dict("records"), [("gate_file", "gate file", "str"), ("exists", "exists", "str"), ("touches_nuts", "touches NUTS", "str"), ("unaffected", "unaffected", "str")]),
        "",
        "## Part 4 - Cost Projection",
        "",
        f"- source: {result['cost_projection']['source']}",
        f"- seconds per curve: {result['cost_projection']['sec_per_curve']:.2f}",
        f"- Cell A/B/C x five likelihoods, 2250 fits: {result['cost_projection']['misspec_2x2_2250_fits_hours']:.2f} hours",
        f"- dose response, 900 fits: {result['cost_projection']['dose_900_fits_hours']:.2f} hours",
        f"- likelihood repair, 750 fits: {result['cost_projection']['likelihood_repair_750_fits_hours']:.2f} hours",
        "",
    ]
    (out / f"{OUTPUT_STEM}.md").write_text("\n".join(lines), encoding="utf-8")


def run(root: Path, max_rung: str | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    real_subset, raw, _existing_real = load_real_subset(root)
    templates = real_templates(real_subset, raw)
    t_full = np.asarray(OmbeGrid().t, dtype=np.float32)
    cell_a_data, cell_a_arrays, _meta = synthetic_dataset(templates, "cell_A_synth_iid", t_full)
    summaries = []
    s_min: dict[str, Any] | None = None
    stop_reason = ""
    for spec in RUNG_SPECS:
        rows, div, cond = run_rung(root, spec, cell_a_data, cell_a_arrays, t_full)
        summary = summarize_rung(rows, cond, len(cell_a_data))
        summaries.append(summary)
        print(f"{spec['rung']}: converged={summary['converged']} nominal={summary['nominal_all']} cov50={summary['coverage_50']:.3f} div={summary['total_divergences']} median_ess={summary['median_ess']:.1f}")
        if summary["converged"]:
            s_min = summary
            if summary["nominal_all"]:
                stop_reason = "first converged rung also reached nominal coverage"
                break
            stop_reason = "first converged rung failed nominal coverage; stopping per gate"
            break
        if max_rung and spec["rung"] == max_rung:
            stop_reason = f"manual max_rung {max_rung} reached"
            break
    ladder = pd.DataFrame(summaries)
    if s_min is None:
        verdict = "GEOMETRY-INTRACTABLE" if len(ladder) and ladder["rung"].iloc[-1] == "S6" else "INCOMPLETE"
        gate_retracted = False
        if not stop_reason:
            stop_reason = "S6 reached without any rung satisfying divergence, Rhat, and ESS convergence criteria"
    elif s_min["nominal_all"]:
        verdict = "SAMPLER-ARTIFACT"
        gate_retracted = True
    else:
        verdict = "COVERAGE-CODE-FAULT"
        gate_retracted = False
    audit = audit_nuts_files(root, s_min)
    unaffected = deterministic_unaffected_rows(root)
    cost = cost_projection(ladder, s_min)
    baseline = ladder[ladder["rung"].eq("S1")].iloc[0].to_dict() if len(ladder[ladder["rung"].eq("S1")]) else {}
    result = {
        "verdict": verdict,
        "S_min": None if s_min is None else s_min["rung"],
        "S_min_summary": s_min,
        "gate_A_halt_retracted": gate_retracted,
        "stop_reason": stop_reason,
        "ladder_rungs_run": ladder["rung"].tolist() if len(ladder) else [],
        "divergence_geometry_note": (
            f"S1 posterior condition number median {baseline.get('condition_number_median', np.nan):.1f}; "
            f"Rhat>1.01 fraction {baseline.get('rhat_gt_1p01_fraction', np.nan):.3f}; "
            f"divergent transitions {baseline.get('total_divergences', np.nan)}. "
            "The tau_f/tau_s ordering is by disjoint prior support, not rejection."
        ),
        "tau_ordering_constraint": {"method": "disjoint prior support", "rejection_rate": 0.0},
        "cost_projection": cost,
        "repo_likelihood_config": {"chains": REPO_NUTS_CHAINS, "warmup": REPO_NUTS_WARMUP, "draws": REPO_NUTS_DRAWS, "grid_stride": NUTS_GRID_STRIDE},
        "target_counts": TARGET_COUNTS,
        "wall_clock_s": float(time.perf_counter() - start),
    }
    write_outputs(root, result, ladder, audit, unaffected)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--max-rung", choices=[s["rung"] for s in RUNG_SPECS])
    args = parser.parse_args()
    result = run(Path(args.root).resolve(), max_rung=args.max_rung)
    print(f"VERDICT: {result['verdict']}")
    print(f"S_min: {result['S_min']}")
    print(f"Stop reason: {result['stop_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
