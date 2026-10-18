from __future__ import annotations

import argparse
import json
import os
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

from .gate_likelihood_repair_f import (
    VARIANT_SEED_OFFSETS,
    mcmc_diagnostics,
    ombe_likelihood_model,
    samples_to_phi,
)
from .gate_misspecification_isolation_f import COVERAGE_LEVELS, SEED, binomial_interval, clean_json, md_table
from .gate_posterior_calibration_repair_f import WIDE_PRIOR_RANGES, ratio_samples_from_phi
from .gate_s8_micropilot_f import (
    KAPPA_CUTOFF,
    SPEC,
    build_data,
    chain_diag_from_extra,
    fisher_kappa_for_rows,
    select_stratified,
    trace_rows,
)

OUTPUT_STEM = "gate_s9_modeinit_F"
CACHE_VERSION = "v1_s9_modeinit20"
VARIANT = "L0_baseline"
LM_PATH = "results/gate_policy_calibrated_flag_F_rows.csv"
S8_CURVES_PATH = "results/gate_s8_micropilot_F_curves.csv"
S8_FAILING_CURVES_DECLARED = {18, 57, 106, 109, 111, 112, 114, 120, 141, 149}
INIT_SD_FRACTION = 0.05


def configure(root: Path) -> None:
    cache = root / ".cache"
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))
    numpyro.set_host_device_count(int(SPEC["chains"]))


def cache_dir(root: Path) -> Path:
    out = root / "results" / "_cache_s9_modeinit"
    out.mkdir(parents=True, exist_ok=True)
    return out


def rows_path(root: Path) -> Path:
    return cache_dir(root) / f"{CACHE_VERSION}_rows.csv"


def traces_path(root: Path) -> Path:
    return cache_dir(root) / f"{CACHE_VERSION}_traces.csv"


def load_lm_fits(root: Path) -> pd.DataFrame:
    path = root / LM_PATH
    if not path.exists():
        raise RuntimeError(f"Missing stored LM fit file: {path}")
    cols = ["raw_index", "fit_error", "fit_tau_f", "fit_tau_s", "fit_alpha", "fit_fvc_l", "fit_tau_o"]
    lm = pd.read_csv(path, usecols=cols, low_memory=False).drop_duplicates("raw_index")
    for col in ["fit_tau_f", "fit_tau_s", "fit_alpha", "fit_fvc_l", "fit_tau_o"]:
        lm[col] = pd.to_numeric(lm[col], errors="coerce")
    lm["lm_available"] = (
        lm["fit_error"].fillna("").eq("")
        & lm[["fit_tau_f", "fit_tau_s", "fit_alpha", "fit_fvc_l", "fit_tau_o"]].apply(np.isfinite).all(axis=1)
    )
    return lm


def clip_to_prior(name: str, values: np.ndarray) -> np.ndarray:
    lo, hi = WIDE_PRIOR_RANGES[name]
    eps = max((hi - lo) * 1e-7, 1e-9)
    return np.clip(values, lo + eps, hi - eps)


def draw_init_values(curve_id: int, lm_row: Any | None) -> tuple[dict[str, jnp.ndarray], str, dict[str, Any]]:
    rng = np.random.default_rng(SEED + 9090 + int(curve_id))
    chains = int(SPEC["chains"])
    init: dict[str, jnp.ndarray] = {}
    meta: dict[str, Any] = {}
    if lm_row is not None and bool(getattr(lm_row, "lm_available", False)):
        source = "lm_ball"
        centers = {
            "tau_f": float(lm_row.fit_tau_f),
            "tau_s": float(lm_row.fit_tau_s),
            "alpha": float(lm_row.fit_alpha),
            "fvc_l": float(lm_row.fit_fvc_l),
            "tau_o": float(lm_row.fit_tau_o),
        }
        for name, center in centers.items():
            lo, hi = WIDE_PRIOR_RANGES[name]
            sd = INIT_SD_FRACTION * (hi - lo)
            vals = rng.normal(center, sd, size=chains)
            init[name] = jnp.asarray(clip_to_prior(name, vals), dtype=jnp.float32)
            meta[f"lm_{name}"] = center
    else:
        source = "prior_fallback"
        for name, (lo, hi) in WIDE_PRIOR_RANGES.items():
            init[name] = jnp.asarray(rng.uniform(lo, hi, size=chains), dtype=jnp.float32)
            meta[f"lm_{name}"] = np.nan
    return init, source, meta


def run_one_curve(row: Any, arrays: dict[int, dict[str, np.ndarray]], t_full: np.ndarray, fisher_kappa: float, lm_row: Any | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    curve_id = int(row.curve_id)
    arr = arrays[curve_id]
    t_obs = arr["t"]
    y_obs = arr["y"]
    init_params, init_source, init_meta = draw_init_values(curve_id, lm_row)
    base: dict[str, Any] = {
        "row_type": "s9_curve",
        "rung": "S9",
        "variant": VARIANT,
        "curve_id": curve_id,
        "truth_severity": str(row.truth_severity),
        "truth_ratio": float(row.truth_ratio),
        "template_raw_index": int(row.template_raw_index),
        "fisher_kappa": float(fisher_kappa),
        "chains": int(SPEC["chains"]),
        "warmup": int(SPEC["warmup"]),
        "draws": int(SPEC["draws"]),
        "target_accept": float(SPEC["target_accept"]),
        "dense_mass": bool(SPEC["dense_mass"]),
        "init_source": init_source,
        "fit_error": "",
        **init_meta,
    }
    start = time.perf_counter()
    try:
        kernel = NUTS(
            lambda y=None: ombe_likelihood_model(jnp.asarray(t_obs), y, VARIANT),
            target_accept_prob=float(SPEC["target_accept"]),
            dense_mass=True,
            max_tree_depth=int(SPEC["max_tree_depth"]),
        )
        mcmc = MCMC(
            kernel,
            num_warmup=int(SPEC["warmup"]),
            num_samples=int(SPEC["draws"]),
            num_chains=int(SPEC["chains"]),
            chain_method="sequential",
            progress_bar=False,
        )
        mcmc.run(
            jax.random.PRNGKey(SEED + curve_id + VARIANT_SEED_OFFSETS[VARIANT]),
            y=jnp.asarray(y_obs),
            init_params=init_params,
            extra_fields=("diverging", "accept_prob", "num_steps", "adapt_state.step_size"),
        )
        samples_chain = mcmc.get_samples(group_by_chain=True)
        extra = mcmc.get_extra_fields(group_by_chain=True)
        diag = mcmc_diagnostics(mcmc)
        flat = {name: np.asarray(samples_chain[name]).reshape(-1) for name in samples_chain}
        phi = samples_to_phi(flat)
        ratio = ratio_samples_from_phi(phi, t_full)
        ratio = ratio[np.isfinite(ratio)]
        chain_diag = chain_diag_from_extra(extra)
        divergence_count = int(np.asarray(extra["diverging"]).sum())
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
                "pit": float(np.mean(ratio < float(row.truth_ratio))) if len(ratio) else np.nan,
                "divergence_count": divergence_count,
                "divergence_fraction": float(divergence_count / (int(SPEC["chains"]) * int(SPEC["draws"]))),
                "wall_clock_s": float(time.perf_counter() - start),
                "adapted_step_size_min": float(min(d["adapted_step_size"] for d in chain_diag)),
                "adapted_step_size_median": float(np.median([d["adapted_step_size"] for d in chain_diag])),
                "mean_accept_prob_median": float(np.median([d["mean_accept_prob"] for d in chain_diag])),
                "max_treedepth_fraction_max": float(max(d["max_treedepth_fraction"] for d in chain_diag)),
                "frozen_chain_count": int(sum(d["adapted_step_size"] < 1e-5 for d in chain_diag)),
                "chain_diagnostics_json": json.dumps(chain_diag),
                **diag,
            }
        )
        for level in COVERAGE_LEVELS:
            lo = (1.0 - level) / 2.0
            hi = 1.0 - lo
            qlo, qhi = np.quantile(ratio, [lo, hi]) if len(ratio) else (np.nan, np.nan)
            base[f"covered_{int(level * 100)}"] = bool(qlo <= float(row.truth_ratio) <= qhi)
        base["pass"] = bool(base["max_rhat"] <= 1.01 and base["min_bulk_ess"] >= 400.0 and divergence_count == 0)
        return base, trace_rows(curve_id, samples_chain, extra)
    except Exception as exc:
        base.update(
            {
                "fit_error": str(exc),
                "pass": False,
                "divergence_count": np.nan,
                "frozen_chain_count": np.nan,
                "wall_clock_s": float(time.perf_counter() - start),
            }
        )
        return base, []


def run_s9(root: Path, selected: pd.DataFrame, arrays: dict[int, dict[str, np.ndarray]], t_full: np.ndarray, kappa: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rpath = rows_path(root)
    tpath = traces_path(root)
    rows = pd.read_csv(rpath, low_memory=False).to_dict("records") if rpath.exists() and rpath.stat().st_size > 1 else []
    traces = pd.read_csv(tpath, low_memory=False) if tpath.exists() and tpath.stat().st_size > 1 else pd.DataFrame()
    done_ids = {int(r["curve_id"]) for r in rows} if rows else set()
    lm = load_lm_fits(root)
    selected_lm = selected.merge(lm, left_on="template_raw_index", right_on="raw_index", how="left")
    kappa_lookup = dict(zip(kappa["curve_id"].astype(int), kappa["fisher_kappa"].astype(float)))
    configure(root)
    for row in selected_lm.sort_values("curve_id").itertuples(index=False):
        curve_id = int(row.curve_id)
        if curve_id in done_ids:
            continue
        out, trace = run_one_curve(row, arrays, t_full, kappa_lookup[curve_id], row)
        rows.append(out)
        pd.DataFrame(rows).sort_values("curve_id").to_csv(rpath, index=False)
        if trace:
            traces = pd.concat([traces[traces["curve_id"].astype(int) != curve_id] if len(traces) and "curve_id" in traces else traces, pd.DataFrame(trace)], ignore_index=True)
            traces.to_csv(tpath, index=False)
        print(
            f"S9 progress {len(rows)}/{len(selected)} curve={curve_id} init={out.get('init_source')} pass={out.get('pass')} "
            f"rhat={float(out.get('max_rhat', np.nan)):.3g} ess={float(out.get('min_bulk_ess', np.nan)):.1f} "
            f"div={out.get('divergence_count')} frozen={out.get('frozen_chain_count')}",
            flush=True,
        )
    return pd.DataFrame(rows).sort_values("curve_id").reset_index(drop=True), traces


def coverage_summary(rows: pd.DataFrame) -> dict[str, Any]:
    out = {"n": int(len(rows))}
    for level in COVERAGE_LEVELS:
        col = f"covered_{int(level * 100)}"
        n = int(rows[col].notna().sum()) if col in rows else 0
        empirical = float(rows[col].mean()) if n else np.nan
        lo, hi = binomial_interval(n, level)
        out[f"coverage_{int(level * 100)}"] = empirical
        out[f"coverage_{int(level * 100)}_ci_low"] = lo
        out[f"coverage_{int(level * 100)}_ci_high"] = hi
        out[f"coverage_{int(level * 100)}_nominal"] = bool(np.isfinite(empirical) and lo <= empirical <= hi)
    return out


def s8_frozen_counts(root: Path) -> pd.DataFrame:
    path = root / S8_CURVES_PATH
    if not path.exists():
        return pd.DataFrame(columns=["curve_id", "s8_pass", "s8_frozen_chain_count"])
    rows = pd.read_csv(path, low_memory=False)
    out = []
    for _, row in rows.iterrows():
        count = np.nan
        try:
            diag = json.loads(row["chain_diagnostics_json"])
            count = int(sum(float(d.get("adapted_step_size", np.nan)) < 1e-5 for d in diag))
        except Exception:
            pass
        out.append({"curve_id": int(row["curve_id"]), "s8_pass": bool(row["pass"]), "s8_frozen_chain_count": count})
    return pd.DataFrame(out)


def pass_fraction(rows: pd.DataFrame) -> dict[str, Any]:
    return {"n": int(len(rows)), "passed": int(rows["pass"].sum()) if len(rows) else 0, "fraction": float(rows["pass"].mean()) if len(rows) else np.nan}


def best_kappa_split(rows: pd.DataFrame) -> dict[str, Any]:
    vals = sorted(set(float(x) for x in rows["fisher_kappa"] if np.isfinite(x)))
    best = {"cutoff": np.nan, "correct": -1, "low_fail_exceptions": np.nan, "high_pass_exceptions": np.nan}
    for cutoff in vals:
        low = rows[rows["fisher_kappa"] < cutoff]
        high = rows[rows["fisher_kappa"] >= cutoff]
        low_fail = int((~low["pass"]).sum())
        high_pass = int(high["pass"].sum())
        correct = int(low["pass"].sum()) + int((~high["pass"]).sum())
        if correct > best["correct"]:
            best = {"cutoff": float(cutoff), "correct": correct, "low_fail_exceptions": low_fail, "high_pass_exceptions": high_pass}
    return best


def summarize(root: Path, curves: pd.DataFrame, traces: pd.DataFrame, counts: dict[str, int]) -> tuple[dict[str, Any], pd.DataFrame]:
    s8 = s8_frozen_counts(root)
    comparison = curves.merge(s8, on="curve_id", how="left")
    comparison["s9_pass"] = comparison["pass"].astype(bool)
    comparison["s9_frozen_chain_count"] = comparison["frozen_chain_count"]
    comparison["was_s8_failure_declared"] = comparison["curve_id"].astype(int).isin(S8_FAILING_CURVES_DECLARED)
    comparison["was_s8_failure_observed"] = comparison["s8_pass"].eq(False)
    s8_failures = comparison[comparison["was_s8_failure_observed"].fillna(False)].copy()
    overall = pass_fraction(curves)
    s8_pass_n = int(comparison["s8_pass"].fillna(False).sum()) if "s8_pass" in comparison else 0
    low = pass_fraction(curves[curves["fisher_kappa"] < KAPPA_CUTOFF])
    high = pass_fraction(curves[curves["fisher_kappa"] >= KAPPA_CUTOFF])
    severity = {str(k): pass_fraction(v) | {"conclusion_allowed": bool(len(v) > 1)} for k, v in curves.groupby("truth_severity", dropna=False)}
    if overall["fraction"] >= 0.90 and int(curves["frozen_chain_count"].sum()) == 0:
        verdict = "INIT-FIXES-IT"
    elif overall["passed"] > s8_pass_n:
        verdict = "INIT-HELPS-PARTIAL"
    else:
        verdict = "INIT-INSUFFICIENT"
    result = {
        "verdict": verdict,
        "realized_stratum_counts": counts,
        "overall_pass": overall,
        "s8_passed_on_same_curves": s8_pass_n,
        "s8_to_s9_pass_delta": int(overall["passed"] - s8_pass_n),
        "kappa_lt_50000_pass": low,
        "kappa_gte_50000_pass": high,
        "best_kappa_split": best_kappa_split(curves),
        "severity_pass": severity,
        "coverage_all": coverage_summary(curves),
        "coverage_converged_only": coverage_summary(curves[curves["pass"]]),
        "s8_failure_comparison": comparison[
            [
                "curve_id",
                "truth_severity",
                "was_s8_failure_declared",
                "was_s8_failure_observed",
                "s8_pass",
                "s9_pass",
                "s8_frozen_chain_count",
                "s9_frozen_chain_count",
                "max_rhat",
                "min_bulk_ess",
                "divergence_count",
            ]
        ].to_dict("records"),
        "s8_failures_now_passed": int(s8_failures["s9_pass"].sum()) if len(s8_failures) else 0,
        "s8_failures_n": int(len(s8_failures)),
        "lm_fallback_n": int(curves["init_source"].eq("prior_fallback").sum()),
        "lm_fallback_count": int(curves["init_source"].eq("prior_fallback").sum()),
        "frozen_chains_s8_total": int(np.nansum(comparison["s8_frozen_chain_count"])) if "s8_frozen_chain_count" in comparison else None,
        "frozen_chains_s9_total": int(np.nansum(comparison["s9_frozen_chain_count"])),
    }
    return result, comparison


def write_outputs(root: Path, result: dict[str, Any], curves: pd.DataFrame, traces: pd.DataFrame, comparison: pd.DataFrame) -> None:
    out = root / "results"
    out.mkdir(exist_ok=True)
    curves_out = curves.sort_values("fisher_kappa").reset_index(drop=True)
    curves_out.to_csv(out / f"{OUTPUT_STEM}_curves.csv", index=False)
    failing = set(curves[curves["pass"].eq(False)]["curve_id"].astype(int))
    trace_out = traces[traces["curve_id"].astype(int).isin(failing)].copy() if len(traces) else pd.DataFrame()
    trace_out.to_csv(out / f"{OUTPUT_STEM}_traces.csv", index=False)
    comparison.to_csv(out / f"{OUTPUT_STEM}_comparison.csv", index=False)
    (out / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Gate - S9 Mode-Initialized Micro-Pilot, Cycle F",
        "",
        "## Protocol Decisions",
        "",
        "- S9 reuses the exact S8 20-curve selection and sampler settings.",
        "- The only sampler change is per-chain initialization from a Gaussian ball around the stored deterministic LM fit.",
        "- LM fits are loaded from `results/gate_policy_calibrated_flag_F_rows.csv` using each synthetic row's `template_raw_index`; no LM refit is performed.",
        "- Initialization sd is 5% of each wide-prior natural-parameter range and values are clipped to prior support.",
        "- Per-curve convergence requires R-hat <= 1.01, ESS >= 400, and zero divergent transitions.",
        "",
        f"## Verdict: {result['verdict']}",
        "",
        f"- realized stratum counts: {result['realized_stratum_counts']}",
        f"- LM fallback count: {result['lm_fallback_n']}",
        f"- overall pass: {result['overall_pass']['passed']} / {result['overall_pass']['n']} ({result['overall_pass']['fraction']:.3f})",
        f"- S8 passed on same curves: {result['s8_passed_on_same_curves']} / {result['overall_pass']['n']}; S9 delta: {result['s8_to_s9_pass_delta']}",
        f"- S8 observed failures now passing: {result['s8_failures_now_passed']} / {result['s8_failures_n']}",
        f"- frozen chains total S8 -> S9: {result['frozen_chains_s8_total']} -> {result['frozen_chains_s9_total']}",
        f"- kappa < 50000 pass: {result['kappa_lt_50000_pass']['passed']} / {result['kappa_lt_50000_pass']['n']} ({result['kappa_lt_50000_pass']['fraction']:.3f})",
        f"- kappa >= 50000 pass: {result['kappa_gte_50000_pass']['passed']} / {result['kappa_gte_50000_pass']['n']} ({result['kappa_gte_50000_pass']['fraction']:.3f})",
        "",
        "## Curves Sorted By Fisher Kappa",
        "",
        md_table(
            curves_out.to_dict("records"),
            [
                ("curve_id", "curve", "int"),
                ("truth_severity", "severity", "str"),
                ("fisher_kappa", "Fisher kappa", "float1"),
                ("max_rhat", "R-hat", "float3"),
                ("min_bulk_ess", "ESS", "float1"),
                ("divergence_count", "div", "int"),
                ("wall_clock_s", "sec", "float2"),
                ("frozen_chain_count", "frozen", "int"),
                ("init_source", "init", "str"),
                ("pass", "pass", "str"),
            ],
        ),
        "",
        "## Critical S8 vs S9 Comparison",
        "",
        md_table(
            comparison.sort_values("curve_id").to_dict("records"),
            [
                ("curve_id", "curve", "int"),
                ("truth_severity", "severity", "str"),
                ("s8_pass", "S8 pass", "str"),
                ("s9_pass", "S9 pass", "str"),
                ("s8_frozen_chain_count", "S8 frozen", "int"),
                ("s9_frozen_chain_count", "S9 frozen", "int"),
                ("was_s8_failure_declared", "declared S8 fail", "str"),
                ("was_s8_failure_observed", "observed S8 fail", "str"),
            ],
        ),
        "",
        "## Coverage",
        "",
        md_table(
            [{"sample": "all", **result["coverage_all"]}, {"sample": "converged_only", **result["coverage_converged_only"]}],
            [
                ("sample", "sample", "str"),
                ("n", "n", "int"),
                ("coverage_50", "cov50", "float3"),
                ("coverage_50_ci_low", "50 lo", "float3"),
                ("coverage_50_ci_high", "50 hi", "float3"),
                ("coverage_80", "cov80", "float3"),
                ("coverage_80_ci_low", "80 lo", "float3"),
                ("coverage_80_ci_high", "80 hi", "float3"),
                ("coverage_95", "cov95", "float3"),
                ("coverage_95_ci_low", "95 lo", "float3"),
                ("coverage_95_ci_high", "95 hi", "float3"),
            ],
        ),
        "",
        "## Severity Pass Fractions",
        "",
        md_table(
            [{"severity": k, **v} for k, v in result["severity_pass"].items()],
            [("severity", "severity", "str"), ("n", "n", "int"), ("passed", "passed", "int"), ("fraction", "fraction", "float3"), ("conclusion_allowed", "conclude?", "str")],
        ),
        "",
        "Trace dump for still-failing curves: `results/gate_s9_modeinit_F_traces.csv`.",
        "",
    ]
    (out / f"{OUTPUT_STEM}.md").write_text("\n".join(lines), encoding="utf-8")


def run(root: Path) -> dict[str, Any]:
    _pilot, full_data, arrays, t_full = build_data(root)
    selected = select_stratified(full_data)
    kappa = fisher_kappa_for_rows(selected, arrays)
    selected = selected.merge(kappa[["curve_id", "fisher_kappa"]], on="curve_id", how="left")
    curves, traces = run_s9(root, selected, arrays, t_full, kappa)
    result, comparison = summarize(root, curves, traces, selected["truth_severity"].value_counts().to_dict())
    result["config"] = {**SPEC, "rung": "S9"}
    result["init_sd_fraction_of_prior_range"] = INIT_SD_FRACTION
    write_outputs(root, result, curves, traces, comparison)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    print(f"VERDICT: {result['verdict']}")
    print(f"Overall pass: {result['overall_pass']}")
    print(f"Frozen chains S8 -> S9: {result['frozen_chains_s8_total']} -> {result['frozen_chains_s9_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
