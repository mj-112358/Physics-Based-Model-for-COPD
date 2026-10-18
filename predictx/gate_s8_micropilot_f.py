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
from .gate_posterior_calibration_repair_f import ratio_samples_from_phi
from .gate_sampler_pilot_f import build_data
from .ombe import OmbeGrid, batched_fisher_information

OUTPUT_STEM = "gate_s8_micropilot_F"
CACHE_VERSION = "v1_s8_micro20"
VARIANT = "L0_baseline"
SPEC = {"rung": "S8", "chains": 4, "warmup": 2000, "draws": 2000, "target_accept": 0.95, "dense_mass": True, "max_tree_depth": 10}
TARGET_COUNTS = {"mild_0.60_0.70": 9, "normal": 9, "moderate": 1, "severe": 1}
KAPPA_CUTOFF = 50_000.0
TRACE_PARAMS = ("tau_f", "tau_s", "alpha", "fvc_l")


def configure(root: Path) -> None:
    cache = root / ".cache"
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))
    numpyro.set_host_device_count(int(SPEC["chains"]))


def cache_dir(root: Path) -> Path:
    out = root / "results" / "_cache_s8_micropilot"
    out.mkdir(parents=True, exist_ok=True)
    return out


def rows_path(root: Path) -> Path:
    return cache_dir(root) / f"{CACHE_VERSION}_rows.csv"


def samples_path(root: Path, curve_id: int) -> Path:
    return cache_dir(root) / f"{CACHE_VERSION}_curve_{curve_id}_samples.npz"


def select_stratified(full_data: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for stratum, target in TARGET_COUNTS.items():
        pool = full_data[full_data["truth_severity"].eq(stratum)].copy()
        n = min(target, len(pool))
        if n:
            parts.append(pool.sample(n=n, random_state=SEED + 8080 + len(parts)))
    if not parts:
        raise RuntimeError("No curves available for S8 micro-pilot selection.")
    selected = pd.concat(parts, ignore_index=True).sort_values(["truth_severity", "curve_id"]).reset_index(drop=True)
    counts = selected["truth_severity"].value_counts().to_dict()
    print(f"S8 realized stratum counts: {counts}", flush=True)
    if len(counts) == 1:
        raise RuntimeError(f"Sampling bug: all selected curves are one stratum: {counts}")
    return selected


def fisher_kappa_for_rows(selected: pd.DataFrame, arrays: dict[int, dict[str, np.ndarray]]) -> pd.DataFrame:
    t = OmbeGrid().t
    phi = np.vstack([arrays[int(cid)]["phi"] for cid in selected["curve_id"]])
    fisher = np.asarray(batched_fisher_information(jnp.asarray(phi), t, 0.10))
    eig = np.linalg.eigvalsh(fisher)
    eig = np.maximum(eig, np.finfo(float).tiny)
    out = selected[["curve_id", "truth_severity", "truth_ratio"]].copy()
    out["fisher_kappa"] = eig[:, -1] / eig[:, 0]
    return out


def natural_samples(samples_chain: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(samples_chain[name], dtype=float) for name in TRACE_PARAMS}


def chain_diag_from_extra(extra: dict[str, Any]) -> list[dict[str, Any]]:
    accept = np.asarray(extra["accept_prob"], dtype=float)
    diverging = np.asarray(extra["diverging"], dtype=bool)
    num_steps = np.asarray(extra["num_steps"], dtype=float)
    step_size = np.asarray(extra["adapt_state.step_size"], dtype=float)
    max_steps = (2 ** int(SPEC["max_tree_depth"])) - 1
    rows = []
    for chain in range(accept.shape[0]):
        rows.append(
            {
                "chain": int(chain),
                "adapted_step_size": float(np.nanmedian(step_size[chain])),
                "mean_accept_prob": float(np.nanmean(accept[chain])),
                "divergent_transitions": int(diverging[chain].sum()),
                "tree_depth_median": float(np.nanmedian(np.ceil(np.log2(num_steps[chain] + 1.0)))),
                "max_treedepth_fraction": float((num_steps[chain] >= max_steps).mean()),
            }
        )
    return rows


def trace_rows(curve_id: int, samples_chain: dict[str, np.ndarray], extra: dict[str, Any]) -> list[dict[str, Any]]:
    chain_diag = {row["chain"]: row for row in chain_diag_from_extra(extra)}
    rows = []
    natural = natural_samples(samples_chain)
    for param, arr in natural.items():
        means = arr.mean(axis=1)
        sds = arr.std(axis=1, ddof=1)
        pooled = float(np.nanstd(arr.reshape(-1), ddof=1))
        spread = float(np.nanmax(means) - np.nanmin(means))
        for chain, (mean, sd) in enumerate(zip(means, sds)):
            diag = chain_diag[chain]
            rows.append(
                {
                    "curve_id": int(curve_id),
                    "chain": int(chain),
                    "param": "FVC" if param == "fvc_l" else param,
                    "mean": float(mean),
                    "sd": float(sd),
                    "between_chain_spread": spread,
                    "rhat_contrib": float(abs(mean - np.nanmean(means)) / pooled) if pooled > 0 else np.nan,
                    **diag,
                }
            )
    return rows


def run_one_curve(row: Any, arrays: dict[int, dict[str, np.ndarray]], t_full: np.ndarray, fisher_kappa: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    curve_id = int(row.curve_id)
    arr = arrays[curve_id]
    t_obs = arr["t"]
    y_obs = arr["y"]
    base: dict[str, Any] = {
        "row_type": "s8_curve",
        "rung": "S8",
        "variant": VARIANT,
        "curve_id": curve_id,
        "truth_severity": str(row.truth_severity),
        "truth_ratio": float(row.truth_ratio),
        "fisher_kappa": float(fisher_kappa),
        "chains": int(SPEC["chains"]),
        "warmup": int(SPEC["warmup"]),
        "draws": int(SPEC["draws"]),
        "target_accept": float(SPEC["target_accept"]),
        "dense_mass": bool(SPEC["dense_mass"]),
        "fit_error": "",
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
        base.update({"fit_error": str(exc), "pass": False, "divergence_count": np.nan, "wall_clock_s": float(time.perf_counter() - start)})
        return base, []


def run_s8(root: Path, selected: pd.DataFrame, arrays: dict[int, dict[str, np.ndarray]], t_full: np.ndarray, kappa: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rpath = rows_path(root)
    if rpath.exists() and rpath.stat().st_size > 1:
        done = pd.read_csv(rpath, low_memory=False)
    else:
        done = pd.DataFrame()
    done_ids = set(done["curve_id"].astype(int)) if len(done) and "curve_id" in done else set()
    rows = done.to_dict("records") if len(done) else []
    configure(root)
    kappa_lookup = dict(zip(kappa["curve_id"].astype(int), kappa["fisher_kappa"].astype(float)))
    selected = selected.sort_values("curve_id").reset_index(drop=True)
    for row in selected.itertuples(index=False):
        curve_id = int(row.curve_id)
        if curve_id in done_ids:
            continue
        out, traces = run_one_curve(row, arrays, t_full, kappa_lookup[curve_id])
        rows.append(out)
        pd.DataFrame(rows).sort_values("curve_id").to_csv(rpath, index=False)
        if traces:
            pd.DataFrame(traces).to_csv(samples_path(root, curve_id).with_suffix(".trace.csv"), index=False)
        completed = len(rows)
        print(
            f"S8 progress {completed}/{len(selected)} curve={curve_id} pass={out.get('pass')} "
            f"rhat={float(out.get('max_rhat', np.nan)):.3g} ess={float(out.get('min_bulk_ess', np.nan)):.1f} "
            f"div={out.get('divergence_count')} step={float(out.get('adapted_step_size_median', np.nan)):.3g}",
            flush=True,
        )
    curve_rows = pd.DataFrame(rows).sort_values("curve_id").reset_index(drop=True)
    trace_rows_all = []
    for curve_id in curve_rows["curve_id"].astype(int):
        path = samples_path(root, curve_id).with_suffix(".trace.csv")
        if path.exists():
            trace_rows_all.append(pd.read_csv(path))
    traces = pd.concat(trace_rows_all, ignore_index=True) if trace_rows_all else pd.DataFrame()
    return curve_rows, traces


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


def summarize(rows: pd.DataFrame, traces: pd.DataFrame, counts: dict[str, int]) -> dict[str, Any]:
    rows = rows.copy()
    overall = pass_fraction(rows)
    low = pass_fraction(rows[rows["fisher_kappa"] < KAPPA_CUTOFF])
    high = pass_fraction(rows[rows["fisher_kappa"] >= KAPPA_CUTOFF])
    severity = {str(k): pass_fraction(v) | {"conclusion_allowed": bool(len(v) > 1)} for k, v in rows.groupby("truth_severity", dropna=False)}
    coverage_all = coverage_summary(rows)
    coverage_converged = coverage_summary(rows[rows["pass"]])
    wall = rows["wall_clock_s"].to_numpy(float)
    wall = wall[np.isfinite(wall)]
    worst = rows[rows["pass"].eq(False)].sort_values("max_rhat", ascending=False).head(3).copy()
    if len(worst) == 0:
        worst = rows.sort_values("max_rhat", ascending=False).head(3).copy()
    worst_curve_ids = [int(x) for x in worst["curve_id"]]
    trace_worst = traces[traces["curve_id"].isin(worst_curve_ids)].copy() if len(traces) else pd.DataFrame()
    chain_summary = pd.DataFrame()
    tiny_chain_fraction = np.nan
    max_tree_chain_fraction = np.nan
    if len(trace_worst):
        chain_summary = trace_worst[
            [
                "curve_id",
                "chain",
                "adapted_step_size",
                "mean_accept_prob",
                "divergent_transitions",
                "max_treedepth_fraction",
            ]
        ].drop_duplicates(["curve_id", "chain"])
        tiny_chain_fraction = float((chain_summary["adapted_step_size"] < 1e-5).mean()) if len(chain_summary) else np.nan
        max_tree_chain_fraction = float((chain_summary["max_treedepth_fraction"] >= 0.95).mean()) if len(chain_summary) else np.nan
    spread_large = False
    if len(trace_worst):
        spread_large = bool(trace_worst.groupby(["curve_id", "param"])["between_chain_spread"].first().median() > 0.05)
    if np.isfinite(tiny_chain_fraction) and tiny_chain_fraction >= 0.25 and spread_large:
        mechanism = (
            f"Mixed frozen-chain non-mixing: {tiny_chain_fraction:.3f} of worst-curve chains have adapted step size <1e-5, "
            f"{max_tree_chain_fraction:.3f} hit max treedepth on nearly all transitions, and per-chain parameter means are widely separated."
        )
    elif spread_large:
        mechanism = "Failing chains show large between-chain spread without tiny median step size; multimodality/non-mixing is more supported than pure step-size death."
    else:
        mechanism = "No single failure mechanism established from step-size and trace summaries."
    split = best_kappa_split(rows)
    kappa_split = bool(low["fraction"] >= 0.80 and high["fraction"] <= 0.40)
    if overall["fraction"] >= 0.90:
        verdict = "BUDGET-FIXES-IT"
    elif kappa_split:
        verdict = "KAPPA-SPLIT"
    else:
        verdict = "BUDGET-INSUFFICIENT"
    return {
        "verdict": verdict,
        "realized_stratum_counts": counts,
        "overall_pass": overall,
        "kappa_lt_50000_pass": low,
        "kappa_gte_50000_pass": high,
        "best_kappa_split": split,
        "severity_pass": severity,
        "wall_clock_s": {
            "median": float(np.median(wall)) if len(wall) else np.nan,
            "iqr": [float(x) for x in np.percentile(wall, [25, 75])] if len(wall) else [np.nan, np.nan],
        },
        "coverage_all": coverage_all,
        "coverage_converged_only": coverage_converged,
        "worst_rhat_curve_ids": worst_curve_ids,
        "worst_chain_tiny_step_fraction": tiny_chain_fraction,
        "worst_chain_max_treedepth_fraction": max_tree_chain_fraction,
        "failure_mechanism": mechanism,
    }


def write_outputs(root: Path, result: dict[str, Any], curves: pd.DataFrame, traces: pd.DataFrame) -> None:
    out = root / "results"
    out.mkdir(exist_ok=True)
    curves_out = curves.sort_values("fisher_kappa").reset_index(drop=True)
    curves_out.to_csv(out / f"{OUTPUT_STEM}_curves.csv", index=False)
    if len(traces):
        worst = set(result["worst_rhat_curve_ids"])
        traces = traces[traces["curve_id"].isin(worst)].copy()
    traces.to_csv(out / f"{OUTPUT_STEM}_traces.csv", index=False)
    (out / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Gate - S8 Micro-Pilot, Cycle F",
        "",
        "## Protocol Decisions",
        "",
        "- S8 is identical to S7 except warmup/draws are 2000/2000 instead of 1000/1000.",
        "- The 20 curves are sampled across Cell A stratum blocks at target proportions 9 mild / 9 normal / 1 moderate / 1 severe.",
        "- Per-curve convergence requires R-hat <= 1.01, ESS >= 400, and zero divergent transitions.",
        "- Coverage is reported at n=20 and again after excluding non-converged curves; n=1 severity strata are reported but not interpreted.",
        "",
        f"## Verdict: {result['verdict']}",
        "",
        f"- realized stratum counts: {result['realized_stratum_counts']}",
        f"- overall pass: {result['overall_pass']['passed']} / {result['overall_pass']['n']} ({result['overall_pass']['fraction']:.3f})",
        f"- kappa < 50000 pass: {result['kappa_lt_50000_pass']['passed']} / {result['kappa_lt_50000_pass']['n']} ({result['kappa_lt_50000_pass']['fraction']:.3f})",
        f"- kappa >= 50000 pass: {result['kappa_gte_50000_pass']['passed']} / {result['kappa_gte_50000_pass']['n']} ({result['kappa_gte_50000_pass']['fraction']:.3f})",
        f"- best kappa split: {result['best_kappa_split']}",
        f"- wall-clock seconds per curve median/IQR: {result['wall_clock_s']['median']:.2f} / {result['wall_clock_s']['iqr']}",
        f"- failure mechanism: {result['failure_mechanism']}",
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
                ("divergence_fraction", "div frac", "float6"),
                ("wall_clock_s", "sec", "float2"),
                ("pass", "pass", "str"),
            ],
        ),
        "",
        "## Coverage",
        "",
        md_table(
            [
                {"sample": "all", **result["coverage_all"]},
                {"sample": "converged_only", **result["coverage_converged_only"]},
            ],
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
        "## Worst-Rhat Failure Instrumentation",
        "",
        f"- worst curve ids: {result['worst_rhat_curve_ids']}",
        f"- worst-curve chains with adapted step size <1e-5: {result.get('worst_chain_tiny_step_fraction', np.nan):.3f}",
        f"- worst-curve chains hitting max treedepth on nearly all transitions: {result.get('worst_chain_max_treedepth_fraction', np.nan):.3f}",
        "- `results/gate_s8_micropilot_F_traces.csv` contains curve, chain, param, mean, sd, rhat_contrib, adapted step size, mean accept probability, divergences, and tree-depth hit fraction.",
        "",
    ]
    if len(traces):
        chain_rows = traces[
            [
                "curve_id",
                "chain",
                "adapted_step_size",
                "mean_accept_prob",
                "divergent_transitions",
                "max_treedepth_fraction",
            ]
        ].drop_duplicates(["curve_id", "chain"])
        lines.extend(
            [
                md_table(
                    chain_rows.to_dict("records"),
                    [
                        ("curve_id", "curve", "int"),
                        ("chain", "chain", "int"),
                        ("adapted_step_size", "step size", "float6"),
                        ("mean_accept_prob", "accept", "float3"),
                        ("divergent_transitions", "div", "int"),
                        ("max_treedepth_fraction", "max depth frac", "float3"),
                    ],
                ),
                "",
            ]
        )
    (out / f"{OUTPUT_STEM}.md").write_text("\n".join(lines), encoding="utf-8")


def run(root: Path) -> dict[str, Any]:
    start = time.perf_counter()
    _pilot, full_data, arrays, t_full = build_data(root)
    selected = select_stratified(full_data)
    counts = selected["truth_severity"].value_counts().to_dict()
    kappa = fisher_kappa_for_rows(selected, arrays)
    selected = selected.merge(kappa[["curve_id", "fisher_kappa"]], on="curve_id", how="left")
    curves, traces = run_s8(root, selected, arrays, t_full, kappa)
    result = summarize(curves, traces, counts)
    result["wall_clock_total_s"] = float(time.perf_counter() - start)
    result["config"] = SPEC
    write_outputs(root, result, curves, traces)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    print(f"VERDICT: {result['verdict']}")
    print(f"Overall pass: {result['overall_pass']}")
    print(f"Failure mechanism: {result['failure_mechanism']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
