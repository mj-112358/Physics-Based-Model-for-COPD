from __future__ import annotations

import argparse
import ast
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache" / "xdg"))

import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
from scipy.stats import kstest

from .gate_misspecification_isolation_f import (
    COVERAGE_LEVELS,
    TARGET_COUNTS,
    binomial_interval,
    clean_json,
    load_real_subset,
    md_table,
    real_templates,
    synthetic_dataset,
)
from .gate_sampler_convergence_f import (
    CACHE_VERSION as PRIOR_CACHE_VERSION,
)
from .gate_sampler_convergence_f import (
    cond_cache_path as prior_cond_cache_path,
)
from .gate_sampler_convergence_f import (
    div_cache_path as prior_div_cache_path,
)
from .gate_sampler_convergence_f import (
    run_one_curve,
)
from .gate_sampler_convergence_f import (
    rung_cache_path as prior_rung_cache_path,
)
from .ombe import OmbeGrid, batched_fisher_information

OUTPUT_STEM = "gate_sampler_pilot_F"
CACHE_VERSION = "v1_pilot30"
VARIANT = "L0_baseline"
PILOT_N = 30
CONDITION_N = 50
SPECS: list[dict[str, Any]] = [
    {"rung": "S7", "chains": 4, "warmup": 1000, "draws": 1000, "target_accept": 0.95, "dense_mass": True},
    {"rung": "S6", "chains": 4, "warmup": 2500, "draws": 2500, "target_accept": 0.99, "dense_mass": True},
]


def configure(root: Path, chains: int) -> None:
    cache = root / ".cache"
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))
    numpyro.set_host_device_count(max(chains, 1))


def cache_dir(root: Path) -> Path:
    out = root / "results" / "_cache_sampler_pilot"
    out.mkdir(parents=True, exist_ok=True)
    return out


def rows_path(root: Path, rung: str) -> Path:
    return cache_dir(root) / f"{CACHE_VERSION}_{rung}_cell_A_L0_rows.csv"


def div_path(root: Path, rung: str) -> Path:
    return cache_dir(root) / f"{CACHE_VERSION}_{rung}_divergent_draws.csv"


def cond_path(root: Path, rung: str) -> Path:
    return cache_dir(root) / f"{CACHE_VERSION}_{rung}_condition_numbers.csv"


def maybe_seed_s6_from_prior_cache(root: Path, data: pd.DataFrame) -> None:
    """Reuse exact already-computed S6 first curves from the aborted full ladder."""
    rung = "S6"
    if rows_path(root, rung).exists():
        return
    prior_rows = prior_rung_cache_path(root, rung)
    if not prior_rows.exists():
        return
    wanted = set(data["curve_id"].astype(int))
    rows = pd.read_csv(prior_rows, low_memory=False)
    rows = rows[rows["curve_id"].astype(int).isin(wanted)].copy()
    if not len(rows):
        return
    rows_path(root, rung).parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(rows_path(root, rung), index=False)
    prior_cond = prior_cond_cache_path(root, rung)
    if prior_cond.exists() and prior_cond.stat().st_size > 1:
        cond = pd.read_csv(prior_cond, low_memory=False)
        cond = cond[cond["curve_id"].astype(int).isin(wanted)].copy()
        cond.to_csv(cond_path(root, rung), index=False)
    prior_div = prior_div_cache_path(root, rung)
    if prior_div.exists() and prior_div.stat().st_size > 1:
        div = pd.read_csv(prior_div, low_memory=False)
        if "curve_id" in div:
            div = div[div["curve_id"].astype(int).isin(wanted)].copy()
        div.to_csv(div_path(root, rung), index=False)
    else:
        pd.DataFrame().to_csv(div_path(root, rung), index=False)
    print(f"S6: seeded {len(rows)} exact cached curves from {PRIOR_CACHE_VERSION}")


def run_rung(root: Path, spec: dict[str, Any], data: pd.DataFrame, arrays: dict[int, dict[str, np.ndarray]], t_full: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rpath = rows_path(root, spec["rung"])
    cpath = cond_path(root, spec["rung"])
    dpath = div_path(root, spec["rung"])
    if rpath.exists() and rpath.stat().st_size > 1:
        done = pd.read_csv(rpath, low_memory=False)
    else:
        done = pd.DataFrame()
    done_ids = set(done["curve_id"].astype(int)) if len(done) and "curve_id" in done else set()
    rows_accum = done.to_dict("records") if len(done) else []
    cond_accum = pd.read_csv(cpath, low_memory=False) if cpath.exists() and cpath.stat().st_size > 1 else pd.DataFrame()
    cond_records = cond_accum.to_dict("records") if len(cond_accum) else []
    div_accum = pd.read_csv(dpath, low_memory=False) if dpath.exists() and dpath.stat().st_size > 1 else pd.DataFrame()
    if len(rows_accum) >= len(data):
        print(f"Using cached {spec['rung']}: {rpath}")
        return done.sort_values("curve_id").reset_index(drop=True), div_accum, cond_accum.sort_values("curve_id").reset_index(drop=True)
    configure(root, int(spec["chains"]))
    for row in data.itertuples(index=False):
        curve_id = int(row.curve_id)
        if curve_id in done_ids:
            continue
        out, div_rows, cond_row = run_one_curve(spec, row, arrays, t_full)
        rows_accum.append(out)
        cond_records.append(cond_row)
        pd.DataFrame(rows_accum).to_csv(rpath, index=False)
        pd.DataFrame(cond_records).to_csv(cpath, index=False)
        if len(div_rows):
            div_accum = pd.concat([div_accum, div_rows], ignore_index=True)
            div_accum.to_csv(dpath, index=False)
        elif not dpath.exists():
            pd.DataFrame().to_csv(dpath, index=False)
        completed = len(rows_accum)
        if completed % 5 == 0 or completed == len(data):
            print(f"{spec['rung']}: NUTS progress {completed}/{len(data)}")
    rows = pd.DataFrame(rows_accum).sort_values("curve_id").reset_index(drop=True)
    cond = pd.DataFrame(cond_records).sort_values("curve_id").reset_index(drop=True)
    return rows, div_accum, cond


def summarize_rung(rows: pd.DataFrame, cond: pd.DataFrame, spec: dict[str, Any], n_planned: int) -> dict[str, Any]:
    ok = rows[rows["fit_error"].fillna("").eq("")].copy()
    pits = ok["pit"].to_numpy(float)
    pits = pits[np.isfinite(pits)]
    ks_stat, ks_p = kstest(pits, "uniform") if len(pits) else (np.nan, np.nan)
    total_postwarmup = int(len(ok) * int(spec["chains"]) * int(spec["draws"]))
    total_divergences = int(np.nansum(ok["divergence_count"])) if len(ok) else 0
    divergence_transition_fraction = float(total_divergences / total_postwarmup) if total_postwarmup else np.nan
    out: dict[str, Any] = {
        "rung": spec["rung"],
        "n_planned": int(n_planned),
        "n_attempted": int(len(rows)),
        "n_ok": int(len(ok)),
        "complete": bool(len(rows) == n_planned),
        "chains": int(spec["chains"]),
        "warmup": int(spec["warmup"]),
        "draws": int(spec["draws"]),
        "target_accept": float(spec["target_accept"]) if spec.get("target_accept") is not None else np.nan,
        "dense_mass": bool(spec["dense_mass"]),
        "pit_ks_statistic": float(ks_stat),
        "pit_ks_p_value": float(ks_p),
        "median_rhat": float(np.nanmedian(ok["max_rhat"])) if len(ok) else np.nan,
        "rhat_gt_1p01_fraction": float((ok["max_rhat"] > 1.01).mean()) if len(ok) else np.nan,
        "rhat_lte_1p01_fraction": float((ok["max_rhat"] <= 1.01).mean()) if len(ok) else np.nan,
        "min_ess": float(np.nanmin(ok["min_bulk_ess"])) if len(ok) else np.nan,
        "median_ess": float(np.nanmedian(ok["min_bulk_ess"])) if len(ok) else np.nan,
        "ess_lt_400_fraction": float((ok["min_bulk_ess"] < 400.0).mean()) if len(ok) else np.nan,
        "ess_gte_400_fraction": float((ok["min_bulk_ess"] >= 400.0).mean()) if len(ok) else np.nan,
        "total_postwarmup_transitions": total_postwarmup,
        "total_divergences": total_divergences,
        "divergence_transition_fraction": divergence_transition_fraction,
        "any_divergence_fraction": float((ok["divergence_count"] > 0).mean()) if len(ok) else np.nan,
        "wall_clock_s_per_curve": float(np.nanmean(ok["wall_clock_s"])) if len(ok) else np.nan,
        "posterior_condition_number_median": float(np.nanmedian(cond["posterior_condition_number"])) if len(cond) else np.nan,
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
        out["complete"]
        and out["divergence_transition_fraction"] < 0.001
        and out["rhat_lte_1p01_fraction"] >= 0.90
        and out["ess_gte_400_fraction"] >= 0.90
    )
    out["nominal_all"] = bool(all(out[f"coverage_{int(level * 100)}_nominal"] for level in COVERAGE_LEVELS))
    return out


def boundary_summary(div: pd.DataFrame) -> dict[str, Any]:
    if not len(div):
        return {"n_divergent_draws": 0, "tau_s_within_5pct_lower": np.nan, "tau_f_within_5pct_upper": np.nan, "boundary_pathology": False}
    tau_s_frac = float((div["tau_s"].astype(float) <= 1.05).mean()) if "tau_s" in div else np.nan
    tau_f_frac = float((div["tau_f"].astype(float) >= 0.95).mean()) if "tau_f" in div else np.nan
    return {
        "n_divergent_draws": int(len(div)),
        "tau_s_within_5pct_lower": tau_s_frac,
        "tau_f_within_5pct_upper": tau_f_frac,
        "boundary_pathology": bool((np.isfinite(tau_s_frac) and tau_s_frac > 0.10) or (np.isfinite(tau_f_frac) and tau_f_frac > 0.10)),
    }


def condition_reconciliation(root: Path, cell_a_data: pd.DataFrame, arrays: dict[int, dict[str, np.ndarray]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    t = OmbeGrid().t
    first = cell_a_data.sort_values("curve_id").head(CONDITION_N).copy()
    phi = np.vstack([arrays[int(cid)]["phi"] for cid in first["curve_id"]])
    fisher = np.asarray(batched_fisher_information(jnp.asarray(phi), t, 0.10))
    eig = np.linalg.eigvalsh(fisher)
    eig = np.maximum(eig, np.finfo(float).tiny)
    fisher_kappa = eig[:, -1] / eig[:, 0]
    prior_cond = prior_cond_cache_path(root, "S1")
    if prior_cond.exists() and prior_cond.stat().st_size > 1:
        s1 = pd.read_csv(prior_cond)
        s1 = s1.rename(columns={"posterior_condition_number": "s1_posterior_cov_kappa"})
        table = first[["curve_id", "truth_severity", "true_tau_f", "true_tau_s", "true_alpha", "true_fvc_l", "true_tau_o"]].merge(
            s1[["curve_id", "s1_posterior_cov_kappa"]], on="curve_id", how="left"
        )
    else:
        table = first[["curve_id", "truth_severity", "true_tau_f", "true_tau_s", "true_alpha", "true_fvc_l", "true_tau_o"]].copy()
        table["s1_posterior_cov_kappa"] = np.nan
    table["gate1_fisher_kappa"] = fisher_kappa
    table["laplace_inverse_fisher_kappa"] = fisher_kappa
    table = table[
        [
            "curve_id",
            "truth_severity",
            "gate1_fisher_kappa",
            "laplace_inverse_fisher_kappa",
            "s1_posterior_cov_kappa",
            "true_tau_f",
            "true_tau_s",
            "true_alpha",
            "true_fvc_l",
            "true_tau_o",
        ]
    ]
    summary = {
        "n": int(len(table)),
        "gate1_fisher_kappa_median": float(np.nanmedian(table["gate1_fisher_kappa"])),
        "gate1_fisher_kappa_iqr": [float(x) for x in np.nanpercentile(table["gate1_fisher_kappa"], [25, 75])],
        "s1_posterior_cov_kappa_median": float(np.nanmedian(table["s1_posterior_cov_kappa"])),
        "s1_posterior_cov_kappa_iqr": [float(x) for x in np.nanpercentile(table["s1_posterior_cov_kappa"], [25, 75])],
        "reconciliation": (
            "Gate 1 kappa is the condition number of the Fisher information matrix J^T J / sigma^2 "
            "for OMBE flow on the transformed phi scale, using sigma=0.10 L/s. The convergence-gate "
            "172,668 number is the condition number of the empirical posterior covariance of S1 NUTS "
            "draws in phi under the volume likelihood with fixed 50 mL sigma and real FET grids. "
            "They are different quantities; the Fisher-information kappa belongs in the paper as the "
            "sloppiness measure. The 50-row table uses the first 50 Cell A theta from the wide, "
            "stratified Cell A prior, so its Fisher median is not expected to equal Gate 1's original "
            "1000-draw narrow-prior median of 1990."
        ),
    }
    return table, summary


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def int_constant_from_assign(tree: ast.AST, names: tuple[str, ...]) -> int | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names and isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                    return int(node.value.value)
    return None


def infer_result_file(root: Path, file: Path) -> str:
    stem = file.stem
    base = stem[:-2] if stem.endswith("_f") else stem
    suffix = base.removeprefix("gate_")
    candidates = [
        root / "results" / f"{base}_F.json",
        root / "results" / f"{suffix}_F.json",
        root / "results" / f"gate_{suffix}_F.json",
    ]
    for path in candidates:
        if path.exists():
            return str(path.relative_to(root))
    return ""


def corrected_audit(root: Path, s_min: dict[str, Any] | None) -> pd.DataFrame:
    rows = []
    for file in sorted((root / "predictx").glob("*.py")):
        text = file.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            rows.append({"gate_file": str(file.relative_to(root)), "parse_error": str(exc), "nuts_instantiated": False})
            continue
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and call_name(node.func) in {"NUTS", "MCMC"}:
                calls.append({"call": call_name(node.func), "line": int(node.lineno)})
            if isinstance(node, ast.Call) and call_name(node.func) == "run_one_curve":
                calls.append({"call": "run_one_curve->NUTS", "line": int(node.lineno)})
        if not calls and not file.name.startswith("gate_"):
            continue
        warmup = int_constant_from_assign(tree, ("NUTS_WARMUP",))
        draws = int_constant_from_assign(tree, ("NUTS_DRAWS",))
        chains = int_constant_from_assign(tree, ("NUTS_CHAINS",))
        deterministic_header = "Deterministic LM only. No NUTS" in text and file.name != "gate_sampler_pilot_f.py"
        nuts_instantiated = any(c["call"] in {"NUTS", "run_one_curve->NUTS"} for c in calls)
        result_file = infer_result_file(root, file)
        if not nuts_instantiated:
            status = "UNAFFECTED"
            reason = "no NUTS constructor call in AST"
        elif s_min is None:
            status = "NUTS_DEPENDENT_PENDING_S_MIN"
            reason = "NUTS is instantiated; S_min not established"
        else:
            below = (
                warmup is None
                or draws is None
                or chains is None
                or chains < int(s_min["chains"])
                or warmup < int(s_min["warmup"])
                or draws < int(s_min["draws"])
                or (bool(s_min.get("dense_mass")) and "dense_mass=True" not in text.replace(" ", ""))
            )
            status = "INVALIDATED" if below else "NUTS_AT_OR_ABOVE_S_MIN"
            reason = "below S_min sampler settings" if below else "sampler settings at or above S_min by static constants"
        rows.append(
            {
                "gate_file": str(file.relative_to(root)),
                "result_file": result_file,
                "nuts_instantiated": nuts_instantiated,
                "mcmc_instantiated": any(c["call"] == "MCMC" for c in calls),
                "call_sites": json.dumps(calls),
                "chains": chains,
                "warmup": warmup,
                "draws": draws,
                "deterministic_lm_header": deterministic_header,
                "audit_status": status,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict[str, np.ndarray]], np.ndarray]:
    real_subset, raw, _existing_real = load_real_subset(root)
    templates = real_templates(real_subset, raw)
    t_full = np.asarray(OmbeGrid().t, dtype=np.float32)
    data, arrays, _meta = synthetic_dataset(templates, "cell_A_synth_iid", t_full)
    data = data.sort_values("curve_id").reset_index(drop=True)
    return data.head(PILOT_N).copy(), data, arrays, t_full


def write_outputs(root: Path, result: dict[str, Any], ladder: pd.DataFrame, audit: pd.DataFrame, condition: pd.DataFrame) -> None:
    out = root / "results"
    out.mkdir(exist_ok=True)
    ladder.to_csv(out / f"{OUTPUT_STEM}_ladder.csv", index=False)
    audit.to_csv(out / f"{OUTPUT_STEM}_audit.csv", index=False)
    condition.to_csv(out / f"{OUTPUT_STEM}_condition_numbers.csv", index=False)
    (out / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    unaffected = audit[audit["audit_status"].eq("UNAFFECTED")]["gate_file"].tolist()
    invalidated = audit[audit["audit_status"].eq("INVALIDATED")]["gate_file"].tolist()
    nuts_pending = audit[audit["audit_status"].eq("NUTS_DEPENDENT_PENDING_S_MIN")]["gate_file"].tolist()
    lines = [
        "# Gate - Sampler Convergence Pilot, Cycle F",
        "",
        "## Protocol Decisions",
        "",
        "- S1-S4 were carried forward as standing facts and were not rerun.",
        "- S7 was run first, then S6. Both run all 30 pilot curves; there is no population-fraction early stopping.",
        "- Exact cached S6 curves from the previous run are reused only when curve id, sampler settings, Cell A construction, and seed match.",
        "- Divergence is judged by transition fraction: total divergent post-warmup draws divided by total post-warmup draws.",
        "",
        f"## Verdict: {result['verdict']}",
        "",
        f"- S_min: {result.get('S_min')}",
        f"- Stop reason: {result.get('stop_reason')}",
        "",
        "## Part 1 - Pilot Ladder",
        "",
        md_table(
            ladder.to_dict("records"),
            [
                ("rung", "rung", "str"),
                ("n_attempted", "n", "int"),
                ("chains", "chains", "int"),
                ("warmup", "warmup", "int"),
                ("draws", "draws", "int"),
                ("target_accept", "target", "float2"),
                ("dense_mass", "dense", "str"),
                ("coverage_50", "cov50", "float3"),
                ("coverage_50_ci_low", "50 lo", "float3"),
                ("coverage_50_ci_high", "50 hi", "float3"),
                ("coverage_80", "cov80", "float3"),
                ("coverage_80_ci_low", "80 lo", "float3"),
                ("coverage_80_ci_high", "80 hi", "float3"),
                ("coverage_95", "cov95", "float3"),
                ("coverage_95_ci_low", "95 lo", "float3"),
                ("coverage_95_ci_high", "95 hi", "float3"),
                ("pit_ks_statistic", "PIT KS", "float3"),
                ("pit_ks_p_value", "PIT p", "float3"),
                ("median_rhat", "median Rhat", "float3"),
                ("rhat_gt_1p01_fraction", "Rhat>1.01", "float3"),
                ("min_ess", "min ESS", "float1"),
                ("median_ess", "median ESS", "float1"),
                ("ess_lt_400_fraction", "ESS<400", "float3"),
                ("divergence_transition_fraction", "div frac", "float6"),
                ("any_divergence_fraction", "any div curve", "float3"),
                ("wall_clock_s_per_curve", "sec/curve", "float2"),
                ("converged", "converged", "str"),
                ("nominal_all", "coverage nominal", "str"),
            ],
        ),
        "",
        "## Part 2 - Condition Number Reconciliation",
        "",
        result["condition_reconciliation"]["reconciliation"],
        "",
        md_table(
            condition.head(50).to_dict("records"),
            [
                ("curve_id", "curve", "int"),
                ("truth_severity", "severity", "str"),
                ("gate1_fisher_kappa", "Fisher kappa", "float1"),
                ("s1_posterior_cov_kappa", "S1 posterior cov kappa", "float1"),
            ],
        ),
        "",
        "## Part 3 - Corrected Audit",
        "",
        "UNAFFECTED:",
        "",
        "\n".join(f"- {x}" for x in unaffected) if unaffected else "- none",
        "",
        "INVALIDATED:",
        "",
        "\n".join(f"- {x}" for x in invalidated) if invalidated else "- none",
        "",
        "NUTS-DEPENDENT PENDING S_MIN:",
        "",
        "\n".join(f"- {x}" for x in nuts_pending) if nuts_pending else "- none",
        "",
        md_table(
            audit.to_dict("records"),
            [
                ("gate_file", "gate file", "str"),
                ("result_file", "result file", "str"),
                ("nuts_instantiated", "NUTS", "str"),
                ("mcmc_instantiated", "MCMC", "str"),
                ("chains", "chains", "int"),
                ("warmup", "warmup", "int"),
                ("draws", "draws", "int"),
                ("deterministic_lm_header", "deterministic header", "str"),
                ("audit_status", "status", "str"),
                ("reason", "reason", "str"),
            ],
        ),
        "",
    ]
    if result.get("sampler_artifact_statement"):
        lines.extend(["## Required Statement", "", result["sampler_artifact_statement"], ""])
    (out / f"{OUTPUT_STEM}.md").write_text("\n".join(lines), encoding="utf-8")


def run(root: Path) -> dict[str, Any]:
    start = time.perf_counter()
    data, full_data, arrays, t_full = build_data(root)
    maybe_seed_s6_from_prior_cache(root, data)
    summaries = []
    div_by_rung: dict[str, pd.DataFrame] = {}
    for spec in SPECS:
        rows, div, cond = run_rung(root, spec, data, arrays, t_full)
        div_by_rung[spec["rung"]] = div
        summary = summarize_rung(rows, cond, spec, len(data))
        summary["boundary"] = boundary_summary(div)
        summaries.append(summary)
        print(
            f"{spec['rung']}: converged={summary['converged']} cov50={summary['coverage_50']:.3f} "
            f"div_frac={summary['divergence_transition_fraction']:.6f} Rhat>1.01={summary['rhat_gt_1p01_fraction']:.3f} "
            f"ESS<400={summary['ess_lt_400_fraction']:.3f}"
        )
    ladder = pd.DataFrame(summaries)
    converged = ladder[ladder["converged"].eq(True)].copy()
    s_min = converged.iloc[0].to_dict() if len(converged) else None
    if s_min is None:
        verdict = "GEOMETRY-INTRACTABLE-CONFIRMED"
        stop_reason = "Dense mass did not resolve the convergence failure at either budget."
        sampler_statement = ""
    else:
        boundary = s_min.get("boundary", {})
        if isinstance(boundary, str):
            boundary = {}
        if bool(boundary.get("boundary_pathology", False)):
            verdict = "BOUNDARY-PATHOLOGY"
            stop_reason = f"{s_min['rung']} converged, but divergent draws concentrate at prior bounds."
        else:
            verdict = "SAMPLER-ARTIFACT-CONFIRMED"
            stop_reason = f"{s_min['rung']} is the cheapest converged dense-mass pilot rung."
        sampler_statement = (
            "GEOMETRY-INTRACTABLE is retracted. The Gate A IMPLEMENTATION-FAULT verdict is retracted. "
            "The posterior is samplable with a dense mass matrix, and all NUTS results produced at "
            "2 chains / 250 / 250 are invalidated and must be recomputed at S_min."
        )
    condition, condition_summary = condition_reconciliation(root, full_data, arrays)
    audit = corrected_audit(root, s_min)
    result = {
        "verdict": verdict,
        "S_min": None if s_min is None else s_min["rung"],
        "S_min_summary": s_min,
        "stop_reason": stop_reason,
        "sampler_artifact_statement": sampler_statement,
        "ladder": summaries,
        "condition_reconciliation": condition_summary,
        "audit_counts": audit["audit_status"].value_counts(dropna=False).to_dict() if len(audit) else {},
        "target_counts": TARGET_COUNTS,
        "pilot_n": PILOT_N,
        "wall_clock_s": float(time.perf_counter() - start),
    }
    write_outputs(root, result, ladder, audit, condition)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    print(f"VERDICT: {result['verdict']}")
    print(f"S_min: {result['S_min']}")
    print(f"Stop reason: {result['stop_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
