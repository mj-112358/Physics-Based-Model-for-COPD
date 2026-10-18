from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .ombe import (
    NATURAL_PARAM_NAMES,
    PARAM_NAMES,
    OmbeGrid,
    batched_fisher_information,
    flow_integral_l,
    midrange_phi,
    ombe_flow,
    phi_from_natural,
    sample_prior_phi,
    unpack_phi,
)

RAW_RDS = ("SPXRAW_E.rds", "SPXRAW_F.rds", "SPXRAW_G.rds")
RAW_XPT = ("SPXRAW_E.xpt", "SPXRAW_F.xpt", "SPXRAW_G.xpt")
SIGMA_LPS = 0.10
NUTS_WARMUP = 2500
NUTS_DRAWS = 2500
NUTS_CHAINS = 4


def run_synthetic_gates(root: Path | None = None) -> dict[str, str]:
    root = root or Path.cwd()
    results_dir = root / "results"
    figures_dir = root / "figures"
    data_dir = root / "data"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    precondition = precondition_check(data_dir, results_dir)
    gate1 = run_gate1(results_dir, figures_dir)
    gate5 = run_gate5_dependency_check(results_dir, figures_dir, gate1)
    write_summary(results_dir, precondition, gate1, gate5)
    return {"gate1": gate1["status"], "gate5": gate5["status"]}


def precondition_check(data_dir: Path, results_dir: Path) -> dict:
    raw_present = []
    for name in (*RAW_RDS, *RAW_XPT):
        path = data_dir / name
        if path.exists():
            raw_present.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "treated_as": _raw_status(path),
                }
            )
    merged = results_dir / "merged_summary.parquet"
    info = {
        "raw_present": raw_present,
        "merged_summary_exists": merged.exists(),
        "merged_summary_bytes": merged.stat().st_size if merged.exists() else 0,
    }
    if not merged.exists():
        raise RuntimeError("results/merged_summary.parquet is absent; summary truth is unavailable.")
    print("Precondition raw files present:")
    for item in raw_present:
        print(f"  {item['path']} {item['bytes']} bytes ({item['treated_as']})")
    print(f"results/merged_summary.parquet present: {info['merged_summary_bytes']} bytes")
    return info


def run_gate1(results_dir: Path, figures_dir: Path) -> dict:
    grid = OmbeGrid()
    t = grid.t
    phi0 = midrange_phi()
    p0 = unpack_phi(phi0)
    integral = float(flow_integral_l(phi0, t))
    fvc = float(p0["fvc_l"])
    rel_err = abs(integral - fvc) / fvc
    if rel_err > 0.01:
        status = "FAIL"
        result = {
            "status": status,
            "reason": "volume conservation assertion failed",
            "midrange_integral_l": integral,
            "midrange_fvc_l": fvc,
            "relative_error": rel_err,
        }
        write_gate1_report(results_dir, result, None)
        return result

    phi = sample_prior_phi(1000, seed=0)
    fisher = np.asarray(batched_fisher_information(phi, t, SIGMA_LPS))
    eigvals, eigvecs = np.linalg.eigh(fisher)
    eigvals = np.maximum(eigvals, np.finfo(float).tiny)
    kappa = eigvals[:, -1] / eigvals[:, 0]

    q25, q50, q75 = np.percentile(kappa, [25, 50, 75])
    if q50 >= 1000:
        status = "PASS"
        interpretation = "sloppiness confirmed, Bayesian inference justified"
    elif q50 < 100:
        status = "FAIL"
        interpretation = "premise unsupported"
    else:
        status = "AMBIGUOUS"
        interpretation = "ambiguous sloppiness band, human judgment required"

    stiff_load = np.median(np.abs(eigvecs[:, :, -1]), axis=0)
    sloppy_load = np.median(np.abs(eigvecs[:, :, 0]), axis=0)
    hist_counts, hist_edges = np.histogram(np.log10(kappa), bins=20)
    result = {
        "status": status,
        "interpretation": interpretation,
        "sigma_lps": SIGMA_LPS,
        "n_prior_samples": 1000,
        "midrange_integral_l": integral,
        "midrange_fvc_l": fvc,
        "volume_relative_error": rel_err,
        "kappa_median": float(q50),
        "kappa_q25": float(q25),
        "kappa_q75": float(q75),
        "kappa_min": float(np.min(kappa)),
        "kappa_max": float(np.max(kappa)),
        "hist_log10_edges": hist_edges.tolist(),
        "hist_counts": hist_counts.tolist(),
        "stiff_median_abs_loadings": dict(zip(PARAM_NAMES, map(float, stiff_load))),
        "sloppy_median_abs_loadings": dict(zip(PARAM_NAMES, map(float, sloppy_load))),
    }
    write_gate1_report(results_dir, result, kappa)
    plot_eigenspectrum(figures_dir / "gate1_eigenspectrum.png", eigvals)
    return result


def run_gate5_dependency_check(results_dir: Path, figures_dir: Path, gate1: dict) -> dict:
    if gate1["status"] == "FAIL":
        result = {
            "status": "NOT-RUN",
            "reason": "Gate 1 failed; SBC is meaningless when the forward model gate fails.",
        }
        write_gate5_report(results_dir, result)
        return result

    try:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.diagnostics import summary
        from numpyro.infer import MCMC, NUTS
    except ModuleNotFoundError:
        result = {
            "status": "FAIL",
            "reason": "NumPyro is not installed, so the required NUTS inference engine is unavailable.",
            "requested_engine": "NumPyro NUTS, 4 chains, 2500 warmup + 2500 draws, 1000 SBC repeats",
        }
        write_gate5_report(results_dir, result)
        make_placeholder_gate5_figures(figures_dir)
        return result

    numpyro.set_host_device_count(NUTS_CHAINS)
    grid = OmbeGrid()
    t = grid.t
    true = {
        "tau_f": 0.35,
        "tau_s": 2.50,
        "alpha": 0.60,
        "fvc_l": 4.00,
        "tau_o": 0.12,
    }

    def model(y=None):
        tau_f = numpyro.sample("tau_f", dist.Uniform(0.15, 0.80))
        tau_s = numpyro.sample("tau_s", dist.Uniform(1.00, 5.00))
        alpha = numpyro.sample("alpha", dist.Uniform(0.25, 0.85))
        fvc_l = numpyro.sample("fvc_l", dist.Uniform(2.00, 6.00))
        tau_o = numpyro.sample("tau_o", dist.Uniform(0.03, 0.40))
        phi = phi_from_natural(tau_f, tau_s, alpha, fvc_l, tau_o)
        mu = ombe_flow(phi, t)
        numpyro.sample("y", dist.Normal(mu, SIGMA_LPS), obs=y)

    true_phi = phi_from_natural(**{k: jnp.array(v) for k, v in true.items()})
    noiseless = ombe_flow(true_phi, t)
    key = jax.random.PRNGKey(20260707)
    y = noiseless + jax.random.normal(key, shape=noiseless.shape) * SIGMA_LPS
    mcmc = MCMC(
        NUTS(model),
        num_warmup=NUTS_WARMUP,
        num_samples=NUTS_DRAWS,
        num_chains=NUTS_CHAINS,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(1), y=y)
    chain_samples = mcmc.get_samples(group_by_chain=True)
    stats = summary(chain_samples, prob=0.90)
    max_rhat = max(float(stats[name]["r_hat"]) for name in NATURAL_PARAM_NAMES)
    min_ess = min(float(stats[name]["n_eff"]) for name in NATURAL_PARAM_NAMES)
    flat_samples = mcmc.get_samples(group_by_chain=False)
    posterior_mean = {name: float(np.asarray(flat_samples[name]).mean()) for name in NATURAL_PARAM_NAMES}
    mean_phi = phi_from_natural(**{k: jnp.array(v) for k, v in posterior_mean.items()})
    posterior_mean_flow = ombe_flow(mean_phi, t)
    recovery_rmse = float(jnp.sqrt(jnp.mean((posterior_mean_flow - noiseless) ** 2)))
    health_pass = max_rhat < 1.01 and min_ess > 400 and recovery_rmse <= SIGMA_LPS
    if not health_pass:
        result = {
            "status": "FAIL",
            "reason": "NUTS health assertion failed on one synthetic curve; SBC not run.",
            "max_rhat": max_rhat,
            "min_ess": min_ess,
            "posterior_mean_flow_rmse_lps": recovery_rmse,
            "sigma_lps": SIGMA_LPS,
            "posterior_mean": posterior_mean,
            "truth": true,
            "nuts_warmup": NUTS_WARMUP,
            "nuts_draws": NUTS_DRAWS,
            "nuts_chains": NUTS_CHAINS,
        }
        write_gate5_report(results_dir, result)
        make_placeholder_gate5_figures(figures_dir, "Gate 5 stopped: NUTS health failed")
        return result

    result = {
        "status": "NOT-RUN",
        "reason": "NUTS health passed, but the required full SBC calibration was not executed; CI calibration remains pending.",
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "posterior_mean_flow_rmse_lps": recovery_rmse,
        "sigma_lps": SIGMA_LPS,
        "posterior_mean": posterior_mean,
        "truth": true,
        "nuts_warmup": NUTS_WARMUP,
        "nuts_draws": NUTS_DRAWS,
        "nuts_chains": NUTS_CHAINS,
    }
    write_gate5_report(results_dir, result)
    make_placeholder_gate5_figures(figures_dir, "Gate 5 not run: full SBC pending")
    return result


def write_gate1_report(results_dir: Path, result: dict, kappa: np.ndarray | None) -> None:
    lines = [
        "# Gate 1 - Sloppiness",
        "",
        "Synthetic-only OMBE forward model:",
        "",
        "`q(t) = [alpha exp(-t/tau_f)/tau_f + (1-alpha) exp(-t/tau_s)/tau_s] * [1 - exp(-t/tau_o)]`",
        "",
        "`F(t) = FVC * q(t) / integral(q)` on a 12 s, 100 Hz grid.",
        "",
        f"- status: {result['status']}",
        f"- interpretation: {result.get('interpretation', result.get('reason', ''))}",
        f"- flow noise sigma: {result.get('sigma_lps', SIGMA_LPS)} L/s",
        f"- midrange integral: {result['midrange_integral_l']:.6f} L",
        f"- midrange FVC: {result['midrange_fvc_l']:.6f} L",
        f"- volume relative error: {result.get('volume_relative_error', result.get('relative_error', float('nan'))):.6%}",
        "",
    ]
    if kappa is not None:
        lines.extend(
            [
                "## Kappa",
                "",
                f"- n prior samples: {result['n_prior_samples']}",
                f"- median: {result['kappa_median']:.6g}",
                f"- IQR: {result['kappa_q25']:.6g} - {result['kappa_q75']:.6g}",
                f"- min: {result['kappa_min']:.6g}",
                f"- max: {result['kappa_max']:.6g}",
                "",
                "## Full Histogram",
                "",
                "| log10(kappa) bin | count |",
                "|---|---:|",
            ]
        )
        for lo, hi, count in zip(result["hist_log10_edges"][:-1], result["hist_log10_edges"][1:], result["hist_counts"]):
            lines.append(f"| {lo:.2f} to {hi:.2f} | {count} |")
        lines.extend(["", "## Median Absolute Eigenvector Loadings", ""])
        lines.extend(["### Stiffest Direction", "", "| parameter | loading |", "|---|---:|"])
        for name, value in result["stiff_median_abs_loadings"].items():
            lines.append(f"| `{name}` | {value:.4f} |")
        lines.extend(["", "### Sloppiest Direction", "", "| parameter | loading |", "|---|---:|"])
        for name, value in result["sloppy_median_abs_loadings"].items():
            lines.append(f"| `{name}` | {value:.4f} |")
        lines.append("")
    (results_dir / "gate1_sloppiness.md").write_text("\n".join(lines), encoding="utf-8")
    (results_dir / "gate1_sloppiness.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def plot_eigenspectrum(path: Path, eigvals: np.ndarray) -> None:
    qs = np.percentile(eigvals, [25, 50, 75], axis=0)
    x = np.arange(1, eigvals.shape[1] + 1)
    y_med = np.log10(qs[1])
    y_lo = np.log10(qs[0])
    y_hi = np.log10(qs[2])
    image = Image.new("RGB", (1000, 650), "white")
    draw = ImageDraw.Draw(image)
    margin_l, margin_r, margin_t, margin_b = 95, 50, 70, 90
    plot_w = image.width - margin_l - margin_r
    plot_h = image.height - margin_t - margin_b
    y_min = float(np.floor(np.min(y_lo)))
    y_max = float(np.ceil(np.max(y_hi)))

    def px(i: int) -> float:
        return margin_l + (x[i] - x[0]) / (x[-1] - x[0]) * plot_w

    def py(value: float) -> float:
        return margin_t + (y_max - value) / (y_max - y_min) * plot_h

    draw.text((margin_l, 25), "Gate 1 eigenspectrum across 1000 prior samples", fill="black")
    draw.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill="black", width=2)
    draw.line((margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h), fill="black", width=2)
    for tick in range(int(y_min), int(y_max) + 1):
        ytick = py(tick)
        draw.line((margin_l - 5, ytick, margin_l, ytick), fill="black")
        draw.text((10, ytick - 8), f"1e{tick}", fill="black")
    for i, label in enumerate(x):
        xtick = px(i)
        draw.line((xtick, margin_t + plot_h, xtick, margin_t + plot_h + 5), fill="black")
        draw.text((xtick - 4, margin_t + plot_h + 10), str(label), fill="black")
    for i in range(len(x)):
        draw.line((px(i), py(y_lo[i]), px(i), py(y_hi[i])), fill=(180, 180, 220), width=8)
    points = [(px(i), py(y_med[i])) for i in range(len(x))]
    draw.line(points, fill=(20, 70, 180), width=4)
    for point in points:
        draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=(20, 70, 180))
    draw.text((margin_l + 280, margin_t + plot_h + 45), "Fisher eigenvalue index", fill="black")
    draw.text((15, margin_t + 10), "log-scale eigenvalue", fill="black")
    image.save(path)


def write_gate5_report(results_dir: Path, result: dict) -> None:
    lines = [
        "# Gate 5 - SBC Calibration",
        "",
        f"- status: {result['status']}",
        f"- reason: {result['reason']}",
    ]
    if "requested_engine" in result:
        lines.append(f"- requested engine: {result['requested_engine']}")
    if "max_rhat" in result:
        lines.extend(
            [
                f"- NUTS chains: {result['nuts_chains']}",
                f"- NUTS warmup: {result['nuts_warmup']}",
                f"- NUTS draws: {result['nuts_draws']}",
                f"- max R-hat: {result['max_rhat']:.6f}",
                f"- min ESS: {result['min_ess']:.6f}",
                f"- posterior mean flow RMSE: {result['posterior_mean_flow_rmse_lps']:.6f} L/s",
                f"- sigma: {result['sigma_lps']:.6f} L/s",
                "",
                "## Truth vs Posterior Mean",
                "",
                "| parameter | truth | posterior_mean |",
                "|---|---:|---:|",
            ]
        )
        for name in NATURAL_PARAM_NAMES:
            lines.append(f"| `{name}` | {result['truth'][name]:.6f} | {result['posterior_mean'][name]:.6f} |")
    lines.extend(
        [
            "",
            "No SBC ranks, chi-square p-values, or coverage claims were emitted.",
            "",
        ]
    )
    (results_dir / "gate5_sbc.md").write_text("\n".join(lines), encoding="utf-8")
    (results_dir / "gate5_sbc.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def make_placeholder_gate5_figures(figures_dir: Path, title: str = "Gate 5 not run: NumPyro unavailable") -> None:
    for name, title in [
        ("gate5_sbc_hist.png", title),
        ("gate5_coverage.png", title),
    ]:
        image = Image.new("RGB", (900, 420), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 880, 400), outline="black", width=2)
        draw.text((230, 195), title, fill="black")
        image.save(figures_dir / name)


def write_summary(results_dir: Path, precondition: dict, gate1: dict, gate5: dict) -> None:
    lines = [
        "# Synthetic Gate Summary",
        "",
        "## Preconditions",
        "",
        f"- `results/merged_summary.parquet` exists: {precondition['merged_summary_exists']} ({precondition['merged_summary_bytes']} bytes)",
        "- raw curve files present:",
    ]
    if precondition["raw_present"]:
        for item in precondition["raw_present"]:
            lines.append(f"  - `{item['path']}`: {item['bytes']} bytes, treated as {item['treated_as']}")
    else:
        lines.append("  - none")
    lines.extend(
        [
            "",
            "## Gates",
            "",
            f"- Gate 1: {gate1['status']} - {gate1.get('interpretation', gate1.get('reason', ''))}",
            f"  - median kappa: {gate1.get('kappa_median', 'not computed')}",
            f"- Gate 5: {gate5['status']} - {gate5['reason']}",
            "",
            "## Blocked Gates",
            "",
            "Gates 2, 3, 4, 6, and 7 are BLOCKED pending raw-curve access, NOT-REACHED.",
            "",
        ]
    )
    (results_dir / "GATE_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def _raw_status(path: Path) -> str:
    if path.suffix == ".rds" and path.stat().st_size <= 128:
        return "empty/stub"
    if path.suffix == ".xpt" and path.stat().st_size < 100_000:
        return "corrupt/error-page stub"
    return "present"


def main() -> int:
    run_synthetic_gates(Path.cwd())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
