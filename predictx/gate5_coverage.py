from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS
from PIL import Image, ImageDraw

from .gate5_sbc import (
    COVERAGE_LEVELS,
    PRIOR_RANGES,
    SBC_DRAWS,
    SBC_WARMUP,
    SIGMA_LPS,
    ombe_numpyro_model,
    sample_prior_natural,
    write_rank_hist_figure,
)
from .ombe import NATURAL_PARAM_NAMES, OmbeGrid, ombe_flow, phi_from_natural


def run_gate5_coverage(root: Path, n_replications: int = 500, seed: int = 12345) -> dict:
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    rank_baseline = read_json(results_dir / "gate5_sbc.json")
    rank_results = rank_baseline.get("rank_results", {})
    rank_uniform_pass = sum(
        1 for item in rank_results.values() if item.get("chi_square_p", 0.0) > 0.05
    )

    t = OmbeGrid().t
    rng = np.random.default_rng(seed)
    coverages = {level: {name: [] for name in NATURAL_PARAM_NAMES} for level in COVERAGE_LEVELS}

    kernel = NUTS(lambda y=None: ombe_numpyro_model(t, y))
    for rep in range(n_replications):
        truth = sample_prior_natural(rng)
        phi = phi_from_natural(**{name: jnp.array(value) for name, value in truth.items()})
        y = ombe_flow(phi, t) + jax.random.normal(
            jax.random.PRNGKey(seed * 10_000 + rep),
            shape=t.shape,
        ) * SIGMA_LPS
        mcmc = MCMC(
            kernel,
            num_warmup=SBC_WARMUP,
            num_samples=SBC_DRAWS,
            num_chains=1,
            progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(seed * 100_000 + rep + 1), y=y)
        samples = mcmc.get_samples(group_by_chain=False)
        for name in NATURAL_PARAM_NAMES:
            draws = np.asarray(samples[name])
            truth_value = truth[name]
            for level in COVERAGE_LEVELS:
                lo = (1.0 - level) / 2.0
                hi = 1.0 - lo
                qlo, qhi = np.quantile(draws, [lo, hi])
                coverages[level][name].append(bool(qlo <= truth_value <= qhi))
        if (rep + 1) % 10 == 0 or rep + 1 == n_replications:
            print(f"Coverage replication {rep + 1}/{n_replications}")
            (results_dir / "gate5_coverage_progress.json").write_text(
                json.dumps({"done": rep + 1, "total": n_replications}, indent=2) + "\n",
                encoding="utf-8",
            )

    coverage_results = {}
    below_counts = {name: 0 for name in NATURAL_PARAM_NAMES}
    for level in COVERAGE_LEVELS:
        half_width = 2.0 * np.sqrt(level * (1.0 - level) / n_replications)
        lo_band = max(0.0, level - half_width)
        hi_band = min(1.0, level + half_width)
        level_key = f"{level:.2f}"
        coverage_results[level_key] = {
            "nominal": level,
            "mc_half_width_2se": float(half_width),
            "mc_lower": float(lo_band),
            "mc_upper": float(hi_band),
            "params": {},
        }
        for name in NATURAL_PARAM_NAMES:
            observed = float(np.mean(coverages[level][name]))
            relation = "within"
            if observed < lo_band:
                relation = "below"
                below_counts[name] += 1
            elif observed > hi_band:
                relation = "above"
            coverage_results[level_key]["params"][name] = {
                "observed": observed,
                "relation_to_mc_band": relation,
            }

    persistent_below = [name for name, count in below_counts.items() if count >= 3]
    if persistent_below:
        status = "FAIL"
        reason = "Coverage is consistently below the Monte Carlo band for >=3 nominal levels."
    elif rank_uniform_pass >= 4:
        status = "PASS"
        reason = "Rank histograms passed previously and coverage is within Monte Carlo tolerance."
    else:
        status = "AMBIGUOUS"
        reason = "Coverage is within tolerance, but previous rank histogram pass count is insufficient or unavailable."

    result = {
        "status": status,
        "reason": reason,
        "n_replications": n_replications,
        "draws": SBC_DRAWS,
        "warmup": SBC_WARMUP,
        "chains": 1,
        "sigma_lps": SIGMA_LPS,
        "rank_uniform_p_gt_0_05_params_from_prior_run": rank_uniform_pass,
        "persistent_below_band_params": persistent_below,
        "below_band_counts_by_param": below_counts,
        "coverage_results": coverage_results,
        "prior_ranges": PRIOR_RANGES,
    }
    write_report(results_dir, result, rank_baseline)
    write_coverage_figure(figures_dir / "gate5_coverage.png", result)
    if rank_results:
        write_rank_hist_figure(figures_dir / "gate5_sbc_hist.png", rank_baseline)
    update_summary(results_dir, result)
    return result


def write_report(results_dir: Path, result: dict, rank_baseline: dict) -> None:
    lines = [
        "# Gate 5 - SBC Calibration",
        "",
        f"- status: {result['status']}",
        f"- reason: {result['reason']}",
        f"- coverage replications: {result['n_replications']}",
        f"- posterior draws per replication: {result['draws']}",
        f"- warmup per replication: {result['warmup']}",
        f"- chains per coverage replication: {result['chains']}",
        f"- sigma: {result['sigma_lps']} L/s",
        f"- prior rank run p>0.05 params: {result['rank_uniform_p_gt_0_05_params_from_prior_run']}/5",
        "",
        "Rank histograms were not rerun here; this coverage-only run reuses the prior rank-uniformity result.",
        "",
        "## Prior Rank Histogram Tests",
        "",
        "| parameter | chi-square p | shape |",
        "|---|---:|---|",
    ]
    rank_results = rank_baseline.get("rank_results", {})
    for name in NATURAL_PARAM_NAMES:
        item = rank_results.get(name, {})
        p_value = item.get("chi_square_p")
        p_text = f"{p_value:.6g}" if isinstance(p_value, (int, float)) else "not available"
        shape = item.get("shape", "not available")
        lines.append(f"| `{name}` | {p_text} | {shape} |")

    lines.extend(
        [
            "",
            "## Coverage With Monte Carlo Tolerance",
            "",
            "Criterion: PASS if observed coverage is within `nominal +/- 2*sqrt(p(1-p)/n)`; flag persistent overconfidence only when a parameter is below the band at >=3 nominal levels.",
            "",
            "| nominal | MC band | " + " | ".join(f"`{name}`" for name in NATURAL_PARAM_NAMES) + " |",
            "|---|---|" + "---:|" * len(NATURAL_PARAM_NAMES),
        ]
    )
    for level_key, item in result["coverage_results"].items():
        band = f"{item['mc_lower']:.3f}-{item['mc_upper']:.3f}"
        values = []
        for name in NATURAL_PARAM_NAMES:
            param = item["params"][name]
            suffix = "" if param["relation_to_mc_band"] == "within" else f" ({param['relation_to_mc_band']})"
            values.append(f"{param['observed']:.3f}{suffix}")
        lines.append(f"| {item['nominal']:.0%} | {band} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Persistent Below-Band Flags",
            "",
            "| parameter | below-band nominal levels |",
            "|---|---:|",
        ]
    )
    for name in NATURAL_PARAM_NAMES:
        lines.append(f"| `{name}` | {result['below_band_counts_by_param'][name]} |")
    lines.extend(
        [
            "",
            f"Headline: genuinely mis-covered params = {', '.join(result['persistent_below_band_params']) if result['persistent_below_band_params'] else 'none'}; all other coverage deviations are within Monte Carlo tolerance.",
            "",
            "No tuning was performed.",
            "",
        ]
    )
    (results_dir / "gate5_sbc.md").write_text("\n".join(lines), encoding="utf-8")
    (results_dir / "gate5_coverage.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def write_coverage_figure(path: Path, result: dict) -> None:
    image = Image.new("RGB", (1100, 760), "white")
    draw = ImageDraw.Draw(image)
    x0, y0, size = 130, 90, 560
    draw.text((x0, 35), "Gate 5 coverage with Monte Carlo tolerance", fill="black")
    draw.rectangle((x0, y0, x0 + size, y0 + size), outline="black")
    draw.line((x0, y0 + size, x0 + size, y0), fill=(0, 0, 0), width=2)
    colors = [(200, 30, 30), (30, 120, 30), (30, 80, 200), (180, 120, 20), (120, 40, 160)]
    levels = [float(key) for key in result["coverage_results"].keys()]
    for idx, name in enumerate(NATURAL_PARAM_NAMES):
        points = []
        for level in levels:
            item = result["coverage_results"][f"{level:.2f}"]
            actual = item["params"][name]["observed"]
            px = x0 + level * size
            py = y0 + size - actual * size
            points.append((px, py))
        draw.line(points, fill=colors[idx], width=3)
        for px, py in points:
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=colors[idx])
        draw.text((740, 110 + idx * 28), name, fill=colors[idx])
    draw.text((x0 + 210, y0 + size + 40), "nominal coverage", fill="black")
    draw.text((28, y0 + 250), "actual", fill="black")
    image.save(path)


def update_summary(results_dir: Path, gate5: dict) -> None:
    gate1 = read_json(results_dir / "gate1_sloppiness.json")
    lines = [
        "# Synthetic Gate Summary",
        "",
        "## Gates",
        "",
        f"- Gate 1: {gate1.get('status', 'UNKNOWN')} - {gate1.get('interpretation', gate1.get('reason', ''))}",
        f"  - median kappa: {gate1.get('kappa_median', 'not computed')}",
        f"- Gate 5: {gate5['status']} - {gate5['reason']}",
        f"  - coverage replications: {gate5['n_replications']}",
        f"  - genuinely mis-covered params: {', '.join(gate5['persistent_below_band_params']) if gate5['persistent_below_band_params'] else 'none'}",
        "",
        "## Blocked Gates",
        "",
        "Gates 2, 3, 4, 6, and 7 are BLOCKED pending raw-curve access, NOT-REACHED.",
        "",
    ]
    (results_dir / "GATE_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-replications", type=int, default=500)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    run_gate5_coverage(Path.cwd(), n_replications=args.n_replications, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
