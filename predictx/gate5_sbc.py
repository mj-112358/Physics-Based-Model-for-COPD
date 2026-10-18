from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from PIL import Image, ImageDraw
from scipy.stats import chisquare

from .ombe import NATURAL_PARAM_NAMES, OmbeGrid, ombe_flow, phi_from_natural

SIGMA_LPS = 0.10
SBC_WARMUP = 1000
SBC_DRAWS = 1000
COVERAGE_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
PRIOR_RANGES = {
    "tau_f": (0.15, 0.80),
    "tau_s": (1.00, 5.00),
    "alpha": (0.25, 0.85),
    "fvc_l": (2.00, 6.00),
    "tau_o": (0.03, 0.40),
}


def ombe_numpyro_model(t: jnp.ndarray, y: jnp.ndarray | None = None) -> None:
    tau_f = numpyro.sample("tau_f", dist.Uniform(*PRIOR_RANGES["tau_f"]))
    tau_s = numpyro.sample("tau_s", dist.Uniform(*PRIOR_RANGES["tau_s"]))
    alpha = numpyro.sample("alpha", dist.Uniform(*PRIOR_RANGES["alpha"]))
    fvc_l = numpyro.sample("fvc_l", dist.Uniform(*PRIOR_RANGES["fvc_l"]))
    tau_o = numpyro.sample("tau_o", dist.Uniform(*PRIOR_RANGES["tau_o"]))
    phi = phi_from_natural(tau_f, tau_s, alpha, fvc_l, tau_o)
    mu = ombe_flow(phi, t)
    numpyro.sample("y", dist.Normal(mu, SIGMA_LPS), obs=y)


def run_gate5_sbc(root: Path, n_replications: int = 200, seed: int = 0) -> dict:
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    gate1 = _read_json(results_dir / "gate1_sloppiness.json")
    if gate1 and gate1.get("status") != "PASS":
        result = {
            "status": "NOT-RUN",
            "reason": "Gate 1 is not PASS; SBC not run.",
            "gate1_status": gate1.get("status"),
        }
        write_sbc_report(results_dir, result)
        return result

    t = OmbeGrid().t
    rng = np.random.default_rng(seed)
    ranks = {name: [] for name in NATURAL_PARAM_NAMES}
    coverages = {level: {name: [] for name in NATURAL_PARAM_NAMES} for level in COVERAGE_LEVELS}

    kernel = NUTS(lambda y=None: ombe_numpyro_model(t, y))
    for rep in range(n_replications):
        truth = sample_prior_natural(rng)
        phi = phi_from_natural(**{name: jnp.array(value) for name, value in truth.items()})
        key_y = jax.random.PRNGKey(seed * 10_000 + rep)
        y = ombe_flow(phi, t) + jax.random.normal(key_y, shape=t.shape) * SIGMA_LPS
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
            ranks[name].append(int(np.sum(draws < truth_value)))
            for level in COVERAGE_LEVELS:
                lo = (1.0 - level) / 2.0
                hi = 1.0 - lo
                qlo, qhi = np.quantile(draws, [lo, hi])
                coverages[level][name].append(bool(qlo <= truth_value <= qhi))

        if (rep + 1) % 10 == 0 or rep + 1 == n_replications:
            print(f"SBC replication {rep + 1}/{n_replications}")
            write_progress(results_dir, rep + 1, n_replications, ranks)

    rank_bins = np.linspace(0, SBC_DRAWS, 21)
    rank_results = {}
    uniform_pass = 0
    for name in NATURAL_PARAM_NAMES:
        counts, edges = np.histogram(ranks[name], bins=rank_bins)
        p_value = float(chisquare(counts).pvalue)
        shape = classify_rank_shape(counts)
        rank_results[name] = {
            "counts": counts.astype(int).tolist(),
            "edges": edges.tolist(),
            "chi_square_p": p_value,
            "shape": shape,
        }
        if p_value > 0.05:
            uniform_pass += 1

    coverage_results = {}
    coverage_pass = True
    for level in COVERAGE_LEVELS:
        by_param = {}
        for name in NATURAL_PARAM_NAMES:
            actual = float(np.mean(coverages[level][name]))
            by_param[name] = actual
            if actual + 1e-12 < level:
                coverage_pass = False
        coverage_results[str(level)] = by_param

    u_shaped = [
        name
        for name, item in rank_results.items()
        if item["shape"] == "U-shaped" and item["chi_square_p"] <= 0.05
    ]
    gross_shape_flags = [
        name
        for name, item in rank_results.items()
        if item["shape"] in {"U-shaped", "inverse-U"}
    ]
    inverse_u = [name for name, item in rank_results.items() if item["shape"] == "inverse-U"]
    if uniform_pass >= 4 and coverage_pass:
        status = "PASS"
        reason = "Rank histograms uniform for >=4/5 params and coverage is on-or-above diagonal."
    elif u_shaped:
        status = "FAIL"
        reason = "Chi-square-significant U-shaped rank histogram detected; posterior is overconfident for listed params."
    elif not coverage_pass:
        status = "FAIL"
        reason = "Rank histogram p-values passed, but expected coverage fell below nominal; interval calibration is overconfident."
    else:
        status = "AMBIGUOUS"
        reason = "SBC did not meet pass criterion, but no U-shaped fatal overconfidence pattern was detected."

    result = {
        "status": status,
        "reason": reason,
        "n_replications": n_replications,
        "warmup": SBC_WARMUP,
        "draws": SBC_DRAWS,
        "chains": 1,
        "sigma_lps": SIGMA_LPS,
        "uniform_p_gt_0_05_params": uniform_pass,
        "coverage_on_or_above_diagonal": coverage_pass,
        "u_shaped_params": u_shaped,
        "inverse_u_params": inverse_u,
        "gross_shape_flags": gross_shape_flags,
        "rank_results": rank_results,
        "coverage_results": coverage_results,
    }
    write_sbc_report(results_dir, result)
    write_sbc_figures(figures_dir, result)
    update_gate_summary(results_dir, result)
    return result


def sample_prior_natural(rng: np.random.Generator) -> dict[str, float]:
    return {
        name: float(rng.uniform(lo, hi))
        for name, (lo, hi) in PRIOR_RANGES.items()
    }


def classify_rank_shape(counts: np.ndarray) -> str:
    edge = float(np.mean([counts[0], counts[-1]]))
    center = float(np.mean(counts[len(counts) // 2 - 2 : len(counts) // 2 + 2]))
    if edge > center * 1.5:
        return "U-shaped"
    if center > edge * 1.5:
        return "inverse-U"
    return "no gross shape flag"


def write_sbc_report(results_dir: Path, result: dict) -> None:
    lines = [
        "# Gate 5 - SBC Calibration",
        "",
        f"- status: {result['status']}",
        f"- reason: {result['reason']}",
    ]
    if "n_replications" in result:
        lines.extend(
            [
                f"- SBC replications: {result['n_replications']}",
                f"- posterior draws per replication: {result['draws']}",
                f"- warmup per replication: {result['warmup']}",
                f"- chains per SBC replication: {result['chains']}",
                f"- sigma: {result['sigma_lps']} L/s",
                f"- rank uniformity p>0.05 params: {result['uniform_p_gt_0_05_params']}/5",
                f"- coverage on-or-above diagonal: {result['coverage_on_or_above_diagonal']}",
                "",
                "## Rank Histogram Tests",
                "",
                "| parameter | chi-square p | shape |",
                "|---|---:|---|",
            ]
        )
        for name in NATURAL_PARAM_NAMES:
            item = result["rank_results"][name]
            lines.append(f"| `{name}` | {item['chi_square_p']:.6g} | {item['shape']} |")
        lines.extend(["", "## Expected Coverage", "", "| level | " + " | ".join(f"`{p}`" for p in NATURAL_PARAM_NAMES) + " |", "|---|" + "---:|" * len(NATURAL_PARAM_NAMES)])
        for level in COVERAGE_LEVELS:
            row = result["coverage_results"][str(level)]
            lines.append("| " + f"{level:.0%}" + " | " + " | ".join(f"{row[p]:.3f}" for p in NATURAL_PARAM_NAMES) + " |")
        lines.extend(
            [
                "",
                f"Chi-square-significant U-shaped params: {', '.join(result['u_shaped_params']) if result['u_shaped_params'] else 'none'}",
                f"Gross shape flags by edge/center heuristic: {', '.join(result.get('gross_shape_flags', [])) if result.get('gross_shape_flags') else 'none'}",
                f"Inverse-U params: {', '.join(result['inverse_u_params']) if result['inverse_u_params'] else 'none'}",
                "",
            ]
        )
    lines.append("Actual SBC numbers are based on this run only; no tuning was performed.")
    lines.append("")
    (results_dir / "gate5_sbc.md").write_text("\n".join(lines), encoding="utf-8")
    (results_dir / "gate5_sbc.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def write_sbc_figures(figures_dir: Path, result: dict) -> None:
    write_rank_hist_figure(figures_dir / "gate5_sbc_hist.png", result)
    write_coverage_figure(figures_dir / "gate5_coverage.png", result)


def write_rank_hist_figure(path: Path, result: dict) -> None:
    image = Image.new("RGB", (1300, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 25), "Gate 5 SBC rank histograms", fill="black")
    panel_w, panel_h = 380, 250
    for idx, name in enumerate(NATURAL_PARAM_NAMES):
        x0 = 40 + (idx % 2) * 620
        y0 = 80 + (idx // 2) * 270
        counts = np.array(result["rank_results"][name]["counts"], dtype=float)
        max_count = max(float(counts.max()), 1.0)
        draw.text((x0, y0 - 20), f"{name} p={result['rank_results'][name]['chi_square_p']:.3g}", fill="black")
        draw.rectangle((x0, y0, x0 + panel_w, y0 + panel_h), outline="black")
        bar_w = panel_w / len(counts)
        for i, count in enumerate(counts):
            h = (count / max_count) * (panel_h - 20)
            draw.rectangle(
                (x0 + i * bar_w, y0 + panel_h - h, x0 + (i + 1) * bar_w - 2, y0 + panel_h),
                fill=(70, 120, 200),
            )
    image.save(path)


def write_coverage_figure(path: Path, result: dict) -> None:
    image = Image.new("RGB", (1000, 720), "white")
    draw = ImageDraw.Draw(image)
    x0, y0, size = 120, 80, 560
    draw.text((x0, 30), "Gate 5 expected coverage", fill="black")
    draw.rectangle((x0, y0, x0 + size, y0 + size), outline="black")
    draw.line((x0, y0 + size, x0 + size, y0), fill=(0, 0, 0), width=2)
    colors = [(200, 30, 30), (30, 120, 30), (30, 80, 200), (180, 120, 20), (120, 40, 160)]
    for idx, name in enumerate(NATURAL_PARAM_NAMES):
        points = []
        for level in COVERAGE_LEVELS:
            actual = result["coverage_results"][str(level)][name]
            px = x0 + level * size
            py = y0 + size - actual * size
            points.append((px, py))
        draw.line(points, fill=colors[idx], width=3)
        for px, py in points:
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=colors[idx])
        draw.text((720, 100 + idx * 28), name, fill=colors[idx])
    draw.text((x0 + 200, y0 + size + 40), "nominal coverage", fill="black")
    draw.text((20, y0 + 250), "actual", fill="black")
    image.save(path)


def write_progress(results_dir: Path, done: int, total: int, ranks: dict[str, list[int]]) -> None:
    payload = {
        "done": done,
        "total": total,
        "rank_counts_so_far": {name: len(values) for name, values in ranks.items()},
    }
    (results_dir / "gate5_sbc_progress.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def update_gate_summary(results_dir: Path, gate5: dict) -> None:
    gate1 = _read_json(results_dir / "gate1_sloppiness.json")
    lines = [
        "# Synthetic Gate Summary",
        "",
        "## Gates",
        "",
        f"- Gate 1: {gate1.get('status', 'UNKNOWN')} - {gate1.get('interpretation', gate1.get('reason', ''))}",
        f"  - median kappa: {gate1.get('kappa_median', 'not computed')}",
        f"- Gate 5: {gate5['status']} - {gate5['reason']}",
        f"  - SBC replications: {gate5.get('n_replications', 'not run')}",
        "",
        "## Blocked Gates",
        "",
        "Gates 2, 3, 4, 6, and 7 are BLOCKED pending raw-curve access, NOT-REACHED.",
        "",
    ]
    (results_dir / "GATE_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-replications", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_gate5_sbc(Path.cwd(), n_replications=args.n_replications, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
