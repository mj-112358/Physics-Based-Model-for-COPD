from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares

SIGMA_LPS = 0.10
ADEQUATE_NRSS_THRESHOLD = 2.76
NATURAL_PARAM_NAMES = ("tau_f", "tau_s", "alpha", "fvc_l", "tau_o")


@dataclass(frozen=True)
class FitResult:
    params: dict[str, float]
    nrss: float
    r2: float
    success: bool
    time_shift_s: float = 0.0


def np_ombe_flow(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    tau_f, tau_s, alpha, fvc_l, tau_o = params
    base = alpha * np.exp(-t / tau_f) / tau_f + (1.0 - alpha) * np.exp(-t / tau_s) / tau_s
    rise = 1.0 - np.exp(-t / tau_o)
    q = np.maximum(base * rise, 1e-12)
    area = np.trapezoid(q, t)
    return fvc_l * q / area


def np_ombe_registered_flow(
    params: np.ndarray,
    t: np.ndarray,
    shift_s: float,
    hz: int = 100,
    max_shift_s: float = 0.08,
) -> np.ndarray:
    """Evaluate OMBE flow after a small timing shift.

    Positive shift delays the model relative to the data: prediction at data
    time t is the unshifted model at t - shift. The OMBE shape is unchanged;
    this only tests onset/phase registration.
    """
    if len(t) == 0:
        return np.array([], dtype=float)
    duration_s = max(float(np.nanmax(t)) + max_shift_s + 1.0 / hz, 1.0 / hz)
    model_t = np.arange(0.0, duration_s + 1.0 / hz, 1.0 / hz)
    model_flow = np_ombe_flow(params, model_t)
    shifted_t = np.asarray(t, dtype=float) - float(shift_s)
    return np.interp(shifted_t, model_t, model_flow, left=0.0, right=0.0)


def fit_ombe_lm(t: np.ndarray, flow_lps: np.ndarray, sigma_lps: float = SIGMA_LPS) -> FitResult:
    fvc_est = float(np.clip(np.trapezoid(np.maximum(flow_lps, 0.0), t), 2.0, 6.0))
    x0 = np.array([0.35, 2.50, 0.60, fvc_est, 0.12])
    lo = np.array([0.15, 1.00, 0.25, 2.00, 0.03])
    hi = np.array([0.80, 5.00, 0.85, 6.00, 0.40])

    def residual(x: np.ndarray) -> np.ndarray:
        return (np_ombe_flow(x, t) - flow_lps) / sigma_lps

    opt = least_squares(residual, x0=x0, bounds=(lo, hi), method="trf", max_nfev=600)
    pred = np_ombe_flow(opt.x, t)
    resid = pred - flow_lps
    nrss = float(np.mean((resid / sigma_lps) ** 2))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((flow_lps - np.mean(flow_lps)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return FitResult(
        params=dict(zip(NATURAL_PARAM_NAMES, map(float, opt.x))),
        nrss=nrss,
        r2=float(r2),
        success=bool(opt.success),
    )


def fit_ombe_registered_lm(
    t: np.ndarray,
    flow_lps: np.ndarray,
    sigma_lps: float = SIGMA_LPS,
    max_shift_s: float = 0.08,
) -> FitResult:
    """Fit OMBE flow while jointly optimizing a bounded onset time shift."""
    fvc_est = float(np.clip(np.trapezoid(np.maximum(flow_lps, 0.0), t), 2.0, 6.0))
    x0 = np.array([0.35, 2.50, 0.60, fvc_est, 0.12, 0.0])
    lo = np.array([0.15, 1.00, 0.25, 2.00, 0.03, -max_shift_s])
    hi = np.array([0.80, 5.00, 0.85, 6.00, 0.40, max_shift_s])

    def residual(x: np.ndarray) -> np.ndarray:
        return (np_ombe_registered_flow(x[:5], t, x[5], max_shift_s=max_shift_s) - flow_lps) / sigma_lps

    opt = least_squares(residual, x0=x0, bounds=(lo, hi), method="trf", max_nfev=800)
    pred = np_ombe_registered_flow(opt.x[:5], t, opt.x[5], max_shift_s=max_shift_s)
    resid = pred - flow_lps
    nrss = float(np.mean((resid / sigma_lps) ** 2))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((flow_lps - np.mean(flow_lps)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return FitResult(
        params=dict(zip(NATURAL_PARAM_NAMES, map(float, opt.x[:5]))),
        nrss=nrss,
        r2=float(r2),
        success=bool(opt.success),
        time_shift_s=float(opt.x[5]),
    )


def generate_fit_curves(n: int = 200, seed: int = 10) -> list[dict]:
    import jax.numpy as jnp

    from .ombe import OmbeGrid, ombe_flow, sample_prior_phi, unpack_phi

    phis = sample_prior_phi(n, seed=seed)
    t = np.asarray(OmbeGrid().t)
    rng = np.random.default_rng(seed + 1)
    curves = []
    for i, phi in enumerate(phis):
        truth = {k: float(v) for k, v in unpack_phi(jnp.asarray(phi)).items()}
        clean = np.asarray(ombe_flow(jnp.asarray(phi), jnp.asarray(t)))
        noisy = clean + rng.normal(0.0, SIGMA_LPS, size=len(clean))
        curves.append({"seq": i, "phi": phi, "truth": truth, "t": t, "clean": clean, "flow": noisy})
    return curves


def corrupt_curve(curve: dict, rng: np.random.Generator) -> np.ndarray:
    y = curve["flow"].copy()
    if rng.random() < 0.5:
        idx = int(rng.integers(35, min(450, len(y) - 80)))
        width = int(rng.integers(20, 45))
        amp = rng.uniform(5.0, 10.0)
        phase = np.linspace(0.0, np.pi, width)
        y[idx : idx + width] += amp * np.sin(phase)
    else:
        idx = int(rng.integers(20, 80))
        y[idx:] = rng.normal(0.0, SIGMA_LPS, size=len(y) - idx)
    return y


def validate_fits(n: int = 200, seed: int = 10) -> dict:
    curves = generate_fit_curves(n=n, seed=seed)
    fits = []
    for curve in curves:
        fit = fit_ombe_lm(curve["t"], curve["flow"])
        row = {
            "seq": curve["seq"],
            "nrss": fit.nrss,
            "r2": fit.r2,
            "success": fit.success,
            "adequate": fit.nrss <= ADEQUATE_NRSS_THRESHOLD,
        }
        for name in NATURAL_PARAM_NAMES:
            truth = curve["truth"][name]
            est = fit.params[name]
            row[f"{name}_truth"] = truth
            row[f"{name}_est"] = est
            row[f"{name}_rel_error_pct"] = 100.0 * (est - truth) / truth
        fits.append(row)

    rng = np.random.default_rng(seed + 2)
    corrupt_rows = []
    for curve in curves[:50]:
        corrupted = corrupt_curve(curve, rng)
        fit = fit_ombe_lm(curve["t"], corrupted)
        corrupt_rows.append(
            {
                "seq": curve["seq"],
                "nrss": fit.nrss,
                "r2": fit.r2,
                "flagged_inadequate": fit.nrss > ADEQUATE_NRSS_THRESHOLD,
            }
        )

    result = summarize_fit_rows(fits, corrupt_rows)
    result["fits"] = fits
    result["corrupt_rows"] = corrupt_rows
    result["curves"] = curves
    return result


def summarize_fit_rows(fits: list[dict], corrupt_rows: list[dict]) -> dict:
    n = len(fits)
    adequate = sum(row["adequate"] for row in fits)
    param_summary = {}
    for name in NATURAL_PARAM_NAMES:
        abs_err = np.abs([row[f"{name}_rel_error_pct"] for row in fits])
        param_summary[name] = {
            "median_abs_rel_error_pct": float(np.median(abs_err)),
            "iqr_abs_rel_error_pct": [float(x) for x in np.percentile(abs_err, [25, 75])],
        }
    corrupt_flagged = sum(row["flagged_inadequate"] for row in corrupt_rows)
    return {
        "status": "PASS" if adequate >= int(0.95 * n) and corrupt_flagged >= 45 else "FAIL",
        "n": n,
        "adequate_threshold_nrss": ADEQUATE_NRSS_THRESHOLD,
        "adequate_clean": adequate,
        "adequate_clean_pct": 100.0 * adequate / n,
        "median_nrss": float(np.median([row["nrss"] for row in fits])),
        "median_r2": float(np.median([row["r2"] for row in fits])),
        "param_summary": param_summary,
        "corrupt_n": len(corrupt_rows),
        "corrupt_flagged": corrupt_flagged,
        "corrupt_flagged_pct": 100.0 * corrupt_flagged / len(corrupt_rows),
    }


def write_fit_report(result: dict, results_dir: Path, figures_dir: Path) -> None:
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    lines = [
        "# Module 2 - OMBE Least-Squares Fitter",
        "",
        f"- status: {result['status']}",
        f"- synthetic curves: {result['n']}",
        f"- normalized RSS threshold: {result['adequate_threshold_nrss']:.2f}",
        "- threshold rationale: normalized RSS is mean squared residual divided by known synthetic noise variance; 2.76 is used as a fixed Steltner-like adequacy ceiling, not tuned to this run.",
        f"- clean curves adequate: {result['adequate_clean']}/{result['n']} ({result['adequate_clean_pct']:.1f}%)",
        f"- median normalized RSS: {result['median_nrss']:.3f}",
        f"- median R^2: {result['median_r2']:.4f}",
        f"- corrupted curves flagged inadequate: {result['corrupt_flagged']}/{result['corrupt_n']} ({result['corrupt_flagged_pct']:.1f}%)",
        "",
        "## Parameter Recovery",
        "",
        "| parameter | median abs relative error | IQR |",
        "|---|---:|---|",
    ]
    for name, summary in result["param_summary"].items():
        lines.append(
            f"| `{name}` | {summary['median_abs_rel_error_pct']:.2f}% | "
            f"{summary['iqr_abs_rel_error_pct'][0]:.2f}% to {summary['iqr_abs_rel_error_pct'][1]:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Gate 1 identified `log_fvc` as the stiffest direction and `tau_o`, `alpha`, and time constants as sloppy directions. This report should be read in that identifiability context.",
            "",
        ]
    )
    (results_dir / "mod2_fit.md").write_text("\n".join(lines), encoding="utf-8")
    draw_fit_quality(result, figures_dir / "mod2_fit_quality.png")


def draw_fit_quality(result: dict, path: Path) -> None:
    image = Image.new("RGB", (1000, 650), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 25), "Module 2 fit quality", fill="black")
    x0, y0, w, h = 90, 80, 820, 450
    draw.rectangle((x0, y0, x0 + w, y0 + h), outline="black")
    nrss = np.array([row["nrss"] for row in result["fits"]])
    r2 = np.array([row["r2"] for row in result["fits"]])
    x_min, x_max = 0.0, max(5.0, float(np.percentile(nrss, 99)))
    y_min, y_max = max(0.0, float(np.min(r2)) - 0.02), 1.0
    for xval, yval in zip(nrss, r2):
        px = x0 + np.clip((xval - x_min) / (x_max - x_min), 0, 1) * w
        py = y0 + (y_max - yval) / (y_max - y_min) * h
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(30, 90, 180))
    threshold_x = x0 + (result["adequate_threshold_nrss"] - x_min) / (x_max - x_min) * w
    draw.line((threshold_x, y0, threshold_x, y0 + h), fill=(200, 40, 40), width=3)
    draw.text((x0 + 300, y0 + h + 35), "normalized RSS", fill="black")
    draw.text((20, y0 + 200), "R^2", fill="black")
    draw.text((threshold_x + 5, y0 + 10), "2.76", fill=(200, 40, 40))
    image.save(path)


def main() -> int:
    result = validate_fits()
    write_fit_report(result, Path("results"), Path("figures"))
    print(f"Module 2 {result['status']}: clean adequate {result['adequate_clean']}/{result['n']}, corrupt flagged {result['corrupt_flagged']}/{result['corrupt_n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
