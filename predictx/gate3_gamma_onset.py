from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import chi2

from .gate3_diagnosis import (
    classify_peak_residual,
    clinical_metrics_from_volume,
    coerce_bool,
    cumulative_trapezoid,
    extract_curve,
    pct_error,
    savgol_sigma,
    select_sample,
)
from .gate23_f import find_or_unzip_raw, read_raw
from .ombe_gamma import (
    NATURAL_PARAM_NAMES,
    OmbeGammaGrid,
    batched_fisher_information,
    flow_integral_l,
    midrange_phi,
    sample_prior_phi,
)

BASELINE_DIPOLE_FRACTION = 0.845
BASELINE_FLOW_NRSS_UNREGISTERED = 19.572
BASELINE_FLOW_NRSS_REGISTERED = 6.951
BASELINE_ALL_WITHIN_5PCT = 0.840
BASELINE_RATIO_WITHIN_0P02 = 0.675
REGISTERED_BASELINE_TAU_O_BOUND_N = 100
REGISTERED_BASELINE_TAU_S_BOUND_N = 97

GAMMA_PARAM_NAMES = ("tau_f", "tau_s", "alpha", "fvc_l", "tau_o", "k")
GAMMA_LO = np.array([0.15, 1.00, 0.25, 2.00, 0.03, 1.00])
GAMMA_HI = np.array([0.80, 5.00, 0.85, 6.00, 0.40, 8.00])


def np_ombe_gamma_flow(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    tau_f, tau_s, alpha, fvc_l, tau_o, k = params
    x = np.maximum(t / tau_o, 1e-12)
    gamma = np.where(t > 0.0, np.power(x, k) * np.exp(-x), 0.0)
    base = alpha * np.exp(-t / tau_f) / tau_f + (1.0 - alpha) * np.exp(-t / tau_s) / tau_s
    q = np.maximum(gamma * base, 1e-30)
    area = np.trapezoid(q, t)
    return fvc_l * q / area


def np_ombe_gamma_volume(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    return cumulative_trapezoid(np_ombe_gamma_flow(params, t), t)


def assert_volume_conservation() -> dict:
    t = OmbeGammaGrid().t
    phi = midrange_phi()
    integral = float(flow_integral_l(phi, t))
    fvc = 4.0
    rel_error = abs(integral - fvc) / fvc
    if rel_error > 0.01:
        raise RuntimeError(f"OMBE-gamma volume conservation failed: integral={integral}, fvc={fvc}, rel={rel_error}")
    return {"midrange_integral_l": integral, "midrange_fvc_l": fvc, "relative_error": rel_error}


def run_fisher(seed: int = 41, n: int = 1000) -> dict:
    phis = sample_prior_phi(n, seed=seed)
    t = OmbeGammaGrid().t
    fim = np.asarray(batched_fisher_information(jnp.asarray(phis), t, 0.10))
    eigvals = []
    stiff_loadings = []
    sloppy_loadings = []
    for mat in fim:
        vals, vecs = np.linalg.eigh(mat)
        vals = np.asarray(vals, dtype=float)
        eigvals.append(vals)
        stiff_loadings.append(np.abs(vecs[:, -1]))
        sloppy_loadings.append(np.abs(vecs[:, 0]))
    eigvals = np.asarray(eigvals)
    min_vals = eigvals[:, 0]
    max_vals = eigvals[:, -1]
    kappa = np.where(min_vals > 0, max_vals / min_vals, np.inf)
    finite = kappa[np.isfinite(kappa)]
    summary = {
        "n": int(n),
        "finite_kappa_n": int(len(finite)),
        "kappa_median": float(np.median(finite)) if len(finite) else np.inf,
        "kappa_iqr": [float(x) for x in np.percentile(finite, [25, 75])] if len(finite) else [np.inf, np.inf],
        "kappa_p95": float(np.percentile(finite, 95)) if len(finite) else np.inf,
        "kappa_max": float(np.max(finite)) if len(finite) else np.inf,
        "kappa_gt_1e4_fraction": float(np.mean(kappa > 1e4)),
        "kappa_gt_1e6_fraction": float(np.mean(kappa > 1e6)),
        "stiff_loading_median": dict(zip(NATURAL_PARAM_NAMES, map(float, np.median(stiff_loadings, axis=0)))),
        "sloppy_loading_median": dict(zip(NATURAL_PARAM_NAMES, map(float, np.median(sloppy_loadings, axis=0)))),
    }
    k_loading = summary["sloppy_loading_median"]["k"]
    summary["identifiability_flag"] = bool(
        summary["kappa_median"] > 1e4 or summary["kappa_gt_1e4_fraction"] > 0.50 or k_loading >= 0.50
    )
    if summary["kappa_median"] > 1e4:
        summary["identifiability_reason"] = "median kappa exceeds 1e4"
    elif summary["kappa_gt_1e4_fraction"] > 0.50:
        summary["identifiability_reason"] = "more than half of prior samples have kappa > 1e4"
    elif k_loading >= 0.50:
        summary["identifiability_reason"] = "k dominates the median sloppy direction"
    else:
        summary["identifiability_reason"] = "no pre-specified identifiability failure flag"
    return summary


def fit_gamma_curve(fit_row: Any, raw_row: Any) -> tuple[dict, dict]:
    curve = extract_curve(raw_row)
    sigma_flow = savgol_sigma(curve["fit_flow"])
    old_params = np.array(
        [
            float(fit_row.fit_tau_f),
            float(fit_row.fit_tau_s),
            float(fit_row.fit_alpha),
            float(fit_row.fit_fvc_l),
            float(fit_row.fit_tau_o),
            2.5,
        ]
    )
    x0 = np.clip(old_params, GAMMA_LO + 1e-6, GAMMA_HI - 1e-6)
    scale = sigma_flow if np.isfinite(sigma_flow) and sigma_flow > 0 else 0.10

    def residual(x: np.ndarray) -> np.ndarray:
        return (np_ombe_gamma_flow(x, curve["fit_t"]) - curve["fit_flow"]) / scale

    opt = least_squares(residual, x0=x0, bounds=(GAMMA_LO, GAMMA_HI), method="trf", max_nfev=1200)
    pred_flow = np_ombe_gamma_flow(opt.x, curve["fit_t"])
    flow_resid = curve["fit_flow"] - pred_flow
    peak = classify_peak_residual(curve["fit_t"], flow_resid, curve["pef_t"], sigma_flow)

    dof = max(len(curve["fit_t"]) - len(GAMMA_PARAM_NAMES), 1)
    rss = float(np.sum(flow_resid**2))
    nrss_norm = float(rss / (dof * sigma_flow**2)) if np.isfinite(sigma_flow) and sigma_flow > 0 else np.nan
    threshold = float(chi2.ppf(0.95, dof) / dof)
    ss_tot = float(np.sum((curve["fit_flow"] - np.mean(curve["fit_flow"])) ** 2))
    r2 = 1.0 - float(np.sum(flow_resid**2)) / ss_tot if ss_tot > 0 else np.nan

    pred_volume = np_ombe_gamma_volume(opt.x, curve["vol_t"])
    pred_fev1, pred_fvc, pred_ratio = clinical_metrics_from_volume(curve["vol_t"], pred_volume)
    truth_fev1, truth_fvc, truth_ratio = clinical_metrics_from_volume(curve["vol_t"], curve["vol_l"])
    fev1_pct = pct_error(pred_fev1, truth_fev1)
    fvc_pct = pct_error(pred_fvc, truth_fvc)
    ratio_pct = pct_error(pred_ratio, truth_ratio)
    ratio_abs = float(abs(pred_ratio - truth_ratio)) if np.isfinite(pred_ratio) and np.isfinite(truth_ratio) else np.nan

    row = fit_row._asdict()
    row.update(
        {
            "gamma_error": "",
            "gamma_success": bool(opt.success),
            "gamma_flow_nrss_norm": nrss_norm,
            "gamma_flow_threshold_95": threshold,
            "gamma_flow_adequate": bool(nrss_norm <= threshold) if np.isfinite(nrss_norm) else False,
            "gamma_flow_r2": float(r2),
            "sigma_flow_lps": sigma_flow,
            **peak,
            "gamma_fev1_l": pred_fev1,
            "gamma_fvc_l": pred_fvc,
            "gamma_ratio": pred_ratio,
            "truth_fev1_l": truth_fev1,
            "truth_fvc_l": truth_fvc,
            "truth_ratio": truth_ratio,
            "gamma_fev1_pct_error": fev1_pct,
            "gamma_fvc_pct_error": fvc_pct,
            "gamma_ratio_pct_error": ratio_pct,
            "gamma_ratio_abs_error": ratio_abs,
            "gamma_all_within_5pct": bool(abs(fev1_pct) <= 5.0 and abs(fvc_pct) <= 5.0 and abs(ratio_pct) <= 5.0),
            "gamma_ratio_within_0p02": bool(ratio_abs <= 0.02),
        }
    )
    for name, value in zip(GAMMA_PARAM_NAMES, opt.x):
        row[f"gamma_{name}"] = float(value)
    plot = {
        "label": f"SEQN {int(fit_row.SEQN)} m{int(fit_row.SPAMANU)} k {opt.x[-1]:.2f}",
        "t": curve["fit_t"],
        "flow": curve["fit_flow"],
        "pred_flow": pred_flow,
        "residual": flow_resid,
        "volume_t": curve["vol_t"],
        "volume": curve["vol_l"],
        "pred_volume": pred_volume,
        "pef_t": curve["pef_t"],
    }
    return row, plot


def run_gamma_sample(root: Path) -> tuple[pd.DataFrame, list[dict], dict]:
    fits = pd.read_csv(root / "results" / "gate3_F_curve_fits.csv")
    fits["obstructed"] = coerce_bool(fits["obstructed"])
    fits["fit_error"] = fits["fit_error"].fillna("")
    sample = select_sample(fits)

    raw_path, raw_note = find_or_unzip_raw(root)
    raw, raw_info = read_raw(raw_path)
    raw_pre = raw[raw["SPATTYPE"] == "Pre"]
    raw_by_key = {(float(row.SEQN), float(row.SPAMANU)): row for row in raw_pre.itertuples(index=False)}

    rows = []
    plots = []
    for idx, fit_row in enumerate(sample.itertuples(index=False)):
        base = fit_row._asdict()
        key = (float(fit_row.SEQN), float(fit_row.SPAMANU))
        if key not in raw_by_key:
            base["gamma_error"] = "raw curve not found"
            rows.append(base)
            continue
        try:
            row, plot = fit_gamma_curve(fit_row, raw_by_key[key])
            rows.append(row)
            if len(plots) < 6:
                plots.append(plot)
        except Exception as exc:
            base["gamma_error"] = str(exc)
            rows.append(base)
        if (idx + 1) % 50 == 0 or idx + 1 == len(sample):
            print(f"OMBE-gamma sample fit progress: {idx + 1}/{len(sample)} curves")
    return pd.DataFrame(rows), plots, {"raw_source_note": raw_note, "raw_info": raw_info}


def bound_count(rows: pd.DataFrame, col: str, lo: float, hi: float) -> dict:
    valid = rows[rows["gamma_error"].fillna("") == ""]
    return {
        "lower": int((valid[col] <= lo + 1e-4).sum()),
        "upper": int((valid[col] >= hi - 1e-4).sum()),
        "any": int(((valid[col] <= lo + 1e-4) | (valid[col] >= hi - 1e-4)).sum()),
        "fraction_any": float(((valid[col] <= lo + 1e-4) | (valid[col] >= hi - 1e-4)).mean()) if len(valid) else np.nan,
    }


def summarize_sample(rows: pd.DataFrame) -> dict:
    valid = rows[rows["gamma_error"].fillna("") == ""].copy()
    clinical_valid = valid.dropna(subset=["gamma_fev1_pct_error", "gamma_fvc_pct_error", "gamma_ratio_pct_error"])
    peak_counts = valid["peak_class"].value_counts(dropna=False).to_dict()
    result = {
        "sample_n": int(len(rows)),
        "sample_valid_n": int(len(valid)),
        "sample_error_n": int(len(rows) - len(valid)),
        "peak_class_counts": {str(k): int(v) for k, v in peak_counts.items()},
        "dipole_fraction": float((valid["peak_class"] == "dipole").mean()) if len(valid) else np.nan,
        "single_signed_fraction": float(valid["peak_class"].isin(["single_signed_undershoot", "single_signed_overshoot"]).mean()) if len(valid) else np.nan,
        "flow_nrss_median": float(np.nanmedian(valid["gamma_flow_nrss_norm"])) if len(valid) else np.nan,
        "flow_nrss_iqr": [float(x) for x in np.nanpercentile(valid["gamma_flow_nrss_norm"], [25, 75])] if len(valid) else [np.nan, np.nan],
        "flow_adequate_n": int(valid["gamma_flow_adequate"].sum()) if len(valid) else 0,
        "flow_adequate_fraction": float(valid["gamma_flow_adequate"].mean()) if len(valid) else np.nan,
        "all_within_5pct_n": int(clinical_valid["gamma_all_within_5pct"].sum()) if len(clinical_valid) else 0,
        "all_within_5pct_fraction": float(clinical_valid["gamma_all_within_5pct"].mean()) if len(clinical_valid) else np.nan,
        "ratio_within_0p02_n": int(clinical_valid["gamma_ratio_within_0p02"].sum()) if len(clinical_valid) else 0,
        "ratio_within_0p02_fraction": float(clinical_valid["gamma_ratio_within_0p02"].mean()) if len(clinical_valid) else np.nan,
        "fev1_abs_pct_error_median": float(np.nanmedian(np.abs(clinical_valid["gamma_fev1_pct_error"]))) if len(clinical_valid) else np.nan,
        "fvc_abs_pct_error_median": float(np.nanmedian(np.abs(clinical_valid["gamma_fvc_pct_error"]))) if len(clinical_valid) else np.nan,
        "ratio_abs_error_median": float(np.nanmedian(clinical_valid["gamma_ratio_abs_error"])) if len(clinical_valid) else np.nan,
        "tau_o_bound": bound_count(rows, "gamma_tau_o", GAMMA_LO[4], GAMMA_HI[4]),
        "tau_s_bound": bound_count(rows, "gamma_tau_s", GAMMA_LO[1], GAMMA_HI[1]),
        "k_bound": bound_count(rows, "gamma_k", GAMMA_LO[5], GAMMA_HI[5]),
        "k_median": float(np.nanmedian(valid["gamma_k"])) if len(valid) else np.nan,
        "k_iqr": [float(x) for x in np.nanpercentile(valid["gamma_k"], [25, 75])] if len(valid) else [np.nan, np.nan],
    }
    result["step3_gamma_helps"] = bool(
        result["dipole_fraction"] < 0.30
        and result["tau_o_bound"]["any"] < 50
        and result["tau_s_bound"]["any"] < 50
        and result["k_bound"]["any"] < 50
        and result["ratio_within_0p02_fraction"] > BASELINE_RATIO_WITHIN_0P02 + 0.05
        and result["all_within_5pct_fraction"] >= BASELINE_ALL_WITHIN_5PCT
    )
    result["step3_gate_reason"] = (
        f"dipoles<30%={result['dipole_fraction'] < 0.30}; "
        f"tau_o/tau_s/k bound-piling<50 each={result['tau_o_bound']['any'] < 50 and result['tau_s_bound']['any'] < 50 and result['k_bound']['any'] < 50}; "
        f"ratio-within-0.02 materially above baseline={result['ratio_within_0p02_fraction'] > BASELINE_RATIO_WITHIN_0P02 + 0.05}; "
        f"all-within-5% not below baseline={result['all_within_5pct_fraction'] >= BASELINE_ALL_WITHIN_5PCT}"
    )
    return result


def draw_examples(plots: list[dict], figures_dir: Path) -> None:
    if not plots:
        return
    figures_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(len(plots), 2, figsize=(14, 2.6 * len(plots)), constrained_layout=True)
    if len(plots) == 1:
        axes = np.array([axes])
    for idx, plot in enumerate(plots):
        ax_flow = axes[idx, 0]
        ax_vol = axes[idx, 1]
        ax_flow.plot(plot["t"], plot["flow"], color="#1f5aa6", lw=1.0, label="flow")
        ax_flow.plot(plot["t"], plot["pred_flow"], color="#d62728", lw=1.2, label="OMBE-gamma")
        ax_flow.plot(plot["t"], plot["residual"], color="#111111", lw=0.8, label="residual")
        ax_flow.axvline(plot["pef_t"], color="#f59e0b", lw=0.9)
        ax_flow.axhline(0.0, color="#999999", lw=0.7)
        ax_flow.set_title(f"{plot['label']}: flow")
        ax_flow.set_ylabel("L/s")
        ax_vol.plot(plot["volume_t"], plot["volume"], color="#1f5aa6", lw=1.0, label="parsed volume")
        ax_vol.plot(plot["volume_t"], plot["pred_volume"], color="#d62728", lw=1.2, label="OMBE-gamma volume")
        ax_vol.set_title(f"{plot['label']}: volume")
        ax_vol.set_ylabel("L")
    axes[0, 0].legend(loc="upper right")
    axes[0, 1].legend(loc="lower right")
    for ax in axes.ravel():
        ax.set_xlabel("time from t0 (s)")
    fig.suptitle("OMBE-gamma sample fits", y=1.01)
    fig.savefig(figures_dir / "gate3_F_gamma_onset_examples.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_report(result: dict, rows: pd.DataFrame, results_dir: Path) -> None:
    lines = [
        "# OMBE-Gamma Onset Gate",
        "",
        f"- status: {result['status']}",
        f"- reason: {result['reason']}",
        "- scope: cycle F 200-curve A-grade non-obstructed sample unless Step 3 passes.",
        "- model: `gamma(t)=(t/tau_o)^k exp(-t/tau_o)` times the bi-exponential emptying term, normalized so integral flow equals FVC.",
        "",
        "## Step 1 - Volume Conservation",
        "",
        f"- midrange integral: {result['volume_conservation']['midrange_integral_l']:.6f} L",
        f"- midrange FVC: {result['volume_conservation']['midrange_fvc_l']:.6f} L",
        f"- relative error: {100 * result['volume_conservation']['relative_error']:.6f}%",
        "",
        "## Step 2 - Identifiability",
        "",
        f"- Fisher samples: {result['fisher']['n']}",
        f"- kappa median: {result['fisher']['kappa_median']:.3g}",
        f"- kappa IQR: {result['fisher']['kappa_iqr'][0]:.3g} to {result['fisher']['kappa_iqr'][1]:.3g}",
        f"- kappa p95: {result['fisher']['kappa_p95']:.3g}",
        f"- kappa >1e4 fraction: {result['fisher']['kappa_gt_1e4_fraction']:.3f}",
        f"- kappa >1e6 fraction: {result['fisher']['kappa_gt_1e6_fraction']:.3f}",
        f"- identifiability flag: {result['fisher']['identifiability_flag']} ({result['fisher']['identifiability_reason']})",
        "",
        "Median stiff-direction loadings:",
        "",
        "| parameter | loading |",
        "|---|---:|",
    ]
    for name, value in result["fisher"]["stiff_loading_median"].items():
        lines.append(f"| `{name}` | {value:.3f} |")
    lines.extend(["", "Median sloppy-direction loadings:", "", "| parameter | loading |", "|---|---:|"])
    for name, value in result["fisher"]["sloppy_loading_median"].items():
        lines.append(f"| `{name}` | {value:.3f} |")
    lines.extend(
        [
            "",
            "## Step 3 - 200-Curve Flow-Domain Fit",
            "",
            "| metric | bi-exponential baseline | registered baseline | OMBE-gamma |",
            "|---|---:|---:|---:|",
            f"| dipole fraction | {BASELINE_DIPOLE_FRACTION:.3f} | 0.535 | {result['sample']['dipole_fraction']:.3f} |",
            f"| flow NRSS_norm median | {BASELINE_FLOW_NRSS_UNREGISTERED:.3f} | {BASELINE_FLOW_NRSS_REGISTERED:.3f} | {result['sample']['flow_nrss_median']:.3f} |",
            f"| all-three-within-5% | {BASELINE_ALL_WITHIN_5PCT:.3f} | 0.680 | {result['sample']['all_within_5pct_fraction']:.3f} |",
            f"| FEV1/FVC within 0.02 | {BASELINE_RATIO_WITHIN_0P02:.3f} | 0.435 | {result['sample']['ratio_within_0p02_fraction']:.3f} |",
            "",
            f"- flow adequate fraction: {result['sample']['flow_adequate_fraction']:.3f}",
            f"- FEV1 median abs error: {result['sample']['fev1_abs_pct_error_median']:.2f}%",
            f"- FVC median abs error: {result['sample']['fvc_abs_pct_error_median']:.2f}%",
            f"- FEV1/FVC median absolute error: {result['sample']['ratio_abs_error_median']:.4f}",
            f"- gamma k median: {result['sample']['k_median']:.3f}",
            f"- gamma k IQR: {result['sample']['k_iqr'][0]:.3f} to {result['sample']['k_iqr'][1]:.3f}",
            "",
            "Parameter-at-bound counts:",
            "",
            "| parameter | lower | upper | any | fraction any |",
            "|---|---:|---:|---:|---:|",
            f"| `tau_o` | {result['sample']['tau_o_bound']['lower']} | {result['sample']['tau_o_bound']['upper']} | {result['sample']['tau_o_bound']['any']} | {result['sample']['tau_o_bound']['fraction_any']:.3f} |",
            f"| `tau_s` | {result['sample']['tau_s_bound']['lower']} | {result['sample']['tau_s_bound']['upper']} | {result['sample']['tau_s_bound']['any']} | {result['sample']['tau_s_bound']['fraction_any']:.3f} |",
            f"| `k` | {result['sample']['k_bound']['lower']} | {result['sample']['k_bound']['upper']} | {result['sample']['k_bound']['any']} | {result['sample']['k_bound']['fraction_any']:.3f} |",
            "",
            f"- Step 3 gamma-helps gate: {result['sample']['step3_gamma_helps']}",
            f"- Step 3 gate reason: {result['sample']['step3_gate_reason']}",
            "",
            "## Verdict",
            "",
            result["verdict"],
            "",
        ]
    )
    results_dir.mkdir(exist_ok=True)
    (results_dir / "gate3_F_gamma_onset.md").write_text("\n".join(lines), encoding="utf-8")
    (results_dir / "gate3_F_gamma_onset.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    rows.to_csv(results_dir / "gate3_F_gamma_onset_sample.csv", index=False)


def run(root: Path) -> dict:
    volume = assert_volume_conservation()
    fisher = run_fisher()
    rows, plots, raw_meta = run_gamma_sample(root)
    sample = summarize_sample(rows)

    if fisher["identifiability_flag"] and not sample["step3_gamma_helps"]:
        status = "FAIL"
        reason = "OMBE-gamma fails identifiability and does not clear the 200-curve gamma-helps gate."
        verdict = "OMBE-gamma is not adopted. The augmentation adds complexity without clearing the onset, clinical, and identifiability gates."
    elif fisher["identifiability_flag"]:
        status = "FAIL_IDENTIFIABILITY"
        reason = "OMBE-gamma improves the sample gate but fails the identifiability gate."
        verdict = "Do not adopt OMBE-gamma unless the identifiability issue is resolved; fall back to volume-domain OMBE with the onset limitation documented."
    elif not sample["step3_gamma_helps"]:
        status = "FAIL_SAMPLE"
        reason = "OMBE-gamma is identifiable by the pre-specified checks but does not clear the 200-curve gamma-helps gate."
        verdict = "Do not run full-scale or SBC. The gamma onset did not jointly fix dipoles, bound-piling, and clinical ratio precision."
    else:
        status = "SAMPLE_PASS"
        reason = "OMBE-gamma clears identifiability and the 200-curve gamma-helps gate; full-scale/SBC is now allowed."
        verdict = "Proceed to full cycle-F clinical reproduction and OMBE-gamma SBC calibration."

    result = {
        "status": status,
        "reason": reason,
        "volume_conservation": volume,
        "fisher": fisher,
        "sample": sample,
        "raw_source_note": raw_meta["raw_source_note"],
        "raw_info": raw_meta["raw_info"],
        "verdict": verdict,
    }
    draw_examples(plots, root / "figures")
    write_report(result, rows, root / "results")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    print(f"OMBE-gamma {result['status']}: {result['reason']}")
    print(f"kappa median: {result['fisher']['kappa_median']:.3g}")
    print(f"dipole fraction: {result['sample']['dipole_fraction']:.3f}")
    print(f"ratio within 0.02: {result['sample']['ratio_within_0p02_fraction']:.3f}")
    return 0 if result["status"] == "SAMPLE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
