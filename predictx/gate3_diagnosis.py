from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.stats import chi2

from .fit import np_ombe_flow
from .gate23_f import (
    HZ,
    backextrap_t0,
    detect_forced_exhale_segment,
    find_or_unzip_raw,
    parse_increments,
    read_raw,
)

SAMPLE_N = 200
SAMPLE_SEED = 20260707
N_PARAMS = 5
PEAK_WINDOW_S = 0.12


def coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.map({True: True, False: False, "True": True, "False": False, "true": True, "false": False}).fillna(False)


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    if len(y) == 0:
        return np.array([], dtype=float)
    increments = (y[:-1] + y[1:]) * 0.5 * np.diff(x)
    return np.concatenate([[0.0], np.cumsum(increments)])


def np_ombe_volume(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    return cumulative_trapezoid(np_ombe_flow(params, t), t)


def mad_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return float(mad / 0.6745)


def savgol_sigma(values: np.ndarray) -> float:
    n = len(values)
    if n < 11:
        return np.nan
    window = min(31, n if n % 2 == 1 else n - 1)
    if window < 11:
        return np.nan
    polyorder = min(3, window - 2)
    smooth = savgol_filter(values, window_length=window, polyorder=polyorder, mode="interp")
    return mad_sigma(values - smooth)


def extract_curve(row: Any) -> dict:
    incr = parse_increments(row.SPXRAW)
    fact = float(row.SPAFACT)
    flow = incr * 0.1 * fact
    volume_ml = np.cumsum(incr) * fact
    time = np.arange(len(incr), dtype=float) / HZ
    onset_idx, pef_idx, end_idx = detect_forced_exhale_segment(flow, hz=HZ)
    t0_s = backextrap_t0(time, flow, volume_ml, onset_idx, pef_idx)
    start_idx = max(0, int(np.searchsorted(time, max(t0_s, 0.0), side="left")))
    end_idx = max(end_idx, start_idx + 20)
    end_idx = min(end_idx, len(flow) - 1)

    fit_t = time[start_idx : end_idx + 1] - t0_s
    fit_flow = flow[start_idx : end_idx + 1]
    fit_t = fit_t - fit_t[0]

    v0_ml = float(np.interp(t0_s, time, volume_ml))
    vol_t = time[start_idx : end_idx + 1] - t0_s
    vol_l = (volume_ml[start_idx : end_idx + 1] - v0_ml) / 1000.0
    if vol_t[0] > 1e-9:
        vol_t = np.concatenate([[0.0], vol_t])
        vol_l = np.concatenate([[0.0], vol_l])
    else:
        vol_l = vol_l - vol_l[0]
        vol_t = vol_t - vol_t[0]

    pef_t = float(time[pef_idx] - time[start_idx])
    return {
        "fit_t": fit_t,
        "fit_flow": fit_flow,
        "vol_t": vol_t,
        "vol_l": vol_l,
        "pef_t": pef_t,
        "duration_s": float(vol_t[-1]),
        "truth_fev1_l": float(np.interp(min(1.0, vol_t[-1]), vol_t, vol_l)),
        "truth_fvc_l": float(vol_l[-1]),
    }


def classify_peak_residual(fit_t: np.ndarray, residual: np.ndarray, pef_t: float, sigma_hat: float) -> dict:
    pre = (fit_t >= max(0.0, pef_t - PEAK_WINDOW_S)) & (fit_t < pef_t)
    post = (fit_t > pef_t) & (fit_t <= pef_t + PEAK_WINDOW_S)
    peak = (fit_t >= max(0.0, pef_t - PEAK_WINDOW_S)) & (fit_t <= pef_t + PEAK_WINDOW_S)
    if pre.sum() < 3 or post.sum() < 3 or peak.sum() < 6:
        return {
            "peak_class": "unclassified",
            "peak_pre_mean_lps": np.nan,
            "peak_post_mean_lps": np.nan,
            "peak_positive_fraction": np.nan,
            "peak_negative_fraction": np.nan,
        }
    pre_mean = float(np.mean(residual[pre]))
    post_mean = float(np.mean(residual[post]))
    positive_fraction = float(np.mean(residual[peak] > 0))
    negative_fraction = float(np.mean(residual[peak] < 0))
    significant_scale = sigma_hat if np.isfinite(sigma_hat) and sigma_hat > 0 else 0.0
    sign_change = np.sign(pre_mean) * np.sign(post_mean) < 0
    significant = abs(pre_mean) > significant_scale and abs(post_mean) > significant_scale
    if sign_change and significant:
        peak_class = "dipole"
    elif positive_fraction >= 0.80:
        peak_class = "single_signed_undershoot"
    elif negative_fraction >= 0.80:
        peak_class = "single_signed_overshoot"
    else:
        peak_class = "mixed"
    return {
        "peak_class": peak_class,
        "peak_pre_mean_lps": pre_mean,
        "peak_post_mean_lps": post_mean,
        "peak_positive_fraction": positive_fraction,
        "peak_negative_fraction": negative_fraction,
    }


def fit_ombe_volume_lm(t: np.ndarray, volume_l: np.ndarray) -> dict:
    fvc_est = float(np.clip(volume_l[-1], 2.0, 6.0))
    x0 = np.array([0.35, 2.50, 0.60, fvc_est, 0.12])
    lo = np.array([0.15, 1.00, 0.25, 2.00, 0.03])
    hi = np.array([0.80, 5.00, 0.85, 6.00, 0.40])
    sigma_l = savgol_sigma(volume_l)
    scale = sigma_l if np.isfinite(sigma_l) and sigma_l > 0 else 1.0

    def residual(x: np.ndarray) -> np.ndarray:
        return (np_ombe_volume(x, t) - volume_l) / scale

    opt = least_squares(residual, x0=x0, bounds=(lo, hi), method="trf", max_nfev=1000)
    pred_volume = np_ombe_volume(opt.x, t)
    raw_residual = pred_volume - volume_l
    dof = max(len(t) - N_PARAMS, 1)
    rss = float(np.sum(raw_residual**2))
    nrss_norm = float(rss / (dof * sigma_l**2)) if np.isfinite(sigma_l) and sigma_l > 0 else np.nan
    threshold = float(chi2.ppf(0.95, dof) / dof)
    ss_tot = float(np.sum((volume_l - np.mean(volume_l)) ** 2))
    r2 = 1.0 - float(np.sum(raw_residual**2)) / ss_tot if ss_tot > 0 else np.nan
    return {
        "params": opt.x,
        "success": bool(opt.success),
        "sigma_volume_l": float(sigma_l),
        "volume_nrss_norm": nrss_norm,
        "volume_nrss_threshold_95": threshold,
        "volume_adequate_norm": bool(nrss_norm <= threshold) if np.isfinite(nrss_norm) else False,
        "volume_r2": float(r2),
        "pred_volume_l": pred_volume,
    }


def clinical_metrics_from_volume(t: np.ndarray, volume_l: np.ndarray) -> tuple[float, float, float]:
    fev1 = float(np.interp(min(1.0, t[-1]), t, volume_l))
    fvc = float(volume_l[-1])
    ratio = fev1 / fvc if fvc > 0 else np.nan
    return fev1, fvc, ratio


def pct_error(est: float, truth: float) -> float:
    return float(100.0 * (est - truth) / truth) if truth and np.isfinite(truth) else np.nan


def select_sample(fits: pd.DataFrame) -> pd.DataFrame:
    fits = fits.copy()
    fits["obstructed"] = coerce_bool(fits["obstructed"])
    fits["fit_error"] = fits["fit_error"].fillna("")
    param_cols = ["fit_tau_f", "fit_tau_s", "fit_alpha", "fit_fvc_l", "fit_tau_o"]
    eligible = fits[
        (fits["SPAQEFF"] == "A")
        & (~fits["obstructed"])
        & (fits["fit_error"] == "")
        & fits[param_cols].apply(np.isfinite).all(axis=1)
    ].copy()
    if len(eligible) < SAMPLE_N:
        raise RuntimeError(f"Only {len(eligible)} eligible A-grade non-obstructed curves; need {SAMPLE_N}.")
    sample = eligible.sample(n=SAMPLE_N, random_state=SAMPLE_SEED).sort_values(["SEQN", "SPAMANU"])
    return sample


def analyze_sample(root: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    fits_path = root / "results" / "gate3_F_curve_fits.csv"
    if not fits_path.exists():
        raise RuntimeError(f"Missing prior Gate 3 fit table: {fits_path}")
    fits = pd.read_csv(fits_path)
    fits["obstructed"] = coerce_bool(fits["obstructed"])
    fits["fit_error"] = fits["fit_error"].fillna("")
    eligible_n = int(((fits["SPAQEFF"] == "A") & (~fits["obstructed"]) & (fits["fit_error"] == "")).sum())
    sample = select_sample(fits)

    raw_path, raw_note = find_or_unzip_raw(root)
    raw, raw_info = read_raw(raw_path)
    raw_pre = raw[raw["SPATTYPE"] == "Pre"]
    raw_by_key = {(float(row.SEQN), float(row.SPAMANU)): row for row in raw_pre.itertuples(index=False)}

    rows = []
    plot_payloads = []
    for idx, fit_row in enumerate(sample.itertuples(index=False)):
        base = fit_row._asdict()
        key = (float(fit_row.SEQN), float(fit_row.SPAMANU))
        if key not in raw_by_key:
            base["diagnostic_error"] = "raw curve not found"
            rows.append(base)
            continue
        try:
            curve = extract_curve(raw_by_key[key])
            flow_params = np.array(
                [fit_row.fit_tau_f, fit_row.fit_tau_s, fit_row.fit_alpha, fit_row.fit_fvc_l, fit_row.fit_tau_o],
                dtype=float,
            )
            flow_pred = np_ombe_flow(flow_params, curve["fit_t"])
            flow_residual = curve["fit_flow"] - flow_pred
            sigma_flow = savgol_sigma(curve["fit_flow"])
            peak = classify_peak_residual(curve["fit_t"], flow_residual, curve["pef_t"], sigma_flow)

            volume_fit = fit_ombe_volume_lm(curve["vol_t"], curve["vol_l"])
            fit_fev1, fit_fvc, fit_ratio = clinical_metrics_from_volume(curve["vol_t"], volume_fit["pred_volume_l"])
            truth_fev1, truth_fvc, truth_ratio = clinical_metrics_from_volume(curve["vol_t"], curve["vol_l"])
            fev1_pct = pct_error(fit_fev1, truth_fev1)
            fvc_pct = pct_error(fit_fvc, truth_fvc)
            ratio_pct = pct_error(fit_ratio, truth_ratio)
            ratio_abs = float(abs(fit_ratio - truth_ratio)) if np.isfinite(fit_ratio) and np.isfinite(truth_ratio) else np.nan
            base.update(
                {
                    "diagnostic_error": "",
                    "sigma_flow_lps": sigma_flow,
                    **peak,
                    "truth_fev1_l": truth_fev1,
                    "truth_fvc_l": truth_fvc,
                    "truth_ratio": truth_ratio,
                    "fit_fev1_l": fit_fev1,
                    "fit_fvc_l": fit_fvc,
                    "fit_ratio": fit_ratio,
                    "fev1_pct_error": fev1_pct,
                    "fvc_pct_error": fvc_pct,
                    "ratio_pct_error": ratio_pct,
                    "ratio_abs_error": ratio_abs,
                    "clinical_all_within_5pct": bool(abs(fev1_pct) <= 5.0 and abs(fvc_pct) <= 5.0 and abs(ratio_pct) <= 5.0),
                    "clinical_ratio_within_5pct": bool(abs(ratio_pct) <= 5.0),
                    "clinical_ratio_within_0p02": bool(ratio_abs <= 0.02),
                    "volume_fit_success": volume_fit["success"],
                    "sigma_volume_l": volume_fit["sigma_volume_l"],
                    "volume_nrss_norm": volume_fit["volume_nrss_norm"],
                    "volume_nrss_threshold_95": volume_fit["volume_nrss_threshold_95"],
                    "volume_adequate_norm": volume_fit["volume_adequate_norm"],
                    "volume_r2": volume_fit["volume_r2"],
                }
            )
            if len(plot_payloads) < 6:
                plot_payloads.append(
                    {
                        "label": f"SEQN {int(fit_row.SEQN)} m{int(fit_row.SPAMANU)} {peak['peak_class']}",
                        "t": curve["vol_t"],
                        "volume": curve["vol_l"],
                        "volume_fit": volume_fit["pred_volume_l"],
                        "flow_t": curve["fit_t"],
                        "flow": curve["fit_flow"],
                        "flow_pred": flow_pred,
                        "flow_residual": flow_residual,
                        "pef_t": curve["pef_t"],
                    }
                )
        except Exception as exc:
            base["diagnostic_error"] = str(exc)
        rows.append(base)
        if (idx + 1) % 50 == 0 or idx + 1 == len(sample):
            print(f"Gate 3 fatality diagnostic progress: {idx + 1}/{len(sample)} curves")
    meta = {
        "raw_source_note": raw_note,
        "raw_info": raw_info,
        "sample_n": SAMPLE_N,
        "sample_seed": SAMPLE_SEED,
        "eligible_A_non_obstructed_no_fit_error_n": eligible_n,
    }
    return pd.DataFrame(rows), meta, pd.DataFrame(plot_payloads)


def summarize(rows: pd.DataFrame, meta: dict) -> dict:
    valid = rows[rows["diagnostic_error"].fillna("") == ""].copy()
    peak_counts = valid["peak_class"].value_counts(dropna=False).to_dict()
    n_valid = len(valid)
    volume_valid = valid[np.isfinite(valid["volume_nrss_norm"])].copy()
    clinical_valid = valid.dropna(subset=["fev1_pct_error", "fvc_pct_error", "ratio_pct_error"]).copy()
    result = {
        "status": "UNKNOWN",
        "reason": "",
        **meta,
        "fit_domain_current": "predictx/fit.py fits the FLOW-time curve: fit_ombe_lm(t, flow_lps) minimizes np_ombe_flow(params, t) - flow_lps.",
        "steltner_comparability": "Unresolved: local repo search found only internal mentions of Steltner/2.76, and web search did not locate a primary source for whether the benchmark used flow-time or volume-time residuals. Do not claim comparability until the paper is supplied or verified.",
        "n_analyzed": int(len(rows)),
        "n_valid": int(n_valid),
        "diagnostic_error_n": int(len(rows) - n_valid),
        "peak_class_counts": {str(k): int(v) for k, v in peak_counts.items()},
        "peak_class_fractions": {str(k): float(v / n_valid) for k, v in peak_counts.items()} if n_valid else {},
        "dipole_fraction": float((valid["peak_class"] == "dipole").mean()) if n_valid else np.nan,
        "single_signed_fraction": float(valid["peak_class"].isin(["single_signed_undershoot", "single_signed_overshoot"]).mean()) if n_valid else np.nan,
        "single_signed_undershoot_fraction": float((valid["peak_class"] == "single_signed_undershoot").mean()) if n_valid else np.nan,
        "single_signed_overshoot_fraction": float((valid["peak_class"] == "single_signed_overshoot").mean()) if n_valid else np.nan,
        "volume_nrss_valid_n": int(len(volume_valid)),
        "volume_nrss_norm_median": float(np.nanmedian(valid["volume_nrss_norm"])) if n_valid else np.nan,
        "volume_nrss_norm_iqr": [float(x) for x in np.nanpercentile(valid["volume_nrss_norm"], [25, 75])] if n_valid else [np.nan, np.nan],
        "volume_nrss_threshold_median": float(np.nanmedian(valid["volume_nrss_threshold_95"])) if n_valid else np.nan,
        "volume_adequate_n": int(valid["volume_adequate_norm"].sum()) if n_valid else 0,
        "volume_adequate_fraction": float(valid["volume_adequate_norm"].mean()) if n_valid else np.nan,
        "volume_r2_median": float(np.nanmedian(valid["volume_r2"])) if n_valid else np.nan,
        "clinical_valid_n": int(len(clinical_valid)),
        "clinical_all_within_5pct_n": int(clinical_valid["clinical_all_within_5pct"].sum()) if len(clinical_valid) else 0,
        "clinical_all_within_5pct_fraction": float(clinical_valid["clinical_all_within_5pct"].mean()) if len(clinical_valid) else np.nan,
        "clinical_ratio_within_5pct_n": int(clinical_valid["clinical_ratio_within_5pct"].sum()) if len(clinical_valid) else 0,
        "clinical_ratio_within_5pct_fraction": float(clinical_valid["clinical_ratio_within_5pct"].mean()) if len(clinical_valid) else np.nan,
        "clinical_ratio_within_0p02_n": int(clinical_valid["clinical_ratio_within_0p02"].sum()) if len(clinical_valid) else 0,
        "clinical_ratio_within_0p02_fraction": float(clinical_valid["clinical_ratio_within_0p02"].mean()) if len(clinical_valid) else np.nan,
        "fev1_abs_pct_error_median": float(np.nanmedian(np.abs(clinical_valid["fev1_pct_error"]))) if len(clinical_valid) else np.nan,
        "fvc_abs_pct_error_median": float(np.nanmedian(np.abs(clinical_valid["fvc_pct_error"]))) if len(clinical_valid) else np.nan,
        "ratio_abs_pct_error_median": float(np.nanmedian(np.abs(clinical_valid["ratio_pct_error"]))) if len(clinical_valid) else np.nan,
        "ratio_abs_error_median": float(np.nanmedian(clinical_valid["ratio_abs_error"])) if len(clinical_valid) else np.nan,
    }
    if result["dipole_fraction"] >= 0.50:
        status = "FIXABLE_TIMING_SUSPECT"
        reason = "Dipole peak residuals dominate, supporting onset/phase misregistration as the leading fixable cause."
    elif result["single_signed_fraction"] >= 0.50 and result["clinical_all_within_5pct_fraction"] >= 0.85:
        status = "FLOW_LIMITATION_CLINICAL_OK"
        reason = "Peak residuals are mostly single-signed, but volume-fit clinical metrics reproduce parsed truth within 5% for >=85% of sampled curves."
    elif result["single_signed_fraction"] >= 0.50:
        status = "GENUINE_INADEQUACY"
        reason = "Peak residuals are mostly single-signed and volume-fit clinical metrics do not clear the 5% reproduction criterion."
    else:
        status = "MIXED"
        reason = "Peak residual classes are mixed; no single diagnosis dominates."
    result["status"] = status
    result["reason"] = reason
    return result


def draw_examples(plot_payloads: pd.DataFrame, figures_dir: Path) -> None:
    if plot_payloads.empty:
        return
    figures_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(len(plot_payloads), 2, figsize=(14, 2.6 * len(plot_payloads)), constrained_layout=True)
    if len(plot_payloads) == 1:
        axes = np.array([axes])
    for idx, row in enumerate(plot_payloads.itertuples(index=False)):
        ax_flow = axes[idx, 0]
        ax_vol = axes[idx, 1]
        ax_flow.plot(row.flow_t, row.flow, color="#1f5aa6", lw=1.0, label="flow")
        ax_flow.plot(row.flow_t, row.flow_pred, color="#d62728", lw=1.2, label="flow fit")
        ax_flow.plot(row.flow_t, row.flow_residual, color="#111111", lw=0.8, label="residual")
        ax_flow.axvline(row.pef_t, color="#f59e0b", lw=1.0)
        ax_flow.axhline(0.0, color="#999999", lw=0.7)
        ax_flow.set_title(f"{row.label}: flow residual")
        ax_flow.set_ylabel("L/s")
        ax_vol.plot(row.t, row.volume, color="#1f5aa6", lw=1.0, label="volume")
        ax_vol.plot(row.t, row.volume_fit, color="#d62728", lw=1.2, label="volume fit")
        ax_vol.set_title(f"{row.label}: volume fit")
        ax_vol.set_ylabel("L")
    axes[0, 0].legend(loc="upper right")
    axes[0, 1].legend(loc="lower right")
    for ax in axes.ravel():
        ax.set_xlabel("time from t0 (s)")
    fig.suptitle("Gate 3 diagnosis: flow peak residual vs volume-time fit", y=1.01)
    fig.savefig(figures_dir / "gate3_F_diagnosis_examples.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_report(result: dict, rows: pd.DataFrame, results_dir: Path) -> None:
    lines = [
        "# Gate 3 Diagnosis - Flow Peak Fatality Check",
        "",
        f"- status: {result['status']}",
        f"- reason: {result['reason']}",
        f"- sample: {result['sample_n']} A-grade non-obstructed curves selected with fixed seed {result['sample_seed']}",
        f"- eligible A-grade non-obstructed curves without prior fit error: {result['eligible_A_non_obstructed_no_fit_error_n']}",
        f"- valid analyzed curves: {result['n_valid']}/{result['n_analyzed']}",
        "",
        "## Check 1 - Dipole vs Undershoot",
        "",
        "- residual convention: actual flow minus fitted flow.",
        f"- dipole fraction: {result['dipole_fraction']:.3f}",
        f"- single-signed fraction: {result['single_signed_fraction']:.3f}",
        f"- single-signed undershoot fraction: {result['single_signed_undershoot_fraction']:.3f}",
        f"- single-signed overshoot fraction: {result['single_signed_overshoot_fraction']:.3f}",
        "",
        "| peak class | n | fraction |",
        "|---|---:|---:|",
    ]
    for key, n in result["peak_class_counts"].items():
        frac = result["peak_class_fractions"].get(key, np.nan)
        lines.append(f"| {key} | {n} | {frac:.3f} |")
    lines.extend(
        [
            "",
            "## Check 2 - Fit Domain and Clinical Integrals",
            "",
            f"- current fitter domain: {result['fit_domain_current']}",
            "- volume refit: OMBE parameters were re-fit to parsed volume-time curves for the same 200 curves.",
            f"- volume NRSS valid curves: {result['volume_nrss_valid_n']}",
            f"- volume NRSS_norm median: {result['volume_nrss_norm_median']:.3f}",
            f"- volume NRSS_norm IQR: {result['volume_nrss_norm_iqr'][0]:.3f} to {result['volume_nrss_norm_iqr'][1]:.3f}",
            f"- volume chi-square threshold median: {result['volume_nrss_threshold_median']:.3f}",
            f"- volume adequate by corrected NRSS: {result['volume_adequate_n']}/{result['n_valid']} ({100 * result['volume_adequate_fraction']:.1f}%)",
            f"- volume R^2 median: {result['volume_r2_median']:.5f}",
            "",
            "Clinical metrics from volume-fit OMBE vs parsed same-curve volume truth:",
            f"- all FEV1, FVC, and ratio within 5%: {result['clinical_all_within_5pct_n']}/{result['clinical_valid_n']} ({100 * result['clinical_all_within_5pct_fraction']:.1f}%)",
            f"- FEV1/FVC ratio within 5%: {result['clinical_ratio_within_5pct_n']}/{result['clinical_valid_n']} ({100 * result['clinical_ratio_within_5pct_fraction']:.1f}%)",
            f"- FEV1/FVC ratio within 0.02 absolute: {result['clinical_ratio_within_0p02_n']}/{result['clinical_valid_n']} ({100 * result['clinical_ratio_within_0p02_fraction']:.1f}%)",
            f"- median abs FEV1 error: {result['fev1_abs_pct_error_median']:.2f}%",
            f"- median abs FVC error: {result['fvc_abs_pct_error_median']:.2f}%",
            f"- median abs FEV1/FVC percent error: {result['ratio_abs_pct_error_median']:.2f}%",
            f"- median abs FEV1/FVC absolute error: {result['ratio_abs_error_median']:.4f}",
            "",
            "## Check 3 - Steltner Comparability",
            "",
            result["steltner_comparability"],
            "",
            "## Verdict",
            "",
            "- Dipole-dominant means the leading cause is fit registration/timing, not settled physical inadequacy.",
            "- Single-signed with clinical metrics within 5% means the flow-peak miss is clinically non-fatal for FEV1/FVC outputs.",
            "- Single-signed with clinical metrics failing means OMBE needs an augmented onset model.",
            "",
            f"Observed verdict: {result['status']} - {result['reason']}",
            "",
        ]
    )
    results_dir.mkdir(exist_ok=True)
    (results_dir / "gate3_F_diagnosis.md").write_text("\n".join(lines), encoding="utf-8")
    (results_dir / "gate3_F_diagnosis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    rows.to_csv(results_dir / "gate3_F_diagnosis_sample.csv", index=False)


def run(root: Path) -> dict:
    rows, meta, plot_payloads = analyze_sample(root)
    result = summarize(rows, meta)
    draw_examples(plot_payloads, root / "figures")
    write_report(result, rows, root / "results")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    print(f"Gate 3 diagnosis {result['status']}: {result['reason']}")
    print(f"Dipole fraction: {result['dipole_fraction']:.3f}")
    print(f"Clinical all within 5%: {result['clinical_all_within_5pct_fraction']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
