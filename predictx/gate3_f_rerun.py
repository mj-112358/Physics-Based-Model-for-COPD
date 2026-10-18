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

N_PARAMS = 5


def mad_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return float(mad / 0.6745)


def savgol_noise_sigma(flow_lps: np.ndarray) -> float:
    n = len(flow_lps)
    if n < 11:
        return np.nan
    window = min(31, n if n % 2 == 1 else n - 1)
    if window < 11:
        window = 11 if n >= 11 else n | 1
    if window >= n:
        window = n - 1 if n % 2 == 0 else n
    if window < 5:
        return np.nan
    polyorder = min(3, window - 2)
    smooth = savgol_filter(flow_lps, window_length=window, polyorder=polyorder, mode="interp")
    return mad_sigma(flow_lps - smooth)


def curve_segment_and_fit(row: Any, fit_row: Any) -> dict:
    incr = parse_increments(row.SPXRAW)
    fact = float(row.SPAFACT)
    flow = incr * 0.1 * fact
    volume = np.cumsum(incr) * fact
    time = np.arange(len(incr), dtype=float) / HZ
    onset_idx, pef_idx, end_idx = detect_forced_exhale_segment(flow, hz=HZ)
    t0_s = backextrap_t0(time, flow, volume, onset_idx, pef_idx)
    start_idx = max(0, int(np.searchsorted(time, max(t0_s, 0.0), side="left")))
    end_idx = max(end_idx, start_idx + 20)
    end_idx = min(end_idx, len(flow) - 1)

    segment_t = time[start_idx : end_idx + 1] - t0_s
    segment_flow = flow[start_idx : end_idx + 1]
    segment_t = segment_t - segment_t[0]
    params = np.array(
        [
            fit_row.fit_tau_f,
            fit_row.fit_tau_s,
            fit_row.fit_alpha,
            fit_row.fit_fvc_l,
            fit_row.fit_tau_o,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(params)):
        raise ValueError("fit parameters are not finite")
    pred = np_ombe_flow(params, segment_t)
    residual = segment_flow - pred
    sigma_hat = savgol_noise_sigma(segment_flow)
    n = len(segment_t)
    dof = max(n - N_PARAMS, 1)
    rss = float(np.sum(residual**2))
    nrss_norm = float(rss / (dof * sigma_hat**2)) if np.isfinite(sigma_hat) and sigma_hat > 0 else np.nan
    nrss_threshold_95 = float(chi2.ppf(0.95, dof) / dof)

    total_rms = float(np.sqrt(np.mean(residual**2)))
    pef_t = float(time[pef_idx] - time[start_idx])
    masks = {
        "onset_to_pef": segment_t <= pef_t,
        "pef_to_1s": (segment_t > pef_t) & (segment_t <= 1.0),
        "tail_gt_1s": segment_t > 1.0,
    }
    region_rms = {}
    region_frac = {}
    for name, mask in masks.items():
        if np.any(mask):
            rms = float(np.sqrt(np.mean(residual[mask] ** 2)))
            region_rms[name] = rms
            region_frac[name] = float(rms / total_rms) if total_rms > 0 else np.nan
        else:
            region_rms[name] = np.nan
            region_frac[name] = np.nan
    peak_concentrated = (
        region_rms["onset_to_pef"] > 2.0 * region_rms["tail_gt_1s"]
        if np.isfinite(region_rms["onset_to_pef"]) and np.isfinite(region_rms["tail_gt_1s"])
        else np.nan
    )

    return {
        "n_fit_samples": int(n),
        "dof": int(dof),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "pef_segment_t_s": pef_t,
        "sigma_hat_lps": float(sigma_hat),
        "rss": rss,
        "nrss_norm": nrss_norm,
        "nrss_threshold_95": nrss_threshold_95,
        "adequate_norm": bool(nrss_norm <= nrss_threshold_95) if np.isfinite(nrss_norm) else False,
        "total_rms_lps": total_rms,
        "onset_to_pef_rms_lps": region_rms["onset_to_pef"],
        "pef_to_1s_rms_lps": region_rms["pef_to_1s"],
        "tail_gt_1s_rms_lps": region_rms["tail_gt_1s"],
        "onset_to_pef_rms_frac_total": region_frac["onset_to_pef"],
        "pef_to_1s_rms_frac_total": region_frac["pef_to_1s"],
        "tail_gt_1s_rms_frac_total": region_frac["tail_gt_1s"],
        "peak_concentrated": peak_concentrated,
    }


def summarize_group(df: pd.DataFrame, group_col: str) -> list[dict]:
    rows = []
    for key, group in df.groupby(group_col, dropna=False):
        valid_norm = group[np.isfinite(group["nrss_norm"])]
        valid_peak = group[group["peak_concentrated"].notna()]
        rows.append(
            {
                "group": str(key),
                "n": int(len(group)),
                "norm_valid_n": int(len(valid_norm)),
                "adequate_norm_n": int(valid_norm["adequate_norm"].sum()) if len(valid_norm) else 0,
                "adequate_norm_fraction": float(valid_norm["adequate_norm"].mean()) if len(valid_norm) else np.nan,
                "peak_valid_n": int(len(valid_peak)),
                "peak_concentrated_n": int(valid_peak["peak_concentrated"].sum()) if len(valid_peak) else 0,
                "peak_concentrated_fraction": float(valid_peak["peak_concentrated"].mean()) if len(valid_peak) else np.nan,
                "median_onset_to_pef_rms_frac_total": float(np.nanmedian(group["onset_to_pef_rms_frac_total"])),
                "median_pef_to_1s_rms_frac_total": float(np.nanmedian(group["pef_to_1s_rms_frac_total"])),
                "median_tail_gt_1s_rms_frac_total": float(np.nanmedian(group["tail_gt_1s_rms_frac_total"])),
            }
        )
    return rows


def finite_quantiles(values: pd.Series, probs: list[float]) -> list[float]:
    arr = values.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [np.nan for _ in probs]
    return [float(x) for x in np.quantile(arr, probs)]


def select_representatives(df: pd.DataFrame) -> pd.DataFrame:
    finite = df[np.isfinite(df["nrss_norm"])].copy()
    selected = []

    def add(label: str, subset: pd.DataFrame, mode: str = "median") -> None:
        if subset.empty:
            return
        if mode == "worst":
            row = subset.sort_values("nrss_norm", ascending=False).iloc[0].copy()
        elif mode == "best":
            row = subset.sort_values("nrss_norm").iloc[0].copy()
        else:
            med = subset["nrss_norm"].median()
            row = subset.iloc[(subset["nrss_norm"] - med).abs().argsort().iloc[0]].copy()
        row["figure_label"] = label
        selected.append(row)

    add("A healthy", finite[(finite["SPAQEFF"] == "A") & (~finite["obstructed"])])
    add("A obstructed", finite[(finite["SPAQEFF"] == "A") & (finite["obstructed"])])
    add("B healthy", finite[(finite["SPAQEFF"] == "B") & (~finite["obstructed"])])
    add("worst A", finite[finite["SPAQEFF"] == "A"], mode="worst")
    add("best A", finite[finite["SPAQEFF"] == "A"], mode="best")
    add("worst overall", finite, mode="worst")
    return pd.DataFrame(selected)


def make_plot_data(raw_by_key: dict[tuple[float, float], Any], row: Any) -> dict:
    raw_row = raw_by_key[(float(row.SEQN), float(row.SPAMANU))]
    incr = parse_increments(raw_row.SPXRAW)
    fact = float(raw_row.SPAFACT)
    flow = incr * 0.1 * fact
    volume = np.cumsum(incr) * fact
    time = np.arange(len(incr), dtype=float) / HZ
    onset_idx, pef_idx, end_idx = detect_forced_exhale_segment(flow, hz=HZ)
    t0_s = backextrap_t0(time, flow, volume, onset_idx, pef_idx)
    start_idx = max(0, int(np.searchsorted(time, max(t0_s, 0.0), side="left")))
    end_idx = max(end_idx, start_idx + 20)
    end_idx = min(end_idx, len(flow) - 1)
    segment_t = time[start_idx : end_idx + 1] - t0_s
    segment_flow = flow[start_idx : end_idx + 1]
    segment_t = segment_t - segment_t[0]
    params = np.array([row.fit_tau_f, row.fit_tau_s, row.fit_alpha, row.fit_fvc_l, row.fit_tau_o], dtype=float)
    pred = np_ombe_flow(params, segment_t)
    return {
        "t": segment_t,
        "flow": segment_flow,
        "pred": pred,
        "residual": segment_flow - pred,
        "pef_t": float(time[pef_idx] - time[start_idx]),
    }


def draw_residual_shape_figure(df: pd.DataFrame, raw: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(exist_ok=True)
    reps = select_representatives(df)
    if reps.empty:
        return
    raw_pre = raw[raw["SPATTYPE"] == "Pre"]
    raw_by_key = {(float(row.SEQN), float(row.SPAMANU)): row for row in raw_pre.itertuples(index=False)}

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    for ax, row in zip(axes.ravel(), reps.itertuples(index=False)):
        try:
            data = make_plot_data(raw_by_key, row)
        except Exception as exc:
            ax.set_title(f"{row.figure_label}: plot failed {exc}")
            ax.axis("off")
            continue
        t = data["t"]
        pef_t = data["pef_t"]
        ax.axvspan(0.0, pef_t, color="#fee8c8", alpha=0.65, label="onset-PEF")
        ax.axvspan(pef_t, 1.0, color="#dbeafe", alpha=0.45, label="PEF-1s")
        ax.axvspan(1.0, float(t[-1]), color="#dcfce7", alpha=0.35, label="tail")
        ax.plot(t, data["flow"], color="#1f5aa6", lw=1.2, label="flow")
        ax.plot(t, data["pred"], color="#d62728", lw=1.4, label="OMBE")
        ax.plot(t, data["residual"], color="#111111", lw=0.8, alpha=0.8, label="residual")
        ax.axhline(0.0, color="#777777", lw=0.8)
        ax.set_title(
            f"{row.figure_label}: {row.SPAQEFF}, {'obs' if row.obstructed else 'non-obs'}, "
            f"NRSS {row.nrss_norm:.2f}, peak {row.peak_concentrated}"
        )
        ax.set_xlabel("time from t0 (s)")
        ax.set_ylabel("L/s")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6)
    fig.suptitle("Gate 3 cycle F residual shape: curve, OMBE fit, residual, and regions", y=1.02)
    fig.savefig(figures_dir / "gate3_F_residual_shape.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_report(result: dict, rows: pd.DataFrame, results_dir: Path) -> None:
    lines = [
        "# Gate 3 Re-Run - Cycle F Corrected NRSS + Residual Shape",
        "",
        f"- status: {result['status']}",
        f"- reason: {result['reason']}",
        "- scope: F only; E and G are BLOCKED. Obstructed-subgroup numbers are provisional pending E+G.",
        "- correction: prior `NRSS<=2.76` used a synthetic fixed sigma and is not comparable to real curves with unknown per-curve noise.",
        "- discipline: this is a noise-normalization correctness fix, not a softer metric chosen after seeing the prior fail.",
        "",
        "## Per-Curve Noise",
        "",
        "- sigma_hat estimator: MAD of flow minus Savitzky-Golay smooth, divided by 0.6745.",
        f"- sigma_hat valid curves: {result['sigma_hat_valid_n']}/{result['n_curves']}",
        f"- sigma_hat median: {result['sigma_hat_quantiles_lps']['median']:.4f} L/s",
        f"- sigma_hat IQR: {result['sigma_hat_quantiles_lps']['p25']:.4f} to {result['sigma_hat_quantiles_lps']['p75']:.4f} L/s",
        f"- sigma_hat p05-p95: {result['sigma_hat_quantiles_lps']['p05']:.4f} to {result['sigma_hat_quantiles_lps']['p95']:.4f} L/s",
        "",
        "## Properly Normalized NRSS",
        "",
        "- formula: `NRSS_norm = RSS / (dof * sigma_hat^2)`, where `dof = n_fit_samples - 5`.",
        "- threshold: curve-specific `chi2.ppf(0.95, dof) / dof`; under a correct fit this normalized chi-square is approximately 1.",
        f"- NRSS valid curves: {result['nrss_valid_n']}/{result['n_curves']}",
        f"- fit-error curves: {result['fit_error_n']}",
        f"- NRSS_norm median: {result['nrss_norm_quantiles']['median']:.3f}",
        f"- NRSS_norm IQR: {result['nrss_norm_quantiles']['p25']:.3f} to {result['nrss_norm_quantiles']['p75']:.3f}",
        f"- threshold median: {result['nrss_threshold_quantiles']['median']:.3f}",
        f"- adequate by corrected NRSS: {result['adequate_norm_n']}/{result['n_curves']} ({100 * result['adequate_norm_fraction_all']:.1f}% of all curves; {100 * result['adequate_norm_fraction_valid']:.1f}% of NRSS-valid curves)",
        "",
        "## Residual Shape",
        "",
        f"- shape-valid curves: {result['shape_valid_n']}/{result['n_curves']}",
        f"- peak-concentrated curves: {result['peak_concentrated_n']}/{result['shape_valid_n']} ({100 * result['peak_concentrated_fraction']:.1f}% of shape-valid curves)",
        f"- median onset-to-PEF RMS / total RMS: {result['region_fraction_medians']['onset_to_pef']:.3f}",
        f"- median PEF-to-1s RMS / total RMS: {result['region_fraction_medians']['pef_to_1s']:.3f}",
        f"- median tail >1s RMS / total RMS: {result['region_fraction_medians']['tail_gt_1s']:.3f}",
        "",
        "## Stratified by SPAQEFF",
        "",
        "| SPAQEFF | n | NRSS-valid | adequate corrected NRSS | adequate fraction | peak-valid | peak-concentrated | peak fraction | onset/total | PEF-1s/total | tail/total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["by_spaqeff"]:
        lines.append(
            f"| {row['group']} | {row['n']} | {row['norm_valid_n']} | {row['adequate_norm_n']} | "
            f"{row['adequate_norm_fraction']:.3f} | {row['peak_valid_n']} | {row['peak_concentrated_n']} | "
            f"{row['peak_concentrated_fraction']:.3f} | {row['median_onset_to_pef_rms_frac_total']:.3f} | "
            f"{row['median_pef_to_1s_rms_frac_total']:.3f} | {row['median_tail_gt_1s_rms_frac_total']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Stratified by Obstruction - F Only Provisional Pending E+G",
            "",
            "| obstruction group | n | NRSS-valid | adequate corrected NRSS | adequate fraction | peak-valid | peak-concentrated | peak fraction | onset/total | PEF-1s/total | tail/total |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["by_obstruction"]:
        lines.append(
            f"| {row['group']} | {row['n']} | {row['norm_valid_n']} | {row['adequate_norm_n']} | "
            f"{row['adequate_norm_fraction']:.3f} | {row['peak_valid_n']} | {row['peak_concentrated_n']} | "
            f"{row['peak_concentrated_fraction']:.3f} | {row['median_onset_to_pef_rms_frac_total']:.3f} | "
            f"{row['median_pef_to_1s_rms_frac_total']:.3f} | {row['median_tail_gt_1s_rms_frac_total']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Key Inversion Check",
            "",
            f"- A-grade corrected NRSS adequate fraction: {result['inversion_check']['A_adequate_fraction']:.3f}",
            f"- B-grade corrected NRSS adequate fraction: {result['inversion_check']['B_adequate_fraction']:.3f}",
            f"- non-obstructed corrected NRSS adequate fraction: {result['inversion_check']['non_obstructed_adequate_fraction']:.3f}",
            f"- obstructed corrected NRSS adequate fraction: {result['inversion_check']['obstructed_adequate_fraction']:.3f}",
            f"- A-grade non-obstructed peak-concentrated fraction: {result['inversion_check']['A_non_obstructed_peak_fraction']:.3f}",
            f"- inversion resolution: {result['inversion_check']['interpretation']}",
            "",
            "## Verdict Logic",
            "",
            result["verdict_logic"],
            "",
        ]
    )
    results_dir.mkdir(exist_ok=True)
    (results_dir / "gate3_F_ombe_rerun.md").write_text("\n".join(lines), encoding="utf-8")
    (results_dir / "gate3_F_ombe_rerun.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    rows.to_csv(results_dir / "gate3_F_residual_shape.csv", index=False)


def run(root: Path) -> dict:
    results_dir = root / "results"
    figures_dir = root / "figures"
    fits_path = results_dir / "gate3_F_curve_fits.csv"
    if not fits_path.exists():
        raise RuntimeError(f"Missing prior Gate 3 fit table: {fits_path}")
    fits = pd.read_csv(fits_path)
    fits["fit_error"] = fits["fit_error"].fillna("")

    raw_path, raw_note = find_or_unzip_raw(root)
    raw, raw_info = read_raw(raw_path)
    raw_pre = raw[raw["SPATTYPE"] == "Pre"]
    raw_by_key = {(float(row.SEQN), float(row.SPAMANU)): row for row in raw_pre.itertuples(index=False)}

    diagnostic_rows = []
    for idx, fit_row in enumerate(fits.itertuples(index=False)):
        base = fit_row._asdict()
        key = (float(fit_row.SEQN), float(fit_row.SPAMANU))
        if fit_row.fit_error:
            base.update({"diagnostic_error": f"prior fit error: {fit_row.fit_error}"})
        elif key not in raw_by_key:
            base.update({"diagnostic_error": "raw curve not found"})
        else:
            try:
                base.update(curve_segment_and_fit(raw_by_key[key], fit_row))
                base["diagnostic_error"] = ""
            except Exception as exc:
                base.update({"diagnostic_error": str(exc)})
        diagnostic_rows.append(base)
        if (idx + 1) % 5000 == 0 or idx + 1 == len(fits):
            print(f"Gate 3 rerun diagnostic progress: {idx + 1}/{len(fits)} curves")
    rows = pd.DataFrame(diagnostic_rows)
    for col in ("nrss_norm", "nrss_threshold_95", "sigma_hat_lps", "onset_to_pef_rms_frac_total", "pef_to_1s_rms_frac_total", "tail_gt_1s_rms_frac_total"):
        if col not in rows:
            rows[col] = np.nan
    if "adequate_norm" not in rows:
        rows["adequate_norm"] = False
    if "peak_concentrated" not in rows:
        rows["peak_concentrated"] = np.nan

    nrss_valid = rows[np.isfinite(rows["nrss_norm"])]
    shape_valid = rows[rows["peak_concentrated"].notna()]
    sigma_q = finite_quantiles(rows["sigma_hat_lps"], [0.05, 0.25, 0.50, 0.75, 0.95])
    nrss_q = finite_quantiles(rows["nrss_norm"], [0.25, 0.50, 0.75])
    thresh_q = finite_quantiles(rows["nrss_threshold_95"], [0.25, 0.50, 0.75])
    adequate_norm_n = int(rows["adequate_norm"].fillna(False).sum())

    by_spaqeff = summarize_group(rows, "SPAQEFF")
    by_obstruction = summarize_group(rows, "obstruction_group")

    def group_fraction(group_col: str, group: str, metric: str) -> float:
        subset = rows[rows[group_col] == group]
        if subset.empty:
            return np.nan
        if metric == "adequate":
            subset = subset[np.isfinite(subset["nrss_norm"])]
            return float(subset["adequate_norm"].mean()) if len(subset) else np.nan
        if metric == "peak":
            subset = subset[subset["peak_concentrated"].notna()]
            return float(subset["peak_concentrated"].mean()) if len(subset) else np.nan
        raise ValueError(metric)

    a_non_obs = rows[(rows["SPAQEFF"] == "A") & (~rows["obstructed"])]
    a_non_obs_peak = a_non_obs[a_non_obs["peak_concentrated"].notna()]
    a_non_obs_peak_fraction = float(a_non_obs_peak["peak_concentrated"].mean()) if len(a_non_obs_peak) else np.nan

    adequate_fraction_all = adequate_norm_n / len(rows) if len(rows) else 0.0
    peak_fraction = float(shape_valid["peak_concentrated"].mean()) if len(shape_valid) else np.nan
    region_medians = {
        "onset_to_pef": float(np.nanmedian(rows["onset_to_pef_rms_frac_total"])),
        "pef_to_1s": float(np.nanmedian(rows["pef_to_1s_rms_frac_total"])),
        "tail_gt_1s": float(np.nanmedian(rows["tail_gt_1s_rms_frac_total"])),
    }
    corrected_pass = adequate_fraction_all >= 0.85
    clean_peak_problem = bool(np.isfinite(a_non_obs_peak_fraction) and a_non_obs_peak_fraction >= 0.50)
    if corrected_pass and not clean_peak_problem:
        status = "PASS"
        reason = "Corrected NRSS clears 85% and clean A-grade non-obstructed residuals are not mostly peak-concentrated."
        verdict_logic = "Corrected NRSS clears the adequacy bar and the clean-healthy residual-shape check does not show dominant peak concentration."
    elif corrected_pass and clean_peak_problem:
        status = "LIMITATION"
        reason = "Corrected NRSS clears 85%, but clean A-grade non-obstructed curves show peak-concentrated residuals."
        verdict_logic = "Corrected NRSS passes in aggregate, but residual shape identifies a fast-rise limitation in clean non-obstructed curves."
    else:
        status = "FAIL"
        reason = "Corrected noise-normalized NRSS still fails the 85% adequacy bar."
        verdict_logic = "Corrected NRSS still fails under per-curve noise normalization, so this is a genuine model-adequacy problem under the specified gate."

    inversion_interpretation = (
        "A-grade non-obstructed curves are peak-concentrated, supporting a fast-rise/onset limitation."
        if clean_peak_problem
        else "A-grade non-obstructed curves are not mostly peak-concentrated under the 2x-tail criterion."
    )

    result = {
        "status": status,
        "reason": reason,
        "raw_source_note": raw_note,
        "raw_info": raw_info,
        "n_curves": int(len(rows)),
        "fit_error_n": int((rows["fit_error"].fillna("") != "").sum()),
        "diagnostic_error_n": int((rows["diagnostic_error"].fillna("") != "").sum()),
        "sigma_hat_valid_n": int(np.isfinite(rows["sigma_hat_lps"]).sum()),
        "sigma_hat_quantiles_lps": {
            "p05": sigma_q[0],
            "p25": sigma_q[1],
            "median": sigma_q[2],
            "p75": sigma_q[3],
            "p95": sigma_q[4],
        },
        "nrss_valid_n": int(len(nrss_valid)),
        "nrss_norm_quantiles": {"p25": nrss_q[0], "median": nrss_q[1], "p75": nrss_q[2]},
        "nrss_threshold_quantiles": {"p25": thresh_q[0], "median": thresh_q[1], "p75": thresh_q[2]},
        "adequate_norm_n": adequate_norm_n,
        "adequate_norm_fraction_all": float(adequate_fraction_all),
        "adequate_norm_fraction_valid": float(nrss_valid["adequate_norm"].mean()) if len(nrss_valid) else np.nan,
        "shape_valid_n": int(len(shape_valid)),
        "peak_concentrated_n": int(shape_valid["peak_concentrated"].sum()) if len(shape_valid) else 0,
        "peak_concentrated_fraction": peak_fraction,
        "region_fraction_medians": region_medians,
        "by_spaqeff": by_spaqeff,
        "by_obstruction": by_obstruction,
        "inversion_check": {
            "A_adequate_fraction": group_fraction("SPAQEFF", "A", "adequate"),
            "B_adequate_fraction": group_fraction("SPAQEFF", "B", "adequate"),
            "non_obstructed_adequate_fraction": group_fraction("obstruction_group", "non_obstructed_derived_ratio_ge_0.70", "adequate"),
            "obstructed_adequate_fraction": group_fraction("obstruction_group", "obstructed_derived_ratio_lt_0.70", "adequate"),
            "A_non_obstructed_peak_fraction": a_non_obs_peak_fraction,
            "interpretation": inversion_interpretation,
        },
        "verdict_logic": verdict_logic,
    }
    draw_residual_shape_figure(rows, raw, figures_dir)
    write_report(result, rows, results_dir)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    print(f"Gate 3 rerun {result['status']}: {result['reason']}")
    print(
        f"Corrected NRSS adequate: {result['adequate_norm_n']}/{result['n_curves']} "
        f"({100 * result['adequate_norm_fraction_all']:.1f}%)"
    )
    print(
        f"Peak-concentrated: {result['peak_concentrated_n']}/{result['shape_valid_n']} "
        f"({100 * result['peak_concentrated_fraction']:.1f}%)"
    )
    return 0 if result["status"] in {"PASS", "LIMITATION"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
