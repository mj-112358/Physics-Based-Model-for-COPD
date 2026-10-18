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
from scipy.stats import chi2

from .fit import fit_ombe_registered_lm, np_ombe_registered_flow
from .gate3_diagnosis import (
    N_PARAMS,
    SAMPLE_N,
    SAMPLE_SEED,
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

BASELINE_DIPOLE_FRACTION = 0.845
BASELINE_ALL_WITHIN_5PCT = 0.840
BASELINE_RATIO_WITHIN_5PCT = 0.855
BASELINE_RATIO_WITHIN_0P02 = 0.675
MATERIAL_IMPROVEMENT_POINTS = 0.05


def bool_from_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def predicted_volume_from_registered_flow(params: np.ndarray, shift_s: float, t: np.ndarray) -> np.ndarray:
    flow = np_ombe_registered_flow(params, t, shift_s)
    return cumulative_trapezoid(flow, t)


def analyze_registered_curve(fit_row: Any, raw_row: Any) -> tuple[dict, dict]:
    curve = extract_curve(raw_row)
    sigma_flow = savgol_sigma(curve["fit_flow"])
    fit = fit_ombe_registered_lm(
        curve["fit_t"],
        curve["fit_flow"],
        sigma_lps=sigma_flow if np.isfinite(sigma_flow) and sigma_flow > 0 else 0.10,
    )
    params = np.array([fit.params["tau_f"], fit.params["tau_s"], fit.params["alpha"], fit.params["fvc_l"], fit.params["tau_o"]])
    pred_flow = np_ombe_registered_flow(params, curve["fit_t"], fit.time_shift_s)
    residual = curve["fit_flow"] - pred_flow
    peak = classify_peak_residual(curve["fit_t"], residual, curve["pef_t"], sigma_flow)
    dof = max(len(curve["fit_t"]) - (N_PARAMS + 1), 1)
    rss = float(np.sum(residual**2))
    nrss_norm = float(rss / (dof * sigma_flow**2)) if np.isfinite(sigma_flow) and sigma_flow > 0 else np.nan
    threshold = float(chi2.ppf(0.95, dof) / dof)

    pred_volume = predicted_volume_from_registered_flow(params, fit.time_shift_s, curve["vol_t"])
    pred_fev1, pred_fvc, pred_ratio = clinical_metrics_from_volume(curve["vol_t"], pred_volume)
    truth_fev1, truth_fvc, truth_ratio = clinical_metrics_from_volume(curve["vol_t"], curve["vol_l"])
    fev1_pct = pct_error(pred_fev1, truth_fev1)
    fvc_pct = pct_error(pred_fvc, truth_fvc)
    ratio_pct = pct_error(pred_ratio, truth_ratio)
    ratio_abs = float(abs(pred_ratio - truth_ratio)) if np.isfinite(pred_ratio) and np.isfinite(truth_ratio) else np.nan

    row = fit_row._asdict()
    row.update(
        {
            "registered_error": "",
            "registered_success": fit.success,
            "registered_shift_s": fit.time_shift_s,
            "registered_shift_samples": fit.time_shift_s * 100.0,
            "registered_tau_f": fit.params["tau_f"],
            "registered_tau_s": fit.params["tau_s"],
            "registered_alpha": fit.params["alpha"],
            "registered_fvc_parameter_l": fit.params["fvc_l"],
            "registered_tau_o": fit.params["tau_o"],
            "sigma_flow_lps": sigma_flow,
            "registered_flow_nrss_norm": nrss_norm,
            "registered_flow_threshold_95": threshold,
            "registered_flow_adequate": bool(nrss_norm <= threshold) if np.isfinite(nrss_norm) else False,
            "registered_flow_r2": fit.r2,
            **peak,
            "truth_fev1_l": truth_fev1,
            "truth_fvc_l": truth_fvc,
            "truth_ratio": truth_ratio,
            "registered_fev1_l": pred_fev1,
            "registered_fvc_l": pred_fvc,
            "registered_ratio": pred_ratio,
            "registered_fev1_pct_error": fev1_pct,
            "registered_fvc_pct_error": fvc_pct,
            "registered_ratio_pct_error": ratio_pct,
            "registered_ratio_abs_error": ratio_abs,
            "registered_all_within_5pct": bool(abs(fev1_pct) <= 5.0 and abs(fvc_pct) <= 5.0 and abs(ratio_pct) <= 5.0),
            "registered_ratio_within_5pct": bool(abs(ratio_pct) <= 5.0),
            "registered_ratio_within_0p02": bool(ratio_abs <= 0.02),
        }
    )
    plot = {
        "label": f"SEQN {int(fit_row.SEQN)} m{int(fit_row.SPAMANU)} shift {fit.time_shift_s * 100.0:.1f} samples",
        "t": curve["fit_t"],
        "flow": curve["fit_flow"],
        "pred_flow": pred_flow,
        "residual": residual,
        "volume_t": curve["vol_t"],
        "volume": curve["vol_l"],
        "pred_volume": pred_volume,
        "pef_t": curve["pef_t"],
    }
    return row, plot


def summarize_registered(rows: pd.DataFrame, prefix: str = "") -> dict:
    valid = rows[rows["registered_error"].fillna("") == ""].copy()
    clinical_valid = valid.dropna(
        subset=["registered_fev1_pct_error", "registered_fvc_pct_error", "registered_ratio_pct_error"]
    )
    peak_counts = valid["peak_class"].value_counts(dropna=False).to_dict()
    result = {
        f"{prefix}n": int(len(rows)),
        f"{prefix}valid_n": int(len(valid)),
        f"{prefix}error_n": int(len(rows) - len(valid)),
        f"{prefix}flow_nrss_valid_n": int(np.isfinite(valid["registered_flow_nrss_norm"]).sum()) if len(valid) else 0,
        f"{prefix}flow_nrss_median": float(np.nanmedian(valid["registered_flow_nrss_norm"])) if len(valid) else np.nan,
        f"{prefix}flow_nrss_iqr": [float(x) for x in np.nanpercentile(valid["registered_flow_nrss_norm"], [25, 75])] if len(valid) else [np.nan, np.nan],
        f"{prefix}flow_threshold_median": float(np.nanmedian(valid["registered_flow_threshold_95"])) if len(valid) else np.nan,
        f"{prefix}flow_adequate_n": int(valid["registered_flow_adequate"].sum()) if len(valid) else 0,
        f"{prefix}flow_adequate_fraction": float(valid["registered_flow_adequate"].mean()) if len(valid) else np.nan,
        f"{prefix}dipole_n": int((valid["peak_class"] == "dipole").sum()) if len(valid) else 0,
        f"{prefix}dipole_fraction": float((valid["peak_class"] == "dipole").mean()) if len(valid) else np.nan,
        f"{prefix}single_signed_n": int(valid["peak_class"].isin(["single_signed_undershoot", "single_signed_overshoot"]).sum()) if len(valid) else 0,
        f"{prefix}single_signed_fraction": float(valid["peak_class"].isin(["single_signed_undershoot", "single_signed_overshoot"]).mean()) if len(valid) else np.nan,
        f"{prefix}peak_class_counts": {str(k): int(v) for k, v in peak_counts.items()},
        f"{prefix}shift_median_samples": float(np.nanmedian(valid["registered_shift_samples"])) if len(valid) else np.nan,
        f"{prefix}shift_iqr_samples": [float(x) for x in np.nanpercentile(valid["registered_shift_samples"], [25, 75])] if len(valid) else [np.nan, np.nan],
        f"{prefix}clinical_valid_n": int(len(clinical_valid)),
        f"{prefix}all_within_5pct_n": int(clinical_valid["registered_all_within_5pct"].sum()) if len(clinical_valid) else 0,
        f"{prefix}all_within_5pct_fraction": float(clinical_valid["registered_all_within_5pct"].mean()) if len(clinical_valid) else np.nan,
        f"{prefix}ratio_within_5pct_n": int(clinical_valid["registered_ratio_within_5pct"].sum()) if len(clinical_valid) else 0,
        f"{prefix}ratio_within_5pct_fraction": float(clinical_valid["registered_ratio_within_5pct"].mean()) if len(clinical_valid) else np.nan,
        f"{prefix}ratio_within_0p02_n": int(clinical_valid["registered_ratio_within_0p02"].sum()) if len(clinical_valid) else 0,
        f"{prefix}ratio_within_0p02_fraction": float(clinical_valid["registered_ratio_within_0p02"].mean()) if len(clinical_valid) else np.nan,
        f"{prefix}fev1_abs_pct_error_median": float(np.nanmedian(np.abs(clinical_valid["registered_fev1_pct_error"]))) if len(clinical_valid) else np.nan,
        f"{prefix}fvc_abs_pct_error_median": float(np.nanmedian(np.abs(clinical_valid["registered_fvc_pct_error"]))) if len(clinical_valid) else np.nan,
        f"{prefix}ratio_abs_pct_error_median": float(np.nanmedian(np.abs(clinical_valid["registered_ratio_pct_error"]))) if len(clinical_valid) else np.nan,
        f"{prefix}ratio_abs_error_median": float(np.nanmedian(clinical_valid["registered_ratio_abs_error"])) if len(clinical_valid) else np.nan,
    }
    return result


def summarize_by_group(rows: pd.DataFrame, group_col: str) -> list[dict]:
    out = []
    for key, group in rows.groupby(group_col, dropna=False):
        summary = summarize_registered(group)
        out.append(
            {
                "group": str(key),
                "n": summary["n"],
                "valid_n": summary["valid_n"],
                "all_within_5pct_fraction": summary["all_within_5pct_fraction"],
                "ratio_within_0p02_fraction": summary["ratio_within_0p02_fraction"],
                "flow_adequate_fraction": summary["flow_adequate_fraction"],
                "dipole_fraction": summary["dipole_fraction"],
                "median_ratio_abs_error": summary["ratio_abs_error_median"],
            }
        )
    return out


def draw_registered_examples(rows: pd.DataFrame, plots: list[dict], figures_dir: Path, name: str) -> None:
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
        ax_flow.plot(plot["t"], plot["pred_flow"], color="#d62728", lw=1.2, label="registered fit")
        ax_flow.plot(plot["t"], plot["residual"], color="#111111", lw=0.8, label="residual")
        ax_flow.axvline(plot["pef_t"], color="#f59e0b", lw=0.9)
        ax_flow.axhline(0.0, color="#999999", lw=0.7)
        ax_flow.set_title(f"{plot['label']}: flow")
        ax_flow.set_ylabel("L/s")
        ax_vol.plot(plot["volume_t"], plot["volume"], color="#1f5aa6", lw=1.0, label="parsed volume")
        ax_vol.plot(plot["volume_t"], plot["pred_volume"], color="#d62728", lw=1.2, label="registered OMBE volume")
        ax_vol.set_title(f"{plot['label']}: clinical volume")
        ax_vol.set_ylabel("L")
    axes[0, 0].legend(loc="upper right")
    axes[0, 1].legend(loc="lower right")
    for ax in axes.ravel():
        ax.set_xlabel("time from t0 (s)")
    fig.suptitle("Gate 3 registered-fit examples", y=1.01)
    fig.savefig(figures_dir / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_registered_subset(subset: pd.DataFrame, raw_by_key: dict[tuple[float, float], Any], progress_label: str) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    plots = []
    for idx, fit_row in enumerate(subset.itertuples(index=False)):
        base = fit_row._asdict()
        key = (float(fit_row.SEQN), float(fit_row.SPAMANU))
        if key not in raw_by_key:
            base["registered_error"] = "raw curve not found"
            rows.append(base)
            continue
        try:
            row, plot = analyze_registered_curve(fit_row, raw_by_key[key])
            rows.append(row)
            if len(plots) < 6:
                plots.append(plot)
        except Exception as exc:
            base["registered_error"] = str(exc)
            rows.append(base)
        if (idx + 1) % 1000 == 0 or idx + 1 == len(subset):
            print(f"{progress_label}: {idx + 1}/{len(subset)} curves")
    return pd.DataFrame(rows), plots


def fixed_from_sample(summary: dict) -> tuple[bool, str]:
    dipole_ok = summary["sample_dipole_fraction"] < 0.30
    all_ok = summary["sample_all_within_5pct_fraction"] > 0.90
    ratio_delta = summary["sample_ratio_within_0p02_fraction"] - BASELINE_RATIO_WITHIN_0P02
    ratio_ok = ratio_delta >= MATERIAL_IMPROVEMENT_POINTS
    fixed = bool(dipole_ok and all_ok and ratio_ok)
    reason = (
        f"dipoles<30%={dipole_ok}; all-three-within-5%>90%={all_ok}; "
        f"FEV1/FVC within 0.02 improved by >=5 points={ratio_ok} (delta={100 * ratio_delta:.1f} points)"
    )
    return fixed, reason


def write_report(result: dict, sample_rows: pd.DataFrame, full_rows: pd.DataFrame | None, results_dir: Path) -> None:
    lines = [
        "# Gate 3 Registered Onset Fix",
        "",
        f"- status: {result['status']}",
        f"- reason: {result['reason']}",
        "- method: jointly optimized a bounded sub-sample onset time shift with the five OMBE parameters.",
        "- shift convention: positive shift delays the model relative to data; prediction at data time t is unshifted OMBE at t-shift.",
        "- max shift bound: +/-0.08 s.",
        "- criterion change discipline: flow-onset weakness remains documented; clinical Gate 3 uses volume-derived outputs because FEV1, FVC, and FEV1/FVC are volume integrals.",
        "",
        "## Step 2 - 200-Curve A-Grade Non-Obstructed Sample",
        "",
        f"- baseline dipole fraction: {BASELINE_DIPOLE_FRACTION:.3f}",
        f"- registered dipole fraction: {result['sample_dipole_fraction']:.3f}",
        f"- registered flow NRSS_norm median: {result['sample_flow_nrss_median']:.3f}",
        f"- registered flow adequate fraction: {result['sample_flow_adequate_fraction']:.3f}",
        f"- registered shift median: {result['sample_shift_median_samples']:.2f} samples",
        f"- registered shift IQR: {result['sample_shift_iqr_samples'][0]:.2f} to {result['sample_shift_iqr_samples'][1]:.2f} samples",
        "",
        "| clinical metric | baseline | registered | delta points |",
        "|---|---:|---:|---:|",
        f"| all FEV1/FVC/FVC within 5% | {BASELINE_ALL_WITHIN_5PCT:.3f} | {result['sample_all_within_5pct_fraction']:.3f} | {100 * (result['sample_all_within_5pct_fraction'] - BASELINE_ALL_WITHIN_5PCT):.1f} |",
        f"| FEV1/FVC within 5% | {BASELINE_RATIO_WITHIN_5PCT:.3f} | {result['sample_ratio_within_5pct_fraction']:.3f} | {100 * (result['sample_ratio_within_5pct_fraction'] - BASELINE_RATIO_WITHIN_5PCT):.1f} |",
        f"| FEV1/FVC within 0.02 absolute | {BASELINE_RATIO_WITHIN_0P02:.3f} | {result['sample_ratio_within_0p02_fraction']:.3f} | {100 * (result['sample_ratio_within_0p02_fraction'] - BASELINE_RATIO_WITHIN_0P02):.1f} |",
        "",
        f"- fixed criteria: {result['sample_fixed_reason']}",
        "",
    ]
    if result["full_ran"]:
        lines.extend(
            [
                "## Step 3 - Full Cycle F Registered Gate",
                "",
                f"- curves: {result['full_n']}",
                f"- valid curves: {result['full_valid_n']}",
                f"- all FEV1/FVC/FVC within 5%: {result['full_all_within_5pct_n']}/{result['full_clinical_valid_n']} ({100 * result['full_all_within_5pct_fraction']:.1f}%)",
                f"- FEV1/FVC within 0.02 absolute: {result['full_ratio_within_0p02_n']}/{result['full_clinical_valid_n']} ({100 * result['full_ratio_within_0p02_fraction']:.1f}%)",
                f"- flow NRSS adequate fraction, secondary limitation metric: {result['full_flow_adequate_fraction']:.3f}",
                f"- dipole fraction after registration: {result['full_dipole_fraction']:.3f}",
                "",
                "### Stratified by SPAQEFF",
                "",
                "| SPAQEFF | n | valid | all within 5% | ratio within 0.02 | flow adequate | dipole | median ratio abs err |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in result["full_by_spaqeff"]:
            lines.append(
                f"| {row['group']} | {row['n']} | {row['valid_n']} | {row['all_within_5pct_fraction']:.3f} | "
                f"{row['ratio_within_0p02_fraction']:.3f} | {row['flow_adequate_fraction']:.3f} | "
                f"{row['dipole_fraction']:.3f} | {row['median_ratio_abs_error']:.4f} |"
            )
        lines.extend(
            [
                "",
                "### Stratified by Obstruction - F Only Provisional Pending E+G",
                "",
                "| obstruction group | n | valid | all within 5% | ratio within 0.02 | flow adequate | dipole | median ratio abs err |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in result["full_by_obstruction"]:
            lines.append(
                f"| {row['group']} | {row['n']} | {row['valid_n']} | {row['all_within_5pct_fraction']:.3f} | "
                f"{row['ratio_within_0p02_fraction']:.3f} | {row['flow_adequate_fraction']:.3f} | "
                f"{row['dipole_fraction']:.3f} | {row['median_ratio_abs_error']:.4f} |"
            )
    else:
        lines.extend(
            [
                "## Step 3 - Full Cycle F Registered Gate",
                "",
                "- NOT RUN: sample did not clear the fixed criteria, so the script did not spend a full 31,851-curve rerun.",
                "",
            ]
        )
    lines.extend(
        [
            "## Step 4 - Residual Clinical Failures",
            "",
        ]
    )
    if result["full_ran"]:
        lines.extend(
            [
                f"- full curves failing FEV1/FVC within 0.02: {result['full_ratio_0p02_failure_n']}",
                f"- failed obstructed curves: {result['full_ratio_0p02_failed_obstructed_n']}",
                f"- failed truncated <6s curves: {result['full_ratio_0p02_failed_truncated_lt6s_n']}",
                f"- failed median registered shift: {result['full_ratio_0p02_failed_shift_median_samples']:.2f} samples",
                "",
            ]
        )
    else:
        valid_sample = sample_rows[sample_rows["registered_error"].fillna("") == ""].copy()
        failures = valid_sample[~valid_sample["registered_ratio_within_0p02"].fillna(False).astype(bool)]
        lines.extend(
            [
                f"- sample curves failing FEV1/FVC within 0.02: {len(failures)}",
                f"- sample failed median registered shift: {float(np.nanmedian(failures['registered_shift_samples'])) if len(failures) else np.nan:.2f} samples",
                "",
            ]
        )
    results_dir.mkdir(exist_ok=True)
    (results_dir / "gate3_F_registered_fix.md").write_text("\n".join(lines), encoding="utf-8")
    (results_dir / "gate3_F_registered_fix.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    sample_rows.to_csv(results_dir / "gate3_F_registered_sample.csv", index=False)
    if full_rows is not None:
        full_rows.to_csv(results_dir / "gate3_F_registered_full.csv", index=False)


def run(root: Path) -> dict:
    fits = pd.read_csv(root / "results" / "gate3_F_curve_fits.csv")
    fits["obstructed"] = coerce_bool(fits["obstructed"])
    fits["fit_error"] = fits["fit_error"].fillna("")
    sample = select_sample(fits)

    raw_path, raw_note = find_or_unzip_raw(root)
    raw, raw_info = read_raw(raw_path)
    raw_pre = raw[raw["SPATTYPE"] == "Pre"]
    raw_by_key = {(float(row.SEQN), float(row.SPAMANU)): row for row in raw_pre.itertuples(index=False)}

    sample_rows, sample_plots = run_registered_subset(sample, raw_by_key, "Registered sample")
    sample_summary = summarize_registered(sample_rows, prefix="sample_")
    sample_fixed, sample_reason = fixed_from_sample(sample_summary)
    result = {
        "status": "SAMPLE_FIXED" if sample_fixed else "SAMPLE_NOT_FIXED",
        "reason": "registered sample cleared the fixed criteria" if sample_fixed else "registered sample did not clear the fixed criteria",
        "raw_source_note": raw_note,
        "raw_info": raw_info,
        "sample_seed": SAMPLE_SEED,
        "sample_n_requested": SAMPLE_N,
        "sample_fixed": sample_fixed,
        "sample_fixed_reason": sample_reason,
        "full_ran": False,
        **sample_summary,
    }
    draw_registered_examples(sample_rows, sample_plots, root / "figures", "gate3_F_registered_sample_examples.png")

    full_rows = None
    if sample_fixed:
        candidates = fits[(fits["SPATTYPE"] if "SPATTYPE" in fits.columns else True)] if False else fits.copy()
        candidates = candidates[(candidates["fit_error"] == "")]
        full_rows, full_plots = run_registered_subset(candidates, raw_by_key, "Registered full")
        full_summary = summarize_registered(full_rows, prefix="full_")
        result.update(full_summary)
        result["full_ran"] = True
        result["full_by_spaqeff"] = summarize_by_group(full_rows, "SPAQEFF")
        result["full_by_obstruction"] = summarize_by_group(full_rows, "obstruction_group")
        valid_full = full_rows[full_rows["registered_error"].fillna("") == ""].copy()
        failures = valid_full[~valid_full["registered_ratio_within_0p02"].fillna(False).astype(bool)].copy()
        result["full_ratio_0p02_failure_n"] = int(len(failures))
        result["full_ratio_0p02_failed_obstructed_n"] = int(failures["obstructed"].sum()) if len(failures) else 0
        result["full_ratio_0p02_failed_truncated_lt6s_n"] = int((failures["duration_s"] < 6.0).sum()) if "duration_s" in failures else 0
        result["full_ratio_0p02_failed_shift_median_samples"] = float(np.nanmedian(failures["registered_shift_samples"])) if len(failures) else np.nan
        draw_registered_examples(full_rows, full_plots, root / "figures", "gate3_F_registered_full_examples.png")
        result["status"] = "FULL_RAN"
        result["reason"] = "registered sample cleared criteria, so full cycle-F registered gate ran"

    write_report(result, sample_rows, full_rows, root / "results")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    print(f"Gate 3 registered fix {result['status']}: {result['reason']}")
    print(f"Sample dipole fraction: {result['sample_dipole_fraction']:.3f}")
    print(f"Sample all within 5%: {result['sample_all_within_5pct_fraction']:.3f}")
    print(f"Sample FEV1/FVC within 0.02: {result['sample_ratio_within_0p02_fraction']:.3f}")
    return 0 if result["sample_fixed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
