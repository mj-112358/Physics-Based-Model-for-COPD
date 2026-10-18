from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyreadstat
from PIL import Image, ImageDraw

from .dataset_resolver import resolve_dataset_path
from .fit import ADEQUATE_NRSS_THRESHOLD, fit_ombe_lm, np_ombe_flow
from .raw import interval_end_times
from .segment import backextrap_t0, derive_fev1_fvc, detect_forced_exhale_segment

RAW_REQUIRED = (
    "SEQN",
    "SPATTYPE",
    "SPAMANU",
    "SPAFACT",
    "SPAPLAT",
    "SPAACC",
    "SPAQEFF",
    "SPXPTS",
    "SPXRAW",
)
RAW_OPTIONAL = ("SPAPOS",)
HZ = 100
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20260710


def decode_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    if isinstance(value, str):
        return value.strip()
    return value


def find_or_unzip_raw(root: Path, cycle: str = "F") -> tuple[Path, str]:
    resolved = resolve_dataset_path(root, "SPXRAW", cycle)
    return resolved.path, resolved.note


def read_raw(path: Path, cycle: str = "F") -> tuple[pd.DataFrame, dict]:
    _, meta = pyreadstat.read_sas7bdat(str(path), metadataonly=True)
    columns = list(meta.column_names)
    missing = [name for name in RAW_REQUIRED if name not in columns]
    if missing:
        raise RuntimeError(f"SPXRAW_{cycle} missing required columns: {missing}; observed columns: {columns}")
    usecols = [name for name in RAW_REQUIRED + RAW_OPTIONAL if name in columns]
    df, _ = pyreadstat.read_sas7bdat(str(path), usecols=usecols)
    if df.empty:
        raise RuntimeError(f"SPXRAW_{cycle} is empty after read_sas7bdat")
    for col in ("SPATTYPE", "SPAPLAT", "SPAACC", "SPAQEFF", "SPAPOS"):
        if col in df.columns:
            df[col] = df[col].map(decode_value)
    info = {
        "path": str(path),
        "metadata_rows": int(meta.number_rows),
        "loaded_rows": int(len(df)),
        "columns": list(df.columns),
    }
    return df, info


def read_ground_truth(root: Path, cycle: str = "F") -> pd.DataFrame:
    resolved = resolve_dataset_path(root, "SPX", cycle)
    path = resolved.path
    df = pd.read_sas(path, format="xport")
    required = ("SEQN", "SPXNFEV1", "SPXNFVC")
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise RuntimeError(f"SPX_{cycle} missing required ground-truth columns: {missing}")
    keep = ["SEQN", "SPXNFEV1", "SPXNFVC"]
    if "SPXNQFVC" in df.columns:
        keep.append("SPXNQFVC")
    return df[keep].copy()


def parse_increments(raw: Any) -> np.ndarray:
    raw = decode_value(raw)
    if not isinstance(raw, str) or not raw:
        raise ValueError("SPXRAW is empty or non-string")
    values = np.fromstring(raw, sep=",", dtype=np.float64)
    if values.size == 0:
        raise ValueError("SPXRAW parsed to zero points")
    return values


def derive_curve(row: Any) -> dict:
    incr = parse_increments(row.SPXRAW)
    expected = int(row.SPXPTS)
    if len(incr) != expected:
        raise ValueError(f"SPXRAW length {len(incr)} != SPXPTS {expected}")
    fact = float(row.SPAFACT)
    flow_lps = incr * 0.1 * fact
    volume_ml = np.cumsum(incr) * fact
    # Each SPXRAW value is the volume change over a completed 0.01 s
    # interval. The first cumulative volume therefore belongs at 0.01 s,
    # rather than at time zero. This convention also matches raw.py.
    time_s = interval_end_times(len(incr), HZ)
    onset_idx, pef_idx, end_idx = detect_forced_exhale_segment(flow_lps, hz=HZ)
    t0_s = backextrap_t0(time_s, flow_lps, volume_ml, onset_idx, pef_idx)
    fev1_ml, fvc_ml = derive_fev1_fvc(time_s, volume_ml, t0_s, end_idx)
    v0 = float(np.interp(t0_s, time_s, volume_ml))
    onset_volume = float(volume_ml[onset_idx])
    bev_ml = abs(v0 - onset_volume)
    bev_limit_ml = max(100.0, 0.05 * max(fvc_ml, 0.0))
    return {
        "n_points": int(len(incr)),
        "btps_factor": fact,
        "onset_idx": int(onset_idx),
        "pef_idx": int(pef_idx),
        "end_idx": int(end_idx),
        "t0_s": float(t0_s),
        "duration_s": float(time_s[end_idx] - max(t0_s, 0.0)),
        "derived_fev1_ml": float(fev1_ml),
        "derived_fvc_ml": float(fvc_ml),
        "derived_ratio": float(fev1_ml / fvc_ml) if fvc_ml > 0 else np.nan,
        "bev_ml": float(bev_ml),
        "bev_limit_ml": float(bev_limit_ml),
        "bev_flag": bool(bev_ml > bev_limit_ml),
        "final_volume_ml": float(volume_ml[-1]),
    }


def derive_all(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    errors = []
    for idx, row in enumerate(raw.itertuples(index=False)):
        base = {
            "SEQN": float(row.SEQN),
            "SPATTYPE": row.SPATTYPE,
            "SPAMANU": float(row.SPAMANU),
            "SPAFACT": float(row.SPAFACT),
            "SPAPLAT": row.SPAPLAT,
            "SPAACC": row.SPAACC,
            "SPAQEFF": row.SPAQEFF,
            "SPXPTS": int(row.SPXPTS),
        }
        try:
            base.update(derive_curve(row))
            base["parse_error"] = ""
        except Exception as exc:
            base["parse_error"] = str(exc)
            errors.append(base)
        rows.append(base)
        if (idx + 1) % 5000 == 0 or idx + 1 == len(raw):
            print(f"Gate 2 parse progress: {idx + 1}/{len(raw)} rows")
    result = pd.DataFrame(rows)
    result.attrs["parse_error_count"] = len(errors)
    return result


def pct_error(derived: pd.Series, truth: pd.Series) -> pd.Series:
    return 100.0 * (derived - truth) / truth


def bootstrap_binary_rate_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    draws = np.empty(BOOTSTRAP_B, dtype=float)
    bool_values = values.astype(bool)
    for b in range(BOOTSTRAP_B):
        draws[b] = float(bool_values[rng.choice(idx, size=n, replace=True)].mean())
    return float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def underpowered_rate(point: float, low: float, high: float, n: int) -> bool:
    if n < 30 or not np.isfinite(point) or not np.isfinite(low) or not np.isfinite(high):
        return True
    return bool((high - low) / 2.0 > point)


def pct_text(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return "NA"
    return f"{v:.2f}%" if np.isfinite(v) else "NA"


def comparison_summary(df: pd.DataFrame, label: str) -> dict:
    valid = df.dropna(subset=["derived_fev1_ml", "derived_fvc_ml", "SPXNFEV1", "SPXNFVC"]).copy()
    valid = valid[(valid["SPXNFEV1"] > 0) & (valid["SPXNFVC"] > 0)]
    if valid.empty:
        return {"label": label, "n": 0}
    fev1_err = pct_error(valid["derived_fev1_ml"], valid["SPXNFEV1"])
    fvc_err = pct_error(valid["derived_fvc_ml"], valid["SPXNFVC"])
    ratio_err = pct_error(valid["derived_ratio"], valid["SPXNFEV1"] / valid["SPXNFVC"])
    bev_flags = valid["bev_flag"].astype(bool).to_numpy() if "bev_flag" in valid else np.array([], dtype=bool)
    bev_rate = float(bev_flags.mean()) if len(bev_flags) else np.nan
    bev_ci_low, bev_ci_high = bootstrap_binary_rate_ci(bev_flags, BOOTSTRAP_SEED + sum(ord(ch) for ch in label))
    return {
        "label": label,
        "n": int(len(valid)),
        "fev1_median_signed_pct_error": float(np.median(fev1_err)),
        "fev1_median_abs_pct_error": float(np.median(np.abs(fev1_err))),
        "fev1_abs_pct_error_iqr": [float(x) for x in np.percentile(np.abs(fev1_err), [25, 75])],
        "fvc_median_signed_pct_error": float(np.median(fvc_err)),
        "fvc_median_abs_pct_error": float(np.median(np.abs(fvc_err))),
        "fvc_abs_pct_error_iqr": [float(x) for x in np.percentile(np.abs(fvc_err), [25, 75])],
        "ratio_median_signed_pct_error": float(np.median(ratio_err)),
        "ratio_median_abs_pct_error": float(np.median(np.abs(ratio_err))),
        "bev_flag_fraction": bev_rate,
        "bev_flag_n": int(bev_flags.sum()) if len(bev_flags) else 0,
        "bev_flag_den": int(len(bev_flags)),
        "bev_flag_ci_low": bev_ci_low,
        "bev_flag_ci_high": bev_ci_high,
        "bev_flag_underpowered": underpowered_rate(bev_rate, bev_ci_low, bev_ci_high, len(bev_flags)),
    }


def participant_sets(derived: pd.DataFrame, truth: pd.DataFrame) -> dict[str, pd.DataFrame]:
    merged = derived.merge(truth, on="SEQN", how="left")
    pre = merged[(merged["SPATTYPE"] == "Pre") & (merged["parse_error"] == "")]
    acceptable = pre[pre["SPAACC"] == "Y"].copy()
    acceptable = acceptable.dropna(subset=["SPXNFEV1", "SPXNFVC"])
    acceptable = acceptable.sort_values(["SEQN", "SPAMANU"])

    first = acceptable.groupby("SEQN", as_index=False).first()
    first_a = acceptable[acceptable["SPAQEFF"] == "A"].groupby("SEQN", as_index=False).first()

    max_any = (
        acceptable.groupby("SEQN", as_index=False)
        .agg(
            derived_fev1_ml=("derived_fev1_ml", "max"),
            derived_fvc_ml=("derived_fvc_ml", "max"),
            SPXNFEV1=("SPXNFEV1", "first"),
            SPXNFVC=("SPXNFVC", "first"),
            derived_ratio=("derived_ratio", "max"),
            bev_flag=("bev_flag", "max"),
            SPAQEFF=("SPAQEFF", "first"),
            SPAACC=("SPAACC", "first"),
        )
    )
    a_curves = acceptable[acceptable["SPAQEFF"] == "A"].copy()
    max_a = (
        a_curves.groupby("SEQN", as_index=False)
        .agg(
            derived_fev1_ml=("derived_fev1_ml", "max"),
            derived_fvc_ml=("derived_fvc_ml", "max"),
            SPXNFEV1=("SPXNFEV1", "first"),
            SPXNFVC=("SPXNFVC", "first"),
            derived_ratio=("derived_ratio", "max"),
            bev_flag=("bev_flag", "max"),
            SPAQEFF=("SPAQEFF", "first"),
            SPAACC=("SPAACC", "first"),
        )
    )
    return {
        "all_acceptable_A_pre_curves": a_curves,
        "first_acceptable_pre_per_participant": first,
        "first_acceptable_A_pre_per_participant": first_a,
        "max_acceptable_pre_per_participant": max_any,
        "max_acceptable_A_pre_per_participant": max_a,
        "pre_merged": pre,
        "acceptable_pre": acceptable,
    }


def diagnose_gate2(summary: dict, criterion: dict) -> str:
    if criterion.get("n", 0) == 0:
        return "no criterion rows available after filtering"
    fev1_signed = criterion["fev1_median_signed_pct_error"]
    fvc_signed = criterion["fvc_median_signed_pct_error"]
    fev1_abs_iqr = criterion["fev1_abs_pct_error_iqr"]
    fvc_abs_iqr = criterion["fvc_abs_pct_error_iqr"]
    width = max(fev1_abs_iqr[1] - fev1_abs_iqr[0], fvc_abs_iqr[1] - fvc_abs_iqr[0])
    if np.sign(fev1_signed) == np.sign(fvc_signed) and abs(fev1_signed - fvc_signed) < 3.0 and width < 7.5:
        return "constant multiplicative pattern: BTPS/unit scaling is the leading suspect"
    if abs(fvc_signed) > abs(fev1_signed) + 3.0:
        return "constant/additive or endpoint pattern: t0/end-of-test segmentation is the leading suspect"
    return "scattered pattern: segmentation variability is the leading suspect"


def draw_curve_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], t: np.ndarray, flow: np.ndarray, volume: np.ndarray, title: str) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline="black")
    draw.text((x0 + 6, y0 + 4), title, fill="black")
    w = x1 - x0 - 20
    h = y1 - y0 - 30
    px0 = x0 + 10
    py0 = y0 + 24
    max_t = max(float(t[-1]), 1.0)
    flow_max = max(float(np.nanmax(np.abs(flow))), 0.5)
    vol_max = max(float(np.nanmax(volume)), 1.0)
    mid_y = py0 + h * 0.42
    points_flow = []
    points_vol = []
    step = max(1, len(t) // 500)
    for ti, fi, vi in zip(t[::step], flow[::step], volume[::step]):
        x = px0 + (float(ti) / max_t) * w
        y_flow = mid_y - (float(fi) / flow_max) * (h * 0.36)
        y_vol = py0 + h - (float(vi) / vol_max) * (h * 0.36)
        points_flow.append((x, y_flow))
        points_vol.append((x, y_vol))
    if len(points_flow) > 1:
        draw.line(points_flow, fill=(30, 90, 180), width=2)
        draw.line(points_vol, fill=(30, 140, 80), width=2)
    draw.text((x1 - 120, y0 + 4), "blue=flow green=vol", fill="black")


def draw_gate2_figures(raw: pd.DataFrame, derived: pd.DataFrame, figures_dir: Path, cycle: str = "F") -> None:
    figures_dir.mkdir(exist_ok=True)
    candidates = derived[(derived["SPATTYPE"] == "Pre") & (derived["SPAACC"] == "Y") & (derived["parse_error"] == "")].head(20)
    if candidates.empty:
        return
    raw_by_key = {
        (float(row.SEQN), float(row.SPAMANU)): row
        for row in raw.itertuples(index=False)
    }
    panels = []
    for row in candidates.itertuples(index=False):
        raw_row = raw_by_key.get((float(row.SEQN), float(row.SPAMANU)))
        if raw_row is None:
            continue
        incr = parse_increments(raw_row.SPXRAW)
        fact = float(raw_row.SPAFACT)
        flow = incr * 0.1 * fact
        volume = np.cumsum(incr) * fact
        time = np.arange(len(incr), dtype=float) / HZ
        panels.append((time, flow, volume, f"SEQN {int(row.SEQN)} manu {int(row.SPAMANU)} {row.SPAQEFF}"))
    if not panels:
        return
    image = Image.new("RGB", (1200, 520), "white")
    draw = ImageDraw.Draw(image)
    draw_curve_panel(draw, (40, 50, 1160, 480), *panels[0])
    draw.text((40, 20), f"Gate 2 cycle {cycle} single real SPXRAW curve", fill="black")
    image.save(figures_dir / f"gate2_{cycle}_single.png")

    grid = Image.new("RGB", (1400, 1200), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((35, 20), f"Gate 2 cycle {cycle} real SPXRAW grid", fill="black")
    for idx, panel in enumerate(panels[:20]):
        col = idx % 4
        row = idx // 4
        x0 = 25 + col * 340
        y0 = 55 + row * 220
        draw_curve_panel(draw, (x0, y0, x0 + 320, y0 + 200), *panel)
    grid.save(figures_dir / f"gate2_{cycle}_grid.png")


def write_gate2_report(result: dict, results_dir: Path, cycle: str = "F") -> None:
    results_dir.mkdir(exist_ok=True)
    lines = [
        f"# Gate 2 - Cycle {cycle} Parse + FEV1 Recovery",
        "",
        f"- status: {result['status']}",
        f"- reason: {result['reason']}",
        f"- cycle: {cycle}",
        f"- parser callable: {result['parser_callable']}",
        f"- raw source: {result['raw_info']['path']}",
        f"- raw source note: {result['raw_source_note']}",
        f"- raw metadata rows: {result['raw_info']['metadata_rows']}",
        f"- raw loaded rows: {result['raw_info']['loaded_rows']}",
        f"- required columns present: {', '.join(RAW_REQUIRED)}",
        f"- parse errors: {result['parse_error_count']}",
        "",
        "## Row Counts",
        "",
    ]
    for key, value in result["counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Ground-Truth Comparisons",
            "",
            "| comparison set | n | median signed FEV1 err | median abs FEV1 err | median signed FVC err | median abs FVC err | median abs ratio err | BEV flag frac |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in result["comparison_summaries"]:
        if item.get("n", 0) == 0:
            lines.append(f"| {item['label']} | 0 | NA | NA | NA | NA | NA | NA |")
            continue
        lines.append(
            f"| {item['label']} | {item['n']} | "
            f"{pct_text(item['fev1_median_signed_pct_error'])} | "
            f"{pct_text(item['fev1_median_abs_pct_error'])} | "
            f"{pct_text(item['fvc_median_signed_pct_error'])} | "
            f"{pct_text(item['fvc_median_abs_pct_error'])} | "
            f"{pct_text(item['ratio_median_abs_pct_error'])} | "
            f"{item['bev_flag_fraction']:.3f} ({item.get('bev_flag_n', '')}/{item.get('bev_flag_den', '')}; "
            f"CI {item.get('bev_flag_ci_low', np.nan):.3f}-{item.get('bev_flag_ci_high', np.nan):.3f}) |"
        )
    criterion = result["criterion"]
    lines.extend(
        [
            "",
            "## Criterion",
            "",
            f"Criterion set: max acceptable (`SPAACC=='Y'`) pre-bronchodilator value per participant, matching the observed `SPX_{cycle}` best-test/composite behavior.",
            f"- criterion n: {criterion.get('n', 0)}",
            f"- median absolute FEV1 error: {pct_text(criterion.get('fev1_median_abs_pct_error', float('nan')))}",
            f"- median absolute FVC error: {pct_text(criterion.get('fvc_median_abs_pct_error', float('nan')))}",
            "- pass threshold: both <= 5%",
            f"- diagnosis if failed: {result['diagnosis']}",
            "- residual note: the remaining one-direction low bias is reported as composite/selection mismatch plus possible mild back-extrapolation clipping; no scaling correction was applied.",
            "",
            "Parsing formula used exactly: comma-split SPXRAW to integer volume increments; `flow_Lps=incr*0.1*SPAFACT`; `volume_mL=cumsum(incr)*SPAFACT`; no differentiation.",
            "",
        ]
    )
    (results_dir / f"gate2_{cycle}_parse.md").write_text("\n".join(lines), encoding="utf-8")
    (results_dir / f"gate2_{cycle}_parse.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def run_gate2(root: Path, cycle: str = "F") -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_dir = root / "results"
    figures_dir = root / "figures"
    raw_path, raw_note = find_or_unzip_raw(root, cycle)
    raw, raw_info = read_raw(raw_path, cycle)
    print(f"SPXRAW_{cycle} row count: {len(raw)}")
    print(f"SPXRAW_{cycle} columns: {list(raw.columns)}")
    print(f"Parser callable shared across cycles: {derive_curve.__module__}.{derive_curve.__name__}")
    truth = read_ground_truth(root, cycle)
    derived = derive_all(raw)
    sets = participant_sets(derived, truth)
    counts = {
        "raw_rows": int(len(raw)),
        "pre_rows": int((derived["SPATTYPE"] == "Pre").sum()),
        "acceptable_pre_rows": int(((derived["SPATTYPE"] == "Pre") & (derived["SPAACC"] == "Y")).sum()),
        "A_pre_rows": int(((derived["SPATTYPE"] == "Pre") & (derived["SPAQEFF"] == "A")).sum()),
        "acceptable_A_pre_rows": int(
            ((derived["SPATTYPE"] == "Pre") & (derived["SPAACC"] == "Y") & (derived["SPAQEFF"] == "A")).sum()
        ),
        "participants_with_truth": int(truth[["SPXNFEV1", "SPXNFVC"]].dropna().shape[0]),
        "participants_first_acceptable": int(len(sets["first_acceptable_pre_per_participant"])),
        "participants_first_acceptable_A": int(len(sets["first_acceptable_A_pre_per_participant"])),
    }
    comparison_summaries = [
        comparison_summary(sets["all_acceptable_A_pre_curves"], "all acceptable A Pre curves"),
        comparison_summary(sets["first_acceptable_A_pre_per_participant"], "first acceptable A Pre per participant"),
        comparison_summary(sets["first_acceptable_pre_per_participant"], "first acceptable Pre per participant"),
        comparison_summary(sets["max_acceptable_A_pre_per_participant"], "max acceptable A Pre per participant"),
        comparison_summary(sets["max_acceptable_pre_per_participant"], "max acceptable Pre per participant"),
    ]
    criterion = next(item for item in comparison_summaries if item["label"] == "max acceptable Pre per participant")
    gate2_pass = (
        criterion.get("n", 0) > 0
        and criterion["fev1_median_abs_pct_error"] <= 5.0
        and criterion["fvc_median_abs_pct_error"] <= 5.0
    )
    result = {
        "cycle": cycle,
        "status": "PASS" if gate2_pass else "FAIL",
        "reason": "Gate 2 parser recovery criterion passed under max/best-test comparison." if gate2_pass else "Gate 2 parser recovery criterion failed; stopping before OMBE fit.",
        "parser_callable": f"{derive_curve.__module__}.{derive_curve.__name__}",
        "raw_source_note": raw_note,
        "raw_info": raw_info,
        "parse_error_count": int(derived.attrs.get("parse_error_count", 0)),
        "counts": counts,
        "comparison_summaries": comparison_summaries,
        "criterion": criterion,
        "diagnosis": "none; Gate 2 passed" if gate2_pass else diagnose_gate2(comparison_summaries[0], criterion),
    }
    draw_gate2_figures(raw, derived, figures_dir, cycle)
    write_gate2_report(result, results_dir, cycle)
    return result, raw, derived, truth


def fit_curve_for_gate3(row: Any) -> dict:
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
    if len(segment_t) < 20:
        raise ValueError("fit segment has fewer than 20 samples")
    fit = fit_ombe_lm(segment_t, segment_flow)
    result = {
        "nrss": fit.nrss,
        "r2": fit.r2,
        "fit_success": fit.success,
        "adequate": bool(fit.nrss <= ADEQUATE_NRSS_THRESHOLD),
        "fit_duration_s": float(segment_t[-1] - segment_t[0]),
    }
    for name, value in fit.params.items():
        result[f"fit_{name}"] = float(value)
    return result


def summarize_fraction(df: pd.DataFrame, group_col: str) -> list[dict]:
    rows = []
    for key, group in df.groupby(group_col, dropna=False):
        n = len(group)
        rows.append(
            {
                "group": str(key),
                "n": int(n),
                "adequate": int(group["adequate"].sum()),
                "adequate_fraction": float(group["adequate"].mean()) if n else np.nan,
            }
        )
    return rows


def severity_group(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "missing"
    if ratio >= 0.70:
        return "non-obstructed"
    if ratio < 0.50:
        return "<0.50"
    if ratio < 0.60:
        return "0.50-0.60"
    return "0.60-0.70"


def prepare_fit_panel(row: Any, raw_by_key: dict[tuple[float, float], Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str] | None:
    raw_row = raw_by_key.get((float(row.SEQN), float(row.SPAMANU)))
    if raw_row is None:
        return None
    try:
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
        label = "obstructed" if bool(row.obstructed) else "non-obstructed"
        title = f"{label} NRSS {row.nrss:.1f}"
        return segment_t, segment_flow, pred, volume[start_idx : end_idx + 1], title
    except Exception:
        return None


def draw_fit_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    t: np.ndarray,
    flow: np.ndarray,
    pred: np.ndarray,
    title: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline="black")
    draw.text((x0 + 6, y0 + 4), title, fill="black")
    w = x1 - x0 - 20
    h = y1 - y0 - 34
    px0 = x0 + 10
    py0 = y0 + 26
    max_t = max(float(t[-1]), 1.0)
    flow_max = max(float(np.nanmax(np.abs(np.concatenate([flow, pred])))), 0.5)
    mid_y = py0 + h * 0.62
    step = max(1, len(t) // 600)
    measured = []
    fitted = []
    for ti, fi, pi in zip(t[::step], flow[::step], pred[::step]):
        x = px0 + (float(ti) / max_t) * w
        measured.append((x, mid_y - (float(fi) / flow_max) * (h * 0.48)))
        fitted.append((x, mid_y - (float(pi) / flow_max) * (h * 0.48)))
    if len(measured) > 1:
        draw.line(measured, fill=(30, 90, 180), width=2)
        draw.line(fitted, fill=(210, 45, 45), width=2)
    draw.line((px0, mid_y, px0 + w, mid_y), fill=(180, 180, 180), width=1)
    draw.text((x1 - 145, y0 + 4), "blue=flow red=OMBE", fill="black")


def draw_gate3_examples(fits: pd.DataFrame, raw: pd.DataFrame, figures_dir: Path, cycle: str = "F") -> None:
    figures_dir.mkdir(exist_ok=True)
    if fits.empty:
        return
    chosen_rows = []
    for obstructed in (False, True):
        group = fits[fits["obstructed"] == obstructed].dropna(subset=["nrss"])
        if group.empty:
            continue
        chosen_rows.append(group.sort_values("nrss").head(1))
        chosen_rows.append(group.sort_values("nrss", ascending=False).head(1))
    chosen = pd.concat(chosen_rows, ignore_index=True) if chosen_rows else fits.dropna(subset=["nrss"]).head(0)
    raw_by_key = {
        (float(row.SEQN), float(row.SPAMANU)): row
        for row in raw.itertuples(index=False)
    }
    image = Image.new("RGB", (1200, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.text((35, 20), f"Gate 3 cycle {cycle} OMBE examples: best/worst non-obstructed and obstructed real curves", fill="black")
    for idx, row in enumerate(chosen.itertuples(index=False)):
        panel = prepare_fit_panel(row, raw_by_key)
        if panel is None:
            continue
        t, flow, pred, _volume, title = panel
        x0 = 35 + (idx % 2) * 570
        y0 = 60 + (idx // 2) * 300
        draw_fit_panel(draw, (x0, y0, x0 + 540, y0 + 270), t, flow, pred, title)
    image.save(figures_dir / f"gate3_{cycle}_examples.png")


def run_gate3(root: Path, raw: pd.DataFrame, derived: pd.DataFrame, truth: pd.DataFrame, cycle: str = "F") -> dict:
    results_dir = root / "results"
    figures_dir = root / "figures"
    merged = raw.merge(
        derived[["SEQN", "SPAMANU", "SPATTYPE", "SPAACC", "SPAQEFF", "parse_error", "duration_s", "derived_ratio"]],
        on=["SEQN", "SPAMANU", "SPATTYPE", "SPAACC", "SPAQEFF"],
        how="left",
    ).merge(truth, on="SEQN", how="left")
    rows = []
    candidates = merged[(merged["SPATTYPE"] == "Pre") & (merged["SPAACC"] == "Y") & (merged["parse_error"] == "")]
    for idx, row in enumerate(candidates.itertuples(index=False)):
        ratio = float(row.derived_ratio) if pd.notna(row.derived_ratio) else np.nan
        out = {
            "SEQN": float(row.SEQN),
            "SPAMANU": float(row.SPAMANU),
            "SPAQEFF": row.SPAQEFF,
            "duration_s": float(row.duration_s),
            "fev1_fvc_ratio": ratio,
            "obstructed": bool(ratio < 0.70) if np.isfinite(ratio) else False,
            "obstruction_group": "obstructed_derived_ratio_lt_0.70" if np.isfinite(ratio) and ratio < 0.70 else ("non_obstructed_derived_ratio_ge_0.70" if np.isfinite(ratio) else "missing_derived_ratio"),
            "severity_bin": severity_group(ratio),
        }
        try:
            out.update(fit_curve_for_gate3(row))
            out["fit_error"] = ""
        except Exception as exc:
            out.update({"nrss": np.nan, "r2": np.nan, "fit_success": False, "adequate": False, "fit_duration_s": np.nan})
            out["fit_error"] = str(exc)
        rows.append(out)
        if (idx + 1) % 1000 == 0 or idx + 1 == len(candidates):
            print(f"Gate 3 fit progress: {idx + 1}/{len(candidates)} curves")
    fits = pd.DataFrame(rows)
    adequate_fraction = float(fits["adequate"].mean()) if len(fits) else 0.0
    failed = fits[~fits["adequate"]]
    result = {
        "cycle": cycle,
        "status": "PASS" if adequate_fraction >= 0.85 else "FAIL",
        "reason": "Gate 3 real-curve OMBE fit fraction passed." if adequate_fraction >= 0.85 else "Gate 3 real-curve OMBE fit fraction failed.",
        "cycle_scope": f"cycle {cycle}; obstructed-subgroup numbers remain cycle-specific.",
        "n_fit_curves": int(len(fits)),
        "adequate_threshold_nrss": ADEQUATE_NRSS_THRESHOLD,
        "adequate_count": int(fits["adequate"].sum()) if len(fits) else 0,
        "adequate_fraction": adequate_fraction,
        "nrss_median": float(np.nanmedian(fits["nrss"])) if len(fits) else np.nan,
        "nrss_iqr": [float(x) for x in np.nanpercentile(fits["nrss"], [25, 75])] if len(fits) else [np.nan, np.nan],
        "r2_median": float(np.nanmedian(fits["r2"])) if len(fits) else np.nan,
        "r2_iqr": [float(x) for x in np.nanpercentile(fits["r2"], [25, 75])] if len(fits) else [np.nan, np.nan],
        "by_spaqeff": summarize_fraction(fits, "SPAQEFF") if len(fits) else [],
        "by_obstruction": summarize_fraction(fits, "obstruction_group") if len(fits) else [],
        "by_obstructed_severity": summarize_fraction(fits[fits["obstructed"]], "severity_bin") if len(fits) else [],
        "failure_characterization": {
            "n_failed": int(len(failed)),
            "failed_obstructed_n": int(failed["obstructed"].sum()) if len(failed) else 0,
            "failed_truncated_lt6s_n": int((failed["duration_s"] < 6.0).sum()) if len(failed) else 0,
            "failed_spaqeff_D_n": int((failed["SPAQEFF"] == "D").sum()) if len(failed) else 0,
            "fit_error_n": int((failed["fit_error"] != "").sum()) if len(failed) else 0,
        },
    }
    fits.to_csv(results_dir / f"gate3_{cycle}_curve_fits.csv", index=False)
    draw_gate3_examples(fits, raw, figures_dir, cycle)
    write_gate3_report(result, results_dir, cycle)
    return result


def write_gate3_report(result: dict, results_dir: Path, cycle: str = "F") -> None:
    lines = [
        f"# Gate 3 - Cycle {cycle} OMBE Fit on Real Curves",
        "",
        f"- status: {result['status']}",
        f"- reason: {result['reason']}",
        f"- scope: {result['cycle_scope']}",
        f"- adequate threshold NRSS: {result['adequate_threshold_nrss']:.2f}",
        "- threshold note: unchanged from synthetic Module 2/Steltner-like threshold; not tuned on real data.",
        f"- fit curves: {result['n_fit_curves']}",
        f"- adequate fits: {result['adequate_count']}/{result['n_fit_curves']} ({100 * result['adequate_fraction']:.1f}%)",
        f"- NRSS median: {result['nrss_median']:.3f}",
        f"- NRSS IQR: {result['nrss_iqr'][0]:.3f} to {result['nrss_iqr'][1]:.3f}",
        f"- R^2 median: {result['r2_median']:.4f}",
        f"- R^2 IQR: {result['r2_iqr'][0]:.4f} to {result['r2_iqr'][1]:.4f}",
        "",
        "## Fit Fraction by SPAQEFF",
        "",
        "| SPAQEFF | n | adequate | adequate fraction |",
        "|---|---:|---:|---:|",
    ]
    for row in result["by_spaqeff"]:
        lines.append(f"| {row['group']} | {row['n']} | {row['adequate']} | {row['adequate_fraction']:.3f} |")
    lines.extend(
        [
            "",
            f"## Obstructed Fit Fraction - Cycle {cycle}",
            "",
            f"All obstructed-subgroup numbers here use derived per-curve FEV1/FVC and are cycle {cycle} only.",
            "",
            "| obstruction group | n | adequate | adequate fraction |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in result["by_obstruction"]:
        lines.append(f"| {row['group']} | {row['n']} | {row['adequate']} | {row['adequate_fraction']:.3f} |")
    lines.extend(
        [
            "",
            f"## Obstructed Severity - Cycle {cycle}",
            "",
            "| derived FEV1/FVC severity bin | n | adequate | adequate fraction |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in result["by_obstructed_severity"]:
        lines.append(f"| {row['group']} | {row['n']} | {row['adequate']} | {row['adequate_fraction']:.3f} |")
    fc = result["failure_characterization"]
    lines.extend(
        [
            "",
            "## Failure Characterization",
            "",
            f"- failed curves: {fc['n_failed']}",
            f"- failed obstructed curves: {fc['failed_obstructed_n']} (cycle {cycle})",
            f"- failed truncated <6s curves: {fc['failed_truncated_lt6s_n']}",
            f"- failed SPAQEFF=='D' curves: {fc['failed_spaqeff_D_n']}",
            f"- fit-error curves: {fc['fit_error_n']}",
            "",
        ]
    )
    (results_dir / f"gate3_{cycle}_ombe.md").write_text("\n".join(lines), encoding="utf-8")
    (results_dir / f"gate3_{cycle}_ombe.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def write_gate3_not_run(results_dir: Path, gate2: dict, cycle: str = "F") -> None:
    result = {
        "cycle": cycle,
        "status": "NOT-RUN",
        "reason": "Gate 2 failed, so OMBE fit on real curves was not run.",
        "gate2_status": gate2["status"],
        "gate2_reason": gate2["reason"],
        "cycle_scope": f"cycle {cycle}.",
    }
    (results_dir / f"gate3_{cycle}_ombe.md").write_text(
        "\n".join(
            [
                f"# Gate 3 - Cycle {cycle} OMBE Fit on Real Curves",
                "",
                "- status: NOT-RUN",
                "- reason: Gate 2 failed, so OMBE fit on real curves was not run.",
                f"- Gate 2 reason: {gate2['reason']}",
                f"- scope: cycle {cycle}.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (results_dir / f"gate3_{cycle}_ombe.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--cycle", default="F", choices=("E", "F", "G"))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    cycle = args.cycle.upper()
    gate2, raw, derived, truth = run_gate2(root, cycle)
    if gate2["status"] != "PASS":
        write_gate3_not_run(root / "results", gate2, cycle)
        print(f"Gate 2 {gate2['status']}: {gate2['reason']}")
        print("Gate 3 NOT-RUN: Gate 2 failed.")
        return 1
    gate3 = run_gate3(root, raw, derived, truth, cycle)
    print(f"Gate 2 {gate2['status']}: {gate2['reason']}")
    print(f"Gate 3 {gate3['status']}: {gate3['reason']}")
    return 0 if gate3["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
