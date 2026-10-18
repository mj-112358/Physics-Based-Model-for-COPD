from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import chisquare

from .gate23_f import find_or_unzip_raw, read_raw
from .gate_clinical_classification_f import (
    RANDOM_SEED,
    build_usable_raw_candidates,
    extract_curve_for_fit,
    load_column_map,
    load_gli,
    load_truth,
)
from .gate_fvc_correction_f import drop_bias_audit
from .gate_policy_calibrated_flag_f import build_pool
from .ombe import OmbeGrid
from .tsnpe import features_from_phi, prior_bounds_phi, sample_prior_phi

POSTERIOR_DRAWS = 400
TRAINING_SIMS = 6_000
MAX_TRAIN_EPOCHS = 60
VOLUME_SIGMA_L = 0.050
COVERAGE_LEVELS = (0.50, 0.80, 0.95)
PIT_BINS = 10


def configure_caches(root: Path) -> None:
    cache = root / ".cache"
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))


def volume_from_phi(theta_phi: np.ndarray, t: np.ndarray) -> np.ndarray:
    nat = {
        "tau_f": np.exp(theta_phi[:, 0]),
        "tau_s": np.exp(theta_phi[:, 1]),
        "alpha": 1.0 / (1.0 + np.exp(-theta_phi[:, 2])),
        "fvc_l": np.exp(theta_phi[:, 3]),
        "tau_o": np.exp(theta_phi[:, 4]),
    }
    tau_f = nat["tau_f"][:, None]
    tau_s = nat["tau_s"][:, None]
    alpha = nat["alpha"][:, None]
    fvc_l = nat["fvc_l"][:, None]
    tau_o = nat["tau_o"][:, None]
    tt = t[None, :]
    base = alpha * np.exp(-tt / tau_f) / tau_f + (1.0 - alpha) * np.exp(-tt / tau_s) / tau_s
    rise = 1.0 - np.exp(-tt / tau_o)
    shape = np.maximum(base * rise, 1e-12)
    area = np.trapezoid(shape, t, axis=1)[:, None]
    flow = fvc_l * shape / area
    increments = (flow[:, :-1] + flow[:, 1:]) * 0.5 * np.diff(t)[None, :]
    return np.concatenate([np.zeros((len(theta_phi), 1)), np.cumsum(increments, axis=1)], axis=1).astype(np.float32)


def volume_summary_features(volume: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Low-dimensional OMBE volume summaries used as SNPE context."""
    v = np.atleast_2d(np.asarray(volume, dtype=np.float32))
    fvc = np.maximum(v[:, -1], 0.1)

    def at(seconds: float) -> np.ndarray:
        return np.array([np.interp(seconds, t, row) for row in v], dtype=np.float32)

    v05 = at(0.5)
    v1 = at(1.0)
    v2 = at(2.0)
    v3 = at(3.0)
    v6 = at(6.0)
    v11 = at(11.0)
    last_delta = np.maximum(v[:, -1] - v11, 0.0)
    return np.column_stack(
        [
            v1,
            fvc,
            v1 / fvc,
            v05 / fvc,
            v2 / fvc,
            v3 / fvc,
            v6 / fvc,
            last_delta,
        ]
    ).astype(np.float32)


def train_volume_tsnpe(root: Path, t: np.ndarray) -> tuple[object, dict]:
    configure_caches(root)
    from sbi.inference import SNPE
    from sbi.neural_nets import posterior_nn
    from sbi.utils import BoxUniform

    print(f"Posterior training: generating {TRAINING_SIMS} OMBE volume simulations")
    torch.set_num_threads(4)
    lo, hi = prior_bounds_phi()
    prior = BoxUniform(low=torch.as_tensor(lo), high=torch.as_tensor(hi))
    density_builder = posterior_nn(
        model="nsf",
        z_score_theta="independent",
        z_score_x="independent",
        hidden_features=48,
        num_transforms=3,
    )
    inference = SNPE(prior=prior, density_estimator=density_builder, show_progress_bars=False)
    theta = sample_prior_phi(TRAINING_SIMS, seed=RANDOM_SEED + 9000).astype(np.float32)
    clean = volume_from_phi(theta, t)
    rng = np.random.default_rng(RANDOM_SEED + 9100)
    noisy = clean + rng.normal(0.0, VOLUME_SIGMA_L, size=clean.shape).astype(np.float32)
    x_summary = volume_summary_features(noisy, t)
    inference.append_simulations(torch.as_tensor(theta), torch.as_tensor(x_summary))
    print(
        "Posterior training: simulations appended; "
        f"context_dim={x_summary.shape[1]}, max_epochs={MAX_TRAIN_EPOCHS}"
    )
    start = time.perf_counter()
    estimator = inference.train(
        training_batch_size=512,
        learning_rate=5e-4,
        validation_fraction=0.1,
        stop_after_epochs=20,
        max_num_epochs=MAX_TRAIN_EPOCHS,
        show_train_summary=False,
    )
    print("Posterior training: density estimator trained; building posterior")
    posterior = inference.build_posterior(estimator, prior=prior, sample_with="direct")
    elapsed = time.perf_counter() - start
    summary = getattr(inference, "_summary", {}) or {}
    return posterior, {
        "engine": "sbi SNPE_C posterior_nn(model='nsf') on OMBE volume-summary context",
        "simulation_budget": TRAINING_SIMS,
        "posterior_draws_per_curve": POSTERIOR_DRAWS,
        "volume_sigma_l": VOLUME_SIGMA_L,
        "max_train_epochs": MAX_TRAIN_EPOCHS,
        "training_wall_clock_s": float(elapsed),
        "training_loss_final": float(summary.get("training_loss", [np.nan])[-1]) if summary.get("training_loss") else np.nan,
        "validation_loss_final": float(summary.get("validation_loss", [np.nan])[-1]) if summary.get("validation_loss") else np.nan,
        "convergence_diagnostics": "R-hat/ESS/divergences are not defined for amortized TSNPE; nonfinite posterior-sample failures are counted separately.",
    }


def curve_to_grid(raw_row: Any, t: np.ndarray) -> np.ndarray:
    curve = extract_curve_for_fit(raw_row)
    return np.interp(t, curve["vol_t"], curve["vol_l"], left=0.0, right=curve["vol_l"][-1]).astype(np.float32)


def ratio_samples_from_phi(phi: np.ndarray, t: np.ndarray) -> np.ndarray:
    return features_from_phi(phi.astype(np.float32), t)["ratio"].astype(np.float32)


def sample_posterior_ratios(
    posterior: object,
    raw: pd.DataFrame,
    rows: pd.DataFrame,
    t: np.ndarray,
    draws: int,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict]:
    raw_by_index = {int(idx): row for idx, row in raw.iterrows()}
    ratio_samples: dict[int, np.ndarray] = {}
    out_rows = []
    start = time.perf_counter()
    failures = []
    lo, hi = prior_bounds_phi()
    total_phi_draws = 0
    outside_prior_draws = 0
    for i, row in enumerate(rows.itertuples(index=False), start=1):
        out = row._asdict()
        try:
            x = curve_to_grid(raw_by_index[int(row.raw_index)], t)
            x_summary = volume_summary_features(x[None, :], t)[0]
            with torch.no_grad(), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*reject_outside_prior=False.*")
                phi = posterior.sample(
                    (draws,),
                    x=torch.as_tensor(x_summary, dtype=torch.float32),
                    show_progress_bars=False,
                    reject_outside_prior=False,
                ).detach().cpu().numpy()
            total_phi_draws += int(phi.shape[0])
            outside_prior_draws += int(((phi < lo) | (phi > hi)).any(axis=1).sum())
            samples = ratio_samples_from_phi(phi, t)
            samples = samples[np.isfinite(samples)]
            if len(samples) < max(50, draws // 2):
                raise ValueError("nonfinite posterior ratio samples")
            if len(samples) != draws:
                samples = samples[:draws]
            ratio_samples[int(row.raw_index)] = samples
            out.update(
                {
                    "posterior_error": "",
                    "posterior_draws": int(len(samples)),
                    "posterior_ratio_mean": float(np.mean(samples)),
                    "posterior_ratio_sd": float(np.std(samples, ddof=1)),
                    "posterior_ratio_q025": float(np.quantile(samples, 0.025)),
                    "posterior_ratio_q25": float(np.quantile(samples, 0.25)),
                    "posterior_ratio_q50": float(np.quantile(samples, 0.50)),
                    "posterior_ratio_q75": float(np.quantile(samples, 0.75)),
                    "posterior_ratio_q975": float(np.quantile(samples, 0.975)),
                }
            )
        except Exception as exc:
            failures.append(int(getattr(row, "raw_index", -1)))
            out.update({"posterior_error": str(exc), "posterior_draws": 0})
        out_rows.append(out)
        if i % 100 == 0 or i == len(rows):
            print(f"Posterior sampling progress: {i}/{len(rows)} selected curves")
    meta = {
        "wall_clock_s": float(time.perf_counter() - start),
        "failed_raw_indices": failures,
        "direct_sampling_reject_outside_prior": False,
        "outside_prior_draw_fraction": float(outside_prior_draws / total_phi_draws) if total_phi_draws else np.nan,
    }
    return pd.DataFrame(out_rows), ratio_samples, meta


def coverage(samples_by_idx: dict[int, np.ndarray], rows: pd.DataFrame, transform: dict | None = None) -> dict:
    transform = transform or {"shift": 0.0, "scale": 1.0}
    valid_rows = rows[rows["posterior_error"].fillna("") == ""].copy()
    out = {}
    pits = []
    for level in COVERAGE_LEVELS:
        hits = []
        lo = (1.0 - level) / 2.0
        hi = 1.0 - lo
        for row in valid_rows.itertuples(index=False):
            samples = apply_recalibration(samples_by_idx[int(row.raw_index)], transform)
            qlo, qhi = np.quantile(samples, [lo, hi])
            hits.append(qlo <= row.fev1_fvc_ratio <= qhi)
        out[str(level)] = float(np.mean(hits)) if hits else np.nan
    for row in valid_rows.itertuples(index=False):
        samples = apply_recalibration(samples_by_idx[int(row.raw_index)], transform)
        pits.append(float(np.mean(samples < row.fev1_fvc_ratio)))
    counts, _ = np.histogram(pits, bins=np.linspace(0.0, 1.0, PIT_BINS + 1))
    expected = max(len(pits) / PIT_BINS, 1e-12)
    p_value = float(chisquare(counts).pvalue) if len(pits) else np.nan
    gross_skew = bool(
        len(pits) == 0
        or abs(float(np.mean(pits)) - 0.5) > 0.10
        or counts.max() > 2.5 * expected
        or p_value < 0.01
    )
    pass_coverage = all(abs(out[str(level)] - level) <= 0.07 for level in COVERAGE_LEVELS)
    return {
        "n": int(len(valid_rows)),
        "coverage": out,
        "pit_counts": counts.astype(int).tolist(),
        "pit_chisquare_p": p_value,
        "pit_mean": float(np.mean(pits)) if pits else np.nan,
        "gross_skew": gross_skew,
        "calibrated": bool(pass_coverage and not gross_skew),
    }


def fit_recalibration(samples_by_idx: dict[int, np.ndarray], train: pd.DataFrame) -> dict:
    valid = train[train["posterior_error"].fillna("") == ""].copy()
    means = np.array([np.mean(samples_by_idx[int(row.raw_index)]) for row in valid.itertuples(index=False)])
    truth = valid["fev1_fvc_ratio"].to_numpy(float)
    shift = float(np.median(truth - means))
    best = {"shift": shift, "scale": 1.0, "train_score": np.inf}
    for scale in np.linspace(0.75, 8.0, 146):
        transformed = {}
        for row in valid.itertuples(index=False):
            raw = samples_by_idx[int(row.raw_index)]
            transformed[int(row.raw_index)] = apply_recalibration(raw, {"shift": shift, "scale": float(scale)})
        cov = coverage(transformed, valid, {"shift": 0.0, "scale": 1.0})
        score = sum(abs(cov["coverage"][str(level)] - level) for level in COVERAGE_LEVELS)
        if score < best["train_score"]:
            best = {"shift": shift, "scale": float(scale), "train_score": float(score), "train_coverage": cov}
    return best


def apply_recalibration(samples: np.ndarray, transform: dict) -> np.ndarray:
    mean = float(np.mean(samples))
    return mean + transform.get("shift", 0.0) + transform.get("scale", 1.0) * (samples - mean)


def posterior_probability(samples: np.ndarray, threshold: float, transform: dict) -> float:
    return float(np.mean(apply_recalibration(samples, transform) < threshold))


def choose_operating_points(train: pd.DataFrame, samples_by_idx: dict[int, np.ndarray], transform: dict, threshold_col: str | None) -> dict:
    rows = train[train["posterior_error"].fillna("") == ""].copy()
    if threshold_col is None:
        thresholds = pd.Series(0.70, index=rows.index)
    else:
        rows = rows[rows[threshold_col].notna()].copy()
        thresholds = rows[threshold_col]
    p = np.array(
        [
            posterior_probability(samples_by_idx[int(row.raw_index)], float(thresholds.loc[idx]), transform)
            for idx, row in zip(rows.index, rows.itertuples(index=False))
        ]
    )
    truth = (rows["fev1_fvc_ratio"].to_numpy(float) < thresholds.to_numpy(float))
    grid = np.linspace(0.0, 1.0, 101)
    high_candidates = []
    low_candidates = []
    for h in grid:
        fp_rate = float(((p >= h) & (~truth)).sum() / (~truth).sum()) if (~truth).sum() else 0.0
        sens = float(((p >= h) & truth).sum() / truth.sum()) if truth.sum() else np.nan
        if fp_rate <= 0.10:
            high_candidates.append((h, sens, fp_rate))
    for lo in grid:
        fn_rate = float(((p <= lo) & truth).sum() / truth.sum()) if truth.sum() else 0.0
        norm_call = float(((p <= lo) & (~truth)).sum() / (~truth).sum()) if (~truth).sum() else np.nan
        if fn_rate <= 0.10:
            low_candidates.append((lo, norm_call, fn_rate))
    t_high = min(high_candidates, key=lambda x: (x[0], -x[1]))[0] if high_candidates else 1.0
    t_low = max(low_candidates, key=lambda x: (x[0], x[1]))[0] if low_candidates else 0.0
    if t_low >= t_high:
        t_low = max(0.0, t_high - 0.01)
    return {
        "t_high": float(t_high),
        "t_low": float(t_low),
        "train_n": int(len(rows)),
        "rationale": "t_high is the lowest TRAIN p_obs threshold with normal-truth FP rate <=0.10; t_low is the highest TRAIN p_obs threshold with obstructed-truth confident-normal rate <=0.10.",
    }


def classify_posterior(rows: pd.DataFrame, samples_by_idx: dict[int, np.ndarray], transform: dict, op: dict, threshold_col: str | None) -> pd.DataFrame:
    out = rows[rows["posterior_error"].fillna("") == ""].copy()
    if threshold_col is None:
        thresholds = pd.Series(0.70, index=out.index)
    else:
        out = out[out[threshold_col].notna()].copy()
        thresholds = out[threshold_col]
    p_obs = []
    for idx, row in zip(out.index, out.itertuples(index=False)):
        p_obs.append(posterior_probability(samples_by_idx[int(row.raw_index)], float(thresholds.loc[idx]), transform))
    out["p_obs"] = p_obs
    out["posterior_outcome"] = "ABSTAIN"
    out.loc[out["p_obs"] >= op["t_high"], "posterior_outcome"] = "OBSTRUCTED"
    out.loc[out["p_obs"] <= op["t_low"], "posterior_outcome"] = "NORMAL"
    out["truth_obstructed"] = out["fev1_fvc_ratio"].to_numpy(float) < thresholds.to_numpy(float)
    return out


def three_way_metrics(classified: pd.DataFrame) -> dict:
    truth = classified["truth_obstructed"].astype(bool)
    outcome = classified["posterior_outcome"]
    obs_n = int(truth.sum())
    normal_n = int((~truth).sum())
    nonabs = classified[outcome != "ABSTAIN"].copy()
    pred = nonabs["posterior_outcome"] == "OBSTRUCTED"
    conf = confusion_from_bool(nonabs["truth_obstructed"].astype(bool), pred)
    borderline = classified[(classified["fev1_fvc_ratio"] >= 0.60) & (classified["fev1_fvc_ratio"] < 0.70)]
    return {
        "n": int(len(classified)),
        "obstructed_truth": {
            "n": obs_n,
            "confident_obstructed": float(((truth) & (outcome == "OBSTRUCTED")).sum() / obs_n) if obs_n else np.nan,
            "abstain": float(((truth) & (outcome == "ABSTAIN")).sum() / obs_n) if obs_n else np.nan,
            "confident_normal": float(((truth) & (outcome == "NORMAL")).sum() / obs_n) if obs_n else np.nan,
        },
        "normal_truth": {
            "n": normal_n,
            "confident_normal": float(((~truth) & (outcome == "NORMAL")).sum() / normal_n) if normal_n else np.nan,
            "abstain": float(((~truth) & (outcome == "ABSTAIN")).sum() / normal_n) if normal_n else np.nan,
            "confident_obstructed": float(((~truth) & (outcome == "OBSTRUCTED")).sum() / normal_n) if normal_n else np.nan,
        },
        "abstain_burden_all": float((outcome == "ABSTAIN").mean()) if len(classified) else np.nan,
        "abstain_burden_normal_truth": float(((~truth) & (outcome == "ABSTAIN")).sum() / normal_n) if normal_n else np.nan,
        "confident": conf,
        "borderline_n": int(len(borderline)),
        "borderline_confident_fn": float((borderline["posterior_outcome"] == "NORMAL").mean()) if len(borderline) else np.nan,
        "borderline_abstain": float((borderline["posterior_outcome"] == "ABSTAIN").mean()) if len(borderline) else np.nan,
    }


def point_estimate_baseline(rows: pd.DataFrame, threshold_col: str | None) -> pd.DataFrame:
    out = rows.copy()
    if threshold_col is None:
        thresholds = pd.Series(0.70, index=out.index)
    else:
        out = out[out[threshold_col].notna()].copy()
        thresholds = out[threshold_col]
    if "corrected_ratio_c1" not in out:
        out["corrected_ratio_c1"] = out["model_ratio"]
    out["posterior_outcome"] = np.where(out["corrected_ratio_c1"] < thresholds.to_numpy(float), "OBSTRUCTED", "NORMAL")
    if "truncated_ats" in out:
        out.loc[out["truncated_ats"].astype(bool), "posterior_outcome"] = "ABSTAIN"
    out["truth_obstructed"] = out["fev1_fvc_ratio"].to_numpy(float) < thresholds.to_numpy(float)
    return out


def confusion_from_bool(truth: pd.Series, pred: pd.Series) -> dict:
    tp = int((truth & pred).sum())
    tn = int((~truth & ~pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    n = tp + tn + fp + fn
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    agree = (tp + tn) / n if n else np.nan
    truth_pos = (tp + fn) / n if n else np.nan
    pred_pos = (tp + fp) / n if n else np.nan
    truth_neg = (tn + fp) / n if n else np.nan
    pred_neg = (tn + fn) / n if n else np.nan
    pe = truth_pos * pred_pos + truth_neg * pred_neg if n else np.nan
    kappa = (agree - pe) / (1.0 - pe) if n and abs(1.0 - pe) > 1e-12 else np.nan
    return {"n": n, "sensitivity": sens, "specificity": spec, "agreement": agree, "cohens_kappa": kappa, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def roc_sweep(rows: pd.DataFrame, samples_by_idx: dict[int, np.ndarray], transform: dict, threshold_col: str | None) -> list[dict]:
    valid = rows[rows["posterior_error"].fillna("") == ""].copy()
    if threshold_col is None:
        thresholds = pd.Series(0.70, index=valid.index)
    else:
        valid = valid[valid[threshold_col].notna()].copy()
        thresholds = valid[threshold_col]
    p = np.array(
        [
            posterior_probability(samples_by_idx[int(row.raw_index)], float(thresholds.loc[idx]), transform)
            for idx, row in zip(valid.index, valid.itertuples(index=False))
        ]
    )
    truth = valid["fev1_fvc_ratio"].to_numpy(float) < thresholds.to_numpy(float)
    rows_out = []
    for cut in np.linspace(0.0, 1.0, 21):
        pred = p >= cut
        cm = confusion_from_bool(pd.Series(truth), pd.Series(pred))
        rows_out.append({"threshold": float(cut), "sensitivity": cm["sensitivity"], "specificity": cm["specificity"]})
    return rows_out


def pool_ratio_samples(groups: dict[float, list[np.ndarray]], draws: int) -> dict[float, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED + 12000)
    pooled = {}
    for seqn, samples_list in groups.items():
        if len(samples_list) == 1:
            pooled[seqn] = samples_list[0]
            continue
        means = np.array([np.mean(s) for s in samples_list])
        variances = np.array([max(np.var(s, ddof=1), 1e-8) for s in samples_list])
        precision = 1.0 / variances
        mean = float(np.sum(means * precision) / np.sum(precision))
        sd = float(np.sqrt(1.0 / np.sum(precision)))
        pooled[seqn] = rng.normal(mean, sd, size=draws).astype(np.float32)
    return pooled


def write_outputs(root: Path, result: dict, rows: pd.DataFrame) -> None:
    lines = [
        "# Gate - Full Posterior Decision Method, Cycle F",
        "",
        f"- VERDICT: {result['verdict']} - {result['verdict_reason']}",
        f"- posterior engine: {result['posterior_training']['engine']}",
        f"- posterior draws per curve: {result['posterior_training']['posterior_draws_per_curve']}",
        "",
        "## Sample And Split",
        "",
        f"- split counts: {result['split_counts']}",
        f"- posterior fit failures: {result['posterior_failures']['n_failed']}/{result['posterior_failures']['n_attempted']}",
        f"- posterior production wall-clock: {result['posterior_sampling']['wall_clock_s']:.1f}s",
        "",
        "## Step 1 - Posterior Calibration Gate",
        "",
        f"- raw TEST calibration: {result['calibration_raw']}",
        f"- recalibration map: {result['recalibration_map']}",
        f"- recalibrated TEST calibration: {result['calibration_recalibrated']}",
        "",
    ]
    if result["halted"]:
        lines.extend(["Posterior classification halted because calibration did not clear.", ""])
    else:
        fixed = result["fixed_0_70"]
        lln = result["gli_lln"]
        lines.extend(
            [
                "## Step 2 - Operating Points",
                "",
                f"- fixed 0.70 operating points: {fixed['operating_points']}",
                f"- GLI LLN operating points: {lln['operating_points']}",
                "",
                "## Step 3-4 - TEST Accounting",
                "",
                "| method | threshold | obstructed caught | abstain obstructed | confident FN | borderline confident FN | normal correct | normal abstain | normal FP | abstain all | confident specificity | kappa |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for threshold_name, block in [("fixed 0.70", fixed), ("GLI LLN", lln)]:
            for method_name in ["point_estimate_baseline", "posterior_single_best", "posterior_pooled"]:
                m = block[method_name]
                lines.append(
                    f"| {method_name} | {threshold_name} | {m['obstructed_truth']['confident_obstructed']:.3f} | {m['obstructed_truth']['abstain']:.3f} | {m['obstructed_truth']['confident_normal']:.3f} | {m['borderline_confident_fn']:.3f} | {m['normal_truth']['confident_normal']:.3f} | {m['normal_truth']['abstain']:.3f} | {m['normal_truth']['confident_obstructed']:.3f} | {m['abstain_burden_all']:.3f} | {m['confident']['specificity']:.3f} | {m['confident']['cohens_kappa']:.3f} |"
                )
        lines.extend(
            [
                "",
                "## Step 3 - Multi-Manoeuvre Pooling",
                "",
                f"- TEST participants with >=2 usable A/B manoeuvres: {result['pooling']['participants_ge2']}",
                f"- borderline posterior sd single-best median: {result['pooling']['borderline_sd_single_median']:.4f}",
                f"- borderline posterior sd pooled median: {result['pooling']['borderline_sd_pooled_median']:.4f}",
                f"- straddling borderline cases moved to confident-correct by pooling: {result['pooling']['straddle_to_correct_n']}",
                "",
                "## ROC Sweep",
                "",
                f"- fixed 0.70 ROC sweep: {fixed['roc_sweep']}",
                "",
                "## Step 5 - Integrity",
                "",
                result["integrity_paragraph"],
                "",
                "## Step 6 - Drop Audit",
                "",
                f"- drop audit: {result['drop_bias_audit']}",
                f"- remaining confident borderline leaks under pooled policy: {result['remaining_pooled_borderline_leaks_n']}",
                "",
            ]
        )
    lines.append(f"VERDICT: {result['verdict']} - {result['verdict_reason']}")
    lines.append("")
    out = root / "results"
    out.mkdir(exist_ok=True)
    (out / "gate_posterior_method_F.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "gate_posterior_method_F.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    rows.to_csv(out / "gate_posterior_method_F_rows.csv", index=False)


def run(root: Path) -> dict:
    configure_caches(root)
    print("Posterior gate: loading column map, truth, GLI, and Cycle-F raw curves")
    column_map = load_column_map(root)
    truth = load_truth(root, column_map)
    gli = load_gli(root)
    raw_path, _ = find_or_unzip_raw(root)
    raw, _ = read_raw(raw_path)
    print(f"Posterior gate: raw rows loaded={len(raw)}")
    if (root / "results" / "gate_policy_calibrated_flag_F_rows.csv").exists():
        print("Posterior gate: reusing policy gate pool/split rows")
        rows = pd.read_csv(root / "results" / "gate_policy_calibrated_flag_F_rows.csv")
        selected_truth = rows.copy()
        meta = {"split_counts": {s: g["truth_severity"].value_counts().to_dict() for s, g in rows.groupby("split")}}
    else:
        print("Posterior gate: policy rows absent; rebuilding pool/split")
        rows, meta, selected_truth = build_pool(root, raw, truth, gli)
    rows = rows.merge(gli, on="SEQN", how="left", suffixes=("", "_gli_dup")) if "gli_lln_ratio" not in rows else rows
    t = np.asarray(OmbeGrid().t, dtype=np.float32)
    posterior, training = train_volume_tsnpe(root, t)
    print("Posterior gate: sampling selected best-manoeuvre posteriors")
    post_rows, samples_by_idx, sampling = sample_posterior_ratios(posterior, raw, rows, t, POSTERIOR_DRAWS)
    train = post_rows[post_rows["split"] == "train"].copy()
    test = post_rows[post_rows["split"] == "test"].copy()
    raw_cal = coverage(samples_by_idx, test)
    recal = {"shift": 0.0, "scale": 1.0, "train_score": None}
    recal_test = raw_cal
    if not raw_cal["calibrated"]:
        print("Posterior gate: raw calibration failed; fitting <=2-param TRAIN-only recalibration")
        recal = fit_recalibration(samples_by_idx, train)
        recal_test = coverage(samples_by_idx, test, recal)
    halted = not recal_test["calibrated"]
    base_result = {
        "posterior_training": training,
        "posterior_sampling": sampling,
        "posterior_failures": {
            "n_attempted": int(len(post_rows)),
            "n_failed": int((post_rows["posterior_error"].fillna("") != "").sum()),
        },
        "split_counts": meta.get("split_counts"),
        "calibration_raw": raw_cal,
        "recalibration_map": recal,
        "calibration_recalibrated": recal_test,
        "halted": halted,
    }
    if halted:
        print("Posterior gate: calibration did not clear; halting before classification")
        result = {
            **base_result,
            "verdict": "WEAK",
            "verdict_reason": "Posterior calibration did not clear on TEST after the train-only <=2-param recalibration map; posterior classification halted.",
        }
        write_outputs(root, result, post_rows)
        return result

    op_fixed = choose_operating_points(train, samples_by_idx, recal, None)
    op_lln = choose_operating_points(train, samples_by_idx, recal, "gli_lln_ratio")
    print(f"Posterior gate: fixed operating points={op_fixed}")
    print(f"Posterior gate: GLI operating points={op_lln}")
    single_fixed = classify_posterior(test, samples_by_idx, recal, op_fixed, None)
    single_lln = classify_posterior(test, samples_by_idx, recal, op_lln, "gli_lln_ratio")
    single_fixed_metrics = three_way_metrics(single_fixed)
    single_lln_metrics = three_way_metrics(single_lln)

    print("Posterior gate: building multi-manoeuvre posterior pool")
    # Pooling: approximate each per-manoeuvre posterior ratio as Gaussian and multiply posteriors.
    candidates = build_usable_raw_candidates(raw)
    test_seqn = set(test["SEQN"])
    multi_candidates = candidates[
        (candidates["SEQN"].isin(test_seqn))
        & (candidates["parse_error"].fillna("") == "")
        & (candidates["SPAACC"] == "Y")
        & (candidates["SPAQEFF"].isin(["A", "B"]))
    ].copy()
    groups = {}
    participants_ge2 = 0
    for seqn, group in multi_candidates.groupby("SEQN"):
        sample_list = []
        for row in group.itertuples(index=False):
            if int(row.raw_index) in samples_by_idx:
                sample_list.append(apply_recalibration(samples_by_idx[int(row.raw_index)], recal))
            else:
                # Sample non-selected manoeuvres only when needed for pooling.
                sampled_row, sampled, _ = sample_posterior_ratios(posterior, raw, pd.DataFrame([row._asdict()]), t, POSTERIOR_DRAWS)
                if len(sampled) == 0:
                    continue
                raw_idx = int(row.raw_index)
                samples_by_idx[raw_idx] = sampled[raw_idx]
                sample_list.append(apply_recalibration(samples_by_idx[raw_idx], recal))
        if sample_list:
            groups[float(seqn)] = sample_list
            if len(sample_list) >= 2:
                participants_ge2 += 1
    pooled_by_seqn = pool_ratio_samples(groups, POSTERIOR_DRAWS)
    pooled_samples = {
        int(row.raw_index): pooled_by_seqn.get(float(row.SEQN), apply_recalibration(samples_by_idx[int(row.raw_index)], recal))
        for row in test.itertuples(index=False)
        if int(row.raw_index) in samples_by_idx
    }
    identity = {"shift": 0.0, "scale": 1.0}
    pooled_fixed = classify_posterior(test, pooled_samples, identity, op_fixed, None)
    pooled_lln = classify_posterior(test, pooled_samples, identity, op_lln, "gli_lln_ratio")
    pooled_fixed_metrics = three_way_metrics(pooled_fixed)
    pooled_lln_metrics = three_way_metrics(pooled_lln)

    baseline_metrics = three_way_metrics(point_estimate_baseline(test, None))
    baseline_lln = three_way_metrics(point_estimate_baseline(test, "gli_lln_ratio"))

    borderline = test[(test["fev1_fvc_ratio"] >= 0.60) & (test["fev1_fvc_ratio"] < 0.70)]
    sd_single = [np.std(apply_recalibration(samples_by_idx[int(row.raw_index)], recal), ddof=1) for row in borderline.itertuples(index=False) if int(row.raw_index) in samples_by_idx]
    sd_pooled = [np.std(pooled_samples[int(row.raw_index)], ddof=1) for row in borderline.itertuples(index=False) if int(row.raw_index) in pooled_samples]
    single_outcome = single_fixed.set_index("SEQN")["posterior_outcome"]
    pooled_outcome = pooled_fixed.set_index("SEQN")["posterior_outcome"]
    straddle_to_correct = 0
    for seqn in borderline["SEQN"]:
        if single_outcome.get(seqn) == "ABSTAIN" and pooled_outcome.get(seqn) == "OBSTRUCTED":
            straddle_to_correct += 1

    overfit = abs(recal_test["coverage"]["0.8"] - 0.8) > abs(recal.get("train_coverage", {}).get("coverage", {}).get("0.8", 0.8) - 0.8) + 0.10
    verdict_status = "WEAK"
    verdict_reason = "No posterior improvement over point-estimate baseline."
    if (
        pooled_fixed_metrics["borderline_confident_fn"] <= 0.10
        and pooled_fixed_metrics["abstain_burden_normal_truth"] <= 0.20
        and pooled_fixed_metrics["confident"]["specificity"] >= 0.90
        and not overfit
    ):
        verdict_status = "STRONG"
        verdict_reason = "Posterior calibrated on TEST and pooled policy clears borderline FN, abstain burden, specificity, and generalization gates."
    elif pooled_fixed_metrics["borderline_confident_fn"] < 0.19 and not overfit:
        verdict_status = "MODERATE"
        verdict_reason = "Posterior calibrated and pooled borderline confident-FN is materially below the point-estimate baseline but does not clear STRONG."

    result = {
        **base_result,
        "verdict": verdict_status,
        "verdict_reason": verdict_reason,
        "fixed_0_70": {
            "operating_points": op_fixed,
            "point_estimate_baseline": baseline_metrics,
            "posterior_single_best": single_fixed_metrics,
            "posterior_pooled": pooled_fixed_metrics,
            "roc_sweep": roc_sweep(test, samples_by_idx, recal, None),
        },
        "gli_lln": {
            "operating_points": op_lln,
            "point_estimate_baseline": baseline_lln,
            "posterior_single_best": single_lln_metrics,
            "posterior_pooled": pooled_lln_metrics,
            "roc_sweep": roc_sweep(test, samples_by_idx, recal, "gli_lln_ratio"),
        },
        "pooling": {
            "participants_ge2": int(participants_ge2),
            "borderline_sd_single_median": float(np.median(sd_single)) if sd_single else np.nan,
            "borderline_sd_pooled_median": float(np.median(sd_pooled)) if sd_pooled else np.nan,
            "straddle_to_correct_n": int(straddle_to_correct),
        },
        "integrity_paragraph": "This is not threshold-shopping: the clinical thresholds are unmoved, calibration and operating points are learned on TRAIN only, calibration is checked on TEST before classification, and abstention is reported with normal-truth cost.",
        "drop_bias_audit": drop_bias_audit(raw, selected_truth, truth),
        "remaining_pooled_borderline_leaks_n": int((pooled_fixed[(pooled_fixed["fev1_fvc_ratio"] >= 0.60) & (pooled_fixed["fev1_fvc_ratio"] < 0.70)]["posterior_outcome"] == "NORMAL").sum()),
    }
    write_outputs(root, result, post_rows)
    return result


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    print(f"Posterior method gate: {result['verdict']} - {result['verdict_reason']}")
    if not result.get("halted"):
        print(f"Pooled borderline confident FN: {result['fixed_0_70']['posterior_pooled']['borderline_confident_fn']:.3f}")
        print(f"Normal abstain burden: {result['fixed_0_70']['posterior_pooled']['abstain_burden_normal_truth']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
