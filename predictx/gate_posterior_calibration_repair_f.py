from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache" / "xdg"))

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import torch
from numpyro.infer import MCMC, NUTS
from scipy.stats import chisquare

from .gate5_sbc import PRIOR_RANGES
from .gate23_f import find_or_unzip_raw, read_raw
from .gate_clinical_classification_f import (
    RANDOM_SEED,
    extract_curve_for_fit,
    load_column_map,
    load_gli,
    load_truth,
)
from .gate_policy_calibrated_flag_f import build_pool
from .ombe import OmbeGrid, ombe_volume, phi_from_natural
from .tsnpe import features_from_phi

PARAM_NAMES = ("tau_f", "tau_s", "alpha", "fvc_l", "tau_o")
PHI_NAMES = ("log_tau_f", "log_tau_s", "logit_alpha", "log_fvc", "log_tau_o")
COVERAGE_LEVELS = (0.50, 0.80, 0.95)
PIT_BINS = 10
PIT_P_THRESHOLD = 0.01
VOLUME_SIGMA_L = 0.050

BASELINE_SIMS = 6_000
BASELINE_DRAWS = 400
REPAIR_SIMS = 40_000
REPAIR_DRAWS = 2_000
MAX_TRAIN_EPOCHS_BASELINE = 60
MAX_TRAIN_EPOCHS_REPAIR = 80

NUTS_SUBSET_N = 100
NUTS_WARMUP = 300
NUTS_DRAWS = 300
NUTS_CHAINS = 2
NUTS_GRID_STRIDE = 5

WIDE_PRIOR_RANGES = {
    "tau_f": (0.01, 1.00),
    "tau_s": (1.00, 20.00),
    "alpha": (0.001, 0.999),
    "fvc_l": (0.25, 12.00),
    "tau_o": (0.0005, 5.00),
}


def configure_caches(root: Path) -> None:
    cache = root / ".cache"
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def prior_bounds_phi_ranges(ranges: dict[str, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array(
        [
            np.log(ranges["tau_f"][0]),
            np.log(ranges["tau_s"][0]),
            np.log(ranges["alpha"][0] / (1.0 - ranges["alpha"][0])),
            np.log(ranges["fvc_l"][0]),
            np.log(ranges["tau_o"][0]),
        ],
        dtype=np.float32,
    )
    hi = np.array(
        [
            np.log(ranges["tau_f"][1]),
            np.log(ranges["tau_s"][1]),
            np.log(ranges["alpha"][1] / (1.0 - ranges["alpha"][1])),
            np.log(ranges["fvc_l"][1]),
            np.log(ranges["tau_o"][1]),
        ],
        dtype=np.float32,
    )
    return lo, hi


def sample_prior_phi_ranges(n: int, ranges: dict[str, tuple[float, float]], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tau_f = rng.uniform(*ranges["tau_f"], size=n)
    tau_s = rng.uniform(*ranges["tau_s"], size=n)
    bad = tau_s <= tau_f
    attempts = 0
    while np.any(bad):
        tau_s[bad] = rng.uniform(*ranges["tau_s"], size=int(bad.sum()))
        bad = tau_s <= tau_f
        attempts += 1
        if attempts > 50:
            raise RuntimeError("Could not sample tau_s > tau_f from requested ranges.")
    alpha = rng.uniform(*ranges["alpha"], size=n)
    fvc = rng.uniform(*ranges["fvc_l"], size=n)
    tau_o = rng.uniform(*ranges["tau_o"], size=n)
    return np.column_stack(
        [
            np.log(tau_f),
            np.log(tau_s),
            np.log(alpha / (1.0 - alpha)),
            np.log(fvc),
            np.log(tau_o),
        ]
    ).astype(np.float32)


def stable_seed(label: str, offset: int = 0) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return RANDOM_SEED + offset + int(digest[:8], 16) % 100_000


def natural_ranges_for_report(ranges: dict[str, tuple[float, float]]) -> dict[str, dict[str, float]]:
    return {name: {"low": float(lo), "high": float(hi)} for name, (lo, hi) in ranges.items()}


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
    return np.column_stack([v1, fvc, v1 / fvc, v05 / fvc, v2 / fvc, v3 / fvc, v6 / fvc, last_delta]).astype(
        np.float32
    )


def curve_to_grid(raw_row: Any, t: np.ndarray) -> np.ndarray:
    curve = extract_curve_for_fit(raw_row)
    return np.interp(t, curve["vol_t"], curve["vol_l"], left=0.0, right=curve["vol_l"][-1]).astype(np.float32)


def ratio_samples_from_phi(phi: np.ndarray, t: np.ndarray) -> np.ndarray:
    return features_from_phi(phi.astype(np.float32), t)["ratio"].astype(np.float32)


def train_snpe(
    root: Path,
    t: np.ndarray,
    ranges: dict[str, tuple[float, float]],
    n_sims: int,
    label: str,
    max_epochs: int,
) -> tuple[object, dict]:
    configure_caches(root)
    from sbi.inference import SNPE
    from sbi.neural_nets import posterior_nn
    from sbi.utils import BoxUniform

    torch.set_num_threads(4)
    lo, hi = prior_bounds_phi_ranges(ranges)
    prior = BoxUniform(low=torch.as_tensor(lo), high=torch.as_tensor(hi))
    density_builder = posterior_nn(
        model="nsf",
        z_score_theta="independent",
        z_score_x="independent",
        hidden_features=64,
        num_transforms=5,
    )
    inference = SNPE(prior=prior, density_estimator=density_builder, show_progress_bars=False)
    print(f"{label}: generating {n_sims} OMBE volume simulations")
    theta = sample_prior_phi_ranges(n_sims, ranges, seed=stable_seed(label)).astype(np.float32)
    clean = volume_from_phi(theta, t)
    rng = np.random.default_rng(stable_seed(label, 33_000))
    noisy = clean + rng.normal(0.0, VOLUME_SIGMA_L, size=clean.shape).astype(np.float32)
    x_summary = volume_summary_features(noisy, t)
    inference.append_simulations(torch.as_tensor(theta), torch.as_tensor(x_summary))
    print(f"{label}: training SNPE, context_dim={x_summary.shape[1]}, max_epochs={max_epochs}")
    start = time.perf_counter()
    estimator = inference.train(
        training_batch_size=512,
        learning_rate=5e-4,
        validation_fraction=0.1,
        stop_after_epochs=20,
        max_num_epochs=max_epochs,
        show_train_summary=False,
    )
    posterior = inference.build_posterior(estimator, prior=prior, sample_with="direct")
    elapsed = time.perf_counter() - start
    summary = getattr(inference, "_summary", {}) or {}
    return posterior, {
        "label": label,
        "engine": "sbi SNPE_C posterior_nn(model='nsf') on OMBE volume-summary context",
        "simulation_budget": int(n_sims),
        "volume_sigma_l": VOLUME_SIGMA_L,
        "max_train_epochs": int(max_epochs),
        "training_wall_clock_s": float(elapsed),
        "training_loss_final": float(summary.get("training_loss", [np.nan])[-1]) if summary.get("training_loss") else np.nan,
        "validation_loss_final": float(summary.get("validation_loss", [np.nan])[-1]) if summary.get("validation_loss") else np.nan,
    }


def sample_snpe(
    posterior: object,
    raw: pd.DataFrame,
    rows: pd.DataFrame,
    t: np.ndarray,
    ranges: dict[str, tuple[float, float]],
    draws: int,
    label: str,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[int, np.ndarray], dict]:
    raw_by_index = {int(idx): row for idx, row in raw.iterrows()}
    lo, hi = prior_bounds_phi_ranges(ranges)
    ratio_by_idx: dict[int, np.ndarray] = {}
    phi_by_idx: dict[int, np.ndarray] = {}
    out_rows = []
    failures = []
    outside = 0
    total = 0
    start = time.perf_counter()
    for i, row in enumerate(rows.itertuples(index=False), start=1):
        out = row._asdict()
        raw_idx = int(row.raw_index)
        try:
            x = curve_to_grid(raw_by_index[raw_idx], t)
            x_summary = volume_summary_features(x[None, :], t)[0]
            with torch.no_grad(), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*reject_outside_prior=False.*")
                phi = posterior.sample(
                    (draws,),
                    x=torch.as_tensor(x_summary, dtype=torch.float32),
                    show_progress_bars=False,
                    reject_outside_prior=False,
                ).detach().cpu().numpy()
            total += int(phi.shape[0])
            outside += int(((phi < lo) | (phi > hi)).any(axis=1).sum())
            ratio = ratio_samples_from_phi(phi, t)
            finite = np.isfinite(ratio)
            phi = phi[finite]
            ratio = ratio[finite]
            if len(ratio) < max(100, draws // 2):
                raise ValueError("too few finite posterior ratio draws")
            phi_by_idx[raw_idx] = phi.astype(np.float32)
            ratio_by_idx[raw_idx] = ratio.astype(np.float32)
            out.update(
                {
                    f"{label}_posterior_error": "",
                    f"{label}_posterior_draws": int(len(ratio)),
                    f"{label}_ratio_mean": float(np.mean(ratio)),
                    f"{label}_ratio_sd": float(np.std(ratio, ddof=1)),
                    f"{label}_ratio_q025": float(np.quantile(ratio, 0.025)),
                    f"{label}_ratio_q50": float(np.quantile(ratio, 0.50)),
                    f"{label}_ratio_q975": float(np.quantile(ratio, 0.975)),
                }
            )
        except Exception as exc:
            failures.append(raw_idx)
            out.update({f"{label}_posterior_error": str(exc), f"{label}_posterior_draws": 0})
        out_rows.append(out)
        if i % 100 == 0 or i == len(rows):
            print(f"{label}: posterior sampling {i}/{len(rows)} curves")
    meta = {
        "draws_per_curve": int(draws),
        "wall_clock_s": float(time.perf_counter() - start),
        "failed_raw_indices": failures,
        "outside_prior_draw_fraction": float(outside / total) if total else np.nan,
        "direct_sampling_reject_outside_prior": False,
    }
    return pd.DataFrame(out_rows), ratio_by_idx, phi_by_idx, meta


def pit_values(samples_by_idx: dict[int, np.ndarray], rows: pd.DataFrame) -> np.ndarray:
    pits = []
    for row in rows.itertuples(index=False):
        raw_idx = int(row.raw_index)
        if raw_idx in samples_by_idx and np.isfinite(row.fev1_fvc_ratio):
            pits.append(float(np.mean(samples_by_idx[raw_idx] < float(row.fev1_fvc_ratio))))
    return np.asarray(pits, dtype=float)


def make_pit_map(train_pits: np.ndarray) -> dict[str, Any]:
    u = np.sort(np.clip(train_pits[np.isfinite(train_pits)], 0.0, 1.0))
    n = len(u)
    y = (np.arange(n) + 0.5) / n
    x_ext = np.concatenate([[0.0], u, [1.0]])
    y_ext = np.concatenate([[0.0], y, [1.0]])
    return {
        "type": "empirical_monotonic_pit_recalibration",
        "n_train": int(n),
        "x_raw_pit_knots": x_ext.tolist(),
        "y_calibrated_cdf_knots": y_ext.tolist(),
    }


def apply_pit_map(u: np.ndarray, pit_map: dict[str, Any]) -> np.ndarray:
    return np.interp(np.clip(u, 0.0, 1.0), pit_map["x_raw_pit_knots"], pit_map["y_calibrated_cdf_knots"])


def inverse_pit_map(q: float | np.ndarray, pit_map: dict[str, Any]) -> np.ndarray:
    return np.interp(q, pit_map["y_calibrated_cdf_knots"], pit_map["x_raw_pit_knots"])


def calibration_eval(
    samples_by_idx: dict[int, np.ndarray],
    rows: pd.DataFrame,
    pit_map: dict[str, Any] | None = None,
) -> dict:
    valid = rows[[int(r.raw_index) in samples_by_idx and np.isfinite(r.fev1_fvc_ratio) for r in rows.itertuples(index=False)]]
    raw_pits = pit_values(samples_by_idx, valid)
    pits = apply_pit_map(raw_pits, pit_map) if pit_map is not None else raw_pits
    coverage = {}
    for level in COVERAGE_LEVELS:
        lo = (1.0 - level) / 2.0
        hi = 1.0 - lo
        if pit_map is not None:
            raw_lo, raw_hi = inverse_pit_map(np.array([lo, hi]), pit_map)
        else:
            raw_lo, raw_hi = lo, hi
        hits = []
        for row in valid.itertuples(index=False):
            samples = samples_by_idx[int(row.raw_index)]
            qlo, qhi = np.quantile(samples, [raw_lo, raw_hi])
            hits.append(qlo <= float(row.fev1_fvc_ratio) <= qhi)
        coverage[str(level)] = float(np.mean(hits)) if hits else np.nan
    counts, _ = np.histogram(pits, bins=np.linspace(0.0, 1.0, PIT_BINS + 1))
    expected = max(len(pits) / PIT_BINS, 1e-12)
    p_value = float(chisquare(counts).pvalue) if len(pits) else np.nan
    gross_skew = bool(
        len(pits) == 0
        or p_value < PIT_P_THRESHOLD
        or counts.max() > 2.5 * expected
        or abs(float(np.mean(pits)) - 0.5) > 0.10
    )
    coverage_pass = all(abs(coverage[str(level)] - level) <= 0.07 for level in COVERAGE_LEVELS)
    return {
        "n": int(len(valid)),
        "coverage": coverage,
        "pit_counts": counts.astype(int).tolist(),
        "pit_chisquare_p": p_value,
        "pit_p_threshold": PIT_P_THRESHOLD,
        "pit_mean": float(np.mean(pits)) if len(pits) else np.nan,
        "gross_skew": gross_skew,
        "calibrated": bool(coverage_pass and not gross_skew),
    }


def bound_mass(
    phi_by_idx: dict[int, np.ndarray],
    rows: pd.DataFrame,
    ranges: dict[str, tuple[float, float]],
) -> dict[str, dict[str, float]]:
    lo, hi = prior_bounds_phi_ranges(ranges)
    selected = [phi_by_idx[int(row.raw_index)] for row in rows.itertuples(index=False) if int(row.raw_index) in phi_by_idx]
    phi = np.concatenate(selected, axis=0) if selected else np.empty((0, 5), dtype=np.float32)
    out = {}
    for j, name in enumerate(PHI_NAMES):
        width = float(hi[j] - lo[j])
        near = 0.02 * width
        vals = phi[:, j] if len(phi) else np.array([])
        out[name] = {
            "n_draws": int(len(vals)),
            "below_low": float(np.mean(vals < lo[j])) if len(vals) else np.nan,
            "above_high": float(np.mean(vals > hi[j])) if len(vals) else np.nan,
            "outside": float(np.mean((vals < lo[j]) | (vals > hi[j]))) if len(vals) else np.nan,
            "near_low_within_2pct_width": float(np.mean((vals >= lo[j]) & (vals <= lo[j] + near))) if len(vals) else np.nan,
            "near_high_within_2pct_width": float(np.mean((vals <= hi[j]) & (vals >= hi[j] - near))) if len(vals) else np.nan,
        }
    return out


def load_pool(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    configure_caches(root)
    column_map = load_column_map(root)
    truth = load_truth(root, column_map)
    gli = load_gli(root)
    raw_path, _ = find_or_unzip_raw(root)
    raw, _ = read_raw(raw_path)
    policy_rows = root / "results" / "gate_policy_calibrated_flag_F_rows.csv"
    if policy_rows.exists():
        rows = pd.read_csv(policy_rows)
        selected_truth = rows.copy()
        meta = {"source": str(policy_rows), "split_counts": {s: g["truth_severity"].value_counts().to_dict() for s, g in rows.groupby("split")}}
    else:
        rows, meta, selected_truth = build_pool(root, raw, truth, gli)
    rows = rows.merge(gli, on="SEQN", how="left", suffixes=("", "_gli_dup")) if "gli_lln_ratio" not in rows else rows
    return rows, raw, truth, selected_truth, meta


def run_snpe_step(
    root: Path,
    raw: pd.DataFrame,
    rows: pd.DataFrame,
    t: np.ndarray,
    ranges: dict[str, tuple[float, float]],
    n_sims: int,
    draws: int,
    label: str,
    max_epochs: int,
) -> dict:
    posterior, training = train_snpe(root, t, ranges, n_sims, label, max_epochs)
    sample_rows, ratio_by_idx, phi_by_idx, sampling = sample_snpe(posterior, raw, rows, t, ranges, draws, label)
    train = sample_rows[sample_rows["split"] == "train"].copy()
    test = sample_rows[sample_rows["split"] == "test"].copy()
    return {
        "label": label,
        "training": training,
        "sampling": sampling,
        "train_calibration": calibration_eval(ratio_by_idx, train),
        "test_calibration": calibration_eval(ratio_by_idx, test),
        "train_bound_mass": bound_mass(phi_by_idx, train, ranges),
        "sample_rows": sample_rows,
        "ratio_by_idx": ratio_by_idx,
        "phi_by_idx": phi_by_idx,
    }


def nuts_volume_model(t: jnp.ndarray, ranges: dict[str, tuple[float, float]], y: jnp.ndarray | None = None) -> None:
    tau_f = numpyro.sample("tau_f", dist.Uniform(*ranges["tau_f"]))
    tau_s = numpyro.sample("tau_s", dist.Uniform(*ranges["tau_s"]))
    alpha = numpyro.sample("alpha", dist.Uniform(*ranges["alpha"]))
    fvc_l = numpyro.sample("fvc_l", dist.Uniform(*ranges["fvc_l"]))
    tau_o = numpyro.sample("tau_o", dist.Uniform(*ranges["tau_o"]))
    phi = phi_from_natural(tau_f, tau_s, alpha, fvc_l, tau_o)
    mu = ombe_volume(phi, t)
    numpyro.sample("y", dist.Normal(mu, VOLUME_SIGMA_L), obs=y)


def samples_to_phi(samples: dict[str, np.ndarray]) -> np.ndarray:
    tau_f = np.ravel(samples["tau_f"])
    tau_s = np.ravel(samples["tau_s"])
    alpha = np.clip(np.ravel(samples["alpha"]), 1e-6, 1.0 - 1e-6)
    fvc_l = np.ravel(samples["fvc_l"])
    tau_o = np.ravel(samples["tau_o"])
    return np.column_stack([np.log(tau_f), np.log(tau_s), np.log(alpha / (1.0 - alpha)), np.log(fvc_l), np.log(tau_o)]).astype(np.float32)


def compute_mcmc_diagnostics(mcmc: MCMC) -> dict:
    samples = mcmc.get_samples(group_by_chain=True)
    rhat_by_param = {}
    ess_by_param = {}
    for name in PARAM_NAMES:
        arr = np.asarray(samples[name], dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 4:
            rhat_by_param[name] = np.nan
            ess_by_param[name] = np.nan
            continue
        rhat_by_param[name] = split_rhat(arr)
        ess_by_param[name] = bulk_ess(arr)
    rhat_vals = np.array(list(rhat_by_param.values()), dtype=float)
    ess_vals = np.array(list(ess_by_param.values()), dtype=float)
    return {
        "max_rhat": float(np.nanmax(rhat_vals)) if np.isfinite(rhat_vals).any() else np.nan,
        "min_bulk_ess": float(np.nanmin(ess_vals)) if np.isfinite(ess_vals).any() else np.nan,
        "rhat_by_param": rhat_by_param,
        "bulk_ess_by_param": ess_by_param,
        "diagnostic_method": "in-script split R-hat and initial-positive-sequence bulk ESS approximation",
    }


def split_rhat(chains: np.ndarray) -> float:
    chains = np.asarray(chains, dtype=float)
    n_chains, n_draws = chains.shape
    half = n_draws // 2
    split = np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)
    m, n = split.shape
    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)
    within = float(chain_vars.mean())
    if within <= 0.0:
        return np.nan
    between = float(n * chain_means.var(ddof=1))
    var_hat = ((n - 1) / n) * within + between / n
    return float(np.sqrt(max(var_hat / within, 0.0)))


def autocorr_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return np.ones(len(x))
    corr = np.correlate(x, x, mode="full")[len(x) - 1 :]
    return corr / denom


def bulk_ess(chains: np.ndarray) -> float:
    chains = np.asarray(chains, dtype=float)
    n_chains, n_draws = chains.shape
    rhos = np.array([autocorr_1d(chain) for chain in chains])
    rho = np.nanmean(rhos, axis=0)
    positive_sum = 0.0
    for lag in range(1, len(rho), 2):
        pair = rho[lag]
        if lag + 1 < len(rho):
            pair += rho[lag + 1]
        if pair < 0.0:
            break
        positive_sum += pair
    tau = max(1.0 + 2.0 * positive_sum, 1.0)
    return float(n_chains * n_draws / tau)


def run_nuts_subset(
    raw: pd.DataFrame,
    test_rows: pd.DataFrame,
    snpe_ratio_by_idx: dict[int, np.ndarray],
    t_full: np.ndarray,
    ranges: dict[str, tuple[float, float]],
) -> dict:
    available = test_rows[[int(r.raw_index) in snpe_ratio_by_idx for r in test_rows.itertuples(index=False)]].copy()
    subset = available.sample(n=min(NUTS_SUBSET_N, len(available)), random_state=RANDOM_SEED + 44_000).copy()
    t_nuts = np.asarray(t_full[::NUTS_GRID_STRIDE], dtype=np.float32)
    raw_by_index = {int(idx): row for idx, row in raw.iterrows()}
    kernel = NUTS(lambda y=None: nuts_volume_model(jnp.asarray(t_nuts), ranges, y))
    ratio_by_idx: dict[int, np.ndarray] = {}
    diagnostics = []
    start = time.perf_counter()
    for i, row in enumerate(subset.itertuples(index=False), start=1):
        raw_idx = int(row.raw_index)
        y = curve_to_grid(raw_by_index[raw_idx], t_full)[::NUTS_GRID_STRIDE]
        mcmc = MCMC(
            kernel,
            num_warmup=NUTS_WARMUP,
            num_samples=NUTS_DRAWS,
            num_chains=NUTS_CHAINS,
            chain_method="sequential",
            progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(RANDOM_SEED + raw_idx), y=jnp.asarray(y))
        samples_chain = mcmc.get_samples(group_by_chain=True)
        diag = compute_mcmc_diagnostics(mcmc)
        diag["raw_index"] = raw_idx
        diagnostics.append(diag)
        phi = samples_to_phi({name: np.asarray(samples_chain[name]) for name in PARAM_NAMES})
        ratio_by_idx[raw_idx] = ratio_samples_from_phi(phi, t_full)
        if i % 10 == 0 or i == len(subset):
            print(f"NUTS subset progress: {i}/{len(subset)} curves")
    nuts_cal = calibration_eval(ratio_by_idx, subset)
    snpe_subset = {int(row.raw_index): snpe_ratio_by_idx[int(row.raw_index)] for row in subset.itertuples(index=False)}
    snpe_cal = calibration_eval(snpe_subset, subset)
    max_rhat = [d.get("max_rhat", np.nan) for d in diagnostics]
    min_ess = [d.get("min_bulk_ess", np.nan) for d in diagnostics]
    return {
        "n_subset": int(len(subset)),
        "warmup": NUTS_WARMUP,
        "draws": NUTS_DRAWS,
        "chains": NUTS_CHAINS,
        "grid_stride": NUTS_GRID_STRIDE,
        "grid_hz": float(100 / NUTS_GRID_STRIDE),
        "wall_clock_s": float(time.perf_counter() - start),
        "snpe_subset_calibration": snpe_cal,
        "nuts_subset_calibration": nuts_cal,
        "diagnostics_summary": {
            "max_rhat_median": float(np.nanmedian(max_rhat)) if max_rhat else np.nan,
            "max_rhat_max": float(np.nanmax(max_rhat)) if max_rhat else np.nan,
            "min_bulk_ess_median": float(np.nanmedian(min_ess)) if min_ess else np.nan,
            "min_bulk_ess_min": float(np.nanmin(min_ess)) if min_ess else np.nan,
        },
        "diagnostics": diagnostics,
    }


def write_report(root: Path, result: dict, rows: pd.DataFrame) -> None:
    out = root / "results"
    out.mkdir(exist_ok=True)
    lines = [
        "# Gate - Posterior Calibration Repair, Cycle F",
        "",
        f"- FINAL CALIBRATION VERDICT: {result['final_verdict']}",
        f"- responsible cause attribution: {result['cause_attribution']}",
        f"- posterior classification unblocked: {result['posterior_classification_unblocked']}",
        "",
        "## Sample",
        "",
        f"- source: {result['pool_meta'].get('source', 'rebuilt')}",
        f"- split counts: {result['pool_meta'].get('split_counts')}",
        "",
        "## Step A - Widen Prior",
        "",
        f"- current prior bounds: {result['step_a']['current_prior_bounds']}",
        f"- current TRAIN bound mass: {result['step_a']['current_train_bound_mass']}",
        f"- widened prior bounds: {result['step_a']['wide_prior_bounds']}",
        f"- widened outside-prior draw fraction: {result['step_a']['wide_sampling']['outside_prior_draw_fraction']:.6g}",
        f"- widened TEST calibration: {result['step_a']['wide_test_calibration']}",
        "",
        "## Step B - Increased Training Budget",
        "",
        f"- sims: {result['step_b']['training']['simulation_budget']}",
        f"- draws per curve: {result['step_b']['sampling']['draws_per_curve']}",
        f"- outside-prior draw fraction: {result['step_b']['sampling']['outside_prior_draw_fraction']:.6g}",
        f"- TEST calibration: {result['step_b']['test_calibration']}",
        "",
        "## Step C - NUTS Cross-Check",
        "",
        f"- status: {result['step_c']['status']}",
        f"- NUTS settings/result: {result['step_c']}",
        "",
        "## Step D - Shape-Correct Coverage Calibration",
        "",
        f"- status: {result['step_d']['status']}",
        f"- map form: {result['step_d'].get('map_form')}",
        f"- TRAIN calibration after map: {result['step_d'].get('train_calibration')}",
        f"- TEST calibration after map: {result['step_d'].get('test_calibration')}",
        "",
        "## Decision",
        "",
        result["decision_text"],
        "",
    ]
    (out / "gate_posterior_calibration_repair_F.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "gate_posterior_calibration_repair_F.json").write_text(json.dumps(to_jsonable(result), indent=2) + "\n", encoding="utf-8")
    rows.to_csv(out / "gate_posterior_calibration_repair_F_rows.csv", index=False)


def run(root: Path) -> dict:
    configure_caches(root)
    rows, raw, truth, _selected_truth, pool_meta = load_pool(root)
    t = np.asarray(OmbeGrid().t, dtype=np.float32)
    print(f"Calibration repair: pool rows={len(rows)}, raw rows={len(raw)}")

    baseline = run_snpe_step(
        root,
        raw,
        rows,
        t,
        PRIOR_RANGES,
        BASELINE_SIMS,
        BASELINE_DRAWS,
        "current_prior",
        MAX_TRAIN_EPOCHS_BASELINE,
    )
    wide = run_snpe_step(
        root,
        raw,
        rows,
        t,
        WIDE_PRIOR_RANGES,
        BASELINE_SIMS,
        BASELINE_DRAWS,
        "wide_prior",
        MAX_TRAIN_EPOCHS_BASELINE,
    )

    step_a = {
        "current_prior_bounds": natural_ranges_for_report(PRIOR_RANGES),
        "current_train_bound_mass": baseline["train_bound_mass"],
        "current_sampling": baseline["sampling"],
        "current_test_calibration": baseline["test_calibration"],
        "wide_prior_bounds": natural_ranges_for_report(WIDE_PRIOR_RANGES),
        "wide_sampling": wide["sampling"],
        "wide_train_bound_mass": wide["train_bound_mass"],
        "wide_train_calibration": wide["train_calibration"],
        "wide_test_calibration": wide["test_calibration"],
    }

    high_budget = run_snpe_step(
        root,
        raw,
        rows,
        t,
        WIDE_PRIOR_RANGES,
        REPAIR_SIMS,
        REPAIR_DRAWS,
        "wide_prior_40k",
        MAX_TRAIN_EPOCHS_REPAIR,
    )
    step_b = {
        "training": high_budget["training"],
        "sampling": high_budget["sampling"],
        "train_bound_mass": high_budget["train_bound_mass"],
        "train_calibration": high_budget["train_calibration"],
        "test_calibration": high_budget["test_calibration"],
        "overconfidence_moved_toward_nominal": bool(
            abs(high_budget["test_calibration"]["coverage"]["0.5"] - 0.5)
            < abs(wide["test_calibration"]["coverage"]["0.5"] - 0.5)
        ),
    }

    step_c: dict[str, Any]
    if high_budget["test_calibration"]["calibrated"]:
        step_c = {"status": "NOT-RUN", "reason": "Step B cleared calibration; NUTS cross-check not needed for repair isolation."}
    else:
        test_rows = high_budget["sample_rows"][high_budget["sample_rows"]["split"] == "test"].copy()
        step_c = {"status": "RAN", **run_nuts_subset(raw, test_rows, high_budget["ratio_by_idx"], t, WIDE_PRIOR_RANGES)}

    if high_budget["test_calibration"]["calibrated"]:
        step_d = {"status": "NOT-RUN", "reason": "Step B cleared calibration before shape calibration."}
        final_cal = high_budget["test_calibration"]
    else:
        train_rows = high_budget["sample_rows"][high_budget["sample_rows"]["split"] == "train"].copy()
        test_rows = high_budget["sample_rows"][high_budget["sample_rows"]["split"] == "test"].copy()
        train_pits = pit_values(high_budget["ratio_by_idx"], train_rows)
        pit_map = make_pit_map(train_pits)
        train_cal = calibration_eval(high_budget["ratio_by_idx"], train_rows, pit_map)
        test_cal = calibration_eval(high_budget["ratio_by_idx"], test_rows, pit_map)
        step_d = {
            "status": "RAN",
            "map_form": "empirical monotonic PIT recalibration: F_cal(y)=H_train(F_raw(y)); central intervals use H_train inverse quantiles",
            "map": {
                "type": pit_map["type"],
                "n_train": pit_map["n_train"],
                "n_knots": len(pit_map["x_raw_pit_knots"]),
            },
            "train_calibration": train_cal,
            "test_calibration": test_cal,
            "generalizes_train_to_test": bool(
                all(
                    abs(train_cal["coverage"][str(level)] - test_cal["coverage"][str(level)]) <= 0.10
                    for level in COVERAGE_LEVELS
                )
                and abs(train_cal["pit_mean"] - test_cal["pit_mean"]) <= 0.10
            ),
        }
        final_cal = test_cal

    final_verdict = "CALIBRATED" if final_cal["calibrated"] else "NOT-CALIBRATED"
    unblocked = bool(final_cal["calibrated"])
    if unblocked:
        decision_text = "Posterior calibration clears on TEST; posterior classification is now unblocked, but classification/ROC/pooling were not run in this calibration-only gate."
    else:
        decision_text = (
            "Posterior calibration does not clear after prior widening, 40k-simulation SNPE, NUTS cross-check, "
            "and TRAIN-only PIT shape calibration. Posterior classification remains blocked; the point-estimate "
            "method is the defensible ceiling for this model/data path until the posterior engine or likelihood is changed."
        )

    if step_b["test_calibration"]["calibrated"]:
        cause = "SNPE training budget and widened prior were sufficient."
    elif step_d.get("test_calibration", {}).get("calibrated"):
        cause = "Residual miscalibration was posterior-CDF shape error repairable by TRAIN-only monotonic PIT calibration."
    elif step_c.get("status") == "RAN" and step_c.get("nuts_subset_calibration", {}).get("calibrated") and not step_c.get("snpe_subset_calibration", {}).get("calibrated"):
        cause = "Amortized SNPE failure: NUTS calibrated on the same subset while SNPE did not."
    else:
        cause = "Not isolated to a simple prior-bound or training-budget issue; residual shape/model-likelihood mismatch remains."

    result = {
        "final_verdict": final_verdict,
        "posterior_classification_unblocked": unblocked,
        "cause_attribution": cause,
        "pool_meta": pool_meta,
        "constants": {
            "coverage_levels": COVERAGE_LEVELS,
            "pit_bins": PIT_BINS,
            "pit_p_threshold": PIT_P_THRESHOLD,
            "volume_sigma_l": VOLUME_SIGMA_L,
        },
        "step_a": step_a,
        "step_b": step_b,
        "step_c": step_c,
        "step_d": step_d,
        "decision_text": decision_text,
    }
    final_rows = high_budget["sample_rows"]
    write_report(root, result, final_rows)
    return result


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    print(f"Posterior calibration repair: {result['final_verdict']} - {result['cause_attribution']}")
    print(f"Posterior classification unblocked: {result['posterior_classification_unblocked']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
