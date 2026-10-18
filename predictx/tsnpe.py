from __future__ import annotations

import argparse
import importlib.util
import json
import os
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import torch
from PIL import Image, ImageDraw
from scipy.stats import chisquare

from .gate5_sbc import PRIOR_RANGES, SIGMA_LPS, ombe_numpyro_model
from .ombe import NATURAL_PARAM_NAMES, OmbeGrid, flow_integral_l, midrange_phi, sample_prior_phi, unpack_phi

BOUNDARY_STATEMENT = (
    "This validates that TSNPE approximates NUTS on curves drawn from the OMBE\n"
    "generative model. It does NOT establish that either engine fits real NHANES\n"
    "curves - that is Gate 3, which requires the raw-curve file and is outstanding.\n"
    "Modules 2 and 3 confirm the inference machinery is internally correct, not that\n"
    "OMBE describes real physiology."
)


@dataclass(frozen=True)
class TsnpeConfig:
    rounds: int = 5
    sims_per_round: int = 2000
    heldout: int = 200
    tsnpe_samples: int = 1000
    nuts_warmup: int = 2500
    nuts_draws: int = 2500
    nuts_chains: int = 4
    nuts_chain_method: str = "sequential"
    nuts_rhat_threshold: float = 1.01
    nuts_ess_threshold: float = 400.0
    max_nuts_convergence_fail_frac: float = 0.10
    stop_after_epochs: int = 20
    max_num_epochs: int = 2_147_483_647
    seed: int = 31013
    sigma_lps: float = SIGMA_LPS
    torch_threads: int = 4
    rerun_20k_on_converged_fail: bool = False


def configure_local_caches(root: Path) -> None:
    cache_root = root / ".cache"
    mpl = cache_root / "matplotlib"
    xdg = cache_root / "xdg"
    mpl.mkdir(parents=True, exist_ok=True)
    xdg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg))


class FlowCnnEmbedding(torch.nn.Module):
    def __init__(self, input_length: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.input_length = input_length
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(64, embedding_dim),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x[:, None, :]
        elif x.ndim != 3:
            raise ValueError(f"Expected x with 2 or 3 dims, got shape {tuple(x.shape)}")
        return self.net(x.float())


def assert_ombe_volume_conservation() -> None:
    phi = midrange_phi()
    t = OmbeGrid().t
    integral = float(flow_integral_l(phi, t))
    fvc = float(unpack_phi(phi)["fvc_l"])
    rel = abs(integral - fvc) / fvc
    if rel > 0.01:
        raise RuntimeError(f"OMBE volume conservation failed: integral={integral}, fvc={fvc}, rel={rel}")


def prior_bounds_phi() -> tuple[np.ndarray, np.ndarray]:
    lo = np.array(
        [
            np.log(PRIOR_RANGES["tau_f"][0]),
            np.log(PRIOR_RANGES["tau_s"][0]),
            np.log(PRIOR_RANGES["alpha"][0] / (1.0 - PRIOR_RANGES["alpha"][0])),
            np.log(PRIOR_RANGES["fvc_l"][0]),
            np.log(PRIOR_RANGES["tau_o"][0]),
        ],
        dtype=np.float32,
    )
    hi = np.array(
        [
            np.log(PRIOR_RANGES["tau_f"][1]),
            np.log(PRIOR_RANGES["tau_s"][1]),
            np.log(PRIOR_RANGES["alpha"][1] / (1.0 - PRIOR_RANGES["alpha"][1])),
            np.log(PRIOR_RANGES["fvc_l"][1]),
            np.log(PRIOR_RANGES["tau_o"][1]),
        ],
        dtype=np.float32,
    )
    return lo, hi


def simulate_noisy_flows(theta_phi: np.ndarray, t: np.ndarray, sigma_lps: float, seed: int) -> np.ndarray:
    theta_jax = jnp.asarray(theta_phi, dtype=jnp.float32)
    t_jax = jnp.asarray(t, dtype=jnp.float32)
    clean = np.asarray(jax.vmap(lambda p: ombe_numpy_flow_jax(p, t_jax))(theta_jax))
    rng = np.random.default_rng(seed)
    return (clean + rng.normal(0.0, sigma_lps, size=clean.shape)).astype(np.float32)


@jax.jit
def ombe_numpy_flow_jax(phi: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    from .ombe import ombe_flow

    return ombe_flow(phi, t)


def phi_to_natural(theta_phi: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "tau_f": np.exp(theta_phi[:, 0]),
        "tau_s": np.exp(theta_phi[:, 1]),
        "alpha": 1.0 / (1.0 + np.exp(-theta_phi[:, 2])),
        "fvc_l": np.exp(theta_phi[:, 3]),
        "tau_o": np.exp(theta_phi[:, 4]),
    }


def natural_dict_to_phi(samples: dict[str, np.ndarray]) -> np.ndarray:
    alpha = np.asarray(samples["alpha"])
    return np.column_stack(
        [
            np.log(np.asarray(samples["tau_f"])),
            np.log(np.asarray(samples["tau_s"])),
            np.log(alpha / (1.0 - alpha)),
            np.log(np.asarray(samples["fvc_l"])),
            np.log(np.asarray(samples["tau_o"])),
        ]
    ).astype(np.float32)


def features_from_phi(theta_phi: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
    nat = phi_to_natural(theta_phi)
    tau_f = nat["tau_f"][:, None]
    tau_s = nat["tau_s"][:, None]
    alpha = nat["alpha"][:, None]
    fvc_l = nat["fvc_l"][:, None]
    tau_o = nat["tau_o"][:, None]
    tt = t[None, :]
    base = alpha * np.exp(-tt / tau_f) / tau_f + (1.0 - alpha) * np.exp(-tt / tau_s) / tau_s
    rise = 1.0 - np.exp(-tt / tau_o)
    q = np.maximum(base * rise, 1e-12)
    area = np.trapezoid(q, t, axis=1)[:, None]
    flow = fvc_l * q / area
    dt = float(t[1] - t[0])
    increments = (flow[:, :-1] + flow[:, 1:]) * 0.5 * dt
    volume = np.concatenate([np.zeros((len(theta_phi), 1)), np.cumsum(increments, axis=1)], axis=1)
    fev1_idx = int(round(1.0 / dt))
    fev1 = volume[:, fev1_idx]
    fvc = nat["fvc_l"]
    return {"fev1_l": fev1, "ratio": fev1 / fvc}


def interval_overlap_fraction(a: tuple[float, float], b: tuple[float, float]) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    union_lo = min(a[0], b[0])
    union_hi = max(a[1], b[1])
    union = union_hi - union_lo
    if union <= 0.0:
        return 0.0
    return max(0.0, hi - lo) / union


def extract_loss_summary(inference: object) -> dict:
    summary = getattr(inference, "_summary", {}) or {}
    result = {}
    for key in ("training_loss", "validation_loss"):
        values = summary.get(key)
        if values is not None and len(values) > 0:
            result[key] = float(values[-1])
    for attr in ("_best_val_loss", "_val_loss"):
        value = getattr(inference, attr, None)
        if value is not None:
            try:
                result[attr.strip("_")] = float(value)
            except TypeError:
                pass
    return result


def dependency_versions() -> dict[str, str]:
    packages = ("sbi", "torch", "nflows", "zuko", "numpyro", "jax", "numpy", "scipy")
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def train_tsnpe(config: TsnpeConfig, t: np.ndarray, root: Path) -> tuple[object, dict]:
    if importlib.util.find_spec("sbi") is None:
        raise RuntimeError("The required `sbi` package is not installed.")

    configure_local_caches(root)
    from sbi.inference import SNPE
    from sbi.neural_nets import posterior_nn
    from sbi.utils import BoxUniform

    torch.set_num_threads(config.torch_threads)
    lo, hi = prior_bounds_phi()
    prior = BoxUniform(low=torch.as_tensor(lo), high=torch.as_tensor(hi))
    embedding = FlowCnnEmbedding(input_length=len(t), embedding_dim=64)
    density_builder = posterior_nn(
        model="nsf",
        z_score_theta="independent",
        z_score_x="none",
        hidden_features=64,
        num_transforms=5,
        embedding_net=embedding,
    )
    inference = SNPE(prior=prior, density_estimator=density_builder, show_progress_bars=False)

    simulation_budget = 0
    for round_idx in range(config.rounds):
        theta = sample_prior_phi(config.sims_per_round, seed=config.seed + round_idx)
        x = simulate_noisy_flows(
            theta,
            t,
            sigma_lps=config.sigma_lps,
            seed=config.seed + 10_000 + round_idx,
        )
        inference.append_simulations(
            torch.as_tensor(theta, dtype=torch.float32),
            torch.as_tensor(x, dtype=torch.float32),
        )
        simulation_budget += config.sims_per_round
        print(f"TSNPE training round {round_idx + 1}/{config.rounds}; cumulative simulations={simulation_budget}")

    density_estimator = inference.train(
        training_batch_size=200,
        learning_rate=5e-4,
        validation_fraction=0.1,
        stop_after_epochs=config.stop_after_epochs,
        max_num_epochs=config.max_num_epochs,
        show_train_summary=False,
    )
    posterior = inference.build_posterior(density_estimator, prior=prior, sample_with="direct")
    training = {
        "rounds": config.rounds,
        "sims_per_round": config.sims_per_round,
        "simulation_budget": simulation_budget,
        "density_estimator": "sbi SNPE_C posterior_nn(model='nsf')",
        "num_transforms": 5,
        "flow_type": "rational-quadratic spline NSF",
        "embedding": "1D-CNN to 64-dimensional summary",
        "proposal": "prior simulations in appended rounds; no observation-specific proposal",
        "loss_summary": extract_loss_summary(inference),
    }
    return posterior, training


def nuts_convergence_diagnostics(samples_by_chain: dict[str, np.ndarray], config: TsnpeConfig) -> dict:
    from numpyro.diagnostics import summary

    summaries = summary(samples_by_chain, group_by_chain=True)
    by_param = {}
    for name in NATURAL_PARAM_NAMES:
        item = summaries[name]
        rhat = float(item["r_hat"])
        ess = float(item["n_eff"])
        by_param[name] = {
            "r_hat": rhat,
            "ess": ess,
            "converged": bool(
                np.isfinite(rhat)
                and np.isfinite(ess)
                and rhat < config.nuts_rhat_threshold
                and ess > config.nuts_ess_threshold
            ),
        }
    max_rhat = max(item["r_hat"] for item in by_param.values())
    min_ess = min(item["ess"] for item in by_param.values())
    return {
        "by_param": by_param,
        "max_rhat": float(max_rhat),
        "min_ess": float(min_ess),
        "converged": bool(all(item["converged"] for item in by_param.values())),
        "rhat_threshold": config.nuts_rhat_threshold,
        "ess_threshold": config.nuts_ess_threshold,
    }


def run_nuts_for_observation(
    y: np.ndarray,
    t_jax: jnp.ndarray,
    seed: int,
    config: TsnpeConfig,
) -> tuple[np.ndarray, dict]:
    from numpyro.infer import MCMC, NUTS

    kernel = NUTS(lambda y=None: ombe_numpyro_model(t_jax, y))
    mcmc = MCMC(
        kernel,
        num_warmup=config.nuts_warmup,
        num_samples=config.nuts_draws,
        num_chains=config.nuts_chains,
        chain_method=config.nuts_chain_method,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), y=jnp.asarray(y, dtype=jnp.float32))
    samples_by_chain = {
        name: np.asarray(value)
        for name, value in mcmc.get_samples(group_by_chain=True).items()
        if name in NATURAL_PARAM_NAMES
    }
    diagnostics = nuts_convergence_diagnostics(samples_by_chain, config)
    flat_samples = {
        name: value.reshape((-1,) + value.shape[2:])
        for name, value in samples_by_chain.items()
    }
    return natural_dict_to_phi(flat_samples), diagnostics


def build_heldout_reference(config: TsnpeConfig, t: np.ndarray) -> dict:
    theta_truth = sample_prior_phi(config.heldout, seed=config.seed + 50_000)
    x_obs = simulate_noisy_flows(theta_truth, t, config.sigma_lps, seed=config.seed + 60_000)
    return {
        "theta_truth": theta_truth,
        "x_obs": x_obs,
        "truth_nat": phi_to_natural(theta_truth),
        "nuts_phi_by_idx": {},
        "nuts_diagnostics_by_idx": {},
        "failed_indices": [],
        "converged_indices": [],
    }


def compare_on_heldout(
    posterior: object,
    config: TsnpeConfig,
    t: np.ndarray,
    reference: dict | None = None,
    run_label: str = "initial",
) -> tuple[dict, dict]:
    reference = reference or build_heldout_reference(config, t)
    x_obs = reference["x_obs"]
    truth_nat = reference["truth_nat"]
    t_jax = jnp.asarray(t, dtype=jnp.float32)
    max_failed = int(np.floor(config.max_nuts_convergence_fail_frac * config.heldout))

    fev1_diff_ok = []
    ratio_diff_ok = []
    mean_abs_fev1_diff_l = []
    mean_abs_ratio_diff = []
    std_ratio_ok = {name: [] for name in NATURAL_PARAM_NAMES}
    ci_overlap_ok = {name: [] for name in NATURAL_PARAM_NAMES}
    ci_overlap_values = {name: [] for name in NATURAL_PARAM_NAMES}
    std_ratio_values = {name: [] for name in NATURAL_PARAM_NAMES}
    tsnpe_coverage = {name: [] for name in NATURAL_PARAM_NAMES}
    ranks = {name: [] for name in NATURAL_PARAM_NAMES}
    mean_abs_param_diff_rel = {name: [] for name in NATURAL_PARAM_NAMES}
    evaluated_indices = []

    for idx in range(config.heldout):
        if idx not in reference["nuts_diagnostics_by_idx"]:
            nuts_phi, nuts_diag = run_nuts_for_observation(
                x_obs[idx],
                t_jax,
                seed=config.seed + 70_000 + idx,
                config=config,
            )
            if not np.isfinite(nuts_phi).all():
                raise RuntimeError(f"NUTS returned non-finite samples at heldout index {idx}")
            reference["nuts_diagnostics_by_idx"][idx] = nuts_diag
            if nuts_diag["converged"]:
                reference["nuts_phi_by_idx"][idx] = nuts_phi
                reference["converged_indices"].append(idx)
            else:
                reference["failed_indices"].append(idx)
            if len(reference["failed_indices"]) > max_failed:
                comparison = stopped_for_nuts_convergence(config, reference, run_label)
                print(
                    f"Module 3 {run_label}: NUTS convergence gate failed after "
                    f"{len(reference['nuts_diagnostics_by_idx'])}/{config.heldout} curves"
                )
                return comparison, reference

        nuts_diag = reference["nuts_diagnostics_by_idx"][idx]
        if not nuts_diag["converged"]:
            if (idx + 1) % 10 == 0 or idx + 1 == config.heldout:
                print(
                    f"Module 3 {run_label}: NUTS checked {idx + 1}/{config.heldout}; "
                    f"converged={len(reference['converged_indices'])}, failed={len(reference['failed_indices'])}"
                )
            continue

        nuts_phi = reference["nuts_phi_by_idx"][idx]
        x_torch = torch.as_tensor(x_obs[idx], dtype=torch.float32)
        with torch.no_grad():
            tsnpe_phi = posterior.sample(
                (config.tsnpe_samples,),
                x=x_torch,
                show_progress_bars=False,
                reject_outside_prior=True,
        ).detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(tsnpe_phi).all():
            raise RuntimeError(f"TSNPE returned non-finite samples at heldout index {idx}")
        evaluated_indices.append(idx)

        tsnpe_features = features_from_phi(tsnpe_phi, t)
        nuts_features = features_from_phi(nuts_phi, t)
        fev1_diff = abs(float(np.mean(tsnpe_features["fev1_l"]) - np.mean(nuts_features["fev1_l"])))
        ratio_diff = abs(float(np.mean(tsnpe_features["ratio"]) - np.mean(nuts_features["ratio"])))
        mean_abs_fev1_diff_l.append(fev1_diff)
        mean_abs_ratio_diff.append(ratio_diff)
        fev1_diff_ok.append(fev1_diff < 0.050)
        ratio_diff_ok.append(ratio_diff < 0.020)

        tsnpe_nat = phi_to_natural(tsnpe_phi)
        nuts_nat = phi_to_natural(nuts_phi)
        for name in NATURAL_PARAM_NAMES:
            ts = tsnpe_nat[name]
            ns = nuts_nat[name]
            ts_std = float(np.std(ts, ddof=1))
            ns_std = float(np.std(ns, ddof=1))
            ratio = np.inf if ns_std <= 0.0 else ts_std / ns_std
            std_ratio_values[name].append(float(ratio))
            std_ratio_ok[name].append(0.7 <= ratio <= 1.5)
            ts_ci = tuple(np.quantile(ts, [0.05, 0.95]))
            ns_ci = tuple(np.quantile(ns, [0.05, 0.95]))
            overlap = interval_overlap_fraction(ts_ci, ns_ci)
            ci_overlap_values[name].append(float(overlap))
            ci_overlap_ok[name].append(overlap > 0.70)
            truth_value = float(truth_nat[name][idx])
            tsnpe_coverage[name].append(bool(ts_ci[0] <= truth_value <= ts_ci[1]))
            ranks[name].append(int(np.sum(ts < truth_value)))
            denominator = max(abs(truth_value), 1e-12)
            mean_abs_param_diff_rel[name].append(abs(float(np.mean(ts) - np.mean(ns))) / denominator)

        if (idx + 1) % 10 == 0 or idx + 1 == config.heldout:
            print(
                f"Module 3 {run_label}: heldout {idx + 1}/{config.heldout}; "
                f"NUTS converged={len(reference['converged_indices'])}, failed={len(reference['failed_indices'])}"
            )

    convergence = nuts_convergence_summary(config, reference)
    if not convergence["gate_passed"]:
        return stopped_for_nuts_convergence(config, reference, run_label), reference
    if not evaluated_indices:
        comparison = stopped_for_nuts_convergence(config, reference, run_label)
        comparison["reason"] = "No heldout curves had a converged NUTS reference; agreement metrics not emitted."
        return comparison, reference

    rank_results = {}
    rank_pass = 0
    rank_bins = np.linspace(0, config.tsnpe_samples, 21)
    for name in NATURAL_PARAM_NAMES:
        counts, edges = np.histogram(ranks[name], bins=rank_bins)
        p_value = float(chisquare(counts).pvalue)
        rank_results[name] = {
            "counts": counts.astype(int).tolist(),
            "edges": edges.tolist(),
            "chi_square_p": p_value,
        }
        if p_value > 0.05:
            rank_pass += 1

    per_param = {}
    worst_param_scores = {}
    for name in NATURAL_PARAM_NAMES:
        per_param[name] = {
            "std_ratio_median": float(np.median(std_ratio_values[name])),
            "std_ratio_iqr": [float(x) for x in np.percentile(std_ratio_values[name], [25, 75])],
            "std_ratio_in_0_7_1_5_fraction": float(np.mean(std_ratio_ok[name])),
            "ci_overlap_median": float(np.median(ci_overlap_values[name])),
            "ci_overlap_gt_0_70_fraction": float(np.mean(ci_overlap_ok[name])),
            "tsnpe_90pct_coverage": float(np.mean(tsnpe_coverage[name])),
            "sbc_rank_chi_square_p": rank_results[name]["chi_square_p"],
            "mean_abs_posterior_mean_diff_relative_to_truth": float(np.median(mean_abs_param_diff_rel[name])),
        }
        worst_param_scores[name] = (
            1.0 - per_param[name]["ci_overlap_median"]
            + abs(np.log(max(per_param[name]["std_ratio_median"], 1e-12)))
            + per_param[name]["mean_abs_posterior_mean_diff_relative_to_truth"]
        )
    worst_param = max(worst_param_scores, key=worst_param_scores.get)

    metrics = {
        "fev1_mean_diff_lt_50ml": {
            "fraction": float(np.mean(fev1_diff_ok)),
            "median_abs_diff_l": float(np.median(mean_abs_fev1_diff_l)),
            "pass": bool(np.mean(fev1_diff_ok) >= 0.85),
        },
        "ratio_mean_diff_lt_0_02": {
            "fraction": float(np.mean(ratio_diff_ok)),
            "median_abs_diff": float(np.median(mean_abs_ratio_diff)),
            "pass": bool(np.mean(ratio_diff_ok) >= 0.85),
        },
        "posterior_std_ratio_0_7_1_5": {
            "fraction": float(np.mean([v for values in std_ratio_ok.values() for v in values])),
            "pass": bool(np.mean([v for values in std_ratio_ok.values() for v in values]) >= 0.85),
        },
        "ci_90_overlap_gt_70pct": {
            "fraction": float(np.mean([v for values in ci_overlap_ok.values() for v in values])),
            "pass": bool(np.mean([v for values in ci_overlap_ok.values() for v in values]) >= 0.85),
        },
        "sbc_rank_chi_square_p_gt_0_05": {
            "params_passing": int(rank_pass),
            "pass": bool(rank_pass >= 4),
        },
        "tsnpe_90pct_coverage_ge_85pct": {
            "overall_coverage": float(np.mean([v for values in tsnpe_coverage.values() for v in values])),
            "pass": bool(np.mean([v for values in tsnpe_coverage.values() for v in values]) >= 0.85),
        },
    }
    metric_pass_count = sum(1 for item in metrics.values() if item["pass"])
    catastrophic = (
        metrics["ci_90_overlap_gt_70pct"]["fraction"] < 0.10
        or metrics["posterior_std_ratio_0_7_1_5"]["fraction"] < 0.10
    )
    comparison = {
        "run_label": run_label,
        "heldout": config.heldout,
        "heldout_evaluated": len(evaluated_indices),
        "evaluated_indices": evaluated_indices,
        "nuts_warmup": config.nuts_warmup,
        "nuts_draws": config.nuts_draws,
        "nuts_chains": config.nuts_chains,
        "nuts_chain_method": config.nuts_chain_method,
        "nuts_convergence": convergence,
        "tsnpe_samples": config.tsnpe_samples,
        "metrics": metrics,
        "metric_pass_count": metric_pass_count,
        "catastrophic_divergence": bool(catastrophic),
        "rank_results": rank_results,
        "per_parameter": per_param,
        "worst_parameter": worst_param,
        "tau_o_worst_expected": worst_param == "tau_o",
        "ratio_bias_flag": bool(metrics["ratio_mean_diff_lt_0_02"]["median_abs_diff"] > 0.020),
        "tau_f_persists_as_worst": worst_param == "tau_f",
        "mean_abs_fev1_diff_l_values": mean_abs_fev1_diff_l,
        "mean_abs_ratio_diff_values": mean_abs_ratio_diff,
    }
    return comparison, reference


def nuts_convergence_summary(config: TsnpeConfig, reference: dict) -> dict:
    diagnostics = reference["nuts_diagnostics_by_idx"]
    failed = list(reference["failed_indices"])
    converged = list(reference["converged_indices"])
    failed_fraction = len(failed) / config.heldout if config.heldout else 1.0
    max_rhat = None
    min_ess = None
    per_param_max_rhat = {name: -np.inf for name in NATURAL_PARAM_NAMES}
    per_param_min_ess = {name: np.inf for name in NATURAL_PARAM_NAMES}
    for diag in diagnostics.values():
        max_rhat = diag["max_rhat"] if max_rhat is None else max(max_rhat, diag["max_rhat"])
        min_ess = diag["min_ess"] if min_ess is None else min(min_ess, diag["min_ess"])
        for name in NATURAL_PARAM_NAMES:
            per_param_max_rhat[name] = max(per_param_max_rhat[name], diag["by_param"][name]["r_hat"])
            per_param_min_ess[name] = min(per_param_min_ess[name], diag["by_param"][name]["ess"])
    return {
        "attempted": len(diagnostics),
        "converged": len(converged),
        "failed": len(failed),
        "failed_fraction_of_heldout": float(failed_fraction),
        "max_allowed_failed_fraction": config.max_nuts_convergence_fail_frac,
        "rhat_threshold": config.nuts_rhat_threshold,
        "ess_threshold": config.nuts_ess_threshold,
        "gate_passed": bool(
            len(diagnostics) == config.heldout
            and failed_fraction <= config.max_nuts_convergence_fail_frac
        ),
        "failed_indices": failed,
        "converged_indices": converged,
        "max_rhat_observed": None if max_rhat is None else float(max_rhat),
        "min_ess_observed": None if min_ess is None else float(min_ess),
        "per_param_max_rhat": {
            name: None if value == -np.inf else float(value)
            for name, value in per_param_max_rhat.items()
        },
        "per_param_min_ess": {
            name: None if value == np.inf else float(value)
            for name, value in per_param_min_ess.items()
        },
    }


def stopped_for_nuts_convergence(config: TsnpeConfig, reference: dict, run_label: str) -> dict:
    return {
        "run_label": run_label,
        "heldout": config.heldout,
        "heldout_evaluated": 0,
        "nuts_warmup": config.nuts_warmup,
        "nuts_draws": config.nuts_draws,
        "nuts_chains": config.nuts_chains,
        "nuts_chain_method": config.nuts_chain_method,
        "nuts_convergence": nuts_convergence_summary(config, reference),
        "tsnpe_samples": config.tsnpe_samples,
        "metrics": None,
        "metric_pass_count": 0,
        "catastrophic_divergence": None,
        "rank_results": {},
        "per_parameter": {},
        "worst_parameter": None,
        "tau_o_worst_expected": None,
        "ratio_bias_flag": None,
        "tau_f_persists_as_worst": None,
        "reason": "NUTS convergence gate failed for >10% of heldout curves; agreement metrics not emitted.",
    }


def run_tsnpe(
    results_dir: Path,
    figures_dir: Path,
    config: TsnpeConfig | None = None,
    root: Path | None = None,
) -> dict:
    config = config or TsnpeConfig()
    root = root or Path.cwd()
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    prior_underwarmed = load_prior_underwarmed_result(results_dir / "mod3_tsnpe.json")
    numpyro.set_host_device_count(config.nuts_chains)
    assert_ombe_volume_conservation()
    if importlib.util.find_spec("sbi") is None:
        result = {
            "status": "FAIL",
            "reason": "The required `sbi` package is not installed; Module 3 cannot run.",
            "boundary_statement": BOUNDARY_STATEMENT,
            "environment": dependency_versions(),
            "prior_underwarmed_result": prior_underwarmed,
        }
        write_report(result, results_dir, figures_dir)
        write_stack_summary(results_dir, result)
        return result

    t = np.asarray(OmbeGrid().t, dtype=np.float32)
    posterior, training = train_tsnpe(config, t, root)
    comparison, reference = compare_on_heldout(posterior, config, t, run_label="converged_10k")
    result = build_module3_result(config, training, comparison, prior_underwarmed)

    if (
        result["status"] == "FAIL"
        and comparison.get("nuts_convergence", {}).get("gate_passed")
        and config.rerun_20k_on_converged_fail
        and config.rounds * config.sims_per_round < 20_000
    ):
        boosted = replace(config, rounds=10, sims_per_round=2000, rerun_20k_on_converged_fail=False)
        print("Module 3: converged 10k run failed; starting one 20k-simulation rerun.")
        posterior_20k, training_20k = train_tsnpe(boosted, t, root)
        comparison_20k, _ = compare_on_heldout(
            posterior_20k,
            boosted,
            t,
            reference=reference,
            run_label="budget_20k",
        )
        result_20k = build_module3_result(boosted, training_20k, comparison_20k, prior_underwarmed)
        result_20k["initial_converged_run"] = compact_run_result(result)
        result = result_20k
        if result["status"] == "FAIL":
            result["reason"] = (
                "TSNPE still failed against a converged NUTS baseline after the requested one-time 20k-simulation rerun."
            )
    write_report(result, results_dir, figures_dir)
    write_stack_summary(results_dir, result)
    return result


def build_module3_result(
    config: TsnpeConfig,
    training: dict,
    comparison: dict,
    prior_underwarmed: dict | None,
) -> dict:
    if not comparison.get("nuts_convergence", {}).get("gate_passed"):
        status = "INCONCLUSIVE"
        reason = comparison.get(
            "reason",
            "NUTS convergence gate failed; TSNPE-NUTS agreement metrics were not emitted.",
        )
    elif comparison["metric_pass_count"] >= 4 and not comparison["catastrophic_divergence"]:
        status = "PASS"
        reason = "TSNPE agreed with converged NUTS on >=4/6 synthetic metrics with no catastrophic divergence."
    else:
        status = "FAIL"
        reason = "TSNPE did not meet the >=4/6 synthetic agreement criterion against a converged NUTS baseline."
    result = {
        "status": status,
        "reason": reason,
        "boundary_statement": BOUNDARY_STATEMENT,
        "environment": dependency_versions(),
        "config": config.__dict__,
        "training": training,
        "comparison": comparison,
        "prior_underwarmed_result": prior_underwarmed,
    }
    return result


def load_prior_underwarmed_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    nested = result.get("prior_underwarmed_result")
    if nested:
        return nested
    comparison = result.get("comparison", {})
    config = result.get("config", {})
    if comparison.get("nuts_warmup") == 500 and comparison.get("nuts_draws") == 500:
        return compact_run_result(result)
    if config.get("nuts_warmup") == 500 and config.get("nuts_draws") == 500:
        return compact_run_result(result)
    return None


def compact_run_result(result: dict) -> dict:
    comparison = result.get("comparison", {})
    training = result.get("training", {})
    metrics = comparison.get("metrics")
    compact = {
        "status": result.get("status"),
        "reason": result.get("reason"),
        "training": {
            "rounds": training.get("rounds"),
            "sims_per_round": training.get("sims_per_round"),
            "simulation_budget": training.get("simulation_budget"),
            "loss_summary": training.get("loss_summary"),
        },
        "comparison": {
            "run_label": comparison.get("run_label"),
            "heldout": comparison.get("heldout"),
            "heldout_evaluated": comparison.get("heldout_evaluated", comparison.get("heldout")),
            "nuts_warmup": comparison.get("nuts_warmup"),
            "nuts_draws": comparison.get("nuts_draws"),
            "nuts_chains": comparison.get("nuts_chains", 1),
            "nuts_convergence": comparison.get("nuts_convergence"),
            "metric_pass_count": comparison.get("metric_pass_count"),
            "catastrophic_divergence": comparison.get("catastrophic_divergence"),
            "metrics": metrics,
            "worst_parameter": comparison.get("worst_parameter"),
            "ratio_bias_flag": comparison.get("ratio_bias_flag"),
            "tau_f_persists_as_worst": comparison.get("tau_f_persists_as_worst"),
        },
    }
    return compact


def write_report(result: dict, results_dir: Path, figures_dir: Path) -> None:
    (results_dir / "mod3_tsnpe.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Module 3 - TSNPE Training + NUTS Agreement",
        "",
        f"- status: {result['status']}",
        f"- reason: {result['reason']}",
        "",
        "## Boundary",
        "",
        result["boundary_statement"],
        "",
    ]
    if "training" in result:
        training = result["training"]
        comparison = result["comparison"]
        environment = result.get("environment", {})
        lines.extend(
            [
                "## Environment",
                "",
                f"- sbi: {environment.get('sbi', 'not recorded')}",
                f"- torch: {environment.get('torch', 'not recorded')}",
                f"- nflows: {environment.get('nflows', 'not recorded')}",
                f"- zuko: {environment.get('zuko', 'not recorded')}",
                f"- numpyro: {environment.get('numpyro', 'not recorded')}",
                f"- jax: {environment.get('jax', 'not recorded')}",
                "",
                "## Training",
                "",
                f"- sbi estimator: {training['density_estimator']}",
                f"- flow: {training['flow_type']}",
                f"- coupling/spline transforms: {training['num_transforms']}",
                f"- embedding: {training['embedding']}",
                f"- training rounds: {training['rounds']}",
                f"- simulations per round: {training['sims_per_round']}",
                f"- total simulation budget: {training['simulation_budget']}",
                f"- proposal note: {training['proposal']}",
                f"- final training loss: {format_optional(training['loss_summary'].get('training_loss'))}",
                f"- final validation loss: {format_optional(training['loss_summary'].get('validation_loss'))}",
                f"- best validation loss: {format_optional(training['loss_summary'].get('best_val_loss'))}",
            ]
        )
        append_prior_comparison(lines, result.get("prior_underwarmed_result"), comparison)
        if "initial_converged_run" in result:
            lines.extend(["", "## Initial Converged-NUTS Run", ""])
            append_comparison_summary(lines, result["initial_converged_run"]["comparison"])
        append_comparison_report(lines, comparison)
        if comparison.get("metrics") is None:
            write_failure_figure(result, figures_dir / "mod3_agreement.png")
        else:
            write_agreement_figure(result, figures_dir / "mod3_agreement.png")
    else:
        lines.extend(
            [
                "No FEV1-diff, ratio-diff, posterior-std, CI-overlap, SBC, or TSNPE coverage metrics were emitted.",
                "",
            ]
        )
        write_failure_figure(result, figures_dir / "mod3_agreement.png")
    (results_dir / "mod3_tsnpe.md").write_text("\n".join(lines), encoding="utf-8")


def append_prior_comparison(lines: list[str], prior: dict | None, current: dict) -> None:
    if not prior:
        return
    prior_comp = prior.get("comparison", {})
    prior_metrics = prior_comp.get("metrics") or {}
    current_metrics = current.get("metrics") or {}
    lines.extend(
        [
            "",
            "## Prior vs Converged-NUTS Baseline",
            "",
            "| metric | prior 500/500 single-chain NUTS | current converged-NUTS run |",
            "|---|---:|---:|",
            f"| NUTS warmup/draws/chains | {prior_comp.get('nuts_warmup')}/{prior_comp.get('nuts_draws')}/{prior_comp.get('nuts_chains', 1)} | {current.get('nuts_warmup')}/{current.get('nuts_draws')}/{current.get('nuts_chains')} |",
            f"| NUTS convergence gate | not checked | {current.get('nuts_convergence', {}).get('converged')}/{current.get('nuts_convergence', {}).get('attempted')} converged |",
            f"| metric pass count | {prior_comp.get('metric_pass_count')}/6 | {current.get('metric_pass_count')}/6 |",
            f"| FEV1 median diff | {metric_median(prior_metrics, 'fev1_mean_diff_lt_50ml', 'median_abs_diff_l', scale=1000, suffix=' mL')} | {metric_median(current_metrics, 'fev1_mean_diff_lt_50ml', 'median_abs_diff_l', scale=1000, suffix=' mL')} |",
            f"| FEV1/FVC median diff | {metric_median(prior_metrics, 'ratio_mean_diff_lt_0_02', 'median_abs_diff')} | {metric_median(current_metrics, 'ratio_mean_diff_lt_0_02', 'median_abs_diff')} |",
            f"| std-ratio pass fraction | {metric_fraction(prior_metrics, 'posterior_std_ratio_0_7_1_5')} | {metric_fraction(current_metrics, 'posterior_std_ratio_0_7_1_5')} |",
            f"| CI-overlap pass fraction | {metric_fraction(prior_metrics, 'ci_90_overlap_gt_70pct')} | {metric_fraction(current_metrics, 'ci_90_overlap_gt_70pct')} |",
            f"| TSNPE 90% coverage | {metric_coverage(prior_metrics)} | {metric_coverage(current_metrics)} |",
            f"| worst parameter | {prior_comp.get('worst_parameter')} | {current.get('worst_parameter')} |",
        ]
    )


def append_comparison_summary(lines: list[str], comparison: dict) -> None:
    convergence = comparison.get("nuts_convergence", {})
    lines.extend(
        [
            f"- run label: {comparison.get('run_label')}",
            f"- NUTS warmup/draws/chains: {comparison.get('nuts_warmup')}/{comparison.get('nuts_draws')}/{comparison.get('nuts_chains')}",
            f"- NUTS converged: {convergence.get('converged')}/{convergence.get('attempted')}",
            f"- metric pass count: {comparison.get('metric_pass_count')}/6",
            f"- worst parameter: {comparison.get('worst_parameter')}",
        ]
    )
    metrics = comparison.get("metrics")
    if metrics:
        lines.extend(
            [
                f"- FEV1/FVC median diff: {metrics['ratio_mean_diff_lt_0_02']['median_abs_diff']:.4f}",
                f"- std-ratio pass fraction: {metrics['posterior_std_ratio_0_7_1_5']['fraction']:.3f}",
                f"- CI-overlap pass fraction: {metrics['ci_90_overlap_gt_70pct']['fraction']:.3f}",
            ]
        )


def append_comparison_report(lines: list[str], comparison: dict) -> None:
    convergence = comparison.get("nuts_convergence", {})
    lines.extend(
        [
            "",
            "## NUTS Convergence Gate",
            "",
            f"- run label: {comparison.get('run_label')}",
            f"- held-out synthetic curves: {comparison['heldout']}",
            f"- NUTS warmup/draws/chains: {comparison['nuts_warmup']}/{comparison['nuts_draws']}/{comparison.get('nuts_chains')}",
            f"- NUTS chain method: {comparison.get('nuts_chain_method')}",
            f"- R-hat threshold: < {convergence.get('rhat_threshold')}",
            f"- ESS threshold: > {convergence.get('ess_threshold')}",
            f"- NUTS attempted: {convergence.get('attempted')}/{comparison['heldout']}",
            f"- NUTS converged: {convergence.get('converged')}/{comparison['heldout']}",
            f"- NUTS failed: {convergence.get('failed')}/{comparison['heldout']} ({100 * (convergence.get('failed_fraction_of_heldout') or 0):.1f}%)",
            f"- convergence gate passed: {convergence.get('gate_passed')}",
            f"- max R-hat observed: {format_optional(convergence.get('max_rhat_observed'))}",
            f"- min ESS observed: {format_optional(convergence.get('min_ess_observed'))}",
        ]
    )
    if comparison.get("metrics") is None:
        lines.extend(
            [
                "",
                "Agreement metrics were not emitted because the NUTS reference did not pass the convergence gate.",
                "",
            ]
        )
        return
    metrics = comparison["metrics"]
    lines.extend(
        [
            "",
            "## Agreement Metrics",
            "",
            f"- held-out curves with converged NUTS used for agreement: {comparison['heldout_evaluated']}",
            f"- TSNPE posterior samples per curve: {comparison['tsnpe_samples']}",
            f"- metric pass count: {comparison['metric_pass_count']}/6",
            f"- catastrophic divergence: {comparison['catastrophic_divergence']}",
            "",
            "| metric | value | pass |",
            "|---|---:|---|",
            f"| FEV1 mean diff <50 mL | {metrics['fev1_mean_diff_lt_50ml']['fraction']:.3f} (median {1000 * metrics['fev1_mean_diff_lt_50ml']['median_abs_diff_l']:.1f} mL) | {metrics['fev1_mean_diff_lt_50ml']['pass']} |",
            f"| FEV1/FVC mean diff <0.02 | {metrics['ratio_mean_diff_lt_0_02']['fraction']:.3f} (median {metrics['ratio_mean_diff_lt_0_02']['median_abs_diff']:.4f}) | {metrics['ratio_mean_diff_lt_0_02']['pass']} |",
            f"| posterior std ratio 0.7-1.5 | {metrics['posterior_std_ratio_0_7_1_5']['fraction']:.3f} | {metrics['posterior_std_ratio_0_7_1_5']['pass']} |",
            f"| 90% CI overlap >70% | {metrics['ci_90_overlap_gt_70pct']['fraction']:.3f} | {metrics['ci_90_overlap_gt_70pct']['pass']} |",
            f"| SBC rank chi-square p>0.05 | {metrics['sbc_rank_chi_square_p_gt_0_05']['params_passing']}/5 params | {metrics['sbc_rank_chi_square_p_gt_0_05']['pass']} |",
            f"| TSNPE 90% coverage >=85% | {metrics['tsnpe_90pct_coverage_ge_85pct']['overall_coverage']:.3f} | {metrics['tsnpe_90pct_coverage_ge_85pct']['pass']} |",
            "",
            "CI overlap is measured as intersection length divided by union length for each parameter's 90% credible interval, then aggregated across curve-parameter pairs.",
            "",
            "## Per-Parameter Agreement",
            "",
            "| parameter | std ratio median | std ratio pass frac | CI overlap median | CI pass frac | TSNPE 90% coverage | SBC p | median abs mean diff / truth |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in NATURAL_PARAM_NAMES:
        item = comparison["per_parameter"][name]
        lines.append(
            f"| `{name}` | {item['std_ratio_median']:.3f} | "
            f"{item['std_ratio_in_0_7_1_5_fraction']:.3f} | "
            f"{item['ci_overlap_median']:.3f} | "
            f"{item['ci_overlap_gt_0_70_fraction']:.3f} | "
            f"{item['tsnpe_90pct_coverage']:.3f} | "
            f"{item['sbc_rank_chi_square_p']:.6g} | "
            f"{item['mean_abs_posterior_mean_diff_relative_to_truth']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"FEV1/FVC median-diff bias flag (>0.02): {comparison['ratio_bias_flag']}.",
            f"Worst agreement parameter: `{comparison['worst_parameter']}`.",
            f"tau_f persists as worst against converged baseline: {comparison['tau_f_persists_as_worst']}.",
            f"tau_o worst as expected from sloppy-direction diagnostics: {comparison['tau_o_worst_expected']}.",
        ]
    )
    if comparison["tau_f_persists_as_worst"]:
        lines.append("If tau_f persists as worst, the CNN summary network losing fast-rise information is the likely cause; this run diagnoses that but does not change the architecture.")
    if not comparison["tau_o_worst_expected"]:
        lines.append("Flag: tau_o was not the worst parameter, which is inconsistent with the Gate 1 / Gate 5 / Module 2 pattern.")
    lines.append("")


def metric_fraction(metrics: dict, name: str) -> str:
    item = metrics.get(name)
    if not item:
        return "not emitted"
    return f"{item.get('fraction'):.3f}"


def metric_coverage(metrics: dict) -> str:
    item = metrics.get("tsnpe_90pct_coverage_ge_85pct")
    if not item:
        return "not emitted"
    return f"{item.get('overall_coverage'):.3f}"


def metric_median(metrics: dict, metric_name: str, value_name: str, scale: float = 1.0, suffix: str = "") -> str:
    item = metrics.get(metric_name)
    if not item or value_name not in item:
        return "not emitted"
    return f"{scale * item[value_name]:.4g}{suffix}"


def format_optional(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6g}"
    return "not available from sbi summary"


def write_agreement_figure(result: dict, path: Path) -> None:
    comparison = result["comparison"]
    image = Image.new("RGB", (1250, 760), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 30), "Module 3 TSNPE vs NUTS agreement", fill="black")
    metrics = comparison["metrics"]
    labels = [
        ("FEV1 diff", metrics["fev1_mean_diff_lt_50ml"]["fraction"]),
        ("Ratio diff", metrics["ratio_mean_diff_lt_0_02"]["fraction"]),
        ("Std ratio", metrics["posterior_std_ratio_0_7_1_5"]["fraction"]),
        ("CI overlap", metrics["ci_90_overlap_gt_70pct"]["fraction"]),
        ("Coverage", metrics["tsnpe_90pct_coverage_ge_85pct"]["overall_coverage"]),
    ]
    x0, y0, w, h = 80, 90, 700, 280
    draw.rectangle((x0, y0, x0 + w, y0 + h), outline="black")
    draw.line((x0, y0 + h - 0.85 * h, x0 + w, y0 + h - 0.85 * h), fill=(190, 50, 50), width=2)
    bar_w = 80
    for idx, (label, value) in enumerate(labels):
        px = x0 + 60 + idx * 125
        bh = max(0, min(h, value * h))
        color = (45, 115, 180) if value >= 0.85 else (210, 120, 45)
        draw.rectangle((px, y0 + h - bh, px + bar_w, y0 + h), fill=color)
        draw.text((px - 10, y0 + h + 18), label, fill="black")
        draw.text((px + 8, y0 + h - bh - 22), f"{value:.2f}", fill="black")
    draw.text((x0 + w + 15, y0 + h - 0.85 * h - 8), "0.85", fill=(190, 50, 50))

    px0, py0, pw, ph = 80, 450, 980, 220
    draw.rectangle((px0, py0, px0 + pw, py0 + ph), outline="black")
    max_score = max(
        item["mean_abs_posterior_mean_diff_relative_to_truth"]
        for item in comparison["per_parameter"].values()
    )
    max_score = max(max_score, 1e-6)
    for idx, name in enumerate(NATURAL_PARAM_NAMES):
        item = comparison["per_parameter"][name]
        value = item["mean_abs_posterior_mean_diff_relative_to_truth"]
        bw = (value / max_score) * (pw - 220)
        y = py0 + 25 + idx * 36
        color = (190, 60, 60) if name == comparison["worst_parameter"] else (75, 135, 90)
        draw.text((px0 + 15, y), name, fill="black")
        draw.rectangle((px0 + 170, y, px0 + 170 + bw, y + 18), fill=color)
        draw.text((px0 + 180 + bw, y), f"{value:.3f}", fill="black")
    draw.text((px0 + 250, py0 + ph + 25), "median abs posterior-mean difference / truth", fill="black")
    image.save(path)


def write_failure_figure(result: dict, path: Path) -> None:
    image = Image.new("RGB", (900, 420), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 880, 400), outline="black", width=2)
    draw.text((80, 185), f"Module 3 {result['status']}: {result['reason']}", fill="black")
    image.save(path)


def write_stack_summary(results_dir: Path, mod3: dict) -> None:
    mod1 = read_metric_doc(results_dir / "mod1_segmentation.md")
    mod2 = read_metric_doc(results_dir / "mod2_fit.md")
    lines = [
        "# Synthetic Stack Summary",
        "",
        f"- Module 1 segmentation: {mod1 if mod1 else 'see results/mod1_segmentation.md'}",
        f"- Module 2 least-squares fitter: {mod2 if mod2 else 'see results/mod2_fit.md'}",
        f"- Module 3 TSNPE: {mod3['status']}",
        f"  - reason: {mod3['reason']}",
    ]
    if "comparison" in mod3:
        comparison = mod3["comparison"]
        metrics = comparison["metrics"]
        convergence = comparison.get("nuts_convergence", {})
        lines.extend(
            [
                f"  - NUTS warmup/draws/chains: {comparison.get('nuts_warmup')}/{comparison.get('nuts_draws')}/{comparison.get('nuts_chains')}",
                f"  - NUTS converged: {convergence.get('converged')}/{convergence.get('attempted')}",
                f"  - metric pass count: {comparison['metric_pass_count']}/6",
                f"  - held-out curves evaluated: {comparison.get('heldout_evaluated')}/{comparison['heldout']}",
            ]
        )
        if metrics:
            lines.extend(
                [
                    f"  - FEV1 diff pass fraction: {metrics['fev1_mean_diff_lt_50ml']['fraction']:.3f}",
                    f"  - FEV1/FVC diff pass fraction: {metrics['ratio_mean_diff_lt_0_02']['fraction']:.3f}",
                    f"  - TSNPE 90% coverage: {metrics['tsnpe_90pct_coverage_ge_85pct']['overall_coverage']:.3f}",
                    f"  - worst agreement parameter: {comparison['worst_parameter']}",
                ]
            )
        else:
            lines.append("  - agreement metrics: not emitted because NUTS convergence gate failed")
    lines.extend(["", "## Boundary", "", mod3["boundary_statement"], ""])
    (results_dir / "SYNTHETIC_STACK_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def read_metric_doc(path: Path) -> str:
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("- status:"):
            return line.replace("- status:", "").strip()
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--sims-per-round", type=int, default=2000)
    parser.add_argument("--heldout", type=int, default=200)
    parser.add_argument("--tsnpe-samples", type=int, default=1000)
    parser.add_argument("--nuts-warmup", type=int, default=2500)
    parser.add_argument("--nuts-draws", type=int, default=2500)
    parser.add_argument("--nuts-chains", type=int, default=4)
    parser.add_argument("--nuts-chain-method", default="sequential", choices=["parallel", "sequential", "vectorized"])
    parser.add_argument("--nuts-rhat-threshold", type=float, default=1.01)
    parser.add_argument("--nuts-ess-threshold", type=float, default=400.0)
    parser.add_argument("--max-nuts-convergence-fail-frac", type=float, default=0.10)
    parser.add_argument("--stop-after-epochs", type=int, default=20)
    parser.add_argument("--max-num-epochs", type=int, default=2_147_483_647)
    parser.add_argument("--seed", type=int, default=31013)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--rerun-20k-on-converged-fail", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TsnpeConfig(
        rounds=args.rounds,
        sims_per_round=args.sims_per_round,
        heldout=args.heldout,
        tsnpe_samples=args.tsnpe_samples,
        nuts_warmup=args.nuts_warmup,
        nuts_draws=args.nuts_draws,
        nuts_chains=args.nuts_chains,
        nuts_chain_method=args.nuts_chain_method,
        nuts_rhat_threshold=args.nuts_rhat_threshold,
        nuts_ess_threshold=args.nuts_ess_threshold,
        max_nuts_convergence_fail_frac=args.max_nuts_convergence_fail_frac,
        stop_after_epochs=args.stop_after_epochs,
        max_num_epochs=args.max_num_epochs,
        seed=args.seed,
        torch_threads=args.torch_threads,
        rerun_20k_on_converged_fail=args.rerun_20k_on_converged_fail,
    )
    result = run_tsnpe(Path("results"), Path("figures"), config=config, root=Path.cwd())
    print(f"Module 3 {result['status']}: {result['reason']}")
    if "comparison" in result:
        metrics = result["comparison"]["metrics"]
        print(f"Metric pass count: {result['comparison']['metric_pass_count']}/6")
        convergence = result["comparison"].get("nuts_convergence", {})
        print(f"NUTS converged: {convergence.get('converged')}/{convergence.get('attempted')}")
        if metrics:
            print(f"FEV1 diff pass fraction: {metrics['fev1_mean_diff_lt_50ml']['fraction']:.3f}")
            print(f"FEV1/FVC diff pass fraction: {metrics['ratio_mean_diff_lt_0_02']['fraction']:.3f}")
            print(f"TSNPE 90% coverage: {metrics['tsnpe_90pct_coverage_ge_85pct']['overall_coverage']:.3f}")
            print(f"Worst parameter: {result['comparison']['worst_parameter']}")
        else:
            print("Agreement metrics not emitted because NUTS convergence gate failed.")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
