from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

PARAM_NAMES = ("log_tau_f", "log_tau_s", "logit_alpha", "log_fvc", "log_tau_o", "log_k_minus_1")
NATURAL_PARAM_NAMES = ("tau_f", "tau_s", "alpha", "fvc_l", "tau_o", "k")


@dataclass(frozen=True)
class OmbeGammaGrid:
    duration_s: float = 12.0
    hz: int = 100

    @property
    def t(self) -> jnp.ndarray:
        return jnp.arange(0.0, self.duration_s + 1.0 / self.hz, 1.0 / self.hz)


def unpack_phi(phi: jnp.ndarray) -> dict[str, jnp.ndarray]:
    return {
        "tau_f": jnp.exp(phi[0]),
        "tau_s": jnp.exp(phi[1]),
        "alpha": jax.nn.sigmoid(phi[2]),
        "fvc_l": jnp.exp(phi[3]),
        "tau_o": jnp.exp(phi[4]),
        "k": 1.0 + jnp.exp(phi[5]),
    }


def phi_from_natural(
    tau_f: jnp.ndarray,
    tau_s: jnp.ndarray,
    alpha: jnp.ndarray,
    fvc_l: jnp.ndarray,
    tau_o: jnp.ndarray,
    k: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.array(
        [
            jnp.log(tau_f),
            jnp.log(tau_s),
            jnp.log(alpha / (1.0 - alpha)),
            jnp.log(fvc_l),
            jnp.log(tau_o),
            jnp.log(k - 1.0),
        ]
    )


def ombe_gamma_flow(phi: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """Return OMBE-gamma flow in L/s.

    The gamma onset is a true rounded peak term:
      gamma(t) = (t/tau_o)^k exp(-t/tau_o)
      base(t) = alpha exp(-t/tau_f)/tau_f + (1-alpha) exp(-t/tau_s)/tau_s
      flow(t) = FVC * gamma(t) base(t) / integral(gamma base)

    Normalization keeps volume conservation exact on the grid used for fitting
    and Fisher calculations.
    """
    p = unpack_phi(phi)
    x = jnp.maximum(t / p["tau_o"], 1e-12)
    gamma = jnp.where(t > 0.0, jnp.power(x, p["k"]) * jnp.exp(-x), 0.0)
    base = (
        p["alpha"] * jnp.exp(-t / p["tau_f"]) / p["tau_f"]
        + (1.0 - p["alpha"]) * jnp.exp(-t / p["tau_s"]) / p["tau_s"]
    )
    q = jnp.maximum(gamma * base, 1e-30)
    area = jnp.trapezoid(q, t)
    return p["fvc_l"] * q / area


def ombe_gamma_volume(phi: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    flow = ombe_gamma_flow(phi, t)
    increments = (flow[:-1] + flow[1:]) * 0.5 * jnp.diff(t)
    return jnp.concatenate([jnp.array([0.0]), jnp.cumsum(increments)])


def flow_integral_l(phi: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    return jnp.trapezoid(ombe_gamma_flow(phi, t), t)


def sample_prior_phi(n: int, seed: int = 0) -> np.ndarray:
    """Sample the documented 6D OMBE-gamma prior in transformed coordinates."""
    rng = np.random.default_rng(seed)
    tau_f = rng.uniform(0.15, 0.80, size=n)
    tau_s = rng.uniform(1.00, 5.00, size=n)
    alpha = rng.uniform(0.25, 0.85, size=n)
    fvc = rng.uniform(2.00, 6.00, size=n)
    tau_o = rng.uniform(0.03, 0.40, size=n)
    k = rng.uniform(1.05, 5.00, size=n)
    phi = np.column_stack(
        [
            np.log(tau_f),
            np.log(tau_s),
            np.log(alpha / (1.0 - alpha)),
            np.log(fvc),
            np.log(tau_o),
            np.log(k - 1.0),
        ]
    )
    if not np.all(phi[:, 0] < phi[:, 1]):
        raise RuntimeError("Prior violated tau_f < tau_s constraint.")
    return phi


def midrange_phi() -> jnp.ndarray:
    return phi_from_natural(
        jnp.array(0.35),
        jnp.array(2.50),
        jnp.array(0.60),
        jnp.array(4.00),
        jnp.array(0.12),
        jnp.array(2.50),
    )


def fisher_information(phi: jnp.ndarray, t: jnp.ndarray, sigma_lps: float) -> jnp.ndarray:
    jac = jax.jacfwd(lambda p: ombe_gamma_flow(p, t))(phi)
    return (jac.T @ jac) / (sigma_lps**2)


batched_fisher_information = jax.jit(jax.vmap(fisher_information, in_axes=(0, None, None)))
