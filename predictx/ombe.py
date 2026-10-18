from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

PARAM_NAMES = ("log_tau_f", "log_tau_s", "logit_alpha", "log_fvc", "log_tau_o")
NATURAL_PARAM_NAMES = ("tau_f", "tau_s", "alpha", "fvc_l", "tau_o")


@dataclass(frozen=True)
class OmbeGrid:
    duration_s: float = 12.0
    hz: int = 100

    @property
    def t(self) -> jnp.ndarray:
        return jnp.arange(0.0, self.duration_s + 1.0 / self.hz, 1.0 / self.hz)


def unpack_phi(phi: jnp.ndarray) -> dict[str, jnp.ndarray]:
    tau_f = jnp.exp(phi[0])
    tau_s = jnp.exp(phi[1])
    alpha = jax.nn.sigmoid(phi[2])
    fvc_l = jnp.exp(phi[3])
    tau_o = jnp.exp(phi[4])
    return {
        "tau_f": tau_f,
        "tau_s": tau_s,
        "alpha": alpha,
        "fvc_l": fvc_l,
        "tau_o": tau_o,
    }


def phi_from_natural(
    tau_f: jnp.ndarray,
    tau_s: jnp.ndarray,
    alpha: jnp.ndarray,
    fvc_l: jnp.ndarray,
    tau_o: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.array(
        [
            jnp.log(tau_f),
            jnp.log(tau_s),
            jnp.log(alpha / (1.0 - alpha)),
            jnp.log(fvc_l),
            jnp.log(tau_o),
        ]
    )


def ombe_flow(phi: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """Return synthetic OMBE flow in L/s on grid t.

    Model definition:
      q(t) = [a exp(-t/tau_f)/tau_f + (1-a) exp(-t/tau_s)/tau_s]
             * [1 - exp(-t/tau_o)]
      F(t) = FVC * q(t) / integral(q)

    The final normalization makes volume conservation explicit on the same grid
    used for inference and Fisher calculations.
    """
    p = unpack_phi(phi)
    base = (
        p["alpha"] * jnp.exp(-t / p["tau_f"]) / p["tau_f"]
        + (1.0 - p["alpha"]) * jnp.exp(-t / p["tau_s"]) / p["tau_s"]
    )
    rise = 1.0 - jnp.exp(-t / p["tau_o"])
    q = jnp.maximum(base * rise, 1e-12)
    area = jnp.trapezoid(q, t)
    return p["fvc_l"] * q / area


def ombe_volume(phi: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    flow = ombe_flow(phi, t)
    increments = (flow[:-1] + flow[1:]) * 0.5 * jnp.diff(t)
    return jnp.concatenate([jnp.array([0.0]), jnp.cumsum(increments)])


def flow_integral_l(phi: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    return jnp.trapezoid(ombe_flow(phi, t), t)


def sample_prior_phi(n: int, seed: int = 0) -> np.ndarray:
    """Sample a fixed, documented synthetic prior in transformed coordinates.

    Ranges:
      tau_f: 0.15..0.80 s
      tau_s: 1.00..5.00 s, always greater than tau_f
      alpha: 0.25..0.85
      FVC: 2.00..6.00 L
      tau_o: 0.03..0.40 s
    """
    rng = np.random.default_rng(seed)
    tau_f = rng.uniform(0.15, 0.80, size=n)
    tau_s = rng.uniform(1.00, 5.00, size=n)
    alpha = rng.uniform(0.25, 0.85, size=n)
    fvc = rng.uniform(2.00, 6.00, size=n)
    tau_o = rng.uniform(0.03, 0.40, size=n)
    phi = np.column_stack(
        [
            np.log(tau_f),
            np.log(tau_s),
            np.log(alpha / (1.0 - alpha)),
            np.log(fvc),
            np.log(tau_o),
        ]
    )
    if not np.all(phi[:, 0] < phi[:, 1]):
        raise RuntimeError("Prior violated tau_f < tau_s constraint.")
    return phi


def midrange_phi() -> jnp.ndarray:
    tau_f = 0.35
    tau_s = 2.50
    alpha = 0.60
    fvc = 4.00
    tau_o = 0.12
    return jnp.array(
        [
            np.log(tau_f),
            np.log(tau_s),
            np.log(alpha / (1.0 - alpha)),
            np.log(fvc),
            np.log(tau_o),
        ],
        dtype=jnp.float32,
    )


def fisher_information(phi: jnp.ndarray, t: jnp.ndarray, sigma_lps: float) -> jnp.ndarray:
    jac = jax.jacfwd(lambda p: ombe_flow(p, t))(phi)
    return (jac.T @ jac) / (sigma_lps**2)


batched_fisher_information = jax.jit(jax.vmap(fisher_information, in_axes=(0, None, None)))
