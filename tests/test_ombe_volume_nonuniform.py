from __future__ import annotations

import unittest

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from predictx.ombe import midrange_phi, ombe_flow, ombe_volume


class OmbeVolumeNonuniformGridTest(unittest.TestCase):
    def test_ombe_volume_uses_each_time_interval(self) -> None:
        phi = midrange_phi()
        t = jnp.asarray([0.0, 0.003, 0.013, 0.023, 0.050, 0.100, 0.200], dtype=jnp.float32)
        flow = np.asarray(ombe_flow(phi, t), dtype=float)
        expected = np.concatenate([[0.0], np.cumsum((flow[:-1] + flow[1:]) * 0.5 * np.diff(np.asarray(t)))])
        constant_dt = np.concatenate([[0.0], np.cumsum((flow[:-1] + flow[1:]) * 0.5 * float(t[1] - t[0]))])

        got = np.asarray(ombe_volume(phi, t), dtype=float)

        self.assertTrue(np.allclose(got, expected, rtol=1e-6, atol=1e-6))
        self.assertFalse(np.allclose(got[-1], constant_dt[-1], rtol=1e-2, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
