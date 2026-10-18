from __future__ import annotations

import numpy as np

from predictx.gate_frontier_f import select_safe_point
from predictx.gate_truncation_recovery_f import PREDICTION_HORIZON_S, fit_ombe_truncated, model_grid_with_stop
from predictx.raw import interval_end_times


def test_raw_increment_timestamps_represent_interval_ends() -> None:
    assert interval_end_times(3).tolist() == [0.01, 0.02, 0.03]


def test_truncation_grid_does_not_depend_on_complete_curve_duration() -> None:
    short = np.arange(0.0, 6.01, 0.01)
    long = np.arange(0.0, 10.01, 0.01)

    short_grid, short_fit = model_grid_with_stop(short, 2.0)
    long_grid, long_fit = model_grid_with_stop(long, 2.0)

    assert np.array_equal(short_grid, long_grid)
    assert np.array_equal(short_fit, long_fit)
    assert short_grid[-1] == PREDICTION_HORIZON_S


def test_truncation_fit_ignores_volume_after_cut_time() -> None:
    t = np.arange(0.0, 8.01, 0.01)
    base = 4.0 * (1.0 - np.exp(-t / 1.8))
    altered = base.copy()
    altered[t > 2.0] += 5.0

    fit_a = fit_ombe_truncated(t, base, 2.0)
    fit_b = fit_ombe_truncated(t, altered, 2.0)

    assert np.allclose(fit_a["params"], fit_b["params"], rtol=0.0, atol=1e-10)
    assert np.isclose(fit_a["fvc_l"], fit_b["fvc_l"], rtol=0.0, atol=1e-10)


def test_frontier_selection_uses_training_metrics() -> None:
    rows = [
        {
            "target_coverage": 0.70,
            "train_safe": True,
            "train_selection_coverage": 0.70,
            "train_selection_kappa_reweighted": 0.90,
            "confident_coverage": 0.99,
        },
        {
            "target_coverage": 0.80,
            "train_safe": True,
            "train_selection_coverage": 0.80,
            "train_selection_kappa_reweighted": 0.80,
            "confident_coverage": 0.10,
        },
    ]

    selected = select_safe_point(rows, "train_safe")

    assert selected is not None
    assert selected["target_coverage"] == 0.80
