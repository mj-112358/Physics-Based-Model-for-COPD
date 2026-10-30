from __future__ import annotations

import numpy as np
import pandas as pd

from tools.evaluate_physics_informed_selector import (
    BASE_FEATURES,
    PHYSICS_FEATURES,
    BasePrediction,
    _features,
    _top_score,
)


def test_top_score_is_deterministic_and_exact() -> None:
    score = np.array([0.8, 0.8, 0.4, 0.9, 0.1])
    selected = _top_score(score, coverage=0.6)
    assert selected.tolist() == [True, True, False, True, False]
    assert int(selected.sum()) == 3


def test_physics_feature_set_is_a_strict_ablation_extension() -> None:
    assert set(BASE_FEATURES) < set(PHYSICS_FEATURES)
    assert not any("truth" in name.lower() for name in PHYSICS_FEATURES)
    assert not any("fvc_ml" == name.lower() for name in PHYSICS_FEATURES)


def test_inference_features_use_predictions_and_observations_only() -> None:
    frame = pd.DataFrame(
        {
            "duration_s": [2.0, 7.0],
            "dV_last_L": [0.12, 0.01],
            "truncated_ats": [True, False],
            "model_fvc_mL": [3500.0, 4200.0],
            "volume_r2": [0.98, 0.96],
        }
    )
    base = BasePrediction(
        ratio=np.array([0.68, 0.82]),
        fvc_l=np.array([3.4, 4.1]),
        physics_ratio=np.array([0.66, 0.82]),
    )
    features = _features(frame, base)
    assert list(features.columns) == list(PHYSICS_FEATURES)
    assert np.isclose(features.loc[0, "physics_ridge_fvc_gap_l"], 0.1)
    assert bool(features.loc[0, "physics_ridge_disagreement"]) is False
