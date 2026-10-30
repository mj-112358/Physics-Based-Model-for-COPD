from __future__ import annotations

import numpy as np
import pandas as pd

from predictx.gate_deferral_recovery_f import corrected_maneuvers
from predictx.gate_r3_ats_besttest_f import besttest_details, raw_maneuvers_with_correction
from predictx.state_aware_endpoints import fit_incomplete_ratio_offset, route_endpoints_by_measurement_state


def test_route_uses_direct_complete_and_observed_fev1_for_recovery() -> None:
    source = pd.DataFrame(
        {
            "row_type": ["maneuver_fit", "maneuver_fit"],
            "SEQN": [1.0, 2.0],
            "truncated_ats": [False, True],
            "observed_fev1_l": [2.0, 1.8],
            "observed_fvc_l": [3.0, 2.0],
            "observed_ratio": [2.0 / 3.0, 0.9],
            "model_fev1_l": [9.0, 1.6],
            "model_fvc_l": [9.5, 3.0],
            "model_ratio": [9.0 / 9.5, 1.6 / 3.0],
            "volume_fit_success": [False, True],
            "volume_r2": [0.1, 0.99],
            "fit_error": ["irrelevant complete-curve fit failure", ""],
        }
    )

    routed = route_endpoints_by_measurement_state(
        source,
        c1=0.01,
        observed_fev1_col="observed_fev1_l",
        observed_fvc_col="observed_fvc_l",
        observed_ratio_col="observed_ratio",
    )

    assert routed.loc[0, "endpoint_source"] == "direct_complete"
    assert routed.loc[0, "model_fev1_l"] == 2.0
    assert routed.loc[0, "model_fvc_l"] == 3.0
    assert routed.loc[0, "corrected_ratio_c1"] == 2.0 / 3.0
    assert bool(routed.loc[0, "volume_fit_success"])
    assert routed.loc[0, "fit_error"] == ""

    assert routed.loc[1, "endpoint_source"] == "physics_recovered_fvc"
    assert routed.loc[1, "model_fev1_l"] == 1.8
    assert routed.loc[1, "model_fvc_l"] == 3.0
    assert np.isclose(routed.loc[1, "model_ratio"], 0.6)
    assert np.isclose(routed.loc[1, "corrected_ratio_c1"], 0.59)
    assert routed.loc[1, "physics_model_fev1_l"] == 1.6


def test_routing_is_idempotent_and_maneuver_helpers_preserve_it() -> None:
    source = pd.DataFrame(
        {
            "row_type": ["maneuver_fit"],
            "SEQN": [1.0],
            "truncated_ats": [False],
            "observed_fev1_l": [2.0],
            "observed_fvc_l": [4.0],
            "observed_ratio": [0.5],
            "model_fev1_l": [1.7],
            "model_fvc_l": [3.5],
            "model_ratio": [1.7 / 3.5],
            "volume_fit_success": [True],
            "volume_r2": [0.98],
            "fit_error": [""],
        }
    )
    once = route_endpoints_by_measurement_state(
        source,
        c1=0.02,
        observed_fev1_col="observed_fev1_l",
        observed_fvc_col="observed_fvc_l",
        observed_ratio_col="observed_ratio",
    )
    twice = route_endpoints_by_measurement_state(
        once,
        c1=0.02,
        observed_fev1_col="observed_fev1_l",
        observed_fvc_col="observed_fvc_l",
        observed_ratio_col="observed_ratio",
    )
    assert twice.loc[0, "physics_model_fvc_l"] == 3.5
    assert twice.loc[0, "corrected_ratio_c1"] == 0.5
    assert corrected_maneuvers(twice, 0.02).loc[0, "corrected_ratio_c1"] == 0.5
    assert raw_maneuvers_with_correction(twice, 0.02).loc[0, "corrected_ratio_c1"] == 0.5


def test_calibration_uses_only_incomplete_training_rows() -> None:
    train = pd.DataFrame(
        {
            "truncated_ats": [True, True, False],
            "parsed_fev1_l": [1.0, 2.0, 99.0],
            "model_fvc_mL": [2000.0, 4000.0, 1000.0],
            "fev1_fvc_ratio": [0.45, 0.55, 0.0],
        }
    )
    fitted = fit_incomplete_ratio_offset(
        train,
        observed_fev1_col="parsed_fev1_l",
        physics_fvc_col="model_fvc_mL",
        truth_ratio_col="fev1_fvc_ratio",
        physics_fvc_scale=0.001,
    )
    assert fitted["training_rows"] == 2
    assert np.isclose(fitted["bias"], 0.0)


def test_besttest_applies_correction_only_when_selected_fvc_is_recovered() -> None:
    common = {
        "SPAACC": ["Y"],
        "volume_fit_success": [True],
        "model_fev1_l_numeric": [2.0],
        "model_fvc_l_numeric": [4.0],
        "corrected_ratio_c1": [0.5],
        "raw_index": [1],
    }
    direct = besttest_details(
        pd.DataFrame({**common, "endpoint_source": ["direct_complete"]}),
        c1=0.01,
        boundary_guard=False,
    )
    recovered = besttest_details(
        pd.DataFrame({**common, "endpoint_source": ["physics_recovered_fvc"]}),
        c1=0.01,
        boundary_guard=False,
    )
    assert np.isclose(direct["ats_besttest_ratio_corrected"], 0.5)
    assert not direct["ats_besttest_c1_applied"]
    assert np.isclose(recovered["ats_besttest_ratio_corrected"], 0.49)
    assert recovered["ats_besttest_c1_applied"]
