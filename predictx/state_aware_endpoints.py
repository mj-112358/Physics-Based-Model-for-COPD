from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PHYSICS_AUDIT_COLUMNS = {
    "model_fev1_l": "physics_model_fev1_l",
    "model_fvc_l": "physics_model_fvc_l",
    "model_ratio": "physics_model_ratio",
    "volume_fit_success": "physics_volume_fit_success",
    "volume_r2": "physics_volume_r2",
    "fit_error": "physics_fit_error",
}


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.map(
            {
                True: True,
                False: False,
                "True": True,
                "False": False,
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise KeyError(f"Required endpoint column is missing: {column}")
    return pd.to_numeric(frame[column], errors="coerce")


def route_endpoints_by_measurement_state(
    frame: pd.DataFrame,
    *,
    c1: float,
    observed_fev1_col: str,
    observed_fvc_col: str,
    observed_ratio_col: str,
    truncated_col: str = "truncated_ats",
) -> pd.DataFrame:
    """Route direct and recovered endpoints without using reference outcomes.

    Complete manoeuvres retain their directly measured FEV1, FVC, and ratio.
    Incomplete manoeuvres retain directly measured FEV1 and use only the
    physics-recovered FVC.  The Cycle-F-trained constant correction is applied
    exclusively to ratios containing a recovered FVC.

    Original physics outputs are copied to ``physics_*`` audit columns before
    the compatibility ``model_*`` columns are replaced by the routed endpoint.
    Calling this function repeatedly is idempotent because existing audit
    columns remain the source of the physics branch.
    """
    if truncated_col not in frame:
        raise KeyError(f"Required measurement-state column is missing: {truncated_col}")

    out = frame.copy()
    truncated = _as_bool(out[truncated_col])
    observed_fev1 = _numeric(out, observed_fev1_col)
    observed_fvc = _numeric(out, observed_fvc_col)
    observed_ratio = _numeric(out, observed_ratio_col)

    for source, audit in PHYSICS_AUDIT_COLUMNS.items():
        if audit not in out:
            if source not in out:
                raise KeyError(f"Required physics column is missing: {source}")
            out[audit] = out[source]

    physics_fvc = pd.to_numeric(out["physics_model_fvc_l"], errors="coerce")
    physics_success = _as_bool(out["physics_volume_fit_success"])
    recovered_ratio = observed_fev1 / physics_fvc

    direct_valid = np.isfinite(observed_fev1) & np.isfinite(observed_fvc) & np.isfinite(observed_ratio)
    recovered_valid = np.isfinite(observed_fev1) & np.isfinite(physics_fvc) & physics_success

    out["model_fev1_l"] = observed_fev1
    out["model_fvc_l"] = np.where(truncated, physics_fvc, observed_fvc)
    out["model_ratio"] = np.where(truncated, recovered_ratio, observed_ratio)
    out["corrected_ratio_c1"] = np.where(truncated, recovered_ratio - float(c1), observed_ratio)
    out["endpoint_source"] = np.where(truncated, "physics_recovered_fvc", "direct_complete")
    out["c1_applied"] = truncated
    out["volume_fit_success"] = np.where(truncated, recovered_valid, direct_valid)
    out["volume_r2"] = np.where(
        truncated,
        pd.to_numeric(out["physics_volume_r2"], errors="coerce"),
        np.where(direct_valid, 1.0, np.nan),
    )
    physics_error = out["physics_fit_error"].fillna("").astype(str)
    out["fit_error"] = np.where(truncated, physics_error, np.where(direct_valid, "", "invalid direct endpoint"))

    complete = ~truncated
    if not np.allclose(
        pd.to_numeric(out.loc[complete, "model_fvc_l"], errors="coerce"),
        observed_fvc.loc[complete],
        equal_nan=True,
    ):
        raise AssertionError("Complete manoeuvres were not routed to directly observed FVC")
    if not np.allclose(
        pd.to_numeric(out.loc[truncated, "model_fev1_l"], errors="coerce"),
        observed_fev1.loc[truncated],
        equal_nan=True,
    ):
        raise AssertionError("Recovered endpoints did not retain directly observed FEV1")
    if not np.allclose(
        pd.to_numeric(out.loc[truncated, "model_fvc_l"], errors="coerce"),
        physics_fvc.loc[truncated],
        equal_nan=True,
    ):
        raise AssertionError("Incomplete manoeuvres were not routed to physics-recovered FVC")
    return out


def fit_incomplete_ratio_offset(
    train: pd.DataFrame,
    *,
    observed_fev1_col: str,
    physics_fvc_col: str,
    truth_ratio_col: str,
    truncated_col: str = "truncated_ats",
    physics_fvc_scale: float = 1.0,
) -> dict[str, Any]:
    """Fit the one-parameter ratio calibration on incomplete training rows only."""
    truncated = _as_bool(train[truncated_col])
    observed_fev1 = _numeric(train, observed_fev1_col)
    physics_fvc = _numeric(train, physics_fvc_col) * float(physics_fvc_scale)
    truth_ratio = _numeric(train, truth_ratio_col)
    raw_ratio = observed_fev1 / physics_fvc
    valid = truncated & np.isfinite(raw_ratio) & np.isfinite(truth_ratio)
    residual = raw_ratio.loc[valid] - truth_ratio.loc[valid]
    if residual.empty:
        raise RuntimeError("No valid incomplete training rows are available for state-aware ratio calibration")
    bias = float(np.median(residual))
    corrected = residual - bias
    return {
        "kind": "constant",
        "bias": bias,
        "training_rows": int(valid.sum()),
        "training_scope": "Cycle F TRAIN incomplete manoeuvres only",
        "physics_fvc_scale": float(physics_fvc_scale),
        "formula": f"corrected_ratio = observed_fev1 / recovered_fvc - ({bias:.12f})",
        "residual_median": float(np.median(corrected)),
        "residual_iqr": [float(np.percentile(corrected, 25)), float(np.percentile(corrected, 75))],
    }
