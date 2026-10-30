#!/usr/bin/env python3
"""Reproduce cohort flow, patient-benefit strata, and model diagnostics.

The script consumes the row-level artifacts already used by the manuscript and
the public NHANES raw files.  It writes machine-readable cohort, benefit,
bound-sensitivity, profile-likelihood, optimization, and interval-calibration
outputs without changing the frozen workflow or any decision threshold.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/predictx-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/predictx-xdg-cache")

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from predictx.gate23_f import find_or_unzip_raw, read_raw
from predictx.gate3_diagnosis import np_ombe_volume, savgol_sigma
from predictx.gate_clinical_classification_f import extract_curve_for_fit
from predictx.gate_three_cycle_efg import (
    gate2_result_for_cycle,
    participant_candidates,
    sample_by_stratum,
)
from predictx.gate_truncation_recovery_f import model_grid_with_stop


SEED = 20261029
TRUNCATION_TIMES = (2.0, 3.0, 4.0, 5.0)
PARAMETER_NAMES = ("tau_f", "tau_s", "alpha", "fvc_l", "tau_o")
BOUNDS = {
    "nominal": ([0.15, 1.00, 0.25, 2.00, 0.03], [0.80, 5.00, 0.85, 6.00, 0.40]),
    "fvc_wide": ([0.15, 1.00, 0.25, 0.75, 0.03], [0.80, 5.00, 0.85, 7.50, 0.40]),
    "all_wide": ([0.10, 0.75, 0.10, 0.75, 0.01], [1.00, 8.00, 0.95, 7.50, 0.60]),
}


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(clean(payload), indent=2) + "\n", encoding="utf-8")


def quantile_summary(values: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "n": int(len(x)),
        "median": float(x.median()) if len(x) else np.nan,
        "q1": float(x.quantile(0.25)) if len(x) else np.nan,
        "q3": float(x.quantile(0.75)) if len(x) else np.nan,
        "minimum": float(x.min()) if len(x) else np.nan,
        "maximum": float(x.max()) if len(x) else np.nan,
    }


def artificial_cohort(root: Path, refresh: bool) -> pd.DataFrame:
    output = root / "results" / "manuscript_artificial_cohort_ids.csv"
    if output.exists() and not refresh:
        return pd.read_csv(output)

    parts: list[pd.DataFrame] = []
    f_rows = pd.read_csv(root / "results" / "gate_truncation_recovery_F_rows.csv")
    f = f_rows[
        f_rows["estimator"].eq("E1_OMBE") & f_rows["T_label"].astype(str).eq("2")
    ][
        ["SEQN", "raw_index", "truth_severity", "true_fev1_l", "true_fvc_l", "true_ratio"]
    ].copy()
    f.insert(0, "cycle", "F")
    f.insert(1, "global_key", "F:" + f["SEQN"].astype(int).astype(str))
    parts.append(f)

    for cycle in ("E", "G"):
        print(f"Reconstructing Cycle {cycle} controlled-truncation membership", flush=True)
        _gate2, raw, _derived, truth = gate2_result_for_cycle(root, cycle)
        selected_a, selected_complete = participant_candidates(raw, truth, cycle)
        sample, _meta = sample_by_stratum(selected_a, selected_complete, cycle)
        parts.append(
            sample[
                [
                    "cycle",
                    "global_key",
                    "SEQN",
                    "raw_index",
                    "truth_severity",
                    "true_fev1_l",
                    "true_fvc_l",
                    "true_ratio",
                ]
            ].copy()
        )

    out = pd.concat(parts, ignore_index=True)
    out.to_csv(output, index=False)
    return out


def cohort_record(
    name: str,
    rows: pd.DataFrame,
    demographics: pd.DataFrame,
    severity_column: str,
    cycles: str,
    criteria: str,
) -> dict[str, Any]:
    ids = rows[["cycle", "SEQN"]].drop_duplicates()
    merged = ids.merge(
        demographics[["cycle", "SEQN", "age_years"]], on=["cycle", "SEQN"], how="left"
    )
    severity = rows.drop_duplicates(["cycle", "SEQN"])[severity_column].value_counts().to_dict()
    return {
        "experiment": name,
        "cycles": cycles,
        "participants": int(len(ids)),
        "age_years": quantile_summary(merged["age_years"]),
        "severity_distribution": {str(k): int(v) for k, v in severity.items()},
        "criteria": criteria,
    }


def build_cohort_audit(root: Path, artificial: pd.DataFrame) -> dict[str, Any]:
    demographics = pd.read_parquet(root / "results" / "merged_summary.parquet")
    demographics = demographics[
        np.isfinite(demographics["fev1_mL"])
        & np.isfinite(demographics["fvc_mL"])
        & (demographics["fvc_mL"] > 0)
    ].copy()
    ratio = demographics["fev1_mL"] / demographics["fvc_mL"]
    demographics["truth_severity"] = pd.cut(
        ratio,
        [-np.inf, 0.50, 0.60, 0.70, np.inf],
        labels=["severe", "moderate", "mild_0.60_0.70", "normal"],
        right=False,
    ).astype(str)

    source = demographics[["cycle", "SEQN", "truth_severity"]].copy()
    artificial_obstructed = artificial[artificial["true_ratio"] < 0.70].copy()

    natural_all = pd.read_csv(root / "results" / "gate_truncation_hardening_F_recovered_curves.csv")
    natural = natural_all[
        natural_all["dataset"].eq("field_realistic") & natural_all["T_label"].eq("field_all")
    ][["SEQN", "truth_severity"]].copy()
    natural.insert(0, "cycle", "F")

    workflow_all = pd.read_csv(root / "results" / "gate_workflow_policy_F_rows.csv", low_memory=False)
    workflow = workflow_all[workflow_all["split"].eq("test")][
        ["SEQN", "truth_severity"]
    ].copy()
    workflow.insert(0, "cycle", "F")

    frozen_parts = []
    for cycle in ("E", "G"):
        part = pd.read_csv(root / "results" / f"frozen_validation_{cycle}_participants.csv", low_memory=False)
        part = part[["SEQN", "truth_severity"]].copy()
        part.insert(0, "cycle", cycle)
        frozen_parts.append(part)
    frozen = pd.concat(frozen_parts, ignore_index=True)

    records = [
        cohort_record(
            "NHANES summary-truth source",
            source,
            demographics,
            "truth_severity",
            "E/F/G",
            "Finite released FEV1 and FVC; participant-level summary truth.",
        ),
        cohort_record(
            "Controlled truncation sample",
            artificial,
            demographics,
            "truth_severity",
            "E/F/G",
            "Pre-bronchodilator, SPAACC=Y, SPAQEFF=A, FET>=6 s, terminal-second volume<=0.025 L; best complete curve per participant; cap 200 per severity-cycle.",
        ),
        cohort_record(
            "Controlled truncation obstructed set",
            artificial_obstructed,
            demographics,
            "truth_severity",
            "E/F/G",
            "Controlled-truncation sample with complete-curve FEV1/FVC<0.70.",
        ),
        cohort_record(
            "Natural incomplete pairs",
            natural,
            demographics,
            "truth_severity",
            "F",
            "Clean target plus separate flagged Pre manoeuvre; shortest usable incomplete candidate; one pair per participant.",
        ),
        cohort_record(
            "Cycle F policy-development test",
            workflow,
            demographics,
            "truth_severity",
            "F",
            "Participant-separated TEST partition after frozen TRAIN-only estimation.",
        ),
        cohort_record(
            "Frozen cross-cycle workflow",
            frozen,
            demographics,
            "truth_severity",
            "E/G",
            "Finite released truth plus at least one usable A/B pre-bronchodilator candidate; Cycle F configuration frozen before scoring.",
        ),
    ]

    art_f = set(artificial.loc[artificial["cycle"].eq("F"), "SEQN"])
    art_e = set(artificial.loc[artificial["cycle"].eq("E"), "SEQN"])
    art_g = set(artificial.loc[artificial["cycle"].eq("G"), "SEQN"])
    natural_ids = set(natural["SEQN"])
    workflow_ids = set(workflow["SEQN"])
    frozen_e = set(frozen.loc[frozen["cycle"].eq("E"), "SEQN"])
    frozen_g = set(frozen.loc[frozen["cycle"].eq("G"), "SEQN"])
    overlap = {
        "F_controlled_and_natural": len(art_f & natural_ids),
        "F_controlled_and_workflow_test": len(art_f & workflow_ids),
        "F_natural_and_workflow_test": len(natural_ids & workflow_ids),
        "F_all_three": len(art_f & natural_ids & workflow_ids),
        "E_controlled_within_frozen_workflow": len(art_e & frozen_e),
        "G_controlled_within_frozen_workflow": len(art_g & frozen_g),
        "cross_cycle_identifier_overlap": 0,
    }

    payload = {
        "source_raw_manoeuvres": 108_939,
        "source_participants_with_truth": 20_050,
        "usable_ab_participants": 19_459,
        "cohorts": records,
        "overlap": overlap,
        "severity_definition": {
            "severe": "ratio <0.50",
            "moderate": "0.50<=ratio<0.60",
            "mild_0.60_0.70": "0.60<=ratio<0.70",
            "normal": "ratio>=0.70",
        },
    }
    dump_json(root / "results" / "manuscript_cohort_audit.json", payload)
    flat = []
    for record in records:
        row = {
            "experiment": record["experiment"],
            "cycles": record["cycles"],
            "participants": record["participants"],
            "age_median": record["age_years"]["median"],
            "age_q1": record["age_years"]["q1"],
            "age_q3": record["age_years"]["q3"],
            "criteria": record["criteria"],
        }
        row.update({f"severity_{k}": v for k, v in record["severity_distribution"].items()})
        flat.append(row)
    pd.DataFrame(flat).to_csv(root / "results" / "manuscript_cohort_audit.csv", index=False)
    return payload


def confusion(truth: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = int(np.sum(truth & pred))
    tn = int(np.sum(~truth & ~pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    return {
        "n": int(len(truth)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "error_rate": (fp + fn) / len(truth) if len(truth) else np.nan,
    }


def build_benefit_audit(root: Path) -> dict[str, Any]:
    detail = pd.read_csv(root / "results" / "gate_truncation_recovery_F_rows.csv")
    detail = detail[
        detail["estimator"].isin(["E0_DIRECT", "E1_OMBE"])
        & detail["T_label"].astype(str).isin(["2", "3", "4", "5"])
    ].copy()

    duration_rows = []
    for t in ("2", "3", "4", "5"):
        for estimator, label in (("E0_DIRECT", "direct"), ("E1_OMBE", "physics")):
            group = detail[detail["T_label"].astype(str).eq(t) & detail["estimator"].eq(estimator)]
            metric = confusion(group["truth_obstructed_fixed"], group["est_obstructed_fixed"])
            duration_rows.append({"duration_s": int(t), "method": label, **metric})

    severity_rows = []
    for (t, severity), group in detail[
        detail["estimator"].eq("E1_OMBE") & detail["truth_obstructed_fixed"].astype(bool)
    ].groupby([detail["T_label"].astype(str), "truth_severity"]):
        severity_rows.append(
            {
                "duration_s": int(t),
                "severity": severity,
                "n": int(len(group)),
                "detection_rate": float(group["est_obstructed_fixed"].astype(bool).mean()),
            }
        )

    physics = detail[detail["estimator"].eq("E1_OMBE")].copy()
    physics["distance_band"] = pd.cut(
        0.70 - physics["true_ratio"],
        [-np.inf, 0.05, 0.10, 0.20, np.inf],
        labels=["0-0.05", "0.05-0.10", "0.10-0.20", ">=0.20"],
        right=False,
    )
    distance_rows = []
    for (t, band), group in physics[
        physics["truth_obstructed_fixed"].astype(bool)
    ].groupby([physics["T_label"].astype(str), "distance_band"], observed=True):
        distance_rows.append(
            {
                "duration_s": int(t),
                "distance_band": str(band),
                "n": int(len(group)),
                "detection_rate": float(group["est_obstructed_fixed"].astype(bool).mean()),
            }
        )

    two = detail[detail["T_label"].astype(str).eq("2")].copy()
    two["fvc_band"] = pd.cut(
        two["true_fvc_l"],
        [-np.inf, 2.0, 3.0, 4.0, np.inf],
        labels=["<2 L", "2-3 L", "3-4 L", ">=4 L"],
        right=False,
    )
    fvc_rows = []
    for band, group in two.groupby("fvc_band", observed=True):
        direct = group[group["estimator"].eq("E0_DIRECT")]
        model = group[group["estimator"].eq("E1_OMBE")]
        direct_obstructed = direct[direct["truth_obstructed_fixed"].astype(bool)]
        model_obstructed = model[model["truth_obstructed_fixed"].astype(bool)]
        fvc_rows.append(
            {
                "fvc_band": str(band),
                "n": int(model["SEQN"].nunique()),
                "obstructed_n": int(len(model_obstructed)),
                "direct_detection": float(direct_obstructed["est_obstructed_fixed"].astype(bool).mean()),
                "physics_detection": float(model_obstructed["est_obstructed_fixed"].astype(bool).mean()),
                "physics_median_abs_fvc_error_ml": float(
                    np.median(np.abs((model["est_fvc_l"] - model["true_fvc_l"]) * 1000.0))
                ),
                "physics_floor_hits": int(np.isclose(model["est_fvc_l"], 2.0, atol=1e-6).sum()),
            }
        )

    frozen = []
    for cycle in ("E", "G"):
        part = pd.read_csv(root / "results" / f"frozen_validation_{cycle}_participants.csv", low_memory=False)
        part["cycle"] = cycle
        frozen.append(part)
    frozen_df = pd.concat(frozen, ignore_index=True)
    frozen_df["maneuver_band"] = pd.cut(
        frozen_df["maneuver_count"],
        [-np.inf, 2, 3, 4, np.inf],
        labels=["1-2", "3", "4", "5+"],
        right=True,
    )
    maneuver_rows = []
    for band, group in frozen_df.groupby("maneuver_band", observed=True):
        selected = group["physics_finalized"].astype(bool)
        truth = group["truth_obstructed_fixed"].astype(bool)
        pred = group["physics_prediction_obstructed"].astype(bool)
        maneuver_rows.append(
            {
                "maneuver_band": str(band),
                "n": int(len(group)),
                "coverage": float(selected.mean()),
                "accuracy_when_finalized": float((truth[selected] == pred[selected]).mean()),
            }
        )

    interval = frozen_df[
        frozen_df["candidate_ci_low"].notna() & frozen_df["candidate_ci_high"].notna()
    ].copy()
    interval["ci_width"] = interval["candidate_ci_high"] - interval["candidate_ci_low"]
    interval["distance_to_boundary"] = np.abs(interval["candidate_median"] - 0.70)
    interval["straddles_boundary"] = (
        (interval["candidate_ci_low"] < 0.70) & (interval["candidate_ci_high"] >= 0.70)
    )
    interval["finalized"] = interval["physics_finalized"].astype(bool)
    interval["covers_truth_ratio"] = (
        (interval["fev1_fvc_ratio"] >= interval["candidate_ci_low"])
        & (interval["fev1_fvc_ratio"] <= interval["candidate_ci_high"])
    )
    interval_rows = interval[
        [
            "cycle",
            "SEQN",
            "ci_width",
            "distance_to_boundary",
            "straddles_boundary",
            "finalized",
            "covers_truth_ratio",
        ]
    ].to_dict("records")

    truth = frozen_df["truth_obstructed_fixed"].astype(bool).to_numpy()
    ridge_ratio = frozen_df["state_aware_ridge_ratio"].to_numpy(float)
    ridge_pred = ridge_ratio < 0.70
    ridge_margin = np.abs(ridge_ratio - 0.70)
    risk_rows = []
    for coverage in np.linspace(0.10, 1.00, 19):
        n = max(1, int(round(coverage * len(frozen_df))))
        take = np.argsort(-ridge_margin)[:n]
        metric = confusion(truth[take], ridge_pred[take])
        risk_rows.append({"coverage": n / len(frozen_df), "method": "ridge_margin", **metric})

    physics_pred = frozen_df["physics_prediction_obstructed"].astype(bool).to_numpy()
    physics_selected = frozen_df["physics_finalized"].astype(bool).to_numpy()
    consensus = physics_selected & (physics_pred == ridge_pred)
    consensus_metric = confusion(truth[consensus], physics_pred[consensus])
    consensus_metric["coverage"] = float(consensus.mean())

    payload = {
        "controlled_duration": duration_rows,
        "controlled_severity": severity_rows,
        "controlled_distance_to_boundary": distance_rows,
        "controlled_fvc_magnitude": fvc_rows,
        "frozen_maneuver_availability": maneuver_rows,
        "r2_interval_cases": {
            "n": int(len(interval)),
            "empirical_truth_coverage": float(interval["covers_truth_ratio"].mean()),
            "boundary_straddle_n": int(interval["straddles_boundary"].sum()),
            "boundary_straddle_finalized_n": int(
                (interval["straddles_boundary"] & interval["finalized"]).sum()
            ),
            "rows": interval_rows,
        },
        "risk_coverage": risk_rows,
        "frozen_consensus_operating_point": consensus_metric,
    }
    dump_json(root / "results" / "manuscript_patient_benefit.json", payload)
    flat = []
    for facet, rows in (
        ("duration", duration_rows),
        ("severity", severity_rows),
        ("distance", distance_rows),
        ("fvc", fvc_rows),
        ("maneuver", maneuver_rows),
        ("risk_coverage", risk_rows),
    ):
        flat.extend({"facet": facet, **row} for row in rows)
    pd.DataFrame(flat).to_csv(root / "results" / "manuscript_patient_benefit.csv", index=False)
    return payload


def fit_bounded(
    full_t: np.ndarray,
    full_y: np.ndarray,
    stop_s: float,
    lo_values: list[float],
    hi_values: list[float],
    fixed_fvc: float | None = None,
) -> dict[str, Any]:
    grid, fit_idx = model_grid_with_stop(full_t, stop_s)
    y_fit = np.interp(grid[fit_idx], full_t, full_y)
    sigma = savgol_sigma(y_fit)
    scale = sigma if np.isfinite(sigma) and sigma > 0 else max(float(np.std(y_fit)), 0.01)
    lo = np.asarray(lo_values, dtype=float)
    hi = np.asarray(hi_values, dtype=float)
    observed_end = float(y_fit[-1])
    x0 = np.array([0.35, 2.50, 0.60, np.clip(observed_end, lo[3], hi[3]), 0.12])

    if fixed_fvc is None:
        def residual(x: np.ndarray) -> np.ndarray:
            return (np_ombe_volume(x, grid)[fit_idx] - y_fit) / scale

        opt = least_squares(
            residual,
            x0=np.clip(x0, lo + 1e-9, hi - 1e-9),
            bounds=(lo, hi),
            method="trf",
            jac="3-point",
            max_nfev=600,
        )
        params = opt.x
    else:
        free = [0, 1, 2, 4]
        x0_free = x0[free]
        lo_free = lo[free]
        hi_free = hi[free]

        def residual(x_free: np.ndarray) -> np.ndarray:
            params_local = np.empty(5, dtype=float)
            params_local[free] = x_free
            params_local[3] = fixed_fvc
            return (np_ombe_volume(params_local, grid)[fit_idx] - y_fit) / scale

        opt = least_squares(
            residual,
            x0=np.clip(x0_free, lo_free + 1e-9, hi_free - 1e-9),
            bounds=(lo_free, hi_free),
            method="trf",
            jac="3-point",
            max_nfev=600,
        )
        params = np.empty(5, dtype=float)
        params[free] = opt.x
        params[3] = fixed_fvc

    objective = float(np.dot(opt.fun, opt.fun))
    condition = np.nan
    try:
        singular = np.linalg.svd(np.asarray(opt.jac, dtype=float), compute_uv=False)
        condition = float((singular[0] / singular[-1]) ** 2) if singular[-1] > 0 else np.inf
    except Exception:
        pass
    return {
        "success": bool(opt.success),
        "params": params,
        "objective": objective,
        "n_observations": int(len(y_fit)),
        "nfev": int(opt.nfev),
        "condition_jtj": condition,
        "scale_l": float(scale),
        "message": str(opt.message),
    }


def build_model_diagnostics(root: Path, artificial: pd.DataFrame, refresh: bool) -> dict[str, Any]:
    sensitivity_csv = root / "results" / "manuscript_bound_sensitivity_rows.csv"
    profile_csv = root / "results" / "manuscript_profile_likelihood_rows.csv"
    if sensitivity_csv.exists() and not refresh:
        sensitivity = pd.read_csv(sensitivity_csv)
    else:
        raw_path, _note = find_or_unzip_raw(root, "F")
        raw, _info = read_raw(raw_path, "F")
        f = artificial[artificial["cycle"].eq("F")].copy()
        curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for row in f.itertuples(index=False):
            curve = extract_curve_for_fit(raw.loc[int(row.raw_index)])
            curves[int(row.raw_index)] = (
                np.asarray(curve["vol_t"], dtype=float),
                np.asarray(curve["vol_l"], dtype=float),
            )

        sensitivity_rows = []
        total = len(f) * len(TRUNCATION_TIMES) * len(BOUNDS)
        done = 0
        for row in f.itertuples(index=False):
            full_t, full_y = curves[int(row.raw_index)]
            for stop_s in TRUNCATION_TIMES:
                observed_fev1 = float(np.interp(1.0, full_t, full_y))
                for config, (lo, hi) in BOUNDS.items():
                    fit = fit_bounded(full_t, full_y, stop_s, lo, hi)
                    params = np.asarray(fit["params"], dtype=float)
                    ratio = observed_fev1 / params[3]
                    active = np.isclose(params, lo, atol=1e-5) | np.isclose(params, hi, atol=1e-5)
                    sensitivity_rows.append(
                        {
                            "SEQN": row.SEQN,
                            "raw_index": int(row.raw_index),
                            "truth_severity": row.truth_severity,
                            "duration_s": int(stop_s),
                            "configuration": config,
                            "success": fit["success"],
                            "objective": fit["objective"],
                            "nfev": fit["nfev"],
                            "condition_jtj": fit["condition_jtj"],
                            "true_fvc_l": row.true_fvc_l,
                            "true_ratio": row.true_ratio,
                            "estimated_fvc_l": params[3],
                            "estimated_ratio": ratio,
                            "truth_obstructed": bool(row.true_ratio < 0.70),
                            "predicted_obstructed": bool(ratio < 0.70),
                            **{name: params[i] for i, name in enumerate(PARAMETER_NAMES)},
                            **{f"active_{name}": bool(active[i]) for i, name in enumerate(PARAMETER_NAMES)},
                        }
                    )
                    done += 1
                    if done % 500 == 0 or done == total:
                        print(f"Bound-sensitivity fits: {done}/{total}", flush=True)
        sensitivity = pd.DataFrame(sensitivity_rows)
        sensitivity.to_csv(sensitivity_csv, index=False)

    if profile_csv.exists() and not refresh:
        profile = pd.read_csv(profile_csv)
    else:
        raw_path, _note = find_or_unzip_raw(root, "F")
        raw, _info = read_raw(raw_path, "F")
        f = artificial[artificial["cycle"].eq("F")].copy()
        curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for row in f.itertuples(index=False):
            curve = extract_curve_for_fit(raw.loc[int(row.raw_index)])
            curves[int(row.raw_index)] = (
                np.asarray(curve["vol_t"], dtype=float),
                np.asarray(curve["vol_l"], dtype=float),
            )
        rng = np.random.default_rng(SEED)
        profile_sample = pd.concat(
            [
                group.sample(
                    n=min(15, len(group)),
                    random_state=int(rng.integers(1, 1_000_000)),
                )
                for _severity, group in f.groupby("truth_severity", sort=False)
            ],
            ignore_index=True,
        )
        profile_rows = []
        for i, row in enumerate(profile_sample.itertuples(index=False), start=1):
            full_t, full_y = curves[int(row.raw_index)]
            for stop_s in TRUNCATION_TIMES:
                nominal = fit_bounded(full_t, full_y, stop_s, *BOUNDS["fvc_wide"])
                center = float(nominal["params"][3])
                grid = np.unique(
                    np.concatenate(
                        [
                            np.linspace(max(0.75, center * 0.60), min(7.50, center * 1.40), 41),
                            [center],
                        ]
                    )
                )
                for fvc_value in grid:
                    prof = fit_bounded(
                        full_t,
                        full_y,
                        stop_s,
                        *BOUNDS["fvc_wide"],
                        fixed_fvc=float(fvc_value),
                    )
                    profile_rows.append(
                        {
                            "SEQN": row.SEQN,
                            "raw_index": int(row.raw_index),
                            "truth_severity": row.truth_severity,
                            "duration_s": int(stop_s),
                            "profile_fvc_l": float(fvc_value),
                            "profile_objective": prof["objective"],
                            "nominal_objective": nominal["objective"],
                            "delta_objective": prof["objective"] - nominal["objective"],
                            "profile_lr": float(
                                nominal["n_observations"]
                                * np.log(max(prof["objective"], 1e-12) / max(nominal["objective"], 1e-12))
                            ),
                            "n_observations": nominal["n_observations"],
                            "nominal_fvc_l": center,
                            "true_fvc_l": row.true_fvc_l,
                            "condition_jtj": nominal["condition_jtj"],
                        }
                    )
            print(f"Profile-likelihood curves: {i}/{len(profile_sample)}", flush=True)
        profile = pd.DataFrame(profile_rows)
        profile.to_csv(profile_csv, index=False)

    sensitivity_summary = []
    for (config, duration), group in sensitivity.groupby(["configuration", "duration_s"]):
        successful = group[group["success"].astype(bool)].copy()
        truth = successful["truth_obstructed"].astype(bool).to_numpy()
        pred = successful["predicted_obstructed"].astype(bool).to_numpy()
        metric = confusion(truth, pred)
        sensitivity_summary.append(
            {
                "configuration": config,
                "duration_s": int(duration),
                "n": int(len(group)),
                "success_rate": float(group["success"].astype(bool).mean()),
                "median_abs_fvc_error_ml": float(
                    np.median(np.abs(successful["estimated_fvc_l"] - successful["true_fvc_l"]) * 1000.0)
                ),
                "p95_abs_fvc_error_ml": float(
                    np.percentile(np.abs(successful["estimated_fvc_l"] - successful["true_fvc_l"]) * 1000.0, 95)
                ),
                "median_nfev": float(successful["nfev"].median()),
                "any_active_bound_rate": float(
                    successful[[f"active_{name}" for name in PARAMETER_NAMES]].any(axis=1).mean()
                ),
                **metric,
            }
        )

    paired_bound = []
    nominal = sensitivity[sensitivity["configuration"].eq("nominal")]
    for config in ("fvc_wide", "all_wide"):
        alt = sensitivity[sensitivity["configuration"].eq(config)]
        joined = nominal.merge(
            alt,
            on=["SEQN", "raw_index", "duration_s"],
            suffixes=("_nominal", "_alternate"),
            validate="one_to_one",
        )
        joined["fvc_shift_ml"] = (
            joined["estimated_fvc_l_alternate"] - joined["estimated_fvc_l_nominal"]
        ) * 1000.0
        joined["classification_flip"] = (
            joined["predicted_obstructed_nominal"] != joined["predicted_obstructed_alternate"]
        )
        for duration, group in joined.groupby("duration_s"):
            paired_bound.append(
                {
                    "configuration": config,
                    "duration_s": int(duration),
                    "n": int(len(group)),
                    "median_absolute_fvc_shift_ml": float(np.median(np.abs(group["fvc_shift_ml"]))),
                    "p95_absolute_fvc_shift_ml": float(np.percentile(np.abs(group["fvc_shift_ml"]), 95)),
                    "classification_flip_rate": float(group["classification_flip"].mean()),
                }
            )

    profile_summary = []
    for duration, group in profile.groupby("duration_s"):
        per_curve = []
        for (_seqn, _raw), curve in group.groupby(["SEQN", "raw_index"]):
            accepted = curve[curve["profile_lr"] <= 3.841459]
            width = accepted["profile_fvc_l"].max() - accepted["profile_fvc_l"].min() if len(accepted) else np.nan
            per_curve.append(
                {
                    "width_l": width,
                    "touches_grid_edge": bool(
                        len(accepted)
                        and (
                            np.isclose(accepted["profile_fvc_l"].min(), curve["profile_fvc_l"].min())
                            or np.isclose(accepted["profile_fvc_l"].max(), curve["profile_fvc_l"].max())
                        )
                    ),
                    "condition_jtj": float(curve["condition_jtj"].iloc[0]),
                }
            )
        pc = pd.DataFrame(per_curve)
        profile_summary.append(
            {
                "duration_s": int(duration),
                "curves": int(len(pc)),
                "median_profile_width_l": float(pc["width_l"].median()),
                "q1_profile_width_l": float(pc["width_l"].quantile(0.25)),
                "q3_profile_width_l": float(pc["width_l"].quantile(0.75)),
                "grid_edge_rate": float(pc["touches_grid_edge"].mean()),
                "median_log10_condition_jtj": float(np.median(np.log10(pc["condition_jtj"].replace(np.inf, np.nan).dropna()))),
            }
        )

    optimization = json.loads((root / "results" / "gate_fit_optimization_F.json").read_text())
    interval = json.loads((root / "results" / "manuscript_patient_benefit.json").read_text())[
        "r2_interval_cases"
    ]
    payload = {
        "bound_configurations": BOUNDS,
        "bound_sensitivity_summary": sensitivity_summary,
        "paired_bound_changes": paired_bound,
        "profile_likelihood_summary": profile_summary,
        "optimization_reliability": {
            "audit_curves": optimization["part2_multistart_landscape"]["n"],
            "starts_per_curve": optimization["part2_multistart_landscape"]["k"],
            "material_non_default_best_n": optimization["part2_multistart_landscape"]["material_non_default_best_n"],
            "material_local_minimum_rate": optimization["part2_multistart_landscape"]["material_local_minimum_rate"],
        },
        "individual_r2_interval_coverage": {
            "n": interval["n"],
            "empirical_truth_coverage": interval["empirical_truth_coverage"],
            "boundary_straddle_n": interval["boundary_straddle_n"],
            "boundary_straddle_finalized_n": interval["boundary_straddle_finalized_n"],
        },
        "interpretation": {
            "population_bootstrap": "Participant resampling estimates uncertainty in cohort-level performance metrics.",
            "individual_interval": "R2 resampling across repeated manoeuvres produces a participant-specific decision interval and is audited against the released participant ratio.",
            "profile_threshold": "FVC profile set uses delta standardized sum of squares <= chi-square(1,0.95)=3.841459 on a fixed grid.",
        },
    }
    dump_json(root / "results" / "manuscript_model_diagnostics.json", payload)
    pd.DataFrame(sensitivity_summary).to_csv(
        root / "results" / "manuscript_bound_sensitivity_summary.csv", index=False
    )
    pd.DataFrame(profile_summary).to_csv(
        root / "results" / "manuscript_profile_likelihood_summary.csv", index=False
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--refresh-cohort", action="store_true")
    parser.add_argument("--refresh-diagnostics", action="store_true")
    parser.add_argument("--skip-diagnostics", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()

    artificial = artificial_cohort(root, args.refresh_cohort)
    cohort = build_cohort_audit(root, artificial)
    benefit = build_benefit_audit(root)
    diagnostics = None
    if not args.skip_diagnostics:
        diagnostics = build_model_diagnostics(root, artificial, args.refresh_diagnostics)
    print(
        json.dumps(
            clean(
                {
                    "cohorts": {r["experiment"]: r["participants"] for r in cohort["cohorts"]},
                    "consensus": benefit["frozen_consensus_operating_point"],
                    "diagnostics_complete": diagnostics is not None,
                }
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
