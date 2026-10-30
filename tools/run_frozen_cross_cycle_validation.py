from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from predictx.dataset_resolver import resolve_dataset_path
from predictx.gate23_f import read_raw
from predictx.gate_clinical_classification_f import (
    build_usable_raw_candidates,
    extract_curve_for_fit,
    select_best_manoeuvre,
    severity_category,
)
from predictx.gate_deferral_recovery_f import (
    apply_recovery,
    corrected_maneuvers,
)
from predictx.gate_policy_calibrated_flag_f import add_ats_features, ats_flag
from predictx.gate_r3_ats_besttest_f import (
    apply_ats_policy,
    build_case_rows,
    raw_maneuvers_with_correction,
)
from predictx.gate_reproducibility_volume_f import (
    bool_series,
    fit_one_volume,
    participant_sd_rows,
    raw_index_map,
)
from predictx.gate_workflow_policy_f import add_repeatability, add_workflow_columns
from predictx.state_aware_endpoints import fit_incomplete_ratio_offset, route_endpoints_by_measurement_state
from tools.evaluate_manuscript_baselines import (
    BOOTSTRAP_REPLICATES,
    FIXED_RATIO_THRESHOLD,
    RIDGE_ALPHAS,
    RIDGE_FEATURES,
    RidgeRegressor,
    choose_fixed_specificity_threshold,
    choose_ridge_alpha,
    participant_folds,
    score_binary,
    top_margin_mask,
)

SEED = 20261029
TARGET_SPECIFICITY = 0.9774774774774775
FEV1_FEV6_THRESHOLD = 0.73
FIT_BATCH = 250
FINAL_OBSTRUCTED = {"ACCEPT_OBSTRUCTED", "TRUNCATED_RECOVERED", "RECOVERED_OBSTRUCTED"}
HYBRID_FEATURES = (*RIDGE_FEATURES, "physics_recovered_fvc_l")


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bootstrap_metrics(truth: np.ndarray, pred: np.ndarray, selected: np.ndarray, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    n = len(truth)
    names = ("coverage", "sensitivity", "specificity", "kappa")
    draws = {name: np.empty(BOOTSTRAP_REPLICATES) for name in names}
    for b in range(BOOTSTRAP_REPLICATES):
        take = rng.integers(0, n, size=n)
        metric = score_binary(truth[take], pred[take], selected[take])
        for name in names:
            draws[name][b] = metric[name]
    return {
        name: [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]
        for name, values in draws.items()
    }


def bootstrap_error(error: np.ndarray, seed: int) -> dict[str, Any]:
    arr = np.asarray(error, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mae_ml": np.nan, "median_signed_ml": np.nan, "mae_95_ci": [np.nan, np.nan]}
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_REPLICATES)
    for b in range(BOOTSTRAP_REPLICATES):
        sample = arr[rng.integers(0, arr.size, size=arr.size)]
        draws[b] = np.mean(np.abs(sample))
    return {
        "n": int(arr.size),
        "mae_ml": float(np.mean(np.abs(arr))),
        "median_signed_ml": float(np.median(arr)),
        "mae_95_ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def paired_bootstrap_mae_difference(error_a: np.ndarray, error_b: np.ndarray, seed: int) -> dict[str, Any]:
    a = np.asarray(error_a, dtype=float)
    b = np.asarray(error_b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if a.size == 0:
        return {"n": 0, "difference_mae_ml": np.nan, "95_ci": [np.nan, np.nan]}
    observed = float(np.mean(np.abs(a)) - np.mean(np.abs(b)))
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_REPLICATES)
    for i in range(BOOTSTRAP_REPLICATES):
        take = rng.integers(0, a.size, size=a.size)
        draws[i] = np.mean(np.abs(a[take])) - np.mean(np.abs(b[take]))
    return {
        "n": int(a.size),
        "difference_mae_ml": observed,
        "95_ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def paired_bootstrap_difference(
    truth: np.ndarray,
    pred_a: np.ndarray,
    selected_a: np.ndarray,
    pred_b: np.ndarray,
    selected_b: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(truth)
    names = ("coverage", "sensitivity", "specificity", "kappa")
    observed_a = score_binary(truth, pred_a, selected_a)
    observed_b = score_binary(truth, pred_b, selected_b)
    draws = {name: np.empty(BOOTSTRAP_REPLICATES) for name in names}
    for i in range(BOOTSTRAP_REPLICATES):
        take = rng.integers(0, n, size=n)
        ma = score_binary(truth[take], pred_a[take], selected_a[take])
        mb = score_binary(truth[take], pred_b[take], selected_b[take])
        for name in names:
            draws[name][i] = ma[name] - mb[name]
    return {
        name: {
            "difference": float(observed_a[name] - observed_b[name]),
            "95_ci": [float(np.nanpercentile(draws[name], 2.5)), float(np.nanpercentile(draws[name], 97.5))],
        }
        for name in names
    }


def metric_record(
    name: str,
    truth: np.ndarray,
    pred: np.ndarray,
    selected: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    out = score_binary(truth, pred, selected)
    out["method"] = name
    out["bootstrap_95_ci"] = bootstrap_metrics(truth, pred, selected, seed)
    return clean(out)


def fit_candidates_with_checkpoint(
    root: Path,
    cycle: str,
    raw: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    checkpoint = root / "results" / f"frozen_validation_{cycle}_maneuver_fits.csv"
    raw_by_index = raw_index_map(raw)
    done: set[int] = set()
    if checkpoint.exists():
        prior = pd.read_csv(checkpoint, usecols=["raw_index"])
        done = set(prior["raw_index"].astype(int))
        print(f"Cycle {cycle}: resuming after {len(done)} manoeuvre fits", flush=True)
    batch: list[dict[str, Any]] = []
    total = len(candidates)
    for i, candidate in enumerate(candidates.itertuples(index=False), start=1):
        raw_index = int(candidate.raw_index)
        if raw_index in done:
            continue
        row = fit_one_volume(raw_by_index, candidate)
        try:
            curve = extract_curve_for_fit(raw_by_index[raw_index])
            fet = float(curve["vol_t"][-1])
            start = max(0.0, fet - 1.0)
            v_start = float(np.interp(start, curve["vol_t"], curve["vol_l"]))
            row["fet_s"] = fet
            row["dV_last_L"] = float(curve["vol_l"][-1] - v_start)
            row["truncated_ats"] = bool(row["dV_last_L"] > 0.025 or fet < 6.0)
        except Exception:
            row["fet_s"] = np.nan
            row["dV_last_L"] = np.nan
            row["truncated_ats"] = True
        raw_row = raw_by_index[raw_index]
        row["SPAPLAT"] = str(getattr(raw_row, "SPAPLAT", ""))
        batch.append(row)
        if len(batch) >= FIT_BATCH:
            pd.DataFrame(batch).to_csv(checkpoint, mode="a", header=not checkpoint.exists(), index=False)
            batch = []
            print(f"Cycle {cycle}: fitted {i}/{total} manoeuvres", flush=True)
    if batch:
        pd.DataFrame(batch).to_csv(checkpoint, mode="a", header=not checkpoint.exists(), index=False)
    fits = pd.read_csv(checkpoint, low_memory=False)
    fits = fits.drop_duplicates("raw_index", keep="last")
    if len(fits) != total:
        raise RuntimeError(f"Cycle {cycle} fit checkpoint has {len(fits)} rows; expected {total}")
    fits["volume_fit_success"] = bool_series(fits["volume_fit_success"])
    fits["truncated_ats"] = bool_series(fits["truncated_ats"])
    return fits


def choose_ridge_alpha_for_features(
    train: pd.DataFrame,
    features: tuple[str, ...],
) -> tuple[float, list[dict[str, float]]]:
    x = train[list(features)].to_numpy(float)
    y = train["fvc_mL"].to_numpy(float) / 1000.0
    folds = participant_folds(train["SEQN"].to_numpy())
    rows: list[dict[str, float]] = []
    for alpha in RIDGE_ALPHAS:
        prediction = np.full(len(train), np.nan)
        for fold in range(5):
            fit_mask = folds != fold
            validation_mask = folds == fold
            model = RidgeRegressor(alpha).fit(x[fit_mask], y[fit_mask])
            prediction[validation_mask] = model.predict(x[validation_mask])
        rows.append(
            {
                "alpha": float(alpha),
                "cv_mae_l": float(np.mean(np.abs(prediction - y))),
            }
        )
    best = min(rows, key=lambda row: (row["cv_mae_l"], row["alpha"]))
    return float(best["alpha"]), rows


def load_frozen_ridges(root: Path) -> dict[str, Any]:
    policy = pd.read_csv(root / "results" / "gate_policy_calibrated_flag_F_rows.csv", low_memory=False)
    merged = pq.read_table(root / "results" / "merged_summary.parquet").to_pandas()
    f = merged[merged["cycle"].eq("F")].drop_duplicates("SEQN")
    data = policy.merge(f[["SEQN", "age_years", "height_cm", "sex"]], on="SEQN", how="left", validate="one_to_one")
    data["physics_recovered_fvc_l"] = pd.to_numeric(data["model_fvc_mL"], errors="coerce") / 1000.0
    train = data[data["split"].eq("train")].copy()
    best_alpha, alpha_search = choose_ridge_alpha(train)
    full = RidgeRegressor(best_alpha).fit(
        train[list(RIDGE_FEATURES)].to_numpy(float),
        train["fvc_mL"].to_numpy(float) / 1000.0,
    )
    truncated_train = train[train["truncated_ats"].astype(bool)].copy()
    if len(truncated_train) < 20:
        raise RuntimeError("Too few Cycle F TRAIN truncated rows for the censored-only ridge audit")
    truncated_alpha, truncated_search = choose_ridge_alpha(truncated_train)
    truncated = RidgeRegressor(truncated_alpha).fit(
        truncated_train[list(RIDGE_FEATURES)].to_numpy(float),
        truncated_train["fvc_mL"].to_numpy(float) / 1000.0,
    )
    hybrid_alpha, hybrid_search = choose_ridge_alpha_for_features(truncated_train, HYBRID_FEATURES)
    hybrid = RidgeRegressor(hybrid_alpha).fit(
        truncated_train[list(HYBRID_FEATURES)].to_numpy(float),
        truncated_train["fvc_mL"].to_numpy(float) / 1000.0,
    )

    full_train_ratio = train["parsed_fev1_l"].to_numpy(float) / full.predict(train[list(RIDGE_FEATURES)].to_numpy(float))
    truncated_train_ratio = train["parsed_fev1_l"].to_numpy(float) / truncated.predict(train[list(RIDGE_FEATURES)].to_numpy(float))
    truth = train["fev1_fvc_ratio"].to_numpy(float) < 0.70
    full_threshold = choose_fixed_specificity_threshold(full_train_ratio, truth, TARGET_SPECIFICITY)
    state_ratio = np.where(train["truncated_ats"].astype(bool), truncated_train_ratio, train["parsed_ratio"].to_numpy(float))
    hybrid_train_fvc = hybrid.predict(train[list(HYBRID_FEATURES)].to_numpy(float))
    hybrid_state_ratio = np.where(
        train["truncated_ats"].astype(bool),
        train["parsed_fev1_l"].to_numpy(float) / hybrid_train_fvc,
        train["parsed_ratio"].to_numpy(float),
    )
    state_threshold = choose_fixed_specificity_threshold(state_ratio, truth, TARGET_SPECIFICITY)
    hybrid_threshold = choose_fixed_specificity_threshold(hybrid_state_ratio, truth, TARGET_SPECIFICITY)
    direct_threshold = choose_fixed_specificity_threshold(
        train["parsed_ratio"].to_numpy(float), truth, TARGET_SPECIFICITY
    )
    physics_ratio_calibration = fit_incomplete_ratio_offset(
        train,
        observed_fev1_col="parsed_fev1_l",
        physics_fvc_col="model_fvc_mL",
        truth_ratio_col="fev1_fvc_ratio",
        physics_fvc_scale=0.001,
    )

    def parameters(model: RidgeRegressor, features: tuple[str, ...] = RIDGE_FEATURES) -> dict[str, Any]:
        return {
            "alpha": model.alpha,
            "features": list(features),
            "medians": model.medians_.tolist(),
            "means": model.mean_.tolist(),
            "scales": model.scale_.tolist(),
            "coefficients": model.coef_.tolist(),
            "intercept": model.intercept_,
        }

    return {
        "full": full,
        "truncated": truncated,
        "hybrid": hybrid,
        "frozen": {
            "training_cycle": "F",
            "training_partition": "TRAIN only",
            "full_model": parameters(full),
            "truncated_only_model": parameters(truncated),
            "physics_plus_ridge_hybrid": parameters(hybrid, HYBRID_FEATURES),
            "truncated_train_n": int(len(truncated_train)),
            "full_alpha_search": alpha_search,
            "truncated_alpha_search": truncated_search,
            "hybrid_alpha_search": hybrid_search,
            "full_fixed_specificity_threshold": full_threshold,
            "state_aware_fixed_specificity_threshold": state_threshold,
            "hybrid_state_aware_fixed_specificity_threshold": hybrid_threshold,
            "direct_ratio_fixed_specificity_threshold": direct_threshold,
            "state_aware_physics_ratio_calibration": physics_ratio_calibration,
        },
    }


def downloaded_truth(root: Path, cycle: str) -> pd.DataFrame:
    path = resolve_dataset_path(root, "SPX", cycle).path
    df = pd.read_sas(path, format="xport", encoding=None)
    keep = ["SEQN", "SPXNFEV1", "SPXNFVC", "SPXNFEV6", "SPXNQFVC", "SPXNQFV1", "SPXNQEFF", "SPDNACC"]
    out = df[[col for col in keep if col in df.columns]].copy()
    out = out.rename(
        columns={
            "SPXNFEV1": "fev1_mL",
            "SPXNFVC": "fvc_mL",
            "SPXNFEV6": "fev6_mL",
            "SPXNQFVC": "fvc_quality",
            "SPXNQFV1": "fev1_quality",
            "SPXNQEFF": "effort_quality",
            "SPDNACC": "acceptable_curve_count",
        }
    )
    out["fev1_fvc_ratio"] = out["fev1_mL"] / out["fvc_mL"]
    return out


def prepare_cycle(root: Path, cycle: str, ridges: dict[str, Any], c1: float) -> tuple[dict[str, Any], pd.DataFrame]:
    raw_path = resolve_dataset_path(root, "SPXRAW", cycle).path
    raw, raw_info = read_raw(raw_path, cycle)
    truth = downloaded_truth(root, cycle)
    merged = pq.read_table(root / "results" / "merged_summary.parquet").to_pandas()
    cov = merged[merged["cycle"].eq(cycle)][["SEQN", "age_years", "height_cm", "sex"]].drop_duplicates("SEQN")
    truth = truth.merge(cov, on="SEQN", how="left", validate="one_to_one")
    truth = truth[np.isfinite(truth["fev1_fvc_ratio"])].copy()
    truth["truth_severity"] = truth["fev1_fvc_ratio"].map(severity_category)

    candidates_all = build_usable_raw_candidates(raw)
    candidates = candidates_all[candidates_all["SEQN"].isin(set(truth["SEQN"]))].copy()
    candidates = candidates[candidates["parse_error"].fillna("").eq("")].copy()
    selected = select_best_manoeuvre(candidates)
    fits = fit_candidates_with_checkpoint(root, cycle, raw, candidates)
    fits = fits.merge(truth[["SEQN", "fev1_fvc_ratio", "truth_severity"]], on="SEQN", how="left", validate="many_to_one")
    fits["row_type"] = "maneuver_fit"
    fits = route_endpoints_by_measurement_state(
        fits,
        c1=c1,
        observed_fev1_col="observed_fev1_l",
        observed_fvc_col="observed_fvc_l",
        observed_ratio_col="observed_ratio",
    )

    valid_for_summary = fits[
        fits["volume_fit_success"].astype(bool)
        & fits["fit_error"].fillna("").eq("")
        & np.isfinite(fits["model_ratio"].to_numpy(float))
    ].copy()
    counts = valid_for_summary.groupby("SEQN").size()
    repeat_ids = set(counts[counts >= 2].index)
    pp = participant_sd_rows(valid_for_summary[valid_for_summary["SEQN"].isin(repeat_ids)].copy())
    repro = pd.concat([fits, pp], ignore_index=True, sort=False)

    selected_ats = add_ats_features(raw, selected)
    # Preserve the parser-derived ATS endpoint fields from ``selected_ats``.
    # The checkpoint also carries endpoint fields for auditability, so merge
    # only the fit-specific quantities required downstream.
    fit_columns = [
        "raw_index",
        "model_fev1_l",
        "model_fvc_l",
        "model_ratio",
        "volume_fit_success",
        "volume_r2",
        "volume_nrss_norm",
        "fit_tau_f",
        "fit_tau_s",
        "fit_alpha",
        "fit_fvc_l",
        "fit_tau_o",
        "fit_error",
        "endpoint_source",
        "c1_applied",
        "corrected_ratio_c1",
        "physics_model_fev1_l",
        "physics_model_fvc_l",
        "physics_model_ratio",
        "physics_volume_fit_success",
        "physics_volume_r2",
        "physics_fit_error",
    ]
    selected_fit = selected_ats.merge(
        fits[[column for column in fit_columns if column in fits.columns]],
        on="raw_index",
        how="left",
        validate="one_to_one",
    )
    selected_fit = selected_fit.merge(truth, on="SEQN", how="inner", validate="one_to_one", suffixes=("", "_truth"))
    selected_fit["model_fev1_mL"] = selected_fit["model_fev1_l"] * 1000.0
    selected_fit["model_fvc_mL"] = selected_fit["model_fvc_l"] * 1000.0
    selected_fit["truncated_ats"] = ats_flag(selected_fit)
    selected_fit = route_endpoints_by_measurement_state(
        selected_fit,
        c1=c1,
        observed_fev1_col="parsed_fev1_l",
        observed_fvc_col="parsed_fvc_l",
        observed_ratio_col="parsed_ratio",
    )
    selected_fit["model_fev1_mL"] = selected_fit["model_fev1_l"] * 1000.0
    selected_fit["model_fvc_mL"] = selected_fit["model_fvc_l"] * 1000.0
    selected_fit["split"] = f"frozen_{cycle}"
    selected_fit["truth_severity"] = selected_fit["fev1_fvc_ratio"].map(severity_category)

    workflow = add_workflow_columns(add_repeatability(selected_fit, repro))
    recovered = apply_recovery(workflow, corrected_maneuvers(repro, c1))
    cases = build_case_rows(recovered, raw_maneuvers_with_correction(repro, c1), c1)
    final = apply_ats_policy(cases, include_residual_r1=True)

    x = final[list(RIDGE_FEATURES)].to_numpy(float)
    full_fvc = ridges["full"].predict(x)
    truncated_fvc = ridges["truncated"].predict(x)
    final["physics_recovered_fvc_l"] = pd.to_numeric(final["physics_model_fvc_l"], errors="coerce")
    hybrid_fvc = ridges["hybrid"].predict(final[list(HYBRID_FEATURES)].to_numpy(float))
    archived_ratio = final["parsed_fev1_l"].to_numpy(float) / full_fvc
    truncated_ratio = final["parsed_fev1_l"].to_numpy(float) / truncated_fvc
    state_ratio = np.where(final["truncated_ats"].astype(bool), truncated_ratio, final["parsed_ratio"].to_numpy(float))
    hybrid_state_fvc = np.where(
        final["truncated_ats"].astype(bool), hybrid_fvc, final["parsed_fvc_l"].to_numpy(float)
    )
    hybrid_state_ratio = np.where(
        final["truncated_ats"].astype(bool),
        final["parsed_fev1_l"].to_numpy(float) / hybrid_fvc,
        final["parsed_ratio"].to_numpy(float),
    )

    y = final["fev1_fvc_ratio"].to_numpy(float) < FIXED_RATIO_THRESHOLD
    physics_selected = final["post_ats_finalized"].astype(bool).to_numpy()
    physics_pred = final["post_ats_decision"].isin(FINAL_OBSTRUCTED).to_numpy()
    physics_n = int(physics_selected.sum())
    direct_ratio = final["parsed_ratio"].to_numpy(float)
    direct_pred = direct_ratio < 0.70
    qc_selected = ~final["truncated_ats"].astype(bool).to_numpy()
    fev6 = final["fev6_mL"].to_numpy(float)
    fev6_valid = np.isfinite(fev6) & (fev6 > 0)
    fev6_ratio = final["fev1_mL"].to_numpy(float) / fev6
    fev6_pred = fev6_ratio < FEV1_FEV6_THRESHOLD
    archived_pred = archived_ratio < 0.70
    state_pred = state_ratio < 0.70
    hybrid_pred = hybrid_state_ratio < 0.70
    consensus_selected = physics_selected & (physics_pred == state_pred)
    consensus_n = int(consensus_selected.sum())

    methods: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "state_aware_physics_workflow": (physics_pred, physics_selected),
        "quality_control_first": (direct_pred, qc_selected),
        "fev1_fev6": (fev6_pred, fev6_valid),
        "direct_ratio_matched_coverage": (direct_pred, top_margin_mask(direct_ratio, physics_n)),
        "direct_ratio_on_physics_cases": (direct_pred, physics_selected),
        "archived_ridge_all": (archived_pred, np.ones(len(final), dtype=bool)),
        "archived_ridge_matched_coverage": (archived_pred, top_margin_mask(archived_ratio, physics_n)),
        "state_aware_truncated_ridge": (state_pred, np.ones(len(final), dtype=bool)),
        "state_aware_ridge_matched_coverage": (state_pred, top_margin_mask(state_ratio, physics_n)),
        "state_aware_ridge_on_physics_cases": (state_pred, physics_selected),
        "physics_plus_ridge_hybrid": (hybrid_pred, np.ones(len(final), dtype=bool)),
        "physics_plus_ridge_hybrid_matched_coverage": (
            hybrid_pred,
            top_margin_mask(hybrid_state_ratio, physics_n),
        ),
        "physics_plus_ridge_hybrid_on_physics_cases": (hybrid_pred, physics_selected),
        "physics_ridge_consensus": (physics_pred, consensus_selected),
        "direct_ratio_at_consensus_coverage": (direct_pred, top_margin_mask(direct_ratio, consensus_n)),
        "state_aware_ridge_at_consensus_coverage": (state_pred, top_margin_mask(state_ratio, consensus_n)),
        "state_aware_ridge_on_consensus_cases": (state_pred, consensus_selected),
        "direct_ratio_fixed_specificity": (
            direct_ratio < ridges["frozen"]["direct_ratio_fixed_specificity_threshold"]["threshold"],
            np.ones(len(final), dtype=bool),
        ),
        "archived_ridge_fixed_specificity": (
            archived_ratio < ridges["frozen"]["full_fixed_specificity_threshold"]["threshold"],
            np.ones(len(final), dtype=bool),
        ),
        "state_aware_ridge_fixed_specificity": (
            state_ratio < ridges["frozen"]["state_aware_fixed_specificity_threshold"]["threshold"],
            np.ones(len(final), dtype=bool),
        ),
        "physics_plus_ridge_hybrid_fixed_specificity": (
            hybrid_state_ratio
            < ridges["frozen"]["hybrid_state_aware_fixed_specificity_threshold"]["threshold"],
            np.ones(len(final), dtype=bool),
        ),
    }
    metrics = {
        name: metric_record(name, y, pred, selected_mask, SEED + 100 * (ord(cycle) - 64) + i)
        for i, (name, (pred, selected_mask)) in enumerate(methods.items(), start=1)
    }

    paired = {
        name: paired_bootstrap_difference(y, physics_pred, physics_selected, pred, mask, SEED + 500 + i)
        for i, (name, (pred, mask)) in enumerate(methods.items(), start=1)
        if name != "state_aware_physics_workflow"
    }
    paired_consensus = {
        name: paired_bootstrap_difference(
            y,
            physics_pred,
            consensus_selected,
            methods[name][0],
            methods[name][1],
            SEED + 650 + i,
        )
        for i, name in enumerate(
            ["direct_ratio_at_consensus_coverage", "state_aware_ridge_at_consensus_coverage"], start=1
        )
    }

    model_fvc = final["model_fvc_l"].to_numpy(float)
    standalone_physics_fvc = final["physics_model_fvc_l"].to_numpy(float)
    truth_fvc_l = final["fvc_mL"].to_numpy(float) / 1000.0
    state_ridge_fvc = np.where(
        final["truncated_ats"].astype(bool), truncated_fvc, final["parsed_fvc_l"].to_numpy(float)
    )
    no_complete_ids = set(
        fits.groupby("SEQN")["truncated_ats"].apply(lambda s: bool(s.astype(bool).all())).loc[lambda s: s].index
    )
    no_complete = final["SEQN"].isin(no_complete_ids).to_numpy()
    fvc_errors = {
        "direct_observed": bootstrap_error((final["parsed_fvc_l"].to_numpy(float) - truth_fvc_l) * 1000.0, SEED + 700),
        "physics_model": bootstrap_error((model_fvc - truth_fvc_l) * 1000.0, SEED + 701),
        "state_aware_physics": bootstrap_error((model_fvc - truth_fvc_l) * 1000.0, SEED + 711),
        "standalone_physics_all": bootstrap_error(
            (standalone_physics_fvc - truth_fvc_l) * 1000.0, SEED + 712
        ),
        "archived_ridge": bootstrap_error((full_fvc - truth_fvc_l) * 1000.0, SEED + 702),
        "truncated_only_ridge": bootstrap_error((truncated_fvc - truth_fvc_l) * 1000.0, SEED + 703),
        "state_aware_ridge": bootstrap_error((state_ridge_fvc - truth_fvc_l) * 1000.0, SEED + 713),
        "state_aware_physics_plus_ridge_hybrid": bootstrap_error(
            (hybrid_state_fvc - truth_fvc_l) * 1000.0, SEED + 714
        ),
        "no_complete_direct_observed": bootstrap_error((final.loc[no_complete, "parsed_fvc_l"].to_numpy(float) - truth_fvc_l[no_complete]) * 1000.0, SEED + 704),
        "no_complete_physics_model": bootstrap_error((model_fvc[no_complete] - truth_fvc_l[no_complete]) * 1000.0, SEED + 705),
        "no_complete_archived_ridge": bootstrap_error((full_fvc[no_complete] - truth_fvc_l[no_complete]) * 1000.0, SEED + 706),
        "no_complete_truncated_only_ridge": bootstrap_error((truncated_fvc[no_complete] - truth_fvc_l[no_complete]) * 1000.0, SEED + 707),
        "no_complete_physics_plus_ridge_hybrid": bootstrap_error(
            (hybrid_state_fvc[no_complete] - truth_fvc_l[no_complete]) * 1000.0, SEED + 715
        ),
    }
    physics_error_ml = (model_fvc - truth_fvc_l) * 1000.0
    fvc_paired = {
        "physics_minus_direct_observed": paired_bootstrap_mae_difference(
            physics_error_ml,
            (final["parsed_fvc_l"].to_numpy(float) - truth_fvc_l) * 1000.0,
            SEED + 720,
        ),
        "physics_minus_archived_ridge": paired_bootstrap_mae_difference(
            physics_error_ml, (full_fvc - truth_fvc_l) * 1000.0, SEED + 721
        ),
        "physics_minus_truncated_only_ridge": paired_bootstrap_mae_difference(
            physics_error_ml, (truncated_fvc - truth_fvc_l) * 1000.0, SEED + 722
        ),
        "state_aware_physics_minus_state_aware_ridge": paired_bootstrap_mae_difference(
            physics_error_ml, (state_ridge_fvc - truth_fvc_l) * 1000.0, SEED + 724
        ),
        "state_aware_physics_minus_hybrid": paired_bootstrap_mae_difference(
            physics_error_ml, (hybrid_state_fvc - truth_fvc_l) * 1000.0, SEED + 725
        ),
        "hybrid_minus_state_aware_ridge": paired_bootstrap_mae_difference(
            (hybrid_state_fvc - truth_fvc_l) * 1000.0,
            (state_ridge_fvc - truth_fvc_l) * 1000.0,
            SEED + 726,
        ),
        "no_complete_physics_minus_truncated_only_ridge": paired_bootstrap_mae_difference(
            physics_error_ml[no_complete],
            (truncated_fvc[no_complete] - truth_fvc_l[no_complete]) * 1000.0,
            SEED + 723,
        ),
        "no_complete_hybrid_minus_truncated_only_ridge": paired_bootstrap_mae_difference(
            (hybrid_state_fvc[no_complete] - truth_fvc_l[no_complete]) * 1000.0,
            (truncated_fvc[no_complete] - truth_fvc_l[no_complete]) * 1000.0,
            SEED + 727,
        ),
    }

    final["physics_prediction_obstructed"] = physics_pred
    final["physics_finalized"] = physics_selected
    final["archived_ridge_fvc_l"] = full_fvc
    final["archived_ridge_ratio"] = archived_ratio
    final["truncated_only_ridge_fvc_l"] = truncated_fvc
    final["state_aware_ridge_ratio"] = state_ratio
    final["state_aware_ridge_fvc_l"] = state_ridge_fvc
    final["physics_plus_ridge_hybrid_fvc_l"] = hybrid_state_fvc
    final["physics_plus_ridge_hybrid_ratio"] = hybrid_state_ratio
    final["no_complete_acceptable_manoeuvre"] = no_complete
    final.to_csv(root / "results" / f"frozen_validation_{cycle}_participants.csv", index=False)

    deferred = final[~final["post_ats_finalized"].astype(bool)].copy()
    deferral_counts = (
        deferred["primary_deferral_reason"].replace("", "UNSPECIFIED").fillna("UNSPECIFIED").value_counts().to_dict()
    )
    cohort = {
        "summary_rows": int(len(downloaded_truth(root, cycle))),
        "truth_with_fev1_fvc": int(len(truth)),
        "raw_rows": int(len(raw)),
        "raw_participants": int(raw["SEQN"].nunique()),
        "acceptable_ab_candidate_rows": int(len(candidates)),
        "eligible_participants_with_candidate": int(len(selected)),
        "successful_selected_routed_endpoints": int(final["volume_fit_success"].astype(bool).sum()),
        "selected_physics_recovery_n": int(final["endpoint_source"].eq("physics_recovered_fvc").sum()),
        "successful_selected_physics_fits": int(
            (
                final["endpoint_source"].eq("physics_recovered_fvc")
                & final["physics_volume_fit_success"].astype(bool)
            ).sum()
        ),
        "no_complete_acceptable_manoeuvre_n": int(no_complete.sum()),
        "class_distribution": final["truth_severity"].value_counts().to_dict(),
        "deferred_n": int(len(deferred)),
        "deferral_reasons": deferral_counts,
        "deferred_severity": deferred["truth_severity"].value_counts().to_dict(),
        "raw_info": raw_info,
    }
    no_complete_metrics = {
        name: metric_record(name, y[no_complete], pred[no_complete], mask[no_complete], SEED + 800 + i)
        for i, (name, (pred, mask)) in enumerate(methods.items(), start=1)
    }
    result = {
        "cycle": cycle,
        "status": "post-audit locked cross-cycle re-evaluation",
        "cohort": cohort,
        "metrics": metrics,
        "paired_differences_physics_minus_comparator": paired,
        "paired_differences_consensus_minus_comparator": paired_consensus,
        "fvc_error": fvc_errors,
        "paired_fvc_mae_differences": fvc_paired,
        "no_complete_acceptable_manoeuvre_metrics": no_complete_metrics,
    }
    return clean(result), final


def consolidated_table(results: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for cycle, result in results.items():
        for method, metric in result["metrics"].items():
            rows.append(
                {
                    "cycle": cycle,
                    "method": method,
                    "n": metric["n"],
                    "coverage": metric["coverage"],
                    "sensitivity": metric["sensitivity"],
                    "specificity": metric["specificity"],
                    "kappa": metric["kappa"],
                    "false_negatives": metric["fn"],
                    "false_positives": metric["fp"],
                }
            )
    return pd.DataFrame(rows)


def scientific_conclusion(results: dict[str, Any]) -> dict[str, Any]:
    comparisons = []
    for cycle, result in results.items():
        p = result["metrics"]["state_aware_physics_workflow"]
        r = result["metrics"]["archived_ridge_matched_coverage"]
        s = result["metrics"]["state_aware_ridge_matched_coverage"]
        comparisons.append(
            {
                "cycle": cycle,
                "physics_kappa": p["kappa"],
                "archived_ridge_matched_kappa": r["kappa"],
                "state_aware_ridge_matched_kappa": s["kappa"],
                "hybrid_matched_kappa": result["metrics"]["physics_plus_ridge_hybrid_matched_coverage"]["kappa"],
                "physics_coverage": p["coverage"],
                "physics_fvc_mae_ml": result["fvc_error"]["state_aware_physics"]["mae_ml"],
                "archived_ridge_fvc_mae_ml": result["fvc_error"]["archived_ridge"]["mae_ml"],
                "truncated_only_ridge_fvc_mae_ml": result["fvc_error"]["truncated_only_ridge"]["mae_ml"],
                "state_aware_ridge_fvc_mae_ml": result["fvc_error"]["state_aware_ridge"]["mae_ml"],
                "hybrid_fvc_mae_ml": result["fvc_error"]["state_aware_physics_plus_ridge_hybrid"]["mae_ml"],
                "consensus_kappa": result["metrics"]["physics_ridge_consensus"]["kappa"],
                "ridge_at_consensus_coverage_kappa": result["metrics"]["state_aware_ridge_at_consensus_coverage"]["kappa"],
                "direct_at_consensus_coverage_kappa": result["metrics"]["direct_ratio_at_consensus_coverage"]["kappa"],
            }
        )
    physics_wins_kappa = sum(
        row["physics_kappa"] > row["state_aware_ridge_matched_kappa"] for row in comparisons
    )
    hybrid_wins_fvc = sum(
        row["hybrid_fvc_mae_ml"] < row["state_aware_ridge_fvc_mae_ml"] for row in comparisons
    )
    return {
        "cycle_comparisons": comparisons,
        "physics_wins_state_aware_matched_kappa_cycles": int(physics_wins_kappa),
        "hybrid_wins_state_aware_ridge_fvc_cycles": int(hybrid_wins_fvc),
        "evidence_summary": (
            "The corrected measurement-state implementation routes complete manoeuvres to direct endpoints and incomplete "
            "manoeuvres to observed FEV1 with recovered FVC. The standalone physics branch did not exceed state-aware ridge "
            "at matched coverage. A Cycle-F-trained physics-plus-ridge hybrid is reported separately to test whether recovered "
            "FVC contributes incremental endpoint information beyond the clinical and waveform-prefix features already available "
            "to ridge. Agreement selection is retained as a distinct reliability analysis."
        ),
        "interpretation_rule": (
            "Physics is credited with incremental predictive value only when its paired cross-cycle difference excludes zero "
            "or when it supplies a separately measured advantage in endpoint error, deferral safety, or robustness."
        ),
        "implementation_revision": (
            "Post-audit correction: prior scoring applied physics-derived endpoints to complete manoeuvres and used modeled "
            "FEV1. Current scoring uses direct complete endpoints and observed FEV1 for censored recovery. The archived prior "
            "outputs are retained under results/pre_state_routing_audit."
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any], table: pd.DataFrame) -> None:
    lines = [
        "# Locked Post-Audit Cycle E and G Evaluation",
        "",
        "All model coefficients, the Cycle F constant ratio correction, physical parameter bounds, workflow thresholds, recovery rules, and selection guards were frozen before Cycle E and G outcomes were scored.",
        "",
        "## Consolidated Comparison",
        "",
        "| Cycle | Method | n | Coverage | Sensitivity | Specificity | Kappa | FN | FP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.cycle} | {row.method} | {row.n} | {row.coverage:.3f} | {row.sensitivity:.3f} | "
            f"{row.specificity:.3f} | {row.kappa:.3f} | {row.false_negatives} | {row.false_positives} |"
        )
    lines.extend(["", "## Cohort Accounting", ""])
    for cycle, result in payload["cycles"].items():
        c = result["cohort"]
        lines.extend(
            [
                f"### Cycle {cycle}",
                "",
                f"The evaluation included {c['eligible_participants_with_candidate']} participants with an acceptable A/B pre-bronchodilator manoeuvre. "
                f"The physics workflow deferred {c['deferred_n']} participants, and {c['no_complete_acceptable_manoeuvre_n']} participants had no complete acceptable manoeuvre under the frozen ATS endpoint rule.",
                "",
                f"Deferral reasons: {c['deferral_reasons']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "The archived ridge result is retained for audit continuity, but it uses the terminal volume of the full selected manoeuvre and is therefore not a clean censored-prefix comparator. The state-aware truncated-only ridge is the more relevant learned baseline for participants whose endpoint is incomplete.",
            "",
            "The corrected workflow routes complete manoeuvres to direct endpoints and invokes physics recovery only for incomplete endpoints. The consolidated table reports standalone physics, state-aware ridge, and the Cycle-F-trained physics-plus-ridge hybrid under identical participant eligibility and coverage definitions.",
            "",
            "Paired participant-bootstrap differences, endpoint-error estimates, no-complete-manoeuvre results, and full machine-readable cohort details are stored in the companion JSON.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cycles", nargs="+", default=["E", "G"], choices=["E", "G"])
    args = parser.parse_args()
    root = args.root.resolve()
    ridges = load_frozen_ridges(root)
    c1 = float(ridges["frozen"]["state_aware_physics_ratio_calibration"]["bias"])
    frozen = {
        "frozen_before_current_cross_cycle_scoring": True,
        "prior_cycle_exposure": (
            "Cycles E and G appeared in earlier ancillary truncation and subgroup analyses; the present participant-level "
            "final-workflow scoring was locked, but is not a pristine independent validation."
        ),
        "c1_ratio_correction": c1,
        "c1_ratio_correction_provenance": ridges["frozen"]["state_aware_physics_ratio_calibration"],
        "endpoint_routing": {
            "complete": "direct observed FEV1, FVC, and ratio",
            "incomplete": "direct observed FEV1 divided by physics-recovered FVC, followed by Cycle F TRAIN correction",
            "reference_outcomes_used_for_routing": False,
        },
        "physics_parameter_bounds": {
            "tau_f_s": [0.15, 0.80],
            "tau_s_s": [1.00, 5.00],
            "alpha": [0.25, 0.85],
            "fvc_l": [2.00, 6.00],
            "tau_o_s": [0.03, 0.40],
        },
        "ats_endpoint_rule": {"minimum_duration_s": 6.0, "maximum_last_second_volume_l": 0.025},
        "workflow_thresholds": {
            "fixed_obstruction_ratio": 0.70,
            "severe_referral_below": 0.50,
            "borderline_repeat_interval": [0.60, 0.72],
            "repeatability_ratio_sd": 0.030,
        },
        "recovery_thresholds": {
            "R1_volume_r2": 0.98,
            "R1_obstructed_ratio_below": 0.60,
            "R1_normal_ratio_at_least": 0.80,
            "R2_obstructed_median_below": 0.67,
            "R2_normal_median_at_least": 0.73,
            "R3_volume_r2": 0.95,
            "R3_cluster_width": 0.020,
            "R3_obstructed_high_below": 0.65,
            "R3_normal_low_at_least": 0.75,
        },
        "ridge": ridges["frozen"],
    }
    (root / "results" / "frozen_validation_configuration.json").write_text(
        json.dumps(clean(frozen), indent=2) + "\n", encoding="utf-8"
    )

    cycle_results: dict[str, Any] = {}
    for cycle in args.cycles:
        result, _ = prepare_cycle(root, cycle, ridges, c1)
        cycle_results[cycle] = result
        (root / "results" / f"frozen_validation_{cycle}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )

    table = consolidated_table(cycle_results)
    table.to_csv(root / "results" / "frozen_validation_consolidated.csv", index=False)
    payload = {
        "study_status": "post-audit locked cross-cycle re-evaluation; cycles share the NHANES program and outcome construction",
        "source_files": {
            "SPXRAW_E.zip": sha256(root / "data" / "nhanes" / "public" / "SPXRAW_E.zip"),
            "SPX_E.xpt": sha256(root / "data" / "nhanes" / "public" / "SPX_E.xpt"),
            "SPXRAW_G.zip": sha256(root / "data" / "nhanes" / "public" / "SPXRAW_G.zip"),
            "SPX_G.xpt": sha256(root / "data" / "nhanes" / "public" / "SPX_G.xpt"),
        },
        "frozen_configuration": frozen,
        "cycles": cycle_results,
        "scientific_conclusion": scientific_conclusion(cycle_results),
    }
    out_json = root / "results" / "frozen_cross_cycle_validation.json"
    out_md = root / "results" / "frozen_cross_cycle_validation.md"
    out_json.write_text(json.dumps(clean(payload), indent=2) + "\n", encoding="utf-8")
    write_markdown(out_md, payload, table)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
