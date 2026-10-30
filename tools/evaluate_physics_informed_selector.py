from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import ExtraTreesClassifier

from tools.evaluate_manuscript_baselines import (
    FIXED_RATIO_THRESHOLD,
    RIDGE_ALPHAS,
    RIDGE_FEATURES,
    RidgeRegressor,
    participant_folds,
    score_binary,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261029
BOOTSTRAP_REPLICATES = 2000
COVERAGE = 0.75
MINIMUM_DEVELOPMENT_COVERAGE = 0.75
COVERAGE_GRID = tuple(float(value) for value in np.arange(0.75, 0.951, 0.025))
BASE_FEATURES = ("ridge_margin", "duration_s", "dV_last_L", "truncated_ats")
PHYSICS_FEATURES = (
    *BASE_FEATURES,
    "physics_margin",
    "physics_ridge_disagreement",
    "signed_ratio_disagreement",
    "physics_ridge_fvc_gap_l",
    "physics_volume_r2",
)


@dataclass
class BasePrediction:
    ratio: np.ndarray
    fvc_l: np.ndarray
    physics_ratio: np.ndarray


def _numeric(frame: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(float)


def _ridge_alpha(frame: pd.DataFrame) -> float:
    truncated = frame[frame["truncated_ats"].astype(bool)].copy()
    x = truncated[list(RIDGE_FEATURES)].to_numpy(float)
    y = _numeric(truncated, "fvc_mL") / 1000.0
    folds = participant_folds(truncated["SEQN"].to_numpy())
    rows: list[tuple[float, float]] = []
    for alpha in RIDGE_ALPHAS:
        pred = np.full(len(truncated), np.nan)
        for fold in np.unique(folds):
            fit = folds != fold
            val = folds == fold
            pred[val] = RidgeRegressor(float(alpha)).fit(x[fit], y[fit]).predict(x[val])
        rows.append((float(np.mean(np.abs(pred - y))), float(alpha)))
    return min(rows)[1]


def _fit_base(train: pd.DataFrame, alpha: float) -> tuple[RidgeRegressor, float]:
    truncated = train[train["truncated_ats"].astype(bool)].copy()
    ridge = RidgeRegressor(alpha).fit(
        truncated[list(RIDGE_FEATURES)].to_numpy(float),
        _numeric(truncated, "fvc_mL") / 1000.0,
    )
    physics_ratio = _numeric(truncated, "parsed_fev1_l") / (_numeric(truncated, "model_fvc_mL") / 1000.0)
    truth_ratio = _numeric(truncated, "fev1_fvc_ratio")
    correction = float(np.nanmedian(physics_ratio - truth_ratio))
    return ridge, correction


def _predict_base(frame: pd.DataFrame, ridge: RidgeRegressor, correction: float) -> BasePrediction:
    truncated = frame["truncated_ats"].astype(bool).to_numpy()
    ridge_fvc = ridge.predict(frame[list(RIDGE_FEATURES)].to_numpy(float))
    observed_fvc = _numeric(frame, "parsed_fvc_l")
    fvc = np.where(truncated, ridge_fvc, observed_fvc)
    ratio = np.where(truncated, _numeric(frame, "parsed_fev1_l") / ridge_fvc, _numeric(frame, "parsed_ratio"))
    physics_fvc = _numeric(frame, "model_fvc_mL") / 1000.0
    physics_ratio = np.where(
        truncated,
        _numeric(frame, "parsed_fev1_l") / physics_fvc - correction,
        _numeric(frame, "parsed_ratio"),
    )
    return BasePrediction(ratio=ratio, fvc_l=fvc, physics_ratio=physics_ratio)


def _features(frame: pd.DataFrame, base: BasePrediction) -> pd.DataFrame:
    physics_fvc = _numeric(frame, "model_fvc_mL") / 1000.0
    return pd.DataFrame(
        {
            "ridge_margin": np.abs(base.ratio - FIXED_RATIO_THRESHOLD),
            "duration_s": _numeric(frame, "duration_s"),
            "dV_last_L": _numeric(frame, "dV_last_L"),
            "truncated_ats": frame["truncated_ats"].astype(bool).to_numpy(float),
            "physics_margin": np.abs(base.physics_ratio - FIXED_RATIO_THRESHOLD),
            "physics_ridge_disagreement": (base.physics_ratio < FIXED_RATIO_THRESHOLD)
            != (base.ratio < FIXED_RATIO_THRESHOLD),
            "signed_ratio_disagreement": base.physics_ratio - base.ratio,
            "physics_ridge_fvc_gap_l": physics_fvc - base.fvc_l,
            "physics_volume_r2": _numeric(frame, "volume_r2"),
        }
    ).replace([np.inf, -np.inf], np.nan)


def _impute_fit(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    medians = train.median(axis=0).fillna(0.0)
    return train.fillna(medians).to_numpy(float), test.fillna(medians).to_numpy(float)


def _selector() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=5,
        max_features=0.7,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def _top_score(score: np.ndarray, coverage: float = COVERAGE) -> np.ndarray:
    n = round(len(score) * coverage)
    order = np.argsort(-score, kind="stable")
    selected = np.zeros(len(score), dtype=bool)
    selected[order[:n]] = True
    return selected


def _metric_subset(metric: dict) -> dict:
    return {
        key: (int(value) if key in {"n", "tp", "tn", "fp", "fn"} else float(value))
        for key, value in metric.items()
    }


def _paired_bootstrap(
    truth: np.ndarray,
    prediction_a: np.ndarray,
    selected_a: np.ndarray,
    prediction_b: np.ndarray,
    selected_b: np.ndarray,
    *,
    seed: int,
) -> dict[str, dict[str, float | list[float]]]:
    names = ("sensitivity", "specificity", "kappa")
    observed_a = score_binary(truth, prediction_a, selected_a)
    observed_b = score_binary(truth, prediction_b, selected_b)
    draws = {name: np.empty(BOOTSTRAP_REPLICATES) for name in names}
    rng = np.random.default_rng(seed)
    for replicate in range(BOOTSTRAP_REPLICATES):
        take = rng.integers(0, len(truth), size=len(truth))
        metric_a = score_binary(truth[take], prediction_a[take], selected_a[take])
        metric_b = score_binary(truth[take], prediction_b[take], selected_b[take])
        for name in names:
            draws[name][replicate] = metric_a[name] - metric_b[name]
    return {
        name: {
            "difference": float(observed_a[name] - observed_b[name]),
            "95_ci": [
                float(np.nanpercentile(draws[name], 2.5)),
                float(np.nanpercentile(draws[name], 97.5)),
            ],
        }
        for name in names
    }


def _bootstrap_intervals(
    truth: np.ndarray,
    prediction: np.ndarray,
    selected: np.ndarray,
    *,
    seed: int,
) -> dict[str, list[float]]:
    names = ("coverage", "sensitivity", "specificity", "kappa")
    draws = {name: np.empty(BOOTSTRAP_REPLICATES) for name in names}
    rng = np.random.default_rng(seed)
    for replicate in range(BOOTSTRAP_REPLICATES):
        take = rng.integers(0, len(truth), size=len(truth))
        metric = score_binary(truth[take], prediction[take], selected[take])
        for name in names:
            draws[name][replicate] = metric[name]
    return {
        name: [
            float(np.nanpercentile(values, 2.5)),
            float(np.nanpercentile(values, 97.5)),
        ]
        for name, values in draws.items()
    }


def _load_f() -> pd.DataFrame:
    policy = pd.read_csv(ROOT / "results/gate_policy_calibrated_flag_F_rows.csv", low_memory=False)
    merged = pq.read_table(ROOT / "results/merged_summary.parquet").to_pandas()
    cov = merged[merged["cycle"].eq("F")][["SEQN", "age_years", "height_cm", "sex"]].drop_duplicates("SEQN")
    return policy.merge(cov, on="SEQN", how="left", validate="one_to_one").query("split == 'train'").reset_index(drop=True)


def _oof_meta(data: pd.DataFrame, alpha: float, columns: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outer = participant_folds(data["SEQN"].to_numpy())
    score = np.full(len(data), np.nan)
    ratio = np.full(len(data), np.nan)
    for fold in sorted(np.unique(outer)):
        development = data.loc[outer != fold].reset_index(drop=True)
        validation = data.loc[outer == fold].reset_index(drop=True)
        dev_folds = participant_folds(development["SEQN"].to_numpy())
        dev_base_ratio = np.full(len(development), np.nan)
        dev_base_fvc = np.full(len(development), np.nan)
        dev_physics_ratio = np.full(len(development), np.nan)
        for inner in sorted(np.unique(dev_folds)):
            fit = development.loc[dev_folds != inner]
            val_mask = dev_folds == inner
            ridge, correction = _fit_base(fit, alpha)
            predicted = _predict_base(development.loc[val_mask], ridge, correction)
            dev_base_ratio[val_mask] = predicted.ratio
            dev_base_fvc[val_mask] = predicted.fvc_l
            dev_physics_ratio[val_mask] = predicted.physics_ratio
        dev_base = BasePrediction(dev_base_ratio, dev_base_fvc, dev_physics_ratio)
        dev_x = _features(development, dev_base)[list(columns)]
        dev_truth = _numeric(development, "fev1_fvc_ratio") < FIXED_RATIO_THRESHOLD
        dev_pred = dev_base.ratio < FIXED_RATIO_THRESHOLD
        correct = dev_pred == dev_truth

        ridge, correction = _fit_base(development, alpha)
        val_base = _predict_base(validation, ridge, correction)
        val_x = _features(validation, val_base)[list(columns)]
        x_train, x_val = _impute_fit(dev_x, val_x)
        model = _selector().fit(x_train, correct)
        val_indices = np.flatnonzero(outer == fold)
        score[val_indices] = model.predict_proba(x_val)[:, list(model.classes_).index(True)]
        ratio[val_indices] = val_base.ratio
    truth = _numeric(data, "fev1_fvc_ratio") < FIXED_RATIO_THRESHOLD
    return score, ratio, truth


def _final_meta(data: pd.DataFrame, alpha: float, columns: tuple[str, ...]):
    folds = participant_folds(data["SEQN"].to_numpy())
    ratio = np.full(len(data), np.nan)
    fvc = np.full(len(data), np.nan)
    physics_ratio = np.full(len(data), np.nan)
    for fold in sorted(np.unique(folds)):
        ridge, correction = _fit_base(data.loc[folds != fold], alpha)
        predicted = _predict_base(data.loc[folds == fold], ridge, correction)
        ratio[folds == fold] = predicted.ratio
        fvc[folds == fold] = predicted.fvc_l
        physics_ratio[folds == fold] = predicted.physics_ratio
    base = BasePrediction(ratio, fvc, physics_ratio)
    x = _features(data, base)[list(columns)]
    truth = _numeric(data, "fev1_fvc_ratio") < FIXED_RATIO_THRESHOLD
    correct = (ratio < FIXED_RATIO_THRESHOLD) == truth
    medians = x.median(axis=0).fillna(0.0)
    model = _selector().fit(x.fillna(medians).to_numpy(float), correct)
    ridge, correction = _fit_base(data, alpha)
    return model, medians, ridge, correction


def _score_cycle(cycle: str, trained: dict[str, tuple]) -> tuple[dict, pd.DataFrame]:
    frame = pd.read_csv(ROOT / f"results/frozen_validation_{cycle}_participants.csv", low_memory=False)
    frame["model_fvc_mL"] = _numeric(frame, "physics_model_fvc_l") * 1000.0
    frame["volume_r2"] = _numeric(frame, "physics_volume_r2")
    truth = frame["truth_obstructed_fixed"].astype(bool).to_numpy()
    ratio = _numeric(frame, "state_aware_ridge_ratio")
    pred = ratio < FIXED_RATIO_THRESHOLD
    outcomes: dict[str, dict] = {}
    masks: dict[str, np.ndarray] = {}
    reliability_scores: dict[str, np.ndarray] = {}
    physics_base: BasePrediction | None = None
    physics_feature_frame: pd.DataFrame | None = None
    for name, (model, medians, ridge, correction, columns) in trained.items():
        base = _predict_base(frame, ridge, correction)
        if not np.allclose(base.ratio, ratio, atol=1e-10, equal_nan=True):
            raise AssertionError(f"{cycle}: reconstructed state-aware ridge differs from frozen artifact")
        x = _features(frame, base)[list(columns)].fillna(medians).to_numpy(float)
        reliability = model.predict_proba(x)[:, list(model.classes_).index(True)]
        reliability_scores[name] = reliability
        if name == "physics_reliability":
            physics_base = base
            physics_feature_frame = _features(frame, base)
        masks[name] = _top_score(reliability)
        outcomes[name] = _metric_subset(score_binary(truth, pred, masks[name]))
    masks["ridge_margin"] = _top_score(np.abs(ratio - FIXED_RATIO_THRESHOLD))
    outcomes["ridge_margin"] = _metric_subset(score_binary(truth, pred, masks["ridge_margin"]))
    direct = _numeric(frame, "parsed_ratio")
    direct_pred = direct < FIXED_RATIO_THRESHOLD
    masks["direct_margin"] = _top_score(np.abs(direct - FIXED_RATIO_THRESHOLD))
    outcomes["direct_margin"] = _metric_subset(score_binary(truth, direct_pred, masks["direct_margin"]))
    for index, name in enumerate(outcomes, start=1):
        method_prediction = direct_pred if name == "direct_margin" else pred
        outcomes[name]["bootstrap_95_ci"] = _bootstrap_intervals(
            truth,
            method_prediction,
            masks[name],
            seed=SEED + 10_000 * ord(cycle) + index,
        )

    comparisons = {}
    for index, name in enumerate(("ridge_reliability", "ridge_margin", "direct_margin"), start=1):
        comparator_prediction = direct_pred if name == "direct_margin" else pred
        comparisons[f"physics_reliability_minus_{name}"] = _paired_bootstrap(
            truth,
            pred,
            masks["physics_reliability"],
            comparator_prediction,
            masks[name],
            seed=SEED + 100 * ord(cycle) + index,
        )

    severity = frame["truth_severity"].fillna("Unknown").astype(str)
    severe = severity.str.lower().isin({"severe", "very severe"}).to_numpy()
    selected = masks["physics_reliability"]
    deferred = ~selected
    no_complete = frame["no_complete_acceptable_manoeuvre"].astype(bool).to_numpy()
    if physics_base is None or physics_feature_frame is None:
        raise AssertionError("Physics feature audit was not generated")
    participant_rows = pd.DataFrame(
        {
            "SEQN": frame["SEQN"].astype(int),
            "cycle": cycle,
            "truth_obstructed": truth,
            "truth_severity": severity,
            "state_aware_ridge_prediction": pred,
            "state_aware_ridge_ratio": ratio,
            "physics_ratio": physics_base.physics_ratio,
            "physics_reliability_score": reliability_scores["physics_reliability"],
            "nonphysics_reliability_score": reliability_scores["ridge_reliability"],
            "physics_informed_selected": selected,
            "ridge_reliability_selected": masks["ridge_reliability"],
            "ridge_margin_selected": masks["ridge_margin"],
            "direct_margin_selected": masks["direct_margin"],
            "no_complete_acceptable_manoeuvre": frame["no_complete_acceptable_manoeuvre"].astype(bool),
        }
    )
    audit = {
        "cycle": cycle,
        "coverage_target": COVERAGE,
        "metrics": outcomes,
        "paired_participant_bootstrap_differences": comparisons,
        "no_complete_acceptable_manoeuvre_metrics": {
            name: _metric_subset(
                score_binary(
                    truth[no_complete],
                    (direct_pred if name == "direct_margin" else pred)[no_complete],
                    masks[name][no_complete],
                )
            )
            for name in outcomes
        },
        "physics_informed_selection": {
            "selected_n": int(selected.sum()),
            "deferred_n": int(deferred.sum()),
            "selected_severity": severity[selected].value_counts().to_dict(),
            "deferred_severity": severity[deferred].value_counts().to_dict(),
            "selected_no_complete_acceptable_manoeuvre_n": int(
                (selected & frame["no_complete_acceptable_manoeuvre"].astype(bool).to_numpy()).sum()
            ),
            "deferred_no_complete_acceptable_manoeuvre_n": int(
                (deferred & frame["no_complete_acceptable_manoeuvre"].astype(bool).to_numpy()).sum()
            ),
            "severe_selected_n": int((selected & severe).sum()),
            "severe_false_negatives": int((selected & severe & truth & ~pred).sum()),
            "deferred_feature_counts": {
                "truncated_measurement": int(
                    (deferred & frame["truncated_ats"].astype(bool).to_numpy()).sum()
                ),
                "physics_ridge_class_disagreement": int(
                    (deferred & physics_feature_frame["physics_ridge_disagreement"].astype(bool).to_numpy()).sum()
                ),
                "ridge_ratio_within_0_05_of_boundary": int(
                    (deferred & (np.abs(ratio - FIXED_RATIO_THRESHOLD) < 0.05)).sum()
                ),
                "physics_volume_r2_below_0_90": int(
                    (
                        deferred
                        & (physics_feature_frame["physics_volume_r2"].fillna(-np.inf).to_numpy(float) < 0.90)
                    ).sum()
                ),
                "physics_ridge_fvc_gap_above_0_25_l": int(
                    (
                        deferred
                        & (np.abs(physics_feature_frame["physics_ridge_fvc_gap_l"].to_numpy(float)) > 0.25)
                    ).sum()
                ),
            },
        },
    }
    return audit, participant_rows


def main() -> None:
    data = _load_f()
    alpha = _ridge_alpha(data)
    trained = {}
    development: dict[str, dict] = {}
    for name, columns in (("ridge_reliability", BASE_FEATURES), ("physics_reliability", PHYSICS_FEATURES)):
        score, ratio, truth = _oof_meta(data, alpha, columns)
        coverage_search = []
        for coverage in COVERAGE_GRID:
            metric = _metric_subset(
                score_binary(truth, ratio < FIXED_RATIO_THRESHOLD, _top_score(score, coverage))
            )
            metric["target_coverage"] = coverage
            coverage_search.append(metric)
        best = max(coverage_search, key=lambda row: (row["kappa"], row["specificity"], row["sensitivity"]))
        development[name] = {
            "features": list(columns),
            "participant_separated_nested_oof": True,
            "minimum_coverage_constraint": MINIMUM_DEVELOPMENT_COVERAGE,
            "coverage_search": coverage_search,
            "selected_operating_point": best,
        }
        model, medians, ridge, correction = _final_meta(data, alpha, columns)
        trained[name] = (model, medians, ridge, correction, columns)
    selected_coverage = development["physics_reliability"]["selected_operating_point"]["target_coverage"]
    if not np.isclose(selected_coverage, COVERAGE):
        raise AssertionError(f"Expected the Cycle F selector to freeze {COVERAGE:.3f} coverage; got {selected_coverage}")
    cycle_results = {}
    participant_tables = []
    for cycle in ("E", "G"):
        cycle_results[cycle], participant_table = _score_cycle(cycle, trained)
        participant_tables.append(participant_table)

    endpoint = json.loads((ROOT / "results/frozen_cross_cycle_validation.json").read_text())
    endpoint_comparison = {
        cycle: {
            "state_aware_ridge_fvc_mae_ml": endpoint["cycles"][cycle]["fvc_error"]["state_aware_ridge"]["mae_ml"],
            "physics_plus_ridge_fvc_mae_ml": endpoint["cycles"][cycle]["fvc_error"]["state_aware_physics_plus_ridge_hybrid"]["mae_ml"],
            "paired_hybrid_minus_ridge_mae_ml": endpoint["cycles"][cycle]["paired_fvc_mae_differences"]["hybrid_minus_state_aware_ridge"],
        }
        for cycle in ("E", "G")
    }
    payload = {
        "status": "post-audit locked cross-cycle re-evaluation",
        "development_cycle": "F TRAIN only",
        "cross_cycle_evaluation_cycles": ["E", "G"],
        "method": {
            "base_prediction": "state-aware ridge classification",
            "reliability_model": "ExtraTreesClassifier",
            "reliability_model_parameters": {
                "n_estimators": 500,
                "min_samples_leaf": 5,
                "max_features": 0.7,
                "class_weight": "balanced",
                "random_state": SEED,
            },
            "ridge_alpha": alpha,
            "participant_separation": "nested participant-hash cross-fitting on Cycle F TRAIN",
            "coverage_rule": "maximize Cycle F nested-OOF kappa subject to coverage >= 0.75; freeze before cross-cycle scoring",
            "cycle_E_G_labels_enter_training_or_threshold_fitting": False,
        },
        "development": development,
        "cross_cycle": cycle_results,
        "endpoint_comparison": endpoint_comparison,
        "overall_success_rule": {
            "classification": "physics-informed reliability kappa exceeds same-capacity non-physics reliability and ridge-margin selection at identical coverage in both cycles",
            "endpoint": "Cycle-F-trained physics-plus-ridge hybrid FVC MAE is below state-aware ridge in both cycles",
        },
    }
    classification_pass = all(
        cycle_results[cycle]["paired_participant_bootstrap_differences"][comparator]["kappa"]["95_ci"][0] > 0
        for cycle in ("E", "G")
        for comparator in (
            "physics_reliability_minus_ridge_reliability",
            "physics_reliability_minus_ridge_margin",
        )
    )
    endpoint_pass = all(
        endpoint_comparison[cycle]["paired_hybrid_minus_ridge_mae_ml"]["95_ci"][1] < 0
        for cycle in ("E", "G")
    )
    payload["overall_success_rule"]["classification_pass"] = classification_pass
    payload["overall_success_rule"]["endpoint_pass"] = endpoint_pass
    payload["overall_success_rule"]["overall_pass"] = classification_pass and endpoint_pass

    output = ROOT / "results/physics_informed_selector_cross_cycle.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    pd.concat(participant_tables, ignore_index=True).to_csv(
        ROOT / "results/physics_informed_selector_participants.csv", index=False
    )

    lines = [
        "# Physics-Informed Selective Reliability Evaluation",
        "",
        (
            "The reliability rule was developed with nested participant-separated cross-fitting on Cycle F TRAIN only. "
            "The operating coverage and all model parameters were then fixed for the Cycle E and G comparison."
        ),
        "",
        "| Cycle | Method | Coverage | Sensitivity | Specificity | Kappa | FN | FP |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cycle in ("E", "G"):
        for name, metric in cycle_results[cycle]["metrics"].items():
            lines.append(
                f"| {cycle} | {name} | {metric['coverage']:.3f} | {metric['sensitivity']:.3f} | "
                f"{metric['specificity']:.3f} | {metric['kappa']:.3f} | {metric['fn']} | {metric['fp']} |"
            )
    lines.extend(["", f"Overall audited success rule: **{'PASS' if payload['overall_success_rule']['overall_pass'] else 'FAIL'}**.", ""])
    (ROOT / "results/physics_informed_selector_cross_cycle.md").write_text("\n".join(lines))
    print(json.dumps(payload["overall_success_rule"], indent=2))
    for cycle in ("E", "G"):
        print(cycle, json.dumps(cycle_results[cycle]["metrics"], indent=2))


if __name__ == "__main__":
    main()
