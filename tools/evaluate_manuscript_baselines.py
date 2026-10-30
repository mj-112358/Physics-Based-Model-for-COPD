from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SEED = 20261029
BOOTSTRAP_REPLICATES = 2000
FIXED_RATIO_THRESHOLD = 0.70
FEV1_FEV6_THRESHOLD = 0.73
RIDGE_ALPHAS = np.logspace(-4, 4, 41)
RIDGE_FEATURES = (
    "parsed_fev1_l",
    "parsed_fvc_l",
    "duration_s",
    "age_years",
    "height_cm",
    "sex",
)
FINAL_OBSTRUCTED_ACTIONS = {
    "ACCEPT_OBSTRUCTED",
    "TRUNCATED_RECOVERED",
    "RECOVERED_OBSTRUCTED",
}


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def kappa_from_counts(tp: int, tn: int, fp: int, fn: int) -> float:
    n = tp + tn + fp + fn
    if n == 0:
        return np.nan
    observed = (tp + tn) / n
    truth_positive = (tp + fn) / n
    predicted_positive = (tp + fp) / n
    expected = truth_positive * predicted_positive + (1 - truth_positive) * (1 - predicted_positive)
    return (observed - expected) / (1 - expected) if expected < 1 else np.nan


def score_binary(truth: np.ndarray, prediction: np.ndarray, selected: np.ndarray | None = None) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    selected = np.ones(len(truth), dtype=bool) if selected is None else np.asarray(selected, dtype=bool)
    y = truth[selected]
    p = prediction[selected]
    tp = int(np.sum(y & p))
    tn = int(np.sum(~y & ~p))
    fp = int(np.sum(~y & p))
    fn = int(np.sum(y & ~p))
    return {
        "n": int(len(y)),
        "coverage": float(np.mean(selected)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "kappa": float(kappa_from_counts(tp, tn, fp, fn)),
    }


def bootstrap_intervals(
    truth: np.ndarray,
    prediction: np.ndarray,
    selected: np.ndarray,
    *,
    seed: int,
) -> dict[str, list[float]]:
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    selected = np.asarray(selected, dtype=bool)
    rng = np.random.default_rng(seed)
    n = len(truth)
    draws = {name: np.empty(BOOTSTRAP_REPLICATES) for name in ("coverage", "sensitivity", "specificity", "kappa")}
    for b in range(BOOTSTRAP_REPLICATES):
        take = rng.integers(0, n, size=n)
        metric = score_binary(truth[take], prediction[take], selected[take])
        for name in draws:
            draws[name][b] = metric[name]
    return {
        name: [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]
        for name, values in draws.items()
    }


class RidgeRegressor:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.medians_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.medians_ = np.nanmedian(x, axis=0)
        x = np.where(np.isfinite(x), x, self.medians_)
        self.mean_ = np.mean(x, axis=0)
        self.scale_ = np.std(x, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        z = (x - self.mean_) / self.scale_
        y_mean = float(np.mean(y))
        penalty = self.alpha * np.eye(z.shape[1])
        self.coef_ = np.linalg.solve(z.T @ z + penalty, z.T @ (y - y_mean))
        self.intercept_ = y_mean
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if any(value is None for value in (self.mean_, self.scale_, self.coef_, self.intercept_, self.medians_)):
            raise RuntimeError("RidgeRegressor must be fitted before prediction")
        x = np.asarray(x, dtype=float)
        x = np.where(np.isfinite(x), x, self.medians_)
        z = (x - self.mean_) / self.scale_
        return self.intercept_ + z @ self.coef_


def participant_folds(seqn: np.ndarray, k: int = 5) -> np.ndarray:
    seqn = np.asarray(seqn, dtype=np.int64)
    return np.abs(seqn * 2654435761 + SEED) % k


def choose_ridge_alpha(train: pd.DataFrame) -> tuple[float, list[dict[str, float]]]:
    x = train[list(RIDGE_FEATURES)].to_numpy(dtype=float)
    y = train["fvc_mL"].to_numpy(dtype=float) / 1000.0
    folds = participant_folds(train["SEQN"].to_numpy())
    rows = []
    for alpha in RIDGE_ALPHAS:
        errors = []
        for fold in range(5):
            fit_mask = folds != fold
            val_mask = folds == fold
            model = RidgeRegressor(alpha).fit(x[fit_mask], y[fit_mask])
            errors.extend(np.abs(model.predict(x[val_mask]) - y[val_mask]))
        rows.append({"alpha": float(alpha), "cv_mae_l": float(np.mean(errors))})
    best = min(rows, key=lambda row: (row["cv_mae_l"], row["alpha"]))
    return float(best["alpha"]), rows


def top_margin_mask(ratio: np.ndarray, n_selected: int) -> np.ndarray:
    ratio = np.asarray(ratio, dtype=float)
    order = np.argsort(-np.abs(ratio - FIXED_RATIO_THRESHOLD), kind="stable")
    selected = np.zeros(len(ratio), dtype=bool)
    selected[order[:n_selected]] = True
    return selected


def choose_fixed_specificity_threshold(train_ratio: np.ndarray, train_truth: np.ndarray, target_specificity: float) -> dict[str, Any]:
    candidates = []
    for threshold in np.unique(train_ratio):
        metric = score_binary(train_truth, train_ratio < threshold)
        if metric["specificity"] >= target_specificity:
            candidates.append((metric["sensitivity"], -abs(metric["specificity"] - target_specificity), -abs(threshold - FIXED_RATIO_THRESHOLD), threshold, metric))
    if not candidates:
        raise RuntimeError("No threshold achieved the requested training specificity")
    _, _, _, threshold, metric = max(candidates)
    return {"threshold": float(threshold), "training_metrics": metric}


def add_intervals(name: str, result: dict[str, Any], truth: np.ndarray, prediction: np.ndarray, selected: np.ndarray, seed_offset: int) -> None:
    result["bootstrap_95_ci"] = bootstrap_intervals(truth, prediction, selected, seed=SEED + seed_offset)
    result["name"] = name


def run(root: Path) -> dict[str, Any]:
    results = root / "results"
    policy = pd.read_csv(results / "gate_policy_calibrated_flag_F_rows.csv", low_memory=False)
    final_rows = pd.read_csv(results / "gate_r3_ats_besttest_F_rows.csv", low_memory=False)
    summary = pq.read_table(results / "merged_summary.parquet").to_pandas()
    summary_f = summary[summary["cycle"].eq("F")].drop_duplicates("SEQN")
    covariates = summary_f[["SEQN", "age_years", "height_cm", "sex", "fev6_mL"]]
    data = policy.merge(covariates, on="SEQN", how="left", validate="one_to_one")
    if data[list(RIDGE_FEATURES)].isna().all(axis=0).any():
        raise RuntimeError("At least one ridge feature is entirely missing after the Cycle F join")

    train = data[data["split"].eq("train")].copy()
    test = data[data["split"].eq("test")].copy()
    if len(test) != len(final_rows) or set(test["SEQN"]) != set(final_rows["SEQN"]):
        raise RuntimeError("The policy TEST split and final-policy rows do not identify the same participants")
    test = test.sort_values("SEQN").reset_index(drop=True)
    final_rows = final_rows.sort_values("SEQN").reset_index(drop=True)

    train_truth = train["fev1_fvc_ratio"].to_numpy(dtype=float) < FIXED_RATIO_THRESHOLD
    test_truth = test["fev1_fvc_ratio"].to_numpy(dtype=float) < FIXED_RATIO_THRESHOLD
    final_truth = final_rows["truth_obstructed_fixed"].astype(bool).to_numpy()
    if not np.array_equal(test_truth, final_truth):
        raise RuntimeError("Fixed-ratio truth differs between baseline and final-policy artifacts")

    final_selected = final_rows["post_ats_finalized"].astype(bool).to_numpy()
    final_prediction = final_rows["post_ats_decision"].isin(FINAL_OBSTRUCTED_ACTIONS).to_numpy()
    final_metrics = score_binary(test_truth, final_prediction, final_selected)
    final_n = final_metrics["n"]

    direct_ratio = test["parsed_ratio"].to_numpy(dtype=float)
    direct_prediction = direct_ratio < FIXED_RATIO_THRESHOLD
    qc_selected = ~test["truncated_ats"].astype(bool).to_numpy()
    qc_metrics = score_binary(test_truth, direct_prediction, qc_selected)
    add_intervals("quality_control_first", qc_metrics, test_truth, direct_prediction, qc_selected, 1)

    direct_margin_selected = top_margin_mask(direct_ratio, final_n)
    direct_margin_metrics = score_binary(test_truth, direct_prediction, direct_margin_selected)
    add_intervals("direct_ratio_matched_coverage", direct_margin_metrics, test_truth, direct_prediction, direct_margin_selected, 2)

    fev6_valid = np.isfinite(test["fev6_mL"].to_numpy(dtype=float)) & (test["fev6_mL"].to_numpy(dtype=float) > 0)
    fev6_ratio = test["fev1_mL"].to_numpy(dtype=float) / test["fev6_mL"].to_numpy(dtype=float)
    fev6_prediction = fev6_ratio < FEV1_FEV6_THRESHOLD
    fev6_metrics = score_binary(test_truth, fev6_prediction, fev6_valid)
    add_intervals("fev1_fev6", fev6_metrics, test_truth, fev6_prediction, fev6_valid, 3)

    best_alpha, alpha_search = choose_ridge_alpha(train)
    ridge = RidgeRegressor(best_alpha).fit(
        train[list(RIDGE_FEATURES)].to_numpy(dtype=float),
        train["fvc_mL"].to_numpy(dtype=float) / 1000.0,
    )
    train_predicted_fvc = ridge.predict(train[list(RIDGE_FEATURES)].to_numpy(dtype=float))
    test_predicted_fvc = ridge.predict(test[list(RIDGE_FEATURES)].to_numpy(dtype=float))
    train_ridge_ratio = train["parsed_fev1_l"].to_numpy(dtype=float) / train_predicted_fvc
    test_ridge_ratio = test["parsed_fev1_l"].to_numpy(dtype=float) / test_predicted_fvc
    ridge_prediction = test_ridge_ratio < FIXED_RATIO_THRESHOLD

    ridge_all_metrics = score_binary(test_truth, ridge_prediction)
    ridge_all_metrics["fvc_mae_ml"] = float(np.mean(np.abs(test_predicted_fvc * 1000.0 - test["fvc_mL"].to_numpy(dtype=float))))
    ridge_all_metrics["fvc_median_signed_error_ml"] = float(np.median(test_predicted_fvc * 1000.0 - test["fvc_mL"].to_numpy(dtype=float)))
    add_intervals("ridge_all", ridge_all_metrics, test_truth, ridge_prediction, np.ones(len(test), dtype=bool), 4)

    ridge_margin_selected = top_margin_mask(test_ridge_ratio, final_n)
    ridge_margin_metrics = score_binary(test_truth, ridge_prediction, ridge_margin_selected)
    add_intervals("ridge_matched_coverage", ridge_margin_metrics, test_truth, ridge_prediction, ridge_margin_selected, 5)

    threshold_selection = choose_fixed_specificity_threshold(train_ridge_ratio, train_truth, final_metrics["specificity"])
    fixed_specificity_prediction = test_ridge_ratio < threshold_selection["threshold"]
    ridge_fixed_specificity_metrics = score_binary(test_truth, fixed_specificity_prediction)
    add_intervals(
        "ridge_fixed_specificity",
        ridge_fixed_specificity_metrics,
        test_truth,
        fixed_specificity_prediction,
        np.ones(len(test), dtype=bool),
        6,
    )

    add_intervals("final_selective_policy", final_metrics, test_truth, final_prediction, final_selected, 7)

    prediction_rows = pd.DataFrame(
        {
            "SEQN": test["SEQN"],
            "truth_obstructed_fixed": test_truth,
            "truncated_ats": test["truncated_ats"].astype(bool),
            "direct_ratio": direct_ratio,
            "qc_selected": qc_selected,
            "direct_margin_selected": direct_margin_selected,
            "fev1_fev6_ratio": fev6_ratio,
            "ridge_predicted_fvc_l": test_predicted_fvc,
            "ridge_ratio": test_ridge_ratio,
            "ridge_margin_selected": ridge_margin_selected,
            "final_selected": final_selected,
            "final_prediction_obstructed": final_prediction,
        }
    )
    prediction_rows.to_csv(results / "manuscript_baseline_predictions.csv", index=False)

    result = {
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "cycle": "F",
        "split": {"train_n": int(len(train)), "test_n": int(len(test))},
        "truth_definition": "released summary FEV1/FVC < 0.70",
        "final_policy_status": "internal policy-development evaluation; Cycle F informed policy comparison",
        "quality_control_first_definition": "defer ATS-truncated manoeuvres; classify remaining manoeuvres from the directly measured ratio",
        "fev1_fev6_definition": f"released summary FEV1/FEV6 < {FEV1_FEV6_THRESHOLD:.2f}",
        "ridge_definition": {
            "target": "released summary FVC in litres",
            "features": list(RIDGE_FEATURES),
            "training_partition": "Cycle F TRAIN only",
            "alpha_selection": "five participant-deterministic folds minimizing FVC mean absolute error",
            "selected_alpha": best_alpha,
            "alpha_search": alpha_search,
        },
        "matched_coverage_n": final_n,
        "metrics": {
            "quality_control_first": qc_metrics,
            "direct_ratio_matched_coverage": direct_margin_metrics,
            "fev1_fev6": fev6_metrics,
            "ridge_all": ridge_all_metrics,
            "ridge_matched_coverage": ridge_margin_metrics,
            "ridge_fixed_specificity": ridge_fixed_specificity_metrics,
            "final_selective_policy": final_metrics,
        },
        "fixed_specificity_threshold": threshold_selection,
        "scope_note": "All comparator metrics use the same Cycle F internal split. No Cycle E or G workflow feature table is present in this checkout, so these results are not independent external validation.",
    }
    return clean_json(result)


def format_metric(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


def write_report(root: Path, result: dict[str, Any]) -> None:
    results = root / "results"
    (results / "manuscript_baseline_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    labels = {
        "quality_control_first": "QC-first direct measurement",
        "direct_ratio_matched_coverage": "Direct-ratio margin at matched coverage",
        "fev1_fev6": "FEV1/FEV6 at 0.73",
        "ridge_all": "Prefix-summary ridge, all cases",
        "ridge_matched_coverage": "Prefix-summary ridge at matched coverage",
        "ridge_fixed_specificity": "Prefix-summary ridge at fixed specificity",
        "final_selective_policy": "Proposed selective policy",
    }
    lines = [
        "# Manuscript Baseline Comparison",
        "",
        "All results use the existing Cycle F TRAIN/TEST split. The final policy remains an internal policy-development result because Cycle F informed policy comparison.",
        "",
        "| Comparator | n | Coverage | Sensitivity | Specificity | Kappa |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in labels.items():
        metric = result["metrics"][key]
        lines.append(
            f"| {label} | {metric['n']} | {format_metric(metric['coverage'])} | "
            f"{format_metric(metric['sensitivity'])} | {format_metric(metric['specificity'])} | {format_metric(metric['kappa'])} |"
        )
    lines.extend(
        [
            "",
            f"The ridge baseline used alpha={result['ridge_definition']['selected_alpha']:.6g}, selected on TRAIN participants only. "
            f"Its all-case FVC mean absolute error was {result['metrics']['ridge_all']['fvc_mae_ml']:.1f} mL.",
            "",
            "The quality-control-first comparator defers every ATS-truncated manoeuvre before classification. "
            "The matched-coverage comparators rank cases by absolute distance from the 0.70 decision boundary and retain exactly the final policy's number of classified cases.",
            "",
            "The committed repository does not contain the Cycle E or G participant-level workflow features or the full natural-pair row set. "
            "An untouched-cycle policy evaluation and a natural-pair FEV1/FEV6 analysis therefore cannot be reproduced from this checkout.",
        ]
    )
    (results / "manuscript_baseline_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate clinical and computational baselines used in the manuscript")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = run(args.root)
    write_report(args.root, result)
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
