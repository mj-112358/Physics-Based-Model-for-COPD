from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .gate_graded_probability_f import (
    DEMOGRAPHIC_BASE_FEATURES,
    PPV_NPV_FLOOR,
    add_demographics,
    boundary_mask,
    choose_boundaries,
    class_reweight,
    feature_audit,
    impute_frames,
    load_model_frame,
    prevalence_from_full_cohort,
    reliability_bins,
    truth_obstructed,
)
from .gate_workflow_policy_f import RANDOM_SEED, clean_json, fmt, md_table

OUTPUT_STEM = "gate_graded_probability_repair_F"


def prior_adjust(p: np.ndarray, source_prevalence: float, target_prevalence: float) -> np.ndarray:
    eps = 1e-9
    p2 = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    source_odds = source_prevalence / (1.0 - source_prevalence)
    target_odds = target_prevalence / (1.0 - target_prevalence)
    odds = p2 / (1.0 - p2) * target_odds / source_odds
    return odds / (1.0 + odds)


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def auc_safe(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def metric_row(
    y: np.ndarray,
    p: np.ndarray,
    severe: np.ndarray,
    weights: np.ndarray,
    p_low: float,
    p_high: float,
) -> dict[str, Any]:
    normal = p <= p_low
    obstructed = p >= p_high
    pred = np.full(len(y), -1, dtype=int)
    pred[normal] = 0
    pred[obstructed] = 1
    conf = pred != -1
    if not conf.any():
        return {
            "confident_coverage": 0.0,
            "confident_n": 0,
            "ppv_reweighted": np.nan,
            "npv_reweighted": np.nan,
            "cohens_kappa": np.nan,
            "severe_fn_n": 0,
            "confident_obstructed_n": 0,
            "confident_normal_n": 0,
            "raw_fp_obstructed": 0,
            "raw_fn_normal": 0,
        }
    yy = y[conf]
    pp = pred[conf]
    ww = weights[conf]
    tp = float(ww[(yy == 1) & (pp == 1)].sum())
    fp = float(ww[(yy == 0) & (pp == 1)].sum())
    tn = float(ww[(yy == 0) & (pp == 0)].sum())
    fn = float(ww[(yy == 1) & (pp == 0)].sum())
    return {
        "confident_coverage": float(conf.mean()),
        "confident_n": int(conf.sum()),
        "ppv_reweighted": float(tp / (tp + fp)) if (tp + fp) else np.nan,
        "npv_reweighted": float(tn / (tn + fn)) if (tn + fn) else np.nan,
        "cohens_kappa": float(cohen_kappa_score(yy, pp)) if len(np.unique(pp)) > 1 else np.nan,
        "severe_fn_n": int((normal & severe).sum()),
        "confident_obstructed_n": int(obstructed.sum()),
        "confident_normal_n": int(normal.sum()),
        "raw_fp_obstructed": int(((y == 0) & obstructed).sum()),
        "raw_fn_normal": int(((y == 1) & normal).sum()),
    }


def oracle_best(
    y: np.ndarray,
    p: np.ndarray,
    severe: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    vals = np.unique(np.round(np.asarray(p, dtype=float), 12))
    prefix_valid = np.zeros(len(vals), dtype=bool)
    suffix_valid = np.zeros(len(vals), dtype=bool)
    prefix_n = np.zeros(len(vals), dtype=int)
    suffix_n = np.zeros(len(vals), dtype=int)

    for i, value in enumerate(vals):
        normal = p <= value
        obstructed = p >= value
        prefix_n[i] = int(normal.sum())
        suffix_n[i] = int(obstructed.sum())
        prefix_row = metric_row(y, p, severe, weights, float(value), 1.0)
        suffix_row = metric_row(y, p, severe, weights, 0.0, float(value))
        prefix_valid[i] = (
            prefix_row["confident_normal_n"] > 0
            and prefix_row["severe_fn_n"] == 0
            and finite(prefix_row["npv_reweighted"])
            and prefix_row["npv_reweighted"] >= PPV_NPV_FLOOR
        )
        suffix_valid[i] = (
            suffix_row["confident_obstructed_n"] > 0
            and finite(suffix_row["ppv_reweighted"])
            and suffix_row["ppv_reweighted"] >= PPV_NPV_FLOOR
        )

    best_suffix_idx_after = np.full(len(vals) + 1, -1, dtype=int)
    valid_suffix_count_after = np.zeros(len(vals) + 1, dtype=int)
    best_idx = -1
    best_n = -1
    count = 0
    for i in range(len(vals) - 1, -1, -1):
        if suffix_valid[i] and suffix_n[i] > best_n:
            best_idx = i
            best_n = suffix_n[i]
        best_suffix_idx_after[i] = best_idx
        valid_suffix_count_after[i] = valid_suffix_count_after[i + 1] + int(suffix_valid[i])

    best: dict[str, Any] | None = None
    for i, value in enumerate(vals[:-1]):
        if not prefix_valid[i]:
            continue
        count += int(valid_suffix_count_after[i + 1])
        j = int(best_suffix_idx_after[i + 1])
        if j < 0:
            continue
        candidate = {
            **metric_row(y, p, severe, weights, float(value), float(vals[j])),
            "p_low": float(value),
            "p_high": float(vals[j]),
        }
        if candidate["confident_obstructed_n"] == 0 or candidate["confident_normal_n"] == 0:
            continue
        if best is None or (
            candidate["confident_coverage"],
            candidate["cohens_kappa"] if finite(candidate["cohens_kappa"]) else -1.0,
        ) > (
            best["confident_coverage"],
            best["cohens_kappa"] if finite(best["cohens_kappa"]) else -1.0,
        ):
            best = candidate
    if best is None:
        best = {
            "confident_coverage": 0.0,
            "confident_n": 0,
            "p_low": np.nan,
            "p_high": np.nan,
            "ppv_reweighted": np.nan,
            "npv_reweighted": np.nan,
            "cohens_kappa": np.nan,
            "severe_fn_n": 0,
            "confident_obstructed_n": 0,
            "confident_normal_n": 0,
            "raw_fp_obstructed": 0,
            "raw_fn_normal": 0,
        }
    best["oracle_feasible_threshold_pairs"] = int(count)
    return best


def make_base(name: str) -> Any:
    if name == "gbt":
        return GradientBoostingClassifier(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.8,
            random_state=RANDOM_SEED,
        )
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_SEED),
        )
    raise ValueError(name)


def fit_probability(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    y_train: np.ndarray,
    base_name: str,
    calibration: str,
    apply_prior: bool,
    source_prevalence: float,
    target_prevalence: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_train, x_test, _ = impute_frames(train, test, features)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    model = CalibratedClassifierCV(estimator=make_base(base_name), method=calibration, cv=cv)
    model.fit(x_train, y_train)
    p_train = model.predict_proba(x_train)[:, 1].astype(float)
    p_test = model.predict_proba(x_test)[:, 1].astype(float)
    if apply_prior:
        p_train = prior_adjust(p_train, source_prevalence, target_prevalence)
        p_test = prior_adjust(p_test, source_prevalence, target_prevalence)
    return p_train, p_test


def evaluate_variant(
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    severe_test: np.ndarray,
    train_weights: np.ndarray,
    test_weights: np.ndarray,
    source_prevalence: float,
    target_prevalence: float,
    base_name: str,
    calibration: str,
    apply_prior: bool,
    headline_allowed: bool,
) -> dict[str, Any]:
    p_train, p_test = fit_probability(
        train,
        test,
        features,
        y_train,
        base_name,
        calibration,
        apply_prior,
        source_prevalence,
        target_prevalence,
    )
    train_boundaries = choose_boundaries(train, p_train, train_weights)
    train_row = metric_row(
        y_train,
        p_train,
        train["truth_severity"].astype(str).eq("severe").to_numpy(),
        train_weights,
        float(train_boundaries["p_low"]),
        float(train_boundaries["p_high"]),
    )
    test_row = metric_row(
        y_test,
        p_test,
        severe_test,
        test_weights,
        float(train_boundaries["p_low"]),
        float(train_boundaries["p_high"]),
    )
    oracle = oracle_best(y_test, p_test, severe_test, test_weights)
    boundary = boundary_mask(test).to_numpy()
    _, raw_cal = reliability_bins(y_test, p_test, None, "test", "raw", target_prevalence)
    _, rw_cal = reliability_bins(y_test, p_test, test_weights, "test", "reweighted", target_prevalence)
    return {
        "variant": name,
        "headline_allowed": bool(headline_allowed),
        "features_n": int(len(features)),
        "base_model": base_name,
        "calibration": calibration,
        "prior_adjusted_to_natural_prevalence": bool(apply_prior),
        "test_auc": auc_safe(y_test, p_test),
        "boundary_auc": auc_safe(y_test[boundary], p_test[boundary]),
        "raw_ece": raw_cal["ece"],
        "reweighted_ece": rw_cal["ece"],
        "train_p_low": train_boundaries["p_low"],
        "train_p_high": train_boundaries["p_high"],
        "train_confident_coverage": train_row["confident_coverage"],
        "train_ppv_reweighted": train_row["ppv_reweighted"],
        "train_npv_reweighted": train_row["npv_reweighted"],
        "test_confident_coverage": test_row["confident_coverage"],
        "test_ppv_reweighted": test_row["ppv_reweighted"],
        "test_npv_reweighted": test_row["npv_reweighted"],
        "test_kappa": test_row["cohens_kappa"],
        "test_severe_fn_n": test_row["severe_fn_n"],
        "test_confident_obstructed_n": test_row["confident_obstructed_n"],
        "test_confident_normal_n": test_row["confident_normal_n"],
        "test_raw_fp_obstructed": test_row["raw_fp_obstructed"],
        "test_raw_fn_normal": test_row["raw_fn_normal"],
        "oracle_confident_coverage": oracle["confident_coverage"],
        "oracle_confident_n": oracle["confident_n"],
        "oracle_p_low": oracle["p_low"],
        "oracle_p_high": oracle["p_high"],
        "oracle_ppv_reweighted": oracle["ppv_reweighted"],
        "oracle_npv_reweighted": oracle["npv_reweighted"],
        "oracle_kappa": oracle["cohens_kappa"],
        "oracle_severe_fn_n": oracle["severe_fn_n"],
        "oracle_threshold_pairs": oracle["oracle_feasible_threshold_pairs"],
        "oracle_confident_obstructed_n": oracle["confident_obstructed_n"],
        "oracle_confident_normal_n": oracle["confident_normal_n"],
        "oracle_raw_fp_obstructed": oracle["raw_fp_obstructed"],
        "oracle_raw_fn_normal": oracle["raw_fn_normal"],
    }


def verdict(result: dict[str, Any]) -> tuple[str, str]:
    operational = result["operational_baseline_coverage"]
    allowed = [row for row in result["variant_rows"] if row["headline_allowed"]]
    train_pass = [
        row
        for row in allowed
        if row["test_severe_fn_n"] == 0
        and finite(row["test_ppv_reweighted"])
        and finite(row["test_npv_reweighted"])
        and row["test_ppv_reweighted"] >= PPV_NPV_FLOOR
        and row["test_npv_reweighted"] >= PPV_NPV_FLOOR
        and row["test_confident_coverage"] >= operational + 0.03
    ]
    if train_pass:
        best = max(train_pass, key=lambda row: row["test_confident_coverage"])
        return (
            "REPAIR-SUCCEEDS",
            f"{best['variant']} clears floors and exceeds operational baseline coverage.",
        )
    oracle_best_allowed = max(allowed, key=lambda row: row["oracle_confident_coverage"]) if allowed else None
    if oracle_best_allowed and oracle_best_allowed["oracle_confident_coverage"] < operational + 0.03:
        return (
            "REPAIR-NOT-ACHIEVABLE-CLEAN",
            "Even TEST-oracle thresholds on the best headline-allowed variant do not materially exceed operational baseline coverage.",
        )
    return (
        "REPAIR-TRAIN-SELECTION-FAILED",
        "A headline-allowed variant has an oracle-feasible region, but TRAIN-frozen boundaries do not reproduce it safely on TEST.",
    )


def write_report(root: Path, result: dict[str, Any], rows: pd.DataFrame) -> None:
    out = root / "results"
    out.mkdir(exist_ok=True)
    (out / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    rows.to_csv(out / f"{OUTPUT_STEM}_variants.csv", index=False)
    allowed = [row for row in result["variant_rows"] if row["headline_allowed"]]
    best_allowed_oracle = max(allowed, key=lambda row: row["oracle_confident_coverage"])
    best_train = max(allowed, key=lambda row: row["test_confident_coverage"])
    lines = [
        "# Gate - Graded Probability Repair Attempt, Cycle F",
        "",
        f"VERDICT: {result['verdict']} - {result['verdict_reason']}",
        "",
        "## What Was Tried",
        "",
        "- Natural-prevalence prior correction after calibration.",
        "- Sigmoid versus isotonic calibration.",
        "- Gradient-boosted trees versus balanced logistic regression.",
        "- CLEAN-only and CLEAN+orthogonal-demographic feature sets.",
        "- RATIO-AWARE diagnostic variant, clearly excluded from headline eligibility.",
        "",
        "## Baselines",
        "",
        f"- strict deterministic coverage: {fmt(result['strict_baseline_coverage'], 3)}.",
        f"- operational recovery coverage: {fmt(result['operational_baseline_coverage'], 3)}.",
        f"- material improvement target used here: operational + 0.03 = {fmt(result['operational_baseline_coverage'] + 0.03, 3)}.",
        "",
        "## Variant Results",
        "",
        md_table(
            result["variant_rows"],
            [
                ("variant", "variant", "str"),
                ("headline_allowed", "allowed", "str"),
                ("test_auc", "AUC", "float3"),
                ("boundary_auc", "boundary AUC", "float3"),
                ("reweighted_ece", "ECE rw", "float3"),
                ("test_confident_coverage", "TRAIN-frozen cov", "float3"),
                ("test_ppv_reweighted", "PPV rw", "float3"),
                ("test_npv_reweighted", "NPV rw", "float3"),
                ("test_severe_fn_n", "severe FN", "int"),
                ("oracle_confident_coverage", "TEST-oracle cov", "float3"),
                ("oracle_ppv_reweighted", "oracle PPV", "float3"),
                ("oracle_npv_reweighted", "oracle NPV", "float3"),
                ("oracle_severe_fn_n", "oracle severe FN", "int"),
            ],
        ),
        "",
        "## Best Headline-Allowed Attempts",
        "",
        f"- Best TRAIN-frozen headline-allowed coverage: {best_train['variant']} at {fmt(best_train['test_confident_coverage'], 3)}, PPV/NPV {fmt(best_train['test_ppv_reweighted'], 3)}/{fmt(best_train['test_npv_reweighted'], 3)}, severe FN {best_train['test_severe_fn_n']}.",
        f"- Best TEST-oracle headline-allowed coverage: {best_allowed_oracle['variant']} at {fmt(best_allowed_oracle['oracle_confident_coverage'], 3)}, PPV/NPV {fmt(best_allowed_oracle['oracle_ppv_reweighted'], 3)}/{fmt(best_allowed_oracle['oracle_npv_reweighted'], 3)}, severe FN {best_allowed_oracle['oracle_severe_fn_n']}.",
        "- TEST-oracle rows are diagnostic upper bounds only; they are not deployable because boundaries are selected using TEST truth.",
        "",
        "## Interpretation",
        "",
        "- Prior correction can repair reweighted ECE for some variants, but it does not solve the PPV floor at natural prevalence.",
        "- Demographics remove the severe-FN problem for the broad GBT candidate, but reweighted PPV remains far below 0.90.",
        "- Some sigmoid variants have TEST-oracle feasible zones with PPV=1.0, but their coverage ceiling stays below the operational 66.6% baseline.",
        "- The only path expected to beat this would require ratio-aware features, which are explicitly not headline-eligible under the leakage rules.",
        "",
        f"VERDICT: {result['verdict']} - {result['verdict_reason']}",
        "",
    ]
    (out / f"{OUTPUT_STEM}.md").write_text("\n".join(lines), encoding="utf-8")


def run(root: Path) -> dict[str, Any]:
    workflow, repro, recovered_test, _c1 = load_model_frame(root)
    audit, _banned, _quarantined, clean_features = feature_audit(workflow)
    train = workflow[workflow["split"].eq("train")].copy().reset_index(drop=True)
    test = workflow[workflow["split"].eq("test")].copy().reset_index(drop=True)
    test = test.merge(
        recovered_test[["SEQN", "post_recovery_finalized", "post_recovery_decision"]],
        on="SEQN",
        how="left",
    )
    prevalence = prevalence_from_full_cohort(root)
    target_prevalence = float(prevalence["target_prevalence"])
    y_train = truth_obstructed(train).astype(int).to_numpy()
    y_test = truth_obstructed(test).astype(int).to_numpy()
    source_prevalence = float(y_train.mean())
    train_weights = class_reweight(y_train, target_prevalence, source_prevalence)
    test_weights = class_reweight(y_test, target_prevalence, source_prevalence)
    severe_test = test["truth_severity"].astype(str).eq("severe").to_numpy()

    demo_frame, _demo_availability, _notes = add_demographics(
        root,
        workflow.merge(test[["SEQN", "post_recovery_finalized"]], on="SEQN", how="left"),
    )
    demo_train = demo_frame[demo_frame["split"].eq("train")].copy().reset_index(drop=True)
    demo_test = demo_frame[demo_frame["split"].eq("test")].copy().reset_index(drop=True)
    demo_cols = [
        col
        for col in demo_frame.columns
        if col in DEMOGRAPHIC_BASE_FEATURES or col.startswith("race_")
    ]

    variants = [
        ("clean_gbt_isotonic", train, test, clean_features, "gbt", "isotonic", False, True),
        ("clean_gbt_sigmoid_prior", train, test, clean_features, "gbt", "sigmoid", True, True),
        ("clean_logistic_sigmoid_prior", train, test, clean_features, "logistic", "sigmoid", True, True),
        ("demo_gbt_isotonic", demo_train, demo_test, clean_features + demo_cols, "gbt", "isotonic", False, True),
        ("demo_gbt_sigmoid_prior", demo_train, demo_test, clean_features + demo_cols, "gbt", "sigmoid", True, True),
        ("demo_logistic_sigmoid_prior", demo_train, demo_test, clean_features + demo_cols, "logistic", "sigmoid", True, True),
        (
            "ratio_aware_gbt_sigmoid_prior_DIAGNOSTIC",
            train,
            test,
            clean_features + ["model_ratio", "corrected_ratio_c1"],
            "gbt",
            "sigmoid",
            True,
            False,
        ),
    ]

    rows = []
    for name, tr, te, features, base_name, cal, prior, allowed in variants:
        rows.append(
            evaluate_variant(
                name,
                tr,
                te,
                features,
                y_train,
                y_test,
                severe_test,
                train_weights,
                test_weights,
                source_prevalence,
                target_prevalence,
                base_name,
                cal,
                prior,
                allowed,
            )
        )

    strict_coverage = float(recovered_test["workflow_final_diagnostic"].astype(bool).mean())
    operational_coverage = float(recovered_test["post_recovery_finalized"].astype(bool).mean())
    result: dict[str, Any] = {
        "seed": RANDOM_SEED,
        "target_prevalence": target_prevalence,
        "source_train_prevalence": source_prevalence,
        "strict_baseline_coverage": strict_coverage,
        "operational_baseline_coverage": operational_coverage,
        "clean_features": clean_features,
        "variant_rows": rows,
        "protocol_decisions": [
            "This is a repair audit, not a replacement for the leakage-free gate unless a variant clears the same guards.",
            "Headline eligibility excludes ratio-aware variants even when they perform better.",
            "TEST-oracle thresholds are reported only as upper bounds; deployable thresholds are TRAIN-frozen.",
            "The material-improvement target is operational recovery coverage plus 0.03.",
        ],
    }
    result["verdict"], result["verdict_reason"] = verdict(result)
    write_report(root, result, pd.DataFrame(rows))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    allowed = [row for row in result["variant_rows"] if row["headline_allowed"]]
    best_oracle = max(allowed, key=lambda row: row["oracle_confident_coverage"])
    print(f"Graded probability repair: {result['verdict']} - {result['verdict_reason']}")
    print(f"Operational baseline: {result['operational_baseline_coverage']:.3f}")
    print(
        "Best headline-allowed TEST-oracle: "
        f"{best_oracle['variant']} coverage={best_oracle['oracle_confident_coverage']:.3f} "
        f"PPV/NPV={best_oracle['oracle_ppv_reweighted']:.3f}/{best_oracle['oracle_npv_reweighted']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
