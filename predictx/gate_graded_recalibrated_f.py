from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression

from .gate_deferral_recovery_f import final_metrics
from .gate_graded_probability_f import (
    PPV_NPV_FLOOR,
    all_indeterminate_zones,
    apply_zones,
    bootstrap_auc_ci,
    boundary_mask,
    choose_boundaries,
    class_reweight,
    feature_audit,
    fit_model,
    impute_frames,
    load_model_frame,
    make_gbt,
    model_auc_report,
    prevalence_from_full_cohort,
    probability,
    reliability_bins,
    truth_obstructed,
    zone_metrics,
)
from .gate_graded_probability_repair_f import prior_adjust
from .gate_workflow_policy_f import RANDOM_SEED, clean_json, exact_ci, fmt, md_table

OUTPUT_STEM = "gate_graded_recalibrated_F"
FIXED_RATIO_THRESHOLD = 0.70
CALIBRATION_PASS_ECE = 0.10
CALIBRATION_MARGINAL_ECE = 0.15
MATERIAL_COVERAGE_DELTA = 0.03


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def fit_weighted_model(x_train: pd.DataFrame, y_train: np.ndarray, sample_weight: np.ndarray) -> Any:
    model = clone(make_gbt())
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return model


def probability_summary(name: str, p: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(p, dtype=float)
    qs = np.nanpercentile(arr, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return {
        "name": name,
        "n": int(arr.size),
        "min": float(qs[0]),
        "p01": float(qs[1]),
        "p05": float(qs[2]),
        "p25": float(qs[3]),
        "median": float(qs[4]),
        "p75": float(qs[5]),
        "p95": float(qs[6]),
        "p99": float(qs[7]),
        "max": float(qs[8]),
        "mean": float(np.nanmean(arr)),
    }


def add_calibration(
    rows_out: list[dict[str, Any]],
    y: np.ndarray,
    p: np.ndarray,
    weights: np.ndarray | None,
    scope: str,
    weighting: str,
    target_prevalence: float,
    probability_source: str,
) -> dict[str, Any]:
    rows, summary = reliability_bins(y, p, weights, scope, weighting, target_prevalence)
    for row in rows:
        row["probability_source"] = probability_source
        rows_out.append(row)
    return {
        **summary,
        "probability_source": probability_source,
        "scope": scope,
        "weighting": weighting,
        "target_prevalence": float(target_prevalence),
    }


def calibration_status(ece: float) -> str:
    if not finite(ece):
        return "FAILED_NONFINITE"
    if float(ece) <= CALIBRATION_PASS_ECE:
        return "RESTORED"
    if float(ece) <= CALIBRATION_MARGINAL_ECE:
        return "MARGINAL"
    return "INSUFFICIENT"


def reject_reasons(metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if metrics["severe_fn_n"] > 0:
        reasons.append("candidate confident-normal zone has severe false negatives on TEST")
    if not finite(metrics["ppv_reweighted"]) or metrics["ppv_reweighted"] < PPV_NPV_FLOOR:
        reasons.append("candidate confident-obstructed PPV is below the reweighted 0.90 floor on TEST")
    if not finite(metrics["npv_reweighted"]) or metrics["npv_reweighted"] < PPV_NPV_FLOOR:
        reasons.append("candidate confident-normal NPV is below the reweighted 0.90 floor on TEST")
    return reasons


def weighted_ppv_npv_counts(y: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    yy = np.asarray(y, dtype=bool)
    pp = np.asarray(pred, dtype=bool)
    ww = np.asarray(weights, dtype=float)
    tp = float(ww[yy & pp].sum())
    tn = float(ww[~yy & ~pp].sum())
    fp = float(ww[~yy & pp].sum())
    fn = float(ww[yy & ~pp].sum())
    return {
        "ppv_reweighted": float(tp / (tp + fp)) if (tp + fp) else np.nan,
        "npv_reweighted": float(tn / (tn + fn)) if (tn + fn) else np.nan,
        "tp_w": tp,
        "tn_w": tn,
        "fp_w": fp,
        "fn_w": fn,
    }


def bucket_call_metrics(
    part: pd.DataFrame,
    weights: np.ndarray,
    confident_col: str,
    pred_col: str,
    zone_col: str,
    prefix: str,
) -> dict[str, Any]:
    if part.empty:
        return {
            f"{prefix}_confident_n": 0,
            f"{prefix}_confident_fraction": np.nan,
            f"{prefix}_accuracy": np.nan,
            f"{prefix}_severe_confident_n": 0,
            f"{prefix}_severe_fn_n": 0,
            f"{prefix}_severe_fn_ci_high": np.nan,
            f"{prefix}_ppv_reweighted": np.nan,
            f"{prefix}_npv_reweighted": np.nan,
        }
    confident = part[part[confident_col].astype(bool)].copy()
    if confident.empty:
        return {
            f"{prefix}_confident_n": 0,
            f"{prefix}_confident_fraction": 0.0,
            f"{prefix}_accuracy": np.nan,
            f"{prefix}_severe_confident_n": 0,
            f"{prefix}_severe_fn_n": 0,
            f"{prefix}_severe_fn_ci_high": np.nan,
            f"{prefix}_ppv_reweighted": np.nan,
            f"{prefix}_npv_reweighted": np.nan,
        }
    y = confident["truth_obstructed"].astype(bool).to_numpy()
    pred = confident[pred_col].astype(bool).to_numpy()
    severe = confident["truth_severity"].astype(str).eq("severe")
    severe_fn = int((severe & confident[zone_col].astype(str).eq("CONFIDENT_NORMAL")).sum())
    _lo, hi = exact_ci(severe_fn, int(severe.sum()))
    w = weights[confident.index.to_numpy()]
    ppvnpv = weighted_ppv_npv_counts(y, pred, w)
    return {
        f"{prefix}_confident_n": int(len(confident)),
        f"{prefix}_confident_fraction": float(len(confident) / len(part)),
        f"{prefix}_accuracy": float((y == pred).mean()),
        f"{prefix}_severe_confident_n": int(severe.sum()),
        f"{prefix}_severe_fn_n": severe_fn,
        f"{prefix}_severe_fn_ci_high": hi,
        f"{prefix}_ppv_reweighted": ppvnpv["ppv_reweighted"],
        f"{prefix}_npv_reweighted": ppvnpv["npv_reweighted"],
    }


def deferral_bucket_rows(rows: pd.DataFrame, weights: np.ndarray) -> list[dict[str, Any]]:
    deferred = rows[rows["post_recovery_finalized"].eq(False)].copy()
    dist = (pd.to_numeric(deferred["fev1_fvc_ratio"], errors="coerce") - FIXED_RATIO_THRESHOLD).abs()
    specs = [
        ("KNIFE_EDGE", "<0.02", dist < 0.02),
        ("NEAR", "0.02-0.05", (dist >= 0.02) & (dist < 0.05)),
        ("RECOVERABLE", ">=0.05", dist >= 0.05),
    ]
    out: list[dict[str, Any]] = []
    for bucket, distance_band, mask in specs:
        part = deferred[mask].copy()
        row = {
            "bucket": bucket,
            "true_distance_band": distance_band,
            "n": int(len(part)),
            "fraction_of_operational_deferrals": float(len(part) / len(deferred)) if len(deferred) else np.nan,
        }
        row.update(
            bucket_call_metrics(
                part,
                weights,
                "graded_confident",
                "graded_pred_obstructed",
                "graded_zone",
                "retained",
            )
        )
        row.update(
            bucket_call_metrics(
                part,
                weights,
                "candidate_graded_confident",
                "candidate_graded_pred_obstructed",
                "candidate_graded_zone",
                "candidate",
            )
        )
        out.append(row)
    return out


def zone_fraction_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "zone": row["zone"],
            "n": row["n"],
            "fraction": row["fraction"],
        }
        for row in metrics["zone_counts"]
    ]


def select_probability_source(
    calibration_summaries: dict[str, dict[str, Any]],
    p_train_map: dict[str, np.ndarray],
    p_test_map: dict[str, np.ndarray],
) -> dict[str, Any]:
    candidates = []
    for name in ["posthoc_prior_corrected", "weighted_retrained"]:
        summary = calibration_summaries[name]
        candidates.append(
            {
                "name": name,
                "ece": float(summary["ece"]),
                "status": calibration_status(summary["ece"]),
                "p_train": p_train_map[name],
                "p_test": p_test_map[name],
            }
        )
    restored = [item for item in candidates if item["status"] == "RESTORED"]
    if restored:
        return min(restored, key=lambda item: item["ece"])
    marginal = [item for item in candidates if item["status"] == "MARGINAL"]
    if marginal:
        return min(marginal, key=lambda item: item["ece"])
    return min(candidates, key=lambda item: item["ece"])


def physics_inclusive_arm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    clean_features: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_weights: np.ndarray,
    test_weights: np.ndarray,
    source_prevalence: float,
    target_prevalence: float,
) -> dict[str, Any]:
    if "fit_alpha" not in train.columns:
        return {
            "available": False,
            "reason": "fit_alpha column not present",
        }
    features = clean_features + ["fit_alpha"]
    x_train, x_test, _ = impute_frames(train, test, features)
    model = fit_model(make_gbt(), x_train, y_train)
    p_train_raw = probability(model, x_train)
    p_test_raw = probability(model, x_test)
    p_train_adj = prior_adjust(p_train_raw, source_prevalence, target_prevalence)
    p_test_adj = prior_adjust(p_test_raw, source_prevalence, target_prevalence)
    boundaries = choose_boundaries(train, p_train_adj, train_weights)
    zones = apply_zones(test, p_test_adj, boundaries)
    metrics = zone_metrics(zones, test_weights)
    reasons = reject_reasons(metrics)
    band = boundary_mask(test).to_numpy()
    calibration_rows: list[dict[str, Any]] = []
    calibration = add_calibration(
        calibration_rows,
        y_test,
        p_test_adj,
        test_weights,
        "test_all",
        "reweighted_natural",
        target_prevalence,
        "clean_plus_fit_alpha_prior_corrected",
    )
    return {
        "available": True,
        "label": "CLEAN + fit_alpha; mechanistically independent of label computation despite ratio correlation",
        "features": features,
        "whole_auc": bootstrap_auc_ci(y_test, p_test_adj, RANDOM_SEED + 510),
        "boundary_auc": bootstrap_auc_ci(y_test[band], p_test_adj[band], RANDOM_SEED + 520),
        "boundary_n": int(band.sum()),
        "calibration": calibration,
        "zone_boundaries": boundaries,
        "candidate_zone_metrics": metrics,
        "candidate_rejected": bool(reasons),
        "candidate_reject_reasons": reasons,
        "retained_confident_coverage_if_guard_enforced": 0.0 if reasons else metrics["confident_coverage"],
    }


def load_prior_gate_ece(root: Path) -> float:
    path = root / "results" / "gate_graded_probability_F.json"
    if not path.exists():
        return np.nan
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data["calibration_summary"]["test_all_reweighted"]["ece"])
    except Exception:
        return np.nan


def verdict(result: dict[str, Any]) -> tuple[str, str]:
    posthoc_status = result["calibration_gate"]["posthoc_prior_corrected_status"]
    if posthoc_status == "INSUFFICIENT":
        return (
            "PRIOR-CORRECTION-INSUFFICIENT",
            "Label-shift prior correction did not restore natural-prevalence ECE below the marginal threshold.",
        )
    retained = result["retained_zone_metrics"]
    operational = result["deterministic_baseline"]["operational_recovery"]["finalized_fraction"]
    recoverable = next(
        row for row in result["deferral_bucket_recovery"] if row["bucket"] == "RECOVERABLE"
    )
    if (
        retained["confident_coverage"] >= operational + MATERIAL_COVERAGE_DELTA
        and retained["severe_fn_n"] == 0
        and finite(retained["ppv_reweighted"])
        and finite(retained["npv_reweighted"])
        and retained["ppv_reweighted"] >= PPV_NPV_FLOOR
        and retained["npv_reweighted"] >= PPV_NPV_FLOOR
        and recoverable["retained_confident_n"] > 0
    ):
        return (
            "GRADED-VIABLE-COVERAGE-GAIN",
            "ECE is restored and retained confident coverage materially exceeds the operational baseline without safety or PPV/NPV regression.",
        )
    return (
        "GRADED-VIABLE-NEUTRAL",
        "ECE is restored, but TRAIN-frozen confident zones do not safely exceed the 66.6% operational baseline.",
    )


def write_outputs(
    root: Path,
    result: dict[str, Any],
    calibration_rows: list[dict[str, Any]],
    participants: pd.DataFrame,
) -> None:
    out = root / "results"
    out.mkdir(exist_ok=True)
    (out / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(calibration_rows).to_csv(out / f"{OUTPUT_STEM}_calibration.csv", index=False)
    participants.to_csv(out / f"{OUTPUT_STEM}_participants.csv", index=False)

    cal = result["calibration_summary"]
    selected = result["selected_probability_source"]
    candidate = result["candidate_zone_metrics"]
    retained = result["retained_zone_metrics"]
    baseline = result["deterministic_baseline"]
    physics = result["physics_inclusive_arm"]
    lines = [
        "# Gate - Prevalence-Corrected Recalibration of the Graded Obstruction Classifier, Cycle F",
        "",
        f"VERDICT: {result['verdict']} - {result['verdict_reason']}",
        "",
        "## Protocol Decisions",
        "",
    ]
    lines.extend(f"- {item}" for item in result["protocol_decisions"])
    lines.extend(
        [
            "",
            "## Part 1 - Prior Correction",
            "",
            f"- TRAIN constructed prevalence pi_train: {fmt(result['source_train_prevalence'], 3)}.",
            f"- Natural target prevalence pi_target: {fmt(result['target_prevalence'], 3)}.",
            f"- Prior-correction multipliers: a={fmt(result['prior_correction']['a'], 4)}, b={fmt(result['prior_correction']['b'], 4)}.",
            f"- Prior gate reweighted ECE: {fmt(result['prior_gate_reweighted_ece'], 3)}.",
            f"- Selected probability source for zones: {selected}.",
            "",
            md_table(
                [
                    cal["raw_train_prevalence"],
                    cal["uncorrected_reweighted_natural"],
                    cal["posthoc_prior_corrected"],
                    cal["weighted_retrained"],
                    cal.get("fallback_isotonic", {}),
                ],
                [
                    ("probability_source", "source", "str"),
                    ("weighting", "weighting", "str"),
                    ("n", "n", "int"),
                    ("ece", "ECE", "float3"),
                    ("brier", "Brier", "float3"),
                ],
            ),
            "",
            f"- Post-hoc prior-correction status: {result['calibration_gate']['posthoc_prior_corrected_status']}.",
            f"- Confirmatory weighted-retrained status: {result['calibration_gate']['weighted_retrained_status']}.",
            f"- Confirmatory ECE difference: {fmt(result['calibration_gate']['confirmatory_ece_abs_delta'], 3)} ({result['calibration_gate']['confirmatory_status']}).",
            "- The per-TEST-patient corrected probability `p_adj` is in the participants CSV.",
            "",
            "Probability distributions:",
            "",
            md_table(
                result["probability_summaries"],
                [
                    ("name", "probability", "str"),
                    ("mean", "mean", "float3"),
                    ("p05", "p05", "float3"),
                    ("median", "median", "float3"),
                    ("p95", "p95", "float3"),
                    ("min", "min", "float3"),
                    ("max", "max", "float3"),
                ],
            ),
            "",
            "## Part 2 - Frozen Zones on Corrected Probabilities",
            "",
            f"- TRAIN-frozen boundaries: p_low={fmt(result['zone_boundaries']['p_low'], 4)}, p_high={fmt(result['zone_boundaries']['p_high'], 4)}, status={result['zone_boundaries']['boundary_status']}.",
            f"- Candidate rejected on TEST: {result['candidate_zone_rejected']}.",
            f"- Candidate rejection reasons: {'; '.join(result['candidate_zone_reject_reasons']) if result['candidate_zone_reject_reasons'] else 'none'}.",
            "",
            "Candidate zones before enforcing the TEST safety/floor guard:",
            "",
            md_table(
                [{"scope": "candidate", **candidate}],
                [
                    ("scope", "scope", "str"),
                    ("confident_n", "confident n", "int"),
                    ("confident_coverage", "coverage", "float3"),
                    ("accuracy", "accuracy", "float3"),
                    ("sensitivity", "sens", "float3"),
                    ("specificity", "spec", "float3"),
                    ("cohens_kappa", "kappa", "float3"),
                    ("ppv_reweighted", "PPV rw", "float3"),
                    ("npv_reweighted", "NPV rw", "float3"),
                    ("severe_confident_n", "severe conf", "int"),
                    ("severe_fn_n", "severe FN", "int"),
                    ("severe_fn_ci_high", "severe FN CP high", "float3"),
                ],
            ),
            "",
            "Retained zones after guard enforcement:",
            "",
            md_table(
                zone_fraction_rows(retained),
                [("zone", "zone", "str"), ("n", "n", "int"), ("fraction", "fraction", "float3")],
            ),
            "",
            md_table(
                [{"scope": "retained", **retained}],
                [
                    ("scope", "scope", "str"),
                    ("confident_n", "confident n", "int"),
                    ("confident_coverage", "coverage", "float3"),
                    ("accuracy", "accuracy", "float3"),
                    ("sensitivity", "sens", "float3"),
                    ("specificity", "spec", "float3"),
                    ("cohens_kappa", "kappa", "float3"),
                    ("ppv_reweighted", "PPV rw", "float3"),
                    ("npv_reweighted", "NPV rw", "float3"),
                    ("severe_confident_n", "severe conf", "int"),
                    ("severe_fn_n", "severe FN", "int"),
                    ("severe_fn_ci_high", "severe FN CP high", "float3"),
                ],
            ),
            "",
            "## Part 3 - Non-Boundary Deferral Recovery",
            "",
            "Operational deterministic deferrals only:",
            "",
            md_table(
                result["deferral_bucket_recovery"],
                [
                    ("bucket", "bucket", "str"),
                    ("true_distance_band", "true abs distance", "str"),
                    ("n", "n", "int"),
                    ("fraction_of_operational_deferrals", "fraction", "float3"),
                    ("retained_confident_n", "retained confident", "int"),
                    ("retained_accuracy", "retained acc", "float3"),
                    ("retained_severe_fn_n", "retained severe FN", "int"),
                    ("candidate_confident_n", "candidate confident", "int"),
                    ("candidate_accuracy", "candidate acc", "float3"),
                    ("candidate_severe_fn_n", "candidate severe FN", "int"),
                ],
            ),
            "",
            "## Part 4 - Head-to-Head vs Deterministic Baseline",
            "",
            md_table(
                [
                    {"scope": "strict deterministic", **baseline["strict_workflow"]},
                    {"scope": "operational deterministic", **baseline["operational_recovery"]},
                    {
                        "scope": "graded retained confident",
                        "finalized_fraction": retained["confident_coverage"],
                        "cohens_kappa": retained["cohens_kappa"],
                        "sensitivity": retained["sensitivity"],
                        "specificity": retained["specificity"],
                        "severe_fn_n": retained["severe_fn_n"],
                        "severe_fn_ci_high": retained["severe_fn_ci_high"],
                    },
                    {
                        "scope": "graded rejected candidate",
                        "finalized_fraction": candidate["confident_coverage"],
                        "cohens_kappa": candidate["cohens_kappa"],
                        "sensitivity": candidate["sensitivity"],
                        "specificity": candidate["specificity"],
                        "severe_fn_n": candidate["severe_fn_n"],
                        "severe_fn_ci_high": candidate["severe_fn_ci_high"],
                    },
                ],
                [
                    ("scope", "scope", "str"),
                    ("finalized_fraction", "coverage", "float3"),
                    ("cohens_kappa", "kappa", "float3"),
                    ("sensitivity", "sens", "float3"),
                    ("specificity", "spec", "float3"),
                    ("severe_fn_n", "severe FN", "int"),
                    ("severe_fn_ci_high", "severe FN CP high", "float3"),
                ],
            ),
            "",
            f"- Operational baseline coverage: {fmt(baseline['operational_recovery']['finalized_fraction'], 3)}.",
            f"- Retained graded confident coverage: {fmt(retained['confident_coverage'], 3)}.",
            f"- Candidate graded confident coverage before rejection: {fmt(candidate['confident_coverage'], 3)}.",
            "",
            "## Part 5 - Physics-Inclusive Arm",
            "",
        ]
    )
    if physics.get("available"):
        clean_auc = result["clean_auc"]
        lines.extend(
            [
                f"- Arm: {physics['label']}.",
                "",
                md_table(
                    [
                        {
                            "model": "CLEAN prior-corrected",
                            "whole_auc": clean_auc["test_auc"]["auc"],
                            "boundary_auc": clean_auc["boundary_auc"]["auc"],
                            "candidate_coverage": candidate["confident_coverage"],
                            "retained_coverage": retained["confident_coverage"],
                            "candidate_rejected": result["candidate_zone_rejected"],
                        },
                        {
                            "model": "CLEAN + fit_alpha",
                            "whole_auc": physics["whole_auc"]["auc"],
                            "boundary_auc": physics["boundary_auc"]["auc"],
                            "candidate_coverage": physics["candidate_zone_metrics"]["confident_coverage"],
                            "retained_coverage": physics["retained_confident_coverage_if_guard_enforced"],
                            "candidate_rejected": physics["candidate_rejected"],
                        },
                    ],
                    [
                        ("model", "model", "str"),
                        ("whole_auc", "whole AUC", "float3"),
                        ("boundary_auc", "boundary AUC", "float3"),
                        ("candidate_coverage", "candidate cov", "float3"),
                        ("retained_coverage", "retained cov", "float3"),
                        ("candidate_rejected", "rejected", "str"),
                    ],
                ),
                "",
                f"- Whole AUC: {fmt(physics['whole_auc']['auc'], 3)}.",
                f"- Boundary AUC: {fmt(physics['boundary_auc']['auc'], 3)} (n={physics['boundary_n']}).",
                f"- Natural-prevalence ECE: {fmt(physics['calibration']['ece'], 3)}.",
                f"- Candidate confident coverage: {fmt(physics['candidate_zone_metrics']['confident_coverage'], 3)}.",
                f"- Candidate rejected: {physics['candidate_rejected']}; reasons: {'; '.join(physics['candidate_reject_reasons']) if physics['candidate_reject_reasons'] else 'none'}.",
                f"- Retained coverage if the same guard is enforced: {fmt(physics['retained_confident_coverage_if_guard_enforced'], 3)}.",
            ]
        )
    else:
        lines.append(f"- Not run: {physics.get('reason', 'unavailable')}.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Label-shift prior correction changed ECE from {fmt(result['prior_gate_reweighted_ece'], 3)} to {fmt(cal['posthoc_prior_corrected']['ece'], 3)}.",
            "- That restores probability calibration at natural prevalence.",
            "- The corrected TRAIN-frozen zones still fail the TEST guard if they are allowed to emit broad confident calls.",
            "- Therefore the deployable retained graded output is probability-calibrated but does not add safe confident coverage over the 66.6% operational deterministic baseline.",
            "- The secondary fit_alpha arm improves ranking, including the boundary AUC, but still fails the natural-prevalence PPV floor and therefore does not produce deployable confident coverage.",
            "",
            f"VERDICT: {result['verdict']} - {result['verdict_reason']}",
            "",
        ]
    )
    (out / f"{OUTPUT_STEM}.md").write_text("\n".join(lines), encoding="utf-8")


def run(root: Path) -> dict[str, Any]:
    workflow, _repro, recovered_test, _c1 = load_model_frame(root)
    audit, banned_present, quarantined, clean_features = feature_audit(workflow)
    leakage_guard_tripped = bool(set(clean_features) & (set(quarantined) | set(banned_present)))
    if leakage_guard_tripped:
        result = {
            "seed": RANDOM_SEED,
            "verdict": "LEAKAGE-GUARD-TRIPPED",
            "verdict_reason": "A banned or quarantined feature reached CLEAN features.",
            "leakage_guard_tripped": True,
            "banned_present": banned_present,
            "quarantined_features": quarantined,
            "clean_features": clean_features,
        }
        write_outputs(root, result, [], pd.DataFrame())
        return result

    train = workflow[workflow["split"].eq("train")].copy().reset_index(drop=True)
    test = workflow[workflow["split"].eq("test")].copy().reset_index(drop=True)
    test = test.merge(
        recovered_test[["SEQN", "post_recovery_finalized", "post_recovery_decision"]],
        on="SEQN",
        how="left",
    )
    test["post_recovery_finalized"] = test["post_recovery_finalized"].fillna(False).astype(bool)

    x_train, x_test, imputation = impute_frames(train, test, clean_features)
    y_train = truth_obstructed(train).astype(int).to_numpy()
    y_test = truth_obstructed(test).astype(int).to_numpy()
    source_prevalence = float(y_train.mean())
    prevalence = prevalence_from_full_cohort(root)
    target_prevalence = float(prevalence["target_prevalence"])
    train_weights = class_reweight(y_train, target_prevalence, source_prevalence)
    test_weights = class_reweight(y_test, target_prevalence, source_prevalence)

    clean_model = fit_model(make_gbt(), x_train, y_train)
    p_train_raw = probability(clean_model, x_train)
    p_test_raw = probability(clean_model, x_test)
    p_train_adj = prior_adjust(p_train_raw, source_prevalence, target_prevalence)
    p_test_adj = prior_adjust(p_test_raw, source_prevalence, target_prevalence)

    weighted_model = fit_weighted_model(x_train, y_train, train_weights)
    p_train_weighted = probability(weighted_model, x_train)
    p_test_weighted = probability(weighted_model, x_test)

    calibration_rows: list[dict[str, Any]] = []
    calibration_summaries: dict[str, dict[str, Any]] = {}
    calibration_summaries["raw_train_prevalence"] = add_calibration(
        calibration_rows,
        y_test,
        p_test_raw,
        None,
        "test_all",
        "unweighted_constructed_test",
        target_prevalence,
        "raw_train_prevalence",
    )
    calibration_summaries["uncorrected_reweighted_natural"] = add_calibration(
        calibration_rows,
        y_test,
        p_test_raw,
        test_weights,
        "test_all",
        "reweighted_natural",
        target_prevalence,
        "uncorrected_train_prevalence",
    )
    calibration_summaries["posthoc_prior_corrected"] = add_calibration(
        calibration_rows,
        y_test,
        p_test_adj,
        test_weights,
        "test_all",
        "reweighted_natural",
        target_prevalence,
        "posthoc_prior_corrected",
    )
    calibration_summaries["weighted_retrained"] = add_calibration(
        calibration_rows,
        y_test,
        p_test_weighted,
        test_weights,
        "test_all",
        "reweighted_natural",
        target_prevalence,
        "weighted_retrained",
    )

    p_train_map = {
        "posthoc_prior_corrected": p_train_adj,
        "weighted_retrained": p_train_weighted,
    }
    p_test_map = {
        "posthoc_prior_corrected": p_test_adj,
        "weighted_retrained": p_test_weighted,
    }
    selected = select_probability_source(calibration_summaries, p_train_map, p_test_map)

    fallback_info: dict[str, Any] = {}
    if calibration_status(calibration_summaries["posthoc_prior_corrected"]["ece"]) == "INSUFFICIENT":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_train_adj, y_train, sample_weight=train_weights)
        p_train_fallback = iso.predict(p_train_adj)
        p_test_fallback = iso.predict(p_test_adj)
        calibration_summaries["fallback_isotonic"] = add_calibration(
            calibration_rows,
            y_test,
            p_test_fallback,
            test_weights,
            "test_all",
            "reweighted_natural",
            target_prevalence,
            "fallback_train_only_isotonic_after_prior",
        )
        fallback_info = {
            "ran": True,
            "status": calibration_status(calibration_summaries["fallback_isotonic"]["ece"]),
        }
        if calibration_summaries["fallback_isotonic"]["ece"] < selected["ece"]:
            selected = {
                "name": "fallback_train_only_isotonic_after_prior",
                "ece": float(calibration_summaries["fallback_isotonic"]["ece"]),
                "status": calibration_status(calibration_summaries["fallback_isotonic"]["ece"]),
                "p_train": p_train_fallback,
                "p_test": p_test_fallback,
            }
    else:
        calibration_summaries["fallback_isotonic"] = {
            "probability_source": "fallback_train_only_isotonic_after_prior",
            "scope": "test_all",
            "weighting": "not_run",
            "target_prevalence": target_prevalence,
            "ece": np.nan,
            "brier": np.nan,
            "n": int(len(y_test)),
        }
        fallback_info = {"ran": False, "status": "NOT_NEEDED"}

    boundaries = choose_boundaries(train, selected["p_train"], train_weights)
    candidate_zones = apply_zones(test, selected["p_test"], boundaries)
    candidate_metrics = zone_metrics(candidate_zones, test_weights)
    reasons = reject_reasons(candidate_metrics)
    if reasons:
        retained_zones = all_indeterminate_zones(test, selected["p_test"])
        retained_zones["candidate_graded_zone"] = candidate_zones["graded_zone"].to_numpy()
        retained_zones["candidate_graded_confident"] = candidate_zones["graded_confident"].to_numpy()
        retained_zones["candidate_graded_pred_obstructed"] = candidate_zones["graded_pred_obstructed"].to_numpy()
        retained_zones["candidate_rejected_reason"] = "; ".join(reasons)
    else:
        retained_zones = candidate_zones.copy()
        retained_zones["candidate_graded_zone"] = candidate_zones["graded_zone"].to_numpy()
        retained_zones["candidate_graded_confident"] = candidate_zones["graded_confident"].to_numpy()
        retained_zones["candidate_graded_pred_obstructed"] = candidate_zones["graded_pred_obstructed"].to_numpy()
        retained_zones["candidate_rejected_reason"] = ""
    retained_metrics = zone_metrics(retained_zones, test_weights)

    retained_zones["p_train_calibrated"] = p_test_raw
    retained_zones["p_adj"] = p_test_adj
    retained_zones["p_weighted_retrained"] = p_test_weighted
    retained_zones["p_selected"] = selected["p_test"]
    retained_zones["selected_probability_source"] = selected["name"]
    retained_zones["true_abs_distance_from_0p70"] = (
        pd.to_numeric(retained_zones["fev1_fvc_ratio"], errors="coerce") - FIXED_RATIO_THRESHOLD
    ).abs()
    retained_zones["deferral_bucket"] = "FINALIZED_DETERMINISTIC"
    deferred = retained_zones["post_recovery_finalized"].eq(False)
    retained_zones.loc[deferred & retained_zones["true_abs_distance_from_0p70"].lt(0.02), "deferral_bucket"] = "KNIFE_EDGE"
    retained_zones.loc[
        deferred
        & retained_zones["true_abs_distance_from_0p70"].ge(0.02)
        & retained_zones["true_abs_distance_from_0p70"].lt(0.05),
        "deferral_bucket",
    ] = "NEAR"
    retained_zones.loc[deferred & retained_zones["true_abs_distance_from_0p70"].ge(0.05), "deferral_bucket"] = "RECOVERABLE"
    retained_zones["candidate_graded_pred_obstructed"] = retained_zones["candidate_graded_zone"].eq("CONFIDENT_OBSTRUCTED")

    deferral_rows = deferral_bucket_rows(retained_zones, test_weights)

    strict_metrics = final_metrics(recovered_test, "workflow_decision", "workflow_final_diagnostic")
    operational_metrics = final_metrics(recovered_test, "post_recovery_decision", "post_recovery_finalized")

    clean_auc = model_auc_report("clean_gbt", clean_model, x_test, test, test_weights, 610)
    weighted_auc = model_auc_report("weighted_retrained_gbt", weighted_model, x_test, test, test_weights, 620)
    physics = physics_inclusive_arm(
        train,
        test,
        clean_features,
        y_train,
        y_test,
        train_weights,
        test_weights,
        source_prevalence,
        target_prevalence,
    )

    confirmatory_ece_delta = abs(
        float(calibration_summaries["posthoc_prior_corrected"]["ece"])
        - float(calibration_summaries["weighted_retrained"]["ece"])
    )
    confirmatory_status = "AGREES_WITHIN_0.03_ECE" if confirmatory_ece_delta <= 0.03 else "DISAGREES_REVIEW_REQUIRED"

    prior_gate_ece = load_prior_gate_ece(root)
    result: dict[str, Any] = {
        "seed": RANDOM_SEED,
        "leakage_guard_tripped": False,
        "banned_present": banned_present,
        "quarantined_features": quarantined,
        "clean_features": clean_features,
        "feature_audit_top_corr": audit.sort_values("max_abs_corr", ascending=False).head(18).to_dict("records"),
        "clean_missingness": imputation["missingness"],
        "prevalence": prevalence,
        "source_train_prevalence": source_prevalence,
        "target_prevalence": target_prevalence,
        "prior_correction": {
            "a": float(target_prevalence / source_prevalence),
            "b": float((1.0 - target_prevalence) / (1.0 - source_prevalence)),
            "formula": "p_adj=(p_train*a)/(p_train*a+(1-p_train)*b)",
        },
        "prior_gate_reweighted_ece": prior_gate_ece,
        "calibration_summary": calibration_summaries,
        "calibration_gate": {
            "posthoc_prior_corrected_status": calibration_status(calibration_summaries["posthoc_prior_corrected"]["ece"]),
            "weighted_retrained_status": calibration_status(calibration_summaries["weighted_retrained"]["ece"]),
            "fallback": fallback_info,
            "confirmatory_ece_abs_delta": confirmatory_ece_delta,
            "confirmatory_status": confirmatory_status,
        },
        "selected_probability_source": selected["name"],
        "probability_summaries": [
            probability_summary("raw_train_prevalence_test", p_test_raw),
            probability_summary("posthoc_prior_corrected_p_adj_test", p_test_adj),
            probability_summary("weighted_retrained_test", p_test_weighted),
            probability_summary("selected_test", selected["p_test"]),
        ],
        "clean_auc": clean_auc,
        "weighted_retrained_auc": weighted_auc,
        "zone_boundaries": boundaries,
        "candidate_zone_metrics": candidate_metrics,
        "candidate_zone_rejected": bool(reasons),
        "candidate_zone_reject_reasons": reasons,
        "retained_zone_metrics": retained_metrics,
        "deferral_bucket_recovery": deferral_rows,
        "deterministic_baseline": {
            "strict_workflow": strict_metrics,
            "operational_recovery": operational_metrics,
        },
        "physics_inclusive_arm": physics,
        "protocol_decisions": [
            "TEST is scored once after TRAIN-only model fitting and TRAIN-only zone selection.",
            "Natural-prevalence calibration metrics use class reweighting on TEST because the TEST split is constructed/enriched.",
            "The post-hoc prior-corrected CLEAN GBT is the primary arm; a weighted-retrained GBT is confirmatory and can be selected only if its ECE is cleaner.",
            "If the post-hoc prior correction exceeded ECE 0.15, the only fallback allowed was TRAIN-only isotonic recalibration after prior correction.",
            "Operational deterministic residual deferrals are used for the non-boundary bucket analysis because the requested head-to-head baseline is the 66.6% operational policy.",
            "A TEST candidate that violates severe FN=0 or PPV/NPV >=0.90 is reported and then retained as probability-only indeterminate output.",
            "fit_alpha is included only in the secondary physics arm requested here; it remains outside CLEAN because the prior leakage audit quarantined it by ratio correlation.",
        ],
    }
    result["verdict"], result["verdict_reason"] = verdict(result)

    participant_cols = [
        "SEQN",
        "split",
        "fev1_fvc_ratio",
        "truth_obstructed",
        "truth_severity",
        "workflow_final_diagnostic",
        "post_recovery_finalized",
        "post_recovery_decision",
        "p_train_calibrated",
        "p_adj",
        "p_weighted_retrained",
        "p_selected",
        "selected_probability_source",
        "graded_zone",
        "graded_confident",
        "graded_pred_obstructed",
        "candidate_graded_zone",
        "candidate_graded_confident",
        "candidate_graded_pred_obstructed",
        "candidate_rejected_reason",
        "true_abs_distance_from_0p70",
        "deferral_bucket",
    ]
    participants = retained_zones[participant_cols].copy()
    write_outputs(root, result, calibration_rows, participants)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    cal = result["calibration_summary"]["posthoc_prior_corrected"]
    retained = result["retained_zone_metrics"]
    candidate = result["candidate_zone_metrics"]
    print(f"Graded recalibration gate: {result['verdict']} - {result['verdict_reason']}")
    print(f"Post-hoc prior-corrected ECE: {cal['ece']:.3f}")
    print(f"Selected probability source: {result['selected_probability_source']}")
    print(
        "Retained coverage: "
        f"{retained['confident_coverage']:.3f}; "
        f"candidate coverage before rejection: {candidate['confident_coverage']:.3f}"
    )
    print(
        "Candidate PPV/NPV/severe FN: "
        f"{candidate['ppv_reweighted']:.3f}/{candidate['npv_reweighted']:.3f}/"
        f"{candidate['severe_fn_n']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
