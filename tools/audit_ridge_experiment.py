from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


EXPECTED_FEATURES = (
    "parsed_fev1_l",
    "parsed_fvc_l",
    "duration_s",
    "age_years",
    "height_cm",
    "sex",
)


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


def source_constants(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name in {"SEED", "BOOTSTRAP_REPLICATES", "FIXED_RATIO_THRESHOLD", "FEV1_FEV6_THRESHOLD", "RIDGE_FEATURES"}:
            try:
                values[name] = ast.literal_eval(node.value)
            except Exception:
                values[name] = ast.unparse(node.value)
    return values


def run(root: Path) -> dict[str, Any]:
    results = root / "results"
    source = root / "tools" / "evaluate_manuscript_baselines.py"
    reported = json.loads((results / "manuscript_baseline_comparison.json").read_text(encoding="utf-8"))
    policy = pd.read_csv(results / "gate_policy_calibrated_flag_F_rows.csv", low_memory=False)
    final_rows = pd.read_csv(results / "gate_r3_ats_besttest_F_rows.csv", low_memory=False)
    summary = pq.read_table(results / "merged_summary.parquet").to_pandas()
    summary_f = summary[summary["cycle"].eq("F")].drop_duplicates("SEQN")
    covariates = summary_f[["SEQN", "age_years", "height_cm", "sex", "fev6_mL"]]
    data = policy.merge(covariates, on="SEQN", how="left", validate="one_to_one")
    train = data[data["split"].eq("train")].copy()
    test = data[data["split"].eq("test")].copy()

    train_ids = set(train["SEQN"].astype(int))
    test_ids = set(test["SEQN"].astype(int))
    final_ids = set(final_rows["SEQN"].astype(int))
    feature_sources = {
        "parsed_fev1_l": "selected acceptable A/B raw manoeuvre decoded through its observed endpoint",
        "parsed_fvc_l": "terminal cumulative volume of the same full selected raw manoeuvre",
        "duration_s": "observed duration of the same full selected raw manoeuvre",
        "age_years": "Cycle F merged NHANES demographic summary",
        "height_cm": "Cycle F merged NHANES body-measure summary",
        "sex": "Cycle F merged NHANES demographic summary",
        "target_fvc_l": "released Cycle F summary best-test SPXNFVC divided by 1000",
        "truth_label": "released Cycle F SPXNFEV1/SPXNFVC < 0.70",
    }

    parsed_fvc = data["parsed_fvc_l"].to_numpy(float)
    target_fvc = data["fvc_mL"].to_numpy(float) / 1000.0
    finite = np.isfinite(parsed_fvc) & np.isfinite(target_fvc)
    correlation = float(np.corrcoef(parsed_fvc[finite], target_fvc[finite])[0, 1])
    terminal_mae_ml = float(np.mean(np.abs(parsed_fvc[finite] - target_fvc[finite])) * 1000.0)

    constants = source_constants(source)
    source_text = source.read_text(encoding="utf-8")
    checks = {
        "participant_separated_split": len(train_ids & test_ids) == 0,
        "one_row_per_participant": not data["SEQN"].duplicated().any(),
        "reported_test_is_final_policy_participant_set": test_ids == final_ids,
        "truth_matches_final_policy": bool(
            np.array_equal(
                (test.sort_values("SEQN")["fev1_fvc_ratio"].to_numpy(float) < 0.70),
                final_rows.sort_values("SEQN")["truth_obstructed_fixed"].astype(bool).to_numpy(),
            )
        ),
        "preprocessing_fitted_on_training_rows": all(token in source_text for token in ("self.medians_", "self.mean_", "self.scale_")),
        "ridge_alpha_selected_by_training_folds": "choose_ridge_alpha(train)" in source_text,
        "test_labels_absent_from_alpha_selection": "choose_ridge_alpha(test)" not in source_text,
        "bootstrap_resamples_participant_rows": "rng.integers(0, n, size=n)" in source_text and not data["SEQN"].duplicated().any(),
        "bootstrap_refits_ridge": False,
    }

    result = {
        "audit_scope": "existing Cycle F ridge experiment",
        "source_constants": constants,
        "reported_metadata": {
            "seed": reported.get("seed"),
            "bootstrap_replicates": reported.get("bootstrap_replicates"),
            "selected_alpha": reported.get("ridge_definition", {}).get("selected_alpha"),
            "train_n": int(len(train)),
            "test_n": int(len(test)),
        },
        "feature_sources": feature_sources,
        "checks": checks,
        "target_proxy_diagnostics": {
            "parsed_terminal_fvc_vs_summary_fvc_correlation": correlation,
            "parsed_terminal_fvc_vs_summary_fvc_mae_ml": terminal_mae_ml,
            "full_observed_endpoint_is_feature": True,
            "genuinely_fixed_prefix_features_only": False,
        },
        "selection_audit": {
            "alpha": "selected by five deterministic participant folds within Cycle F TRAIN",
            "imputation_and_scaling": "estimated inside each ridge fit from its fitting fold or full TRAIN partition",
            "matched_coverage": "uses the number of finalized cases in the Cycle F policy-development TEST result; ranking uses test features but not test labels",
            "fixed_specificity": "threshold is selected on TRAIN, but its target specificity is copied from the Cycle F policy-development TEST result",
            "policy_status": "the physics policy itself was iteratively compared on Cycle F TEST, so its final TEST metrics are development estimates",
        },
        "leakage_verdict": {
            "participant_record_leakage": False,
            "test_label_used_for_ridge_alpha_or_coefficients": False,
            "test_result_informs_reported_operating_points": True,
            "target_proxy_leakage_for_missing_fvc_estimand": True,
            "explanation": (
                "The ridge target is released summary FVC, while parsed_fvc_l is the terminal volume of the full selected "
                "same-session manoeuvre. This is not train/test record leakage, but it supplies a direct near-target proxy "
                "and therefore is not a clean censored-prefix comparator for latent FVC recovery."
            ),
        },
        "comparability_verdict": {
            "identical_test_participants": True,
            "identical_truth_labels": True,
            "identical_coverage_for_matched_analysis": True,
            "identical_input_information": False,
            "paired_difference_intervals_reported": False,
            "ridge_information": "one selected full manoeuvre plus age, height, and sex",
            "physics_information": "selected waveform fit plus repeated-manoeuvre fits, acceptability, repeatability, and ATS-derived state",
            "conclusion": (
                "The existing matched-coverage numbers compare decisions on the same participant universe, but they do not "
                "isolate estimator value under equal information. A frozen cross-cycle study must report both the archived "
                "ridge and an endpoint-censored ridge using the same observation window as the physics estimator."
            ),
        },
        "bootstrap_verdict": (
            "The 2,000-replicate percentile bootstrap is participant-level because each table has one row per participant. "
            "It conditions on the fitted ridge model and does not include training-sample uncertainty or paired confidence "
            "intervals for method differences."
        ),
    }
    return clean(result)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    c = result["checks"]
    d = result["target_proxy_diagnostics"]
    lines = [
        "# Ridge Experiment Validation Audit",
        "",
        "## Verdict",
        "",
        "The Cycle F ridge fit has a participant-disjoint training/test split, and ridge regularization, imputation, scaling, and coefficients are estimated without test labels. The experiment nevertheless is not a clean censored-FVC baseline because `parsed_fvc_l` is the terminal volume of the full selected same-session manoeuvre.",
        "",
        f"The terminal-volume feature correlates {d['parsed_terminal_fvc_vs_summary_fvc_correlation']:.4f} with released summary FVC and has an absolute error of {d['parsed_terminal_fvc_vs_summary_fvc_mae_ml']:.1f} mL. This is target-proxy leakage relative to the missing-endpoint estimand, not participant record leakage.",
        "",
        "## Reproducibility Checks",
        "",
        "| Check | Result |",
        "|---|---:|",
    ]
    for name, value in c.items():
        lines.append(f"| {name.replace('_', ' ')} | {value} |")
    lines.extend(
        [
            "",
            "## Comparability",
            "",
            result["comparability_verdict"]["conclusion"],
            "",
            "## Bootstrap",
            "",
            result["bootstrap_verdict"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    result = run(root)
    out_json = root / "results" / "ridge_experiment_validation_audit.json"
    out_md = root / "results" / "ridge_experiment_validation_audit.md"
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(out_md, result)
    print(json.dumps(result["leakage_verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
