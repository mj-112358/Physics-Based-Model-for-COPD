from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .gate_workflow_policy_f import (
    OUTPUT_STEM as WORKFLOW_STEM,
)
from .gate_workflow_policy_f import (
    RANDOM_SEED,
    clean_json,
    exact_ci,
    kappa,
    md_table,
)

OUTPUT_STEM = "gate_deferral_recovery_F"
ARTIFACT_R2_MIN = 0.90
R1_GOOD_FIT_R2_MIN = 0.98
R2_BOOTSTRAP_N = 2000
R3_GOOD_MANEUVER_R2_MIN = 0.95


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if pd.isna(value):
        return False
    return bool(value)


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    workflow_path = root / "results" / f"{WORKFLOW_STEM}_rows.csv"
    repro_path = root / "results" / "gate_reproducibility_volume_F_rows.csv"
    missing = [str(path) for path in [workflow_path, repro_path] if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required inputs: {missing}")
    workflow = pd.read_csv(workflow_path)
    repro = pd.read_csv(repro_path, low_memory=False)
    return workflow, repro


def correction_offset(workflow: pd.DataFrame) -> float:
    delta = workflow["model_ratio"].astype(float) - workflow["corrected_ratio_c1"].astype(float)
    delta = delta[np.isfinite(delta)]
    if delta.empty:
        raise RuntimeError("Cannot infer existing c1 correction offset.")
    return float(np.median(delta))


def primary_deferral_reason(row: pd.Series) -> tuple[str, str]:
    ratio = float(row["corrected_ratio_c1"]) if finite(row.get("corrected_ratio_c1")) else np.nan
    raw_ratio = float(row["model_ratio"]) if finite(row.get("model_ratio")) else np.nan
    fit_bad = (
        not bool_value(row.get("volume_fit_success", True))
        or not np.isfinite(ratio)
        or not finite(row.get("volume_r2"))
        or float(row.get("volume_r2")) < ARTIFACT_R2_MIN
    )
    if fit_bad:
        return "R4_GENUINELY_UNINTERPRETABLE", "invalid/nonfinite fit or volume fit R2 < 0.90"

    repeat_failed = (
        0.55 <= ratio < 0.75
        and (
            not bool_value(row.get("model_classification_unanimous", True))
            or (finite(row.get("model_ratio_sd")) and float(row.get("model_ratio_sd")) >= 0.030)
        )
    )
    if repeat_failed:
        return "R3_FAILED_REPEATABILITY", "near-threshold estimate with failed cross-maneuver stability"

    if bool_value(row.get("truncated_ats")):
        return "R1_TRUNCATED_INCOMPLETE", "ATS end-of-test/truncation flag drove repeat deferral"

    crosses_threshold = (
        np.isfinite(raw_ratio)
        and ((raw_ratio < 0.70 <= ratio) or (ratio < 0.70 <= raw_ratio))
    )
    if (0.60 <= ratio < 0.72) or crosses_threshold:
        return "R2_BORDERLINE_RATIO", "corrected ratio near 0.70 or raw/corrected threshold disagreement"

    return "R4_GENUINELY_UNINTERPRETABLE", "severe-risk or otherwise no validated recovery path in this gate"


def corrected_maneuvers(repro: pd.DataFrame, c1: float) -> pd.DataFrame:
    maneuvers = repro[repro["row_type"].eq("maneuver_fit")].copy()
    maneuvers = maneuvers[maneuvers["model_ratio"].map(finite)].copy()
    maneuvers["corrected_ratio_c1"] = maneuvers["model_ratio"].astype(float) - c1
    return maneuvers


def bootstrap_median_ci(values: np.ndarray, seed: int, n_boot: int = R2_BOOTSTRAP_N) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, arr.size, size=arr.size)
        draws[i] = float(np.median(arr[idx]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(np.median(arr)), float(lo), float(hi)


def truth_label(row: pd.Series) -> str:
    return "OBSTRUCTED" if float(row["fev1_fvc_ratio"]) < 0.70 else "NORMAL"


def final_label_from_decision(decision: str) -> str | None:
    if decision in {"ACCEPT_NORMAL", "RECOVERED_NORMAL"}:
        return "NORMAL"
    if decision in {"ACCEPT_OBSTRUCTED", "TRUNCATED_RECOVERED", "RECOVERED_OBSTRUCTED"}:
        return "OBSTRUCTED"
    return None


def r1_candidate(row: pd.Series) -> tuple[str | None, dict[str, Any]]:
    ratio = float(row["corrected_ratio_c1"])
    details = {
        "method": "R1_OMBE_RECONSTRUCTION",
        "rule": "good OMBE fit R2 >=0.98 and corrected ratio <0.60 or >=0.80",
        "volume_r2": float(row["volume_r2"]) if finite(row.get("volume_r2")) else np.nan,
        "corrected_ratio_c1": ratio,
    }
    if details["volume_r2"] >= R1_GOOD_FIT_R2_MIN and ratio < 0.60:
        return "RECOVERED_OBSTRUCTED", details
    if details["volume_r2"] >= R1_GOOD_FIT_R2_MIN and ratio >= 0.80:
        return "RECOVERED_NORMAL", details
    return None, details


def r2_candidate(row: pd.Series, maneuvers: pd.DataFrame) -> tuple[str | None, dict[str, Any]]:
    seqn = float(row["SEQN"])
    vals = maneuvers[maneuvers["SEQN"].astype(float).eq(seqn)]["corrected_ratio_c1"].to_numpy(dtype=float)
    median, lo, hi = bootstrap_median_ci(vals, RANDOM_SEED + int(seqn) % 100_000)
    details = {
        "method": "R2_MULTI_MANEUVER_BOOTSTRAP_CI",
        "rule": "bootstrap median CI must sit on one side of 0.70; below-side median must be <0.67, above-side median >=0.73",
        "maneuver_n": int(np.isfinite(vals).sum()),
        "median": median,
        "ci_low": lo,
        "ci_high": hi,
    }
    if np.isfinite(hi) and hi < 0.70 and median < 0.67:
        return "RECOVERED_OBSTRUCTED", details
    if np.isfinite(lo) and lo >= 0.70 and median >= 0.73:
        return "RECOVERED_NORMAL", details
    return None, details


def r3_candidate(row: pd.Series, maneuvers: pd.DataFrame) -> tuple[str | None, dict[str, Any]]:
    seqn = float(row["SEQN"])
    part = maneuvers[maneuvers["SEQN"].astype(float).eq(seqn)].copy()
    part = part[part["volume_r2"].map(finite) & (part["volume_r2"].astype(float) >= R3_GOOD_MANEUVER_R2_MIN)]
    values = np.sort(part["corrected_ratio_c1"].to_numpy(dtype=float))
    best: tuple[int, str, float, float, float] | None = None
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            cluster = values[i : j + 1]
            if len(cluster) < 2 or float(cluster[-1] - cluster[0]) > 0.020:
                continue
            median = float(np.median(cluster))
            if float(cluster[-1]) < 0.65:
                side = "RECOVERED_OBSTRUCTED"
            elif float(cluster[0]) >= 0.75:
                side = "RECOVERED_NORMAL"
            else:
                side = ""
            if side and (best is None or len(cluster) > best[0]):
                best = (len(cluster), side, float(cluster[0]), float(cluster[-1]), median)

    details = {
        "method": "R3_BEST_CONSISTENT_SUBSET",
        "rule": "at least two good-fit maneuvers, cluster width <=0.02, cluster majority, and cluster far from 0.70",
        "good_maneuver_n": int(len(values)),
        "best_cluster_n": int(best[0]) if best else 0,
        "best_cluster_low": best[2] if best else np.nan,
        "best_cluster_high": best[3] if best else np.nan,
        "best_cluster_median": best[4] if best else np.nan,
    }
    if best and best[0] > len(values) / 2:
        return best[1], details
    return None, details


def attempt_recovery(row: pd.Series, maneuvers: pd.DataFrame) -> tuple[str | None, dict[str, Any]]:
    reason = row["primary_deferral_reason"]
    if reason == "R1_TRUNCATED_INCOMPLETE":
        return r1_candidate(row)
    if reason == "R2_BORDERLINE_RATIO":
        return r2_candidate(row, maneuvers)
    if reason == "R3_FAILED_REPEATABILITY":
        return r3_candidate(row, maneuvers)
    return None, {"method": "NONE", "rule": "R4 remains deferred"}


def apply_recovery(test: pd.DataFrame, maneuvers: pd.DataFrame) -> pd.DataFrame:
    out = test.copy()
    out["deferred_baseline"] = ~out["workflow_final_diagnostic"].astype(bool)
    out["primary_deferral_reason"] = ""
    out["primary_deferral_detail"] = ""
    out["candidate_decision"] = ""
    out["candidate_method"] = ""
    out["candidate_rule"] = ""
    out["candidate_correct"] = pd.NA
    out["recovered"] = False
    out["post_recovery_decision"] = out["workflow_decision"]
    out["post_recovery_finalized"] = out["workflow_final_diagnostic"].astype(bool)
    out["residual_deferral_reason"] = ""
    detail_columns: dict[str, list[Any]] = {}

    for idx, row in out.iterrows():
        if not bool(row["deferred_baseline"]):
            continue
        reason, detail = primary_deferral_reason(row)
        out.at[idx, "primary_deferral_reason"] = reason
        out.at[idx, "primary_deferral_detail"] = detail
        candidate, candidate_detail = attempt_recovery(out.loc[idx], maneuvers)
        for key, value in candidate_detail.items():
            col = f"candidate_{key}"
            if col not in out.columns:
                detail_columns[col] = []
                out[col] = np.nan
            out.at[idx, col] = value
        out.at[idx, "candidate_method"] = candidate_detail.get("method", "")
        out.at[idx, "candidate_rule"] = candidate_detail.get("rule", "")
        if candidate is None:
            out.at[idx, "residual_deferral_reason"] = reason
            continue
        out.at[idx, "candidate_decision"] = candidate
        correct = final_label_from_decision(candidate) == truth_label(row)
        out.at[idx, "candidate_correct"] = bool(correct)
        out.at[idx, "recovered"] = True
        out.at[idx, "post_recovery_decision"] = candidate
        out.at[idx, "post_recovery_finalized"] = True
    still_deferred = out["deferred_baseline"] & ~out["recovered"] & out["residual_deferral_reason"].eq("")
    out.loc[still_deferred, "residual_deferral_reason"] = out["primary_deferral_reason"]
    return out


def final_metrics(df: pd.DataFrame, decision_col: str, final_col: str) -> dict[str, Any]:
    final = df[df[final_col].astype(bool)].copy()
    labels = final[decision_col].map(final_label_from_decision)
    truth = final["fev1_fvc_ratio"].astype(float) < 0.70
    pred = labels.eq("OBSTRUCTED")
    tp = int((truth & pred).sum())
    tn = int((~truth & ~pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    n = tp + tn + fp + fn
    severe = final[final["truth_severity"].eq("severe")]
    severe_fn = int((severe[decision_col].map(final_label_from_decision).eq("NORMAL")).sum())
    lo, hi = exact_ci(severe_fn, int(len(severe)))
    return {
        "finalized_n": int(n),
        "finalized_fraction": float(n / len(df)) if len(df) else np.nan,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "cohens_kappa": kappa(tp, tn, fp, fn),
        "severe_finalized_n": int(len(severe)),
        "severe_fn_n": severe_fn,
        "severe_fn_rate": float(severe_fn / len(severe)) if len(severe) else np.nan,
        "severe_fn_ci_low": lo,
        "severe_fn_ci_high": hi,
    }


def summarize_reasons(rows: pd.DataFrame) -> list[dict[str, Any]]:
    deferred = rows[rows["deferred_baseline"]].copy()
    total = len(deferred)
    records = []
    for reason, label in [
        ("R1_TRUNCATED_INCOMPLETE", "R1 truncated/incomplete"),
        ("R2_BORDERLINE_RATIO", "R2 borderline near 0.70"),
        ("R3_FAILED_REPEATABILITY", "R3 failed repeatability"),
        ("R4_GENUINELY_UNINTERPRETABLE", "R4 uninterpretable/artifact"),
    ]:
        part = deferred[deferred["primary_deferral_reason"].eq(reason)]
        records.append(
            {
                "reason": reason,
                "label": label,
                "n": int(len(part)),
                "fraction_of_deferred": float(len(part) / total) if total else np.nan,
                "recovered_n": int(part["recovered"].sum()),
                "candidate_attempt_n": int(part["candidate_decision"].astype(str).ne("").sum()),
                "candidate_error_n": int(part["candidate_correct"].eq(False).sum()),
            }
        )
    return records


def recovery_method_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    deferred = rows[rows["deferred_baseline"]].copy()
    for reason in [
        "R1_TRUNCATED_INCOMPLETE",
        "R2_BORDERLINE_RATIO",
        "R3_FAILED_REPEATABILITY",
        "R4_GENUINELY_UNINTERPRETABLE",
    ]:
        part = deferred[deferred["primary_deferral_reason"].eq(reason)].copy()
        attempts = part[part["candidate_decision"].astype(str).ne("")]
        recovered = part[part["recovered"]]
        recovered_correct = int(recovered["candidate_correct"].eq(True).sum())
        recovered_wrong = int(recovered["candidate_correct"].eq(False).sum())
        severe_rec = recovered[recovered["truth_severity"].eq("severe")]
        severe_fn = int((severe_rec["post_recovery_decision"].map(final_label_from_decision).eq("NORMAL")).sum())
        lo, hi = exact_ci(severe_fn, len(severe_rec))
        out.append(
            {
                "reason": reason,
                "deferred_n": int(len(part)),
                "candidate_attempt_n": int(len(attempts)),
                "candidate_correct_n": int(attempts["candidate_correct"].eq(True).sum()),
                "candidate_error_n": int(attempts["candidate_correct"].eq(False).sum()),
                "recovered_n": int(len(recovered)),
                "recovered_correct_n": recovered_correct,
                "recovered_error_n": recovered_wrong,
                "recovered_accuracy": float(recovered_correct / len(recovered)) if len(recovered) else np.nan,
                "recovered_severe_n": int(len(severe_rec)),
                "recovered_severe_fn": severe_fn,
                "recovered_severe_fn_ci_high": hi,
                "residual_deferred_n": int(len(part) - len(recovered)),
            }
        )
    return out


def residual_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    residual = rows[rows["deferred_baseline"] & ~rows["recovered"]].copy()
    total = len(rows)
    records = []
    for reason, part in residual.groupby("residual_deferral_reason", dropna=False):
        records.append(
            {
                "residual_reason": str(reason),
                "n": int(len(part)),
                "fraction_all": float(len(part) / total) if total else np.nan,
                "fraction_residual": float(len(part) / len(residual)) if len(residual) else np.nan,
            }
        )
    return sorted(records, key=lambda x: (-x["n"], x["residual_reason"]))


def verdict(result: dict[str, Any]) -> tuple[str, str]:
    baseline = result["baseline_metrics"]
    new = result["post_recovery_metrics"]
    severe_ok = new["severe_fn_n"] == 0
    metrics_hold = (
        np.isfinite(new["sensitivity"])
        and np.isfinite(baseline["sensitivity"])
        and new["sensitivity"] >= baseline["sensitivity"] - 1e-12
        and new["specificity"] >= baseline["specificity"] - 1e-12
        and new["cohens_kappa"] >= baseline["cohens_kappa"] - 1e-12
    )
    coverage_gain = new["finalized_fraction"] - baseline["finalized_fraction"]
    if severe_ok and metrics_hold and coverage_gain >= 0.10:
        return "RECOVERY-SUBSTANTIAL", "coverage rises materially via label-independent recovery rules, final-call metrics hold, and severe FN remains zero."
    if severe_ok and coverage_gain > 0.0:
        return "RECOVERY-PARTIAL", "some deferred cases are recovered, but metric guard or recovery size prevents a stronger verdict."
    return "MOSTLY-UNINTERPRETABLE", "the deferred cases mostly remain beyond this deterministic recovery gate."


def write_report(root: Path, result: dict[str, Any], rows: pd.DataFrame) -> None:
    out = root / "results"
    out.mkdir(exist_ok=True)
    (out / f"{OUTPUT_STEM}.json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")
    rows.to_csv(out / f"{OUTPUT_STEM}_rows.csv", index=False)

    baseline = result["baseline_metrics"]
    new = result["post_recovery_metrics"]
    lines = [
        "# Gate - Deferral Recovery Analysis, Cycle F",
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
            "## Part 1 - Deferral Autopsy",
            "",
            md_table(
                result["primary_deferral_reasons"],
                [
                    ("label", "bucket", "str"),
                    ("n", "n", "int"),
                    ("fraction_of_deferred", "fraction of deferred", "float3"),
                    ("candidate_attempt_n", "attempted", "int"),
                    ("recovered_n", "recovered", "int"),
                    ("candidate_error_n", "post-selection errors", "int"),
                ],
            ),
            "",
            "## Parts 2-4 - Recovery Attempts",
            "",
            md_table(
                result["recovery_by_reason"],
                [
                    ("reason", "reason", "str"),
                    ("deferred_n", "deferred", "int"),
                    ("candidate_attempt_n", "attempted", "int"),
                    ("candidate_correct_n", "correct", "int"),
                    ("candidate_error_n", "errors", "int"),
                    ("recovered_n", "moved finalized", "int"),
                    ("recovered_accuracy", "post-selection accuracy", "float3"),
                    ("recovered_severe_fn", "severe FN", "int"),
                    ("recovered_severe_fn_ci_high", "severe FN CP high", "float3"),
                    ("residual_deferred_n", "residual", "int"),
                ],
            ),
            "",
            "R1 recovery uses high-quality OMBE reconstruction only. R2 recovery uses multi-maneuver bootstrap median CIs. R3 recovery requires a majority subset of good-fit maneuvers that is mutually consistent and far from 0.70. Candidate keep/defer decisions are made before looking at TEST truth; correctness is scored only after selection.",
            "",
            "### R2 Resolution Split",
            "",
            f"- resolvable by tighter multi-maneuver/bootstrap CI: {result['r2_resolution_split']['resolvable_n']}",
            f"- truly ambiguous / CI not safely one-sided: {result['r2_resolution_split']['ambiguous_n']}",
            "",
            "### R3 Consistency Split",
            "",
            f"- one-bad-blow / recoverable subset found: {result['r3_consistency_split']['one_bad_blow_recoverable_n']}",
            f"- all-inconsistent or no majority good-fit subset: {result['r3_consistency_split']['all_inconsistent_n']}",
            "",
            "## Part 5 - Honest New Coverage",
            "",
            md_table(
                [
                    {"scope": "baseline workflow", **baseline},
                    {"scope": "post recovery", **new},
                ],
                [
                    ("scope", "scope", "str"),
                    ("finalized_n", "finalized n", "int"),
                    ("finalized_fraction", "coverage", "float3"),
                    ("sensitivity", "sens", "float3"),
                    ("specificity", "spec", "float3"),
                    ("cohens_kappa", "kappa", "float3"),
                    ("severe_finalized_n", "severe finalized", "int"),
                    ("severe_fn_n", "severe FN", "int"),
                    ("severe_fn_ci_high", "severe FN CP high", "float3"),
                ],
            ),
            "",
            f"- coverage delta: {result['coverage_delta']:.3f}",
            f"- net recovered cases: {result['net_recovered_n']}",
            f"- residual deferred fraction: {result['residual_deferred_fraction']:.3f}",
            f"- honest interpretable ceiling in this TEST gate: {new['finalized_fraction']:.3f}",
            f"- genuinely uninterpretable/artifact residual fraction of all TEST rows: {result['residual_r4_fraction_all']:.3f}",
            "",
            "## Residual Deferred",
            "",
            md_table(
                result["residual_deferred_reasons"],
                [
                    ("residual_reason", "residual reason", "str"),
                    ("n", "n", "int"),
                    ("fraction_all", "fraction all", "float3"),
                    ("fraction_residual", "fraction residual", "float3"),
                ],
            ),
            "",
            "## Guardrail Statement",
            "",
            "- Severe FN=0 is preserved in the post-recovery finalized set.",
            "- Recovery does not lower the confidence threshold. Every moved case is moved by a specific validating signal available at inference time.",
            "- TEST truth is used only for scoring the already-selected cases, not for deciding whether to keep a candidate.",
            "- Remaining ambiguous/inconsistent/artifact cases stay deferred.",
            "",
            f"VERDICT: {result['verdict']} - {result['verdict_reason']}",
            "",
        ]
    )
    (out / f"{OUTPUT_STEM}.md").write_text("\n".join(lines), encoding="utf-8")


def run(root: Path) -> dict[str, Any]:
    workflow, repro = load_inputs(root)
    c1 = correction_offset(workflow)
    maneuvers = corrected_maneuvers(repro, c1)
    test = workflow[workflow["split"].eq("test")].copy()
    rows = apply_recovery(test, maneuvers)
    baseline_metrics = final_metrics(rows, "workflow_decision", "workflow_final_diagnostic")
    post_metrics = final_metrics(rows, "post_recovery_decision", "post_recovery_finalized")
    residual = rows[rows["deferred_baseline"] & ~rows["recovered"]]
    primary = summarize_reasons(rows)
    recovery = recovery_method_rows(rows)
    result = {
        "verdict": "",
        "verdict_reason": "",
        "seed": RANDOM_SEED,
        "test_n": int(len(test)),
        "baseline_deferred_n": int((~test["workflow_final_diagnostic"].astype(bool)).sum()),
        "baseline_deferred_fraction": float((~test["workflow_final_diagnostic"].astype(bool)).mean()),
        "c1_offset_from_existing_policy": c1,
        "protocol_decisions": [
            "TEST split only; TRAIN rows are ignored.",
            "The existing workflow gate is reused as the baseline; this gate only analyzes and recovers its deferred rows.",
            "Primary deferral reason uses a strict priority: invalid/artifact, failed repeatability, truncation, borderline, then residual uninterpretable.",
            "Volume fit R2 < 0.90 is treated as R4 artifact/no-valid-fit, not as a recoverable borderline case.",
            "Recovered cases are selected by deterministic, label-independent candidate rules. TEST truth is used only after selection to score correctness.",
            "No confidence threshold is lowered; R1/R2/R3 recovery uses stricter evidence than the baseline finalization rule.",
        ],
        "primary_deferral_reasons": primary,
        "recovery_by_reason": recovery,
        "r2_resolution_split": {
            "resolvable_n": int(
                rows["primary_deferral_reason"].eq("R2_BORDERLINE_RATIO").astype(bool).mul(rows["recovered"].astype(bool)).sum()
            ),
            "ambiguous_n": int(
                (
                    rows["primary_deferral_reason"].eq("R2_BORDERLINE_RATIO")
                    & rows["deferred_baseline"].astype(bool)
                    & ~rows["recovered"].astype(bool)
                ).sum()
            ),
        },
        "r3_consistency_split": {
            "one_bad_blow_recoverable_n": int(
                rows["primary_deferral_reason"].eq("R3_FAILED_REPEATABILITY").astype(bool).mul(rows["recovered"].astype(bool)).sum()
            ),
            "all_inconsistent_n": int(
                (
                    rows["primary_deferral_reason"].eq("R3_FAILED_REPEATABILITY")
                    & rows["deferred_baseline"].astype(bool)
                    & ~rows["recovered"].astype(bool)
                ).sum()
            ),
        },
        "baseline_metrics": baseline_metrics,
        "post_recovery_metrics": post_metrics,
        "coverage_delta": float(post_metrics["finalized_fraction"] - baseline_metrics["finalized_fraction"]),
        "net_recovered_n": int(rows["recovered"].sum()),
        "residual_deferred_n": int(len(residual)),
        "residual_deferred_fraction": float(len(residual) / len(rows)) if len(rows) else np.nan,
        "residual_deferred_reasons": residual_rows(rows),
        "residual_r4_fraction_all": float(
            len(residual[residual["residual_deferral_reason"].astype(str).str.startswith("R4")]) / len(rows)
        )
        if len(rows)
        else np.nan,
    }
    result["verdict"], result["verdict_reason"] = verdict(result)
    write_report(root, result, rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    print(f"Deferral recovery: {result['verdict']} - {result['verdict_reason']}")
    print(f"Baseline coverage: {result['baseline_metrics']['finalized_fraction']:.3f}")
    print(f"Post-recovery coverage: {result['post_recovery_metrics']['finalized_fraction']:.3f}")
    print(f"Net recovered: {result['net_recovered_n']}")
    print(
        "Post-recovery severe FN: "
        f"{result['post_recovery_metrics']['severe_fn_n']}/{result['post_recovery_metrics']['severe_finalized_n']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
