from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta

OUTPUT_STEM = "gate_workflow_policy_F"
RANDOM_SEED = 20260710
BOOTSTRAP_N = 2000

DECISION_ORDER = [
    "ACCEPT_NORMAL",
    "ACCEPT_OBSTRUCTED",
    "BORDERLINE_REPEAT",
    "TRUNCATED_RECOVERED",
    "REFER_SEVERE_OR_UNSTABLE",
    "UNUSABLE",
]

FINAL_DIAGNOSTIC = {"ACCEPT_NORMAL", "ACCEPT_OBSTRUCTED", "TRUNCATED_RECOVERED"}
OBSTRUCTED_ACTION = {"ACCEPT_OBSTRUCTED", "TRUNCATED_RECOVERED", "REFER_SEVERE_OR_UNSTABLE"}
NONFINAL_ACTION = {"BORDERLINE_REPEAT", "REFER_SEVERE_OR_UNSTABLE", "UNUSABLE"}


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if np.isfinite(out) else None
    if value is pd.NA:
        return None
    try:
        if pd.isna(value) and not isinstance(value, (str, bytes)):
            return None
    except Exception:
        pass
    return value


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if pd.isna(value):
        return False
    return bool(value)


def fmt(value: Any, digits: int = 3) -> str:
    try:
        out = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(out):
        return "NA"
    return f"{out:.{digits}f}"


def exact_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def rate_record(name: str, numerator: int, denominator: int) -> dict[str, Any]:
    lo, hi = exact_ci(numerator, denominator)
    return {
        "metric": name,
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": float(numerator / denominator) if denominator else np.nan,
        "ci_low": lo,
        "ci_high": hi,
    }


def kappa(tp: int, tn: int, fp: int, fn: int) -> float:
    n = tp + tn + fp + fn
    if n == 0:
        return np.nan
    po = (tp + tn) / n
    p_yes_true = (tp + fn) / n
    p_yes_pred = (tp + fp) / n
    p_no_true = (tn + fp) / n
    p_no_pred = (tn + fn) / n
    pe = p_yes_true * p_yes_pred + p_no_true * p_no_pred
    if np.isclose(1.0 - pe, 0.0):
        return np.nan
    return float((po - pe) / (1.0 - pe))


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str, str]]) -> str:
    lines = ["| " + " | ".join(label for _, label, _ in columns) + " |"]
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        cells = []
        for key, _, kind in columns:
            value = row.get(key)
            if kind == "int":
                cells.append(str(int(value)) if value is not None and pd.notna(value) else "NA")
            elif kind.startswith("float"):
                digits = int(kind.replace("float", "") or "3")
                cells.append(fmt(value, digits))
            elif kind == "rate":
                n = row.get(f"{key}_n", row.get("denominator"))
                k = row.get(f"{key}_k", row.get("numerator"))
                if value is None or not finite(value):
                    cells.append("NA")
                elif k is not None and n is not None and pd.notna(k) and pd.notna(n):
                    cells.append(f"{float(value):.3f} ({int(k)}/{int(n)})")
                else:
                    cells.append(f"{float(value):.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_path = root / "results" / "gate_policy_calibrated_flag_F_rows.csv"
    repro_path = root / "results" / "gate_reproducibility_volume_F_rows.csv"
    missing = [str(path) for path in [policy_path, repro_path] if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required inputs: {missing}")
    policy = pd.read_csv(policy_path)
    repro = pd.read_csv(repro_path, low_memory=False)
    return policy, repro


def correction_offset(policy: pd.DataFrame) -> float:
    diff = policy["model_ratio"].astype(float) - policy["corrected_ratio_c1"].astype(float)
    diff = diff[np.isfinite(diff)]
    if diff.empty:
        raise RuntimeError("Cannot infer c1 correction offset from policy rows.")
    return float(np.median(diff))


def add_repeatability(policy: pd.DataFrame, repro: pd.DataFrame) -> pd.DataFrame:
    summaries = repro[repro["row_type"].eq("participant_summary")].copy()
    keep = [
        "SEQN",
        "maneuver_count",
        "model_ratio_sd",
        "model_fev1_sd_l",
        "model_fvc_sd_l",
        "model_classification_unanimous",
        "model_classification_majority_fraction",
    ]
    available = [col for col in keep if col in summaries.columns]
    summaries = summaries[available].drop_duplicates("SEQN")
    out = policy.merge(summaries, on="SEQN", how="left", suffixes=("", "_repeat"))
    return out


def workflow_decision(row: pd.Series) -> str:
    ratio = row.get("corrected_ratio_c1")
    raw_ratio = row.get("model_ratio")
    if not finite(ratio):
        return "UNUSABLE"
    if "volume_fit_success" in row and not bool_value(row.get("volume_fit_success")):
        return "UNUSABLE"
    if str(row.get("fit_error", "")).strip() not in {"", "nan", "None"}:
        return "UNUSABLE"

    ratio = float(ratio)
    raw_ratio = float(raw_ratio) if finite(raw_ratio) else np.nan
    truncated = bool_value(row.get("truncated_ats"))
    repeat_known = not pd.isna(row.get("model_classification_unanimous", np.nan))
    repeated_unstable = repeat_known and not bool_value(row.get("model_classification_unanimous"))
    ratio_sd = float(row.get("model_ratio_sd")) if finite(row.get("model_ratio_sd")) else np.nan
    near_decision_boundary = 0.55 <= ratio < 0.75
    high_spread = np.isfinite(ratio_sd) and ratio_sd >= 0.030

    if ratio < 0.50:
        return "REFER_SEVERE_OR_UNSTABLE"
    if near_decision_boundary and (repeated_unstable or high_spread):
        return "REFER_SEVERE_OR_UNSTABLE"
    if truncated and ratio < 0.70:
        return "TRUNCATED_RECOVERED"
    if truncated:
        return "BORDERLINE_REPEAT"
    if np.isfinite(raw_ratio) and ((raw_ratio < 0.70 <= ratio) or (ratio < 0.70 <= raw_ratio)):
        return "BORDERLINE_REPEAT"
    if 0.60 <= ratio < 0.72:
        return "BORDERLINE_REPEAT"
    if ratio < 0.60:
        return "ACCEPT_OBSTRUCTED"
    return "ACCEPT_NORMAL"


def add_workflow_columns(policy: pd.DataFrame) -> pd.DataFrame:
    out = policy.copy()
    out["workflow_decision"] = out.apply(workflow_decision, axis=1)
    out["truth_obstructed_fixed"] = out["fev1_fvc_ratio"].astype(float) < 0.70
    out["truth_normal_fixed"] = ~out["truth_obstructed_fixed"]
    out["workflow_accepted_normal"] = out["workflow_decision"].eq("ACCEPT_NORMAL")
    out["workflow_obstructed_action"] = out["workflow_decision"].isin(OBSTRUCTED_ACTION)
    out["workflow_final_diagnostic"] = out["workflow_decision"].isin(FINAL_DIAGNOSTIC)
    out["workflow_nonfinal_action"] = out["workflow_decision"].isin(NONFINAL_ACTION)
    out["workflow_repeat_or_referral"] = out["workflow_decision"].isin(NONFINAL_ACTION | {"BORDERLINE_REPEAT"})
    out["workflow_predicted_obstructed_when_final"] = out["workflow_decision"].isin(
        {"ACCEPT_OBSTRUCTED", "TRUNCATED_RECOVERED"}
    )
    out["workflow_false_negative"] = out["truth_obstructed_fixed"] & out["workflow_accepted_normal"]
    out["workflow_false_positive_final"] = (
        out["truth_normal_fixed"] & out["workflow_decision"].isin({"ACCEPT_OBSTRUCTED", "TRUNCATED_RECOVERED"})
    )
    out["workflow_normal_burden"] = out["truth_normal_fixed"] & ~out["workflow_decision"].eq("ACCEPT_NORMAL")
    return out


def decision_counts(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    total = len(df)
    counts = df["workflow_decision"].value_counts().to_dict()
    for decision in DECISION_ORDER:
        n = int(counts.get(decision, 0))
        rows.append({"decision": decision, "n": n, "fraction": float(n / total) if total else np.nan})
    return rows


def confusion_for_final(df: pd.DataFrame) -> dict[str, Any]:
    final = df[df["workflow_final_diagnostic"]].copy()
    truth = final["truth_obstructed_fixed"].astype(bool)
    pred = final["workflow_predicted_obstructed_when_final"].astype(bool)
    tp = int((truth & pred).sum())
    tn = int((~truth & ~pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    n = tp + tn + fp + fn
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "agreement": float((tp + tn) / n) if n else np.nan,
        "cohens_kappa": kappa(tp, tn, fp, fn),
    }


def scope_metrics(df: pd.DataFrame, scope: str) -> dict[str, Any]:
    out: dict[str, Any] = {"scope": scope, "n": int(len(df))}
    obs = df[df["truth_obstructed_fixed"]]
    normals = df[df["truth_normal_fixed"]]
    severe = df[df["truth_severity"].eq("severe")]
    moderate = df[df["truth_severity"].eq("moderate")]
    borderline = df[df["truth_severity"].eq("mild_0.60_0.70")]
    final = df[df["workflow_final_diagnostic"]]
    nonfinal = df[df["workflow_nonfinal_action"]]

    rates = [
        rate_record("accepted_normal_fn_all_obstructed", int(obs["workflow_accepted_normal"].sum()), len(obs)),
        rate_record("accepted_normal_fn_severe", int(severe["workflow_accepted_normal"].sum()), len(severe)),
        rate_record("accepted_normal_fn_moderate", int(moderate["workflow_accepted_normal"].sum()), len(moderate)),
        rate_record("accepted_normal_fn_borderline", int(borderline["workflow_accepted_normal"].sum()), len(borderline)),
        rate_record("normal_non_accept_burden", int(normals["workflow_normal_burden"].sum()), len(normals)),
        rate_record("final_diagnostic_call_rate", len(final), len(df)),
        rate_record("repeat_referral_unusable_rate", len(nonfinal), len(df)),
    ]
    out["rates"] = {item["metric"]: item for item in rates}
    out["decision_counts"] = decision_counts(df)
    out["final_confusion_fixed_0p70"] = confusion_for_final(df)
    out["severity_decision_counts"] = (
        df.groupby(["truth_severity", "workflow_decision"], dropna=False)
        .size()
        .reset_index(name="n")
        .to_dict("records")
    )
    return out


def bootstrap_rate(
    df: pd.DataFrame,
    numerator_col: str,
    denominator_mask: pd.Series,
    seed: int,
    n_boot: int = BOOTSTRAP_N,
) -> dict[str, Any]:
    denom_df = df[denominator_mask].copy()
    if denom_df.empty:
        return {"n": 0, "point": np.nan, "ci_low": np.nan, "ci_high": np.nan, "bootstrap_n": n_boot}
    rng = np.random.default_rng(seed)
    vals = denom_df[numerator_col].astype(bool).to_numpy()
    point = float(vals.mean())
    boots = []
    n = len(vals)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float(vals[idx].mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": int(n),
        "point": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "bootstrap_n": n_boot,
    }


def heldout_bootstrap_summary(test: pd.DataFrame) -> dict[str, Any]:
    return {
        "accepted_normal_fn_all_obstructed": bootstrap_rate(
            test,
            "workflow_accepted_normal",
            test["truth_obstructed_fixed"],
            RANDOM_SEED + 1,
        ),
        "accepted_normal_fn_borderline": bootstrap_rate(
            test,
            "workflow_accepted_normal",
            test["truth_severity"].eq("mild_0.60_0.70"),
            RANDOM_SEED + 2,
        ),
        "normal_non_accept_burden": bootstrap_rate(
            test,
            "workflow_normal_burden",
            test["truth_normal_fixed"],
            RANDOM_SEED + 3,
        ),
        "final_diagnostic_call_rate": bootstrap_rate(
            test.assign(final=test["workflow_final_diagnostic"]),
            "final",
            pd.Series(True, index=test.index),
            RANDOM_SEED + 4,
        ),
    }


def stopping_step_decision(ratios: list[float], maxed: bool = False) -> tuple[bool, str, str]:
    ratios = [float(x) for x in ratios if np.isfinite(float(x))]
    if not ratios:
        return True, "UNUSABLE", "no finite corrected ratios"
    k = len(ratios)
    center = float(np.median(ratios))
    spread = float(np.max(ratios) - np.min(ratios)) if k > 1 else 0.0
    if k == 1:
        if center < 0.50:
            return True, "REFER_SEVERE_OR_UNSTABLE", "single maneuver severe-range ratio"
        if center >= 0.80:
            return True, "ACCEPT_NORMAL", "single maneuver far above threshold"
        return False, "BORDERLINE_REPEAT", "single maneuver not far enough from threshold"

    all_obstructed = all(r < 0.70 for r in ratios)
    all_normal = all(r >= 0.70 for r in ratios)
    stable = spread <= 0.030
    if center < 0.50:
        return True, "REFER_SEVERE_OR_UNSTABLE", "severe-range median after repeated maneuvers"
    if stable and all_obstructed and center < 0.60:
        return True, "ACCEPT_OBSTRUCTED", "stable repeated moderate-range obstruction"
    if stable and all_normal and center >= 0.72:
        return True, "ACCEPT_NORMAL", "stable repeated normal-range result"
    if maxed:
        if 0.60 <= center < 0.72 or not (all_obstructed or all_normal):
            return True, "BORDERLINE_REPEAT", "max maneuver budget reached with borderline or split classification"
        if center < 0.70:
            return True, "ACCEPT_OBSTRUCTED", "max maneuver budget reached with stable obstructed side"
        return True, "ACCEPT_NORMAL", "max maneuver budget reached with stable normal side"
    return False, "BORDERLINE_REPEAT", "additional maneuver expected to reduce decision uncertainty"


def stopping_simulation(repro: pd.DataFrame, c1_offset: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    maneuvers = repro[repro["row_type"].eq("maneuver_fit")].copy()
    maneuvers = maneuvers[
        maneuvers["model_ratio"].map(finite)
        & maneuvers["fev1_fvc_ratio"].map(finite)
        & maneuvers["SPAMANU"].map(finite)
    ].copy()
    maneuvers["corrected_ratio_c1"] = maneuvers["model_ratio"].astype(float) - c1_offset
    rows = []
    for seqn, group in maneuvers.groupby("SEQN"):
        group = group.sort_values(["SPAMANU", "raw_index"]).copy()
        max_available = min(3, len(group))
        if max_available == 0:
            continue
        seen: list[float] = []
        final_decision = "UNUSABLE"
        stop_k = max_available
        stop_reason = "no usable maneuver"
        for idx, ratio in enumerate(group["corrected_ratio_c1"].iloc[:max_available], start=1):
            seen.append(float(ratio))
            stop, decision, reason = stopping_step_decision(seen, maxed=(idx == max_available))
            if stop:
                final_decision = decision
                stop_k = idx
                stop_reason = reason
                break
        truth_ratio = float(group["fev1_fvc_ratio"].iloc[0])
        truth_severity = str(group["truth_severity"].iloc[0])
        rows.append(
            {
                "SEQN": float(seqn),
                "available_maneuvers": int(len(group)),
                "max_simulated_maneuvers": int(max_available),
                "stop_k": int(stop_k),
                "stopping_decision": final_decision,
                "stopping_reason": stop_reason,
                "truth_ratio": truth_ratio,
                "truth_severity": truth_severity,
                "truth_obstructed_fixed": bool(truth_ratio < 0.70),
                "accepted_normal": bool(final_decision == "ACCEPT_NORMAL"),
                "normal_burden": bool(truth_ratio >= 0.70 and final_decision != "ACCEPT_NORMAL"),
                "maneuvers_saved_vs_3": int(max(0, min(3, len(group)) - stop_k)),
            }
        )
    sim = pd.DataFrame(rows)
    eligible_3 = sim[sim["available_maneuvers"] >= 3].copy()
    summary = {
        "method": "deterministic stopping proxy, not Bayesian EIG",
        "participants_simulated": int(len(sim)),
        "participants_with_at_least_3_maneuvers": int(len(eligible_3)),
        "mean_stop_k_all": float(sim["stop_k"].mean()) if len(sim) else np.nan,
        "mean_stop_k_eligible_3": float(eligible_3["stop_k"].mean()) if len(eligible_3) else np.nan,
        "mean_maneuvers_saved_vs_3_eligible_3": float(eligible_3["maneuvers_saved_vs_3"].mean()) if len(eligible_3) else np.nan,
        "decision_counts_all": sim["stopping_decision"].value_counts().to_dict() if len(sim) else {},
        "accepted_normal_fn_all_obstructed": rate_record(
            "stopping_accepted_normal_fn_all_obstructed",
            int((sim["truth_obstructed_fixed"] & sim["accepted_normal"]).sum()) if len(sim) else 0,
            int(sim["truth_obstructed_fixed"].sum()) if len(sim) else 0,
        ),
        "accepted_normal_fn_severe": rate_record(
            "stopping_accepted_normal_fn_severe",
            int((sim["truth_severity"].eq("severe") & sim["accepted_normal"]).sum()) if len(sim) else 0,
            int(sim["truth_severity"].eq("severe").sum()) if len(sim) else 0,
        ),
        "normal_non_accept_burden": rate_record(
            "stopping_normal_non_accept_burden",
            int(sim["normal_burden"].sum()) if len(sim) else 0,
            int((~sim["truth_obstructed_fixed"]).sum()) if len(sim) else 0,
        ),
    }
    return sim, summary


def outcome_verdict(test_metrics: dict[str, Any], stopping: dict[str, Any]) -> tuple[str, str]:
    severe = test_metrics["rates"]["accepted_normal_fn_severe"]
    borderline = test_metrics["rates"]["accepted_normal_fn_borderline"]
    normal_burden = test_metrics["rates"]["normal_non_accept_burden"]
    final_call = test_metrics["rates"]["final_diagnostic_call_rate"]
    if severe["numerator"] == 0 and borderline["rate"] <= 0.10 and normal_burden["rate"] <= 0.35:
        return (
            "WORKFLOW-RECOVERED",
            "held-out severe accepted-normal FN is zero, borderline accepted-normal FN is <=0.10, and normal non-accept burden is <=0.35.",
        )
    if severe["numerator"] == 0 and borderline["rate"] <= 0.20 and final_call["rate"] >= 0.50:
        return (
            "WORKFLOW-PARTIAL",
            "held-out severe accepted-normal FN is zero and the workflow resolves most patients, but borderline FN or normal burden remains nontrivial.",
        )
    return (
        "WORKFLOW-WEAK",
        "the deterministic workflow either misses severe/borderline obstruction or pushes too many cases into repeat/referral.",
    )


def rate_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for metric, rec in metrics["rates"].items():
        rows.append(
            {
                "metric": metric,
                "rate": rec["rate"],
                "numerator": rec["numerator"],
                "denominator": rec["denominator"],
                "ci": f"{fmt(rec['ci_low'], 3)}-{fmt(rec['ci_high'], 3)}",
            }
        )
    return rows


def write_outputs(
    root: Path,
    result: dict[str, Any],
    workflow_rows: pd.DataFrame,
    stopping_rows: pd.DataFrame,
) -> None:
    out = root / "results"
    out.mkdir(exist_ok=True)
    stem = out / OUTPUT_STEM
    workflow_rows.to_csv(stem.with_name(f"{OUTPUT_STEM}_rows.csv"), index=False)
    stopping_rows.to_csv(stem.with_name(f"{OUTPUT_STEM}_stopping_rows.csv"), index=False)
    stem.with_suffix(".json").write_text(json.dumps(clean_json(result), indent=2) + "\n", encoding="utf-8")

    test_metrics = result["workflow_metrics"]["heldout_test"]
    all_metrics = result["workflow_metrics"]["all_policy_rows"]
    final = test_metrics["final_confusion_fixed_0p70"]
    stopping = result["stopping_simulation"]
    lines = [
        "# Gate - Deterministic Workflow Policy, Cycle F",
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
            "## Inputs",
            "",
            md_table(
                result["inputs"],
                [
                    ("artifact", "artifact", "str"),
                    ("rows", "rows", "int"),
                    ("role", "role", "str"),
                ],
            ),
            "",
            "## Workflow Rule",
            "",
            "| priority | condition | output |",
            "|---:|---|---|",
            "| 1 | missing/nonfinite fit or ratio | UNUSABLE |",
            "| 2 | corrected ratio <0.50 | REFER_SEVERE_OR_UNSTABLE |",
            "| 3 | near threshold and repeated-maneuver instability or ratio SD >=0.030 | REFER_SEVERE_OR_UNSTABLE |",
            "| 4 | ATS truncation flag and corrected ratio <0.70 | TRUNCATED_RECOVERED |",
            "| 5 | ATS truncation flag with non-obstructed corrected ratio | BORDERLINE_REPEAT |",
            "| 6 | raw/corrected ratio crosses 0.70, or corrected ratio in [0.60,0.72) | BORDERLINE_REPEAT |",
            "| 7 | corrected ratio <0.60 | ACCEPT_OBSTRUCTED |",
            "| 8 | otherwise | ACCEPT_NORMAL |",
            "",
            "## Held-Out TEST Workflow Rates",
            "",
            md_table(
                rate_rows(test_metrics),
                [
                    ("metric", "metric", "str"),
                    ("rate", "rate", "float3"),
                    ("numerator", "num", "int"),
                    ("denominator", "den", "int"),
                    ("ci", "exact 95% CI", "str"),
                ],
            ),
            "",
            "Held-out bootstrap checks:",
            "",
            md_table(
                [
                    {"metric": k, **v}
                    for k, v in result["heldout_bootstrap"].items()
                ],
                [
                    ("metric", "metric", "str"),
                    ("point", "point", "float3"),
                    ("ci_low", "boot low", "float3"),
                    ("ci_high", "boot high", "float3"),
                    ("n", "n", "int"),
                    ("bootstrap_n", "B", "int"),
                ],
            ),
            "",
            "## Held-Out Decision Distribution",
            "",
            md_table(
                test_metrics["decision_counts"],
                [
                    ("decision", "decision", "str"),
                    ("n", "n", "int"),
                    ("fraction", "fraction", "float3"),
                ],
            ),
            "",
            "## Final Diagnostic Calls Only",
            "",
            "This excludes `BORDERLINE_REPEAT`, `REFER_SEVERE_OR_UNSTABLE`, and `UNUSABLE`; those are safe workflow actions, not final normal/obstructed diagnoses.",
            "",
            md_table(
                [
                    {
                        "n": final["n"],
                        "tp": final["tp"],
                        "tn": final["tn"],
                        "fp": final["fp"],
                        "fn": final["fn"],
                        "sensitivity": final["sensitivity"],
                        "specificity": final["specificity"],
                        "kappa": final["cohens_kappa"],
                    }
                ],
                [
                    ("n", "n", "int"),
                    ("tp", "TP", "int"),
                    ("tn", "TN", "int"),
                    ("fp", "FP", "int"),
                    ("fn", "FN", "int"),
                    ("sensitivity", "sens", "float3"),
                    ("specificity", "spec", "float3"),
                    ("kappa", "kappa", "float3"),
                ],
            ),
            "",
            "## All Policy Rows",
            "",
            md_table(
                rate_rows(all_metrics),
                [
                    ("metric", "metric", "str"),
                    ("rate", "rate", "float3"),
                    ("numerator", "num", "int"),
                    ("denominator", "den", "int"),
                    ("ci", "exact 95% CI", "str"),
                ],
            ),
            "",
            "## Deterministic Stopping Proxy",
            "",
            "This is not Bayesian EIG. It is a deterministic value-of-additional-maneuver proxy using first/second/third maneuver stability from existing volume-domain fits.",
            "",
            md_table(
                [
                    {
                        "participants": stopping["participants_simulated"],
                        "eligible_3": stopping["participants_with_at_least_3_maneuvers"],
                        "mean_stop_k": stopping["mean_stop_k_all"],
                        "mean_stop_k_3": stopping["mean_stop_k_eligible_3"],
                        "saved": stopping["mean_maneuvers_saved_vs_3_eligible_3"],
                    }
                ],
                [
                    ("participants", "participants", "int"),
                    ("eligible_3", ">=3 maneuvers", "int"),
                    ("mean_stop_k", "mean stop k", "float2"),
                    ("mean_stop_k_3", "mean stop k, >=3", "float2"),
                    ("saved", "saved vs 3", "float2"),
                ],
            ),
            "",
            md_table(
                [
                    stopping["accepted_normal_fn_all_obstructed"],
                    stopping["accepted_normal_fn_severe"],
                    stopping["normal_non_accept_burden"],
                ],
                [
                    ("metric", "metric", "str"),
                    ("rate", "rate", "float3"),
                    ("numerator", "num", "int"),
                    ("denominator", "den", "int"),
                    ("ci_low", "CI low", "float3"),
                    ("ci_high", "CI high", "float3"),
                ],
            ),
            "",
            f"Stopping decision counts: {stopping['decision_counts_all']}",
            "",
            "## Proposal Recovery Statement",
            "",
            "- Bayesian posterior uncertainty remains blocked; this gate makes no posterior claim.",
            "- The triage component is now explicit as a deterministic policy over validated OMBE, ATS truncation, and repeatability signals.",
            "- The stopping component is attemptable and now has a deterministic proxy audit; it should be framed as value-of-additional-maneuver stability, not full Bayesian expected information gain.",
            "",
            f"VERDICT: {result['verdict']} - {result['verdict_reason']}",
            "",
        ]
    )
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def run(root: Path) -> dict[str, Any]:
    policy, repro = load_inputs(root)
    c1 = correction_offset(policy)
    workflow = add_workflow_columns(add_repeatability(policy, repro))
    test = workflow[workflow["split"].eq("test")].copy()
    all_metrics = scope_metrics(workflow, "all_policy_rows")
    test_metrics = scope_metrics(test, "heldout_test")
    stopping_rows, stopping_summary = stopping_simulation(repro, c1)
    boot = heldout_bootstrap_summary(test)
    verdict, reason = outcome_verdict(test_metrics, stopping_summary)
    result = {
        "verdict": verdict,
        "verdict_reason": reason,
        "seed": RANDOM_SEED,
        "bootstrap_n": BOOTSTRAP_N,
        "c1_offset_from_existing_policy": c1,
        "inputs": [
            {
                "artifact": "results/gate_policy_calibrated_flag_F_rows.csv",
                "rows": int(len(policy)),
                "role": "bias-corrected selected-maneuver policy sample",
            },
            {
                "artifact": "results/gate_reproducibility_volume_F_rows.csv",
                "rows": int(len(repro)),
                "role": "cross-maneuver repeatability and stopping proxy",
            },
        ],
        "protocol_decisions": [
            "Cycle F is the first workflow target because the existing calibrated policy artifacts are strongest there.",
            "The headline evaluation uses held-out TEST rows from the existing policy gate; all rows are reported secondarily.",
            "No NUTS, SNPE, posterior draws, refits, threshold movement, or dashboard code is used.",
            "Only ACCEPT_NORMAL is counted as a false-negative workflow failure for obstructed truth; repeat/referral/unusable are safe non-final actions.",
            "Predicted severe-range corrected ratio <0.50 is routed to referral instead of accepted as a routine obstructed call.",
            "Borderline corrected ratios in [0.60,0.72), raw/corrected threshold disagreement, and ATS-truncated normal calls are routed to repeat.",
            "Repeated-maneuver instability near the decision boundary routes to referral/unstable rather than a confident diagnosis.",
            "The stopping audit is explicitly a deterministic stability proxy, not Bayesian expected information gain.",
        ],
        "workflow_metrics": {
            "heldout_test": test_metrics,
            "all_policy_rows": all_metrics,
        },
        "heldout_bootstrap": boot,
        "stopping_simulation": stopping_summary,
    }
    write_outputs(root, result, workflow, stopping_rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(Path(args.root).resolve())
    test = result["workflow_metrics"]["heldout_test"]["rates"]
    print(f"Workflow policy gate: {result['verdict']} - {result['verdict_reason']}")
    print(
        "Held-out severe accepted-normal FN: "
        f"{test['accepted_normal_fn_severe']['numerator']}/{test['accepted_normal_fn_severe']['denominator']}"
    )
    print(
        "Held-out borderline accepted-normal FN: "
        f"{test['accepted_normal_fn_borderline']['rate']:.3f}"
    )
    print(
        "Held-out normal non-accept burden: "
        f"{test['normal_non_accept_burden']['rate']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
