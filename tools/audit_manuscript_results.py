#!/usr/bin/env python3
"""Recompute endpoint-error and lower-bound diagnostics used in the manuscript."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "gate_truncation_recovery_F_rows.csv"
OUTPUT = ROOT / "results" / "manuscript_endpoint_audit.json"


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def main() -> None:
    with SOURCE.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["estimator"] == "E1_OMBE" and row["fit_success"] == "True"
        ]

    endpoint_summary: list[dict[str, float | int | str]] = []
    severity_summary: list[dict[str, float | int | str]] = []
    for endpoint in ("2", "3", "4", "5"):
        subset = [row for row in rows if row["T_label"] == endpoint]
        signed_error_ml = [
            (float(row["est_fvc_l"]) - float(row["true_fvc_l"])) * 1000
            for row in subset
        ]
        absolute_error_ml = [abs(value) for value in signed_error_ml]
        floor_rows = [row for row in subset if abs(float(row["est_fvc_l"]) - 2.0) < 1e-6]
        endpoint_summary.append(
            {
                "endpoint_s": int(endpoint),
                "n": len(subset),
                "median_signed_error_ml": round(statistics.median(signed_error_ml), 1),
                "median_absolute_error_ml": round(statistics.median(absolute_error_ml), 1),
                "p95_absolute_error_ml": round(percentile(absolute_error_ml, 0.95), 1),
                "fvc_floor_hits": len(floor_rows),
                "reference_fvc_below_2l": sum(float(row["true_fvc_l"]) < 2.0 for row in subset),
                "floor_hits_with_reference_below_2l": sum(
                    float(row["true_fvc_l"]) < 2.0 for row in floor_rows
                ),
            }
        )

        if endpoint in {"2", "3"}:
            for severity in ("severe", "moderate", "mild_0.60_0.70", "normal"):
                group = [row for row in subset if row["truth_severity"] == severity]
                severity_summary.append(
                    {
                        "endpoint_s": int(endpoint),
                        "severity": severity,
                        "n": len(group),
                        "false_negatives": sum(
                            row["truth_obstructed_fixed"] == "True"
                            and row["est_obstructed_fixed"] == "False"
                            for row in group
                        ),
                        "false_positives": sum(
                            row["truth_obstructed_fixed"] == "False"
                            and row["est_obstructed_fixed"] == "True"
                            for row in group
                        ),
                        "fvc_floor_hits": sum(
                            abs(float(row["est_fvc_l"]) - 2.0) < 1e-6 for row in group
                        ),
                        "reference_fvc_below_2l": sum(
                            float(row["true_fvc_l"]) < 2.0 for row in group
                        ),
                    }
                )

    payload = {
        "source": str(SOURCE.relative_to(ROOT)),
        "estimator": "E1_OMBE",
        "cycle": "F",
        "endpoint_summary": endpoint_summary,
        "severity_summary": severity_summary,
        "severity_labels": dict(Counter(row["truth_severity"] for row in rows if row["T_label"] == "2")),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
