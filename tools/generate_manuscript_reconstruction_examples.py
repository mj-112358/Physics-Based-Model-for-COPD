#!/usr/bin/env python3
"""Generate representative real-NHANES reconstruction panels for the paper.

Within each prespecified severity-duration stratum, the displayed participant is
the correctly classified, non-bound-active fit whose absolute FVC percentage
error is closest to the stratum median.  This avoids selecting unusually strong
examples while keeping the figure reproducible.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/predictx-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/predictx-xdg-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predictx.gate23_f import find_or_unzip_raw, read_raw
from predictx.gate3_diagnosis import np_ombe_volume
from predictx.gate_clinical_classification_f import extract_curve_for_fit
from predictx.gate_truncation_recovery_f import fit_ombe_truncated, model_grid_with_stop


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "results" / "gate_truncation_recovery_F_rows.csv"
OUTPUT = ROOT / "paper_assets" / "real_reconstruction_examples.png"
AUDIT = ROOT / "results" / "manuscript_reconstruction_examples.csv"

STRATA = (
    ("severe", "2", "A", "Severe obstruction", 2.0),
    ("moderate", "3", "B", "Moderate obstruction", 3.0),
    ("mild_0.60_0.70", "4", "C", "Mild obstruction", 4.0),
)


def choose_examples(rows: pd.DataFrame) -> pd.DataFrame:
    eligible = rows[
        rows["estimator"].eq("E1_OMBE")
        & rows["fit_success"].astype(bool)
        & rows["est_obstructed_fixed"].eq(rows["truth_obstructed_fixed"])
        & rows["est_fvc_l"].between(2.05, 5.95)
    ].copy()

    selected = []
    for severity, t_label, panel, display, stop_s in STRATA:
        group = eligible[
            eligible["truth_severity"].eq(severity)
            & eligible["T_label"].astype(str).eq(t_label)
        ].copy()
        median_error = float(group["fvc_abs_pct_error"].median())
        group["distance_to_median_error"] = (
            group["fvc_abs_pct_error"] - median_error
        ).abs()
        row = group.sort_values(
            ["distance_to_median_error", "SEQN", "raw_index"]
        ).iloc[0].copy()
        row["panel"] = panel
        row["display_severity"] = display
        row["stop_s"] = stop_s
        row["stratum_median_abs_fvc_pct_error"] = median_error
        selected.append(row)
    return pd.DataFrame(selected)


def main() -> None:
    rows = pd.read_csv(ROWS)
    selected = choose_examples(rows)
    raw_path, _note = find_or_unzip_raw(ROOT, "F")
    raw, _info = read_raw(raw_path, "F")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.7,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25), constrained_layout=True)
    audit_rows: list[dict[str, object]] = []

    for ax, (_, row) in zip(axes, selected.iterrows()):
        curve = extract_curve_for_fit(raw.loc[int(row["raw_index"])])
        full_t = np.asarray(curve["vol_t"], dtype=float)
        full_y = np.asarray(curve["vol_l"], dtype=float)
        stop_s = float(row["stop_s"])
        fit = fit_ombe_truncated(full_t, full_y, stop_s)
        grid, fit_idx = model_grid_with_stop(full_t, stop_s)
        recovered = np_ombe_volume(np.asarray(fit["params"], dtype=float), grid)
        prefix_y = np.interp(grid[fit_idx], full_t, full_y)

        ax.plot(full_t, full_y, color="#7F7F7F", lw=1.8, label="Complete reference")
        ax.plot(grid, recovered, color="#1F4E79", lw=2.1, label="Recovered trajectory")
        ax.plot(grid[fit_idx], prefix_y, color="#ED7D31", lw=2.8, label="Observed prefix")
        ax.axvline(stop_s, color="#BFBFBF", lw=1.0, ls="--")
        ax.scatter([stop_s], [prefix_y[-1]], color="#ED7D31", s=18, zorder=5)
        ax.set_xlim(0.0, min(float(full_t[-1]), 12.0))
        ax.set_ylim(0.0, max(float(full_y.max()), float(recovered.max())) * 1.08)
        ax.grid(axis="y", color="#E6E6E6", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Time from corrected onset (s)")
        if ax is axes[0]:
            ax.set_ylabel("Expired volume (L)")
        ax.set_title(
            f"{row['panel']}  {row['display_severity']}  |  {stop_s:.0f}-s prefix",
            loc="left",
            fontweight="bold",
        )
        ax.text(
            0.98,
            0.08,
            f"Reference FVC {row['true_fvc_l']:.2f} L\nRecovered FVC {fit['fvc_l']:.2f} L\n"
            f"Ratio {row['true_ratio']:.3f} -> {row['true_fev1_l']/fit['fvc_l']:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#D9D9D9", "alpha": 0.92, "pad": 3},
        )

        audit_rows.append(
            {
                "panel": row["panel"],
                "cycle": "F",
                "SEQN": int(row["SEQN"]),
                "raw_index": int(row["raw_index"]),
                "severity": row["truth_severity"],
                "stop_s": stop_s,
                "reference_fvc_l": float(row["true_fvc_l"]),
                "recovered_fvc_l": float(fit["fvc_l"]),
                "reference_ratio": float(row["true_ratio"]),
                "recovered_ratio": float(row["true_fev1_l"] / fit["fvc_l"]),
                "stratum_median_abs_fvc_pct_error": float(
                    row["stratum_median_abs_fvc_pct_error"]
                ),
                "selection_rule": "closest to stratum median absolute FVC percent error among correctly classified non-bound-active fits",
            }
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.035))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    pd.DataFrame(audit_rows).to_csv(AUDIT, index=False)
    print(OUTPUT)
    print(AUDIT)


if __name__ == "__main__":
    main()
