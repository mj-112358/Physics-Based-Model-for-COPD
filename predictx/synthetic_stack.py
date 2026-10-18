from __future__ import annotations

from pathlib import Path

from .fit import validate_fits, write_fit_report
from .ombe import OmbeGrid, flow_integral_l, midrange_phi, unpack_phi
from .segment import validate_segmentation, write_segmentation_report
from .tsnpe import run_tsnpe


def assert_ombe_volume_conservation() -> None:
    phi = midrange_phi()
    t = OmbeGrid().t
    integral = float(flow_integral_l(phi, t))
    fvc = float(unpack_phi(phi)["fvc_l"])
    rel = abs(integral - fvc) / fvc
    if rel > 0.01:
        raise RuntimeError(f"OMBE volume conservation failed: integral={integral}, fvc={fvc}, rel={rel}")


def main() -> int:
    results_dir = Path("results")
    figures_dir = Path("figures")
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    assert_ombe_volume_conservation()

    mod1 = validate_segmentation()
    write_segmentation_report(mod1, results_dir, figures_dir)
    if mod1["status"] != "PASS":
        write_summary(results_dir, mod1, None, None)
        print("Module 1 failed; stopping before Modules 2 and 3.")
        return 1

    mod2 = validate_fits()
    write_fit_report(mod2, results_dir, figures_dir)
    mod3 = run_tsnpe(results_dir, figures_dir)
    write_summary(results_dir, mod1, mod2, mod3)
    print(f"Module 1 {mod1['status']}; Module 2 {mod2['status']}; Module 3 {mod3['status']}")
    return 0


def write_summary(results_dir: Path, mod1: dict, mod2: dict | None, mod3: dict | None) -> None:
    lines = [
        "# Synthetic Stack Summary",
        "",
        f"- Module 1 segmentation: {mod1['status']}",
        f"  - t0 within 2 samples: {mod1['t0_within_2_samples']}/{mod1['n']} ({mod1['t0_within_2_samples_pct']:.1f}%)",
        f"  - FEV1 within 2%: {mod1['fev1_within_2_pct']}/{mod1['n']} ({mod1['fev1_within_2_pct_pct']:.1f}%)",
    ]
    if mod2 is not None:
        lines.extend(
            [
                f"- Module 2 least-squares fitter: {mod2['status']}",
                f"  - clean adequate: {mod2['adequate_clean']}/{mod2['n']} ({mod2['adequate_clean_pct']:.1f}%)",
                f"  - corrupt flagged: {mod2['corrupt_flagged']}/{mod2['corrupt_n']} ({mod2['corrupt_flagged_pct']:.1f}%)",
                f"  - median NRSS: {mod2['median_nrss']:.3f}",
            ]
        )
    else:
        lines.append("- Module 2 least-squares fitter: NOT-REACHED")
    if mod3 is not None:
        lines.extend(
            [
                f"- Module 3 TSNPE: {mod3['status']}",
                f"  - reason: {mod3['reason']}",
            ]
        )
    else:
        lines.append("- Module 3 TSNPE: NOT-REACHED")
    lines.append("")
    (results_dir / "SYNTHETIC_STACK_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
