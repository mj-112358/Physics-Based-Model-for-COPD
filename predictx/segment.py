from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class SyntheticRecording:
    seq: int
    phi: np.ndarray
    time_s: np.ndarray
    flow_lps: np.ndarray
    volume_ml: np.ndarray
    true_onset_idx: int
    true_t0_s: float
    true_fev1_ml: float
    true_fvc_ml: float
    true_exhale_flow_lps: np.ndarray
    true_exhale_time_s: np.ndarray


def moving_average(x: np.ndarray, width: int = 5) -> np.ndarray:
    if width <= 1:
        return x.copy()
    kernel = np.ones(width) / width
    return np.convolve(x, kernel, mode="same")


def clean_ats_zeroed_curve(phi: np.ndarray, hz: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import jax.numpy as jnp

    from .ombe import OmbeGrid, ombe_flow, ombe_volume

    grid = OmbeGrid(hz=hz)
    t = np.asarray(grid.t)
    flow = np.asarray(ombe_flow(jnp.asarray(phi), jnp.asarray(t)))
    volume_l = np.asarray(ombe_volume(jnp.asarray(phi), jnp.asarray(t)))
    pef_idx = int(np.argmax(flow))
    offset = float(t[pef_idx] - volume_l[pef_idx] / flow[pef_idx])
    flow_zeroed = flow
    volume_zeroed_l = volume_l
    t_zeroed = t
    for _ in range(4):
        t_zeroed = np.arange(0.0, t[-1] - offset, 1.0 / hz)
        source_t = t_zeroed + offset
        flow_zeroed = np.interp(source_t, t, flow)
        volume_zeroed_l = np.interp(source_t, t, volume_l) - np.interp(offset, t, volume_l)
        clean_pef_idx = int(np.argmax(flow_zeroed))
        discrete_t0 = t_zeroed[clean_pef_idx] - volume_zeroed_l[clean_pef_idx] / flow_zeroed[clean_pef_idx]
        if abs(discrete_t0) < 1e-5:
            break
        offset += float(discrete_t0)
    return t_zeroed, flow_zeroed, volume_zeroed_l * 1000.0


def synthesize_recording(
    phi: np.ndarray,
    rng: np.random.Generator,
    seq: int,
    hz: int = 100,
    noise_lps: float = 0.03,
) -> SyntheticRecording:
    t_exhale, clean_flow, clean_volume_ml = clean_ats_zeroed_curve(phi, hz=hz)
    baseline_n = int(rng.integers(10, 51))
    inhale_n = int(rng.integers(60, 151))
    baseline = rng.normal(0.0, noise_lps, size=baseline_n)
    exhale = clean_flow + rng.normal(0.0, noise_lps, size=len(clean_flow))
    phase = np.linspace(0.0, np.pi, inhale_n)
    inhale_amp = rng.uniform(0.6, 1.8)
    inhale = -inhale_amp * np.sin(phase) + rng.normal(0.0, noise_lps, size=inhale_n)
    flow = np.concatenate([baseline, exhale, inhale])
    time_s = np.arange(len(flow), dtype=float) / hz
    volume_ml = np.cumsum(flow * (1000.0 / hz))
    true_fev1_ml = float(np.interp(1.0, t_exhale, clean_volume_ml))
    true_fvc_ml = float(clean_volume_ml[-1])
    return SyntheticRecording(
        seq=seq,
        phi=phi,
        time_s=time_s,
        flow_lps=flow,
        volume_ml=volume_ml,
        true_onset_idx=baseline_n,
        true_t0_s=baseline_n / hz,
        true_fev1_ml=true_fev1_ml,
        true_fvc_ml=true_fvc_ml,
        true_exhale_flow_lps=clean_flow,
        true_exhale_time_s=t_exhale,
    )


def generate_synthetic_recordings(n: int = 200, seed: int = 0) -> list[SyntheticRecording]:
    from .ombe import sample_prior_phi

    phis = sample_prior_phi(n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    return [synthesize_recording(phi, rng, seq=i) for i, phi in enumerate(phis)]


def detect_forced_exhale_segment(flow_lps: np.ndarray, hz: int = 100) -> tuple[int, int, int]:
    smooth = moving_average(flow_lps, width=5)
    pef_idx = int(np.argmax(smooth))
    pef = float(smooth[pef_idx])
    baseline = smooth[: min(10, max(5, pef_idx))]
    baseline_med = float(np.median(baseline))
    sigma = 1.4826 * float(np.median(np.abs(baseline - baseline_med)))
    threshold = max(0.05, baseline_med + 5.0 * sigma, 0.02 * pef)
    onset_idx = 0
    for idx in range(0, max(1, pef_idx - 3)):
        window = smooth[idx : idx + 5]
        if len(window) == 5 and np.mean(window > threshold) >= 0.8:
            onset_idx = idx
            break

    end_idx = len(flow_lps) - 1
    post = smooth[pef_idx:]
    neg_threshold = min(-0.20, -0.10 * pef)
    for offset in range(5, len(post) - 5):
        window = post[offset : offset + 5]
        if np.mean(window < neg_threshold) >= 0.8:
            end_idx = max(pef_idx + offset - 1, pef_idx)
            break
    return onset_idx, pef_idx, end_idx


def backextrap_t0(
    time_s: np.ndarray,
    flow_lps: np.ndarray,
    volume_ml: np.ndarray,
    onset_idx: int,
    pef_idx: int,
) -> float:
    pef_ml_s = flow_lps[pef_idx] * 1000.0
    if pef_ml_s <= 0:
        raise ValueError("PEF must be positive for back-extrapolation.")
    baseline_volume = volume_ml[onset_idx]
    return float(time_s[pef_idx] - (volume_ml[pef_idx] - baseline_volume) / pef_ml_s)


def derive_fev1_fvc(
    time_s: np.ndarray,
    volume_ml: np.ndarray,
    t0_s: float,
    end_idx: int,
) -> tuple[float, float]:
    v0 = float(np.interp(t0_s, time_s, volume_ml))
    v1 = float(np.interp(t0_s + 1.0, time_s, volume_ml))
    vend = float(volume_ml[end_idx])
    return v1 - v0, vend - v0


def baseline_correct_flow_volume(flow_lps: np.ndarray, onset_idx: int, hz: int = 100) -> tuple[np.ndarray, np.ndarray]:
    baseline_window = flow_lps[: max(1, onset_idx)]
    baseline_flow = float(np.median(baseline_window)) if len(baseline_window) else 0.0
    corrected_flow = flow_lps - baseline_flow
    corrected_volume_ml = np.cumsum(corrected_flow * (1000.0 / hz))
    return corrected_flow, corrected_volume_ml


def validate_segmentation(n: int = 200, seed: int = 0) -> dict:
    records = generate_synthetic_recordings(n=n, seed=seed)
    rows = []
    for record in records:
        onset_idx, pef_idx, end_idx = detect_forced_exhale_segment(record.flow_lps)
        corrected_flow, corrected_volume = baseline_correct_flow_volume(record.flow_lps, onset_idx)
        t0 = backextrap_t0(record.time_s, corrected_flow, corrected_volume, onset_idx, pef_idx)
        fev1, fvc = derive_fev1_fvc(record.time_s, corrected_volume, t0, end_idx)
        t0_sample_error = (t0 - record.true_t0_s) * 100.0
        fev1_pct_error = 100.0 * (fev1 - record.true_fev1_ml) / record.true_fev1_ml
        rows.append(
            {
                "seq": record.seq,
                "true_onset_idx": record.true_onset_idx,
                "detected_onset_idx": onset_idx,
                "pef_idx": pef_idx,
                "end_idx": end_idx,
                "t0_s": t0,
                "true_t0_s": record.true_t0_s,
                "t0_sample_error": t0_sample_error,
                "fev1_pct_error": fev1_pct_error,
                "fvc_pct_error": 100.0 * (fvc - record.true_fvc_ml) / record.true_fvc_ml,
            }
        )
    t0_errors = np.array([row["t0_sample_error"] for row in rows])
    fev1_abs = np.abs([row["fev1_pct_error"] for row in rows])
    pass_t0 = np.abs(t0_errors) <= 2.0
    pass_fev1 = fev1_abs <= 2.0
    return {
        "records": records,
        "rows": rows,
        "n": n,
        "t0_within_2_samples": int(pass_t0.sum()),
        "t0_within_2_samples_pct": float(pass_t0.mean() * 100.0),
        "fev1_within_2_pct": int(pass_fev1.sum()),
        "fev1_within_2_pct_pct": float(pass_fev1.mean() * 100.0),
        "joint_pass": int((pass_t0 & pass_fev1).sum()),
        "t0_error_median_samples": float(np.median(t0_errors)),
        "t0_error_iqr_samples": [float(x) for x in np.percentile(t0_errors, [25, 75])],
        "fev1_abs_pct_error_median": float(np.median(fev1_abs)),
        "fev1_abs_pct_error_iqr": [float(x) for x in np.percentile(fev1_abs, [25, 75])],
        "status": "PASS" if pass_t0.mean() >= 0.95 and pass_fev1.mean() >= 0.95 else "FAIL",
    }


def write_segmentation_report(result: dict, results_dir: Path, figures_dir: Path) -> None:
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    lines = [
        "# Module 1 - Segmentation + Back-Extrapolation",
        "",
        f"- status: {result['status']}",
        f"- synthetic curves: {result['n']}",
        "- recording synthesis: 10-50 baseline samples, OMBE exhale, negative-going inhale, Gaussian flow noise sigma=0.03 L/s",
        "- synthetic truth: true onset is the ATS-zeroed start of the exhale segment",
        "",
        "## Criteria",
        "",
        f"- t0 within 2 samples: {result['t0_within_2_samples']}/{result['n']} ({result['t0_within_2_samples_pct']:.1f}%)",
        f"- FEV1 within 2%: {result['fev1_within_2_pct']}/{result['n']} ({result['fev1_within_2_pct_pct']:.1f}%)",
        f"- joint pass: {result['joint_pass']}/{result['n']}",
        "",
        "## Error Distributions",
        "",
        f"- t0 sample error median: {result['t0_error_median_samples']:.3f}",
        f"- t0 sample error IQR: {result['t0_error_iqr_samples'][0]:.3f} to {result['t0_error_iqr_samples'][1]:.3f}",
        f"- abs FEV1 percent error median: {result['fev1_abs_pct_error_median']:.3f}%",
        f"- abs FEV1 percent error IQR: {result['fev1_abs_pct_error_iqr'][0]:.3f}% to {result['fev1_abs_pct_error_iqr'][1]:.3f}%",
        "",
    ]
    failures = [
        row
        for row in result["rows"]
        if abs(row["t0_sample_error"]) > 2.0 or abs(row["fev1_pct_error"]) > 2.0
    ]
    if failures:
        lines.extend(["## Failing Synthetic SEQs", "", "| seq | t0_sample_error | fev1_pct_error |", "|---:|---:|---:|"])
        for row in failures[:50]:
            lines.append(f"| {row['seq']} | {row['t0_sample_error']:.3f} | {row['fev1_pct_error']:.3f} |")
        lines.append("")
    (results_dir / "mod1_segmentation.md").write_text("\n".join(lines), encoding="utf-8")
    draw_examples(result, figures_dir / "mod1_examples.png")


def draw_examples(result: dict, path: Path) -> None:
    records = result["records"][:8]
    rows = {row["seq"]: row for row in result["rows"]}
    image = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 20), "Module 1 synthetic segmentation examples", fill="black")
    panel_w, panel_h = 540, 160
    for i, record in enumerate(records):
        x0 = 40 + (i % 2) * 580
        y0 = 70 + (i // 2) * 175
        flow = record.flow_lps
        t = record.time_s
        row = rows[record.seq]
        draw.rectangle((x0, y0, x0 + panel_w, y0 + panel_h), outline="black")
        y_mid = y0 + panel_h * 0.55
        scale_x = panel_w / (t[-1] - t[0])
        max_abs = max(0.5, float(np.max(np.abs(flow))))
        scale_y = (panel_h * 0.45) / max_abs
        points = [(x0 + (tt - t[0]) * scale_x, y_mid - ff * scale_y) for tt, ff in zip(t[::5], flow[::5])]
        if len(points) > 1:
            draw.line(points, fill=(30, 80, 180), width=2)
        true_x = x0 + record.true_t0_s * scale_x
        rec_x = x0 + row["t0_s"] * scale_x
        draw.line((true_x, y0, true_x, y0 + panel_h), fill=(0, 150, 0), width=2)
        draw.line((rec_x, y0, rec_x, y0 + panel_h), fill=(200, 40, 40), width=2)
        draw.text((x0 + 5, y0 + 5), f"seq {record.seq}", fill="black")
    draw.text((40, 765), "green=true ATS onset, red=recovered back-extrapolated t0", fill="black")
    image.save(path)


def main() -> int:
    result = validate_segmentation()
    write_segmentation_report(result, Path("results"), Path("figures"))
    print(f"Module 1 {result['status']}: t0 {result['t0_within_2_samples']}/{result['n']}, FEV1 {result['fev1_within_2_pct']}/{result['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
