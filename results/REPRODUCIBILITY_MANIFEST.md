# Reproducibility Manifest

## Corrected analyses in this revision

The Cycle F parser, fixed-horizon truncation analysis, graded recalibration,
and coverage frontier were regenerated on 2026-10-18. The source revision is
the initial public repository revision associated with these files.

Commands:

```bash
python -c 'from pathlib import Path; from predictx.gate23_f import run_gate2; run_gate2(Path.cwd(), "F")'
python -m predictx.gate_truncation_recovery_f --root .
python -m predictx.gate_graded_recalibrated_f --root .
python -m predictx.gate_frontier_f --root .
python -m ruff check predictx tests
python -m pytest -q
python -m compileall -q predictx tests
```

## Public Cycle F inputs

| File | SHA-256 |
|---|---|
| `SPXRAW_F.zip` | `e266b2a1d3ad7cb705f2560851e42560381c5f4697f626d1ce78170c5cb4d706` |
| `SPX_F.xpt` | `99477440c27a4b4423a47bf42632a14fca3281545e42937fc42e0523f0abbcfb` |

Both files were downloaded from the official CDC NHANES 2009-2010 data
release. Raw inputs are excluded from version control.

## Software environment

| Component | Version |
|---|---:|
| Python | 3.12.10 |
| NumPy | 2.5.4 |
| pandas | 2.3.3 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.1 |
| pyreadstat | 1.3.6 |
| PyArrow | 22.0.0 |
| Matplotlib | 3.11.2 |

The optional JAX and posterior-inference dependencies were not installed for
this regeneration. The JAX-specific unit test was therefore skipped. No
posterior-inference result was regenerated or upgraded to supported status.

## Verification outcome

- Static checks: passed.
- Unit and protocol-integrity tests: 5 passed, 1 optional JAX test skipped.
- Bytecode compilation: passed.
- Cycle F parser gate: passed on 7,079 participants.
- Corrected truncation gate: `PHYSICS-JUSTIFIED` under the registered internal
  comparison rule, with important clinical limitations documented in
  `docs/RESULTS.md` and `docs/LIMITATIONS.md`.
- CLEAN frontier gate: `FRONTIER-UNSAFE` on held-out TEST safety constraints.
