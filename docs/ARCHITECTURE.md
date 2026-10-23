# Software architecture

## Design approach

The codebase separates source ingestion, feature reconstruction, physiological modeling and study-specific evaluation. Files prefixed `gate_` are **experimental analysis entry points**, preserving the provenance of alternative research questions, diagnostics and protocol revisions. They are not intended to be imported as a public high-level API.

| Layer | Principal modules | Responsibility |
| --- | --- | --- |
| Data access | `io.py`, `dataset_resolver.py`, `schema.py` | Resolve source datasets, inspect schemas, validate NHANES input |
| Signal reconstruction | `raw.py`, `merge.py` | Restore volume/flow time series, align participant and maneuver information |
| Reference measurements | `gli.py`, `gli_reference.R`, `obstruction.py` | Generate reference inputs and evaluate spirometric ratios |
| Mechanistic computation | `ombe.py`, `ombe_gamma.py`, `fit.py` | Evaluate model functions and solve parameter estimation problems |
| Analytical protocols | `gate*.py` | Implement cohort selection, truncation, benchmarks, inference and audits |
| Outputs | `results/`, `figures/` | Store versioned study artifacts for inspection |
| Verification | `tests/`, `.github/workflows/quality.yml` | Exercise software invariants and automation |

## Key methodological decisions

1. **Sampling correctness:** distinguish volume increments from cumulative volume and preserve the 100 Hz sampling interval.
2. **Reference integrity:** maintain separation between raw maneuver-level measurements and released same-session summary fields.
3. **Controlled extrapolation:** hold the prediction horizon fixed and prevent access to samples acquired after a simulated termination.
4. **Participant-level inference:** separate cohorts by participant identifier to avoid maneuver-level overlap across splits.
5. **Evidence provenance:** compare historical result artifacts against current protocol documentation before reporting them.

## Extension and maintenance guidelines

- Put reusable numerical primitives in `predictx/` with explicit inputs, outputs and units.
- Keep experiment-specific data filtering, seeds, thresholds and artifact filenames in dedicated protocol modules.
- Prefer structured Python logging and parameterized functions over print-based messages in reusable libraries.
- Record protocol changes in documentation and add regression tests for any correction that changes scientific outputs.
- Preserve raw inputs unchanged; write derived data and plots to designated output directories.
- Preserve negative as well as positive evaluations. Analytical integrity is part of a professional research implementation.

For authoritative mathematical definitions, see [Methods](METHODS.md). For execution and data layout, see [Reproducibility](REPRODUCIBILITY.md).
