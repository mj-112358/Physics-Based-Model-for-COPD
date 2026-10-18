# Obstruction Composition

## Phase 0 GLI Input Export

- merged rows: 30442
- exported rows: 26908
- dropped for missing age/height/sex: 3534

## Phase 1 Assertion

Post-fix `effort_quality.value_counts(dropna=False)`:

```text
effort_quality
A       18723
None    10177
C        1001
B         541
```

## GLI Reference

- rspiro version: 0.5
- exact GLI functions called: pred_GLI; LLN_GLI
- GLI input rows: 26908
- GLI reference rows written: 26047
- fallback input rows mapped to GLI Other/mixed: 10430
- fallback output rows mapped to GLI Other/mixed: 10033
- rows dropped because rspiro returned non-finite GLI values: 860
- finite rows dropped by predicted FEV1 sanity range 0.5-6 L: 1

See `results/gli_mapping_notes.md` for mapping comments and validation notes.

## SEQN Join Log

| join scope | matched | unmatched |
|---|---:|---:|
| merged -> GLI reference | 26047 | 4395 |
| ratio-not-null rows -> GLI reference | 19959 | 91 |
| quality-filtered rows -> GLI reference | 17549 | 77 |

## Filter Log

| filter | before | after | dropped |
|---|---:|---:|---:|
| Restrict to fev1_fvc_ratio.notna() | 30442 | 20050 | 10392 |
| Quality filter: effort_quality in {A,B} AND fev1_quality in {A,B} | 20050 | 17626 | 2424 |
| Require matched finite GLI LLN reference | 17626 | 17549 | 77 |

Repeatability note: `acceptable_curve_count >= 3` is not used here; true repeatability requires raw curves and is deferred.

## Obstruction Composition

| definition | n_total | n_obstructed | n_0_60_to_0_70 | n_0_50_to_0_60 | n_lt_0_50 | n_obstructed_gte_0_70 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed <0.70 | 17549 | 1703 | 1334 | 265 | 104 | 0 |
| LLN | 17549 | 1825 | 790 | 265 | 104 | 666 |

## LLN Adds/Removes Versus Fixed <0.70

| age band | LLN adds vs fixed | LLN removes vs fixed |
|---|---:|---:|
| <50 | 665 | 11 |
| >=50 | 1 | 533 |

Pattern flag: expected age-band add/remove pattern did NOT appear; GLI mapping or units may be wrong.

## Decision Inputs

| definition | usable moderate 0.50-0.60 | usable severe <0.50 | combined moderate+severe |
|---|---:|---:|---:|
| Fixed <0.70 | 265 | 104 | 369 |
| LLN | 265 | 104 | 369 |

Severe n under LLN = 104. Combined moderate+severe = 369. Below ~300 combined means collapse to a single 'moderate-or-worse' stratum for coverage claims.

Headline: fixed severe = 104; fixed moderate+severe = 369; LLN severe = 104; LLN moderate+severe = 369.
