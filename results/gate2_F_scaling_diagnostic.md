# Gate 2 Cycle F Scaling Diagnostic

No fix applied. This diagnoses whether the Gate 2 low bias is multiplicative/BTPS-related or definitional.

- raw source: <PROJECT_ROOT>/NHANES_SpiroSBI/2009-2010/spxraw_f.sas7bdat
- raw rows: 36421
- parse errors: 45

## derived/SPXN Ratio vs SPAFACT

| set | n | median FEV1 ratio | FEV1 ratio IQR | median FVC ratio | FVC ratio IQR | median SPAFACT | SPAFACT IQR | Pearson r FEV1~SPAFACT | Pearson r FVC~SPAFACT | Spearman r FEV1~SPAFACT | Spearman r FVC~SPAFACT |
|---|---:|---:|---|---:|---|---:|---|---:|---:|---:|---:|
| first acceptable A Pre per participant | 6638 | 0.9438 | 0.9140-0.9622 | 0.9346 | 0.9054-0.9564 | 1.0840 | 1.0750-1.0910 | 0.0128 | 0.0232 | -0.0056 | 0.0143 |
| all acceptable A Pre curves | 18617 | 0.9527 | 0.9260-0.9681 | 0.9412 | 0.9140-0.9609 | 1.0840 | 1.0750-1.0910 | 0.0299 | 0.0217 | 0.0146 | 0.0144 |

## First-Acceptable vs Max Comparison

| set | n | median signed FEV1 err | median abs FEV1 err | median signed FVC err | median abs FVC err |
|---|---:|---:|---:|---:|---:|
| first acceptable A Pre per participant | 6638 | -5.62% | 5.62% | -6.54% | 6.54% |
| max acceptable A Pre per participant | 6638 | -3.02% | 3.02% | -3.82% | 3.82% |
| max acceptable Pre per participant | 7079 | -2.89% | 2.91% | -3.49% | 3.53% |

## Interpretation

- derived/SPXN ratio has weak correlation with SPAFACT; this does not support a per-curve BTPS-factor application error.
- ratio IQR is not tight; segmentation/definition contributes beyond a pure scalar.
- max acceptable comparison materially shrinks the error, so part of the Gate 2 failure is definitional: SPXN behaves closer to best-test than first-acceptable.
