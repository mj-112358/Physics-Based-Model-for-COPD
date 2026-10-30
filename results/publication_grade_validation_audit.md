# Publication-Grade Validation Audit

## Executive verdict

The executed evidence does not support superior endpoint reconstruction or superior standalone classification by the physics-constrained workflow relative to ridge regression. In frozen Cycle E/G scoring, physics reduced FVC error relative to the raw observed endpoint by 20.9 mL and 18.0 mL, but its FVC mean absolute error exceeded the incomplete-state ridge by 91.5 mL (95% CI 88.0–95.3) and 87.7 mL (83.9–91.3). At matched coverage, physics kappa was lower than state-aware ridge by 0.093 (0.065–0.121) in Cycle E and 0.102 (0.069–0.135) in Cycle G.

The reproducible incremental contribution is selective deferral. Requiring the frozen physics and state-aware ridge decisions to agree retained 84.0% of Cycle E and 83.6% of Cycle G. Relative to ridge confidence ranking at equal coverage, this agreement rule improved kappa by 0.023 (95% CI 0.0003–0.045) and 0.036 (0.012–0.062). Physics and ridge predictions are identical on the retained agreement subset; the gain therefore arises from case selection, not predictive fusion.

## 1. Existing ridge experiment audit

### Partition integrity

- Cycle F contains 562 TRAIN and 563 TEST participants, each represented once, with zero sequence-identifier overlap.
- The reported final-policy table contains the same 563 TEST participants and identical truth labels.
- Ridge regularization was selected by five deterministic participant folds within TRAIN.
- Median imputation, centering, scaling, and coefficients were fitted inside each training fold and refitted on the complete TRAIN partition.
- No TEST labels entered regularization selection or coefficient estimation.

### Feature provenance and leakage assessment

The archived ridge features are selected-manoeuvre FEV1, terminal observed volume, duration, age, height, and sex. The target is released summary FVC, and the binary truth is released summary FEV1/FVC below 0.70. The terminal-volume feature correlates 0.9934 with summary FVC and differs from it by 185.7 mL mean absolute error.

There is no participant record leakage and no use of TEST labels during model fitting. There is, however, target-proxy leakage relative to the latent missing-FVC estimand: the archived ridge observes the terminal volume of the full selected same-session manoeuvre. It is consequently an audit comparator, not an equal-information censored-prefix baseline.

The matched-coverage count and target specificity were copied from the Cycle F policy-development TEST result. Ranking and threshold fitting did not use TEST labels, but the reported operating points are indirectly TEST-informed. The physics policy was also iteratively compared on this partition. All Cycle F workflow and baseline figures are therefore internal policy-development estimates.

### Bootstrap audit

The archived analysis uses 2,000 percentile replicates and resamples one-row-per-participant evaluation tables. It is a participant bootstrap. It conditions on the fitted ridge model, does not refit the model within each replicate, and originally omitted paired method-difference intervals. The frozen evaluation adds paired participant-bootstrap intervals for classification metrics and FVC mean absolute error.

## 2. Data acquisition and parser concordance

Official CDC files were downloaded and hashed:

- SPXRAW_E.zip: `190f2e8bfc82f0b5a3bf7ba1277cdab34a9ea3fea3b8be5ca8a9ecee8c957a2e`
- SPX_E.xpt: `88b4012ccd64f3f276f1afe64feddc26705f62891e0fdd3f16ea36d68dcefe03`
- SPXRAW_G.zip: `e10fda23ead8cb7efcbc6169b03f4ecc1b952de7c2cf380f84cadcc290f7b3ae`
- SPX_G.xpt: `58177b07ad94a2c005b0c9494cddb82fc94ec8f5c6c13a91b83da035f20770d9`

The raw files contain 34,857 Cycle E and 37,661 Cycle G manoeuvre rows. The existing parser passed its registered concordance gate in both cycles. Among participants with released summary endpoints, median absolute FEV1/FVC endpoint errors were 3.143%/3.716% in Cycle E and 2.948%/3.611% in Cycle G. Signed errors remained negative, so parser error is not negligible and remains part of every downstream comparison.

## 3. Frozen configuration and validation status

The current scoring run froze the Cycle F ratio correction, five physical bounds, ATS endpoint rule, recovery rules, workflow thresholds, ridge coefficients, imputation/scaling statistics, regularization, and classification thresholds before Cycle E/G participant outcomes were scored. No Cycle E/G outcome was used to modify these values.

Cycles E and G had appeared in earlier repository truncation and subgroup analyses. The present work is therefore a locked participant-level final-workflow transport test, not a pristine independent validation. It also remains within NHANES and shares acquisition semantics and released endpoint construction. It is not external clinical validation.

Code audit identified a manuscript–implementation mismatch. The manuscript described direct FVC for complete endpoints and physics recovery only for incomplete endpoints. The frozen implementation applies corrected model-derived ratios in the primary workflow and in its repeated-manoeuvre evidence path. The reported Cycle E/G physics results correspond to the executed code. Direct-for-complete routing is an unexecuted revision and cannot be claimed as part of these results.

## 4. Cohort accounting

| Quantity | Cycle E | Cycle G |
|---|---:|---:|
| Released summary rows | 7,743 | 7,495 |
| Valid FEV1/FVC truth | 6,319 | 6,652 |
| Raw manoeuvre rows | 34,857 | 37,661 |
| Acceptable A/B candidate rows | 21,789 | 24,005 |
| Eligible participants with candidate | 6,026 | 6,408 |
| No complete acceptable manoeuvre | 458 | 519 |
| Obstructed participants | 607 | 562 |
| Physics deferrals | 786 | 834 |

Cycle E deferrals comprised 336 uninterpretable signals, 247 failed-repeatability cases, 159 boundary cases, and 44 unresolved truncations. Cycle G contained 395, 253, 125, and 61 respectively. Deferred participants included 135 obstructed Cycle E participants and 120 obstructed Cycle G participants, including 18 and 20 severe cases.

## 5. Frozen cross-cycle results

| Cycle | Method | Coverage | Sensitivity | Specificity | Kappa | FN | FP |
|---|---|---:|---:|---:|---:|---:|---:|
| E | Physics workflow | 0.870 | 0.958 | 0.967 | 0.817 | 20 | 158 |
| E | QC-first | 0.835 | 0.849 | 0.991 | 0.868 | 79 | 40 |
| E | FEV1/FEV6 | 1.000 | 0.692 | 0.990 | 0.756 | 187 | 53 |
| E | Direct ratio, matched coverage | 0.870 | 0.946 | 0.996 | 0.934 | 16 | 21 |
| E | State-aware ridge, matched coverage | 0.870 | 0.957 | 0.991 | 0.910 | 14 | 44 |
| E | Physics–ridge agreement | 0.840 | 0.962 | 0.991 | 0.930 | 16 | 40 |
| G | Physics workflow | 0.870 | 0.980 | 0.966 | 0.805 | 9 | 177 |
| G | QC-first | 0.835 | 0.819 | 0.993 | 0.853 | 86 | 35 |
| G | FEV1/FEV6 | 1.000 | 0.685 | 0.992 | 0.756 | 177 | 47 |
| G | Direct ratio, matched coverage | 0.870 | 0.938 | 0.998 | 0.942 | 14 | 11 |
| G | State-aware ridge, matched coverage | 0.870 | 0.948 | 0.993 | 0.907 | 14 | 35 |
| G | Physics–ridge agreement | 0.836 | 0.982 | 0.995 | 0.955 | 7 | 26 |

The standalone physics workflow transported with nearly identical coverage and specificity, supporting operational stability. It did not outperform direct-ratio or ridge classification at matched coverage. Physics produced higher sensitivity than direct selection in Cycle G and higher sensitivity than state-aware ridge in Cycle G, paired with materially lower specificity and kappa.

At agreement coverage, physics–ridge agreement improved kappa over ridge confidence ranking by 0.023 in Cycle E and 0.036 in Cycle G. Against direct-ratio selection at the same coverage, the kappa differences were -0.004 (-0.023 to 0.017) and 0.012 (-0.010 to 0.035), neither excluding zero.

## 6. Endpoint accuracy and no-complete-manoeuvre subgroup

| Cycle | Endpoint estimator | FVC MAE (mL) | 95% CI |
|---|---|---:|---:|
| E | Direct observed | 197.5 | 194.1–201.0 |
| E | Physics | 176.6 | 172.9–180.6 |
| E | Incomplete-state ridge | 85.1 | 82.5–87.7 |
| G | Direct observed | 192.6 | 189.1–196.3 |
| G | Physics | 174.6 | 170.9–178.4 |
| G | Incomplete-state ridge | 86.9 | 84.4–89.6 |

For participants with no complete acceptable manoeuvre, physics FVC MAE was 293.3 mL in Cycle E and 316.5 mL in Cycle G, compared with 130.9 and 125.9 mL for ridge. The paired physics-minus-ridge disadvantages were 162.4 mL (135.5–188.7) and 190.6 mL (163.4–216.2). This subgroup directly contradicts a claim of superior physics-based latent-FVC recovery.

Physics finalized 63.3% and 58.6% of the no-complete subgroup. Sensitivity was 91.3% and 100%, while specificity fell to 79.0% and 72.2%. The agreement gate reduced subgroup coverage to 52.2% and 43.0%, with kappa 0.703 and 0.799. These data support conservative deferral rather than autonomous recovery.

## 7. Scientific contribution after validation

- **Endpoint accuracy:** no incremental advantage over ridge; physics is significantly worse.
- **Standalone classification:** no matched-coverage advantage over ridge or direct-ratio selection.
- **Safer deferral:** supported. Cross-model agreement improves kappa over ridge confidence ranking at equal coverage in both cycles and concentrates unresolved risk in explicit deferrals.
- **Robustness:** the physics workflow shows stable coverage and specificity across E/G and earlier fixed perturbation tests, but no new physical-device experiment was executed in this audit.
- **Interpretability:** structural constraints, endpoint provenance, and deferral reasons are explicit. This is a design property; the audit does not convert it into a clinical-outcome effect.
- **External validity:** not established. E/G are within-program transport cohorts, and prior ancillary exposure prevents describing them as pristine independent validation.

## 8. Required manuscript revisions

1. Replace the abstract's Cycle F development headline with the frozen E/G comparison and report ridge superiority in endpoint accuracy.
2. Reframe novelty as an agreement-based selective-inference contribution, not superior physics reconstruction.
3. Label all Cycle F policy figures as internal development estimates.
4. Describe Cycle E/G as frozen within-NHANES transport, including prior ancillary exposure; do not call it external validation.
5. Disclose the archived ridge terminal-volume target proxy and separate record leakage from estimand leakage.
6. Report state-aware incomplete-manoeuvre ridge as the principal learned comparator.
7. Add matched-coverage, exact-subset, and equal-consensus-coverage comparisons with paired confidence intervals.
8. Add FVC error and the no-complete-acceptable-manoeuvre subgroup.
9. Report deferral counts, causes, class distribution, and severe cases among deferred participants.
10. Correct the direct-for-complete manuscript claim or modify and prospectively re-freeze the implementation; the present results correspond to model-derived ratios across the primary workflow.
11. Remove statements implying that physics–ridge agreement is a predictive ensemble. It is a selective agreement gate; retained predictions equal ridge predictions.
12. Restrict clinical claims to pre-bronchodilator airflow-obstruction screening and require prospective post-bronchodilator, cross-device clinical validation.

## 9. Reproducibility artifacts

- `tools/audit_ridge_experiment.py`
- `tools/run_frozen_cross_cycle_validation.py`
- `results/ridge_experiment_validation_audit.json`
- `results/frozen_validation_configuration.json`
- `results/frozen_validation_E.json`
- `results/frozen_validation_G.json`
- `results/frozen_cross_cycle_validation.json`
- `results/frozen_validation_consolidated.csv`
- `results/frozen_validation_E_participants.csv`
- `results/frozen_validation_G_participants.csv`

The machine-readable JSON contains 2,000-replicate confidence intervals, paired differences, FVC errors, no-complete subgroup metrics, class distributions, deferral causes, and cryptographic source hashes.
