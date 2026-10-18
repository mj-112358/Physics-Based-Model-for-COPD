# Gate - Covariate-Assisted Resolution of Deferred Cases, Cycle F

VERDICT: CONTEXT-ADDS-NOTHING - covariate-resolved calls did not beat the naive 0.70 spirometry flip on the same cases, so Tier 2 is discarded.

## Protocol Decisions

- TEST split only.
- Operated only on residual deferred cases from gate_deferral_recovery_f: deferred_baseline == True and recovered == False.
- The clinical context rule was declared before evaluation and was not fit to outcomes.
- Borderline band was fixed as corrected_ratio_c1 in [0.65, 0.75] for R2/R3 residuals only.
- RDQ has no direct dyspnea variable in these cycles; RDQ135 wheeze-limited activity is reported as a dyspnea/activity proxy.
- Pack-years are approximate and only used for availability reporting, not for the tiebreak rule.
- Covariates never override already-finalized spirometry calls.
- Tier 2 is retained only if context beats the naive 0.70 flip on the exact same cases and severe FN remains zero.

## Part 0 - Scope

| residual reason | n | fraction residual |
|---|---|---|
| R3_FAILED_REPEATABILITY | 103 | 0.548 |
| R2_BORDERLINE_RATIO | 39 | 0.207 |
| R4_GENUINELY_UNINTERPRETABLE | 33 | 0.176 |
| R1_TRUNCATED_INCOMPLETE | 13 | 0.069 |

## Part 1 - Covariate Availability

| covariate | available n | deferred n | available fraction |
|---|---|---|---|
| DEMO.RIDAGEYR age | 188 | 188 | 1.000 |
| DEMO.RIAGENDR sex | 188 | 188 | 1.000 |
| DEMO.RIDRETH1 race/ethnicity | 188 | 188 | 1.000 |
| BMX.BMXHT height | 188 | 188 | 1.000 |
| BMX.BMXBMI BMI | 188 | 188 | 1.000 |
| SMQ smoking status | 158 | 188 | 0.840 |
| SMQ pack-years derivable | 155 | 188 | 0.824 |
| MCQ emphysema/chronic bronchitis | 156 | 188 | 0.830 |
| RDQ cough | 143 | 188 | 0.761 |
| RDQ wheeze | 188 | 188 | 1.000 |
| RDQ dyspnea/activity proxy | 186 | 188 | 0.989 |
| RDQ any respiratory symptom | 151 | 188 | 0.803 |

## Part 2 - Declared Clinical Tiebreak Rule

- scope: only residual deferred TEST cases; only R2 and R3 with corrected_ratio_c1 in [0.65, 0.75]
- high_risk: age >= 40 AND ever-smoker AND (self-reported emphysema/chronic bronchitis OR >=1 RDQ respiratory symptom) -> OBSTRUCTED
- low_risk: never-smoker AND no self-reported emphysema/chronic bronchitis AND no RDQ respiratory symptoms -> NORMAL
- intermediate: all other cases stay deferred
- naive_comparator: on the exact same resolved cases, classify corrected_ratio_c1 < 0.70 as OBSTRUCTED and otherwise NORMAL

Eligible borderline residual cases: 126
Eligible risk counts: {'INTERMEDIATE': 73, 'LOW': 31, 'HIGH': 22}

## Part 3 - Honest Evaluation

| method | resolved n | correct | errors | accuracy | severe n | severe FN | severe FN CP high |
|---|---|---|---|---|---|---|---|
| context rule | 53 | 26 | 27 | 0.491 | 0 | 0 | NA |
| naive 0.70 flip | 53 | 29 | 24 | 0.547 | 0 | 0 | NA |

| method | participants with proxy | held-out blow correct | held-out blow n | held-out blow acc | participant majority correct | participant majority n | participant majority acc |
|---|---|---|---|---|---|---|---|
| context rule | 52 | 89 | 182 | 0.489 | 23 | 52 | 0.442 |
| naive 0.70 flip | 52 | 123 | 182 | 0.676 | 35 | 52 | 0.673 |

| resolved n | HIGH -> obstructed | LOW -> normal | still deferred |
|---|---|---|---|
| 53 | 22 | 31 | 135 |

Candidate coverage if retained: 0.760.
Candidate retained: False.

## Part 4 - Tiered Output

| tier | finalized n | coverage | held-out blow correct | held-out blow n | held-out blow acc | retained |
|---|---|---|---|---|---|---|
| TIER 1 spirometry-alone retained | 375 | 0.666 | 1181 | 1208 | 0.978 | True |
| TIER 2 spirometry + clinical context candidate | 53 | 0.094 | 89 | 182 | 0.489 | False |

| scope | finalized n | coverage | sens | spec | kappa | severe finalized | severe FN | severe FN CP high |
|---|---|---|---|---|---|---|---|---|
| final retained policy | 375 | 0.666 | 0.943 | 0.977 | 0.923 | 6 | 0 | 0.459 |

Tier 2 is discarded because it does not beat the naive 0.70 comparator on the same cases. Final retained coverage therefore remains the current 66.6%.

VERDICT: CONTEXT-ADDS-NOTHING - covariate-resolved calls did not beat the naive 0.70 spirometry flip on the same cases, so Tier 2 is discarded.
