# Locked Post-Audit Cycle E and G Evaluation

All model coefficients, the Cycle F constant ratio correction, physical parameter bounds, workflow thresholds, recovery rules, and selection guards were frozen before Cycle E and G outcomes were scored.

## Consolidated Comparison

| Cycle | Method | n | Coverage | Sensitivity | Specificity | Kappa | FN | FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E | state_aware_physics_workflow | 5405 | 0.897 | 0.949 | 0.973 | 0.839 | 25 | 133 |
| E | quality_control_first | 5033 | 0.835 | 0.849 | 0.991 | 0.868 | 79 | 40 |
| E | fev1_fev6 | 6026 | 1.000 | 0.692 | 0.990 | 0.756 | 187 | 53 |
| E | direct_ratio_matched_coverage | 5405 | 0.897 | 0.929 | 0.995 | 0.926 | 24 | 23 |
| E | direct_ratio_on_physics_cases | 5405 | 0.897 | 0.868 | 0.991 | 0.875 | 65 | 45 |
| E | archived_ridge_all | 6026 | 1.000 | 0.970 | 0.918 | 0.678 | 18 | 443 |
| E | archived_ridge_matched_coverage | 5405 | 0.897 | 0.986 | 0.963 | 0.837 | 8 | 181 |
| E | state_aware_truncated_ridge | 6026 | 1.000 | 0.863 | 0.979 | 0.824 | 83 | 113 |
| E | state_aware_ridge_matched_coverage | 5405 | 0.897 | 0.941 | 0.991 | 0.904 | 22 | 47 |
| E | state_aware_ridge_on_physics_cases | 5405 | 0.897 | 0.878 | 0.987 | 0.860 | 60 | 66 |
| E | physics_plus_ridge_hybrid | 6026 | 1.000 | 0.863 | 0.978 | 0.819 | 83 | 120 |
| E | physics_plus_ridge_hybrid_matched_coverage | 5405 | 0.897 | 0.941 | 0.989 | 0.894 | 22 | 55 |
| E | physics_plus_ridge_hybrid_on_physics_cases | 5405 | 0.897 | 0.878 | 0.986 | 0.857 | 60 | 69 |
| E | physics_ridge_consensus | 5273 | 0.875 | 0.945 | 0.989 | 0.911 | 25 | 51 |
| E | direct_ratio_at_consensus_coverage | 5273 | 0.875 | 0.944 | 0.996 | 0.933 | 17 | 22 |
| E | state_aware_ridge_at_consensus_coverage | 5273 | 0.875 | 0.952 | 0.991 | 0.907 | 16 | 44 |
| E | state_aware_ridge_on_consensus_cases | 5273 | 0.875 | 0.945 | 0.989 | 0.911 | 25 | 51 |
| E | direct_ratio_fixed_specificity | 6026 | 1.000 | 0.947 | 0.955 | 0.781 | 32 | 244 |
| E | archived_ridge_fixed_specificity | 6026 | 1.000 | 0.876 | 0.979 | 0.832 | 75 | 113 |
| E | state_aware_ridge_fixed_specificity | 6026 | 1.000 | 0.895 | 0.975 | 0.828 | 64 | 133 |
| E | physics_plus_ridge_hybrid_fixed_specificity | 6026 | 1.000 | 0.900 | 0.970 | 0.809 | 61 | 163 |
| G | state_aware_physics_workflow | 5712 | 0.891 | 0.973 | 0.974 | 0.836 | 12 | 139 |
| G | quality_control_first | 5349 | 0.835 | 0.819 | 0.993 | 0.853 | 86 | 35 |
| G | fev1_fev6 | 6408 | 1.000 | 0.685 | 0.992 | 0.756 | 177 | 47 |
| G | direct_ratio_matched_coverage | 5712 | 0.891 | 0.935 | 0.998 | 0.944 | 17 | 11 |
| G | direct_ratio_on_physics_cases | 5712 | 0.891 | 0.860 | 0.994 | 0.883 | 62 | 30 |
| G | archived_ridge_all | 6408 | 1.000 | 0.977 | 0.928 | 0.680 | 13 | 423 |
| G | archived_ridge_matched_coverage | 5712 | 0.891 | 0.983 | 0.970 | 0.846 | 9 | 154 |
| G | state_aware_truncated_ridge | 6408 | 1.000 | 0.843 | 0.982 | 0.814 | 88 | 105 |
| G | state_aware_ridge_matched_coverage | 5712 | 0.891 | 0.943 | 0.993 | 0.903 | 17 | 40 |
| G | state_aware_ridge_on_physics_cases | 5712 | 0.891 | 0.880 | 0.991 | 0.876 | 53 | 48 |
| G | physics_plus_ridge_hybrid | 6408 | 1.000 | 0.842 | 0.978 | 0.796 | 89 | 127 |
| G | physics_plus_ridge_hybrid_matched_coverage | 5712 | 0.891 | 0.944 | 0.989 | 0.878 | 17 | 58 |
| G | physics_plus_ridge_hybrid_on_physics_cases | 5712 | 0.891 | 0.880 | 0.990 | 0.869 | 53 | 54 |
| G | physics_ridge_consensus | 5558 | 0.867 | 0.970 | 0.993 | 0.936 | 12 | 37 |
| G | direct_ratio_at_consensus_coverage | 5558 | 0.867 | 0.942 | 0.998 | 0.946 | 13 | 10 |
| G | state_aware_ridge_at_consensus_coverage | 5558 | 0.867 | 0.947 | 0.994 | 0.909 | 14 | 34 |
| G | state_aware_ridge_on_consensus_cases | 5558 | 0.867 | 0.970 | 0.993 | 0.936 | 12 | 37 |
| G | direct_ratio_fixed_specificity | 6408 | 1.000 | 0.929 | 0.965 | 0.791 | 40 | 202 |
| G | archived_ridge_fixed_specificity | 6408 | 1.000 | 0.858 | 0.981 | 0.817 | 80 | 112 |
| G | state_aware_ridge_fixed_specificity | 6408 | 1.000 | 0.884 | 0.978 | 0.818 | 65 | 131 |
| G | physics_plus_ridge_hybrid_fixed_specificity | 6408 | 1.000 | 0.893 | 0.970 | 0.788 | 60 | 178 |

## Cohort Accounting

### Cycle E

The evaluation included 6026 participants with an acceptable A/B pre-bronchodilator manoeuvre. The physics workflow deferred 621 participants, and 458 participants had no complete acceptable manoeuvre under the frozen ATS endpoint rule.

Deferral reasons: {'R3_FAILED_REPEATABILITY': 232, 'R4_GENUINELY_UNINTERPRETABLE': 226, 'R2_BORDERLINE_RATIO': 129, 'R1_TRUNCATED_INCOMPLETE': 34}

### Cycle G

The evaluation included 6408 participants with an acceptable A/B pre-bronchodilator manoeuvre. The physics workflow deferred 696 participants, and 519 participants had no complete acceptable manoeuvre under the frozen ATS endpoint rule.

Deferral reasons: {'R4_GENUINELY_UNINTERPRETABLE': 278, 'R3_FAILED_REPEATABILITY': 251, 'R2_BORDERLINE_RATIO': 121, 'R1_TRUNCATED_INCOMPLETE': 46}

## Interpretation

The archived ridge result is retained for audit continuity, but it uses the terminal volume of the full selected manoeuvre and is therefore not a clean censored-prefix comparator. The state-aware truncated-only ridge is the more relevant learned baseline for participants whose endpoint is incomplete.

The corrected workflow routes complete manoeuvres to direct endpoints and invokes physics recovery only for incomplete endpoints. The consolidated table reports standalone physics, state-aware ridge, and the Cycle-F-trained physics-plus-ridge hybrid under identical participant eligibility and coverage definitions.

Paired participant-bootstrap differences, endpoint-error estimates, no-complete-manoeuvre results, and full machine-readable cohort details are stored in the companion JSON.
