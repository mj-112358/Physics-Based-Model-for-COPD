# Pooled Severe Cohort EFG

SEVERE POWER: SEVERE-POWERED-RECOVERED

## Paired Severe Cohort

| cycle | paired severe n |
|---|---|
| E | 15 |
| F | 13 |
| G | 18 |

- pooled paired severe n: 46

The pooled severe paired-incomplete cohort is powered at n>=20, so severe-specific incomplete-maneuver recovery was run.

## Pooled Incomplete Severe Recovery

| estimator | endpoint | FN | specificity | fit failures |
|---|---|---|---|---|
| E0_DIRECT | field_incomplete | 0.152 (7/46; CI 0.065-0.261) | NA (0/0; CI NA-NA) UNDERPOWERED | 0 |
| E1_OMBE | field_incomplete | 0.000 (0/46; exact 95% CI 0.000-0.077) | NA (0/0; CI NA-NA) UNDERPOWERED | 0 |
| E2_SPLINE | field_incomplete | 0.652 (30/46; CI 0.500-0.783) | NA (0/0; CI NA-NA) UNDERPOWERED | 25 |
| E3_LINEAR_TAIL | field_incomplete | 0.043 (2/46; CI 0.000-0.109) UNDERPOWERED | NA (0/0; CI NA-NA) UNDERPOWERED | 0 |
| E4_MONO_EXPONENTIAL | field_incomplete | 0.000 (0/46; exact 95% CI 0.000-0.077) | NA (0/0; CI NA-NA) UNDERPOWERED | 0 |
| H1 | field_incomplete | 0.087 (4/46; CI 0.022-0.174) | NA (0/0; CI NA-NA) UNDERPOWERED | 0 |

Zero-event severe pooled field-incomplete intervals use exact two-sided Clopper-Pearson 95% upper bound 0.077062; rule-of-three upper bound is 0.065217.

## Pooled Severity-Resolved Clean-Cut Recovery

| estimator | T | severity | FN |
|---|---|---|---|
| E1_OMBE | 2 | mild_0.60_0.70 | 0.738 (443/600; CI 0.702-0.772) |
| E1_OMBE | 2 | moderate | 0.316 (71/225; CI 0.258-0.378) |
| E1_OMBE | 2 | severe | 0.000 (0/67; CI 0.000-0.000) |
| E1_OMBE | 3 | mild_0.60_0.70 | 0.767 (460/600; CI 0.733-0.802) |
| E1_OMBE | 3 | moderate | 0.253 (57/225; CI 0.196-0.311) |
| E1_OMBE | 3 | severe | 0.000 (0/67; CI 0.000-0.000) |
| E1_OMBE | 4 | mild_0.60_0.70 | 0.738 (443/600; CI 0.703-0.773) |
| E1_OMBE | 4 | moderate | 0.107 (24/225; CI 0.067-0.147) |
| E1_OMBE | 4 | severe | 0.000 (0/67; CI 0.000-0.000) |
| E1_OMBE | 5 | mild_0.60_0.70 | 0.658 (395/600; CI 0.620-0.697) |
| E1_OMBE | 5 | moderate | 0.018 (4/225; CI 0.004-0.036) |
| E1_OMBE | 5 | severe | 0.000 (0/67; CI 0.000-0.000) |
| H1 | 2 | mild_0.60_0.70 | 0.730 (438/600; CI 0.695-0.765) |
| H1 | 2 | moderate | 0.302 (68/225; CI 0.244-0.360) |
| H1 | 2 | severe | 0.000 (0/67; CI 0.000-0.000) |
| H1 | 3 | mild_0.60_0.70 | 0.760 (456/600; CI 0.727-0.793) |
| H1 | 3 | moderate | 0.218 (49/225; CI 0.164-0.271) |
| H1 | 3 | severe | 0.000 (0/67; CI 0.000-0.000) |
| H1 | 4 | mild_0.60_0.70 | 0.723 (434/600; CI 0.688-0.758) |
| H1 | 4 | moderate | 0.084 (19/225; CI 0.053-0.120) |
| H1 | 4 | severe | 0.000 (0/67; CI 0.000-0.000) |
| H1 | 5 | mild_0.60_0.70 | 0.645 (387/600; CI 0.607-0.683) |
| H1 | 5 | moderate | 0.018 (4/225; CI 0.004-0.036) |
| H1 | 5 | severe | 0.000 (0/67; CI 0.000-0.000) |

## Severe-Band Honesty Check

| estimator | T | F-only FN | pooled FN | perfect survives |
|---|---|---|---|---|
| E1_OMBE | 2 | 0.000 (0/17; CI NA-NA) | 0.000 (0/67; CI NA-NA) | True |
| E1_OMBE | 3 | 0.000 (0/17; CI NA-NA) | 0.000 (0/67; CI NA-NA) | True |
| E1_OMBE | 4 | 0.000 (0/17; CI NA-NA) | 0.000 (0/67; CI NA-NA) | True |
| E1_OMBE | 5 | 0.000 (0/17; CI NA-NA) | 0.000 (0/67; CI NA-NA) | True |
| H1 | 2 | 0.000 (0/17; CI NA-NA) | 0.000 (0/67; CI NA-NA) | True |
| H1 | 3 | 0.000 (0/17; CI NA-NA) | 0.000 (0/67; CI NA-NA) | True |
| H1 | 4 | 0.000 (0/17; CI NA-NA) | 0.000 (0/67; CI NA-NA) | True |
| H1 | 5 | 0.000 (0/17; CI NA-NA) | 0.000 (0/67; CI NA-NA) | True |
