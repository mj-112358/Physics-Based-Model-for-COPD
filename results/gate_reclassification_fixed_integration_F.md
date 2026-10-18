# Gate - Reclassification After Fixed Integration, Cycle F

- VERDICT: UNCHANGED-POINT-GATES - clinical/FVC/policy rerun values are unchanged because these gates already used the nonuniform NumPy volume path
- Scope: clinical classification, FVC convention, and ATS flag policy gates only.
- No gamma, registration, ML residual, or threshold movement was introduced.

## Step 0 - Integration Audit

| module/function | use | status | evidence |
|---|---|---|---|
| predictx/ombe.py::ombe_volume | shared JAX OMBE volume | fixed | uses jnp.diff(t) for trapezoid interval widths |
| predictx/ombe_gamma.py::ombe_gamma_volume | JAX OMBE-gamma volume, out of scope for this gate | fixed | uses jnp.diff(t) for trapezoid interval widths |
| predictx/gate3_diagnosis.py::np_ombe_volume | NumPy OMBE volume used by clinical/FVC/policy point gates | already-correct | calls cumulative_trapezoid(...), which uses np.diff(x) |
| predictx/gate_clinical_classification_f.py::fit_selected_rows | clinical classification point estimates | already-correct-through-import | imports fit_ombe_volume_lm from gate3_diagnosis.py |
| predictx/gate_fvc_correction_f.py::run | FVC convention gate | already-correct-through-clinical-rows | uses gate_clinical_classification_F_rows.csv and fit_selected_rows |
| predictx/gate_policy_calibrated_flag_f.py::run | bias correction + ATS flag policy | already-correct-through-import | imports fit_selected_rows from gate_clinical_classification_f.py |
| predictx/gate_likelihood_repair_f.py and posterior gates | JAX likelihood/posterior paths | fixed-by-shared-ombe | these import predictx.ombe.ombe_volume, now nonuniform-grid safe |

Nonuniform-grid regression assertion: PASS. Correct final volume=4.000000 L; constant-dt final volume would be 0.209169 L.

Important reconciliation: the shared JAX volume helper was broken before this patch, but the Cycle-F point-estimate gates audited here already used `gate3_diagnosis.np_ombe_volume`, which integrates with `np.diff(t)`. Therefore old-vs-new point-estimate changes are expected to be zero or numerical noise.

## Step 1 - Clinical Classification Old vs New

| definition | metric | old | new | delta |
|---|---|---:|---:|---:|
| fixed | n | 414.0000 | 414.0000 | +0.0000 |
| fixed | tp | 112.0000 | 112.0000 | +0.0000 |
| fixed | tn | 287.0000 | 287.0000 | +0.0000 |
| fixed | fp | 13.0000 | 13.0000 | +0.0000 |
| fixed | fn | 2.0000 | 2.0000 | +0.0000 |
| fixed | sensitivity | 0.9825 | 0.9825 | +0.0000 |
| fixed | specificity | 0.9567 | 0.9567 | +0.0000 |
| fixed | agreement | 0.9638 | 0.9638 | +0.0000 |
| fixed | cohens_kappa | 0.9118 | 0.9118 | +0.0000 |
| lln | n | 412.0000 | 412.0000 | +0.0000 |
| lln | tp | 114.0000 | 114.0000 | +0.0000 |
| lln | tn | 276.0000 | 276.0000 | +0.0000 |
| lln | fp | 18.0000 | 18.0000 | +0.0000 |
| lln | fn | 4.0000 | 4.0000 | +0.0000 |
| lln | sensitivity | 0.9661 | 0.9661 | +0.0000 |
| lln | specificity | 0.9388 | 0.9388 | +0.0000 |
| lln | agreement | 0.9466 | 0.9466 | +0.0000 |
| lln | cohens_kappa | 0.8738 | 0.8738 | +0.0000 |

### Obstructed-Set Errors

| metric | old median | new median | delta | old IQR | new IQR |
|---|---:|---:|---:|---:|---:|
| fev1_abs_pct | 1.3758 | 1.3758 | +0.0000 | [0.6150267820737756, 3.3149471155258303] | [0.6150267820737756, 3.3149471155258303] |
| fev1_signed_pct | -1.1576 | -1.1576 | +0.0000 | [-3.3149471155258303, 0.0591870551884093] | [-3.3149471155258303, 0.0591870551884093] |
| fvc_abs_pct | 5.7793 | 5.7793 | +0.0000 | [3.499868794265809, 8.161635330424595] | [3.499868794265809, 8.161635330424595] |
| fvc_signed_pct | -5.1777 | -5.1777 | +0.0000 | [-7.929056742066506, -2.7313420706037506] | [-7.929056742066506, -2.7313420706037506] |
| ratio_abs | 0.0219 | 0.0219 | +0.0000 | [0.011916743563101947, 0.03827251796730441] | [0.011916743563101947, 0.03827251796730441] |
| ratio_signed | 0.0197 | 0.0197 | +0.0000 | [0.007252559074206771, 0.03269159675570968] | [0.007252559074206771, 0.03269159675570968] |

### Severity Confusion

Old severity table:

```json
{
  "severe": {
    "severe": 24,
    "moderate": 7,
    "mild_0.60_0.70": 0,
    "normal": 0
  },
  "moderate": {
    "severe": 5,
    "moderate": 46,
    "mild_0.60_0.70": 30,
    "normal": 2
  },
  "mild_0.60_0.70": {
    "severe": 0,
    "moderate": 0,
    "mild_0.60_0.70": 0,
    "normal": 0
  },
  "normal": {
    "severe": 0,
    "moderate": 7,
    "mild_0.60_0.70": 6,
    "normal": 287
  }
}
```

New severity table:

```json
{
  "severe": {
    "severe": 24,
    "moderate": 7,
    "mild_0.60_0.70": 0,
    "normal": 0
  },
  "moderate": {
    "severe": 5,
    "moderate": 46,
    "mild_0.60_0.70": 30,
    "normal": 2
  },
  "mild_0.60_0.70": {
    "severe": 0,
    "moderate": 0,
    "mild_0.60_0.70": 0,
    "normal": 0
  },
  "normal": {
    "severe": 0,
    "moderate": 7,
    "mild_0.60_0.70": 6,
    "normal": 287
  }
}
```

Severe demotion-to-normal old=0.000, new=0.000, delta=+0.000.

## Step 2 - FVC Convention Old vs New

| field | old | new |
|---|---:|---:|
| verdict | PARTIAL | PARTIAL |
| preferred variant | variant_b_observed_raw_eot | variant_b_observed_raw_eot |
| preferred signed ratio median | 0.0108 | 0.0108 |

### Obstructed Signed Error Mechanism

```json
{
  "old": {
    "group": "obstructed",
    "n": 114,
    "signed_fev1_pct_median": -1.1576281358070144,
    "signed_fev1_pct_iqr": [
      -3.3149471155258303,
      0.0591870551884093
    ],
    "signed_fvc_pct_median": -5.177661538237652,
    "signed_fvc_pct_iqr": [
      -7.929056742066506,
      -2.7313420706037506
    ],
    "signed_ratio_median": 0.019731369676320998,
    "signed_ratio_iqr": [
      0.007252559074206771,
      0.03269159675570968
    ]
  },
  "new": {
    "group": "obstructed",
    "n": 114,
    "signed_fev1_pct_median": -1.1576281358070144,
    "signed_fev1_pct_iqr": [
      -3.3149471155258303,
      0.0591870551884093
    ],
    "signed_fvc_pct_median": -5.177661538237652,
    "signed_fvc_pct_iqr": [
      -7.929056742066506,
      -2.7313420706037506
    ],
    "signed_ratio_median": 0.019731369676320998,
    "signed_ratio_iqr": [
      0.007252559074206771,
      0.03269159675570968
    ]
  }
}
```

### FVC Convention Comparison

| convention | old signed ratio median | new signed ratio median | delta | old IQR | new IQR |
|---|---:|---:|---:|---:|---:|
| baseline | 0.0197 | 0.0197 | +0.0000 | [0.007252559074206771, 0.03269159675570968] | [0.007252559074206771, 0.03269159675570968] |
| variant_a_model_raw_eot | 0.0197 | 0.0197 | +0.0000 | [0.00725255907420666, 0.032691596755709595] | [0.00725255907420666, 0.032691596755709595] |
| variant_b_observed_raw_eot | 0.0108 | 0.0108 | +0.0000 | [0.001798732374098319, 0.020027901275013096] | [0.001798732374098319, 0.020027901275013096] |

## Step 3 - Bias Correction + Flag Policy Old vs New

| correction parameter | old | new | delta |
|---|---:|---:|---:|
| c1 constant bias | 0.008637 | 0.008637 | +0.000000 |
| c2 intercept | -0.102225 | -0.102225 | +0.000000 |
| c2 slope | 0.141564 | 0.141564 | +0.000000 |

| block | metric | old | new | delta |
|---|---|---:|---:|---:|
| baseline | residual_fn | 0.2364 | 0.2364 | +0.0000 |
| baseline | borderline_fn | 0.2880 | 0.2880 | +0.0000 |
| baseline | normal_truth_flag_burden | 0.0000 | 0.0000 | +0.0000 |
| baseline | nonflag_sensitivity | 0.7636 | 0.7636 | +0.0000 |
| baseline | nonflag_specificity | 0.9440 | 0.9440 | +0.0000 |
| baseline | nonflag_kappa | 0.6909 | 0.6909 | +0.0000 |
| c1_correction_only | residual_fn | 0.1693 | 0.1693 | +0.0000 |
| c1_correction_only | borderline_fn | 0.2040 | 0.2040 | +0.0000 |
| c1_correction_only | normal_truth_flag_burden | 0.0000 | 0.0000 | +0.0000 |
| c1_correction_only | nonflag_sensitivity | 0.8307 | 0.8307 | +0.0000 |
| c1_correction_only | nonflag_specificity | 0.9400 | 0.9400 | +0.0000 |
| c1_correction_only | nonflag_kappa | 0.7591 | 0.7591 | +0.0000 |
| c1_flag_unbanded | residual_fn | 0.1597 | 0.1597 | +0.0000 |
| c1_flag_unbanded | borderline_fn | 0.1920 | 0.1920 | +0.0000 |
| c1_flag_unbanded | normal_truth_flag_burden | 0.1280 | 0.1280 | +0.0000 |
| c1_flag_unbanded | nonflag_sensitivity | 0.8387 | 0.8387 | +0.0000 |
| c1_flag_unbanded | nonflag_specificity | 0.9312 | 0.9312 | +0.0000 |
| c1_flag_unbanded | nonflag_kappa | 0.7520 | 0.7520 | +0.0000 |
| c2_correction_only | residual_fn | 0.2780 | 0.2780 | +0.0000 |
| c2_correction_only | borderline_fn | 0.3400 | 0.3400 | +0.0000 |
| c2_correction_only | normal_truth_flag_burden | 0.0000 | 0.0000 | +0.0000 |
| c2_correction_only | nonflag_sensitivity | 0.7220 | 0.7220 | +0.0000 |
| c2_correction_only | nonflag_specificity | 0.9480 | 0.9480 | +0.0000 |
| c2_correction_only | nonflag_kappa | 0.6507 | 0.6507 | +0.0000 |
| c2_flag_unbanded | residual_fn | 0.2556 | 0.2556 | +0.0000 |
| c2_flag_unbanded | borderline_fn | 0.3120 | 0.3120 | +0.0000 |
| c2_flag_unbanded | normal_truth_flag_burden | 0.1320 | 0.1320 | +0.0000 |
| c2_flag_unbanded | nonflag_sensitivity | 0.7386 | 0.7386 | +0.0000 |
| c2_flag_unbanded | nonflag_specificity | 0.9401 | 0.9401 | +0.0000 |
| c2_flag_unbanded | nonflag_kappa | 0.6495 | 0.6495 | +0.0000 |

## Step 4 - Reconciliation

a. Fixed-threshold sensitivity/specificity/kappa moved +0.000000, +0.000000, +0.000000 respectively.

b. Signed obstructed-set ratio median changed from 0.0197 to 0.0197 (delta +0.0000).

c. Obstructed signed FVC percent median changed from -5.18% to -5.18%.

d. Baseline TEST borderline residual false-negative rate changed from 0.288 to 0.288.

e. No point-estimate conclusion is retracted from this rerun because the point-estimate gates were already using the NumPy nonuniform-grid volume fitter. The review premise that clinical/FVC/policy used the broken JAX dt path is not supported by the code audit. Retract only the narrower implementation statement that the shared JAX OMBE volume helper was safe on nonuniform real-curve grids before this patch.

## Step 5 - Integrity Note

The c1 correction remains numerically identical in this rerun, so its empirical effect is unchanged. However, it should not be justified as a repair for the shared JAX constant-dt bug in these point-estimate gates; that bug did not feed this path. The defensible justification remains the observed held-out signed-ratio bias and ATS truncation behavior in the point-estimate gate itself.

HEADLINE: fixed sensitivity 0.982, fixed specificity 0.957, fixed kappa 0.912, signed ratio median 0.0197, policy baseline borderline FN 0.288.
