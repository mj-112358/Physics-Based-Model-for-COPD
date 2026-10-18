# Gate - Two-Path Reconciliation, Cycle F

- VERDICT: BOTH-CORRECT-DIFFERENT-READOUT
- responsible difference: posterior path uses a downsampled fitting grid/objective; deterministic stride-grid LM moves theta and ratios relative to native-grid point LM
- no NUTS, no sampling, no downstream classification rerun
- sample: 20 obstructed Cycle-F selected manoeuvres with largest absolute signed ratio error under the NumPy point path

## Sample

| SEQN | raw_index | truth ratio | NumPy model ratio | NumPy signed error |
|---|---|---|---|---|
| 55814 | 14677 | 0.5464 | 0.2146 | -0.3318 |
| 61449 | 34068 | 0.5962 | 0.8985 | 0.3023 |
| 52119 | 1773 | 0.5316 | 0.2308 | -0.3008 |
| 56291 | 16322 | 0.3867 | 0.5817 | 0.1950 |
| 56985 | 18671 | 0.4315 | 0.2501 | -0.1814 |
| 56254 | 16214 | 0.4509 | 0.2996 | -0.1513 |
| 55047 | 11891 | 0.5043 | 0.3757 | -0.1285 |
| 60057 | 29127 | 0.5928 | 0.7179 | 0.1251 |
| 52630 | 3609 | 0.3786 | 0.2560 | -0.1226 |
| 59251 | 26302 | 0.5809 | 0.6774 | 0.0966 |
| 52635 | 3636 | 0.5563 | 0.4601 | -0.0962 |
| 56683 | 17715 | 0.5934 | 0.6833 | 0.0900 |
| 59863 | 28481 | 0.5606 | 0.6409 | 0.0803 |
| 54139 | 8605 | 0.5402 | 0.6185 | 0.0783 |
| 55189 | 12356 | 0.5636 | 0.6367 | 0.0731 |
| 54982 | 11657 | 0.5848 | 0.6549 | 0.0701 |
| 56816 | 18180 | 0.5724 | 0.6423 | 0.0699 |
| 54834 | 11082 | 0.3958 | 0.3302 | -0.0655 |
| 54267 | 9046 | 0.5649 | 0.6249 | 0.0600 |
| 53182 | 5463 | 0.5040 | 0.5638 | 0.0597 |

## Step 1 - Forward-Evaluation Equivalence

Fixed theta is the saved NumPy LM theta from `gate_clinical_classification_F_rows.csv`. NumPy path is `gate3_diagnosis.py::np_ombe_volume(theta, t)`. JAX path is `ombe.py::ombe_volume(phi, t)` after the `jnp.diff(t)` fix.

| raw_index | n | duration s | max |dV| mL | max diff t | dFEV1 mL | dFVC mL | dRatio |
|---|---|---|---|---|---|---|---|
| 14677 | 1314 | 13.128 | 0.000 | 13.128 | -0.000 | -0.000 | 0.000000 |
| 34068 | 1518 | 15.166 | 0.000 | 11.266 | -0.000 | 0.000 | -0.000000 |
| 1773 | 989 | 9.878 | 0.000 | 4.998 | -0.000 | -0.000 | 0.000000 |
| 16322 | 869 | 8.676 | 0.000 | 2.716 | 0.000 | 0.000 | 0.000000 |
| 18671 | 989 | 9.879 | 0.000 | 9.789 | 0.000 | 0.000 | -0.000000 |
| 16214 | 1380 | 13.783 | 0.000 | 4.433 | 0.000 | -0.000 | 0.000000 |
| 11891 | 1581 | 15.790 | 0.000 | 7.370 | 0.000 | -0.000 | 0.000000 |
| 29127 | 872 | 8.709 | 0.000 | 8.359 | 0.000 | 0.000 | -0.000000 |
| 3609 | 1018 | 10.168 | 0.000 | 5.858 | -0.000 | 0.000 | -0.000000 |
| 26302 | 1110 | 11.087 | 0.000 | 10.707 | 0.000 | -0.000 | 0.000000 |
| 3636 | 1441 | 14.399 | 0.000 | 10.929 | 0.000 | -0.000 | 0.000000 |
| 17715 | 1509 | 15.080 | 0.000 | 3.460 | 0.000 | 0.000 | 0.000000 |
| 28481 | 1186 | 11.845 | 0.000 | 11.305 | 0.000 | 0.000 | -0.000000 |
| 8605 | 1311 | 13.094 | 0.000 | 11.804 | 0.000 | -0.000 | 0.000000 |
| 12356 | 1722 | 17.202 | 0.000 | 13.162 | -0.000 | -0.000 | 0.000000 |
| 11657 | 1617 | 16.150 | 0.000 | 2.300 | -0.000 | -0.000 | -0.000000 |
| 18180 | 1024 | 10.222 | 0.000 | 5.192 | -0.000 | -0.000 | 0.000000 |
| 11082 | 1601 | 15.998 | 0.000 | 3.378 | 0.000 | 0.000 | 0.000000 |
| 9046 | 1203 | 12.014 | 0.000 | 8.064 | -0.000 | -0.000 | 0.000000 |
| 5463 | 1496 | 14.941 | 0.000 | 3.361 | -0.000 | -0.000 | 0.000000 |

Gate result: max volume difference >5 mL on any curve = False.

## Step 2 - Readout Equivalence

Point convention: `extract_curve_for_fit` applies segmentation/back-extrapolated t0, builds `vol_t`, then `clinical_metrics_from_volume` reads FEV1 by `np.interp(min(1.0, t[-1]), t, volume)`, FVC as `volume[-1]`, and ratio as FEV1/FVC.

Likelihood/TSNPE convention: likelihood gates evaluate posterior ratios on `OmbeGrid().t` through `gate_posterior_calibration_repair_f.ratio_samples_from_phi -> tsnpe.features_from_phi`, which uses a uniform-grid feature readout and returns FVC as the natural `fvc_l` parameter.

KOH convention: KOH fits/readouts use each curve's `vol_t[::grid_stride]` with `grid_stride=5`; ratio is FEV1 by interpolation on that downsampled grid divided by predicted volume at that grid's final sample.

| raw_index | truth | point ratio | likelihood ratio | likelihood dRatio | KOH ratio | KOH dRatio | likelihood dFEV1 mL | likelihood dFVC mL |
|---|---|---|---|---|---|---|---|---|
| 14677 | 0.5464 | 0.2146 | 0.2188 | 0.0042 | 0.2137 | -0.0009 | 8.49 | 0.00 |
| 34068 | 0.5962 | 0.8985 | 0.8985 | -0.0000 | 0.8966 | -0.0019 | -0.03 | 0.00 |
| 1773 | 0.5316 | 0.2308 | 0.2188 | -0.0120 | 0.2300 | -0.0008 | -23.90 | -0.00 |
| 16322 | 0.3867 | 0.5817 | 0.5561 | -0.0257 | 0.5788 | -0.0029 | -51.32 | -0.00 |
| 18671 | 0.4315 | 0.2501 | 0.2374 | -0.0126 | 0.2491 | -0.0010 | -25.27 | -0.00 |
| 16214 | 0.4509 | 0.2996 | 0.3074 | 0.0078 | 0.2982 | -0.0014 | 15.59 | -0.00 |
| 11891 | 0.5043 | 0.3757 | 0.3909 | 0.0151 | 0.3703 | -0.0055 | 30.28 | -0.00 |
| 29127 | 0.5928 | 0.7179 | 0.7055 | -0.0124 | 0.7141 | -0.0038 | -24.88 | -0.00 |
| 3609 | 0.3786 | 0.2560 | 0.2453 | -0.0107 | 0.2549 | -0.0011 | -21.45 | -0.00 |
| 26302 | 0.5809 | 0.6774 | 0.6740 | -0.0035 | 0.6746 | -0.0028 | -8.26 | -0.00 |
| 3636 | 0.5563 | 0.4601 | 0.4716 | 0.0115 | 0.4531 | -0.0070 | 23.04 | -0.00 |
| 17715 | 0.5934 | 0.6833 | 0.6833 | -0.0000 | 0.6808 | -0.0026 | -0.00 | 0.00 |
| 28481 | 0.5606 | 0.6409 | 0.6405 | -0.0004 | 0.6367 | -0.0042 | -0.87 | -0.00 |
| 8605 | 0.5402 | 0.6185 | 0.6220 | 0.0036 | 0.6152 | -0.0033 | 7.96 | -0.00 |
| 12356 | 0.5636 | 0.6367 | 0.6524 | 0.0157 | 0.6333 | -0.0034 | 64.76 | 0.00 |
| 11657 | 0.5848 | 0.6549 | 0.6549 | 0.0000 | 0.6525 | -0.0024 | 0.07 | 0.00 |
| 18180 | 0.5724 | 0.6423 | 0.6397 | -0.0026 | 0.6393 | -0.0030 | -6.36 | -0.00 |
| 11082 | 0.3958 | 0.3302 | 0.3451 | 0.0149 | 0.3283 | -0.0019 | 29.82 | -0.00 |
| 9046 | 0.5649 | 0.6249 | 0.6249 | 0.0000 | 0.6215 | -0.0034 | 0.03 | 0.00 |
| 5463 | 0.5040 | 0.5638 | 0.5735 | 0.0098 | 0.5604 | -0.0033 | 29.10 | -0.00 |

Gate result: likelihood ratio differs by >0.002 on any curve = True; KOH ratio differs by >0.002 on any curve = True.

## Step 3 - Fit-Objective Equivalence

JAX LM refit uses the same natural-parameter bounds and the same deterministic least-squares objective as the NumPy LM, but evaluates volume through corrected `ombe_volume`.

| raw_index | native success | native max |d%| | native dRatio | stride success | stride max |d%| | stride dRatio | stride-from-NumPy max |d%| | stride-from-NumPy dRatio | NumPy tau_o bound | native tau_o bound | stride tau_o bound |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 14677 | True | 0.0000 | 0.000000 | True | 0.0000 | 0.000000 | 0.0000 | 0.000000 | True | True | True |
| 34068 | True | 0.0004 | 0.000000 | True | 1.9217 | 0.002315 | 1.9312 | 0.002326 | True | True | True |
| 1773 | True | 0.0000 | 0.000000 | True | 0.0000 | 0.000000 | 0.0000 | 0.000000 | True | True | True |
| 16322 | True | 0.0001 | -0.000000 | True | 1.5578 | 0.003618 | 1.5571 | 0.003617 | True | True | True |
| 18671 | True | 0.0000 | 0.000000 | True | 0.6886 | 0.000919 | 0.6886 | 0.000919 | True | True | True |
| 16214 | True | 0.0000 | -0.000000 | True | 0.5590 | 0.001352 | 0.5589 | 0.001352 | True | True | True |
| 11891 | True | 0.0000 | -0.000000 | True | 2.9005 | 0.005751 | 2.9005 | 0.005751 | True | True | True |
| 29127 | True | 0.0000 | 0.000000 | True | 3.8260 | 0.004219 | 3.8234 | 0.004220 | True | True | True |
| 3609 | True | 0.0000 | 0.000000 | True | 0.6587 | 0.000954 | 0.6587 | 0.000954 | True | True | True |
| 26302 | True | 0.0000 | 0.000000 | True | 1.5671 | 0.003962 | 1.5671 | 0.003962 | True | True | True |
| 3636 | True | 0.0156 | -0.000000 | True | 30.7656 | 0.009155 | 30.7656 | 0.009155 | False | False | True |
| 17715 | True | 0.0004 | 0.000000 | True | 4.2334 | 0.002656 | 4.2330 | 0.002656 | True | True | True |
| 28481 | True | 0.0000 | -0.000000 | True | 3.6630 | 0.004475 | 3.6631 | 0.004475 | True | True | True |
| 8605 | True | 0.0000 | -0.000000 | True | 2.4836 | 0.004050 | 2.4821 | 0.004049 | True | True | True |
| 12356 | True | 0.0000 | 0.000000 | True | 1.8708 | 0.004196 | 1.8705 | 0.004196 | True | True | True |
| 11657 | True | 0.0004 | 0.000000 | True | 9.0977 | 0.002628 | 9.0918 | 0.002627 | False | False | False |
| 18180 | True | 0.0000 | 0.000000 | True | 3.2818 | 0.003544 | 3.2808 | 0.003544 | True | True | True |
| 11082 | True | 0.0000 | 0.000000 | True | 0.6319 | 0.001969 | 0.6319 | 0.001969 | True | True | True |
| 9046 | True | 0.0000 | 0.000000 | True | 3.1702 | 0.004142 | 3.1686 | 0.004142 | True | True | True |
| 5463 | True | 0.0000 | 0.000000 | True | 2.4864 | 0.003932 | 2.4851 | 0.003932 | True | True | True |

### Step 3 Per-Parameter Relative Differences

| raw_index | native tau_f d% | native tau_s d% | native alpha d% | native fvc d% | native tau_o d% | stride tau_f d% | stride tau_s d% | stride alpha d% | stride fvc d% | stride tau_o d% |
|---|---|---|---|---|---|---|---|---|---|---|
| 14677 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 34068 | -0.00 | 0.00 | -0.00 | 0.00 | 0.00 | -1.93 | -0.00 | 0.00 | 0.00 | -0.00 |
| 1773 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 16322 | 0.00 | 0.00 | 0.00 | 0.00 | -0.00 | -1.56 | 0.00 | 0.43 | 0.00 | -0.00 |
| 18671 | 0.00 | 0.00 | 0.00 | 0.00 | -0.00 | -0.00 | 0.00 | 0.69 | 0.00 | 0.00 |
| 16214 | -0.00 | 0.00 | -0.00 | 0.00 | 0.00 | -0.00 | -0.00 | 0.56 | 0.00 | 0.00 |
| 11891 | 0.00 | 0.00 | -0.00 | 0.00 | 0.00 | -0.00 | -0.00 | 2.90 | 0.00 | -0.00 |
| 29127 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | -3.82 | -2.89 | -0.16 | 0.00 | 0.00 |
| 3609 | -0.00 | 0.00 | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 | 0.66 | 0.00 | 0.00 |
| 26302 | -0.00 | 0.00 | -0.00 | 0.00 | 0.00 | -1.57 | -0.00 | 0.13 | 0.00 | 0.00 |
| 3636 | -0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.64 | 0.00 | -1.59 | 0.00 | -30.77 |
| 17715 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | -0.00 | -0.00 | 4.23 | -0.00 | -0.00 |
| 28481 | -0.00 | -0.00 | -0.00 | -0.00 | 0.00 | -3.66 | -1.09 | 0.48 | -0.03 | 0.00 |
| 8605 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | -2.48 | -1.08 | 0.14 | -0.02 | 0.00 |
| 12356 | -0.00 | -0.00 | -0.00 | 0.00 | 0.00 | -1.87 | 0.00 | 0.52 | 0.01 | -0.00 |
| 11657 | -0.00 | -0.00 | 0.00 | 0.00 | 0.00 | -0.00 | 0.12 | 0.00 | 0.00 | -9.09 |
| 18180 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | -3.28 | -1.22 | -0.21 | -0.04 | 0.00 |
| 11082 | 0.00 | -0.00 | 0.00 | 0.00 | -0.00 | -0.00 | -0.00 | 0.63 | 0.00 | -0.00 |
| 9046 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | -3.17 | -1.29 | 0.07 | -0.04 | 0.00 |
| 5463 | -0.00 | -0.00 | -0.00 | -0.00 | 0.00 | -2.49 | -0.66 | 0.43 | -0.02 | 0.00 |

Gate result: native theta differs by >2% on any curve = False; stride theta differs by >2% on any curve = True; stride initialized from NumPy theta differs by >2% = True.

## Step 4 - Grid-Stride Check

| raw_index | native n | stride n | native ratio | stride ratio | stride dRatio | fraction of 0.0178 gap | dFEV1 mL | dFVC mL |
|---|---|---|---|---|---|---|---|---|
| 14677 | 1314 | 263 | 0.2146 | 0.2137 | -0.00086 | 0.048 | -1.72 | -0.00 |
| 34068 | 1518 | 304 | 0.8985 | 0.8966 | -0.00186 | 0.104 | -11.14 | 0.00 |
| 1773 | 989 | 198 | 0.2308 | 0.2300 | -0.00080 | 0.045 | -1.59 | 0.00 |
| 16322 | 869 | 174 | 0.5817 | 0.5788 | -0.00290 | 0.163 | -5.81 | 0.00 |
| 18671 | 989 | 198 | 0.2501 | 0.2491 | -0.00098 | 0.055 | -1.95 | 0.00 |
| 16214 | 1380 | 276 | 0.2996 | 0.2982 | -0.00135 | 0.076 | -2.71 | -0.00 |
| 11891 | 1581 | 317 | 0.3757 | 0.3703 | -0.00546 | 0.307 | -10.91 | -0.00 |
| 29127 | 872 | 175 | 0.7179 | 0.7141 | -0.00379 | 0.213 | -7.59 | 0.00 |
| 3609 | 1018 | 204 | 0.2560 | 0.2549 | -0.00109 | 0.061 | -2.18 | 0.00 |
| 26302 | 1110 | 222 | 0.6774 | 0.6746 | -0.00283 | 0.159 | -6.73 | -0.00 |
| 3636 | 1441 | 289 | 0.4601 | 0.4531 | -0.00695 | 0.391 | -13.90 | -0.00 |
| 17715 | 1509 | 302 | 0.6833 | 0.6808 | -0.00257 | 0.144 | -15.39 | -0.00 |
| 28481 | 1186 | 238 | 0.6409 | 0.6367 | -0.00419 | 0.235 | -9.15 | 0.00 |
| 8605 | 1311 | 263 | 0.6185 | 0.6152 | -0.00327 | 0.184 | -7.29 | -0.00 |
| 12356 | 1722 | 345 | 0.6367 | 0.6333 | -0.00341 | 0.192 | -14.03 | -0.00 |
| 11657 | 1617 | 324 | 0.6549 | 0.6525 | -0.00240 | 0.135 | -14.38 | -0.00 |
| 18180 | 1024 | 205 | 0.6423 | 0.6393 | -0.00302 | 0.170 | -7.41 | 0.00 |
| 11082 | 1601 | 321 | 0.3302 | 0.3283 | -0.00189 | 0.106 | -3.79 | 0.00 |
| 9046 | 1203 | 241 | 0.6249 | 0.6215 | -0.00343 | 0.193 | -7.93 | -0.00 |
| 5463 | 1496 | 300 | 0.5638 | 0.5604 | -0.00334 | 0.188 | -9.95 | 0.00 |

Gate result: median absolute stride-only ratio difference >0.002 = True.

## Summary Statistics

```json
{
  "step1": {
    "max_abs_volume_diff_ml": {
      "n": 20,
      "median": 3.4416913763379853e-12,
      "iqr": [
        2.3869795029440866e-12,
        5.440092820663267e-12
      ],
      "max": 9.769962616701378e-12
    },
    "fev1_diff_ml_jax_minus_np": {
      "n": 20,
      "median": 0.0,
      "iqr": [
        -2.7755575615628914e-13,
        3.3306690738754696e-13
      ],
      "max": 1.5543122344752192e-12
    },
    "fvc_diff_ml_jax_minus_np": {
      "n": 20,
      "median": -8.881784197001252e-13,
      "iqr": [
        -2.0539125955565396e-12,
        5.551115123125783e-13
      ],
      "max": 5.551115123125783e-12
    },
    "ratio_diff_jax_minus_np": {
      "n": 20,
      "median": 8.326672684688674e-17,
      "iqr": [
        -1.1102230246251565e-16,
        2.498001805406602e-16
      ],
      "max": 9.992007221626409e-16
    }
  },
  "step2": {
    "likelihood_ratio_diff_vs_point": {
      "n": 20,
      "median": 5.88872465734358e-06,
      "iqr": [
        -0.00529091816276496,
        0.008288935875315942
      ],
      "max": 0.015740024655800045
    },
    "koh_ratio_diff_vs_point": {
      "n": 20,
      "median": -0.0028694814689601,
      "iqr": [
        -0.0034132708186180027,
        -0.001730638555659303
      ],
      "max": -0.0007960296896588615
    },
    "likelihood_fev1_diff_ml_vs_point": {
      "n": 20,
      "median": 0.016795777791456246,
      "iqr": [
        -11.560899972517912,
        17.452117280467416
      ],
      "max": 64.76177043019771
    },
    "likelihood_fvc_diff_ml_vs_point": {
      "n": 20,
      "median": -4.440892098500626e-13,
      "iqr": [
        -4.811528842019186e-06,
        5.551115123125783e-14
      ],
      "max": 0.00034369164847447564
    },
    "koh_fev1_diff_ml_vs_point": {
      "n": 20,
      "median": -7.500205316958719,
      "iqr": [
        -10.96970322359095,
        -3.515806830833829
      ],
      "max": -1.592059379317723
    },
    "koh_fvc_diff_ml_vs_point": {
      "n": 20,
      "median": 0.0,
      "iqr": [
        -4.440892098500626e-13,
        8.881784197001252e-13
      ],
      "max": 4.440892098500626e-12
    }
  },
  "step3": {
    "max_abs_native_param_rel_diff_pct": {
      "n": 20,
      "median": 1.38904233244713e-05,
      "iqr": [
        3.294063669676966e-06,
        6.160420095641748e-05
      ],
      "max": 0.015649957731134066
    },
    "ratio_diff_jax_native_lm_minus_numpy_lm": {
      "n": 20,
      "median": 8.093061776293098e-10,
      "iqr": [
        -2.18132009677241e-09,
        5.4721622094744404e-09
      ],
      "max": 4.5917836577480386e-07
    },
    "max_abs_stride_param_rel_diff_pct": {
      "n": 20,
      "median": 2.202604686410228,
      "iqr": [
        0.6811548693749564,
        3.3771124282627323
      ],
      "max": 30.76559508683676
    },
    "ratio_diff_jax_stride_lm_minus_numpy_lm": {
      "n": 20,
      "median": 0.003581010658264827,
      "iqr": [
        0.0018151956682603276,
        0.00415528212344668
      ],
      "max": 0.00915525185514049
    },
    "max_abs_stride_from_numpy_param_rel_diff_pct": {
      "n": 20,
      "median": 2.2066548793345833,
      "iqr": [
        0.6811520485744647,
        3.3763916759982067
      ],
      "max": 30.76559070683738
    },
    "ratio_diff_jax_stride_from_numpy_lm_minus_numpy_lm": {
      "n": 20,
      "median": 0.003580805007817711,
      "iqr": [
        0.001815168195317865,
        0.0041552532125315456
      ],
      "max": 0.009155251533059627
    }
  },
  "step4": {
    "ratio_diff_stride_minus_native": {
      "n": 20,
      "median": -0.0028694814689599335,
      "iqr": [
        -0.003413270818617864,
        -0.001730638555660649
      ],
      "max": -0.0007960296896589447
    },
    "abs_stride_ratio_diff_fraction_of_0p0178_gap": {
      "n": 20,
      "median": 0.16120682409887266,
      "iqr": [
        0.09722688514947467,
        0.19175678756280135
      ],
      "max": 0.3905809822870706
    },
    "fev1_diff_ml_stride_minus_native": {
      "n": 20,
      "median": -7.500205316958275,
      "iqr": [
        -10.969703223592585,
        -3.5158068308336343
      ],
      "max": -1.592059379317723
    },
    "fvc_diff_ml_stride_minus_native": {
      "n": 20,
      "median": -2.220446049250313e-13,
      "iqr": [
        -9.992007221626409e-13,
        9.992007221626409e-13
      ],
      "max": 7.327471962526033e-12
    }
  }
}
```

## Step 5 - Verdict and Consequence

Verdict: **BOTH-CORRECT-DIFFERENT-READOUT**.

posterior path uses a downsampled fitting grid/objective; deterministic stride-grid LM moves theta and ratios relative to native-grid point LM

Bias medians on the 20-curve stress sample: {'point_native_signed_ratio': 0.06497551660064843, 'likelihood_tsnpe_signed_ratio': 0.06842193237577254, 'koh_downsampled_signed_ratio': 0.061750823542583266, 'numpy_stride_signed_ratio': 0.06175082354258299}

Direct answer - c1: The c1 correction is a convention/readout patch for the point-estimate clinical convention, not evidence that the OMBE integrator itself is wrong.

Direct answer - tail undershoot: The tail-undershoot mechanism survives only for the point-gate native-grid clinical FVC convention; it is not supported as a universal OMBE/JAX posterior mechanism.
