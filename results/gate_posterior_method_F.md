# Gate - Full Posterior Decision Method, Cycle F

- VERDICT: WEAK - Posterior calibration did not clear on TEST after the train-only <=2-param recalibration map; posterior classification halted.
- posterior engine: sbi SNPE_C posterior_nn(model='nsf') on OMBE volume-summary context
- posterior draws per curve: 400

## Sample And Split

- split counts: {'test': {'mild_0.60_0.70': 250, 'normal': 250, 'moderate': 47, 'severe': 16}, 'train': {'mild_0.60_0.70': 250, 'normal': 250, 'moderate': 47, 'severe': 15}}
- posterior fit failures: 0/1125
- posterior production wall-clock: 15.0s

## Step 1 - Posterior Calibration Gate

- raw TEST calibration: {'n': 563, 'coverage': {'0.5': 0.325044404973357, '0.8': 0.6785079928952042, '0.95': 0.8880994671403197}, 'pit_counts': [109, 125, 89, 45, 26, 29, 32, 20, 20, 68], 'pit_chisquare_p': 2.134518818573328e-47, 'pit_mean': 0.3715230905861457, 'gross_skew': True, 'calibrated': False}
- recalibration map: {'shift': -0.01780360164831385, 'scale': 1.2, 'train_score': 0.12864768683274008, 'train_coverage': {'n': 562, 'coverage': {'0.5': 0.5729537366548043, '0.8': 0.8042704626334519, '0.95': 0.8985765124555161}, 'pit_counts': [19, 48, 72, 85, 83, 62, 40, 27, 33, 93], 'pit_chisquare_p': 5.990037738237564e-20, 'pit_mean': 0.5186120996441281, 'gross_skew': True, 'calibrated': False}}
- recalibrated TEST calibration: {'n': 563, 'coverage': {'0.5': 0.5310834813499112, '0.8': 0.7904085257548845, '0.95': 0.8827708703374778}, 'pit_counts': [21, 54, 76, 81, 67, 57, 36, 37, 37, 97], 'pit_chisquare_p': 6.537096680682404e-16, 'pit_mean': 0.5236856127886323, 'gross_skew': True, 'calibrated': False}

Posterior classification halted because calibration did not clear.

VERDICT: WEAK - Posterior calibration did not clear on TEST after the train-only <=2-param recalibration map; posterior classification halted.
