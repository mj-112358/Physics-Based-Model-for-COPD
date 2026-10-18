# Module 1 - Segmentation + Back-Extrapolation

- status: PASS
- synthetic curves: 200
- recording synthesis: 10-50 baseline samples, OMBE exhale, negative-going inhale, Gaussian flow noise sigma=0.03 L/s
- synthetic truth: true onset is the ATS-zeroed start of the exhale segment

## Criteria

- t0 within 2 samples: 200/200 (100.0%)
- FEV1 within 2%: 200/200 (100.0%)
- joint pass: 200/200

## Error Distributions

- t0 sample error median: -0.043
- t0 sample error IQR: -0.361 to 0.253
- abs FEV1 percent error median: 0.340%
- abs FEV1 percent error IQR: 0.189% to 0.588%
