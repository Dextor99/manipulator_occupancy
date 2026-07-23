GT recheck density: `dense`, time step: `0.04` s, d_stop: `0.08` m.

| scenario | method | n | checked samples | violation rate | Dmin mean / m | Dmin min / m | Dmin p05 / m |
|---|---|---:|---:|---:|---:|---:|---:|
| body_crossing | ccro_nubs | 15 | 4554 | 0.07 | 0.183 | 0.060 | 0.080 |
| body_crossing | critical_point_nubs | 15 | 5219 | 0.40 | 0.127 | -0.025 | -0.004 |
| body_crossing | reference_only | 15 | 1370 | 1.00 | 0.076 | 0.072 | 0.073 |
| body_crossing | ssm | 15 | 7236 | 0.93 | 0.006 | -0.026 | -0.025 |
| body_crossing | ssm_apf | 15 | 7236 | 0.93 | 0.008 | -0.025 | -0.025 |
| ee_crossing | ccro_nubs | 15 | 6037 | 0.60 | 0.043 | -0.051 | -0.049 |
| ee_crossing | critical_point_nubs | 15 | 6362 | 0.73 | 0.023 | -0.051 | -0.043 |
| ee_crossing | reference_only | 15 | 2702 | 0.20 | 0.090 | 0.074 | 0.075 |
| ee_crossing | ssm | 15 | 6047 | 0.67 | 0.035 | -0.050 | -0.043 |
| ee_crossing | ssm_apf | 15 | 6342 | 0.67 | 0.033 | -0.048 | -0.038 |
| far_safe | ccro_nubs | 10 | 2000 | 0.00 | 0.662 | 0.614 | 0.621 |
| initial_high_risk | ccro_nubs | 10 | 5010 | 1.00 | -0.048 | -0.053 | -0.052 |
