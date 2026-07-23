GT recheck density: `dense`, time step: `0.04` s, d_stop: `0.08` m.

| scenario | method | n | checked samples | violation rate | Dmin mean / m | Dmin min / m | Dmin p05 / m |
|---|---|---:|---:|---:|---:|---:|---:|
| body_crossing | ccro_nubs | 15 | 3002 | 0.00 | 0.148 | 0.121 | 0.121 |
| body_crossing | reference_only | 15 | 1336 | 1.00 | 0.075 | 0.070 | 0.070 |
| body_crossing | ssm | 15 | 6966 | 0.67 | 0.044 | -0.010 | -0.008 |
| ee_crossing | ccro_nubs | 15 | 3001 | 0.00 | 0.138 | 0.120 | 0.122 |
| ee_crossing | reference_only | 15 | 1327 | 1.00 | 0.076 | 0.071 | 0.072 |
| ee_crossing | ssm | 15 | 7515 | 1.00 | -0.038 | -0.056 | -0.052 |
| far_safe | ccro_nubs | 10 | 2000 | 0.00 | 0.662 | 0.614 | 0.621 |
| initial_high_risk | ccro_nubs | 10 | 5010 | 1.00 | -0.048 | -0.053 | -0.052 |
