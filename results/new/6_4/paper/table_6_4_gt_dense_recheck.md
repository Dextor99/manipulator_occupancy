GT recheck density: `dense`, time step: `0.04` s, d_stop: `0.08` m.

| scenario | method | n | checked samples | violation rate | Dmin mean / m | Dmin min / m | Dmin p05 / m |
|---|---|---:|---:|---:|---:|---:|---:|
| body_crossing | ccro_nubs | 15 | 4864 | 0.07 | 0.178 | 0.006 | 0.064 |
| body_crossing | reference_only | 15 | 1370 | 1.00 | 0.076 | 0.072 | 0.073 |
| body_crossing | ssm | 15 | 7236 | 0.93 | 0.006 | -0.026 | -0.025 |
| ee_crossing | ccro_nubs | 15 | 5841 | 0.53 | 0.059 | -0.051 | -0.049 |
| ee_crossing | reference_only | 15 | 2702 | 0.20 | 0.090 | 0.074 | 0.075 |
| ee_crossing | ssm | 15 | 6047 | 0.67 | 0.035 | -0.050 | -0.043 |
| far_safe | ccro_nubs | 10 | 2000 | 0.00 | 0.662 | 0.614 | 0.621 |
| initial_high_risk | ccro_nubs | 10 | 5010 | 1.00 | -0.048 | -0.053 | -0.052 |
