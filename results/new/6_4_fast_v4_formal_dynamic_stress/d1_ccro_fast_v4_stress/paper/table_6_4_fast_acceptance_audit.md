# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing_fast | 0.15 | Dynamic | CCRO-fast-v4 | 10 | 1.00 | 0.00 | 0.00 | - | 0.30 | 0.00 | 1.00 | 0.00 | 0.0024 | -0.1040 | -0.0940 |
| body_crossing_fast | 0.25 | Dynamic | CCRO-fast-v4 | 10 | 1.00 | 0.00 | 0.00 | - | 0.50 | 0.00 | 1.00 | 0.00 | 0.0024 | -0.0982 | -0.0956 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D1F_000 | CCRO-fast-v4 | 0.0092 | 0.0092 | 0.0162 | 0.0070 | -0.0738 | -0.0708 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_001 | CCRO-fast-v4 | 0.0314 | 0.0319 | 0.0323 | 0.0004 | -0.0577 | -0.0481 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_002 | CCRO-fast-v4 | 0.0204 | 0.0204 | 0.0274 | 0.0071 | -0.0626 | -0.0596 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_003 | CCRO-fast-v4 | 0.0179 | 0.0188 | 0.0203 | 0.0015 | -0.0697 | -0.0612 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_004 | CCRO-fast-v4 | 0.0073 | 0.0073 | 0.0073 | 0.0000 | -0.0827 | -0.0727 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_005 | CCRO-fast-v4 | -0.0140 | -0.0140 | -0.0140 | 0.0000 | -0.1040 | -0.0940 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_006 | CCRO-fast-v4 | 0.0384 | 0.0384 | 0.0384 | 0.0000 | -0.0516 | -0.0416 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_007 | CCRO-fast-v4 | 0.0292 | 0.0323 | 0.0350 | 0.0027 | -0.0550 | -0.0477 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_008 | CCRO-fast-v4 | 0.0220 | 0.0220 | 0.0243 | 0.0023 | -0.0657 | -0.0580 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_009 | CCRO-fast-v4 | 0.0284 | 0.0284 | 0.0311 | 0.0027 | -0.0589 | -0.0516 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_010 | CCRO-fast-v4 | 0.0103 | 0.0103 | 0.0102 | -0.0001 | -0.0798 | -0.0697 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_011 | CCRO-fast-v4 | -0.0008 | 0.0008 | 0.0012 | 0.0003 | -0.0888 | -0.0792 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_012 | CCRO-fast-v4 | 0.0281 | 0.0293 | 0.0353 | 0.0060 | -0.0547 | -0.0507 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_013 | CCRO-fast-v4 | -0.0111 | -0.0096 | -0.0081 | 0.0015 | -0.0981 | -0.0896 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_014 | CCRO-fast-v4 | 0.0315 | 0.0315 | 0.0315 | 0.0000 | -0.0585 | -0.0485 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_015 | CCRO-fast-v4 | 0.0279 | 0.0279 | 0.0280 | 0.0000 | -0.0620 | -0.0521 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_016 | CCRO-fast-v4 | 0.0357 | 0.0357 | 0.0357 | 0.0000 | -0.0543 | -0.0443 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_017 | CCRO-fast-v4 | -0.0071 | -0.0071 | 0.0035 | 0.0106 | -0.0865 | -0.0871 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_018 | CCRO-fast-v4 | -0.0173 | -0.0156 | -0.0082 | 0.0074 | -0.0982 | -0.0956 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_019 | CCRO-fast-v4 | 0.0067 | 0.0125 | 0.0112 | -0.0013 | -0.0788 | -0.0675 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
