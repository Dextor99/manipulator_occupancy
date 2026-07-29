# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing_fast | 0.15 | G1 | CCRO-fast-v4 | 10 | 1.00 | 0.90 | 0.90 | 1.00 | 1.00 | 0.90 | 1.00 | 0.90 | 0.0005 | -0.0128 | -0.0028 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D1F_000 | CCRO-fast-v4 | 0.0723 | 0.0915 | 0.0914 | -0.0001 | 0.0014 | 0.0115 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_001 | CCRO-fast-v4 | 0.0674 | 0.0924 | 0.0928 | 0.0003 | 0.0028 | 0.0124 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_002 | CCRO-fast-v4 | 0.0748 | 0.0914 | 0.0914 | 0.0000 | 0.0014 | 0.0114 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_003 | CCRO-fast-v4 | 0.0722 | 0.0918 | 0.0916 | -0.0002 | 0.0016 | 0.0118 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_004 | CCRO-fast-v4 | 0.0746 | 0.0921 | 0.0921 | 0.0000 | 0.0021 | 0.0121 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_005 | CCRO-fast-v4 | 0.0735 | 0.0924 | 0.0951 | 0.0027 | 0.0051 | 0.0124 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_006 | CCRO-fast-v4 | 0.0684 | 0.0911 | 0.0911 | 0.0000 | 0.0011 | 0.0111 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_007 | CCRO-fast-v4 | 0.0654 | 0.0772 | 0.0772 | 0.0000 | -0.0128 | -0.0028 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_008 | CCRO-fast-v4 | 0.0694 | 0.0932 | 0.0957 | 0.0025 | 0.0057 | 0.0132 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_009 | CCRO-fast-v4 | 0.0731 | 0.0909 | 0.0905 | -0.0004 | 0.0005 | 0.0109 | 1 | 1 | 1 | 1 | 1 | usable |
