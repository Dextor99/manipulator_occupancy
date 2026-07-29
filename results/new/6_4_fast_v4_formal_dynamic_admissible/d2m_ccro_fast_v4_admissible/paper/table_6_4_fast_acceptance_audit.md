# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ee_crossing_fast | 0.15 | Dynamic | CCRO-fast-v4 | 10 | 1.00 | 0.60 | 0.60 | 1.00 | 1.00 | 0.60 | 1.00 | 0.60 | 0.0023 | -0.0381 | -0.0279 |
| ee_crossing_fast | 0.25 | Dynamic | CCRO-fast-v4 | 10 | 1.00 | 0.70 | 0.60 | 1.00 | 1.00 | 0.60 | 1.00 | 0.60 | 0.0022 | -0.0426 | -0.0349 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D2MF_000 | CCRO-fast-v4 | 0.0795 | 0.0907 | 0.0914 | 0.0007 | 0.0014 | 0.0107 | 1 | 1 | 1 | 1 | 1 | usable |
| D2MF_001 | CCRO-fast-v4 | 0.0574 | 0.0735 | 0.0724 | -0.0011 | -0.0176 | -0.0065 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_002 | CCRO-fast-v4 | 0.0449 | 0.0583 | 0.0622 | 0.0040 | -0.0278 | -0.0217 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_003 | CCRO-fast-v4 | 0.0719 | 0.0924 | 0.0940 | 0.0016 | 0.0040 | 0.0124 | 1 | 1 | 1 | 1 | 1 | usable |
| D2MF_004 | CCRO-fast-v4 | 0.0784 | 0.0932 | 0.0946 | 0.0014 | 0.0046 | 0.0132 | 1 | 1 | 1 | 1 | 1 | usable |
| D2MF_005 | CCRO-fast-v4 | 0.0637 | 0.0754 | 0.0771 | 0.0017 | -0.0129 | -0.0046 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_006 | CCRO-fast-v4 | 0.0627 | 0.0918 | 0.0912 | -0.0005 | 0.0012 | 0.0118 | 1 | 1 | 1 | 1 | 1 | usable |
| D2MF_007 | CCRO-fast-v4 | 0.0648 | 0.0917 | 0.0972 | 0.0054 | 0.0072 | 0.0117 | 1 | 1 | 1 | 1 | 1 | usable |
| D2MF_008 | CCRO-fast-v4 | 0.0460 | 0.0811 | 0.0912 | 0.0101 | 0.0012 | 0.0011 | 1 | 1 | 1 | 1 | 1 | usable |
| D2MF_009 | CCRO-fast-v4 | 0.0436 | 0.0521 | 0.0519 | -0.0002 | -0.0381 | -0.0279 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_010 | CCRO-fast-v4 | 0.0541 | 0.0682 | 0.0682 | -0.0000 | -0.0218 | -0.0118 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_011 | CCRO-fast-v4 | 0.0580 | 0.0915 | 0.0930 | 0.0015 | 0.0030 | 0.0115 | 1 | 1 | 1 | 1 | 1 | usable |
| D2MF_012 | CCRO-fast-v4 | 0.0749 | 0.0914 | 0.0920 | 0.0006 | 0.0020 | 0.0114 | 1 | 1 | 1 | 1 | 1 | usable |
| D2MF_013 | CCRO-fast-v4 | 0.0609 | 0.0838 | 0.0924 | 0.0086 | 0.0024 | 0.0038 | 1 | 1 | 1 | 1 | 1 | usable |
| D2MF_014 | CCRO-fast-v4 | 0.0796 | 0.0936 | 0.0936 | 0.0000 | 0.0036 | 0.0136 | 1 | 1 | 1 | 1 | 1 | usable |
| D2MF_015 | CCRO-fast-v4 | 0.0522 | 0.0667 | 0.0692 | 0.0025 | -0.0208 | -0.0133 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_016 | CCRO-fast-v4 | 0.0741 | 0.0933 | 0.0938 | 0.0005 | 0.0038 | 0.0133 | 1 | 1 | 1 | 1 | 1 | usable |
| D2MF_017 | CCRO-fast-v4 | 0.0716 | 0.0927 | 0.0983 | 0.0056 | 0.0083 | 0.0127 | 1 | 1 | 1 | 1 | 1 | usable |
| D2MF_018 | CCRO-fast-v4 | 0.0401 | 0.0451 | 0.0474 | 0.0024 | -0.0426 | -0.0349 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_019 | CCRO-fast-v4 | 0.0578 | 0.0868 | 0.0874 | 0.0006 | -0.0026 | 0.0068 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
