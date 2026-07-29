# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ee_crossing_fast | 0.15 | Dynamic | CCRO-fast-v4 | 10 | 1.00 | 0.00 | 0.00 | - | 0.30 | 0.00 | 1.00 | 0.00 | 0.0015 | -0.1046 | -0.0966 |
| ee_crossing_fast | 0.25 | Dynamic | CCRO-fast-v4 | 10 | 1.00 | 0.00 | 0.00 | - | 0.50 | 0.00 | 1.00 | 0.00 | 0.0026 | -0.0888 | -0.0788 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D2MF_000 | CCRO-fast-v4 | 0.0256 | 0.0316 | 0.0328 | 0.0012 | -0.0572 | -0.0484 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_001 | CCRO-fast-v4 | 0.0084 | 0.0084 | 0.0084 | 0.0000 | -0.0816 | -0.0716 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_002 | CCRO-fast-v4 | -0.0166 | -0.0166 | -0.0146 | 0.0019 | -0.1046 | -0.0966 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_003 | CCRO-fast-v4 | 0.0240 | 0.0241 | 0.0237 | -0.0004 | -0.0663 | -0.0559 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_004 | CCRO-fast-v4 | -0.0016 | -0.0016 | 0.0036 | 0.0052 | -0.0864 | -0.0816 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_005 | CCRO-fast-v4 | -0.0122 | -0.0122 | -0.0105 | 0.0018 | -0.1005 | -0.0922 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_006 | CCRO-fast-v4 | -0.0082 | -0.0082 | -0.0082 | 0.0000 | -0.0982 | -0.0882 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_007 | CCRO-fast-v4 | -0.0028 | 0.0076 | 0.0102 | 0.0026 | -0.0798 | -0.0724 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_008 | CCRO-fast-v4 | 0.0390 | 0.0390 | 0.0415 | 0.0025 | -0.0485 | -0.0410 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_009 | CCRO-fast-v4 | 0.0228 | 0.0228 | 0.0226 | -0.0001 | -0.0674 | -0.0572 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_010 | CCRO-fast-v4 | -0.0047 | 0.0115 | 0.0106 | -0.0010 | -0.0794 | -0.0685 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_011 | CCRO-fast-v4 | 0.0396 | 0.0596 | 0.0658 | 0.0062 | -0.0242 | -0.0204 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_012 | CCRO-fast-v4 | -0.0147 | 0.0041 | 0.0072 | 0.0031 | -0.0828 | -0.0759 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_013 | CCRO-fast-v4 | 0.0086 | 0.0193 | 0.0306 | 0.0114 | -0.0594 | -0.0607 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_014 | CCRO-fast-v4 | 0.0259 | 0.0421 | 0.0442 | 0.0021 | -0.0458 | -0.0379 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_015 | CCRO-fast-v4 | 0.0397 | 0.0397 | 0.0397 | 0.0000 | -0.0503 | -0.0403 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_016 | CCRO-fast-v4 | 0.0321 | 0.0321 | 0.0360 | 0.0039 | -0.0540 | -0.0479 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_017 | CCRO-fast-v4 | 0.0012 | 0.0012 | 0.0012 | 0.0000 | -0.0888 | -0.0788 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_018 | CCRO-fast-v4 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | -0.0885 | -0.0785 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_019 | CCRO-fast-v4 | 0.0216 | 0.0216 | 0.0216 | 0.0000 | -0.0684 | -0.0584 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
