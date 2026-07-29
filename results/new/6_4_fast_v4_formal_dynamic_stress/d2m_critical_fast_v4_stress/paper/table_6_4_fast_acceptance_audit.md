# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ee_crossing_fast | 0.15 | Dynamic | Critical-fast-v4 | 10 | 1.00 | 0.00 | 0.00 | - | 0.60 | 0.00 | 1.00 | 0.00 | 0.0013 | -0.1046 | -0.0966 |
| ee_crossing_fast | 0.25 | Dynamic | Critical-fast-v4 | 10 | 1.00 | 0.00 | 0.00 | - | 0.90 | 0.00 | 1.00 | 0.00 | 0.0016 | -0.0955 | -0.0837 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D2MF_000 | Critical-fast-v4 | 0.0256 | 0.0316 | 0.0328 | 0.0012 | -0.0572 | -0.0484 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_001 | Critical-fast-v4 | 0.0084 | 0.0084 | 0.0084 | 0.0000 | -0.0816 | -0.0716 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_002 | Critical-fast-v4 | -0.0166 | -0.0166 | -0.0146 | 0.0019 | -0.1046 | -0.0966 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_003 | Critical-fast-v4 | 0.0240 | 0.0241 | 0.0240 | -0.0002 | -0.0660 | -0.0559 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_004 | Critical-fast-v4 | -0.0016 | -0.0016 | 0.0036 | 0.0052 | -0.0864 | -0.0816 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_005 | Critical-fast-v4 | -0.0122 | -0.0122 | -0.0105 | 0.0018 | -0.1005 | -0.0922 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_006 | Critical-fast-v4 | -0.0082 | -0.0082 | -0.0082 | 0.0000 | -0.0982 | -0.0882 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_007 | Critical-fast-v4 | -0.0028 | 0.0004 | 0.0004 | 0.0000 | -0.0896 | -0.0796 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_008 | Critical-fast-v4 | 0.0390 | 0.0390 | 0.0415 | 0.0025 | -0.0485 | -0.0410 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_009 | Critical-fast-v4 | 0.0228 | 0.0228 | 0.0230 | 0.0003 | -0.0670 | -0.0572 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_010 | Critical-fast-v4 | -0.0047 | -0.0037 | -0.0055 | -0.0018 | -0.0955 | -0.0837 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_011 | Critical-fast-v4 | 0.0396 | 0.0569 | 0.0569 | 0.0000 | -0.0331 | -0.0231 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_012 | Critical-fast-v4 | -0.0147 | 0.0044 | 0.0074 | 0.0029 | -0.0826 | -0.0756 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_013 | Critical-fast-v4 | 0.0086 | 0.0113 | 0.0201 | 0.0088 | -0.0699 | -0.0687 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_014 | Critical-fast-v4 | 0.0259 | 0.0307 | 0.0333 | 0.0026 | -0.0567 | -0.0493 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_015 | Critical-fast-v4 | 0.0397 | 0.0397 | 0.0397 | 0.0000 | -0.0503 | -0.0403 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_016 | Critical-fast-v4 | 0.0321 | 0.0321 | 0.0360 | 0.0039 | -0.0540 | -0.0479 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_017 | Critical-fast-v4 | 0.0012 | 0.0012 | 0.0012 | 0.0000 | -0.0888 | -0.0788 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_018 | Critical-fast-v4 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | -0.0885 | -0.0785 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_019 | Critical-fast-v4 | 0.0216 | 0.0216 | 0.0216 | 0.0000 | -0.0684 | -0.0584 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
