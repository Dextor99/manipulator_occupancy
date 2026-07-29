# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ee_crossing_fast | 0.15 | Dynamic | Critical-fast-v4 | 10 | 1.00 | 0.50 | 0.20 | 1.00 | 1.00 | 0.20 | 1.00 | 0.20 | 0.0025 | -0.0455 | -0.0348 |
| ee_crossing_fast | 0.25 | Dynamic | Critical-fast-v4 | 10 | 1.00 | 0.70 | 0.70 | 1.00 | 0.90 | 0.70 | 1.00 | 0.70 | 0.0039 | -0.0498 | -0.0399 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D2MF_000 | Critical-fast-v4 | 0.0795 | 0.0892 | 0.0883 | -0.0009 | -0.0017 | 0.0092 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D2MF_001 | Critical-fast-v4 | 0.0574 | 0.0734 | 0.0722 | -0.0012 | -0.0178 | -0.0066 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_002 | Critical-fast-v4 | 0.0449 | 0.0572 | 0.0573 | 0.0002 | -0.0327 | -0.0228 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_003 | Critical-fast-v4 | 0.0719 | 0.0875 | 0.0877 | 0.0002 | -0.0023 | 0.0075 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D2MF_004 | Critical-fast-v4 | 0.0784 | 0.1026 | 0.1069 | 0.0043 | 0.0169 | 0.0226 | 1 | 1 | 1 | 1 | 1 | usable |
| D2MF_005 | Critical-fast-v4 | 0.0637 | 0.0738 | 0.0739 | 0.0001 | -0.0161 | -0.0062 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_006 | Critical-fast-v4 | 0.0627 | 0.0865 | 0.0862 | -0.0003 | -0.0038 | 0.0065 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D2MF_007 | Critical-fast-v4 | 0.0648 | 0.0898 | 0.1037 | 0.0139 | 0.0137 | 0.0098 | 1 | 1 | 1 | 1 | 1 | usable |
| D2MF_008 | Critical-fast-v4 | 0.0460 | 0.0627 | 0.0719 | 0.0092 | -0.0181 | -0.0173 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_009 | Critical-fast-v4 | 0.0436 | 0.0452 | 0.0445 | -0.0007 | -0.0455 | -0.0348 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_010 | Critical-fast-v4 | 0.0541 | 0.0602 | 0.0605 | 0.0002 | -0.0295 | -0.0198 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_011 | Critical-fast-v4 | 0.0580 | 0.0884 | 0.0935 | 0.0051 | 0.0035 | 0.0084 | 1 | 1 | 1 | 1 | 1 | usable |
| D2MF_012 | Critical-fast-v4 | 0.0749 | 0.0908 | 0.0908 | -0.0000 | 0.0008 | 0.0108 | 1 | 1 | 1 | 1 | 1 | usable |
| D2MF_013 | Critical-fast-v4 | 0.0609 | 0.0843 | 0.0936 | 0.0093 | 0.0036 | 0.0043 | 1 | 1 | 1 | 1 | 1 | usable |
| D2MF_014 | Critical-fast-v4 | 0.0796 | 0.0994 | 0.0984 | -0.0009 | 0.0084 | 0.0194 | 1 | 1 | 1 | 1 | 1 | usable |
| D2MF_015 | Critical-fast-v4 | 0.0522 | 0.0652 | 0.0683 | 0.0031 | -0.0217 | -0.0148 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_016 | Critical-fast-v4 | 0.0741 | 0.0939 | 0.0959 | 0.0020 | 0.0059 | 0.0139 | 1 | 1 | 1 | 1 | 1 | usable |
| D2MF_017 | Critical-fast-v4 | 0.0716 | 0.1034 | 0.1123 | 0.0089 | 0.0223 | 0.0234 | 1 | 1 | 1 | 1 | 1 | usable |
| D2MF_018 | Critical-fast-v4 | 0.0401 | 0.0401 | 0.0402 | 0.0001 | -0.0498 | -0.0399 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_019 | Critical-fast-v4 | 0.0578 | 0.0849 | 0.0957 | 0.0108 | 0.0057 | 0.0049 | 1 | 1 | 1 | 1 | 1 | usable |
