# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ee_crossing_fast | 0.15 | Dynamic | Critical-fast-v4 | 10 | 1.00 | 0.00 | 0.00 | - | 0.70 | 0.00 | 1.00 | 0.00 | 0.0027 | -0.1927 | -0.1848 |
| ee_crossing_fast | 0.25 | Dynamic | Critical-fast-v4 | 10 | 1.00 | 0.00 | 0.00 | - | 0.80 | 0.00 | 1.00 | 0.00 | 0.0033 | -0.1705 | -0.1673 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D2MF_000 | Critical-fast-v4 | -0.1078 | -0.1035 | -0.0972 | 0.0062 | -0.1872 | -0.1835 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_001 | Critical-fast-v4 | -0.0896 | -0.0896 | -0.0850 | 0.0046 | -0.1750 | -0.1696 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_002 | Critical-fast-v4 | -0.0422 | -0.0409 | -0.0409 | -0.0000 | -0.1309 | -0.1209 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_003 | Critical-fast-v4 | -0.1048 | -0.1048 | -0.1027 | 0.0020 | -0.1927 | -0.1848 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_004 | Critical-fast-v4 | -0.0435 | -0.0447 | -0.0330 | 0.0117 | -0.1230 | -0.1247 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_005 | Critical-fast-v4 | -0.0189 | -0.0101 | -0.0081 | 0.0020 | -0.0981 | -0.0901 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_006 | Critical-fast-v4 | -0.0198 | -0.0198 | -0.0204 | -0.0006 | -0.1104 | -0.0998 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_007 | Critical-fast-v4 | -0.0316 | -0.0291 | -0.0296 | -0.0005 | -0.1196 | -0.1091 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_008 | Critical-fast-v4 | -0.0842 | -0.0842 | -0.0842 | 0.0000 | -0.1742 | -0.1642 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_009 | Critical-fast-v4 | -0.0614 | -0.0578 | -0.0568 | 0.0010 | -0.1468 | -0.1378 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_010 | Critical-fast-v4 | -0.0526 | -0.0467 | -0.0380 | 0.0087 | -0.1280 | -0.1267 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_011 | Critical-fast-v4 | -0.0975 | -0.0873 | -0.0797 | 0.0076 | -0.1697 | -0.1673 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_012 | Critical-fast-v4 | -0.0128 | -0.0128 | -0.0134 | -0.0006 | -0.1034 | -0.0928 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_013 | Critical-fast-v4 | -0.0538 | -0.0538 | -0.0518 | 0.0020 | -0.1418 | -0.1338 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_014 | Critical-fast-v4 | -0.0242 | -0.0234 | -0.0218 | 0.0016 | -0.1118 | -0.1034 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_015 | Critical-fast-v4 | -0.0588 | -0.0569 | -0.0520 | 0.0049 | -0.1420 | -0.1369 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_016 | Critical-fast-v4 | -0.0766 | -0.0697 | -0.0604 | 0.0093 | -0.1504 | -0.1497 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_017 | Critical-fast-v4 | -0.0834 | -0.0798 | -0.0805 | -0.0007 | -0.1705 | -0.1598 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_018 | Critical-fast-v4 | -0.0381 | -0.0352 | -0.0384 | -0.0032 | -0.1284 | -0.1152 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_019 | Critical-fast-v4 | -0.0829 | -0.0823 | -0.0789 | 0.0035 | -0.1689 | -0.1623 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
