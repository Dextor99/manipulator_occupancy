# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ee_crossing_fast | 0.15 | Dynamic | CCRO-fast-v4 | 10 | 1.00 | 0.00 | 0.00 | - | 0.80 | 0.00 | 1.00 | 0.00 | 0.0016 | -0.1923 | -0.1838 |
| ee_crossing_fast | 0.25 | Dynamic | CCRO-fast-v4 | 10 | 1.00 | 0.00 | 0.00 | - | 1.00 | 0.00 | 1.00 | 0.00 | 0.0024 | -0.1779 | -0.1721 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D2MF_000 | CCRO-fast-v4 | -0.1078 | -0.1038 | -0.1023 | 0.0015 | -0.1923 | -0.1838 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_001 | CCRO-fast-v4 | -0.0896 | -0.0896 | -0.0850 | 0.0046 | -0.1750 | -0.1696 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_002 | CCRO-fast-v4 | -0.0422 | -0.0387 | -0.0367 | 0.0019 | -0.1267 | -0.1187 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_003 | CCRO-fast-v4 | -0.1048 | -0.1007 | -0.1005 | 0.0002 | -0.1905 | -0.1807 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_004 | CCRO-fast-v4 | -0.0435 | -0.0221 | -0.0200 | 0.0021 | -0.1100 | -0.1021 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_005 | CCRO-fast-v4 | -0.0189 | -0.0118 | -0.0117 | 0.0000 | -0.1017 | -0.0918 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_006 | CCRO-fast-v4 | -0.0198 | -0.0100 | -0.0100 | -0.0001 | -0.1000 | -0.0900 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_007 | CCRO-fast-v4 | -0.0316 | -0.0289 | -0.0257 | 0.0032 | -0.1157 | -0.1089 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_008 | CCRO-fast-v4 | -0.0842 | -0.0842 | -0.0842 | 0.0000 | -0.1742 | -0.1642 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D2MF_009 | CCRO-fast-v4 | -0.0614 | -0.0490 | -0.0468 | 0.0022 | -0.1368 | -0.1290 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_010 | CCRO-fast-v4 | -0.0526 | -0.0314 | -0.0312 | 0.0003 | -0.1212 | -0.1114 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_011 | CCRO-fast-v4 | -0.0975 | -0.0921 | -0.0879 | 0.0042 | -0.1779 | -0.1721 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_012 | CCRO-fast-v4 | -0.0128 | -0.0010 | -0.0026 | -0.0017 | -0.0926 | -0.0810 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_013 | CCRO-fast-v4 | -0.0538 | -0.0357 | -0.0343 | 0.0014 | -0.1243 | -0.1157 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_014 | CCRO-fast-v4 | -0.0242 | -0.0234 | -0.0218 | 0.0016 | -0.1118 | -0.1034 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_015 | CCRO-fast-v4 | -0.0588 | -0.0551 | -0.0515 | 0.0036 | -0.1415 | -0.1351 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_016 | CCRO-fast-v4 | -0.0766 | -0.0746 | -0.0671 | 0.0075 | -0.1571 | -0.1546 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_017 | CCRO-fast-v4 | -0.0834 | -0.0686 | -0.0664 | 0.0022 | -0.1564 | -0.1486 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_018 | CCRO-fast-v4 | -0.0381 | -0.0237 | -0.0226 | 0.0011 | -0.1126 | -0.1037 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D2MF_019 | CCRO-fast-v4 | -0.0829 | -0.0823 | -0.0788 | 0.0035 | -0.1688 | -0.1623 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
