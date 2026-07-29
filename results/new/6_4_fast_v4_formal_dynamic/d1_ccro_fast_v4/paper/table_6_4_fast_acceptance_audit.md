# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing_fast | 0.15 | Dynamic | CCRO-fast-v4 | 10 | 1.00 | 0.00 | 0.00 | - | 0.40 | 0.00 | 1.00 | 0.00 | 0.0031 | -0.1917 | -0.1833 |
| body_crossing_fast | 0.25 | Dynamic | CCRO-fast-v4 | 10 | 1.00 | 0.00 | 0.00 | - | 0.40 | 0.00 | 1.00 | 0.00 | 0.0011 | -0.1944 | -0.1799 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D1F_000 | CCRO-fast-v4 | -0.0943 | -0.0943 | -0.0854 | 0.0089 | -0.1754 | -0.1743 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_001 | CCRO-fast-v4 | -0.0994 | -0.0981 | -0.0966 | 0.0015 | -0.1866 | -0.1781 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_002 | CCRO-fast-v4 | -0.0983 | -0.0983 | -0.0991 | -0.0009 | -0.1891 | -0.1783 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_003 | CCRO-fast-v4 | -0.1014 | -0.1014 | -0.1017 | -0.0003 | -0.1917 | -0.1814 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_004 | CCRO-fast-v4 | -0.1008 | -0.0991 | -0.0979 | 0.0012 | -0.1879 | -0.1791 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_005 | CCRO-fast-v4 | -0.0318 | -0.0315 | -0.0205 | 0.0110 | -0.1105 | -0.1115 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_006 | CCRO-fast-v4 | -0.1035 | -0.1033 | -0.0996 | 0.0037 | -0.1896 | -0.1833 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_007 | CCRO-fast-v4 | -0.0190 | -0.0190 | -0.0178 | 0.0012 | -0.1078 | -0.0990 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_008 | CCRO-fast-v4 | -0.0879 | -0.0879 | -0.0844 | 0.0036 | -0.1744 | -0.1679 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_009 | CCRO-fast-v4 | -0.0731 | -0.0731 | -0.0715 | 0.0016 | -0.1615 | -0.1531 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_010 | CCRO-fast-v4 | -0.1037 | -0.0999 | -0.1044 | -0.0045 | -0.1944 | -0.1799 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_011 | CCRO-fast-v4 | -0.0911 | -0.0712 | -0.0712 | 0.0000 | -0.1612 | -0.1512 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_012 | CCRO-fast-v4 | -0.0743 | -0.0743 | -0.0744 | -0.0000 | -0.1644 | -0.1543 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_013 | CCRO-fast-v4 | -0.0964 | -0.0964 | -0.0978 | -0.0015 | -0.1878 | -0.1764 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_014 | CCRO-fast-v4 | -0.0302 | -0.0302 | -0.0270 | 0.0032 | -0.1170 | -0.1102 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_015 | CCRO-fast-v4 | -0.0846 | -0.0739 | -0.0728 | 0.0012 | -0.1628 | -0.1539 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_016 | CCRO-fast-v4 | -0.0996 | -0.0996 | -0.1019 | -0.0023 | -0.1919 | -0.1796 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_017 | CCRO-fast-v4 | -0.0323 | -0.0323 | -0.0204 | 0.0118 | -0.1104 | -0.1123 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_018 | CCRO-fast-v4 | -0.0845 | -0.0816 | -0.0802 | 0.0015 | -0.1702 | -0.1616 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_019 | CCRO-fast-v4 | -0.0253 | -0.0253 | -0.0236 | 0.0017 | -0.1136 | -0.1053 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
