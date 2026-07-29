# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing_fast | 0.15 | Dynamic | Critical-fast-v4 | 10 | 1.00 | 0.00 | 0.00 | - | 0.20 | 0.00 | 1.00 | 0.00 | 0.0020 | -0.1917 | -0.1814 |
| body_crossing_fast | 0.25 | Dynamic | Critical-fast-v4 | 10 | 1.00 | 0.00 | 0.00 | - | 0.40 | 0.00 | 1.00 | 0.00 | -0.0000 | -0.1953 | -0.1800 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D1F_000 | Critical-fast-v4 | -0.0943 | -0.0943 | -0.0854 | 0.0089 | -0.1754 | -0.1743 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_001 | Critical-fast-v4 | -0.0994 | -0.0994 | -0.0994 | -0.0000 | -0.1894 | -0.1794 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_002 | Critical-fast-v4 | -0.0983 | -0.0983 | -0.0991 | -0.0009 | -0.1891 | -0.1783 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_003 | Critical-fast-v4 | -0.1014 | -0.1014 | -0.1017 | -0.0003 | -0.1917 | -0.1814 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_004 | Critical-fast-v4 | -0.1008 | -0.1008 | -0.1006 | 0.0002 | -0.1906 | -0.1808 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_005 | Critical-fast-v4 | -0.0318 | -0.0318 | -0.0196 | 0.0122 | -0.1096 | -0.1118 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_006 | Critical-fast-v4 | -0.1035 | -0.0918 | -0.0982 | -0.0064 | -0.1882 | -0.1718 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_007 | Critical-fast-v4 | -0.0190 | -0.0190 | -0.0178 | 0.0012 | -0.1078 | -0.0990 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_008 | Critical-fast-v4 | -0.0879 | -0.0879 | -0.0844 | 0.0036 | -0.1744 | -0.1679 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_009 | Critical-fast-v4 | -0.0731 | -0.0731 | -0.0715 | 0.0016 | -0.1615 | -0.1531 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_010 | Critical-fast-v4 | -0.1037 | -0.1000 | -0.1053 | -0.0053 | -0.1953 | -0.1800 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_011 | Critical-fast-v4 | -0.0911 | -0.0911 | -0.0925 | -0.0014 | -0.1825 | -0.1711 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_012 | Critical-fast-v4 | -0.0743 | -0.0743 | -0.0744 | -0.0000 | -0.1644 | -0.1543 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_013 | Critical-fast-v4 | -0.0964 | -0.0964 | -0.0978 | -0.0015 | -0.1878 | -0.1764 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_014 | Critical-fast-v4 | -0.0302 | -0.0241 | -0.0198 | 0.0043 | -0.1098 | -0.1041 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_015 | Critical-fast-v4 | -0.0846 | -0.0791 | -0.0866 | -0.0075 | -0.1766 | -0.1591 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_016 | Critical-fast-v4 | -0.0996 | -0.0996 | -0.1019 | -0.0023 | -0.1919 | -0.1796 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_017 | Critical-fast-v4 | -0.0323 | -0.0323 | -0.0204 | 0.0118 | -0.1104 | -0.1123 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_018 | Critical-fast-v4 | -0.0845 | -0.0663 | -0.0663 | 0.0000 | -0.1563 | -0.1463 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_019 | Critical-fast-v4 | -0.0253 | -0.0253 | -0.0236 | 0.0017 | -0.1136 | -0.1053 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
