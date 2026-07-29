# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing_fast | 0.15 | G1 | CCRO-fast-v4 | 10 | 1.00 | 0.90 | 0.50 | 1.00 | 0.00 | 0.50 | 1.00 | 0.00 | 0.0006 | -0.0123 | -0.0019 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D1F_000 | CCRO-fast-v4 | 0.0774 | 0.0841 | 0.0841 | -0.0000 | -0.0059 | 0.0041 | 1 | 1 | 0 | 0 | 1 | online_timeout |
| D1F_001 | CCRO-fast-v4 | 0.0733 | 0.1018 | 0.1032 | 0.0013 | 0.0132 | 0.0218 | 1 | 1 | 1 | 0 | 1 | online_timeout |
| D1F_002 | CCRO-fast-v4 | 0.0730 | 0.0997 | 0.1029 | 0.0031 | 0.0129 | 0.0197 | 1 | 1 | 1 | 0 | 1 | online_timeout |
| D1F_003 | CCRO-fast-v4 | 0.0760 | 0.0921 | 0.0927 | 0.0006 | 0.0027 | 0.0121 | 1 | 1 | 1 | 0 | 1 | online_timeout |
| D1F_004 | CCRO-fast-v4 | 0.0721 | 0.0886 | 0.0887 | 0.0002 | -0.0013 | 0.0086 | 1 | 1 | 0 | 0 | 1 | online_timeout |
| D1F_005 | CCRO-fast-v4 | 0.0773 | 0.0930 | 0.0930 | 0.0000 | 0.0030 | 0.0130 | 1 | 1 | 1 | 0 | 1 | online_timeout |
| D1F_006 | CCRO-fast-v4 | 0.0776 | 0.0865 | 0.0866 | 0.0000 | -0.0034 | 0.0065 | 1 | 1 | 0 | 0 | 1 | online_timeout |
| D1F_007 | CCRO-fast-v4 | 0.0767 | 0.0892 | 0.0888 | -0.0004 | -0.0012 | 0.0092 | 1 | 1 | 0 | 0 | 1 | online_timeout |
| D1F_008 | CCRO-fast-v4 | 0.0764 | 0.1009 | 0.1026 | 0.0016 | 0.0126 | 0.0209 | 1 | 1 | 1 | 0 | 1 | online_timeout |
| D1F_009 | CCRO-fast-v4 | 0.0780 | 0.0781 | 0.0777 | -0.0003 | -0.0123 | -0.0019 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
