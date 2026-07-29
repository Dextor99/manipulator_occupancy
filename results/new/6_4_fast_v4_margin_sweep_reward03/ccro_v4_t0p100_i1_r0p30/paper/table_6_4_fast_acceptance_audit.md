# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing_fast | 0.15 | G1 | CCRO-fast-v4 | 10 | 1.00 | 0.90 | 0.50 | 1.00 | 0.90 | 0.50 | 1.00 | 0.50 | 0.0004 | -0.0123 | -0.0019 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D1F_000 | CCRO-fast-v4 | 0.0774 | 0.0809 | 0.0809 | 0.0000 | -0.0091 | 0.0009 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_001 | CCRO-fast-v4 | 0.0733 | 0.0959 | 0.0964 | 0.0005 | 0.0064 | 0.0159 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_002 | CCRO-fast-v4 | 0.0730 | 0.0966 | 0.0994 | 0.0028 | 0.0094 | 0.0166 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_003 | CCRO-fast-v4 | 0.0760 | 0.0914 | 0.0920 | 0.0006 | 0.0020 | 0.0114 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_004 | CCRO-fast-v4 | 0.0721 | 0.0852 | 0.0860 | 0.0008 | -0.0040 | 0.0052 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_005 | CCRO-fast-v4 | 0.0773 | 0.0902 | 0.0902 | -0.0001 | 0.0002 | 0.0102 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_006 | CCRO-fast-v4 | 0.0776 | 0.0836 | 0.0837 | 0.0001 | -0.0063 | 0.0036 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_007 | CCRO-fast-v4 | 0.0767 | 0.0892 | 0.0888 | -0.0004 | -0.0012 | 0.0092 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_008 | CCRO-fast-v4 | 0.0764 | 0.0969 | 0.0969 | -0.0000 | 0.0069 | 0.0169 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_009 | CCRO-fast-v4 | 0.0780 | 0.0781 | 0.0777 | -0.0003 | -0.0123 | -0.0019 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
