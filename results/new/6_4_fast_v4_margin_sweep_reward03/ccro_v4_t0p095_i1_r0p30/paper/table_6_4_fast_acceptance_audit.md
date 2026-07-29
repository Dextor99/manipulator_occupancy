# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing_fast | 0.15 | G1 | CCRO-fast-v4 | 10 | 1.00 | 1.00 | 0.40 | 1.00 | 0.90 | 0.40 | 1.00 | 0.40 | 0.0002 | -0.0100 | 0.0005 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D1F_000 | CCRO-fast-v4 | 0.0774 | 0.0825 | 0.0825 | 0.0000 | -0.0075 | 0.0025 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_001 | CCRO-fast-v4 | 0.0733 | 0.0933 | 0.0935 | 0.0002 | 0.0035 | 0.0133 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_002 | CCRO-fast-v4 | 0.0730 | 0.0924 | 0.0946 | 0.0022 | 0.0046 | 0.0124 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_003 | CCRO-fast-v4 | 0.0760 | 0.0901 | 0.0905 | 0.0005 | 0.0005 | 0.0101 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_004 | CCRO-fast-v4 | 0.0721 | 0.0868 | 0.0868 | 0.0000 | -0.0032 | 0.0068 | 1 | 1 | 0 | 0 | 1 | online_timeout |
| D1F_005 | CCRO-fast-v4 | 0.0773 | 0.0889 | 0.0889 | 0.0000 | -0.0011 | 0.0089 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_006 | CCRO-fast-v4 | 0.0776 | 0.0838 | 0.0838 | 0.0000 | -0.0062 | 0.0038 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_007 | CCRO-fast-v4 | 0.0767 | 0.0877 | 0.0878 | 0.0001 | -0.0022 | 0.0077 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_008 | CCRO-fast-v4 | 0.0764 | 0.0938 | 0.0937 | -0.0001 | 0.0037 | 0.0138 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_009 | CCRO-fast-v4 | 0.0780 | 0.0805 | 0.0800 | -0.0005 | -0.0100 | 0.0005 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
