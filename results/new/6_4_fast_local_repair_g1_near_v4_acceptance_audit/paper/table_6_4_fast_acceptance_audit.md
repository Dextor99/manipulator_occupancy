# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing_fast | 0.15 | G1 | Critical-fast-v4 | 10 | 0.80 | 0.60 | 0.10 | 1.00 | 0.90 | 0.10 | 1.00 | 0.10 | 0.0011 | -0.0144 | -0.0040 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D1F_000 | Critical-fast-v4 | 0.0774 | 0.0774 | 0.0774 | 0.0000 | -0.0126 | -0.0026 | 0 | 0 | 0 | 1 | 1 | qp_not_solved |
| D1F_001 | Critical-fast-v4 | 0.0733 | 0.0856 | 0.0886 | 0.0030 | -0.0014 | 0.0056 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_002 | Critical-fast-v4 | 0.0730 | 0.0954 | 0.1013 | 0.0060 | 0.0113 | 0.0154 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_003 | Critical-fast-v4 | 0.0760 | 0.0760 | 0.0756 | -0.0004 | -0.0144 | -0.0040 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_004 | Critical-fast-v4 | 0.0721 | 0.0793 | 0.0801 | 0.0007 | -0.0099 | -0.0007 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_005 | Critical-fast-v4 | 0.0773 | 0.0863 | 0.0864 | 0.0002 | -0.0036 | 0.0063 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_006 | Critical-fast-v4 | 0.0776 | 0.0857 | 0.0867 | 0.0010 | -0.0033 | 0.0057 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_007 | Critical-fast-v4 | 0.0767 | 0.0878 | 0.0880 | 0.0002 | -0.0020 | 0.0078 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_008 | Critical-fast-v4 | 0.0764 | 0.0835 | 0.0837 | 0.0002 | -0.0063 | 0.0035 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_009 | Critical-fast-v4 | 0.0780 | 0.0780 | 0.0779 | -0.0001 | -0.0121 | -0.0020 | 0 | 0 | 0 | 1 | 1 | qp_not_solved |
