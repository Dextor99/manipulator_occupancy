# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing_fast | 0.15 | G1 | CCRO-fast-v4 | 10 | 1.00 | 1.00 | 0.80 | 1.00 | 1.00 | 0.80 | 1.00 | 0.80 | 0.0006 | -0.0069 | 0.0035 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D1F_000 | CCRO-fast-v4 | 0.0755 | 0.0938 | 0.0938 | -0.0000 | 0.0038 | 0.0138 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_001 | CCRO-fast-v4 | 0.0763 | 0.0933 | 0.0936 | 0.0003 | 0.0036 | 0.0133 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_002 | CCRO-fast-v4 | 0.0767 | 0.0932 | 0.0950 | 0.0018 | 0.0050 | 0.0132 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_003 | CCRO-fast-v4 | 0.0757 | 0.0835 | 0.0831 | -0.0004 | -0.0069 | 0.0035 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_004 | CCRO-fast-v4 | 0.0769 | 0.0924 | 0.0924 | 0.0000 | 0.0024 | 0.0124 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_005 | CCRO-fast-v4 | 0.0787 | 0.0931 | 0.0969 | 0.0038 | 0.0069 | 0.0131 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_006 | CCRO-fast-v4 | 0.0766 | 0.0845 | 0.0845 | 0.0000 | -0.0055 | 0.0045 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_007 | CCRO-fast-v4 | 0.0764 | 0.0937 | 0.0952 | 0.0015 | 0.0052 | 0.0137 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_008 | CCRO-fast-v4 | 0.0772 | 0.0940 | 0.0930 | -0.0011 | 0.0030 | 0.0140 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_009 | CCRO-fast-v4 | 0.0776 | 0.0926 | 0.0926 | 0.0000 | 0.0026 | 0.0126 | 1 | 1 | 1 | 1 | 1 | usable |
