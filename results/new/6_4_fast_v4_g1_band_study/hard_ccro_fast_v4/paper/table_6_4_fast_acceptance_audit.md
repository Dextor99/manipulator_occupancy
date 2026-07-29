# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing_fast | 0.15 | G1 | CCRO-fast-v4 | 10 | 1.00 | 0.30 | 0.20 | 1.00 | 0.90 | 0.20 | 1.00 | 0.10 | 0.0020 | -0.0260 | -0.0186 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D1F_000 | CCRO-fast-v4 | 0.0569 | 0.0737 | 0.0737 | 0.0000 | -0.0163 | -0.0063 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_001 | CCRO-fast-v4 | 0.0589 | 0.0853 | 0.0934 | 0.0081 | 0.0034 | 0.0053 | 1 | 1 | 1 | 0 | 1 | online_timeout |
| D1F_002 | CCRO-fast-v4 | 0.0606 | 0.0755 | 0.0761 | 0.0007 | -0.0139 | -0.0045 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_003 | CCRO-fast-v4 | 0.0606 | 0.0858 | 0.0864 | 0.0006 | -0.0036 | 0.0058 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_004 | CCRO-fast-v4 | 0.0557 | 0.0646 | 0.0654 | 0.0008 | -0.0246 | -0.0154 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_005 | CCRO-fast-v4 | 0.0619 | 0.0872 | 0.0949 | 0.0077 | 0.0049 | 0.0072 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_006 | CCRO-fast-v4 | 0.0572 | 0.0720 | 0.0720 | 0.0001 | -0.0180 | -0.0080 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_007 | CCRO-fast-v4 | 0.0552 | 0.0704 | 0.0701 | -0.0002 | -0.0199 | -0.0096 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_008 | CCRO-fast-v4 | 0.0572 | 0.0724 | 0.0725 | 0.0001 | -0.0175 | -0.0076 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_009 | CCRO-fast-v4 | 0.0613 | 0.0614 | 0.0640 | 0.0025 | -0.0260 | -0.0186 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
