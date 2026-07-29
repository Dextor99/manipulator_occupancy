# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing_fast | 0.15 | G1 | Critical-fast-v4 | 10 | 1.00 | 0.10 | 0.10 | 1.00 | 0.90 | 0.10 | 1.00 | 0.10 | 0.0015 | -0.0349 | -0.0248 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D1F_000 | Critical-fast-v4 | 0.0569 | 0.0724 | 0.0715 | -0.0010 | -0.0185 | -0.0076 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_001 | Critical-fast-v4 | 0.0589 | 0.0742 | 0.0808 | 0.0066 | -0.0092 | -0.0058 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_002 | Critical-fast-v4 | 0.0606 | 0.0654 | 0.0654 | 0.0000 | -0.0246 | -0.0146 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_003 | Critical-fast-v4 | 0.0606 | 0.0887 | 0.0938 | 0.0050 | 0.0038 | 0.0087 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_004 | Critical-fast-v4 | 0.0557 | 0.0646 | 0.0652 | 0.0006 | -0.0248 | -0.0154 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_005 | Critical-fast-v4 | 0.0619 | 0.0732 | 0.0779 | 0.0046 | -0.0121 | -0.0068 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_006 | Critical-fast-v4 | 0.0572 | 0.0694 | 0.0684 | -0.0010 | -0.0216 | -0.0106 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_007 | Critical-fast-v4 | 0.0552 | 0.0552 | 0.0551 | -0.0002 | -0.0349 | -0.0248 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_008 | Critical-fast-v4 | 0.0572 | 0.0696 | 0.0690 | -0.0006 | -0.0210 | -0.0104 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_009 | Critical-fast-v4 | 0.0613 | 0.0613 | 0.0624 | 0.0011 | -0.0276 | -0.0187 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
