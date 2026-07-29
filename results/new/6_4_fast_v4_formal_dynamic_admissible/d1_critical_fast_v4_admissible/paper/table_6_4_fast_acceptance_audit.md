# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing_fast | 0.15 | Dynamic | Critical-fast-v4 | 10 | 1.00 | 0.20 | 0.20 | 1.00 | 0.80 | 0.20 | 1.00 | 0.20 | 0.0017 | -0.0467 | -0.0367 |
| body_crossing_fast | 0.25 | Dynamic | Critical-fast-v4 | 10 | 0.70 | 0.20 | 0.00 | - | 1.00 | 0.00 | 1.00 | 0.00 | 0.0018 | -0.0332 | -0.0232 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D1F_000 | Critical-fast-v4 | 0.0576 | 0.0646 | 0.0653 | 0.0006 | -0.0247 | -0.0154 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_001 | Critical-fast-v4 | 0.0433 | 0.0433 | 0.0433 | 0.0001 | -0.0467 | -0.0367 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_002 | Critical-fast-v4 | 0.0595 | 0.0781 | 0.0790 | 0.0009 | -0.0110 | -0.0019 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_003 | Critical-fast-v4 | 0.0410 | 0.0478 | 0.0489 | 0.0010 | -0.0411 | -0.0322 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_004 | Critical-fast-v4 | 0.0472 | 0.0472 | 0.0500 | 0.0028 | -0.0400 | -0.0328 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_005 | Critical-fast-v4 | 0.0444 | 0.0481 | 0.0481 | 0.0000 | -0.0419 | -0.0319 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_006 | Critical-fast-v4 | 0.0689 | 0.0864 | 0.0909 | 0.0046 | 0.0009 | 0.0064 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_007 | Critical-fast-v4 | 0.0592 | 0.0659 | 0.0655 | -0.0004 | -0.0245 | -0.0141 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_008 | Critical-fast-v4 | 0.0772 | 0.0744 | 0.0743 | -0.0000 | -0.0157 | -0.0056 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_009 | Critical-fast-v4 | 0.0786 | 0.0947 | 0.1023 | 0.0076 | 0.0123 | 0.0147 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_010 | Critical-fast-v4 | 0.0552 | 0.0568 | 0.0568 | -0.0000 | -0.0332 | -0.0232 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_011 | Critical-fast-v4 | 0.0435 | 0.0594 | 0.0661 | 0.0067 | -0.0239 | -0.0206 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_012 | Critical-fast-v4 | 0.0719 | 0.0719 | 0.0715 | -0.0004 | -0.0185 | -0.0081 | 0 | 0 | 0 | 1 | 1 | qp_not_solved |
| D1F_013 | Critical-fast-v4 | 0.0765 | 0.0786 | 0.0784 | -0.0002 | -0.0116 | -0.0014 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_014 | Critical-fast-v4 | 0.0776 | 0.0776 | 0.0776 | 0.0000 | -0.0124 | -0.0024 | 0 | 0 | 0 | 1 | 1 | qp_not_solved |
| D1F_015 | Critical-fast-v4 | 0.0721 | 0.0751 | 0.0752 | 0.0001 | -0.0148 | -0.0049 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_016 | Critical-fast-v4 | 0.0735 | 0.0872 | 0.0878 | 0.0006 | -0.0022 | 0.0072 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_017 | Critical-fast-v4 | 0.0764 | 0.0764 | 0.0786 | 0.0022 | -0.0114 | -0.0036 | 0 | 0 | 0 | 1 | 1 | qp_not_solved |
| D1F_018 | Critical-fast-v4 | 0.0752 | 0.0853 | 0.0843 | -0.0010 | -0.0057 | 0.0053 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_019 | Critical-fast-v4 | 0.0725 | 0.0790 | 0.0888 | 0.0098 | -0.0012 | -0.0010 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
