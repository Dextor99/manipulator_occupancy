# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing_fast | 0.15 | G1 | Critical-fast-v4 | 10 | 1.00 | 0.80 | 0.30 | 1.00 | 1.00 | 0.30 | 1.00 | 0.30 | 0.0030 | -0.0169 | -0.0071 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D1F_000 | Critical-fast-v4 | 0.0723 | 0.0760 | 0.0762 | 0.0002 | -0.0138 | -0.0040 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_001 | Critical-fast-v4 | 0.0674 | 0.0907 | 0.1008 | 0.0101 | 0.0108 | 0.0107 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_002 | Critical-fast-v4 | 0.0748 | 0.0818 | 0.0818 | 0.0000 | -0.0082 | 0.0018 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_003 | Critical-fast-v4 | 0.0722 | 0.0858 | 0.0890 | 0.0032 | -0.0010 | 0.0058 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_004 | Critical-fast-v4 | 0.0746 | 0.0824 | 0.0824 | 0.0000 | -0.0076 | 0.0024 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_005 | Critical-fast-v4 | 0.0735 | 0.0959 | 0.1018 | 0.0060 | 0.0118 | 0.0159 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_006 | Critical-fast-v4 | 0.0684 | 0.0806 | 0.0860 | 0.0055 | -0.0040 | 0.0006 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_007 | Critical-fast-v4 | 0.0654 | 0.0729 | 0.0731 | 0.0002 | -0.0169 | -0.0071 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_008 | Critical-fast-v4 | 0.0694 | 0.0880 | 0.0930 | 0.0051 | 0.0030 | 0.0080 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_009 | Critical-fast-v4 | 0.0731 | 0.0852 | 0.0851 | -0.0000 | -0.0049 | 0.0052 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
