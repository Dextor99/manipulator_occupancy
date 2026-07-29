# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing_fast | 0.15 | G1 | CCRO-fast-v4 | 10 | 1.00 | 0.90 | 0.50 | 1.00 | 0.00 | 0.50 | 1.00 | 0.00 | 0.0007 | -0.0122 | -0.0019 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D1F_000 | CCRO-fast-v4 | 0.0774 | 0.0822 | 0.0820 | -0.0002 | -0.0080 | 0.0022 | 1 | 1 | 0 | 0 | 1 | online_timeout |
| D1F_001 | CCRO-fast-v4 | 0.0733 | 0.1054 | 0.1069 | 0.0015 | 0.0169 | 0.0254 | 1 | 1 | 1 | 0 | 1 | online_timeout |
| D1F_002 | CCRO-fast-v4 | 0.0730 | 0.1008 | 0.1051 | 0.0043 | 0.0151 | 0.0208 | 1 | 1 | 1 | 0 | 1 | online_timeout |
| D1F_003 | CCRO-fast-v4 | 0.0760 | 0.0921 | 0.0927 | 0.0006 | 0.0027 | 0.0121 | 1 | 1 | 1 | 0 | 1 | online_timeout |
| D1F_004 | CCRO-fast-v4 | 0.0721 | 0.0837 | 0.0849 | 0.0012 | -0.0051 | 0.0037 | 1 | 1 | 0 | 0 | 1 | online_timeout |
| D1F_005 | CCRO-fast-v4 | 0.0773 | 0.0942 | 0.0938 | -0.0004 | 0.0038 | 0.0142 | 1 | 1 | 1 | 0 | 1 | online_timeout |
| D1F_006 | CCRO-fast-v4 | 0.0776 | 0.0859 | 0.0859 | 0.0000 | -0.0041 | 0.0059 | 1 | 1 | 0 | 0 | 1 | online_timeout |
| D1F_007 | CCRO-fast-v4 | 0.0767 | 0.0890 | 0.0887 | -0.0003 | -0.0013 | 0.0090 | 1 | 1 | 0 | 0 | 1 | online_timeout |
| D1F_008 | CCRO-fast-v4 | 0.0764 | 0.1019 | 0.1027 | 0.0008 | 0.0127 | 0.0219 | 1 | 1 | 1 | 0 | 1 | online_timeout |
| D1F_009 | CCRO-fast-v4 | 0.0780 | 0.0781 | 0.0778 | -0.0003 | -0.0122 | -0.0019 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
