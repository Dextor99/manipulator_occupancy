# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing_fast | 0.15 | G1 | CCRO-fast-v4 | 10 | 1.00 | 1.00 | 0.50 | 1.00 | 0.00 | 0.50 | 1.00 | 0.00 | 0.0004 | -0.0080 | 0.0022 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D1F_000 | CCRO-fast-v4 | 0.0774 | 0.0861 | 0.0861 | 0.0000 | -0.0039 | 0.0061 | 1 | 1 | 0 | 0 | 1 | online_timeout |
| D1F_001 | CCRO-fast-v4 | 0.0733 | 0.0995 | 0.1002 | 0.0007 | 0.0102 | 0.0195 | 1 | 1 | 1 | 0 | 1 | online_timeout |
| D1F_002 | CCRO-fast-v4 | 0.0730 | 0.0981 | 0.1009 | 0.0028 | 0.0109 | 0.0181 | 1 | 1 | 1 | 0 | 1 | online_timeout |
| D1F_003 | CCRO-fast-v4 | 0.0760 | 0.0921 | 0.0927 | 0.0006 | 0.0027 | 0.0121 | 1 | 1 | 1 | 0 | 1 | online_timeout |
| D1F_004 | CCRO-fast-v4 | 0.0721 | 0.0868 | 0.0868 | 0.0000 | -0.0032 | 0.0068 | 1 | 1 | 0 | 0 | 1 | online_timeout |
| D1F_005 | CCRO-fast-v4 | 0.0773 | 0.0917 | 0.0917 | 0.0000 | 0.0017 | 0.0117 | 1 | 1 | 1 | 0 | 1 | online_timeout |
| D1F_006 | CCRO-fast-v4 | 0.0776 | 0.0874 | 0.0873 | -0.0001 | -0.0027 | 0.0074 | 1 | 1 | 0 | 0 | 1 | online_timeout |
| D1F_007 | CCRO-fast-v4 | 0.0767 | 0.0890 | 0.0888 | -0.0003 | -0.0012 | 0.0090 | 1 | 1 | 0 | 0 | 1 | online_timeout |
| D1F_008 | CCRO-fast-v4 | 0.0764 | 0.0991 | 0.0990 | -0.0001 | 0.0090 | 0.0191 | 1 | 1 | 1 | 0 | 1 | online_timeout |
| D1F_009 | CCRO-fast-v4 | 0.0780 | 0.0822 | 0.0820 | -0.0001 | -0.0080 | 0.0022 | 1 | 1 | 0 | 0 | 1 | online_timeout |
