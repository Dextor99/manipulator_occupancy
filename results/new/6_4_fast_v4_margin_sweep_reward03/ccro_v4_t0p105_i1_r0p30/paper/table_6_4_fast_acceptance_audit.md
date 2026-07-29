# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing_fast | 0.15 | G1 | CCRO-fast-v4 | 10 | 1.00 | 0.80 | 0.40 | 1.00 | 0.90 | 0.40 | 1.00 | 0.40 | 0.0007 | -0.0122 | -0.0019 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D1F_000 | CCRO-fast-v4 | 0.0774 | 0.0795 | 0.0795 | 0.0000 | -0.0105 | -0.0005 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_001 | CCRO-fast-v4 | 0.0733 | 0.0965 | 0.0980 | 0.0015 | 0.0080 | 0.0165 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_002 | CCRO-fast-v4 | 0.0730 | 0.0997 | 0.1046 | 0.0049 | 0.0146 | 0.0197 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_003 | CCRO-fast-v4 | 0.0760 | 0.0917 | 0.0922 | 0.0005 | 0.0022 | 0.0117 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_004 | CCRO-fast-v4 | 0.0721 | 0.0837 | 0.0849 | 0.0012 | -0.0051 | 0.0037 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_005 | CCRO-fast-v4 | 0.0773 | 0.0895 | 0.0893 | -0.0002 | -0.0007 | 0.0095 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_006 | CCRO-fast-v4 | 0.0776 | 0.0833 | 0.0833 | 0.0000 | -0.0067 | 0.0033 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_007 | CCRO-fast-v4 | 0.0767 | 0.0890 | 0.0887 | -0.0003 | -0.0013 | 0.0090 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_008 | CCRO-fast-v4 | 0.0764 | 0.0993 | 0.0993 | -0.0001 | 0.0093 | 0.0193 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_009 | CCRO-fast-v4 | 0.0780 | 0.0781 | 0.0778 | -0.0003 | -0.0122 | -0.0019 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
