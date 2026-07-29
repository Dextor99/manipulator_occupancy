# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing_fast | 0.15 | Dynamic | CCRO-fast-v4 | 10 | 1.00 | 0.30 | 0.20 | 1.00 | 1.00 | 0.20 | 1.00 | 0.20 | 0.0008 | -0.0395 | -0.0294 |
| body_crossing_fast | 0.25 | Dynamic | CCRO-fast-v4 | 10 | 1.00 | 0.70 | 0.30 | 1.00 | 1.00 | 0.30 | 1.00 | 0.30 | 0.0015 | -0.0268 | -0.0169 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D1F_000 | CCRO-fast-v4 | 0.0576 | 0.0672 | 0.0684 | 0.0012 | -0.0216 | -0.0128 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_001 | CCRO-fast-v4 | 0.0433 | 0.0519 | 0.0512 | -0.0007 | -0.0388 | -0.0281 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_002 | CCRO-fast-v4 | 0.0595 | 0.0767 | 0.0767 | -0.0001 | -0.0133 | -0.0033 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_003 | CCRO-fast-v4 | 0.0410 | 0.0535 | 0.0546 | 0.0011 | -0.0354 | -0.0265 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_004 | CCRO-fast-v4 | 0.0472 | 0.0506 | 0.0536 | 0.0030 | -0.0364 | -0.0294 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_005 | CCRO-fast-v4 | 0.0444 | 0.0512 | 0.0505 | -0.0007 | -0.0395 | -0.0288 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_006 | CCRO-fast-v4 | 0.0689 | 0.0927 | 0.0926 | -0.0001 | 0.0026 | 0.0127 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_007 | CCRO-fast-v4 | 0.0592 | 0.0679 | 0.0679 | 0.0000 | -0.0221 | -0.0121 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_008 | CCRO-fast-v4 | 0.0772 | 0.0844 | 0.0844 | -0.0001 | -0.0056 | 0.0044 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_009 | CCRO-fast-v4 | 0.0786 | 0.0931 | 0.0971 | 0.0040 | 0.0071 | 0.0131 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_010 | CCRO-fast-v4 | 0.0552 | 0.0631 | 0.0632 | 0.0000 | -0.0268 | -0.0169 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_011 | CCRO-fast-v4 | 0.0435 | 0.0702 | 0.0757 | 0.0055 | -0.0143 | -0.0098 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_012 | CCRO-fast-v4 | 0.0719 | 0.0821 | 0.0819 | -0.0003 | -0.0081 | 0.0021 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_013 | CCRO-fast-v4 | 0.0765 | 0.0782 | 0.0781 | -0.0001 | -0.0119 | -0.0018 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_014 | CCRO-fast-v4 | 0.0776 | 0.0866 | 0.0864 | -0.0002 | -0.0036 | 0.0066 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_015 | CCRO-fast-v4 | 0.0721 | 0.0887 | 0.0889 | 0.0001 | -0.0011 | 0.0087 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_016 | CCRO-fast-v4 | 0.0735 | 0.0916 | 0.0929 | 0.0013 | 0.0029 | 0.0116 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_017 | CCRO-fast-v4 | 0.0764 | 0.0808 | 0.0827 | 0.0019 | -0.0073 | 0.0008 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_018 | CCRO-fast-v4 | 0.0752 | 0.0918 | 0.0918 | -0.0000 | 0.0018 | 0.0118 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_019 | CCRO-fast-v4 | 0.0725 | 0.0884 | 0.0951 | 0.0068 | 0.0051 | 0.0084 | 1 | 1 | 1 | 1 | 1 | usable |
