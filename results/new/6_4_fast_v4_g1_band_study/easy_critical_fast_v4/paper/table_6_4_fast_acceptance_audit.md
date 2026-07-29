# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing_fast | 0.15 | G1 | Critical-fast-v4 | 10 | 1.00 | 0.90 | 0.40 | 1.00 | 1.00 | 0.40 | 1.00 | 0.40 | 0.0018 | -0.0138 | -0.0025 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D1F_000 | Critical-fast-v4 | 0.0755 | 0.0866 | 0.0866 | -0.0000 | -0.0034 | 0.0066 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_001 | Critical-fast-v4 | 0.0763 | 0.0863 | 0.0865 | 0.0002 | -0.0035 | 0.0063 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_002 | Critical-fast-v4 | 0.0767 | 0.0873 | 0.0923 | 0.0050 | 0.0023 | 0.0073 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_003 | Critical-fast-v4 | 0.0757 | 0.0775 | 0.0762 | -0.0012 | -0.0138 | -0.0025 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_004 | Critical-fast-v4 | 0.0769 | 0.0860 | 0.0863 | 0.0002 | -0.0037 | 0.0060 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_005 | Critical-fast-v4 | 0.0787 | 0.0964 | 0.1035 | 0.0071 | 0.0135 | 0.0164 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_006 | Critical-fast-v4 | 0.0766 | 0.0812 | 0.0807 | -0.0005 | -0.0093 | 0.0012 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
| D1F_007 | Critical-fast-v4 | 0.0764 | 0.0943 | 0.1012 | 0.0069 | 0.0112 | 0.0143 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_008 | Critical-fast-v4 | 0.0772 | 0.0904 | 0.0904 | 0.0000 | 0.0004 | 0.0104 | 1 | 1 | 1 | 1 | 1 | usable |
| D1F_009 | Critical-fast-v4 | 0.0776 | 0.0870 | 0.0872 | 0.0002 | -0.0028 | 0.0070 | 1 | 1 | 0 | 1 | 1 | dense_safe_but_online_margin_rejected |
