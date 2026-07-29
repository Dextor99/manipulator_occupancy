# 6.4 Fast Local Repair Acceptance Audit

## Gate Funnel

| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing_fast | 0.15 | Dynamic | Critical-fast-v4 | 10 | 0.80 | 0.00 | 0.00 | - | 0.20 | 0.00 | 1.00 | 0.00 | 0.0021 | -0.1040 | -0.0940 |
| body_crossing_fast | 0.25 | Dynamic | Critical-fast-v4 | 10 | 0.80 | 0.00 | 0.00 | - | 0.70 | 0.00 | 1.00 | 0.00 | 0.0029 | -0.0989 | -0.0973 |

## Trial-Level Audit

| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D1F_000 | Critical-fast-v4 | 0.0092 | 0.0092 | 0.0162 | 0.0070 | -0.0738 | -0.0708 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_001 | Critical-fast-v4 | 0.0314 | 0.0314 | 0.0321 | 0.0007 | -0.0579 | -0.0486 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_002 | Critical-fast-v4 | 0.0204 | 0.0204 | 0.0274 | 0.0071 | -0.0626 | -0.0596 | 0 | 0 | 0 | 1 | 1 | qp_not_solved |
| D1F_003 | Critical-fast-v4 | 0.0179 | 0.0179 | 0.0185 | 0.0006 | -0.0715 | -0.0621 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_004 | Critical-fast-v4 | 0.0073 | 0.0073 | 0.0073 | 0.0000 | -0.0827 | -0.0727 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_005 | Critical-fast-v4 | -0.0140 | -0.0140 | -0.0140 | 0.0000 | -0.1040 | -0.0940 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_006 | Critical-fast-v4 | 0.0384 | 0.0384 | 0.0384 | 0.0000 | -0.0516 | -0.0416 | 0 | 0 | 0 | 1 | 1 | qp_not_solved |
| D1F_007 | Critical-fast-v4 | 0.0292 | 0.0292 | 0.0303 | 0.0010 | -0.0597 | -0.0508 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_008 | Critical-fast-v4 | 0.0220 | 0.0220 | 0.0243 | 0.0023 | -0.0657 | -0.0580 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_009 | Critical-fast-v4 | 0.0284 | 0.0284 | 0.0311 | 0.0027 | -0.0589 | -0.0516 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_010 | Critical-fast-v4 | 0.0103 | 0.0103 | 0.0102 | -0.0001 | -0.0798 | -0.0697 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_011 | Critical-fast-v4 | -0.0008 | -0.0008 | -0.0003 | 0.0005 | -0.0903 | -0.0808 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_012 | Critical-fast-v4 | 0.0281 | 0.0281 | 0.0355 | 0.0074 | -0.0545 | -0.0519 | 0 | 0 | 0 | 1 | 1 | qp_not_solved |
| D1F_013 | Critical-fast-v4 | -0.0111 | -0.0114 | -0.0080 | 0.0034 | -0.0980 | -0.0914 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_014 | Critical-fast-v4 | 0.0315 | 0.0315 | 0.0315 | 0.0000 | -0.0585 | -0.0485 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_015 | Critical-fast-v4 | 0.0279 | 0.0279 | 0.0280 | 0.0000 | -0.0620 | -0.0521 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
| D1F_016 | Critical-fast-v4 | 0.0357 | 0.0357 | 0.0357 | 0.0000 | -0.0543 | -0.0443 | 0 | 0 | 0 | 1 | 1 | qp_not_solved |
| D1F_017 | Critical-fast-v4 | -0.0071 | -0.0071 | 0.0035 | 0.0106 | -0.0865 | -0.0871 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_018 | Critical-fast-v4 | -0.0173 | -0.0173 | -0.0089 | 0.0084 | -0.0989 | -0.0973 | 1 | 0 | 0 | 0 | 1 | dense_not_safe |
| D1F_019 | Critical-fast-v4 | 0.0067 | 0.0122 | 0.0112 | -0.0010 | -0.0788 | -0.0678 | 1 | 0 | 0 | 1 | 1 | dense_not_safe |
