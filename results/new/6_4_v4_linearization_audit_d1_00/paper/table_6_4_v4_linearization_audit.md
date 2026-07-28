# 6.4 v4 linearization audit

| item | value |
|---|---:|
| reference dense min / m | 0.0729 |
| candidate dense min / m | 0.0729 |
| actual dense gain / m | 0.0000 |
| QP min predicted distance / m | 0.0925 |
| delta norm / rad | 0.0000 |
| max abs delta / rad | 0.0000 |
| online accepted | False |
| elapsed / ms | 71.88 |

## Jacobian Check

| link | relative error | absolute error norm |
|---|---:|---:|
| gripper_base_link | 0.0000 | 0.0000 |

## Active Constraints

| i | link | time / s | d / m | grad norm | A norm | predicted gain / m | predicted d / m | cp-row norms |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | gripper_base_link | 1.0000 | 0.0729 | 1.0565 | 0.0000 | 0.0000 | 0.0729 | 0.000,0.000,0.000,0.000 |
| 1 | wrist3_Link | 1.0000 | 0.0783 | 0.9314 | 0.0000 | 0.0000 | 0.0783 | 0.000,0.000,0.000,0.000 |
| 2 | wrist2_Link | 1.0000 | 0.0820 | 0.8996 | 0.0000 | 0.0000 | 0.0820 | 0.000,0.000,0.000,0.000 |
| 3 | left_link | 1.0000 | 0.0838 | 1.0614 | 0.0000 | 0.0000 | 0.0838 | 0.000,0.000,0.000,0.000 |
| 4 | gripper_base_link | 0.9000 | 0.0925 | 0.9992 | 0.3261 | 0.0000 | 0.0925 | 0.015,0.026,0.063,0.319 |
| 5 | wrist3_Link | 0.9000 | 0.0964 | 0.9805 | 0.3199 | 0.0000 | 0.0964 | 0.014,0.025,0.062,0.313 |
| 6 | wrist2_Link | 0.9000 | 0.1002 | 0.9518 | 0.3106 | 0.0000 | 0.1002 | 0.014,0.025,0.060,0.303 |
| 7 | left_link | 0.9000 | 0.1039 | 1.0983 | 0.3584 | 0.0000 | 0.1039 | 0.016,0.028,0.069,0.350 |
