# 6.4 v4 linearization audit

| item | value |
|---|---:|
| reference dense min / m | 0.0774 |
| candidate dense min / m | 0.0853 |
| actual dense gain / m | 0.0080 |
| QP min predicted distance / m | 0.0867 |
| delta norm / rad | 0.0712 |
| max abs delta / rad | 0.0255 |
| online accepted | False |
| elapsed / ms | 77.87 |

## Jacobian Check

| link | relative error | absolute error norm |
|---|---:|---:|
| foreArm_Link | 0.0000 | 0.0000 |

## Active Constraints

| i | link | time / s | d / m | grad norm | A norm | predicted gain / m | predicted d / m | cp-row norms |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | foreArm_Link | 0.4000 | 0.0774 | 0.2120 | 0.2120 | 0.0079 | 0.0853 | 0.000,0.212,0.000,0.000 |
| 1 | foreArm_Link | 0.5000 | 0.0782 | 0.2130 | 0.2026 | 0.0081 | 0.0863 | 0.049,0.135,0.135,0.049 |
| 2 | foreArm_Link | 0.6000 | 0.0787 | 0.2506 | 0.2506 | 0.0090 | 0.0877 | 0.000,0.000,0.251,0.000 |
| 3 | foreArm_Link | 0.3000 | 0.0788 | 0.1723 | 0.1678 | 0.0079 | 0.0867 | 0.133,0.098,0.027,0.014 |
| 4 | foreArm_Link | 0.7000 | 0.0820 | 0.2859 | 0.2785 | 0.0068 | 0.0888 | 0.024,0.044,0.163,0.220 |
| 5 | foreArm_Link | 0.2000 | 0.0833 | 0.1416 | 0.1416 | 0.0051 | 0.0884 | 0.142,0.000,0.000,0.000 |
| 6 | foreArm_Link | 0.8000 | 0.0877 | 0.3163 | 0.3163 | 0.0028 | 0.0905 | 0.000,0.000,0.000,0.316 |
| 7 | foreArm_Link | 0.1000 | 0.0906 | 0.1302 | 0.0425 | 0.0010 | 0.0917 | 0.041,0.008,0.003,0.002 |
