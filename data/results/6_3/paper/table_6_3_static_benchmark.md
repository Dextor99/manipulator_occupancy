| 场景 | 方法 | Dense feasible | Budget accepted | $D_{\min}$ / m | $J_{\mathrm{smooth}}$ | $T_{\mathrm{plan}}$ / ms | timeout |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P1 | RRT-Connect + smoothing | 1.000 | 1.000 | 0.1294 ± 0.0439 | 0.5915 ± 0.655 | 1286 ± 367 | 0 |
| P1 | MINCO-risk (adapted) | 1.000 | 1.000 | 0.2913 ± 0.0239 | 3.697 ± 0 | 2461 ± 68.9 | 0 |
| P1 | NUBS w/o risk (ablation) | 0.000 | 0.000 | - | - | 2.531 ± 0 | 0 |
| P1 | Critical-point-NUBS | 0.600 | 0.600 | 0.08952 ± 0.00525 | 0.1136 ± 0.00613 | 7784 ± 1.82e+03 | 1 |
| P1 | CCRO-NUBS | 1.000 | 0.600 | 0.1199 ± 0.00271 | 0.1278 ± 0.00719 | 8790 ± 2.71e+03 | 4 |
| P2 | RRT-Connect + smoothing | 1.000 | 1.000 | 0.1279 ± 0.0254 | 0.4426 ± 0.659 | 1461 ± 618 | 0 |
| P2 | MINCO-risk (adapted) | 0.900 | 0.900 | 0.1848 ± 0.0919 | 2.369 ± 1.19 | 2496 ± 102 | 0 |
| P2 | NUBS w/o risk (ablation) | 0.000 | 0.000 | - | - | 2.531 ± 0 | 0 |
| P2 | Critical-point-NUBS | 1.000 | 1.000 | 0.1613 ± 0.004 | 0.1525 ± 0.0193 | 7393 ± 1.5e+03 | 0 |
| P2 | CCRO-NUBS | 1.000 | 1.000 | 0.1122 ± 0.00217 | 0.1187 ± 0.012 | 4056 ± 913 | 0 |
| P3 | RRT-Connect + smoothing | 1.000 | 1.000 | 0.1206 ± 0.0308 | 1.508 ± 1.47 | 3703 ± 1.53e+03 | 0 |
| P3 | MINCO-risk (adapted) | 1.000 | 1.000 | 0.1772 ± 0.0173 | 3.697 ± 0 | 2856 ± 58.8 | 0 |
| P3 | NUBS w/o risk (ablation) | 0.000 | 0.000 | - | - | 2.531 ± 0 | 0 |
| P3 | Critical-point-NUBS | 0.800 | 0.600 | 0.1621 ± 0.00351 | 0.2228 ± 0.0271 | 8814 ± 1.63e+03 | 3 |
| P3 | CCRO-NUBS | 1.000 | 0.000 | 0.1106 ± 0.00207 | 0.1541 ± 0.0138 | 1.963e+04 ± 3.66e+03 | 10 |

Note: $D_{\min}$ and $J_{\mathrm{smooth}}$ are computed only over dense-feasible trajectories. Budget accepted requires dense feasibility and raw planning time no greater than 10 s; the 10 s budget is an offline evaluation criterion, not a hard solver termination.
