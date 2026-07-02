| material | decision | reason |
|---|---|---|
| CCRO-NUBS full-body | use | Our method result; dense verifier accepted in A/B/C. |
| NUBS-base | ablation | Internal no-risk variant; not an external benchmark. |
| NUBS-EEF-risk | ablation | Internal end-effector-only risk variant. |
| MINCO-risk | use | Continuous polynomial trajectory optimization baseline. |
| MINCO-base | auxiliary | No risk term; useful only as lower baseline. |
| RRT-Connect + smoothing | use | Sampling baseline; already has 30 seeds per scenario. |
| CHOMP / TrajOpt | missing | Add at least one optimization-style baseline. |
| GPMP2 | missing | Add GPMP2-style continuous-time optimizer or document omission. |
