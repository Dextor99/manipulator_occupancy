| scenario | method | n | task safe | replan success | finish | violation | Dmin GT / m | bridge Dmin / m | replans | accepted | planner ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing | CCRO-NUBS | 15 | 0.93 | 0.87 | 0.93 | 0.07 | 0.188 | 0.009 | 2.00 | 1.07 | 1697.5 |
| body_crossing | Critical-point-NUBS | 15 | 0.60 | 0.60 | 0.60 | 0.40 | 0.129 | 0.011 | 2.00 | 0.60 | 340.2 |
| body_crossing | Reference-only | 15 | 0.00 | 0.00 | 0.00 | 1.00 | 0.077 | - | 0.00 | 0.00 | - |
| body_crossing | SSM | 15 | 0.07 | 0.00 | 0.07 | 0.87 | 0.007 | - | 0.00 | 0.00 | - |
| body_crossing | SSM+APF | 15 | 0.07 | 0.00 | 0.07 | 0.87 | 0.009 | - | 0.00 | 0.00 | - |
| ee_crossing | CCRO-NUBS | 15 | 0.40 | 0.40 | 0.40 | 0.60 | 0.045 | 0.025 | 2.00 | 0.40 | 3120.0 |
| ee_crossing | Critical-point-NUBS | 15 | 0.27 | 0.07 | 0.27 | 0.73 | 0.023 | 0.026 | 2.00 | 0.07 | 1111.8 |
| ee_crossing | Reference-only | 15 | 0.80 | 0.00 | 0.80 | 0.20 | 0.091 | - | 0.00 | 0.00 | - |
| ee_crossing | SSM | 15 | 0.33 | 0.00 | 0.33 | 0.67 | 0.035 | - | 0.00 | 0.00 | - |
| ee_crossing | SSM+APF | 15 | 0.27 | 0.00 | 0.27 | 0.67 | 0.034 | - | 0.00 | 0.00 | - |
| far_safe | CCRO-NUBS | 10 | 1.00 | 0.00 | 1.00 | 0.00 | 0.673 | - | 0.00 | 0.00 | - |
| initial_high_risk | CCRO-NUBS | 10 | 0.00 | 0.00 | 0.00 | 1.00 | -0.047 | - | 0.00 | 0.00 | - |

Notes:

- `violation` is GT safety-distance violation rate under the executed closed loop.
- `initial_high_risk` is a safety-hold test: `finish=0` and `violation=1` are expected because the obstacle is initialized inside the hold region; acceptance is judged by immediate hold, zero replans, and zero candidate switches.
- `task safe` reports task completion without GT safety violation; `replan success` reports at least one accepted candidate switch after a trigger.
- Candidate switching uses online medium validation and is followed by dense GT offline audit; optimizer convergence flags are reported separately in `table_6_4_candidate_validation_audit.md`.
