| scenario | method | n | task safe | replan success | finish | violation | Dmin GT / m | bridge GT / m | bridge pred / m | replans | accepted | planner ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing | CCRO-NUBS | 15 | 0.93 | 0.73 | 0.93 | 0.07 | 0.182 | 0.277 | 0.223 | 2.00 | 0.87 | 1538.8 |
| body_crossing | Critical-point-NUBS | 15 | 0.67 | 0.00 | 0.67 | 0.20 | 0.134 | 0.300 | 0.203 | 2.00 | 0.00 | 6379.3 |
| body_crossing | Reference-only | 15 | 0.00 | 0.00 | 0.00 | 1.00 | 0.077 | - | - | 0.00 | 0.00 | - |
| body_crossing | SSM | 15 | 0.07 | 0.00 | 0.07 | 0.87 | 0.007 | - | - | 0.00 | 0.00 | - |
| body_crossing | SSM+APF | 15 | 0.07 | 0.00 | 0.07 | 0.87 | 0.009 | - | - | 0.00 | 0.00 | - |
| ee_crossing | CCRO-NUBS | 15 | 0.27 | 0.47 | 0.27 | 0.73 | 0.023 | 0.156 | 0.094 | 2.00 | 0.47 | 2871.8 |
| ee_crossing | Critical-point-NUBS | 15 | 0.00 | 0.00 | 0.00 | 1.00 | -0.019 | 0.156 | 0.068 | 2.00 | 0.00 | 11001.2 |
| ee_crossing | Reference-only | 15 | 0.80 | 0.00 | 0.80 | 0.20 | 0.091 | - | - | 0.00 | 0.00 | - |
| ee_crossing | SSM | 15 | 0.33 | 0.00 | 0.33 | 0.67 | 0.035 | - | - | 0.00 | 0.00 | - |
| ee_crossing | SSM+APF | 15 | 0.27 | 0.00 | 0.27 | 0.67 | 0.034 | - | - | 0.00 | 0.00 | - |
| far_safe | CCRO-NUBS | 10 | 1.00 | 0.00 | 1.00 | 0.00 | 0.673 | - | - | 0.00 | 0.00 | - |
| initial_high_risk | CCRO-NUBS | 10 | 0.00 | 0.00 | 0.00 | 1.00 | -0.047 | - | - | 0.00 | 0.00 | - |

Notes:

- `violation` is GT safety-distance violation rate under the executed closed loop.
- `initial_high_risk` is a safety-hold test: `finish=0` and `violation=1` are expected because the obstacle is initialized inside the hold region; acceptance is judged by immediate hold, zero replans, and zero candidate switches.
- `task safe` reports task completion without GT safety violation; `replan success` reports at least one accepted candidate switch after a trigger.
- `bridge GT` is the minimum GT distance actually observed during the pending interval; `bridge pred` is the online forecast distance under the expected slowed execution.
- Candidate switching uses online medium validation and is followed by dense GT offline audit; optimizer convergence flags are reported separately in `table_6_4_candidate_validation_audit.md`.
