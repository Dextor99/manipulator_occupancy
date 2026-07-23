| scenario | method | n | success | finish | violation | Dmin GT / m | replans | accepted | first hold / s | planner ms | false replans |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing | CCRO-NUBS | 15 | 1.00 | 1.00 | 0.00 | 0.150 | 1.93 | 1.93 | - | 1161.3 | 0 |
| body_crossing | Reference-only | 15 | 0.00 | 0.00 | 1.00 | 0.077 | 0.00 | 0.00 | - | - | 0 |
| body_crossing | SSM | 15 | 0.13 | 0.13 | 0.53 | 0.046 | 0.00 | 0.00 | 3.837 | - | 0 |
| ee_crossing | CCRO-NUBS | 15 | 1.00 | 1.00 | 0.00 | 0.146 | 2.00 | 1.87 | - | 1266.1 | 0 |
| ee_crossing | Reference-only | 15 | 0.00 | 0.00 | 1.00 | 0.077 | 0.00 | 0.00 | - | - | 0 |
| ee_crossing | SSM | 15 | 0.00 | 0.00 | 1.00 | -0.034 | 0.00 | 0.00 | 3.523 | - | 0 |
| far_safe | CCRO-NUBS | 10 | 1.00 | 1.00 | 0.00 | 0.673 | 0.00 | 0.00 | - | - | 0 |
| initial_high_risk | CCRO-NUBS | 10 | 1.00 | 0.00 | 1.00 | -0.047 | 0.00 | 0.00 | 0.000 | - | 0 |

Notes:

- `violation` is GT safety-distance violation rate under the executed closed loop.
- `initial_high_risk` is a safety-hold test: `finish=0` and `violation=1` are expected because the obstacle is initialized inside the hold region; acceptance is judged by immediate hold, zero replans, and zero candidate switches.
- Candidate switching uses independent dense validation as the acceptance gate; optimizer convergence flags are reported separately in `table_6_4_candidate_validation_audit.md`.
