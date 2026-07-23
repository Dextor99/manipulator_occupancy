| scenario | method | n | success | finish | violation | Dmin GT / m | replans | accepted | first hold / s | planner ms | false replans |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing | CCRO-NUBS | 3 | 1.00 | 1.00 | 0.00 | 0.221 | 2.00 | 1.67 | - | 2439.3 | 0 |
| body_crossing | Reference-only | 3 | 0.00 | 0.00 | 1.00 | 0.077 | 0.00 | 0.00 | - | - | 0 |
| body_crossing | SSM | 3 | 0.00 | 0.00 | 1.00 | 0.003 | 0.00 | 0.00 | 3.800 | - | 0 |

Notes:

- `violation` is GT safety-distance violation rate under the executed closed loop.
- `initial_high_risk` is a safety-hold test: `finish=0` and `violation=1` are expected because the obstacle is initialized inside the hold region; acceptance is judged by immediate hold, zero replans, and zero candidate switches.
- Candidate switching uses independent dense validation as the acceptance gate; optimizer convergence flags are reported separately in `table_6_4_candidate_validation_audit.md`.
