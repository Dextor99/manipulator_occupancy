| scenario | method | n | task safe | replan success | finish | violation | Dmin GT / m | bridge GT / m | bridge pred / m | replans | accepted | planner ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing_main | CCRO-NUBS | 3 | 0.33 | 0.00 | 0.33 | 0.00 | 0.089 | 0.117 | 0.000 | 2.00 | 0.00 | 3484.1 |

Notes:

- `violation` is GT safety-distance violation rate under the executed closed loop.
- `initial_high_risk` is a safety-hold test: `finish=0` and `violation=1` are expected because the obstacle is initialized inside the hold region; acceptance is judged by immediate hold, zero replans, and zero candidate switches.
- `task safe` reports task completion without GT safety violation; `replan success` reports at least one accepted candidate switch after a trigger.
- `bridge GT` is the minimum GT distance actually observed during the pending interval; `bridge pred` is the online forecast distance under the expected slowed execution.
- Candidate switching uses online medium validation and is followed by dense GT offline audit; optimizer convergence flags are reported separately in `table_6_4_candidate_validation_audit.md`.
