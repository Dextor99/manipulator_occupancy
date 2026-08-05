# 6.5.1 Real-Platform Baseline Results

Generated at: `2026-08-05T07:19:18.098942+00:00`
Mode: `offline`

| Condition | Trials | Completion | Joint RMSE mean / rad | Joint RMSE P95 / rad | Max joint error / rad | Terminal max error / rad | Min clearance / m | False HOLD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 No-obstacle NUBS | 10 | 1.00 | 0.00319 | 0.00322 | 0.01050 | 0.00404 | N/A | 0 |
| B2 Static CCRO-NUBS | 10 | 1.00 | 0.00319 | 0.00323 | 0.01057 | 0.00544 | 0.09853 | 0 |

## Admission Checks

| Check | Result |
|---|---:|
| B0_valid_frame_rate_ge_95pct | PASS |
| B0_dropout_le_100ms | PASS |
| B0_no_false_hold | PASS |
| B1_all_trials_completed | PASS |
| B1_max_joint_error_le_0p03rad | PASS |
| B1_terminal_max_error_le_0p02rad | PASS |
| B1_no_false_hold | PASS |
| B2_dense_verification_accepted | PASS |
| B2_all_trials_completed | PASS |
| B2_min_clearance_ge_0p08m | PASS |
| B2_no_false_hold | PASS |

Overall: **PASS**