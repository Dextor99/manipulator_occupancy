# 6.5.1 Real-Platform Baseline Results

> 本结果用于验证6.5.1实验程序、轨迹生成、风险评价、dense复核和统计链路的可重复性；若 `sensor_live=false` 或 `robot_commanded=false`，则未直接驱动AUBO i16或采集完整RealSense实机执行数据，不作为正式实机结果。

Generated at: `2026-08-05T07:29:00.471981+00:00`
Mode: `offline`
Software gate: `PASS`
Real experiment gate: `NOT_RUN`

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