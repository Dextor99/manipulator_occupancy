# CCRO-NUBS P0–P7 实施总结与剩余实物工作

> 自动执行日期：2026-06-23；环境：`/home/hzy/miniconda3/envs/py310/bin/python`。

## 1. 状态总览

| 阶段 | 自动结果 | 当前判断 |
|---|---|---|
| P0 冻结与回归 | PASS | 完成 |
| P1 多种子统计 | PASS，360 次评价 | 完成首轮统计 |
| P2 时间优化 | PASS | 完成 |
| P3 录制/真值/影子 | 仿真 PASS，真值 PENDING | 可完成软件部分已完成 |
| P4 A4/A5/A6 | PASS | 仿真完成 |
| P5 实时性 | PASS，p95 997.52 ms | 首要目标完成，增强目标未完成 |
| P6 统一消融 | PASS | 完成当前内部基线 |
| P7 真机 | dry-run PASS，真机 PENDING | 软件闭锁完成，实物待补 |

## 2. 关键结果

```text
P1 temporal_full pass_rate      0.9778
P2 variable-T duration          6.15818 s
P2 variable-T jerk              0.06158182
P3 synthetic prediction RMSE    0.01708 m
P4 A/C A6 Dmin                  0.06160 / 0.07690 m
P4 control worst p95            8.64 ms
P5 planner p95                  997.52 ms
P5 dense feasible / converged   1.00 / 0.50
P6 lambda_time sweep success    4/4
P7 unsafe real switch           blocked
```

## 3. 不能宣称完成的内容

1. P3 没有同步三维中心真值，真实预测 RMSE/覆盖率仍为空；
2. 两条旧录制存在时间戳倒退，不能纳入有效序列；
3. P5 的 C 场景常在迭代上限停止，虽 dense 可行但不是严格收敛；
4. 尚无自碰撞验证，`self_collision_checked=false`；
5. 未复现外部论文官方实现，当前 P6 是项目内部统一基线；
6. P7 没有任何自动真机切换结果，默认配置禁止下发。

## 4. 一键复现

```bash
bash scripts/run_ccro_p0_p7.sh
```

各阶段也可单独运行 `scripts/setup_ccro_pN.sh`。P4 包含墙钟异步实验，P5 固定运行 30 次，因此全流程需要一定时间。

回归说明：P0–P7 相关 planning/gate 测试通过。直接运行全仓裸 `pytest` 还会收集 `robot/01_calibrate_robot/` 下依赖未安装 AUBO SDK 的历史硬件脚本；`tests/` 中另有两个既存 mock/tracker 期望与当前实现不一致（对象数、位置 EMA 后速度值），本轮没有越权修改这些非 P0–P7 行为。

## 5. 下一轮实物补充顺序

1. 清理/重录时间戳异常序列，使用 AprilTag、导轨或人工三维框填写 `data/annotations/ccro_p3_truth.csv`；
2. 只做 P7-A 影子模式，积累候选接受/拒绝和时间同步证据；
3. 实现并测试自碰撞、通信 watchdog、急停链路和厂家限位；
4. P7-B 只开放速度缩放和停止；
5. 全部闸门证据通过后，才在软质障碍、低速条件下人工批准 P7-C 候选切换。
