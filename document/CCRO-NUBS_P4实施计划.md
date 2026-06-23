# CCRO-NUBS P4：A4/A5/A6 统一 Mesh 虚拟闭环实施计划

## 1. 目标

在同一阶段 4 轨迹、动态球预测、URDF collision Mesh 和物理时间轴上比较：

- A4：单次固定参考，不重规划；
- A5：风险触发异步 NUBS 重规划；
- A6：A5 + 虚拟执行器 + 速度缩放 + 全身风险梯度排斥 + HIGH/状态失配零速保持。

程序全程只积分 `q_sim`，不调用真实机械臂 SDK。

## 2. 实现

- `planning/safety_executor.py`：独立安全执行层，状态误差门控、速度限幅和零速接管；
- `experiments/exp_ccro_p4.py`：逐场景生成 A4/A5/A6 并重新计算 Mesh 距离；
- `config/ccro_p4.yaml`：控制器参数与验收门槛；
- `tests/planning/test_ccro_p4.py`：HIGH、状态失配和正常跟踪测试。

A6 不修改规划线程；规划过程中仍按 20 Hz 控制。候选仍由阶段 4 dense verifier 验收，执行层只对已激活参考进行速度级保护。

## 3. 指标

```text
D_min, T_viol, goal_error, finished
planning_control_cycles
control mean/p95/max
HIGH hold 次数、state mismatch 次数
```

动态 A/C 要求 A5 优于 A4；A6 不低于 d_stop，且控制 p95 < 50 ms。远障碍 B 不误触发；当前碰撞 D 必须零速接管。

## 4. 运行

```bash
bash scripts/setup_ccro_p4.sh
```

真机迁移前必须把 `q_sim` 替换为独立反馈状态，并验证通信 watchdog、急停、自碰撞和制造商速度/加速度限制。

## 5. 实际执行结果（2026-06-23）

| 场景 | A4 Dmin / m | A5 Dmin / m | A6 Dmin / m | A5 接受重规划 | A6 control p95 / ms |
|---|---:|---:|---:|---:|---:|
| A 动态身体 | 0.04984 | 0.06174 | 0.06160 | 1 | 8.04 |
| B 远障碍 | 1.08211 | 1.08214 | 1.08214 | 0 | 0.85 |
| C 动态末端 | 0.04253 | 0.08280 | 0.07690 | 1 | 8.64 |
| D 当前碰撞 | 0 | 0 | 0（零速保持） | 0 | 8.40 |

四场景均通过，A6 在 A/C 到达目标且 `T_viol=0`，D 正确拒绝运动。A6 并不保证 Dmin 一定高于 A5；其职责是规划期间和异常状态下的执行保护，本结果中 A/C 的距离均保持在 `d_stop` 以上。
