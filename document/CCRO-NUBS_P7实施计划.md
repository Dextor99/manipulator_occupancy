# CCRO-NUBS P7：真实机械臂分阶段验证与失效闭锁计划

## 1. 当前范围

P7 软件准备已实现，但不自动连接或运动真实机械臂。默认配置 `allow_real_robot_commands=false`，即使误传模式也会 fail-closed。当前可自动完成的是 dry-run 和影子模式的数据/闸门验证；真实执行结果留待设备补充。

## 2. 三阶段顺序

### A. 影子模式

机械臂仍运行人工确认的原参考轨迹；CCRO 只记录风险、候选、dense 验证、规划耗时和假定切换收益，不发送候选。

### B. 安全层接管

只允许速度缩放与零速停止。先空场景 10 次，再软质静态障碍 10 次，再慢速动态障碍 10 次。任何时间戳、通信或 watchdog 异常立即零速并人工复位。

### C. 低速候选切换

只有全部条件为真才允许：

```text
SelfCollisionOK, WatchdogOK, CommunicationOK, TimestampOK,
SwitchStateErrorOK, DenseValidationOK, EmergencyStopOK,
ManufacturerLimitsVerified
```

并要求配置显式开启、操作者输入一次性批准短语。初始限速 `0.10 rad/s`、加速度 `0.20 rad/s^2`；这些不是厂家额定值，仍需按手册/SDK 核实。

## 3. 新增文件

- `robot/ccro_safety_gate.py`：纯函数式 fail-closed 闸门；
- `experiments/exp_ccro_p7_dry_run.py`：验证默认配置不可能下发切换；
- `config/ccro_p7.yaml`：真机状态唯一来源；
- `tests/test_ccro_p7_gate.py`：缺项、口令和全通过测试。

## 4. 当前验收

```text
dry_run_accepted = true
unsafe_switch_blocked = true
real_robot_complete = false
```

运行：

```bash
bash scripts/setup_ccro_p7.sh
```

真实试验完成后，将每项 readiness 的证据文件路径写入结果，而不是简单把布尔值改成 true。特别是当前 dense verifier 明确 `self_collision_checked=false`，所以候选切换必须继续锁止。

## 5. 实际 dry-run 结果（2026-06-23）

```text
gate tests                 3 passed
dry_run_accepted           true
unsafe_switch_blocked      true
real_robot_complete        false
```

仍缺：自碰撞、watchdog、通信、时间戳、切换状态误差、急停和厂家限位七类实物证据。默认配置保持闭锁。
