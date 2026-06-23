# CCRO-NUBS 阶段 4 实施方案与验收报告

> 目标：在阶段 3 的全身时空风险优化基础上，实现事件触发、独立进程求解、预测延迟补偿、候选轨迹稠密复核和 HIGH 风险安全保持，并在参考轨迹级虚拟闭环中验证动态身体/末端障碍、无误触发和立即接管。

---

## 1. 阶段范围

阶段 4 当前完成的是：

```text
程序化动态障碍真值
未来轨迹风险监控
LOW / MEDIUM / HIGH 状态机
单槽独立进程重规划
3 s 硬超时
预计切换时刻的 q/qd/qdd 延迟补偿
候选轨迹 dense Mesh 复核
超时、失败、拒绝和 HIGH 风险安全保持
参考轨迹级虚拟闭环
```

当前不包含：

- 真机轨迹发送；
- 自碰撞保证；
- RGB-D 录制序列的影子重放；
- 实际伺服器跟踪误差、通信延迟和 watchdog；
- `exp_45_controller.py` 的排斥速度闭环；
- 约 `1 Hz` 的稳定重规划性能。

所以本阶段结果可以支撑“动态全身风险能够触发异步候选轨迹重规划，并在参考轨迹级虚拟闭环中安全切换或保持”，不能表述为“真机实时在线重规划已经完成”。

---

## 2. 对旧阶段 4 原型的修正

旧原型可以运行，但存在以下问题：

| 原问题 | 后果 | 当前处理 |
|---|---|---|
| `remaining_waypoints()` 返回 `(6,M-1)` | warm-start 总被丢弃 | 修复为 `(M-1,6)` 并增加测试 |
| 每次切换后障碍预测时间归零 | 动态对象位置错位 | 新增 `ShiftedForecast`，显式查询全局时间 |
| L-BFGS-B 在控制循环同步执行 | 计算期间安全循环停顿 | 使用 Linux `fork` 独立进程和单槽任务 |
| 无硬超时 | 优化卡住可能无限等待 | 3 s 截止，超时终止子进程并安全保持 |
| LOW 不能直接进入 HIGH | 当前碰撞可能先触发优化 | 当前距离或短时刹车窗口直接 HIGH |
| 所有未来碰撞都直接 HIGH | 远期风险没有规划机会 | 引入 `time_to_stop` 与 `emergency_lead_time` |
| 场景只有静态球 | 没有验证动态时间语义 | 改为身体和末端匀速横穿场景 |
| 改善量计算但未进入验收 | `accepted=true` 不能证明收益 | 全过程 `D_min` 改善至少 2 mm 成为硬门槛 |
| 最终候选 D_min 冒充执行全过程 D_min | 指标结论偏乐观 | 分开记录 passive/active 实际时间线 D_min |

---

## 3. 软件结构

```text
planning/obstacle_forecast.py
  ShiftedForecast(source, time_offset, local_horizon)

planning/trajectory_buffer.py
  绝对时间索引、正确 warm-start、pause/resume

planning/replanner.py
  FutureRiskReport / ReplanEvent / SafetyEvent
  独立进程 worker、提交、轮询、硬超时、切换和安全保持

experiments/exp_ccro_stage4.py
  动态场景、wall-clock 50 ms 控制循环、全过程指标和验收

tests/planning/test_ccro_stage4.py
  时间偏移、warm-start、HIGH、非阻塞接受和超时测试

config/ccro_stage4.yaml
  阈值、预算、场景和实验唯一参数源
```

---

## 4. 时间轴设计

阶段 4 同时存在三个时间量：

```text
t_global       实验/系统全局时间
tau_active     当前活动轨迹局部时间
tau_candidate  候选轨迹局部时间
```

未来风险监控使用：

```text
q = active.evaluate(tau_active + delta)
O = global_forecast.occupancy_at(t_global + delta)
```

候选轨迹在计划切换时刻 `t_switch` 开始，其局部预测由：

```text
ShiftedForecast.occupancy_at(tau_candidate)
  = global_forecast.occupancy_at(t_switch + tau_candidate)
```

这保证重规划后障碍不会被错误重置到初始位置。

---

## 5. 延迟补偿和连续切换

MEDIUM 风险触发时，不立即冻结当前轨迹。调度器预留：

```text
t_switch = t_submit + switch_delay
switch_delay = 3.0 s
```

从当前活动轨迹预测切换时刻的状态：

```text
q_head     = q_active(t_switch)
qd_head    = qd_active(t_switch)
qdd_head   = qdd_active(t_switch)
```

候选轨迹以该非零状态作为硬边界。规划进程运行期间，原轨迹继续执行；候选提前完成后等待切换时隙。切换时由独立复核器保证：

```text
continuity_q_ok
continuity_qd_ok
continuity_qdd_ok
```

候选剩余总时长为：

```text
T_candidate = T_active_remaining - switch_delay
```

各分段按原非均匀时间比例缩放，因此不会因为每次重规划重新附加完整 8 秒而无限延长任务。

---

## 6. 异步任务与故障处理

### 6.1 单槽任务

同一时刻最多存在一个规划进程：

```text
submit_replan()   构造候选边界、warm-start 和 ShiftedForecast，立即返回
poll_candidate()  控制周期中非阻塞读取结果
```

优化和 dense 复核均在子进程内完成，跨进程只传递数组和普通字典。

### 6.2 硬超时

```text
planning_budget = 3.0 s
```

截止时仍无结果：

1. `terminate()` 子进程；
2. 记录 `outcome=timeout` 和 `planning_budget_exceeded`；
3. 冻结活动轨迹时间；
4. 进入 HIGH 并记录一次安全接管。

单元测试通过 `artificial_worker_delay` 验证了这一分支。

### 6.3 过期和当前危险

控制循环仍对当前实际参考构型计算 dense 距离。如果规划进行中出现：

```text
current D_min <= d_stop
```

立即终止候选任务并保持，不等待优化结果。

---

## 7. 风险状态机

参数：

```text
d_replan             0.130 m
d_safe               0.110 m
d_accept             0.060 m
d_stop               0.035 m
hysteresis_enter      0.015 m
hysteresis_exit       0.010 m
emergency_lead_time   0.30 s
```

状态规则：

```text
LOW:
  future D_min > d_replan - hysteresis_exit
  -> 原轨迹继续

MEDIUM:
  远期 future D_min 进入预警区，且当前距离仍安全
  -> 提交一个异步候选任务

HIGH:
  current D_min <= d_stop
  或 time_to_stop <= emergency_lead_time
  或优化超时/失败/候选拒绝
  -> 终止规划并安全保持
```

安全接管采用锁存，连续周期内不会为同一状态重复累计事件。

---

## 8. 候选接受门槛

候选必须同时满足：

```text
solver_ok
finite_ok
goal_ok
distance_ok: dense D_min >= d_stop
position_ok
velocity_ok
acceleration_ok
continuity_q_ok
continuity_qd_ok
continuity_qdd_ok
forecast_horizon_ok
candidate dense D_min >= d_accept
完成时间未超过 3 s
```

场景总验收还要求：

```text
active 全过程 D_min >= d_stop
active 全过程 D_min > passive 全过程 D_min + 0.002 m
最终目标误差 <= 1e-4
没有安全接管
```

候选复核值和执行全过程值分别保存，不相互替代。

---

## 9. 实验场景

| 场景 | 类型 | 目标 | 验收预期 |
|---|---|---|---|
| A | `0.07 m/s` 动态身体横穿 | `foreArm_Link` | 一次候选接受、提高全过程距离、到达目标 |
| B | 远静态球 | 全身 | LOW，无误触发、到达目标 |
| C | `0.12 m/s` 动态末端横穿 | `right_link` | 一次候选接受、提高全过程距离、到达目标 |
| D | 当前前臂碰撞 | `foreArm_Link` | LOW 直接 HIGH，只产生一次安全接管 |

动态横穿方向不是固定选一个切向量，而是在多个切向方向中搜索，要求初始占据远离机械臂且轨迹附近形成受控间隙，避免“名义末端场景实际先穿过身体”的无效场景。

---

## 10. 实际结果

最终运行：

```text
data/results/ccro_stage4/metrics.json
accepted = true
```

| 场景 | passive D_min / m | active D_min / m | 改善 / m | 重规划 | 接受 | 安全接管 | 目标 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A 动态身体 | 0.049840 | 0.061741 | +0.011900 | 1 | 1 | 0 | 到达 |
| B 远障碍 | 1.082109 | 1.082139 | +0.000031 | 0 | 0 | 0 | 到达 |
| C 动态末端 | 0.042528 | 0.082756 | +0.040229 | 1 | 1 | 0 | 到达 |
| D 当前碰撞 | 0.000000 | 0.000000 | - | 0 | 0 | 1 | 正确保持 |

A、C 候选的全部复核检查均为 `true`，目标误差分别为：

```text
A 5.59e-16
C 2.84e-16
```

重规划和 dense 复核合计耗时：

```text
n       2
mean    1460.24 ms
p95     1502.45 ms
max     1507.14 ms
budget  3000 ms
```

两个任务均在预算内完成，并且规划期间控制侧各运行 60 个 `50 ms` 周期，说明优化不再阻塞控制循环。

但 `p95 > 1000 ms`，所以当前实测只支持约 `0.67 Hz` 的候选生成吞吐，不支持“已达到 1 Hz”的表述。样本数仅 2，耗时数据是功能性验证，不是统计充分的实时性结论。

---

## 11. 自动测试

阶段 4 新增 5 项测试：

1. `ShiftedForecast` 的全局/局部时间映射；
2. warm-start shape 和 pause/resume 连续性；
3. LOW 状态当前危险能否立即进入 HIGH；
4. 子进程提交是否非阻塞、候选能否接受；
5. 超时是否终止任务、拒绝切换并保持位置。

阶段 1–4 联合回归：

```text
19 passed
```

---

## 12. 运行方法

```bash
cd /home/hzy/Code/manipulator_occupancy
bash scripts/setup_ccro_stage4.sh

/home/hzy/miniconda3/bin/conda run -n py310 \
  python -m experiments.exp_ccro_stage4
```

输出：

```text
data/results/ccro_stage4/
  config.yaml
  metrics.json
  table_stage4.md
```

---

## 13. 最终判定与下一步

```text
异步动态风险触发：通过
全局预测时间对齐：通过
非零预测状态连续切换：通过
硬超时和 HIGH 保持：通过单元/故障注入测试
动态身体和末端程序化场景：通过
参考轨迹级虚拟闭环：通过
1 Hz 实时性：未达到
录制序列影子模式：未完成
排斥速度虚拟执行器：未集成
真机自动切换：禁止
```

下一步应优先做 20–30 次重复耗时统计、录制 RGB-D/关节序列影子模式，以及将 `exp_45_controller.py` 的速度缩放和排斥量接入带 `q_sim` 积分的执行器。只有补齐自碰撞、watchdog、通信异常停止和低速真机切换验证后，才允许将本阶段扩展为真实机械臂在线重规划。
