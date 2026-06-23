# CCRO-NUBS 阶段 3 实施计划与验收报告

> 目标：在阶段 2 的固定时间全身 Mesh 风险优化上加入动态障碍预测占据，使机器人构型 `q(t)` 与障碍位置 `O(t)` 按同一物理时间查询；通过程序化已知真值场景、消融对比、梯度校验和独立稠密复核完成阶段验收。

---

## 1. 阶段边界与完成定义

阶段 3 输入：

```text
阶段 1 的固定时间 6-DOF NUBS 轨迹后端
阶段 2 的 URDF collision Mesh 全身表面模型
q_start / q_goal 及固定非均匀分段时间 T
动态对象在规划起点的中心、速度、尺寸和不确定性
预测有效时域 valid_horizon
```

阶段 3 输出：

```text
可由轨迹相对时间 tau 查询的动态占据
全身时空风险 J_risk(P_inner)
对 P_inner 的链式梯度
优化后的固定时间 NUBS 候选轨迹
独立 dense Mesh + 细时间网格验收结果
A/B/C 场景、四种对比方法、图、表和 JSON 指标
```

本阶段明确不包含：

- 总时长或分段时间联合优化；
- 在线事件触发、滚动预测与轨迹拼接；
- 自碰撞检测；
- 真实机械臂指令发送；
- 将现有 `0.4 s` 在线预测无条件外推为 `8 s` 可靠预测。

因此，本阶段完成后可以宣称“动态预测占据已与固定时间 NUBS 轨迹物理时间对齐，并在程序化动态场景中进入全身轨迹优化”，不能宣称“已经实现真机动态在线重规划”。

---

## 2. 与前两阶段的继承关系

阶段 3 不重新实现轨迹或 Mesh 模型，直接复用：

```text
planning/nubs_trajectory.py       固定时间 NUBS 构造、采样和能量
planning/optimizer.py             JointLimits、运动学限制惩罚
planning/robot_surface_model.py   collision Mesh 多分辨率表面点
planning/mesh_risk.py             非均匀时间网格梯形积分权重
planning/verifier.py              阶段 2 验收结构并扩展动态版本
```

保持不变的约定：

1. `P_inner` 是内部插值构型，不是 B 样条控制点；
2. `q_goal` 是轨迹尾端硬约束，不重复加入软目标项；
3. `T=[1.6, 1.8, 2.2, 2.4] s` 固定，总时长为 `8.0 s`；
4. 优化使用 medium 表面，验收使用 dense 表面；
5. 求解器成功只表示数值收敛，候选轨迹仍必须经过独立验收器。

---

## 3. 软件设计

```text
planning/obstacle_forecast.py
  ForecastSphere / ObstacleOccupancy / ObstacleForecast
  ConstantVelocitySphereForecast / FrozenSphereForecast / CompositeForecast

planning/spatiotemporal_risk.py
  动态球占据距离、等连杆权重全身风险、构型梯度、轨迹时间积分

planning/dynamic_optimizer.py
  固定 T 的 minimum-jerk + 时空风险 + 运动学限制优化

planning/verifier.py
  DynamicTrajectoryVerifier：细时间网格和 dense Mesh 独立复核

experiments/exp_ccro_stage3.py
  构造已知真值动态场景、运行四种方法、输出指标与曲线

config/ccro_stage3.yaml
  阶段参数唯一来源

tests/planning/test_ccro_stage3.py
  预测、时间对齐、拒绝逻辑、梯度及风险下降测试
```

模块依赖方向为：

```text
ObstacleForecast -> SpatioTemporalRiskEvaluator
                 -> DynamicRiskNUBSOptimizer
                 -> DynamicTrajectoryVerifier
```

优化器和验收器共享“风险定义”，但使用不同的表面密度和时间网格；验收器不复用优化器的采样结果，避免同一离散误差同时污染求解与验收。

---

## 4. 动态占据模型

### 4.1 时间语义

`tau` 是从本次规划开始计时的物理秒：

```text
tau = 0                   当前规划状态
tau = trajectory time     对应 q(tau) 的障碍占据
```

程序化实验的真值和优化器均调用：

```python
forecast.occupancy_at(tau)
```

不会使用归一化参数，也不会在每个轨迹采样点把障碍重新置为 `tau=0`。

### 4.2 匀速球预测

对象中心：

```text
c(tau) = c0 + v * tau
```

保守半径：

```text
r(tau) = r0 + margin + uncertainty
       + uncertainty_growth * tau
       + velocity_radius_scale * ||v|| * tau
```

`tau=0` 会返回当前占据。预测有效时域由 `valid_horizon` 显式给出，超出时域有两种策略：

- `error`：立即拒绝查询，适合离线严格验收；
- `hold_inflate`：中心保持在时域末端、半径继续膨胀，并标记 `extrapolated=True`。

本次验收采用 `error`，所有轨迹样本的 `extrapolated_sample_count=0`。

### 4.3 当前帧消融

`FrozenSphereForecast` 固定 `tau=0` 的中心和半径，用来回答：如果规划器只看当前障碍而忽略未来运动，会发生什么。它是消融基线，不是可执行安全策略。

---

## 5. 全身时空风险与梯度

对第 `l` 个连杆表面点 `x_lk(q)` 和动态球 `j`：

```text
d_lk(q,tau) = min_j max(||x_lk(q)-c_j(tau)||-r_j(tau), 0)
phi(d) = max(d_safe-d, 0)^2
R_l(q,tau) = mean_k phi(d_lk(q,tau))
R_body(q,tau) = sum_l w_l R_l / sum_l w_l
```

轨迹风险使用实际时间网格的梯形积分：

```text
J_risk = sum_i w_i R_body(q(t_i), t_i)
```

每个连杆内部平均、连杆之间默认等权，避免面积较大的大臂淹没腕部和夹爪风险。

构型风险采用中心有限差分：

```text
dR_i/dq_j = [R(q_i+eps_q e_j,t_i)-R(q_i-eps_q e_j,t_i)]/(2 eps_q)
```

固定 `T` 时，NUBS 样本对内部插值点的灵敏度只需在优化器初始化时计算一次：

```text
S_i = dq(t_i)/dP_inner
dJ_risk/dP_inner = sum_i w_i (dR_i/dq_i) S_i
```

最终目标：

```text
J = lambda_smooth * J_jerk
  + lambda_risk * J_risk
  + J_joint_limits
```

本阶段不含 `J_time`，也不计算时间变量梯度。

---

## 6. 优化和独立验收

### 6.1 优化配置

```text
优化变量              P_inner，shape=(3,6)
求解器                L-BFGS-B
风险时间节点          每段 5 个内部间隔，共 21 个去重节点
优化表面              medium，实际 1999 点（按连杆最小点数取整）
lambda_smooth          0.05
lambda_risk            5000
d_safe / d_activate    0.12 / 0.18 m
max_iterations         80
```

### 6.2 验收配置

```text
验收时间步长          0.025 s，8 s 轨迹共 321 个时刻
验收表面              dense，实际 9000 点
紧急拒绝距离 d_stop   0.04 m
目标/连续性            q、qd、qdd 独立检查
运动学限制            q、qd、qdd 独立检查
预测外推              必须为 0 个样本
```

`d_safe=0.12 m` 是软风险开始惩罚的设计裕量，`d_stop=0.04 m` 是候选轨迹硬拒绝阈值。因此通过的轨迹允许局部距离处于 `[d_stop,d_safe)`；本次 A/B/C 的全身时空轨迹均没有进入 `d_stop`。

验收条件：

```text
solver_ok
finite_ok
goal_ok
distance_ok: dense D_min >= d_stop
position_ok / velocity_ok / acceleration_ok
continuity_q_ok / continuity_qd_ok / continuity_qdd_ok
forecast_horizon_ok: extrapolated_sample_count == 0
```

---

## 7. 实验设计

### 7.1 已知真值场景构造

先生成相同的无风险 NUBS 基线，再从其运动扫掠表面选择远离起终构型的点。对每个障碍设置速度 `v` 和预定碰撞时刻 `t_c`：

```text
c_collision = selected_surface_point + small_outward_offset
c0 = c_collision - v * t_c
```

因此：

```text
c0 + v*t_c = c_collision
```

这使时间对齐具有解析真值。三组实验的最大 `time_alignment_error` 为约 `1.12e-16 m`。

### 7.2 A/B/C 场景

| 场景 | 动态对象 | 目标连杆 | 设计目的 |
|---|---:|---|---|
| A | 1，0.12 m/s | 末端、夹爪 | 验证末端动态穿越和基本时空规避 |
| B | 1，0.11 m/s | 上臂、前臂、腕 1 | 构造末端模型漏检的身体碰撞反例 |
| C | 2，0.10 m/s | 前臂、腕 1、腕 2 | 验证多对象组合占据和更复杂规避 |

程序化真值的有效时域专门设置为完整 `8 s`，用于验证算法和时间对齐。它不代表现有在线感知已经具有 8 秒可靠预测能力；在线系统当前的约 `0.4 s` 时域必须在阶段 4 用滚动重规划和安全层处理。

### 7.3 对比方法

| 方法 | 风险输入 | 风险连杆 |
|---|---|---|
| `baseline` | 无 | 无 |
| `current_full` | 固定当前帧 | 全身 |
| `temporal_ee` | 正确动态预测 | 仅末端和夹爪 |
| `temporal_full` | 正确动态预测 | collision Mesh 全身 |

四种方法使用相同起终状态、固定分段时间、轨迹初值、运动学限制和最终验收器。

### 7.4 阶段验收门槛

1. `temporal_full` 在 A/B/C 全部求解成功并通过独立验收；
2. 每组 `temporal_full` 的真实时空风险低于无风险基线；
3. B 场景中 `temporal_ee` 被拒绝而 `temporal_full` 通过；
4. 至少一组 `current_full` 因忽略未来运动被拒绝；
5. 完整目标梯度相对误差不超过 `5e-2`、余弦相似度大于 `0.99`；
6. 录制动态预警结果至少提供正的时间预警增益；若缺少三维中心真值，必须明确标记，不伪造预测 RMSE。

---

## 8. 运行方法

统一使用 `py310`：

```bash
cd /home/hzy/Code/manipulator_occupancy
bash scripts/setup_ccro_stage3.sh
```

运行完整阶段实验：

```bash
/home/hzy/miniconda3/bin/conda run -n py310 \
  python -m experiments.exp_ccro_stage3
```

仅运行阶段 3 测试：

```bash
/home/hzy/miniconda3/bin/conda run -n py310 \
  python -m pytest -p no:cacheprovider tests/planning/test_ccro_stage3.py -q
```

输出：

```text
data/results/ccro_stage3/
  config.yaml
  metrics.json
  table_stage3.md
  scenario_A.png
  scenario_B.png
  scenario_C.png
```

实验程序只有在所有门槛满足时才以退出码 0 结束；否则返回退出码 2，便于 CI 或批处理直接判定失败。

---

## 9. 已完成实验结果

2026-06-23 在 `py310` 中完成首轮固定种子实验，整体结果 `accepted=true`。

| 场景 | 方法 | 验收 | dense D_min / m | `time < d_stop` / s | 真实时空风险 | 优化耗时 / ms |
|---|---|---:|---:|---:|---:|---:|
| A | baseline | 否 | 0.000000 | 1.0 | 1.4321e-3 | - |
| A | current_full | 否 | 0.000000 | 1.0 | 1.4321e-3 | 20.85 |
| A | temporal_ee | 是 | 0.114228 | 0.0 | 1.5914e-8 | 1165.51 |
| A | temporal_full | 是 | 0.113778 | 0.0 | 2.0106e-8 | 2135.88 |
| B | baseline | 否 | 0.000000 | 0.8 | 2.2017e-4 | - |
| B | current_full | 否 | 0.000000 | 0.8 | 2.2017e-4 | 19.27 |
| B | temporal_ee | 否 | 0.000000 | 0.7 | 1.0639e-4 | 556.99 |
| B | temporal_full | 是 | 0.097733 | 0.0 | 7.4078e-8 | 1303.19 |
| C | baseline | 否 | 0.000000 | 2.3 | 5.2931e-3 | - |
| C | current_full | 否 | 0.000000 | 2.3 | 5.2931e-3 | 23.32 |
| C | temporal_ee | 否 | 0.012893 | 0.4 | 5.6072e-5 | 2029.57 |
| C | temporal_full | 是 | 0.092437 | 0.0 | 8.7029e-7 | 3872.76 |

结果解释：

- A 验证基本动态末端穿越，末端和全身时空方法均能规避；
- B 的最近碰撞连杆为 `foreArm_Link`，末端时空法仍碰撞，全身法通过，身体反例成立；
- C 的两个动态对象使末端法最低距离只有 `0.012893 m`，全身法提高到 `0.092437 m`；
- 当前帧法在优化时看到的风险为 0，因此 0 次迭代即“收敛”，但按真实动态轨迹复核三组均碰撞。这正是阶段 3 要解决的时间错位问题；
- 所有全身时空结果的 `q/qd/qdd`、目标端状态和头部连续性检查均通过，且外推样本为 0。

梯度检查：

```text
relative_error      1.410921e-6
cosine_similarity   0.9999999999998568
max_absolute_error  1.058110e-4
门槛                relative_error <= 5e-2
```

录制的 ch4_3 动态预警结果共 8 次试验：

```text
时空方法 T_lead       5.128593 s
当前帧方法 T_lead     4.735648 s
预警提前量增益        +0.392945 s
```

但该结果文件没有稳定的三维对象中心真值，故本阶段没有报告或伪造位置预测 RMSE；录制结果只作为时间预警收益的辅助证据，轨迹优化的因果验证由已知真值程序化场景承担。

---

## 10. 自动测试与最终判定

阶段 3 新增 4 项测试：

1. `tau=0` 当前占据、匀速中心、半径增长和时域越界；
2. 动态风险是否在解析碰撞时刻正确激活；
3. dense 动态验收器是否拒绝定时身体碰撞且不产生外推；
4. 时空目标梯度及优化后风险下降。

测试结果：

```text
tests/planning/test_ccro_stage3.py: 4 passed
```

最终判定：

```text
程序化固定时间阶段 3：完成并通过
动态占据时间对齐：通过
全身时空风险进入 NUBS 优化：通过
末端模型身体漏检反例：通过
独立 dense 安全复核：通过
录制数据三维预测误差验证：未完成（原始结果缺三维中心真值）
真机在线动态重规划：未开始，属于阶段 4
```

阶段 4 的合理起点不是继续延长匀速预测，而是实现短时滚动预测、风险事件触发、从实时 `q/qd/qdd` 生成候选轨迹、轨迹切换连续性和速度级安全层协同。
