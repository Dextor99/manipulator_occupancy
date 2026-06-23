# CCRO-NUBS 阶段 2 实施计划

> 目标：在阶段 1 的固定时间 6-DOF NUBS 后端上加入静态障碍 URDF/Mesh 全身风险代价，完成构型风险梯度、L-BFGS-B 优化、独立高密度复核及末端模型漏检反例验证。

---

## 1. 阶段范围

阶段 2 输入：

```text
q_start, qd_start, qdd_start
q_goal, qd_goal, qdd_goal
固定非均匀分段时间 T
静态障碍点云 O
URDF collision meshes
关节位置、速度和加速度规划限制
```

阶段 2 输出：

```text
优化后的 P_inner
连续 NUBS 轨迹 q(t), qd(t), qdd(t), jerk(t)
优化前后 J_risk、D_min 和 jerk 能量
solver 状态和耗时
高密度候选轨迹复核结果
```

阶段 2 不包含：

- 动态障碍预测；
- `P_inner + T` 联合优化；
- 在线重规划；
- 真实机械臂轨迹发送；
- 自碰撞保证。

时间变量属于阶段 2B；动态障碍和重规划分别属于阶段 3、4。

---

## 2. 阶段 1 复用关系

阶段 2 直接复用：

```text
planning/_nubs_cpp*.so
planning/nubs_trajectory.py
planning/optimizer.py::JointLimits
config/ccro_stage1.yaml 中已验证的边界和单位约定
```

阶段 1 的以下约束保持不变：

1. `q_goal` 是 NUBS 尾端硬边界，因此不添加重复的 `J_goal`；
2. 分段时间固定，因此不添加 `J_time`；
3. `P_inner` 是内部插值构型，不是 B 样条控制点；
4. 所有物理量使用 SI 单位；
5. 优化器返回成功不等于轨迹可以执行，必须经过独立复核。

---

## 3. 软件模块

```text
planning/robot_surface_model.py
  URDF collision mesh -> 确定性局部表面采样 -> FK 世界表面点

planning/mesh_risk.py
  静态障碍 KD-tree -> 多表面点 R_body -> dR/dq -> 轨迹风险积分

planning/static_optimizer.py
  NUBS 能量梯度 + 风险链式梯度 + L-BFGS-B

planning/verifier.py
  dense Mesh + 细时间采样的目标、距离和运动学复核

experiments/exp_ccro_stage2.py
  程序化场景 A/B/C、对比方法、结果表和曲线
```

---

## 4. RobotSurfaceModel

### 4.1 几何选择

阶段 2 明确使用：

```text
urdf/aubo_i16_gripper.urdf 中的 collision mesh
```

不再含混使用 visual mesh。collision mesh 的优点是：

- 与碰撞评价语义一致；
- STL 可由 Open3D 直接稳定读取；
- 当前 AUBO 和夹爪 collision mesh 的尺寸已经是米制；
- 比渲染用 DAE 更适合重复采样。

### 4.2 关节边界

当前 URDF 除六个机械臂关节外还包含夹爪 prismatic joints。阶段 2 的优化变量严格限定为配置中的六个关节：

```text
shoulder_joint
upperArm_joint
foreArm_joint
wrist1_joint
wrist2_joint
wrist3_joint
```

夹爪关节保持 URDF 零位，不进入优化变量，避免 6D NUBS 与 8 个 movable joints 顺序混淆。

### 4.3 表面积采样

设第 `l` 个 Mesh 表面积为 `A_l`，dense 目标总点数为 `N_dense`：

```text
N_l = max(N_min, round(N_dense * A_l / sum_j A_j))
```

程序固定 Open3D 随机种子，将 dense 样本作为母集，通过确定性索引派生 medium 和 coarse：

```text
coarse < medium < dense
```

这保证三档点集具有包含关系，不会因为三次独立随机采样造成优化与复核差异无法解释。

### 4.4 缓存

缓存键包含：

- URDF 文件内容；
- 所有 Mesh 文件内容；
- geometry 类型；
- 三档密度；
- 最小连杆点数；
- 随机种子；
- Mesh origin 和 scale。

缓存路径：

```text
data/cache/robot_surface/surface_<hash>.npz
```

任何关键输入变化都会生成新的缓存键。

---

## 5. 静态 Mesh 全身风险

### 5.1 距离方向

对静态障碍点云建立一次 `cKDTree(O)`。每个轨迹构型下，将机器人表面点作为 query：

```text
d_lk(q) = min_{o in O} ||x_lk(q) - o||
```

该方向与风险公式一致：每个机器人表面点都有一个最近障碍距离。

### 5.2 优化代价

每个连杆内部先平均，连杆之间默认等权：

```text
phi(d) = max(d_safe - d, 0)^2

R_l(q) = 1/N_l * sum_k phi(d_lk(q))

R_body(q) = sum_l w_l R_l(q) / sum_l w_l
```

采用“连杆内平均 + 连杆间等权”而不是直接对所有点求和，是为了避免大臂因为 Mesh 面积和采样点多而完全淹没手腕、法兰和夹爪风险。

### 5.3 `D_min` 与 `R_body` 分工

```text
R_body: 优化目标和梯度
D_min: 最近风险连杆、图表、触发和高密度复核
```

禁止使用单个 `D_min` 直接替代多点优化代价。

### 5.4 轨迹积分

每段使用相同数量的时间节点，避免短分段被全局均匀采样漏掉：

```text
J_risk = sum_i w_i R_body(q(t_i))
```

权重 `w_i` 使用非均匀时间网格梯形积分权重。

---

## 6. 风险梯度

### 6.1 构型有限差分

对每个风险采样时刻：

```text
dR/dq_j = [R(q + eps_q e_j) - R(q - eps_q e_j)] / (2 eps_q)
```

默认 `eps_q=2e-4 rad`。只有当 `R_body>0` 且 `D_min<d_activate` 时计算六个关节扰动。

### 6.2 NUBS 样本灵敏度

固定 `T` 和边界状态时，`q(t)` 对 `P_inner` 是线性映射。程序初始化优化器时只计算一次：

```text
S_i = dq(t_i) / dP_inner
```

当前实现用中心有限差分构造 `S_i`，但该矩阵在后续 L-BFGS 迭代中保持不变。

### 6.3 链式传播

```text
dJ_risk/dP = sum_i (w_i dR_i/dq_i) * (dq_i/dP)
```

再与 NUBS 内置解析/传播的平滑能量梯度相加：

```text
dJ/dP = lambda_s dJ_smooth/dP
      + lambda_r dJ_risk/dP
      + dJ_limits/dP
```

### 6.4 完整外层有限差分复核

实验对完整目标再次执行：

```text
g_fd[j] = [J(P + eps e_j) - J(P - eps e_j)] / (2 eps)
```

同时报告相对误差和余弦相似度。当前正式结果为：

```text
relative_error = 0.00301
cosine_similarity = 0.999997
```

满足默认 `relative_error <= 0.05` 的离线 Level-1 验收门槛。

---

## 7. 静态风险优化器

固定时间目标：

```text
J = lambda_smooth * J_smooth
  + lambda_risk * J_risk
  + lambda_position * J_q
  + lambda_velocity * J_qd
  + lambda_acceleration * J_qdd
```

默认使用：

```text
lambda_smooth = 0.05
lambda_risk = 5000
```

权重不是硬件安全常数，而是当前程序化静态场景的第一版数值尺度。正式论文参数敏感性实验应扫描风险权重并报告安全距离、jerk 和耗时的权衡。

L-BFGS-B bounds 只约束 `P_inner`。段内 `q/qd/qdd` 仍通过采样惩罚和最终复核检查。

---

## 8. 独立候选轨迹复核

优化使用 medium 表面点，复核使用 dense 表面点。复核器检查：

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
```

阶段 2 的复核报告明确包含：

```text
self_collision_checked = false
```

即当前轨迹通过的是“外部静态障碍 + 关节运动学 + 边界连续性”复核，不能表述为已经完成自碰撞保证，也不能直接进入真实机械臂执行。

---

## 9. 程序化场景

障碍不是手工写死世界坐标，而是从无风险 NUBS 的中间扫掠表面自动构造：

1. 在轨迹中间区间扫描目标连杆表面；
2. 计算候选表面点到起点和终点全身表面的距离；
3. 选择端点净空最大的中间扫掠点；
4. 沿局部表面向外方向偏置；
5. 生成确定性实心球点云。

这样可保证障碍主要影响轨迹中段，而不是让硬边界起点或终点天然不可行。

### 场景 A

```text
目标连杆: wrist3 / gripper / fingers
障碍数: 1
目的: 验证基本末端绕行
```

### 场景 B

```text
目标连杆: upperArm / foreArm / wrist1
障碍数: 1
目的: 构造末端安全但中间连杆危险的反例
```

### 场景 C

```text
目标连杆: foreArm / wrist1 / wrist2
障碍数: 2
目的: 验证多障碍情况下的固定时间优化能力
```

对比方法：

```text
baseline: 无风险 NUBS
ee_only: 仅末端连杆风险
full_body: 全部 collision mesh 连杆风险
```

---

## 10. 环境和运行

一键核对依赖并运行阶段 1+2 测试：

```bash
cd /home/hzy/Code/manipulator_occupancy
bash scripts/setup_ccro_stage2.sh
```

运行正式阶段 2 实验：

```bash
/home/hzy/miniconda3/bin/conda run -n py310 \
  python -m experiments.exp_ccro_stage2
```

自定义配置或输出目录：

```bash
/home/hzy/miniconda3/bin/conda run -n py310 \
  python -m experiments.exp_ccro_stage2 \
  --config config/ccro_stage2.yaml \
  --output data/results/ccro_stage2_custom
```

默认 A/B/C 全量实验约需数分钟，其中主要耗时来自：

- 风险构型有限差分；
- 多次 L-BFGS 目标调用；
- 每条候选轨迹的 dense Mesh 复核。

---

## 11. 自动测试

`tests/planning/test_ccro_stage2.py` 覆盖：

1. 表面采样缓存和多分辨率确定性；
2. 近场风险为正、远场风险为零；
3. 风险构型梯度为有限值；
4. 复核器能够拒绝碰撞轨迹；
5. 完整目标梯度与外层有限差分一致；
6. 静态优化能够降低风险。

运行：

```bash
/home/hzy/miniconda3/bin/conda run -n py310 \
  python -m pytest -p no:cacheprovider \
  tests/planning/test_nubs_stage1.py \
  tests/planning/test_ccro_stage2.py -q
```

---

## 12. 阶段 2 验收条件

必须同时满足：

```text
[ ] 阶段 1 回归测试继续通过
[ ] RobotSurfaceModel 多分辨率和缓存测试通过
[ ] 完整目标梯度 relative_error <= 0.05
[ ] A/B/C 的 full_body 风险均低于 baseline
[ ] A/B/C 的 full_body 候选均通过 dense 复核
[ ] 场景 B 的 ee_only 无法解决中间连杆碰撞，而 full_body 可以
[ ] solver 失败、复核失败和风险未下降分别记录
[ ] mean/p95/max 耗时字段可由多 trial 扩展统计
[ ] 明确记录 self_collision_checked=false
```

当前第一轮正式结果：

| 场景 | baseline D_min/m | ee_only D_min/m | full_body D_min/m | full_body 复核 |
|---|---:|---:|---:|---:|
| A | 0.000301 | 0.081299 | 0.106467 | PASS |
| B | 0.000437 | 0.000437 | 0.101620 | PASS |
| C | 0.000293 | 0.054373 | 0.102939 | PASS |

场景 B 的最近风险连杆不是末端，`ee_only` 与 baseline 的最小距离相同，而 `full_body` 通过复核，因此全身反例成立。

---

## 13. 输出文件

```text
data/results/ccro_stage2/
  config.yaml
  metrics.json
  table_stage2.md
  scenario_A_obstacle.npz
  scenario_B_obstacle.npz
  scenario_C_obstacle.npz
  scenario_A.png
  scenario_B.png
  scenario_C.png
```

`metrics.json` 保存每个场景、每种方法的 solver、风险、距离、复核、迭代和耗时信息。

---

## 14. 已知限制和下一步

1. 当前程序化障碍用于验证方法正确性，不替代真实 RGB-D 静态点云实验。
2. 当前使用构型有限差分，场景 C 全身优化约需二十余秒；进入在线阶段前需要风险时间窗和连杆上游关节筛选。
3. dense 复核当前采用离散时间采样，不是连续碰撞检测。
4. 未实现自碰撞检测，因此禁止将阶段 2 轨迹直接发送给真实机械臂。
5. 当前只优化 `P_inner`。下一步先评估阶段 2B 的固定时间比例总时长优化，再进入动态障碍阶段。
6. 论文静态主实验还需加入多次 trial、参数敏感性和录制静态点云外部有效性结果。

---

> 文档版本：v1.0  
> Python 环境：`py310`  
> 对应配置：`config/ccro_stage2.yaml`  
> 对应总方案：`document/CCRO-NUBS_实现方案.md`
