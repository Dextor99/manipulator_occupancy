# CCRO-NUBS 论文实现方案

> 基于构型耦合时空风险占据与非均匀 B 样条优化的机械臂动态安全运动生成方法

---

## 1. 文档目标与实现边界

本文档用于把 `CCRO-NUBS_论文框架与实现路线.md` 中的方法设想落成可执行的工程任务、模块接口、测试顺序和验收标准。

本路线遵循以下边界：

1. 论文主体优先实现关节空间 `q_start -> q_goal` 的连续轨迹，不在第一版中直接优化任务空间末端位姿。
2. 先完成离线、固定分段时间的 NUBS 优化，再增加时间变量、动态占据和在线重规划。
3. 优化层实现的是“风险代价驱动轨迹优化”，不在没有显式不等式约束和连续时间证明的情况下宣称严格安全保证。
4. `D_min` 用于触发、安全评价和候选轨迹复核；优化目标使用多表面点平滑聚合的 `R_body`，不直接优化单一最近点。
5. 低频轨迹重规划与速度级安全保护必须解耦。优化超时、失败或候选轨迹被拒绝时，安全层仍可独立减速或停止。
6. 真实机械臂自动轨迹切换是加强验证，不是论文方法成立的唯一前提。

---

## 2. 项目现状与真实进度

### 2.1 现有处理链路

```text
RGB-D 点云
  -> 坐标变换、工作空间裁剪、体素降采样
  -> URDF/Mesh 构型点云与自身点云剔除
  -> DBSCAN 对象提取
  -> 跨帧关联、位置 EMA、速度估计
  -> 短时匀速预测与不确定性膨胀
  -> Mesh 全身距离、构型有限差分梯度
  -> 速度缩放、排斥速度、停止保护
```

现有链路是 CCRO-NUBS 的感知和安全执行前端，但尚未形成“未来风险 -> NUBS 连续轨迹优化 -> 候选轨迹复核 -> 轨迹切换”的闭环。

### 2.2 可复用模块与限制

| 功能 | 现有位置 | 可复用内容 | 接入前必须处理的问题 |
|---|---|---|---|
| URDF 解析和 FK | `robot/urdf_model.py` | 关节树、连杆变换 | 只读取首个 visual mesh；未解析可用的速度/加速度限制 |
| Mesh 局部预采样 | `test_remove_robot_points_fast.py::RobotPointRemover` | 局部点缓存和 FK 变换 | 核心类位于测试脚本；实验代码依赖私有字段；点数按连杆平均分配 |
| 对象跟踪 | `perception/occupancy_tracker.py` | 对象 ID、位置和速度 | 第一版规划只使用短时近似匀速模型 |
| 风险预测 | `risk/prediction.py` | 中心外推和半径膨胀 | 当前默认时域仅 `0.4 s`，不能无说明地用于 `2-4 s` 规划 |
| 全身距离和梯度 | `experiments/exp_44_main.py` | 表面点云、KD-tree、中心有限差分 | 当前主要优化单一最小距离，需增加多表面点聚合风险 |
| 速度级安全层 | `experiments/exp_45_controller.py` | 缩放、排斥和安全滤波 | 当前是离线/虚拟闭环验证；实测 p95 耗时仍需优化 |
| NUBS 后端 | `NUBSTrajectory-main/` + `planning/` | 轨迹构造、求值、能量、中间点梯度和 6-DOF Python 接口 | 阶段 1 已通过；NUBS 源码目录仍需正式纳入版本管理 |

### 2.3 现有实验的可用性

| 现有实验 | 对 CCRO-NUBS 的作用 | 状态判定 |
|---|---|---|
| 4.2 构型耦合自身点云解耦 | 支撑感知前端 | 已有脚本和数据 |
| 4.3 对象级时空风险预警 | 支撑动态风险输入 | 已有脚本和数据 |
| 4.4 构型排斥方向 | 提供全身距离梯度参考实现 | 已有末端和中间连杆数据 |
| 4.5 速度级安全运动 | 作为未来高频安全层基础 | 离线回放和虚拟积分已有，不等价于 NUBS 轨迹重规划 |
| 4.6 消融与耗时 | 支撑现有前端模块分析 | 消融可用；完整端到端耗时结论不充分 |

### 2.4 按论文四阶段判定当前位置

```text
感知/风险前端        基本完成，需生产化整理
NUBS 独立 C++ 库       已存在，未接入机械臂系统
阶段 1：无障碍 NUBS    已完成，自动验收 accepted=true
阶段 2：静态风险     已完成程序化 A/B/C 首轮验证，录制静态点云实验待补
阶段 3：动态风险     程序化固定时间 A/B/C 已完成并通过；录制序列缺三维中心真值，未做位置 RMSE
阶段 4：事件重规划   程序化异步参考轨迹闭环已通过；约 0.67 Hz，录制序列与真机切换未完成
```

---

## 3. 总体软件架构

### 3.1 目标目录

正式模块不放在 `experiments/` 中。`experiments/` 只保留参数解析、场景组织、结果保存和绘图。

```text
planning/
  __init__.py
  types.py                    # JointState、ObstacleForecast、TrajectorySample 等数据类
  robot_surface_model.py      # 正式的 URDF/Mesh 多分辨率表面模型
  nubs_trajectory.py          # Python 层 NUBS 轨迹对象，封装 C++ 扩展
  mesh_risk.py                # 静态/时变 Mesh 全身风险代价
  objective.py                # 平滑、风险、关节限制等组合目标
  optimizer.py                # 固定 T、总时长、完整 T 的分阶段优化
  verifier.py                 # 候选轨迹独立高密度复核
  trajectory_buffer.py        # 当前轨迹、剩余轨迹和采样缓冲
  replanner.py                # 风险触发、warm start、超时和候选切换

planning/cpp/
  nubs_pybind.cpp             # pybind11 封装
  CMakeLists.txt

config/
  planning.yaml               # 轨迹、关节限制、代价权重、超时和验收阈值

tests/planning/
  test_robot_surface_model.py
  test_nubs_binding.py
  test_nubs_gradient.py
  test_mesh_risk.py
  test_optimizer_static.py
  test_verifier.py
  test_replanner.py

experiments/
  exp_ccro_stage1.py
  exp_ccro_stage2.py
  exp_ccro_stage3.py
  exp_ccro_stage4.py
```

### 3.2 核心数据类

```python
@dataclass(frozen=True)
class JointState:
    q: np.ndarray       # shape (6,), rad
    qd: np.ndarray      # shape (6,), rad/s
    qdd: np.ndarray     # shape (6,), rad/s^2
    timestamp: float


@dataclass(frozen=True)
class TrajectorySample:
    times: np.ndarray   # shape (N,), s
    q: np.ndarray       # shape (N, 6)
    qd: np.ndarray      # shape (N, 6)
    qdd: np.ndarray     # shape (N, 6)
    jerk: np.ndarray    # shape (N, 6)


@dataclass
class OptimizationResult:
    success: bool
    status: str
    message: str
    trajectory: "NUBSTrajectory6D | None"
    p_inner: np.ndarray
    durations: np.ndarray
    cost_total: float
    cost_terms: dict[str, float]
    iterations: int
    elapsed_ms: float
    gradient_norm: float


@dataclass
class VerificationResult:
    accepted: bool
    reasons: list[str]
    goal_error: float
    min_distance: float
    nearest_link: str | None
    q_violation: float
    qd_violation: float
    qdd_violation: float
    continuity_error: dict[str, float]
    validation_ms: float
```

所有轨迹模块统一使用 SI 单位：弧度、秒、米。角度制只能出现在人机交互或实验配置转换入口。

### 3.3 配置文件边界

`config/planning.yaml` 至少包含：

```yaml
robot:
  joint_names: []              # 必须与 URDF 和机械臂反馈顺序一致
  q_min: []
  q_max: []
  qd_max: []
  qdd_max: []
  execution_limit_scale: 0.8   # 规划限制比硬件极限保守

trajectory:
  system_order: 3              # minimum jerk
  segment_count: 4
  initial_total_duration: 4.0
  min_segment_duration: 0.1
  samples_per_segment: 8

risk:
  d_activate: 0.20
  d_safe: 0.15
  d_stop: 0.05
  fd_epsilon_q: 1.0e-4
  aggregation: squared_hinge

optimization:
  max_iterations: 100
  fixed_time_timeout_s: 5.0
  online_timeout_s: 0.8
  gradient_tolerance: 1.0e-5
  lambda_smooth: 1.0
  lambda_risk: 10.0
  lambda_position: 10.0
  lambda_velocity: 10.0
  lambda_acceleration: 10.0

validation:
  time_step_s: 0.01
  dense_mesh_points: 50000
  epsilon_goal: 1.0e-4
  epsilon_continuity_q: 1.0e-5
  epsilon_continuity_qd: 1.0e-4
  epsilon_continuity_qdd: 1.0e-3
```

表中的值是初始工程建议，不是已验证的 AUBO i16 硬件限制。`q_min/q_max/qd_max/qdd_max` 必须根据用户手册、SDK 限制和实验安全策略明确填写，不得直接使用 URDF 中的 `velocity="0"`。

---

## 4. P0：实现前基线整理

P0 不改变方法，但会决定后续结果是否可复现。

### 4.1 版本与依赖管理

1. 决定 `NUBSTrajectory-main/` 的归属：
   - 推荐作为 `third_party/nubs_trajectory/` 纳入当前项目；或
   - 作为独立 Git submodule 并固定 commit。
2. 不在未跟踪目录上继续添加系统对接代码，否则后续论文结果无法对应确定版本。
3. `py310` 作为项目可用 Python 环境，统一命令格式：

```bash
/home/hzy/miniconda3/bin/conda run -n py310 python -m pytest -p no:cacheprovider tests
/home/hzy/miniconda3/bin/conda run -n py310 python -m experiments.exp_ccro_stage1
```

4. 如 `py310` 中某个依赖缺失，应在环境内补齐并记录，不将其解读为环境不可用。

### 4.2 结果基线

- 保留 4.2-4.6 现有原始录制数据和结果，不在 CCRO-NUBS 实现中覆盖。
- CCRO-NUBS 新结果使用 `data/results/ccro_stage1` 至 `ccro_stage4`。
- 每次正式实验保存：完整配置副本、Git commit、随机种子、输入数据清单、指标 JSON 和失败原因。

### 4.3 P0 验收条件

```text
[ ] NUBS 源码版本已固定
[ ] py310 可执行 Python 单测
[ ] NUBS CTest 可执行且结果已记录
[ ] planning.yaml 中的关节顺序与机械臂反馈顺序已交叉验证
[ ] 关节规划限制有可追溯来源
```

---

## 5. P1：机器人表面模型生产化

这一步是 NUBS 风险积分的前置工作，避免优化器长期依赖测试脚本和私有字段。

### 5.1 `RobotSurfaceModel` 接口

```python
class RobotSurfaceModel:
    def __init__(
        self,
        urdf_path: str,
        sample_counts: dict[str, int],
        cache_path: str | None,
        seed: int,
    ): ...

    @property
    def joint_names(self) -> tuple[str, ...]: ...

    @property
    def link_names(self) -> tuple[str, ...]: ...

    def surface(
        self,
        q: np.ndarray,
        density: str = "medium",
        links: set[str] | None = None,
    ) -> np.ndarray: ...

    def surface_by_link(
        self,
        q: np.ndarray,
        density: str = "medium",
    ) -> dict[str, np.ndarray]: ...
```

### 5.2 实现要求

1. 采样点在连杆局部坐标系中只生成一次，运行时仅做 FK 变换。
2. 同一高密度样本派生 `coarse/medium/dense` 三档点集，不为每一档独立随机采样。
3. 采样点数优先按 Mesh 表面积分配，同时为手腕、法兰和夹爪保留最小点数。
4. 缓存中记录 URDF 文件哈希、Mesh 文件哈希、采样数、随机种子和单位缩放。缓存元数据不一致时自动拒绝加载。
5. 明确当前使用 visual mesh 还是 collision mesh。论文和代码表述必须一致。
6. 不在规划模块中直接访问 `_local_samples` 或 `_transform_to_world`。

### 5.3 距离查询约定

对环境障碍点集 `O` 建立 KD-tree，批量查询机器人表面点到障碍的距离：

```python
class SurfaceDistanceResult:
    distances: np.ndarray        # 每个机器人表面点的距离
    min_distance: float
    nearest_link: str | None
    robot_point: np.ndarray | None
    obstacle_point: np.ndarray | None
```

`min_distance` 用于报告和复核；`distances` 用于后续多点风险聚合。

### 5.4 P1 验收条件

- 同一随机种子和配置重复初始化时，局部采样完全一致。
- 新模型与 `exp_44_main.py` 在同一 `q` 和障碍点云上的 `D_min` 误差不超过采样分辨率容差。
- `surface_by_link()` 可稳定识别末端与中间连杆危险。
- 执行一次 4.4 代表序列回归测试，方法方向与原结果一致。

---

## 6. 阶段 1：打通 6-DOF NUBS 后端

### 6.1 目标

给定起始状态、终点关节构型、中间插值构型和分段时间，生成满足边界的 6 自由度 minimum-jerk NUBS 轨迹，并验证能量与梯度。

阶段 1 不加障碍、不改变分段时间、不接真实机械臂。

### 6.2 不重写第二套 NUBS 核心

默认路线是直接封装已有 `NUBSTrajectory.hpp`，而不是先重写完整纯 Python de Boor、带状系统和梯度传播。纯 Python 参考实现只在 C++ 接口无法验证某个局部公式时使用，不作为主后端。

理由：

- 已有 C++ 库包含 fixed-order minimum jerk 和非均匀分段时间。
- 已有 NUBS/MINCO、能量梯度、小时长和局部有限差分测试。
- 重写第二套核心容易产生两个数值行为不一致的实现，也会延迟真正的风险优化。

### 6.3 pybind11 第一版接口

```python
class NUBSTrajectory6D:
    def generate(
        self,
        p_inner: np.ndarray,       # shape (M-1, 6)
        head_state: np.ndarray,     # shape (6, 3): q, qd, qdd
        tail_state: np.ndarray,     # shape (6, 3): q, qd, qdd
        durations: np.ndarray,      # shape (M,), each > T_min
    ) -> None: ...

    @property
    def total_duration(self) -> float: ...

    def evaluate(self, t: float, derivative_order: int = 0) -> np.ndarray: ...

    def sample(
        self,
        times: np.ndarray,
        max_derivative: int = 3,
    ) -> TrajectorySample: ...

    def control_points(self) -> np.ndarray: ...

    def energy(self) -> float: ...

    def energy_and_gradient(self) -> tuple[float, np.ndarray, np.ndarray]:
        """return energy, dE/dP_inner, dE/dT"""
```

所有输入都必须检查 shape、有限性、关节维数和分段时间正值。C++ 异常统一转换为明确的 Python 异常，不允许静默返回 NaN 轨迹。

### 6.4 边界条件与目标项

论文主体将 `q_goal` 放入 NUBS 尾端状态的位置列，因此：

```text
q(0) = q_start
qd(0) = qd_start
qdd(0) = qdd_start
q(T) = q_goal
qd(T) = qd_goal    # 第一版默认 0
qdd(T) = qdd_goal  # 第一版默认 0
```

`q_goal` 是硬边界时，不再在目标函数中重复加入 `lambda_goal * J_goal`。目标误差保留为数值正确性和候选轨迹验收指标。只有后续改为软终端条件时，才恢复 `J_goal`。

### 6.5 初始中间构型

第一版按累计时间比例插值：

```python
alpha_i = durations[:i].sum() / durations.sum()
p_inner[i - 1] = (1.0 - alpha_i) * q_start + alpha_i * q_goal
```

空间线性插值只是 warm start，不视为已经安全或满足速度/加速度限制。

### 6.6 阶段 1 优化器

固定 `T` 时优化变量为 `P_inner`：

```text
J = lambda_s * J_smooth + lambda_q * J_q
    + lambda_v * J_v + lambda_a * J_a
```

无障碍基准中可先只使用 `J_smooth`，验证自由中间插值点优化后能够恢复与相同边界单段 minimum-jerk 参考一致的轨迹。随后再加入采样型关节位置、速度和加速度惩罚。

优化使用 `scipy.optimize.minimize(method="L-BFGS-B", jac=True)`。阶段 1 先使用 NUBS 内置平滑能量梯度，不在 Python 中对该能量再做一层全变量有限差分。

### 6.7 阶段 1 测试顺序

1. 先运行原 NUBS CTest，确认封装前后库行为没有改变。
2. 对 `Dim=6, S=3` 增加起终边界和中间插值点测试。
3. 将 Python 封装的采样结果与 C++ 测试向量比较。
4. 将 `dE/dP_inner` 与完整中心有限差分比较。
5. 对不同段数 `M in {1, 3, 5, 8}` 和非均匀时间运行随机回归测试。
6. 与项目当前的五次 minimum-jerk 轨迹生成器比较边界、采样曲线和 jerk 积分。

### 6.8 阶段 1 验收标准

| 项目 | 初始验收值 |
|---|---:|
| `q(0)` / `q(T)` 误差 | `< 1e-7 rad` |
| `qd(0)` / `qd(T)` 误差 | `< 1e-6 rad/s` |
| `qdd(0)` / `qdd(T)` 误差 | `< 1e-5 rad/s^2` |
| 中间插值构型误差 | `< 1e-7 rad` |
| 平滑能量梯度相对误差 | `< 1e-3` |
| 优化输出 | 无 NaN/Inf，solver 状态和失败原因完整 |
| 轨迹限制 | `q/qd/qdd` 无未报告超限 |

这些是数值单测阈值，若不同平台上的线性代数误差超出，应先根据量级和参考结果调整容差，不得为让测试通过而删除边界校验。

### 6.9 阶段 1 输出

```text
data/results/ccro_stage1/
  config.yaml
  test_cases.json
  gradient_check.json
  timing.json
  table_nubs_backend.md
  trajectory_examples.png
```

---

## 7. 阶段 2：静态障碍 Mesh 全身风险优化

### 7.1 目标

在固定分段时间下，使 `P_inner` 优化能够降低静态障碍点云对机器人全身的风险，并通过独立高密度复核。

### 7.2 优化风险不直接使用 `D_min`

对轨迹时刻 `t_i` 下的每个连杆表面点查询障碍距离 `d_lk`，定义：

```text
R_body(q, t) = sum_l w_l / N_l * sum_k phi(d_safe - d_lk(q, t))
```

第一版建议使用平方 hinge：

```text
z = d_safe - d
phi(z) = 0       , z <= 0
phi(z) = z^2     , z > 0
```

该函数在 `z=0` 一阶连续，数值解释直接。若后续需要更平滑的梯度，再切换到数值稳定的 softplus：

```python
softplus = np.logaddexp(0.0, beta * z) / beta
```

需注意 softplus 在 `d=d_safe` 时的值是 `log(2)/beta`，且在安全区内存在小的正尾项，不能将其描述为严格为零。

`D_min(q,t)` 另行计算，只用于激活窗口、报告和复核。

### 7.3 风险积分

为避免短分段被全局均匀采样漏掉，采样按每个 NUBS 分段进行。第一版使用每段固定数量的梯形积分点，后续可切换到 Gauss-Legendre 积分。

```text
J_risk ~= sum_i quadrature_weight_i * R_body(q(t_i), t_i)
```

代价实现必须同时返回：

```python
RiskEvaluation(
    cost: float,
    min_distance: float,
    nearest_link: str | None,
    active_sample_count: int,
    per_link_cost: dict[str, float],
)
```

### 7.4 梯度路线

梯度必须按以下顺序实现，不直接跳到在线半解析版。

#### Level 0：完整目标的外层有限差分基准

对每个 `P_inner[r,j]` 重新生成轨迹并计算完整 `J`，得到中心有限差分。它只用于小场景和梯度正确性基准，不作为最终规划性能实现。

#### Level 1：轨迹采样构型梯度 + NUBS 传播

1. 对每个活跃时间采样点计算 `dR/dq`：

```text
dR/dq_j ~= [R(q + eps*e_j) - R(q - eps*e_j)] / (2*eps)
```

2. 使用 NUBS 位置基函数将采样梯度累积为 `dJ/dC`。
3. 调用 NUBS 伴随/梯度传播获得 `dJ/dP_inner`。
4. 与 Level 0 在相同 `P_inner`上比较相对误差和余弦相似度。

为实现这一步，pybind/C++ 层需增加下列接口之一：

```python
basis_matrix(times, derivative_order=0)
propagate_coefficient_gradient(dJ_dC, dJ_dT_direct=None)
```

或直接暴露更高层的：

```python
propagate_sample_gradients(times, dJ_dq, quadrature_weights)
```

第二种接口更不容易在 Python 端搞错控制点排列，推荐作为最终对外接口。

#### Level 2：风险时间窗和关节维度筛选

仅对 `D_min < d_activate` 的采样时刻计算风险梯度，并根据最近风险连杆只扰动可影响该连杆的上游关节。

#### Level 3：连杆点 Jacobian 半解析梯度

在 Level 1/2 结果稳定后，使用最近表面点或活跃表面点的点 Jacobian 计算 `dd/dq`。该阶段主要用于降低在线重规划耗时，不是静态正确性验证的前置条件。

### 7.5 静态优化目标

```text
J = lambda_s * J_smooth
  + lambda_r * J_risk
  + lambda_q * J_q
  + lambda_v * J_v
  + lambda_a * J_a
```

时间固定时不加 `J_time`；终点是硬边界时不加 `J_goal`。

权重调试顺序：

1. 关闭风险项，确认无障碍轨迹和限制惩罚正常。
2. 只开启风险与平滑项，调整两者的数值量级。
3. 开启 `q/qd/qdd` 惩罚，保证各项梯度不被单一权重完全淹没。
4. 不以单一场景的手工权重作为所有实验结论，至少在三类静态场景中复用同一组主权重。

### 7.6 候选轨迹复核器

复核器必须与优化代价独立，使用更高 Mesh 密度和更细时间步长：

```text
GoalOK        = ||q(T) - q_goal|| <= epsilon_goal
FiniteOK      = 所有轨迹值均为有限数
DistanceOK    = dense D_min >= d_stop
PositionOK    = q_min <= q(t) <= q_max
VelocityOK    = |qd(t)| <= qd_max
AccelerationOK= |qdd(t)| <= qdd_max
ContinuityOK  = 起始 q/qd/qdd 与当前状态匹配
SolverOK      = 求解器正常结束且未超时
```

真实机械臂阶段还必须增加：

```text
SelfCollisionOK = 不属于允许对的连杆之间无自碰撞
TimestampOK     = 采样时间严格递增
SwitchOK        = 控制周期内无位置/速度跳变
```

当前项目没有自碰撞模块。离线阶段必须将其标记为“未检查”，不得默认为通过；在允许真实机械臂执行前必须补齐。

### 7.7 静态实验场景

| 场景 | 目的 | 必要对比 |
|---|---|---|
| A：末端原路径附近单障碍 | 验证基本绕行 | 无风险 NUBS、末端风险、全身风险 |
| B：肘部/小臂附近障碍 | 验证末端模型的漏检 | 末端风险 vs Mesh 全身风险 |
| C：多障碍/窄通道 | 验证优化边界 | 必须同时报告失败率，不预设必然成功 |

场景优先使用程序化点云构造以便重复，随后再用录制 RGB-D 点云做外部有效性验证。

### 7.8 阶段 2 验收指标

不将“规划成功率 > 90%”或“风险降低 > 50%”直接设为未经先导实验的硬指标。第一轮先完成下列可证伪验收：

```text
[ ] 同一优化结果在中密度代价和高密度复核下都有完整报告
[ ] 全身风险优化后 J_risk 相对初始轨迹下降
[ ] 候选轨迹满足目标、有限性和 q/qd/qdd 判据
[ ] 如候选轨迹 D_min < d_stop，必须被复核器拒绝
[ ] 场景 B 存在末端安全但中间连杆危险的反例
[ ] 规划时间报告 mean/p95/max，不只报告平均值
[ ] 优化失败、超时和复核失败分开统计
```

---

## 8. 阶段 2B：时间变量的保守开放

时间优化不与第一版静态绕行同时开启。建议顺序为：

1. 固定每段 `T_i`，只优化 `P_inner`。
2. 保持各段时间比例，只优化一个总时长 `T_total`。这可直接复用 NUBS 现有 fixed-ratio total-duration 梯度。
3. 只有在总时长优化稳定后，才开放每段时间。

完整分段时间使用无约束参数 `z_i` 保证正值：

```text
T_i = T_min + softplus(z_i)
```

开放时间后的目标为：

```text
J = J_fixed_time + lambda_t * sum_i T_i
```

外部风险项对时间的梯度先用总时长有限差分或完整外层有限差分验证，不能只使用 NUBS 平滑能量的 `dE/dT` 就宣称完整目标时间梯度正确。

---

## 9. 阶段 3：动态障碍时空风险优化

### 9.1 目标

将对象跟踪结果转换为可在任意轨迹时刻查询的预测占据，使风险代价中的机器人构型与障碍未来位置按同一物理时间对齐。

### 9.2 `ObstacleForecast` 接口

```python
class ObstacleForecast(Protocol):
    @property
    def valid_horizon(self) -> float: ...

    def occupancy_at(self, tau: float) -> "ObstacleOccupancy": ...


@dataclass(frozen=True)
class ObstacleOccupancy:
    points: np.ndarray | None
    spheres: tuple[RiskSphere, ...]
    timestamp_offset: float
    extrapolated: bool
```

实现至少包含：

- `StaticPointCloudForecast`：在任意 `tau` 返回同一静态点云。
- `ConstantVelocityObjectForecast`：使用 `p(tau)=p0+v*tau`，同时使半径/不确定性随 `tau` 增大。
- `CompositeForecast`：合并静态背景、当前动态对象和多个跟踪对象。

动态对象的 `tau=0` 必须包含当前占据，不能只生成从 `prediction_step` 开始的未来球而遗漏当前障碍。

### 9.3 预测时域与规划时域一致性

当前 `safety.yaml` 默认预测时域为 `0.4 s`，因此不能直接在调度器中写死 `planning_horizon=3.0 s` 而不定义后 `2.6 s` 的障碍行为。

阶段 3 采用以下规则：

1. 离线匀速数据可将预测时域提高到 `1.0 s` 左右，但必须单独评估预测误差。
2. `H_eval <= forecast.valid_horizon`；超出时域的占据不得默认消失。
3. 如必须规划更长的完整到达轨迹，超出有效预测时域后可保持最后预测中心并继续膨胀，但必须将其标记为 `extrapolated=True` 并在实验中统计。
4. 在线轨迹评估只依赖短时有效风险窗口，风险窗口之外由下一次感知更新和高频安全层补充。

### 9.4 时空风险积分

```text
J_risk = sum_i w_i * R_body(q(t_i), occupancy_at(t_i))
```

每个轨迹采样点的 `t_i` 是相对当次规划起点的物理秒，不是归一化参数。录制数据回放时要明确区分：

```text
record_timestamp      录制序列中的绝对/相对时间
planning_start_time   当次优化的起始时间
trajectory_tau        轨迹内部时间
```

障碍预测必须以 `planning_start_time + trajectory_tau` 查询，避免每次重规划都把障碍时间重置错位。

### 9.5 阶段 3 实现顺序

1. 使用程序化匀速球/点云验证时间对齐。
2. 固定 `T`，只优化 `P_inner`，比较当前帧静态风险与时空风险。
3. 使用 ch4_3 录制序列验证末端和中间连杆场景。
4. 单独统计预测失配、过度膨胀和有效时域外推比例。
5. 固定时间版本稳定后，先开放总时长，最后才开放完整分段时间。

### 9.6 对比方法

```text
B0: 无风险 NUBS
B1: NUBS + 当前帧末端风险
B2: NUBS + 当前帧 Mesh 全身风险
B3: NUBS + Mesh 全身时空风险
```

四种方法必须使用相同起终状态、段数、初始时间、优化迭代/超时预算和高密度复核器。

### 9.7 阶段 3 输出指标

```text
目标到达率
全身 D_min
低于 d_safe 和 d_stop 的累计时间
J_risk 及风险降低率
路径长度、轨迹时长和 jerk 相对增量
优化成功/超时/数值失败率
高密度复核拒绝率
预测失配数量与安全层接管次数
规划耗时 mean/p95/max
```

---

## 10. 阶段 4：风险触发重规划与安全层协同

### 10.1 目标

实现事件触发的剩余轨迹优化，在优化尚未完成、失败或候选轨迹被拒绝时，保留速度级安全层的独立控制权。

### 10.2 轨迹缓冲和 warm start

`TrajectoryBuffer` 至少提供：

```python
class TrajectoryBuffer:
    def set_active(self, trajectory, start_timestamp: float) -> None: ...
    def sample_now(self, timestamp: float) -> JointState: ...
    def sample_future(self, timestamp: float, horizon: float) -> TrajectorySample: ...
    def remaining_waypoints(self, timestamp: float, durations: np.ndarray) -> np.ndarray: ...
```

warm start 不是直接拷贝上一轨迹的 `P_inner`，因为新轨迹的起始时刻和剩余分段已变化。正确步骤为：

1. 以当前反馈 `q_now/qd_now/qdd_now` 作为新轨迹头部硬边界。
2. 按新分段节点时间采样上一轨迹的剩余部分。
3. 将这些采样构型作为新 `P_inner` 初值。
4. 如上一轨迹已不可用，再回退到当前状态到 `q_goal` 的插值初值。

### 10.3 触发状态机

```text
LOW:
  未来评估时域内 D_min >= d_replan
  -> 继续执行当前轨迹，不调用优化器

MEDIUM:
  d_stop < D_min < d_replan
  -> 触发低频重规划
  -> 安全层可同时限速

HIGH:
  D_min <= d_stop 或碰撞时间余量低于刹车预算
  -> 安全层立即停止/制动
  -> 不等待优化器返回
```

`d_replan` 建议不小于 `d_safe`，并使用进入/退出滞回阈值和最小重规划间隔，避免在阈值附近每帧触发。

### 10.4 重规划任务模型

```python
class ReplanManager:
    def evaluate_active_trajectory(
        self,
        now: JointState,
        active: TrajectoryBuffer,
        forecast: ObstacleForecast,
    ) -> FutureRiskReport: ...

    def request_replan(
        self,
        now: JointState,
        q_goal: np.ndarray,
        active: TrajectoryBuffer,
        forecast: ObstacleForecast,
    ) -> None: ...

    def poll_candidate(self) -> OptimizationResult | None: ...

    def accept_candidate(
        self,
        result: OptimizationResult,
        verification: VerificationResult,
    ) -> bool: ...
```

优化任务必须有硬超时预算。第一版可放在独立进程，主安全循环只提交任务和轮询结果，不在感知/安全线程中同步等待 L-BFGS。

### 10.5 候选轨迹接受门槛

```text
Accept = GoalOK
      and FiniteOK
      and DistanceOK
      and PositionOK
      and VelocityOK
      and AccelerationOK
      and ContinuityOK
      and SolverOK
      and ValidationOK
```

在真实机械臂自动切换时还要加入 `SelfCollisionOK` 和 `SwitchOK`。任一条件不满足都必须拒绝候选轨迹，且日志中保存具体原因，不使用单一 `success=False` 覆盖所有失败类型。

### 10.6 与现有安全层集成

每个控制周期执行：

```text
q_ref, qd_ref = active_trajectory.sample(now)
distance/risk = fast_safety_evaluation(current_observation, q_now)
command = safety_layer(qd_ref, distance, repulsive_velocity)
send(command)
```

规划器只能更换 `active_trajectory`，不能关闭或绕过安全层。

需要特别注意：当前真机接口主要是 `movej`、直线往复和速度缩放，尚没有验证过的时间标记关节轨迹缓冲执行器。因此阶段 4 的默认产物是虚拟闭环和影子模式，不直接允许实机自动切换。

### 10.7 验证顺序

#### A. 纯调度器单测

- LOW 风险不调用优化器。
- MEDIUM 风险只创建一个未完成任务，不重复堆积。
- HIGH 风险立即输出安全接管状态。
- 超时、优化失败和复核失败均不切换轨迹。
- 滞回和最小间隔能防止触发抖动。

#### B. 录制序列影子模式

- 使用真实 RGB-D 和关节反馈录制数据。
- 只记录触发、优化、复核和假定切换结果。
- 不将候选轨迹发送给机械臂。

#### C. 虚拟闭环

- 虚拟执行器按已接受轨迹更新 `q_sim/qd_sim/qdd_sim`。
- 各方法使用各自的虚拟构型重新计算 Mesh 风险。
- 报告目标到达、风险、重规划和安全层介入。

#### D. 真实机械臂分阶段验证

1. 只显示候选轨迹和复核状态。
2. 实机仍执行原参考轨迹，NUBS 不接管。
3. 补齐自碰撞、轨迹缓冲、watchdog 和通信异常停止后，低速允许已复核轨迹切换。
4. 从空场景、软性静态障碍、慢速动态障碍逐级扩展。

### 10.8 阶段 4 实时性目标的表述

第一版重规划工程目标为约 `1 Hz`，但只有当 `p95` 和超时率实测通过时才可如此表述。当前 4.6 结果中控制生成段 `p95` 约为 `200 ms`，且缺少完整感知耗时，因此尚不能宣称完整高频链路已达标。

每次重规划至少记录：

```text
T_risk_eval
T_warm_start
T_optimize
T_validate
T_total
solver_iterations
timeout
candidate_accepted
reject_reasons
```

---

## 11. 实验矩阵与论文对应

### 11.1 主实验

| CCRO-NUBS 实验 | 对应内容 | 依赖 | 当前状态 |
|---|---|---|---|
| E1 感知与风险前端 | 复用 4.2/4.3 | 现有结果 | 已有 |
| E2 构型耦合全身风险 | 复用 4.4 并补采样密度实验 | P1 | 部分已有 |
| E3 NUBS 后端验证 | 边界、MINCO、梯度、耗时 | 阶段 1 | 已完成 |
| E4 静态 Mesh 全身风险规划 | A/B/C 静态场景 | 阶段 2 | 程序化场景已完成；录制点云待补 |
| E5 动态时空风险轨迹优化 | 程序化场景 + 录制预警指标 | 阶段 3 | 程序化 A/B/C 已通过；录制数据仅支持预警提前量，缺三维中心真值 |
| E6 风险触发重规划 | 程序化异步参考轨迹闭环 | 阶段 4 | 动态 A/C、无触发 B、HIGH 接管 D 已通过；录制影子模式待补 |
| E7 真实机械臂 | 分阶段安全验证 | 执行器和自碰撞补齐 | 扩展 |

### 11.2 对比与消融

```text
A1: 无风险 NUBS
A2: NUBS + 当前帧 Mesh 全身风险
A3: NUBS + 时空风险，仅末端 Mesh
A4: NUBS + Mesh 全身时空风险，无在线重规划
A5: A4 + 风险触发重规划，无速度级安全层
A6: 完整方法，重规划 + 速度级安全层
```

如计算预算允许，增加：

```text
A7: 固定 T vs 可优化总时长
A8: 单一 D_min 代价 vs 多表面点 R_body
A9: 全量构型有限差分 vs 风险窗口有限差分
```

### 11.3 统一结果字段

每个场景和方法至少输出：

```text
scenario_id, method, seed
goal_error, reached_goal
min_distance, time_below_d_safe, time_below_d_stop
risk_before, risk_after, risk_reduction
path_length, total_duration, jerk_integral
q_violation, qd_violation, qdd_violation
solver_success, timeout, numeric_failure, iterations
validation_accepted, reject_reasons
planning_ms, validation_ms, total_ms
replan_count, replan_success_count, safety_takeover_count
```

不使用缺失值代表零；不把求解器返回、复核通过和实际目标到达合并为同一个 `success`。

---

## 12. 测试和回归策略

### 12.1 单元测试

- `RobotSurfaceModel`：缓存一致性、单位、按连杆输出、空 Mesh 处理。
- `NUBSTrajectory6D`：边界、插值、导数、非法时间、NaN 输入。
- `MeshRiskEvaluator`：空障碍、安全区、边界、穿透、多连杆权重。
- `TrajectoryObjective`：分项代价与总代价一致，梯度 shape 和有限性。
- `TrajectoryVerifier`：为每一种拒绝原因构造一条确定性轨迹。
- `ReplanManager`：状态转移、滞回、超时、重复任务抑制。

### 12.2 梯度测试

每种梯度保留两个判据：

```text
relative_error = ||g - g_fd|| / max(||g_fd||, epsilon)
cosine         = dot(g, g_fd) / (||g|| * ||g_fd||)
```

在风险代价最近点发生切换时，相对误差可能被局部非光滑放大。此时应同时查看梯度方向、小步长下的代价下降和多点聚合结果，不能仅为了达到严格相对误差而人为挑选完全平滑的样本。

### 12.3 集成测试门槛

```text
阶段 1 通过 -> 才允许合并风险目标
阶段 2 固定 T 通过 -> 才允许开放时间变量
静态时间优化通过 -> 才允许动态时间对齐
录制序列影子模式通过 -> 才允许虚拟轨迹切换
虚拟闭环通过 + 执行安全项补齐 -> 才评估真实机械臂切换
```

---

## 13. 时间线与开发优先级

时间预估以一名熟悉当前项目和 C++/Python 混合编译的开发者为参考，不包括硬件排期和重新录制数据的等待时间。

| 优先级 | 任务 | 参考工作量 | 出口条件 |
|---|---|---:|---|
| P0 | 版本、py310 测试、规划限制和配置 | 1-2 天 | CTest/Python 测试可复现 |
| P1 | `RobotSurfaceModel` 正式化 | 2-4 天 | 与 4.4 距离结果回归一致 |
| 阶段 1 | 6-DOF NUBS pybind 和无障碍验证 | 3-5 天 | 边界、能量、梯度和限制通过 |
| 阶段 2 | 静态全身风险优化和复核 | 5-8 天 | A/B/C 场景与失败统计完整 |
| 阶段 2B | 总时长及分段时间优化 | 3-5 天 | 完整目标时间梯度验证通过 |
| 阶段 3 | 动态时空风险和录制序列 | 4-7 天 | 时间对齐和预测失配统计完整 |
| 阶段 4 | 影子重规划和虚拟闭环 | 5-8 天 | 超时、复核、切换和接管逻辑通过 |
| 真机扩展 | 轨迹执行器、自碰撞、watchdog | 单独排期 | 通过安全审查和低速逐级验证 |

推荐的第一批实际编码范围仅包含：

```text
P0 版本/环境/配置基线
  -> P1 RobotSurfaceModel
  -> 阶段 1 NUBS pybind
  -> 6-DOF 无障碍固定时间轨迹测试
```

不建议在第一批中同时修改动态预测、实机执行和重规划调度，否则难以判断数值失败来自 NUBS、风险梯度还是执行接口。

---

## 14. 关键风险与降级方案

| 风险 | 表现 | 优先处理 | 允许的降级方案 |
|---|---|---|---|
| NUBS 封装构建失败 | pybind/CMake 无法导入 | 先独立编译最小 6D 模块 | 用 C++ 可执行文件跑阶段 1，不立即重写整套 Python NUBS |
| 优化不收敛 | ABNORMAL_TERMINATION / 梯度爆炸 | 关闭风险项分项检查，对目标和梯度做量级报告 | 固定 T、降低段数、使用外层 FD 基准 |
| 最近点切换导致梯度抖动 | 小扰动梯度方向突变 | 多表面点风险聚合，减小 FD 步长扫描范围 | 加大体素分辨率、风险时间窗和梯度裁剪 |
| Mesh 复核过慢 | validation p95 过高 | 复用局部采样缓存和按连杆批量变换 | 优化可用 medium，但真机候选不允许跳过 dense 复核 |
| 预测失配 | 优化轨迹对真实障碍无效 | 缩短有效时域、增加不确定性、提高感知更新 | 拒绝候选轨迹并由安全层接管 |
| 重规划超时 | 中风险时无新轨迹 | 风险窗口、warm start、局部关节有限差分 | 保留原轨迹的限速状态或停止 |
| 安全控制 p95 过高 | 控制周期尾延迟 | 补齐完整埋点，优化表面距离梯度 | 降低执行速度并放大刹车裕量，不宣称高频已达标 |
| 真机轨迹切换不连续 | 位置/速度突跳 | 使用当前反馈硬约束新轨迹头状态，增加缓冲执行器 | 停留在影子模式，不允许自动切换 |

---

## 15. 论文成立节点

不同完成程度对应的可宣称能力如下：

```text
完成阶段 1：
  可宣称已接入 NUBS minimum-jerk 连续轨迹后端。

完成阶段 2：
  可宣称 Mesh 全身风险已进入 NUBS 轨迹优化，
  并在静态障碍下验证了风险降低与安全复核。

完成阶段 3：
  可宣称动态障碍的时空预测占据已与轨迹物理时间对齐。

完成阶段 4 的影子模式和虚拟闭环：
  论文“构型耦合时空风险进入连续轨迹优化，
  并用于风险触发的候选轨迹重规划”的主体即成立。

只有完成真机低速自动切换与端到端实时性验证：
  才宣称具备真实机械臂动态在线重规划执行能力。
```

---

## 16. 第一批实施清单

为了尽快形成第一个可验收里程碑，实际编码建议严格按以下顺序执行：

1. 固定 NUBS 源码位置和版本，在 `py310` 中运行原 CTest/依赖检查。
2. 新建 `config/planning.yaml`，先填写已确认的关节顺序，再从手册/SDK 核对限制。
3. 将 `RobotPointRemover` 的 Mesh 预采样和 FK 表面变换抽取到 `planning/robot_surface_model.py`，保留旧接口回归对比。
4. 实现 `planning/cpp/nubs_pybind.cpp`，只封装 6D minimum jerk 所需最小接口。
5. 实现 `planning/nubs_trajectory.py`，处理 numpy shape、单位、异常和批量采样。
6. 新增 `tests/planning/test_nubs_binding.py` 和 `test_nubs_gradient.py`。
7. 实现无障碍固定时间 `P_inner` 优化与 `exp_ccro_stage1.py`。
8. 生成阶段 1 的边界误差、梯度误差、限制违反和耗时报告。
9. 只有阶段 1 全部通过后，才创建 `mesh_risk.py` 并进入静态风险优化。

---

> **文档版本**：v2.1  
> **最后更新**：2026-06-23  
> **Python 环境**：`py310`（可用，依赖按实施步骤核查）  
> **对应论文框架**：`document/CCRO-NUBS_论文框架与实现路线.md`
