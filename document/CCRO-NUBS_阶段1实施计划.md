# CCRO-NUBS 阶段 1 实施计划

> 目标：在不引入障碍风险的条件下，打通 6 自由度 minimum-jerk NUBS 的 C++ 构造、Python 调用、固定时间中间构型优化、自动测试和实验结果生成。

---

## 1. 阶段边界

阶段 1 只回答以下问题：

1. 现有 `NUBSTrajectory.hpp` 能否稳定生成 AUBO 六关节连续轨迹；
2. 起终位置、速度和加速度以及中间插值构型是否满足；
3. minimum-jerk 能量和对 `P_inner` 的梯度是否正确；
4. SciPy L-BFGS-B 能否在固定分段时间下优化中间插值构型；
5. 输出轨迹是否满足阶段配置中的关节位置、速度和加速度限制；
6. 所有结果能否在 `py310` 中通过一个命令重复生成。

阶段 1 明确不包含：

- 静态或动态障碍风险；
- Mesh 距离积分；
- 分段时间联合优化；
- 风险触发重规划；
- 真实机械臂指令发送。

这些内容提前混入会导致 NUBS 数值错误、风险梯度错误和执行接口错误难以区分。

---

## 2. 数学变量约定

系统阶数固定为 `s=3`，B 样条次数为：

```text
p = 2s - 1 = 5
```

轨迹维度固定为 6：

```text
q(t) in R^6
```

给定 `M` 段轨迹：

```text
durations = [T_1, ..., T_M]
P_inner.shape = (M-1, 6)
```

`P_inner` 是内部时间节点处的插值关节构型，不是最终 B 样条控制点。NUBS 内部通过带状线性系统恢复：

```text
control_points.shape = (M + 2s - 1, 6)
```

边界状态固定为：

```text
head_state = [q_start, qd_start, qdd_start]
tail_state = [q_goal, qd_goal, qdd_goal]
```

其中阶段 1 默认 `qd_goal=0`、`qdd_goal=0`。因为 `q_goal` 已经是尾端硬边界，优化目标中不重复加入目标误差项。

---

## 3. 实现结构

```text
NUBSTrajectory-main/include/NUBSTrajectory.hpp
        |
        v
planning/cpp/nubs_pybind.cpp
        |
        v
planning/_nubs_cpp*.so
        |
        v
planning/nubs_trajectory.py
        |
        +--> planning/optimizer.py
        |
        +--> experiments/exp_ccro_stage1.py
        |
        +--> tests/planning/test_nubs_stage1.py
```

### 3.1 C++ 绑定职责

`planning/cpp/nubs_pybind.cpp` 直接持有：

```cpp
nubs::QuinticNUBS<6>
```

绑定只负责高性能核心能力：

- `generate()`；
- 单时刻 `evaluate()`；
- 批量 `sample()`；
- `energy()`；
- `energy_and_gradient()`；
- 读取分段时间和控制点。

阶段 1 不修改 NUBS 数学内核，不复制 de Boor 和带状求解代码。

### 3.2 Python 轨迹包装职责

`planning/nubs_trajectory.py` 负责：

- 检查所有数组维数、shape 和有限性；
- 明确输入单位为 rad、rad/s、rad/s² 和 s；
- 生成边界状态矩阵；
- 按累计时间比例生成初始 `P_inner`；
- 提供批量采样、密集采样、边界误差和中间插值误差；
- 当扩展未构建时返回明确的构建命令。

### 3.3 固定时间优化器职责

`planning/optimizer.py` 只优化：

```text
x = flatten(P_inner)
```

分段时间 `T_i` 保持不变。目标为：

```text
J = lambda_smooth * J_smooth
  + lambda_position * J_q
  + lambda_velocity * J_qd
  + lambda_acceleration * J_qdd
```

其中：

- `J_smooth` 直接使用 NUBS 的精确 minimum-jerk 能量和内置梯度；
- 限制惩罚使用分段密集采样；
- 限制惩罚激活时，对惩罚部分使用 `P_inner` 中心有限差分；
- `P_inner` 本身使用 `q_min/q_max` 作为 L-BFGS-B bounds；
- B 样条段内仍可能超出 waypoint bounds，因此最终必须密集采样检查。

阶段 1 的示例配置通过较保守时长使速度和加速度惩罚不激活，优先验证 NUBS 内置能量梯度。

---

## 4. 文件清单

| 文件 | 作用 |
|---|---|
| `planning/cpp/CMakeLists.txt` | 定位 pybind11、Eigen3 和 NUBS 头文件 |
| `planning/cpp/nubs_pybind.cpp` | 6-DOF minimum-jerk C++ 绑定 |
| `planning/nubs_trajectory.py` | Python 轨迹 API 和输入验证 |
| `planning/optimizer.py` | 固定时间 L-BFGS-B 优化器 |
| `config/ccro_stage1.yaml` | 阶段 1 的场景、限制、优化和验收参数 |
| `experiments/exp_ccro_stage1.py` | 梯度检查、优化、指标、图和表格入口 |
| `tests/planning/test_nubs_stage1.py` | 边界、梯度、优化和非法输入测试 |
| `scripts/build_ccro_stage1.sh` | 只构建 C++ 扩展 |
| `scripts/setup_ccro_stage1.sh` | 安装阶段依赖、构建并测试 |
| `requirements-stage1.txt` | 阶段 1 Python 依赖 |

构建产物位于：

```text
planning/_nubs_cpp*.so
planning/cpp/build/
```

实验产物位于：

```text
data/results/ccro_stage1/
  config.yaml
  metrics.json
  table_stage1.md
  trajectory_stage1.png
```

---

## 5. 环境配置

项目统一使用已有的 `py310` 环境：

```bash
cd /home/hzy/Code/manipulator_occupancy
bash scripts/setup_ccro_stage1.sh
```

脚本执行顺序：

1. 在 `py310` 中安装/核对 `requirements-stage1.txt`；
2. 获取 `py310` 的 Python prefix 和 pybind11 CMake 目录；
3. 使用 `py310` 中的 CMake、Eigen3 和系统 C++ 编译器构建扩展；
4. 导入 `planning._nubs_cpp` 做烟雾测试；
5. 运行阶段 1 自动测试。

如只需重新编译扩展：

```bash
bash scripts/build_ccro_stage1.sh
```

如需手动运行测试：

```bash
/home/hzy/miniconda3/bin/conda run -n py310 \
  python -m pytest -p no:cacheprovider tests/planning/test_nubs_stage1.py -q
```

---

## 6. 运行实验

默认配置运行：

```bash
/home/hzy/miniconda3/bin/conda run -n py310 \
  python -m experiments.exp_ccro_stage1
```

指定配置或输出目录：

```bash
/home/hzy/miniconda3/bin/conda run -n py310 \
  python -m experiments.exp_ccro_stage1 \
  --config config/ccro_stage1.yaml \
  --output data/results/ccro_stage1_custom
```

实验入口会：

1. 加载起终状态和显式非均匀分段时间；未配置 `segment_durations` 时才回退到均匀时间；
2. 生成线性时间比例的 `P_inner`；
3. 构造初始 NUBS 轨迹；
4. 将内置 `dE/dP_inner` 与外部中心有限差分比较；
5. 调用 L-BFGS-B 优化 `P_inner`；
6. 以 `0.01 s` 默认步长密集采样；
7. 检查边界、插值、能量、关节限制和求解状态；
8. 统计构造与 101 点批量采样耗时；
9. 输出 JSON、Markdown 表格和轨迹图；
10. 任一核心验收项失败时以非零退出码结束。

---

## 7. 梯度验证

对每个内部插值构型变量计算：

```text
g_fd[r,j] = [E(P + eps*e_rj) - E(P - eps*e_rj)] / (2*eps)
```

同时报告：

```text
relative_error = ||g_nubs - g_fd|| / max(||g_fd||, 1e-12)
cosine = dot(g_nubs, g_fd) / (||g_nubs|| ||g_fd||)
max_absolute_error
```

默认门槛：

```text
relative_error <= 1e-3
cosine 接近 1
```

如果相对误差失败，排查顺序为：

1. 检查 `P_inner` 的 `(M-1, 6)` 排列；
2. 扫描 `eps in {1e-4, 1e-5, 1e-6, 1e-7}`；
3. 对比 NUBS 的 local FD 和 full FD 梯度；
4. 检查小分段时间和条件数；
5. 最后才调整测试容差。

---

## 8. 验收标准

阶段 1 必须同时满足：

| 检查项 | 默认门槛 |
|---|---:|
| 起点/终点位置误差 | `<= 1e-7 rad` |
| 起点/终点速度误差 | `<= 1e-6 rad/s` |
| 起点/终点加速度误差 | `<= 1e-5 rad/s²` |
| 内部插值构型误差 | `<= 1e-7 rad` |
| 能量梯度相对误差 | `<= 1e-3` |
| 优化后能量 | 不高于初始能量（允许数值容差） |
| 求解结果 | solver 正常结束，无 NaN/Inf |
| 关节位置/速度/加速度 | 密集采样下无未报告超限 |

此外必须保留以下信息，即使验收通过：

- SciPy solver status 和 message；
- 迭代次数、函数调用次数和最终梯度范数；
- NUBS 精确能量与密集采样 jerk 积分差异；
- 构造、批量采样和优化耗时；
- 使用的完整配置副本。

---

## 9. 已知限制

1. 配置中的关节限制是阶段 1 的保守实验限制，不是制造商硬件额定值；真实执行前必须从 AUBO 手册和 SDK 重新核对。
2. 当前只开放 `P_inner`，不优化总时长或各段时长。
3. 当前限制惩罚对活跃约束使用有限差分，适合离线正确性验证，不是后续在线最终实现。
4. 阶段 1 不检查外部碰撞和自碰撞，因为没有障碍且不向真实机械臂发送轨迹。
5. NUBS 头文件目前位于未跟踪目录，正式论文实验前必须固定其版本。
6. 端点求值在 C++ 内部使用终点左极限，允许存在浮点量级的尾端误差。

---

## 10. 阶段完成后的下一步

只有以下文件都生成且 `metrics.json` 中 `accepted=true` 后，才进入阶段 2：

```text
planning/_nubs_cpp*.so
data/results/ccro_stage1/metrics.json
data/results/ccro_stage1/table_stage1.md
data/results/ccro_stage1/trajectory_stage1.png
```

阶段 2 的第一项工作不是动态预测，而是实现正式 `RobotSurfaceModel` 和静态障碍多表面点风险积分。

---

> 文档版本：v1.0  
> Python 环境：`py310`  
> 对应总方案：`document/CCRO-NUBS_实现方案.md`
