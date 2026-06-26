# CCRO-NUBS 实时性优化记录

生成时间：2026-06-26

## 1. 当前判断

P0–P7 的仿真流程已经跑通，现阶段主要问题不是算法链路断裂，而是虚拟闭环控制段实时性不足。

原 `data/results/ch4_6/timing.json` 显示：

- `T_pre_ms/T_dec_ms/T_obj_ms/T_trk_ms/T_risk_ms` 均为 0；
- `T_rep_ms = T_cmd_ms = T_frame_ms`；
- 控制段平均耗时 `40.8883 ms`；
- 控制段 p95 耗时 `199.8850 ms`。

因此该 timing 不能作为完整端到端感知闭环实时性结论，只能说明原 4.5 虚拟闭环中的排斥速度/安全滤波实现较慢。

## 2. 瓶颈来源

主要瓶颈为：

1. `OursFullController` 先计算 `repulsive_velocity("ours")`，随后又单独调用 `distance_gradient()`；
2. `distance_gradient()` 原实现对每个关节做中心有限差分，每个关节需要两次完整 `distance_for_q()`；
3. `distance_for_q()` 会重复生成机器人表面点云、体素降采样并构建 KD-tree；
4. `ours` 动态预测会堆叠未来点云，使障碍点数量放大；
5. 虚拟闭环默认 `mesh_samples=50000`，更适合离线复核/可视化，不适合在线控制。

## 3. 已实施修改

### 3.1 4.5 控制器链路优化

修改文件：

- `experiments/exp_44_main.py`
- `experiments/exp_45_controller.py`
- `experiments/exp_45_runner.py`
- `experiments/exp_45_virtual_loop.py`

核心修改：

1. 新增 `RepulsionDetails44`，使排斥速度函数同时返回：
   - `velocity`
   - `gradient`
   - `distance`
   - `active_point_count`
2. `OursFullController` 复用 `repulsive_velocity_details("ours")` 返回的梯度，不再额外调用一次 `distance_gradient()`。
3. `distance_gradient()` 默认从完整有限差分改为 `fast_distance_gradient()`：
   - 先确定最近障碍点与最近机器人局部采样点；
   - 梯度只追踪该局部点随关节扰动后的距离变化；
   - 不再对每个关节重复构建完整表面点云和 KD-tree。
4. 新增同一构型下的机器人表面点云缓存和 KD-tree 缓存。
5. `ours` 当前距离优先复用 `frame.ref.d_ref`，避免重复计算当前风险距离。
6. 在线/虚拟闭环默认 `mesh_samples` 从 `50000/20000` 调整为 `10000`；dense 或论文图示仍可显式传入更高采样数。

### 3.2 P4/A6 在线执行层优化

新增文件：

- `planning/fast_sphere_risk.py`

修改文件：

- `experiments/exp_ccro_p4.py`
- `config/ccro_p4.yaml`
- `tests/planning/test_ccro_p4.py`

核心修改：

1. 新增 `FastSphereRiskEvaluator`：
   - 将每个机器人连杆用少量局部包围球表示；
   - 障碍使用阶段三/四已有对象级预测球；
   - 在线执行层计算 `distance + gradient` 时不再使用全 mesh 有限差分。
2. P4/A6 的速度级安全执行层使用 `FastSphereRiskEvaluator`。
3. A6 的每一步安全结果仍用 dense mesh evaluator 做距离复核，因此不会把验收标准替换成粗模型。

## 4. 新实验结果

### 4.1 P4/A6 快速执行层

命令：

```bash
/home/hzy/miniconda3/envs/py310/bin/python -m experiments.exp_ccro_p4 --config config/ccro_p4.yaml
```

结果文件：

- `data/results/ccro_p4/table_p4.md`
- `data/results/ccro_p4/metrics.json`

结果摘要：

| 场景 | A4 Dmin | A5 Dmin | A6 dense Dmin | A6 控制 p95 |
| --- | ---: | ---: | ---: | ---: |
| A | 0.04984 | 0.06185 | 0.07532 | 6.73 ms |
| B | 1.08211 | 1.08214 | 1.08214 | 0.87 ms |
| C | 0.04253 | 0.08277 | 0.06389 | 6.52 ms |
| D | 0.00000 | 0.00000 | 0.00000 | 6.11 ms |

结论：

- 四个场景均 `accepted=true`；
- A/C 动态场景 dense 复核距离均高于 `d_stop=0.035 m`；
- A6 控制 p95 最差约 `6.73 ms`，满足 `20 ms` 控制段目标。

注意：

- A 场景 A6 出现较多 `state_mismatch_hold`，说明安全执行层较保守，能保护但不保证任务完成；
- A6 应表述为“速度级安全保护层”，不是替代 A5 轨迹重规划的主任务控制器。

### 4.2 4.5 虚拟闭环微基准

命令：

```bash
/home/hzy/miniconda3/envs/py310/bin/python -m experiments.exp_45_virtual_loop \
  --record-dir data/recordings/ch4_3_dynamic_01 \
  --empty-record-dir data/recordings/ch4_3_empty \
  --scenario B \
  --controller ours_full \
  --trial-id 1 \
  --output data/results/ch4_5_virtual_fast_10k \
  --max-frames 64

/home/hzy/miniconda3/envs/py310/bin/python -m experiments.exp_46_timing \
  --logs data/results/ch4_5_virtual_fast_10k \
  --output data/results/ch4_5_virtual_fast_10k/timing.json
```

结果文件：

- `data/results/ch4_5_virtual_fast_10k/trial_B_ours_full_01.json`
- `data/results/ch4_5_virtual_fast_10k/timing.json`

结果摘要：

| 版本 | mesh samples | 控制 mean | 控制 p95 | R_avoid | T_viol |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 ch4_6 汇总 | 原始日志 | 40.8883 ms | 199.8850 ms | - | - |
| 优化后，50k mesh | 50000 | 5.3909 ms | 25.7012 ms | 1.0000 | 0.0000 |
| 优化后，10k mesh | 10000 | 3.3717 ms | 14.8642 ms | 1.0000 | 0.0000 |

结论：

- 仅代码缓存与梯度复用后，50k mesh 下 p95 已从约 `199.9 ms` 降至约 `25.7 ms`；
- 使用在线控制默认的 10k mesh 后，p95 进一步降至约 `14.9 ms`；
- 微基准中 `Ours-Full` 仍保持 `R_avoid=1.0` 和 `T_viol=0`。

## 5. 当前可写入论文的表述

建议表述为：

> 早期虚拟闭环采用全身点云距离和有限差分梯度，控制段 p95 耗时约 199.9 ms，不满足实时控制目标。针对该问题，本文将在线执行层改为最近点局部梯度近似、同构型表面/KD-tree 缓存以及对象级球风险近似；dense mesh 仍保留为离线复核。优化后，在 10k 在线表面采样下，4.5 虚拟闭环微基准控制段 p95 降至约 14.9 ms，P4/A6 速度级安全层 p95 最差约 6.7 ms，同时 dense 复核保持安全距离约束。

同时需要保留限制：

> 该结果仍不是完整 RGB-D 端到端实时性结论，因为当前 timing 仍未计入相机读取、自滤除、聚类、跟踪和预测等模块。后续真机实验需补充分模块端到端计时。

## 6. 后续建议

1. 对 4.5 全部 B/C 试次重新运行优化版，生成新的 `metrics_fast.json` 与 timing 表；
2. 给 timing 增加完整感知字段：
   - `T_read`
   - `T_preprocess`
   - `T_self_filter`
   - `T_cluster`
   - `T_track`
   - `T_predict`
   - `T_control`
   - `T_total`
3. 将 `fast_distance_gradient()` 后续升级为真正解析雅可比：
   - 最近机器人点 `x_r(q)`；
   - 最近障碍点/对象边界点 `x_o`；
   - `∂d/∂q = n^T J_r(q)`。
4. 保持分层策略：
   - 在线控制：对象级球/胶囊体 + 缓存；
   - 候选筛选：coarse/medium mesh；
   - 最终验收：dense mesh。

