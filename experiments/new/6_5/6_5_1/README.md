# 6.5.1 实机准入实验程序

本目录只服务第 6.5.1 节：真实平台与低速轨迹执行基线。不要用它直接进入 6.5.2 动态闭环切换。

## 运行环境

建议使用项目当前的 `py310` 环境：

```bash
cd /home/hzy/Code/manipulator_occupancy
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/run_651_baseline.py --help
```

## 1. 离线软件链路预验证

用于验证 NUBS、RobotSurfaceModel、MeshRiskEvaluator、静态优化器、dense verifier、统计和导出链路。
不会读取 RealSense，也不会向 AUBO 发送命令。

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/run_651_baseline.py \
  --mode offline \
  --output results/new/6_5/6_5_1/offline_reproducible
```

结果中的关键标志应为：

```json
{
  "execution_mode": "offline_reproducible",
  "sensor_live": false,
  "robot_state_live": false,
  "robot_commanded": false,
  "software_gate": "PASS",
  "real_experiment_gate": "NOT_RUN"
}
```

## 2. live-shadow：真实 B0 静止稳定性

连接 RealSense 和 AUBO 状态反馈，执行真实点云、自滤除和时间戳记录，但不发送任何运动指令。

先做 10 s 单构型试运行：

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/run_651_baseline.py \
  --mode live-shadow \
  --live-conditions start \
  --b0-duration-s 10 \
  --output results/new/6_5/6_5_1/real_platform/b0_smoke_start
```

确认 `b0_live_shadow.csv` 字段完整后，再做正式 3 x 60 s：

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/run_651_baseline.py \
  --mode live-shadow \
  --live-conditions start,mid,goal \
  --b0-duration-s 60 \
  --output results/new/6_5/6_5_1/real_platform/b0_formal
```

程序会在每个构型前暂停，等待人工确认机械臂已保持在对应构型。

如果要预览 `robot/safety_guided_motion.py --real-robot --range 0.20` 对应的 Cartesian Y 轴路径，先运行只读预览程序。该程序不会发送机械臂运动命令。

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/preview_safety_y_motion.py \
  --center-source live-current \
  --range 0.20 \
  --motion-x-offset 0.0 \
  --output results/new/6_5/6_5_1/real_platform/safety_y_preview_live_current
```

输出包含：

```text
preview_safety_y_motion.json
preview_safety_y_path.csv
preview_safety_y_motion.png
```

默认映射为 `start=Y_minus, mid=center, goal=Y_plus`。注意：该预览只显示 TCP 直线路径，不等价于完整机械臂全身碰撞证明，因为控制器内部 IK 过程没有在这里回放。

如果要检查 6.5.1 脚本曾经使用的 `start/mid/goal` 构型，以及从当前关节到这些构型的近似 `movej` 扫掠路径，使用：

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/preview_651_b0_positions.py \
  --current-source live-current \
  --output results/new/6_5/6_5_1/real_platform/b0_position_preview_live_current
```

该程序不会发送运动命令。它输出 `b0_position_path_preview.png`、`b0_position_path_samples.csv` 和 `b0_position_preview.json`，用于检查当前位姿到 `start -> mid -> goal` 的关节线性近似路径。

## 3. live-execute：安全预检

当前仓库尚未封装 AUBO 支持的 NUBS 关节轨迹队列/批量下发接口。为避免 Python 循环逐点发送轨迹，本模式只执行预检并拒绝真实运动。

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/run_651_baseline.py \
  --mode live-execute \
  --output results/new/6_5/6_5_1/real_platform/live_execute_preflight
```

预期结果：

```json
{
  "robot_commanded": false,
  "real_experiment_gate": "BLOCKED_NO_SUPPORTED_AUBO_TRAJECTORY_API"
}
```

只有补齐受控轨迹下发接口后，才能继续真实 B1/B2 执行。

## 4. 新版 6.5.1：静止机械臂感知补采集

新版 6.5.1 只验证真实 RGB-D 感知、机械臂本体滤除、障碍物检测/跟踪、STRO 预测和 CCRO 最近连杆风险判断。采集程序不会向机器人发送运动命令。

程序：

```bash
experiments/new/6_5/6_5_1/run_651_perception_capture.py
```

### 4.1 先做 5 秒 smoke test

确认 RealSense、AUBO 状态、自滤除、聚类和日志字段都正常：

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/run_651_perception_capture.py \
  --scenario E0 \
  --repeat 0 \
  --duration 5 \
  --output results/new/6_5/6_5_1/perception_smoke
```

输出：

```text
results/new/6_5/6_5_1/perception_smoke/
├── metrics.json
├── summary.md
└── trials/E0_empty_self_filter_r00/
    ├── frames.csv
    └── summary.json
```

检查 `frames.csv` 中至少应有：

```text
cluster_count, stable_track_count, nearest_distance_m,
nearest_link, predicted_distance_m, predicted_nearest_link,
risk_state_current, risk_state_predicted, q1_rad...q6_rad
```

### 4.2 正式补采：E0 空场景

机械臂保持静止，工作空间内不要放临时障碍物。

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/run_651_perception_capture.py \
  --scenario E0 \
  --repeat 1 \
  --output results/new/6_5/6_5_1/perception_formal
```

### 4.3 正式补采：S1/S2/S3 静态障碍

每个场景重复 3 次。程序会按阶段提示：

```text
pre_empty      3 s：障碍物未进入
obstacle      10 s：放置泡沫障碍物并保持静止
post_removed   3 s：移走障碍物
```

建议摆放：

```text
S1：肘部 / 中间连杆附近
S2：前臂中部附近
S3：腕部 / 夹爪附近
```

运行命令示例：

```bash
for r in 1 2 3; do
  /home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/run_651_perception_capture.py \
    --scenario S1 --repeat $r \
    --output results/new/6_5/6_5_1/perception_formal
done

for r in 1 2 3; do
  /home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/run_651_perception_capture.py \
    --scenario S2 --repeat $r \
    --output results/new/6_5/6_5_1/perception_formal
done

for r in 1 2 3; do
  /home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/run_651_perception_capture.py \
    --scenario S3 --repeat $r \
    --output results/new/6_5/6_5_1/perception_formal
done
```

### 4.4 正式补采：D1/D2 动态障碍

每个场景重复 5 次。程序会按阶段提示：

```text
pre_empty      3 s：泡沫球未进入
dynamic       15 s：实验人员用长杆移动泡沫球
post_removed   3 s：泡沫球离开并恢复空场景
```

建议路径：

```text
D1：横向经过前臂或肘部区域
D2：斜向接近腕部或小臂后离开
```

运行命令示例：

```bash
for r in 1 2 3 4 5; do
  /home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/run_651_perception_capture.py \
    --scenario D1 --repeat $r \
    --temporal-denoise \
    --output results/new/6_5/6_5_1/perception_formal
done

for r in 1 2 3 4 5; do
  /home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/run_651_perception_capture.py \
    --scenario D2 --repeat $r \
    --temporal-denoise \
    --output results/new/6_5/6_5_1/perception_formal
done
```

### 4.5 采集完成后查看汇总

```bash
sed -n '1,120p' results/new/6_5/6_5_1/perception_formal/summary.md
```

每个 trial 的逐帧日志位于：

```text
results/new/6_5/6_5_1/perception_formal/trials/<scenario>_<name>_rXX/frames.csv
```

生成每个 trial 的距离/速度/风险曲线：

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/plot_651_perception.py \
  --root results/new/6_5/6_5_1/perception_formal
```

每个 trial 目录下会生成：

```text
perception_curve.png
```

## 5. 论文插图关键帧补采

正式 6.5.1 结果目录只保存了 `frames.csv`、`summary.json` 和 `perception_curve.png`，不能从中恢复真实 RGB、深度图或点云画面。若论文需要静态检测、D1 动态横穿、D2 动态接近的真实检测帧，可使用独立补采程序：

```bash
experiments/new/6_5/6_5_1/capture_651_visual_snapshots.py
```

该程序不会向机器人发送运动指令，只读取 RealSense 和 AUBO 状态。建议只补采 3 个代表性场景：

```bash
# S2：前臂附近静态障碍关键帧
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/capture_651_visual_snapshots.py \
  --scenario S2 \
  --repeat 1 \
  --output results/new/6_5/6_5_1/perception_visual_snapshots

# D1：横向经过肘部/上臂区域关键帧
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/capture_651_visual_snapshots.py \
  --scenario D1 \
  --repeat 1 \
  --output results/new/6_5/6_5_1/perception_visual_snapshots

# D2：斜向接近腕部/末端区域关键帧
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/capture_651_visual_snapshots.py \
  --scenario D2 \
  --repeat 1 \
  --output results/new/6_5/6_5_1/perception_visual_snapshots
```

输出结构：

```text
results/new/6_5/6_5_1/perception_visual_snapshots/
└── trials/<scenario>_<name>_rXX/
    ├── snapshot_index.json
    └── snapshots/<scenario>_rXX_<event>/
        ├── rgb.png
        ├── depth_colormap.png
        ├── rgb_overlay.png
        ├── scene_points.npz
        ├── robot_points.npz
        ├── clusters.npz
        └── snapshot_meta.json
```

其中 `rgb_overlay.png` 可直接作为论文候选插图；`scene_points.npz`、`robot_points.npz` 和 `clusters.npz` 可用于后续重绘点云视图。动态场景会尽量保存 `predicted_risk_onset`、`current_risk_onset`、`minimum_clearance` 和 `risk_cleared` 等关键事件帧。
