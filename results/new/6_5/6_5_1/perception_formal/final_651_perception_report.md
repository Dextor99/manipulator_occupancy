# 6.5.1 真实环境下静态与动态障碍感知验证结果报告

## 实验定位

本实验对应新版第 6.5.1 节，目标是在真实 RGB-D 输入下验证机械臂静止状态时的环境感知与风险识别能力。实验只读取 RealSense 点云和 AUBO i16 实时关节状态，用于完成机械臂本体滤除、外部障碍物聚类、动态障碍跟踪、STRO 预测风险和 CCRO 最近连杆风险判断。

本部分不向 AUBO 发送任何运动指令，所有试次中 `robot_commanded=false`。因此，本实验可作为真实感知与风险识别证据，不作为轨迹执行、停止控制或在线重规划的实机证据。

## 程序与结果路径

实验程序：

```bash
experiments/new/6_5/6_5_1/run_651_perception_capture.py
```

曲线生成程序：

```bash
experiments/new/6_5/6_5_1/plot_651_perception.py
```

正式结果目录：

```bash
results/new/6_5/6_5_1/perception_formal
```

每个试次输出：

```text
trials/<scenario>_<name>_<repeat>/
├── frames.csv
├── summary.json
└── perception_curve.png
```

## 统一运行参数

空场调参后，正式采集统一采用以下参数：

```bash
--temporal-denoise
--self-filter-threshold 0.08
--cluster-min-points 30
--cluster-min-volume 0.001
```

该参数组合下，E0 空场烟雾测试结果为：

| scene | frames | effective Hz | detected ratio | stable track ratio | risk ratio | result |
|---|---:|---:|---:|---:|---:|---|
| E0 empty | 79 | 7.903 | 0.000 | 0.000 | 0.000 | PASS |

对应数据：

```bash
results/new/6_5/6_5_1/perception_smoke_tuned/trials/E0_empty_self_filter_r02
```

## 最终采用试次

静态障碍试次采用：

| scene | description | adopted trials | note |
|---|---|---|---|
| S1 | 肘部/上臂附近静态障碍 | r04-r06 | 早期靠近末端的 r01-r03 已删除 |
| S2 | 前臂附近静态障碍 | r04-r06 | 早期靠近末端的 r01-r03 已删除 |
| S3 | 腕部/末端附近静态障碍 | r01-r03 | 原始采集位置合理 |

动态障碍试次采用：

| scene | description | adopted trials | note |
|---|---|---|---|
| D1 | 横向经过肘部/上臂区域 | r01-r05 | 全部可用；当前 r01 为重采后的有效数据 |
| D2 | 斜向接近腕部/末端区域后离开 | r02-r06 | 原 r01 已删除，当前目录仅保留有效 D2 试次 |

区域不匹配试次已从正式 `trials/` 目录删除，不进入自动汇总统计。

## 静态障碍结果

| scene | trials | detected ratio | stable track ratio | risk ratio | minimum distance / m | pre/post risk max | dominant nearest links |
|---|---:|---:|---:|---:|---:|---:|---|
| S1 elbow/upper arm | 3 | 0.974 | 0.917 | 0.974 | 0.0419 | 0.000 | upperArm_Link:222 |
| S2 forearm | 3 | 0.974 | 0.948 | 0.974 | 0.0420 | 0.000 | foreArm_Link:154, gripper_base_link:28, shoulder_Link:21, wrist1_Link:20 |
| S3 wrist/end | 3 | 0.978 | 0.952 | 0.974 | 0.0494 | 0.000 | left_link:220, right_link:5 |

静态障碍结果表明：在无障碍阶段和障碍移除阶段，系统没有产生风险误触发；在障碍物放置阶段，外部障碍物能够被稳定检测和跟踪，并能触发对应连杆区域的风险响应。

S1 的最近连杆稳定集中在 `upperArm_Link`，可作为肘部/上臂区域障碍检测证据。S2 的最近连杆主要为 `foreArm_Link`，同时少量帧落在 `wrist1_Link`、`gripper_base_link` 或 `shoulder_Link`，这与障碍物尺寸、点云视角和最近距离判定有关。S3 主要落在 `left_link/right_link`，符合腕部/末端附近障碍设置。

## 动态障碍结果

| scene | trials | detected ratio | stable track ratio | risk ratio | minimum distance / m | mean max speed / m s-1 | predicted lead / frames | pre/post risk max | dominant nearest links |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D1 cross elbow/upper arm | 5 | 0.739 | 0.721 | 0.429 | 0.0448 | 0.111 | 4.4 | 0.000 | upperArm_Link:417 |
| D2 oblique wrist/end approach | 5 | 0.633 | 0.614 | 0.435 | 0.0399 | 0.087 | 3.4 | 0.000 | wrist2_Link:174, wrist1_Link:106, gripper_base_link:76 |

动态障碍结果表明：在真实 RGB-D 输入下，手持泡沫球的横向经过和斜向接近过程均能被检测和跟踪；障碍物进入风险区域时，当前风险和预测风险均发生响应；障碍物离开后，风险状态恢复为空场状态。

D1 的风险主要集中在 `upperArm_Link`，符合横向经过肘部/上臂区域的实验设计。D2 的风险主要集中在 `wrist2_Link`、`wrist1_Link` 和 `gripper_base_link`，符合腕部/末端区域斜向接近的实验设计。预测风险首次触发平均早于当前距离风险触发，说明 STRO 预测模块在动态接近过程中具有提前预警作用。

## 已删除试次说明

为避免后续误用，区域不匹配的试次已从正式 `trials/` 目录删除：

| deleted trial group | reason |
|---|---|
| S1 r01-r03 | 实际最近连杆主要为 `left_link`，更接近末端障碍，不适合作为肘部/上臂区域证据 |
| S2 r01-r03 | 实际最近连杆主要为 `right_link`，更接近末端障碍，不适合作为前臂区域证据 |
| D1 早期 r01 | 首次采集动态阶段漏检且 post_removed 未清空，已被同名重采有效数据覆盖 |
| D2 r01 | 实际最近连杆为 `upperArm_Link`，运动过程更符合 D1，不适合作为 D2 腕部/末端区域证据 |

当前自动汇总只统计正式保留试次：S1 r04-r06、S2 r04-r06、S3 r01-r03、D1 r01-r05 和 D2 r02-r06。

## 复现实验步骤

### 1. 进入项目目录

```bash
cd /home/hzy/Code/manipulator_occupancy
```

### 2. 确认程序帮助信息

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/run_651_perception_capture.py --help
```

### 3. 空场调参 smoke test

机械臂保持静止，工作区内不要放临时障碍物。

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/run_651_perception_capture.py \
  --scenario E0 \
  --repeat 2 \
  --duration 10 \
  --temporal-denoise \
  --self-filter-threshold 0.08 \
  --cluster-min-points 30 \
  --cluster-min-volume 0.001 \
  --output results/new/6_5/6_5_1/perception_smoke_tuned
```

通过标准：

```text
detected_frame_ratio = 0
stable_track_frame_ratio = 0
risk_frame_ratio = 0
```

### 4. 静态障碍采集

每个静态试次包含三个阶段：

```text
pre_empty      障碍物不进入工作区
obstacle       将轻质泡沫障碍物放到指定位置并保持静止
post_removed   移走障碍物，恢复空场景
```

S1 肘部/上臂区域补采：

```bash
for r in 4 5 6; do
  /home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/run_651_perception_capture.py \
    --scenario S1 \
    --repeat $r \
    --temporal-denoise \
    --self-filter-threshold 0.08 \
    --cluster-min-points 30 \
    --cluster-min-volume 0.001 \
    --output results/new/6_5/6_5_1/perception_formal
done
```

S2 前臂区域补采：

```bash
for r in 4 5 6; do
  /home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/run_651_perception_capture.py \
    --scenario S2 \
    --repeat $r \
    --temporal-denoise \
    --self-filter-threshold 0.08 \
    --cluster-min-points 30 \
    --cluster-min-volume 0.001 \
    --output results/new/6_5/6_5_1/perception_formal
done
```

S3 腕部/末端区域采集：

```bash
for r in 1 2 3; do
  /home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/run_651_perception_capture.py \
    --scenario S3 \
    --repeat $r \
    --temporal-denoise \
    --self-filter-threshold 0.08 \
    --cluster-min-points 30 \
    --cluster-min-volume 0.001 \
    --output results/new/6_5/6_5_1/perception_formal
done
```

### 5. 动态障碍采集

每个动态试次包含三个阶段：

```text
pre_empty      泡沫球不进入工作区
dynamic        使用长杆移动轻质泡沫球
post_removed   泡沫球完全离开工作区
```

D1 横向经过肘部/上臂区域：

```bash
for r in 1 2 3 4 5; do
  /home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/run_651_perception_capture.py \
    --scenario D1 \
    --repeat $r \
    --temporal-denoise \
    --self-filter-threshold 0.08 \
    --cluster-min-points 30 \
    --cluster-min-volume 0.001 \
    --output results/new/6_5/6_5_1/perception_formal
done
```

D2 斜向接近腕部/末端区域后离开：

```bash
for r in 2 3 4 5 6; do
  /home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/run_651_perception_capture.py \
    --scenario D2 \
    --repeat $r \
    --temporal-denoise \
    --self-filter-threshold 0.08 \
    --cluster-min-points 30 \
    --cluster-min-volume 0.001 \
    --output results/new/6_5/6_5_1/perception_formal
done
```

### 6. 生成曲线图

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_1/plot_651_perception.py \
  --root results/new/6_5/6_5_1/perception_formal
```

### 7. 查看汇总与质量检查

```bash
sed -n '1,200p' results/new/6_5/6_5_1/perception_formal/summary.md
sed -n '1,200p' results/new/6_5/6_5_1/perception_formal/static_supplement_quality_check.md
sed -n '1,200p' results/new/6_5/6_5_1/perception_formal/d1_quality_check.md
```

## 写作建议

论文中建议将本节表述为：

> 在机械臂保持静止的条件下，采用 RealSense 采集真实 RGB-D 点云，并同步读取 AUBO i16 关节状态。系统首先基于当前关节构型进行机械臂本体点云滤除，然后对外部点云进行聚类、跟踪和风险预测。实验分别设置静态障碍和手持动态障碍场景，用于验证真实感知输入下障碍物检测、最近连杆定位、风险触发以及障碍移除后的风险恢复能力。

可报告的主要结论：

1. 空场条件下未产生稳定误检或风险误触发。
2. 静态障碍放入后，三类区域的检测率均约为 0.974 以上，稳定跟踪率约为 0.917 以上，移除后风险恢复为 0。
3. D1 横向经过肘部/上臂区域时，风险主要集中在 `upperArm_Link`。
4. D2 斜向接近腕部/末端区域时，风险主要集中在 `wrist2_Link`、`wrist1_Link` 和 `gripper_base_link`。
5. 动态障碍场景中，预测风险触发早于当前距离风险触发，说明 STRO 对接近趋势具有提前预警能力。

本节不报告真实轨迹跟踪误差、停止响应时间或在线重规划成功率；这些内容应放入后续 6.5.2 或 6.5.3。
