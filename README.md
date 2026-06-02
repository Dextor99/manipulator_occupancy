# Manipulator Occupancy — 机械臂统一时空占据感知系统

机械臂工作空间中的 **实时占据感知** 与 **安全检测** 系统。通过 RGB-D 相机（RealSense）感知环境点云，对机械臂自身进行滤除（Self-Filter），对剩余物体进行聚类、几何拟合、跟踪与运动预测，最终输出安全决策（减速/停止）。

Non-ROS Python prototype for unified spatio-temporal occupancy perception and lightweight risk modeling around a manipulator.

---

## 📦 项目结构

```
manipulator_occupancy/
├── main.py                         # 主程序入口，组装完整 pipeline
│
├── calibration/                    # 相机标定与坐标变换工具
│   └── transform_utils.py          # 加载 4×4 变换矩阵、点云坐标变换
│
├── camera/                         # 相机驱动与点云预处理
│   ├── Realsense/                  # RealSense 相机相关工具
│   │   ├── Realsense.py            # RealSense 基础读取器
│   │   ├── realsense_save_rgbd.py  # 保存 RGB-D 图像到本地
│   │   ├── capture_images.py       # 采集 RGB-D 图像
│   │   ├── get_RGB.py              # 获取 RGB 图像
│   │   ├── get_RGB_without_undistorted.py  # 获取未去畸变的 RGB 图
│   │   ├── get_PIC_RGB&D.py        # 同时获取 RGB 和深度图
│   │   ├── get_camera_matrix.py    # 计算相机内参矩阵
│   │   ├── depth2clouds.py         # 深度图转点云
│   │   ├── depth2clouds_ROI.py     # 感兴趣区域的深度转点云
│   │   ├── depth2clouds_mask.py    # 带掩码的深度转点云
│   │   ├── tiff2ply.py             # TIFF 深度图转 PLY 点云
│   │   ├── visual_tiff_in.py       # 可视化 TIFF 深度图
│   │   ├── chess_board.py          # 棋盘格角点检测
│   │   ├── chess_board_mouse.py    # 鼠标交互式棋盘格角点提取
│   │   ├── live_view.py            # 实时相机预览
│   │   ├── inout.py                # 文件读写工具
│   │   ├── camera.json / camera0.json # 相机参数配置
│   │   ├── eye_in_hand_hzy.py / eye_in_hand_lza.py  # 眼在手上标定
│   │   ├── eye_to_hand_hzy1.py / eye_to_hand_lza.py  # 眼在手外标定
│   │   └── record/                 # 手眼标定记录数据
│   │
│   ├── realsense_pipeline_reader.py # Pipeline 版读取器，输出 Frame（含点云）
│   ├── realsense_reader.py         # 简版 RealSense 读取器
│   ├── mock_reader.py              # Mock 数据读取器（离线测试用）
│   ├── depth_to_pointcloud.py      # 深度图 → 3D 点云（利用相机内参）
│   └── pointcloud_preprocess.py    # 预处理：去无效点、裁切、体素降采样
│
├── config/                         # 配置文件目录
│   ├── camera_extrinsic.json       # 相机外参（base_T_cam 4×4 矩阵）
│   ├── camera_intrinsic.json       # 相机内参（fx, fy, cx, cy）
│   ├── workspace.yaml              # 工作空间边界（x/y/z 范围 + 球约束）
│   ├── robot_capsules.yaml         # 胶囊体碰撞模型定义
│   ├── robot_model.yaml            # 机械臂模型参数
│   ├── safety.yaml                 # 安全策略参数（阈值、聚类等）
│   └── apply_calibration.py        # 应用标定结果到配置文件
│
├── control/                        # 机械臂控制模块
│   ├── robot_command.py            # 机器人指令接口（Mock + 安全响应）
│   ├── speed_scaler.py             # 速度缩放
│   └── trajectory_generator.py     # Minimum Jerk 轨迹生成
│
├── experiments/                    # 实验脚本
│   ├── exp_self_filter.py          # 测试自滤除功能
│   ├── exp_prediction.py           # 测试完整 pipeline（20帧）
│   ├── exp_safety_motion.py        # 测试安全决策（50帧）
│   └── exp_geometry_compare.py     # 比较不同拟合方式（球体/AABB/OBB）
│
├── perception/                     # 核心感知模块
│   ├── self_filter.py              # 自滤除：用胶囊体模型移除机器人点云
│   ├── clustering.py               # 聚类（Open3D DBSCAN → 纯 NumPy 降级）
│   ├── geometry_fit.py             # 几何拟合：球体 / AABB / OBB
│   ├── occupancy_object.py         # 数据类（ShapeModel + OccupancyObject）
│   └── occupancy_tracker.py        # 跨帧跟踪关联、速度估计、置信度
│
├── risk/                           # 风险评估与安全策略
│   ├── distance_check.py           # 胶囊体 ↔ 风险球最近距离
│   ├── prediction.py               # 运动预测（匀速模型多步外推）
│   └── safety_policy.py            # 安全决策：SAFE/WARNING/SLOW/STOP
│
├── robot/                          # 机械臂建模与状态读取
│   ├── capsule_model.py            # 胶囊体模型（端点+半径）及 Mock 数据
│   ├── fk_model.py                 # 正运动学（基于 urdfpy 或 Mock）
│   ├── urdf_model.py               # 轻量级 URDF 解析 + FK（纯 NumPy）
│   ├── robot_state_reader.py       # 状态读取器（Mock/真实 AUBO SDK）
│   ├── robot_visualizer.py         # Open3D 可视化工具
│   └── 01_calibrate_robot/         # 📦 子模块：机械臂标定项目（C++/Python）
│
├── tests/                          # 单元测试（pytest）
│   ├── test_capsule_distance.py    # 胶囊体距离计算测试
│   ├── test_depth_to_pointcloud.py # 深度图转点云测试
│   ├── test_geometry_fit.py        # 几何拟合测试
│   ├── test_pipeline_mock.py       # Mock pipeline 测试
│   ├── test_prediction_policy.py   # 预测+安全策略测试
│   ├── test_self_filter.py         # 自滤除测试
│   ├── test_tracker.py             # 跟踪器测试
│   └── test_transform_utils.py     # 坐标变换测试
│
├── urdf/                           # URDF 机械臂模型
│   ├── aubo_i16.urdf / aubo_i16_gripper.urdf / aubo_i16_old.urdf  # 遨博 i16
│   ├── doosan.urdf / doosan_copy.urdf # 斗山机械臂
│   ├── m0609_white.urdf            # M0609 机械臂
│   ├── ur5.urdf                    # UR5 机械臂
│   ├── camera_board.urdf / camera_board2.urdf # 相机安装板
│   ├── gripper.urdf                # 夹爪
│   ├── EG2-4C-URDF-11_20/         # EG2-4C 夹爪（含 ROS 配置）
│   └── meshes/                     # 各型号碰撞/视觉网格模型
│
├── utils/                          # 工具模块
│   └── config.py                   # JSON/YAML 配置加载器
│
├── visualization/                  # 可视化
│   ├── open3d_viewer.py            # 实时可视化（点云+胶囊+占据物+风险球）
│   └── plot_logger.py              # CSV 日志记录
│
├── test_*.py                       # 根目录下的测试/验证脚本
│   ├── test_clustering_filtering.py  # 聚类+自滤除联合测试
│   ├── test_remove_robot_points.py / test_remove_robot_points_fast.py # 机器人点云移除
│   ├── test_transform_crop.py        # 坐标变换+裁切
│   ├── test_urdf_only.py / test_urdf_visualization.py / test_urdf_visualization_urdf_only.py # URDF
│   ├── test.py                      # 简易测试
│   └── verify_robot_fk.py           # 正运动学验证
│
├── data/logs/                      # Pipeline 运行日志（CSV）
├── requirements.txt                # Python 依赖
└── 机械臂统一时空占据感知实施计划.md  # 项目实施文档
```

---

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

| 依赖 | 用途 |
|------|------|
| `numpy`, `scipy` | 数值计算 |
| `open3d` | 3D 点云处理与可视化 |
| `pyrealsense2` | RealSense 相机驱动（可选） |
| `pyyaml` | 配置解析 |
| `urdfpy`, `trimesh` | URDF 加载与网格处理 |
| `matplotlib` | 绘图 |
| `pytest` | 测试 |

### 运行

**Mock 模式（无需硬件）：**

```bash
# 主 pipeline，模拟数据 100 帧 + 可视化
python main.py --source mock --max-frames 100 --visualize

# 实验脚本
python experiments/exp_self_filter.py
python experiments/exp_prediction.py
python experiments/exp_geometry_compare.py
python experiments/exp_safety_motion.py
```

**RealSense 模式（需连接相机）：**

```bash
python main.py --source realsense --visualize
```

**运行测试：**

```bash
pytest tests/ -v
```

---

## 🔧 Pipeline 流程

```
RGB-D 相机 (RealSense/Mock)
        │
        ▼  depth_to_pointcloud
  深度图 → 相机坐标系点云
        │
        ▼  transform_points
  坐标变换 → 基坐标系点云
        │
        ▼  crop_workspace + voxel_downsample
  预处理 → 裁切工作空间 + 体素降采样
        │
        ▼  filter_robot_self_points
  自滤除 → 用胶囊体移除机器人点云
        │
        ▼  cluster_points (DBSCAN)
  聚类 → 分割出独立物体
        │
        ▼  fit_sphere / fit_aabb / fit_obb
  几何拟合 → 球体/AABB/OBB 描述
        │
        ▼  OccupancyTracker.update
  跨帧跟踪 → 关联 + 速度估计
        │
        ▼  predict_risk_spheres
  运动预测 → 多时间步外推风险球
        │
        ▼  min_capsule_sphere_distance
  距离检测 → 胶囊体 vs 风险球
        │
        ▼  SafetyPolicy.evaluate
  安全决策 → SAFE / WARNING / SLOW / STOP
        │
        ▼
  输出: 速度缩放系数 / 紧急停止信号
```

---

## 🧠 核心算法

| 模块 | 方法 |
|------|------|
| **自滤除** | 胶囊体有符号距离，点到线段投影 |
| **聚类** | DBSCAN（Open3D → 纯 NumPy 降级） |
| **几何拟合** | 球体（均值+最大距）、AABB、OBB（SVD） |
| **跨帧跟踪** | 最近邻关联 + 指数移动平均速度 |
| **运动预测** | 匀速模型 + 不确定性膨胀 |
| **安全策略** | 三级阈值，线性速度缩放 |
| **轨迹规划** | Minimum Jerk 五次多项式 |

---

## ⚙️ 硬件配置注意事项

- `config/camera_extrinsic.json` — 默认为单位矩阵，实际使用前需替换为标定后的 `base_T_cam`
- `config/robot_model.yaml` — 可选 URDF 路径，缺失时使用 Mock 模型
- `camera/realsense_reader.py` — 未安装 `pyrealsense2` 时会报清晰错误
- `control/robot_command.py` — 目前仅 Mock 输出，接入真实机械臂 SDK 需实现同一接口

---

## 📁 子模块

[**`robot/01_calibrate_robot/`**](https://github.com/307321587/01_calibrate_robot.git) — 机械臂标定项目，包含：
- C++ 底层控制与 SDK 封装（AUBO i16）
- Pybind11 Python 接口
- 手眼标定数据采集与解算
- 克隆后初始化：`git submodule update --init --recursive`

---

## 🧪 支持的机械臂

| 型号 | URDF |
|------|------|
| 遨博 AUBO i16 | `urdf/aubo_i16.urdf` |
| 遨博 + 夹爪 | `urdf/aubo_i16_gripper.urdf` |
| 斗山 Doosan | `urdf/doosan.urdf` |
| M0609 | `urdf/m0609_white.urdf` |
| UR5 | `urdf/ur5.urdf` |
| EG2-4C 夹爪 | `urdf/EG2-4C-URDF-11_20/` |

---

## 📝 License

本项目仅供学习和研究使用。
