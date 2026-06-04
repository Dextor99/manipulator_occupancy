#!/usr/bin/env python3
"""
安全引导运动 ── 基于 live_tracking.py 的障碍感知，
模拟机械臂 Y 轴 ±30cm 往返运动，根据障碍距离平滑调速。

核心复用
--------
每帧循环 100% 复用 live_tracking 的链路：
  点云采集 → 聚类 → 跟踪+速度 → 预测

拿到 tracked_objects + risk_spheres 后新增：
  ① YAxisMotionPlanner      → 模拟 Y 轴往返运动，输出 y_pos
  ② MovingCapsuleProvider   → 根据 y_pos 偏移胶囊体
  ③ Distance Check          → 用偏移胶囊体对 risk_spheres 做距离检查
  ④ AdaptiveSafetyController → 输出 speed_scale → 写回步进器
  ⑤ 可视化：移动胶囊体 + 距离连线 + HUD

策略
----
  静态障碍 (v < 0.01)  → 按实际距离线性减速
  动态接近 (closing>0) → 预判提前减速 (effective_dist -= closing × 0.3s)
  动态远离 (closing=0) → 正常通过
  速度渐变：max_decel = 2.0/s, max_accel = 0.5/s （缓启动不突兀）

用法
----
  # 模拟模式（结构化点云 + 模拟机械臂 Y 轴运动）
  python robot/safety_guided_motion.py --mock --visualize

  # 真实机械臂 + RealSense
  python robot/safety_guided_motion.py --real-robot --visualize --remove-planes
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中（无论从何处执行）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import open3d as o3d

# ── 复用 live_tracking 的链路模块 ──
from test_remove_robot_points_fast import SceneProcessor
from test_clustering_filtering import (
    FastClusteringFilter,
    SphereSmoother,
    TemporalDenoiser,
    _random_colors,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    CLUSTER_MIN_POINTS,
    CLUSTER_MIN_VOLUME,
    EDGE_MARGIN,
    FRAME_INTERVAL_MS,
)
from perception.geometry_fit import (
    fit_obb,
    create_obb_wireframe,
    create_sphere_wireframe,
    create_text_label,
    make_occupancy_object,
)
from perception.occupancy_tracker import OccupancyTracker
from perception.occupancy_object import OccupancyObject
from risk.prediction import predict_risk_spheres, RiskSphere
# 距离判定改用机械臂点云，不再使用胶囊体
from robot.robot_commander import RobotCommander

# ═══════════════════════════════════════════════════════════════
# 结构化模拟数据（同 live_tracking._StructuredMockReader）
# ═══════════════════════════════════════════════════════════════


class _StructuredMockReader:
    """在基坐标系下生成：桌面(z=0) + 移动球体 + 静态盒子。"""

    def __init__(self):
        self.index = 0
        self.rng = np.random.default_rng(2)

    def read(self):
        timestamp = self.index * 0.1
        self.index += 1
        moving_center = np.array([-0.3 + 0.035 * self.index, 0.25, 0.30])
        static_center = np.array([0.55, -0.25, 0.20])
        points = np.vstack([
            self._sphere_points(moving_center, 0.08, 300),
            self._box_points(static_center, np.array([0.16, 0.12, 0.10]), 250),
            self._table_points(400),
        ])
        return type("MockFrame", (), {"points_cam": points, "timestamp": timestamp})()

    @staticmethod
    def _sphere_points(center, radius, count):
        rng = np.random.default_rng()
        d = rng.normal(size=(count, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return center + d * (radius * np.cbrt(rng.random((count, 1))))

    @staticmethod
    def _box_points(center, size, count):
        rng = np.random.default_rng()
        return center + (rng.random((count, 3)) - 0.5) * size

    @staticmethod
    def _table_points(count):
        rng = np.random.default_rng()
        xy = rng.uniform(-0.7, 0.7, size=(count, 2))
        return np.hstack([xy, np.zeros((count, 1))])

    def stop(self):
        pass


# ═══════════════════════════════════════════════════════════════
# 1. Y 轴运动规划器
# ═══════════════════════════════════════════════════════════════

class YAxisMotionPlanner:
    """在 Y 轴 ±range 之间正弦波形平滑往返，速度受 speed_scale 控制。

    使用累积相位法（而非 t_elapsed），speed_scale 变化时位置连续不跳变。
    """

    def __init__(self, range_m: float = 0.40, base_speed: float = 0.1):
        self.range = range_m
        self.base_speed = base_speed
        self.y_pos = 0.0
        self._phase = 0.0  # 累积相位 rad
        self._omega = 0.8  # rad/s

    def step(self, dt: float, speed_scale: float) -> float:
        """累积相位积分：phase += ω * speed * dt → 位置平滑过渡。"""
        self._phase += self._omega * max(speed_scale, 0.0) * dt
        self.y_pos = self.range * np.sin(self._phase)
        return self.y_pos


# ═══════════════════════════════════════════════════════════════
# 2. 移动胶囊体提供器
# ═══════════════════════════════════════════════════════════════

# 不再使用 MovingCapsuleProvider
# 距离判定直接用机械臂点云（来自相机或模拟生成）


# ═══════════════════════════════════════════════════════════════
# 3. 自适应安全控制器
# ═══════════════════════════════════════════════════════════════

class AdaptiveSafetyController:
    """基于障碍距离 + 接近速度的智能调速。

    根据距离变化率（closing velocity）区分三类场景：

    ---------+--------------------+----------------------------------
     场景     | 判定条件           | 行为
    ---------+--------------------+----------------------------------
     静态     | obs_speed<0.01    | 梯度减速，到 d_stop 才停
     迎面接近 | closing > 0.03    | 预判提前减速，相对速度越大越激进
     同向/远离 | closing ≤ 0.03   | 保持安全即可，不主动减速
    ---------+--------------------+----------------------------------

    设计要点
    --------
    - d_stop=0.08m > ROBOT_REMOVAL_THRESHOLD=0.05m：
      停止距离大于自身过滤阈值，避免"最近障碍点被删除→距离骤升→误判安全"。
    - surface_margin=0.02m：
      补偿点云稀疏/体素降采样/遮挡导致的点到点距离乐观偏差。
    - HOLD_RECOVERY 仅当距离恢复到 d_safe 以上才触发：
      自然距离控制已覆盖大部分恢复场景，HOLD_RECOVERY 仅作为兜底。

    速度渐变：减速 2.0/s, 加速 0.5/s – 缓启动不突兀。
    """

    def __init__(
        self,
        d_safe: float = 0.22,
        d_slow: float = 0.14,
        d_stop: float = 0.08,
        max_decel: float = 2.0,
        max_accel: float = 0.5,
        dynamic_lookahead: float = 0.15,
        close_threshold: float = 0.05,  # m/s — 超过此值视为"迎面接近"
        dist_smooth_alpha: float = 0.3,  # 距离 EMA 平滑系数
        surface_margin: float = 0.02,  # 点云稀疏补偿（体素降采样/遮挡导致的距离乐观）
        hold_timeout: float = 2.0,  # 停止超过此秒数且障碍不靠近 → 尝试恢复
        hold_recovery_speed: float = 0.15,  # 恢复时的初始 speed_scale
    ):
        self.d_safe = d_safe
        self.d_slow = d_slow
        self.d_stop = d_stop
        self.max_decel = max_decel
        self.max_accel = max_accel
        self.dynamic_lookahead = dynamic_lookahead
        self.close_threshold = close_threshold
        self.dist_smooth_alpha = dist_smooth_alpha
        self.surface_margin = surface_margin
        self.hold_timeout = hold_timeout
        self.hold_recovery_speed = hold_recovery_speed

        self.speed_scale = 1.0
        self.last_decision = "SAFE"
        self.last_nearest_id: int | None = None
        self.last_effective_dist = float("inf")

        # ── 帧间状态（按 object_id 追踪） ──
        self._prev_dist: dict[int, float] = {}
        self._smoothed_dist: dict[int, float] = {}

        # ── 停止超时恢复状态 ──
        self._hold_start_time: float | None = None

    def evaluate(
        self,
        distance: float,
        nearest_obj: OccupancyObject | None,
        capsules: list | None = None,
        dt: float = 0.05,
    ) -> float:
        """返回平滑后的 speed_scale (0.0~1.0)。

        Parameters
        ----------
        distance : float
            机械臂表面到障碍物表面的 KD-tree 距离 (m), inf = 无障。
        nearest_obj : OccupancyObject | None
            最近障碍的跟踪对象，含 velocity / id 等信息。
        dt : float
            帧间隔 (s)。
        """
        # ── 无障碍 → 全速 ──
        if np.isinf(distance) or nearest_obj is None:
            target = 1.0
            self.last_decision = "SAFE"
            self.last_nearest_id = None
            self.last_effective_dist = float("inf")
            self._prev_dist.clear()
            self._smoothed_dist.clear()
            self._hold_start_time = None
        else:
            oid = nearest_obj.id

            # a) 距离 EMA 平滑，抑制单帧抖动
            raw_dist = distance
            if oid in self._smoothed_dist:
                smoothed = self.dist_smooth_alpha * raw_dist + \
                           (1.0 - self.dist_smooth_alpha) * self._smoothed_dist[oid]
            else:
                smoothed = raw_dist
            self._smoothed_dist[oid] = smoothed

            # b) 接近速率 (closing > 0 = 间距在缩小)
            if oid in self._prev_dist:
                closing = (self._prev_dist[oid] - smoothed) / max(dt, 1e-6)
            else:
                closing = 0.0
            self._prev_dist[oid] = smoothed

            obstacle_speed = float(np.linalg.norm(nearest_obj.velocity))

            # c) 场景分类 + 有效距离
            if obstacle_speed < 0.01 and closing <= self.close_threshold:
                # ── 静态障碍：梯度减速 + surface_margin 补偿点云稀疏乐观偏差 ──
                effective_dist = max(0.0, smoothed - self.surface_margin)
                self.last_decision = "STATIC"
            elif closing > self.close_threshold:
                # ── 迎面接近：预判减速 ──
                effective_dist = max(0.0, smoothed - closing * self.dynamic_lookahead)
                self.last_decision = "APPROACHING"
            else:
                # ── 同向/远离：不主动减速，仅 d_stop 处保底 ──
                effective_dist = smoothed + 0.04
                self.last_decision = "FOLLOWING"

            # d) 三段映射 → 目标 speed_scale
            if effective_dist >= self.d_safe:
                target = 1.0
            elif effective_dist <= self.d_stop:
                target = 0.0
            else:
                target = (effective_dist - self.d_stop) / (self.d_safe - self.d_stop)

            # e) 停止超时恢复（安全的兜底机制）
            #    ⚠ 仅当障碍物已明显远离（smoothed > d_safe）才允许强制恢复。
            #    正常情况下 distance>d_safe → target=1.0 → target<0.01 不成立，
            #    所以此分支实际不会覆盖正常距离控制，仅作为极端情况下的安全兜底。
            #    避免"停止 2s → 主动恢复 → 顶着障碍走"的危险循环。
            if target < 0.01 and closing <= self.close_threshold:
                if self._hold_start_time is None:
                    self._hold_start_time = time.perf_counter()
                elif time.perf_counter() - self._hold_start_time >= self.hold_timeout:
                    if smoothed > self.d_safe:
                        target = self.hold_recovery_speed
                        self.last_decision = "HOLD_RECOVERY"
            else:
                self._hold_start_time = None

            self.last_nearest_id = oid
            self.last_effective_dist = effective_dist

        # ── 速度渐变 ──
        max_change = self.max_decel * dt if target < self.speed_scale else self.max_accel * dt
        delta = np.clip(target - self.speed_scale, -max_change, max_change)
        self.speed_scale = float(np.clip(self.speed_scale + delta, 0.0, 1.0))
        return self.speed_scale


# ═══════════════════════════════════════════════════════════════
# 4. 可视化辅助（轻量：只有 PointCloud + LineSet，无 TriangleMesh）
# ═══════════════════════════════════════════════════════════════

def _risk_color(dist: float, controller: AdaptiveSafetyController) -> tuple[float, float, float]:
    """根据距离返回颜色 (绿→黄→红)。"""
    if np.isinf(dist):
        return (0.2, 0.8, 0.2)  # 绿
    if dist <= controller.d_stop:
        return (0.8, 0.1, 0.1)  # 深红
    if dist <= controller.d_slow:
        t = (dist - controller.d_stop) / max(controller.d_slow - controller.d_stop, 1e-6)
        return (0.9 * (1 - t) + 0.9 * t, 0.2 * (1 - t) + 0.9 * t, 0.1)  # 红→黄
    if dist <= controller.d_safe:
        t = (dist - controller.d_slow) / max(controller.d_safe - controller.d_slow, 1e-6)
        return (0.9 * (1 - t) + 0.2 * t, 0.9, 0.1 * (1 - t) + 0.4 * t)  # 黄→绿
    return (0.2, 0.8, 0.2)  # 绿


MAX_VIS_CLUSTERS = 12


def _mock_robot_points(y_pos: float, n_points: int = 1200) -> np.ndarray:
    """生成机械臂本体的模拟点云（多段实心圆柱骨架），随 y_pos 沿 Y 轴偏移。

    再现 AUBO i16 大致骨架 —— 基座→肩→上臂→肘→前臂→腕，
    每段用实心圆柱点云近似，确保可视化能看到"臂"的形态。
    """
    rng = np.random.default_rng(0)
    # 关节位置（基坐标系，Y = 0）
    joints = [
        [0.00, 0.0, 0.00],
        [0.00, 0.0, 0.08],
        [0.00, 0.0, 0.32],
        [0.00, 0.0, 0.38],
        [0.12, 0.0, 0.52],
        [0.15, 0.0, 0.58],
    ]
    radii = [0.055, 0.045, 0.035, 0.035, 0.028, 0.022]

    n_seg = len(joints) - 1
    base = n_points // n_seg
    seg_counts = [base] * (n_seg - 1) + [n_points - base * (n_seg - 1)]

    all_pts = []
    for idx, count in enumerate(seg_counts):
        p0, p1 = np.array(joints[idx]), np.array(joints[idx + 1])
        r0, r1 = radii[idx], radii[idx + 1]
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-8:
            continue
        du = d / L
        # 健壮的正交基
        ref = np.array([1.0, 0.0, 0.0]) if abs(du[1]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp1 = np.cross(du, ref)
        perp1 /= float(np.linalg.norm(perp1))
        perp2 = np.cross(du, perp1)

        t = rng.random(count) * L
        theta = rng.random(count) * 2 * np.pi
        r_at_t = r0 + (r1 - r0) * t / L
        r_eff = r_at_t * np.sqrt(rng.random(count))  # sqrt → 截面均匀分布

        pts = (p0[np.newaxis] + du[np.newaxis] * t[:, np.newaxis]
               + perp1[np.newaxis] * (r_eff * np.cos(theta))[:, np.newaxis]
               + perp2[np.newaxis] * (r_eff * np.sin(theta))[:, np.newaxis])
        all_pts.append(pts)

    arm = np.vstack(all_pts)
    arm[:, 1] += y_pos
    return arm


def _find_nearest_distance(
    robot_pts: np.ndarray, risk_spheres: list[RiskSphere],
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """用 KD-tree 找机器人点云到障碍球的最短距离。返回 (距离, 机器人最近点, 球心)。"""
    if len(robot_pts) == 0 or not risk_spheres:
        return float("inf"), None, None
    from scipy.spatial import cKDTree
    tree = cKDTree(robot_pts)
    best_d = float("inf")
    best_rob_pt: np.ndarray | None = None
    best_obs_pt: np.ndarray | None = None
    for rs in risk_spheres:
        d, idx = tree.query(rs.center)
        d = max(d - rs.radius, 0.0)  # 表面对表面
        if d < best_d:
            best_d = d
            best_rob_pt = robot_pts[idx]
            best_obs_pt = rs.center
    return best_d, best_rob_pt, best_obs_pt


def _find_nearest_distance_rs(
    robot_pts: np.ndarray, risk_spheres: list[RiskSphere],
) -> tuple[float, int | None]:
    """用 KD-tree 找机器人点云到风险球的最短距离，返回 (距离, object_id)。

    已废弃 — 改用 _find_nearest_cluster_distance 直接使用原始点云。
    保留仅用于可视化连线（_update_visualization 中仍使用 RiskSphere）。
    """
    if len(robot_pts) == 0 or not risk_spheres:
        return float("inf"), None
    from scipy.spatial import cKDTree
    tree = cKDTree(robot_pts)
    best_d = float("inf")
    best_id: int | None = None
    for rs in risk_spheres:
        d, idx = tree.query(rs.center)
        d = max(d - rs.radius, 0.0)  # 表面对表面
        if d < best_d:
            best_d = d
            best_id = rs.object_id
    return best_d, best_id


def _find_nearest_cluster_distance(
    robot_pts: np.ndarray,
    valid_clusters: list,
    tracked_objects: list[OccupancyObject],
) -> tuple[float, OccupancyObject | None, int | None]:
    """用 KD-tree 找机器人点云到障碍物原始点云簇的最短表面距离（点对点）。

    直接用障碍物簇的原始点云，不经过 RiskSphere / 包围球 / 包围盒膨胀，
    比 RiskSphere 表面距离更精确 —— 不会被膨胀半径吞掉真实间距。

    Returns
    -------
    min_dist : float
        最短欧氏距离 (m)，无障返回 inf
    nearest_obj : OccupancyObject | None
        最近障碍物的跟踪对象
    nearest_obj_id : int | None
        最近障碍物的跟踪 ID
    """
    if len(robot_pts) == 0 or not valid_clusters or not tracked_objects:
        return float("inf"), None, None

    from scipy.spatial import cKDTree

    # 拼合所有有效簇的原始点云
    all_obs_pts = np.vstack([c.points for c in valid_clusters])
    if len(all_obs_pts) == 0:
        return float("inf"), None, None

    obs_tree = cKDTree(all_obs_pts)

    # 对每个机器人点查最近障碍点
    dists, idxs = obs_tree.query(robot_pts)
    min_idx = int(np.argmin(dists))
    min_dist = float(dists[min_idx])
    nearest_obs_pt = all_obs_pts[idxs[min_idx]]

    # 空间关联：最近点属于哪个簇 → 对应哪个 tracked_object
    cluster_boundaries = np.cumsum([len(c.points) for c in valid_clusters])
    global_idx = int(idxs[min_idx])
    for ci, boundary in enumerate(cluster_boundaries):
        if global_idx < boundary:
            cluster_center = valid_clusters[ci].center
            best_obj = None
            best_d = 0.3  # 空间关联阈值 0.3m
            for obj in tracked_objects:
                d = float(np.linalg.norm(obj.center - cluster_center))
                if d < best_d:
                    best_d = d
                    best_obj = obj
            return min_dist, best_obj, best_obj.id if best_obj else None

    return min_dist, None, None


# ── 可视化初始化：同 live_tracking.py，预分配轻量 PointCloud / LineSet ──


def _init_visualization(enabled: bool) -> dict | None:
    """全部预分配，之后每帧只 update_geometry，绝不 remove_geometry（HUD 除外）。"""
    if not enabled:
        return None
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="安全引导运动 — 模拟 Y 轴 ±30cm",
                          width=1280, height=800)
    except Exception as e:
        print(f"[警告] 可视化窗口创建失败: {e}")
        print("[警告] 将在无可视化模式下运行（尝试 --visualize 或检查 DISPLAY 环境变量）")
        return None

    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))

    # 机械臂点云（红色）
    robot_pcd = o3d.geometry.PointCloud()
    robot_pcd.paint_uniform_color([0.8, 0.2, 0.2])
    vis.add_geometry(robot_pcd)

    # 场景点云（绿色）
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.paint_uniform_color([0.0, 0.6, 0.0])
    vis.add_geometry(scene_pcd)

    # 平面点云（蓝色）
    plane_pcd = o3d.geometry.PointCloud()
    plane_pcd.paint_uniform_color([0.2, 0.4, 0.9])
    vis.add_geometry(plane_pcd)

    # 预分配 LineSet 槽位（空几何体可以直接 add_geometry）
    def _add_empty_ls():
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
        vis.add_geometry(ls)
        return ls

    obb_lines = [_add_empty_ls() for _ in range(MAX_VIS_CLUSTERS)]
    vel_lines = [_add_empty_ls() for _ in range(MAX_VIS_CLUSTERS)]
    traj_lines = [_add_empty_ls() for _ in range(MAX_VIS_CLUSTERS)]
    dist_line = _add_empty_ls()

    vis.poll_events()
    vis.update_renderer()

    return dict(
        vis=vis,
        robot_pcd=robot_pcd,
        scene_pcd=scene_pcd,
        plane_pcd=plane_pcd,
        obb_lines=obb_lines,
        vel_lines=vel_lines,
        traj_lines=traj_lines,
        dist_line=dist_line,
        hud_geoms=[],
    )


# ── 每帧可视化更新：同 live_tracking.py，update_geometry 原地替换 ──


def _update_visualization(
    geo: dict,
    robot_pts: np.ndarray,
    cluster_result: FastClusteringFilter,
    valid: list,
    tracked_objects: list[OccupancyObject],
    risk_spheres_all: list[RiskSphere],
    y_pos: float,
    frame_idx: int,
    dt: float,
    controller: AdaptiveSafetyController,
    use_mock_camera: bool,
):
    vis = geo["vis"]
    n_show = min(len(valid), MAX_VIS_CLUSTERS)

    # ── 1. 机械臂点云 ──
    rp = geo["robot_pcd"]
    if use_mock_camera:
        mock_pts = _mock_robot_points(y_pos)
        rp.points = o3d.utility.Vector3dVector(mock_pts)
    else:
        rp.points = o3d.utility.Vector3dVector(robot_pts)
    vis.update_geometry(rp)

    # ── 2. 场景点云 ──
    sp = geo["scene_pcd"]
    if n_show > 0:
        colors = _random_colors(n_show)
        all_pts = np.vstack([valid[i].points for i in range(n_show)])
        all_cls = np.repeat(colors, [len(valid[i].points) for i in range(n_show)], axis=0)
        sp.points = o3d.utility.Vector3dVector(all_pts)
        sp.colors = o3d.utility.Vector3dVector(all_cls)
    else:
        sp.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
    vis.update_geometry(sp)

    # ── 3. 平面点云 ──
    pp = geo["plane_pcd"]
    pp.points = o3d.utility.Vector3dVector(cluster_result.plane_points)
    vis.update_geometry(pp)

    # ── 4. OBB 线框 ──
    palette = _random_colors(max(n_show, 1))
    for i in range(MAX_VIS_CLUSTERS):
        ls = geo["obb_lines"][i]
        if i < n_show:
            obb = create_obb_wireframe(valid[i].points, color=palette[i])
            ls.points = obb.points
            ls.lines = obb.lines
            ls.paint_uniform_color(palette[i])
        else:
            ls.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
        vis.update_geometry(ls)

    # ── 5. 速度箭头 + 预测轨迹 ──
    n_trk = min(len(tracked_objects), MAX_VIS_CLUSTERS)
    for i in range(MAX_VIS_CLUSTERS):
        vl = geo["vel_lines"][i]
        tl = geo["traj_lines"][i]
        if i < n_trk:
            obj = tracked_objects[i]
            s = float(np.linalg.norm(obj.velocity))
            if s > 0.01:
                tip = obj.center + (obj.velocity / s) * min(s, 0.5)
                vl.points = o3d.utility.Vector3dVector(np.vstack([obj.center, tip]))
                vl.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
                vl.paint_uniform_color([1.0, 0.2, 0.0])
            else:
                vl.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
            vis.update_geometry(vl)

            ors = [rs for rs in risk_spheres_all
                   if hasattr(rs, 'object_id') and rs.object_id == obj.id]
            if len(ors) >= 2:
                pts = np.array([rs.center for rs in ors])
                lines = np.array([[j, j + 1] for j in range(len(ors) - 1)], dtype=np.int32)
                tl.points = o3d.utility.Vector3dVector(pts)
                tl.lines = o3d.utility.Vector2iVector(lines)
                tl.paint_uniform_color([1.0, 0.2, 0.0])
            else:
                tl.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
            vis.update_geometry(tl)
        else:
            for lsi in (vl, tl):
                lsi.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
                lsi.lines = o3d.utility.Vector2iVector(np.empty((0, 2), dtype=np.int32))
                vis.update_geometry(lsi)

    # ── 6. 最近距离连线（用机器人点云 KD-tree，不再用胶囊体） ──
    dl = geo["dist_line"]
    dl_updated = False
    rob_pts_for_dist = _mock_robot_points(y_pos) if use_mock_camera else robot_pts
    if len(rob_pts_for_dist) > 0 and risk_spheres_all:
        _, rob_pt, obs_pt = _find_nearest_distance(rob_pts_for_dist, risk_spheres_all)
        if rob_pt is not None and obs_pt is not None:
            dl.points = o3d.utility.Vector3dVector(np.vstack([rob_pt, obs_pt]))
            dl.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
            dl.paint_uniform_color(_risk_color(controller.last_effective_dist, controller))
            dl_updated = True
    if not dl_updated:
        dl.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
        dl.lines = o3d.utility.Vector2iVector(np.empty((0, 2), dtype=np.int32))
    vis.update_geometry(dl)

    # ── 7. HUD 文字（结构变化 → 少量 remove/add） ──
    for g in geo.get("hud_geoms", []):
        vis.remove_geometry(g, reset_bounding_box=False)
    geo["hud_geoms"] = []

    dir_char = "→" if controller.speed_scale > 0.01 else "■"
    dist_str = (f"{controller.last_effective_dist:.3f}m"
                if not np.isinf(controller.last_effective_dist) else "---")
    info_lines = [
        f"[{frame_idx:4d}]  Y={y_pos:+.3f}m  {dir_char}",
        f"Speed: {controller.speed_scale:.0%}  ({controller.last_decision})",
        f"Dist:  {dist_str}  obj=#{controller.last_nearest_id or '--'}",
        f"dt={dt*1000:.0f}ms",
    ]
    for li, line in enumerate(info_lines):
        lb = create_text_label(
            np.array([-0.65, 0.55 - li * 0.07, 0.85]),
            line, color=_risk_color(controller.last_effective_dist, controller), size=0.028,
        )
        vis.add_geometry(lb, reset_bounding_box=False)
        geo["hud_geoms"].append(lb)

    vis.update_renderer()


# ═══════════════════════════════════════════════════════════════
# 5. 主流程
# ═══════════════════════════════════════════════════════════════

def run_safety_guided_motion(
    config_dir: str = "config",
    urdf_path: str = "urdf/aubo_i16_gripper.urdf",
    use_real_robot: bool = False,
    use_mock_camera: bool = True,
    robot_ip: str = "192.168.123.96",
    visualize: bool = False,
    single_shot: bool = False,
    show_timing: bool = False,
    no_safety: bool = False,
    plane_removal: dict | None = None,
    temporal_denoise: dict | None = None,
    **kwargs,
):
    """安全引导运动主循环。"""

    # ── 数据源 ──
    if use_mock_camera:
        reader = _StructuredMockReader()
        print("[数据] 模拟模式：桌面 + 移动球体 + 静态盒子")
        if plane_removal is None:
            plane_removal = {"enabled": True, "distance_threshold": 0.02, "max_planes": 1}
            print("[数据] 已自动启用平面分离")
    else:
        processor = SceneProcessor(
            config_dir=config_dir, urdf_path=urdf_path,
            use_real_robot=use_real_robot, use_mock_camera=False,
        )
        print("[数据] RealSense 模式（带机械臂本体去除）")

    # ── 去噪 ──
    denoiser = None
    if temporal_denoise and temporal_denoise.get("enabled", False):
        denoiser = TemporalDenoiser(
            voxel_size=temporal_denoise.get("voxel_size", 0.04),
            confidence_threshold=temporal_denoise.get("confidence_threshold", 2),
            decay=temporal_denoise.get("decay", 0.5),
        )
        print("[去噪] 已启用")

    # ── 管道组件 ──
    sphere_smoother = SphereSmoother(alpha=0.25, max_miss=5)
    tracker = OccupancyTracker(
        association_distance=0.25, alpha=0.3,
        pos_alpha=0.3, motion_gate=0.010,
        velocity_dead_zone=0.02, shape_alpha=0.4,
    )

    # ── ★ 安全引导运动新组件 ──
    motion_planner = YAxisMotionPlanner(
        range_m=kwargs.get("range_m", 0.40),
        base_speed=kwargs.get("base_speed", 0.1),
    )
    controller = AdaptiveSafetyController(
        d_safe=kwargs.get("d_safe", 0.22),
        d_slow=kwargs.get("d_slow", 0.14),
        d_stop=kwargs.get("d_stop", 0.08),
        max_decel=kwargs.get("max_decel", 2.0),
        max_accel=kwargs.get("max_accel", 0.5),
        dynamic_lookahead=kwargs.get("dynamic_lookahead", 0.15),
    )

    # ── ★ 真实机械臂运动控制 ──
    commander: RobotCommander | None = None
    if use_real_robot:
        # 从 SceneProcessor 内部提取已加载的 SDK 模块，
        # 注入 RobotCommander（主进程读取关节用），避免双重连接
        state_reader = getattr(processor, '_state_reader', None)
        robot_mod = state_reader.sdk_module if hasattr(state_reader, 'sdk_module') else None
        commander = RobotCommander(ip=robot_ip, base_speed=kwargs.get("base_speed", 0.05),
                                   robot_mod=robot_mod)
        ok = commander.connect(home_joints_deg=[0.0, 0.0, 90.0, 0.0, 90.0, 0.0])
        if not ok:
            print("[错误] 无法连接机器人，退出")
            return
        # 启动后台 Y 轴运动线程（传入 base_omega 保持速度和模拟模式一致）
        base_omega = kwargs.get("base_omega", 0.8)
        commander.start_y_oscillate(range_m=kwargs.get("range_m", 0.40), base_omega=base_omega)
        robot_y_pos = commander.get_y_pos()
        print(f"[运动] 真实机械臂模式：Y 轴 ±0.40m 往返, base_speed=0.05m/s")
        print(f"[运动] 初始 Y ≈ {robot_y_pos:.3f}m")
    else:
        print("[运动] 模拟 Y 轴 ±0.30m 往返, base_speed=0.1m/s")
    if no_safety:
        print("[安全] 已关闭 (--no-safety) — 固定 speed=1.0")
    else:
        print(f"[安全] d_safe={controller.d_safe:.2f}  "
              f"d_slow={controller.d_slow:.2f}  d_stop={controller.d_stop:.2f}  "
              f"margin={controller.surface_margin:.2f}")
        print(f"[安全] max_decel={controller.max_decel:.1f}/s  max_accel={controller.max_accel:.1f}/s")

    # ── 可视化（PointCloud + LineSet 轻量模式，同 live_tracking.py） ──
    geo = _init_visualization(visualize)
    vis = geo["vis"] if geo is not None else None
    _view_fitted = False

    print("\n=== 安全引导运动: 聚类→跟踪→预测→速度调节 ===\n")

    # ── 真实机械臂模式：用初始帧的 robot_pts 作为参考臂形 ──
    # 初始时臂在中心位置，URDF 移除最准 → robot_pts 完整且处于相机坐标系
    # 之后每帧按 SDK 报告的 Y 位移平移参考点，确保距离检查坐标系正确
    ref_robot_pts = None
    ref_y0 = 0.0

    # ── 帧生成 ──
    def _frames():
        if use_mock_camera:
            while True:
                yield reader.read()
                time.sleep(FRAME_INTERVAL_MS / 1000)  # 节流 mock 到 ~10fps
        else:
            yield from processor.run()

    # ── 后台采集线程：避免主线程被 wait_for_frames 阻塞 ──
    import queue as _queue
    _frame_queue: _queue.Queue = _queue.Queue(maxsize=3)
    _acq_alive = True

    def _acquisition_worker():
        for raw in _frames():
            if not _acq_alive:
                break
            try:
                _frame_queue.put_nowait(raw)
            except _queue.Full:
                pass  # 队列满则丢弃该帧（主线程处理不过来）
            if single_shot:
                break

    threading.Thread(target=_acquisition_worker, daemon=True).start()

    t_last = time.perf_counter()

    try:
        frame_idx = 0
        while True:
            # ── 处理窗口事件（不渲染，渲染在每帧可视化之后） ──
            if vis is not None:
                try:
                    if not vis.poll_events():
                        break
                except Exception:
                    break

            # ── 非阻塞取帧 ──
            try:
                raw = _frame_queue.get_nowait()
            except _queue.Empty:
                time.sleep(0.005)
                continue

            frame_idx += 1
            t0 = time.perf_counter()

            # ── 获取点云 ──
            if use_mock_camera:
                scene_pts = np.asarray(raw.points_cam, dtype=np.float64)
                robot_pts = np.empty((0, 3))
                timestamp = getattr(raw, "timestamp", time.time())
            else:
                scene_pts = np.asarray(
                    getattr(raw, "scene_points", np.empty((0, 3))), dtype=np.float64)
                robot_pts = np.asarray(
                    getattr(raw, "robot_points", np.empty((0, 3))), dtype=np.float64)
                timestamp = getattr(raw, "timestamp", time.time())

            # ── 时域去噪 ──
            n_denoised = 0
            if denoiser is not None:
                scene_pts = denoiser.filter(scene_pts)
                n_denoised = len(denoiser.last_noise)

            # ── 聚类 ──
            ws = {"x": [-1.5, 1.5], "y": [-1.5, 1.5], "z": [-0.5, 1.8]} if use_mock_camera \
                else getattr(processor, "_workspace", None) or getattr(processor, "workspace", None)

            cluster_result = FastClusteringFilter(
                scene_pts, robot_pts, workspace=ws, plane_removal=plane_removal,
            )
            t1 = time.perf_counter()

            # ── 半径平滑 + 稳定筛选 ──
            if cluster_result.clusters:
                sphere_smoother.update([
                    (c.center, float(np.max(np.linalg.norm(c.points - c.center, axis=1))) + 0.02)
                    for c in cluster_result.clusters
                ])
                valid = [
                    cl for cl, tr in zip(cluster_result.clusters, sphere_smoother.tracks)
                    if tr.age >= sphere_smoother.min_age
                ]
            else:
                valid = []
                sphere_smoother.update([])

            # ── 跟踪 + 速度 ──
            detections = [make_occupancy_object(cl.points, timestamp=timestamp) for cl in valid]
            tracked_objects = tracker.update(detections, timestamp=timestamp)

            # ── 预测 ──
            risk_spheres_all = predict_risk_spheres(
                tracked_objects, horizon=0.5, step=0.1)

            # ── dt ──
            now = time.perf_counter()
            dt = max(now - t_last, 0.001)
            t_last = now

            # =====================================================
            # ★ 安全引导运动层
            # =====================================================

            if no_safety:
                # ── 无安全模式：固定 speed=1.0，跳过距离检查 + 安全控制 ──
                speed_scale = 1.0
                if commander is not None:
                    y_pos = commander.get_y_pos()
                    commander.set_speed_scale(1.0)
                else:
                    y_pos = motion_planner.step(dt, 1.0)
                min_dist = float("inf")
                nearest_obj = None
                controller.speed_scale = 1.0
                controller.last_decision = "NO_SAFETY"
            else:
                # a) 读取当前位置（先不动作，用于距离检查）
                if commander is not None:
                    y_pos = commander.get_y_pos()
                    # ▸ ▸ 等距离检查 + 安全计算后，再写入 speed_scale（步骤 d）
                else:
                    y_pos = motion_planner.y_pos  # 读取当前位置（不步进）

                # b) 距离检查（用机械臂点云 × 障碍物原始点云簇 KD-tree — 点对点，不经过 RiskSphere 膨胀）
                # 记录第一帧有效的 robot_pts 作为参考臂形（初始位置 URDF 移除最准）
                if ref_robot_pts is None and len(robot_pts) > 100:
                    ref_robot_pts = robot_pts.copy()
                    ref_y0 = y_pos
                    print(f"[初始化] 参考臂形已记录: {len(ref_robot_pts)} 点, ref_y0={ref_y0:.3f}m")

                if len(robot_pts) > 100:
                    rob_pts_for_dist = robot_pts
                elif ref_robot_pts is not None:
                    # 用参考臂形平移 Y 位移 ≈ 真实臂在相机坐标系中的当前位置
                    dy = y_pos - ref_y0
                    rob_pts_for_dist = ref_robot_pts + np.array([0.0, dy, 0.0], dtype=float)
                else:
                    rob_pts_for_dist = _mock_robot_points(y_pos)

                # 直接用原始点云簇 KD-tree（点对点，不经过 RiskSphere 膨胀）
                min_dist, nearest_obj, _ = _find_nearest_cluster_distance(
                    rob_pts_for_dist, valid, tracked_objects)

                # c) 安全控制器 → 用当前帧数据计算 speed_scale
                speed_scale = controller.evaluate(min_dist, nearest_obj, None, dt)

                # d) 将安全速度写入运动执行器（下一帧生效）
                if commander is not None:
                    commander.set_speed_scale(speed_scale)
                else:
                    y_pos = motion_planner.step(dt, speed_scale)

            t2 = time.perf_counter()

            # ── 诊断 ──
            n_rob = len(robot_pts)
            n_rs = len(risk_spheres_all)

            # ── 控制台输出 ──
            n_valid = len(cluster_result.clusters)
            n_plane = len(cluster_result.plane_points)
            n_tracked = len(tracked_objects)
            # 用 Y 变化方向显示箭头
            _last_y = getattr(motion_planner, '_last_y', y_pos) if not commander else y_pos
            dir_char = "→" if y_pos >= _last_y else "←"
            if not commander:
                motion_planner._last_y = y_pos
            dist_str = (f"{min_dist:.3f}" if not np.isinf(min_dist) else "---")
            obj_str = f"#{nearest_obj.id}" if nearest_obj is not None else "--"

            timing = f"  ⏱ {(t1-t0)*1000:.0f}+{(t2-t1)*1000:.0f}={(t2-t0)*1000:.0f}ms" if show_timing else ""
            d_tag = f"  denoised={n_denoised}" if n_denoised > 0 else ""

            print(
                f"[{frame_idx:4d}]  Y={y_pos:+.3f}m {dir_char}  "
                f"speed={speed_scale:.0%}  "
                f"dist={dist_str}  obj={obj_str}  "
                f"dec={controller.last_decision}  "
                f"rob={n_rob} rs={n_rs}{d_tag}{timing}"
            )

            # ── 可视化 ──
            if vis is not None:
                _update_visualization(
                    geo=geo,
                    robot_pts=robot_pts,
                    cluster_result=cluster_result, valid=valid,
                    tracked_objects=tracked_objects,
                    risk_spheres_all=risk_spheres_all,
                    y_pos=y_pos, frame_idx=frame_idx, dt=dt,
                    controller=controller,
                    use_mock_camera=use_mock_camera,
                )
                if not _view_fitted:
                    vis.reset_view_point(True)
                    opt = vis.get_render_option()
                    opt.point_size = 2.0
                    _view_fitted = True

            # 帧率
            elapsed = (time.perf_counter() - t0) * 1000
            sleep = max(0, FRAME_INTERVAL_MS - elapsed)
            if sleep > 0 and not single_shot:
                time.sleep(sleep / 1000)

            if single_shot:
                print("\n[single shot] 等待窗口关闭...")
                while vis is not None:
                    try:
                        if not vis.poll_events():
                            break
                    except Exception:
                        break
                    time.sleep(0.05)
                break

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        # 停后台采集线程
        _acq_alive = False
        time.sleep(0.05)
        # 停真实机械臂
        if commander is not None:
            commander.stop()
        if vis is not None:
            try:
                vis.destroy_window()
            except Exception:
                pass
        if not use_mock_camera:
            processor.stop()


# ═══════════════════════════════════════════════════════════════
# 6. 命令行
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="安全引导运动：Y 轴往返 + 障碍自适应调速")
    p.add_argument("--config", default="config")
    p.add_argument("--urdf", default="urdf/aubo_i16_gripper.urdf")
    p.add_argument("--real-robot", action="store_true",
                   help="真实机械臂模式：连接 AUBO i16 + RealSense")
    p.add_argument("--robot-ip", type=str, default="192.168.123.96",
                   help="AUBO 机器人 IP 地址")
    p.add_argument("--mock", action="store_true", default=False,
                   help="模拟模式（默认，如未指定 --real-robot 则自动启用）")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--single", action="store_true")
    p.add_argument("--timing", action="store_true")
    p.add_argument("--remove-planes", action="store_true")
    p.add_argument("--no-safety", action="store_true",
                   help="关闭安全引导，固定 speed_scale=1.0，纯运动")
    p.add_argument("--plane-dist", type=float, default=0.02)
    p.add_argument("--max-planes", type=int, default=1)
    p.add_argument("--temporal-denoise", action="store_true")
    p.add_argument("--denoise-voxel", type=float, default=0.04)
    p.add_argument("--denoise-conf", type=int, default=2)
    p.add_argument("--denoise-decay", type=float, default=0.5)

    # 安全 & 运动参数（覆盖默认）
    p.add_argument("--d-safe", type=float, default=None)
    p.add_argument("--d-slow", type=float, default=None)
    p.add_argument("--d-stop", type=float, default=None)
    p.add_argument("--max-decel", type=float, default=None)
    p.add_argument("--max-accel", type=float, default=None)
    p.add_argument("--base-speed", type=float, default=None)
    p.add_argument("--range", type=float, default=None)

    args = p.parse_args()

    # 获取速度参数
    # 未指定任何模式 → 默认 mock
    use_mock = args.mock or not args.real_robot

    kwargs = {}
    if args.d_safe is not None:
        kwargs["d_safe"] = args.d_safe
    if args.d_slow is not None:
        kwargs["d_slow"] = args.d_slow
    if args.d_stop is not None:
        kwargs["d_stop"] = args.d_stop
    if args.max_decel is not None:
        kwargs["max_decel"] = args.max_decel
    if args.max_accel is not None:
        kwargs["max_accel"] = args.max_accel
    if args.base_speed is not None:
        kwargs["base_speed"] = args.base_speed
    if args.range is not None:
        kwargs["range_m"] = args.range

    run_safety_guided_motion(
        config_dir=args.config, urdf_path=args.urdf,
        use_real_robot=args.real_robot, use_mock_camera=use_mock,
        robot_ip=args.robot_ip,
        visualize=args.visualize, single_shot=args.single,
        show_timing=args.timing,
        no_safety=args.no_safety,
        plane_removal=(
            {"enabled": True, "distance_threshold": args.plane_dist, "max_planes": args.max_planes}
            if args.remove_planes else None
        ),
        temporal_denoise=(
            {"enabled": True, "voxel_size": args.denoise_voxel,
             "confidence_threshold": args.denoise_conf, "decay": args.denoise_decay}
            if args.temporal_denoise else None
        ),
        **kwargs,
    )


if __name__ == "__main__":
    main()
