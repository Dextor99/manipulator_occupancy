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
from risk.distance_check import min_capsule_sphere_distance
from robot.capsule_model import Capsule, mock_capsules, capsules_from_config
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

    使用时间基准正弦波（与 RobotCommander._motion_loop 一致），
    speed_scale 影响波形角速度，实现平滑加减速。
    """

    def __init__(self, range_m: float = 0.30, base_speed: float = 0.1):
        self.range = range_m
        self.base_speed = base_speed
        self.y_pos = 0.0
        self._t_start = time.perf_counter()
        self._omega = 0.8  # rad/s，与 RobotCommander 一致

    def step(self, dt: float, speed_scale: float) -> float:
        """根据 speed_scale 更新位置，返回当前 y_pos（dt 保留为接口兼容）。"""
        t_elapsed = time.perf_counter() - self._t_start
        phase = self._omega * max(speed_scale, 0.0) * t_elapsed
        self.y_pos = self.range * np.sin(phase)
        return self.y_pos


# ═══════════════════════════════════════════════════════════════
# 2. 移动胶囊体提供器
# ═══════════════════════════════════════════════════════════════

class MovingCapsuleProvider:
    """根据 y_pos 偏移机械臂胶囊体（基座不动，末端偏移最大）。"""

    OFFSET_FACTORS = {
        "base_link": 0.0,
        "upper_arm": 0.7,
        "forearm": 0.85,
        "wrist": 0.95,
    }

    def __init__(self, base_capsules: list[Capsule] | None = None):
        self.base_capsules = base_capsules or mock_capsules()

    def get_capsules(self, y_pos: float) -> list[Capsule]:
        moved = []
        for cap in self.base_capsules:
            factor = self.OFFSET_FACTORS.get(cap.name, 0.5)
            offset = np.array([0.0, factor * y_pos, 0.0])
            moved.append(Capsule(
                name=cap.name,
                a=cap.a + offset,
                b=cap.b + offset,
                radius=cap.radius,
            ))
        return moved


# ═══════════════════════════════════════════════════════════════
# 3. 自适应安全控制器
# ═══════════════════════════════════════════════════════════════

class AdaptiveSafetyController:
    """基于障碍距离平滑调节速度倍率。

    - 静态障碍 → 按实际距离减速
    - 动态接近 → 预判未来位置，提前减速
    - 动态远离 → 正常通过

    速度渐变不跳变，避免突兀启停。
    """

    def __init__(
        self,
        d_safe: float = 0.22,
        d_slow: float = 0.12,
        d_stop: float = 0.06,
        max_decel: float = 2.0,
        max_accel: float = 0.5,
        dynamic_lookahead: float = 0.3,
    ):
        self.d_safe = d_safe
        self.d_slow = d_slow
        self.d_stop = d_stop
        self.max_decel = max_decel
        self.max_accel = max_accel
        self.dynamic_lookahead = dynamic_lookahead

        self.speed_scale = 1.0
        self.last_decision = "SAFE"
        self.last_nearest_id: int | None = None
        self.last_effective_dist = float("inf")

    def evaluate(
        self,
        distance: float,
        nearest_obj: OccupancyObject | None,
        capsules: list[Capsule],
        dt: float,
    ) -> float:
        """返回平滑后的 speed_scale (0.0~1.0)。"""
        # ── 无障碍 → 全速 ──
        if np.isinf(distance) or nearest_obj is None:
            target = 1.0
            self.last_decision = "SAFE"
            self.last_nearest_id = None
            self.last_effective_dist = float("inf")
        else:
            speed = float(np.linalg.norm(nearest_obj.velocity))
            effective_dist = distance

            if speed > 0.01:
                # 计算接近速度
                wrist_center = self._get_wrist_center(capsules)
                if wrist_center is not None:
                    to_robot = wrist_center - nearest_obj.center
                    dist_to_robot = float(np.linalg.norm(to_robot))
                    if dist_to_robot > 0.001:
                        closing = max(
                            0.0,
                            float(np.dot(nearest_obj.velocity, to_robot / dist_to_robot)),
                        )
                    else:
                        closing = 0.0
                else:
                    closing = 0.0

                if closing > 0.01:
                    effective_dist = max(0.0, distance - closing * self.dynamic_lookahead)
                    self.last_decision = "DYNAMIC_APPROACH"
                else:
                    self.last_decision = "DYNAMIC_FLEE" if speed > 0.01 else "STATIC"
            else:
                self.last_decision = "STATIC"

            # ── 三段映射 → 目标 speed_scale ──
            if effective_dist >= self.d_safe:
                target = 1.0
            elif effective_dist <= self.d_stop:
                target = 0.0
            else:
                target = (effective_dist - self.d_stop) / (self.d_safe - self.d_stop)

            self.last_nearest_id = nearest_obj.id
            self.last_effective_dist = effective_dist

        # ── 速度渐变 ──
        max_change = self.max_decel * dt if target < self.speed_scale else self.max_accel * dt
        delta = np.clip(target - self.speed_scale, -max_change, max_change)
        self.speed_scale = float(np.clip(self.speed_scale + delta, 0.0, 1.0))
        return self.speed_scale

    @staticmethod
    def _get_wrist_center(capsules: list[Capsule]) -> np.ndarray | None:
        for cap in capsules:
            if cap.name == "wrist":
                return (cap.a + cap.b) * 0.5
        return None


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
    motion_planner = YAxisMotionPlanner(range_m=0.30, base_speed=0.1)
    capsule_provider = MovingCapsuleProvider()
    controller = AdaptiveSafetyController(
        d_safe=0.22, d_slow=0.12, d_stop=0.06,
        max_decel=2.0, max_accel=0.5, dynamic_lookahead=0.3,
    )

    # ── ★ 真实机械臂运动控制 ──
    commander: RobotCommander | None = None
    if use_real_robot:
        # 从 SceneProcessor 内部提取已加载的 SDK 模块，
        # 注入 RobotCommander（主进程读取关节用），避免双重连接
        state_reader = getattr(processor, '_state_reader', None)
        robot_mod = state_reader.sdk_module if hasattr(state_reader, 'sdk_module') else None
        commander = RobotCommander(ip=robot_ip, base_speed=0.05,
                                   robot_mod=robot_mod)
        ok = commander.connect(home_joints_deg=[0.0, 0.0, 90.0, 0.0, 90.0, 0.0])
        if not ok:
            print("[错误] 无法连接机器人，退出")
            return
        # 启动后台 Y 轴运动线程
        commander.start_y_oscillate(range_m=0.30)
        robot_y_pos = commander.get_y_pos()
        print(f"[运动] 真实机械臂模式：Y 轴 ±0.30m 往返, base_speed=0.05m/s")
        print(f"[运动] 初始 Y ≈ {robot_y_pos:.3f}m")
    else:
        print("[运动] 模拟 Y 轴 ±0.30m 往返, base_speed=0.1m/s")
    print("[安全] d_safe=0.22  d_slow=0.12  d_stop=0.06")
    print(f"[安全] max_decel={controller.max_decel:.1f}/s  max_accel={controller.max_accel:.1f}/s")

    # ── 可视化（PointCloud + LineSet 轻量模式，同 live_tracking.py） ──
    geo = _init_visualization(visualize)
    vis = geo["vis"] if geo is not None else None
    _view_fitted = False

    print("\n=== 安全引导运动: 聚类→跟踪→预测→速度调节 ===\n")

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

            # a) 机器人步进（模拟 / 真实）
            if commander is not None:
                # 真实机械臂：将安全控制器的 speed_scale 写入后台线程
                commander.set_speed_scale(controller.speed_scale)
                y_pos = commander.get_y_pos()
            else:
                # 模拟模式
                y_pos = motion_planner.step(dt, controller.speed_scale)

            # b) 胶囊体偏移
            moved_capsules = capsule_provider.get_capsules(y_pos)

            # c) 距离检查（用偏移后的胶囊体 × 所有风险球）
            min_dist, nearest_id = min_capsule_sphere_distance(moved_capsules, risk_spheres_all)

            # d) 找最近物体对象
            nearest_obj = None
            if nearest_id is not None:
                for obj in tracked_objects:
                    if obj.id == nearest_id:
                        nearest_obj = obj
                        break

            # e) 安全控制器 → speed_scale
            speed_scale = controller.evaluate(min_dist, nearest_obj, moved_capsules, dt)

            t2 = time.perf_counter()

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
                f"dec={controller.last_decision}{d_tag}{timing}"
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
