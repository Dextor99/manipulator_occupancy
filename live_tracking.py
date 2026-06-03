"""
实时占据感知：复用 test_clustering_filtering.py 的聚类结果 →
多目标跟踪 + 速度估计 + 运动预测 + 高级可视化。

设计思路
--------
数据采集 / 机械臂去除 / 聚类过滤 → 全部复用 test_clustering_filtering 的成果
在此基础上增加：多目标跟踪（OccupancyTracker）、3D 速度估计（EMA）、
运动预测（predict_risk_spheres）、高级可视化（速度箭头/预测轨迹/ID标签）

用法
----
  # 模拟数据（移动球体 + 静态盒子 + 桌面，自动分离平面）
  python live_tracking.py --mock-camera --visualize

  # 真实机械臂 + RealSense（如有桌面加 --remove-planes）
  python live_tracking.py --real-robot --visualize --remove-planes

  # 单帧调试
  python live_tracking.py --mock-camera --single --visualize
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from test_remove_robot_points_fast import SceneProcessor, ProcessedFrame
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
from risk.prediction import predict_risk_spheres, RiskSphere

# ── 可视化常量 ─────────────────────────────────────────────────
MAX_VIS_CLUSTERS = 12
VEL_ARROW_SCALE = 1.0   # 1m/s → 1m 箭头


# ── 可视化辅助 ─────────────────────────────────────────────────

def make_velocity_arrow(center: np.ndarray, velocity: np.ndarray,
                        scale: float = 1.0) -> o3d.geometry.LineSet:
    speed = np.linalg.norm(velocity)
    if speed < 0.005:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
        return ls
    tip = center + (velocity / speed) * min(speed * scale, 0.5)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.vstack([center, tip]))
    ls.lines = o3d.utility.Vector2iVector([[0, 1]])
    ls.paint_uniform_color([1.0, 0.2, 0.0])
    return ls


def make_prediction_trajectory(risk_spheres: list[RiskSphere],
                               color=(1.0, 0.6, 0.0)) -> o3d.geometry.LineSet:
    if len(risk_spheres) < 2:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
        return ls
    pts = np.array([rs.center for rs in risk_spheres])
    lines = [[i, i + 1] for i in range(len(pts) - 1)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color(color)
    return ls


# ── 结构化模拟数据 ─────────────────────────────────────────────

class _StructuredMockReader:
    """在基坐标系下生成：桌面（z=0平面）+ 移动球体 + 静态盒子。"""

    def __init__(self, seed: int = 2):
        self.index = 0
        self.rng = np.random.default_rng(seed)

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


# ── 主流程 ─────────────────────────────────────────────────────

def run_live(
    config_dir: str = "config",
    urdf_path: str = "urdf/aubo_i16_gripper.urdf",
    use_real_robot: bool = False,
    use_mock_camera: bool = False,
    visualize: bool = True,
    single_shot: bool = False,
    show_timing: bool = False,
    plane_removal: dict | None = None,
    temporal_denoise: dict | None = None,
    **kwargs,
):
    """实时管线：采集+聚类 → 跟踪+速度 → 预测+可视化。"""

    # ── 选择数据源 ──
    # 与 test_clustering_filtering.py 一致：
    #   mock → 结构化桌面+球体+盒子
    #   real → SceneProcessor（RealSense + 机械臂去除 + 裁剪）
    if use_mock_camera:
        reader = _StructuredMockReader()
        print("[数据] 模拟模式：桌面 + 移动球体 + 静态盒子")
        if plane_removal is None:
            plane_removal = {"enabled": True, "distance_threshold": 0.02, "max_planes": 1}
            print("[数据] 已自动启用平面分离")
        # mock 模式无机械臂点
        frame_data_obj = {"robot_points": np.empty((0, 3))}
    else:
        processor = SceneProcessor(
            config_dir=config_dir,
            urdf_path=urdf_path,
            use_real_robot=use_real_robot,
            use_mock_camera=False,
        )
        print("[数据] RealSense 模式（带机械臂本体去除）")

    # ── 时域去噪 ──
    denoiser = None
    if temporal_denoise and temporal_denoise.get("enabled", False):
        denoiser = TemporalDenoiser(
            voxel_size=temporal_denoise.get("voxel_size", 0.04),
            confidence_threshold=temporal_denoise.get("confidence_threshold", 2),
            decay=temporal_denoise.get("decay", 0.5),
        )
        print("[去噪] 已启用")

    # ── 球体半径时域平滑 ──
    sphere_smoother = SphereSmoother(alpha=0.25, max_miss=5)

    # ── ★ 多目标跟踪 + 速度估计 ──
    tracker = OccupancyTracker(association_distance=0.25, alpha=0.3)

    # ── 可视化初始化 ──
    vis = None
    geo = _init_visualization(visualize)
    if geo is not None:
        vis = geo["vis"]
    _view_fitted = False

    print("\n=== Live: 采集+聚类 → 跟踪+速度 → 预测+可视化 ===\n")

    # ── 帧生成 ──
    def _frames():
        if use_mock_camera:
            while True:
                yield reader.read()
        else:
            yield from processor.run()

    frame_iter = [next(_frames())] if single_shot else _frames()

    try:
        for frame_idx, raw in enumerate(frame_iter):
            t0 = time.perf_counter()

            # ── 获取点云 ──
            if use_mock_camera:
                scene_pts = np.asarray(raw.points_cam, dtype=np.float64)
                robot_pts = np.empty((0, 3))
                timestamp = getattr(raw, "timestamp", time.time())
            else:
                scene_pts = np.asarray(getattr(raw, "scene_points", np.empty((0, 3))), dtype=np.float64)
                robot_pts = np.asarray(getattr(raw, "robot_points", np.empty((0, 3))), dtype=np.float64)
                timestamp = getattr(raw, "timestamp", time.time())

            # ── [可选] 时域去噪 ──
            n_denoised = 0
            if denoiser is not None:
                scene_pts = denoiser.filter(scene_pts)
                n_denoised = len(denoiser.last_noise)

            # ── ★ 聚类+过滤（完全复用 test_clustering_filtering 的逻辑） ──
            if use_mock_camera:
                ws = {"x": [-1.5, 1.5], "y": [-1.5, 1.5], "z": [-0.5, 1.8]}
            else:
                ws = getattr(processor, "_workspace", None) or getattr(processor, "workspace", None)

            cluster_result = FastClusteringFilter(
                scene_pts, robot_pts,
                workspace=ws,
                plane_removal=plane_removal,
            )
            t1 = time.perf_counter()

            # ── 半径时域平滑 + 稳定簇筛选 ──
            valid_all = cluster_result.clusters
            if valid_all:
                raw_centers = [c.center for c in valid_all]
                raw_radii = [float(np.max(np.linalg.norm(c.points - c.center, axis=1))) + 0.02
                             for c in valid_all]
                sphere_smoother.update(list(zip(raw_centers, raw_radii)))
                stable = []
                for cl, track in zip(valid_all, sphere_smoother.tracks):
                    if track.age >= sphere_smoother.min_age:
                        stable.append(cl)
                valid = stable
            else:
                valid = []
                sphere_smoother.update([])

            # ── ★ 多目标跟踪 + 3D 速度估计（EMA） ──
            detections = [make_occupancy_object(cl.points, timestamp=timestamp)
                          for cl in valid]
            tracked_objects = tracker.update(detections, timestamp=timestamp)

            # ── ★ 基于速度的运动预测（外推球体膨胀） ──
            risk_spheres_all = predict_risk_spheres(tracked_objects, horizon=0.5, step=0.1)
            t2 = time.perf_counter()

            # ── 控制台 ──
            n_valid = len(cluster_result.clusters)
            n_filtered = len(cluster_result.filtered_out)
            n_noise = len(cluster_result.noise_points)
            n_plane = len(cluster_result.plane_points)
            n_tracked = len(tracked_objects)
            n_stable = len(valid)

            d_tag = f"  denoised={n_denoised}" if n_denoised > 0 else ""
            ms_cluster = (t1 - t0) * 1000
            ms_track = (t2 - t1) * 1000
            ms_total = (t2 - t0) * 1000
            timing = f"  ⏱ {ms_cluster:.0f}+{ms_track:.0f}={ms_total:.0f}ms" if show_timing else ""

            print(f"[{frame_idx:4d}] clusters={n_valid}  stable={n_stable}  "
                  f"track={n_tracked}  filter={n_filtered}  noise={n_noise}  "
                  f"plane={n_plane}{d_tag}{timing}")

            # 每个物体的速度和预测
            for obj in tracked_objects:
                s = float(np.linalg.norm(obj.velocity))
                print(f"         #{obj.id} age={obj.age}  "
                      f"c=({obj.center[0]:.2f},{obj.center[1]:.2f},{obj.center[2]:.2f})  "
                      f"v={s:.3f}m/s  r={obj.radius:.3f}m")

            if risk_spheres_all:
                from collections import defaultdict
                for oid, spheres in defaultdict(list, {
                    oid: [rs for rs in risk_spheres_all if rs.object_id == oid]
                    for oid in {rs.object_id for rs in risk_spheres_all}
                }).items():
                    last = spheres[-1]
                    print(f"         pred #{oid}: t={last.tau:.1f}s → "
                          f"({last.center[0]:.2f},{last.center[1]:.2f},{last.center[2]:.2f})  "
                          f"r={last.radius:.3f}m")

            # ── 可视化 ──
            if vis is not None:
                _update_visualization(
                    vis=vis, geo=geo,
                    robot_pts=robot_pts,
                    cluster_result=cluster_result,
                    valid=valid,
                    tracked_objects=tracked_objects,
                    risk_spheres_all=risk_spheres_all,
                    denoiser=denoiser,
                )
                if not _view_fitted and (len(robot_pts) > 0 or len(scene_pts) > 0):
                    vis.reset_view_point(True)
                    opt = vis.get_render_option()
                    opt.point_size = 2.0
                    _view_fitted = True
                if not vis.poll_events():
                    break

            # 帧率
            elapsed_ms = (time.perf_counter() - t0) * 1000
            sleep = max(0, FRAME_INTERVAL_MS - elapsed_ms)
            if sleep > 0 and not single_shot:
                time.sleep(sleep / 1000)

            if single_shot:
                print("\n[single shot] 关闭窗口退出")
                if vis is not None:
                    vis.run()
                break

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        if vis is not None:
            vis.destroy_window()
        if not use_mock_camera:
            processor.stop()


# ── 可视化初始化 + 更新 ─────────────────────────────────────────

def _init_visualization(enabled: bool) -> dict | None:
    if not enabled:
        return None
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Live — 聚类→跟踪→速度→预测",
                      width=1280, height=800)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))

    robot_pcd = o3d.geometry.PointCloud()
    robot_pcd.paint_uniform_color([0.8, 0.2, 0.2])
    vis.add_geometry(robot_pcd)

    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.paint_uniform_color([0.0, 0.6, 0.0])
    vis.add_geometry(scene_pcd)

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.paint_uniform_color([0.6, 0.6, 0.6])
    vis.add_geometry(filtered_pcd)

    noise_pcd = o3d.geometry.PointCloud()
    noise_pcd.paint_uniform_color([0.3, 0.3, 0.3])
    vis.add_geometry(noise_pcd)

    plane_pcd = o3d.geometry.PointCloud()
    plane_pcd.paint_uniform_color([0.2, 0.4, 0.9])
    vis.add_geometry(plane_pcd)

    denoise_pcd = o3d.geometry.PointCloud()
    denoise_pcd.paint_uniform_color([0.6, 0.2, 0.8])
    vis.add_geometry(denoise_pcd)

    center_pts = o3d.geometry.PointCloud()
    vis.add_geometry(center_pts)

    obb, sphere, vel, traj = [], [], [], []
    for _ in range(MAX_VIS_CLUSTERS):
        for lst in (obb, sphere, vel, traj):
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
            vis.add_geometry(ls)
            lst.append(ls)

    return dict(vis=vis, robot_pcd=robot_pcd, scene_pcd=scene_pcd,
                filtered_pcd=filtered_pcd, noise_pcd=noise_pcd,
                plane_pcd=plane_pcd, denoise_pcd=denoise_pcd,
                center_pts=center_pts,
                obb_lines=obb, sphere_lines=sphere,
                vel_lines=vel, traj_lines=traj, id_labels=[])


def _update_visualization(
    vis: o3d.visualization.Visualizer,
    geo: dict,
    robot_pts: np.ndarray,
    cluster_result: FastClusteringFilter,
    valid: list,
    tracked_objects: list,
    risk_spheres_all: list[RiskSphere],
    denoiser: TemporalDenoiser | None,
):
    n_show = min(len(valid), MAX_VIS_CLUSTERS)

    robot_pcd = geo["robot_pcd"]
    robot_pcd.points = o3d.utility.Vector3dVector(robot_pts)
    vis.update_geometry(robot_pcd)

    scene_pcd = geo["scene_pcd"]
    if n_show > 0:
        colors = _random_colors(n_show)
        all_pts = np.vstack([valid[i].points for i in range(n_show)])
        all_cols = np.repeat(colors, [len(valid[i].points) for i in range(n_show)], axis=0)
        scene_pcd.points = o3d.utility.Vector3dVector(all_pts)
        scene_pcd.colors = o3d.utility.Vector3dVector(all_cols)
    else:
        scene_pcd.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
    vis.update_geometry(scene_pcd)

    # 被过滤掉的簇点和噪声点 → 不显示（只留有效簇和平面）
    geo["filtered_pcd"].points = o3d.utility.Vector3dVector(np.empty((0, 3)))
    vis.update_geometry(geo["filtered_pcd"])
    geo["noise_pcd"].points = o3d.utility.Vector3dVector(np.empty((0, 3)))
    vis.update_geometry(geo["noise_pcd"])
    geo["plane_pcd"].points = o3d.utility.Vector3dVector(cluster_result.plane_points)
    vis.update_geometry(geo["plane_pcd"])
    if denoiser is not None:
        geo["denoise_pcd"].points = o3d.utility.Vector3dVector(denoiser.last_noise)
        vis.update_geometry(geo["denoise_pcd"])

    # 中心
    cp = geo["center_pts"]
    if n_show > 0:
        centers = np.array([valid[i].center for i in range(n_show)])
        cp.points = o3d.utility.Vector3dVector(centers)
        cp.colors = o3d.utility.Vector3dVector(np.array([colors[i] for i in range(n_show)]))
    else:
        cp.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
    vis.update_geometry(cp)

    # stable → tracked 映射
    stable_to_obj = {}
    if n_show > 0 and tracked_objects:
        from scipy.spatial import cKDTree
        tc = np.array([obj.center for obj in tracked_objects])
        if len(tc) > 0:
            tree = cKDTree(tc)
            for vi in range(n_show):
                d, idx = tree.query(valid[vi].center, k=1)
                if d < 0.30:
                    stable_to_obj[vi] = tracked_objects[idx]

    # OBB + 包围球 + 速度箭头 + 轨迹
    for i in range(MAX_VIS_CLUSTERS):
        if i < n_show:
            cl, ci = valid[i], colors[i]
            obb = create_obb_wireframe(cl.points, color=ci)
            geo["obb_lines"][i].points = obb.points
            geo["obb_lines"][i].lines = obb.lines
            geo["obb_lines"][i].paint_uniform_color(ci)
            vis.update_geometry(geo["obb_lines"][i])

            om = fit_obb(cl.points)
            sw = create_sphere_wireframe(om.center,
                float(np.max(np.linalg.norm(cl.points - om.center, axis=1))) + 0.02,
                color=(1.0, 0.9, 0.0))
            geo["sphere_lines"][i].points = sw.points
            geo["sphere_lines"][i].lines = sw.lines
            geo["sphere_lines"][i].paint_uniform_color((1.0, 0.9, 0.0))
            vis.update_geometry(geo["sphere_lines"][i])
        else:
            for ls in (geo["obb_lines"][i], geo["sphere_lines"][i]):
                ls.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
                vis.update_geometry(ls)

        # 速度箭头
        if i < n_show and i in stable_to_obj:
            obj = stable_to_obj[i]
            arr = make_velocity_arrow(obj.center, obj.velocity, VEL_ARROW_SCALE)
            geo["vel_lines"][i].points = arr.points
            geo["vel_lines"][i].lines = arr.lines
            geo["vel_lines"][i].paint_uniform_color([1.0, 0.2, 0.0])
        else:
            geo["vel_lines"][i].points = o3d.utility.Vector3dVector(np.empty((0, 3)))
        vis.update_geometry(geo["vel_lines"][i])

        # 预测轨迹
        if i < n_show and i in stable_to_obj:
            obj = stable_to_obj[i]
            ors = [rs for rs in risk_spheres_all if rs.object_id == obj.id]
            tr = make_prediction_trajectory(ors, colors[i] if i < n_show else (1.0, 0.6, 0.0))
            geo["traj_lines"][i].points = tr.points
            geo["traj_lines"][i].lines = tr.lines
            geo["traj_lines"][i].paint_uniform_color(colors[i] if i < n_show else (1.0, 0.6, 0.0))
        else:
            geo["traj_lines"][i].points = o3d.utility.Vector3dVector(np.empty((0, 3)))
        vis.update_geometry(geo["traj_lines"][i])

    # ID 标签
    for label in geo["id_labels"]:
        vis.remove_geometry(label, reset_bounding_box=False)
    geo["id_labels"].clear()
    for i in range(min(n_show, 6)):
        if i in stable_to_obj:
            obj = stable_to_obj[i]
            s = float(np.linalg.norm(obj.velocity))
            lb = create_text_label(obj.center + (0, 0, 0.08),
                                   f"#{obj.id}  {s:.2f}m/s",
                                   color=(1, 1, 1), size=0.03)
            vis.add_geometry(lb, reset_bounding_box=False)
            geo["id_labels"].append(lb)

    vis.update_renderer()


# ── 命令行 ─────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Live: 聚类→跟踪→速度→预测")
    p.add_argument("--config", default="config")
    p.add_argument("--urdf", default="urdf/aubo_i16_gripper.urdf")
    p.add_argument("--real-robot", action="store_true")
    p.add_argument("--mock-camera", action="store_true")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--single", action="store_true", help="单帧模式")
    p.add_argument("--timing", action="store_true", help="打印耗时")
    p.add_argument("--remove-planes", action="store_true")
    p.add_argument("--plane-dist", type=float, default=0.02)
    p.add_argument("--max-planes", type=int, default=1)
    p.add_argument("--temporal-denoise", action="store_true")
    p.add_argument("--denoise-voxel", type=float, default=0.04)
    p.add_argument("--denoise-conf", type=int, default=2)
    p.add_argument("--denoise-decay", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=DBSCAN_EPS)
    p.add_argument("--min-samples", type=int, default=DBSCAN_MIN_SAMPLES)
    p.add_argument("--min-points", type=int, default=CLUSTER_MIN_POINTS)
    p.add_argument("--min-volume", type=float, default=CLUSTER_MIN_VOLUME)
    p.add_argument("--edge-margin", type=float, default=EDGE_MARGIN)
    args = p.parse_args()

    run_live(
        config_dir=args.config, urdf_path=args.urdf,
        use_real_robot=args.real_robot, use_mock_camera=args.mock_camera,
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
    )


if __name__ == "__main__":
    main()
