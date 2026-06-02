"""实时场景点云显示 — 自动去除机械臂本体点云。

对每一帧：
  1. 读取 RealSense 点云 → 变换到基坐标系 → 裁剪 → 体素降采样
  2. 读取关节角 → FK → 将 URDF 网格变换到世界坐标系
  3. 在网格表面均匀采样，计算场景点到网格表面的最近距离
  4. 移除距离 < threshold（默认 1 cm）的点（机械臂本体）
  5. 仅显示环境点云（不渲染机械臂模型）
  python test_remove_robot_points.py --real-robot
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import time
from pathlib import Path

import numpy as np

from calibration.transform_utils import load_transform_json, transform_points
from camera.pointcloud_preprocess import crop_workspace, voxel_downsample
from camera.realsense_pipeline_reader import RealSensePipelineReader
from robot.urdf_model import URDFModel
from robot.robot_state_reader import RealRobotStateReader, MockRobotStateReader
from utils.config import load_config_dir

# ── 默认参数 ────────────────────────────────────────────────────
ROBOT_REMOVAL_THRESHOLD = 0.03      # 1 cm
MESH_SAMPLE_POINTS = 50000           # 网格表面采样点数


# ── mock camera for testing ────────────────────────────────────

@dataclasses.dataclass
class _MockFrame:
    points_cam: np.ndarray
    timestamp: float = 0.0


class _MockReader:
    """生成工作空间内的随机点云用于测试。"""
    def __init__(self, n_points: int = 5000):
        self._n = n_points
        self._t = 0.0

    def read(self):
        self._t += 0.033
        n_bg = self._n - 200
        bg = np.random.uniform(-0.8, 0.8, (n_bg, 3))
        cx, cy, cz = 0.3 * math.cos(self._t * 0.5), 0.0, 0.3 + 0.2 * math.sin(self._t * 0.3)
        cluster = np.random.normal(0, 0.03, (200, 3)) + [cx, cy, cz]
        return _MockFrame(points_cam=np.vstack([bg, cluster]))

    def stop(self):
        pass


# ── scene helpers ──────────────────────────────────────────────

def _static_geometries():
    """坐标系指示：原点和线框球（辅助观察）。"""
    import open3d as o3d

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([0.25, 0.25, 0.70])
    wire = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    wire.paint_uniform_color([0.25, 0.25, 0.70])

    origin = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    origin.paint_uniform_color([1.0, 0.0, 0.0])

    return [origin, wire]


# ── 网格加载与采样 ─────────────────────────────────────────────

def _scale_to_meters(mesh) -> None:
    """如果包围盒尺寸 > 10 则视为毫米单位，缩放到米。"""
    extent = mesh.get_axis_aligned_bounding_box().get_extent()
    if np.linalg.norm(extent) > 10:
        mesh.scale(0.001, center=(0, 0, 0))


def load_robot_meshes(urdf: URDFModel) -> dict[str, tuple]:
    """加载所有 link 网格并缩放到米制。

    Returns
    -------
    {link_name: (mesh_local, identity_4x4)}  — 本地坐标网格 + 占位变换。
    """
    import open3d as o3d

    meshes: dict[str, tuple] = {}
    for link_name in urdf.links:
        mesh_path = urdf.resolve_mesh(link_name)
        if mesh_path is None:
            continue
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if not mesh.has_triangles():
            continue
        _scale_to_meters(mesh)
        mesh.compute_vertex_normals()
        meshes[link_name] = (mesh, np.eye(4))

    return meshes


def sample_robot_surface(meshes: dict[str, tuple],
                         fk: dict[str, np.ndarray],
                         urdf: URDFModel,
                         n_total: int = MESH_SAMPLE_POINTS) -> np.ndarray:
    """在所有 link 网格表面均匀采样，返回世界坐标系下的 (N, 3) 点云。"""
    import copy
    import open3d as o3d

    if not meshes:
        return np.empty((0, 3))

    n_links = len(meshes)
    pts_per_link = max(n_total // n_links, 100)

    all_pts = []
    for link_name, (mesh, _) in meshes.items():
        T = fk.get(link_name, np.eye(4))
        vo = urdf.links[link_name]['visual_origin']
        mesh_world = copy.deepcopy(mesh)
        mesh_world.transform(T @ vo)
        sampled = mesh_world.sample_points_uniformly(number_of_points=pts_per_link)
        pts = np.asarray(sampled.points)
        if len(pts):
            all_pts.append(pts)

    if not all_pts:
        return np.empty((0, 3))
    return np.vstack(all_pts)


# ── main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='场景点云可视化 — 自动去除机械臂本体点云')
    parser.add_argument('--config', default='config')
    parser.add_argument('--urdf', default='urdf/aubo_i16_gripper.urdf')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--threshold', type=float, default=ROBOT_REMOVAL_THRESHOLD,
                        help=f'去除阈值 (米，默认 {ROBOT_REMOVAL_THRESHOLD})')
    parser.add_argument('--mesh-samples', type=int, default=MESH_SAMPLE_POINTS,
                        help=f'网格采样点数 (默认 {MESH_SAMPLE_POINTS})')
    parser.add_argument('--mock-camera', action='store_true',
                        help='使用随机点云代替 RealSense（无需相机）')
    parser.add_argument('--real-robot', action='store_true',
                        help='从真实 AUBO 机器人读取关节角')
    args = parser.parse_args()

    threshold = args.threshold
    n_mesh_samples = args.mesh_samples

    # ── 配置 ──
    config = load_config_dir(args.config)
    extrinsic = load_transform_json(Path(args.config) / 'camera_extrinsic.json')
    workspace = config['workspace']

    # ── robot state reader ──
    if args.real_robot:
        state_reader = RealRobotStateReader()
        if not state_reader.connect():
            print('Failed to connect to real robot, falling back to mock')
            state_reader = MockRobotStateReader()
    else:
        state_reader = MockRobotStateReader()
    print(f'Using robot state reader: {type(state_reader).__name__}')

    # ── URDF ──
    urdf = URDFModel(args.urdf)
    movable = urdf.movable_joints()
    print(f'URDF  joints={len(urdf.joints)}  movable={movable}')

    # ── camera ──
    if args.mock_camera:
        reader = _MockReader()
        print('Using mock camera (random point cloud)')
    else:
        reader = RealSensePipelineReader(width=args.width, height=args.height)

    # ── load robot meshes ──
    import open3d as o3d
    robot_meshes = load_robot_meshes(urdf)
    print(f'Loaded {len(robot_meshes)} link meshes for robot point removal')

    # ── open3d visualiser ──
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Scene Point Cloud (Robot Points Removed)',
                      width=1024, height=768)

    for geom in _static_geometries():
        vis.add_geometry(geom)

    # 仅显示环境点云（机械臂点已去除）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
    pcd.paint_uniform_color([0.0, 0.5, 0.0])   # 绿色 — 环境点云
    vis.add_geometry(pcd)

    # ── main loop ──
    print('=== Running (Q to exit) ===')
    frame = 0

    # 复用 PointCloud 对象避免每帧构造
    robot_pcd = o3d.geometry.PointCloud()
    scene_pcd = o3d.geometry.PointCloud()

    try:
        while vis.poll_events():
            t0 = time.perf_counter()

            # 1. 读取相机帧并预处理
            frame_data = reader.read()
            if len(frame_data.points_cam):
                pts = transform_points(frame_data.points_cam, extrinsic)
                pts = crop_workspace(pts, workspace)
                pts = voxel_downsample(pts, workspace.get('voxel_size', 0.02))
            else:
                pts = np.empty((0, 3))

            # 2. 读取关节角 → FK → 机械臂网格表面采样 → 距离过滤
            if len(pts) > 0:
                angles = state_reader.get_joint_positions()
                fk = urdf.link_transforms(angles)

                robot_pts = sample_robot_surface(robot_meshes, fk, urdf, n_mesh_samples)

                if len(robot_pts) > 0:
                    # 计算每个场景点到最近机械臂表面的距离
                    robot_pcd.points = o3d.utility.Vector3dVector(robot_pts)
                    scene_pcd.points = o3d.utility.Vector3dVector(pts)
                    dists = np.asarray(scene_pcd.compute_point_cloud_distance(robot_pcd))

                    # 保留距离 > threshold 的点（环境点），移除距离近的点（机械臂本体）
                    keep = dists > threshold
                    pts = pts[keep]

            # 3. 更新可视化
            pcd.points = o3d.utility.Vector3dVector(pts)
            vis.update_geometry(pcd)
            vis.update_renderer()

            elapsed = (time.perf_counter() - t0) * 1000
            frame += 1
            if frame % 30 == 0:
                print(f'[{frame:4d}]  {elapsed:.1f} ms/frame  '
                      f'scene_pts={len(pts)}  '
                      f'threshold={threshold:.3f}m')

    finally:
        vis.destroy_window()
        reader.stop()
        if args.real_robot and hasattr(state_reader, 'disconnect'):
            state_reader.disconnect()


if __name__ == '__main__':
    main()
