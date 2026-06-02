"""RealSense 点云 + 机器人模型实时可视化（同步版）。

启动时通过 --mode 指定显示模式：
  mesh       — URDF 三角网格渲染
  simplified — 胶囊体/球体简化模型
  none       — 仅显示场景点云

全同步管线：每个线程读取相机帧、处理点云、读取关节角、FK 更新，
保证点云和机械臂模型在同一帧内对齐。
python test_urdf_visualization.py --real-robot
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
    import open3d as o3d

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([0.25, 0.25, 0.70])
    wire = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    wire.paint_uniform_color([0.25, 0.25, 0.70])

    origin = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    origin.paint_uniform_color([1.0, 0.0, 0.0])

    return [origin, wire]


# ── capsule / sphere mesh builders ─────────────────────────────

def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3×3 rotation that maps unit vector *a* to unit vector *b* (Rodrigues)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)


def _capsule_mesh(a: np.ndarray, b: np.ndarray, radius: float,
                  color: tuple[float, float, float],
                  segments: int = 16) -> 'o3d.geometry.TriangleMesh':
    """Create a capsule (cylinder + hemisphere) mesh from *a* to *b*."""
    import open3d as o3d

    a, b = np.asarray(a), np.asarray(b)
    v = b - a
    length = np.linalg.norm(v)

    if length < 1e-8:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=segments)
        sphere.translate(a)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        return sphere

    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius, length,
                                                     segments, segments * 2)
    R = _rotation_between(np.array([0.0, 0.0, 1.0]), v / length)
    cyl.rotate(R, center=(0, 0, 0))
    cyl.translate((a + b) / 2)

    sa = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=segments)
    sa.translate(a)
    sb = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=segments)
    sb.translate(b)

    merged = cyl + sa + sb
    merged.paint_uniform_color(color)
    merged.compute_vertex_normals()
    return merged


def _sphere_mesh(center: np.ndarray, radius: float,
                 color: tuple[float, float, float],
                 segments: int = 16) -> 'o3d.geometry.TriangleMesh':
    """Create a sphere mesh at *center*."""
    import open3d as o3d

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=segments)
    sphere.translate(np.asarray(center))
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    return sphere


# ── simplified robot model ─────────────────────────────────────
#
# Each capsule is defined in a link's LOCAL frame.  Meshes are created ONCE at
# startup in local coordinates and transformed each frame via FK delta.

CAPSULE_SEGMENTS = [
    # (name, link_name, a_local, b_local, radius, color)
    ("base",      "base_link",       [0, 0, 0],     [0, 0, 0.163],    0.09,  (0.85, 0.35, 0.30)),
    ("shoulder",  "shoulder_Link",   [0, 0, 0],     [0, 0.191, 0],    0.085, (0.80, 0.40, 0.30)),
    ("upper_arm", "upperArm_Link",   [0, 0, 0],     [0.48008, 0, 0],  0.075, (0.75, 0.45, 0.30)),
    ("forearm",   "foreArm_Link",    [0, 0, 0],     [0.36992, 0, 0],  0.065, (0.70, 0.50, 0.30)),
    ("wrist1",    "wrist1_Link",     [0, 0, 0],     [0, 0.1175, 0],   0.055, (0.65, 0.55, 0.30)),
    ("wrist2",    "wrist2_Link",     [0, 0, 0],     [0, -0.1035, 0],  0.05,  (0.60, 0.55, 0.35)),
    ("wrist3",    "wrist3_Link",     [0, 0, 0],     [0, 0, 0.07],     0.04,  (0.55, 0.55, 0.40)),
]

GRIPPER_SPHERES = [
    # (name, link_name, center_local, radius, color)
    ("grip_base",  "gripper_base_link", [0, 0, 0],       0.035, (0.50, 0.50, 0.50)),
    ("left_finger",  "left_link",       [0, 0, 0.025],   0.020, (0.50, 0.50, 0.50)),
    ("right_finger", "right_link",      [0, 0, 0.025],   0.020, (0.50, 0.50, 0.50)),
]


def _build_simple_robot(fk: dict[str, np.ndarray]) -> list:
    """Create simplified meshes in local frames, transform to world via FK.

    Returns list of [mesh, link_name, current_4x4_world_T].
    """
    items: list = []

    for _name, link_name, a_loc, b_loc, radius, color in CAPSULE_SEGMENTS:
        T = fk.get(link_name)
        if T is None:
            continue
        mesh = _capsule_mesh(np.array(a_loc), np.array(b_loc), radius, color)
        mesh.transform(T)
        items.append([mesh, link_name, T])

    for _name, link_name, c_loc, radius, color in GRIPPER_SPHERES:
        T = fk.get(link_name)
        if T is None:
            continue
        mesh = _sphere_mesh(np.array(c_loc), radius, color)
        mesh.transform(T)
        items.append([mesh, link_name, T])

    return items


def _update_simple_robot(items: list, fk: dict[str, np.ndarray], vis) -> None:
    """Transform all simplified meshes to current FK (delta-transform)."""
    for entry in items:
        mesh, link_name, cur_T = entry
        new_T = fk.get(link_name, np.eye(4))
        delta = new_T @ np.linalg.inv(cur_T)
        mesh.transform(delta)
        entry[2] = new_T
        vis.update_geometry(mesh)


# ── URDF mesh loader ──────────────────────────────────────────

def _scale_to_meters(mesh) -> None:
    """Scale mesh to metres if its bounding box suggests it's in mm."""
    extent = mesh.get_axis_aligned_bounding_box().get_extent()
    if np.linalg.norm(extent) > 10:
        mesh.scale(0.001, center=(0, 0, 0))


def _load_robot_meshes(urdf: URDFModel, vis, angles: dict[str, float]):
    """Load all link meshes, scale to metres, apply initial FK, add to viewer.

    Returns {link_name: (mesh, current_4x4_transform)}.
    """
    import open3d as o3d

    fk = urdf.link_transforms(angles)
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

        vo = urdf.links[link_name]['visual_origin']
        T = fk.get(link_name, np.eye(4)) @ vo
        mesh.transform(T)
        vis.add_geometry(mesh)
        meshes[link_name] = (mesh, T)

    return meshes


# ── main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='RealSense 点云 + 机器人模型实时可视化')
    parser.add_argument('--config', default='config')
    parser.add_argument('--urdf', default='urdf/aubo_i16_gripper.urdf')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--mock-camera', action='store_true',
                        help='使用随机点云代替 RealSense（无需相机）')
    parser.add_argument('--real-robot', action='store_true',
                        help='从真实 AUBO 机器人读取关节角（需连接机器人）')
    parser.add_argument('--mode', default='mesh',
                        choices=['mesh', 'none', 'simplified'],
                        help='显示模式 (默认 mesh)')
    args = parser.parse_args()

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

    # ── open3d visualiser ──
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    mode_label = {'mesh': 'Mesh', 'none': 'None', 'simplified': 'Capsule'}[args.mode]
    vis.create_window(window_name=f'Point Cloud + Robot [{mode_label}]',
                      width=1024, height=768)

    for geom in _static_geometries():
        vis.add_geometry(geom)

    # point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
    pcd.paint_uniform_color([0.0, 0.5, 0.0])
    vis.add_geometry(pcd)

    # ── robot geometry (only the selected mode) ──
    zero = {n: 0.0 for n in movable}
    robot_meshes = None
    simple_items = None

    if args.mode == 'mesh':
        robot_meshes = _load_robot_meshes(urdf, vis, zero)
        print(f'Loaded {len(robot_meshes)} link meshes (mesh mode)')
    elif args.mode == 'simplified':
        fk_zero = urdf.link_transforms(zero)
        simple_items = _build_simple_robot(fk_zero)
        for entry in simple_items:
            vis.add_geometry(entry[0])
        print(f'Simplified robot: {len(simple_items)} capsules/spheres')

    # ── loop (全同步：点云与机械臂数据同帧) ──
    print('=== Running (Q to exit) ===')
    frame = 0

    try:
        while vis.poll_events():
            t0 = time.perf_counter()

            # 1. 读取相机帧并处理点云
            frame_data = reader.read()
            if len(frame_data.points_cam):
                pts = transform_points(frame_data.points_cam, extrinsic)
                pts = crop_workspace(pts, workspace)
                pts = voxel_downsample(pts, workspace.get('voxel_size', 0.02))
                pcd.points = o3d.utility.Vector3dVector(pts)
                vis.update_geometry(pcd)

            # 2. 读取关节角 → FK → 更新机械臂（与点云同一时刻）
            angles = state_reader.get_joint_positions()
            fk = urdf.link_transforms(angles)

            if args.mode == 'mesh' and robot_meshes is not None:
                for link_name, (mesh, cur_T) in list(robot_meshes.items()):
                    vo = urdf.links[link_name]['visual_origin']
                    new_T = fk.get(link_name, np.eye(4)) @ vo
                    delta = new_T @ np.linalg.inv(cur_T)
                    mesh.transform(delta)
                    robot_meshes[link_name] = (mesh, new_T)
                    if frame % 2 == 0:
                        vis.update_geometry(mesh)
            elif args.mode == 'simplified' and simple_items is not None:
                if frame % 2 == 0:
                    _update_simple_robot(simple_items, fk, vis)

            vis.update_renderer()

            elapsed = (time.perf_counter() - t0) * 1000
            frame += 1
            if frame % 30 == 0:
                print(f'[{frame:4d}]  {elapsed:.1f} ms/frame  mode={args.mode}')

    finally:
        vis.destroy_window()
        reader.stop()
        if args.real_robot and hasattr(state_reader, 'disconnect'):
            state_reader.disconnect()


if __name__ == '__main__':
    main()
