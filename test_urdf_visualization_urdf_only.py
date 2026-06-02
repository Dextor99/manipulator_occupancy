"""RealSense 点云 + URDF 机器人模型实时可视化（独立程序）。"""
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


@dataclasses.dataclass
class _MockFrame:
    points_cam: np.ndarray
    timestamp: float = 0.0


class _MockReader:
    """生成工作空间内的随机点云用于测试 URDF 叠加。"""
    def __init__(self, n_points: int = 5000):
        self._n = n_points
        self._t = 0.0

    def read(self):
        self._t += 0.033
        # 随机点 + 一个缓慢移动的簇模拟物体
        n_bg = self._n - 200
        bg = np.random.uniform(-0.8, 0.8, (n_bg, 3))
        cx, cy, cz = 0.3 * math.cos(self._t * 0.5), 0.0, 0.3 + 0.2 * math.sin(self._t * 0.3)
        cluster = np.random.normal(0, 0.03, (200, 3)) + [cx, cy, cz]
        return _MockFrame(points_cam=np.vstack([bg, cluster]))

    def stop(self):
        pass


# ── helpers ───────────────────────────────────────────────────


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


def _scale_to_meters(mesh) -> None:
    """Scale mesh to meters if its bounding box suggests it's in mm."""
    extent = mesh.get_axis_aligned_bounding_box().get_extent()
    if np.linalg.norm(extent) > 10:
        mesh.scale(0.001, center=(0, 0, 0))


def _load_robot(urdf: URDFModel, vis, angles: dict[str, float]):
    """Load all link meshes, scale to meters, apply initial FK, add to visualizer.

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


# ── main ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='RealSense 点云 + URDF 机器人实时可视化')
    parser.add_argument('--config', default='config')
    parser.add_argument('--urdf', default='urdf/aubo_i16_gripper.urdf')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--mock-camera', action='store_true',
                        help='使用随机点云代替 RealSense（无需相机）')
    parser.add_argument('--real-robot', action='store_true',
                        help='从真实 AUBO 机器人读取关节角（需连接机器人）')
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

    # ── open3d visualizer ──
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Point Cloud + URDF Robot',
                      width=1024, height=768)

    for geom in _static_geometries():
        vis.add_geometry(geom)

    # point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
    pcd.paint_uniform_color([0.0, 0.5, 0.0])
    vis.add_geometry(pcd)

    # robot meshes
    zero = {n: 0.0 for n in movable}
    robot = _load_robot(urdf, vis, zero)
    print(f'Loaded {len(robot)} link meshes')

    # ── loop ──
    print('=== Running  (Q to exit) ===')
    frame = 0

    try:
        while vis.poll_events():
            t0 = time.perf_counter()

            # ---- point cloud ----
            frame_data = reader.read()
            if len(frame_data.points_cam):
                pts = transform_points(frame_data.points_cam, extrinsic)
                pts = crop_workspace(pts, workspace)
                pts = voxel_downsample(pts, workspace.get('voxel_size', 0.02))
                pcd.points = o3d.utility.Vector3dVector(pts)
                vis.update_geometry(pcd)

            # ---- robot FK ----
            angles = state_reader.get_joint_positions()
            fk = urdf.link_transforms(angles)
            for link_name, (mesh, cur_T) in list(robot.items()):
                vo = urdf.links[link_name]['visual_origin']
                new_T = fk.get(link_name, np.eye(4)) @ vo
                delta = new_T @ np.linalg.inv(cur_T)
                mesh.transform(delta)
                robot[link_name] = (mesh, new_T)
                vis.update_geometry(mesh)

            vis.update_renderer()

            elapsed = (time.perf_counter() - t0) * 1000
            frame += 1
            if frame % 30 == 0:
                print(f'[{frame:4d}]  {elapsed:.1f} ms/frame')

    finally:
        vis.destroy_window()
        reader.stop()
        if args.real_robot and hasattr(state_reader, 'disconnect'):
            state_reader.disconnect()


if __name__ == '__main__':
    main()

#python test_urdf_visualization_urdf_only.py --real-robot
