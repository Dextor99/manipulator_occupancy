"""仅 URDF 机器人模型动画（无需 RealSense 相机），用于验证 FK 和渲染。"""
from __future__ import annotations

import argparse

import numpy as np

from robot.urdf_model import URDFModel
from robot.robot_state_reader import RealRobotStateReader, MockRobotStateReader


def _scale_to_meters(mesh) -> None:
    """Scale mesh to meters if its bounding box suggests it's in mm."""
    extent = mesh.get_axis_aligned_bounding_box().get_extent()
    if np.linalg.norm(extent) > 10:
        mesh.scale(0.001, center=(0, 0, 0))


def main():
    parser = argparse.ArgumentParser(description='URDF 机器人模型动画')
    parser.add_argument('--real-robot', action='store_true',
                        help='从真实 AUBO 机器人读取关节角（需连接机器人）')
    args = parser.parse_args()

    import open3d as o3d

    # ── robot state reader ──
    if args.real_robot:
        reader = RealRobotStateReader()
        if not reader.connect():
            print('Failed to connect to real robot, falling back to mock')
            reader = MockRobotStateReader()
    else:
        reader = MockRobotStateReader()
    print(f'Using robot state reader: {type(reader).__name__}')

    urdf = URDFModel('urdf/aubo_i16_gripper.urdf')
    movable = urdf.movable_joints()
    print(f'URDF joints={len(urdf.joints)}  movable={len(movable)}')

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='URDF Robot Animation', width=1024, height=768)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(axis)

    zero = {n: 0.0 for n in movable}
    fk0 = urdf.link_transforms(zero)
    meshes: dict[str, tuple] = {}

    for link_name in urdf.links:
        p = urdf.resolve_mesh(link_name)
        if p is None:
            continue
        m = o3d.io.read_triangle_mesh(p)
        if not m.has_triangles():
            continue
        _scale_to_meters(m)
        m.compute_vertex_normals()
        vo = urdf.links[link_name]['visual_origin']
        T = fk0.get(link_name, np.eye(4)) @ vo
        m.transform(T)
        vis.add_geometry(m)
        meshes[link_name] = (m, T)

    print(f'Loaded {len(meshes)} meshes')

    # print FK origin positions
    print('FK at zero:')
    for ln, T in sorted(fk0.items()):
        print(f'  {ln:25s}  {T[:3,3]}')

    frame = 0
    try:
        while vis.poll_events():
            angles = reader.get_joint_positions()
            fk = urdf.link_transforms(angles)

            for link_name, (m, cur_T) in list(meshes.items()):
                vo = urdf.links[link_name]['visual_origin']
                new_T = fk.get(link_name, np.eye(4)) @ vo
                delta = new_T @ np.linalg.inv(cur_T)
                m.transform(delta)
                meshes[link_name] = (m, new_T)
                vis.update_geometry(m)

            vis.update_renderer()
            frame += 1
            if frame % 30 == 0:
                wrist3_pos = fk.get('wrist3_Link', np.eye(4))[:3, 3]
                print(f'[{frame:4d}]  wrist3={wrist3_pos}')

    finally:
        vis.destroy_window()
        if args.real_robot and hasattr(reader, 'disconnect'):
            reader.disconnect()


if __name__ == '__main__':
    main()
