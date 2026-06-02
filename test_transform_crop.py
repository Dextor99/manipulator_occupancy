"""RealSense 点云 → 基坐标系变换 → 球域切割 → 实时可视化"""
import argparse
from pathlib import Path

import numpy as np

from calibration.transform_utils import load_transform_json, transform_points
from camera.pointcloud_preprocess import crop_workspace, voxel_downsample
from camera.realsense_pipeline_reader import RealSensePipelineReader
from utils.config import load_config_dir


def build_scene_geometries():
    """创建静态场景元素（坐标系原点、球体线框），返回列表。"""
    import open3d as o3d

    origin = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    origin.paint_uniform_color([1.0, 0.0, 0.0])

    sphere_frame = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    sphere_frame.paint_uniform_color([0.3, 0.3, 0.8])
    sphere_frame.compute_vertex_normals()
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(sphere_frame)
    wireframe.paint_uniform_color([0.3, 0.3, 0.8])

    return [origin, wireframe]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    config = load_config_dir(args.config)
    extrinsic = load_transform_json(Path(args.config) / "camera_extrinsic.json")
    workspace = config["workspace"]

    reader = RealSensePipelineReader(width=args.width, height=args.height)

    import open3d as o3d

    # 初始化可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Base Frame - 1m Sphere Workspace", width=1024, height=768)

    # 添加静态场景
    for geom in build_scene_geometries():
        vis.add_geometry(geom)

    # 点云对象（只更新 points）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
    pcd.paint_uniform_color([0.0, 0.5, 0.0])
    vis.add_geometry(pcd)

    print("=== 实时可视化（点击窗口按 Q 退出）===")
    frame_count = 0
    try:
        while vis.poll_events():
            frame = reader.read()
            n_raw = len(frame.points_cam)
            if n_raw == 0:
                continue

            # 1. 相机坐标系 → 基坐标系
            points_base = transform_points(frame.points_cam, extrinsic)

            # 2. 工作空间切割（AABB + 球形）
            cropped = crop_workspace(points_base, workspace)
            downsampled = voxel_downsample(cropped, workspace.get("voxel_size", 0.02))

            # 更新点云
            pcd.points = o3d.utility.Vector3dVector(downsampled)
            vis.update_geometry(pcd)
            vis.update_renderer()

            frame_count += 1
            if frame_count % 30 == 0:
                n_base = len(points_base)
                n_final = len(downsampled)
                print(f"[{frame_count:4d}]  cam: {n_raw:6d}  base: {n_base:6d}  保留: {n_final:6d}")

    finally:
        vis.destroy_window()
        reader.stop()


if __name__ == "__main__":
    main()
