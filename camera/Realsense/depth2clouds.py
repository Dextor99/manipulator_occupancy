import cv2
import numpy as np
import open3d as o3d
import json
import os


def load_camera_params(json_path):
    """从JSON文件加载相机参数"""
    with open(json_path, 'r') as f:
        params = json.load(f)

    camera_matrix = np.array(params['camera_matrix'])
    dist_coeffs = np.array(params['discoeffs'])
    depth_scale = params.get('depth_scale', 1.0)  # 默认1.0（米制单位）

    # 提取内参
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    return fx, fy, cx, cy, depth_scale


def depth_to_pointcloud(depth_path, camera_json_path):
    """
    将深度图转换为点云（无RGB信息）
    参数:
        depth_path: 深度图路径（PNG格式，16位）
        camera_json_path: 相机参数JSON文件路径
    """
    # 加载相机参数
    fx, fy, cx, cy, depth_scale = load_camera_params(camera_json_path)

    # 读取深度图（16位单通道）
    depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if depth_img is None:
        raise ValueError(f"无法加载深度图: {depth_path}")

    # 转换为米制单位（假设原始深度图单位与depth_scale一致）
    depth = depth_img.astype(np.float32) * depth_scale

    # 创建点云网格
    height, width = depth.shape
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)

    # 计算3D坐标（Z方向为深度）
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 过滤无效点（深度<=0或NaN）
    valid_mask = (z > 0.4) & ~np.isnan(z) & (z < 1)
    points = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=-1)

    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd


def visualize_pointcloud(pcd):
    """可视化点云（带坐标系）"""
    # 添加坐标系参考（红色X，绿色Y，蓝色Z）
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0])

    o3d.visualization.draw_geometries(
        [pcd, coord_frame],
        window_name="Depth Point Cloud",
        width=800,
        height=600,
        zoom=0.5
    )


def save_pointcloud(pcd, output_path="pointcloud.ply"):
    """保存点云为PLY文件"""
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"点云已保存至: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    # 文件路径配置
    depth_image_path = "PIC/camera_capture_0709/depth/000000.png"  # 深度图路径
    # depth_image_path = "../../GGCNN/out/label1/0000_label1_perfect_depth.tiff"  # 深度图路径
    camera_json_path = "camera.json"  # 相机参数文件

    # 检查文件是否存在
    if not os.path.exists(depth_image_path):
        raise FileNotFoundError(f"深度图文件不存在: {depth_image_path}")
    if not os.path.exists(camera_json_path):
        raise FileNotFoundError(f"相机参数文件不存在: {camera_json_path}")

    # 生成点云
    print("正在从深度图生成点云...")
    pcd = depth_to_pointcloud(depth_image_path, camera_json_path)
    print(f"生成点云完成，包含 {len(pcd.points)} 个点")

    # 可视化
    print("正在显示点云...")
    visualize_pointcloud(pcd)

    # 保存点云（可选）
    # save_pointcloud(pcd)