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
    depth_scale = params.get('depth_scale', 1.0)  # 默认1.0（米制单位）

    # 提取内参
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    return fx, fy, cx, cy, depth_scale


def masked_depth_to_pointcloud(depth_img, mask_img, fx, fy, cx, cy, depth_scale=1.0, max_depth=1.0):
    """
    将深度图转换为点云（只处理掩码区域内点）
    参数:
        depth_img: 深度图数据
        mask_img: 掩码图像（物体区域为255）
        fx, fy: 相机焦距
        cx, cy: 相机光心
        depth_scale: 深度值缩放比例
        max_depth: 最大有效深度（米）
    返回:
        点云坐标数组 (N,3)
    """
    # 转换为米制单位
    depth = depth_img.astype(np.float32) * depth_scale

    # 确保掩码是二值图像
    if len(mask_img.shape) == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

    # 获取掩码区域内的像素坐标
    y_coords, x_coords = np.where(mask == 255)

    # 计算3D坐标
    z = depth[y_coords, x_coords]
    x = (x_coords - cx) * z / fx
    y = (y_coords - cy) * z / fy

    # 过滤无效点
    valid_mask = (z > 0.4) & ~np.isnan(z) & (z < max_depth)

    return np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=-1)


def save_pointcloud(points, output_dir="pointclouds", filename="object_pointcloud.ply"):
    """保存点云到PLY文件"""
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 保存为PLY文件
    output_path = os.path.join(output_dir, filename)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"点云已保存至: {output_path}")
def visualize_pointcloud(points):
    """可视化点云（带坐标系）"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 添加坐标系（红色X，绿色Y，蓝色Z）
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([pcd, coord_frame],
                                      window_name="Masked Object Point Cloud",
                                      width=800,
                                      height=600)


if __name__ == "__main__":
    # 文件路径配置
    record_dir = "record/sanjiaoguan"
    pointclouds_dir = "pointclouds"
    depth_path = os.path.join(record_dir, "grasp_point_Depth.png")
    mask_path = os.path.join(record_dir, "object_mask.png")
    camera_json_path = "camera.json"

    # 检查文件是否存在
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"深度图文件不存在: {depth_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"掩码图像不存在: {mask_path}")
    if not os.path.exists(camera_json_path):
        raise FileNotFoundError(f"相机参数文件不存在: {camera_json_path}")

    # 加载相机参数
    fx, fy, cx, cy, depth_scale = load_camera_params(camera_json_path)
    print(f"相机参数: fx={fx}, fy={fy}, cx={cx}, cy={cy}, depth_scale={depth_scale}")

    # 读取深度图和掩码
    depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    mask_img = cv2.imread(mask_path)

    if depth_img is None:
        raise ValueError("无法加载深度图")
    if mask_img is None:
        raise ValueError("无法加载掩码图像")

    # 生成点云（只处理掩码区域）
    print("正在从掩码区域生成点云...")
    points = masked_depth_to_pointcloud(
        depth_img, mask_img, fx, fy, cx, cy,
        depth_scale, max_depth=1.0
    )

    print(f"生成点云完成，包含 {len(points)} 个点")
    # 保存点云
    save_pointcloud(points, pointclouds_dir)
    # 可视化
    print("正在显示物体点云...")
    visualize_pointcloud(points)