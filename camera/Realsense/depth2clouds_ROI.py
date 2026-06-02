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


def depth_to_pointcloud(depth_img, fx, fy, cx, cy, depth_scale=1.0, max_depth=1.0, detection_size=600):
    """
    将深度图转换为点云（只处理检测区域内点）
    参数:
        depth_img: 深度图数据
        fx, fy: 相机焦距
        cx, cy: 相机光心
        depth_scale: 深度值缩放比例
        max_depth: 最大有效深度（米）
        detection_size: 检测区域大小（像素）
    """
    # 转换为米制单位
    depth = depth_img.astype(np.float32) * depth_scale

    height, width = depth.shape
    center_x, center_y = width // 2, height // 2

    # 计算检测区域边界
    half_size = detection_size // 2
    x_start = max(0, center_x - half_size)
    x_end = min(width, center_x + half_size)
    y_start = max(0, center_y - half_size)
    y_end = min(height, center_y + half_size)

    # 创建检测区域内的网格
    u = np.arange(x_start, x_end)
    v = np.arange(y_start, y_end)
    u, v = np.meshgrid(u, v)

    # 提取检测区域深度
    roi_depth = depth[y_start:y_end, x_start:x_end]

    # 计算3D坐标
    z = roi_depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 复合过滤条件（有效深度范围：0 < z < max_depth）
    valid_mask = (z > 0.4) & ~np.isnan(z) & (z < max_depth)

    return np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=-1)


def visualize_pointcloud(points):
    """可视化点云（带坐标系）"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 添加坐标系（红色X，绿色Y，蓝色Z）
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([pcd, coord_frame],
                                      window_name="Detection Area Point Cloud",
                                      width=800,
                                      height=600)


if __name__ == "__main__":
    # 文件路径配置
    depth_path = "PIC/camera_capture_0709/depth/000000.png"  # 深度图路径
    camera_json_path = "camera0.json"  # 相机参数文件

    # 检查文件是否存在
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"深度图文件不存在: {depth_path}")
    if not os.path.exists(camera_json_path):
        raise FileNotFoundError(f"相机参数文件不存在: {camera_json_path}")

    # 加载相机参数
    fx, fy, cx, cy, depth_scale = load_camera_params(camera_json_path)
    print(f"相机参数: fx={fx}, fy={fy}, cx={cx}, cy={cy}, depth_scale={depth_scale}")

    # 读取深度图（16位单通道）
    depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if depth_img is None:
        raise ValueError("无法加载深度图")

    # 生成点云（只处理检测区域）
    print("正在从检测区域生成点云...")
    points = depth_to_pointcloud(depth_img, fx, fy, cx, cy, depth_scale,
                                 detection_size=600)  # 与主程序相同的检测区域大小

    print(f"生成点云完成，包含 {len(points)} 个点")

    # 可视化
    print("正在显示点云...")
    visualize_pointcloud(points)