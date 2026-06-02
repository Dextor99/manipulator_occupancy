"""
将 TIFF 深度图转换为点云并用 Open3D 可视化

用法：
    python visual_tiff_in.py                              # 文件选择框
    python visual_tiff_in.py --depth ../../captured/capture_20260526_203612/depth/000001.tiff
    python visual_tiff_in.py --depth depth.tiff --rgb color.png
"""

import argparse
import sys
import numpy as np
import cv2
import open3d as o3d

try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    tk = None

# D435I 1280x720 内参（程序启动时读取的实际值）
CAMERA_MATRIX = np.array([
    [918.27160645, 0., 643.14483643],
    [0., 918.02313232, 357.28491211],
    [0., 0., 1.]
], dtype=np.float64)


def load_depth_tiff(path):
    """读取 TIFF 深度图，统一返回 float32 单位米"""
    depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise ValueError(f"无法读取: {path}")

    if depth.dtype == np.uint16:
        print("  格式: uint16 (mm) → 转换为米")
        depth = depth.astype(np.float32) / 1000.0
    elif depth.dtype == np.float32:
        print("  格式: float32 (m)")
    else:
        depth = depth.astype(np.float32)

    print(f"  尺寸: {depth.shape}, 范围: {depth.min():.3f}–{depth.max():.3f} m")
    return depth


def depth_to_pcd(depth_image, intrinsic, rgb_image=None, max_depth=3.0, step=2):
    """
    深度图 → Open3D 点云

    参数:
        depth_image: (H, W) float32，单位米
        intrinsic: 内参矩阵 3x3
        rgb_image: (H, W, 3) BGR 彩色图（可选）
        max_depth: 最大有效深度（米）
        step: 采样步长（>1 降采样）

    返回:
        open3d.geometry.PointCloud
    """
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    # 有效深度掩码 + 下采样
    mask = (depth_image > 0) & (depth_image < max_depth)
    mask[::step, :] = False
    mask[:, ::step] = False

    v_valid, u_valid = np.where(mask)
    z_valid = depth_image[v_valid, u_valid]

    # 反投影
    x = (u_valid.astype(np.float32) - cx) * z_valid / fx
    y = (v_valid.astype(np.float32) - cy) * z_valid / fy
    points = np.stack((x, y, z_valid), axis=-1)

    print(f"  点云点数: {len(points)}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    # 上色
    if rgb_image is not None:
        rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        colors = rgb[v_valid, u_valid].astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # 深度伪彩
        z_norm = (z_valid - z_valid.min()) / (z_valid.max() - z_valid.min() + 1e-8)
        cmap = np.array([[0.1, 0.1, 0.7], [0.1, 0.7, 0.1],
                         [0.7, 0.7, 0.1], [0.7, 0.1, 0.1]])
        idx = (z_norm * (cmap.shape[0] - 1)).astype(np.int32)
        colors = cmap[idx]
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def visualize_pcd(pcd, title="Depth Point Cloud"):
    """用 Open3D 可视化点云"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1024, height=768)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.12, 0.12, 0.12])

    ctrl = vis.get_view_control()
    ctrl.set_front([0.0, -0.3, -1.0])
    ctrl.set_lookat(pcd.get_center())
    ctrl.set_up([0.0, -1.0, 0.0])
    ctrl.set_zoom(0.6)

    print("鼠标拖拽旋转 · 滚轮缩放 · ESC 退出")
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="TIFF 深度图 → 点云可视化")
    parser.add_argument("--depth", default=None, help="深度图路径 (.tiff)")
    parser.add_argument("--rgb", default=None, help="彩色图路径 (.png)，可选")
    parser.add_argument("--fx", type=float, default=CAMERA_MATRIX[0, 0], help="fx")
    parser.add_argument("--fy", type=float, default=CAMERA_MATRIX[1, 1], help="fy")
    parser.add_argument("--cx", type=float, default=CAMERA_MATRIX[0, 2], help="cx")
    parser.add_argument("--cy", type=float, default=CAMERA_MATRIX[1, 2], help="cy")
    parser.add_argument("--max-depth", type=float, default=3.0, help="最大深度 (米)")
    parser.add_argument("--step", type=int, default=2, help="采样步长")
    args = parser.parse_args()

    # 选择深度文件
    depth_path = args.depth
    if not depth_path and tk:
        root = tk.Tk()
        root.withdraw()
        depth_path = filedialog.askopenfilename(
            title="选择深度 TIFF 文件",
            filetypes=[("TIFF", "*.tiff *.tif"), ("All", "*.*")]
        )
        if not depth_path:
            print("未选择文件")
            sys.exit(1)
    elif not depth_path:
        print("请指定 --depth 或安装 tkinter 以使用文件选择框")
        sys.exit(1)

    # 加载
    print(f"深度图: {depth_path}")
    depth = load_depth_tiff(depth_path)

    # 加载彩色图
    rgb = None
    if args.rgb:
        rgb = cv2.imread(args.rgb)
        if rgb is None:
            print(f"警告: 无法读取 {args.rgb}")
        else:
            if rgb.shape[:2] != depth.shape[:2]:
                rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))
            print(f"彩色图: {args.rgb} {rgb.shape}")

    # 内参
    intrinsic = np.array([
        [args.fx, 0, args.cx],
        [0, args.fy, args.cy],
        [0, 0, 1]
    ], dtype=np.float64)
    print(f"相机内参:\n{intrinsic}")

    # 转点云
    print("生成点云...")
    pcd = depth_to_pcd(depth, intrinsic, rgb, max_depth=args.max_depth, step=args.step)

    # 可视化
    visualize_pcd(pcd)


if __name__ == "__main__":
    main()
