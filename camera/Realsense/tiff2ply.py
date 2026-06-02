"""
将 TIFF 深度图转换为点云并保存为 PLY 文件

用法：
    python tiff2ply.py                                  # 弹出文件选择框
    python tiff2ply.py --depth captured/xxx/depth/000000.tiff
    python tiff2ply.py --depth depth.tiff --rgb color.png --save out.ply
"""

import argparse
import os
import sys
import numpy as np
import cv2

try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    tk = None


def depth_to_pointcloud_ply(depth_image, intrinsic, rgb_image=None, max_depth=3.0, step=2):
    """
    将深度图转换为点云并返回 PLY 格式文本。

    参数:
        depth_image: (H, W) float32，单位米
        intrinsic: 相机内参矩阵 3x3
        rgb_image: (H, W, 3) BGR 彩色图（可选）
        max_depth: 最大深度阈值（米），超出滤除
        step: 采样步长，>1 可降采样

    返回:
        vertices: (N, 3) float32 点云坐标
        colors: (N, 3) uint8 颜色
    """
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    h, w = depth_image.shape

    # 有效掩码：深度在范围内
    mask = (depth_image > 0) & (depth_image < max_depth)

    # 按步长下采样
    mask[::step, :] = False
    mask[:, ::step] = False

    # 提取有效像素坐标和深度
    v_valid, u_valid = np.where(mask)
    z_valid = depth_image[v_valid, u_valid]

    # 反投影到 3D
    x = (u_valid.astype(np.float32) - cx) * z_valid / fx
    y = (v_valid.astype(np.float32) - cy) * z_valid / fy
    vertices = np.stack((x, y, z_valid), axis=-1)

    # 颜色
    if rgb_image is not None:
        colors = rgb_image[v_valid, u_valid]  # BGR
        colors = colors[:, ::-1]  # BGR → RGB
    else:
        # 按深度伪彩
        z_norm = (z_valid - z_valid.min()) / (z_valid.max() - z_valid.min() + 1e-8)
        cmap = np.array([[0, 0, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]], dtype=np.uint8)
        idx = (z_norm * (cmap.shape[0] - 1)).astype(np.int32)
        colors = cmap[idx]

    print(f"点云点数: {len(vertices)}")
    return vertices, colors


def write_ply(path, vertices, colors):
    """将点云写入 PLY 文件"""
    n = len(vertices)
    header = f"""ply
format ascii 1.0
element vertex {n}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(path, "w") as f:
        f.write(header)
        for v, c in zip(vertices, colors):
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
    print(f"已保存: {path}")


def load_depth_tiff(path):
    """读取 TIFF 深度图，统一返回 float32 单位米"""
    depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise ValueError(f"无法读取: {path}")

    if depth.dtype == np.uint16:
        print("  depth format: uint16 (mm) → 转换为米")
        depth = depth.astype(np.float32) / 1000.0
    elif depth.dtype == np.float32:
        print("  depth format: float32 (m)")
    else:
        depth = depth.astype(np.float32)

    print(f"  尺寸: {depth.shape}, 深度范围: {depth.min():.3f}–{depth.max():.3f} m")
    return depth


def main():
    parser = argparse.ArgumentParser(description="TIFF 深度图转点云 (PLY)")
    parser.add_argument("--depth", default=None, help="深度图路径 (.tiff)")
    parser.add_argument("--rgb", default=None, help="彩色图路径 (.png)，可选")
    parser.add_argument("--save", default=None, help="输出 PLY 路径（默认与 depth 同目录）")
    parser.add_argument("--fx", type=float, default=918.27, help="fx")
    parser.add_argument("--fy", type=float, default=918.02, help="fy")
    parser.add_argument("--cx", type=float, default=643.14, help="cx")
    parser.add_argument("--cy", type=float, default=357.28, help="cy")
    parser.add_argument("--max-depth", type=float, default=3.0, help="最大深度阈值（米）")
    parser.add_argument("--step", type=int, default=2, help="采样步长")
    parser.add_argument("--view", action="store_true", help="用 matplotlib 简单查看点云")
    args = parser.parse_args()

    depth_path = args.depth
    # 若未指定文件，弹出选择框
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

    # 加载深度
    print(f"深度图: {depth_path}")
    depth_image = load_depth_tiff(depth_path)

    # 加载彩色图
    rgb_image = None
    if args.rgb:
        rgb_image = cv2.imread(args.rgb)
        if rgb_image is None:
            print(f"警告: 无法读取彩色图 {args.rgb}")
        else:
            if rgb_image.shape[:2] != depth_image.shape[:2]:
                rgb_image = cv2.resize(rgb_image, (depth_image.shape[1], depth_image.shape[0]))
            print(f"彩色图: {args.rgb} {rgb_image.shape}")

    # 相机内参
    intrinsic = np.array([
        [args.fx, 0, args.cx],
        [0, args.fy, args.cy],
        [0, 0, 1]
    ], dtype=np.float64)
    print(f"相机内参:\n{intrinsic}")

    # 生成点云
    print("生成点云...")
    vertices, colors = depth_to_pointcloud_ply(
        depth_image, intrinsic, rgb_image,
        max_depth=args.max_depth, step=args.step
    )

    # 保存 PLY
    if args.save:
        save_path = args.save
    else:
        save_path = os.path.splitext(depth_path)[0] + ".ply"
    write_ply(save_path, vertices, colors)

    # 可视化（可选，matplotlib 3D 散点图）
    if args.view and len(vertices) < 50000:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        print("正在显示...")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                   c=colors / 255.0, s=1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Depth Point Cloud")
        plt.tight_layout()
        plt.show()
    elif args.view:
        print(f"点数太多 ({len(vertices)}), 跳过 matplotlib 显示，用 MeshLab 等工具打开 PLY 文件查看")


if __name__ == "__main__":
    main()
