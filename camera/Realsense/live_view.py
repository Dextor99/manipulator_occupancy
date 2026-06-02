"""
RealSense D435i 实时三视图：RGB + 深度 + 点云

在一组 OpenCV 窗口中实时显示 RGB 图和深度图，
同时在 Open3D 窗口中显示由深度图转换的 3D 点云。

用法：
    python live_view.py
    python live_view.py --width 848 --height 480 --step 1
"""

import argparse
import sys
import numpy as np
import cv2
import open3d as o3d

try:
    import pyrealsense2 as rs
except ImportError:
    print("请先安装 pyrealsense2: pip install pyrealsense2")
    sys.exit(1)


class RealsenseCapture:
    """RealSense 相机取图封装"""

    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)

        self.profile = self.pipeline.start(config)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)

        # 相机内参
        color_stream = self.profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()
        self.intrinsic = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ], dtype=np.float64)
        self.dist_coeffs = np.array(intr.coeffs)

    def get_frames(self):
        """获取一帧对齐后的 RGB 和深度图"""
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
        return color, depth

    def stop(self):
        self.pipeline.stop()


def depth_to_pcd(depth_image, intrinsic, rgb_image, max_depth=3.0, step=2):
    """
    深度图 → Open3D 点云（带颜色）
    """
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    mask = (depth_image > 0) & (depth_image < max_depth)

    # 降采样
    if step > 1:
        mask[::step, :] = False
        mask[:, ::step] = False

    v_valid, u_valid = np.where(mask)
    z_valid = depth_image[v_valid, u_valid]

    # 反投影
    x = (u_valid.astype(np.float32) - cx) * z_valid / fx
    y = (v_valid.astype(np.float32) - cy) * z_valid / fy
    points = np.stack((x, y, z_valid), axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    # 颜色
    rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    colors = rgb[v_valid, u_valid].astype(np.float64) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def depth_colormap(depth_image, max_dist=1.5):
    """深度图 → 伪彩图（用于显示）"""
    depth_valid = np.clip(depth_image, 0, max_dist) / max_dist
    depth_8bit = (depth_valid * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)


def warmup(cam, n_frames=10):
    """丢弃预热帧，稳定自动曝光"""
    print("相机预热中...")
    for i in range(n_frames):
        cam.get_frames()
        print(f"  预热 {i+1}/{n_frames}", end="\r")
    print("\n预热完成")


def main():
    parser = argparse.ArgumentParser(description="RealSense 实时三视图：RGB + 深度 + 点云")
    parser.add_argument("--width", type=int, default=1280, help="图像宽度")
    parser.add_argument("--height", type=int, default=720, help="图像高度")
    parser.add_argument("--max-depth", type=float, default=3.0, help="最大有效深度 (米)")
    parser.add_argument("--step", type=int, default=2, help="点云采样步长 (1=全分辨率)")
    args = parser.parse_args()

    # --- 启动相机 ---
    print("正在启动 RealSense 相机...")
    cam = RealsenseCapture(width=args.width, height=args.height)
    print(f"分辨率: {args.width}x{args.height}")
    print(f"内参:\n{cam.intrinsic}")

    warmup(cam)

    # --- Open3D 点云窗口 ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud", width=1280, height=720)

    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.12, 0.12, 0.12])

    ctrl = vis.get_view_control()

    # 初始空点云（首帧填入真实数据）
    pcd = o3d.geometry.PointCloud()
    is_init = True

    # 操作提示
    print("\n实时显示中 ...")
    print("  Open3D 窗口: 鼠标拖拽旋转 · 滚轮缩放")
    print("  OpenCV 窗口: 按 q 退出\n")

    try:
        while True:
            color, depth = cam.get_frames()
            if color is None:
                continue

            # --- 更新点云 ---
            pcd_new = depth_to_pcd(depth, cam.intrinsic, color,
                                   max_depth=args.max_depth, step=args.step)
            n_pts = len(pcd_new.points)
            if n_pts == 0:
                continue

            if is_init:
                pcd.points = pcd_new.points
                pcd.colors = pcd_new.colors
                vis.add_geometry(pcd)

                ctrl.set_front([0.0, -0.3, -1.0])
                ctrl.set_lookat(pcd.get_center())
                ctrl.set_up([0.0, -1.0, 0.0])
                ctrl.set_zoom(0.6)
                is_init = False
            else:
                pcd.points = pcd_new.points
                pcd.colors = pcd_new.colors
                vis.update_geometry(pcd)

            # 刷新 Open3D 窗口（非阻塞）
            if not vis.poll_events():
                break
            vis.update_renderer()

            # --- 显示 RGB + 深度（两个独立 OpenCV 窗口，原生分辨率） ---
            dc = depth_colormap(depth)
            cv2.imshow("RGB", color)
            cv2.imshow("Depth (q: quit)", dc)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户退出")
                break

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cam.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()
        print("已退出")


if __name__ == "__main__":
    main()
