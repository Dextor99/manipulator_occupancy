"""
RealSense D435i 取图程序
功能：实时预览 RGB/深度图，按空格保存图像（含对齐深度），按 q 退出

用法：
    python capture_images.py                     # RGB + 深度（默认 1280x720）
    python capture_images.py --rgb-only           # 仅 RGB
    python capture_images.py --width 848 --height 480
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ImportError:
    print("请先安装 pyrealsense2: pip install pyrealsense2")
    sys.exit(1)

# D435I 支持的深度分辨率（彩色通常都支持）
VALID_DEPTH_RES = {(1280, 720), (848, 480), (640, 480), (640, 360), (480, 270)}


class RealsenseCapture:
    """RealSense 相机取图封装"""

    def __init__(self, width=1280, height=720, rgb_only=False):
        self.width = width
        self.height = height
        self.rgb_only = rgb_only
        self.save_count = 0

        # 初始化管道
        self.pipeline = rs.pipeline()
        config = rs.config()

        # 配置彩色流
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        # 配置深度流
        if not rgb_only:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)

        # 启动相机
        self.profile = self.pipeline.start(config)

        # 获取深度缩放系数
        if not rgb_only:
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            self.align = rs.align(rs.stream.color)
        else:
            self.depth_scale = None
            self.align = None

        # 获取相机内参
        color_profile = self.profile.get_stream(rs.stream.color)
        intr = color_profile.as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array(intr.coeffs)

    def get_frames(self):
        """获取一帧 RGB 和对齐后的深度图像"""
        frames = self.pipeline.wait_for_frames()

        if not self.rgb_only:
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
        else:
            color_frame = frames.get_color_frame()
            depth_frame = None

        if not color_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())

        depth_image = None
        if depth_frame:
            depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale

        return color_image, depth_image

    def stop(self):
        self.pipeline.stop()


def create_save_dir(base_dir="captured", rgb_only=False):
    """创建带时间戳的保存目录"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, f"capture_{timestamp}")
    os.makedirs(os.path.join(save_dir, "rgb"), exist_ok=True)
    if not rgb_only:
        os.makedirs(os.path.join(save_dir, "depth"), exist_ok=True)
    print(f"图像保存至: {save_dir}")
    return save_dir


def depth_colormap(depth_image, max_dist=1.5):
    """将深度图转为彩色图用于显示"""
    depth_valid = np.clip(depth_image, 0, max_dist) / max_dist
    depth_8bit = (depth_valid * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)


def warmup(cam):
    """丢弃预热帧，让自动曝光稳定"""
    print("相机预热中...")
    for i in range(5):
        color_image, _ = cam.get_frames()
        if color_image is not None:
            print(f"  预热 {i+1}/5", end="\r")
    print("\n预热完成")


def main():
    parser = argparse.ArgumentParser(description="RealSense D435i 取图程序")
    parser.add_argument("--width", type=int, default=1280, help="图像宽度 (默认 1280)")
    parser.add_argument("--height", type=int, default=720, help="图像高度 (默认 720)")
    parser.add_argument("--rgb-only", action="store_true", help="仅采集 RGB 图像")
    parser.add_argument("--save-dir", default="captured", help="保存根目录 (默认 captured)")
    args = parser.parse_args()

    # 检查深度分辨率是否有效
    if not args.rgb_only and (args.width, args.height) not in VALID_DEPTH_RES:
        print(f"警告: 深度流可能不支持 {args.width}x{args.height}")
        print(f"      推荐分辨率: {', '.join(f'{w}x{h}' for w, h in sorted(VALID_DEPTH_RES))}")

    # 初始化相机
    print("正在启动 RealSense 相机...")
    cam = RealsenseCapture(width=args.width, height=args.height, rgb_only=args.rgb_only)
    print(f"分辨率: {args.width}x{args.height}")
    print(f"相机内参:\n{cam.camera_matrix}")
    print(f"畸变系数: {cam.dist_coeffs}")
    print("按 SPACE 保存图像，按 q 退出\n")

    save_dir = create_save_dir(args.save_dir, rgb_only=args.rgb_only)
    warmup(cam)

    # 判断显示宽度，缩放至屏幕合适大小
    display_width = args.width * (1 if args.rgb_only else 2)
    display_height = args.height
    scale = min(1.0, 1920 / display_width, 1000 / display_height)

    win_name = "RealSense Capture"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    if scale < 1.0:
        cv2.resizeWindow(win_name, int(display_width * scale), int(display_height * scale))

    try:
        while True:
            color_image, depth_image = cam.get_frames()
            if color_image is None:
                continue

            # 构建显示画面
            if depth_image is not None:
                depth_color = depth_colormap(depth_image)
                display = np.hstack((color_image, depth_color))
            else:
                display = color_image

            # 叠加提示信息
            info = f"Saved: {cam.save_count}  |  SPACE: save  |  q: quit"
            cv2.putText(display, info, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(win_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                # 保存 RGB
                rgb_path = os.path.join(save_dir, "rgb", f"{cam.save_count:06d}.png")
                cv2.imwrite(rgb_path, color_image)

                if depth_image is not None:
                    # 保存深度（单位：米，float32 格式存为 TIFF）
                    depth_path = os.path.join(save_dir, "depth", f"{cam.save_count:06d}.tiff")
                    cv2.imwrite(depth_path, depth_image)

                    print(f"[{cam.save_count}] 已保存: {rgb_path}  +  depth")
                else:
                    print(f"[{cam.save_count}] 已保存: {rgb_path}")

                cam.save_count += 1

            elif key == ord('q'):
                print("用户退出")
                break

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print(f"共保存 {cam.save_count} 张图像")
        print(f"保存路径: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    main()
