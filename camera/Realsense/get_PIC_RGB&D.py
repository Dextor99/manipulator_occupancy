import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 配置参数
file_name = "camera_capture_0709"
save_dir = "./camera/PIC/"
os.makedirs(os.path.join(save_dir, file_name, "color"), exist_ok=True)
os.makedirs(os.path.join(save_dir, file_name, "depth"), exist_ok=True)


def main():
    # 配置相机管道
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    print("启动相机...")
    profile = pipeline.start(config)

    # 获取深度比例系数
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    capture_count = 0
    try:
        while True:
            # 等待帧
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # 转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 显示采集状态
            display_image = color_image.copy()
            cv2.putText(display_image, f"已采集: {capture_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_image, "空格: 采集 | Q: 退出", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('相机采集 (彩色)', display_image)

            # 显示深度图（仅预览）
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET)
            cv2.imshow('深度预览', depth_colormap)

            # 处理按键
            key = cv2.waitKey(1)
            if key & 0xFF == ord(' '):  # 空格采集
                # 保存彩色图像
                color_path = os.path.join(save_dir, file_name, "color", f"{capture_count:06d}.jpg")
                cv2.imwrite(color_path, color_image)

                # 保存深度图像（原始16位数据）
                depth_path = os.path.join(save_dir, file_name, "depth", f"{capture_count:06d}.png")
                cv2.imwrite(depth_path, depth_image)

                print(f"已保存: {color_path} 和 {depth_path}")
                capture_count += 1

            elif key & 0xFF in [ord('q'), 27]:  # Q或ESC退出
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\n采集结束。共保存{capture_count}组图像到:")
        print(f"彩色图像: {os.path.join(save_dir, file_name, 'color')}")
        print(f"深度图像: {os.path.join(save_dir, file_name, 'depth')}")


if __name__ == "__main__":
    main()