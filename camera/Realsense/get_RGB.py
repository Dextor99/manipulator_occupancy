import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime

# 创建保存图像的文件夹
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"saved_rgb_images{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# 初始化 RealSense 管道
pipeline = rs.pipeline()
config = rs.config()

# 配置彩色流
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# 启动摄像头
pipeline.start(config)

camera_matrix = np.array([[901.064825254417, 0, 635.368228630993],
                          [0, 901.091855452099, 374.049335829249],
                          [0.0, 0.0, 1]], dtype=np.float64)  # 相机内参
distortion_coeffs = np.array([-0.057804023735415, 0.182556905414127, 0, 0, 0],
                             dtype=np.float64)  # 畸变系数

print("按空格键或鼠标左键保存当前RGB图像，按Q键退出程序")

# 全局变量用于存储当前帧
current_frame = None


def save_current_image():
    global current_frame
    if current_frame is not None:
        # 生成带时间戳的文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(output_dir, f"rgb_{timestamp}.png")

        # 保存图像
        cv2.imwrite(filename, current_frame)
        print(f"已保存图像: {filename}")

        # 在图像上显示保存成功提示
        display_image = current_frame.copy()
        cv2.putText(display_image, f"Saved: {filename}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("RGB Capture", display_image)
        cv2.waitKey(500)  # 短暂显示保存提示


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键按下
        save_current_image()


# 创建窗口并设置鼠标回调
cv2.namedWindow("RGB Capture")
cv2.setMouseCallback("RGB Capture", mouse_callback)

try:
    while True:
        # 获取帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # 转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())

        # 校正畸变
        undistorted_image = cv2.undistort(color_image, camera_matrix, distortion_coeffs)

        # 更新当前帧
        current_frame = undistorted_image.copy()

        # 显示图像
        display_image = undistorted_image.copy()

        # 添加操作提示
        cv2.putText(display_image, "Press SPACE or LEFT CLICK to save image, Q to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("RGB Capture", display_image)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # 空格键保存图像
            save_current_image()
        elif key == ord('q'):  # q键退出
            break

finally:
    # 清理资源
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"所有保存的图像位于: {os.path.abspath(output_dir)}")
    print("程序已退出。")