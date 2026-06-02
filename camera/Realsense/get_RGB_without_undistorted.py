import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime

# 配置保存路径和文件名
save_dir = "E:/Code/Metal_waste_grasp/camera/record/camera_intrinsics/640_480_lab"  # 替换为你想要的保存路径
os.makedirs(save_dir, exist_ok=True)  # 创建目录（如果不存在）

# 初始化RealSense管道
pipeline = rs.pipeline()
config = rs.config()

# 启用彩色流（1280x720分辨率，BGR格式，30fps）
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动摄像头
pipeline.start(config)

print("按空格键或鼠标左键保存图像，按Q键退出程序")

# 全局变量存储当前帧
current_frame = None
frame_count = 0


def save_current_frame():
    global frame_count
    if current_frame is not None:
        # 生成带时间戳的文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}_{frame_count:04d}.png"
        save_path = os.path.join(save_dir, filename)

        # 保存图像
        cv2.imwrite(save_path, current_frame)
        print(f"已保存: {save_path}")
        frame_count += 1

        # 显示保存提示
        display_frame = current_frame.copy()
        cv2.putText(display_frame, "SAVED: " + filename, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("RealSense Camera", display_frame)
        cv2.waitKey(300)  # 显示提示信息0.3秒


# 鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键点击
        save_current_frame()


# 创建窗口并设置鼠标回调
cv2.namedWindow("RealSense Camera")
cv2.setMouseCallback("RealSense Camera", mouse_callback)

try:
    while True:
        # 等待获取一帧数据
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # 转换为OpenCV格式并更新当前帧
        current_frame = np.asanyarray(color_frame.get_data())

        # 显示图像
        cv2.imshow("RealSense Camera", current_frame)

        # 按键处理
        key = cv2.waitKey(1)

        if key == ord(' '):  # 空格键保存
            save_current_frame()
        elif key == ord('q'):  # Q键退出
            break

finally:
    # 清理资源
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"所有图像已保存到: {os.path.abspath(save_dir)}")
    print("程序已退出")