import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime
from Realsense import RealsenseCamera


# 初始化 RealSense 管道
pipeline = rs.pipeline()
config = rs.config()

# 配置彩色流
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动摄像头
pipeline.start(config)

camera = RealsenseCamera(width=640, height=480)
# 获取相机内参和畸变系数
camera_matrix, distortion_coeffs = camera.get_intrinsics()
print(camera_matrix)
print(distortion_coeffs)

pipeline.stop()
cv2.destroyAllWindows()