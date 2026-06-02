import cv2.aruco
import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import os
import json

sys.path.append('../robot_out/build/x64-Release')
import cdrflex as cd

# 全局变量
all_status = []
file_name = "hand_eye_1012_wjc_640_480_2"
save_flag = False
save_dir = f'E:/Code/Metal_waste_grasp/camera/record/eye_to_hand/{file_name}'

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

def get_intrinsics(profile):
    intr = profile.as_video_stream_profile().get_intrinsics()
    camera_matrix = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    coeffs = intr.coeffs
    print(f"Camera Intrinsics:{camera_matrix}")
    print(f'coeffs:{coeffs}')
    return camera_matrix, coeffs

def save_hand_eye_calibrate_data(num):
    robot_status = cd.get_current_posx()
    print(robot_status)
    all_status.append({'num': num, 'robot': robot_status})

def mouse_callback(event, x, y, flags, param):
    global save_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        save_flag = True

if __name__ == "__main__":
    cd.tmain(50)
    num = 0

    # 配置相机流
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    color_profile = profile.get_stream(rs.stream.color)
    camera_matrix, coeff = get_intrinsics(color_profile)

    # 设置窗口和鼠标回调
    cv2.namedWindow('RealSense', cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback('RealSense', mouse_callback)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            cv2.imshow('RealSense', color_image)

            key = cv2.waitKey(1)

            # 空格键保存
            if key & 0xFF == ord(' '):
                cv2.imwrite(f'{save_dir}/{num:06d}.jpg', color_image)
                save_hand_eye_calibrate_data(num)
                print(f"保存图像和JSON数据: {num:06d}.jpg")
                num += 1

            # 鼠标左键保存
            if save_flag:
                cv2.imwrite(f'{save_dir}/{num:06d}.jpg', color_image)
                save_hand_eye_calibrate_data(num)
                print(f"保存图像和JSON数据: {num:06d}.jpg")
                num += 1
                save_flag = False

            # 退出
            if key & 0xFF == ord('q') or key == 27:
                break

    finally:
        # 确保保存JSON数据
        json_path = f'{save_dir}/record.json'
        with open(json_path, 'w') as f:
            json.dump(all_status, f, indent=1)
            print(f"已保存JSON数据到 {json_path}")

        # 释放资源
        pipeline.stop()
        cv2.destroyAllWindows()
        cd.stop(1)
        print('程序结束')