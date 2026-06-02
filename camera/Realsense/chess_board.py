import cv2.aruco
import pyrealsense2 as rs
import numpy as np
import cv2
import sys
sys.path.append('../robot_out/build/x64-Release')
import cdrflex as cd
import json
import os
all_status=[]
file_name="eye_in_hand_wjc_640480_0726"

def get_intrinsics(profile):
    intr = profile.as_video_stream_profile().get_intrinsics()
    camera_matrix=np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    coeffs=intr.coeffs
    print(f"Camera Intrinsics:{camera_matrix}")
    print(f'coeffs:{coeffs}')
    return camera_matrix,coeffs


def save_hand_eye_calibrate_data(num):
    robot_status=cd.get_current_posx()
    print(robot_status)
    all_status.append({'num':num,'robot':robot_status,})

if __name__ == "__main__":
    cd.tmain(50)
    num=0
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    color_profile = profile.get_stream(rs.stream.color)
    camera_matrix,coeff=get_intrinsics(color_profile)
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_KEEPRATIO)
            cv2.imshow('RealSense', color_image)
            key = cv2.waitKey(1)
            # 如果key为空格
            if key & 0xFF == ord(' '):
                cv2.imwrite(f'E:/Code/Metal_waste_grasp/camera/record/eye_in_hand/{file_name}/{num:06d}.jpg',color_image)
                save_hand_eye_calibrate_data(num)
                num=num+1
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        if os.path.exists('E:/Code/Metal_waste_grasp/camera/record/eye_in_hand/{file_name}/record.json')==False:
            with open(f'E:/Code/Metal_waste_grasp/camera/record/eye_in_hand/{file_name}/record.json','w') as f:
                json.dump(all_status,f,indent=1)

        pipeline.stop()
        cd.stop(1)
        print('done')