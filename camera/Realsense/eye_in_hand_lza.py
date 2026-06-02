import json
import cv2
import numpy as np
from math import *
from scipy.spatial.transform import Rotation as R

# 注意单位需要统一,若机器人移动为m此处也需要为米
square_size = 20  # 标定板每个格子的尺寸（假设单位为mm）
# dist_coeffs = np.array([-0.057804023735415, 0.182556905414127, 0, 0, 0], dtype=np.float64)  # 畸变系数
# camera_matrix = np.array([[901.064825254417, 0, 635.368228630993],
#                           [0, 901.091855452099, 374.049335829249],
#                           [0., 0., 1.]])
dist_coeffs = np.array([0.0705168032377493, -0.000346322281824665, 0, 0, 0], dtype=np.float64)  # 畸变系数
camera_matrix = np.array([[602.179167691128, 0, 319.098856377405],
                          [0, 602.541260479858, 247.066408398418],
                          [0., 0., 1.]])
# camera_matrix=np.array([[910.9901123,0.,638.86859131],
#  [  0.,911.05603027,371.0897522],
#  [  0.,0.,   1.        ]])

file_path = "eye_in_hand_wjc_640480_0726"


def euler2rot(euler):
    r = R.from_euler('ZYZ', euler, degrees=True)
    rotation_matrix = r.as_matrix()
    A_Z = np.array([[cos(euler[0] / 180 * pi), -sin(euler[0] / 180 * pi), 0],
                    [sin(euler[0] / 180 * pi), cos(euler[0] / 180 * pi), 0], [0, 0, 1]])
    B_y = np.array([[cos(euler[1] / 180 * pi), 0, sin(euler[1] / 180 * pi)], [0, 1, 0],
                    [-sin(euler[1] / 180 * pi), 0, cos(euler[1] / 180 * pi)]])
    C_z = np.array([[cos(euler[2] / 180 * pi), -sin(euler[2] / 180 * pi), 0],
                    [sin(euler[2] / 180 * pi), cos(euler[2] / 180 * pi), 0], [0, 0, 1]])
    R_zyz = A_Z @ B_y @ C_z
    return rotation_matrix


def get_RT_from_chessboard(img_path):
    '''
    :param img_path: 读取图片路径
    :param chess_board_x_num: 棋盘格x方向格子数
    :param chess_board_y_num: 棋盘格y方向格子数
    :param K: 相机内参
    :param chess_board_len: 单位棋盘格长度,mm
    :return: 相机外参
    '''
    # img=cv2.imread(image)
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(image, (11, 8),
                                             flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)

    corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

    obj_points = np.zeros((11 * 8, 3), dtype=np.float64)
    obj_points[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2) * square_size
    # print(object_points)

    if ret:  # 如果成功找到了角点
        # 通过角点坐标和标定板的实际尺寸来计算标定板的位姿
        _, rvecs, tvecs, _ = cv2.solvePnPRansac(obj_points, corners_subpix, camera_matrix, dist_coeffs)
        RT = np.column_stack(((cv2.Rodrigues(rvecs))[0], tvecs))
        RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
        # 计算重投影误差
        repoj_corners_subpix, _ = cv2.projectPoints(obj_points, rvecs, tvecs, camera_matrix, dist_coeffs)
        repoj_error = np.mean(np.linalg.norm(corners_subpix - repoj_corners_subpix, axis=1))
        print(f'reproje error:{repoj_error}')
        return RT, repoj_error
    else:
        return None


if __name__ == "__main__":

    with open(f'E:/Code/Metal_waste_grasp/camera/record/eye_in_hand/{file_path}/record.json', 'r') as f:
        datas = json.load(f)

    Rgri2base_s = []
    tgri2base_s = []
    Rtar2cam_s = []
    ttar2cam_s = []
    reproj_error_s = []

    for data in datas:
        robot_t_euler = data['robot']

        Rgri2base = euler2rot(robot_t_euler[3:6])
        tgri2base = np.array(robot_t_euler[0:3])
        Tgri2base = np.vstack((np.hstack((Rgri2base, tgri2base[:, np.newaxis])), np.array([0, 0, 0, 1])))

        Ttar2cam, reproj_error = get_RT_from_chessboard(
            f'E:/Code/Metal_waste_grasp/camera/record/eye_in_hand/{file_path}/{data["num"]:06d}.jpg')

        Rgri2base_s.append(Tgri2base[0:3, 0:3])
        tgri2base_s.append(Tgri2base[0:3, 3])
        Rtar2cam_s.append(Ttar2cam[0:3, 0:3])
        ttar2cam_s.append(Ttar2cam[0:3, 3])
        reproj_error_s.append(reproj_error)
    mean_reproj_error = np.mean(reproj_error_s)
    print(f'mean_reproj_error:{mean_reproj_error}')
    # index=np.where(np.array(reproj_error_s)<1.5*mean_reproj_error)
    index = range(len(reproj_error_s))
    Rcam2gri, tcam2gri = cv2.calibrateHandEye(Rgri2base_s, tgri2base_s, Rtar2cam_s, ttar2cam_s, cv2.CALIB_HAND_EYE_TSAI)
    # Rcam2gri, tcam2gri = cv2.calibrateHandEye(np.array(Rgri2base_s)[index], np.array(tgri2base_s)[index],
    #                                           np.array(Rtar2cam_s)[index], np.array(ttar2cam_s)[index],
    #                                           cv2.CALIB_HAND_EYE_ANDREFF)
    print(Rcam2gri)
    print(tcam2gri)
    # 重投影计算误差
    Ttar2base_s = []
    for Rgri2base, tgri2base, Rtar2cam, ttar2cam in zip(Rgri2base_s, tgri2base_s, Rtar2cam_s, ttar2cam_s):
        Tgri2base = np.eye(4)
        Tgri2base[0:3, 0:3] = Rgri2base
        Tgri2base[0:3, 3] = tgri2base
        Ttar2cam = np.eye(4)
        Ttar2cam[0:3, 0:3] = Rtar2cam
        Ttar2cam[0:3, 3] = ttar2cam
        Tcam2gri = np.eye(4)
        Tcam2gri[0:3, 0:3] = Rcam2gri
        Tcam2gri[0:3, 3] = tcam2gri.squeeze()
        Ttar2base = Tgri2base @ Tcam2gri @ Ttar2cam
        Ttar2base_s.append(Ttar2base)
        # print("tar2base: ")
        # print(Ttar2base)

    # 平均位姿
    Ttar2base_avg = np.mean(np.array(Ttar2base_s), axis=0)
    Rtar2base_avg = Ttar2base_avg[0:3, 0:3]
    ttar2base_avg = Ttar2base_avg[0:3, 3]

    mean_angle = []
    mean_trans = []
    # 计算与平均位姿之间的差值
    for Ttar2base in Ttar2base_s:
        R_diff = Rtar2base_avg.transpose() @ Ttar2base[0:3, 0:3]
        t_diff = ttar2base_avg - Ttar2base[0:3, 3]

        R_diff_angle, _ = cv2.Rodrigues(R_diff)
        R_diff_angle = np.linalg.norm(R_diff_angle)

        t_diff_norm = np.linalg.norm(t_diff)

        mean_angle.append(R_diff_angle * 180 / np.pi)
        mean_trans.append(t_diff_norm)

    mean_angle = np.mean(mean_angle)
    mean_trans = np.mean(mean_trans)

    if mean_angle < 5 and mean_trans < 5:
        print("calibrate success")
        print(f'mean_angle:{mean_angle}')
        print(f'mean_trans:{mean_trans}')