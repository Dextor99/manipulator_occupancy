
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import glob
import os
from inout import load_json, save_json

square_size = 20  # 标定板每个格子的尺寸（假设单位为米)
reproj_threhold = 0.5


def euler2rot(euler):
    r = R.from_euler('ZYZ', euler, degrees=True)
    rotation_matrix = r.as_matrix()
    return rotation_matrix


def detect_aruco(img, camera_matrix):
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict)
    (corners, ids, rejected) = detector.detectMarkers(img)
    marker_length = 0.06
    corners_3d = np.array([[-marker_length / 2, marker_length / 2, 0], [marker_length / 2, marker_length / 2, 0],
                           [marker_length / 2, -marker_length / 2, 0], [-marker_length / 2, -marker_length / 2, 0]])

    if ids is not None:
        # cv2.aruco.drawDetectedMarkers(img, corners, ids)
        # rvecs, tvecs,_=cv2.aruco.estimatePoseSingleMarkers(corners, 0.06, camera_matrix, np.zeros(5))
        flag, rvecs, tvecs = cv2.solvePnP(corners_3d, corners[0], camera_matrix, np.zeros(5))
        rvecs = np.squeeze(rvecs)
        tvecs = np.squeeze(tvecs)
        cv2.drawFrameAxes(img, camera_matrix, np.zeros(5), rvecs, tvecs, 0.1)
        cv2.imshow('axis', img)
        cv2.waitKey()
        # print(f"rotation: {rvec} translation:{tvec}")
        return rvecs, tvecs
    return None, None


def get_RT_from_chessboard(undistort_img, camera_matrix):
    '''
    :param img_path: 读取图片路径
    :param chess_board_x_num: 棋盘格x方向格子数
    :param chess_board_y_num: 棋盘格y方向格子数
    :param K: 相机内参
    :param chess_board_len: 单位棋盘格长度,mm
    :return: 相机外参
    '''
    gray = cv2.cvtColor(undistort_img, cv2.COLOR_BGR2GRAY)
    # size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(undistort_img, (11, 8),
                                             flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:  # 如果成功找到了角点
        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                          (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

        obj_points = np.zeros((11 * 8, 3), dtype=np.float64)
        obj_points[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2) * square_size
        # print(object_points)
        # 通过角点坐标和标定板的实际尺寸来计算标定板的位姿
        _, rvecs, tvecs, _ = cv2.solvePnPRansac(obj_points, corners_subpix, camera_matrix,
                                                np.array([0, 0, 0, 0, 0], dtype=np.float64))
        RT = np.column_stack(((cv2.Rodrigues(rvecs))[0], tvecs))
        RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
        # 计算重投影误差
        repoj_corners_subpix, _ = cv2.projectPoints(obj_points, rvecs, tvecs, camera_matrix,
                                                    np.array([0, 0, 0, 0, 0], dtype=np.float64))
        repoj_error = np.mean(np.linalg.norm(corners_subpix - repoj_corners_subpix, axis=1))
        if repoj_error > reproj_threhold:
            return None, None
        # unsort_img=cv2.undistort(img,camera_matrix,dist_coeffs)
        cv2.drawFrameAxes(undistort_img, camera_matrix, np.zeros(5), rvecs, tvecs, 100)
        # cv2.imshow('axis', undistort_img)
        # cv2.waitKey()
        print(f'reproje error:{repoj_error}')
        rvecs = np.squeeze(rvecs)
        tvecs = np.squeeze(tvecs)
        return rvecs, tvecs
    else:
        return None, None


def cal_mean_diff(Rbase2gri_s, tbase2gri_s, Rtar2cam_s, ttar2cam_s, Rcam2base, tcam2base):
    Tgri2tar_s = []
    # 重投影计算误差
    for Rbase2gri, tbase2gri, Rtar2cam, ttar2cam in zip(Rbase2gri_s, tbase2gri_s, Rtar2cam_s, ttar2cam_s):
        base2gri = np.eye(4)
        base2gri[0:3, 0:3] = Rbase2gri
        base2gri[0:3, 3] = tbase2gri
        tar2cam = np.eye(4)
        tar2cam[0:3, 0:3] = Rtar2cam
        tar2cam[0:3, 3] = ttar2cam
        cam2base = np.eye(4)
        cam2base[0:3, 0:3] = Rcam2base
        cam2base[0:3, 3] = tcam2base.squeeze()
        gri2tar = base2gri @ cam2base @ tar2cam
        Tgri2tar_s.append(gri2tar)
        # print(gri2tar)

    # 平均位姿
    Ttar2base_avg = np.mean(np.array(Tgri2tar_s), axis=0)
    Rtar2base_avg = Ttar2base_avg[0:3, 0:3]
    ttar2base_avg = Ttar2base_avg[0:3, 3]

    mean_angle = []
    mean_trans = []
    # 计算与平均位姿之间的差值
    for Tgri2tar in Tgri2tar_s:
        R_diff = Rtar2base_avg.transpose() @ Tgri2tar[0:3, 0:3]
        t_diff = ttar2base_avg - Tgri2tar[0:3, 3]

        R_diff_rads, _ = cv2.Rodrigues(R_diff)
        R_diff_angle = np.linalg.norm(R_diff_rads * 180 / np.pi)

        t_diff_norm = np.linalg.norm(t_diff)

        mean_angle.append(R_diff_angle)
        mean_trans.append(t_diff_norm)

    mean_angle = np.mean(mean_angle)
    mean_trans = np.mean(mean_trans)
    return mean_angle, mean_trans, Rtar2base_avg, ttar2base_avg, Tgri2tar_s


if __name__ == "__main__":
    # 标定不能使用4：3分辨率，尤其是640 480 会标定失败
    root_path = 'record/eye_to_hand/hand_eye_1012_wjc_640_480'
    # root_path="record/effector_real_202502052149"
    # camera_path = os.path.join(root_path, 'camera.json')
    # camera_matrix = np.array(load_json(camera_path)['camera_matrix'])
    # dist_coeffs = np.array(load_json(camera_path)['discoeffs'])
    camera_matrix = np.array([[602.179167691128,	0	,319.098856377405],
                       [0,	602.541260479858	,247.066408398418],
                       [0.0, 0.0, 1]], dtype=np.float64)  # 相机内参
    dist_coeffs = np.array([0.0705168032377493	,-0.000346322281824665, 0, 0, 0],
                                dtype=np.float64)  # 畸变系数
    # camera_matrix = np.array([[901.064825254417, 0, 635.368228630993],
    #                    [0, 901.091855452099, 374.049335829249],
    #                    [0.0, 0.0, 1]], dtype=np.float64)  # 相机内参
    # dist_coeffs = np.array([-0.057804023735415, 0.182556905414127, 0, 0, 0],
    #                             dtype=np.float64)  # 畸变系数

    # dist_coeffs=np.zeros(5)
    img_paths = glob.glob(os.path.join(root_path, '*.jpg'))
    img_paths = sorted(img_paths)
    calibration_save_path = os.path.join(root_path, 'calibration.json')
    calibration_save = {}

    Rtar2cam_s = []
    ttar2cam_s = []
    delete_index = []
    for index, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        undistort_img = cv2.undistort(img, camera_matrix, dist_coeffs)
        rvec, tvec = get_RT_from_chessboard(undistort_img, camera_matrix)
        if rvec is not None:
            Rtar2cam, _ = cv2.Rodrigues(rvec)
            ttar2cam = np.array(tvec)
            # print(f'{index:06d}:\n{Rtar2cam}')
            Rtar2cam_s.append(Rtar2cam)
            ttar2cam_s.append(ttar2cam)
        else:
            delete_index.append(index)

    datas = load_json(os.path.join(root_path, 'record.json'))

    # 直接读取

    Rbase2gri_s = []
    tbase2gri_s = []

    # for index, data in enumerate(datas):
    #     if index in delete_index:
    #         continue
    #     robot_rot = np.array(data['R_e2b'])
    #     robot_t = np.array(data['t_e2b'])
    #     Tgri2base = np.vstack((np.hstack((robot_rot, robot_t[:, np.newaxis])), np.array([0, 0, 0, 1])))
    #     Tbase2gri = np.linalg.inv(Tgri2base)
    #
    #     Rbase2gri_s.append(Tbase2gri[0:3, 0:3])
    #     tbase2gri_s.append(Tbase2gri[0:3, 3])

    for index, data in enumerate(datas):
        if index in delete_index:
            continue
        if 'robot' not in data:
            print(f"警告：第 {index} 条数据没有 'robot' 键")
            continue

        # 从"robot"键获取数据
        robot_data = np.array(data['robot'])

        # 平移向量 (t_e2b)
        robot_t = robot_data[:3]

        # 旋转向量 (R_e2b)
        robot_rot = robot_data[3:]

        # 如果旋转向量是欧拉角，转为旋转矩阵
        R_e2b = euler2rot(robot_rot)  # euler2rot是你定义的欧拉角转旋转矩阵函数

        Tgri2base = np.vstack((np.hstack((R_e2b, robot_t[:, np.newaxis])), np.array([0, 0, 0, 1])))
        Tbase2gri = np.linalg.inv(Tgri2base)

        Rbase2gri_s.append(Tbase2gri[0:3, 0:3])
        tbase2gri_s.append(Tbase2gri[0:3, 3])
    Rcam2base, tcam2base = cv2.calibrateHandEye(Rbase2gri_s, tbase2gri_s, Rtar2cam_s, ttar2cam_s,
                                                cv2.CALIB_HAND_EYE_TSAI)
    print(f'优化前旋转：\n{Rcam2base}')
    print(f'优化前平移：\n{tcam2base}')
    calibration_save['Rcam2base'] = np.squeeze(Rcam2base).tolist()
    calibration_save['tcam2base'] = np.squeeze(tcam2base).tolist()

    mean_angle, mean_trans, Rtar2base_avg, ttar2base_avg, Tgri2tar_s = cal_mean_diff(Rbase2gri_s, tbase2gri_s,
                                                                                     Rtar2cam_s, ttar2cam_s, Rcam2base,
                                                                                     tcam2base)
    if mean_angle < 0.5 and mean_trans < 0.005:
        print("标定成功")
        print(f'平均角度误差(度):{mean_angle}')
        print(f'平均平移误差(米):{mean_trans}')
    else:
        print("标定失败，重新采集数据集")
        print(f'平均角度误差(度):{mean_angle}')
        print(f'平均平移误差(米):{mean_trans}')

    # 取小于平均值的位姿再进行标定
    opt_Rtar2cam_s = []
    opt_ttar2cam_s = []
    opt_Rbase2gri_s = []
    opt_tbase2gri_s = []

    for index, Tgri2tar in enumerate(Tgri2tar_s):
        R_diff = Rtar2base_avg.transpose() @ Tgri2tar[0:3, 0:3]
        t_diff = ttar2base_avg - Tgri2tar[0:3, 3]

        R_diff_rads, _ = cv2.Rodrigues(R_diff)
        R_diff_angle = np.linalg.norm(R_diff_rads * 180 / np.pi)

        t_diff_norm = np.linalg.norm(t_diff)

        if (R_diff_angle < mean_angle and t_diff_norm < mean_trans):
            opt_Rbase2gri_s.append(Rbase2gri_s[index])
            opt_tbase2gri_s.append(tbase2gri_s[index])
            opt_Rtar2cam_s.append(Rtar2cam_s[index])
            opt_ttar2cam_s.append(ttar2cam_s[index])
    opt_Rcam2base, opt_tcam2base = cv2.calibrateHandEye(opt_Rbase2gri_s, opt_tbase2gri_s, opt_Rtar2cam_s,
                                                        opt_ttar2cam_s, cv2.CALIB_HAND_EYE_TSAI)
    print(f'优化后旋转：\n{opt_Rcam2base}')
    print(f'优化后平移：\n{opt_tcam2base}')
    calibration_save['opt_Rcam2base'] = np.squeeze(opt_Rcam2base).tolist()
    calibration_save['opt_tcam2base'] = np.squeeze(opt_tcam2base).tolist()

    mean_angle, mean_trans, Rtar2base_avg, ttar2base_avg, Tgri2tar_s = cal_mean_diff(opt_Rbase2gri_s, opt_tbase2gri_s,
                                                                                     opt_Rtar2cam_s, opt_ttar2cam_s,
                                                                                     opt_Rcam2base, opt_tcam2base)

    if mean_angle < 0.5 and mean_trans < 0.005:
        print("标定成功")
        print(f'平均角度误差(度):{mean_angle}')
        print(f'平均平移误差(米):{mean_trans}')
    else:
        print("标定失败，重新采集数据集")
        print(f'平均角度误差(度):{mean_angle}')
        print(f'平均平移误差(米):{mean_trans}')
    save_json(calibration_save_path, calibration_save)