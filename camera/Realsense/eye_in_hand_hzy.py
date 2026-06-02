import json
import cv2
import numpy as np
from math import *
import pandas as pd
import matplotlib.pyplot as plt

class Camera_lab:
    def __init__(self):
        self.K = np.array([[910.516824109979, 0., 647.146704836095 ],
                      [0., 910.512191358783, 358.305911693392],
                      [0.0, 0.0, 1]], dtype=np.float64) # 相机内参
        self.dist_coeffs = np.array([0.149634537456417, -0.308588264341989, 0, 0, 0], dtype=np.float64)  # 畸变系数

class Camera_wjc:
    def __init__(self):
        # self.K = np.array([[901.064825254417, 0, 635.368228630993],
        #               [0, 901.091855452099, 374.049335829249],
        #               [0.0, 0.0, 1]], dtype=np.float64) # 相机内参
        # self.dist_coeffs = np.array([-0.057804023735415, 0.182556905414127, 0, 0, 0], dtype=np.float64)  # 畸变系数
        self.K = np.array([[602.179167691128,0.,319.098856377405],
                           [0.,602.541260479858,247.066408398418],
                           [0.0, 0.0, 1]], dtype=np.float64)  # 相机内参
        self.dist_coeffs = np.array([0.0705168032377493, -0.000346322281824665, 0.0, 0.0, 0.0], dtype=np.float64)  # 畸变系数
class Camera0:
    def __init__(self):
        self.K = np.array([[918.27160645  ,0.,643.14483643 ],
                      [0.,918.02313232 ,357.28491211],
                      [0.0, 0.0, 1]], dtype=np.float64) # 相机内参
        # self.K = np.array([[1377.407471, 0., 964.717285],
        #                    [0., 1377.03479, 535.927368],
        #                    [0.0, 0.0, 1]], dtype=np.float64)  # 相机内参
        self.dist_coeffs = np.array([0,0,0,0,0], dtype=np.float64)  # 畸变系数

class Camera1:
    def __init__(self):
        self.K = np.array([[1512.201846, 0.0, 586.244918 ],
                      [0.0, 1511.945664, 791.240811],
                      [0.0, 0.0, 1]], dtype=np.float64) # 相机内参
        self.dist_coeffs = np.array([-0.125160,0.187903,-0.000024,0.000586,-0.034650], dtype=np.float64)  # 畸变系数


class Camera2:
    def __init__(self):
        self.K = np.array([[1511.62, 0.0, 606.464],
                           [0.0, 1510.29, 811.054],
                           [0.0, 0.0, 1]], dtype=np.float64)  # 相机内参
        self.dist_coeffs = np.array([ -0.116688,  0.148341, -0.00033, 9.4e-05, 0.029323],
                                    dtype=np.float64)  # 畸变系数


class Doosan():
    # 用于根据位姿计算变换矩阵
    def getEnd2Base(self, Tx, Ty, Tz, A, B, C):
        thetaX = A / 180 * pi
        thetaY = B / 180 * pi
        thetaZ = C / 180 * pi
        R = getZYZRotationMatrix(thetaX, thetaY, thetaZ)
        t = np.array([[Tx], [Ty], [Tz]])
        RT1 = np.column_stack([R, t])  # 列合并
        RT1 = np.row_stack((RT1, np.array([0, 0, 0, 1])))
        # RT1=np.linalg.inv(RT1)
        return RT1


class Huashu():
    # 用于根据位姿计算变换矩阵
    def getEnd2Base(self, Tx, Ty, Tz, A, B, C):
        thetaX = A / 180 * pi
        thetaY = B / 180 * pi
        thetaZ = C / 180 * pi
        R = getZYXRotationMatrix(thetaX, thetaY, thetaZ)
        t = np.array([[Tx], [Ty], [Tz]])
        RT1 = np.column_stack([R, t])  # 列合并
        RT1 = np.row_stack((RT1, np.array([0, 0, 0, 1])))
        # RT1=np.linalg.inv(RT1)
        return RT1


# 用于根据欧拉角计算旋转矩阵
def getZYZRotationMatrix(A, B, C):
    A_Z = np.array([[cos(A), -sin(A), 0], [sin(A), cos(A), 0], [0, 0, 1]])
    B_y = np.array([[cos(B), 0, sin(B)], [0, 1, 0], [-sin(B), 0, cos(B)]])
    C_z = np.array([[cos(C), -sin(C), 0], [sin(C), cos(C), 0], [0, 0, 1]])
    R = A_Z @ B_y @ C_z
    return R


def getZYXRotationMatrix(A, B, C):
    A_X = np.array([[1, 0, 0],[0,cos(C), -sin(C)], [0,sin(C), cos(C)]])
    B_Y = np.array([[cos(B), 0, sin(B)], [0, 1, 0], [-sin(B), 0, cos(B)]])
    C_Z = np.array([[cos(A), -sin(A), 0], [sin(A), cos(A), 0], [0, 0, 1]])
    R = C_Z @ B_Y @ A_X
    return R


# 用来从棋盘格图片得到相机外参，返回变换矩阵
def getBoard2Camera(img_path, chess_board_x_num, chess_board_y_num, K, dist_coeffs, chess_board_len):

    img = cv2.imread(img_path)
    # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (chess_board_x_num, chess_board_y_num), flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
    # if ret:
    #     # 绘制角点
    #     cv2.drawChessboardCorners(img, (chess_board_x_num, chess_board_y_num), corners, ret)
    #     cv2.imshow('img', img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    corner_points = np.zeros((2, corners.shape[0]), dtype=np.float64)
    for i in range(corners.shape[0]):
        corner_points[:, i] = corners[i, 0, :]
    obj_points = np.zeros((chess_board_x_num*chess_board_y_num,3),dtype=np.float64)
    obj_points[:,:2] = np.mgrid[0:chess_board_x_num,0:chess_board_y_num].T.reshape(-1,2) * chess_board_len
    retval, rvec, tvec = cv2.solvePnP(obj_points, corner_points.T, K, dist_coeffs)
    RT = np.column_stack(((cv2.Rodrigues(rvec))[0], tvec))
    RT = np.row_stack((RT, np.array([0, 0, 0, 1])))

    return RT

chess_board_x_num = 11  # 棋盘格x方向格子数
chess_board_y_num = 8  # 棋盘格y方向格子数
chess_board_len = 20  # 单位棋盘格长度,mm

camera = Camera_wjc()
robot = Doosan()

file_path = "eye_in_hand_wjc_640480_0726"  # 替换为您的实际文件夹名称
folder = f'E:/Code/Metal_waste_grasp/camera/record/eye_in_hand/{file_path}'
file_num = 30  # 照片数量

# 计算board to cam 变换矩阵
R_all_chess_to_cam = []
T_all_chess_to_cam = []
RT_all_chess_to_cam = []

# for i in range(file_num):
    # image_path = folder + '/' + str(i) + '.bmp'
    # board2camera = getBoard2Camera(image_path, chess_board_x_num, chess_board_y_num, camera.K,camera.dist_coeffs, chess_board_len)
    # R_all_chess_to_cam.append(board2camera[:3, :3])
    # T_all_chess_to_cam.append(board2camera[:3, 3].reshape((3, 1)))
    # RT_all_chess_to_cam.append(board2camera)
for i in range(file_num):
    image_path = f'{folder}/{i:06d}.jpg'
    board2camera = getBoard2Camera(image_path, chess_board_x_num, chess_board_y_num, camera.K, camera.dist_coeffs, chess_board_len)
    R_all_chess_to_cam.append(board2camera[:3, :3])
    T_all_chess_to_cam.append(board2camera[:3, 3].reshape((3, 1)))
    RT_all_chess_to_cam.append(board2camera)

# 计算end to base变换矩阵
# 读取机器人位姿的JSON文件
file_address = folder + '/' + "record.json"  # 从记录文件读取机器人六个位姿
robot_datas = pd.read_json(file_address)  # 从 JSON 文件读取数据
# 计算end to base变换矩阵
R_all_end_to_base = []
T_all_end_to_base = []
RT_all_end_to_base = []
for i in range(file_num):
    robot_pose = robot_datas.iloc[i]['robot']  # 提取机器人位姿
    end2base = robot.getEnd2Base(
        robot_pose[0],  # x
        robot_pose[1],  # y
        robot_pose[2],  # z
        robot_pose[3],  # A
        robot_pose[4],  # B
        robot_pose[5]   # C
    )
    R_all_end_to_base.append(end2base[:3, :3])
    T_all_end_to_base.append(end2base[:3, 3].reshape((3, 1)))
    RT_all_end_to_base.append(end2base)

# file_address =  folder + '/'+ "robot_poses.xlsx"  # 从记录文件读取机器人六个位姿
# sheet_1 = pd.read_excel(file_address)
# R_all_end_to_base = []
# T_all_end_to_base = []
# RT_all_end_to_base = []
# for i in range(file_num):
#     end2base = robot.getEnd2Base(sheet_1.iloc[i]['x'], sheet_1.iloc[i]['y'], sheet_1.iloc[i]['z'], sheet_1.iloc[i]['A'], sheet_1.iloc[i]['B'], sheet_1.iloc[i]['C'])
#     R_all_end_to_base.append(end2base[:3, :3])
#     T_all_end_to_base.append(end2base[:3, 3].reshape((3, 1)))
#     RT_all_end_to_base.append(end2base)

# 手眼标定函数
R, T = cv2.calibrateHandEye(R_all_end_to_base, T_all_end_to_base, R_all_chess_to_cam,
                            T_all_chess_to_cam)  # 手眼标定
cam2end = np.column_stack((R, T))
cam2end = np.row_stack((cam2end, np.array([0, 0, 0, 1])))  # 即为cam to end变换矩阵
print('相机相对于末端的变换矩阵为：')
print(cam2end)

# 保存ndarray到文件
np.save('hand_eye_calib.npy', cam2end)

# 棋盘角点坐标
obj_points = np.zeros((chess_board_x_num * chess_board_y_num, 3), dtype=np.float64)
obj_points[:, :2] = np.mgrid[0:chess_board_x_num, 0:chess_board_y_num].T.reshape(-1, 2) * chess_board_len
# 创建全为1的数组
ones_row = np.ones((1, chess_board_x_num * chess_board_y_num))
# 将全为1的数组与原数组进行垂直叠加
RT_chess_point = np.vstack([obj_points.T, ones_row])


RT_all_xyz_1 = []
for i in range(file_num):
    RT_end_to_base = RT_all_end_to_base[i]
    RT_chess_to_cam = RT_all_chess_to_cam[i]
    RT_chess_to_base = np.dot(RT_end_to_base , np.dot(cam2end , RT_chess_to_cam)) #棋盘到基坐标的变换矩阵
    print('第', i+1, '次')
    # print(RT_chess_to_base)
    chess_point_in_base = RT_chess_to_base@RT_chess_point
    array_1 = chess_point_in_base[:3, :]
    array_1 = array_1.T
    RT_all_xyz_1.append(array_1)
    print(np.max(array_1[:,2]))
stacked_xyz_1 = np.stack(RT_all_xyz_1)
print(stacked_xyz_1.shape)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 生成颜色映射
colors = plt.cm.viridis(np.linspace(0, 1, chess_board_x_num*chess_board_y_num))
# 遍历每一组的点
for group in range(stacked_xyz_1.shape[0]):
    x = stacked_xyz_1[group, :, 0]  # 取出每一组点的 x 坐标
    y = stacked_xyz_1[group, :, 1]  # 取出每一组点的 y 坐标
    z = stacked_xyz_1[group, :, 2]  # 取出每一组点的 z 坐标

    # 使用scatter函数绘制三维散点图
    ax.scatter(x, y, z, c=colors)

# 显示图像
plt.show()

# 求每组中每个对应点的平均值
mean_points = np.mean(stacked_xyz_1, axis=0)
# 计算每个点与平均值的距离
distances = np.linalg.norm(stacked_xyz_1 - mean_points, axis=2)
mean_dis = np.mean(distances, axis=0)
print(distances)
print('54个角点坐标的平均误差')
print(mean_dis)
# 求每组中每个对应点的距离平均值的最大距离
max_distances = np.max(distances, axis=0)
print('54个角点坐标的最大误差')
print(max_distances)

# 绘制平均距离
plt.figure(num=1, figsize=(10, 6))
plt.plot(mean_dis, marker='o')
plt.title('Average Distances')
plt.xlabel('Point')
plt.ylabel('Average Distance/(mm)')
plt.ylim(bottom=0)
plt.grid(True)
plt.show()

# 绘制最大距离
plt.figure(num=2, figsize=(10, 6))
plt.plot(max_distances, marker='x')
plt.title('Maximum Distances')
plt.xlabel('Point')
plt.ylabel('Maximum Distance/(mm)')
plt.ylim(bottom=0)
plt.grid(True)
plt.show()

# 创建一个图形窗口，并分割成两个坐标系
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制平均距离
ax1.plot(mean_dis, marker='o', color='blue', label='Average Distance')
ax1.set_xlabel('Coordinate Index')
ax1.set_ylabel('Average Distance', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True)
ax1.legend(loc='upper left')

# 创建第二个坐标系
ax2 = ax1.twinx()
ax2.plot(max_distances, marker='x', color='red', label='Max Distance')
ax2.set_ylabel('Max Distance', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.legend(loc='upper right')

plt.title('Comparison of Average and Max Distances for Each Coordinate')
plt.show()