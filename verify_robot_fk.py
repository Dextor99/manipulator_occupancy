"""验证 URDF FK vs 真实机器人位姿（只读，不移动机器人）。

用法：
  conda run -n py310 python verify_robot_fk.py

输出对比 URDF FK 计算的末端位姿和 SDK get_status() 的实际位姿。
若偏差 > 5cm 请检查关节映射。  若偏差 ~16cm 且主要在 Z 轴，
通常是末端工具(TCP)偏移 —— 对比 wrist3_Link 即可。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

# 添加 SDK 路径
SDK_PATH = str(
    Path(__file__).resolve().parent
    / "robot/01_calibrate_robot/build/modules/pybind"
)
if SDK_PATH not in sys.path:
    sys.path.append(SDK_PATH)

from robot.urdf_model import URDFModel
from robot.robot_state_reader import RealRobotStateReader

# ── 当前假设的关节映射（SDK index → URDF joint name） ──
JOINT_MAP = [
    ("shoulder_joint",  0),   # J1 = 基座旋转
    ("upperArm_joint",  1),   # J2 = 肩部
    ("foreArm_joint",   2),   # J3 = 肘部
    ("wrist1_joint",    3),   # J4 = 腕部 1
    ("wrist2_joint",    4),   # J5 = 腕部 2
    ("wrist3_joint",    5),   # J6 = 腕部 3
]


def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → intrinsic XYZ RPY (rx, ry, rz)."""
    return Rot.from_matrix(R).as_euler("xyz")


def main():
    urdf = URDFModel("urdf/aubo_i16_gripper.urdf")

    # 连接机器人（只读，不移动）
    reader = RealRobotStateReader()
    if not reader.connect():
        print("无法连接机器人，退出。")
        sys.exit(1)

    try:
        print("=" * 60)
        print("  读取机器人当前状态 ...")
        print("=" * 60)

        # 1) 关节角
        angles = reader.get_joint_positions()
        print("\n【SDK 关节角】")
        for name, idx in JOINT_MAP:
            val = angles[name]
            print(f"  SDK joint[{idx}] → {name:20s} = {val:+.4f} rad  ({np.rad2deg(val):+7.2f}°)")

        # 2) SDK 末端位姿
        ee = reader.get_end_effector_pose()
        if ee is None:
            print("无法读取末端位姿，跳过对比。")
            return
        actual_pos = ee["pos"]
        actual_rpy = ee["rpy"]
        print(f"\n【SDK get_status() 末端位姿】")
        print(f"  Position: ({actual_pos[0]:.4f}, {actual_pos[1]:.4f}, {actual_pos[2]:.4f})")
        print(f"  RPY:      ({actual_rpy[0]:.4f}, {actual_rpy[1]:.4f}, {actual_rpy[2]:.4f}) rad")
        print(f"  RPY(deg): ({np.rad2deg(actual_rpy[0]):.1f}, {np.rad2deg(actual_rpy[1]):.1f}, {np.rad2deg(actual_rpy[2]):.1f})°")

        # 3) URDF FK
        angles.update({"left_joint": -0.02, "right_joint": -0.02})
        fk = urdf.link_transforms(angles)

        # 对比 gripper_base_link（法兰）和 wrist3_Link
        for link_name in ("gripper_base_link", "wrist3_Link"):
            if link_name not in fk:
                continue
            T = fk[link_name]
            pos = T[:3, 3]
            R = T[:3, :3]
            rpy = matrix_to_rpy(R)

            pos_err = np.linalg.norm(pos - actual_pos)
            rpy_err = np.linalg.norm(rpy - actual_rpy)
            print(f"\n【URDF FK {link_name}】")
            print(f"  Position: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
            print(f"  RPY:      ({rpy[0]:.4f}, {rpy[1]:.4f}, {rpy[2]:.4f}) rad")
            print(f"  ── vs SDK ──")
            print(f"  |Δpos| = {pos_err:.4f} m")
            print(f"  |Δrpy| = {rpy_err:.4f} rad")

            if pos_err < 0.01:
                print(f"  ✓ 位置匹配 (< 1cm)")
            elif pos_err < 0.05:
                print(f"  ~ 位置偏差 {pos_err*100:.1f}cm — 可能是工具(TCP)偏移")
            else:
                print(f"  ✗ 位置偏差 {pos_err*100:.1f}cm — 关节映射可能错误")

        # 4) 连续监视模式
        print("\n" + "=" * 60)
        print("  连续监视模式（每 0.5s 刷新，Ctrl+C 退出）")
        print("  手动推动机器人观察 FK 追踪是否准确")
        print("=" * 60)
        try:
            while True:
                a = reader.get_joint_positions()
                a.update({"left_joint": -0.02, "right_joint": -0.02})
                fk_now = urdf.link_transforms(a)
                ee_now = reader.get_end_effector_pose()
                if ee_now is not None:
                    T = fk_now.get("gripper_base_link",
                                   fk_now.get("wrist3_Link", np.eye(4)))
                    err = np.linalg.norm(T[:3, 3] - ee_now["pos"])
                    raw = f"  |Δpos| = {err*1000:.0f} mm  joints="
                    for name, _ in JOINT_MAP:
                        raw += f"{np.rad2deg(a[name]):6.1f}°"
                    print(raw, end="\r")
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n\n停止监视。")

    finally:
        reader.disconnect()


if __name__ == "__main__":
    main()
