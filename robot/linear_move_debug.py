#!/usr/bin/env python3
"""AUBO Cartesian/joint motion debug helper.

This script is intentionally separate from safety_guided_motion.py.

典型用法
--------
预览：从归位后的当前 TCP 位姿开始，沿 Y 轴正方向移动 50 mm。
注意：不加 --execute 时只打印计划，不会真正控制机械臂运动。
  python robot/linear_move_debug.py --axis y --distance-mm 50 --execute

执行：从归位后的当前 TCP 位姿开始，沿 Y 轴正方向移动 50 mm。
--steps 5 表示把这段位移拆成 5 个小的笛卡尔目标点依次执行。
默认 method=movel，调用方式和 robot_control_ui.py 里的直线运动一致：
读取 get_status()，修改 XYZ，再调用 robot.movel(pose)。
  python robot/linear_move_debug.py --axis y --distance-mm 50 --steps 5 --execute

调试 movej：先归位到 [0, 0, 90, 0, 90, 0] 度，
然后让第 6 个关节在当前位置基础上转动 -10 度。
  python robot/linear_move_debug.py --joint 6 --joint-delta-deg -10 --execute

尝试 SDK 的直线插补接口：沿 X 轴正方向移动 30 mm。
--method line 会调用 robot.movel_line_async(pose)。
只有当前编译出的 .so 导出了 movel_line_async 时才能使用。
  python robot/linear_move_debug.py --axis x --distance-mm 30 --method line --execute
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import math
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SDK_RELPATH = "robot/01_calibrate_robot/build/modules/pybind"
AUBO_LIB_RELPATH = "robot/01_calibrate_robot/third_party/aubo/lib/x64"
HOME_JOINTS_DEG = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]


def _preload_deps() -> None:
    """Preload native libraries whose RPATH is often missing in the SDK .so."""
    conda_lib = Path(sys.prefix) / "lib"
    aubo_lib = ROOT / AUBO_LIB_RELPATH
    for folder, names in (
        (conda_lib, (
            "libopenblas.so.0",
            "libjsoncpp.so.27",
            "libvisp_core.so.3.7",
            "libvisp_robot.so.3.7",
        )),
        (aubo_lib, ()),
    ):
        if not folder.exists():
            continue
        for name in names:
            path = folder / name
            if path.exists():
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def _find_sdk_so(sdk_dir: Path) -> Path:
    tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    candidates = list(sdk_dir.glob(f"*{tag}*.so")) + list(sdk_dir.glob("robot*.so"))
    if not candidates:
        raise FileNotFoundError(f"no robot SDK .so found in {sdk_dir}")
    return candidates[0]


def load_robot_module(sdk_dir: str | None = None):
    """Load the pybind11 SDK module named 'robot' without importing robot/ package."""
    _preload_deps()
    so_dir = Path(sdk_dir) if sdk_dir else ROOT / SDK_RELPATH
    so_path = _find_sdk_so(so_dir)

    robot_pkg = sys.modules.pop("robot", None)
    try:
        spec = importlib.util.spec_from_file_location("robot", so_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to create import spec for {so_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod, so_path
    finally:
        if robot_pkg is not None:
            sys.modules["robot"] = robot_pkg


def fmt_pose(pose: list[float]) -> str:
    return (
        f"X={pose[0] * 1000:+8.1f} mm  "
        f"Y={pose[1] * 1000:+8.1f} mm  "
        f"Z={pose[2] * 1000:+8.1f} mm  "
        f"RX={math.degrees(pose[3]):+7.2f} deg  "
        f"RY={math.degrees(pose[4]):+7.2f} deg  "
        f"RZ={math.degrees(pose[5]):+7.2f} deg"
    )


def fmt_joints(joints: list[float]) -> str:
    return "[" + ", ".join(f"{math.degrees(j):+7.2f}" for j in joints) + "] deg"


def move_home(robot, execute: bool) -> None:
    """Move to the known safe home joint pose before any debug motion."""
    home_rad = [math.radians(v) for v in HOME_JOINTS_DEG]
    print(f"[home] target: {HOME_JOINTS_DEG} deg")
    if not execute:
        print("[dry-run] home move is also preview-only; add --execute to send robot.movej(home)")
        return

    robot.movej(home_rad)
    time.sleep(0.2)
    print(f"[home] after : {fmt_joints(list(robot.get_joint()))}")


def axis_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis.lower()]


def run_joint_jog(robot, joint: int, delta_deg: float, execute: bool) -> None:
    joints = list(robot.get_joint())
    target = joints.copy()
    target[joint - 1] += math.radians(delta_deg)

    print(f"[joint] current: {fmt_joints(joints)}")
    print(f"[joint] target : {fmt_joints(target)}")
    if not execute:
        print("[dry-run] add --execute to send robot.movej(target)")
        return

    robot.movej(target)
    time.sleep(0.2)
    print(f"[joint] after  : {fmt_joints(list(robot.get_joint()))}")


def build_cartesian_waypoints(
    start_pose: list[float],
    axis: str,
    distance_mm: float,
    steps: int,
) -> list[list[float]]:
    idx = axis_index(axis)
    waypoints = []
    for step in range(1, steps + 1):
        pose = start_pose.copy()
        pose[idx] += (distance_mm / 1000.0) * step / steps
        waypoints.append(pose)
    return waypoints


def run_cartesian_move(
    robot,
    axis: str,
    distance_mm: float,
    steps: int,
    method: str,
    settle_s: float,
    execute: bool,
) -> None:
    start = list(robot.get_status())
    waypoints = build_cartesian_waypoints(start, axis, distance_mm, steps)

    print(f"[pose] start : {fmt_pose(start)}")
    print(f"[plan] axis={axis.upper()} distance={distance_mm:+.1f} mm steps={steps} method={method}")
    for i, wp in enumerate(waypoints, 1):
        print(f"  [{i:02d}/{steps:02d}] {fmt_pose(wp)}")

    if not execute:
        print("[dry-run] add --execute to send motion commands")
        return

    if method == "line" and not hasattr(robot, "movel_line_async"):
        raise RuntimeError("this SDK module does not export movel_line_async")

    for i, wp in enumerate(waypoints, 1):
        print(f"[move] waypoint {i}/{steps}")
        if method == "movel":
            # This mirrors robot_control_ui.py: get_status(), change XYZ, robot.movel(pose).
            # In the current binding, movel() computes IK then calls JointMove.
            robot.movel(wp)
        elif method == "async":
            robot.movel_async(wp)
            time.sleep(settle_s)
        elif method == "line":
            robot.movel_line_async(wp)
            time.sleep(settle_s)
        else:
            raise ValueError(f"unknown method: {method}")

    time.sleep(0.2)
    print(f"[pose] after : {fmt_pose(list(robot.get_status()))}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Debug AUBO movej/movel motion from the current robot state."
    )
    parser.add_argument("--sdk-dir", default=None, help="override SDK pybind .so directory")
    parser.add_argument("--execute", action="store_true", help="actually send motion commands")

    cart = parser.add_argument_group("Cartesian move from current TCP pose")
    cart.add_argument("--axis", choices=("x", "y", "z"), default="y")
    cart.add_argument("--distance-mm", type=float, default=30.0)
    cart.add_argument("--steps", type=int, default=3)
    cart.add_argument("--method", choices=("movel", "async", "line"), default="movel")
    cart.add_argument("--settle-s", type=float, default=0.4, help="wait after async/line waypoint")

    joint = parser.add_argument_group("Raw movej joint jog")
    joint.add_argument("--joint", type=int, choices=range(1, 7), metavar="1..6")
    joint.add_argument("--joint-delta-deg", type=float, default=0.0)

    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be >= 1")

    robot, so_path = load_robot_module(args.sdk_dir)
    print(f"[sdk] loaded: {so_path}")

    try:
        ok = robot.init()
        if not ok:
            print("[error] robot.init() failed")
            return 1
        move_home(robot, args.execute)

        if args.joint is not None:
            run_joint_jog(robot, args.joint, args.joint_delta_deg, args.execute)
        else:
            run_cartesian_move(
                robot=robot,
                axis=args.axis,
                distance_mm=args.distance_mm,
                steps=args.steps,
                method=args.method,
                settle_s=args.settle_s,
                execute=args.execute,
            )
    finally:
        try:
            robot.log_out()
            print("[sdk] logged out")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
