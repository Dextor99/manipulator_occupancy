#!/usr/bin/env python3
"""
运动控制子进程（独立进程入口）。

由 RobotCommander 通过 subprocess.Popen 启动。
通过 mmap 共享文件与主进程通信：

  offset 0 (double): speed_scale  (主进程→子进程)
  offset 8 (double): y_pos       (子进程→主进程)
  offset 16 (int8):  running     (主进程→子进程, 0=停止)

用法
----
  python robot/motion_worker.py <shm_path> <robot_ip> <range_m> [base_omega]

真实连续运动使用 robot.movel_line(..., block=False) 一次发送到端点，
运行中监控安全速度和端点位置。旧的 movel_async() 是 IK + JointMove 的
非阻塞接口，不再用于安全引导运动。
"""

from __future__ import annotations

import math
import mmap
import os
import struct
import sys
import time
from pathlib import Path


def main():
    if len(sys.argv) < 4:
        print("用法: motion_worker.py <shm_path> <robot_ip> <range_m> [base_omega]")
        sys.exit(1)

    shm_path = sys.argv[1]
    robot_ip = sys.argv[2]
    range_m = float(sys.argv[3])
    base_omega = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8

    # ── 打开共享内存 ──
    fd = os.open(shm_path, os.O_RDWR)
    shm = mmap.mmap(fd, 32)

    def _read_double(offset: int) -> float:
        shm.seek(offset)
        return struct.unpack('d', shm.read(8))[0]

    def _read_int8(offset: int) -> int:
        shm.seek(offset)
        return shm.read(1)[0]

    def _write_double(offset: int, val: float):
        shm.seek(offset)
        shm.write(struct.pack('d', val))

    # ── 预加载 SDK 依赖 ──
    import ctypes

    conda_lib = Path(sys.prefix) / "lib"
    for _lib in ("libopenblas.so.0", "libjsoncpp.so.27",
                 "libvisp_core.so.3.7", "libvisp_robot.so.3.7"):
        _p = conda_lib / _lib
        if _p.exists():
            ctypes.CDLL(str(_p), mode=ctypes.RTLD_GLOBAL)

    # ── 加载 SDK .so ──
    from importlib import util as _util

    root = Path(__file__).resolve().parent
    so_dir = root / "01_calibrate_robot/build/modules/pybind"
    if not so_dir.exists():
        so_dir = root.parent / "robot/01_calibrate_robot/build/modules/pybind"

    tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    candidates = list(so_dir.glob(f"*{tag}*.so")) + list(so_dir.glob("robot*.so"))
    if not candidates:
        print(f"[MotionWorker] 未找到 .so (在 {so_dir})", flush=True)
        sys.exit(1)

    so_path = str(candidates[0])
    robot_pkg = sys.modules.pop("robot", None)
    try:
        spec = _util.spec_from_file_location("robot", so_path)
        if spec is None or spec.loader is None:
            sys.exit(1)
        mod = _util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as exc:
        print(f"[MotionWorker] 加载 SDK 失败: {exc}", flush=True)
        sys.exit(1)
    finally:
        if robot_pkg is not None:
            sys.modules["robot"] = robot_pkg

    # ── 连接 ──
    try:
        if not mod.init():
            print("[MotionWorker] robot.init() 失败", flush=True)
            sys.exit(1)
    except Exception as exc:
        print(f"[MotionWorker] 连接失败: {exc}", flush=True)
        sys.exit(1)
    print(f"[MotionWorker] SDK 连接成功 ({robot_ip})", flush=True)

    # ── 归位 ──
    home_joints = [0.0, 0.0, math.radians(90), 0.0, math.radians(90), 0.0]
    try:
        mod.movej(home_joints)
    except Exception as exc:
        print(f"[MotionWorker] movej 归位失败: {exc}", flush=True)
        sys.exit(1)
    print("[MotionWorker] 归位完成", flush=True)

    # ── 读取基准位姿 ──
    try:
        status = list(mod.get_status())
    except Exception as exc:
        print(f"[MotionWorker] 读取位姿失败: {exc}", flush=True)
        sys.exit(1)

    x0, y0, z0, rx0, ry0, rz0 = status
    print(f"[MotionWorker] 末端位姿: "
          f"X={x0:.3f} Y={y0:.3f} Z={z0:.3f} "
          f"RX={math.degrees(rx0):.1f}° RY={math.degrees(ry0):.1f}° "
          f"RZ={math.degrees(rz0):.1f}°", flush=True)

    if not hasattr(mod, "movel_line") or not hasattr(mod, "move_control_stop"):
        print("[MotionWorker] 当前 SDK .so 缺少 movel_line/move_control_stop，请重新编译 pybind robot 模块", flush=True)
        sys.exit(1)

    # ── 运动循环（非阻塞 LineMove 到端点，支持暂停/恢复） ──
    # TeachStart(MOV_Y) 在程序化 stop/continue/反向时容易让控制器进入 stop state。
    # 这里改为一次发送到 y_min/y_max 端点的非阻塞直线运动，机械臂连续运行；
    # 主循环只负责监控安全速度、端点和必要时打断/重发目标。
    #
    # 当前策略：
    # 1) 每次读取真实 TCP Y 位置；
    # 2) 当前方向目标是 y_max 或 y_min；
    # 3) speed_scale≈0 时 move_control_stop，停在当前位置；
    # 4) 障碍离开后，从当前位置继续向原方向端点移动。
    y_min = y0 - range_m
    y_max = y0 + range_m
    direction = 1.0
    max_line_vel = min(max(abs(range_m * base_omega), 0.02), 0.08)
    max_line_acc = min(max(max_line_vel * 2.0, 0.05), 0.20)
    edge_margin_m = 0.010
    min_line_vel = 0.006
    active_motion = False
    active_dir: float | None = None
    active_vel = 0.0
    paused_for_safety = False
    print("[MotionWorker] 开始运动循环 (非阻塞 movel_line 连续往返, 支持暂停)", flush=True)
    print(f"[MotionWorker] line_vel<= {max_line_vel:.3f} m/s  line_acc<= {max_line_acc:.3f} m/s^2", flush=True)
    print(f"[MotionWorker] Y range: {y_min:+.3f} .. {y_max:+.3f} m", flush=True)

    def _stop_motion(reason: str):
        nonlocal active_motion, active_dir, active_vel
        if active_motion:
            try:
                mod.move_control_stop(False)
                print(f"[MotionWorker] 停止当前直线运动: {reason}", flush=True)
            except Exception as exc:
                print(f"[MotionWorker] move_control_stop 异常: {exc}", flush=True)
            time.sleep(0.15)
        active_motion = False
        active_dir = None
        active_vel = 0.0

    def _clear_controller_stop():
        """清除控制器的 stop state，为新的 movel_line 做准备。
        move_control_continue 可能返回非零（如 11028=仍在停止），重试至多 3 次。
        """
        if not hasattr(mod, "move_control_continue"):
            return True
        for attempt in range(3):
            try:
                ret = mod.move_control_continue(False)
            except Exception as exc:
                print(f"[MotionWorker] move_control_continue 异常(attempt {attempt}): {exc}", flush=True)
                ret = -1
            if ret == 0:
                if attempt > 0:
                    print(f"[MotionWorker] move_control_continue 恢复成功 (attempt {attempt})", flush=True)
                return True
            time.sleep(0.15 + attempt * 0.10)
        print(f"[MotionWorker] move_control_continue 多次尝试后仍返回 {ret}", flush=True)
        return False

    def _start_line(new_dir: float, new_vel: float, current_y: float):
        nonlocal active_motion, active_dir, active_vel
        target_y = y_max if new_dir > 0.0 else y_min
        if abs(target_y - current_y) <= edge_margin_m:
            return True
        if not _clear_controller_stop():
            time.sleep(0.3)
        target_pose = [x0, target_y, z0, rx0, ry0, rz0]
        ret = mod.movel_line(target_pose, new_vel, max_line_acc, False, False)
        if ret != 0:
            print(f"[MotionWorker] movel_line(nonblock) 返回错误 ret={ret}", flush=True)
            time.sleep(0.20)
            return False
        active_motion = True
        active_dir = new_dir
        active_vel = new_vel
        print(f"[MotionWorker] 直线运动到 {'+Y' if new_dir > 0 else '-Y'} 端点 target={target_y:+.3f} vel={new_vel:.3f}m/s", flush=True)
        return True

    try:
        while _read_int8(16):
            speed = max(_read_double(0), 0.0)

            try:
                actual = list(mod.get_status())
                current_y = actual[1]
            except Exception:
                current_y = y0

            _write_double(8, current_y)

            if current_y >= y_max - edge_margin_m:
                if direction > 0.0:
                    _stop_motion(f"到达 +Y 端点附近 Y={current_y:+.3f}")
                direction = -1.0
            elif current_y <= y_min + edge_margin_m:
                if direction < 0.0:
                    _stop_motion(f"到达 -Y 端点附近 Y={current_y:+.3f}")
                direction = 1.0

            if speed <= 0.005:
                if not paused_for_safety:
                    _stop_motion("安全速度为 0")
                    paused_for_safety = True
            else:
                if paused_for_safety:
                    paused_for_safety = False
                desired_vel = max(min_line_vel, max_line_vel * min(speed, 1.0))
                vel_changed = (
                    active_vel <= 0.0
                    or abs(desired_vel - active_vel) / max(active_vel, 1e-6) > 0.25
                )
                dir_changed = active_dir is None or direction != active_dir
                if (not active_motion) or dir_changed or vel_changed:
                    if active_motion:
                        _stop_motion("重设方向/速度")
                    if not _start_line(direction, desired_vel, current_y):
                        continue

            time.sleep(0.030)

    except KeyboardInterrupt:
        pass
    finally:
        _stop_motion("退出")
        try:
            mod.log_out()
        except Exception:
            pass
        shm.close()
        os.close(fd)
        print("[MotionWorker] 退出", flush=True)


if __name__ == "__main__":
    main()
