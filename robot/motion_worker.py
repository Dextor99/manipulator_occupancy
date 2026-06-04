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

    # ── 运动循环（累积相位，支持暂停/恢复） ──
    # phase += ω × speed × dt
    # speed=0 → phase 冻结 → 机械臂停在当前位置不动
    # speed>0 → phase 从暂停处继续 → 运动方向、位置连续无跳变
    #
    # 暂停机制：speed ≈ 0 时不发送 movel_async，机械臂保持当前位置。
    # 速度恢复后继续发送新目标。避免 speed_scale=0 时重复发送相同目标
    # 导致机器人持续向障碍施力。
    phase = 0.0
    t_last = time.perf_counter()
    print("[MotionWorker] 开始运动循环 (累积相位, 支持暂停)", flush=True)

    try:
        while _read_int8(16):
            now = time.perf_counter()
            dt = now - t_last
            t_last = now

            speed = max(_read_double(0), 0.0)

            # 累积相位：speed=0 时项为 0，相位自然冻结
            phase += base_omega * speed * dt
            y_target = y0 + range_m * math.sin(phase)

            target_pose = [x0, y_target, z0, rx0, ry0, rz0]

            # ── speed ≈ 0 → 暂停：不发 movel_async，机器人保持当前位置 ──
            if speed > 0.005:
                try:
                    mod.movel_async(target_pose)
                except Exception as e:
                    print(f"[MotionWorker] movel_async 异常: {e}", flush=True)
                    break
            # speed ≤ 0.005：跳过 movel，不向机器人发送新指令
            # 伺服保持最后一条 movel_async 的目标位置

            # 写入真实末端 Y 位置（无论是否暂停，都读取实际位置）
            try:
                actual = list(mod.get_status())
                _write_double(8, actual[1])
            except Exception:
                _write_double(8, y_target)

            # 固定 20ms 间隔（~50Hz），给控制器足够时间处理每条指令
            time.sleep(0.020)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            mod.log_out()
        except Exception:
            pass
        shm.close()
        os.close(fd)
        print("[MotionWorker] 退出", flush=True)


if __name__ == "__main__":
    main()
