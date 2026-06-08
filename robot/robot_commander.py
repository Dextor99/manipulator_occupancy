"""
真实机械臂运动控制接口 — 基于 AUBO i16 SDK 模块级 API。

完全使用模块级 API（robot.init / movej / movel_line / get_joint / get_status），
避免 vpRobotAuboRobots.rtInit 的 readTxt 断言崩溃。

Y 轴往返运动通过笛卡尔空间直线运动实现（movel_line 直接控制末端 Y 坐标），
比之前通过 J1 近似更精确，末端沿 Y 方向走直线。

用法
----
  commander = RobotCommander(ip="192.168.123.96")
  commander.connect()                           # homing → [0,0,90,0,90,0]
  commander.start_y_oscillate(amp=0.30)         # 子进程开始 Y 轴直线往返

  while running:
      y = commander.get_y_pos()                 # 读取当前真实 Y 位置
      commander.set_speed_scale(0.5)            # 调速 (0~1)
      time.sleep(0.1)

  commander.stop()

独立子进程
----------
start_y_oscillate 通过 subprocess.Popen 启动 robot/motion_worker.py，
在主进程完全隔离的独立进程中运行 SDK movel_line() 循环。
通过 mmap 共享文件（~32 字节）交换 speed_scale / y_pos / running 标志。
"""

from __future__ import annotations

import math
import mmap
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np


class RobotCommander:
    """AUBO i16 机器人运动控制。

    运动在独立子进程中运行（subprocess.Popen），
    SDK movel_line() 的阻塞调用完全不占用主进程 GIL/CPU。
    """

    def __init__(self, ip: str = "192.168.123.96", base_speed: float = 0.05,
                 robot_mod=None):
        self.ip = ip
        self.base_speed = base_speed
        self._mod = robot_mod  # （仅主进程读取关节时使用）
        self._connected = False

        # ── 子进程控制（subprocess + mmap） ──
        self._process: subprocess.Popen | None = None
        self._shm: mmap.mmap | None = None
        self._shm_fd: int | None = None
        self._shm_path: str | None = None

        # mmap 布局：offset 0=speed_scale(double), 8=y_pos(double), 16=running(int8)
        self._SHM_SPEED = 0
        self._SHM_Y = 8
        self._SHM_RUNNING = 16
        self._SHM_SIZE = 32

        # ── 主进程 SDK 访问锁（get_joint / get_status 用） ──
        self._lock = threading.Lock()

        # 归位后的末端位姿
        self._home_pose: list[float] | None = None

    # ══════════════════════════════════════════════════════════════
    # 连接 & 归位（主进程，单次）
    # ══════════════════════════════════════════════════════════════

    def connect(self, home_joints_deg: list[float] | None = None) -> bool:
        """连接机器人，归位到目标位姿。

        归位在主进程执行（单次阻塞可接受），
        之后启动的子进程会建立独自的 SDK 连接。
        """
        if self._connected:
            return True

        if self._mod is None:
            mod = self._get_module()
            if mod is None:
                return False
            self._mod = mod
            try:
                ok = self._mod.init()
                if not ok:
                    print("[RobotCommander] robot.init() 失败")
                    return False
            except Exception as exc:
                print(f"[RobotCommander] 连接失败: {exc}")
                return False
        else:
            pass  # 注入模块，已 init

        print(f"[RobotCommander] 已连接 {self.ip}")

        # ── 归位 ──
        if home_joints_deg is None:
            home_joints_deg = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]
        target_rad = [math.radians(j) for j in home_joints_deg]
        print(f"[RobotCommander] 归位至 {home_joints_deg}…")
        try:
            self._mod.movej(target_rad)
        except Exception as exc:
            print(f"[RobotCommander] movej 归位失败: {exc}")
            return False
        print("[RobotCommander] 归位完成")

        # ── 读取当前末端位姿 ──
        try:
            status = list(self._mod.get_status())
            self._home_pose = status
            print(f"[RobotCommander] 末端位姿: "
                  f"X={status[0]:.3f} Y={status[1]:.3f} Z={status[2]:.3f} "
                  f"RX={math.degrees(status[3]):.1f}° RY={math.degrees(status[4]):.1f}° "
                  f"RZ={math.degrees(status[5]):.1f}°")
        except Exception as exc:
            print(f"[RobotCommander] 读取末端位姿失败: {exc}")
            return False

        try:
            cur = list(self._mod.get_joint())
            print(f"[RobotCommander] 当前关节: {[f'{math.degrees(j):.1f}°' for j in cur]}")
        except Exception:
            pass

        self._connected = True
        return True

    # ══════════════════════════════════════════════════════════════
    # Y 轴运动控制（子进程）
    # ══════════════════════════════════════════════════════════════

    def start_y_oscillate(self, range_m: float = 0.40, base_omega: float = 0.8):
        """启动子进程：Y 轴 ±range_m 正弦往返。

        通过 subprocess.Popen 启动 robot/motion_worker.py，
        独立进程运行 SDK movel_line() 循环，完全不阻塞主进程。
        """
        if not self._connected:
            print("[RobotCommander] 未连接")
            return
        if self._process is not None and self._process.poll() is None:
            print("[RobotCommander] 运动进程已在运行")
            return

        # ── 创建共享内存文件 ──
        shm_file = tempfile.NamedTemporaryFile(prefix='robot_shm_', delete=False)
        shm_file.write(b'\x00' * self._SHM_SIZE)
        shm_file.flush()
        self._shm_path = shm_file.name
        shm_file.close()

        # mmap 打开
        self._shm_fd = os.open(self._shm_path, os.O_RDWR)
        self._shm = mmap.mmap(self._shm_fd, self._SHM_SIZE)

        # 写入 running = 1
        self._shm.seek(self._SHM_RUNNING)
        self._shm.write(b'\x01')

        # 写入初始 speed = 1.0
        self._write_double(self._SHM_SPEED, 1.0)

        # ── 启动子进程 ──
        worker_script = str(Path(__file__).parent / "motion_worker.py")
        self._process = subprocess.Popen(
            [sys.executable, worker_script,
             self._shm_path, self.ip, str(range_m), str(base_omega)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # 子进程不需要继承主进程的 Open3D/SDK 连接
            close_fds=True,
        )

        # 启动一个线程读取子进程输出并打印
        def _reader():
            try:
                for line in iter(self._process.stdout.readline, b''):
                    print(f"  [motion] {line.decode().strip()}")
            except Exception:
                pass

        threading.Thread(target=_reader, daemon=True).start()

        # 等子进程完成初始化（连接、归位 ~2-3s）
        time.sleep(3.0)
        if self._process.poll() is not None:
            print("[RobotCommander] 运动进程异常退出")
            return
        print(f"[RobotCommander] Y 轴 ±{range_m}m 正弦往返 "
              f"(PID={self._process.pid})")

    def set_speed_scale(self, scale: float):
        """设置速度倍率 (0~1)，通过 mmap 共享内存传递。"""
        self._write_double(self._SHM_SPEED, max(0.0, min(1.0, scale)))

    def get_y_pos(self) -> float:
        """读取当前 Y 位置（mmap 共享内存）。"""
        if self._shm is None:
            return 0.0
        return self._read_double(self._SHM_Y)

    def get_pose(self) -> list[float] | None:
        """读取当前末端位姿（主进程 SDK 连接）。"""
        if not self._connected or self._mod is None:
            return None
        try:
            return list(self._mod.get_status())
        except Exception:
            return None

    def stop(self):
        """停止机器人运动。"""
        # 发送停止信号
        if self._shm is not None:
            self._shm.seek(self._SHM_RUNNING)
            self._shm.write(b'\x00')

        # 等待子进程退出
        if self._process is not None:
            try:
                self._process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            self._process = None

        # 清理 mmap
        if self._shm is not None:
            self._shm.close()
            self._shm = None
        if self._shm_fd is not None:
            os.close(self._shm_fd)
            self._shm_fd = None
        if self._shm_path is not None:
            try:
                os.unlink(self._shm_path)
            except Exception:
                pass
            self._shm_path = None

        # 主进程 SDK 发送停止
        if self._connected and self._mod is not None:
            try:
                cur = list(self._mod.get_joint())
                self._mod.movej(cur)
            except Exception:
                pass
        self._connected = False
        print("[RobotCommander] 已停止")

    # ══════════════════════════════════════════════════════════════
    # mmap 辅助
    # ══════════════════════════════════════════════════════════════

    def _read_double(self, offset: int) -> float:
        if self._shm is None:
            return 0.0
        self._shm.seek(offset)
        return struct.unpack('d', self._shm.read(8))[0]

    def _write_double(self, offset: int, val: float):
        if self._shm is None:
            return
        self._shm.seek(offset)
        self._shm.write(struct.pack('d', val))

    # ══════════════════════════════════════════════════════════════
    # 工具方法
    # ══════════════════════════════════════════════════════════════

    def get_joints(self) -> list[float] | None:
        """读取当前 6 个关节角 (rad)（主进程 SDK）。"""
        if not self._connected or self._mod is None:
            return None
        try:
            return list(self._mod.get_joint())
        except Exception:
            return None

    @property
    def speed_scale(self) -> float:
        if self._shm is None:
            return 0.0
        return self._read_double(self._SHM_SPEED)

    # ══════════════════════════════════════════════════════════════
    # SDK 模块加载
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _get_module():
        """获取已加载的 robot 模块。"""
        import importlib

        mod = sys.modules.get("robot")
        if mod is not None:
            if hasattr(mod, "init") and hasattr(mod, "movel"):
                return mod

        root = Path(__file__).resolve().parent.parent
        so_dir = root / "robot/01_calibrate_robot/build/modules/pybind"

        tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
        candidates = list(so_dir.glob(f"*{tag}*.so")) + list(so_dir.glob("robot*.so"))
        if not candidates:
            print(f"[RobotCommander] 未找到 AUBO SDK .so (在 {so_dir})")
            return None
        so_path = str(candidates[0])

        robot_pkg = sys.modules.pop("robot", None)
        try:
            spec = importlib.util.spec_from_file_location("robot", so_path)
            if spec is None or spec.loader is None:
                return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        except Exception as exc:
            print(f"[RobotCommander] 加载 AUBO SDK 失败: {exc}")
            return None
        finally:
            if robot_pkg is not None:
                sys.modules["robot"] = robot_pkg


if __name__ == "__main__":
    import sys

    commander = RobotCommander(ip=sys.argv[1] if len(sys.argv) > 1 else "192.168.123.96")
    if not commander.connect():
        sys.exit(1)
    commander.start_y_oscillate(range_m=0.40)
    print("运动中，按 Ctrl+C 停止")
    try:
        while True:
            print(f"  Y = {commander.get_y_pos():+.3f} m  speed = {commander.speed_scale:.0%}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        commander.stop()
