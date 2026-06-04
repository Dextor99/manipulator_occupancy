"""
真实机械臂运动控制接口 — 基于 AUBO i16 SDK 模块级 API。

完全使用模块级 API（robot.init / movej / movel / get_joint / get_status），
避免 vpRobotAuboRobots.rtInit 的 readTxt 断言崩溃。

Y 轴往返运动通过笛卡尔空间直线运动实现（movel 直接控制末端 Y 坐标），
比之前通过 J1 近似更精确，末端沿 Y 方向走直线。

用法
----
  commander = RobotCommander(ip="192.168.123.96")
  commander.connect()                           # homing → [0,0,90,0,90,0]
  commander.start_y_oscillate(amp=0.30)         # 后台线程开始 Y 轴直线往返

  while running:
      y = commander.get_y_pos()                 # 读取当前真实 Y 位置
      commander.set_speed_scale(0.5)            # 调速 (0~1)
      time.sleep(0.1)

  commander.stop()
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import numpy as np


class RobotCommander:
    """AUBO i16 机器人运动控制（模块级 API，movel 笛卡尔空间直线运动）。"""

    def __init__(self, ip: str = "192.168.123.96", base_speed: float = 0.05):
        self.ip = ip
        self.base_speed = base_speed  # m/s，Y 方向最大速度
        self._mod = None
        self._connected = False

        # 运动控制后台线程
        self._motion_thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()

        # 线程间共享状态
        self._current_y_pos: float = 0.0
        self._speed_scale: float = 1.0

        # 归位后的末端位姿（固定 x, z, rx, ry, rz，只变 y）
        self._home_pose: list[float] | None = None

    # ══════════════════════════════════════════════════════════════
    # 连接 & 归位
    # ══════════════════════════════════════════════════════════════

    def connect(self, home_joints_deg: list[float] | None = None) -> bool:
        """连接机器人，归位到目标位姿。

        使用模块级 robot.init() + robot.movej()，然后读取末端位姿
        作为笛卡尔运动的参考基准。

        Parameters
        ----------
        home_joints_deg : 6 关节角 (度)，默认 [0, 0, 90, 0, 90, 0]
        """
        if self._connected:
            return True

        mod = self._get_module()
        if mod is None:
            return False
        self._mod = mod

        # ── 连接 ──
        try:
            ok = mod.init()
            if not ok:
                print("[RobotCommander] robot.init() 失败")
                return False
        except Exception as exc:
            print(f"[RobotCommander] 连接失败: {exc}")
            return False
        print(f"[RobotCommander] 已连接 {self.ip}")

        # ── 归位 ──
        if home_joints_deg is None:
            home_joints_deg = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]
        target_rad = [math.radians(j) for j in home_joints_deg]
        print(f"[RobotCommander] 归位至 {home_joints_deg}…")
        try:
            mod.movej(target_rad)
        except Exception as exc:
            print(f"[RobotCommander] movej 归位失败: {exc}")
            return False
        print("[RobotCommander] 归位完成")

        # ── 读取当前末端位姿作为基准 ──
        try:
            status = list(mod.get_status())
            self._home_pose = status
            print(f"[RobotCommander] 末端位姿: "
                  f"X={status[0]:.3f} Y={status[1]:.3f} Z={status[2]:.3f} "
                  f"RX={math.degrees(status[3]):.1f}° RY={math.degrees(status[4]):.1f}° "
                  f"RZ={math.degrees(status[5]):.1f}°")
            with self._lock:
                self._current_y_pos = status[1]
        except Exception as exc:
            print(f"[RobotCommander] 读取末端位姿失败: {exc}")
            return False

        # 同时读关节角确认
        try:
            cur = list(mod.get_joint())
            print(f"[RobotCommander] 当前关节: {[f'{math.degrees(j):.1f}°' for j in cur]}")
        except Exception:
            pass

        self._connected = True
        return True

    # ══════════════════════════════════════════════════════════════
    # Y 轴运动控制（后台线程）
    # ══════════════════════════════════════════════════════════════

    def start_y_oscillate(self, range_m: float = 0.30):
        """启动后台运动线程：Y 轴 ±range_m 直线往返（笛卡尔空间）。

        使用 movel 直接控制末端在 Y 方向的正弦往返运动，
        保持 X, Z, RX, RY, RZ 不变。末端沿 Y 走直线，精确可控。
        """
        if not self._connected or self._mod is None:
            print("[RobotCommander] 未连接")
            return
        if self._motion_thread is not None:
            print("[RobotCommander] 运动线程已在运行")
            return

        self._running = True
        self._motion_thread = threading.Thread(
            target=self._motion_loop,
            args=(range_m,),
            daemon=True,
        )
        self._motion_thread.start()
        print(f"[RobotCommander] Y 轴 ±{range_m}m 笛卡尔直线往返已启动")

    def set_speed_scale(self, scale: float):
        """线程安全地设置速度倍率 (0~1)。"""
        with self._lock:
            self._speed_scale = max(0.0, min(1.0, scale))

    def get_y_pos(self) -> float:
        """线程安全地读取当前 Y 位置（缓存值，后台线程每帧更新）。"""
        with self._lock:
            return self._current_y_pos

    def get_pose(self) -> list[float] | None:
        """读取当前末端位姿 [x, y, z, rx, ry, rz]（米+弧度）。"""
        if not self._connected or self._mod is None:
            return None
        try:
            return list(self._mod.get_status())
        except Exception:
            return None

    def stop(self):
        """停止机器人运动。"""
        self._running = False
        if self._motion_thread is not None:
            self._motion_thread.join(timeout=3.0)
            self._motion_thread = None

        # 原地停止（发一次 movej 到当前位置）
        if self._connected and self._mod is not None:
            try:
                cur = list(self._mod.get_joint())
                self._mod.movej(cur)
            except Exception:
                pass
        self._connected = False
        print("[RobotCommander] 已停止")

    # ══════════════════════════════════════════════════════════════
    # 后台运动线程
    # ══════════════════════════════════════════════════════════════

    def _motion_loop(self, range_m: float):
        """后台运动线程：movel 笛卡尔空间正弦波往返。

        核心思想
        --------
        归位后记录末端位姿 [x0, y0, z0, rx0, ry0, rz0]。
        然后固定 x0, z0 和姿态，只改变 y 坐标：
          y_target(t) = y0 + range_m × sin(ω × speed_scale × t)

        调用 movel([x0, y_target, z0, rx0, ry0, rz0]) 即可让末端
        沿 Y 方向做精确直线正弦往返。
        """
        mod = self._mod
        if self._home_pose is None:
            print("[RobotCommander] 未记录末端位姿，无法运动")
            return

        x0, y0, z0, rx0, ry0, rz0 = self._home_pose
        base_omega = 0.8  # rad/s，~7.85s 一次完整往返
        t_start = time.perf_counter()

        while self._running:
            with self._lock:
                speed = self._speed_scale

            # 时间基准正弦 Y 目标
            t_elapsed = time.perf_counter() - t_start
            phase = base_omega * speed * t_elapsed
            y_target = y0 + range_m * math.sin(phase)

            target_pose = [x0, y_target, z0, rx0, ry0, rz0]

            try:
                mod.movel(target_pose)
            except Exception as e:
                print(f"[RobotCommander] movel 异常: {e}")
                break

            # 更新 Y 位置
            with self._lock:
                self._current_y_pos = y_target

        print("[RobotCommander] 运动线程退出")

    # ══════════════════════════════════════════════════════════════
    # 工具方法
    # ══════════════════════════════════════════════════════════════

    def get_joints(self) -> list[float] | None:
        """读取当前 6 个关节角 (rad)。"""
        if not self._connected or self._mod is None:
            return None
        try:
            return list(self._mod.get_joint())
        except Exception:
            return None

    @property
    def speed_scale(self) -> float:
        with self._lock:
            return self._speed_scale

    # ══════════════════════════════════════════════════════════════
    # SDK 模块加载
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _get_module():
        """获取已加载的 robot 模块。"""
        import sys

        # 检查是否已有 robot 模块
        mod = sys.modules.get("robot")
        if mod is not None:
            if hasattr(mod, "init") and hasattr(mod, "movel"):
                return mod

        # 若没有，尝试加载
        import importlib

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
