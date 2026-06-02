"""Robot state readers: mock (fallback) and real (AUBO i16 SDK)."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np


class RobotStateReader:
    def get_joint_positions(self) -> dict[str, float]:
        raise NotImplementedError

    def get_end_effector_pose(self) -> dict | None:
        """{pos: ndarray(3,), rpy: ndarray(3,)} in base frame, or None."""
        return None


class MockRobotStateReader(RobotStateReader):
    """Returns URDF joint names with smooth sinusoidal motion (for testing)."""

    def __init__(self):
        self._index = 0

    def get_joint_positions(self) -> dict[str, float]:
        self._index += 1
        t = self._index * 0.02
        return {
            "shoulder_joint": 0.5 * math.sin(t * 0.30),
            "upperArm_joint": 0.3 * math.sin(t * 0.50 + 0.5),
            "foreArm_joint": 0.8 + 0.25 * math.sin(t * 0.40 + 1.0),
            "wrist1_joint": 0.2 * math.sin(t * 0.60 + 1.5),
            "wrist2_joint": 0.3 * math.sin(t * 0.35 + 2.0),
            "wrist3_joint": 0.1 * math.sin(t * 0.45 + 2.5),
            "left_joint": -0.02,
            "right_joint": -0.02,
        }


class RealRobotStateReader(RobotStateReader):
    """Reads live joint positions from an AUBO i16 robot via the C++ SDK.

    Falls back to MockRobotStateReader if the robot cannot be reached.
    """

    SDK_RELPATH = "robot/01_calibrate_robot/build/modules/pybind"
    AUBO_LIB_RELPATH = "robot/01_calibrate_robot/third_party/aubo/lib/x64"

    def __init__(self, sdk_path: str | None = None):
        self._connected = False
        self._mod = None
        self._sdk_path = sdk_path

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _project_root() -> Path:
        """Absolute path to project root (parent of robot/ package)."""
        return Path(__file__).resolve().parent.parent

    def _preload_deps(self) -> None:
        """Pre-load native dependencies with RTLD_GLOBAL before importing the SDK.

        The .so has a broken RPATH (points to a non-existent /home/hzy/software/…).
        We can't fix LD_LIBRARY_PATH at this point (glibc reads it at process start),
        so we manually dlopen the missing libs with RTLD_GLOBAL so their symbols
        are visible when the SDK .so is loaded.
        """
        import ctypes

        conda_lib = Path(sys.prefix) / "lib"
        aubo_lib = self._project_root() / self.AUBO_LIB_RELPATH

        # Dependencies that RPATH can't find (conda env libs)
        conda_deps = [
            "libopenblas.so.0",
            "libjsoncpp.so.27",
            "libvisp_core.so.3.7",
            "libvisp_robot.so.3.7",
        ]
        for lib in conda_deps:
            path = conda_lib / lib
            if path.exists():
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)

    def _find_sdk_so(self, so_dir: str) -> str | None:
        """Find the robot SDK .so matching the current Python version."""
        import sys

        so_dir_obj = Path(so_dir)
        tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
        candidates = list(so_dir_obj.glob(f"*{tag}*"))
        if not candidates:
            # fallback: any robot*.so
            candidates = list(so_dir_obj.glob("robot*.so"))
            if candidates:
                print(f"[RealRobotStateReader] no .so for {tag}, trying {candidates[0].name}")
        return str(candidates[0]) if candidates else None

    # ── public API ────────────────────────────────────────────

    def connect(self) -> bool:
        if self._connected:
            return True
        try:
            import importlib.util
            import sys

            so_dir = self._sdk_path or str(
                self._project_root() / self.SDK_RELPATH
            )

            # Pre-load native deps (broken RPATH in .so means dlopen can't find them)
            self._preload_deps()

            so_path = self._find_sdk_so(so_dir)
            if so_path is None:
                print(f"[RealRobotStateReader] no robot SDK .so found in {so_dir}")
                return False

            # The .so exports PyInit_robot (from pybind11 module name "robot"),
            # but the project also has a robot/ package.  Temporarily remove the
            # package from sys.modules so the SDK .so can be loaded with its
            # correct module name, then restore it.
            robot_pkg = sys.modules.pop("robot", None)

            spec = importlib.util.spec_from_file_location("robot", so_path)
            if spec is None or spec.loader is None:
                print("[RealRobotStateReader] failed to create SDK spec")
                return False
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._mod = mod

            # Restore robot/ package in sys.modules
            if robot_pkg is not None:
                sys.modules["robot"] = robot_pkg

            ok = self._mod.init()
            if ok:
                self._connected = True
                print("[RealRobotStateReader] connected to AUBO robot")
            else:
                print("[RealRobotStateReader] robot.init() returned False")
            return ok
        except Exception as exc:
            print(f"[RealRobotStateReader] connection failed: {exc}")
            return False

    def disconnect(self):
        if self._connected and self._mod is not None:
            try:
                self._mod.log_out()
            except Exception:
                pass
        self._connected = False
        print("[RealRobotStateReader] disconnected")

    def get_joint_positions(self) -> dict[str, float]:
        if not self._connected or self._mod is None:
            return MockRobotStateReader().get_joint_positions()
        try:
            raw = list(self._mod.get_joint())  # 6 floats (radians)
            return {
                "shoulder_joint": raw[0],
                "upperArm_joint": raw[1],
                "foreArm_joint": raw[2],
                "wrist1_joint": raw[3],
                "wrist2_joint": raw[4],
                "wrist3_joint": raw[5],
                "left_joint": -0.02,   # gripper — fixed open
                "right_joint": -0.02,
            }
        except Exception as exc:
            print(f"[RealRobotStateReader] read error: {exc}")
            return MockRobotStateReader().get_joint_positions()

    def get_end_effector_pose(self) -> dict | None:
        """Read [x,y,z,rx,ry,rz] from SDK for FK cross-validation."""
        if not self._connected or self._mod is None:
            return None
        try:
            s = list(self._mod.get_status())
            return {"pos": np.array(s[0:3]), "rpy": np.array(s[3:6])}
        except Exception:
            return None
