"""Record RGB-D point clouds and robot joint states for Chapter 4.2 experiments."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from camera.mock_reader import MockRGBDReader
from camera.realsense_pipeline_reader import RealSensePipelineReader
from robot.robot_state_reader import MockRobotStateReader, RealRobotStateReader
from utils.config import load_config_dir


class Recorder:
    """Write one RGB-D + joint-state sequence as frame_XXXXX.npz files."""

    def __init__(
        self,
        save_dir: str | Path,
        config_dir: str | Path = "config",
        scene: str = "A",
        obstacle_desc: str = "",
        save_images: bool = False,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.frame_dir = self.save_dir / "frames"
        self.frame_dir.mkdir(exist_ok=True)

        self.config_dir = Path(config_dir)
        self.scene = scene
        self.obstacle_desc = obstacle_desc
        self.save_images = save_images
        self.frame_count = 0
        self.started_at = time.time()

        config = load_config_dir(self.config_dir)
        self.manifest: dict[str, Any] = {
            "scene": scene,
            "obstacle_desc": obstacle_desc,
            "created_at": self.started_at,
            "config_dir": str(self.config_dir),
            "workspace": config.get("workspace", {}),
            "safety": config.get("safety", {}),
            "intrinsic": config.get("intrinsic", {}),
            "extrinsic": config.get("extrinsic", {}),
            "frames": 0,
            "save_images": save_images,
            "joint_names": [],
        }

    def record_frame(self, frame: Any, joints: dict[str, float]) -> None:
        if not self.manifest["joint_names"]:
            self.manifest["joint_names"] = list(joints.keys())
        joint_names = self.manifest["joint_names"]
        joint_values = np.array([joints.get(name, 0.0) for name in joint_names], dtype=np.float64)

        payload: dict[str, Any] = {
            "points_cam": np.asarray(frame.points_cam, dtype=np.float32),
            "joint_values": joint_values,
            "timestamp": np.array(getattr(frame, "timestamp", time.time()), dtype=np.float64),
        }
        if self.save_images:
            if hasattr(frame, "color"):
                payload["color_image"] = np.asarray(frame.color)
            if hasattr(frame, "depth"):
                payload["depth_image"] = np.asarray(frame.depth)

        out = self.frame_dir / f"frame_{self.frame_count:05d}.npz"
        np.savez_compressed(out, **payload)
        self.frame_count += 1

    def close(self) -> None:
        self.manifest["frames"] = self.frame_count
        self.manifest["duration_wall_s"] = time.time() - self.started_at
        with (self.save_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, indent=2, ensure_ascii=False)


def load_sequence(record_dir: str | Path):
    """Yield recorded frames as dicts with points_cam, joint_dict, and timestamp."""
    record_dir = Path(record_dir)
    manifest_path = record_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    joint_names = manifest.get("joint_names", [])

    for frame_path in sorted((record_dir / "frames").glob("frame_*.npz")):
        data = np.load(frame_path)
        if "joint_values" in data.files:
            joint_values = data["joint_values"]
        elif "joint_angles" in data.files:
            joint_values = data["joint_angles"]
        else:
            print(f"[recorder] skip {frame_path}: missing joint_values")
            continue
        if not joint_names:
            joint_names = [f"joint_{i}" for i in range(len(joint_values))]
        yield {
            "path": frame_path,
            "points_cam": data["points_cam"].astype(np.float64),
            "joint_dict": {name: float(joint_values[i]) for i, name in enumerate(joint_names)},
            "timestamp": float(data["timestamp"]),
            "manifest": manifest,
        }


def _make_camera(source: str, width: int, height: int, fps: int):
    if source == "mock":
        return MockRGBDReader(dt=1.0 / max(fps, 1))
    if source == "realsense":
        return RealSensePipelineReader(width=width, height=height, fps=fps)
    raise ValueError(f"unknown camera source: {source}")


def _make_state_reader(use_real_robot: bool):
    if use_real_robot:
        reader = RealRobotStateReader()
        if reader.connect():
            return reader
        print("[recorder] failed to connect real robot; falling back to mock joints")
    return MockRobotStateReader()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record data for Chapter 4.2 decoupling experiments.")
    parser.add_argument("--output", required=True, help="Recording directory, e.g. data/recordings/ch4_2_scene_A")
    parser.add_argument("--config", default="config")
    parser.add_argument("--scene", choices=["A", "B", "B2", "C"], required=True)
    parser.add_argument("--duration", type=float, default=30.0, help="Recording duration in seconds.")
    parser.add_argument("--source", choices=["realsense", "mock"], default="realsense")
    parser.add_argument("--real-robot", action="store_true", help="Read joints from the real robot SDK.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save-images", action="store_true", help="Also store color/depth arrays per frame.")
    parser.add_argument("--obstacle-desc", default="")
    args = parser.parse_args()

    camera = _make_camera(args.source, args.width, args.height, args.fps)
    state_reader = _make_state_reader(args.real_robot)
    recorder = Recorder(
        args.output,
        config_dir=args.config,
        scene=args.scene,
        obstacle_desc=args.obstacle_desc,
        save_images=args.save_images,
    )
    recorder.manifest["requested_camera"] = {
        "source": args.source,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
    }
    if hasattr(camera, "intrinsic"):
        recorder.manifest["intrinsic"] = dict(camera.intrinsic)
        if hasattr(camera, "depth_scale"):
            recorder.manifest["intrinsic"]["depth_scale"] = float(camera.depth_scale)

    print(f"[recorder] writing to {Path(args.output).resolve()}")
    print("[recorder] press Ctrl+C to stop early")
    deadline = time.time() + args.duration
    try:
        while time.time() < deadline:
            frame = camera.read()
            joints = state_reader.get_joint_positions()
            recorder.record_frame(frame, joints)
            if recorder.frame_count % 30 == 0:
                print(f"[recorder] frames={recorder.frame_count}")
    except KeyboardInterrupt:
        print("\n[recorder] interrupted")
    finally:
        recorder.close()
        camera.stop()
        if hasattr(state_reader, "disconnect"):
            state_reader.disconnect()
        print(f"[recorder] saved {recorder.frame_count} frames")


if __name__ == "__main__":
    main()
