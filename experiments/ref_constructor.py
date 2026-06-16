"""Offline reference obstacle and distance construction for Chapter 4 experiments."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from camera.pointcloud_preprocess import voxel_downsample
from experiments.decoupling_eval import CommonPreprocessor
from experiments.recorder import load_sequence
from robot.urdf_model import URDFModel
from test_clustering_filtering import FastClusteringFilter
from test_remove_robot_points_fast import RobotPointRemover
from utils.config import load_config_dir


@dataclasses.dataclass
class ReferenceFrame:
    frame_index: int
    timestamp: float
    joint_dict: dict[str, float]
    common_points: np.ndarray
    obs_points: np.ndarray
    robot_points: np.ndarray
    d_ref: float
    obs_center: np.ndarray | None
    obs_speed: float


class ReferenceConstructor:
    """Construct P_obs^ref(t) and D_ref(t) for offline evaluation only."""

    def __init__(
        self,
        config_dir: str | Path = "config",
        urdf_path: str | Path = "urdf/aubo_i16_gripper.urdf",
        bg_eps: float = 0.03,
        robot_exclusion: float = 0.035,
        voxel_size: float | None = None,
        mesh_samples: int = 50000,
        remove_planes: bool = True,
    ):
        self.config_dir = Path(config_dir)
        self.config = load_config_dir(config_dir)
        self.preprocessor = CommonPreprocessor(config_dir, voxel_size=voxel_size)
        self.urdf = URDFModel(urdf_path)
        self.remover = RobotPointRemover(self.urdf, n_samples=mesh_samples, threshold=robot_exclusion)
        self.bg_eps = float(bg_eps)
        self.robot_exclusion = float(robot_exclusion)
        self.mesh_samples = int(mesh_samples)
        self.cluster_kwargs: dict[str, Any] = {
            "eps": self.config["safety"].get("cluster_eps", 0.05),
            "min_samples": self.config["safety"].get("cluster_min_points", 25),
            "min_points": self.config["safety"].get("cluster_min_points", 25),
            "workspace": self.config["workspace"],
        }
        if remove_planes:
            self.cluster_kwargs["plane_removal"] = {
                "enabled": True,
                "distance_threshold": 0.02,
                "max_planes": 1,
                "min_plane_points": 80,
            }

    def build_empty_tree(self, empty_record_dir: str | Path | None) -> cKDTree | None:
        if empty_record_dir is None:
            return None
        points = []
        for frame in load_sequence(empty_record_dir):
            p = self.preprocessor(frame["points_cam"])
            if len(p):
                points.append(p)
        if not points:
            return None
        bg = voxel_downsample(np.vstack(points), self.preprocessor.voxel_size)
        return cKDTree(bg)

    def robot_surface(self, joint_dict: dict[str, float]) -> np.ndarray:
        fk = self.urdf.link_transforms(joint_dict)
        return self.remover._transform_to_world(fk)

    def reference_obstacle(
        self,
        common_points: np.ndarray,
        robot_points: np.ndarray,
        empty_tree: cKDTree | None,
    ) -> np.ndarray:
        if len(common_points) == 0:
            return np.empty((0, 3))

        mask = np.ones(len(common_points), dtype=bool)
        if empty_tree is not None:
            d_bg, _ = empty_tree.query(common_points, k=1)
            mask &= d_bg > self.bg_eps

        if len(robot_points):
            robot_tree = cKDTree(voxel_downsample(robot_points, max(self.robot_exclusion * 0.5, 0.005)))
            d_robot, _ = robot_tree.query(common_points, k=1)
            mask &= d_robot > self.robot_exclusion

        candidates = common_points[mask]
        if len(candidates) == 0:
            return np.empty((0, 3))

        clusters = FastClusteringFilter(candidates, robot_points, **self.cluster_kwargs).clusters
        if not clusters:
            return np.empty((0, 3))
        return np.vstack([cluster.points for cluster in clusters])

    @staticmethod
    def reference_distance(obs_points: np.ndarray, robot_points: np.ndarray) -> float:
        if len(obs_points) == 0 or len(robot_points) == 0:
            return math.inf
        tree = cKDTree(voxel_downsample(robot_points, 0.01))
        distances, _ = tree.query(obs_points, k=1)
        return float(np.min(distances))

    def build_series(
        self,
        test_record_dir: str | Path,
        empty_record_dir: str | Path | None = None,
        max_frames: int | None = None,
    ) -> list[ReferenceFrame]:
        empty_tree = self.build_empty_tree(empty_record_dir)
        output: list[ReferenceFrame] = []
        prev_center: np.ndarray | None = None
        prev_time: float | None = None

        for idx, frame in enumerate(load_sequence(test_record_dir)):
            if max_frames is not None and idx >= max_frames:
                break
            common = self.preprocessor(frame["points_cam"])
            robot = self.robot_surface(frame["joint_dict"])
            obs = self.reference_obstacle(common, robot, empty_tree)
            d_ref = self.reference_distance(obs, robot)
            center = obs.mean(axis=0) if len(obs) else None
            speed = 0.0
            if center is not None and prev_center is not None and prev_time is not None:
                dt = max(frame["timestamp"] - prev_time, 1e-6)
                speed = float(np.linalg.norm(center - prev_center) / dt)
            if center is not None:
                prev_center = center
                prev_time = frame["timestamp"]
            output.append(
                ReferenceFrame(
                    frame_index=idx,
                    timestamp=frame["timestamp"],
                    joint_dict=frame["joint_dict"],
                    common_points=common,
                    obs_points=obs,
                    robot_points=robot,
                    d_ref=d_ref,
                    obs_center=center,
                    obs_speed=speed,
                )
            )
        return output

    @staticmethod
    def danger_index(series: list[ReferenceFrame], d_stop: float) -> int | None:
        for i, frame in enumerate(series):
            if frame.d_ref <= d_stop:
                return i
        return None

    @staticmethod
    def leave_index(series: list[ReferenceFrame], d_safe: float, start: int = 0) -> int | None:
        for i in range(max(start, 0), len(series)):
            if series[i].d_ref > d_safe:
                return i
        return None


def save_reference_series(series: list[ReferenceFrame], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "frame_index": f.frame_index,
            "timestamp": f.timestamp,
            "d_ref": f.d_ref,
            "obs_count": int(len(f.obs_points)),
            "obs_center": None if f.obs_center is None else f.obs_center.tolist(),
            "obs_speed": f.obs_speed,
        }
        for f in series
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return output_path
