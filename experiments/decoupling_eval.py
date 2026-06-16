"""Evaluation utilities for Chapter 4.2 external occupancy decoupling."""
from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree

from calibration.transform_utils import load_transform_json
from camera.pointcloud_preprocess import crop_workspace, remove_invalid_points, voxel_downsample
from experiments.recorder import load_sequence
from robot.urdf_model import URDFModel
from risk.safety_policy import RiskLevel, SafetyPolicy
from test_clustering_filtering import FastClusteringFilter
from test_remove_robot_points_fast import RobotPointRemover
from utils.config import load_config_dir


METHODS = ("workspace", "ksi_like", "ours")


@dataclasses.dataclass
class MethodOutput:
    external_points: np.ndarray
    robot_points: np.ndarray
    decouple_time_ms: float
    common_points: np.ndarray


@dataclasses.dataclass
class FrameResult:
    method: str
    frame_index: int
    timestamp: float
    raw_points: int
    common_points: int
    external_points: int
    robot_points: int
    cluster_count: int
    cluster_centers: list[list[float]]
    safety_state: str
    min_distance: float
    decouple_time_ms: float
    raw_in_omega: int = 0
    external_in_omega: int = 0
    detected_in_omega: bool = False
    raw_in_omegas: list[int] = dataclasses.field(default_factory=list)
    external_in_omegas: list[int] = dataclasses.field(default_factory=list)
    detected_in_omegas: list[bool] = dataclasses.field(default_factory=list)


def parse_omega(omega: str | Iterable[float] | None) -> np.ndarray | None:
    if omega is None:
        return None
    if isinstance(omega, str):
        values = [float(v.strip()) for v in omega.split(",") if v.strip()]
    else:
        values = [float(v) for v in omega]
    if len(values) != 6:
        raise ValueError("omega must contain 6 values: x_min,x_max,y_min,y_max,z_min,z_max")
    return np.asarray(values, dtype=float)


def parse_omegas(omegas: str | Iterable[Iterable[float]] | None) -> list[np.ndarray]:
    """Parse one or more obstacle AABBs.

    CLI format: "x0,x1,y0,y1,z0,z1;x0,x1,y0,y1,z0,z1".
    """
    if omegas is None:
        return []
    if isinstance(omegas, str):
        parsed = []
        for part in omegas.split(";"):
            part = part.strip()
            if part:
                item = parse_omega(part)
                if item is not None:
                    parsed.append(item)
        return parsed
    parsed = []
    for omega in omegas:
        item = parse_omega(omega)
        if item is not None:
            parsed.append(item)
    return parsed


def points_in_omega(points: np.ndarray, omega: np.ndarray | None) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if omega is None or len(points) == 0:
        return np.zeros(len(points), dtype=bool)
    x0, x1, y0, y1, z0, z1 = omega
    return (
        (points[:, 0] >= x0)
        & (points[:, 0] <= x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] <= y1)
        & (points[:, 2] >= z0)
        & (points[:, 2] <= z1)
    )


class CommonPreprocessor:
    """Transform camera points into base frame, crop workspace, and voxel downsample."""

    def __init__(
        self,
        config_dir: str | Path,
        voxel_size: float | None = None,
        max_raw_points: int | None = 100000,
    ):
        config = load_config_dir(config_dir)
        self.workspace = config["workspace"]
        self.voxel_size = float(voxel_size if voxel_size is not None else self.workspace.get("voxel_size", 0.02))
        self.max_raw_points = max_raw_points
        self.extrinsic = load_transform_json(Path(config_dir) / "camera_extrinsic.json")
        self.R = self.extrinsic[:3, :3]
        self.t = self.extrinsic[:3, 3]

    def __call__(self, points_cam: np.ndarray) -> np.ndarray:
        points = remove_invalid_points(points_cam)
        if len(points) == 0:
            return points.reshape(0, 3)
        if self.max_raw_points is not None and self.max_raw_points > 0 and len(points) > self.max_raw_points:
            # Deterministic thinning keeps replay results repeatable while avoiding
            # 1280x720 full-cloud memory spikes during offline evaluation.
            idx = np.linspace(0, len(points) - 1, int(self.max_raw_points)).astype(np.int64)
            points = points[idx]
        points_base = points @ self.R.T + self.t
        cropped = crop_workspace(points_base, self.workspace)
        return voxel_downsample(cropped, self.voxel_size)


class BaseMethod:
    name = "base"

    def filter(self, points_cam: np.ndarray, joint_dict: dict[str, float]) -> MethodOutput:
        raise NotImplementedError


class WorkspaceMethod(BaseMethod):
    name = "workspace"

    def __init__(self, preprocessor: CommonPreprocessor):
        self.preprocessor = preprocessor

    def filter(self, points_cam: np.ndarray, joint_dict: dict[str, float]) -> MethodOutput:
        common = self.preprocessor(points_cam)
        return MethodOutput(common, np.empty((0, 3)), 0.0, common)


class MeshRobotMixin:
    def __init__(
        self,
        preprocessor: CommonPreprocessor,
        urdf_path: str | Path,
        threshold: float,
        mesh_samples: int = 50000,
    ):
        self.preprocessor = preprocessor
        self.urdf = URDFModel(urdf_path)
        self.remover = RobotPointRemover(
            self.urdf,
            n_samples=mesh_samples,
            threshold=threshold,
            process_interval=1,
        )
        self.threshold = threshold

    def robot_surface(self, joint_dict: dict[str, float]) -> np.ndarray:
        fk = self.urdf.link_transforms(joint_dict)
        return self.remover._transform_to_world(fk)


class KSILikeMethod(MeshRobotMixin, BaseMethod):
    """Cluster-level kinematic self-identification baseline.

    This intentionally removes whole clusters using robot-model matching. It is
    weaker than Ours because it cannot keep the non-robot part of a mixed cluster.
    """

    name = "ksi_like"

    def __init__(
        self,
        preprocessor: CommonPreprocessor,
        urdf_path: str | Path,
        match_margin: float = 0.06,
        near_ratio: float = 0.25,
        mesh_samples: int = 30000,
        cluster_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(preprocessor, urdf_path, threshold=match_margin, mesh_samples=mesh_samples)
        self.match_margin = match_margin
        self.near_ratio = near_ratio
        self.cluster_kwargs = cluster_kwargs or {}

    def filter(self, points_cam: np.ndarray, joint_dict: dict[str, float]) -> MethodOutput:
        common = self.preprocessor(points_cam)
        t0 = time.perf_counter()
        robot_pts = self.robot_surface(joint_dict)
        if len(common) == 0 or len(robot_pts) == 0:
            return MethodOutput(common, robot_pts, (time.perf_counter() - t0) * 1000.0, common)

        cluster_result = FastClusteringFilter(common, robot_pts, **self.cluster_kwargs)
        tree = cKDTree(voxel_downsample(robot_pts, max(self.match_margin * 0.5, 0.005)))
        external_clusters = []
        robot_clusters = []
        for cluster in cluster_result.clusters:
            distances, _ = tree.query(cluster.points, k=1)
            min_dist = float(distances.min()) if len(distances) else math.inf
            near = float(np.mean(distances < self.match_margin)) if len(distances) else 0.0
            if min_dist < self.match_margin or near >= self.near_ratio:
                robot_clusters.append(cluster.points)
            else:
                external_clusters.append(cluster.points)

        # Keep DBSCAN noise as external sparse points; only stable robot-like clusters are removed.
        if len(cluster_result.noise_points):
            external_clusters.append(cluster_result.noise_points)
        external = np.vstack(external_clusters) if external_clusters else np.empty((0, 3))
        dec_ms = (time.perf_counter() - t0) * 1000.0
        robot_like = np.vstack(robot_clusters) if robot_clusters else np.empty((0, 3))
        return MethodOutput(external, robot_like if len(robot_like) else robot_pts, dec_ms, common)


class OursMethod(MeshRobotMixin, BaseMethod):
    name = "ours"

    def filter(self, points_cam: np.ndarray, joint_dict: dict[str, float]) -> MethodOutput:
        common = self.preprocessor(points_cam)
        t0 = time.perf_counter()
        fk = self.urdf.link_transforms(joint_dict)
        external, robot_pts = self.remover.remove(common, fk)
        dec_ms = (time.perf_counter() - t0) * 1000.0
        return MethodOutput(external, robot_pts, dec_ms, common)


class DecouplingEvaluator:
    def __init__(
        self,
        config_dir: str | Path = "config",
        urdf_path: str | Path = "urdf/aubo_i16_gripper.urdf",
        delta_r: float = 0.05,
        delta_eval: float = 0.10,
        voxel_size: float | None = None,
        max_raw_points: int | None = 100000,
        mesh_samples: int = 50000,
        remove_planes: bool = False,
    ):
        self.config_dir = Path(config_dir)
        self.config = load_config_dir(config_dir)
        self.preprocessor = CommonPreprocessor(config_dir, voxel_size=voxel_size, max_raw_points=max_raw_points)
        self.urdf_path = Path(urdf_path)
        self.delta_r = float(delta_r)
        self.delta_eval = float(delta_eval)
        self.mesh_samples = int(mesh_samples)
        self.safety = SafetyPolicy(
            d_safe=self.config["safety"].get("d_safe", 0.15),
            d_slow=self.config["safety"].get("d_slow", 0.10),
            d_stop=self.config["safety"].get("d_stop", 0.05),
        )
        self.cluster_kwargs = {
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

    def build_method(self, method: str) -> BaseMethod:
        if method == "workspace":
            return WorkspaceMethod(self.preprocessor)
        if method == "ksi_like":
            return KSILikeMethod(
                self.preprocessor,
                self.urdf_path,
                match_margin=max(self.delta_r, 0.04),
                mesh_samples=max(self.mesh_samples // 2, 5000),
                cluster_kwargs=self.cluster_kwargs,
            )
        if method == "ours":
            return OursMethod(
                self.preprocessor,
                self.urdf_path,
                threshold=self.delta_r,
                mesh_samples=self.mesh_samples,
            )
        raise ValueError(f"unknown method: {method}")

    def evaluate_recording(
        self,
        record_dir: str | Path,
        methods: Iterable[str] = METHODS,
        scene: str | None = None,
        omega: np.ndarray | None = None,
        omegas: list[np.ndarray] | None = None,
        n_min_obj: int = 30,
    ) -> dict[str, Any]:
        obstacle_omegas = list(omegas or ([] if omega is None else [omega]))
        method_objects = {name: self.build_method(name) for name in methods}
        per_frame: list[FrameResult] = []
        metric_inputs: dict[str, list[FrameResult]] = {name: [] for name in method_objects}
        residual_counts: dict[str, list[tuple[int, int]]] = {name: [] for name in method_objects}

        # A separate high-density surface model for evaluation denominators.
        eval_urdf = URDFModel(self.urdf_path)
        eval_remover = RobotPointRemover(eval_urdf, n_samples=self.mesh_samples, threshold=self.delta_r)

        seen_frames = 0
        for frame_index, frame in enumerate(load_sequence(record_dir)):
            seen_frames += 1
            if scene is None:
                scene = frame["manifest"].get("scene", "A")
            joint_dict = frame["joint_dict"]
            eval_robot_pts = eval_remover._transform_to_world(eval_urdf.link_transforms(joint_dict))
            eval_tree = cKDTree(voxel_downsample(eval_robot_pts, max(self.delta_eval * 0.5, 0.005)))

            for method_name, method in method_objects.items():
                output = method.filter(frame["points_cam"], joint_dict)
                clusters = FastClusteringFilter(output.external_points, eval_robot_pts, **self.cluster_kwargs).clusters
                min_dist = self._min_cluster_robot_distance(clusters, eval_tree)
                decision = self.safety.evaluate(min_dist)

                raw_in_omegas: list[int] = []
                external_in_omegas: list[int] = []
                detected_in_omegas: list[bool] = []
                for item in obstacle_omegas:
                    raw_in_omegas.append(int(points_in_omega(output.common_points, item).sum()))
                    external_in_omegas.append(int(points_in_omega(output.external_points, item).sum()))
                    detected_in_omegas.append(any(self._point_in_omega(cluster.center, item) for cluster in clusters))
                raw_in_omega = raw_in_omegas[0] if raw_in_omegas else 0
                external_in_omega = external_in_omegas[0] if external_in_omegas else 0
                detected_in_omega = detected_in_omegas[0] if detected_in_omegas else False

                near_common = self._count_near_robot(output.common_points, eval_tree, self.delta_eval)
                near_external = self._count_near_robot(output.external_points, eval_tree, self.delta_eval)
                residual_counts[method_name].append((near_external, near_common))

                row = FrameResult(
                    method=method_name,
                    frame_index=frame_index,
                    timestamp=frame["timestamp"],
                    raw_points=len(frame["points_cam"]),
                    common_points=len(output.common_points),
                    external_points=len(output.external_points),
                    robot_points=len(output.robot_points),
                    cluster_count=len(clusters),
                    cluster_centers=[cluster.center.tolist() for cluster in clusters],
                    safety_state=decision.level.value,
                    min_distance=float(decision.min_distance),
                    decouple_time_ms=float(output.decouple_time_ms),
                    raw_in_omega=raw_in_omega,
                    external_in_omega=external_in_omega,
                    detected_in_omega=detected_in_omega,
                    raw_in_omegas=raw_in_omegas,
                    external_in_omegas=external_in_omegas,
                    detected_in_omegas=detected_in_omegas,
                )
                per_frame.append(row)
                metric_inputs[method_name].append(row)

        if seen_frames == 0:
            raise RuntimeError(f"no recorded frames found in {record_dir}")
        scene = scene or "A"
        metrics = {
            method: self._aggregate(rows, residual_counts[method], scene, obstacle_omegas, n_min_obj)
            for method, rows in metric_inputs.items()
        }
        return {
            "record_dir": str(record_dir),
            "scene": scene,
            "delta_r": self.delta_r,
            "delta_eval": self.delta_eval,
            "methods": list(method_objects),
            "omegas": [item.tolist() for item in obstacle_omegas],
            "metrics": metrics,
            "per_frame": [dataclasses.asdict(row) for row in per_frame],
        }

    @staticmethod
    def _point_in_omega(point: np.ndarray, omega: np.ndarray) -> bool:
        x0, x1, y0, y1, z0, z1 = omega
        x, y, z = point
        return bool(x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1)

    @staticmethod
    def _count_near_robot(points: np.ndarray, tree: cKDTree, threshold: float) -> int:
        if len(points) == 0:
            return 0
        distances, _ = tree.query(points, k=1)
        return int(np.count_nonzero(distances < threshold))

    @staticmethod
    def _min_cluster_robot_distance(clusters: list[Any], tree: cKDTree) -> float:
        best = math.inf
        for cluster in clusters:
            if len(cluster.points) == 0:
                continue
            distances, _ = tree.query(cluster.points, k=1)
            best = min(best, float(distances.min()))
        return best

    @staticmethod
    def _aggregate(
        rows: list[FrameResult],
        residual_counts: list[tuple[int, int]],
        scene: str,
        omegas: list[np.ndarray],
        n_min_obj: int,
    ) -> dict[str, float | int | None]:
        out: dict[str, float | int | None] = {
            "frames": len(rows),
            "T_dec_ms_mean": float(np.mean([r.decouple_time_ms for r in rows])) if rows else None,
            "T_dec_ms_std": float(np.std([r.decouple_time_ms for r in rows])) if rows else None,
        }

        if scene == "A":
            out["N_res"] = float(np.mean([r.cluster_count for r in rows])) if rows else None
            near_ext = sum(a for a, _ in residual_counts)
            near_common = sum(b for _, b in residual_counts)
            out["R_res"] = float(near_ext / near_common) if near_common > 0 else None
            false = [
                r.safety_state
                for r in rows
                if r.safety_state != RiskLevel.SAFE.value
            ]
            out["R_false"] = float(len(false) / len(rows)) if rows else None

        if scene in {"B", "B2"} and omegas:
            out["obstacle_count"] = len(omegas)
            raw_total_all = 0
            keep_total_all = 0
            det_hits_all = 0
            visible_pairs = 0
            sigmas = []

            for idx, omega in enumerate(omegas):
                visible = [
                    r
                    for r in rows
                    if len(r.raw_in_omegas) > idx and r.raw_in_omegas[idx] > n_min_obj
                ]
                raw_total = sum(r.raw_in_omegas[idx] for r in visible)
                keep_total = sum(r.external_in_omegas[idx] for r in visible)
                r_keep = float(keep_total / raw_total) if raw_total > 0 else None
                det_hits = sum(r.detected_in_omegas[idx] for r in visible)
                r_det = float(det_hits / len(visible)) if visible else None
                centers = [
                    np.asarray(center, dtype=float)
                    for r in visible
                    if r.detected_in_omegas[idx]
                    for center in r.cluster_centers
                    if DecouplingEvaluator._point_in_omega(np.asarray(center, dtype=float), omega)
                ]
                sigma = None
                if centers:
                    c = np.vstack(centers)
                    mean = c.mean(axis=0)
                    sigma = float(np.sqrt(np.mean(np.sum((c - mean) ** 2, axis=1))))
                    sigmas.append(sigma)

                suffix = f"_{idx + 1}"
                out[f"visible_frames{suffix}"] = len(visible)
                out[f"R_keep{suffix}"] = r_keep
                out[f"R_over{suffix}"] = float(1.0 - r_keep) if r_keep is not None else None
                out[f"R_det{suffix}"] = r_det
                out[f"sigma_c{suffix}"] = sigma

                raw_total_all += raw_total
                keep_total_all += keep_total
                det_hits_all += det_hits
                visible_pairs += len(visible)

            r_keep_all = float(keep_total_all / raw_total_all) if raw_total_all > 0 else None
            out["visible_pairs"] = visible_pairs
            out["visible_frames"] = visible_pairs
            out["R_keep"] = r_keep_all
            out["R_over"] = float(1.0 - r_keep_all) if r_keep_all is not None else None
            out["R_det"] = float(det_hits_all / visible_pairs) if visible_pairs else None
            out["sigma_c"] = float(np.mean(sigmas)) if sigmas else None

        return out


def save_results(result: dict[str, Any], output_dir: str | Path, stem: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return out_path


def markdown_table(metrics: dict[str, Any], scene: str) -> str:
    if scene == "A":
        headers = ["方法", "N_res ↓", "R_res ↓", "R_false ↓", "T_dec(ms) ↓"]
        rows = [
            [
                method,
                _fmt(vals.get("N_res")),
                _fmt(vals.get("R_res")),
                _fmt(vals.get("R_false")),
                _fmt(vals.get("T_dec_ms_mean")),
            ]
            for method, vals in metrics.items()
        ]
    elif scene in {"B", "B2"}:
        headers = ["方法", "R_keep ↑", "R_det ↑", "sigma_c ↓", "R_over ↓", "T_dec(ms) ↓"]
        rows = [
            [
                method,
                _fmt(vals.get("R_keep")),
                _fmt(vals.get("R_det")),
                _fmt(vals.get("sigma_c")),
                _fmt(vals.get("R_over")),
                _fmt(vals.get("T_dec_ms_mean")),
            ]
            for method, vals in metrics.items()
        ]
    else:
        headers = ["方法", "frames", "T_dec(ms)"]
        rows = [[method, str(vals.get("frames", "")), _fmt(vals.get("T_dec_ms_mean"))] for method, vals in metrics.items()]

    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([line, sep, *body])


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.4f}"
    return str(value)
