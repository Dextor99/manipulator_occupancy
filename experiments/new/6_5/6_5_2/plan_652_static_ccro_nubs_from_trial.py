#!/usr/bin/env python3
"""Plan a real-obstacle static CCRO-NUBS candidate in joint space.

This is the 6.5.2 path that matches the paper method: NUBS interpolation
configurations in 6-DOF joint space plus full-body CCRO mesh risk.  It is not a
B-spline/Bezier Cartesian detour.

The script reads an existing live-perception trial, uses its observed obstacle
point cloud and q_start, then optimizes a joint-space NUBS trajectory to an
operator-provided or estimated q_goal.  No robot command is sent.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from experiments.exp_ccro_stage2 import _baseline, _limits, _load, _risk_optimizer  # noqa: E402
from planning.mesh_risk import MeshRiskEvaluator, StaticObstacleField  # noqa: E402
from planning.nubs_trajectory import NUBSTrajectory6D  # noqa: E402
from planning.robot_surface_model import RobotSurfaceModel  # noqa: E402
from planning.static_optimizer import StaticRiskNUBSOptimizer  # noqa: E402
from planning.verifier import TrajectoryVerifier  # noqa: E402
from preview_652_candidate_ik_urdf import plot_pose_sequence as plot_urdf_pose_sequence  # noqa: E402
from run_652_static_avoidance import (  # noqa: E402
    DEFAULT_CONFIG,
    make_evaluator_and_verifier,
    make_surface_model,
    save_distance_curve,
    save_joint_preview,
    sample_trajectory_distances,
    trajectory_rows,
    write_csv,
    write_json,
)


DEFAULT_TRIAL = (
    ROOT
    / "results"
    / "new"
    / "6_5"
    / "6_5_2"
    / "planar_static_live"
    / "rs1_lateral_table_obstacle"
    / "trials"
    / "rs1_lateral_table_obstacle_r05"
)

# Estimated from the earlier low-speed Y=-0.4 smoke execution.  This should be
# replaced by a fresh goal-state capture before formal execution.
DEFAULT_Q_GOAL_RAD = "-0.36184728145599365,-0.22320318222045898,1.315380573272705,-0.03216493874788284,1.5707743167877197,-0.3618289530277252"


class TabletopPreferenceStaticRiskNUBSOptimizer(StaticRiskNUBSOptimizer):
    """Static CCRO-NUBS optimizer with soft task-space minimal-change terms."""

    def __init__(
        self,
        *args,
        surface_model: RobotSurfaceModel,
        reference: NUBSTrajectory6D,
        tcp_link: str,
        lambda_tcp_z: float,
        tcp_z_tolerance_m: float,
        lambda_tcp_xy: float,
        tcp_xy_tolerance_m: float,
        lambda_joint_deviation: float,
        joint_deviation_tolerance_rad: float,
        tcp_preference_samples: int,
        route_family: str,
        lambda_route_corridor: float,
        route_corridor_margin_m: float,
        route_corridor_influence_m: float,
        lambda_side_z_corridor: float,
        side_z_tolerance_m: float,
        obstacle_points: np.ndarray,
        table_z_m: float,
        clearance_m: float,
        vertical_uncertainty_m: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.surface_model = surface_model
        self.tcp_link = tcp_link
        self.lambda_tcp_z = float(lambda_tcp_z)
        self.tcp_z_tolerance_m = float(max(tcp_z_tolerance_m, 0.0))
        self.lambda_tcp_xy = float(lambda_tcp_xy)
        self.tcp_xy_tolerance_m = float(max(tcp_xy_tolerance_m, 0.0))
        self.lambda_joint_deviation = float(lambda_joint_deviation)
        self.joint_deviation_tolerance_rad = float(max(joint_deviation_tolerance_rad, 0.0))
        self.route_family = str(route_family)
        self.lambda_route_corridor = float(lambda_route_corridor)
        self.route_corridor_margin_m = float(route_corridor_margin_m)
        self.route_corridor_influence_m = float(route_corridor_influence_m)
        self.lambda_side_z_corridor = float(lambda_side_z_corridor)
        self.side_z_tolerance_m = float(max(side_z_tolerance_m, 0.0))
        self.clearance_m = float(clearance_m)
        self.vertical_uncertainty_m = float(vertical_uncertainty_m)
        self.obstacle_points = np.asarray(obstacle_points, dtype=np.float64)
        self.obstacle_xy_center = np.mean(self.obstacle_points[:, :2], axis=0)
        self.obstacle_xy_radius = float(
            np.percentile(
                np.linalg.norm(self.obstacle_points[:, :2] - self.obstacle_xy_center[None, :], axis=1),
                95,
            )
        )
        base_vec = -self.obstacle_xy_center
        self.base_side_direction = base_vec / max(float(np.linalg.norm(base_vec)), 1.0e-9)
        self.outer_side_direction = -self.base_side_direction
        self.obstacle_top_p99_m = float(np.percentile(self.obstacle_points[:, 2], 99))
        self.table_z_m = float(table_z_m)
        count = max(2, int(tcp_preference_samples))
        self.tcp_preference_times = np.linspace(0.0, reference.total_duration, count)
        self.reference_q = reference.sample(self.tcp_preference_times, max_derivative=0).q
        self.reference_tcp_xyz = np.asarray(
            [self._tcp_xyz(q) for q in self.reference_q],
            dtype=np.float64,
        )
        self.reference_tcp_z = np.asarray(
            self.reference_tcp_xyz[:, 2],
            dtype=np.float64,
        )
        self.corridor_mask = self._make_corridor_mask()

    def _joint_dict(self, q: np.ndarray) -> dict[str, float]:
        return {name: float(q[i]) for i, name in enumerate(self.surface_model.joint_names)}

    def _tcp_z(self, q: np.ndarray) -> float:
        return float(self._tcp_xyz(q)[2])

    def _tcp_xyz(self, q: np.ndarray) -> np.ndarray:
        fk = self.surface_model.urdf.link_transforms(self._joint_dict(q))
        return np.asarray(fk[self.tcp_link][:3, 3], dtype=np.float64)

    def tcp_z_preference_cost(self, flat_points: np.ndarray) -> float:
        if self.lambda_tcp_z <= 0.0:
            return 0.0
        points = np.asarray(flat_points, dtype=np.float64).reshape(self.inner_shape)
        trajectory = self._trajectory(points)
        samples = trajectory.sample(self.tcp_preference_times, max_derivative=0).q
        z = np.asarray([self._tcp_z(q) for q in samples], dtype=np.float64)
        dz = np.abs(z - self.reference_tcp_z)
        hinge = np.maximum(dz - self.tcp_z_tolerance_m, 0.0)
        return float(np.mean(hinge * hinge))

    def tcp_xy_preference_cost(self, flat_points: np.ndarray) -> float:
        if self.lambda_tcp_xy <= 0.0:
            return 0.0
        points = np.asarray(flat_points, dtype=np.float64).reshape(self.inner_shape)
        trajectory = self._trajectory(points)
        samples = trajectory.sample(self.tcp_preference_times, max_derivative=0).q
        xyz = np.asarray([self._tcp_xyz(q) for q in samples], dtype=np.float64)
        deviation = np.linalg.norm(xyz[:, :2] - self.reference_tcp_xyz[:, :2], axis=1)
        hinge = np.maximum(deviation - self.tcp_xy_tolerance_m, 0.0)
        return float(np.mean(hinge * hinge))

    def joint_deviation_preference_cost(self, flat_points: np.ndarray) -> float:
        if self.lambda_joint_deviation <= 0.0:
            return 0.0
        points = np.asarray(flat_points, dtype=np.float64).reshape(self.inner_shape)
        trajectory = self._trajectory(points)
        samples = trajectory.sample(self.tcp_preference_times, max_derivative=0).q
        deviation = np.linalg.norm(samples - self.reference_q, axis=1)
        hinge = np.maximum(deviation - self.joint_deviation_tolerance_rad, 0.0)
        return float(np.mean(hinge * hinge))

    def _make_corridor_mask(self) -> np.ndarray:
        distances = np.linalg.norm(
            self.reference_tcp_xyz[:, :2] - self.obstacle_xy_center[None, :],
            axis=1,
        )
        threshold = self.obstacle_xy_radius + self.route_corridor_influence_m
        mask = distances <= threshold
        if not np.any(mask):
            center = int(np.argmin(distances))
            mask[max(0, center - 2): min(len(mask), center + 3)] = True
        return mask

    def route_corridor_cost(self, flat_points: np.ndarray) -> float:
        route_enabled = self.lambda_route_corridor > 0.0 and self.route_family != "none"
        side_z_enabled = (
            self.lambda_side_z_corridor > 0.0
            and self.route_family in {"base_side", "outer_side"}
        )
        if not route_enabled and not side_z_enabled:
            return 0.0
        points = np.asarray(flat_points, dtype=np.float64).reshape(self.inner_shape)
        trajectory = self._trajectory(points)
        samples = trajectory.sample(self.tcp_preference_times, max_derivative=0).q
        xyz = np.asarray([self._tcp_xyz(q) for q in samples], dtype=np.float64)
        active = xyz[self.corridor_mask]
        if len(active) == 0:
            return 0.0
        cost = 0.0
        if self.route_family in {"base_side", "outer_side"}:
            if route_enabled:
                direction = self.base_side_direction if self.route_family == "base_side" else self.outer_side_direction
                required = self.obstacle_xy_radius + self.route_corridor_margin_m
                signed = (active[:, :2] - self.obstacle_xy_center[None, :]) @ direction
                violation = np.maximum(required - signed, 0.0)
                cost += self.lambda_route_corridor * float(np.mean(violation * violation))
            if side_z_enabled:
                ref_active = self.reference_tcp_xyz[self.corridor_mask]
                z_deviation = np.abs(active[:, 2] - ref_active[:, 2])
                z_violation = np.maximum(z_deviation - self.side_z_tolerance_m, 0.0)
                cost += self.lambda_side_z_corridor * float(np.mean(z_violation * z_violation))
            return float(cost)
        if self.route_family == "overpass":
            if not route_enabled:
                return 0.0
            bbox_min = np.min(self.obstacle_points[:, :2], axis=0) - self.route_corridor_margin_m
            bbox_max = np.max(self.obstacle_points[:, :2], axis=0) + self.route_corridor_margin_m
            inside = (
                (active[:, 0] >= bbox_min[0])
                & (active[:, 0] <= bbox_max[0])
                & (active[:, 1] >= bbox_min[1])
                & (active[:, 1] <= bbox_max[1])
            )
            if not np.any(inside):
                return 0.0
            required_z = self.obstacle_top_p99_m + self.clearance_m + self.vertical_uncertainty_m
            violation = np.maximum(required_z - active[inside, 2], 0.0)
            return float(self.lambda_route_corridor * np.mean(violation * violation))
        return 0.0

    def route_corridor_report(self, trajectory: NUBSTrajectory6D) -> dict[str, Any]:
        samples = trajectory.sample(self.tcp_preference_times, max_derivative=0).q
        xyz = np.asarray([self._tcp_xyz(q) for q in samples], dtype=np.float64)
        active = xyz[self.corridor_mask]
        report: dict[str, Any] = {
            "route_family": self.route_family,
            "enabled": bool(self.lambda_route_corridor > 0.0 and self.route_family != "none"),
            "lambda_route_corridor": self.lambda_route_corridor,
            "route_corridor_margin_m": self.route_corridor_margin_m,
            "route_corridor_influence_m": self.route_corridor_influence_m,
            "active_sample_count": int(len(active)),
            "obstacle_xy_center": self.obstacle_xy_center.tolist(),
            "obstacle_xy_radius_p95_m": self.obstacle_xy_radius,
            "accepted": True,
            "max_violation_m": 0.0,
        }
        if not report["enabled"] or len(active) == 0:
            return report
        if self.route_family in {"base_side", "outer_side"}:
            direction = self.base_side_direction if self.route_family == "base_side" else self.outer_side_direction
            required = self.obstacle_xy_radius + self.route_corridor_margin_m
            signed = (active[:, :2] - self.obstacle_xy_center[None, :]) @ direction
            violation = np.maximum(required - signed, 0.0)
            ref_active = self.reference_tcp_xyz[self.corridor_mask]
            z_deviation = np.abs(active[:, 2] - ref_active[:, 2])
            z_violation = np.maximum(z_deviation - self.side_z_tolerance_m, 0.0)
            side_z_enabled = self.lambda_side_z_corridor > 0.0
            side_z_accepted = bool((not side_z_enabled) or np.max(z_violation) <= 1.0e-3)
            report.update(
                {
                    "direction_xy": direction.tolist(),
                    "required_signed_distance_m": float(required),
                    "min_signed_distance_m": float(np.min(signed)),
                    "max_lateral_violation_m": float(np.max(violation)),
                    "lambda_side_z_corridor": self.lambda_side_z_corridor,
                    "side_z_tolerance_m": self.side_z_tolerance_m,
                    "max_side_z_deviation_m": float(np.max(z_deviation)),
                    "max_side_z_violation_m": float(np.max(z_violation)),
                    "side_z_accepted": side_z_accepted,
                    "max_violation_m": float(max(np.max(violation), np.max(z_violation))),
                    "accepted": bool(np.max(violation) <= 1.0e-3 and side_z_accepted),
                }
            )
        elif self.route_family == "overpass":
            bbox_min = np.min(self.obstacle_points[:, :2], axis=0) - self.route_corridor_margin_m
            bbox_max = np.max(self.obstacle_points[:, :2], axis=0) + self.route_corridor_margin_m
            inside = (
                (active[:, 0] >= bbox_min[0])
                & (active[:, 0] <= bbox_max[0])
                & (active[:, 1] >= bbox_min[1])
                & (active[:, 1] <= bbox_max[1])
            )
            required_z = self.obstacle_top_p99_m + self.clearance_m + self.vertical_uncertainty_m
            if np.any(inside):
                violation = np.maximum(required_z - active[inside, 2], 0.0)
                report.update(
                    {
                        "required_z_m": float(required_z),
                        "inside_footprint_sample_count": int(np.count_nonzero(inside)),
                        "min_z_inside_m": float(np.min(active[inside, 2])),
                        "max_violation_m": float(np.max(violation)),
                        "accepted": bool(np.max(violation) <= 1.0e-3),
                    }
                )
            else:
                report.update({"required_z_m": float(required_z), "inside_footprint_sample_count": 0})
        return report

    def preference_cost(self, flat_points: np.ndarray) -> float:
        return float(
            self.lambda_tcp_z * self.tcp_z_preference_cost(flat_points)
            + self.lambda_tcp_xy * self.tcp_xy_preference_cost(flat_points)
            + self.lambda_joint_deviation * self.joint_deviation_preference_cost(flat_points)
            + self.route_corridor_cost(flat_points)
        )

    def _preference_gradient(self, flat_points: np.ndarray, base_cost: float) -> np.ndarray:
        if (
            flat_points.size == 0
            or (
                self.lambda_tcp_z <= 0.0
                and self.lambda_tcp_xy <= 0.0
                and self.lambda_joint_deviation <= 0.0
                and self.lambda_route_corridor <= 0.0
                and self.lambda_side_z_corridor <= 0.0
            )
        ):
            return np.zeros_like(flat_points)
        grad = np.zeros_like(flat_points)
        for idx in range(flat_points.size):
            plus = flat_points.copy()
            minus = flat_points.copy()
            plus[idx] += self.fd_epsilon
            minus[idx] -= self.fd_epsilon
            grad[idx] = (
                self.preference_cost(plus) - self.preference_cost(minus)
            ) / (2.0 * self.fd_epsilon)
        return grad

    def cost_only(self, flat_points: np.ndarray) -> float:
        base = super().cost_only(flat_points)
        pref = self.preference_cost(np.asarray(flat_points, dtype=np.float64))
        return float(base + pref)

    def objective(self, flat_points: np.ndarray) -> tuple[float, np.ndarray]:
        cost, grad = super().objective(flat_points)
        flat = np.asarray(flat_points, dtype=np.float64)
        pref = self.preference_cost(flat)
        pref_grad = self._preference_gradient(flat, pref)
        return float(cost + pref), grad + pref_grad


def make_tabletop_optimizer(
    config: dict[str, Any],
    head: np.ndarray,
    tail: np.ndarray,
    durations: np.ndarray,
    limits,
    evaluator: MeshRiskEvaluator,
    obstacle: StaticObstacleField,
    reference: NUBSTrajectory6D,
    surface_model: RobotSurfaceModel,
    args: argparse.Namespace,
):
    opt = config["optimizer"]
    risk = config["risk"]
    return TabletopPreferenceStaticRiskNUBSOptimizer(
        head,
        tail,
        durations,
        limits,
        evaluator,
        obstacle,
        lambda_risk=opt["lambda_risk"],
        risk_samples_per_segment=risk["risk_samples_per_segment"],
        risk_links=None,
        sensitivity_epsilon=opt["sensitivity_epsilon"],
        lambda_smooth=opt["lambda_smooth"],
        lambda_position=opt["lambda_position"],
        lambda_velocity=opt["lambda_velocity"],
        lambda_acceleration=opt["lambda_acceleration"],
        samples_per_segment=opt["samples_per_segment"],
        finite_difference_epsilon=opt["finite_difference_epsilon"],
        max_iterations=opt["max_iterations"],
        gradient_tolerance=opt["gradient_tolerance"],
        surface_model=surface_model,
        reference=reference,
        tcp_link=args.tcp_link,
        lambda_tcp_z=args.lambda_tcp_z,
        tcp_z_tolerance_m=args.tcp_z_tolerance_m,
        lambda_tcp_xy=args.lambda_tcp_xy,
        tcp_xy_tolerance_m=args.tcp_xy_tolerance_m,
        lambda_joint_deviation=args.lambda_joint_deviation,
        joint_deviation_tolerance_rad=args.joint_deviation_tolerance_rad,
        tcp_preference_samples=args.tcp_preference_samples,
        route_family=args.route_family,
        lambda_route_corridor=args.lambda_route_corridor,
        route_corridor_margin_m=args.route_corridor_margin_m,
        route_corridor_influence_m=args.route_corridor_influence_m,
        lambda_side_z_corridor=args.lambda_side_z_corridor,
        side_z_tolerance_m=args.side_z_tolerance_m,
        obstacle_points=args._obstacle_points_for_corridor,
        table_z_m=args._table_z_for_corridor,
        clearance_m=args.clearance_m,
        vertical_uncertainty_m=args.vertical_uncertainty_m,
    )


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def parse_vector(value: str, *, name: str) -> np.ndarray:
    arr = np.asarray([float(item.strip()) for item in value.split(",") if item.strip()], dtype=np.float64)
    if arr.shape != (6,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain six finite comma-separated values")
    return arr


def load_trial_q_start(trial_dir: Path) -> np.ndarray:
    summary = json.loads((trial_dir / "summary.json").read_text(encoding="utf-8"))
    q = summary.get("q_start_mean_rad") or summary.get("obstacle_model", {}).get("q_mean")
    arr = np.asarray(q, dtype=np.float64)
    if arr.shape != (6,) or not np.all(np.isfinite(arr)):
        raise RuntimeError(f"cannot load q_start from {trial_dir / 'summary.json'}")
    return arr


def load_initial_inner_points(initial_plan_dir: Path | None, expected_shape: tuple[int, int]) -> np.ndarray | None:
    if initial_plan_dir is None:
        return None
    data_path = initial_plan_dir / "ccro_nubs_trajectories.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"missing initial plan data: {data_path}")
    data = np.load(data_path)
    points = np.asarray(data["candidate_inner"], dtype=np.float64)
    if points.shape != expected_shape or not np.all(np.isfinite(points)):
        raise RuntimeError(
            f"initial candidate_inner has shape {points.shape}, expected {expected_shape}"
        )
    return points


def sample_rows(name: str, trajectory: NUBSTrajectory6D, dt: float) -> list[dict[str, Any]]:
    return trajectory_rows(name, trajectory, dt)


def tcp_path_z_stats(surface_model: RobotSurfaceModel, trajectory: NUBSTrajectory6D, tcp_link: str, samples: int = 241) -> dict[str, Any]:
    times = np.linspace(0.0, trajectory.total_duration, max(2, int(samples)))
    z_values = []
    for t in times:
        q = trajectory.evaluate(float(t))
        joints = {name: float(q[i]) for i, name in enumerate(surface_model.joint_names)}
        fk = surface_model.urdf.link_transforms(joints)
        z_values.append(float(fk[tcp_link][2, 3]))
    z = np.asarray(z_values, dtype=np.float64)
    return {
        "tcp_link": tcp_link,
        "z_min_m": float(np.min(z)),
        "z_max_m": float(np.max(z)),
        "z_range_m": float(np.ptp(z)),
        "z_start_m": float(z[0]),
        "z_goal_m": float(z[-1]),
    }


def rotation_angle_rad(rotation: np.ndarray) -> float:
    trace = float(np.trace(rotation))
    value = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(value))


def trajectory_preference_metrics(
    surface_model: RobotSurfaceModel,
    reference: NUBSTrajectory6D,
    candidate: NUBSTrajectory6D,
    tcp_link: str,
    samples: int = 241,
) -> dict[str, Any]:
    times = np.linspace(0.0, candidate.total_duration, max(2, int(samples)))
    ref_q = reference.sample(times, max_derivative=0).q
    cand_q = candidate.sample(times, max_derivative=0).q
    ref_xyz = []
    cand_xyz = []
    orientation_errors = []
    for qr, qc in zip(ref_q, cand_q):
        ref_fk = surface_model.urdf.link_transforms({name: float(qr[i]) for i, name in enumerate(surface_model.joint_names)})
        cand_fk = surface_model.urdf.link_transforms({name: float(qc[i]) for i, name in enumerate(surface_model.joint_names)})
        ref_xyz.append(np.asarray(ref_fk[tcp_link][:3, 3], dtype=np.float64))
        cand_xyz.append(np.asarray(cand_fk[tcp_link][:3, 3], dtype=np.float64))
        relative = ref_fk[tcp_link][:3, :3].T @ cand_fk[tcp_link][:3, :3]
        orientation_errors.append(rotation_angle_rad(relative))
    ref_xyz_arr = np.asarray(ref_xyz, dtype=np.float64)
    cand_xyz_arr = np.asarray(cand_xyz, dtype=np.float64)
    xyz_dev = np.linalg.norm(cand_xyz_arr - ref_xyz_arr, axis=1)
    xy_dev = np.linalg.norm(cand_xyz_arr[:, :2] - ref_xyz_arr[:, :2], axis=1)
    z_dev = np.abs(cand_xyz_arr[:, 2] - ref_xyz_arr[:, 2])
    orientation = np.asarray(orientation_errors, dtype=np.float64)
    joint_dev = np.linalg.norm(cand_q - ref_q, axis=1)
    joint_step = np.linalg.norm(np.diff(cand_q, axis=0), axis=1) if len(cand_q) > 1 else np.zeros(1)
    tcp_step = np.linalg.norm(np.diff(cand_xyz_arr, axis=0), axis=1) if len(cand_xyz_arr) > 1 else np.zeros(1)
    tcp_xy_step = np.linalg.norm(np.diff(cand_xyz_arr[:, :2], axis=0), axis=1) if len(cand_xyz_arr) > 1 else np.zeros(1)
    return {
        "tcp_link": tcp_link,
        "max_tcp_xyz_deviation_m": float(np.max(xyz_dev)),
        "mean_tcp_xyz_deviation_m": float(np.mean(xyz_dev)),
        "max_tcp_xy_deviation_m": float(np.max(xy_dev)),
        "mean_tcp_xy_deviation_m": float(np.mean(xy_dev)),
        "max_tcp_z_deviation_m": float(np.max(z_dev)),
        "mean_tcp_z_deviation_m": float(np.mean(z_dev)),
        "max_tcp_orientation_deviation_rad": float(np.max(orientation)),
        "mean_tcp_orientation_deviation_rad": float(np.mean(orientation)),
        "max_tcp_orientation_deviation_deg": float(np.rad2deg(np.max(orientation))),
        "mean_tcp_orientation_deviation_deg": float(np.rad2deg(np.mean(orientation))),
        "max_joint_deviation_rad": float(np.max(joint_dev)),
        "mean_joint_deviation_rad": float(np.mean(joint_dev)),
        "joint_path_length_rad": float(np.sum(joint_step)),
        "tcp_path_length_m": float(np.sum(tcp_step)),
        "tcp_xy_path_length_m": float(np.sum(tcp_xy_step)),
    }


def task_constraint_report(
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    z_tol = float(args.tcp_z_hard_tolerance_m)
    orient_tol = float(args.tcp_orientation_hard_deg)
    checks = {
        "tcp_z_corridor_ok": bool(z_tol <= 0.0 or metrics["max_tcp_z_deviation_m"] <= z_tol),
        "tcp_orientation_ok": bool(
            orient_tol <= 0.0 or metrics["max_tcp_orientation_deviation_deg"] <= orient_tol
        ),
        "table_clearance_ok": True,
        "base_self_clearance_ok": True,
    }
    reasons = [name for name, ok in checks.items() if not ok]
    enabled = bool(z_tol > 0.0 or orient_tol > 0.0)
    return {
        "accepted": bool(all(checks.values())),
        "checks": checks,
        "reasons": reasons,
        "enabled": enabled,
        "tcp_z_hard_tolerance_m": z_tol,
        "tcp_orientation_hard_deg": orient_tol,
        "table_clearance_checked": False,
        "base_self_clearance_checked": False,
        "note": (
            "TCP height and orientation are enforced here. Table/base/self checks "
            "are reported as not independently implemented in this script and must "
            "remain covered by visual/dense safety review before hardware execution."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    trial_dir = args.trial_dir.resolve()
    output_dir = (args.output or (trial_dir / "ccro_nubs_jointspace_plan")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    config = _load(args.config)
    if args.max_iterations_override is not None:
        config["optimizer"]["max_iterations"] = int(args.max_iterations_override)
    q_start = parse_vector(args.q_start_rad, name="--q-start-rad") if args.q_start_rad else load_trial_q_start(trial_dir)
    q_goal = parse_vector(args.q_goal_rad, name="--q-goal-rad")
    durations = np.asarray([float(v) for v in args.segment_durations.split(",") if v.strip()], dtype=np.float64)
    if len(durations) < 1 or np.any(durations <= 0):
        raise ValueError("--segment-durations must contain positive durations")

    head = NUBSTrajectory6D.make_boundary_state(q_start)
    tail = NUBSTrajectory6D.make_boundary_state(q_goal)
    obstacle_points = np.asarray(np.load(trial_dir / "obstacle_points.npz")["points"], dtype=np.float64)
    if len(obstacle_points) > args.max_obstacle_points:
        rng = np.random.default_rng(args.seed)
        obstacle_points = obstacle_points[rng.choice(len(obstacle_points), args.max_obstacle_points, replace=False)]
    obstacle = StaticObstacleField.from_points(obstacle_points)
    table_z = float(json.loads((trial_dir / "summary.json").read_text(encoding="utf-8")).get("table_z_m", np.percentile(obstacle_points[:, 2], 2)))
    args._obstacle_points_for_corridor = obstacle_points
    args._table_z_for_corridor = table_z

    surface_model = make_surface_model(config)
    evaluator, verifier, limits = make_evaluator_and_verifier(config, surface_model)

    baseline = _baseline(config, head, tail, durations)
    reference = baseline.trajectory
    legacy_lambda_tcp_xyz = float(args.lambda_tcp_xyz or 0.0)
    if legacy_lambda_tcp_xyz > 0.0 and args.lambda_tcp_xy <= 0.0:
        args.lambda_tcp_xy = legacy_lambda_tcp_xyz
    route_corridor_enabled = bool(args.route_family != "none" and args.lambda_route_corridor > 0.0)
    side_z_corridor_enabled = bool(
        args.route_family in {"base_side", "outer_side"} and args.lambda_side_z_corridor > 0.0
    )
    if (
        args.lambda_tcp_z > 0.0
        or args.lambda_tcp_xy > 0.0
        or args.lambda_joint_deviation > 0.0
        or route_corridor_enabled
        or side_z_corridor_enabled
    ):
        optimizer = make_tabletop_optimizer(
            config, head, tail, durations, limits, evaluator, obstacle, reference, surface_model, args
        )
        optimizer_type = "joint_space_CCRO_NUBS_with_task_space_route_family_preferences"
    else:
        optimizer = _risk_optimizer(config, head, tail, durations, limits, evaluator, obstacle, None)
        optimizer_type = "joint_space_CCRO_NUBS"
    initial_inner = load_initial_inner_points(args.initial_plan_dir, baseline.p_inner.shape)
    initial_source = "linear_reference_inner_points"
    if initial_inner is not None:
        initial_source = str(args.initial_plan_dir.resolve())
    result = optimizer.optimize(baseline.p_inner if initial_inner is None else initial_inner)
    candidate = result.trajectory

    validation = verifier.verify(
        candidate,
        obstacle,
        current_q=head[:, 0],
        current_qd=head[:, 1],
        current_qdd=head[:, 2],
        q_goal=tail[:, 0],
        solver_success=result.success,
    )
    reference_validation = verifier.verify(
        reference,
        obstacle,
        current_q=head[:, 0],
        current_qd=head[:, 1],
        current_qdd=head[:, 2],
        q_goal=tail[:, 0],
        solver_success=baseline.success,
    )

    reference_dist = sample_trajectory_distances(
        reference,
        evaluator,
        obstacle,
        dt=args.audit_dt,
        density=config["risk"]["validation_density"],
    )
    candidate_dist = sample_trajectory_distances(
        candidate,
        evaluator,
        obstacle,
        dt=args.audit_dt,
        density=config["risk"]["validation_density"],
    )

    fields = [
        "trajectory",
        "t_s",
        *[f"q{j+1}_rad" for j in range(6)],
        *[f"qd{j+1}_rad_s" for j in range(6)],
        *[f"qdd{j+1}_rad_s2" for j in range(6)],
    ]
    write_csv(output_dir / "reference_trajectory.csv", sample_rows("reference_nubs", reference, args.trajectory_dt), fields)
    write_csv(output_dir / "ccro_nubs_candidate_trajectory.csv", sample_rows("ccro_nubs_candidate", candidate, args.trajectory_dt), fields)
    np.savez_compressed(
        output_dir / "ccro_nubs_trajectories.npz",
        reference_inner=baseline.p_inner,
        candidate_inner=result.p_inner,
        durations=durations,
        q_start=q_start,
        q_goal=q_goal,
        obstacle_points=obstacle_points,
    )
    save_distance_curve(
        output_dir / "figures" / "distance_risk_curve.png",
        reference_dist,
        candidate_dist,
        config["validation"]["d_accept"],
    )
    save_joint_preview(output_dir / "figures" / "joint_trajectory_preview.png", reference, candidate)

    times = np.linspace(0.0, candidate.total_duration, max(2, int(np.ceil(candidate.total_duration / args.urdf_dt)) + 1))
    q_path = candidate.sample(times, max_derivative=0).q
    plot_urdf_pose_sequence(
        output_dir / "figures" / "ccro_nubs_urdf_pose_sequence.png",
        surface_model,
        q_path,
        obstacle_points,
        table_z,
        config["risk"]["validation_density"],
    )

    candidate_metrics = trajectory_preference_metrics(surface_model, reference, candidate, args.tcp_link)
    task_constraints = task_constraint_report(candidate_metrics, args)
    route_family_report = (
        optimizer.route_corridor_report(candidate)
        if isinstance(optimizer, TabletopPreferenceStaticRiskNUBSOptimizer)
        else {
            "route_family": args.route_family,
            "enabled": False,
            "accepted": True,
            "max_violation_m": 0.0,
        }
    )
    accepted = bool(
        validation.accepted
        and (not task_constraints["enabled"] or task_constraints["accepted"])
        and (not route_family_report["enabled"] or route_family_report["accepted"])
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "robot_commanded": False,
        "trajectory_type": "joint_space_NUBS_not_B_spline",
        "trial_dir": str(trial_dir),
        "output_dir": str(output_dir),
        "status": "PLAN_ACCEPTED" if accepted else "PLAN_REJECTED",
        "accepted_for_real_execution": accepted,
        "optimizer_type": optimizer_type,
        "q_start_rad": q_start.tolist(),
        "q_goal_rad": q_goal.tolist(),
        "q_goal_source_note": args.q_goal_note,
        "durations": durations.tolist(),
        "initial_inner_source": initial_source,
        "reference": {
            "solver_success": bool(baseline.success),
            "min_distance_m": reference_dist["min_distance_m"],
            "nearest_link": reference_dist["nearest_link"],
            "dense_verification": asdict(reference_validation),
        },
        "candidate": {
            "optimizer_success": bool(result.success),
            "optimizer_message": result.message,
            "elapsed_ms": result.elapsed_ms,
            "iterations": result.iterations,
            "initial_min_distance_m": result.initial_min_distance,
            "final_min_distance_m": result.final_min_distance,
            "sampled_min_distance_m": candidate_dist["min_distance_m"],
            "nearest_link": candidate_dist["nearest_link"],
            "dense_verification": asdict(validation),
            "tcp_z_stats": tcp_path_z_stats(surface_model, candidate, args.tcp_link),
            "minimal_change_metrics": candidate_metrics,
            "task_constraints": task_constraints,
            "route_family_constraints": route_family_report,
        },
        "reference_tcp_z_stats": tcp_path_z_stats(surface_model, reference, args.tcp_link),
        "reference_candidate_minimal_change_metrics": candidate_metrics,
        "tabletop_preference": {
            "enabled": bool(
                args.lambda_tcp_z > 0.0
                or args.lambda_tcp_xy > 0.0
                or args.lambda_joint_deviation > 0.0
                or route_corridor_enabled
                or side_z_corridor_enabled
            ),
            "lambda_tcp_z": args.lambda_tcp_z,
            "tcp_z_tolerance_m": args.tcp_z_tolerance_m,
            "lambda_tcp_xy": args.lambda_tcp_xy,
            "tcp_xy_tolerance_m": args.tcp_xy_tolerance_m,
            "legacy_lambda_tcp_xyz": legacy_lambda_tcp_xyz,
            "lambda_joint_deviation": args.lambda_joint_deviation,
            "joint_deviation_tolerance_rad": args.joint_deviation_tolerance_rad,
            "tcp_preference_samples": args.tcp_preference_samples,
            "tcp_link": args.tcp_link,
            "route_family": args.route_family,
            "lambda_route_corridor": args.lambda_route_corridor,
            "route_corridor_margin_m": args.route_corridor_margin_m,
            "route_corridor_influence_m": args.route_corridor_influence_m,
            "lambda_side_z_corridor": args.lambda_side_z_corridor,
            "side_z_tolerance_m": args.side_z_tolerance_m,
            "vertical_uncertainty_m": args.vertical_uncertainty_m,
            "clearance_m": args.clearance_m,
        },
        "route_family_constraints": route_family_report,
        "task_constraints": task_constraints,
        "files": [
            "reference_trajectory.csv",
            "ccro_nubs_candidate_trajectory.csv",
            "ccro_nubs_trajectories.npz",
            "figures/distance_risk_curve.png",
            "figures/joint_trajectory_preview.png",
            "figures/ccro_nubs_urdf_pose_sequence.png",
        ],
        "execution_note": (
            "No robot command was sent. Execute only after inspecting URDF pose sequence, "
            "dense verification, and confirming a bounded joint trajectory execution interface."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, default=DEFAULT_TRIAL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--q-start-rad", default="")
    parser.add_argument("--q-goal-rad", default=DEFAULT_Q_GOAL_RAD)
    parser.add_argument(
        "--q-goal-note",
        default="Estimated from previous Y=-0.4 low-speed smoke execution; capture a fresh goal state before formal hardware execution.",
    )
    parser.add_argument("--segment-durations", default="2.0,2.0,2.0,2.0")
    parser.add_argument("--trajectory-dt", type=float, default=0.04)
    parser.add_argument("--audit-dt", type=float, default=0.04)
    parser.add_argument("--urdf-dt", type=float, default=0.20)
    parser.add_argument("--max-obstacle-points", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260652)
    parser.add_argument("--tcp-link", default="gripper_base_link")
    parser.add_argument(
        "--lambda-tcp-z",
        type=float,
        default=0.0,
        help="softly penalize TCP height changes relative to the reference trajectory; 0 disables",
    )
    parser.add_argument("--tcp-z-tolerance-m", type=float, default=0.015)
    parser.add_argument(
        "--lambda-tcp-xy",
        type=float,
        default=0.0,
        help="softly penalize in-plane TCP path deviations relative to the reference trajectory",
    )
    parser.add_argument("--tcp-xy-tolerance-m", type=float, default=0.03)
    parser.add_argument(
        "--lambda-tcp-xyz",
        type=float,
        default=0.0,
        help="deprecated compatibility alias; if --lambda-tcp-xy is zero this value is used for XY deviation",
    )
    parser.add_argument(
        "--lambda-joint-deviation",
        type=float,
        default=0.0,
        help="softly penalize joint-space deviations relative to the reference trajectory",
    )
    parser.add_argument("--joint-deviation-tolerance-rad", type=float, default=0.10)
    parser.add_argument(
        "--tcp-z-hard-tolerance-m",
        type=float,
        default=0.0,
        help="0 disables TCP height as a hard gate for general static avoidance",
    )
    parser.add_argument(
        "--tcp-orientation-hard-deg",
        type=float,
        default=0.0,
        help="0 disables orientation as a hard task gate; keep it reported for audit",
    )
    parser.add_argument("--tcp-preference-samples", type=int, default=25)
    parser.add_argument("--route-family", choices=["none", "base_side", "outer_side", "overpass"], default="none")
    parser.add_argument(
        "--lambda-route-corridor",
        type=float,
        default=0.0,
        help="soft penalty weight used to preserve a geometric route family",
    )
    parser.add_argument("--route-corridor-margin-m", type=float, default=0.08)
    parser.add_argument("--route-corridor-influence-m", type=float, default=0.25)
    parser.add_argument(
        "--lambda-side-z-corridor",
        type=float,
        default=0.0,
        help="side-route-only penalty that keeps base/outer candidates near the reference TCP height",
    )
    parser.add_argument(
        "--side-z-tolerance-m",
        type=float,
        default=0.05,
        help="maximum obstacle-near TCP height deviation allowed for side route families",
    )
    parser.add_argument("--clearance-m", type=float, default=0.08)
    parser.add_argument("--vertical-uncertainty-m", type=float, default=0.02)
    parser.add_argument("--max-iterations-override", type=int, default=None)
    parser.add_argument(
        "--initial-plan-dir",
        type=Path,
        default=None,
        help="reuse candidate_inner from an existing plan as the optimizer initial condition",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
