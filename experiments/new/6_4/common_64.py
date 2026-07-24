"""Shared helpers for the revised Chapter 6.4 experiment."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_ccro_stage4 import _baseline, _limits, _load, _states  # noqa: E402
from planning.dynamic_optimizer import DynamicRiskNUBSOptimizer  # noqa: E402
from planning.nubs_trajectory import NUBSTrajectory6D  # noqa: E402
from planning.obstacle_forecast import ConstantVelocitySphereForecast, ShiftedForecast  # noqa: E402
from planning.robot_surface_model import RobotSurfaceModel  # noqa: E402
from planning.spatiotemporal_risk import DynamicConfigurationRisk, SpatioTemporalRiskEvaluator  # noqa: E402
from planning.verifier import DynamicTrajectoryVerifier  # noqa: E402

from . import config_64 as cfg


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "value"):
        return value.value
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def load_stage4_config(path: Path = cfg.STAGE4_CONFIG) -> dict[str, Any]:
    return _load(path)


def load_surface_model(config: dict[str, Any]) -> RobotSurfaceModel:
    robot, surface = config["robot"], config["surface"]
    return RobotSurfaceModel(
        ROOT / robot["urdf_path"],
        robot["joint_names"],
        surface["density_totals"],
        seed=surface["random_seed"],
        min_points_per_link=surface["min_points_per_link"],
        cache_dir=surface["cache_dir"],
        geometry=surface["geometry"],
    )


def make_reference(config: dict[str, Any]) -> tuple[NUBSTrajectory6D, np.ndarray, np.ndarray, np.ndarray]:
    head, tail, durations = _states(config)
    limits = _limits(config)
    result = _baseline(config, head, tail, durations, limits)
    return result.trajectory, head, tail, durations


def make_risk_stack(config: dict[str, Any], model: RobotSurfaceModel, forecast):
    limits = _limits(config)
    risk_cfg = config["risk"]
    evaluator = SpatioTemporalRiskEvaluator(
        model,
        d_safe=max(float(risk_cfg["d_safe"]), cfg.D_ONLINE_ACCEPT),
        d_activate=max(float(risk_cfg["d_activate"]), cfg.D_REPLAN_OUT),
        fd_epsilon_q=float(risk_cfg["fd_epsilon_q"]),
        density=cfg.SURFACE_DENSITY_LOOP,
    )
    verifier = DynamicTrajectoryVerifier(
        evaluator,
        limits,
        d_stop=cfg.D_ONLINE_ACCEPT,
        time_step=cfg.DT,
        density=cfg.SURFACE_DENSITY_VERIFY,
        epsilon_goal=1.0e-2,
        epsilon_continuity_q=5.0e-3,
        epsilon_continuity_qd=3.0e-3,
        epsilon_continuity_qdd=3.0e-3,
        limit_tolerance=1.0e-8,
    )
    return evaluator, verifier, limits


class CriticalPointSpatioTemporalRiskEvaluator(SpatioTemporalRiskEvaluator):
    """Sparse critical-point dynamic risk with 6.2-style equivalent radii."""

    def __init__(self, surface_model: RobotSurfaceModel, **kwargs) -> None:
        super().__init__(surface_model, **kwargs)
        self._local_critical_points = self._build_local_critical_points()

    def _build_local_critical_points(self) -> dict[str, list[tuple[str, np.ndarray, float]]]:
        selected_by_link: dict[str, list[tuple[str, np.ndarray, float]]] = {}
        seen: set[tuple[str, int]] = set()
        for region, links in cfg.CRITICAL_POINT_LINKS.items():
            radius = float(cfg.CRITICAL_POINT_RADII[region])
            for link in links:
                if link not in self.surface_model.link_names:
                    continue
                points = self.surface_model.local_samples(link, density="coarse")
                if len(points) == 0:
                    continue
                centroid = points.mean(axis=0)
                distances = np.linalg.norm(points - centroid[None, :], axis=1)
                selected = [int(np.argmax(distances))]
                if cfg.CRITICAL_POINTS_PER_REGION > 1:
                    farthest = int(np.argmax(np.linalg.norm(points - points[selected[0]][None, :], axis=1)))
                    selected.append(farthest)
                selected_by_link.setdefault(link, [])
                for local_index, point_index in enumerate(selected[: cfg.CRITICAL_POINTS_PER_REGION]):
                    key = (link, int(point_index))
                    if key in seen:
                        continue
                    seen.add(key)
                    selected_by_link[link].append(
                        (
                            f"{region}_{link}_{local_index}",
                            points[point_index].copy(),
                            radius,
                        )
                    )
        return selected_by_link

    def critical_point_count(self) -> int:
        return int(sum(len(points) for points in self._local_critical_points.values()))

    def _evaluate_no_gradient(self, q, occupancy, links, density) -> DynamicConfigurationRisk:
        del density
        values = np.asarray(q, dtype=np.float64)
        critical_by_link = self._local_critical_points
        if links is not None:
            critical_by_link = {link: points for link, points in critical_by_link.items() if link in links}
        if not critical_by_link or not occupancy.spheres:
            return DynamicConfigurationRisk(
                0.0, math.inf, None, None, None, None, {}, occupancy.extrapolated
            )
        fk = self.surface_model.urdf.link_transforms(self.surface_model._joint_dict(values))
        per_link: dict[str, float] = {}
        total_cost = 0.0
        total_weight = 0.0
        min_distance = math.inf
        nearest_link = None
        nearest_object_id = None
        nearest_robot = None
        nearest_obstacle = None
        for link, critical_points in critical_by_link.items():
            transform = fk.get(link)
            if transform is None:
                continue
            local_points = np.asarray([item[1] for item in critical_points], dtype=np.float64)
            point_radii = np.asarray([item[2] for item in critical_points], dtype=np.float64)
            world_points = local_points @ transform[:3, :3].T + transform[:3, 3]
            distance_columns = [
                np.linalg.norm(world_points - sphere.center[None, :], axis=1) - point_radii - sphere.radius
                for sphere in occupancy.spheres
            ]
            distance_matrix = np.column_stack(distance_columns)
            sphere_indices = np.argmin(distance_matrix, axis=1)
            distances = distance_matrix[np.arange(len(world_points)), sphere_indices]
            local_index = int(np.argmin(distances))
            local_distance = float(distances[local_index])
            if local_distance < min_distance:
                sphere = occupancy.spheres[int(sphere_indices[local_index])]
                point = world_points[local_index]
                min_distance = local_distance
                nearest_link = link
                nearest_object_id = int(sphere.object_id)
                nearest_robot = point.copy()
                direction = point - sphere.center
                norm = float(np.linalg.norm(direction))
                direction = np.array([1.0, 0.0, 0.0]) if norm < 1.0e-12 else direction / norm
                nearest_obstacle = sphere.center + sphere.radius * direction
            hinge = np.maximum(self.d_safe - distances, 0.0)
            link_cost = float(np.mean(hinge * hinge))
            per_link[link] = link_cost
            weight = float(self.link_weights.get(link, 1.0))
            if weight < 0.0 or not np.isfinite(weight):
                raise ValueError(f"invalid link weight for {link}")
            total_cost += weight * link_cost
            total_weight += weight
        return DynamicConfigurationRisk(
            cost=0.0 if total_weight <= 0.0 else total_cost / total_weight,
            min_distance=min_distance,
            nearest_link=nearest_link,
            nearest_object_id=nearest_object_id,
            robot_point=nearest_robot,
            obstacle_point=nearest_obstacle,
            per_link_cost=per_link,
            extrapolated=occupancy.extrapolated,
        )


def make_critical_risk_stack(config: dict[str, Any], model: RobotSurfaceModel):
    limits = _limits(config)
    risk_cfg = config["risk"]
    evaluator = CriticalPointSpatioTemporalRiskEvaluator(
        model,
        d_safe=max(float(risk_cfg["d_safe"]), cfg.D_ONLINE_ACCEPT),
        d_activate=max(float(risk_cfg["d_activate"]), cfg.D_REPLAN_OUT),
        fd_epsilon_q=float(risk_cfg["fd_epsilon_q"]),
        density=cfg.SURFACE_DENSITY_LOOP,
    )
    verifier = DynamicTrajectoryVerifier(
        evaluator,
        limits,
        d_stop=cfg.D_ONLINE_ACCEPT,
        time_step=cfg.DT,
        density=cfg.SURFACE_DENSITY_VERIFY,
        epsilon_goal=1.0e-2,
        epsilon_continuity_q=5.0e-3,
        epsilon_continuity_qd=3.0e-3,
        epsilon_continuity_qdd=3.0e-3,
        limit_tolerance=1.0e-8,
    )
    return evaluator, verifier


def constant_forecast(center: np.ndarray, velocity: np.ndarray, radius: float):
    return ConstantVelocitySphereForecast(
        np.asarray(center, dtype=np.float64),
        np.asarray(velocity, dtype=np.float64),
        float(radius),
        cfg.FORECAST_HORIZON,
        margin=0.035,
        uncertainty=0.015,
        uncertainty_growth=0.0030,
        velocity_radius_scale=0.080,
        beyond_horizon="hold_inflate",
    )


def _trajectory_min_distance(evaluator, trajectory: NUBSTrajectory6D, forecast, duration: float) -> float:
    sample_count = max(5, int(np.ceil(float(duration) / cfg.DT)) + 1)
    times = np.linspace(0.0, float(duration), sample_count)
    risk = evaluator.trajectory(trajectory, forecast, times, density=cfg.SURFACE_DENSITY_LOOP, with_gradient=False)
    return float(risk.min_distance)


def _clearance_guided_seed(
    evaluator,
    head: np.ndarray,
    tail: np.ndarray,
    durations: np.ndarray,
    base_seed: np.ndarray,
    limits,
    forecast,
) -> tuple[np.ndarray, dict[str, Any]]:
    base_trajectory = NUBSTrajectory6D().generate(base_seed, head, tail, durations)
    total_duration = float(np.sum(durations))
    sample_times = np.linspace(0.0, total_duration, max(5, int(np.ceil(total_duration / cfg.DT)) + 1))
    best_distance = math.inf
    best_q = None
    best_time = None
    best_gradient = None
    for sample_time in sample_times:
        q = base_trajectory.evaluate(float(sample_time))
        risk = evaluator.configuration(
            q,
            forecast,
            float(sample_time),
            density=cfg.SURFACE_DENSITY_LOOP,
            with_gradient=True,
        )
        if risk.min_distance < best_distance:
            best_distance = float(risk.min_distance)
            best_q = q
            best_time = float(sample_time)
            best_gradient = None if risk.gradient_q is None else np.asarray(risk.gradient_q, dtype=np.float64)
    if best_q is None or best_gradient is None or not np.any(np.isfinite(best_gradient)):
        return base_seed, {"seed_strategy": "reference", "detour_reason": "no_gradient", "reference_seed_min_distance": best_distance}
    direction = -best_gradient
    norm = float(np.linalg.norm(direction))
    if norm < 1.0e-12:
        return base_seed, {"seed_strategy": "reference", "detour_reason": "zero_gradient", "reference_seed_min_distance": best_distance}
    direction = direction / norm
    local_times = np.cumsum(durations)[:-1]
    if len(local_times) == 0:
        return base_seed, {"seed_strategy": "reference", "detour_reason": "no_inner_points", "reference_seed_min_distance": best_distance}
    weights = np.sin(np.pi * local_times / max(total_duration, 1.0e-9)) ** 2
    detour_seed = base_seed + weights[:, None] * cfg.CLEARANCE_DETOUR_STEP * direction[None, :]
    detour_seed = np.minimum(np.maximum(detour_seed, limits.q_min[None, :]), limits.q_max[None, :])
    detour_trajectory = NUBSTrajectory6D().generate(detour_seed, head, tail, durations)
    reference_distance = _trajectory_min_distance(evaluator, base_trajectory, forecast, total_duration)
    detour_distance = _trajectory_min_distance(evaluator, detour_trajectory, forecast, total_duration)
    if reference_distance < cfg.CLEARANCE_DETOUR_TRIGGER and detour_distance > reference_distance:
        return detour_seed, {
            "seed_strategy": "clearance_guided",
            "reference_seed_min_distance": reference_distance,
            "detour_seed_min_distance": detour_distance,
            "detour_time": best_time,
        }
    return base_seed, {
        "seed_strategy": "reference",
        "reference_seed_min_distance": reference_distance,
        "detour_seed_min_distance": detour_distance,
        "detour_time": best_time,
    }


def optimize_candidate(
    config: dict[str, Any],
    evaluator: SpatioTemporalRiskEvaluator,
    limits,
    forecast,
    *,
    q_now: np.ndarray,
    qd_now: np.ndarray,
    qdd_now: np.ndarray,
    q_goal: np.ndarray,
    remaining_duration: float,
    verifier: DynamicTrajectoryVerifier,
    qd_goal: np.ndarray | None = None,
    qdd_goal: np.ndarray | None = None,
    risk_links: set[str] | None = None,
    warm_start_trajectory: NUBSTrajectory6D | None = None,
    warm_start_tau: float | None = None,
    optimization_budget_s: float | None = None,
) -> dict[str, Any]:
    head = NUBSTrajectory6D.make_boundary_state(q_now, qd_now, qdd_now)
    tail = NUBSTrajectory6D.make_boundary_state(
        q_goal,
        np.zeros(6) if qd_goal is None else qd_goal,
        np.zeros(6) if qdd_goal is None else qdd_goal,
    )
    segment_count = max(3, int(config["trajectory"]["segment_count"]))
    durations = np.full(segment_count, max(float(remaining_duration), 1.2) / segment_count)
    optimizer = DynamicRiskNUBSOptimizer(
        head,
        tail,
        durations,
        limits,
        evaluator,
        forecast,
        lambda_risk=5.0 * float(config["optimizer"]["lambda_risk"]),
        risk_samples_per_segment=cfg.RISK_SAMPLES_PER_SEGMENT,
        risk_links=risk_links,
        lambda_smooth=float(config["optimizer"]["lambda_smooth"]),
        lambda_position=float(config["optimizer"]["lambda_position"]),
        lambda_velocity=float(config["optimizer"]["lambda_velocity"]),
        lambda_acceleration=float(config["optimizer"]["lambda_acceleration"]),
        samples_per_segment=cfg.OPTIMIZER_SAMPLES_PER_SEGMENT,
        finite_difference_epsilon=float(config["optimizer"]["finite_difference_epsilon"]),
        sensitivity_epsilon=float(config["optimizer"]["sensitivity_epsilon"]),
        max_iterations=cfg.OPTIMIZER_MAX_ITERATIONS,
        gradient_tolerance=float(config["optimizer"]["gradient_tolerance"]),
    )
    p_inner_initial = None
    if warm_start_trajectory is not None and warm_start_tau is not None and len(durations) > 1:
        local_times = np.cumsum(durations)[:-1]
        p_inner_initial = np.vstack(
            [
                warm_start_trajectory.evaluate(
                    min(float(warm_start_tau) + float(local_time), warm_start_trajectory.total_duration)
                )
                for local_time in local_times
            ]
        )
    seed_info: dict[str, Any] = {"seed_strategy": "linear", "warm_start_used": p_inner_initial is not None}
    if p_inner_initial is not None:
        p_inner_initial, seed_info = _clearance_guided_seed(
            evaluator,
            head,
            tail,
            durations,
            p_inner_initial,
            limits,
            forecast,
        )
        seed_info["warm_start_used"] = True
    result = optimizer.optimize(p_inner_initial=p_inner_initial, time_limit_s=optimization_budget_s)
    verification = verifier.verify(
        result.trajectory,
        forecast,
        current_q=q_now,
        current_qd=qd_now,
        current_qdd=qdd_now,
        q_goal=q_goal,
        solver_success=result.success,
    )
    checks_without_solver = dict(verification.checks)
    checks_without_solver["solver_ok"] = True
    accepted = bool(all(checks_without_solver.values()))
    reasons = [name for name, passed in checks_without_solver.items() if not passed]
    return {
        "trajectory": result.trajectory,
        "optimization": {
            "success": result.success,
            "status": result.status,
            "message": result.message,
            "elapsed_ms": result.elapsed_ms,
            "initial_cost": result.initial_cost,
            "final_cost": result.final_cost,
            "initial_min_distance": result.initial_min_distance,
            "final_min_distance": result.final_min_distance,
            "initial_risk": result.initial_risk,
            "final_risk": result.final_risk,
            "iterations": result.iterations,
            "function_evaluations": result.function_evaluations,
            "gradient_norm": result.gradient_norm,
            "warm_start_used": p_inner_initial is not None,
            **seed_info,
        },
        "verification": {
            "accepted": accepted,
            "strict_accepted": verification.accepted,
            "reasons": reasons,
            "strict_reasons": verification.reasons,
            "checks": verification.checks,
            "min_distance": verification.min_distance,
            "goal_error": verification.goal_error,
            "validation_ms": verification.validation_ms,
        },
    }


def min_distance_to_sphere(model: RobotSurfaceModel, q: np.ndarray, center: np.ndarray, radius: float, density: str) -> tuple[float, str | None]:
    best = math.inf
    best_link = None
    for link, points in model.surface_by_link(q, density).items():
        if len(points) == 0:
            continue
        distance = float(np.min(np.linalg.norm(points - center[None, :], axis=1) - float(radius)))
        if distance < best:
            best = distance
            best_link = link
    return best, best_link


def git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
