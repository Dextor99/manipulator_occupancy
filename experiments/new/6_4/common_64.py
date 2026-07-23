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
from planning.spatiotemporal_risk import SpatioTemporalRiskEvaluator  # noqa: E402
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
        d_safe=max(float(risk_cfg["d_safe"]), cfg.D_ACCEPT),
        d_activate=max(float(risk_cfg["d_activate"]), cfg.D_REPLAN_OUT),
        fd_epsilon_q=float(risk_cfg["fd_epsilon_q"]),
        density=cfg.SURFACE_DENSITY_LOOP,
    )
    verifier = DynamicTrajectoryVerifier(
        evaluator,
        limits,
        d_stop=cfg.D_ACCEPT,
        time_step=cfg.DT,
        density=cfg.SURFACE_DENSITY_VERIFY,
        epsilon_goal=1.0e-2,
        epsilon_continuity_q=5.0e-3,
        epsilon_continuity_qd=3.0e-3,
        epsilon_continuity_qdd=3.0e-3,
        limit_tolerance=1.0e-8,
    )
    return evaluator, verifier, limits


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
) -> dict[str, Any]:
    head = NUBSTrajectory6D.make_boundary_state(q_now, qd_now, qdd_now)
    tail = NUBSTrajectory6D.make_boundary_state(q_goal, np.zeros(6), np.zeros(6))
    segment_count = max(3, int(config["trajectory"]["segment_count"]))
    durations = np.full(segment_count, max(float(remaining_duration), 1.2) / segment_count)
    optimizer = DynamicRiskNUBSOptimizer(
        head,
        tail,
        durations,
        limits,
        evaluator,
        forecast,
        lambda_risk=2.0 * float(config["optimizer"]["lambda_risk"]),
        risk_samples_per_segment=cfg.RISK_SAMPLES_PER_SEGMENT,
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
    result = optimizer.optimize()
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
