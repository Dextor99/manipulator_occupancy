"""Shared pure policy and Static20 verification for goal-directed rolling Fast."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from planning.obstacle_forecast import CompositeForecast, ConstantVelocitySphereForecast


def make_static20_forecast(
    geometry: dict[str, Any],
    *,
    observation_inflation_m: float = 0.020,
    valid_horizon_s: float = 60.0,
) -> CompositeForecast:
    if observation_inflation_m < 0.0 or valid_horizon_s <= 0.0:
        raise ValueError("Static20 inflation must be non-negative and horizon positive")
    centers = np.asarray(geometry["component_centers"], dtype=np.float64)
    radii = np.asarray(geometry["component_base_radii"], dtype=np.float64)
    return CompositeForecast(
        [
            ConstantVelocitySphereForecast(
                center=center,
                velocity=np.zeros(3),
                radius=float(radius),
                valid_horizon=float(valid_horizon_s),
                object_id=index,
                margin=float(observation_inflation_m),
                uncertainty=0.0,
                uncertainty_growth=0.0,
                velocity_radius_scale=0.0,
                beyond_horizon="error",
            )
            for index, (center, radius) in enumerate(zip(centers, radii), 1)
        ]
    )


def transported_reference_goal(reference: Any, q_now: np.ndarray, anchor_s: float, horizon_s: float):
    q_anchor, _, _ = reference.state_at(anchor_s)
    goal_time = min(float(reference.times[-1]), anchor_s + horizon_s)
    q_ref_goal, qd_ref_goal, qdd_ref_goal = reference.state_at(goal_time)
    increment = q_ref_goal - q_anchor
    q_goal = np.asarray(q_now, dtype=np.float64) + increment
    return (q_goal, qd_ref_goal, qdd_ref_goal), {
        "reference_anchor_time_s": float(anchor_s),
        "reference_goal_time_s": goal_time,
        "reference_increment_rad": increment.tolist(),
        "transported_goal_rad": q_goal.tolist(),
        "offset_from_reference_at_anchor_rad": (np.asarray(q_now) - q_anchor).tolist(),
        "offset_from_reference_at_goal_rad": (q_goal - q_ref_goal).tolist(),
    }


def bounded_terminal_goal(q_now: np.ndarray, q_final: np.ndarray, *, max_step_rad: float):
    delta = np.asarray(q_final, dtype=np.float64) - np.asarray(q_now, dtype=np.float64)
    peak = float(np.max(np.abs(delta)))
    scale = 1.0 if peak <= max_step_rad else float(max_step_rad / peak)
    step = scale * delta
    target = np.asarray(q_now, dtype=np.float64) + step
    zeros = np.zeros(6, dtype=np.float64)
    return (target, zeros, zeros), {
        "terminal_goal_error_max_abs_rad": peak,
        "terminal_step_scale": scale,
        "terminal_step_rad": step.tolist(),
        "transported_goal_rad": target.tolist(),
    }


def tcp_position(model: Any, q: np.ndarray, link: str) -> np.ndarray:
    transforms = model.urdf.link_transforms(
        {name: float(q[index]) for index, name in enumerate(model.joint_names)}
    )
    if link not in transforms:
        raise KeyError(f"TCP link {link!r} is absent from the robot model")
    return np.asarray(transforms[link][:3, 3], dtype=np.float64)


def risk_guided_goal(evaluator: Any, forecast: Any, nominal_goal: Any, *, max_delta_rad: float):
    q_goal, qd_goal, qdd_goal = nominal_goal
    risk = evaluator.configuration(
        np.asarray(q_goal), forecast, 0.0, density="medium", with_gradient=True
    )
    gradient = np.asarray(risk.gradient_q, dtype=np.float64)
    peak = float(np.max(np.abs(gradient)))
    audit = {
        "nominal_goal_distance_m": float(risk.min_distance),
        "nearest_link": risk.nearest_link,
        "risk_cost": float(risk.cost),
        "gradient_q": gradient.tolist(),
        "gradient_inf_norm": peak,
        "max_delta_rad": float(max_delta_rad),
    }
    if peak <= 1.0e-12 or max_delta_rad <= 0.0:
        return None, {**audit, "available": False, "reason": "zero_risk_gradient_or_zero_bound"}
    delta = -gradient * (float(max_delta_rad) / peak)
    guided_q = np.asarray(q_goal) + delta
    guided = evaluator.configuration_clearance(guided_q, forecast, 0.0, density="medium")
    audit.update(
        {
            "available": guided.min_distance > risk.min_distance,
            "delta_q_risk_rad": delta.tolist(),
            "delta_q_risk_max_abs_rad": float(np.max(np.abs(delta))),
            "linear_probe_goal_distance_m": float(guided.min_distance),
            "linear_probe_improvement_m": float(guided.min_distance - risk.min_distance),
        }
    )
    if not audit["available"]:
        audit["reason"] = "bounded_gradient_step_did_not_improve_goal_clearance"
        return None, audit
    return (guided_q, np.asarray(qd_goal), np.asarray(qdd_goal)), audit


def complete_verifier_pass(result: dict[str, Any], *, online_accept_m: float, fast_budget_ms: float) -> bool:
    return bool(
        result["online_pipeline_elapsed_ms"] <= fast_budget_ms
        and result["candidate_online_min_distance_m"] >= online_accept_m
        and all(ok for name, ok in result["verification_checks"].items() if name != "solver_ok")
    )


def terminal_side_release_allowed(progress_phase: str, verifier_passed: bool) -> bool:
    return bool(progress_phase == "terminal_goal" and verifier_passed)


def verify_static20_virtual_candidate(
    trial: Any,
    runtime_args: Any,
    config: dict[str, Any],
    model: Any,
    trajectory: Any,
    fresh_geometry: dict[str, Any],
    *,
    observation_inflation_m: float = 0.020,
) -> tuple[dict[str, Any], Any]:
    """Fresh RGB-D verifier with no execution-authority semantics."""
    horizon = max(2.0, float(trajectory.total_duration) + 1.0)
    forecast = make_static20_forecast(
        fresh_geometry,
        observation_inflation_m=observation_inflation_m,
        valid_horizon_s=horizon,
    )
    _, verifier, _ = trial.make_risk_stack(config, model, forecast)
    verifier.d_stop = float(runtime_args.online_accept_m)
    samples = trajectory.sample(np.asarray([0.0, trajectory.total_duration]))
    result = verifier.verify(
        trajectory,
        forecast,
        current_q=samples.q[0],
        current_qd=samples.qd[0],
        current_qdd=samples.qdd[0],
        q_goal=samples.q[-1],
        solver_success=True,
    )
    payload = {
        "status": "STATIC20_VIRTUAL_CANDIDATE_PASS" if result.accepted else "STATIC20_VIRTUAL_CANDIDATE_FAIL",
        "virtual_authorization": bool(result.accepted),
        "execution_authorization": False,
        "robot_executed": False,
        "static_observation_inflation_m": float(observation_inflation_m),
        "verification": asdict(result),
    }
    return payload, forecast
