"""Minimal geometric bypass-goal generation for one Fast CCRO-NUBS event."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalized(values: np.ndarray, *, fallback: np.ndarray | None = None) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm > 1.0e-9:
        return vector / norm
    if fallback is None:
        raise ValueError("cannot normalize a zero vector")
    fallback = np.asarray(fallback, dtype=np.float64)
    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm <= 1.0e-9:
        raise ValueError("fallback direction is also zero")
    return fallback / fallback_norm


def task_and_side_directions(
    tcp_position: np.ndarray,
    goal_position: np.ndarray,
    risk_position: np.ndarray,
    predicted_obstacle_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    task = normalized(np.asarray(goal_position) - np.asarray(tcp_position))
    away = np.asarray(risk_position) - np.asarray(predicted_obstacle_position)
    lateral = away - task * float(np.dot(task, away))
    fallback = np.cross(task, np.asarray([0.0, 0.0, 1.0]))
    if np.linalg.norm(fallback) <= 1.0e-9:
        fallback = np.cross(task, np.asarray([0.0, 1.0, 0.0]))
    side = normalized(lateral, fallback=fallback)
    return task, side, {
        "task_direction": task.tolist(),
        "away_vector": away.tolist(),
        "lateral_away_vector": lateral.tolist(),
        "side_direction": side.tolist(),
    }


def damped_cartesian_delta_to_joint(
    model: Any,
    q_now: np.ndarray,
    cartesian_delta: np.ndarray,
    *,
    tcp_link: str = "gripper_base_link",
    damping: float = 1.0e-3,
    max_joint_delta_rad: float = 0.12,
) -> tuple[np.ndarray, dict[str, Any]]:
    jacobian = model.point_jacobian(
        np.asarray(q_now, dtype=np.float64), tcp_link, np.zeros(3, dtype=np.float64)
    )
    desired = np.asarray(cartesian_delta, dtype=np.float64)
    delta = jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + float(damping) * np.eye(3), desired
    )
    raw_peak = float(np.max(np.abs(delta)))
    scale = 1.0
    if raw_peak > max_joint_delta_rad:
        scale = float(max_joint_delta_rad / raw_peak)
        delta *= scale
    achieved = jacobian @ delta
    return np.asarray(q_now, dtype=np.float64) + delta, {
        "requested_tcp_delta_m": desired.tolist(),
        "linearized_tcp_delta_m": achieved.tolist(),
        "joint_delta_rad": delta.tolist(),
        "raw_joint_delta_max_abs_rad": raw_peak,
        "joint_delta_scale": scale,
        "joint_delta_max_abs_rad": float(np.max(np.abs(delta))),
    }


def bypass_goal_candidates(
    model: Any,
    q_now: np.ndarray,
    *,
    tcp_position: np.ndarray,
    goal_position: np.ndarray,
    risk_position: np.ndarray,
    predicted_obstacle_position: np.ndarray,
    forward_m: float = 0.05,
    side_lengths_m: tuple[float, ...] = (0.04, 0.06, 0.08),
    tcp_link: str = "gripper_base_link",
    max_joint_delta_rad: float = 0.12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task, side, direction_audit = task_and_side_directions(
        tcp_position, goal_position, risk_position, predicted_obstacle_position
    )
    rows = []
    for sign in (1.0, -1.0):
        for side_m in side_lengths_m:
            desired = float(forward_m) * task + sign * float(side_m) * side
            q_goal, mapping = damped_cartesian_delta_to_joint(
                model,
                q_now,
                desired,
                tcp_link=tcp_link,
                max_joint_delta_rad=max_joint_delta_rad,
            )
            rows.append(
                {
                    "side_sign": int(sign),
                    "forward_m": float(forward_m),
                    "side_m": float(side_m),
                    "q_goal": q_goal,
                    "mapping": mapping,
                }
            )
    return rows, direction_audit
