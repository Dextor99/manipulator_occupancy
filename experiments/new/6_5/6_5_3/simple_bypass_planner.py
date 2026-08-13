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


def risk_link_bypass_goal_candidates(
    model: Any,
    q_now: np.ndarray,
    *,
    tcp_position: np.ndarray,
    goal_position: np.ndarray,
    risk_link: str,
    risk_position: np.ndarray,
    predicted_obstacle_position: np.ndarray,
    risk_point_q: np.ndarray | None = None,
    forward_m: float = 0.05,
    side_lengths_m: tuple[float, ...] = (0.04, 0.06, 0.08),
    tcp_link: str = "gripper_base_link",
    damping: float = 1.0e-3,
    task_weight: float = 1.0,
    max_joint_delta_rad: float = 0.12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate three away-side goals driven by the limiting robot link.

    The three Cartesian rows of the limiting-link point Jacobian request the
    lateral clearance displacement.  One additional TCP row preserves forward
    task progress.  This avoids using the gripper Jacobian as a proxy when a
    body link such as ``left_link`` is the actual clearance limiter.
    """
    q_values = np.asarray(q_now, dtype=np.float64)
    task, side, direction_audit = task_and_side_directions(
        tcp_position, goal_position, risk_position, predicted_obstacle_position
    )
    risk_configuration = q_values if risk_point_q is None else np.asarray(risk_point_q, dtype=np.float64)
    transforms = model.urdf.link_transforms(
        {
            name: float(risk_configuration[index])
            for index, name in enumerate(model.joint_names)
        }
    )
    if risk_link not in transforms:
        raise KeyError(f"risk link {risk_link!r} is absent from the robot model")
    risk_transform = np.asarray(transforms[risk_link], dtype=np.float64)
    risk_local = risk_transform[:3, :3].T @ (
        np.asarray(risk_position, dtype=np.float64) - risk_transform[:3, 3]
    )
    risk_jacobian = np.asarray(
        model.point_jacobian(q_values, risk_link, risk_local), dtype=np.float64
    )
    tcp_jacobian = np.asarray(
        model.point_jacobian(q_values, tcp_link, np.zeros(3, dtype=np.float64)),
        dtype=np.float64,
    )
    task_row = task @ tcp_jacobian
    system = np.vstack([risk_jacobian, float(task_weight) * task_row[None, :]])

    rows: list[dict[str, Any]] = []
    for side_m in side_lengths_m:
        requested_risk = float(side_m) * side
        target = np.r_[requested_risk, float(task_weight) * float(forward_m)]
        delta = system.T @ np.linalg.solve(
            system @ system.T + float(damping) * np.eye(system.shape[0]), target
        )
        raw_peak = float(np.max(np.abs(delta)))
        scale = 1.0
        if raw_peak > max_joint_delta_rad:
            scale = float(max_joint_delta_rad / raw_peak)
            delta *= scale
        achieved_risk = risk_jacobian @ delta
        achieved_tcp = tcp_jacobian @ delta
        rows.append(
            {
                "side_sign": 1,
                "forward_m": float(forward_m),
                "side_m": float(side_m),
                "q_goal": q_values + delta,
                "mapping": {
                    "risk_link": risk_link,
                    "risk_point_base_m": np.asarray(risk_position).tolist(),
                    "risk_point_local_m": risk_local.tolist(),
                    "requested_risk_delta_m": requested_risk.tolist(),
                    "linearized_risk_delta_m": achieved_risk.tolist(),
                    "requested_task_progress_m": float(forward_m),
                    "linearized_tcp_delta_m": achieved_tcp.tolist(),
                    "linearized_task_progress_m": float(np.dot(task, achieved_tcp)),
                    "joint_delta_rad": delta.tolist(),
                    "raw_joint_delta_max_abs_rad": raw_peak,
                    "joint_delta_scale": scale,
                    "joint_delta_max_abs_rad": float(np.max(np.abs(delta))),
                },
            }
        )
    direction_audit.update(
        {
            "risk_link": risk_link,
            "risk_point_base_m": np.asarray(risk_position).tolist(),
            "risk_point_local_m": risk_local.tolist(),
            "risk_point_configuration_rad": risk_configuration.tolist(),
            "candidate_side_policy": "away_only",
        }
    )
    return rows, direction_audit


def goal_directed_side_continuation_candidates(
    model: Any,
    q_now: np.ndarray,
    *,
    tcp_position: np.ndarray,
    goal_position: np.ndarray,
    risk_link: str,
    risk_position: np.ndarray,
    risk_point_q: np.ndarray,
    established_side: np.ndarray,
    forward_m: float = 0.05,
    side_m: float = 0.04,
    side_weights: tuple[float, ...] = (1.0, 0.5, 0.0),
    tcp_link: str = "gripper_base_link",
    damping: float = 1.0e-3,
    task_weight: float = 1.0,
    max_joint_delta_rad: float = 0.12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Progress toward goal while preserving a previously validated bypass side.

    The side vector fixes only which side of the obstacle is retained.  Its
    magnitude is varied from strong to release; it is not repeatedly inferred
    from the newest obstacle point and is never interpreted as a direction that
    must grow without bound.
    """
    q_values = np.asarray(q_now, dtype=np.float64)
    task = normalized(np.asarray(goal_position) - np.asarray(tcp_position))
    side_raw = np.asarray(established_side, dtype=np.float64)
    side_lateral = side_raw - task * float(np.dot(task, side_raw))
    side = normalized(side_lateral, fallback=side_raw)
    risk_configuration = np.asarray(risk_point_q, dtype=np.float64)
    transforms = model.urdf.link_transforms(
        {name: float(risk_configuration[index]) for index, name in enumerate(model.joint_names)}
    )
    risk_transform = np.asarray(transforms[risk_link], dtype=np.float64)
    risk_local = risk_transform[:3, :3].T @ (
        np.asarray(risk_position, dtype=np.float64) - risk_transform[:3, 3]
    )
    risk_jacobian = np.asarray(model.point_jacobian(q_values, risk_link, risk_local), dtype=np.float64)
    tcp_jacobian = np.asarray(
        model.point_jacobian(q_values, tcp_link, np.zeros(3, dtype=np.float64)), dtype=np.float64
    )
    task_row = task @ tcp_jacobian
    system = np.vstack([risk_jacobian, float(task_weight) * task_row[None, :]])
    rows = []
    for index, weight in enumerate(side_weights):
        if weight < 0.0 or weight > 1.0:
            raise ValueError("side continuation weights must lie in [0, 1]")
        requested_risk = float(side_m) * float(weight) * side
        target = np.r_[requested_risk, float(task_weight) * float(forward_m)]
        delta = system.T @ np.linalg.solve(
            system @ system.T + float(damping) * np.eye(system.shape[0]), target
        )
        raw_peak = float(np.max(np.abs(delta)))
        scale = 1.0
        if raw_peak > max_joint_delta_rad:
            scale = float(max_joint_delta_rad / raw_peak)
            delta *= scale
        achieved_risk = risk_jacobian @ delta
        achieved_tcp = tcp_jacobian @ delta
        rows.append(
            {
                "candidate": index + 1,
                "phase": ("strong" if weight == 1.0 else "release" if weight == 0.0 else "weak"),
                "side_weight": float(weight),
                "side_m": float(side_m) * float(weight),
                "forward_m": float(forward_m),
                "q_goal": q_values + delta,
                "mapping": {
                    "risk_link": risk_link,
                    "risk_point_base_m": np.asarray(risk_position).tolist(),
                    "risk_point_local_m": risk_local.tolist(),
                    "requested_risk_delta_m": requested_risk.tolist(),
                    "linearized_risk_delta_m": achieved_risk.tolist(),
                    "requested_task_progress_m": float(forward_m),
                    "linearized_tcp_delta_m": achieved_tcp.tolist(),
                    "linearized_task_progress_m": float(np.dot(task, achieved_tcp)),
                    "joint_delta_rad": delta.tolist(),
                    "raw_joint_delta_max_abs_rad": raw_peak,
                    "joint_delta_scale": scale,
                    "joint_delta_max_abs_rad": float(np.max(np.abs(delta))),
                },
            }
        )
    return rows, {
        "task_direction": task.tolist(),
        "established_bypass_side": side.tolist(),
        "side_policy": "lock_side_not_constant_direction",
        "candidate_phases": [row["phase"] for row in rows],
        "risk_link": risk_link,
    }
