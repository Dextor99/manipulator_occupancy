"""Full joint-space CCRO-NUBS planning for a confirmed stationary obstacle."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import importlib
import copy

import numpy as np

from experiments.exp_ccro_stage2 import _baseline, _risk_optimizer
from planning.mesh_risk import StaticObstacleField
from planning.nubs_trajectory import NUBSTrajectory6D


def _sphere_points(geometry: dict[str, Any], *, samples: int = 96) -> np.ndarray:
    centers = np.asarray(geometry.get("component_centers", []), dtype=float)
    radii = np.asarray(geometry.get("component_base_radii", []), dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0:
        raise ValueError("stationary geometry has no component centers")
    if len(radii) != len(centers):
        radii = np.full(len(centers), float(geometry.get("radius_m", 0.055)))
    points = []
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    for center, radius in zip(centers, radii):
        for i in range(samples):
            z = 1.0 - 2.0 * (i + 0.5) / samples
            a = 2.0 * np.pi * i / phi
            r = max(0.0, 1.0 - z * z) ** 0.5
            points.append(center + float(radius) * np.array([r * np.cos(a), r * np.sin(a), z]))
    return np.asarray(points, dtype=float)


def _make_stationary_full_config(config: dict[str, Any], *, min_clearance_m: float) -> dict[str, Any]:
    """High-fidelity planning config; the dense 0.09 m verifier is unchanged."""
    cfg = copy.deepcopy(config)
    risk = cfg.setdefault("risk", {})
    optimizer = cfg.setdefault("optimizer", {})
    validation = cfg.setdefault("validation", {})
    risk["risk_samples_per_segment"] = 12
    risk["optimizer_density"] = "medium"
    risk["d_safe"] = max(float(risk.get("d_safe", 0.11)), 0.12)
    optimizer["max_iterations"] = max(int(optimizer.get("max_iterations", 60)), 400)
    validation["d_accept"] = float(min_clearance_m)
    return cfg


def _choose_tcp_link(model: Any) -> str:
    names = list(getattr(model, "link_names", []))
    for name in ("gripper_base_link", "wrist3_Link", "wrist3_link"):
        if name in names:
            return name
    if not names:
        raise RuntimeError("surface model has no links")
    return names[-1]


def _make_route_args(*, route_family: str, tcp_link: str,
                     obstacle_points: np.ndarray, min_clearance_m: float) -> argparse.Namespace:
    return argparse.Namespace(
        tcp_link=tcp_link, lambda_tcp_z=0.0, tcp_z_tolerance_m=0.015,
        lambda_tcp_xy=0.0, tcp_xy_tolerance_m=0.03,
        lambda_joint_deviation=0.0, joint_deviation_tolerance_rad=0.10,
        tcp_preference_samples=25, route_family=route_family,
        lambda_route_corridor=0.0 if route_family == "none" else 5000.0,
        route_corridor_margin_m=0.08, route_corridor_influence_m=0.25,
        lambda_side_z_corridor=8000.0 if route_family in {"base_side", "outer_side"} else 0.0,
        side_z_tolerance_m=0.05, clearance_m=float(min_clearance_m),
        vertical_uncertainty_m=0.02,
        _obstacle_points_for_corridor=obstacle_points,
        _table_z_for_corridor=float(np.min(obstacle_points[:, 2])),
    )


def _dense_candidate_audit(*, trajectory: Any, evaluator: Any, obstacle: Any,
                           static_runtime: Any) -> dict[str, Any]:
    profile = static_runtime.sample_trajectory_distances(
        trajectory, evaluator, obstacle, dt=0.04, density="dense"
    )
    distances = np.asarray(profile["sample_distances_m"], dtype=float)
    return {
        "dense_verifier_min_distance_m": float(profile["min_distance_m"]),
        "dense_min_time_s": float(profile["min_time_s"]),
        "dense_nearest_link": profile["nearest_link"],
        "dense_start_distance_m": float(distances[0]),
        "dense_goal_distance_m": float(distances[-1]),
        "dense_sample_count": int(len(distances)),
    }


def _seed_with_side_offset(
    baseline: np.ndarray,
    *,
    q_start: np.ndarray,
    model: Any,
    obstacle_points: np.ndarray,
    offset_m: float,
) -> np.ndarray:
    """Make a bounded joint-space seed toward the obstacle's outside.

    This is only an optimizer initialisation, not a Cartesian command.  The
    final full-body CCRO verifier remains authoritative.  A damped Jacobian
    inverse keeps the perturbation small and leaves the boundary states fixed.
    """
    seed = np.asarray(baseline, dtype=float).copy()
    if offset_m <= 0.0 or seed.size == 0:
        return seed
    obstacle_center = np.mean(np.asarray(obstacle_points, dtype=float), axis=0)
    link_names = list(getattr(model, "link_names", []))
    tcp_link = next((n for n in ("gripper_base_link", "wrist3_Link", "wrist3_link") if n in link_names), None)
    if tcp_link is None and link_names:
        tcp_link = link_names[-1]
    if tcp_link is None or not hasattr(model, "point_jacobian"):
        return seed

    q_ref = np.asarray(q_start, dtype=float).copy()
    # Use the current TCP-to-obstacle direction as the conservative outside
    # direction.  If it is degenerate, retain a deterministic +X direction.
    try:
        fk = model.urdf.link_transforms(model._joint_dict(q_ref))
        tcp = np.asarray(fk[tcp_link][:3, 3], dtype=float)
    except Exception:
        tcp = np.zeros(3, dtype=float)
    direction = tcp[:2] - obstacle_center[:2]
    norm = float(np.linalg.norm(direction))
    if norm < 1.0e-8:
        direction = np.array([1.0, 0.0], dtype=float)
        norm = 1.0
    direction = direction / norm
    desired = np.array([direction[0], direction[1], 0.0], dtype=float) * float(offset_m)

    for row in range(seed.shape[0]):
        # Keep the seed smooth and zero at the trajectory boundaries.
        alpha = float(row + 1) / float(seed.shape[0] + 1)
        q = np.asarray(seed[row], dtype=float)
        try:
            jac = np.asarray(model.point_jacobian(q, tcp_link, np.zeros(3)), dtype=float)
            delta = np.linalg.pinv(jac, rcond=1.0e-3) @ (desired * np.sin(np.pi * alpha))
            delta = np.clip(delta, -0.20, 0.20)
            seed[row] = q + delta
        except Exception:
            continue
    return seed


def plan_stationary_terminal_ccro(*, config: dict[str, Any], model: Any,
                                  q_start: np.ndarray, q_goal: np.ndarray,
                                  geometry: dict[str, Any], output_dir: Path,
                                  min_clearance_m: float = 0.09) -> tuple[dict[str, Any], Any | None]:
    """Optimize a small multi-start family of full static CCRO-NUBS paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    durations = np.asarray([2.0, 2.0, 2.0, 2.0], dtype=float)
    head = NUBSTrajectory6D.make_boundary_state(q_start)
    tail = NUBSTrajectory6D.make_boundary_state(q_goal)
    points = _sphere_points(geometry)
    obstacle = StaticObstacleField.from_points(points)
    static_runtime = importlib.import_module(
        "experiments.new.6_5.6_5_2.run_652_static_avoidance"
    )
    planner_config = _make_stationary_full_config(config, min_clearance_m=min_clearance_m)
    plan652 = importlib.import_module(
        "experiments.new.6_5.6_5_2.plan_652_static_ccro_nubs_from_trial"
    )
    evaluator, verifier, limits = static_runtime.make_evaluator_and_verifier(planner_config, model)
    baseline = _baseline(planner_config, head, tail, durations)
    if not baseline.success:
        return {
            "status": "STATIONARY_FULL_CCRO_HOLD", "authorized": False,
            "reason": "baseline_generation_failed", "optimizer_message": str(baseline.message),
        }, None
    reference = baseline.trajectory
    tcp_link = _choose_tcp_link(model)
    families = [("free", "none"), ("base_side", "base_side"), ("outer_side", "outer_side")]
    candidates: list[dict[str, Any]] = []
    accepted: list[tuple[float, Any, dict[str, Any]]] = []
    for name, route_family in families:
        if route_family == "none":
            optimizer = _risk_optimizer(planner_config, head, tail, durations, limits, evaluator, obstacle, None)
        else:
            route_args = _make_route_args(
                route_family=route_family, tcp_link=tcp_link,
                obstacle_points=points, min_clearance_m=min_clearance_m,
            )
            optimizer = plan652.make_tabletop_optimizer(
                planner_config, head, tail, durations, limits, evaluator,
                obstacle, reference, model, route_args,
            )
        initial = baseline.p_inner.copy()
        result = optimizer.optimize(initial)
        trajectory = result.trajectory if result.success else None
        verification = None
        dense_audit = None
        route_report = {"route_family": route_family, "enabled": False, "accepted": True, "max_violation_m": 0.0}
        if trajectory is not None:
            verification = verifier.verify(
                trajectory, obstacle, current_q=q_start,
                current_qd=np.zeros(6), current_qdd=np.zeros(6),
                q_goal=q_goal, solver_success=result.success,
            )
            dense_audit = _dense_candidate_audit(
                trajectory=trajectory, evaluator=evaluator, obstacle=obstacle,
                static_runtime=static_runtime,
            )
            if route_family != "none" and hasattr(optimizer, "route_corridor_report"):
                route_report = optimizer.route_corridor_report(trajectory)
        minimum = None if verification is None else float(verification.min_distance)
        item = {
            "family": name,
            "route_family": route_family,
            "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status),
            "optimizer_message": str(result.message),
            "optimizer_elapsed_ms": float(result.elapsed_ms),
            "optimizer_iterations": int(result.iterations),
            "optimizer_function_evaluations": int(result.function_evaluations),
            "optimizer_gradient_norm": float(result.gradient_norm),
            "optimizer_initial_risk": float(result.initial_risk),
            "optimizer_final_risk": float(result.final_risk),
            "optimizer_initial_min_distance_m": float(result.initial_min_distance),
            "optimizer_final_min_distance_m": float(result.final_min_distance),
            "dense_verifier_min_distance_m": minimum,
            "checks": {} if verification is None else dict(verification.checks),
            "reasons": [] if verification is None else list(verification.reasons),
            "route_family_report": route_report,
            "seed_to_optimized_l2": float(np.linalg.norm(result.p_inner - initial)),
        }
        if dense_audit is not None:
            item.update(dense_audit)
        route_ok = bool(route_report.get("accepted", True))
        verifier_ok = bool(verification is not None and verification.accepted)
        clearance_ok = bool(minimum is not None and minimum >= float(min_clearance_m))
        candidate_authorized = bool(trajectory is not None and result.success and verifier_ok and clearance_ok and route_ok)
        item["authorized"] = candidate_authorized
        candidates.append(item)
        if candidate_authorized:
            accepted.append((minimum, trajectory, item))
    authorized = bool(accepted)
    best = max(accepted, key=lambda row: row[0]) if accepted else None
    trajectory = None if best is None else best[1]
    best_item = {} if best is None else best[2]
    payload = {
        "status": "STATIONARY_FULL_CCRO_AUTHORIZED" if authorized else "STATIONARY_FULL_CCRO_HOLD",
        "authorized": authorized,
        "optimizer_success": any(item["optimizer_success"] for item in candidates),
        "planner_mode": "high_fidelity_route_family_full_ccro",
        "authorization_clearance_m": float(min_clearance_m),
        "planner_target_d_safe_m": float(planner_config["risk"]["d_safe"]),
        "optimizer_density": planner_config["risk"]["optimizer_density"],
        "risk_samples_per_segment": int(planner_config["risk"]["risk_samples_per_segment"]),
        "max_iterations": int(planner_config["optimizer"]["max_iterations"]),
        "duration_s": float(np.sum(durations)),
        "min_distance_m": best_item.get("min_distance_m"),
        "checks": best_item.get("checks", {}),
        "reasons": best_item.get("reasons", []),
        "selected_family": None if best is None else best_item.get("family"),
        "selected_route_family": None if best is None else best_item.get("route_family"),
        "candidates": candidates,
        "obstacle_point_count": int(len(points)),
        "q_start_rad": q_start.tolist(),
        "q_goal_rad": q_goal.tolist(),
    }
    return payload, trajectory if authorized else None
