"""Full joint-space CCRO-NUBS planning for a confirmed stationary obstacle."""
from __future__ import annotations

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
    static_config = copy.deepcopy(config)
    static_config.setdefault("validation", {})["d_accept"] = float(min_clearance_m)
    evaluator, verifier, limits = static_runtime.make_evaluator_and_verifier(static_config, model)
    baseline = _baseline(config, head, tail, durations)
    seeds = [
        ("baseline", baseline.p_inner),
        ("outside_0p04m", _seed_with_side_offset(
            baseline.p_inner, q_start=q_start, model=model,
            obstacle_points=points, offset_m=0.04)),
        ("outside_0p08m", _seed_with_side_offset(
            baseline.p_inner, q_start=q_start, model=model,
            obstacle_points=points, offset_m=0.08)),
    ]
    candidates: list[dict[str, Any]] = []
    accepted: list[tuple[float, Any, dict[str, Any]]] = []
    for name, initial in seeds:
        # Side seeds can start farther from the local basin; give those
        # initialisations a bounded extra iteration budget without changing
        # the safety contract or the baseline planner configuration.
        optimizer_config = config
        if name != "baseline":
            optimizer_config = copy.deepcopy(config)
            optimizer_config.setdefault("optimizer", {})["max_iterations"] = max(
                int(optimizer_config.get("optimizer", {}).get("max_iterations", 0)),
                600,
            )
        optimizer = _risk_optimizer(
            optimizer_config, head, tail, durations, limits, evaluator, obstacle, None
        )
        result = optimizer.optimize(initial)
        trajectory = result.trajectory if result.success else None
        verification = None
        if trajectory is not None:
            verification = verifier.verify(
                trajectory, obstacle, current_q=q_start,
                current_qd=np.zeros(6), current_qdd=np.zeros(6),
                q_goal=q_goal, solver_success=result.success,
            )
        minimum = None if verification is None else float(verification.min_distance)
        item = {
            "seed": name,
            "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status),
            "optimizer_message": str(result.message),
            "min_distance_m": minimum,
            "checks": {} if verification is None else dict(verification.checks),
            "reasons": [] if verification is None else list(verification.reasons),
        }
        candidates.append(item)
        if (
            trajectory is not None and verification is not None
            and verification.accepted and minimum is not None
            and minimum >= float(min_clearance_m)
        ):
            accepted.append((minimum, trajectory, item))
    authorized = bool(accepted)
    best = max(accepted, key=lambda row: row[0]) if accepted else None
    trajectory = None if best is None else best[1]
    best_item = {} if best is None else best[2]
    payload = {
        "status": "STATIONARY_FULL_CCRO_AUTHORIZED" if authorized else "STATIONARY_FULL_CCRO_HOLD",
        "authorized": authorized,
        "optimizer_success": any(item["optimizer_success"] for item in candidates),
        "duration_s": float(np.sum(durations)),
        "min_distance_m": best_item.get("min_distance_m"),
        "checks": best_item.get("checks", {}),
        "reasons": best_item.get("reasons", []),
        "selected_seed": None if best is None else best_item.get("seed"),
        "candidates": candidates,
        "obstacle_point_count": int(len(points)),
        "q_start_rad": q_start.tolist(),
        "q_goal_rad": q_goal.tolist(),
    }
    return payload, trajectory if authorized else None
