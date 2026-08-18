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


def plan_stationary_terminal_ccro(*, config: dict[str, Any], model: Any,
                                  q_start: np.ndarray, q_goal: np.ndarray,
                                  geometry: dict[str, Any], output_dir: Path,
                                  min_clearance_m: float = 0.09) -> tuple[dict[str, Any], Any | None]:
    """Optimize one full static CCRO-NUBS and verify it before execution."""
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
    optimizer = _risk_optimizer(config, head, tail, durations, limits, evaluator, obstacle, None)
    result = optimizer.optimize(baseline.p_inner)
    trajectory = result.trajectory if result.success else None
    verification = None
    if trajectory is not None:
        verification = verifier.verify(trajectory, obstacle, current_q=q_start,
                                        current_qd=np.zeros(6), current_qdd=np.zeros(6),
                                        q_goal=q_goal, solver_success=result.success)
    checks = {} if verification is None else dict(verification.checks)
    authorized = bool(trajectory is not None and verification is not None and verification.accepted
                      and float(verification.min_distance) >= float(min_clearance_m))
    payload = {
        "status": "STATIONARY_FULL_CCRO_AUTHORIZED" if authorized else "STATIONARY_FULL_CCRO_HOLD",
        "authorized": authorized,
        "optimizer_success": bool(result.success),
        "duration_s": float(np.sum(durations)),
        "min_distance_m": None if verification is None else float(verification.min_distance),
        "checks": checks,
        "reasons": [] if verification is None else list(verification.reasons),
        "obstacle_point_count": int(len(points)),
        "q_start_rad": q_start.tolist(),
        "q_goal_rad": q_goal.tolist(),
    }
    return payload, trajectory if authorized else None
