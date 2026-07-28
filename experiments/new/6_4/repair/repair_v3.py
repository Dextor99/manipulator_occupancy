"""Fast CCRO-NUBS v3 sequential convex local repair."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from planning.nubs_trajectory import NUBSTrajectory6D
from .. import config_64 as cfg
from .active_distance import extract_active_distances, extract_dense_nearest_distances
from .local_qp_solver import solve_local_qp
from .nubs_linearization import build_local_sensitivity


@dataclass(frozen=True)
class RepairV3Result:
    trajectory: NUBSTrajectory6D
    p_inner: np.ndarray
    iterations: int
    accepted_steps: int
    active_constraints: int
    qp_successes: int
    risk_scan_ms: float
    linearization_ms: float
    qp_ms: float
    messages: list[str]


def run_repair_v3(
    evaluator,
    forecast,
    limits,
    p_inner: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    durations: np.ndarray,
    *,
    dense_active: bool = False,
) -> RepairV3Result:
    sample_times = np.arange(0.0, float(np.sum(durations)) + 0.5 * cfg.FAST_SAMPLE_DT, cfg.FAST_SAMPLE_DT)
    points = np.asarray(p_inner, dtype=np.float64).copy()
    trajectory = NUBSTrajectory6D().generate(points, head, tail, durations)
    risk_scan_ms = 0.0
    linearization_ms = 0.0
    qp_ms = 0.0
    accepted = 0
    qp_successes = 0
    active_count = 0
    messages: list[str] = []
    for iteration in range(cfg.FAST_V3_MAX_ITERATIONS):
        t_scan = time.perf_counter()
        if dense_active:
            active = extract_dense_nearest_distances(
                evaluator,
                trajectory,
                forecast,
                sample_times=sample_times,
                top_k=cfg.FAST_V3_ACTIVE_CONSTRAINTS,
            )
        else:
            active = extract_active_distances(
                evaluator,
                trajectory,
                forecast,
                sample_times=sample_times,
                top_k=cfg.FAST_V3_ACTIVE_CONSTRAINTS,
                density=cfg.SURFACE_DENSITY_LOOP,
            )
        risk_scan_ms += (time.perf_counter() - t_scan) * 1000.0
        active_count += len(active)
        if not active:
            messages.append("no active constraints")
            break
        t_lin = time.perf_counter()
        sensitivity = build_local_sensitivity(
            points,
            head,
            tail,
            durations,
            sample_times,
            epsilon=cfg.FAST_V3_SENSITIVITY_EPS,
        )
        linearization_ms += (time.perf_counter() - t_lin) * 1000.0
        t_qp = time.perf_counter()
        qp = solve_local_qp(
            active,
            sensitivity,
            limits,
            trust_region=cfg.FAST_V3_TRUST_REGION,
            d_safe=cfg.D_ONLINE_ACCEPT,
        )
        qp_ms += (time.perf_counter() - t_qp) * 1000.0
        messages.append(qp.message)
        if not qp.success:
            break
        qp_successes += 1
        candidate_points = points + cfg.FAST_V3_RELAXATION * qp.delta.reshape(points.shape)
        candidate = NUBSTrajectory6D().generate(candidate_points, head, tail, durations)
        current_min = min(item.distance for item in active)
        if dense_active:
            next_active = extract_dense_nearest_distances(
                evaluator,
                candidate,
                forecast,
                sample_times=sample_times,
                top_k=1,
            )
        else:
            next_active = extract_active_distances(
                evaluator,
                candidate,
                forecast,
                sample_times=sample_times,
                top_k=1,
                density=cfg.SURFACE_DENSITY_LOOP,
            )
        next_min = cfg.D_ONLINE_ACCEPT if not next_active else min(item.distance for item in next_active)
        if next_min <= current_min + cfg.FAST_V3_MIN_IMPROVEMENT:
            messages.append("rejected: no monotonic distance improvement")
            break
        points = candidate_points
        trajectory = candidate
        accepted += 1
    return RepairV3Result(
        trajectory=trajectory,
        p_inner=points,
        iterations=iteration + 1 if "iteration" in locals() else 0,
        accepted_steps=accepted,
        active_constraints=active_count,
        qp_successes=qp_successes,
        risk_scan_ms=float(risk_scan_ms),
        linearization_ms=float(linearization_ms),
        qp_ms=float(qp_ms),
        messages=messages,
    )
