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
    trajectory_generation_ms: float
    motion_check_ms: float
    candidate_distance_check_ms: float
    scale_attempts: int
    messages: list[str]


def _motion_violations(trajectory: NUBSTrajectory6D, limits, sample_times: np.ndarray) -> dict[str, float]:
    samples = trajectory.sample(sample_times)
    q_low = np.maximum(limits.q_min[None, :] - samples.q, 0.0)
    q_high = np.maximum(samples.q - limits.q_max[None, :], 0.0)
    qd = np.maximum(np.abs(samples.qd) - limits.qd_max[None, :], 0.0)
    qdd = np.maximum(np.abs(samples.qdd) - limits.qdd_max[None, :], 0.0)
    return {
        "q": float(np.max(np.maximum(q_low, q_high), initial=0.0)),
        "qd": float(np.max(qd, initial=0.0)),
        "qdd": float(np.max(qdd, initial=0.0)),
    }


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
    v4_mode: bool = False,
) -> RepairV3Result:
    sample_times = np.arange(0.0, float(np.sum(durations)) + 0.5 * cfg.FAST_SAMPLE_DT, cfg.FAST_SAMPLE_DT)
    motion_times = np.arange(0.0, float(np.sum(durations)) + 0.5 * cfg.DT, cfg.DT)
    points = np.asarray(p_inner, dtype=np.float64).copy()
    t_generate = time.perf_counter()
    trajectory = NUBSTrajectory6D().generate(points, head, tail, durations)
    trajectory_generation_ms = (time.perf_counter() - t_generate) * 1000.0
    risk_scan_ms = 0.0
    linearization_ms = 0.0
    qp_ms = 0.0
    motion_check_ms = 0.0
    candidate_distance_check_ms = 0.0
    scale_attempts = 0
    accepted = 0
    qp_successes = 0
    active_count = 0
    messages: list[str] = []
    max_iterations = cfg.FAST_V4_MAX_ITERATIONS if v4_mode else cfg.FAST_V3_MAX_ITERATIONS
    for iteration in range(max_iterations):
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
            d_safe=cfg.FAST_V4_TARGET_CLEARANCE if v4_mode else cfg.D_ONLINE_ACCEPT,
            clearance_reward=cfg.FAST_V4_CLEARANCE_REWARD if v4_mode else 0.0,
        )
        qp_ms += (time.perf_counter() - t_qp) * 1000.0
        messages.append(qp.message)
        if not qp.success:
            break
        qp_successes += 1
        current_min = min(item.distance for item in active)
        accepted_candidate = None
        scales = cfg.FAST_V4_ACCEPTANCE_SCALES if v4_mode else (cfg.FAST_V3_RELAXATION,)
        for scale in scales:
            scale_attempts += 1
            candidate_points = points + float(scale) * qp.delta.reshape(points.shape)
            t_generate = time.perf_counter()
            candidate = NUBSTrajectory6D().generate(candidate_points, head, tail, durations)
            trajectory_generation_ms += (time.perf_counter() - t_generate) * 1000.0
            t_motion = time.perf_counter()
            motion = _motion_violations(candidate, limits, motion_times)
            motion_check_ms += (time.perf_counter() - t_motion) * 1000.0
            if motion["q"] > 1.0e-8 or motion["qd"] > 1.0e-8 or motion["qdd"] > 1.0e-8:
                messages.append(
                    f"scale {float(scale):.2f} rejected: motion q={motion['q']:.3e} "
                    f"qd={motion['qd']:.3e} qdd={motion['qdd']:.3e}"
                )
                continue
            if dense_active:
                t_distance = time.perf_counter()
                next_active = extract_dense_nearest_distances(
                    evaluator,
                    candidate,
                    forecast,
                    sample_times=sample_times,
                    top_k=1,
                )
            else:
                t_distance = time.perf_counter()
                next_active = extract_active_distances(
                    evaluator,
                    candidate,
                    forecast,
                    sample_times=sample_times,
                    top_k=1,
                    density=cfg.SURFACE_DENSITY_LOOP,
                )
            candidate_distance_check_ms += (time.perf_counter() - t_distance) * 1000.0
            next_min = cfg.D_ONLINE_ACCEPT if not next_active else min(item.distance for item in next_active)
            if next_min <= current_min + cfg.FAST_V3_MIN_IMPROVEMENT:
                messages.append(f"scale {float(scale):.2f} rejected: no monotonic distance improvement")
                continue
            accepted_candidate = (candidate_points, candidate)
            messages.append(f"accepted scale {float(scale):.2f}")
            break
        if accepted_candidate is None:
            messages.append("rejected: no feasible monotonic step")
            break
        points, trajectory = accepted_candidate
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
        trajectory_generation_ms=float(trajectory_generation_ms),
        motion_check_ms=float(motion_check_ms),
        candidate_distance_check_ms=float(candidate_distance_check_ms),
        scale_attempts=int(scale_attempts),
        messages=messages,
    )
