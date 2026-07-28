"""Fast 1 s local repair experiment for Chapter 6.4.

This is the replacement online experiment for the old seconds-level dynamic
NUBS optimizer.  It keeps time allocation fixed, repairs only a few local
interpolation points, uses one high-risk gradient query per repair step, and
reports end-to-end wall-clock timing.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from planning.nubs_trajectory import NUBSTrajectory6D
from planning.verifier import DynamicTrajectoryVerifier
from .repair.active_distance import extract_dense_nearest_distances
from .repair.nubs_linearization import build_local_sensitivity
from .repair.repair_v3 import run_repair_v3

from . import config_64 as cfg
from .common_64 import (
    constant_forecast,
    git_commit_hash,
    load_stage4_config,
    load_surface_model,
    make_critical_risk_stack,
    make_reference,
    make_risk_stack,
    write_json,
)
from .scenarios_64 import _tangent_direction


FAST_METHOD_NAMES = {
    "critical_fast_repair": "Critical-fast-repair",
    "ccro_fast_repair": "CCRO-fast-repair",
    "critical_fast_v3": "Critical-fast-v3",
    "ccro_fast_v3": "CCRO-fast-v3",
    "critical_fast_v4": "Critical-fast-v4",
    "ccro_fast_v4": "CCRO-fast-v4",
}


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def _make_local_reference(reference, tau_start: float) -> tuple[NUBSTrajectory6D, np.ndarray, np.ndarray, np.ndarray]:
    horizon = cfg.FAST_LOCAL_HORIZON
    segments = cfg.FAST_LOCAL_SEGMENTS
    durations = np.full(segments, horizon / segments, dtype=np.float64)
    knot_times = tau_start + np.cumsum(durations)[:-1]
    p_inner = np.vstack([reference.evaluate(float(tau)) for tau in knot_times])
    head = NUBSTrajectory6D.make_boundary_state(
        reference.evaluate(tau_start),
        reference.evaluate(tau_start, derivative_order=1),
        reference.evaluate(tau_start, derivative_order=2),
    )
    tau_goal = tau_start + horizon
    tail = NUBSTrajectory6D.make_boundary_state(
        reference.evaluate(tau_goal),
        reference.evaluate(tau_goal, derivative_order=1),
        reference.evaluate(tau_goal, derivative_order=2),
    )
    return NUBSTrajectory6D().generate(p_inner, head, tail, durations), p_inner, head, tail


def _surface_outward(model, q: np.ndarray, links: tuple[str, ...], rng: np.random.Generator) -> tuple[str, np.ndarray, np.ndarray]:
    link = links[int(rng.integers(0, len(links)))]
    points = model.surface_by_link(q, cfg.SURFACE_DENSITY_LOOP, {link})[link]
    point = points[int(rng.integers(0, len(points)))]
    centroid = points.mean(axis=0)
    outward = point - centroid
    norm = float(np.linalg.norm(outward))
    if norm < 1.0e-9:
        outward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        outward = outward / norm
    return link, point, outward


def _has_controllable_reference_risk(
    evaluator,
    local_ref: NUBSTrajectory6D,
    p_inner: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    forecast,
) -> bool:
    sample_times = np.arange(0.0, local_ref.total_duration + 0.5 * cfg.FAST_SAMPLE_DT, cfg.FAST_SAMPLE_DT)
    active = extract_dense_nearest_distances(
        evaluator,
        local_ref,
        forecast,
        sample_times=sample_times,
        top_k=cfg.FAST_V3_ACTIVE_CONSTRAINTS,
    )
    risky = [item for item in active if item.distance < cfg.D_ONLINE_ACCEPT]
    if not risky:
        return False
    margin = float(cfg.FAST_G1_ENDPOINT_MARGIN)
    if min(item.tau for item in risky) < margin:
        return False
    if max(item.tau for item in risky) > local_ref.total_duration - margin:
        return False
    sensitivity = build_local_sensitivity(
        p_inner,
        head,
        tail,
        local_ref.durations,
        sample_times,
        epsilon=cfg.FAST_V3_SENSITIVITY_EPS,
    )
    for item in risky:
        time_index = int(np.argmin(np.abs(sensitivity.sample_times - item.tau)))
        a_row = np.einsum("j,jv->v", item.gradient_q, sensitivity.sq[time_index], optimize=True)
        if float(np.linalg.norm(a_row)) >= cfg.FAST_G1_MIN_ACTIVE_A_NORM:
            return True
    return False


def make_fast_instances(model, reference, dense_evaluator, *, scenario: str, smoke: bool, g1: bool, g1_near: bool) -> list[dict[str, Any]]:
    rng = np.random.default_rng(cfg.RANDOM_SEED + (8100 if scenario == "D1" else 9100) + int(g1) * 1000)
    links = cfg.BODY_LINKS if scenario == "D1" else cfg.EE_LINKS
    scenario_type = "body_crossing_fast" if scenario == "D1" else "ee_crossing_fast"
    if g1 or g1_near:
        speeds = (0.15,)
        conflict_groups = {"G1": cfg.FAST_G1_CONFLICT_TIME}
        repeats = 10
    else:
        speeds = cfg.SPEED_GROUPS
        conflict_groups = cfg.FAST_CONFLICT_TIME_GROUPS
        repeats = 1 if smoke else 5
    instances: list[dict[str, Any]] = []
    index = 0
    tau_low, tau_high = cfg.FAST_TAU_START_RANGE
    tau_high = min(tau_high, reference.total_duration - cfg.FAST_LOCAL_HORIZON - 0.05)
    desired_per_cell = repeats
    for speed in speeds:
        for lead_label, conflict_time in conflict_groups.items():
            repeat = 0
            attempts = 0
            while repeat < desired_per_cell:
                attempts += 1
                if attempts > 2400:
                    raise RuntimeError(f"failed to generate enough fast instances for {scenario} {speed} {lead_label}")
                tau_start = float(rng.uniform(tau_low, tau_high))
                local_ref, p_inner, head, tail = _make_local_reference(reference, tau_start)
                q_conflict = local_ref.evaluate(float(conflict_time))
                link, surface_point, outward = _surface_outward(model, q_conflict, links, rng)
                radius = float(rng.uniform(0.04, 0.055))
                if g1_near:
                    clearance_candidates = np.linspace(0.12, 0.34, 12, dtype=np.float64)
                    rng.shuffle(clearance_candidates)
                elif g1:
                    clearance_candidates = np.linspace(0.08, 0.28, 11, dtype=np.float64)
                    rng.shuffle(clearance_candidates)
                else:
                    clearance_candidates = np.asarray([float(rng.uniform(0.025, 0.065))], dtype=np.float64)
                direction = _tangent_direction(outward, rng)
                velocity = float(speed) * direction
                selected = None
                for clearance_value in clearance_candidates:
                    clearance = float(clearance_value)
                    crossing_center = surface_point + (radius + clearance) * outward
                    center_at_start = crossing_center - velocity * float(conflict_time)
                    forecast = constant_forecast(center_at_start, velocity, radius)
                    reference_dense_min = _trajectory_min(dense_evaluator, local_ref, forecast, density="dense")
                    if not (g1 or g1_near):
                        selected = (clearance, center_at_start, forecast, reference_dense_min)
                        break
                    low, high = cfg.FAST_G1_NEAR_DENSE_MIN_RANGE if g1_near else cfg.FAST_G1_REFERENCE_DENSE_MIN_RANGE
                    if not (low <= reference_dense_min < high):
                        continue
                    if not _has_controllable_reference_risk(dense_evaluator, local_ref, p_inner, head, tail, forecast):
                        continue
                    selected = (clearance, center_at_start, forecast, reference_dense_min)
                    break
                if selected is None:
                    continue
                clearance, center_at_start, forecast, reference_dense_min = selected
                instances.append(
                    {
                        "instance_id": f"{scenario}F_{index:03d}",
                        "scenario_type": scenario_type,
                        "scenario": scenario,
                        "speed_group": float(speed),
                        "lead_label": lead_label,
                        "conflict_time_after_start": float(conflict_time),
                        "tau_start": tau_start,
                        "tau_goal": tau_start + cfg.FAST_LOCAL_HORIZON,
                        "target_link": link,
                        "obstacle_center0": center_at_start.tolist(),
                        "obstacle_velocity": velocity.tolist(),
                        "obstacle_radius": radius,
                        "reference_clearance_at_conflict": clearance,
                        "reference_dense_min_distance": float(reference_dense_min),
                        "seed": int(cfg.RANDOM_SEED + index),
                        "repeat_index": int(repeat),
                    }
                )
                index += 1
                repeat += 1
    return instances


def _trajectory_min(evaluator, trajectory, forecast, *, density: str) -> float:
    times = np.arange(0.0, trajectory.total_duration + 0.5 * cfg.FAST_SAMPLE_DT, cfg.FAST_SAMPLE_DT)
    risk = evaluator.trajectory(trajectory, forecast, times, density=density, with_gradient=False)
    return float(risk.min_distance)


def _worst_risk(evaluator, trajectory, forecast):
    times = np.arange(0.0, trajectory.total_duration + 0.5 * cfg.FAST_SAMPLE_DT, cfg.FAST_SAMPLE_DT)
    worst = None
    for tau in times:
        q = trajectory.evaluate(float(tau))
        risk = evaluator.configuration(q, forecast, float(tau), density=cfg.SURFACE_DENSITY_LOOP, with_gradient=False)
        if worst is None or risk.min_distance < worst["min_distance"]:
            worst = {"tau": float(tau), "q": q, "min_distance": float(risk.min_distance), "nearest_link": risk.nearest_link}
    return worst


def _repair_points(points: np.ndarray, direction: np.ndarray, tau: float, durations: np.ndarray, step: float, limits) -> np.ndarray:
    if points.size == 0:
        return points.copy()
    knot_times = np.cumsum(durations)[:-1]
    sigma = max(0.18, 0.25 * cfg.FAST_LOCAL_HORIZON)
    weights = np.exp(-0.5 * ((knot_times - float(tau)) / sigma) ** 2)
    if float(np.max(weights)) > 0.0:
        weights = weights / float(np.max(weights))
    repaired = points + float(step) * weights[:, None] * direction[None, :]
    return np.minimum(np.maximum(repaired, limits.q_min[None, :]), limits.q_max[None, :])


def _motion_violations(trajectory: NUBSTrajectory6D, limits) -> dict[str, float]:
    times = np.arange(0.0, trajectory.total_duration + 0.5 * cfg.FAST_SAMPLE_DT, cfg.FAST_SAMPLE_DT)
    samples = trajectory.sample(times)
    q_low = np.maximum(limits.q_min[None, :] - samples.q, 0.0)
    q_high = np.maximum(samples.q - limits.q_max[None, :], 0.0)
    qd = np.maximum(np.abs(samples.qd) - limits.qd_max[None, :], 0.0)
    qdd = np.maximum(np.abs(samples.qdd) - limits.qdd_max[None, :], 0.0)
    return {
        "q": float(np.max(np.maximum(q_low, q_high), initial=0.0)),
        "qd": float(np.max(qd, initial=0.0)),
        "qdd": float(np.max(qdd, initial=0.0)),
    }


def _distance_gradient(evaluator, q: np.ndarray, forecast, tau: float) -> np.ndarray:
    eps = cfg.FAST_DISTANCE_GRAD_EPS
    gradient = np.zeros(6, dtype=np.float64)
    for joint in range(6):
        plus = np.asarray(q, dtype=np.float64).copy()
        minus = np.asarray(q, dtype=np.float64).copy()
        plus[joint] += eps
        minus[joint] -= eps
        d_plus = evaluator.configuration(
            plus,
            forecast,
            float(tau),
            density=cfg.SURFACE_DENSITY_LOOP,
            with_gradient=False,
        ).min_distance
        d_minus = evaluator.configuration(
            minus,
            forecast,
            float(tau),
            density=cfg.SURFACE_DENSITY_LOOP,
            with_gradient=False,
        ).min_distance
        gradient[joint] = (float(d_plus) - float(d_minus)) / (2.0 * eps)
    return gradient


def _git_dirty() -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=cfg.ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except Exception:
        return None


def fast_repair(
    reference,
    model,
    evaluator,
    online_verifier,
    dense_verifier,
    limits,
    instance: dict[str, Any],
) -> dict[str, Any]:
    local_ref, p_inner, head, tail = _make_local_reference(reference, float(instance["tau_start"]))
    durations = local_ref.durations
    forecast = constant_forecast(
        np.asarray(instance["obstacle_center0"], dtype=np.float64),
        np.asarray(instance["obstacle_velocity"], dtype=np.float64),
        float(instance["obstacle_radius"]),
    )
    started = time.perf_counter()
    risk_scan_ms = 0.0
    repair_ms = 0.0
    medium_gate_ms = 0.0
    points = p_inner.copy()
    trajectory = local_ref
    repair_steps = 0
    accepted_updates = 0
    rejected_updates = 0
    best_medium_min = _trajectory_min(evaluator, trajectory, forecast, density=cfg.SURFACE_DENSITY_VERIFY)
    for step_index in range(cfg.FAST_REPAIR_STEPS + 1):
        t0 = time.perf_counter()
        worst = _worst_risk(evaluator, trajectory, forecast)
        risk_scan_ms += (time.perf_counter() - t0) * 1000.0
        if worst is None or worst["min_distance"] >= cfg.D_ONLINE_ACCEPT or step_index == cfg.FAST_REPAIR_STEPS:
            break
        t1 = time.perf_counter()
        direction = _distance_gradient(evaluator, worst["q"], forecast, worst["tau"])
        norm = float(np.linalg.norm(direction))
        if norm < 1.0e-10 or not np.all(np.isfinite(direction)):
            break
        unit = direction / norm
        accepted = None
        for step_size in cfg.FAST_REPAIR_STEP_SIZES:
            trial_points = _repair_points(points, unit, worst["tau"], durations, float(step_size), limits)
            trial_trajectory = NUBSTrajectory6D().generate(trial_points, head, tail, durations)
            motion = _motion_violations(trial_trajectory, limits)
            if motion["q"] > 1.0e-8 or motion["qd"] > 1.0e-8 or motion["qdd"] > 1.0e-8:
                rejected_updates += 1
                continue
            trial_min = _trajectory_min(evaluator, trial_trajectory, forecast, density=cfg.SURFACE_DENSITY_VERIFY)
            if trial_min <= best_medium_min + 1.0e-5:
                rejected_updates += 1
                continue
            accepted = (trial_points, trial_trajectory, float(trial_min), float(step_size))
            break
        if accepted is None:
            repair_ms += (time.perf_counter() - t1) * 1000.0
            break
        points, trajectory, best_medium_min, _ = accepted
        repair_ms += (time.perf_counter() - t1) * 1000.0
        repair_steps += 1
        accepted_updates += 1
    t_online = time.perf_counter()
    online = online_verifier.verify(
        trajectory,
        forecast,
        current_q=head[:, 0],
        current_qd=head[:, 1],
        current_qdd=head[:, 2],
        q_goal=tail[:, 0],
        solver_success=True,
    )
    medium_gate_ms = (time.perf_counter() - t_online) * 1000.0
    online_ms = (time.perf_counter() - started) * 1000.0
    t_dense = time.perf_counter()
    dense = dense_verifier.verify(
        trajectory,
        forecast,
        current_q=head[:, 0],
        current_qd=head[:, 1],
        current_qdd=head[:, 2],
        q_goal=tail[:, 0],
        solver_success=True,
    )
    dense_recheck_ms = (time.perf_counter() - t_dense) * 1000.0
    ref_dense_min = _trajectory_min(dense_verifier.risk_evaluator, local_ref, forecast, density="dense")
    cand_dense_min = float(dense.min_distance)
    return {
        "reference_trajectory": local_ref,
        "candidate_trajectory": trajectory,
        "metrics": {
            "reference_dense_min_distance": ref_dense_min,
            "candidate_dense_min_distance": cand_dense_min,
            "delta_dense_min_distance": float(cand_dense_min - ref_dense_min),
            "online_feasible": bool(online.accepted),
            "dense_geometry_only_feasible": bool(dense.accepted),
            "usable_candidate": bool(online_ms <= cfg.FAST_REPAIR_ACCEPT_MS and online.accepted and dense.accepted),
            "hard_realtime_ok": bool(online_ms <= cfg.FAST_REPAIR_HARD_MAX_MS),
            "beneficial": bool(cand_dense_min >= ref_dense_min + cfg.SWITCH_IMPROVEMENT_MARGIN),
            "repair_steps": repair_steps,
            "accepted_updates": accepted_updates,
            "rejected_updates": rejected_updates,
            "risk_scan_ms": float(risk_scan_ms),
            "repair_ms": float(repair_ms),
            "medium_gate_ms": float(medium_gate_ms),
            "online_ms": float(online_ms),
            "dense_recheck_ms": float(dense_recheck_ms),
            "validation_ms": float(medium_gate_ms + dense_recheck_ms),
            "total_ms": float(online_ms + dense_recheck_ms),
            "online_reasons": online.reasons,
            "dense_reasons": dense.reasons,
        },
    }


def fast_repair_v3(
    reference,
    evaluator,
    online_verifier,
    dense_verifier,
    limits,
    instance: dict[str, Any],
    *,
    dense_active: bool = False,
    v4_mode: bool = False,
) -> dict[str, Any]:
    local_ref, p_inner, head, tail = _make_local_reference(reference, float(instance["tau_start"]))
    durations = local_ref.durations
    forecast = constant_forecast(
        np.asarray(instance["obstacle_center0"], dtype=np.float64),
        np.asarray(instance["obstacle_velocity"], dtype=np.float64),
        float(instance["obstacle_radius"]),
    )
    started = time.perf_counter()
    repaired = run_repair_v3(
        evaluator,
        forecast,
        limits,
        p_inner,
        head,
        tail,
        durations,
        dense_active=dense_active,
        v4_mode=v4_mode,
    )
    t_online = time.perf_counter()
    online = online_verifier.verify(
        repaired.trajectory,
        forecast,
        current_q=head[:, 0],
        current_qd=head[:, 1],
        current_qdd=head[:, 2],
        q_goal=tail[:, 0],
        solver_success=True,
    )
    medium_gate_ms = (time.perf_counter() - t_online) * 1000.0
    online_ms = (time.perf_counter() - started) * 1000.0
    t_dense = time.perf_counter()
    dense = dense_verifier.verify(
        repaired.trajectory,
        forecast,
        current_q=head[:, 0],
        current_qd=head[:, 1],
        current_qdd=head[:, 2],
        q_goal=tail[:, 0],
        solver_success=True,
    )
    dense_recheck_ms = (time.perf_counter() - t_dense) * 1000.0
    ref_online_min = _trajectory_min(online_verifier.risk_evaluator, local_ref, forecast, density=cfg.SURFACE_DENSITY_VERIFY)
    ref_dense_min = _trajectory_min(dense_verifier.risk_evaluator, local_ref, forecast, density="dense")
    cand_dense_min = float(dense.min_distance)
    cand_online_min = float(online.min_distance)
    time_pass = bool(online_ms <= cfg.FAST_REPAIR_ACCEPT_MS)
    hard_time_pass = bool(online_ms <= cfg.FAST_REPAIR_HARD_MAX_MS)
    online_distance_pass = bool("distance_ok" not in online.reasons)
    acceleration_pass = bool("acceleration_ok" not in online.reasons)
    dense_safe = bool(dense.accepted)
    return {
        "reference_trajectory": local_ref,
        "candidate_trajectory": repaired.trajectory,
        "metrics": {
            "reference_dense_min_distance": ref_dense_min,
            "reference_online_min_distance": ref_online_min,
            "candidate_dense_min_distance": cand_dense_min,
            "candidate_online_min_distance": cand_online_min,
            "delta_dense_min_distance": float(cand_dense_min - ref_dense_min),
            "delta_online_min_distance": float(cand_online_min - ref_online_min),
            "candidate_medium_dense_gap": float(cand_online_min - cand_dense_min),
            "online_threshold_margin": float(cand_online_min - cfg.D_ONLINE_ACCEPT),
            "dense_threshold_margin": float(cand_dense_min - cfg.D_STOP),
            "online_distance_pass": online_distance_pass,
            "acceleration_pass": acceleration_pass,
            "time_pass": time_pass,
            "online_feasible": bool(online.accepted),
            "dense_geometry_only_feasible": dense_safe,
            "usable_candidate": bool(time_pass and online.accepted and dense_safe),
            "hard_realtime_ok": hard_time_pass,
            "beneficial": bool(cand_dense_min >= ref_dense_min + cfg.SWITCH_IMPROVEMENT_MARGIN),
            "repair_steps": repaired.accepted_steps,
            "accepted_updates": repaired.accepted_steps,
            "rejected_updates": max(0, repaired.iterations - repaired.accepted_steps),
            "risk_scan_ms": repaired.risk_scan_ms,
            "repair_ms": float(repaired.linearization_ms + repaired.qp_ms),
            "linearization_ms": repaired.linearization_ms,
            "qp_ms": repaired.qp_ms,
            "medium_gate_ms": float(medium_gate_ms),
            "online_ms": float(online_ms),
            "dense_recheck_ms": float(dense_recheck_ms),
            "validation_ms": float(medium_gate_ms + dense_recheck_ms),
            "total_ms": float(online_ms + dense_recheck_ms),
            "active_constraints": repaired.active_constraints,
            "qp_successes": repaired.qp_successes,
            "repair_messages": repaired.messages,
            "online_reasons": online.reasons,
            "dense_reasons": dense.reasons,
        },
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["scenario_type"],
            f"{float(row['speed_group']):.2f}",
            row["lead_label"],
            row["method"],
        )
        groups.setdefault(key, []).append(row)
    summary = []
    for (scenario, speed, lead, method), items in sorted(groups.items()):
        online_accepted = [item for item in items if item["online_feasible"]]
        verified_safety = (
            float(np.mean([item["dense_geometry_only_feasible"] for item in online_accepted]))
            if online_accepted
            else None
        )
        summary.append(
            {
                "scenario_type": scenario,
                "speed_group": speed,
                "lead_label": lead,
                "method": method,
                "n": len(items),
                "qp_solved": float(np.mean([item["qp_successes"] > 0 for item in items])),
                "repair_success": float(np.mean([item["dense_geometry_only_feasible"] for item in items])),
                "online_acceptance": float(np.mean([item["online_feasible"] for item in items])),
                "verified_safety": verified_safety,
                "usable": float(np.mean([item["usable_candidate"] for item in items])),
                "dense_feasible": float(np.mean([item["dense_geometry_only_feasible"] for item in items])),
                "online_distance_pass": float(np.mean([item.get("online_distance_pass", False) for item in items])),
                "time_pass": float(np.mean([item.get("time_pass", False) for item in items])),
                "beneficial": float(np.mean([item["beneficial"] for item in items])),
                "acceleration_ok": float(np.mean([item.get("acceleration_pass", "acceleration_ok" not in item["online_reasons"]) for item in items])),
                "delta_dense": float(np.mean([item["delta_dense_min_distance"] for item in items])),
                "medium_dense_gap_mean": float(np.mean([item.get("candidate_medium_dense_gap", 0.0) for item in items])),
                "online_margin_min": float(np.min([item.get("online_threshold_margin", 0.0) for item in items])),
                "dense_margin_min": float(np.min([item.get("dense_threshold_margin", 0.0) for item in items])),
                "online_ms_mean": float(np.mean([item["online_ms"] for item in items])),
                "online_ms_p95": float(np.percentile([item["online_ms"] for item in items], 95)),
                "online_ms_max": float(np.max([item["online_ms"] for item in items])),
                "dense_recheck_ms_p95": float(np.percentile([item["dense_recheck_ms"] for item in items], 95)),
            }
        )
    return summary


def write_table(summary: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "| scenario | speed | conflict | method | n | usable | dense feasible | accel ok | beneficial | delta Dmin dense / m | online mean / ms | online p95 / ms | online max / ms | dense recheck p95 / ms |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['scenario_type']} | {row['speed_group']} | {row['lead_label']} | {FAST_METHOD_NAMES.get(row['method'], row['method'])} | "
            f"{row['n']} | {_fmt(row['usable'], 2)} | {_fmt(row['dense_feasible'], 2)} | "
            f"{_fmt(row['acceleration_ok'], 2)} | {_fmt(row['beneficial'], 2)} | "
            f"{_fmt(row['delta_dense'])} | {_fmt(row['online_ms_mean'], 1)} | {_fmt(row['online_ms_p95'], 1)} | "
            f"{_fmt(row['online_ms_max'], 1)} | {_fmt(row['dense_recheck_ms_p95'], 1)} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _trial_failure_reason(row: dict[str, Any]) -> str:
    if row.get("usable_candidate", False):
        return "usable"
    if row.get("qp_successes", 0) <= 0:
        return "qp_not_solved"
    if not row.get("dense_geometry_only_feasible", False):
        return "dense_not_safe"
    if not row.get("time_pass", False):
        return "online_timeout"
    if not row.get("acceleration_pass", True):
        return "motion_limit_rejected"
    if not row.get("online_distance_pass", False):
        return "dense_safe_but_online_margin_rejected"
    if not row.get("online_feasible", False):
        return "other_online_gate_rejected"
    return "not_beneficial_only"


def write_acceptance_audit(rows: list[dict[str, Any]], summary: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# 6.4 Fast Local Repair Acceptance Audit",
        "",
        "## Gate Funnel",
        "",
        "| scenario | speed | conflict | method | n | QP solved | repair success dense>0.08 | online acceptance | verified safety among accepted | time pass | online distance pass | accel pass | usable | mean medium-dense gap / m | min online margin / m | min dense margin / m |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['scenario_type']} | {row['speed_group']} | {row['lead_label']} | {FAST_METHOD_NAMES.get(row['method'], row['method'])} | "
            f"{row['n']} | {_fmt(row['qp_solved'], 2)} | {_fmt(row['repair_success'], 2)} | "
            f"{_fmt(row['online_acceptance'], 2)} | {_fmt(row['verified_safety'], 2)} | "
            f"{_fmt(row['time_pass'], 2)} | {_fmt(row['online_distance_pass'], 2)} | {_fmt(row['acceleration_ok'], 2)} | "
            f"{_fmt(row['usable'], 2)} | {_fmt(row['medium_dense_gap_mean'], 4)} | "
            f"{_fmt(row['online_margin_min'], 4)} | {_fmt(row['dense_margin_min'], 4)} |"
        )
    lines.extend(
        [
            "",
            "## Trial-Level Audit",
            "",
            "| trial | method | ref dense | cand dense | cand online | medium-dense gap | online margin | dense margin | QP | dense safe | online accepted | time pass | accel pass | failure reason |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['instance_id']} | {FAST_METHOD_NAMES.get(row['method'], row['method'])} | "
            f"{_fmt(row.get('reference_dense_min_distance'), 4)} | {_fmt(row.get('candidate_dense_min_distance'), 4)} | "
            f"{_fmt(row.get('candidate_online_min_distance'), 4)} | {_fmt(row.get('candidate_medium_dense_gap'), 4)} | "
            f"{_fmt(row.get('online_threshold_margin'), 4)} | {_fmt(row.get('dense_threshold_margin'), 4)} | "
            f"{int(row.get('qp_successes', 0) > 0)} | {int(bool(row.get('dense_geometry_only_feasible', False)))} | "
            f"{int(bool(row.get('online_feasible', False)))} | {int(bool(row.get('time_pass', False)))} | "
            f"{int(bool(row.get('acceleration_pass', False)))} | {_trial_failure_reason(row)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(Path("results/new/6_4_fast_local_repair")))
    parser.add_argument("--config", default=str(cfg.STAGE4_CONFIG))
    parser.add_argument("--scenario", choices=["D1", "D2M"], default="D1")
    parser.add_argument(
        "--method",
        choices=[
            "critical_fast_repair",
            "ccro_fast_repair",
            "critical_fast_v3",
            "ccro_fast_v3",
            "critical_fast_v4",
            "ccro_fast_v4",
        ],
        default=None,
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--g1", action="store_true")
    parser.add_argument("--g1-near", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    trials_dir = output / "trials"
    paper_dir = output / "paper"
    trials_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name("config_64.yaml"), output / "config_64.yaml")
    config = load_stage4_config(Path(args.config))
    model = load_surface_model(config)
    reference, _, _, _ = make_reference(config)
    ccro_evaluator, online_verifier, limits = make_risk_stack(config, model, None)
    critical_evaluator, _ = make_critical_risk_stack(config, model)
    dense_verifier = DynamicTrajectoryVerifier(
        ccro_evaluator,
        limits,
        d_stop=cfg.D_STOP,
        time_step=cfg.FAST_SAMPLE_DT,
        density="dense",
        epsilon_goal=1.0e-2,
        epsilon_continuity_q=5.0e-3,
        epsilon_continuity_qd=3.0e-3,
        epsilon_continuity_qdd=3.0e-3,
        limit_tolerance=1.0e-8,
    )
    instances = make_fast_instances(
        model,
        reference,
        ccro_evaluator,
        scenario=args.scenario,
        smoke=args.smoke,
        g1=args.g1,
        g1_near=args.g1_near,
    )
    methods = (args.method,) if args.method else (
        "critical_fast_repair",
        "ccro_fast_repair",
        "critical_fast_v3",
        "ccro_fast_v3",
        "critical_fast_v4",
        "ccro_fast_v4",
    )
    rows: list[dict[str, Any]] = []
    for instance in instances:
        for method in methods:
            evaluator = critical_evaluator if method.startswith("critical") else ccro_evaluator
            if method.endswith("_v4"):
                repaired = fast_repair_v3(
                    reference,
                    evaluator,
                    online_verifier,
                    dense_verifier,
                    limits,
                    instance,
                    dense_active=not method.startswith("critical"),
                    v4_mode=True,
                )
            elif method.endswith("_v3"):
                repaired = fast_repair_v3(reference, evaluator, online_verifier, dense_verifier, limits, instance)
            else:
                repaired = fast_repair(reference, model, evaluator, online_verifier, dense_verifier, limits, instance)
            row = {
                **{key: value for key, value in instance.items() if key not in {"obstacle_center0", "obstacle_velocity"}},
                "method": method,
                **repaired["metrics"],
            }
            rows.append(row)
            write_json(trials_dir / f"{instance['instance_id']}_{method}.json", row)
            print(
                f"[6.4 fast] {instance['instance_id']} {instance['lead_label']} {FAST_METHOD_NAMES[method]} "
                f"usable={row['usable_candidate']} dense={row['dense_geometry_only_feasible']} "
                f"dD={row['delta_dense_min_distance']:.3f} online={row['online_ms']:.1f} ms"
            )
    summary = _aggregate(rows)
    metrics = {
        "experiment": "6.4 Fast CCRO-NUBS local repair",
        "scope": "1 s local repair; no seconds-level trajectory optimization",
        "git_commit": git_commit_hash(),
        "git_dirty": _git_dirty(),
        "scenario": args.scenario,
        "mode": "g1_near" if args.g1_near else ("g1" if args.g1 else ("smoke" if args.smoke else "formal_stage_a_fast")),
        "timing_targets_ms": {
            "p95_online": cfg.FAST_REPAIR_ACCEPT_MS,
            "hard_max": cfg.FAST_REPAIR_HARD_MAX_MS,
        },
        "trial_count": len(rows),
        "summary": summary,
        "trials": rows,
    }
    write_json(output / "fast_local_repair_64.json", metrics)
    write_table(summary, paper_dir / "table_6_4_fast_local_repair.md")
    write_acceptance_audit(rows, summary, paper_dir / "table_6_4_fast_acceptance_audit.md")
    print(f"[6.4 fast] saved {output / 'fast_local_repair_64.json'}")
    print(f"[6.4 fast] saved {paper_dir / 'table_6_4_fast_local_repair.md'}")
    print(f"[6.4 fast] saved {paper_dir / 'table_6_4_fast_acceptance_audit.md'}")


if __name__ == "__main__":
    main()
