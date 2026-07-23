"""Discrete virtual closed loop for Chapter 6.4."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from planning.obstacle_forecast import ShiftedForecast

from . import config_64 as cfg
from .common_64 import constant_forecast, min_distance_to_sphere, optimize_candidate


@dataclass
class ObservationFilter:
    center: np.ndarray
    velocity: np.ndarray
    radius: float
    next_update: float
    rng: np.random.Generator

    def update(self, timestamp: float, gt_center: np.ndarray, gt_velocity: np.ndarray, gt_radius: float) -> None:
        if timestamp + 1.0e-9 < self.next_update:
            return
        measured_center = gt_center + self.rng.normal(0.0, cfg.OBS_POS_SIGMA, size=3)
        measured_velocity = gt_velocity + self.rng.normal(0.0, cfg.OBS_VEL_SIGMA, size=3)
        self.center = measured_center
        self.velocity = (
            cfg.OBS_VEL_ALPHA * measured_velocity
            + (1.0 - cfg.OBS_VEL_ALPHA) * self.velocity
        )
        self.radius = max(0.025, float(gt_radius + self.rng.normal(0.0, cfg.OBS_RADIUS_SIGMA)))
        self.next_update += cfg.OBSERVATION_DT

    def forecast(self):
        return constant_forecast(self.center, self.velocity, self.radius)


def speed_scale(distance: float) -> float:
    if distance <= cfg.D_STOP:
        return 0.0
    if distance >= cfg.D_SLOW:
        return 1.0
    return float((distance - cfg.D_STOP) / max(cfg.D_SLOW - cfg.D_STOP, 1.0e-9))


def future_min_distance(evaluator, trajectory, local_tau: float, forecast, *, horizon: float) -> dict[str, Any]:
    remaining = max(0.0, trajectory.total_duration - float(local_tau))
    horizon = min(float(horizon), remaining, float(forecast.valid_horizon))
    if horizon <= 1.0e-9:
        return {"distance": math.inf, "time": None, "nearest_link": None}
    best = {"distance": math.inf, "time": None, "nearest_link": None}
    for delta in np.linspace(0.0, horizon, cfg.EVALUATE_STEPS):
        q = trajectory.evaluate(min(float(local_tau + delta), trajectory.total_duration))
        risk = evaluator.configuration(q, forecast, float(delta), density=cfg.SURFACE_DENSITY_LOOP, with_gradient=False)
        if risk.min_distance < best["distance"]:
            best = {"distance": float(risk.min_distance), "time": float(delta), "nearest_link": risk.nearest_link}
    return best


def run_trial(
    *,
    config: dict[str, Any],
    model,
    reference,
    tail: np.ndarray,
    durations: np.ndarray,
    evaluator,
    verifier,
    limits,
    instance: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    gt_center0 = np.asarray(instance["gt_center0"], dtype=np.float64)
    gt_velocity = np.asarray(instance["gt_velocity"], dtype=np.float64)
    gt_radius = float(instance["gt_radius"])
    obs = ObservationFilter(
        center=np.asarray(instance["observed_center0"], dtype=np.float64),
        velocity=np.asarray(instance["observed_velocity"], dtype=np.float64),
        radius=float(instance["observed_radius"]),
        next_update=0.0,
        rng=np.random.default_rng(int(instance["observation_seed"])),
    )
    q_goal = tail[:, 0]
    active = reference
    tau = 0.0
    timestamp = 0.0
    replan_attempts = 0
    accepted_count = 0
    safety_hold = False
    first_safety_hold = None
    first_replan = None
    false_replan = False
    timeline: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    planning_control_cycles = 0
    started = time.perf_counter()

    while timestamp <= cfg.MAX_TRIAL_TIME:
        q = active.evaluate(min(tau, active.total_duration))
        qd = active.evaluate(min(tau, active.total_duration), derivative_order=1)
        qdd = active.evaluate(min(tau, active.total_duration), derivative_order=2)
        gt_center = gt_center0 + gt_velocity * timestamp
        gt_distance, gt_link = min_distance_to_sphere(
            model, q, gt_center, gt_radius, cfg.SURFACE_DENSITY_TRUTH
        )
        obs.update(timestamp, gt_center, gt_velocity, gt_radius)
        forecast = obs.forecast()
        obs_distance, obs_link = min_distance_to_sphere(
            model, q, obs.center, obs.radius, cfg.SURFACE_DENSITY_LOOP
        )
        future = future_min_distance(evaluator, active, tau, forecast, horizon=cfg.EVALUATE_HORIZON)
        alpha = 1.0
        risk_level = "low"

        if method == "reference_only":
            if gt_distance <= cfg.D_STOP:
                safety_hold = False
                risk_level = "failed_gt_violation"
        else:
            alpha = speed_scale(obs_distance)
            if obs_distance <= cfg.D_STOP:
                risk_level = "high"
                safety_hold = True
                if first_safety_hold is None:
                    first_safety_hold = timestamp
            elif future["distance"] < cfg.D_REPLAN_IN:
                risk_level = "medium"

        if method == "ccro_nubs" and risk_level == "medium" and replan_attempts < cfg.MAX_REPLAN_ATTEMPTS:
            if first_replan is None or timestamp - first_replan >= cfg.REPLAN_INTERVAL:
                first_replan = timestamp if first_replan is None else first_replan
                replan_attempts += 1
                local_forecast = ShiftedForecast(forecast, 0.0, max(1.0, active.total_duration - tau))
                remaining_duration = max(float(active.total_duration - tau), 1.2)
                candidate = optimize_candidate(
                    config,
                    evaluator,
                    limits,
                    local_forecast,
                    q_now=q,
                    qd_now=qd,
                    qdd_now=qdd,
                    q_goal=q_goal,
                    remaining_duration=remaining_duration,
                    verifier=verifier,
                )
                elapsed_s = float(candidate["optimization"]["elapsed_ms"]) / 1000.0
                planning_control_cycles += int(math.ceil(elapsed_s / cfg.DT))
                accepted = bool(candidate["verification"]["accepted"])
                accepted_count += int(accepted)
                event = {
                    "attempt": replan_attempts,
                    "submitted_timestamp": timestamp,
                    "completed_timestamp": timestamp + elapsed_s,
                    "planned_switch_timestamp": timestamp + elapsed_s,
                    "outcome": "accepted" if accepted else "rejected",
                    "elapsed_ms": candidate["optimization"]["elapsed_ms"],
                    "solver_success": candidate["optimization"]["success"],
                    "candidate_accepted": accepted,
                    "candidate_min_distance": candidate["verification"]["min_distance"],
                    "rejection_reasons": candidate["verification"]["reasons"],
                    "future_min_distance_at_submit": future["distance"],
                }
                events.append(event)
                if accepted:
                    active = candidate["trajectory"]
                    tau = 0.0
                    alpha = 1.0
                    safety_hold = False
                    risk_level = "switched"

        if method == "reference_only" and gt_distance <= cfg.D_STOP:
            timeline.append(
                {
                    "time": timestamp,
                    "tau": tau,
                    "q": q.tolist(),
                    "D_gt": gt_distance,
                    "nearest_link_gt": gt_link,
                    "risk_level": risk_level,
                    "alpha": alpha,
                    "safety_hold": safety_hold,
                }
            )
            break

        timeline.append(
            {
                "time": timestamp,
                "tau": tau,
                "q": q.tolist(),
                "D_gt": gt_distance,
                "D_obs": obs_distance,
                "D_future_obs": future["distance"],
                "nearest_link_gt": gt_link,
                "nearest_link_obs": obs_link,
                "risk_level": risk_level,
                "alpha": alpha,
                "safety_hold": safety_hold,
            }
        )
        if safety_hold:
            alpha = 0.0
        tau = min(active.total_duration, tau + alpha * cfg.DT)
        timestamp += cfg.DT
        if np.linalg.norm(active.evaluate(active.total_duration) - q_goal) <= cfg.FINISH_TOLERANCE and tau >= active.total_duration - 1.0e-9:
            break

    q_final = active.evaluate(min(tau, active.total_duration))
    min_gt = min(row["D_gt"] for row in timeline) if timeline else math.inf
    violation_steps = sum(1 for row in timeline if row["D_gt"] < cfg.D_STOP)
    finished = bool(np.linalg.norm(q_final - q_goal) <= cfg.FINISH_TOLERANCE and tau >= active.total_duration - 1.0e-9)
    if instance["scenario_type"] == "far_safe" and replan_attempts:
        false_replan = True
    if method == "reference_only":
        success = finished and violation_steps == 0
    elif method == "ssm":
        success = finished and violation_steps == 0
    else:
        if instance["scenario_type"] == "far_safe":
            success = finished and not false_replan and violation_steps == 0
        elif instance["scenario_type"] == "initial_high_risk":
            success = first_safety_hold is not None and accepted_count == 0
        else:
            success = finished and violation_steps == 0 and accepted_count >= 1

    return {
        "trial_id": f"{instance['instance_id']}_{method}",
        "instance_id": instance["instance_id"],
        "scenario_type": instance["scenario_type"],
        "method": method,
        "success": bool(success),
        "finished": finished,
        "duration_s": float(timeline[-1]["time"] if timeline else 0.0),
        "min_distance_gt": float(min_gt),
        "safety_violation_time_s": float(violation_steps * cfg.DT),
        "goal_error": float(np.linalg.norm(q_final - q_goal)),
        "replan_count": int(replan_attempts),
        "accepted_count": int(accepted_count),
        "first_replan_time": first_replan,
        "first_safety_hold_time": first_safety_hold,
        "false_replan": false_replan,
        "planning_control_cycles": int(planning_control_cycles),
        "events": events,
        "timeline": timeline,
        "wall_elapsed_ms": float((time.perf_counter() - started) * 1000.0),
    }
