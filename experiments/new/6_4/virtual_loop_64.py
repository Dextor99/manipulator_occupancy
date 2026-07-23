"""Discrete virtual closed loop for Chapter 6.4."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from planning.obstacle_forecast import ShiftedForecast

from . import config_64 as cfg
from .common_64 import constant_forecast, min_distance_to_sphere, optimize_candidate
from .scenarios_64 import obstacle_center_at, obstacle_velocity_at


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
        if float(np.linalg.norm(gt_velocity)) < 1.0e-9:
            measured_velocity = np.zeros(3, dtype=np.float64)
        else:
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


def future_min_distance_with_links(
    evaluator,
    trajectory,
    local_tau: float,
    forecast,
    *,
    horizon: float,
    links: set[str] | None,
) -> dict[str, Any]:
    remaining = max(0.0, trajectory.total_duration - float(local_tau))
    horizon = min(float(horizon), remaining, float(forecast.valid_horizon))
    if horizon <= 1.0e-9:
        return {"distance": math.inf, "time": None, "nearest_link": None}
    best = {"distance": math.inf, "time": None, "nearest_link": None}
    for delta in np.linspace(0.0, horizon, cfg.EVALUATE_STEPS):
        q = trajectory.evaluate(min(float(local_tau + delta), trajectory.total_duration))
        risk = evaluator.configuration(
            q,
            forecast,
            float(delta),
            links=links,
            density=cfg.SURFACE_DENSITY_LOOP,
            with_gradient=False,
        )
        if risk.min_distance < best["distance"]:
            best = {"distance": float(risk.min_distance), "time": float(delta), "nearest_link": risk.nearest_link}
    return best


def apf_velocity_correction(evaluator, q: np.ndarray, forecast) -> np.ndarray:
    risk = evaluator.configuration(
        q,
        forecast,
        0.0,
        density=cfg.SURFACE_DENSITY_LOOP,
        with_gradient=True,
    )
    if risk.gradient_q is None or risk.min_distance >= cfg.APF_ACTIVATE_DISTANCE:
        return np.zeros(6, dtype=np.float64)
    correction = -cfg.APF_GAIN * np.asarray(risk.gradient_q, dtype=np.float64)
    norm = float(np.linalg.norm(correction))
    if norm > cfg.APF_MAX_STEP:
        correction *= cfg.APF_MAX_STEP / max(norm, 1.0e-12)
    return correction


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
    pending_candidate: dict[str, Any] | None = None
    safety_hold = False
    first_safety_hold = None
    first_replan = None
    false_replan = False
    bridge_min_distances: list[float] = []
    timeline: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    planning_control_cycles = 0
    started = time.perf_counter()
    execution_scale = 1.0
    q_exec = reference.evaluate(0.0).copy()
    critical_links = (
        {instance["target_region"]}
        if instance.get("target_region") not in {None, "far"}
        else None
    )

    while timestamp <= cfg.MAX_TRIAL_TIME:
        q_ref_now = active.evaluate(min(tau, active.total_duration))
        q = q_exec.copy() if method == "ssm_apf" else q_ref_now
        qd_ref_now = active.evaluate(min(tau, active.total_duration), derivative_order=1)
        qdd_ref_now = active.evaluate(min(tau, active.total_duration), derivative_order=2)
        qd = execution_scale * qd_ref_now
        qdd = (execution_scale ** 2) * qdd_ref_now
        motion_start_time = float(instance.get("motion_start_time", 0.0))
        pre_motion_center = (
            None
            if instance.get("pre_motion_center") is None
            else np.asarray(instance["pre_motion_center"], dtype=np.float64)
        )
        gt_center = obstacle_center_at(
            gt_center0,
            gt_velocity,
            timestamp,
            motion_start_time,
            pre_motion_center,
        )
        gt_velocity_now = obstacle_velocity_at(gt_velocity, timestamp, motion_start_time)
        gt_distance, gt_link = min_distance_to_sphere(
            model, q, gt_center, gt_radius, cfg.SURFACE_DENSITY_TRUTH
        )
        obs.update(timestamp, gt_center, gt_velocity_now, gt_radius)
        forecast = obs.forecast()
        obs_distance, obs_link = min_distance_to_sphere(
            model, q, obs.center, obs.radius, cfg.SURFACE_DENSITY_LOOP
        )
        future = future_min_distance(evaluator, active, tau, forecast, horizon=cfg.EVALUATE_HORIZON)
        method_future = (
            future
            if method != "critical_point_nubs"
            else future_min_distance_with_links(
                evaluator,
                active,
                tau,
                forecast,
                horizon=cfg.EVALUATE_HORIZON,
                links=critical_links,
            )
        )
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
            elif method_future["distance"] < cfg.D_REPLAN_IN:
                risk_level = "medium"

        if pending_candidate is not None:
            if timestamp >= pending_candidate["planned_switch_timestamp"]:
                event = pending_candidate["event"]
                if timestamp + 1.0e-9 < pending_candidate["completed_timestamp"]:
                    event["outcome"] = "timeout_before_switch"
                    event["actual_switch_timestamp"] = timestamp
                    event["candidate_accepted"] = False
                    event["rejection_reasons"] = ["planning_budget"]
                    events.append(event)
                    pending_candidate = None
                else:
                    event = pending_candidate["event"]
                    switch_forecast = ShiftedForecast(forecast, 0.0, max(1.0, pending_candidate["trajectory"].total_duration))
                    switch_verification = verifier.verify(
                        pending_candidate["trajectory"],
                        switch_forecast,
                        current_q=q,
                        current_qd=qd,
                        current_qdd=qdd,
                        q_goal=q_goal,
                        solver_success=True,
                    )
                    switch_checks = dict(switch_verification.checks)
                    switch_checks["solver_ok"] = True
                    reference_at_switch = future_min_distance(
                        evaluator,
                        active,
                        tau,
                        forecast,
                        horizon=min(cfg.EVALUATE_HORIZON, pending_candidate["trajectory"].total_duration),
                    )
                    reference_gate_ok = (
                        switch_verification.min_distance
                        >= float(reference_at_switch["distance"]) + cfg.SWITCH_IMPROVEMENT_MARGIN
                    )
                    switch_checks["reference_gate_ok"] = reference_gate_ok
                    switch_accepted = bool(all(switch_checks.values()))
                    switch_reasons = [name for name, passed in switch_checks.items() if not passed]
                    event["actual_switch_timestamp"] = timestamp
                    event["reference_min_distance_at_switch"] = reference_at_switch["distance"]
                    event["switch_validation"] = {
                        **asdict(switch_verification),
                        "accepted_without_solver_flag": switch_accepted,
                    }
                    if switch_accepted:
                        event["outcome"] = "accepted"
                        event["candidate_accepted"] = True
                        event["candidate_min_distance"] = switch_verification.min_distance
                        event["rejection_reasons"] = []
                        events.append(event)
                        active = pending_candidate["trajectory"]
                        tau = 0.0
                        accepted_count += 1
                        alpha = 1.0
                        safety_hold = False
                        risk_level = "switched"
                    else:
                        event["outcome"] = "rejected_at_switch"
                        event["candidate_accepted"] = False
                        event["candidate_min_distance"] = switch_verification.min_distance
                        event["rejection_reasons"] = switch_reasons
                        events.append(event)
                    pending_candidate = None

        if (
            method in {"ccro_nubs", "critical_point_nubs"}
            and pending_candidate is None
            and not safety_hold
            and risk_level == "medium"
            and replan_attempts < cfg.MAX_REPLAN_ATTEMPTS
        ):
            if first_replan is None or timestamp - first_replan >= cfg.REPLAN_INTERVAL:
                first_replan = timestamp if first_replan is None else first_replan
                replan_attempts += 1
                expected_switch_delay = min(cfg.EXPECTED_SWITCH_DELAY, cfg.SWITCH_DELAY)
                deadline_timestamp = timestamp + cfg.SWITCH_DELAY
                planned_switch_timestamp = timestamp + expected_switch_delay
                planning_alpha = min(alpha, cfg.PENDING_SLOW_SCALE)
                predicted_tau = min(active.total_duration, tau + planning_alpha * expected_switch_delay)
                q_plan = active.evaluate(predicted_tau)
                qd_plan = planning_alpha * active.evaluate(predicted_tau, derivative_order=1)
                qdd_plan = (planning_alpha ** 2) * active.evaluate(predicted_tau, derivative_order=2)
                forecast_after_switch = max(1.0, float(forecast.valid_horizon) - expected_switch_delay)
                remaining_duration = min(
                    max(float(active.total_duration - predicted_tau), 1.2),
                    forecast_after_switch,
                )
                local_forecast = ShiftedForecast(
                    forecast,
                    expected_switch_delay,
                    remaining_duration,
                )
                candidate = optimize_candidate(
                    config,
                    evaluator,
                    limits,
                    local_forecast,
                    q_now=q_plan,
                    qd_now=qd_plan,
                    qdd_now=qdd_plan,
                    q_goal=q_goal,
                    remaining_duration=remaining_duration,
                    verifier=verifier,
                    risk_links=critical_links if method == "critical_point_nubs" else None,
                )
                elapsed_s = float(candidate["optimization"]["elapsed_ms"]) / 1000.0
                planning_control_cycles += int(math.ceil(elapsed_s / cfg.DT))
                event = {
                    "attempt": replan_attempts,
                    "submitted_timestamp": timestamp,
                    "completed_timestamp": timestamp + elapsed_s,
                    "planned_switch_timestamp": planned_switch_timestamp,
                    "deadline_timestamp": deadline_timestamp,
                    "outcome": "pending",
                    "elapsed_ms": candidate["optimization"]["elapsed_ms"],
                    "optimizer_converged": candidate["optimization"]["success"],
                    "solver_success": candidate["optimization"]["success"],
                    "optimizer_status": candidate["optimization"]["status"],
                    "optimizer_message": candidate["optimization"]["message"],
                    "optimizer_iterations": candidate["optimization"]["iterations"],
                    "optimizer_function_evaluations": candidate["optimization"]["function_evaluations"],
                    "optimizer_initial_cost": candidate["optimization"]["initial_cost"],
                    "optimizer_final_cost": candidate["optimization"]["final_cost"],
                    "candidate_accepted": False,
                    "candidate_min_distance": candidate["verification"]["min_distance"],
                    "submission_validation": candidate["verification"],
                    "rejection_reasons": [],
                    "future_min_distance_at_submit": method_future["distance"],
                    "predicted_tau_at_switch": predicted_tau,
                    "planning_alpha": planning_alpha,
                }
                bridge = future_min_distance(
                    evaluator,
                    active,
                    tau,
                    forecast,
                    horizon=expected_switch_delay,
                )
                event["bridge_min_distance"] = bridge["distance"]
                event["bridge_time_to_min"] = bridge["time"]
                bridge_min_distances.append(float(bridge["distance"]))
                if timestamp + elapsed_s > deadline_timestamp:
                    event["outcome"] = "timeout_before_switch"
                    event["rejection_reasons"] = ["planning_budget"]
                    events.append(event)
                elif not candidate["verification"]["accepted"]:
                    event["outcome"] = "rejected_before_switch"
                    event["rejection_reasons"] = candidate["verification"]["reasons"]
                    events.append(event)
                else:
                    pending_candidate = {
                        "trajectory": candidate["trajectory"],
                        "event": event,
                        "completed_timestamp": timestamp + elapsed_s,
                        "planned_switch_timestamp": planned_switch_timestamp,
                        "planning_alpha": planning_alpha,
                    }

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
                    "planner_pending": pending_candidate is not None,
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
                "planner_pending": pending_candidate is not None,
            }
        )
        if safety_hold:
            alpha = 0.0
        elif pending_candidate is not None:
            bridge_distance = min(
                float(pending_candidate["event"].get("bridge_min_distance", math.inf)),
                float(method_future["distance"]),
            )
            if bridge_distance < cfg.D_SLOW:
                alpha = min(alpha, pending_candidate["planning_alpha"])
        if method == "ssm_apf":
            correction = apf_velocity_correction(evaluator, q, forecast)
            q_exec = q + (alpha * qd_ref_now + correction) * cfg.DT
            q_exec = np.minimum(np.maximum(q_exec, limits.q_min), limits.q_max)
        tau = min(active.total_duration, tau + alpha * cfg.DT)
        execution_scale = alpha
        timestamp += cfg.DT
        q_finish_check = q_exec if method == "ssm_apf" else active.evaluate(active.total_duration)
        if np.linalg.norm(q_finish_check - q_goal) <= cfg.FINISH_TOLERANCE and tau >= active.total_duration - 1.0e-9:
            break

    q_final = q_exec if method == "ssm_apf" else active.evaluate(min(tau, active.total_duration))
    min_gt = min(row["D_gt"] for row in timeline) if timeline else math.inf
    violation_steps = sum(1 for row in timeline if row["D_gt"] < cfg.D_STOP)
    finished = bool(np.linalg.norm(q_final - q_goal) <= cfg.FINISH_TOLERANCE and tau >= active.total_duration - 1.0e-9)
    task_safe_success = bool(finished and violation_steps == 0)
    replan_success = bool(accepted_count >= 1)
    if instance["scenario_type"] == "far_safe" and replan_attempts:
        false_replan = True
    if method == "reference_only":
        success = task_safe_success
    elif method == "ssm":
        success = task_safe_success
    else:
        if instance["scenario_type"] == "far_safe":
            success = finished and not false_replan and violation_steps == 0
        elif instance["scenario_type"] == "initial_high_risk":
            success = first_safety_hold is not None and accepted_count == 0
        else:
            success = task_safe_success

    return {
        "trial_id": f"{instance['instance_id']}_{method}",
        "instance_id": instance["instance_id"],
        "scenario_type": instance["scenario_type"],
        "speed_group": instance.get("speed_group"),
        "reference_risk_time": instance.get("reference_risk_time"),
        "trigger_to_reference_risk": (
            None
            if first_replan is None or instance.get("reference_risk_time") is None
            else float(instance["reference_risk_time"]) - float(first_replan)
        ),
        "method": method,
        "success": bool(success),
        "task_safe_success": bool(task_safe_success),
        "replan_success": bool(replan_success),
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
        "bridge_min_distance": None if not bridge_min_distances else float(min(bridge_min_distances)),
        "events": events,
        "timeline": timeline,
        "wall_elapsed_ms": float((time.perf_counter() - started) * 1000.0),
    }
