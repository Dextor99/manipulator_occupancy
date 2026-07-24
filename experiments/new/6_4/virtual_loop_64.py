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


def executed_bridge_min_distance(
    evaluator,
    trajectory,
    local_tau: float,
    forecast,
    *,
    horizon: float,
    alpha: float,
) -> dict[str, Any]:
    horizon = min(float(horizon), float(forecast.valid_horizon))
    if horizon <= 1.0e-9:
        return {"distance": math.inf, "time": None, "tau": float(local_tau), "nearest_link": None}
    best = {"distance": math.inf, "time": None, "tau": float(local_tau), "nearest_link": None}
    tau_cursor = float(local_tau)
    for delta in np.arange(0.0, horizon + 0.5 * cfg.DT, cfg.DT):
        q = trajectory.evaluate(min(tau_cursor, trajectory.total_duration))
        risk = evaluator.configuration(q, forecast, float(delta), density=cfg.SURFACE_DENSITY_LOOP, with_gradient=False)
        if risk.min_distance < best["distance"]:
            best = {
                "distance": float(risk.min_distance),
                "time": float(delta),
                "tau": float(tau_cursor),
                "nearest_link": risk.nearest_link,
            }
        tau_cursor = min(trajectory.total_duration, tau_cursor + float(alpha) * cfg.DT)
    return best


def pending_speed_scale(bridge_distance: float, current_distance: float) -> float:
    if current_distance <= cfg.D_STOP:
        return 0.0
    if bridge_distance < cfg.BRIDGE_SLOW_IN:
        return max(cfg.PENDING_MIN_SLOW_SCALE, speed_scale(current_distance))
    if bridge_distance < cfg.BRIDGE_SLOW_OUT:
        return cfg.PENDING_LIGHT_SLOW_SCALE
    return 1.0


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
    critical_evaluator=None,
    critical_verifier=None,
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
    last_replan = None
    false_replan = False
    timeline: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    planning_control_cycles = 0
    started = time.perf_counter()
    execution_scale = 1.0
    q_exec = reference.evaluate(0.0).copy()
    planning_evaluator = critical_evaluator if method == "critical_point_nubs" and critical_evaluator is not None else evaluator
    local_resume: dict[str, Any] | None = None

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
        method_future = future_min_distance(planning_evaluator, active, tau, forecast, horizon=cfg.EVALUATE_HORIZON)
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
            event = pending_candidate["event"]
            event["bridge_min_distance_gt_executed"] = min(
                float(event.get("bridge_min_distance_gt_executed", math.inf)),
                float(gt_distance),
            )
            event["bridge_gt_sample_count"] = int(event.get("bridge_gt_sample_count", 0)) + 1
            if timestamp >= pending_candidate["planned_switch_timestamp"]:
                if timestamp + 1.0e-9 < pending_candidate["completed_timestamp"]:
                    event["outcome"] = "timeout_before_switch"
                    event["actual_switch_timestamp"] = timestamp
                    event["actual_tau_at_switch"] = tau
                    event["tau_prediction_error_at_switch"] = float(
                        abs(tau - float(event.get("predicted_tau_at_switch", tau)))
                    )
                    event["candidate_accepted"] = False
                    event["rejection_reasons"] = ["planning_budget"]
                    events.append(event)
                    pending_candidate = None
                elif not pending_candidate["submission_accepted"]:
                    event["outcome"] = "rejected_before_switch"
                    event["actual_switch_timestamp"] = timestamp
                    event["actual_tau_at_switch"] = tau
                    event["tau_prediction_error_at_switch"] = float(
                        abs(tau - float(event.get("predicted_tau_at_switch", tau)))
                    )
                    event["candidate_accepted"] = False
                    event["rejection_reasons"] = pending_candidate["submission_reasons"]
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
                        q_goal=pending_candidate["q_goal_candidate"],
                        solver_success=True,
                    )
                    switch_checks = dict(switch_verification.checks)
                    switch_checks["solver_ok"] = True
                    event["actual_tau_at_switch"] = tau
                    event["tau_prediction_error_at_switch"] = float(
                        abs(tau - float(event.get("predicted_tau_at_switch", tau)))
                    )
                    reference_at_switch = future_min_distance(
                        evaluator,
                        active,
                        tau,
                        forecast,
                        horizon=min(cfg.EVALUATE_HORIZON, pending_candidate["trajectory"].total_duration),
                    )
                    reference_safe = float(reference_at_switch["distance"]) >= cfg.D_ONLINE_ACCEPT
                    candidate_better = (
                        switch_verification.min_distance
                        >= float(reference_at_switch["distance"]) + cfg.SWITCH_IMPROVEMENT_MARGIN
                    )
                    remaining_ref_time = max(0.0, active.total_duration - tau)
                    candidate_time_ok = (
                        pending_candidate["trajectory"].total_duration
                        <= remaining_ref_time + cfg.SWITCH_TIME_EXTENSION_MARGIN
                    )
                    reference_gate_ok = (not reference_safe) or (candidate_better and candidate_time_ok)
                    switch_checks["reference_gate_ok"] = reference_gate_ok
                    switch_checks["candidate_time_ok"] = candidate_time_ok
                    switch_accepted = bool(all(switch_checks.values()))
                    switch_reasons = [name for name, passed in switch_checks.items() if not passed]
                    event["actual_switch_timestamp"] = timestamp
                    event["reference_min_distance_at_switch"] = reference_at_switch["distance"]
                    event["reference_safe_at_switch"] = reference_safe
                    event["candidate_time_ok"] = candidate_time_ok
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
                        if pending_candidate.get("resume_trajectory") is not None:
                            local_resume = {
                                "trajectory": active,
                                "tau": pending_candidate["resume_tau"],
                            }
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
            if last_replan is None or timestamp - last_replan >= cfg.REPLAN_INTERVAL:
                if first_replan is None:
                    first_replan = timestamp
                last_replan = timestamp
                replan_attempts += 1
                planned_switch_delay = cfg.PLANNED_SWITCH_DELAY
                deadline_timestamp = timestamp + planned_switch_delay
                planned_switch_timestamp = deadline_timestamp
                nominal_bridge = executed_bridge_min_distance(
                    evaluator,
                    active,
                    tau,
                    forecast,
                    horizon=planned_switch_delay,
                    alpha=1.0,
                )
                planning_bridge_distance = min(float(nominal_bridge["distance"]), float(method_future["distance"]))
                planning_alpha = min(alpha, pending_speed_scale(planning_bridge_distance, obs_distance))
                if risk_level == "medium":
                    planning_alpha = min(planning_alpha, cfg.PENDING_LIGHT_SLOW_SCALE)
                predicted_tau = min(active.total_duration, tau + planning_alpha * planned_switch_delay)
                q_plan = active.evaluate(predicted_tau)
                qd_plan = planning_alpha * active.evaluate(predicted_tau, derivative_order=1)
                qdd_plan = (planning_alpha ** 2) * active.evaluate(predicted_tau, derivative_order=2)
                forecast_after_switch = max(1.0, float(forecast.valid_horizon) - planned_switch_delay)
                full_remaining_duration = max(float(active.total_duration - predicted_tau), 1.2)
                if cfg.USE_LOCAL_CANDIDATE:
                    resume_tau = min(active.total_duration, predicted_tau + cfg.LOCAL_REPLAN_HORIZON)
                    q_candidate_goal = active.evaluate(resume_tau)
                    qd_candidate_goal = active.evaluate(resume_tau, derivative_order=1)
                    qdd_candidate_goal = active.evaluate(resume_tau, derivative_order=2)
                    remaining_duration = min(max(float(resume_tau - predicted_tau), 1.2), forecast_after_switch)
                else:
                    resume_tau = None
                    q_candidate_goal = q_goal
                    qd_candidate_goal = np.zeros(6, dtype=np.float64)
                    qdd_candidate_goal = np.zeros(6, dtype=np.float64)
                    remaining_duration = min(full_remaining_duration, forecast_after_switch)
                local_forecast = ShiftedForecast(
                    forecast,
                    planned_switch_delay,
                    remaining_duration,
                )
                candidate = optimize_candidate(
                    config,
                    planning_evaluator,
                    limits,
                    local_forecast,
                    q_now=q_plan,
                    qd_now=qd_plan,
                    qdd_now=qdd_plan,
                    q_goal=q_candidate_goal,
                    remaining_duration=remaining_duration,
                    verifier=verifier,
                    qd_goal=qd_candidate_goal,
                    qdd_goal=qdd_candidate_goal,
                    warm_start_trajectory=active,
                    warm_start_tau=predicted_tau,
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
                    "local_candidate": bool(cfg.USE_LOCAL_CANDIDATE),
                    "resume_tau": resume_tau,
                    "planning_alpha": planning_alpha,
                    "alpha_slot": planning_alpha,
                    "warm_start_used": candidate["optimization"]["warm_start_used"],
                }
                bridge = executed_bridge_min_distance(
                    evaluator,
                    active,
                    tau,
                    forecast,
                    horizon=planned_switch_delay,
                    alpha=planning_alpha,
                )
                event["bridge_min_distance_obs_predicted"] = bridge["distance"]
                event["bridge_time_to_min_obs_predicted"] = bridge["time"]
                event["bridge_tau_to_min_obs_predicted"] = bridge["tau"]
                event["bridge_min_distance_gt_executed"] = float(gt_distance)
                event["bridge_gt_sample_count"] = 1
                if timestamp + elapsed_s > deadline_timestamp:
                    event["expected_outcome_if_waited"] = "timeout_before_switch"
                elif not candidate["verification"]["accepted"]:
                    event["expected_outcome_if_waited"] = "rejected_before_switch"
                    event["rejection_reasons"] = candidate["verification"]["reasons"]
                pending_candidate = {
                    "trajectory": candidate["trajectory"],
                    "event": event,
                    "completed_timestamp": timestamp + elapsed_s,
                    "planned_switch_timestamp": planned_switch_timestamp,
                    "planning_alpha": planning_alpha,
                    "alpha_slot": planning_alpha,
                    "submission_accepted": bool(candidate["verification"]["accepted"]),
                    "submission_reasons": list(candidate["verification"]["reasons"]),
                    "resume_trajectory": active if cfg.USE_LOCAL_CANDIDATE else None,
                    "resume_tau": resume_tau,
                    "q_goal_candidate": q_candidate_goal,
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
            pending_bridge = executed_bridge_min_distance(
                evaluator,
                active,
                tau,
                forecast,
                horizon=max(0.0, pending_candidate["planned_switch_timestamp"] - timestamp),
                alpha=1.0,
            )
            bridge_distance = min(float(pending_bridge["distance"]), float(method_future["distance"]))
            pending_candidate["event"]["bridge_min_distance_obs_predicted"] = min(
                float(pending_candidate["event"].get("bridge_min_distance_obs_predicted", math.inf)),
                bridge_distance,
            )
            alpha = min(alpha, float(pending_candidate["alpha_slot"]))
        if method == "ssm_apf":
            correction = apf_velocity_correction(evaluator, q, forecast)
            q_exec = q + (alpha * qd_ref_now + correction) * cfg.DT
            q_exec = np.minimum(np.maximum(q_exec, limits.q_min), limits.q_max)
        tau = min(active.total_duration, tau + alpha * cfg.DT)
        execution_scale = alpha
        timestamp += cfg.DT
        if local_resume is not None and tau >= active.total_duration - 1.0e-9:
            active = local_resume["trajectory"]
            tau = min(float(local_resume["tau"]), active.total_duration)
            local_resume = None
            execution_scale = 1.0
            continue
        q_finish_check = q_exec if method == "ssm_apf" else active.evaluate(active.total_duration)
        if np.linalg.norm(q_finish_check - q_goal) <= cfg.FINISH_TOLERANCE and tau >= active.total_duration - 1.0e-9:
            break

    if pending_candidate is not None:
        event = pending_candidate["event"]
        event["outcome"] = "unfinished_pending_at_trial_end"
        event["actual_switch_timestamp"] = None
        event["candidate_accepted"] = False
        if not event.get("rejection_reasons"):
            event["rejection_reasons"] = ["trial_end_pending"]
        events.append(event)
        pending_candidate = None

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
        "last_replan_time": last_replan,
        "first_safety_hold_time": first_safety_hold,
        "false_replan": false_replan,
        "planning_control_cycles": int(planning_control_cycles),
        "bridge_min_distance_obs_predicted": (
            None
            if not events
            else min(
                (
                    float(event["bridge_min_distance_obs_predicted"])
                    for event in events
                    if event.get("bridge_min_distance_obs_predicted") is not None
                ),
                default=None,
            )
        ),
        "bridge_min_distance_gt_executed": (
            None
            if not events
            else min(
                (
                    float(event["bridge_min_distance_gt_executed"])
                    for event in events
                    if event.get("bridge_min_distance_gt_executed") is not None
                ),
                default=None,
            )
        ),
        "events": events,
        "timeline": timeline,
        "wall_elapsed_ms": float((time.perf_counter() - started) * 1000.0),
    }
