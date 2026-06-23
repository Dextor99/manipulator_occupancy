"""Stage-four asynchronous replanning and safety-hold experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import yaml

from experiments.exp_ccro_stage3 import _select_sweep_point
from planning.dynamic_optimizer import DynamicRiskNUBSOptimizer
from planning.nubs_trajectory import NUBSTrajectory6D
from planning.obstacle_forecast import ConstantVelocitySphereForecast
from planning.optimizer import FixedTimeNUBSOptimizer, JointLimits
from planning.replanner import FutureRiskReport, ReplanManager, RiskLevel
from planning.robot_surface_model import RobotSurfaceModel
from planning.spatiotemporal_risk import SpatioTemporalRiskEvaluator
from planning.trajectory_buffer import TrajectoryBuffer
from planning.verifier import DynamicTrajectoryVerifier


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "value"):
        return value.value
    raise TypeError(type(value).__name__)


def _states(config):
    cfg = config["trajectory"]
    head = NUBSTrajectory6D.make_boundary_state(
        cfg["q_start"], cfg["qd_start"], cfg["qdd_start"]
    )
    tail = NUBSTrajectory6D.make_boundary_state(
        cfg["q_goal"], cfg["qd_goal"], cfg["qdd_goal"]
    )
    durations = np.asarray(cfg["segment_durations"], dtype=np.float64)
    return head, tail, durations


def _limits(config):
    cfg = config["robot"]
    return JointLimits.from_arrays(
        cfg["q_min"], cfg["q_max"], cfg["qd_max"], cfg["qdd_max"]
    )


def _baseline(config, head, tail, durations, limits):
    cfg = config["optimizer"]
    optimizer = FixedTimeNUBSOptimizer(
        head,
        tail,
        durations,
        limits,
        lambda_smooth=cfg["lambda_smooth"],
        lambda_position=cfg["lambda_position"],
        lambda_velocity=cfg["lambda_velocity"],
        lambda_acceleration=cfg["lambda_acceleration"],
        samples_per_segment=cfg["samples_per_segment"],
        finite_difference_epsilon=cfg["finite_difference_epsilon"],
        max_iterations=cfg["max_iterations"],
        gradient_tolerance=cfg["gradient_tolerance"],
    )
    initial = NUBSTrajectory6D.linear_inner_points(head[:, 0], tail[:, 0], durations)
    return optimizer.optimize(initial)


def _make_forecast(config, scenario, model, baseline):
    fc = config["forecast"]
    kind = scenario["type"]
    if kind == "far":
        center = np.mean(model.surface(baseline.evaluate(0.5 * baseline.total_duration), "coarse"), axis=0)
        center += np.array([1.0, 0.8, 1.0])
        velocity = np.zeros(3)
        target = None
    elif kind == "high":
        points = model.surface_by_link(
            baseline.evaluate(0.0), "medium", set(scenario["target_links"])
        )
        link = next(iter(points))
        center = points[link][0].copy()
        velocity = np.zeros(3)
        target = {"target_link": link, "collision_time": 0.0, "target_clearance": 0.0}
    else:
        selected = _select_sweep_point(
            model, baseline, scenario["target_links"], scenario["time_range"], []
        )
        radius_at_collision = (
            fc["base_radius"]
            + fc["margin"]
            + fc["uncertainty"]
            + fc["uncertainty_growth"] * selected["time"]
            + fc["velocity_radius_scale"] * float(scenario["speed"]) * selected["time"]
        )
        collision_center = selected["surface_point"] + (
            radius_at_collision + float(scenario["target_clearance"])
        ) * selected["outward"]
        directions = []
        for axis in np.eye(3):
            tangent = np.cross(selected["outward"], axis)
            norm = float(np.linalg.norm(tangent))
            if norm > 1.0e-6:
                tangent /= norm
                directions.extend([tangent, -tangent])
        # Diagonal tangents enlarge the search without changing the target point.
        if len(directions) >= 4:
            for first in directions[:2]:
                for second in directions[2:4]:
                    diagonal = first + second
                    norm = float(np.linalg.norm(diagonal))
                    if norm > 1.0e-6:
                        directions.extend([diagonal / norm, -diagonal / norm])
        best = None
        for direction in directions:
            candidate_velocity = float(scenario["speed"]) * direction
            candidate_center = collision_center - candidate_velocity * selected["time"]
            minimum = math.inf
            initial = math.inf
            for index, tau in enumerate(np.linspace(0.0, baseline.total_duration, 49)):
                points = model.surface(baseline.evaluate(float(tau)), "coarse")
                radius = (
                    fc["base_radius"] + fc["margin"] + fc["uncertainty"]
                    + fc["uncertainty_growth"] * tau
                    + fc["velocity_radius_scale"] * float(scenario["speed"]) * tau
                )
                distance = float(
                    np.min(np.maximum(
                        np.linalg.norm(points - (candidate_center + candidate_velocity * tau), axis=1)
                        - radius,
                        0.0,
                    ))
                )
                if index == 0:
                    initial = distance
                minimum = min(minimum, distance)
            score = (int(initial > config["replanner"]["d_replan"] + 0.02), minimum)
            if best is None or score > best[0]:
                best = (score, candidate_center, candidate_velocity, initial, minimum)
        if best is None:
            raise RuntimeError("failed to construct a dynamic crossing direction")
        _, center, velocity, initial_clearance, sampled_clearance = best
        target = {
            "target_link": selected["link"],
            "collision_time": selected["time"],
            "target_clearance": scenario["target_clearance"],
            "time_alignment_error": float(
                np.linalg.norm(center + velocity * selected["time"] - collision_center)
            ),
            "sampled_initial_clearance": initial_clearance,
            "sampled_path_clearance": sampled_clearance,
        }
    forecast = ConstantVelocitySphereForecast(
        center,
        velocity,
        fc["base_radius"],
        fc["valid_horizon"],
        margin=fc["margin"],
        uncertainty=fc["uncertainty"],
        uncertainty_growth=fc["uncertainty_growth"],
        velocity_radius_scale=fc["velocity_radius_scale"],
        beyond_horizon=fc["beyond_horizon"],
    )
    return forecast, {
        "type": kind,
        "initial_center": center.tolist(),
        "velocity": velocity.tolist(),
        "target": target,
    }


def _actual_distance(evaluator, q, forecast, timestamp, density="dense"):
    return evaluator.configuration(
        q, forecast, float(timestamp), density=density, with_gradient=False
    )


def _passive(trajectory, evaluator, forecast, dt):
    timeline = []
    for timestamp in np.arange(0.0, trajectory.total_duration + 0.5 * dt, dt):
        q = trajectory.evaluate(min(float(timestamp), trajectory.total_duration))
        risk = _actual_distance(evaluator, q, forecast, timestamp)
        timeline.append(
            {
                "time": float(timestamp),
                "q": q.tolist(),
                "actual_min_distance": risk.min_distance,
                "nearest_link": risk.nearest_link,
            }
        )
    return timeline


def _make_components(config, model, limits, forecast):
    rc, vc, oc = config["risk"], config["validation"], config["optimizer"]
    evaluator = SpatioTemporalRiskEvaluator(
        model,
        d_safe=rc["d_safe"],
        d_activate=rc["d_activate"],
        fd_epsilon_q=rc["fd_epsilon_q"],
        density=rc["optimizer_density"],
    )
    verifier = DynamicTrajectoryVerifier(
        evaluator,
        limits,
        d_stop=rc["d_stop"],
        time_step=vc["dense_time_step"],
        density=rc["validation_density"],
        epsilon_goal=vc["epsilon_goal"],
        epsilon_continuity_q=vc["epsilon_continuity_q"],
        epsilon_continuity_qd=vc["epsilon_continuity_qd"],
        epsilon_continuity_qdd=vc["epsilon_continuity_qdd"],
        limit_tolerance=vc["limit_tolerance"],
    )

    def factory(head, tail, durations, local_forecast):
        return DynamicRiskNUBSOptimizer(
            head,
            tail,
            durations,
            limits,
            evaluator,
            local_forecast,
            lambda_risk=oc["lambda_risk"],
            risk_samples_per_segment=rc["risk_samples_per_segment"],
            lambda_smooth=oc["lambda_smooth"],
            lambda_position=oc["lambda_position"],
            lambda_velocity=oc["lambda_velocity"],
            lambda_acceleration=oc["lambda_acceleration"],
            samples_per_segment=oc["samples_per_segment"],
            finite_difference_epsilon=oc["finite_difference_epsilon"],
            sensitivity_epsilon=oc["sensitivity_epsilon"],
            max_iterations=oc["max_iterations"],
            gradient_tolerance=oc["gradient_tolerance"],
        )

    return evaluator, verifier, factory


def _active_loop(config, baseline, q_goal, durations, evaluator, verifier, factory, forecast):
    rp = config["replanner"]
    manager = ReplanManager(
        factory,
        evaluator,
        forecast,
        verifier,
        _limits(config),
        d_replan=rp["d_replan"],
        d_stop=rp["d_stop"],
        d_safe=rp["d_safe"],
        d_accept=rp["d_accept"],
        min_improvement=rp["min_improvement"],
        replan_interval=rp["replan_interval"],
        hysteresis_enter=rp["hysteresis_enter"],
        hysteresis_exit=rp["hysteresis_exit"],
        evaluate_steps=rp["evaluate_steps"],
        evaluate_horizon=rp["evaluate_horizon"],
        max_replan_attempts=rp["max_replan_attempts"],
        planning_budget=rp["planning_budget"],
        switch_delay=rp["switch_delay"],
        emergency_lead_time=rp["emergency_lead_time"],
    )
    buffer = TrajectoryBuffer()
    buffer.set_active(baseline, 0.0, q_goal)
    dt = float(config["virtual_loop"]["dt"])
    timeline = []
    started = time.perf_counter()
    planning_cycle_count = 0
    accepted_count = 0
    last_report = FutureRiskReport(RiskLevel.LOW, math.inf, 0.0, math.inf, None, None)
    try:
        while True:
            timestamp = time.perf_counter() - started
            if timestamp > config["forecast"]["valid_horizon"]:
                manager.engage_safety(timestamp, "forecast horizon exhausted")
                break
            poll = manager.poll_candidate(timestamp, buffer, q_goal)
            accepted_count += int(poll.outcome == "accepted")
            if not manager.replan_in_flight and not manager.safety_hold_required:
                last_report = manager.evaluate_active_trajectory(timestamp, buffer)
                if last_report.level == RiskLevel.HIGH:
                    buffer.pause(timestamp)
                    manager.engage_safety(timestamp, "future D_min <= d_stop")
                elif last_report.level == RiskLevel.MEDIUM:
                    manager.submit_replan(
                        timestamp, buffer, q_goal, durations, last_report
                    )
            if manager.replan_in_flight:
                planning_cycle_count += 1
            q, qd, _ = buffer.sample_state(timestamp)
            actual = _actual_distance(evaluator, q, forecast, timestamp)
            if actual.min_distance <= config["risk"]["d_stop"]:
                manager.abort_for_safety(
                    timestamp, buffer, "current D_min <= d_stop"
                )
            timeline.append(
                {
                    "time": timestamp,
                    "tau": buffer.elapsed(timestamp),
                    "q": q.tolist(),
                    "qd": qd.tolist(),
                    "qd_norm": float(np.linalg.norm(qd)),
                    "actual_min_distance": actual.min_distance,
                    "future_min_distance": last_report.min_distance,
                    "risk_level": manager.current_level.value,
                    "planner_in_flight": manager.replan_in_flight,
                    "safety_hold": manager.safety_hold_required,
                }
            )
            if manager.current_level == RiskLevel.HIGH and not manager.replan_in_flight:
                break
            if buffer.is_finished_at(timestamp):
                break
            sleep_for = dt - ((time.perf_counter() - started) - timestamp)
            if sleep_for > 0.0:
                time.sleep(sleep_for)
    finally:
        manager.shutdown()
    q_final = buffer.sample_now(time.perf_counter() - started)
    return {
        "timeline": timeline,
        "events": [asdict(item) for item in manager.replan_events],
        "safety_events": [asdict(item) for item in manager.safety_events],
        "replan_count": manager.replan_count,
        "accepted_count": accepted_count,
        "planning_control_cycles": planning_cycle_count,
        "goal_error": float(np.linalg.norm(q_final - q_goal)),
        "finished": bool(np.linalg.norm(q_final - q_goal) <= config["virtual_loop"]["finish_tolerance"]),
    }


def _summarise_timeline(timeline, dt):
    return {
        "sample_count": len(timeline),
        "min_actual_distance": float(min(row["actual_min_distance"] for row in timeline)),
        "hold_time": float(sum(row.get("safety_hold", False) for row in timeline) * dt),
    }


def run(config_path, output_override=None):
    config_path = Path(config_path).resolve()
    config = _load(config_path)
    output = Path(output_override or config["experiment"]["output_dir"])
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "config.yaml")
    robot, surface = config["robot"], config["surface"]
    model = RobotSurfaceModel(
        ROOT / robot["urdf_path"],
        robot["joint_names"],
        surface["density_totals"],
        seed=surface["random_seed"],
        min_points_per_link=surface["min_points_per_link"],
        cache_dir=surface["cache_dir"],
        geometry=surface["geometry"],
    )
    head, tail, durations = _states(config)
    limits = _limits(config)
    base_result = _baseline(config, head, tail, durations, limits)
    baseline = base_result.trajectory
    q_goal = tail[:, 0]
    scenarios = {}
    accepted_flags = []
    for name, scenario in config["experiment"]["scenarios"].items():
        forecast, forecast_info = _make_forecast(config, scenario, model, baseline)
        evaluator, verifier, factory = _make_components(config, model, limits, forecast)
        passive_timeline = _passive(
            baseline, evaluator, forecast, config["virtual_loop"]["dt"]
        )
        active = _active_loop(
            config, baseline, q_goal, durations, evaluator, verifier, factory, forecast
        )
        dt = float(config["virtual_loop"]["dt"])
        passive_summary = _summarise_timeline(passive_timeline, dt)
        active_summary = _summarise_timeline(active.pop("timeline"), dt)
        expectation = scenario["expected"]
        if expectation == "accepted_replan":
            ok = bool(
                active["accepted_count"] >= 1
                and active["finished"]
                and active_summary["min_actual_distance"] >= config["risk"]["d_stop"]
                and active_summary["min_actual_distance"]
                > passive_summary["min_actual_distance"] + 0.002
                and not active["safety_events"]
            )
        elif expectation == "no_trigger":
            ok = bool(
                active["replan_count"] == 0
                and active["finished"]
                and not active["safety_events"]
            )
        else:
            ok = bool(
                active["accepted_count"] == 0
                and len(active["safety_events"]) == 1
                and active_summary["min_actual_distance"] <= config["risk"]["d_stop"]
            )
        scenarios[name] = {
            "accepted": ok,
            "name": scenario["name"],
            "expectation": expectation,
            "forecast": forecast_info,
            "passive": passive_summary,
            "active": {**active_summary, **active},
        }
        accepted_flags.append(ok)
    elapsed = [
        event["elapsed_ms"]
        for row in scenarios.values()
        for event in row["active"]["events"]
        if event["outcome"] == "accepted"
    ]
    timing = {
        "accepted_replans": len(elapsed),
        "mean_ms": float(np.mean(elapsed)) if elapsed else None,
        "p95_ms": float(np.percentile(elapsed, 95)) if elapsed else None,
        "max_ms": float(np.max(elapsed)) if elapsed else None,
        "hard_budget_ms": 1000.0 * config["replanner"]["planning_budget"],
        "meets_1hz": bool(elapsed and np.percentile(elapsed, 95) <= 1000.0),
    }
    metrics = {
        "accepted": bool(all(accepted_flags)),
        "scope": "asynchronous reference-level virtual closed loop; no real-robot commands",
        "surface_counts": model.sample_counts(),
        "timing": timing,
        "scenarios": scenarios,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    _write_table(metrics, output / "table_stage4.md")
    return metrics


def _write_table(metrics, path):
    lines = [
        "| scenario | expected | pass | passive D_min / m | active D_min / m | replans | accepted | safety | goal error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in metrics["scenarios"].items():
        lines.append(
            f"| {name} | {row['expectation']} | {row['accepted']} | "
            f"{row['passive']['min_actual_distance']:.6f} | "
            f"{row['active']['min_actual_distance']:.6f} | "
            f"{row['active']['replan_count']} | {row['active']['accepted_count']} | "
            f"{len(row['active']['safety_events'])} | {row['active']['goal_error']:.3g} |"
        )
    lines.extend(["", f"Overall: **{'PASS' if metrics['accepted'] else 'FAIL'}**", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "ccro_stage4.yaml"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    metrics = run(args.config, args.output)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default))
    if not metrics["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
