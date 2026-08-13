#!/usr/bin/env python3
"""Simple event-triggered dynamic NUBS bypass; shadow-only first-stage audit.

The program opens real RGB-D and robot feedback for self filtering, but never
commands the robot.  STRO retains its conservative object-level sphere.  Once
triggered, planning and verification share a Fresh fixed two-sphere forecast
and three risk-link-driven, away-side bypass goals.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

trial = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")
capture = importlib.import_module("experiments.new.6_5.6_5_3.shadow_653_rolling_local_virtual")
bypass = importlib.import_module("experiments.new.6_5.6_5_3.simple_bypass_planner")

DEFAULT_SOURCE = (
    ROOT / "results/new/6_5/6_5_3/dynamic_repair_rolling_live_xp10"
    / "trials/D2_opposing_approach_r04"
)
DEFAULT_REFERENCE = ROOT / "results/new/6_5/6_5_3/reference_xp10_line/reference_feedback.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shadow",), default="shadow")
    parser.add_argument("--source-trial", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reference-feedback-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/new/6_5/6_5_3/simple_dynamic_nubs_shadow",
    )
    parser.add_argument("--task-geometry-id", default="D2_SIMPLE_DYNAMIC_NUBS_XP10")
    parser.add_argument("--seed-timeout-s", type=float, default=10.0)
    parser.add_argument("--trigger-timeout-s", type=float, default=15.0)
    parser.add_argument("--forward-m", type=float, default=0.05)
    parser.add_argument("--side-lengths-m", default="0.04,0.06,0.08")
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.12)
    parser.add_argument("--planning-robust-target-m", type=float, default=0.11)
    parser.add_argument("--tcp-link", default="gripper_base_link")
    return parser


def update_motion_state(
    previous: dict[str, Any] | None,
    speed_m_s: float,
    *,
    enter_m_s: float = 0.08,
    exit_m_s: float = 0.04,
    exit_frames: int = 3,
) -> dict[str, Any]:
    """Update dynamic/quasi-static state without filtering out the obstacle."""
    prior = previous or {"dynamic": False, "low_speed_streak": 0}
    dynamic = bool(prior.get("dynamic", False))
    streak = int(prior.get("low_speed_streak", 0))
    speed = float(speed_m_s)
    transition = "none"
    if dynamic:
        streak = streak + 1 if speed < float(exit_m_s) else 0
        if streak >= int(exit_frames):
            dynamic = False
            transition = "dynamic_to_quasi_static"
            streak = 0
    elif speed >= float(enter_m_s):
        dynamic = True
        streak = 0
        transition = "quasi_static_to_dynamic"
    return {
        "dynamic": dynamic,
        "motion_class": "dynamic" if dynamic else "quasi_static",
        "low_speed_streak": streak,
        "transition": transition,
        "measured_speed_m_s": speed,
        "enter_threshold_m_s": float(enter_m_s),
        "exit_threshold_m_s": float(exit_m_s),
        "exit_streak_frames": int(exit_frames),
    }


def obstacle_with_motion_state(
    obstacle: dict[str, Any], motion_state: dict[str, Any]
) -> dict[str, Any]:
    """Attach prediction semantics while retaining the measured velocity."""
    result = dict(obstacle)
    measured_velocity = np.asarray(obstacle["velocity"], dtype=np.float64)
    prediction_velocity = measured_velocity if motion_state["dynamic"] else np.zeros(3)
    result["motion_state"] = dict(motion_state)
    result["motion_class"] = motion_state["motion_class"]
    result["prediction_velocity"] = prediction_velocity.tolist()
    result["prediction_model"] = (
        "local_constant_velocity" if motion_state["dynamic"] else "quasi_static_hold"
    )
    return result


def prediction_velocity(obstacle: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        obstacle.get("prediction_velocity", obstacle["velocity"]), dtype=np.float64
    )


def tcp_position(model: Any, q: np.ndarray, link: str) -> np.ndarray:
    transforms = model.urdf.link_transforms(
        {name: float(q[index]) for index, name in enumerate(model.joint_names)}
    )
    return np.asarray(transforms[link][:3, 3], dtype=np.float64)


def trajectory_minimum(evaluator: Any, forecast: Any, trajectory: Any, *, step_s: float = 0.10):
    times = np.arange(0.0, trajectory.total_duration + 0.5 * step_s, step_s)
    times = np.unique(np.r_[times, trajectory.total_duration])
    rows = []
    for tau in times:
        risk = evaluator.configuration(
            trajectory.evaluate(float(tau)),
            forecast,
            float(tau),
            density="coarse",
            with_gradient=False,
        )
        rows.append(
            {
                "tau_s": float(tau),
                "distance_m": float(risk.min_distance),
                "nearest_link": risk.nearest_link,
            }
        )
    return min(rows, key=lambda row: row["distance_m"]), rows


def stro_prediction(
    trial: Any,
    runtime_args: Any,
    config: dict[str, Any],
    model: Any,
    reference: Any,
    reference_start: float,
    obstacle: dict[str, Any],
) -> tuple[dict[str, Any], Any, Any]:
    forecast = trial.constant_forecast(
        np.asarray(obstacle["center"]),
        prediction_velocity(obstacle),
        float(obstacle["radius"]),
    )
    evaluator, _, _ = trial.make_risk_stack(config, model, forecast)
    rows = []
    best = None
    for tau in np.arange(
        0.0,
        runtime_args.prediction_horizon_s + 0.5 * runtime_args.prediction_step_s,
        runtime_args.prediction_step_s,
    ):
        q_tau = reference.state_at(reference_start + float(tau))[0]
        risk = evaluator.configuration(
            q_tau, forecast, float(tau), density="medium", with_gradient=False
        )
        row = {
            "tau_s": float(tau),
            "distance_m": float(risk.min_distance),
            "nearest_link": risk.nearest_link,
            "robot_point_m": None if risk.robot_point is None else risk.robot_point.tolist(),
            "obstacle_point_m": None if risk.obstacle_point is None else risk.obstacle_point.tolist(),
        }
        rows.append(row)
        if best is None or row["distance_m"] < best["distance_m"]:
            best = {**row, "risk_object": risk}
    return {"minimum": {k: v for k, v in best.items() if k != "risk_object"}, "profile": rows}, best, evaluator


def fit_fixed_pca_two_sphere(points: np.ndarray, *, fit_margin_m: float = 0.005) -> dict[str, Any]:
    """Fit exactly two consecutive PCA-axis spheres and audit full coverage."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 4 or not np.all(np.isfinite(values)):
        raise ValueError("Fresh cluster points must be a finite (N,3) array with N >= 4")
    mean = np.mean(values, axis=0)
    centered = values - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = np.asarray(vh[0], dtype=np.float64)
    projection = centered @ axis
    groups = np.array_split(np.argsort(projection), 2)
    centers = np.asarray([np.mean(values[group], axis=0) for group in groups])
    radii = np.asarray(
        [
            np.max(np.linalg.norm(values[group] - centers[index], axis=1)) + fit_margin_m
            for index, group in enumerate(groups)
        ],
        dtype=np.float64,
    )
    signed = np.min(
        np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2) - radii[None, :],
        axis=1,
    )
    transverse = centered - projection[:, None] * axis[None, :]
    return {
        "source_point_count": int(len(values)),
        "component_count": 2,
        "component_centers": centers,
        "component_base_radii": radii,
        "pca_axis": axis,
        "axial_length_m": float(np.ptp(projection)),
        "transverse_radius_m": float(np.percentile(np.linalg.norm(transverse, axis=1), 90)),
        "fit_margin_m": float(fit_margin_m),
        "max_point_to_union_distance": float(np.max(signed)),
        "coverage_ratio": float(np.mean(signed <= 1.0e-9)),
        "multi_sphere_max_radius": float(np.max(radii)),
        "covered": bool(np.max(signed) <= 1.0e-9),
        "fit_policy": "fixed_pca_two_sphere",
    }


def compact_prediction(
    runtime_args: Any,
    config: dict[str, Any],
    model: Any,
    reference: Any,
    reference_start: float,
    obstacle: dict[str, Any],
    geometry: dict[str, Any],
) -> tuple[dict[str, Any], Any, Any, Any]:
    forecast = trial.constant_multisphere_forecast(
        np.asarray(geometry["component_centers"], dtype=np.float64),
        np.asarray(geometry["component_base_radii"], dtype=np.float64),
        prediction_velocity(obstacle),
        object_id=int(obstacle.get("track_id") or 1),
    )
    evaluator, _, _ = trial.make_risk_stack(config, model, forecast)
    rows = []
    best = None
    for tau in np.arange(
        0.0,
        runtime_args.prediction_horizon_s + 0.5 * runtime_args.prediction_step_s,
        runtime_args.prediction_step_s,
    ):
        q_tau = reference.state_at(reference_start + float(tau))[0]
        risk = evaluator.configuration(
            q_tau,
            forecast,
            float(tau),
            density="medium",
            with_gradient=False,
        )
        row = {
            "tau_s": float(tau),
            "distance_m": float(risk.min_distance),
            "nearest_link": risk.nearest_link,
            "robot_point_m": None if risk.robot_point is None else risk.robot_point.tolist(),
            "obstacle_point_m": None if risk.obstacle_point is None else risk.obstacle_point.tolist(),
        }
        rows.append(row)
        if best is None or row["distance_m"] < best["distance_m"]:
            best = {**row, "risk_object": risk, "q_risk": np.asarray(q_tau)}
    return (
        {
            "minimum": {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in best.items()
                if key not in {"risk_object", "q_risk"}
            },
            "profile": rows,
        },
        best,
        evaluator,
        forecast,
    )


def select_robust_candidate(rows: list[dict[str, Any]], target_m: float) -> dict[str, Any] | None:
    eligible = [
        row
        for row in rows
        if row["task_progress_ok"] and row["coarse_min_distance_m"] >= float(target_m)
    ]
    return None if not eligible else max(eligible, key=lambda row: row["coarse_min_distance_m"])


def verify_fresh_two_sphere(
    runtime_args: Any,
    config: dict[str, Any],
    model: Any,
    trajectory: Any,
    fresh: dict[str, Any],
    geometry: dict[str, Any],
) -> dict[str, Any]:
    forecast = trial.constant_multisphere_forecast(
        np.asarray(geometry["component_centers"], dtype=np.float64),
        np.asarray(geometry["component_base_radii"], dtype=np.float64),
        prediction_velocity(fresh),
        object_id=int(fresh.get("track_id") or 1),
    )
    _, verifier, _ = trial.make_risk_stack(config, model, forecast)
    verifier.d_stop = float(runtime_args.online_accept_m)
    boundary = trajectory.sample(np.asarray([0.0, trajectory.total_duration]))
    result = verifier.verify(
        trajectory,
        forecast,
        current_q=boundary.q[0],
        current_qd=boundary.qd[0],
        current_qdd=boundary.qdd[0],
        q_goal=boundary.q[-1],
        solver_success=True,
    )
    return {
        "accepted": bool(result.accepted),
        "execution_authorization": False,
        "verification": asdict(result),
    }


def plan_one_bypass_round(
    *,
    trial: Any,
    runtime_args: Any,
    config: dict[str, Any],
    model: Any,
    q_now: np.ndarray,
    q_final: np.ndarray,
    obstacle: dict[str, Any],
    best: dict[str, Any],
    evaluator: Any,
    forecast: Any,
    geometry: dict[str, Any],
    args: argparse.Namespace,
    side_lengths: tuple[float, ...],
    round_dir: Path,
) -> dict[str, Any]:
    tcp_now = tcp_position(model, q_now, args.tcp_link)
    tcp_goal = tcp_position(model, q_final, args.tcp_link)
    if best["risk_object"].robot_point is None or best["risk_object"].obstacle_point is None:
        return {"planning_ok": False, "reason": "missing_ccro_surface_points", "candidates": []}
    risk_position = np.asarray(best["risk_object"].robot_point, dtype=np.float64)
    predicted_obstacle_point = np.asarray(best["risk_object"].obstacle_point, dtype=np.float64)
    goals, direction_audit = bypass.risk_link_bypass_goal_candidates(
        model,
        q_now,
        tcp_position=tcp_now,
        goal_position=tcp_goal,
        risk_link=str(best["nearest_link"]),
        risk_position=risk_position,
        predicted_obstacle_position=predicted_obstacle_point,
        risk_point_q=np.asarray(best["q_risk"], dtype=np.float64),
        forward_m=args.forward_m,
        side_lengths_m=side_lengths,
        tcp_link=args.tcp_link,
        max_joint_delta_rad=args.max_joint_delta_rad,
    )
    candidate_rows = []
    task_direction = np.asarray(direction_audit["task_direction"])
    for index, item in enumerate(goals, 1):
        goal_state = (np.asarray(item["q_goal"]), np.zeros(6), np.zeros(6))
        head, tail, durations, inner, _ = trial.make_local_reference(
            q_now, np.zeros(6), runtime_args, reference_goal=goal_state
        )
        trajectory = trial.NUBSTrajectory6D().generate(inner, head, tail, durations)
        minimum, profile = trajectory_minimum(evaluator, forecast, trajectory)
        tcp_end = tcp_position(model, trajectory.evaluate(trajectory.total_duration), args.tcp_link)
        progress = float(np.dot(tcp_end - tcp_now, task_direction))
        candidate_rows.append(
            {
                "candidate": index,
                "side_sign": item["side_sign"],
                "forward_m": item["forward_m"],
                "side_m": item["side_m"],
                "mapping": item["mapping"],
                "coarse_min_distance_m": float(minimum["distance_m"]),
                "coarse_min_tau_s": float(minimum["tau_s"]),
                "coarse_nearest_link": minimum["nearest_link"],
                "task_progress_m": progress,
                "task_progress_ok": bool(progress > 0.0),
                "robust_target_reached": bool(
                    minimum["distance_m"] >= args.planning_robust_target_m
                ),
                "profile": profile,
            }
        )
    selected = select_robust_candidate(candidate_rows, args.planning_robust_target_m)
    if selected is None:
        best_attempted = max(candidate_rows, key=lambda row: row["coarse_min_distance_m"])
        return {
            "planning_ok": False,
            "reason": "no_geometrically_robust_bypass",
            "fast_invoked": False,
            "bypass_generation": {
                "direction": direction_audit,
                "candidates": candidate_rows,
                "selected_candidate": None,
                "selected_coarse_clearance_m": None,
                "best_attempted_candidate": int(best_attempted["candidate"]),
                "best_attempted_coarse_clearance_m": float(
                    best_attempted["coarse_min_distance_m"]
                ),
                "planning_robust_target_m": float(args.planning_robust_target_m),
            },
        }
    selected_goal = goals[int(selected["candidate"]) - 1]
    generation = {
        "direction": direction_audit,
        "candidates": candidate_rows,
        "selected_candidate": int(selected["candidate"]),
        "selected_coarse_clearance_m": float(selected["coarse_min_distance_m"]),
        "robust_target_reached": bool(selected["robust_target_reached"]),
    }
    artifacts: dict[str, Any] = {}
    fast_result = trial.run_fast_repair(
        runtime_args,
        config,
        model,
        q_now=q_now,
        qd_now=np.zeros(6),
        center=np.asarray(obstacle["center"]),
        velocity=prediction_velocity(obstacle),
        radius=float(obstacle["radius"]),
        risk_links=set(model.surface_by_link(q_now, density="coarse")),
        trial_dir=round_dir / "fast",
        reference_goal=(np.asarray(selected_goal["q_goal"]), np.zeros(6), np.zeros(6)),
        rejoin_goals=None,
        obstacle_audit={"simple_dynamic_nubs_shadow": True},
        multisphere_geometry=geometry,
        artifacts_out=artifacts,
    )
    planning_ok = bool(
        fast_result["candidate_online_min_distance_m"] >= runtime_args.online_accept_m
        and fast_result["online_pipeline_elapsed_ms"] <= runtime_args.fast_budget_ms
        and all(
            ok for name, ok in fast_result["verification_checks"].items() if name != "solver_ok"
        )
    )
    return {
        "planning_ok": planning_ok,
        "reason": "planned" if planning_ok else "fast_verification_failed",
        "fast_invoked": True,
        "bypass_generation": generation,
        "fast": fast_result,
        "trajectory": artifacts["candidate_trajectory"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    side_lengths = tuple(float(value) for value in args.side_lengths_m.split(","))
    if len(side_lengths) != 3 or any(value <= 0.0 for value in side_lengths):
        raise ValueError("side-lengths-m must contain exactly three positive values")
    if args.trigger_timeout_s <= 0.0:
        raise ValueError("trigger-timeout-s must be positive")
    if args.planning_robust_target_m < 0.11:
        raise ValueError("planning-robust-target-m must remain at least 0.11 m")
    output = args.output.resolve() / f"r{args.repeat:02d}"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    source = args.source_trial.resolve()
    source_candidate = json.loads(
        (source / "candidate/candidate_summary.json").read_text(encoding="utf-8")
    )
    reference = trial.RecordedReference.load(args.reference_feedback_csv.resolve())
    runtime_args = trial.build_parser().parse_args(["--scene", "D2", "--mode", "shadow"])
    config = trial.load_stage4_config(runtime_args.stage4_config)
    model = trial.load_stage4_surface_model(config)
    q_now = np.asarray(source_candidate["q_now"], dtype=np.float64)
    q_final = reference.state_at(float(reference.times[-1]))[0]
    reference_start = capture.trigger_reference_time(source)
    log: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "INITIALIZED",
        "mode": "shadow",
        "robot_commanded": False,
        "execution_authorization": False,
        "real_rgbd": True,
        "virtual_robot_state": True,
        "task_geometry_id": args.task_geometry_id,
        "stro_obstacle_model": "conservative_equivalent_single_sphere",
        "planning_obstacle_model": "fresh_fixed_pca_two_sphere",
        "bypass_candidate_count": 3,
        "bypass_jacobian_policy": "ccro_nearest_risk_link_plus_tcp_task_progress",
        "planning_robust_target_m": float(args.planning_robust_target_m),
        "online_accept_m": float(runtime_args.online_accept_m),
        "speed_semantics": "classification_only_not_a_planning_gate",
        "motion_state_thresholds": {
            "dynamic_enter_m_s": float(runtime_args.min_dynamic_trigger_speed_m_s),
            "dynamic_exit_m_s": float(runtime_args.dynamic_exit_speed_m_s),
            "dynamic_exit_streak_frames": int(runtime_args.dynamic_exit_streak_frames),
        },
        "production_fast_forecast": (
            "fresh_fixed_pca_two_sphere_with_motion_class_dependent_velocity"
        ),
    }
    processor = None
    started = time.perf_counter()
    try:
        processor = trial.SceneProcessor(
            config_dir=str(runtime_args.config_dir),
            urdf_path=str(runtime_args.urdf),
            width=runtime_args.width,
            height=runtime_args.height,
            threshold=runtime_args.self_filter_threshold,
            voxel_size=runtime_args.voxel_size,
            use_real_robot=True,
            use_mock_camera=False,
        )
        reader = getattr(processor, "_state_reader", None)
        if reader is None or type(reader).__name__ != "RealRobotStateReader":
            raise RuntimeError("real RGB-D with AUBO state reader is required")
        denoiser = (
            trial.TemporalDenoiser(
                runtime_args.denoise_voxel,
                runtime_args.denoise_conf,
                runtime_args.denoise_decay,
            )
            if runtime_args.temporal_denoise
            else None
        )
        trial.require_confirmation(
            True,
            "SIMPLE DYNAMIC NUBS SHADOW ONLY: the robot will not move. Press Enter, then "
            f"within {args.seed_timeout_s:.1f}s move one compact foam obstacle continuously "
            "into the future task corridor and keep it moving through the capture.",
        )
        obstacle = capture.first_external_seed(
            processor, denoiser, runtime_args, timeout_s=args.seed_timeout_s
        )
        trial.write_json(output / "dynamic_seed.json", obstacle)
        if not obstacle.get("accepted", False):
            log["status"] = "SIMPLE_DYNAMIC_NUBS_SEED_FAILED"
            return log
        motion_state = update_motion_state(
            None,
            float(obstacle["speed_m_s"]),
            enter_m_s=runtime_args.min_dynamic_trigger_speed_m_s,
            exit_m_s=runtime_args.dynamic_exit_speed_m_s,
            exit_frames=runtime_args.dynamic_exit_streak_frames,
        )
        obstacle = obstacle_with_motion_state(obstacle, motion_state)
        log["seed_speed_m_s"] = float(obstacle["speed_m_s"])
        log["seed_motion_state"] = motion_state

        trigger_started = time.perf_counter()
        trigger_attempts = []
        short_args = argparse.Namespace(**vars(runtime_args))
        short_args.post_stop_recheck_duration_s = runtime_args.rolling_observation_duration_s
        short_args.post_stop_recheck_min_frames = runtime_args.rolling_observation_min_frames
        short_args.post_stop_recheck_min_span_s = runtime_args.rolling_observation_min_span_s
        best = None
        evaluator = None
        forecast = None
        dynamic_attempts = 0
        quasi_static_attempts = 0
        while time.perf_counter() - trigger_started < args.trigger_timeout_s:
            speed = float(obstacle["speed_m_s"])
            if obstacle["motion_class"] == "dynamic":
                dynamic_attempts += 1
            else:
                quasi_static_attempts += 1
            attempt = {
                "obstacle_center": obstacle["center"],
                "obstacle_velocity": obstacle["velocity"],
                "prediction_velocity": obstacle["prediction_velocity"],
                "obstacle_speed_m_s": speed,
                "obstacle_radius_m": obstacle["radius"],
                "motion_class": obstacle["motion_class"],
                "motion_state": obstacle["motion_state"],
            }
            trigger_attempts.append(attempt)
            prediction, best, evaluator = stro_prediction(
                trial,
                runtime_args,
                config,
                model,
                reference,
                reference_start,
                obstacle,
            )
            attempt["prediction_minimum"] = prediction["minimum"]
            if best["distance_m"] < runtime_args.moving_shadow_replan_in_m:
                forecast = trial.constant_forecast(
                    np.asarray(obstacle["center"]),
                    prediction_velocity(obstacle),
                    float(obstacle["radius"]),
                )
                log["stro_prediction"] = prediction
                log["trigger_motion_state"] = obstacle["motion_state"]
                break
            fresh_wait, wait_frames, _ = trial.capture_post_stop_obstacle(
                processor,
                reader,
                denoiser,
                short_args,
                trigger_cluster_center=np.asarray(obstacle["center"]),
                trigger_velocity=np.asarray(obstacle["velocity"]),
                trigger_timestamp=float(obstacle["last_timestamp"]),
                stop_when_ready=True,
            )
            attempt["next_capture"] = {
                "accepted": bool(fresh_wait.get("accepted", False)),
                "reason": fresh_wait.get("reason"),
                "frame_count": len(wait_frames),
            }
            if fresh_wait.get("accepted", False):
                motion_state = update_motion_state(
                    motion_state,
                    float(fresh_wait["speed_m_s"]),
                    enter_m_s=runtime_args.min_dynamic_trigger_speed_m_s,
                    exit_m_s=runtime_args.dynamic_exit_speed_m_s,
                    exit_frames=runtime_args.dynamic_exit_streak_frames,
                )
                obstacle = obstacle_with_motion_state(fresh_wait, motion_state)
        log["trigger_wait"] = {
            "timeout_s": float(args.trigger_timeout_s),
            "elapsed_s": time.perf_counter() - trigger_started,
            "attempt_count": len(trigger_attempts),
            "dynamic_attempt_count": int(dynamic_attempts),
            "quasi_static_attempt_count": int(quasi_static_attempts),
            "attempts": trigger_attempts,
        }
        if best is None or best["distance_m"] >= runtime_args.moving_shadow_replan_in_m:
            log["status"] = "SIMPLE_DYNAMIC_NUBS_NO_PREDICTED_TRIGGER"
            return log

        # The conservative STRO sphere ends here.  Acquire one independent
        # Fresh point cluster and use its fixed two-sphere fit consistently for
        # coarse screening, Fast and the subsequent Fresh verifier.
        planning_obstacle, planning_frames, planning_points = trial.capture_post_stop_obstacle(
            processor,
            reader,
            denoiser,
            runtime_args,
            trigger_cluster_center=np.asarray(obstacle["center"]),
            trigger_velocity=np.asarray(obstacle["velocity"]),
            trigger_timestamp=float(obstacle["last_timestamp"]),
            stop_when_ready=True,
        )
        trial.write_json(
            output / "planning_fresh_capture.json",
            {"result": planning_obstacle, "frames": planning_frames},
        )
        if not planning_obstacle.get("accepted", False) or planning_points is None:
            log["status"] = "SIMPLE_DYNAMIC_NUBS_PLANNING_FRESH_NOT_READY_HOLD"
            return log
        motion_state = update_motion_state(
            motion_state,
            float(planning_obstacle["speed_m_s"]),
            enter_m_s=runtime_args.min_dynamic_trigger_speed_m_s,
            exit_m_s=runtime_args.dynamic_exit_speed_m_s,
            exit_frames=runtime_args.dynamic_exit_streak_frames,
        )
        obstacle = obstacle_with_motion_state(planning_obstacle, motion_state)
        log["planning_fresh_motion_state"] = motion_state
        trial.write_json(output / "planning_prediction_state.json", obstacle)
        geometry = fit_fixed_pca_two_sphere(planning_points)
        trial.write_json(output / "planning_fresh_two_sphere.json", geometry)
        if not geometry["covered"]:
            log["status"] = "SIMPLE_DYNAMIC_NUBS_PLANNING_GEOMETRY_COVERAGE_HOLD"
            return log

        planning_rounds = []
        for round_index in (1, 2):
            prediction, best, evaluator, forecast = compact_prediction(
                runtime_args,
                config,
                model,
                reference,
                reference_start,
                obstacle,
                geometry,
            )
            round_dir = output / f"planning_round_{round_index:02d}"
            planned = plan_one_bypass_round(
                trial=trial,
                runtime_args=runtime_args,
                config=config,
                model=model,
                q_now=q_now,
                q_final=q_final,
                obstacle=obstacle,
                best=best,
                evaluator=evaluator,
                forecast=forecast,
                geometry=geometry,
                args=args,
                side_lengths=side_lengths,
                round_dir=round_dir,
            )
            round_log = {
                "round": round_index,
                "q_now": q_now.tolist(),
                "obstacle": obstacle,
                "stro_single_sphere_minimum": log["stro_prediction"]["minimum"],
                "planning_two_sphere_minimum": prediction["minimum"],
                "planning_two_sphere_geometry": geometry,
                "planning_ok": bool(planned["planning_ok"]),
                "reason": planned["reason"],
                "fast_invoked": bool(planned.get("fast_invoked", False)),
                "bypass_generation": planned.get("bypass_generation"),
                "fast": planned.get("fast"),
            }
            if not planned["planning_ok"]:
                planning_rounds.append(round_log)
                log["status"] = (
                    "SIMPLE_DYNAMIC_NUBS_ROBUST_BYPASS_HOLD"
                    if planned["reason"] == "no_geometrically_robust_bypass"
                    else "SIMPLE_DYNAMIC_NUBS_FAST_HOLD"
                )
                break
            fresh, fresh_frames, fresh_points = trial.capture_post_stop_obstacle(
                processor,
                reader,
                denoiser,
                runtime_args,
                trigger_cluster_center=np.asarray(obstacle["center"]),
                trigger_velocity=np.asarray(obstacle["velocity"]),
                trigger_timestamp=float(obstacle["last_timestamp"]),
                stop_when_ready=True,
            )
            trial.write_json(
                round_dir / "fresh_recheck.json", {"result": fresh, "frames": fresh_frames}
            )
            round_log["fresh_recheck"] = fresh
            if not fresh.get("accepted", False) or fresh_points is None:
                round_log["fresh_candidate_verification"] = None
                planning_rounds.append(round_log)
                log["status"] = "SIMPLE_DYNAMIC_NUBS_FRESH_NOT_READY_HOLD"
                break
            motion_state = update_motion_state(
                motion_state,
                float(fresh["speed_m_s"]),
                enter_m_s=runtime_args.min_dynamic_trigger_speed_m_s,
                exit_m_s=runtime_args.dynamic_exit_speed_m_s,
                exit_frames=runtime_args.dynamic_exit_streak_frames,
            )
            fresh = obstacle_with_motion_state(fresh, motion_state)
            round_log["fresh_recheck"] = fresh
            round_log["fresh_motion_state"] = motion_state
            trial.write_json(round_dir / "fresh_prediction_state.json", fresh)
            fresh_geometry = fit_fixed_pca_two_sphere(fresh_points)
            trial.write_json(round_dir / "fresh_two_sphere.json", fresh_geometry)
            round_log["fresh_two_sphere_geometry"] = fresh_geometry
            if not fresh_geometry["covered"]:
                round_log["fresh_candidate_verification"] = None
                planning_rounds.append(round_log)
                log["status"] = "SIMPLE_DYNAMIC_NUBS_FRESH_GEOMETRY_COVERAGE_HOLD"
                break
            fresh_verification = verify_fresh_two_sphere(
                runtime_args, config, model, planned["trajectory"], fresh, fresh_geometry
            )
            round_log["fresh_candidate_verification"] = fresh_verification
            planning_rounds.append(round_log)
            if fresh_verification["accepted"]:
                log["status"] = "SIMPLE_DYNAMIC_NUBS_SHADOW_SUCCESS"
                break
            if round_index == 1:
                # Event-triggered replan from exactly the same physical/virtual
                # robot state, using only the latest Fresh object estimate.
                obstacle = fresh
                geometry = fresh_geometry
                continue
            log["status"] = "SIMPLE_DYNAMIC_NUBS_SECOND_FRESH_RECHECK_HOLD"
        log["planning_rounds"] = planning_rounds
        if planning_rounds:
            final_round = planning_rounds[-1]
            log["bypass_generation"] = final_round.get("bypass_generation")
            log["fast"] = final_round.get("fast")
            log["fresh_recheck"] = final_round.get("fresh_recheck")
            log["fresh_candidate_verification"] = final_round.get(
                "fresh_candidate_verification"
            )
        log["elapsed_s"] = time.perf_counter() - started
    except Exception as exc:
        log["status"] = "SIMPLE_DYNAMIC_NUBS_SHADOW_FAILED"
        log["error"] = str(exc)
        log["traceback"] = traceback.format_exc(limit=20)
    finally:
        log.setdefault("elapsed_s", time.perf_counter() - started)
        if processor is not None:
            processor.stop()
        trial.write_json(summary_path, log)
    return log


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    bypass_summary = result.get("bypass_generation") or {}
    fresh_summary = result.get("fresh_candidate_verification") or {}
    fresh_verification = fresh_summary.get("verification") or {}
    print(
        json.dumps(
            {
                "status": result["status"],
                "robot_commanded": result["robot_commanded"],
                "selected_coarse_clearance_m": bypass_summary.get(
                    "selected_coarse_clearance_m"
                ),
                "best_attempted_coarse_clearance_m": bypass_summary.get(
                    "best_attempted_coarse_clearance_m"
                ),
                "fresh_clearance_m": fresh_verification.get("min_distance"),
                "output": str((args.output / f"r{args.repeat:02d}").resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
