#!/usr/bin/env python3
"""Simple event-triggered dynamic NUBS bypass; shadow-only first-stage audit.

The program opens real RGB-D and robot feedback for self filtering, but never
commands the robot.  It plans one bypass event using an object-level sphere,
coarse-screens at most six geometric goals, runs Fast once on the best goal,
then performs one independent Fresh recheck.
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
        np.asarray(obstacle["velocity"]),
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


def verify_fresh_single_sphere(
    runtime_args: Any,
    config: dict[str, Any],
    model: Any,
    trajectory: Any,
    fresh: dict[str, Any],
) -> dict[str, Any]:
    forecast = trial.constant_forecast(
        np.asarray(fresh["center"], dtype=np.float64),
        np.asarray(fresh["velocity"], dtype=np.float64),
        float(fresh["radius"]),
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
    args: argparse.Namespace,
    side_lengths: tuple[float, ...],
    round_dir: Path,
) -> dict[str, Any]:
    tcp_now = tcp_position(model, q_now, args.tcp_link)
    tcp_goal = tcp_position(model, q_final, args.tcp_link)
    risk_position = np.asarray(best["risk_object"].robot_point, dtype=np.float64)
    predicted_center = np.asarray(obstacle["center"], dtype=np.float64) + np.asarray(
        obstacle["velocity"], dtype=np.float64
    ) * float(best["tau_s"])
    goals, direction_audit = bypass.bypass_goal_candidates(
        model,
        q_now,
        tcp_position=tcp_now,
        goal_position=tcp_goal,
        risk_position=risk_position,
        predicted_obstacle_position=predicted_center,
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
    eligible = [row for row in candidate_rows if row["task_progress_ok"]]
    if not eligible:
        return {"planning_ok": False, "reason": "no_forward_bypass_goal", "candidates": candidate_rows}
    selected = max(eligible, key=lambda row: row["coarse_min_distance_m"])
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
        velocity=np.asarray(obstacle["velocity"]),
        radius=float(obstacle["radius"]),
        risk_links=set(model.surface_by_link(q_now, density="coarse")),
        trial_dir=round_dir / "fast",
        reference_goal=(np.asarray(selected_goal["q_goal"]), np.zeros(6), np.zeros(6)),
        rejoin_goals=None,
        obstacle_audit={"simple_dynamic_nubs_shadow": True},
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
        "obstacle_model": "object_level_equivalent_sphere",
        "bypass_candidate_count": 6,
        "planning_robust_target_m": float(args.planning_robust_target_m),
        "online_accept_m": float(runtime_args.online_accept_m),
        "production_fast_forecast": "unchanged_dynamic_single_sphere",
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
        if float(obstacle["speed_m_s"]) < runtime_args.min_dynamic_trigger_speed_m_s:
            log["status"] = "SIMPLE_DYNAMIC_NUBS_DYNAMIC_SPEED_NOT_REACHED"
            log["measured_speed_m_s"] = float(obstacle["speed_m_s"])
            return log

        trigger_started = time.perf_counter()
        trigger_attempts = []
        short_args = argparse.Namespace(**vars(runtime_args))
        short_args.post_stop_recheck_duration_s = runtime_args.rolling_observation_duration_s
        short_args.post_stop_recheck_min_frames = runtime_args.rolling_observation_min_frames
        short_args.post_stop_recheck_min_span_s = runtime_args.rolling_observation_min_span_s
        best = None
        evaluator = None
        forecast = None
        while time.perf_counter() - trigger_started < args.trigger_timeout_s:
            prediction, best, evaluator = stro_prediction(
                trial,
                runtime_args,
                config,
                model,
                reference,
                reference_start,
                obstacle,
            )
            trigger_attempts.append(
                {
                    "obstacle_center": obstacle["center"],
                    "obstacle_velocity": obstacle["velocity"],
                    "obstacle_speed_m_s": obstacle["speed_m_s"],
                    "obstacle_radius_m": obstacle["radius"],
                    "prediction_minimum": prediction["minimum"],
                }
            )
            if best["distance_m"] < runtime_args.moving_shadow_replan_in_m:
                forecast = trial.constant_forecast(
                    np.asarray(obstacle["center"]),
                    np.asarray(obstacle["velocity"]),
                    float(obstacle["radius"]),
                )
                log["stro_prediction"] = prediction
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
            trigger_attempts[-1]["next_capture"] = {
                "accepted": bool(fresh_wait.get("accepted", False)),
                "reason": fresh_wait.get("reason"),
                "frame_count": len(wait_frames),
            }
            if fresh_wait.get("accepted", False):
                obstacle = fresh_wait
        log["trigger_wait"] = {
            "timeout_s": float(args.trigger_timeout_s),
            "elapsed_s": time.perf_counter() - trigger_started,
            "attempt_count": len(trigger_attempts),
            "attempts": trigger_attempts,
        }
        if best is None or best["distance_m"] >= runtime_args.moving_shadow_replan_in_m:
            log["status"] = "SIMPLE_DYNAMIC_NUBS_NO_PREDICTED_TRIGGER"
            return log

        planning_rounds = []
        for round_index in (1, 2):
            if round_index > 1:
                prediction, best, evaluator = stro_prediction(
                    trial,
                    runtime_args,
                    config,
                    model,
                    reference,
                    reference_start,
                    obstacle,
                )
                forecast = trial.constant_forecast(
                    np.asarray(obstacle["center"]),
                    np.asarray(obstacle["velocity"]),
                    float(obstacle["radius"]),
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
                args=args,
                side_lengths=side_lengths,
                round_dir=round_dir,
            )
            round_log = {
                "round": round_index,
                "q_now": q_now.tolist(),
                "obstacle": obstacle,
                "stro_minimum": prediction["minimum"] if round_index > 1 else log["stro_prediction"]["minimum"],
                "planning_ok": bool(planned["planning_ok"]),
                "reason": planned["reason"],
                "bypass_generation": planned.get("bypass_generation"),
                "fast": planned.get("fast"),
            }
            if not planned["planning_ok"]:
                planning_rounds.append(round_log)
                log["status"] = "SIMPLE_DYNAMIC_NUBS_FAST_HOLD"
                break
            fresh, fresh_frames, _ = trial.capture_post_stop_obstacle(
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
            if not fresh.get("accepted", False):
                round_log["fresh_candidate_verification"] = None
                planning_rounds.append(round_log)
                log["status"] = "SIMPLE_DYNAMIC_NUBS_FRESH_NOT_READY_HOLD"
                break
            fresh_verification = verify_fresh_single_sphere(
                runtime_args, config, model, planned["trajectory"], fresh
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
                "fresh_clearance_m": fresh_verification.get("min_distance"),
                "output": str((args.output / f"r{args.repeat:02d}").resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
