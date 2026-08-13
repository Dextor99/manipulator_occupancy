#!/usr/bin/env python3
"""Real-RGB-D, virtual-robot Static20 goal-directed rolling Fast shadow."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib
import json
import math
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
legacy_shadow = importlib.import_module(
    "experiments.new.6_5.6_5_3.shadow_653_rolling_local_virtual"
)
policy = importlib.import_module(
    "experiments.new.6_5.6_5_3.goal_directed_rolling_common"
)

DEFAULT_SOURCE = (
    ROOT / "results/new/6_5/6_5_3/dynamic_repair_rolling_live_xp10"
    / "trials/D2_opposing_approach_r04"
)
DEFAULT_REFERENCE = ROOT / "results/new/6_5/6_5_3/reference_xp10_line/reference_feedback.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trial", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reference-feedback-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/new/6_5/6_5_3/static20_goal_directed_virtual_shadow",
    )
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--task-geometry-id", default="ROF1_STATIC20_GOAL_DIRECTED_XP10")
    parser.add_argument("--seed-timeout-s", type=float, default=10.0)
    parser.add_argument("--max-wall-s", type=float, default=240.0)
    parser.add_argument("--static-inflation-m", type=float, default=0.020)
    parser.add_argument("--risk-goal-max-delta-rad", type=float, default=0.030)
    parser.add_argument("--terminal-goal-max-step-rad", type=float, default=0.030)
    parser.add_argument("--goal-tolerance-rad", type=float, default=0.010)
    parser.add_argument("--tcp-link", default="gripper_base_link")
    return parser


def static_state(fresh: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(fresh)
    if result.get("accepted", False):
        result["measured_velocity"] = list(result.get("velocity", [0.0, 0.0, 0.0]))
        result["measured_speed_m_s"] = float(result.get("speed_m_s", 0.0))
        result["velocity"] = [0.0, 0.0, 0.0]
        result["speed_m_s"] = 0.0
        result["motion_model"] = "static_zero_velocity"
    return result


def rolling_wall_expired(rolling_started: float, max_wall_s: float, *, now: float | None = None) -> bool:
    current = time.perf_counter() if now is None else float(now)
    return bool(current - float(rolling_started) >= float(max_wall_s))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.repeat < 1 or args.seed_timeout_s <= 0.0 or args.max_wall_s <= 0.0:
        raise ValueError("repeat and time limits must be positive")
    if args.static_inflation_m < 0.0 or args.risk_goal_max_delta_rad < 0.0:
        raise ValueError("Static20 and risk bounds must be non-negative")
    source = args.source_trial.resolve()
    output = args.output.resolve() / f"r{args.repeat:02d}"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    source_candidate = json.loads(
        (source / "candidate/candidate_summary.json").read_text(encoding="utf-8")
    )
    reference = trial.RecordedReference.load(args.reference_feedback_csv.resolve())
    runtime_args = trial.build_parser().parse_args(["--scene", "D2", "--mode", "shadow"])
    config = trial.load_stage4_config(runtime_args.stage4_config)
    model = trial.load_stage4_surface_model(config)
    anchor = legacy_shadow.trigger_reference_time(source)
    q_virtual = np.asarray(source_candidate["q_now"], dtype=np.float64)
    q_final = reference.state_at(float(reference.times[-1]))[0]
    initial_tcp = policy.tcp_position(model, q_virtual, args.tcp_link)
    final_tcp = policy.tcp_position(model, q_final, args.tcp_link)
    task_y_direction = float(np.sign(final_tcp[1] - initial_tcp[1]))
    reference_limit = int(math.ceil((float(reference.times[-1]) - anchor) / runtime_args.local_horizon_s))
    terminal_bound = int(
        math.ceil(float(np.max(np.abs(q_final - q_virtual))) / args.terminal_goal_max_step_rad)
    )
    maximum_segments = reference_limit + terminal_bound + 2
    log: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "INITIALIZED",
        "robot_commanded": False,
        "real_rgbd": True,
        "virtual_robot_state": True,
        "execution_authorization": False,
        "static_observation_inflation_m": float(args.static_inflation_m),
        "production_dynamic_forecast_changed": False,
        "source_trial": str(source),
        "task_geometry_id": args.task_geometry_id,
        "maximum_segments_derived": maximum_segments,
        "reference_transport_segment_limit": reference_limit,
        "terminal_segment_bound": terminal_bound,
        "online_accept_m": float(runtime_args.online_accept_m),
        "fast_budget_ms": float(runtime_args.fast_budget_ms),
        "local_horizon_s": float(runtime_args.local_horizon_s),
        "risk_goal_max_delta_rad": float(args.risk_goal_max_delta_rad),
        "segments": [],
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
            "STATIC20 VIRTUAL SHADOW ONLY: the robot will not move. Place the static "
            "tabletop obstacle in the planned corridor, press Enter, and keep it stationary.",
        )
        seed = static_state(
            legacy_shadow.first_external_seed(
                processor, denoiser, runtime_args, timeout_s=args.seed_timeout_s
            )
        )
        trial.write_json(output / "seed.json", seed)
        if not seed.get("accepted", False):
            log["status"] = "GOAL_DIRECTED_STATIC20_SHADOW_SEED_FAILED"
            return log
        previous = {
            "center": np.asarray(seed["center"], dtype=np.float64),
            "velocity": np.zeros(3),
            "last_timestamp": float(seed["last_timestamp"]),
        }
        # Operator setup and the blocking Enter prompt are not part of the
        # rolling-computation budget.  Start the wall clock only after a valid
        # static RGB-D seed exists.
        rolling_started = time.perf_counter()
        short_args = copy.copy(runtime_args)
        short_args.post_stop_recheck_duration_s = runtime_args.rolling_observation_duration_s
        short_args.post_stop_recheck_min_frames = runtime_args.rolling_observation_min_frames
        short_args.post_stop_recheck_min_span_s = runtime_args.rolling_observation_min_span_s
        locked_side = None

        for segment_index in range(1, maximum_segments + 1):
            if rolling_wall_expired(rolling_started, args.max_wall_s):
                log["status"] = "GOAL_DIRECTED_STATIC20_SHADOW_WALL_LIMIT_HOLD"
                break
            final_error = float(np.max(np.abs(q_final - q_virtual)))
            if final_error <= args.goal_tolerance_rad:
                log["status"] = "GOAL_DIRECTED_STATIC20_SHADOW_SUCCESS"
                break
            remaining_reference = float(reference.times[-1]) - anchor
            if remaining_reference > 1.0e-6:
                local_goal, goal_audit = policy.transported_reference_goal(
                    reference, q_virtual, anchor, runtime_args.local_horizon_s
                )
                increment = np.asarray(goal_audit["reference_increment_rad"])
                if float(np.max(np.abs(increment))) >= runtime_args.min_local_motion_rad:
                    phase = "reference_transport"
                else:
                    anchor = float(reference.times[-1])
                    local_goal, goal_audit = policy.bounded_terminal_goal(
                        q_virtual, q_final, max_step_rad=args.terminal_goal_max_step_rad
                    )
                    goal_audit.update({"reference_anchor_time_s": anchor, "reference_goal_time_s": anchor})
                    phase = "terminal_goal"
            else:
                local_goal, goal_audit = policy.bounded_terminal_goal(
                    q_virtual, q_final, max_step_rad=args.terminal_goal_max_step_rad
                )
                goal_audit.update(
                    {
                        "reference_anchor_time_s": float(reference.times[-1]),
                        "reference_goal_time_s": float(reference.times[-1]),
                    }
                )
                phase = "terminal_goal"

            segment_started = time.perf_counter()
            segment_deadline = min(
                rolling_started + args.max_wall_s,
                segment_started + runtime_args.rolling_fast_max_s,
            )
            segment: dict[str, Any] = {
                "segment": segment_index,
                "progress_phase": phase,
                **goal_audit,
                "q_virtual_start": q_virtual.tolist(),
                "attempts": [],
            }
            advanced = False
            while time.perf_counter() < segment_deadline:
                attempt_index = len(segment["attempts"]) + 1
                attempt_dir = output / f"segment_{segment_index:02d}" / f"attempt_{attempt_index:02d}"
                fresh_plan, plan_frames, plan_points = trial.capture_post_stop_obstacle(
                    processor,
                    reader,
                    denoiser,
                    short_args,
                    trigger_cluster_center=np.asarray(previous["center"]),
                    trigger_velocity=np.zeros(3),
                    trigger_timestamp=float(previous["last_timestamp"]),
                    stop_when_ready=True,
                )
                fresh_plan = static_state(fresh_plan)
                trial.write_json(attempt_dir / "fresh_plan.json", {"result": fresh_plan, "frames": plan_frames})
                if plan_points is not None:
                    np.save(attempt_dir / "fresh_plan_points.npy", np.asarray(plan_points))
                attempt: dict[str, Any] = {
                    "attempt": attempt_index,
                    "fresh_plan": fresh_plan,
                    "virtual_authorization": False,
                    "execution_authorization": False,
                }
                if not fresh_plan.get("accepted", False) or plan_points is None:
                    attempt["status"] = "FRESH_PLAN_NOT_READY"
                    segment["attempts"].append(attempt)
                    continue
                plan_geometry = trial.fit_pca_multisphere(
                    plan_points,
                    fit_margin_m=runtime_args.multisphere_fit_margin_m,
                    max_components=runtime_args.multisphere_max_components,
                )
                if not plan_geometry["covered"]:
                    attempt["status"] = "FRESH_PLAN_COVERAGE_FAILED"
                    segment["attempts"].append(attempt)
                    continue
                plan_forecast = policy.make_static20_forecast(
                    plan_geometry,
                    observation_inflation_m=args.static_inflation_m,
                    valid_horizon_s=max(2.0, runtime_args.local_horizon_s + 1.0),
                )
                evaluator, _, _ = trial.make_risk_stack(config, model, plan_forecast)

                def plan_goal(label: str, goal: Any, side_delta: np.ndarray | None = None):
                    artifacts: dict[str, Any] = {}
                    result = trial.run_fast_repair(
                        runtime_args,
                        config,
                        model,
                        q_now=q_virtual,
                        qd_now=np.zeros(6),
                        center=np.asarray(fresh_plan["center"]),
                        velocity=np.zeros(3),
                        radius=float(fresh_plan["radius"]),
                        risk_links=set(model.surface_by_link(q_virtual, density="coarse")),
                        trial_dir=attempt_dir / label,
                        reference_goal=goal,
                        rejoin_goals=None,
                        obstacle_audit={
                            "static20_goal_directed_virtual_shadow": True,
                            "static20_shadow_forecast_override_authorized": True,
                            "segment": segment_index,
                            "attempt": attempt_index,
                            "goal_mode": label,
                        },
                        multisphere_geometry=plan_geometry,
                        artifacts_out=artifacts,
                        forecast_override=plan_forecast,
                    )
                    candidate = artifacts["candidate_trajectory"]
                    verifier_pass = policy.complete_verifier_pass(
                        result,
                        online_accept_m=runtime_args.online_accept_m,
                        fast_budget_ms=runtime_args.fast_budget_ms,
                    )
                    delta = (
                        np.asarray(side_delta)
                        if side_delta is not None
                        else np.asarray(result["tail_delta_q_rad"])
                    )
                    side = trial.avoidance_side_consistent(
                        locked_side,
                        delta,
                        opposite_projection_tolerance_rad=runtime_args.rolling_side_opposite_tolerance_rad,
                    )
                    if policy.terminal_side_release_allowed(phase, verifier_pass) and not side["accepted"]:
                        side = {**side, "accepted": True, "reason": "side_lock_released_for_verified_terminal_goal"}
                    return result, artifacts, candidate, side, bool(verifier_pass and side["accepted"])

                result, artifacts, candidate, side, planning_ok = plan_goal(
                    "transported_task_goal", local_goal
                )
                goal_mode = "transported_task_goal"
                risk_audit = None
                exact_terminal = bool(
                    phase == "terminal_goal"
                    and float(goal_audit.get("terminal_step_scale", 0.0)) >= 1.0 - 1.0e-12
                )
                if exact_terminal:
                    nominal = artifacts["reference_trajectory"]
                    nominal_check, _ = policy.verify_static20_virtual_candidate(
                        trial,
                        runtime_args,
                        config,
                        model,
                        nominal,
                        plan_geometry,
                        observation_inflation_m=args.static_inflation_m,
                    )
                    if nominal_check["virtual_authorization"]:
                        candidate = nominal
                        planning_ok = True
                        side = {**side, "accepted": True, "reason": "verified_exact_terminal_goal"}
                        goal_mode = "exact_terminal_nominal"
                if not planning_ok:
                    guided_goal, risk_audit = policy.risk_guided_goal(
                        evaluator,
                        plan_forecast,
                        local_goal,
                        max_delta_rad=args.risk_goal_max_delta_rad,
                    )
                    if guided_goal is not None:
                        result, artifacts, candidate, side, planning_ok = plan_goal(
                            "risk_guided_goal",
                            guided_goal,
                            np.asarray(risk_audit["delta_q_risk_rad"]),
                        )
                        goal_mode = "risk_guided_goal"
                attempt.update(
                    {
                        "goal_mode": goal_mode,
                        "risk_guided_goal_audit": risk_audit,
                        "planning_result": result,
                        "planning_side": side,
                        "planning_complete_verifier_pass": planning_ok,
                    }
                )
                if not planning_ok:
                    attempt["status"] = "PLANNING_CANDIDATE_REJECTED"
                    segment["attempts"].append(attempt)
                    previous = fresh_plan
                    continue

                fresh_auth, auth_frames, auth_points = trial.capture_post_stop_obstacle(
                    processor,
                    reader,
                    denoiser,
                    short_args,
                    trigger_cluster_center=np.asarray(fresh_plan["center"]),
                    trigger_velocity=np.zeros(3),
                    trigger_timestamp=float(fresh_plan["last_timestamp"]),
                    stop_when_ready=True,
                )
                fresh_auth = static_state(fresh_auth)
                trial.write_json(attempt_dir / "fresh_auth.json", {"result": fresh_auth, "frames": auth_frames})
                if auth_points is not None:
                    np.save(attempt_dir / "fresh_auth_points.npy", np.asarray(auth_points))
                if not fresh_auth.get("accepted", False) or auth_points is None:
                    attempt["status"] = "FRESH_AUTH_NOT_READY"
                    segment["attempts"].append(attempt)
                    previous = fresh_auth if fresh_auth.get("accepted", False) else fresh_plan
                    continue
                auth_geometry = trial.fit_pca_multisphere(
                    auth_points,
                    fit_margin_m=runtime_args.multisphere_fit_margin_m,
                    max_components=runtime_args.multisphere_max_components,
                )
                if not auth_geometry["covered"]:
                    attempt["status"] = "FRESH_AUTH_COVERAGE_FAILED"
                    segment["attempts"].append(attempt)
                    previous = fresh_auth
                    continue
                virtual_auth, _ = policy.verify_static20_virtual_candidate(
                    trial,
                    runtime_args,
                    config,
                    model,
                    candidate,
                    auth_geometry,
                    observation_inflation_m=args.static_inflation_m,
                )
                attempt["fresh_static20_verification"] = virtual_auth
                attempt["virtual_authorization"] = bool(virtual_auth["virtual_authorization"])
                attempt["execution_authorization"] = False
                if not virtual_auth["virtual_authorization"]:
                    attempt["status"] = "FRESH_STATIC20_REJECTED"
                    segment["attempts"].append(attempt)
                    previous = fresh_auth
                    continue

                q_next = np.asarray(candidate.evaluate(candidate.total_duration), dtype=np.float64)
                tcp_start = policy.tcp_position(model, q_virtual, args.tcp_link)
                tcp_end = policy.tcp_position(model, q_next, args.tcp_link)
                tcp_y_progress = float(task_y_direction * (tcp_end[1] - tcp_start[1]))
                goal_before = float(np.max(np.abs(q_final - q_virtual)))
                goal_after = float(np.max(np.abs(q_final - q_next)))
                phase_progress = tcp_y_progress > 0.0 if phase == "reference_transport" else goal_after < goal_before
                attempt.update(
                    {
                        "status": "VIRTUAL_SEGMENT_AUTHORIZED" if phase_progress else "VIRTUAL_SEGMENT_NO_PHASE_PROGRESS",
                        "q_virtual_end": q_next.tolist(),
                        "tcp_y_task_progress_m": tcp_y_progress,
                        "goal_error_before_max_abs_rad": goal_before,
                        "goal_error_after_max_abs_rad": goal_after,
                        "phase_progress_ok": bool(phase_progress),
                    }
                )
                segment["attempts"].append(attempt)
                if not phase_progress:
                    break
                if locked_side is None:
                    locked_side = np.asarray(side["locked_tail_delta_q"], dtype=np.float64)
                q_virtual = q_next
                anchor = (
                    min(float(reference.times[-1]), anchor + runtime_args.local_horizon_s)
                    if phase == "reference_transport"
                    else float(reference.times[-1])
                )
                previous = fresh_auth
                segment.update(
                    {
                        "status": "VIRTUAL_SEGMENT_AUTHORIZED",
                        "authorized_attempt": attempt_index,
                        "q_virtual_end": q_virtual.tolist(),
                    }
                )
                advanced = True
                break
            segment["attempt_count"] = len(segment["attempts"])
            segment["segment_elapsed_s"] = time.perf_counter() - segment_started
            log["segments"].append(segment)
            if not advanced:
                log["status"] = "GOAL_DIRECTED_STATIC20_SHADOW_HOLD"
                break
        else:
            log["status"] = "GOAL_DIRECTED_STATIC20_SHADOW_SEGMENT_BOUND_HOLD"

        final_error = float(np.max(np.abs(q_final - q_virtual)))
        if final_error <= args.goal_tolerance_rad:
            log["status"] = "GOAL_DIRECTED_STATIC20_SHADOW_SUCCESS"
        log["segments_accepted"] = sum(
            item.get("status") == "VIRTUAL_SEGMENT_AUTHORIZED" for item in log["segments"]
        )
        log["final_goal_error_max_abs_rad"] = final_error
        log["reference_anchor_monotonic"] = all(
            log["segments"][i]["reference_anchor_time_s"]
            <= log["segments"][i + 1]["reference_anchor_time_s"]
            for i in range(len(log["segments"]) - 1)
        )
        log["elapsed_s"] = time.perf_counter() - started
        log["rolling_elapsed_s"] = time.perf_counter() - rolling_started
    except Exception as exc:
        log["status"] = "GOAL_DIRECTED_STATIC20_SHADOW_FAILED"
        log["error"] = str(exc)
        log["traceback"] = traceback.format_exc(limit=20)
    finally:
        if processor is not None:
            processor.stop()
        trial.write_json(summary_path, log)
    return log


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "robot_commanded": result["robot_commanded"],
                "segments_accepted": result.get("segments_accepted", 0),
                "final_goal_error_max_abs_rad": result.get("final_goal_error_max_abs_rad"),
                "output": str((args.output / f"r{args.repeat:02d}").resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
