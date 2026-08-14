#!/usr/bin/env python3
"""Protected two-event continuation for the validated simple dynamic NUBS pilot.

The first local segment is exactly the r04-frozen implementation.  At its
measured tail, Fresh #3 first evaluates whether remaining stationary is
physically safe over the 0.5 s prediction horizon.  If not, one additional
Fresh-authorized local segment may execute from the measured joints.  A direct
terminal NUBS to the recorded preset goal is allowed only after a new complete
verification.  No third local replan is permitted; unresolved approaching risk
is reported as requiring operator intervention, not as a "safe hold".
"""

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
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

trial = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")
live = importlib.import_module("experiments.new.6_5.6_5_3.run_653_simple_dynamic_nubs_live")
bypass = importlib.import_module("experiments.new.6_5.6_5_3.simple_bypass_planner")

DEFAULT_OUTPUT = ROOT / "results/new/6_5/6_5_3/simple_dynamic_nubs_event_replan_live"
EVENT_EXECUTE_PHRASE = "CCRO_653_SIMPLE_DYNAMIC_EVENT_REPLAN_EXECUTE_APPROVED"


def build_parser() -> argparse.ArgumentParser:
    parser = live.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        output=DEFAULT_OUTPUT,
        task_geometry_id="D2_SIMPLE_DYNAMIC_NUBS_EVENT_REPLAN_LIVE_XP10",
    )
    parser.add_argument(
        "--terminal-durations-s",
        default="3.0,4.0,5.0,6.0",
        help="bounded terminal NUBS duration candidates; verifier chooses the first safe one",
    )
    parser.add_argument(
        "--event-operator-phrase",
        default="",
        help=f"required with --execute: {EVENT_EXECUTE_PHRASE}",
    )
    parser.add_argument(
        "--post-local-monitor-max-s",
        type=float,
        default=3.0,
        help="bounded Fresh monitoring time while the measured tail is physically safe",
    )
    parser.add_argument(
        "--continuation-side-m",
        type=float,
        default=0.04,
        help="strong retained-side displacement for local #2; weak uses half and release uses zero",
    )
    return parser


def strict_empty_scene(args: argparse.Namespace, frames: list[dict[str, Any]]) -> bool:
    required = int(args.post_stop_recheck_min_frames)
    tail = frames[-required:] if len(frames) >= required else []
    return bool(
        len(tail) == required
        and all(bool(frame.get("frame_valid", False)) for frame in tail)
        and all(not frame.get("all_external_clusters", []) for frame in tail)
        and all(
            float(frame.get("raw_guard_distance_m", -math.inf))
            > float(args.guided_hard_stop_m)
            for frame in tail
        )
    )


def forecast_from_fresh(
    args: argparse.Namespace,
    fresh: dict[str, Any],
    geometry: dict[str, Any] | None,
    frames: list[dict[str, Any]],
) -> tuple[Any | None, str]:
    if fresh.get("accepted", False) and geometry is not None and geometry.get("covered", False):
        return (
            trial.constant_multisphere_forecast(
                np.asarray(geometry["component_centers"], dtype=np.float64),
                np.asarray(geometry["component_base_radii"], dtype=np.float64),
                np.asarray(fresh["velocity"], dtype=np.float64),
            ),
            "FRESH_TRACKED_OBSTACLE",
        )
    if strict_empty_scene(args, frames):
        return (
            trial.constant_multisphere_forecast(
                np.asarray([[100.0, 100.0, 100.0]], dtype=np.float64),
                np.asarray([1.0e-6], dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            ),
            "STRICT_THREE_FRAME_EMPTY_ROI",
        )
    return None, "FRESH_ASSOCIATION_OR_SCENE_CLEAR_NOT_ESTABLISHED"


def stationary_hold_audit(
    args: argparse.Namespace,
    config: dict[str, Any],
    model: Any,
    q_actual: np.ndarray,
    forecast: Any,
) -> dict[str, Any]:
    evaluator, _, _ = trial.make_risk_stack(config, model, None)
    profile = []
    minimum = math.inf
    nearest = None
    minimum_tau = None
    for tau in np.arange(
        0.0,
        float(args.prediction_horizon_s) + 0.5 * float(args.prediction_step_s),
        float(args.prediction_step_s),
    ):
        risk = evaluator.configuration(q_actual, forecast, float(tau), density="medium", with_gradient=False)
        row = {
            "tau_s": float(tau),
            "distance_m": float(risk.min_distance),
            "nearest_link": risk.nearest_link,
        }
        profile.append(row)
        if row["distance_m"] < minimum:
            minimum = row["distance_m"]
            nearest = row["nearest_link"]
            minimum_tau = row["tau_s"]
    return {
        "predicted_min_distance_m": float(minimum),
        "predicted_min_tau_s": minimum_tau,
        "nearest_link": nearest,
        "threshold_m": float(args.moving_shadow_replan_in_m),
        "physical_hold_safe": bool(minimum >= args.moving_shadow_replan_in_m),
        "profile": profile,
    }


def fit_fresh_geometry(
    args: argparse.Namespace, fresh: dict[str, Any], points: np.ndarray | None
) -> dict[str, Any] | None:
    if not fresh.get("accepted", False) or points is None:
        return None
    geometry = trial.fit_pca_multisphere(
        points,
        fit_margin_m=args.multisphere_fit_margin_m,
        max_components=args.multisphere_max_components,
    )
    if not geometry.get("covered", False):
        return None
    return geometry


def capture_next_fresh(
    args: argparse.Namespace,
    processor: Any,
    state_reader: Any,
    denoiser: Any,
    previous: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray | None, dict[str, Any] | None]:
    fresh, frames, points = trial.capture_post_stop_obstacle(
        processor,
        state_reader,
        denoiser,
        args,
        trigger_cluster_center=np.asarray(previous["center"], dtype=np.float64),
        trigger_velocity=np.asarray(previous["velocity"], dtype=np.float64),
        trigger_timestamp=float(previous["last_timestamp"]),
        stop_when_ready=True,
    )
    return fresh, frames, points, fit_fresh_geometry(args, fresh, points)


def monitor_measured_tail(
    args: argparse.Namespace,
    config: dict[str, Any],
    model: Any,
    processor: Any,
    state_reader: Any,
    denoiser: Any,
    q_actual: np.ndarray,
    *,
    initial_fresh: dict[str, Any],
    initial_frames: list[dict[str, Any]],
    initial_geometry: dict[str, Any] | None,
    output_dir: Path,
    max_wall_s: float,
) -> dict[str, Any]:
    """Monitor a stopped measured tail without equating it to task safety."""
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fresh = initial_fresh
    frames = initial_frames
    geometry = initial_geometry
    cycles = []
    while True:
        scene_clear = strict_empty_scene(args, frames)
        forecast, basis = forecast_from_fresh(args, fresh, geometry, frames)
        cycle: dict[str, Any] = {
            "cycle": len(cycles),
            "elapsed_s": time.perf_counter() - started,
            "fresh": fresh,
            "forecast_basis": basis,
            "strict_empty_scene": scene_clear,
        }
        if scene_clear and forecast is not None:
            cycle["decision"] = "STRICT_SCENE_CLEAR"
            cycles.append(cycle)
            result = {
                "status": "STRICT_SCENE_CLEAR",
                "cycles": cycles,
                "fresh": fresh,
                "frames": frames,
                "geometry": geometry,
                "forecast": forecast,
            }
            trial.write_json(output_dir / "monitor_summary.json", {k: v for k, v in result.items() if k != "forecast"})
            return result
        if forecast is None:
            cycle["decision"] = "OBSERVATION_UNCERTAIN"
            cycles.append(cycle)
            result = {
                "status": "OBSERVATION_UNCERTAIN",
                "cycles": cycles,
                "fresh": fresh,
                "frames": frames,
                "geometry": geometry,
                "forecast": None,
            }
            trial.write_json(output_dir / "monitor_summary.json", {k: v for k, v in result.items() if k != "forecast"})
            return result
        hold = stationary_hold_audit(args, config, model, q_actual, forecast)
        cycle["stationary_hold"] = hold
        if not hold["physical_hold_safe"]:
            cycle["decision"] = "REPLAN_REQUIRED"
            cycles.append(cycle)
            result = {
                "status": "REPLAN_REQUIRED",
                "cycles": cycles,
                "fresh": fresh,
                "frames": frames,
                "geometry": geometry,
                "forecast": forecast,
            }
            trial.write_json(output_dir / "monitor_summary.json", {k: v for k, v in result.items() if k != "forecast"})
            return result
        cycle["decision"] = "PHYSICAL_HOLD_SAFE_MONITORING"
        cycles.append(cycle)
        if time.perf_counter() - started >= max_wall_s:
            result = {
                "status": "PHYSICAL_HOLD_SAFE_MONITORING_TIMEOUT",
                "cycles": cycles,
                "fresh": fresh,
                "frames": frames,
                "geometry": geometry,
                "forecast": forecast,
            }
            trial.write_json(output_dir / "monitor_summary.json", {k: v for k, v in result.items() if k != "forecast"})
            return result
        previous = fresh
        fresh, frames, points, geometry = capture_next_fresh(
            args, processor, state_reader, denoiser, previous
        )
        index = len(cycles)
        trial.write_json(
            output_dir / f"fresh_monitor_{index:02d}.json",
            {"result": fresh, "frames": frames},
        )
        if points is not None:
            np.save(output_dir / f"fresh_monitor_{index:02d}_points.npy", points)
        if geometry is not None:
            trial.write_json(output_dir / f"fresh_monitor_{index:02d}_geometry.json", geometry)


def make_terminal_trajectory(
    q_now: np.ndarray, q_goal: np.ndarray, duration_s: float
) -> Any:
    segments = max(5, int(math.ceil(float(duration_s) / 0.20)))
    durations = np.full(segments, float(duration_s) / segments, dtype=np.float64)
    head = trial.NUBSTrajectory6D.make_boundary_state(q_now, np.zeros(6), np.zeros(6))
    tail = trial.NUBSTrajectory6D.make_boundary_state(q_goal, np.zeros(6), np.zeros(6))
    inner = trial.NUBSTrajectory6D.linear_inner_points(q_now, q_goal, durations)
    return trial.NUBSTrajectory6D().generate(inner, head, tail, durations)


def next_recorded_reference_goal(reference: Any, q_actual: np.ndarray, horizon_s: float):
    """Select a nearby forward nominal state without mutating online progress."""
    errors = np.max(np.abs(np.asarray(reference.q) - q_actual[None, :]), axis=1)
    nearest = int(np.argmin(errors))
    steps = max(1, int(round(float(horizon_s) / float(reference.dt_median))))
    index = min(len(reference.q) - 1, nearest + steps)
    return (
        np.asarray(reference.q[index], dtype=np.float64),
        np.asarray(reference.qd[index], dtype=np.float64),
        np.asarray(reference.qdd[index], dtype=np.float64),
    ), {"nearest_reference_index": nearest, "forward_reference_index": index}


def established_bypass_side(
    model: Any,
    q_escape_start: np.ndarray,
    q_now: np.ndarray,
    q_final: np.ndarray,
    *,
    tcp_link: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    p_start = live.simple.tcp_position(model, np.asarray(q_escape_start), tcp_link)
    p_now = live.simple.tcp_position(model, np.asarray(q_now), tcp_link)
    p_final = live.simple.tcp_position(model, np.asarray(q_final), tcp_link)
    task = bypass.normalized(p_final - p_now)
    executed = p_now - p_start
    lateral = executed - task * float(np.dot(task, executed))
    if np.linalg.norm(lateral) <= 1.0e-6:
        raise RuntimeError("local #1 did not establish a measurable bypass side")
    side = bypass.normalized(lateral)
    return side, {
        "tcp_escape_start_m": p_start.tolist(),
        "tcp_now_m": p_now.tolist(),
        "tcp_final_m": p_final.tolist(),
        "executed_tcp_delta_m": executed.tolist(),
        "task_direction": task.tolist(),
        "lateral_executed_delta_m": lateral.tolist(),
        "established_bypass_side": side.tolist(),
        "semantic": "lock_side_not_constant_direction",
    }


def select_goal_directed_continuation(
    rows: list[dict[str, Any]],
    *,
    robust_target_m: float,
    diagnostic_only: bool = False,
) -> dict[str, Any] | None:
    """Rank continuation seeds without weakening the final verifier.

    Legacy event recovery retains its 0.11 m coarse hard gate.  V3 uses that
    number only as a preference: every finite seed may enter Fast, while the
    unchanged 0.09 m complete verifier remains the execution authority.
    """
    if diagnostic_only:
        eligible = [
            row for row in rows if math.isfinite(row["coarse_min_distance_m"])
        ]
        return (
            max(
                eligible,
                key=lambda row: (
                    row["coarse_min_distance_m"] >= robust_target_m,
                    row["coarse_min_distance_m"],
                    row["task_progress_m"],
                    -row["goal_distance_m"],
                ),
            )
            if eligible
            else None
        )
    safe = [
        row
        for row in rows
        if row["coarse_min_distance_m"] >= robust_target_m
        and row["task_progress_ok"]
    ]
    return (
        max(safe, key=lambda row: (row["task_progress_m"], -row["goal_distance_m"]))
        if safe
        else None
    )


def plan_goal_directed_continuation(
    base_fast: Any,
    runtime_args: argparse.Namespace,
    config: dict[str, Any],
    model: Any,
    *,
    q_escape_start: np.ndarray,
    q_now: np.ndarray,
    q_final: np.ndarray,
    fresh: dict[str, Any],
    geometry: dict[str, Any],
    risk_links: set[str],
    trial_dir: Path,
    nominal_reference_goal: tuple[np.ndarray, np.ndarray, np.ndarray],
    artifacts_out: dict[str, Any],
    forward_m: float,
    side_m: float,
    robust_target_m: float,
    max_joint_delta_rad: float,
    tcp_link: str,
    robust_target_is_diagnostic: bool = False,
) -> dict[str, Any]:
    """Select strong/weak/release goal progress, then invoke unchanged Fast."""
    forecast = trial.constant_multisphere_forecast(
        np.asarray(geometry["component_centers"], dtype=np.float64),
        np.asarray(geometry["component_base_radii"], dtype=np.float64),
        np.asarray(fresh["velocity"], dtype=np.float64),
    )
    evaluator, _, _ = trial.make_risk_stack(config, model, forecast)
    nominal = live.nominal_local_risk(
        runtime_args,
        model,
        evaluator,
        forecast,
        np.asarray(q_now, dtype=np.float64),
        nominal_reference_goal,
    )
    risk = nominal["risk_object"]
    if risk.robot_point is None or risk.obstacle_point is None:
        return {
            "status": "REJECTED_GOAL_DIRECTED_MISSING_RISK_POINTS",
            "local_repair_status": "REJECTED_GOAL_DIRECTED_MISSING_RISK_POINTS",
            "local_repair_ready": False,
            "accepted_for_switch": False,
            "rejection_reasons": ["missing_ccro_surface_points"],
        }
    side, side_audit = established_bypass_side(
        model, q_escape_start, q_now, q_final, tcp_link=tcp_link
    )
    tcp_now = live.simple.tcp_position(model, np.asarray(q_now), tcp_link)
    tcp_final = live.simple.tcp_position(model, np.asarray(q_final), tcp_link)
    goals, direction = bypass.goal_directed_side_continuation_candidates(
        model,
        np.asarray(q_now),
        tcp_position=tcp_now,
        goal_position=tcp_final,
        risk_link=str(nominal["nearest_link"]),
        risk_position=np.asarray(risk.robot_point),
        risk_point_q=np.asarray(nominal["q_risk"]),
        established_side=side,
        forward_m=float(forward_m),
        side_m=float(side_m),
        side_weights=(1.0, 0.5, 0.0),
        tcp_link=tcp_link,
        max_joint_delta_rad=float(max_joint_delta_rad),
    )
    task = np.asarray(direction["task_direction"], dtype=np.float64)
    rows = []
    for item in goals:
        goal_state = (np.asarray(item["q_goal"]), np.zeros(6), np.zeros(6))
        head, tail, durations, inner, _ = trial.make_local_reference(
            np.asarray(q_now), np.zeros(6), runtime_args, reference_goal=goal_state
        )
        trajectory = trial.NUBSTrajectory6D().generate(inner, head, tail, durations)
        minimum, profile = live.simple.trajectory_minimum(evaluator, forecast, trajectory)
        tcp_end = live.simple.tcp_position(
            model, trajectory.evaluate(trajectory.total_duration), tcp_link
        )
        progress = float(np.dot(tcp_end - tcp_now, task))
        rows.append(
            {
                **{key: item[key] for key in ("candidate", "phase", "side_weight", "side_m", "forward_m")},
                "mapping": item["mapping"],
                "coarse_min_distance_m": float(minimum["distance_m"]),
                "coarse_min_tau_s": float(minimum["tau_s"]),
                "coarse_nearest_link": minimum["nearest_link"],
                "task_progress_m": progress,
                "goal_distance_m": float(np.linalg.norm(tcp_final - tcp_end)),
                "task_progress_ok": bool(progress > 0.0),
                "profile": profile,
            }
        )
    selected = select_goal_directed_continuation(
        rows,
        robust_target_m=float(robust_target_m),
        diagnostic_only=bool(robust_target_is_diagnostic),
    )
    audit = {
        "policy": "goal_directed_bypass_continuation",
        "q_escape_start_rad": np.asarray(q_escape_start).tolist(),
        "q_actual_continuation_start_rad": np.asarray(q_now).tolist(),
        "side_audit": side_audit,
        "direction": direction,
        "candidates": rows,
        "planning_robust_target_m": float(robust_target_m),
        "planning_robust_target_is_hard_gate": bool(
            not robust_target_is_diagnostic
        ),
        "selected_candidate": None if selected is None else int(selected["candidate"]),
        "selected_phase": None if selected is None else selected["phase"],
        "fast_invoked": selected is not None,
    }
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial.write_json(trial_dir / "goal_directed_continuation_audit.json", audit)
    if selected is None:
        return {
            "status": "REJECTED_NO_GOAL_DIRECTED_ROBUST_CONTINUATION",
            "local_repair_status": "REJECTED_NO_GOAL_DIRECTED_ROBUST_CONTINUATION",
            "local_repair_ready": False,
            "accepted_for_switch": False,
            "rejection_reasons": ["no_goal_directed_candidate_at_or_above_0.11m"],
            "goal_directed_continuation_audit": audit,
        }
    selected_goal = goals[int(selected["candidate"]) - 1]
    result = base_fast(
        runtime_args,
        config,
        model,
        q_now=np.asarray(q_now),
        qd_now=np.zeros(6),
        center=np.asarray(fresh["center"]),
        velocity=np.asarray(fresh["velocity"]),
        radius=float(fresh["radius"]),
        risk_links=risk_links,
        trial_dir=trial_dir,
        reference_goal=(np.asarray(selected_goal["q_goal"]), np.zeros(6), np.zeros(6)),
        rejoin_goals=None,
        obstacle_audit={"track_id": 1, "event_local_index": 2, "phase": "bypass_progression"},
        multisphere_geometry=geometry,
        artifacts_out=artifacts_out,
    )
    result["goal_directed_continuation_audit"] = audit
    return result


def authorize_terminal_goal(
    args: argparse.Namespace,
    config: dict[str, Any],
    model: Any,
    q_now: np.ndarray,
    q_goal: np.ndarray,
    forecast: Any,
    durations: tuple[float, ...],
    output_dir: Path,
) -> tuple[dict[str, Any], Any | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _, verifier, _ = trial.make_risk_stack(config, model, None)
    attempts = []
    selected = None
    for duration in durations:
        trajectory = make_terminal_trajectory(q_now, q_goal, duration)
        verification = verifier.verify(
            trajectory,
            forecast,
            current_q=q_now,
            current_qd=np.zeros(6),
            current_qdd=np.zeros(6),
            q_goal=q_goal,
            solver_success=True,
        )
        row = {
            "duration_s": float(duration),
            "accepted": bool(verification.accepted),
            "min_distance_m": float(verification.min_distance),
            "checks": verification.checks,
            "reasons": verification.reasons,
            "verification_ms": float(verification.validation_ms),
        }
        attempts.append(row)
        if verification.accepted:
            selected = trajectory
            break
    csv_path = output_dir / "authorized_terminal_goal.csv"
    if csv_path.exists():
        csv_path.unlink()
    if selected is not None:
        trial.save_trajectory_csv(csv_path, selected, dt=0.01)
    payload = {
        "status": "TERMINAL_GOAL_AUTHORIZED" if selected is not None else "TERMINAL_GOAL_HOLD",
        "authorized": selected is not None,
        "attempts": attempts,
        "authorized_trajectory_csv": str(csv_path) if selected is not None else None,
        "q_start_rad": q_now.tolist(),
        "q_goal_rad": q_goal.tolist(),
    }
    trial.write_json(output_dir / "authorization_summary.json", payload)
    return payload, selected


def make_event_handler(event_args: argparse.Namespace, terminal_durations: tuple[float, ...]):
    def handler(**context: Any) -> dict[str, Any]:
        args = context["args"]
        config = context["stage4_config"]
        model = context["stage4_model"]
        robot = context["robot"]
        processor = context["processor"]
        state_reader = context["state_reader"]
        denoiser = context["denoiser"]
        trial_dir = Path(context["trial_dir"])
        q_actual = np.asarray(robot.get_joint(), dtype=np.float64)
        q_expected = np.asarray(
            context["local_artifacts"]["candidate_trajectory"].evaluate(
                context["local_artifacts"]["candidate_trajectory"].total_duration
            ),
            dtype=np.float64,
        )
        start_error = trial.joint_error(q_actual, q_expected)
        result: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "handled": True,
            "status": "EVENT_REPLAN_INITIALIZED",
            "command_hold": True,
            "q_actual_local1_tail_rad": q_actual.tolist(),
            "local1_tail_error": start_error,
            "max_local_executions": 2,
        }
        if start_error["max_abs_rad"] > args.candidate_start_tolerance_rad:
            result["status"] = "EVENT_REPLAN_BLOCKED_LOCAL1_TAIL_MISMATCH"
            trial.write_json(trial_dir / "event_replan_summary.json", result)
            return result

        result["fresh3_raw_guard_distance_m"] = float(context["fresh3_guard_distance"])
        if context["fresh3_guard_distance"] <= args.guided_hard_stop_m:
            result["status"] = "HOLD_UNCERTAIN_OPERATOR_INTERVENTION_REQUIRED"
            trial.write_json(trial_dir / "event_replan_summary.json", result)
            return result
        monitor1 = monitor_measured_tail(
            args,
            config,
            model,
            processor,
            state_reader,
            denoiser,
            q_actual,
            initial_fresh=context["fresh3"],
            initial_frames=context["fresh3_frames"],
            initial_geometry=context["fresh3_geometry"],
            output_dir=trial_dir / "post_local1_monitor",
            max_wall_s=float(event_args.post_local_monitor_max_s),
        )
        result["post_local1_monitor"] = {
            key: value for key, value in monitor1.items() if key not in {"forecast", "geometry"}
        }
        if monitor1["status"] == "PHYSICAL_HOLD_SAFE_MONITORING_TIMEOUT":
            result["status"] = "PHYSICAL_HOLD_SAFE_MONITORING_LIMIT_OPERATOR_CONTROL_REQUIRED"
            trial.write_json(trial_dir / "event_replan_summary.json", result)
            return result
        if monitor1["status"] == "OBSERVATION_UNCERTAIN":
            result["status"] = "HOLD_UNCERTAIN_OPERATOR_INTERVENTION_REQUIRED"
            trial.write_json(trial_dir / "event_replan_summary.json", result)
            return result
        forecast = monitor1["forecast"]
        q_goal = np.asarray(context["reference"].q[-1], dtype=np.float64)

        q_terminal_start = q_actual

        if monitor1["status"] == "REPLAN_REQUIRED":
            local2_dir = trial_dir / "event_local_02"
            local2_dir.mkdir(parents=True, exist_ok=True)
            artifacts: dict[str, Any] = {}
            local2_reference_goal, local2_reference_audit = next_recorded_reference_goal(
                context["reference"], q_actual, args.local_horizon_s
            )
            result["local2_reference_goal"] = local2_reference_audit
            if live.ACTIVE_BASE_FAST_REPAIR is None:
                raise RuntimeError("validated base Fast implementation is unavailable")
            candidate = plan_goal_directed_continuation(
                live.ACTIVE_BASE_FAST_REPAIR,
                args,
                config,
                model,
                q_escape_start=np.asarray(context["local_artifacts"]["q_now"], dtype=np.float64),
                q_now=q_actual,
                q_final=q_goal,
                fresh=monitor1["fresh"],
                geometry=monitor1["geometry"],
                risk_links=set(context["risk_links"]),
                trial_dir=local2_dir,
                nominal_reference_goal=local2_reference_goal,
                artifacts_out=artifacts,
                forward_m=float(event_args.forward_m),
                side_m=float(event_args.continuation_side_m),
                robust_target_m=float(event_args.planning_robust_target_m),
                max_joint_delta_rad=float(event_args.max_joint_delta_rad),
                tcp_link=event_args.tcp_link,
            )
            result["local2_candidate"] = candidate
            if not candidate.get("local_repair_ready", False):
                result["status"] = "HOLD_UNSAFE_APPROACHING_OPERATOR_INTERVENTION_REQUIRED"
                trial.write_json(trial_dir / "event_replan_summary.json", result)
                return result

            fresh4, frames4, points4, geometry4 = capture_next_fresh(
                args, processor, state_reader, denoiser, monitor1["fresh"]
            )
            trial.write_json(local2_dir / "fresh4_recheck.json", {"result": fresh4, "frames": frames4})
            if points4 is not None:
                np.save(local2_dir / "fresh4_cluster_points.npy", points4)
            if geometry4 is not None:
                trial.write_json(local2_dir / "fresh4_multisphere.json", geometry4)
            if geometry4 is None:
                result["status"] = "LOCAL2_FRESH_AUTHORIZATION_NOT_READY_OPERATOR_INTERVENTION_REQUIRED"
                trial.write_json(trial_dir / "event_replan_summary.json", result)
                return result
            authorization, _ = trial.authorize_local_repair_execution(
                args,
                config,
                model,
                local_repair_ready=True,
                local_artifacts=artifacts,
                fresh_geometry=geometry4,
                fresh_velocity=np.asarray(fresh4["velocity"], dtype=np.float64),
                trial_dir=local2_dir,
                execution_duration_s=1.0,
            )
            result["local2_authorization"] = authorization
            if not authorization.get("local_execution_authorized", False):
                result["status"] = "LOCAL2_FRESH_REJECTED_OPERATOR_INTERVENTION_REQUIRED"
                trial.write_json(trial_dir / "event_replan_summary.json", result)
                return result
            execution = trial.execute_authorized_trajectory_offline_track(
                robot,
                Path(authorization["authorized_trajectory_csv"]),
                args,
                processor=processor,
                denoiser=denoiser,
                playback_duration_s=None,
                execution_label="Fresh #4-authorized event local repair #2",
            )
            result["local2_execution"] = execution
            if execution.get("status") != "COMPLETED_AUTHORIZED_TRAJECTORY_EXECUTION":
                result["status"] = "LOCAL2_EXECUTION_FAILED_OPERATOR_INTERVENTION_REQUIRED"
                trial.write_json(trial_dir / "event_replan_summary.json", result)
                return result
            q_terminal_start = np.asarray(robot.get_joint(), dtype=np.float64)
            fresh5, frames5, points5, geometry5 = capture_next_fresh(
                args, processor, state_reader, denoiser, fresh4
            )
            trial.write_json(local2_dir / "fresh5_recheck.json", {"result": fresh5, "frames": frames5})
            if points5 is not None:
                np.save(local2_dir / "fresh5_cluster_points.npy", points5)
            if geometry5 is not None:
                trial.write_json(local2_dir / "fresh5_multisphere.json", geometry5)
            monitor2 = monitor_measured_tail(
                args,
                config,
                model,
                processor,
                state_reader,
                denoiser,
                q_terminal_start,
                initial_fresh=fresh5,
                initial_frames=frames5,
                initial_geometry=geometry5,
                output_dir=trial_dir / "post_local2_monitor",
                max_wall_s=float(event_args.post_local_monitor_max_s),
            )
            result["post_local2_monitor"] = {
                key: value for key, value in monitor2.items() if key not in {"forecast", "geometry"}
            }
            if monitor2["status"] == "REPLAN_REQUIRED":
                result["status"] = "HOLD_UNSAFE_APPROACHING_OPERATOR_INTERVENTION_REQUIRED"
                trial.write_json(trial_dir / "event_replan_summary.json", result)
                return result
            if monitor2["status"] == "PHYSICAL_HOLD_SAFE_MONITORING_TIMEOUT":
                result["status"] = "PHYSICAL_HOLD_SAFE_MONITORING_LIMIT_OPERATOR_CONTROL_REQUIRED"
                trial.write_json(trial_dir / "event_replan_summary.json", result)
                return result
            if monitor2["status"] != "STRICT_SCENE_CLEAR":
                result["status"] = "POST_LOCAL2_STATE_UNCERTAIN_OPERATOR_INTERVENTION_REQUIRED"
                trial.write_json(trial_dir / "event_replan_summary.json", result)
                return result
            forecast = monitor2["forecast"]

        if monitor1["status"] != "STRICT_SCENE_CLEAR" and "post_local2_monitor" not in result:
            raise RuntimeError(f"unexpected post-local decision: {monitor1['status']}")
        terminal_dir = trial_dir / "terminal_goal_authorization"
        terminal, _ = authorize_terminal_goal(
            args,
            config,
            model,
            q_terminal_start,
            q_goal,
            forecast,
            terminal_durations,
            terminal_dir,
        )
        result["terminal_authorization"] = terminal
        if not terminal.get("authorized", False):
            result["status"] = "STRICT_SCENE_CLEAR_TERMINAL_GOAL_PATH_BLOCKED"
            trial.write_json(trial_dir / "event_replan_summary.json", result)
            return result

        # One last raw observation is intentionally not replaced by an old
        # candidate forecast: the guarded executor samples raw distance three
        # times immediately before commanding this current-run terminal CSV.
        terminal_execution = trial.execute_authorized_trajectory_offline_track(
            robot,
            Path(terminal["authorized_trajectory_csv"]),
            args,
            processor=processor,
            denoiser=denoiser,
            playback_duration_s=None,
            execution_label="event-replan terminal NUBS to preset goal",
        )
        result["terminal_execution"] = terminal_execution
        if terminal_execution.get("status") == "COMPLETED_AUTHORIZED_TRAJECTORY_EXECUTION":
            result["status"] = "SIMPLE_DYNAMIC_NUBS_RECOVERED_AND_GOAL_REACHED"
            result["command_hold"] = False
        else:
            result["status"] = "TERMINAL_EXECUTION_FAILED_OPERATOR_INTERVENTION_REQUIRED"
        trial.write_json(trial_dir / "event_replan_summary.json", result)
        return result

    return handler


def run(args: argparse.Namespace) -> dict[str, Any]:
    durations = tuple(float(value) for value in args.terminal_durations_s.split(","))
    if not durations or any(value <= 0.0 for value in durations):
        raise ValueError("terminal-durations-s must contain positive values")
    if args.execute and args.event_operator_phrase != EVENT_EXECUTE_PHRASE:
        raise RuntimeError(f"bad event execute phrase; required: {EVENT_EXECUTE_PHRASE}")
    if args.execute and trial.git_is_dirty():
        raise RuntimeError(
            "event-replan live execution requires a clean committed worktree; "
            "commit the reviewed code and provenance manifest before commanding the robot"
        )

    live_args = copy.copy(args)
    live_args.operator_phrase = live.LOCAL_EXECUTE_PHRASE if args.execute else ""
    old_handler = trial.POST_LOCAL_FRESH3_HANDLER
    try:
        trial.POST_LOCAL_FRESH3_HANDLER = make_event_handler(args, durations)
        result = live.run(live_args)
    finally:
        trial.POST_LOCAL_FRESH3_HANDLER = old_handler

    core_path = Path(result.get("core_summary", ""))
    if core_path.exists():
        with core_path.open("r", encoding="utf-8") as handle:
            core = json.load(handle)
        continuation = next(
            (
                event.get("continuation")
                for event in reversed(core.get("events", []))
                if event.get("continuation") is not None
            ),
            None,
        )
        if continuation is not None:
            result["event_continuation"] = continuation
            result["status"] = continuation.get("status", result["status"])
            trial.write_json(Path(result["output"]) / "summary.json", result)
    return result


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
