#!/usr/bin/env python3
"""Final V3 real closed loop: dynamic local NUBS segments to preset goal.

This entry deliberately does not run the parked-robot virtual shadow.  It
reuses the frozen V3 perception/planning policy and the protected real
executor, then owns a bounded loop of one-second LOCAL_BYPASS and
GOAL_DIRECTED NUBS segments.  Every command is checked against a newer
persistent obstacle state; execution retains the 0.10 m raw-cloud hard guard
and a continuous perception/remaining-trajectory monitor.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

trial = importlib.import_module(
    "experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial"
)
live = importlib.import_module(
    "experiments.new.6_5.6_5_3.run_653_simple_dynamic_nubs_live"
)
event = importlib.import_module(
    "experiments.new.6_5.6_5_3.run_653_simple_dynamic_nubs_event_replan_live"
)
v3 = importlib.import_module("experiments.new.6_5.6_5_3.dynamic_nubs_v3")


DEFAULT_OUTPUT = ROOT / "results/new/6_5/6_5_3/dynamic_nubs_v3_final_live"
DEFAULT_R04 = (
    ROOT
    / "results/new/6_5/6_5_3/dynamic_nubs_v3_playback_shadow/r04"
    / "core_live/trials/D2_opposing_approach_r04/v3_playback_shadow"
    / "playback_shadow_summary.json"
)
SCENE_PHRASE = "CCRO_653_DYNAMIC_NUBS_V3_OPPOSING_SCENE_CONFIRMED"
FINAL_EXECUTE_PHRASE = "CCRO_653_DYNAMIC_NUBS_V3_FINAL_LIVE_EXECUTE_APPROVED"

FINAL_PROTOCOL = {
    **v3.V3_PROTOCOL,
    "protocol_id": "653_DYNAMIC_NUBS_V3_FINAL_LIVE",
    "candidate_playback_mode": "real_event_driven_closed_loop_to_goal",
    "real_candidate_execution_enabled": True,
    "parked_robot_shadow_required": False,
    "goal_motion_replan_threshold_m": 0.14,
    "local_execution_clearance_m": 0.09,
    "actual_execution_raw_hard_guard_m": 0.10,
    "strict_empty_scene_required": False,
    "reference_rejoin_required": False,
}


def decide_next_motion(
    *,
    execution_status: str,
    segment_kind: str,
    monitor_stop_reason: str | None,
    goal_error_max_abs_rad: float,
    goal_tolerance_rad: float,
    risk_remains: bool,
    goal_step_safe: bool,
    local_replans: int,
    max_local_replans: int,
) -> str:
    """Return the only allowed next state for the bounded final loop."""
    if (
        execution_status == "COMPLETED_AUTHORIZED_TRAJECTORY_EXECUTION"
        and goal_error_max_abs_rad <= goal_tolerance_rad
    ):
        return "GOAL_REACHED"
    if execution_status == "STOPPED_BY_MOTION_MONITOR":
        if segment_kind != "goal" or monitor_stop_reason != "predicted_goal_risk_replan":
            return "FAIL_CLOSED_HOLD"
        risk_remains = True
    elif execution_status != "COMPLETED_AUTHORIZED_TRAJECTORY_EXECUTION":
        return "FAIL_CLOSED_HOLD"
    if not risk_remains and goal_step_safe:
        return "GOAL_DIRECTED_NUBS"
    if local_replans >= max_local_replans:
        return "MAX_LOCAL_REPLANS_HOLD"
    return "NEXT_LOCAL_NUBS"


def build_parser() -> argparse.ArgumentParser:
    parser = live.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        output=DEFAULT_OUTPUT,
        task_geometry_id="D2_DYNAMIC_NUBS_V3_FINAL_LIVE_XP10",
        planning_robust_target_m=0.11,
    )
    parser.add_argument("--scene-operator-phrase", default="")
    parser.add_argument("--final-live-operator-phrase", default="")
    parser.add_argument("--max-local-replans", type=int, default=3)
    parser.add_argument("--max-closed-loop-segments", type=int, default=12)
    parser.add_argument("--closed-loop-goal-tolerance-rad", type=float, default=0.01)
    parser.add_argument("--continuation-side-m", type=float, default=0.04)
    parser.add_argument("--software-dry-run", action="store_true")
    parser.add_argument("--r04-shadow-summary", type=Path, default=DEFAULT_R04)
    return parser


def software_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    """Exercise only orchestration semantics using the archived r04 stages."""
    source = args.r04_shadow_summary.resolve()
    archived = json.loads(source.read_text(encoding="utf-8"))
    segments = archived.get("segments", [])
    if len(segments) < 3:
        raise RuntimeError("r04 does not contain the expected three local stages")
    checks = []
    for index, segment in enumerate(segments[:3], 1):
        authorization = segment.get("precommand_authorization", {})
        predicted = segment.get("playback_min_predicted_remaining_clearance_m")
        checks.append(
            {
                "segment": index,
                "latest_state_authorized": bool(
                    authorization.get("local_execution_authorized", False)
                ),
                "predicted_clearance_m": predicted,
                "predicted_clearance_pass": bool(
                    predicted is not None and float(predicted) >= 0.09
                ),
                "archived_status": segment.get("status"),
            }
        )
    archived_checks_pass = all(
        row["latest_state_authorized"] and row["predicted_clearance_pass"]
        for row in checks
    )
    trace: list[str] = []
    transition_audit = []
    local_replans = 1
    for index, segment in enumerate(segments[:3], 1):
        trace.append(f"LOCAL_NUBS_{index}")
        archived_tail = segment.get("tail_hold_predicted_clearance_m")
        # r04 segment 3 was physically aborted because the real robot was
        # parked.  No later obstacle frame exists, so the orchestration test
        # explicitly supplies a mock *newer* state in which the obstacle has
        # crossed and the one-second goal step is safe.  This is not presented
        # as a new geometry authorization.
        mock_latest_state = archived_tail is None
        hold_clearance = 0.15 if mock_latest_state else float(archived_tail)
        goal_step_safe = bool(mock_latest_state)
        action = decide_next_motion(
            execution_status="COMPLETED_AUTHORIZED_TRAJECTORY_EXECUTION",
            segment_kind="local",
            monitor_stop_reason=None,
            goal_error_max_abs_rad=1.0,
            goal_tolerance_rad=0.01,
            risk_remains=hold_clearance < 0.14,
            goal_step_safe=goal_step_safe,
            local_replans=local_replans,
            max_local_replans=3,
        )
        transition_audit.append(
            {
                "after_segment": index,
                "hold_clearance_m": hold_clearance,
                "goal_step_safe": goal_step_safe,
                "state_source": (
                    "mock_newer_post_crossing_state"
                    if mock_latest_state
                    else "archived_r04_tail_state"
                ),
                "action": action,
            }
        )
        if action == "NEXT_LOCAL_NUBS":
            local_replans += 1
            continue
        if action == "GOAL_DIRECTED_NUBS":
            trace.append("GOAL_DIRECTED_NUBS")
            final_action = decide_next_motion(
                execution_status="COMPLETED_AUTHORIZED_TRAJECTORY_EXECUTION",
                segment_kind="goal",
                monitor_stop_reason=None,
                goal_error_max_abs_rad=0.0,
                goal_tolerance_rad=0.01,
                risk_remains=False,
                goal_step_safe=True,
                local_replans=local_replans,
                max_local_replans=3,
            )
            transition_audit.append(
                {
                    "after_segment": "mock_goal_segment",
                    "state_source": "mock_goal_feedback",
                    "action": final_action,
                }
            )
            trace.append(final_action)
            break
        trace.append(action)
        break
    passed = bool(
        archived_checks_pass
        and trace
        == [
            "LOCAL_NUBS_1",
            "LOCAL_NUBS_2",
            "LOCAL_NUBS_3",
            "GOAL_DIRECTED_NUBS",
            "GOAL_REACHED",
        ]
    )
    result = {
        "status": (
            "FINAL_LIVE_SOFTWARE_DRY_RUN_PASS"
            if passed
            else "FINAL_LIVE_SOFTWARE_DRY_RUN_FAILED"
        ),
        "robot_commanded": False,
        "source": str(source),
        "scope": "orchestration_only_not_a_new_geometry_authorization",
        "parked_robot_guard_is_not_a_virtual_candidate_gate": True,
        "trace": trace,
        "transition_audit": transition_audit,
        "archived_segment_checks": checks,
    }
    output = args.output.resolve() / "software_dry_run"
    output.mkdir(parents=True, exist_ok=True)
    trial.write_json(output / "summary.json", result)
    return result


def _valid_snapshot(
    worker: Any,
    args: Any,
    *,
    newer_than_seq: int | None = None,
    timeout_s: float = 0.35,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    snapshot = (
        worker.wait_for_newer_state(after_seq=newer_than_seq, timeout_s=timeout_s)
        if newer_than_seq is not None
        else worker.snapshot()
    )
    aligned = v3.time_aligned_snapshot(snapshot, execution_timestamp=time.time())
    reasons = v3._persistent_state_reasons(
        snapshot,
        aligned,
        args,
        require_newer_than_seq=newer_than_seq,
    )
    return snapshot, aligned, reasons


def _command_time_authorize(
    *,
    worker: Any,
    args: Any,
    config: dict[str, Any],
    model: Any,
    artifacts: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    baseline = worker.snapshot()
    snapshot, aligned, reasons = _valid_snapshot(
        worker,
        args,
        newer_than_seq=v3._state_seq(baseline),
        timeout_s=max(0.35, float(args.candidate_pre_execute_settle_s)),
    )
    if reasons:
        return (
            {
                "status": "FINAL_COMMAND_STATE_HOLD",
                "local_execution_authorized": False,
                "reason": reasons,
            },
            snapshot,
        )
    authorization, _ = trial.authorize_local_repair_execution(
        args,
        config,
        model,
        local_repair_ready=True,
        local_artifacts=artifacts,
        fresh_geometry=aligned["geometry"],
        fresh_velocity=np.asarray(snapshot["velocity"], dtype=np.float64),
        trial_dir=output_dir,
    )
    authorization["persistent_state_seq"] = v3._state_seq(snapshot)
    authorization["persistent_state_age_s"] = float(aligned["propagation_dt_s"])
    authorization["raw_guard_distance_m"] = float(
        snapshot["raw_guard_distance_m"]
    )
    return authorization, snapshot


def _execution_monitor(
    *,
    worker: Any,
    args: Any,
    config: dict[str, Any],
    model: Any,
    trajectory: Any,
    segment_kind: str,
):
    evaluator, _, _ = trial.make_risk_stack(config, model, None)
    threshold = (
        float(args.online_accept_m)
        if segment_kind == "local"
        else float(args.replan_in_m)
    )

    def monitor(
        *, elapsed_s: float, actual_q: np.ndarray, obstacle_snapshot: Any
    ) -> dict[str, Any]:
        del actual_q
        snapshot = obstacle_snapshot or worker.snapshot()
        aligned = v3.time_aligned_snapshot(snapshot, execution_timestamp=time.time())
        reasons = v3._persistent_state_reasons(snapshot, aligned, args)
        if reasons:
            return {
                "motion_safe": False,
                "reason": reasons[0],
                "state_failure_reasons": reasons,
                "segment_kind": segment_kind,
            }
        forecast = v3.v3_execution_multisphere_forecast(
            np.asarray(aligned["geometry"]["component_centers"], dtype=np.float64),
            np.asarray(
                aligned["geometry"]["component_base_radii"], dtype=np.float64
            ),
            np.asarray(snapshot["velocity"], dtype=np.float64),
        )
        clearance = v3._remaining_clearance(
            evaluator,
            trajectory,
            forecast,
            playback_time_s=min(float(elapsed_s), float(trajectory.total_duration)),
        )
        safe = bool(clearance["min_distance_m"] >= threshold)
        return {
            "motion_safe": safe,
            "reason": (
                "clear"
                if safe
                else "predicted_goal_risk_replan"
                if segment_kind == "goal"
                else "local_remaining_clearance_below_0p09"
            ),
            "segment_kind": segment_kind,
            "threshold_m": threshold,
            "remaining_clearance": clearance,
            "state_seq": v3._state_seq(snapshot),
            "state_age_s": float(aligned["propagation_dt_s"]),
        }

    return monitor


def _latest_decision_state(
    worker: Any, args: Any, after_seq: int
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    return _valid_snapshot(
        worker,
        args,
        newer_than_seq=after_seq,
        timeout_s=0.35,
    )


def _plan_next_local(
    *,
    worker: Any,
    args: Any,
    config: dict[str, Any],
    model: Any,
    q_escape_start: np.ndarray,
    q_now: np.ndarray,
    q_goal: np.ndarray,
    snapshot: dict[str, Any],
    aligned: dict[str, Any],
    risk_links: set[str],
    output_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if live.ACTIVE_BASE_FAST_REPAIR is None:
        return None, {"status": "BASE_FAST_UNAVAILABLE"}
    goal_artifacts, goal_step = v3._bounded_goal_artifacts(
        q_now,
        q_goal,
        max_joint_delta_rad=float(args.max_joint_delta_rad),
        duration_s=float(args.local_horizon_s),
    )
    nominal = goal_artifacts["candidate_trajectory"]
    reference_goal = (
        np.asarray(nominal.evaluate(nominal.total_duration), dtype=np.float64),
        np.zeros(6),
        np.zeros(6),
    )
    artifacts: dict[str, Any] = {}
    candidate = event.plan_goal_directed_continuation(
        live.ACTIVE_BASE_FAST_REPAIR,
        args,
        config,
        model,
        q_escape_start=q_escape_start,
        q_now=q_now,
        q_final=q_goal,
        fresh=v3._snapshot_fresh(snapshot, aligned),
        geometry=aligned["geometry"],
        risk_links=risk_links,
        trial_dir=output_dir,
        nominal_reference_goal=reference_goal,
        artifacts_out=artifacts,
        forward_m=float(args.forward_m),
        side_m=float(args.continuation_side_m),
        robust_target_m=float(args.planning_robust_target_m),
        max_joint_delta_rad=float(args.max_joint_delta_rad),
        tcp_link=args.tcp_link,
        robust_target_is_diagnostic=True,
    )
    audit = {"candidate": candidate, "nominal_goal_step": goal_step}
    if not candidate.get("local_repair_ready", False):
        return None, {**audit, "status": "LOCAL_REPLAN_NOT_READY"}
    authorization = v3.latest_state_authorize_with_one_replan(
        worker=worker,
        args=args,
        stage4_config=config,
        stage4_model=model,
        q_now=q_now,
        qd_now=np.zeros(6),
        reference_goal=reference_goal,
        rejoin_goals=[],
        risk_links=risk_links,
        trial_dir=output_dir,
        candidate_summary=candidate,
        local_artifacts=artifacts,
        planning_state=snapshot,
    )
    if not authorization["authorized"]:
        return None, {
            **audit,
            "status": "LOCAL_REPLAN_LATEST_STATE_REJECTED",
            "authorization_status": authorization["status"],
            "attempts": authorization["attempts"],
        }
    next_artifacts = authorization["local_artifacts"]
    next_artifacts["v3_local_bypass"] = True
    return next_artifacts, {
        **audit,
        "status": "NEXT_LOCAL_NUBS_READY",
        "authorization_status": authorization["status"],
    }


def run_final_closed_loop(
    *,
    worker: Any,
    args: Any,
    stage4_config: dict[str, Any],
    stage4_model: Any,
    robot: Any,
    processor: Any,
    state_reader: Any,
    denoiser: Any,
    local_artifacts: dict[str, Any],
    trial_dir: Path,
    task_goal_q: np.ndarray,
    risk_links: set[str] | None = None,
) -> dict[str, Any]:
    del state_reader
    root = Path(trial_dir) / "v3_final_live"
    root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "status": "DYNAMIC_NUBS_CLOSED_LOOP_OPERATOR_INTERVENTION_REQUIRED",
        "robot_commanded": False,
        "events": ["DYNAMIC_NUBS_FINAL_CLOSED_LOOP_STARTED"],
        "segments": [],
        "decisions": [],
    }
    if not args.allow_live_candidate_execution:
        result["status"] = "FINAL_LIVE_EXECUTION_NOT_ENABLED"
        return result
    # The exact final-live phrase authorizes the bounded closed loop once.
    # Per-segment interactive prompts would leave a moving obstacle advancing
    # while the robot waits, so every later command relies on the automatic
    # latest-state verifier and execution guards instead.
    args.candidate_execute_confirm = False

    q_goal = np.asarray(task_goal_q, dtype=np.float64)
    q_escape_start = np.asarray(local_artifacts["q_now"], dtype=np.float64)
    artifacts = local_artifacts
    segment_kind = "local"
    local_replans = 1
    latest_seq = v3._state_seq(worker.snapshot())

    for segment_index in range(1, int(args.max_closed_loop_segments) + 1):
        segment_dir = root / f"segment_{segment_index:02d}_{segment_kind}"
        authorization, command_snapshot = _command_time_authorize(
            worker=worker,
            args=args,
            config=stage4_config,
            model=stage4_model,
            artifacts=artifacts,
            output_dir=segment_dir / "command_authorization",
        )
        if not authorization.get("local_execution_authorized", False):
            result["status"] = "FINAL_COMMAND_AUTHORIZATION_HOLD"
            result["decisions"].append(
                {"segment": segment_index, "authorization": authorization}
            )
            break
        latest_seq = int(authorization["persistent_state_seq"])
        trajectory = artifacts["candidate_trajectory"]
        monitor = _execution_monitor(
            worker=worker,
            args=args,
            config=stage4_config,
            model=stage4_model,
            trajectory=trajectory,
            segment_kind=segment_kind,
        )
        try:
            execution = trial.execute_authorized_trajectory_offline_track(
                robot,
                Path(authorization["authorized_trajectory_csv"]),
                args,
                processor=processor,
                denoiser=denoiser,
                playback_duration_s=None,
                execution_label=(
                    f"V3 final-live {segment_kind} segment {segment_index}"
                ),
                guard_provider=worker.guard_snapshot,
                obstacle_state_provider=worker.snapshot,
                motion_monitor_provider=monitor,
            )
        except Exception as exc:
            trial.maybe_move_stop(robot)
            result["robot_commanded"] = True
            result["status"] = "FINAL_EXECUTOR_EXCEPTION_FAIL_CLOSED_HOLD"
            result["decisions"].append(
                {
                    "segment": segment_index,
                    "decision": "HOLD",
                    "exception": f"{type(exc).__name__}: {exc}",
                }
            )
            trial.write_json(root / "final_closed_loop_summary.json", result)
            return result
        segment_row = {
            "segment": segment_index,
            "kind": segment_kind,
            "authorization": authorization,
            "execution": execution,
        }
        result["segments"].append(segment_row)
        result["robot_commanded"] = True
        trial.write_json(segment_dir / "segment_summary.json", segment_row)

        q_actual = np.asarray(robot.get_joint(), dtype=np.float64)
        goal_error = float(np.max(np.abs(q_goal - q_actual)))
        monitor_reason = (
            execution.get("goal_check", {}).get("monitor_stop_reason")
            if execution.get("status") == "STOPPED_BY_MOTION_MONITOR"
            else None
        )
        immediate_action = decide_next_motion(
            execution_status=str(execution.get("status")),
            segment_kind=segment_kind,
            monitor_stop_reason=monitor_reason,
            goal_error_max_abs_rad=goal_error,
            goal_tolerance_rad=float(args.closed_loop_goal_tolerance_rad),
            risk_remains=True,
            goal_step_safe=False,
            local_replans=local_replans,
            max_local_replans=int(args.max_local_replans),
        )
        if immediate_action == "GOAL_REACHED":
            result["status"] = "DYNAMIC_NUBS_CLOSED_LOOP_GOAL_REACHED"
            result["events"].append("DYNAMIC_NUBS_CLOSED_LOOP_GOAL_REACHED")
            result["final_goal_error_max_abs_rad"] = goal_error
            break

        if immediate_action == "FAIL_CLOSED_HOLD":
            result["status"] = "FINAL_EXECUTION_MONITOR_FAIL_CLOSED_HOLD"
            result["decisions"].append(
                {
                    "segment": segment_index,
                    "decision": "HOLD",
                    "reason": monitor_reason or execution.get("status"),
                }
            )
            break
        if execution.get("status") == "STOPPED_BY_MOTION_MONITOR":
            result["events"].append("GOAL_MOTION_STRO_STOP")

        snapshot, aligned, reasons = _latest_decision_state(
            worker, args, after_seq=max(latest_seq, v3._state_seq(command_snapshot))
        )
        if reasons:
            result["status"] = "FINAL_LATEST_STATE_INVALID_HOLD"
            result["decisions"].append(
                {"segment": segment_index, "decision": "HOLD", "reason": reasons}
            )
            break
        latest_seq = v3._state_seq(snapshot)
        forecast = v3.v3_execution_multisphere_forecast(
            np.asarray(aligned["geometry"]["component_centers"], dtype=np.float64),
            np.asarray(
                aligned["geometry"]["component_base_radii"], dtype=np.float64
            ),
            np.asarray(snapshot["velocity"], dtype=np.float64),
        )
        evaluator, verifier, _ = trial.make_risk_stack(
            stage4_config, stage4_model, None
        )
        hold = v3._fixed_configuration_clearance(
            evaluator,
            q_actual,
            forecast,
            horizon_s=float(args.prediction_horizon_s),
        )
        goal_artifacts, goal_step = v3._bounded_goal_artifacts(
            q_actual,
            q_goal,
            max_joint_delta_rad=float(args.max_joint_delta_rad),
            duration_s=float(args.local_horizon_s),
        )
        goal_trajectory = goal_artifacts["candidate_trajectory"]
        goal_verification = verifier.verify(
            goal_trajectory,
            forecast,
            current_q=q_actual,
            current_qd=np.zeros(6),
            current_qdd=np.zeros(6),
            q_goal=np.asarray(
                goal_trajectory.evaluate(goal_trajectory.total_duration),
                dtype=np.float64,
            ),
            solver_success=True,
        )
        risk_remains = bool(
            execution.get("status") == "STOPPED_BY_MOTION_MONITOR"
            or hold["min_distance_m"] < float(args.replan_in_m)
        )
        decision = {
            "after_segment": segment_index,
            "goal_error_max_abs_rad": goal_error,
            "stationary_predicted_clearance_m": hold["min_distance_m"],
            "risk_remains": risk_remains,
            "goal_step": goal_step,
            "goal_step_safe": bool(goal_verification.accepted),
            "goal_step_min_distance_m": float(goal_verification.min_distance),
        }
        next_action = decide_next_motion(
            execution_status=str(execution.get("status")),
            segment_kind=segment_kind,
            monitor_stop_reason=monitor_reason,
            goal_error_max_abs_rad=goal_error,
            goal_tolerance_rad=float(args.closed_loop_goal_tolerance_rad),
            risk_remains=risk_remains,
            goal_step_safe=bool(goal_verification.accepted),
            local_replans=local_replans,
            max_local_replans=int(args.max_local_replans),
        )
        if next_action == "GOAL_DIRECTED_NUBS":
            artifacts = goal_artifacts
            segment_kind = "goal"
            decision["decision"] = "GOAL_DIRECTED_NUBS"
            result["events"].append("LATEST_STATE_GOAL_DIRECTED_CONTINUATION")
            result["decisions"].append(decision)
            continue

        if next_action == "MAX_LOCAL_REPLANS_HOLD":
            decision["decision"] = "MAX_LOCAL_REPLANS_HOLD"
            result["decisions"].append(decision)
            result["status"] = "FINAL_MAX_LOCAL_REPLANS_FAIL_CLOSED_HOLD"
            break
        try:
            next_artifacts, planning = _plan_next_local(
                worker=worker,
                args=args,
                config=stage4_config,
                model=stage4_model,
                q_escape_start=q_escape_start,
                q_now=q_actual,
                q_goal=q_goal,
                snapshot=snapshot,
                aligned=aligned,
                risk_links=set(risk_links or ()),
                output_dir=root / f"local_replan_{local_replans + 1:02d}",
            )
        except Exception as exc:
            trial.maybe_move_stop(robot)
            decision["decision"] = "LOCAL_REPLAN_EXCEPTION_HOLD"
            decision["exception"] = f"{type(exc).__name__}: {exc}"
            result["decisions"].append(decision)
            result["status"] = "FINAL_LOCAL_REPLAN_EXCEPTION_FAIL_CLOSED_HOLD"
            break
        decision["local_planning"] = planning
        if next_artifacts is None:
            decision["decision"] = "LOCAL_REPLAN_HOLD"
            result["decisions"].append(decision)
            result["status"] = "FINAL_LOCAL_REPLAN_FAIL_CLOSED_HOLD"
            break
        artifacts = next_artifacts
        segment_kind = "local"
        local_replans += 1
        decision["decision"] = "NEXT_LOCAL_NUBS"
        decision["local_replan_index"] = local_replans
        result["decisions"].append(decision)
        result["events"].append("LATEST_STATE_NEXT_LOCAL_REPLAN")
    else:
        result["status"] = "FINAL_CLOSED_LOOP_SEGMENT_LIMIT_HOLD"

    result["local_replans_used"] = local_replans
    result["segments_attempted"] = len(result["segments"])
    trial.write_json(root / "final_closed_loop_summary.json", result)
    return result


def validate(args: argparse.Namespace) -> None:
    if args.software_dry_run:
        if args.execute:
            raise RuntimeError("--software-dry-run and --execute are mutually exclusive")
        return
    if not args.execute:
        raise RuntimeError(
            "final-live has no parked shadow mode; use --software-dry-run or "
            "provide --execute with the exact final authorization phrase"
        )
    if args.reference_operator_phrase != live.REFERENCE_OPERATOR_PHRASE:
        raise RuntimeError(
            "bad reference phrase; required: " + live.REFERENCE_OPERATOR_PHRASE
        )
    if args.scene_operator_phrase != SCENE_PHRASE:
        raise RuntimeError(f"bad scene phrase; required: {SCENE_PHRASE}")
    if args.final_live_operator_phrase != FINAL_EXECUTE_PHRASE:
        raise RuntimeError(
            f"bad final-live phrase; required: {FINAL_EXECUTE_PHRASE}"
        )
    if trial.git_is_dirty():
        raise RuntimeError(
            "final-live execution requires a clean committed worktree"
        )
    if args.max_local_replans != 3:
        raise RuntimeError("final-live maximum local replans is frozen at 3")
    if args.max_closed_loop_segments < 3:
        raise ValueError("max-closed-loop-segments must be at least 3")


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate(args)
    if args.software_dry_run:
        return software_dry_run(args)

    final_args = copy.copy(args)
    final_args.operator_phrase = live.LOCAL_EXECUTE_PHRASE
    original_predictor = trial.RISK_SPHERE_PREDICTOR
    original_trigger_gate = trial.RISK_TRIGGER_REQUIRES_DYNAMIC_TRACK
    original_forecast = trial.constant_multisphere_forecast
    original_worker = trial.PERSISTENT_OBSTACLE_WORKER_FACTORY
    original_latest = trial.LATEST_STATE_AUTHORIZATION_POLICY
    original_shadow = trial.POST_AUTHORIZATION_PLAYBACK_SHADOW
    original_closed_loop = trial.POST_AUTHORIZATION_CLOSED_LOOP_HANDLER
    original_adapter = live.fixed_two_sphere_adapter
    original_factory = live.make_r06_fast_wrapper
    try:
        trial.RISK_SPHERE_PREDICTOR = v3.adaptive_multisphere_predictor
        trial.RISK_TRIGGER_REQUIRES_DYNAMIC_TRACK = False
        trial.constant_multisphere_forecast = v3.v3_execution_multisphere_forecast
        trial.PERSISTENT_OBSTACLE_WORKER_FACTORY = v3.make_persistent_perception_worker
        trial.LATEST_STATE_AUTHORIZATION_POLICY = (
            v3.latest_state_authorize_with_one_replan
        )
        trial.POST_AUTHORIZATION_PLAYBACK_SHADOW = None
        trial.POST_AUTHORIZATION_CLOSED_LOOP_HANDLER = run_final_closed_loop
        live.fixed_two_sphere_adapter = v3.adaptive_geometry_adapter
        live.make_r06_fast_wrapper = v3.make_v3_fast_factory(original_factory)
        result = live.run(final_args)
    finally:
        trial.RISK_SPHERE_PREDICTOR = original_predictor
        trial.RISK_TRIGGER_REQUIRES_DYNAMIC_TRACK = original_trigger_gate
        trial.constant_multisphere_forecast = original_forecast
        trial.PERSISTENT_OBSTACLE_WORKER_FACTORY = original_worker
        trial.LATEST_STATE_AUTHORIZATION_POLICY = original_latest
        trial.POST_AUTHORIZATION_PLAYBACK_SHADOW = original_shadow
        trial.POST_AUTHORIZATION_CLOSED_LOOP_HANDLER = original_closed_loop
        live.fixed_two_sphere_adapter = original_adapter
        live.make_r06_fast_wrapper = original_factory

    core_path = Path(result.get("core_summary", ""))
    if core_path.exists():
        core = json.loads(core_path.read_text(encoding="utf-8"))
        final_event = next(
            (
                row
                for row in reversed(core.get("events", []))
                if str(row.get("type", "")).startswith("DYNAMIC_NUBS_CLOSED_LOOP")
                or str(row.get("type", "")).startswith("FINAL_")
            ),
            None,
        )
        if final_event is not None:
            result["status"] = final_event["type"]
            result["final_closed_loop"] = final_event.get("closed_loop_execution")
    result["final_protocol"] = FINAL_PROTOCOL
    output = Path(result["output"])
    trial.write_json(output / "final_protocol.json", FINAL_PROTOCOL)
    trial.write_json(output / "summary.json", result)
    return result


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
