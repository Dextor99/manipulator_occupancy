#!/usr/bin/env python3
"""Protected two-event continuation for the validated simple dynamic NUBS pilot.

The first local segment is exactly the r04-frozen implementation.  At each
measured tail, the latest persistent state evaluates whether remaining
stationary is physically safe over the 0.5 s prediction horizon.  If not, a
new Fresh-authorized local segment is generated from the measured joints and
the loop continues until risk clears or a fail-safe watchdog intervenes.  A
direct terminal NUBS to the recorded preset goal is allowed only after a new
complete verification.
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
import threading
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

trial = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")
live = importlib.import_module("experiments.new.6_5.6_5_3.run_653_simple_dynamic_nubs_live")
bypass = importlib.import_module("experiments.new.6_5.6_5_3.simple_bypass_planner")
v3 = importlib.import_module("experiments.new.6_5.6_5_3.dynamic_nubs_v3")
stationary_ccro = importlib.import_module("experiments.new.6_5.6_5_3.stationary_terminal_ccro")

DEFAULT_OUTPUT = ROOT / "results/new/6_5/6_5_3/simple_dynamic_nubs_event_replan_live"
EVENT_EXECUTE_PHRASE = "CCRO_653_SIMPLE_DYNAMIC_EVENT_REPLAN_EXECUTE_APPROVED"
ROLLING_REPLAN_MONITOR_REASONS = {
    "remaining_predicted_risk",
}
COMMAND_TIME_REPLAN_STATUS = "COMMAND_TIME_REVALIDATION_REPLAN_REQUIRED"
COMMAND_TIME_HOLD_STATUS = "COMMAND_TIME_REVALIDATION_HOLD_PRECOMMAND"


def classify_terminal_authorization(terminal: dict[str, Any]) -> dict[str, Any]:
    """Classify terminal failure without weakening fail-closed behavior."""
    if bool(terminal.get("authorized", False)):
        return {
            "kind": "authorized",
            "distance_blocked": False,
            "attempt_count": len(terminal.get("attempts") or []),
        }
    attempts = list(terminal.get("attempts") or [])
    if not attempts:
        return {"kind": "other_failure", "distance_blocked": False, "attempt_count": 0}
    distance_blocked = all(
        not bool((attempt.get("checks") or {}).get("distance_ok", False))
        for attempt in attempts
    )
    info = {
        "kind": "distance_blocked" if distance_blocked else "other_failure",
        "distance_blocked": bool(distance_blocked),
        "attempt_count": len(attempts),
    }
    if distance_blocked:
        info["minimum_terminal_clearance_m"] = min(
            float(attempt.get("min_distance_m", math.inf)) for attempt in attempts
        )
    return info


def can_continue_local_after_terminal_block(
    monitor_result: dict[str, Any], terminal: dict[str, Any]
) -> bool:
    """A safe stopped tail may still have a blocked path to the goal."""
    if not classify_terminal_authorization(terminal).get("distance_blocked", False):
        return False
    if monitor_result.get("status") != "PREDICTED_RISK_CLEAR":
        return False
    fresh = monitor_result.get("fresh") or {}
    return bool(
        fresh.get("accepted", False)
        and monitor_result.get("geometry") is not None
        and monitor_result.get("forecast") is not None
    )


def wait_for_stationary_safe_recovery_state(
    worker: Any,
    args: argparse.Namespace,
    *,
    timeout_s: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Wait for a fresh safe state while the robot is already stationary."""
    if worker is None:
        return {"ready": False, "reason": "persistent_tracker_unavailable", "samples": []}, None
    started = time.monotonic()
    deadline = started + float(getattr(args, "prediction_horizon_s", 0.5) if timeout_s is None else timeout_s)
    latest = worker.snapshot()
    samples: list[dict[str, Any]] = []
    while True:
        aligned = v3.time_aligned_snapshot(latest, execution_timestamp=time.time())
        reasons = v3._persistent_state_reasons(latest, aligned, args, raw_guard_reason="raw_hard_guard_not_safe")
        raw_guard = float(latest.get("raw_guard_distance_m", float("-inf")))
        non_guard_reasons = [r for r in reasons if r != "raw_hard_guard_not_safe"]
        samples.append({
            "state_seq": int(v3._state_seq(latest)),
            "raw_guard_distance_m": raw_guard,
            "state_age_s": float(aligned["propagation_dt_s"]),
            "reasons": list(reasons),
        })
        if non_guard_reasons:
            return {"ready": False, "reason": "stationary_recovery_state_invalid", "failure_reasons": non_guard_reasons, "samples": samples, "elapsed_s": time.monotonic() - started}, None
        if not reasons and raw_guard > float(args.guided_hard_stop_m):
            return {"ready": True, "reason": "stationary_guard_recovered", "samples": samples, "elapsed_s": time.monotonic() - started}, latest
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return {"ready": False, "reason": "stationary_raw_guard_not_recovered", "samples": samples, "elapsed_s": time.monotonic() - started}, None
        current_seq = v3._state_seq(latest)
        newer = worker.wait_for_newer_state(after_seq=current_seq, timeout_s=min(0.25, remaining))
        latest = newer


def classify_local_authorization(authorization: dict[str, Any]) -> dict[str, Any]:
    """Only a distance-only Fresh rejection is recoverable by replanning."""
    if bool(authorization.get("local_execution_authorized", False)):
        return {"kind": "authorized", "distance_only": False, "failed_checks": []}
    checks = dict(authorization.get("verification_checks") or {})
    failed_checks = [key for key, passed in checks.items() if not bool(passed)]
    distance_only = set(failed_checks) == {"distance_ok"}
    return {
        "kind": "distance_only_replan" if distance_only else "other_failure",
        "distance_only": bool(distance_only),
        "failed_checks": failed_checks,
        "verification_min_distance_m": authorization.get("verification_min_distance_m"),
    }


def classify_monitor_stop(execution: dict[str, Any]) -> dict[str, Any]:
    """Classify a monitor stop without weakening fail-closed behavior."""
    execution_status = execution.get("status")
    goal_check = execution.get("goal_check") or {}
    motion_monitor = goal_check.get("motion_monitor") or {}
    command_time = execution.get("command_time_revalidation") or {}
    final_barrier = execution.get("final_precommand_barrier") or {}
    reason = (
        motion_monitor.get("reason")
        or final_barrier.get("reason")
        or command_time.get("reason")
        or goal_check.get("monitor_stop_reason")
    )
    monitor_stopped = bool(
        execution_status == "STOPPED_BY_MOTION_MONITOR"
        and goal_check.get("monitor_stopped", False)
    )
    command_time_replan = execution_status == COMMAND_TIME_REPLAN_STATUS
    precommand_hold = execution_status == "FINAL_PRECOMMAND_HOLD_PRECOMMAND"
    monitor_stopped = bool(monitor_stopped or command_time_replan)
    rolling_replan_stop = bool(
        command_time_replan
        or (
            monitor_stopped
            and isinstance(reason, str)
            and reason in ROLLING_REPLAN_MONITOR_REASONS
        )
    )
    return {
        "execution_status": execution_status,
        "monitor_stopped": monitor_stopped,
        "rolling_replan_stop": rolling_replan_stop,
        "precommand_replan": command_time_replan,
        "precommand_hold": precommand_hold,
        "reason": reason,
        "replan_requested": bool(
            command_time_replan
            or motion_monitor.get("replan_requested", goal_check.get("replan_requested", False))
        ),
    }


def resolve_monitor_trajectory(context: dict[str, Any]) -> Any:
    trajectory = context.get("trajectory")
    if trajectory is not None:
        return trajectory
    return trial.reconstruct_saved_nubs_candidate(
        Path(context["authorized_csv"]), segments=5
    )


def make_mid_execution_monitor(**context: Any):
    """Monitor the remaining local trajectory using the persistent tracker.

    A predictive stop is deliberately reported through the existing motion
    monitor interface.  The event handler then receives the measured partial
    configuration and performs the single allowed continuation replan.
    """
    trajectory = resolve_monitor_trajectory(context)
    worker = context.get("worker")
    evaluator, verifier, _ = trial.make_risk_stack(
        context["stage4_config"], context["stage4_model"], None
    )
    args = context["args"]
    rolling_continuation = bool(context.get("rolling_continuation", False))
    preplan_state = {"started": False, "done": False, "result": None, "source_state_seq": None}
    preplan_lock = threading.Lock()
    preplan_thread = None
    reference = context.get("reference")
    risk_links = set(context.get("risk_links") or [])
    local_artifacts = context.get("local_artifacts")
    local_index = int(context.get("event_local_index", 1))

    def run_preplan(snapshot: dict[str, Any], elapsed_s: float) -> None:
        planning_started = time.perf_counter()
        try:
            remaining_s = max(0.0, float(trajectory.total_duration) - float(elapsed_s))
            aligned = v3.time_aligned_snapshot(snapshot, execution_timestamp=time.time())
            geometry = copy.deepcopy(aligned.get("geometry"))
            if geometry is None or not geometry.get("covered", False) or reference is None or not risk_links or local_artifacts is None:
                result = {"ready": False, "reason": "rolling_preplan_context_missing"}
            else:
                velocity = np.asarray(snapshot["velocity"], dtype=np.float64)
                delta = velocity * remaining_s
                centers = np.asarray(geometry["component_centers"], dtype=np.float64) + delta[None, :]
                geometry["component_centers"] = centers.tolist()
                fresh = {"accepted": True, "reason": "rolling_preplan_projected_tail_state", "center": (np.asarray(aligned["propagated_center"], dtype=np.float64) + delta).tolist(), "velocity": velocity.tolist(), "radius": float(max(np.asarray(geometry["component_base_radii"], dtype=np.float64))), "last_timestamp": time.time() + remaining_s, "track_id": int(snapshot.get("track_id", 1)), "preplan_only": True}
                out_dir = Path(context.get("root_trial_dir", context.get("trial_dir", ROOT))) / f"event_local_{local_index + 1:02d}" / "preplan"
                out_dir.mkdir(parents=True, exist_ok=True)
                artifacts: dict[str, Any] = {}
                ref_goal, ref_audit = next_recorded_reference_goal(reference, np.asarray(trajectory.evaluate(trajectory.total_duration), dtype=np.float64), args.local_horizon_s)
                candidate = plan_goal_directed_continuation(
                    live.ACTIVE_BASE_FAST_REPAIR, args, context["stage4_config"], context["stage4_model"],
                    q_escape_start=np.asarray(local_artifacts["q_now"], dtype=np.float64),
                    q_now=np.asarray(trajectory.evaluate(trajectory.total_duration), dtype=np.float64),
                    q_final=np.asarray(reference.q[-1], dtype=np.float64), fresh=fresh, geometry=geometry,
                    risk_links=risk_links, trial_dir=out_dir, nominal_reference_goal=ref_goal,
                    artifacts_out=artifacts, forward_m=float(args.forward_m), side_m=float(args.continuation_side_m),
                    robust_target_m=float(args.planning_robust_target_m), max_joint_delta_rad=float(args.max_joint_delta_rad),
                    tcp_link=args.tcp_link, robust_target_is_diagnostic=True,
                )
                result = {"ready": bool(candidate.get("local_repair_ready", False)), "candidate": candidate, "artifacts": artifacts, "projected_fresh": fresh, "projected_geometry": geometry, "reference_goal_audit": ref_audit, "source_state_seq": int(v3._state_seq(snapshot)), "trigger_elapsed_s": float(elapsed_s), "planning_wall_ms": 1000.0 * (time.perf_counter() - planning_started), "speculative_only": True}
            with preplan_lock:
                preplan_state.update({"done": True, "result": result})
        except Exception as exc:
            with preplan_lock:
                preplan_state.update({"done": True, "result": {"ready": False, "reason": "rolling_preplan_exception", "error": repr(exc)}})

    def take_rolling_preplan(*, wait_s: float = 0.05) -> dict[str, Any] | None:
        if preplan_thread is not None and preplan_thread.is_alive() and wait_s > 0.0:
            preplan_thread.join(timeout=float(wait_s))
        with preplan_lock:
            return copy.deepcopy(preplan_state["result"])
    last = {"seq": -1}

    def prearm(*, timeout_s: float = 1.5, required_updates: int = 2) -> dict[str, Any]:
        """Require fresh persistent perception before arming robot motion.

        The monitor is fail-closed during playback, but that first callback is
        too late to protect the command barrier: a missing worker would allow
        the controller to start and then stop a fraction of a second later.
        Synchronize here, before ``offline_track_execute_joints`` is called.
        """
        if worker is None:
            return {"ready": False, "reason": "persistent_tracker_unavailable"}
        if rolling_continuation:
            snapshot = worker.snapshot()
            aligned = v3.time_aligned_snapshot(snapshot, execution_timestamp=time.time())
            reasons = v3._persistent_state_reasons(
                snapshot, aligned, args, raw_guard_reason="raw_hard_guard_not_safe"
            )
            raw = float(snapshot.get("raw_guard_distance_m", float("-inf")))
            seq = int(v3._state_seq(snapshot))
            ready = bool(not reasons and raw > float(args.guided_hard_stop_m))
            if ready:
                last["prearm_seq"] = seq
            return {
                "ready": ready,
                "reason": None if ready else (reasons or "raw_hard_guard_not_safe"),
                "baseline_state_seq": seq,
                "rolling_continuation": True,
                "no_wait": True,
                "samples": [{"state_seq": seq, "state_age_s": float(aligned["propagation_dt_s"]), "raw_guard_distance_m": raw, "reasons": reasons}],
            }
        baseline = worker.snapshot()
        baseline_seq = v3._state_seq(baseline)
        latest = baseline
        samples = []
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while len(samples) < int(required_updates) and time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            latest = worker.wait_for_newer_state(
                after_seq=v3._state_seq(latest),
                timeout_s=min(0.25, remaining),
            )
            seq = v3._state_seq(latest)
            if seq <= baseline_seq or any(item["state_seq"] == seq for item in samples):
                continue
            aligned = v3.time_aligned_snapshot(latest, execution_timestamp=time.time())
            reasons = v3._persistent_state_reasons(
                latest, aligned, args, raw_guard_reason="raw_hard_guard_not_safe"
            )
            samples.append(
                {
                    "state_seq": seq,
                    "state_age_s": float(aligned["propagation_dt_s"]),
                    "raw_guard_distance_m": float(latest.get("raw_guard_distance_m", float("-inf"))),
                    "reasons": reasons,
                }
            )
            if reasons:
                return {
                    "ready": False,
                    "reason": reasons,
                    "baseline_state_seq": baseline_seq,
                    "samples": samples,
                }
        ready = len(samples) >= int(required_updates)
        if ready and samples:
            last["prearm_seq"] = int(samples[-1]["state_seq"])
        return {
            "ready": ready,
            "reason": None if ready else "persistent_tracker_no_fresh_updates",
            "baseline_state_seq": baseline_seq,
            "samples": samples,
        }

    def wait_for_final_fresh_snapshot(*, after_seq: int) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Require a newer valid state, retrying mildly old samples until timeout."""
        if worker is None:
            return {"ready": False, "reason": "persistent_tracker_unavailable", "after_seq": int(after_seq)}, None
        started = time.monotonic()
        deadline = started + max(0.0, float(args.final_precommand_fresh_timeout_s))
        latest = worker.snapshot()
        cursor_seq = int(after_seq)
        samples: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            seq = int(v3._state_seq(latest))
            if seq <= cursor_seq:
                remaining = max(0.0, deadline - time.monotonic())
                newer = worker.wait_for_newer_state(after_seq=seq, timeout_s=min(0.10, remaining))
                if newer is None:
                    continue
                latest = newer
                continue
            aligned = v3.time_aligned_snapshot(latest, execution_timestamp=time.time())
            age = float(aligned["propagation_dt_s"])
            reasons = list(v3._persistent_state_reasons(latest, aligned, args, raw_guard_reason="raw_hard_guard_not_safe"))
            raw = float(latest.get("raw_guard_distance_m", float("-inf")))
            sample = {"state_seq": seq, "state_age_s": age, "raw_guard_distance_m": raw, "base_reasons": list(reasons)}
            samples.append(sample)
            # Physical hard guard and genuine state invalidity fail closed
            # immediately.  A newer sample that is only mildly old is retried.
            if raw <= float(args.guided_hard_stop_m):
                reasons = list(reasons) or ["raw_hard_guard_not_safe"]
                return {"ready": False, "reason": reasons, "after_seq": int(after_seq), "state_seq": seq, "state_age_s": age, "raw_guard_distance_m": raw, "samples": samples, "wait_elapsed_s": time.monotonic() - started}, None
            if reasons:
                return {"ready": False, "reason": reasons, "after_seq": int(after_seq), "state_seq": seq, "state_age_s": age, "raw_guard_distance_m": raw, "samples": samples, "wait_elapsed_s": time.monotonic() - started}, None
            if age > float(args.final_precommand_max_state_age_s):
                sample["reason"] = "final_precommand_state_too_old_retry"
                cursor_seq = seq
                remaining = max(0.0, deadline - time.monotonic())
                newer = worker.wait_for_newer_state(after_seq=cursor_seq, timeout_s=min(0.10, remaining))
                if newer is not None:
                    latest = newer
                continue
            return {"ready": True, "reason": None, "after_seq": int(after_seq), "state_seq": seq, "state_age_s": age, "raw_guard_distance_m": raw, "samples": samples, "wait_elapsed_s": time.monotonic() - started}, latest
        return {"ready": False, "reason": "final_precommand_freshness_timeout", "after_seq": int(after_seq), "samples": samples, "wait_elapsed_s": time.monotonic() - started}, None

    def boundary_dynamics_audit(trajectory_obj: Any, q_actual: np.ndarray) -> dict[str, Any]:
        """Reject startup/stop discontinuities; never clip a verified trajectory."""
        total = float(trajectory_obj.total_duration)
        if total <= 0.03:
            return {"passed": False, "reason": "trajectory_too_short"}
        dt = min(0.005, total / 50.0)
        ts = [0.0, dt, 2.0 * dt, 3.0 * dt]
        te = [total, total - dt, total - 2.0 * dt, total - 3.0 * dt]
        q0, q1, q2, q3 = [np.asarray(trajectory_obj.evaluate(t), dtype=np.float64) for t in ts]
        qe, qe1, qe2, qe3 = [np.asarray(trajectory_obj.evaluate(t), dtype=np.float64) for t in te]
        qd0 = (-3*q0 + 4*q1 - q2) / (2*dt)
        qdd0 = (2*q0 - 5*q1 + 4*q2 - q3) / (dt*dt)
        qde = (3*qe - 4*qe1 + qe2) / (2*dt)
        qdde = (2*qe - 5*qe1 + 4*qe2 - qe3) / (dt*dt)
        start_error = float(np.max(np.abs(q0 - np.asarray(q_actual, dtype=np.float64))))
        checks = {
            "start_position_ok": start_error <= float(args.candidate_start_sync_rad),
            "first_step_ok": float(np.max(np.abs(q1-q0))) <= float(args.candidate_start_sync_rad),
            "final_step_ok": float(np.max(np.abs(qe-qe1))) <= float(args.candidate_start_sync_rad),
            "start_velocity_ok": float(np.max(np.abs(qd0))) <= float(args.boundary_qd_tol_rad_s),
            "end_velocity_ok": float(np.max(np.abs(qde))) <= float(args.boundary_qd_tol_rad_s),
            "start_acceleration_ok": float(np.max(np.abs(qdd0))) <= float(args.boundary_qdd_tol_rad_s2),
            "end_acceleration_ok": float(np.max(np.abs(qdde))) <= float(args.boundary_qdd_tol_rad_s2),
        }
        return {"passed": all(checks.values()), "checks": checks, "failed_checks": [k for k, v in checks.items() if not v], "sample_dt_s": dt, "start_position_error_rad": start_error, "start_qd_max_rad_s": float(np.max(np.abs(qd0))), "end_qd_max_rad_s": float(np.max(np.abs(qde))), "start_qdd_max_rad_s2": float(np.max(np.abs(qdd0))), "end_qdd_max_rad_s2": float(np.max(np.abs(qdde)))}

    def final_precommand_barrier(*, actual_q: np.ndarray) -> dict[str, Any]:
        after_seq = int(last.get("command_time_seq", last.get("prearm_seq", -1)))
        freshness, snapshot = wait_for_final_fresh_snapshot(after_seq=after_seq)
        if not freshness.get("ready", False) or snapshot is None:
            return {"ready": False, "action": "hold", "reason": "final_precommand_freshness_not_ready", "freshness": freshness}
        raw = float(snapshot.get("raw_guard_distance_m", float("-inf")))
        if raw <= float(args.guided_hard_stop_m):
            return {"ready": False, "action": "hold", "reason": "final_precommand_raw_guard_not_safe", "freshness": freshness, "raw_guard_distance_m": raw}
        aligned = v3.time_aligned_snapshot(snapshot, execution_timestamp=time.time())
        forecast = v3.v3_execution_multisphere_forecast(
            np.asarray(aligned["geometry"]["component_centers"], dtype=np.float64),
            np.asarray(aligned["geometry"]["component_base_radii"], dtype=np.float64),
            np.asarray(snapshot["velocity"], dtype=np.float64),
        )
        q_actual = np.asarray(actual_q, dtype=np.float64)
        q_goal = np.asarray(trajectory.evaluate(trajectory.total_duration), dtype=np.float64)
        verification = verifier.verify(
            trajectory, forecast, current_q=q_actual,
            current_qd=np.zeros(6), current_qdd=np.zeros(6),
            q_goal=q_goal, solver_success=True,
        )
        non_distance_failures = [
            key for key, passed in verification.checks.items()
            if key != "distance_ok" and not bool(passed)
        ]
        if non_distance_failures:
            return {"ready": False, "action": "hold", "reason": "final_precommand_verification_failed", "failed_checks": non_distance_failures, "verification_checks": verification.checks, "verification_min_distance_m": float(verification.min_distance), "freshness": freshness}
        if not verification.accepted or float(verification.min_distance) < float(args.online_accept_m):
            return {"ready": False, "action": "replan", "reason": "final_precommand_candidate_clearance_failed", "verification_min_distance_m": float(verification.min_distance), "verification_checks": verification.checks, "freshness": freshness}
        remaining = v3._remaining_clearance(evaluator, trajectory, forecast, playback_time_s=0.0)
        if float(remaining["min_distance_m"]) < float(args.rolling_replan_m):
            return {"ready": False, "action": "replan", "reason": "final_precommand_remaining_predicted_risk", "remaining_clearance_m": float(remaining["min_distance_m"]), "freshness": freshness}
        dynamics = boundary_dynamics_audit(trajectory, np.asarray(actual_q, dtype=np.float64))
        if not dynamics.get("passed", False):
            return {"ready": False, "action": "replan", "reason": "final_precommand_boundary_dynamics_failed", "freshness": freshness, "boundary_dynamics": dynamics}
        last["final_precommand_seq"] = int(v3._state_seq(snapshot))
        return {"ready": True, "action": "execute", "reason": "final_precommand_candidate_valid", "freshness": freshness, "raw_guard_distance_m": raw, "verification_min_distance_m": float(verification.min_distance), "remaining_clearance_m": float(remaining["min_distance_m"]), "boundary_dynamics": dynamics, "state_seq": int(v3._state_seq(snapshot))}

    def monitor(*, elapsed_s: float, actual_q: np.ndarray, obstacle_snapshot: Any = None) -> dict[str, Any]:
        nonlocal preplan_thread
        del obstacle_snapshot
        if worker is None:
            return {"motion_safe": False, "reason": "persistent_tracker_unavailable", "state_age_s": None, "final_precommand_state_seq": int(last.get("final_precommand_seq", -1))}
        snapshot = worker.snapshot()
        aligned = v3.time_aligned_snapshot(snapshot, execution_timestamp=time.time())
        state_reasons = v3._persistent_state_reasons(
            snapshot, aligned, args, raw_guard_reason="raw_hard_guard_not_safe"
        )
        raw_guard = float(snapshot.get("raw_guard_distance_m", float("-inf")))
        if state_reasons or raw_guard <= float(args.guided_hard_stop_m):
            return {
                "motion_safe": False,
                "replan_requested": False,
                "reason": state_reasons or ["raw_hard_guard_not_safe"],
                "raw_guard_distance_m": raw_guard,
                "state_seq": int(snapshot.get("state_seq", -1)),
                "state_age_s": float(aligned["propagation_dt_s"]),
                "final_precommand_state_seq": int(last.get("final_precommand_seq", -1)),
                "state_seq_delta_from_final": int(v3._state_seq(snapshot)) - int(last.get("final_precommand_seq", -1)),
            }
        forecast = v3.v3_execution_multisphere_forecast(
            np.asarray(aligned["geometry"]["component_centers"], dtype=np.float64),
            np.asarray(aligned["geometry"]["component_base_radii"], dtype=np.float64),
            np.asarray(snapshot["velocity"], dtype=np.float64),
        )
        current = evaluator.configuration(
            np.asarray(actual_q, dtype=np.float64), forecast, 0.0,
            density="medium", with_gradient=False,
        )
        remaining = v3._remaining_clearance(
            evaluator, trajectory, forecast,
            playback_time_s=min(float(elapsed_s), float(trajectory.total_duration)),
        )
        current_near_diagnostic = bool(
            float(current.min_distance) <= float(args.moving_shadow_current_stop_m)
        )
        prediction_gate_m = float(args.rolling_replan_m)
        modeled_remaining_min_m = min(float(current.min_distance), float(remaining["min_distance_m"]))
        remaining_exec_s = max(0.0, float(trajectory.total_duration) - float(elapsed_s))
        should_preplan = bool(
            float(elapsed_s) >= float(args.rolling_preplan_trigger_s)
            and remaining_exec_s >= float(args.rolling_preplan_min_lead_s)
            and modeled_remaining_min_m <= float(args.rolling_preplan_clearance_m)
            and not preplan_state["started"]
            and reference is not None
            and local_artifacts is not None
        )
        if should_preplan:
            with preplan_lock:
                preplan_state["started"] = True
                preplan_state["source_state_seq"] = int(v3._state_seq(snapshot))
            preplan_thread = threading.Thread(target=run_preplan, args=(copy.deepcopy(snapshot), float(elapsed_s)), daemon=True, name=f"ccro-preplan-{local_index + 1}")
            preplan_thread.start()
        if modeled_remaining_min_m < prediction_gate_m:
            reason = "remaining_predicted_risk"
            replan = True
        else:
            reason = "predictive_remaining_clear"
            replan = False
        last.update({"seq": int(snapshot.get("state_seq", -1))})
        return {
            "motion_safe": bool(reason == "predictive_remaining_clear"),
            "replan_requested": replan,
            "reason": reason,
            "current_distance_m": float(current.min_distance),
            "remaining_clearance_m": float(remaining["min_distance_m"]),
            "remaining_tau_s": remaining.get("tau_s"),
            "raw_guard_distance_m": raw_guard,
            "state_seq": int(snapshot.get("state_seq", -1)),
            "state_age_s": float(aligned["propagation_dt_s"]),
            "final_precommand_state_seq": int(last.get("final_precommand_seq", -1)),
            "state_seq_delta_from_final": int(v3._state_seq(snapshot)) - int(last.get("final_precommand_seq", -1)),
            "current_near_diagnostic": current_near_diagnostic,
            "current_near_threshold_m": float(args.moving_shadow_current_stop_m),
            "predictive_replan_threshold_m": prediction_gate_m,
            "rolling_preplan": {"triggered": bool(preplan_state["started"]), "done": bool(preplan_state["done"]), "source_state_seq": preplan_state["source_state_seq"]},
        }

    def command_time_revalidate(*, actual_q: np.ndarray) -> dict[str, Any]:
        """Final no-wait candidate check immediately before waypoint command."""
        if worker is None:
            return {"ready": False, "action": "hold", "reason": "persistent_tracker_unavailable"}
        snapshot = worker.snapshot()
        aligned = v3.time_aligned_snapshot(snapshot, execution_timestamp=time.time())
        reasons = v3._persistent_state_reasons(
            snapshot, aligned, args, raw_guard_reason="raw_hard_guard_not_safe"
        )
        raw_guard = float(snapshot.get("raw_guard_distance_m", float("-inf")))
        state_seq = int(v3._state_seq(snapshot))
        if reasons or raw_guard <= float(args.guided_hard_stop_m):
            return {
                "ready": False, "action": "hold",
                "reason": reasons or "raw_hard_guard_not_safe",
                "state_seq": state_seq, "raw_guard_distance_m": raw_guard,
            }
        q_actual = np.asarray(actual_q, dtype=np.float64)
        start_error = trial.joint_error(q_actual, np.asarray(trajectory.evaluate(0.0), dtype=np.float64))
        if start_error["max_abs_rad"] > float(args.candidate_start_sync_rad):
            return {"ready": False, "action": "replan", "reason": "command_time_start_mismatch", "state_seq": state_seq, "start_error": start_error, "raw_guard_distance_m": raw_guard}
        forecast = v3.v3_execution_multisphere_forecast(
            np.asarray(aligned["geometry"]["component_centers"], dtype=np.float64),
            np.asarray(aligned["geometry"]["component_base_radii"], dtype=np.float64),
            np.asarray(snapshot["velocity"], dtype=np.float64),
        )
        current = evaluator.configuration(q_actual, forecast, 0.0, density="medium", with_gradient=False)
        current_near_diagnostic = bool(
            float(current.min_distance) <= float(args.moving_shadow_current_stop_m)
        )
        distance_diag = {
            "current_distance_m": float(current.min_distance),
            "current_near_diagnostic": current_near_diagnostic,
            "current_near_threshold_m": float(args.moving_shadow_current_stop_m),
        }
        q_goal = np.asarray(trajectory.evaluate(trajectory.total_duration), dtype=np.float64)
        verification = verifier.verify(trajectory, forecast, current_q=q_actual, current_qd=np.zeros(6), current_qdd=np.zeros(6), q_goal=q_goal, solver_success=True)
        failures = [key for key, passed in verification.checks.items() if key != "distance_ok" and not bool(passed)]
        if failures:
            return {"ready": False, "action": "hold", "reason": "command_time_verification_failed", "failed_checks": failures, "verification_checks": verification.checks, "verification_min_distance_m": float(verification.min_distance), "state_seq": state_seq, "raw_guard_distance_m": raw_guard, **distance_diag}
        if not verification.accepted or float(verification.min_distance) < float(args.online_accept_m):
            return {"ready": False, "action": "replan", "reason": "command_time_candidate_clearance_failed", "verification_checks": verification.checks, "verification_min_distance_m": float(verification.min_distance), "state_seq": state_seq, "raw_guard_distance_m": raw_guard, **distance_diag}
        remaining = v3._remaining_clearance(evaluator, trajectory, forecast, playback_time_s=0.0)
        prediction_gate_m = float(args.rolling_replan_m)
        modeled_remaining_min_m = min(float(current.min_distance), float(remaining["min_distance_m"]))
        if modeled_remaining_min_m < prediction_gate_m:
            return {"ready": False, "action": "replan", "reason": "remaining_predicted_risk", "state_seq": state_seq, "remaining_clearance_m": float(remaining["min_distance_m"]), "remaining_tau_s": remaining.get("tau_s"), "verification_min_distance_m": float(verification.min_distance), "raw_guard_distance_m": raw_guard, "predictive_replan_threshold_m": prediction_gate_m, **distance_diag}
        last["command_time_seq"] = state_seq
        return {"ready": True, "action": "execute", "reason": "command_time_candidate_valid", "state_seq": state_seq, "remaining_clearance_m": float(remaining["min_distance_m"]), "remaining_tau_s": remaining.get("tau_s"), "verification_min_distance_m": float(verification.min_distance), "verification_checks": verification.checks, "raw_guard_distance_m": raw_guard, "predictive_replan_threshold_m": prediction_gate_m, **distance_diag}

    # ``execute_authorized_trajectory_offline_track`` uses this explicit
    # barrier before sending any waypoint command.  Keep the normal callback
    # fail-closed as well for failures that occur after motion has started.
    monitor.prearm = prearm
    monitor.command_time_revalidate = command_time_revalidate
    monitor.final_precommand_barrier = final_precommand_barrier
    monitor.take_rolling_preplan = take_rolling_preplan
    monitor.rolling_continuation = rolling_continuation
    return monitor


def build_parser() -> argparse.ArgumentParser:
    parser = live.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        output=DEFAULT_OUTPUT,
        task_geometry_id="D2_SIMPLE_DYNAMIC_NUBS_EVENT_REPLAN_LIVE_XP00",
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
        help="strong retained-side displacement for each continuation local segment; weak uses half and release uses zero",
    )
    parser.add_argument(
        "--rolling-replan-m",
        type=float,
        default=0.09,
        help="modeled remaining-clearance threshold during local recovery; raw hard guard remains 0.10 m",
    )
    parser.add_argument(
        "--max-continuous-replan-s",
        type=float,
        default=10.0,
        help="fail-safe watchdog for one continuous dynamic-replan episode; not a local-segment limit",
    )
    parser.add_argument("--final-precommand-fresh-timeout-s", type=float, default=0.35)
    parser.add_argument("--final-precommand-max-state-age-s", type=float, default=0.35)
    parser.add_argument("--boundary-qd-tol-rad-s", type=float, default=0.03)
    parser.add_argument("--boundary-qdd-tol-rad-s2", type=float, default=0.30)
    parser.add_argument("--rolling-preplan-trigger-s", type=float, default=0.40)
    parser.add_argument("--rolling-preplan-clearance-m", type=float, default=0.12)
    parser.add_argument("--rolling-preplan-min-lead-s", type=float, default=0.25)
    parser.add_argument(
        "--stationary-terminal-full-plan", action="store_true",
        help="after confirmed stationary recovery, use one full static CCRO-NUBS terminal plan",
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
            v3.v3_execution_multisphere_forecast(
                np.asarray(geometry["component_centers"], dtype=np.float64),
                np.asarray(geometry["component_base_radii"], dtype=np.float64),
                np.asarray(fresh["velocity"], dtype=np.float64),
            ),
            "FRESH_TRACKED_OBSTACLE",
        )
    if strict_empty_scene(args, frames):
        return (
            v3.v3_execution_multisphere_forecast(
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


def fresh_from_persistent_snapshot(snapshot: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], None, dict[str, Any] | None]:
    """Build an execution-time Fresh state without a blocking three-frame wait."""
    aligned = v3.time_aligned_snapshot(snapshot, execution_timestamp=time.time())
    geometry = aligned["geometry"]
    fresh = {
        "accepted": bool(geometry.get("covered", False)),
        "reason": "persistent_worker_latest_authorization_state",
        "track_id": int(snapshot.get("track_id", 1)),
        "center": np.asarray(aligned["propagated_center"], dtype=np.float64).tolist(),
        "velocity": np.asarray(snapshot["velocity"], dtype=np.float64).tolist(),
        "radius": float(max(np.asarray(geometry["component_base_radii"], dtype=np.float64))),
        "last_timestamp": float(time.time()),
        "max_association_error_m": float(snapshot.get("association_error_m", 0.0)),
        "source": "persistent_worker_latest_authorization_state",
        "state_seq": int(v3._state_seq(snapshot)),
    }
    return fresh, [], None, geometry if fresh["accepted"] else None


def raw_guard_from_persistent_snapshot(snapshot: dict[str, Any]) -> float:
    value = float(snapshot.get("raw_guard_distance_m", float("-inf")))
    return value if math.isfinite(value) else float("-inf")


def wait_for_confirmed_stationary_snapshot(
    worker: Any,
    args: argparse.Namespace,
    *,
    speed_threshold_m_s: float = 0.04,
    required_frames: int = 3,
    timeout_s: float = 1.5,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Require a valid, raw-safe, three-frame stationary track before Full CCRO."""
    if worker is None:
        return {"confirmed": False, "reason": "tracker_unavailable", "samples": []}, None
    latest = worker.snapshot()
    cursor = int(v3._state_seq(latest))
    streak = 0
    samples: list[dict[str, Any]] = []
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        newer = worker.wait_for_newer_state(
            after_seq=cursor,
            timeout_s=min(0.20, max(0.0, deadline - time.monotonic())),
        )
        if newer is None:
            continue
        latest = newer
        cursor = int(v3._state_seq(latest))
        aligned = v3.time_aligned_snapshot(latest, execution_timestamp=time.time())
        reasons = v3._persistent_state_reasons(
            latest, aligned, args, raw_guard_reason="raw_hard_guard_not_safe"
        )
        raw = raw_guard_from_persistent_snapshot(latest)
        velocity = np.asarray(latest.get("velocity", [0.0, 0.0, 0.0]), dtype=float)
        speed = float(np.linalg.norm(velocity))
        geometry = aligned.get("geometry")
        valid = bool(
            not reasons and geometry is not None and geometry.get("covered", False)
            and raw > float(args.guided_hard_stop_m)
        )
        stationary = bool(valid and speed <= float(speed_threshold_m_s))
        streak = streak + 1 if stationary else 0
        samples.append({"state_seq": cursor, "speed_m_s": speed,
                        "raw_guard_distance_m": raw, "stationary": stationary,
                        "streak": streak})
        if streak >= int(required_frames):
            return {
                "confirmed": True, "reason": "stationary_track_confirmed",
                "samples": samples, "state_seq": cursor, "speed_m_s": speed,
            }, latest
    return {"confirmed": False, "reason": "stationary_confirmation_timeout", "samples": samples}, None


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
        # A visible obstacle is not automatically a remaining task risk.  If
        # the stopped tail is safe over the prediction horizon, hand the
        # latest forecast to terminal-goal verification immediately.  The
        # terminal verifier and raw guard remain hard execution gates.
        cycle["decision"] = "PREDICTED_RISK_CLEAR"
        cycles.append(cycle)
        result = {
            "status": "PREDICTED_RISK_CLEAR",
            "cycles": cycles,
            "fresh": fresh,
            "frames": frames,
            "geometry": geometry,
            "forecast": forecast,
            "terminal_probe_required": True,
            "task_path_clear": None,
        }
        trial.write_json(output_dir / "monitor_summary.json", {k: v for k, v in result.items() if k != "forecast"})
        return result


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
    tabletop_gate_is_hard: bool = False,
) -> dict[str, Any] | None:
    """Rank continuation seeds without weakening the final verifier.

    Legacy event recovery retains its 0.11 m coarse hard gate.  V3 uses that
    number only as a preference: every finite seed may enter Fast, while the
    unchanged 0.09 m complete verifier remains the execution authority.
    """
    tabletop_eligible = [row for row in rows if not tabletop_gate_is_hard or bool(row.get("tabletop_feasible", False))]
    if diagnostic_only:
        eligible = [
            row for row in tabletop_eligible if math.isfinite(row["coarse_min_distance_m"])
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
        for row in tabletop_eligible
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
    allow_unestablished_side_fallback: bool = False,
) -> dict[str, Any]:
    """Select strong/weak/release goal progress, then invoke unchanged Fast."""
    forecast = v3.v3_execution_multisphere_forecast(
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
    try:
        side, side_audit = established_bypass_side(
            model, q_escape_start, q_now, q_final, tcp_link=tcp_link
        )
    except RuntimeError:
        if not allow_unestablished_side_fallback:
            raise
        tcp_now_fallback = live.simple.tcp_position(model, np.asarray(q_now), tcp_link)
        tcp_final_fallback = live.simple.tcp_position(model, np.asarray(q_final), tcp_link)
        task_fallback = bypass.normalized(tcp_final_fallback - tcp_now_fallback)
        away = np.asarray(risk.robot_point, dtype=np.float64) - np.asarray(risk.obstacle_point, dtype=np.float64)
        lateral = away - task_fallback * float(np.dot(task_fallback, away))
        if np.linalg.norm(lateral) <= 1.0e-6:
            return {
                "status": "REJECTED_UNESTABLISHED_BYPASS_SIDE",
                "local_repair_status": "REJECTED_UNESTABLISHED_BYPASS_SIDE",
                "local_repair_ready": False,
                "accepted_for_switch": False,
                "rejection_reasons": ["cannot_construct_lateral_escape_side"],
            }
        try:
            side = bypass.tabletop_parallel_lateral_direction(task_fallback, lateral)
        except ValueError:
            return {
                "status": "REJECTED_UNESTABLISHED_TABLETOP_BYPASS_SIDE",
                "local_repair_status": "REJECTED_UNESTABLISHED_TABLETOP_BYPASS_SIDE",
                "local_repair_ready": False,
                "accepted_for_switch": False,
                "rejection_reasons": ["cannot_construct_tabletop_parallel_escape_side"],
            }
        side_audit = {
            "tcp_escape_start_m": tcp_now_fallback.tolist(),
            "tcp_now_m": tcp_now_fallback.tolist(),
            "tcp_final_m": tcp_final_fallback.tolist(),
            "executed_tcp_delta_m": [0.0, 0.0, 0.0],
            "lateral_executed_delta_m": [0.0, 0.0, 0.0],
            "established_bypass_side": side.tolist(),
            "semantic": "risk_away_fallback_before_first_local_motion",
        }
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
        tabletop_parallel_side=True,
        preserve_tcp_height=True,
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
        tabletop_guard = trial.gripper_base_workspace_guard(
            trajectory, model, min_z_m=float(getattr(runtime_args, "gripper_base_min_z_m", 0.46))
        )
        tcp_end = live.simple.tcp_position(
            model, trajectory.evaluate(trajectory.total_duration), tcp_link
        )
        progress = float(np.dot(tcp_end - tcp_now, task))
        rows.append(
            {
                **{key: item[key] for key in ("candidate", "phase", "side_weight", "side_m", "forward_m")},
                "mapping": item["mapping"],
                "tabletop_workspace_guard": tabletop_guard,
                "tabletop_feasible": bool(tabletop_guard["passed"]),
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
        tabletop_gate_is_hard=True,
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
        if not any(bool(row.get("tabletop_feasible", False)) for row in rows):
            return {
                "status": "REJECTED_NO_TABLETOP_SAFE_GOAL_DIRECTED_CONTINUATION",
                "local_repair_status": "REJECTED_NO_TABLETOP_SAFE_GOAL_DIRECTED_CONTINUATION",
                "local_repair_ready": False,
                "accepted_for_switch": False,
                "rejection_reasons": ["no_tabletop_safe_goal_directed_continuation"],
                "goal_directed_continuation_audit": audit,
            }
        return {
            "status": "REJECTED_NO_GOAL_DIRECTED_ROBUST_CONTINUATION",
            "local_repair_status": "REJECTED_NO_GOAL_DIRECTED_ROBUST_CONTINUATION",
            "local_repair_ready": False,
            "accepted_for_switch": False,
            "rejection_reasons": ["no_goal_directed_candidate_at_or_above_0.11m"],
            "goal_directed_continuation_audit": audit,
        }
    selected_goal = goals[int(selected["candidate"]) - 1]
    event_local_index = None
    if trial_dir.name.startswith("event_local_"):
        try:
            event_local_index = int(trial_dir.name.rsplit("_", 1)[-1])
        except ValueError:
            event_local_index = None
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
        obstacle_audit={
            "track_id": int(fresh.get("track_id", 1)),
            "event_local_index": event_local_index,
            "phase": "bypass_progression",
        },
        multisphere_geometry=geometry,
        artifacts_out=artifacts_out,
        # V3: improvement and motion metrics remain diagnostics; the complete
        # absolute verifier decides whether this continuation may proceed.
        accept_verified_seed_without_fast_step=True,
        original_task_reference_goal=nominal_reference_goal,
    )
    result = live.apply_tabletop_height_shape_policy(
        result=result,
        artifacts_out=artifacts_out,
        runtime_args=runtime_args,
        config=config,
        model=model,
        forecast=forecast,
        q_now=np.asarray(q_now),
        qd_now=np.zeros(6),
        trial_dir=trial_dir,
        max_drop_m=0.005,
    )
    if result.get("local_repair_ready"):
        candidate_trajectory = artifacts_out.get("candidate_trajectory")
        guard = None if candidate_trajectory is None else trial.gripper_base_workspace_guard(
            candidate_trajectory, model, min_z_m=float(getattr(runtime_args, "gripper_base_min_z_m", 0.46))
        )
        result["planning_tabletop_guard"] = guard
        if guard is None or not guard.get("passed", False):
            result["status"] = "GOAL_DIRECTED_FAST_CANDIDATE_TABLETOP_GUARD_FAILED"
            result["local_repair_status"] = result["status"]
            result["local_repair_ready"] = False
            result["accepted_for_switch"] = False
            result.setdefault("rejection_reasons", []).append("gripper_base_below_tabletop_guard")
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
    stationary_geometry: dict[str, Any] | None = None,
    use_stationary_full_plan: bool = False,
) -> tuple[dict[str, Any], Any | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if use_stationary_full_plan and stationary_geometry is not None:
        payload, trajectory = stationary_ccro.plan_stationary_terminal_ccro(
            config=config, model=model, q_start=q_now, q_goal=q_goal,
            geometry=stationary_geometry, output_dir=output_dir,
            min_clearance_m=float(args.online_accept_m),
        )
        payload["planner_mode"] = "full_static_ccro_nubs"
        trial.write_json(output_dir / "authorization_summary.json", payload)
        if trajectory is not None:
            trial.save_trajectory_csv(output_dir / "authorized_terminal_goal.csv", trajectory, dt=0.01)
        return payload, trajectory
    if use_stationary_full_plan:
        payload = {
            "status": "STATIONARY_FULL_CCRO_HOLD",
            "authorized": False,
            "planner_mode": "full_static_ccro_nubs",
            "reason": "stationary_geometry_missing",
        }
        trial.write_json(output_dir / "authorization_summary.json", payload)
        return payload, None
    _, verifier, _ = trial.make_risk_stack(config, model, None)
    attempts = []
    selected = None
    for duration in durations:
        trajectory = make_terminal_trajectory(q_now, q_goal, duration)
        tabletop_guard = trial.gripper_base_workspace_guard(
            trajectory, model, min_z_m=float(getattr(args, "gripper_base_min_z_m", 0.46))
        )
        verification = verifier.verify(
            trajectory,
            forecast,
            current_q=q_now,
            current_qd=np.zeros(6),
            current_qdd=np.zeros(6),
            q_goal=q_goal,
            solver_success=True,
        )
        tabletop_ok = bool(tabletop_guard["passed"])
        authorized = bool(verification.accepted and tabletop_ok)
        checks = dict(verification.checks)
        checks["tabletop_ok"] = tabletop_ok
        reasons = list(verification.reasons)
        if not tabletop_ok:
            reasons.append("gripper_base_below_tabletop_guard")
        row = {
            "duration_s": float(duration),
            "accepted": authorized,
            "min_distance_m": float(verification.min_distance),
            "checks": checks,
            "reasons": reasons,
            "tabletop_workspace_guard": tabletop_guard,
            "verification_ms": float(verification.validation_ms),
        }
        attempts.append(row)
        if authorized:
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
        replan_depth = int(context.get("replan_depth", 0))
        replan_started = float(context.get("replan_started_monotonic", time.monotonic()))
        failed_replans = int(context.get("failed_replans", 0))
        force_goal_directed_local = bool(context.get("force_goal_directed_local", False))
        current_local_artifacts = context["local_artifacts"]
        current_execution_summary = context.get("execution_summary")
        current_local_interrupted = bool(context.get("local1_interrupted", False))
        completed_local_index = replan_depth + 1
        if time.monotonic() - replan_started > float(event_args.max_continuous_replan_s):
            result = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "handled": True,
                "status": "CONTINUOUS_REPLAN_WATCHDOG_TIMEOUT_OPERATOR_INTERVENTION_REQUIRED",
                "command_hold": True,
                "replan_depth": replan_depth,
                "watchdog_elapsed_s": time.monotonic() - replan_started,
            }
            trial.write_json(trial_dir / "event_replan_summary.json", result)
            return result
        q_actual = np.asarray(robot.get_joint(), dtype=np.float64)
        early_execution = bool(context.get("local1_interrupted", False))
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
            "local_segment_index": replan_depth + 1,
            "continuous_replan_watchdog_s": float(event_args.max_continuous_replan_s),
            "failed_replans": failed_replans,
        }
        if start_error["max_abs_rad"] > args.candidate_start_tolerance_rad and not early_execution:
            result["status"] = "EVENT_REPLAN_BLOCKED_LOCAL1_TAIL_MISMATCH"
            trial.write_json(trial_dir / "event_replan_summary.json", result)
            return result

        result["fresh3_raw_guard_distance_m"] = float(context["fresh3_guard_distance"])
        result["local1_interrupted"] = early_execution
        result["local1_execution"] = context.get("execution_summary")
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
            output_dir=trial_dir / f"post_local{replan_depth + 1}_monitor",
            max_wall_s=float(event_args.post_local_monitor_max_s),
        )
        tail_monitor = monitor1
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

        needs_goal_directed_local = bool(
            monitor1["status"] == "REPLAN_REQUIRED"
            or (force_goal_directed_local and monitor1["status"] == "PREDICTED_RISK_CLEAR")
        )
        if needs_goal_directed_local:
            result["local_continuation_reason"] = (
                "terminal_direct_path_blocked"
                if force_goal_directed_local
                else "stationary_hold_not_safe"
            )
            local_index = replan_depth + 2
            local2_dir = trial_dir / f"event_local_{local_index:02d}"
            local2_dir.mkdir(parents=True, exist_ok=True)
            artifacts: dict[str, Any] = {}
            local2_reference_goal, local2_reference_audit = next_recorded_reference_goal(
                context["reference"], q_actual, args.local_horizon_s
            )
            result["local2_reference_goal"] = local2_reference_audit
            preplanned = context.get("rolling_preplan")
            use_preplanned = bool(
                isinstance(preplanned, dict)
                and preplanned.get("ready", False)
                and preplanned.get("candidate") is not None
                and preplanned.get("artifacts") is not None
            )
            if use_preplanned:
                candidate = preplanned["candidate"]
                artifacts = preplanned["artifacts"]
                result["rolling_preplan_used"] = True
                result["rolling_preplan_source_state_seq"] = preplanned.get("source_state_seq")
            if live.ACTIVE_BASE_FAST_REPAIR is None:
                raise RuntimeError("validated base Fast implementation is unavailable")
            if not use_preplanned:
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
                robust_target_is_diagnostic=True,
                allow_unestablished_side_fallback=bool(
                    (context.get("execution_summary") or {}).get("status")
                    == COMMAND_TIME_REPLAN_STATUS
                    and not (context.get("execution_summary") or {}).get("robot_commanded", False)
                ),
                )
                result["rolling_preplan_used"] = False
            result["local2_candidate"] = candidate
            result[f"local_{local_index}_candidate"] = candidate
            if not candidate.get("local_repair_ready", False):
                failed_replans += 1
                result["failed_replans"] = failed_replans
                result["status"] = "HOLD_UNSAFE_APPROACHING_OPERATOR_INTERVENTION_REQUIRED"
                trial.write_json(trial_dir / "event_replan_summary.json", result)
                return result

            worker = context.get("persistent_worker")
            authorization_snapshot = None
            if worker is not None:
                authorization_snapshot = worker.snapshot()
                fresh4, frames4, points4, geometry4 = fresh_from_persistent_snapshot(authorization_snapshot)
            else:
                fresh4, frames4, points4, geometry4 = capture_next_fresh(
                    args, processor, state_reader, denoiser, monitor1["fresh"]
                )
            trial.write_json(local2_dir / "fresh4_recheck.json", {"result": fresh4, "frames": frames4})
            if points4 is not None:
                np.save(local2_dir / "fresh4_cluster_points.npy", points4)
            if geometry4 is not None:
                trial.write_json(local2_dir / "fresh4_multisphere.json", geometry4)
            if geometry4 is None:
                failed_replans += 1
                result["failed_replans"] = failed_replans
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
            result[f"local_{local_index}_authorization"] = authorization
            if not authorization.get("local_execution_authorized", False):
                authorization_class = classify_local_authorization(authorization)
                result[f"local_{local_index}_authorization_classification"] = authorization_class
                if authorization_class["distance_only"]:
                    failed_replans += 1
                    result["failed_replans"] = failed_replans
                    result["status"] = "LOCAL_FRESH_DISTANCE_REJECTED_REPLAN_REQUIRED"
                    result.setdefault("rolling_events", []).append({
                        "type": "LOCAL_FRESH_DISTANCE_ONLY_REJECTION",
                        "local_index": int(local_index),
                        "verification_min_distance_m": authorization_class.get("verification_min_distance_m"),
                        "robot_commanded": False,
                        "action": "HOLD_AND_REPLAN_FROM_LATEST_FRESH",
                    })
                    if time.monotonic() - replan_started > float(event_args.max_continuous_replan_s):
                        result["status"] = "CONTINUOUS_REPLAN_WATCHDOG_TIMEOUT_OPERATOR_INTERVENTION_REQUIRED"
                        trial.write_json(trial_dir / "event_replan_summary.json", result)
                        return result
                    if worker is None:
                        result["status"] = "LOCAL_FRESH_REPLAN_TRACKER_UNAVAILABLE_OPERATOR_INTERVENTION_REQUIRED"
                        trial.write_json(trial_dir / "event_replan_summary.json", result)
                        return result
                    # Replan immediately from the rejecting Fresh snapshot;
                    # waiting for another frame only lets a moving obstacle
                    # approach farther before the same-local retry.
                    latest_snapshot = authorization_snapshot
                    if latest_snapshot is None:
                        result["status"] = "LOCAL_FRESH_REPLAN_SNAPSHOT_UNAVAILABLE"
                        trial.write_json(trial_dir / "event_replan_summary.json", result)
                        return result
                    fresh_retry, frames_retry, _, geometry_retry = fresh_from_persistent_snapshot(latest_snapshot)
                    if not fresh_retry.get("accepted", False) or geometry_retry is None:
                        result["status"] = "LOCAL_FRESH_REPLAN_STATE_NOT_READY_OPERATOR_INTERVENTION_REQUIRED"
                        trial.write_json(trial_dir / "event_replan_summary.json", result)
                        return result
                    next_context = dict(context)
                    next_context.update({
                        "local_artifacts": current_local_artifacts,
                        "fresh3": fresh_retry,
                        "fresh3_frames": frames_retry,
                        "fresh3_geometry": geometry_retry,
                        "fresh3_guard_distance": float(latest_snapshot.get("raw_guard_distance_m", float("-inf"))),
                        "execution_summary": current_execution_summary,
                        "local1_interrupted": current_local_interrupted,
                        "replan_depth": replan_depth,
                        "replan_started_monotonic": replan_started,
                        "failed_replans": failed_replans,
                        "force_goal_directed_local": False,
                    })
                    next_result = handler(**next_context)
                    result["fresh_distance_replan"] = next_result
                    result["status"] = next_result.get("status", "LOCAL_FRESH_DISTANCE_REPLAN_CONTINUED")
                    trial.write_json(trial_dir / "event_replan_summary.json", result)
                    return result
                failed_replans += 1
                result["failed_replans"] = failed_replans
                result["status"] = "LOCAL_FRESH_NON_DISTANCE_REJECTED_OPERATOR_INTERVENTION_REQUIRED"
                trial.write_json(trial_dir / "event_replan_summary.json", result)
                return result
            if worker is None:
                result["status"] = "LOCAL_CONTINUATION_TRACKER_UNAVAILABLE_OPERATOR_INTERVENTION_REQUIRED"
                trial.write_json(trial_dir / "event_replan_summary.json", result)
                return result
            local_monitor = make_mid_execution_monitor(
                authorized_csv=Path(authorization["authorized_trajectory_csv"]),
                robot=robot,
                worker=worker,
                processor=processor,
                state_reader=state_reader,
                denoiser=denoiser,
                args=args,
                stage4_config=config,
                stage4_model=model,
                trial_dir=local2_dir,
                root_trial_dir=trial_dir,
                rolling_continuation=True,
                reference=context.get("reference"),
                risk_links=context.get("risk_links"),
                local_artifacts=artifacts,
                event_local_index=local_index,
            )
            execution = trial.execute_authorized_trajectory_offline_track(
                # Continuation segments use the same prearm and predictive
                # monitor as local #1; no local segment is exempt from the
                # persistent-state safety barrier.
                robot,
                Path(authorization["authorized_trajectory_csv"]),
                args,
                processor=processor,
                denoiser=denoiser,
                playback_duration_s=None,
                execution_label=f"Fresh-authorized event local repair #{local_index}",
                motion_monitor_provider=local_monitor,
            )
            next_rolling_preplan = None
            take_preplan = getattr(local_monitor, "take_rolling_preplan", None)
            if callable(take_preplan):
                next_rolling_preplan = take_preplan(wait_s=0.05)
            if next_rolling_preplan is not None:
                execution["rolling_preplan"] = {
                    "ready": bool(next_rolling_preplan.get("ready", False)),
                    "source_state_seq": next_rolling_preplan.get("source_state_seq"),
                    "trigger_elapsed_s": next_rolling_preplan.get("trigger_elapsed_s"),
                    "planning_wall_ms": next_rolling_preplan.get("planning_wall_ms"),
                    "candidate": next_rolling_preplan.get("candidate"),
                    "artifacts": next_rolling_preplan.get("artifacts"),
                }
            result["local2_execution"] = execution
            result[f"local_{local_index}_execution"] = execution
            execution_status = execution.get("status")
            stop_info = classify_monitor_stop(execution)
            completed_execution = execution_status == "COMPLETED_AUTHORIZED_TRAJECTORY_EXECUTION"
            rolling_interruption = bool(stop_info["rolling_replan_stop"])
            current_local_artifacts = artifacts
            current_execution_summary = execution
            current_local_interrupted = bool(rolling_interruption)
            completed_local_index = local_index
            result["local2_execution_interrupted_by_monitor"] = bool(stop_info["monitor_stopped"])
            result[f"local_{local_index}_execution_interrupted_by_monitor"] = bool(
                stop_info["monitor_stopped"]
            )
            result[f"local_{local_index}_monitor_stop_reason"] = stop_info["reason"]
            result[f"local_{local_index}_rolling_replan_stop"] = rolling_interruption
            if not completed_execution and not rolling_interruption:
                result["status"] = "LOCAL_CONTINUATION_EXECUTION_FAILED_OPERATOR_INTERVENTION_REQUIRED"
                result["execution_failure_reason"] = stop_info["reason"]
                trial.write_json(trial_dir / "event_replan_summary.json", result)
                return result
            q_terminal_start = np.asarray(robot.get_joint(), dtype=np.float64)
            latest5_snapshot = None
            if worker is not None:
                latest5_snapshot = worker.snapshot()
                fresh5, frames5, points5, geometry5 = fresh_from_persistent_snapshot(
                    latest5_snapshot
                )
            else:
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
                output_dir=trial_dir / f"post_local{local_index}_monitor",
                max_wall_s=float(event_args.post_local_monitor_max_s),
            )
            tail_monitor = monitor2
            result["post_local2_monitor"] = {
                key: value for key, value in monitor2.items() if key not in {"forecast", "geometry"}
            }
            result[f"post_local{local_index}_monitor"] = result["post_local2_monitor"]
            if monitor2["status"] == "REPLAN_REQUIRED":
                # A second (or later) local segment is not a terminal state.
                # Re-enter the same event handler with the measured tail and
                # latest persistent state.  Only the watchdogs above can stop
                # this rolling loop; local numbering is diagnostic metadata.
                if worker is None or latest5_snapshot is None:
                    result["status"] = "ROLLING_REPLAN_TRACKER_UNAVAILABLE_OPERATOR_INTERVENTION_REQUIRED"
                    trial.write_json(trial_dir / "event_replan_summary.json", result)
                    return result
                raw_guard5 = raw_guard_from_persistent_snapshot(latest5_snapshot)
                if raw_guard5 <= float(args.guided_hard_stop_m):
                    result["status"] = "ROLLING_REPLAN_RAW_GUARD_NOT_SAFE_OPERATOR_INTERVENTION_REQUIRED"
                    result["rolling_replan_raw_guard_distance_m"] = raw_guard5
                    trial.write_json(trial_dir / "event_replan_summary.json", result)
                    return result
                result[f"post_local{local_index}_rolling_snapshot"] = {
                    "state_seq": int(v3._state_seq(latest5_snapshot)),
                    "raw_guard_distance_m": raw_guard5,
                    "fresh_state_seq": int(fresh5.get("state_seq", -1)),
                    "fresh_accepted": bool(fresh5.get("accepted", False)),
                    "geometry_ready": geometry5 is not None,
                }
                next_context = dict(context)
                next_context.update(
                    {
                        "local_artifacts": artifacts,
                        "fresh3": fresh5,
                        "fresh3_frames": frames5,
                        "fresh3_geometry": geometry5,
                        "fresh3_guard_distance": raw_guard5,
                        "execution_summary": execution,
                        "local1_interrupted": bool(rolling_interruption),
                        "replan_depth": replan_depth + 1,
                        "replan_started_monotonic": replan_started,
                        "failed_replans": failed_replans,
                        "rolling_preplan": execution.get("rolling_preplan"),
                    }
                )
                next_result = handler(**next_context)
                result["next_replan"] = next_result
                result["status"] = next_result.get("status", "ROLLING_REPLAN_CONTINUED")
                trial.write_json(trial_dir / "event_replan_summary.json", result)
                return result
            if monitor2["status"] == "PHYSICAL_HOLD_SAFE_MONITORING_TIMEOUT":
                result["status"] = "PHYSICAL_HOLD_SAFE_MONITORING_LIMIT_OPERATOR_CONTROL_REQUIRED"
                trial.write_json(trial_dir / "event_replan_summary.json", result)
                return result
            if monitor2["status"] not in {"STRICT_SCENE_CLEAR", "PREDICTED_RISK_CLEAR"}:
                result["status"] = "POST_LOCAL2_STATE_UNCERTAIN_OPERATOR_INTERVENTION_REQUIRED"
                trial.write_json(trial_dir / "event_replan_summary.json", result)
                return result
            forecast = monitor2["forecast"]

        if monitor1["status"] not in {"STRICT_SCENE_CLEAR", "PREDICTED_RISK_CLEAR"} and "post_local2_monitor" not in result:
            raise RuntimeError(f"unexpected post-local decision: {monitor1['status']}")
        stationary_confirmation = None
        if bool(getattr(event_args, "stationary_terminal_full_plan", False)):
            stationary_confirmation, stationary_snapshot = wait_for_confirmed_stationary_snapshot(
                context.get("persistent_worker"), args
            )
            result["stationary_confirmation"] = stationary_confirmation
            if not stationary_confirmation.get("confirmed", False):
                result["status"] = "STATIONARY_CONFIRMATION_REQUIRED_HOLD"
                trial.write_json(trial_dir / "event_replan_summary.json", result)
                return result
            fresh_latest, frames_latest, _, geometry_latest = fresh_from_persistent_snapshot(stationary_snapshot)
            monitor1["fresh"] = fresh_latest
            monitor1["frames"] = frames_latest
            monitor1["geometry"] = geometry_latest
            monitor1["forecast"], _ = forecast_from_fresh(args, fresh_latest, geometry_latest, frames_latest)
            forecast = monitor1["forecast"]
        terminal_dir = trial_dir / "terminal_goal_authorization"
        terminal, terminal_trajectory = authorize_terminal_goal(
            args,
            config,
            model,
            q_terminal_start,
            q_goal,
            forecast,
            terminal_durations,
            terminal_dir,
            stationary_geometry=(
                (monitor2.get("geometry", monitor1.get("geometry")) if "post_local2_monitor" in result else monitor1.get("geometry"))
                if bool(getattr(event_args, "stationary_terminal_full_plan", False)) else None
            ),
            use_stationary_full_plan=bool(getattr(event_args, "stationary_terminal_full_plan", False)),
        )
        result["terminal_authorization"] = terminal
        result["stationary_terminal_full_plan_requested"] = bool(
            getattr(event_args, "stationary_terminal_full_plan", False)
        )
        result["stationary_terminal_full_plan_core"] = bool(
            getattr(args, "stationary_terminal_full_plan", False)
        )
        result["terminal_planner_requested"] = (
            "full_static_ccro_nubs"
            if bool(getattr(event_args, "stationary_terminal_full_plan", False))
            else "linear_terminal_fallback"
        )
        if bool(getattr(event_args, "stationary_terminal_full_plan", False)):
            result["terminal_planner_mode"] = terminal.get("planner_mode")
        terminal_classification = classify_terminal_authorization(terminal)
        result["terminal_authorization_classification"] = terminal_classification
        if not terminal.get("authorized", False):
            if bool(getattr(event_args, "stationary_terminal_full_plan", False)):
                result["status"] = "STATIONARY_FULL_CCRO_HOLD"
                trial.write_json(trial_dir / "event_replan_summary.json", result)
                return result
            if can_continue_local_after_terminal_block(tail_monitor, terminal):
                worker = context.get("persistent_worker")
                if worker is None:
                    result["status"] = "TERMINAL_PATH_BLOCKED_TRACKER_UNAVAILABLE_OPERATOR_INTERVENTION_REQUIRED"
                    trial.write_json(trial_dir / "event_replan_summary.json", result)
                    return result
                stationary_recovery, latest = wait_for_stationary_safe_recovery_state(
                    worker, args, timeout_s=float(args.prediction_horizon_s)
                )
                result["terminal_blocked_stationary_guard_recovery"] = stationary_recovery
                if not stationary_recovery.get("ready", False) or latest is None:
                    result["status"] = "TERMINAL_PATH_BLOCKED_STATIONARY_GUARD_NOT_RECOVERED_OPERATOR_INTERVENTION_REQUIRED"
                    trial.write_json(trial_dir / "event_replan_summary.json", result)
                    return result
                raw_guard_latest = float(latest.get("raw_guard_distance_m", float("-inf")))
                fresh_latest, frames_latest, _points_latest, geometry_latest = fresh_from_persistent_snapshot(latest)
                if not fresh_latest.get("accepted", False) or geometry_latest is None:
                    result["status"] = "TERMINAL_PATH_BLOCKED_LATEST_GEOMETRY_NOT_READY_OPERATOR_INTERVENTION_REQUIRED"
                    trial.write_json(trial_dir / "event_replan_summary.json", result)
                    return result
                result["terminal_path_blocked_replan_requested"] = True
                result["terminal_path_blocked_from_local_index"] = int(completed_local_index)
                result["terminal_blocked_latest_raw_guard_m"] = raw_guard_latest
                next_context = dict(context)
                next_context.update(
                    {
                        "local_artifacts": current_local_artifacts,
                        "fresh3": fresh_latest,
                        "fresh3_frames": frames_latest,
                        "fresh3_geometry": geometry_latest,
                        "fresh3_guard_distance": raw_guard_latest,
                        "execution_summary": current_execution_summary,
                        "local1_interrupted": current_local_interrupted,
                        "replan_depth": int(completed_local_index - 1),
                        "replan_started_monotonic": replan_started,
                        "failed_replans": failed_replans,
                        "force_goal_directed_local": True,
                    }
                )
                next_result = handler(**next_context)
                result["terminal_path_blocked_replan"] = next_result
                result["status"] = next_result.get("status", "TERMINAL_PATH_BLOCKED_LOCAL_CONTINUATION")
                trial.write_json(trial_dir / "event_replan_summary.json", result)
                return result
            result["status"] = "TERMINAL_AUTHORIZATION_FAILED_OPERATOR_INTERVENTION_REQUIRED"
            trial.write_json(trial_dir / "event_replan_summary.json", result)
            return result
        if terminal_trajectory is None:
            result["status"] = "TERMINAL_AUTHORIZED_TRAJECTORY_MISSING_OPERATOR_INTERVENTION_REQUIRED"
            trial.write_json(trial_dir / "event_replan_summary.json", result)
            return result

        # One last raw observation is intentionally not replaced by an old
        # candidate forecast: the guarded executor samples raw distance three
        # times immediately before commanding this current-run terminal CSV.
        terminal_worker = context.get("persistent_worker")
        if terminal_worker is None:
            result["status"] = "TERMINAL_TRACKER_UNAVAILABLE_OPERATOR_INTERVENTION_REQUIRED"
            trial.write_json(trial_dir / "event_replan_summary.json", result)
            return result
        terminal_monitor = make_mid_execution_monitor(
            authorized_csv=Path(terminal["authorized_trajectory_csv"]),
            trajectory=terminal_trajectory,
            robot=robot,
            worker=terminal_worker,
            processor=processor,
            state_reader=state_reader,
            denoiser=denoiser,
            args=args,
            stage4_config=config,
            stage4_model=model,
            trial_dir=terminal_dir,
        )
        terminal_execution = trial.execute_authorized_trajectory_offline_track(
            robot,
            Path(terminal["authorized_trajectory_csv"]),
            args,
            processor=processor,
            denoiser=denoiser,
            playback_duration_s=None,
            execution_label="event-replan terminal NUBS to preset goal",
            motion_monitor_provider=terminal_monitor,
        )
        result["terminal_execution"] = terminal_execution
        terminal_stop_info = classify_monitor_stop(terminal_execution)
        terminal_completed = terminal_execution.get("status") == "COMPLETED_AUTHORIZED_TRAJECTORY_EXECUTION"
        terminal_rolling_interruption = bool(terminal_stop_info["rolling_replan_stop"])
        if terminal_rolling_interruption:
            # Terminal risk re-entry is not a terminal failure: continue from
            # the measured interrupted pose using the same rolling local loop.
            q_interrupted = np.asarray(robot.get_joint(), dtype=np.float64)
            latest = terminal_worker.snapshot()
            raw_guard_latest = raw_guard_from_persistent_snapshot(latest)
            if raw_guard_latest <= float(args.guided_hard_stop_m):
                result["status"] = "TERMINAL_RISK_REENTRY_RAW_GUARD_NOT_SAFE_OPERATOR_INTERVENTION_REQUIRED"
                result["terminal_risk_reentry_raw_guard_distance_m"] = raw_guard_latest
                trial.write_json(trial_dir / "event_replan_summary.json", result)
                return result
            fresh_latest, frames_latest, points_latest, geometry_latest = fresh_from_persistent_snapshot(latest)
            next_context = dict(context)
            next_context.update(
                {
                    "local_artifacts": {
                        "candidate_trajectory": terminal_trajectory,
                        # Preserve the previously established bypass anchor;
                        # q_actual is read again at the next handler entry.
                        "q_now": np.asarray(
                            context["local_artifacts"]["q_now"], dtype=np.float64
                        ).copy(),
                    },
                    "fresh3": fresh_latest,
                    "fresh3_frames": frames_latest,
                    "fresh3_geometry": geometry_latest,
                    "fresh3_guard_distance": raw_guard_latest,
                    "execution_summary": terminal_execution,
                    "local1_interrupted": True,
                    "replan_depth": replan_depth + 1,
                    "replan_started_monotonic": replan_started,
                    "failed_replans": failed_replans,
                }
            )
            next_result = handler(**next_context)
            result["terminal_risk_replan"] = next_result
            result["status"] = next_result.get("status", "TERMINAL_RISK_REPLAN_CONTINUED")
            trial.write_json(trial_dir / "event_replan_summary.json", result)
            return result
        if terminal_completed:
            result["status"] = "SIMPLE_DYNAMIC_NUBS_RECOVERED_AND_GOAL_REACHED"
            result["command_hold"] = False
        else:
            result["status"] = "TERMINAL_EXECUTION_FAILED_OPERATOR_INTERVENTION_REQUIRED"
            result["terminal_failure_reason"] = terminal_stop_info["reason"]
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
    # Post-STOP static settling has already been audited by the core.  The
    # command-time revalidation immediately before waypoint submission is the
    # final freshness barrier for a moving obstacle.
    live_args.candidate_pre_execute_settle_s = 0.0
    old_handler = trial.POST_LOCAL_FRESH3_HANDLER
    old_worker_factory = trial.PERSISTENT_OBSTACLE_WORKER_FACTORY
    old_mid_monitor = trial.MID_EXECUTION_MONITOR_FACTORY
    try:
        trial.POST_LOCAL_FRESH3_HANDLER = make_event_handler(args, durations)
        trial.PERSISTENT_OBSTACLE_WORKER_FACTORY = v3.make_persistent_perception_worker
        trial.MID_EXECUTION_MONITOR_FACTORY = make_mid_execution_monitor
        result = live.run(live_args)
    finally:
        trial.POST_LOCAL_FRESH3_HANDLER = old_handler
        trial.PERSISTENT_OBSTACLE_WORKER_FACTORY = old_worker_factory
        trial.MID_EXECUTION_MONITOR_FACTORY = old_mid_monitor

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
