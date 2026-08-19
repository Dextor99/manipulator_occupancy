#!/usr/bin/env python3
"""Frozen D2-AH (dynamic approach -> hold) protected live experiment.

The obstacle moves mainly along base +Y, opposite the robot's base -Y task
motion, until a pre-marked physical stop line and then remains stationary.
The planner and all safety thresholds are inherited unchanged from the
event-replan implementation; this wrapper only freezes the scene protocol
and audits that dynamic approach preceded the hold.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

event = importlib.import_module(
    "experiments.new.6_5.6_5_3.run_653_simple_dynamic_nubs_event_replan_live"
)
trial = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")

DEFAULT_OUTPUT = ROOT / "results/new/6_5/6_5_3/d2_approach_hold_complete_live"
SCENE_PHRASE = "CCRO_653_D2_APPROACH_HOLD_V1_CONFIRMED"
SCENE_ID = "D2_APPROACH_HOLD_V1_FIXED_X_XP00"

# Calibrate in a non-execute shadow first.  Formal execution is blocked until
# this value is frozen and committed.
# Calibrated from D2-AH shadow r02 (observed center Y=-0.145657 m).
# The physical tape/stop mark must be placed at this value for formal runs.
# Offline r10 geometry scan: +0.020 m from the previous line gives the
# preset goal ~0.1056 m dense clearance while retaining a blocked approach.
# Operator guidance only; never an authorization gate.
RECOMMENDED_HOLD_CENTER_Y_M = -0.126
RECOMMENDED_HOLD_HALF_WIDTH_M = 0.01
HOLD_SPEED_THRESHOLD_M_S = 0.04
HOLD_CONFIRM_FRAMES = 3
HOLD_CENTER_SPAN_MAX_M = 0.02
HOLD_RAW_GUARD_MIN_M = 0.10

SCENE_PROTOCOL = {
    "scene_id": SCENE_ID,
    "classification": "dynamic_approach_then_hold",
    "robot_task_direction": "approximately base -Y",
    "obstacle_nominal_direction_unit_base": [0.0, 1.0, 0.0],
    "opposing_condition": "obstacle velocity dot robot task velocity < 0",
    "stro_trigger_policy": {
        "horizon_s": 1.2,
        "risk_distance_m": 0.14,
        "trigger_must_precede_hold": True,
        "trigger_requires_dynamic_track": True,
    },
    "approach_hold_policy": {
        "fixed_x_lane": True,
        "recommended_hold_center_y_m": RECOMMENDED_HOLD_CENTER_Y_M,
        "recommended_hold_half_width_m": RECOMMENDED_HOLD_HALF_WIDTH_M,
        "hold_position_is_authorization_gate": False,
        "hold_speed_threshold_m_s": HOLD_SPEED_THRESHOLD_M_S,
        "hold_confirm_frames": HOLD_CONFIRM_FRAMES,
        "hold_center_span_max_m": HOLD_CENTER_SPAN_MAX_M,
        "manual_adjustment_after_hold": False,
    },
    "candidate_acceptance_policy": {
        "online_clearance_m": 0.09,
        "clearance_improvement_preference_m": 0.003,
        "clearance_improvement_is_hard_gate": False,
        "accepted_steps_is_hard_gate": False,
        "candidate_delta_is_hard_gate": False,
    },
    "execution_safety_policy": {
        "raw_hard_guard_m": 0.10,
        "raw_hard_guard_debounce": False,
        "final_fresh_authorization": True,
        "terminal_full_verification": True,
    },
    "terminal_recovery_policy": {
        "stationary_hold_safe_is_goal_path_clear": False,
        "stationary_hold_switches_to_full_terminal_ccro": False,
        "stationary_hold_switches_to_fast_goal_directed": True,
        "command_time_stale_replans_from_latest_state": True,
        "terminal_distance_blocked_reenters_goal_directed_fast": True,
        "terminal_distance_blocked_uses_fast_terminal_bypass": True,
        "stationary_fast_terminal_is_single_complete_path": True,
        "stationary_terminal_full_ccro_enabled": False,
        "terminal_failure_is_fail_closed": True,
        "continuous_replan_watchdog_s": 10.0,
    },
    "operator_rule": (
        "move mainly along base +Y with approximately fixed X; after STRO, "
        "continue into the recommended hold region, stop the obstacle, and "
        "keep it completely stationary without further adjustment"
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = event.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        output=DEFAULT_OUTPUT,
        task_geometry_id=SCENE_ID,
        post_local_monitor_max_s=6.0,
        stro_trigger_horizon_s=1.2,
        # D2-AH live is intentionally Fast-only.  Full stationary CCRO is
        # retained for offline/static experiments, never this online path.
        stationary_terminal_full_plan=False,
        stationary_fast_goal_directed=True,
        command_time_fast_retry=True,
        stationary_fast_terminal_bypass=True,
        stationary_fast_terminal_duration_s=10.0,
        stationary_fast_terminal_segments=16,
        stationary_fast_terminal_rollout_steps=8,
        stationary_fast_terminal_target_ms=5000.0,
        stationary_fast_terminal_max_ms=7000.0,
        stationary_fast_terminal_virtual_max_joint_delta_rad=0.30,
        # Formal production route is deadline-driven; a positive value is
        # reserved for diagnostic replay step caps.
        stationary_fast_terminal_virtual_fast_steps=0,
        stationary_fast_terminal_samples_per_local=3,
        # route target is diagnostic; terminal max budget supplies the hard
        # route deadline while reserving time for finalization/verifier.
        stationary_fast_terminal_route_max_ms=0.0,
        stationary_virtual_topology_floor_m=0.08,
        stationary_center_span_m=HOLD_CENTER_SPAN_MAX_M,
        shadow_hold_observation_s=3.0,
        terminal_durations_s="6.0,8.0",
    )
    parser.add_argument("--scene-operator-phrase", default="")
    parser.add_argument(
        "--stationary-capture-only", action="store_true",
        help="stop after confirmed stationary terminal capture; never command terminal motion",
    )
    return parser


def validate_frozen_request(args: argparse.Namespace) -> None:
    if args.scene_operator_phrase != SCENE_PHRASE:
        raise RuntimeError(f"bad D2-AH operator phrase; required: {SCENE_PHRASE}")
    expected = {
        "forward_m": 0.05,
        "side_lengths_m": "0.04,0.06,0.08",
        "planning_robust_target_m": 0.11,
        "max_joint_delta_rad": 0.12,
        "continuation_side_m": 0.04,
        "stro_trigger_horizon_s": 1.2,
    }
    changed = {
        name: {"actual": getattr(args, name), "required": value}
        for name, value in expected.items()
        if getattr(args, name) != value
    }
    if changed:
        raise RuntimeError(f"D2-AH frozen planner parameters changed: {changed}")
    if args.task_geometry_id != SCENE_ID:
        raise RuntimeError("D2-AH task geometry id must not be overridden")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _event(events: list[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    return next((item for item in events if item.get("type") == event_type), None)


def audit_approach_hold(result: dict[str, Any], *, execute: bool) -> dict[str, Any]:
    """Audit dynamic trigger, dynamic planning and the physical hold window."""
    core_trial_dir = Path(result.get("core_trial_dir", ""))
    if not core_trial_dir.exists() or not (core_trial_dir / "summary.json").exists():
        return {"passed": False, "reason": "core_trial_dir_missing"}
    core_summary = _load_json(core_trial_dir / "summary.json")
    persistent_path = core_trial_dir / "persistent_perception" / "persistent_perception_audit.json"
    persistent = _load_json(persistent_path) if persistent_path.exists() else {}
    events = list(core_summary.get("events", []))
    trigger = _event(events, "TRIGGER")
    seed = _event(events, "PERSISTENT_TRACKER_SEED_READY")
    repair = _event(events, "LOCAL_REPAIR_READY")
    audit: dict[str, Any] = {
        "scene": "D2_APPROACH_HOLD",
        "trigger_found": trigger is not None,
        "recommended_hold_position_available": True,
    }
    if trigger is None or seed is None:
        audit.update({"passed": False, "reason": "dynamic_trigger_evidence_missing"})
        return audit
    predicted = float(trigger.get("predicted_distance_m", float("inf")))
    current = float(trigger.get("current_distance_m", float("nan")))
    raw = float(trigger.get("guard_distance_m", float("nan")))
    speed = float(trigger.get("window_speed_m_s", 0.0))
    dynamic_ok = predicted < 0.14 and current > 0.12 and raw > 0.10 and speed > 0.03
    audit["dynamic_trigger"] = {
        "passed": dynamic_ok,
        "predicted_distance_m": predicted,
        "current_distance_m": current,
        "raw_guard_distance_m": raw,
        "window_speed_m_s": speed,
    }
    planning_speed = 0.0
    if repair:
        candidate = repair.get("candidate", {})
        planning_speed = float(np.linalg.norm(np.asarray(candidate.get("obstacle_velocity", [0.0] * 3), dtype=float)))
    audit["local1_dynamic_planning"] = {
        "passed": planning_speed > 0.03,
        "obstacle_speed_m_s": planning_speed,
    }
    trigger_ts = float(seed.get("recheck", {}).get("last_timestamp", -1.0))
    updates = [
        item for item in persistent.get("updates", [])
        if item.get("associated", False)
        and item.get("geometry_covered", False)
        and float(item.get("timestamp", -1.0)) >= trigger_ts
    ]
    samples: list[dict[str, Any]] = []
    for prev, curr in zip(updates[:-1], updates[1:]):
        dt = float(curr.get("timestamp", 0.0)) - float(prev.get("timestamp", 0.0))
        if dt <= 1e-6:
            continue
        p0 = np.asarray(prev.get("center", [0.0] * 3), dtype=float)
        p1 = np.asarray(curr.get("center", [0.0] * 3), dtype=float)
        samples.append({
            "timestamp": float(curr["timestamp"]),
            "center_m": p1.tolist(),
            "center_y_m": float(p1[1]),
            "speed_m_s": float(np.linalg.norm((p1 - p0) / dt)),
            "raw_guard_distance_m": float(curr.get("raw_guard_distance_m", float("nan"))),
        })
    hold_start: int | None = None
    for start in range(max(0, len(samples) - HOLD_CONFIRM_FRAMES + 1)):
        window = samples[start : start + HOLD_CONFIRM_FRAMES]
        if len(window) < HOLD_CONFIRM_FRAMES:
            break
        speed_ok = all(x["speed_m_s"] <= HOLD_SPEED_THRESHOLD_M_S for x in window)
        raw_ok = all(x["raw_guard_distance_m"] > HOLD_RAW_GUARD_MIN_M for x in window)
        centers = np.asarray([x["center_m"] for x in window], dtype=float)
        center_span = float(np.max(np.linalg.norm(centers[:, None] - centers[None, :], axis=2)))
        if speed_ok and raw_ok and center_span <= HOLD_CENTER_SPAN_MAX_M:
            hold_start = start
            break
    hold_confirmed = hold_start is not None
    hold_window = samples[hold_start : hold_start + HOLD_CONFIRM_FRAMES] if hold_confirmed else []
    hold_ts = hold_window[0]["timestamp"] if hold_window else None
    audit["hold_confirmation"] = {
        "confirmed": hold_confirmed,
        "recommended_center_y_m": RECOMMENDED_HOLD_CENTER_Y_M,
        "recommended_half_width_m": RECOMMENDED_HOLD_HALF_WIDTH_M,
        "is_authorization_gate": False,
        "speed_threshold_m_s": HOLD_SPEED_THRESHOLD_M_S,
        "required_frames": HOLD_CONFIRM_FRAMES,
        "center_span_max_m": HOLD_CENTER_SPAN_MAX_M,
        "observed_center_y_m": float(np.median([x["center_y_m"] for x in hold_window])) if hold_window else None,
        "within_recommended_band": bool(hold_window and abs(float(np.median([x["center_y_m"] for x in hold_window])) - RECOMMENDED_HOLD_CENTER_Y_M) <= RECOMMENDED_HOLD_HALF_WIDTH_M),
        "minimum_raw_guard_m": min((x["raw_guard_distance_m"] for x in hold_window), default=None),
        "hold_timestamp": hold_ts,
    }
    trigger_before_hold = bool(hold_confirmed and hold_ts is not None and hold_ts > trigger_ts)
    audit["trigger_before_hold"] = trigger_before_hold
    event_summary = _load_json(core_trial_dir / "event_replan_summary.json") if (core_trial_dir / "event_replan_summary.json").exists() else {}
    goal_reached = event_summary.get("status") == "SIMPLE_DYNAMIC_NUBS_RECOVERED_AND_GOAL_REACHED"
    audit["goal_reached"] = goal_reached
    protocol_pass = bool(dynamic_ok and planning_speed > 0.03 and hold_confirmed and trigger_before_hold)
    audit["protocol_pass"] = protocol_pass
    audit["full_execute_pass"] = bool(protocol_pass and goal_reached) if execute else None
    audit["passed"] = audit["full_execute_pass"] if execute else protocol_pass
    return audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_frozen_request(args)
    output = args.output.resolve() / f"r{args.repeat:02d}"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite existing D2-AH result directory: {output}; choose a new --repeat")
    print("\n[D2-AH PROTOCOL]\n1. Move obstacle mainly +Y with fixed X.\n2. STRO must trigger while moving.\n3. Continue to the pre-marked stop line.\n4. Hold completely still; do not adjust again.\n")
    result = event.run(args)
    result["scene_protocol"] = SCENE_PROTOCOL
    result["scene_operator_phrase"] = SCENE_PHRASE
    output = Path(result["output"])
    audit = audit_approach_hold(result, execute=bool(args.execute))
    result["approach_hold_audit"] = audit
    trial.write_json(output / "d2_approach_hold_scene_protocol.json", SCENE_PROTOCOL)
    trial.write_json(output / "d2_approach_hold_audit.json", audit)
    trial.write_json(output / "summary.json", result)
    return result


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
