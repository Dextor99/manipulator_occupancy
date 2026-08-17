#!/usr/bin/env python3
"""Frozen D2-COMPLETE-v2 fixed-X opposing protected live experiment.

Only the physical obstacle corridor differs from the validated simple dynamic
NUBS pilot.  Detection, STRO, fixed two-sphere geometry, 0.11 m bypass gate,
Fast, Fresh authorization, 1 s execution and all safety thresholds remain
unchanged.  The obstacle moves mainly along base +Y, opposite to the robot's
base -Y task motion, while remaining in one fixed-X lane.  Small transverse
and speed variations from manual operation are diagnostics, not validity
gates.  The obstacle must never follow the robot's avoidance motion.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

event = importlib.import_module(
    "experiments.new.6_5.6_5_3.run_653_simple_dynamic_nubs_event_replan_live"
)
trial = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")

DEFAULT_OUTPUT = ROOT / "results/new/6_5/6_5_3/simple_dynamic_nubs_complete_live"
SCENE_PHRASE = "CCRO_653_D2_COMPLETE_V2_FIXED_X_OPPOSING_CONFIRMED"
SCENE_PROTOCOL = {
    "scene_id": "D2_COMPLETE_V2_OPPOSING_FIXED_X_XP00",
    "classification": "opposing_approach",
    "robot_task_direction": "approximately base -Y",
    "obstacle_nominal_direction_unit_base": [0.0, 1.0, 0.0],
    "opposing_condition": "obstacle velocity dot robot task velocity < 0",
    "direction_is_diagnostic_not_a_planning_gate": True,
    "stro_trigger_policy": {
        "horizon_s": 1.2,
        "risk_distance_m": 0.14,
        "semantic": (
            "early-warning trigger only; execution prediction remains 0.5 s"
        ),
    },
    "offline_expected_source_trigger_horizon_s": 0.5,
    "candidate_acceptance_policy": {
        "online_clearance_m": 0.09,
        "clearance_improvement_preference_m": 0.003,
        "clearance_improvement_is_hard_gate": False,
        "accepted_steps_is_hard_gate": False,
        "candidate_delta_is_hard_gate": False,
        "semantic": (
            "absolute verifier governs execution; Fast improvement metrics "
            "are diagnostics only"
        ),
    },
    "tabletop_bypass_policy": {
        "gripper_base_min_z_m": 0.46,
        "tabletop_parallel_side": True,
        "preserve_tcp_height_linearized": True,
        "coarse_seed_tabletop_gate": True,
        "fast_output_tabletop_gate": True,
        "terminal_goal_tabletop_gate": True,
        "final_authorization_guard_remains_independent": True,
    },
    "terminal_recovery_policy": {
        "stationary_hold_safe_is_goal_path_clear": False,
        "direct_terminal_distance_blocked_reenters_local": True,
        "terminal_non_distance_failure_is_fail_closed": True,
        "continuous_replan_watchdog_s": 10.0,
    },
    "nominal_speed_m_s": 0.10,
    "fixed_x_lane_m": 0.7749155588,
    "fixed_x_lane_tolerance_m": 0.025,
    "fixed_x_lane_source": "d2_fixed_x_opposing_search_r02",
    "offline_expected": {
        "speed_y_m_s": 0.08,
        "trigger_current_m": 0.1760151840,
        "trigger_predicted_m": 0.1331116060,
        "coarse_m": 0.1250675599,
        "fast_m": 0.1338093914,
        "fast_ms": 48.0826940,
    },
    "operator_rule": (
        "keep X approximately constant; move mainly from base -Y toward +Y "
        "and continue past the encounter; never steer toward robot avoidance"
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = event.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        output=DEFAULT_OUTPUT,
        task_geometry_id=SCENE_PROTOCOL["scene_id"],
        post_local_monitor_max_s=6.0,
        stro_trigger_horizon_s=1.2,
    )
    parser.add_argument("--scene-operator-phrase", default="")
    return parser


def validate_frozen_request(args: argparse.Namespace) -> None:
    if args.scene_operator_phrase != SCENE_PHRASE:
        raise RuntimeError(f"bad fixed-X opposing phrase; required: {SCENE_PHRASE}")
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
        raise RuntimeError(f"D2-COMPLETE-v1 frozen planner parameters changed: {changed}")
    if args.task_geometry_id != SCENE_PROTOCOL["scene_id"]:
        raise RuntimeError("D2-COMPLETE-v1 task geometry id must not be overridden")


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_frozen_request(args)
    output = args.output.resolve() / f"r{args.repeat:02d}"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"refusing to overwrite existing D2 result directory: {output}; "
            "choose a new --repeat value"
        )
    result = event.run(args)
    result["scene_protocol"] = SCENE_PROTOCOL
    result["scene_operator_phrase"] = SCENE_PHRASE
    output = Path(result["output"])
    trial.write_json(output / "d2_complete_scene_protocol.json", SCENE_PROTOCOL)
    trial.write_json(output / "summary.json", result)
    return result


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
