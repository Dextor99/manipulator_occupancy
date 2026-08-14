#!/usr/bin/env python3
"""Frozen D2-COMPLETE-v1 opposing-oblique protected live experiment.

Only the physical obstacle corridor differs from the validated simple dynamic
NUBS pilot.  Detection, STRO, fixed two-sphere geometry, 0.11 m bypass gate,
Fast, Fresh authorization, 1 s execution and all safety thresholds remain
unchanged.  The obstacle follows an independent A-to-B path and continues out
of the ROI; it must never follow the robot's avoidance motion.
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
SCENE_PHRASE = "CCRO_653_D2_COMPLETE_V1_FIXED_CORRIDOR_CONFIRMED"
SCENE_PROTOCOL = {
    "scene_id": "D2_COMPLETE_V1_OPPOSING_OBLIQUE_XP10",
    "classification": "opposing_oblique",
    "robot_task_direction": "approximately base -Y",
    "obstacle_direction_unit_base": [0.906307787, 0.422618262, 0.0],
    "opposing_condition": "obstacle velocity dot robot task velocity < 0",
    "nominal_speed_m_s": 0.10,
    "offline_trigger_center_base_m": [0.60, -0.12, 0.3662453],
    "corridor_point_a_base_m": [0.464, -0.183, 0.3662453],
    "corridor_point_b_base_m": [0.827, -0.014, 0.3662453],
    "operator_rule": "move A toward B and continue out of ROI; never steer toward robot avoidance",
    "offline_scene_search_id": 1494,
    "offline_expected": {
        "trigger_current_m": 0.1374618194,
        "trigger_predicted_m": 0.1145028772,
        "coarse_m": 0.1258061946,
        "fast_m": 0.1345227926,
        "tail_hold_min_m": 0.1488079986,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = event.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        output=DEFAULT_OUTPUT,
        task_geometry_id=SCENE_PROTOCOL["scene_id"],
        post_local_monitor_max_s=6.0,
    )
    parser.add_argument("--scene-operator-phrase", default="")
    return parser


def validate_frozen_request(args: argparse.Namespace) -> None:
    if args.scene_operator_phrase != SCENE_PHRASE:
        raise RuntimeError(f"bad fixed-corridor phrase; required: {SCENE_PHRASE}")
    expected = {
        "forward_m": 0.05,
        "side_lengths_m": "0.04,0.06,0.08",
        "planning_robust_target_m": 0.11,
        "max_joint_delta_rad": 0.12,
        "continuation_side_m": 0.04,
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
