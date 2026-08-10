#!/usr/bin/env python3
"""Move AUBO to the start joint pose of a 6.5.3 candidate preview package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from robot.linear_move_debug import fmt_joints, load_robot_module  # noqa: E402


DEFAULT_PLAN_DIR = (
    ROOT
    / "results"
    / "new"
    / "6_5"
    / "6_5_3"
    / "dynamic_repair_pilot"
    / "trials"
    / "D1_crossing_body_r14"
    / "candidate_preview_package"
)
DEFAULT_OUTPUT = ROOT / "results" / "new" / "6_5" / "6_5_3" / "candidate_preview_execution" / "move_to_start"
REQUIRED_OPERATOR_PHRASE = "CCRO_653_MOVE_TO_CANDIDATE_START_APPROVED"


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def joint_error(actual: np.ndarray, target: np.ndarray) -> dict[str, float]:
    diff = actual - target
    return {"l2_rad": float(np.linalg.norm(diff)), "max_abs_rad": float(np.max(np.abs(diff)))}


def wait_for_joints(robot: Any, target: np.ndarray, timeout_s: float, poll_s: float, tolerance_rad: float) -> dict[str, Any]:
    started = time.perf_counter()
    samples = []
    while time.perf_counter() - started <= timeout_s:
        actual = np.asarray(robot.get_joint(), dtype=np.float64)
        err = joint_error(actual, target)
        samples.append({"t_s": time.perf_counter() - started, "actual_joint_rad": actual.tolist(), **err})
        if err["max_abs_rad"] <= tolerance_rad:
            return {"reached": True, "elapsed_s": time.perf_counter() - started, "actual_joint_rad": actual.tolist(), "error": err, "samples": samples}
        time.sleep(poll_s)
    actual = np.asarray(robot.get_joint(), dtype=np.float64)
    return {"reached": False, "elapsed_s": time.perf_counter() - started, "actual_joint_rad": actual.tolist(), "error": joint_error(actual, target), "samples": samples}


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan_dir = args.plan_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((plan_dir / "summary.json").read_text(encoding="utf-8"))
    q_start = np.asarray(summary["q_start_rad"], dtype=np.float64)
    log: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "robot_commanded": False,
        "plan_dir": str(plan_dir),
        "target_q_start_rad": q_start.tolist(),
        "target_q_start_deg": np.rad2deg(q_start).tolist(),
        "required_operator_phrase": REQUIRED_OPERATOR_PHRASE,
        "operator_phrase_ok": args.operator_phrase == REQUIRED_OPERATOR_PHRASE,
        "parameters": vars(args),
    }
    if not args.execute:
        log["status"] = "DRY_RUN_NO_ROBOT_COMMAND"
        write_json(output / "move_to_candidate_start_log.json", log)
        print(json.dumps(log, indent=2, ensure_ascii=False, default=json_default))
        return log
    if args.operator_phrase != REQUIRED_OPERATOR_PHRASE:
        log["status"] = "BLOCKED_BAD_OPERATOR_PHRASE"
        write_json(output / "move_to_candidate_start_log.json", log)
        raise RuntimeError(f"bad operator phrase; required: {REQUIRED_OPERATOR_PHRASE}")

    robot, so_path = load_robot_module(args.sdk_dir)
    log["sdk_so"] = str(so_path)
    print(f"[sdk] loaded: {so_path}")
    try:
        if not robot.init():
            raise RuntimeError("robot.init() failed")
        actual = np.asarray(robot.get_joint(), dtype=np.float64)
        log["actual_start_rad"] = actual.tolist()
        log["pre_move_error"] = joint_error(actual, q_start)
        print(f"[current] {fmt_joints(actual.tolist())}")
        print(f"[target ] {fmt_joints(q_start.tolist())}")
        print("")
        input("Step 1/1: movej to dynamic candidate start. Keep emergency stop ready. Press Enter to continue; Ctrl-C abort.")
        robot.movej(q_start.tolist())
        reached = wait_for_joints(robot, q_start, args.motion_timeout_s, args.poll_s, args.joint_tolerance_rad)
        log["robot_commanded"] = True
        log["reach_check"] = reached
        if not reached["reached"]:
            raise RuntimeError(f"candidate start not reached: {reached['error']}")
        log["status"] = "REACHED_CANDIDATE_START"
    except Exception as exc:
        log["status"] = "ABORTED_OR_FAILED"
        log["error"] = str(exc)
        raise
    finally:
        try:
            robot.log_out()
        except Exception:
            pass
        write_json(output / "move_to_candidate_start_log.json", log)
        print(f"[log] {output / 'move_to_candidate_start_log.json'}")
    return log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sdk-dir", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--operator-phrase", default="")
    parser.add_argument("--joint-tolerance-rad", type=float, default=0.035)
    parser.add_argument("--motion-timeout-s", type=float, default=45.0)
    parser.add_argument("--poll-s", type=float, default=0.05)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
