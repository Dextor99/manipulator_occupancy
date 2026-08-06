#!/usr/bin/env python3
"""Guarded low-speed AUBO tabletop Y-stroke executor for 6.5.2.

Sequence when ``--execute`` is provided:

1. movej to the initial posture [0, 0, 90, 0, 90, 0] deg;
2. read the TCP pose at that posture;
3. move in Cartesian space to the start pose with X/Z/RPY fixed and Y=+0.4 m;
4. wait for operator confirmation;
5. optionally stop cleanly at the start pose for obstacle placement;
6. move to the goal pose with X/Z/RPY fixed and Y=-0.4 m;
6. wait for operator confirmation;
7. move back to the initial TCP pose, then movej to the initial joint posture.

Default mode is dry-run.  No robot command is sent unless both ``--execute`` and
the exact operator phrase are provided.  Each motion segment has an additional
interactive confirmation.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys
import termios
import time
import tty
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.linear_move_debug import fmt_joints, fmt_pose, load_robot_module  # noqa: E402


DEFAULT_OUTPUT = ROOT / "results" / "new" / "6_5" / "6_5_2" / "planar_y_guarded_execution"
REQUIRED_OPERATOR_PHRASE = "CCRO_652_PLANAR_Y_APPROVED"


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def parse_home_degrees(value: str) -> list[float]:
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 6:
        raise ValueError("--home-joints-deg must contain six comma-separated values")
    return parts


def make_pose(base_pose: list[float], *, y: float, x_offset: float = 0.0) -> list[float]:
    pose = list(base_pose)
    pose[0] = float(pose[0]) + float(x_offset)
    pose[1] = float(y)
    return pose


def require_confirmation(enabled: bool, message: str) -> None:
    if not enabled:
        print(f"[dry-run] would wait for Enter/Space: {message}")
        return
    print("")
    print(message)
    print("Press Enter or Space to continue; press q or Ctrl-C to abort.")
    if sys.stdin.isatty():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            print("")
        if ch not in {"\r", "\n", " "}:
            raise RuntimeError("operator aborted before motion")
    else:
        typed = input("Press Enter to continue, or type q then Enter to abort: ").strip().lower()
        if typed == "q":
            raise RuntimeError("operator aborted before motion")


def check_pose_limits(pose: list[float], args: argparse.Namespace, label: str) -> None:
    x, y, z = pose[:3]
    if not (args.min_x <= x <= args.max_x):
        raise RuntimeError(f"{label} X={x:.3f} outside [{args.min_x:.3f}, {args.max_x:.3f}]")
    if not (args.min_y <= y <= args.max_y):
        raise RuntimeError(f"{label} Y={y:.3f} outside [{args.min_y:.3f}, {args.max_y:.3f}]")
    if not (args.min_z <= z <= args.max_z):
        raise RuntimeError(f"{label} Z={z:.3f} outside [{args.min_z:.3f}, {args.max_z:.3f}]")


def cartesian_distance(a: list[float], b: list[float]) -> float:
    return float(np.linalg.norm(np.asarray(a[:3], dtype=np.float64) - np.asarray(b[:3], dtype=np.float64)))


def joint_distance(a: list[float], b: list[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def wait_for_pose(robot, target_pose: list[float], args: argparse.Namespace, label: str) -> dict[str, Any]:
    started = time.perf_counter()
    samples = []
    last = list(robot.get_status())
    while time.perf_counter() - started <= args.motion_timeout_s:
        last = list(robot.get_status())
        err = cartesian_distance(target_pose, last)
        samples.append({"t_s": time.perf_counter() - started, "position_error_m": err, "actual_pose": last})
        if err <= args.pose_tolerance_m:
            return {
                "reached": True,
                "label": label,
                "actual_pose": last,
                "position_error_m": err,
                "wait_s": time.perf_counter() - started,
                "sample_count": len(samples),
            }
        time.sleep(args.poll_s)
    return {
        "reached": False,
        "label": label,
        "actual_pose": last,
        "position_error_m": cartesian_distance(target_pose, last),
        "wait_s": time.perf_counter() - started,
        "sample_count": len(samples),
    }


def wait_for_joints(robot, target_joints: list[float], args: argparse.Namespace, label: str) -> dict[str, Any]:
    started = time.perf_counter()
    last = list(robot.get_joint())
    samples = 0
    while time.perf_counter() - started <= args.motion_timeout_s:
        last = list(robot.get_joint())
        err = joint_distance(target_joints, last)
        samples += 1
        if err <= args.joint_tolerance_rad:
            return {
                "reached": True,
                "label": label,
                "actual_joint": last,
                "joint_l2_error_rad": err,
                "wait_s": time.perf_counter() - started,
                "sample_count": samples,
            }
        time.sleep(args.poll_s)
    return {
        "reached": False,
        "label": label,
        "actual_joint": last,
        "joint_l2_error_rad": joint_distance(target_joints, last),
        "wait_s": time.perf_counter() - started,
        "sample_count": samples,
    }


def call_cartesian_motion(robot, pose: list[float], args: argparse.Namespace, label: str) -> dict[str, Any]:
    started = time.perf_counter()
    method_used = None
    ret = None
    if hasattr(robot, "movel_line"):
        method_used = "movel_line"
        # Different locally compiled SDK modules may expose this newer helper.
        ret = robot.movel_line(pose, args.line_velocity_m_s, args.line_acc_m_s2, False, True)
        if ret not in (None, 0):
            raise RuntimeError(f"{label}: movel_line returned {ret}")
    elif args.allow_movel_fallback:
        method_used = "movel"
        robot.movel(pose)
    else:
        raise RuntimeError(
            "Current SDK module has no movel_line. Refusing to fall back to movel unless "
            "--allow-movel-fallback is explicitly provided."
        )
    reach = wait_for_pose(robot, pose, args, label)
    if not reach["reached"]:
        raise RuntimeError(
            f"{label}: target not reached within {args.motion_timeout_s:.1f}s; "
            f"position_error={reach['position_error_m']:.4f}m"
        )
    time.sleep(args.settle_s)
    actual = list(robot.get_status())
    actual_joint = list(robot.get_joint())
    payload = {
        "label": label,
        "method": method_used,
        "return_code": ret,
        "target_pose": pose,
        "actual_pose_after": actual,
        "actual_joint_after": actual_joint,
        "position_error_m": cartesian_distance(pose, actual),
        "reach_check": reach,
        "elapsed_s": time.perf_counter() - started,
    }
    if payload["position_error_m"] > args.pose_tolerance_m:
        raise RuntimeError(f"{label}: settled pose error too large: {payload['position_error_m']:.4f}m")
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "robot_commanded": False,
        "execute_requested": bool(args.execute),
        "operator_phrase_ok": args.operator_phrase == REQUIRED_OPERATOR_PHRASE,
        "required_operator_phrase": REQUIRED_OPERATOR_PHRASE,
        "sequence": [],
        "parameters": vars(args),
    }

    home_deg = parse_home_degrees(args.home_joints_deg)
    home_rad = [math.radians(v) for v in home_deg]
    log["home_joints_deg"] = home_deg
    log["home_joints_rad"] = home_rad

    if not args.execute:
        log["status"] = "DRY_RUN_NO_ROBOT_COMMAND"
        log["plan_note"] = (
            "Dry-run only. With --execute, the script will movej to home, read TCP, "
            f"move to X=home{args.x_offset:+.3f} and Y={args.y_start:.3f}, wait, "
            f"move to X=home{args.x_offset:+.3f} and Y={args.y_goal:.3f}, wait, and return home."
        )
        write_json(output_dir / "execution_plan.json", log)
        print(json.dumps(log, indent=2, ensure_ascii=False, default=json_default))
        return log

    if args.operator_phrase != REQUIRED_OPERATOR_PHRASE:
        log["status"] = "BLOCKED_BAD_OPERATOR_PHRASE"
        write_json(output_dir / "execution_plan.json", log)
        raise RuntimeError(f"bad operator phrase; required: {REQUIRED_OPERATOR_PHRASE}")

    robot, so_path = load_robot_module(args.sdk_dir)
    log["sdk_so"] = str(so_path)
    print(f"[sdk] loaded: {so_path}")

    try:
        ok = robot.init()
        if not ok:
            raise RuntimeError("robot.init() failed")

        require_confirmation(
            True,
            "Step 1/4: movej to initial posture [0,0,90,0,90,0] deg. Keep emergency stop ready.",
        )
        robot.movej(home_rad)
        home_reach = wait_for_joints(robot, home_rad, args, "movej_home")
        if not home_reach["reached"]:
            raise RuntimeError(
                f"movej_home: target not reached within {args.motion_timeout_s:.1f}s; "
                f"joint_l2_error={home_reach['joint_l2_error_rad']:.4f}rad"
            )
        time.sleep(args.settle_s)
        home_pose = list(robot.get_status())
        home_joint_actual = list(robot.get_joint())
        log["robot_commanded"] = True
        log["home_reach_check"] = home_reach
        log["home_pose_after_movej"] = home_pose
        log["home_joint_after_movej"] = home_joint_actual
        print(f"[home pose] {fmt_pose(home_pose)}")
        print(f"[home joint] {fmt_joints(home_joint_actual)}")

        start_pose = make_pose(home_pose, y=args.y_start, x_offset=args.x_offset)
        goal_pose = make_pose(home_pose, y=args.y_goal, x_offset=args.x_offset)
        check_pose_limits(home_pose, args, "home")
        check_pose_limits(start_pose, args, "start")
        check_pose_limits(goal_pose, args, "goal")

        print("")
        print("[plan]")
        print(f"  home : {fmt_pose(home_pose)}")
        print(f"  start: {fmt_pose(start_pose)}")
        print(f"  goal : {fmt_pose(goal_pose)}")
        print(f"  x offset from home TCP: {args.x_offset:+.3f} m")
        print(f"  method preference: movel_line; fallback movel allowed={args.allow_movel_fallback}")
        print(f"  line velocity={args.line_velocity_m_s:.3f} m/s, line acc={args.line_acc_m_s2:.3f} m/s^2")

        require_confirmation(
            True,
            f"Step 2/4: move TCP horizontally to start pose X=home{args.x_offset:+.3f} m, Y={args.y_start:+.3f} m.",
        )
        log["sequence"].append(call_cartesian_motion(robot, start_pose, args, "move_to_start"))
        print(f"[after start] {fmt_pose(log['sequence'][-1]['actual_pose_after'])}")

        if args.stop_after_start:
            log["status"] = "STOPPED_AT_START"
            log["stop_note"] = (
                "Robot moved to the planar start pose and the program exited before the "
                "goal stroke. It is now safe to run a separate perception/planning script "
                "after confirming the robot controller is not occupied by this process."
            )
            print("[stop-after-start] reached start pose; exiting without moving to goal.")
            return log

        require_confirmation(
            True,
            f"Step 3/4: after visual safety check, move TCP horizontally to goal pose X=home{args.x_offset:+.3f} m, Y={args.y_goal:+.3f} m.",
        )
        log["sequence"].append(call_cartesian_motion(robot, goal_pose, args, "move_to_goal"))
        print(f"[after goal] {fmt_pose(log['sequence'][-1]['actual_pose_after'])}")

        require_confirmation(
            True,
            "Step 4/4: return to initial TCP pose and then exact initial joint posture.",
        )
        log["sequence"].append(call_cartesian_motion(robot, home_pose, args, "return_to_home_tcp"))
        robot.movej(home_rad)
        final_joint_reach = wait_for_joints(robot, home_rad, args, "final_movej_home")
        if not final_joint_reach["reached"]:
            raise RuntimeError(
                f"final_movej_home: target not reached within {args.motion_timeout_s:.1f}s; "
                f"joint_l2_error={final_joint_reach['joint_l2_error_rad']:.4f}rad"
            )
        time.sleep(args.settle_s)
        final_pose = list(robot.get_status())
        final_joint = list(robot.get_joint())
        log["final_pose"] = final_pose
        log["final_joint"] = final_joint
        log["final_joint_reach_check"] = final_joint_reach
        log["final_home_position_error_m"] = cartesian_distance(home_pose, final_pose)
        log["status"] = "COMPLETED" if log["final_home_position_error_m"] <= args.pose_tolerance_m else "COMPLETED_JOINT_HOME_POSE_OFFSET"
    except Exception as exc:
        log["status"] = "ABORTED_OR_FAILED"
        log["error"] = str(exc)
        raise
    finally:
        try:
            robot.log_out()
        except Exception:
            pass
        write_json(output_dir / "execution_log.json", log)
        print(f"[log] {output_dir / 'execution_log.json'}")
    return log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="actually command the robot")
    parser.add_argument("--operator-phrase", default="")
    parser.add_argument("--sdk-dir", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--home-joints-deg", default="0,0,90,0,90,0")
    parser.add_argument(
        "--x-offset",
        type=float,
        default=0.0,
        help="add this offset to the home TCP X coordinate for both start and goal poses",
    )
    parser.add_argument("--y-start", type=float, default=0.4)
    parser.add_argument("--y-goal", type=float, default=-0.4)
    parser.add_argument("--line-velocity-m-s", type=float, default=0.025)
    parser.add_argument("--line-acc-m-s2", type=float, default=0.06)
    parser.add_argument("--settle-s", type=float, default=0.4)
    parser.add_argument("--poll-s", type=float, default=0.10)
    parser.add_argument("--motion-timeout-s", type=float, default=60.0)
    parser.add_argument("--pose-tolerance-m", type=float, default=0.015)
    parser.add_argument("--joint-tolerance-rad", type=float, default=0.02)
    parser.add_argument("--allow-movel-fallback", action="store_true")
    parser.add_argument(
        "--stop-after-start",
        action="store_true",
        help="move to home and then the planar start pose, save the log, disconnect, and do not continue to the goal",
    )
    parser.add_argument("--min-x", type=float, default=-0.2)
    parser.add_argument("--max-x", type=float, default=0.9)
    parser.add_argument("--min-y", type=float, default=-0.55)
    parser.add_argument("--max-y", type=float, default=0.55)
    parser.add_argument("--min-z", type=float, default=0.25)
    parser.add_argument("--max-z", type=float, default=0.9)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
