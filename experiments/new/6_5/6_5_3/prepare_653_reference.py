#!/usr/bin/env python3
"""Prepare the 6.5.3 low-speed reference motion.

The real dynamic repair experiment reuses the same safe tabletop start used in
6.5.2: initial joints [0, 0, 90, 0, 90, 0] deg, with no additional TCP-X shift and
Y moving from +0.40 m to -0.40 m.  This program can either:

* write a dry-run plan and a simple Cartesian preview; or
* command the robot through the guarded reference stroke and record feedback.

No robot command is sent unless ``--execute`` and the exact operator phrase are
provided.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
EXP652 = ROOT / "experiments" / "new" / "6_5" / "6_5_2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP652) not in sys.path:
    sys.path.insert(0, str(EXP652))

from execute_652_planar_y_guarded import (  # noqa: E402
    call_cartesian_motion,
    cartesian_distance,
    check_pose_limits,
    joint_distance,
    make_pose,
    parse_home_degrees,
    require_confirmation,
    wait_for_joints,
)
from robot.linear_move_debug import fmt_joints, fmt_pose, load_robot_module  # noqa: E402


DEFAULT_OUTPUT = ROOT / "results" / "new" / "6_5" / "6_5_3" / "reference"
REQUIRED_OPERATOR_PHRASE = "CCRO_653_REFERENCE_APPROVED"


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_cartesian_reference(start_pose: list[float], goal_pose: list[float], args: argparse.Namespace) -> list[dict[str, Any]]:
    count = max(2, int(args.preview_samples))
    rows: list[dict[str, Any]] = []
    for i, u in enumerate(np.linspace(0.0, 1.0, count)):
        pose = np.asarray(start_pose, dtype=np.float64) * (1.0 - u) + np.asarray(goal_pose, dtype=np.float64) * u
        if args.reference_shape == "micro_curve":
            # Small in-plane X offset: zero at endpoints, maximum at the middle.
            pose[0] += float(args.curve_amplitude_m) * math.sin(math.pi * float(u))
        rows.append(
            {
                "index": i,
                "u": f"{float(u):.6f}",
                "x": f"{pose[0]:.8f}",
                "y": f"{pose[1]:.8f}",
                "z": f"{pose[2]:.8f}",
                "rx": f"{pose[3]:.8f}",
                "ry": f"{pose[4]:.8f}",
                "rz": f"{pose[5]:.8f}",
            }
        )
    return rows


def estimate_velocity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) < 2:
        return rows
    times = np.asarray([float(r["t_s"]) for r in rows], dtype=np.float64)
    q = np.asarray([[float(r[f"q{j+1}_rad"]) for j in range(6)] for r in rows], dtype=np.float64)
    qd = np.gradient(q, times, axis=0, edge_order=1)
    for i, row in enumerate(rows):
        for j in range(6):
            row[f"qd{j+1}_rad_s"] = f"{qd[i, j]:.8f}"
    return rows


def append_feedback_sample(robot, started: float, rows: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    pose = list(robot.get_status())
    joint = list(robot.get_joint())
    rows.append(
        {
            "t_s": f"{time.perf_counter() - started:.6f}",
            **{f"q{j+1}_rad": f"{float(joint[j]):.8f}" for j in range(6)},
            **{f"pose_{name}": f"{float(pose[i]):.8f}" for i, name in enumerate(("x", "y", "z", "rx", "ry", "rz"))},
        }
    )
    return pose, joint


def call_cartesian_motion_recorded(
    robot,
    pose: list[float],
    args: argparse.Namespace,
    label: str,
    *,
    started: float,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Issue a nonblocking line move and record feedback while it is moving."""
    command_started = time.perf_counter()
    append_feedback_sample(robot, started, rows)  # preserve the actual start state
    if not hasattr(robot, "movel_line"):
        raise RuntimeError("reference recording requires the nonblocking movel_line interface")
    ret = robot.movel_line(pose, args.line_velocity_m_s, args.line_acc_m_s2, False, True)
    if ret not in (None, 0):
        raise RuntimeError(f"{label}: movel_line returned {ret}")

    last_pose = list(robot.get_status())
    last_joint = list(robot.get_joint())
    reached_at: float | None = None
    while time.perf_counter() - command_started <= args.motion_timeout_s:
        last_pose, last_joint = append_feedback_sample(robot, started, rows)
        error = cartesian_distance(pose, last_pose)
        if error <= args.record_goal_tolerance_m:
            if reached_at is None:
                reached_at = time.perf_counter()
            if time.perf_counter() - reached_at >= args.settle_s:
                break
        else:
            reached_at = None
        time.sleep(args.poll_s)
    else:
        raise RuntimeError(
            f"{label}: target not reached to recording tolerance {args.record_goal_tolerance_m:.4f} m "
            f"within {args.motion_timeout_s:.1f}s"
        )
    return {
        "label": label,
        "method": "movel_line_recorded",
        "return_code": ret,
        "target_pose": pose,
        "actual_pose_after": last_pose,
        "actual_joint_after": last_joint,
        "position_error_m": cartesian_distance(pose, last_pose),
        "elapsed_s": time.perf_counter() - command_started,
        "recorded_rows_total": len(rows),
    }


def reference_completeness(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    if len(rows) < 2:
        return {"accepted": False, "checks": {"minimum_rows": False}, "reasons": ["too_few_rows"]}
    times = np.asarray([float(row["t_s"]) for row in rows], dtype=np.float64)
    ys = np.asarray([float(row["pose_y"]) for row in rows], dtype=np.float64)
    qd = np.asarray([[float(row[f"qd{j}_rad_s"]) for j in range(1, 7)] for row in rows], dtype=np.float64)
    expected_span = abs(float(args.y_goal) - float(args.y_start))
    observed_span = float(np.max(ys) - np.min(ys))
    checks = {
        "minimum_rows": len(rows) >= args.minimum_reference_rows,
        "timestamps_strict": bool(np.all(np.diff(times) > 0.0)),
        "starts_near_reference_start": abs(float(ys[0]) - float(args.y_start)) <= args.reference_endpoint_tolerance_m,
        "ends_near_reference_goal": abs(float(ys[-1]) - float(args.y_goal)) <= args.reference_endpoint_tolerance_m,
        "y_span_complete": observed_span >= args.minimum_reference_span_fraction * expected_span,
        "nonzero_joint_velocity": int(np.sum(np.max(np.abs(qd), axis=1) > args.nonzero_qd_threshold)) >= args.minimum_moving_rows,
    }
    return {
        "accepted": bool(all(checks.values())),
        "checks": checks,
        "reasons": [name for name, ok in checks.items() if not ok],
        "rows": len(rows),
        "duration_s": float(times[-1] - times[0]),
        "y_start_observed_m": float(ys[0]),
        "y_end_observed_m": float(ys[-1]),
        "y_span_observed_m": observed_span,
        "y_span_required_m": float(args.minimum_reference_span_fraction * expected_span),
        "moving_rows": int(np.sum(np.max(np.abs(qd), axis=1) > args.nonzero_qd_threshold)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    home_deg = parse_home_degrees(args.home_joints_deg)
    home_rad = [math.radians(v) for v in home_deg]

    log: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "6.5.3 dynamic repair reference",
        "robot_commanded": False,
        "execute_requested": bool(args.execute),
        "operator_phrase_ok": args.operator_phrase == REQUIRED_OPERATOR_PHRASE,
        "required_operator_phrase": REQUIRED_OPERATOR_PHRASE,
        "home_joints_deg": home_deg,
        "home_joints_rad": home_rad,
        "parameters": vars(args),
    }

    if not args.execute:
        # Without the live TCP pose we still document the intended reference.
        log["status"] = "DRY_RUN_NO_ROBOT_COMMAND"
        log["reference_note"] = (
            "Run with --execute to move to home, derive the live TCP pose, then record the "
            "same tabletop Y stroke used by 6.5.2."
        )
        write_json(output_dir / "reference_plan.json", log)
        print(json.dumps(log, indent=2, ensure_ascii=False, default=json_default))
        return log

    if args.operator_phrase != REQUIRED_OPERATOR_PHRASE:
        log["status"] = "BLOCKED_BAD_OPERATOR_PHRASE"
        write_json(output_dir / "reference_plan.json", log)
        raise RuntimeError(f"bad operator phrase; required: {REQUIRED_OPERATOR_PHRASE}")

    robot, so_path = load_robot_module(args.sdk_dir)
    log["sdk_so"] = str(so_path)
    print(f"[sdk] loaded: {so_path}")

    try:
        if not robot.init():
            raise RuntimeError("robot.init() failed")
        require_confirmation(True, "Step 1/3: movej to initial posture [0,0,90,0,90,0] deg.")
        robot.movej(home_rad)
        reach_home = wait_for_joints(robot, home_rad, args, "movej_home")
        if not reach_home["reached"]:
            raise RuntimeError(f"home not reached: {reach_home}")
        time.sleep(args.settle_s)
        home_pose = list(robot.get_status())
        home_joint = list(robot.get_joint())
        start_pose = make_pose(home_pose, y=args.y_start, x_offset=args.x_offset)
        goal_pose = make_pose(home_pose, y=args.y_goal, x_offset=args.x_offset)
        check_pose_limits(home_pose, args, "home")
        check_pose_limits(start_pose, args, "start")
        check_pose_limits(goal_pose, args, "goal")
        log.update(
            {
                "home_pose": home_pose,
                "home_joint": home_joint,
                "start_pose": start_pose,
                "goal_pose": goal_pose,
                "home_reach_check": reach_home,
            }
        )
        print(f"[home]  {fmt_pose(home_pose)}")
        print(f"[start] {fmt_pose(start_pose)}")
        print(f"[goal]  {fmt_pose(goal_pose)}")

        preview_rows = sample_cartesian_reference(start_pose, goal_pose, args)
        write_csv(output_dir / "reference_cartesian_preview.csv", preview_rows, list(preview_rows[0]))

        require_confirmation(True, "Step 2/3: move TCP to reference start pose.")
        move_start = call_cartesian_motion(robot, start_pose, args, "move_to_reference_start")
        print(f"[at start] {fmt_pose(move_start['actual_pose_after'])}")
        if args.stop_after_start:
            log["robot_commanded"] = True
            log["status"] = "STOPPED_AT_REFERENCE_START"
            log["move_start"] = move_start
            write_json(output_dir / "reference_log.json", log)
            return log

        require_confirmation(True, "Step 3/3: execute low-speed reference stroke and record joint feedback.")
        started = time.perf_counter()
        feedback_rows: list[dict[str, Any]] = []
        if args.reference_shape == "line":
            move_goal = call_cartesian_motion_recorded(
                robot,
                goal_pose,
                args,
                "execute_reference_line",
                started=started,
                rows=feedback_rows,
            )
        else:
            # Execute the micro-curve as short guarded line segments. This is for preview
            # and low-speed pilot only; the official dynamic trial can still use line.
            curve_rows = sample_cartesian_reference(start_pose, goal_pose, args)
            move_goal = {"label": "execute_reference_micro_curve", "segments": []}
            for row in curve_rows[1:]:
                pose = [float(row[k]) for k in ("x", "y", "z", "rx", "ry", "rz")]
                move_goal["segments"].append(
                    call_cartesian_motion_recorded(
                        robot,
                        pose,
                        args,
                        "curve_segment",
                        started=started,
                        rows=feedback_rows,
                    )
                )
        estimate_velocity(feedback_rows)
        feedback = {
            "rows": feedback_rows,
            "last_pose": [float(feedback_rows[-1][f"pose_{name}"]) for name in ("x", "y", "z", "rx", "ry", "rz")],
            "last_joint": [float(feedback_rows[-1][f"q{j}_rad"]) for j in range(1, 7)],
        }
        completeness = reference_completeness(feedback_rows, args)
        fields = ["t_s", *[f"q{j+1}_rad" for j in range(6)], *[f"qd{j+1}_rad_s" for j in range(6)], *[f"pose_{name}" for name in ("x", "y", "z", "rx", "ry", "rz")]]
        feedback_name = "reference_feedback.csv" if completeness["accepted"] else "reference_feedback_invalid.csv"
        write_csv(output_dir / feedback_name, feedback["rows"], fields)
        log["robot_commanded"] = True
        log["status"] = "REFERENCE_RECORDED" if completeness["accepted"] else "REFERENCE_INVALID_INCOMPLETE"
        log["move_start"] = move_start
        log["move_goal"] = move_goal
        log["feedback_rows"] = len(feedback["rows"])
        log["feedback_file"] = str(output_dir / feedback_name)
        log["reference_completeness"] = completeness
        log["final_pose"] = feedback["last_pose"]
        log["final_joint"] = feedback["last_joint"]
        log["goal_position_error_m"] = cartesian_distance(goal_pose, feedback["last_pose"])
        if not completeness["accepted"]:
            raise RuntimeError(f"reference completeness gate failed: {completeness['reasons']}")
    finally:
        try:
            robot.log_out()
        except Exception:
            pass
        write_json(output_dir / "reference_log.json", log)
        print(f"[log] {output_dir / 'reference_log.json'}")
    return log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--operator-phrase", default="")
    parser.add_argument("--sdk-dir", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--home-joints-deg", default="0,0,90,0,90,0")
    parser.add_argument("--x-offset", type=float, default=0.0)
    parser.add_argument("--y-start", type=float, default=0.4)
    parser.add_argument("--y-goal", type=float, default=-0.4)
    parser.add_argument("--reference-shape", choices=["line", "micro_curve"], default="line")
    parser.add_argument("--curve-amplitude-m", type=float, default=0.04)
    parser.add_argument("--preview-samples", type=int, default=81)
    # Keep preparation motion consistent with the unified D1/D2 protocol.
    parser.add_argument("--line-velocity-m-s", type=float, default=0.020)
    parser.add_argument("--line-acc-m-s2", type=float, default=0.05)
    parser.add_argument("--record-duration-s", type=float, default=40.0)
    parser.add_argument("--record-goal-tolerance-m", type=float, default=0.002)
    parser.add_argument("--reference-endpoint-tolerance-m", type=float, default=0.015)
    parser.add_argument("--minimum-reference-span-fraction", type=float, default=0.95)
    parser.add_argument("--minimum-reference-rows", type=int, default=200)
    parser.add_argument("--minimum-moving-rows", type=int, default=50)
    parser.add_argument("--nonzero-qd-threshold", type=float, default=1.0e-4)
    parser.add_argument("--settle-s", type=float, default=0.4)
    parser.add_argument("--poll-s", type=float, default=0.04)
    parser.add_argument("--motion-timeout-s", type=float, default=90.0)
    parser.add_argument("--pose-tolerance-m", type=float, default=0.015)
    parser.add_argument("--joint-tolerance-rad", type=float, default=0.02)
    parser.add_argument("--allow-movel-fallback", action="store_true")
    parser.add_argument("--stop-after-start", action="store_true")
    parser.add_argument("--min-x", type=float, default=-0.2)
    parser.add_argument("--max-x", type=float, default=0.9)
    parser.add_argument("--min-y", type=float, default=-0.55)
    parser.add_argument("--max-y", type=float, default=0.55)
    parser.add_argument("--min-z", type=float, default=0.25)
    parser.add_argument("--max-z", type=float, default=0.9)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps({"status": summary.get("status"), "output": str(args.output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
