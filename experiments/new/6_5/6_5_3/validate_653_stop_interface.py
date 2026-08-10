#!/usr/bin/env python3
"""Validate the AUBO stop interface for 6.5.3 without any obstacle.

This is a guarded hardware diagnostic, not a paper trial.  It commands the same
low-speed tabletop reference line used by 6.5.2/6.5.3, waits a fixed delay, then
calls the SDK stop API while recording pose and joint feedback.

No obstacle should be placed in the workspace.  No robot command is sent unless
``--execute`` and the exact operator phrase are provided.
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
    check_pose_limits,
    joint_distance,
    make_pose,
    parse_home_degrees,
    require_confirmation,
    wait_for_joints,
)
from robot.linear_move_debug import fmt_pose, load_robot_module  # noqa: E402


DEFAULT_OUTPUT = ROOT / "results" / "new" / "6_5" / "6_5_3" / "stop_interface_validation"
REQUIRED_OPERATOR_PHRASE = "CCRO_653_STOP_VALIDATION_APPROVED"


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


def cartesian_distance(a: list[float], b: list[float]) -> float:
    return float(np.linalg.norm(np.asarray(a[:3], dtype=np.float64) - np.asarray(b[:3], dtype=np.float64)))


def call_stop(robot: Any, *, verbose: bool = True) -> dict[str, Any]:
    for name in ("move_control_stop", "offline_track_stop", "teach_stop"):
        if hasattr(robot, name):
            try:
                ret = getattr(robot, name)(verbose)
            except TypeError:
                ret = getattr(robot, name)()
            return {"method": name, "return": ret}
    return {"method": None, "return": None, "error": "no supported stop function found"}


def sample_robot(robot: Any, t_s: float, phase: str) -> dict[str, Any]:
    pose = list(robot.get_status())
    q = list(robot.get_joint())
    return {
        "t_s": f"{t_s:.6f}",
        "phase": phase,
        **{f"pose_{name}": f"{float(pose[i]):.8f}" for i, name in enumerate(("x", "y", "z", "rx", "ry", "rz"))},
        **{f"q{j+1}_rad": f"{float(q[j]):.8f}" for j in range(6)},
    }


def wait_for_pose(robot: Any, target_pose: list[float], args: argparse.Namespace, label: str) -> dict[str, Any]:
    started = time.perf_counter()
    last_pose = list(robot.get_status())
    samples = 0
    while time.perf_counter() - started <= args.motion_timeout_s:
        last_pose = list(robot.get_status())
        err = cartesian_distance(target_pose, last_pose)
        samples += 1
        if err <= args.pose_tolerance_m:
            return {
                "label": label,
                "reached": True,
                "elapsed_s": time.perf_counter() - started,
                "position_error_m": err,
                "actual_pose": last_pose,
                "sample_count": samples,
            }
        time.sleep(args.poll_s)
    return {
        "label": label,
        "reached": False,
        "elapsed_s": time.perf_counter() - started,
        "position_error_m": cartesian_distance(target_pose, last_pose),
        "actual_pose": last_pose,
        "sample_count": samples,
    }


def estimate_motion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) < 2:
        return {"sample_count": len(rows)}
    times = np.asarray([float(r["t_s"]) for r in rows], dtype=np.float64)
    xyz = np.asarray([[float(r[f"pose_{k}"]) for k in ("x", "y", "z")] for r in rows], dtype=np.float64)
    q = np.asarray([[float(r[f"q{j+1}_rad"]) for j in range(6)] for r in rows], dtype=np.float64)
    dt = np.diff(times)
    dt[dt <= 1.0e-9] = np.nan
    v_xyz = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / dt
    v_q = np.linalg.norm(np.diff(q, axis=0), axis=1) / dt
    return {
        "sample_count": len(rows),
        "max_tcp_speed_m_s": float(np.nanmax(v_xyz)) if len(v_xyz) else 0.0,
        "tail_tcp_speed_m_s": float(np.nanmedian(v_xyz[-min(5, len(v_xyz)) :])) if len(v_xyz) else 0.0,
        "max_joint_l2_speed_rad_s": float(np.nanmax(v_q)) if len(v_q) else 0.0,
        "tail_joint_l2_speed_rad_s": float(np.nanmedian(v_q[-min(5, len(v_q)) :])) if len(v_q) else 0.0,
        "tcp_path_length_m": float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))),
        "joint_path_length_rad": float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1))),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    home_deg = parse_home_degrees(args.home_joints_deg)
    home_rad = [math.radians(v) for v in home_deg]
    log: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "6.5.3 stop interface validation",
        "robot_commanded": False,
        "execute_requested": bool(args.execute),
        "operator_phrase_ok": args.operator_phrase == REQUIRED_OPERATOR_PHRASE,
        "required_operator_phrase": REQUIRED_OPERATOR_PHRASE,
        "parameters": vars(args),
    }
    if not args.execute:
        log["status"] = "DRY_RUN_NO_ROBOT_COMMAND"
        write_json(output / "stop_validation_log.json", log)
        print(json.dumps(log, indent=2, ensure_ascii=False, default=json_default))
        return log
    if args.operator_phrase != REQUIRED_OPERATOR_PHRASE:
        log["status"] = "BLOCKED_BAD_OPERATOR_PHRASE"
        write_json(output / "stop_validation_log.json", log)
        raise RuntimeError(f"bad operator phrase; required: {REQUIRED_OPERATOR_PHRASE}")

    robot, so_path = load_robot_module(args.sdk_dir)
    log["sdk_so"] = str(so_path)
    rows: list[dict[str, Any]] = []
    stop_info = None
    fields = [
        "t_s",
        "phase",
        *[f"pose_{name}" for name in ("x", "y", "z", "rx", "ry", "rz")],
        *[f"q{j+1}_rad" for j in range(6)],
    ]
    try:
        if not robot.init():
            raise RuntimeError("robot.init() failed")
        if not hasattr(robot, "movel_line"):
            raise RuntimeError("current SDK does not expose movel_line")
        if not hasattr(robot, "move_control_stop"):
            raise RuntimeError("current SDK does not expose move_control_stop")

        require_confirmation(True, "Step 1/3: movej to home, then move to tabletop start. Workspace must be empty.")
        robot.movej(home_rad)
        reach = wait_for_joints(robot, home_rad, args, "movej_home")
        if not reach["reached"]:
            raise RuntimeError(f"home not reached: {reach}")
        time.sleep(args.settle_s)
        home_pose = list(robot.get_status())
        start_pose = make_pose(home_pose, y=args.y_start, x_offset=args.x_offset)
        goal_pose = make_pose(home_pose, y=args.y_goal, x_offset=args.x_offset)
        check_pose_limits(home_pose, args, "home")
        check_pose_limits(start_pose, args, "start")
        check_pose_limits(goal_pose, args, "goal")
        log.update({"home_pose": home_pose, "start_pose": start_pose, "goal_pose": goal_pose})
        print(f"[home]  {fmt_pose(home_pose)}")
        print(f"[start] {fmt_pose(start_pose)}")
        print(f"[goal]  {fmt_pose(goal_pose)}")

        ret = robot.movel_line(start_pose, args.start_line_velocity_m_s, args.start_line_acc_m_s2, False, True)
        log["move_to_start_return"] = ret
        start_reach = wait_for_pose(robot, start_pose, args, "move_to_start")
        log["start_reach_check"] = start_reach
        log["start_pose_actual"] = start_reach["actual_pose"]
        log["start_error_m"] = start_reach["position_error_m"]
        if not start_reach["reached"]:
            raise RuntimeError(
                f"start pose not reached within {args.motion_timeout_s:.1f}s; "
                f"error={log['start_error_m']:.4f} m"
            )
        time.sleep(args.start_settle_s)

        require_confirmation(True, f"Step 2/3: start empty-workspace line motion; automatic stop after {args.stop_after_s:.2f}s.")
        started = time.perf_counter()
        ret = robot.movel_line(goal_pose, args.line_velocity_m_s, args.line_acc_m_s2, False, True)
        log["reference_motion_return"] = ret
        log["robot_commanded"] = True
        stop_called_at = None
        while time.perf_counter() - started <= args.record_after_stop_s + args.stop_after_s:
            t = time.perf_counter() - started
            phase = "before_stop" if stop_called_at is None else "after_stop"
            rows.append(sample_robot(robot, t, phase))
            if stop_called_at is None and t >= args.stop_after_s:
                stop_called_at = time.perf_counter() - started
                stop_info = call_stop(robot, verbose=True)
                rows.append(sample_robot(robot, time.perf_counter() - started, "stop_command_returned"))
            time.sleep(args.poll_s)
        log["stop_called_at_s"] = stop_called_at
        log["stop_info"] = stop_info
        log["status"] = "STOP_VALIDATION_RECORDED"

        before_rows = [r for r in rows if r["phase"] == "before_stop"]
        after_rows = [r for r in rows if r["phase"] in {"after_stop", "stop_command_returned"}]
        log["before_stop_motion"] = estimate_motion(before_rows)
        log["after_stop_motion"] = estimate_motion(after_rows)
        if after_rows:
            stop_pose = [float(after_rows[0][f"pose_{k}"]) for k in ("x", "y", "z", "rx", "ry", "rz")]
            final_pose = [float(after_rows[-1][f"pose_{k}"]) for k in ("x", "y", "z", "rx", "ry", "rz")]
            log["post_stop_tcp_drift_m"] = cartesian_distance(stop_pose, final_pose)
            log["stop_effective_by_drift"] = bool(log["post_stop_tcp_drift_m"] <= args.max_post_stop_drift_m)
        # Write before any SDK cleanup. Some AUBO SDK stop-state cleanup paths can
        # abort the process; the diagnostic data must already be durable.
        write_csv(output / "stop_validation_feedback.csv", rows, fields)
        write_json(output / "stop_validation_log.json", log)
    except Exception as exc:
        log["status"] = "FAILED"
        log["error"] = str(exc)
        write_csv(output / "stop_validation_feedback.csv", rows, fields)
        write_json(output / "stop_validation_log.json", log)
        raise
    finally:
        if stop_info is None:
            try:
                call_stop(robot, verbose=True)
            except Exception:
                pass
        if not args.skip_logout:
            try:
                robot.log_out()
            except Exception:
                pass
        write_csv(output / "stop_validation_feedback.csv", rows, fields)
        write_json(output / "stop_validation_log.json", log)
        print(f"[log] {output / 'stop_validation_log.json'}")
    return log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--operator-phrase", default="")
    parser.add_argument("--sdk-dir", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--home-joints-deg", default="0,0,90,0,90,0")
    parser.add_argument("--x-offset", type=float, default=0.10)
    parser.add_argument("--y-start", type=float, default=0.4)
    parser.add_argument("--y-goal", type=float, default=-0.4)
    parser.add_argument("--line-velocity-m-s", type=float, default=0.010)
    parser.add_argument("--line-acc-m-s2", type=float, default=0.025)
    parser.add_argument("--start-line-velocity-m-s", type=float, default=0.030)
    parser.add_argument("--start-line-acc-m-s2", type=float, default=0.080)
    parser.add_argument("--stop-after-s", type=float, default=3.0)
    parser.add_argument("--record-after-stop-s", type=float, default=4.0)
    parser.add_argument("--start-settle-s", type=float, default=8.0)
    parser.add_argument("--settle-s", type=float, default=0.4)
    parser.add_argument("--poll-s", type=float, default=0.04)
    parser.add_argument("--motion-timeout-s", type=float, default=90.0)
    parser.add_argument("--pose-tolerance-m", type=float, default=0.020)
    parser.add_argument("--joint-tolerance-rad", type=float, default=0.02)
    parser.add_argument("--max-post-stop-drift-m", type=float, default=0.010)
    parser.add_argument(
        "--skip-logout",
        action="store_true",
        default=True,
        help="avoid SDK logout cleanup after stop-state diagnostics; the process exits immediately anyway",
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
    summary = run(args)
    print(json.dumps({"status": summary.get("status"), "stop_info": summary.get("stop_info"), "output": str(args.output)}, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
