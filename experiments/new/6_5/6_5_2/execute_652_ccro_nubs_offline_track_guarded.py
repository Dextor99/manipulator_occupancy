#!/usr/bin/env python3
"""Guarded continuous Offline Track executor for 6.5.2 CCRO-NUBS plans.

This is the preferred hardware playback path for a precomputed joint-space
NUBS/CCRO candidate.  Unlike the debug ``movej`` waypoint player, this script
submits the full joint waypoint sequence to the AUBO Offline Track interface:

    robotServiceOfflineTrackWaypointClear
    robotServiceOfflineTrackWaypointAppend
    robotServiceOfflineTrackMoveStartup

The default mode is dry-run.  Real motion requires ``--execute`` and an exact
operator phrase.  The script still refuses any plan that did not pass dense
verification.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
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
    / "6_5_2"
    / "planar_static_live"
    / "rs1_lateral_table_obstacle"
    / "trials"
    / "rs1_lateral_table_obstacle_r06"
    / "ccro_nubs_jointspace_plan"
)
DEFAULT_OUTPUT = ROOT / "results" / "new" / "6_5" / "6_5_2" / "offline_track_execution"
REQUIRED_OPERATOR_PHRASE = "CCRO_652_OFFLINE_TRACK_APPROVED"


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def load_summary(plan_dir: Path) -> dict[str, Any]:
    return json.loads((plan_dir / "summary.json").read_text(encoding="utf-8"))


def load_candidate(plan_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = plan_dir / "ccro_nubs_candidate_trajectory.csv"
    times: list[float] = []
    qs: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            times.append(float(row["t_s"]))
            qs.append([float(row[f"q{i}_rad"]) for i in range(1, 7)])
    if len(qs) < 2:
        raise RuntimeError(f"too few trajectory rows in {path}")
    return np.asarray(times, dtype=np.float64), np.asarray(qs, dtype=np.float64)


def resample_for_offline_track(
    times: np.ndarray,
    qs: np.ndarray,
    *,
    playback_duration_s: float,
    controller_period_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    if playback_duration_s <= 0.0:
        return times, qs
    if controller_period_s <= 0.0:
        raise ValueError("--controller-waypoint-period-s must be positive")
    count = max(2, int(np.ceil(playback_duration_s / controller_period_s)) + 1)
    source = np.linspace(0.0, 1.0, len(qs))
    target = np.linspace(0.0, 1.0, count)
    out = np.empty((count, qs.shape[1]), dtype=np.float64)
    for j in range(qs.shape[1]):
        out[:, j] = np.interp(target, source, qs[:, j])
    return np.linspace(0.0, playback_duration_s, count), out


def maybe_downsample(times: np.ndarray, qs: np.ndarray, max_waypoints: int | None) -> tuple[np.ndarray, np.ndarray]:
    if max_waypoints is None or max_waypoints <= 0 or len(qs) <= max_waypoints:
        return times, qs
    if max_waypoints < 2:
        raise ValueError("--max-waypoints must be >= 2")
    idx = np.unique(np.round(np.linspace(0, len(qs) - 1, max_waypoints)).astype(int))
    if idx[0] != 0:
        idx = np.r_[0, idx]
    if idx[-1] != len(qs) - 1:
        idx = np.r_[idx, len(qs) - 1]
    return times[idx], qs[idx]


def joint_error(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return {
        "l2_rad": float(np.linalg.norm(diff)),
        "max_abs_rad": float(np.max(np.abs(diff))),
    }


def trajectory_stats(times: np.ndarray, qs: np.ndarray) -> dict[str, Any]:
    dq = np.diff(qs, axis=0)
    dt = np.diff(times)
    dt[dt <= 0.0] = np.nan
    qd = dq / dt[:, None]
    return {
        "waypoints": int(len(qs)),
        "duration_from_csv_s": float(times[-1] - times[0]),
        "max_abs_joint_step_rad": float(np.nanmax(np.abs(dq))) if len(dq) else 0.0,
        "max_abs_csv_qd_rad_s": float(np.nanmax(np.abs(qd))) if len(dq) else 0.0,
    }


def wait_for_goal(robot, q_goal: np.ndarray, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    last = np.asarray(robot.get_joint(), dtype=np.float64)
    while time.perf_counter() - started <= args.motion_timeout_s:
        now = time.perf_counter()
        last = np.asarray(robot.get_joint(), dtype=np.float64)
        err = joint_error(last, q_goal)
        samples.append(
            {
                "t_s": now - started,
                "actual_joint_rad": last.tolist(),
                "goal_l2_error_rad": err["l2_rad"],
                "goal_max_abs_error_rad": err["max_abs_rad"],
            }
        )
        if err["max_abs_rad"] <= args.goal_tolerance_rad:
            return (
                {
                    "reached": True,
                    "elapsed_s": now - started,
                    "goal_error": err,
                    "actual_joint_rad": last.tolist(),
                    "sample_count": len(samples),
                },
                samples,
            )
        time.sleep(args.poll_s)
    err = joint_error(last, q_goal)
    return (
        {
            "reached": False,
            "elapsed_s": time.perf_counter() - started,
            "goal_error": err,
            "actual_joint_rad": last.tolist(),
            "sample_count": len(samples),
        },
        samples,
    )


def require_enter(message: str) -> None:
    input(f"\n{message}\n确认后按 Enter；Ctrl-C 中止。")


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan_dir = args.plan_dir.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(plan_dir)
    times, qs = load_candidate(plan_dir)
    times_exec, qs_exec = resample_for_offline_track(
        times,
        qs,
        playback_duration_s=args.playback_duration_s,
        controller_period_s=args.controller_waypoint_period_s,
    )
    times_exec, qs_exec = maybe_downsample(times_exec, qs_exec, args.max_waypoints)
    accepted = bool(summary.get("accepted_for_real_execution", False))

    log: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "robot_commanded": False,
        "plan_dir": str(plan_dir),
        "plan_status": summary.get("status"),
        "accepted_for_real_execution": accepted,
        "candidate_min_distance_m": summary.get("candidate", {}).get("dense_verification", {}).get("min_distance"),
        "required_operator_phrase": REQUIRED_OPERATOR_PHRASE,
        "operator_phrase_ok": args.operator_phrase == REQUIRED_OPERATOR_PHRASE,
        "trajectory_csv_stats": trajectory_stats(times, qs),
        "execution_waypoint_stats": trajectory_stats(times_exec, qs_exec),
        "offline_track_timing_note": (
            "AUBO Offline Track may consume waypoints at a controller-side period. "
            "The script slows playback by resampling the geometric NUBS path into "
            "more waypoints according to playback_duration_s/controller_waypoint_period_s."
        ),
        "parameters": vars(args),
    }

    if not accepted:
        log["status"] = "BLOCKED_PLAN_NOT_ACCEPTED"
        write_json(output_dir / "offline_track_execution_log.json", log)
        print(json.dumps(log, indent=2, ensure_ascii=False, default=json_default))
        return log

    if not args.execute:
        log["status"] = "DRY_RUN_NO_ROBOT_COMMAND"
        log["q_start_rad"] = qs_exec[0].tolist()
        log["q_goal_rad"] = qs_exec[-1].tolist()
        write_json(output_dir / "offline_track_execution_log.json", log)
        print(json.dumps(log, indent=2, ensure_ascii=False, default=json_default))
        return log

    if args.operator_phrase != REQUIRED_OPERATOR_PHRASE:
        log["status"] = "BLOCKED_BAD_OPERATOR_PHRASE"
        write_json(output_dir / "offline_track_execution_log.json", log)
        raise RuntimeError(f"bad operator phrase; required: {REQUIRED_OPERATOR_PHRASE}")

    robot, so_path = load_robot_module(args.sdk_dir)
    log["sdk_so"] = str(so_path)
    print(f"[sdk] loaded: {so_path}")

    try:
        if not robot.init():
            raise RuntimeError("robot.init() failed")
        if not hasattr(robot, "offline_track_execute_joints"):
            raise RuntimeError("current robot .so does not expose offline_track_execute_joints")

        actual_start = np.asarray(robot.get_joint(), dtype=np.float64)
        start_err = joint_error(actual_start, qs_exec[0])
        log["actual_start_joint_rad"] = actual_start.tolist()
        log["start_error"] = start_err
        print(f"[current] {fmt_joints(actual_start.tolist())}")
        print(f"[target ] {fmt_joints(qs_exec[0].tolist())}")
        print(f"[start error] max_abs={start_err['max_abs_rad']:.5f} rad")
        if start_err["max_abs_rad"] > args.start_tolerance_rad:
            raise RuntimeError(
                f"current joints are not near trajectory start: {start_err['max_abs_rad']:.5f} rad "
                f"> {args.start_tolerance_rad:.5f} rad"
            )

        print("")
        print("[offline-track]")
        print(f"  waypoints: {len(qs_exec)}")
        if args.playback_duration_s > 0.0:
            print(f"  requested playback duration: {args.playback_duration_s:.3f}s")
            print(f"  assumed controller period: {args.controller_waypoint_period_s:.4f}s")
        print(f"  joint_velc limit: {args.joint_velc:.4f} rad/s")
        print(f"  joint_acc limit : {args.joint_acc:.4f} rad/s^2")
        print(f"  dense min clearance: {log['candidate_min_distance_m']}")
        print("  The obstacle must be unchanged from the accepted plan.")
        require_enter("Step 1/1: start AUBO Offline Track continuous CCRO-NUBS execution.")

        started = time.perf_counter()
        log["status"] = "PRE_COMMAND_READY"
        log["pre_command_note"] = "Operator confirmed; about to call offline_track_execute_joints."
        write_json(output_dir / "offline_track_execution_log.json", log)
        ret_info = robot.offline_track_execute_joints(
            qs_exec.tolist(),
            args.joint_velc,
            args.joint_acc,
            False,
            True,
            True,
        )
        log["robot_commanded"] = True
        log["offline_track_return"] = dict(ret_info)
        if int(ret_info.get("startup_ret", -9999)) != 0:
            raise RuntimeError(f"offline track startup failed: {ret_info}")

        goal_check, feedback_samples = wait_for_goal(robot, qs_exec[-1], args)
        log["goal_check"] = goal_check
        log["feedback_samples"] = feedback_samples
        log["elapsed_s"] = time.perf_counter() - started
        if not goal_check["reached"]:
            raise RuntimeError(f"offline track did not reach goal: {goal_check}")
        log["status"] = "COMPLETED_OFFLINE_TRACK_EXECUTION"
    except Exception as exc:
        log["status"] = "ABORTED_OR_FAILED"
        log["error"] = str(exc)
        try:
            if hasattr(robot, "offline_track_stop"):
                robot.offline_track_stop(True)
            elif hasattr(robot, "move_control_stop"):
                robot.move_control_stop(True)
        except Exception:
            pass
        raise
    finally:
        try:
            robot.log_out()
        except Exception:
            pass
        write_json(output_dir / "offline_track_execution_log.json", log)
        print(f"[log] {output_dir / 'offline_track_execution_log.json'}")

    return log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sdk-dir", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--operator-phrase", default="")
    parser.add_argument("--max-waypoints", type=int, default=0, help="0 means no cap after optional resampling")
    parser.add_argument(
        "--playback-duration-s",
        type=float,
        default=24.0,
        help="resample the path to this expected Offline Track playback duration; <=0 keeps CSV rows",
    )
    parser.add_argument(
        "--controller-waypoint-period-s",
        type=float,
        default=0.005,
        help="assumed controller-side waypoint consumption period used for resampling",
    )
    parser.add_argument("--joint-velc", type=float, default=0.02)
    parser.add_argument("--joint-acc", type=float, default=0.04)
    parser.add_argument("--start-tolerance-rad", type=float, default=0.035)
    parser.add_argument("--goal-tolerance-rad", type=float, default=0.035)
    parser.add_argument("--motion-timeout-s", type=float, default=90.0)
    parser.add_argument("--poll-s", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
