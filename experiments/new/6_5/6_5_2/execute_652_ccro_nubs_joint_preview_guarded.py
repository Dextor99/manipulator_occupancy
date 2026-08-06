#!/usr/bin/env python3
"""Guarded empty-workspace joint playback for a 6.5.2 CCRO-NUBS candidate.

This script is only for guarded low-speed visual playback of a planned
joint-space CCRO-NUBS candidate.  It supports two explicitly separated modes:

* empty-workspace preview after removing the obstacle;
* accepted static-obstacle playback after a plan has passed dense verification.

The AUBO pybind module used in this repository exposes ``movej(q)`` but does
not expose a bounded joint trajectory queue with explicit velocity/acceleration
limits.  Therefore the script is deliberately conservative:

* dry-run by default;
* refuses rejected plans unless ``--allow-rejected-empty-space-preview`` is set
  in empty-workspace preview mode;
* requires an exact operator phrase for either execution mode;
* checks that the current robot state is already near the trajectory start;
* downsamples the dense CSV to a small sequence of joint waypoints;
* logs every commanded waypoint and feedback sample.

Do not use static-obstacle playback if the obstacle has moved after planning.
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
    / "rs1_lateral_table_obstacle_r05"
    / "ccro_nubs_jointspace_plan"
)
DEFAULT_OUTPUT = ROOT / "results" / "new" / "6_5" / "6_5_2" / "jointspace_empty_preview"
REQUIRED_OPERATOR_PHRASE = "CCRO_652_EMPTY_PREVIEW_APPROVED"
STATIC_OPERATOR_PHRASE = "CCRO_652_STATIC_EXECUTION_APPROVED"


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
    path = plan_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_candidate_csv(plan_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = plan_dir / "ccro_nubs_candidate_trajectory.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    times: list[float] = []
    qs: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            times.append(float(row["t_s"]))
            qs.append([float(row[f"q{i}_rad"]) for i in range(1, 7)])
    if len(qs) < 2:
        raise RuntimeError(f"candidate trajectory has too few rows: {path}")
    return np.asarray(times, dtype=np.float64), np.asarray(qs, dtype=np.float64)


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def max_abs_step(qs: np.ndarray) -> float:
    if len(qs) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(qs, axis=0))))


def select_waypoints(times: np.ndarray, qs: np.ndarray, *, max_waypoints: int) -> tuple[np.ndarray, np.ndarray]:
    if max_waypoints < 2:
        raise ValueError("--max-waypoints must be >= 2")
    if len(qs) <= max_waypoints:
        return times, qs
    idx = np.linspace(0, len(qs) - 1, max_waypoints)
    idx = np.unique(np.round(idx).astype(int))
    if idx[0] != 0:
        idx = np.r_[0, idx]
    if idx[-1] != len(qs) - 1:
        idx = np.r_[idx, len(qs) - 1]
    return times[idx], qs[idx]


def wait_for_joints(robot, target: np.ndarray, *, timeout_s: float, poll_s: float, tolerance_rad: float) -> dict[str, Any]:
    started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    last = np.asarray(robot.get_joint(), dtype=np.float64)
    while time.perf_counter() - started <= timeout_s:
        last = np.asarray(robot.get_joint(), dtype=np.float64)
        err_l2 = l2(last, target)
        err_max = float(np.max(np.abs(last - target)))
        samples.append(
            {
                "t_s": time.perf_counter() - started,
                "joint_l2_error_rad": err_l2,
                "joint_max_error_rad": err_max,
                "actual_joint_rad": last.tolist(),
            }
        )
        if err_max <= tolerance_rad:
            return {
                "reached": True,
                "wait_s": time.perf_counter() - started,
                "joint_l2_error_rad": err_l2,
                "joint_max_error_rad": err_max,
                "actual_joint_rad": last.tolist(),
                "samples": samples,
            }
        time.sleep(poll_s)
    return {
        "reached": False,
        "wait_s": time.perf_counter() - started,
        "joint_l2_error_rad": l2(last, target),
        "joint_max_error_rad": float(np.max(np.abs(last - target))),
        "actual_joint_rad": last.tolist(),
        "samples": samples,
    }


def require_enter(enabled: bool, message: str) -> None:
    if not enabled:
        return
    input(f"\n{message}\n确认后按 Enter；Ctrl-C 中止。")


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan_dir = args.plan_dir.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(plan_dir)
    dense_accepted = bool(summary.get("accepted_for_real_execution", False))
    times, qs_dense = load_candidate_csv(plan_dir)
    times_pb, qs_pb = select_waypoints(times, qs_dense, max_waypoints=args.max_waypoints)

    log: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "robot_commanded": False,
        "plan_dir": str(plan_dir),
        "dense_accepted": dense_accepted,
        "plan_status": summary.get("status"),
        "candidate_min_distance_m": summary.get("candidate", {})
        .get("dense_verification", {})
        .get("min_distance"),
        "required_operator_phrase": REQUIRED_OPERATOR_PHRASE,
        "required_static_operator_phrase": STATIC_OPERATOR_PHRASE,
        "operator_phrase_ok": args.operator_phrase == REQUIRED_OPERATOR_PHRASE,
        "static_operator_phrase_ok": args.operator_phrase == STATIC_OPERATOR_PHRASE,
        "empty_workspace_preview": bool(args.empty_workspace_preview),
        "static_obstacle_execution": bool(args.static_obstacle_execution),
        "allow_rejected_empty_space_preview": bool(args.allow_rejected_empty_space_preview),
        "dense_rows": int(len(qs_dense)),
        "playback_waypoints": int(len(qs_pb)),
        "dense_max_abs_joint_step_rad": max_abs_step(qs_dense),
        "playback_max_abs_joint_step_rad": max_abs_step(qs_pb),
        "parameters": vars(args),
        "waypoints": [],
    }

    if args.empty_workspace_preview and args.static_obstacle_execution:
        log["status"] = "BLOCKED_CONFLICTING_EXECUTION_MODES"
        write_json(output_dir / "joint_preview_execution_log.json", log)
        raise RuntimeError("choose only one of --empty-workspace-preview or --static-obstacle-execution")

    if args.static_obstacle_execution and not dense_accepted:
        log["status"] = "BLOCKED_REJECTED_STATIC_PLAN"
        write_json(output_dir / "joint_preview_execution_log.json", log)
        raise RuntimeError("refusing static-obstacle execution because dense verifier did not accept the plan")

    if not dense_accepted and not args.allow_rejected_empty_space_preview:
        log["status"] = "BLOCKED_REJECTED_PLAN"
        log["note"] = (
            "This candidate did not pass the 6.5.2 obstacle-scene dense verifier. "
            "For visual inspection only, remove the obstacle and rerun with "
            "--empty-workspace-preview --allow-rejected-empty-space-preview."
        )
        write_json(output_dir / "joint_preview_execution_log.json", log)
        print(json.dumps(log, indent=2, ensure_ascii=False, default=json_default))
        return log

    if args.execute:
        if not (args.empty_workspace_preview or args.static_obstacle_execution):
            log["status"] = "BLOCKED_NO_EXECUTION_MODE"
            write_json(output_dir / "joint_preview_execution_log.json", log)
            raise RuntimeError("refusing execution unless an explicit execution mode is set")
        expected_phrase = STATIC_OPERATOR_PHRASE if args.static_obstacle_execution else REQUIRED_OPERATOR_PHRASE
        if args.operator_phrase != expected_phrase:
            log["status"] = "BLOCKED_BAD_OPERATOR_PHRASE"
            write_json(output_dir / "joint_preview_execution_log.json", log)
            raise RuntimeError(f"bad operator phrase; required: {expected_phrase}")
    else:
        log["status"] = "DRY_RUN_NO_ROBOT_COMMAND"
        log["q_start_rad"] = qs_pb[0].tolist()
        log["q_goal_rad"] = qs_pb[-1].tolist()
        write_json(output_dir / "joint_preview_execution_log.json", log)
        print(json.dumps(log, indent=2, ensure_ascii=False, default=json_default))
        return log

    robot, so_path = load_robot_module(args.sdk_dir)
    log["sdk_so"] = str(so_path)
    print(f"[sdk] loaded: {so_path}")
    try:
        if not robot.init():
            raise RuntimeError("robot.init() failed")

        actual_start = np.asarray(robot.get_joint(), dtype=np.float64)
        start_error = float(np.max(np.abs(actual_start - qs_pb[0])))
        log["actual_start_joint_rad"] = actual_start.tolist()
        log["start_max_abs_error_rad"] = start_error
        print(f"[current] {fmt_joints(actual_start.tolist())}")
        print(f"[target ] {fmt_joints(qs_pb[0].tolist())}")
        print(f"[start error] max_abs={start_error:.5f} rad")
        if start_error > args.start_tolerance_rad:
            raise RuntimeError(
                f"current joints are not near candidate start: {start_error:.5f} rad "
                f"> {args.start_tolerance_rad:.5f} rad. Move to the start pose first."
            )

        print("")
        print("[playback]")
        print(f"  waypoints: {len(qs_pb)}")
        print(f"  max abs joint step: {max_abs_step(qs_pb):.5f} rad")
        print(f"  candidate min distance in obstacle scene: {log['candidate_min_distance_m']}")
        if args.static_obstacle_execution:
            print("  MODE: accepted static-obstacle playback. The obstacle must be unchanged from planning.")
        else:
            print("  MODE: empty-workspace preview. The obstacle must be removed.")
        require_enter(
            True,
            "Step 1/1: guarded low-speed visual playback of the CCRO-NUBS joint path.",
        )

        log["robot_commanded"] = True
        for i, q in enumerate(qs_pb):
            if args.skip_first and i == 0:
                continue
            print(f"[movej] waypoint {i + 1}/{len(qs_pb)}  t={times_pb[i]:.3f}s  {fmt_joints(q.tolist())}")
            started = time.perf_counter()
            robot.movej(q.tolist())
            reach = wait_for_joints(
                robot,
                q,
                timeout_s=args.waypoint_timeout_s,
                poll_s=args.poll_s,
                tolerance_rad=args.joint_tolerance_rad,
            )
            entry = {
                "index": int(i),
                "trajectory_t_s": float(times_pb[i]),
                "target_joint_rad": q.tolist(),
                "move_elapsed_s": time.perf_counter() - started,
                "reach": reach,
            }
            log["waypoints"].append(entry)
            if not reach["reached"]:
                raise RuntimeError(
                    f"waypoint {i + 1}/{len(qs_pb)} not reached; "
                    f"max_error={reach['joint_max_error_rad']:.5f} rad"
                )
            time.sleep(args.settle_s)

        log["final_joint_rad"] = list(robot.get_joint())
        log["status"] = "COMPLETED_STATIC_OBSTACLE_EXECUTION" if args.static_obstacle_execution else "COMPLETED_EMPTY_WORKSPACE_PREVIEW"
    except Exception as exc:
        log["status"] = "ABORTED_OR_FAILED"
        log["error"] = str(exc)
        try:
            if hasattr(robot, "move_control_stop"):
                robot.move_control_stop()
        except Exception:
            pass
        raise
    finally:
        try:
            robot.log_out()
        except Exception:
            pass
        write_json(output_dir / "joint_preview_execution_log.json", log)
        print(f"[log] {output_dir / 'joint_preview_execution_log.json'}")
    return log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sdk-dir", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--operator-phrase", default="")
    parser.add_argument("--empty-workspace-preview", action="store_true")
    parser.add_argument("--static-obstacle-execution", action="store_true")
    parser.add_argument("--allow-rejected-empty-space-preview", action="store_true")
    parser.add_argument("--max-waypoints", type=int, default=21)
    parser.add_argument("--skip-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start-tolerance-rad", type=float, default=0.035)
    parser.add_argument("--joint-tolerance-rad", type=float, default=0.035)
    parser.add_argument("--waypoint-timeout-s", type=float, default=25.0)
    parser.add_argument("--poll-s", type=float, default=0.10)
    parser.add_argument("--settle-s", type=float, default=0.20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
