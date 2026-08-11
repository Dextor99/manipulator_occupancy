#!/usr/bin/env python3
"""Empty-scene AUBO timing calibration for one authorized 6.5.3 local repair.

This utility never moves the robot to the candidate start. It executes only
when the current joints already satisfy the frozen start tolerance, the
workspace hard guard is clear, the repository is clean, and both explicit
operator gates are present.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


trial = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")
DEFAULT_TRAJECTORY = (
    ROOT
    / "results/new/6_5/6_5_3/dynamic_repair_pilot/trials/D1_crossing_body_r35"
    / "local_execution_authorization/authorized_local_repair.csv"
)
DEFAULT_OUTPUT = ROOT / "results/new/6_5/6_5_3/empty_scene_local_track_calibration"
CALIBRATION_PHRASE = "CCRO_653_EMPTY_SCENE_LOCAL_TRACK_APPROVED"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-csv", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--playback-duration-s", type=float, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--robot-ip", default="192.168.123.96")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    parser.add_argument("--urdf", type=Path, default=ROOT / "urdf/aubo_i16_gripper.urdf")
    parser.add_argument("--joint-velc", type=float, default=0.006)
    parser.add_argument("--joint-acc", type=float, default=0.012)
    parser.add_argument("--controller-waypoint-period-s", type=float, default=0.005)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--operator-phrase", default="")
    return parser


def run(args: argparse.Namespace) -> dict:
    trajectory_csv = args.trajectory_csv.resolve()
    label = f"playback_{args.playback_duration_s:.2f}s".replace(".", "p")
    output_dir = args.output.resolve() / "trials" / f"{label}_r{args.repeat:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "INITIALIZING",
        "robot_commanded": False,
        "trajectory_csv": str(trajectory_csv),
        "requested_duration_s": float(args.playback_duration_s),
        "repeat": int(args.repeat),
        "joint_velc": float(args.joint_velc),
        "joint_acc": float(args.joint_acc),
        "controller_waypoint_period_s": float(args.controller_waypoint_period_s),
        "git_commit": trial.git_commit_hash(),
        "git_dirty": trial.git_is_dirty(),
        "required_operator_phrase": CALIBRATION_PHRASE,
    }
    if not args.execute:
        log["status"] = "DRY_RUN_ONLY"
        trial.write_json(output_dir / "summary.json", log)
        return log
    if args.operator_phrase != CALIBRATION_PHRASE:
        log["status"] = "BLOCKED_BAD_OPERATOR_PHRASE"
        trial.write_json(output_dir / "summary.json", log)
        return log
    if log["git_dirty"]:
        log["status"] = "BLOCKED_DIRTY_WORKTREE"
        trial.write_json(output_dir / "summary.json", log)
        return log

    times, qs = trial.load_fast_candidate_csv(trajectory_csv)
    source_duration = float(times[-1] - times[0])
    log["authorized_csv_duration_s"] = source_duration
    if abs(source_duration - args.playback_duration_s) > 0.02:
        log["status"] = "BLOCKED_TIME_AXIS_MISMATCH"
        trial.write_json(output_dir / "summary.json", log)
        return log

    runtime_args = trial.build_parser().parse_args(["--scene", "D1"])
    runtime_args.robot_ip = args.robot_ip
    runtime_args.config_dir = args.config_dir.resolve()
    runtime_args.urdf = args.urdf.resolve()
    runtime_args.candidate_playback_duration_s = float(args.playback_duration_s)
    runtime_args.candidate_joint_velc = float(args.joint_velc)
    runtime_args.candidate_joint_acc = float(args.joint_acc)
    runtime_args.candidate_controller_waypoint_period_s = float(args.controller_waypoint_period_s)

    processor = None
    try:
        processor = trial.SceneProcessor(
            config_dir=str(runtime_args.config_dir),
            urdf_path=str(runtime_args.urdf),
            width=runtime_args.width,
            height=runtime_args.height,
            threshold=runtime_args.self_filter_threshold,
            voxel_size=runtime_args.voxel_size,
            use_real_robot=True,
            use_mock_camera=False,
        )
        state_reader = getattr(processor, "_state_reader", None)
        if state_reader is None or type(state_reader).__name__ != "RealRobotStateReader":
            raise RuntimeError("real AUBO state reader is required")
        robot = getattr(state_reader, "sdk_module", None)
        if robot is None:
            raise RuntimeError("AUBO SDK module is unavailable")
        denoiser = (
            trial.TemporalDenoiser(runtime_args.denoise_voxel, runtime_args.denoise_conf, runtime_args.denoise_decay)
            if runtime_args.temporal_denoise
            else None
        )
        actual_start = np.asarray(robot.get_joint(), dtype=np.float64)
        start_error = trial.joint_error(actual_start, qs[0])
        log["actual_start_joint_rad"] = actual_start.tolist()
        log["authorized_start_joint_rad"] = qs[0].tolist()
        log["start_error"] = start_error
        if start_error["max_abs_rad"] > runtime_args.candidate_start_tolerance_rad:
            log["status"] = "BLOCKED_START_MISMATCH"
            return log

        preview_distances = [
            trial.execution_hard_guard_distance(processor, denoiser, runtime_args)
            for _ in range(3)
        ]
        log["empty_scene_preview_guard_distance_m"] = preview_distances
        if min(preview_distances) <= runtime_args.guided_hard_stop_m:
            log["status"] = "BLOCKED_HARD_GUARD_PREVIEW"
            return log

        trial.require_confirmation(
            True,
            "Empty-scene timing calibration only. Confirm no obstacle/person is in the workspace, "
            "the emergency stop is ready, and the robot is already at the authorized start.",
        )
        execution = trial.execute_fast_candidate_offline_track(
            robot,
            trajectory_csv,
            runtime_args,
            processor=processor,
            denoiser=denoiser,
        )
        log["execution"] = execution
        log["robot_commanded"] = bool(execution.get("robot_commanded", False))
        log["status"] = "CALIBRATION_PASS" if execution.get("status") == "COMPLETED_DYNAMIC_CANDIDATE_EXECUTION" else "CALIBRATION_FAIL"
    except Exception as exc:
        log["status"] = "CALIBRATION_FAIL"
        log["error"] = str(exc)
        log["traceback"] = traceback.format_exc(limit=20)
        if processor is not None:
            state_reader = getattr(processor, "_state_reader", None)
            robot = None if state_reader is None else getattr(state_reader, "sdk_module", None)
            if robot is not None:
                log["stop_return"] = trial.maybe_move_stop(robot)
    finally:
        if processor is not None:
            processor.stop()
        trial.write_json(output_dir / "summary.json", log)
    return log


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False, default=trial.json_default))


if __name__ == "__main__":
    main()
