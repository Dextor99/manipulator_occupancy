#!/usr/bin/env python3
"""Guarded empty-scene return along an authorized local candidate in reverse."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

trial = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")

DEFAULT_TRAJECTORY = (
    ROOT
    / "results/new/6_5/6_5_3/dynamic_repair_pilot/trials/D1_crossing_body_r35"
    / "local_execution_authorization/authorized_local_repair.csv"
)
DEFAULT_OUTPUT = ROOT / "results/new/6_5/6_5_3/empty_scene_local_track_calibration"
RETURN_PHRASE = "CCRO_653_RETURN_AUTHORIZED_LOCAL_TRACK_APPROVED"


def reverse_authorized_waypoints(
    times: np.ndarray,
    qs: np.ndarray,
    *,
    return_duration_s: float,
    controller_period_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    if return_duration_s <= 0.0:
        raise ValueError("return duration must be positive")
    source_times = np.linspace(0.0, return_duration_s, len(qs))
    return trial.resample_for_offline_track(
        source_times,
        qs[::-1].copy(),
        playback_duration_s=return_duration_s,
        controller_period_s=controller_period_s,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-csv", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--return-duration-s", type=float, default=1.0)
    parser.add_argument("--controller-waypoint-period-s", type=float, default=0.005)
    parser.add_argument("--joint-velc", type=float, default=0.006)
    parser.add_argument("--joint-acc", type=float, default=0.012)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--operator-phrase", default="")
    return parser


def run(args: argparse.Namespace) -> dict:
    output_dir = args.output.resolve() / "candidate_return" / f"r{args.repeat:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_csv = args.trajectory_csv.resolve()
    log = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "INITIALIZING",
        "robot_commanded": False,
        "trajectory_csv": str(trajectory_csv),
        "return_duration_s": float(args.return_duration_s),
        "joint_velc": float(args.joint_velc),
        "joint_acc": float(args.joint_acc),
        "controller_waypoint_period_s": float(args.controller_waypoint_period_s),
        "git_commit": trial.git_commit_hash(),
        "git_dirty": trial.git_is_dirty(),
        "required_operator_phrase": RETURN_PHRASE,
    }
    if not args.execute:
        log["status"] = "DRY_RUN_ONLY"
        trial.write_json(output_dir / "summary.json", log)
        return log
    if args.operator_phrase != RETURN_PHRASE:
        log["status"] = "BLOCKED_BAD_OPERATOR_PHRASE"
        trial.write_json(output_dir / "summary.json", log)
        return log
    if log["git_dirty"]:
        log["status"] = "BLOCKED_DIRTY_WORKTREE"
        trial.write_json(output_dir / "summary.json", log)
        return log

    times, qs = trial.load_fast_candidate_csv(trajectory_csv)
    command_times, command_q = reverse_authorized_waypoints(
        times,
        qs,
        return_duration_s=args.return_duration_s,
        controller_period_s=args.controller_waypoint_period_s,
    )
    runtime_args = trial.build_parser().parse_args(["--scene", "D1"])
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
        actual_q = np.asarray(robot.get_joint(), dtype=np.float64)
        endpoint_error = trial.joint_error(actual_q, qs[-1])
        log["actual_start_joint_rad"] = actual_q.tolist()
        log["authorized_endpoint_joint_rad"] = qs[-1].tolist()
        log["endpoint_start_error"] = endpoint_error
        if endpoint_error["max_abs_rad"] > runtime_args.candidate_start_tolerance_rad:
            log["status"] = "BLOCKED_ENDPOINT_MISMATCH"
            return log

        denoiser = trial.TemporalDenoiser(
            runtime_args.denoise_voxel, runtime_args.denoise_conf, runtime_args.denoise_decay
        ) if runtime_args.temporal_denoise else None
        preview = [trial.execution_hard_guard_distance(processor, denoiser, runtime_args) for _ in range(3)]
        log["empty_scene_preview_guard_distance_m"] = preview
        if min(preview) <= runtime_args.guided_hard_stop_m:
            log["status"] = "BLOCKED_HARD_GUARD_PREVIEW"
            return log
        log["command_waypoint_stats"] = trial.trajectory_stats(command_times, command_q)
        trial.require_confirmation(
            True,
            "Empty-scene return along the exact authorized candidate in reverse. "
            "Confirm clear workspace and emergency stop.",
        )
        ret = robot.offline_track_execute_joints(
            command_q.tolist(), args.joint_velc, args.joint_acc, False, True, True
        )
        log["robot_commanded"] = True
        log["offline_track_return"] = dict(ret)
        if int(ret.get("startup_ret", -9999)) != 0:
            raise RuntimeError(f"reverse Offline Track startup failed: {ret}")
        goal_check, feedback = trial.wait_for_candidate_goal_guarded(
            robot,
            qs[0],
            processor=processor,
            denoiser=denoiser,
            args=runtime_args,
            goal_tolerance_rad=runtime_args.candidate_goal_tolerance_rad,
            min_execution_wait_s=0.90 * args.return_duration_s,
            motion_timeout_s=runtime_args.candidate_motion_timeout_s,
            poll_s=runtime_args.poll_s,
            min_motion_rad=runtime_args.candidate_min_observed_motion_rad,
        )
        log["goal_check"] = goal_check
        log["feedback_samples"] = feedback
        log["tracking_metrics"] = trial.candidate_tracking_metrics(
            command_times,
            command_q,
            feedback,
            minimum_motion_rad=runtime_args.candidate_min_observed_motion_rad,
        )
        log["hold_return"] = trial.maybe_move_stop(robot)
        log["status"] = "CANDIDATE_RETURN_PASS" if goal_check.get("reached") else "CANDIDATE_RETURN_FAIL"
    except Exception as exc:
        log["status"] = "CANDIDATE_RETURN_FAIL"
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
