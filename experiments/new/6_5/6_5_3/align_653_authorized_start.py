#!/usr/bin/env python3
"""Guarded empty-scene alignment to an authorized 6.5.3 candidate start.

The robot may move only along a matched segment of the recorded 6.5.2
reference. Free-space movej positioning is intentionally not used.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

trial = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")

DEFAULT_TRAJECTORY = (
    ROOT
    / "results/new/6_5/6_5_3/dynamic_repair_formal/trials/D1_crossing_body_r01"
    / "local_execution_authorization/authorized_local_repair.csv"
)
DEFAULT_REFERENCE = ROOT / "results/new/6_5/6_5_3/reference_xp10_line/reference_feedback.csv"
DEFAULT_OUTPUT = ROOT / "results/new/6_5/6_5_3/empty_scene_local_track_calibration"
ALIGN_PHRASE = "CCRO_653_ALIGN_AUTHORIZED_START_APPROVED"


def execution_blocking_worktree_entries() -> list[str]:
    """Allow new result artifacts, but reject code/config worktree changes."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ["git_status_failed"]
    entries = [line for line in result.stdout.splitlines() if line.strip()]
    return [line for line in entries if not (line.startswith("?? ") and line[3:].startswith("results/"))]


def load_reference_joints(path: Path) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = np.asarray(
        [[float(row[f"q{i}_rad"]) for i in range(1, 7)] for row in rows],
        dtype=np.float64,
    )
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("reference feedback must contain at least two finite joint rows")
    return values


def matched_reference_segment(
    reference_q: np.ndarray,
    actual_q: np.ndarray,
    target_q: np.ndarray,
    *,
    match_tolerance_rad: float,
) -> tuple[np.ndarray, dict]:
    current_errors = np.max(np.abs(reference_q - actual_q[None, :]), axis=1)
    target_errors = np.max(np.abs(reference_q - target_q[None, :]), axis=1)
    current_index = int(np.argmin(current_errors))
    target_index = int(np.argmin(target_errors))
    if current_errors[current_index] > match_tolerance_rad:
        raise RuntimeError(
            f"current state is not on recorded reference: {current_errors[current_index]:.6f} rad"
        )
    if target_errors[target_index] > match_tolerance_rad:
        raise RuntimeError(
            f"authorized start is not on recorded reference: {target_errors[target_index]:.6f} rad"
        )
    step = 1 if target_index >= current_index else -1
    indices = np.arange(current_index, target_index + step, step, dtype=int)
    segment = reference_q[indices].copy()
    segment[0] = actual_q
    segment[-1] = target_q
    return segment, {
        "current_reference_index": current_index,
        "target_reference_index": target_index,
        "current_reference_error_rad": float(current_errors[current_index]),
        "target_reference_error_rad": float(target_errors[target_index]),
        "reference_direction": "forward" if step > 0 else "reverse",
        "source_waypoints": int(len(segment)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-csv", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--reference-feedback-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--alignment-duration-s", type=float, default=4.0)
    parser.add_argument("--controller-waypoint-period-s", type=float, default=0.005)
    parser.add_argument("--joint-velc", type=float, default=0.006)
    parser.add_argument("--joint-acc", type=float, default=0.012)
    parser.add_argument("--reference-match-tolerance-rad", type=float, default=0.01)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--operator-phrase", default="")
    return parser


def run(args: argparse.Namespace) -> dict:
    output_dir = args.output.resolve() / "start_alignment" / f"r{args.repeat:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    blocking_worktree_entries = execution_blocking_worktree_entries()
    log = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "INITIALIZING",
        "robot_commanded": False,
        "trajectory_csv": str(args.trajectory_csv.resolve()),
        "reference_feedback_csv": str(args.reference_feedback_csv.resolve()),
        "alignment_duration_s": float(args.alignment_duration_s),
        "joint_velc": float(args.joint_velc),
        "joint_acc": float(args.joint_acc),
        "git_commit": trial.git_commit_hash(),
        "git_dirty": trial.git_is_dirty(),
        "execution_blocking_worktree_entries": blocking_worktree_entries,
        "required_operator_phrase": ALIGN_PHRASE,
    }
    if not args.execute:
        log["status"] = "DRY_RUN_ONLY"
        trial.write_json(output_dir / "summary.json", log)
        return log
    if args.operator_phrase != ALIGN_PHRASE:
        log["status"] = "BLOCKED_BAD_OPERATOR_PHRASE"
        trial.write_json(output_dir / "summary.json", log)
        return log
    if blocking_worktree_entries:
        log["status"] = "BLOCKED_DIRTY_WORKTREE"
        trial.write_json(output_dir / "summary.json", log)
        return log

    _, candidate_q = trial.load_fast_candidate_csv(args.trajectory_csv.resolve())
    target_q = candidate_q[0]
    reference_q = load_reference_joints(args.reference_feedback_csv.resolve())
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
        start_error = trial.joint_error(actual_q, target_q)
        log["start_error"] = start_error
        if start_error["max_abs_rad"] <= runtime_args.candidate_start_tolerance_rad:
            log["status"] = "ALREADY_AT_AUTHORIZED_START"
            return log
        source_q, match = matched_reference_segment(
            reference_q,
            actual_q,
            target_q,
            match_tolerance_rad=args.reference_match_tolerance_rad,
        )
        source_times = np.linspace(0.0, args.alignment_duration_s, len(source_q))
        command_times, command_q = trial.resample_for_offline_track(
            source_times,
            source_q,
            playback_duration_s=args.alignment_duration_s,
            controller_period_s=args.controller_waypoint_period_s,
        )
        log["reference_match"] = match
        log["actual_start_joint_rad"] = actual_q.tolist()
        log["authorized_target_joint_rad"] = target_q.tolist()
        log["command_waypoint_stats"] = trial.trajectory_stats(command_times, command_q)
        denoiser = trial.TemporalDenoiser(
            runtime_args.denoise_voxel, runtime_args.denoise_conf, runtime_args.denoise_decay
        ) if runtime_args.temporal_denoise else None
        preview = [trial.execution_hard_guard_distance(processor, denoiser, runtime_args) for _ in range(3)]
        log["empty_scene_preview_guard_distance_m"] = preview
        if min(preview) <= runtime_args.guided_hard_stop_m:
            log["status"] = "BLOCKED_HARD_GUARD_PREVIEW"
            return log
        trial.require_confirmation(
            True,
            f"Empty-scene start alignment along recorded reference index "
            f"{match['current_reference_index']} -> {match['target_reference_index']}. "
            "Confirm emergency stop and clear workspace.",
        )
        ret = robot.offline_track_execute_joints(
            command_q.tolist(), args.joint_velc, args.joint_acc, False, True, True
        )
        log["robot_commanded"] = True
        log["offline_track_return"] = dict(ret)
        if int(ret.get("startup_ret", -9999)) != 0:
            raise RuntimeError(f"alignment Offline Track startup failed: {ret}")
        goal_check, feedback = trial.wait_for_candidate_goal_guarded(
            robot,
            target_q,
            processor=processor,
            denoiser=denoiser,
            args=runtime_args,
            goal_tolerance_rad=runtime_args.candidate_goal_tolerance_rad,
            min_execution_wait_s=0.90 * args.alignment_duration_s,
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
        log["status"] = "START_ALIGNMENT_PASS" if goal_check.get("reached") else "START_ALIGNMENT_FAIL"
    except Exception as exc:
        log["status"] = "START_ALIGNMENT_FAIL"
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
