#!/usr/bin/env python3
"""Safely resume the recorded reference from an executed 6.5.3 rejoin hold."""

from __future__ import annotations

import argparse
import copy
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
calibration = importlib.import_module("experiments.new.6_5.6_5_3.calibrate_653_local_delayed_rejoin")

DEFAULT_SOURCE = (
    ROOT
    / "results/new/6_5/6_5_3/dynamic_repair_rolling_live_xp10"
    / "trials/D2_opposing_approach_r04"
)
PHRASE = "CCRO_653_REJOIN_HOLD_RESUME_APPROVED"
REJOIN_STATE_TOLERANCE_RAD = 0.020


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trial", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: <source-trial>/recovery_from_resume_hold",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--operator-phrase", default="")
    return parser


def load_source(source: Path) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    summary_path = source / "summary.json"
    execution_path = source / "candidate/live_full_candidate_execution_log.json"
    resume_path = source / "reference_resume_authorization.json"
    remainder_path = source / "authorized_reference_remainder.csv"
    for path in (summary_path, execution_path, resume_path, remainder_path):
        if not path.is_file():
            raise FileNotFoundError(f"required r04 recovery artifact is missing: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    resume = json.loads(resume_path.read_text(encoding="utf-8"))
    if summary.get("status") != "TRIGGERED_AND_REPAIR_REJOIN_EXECUTED_HOLD":
        raise RuntimeError(f"source trial is not an executed rejoin hold: {summary.get('status')}")
    if execution.get("status") != "COMPLETED_AUTHORIZED_TRAJECTORY_EXECUTION":
        raise RuntimeError("source repair+rejoin did not complete its authorized execution")
    if not execution.get("goal_check", {}).get("reached", False):
        raise RuntimeError("source robot did not physically reach the authorized rejoin endpoint")
    if resume.get("status") != "REFERENCE_RESUME_HOLD" or resume.get("authorized", False):
        raise RuntimeError("source reference remainder is not in a fail-closed HOLD state")
    remainder_t, remainder_q = trial.load_fast_candidate_csv(remainder_path)
    return summary, resume, remainder_t, remainder_q


def run(args: argparse.Namespace) -> dict:
    source = args.source_trial.resolve()
    output_dir = (
        (source / "recovery_from_resume_hold")
        if args.output is None
        else args.output.resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "recovery_summary.json"
    source_summary, source_resume, remainder_t, remainder_q = load_source(source)
    blocking_entries = calibration.execution_blocking_worktree_entries()
    log = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "INITIALIZED",
        "robot_commanded": False,
        "source_trial": str(source),
        "source_trial_git_commit": source_summary.get("git_commit"),
        "source_archive_commit": "31f5272",
        "source_status": source_summary.get("status"),
        "source_resume_status": source_resume.get("status"),
        "runtime_git_commit": trial.git_commit_hash(),
        "runtime_git_dirty": trial.git_is_dirty(),
        "execution_blocking_worktree_entries": blocking_entries,
        "required_operator_phrase": PHRASE,
        "rejoin_joint_rad": remainder_q[0].tolist(),
        "remainder_duration_s": float(remainder_t[-1]),
        "goal_y_m": -0.40,
    }
    if not args.execute:
        log["status"] = "DRY_RUN_READY"
        trial.write_json(summary_path, log)
        return log
    if args.operator_phrase != PHRASE:
        log["status"] = "BLOCKED_BAD_OPERATOR_PHRASE"
        trial.write_json(summary_path, log)
        return log
    if blocking_entries:
        log["status"] = "BLOCKED_DIRTY_WORKTREE"
        trial.write_json(summary_path, log)
        return log

    runtime_args = trial.build_parser().parse_args(
        [
            "--scene", "D2",
            "--mode", "shadow",
            "--x-offset", "0.10",
            "--y-start", "0.40",
            "--y-goal", "-0.40",
            "--line-velocity-m-s", "0.020",
            "--line-acc-m-s2", "0.05",
        ]
    )
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
        reader = getattr(processor, "_state_reader", None)
        if reader is None or type(reader).__name__ != "RealRobotStateReader":
            raise RuntimeError("real AUBO state reader is required")
        robot = getattr(reader, "sdk_module", None)
        if robot is None:
            raise RuntimeError("real AUBO SDK connection is required")
        denoiser = (
            trial.TemporalDenoiser(
                runtime_args.denoise_voxel,
                runtime_args.denoise_conf,
                runtime_args.denoise_decay,
            )
            if runtime_args.temporal_denoise
            else None
        )

        actual_q = np.asarray(robot.get_joint(), dtype=np.float64)
        state_error = trial.joint_error(actual_q, remainder_q[0])
        state_matches = bool(state_error["max_abs_rad"] <= REJOIN_STATE_TOLERANCE_RAD)
        state_check = {
            "status": (
                "REJOIN_STATE_MATCH"
                if state_matches
                else "CURRENT_STATE_NOT_AT_REJOIN"
            ),
            "accepted": state_matches,
            "tolerance_rad": REJOIN_STATE_TOLERANCE_RAD,
            "actual_joint_rad": actual_q.tolist(),
            "expected_rejoin_joint_rad": remainder_q[0].tolist(),
            "joint_error": state_error,
        }
        log["pre_resume_state_check"] = state_check
        trial.write_json(output_dir / "pre_resume_state_check.json", state_check)
        if not state_check["accepted"]:
            log["status"] = "CURRENT_STATE_NOT_AT_REJOIN"
            return log

        old_fresh = json.loads((source / "fresh3_recheck.json").read_text(encoding="utf-8"))["result"]
        capture_args = copy.copy(runtime_args)
        capture_args.post_stop_recheck_duration_s = max(
            1.0, float(runtime_args.post_stop_recheck_duration_s)
        )
        _, scene_frames, _ = trial.capture_post_stop_obstacle(
            processor,
            reader,
            denoiser,
            capture_args,
            trigger_cluster_center=np.asarray(old_fresh["center"], dtype=np.float64),
            trigger_velocity=np.asarray(old_fresh["velocity"], dtype=np.float64),
            trigger_timestamp=float(old_fresh["last_timestamp"]),
            stop_when_ready=False,
        )
        stage4_config = trial.load_stage4_config(runtime_args.stage4_config)
        stage4_model = trial.load_stage4_surface_model(stage4_config)
        scene_clear = trial.authorize_fresh3_scene_clear(
            runtime_args,
            stage4_model,
            fresh3_frames=scene_frames,
            remainder_times=remainder_t,
            remainder_q=remainder_q,
        )
        scene_recheck = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "frame_count": len(scene_frames),
            "frames": scene_frames,
            "scene_clear_audit": scene_clear,
        }
        log["fresh_scene_clear_recheck"] = scene_clear
        trial.write_json(output_dir / "fresh_scene_clear_recheck.json", scene_recheck)
        if not scene_clear["accepted"]:
            log["status"] = "FRESH3_SCENE_NOT_CLEAR_HOLD"
            return log

        trial.require_confirmation(
            True,
            "Rejoin state and three-frame scene-clear audit passed. Confirm emergency stop, "
            "then press Enter to execute the guarded reference remainder to Y=-0.40 m.",
        )
        remainder_execution = trial.execute_guarded_cartesian_reference_remainder(
            robot,
            runtime_args,
            processor=processor,
            denoiser=denoiser,
            target_y_m=runtime_args.y_goal,
        )
        log["robot_commanded"] = True
        log["reference_remainder_execution"] = remainder_execution
        trial.write_json(
            output_dir / "reference_remainder_execution_log.json", remainder_execution
        )
        reached = bool(remainder_execution.get("reached", False))
        log["status"] = (
            "RECOVERED_FROM_REJOIN_HOLD_AND_GOAL_REACHED"
            if reached
            else "RECOVERY_REMAINDER_STOPPED_OR_FAILED"
        )
        log["rejoin_state_match"] = True
        log["scene_clear"] = True
        log["reference_remainder_executed"] = True
        log["goal_reached"] = reached
        log["minimum_guard_distance_m"] = remainder_execution.get(
            "minimum_hard_guard_distance_m"
        )
    except Exception as exc:
        log["status"] = "RECOVERY_FAILED"
        log["error"] = str(exc)
        log["traceback"] = traceback.format_exc(limit=20)
        if processor is not None:
            reader = getattr(processor, "_state_reader", None)
            robot = None if reader is None else getattr(reader, "sdk_module", None)
            if robot is not None:
                log["stop_return"] = trial.maybe_move_stop(robot)
    finally:
        if processor is not None:
            processor.stop()
        trial.write_json(summary_path, log)
    return log


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False, default=trial.json_default))


if __name__ == "__main__":
    main()
