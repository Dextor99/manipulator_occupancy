#!/usr/bin/env python3
"""Guarded one-shot recovery from an executed delayed-rejoin endpoint to goal."""

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
calibration = importlib.import_module("experiments.new.6_5.6_5_3.calibrate_653_local_delayed_rejoin")
DEFAULT_SOURCE = ROOT / "results/new/6_5/6_5_3/local_delayed_rejoin_calibration/r05"
DEFAULT_OUTPUT = ROOT / "results/new/6_5/6_5_3/delayed_rejoin_resume_recovery"
PHRASE = "CCRO_653_DELAYED_REJOIN_RESUME_APPROVED"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-calibration", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--operator-phrase", default="")
    return parser


def run(args: argparse.Namespace) -> dict:
    source = args.source_calibration.resolve()
    source_summary_path = source / "summary.json"
    bridge_csv = source / "delayed_rejoin_authorization/authorized_delayed_rejoin_bridge.csv"
    if not source_summary_path.is_file() or not bridge_csv.is_file():
        raise FileNotFoundError("source calibration summary or authorized bridge is missing")
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    bridge_execution = source_summary.get("bridge_execution", {})
    if not bridge_execution.get("goal_check", {}).get("reached", False):
        raise RuntimeError("source bridge did not physically reach its authorized endpoint")
    _, bridge_q = trial.load_fast_candidate_csv(bridge_csv)
    target_q = bridge_q[-1]
    output_dir = args.output.resolve() / f"r{args.repeat:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    blocking_entries = calibration.execution_blocking_worktree_entries()
    log = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "INITIALIZED",
        "robot_commanded": False,
        "source_calibration": str(source),
        "source_status": source_summary.get("status"),
        "source_bridge_goal_check": bridge_execution.get("goal_check"),
        "authorized_rejoin_joint_rad": target_q.tolist(),
        "required_operator_phrase": PHRASE,
        "git_commit": trial.git_commit_hash(),
        "git_dirty": trial.git_is_dirty(),
        "execution_blocking_worktree_entries": blocking_entries,
    }
    if not args.execute:
        log["status"] = "DRY_RUN_READY"
        trial.write_json(output_dir / "summary.json", log)
        return log
    if args.operator_phrase != PHRASE:
        log["status"] = "BLOCKED_BAD_OPERATOR_PHRASE"
        trial.write_json(output_dir / "summary.json", log)
        return log
    if blocking_entries:
        log["status"] = "BLOCKED_DIRTY_WORKTREE"
        trial.write_json(output_dir / "summary.json", log)
        return log

    runtime_args = trial.build_parser().parse_args(["--scene", "D1", "--mode", "shadow"])
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
        endpoint_error = trial.joint_error(actual_q, target_q)
        log["actual_joint_rad"] = actual_q.tolist()
        log["rejoin_endpoint_error"] = endpoint_error
        if endpoint_error["max_abs_rad"] > runtime_args.candidate_start_tolerance_rad:
            log["status"] = "BLOCKED_REJOIN_ENDPOINT_MISMATCH"
            return log
        preview = [trial.execution_hard_guard_distance(processor, denoiser, runtime_args) for _ in range(3)]
        log["empty_scene_preview_guard_distance_m"] = preview
        if min(preview) <= runtime_args.guided_hard_stop_m:
            log["status"] = "BLOCKED_HARD_GUARD_PREVIEW"
            return log
        trial.require_confirmation(
            True,
            "Robot matches the executed delayed-rejoin endpoint. Confirm clear workspace and emergency stop, "
            "then press Enter for guarded Cartesian resume to the preset goal.",
        )
        log["remainder_execution"] = trial.execute_guarded_cartesian_reference_remainder(
            robot,
            runtime_args,
            processor=processor,
            denoiser=denoiser,
            target_y_m=runtime_args.y_goal,
        )
        log["robot_commanded"] = True
        log["status"] = (
            "DELAYED_REJOIN_RESUME_RECOVERY_PASS"
            if log["remainder_execution"].get("reached", False)
            else "DELAYED_REJOIN_RESUME_RECOVERY_FAIL"
        )
    except Exception as exc:
        log["status"] = "DELAYED_REJOIN_RESUME_RECOVERY_FAIL"
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
        trial.write_json(output_dir / "summary.json", log)
    return log


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False, default=trial.json_default))


if __name__ == "__main__":
    main()
