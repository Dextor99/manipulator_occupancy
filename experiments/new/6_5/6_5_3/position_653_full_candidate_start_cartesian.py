#!/usr/bin/env python3
"""Guarded Cartesian positioning to an authorized full candidate start."""

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

DEFAULT_TRIAL = ROOT / "results/new/6_5/6_5_3/dynamic_repair_formal/trials/D1_crossing_body_r01"
DEFAULT_REFERENCE = ROOT / "results/new/6_5/6_5_3/reference_xp10_line/reference_feedback.csv"
PHRASE = "CCRO_653_EMPTY_SCENE_POSITION_FULL_START_APPROVED"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trial", type=Path, default=DEFAULT_TRIAL)
    parser.add_argument("--reference-feedback-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=ROOT / "results/new/6_5/6_5_3/full_candidate_start_positioning")
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--operator-phrase", default="")
    return parser


def run(args: argparse.Namespace) -> dict:
    entry_commit = trial.git_commit_hash()
    entry_dirty = trial.git_is_dirty()
    output_dir = args.output.resolve() / f"r{args.repeat:02d}"
    full_csv = args.source_trial.resolve() / "post_plan_authorization/authorized_repair_rejoin.csv"
    times, full_q = trial.load_fast_candidate_csv(full_csv)
    reference = trial.RecordedReference.load(args.reference_feedback_csv.resolve())
    errors = np.max(np.abs(reference.q - full_q[0][None, :]), axis=1)
    target_index = int(np.argmin(errors))
    target_y = float(reference.y[target_index])
    log = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "INITIALIZED",
        "robot_commanded": False,
        "full_trajectory_csv": str(full_csv),
        "target_reference_index": target_index,
        "target_reference_time_s": float(reference.times[target_index]),
        "target_y_m": target_y,
        "reference_to_candidate_start_max_rad": float(errors[target_index]),
        "git_commit": entry_commit,
        "git_dirty": entry_dirty,
        "required_operator_phrase": PHRASE,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.execute:
        log["status"] = "DRY_RUN_READY"
        trial.write_json(output_dir / "summary.json", log)
        return log
    if args.operator_phrase != PHRASE:
        log["status"] = "BLOCKED_BAD_OPERATOR_PHRASE"
        trial.write_json(output_dir / "summary.json", log)
        return log
    if entry_dirty:
        log["status"] = "BLOCKED_DIRTY_WORKTREE"
        trial.write_json(output_dir / "summary.json", log)
        return log

    runtime_args = trial.build_parser().parse_args(["--scene", "D1", "--candidate-playback-duration-s", "1.0"])
    processor = None
    try:
        processor = trial.SceneProcessor(
            config_dir=str(runtime_args.config_dir), urdf_path=str(runtime_args.urdf),
            width=runtime_args.width, height=runtime_args.height,
            threshold=runtime_args.self_filter_threshold, voxel_size=runtime_args.voxel_size,
            use_real_robot=True, use_mock_camera=False,
        )
        reader = getattr(processor, "_state_reader", None)
        if reader is None or type(reader).__name__ != "RealRobotStateReader":
            raise RuntimeError("real AUBO state reader is required")
        robot = getattr(reader, "sdk_module", None)
        denoiser = trial.TemporalDenoiser(
            runtime_args.denoise_voxel, runtime_args.denoise_conf, runtime_args.denoise_decay
        ) if runtime_args.temporal_denoise else None
        log["motion"] = trial.execute_guarded_cartesian_reference_remainder(
            robot, runtime_args, processor=processor, denoiser=denoiser, target_y_m=target_y
        )
        log["robot_commanded"] = True
        actual_q = np.asarray(robot.get_joint(), dtype=np.float64)
        start_error = trial.joint_error(actual_q, full_q[0])
        log["candidate_start_error"] = start_error
        log["status"] = (
            "FULL_CANDIDATE_START_POSITIONED"
            if log["motion"].get("reached", False)
            and start_error["max_abs_rad"] <= runtime_args.candidate_start_tolerance_rad
            else "FULL_CANDIDATE_START_POSITIONING_FAIL"
        )
    except Exception as exc:
        log["status"] = "FULL_CANDIDATE_START_POSITIONING_FAIL"
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
