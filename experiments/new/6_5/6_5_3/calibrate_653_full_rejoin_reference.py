#!/usr/bin/env python3
"""Empty-scene execution check for an authorized repair+rejoin and reference remainder."""

from __future__ import annotations

import argparse
import csv
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
PHRASE = "CCRO_653_EMPTY_SCENE_FULL_REJOIN_REFERENCE_APPROVED"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trial", type=Path, default=DEFAULT_TRIAL)
    parser.add_argument("--reference-feedback-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=ROOT / "results/new/6_5/6_5_3/full_rejoin_reference_calibration")
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--operator-phrase", default="")
    return parser


def prepare_paths(args: argparse.Namespace) -> tuple[Path, Path, dict]:
    source = args.source_trial.resolve()
    full_csv = source / "post_plan_authorization/authorized_repair_rejoin.csv"
    authorization_path = source / "post_plan_authorization/authorization_summary.json"
    if not full_csv.is_file() or not authorization_path.is_file():
        raise FileNotFoundError("authorized repair+rejoin artifacts are incomplete")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if not authorization.get("execution_authorized", False):
        raise RuntimeError("source repair+rejoin was not execution-authorized")
    return source, full_csv, authorization


def recorded_trigger_reference_progress(source: Path) -> tuple[int, float]:
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    trigger_frame = int(summary["trigger_frame"])
    with (source / "frames.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["frame"]) == trigger_frame:
                return int(row["reference_index"]), float(row["reference_time_s"])
    raise RuntimeError(f"trigger frame {trigger_frame} is missing from frames.csv")


def run(args: argparse.Namespace) -> dict:
    output_dir = args.output.resolve() / f"r{args.repeat:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    source, full_csv, authorization = prepare_paths(args)
    reference = trial.RecordedReference.load(args.reference_feedback_csv.resolve())
    full_times, full_q = trial.load_fast_candidate_csv(full_csv)
    reference_start_index, reference_start_time = recorded_trigger_reference_progress(source)
    reference_start_match = float(np.max(np.abs(reference.q[reference_start_index] - full_q[0])))
    selected_offset = float(authorization["selected_rejoin_offset_s"])
    rejoin_match = trial.locate_authorized_rejoin_on_reference(reference, full_csv)
    rejoin_time = float(rejoin_match["time_s"])
    remainder_times, remainder_q, _ = reference.remainder_after(rejoin_time)
    remainder_csv = output_dir / "authorized_reference_remainder.csv"
    trial.save_joint_waypoint_csv(remainder_csv, remainder_times, remainder_q)
    endpoint_error = trial.joint_error(full_q[-1], remainder_q[0])
    log = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "INITIALIZED",
        "robot_commanded": False,
        "source_trial": str(source),
        "full_trajectory_csv": str(full_csv),
        "reference_feedback_csv": str(args.reference_feedback_csv.resolve()),
        "reference_start_index": reference_start_index,
        "reference_start_time_s": reference_start_time,
        "reference_start_match_max_rad": reference_start_match,
        "selected_rejoin_offset_s": selected_offset,
        "rejoin_reference_match": rejoin_match,
        "rejoin_absolute_reference_time_s": rejoin_time,
        "full_duration_s": float(full_times[-1] - full_times[0]),
        "remainder_duration_s": float(remainder_times[-1]),
        "full_to_remainder_endpoint_error": endpoint_error,
        "git_commit": trial.git_commit_hash(),
        "git_dirty": trial.git_is_dirty(),
        "required_operator_phrase": PHRASE,
    }
    if endpoint_error["max_abs_rad"] > 1.0e-4:
        log["status"] = "BLOCKED_REJOIN_ENDPOINT_MISMATCH"
        trial.write_json(output_dir / "summary.json", log)
        return log
    if not args.execute:
        log["status"] = "DRY_RUN_READY"
        trial.write_json(output_dir / "summary.json", log)
        return log
    if args.operator_phrase != PHRASE:
        log["status"] = "BLOCKED_BAD_OPERATOR_PHRASE"
        trial.write_json(output_dir / "summary.json", log)
        return log
    if log["git_dirty"]:
        log["status"] = "BLOCKED_DIRTY_WORKTREE"
        trial.write_json(output_dir / "summary.json", log)
        return log

    runtime_args = trial.build_parser().parse_args(
        ["--scene", "D1", "--candidate-playback-duration-s", "1.0"]
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
        denoiser = trial.TemporalDenoiser(
            runtime_args.denoise_voxel, runtime_args.denoise_conf, runtime_args.denoise_decay
        ) if runtime_args.temporal_denoise else None
        preview = [trial.execution_hard_guard_distance(processor, denoiser, runtime_args) for _ in range(3)]
        log["empty_scene_preview_guard_distance_m"] = preview
        if min(preview) <= runtime_args.guided_hard_stop_m:
            log["status"] = "BLOCKED_HARD_GUARD_PREVIEW"
            return log
        log["full_execution"] = trial.execute_authorized_trajectory_offline_track(
            robot,
            full_csv,
            runtime_args,
            processor=processor,
            denoiser=denoiser,
            playback_duration_s=None,
            execution_label="empty-scene authorized repair + rejoin",
        )
        log["robot_commanded"] = True
        log["remainder_execution"] = trial.execute_authorized_trajectory_offline_track(
            robot,
            remainder_csv,
            runtime_args,
            processor=processor,
            denoiser=denoiser,
            playback_duration_s=None,
            controller_period_s=reference.dt_median,
            execution_label="empty-scene authorized reference remainder",
        )
        log["status"] = "FULL_REJOIN_REFERENCE_CALIBRATION_PASS"
    except Exception as exc:
        log["status"] = "FULL_REJOIN_REFERENCE_CALIBRATION_FAIL"
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
