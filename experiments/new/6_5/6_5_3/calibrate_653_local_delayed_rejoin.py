#!/usr/bin/env python3
"""Empty-scene calibration of the v2 local-first delayed-rejoin state machine."""

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
DEFAULT_SOURCE = ROOT / "results/new/6_5/6_5_3/dynamic_repair_formal/trials/D1_crossing_body_r01"
DEFAULT_REFERENCE = ROOT / "results/new/6_5/6_5_3/reference_xp10_line/reference_feedback.csv"
DEFAULT_OUTPUT = ROOT / "results/new/6_5/6_5_3/local_delayed_rejoin_calibration"
PHRASE = "CCRO_653_EMPTY_SCENE_LOCAL_DELAYED_REJOIN_APPROVED"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trial", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reference-feedback-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--operator-phrase", default="")
    return parser


def recorded_trigger_reference_progress(source: Path) -> tuple[int, float]:
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    trigger_frame = int(summary["trigger_frame"])
    with (source / "frames.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["frame"]) == trigger_frame:
                return int(row["reference_index"]), float(row["reference_time_s"])
    raise RuntimeError(f"trigger frame {trigger_frame} is missing from frames.csv")


def load_source(source: Path, runtime_args: argparse.Namespace):
    local_csv = source / "local_execution_authorization/authorized_local_repair.csv"
    authorization_path = source / "local_execution_authorization/authorization_summary.json"
    fresh_path = source / "post_plan_fresh_recheck.json"
    for path in (local_csv, authorization_path, fresh_path, source / "summary.json", source / "frames.csv"):
        if not path.is_file():
            raise FileNotFoundError(path)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if not authorization.get("local_execution_authorized", False):
        raise RuntimeError("source local repair was not Fresh #2 execution-authorized")
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))["result"]
    repair = trial.reconstruct_saved_nubs_candidate(local_csv, segments=runtime_args.local_segments)
    return local_csv, authorization, fresh, repair


def run(args: argparse.Namespace) -> dict:
    output_dir = args.output.resolve() / f"r{args.repeat:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    source = args.source_trial.resolve()
    reference_path = args.reference_feedback_csv.resolve()
    runtime_args = trial.build_parser().parse_args(
        ["--scene", "D1", "--mode", "shadow", "--candidate-playback-duration-s", "1.0"]
    )
    local_csv, source_authorization, source_fresh, repair = load_source(source, runtime_args)
    reference = trial.RecordedReference.load(reference_path)
    reference_index, reference_time = recorded_trigger_reference_progress(source)
    reference.index = reference_index
    rejoin_offsets = np.arange(
        runtime_args.local_horizon_s + runtime_args.rejoin_search_step_s,
        runtime_args.rejoin_max_offset_s + 0.5 * runtime_args.rejoin_search_step_s,
        runtime_args.rejoin_search_step_s,
    )
    rejoin_goals = [(float(offset), reference.state_after(float(offset))) for offset in rejoin_offsets]
    local_times, local_q = trial.load_fast_candidate_csv(local_csv)
    log = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "INITIALIZED",
        "robot_commanded": False,
        "formal_protocol_id": trial.FORMAL_PROTOCOL_ID,
        "calibration_path": "LOCAL_FIRST_DELAYED_REJOIN",
        "source_trial": str(source),
        "source_local_authorization": source_authorization,
        "source_local_csv": str(local_csv),
        "reference_feedback_csv": str(reference_path),
        "reference_index": reference_index,
        "reference_time_s": reference_time,
        "local_duration_s": float(local_times[-1] - local_times[0]),
        "required_operator_phrase": PHRASE,
        "git_commit": trial.git_commit_hash(),
        "git_dirty": trial.git_is_dirty(),
    }
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

    processor = None
    try:
        stage4_config = trial.load_stage4_config(runtime_args.stage4_config)
        stage4_model = trial.load_stage4_surface_model(stage4_config)
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
        preview = [trial.execution_hard_guard_distance(processor, denoiser, runtime_args) for _ in range(3)]
        log["empty_scene_preview_guard_distance_m"] = preview
        if min(preview) <= runtime_args.guided_hard_stop_m:
            log["status"] = "BLOCKED_HARD_GUARD_PREVIEW"
            return log
        actual_start = np.asarray(robot.get_joint(), dtype=np.float64)
        start_error = trial.joint_error(actual_start, local_q[0])
        log["actual_start_joint_rad"] = actual_start.tolist()
        log["authorized_start_joint_rad"] = local_q[0].tolist()
        log["start_error"] = start_error
        if start_error["max_abs_rad"] > runtime_args.candidate_start_tolerance_rad:
            log["status"] = "BLOCKED_START_MISMATCH"
            return log

        log["local_execution"] = trial.execute_authorized_trajectory_offline_track(
            robot,
            local_csv,
            runtime_args,
            processor=processor,
            denoiser=denoiser,
            playback_duration_s=None,
            execution_label="empty-scene Fresh #2-authorized local repair",
        )
        log["robot_commanded"] = True
        if log["local_execution"]["status"] != "COMPLETED_AUTHORIZED_TRAJECTORY_EXECUTION":
            raise RuntimeError(f"local execution timing failed: {log['local_execution'].get('timing_check')}")
        log["hold_reached"] = True

        fresh3, fresh3_frames, fresh3_points = trial.capture_post_stop_obstacle(
            processor,
            reader,
            denoiser,
            runtime_args,
            trigger_cluster_center=np.asarray(source_fresh["center"], dtype=np.float64),
            trigger_velocity=np.asarray(source_fresh["velocity"], dtype=np.float64),
            trigger_timestamp=float(source_fresh["last_timestamp"]),
            stop_when_ready=False,
        )
        fresh3_geometry = None
        if fresh3.get("accepted", False) and fresh3_points is not None:
            fresh3_geometry = trial.fit_pca_multisphere(
                fresh3_points,
                fit_margin_m=runtime_args.multisphere_fit_margin_m,
                max_components=runtime_args.multisphere_max_components,
            )
            if not fresh3_geometry["covered"]:
                fresh3 = {**fresh3, "accepted": False, "reason": "fresh3_multisphere_coverage_failed"}
        trial.write_json(output_dir / "fresh3_recheck.json", {"result": fresh3, "frames": fresh3_frames})
        if fresh3_geometry is not None:
            trial.write_json(output_dir / "fresh3_multisphere.json", fresh3_geometry)

        fresh3_guard = trial.execution_hard_guard_distance(processor, denoiser, runtime_args)
        delayed, _ = trial.authorize_delayed_rejoin_after_fresh3(
            runtime_args,
            stage4_config,
            stage4_model,
            local_artifacts={"candidate_trajectory": repair},
            fresh3=fresh3,
            fresh3_geometry=fresh3_geometry,
            fresh3_frames=fresh3_frames,
            rejoin_goals=rejoin_goals,
            hard_guard_distance_m=fresh3_guard,
            trial_dir=output_dir,
        )
        log["delayed_rejoin_authorization"] = delayed
        if not delayed["authorized"]:
            log["status"] = "HOLD_DELAYED_REJOIN_NOT_AUTHORIZED"
            return log
        bridge_csv = Path(delayed["authorized_trajectory_csv"])
        log["bridge_execution"] = trial.execute_authorized_trajectory_offline_track(
            robot,
            bridge_csv,
            runtime_args,
            processor=processor,
            denoiser=denoiser,
            playback_duration_s=None,
            execution_label="empty-scene Fresh #3-authorized delayed C2 bridge",
        )
        if log["bridge_execution"]["status"] != "COMPLETED_AUTHORIZED_TRAJECTORY_EXECUTION":
            raise RuntimeError(f"bridge execution timing failed: {log['bridge_execution'].get('timing_check')}")
        log["remainder_execution"] = trial.execute_guarded_cartesian_reference_remainder(
            robot,
            runtime_args,
            processor=processor,
            denoiser=denoiser,
            target_y_m=runtime_args.y_goal,
        )
        log["status"] = (
            "LOCAL_DELAYED_REJOIN_CALIBRATION_PASS"
            if log["remainder_execution"].get("reached", False)
            else "LOCAL_DELAYED_REJOIN_CALIBRATION_FAIL"
        )
    except Exception as exc:
        log["status"] = "LOCAL_DELAYED_REJOIN_CALIBRATION_FAIL"
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
