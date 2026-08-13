#!/usr/bin/env python3
"""Real-RGB-D, virtual-robot audit for bounded rolling-local Fast repair."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

trial = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")
DEFAULT_SOURCE = (
    ROOT / "results/new/6_5/6_5_3/dynamic_repair_rolling_live_xp10"
    / "trials/D2_opposing_approach_r04"
)
DEFAULT_REFERENCE = ROOT / "results/new/6_5/6_5_3/reference_xp10_line/reference_feedback.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trial", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reference-feedback-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/new/6_5/6_5_3/rolling_local_virtual_shadow",
    )
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--max-segments", type=int, default=3)
    parser.add_argument("--max-wall-s", type=float, default=10.0)
    parser.add_argument("--seed-timeout-s", type=float, default=8.0)
    return parser


def first_external_seed(
    processor: Any,
    denoiser: Any,
    runtime_args: argparse.Namespace,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    """Find one external cluster seed without assuming its old r04 location."""
    started = time.perf_counter()
    frames = []
    while time.perf_counter() - started < timeout_s:
        frame = processor.process_frame()
        timestamp = float(getattr(frame, "timestamp", time.time()))
        scene = np.asarray(frame.scene_points, dtype=np.float64)
        robot = np.asarray(frame.robot_points, dtype=np.float64)
        valid = bool(
            scene.ndim == 2 and scene.shape[1:] == (3,) and len(scene) > 0
            and robot.ndim == 2 and robot.shape[1:] == (3,) and len(robot) > 0
            and np.all(np.isfinite(scene)) and np.all(np.isfinite(robot))
        )
        if denoiser is not None:
            scene = denoiser.filter(scene)
        plane_removal = None
        if runtime_args.remove_planes:
            plane_removal = {
                "enabled": True,
                "distance_threshold": runtime_args.plane_dist,
                "max_planes": runtime_args.max_planes,
            }
        clustered = trial.FastClusteringFilter(
            scene,
            robot,
            workspace=getattr(processor, "_workspace", None),
            plane_removal=plane_removal,
            eps=runtime_args.cluster_eps,
            min_samples=runtime_args.cluster_min_samples,
            min_points=runtime_args.cluster_min_points,
            min_volume=runtime_args.cluster_min_volume,
        )
        clusters = trial.filter_guard_clusters(list(clustered.clusters), runtime_args)
        frames.append(
            {
                "timestamp": timestamp,
                "frame_valid": valid,
                "cluster_count": len(clusters),
            }
        )
        if not valid or not clusters:
            continue
        selected = max(clusters, key=lambda cluster: len(np.asarray(cluster.points)))
        detection = trial.make_occupancy_object(
            np.asarray(selected.points), timestamp=timestamp, margin=0.0
        )
        return {
            "accepted": True,
            "timestamp": timestamp,
            "center": np.asarray(detection.center, dtype=np.float64),
            "radius": float(detection.radius),
            "point_count": int(len(np.asarray(selected.points))),
            "frames": frames,
        }
    return {"accepted": False, "reason": "external_seed_timeout", "frames": frames}


def trigger_reference_time(source: Path) -> float:
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    trigger_frame = int(summary["trigger_frame"])
    import csv

    with (source / "frames.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["frame"]) == trigger_frame:
                return float(row["reference_time_s"])
    raise RuntimeError(f"trigger frame {trigger_frame} is missing")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_segments < 1 or args.max_wall_s <= 0.0 or args.seed_timeout_s <= 0.0:
        raise ValueError("max-segments, max-wall-s, and seed-timeout-s must be positive")
    source = args.source_trial.resolve()
    output = args.output.resolve() / f"r{args.repeat:02d}"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    source_candidate = json.loads(
        (source / "candidate/candidate_summary.json").read_text(encoding="utf-8")
    )
    reference = trial.RecordedReference.load(args.reference_feedback_csv.resolve())
    runtime_args = trial.build_parser().parse_args(["--scene", "D2", "--mode", "shadow"])
    config = trial.load_stage4_config(runtime_args.stage4_config)
    model = trial.load_stage4_surface_model(config)
    reference_start = trigger_reference_time(source)
    schedule = trial.rolling_local_reference_schedule(
        reference_start,
        local_horizon_s=runtime_args.local_horizon_s,
        max_segments=args.max_segments,
        reference_end_time_s=float(reference.times[-1]),
    )
    log: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "INITIALIZED",
        "robot_commanded": False,
        "real_rgbd": True,
        "virtual_robot_state": True,
        "source_trial": str(source),
        "runtime_git_commit": trial.git_commit_hash(),
        "requested_segments": len(schedule),
        "max_wall_s": float(args.max_wall_s),
        "segments": [],
    }
    processor = None
    started = time.perf_counter()
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
            raise RuntimeError("real RGB-D pipeline with AUBO state reader is required")
        denoiser = (
            trial.TemporalDenoiser(
                runtime_args.denoise_voxel,
                runtime_args.denoise_conf,
                runtime_args.denoise_decay,
            )
            if runtime_args.temporal_denoise
            else None
        )
        trial.require_confirmation(
            True,
            "VIRTUAL SHADOW ONLY: the robot will not move. Press Enter, then introduce "
            f"the moving tabletop obstacle into the D2 corridor within {args.seed_timeout_s:.1f}s "
            "and keep it moving continuously.",
        )
        seed = first_external_seed(
            processor, denoiser, runtime_args, timeout_s=args.seed_timeout_s
        )
        trial.write_json(output / "seed.json", seed)
        if not seed["accepted"]:
            log["status"] = "ROLLING_LOCAL_VIRTUAL_SEED_FAILED"
            return log

        previous = {
            "center": np.asarray(seed["center"], dtype=np.float64),
            "velocity": np.zeros(3),
            "last_timestamp": float(seed["timestamp"]),
        }
        q_virtual = np.asarray(source_candidate["q_now"], dtype=np.float64)
        locked_side = None
        accepted_segments = 0
        for item in schedule:
            if time.perf_counter() - started >= args.max_wall_s:
                log["status"] = "ROLLING_LOCAL_VIRTUAL_WALL_LIMIT_HOLD"
                break
            index = int(item["segment"])
            segment_dir = output / f"segment_{index:02d}"
            fresh_plan, plan_frames, plan_points = trial.capture_post_stop_obstacle(
                processor,
                reader,
                denoiser,
                runtime_args,
                trigger_cluster_center=np.asarray(previous["center"], dtype=np.float64),
                trigger_velocity=np.asarray(previous["velocity"], dtype=np.float64),
                trigger_timestamp=float(previous["last_timestamp"]),
                stop_when_ready=True,
            )
            segment: dict[str, Any] = {
                **item,
                "q_virtual_start": q_virtual.tolist(),
                "fresh_plan": fresh_plan,
                "plan_frame_count": len(plan_frames),
            }
            trial.write_json(
                segment_dir / "fresh_plan.json", {"result": fresh_plan, "frames": plan_frames}
            )
            if not fresh_plan.get("accepted", False) or plan_points is None:
                segment["status"] = "FRESH_PLAN_NOT_READY_HOLD"
                log["segments"].append(segment)
                log["status"] = "ROLLING_LOCAL_VIRTUAL_STOPPED_FAIL_CLOSED"
                break
            plan_geometry = trial.fit_pca_multisphere(
                plan_points,
                fit_margin_m=runtime_args.multisphere_fit_margin_m,
                max_components=runtime_args.multisphere_max_components,
            )
            artifacts: dict[str, Any] = {}
            result = trial.run_fast_repair(
                runtime_args,
                config,
                model,
                q_now=q_virtual,
                qd_now=np.zeros(6),
                center=np.asarray(fresh_plan["center"], dtype=np.float64),
                velocity=np.asarray(fresh_plan["velocity"], dtype=np.float64),
                radius=float(fresh_plan["radius"]),
                risk_links=set(model.surface_by_link(q_virtual, density="coarse")),
                trial_dir=segment_dir,
                reference_goal=reference.state_at(float(item["reference_goal_time_s"])),
                rejoin_goals=None,
                obstacle_audit={"rolling_local_virtual_shadow": True, "segment": index},
                multisphere_geometry=plan_geometry,
                artifacts_out=artifacts,
            )
            side = trial.avoidance_side_consistent(
                locked_side,
                np.asarray(result["tail_delta_q_rad"], dtype=np.float64),
                opposite_projection_tolerance_rad=runtime_args.rolling_side_opposite_tolerance_rad,
            )
            fresh_auth, auth_frames, auth_points = trial.capture_post_stop_obstacle(
                processor,
                reader,
                denoiser,
                runtime_args,
                trigger_cluster_center=np.asarray(fresh_plan["center"], dtype=np.float64),
                trigger_velocity=np.asarray(fresh_plan["velocity"], dtype=np.float64),
                trigger_timestamp=float(fresh_plan["last_timestamp"]),
                stop_when_ready=True,
            )
            trial.write_json(
                segment_dir / "fresh_authorization.json",
                {"result": fresh_auth, "frames": auth_frames},
            )
            local_auth = {
                "status": "LOCAL_EXECUTION_RECHECK_FAILED",
                "local_execution_authorized": False,
                "reason": fresh_auth.get("reason", "fresh_auth_not_ready"),
            }
            if fresh_auth.get("accepted", False) and auth_points is not None and side["accepted"]:
                auth_geometry = trial.fit_pca_multisphere(
                    auth_points,
                    fit_margin_m=runtime_args.multisphere_fit_margin_m,
                    max_components=runtime_args.multisphere_max_components,
                )
                local_auth, _ = trial.authorize_local_repair_execution(
                    runtime_args,
                    config,
                    model,
                    local_repair_ready=bool(result["local_repair_ready"]),
                    local_artifacts=artifacts,
                    fresh_geometry=auth_geometry,
                    fresh_velocity=np.asarray(fresh_auth["velocity"], dtype=np.float64),
                    trial_dir=segment_dir,
                )
            ready = bool(
                result["local_repair_ready"]
                and side["accepted"]
                and local_auth.get("local_execution_authorized", False)
            )
            candidate_csv = segment_dir / "candidate/fast_ccro_nubs_candidate.csv"
            workspace = trial.trajectory_workspace_deviation(
                model,
                candidate_csv,
                reference,
                float(item["reference_plan_start_time_s"]),
            )
            segment.update(
                {
                    "status": "VIRTUAL_LOCAL_SEGMENT_AUTHORIZED" if ready else "VIRTUAL_LOCAL_SEGMENT_REJECTED",
                    "fast": result,
                    "fresh_authorization": fresh_auth,
                    "side_continuity": side,
                    "local_authorization": local_auth,
                    "workspace_deviation": workspace,
                }
            )
            log["segments"].append(segment)
            if not ready:
                log["status"] = "ROLLING_LOCAL_VIRTUAL_STOPPED_FAIL_CLOSED"
                break
            if locked_side is None:
                locked_side = np.asarray(side["locked_tail_delta_q"], dtype=np.float64)
            q_virtual = np.asarray(
                artifacts["candidate_trajectory"].evaluate(
                    artifacts["candidate_trajectory"].total_duration
                ),
                dtype=np.float64,
            )
            segment["q_virtual_end"] = q_virtual.tolist()
            accepted_segments += 1
            previous = fresh_auth
        else:
            log["status"] = "ROLLING_LOCAL_VIRTUAL_ALL_SEGMENTS_AUTHORIZED"
        log["accepted_segments"] = accepted_segments
        log["side_lock_initialized"] = locked_side is not None
        log["elapsed_s"] = time.perf_counter() - started
    except Exception as exc:
        log["status"] = "ROLLING_LOCAL_VIRTUAL_FAILED"
        log["error"] = str(exc)
        log["traceback"] = traceback.format_exc(limit=20)
    finally:
        if processor is not None:
            processor.stop()
        trial.write_json(summary_path, log)
    return log


def main() -> None:
    result = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "robot_commanded": result["robot_commanded"],
                "accepted_segments": result.get("accepted_segments", 0),
                "output": str(build_parser().parse_args().output.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
