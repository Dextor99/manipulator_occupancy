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
    parser.add_argument("--task-geometry-id", default="D2C_COMPACT_TABLETOP_XP10")
    parser.add_argument(
        "--obstacle-motion-mode",
        choices=("dynamic", "static"),
        default="dynamic",
        help="static forces the measured short-window obstacle velocity to zero",
    )
    parser.add_argument(
        "--obstacle-nominal-size-m",
        default="0.10,0.10,0.10",
        help="audit-only nominal obstacle dimensions dx,dy,dz; no planning authority",
    )
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
    """Find a three-frame external seed without assuming its old r04 location."""
    started = time.perf_counter()
    frames = []
    samples: list[dict[str, Any]] = []
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
        if samples:
            previous_center = np.asarray(samples[-1]["center"], dtype=np.float64)
            selected = min(
                clusters,
                key=lambda cluster: float(
                    np.linalg.norm(np.asarray(cluster.center, dtype=np.float64) - previous_center)
                ),
            )
            association_error = float(
                np.linalg.norm(np.asarray(selected.center, dtype=np.float64) - previous_center)
            )
            if association_error > runtime_args.max_track_cluster_association_m:
                samples = []
        else:
            selected = max(clusters, key=lambda cluster: len(np.asarray(cluster.points)))
            association_error = 0.0
        detection = trial.make_occupancy_object(
            np.asarray(selected.points), timestamp=timestamp, margin=0.0
        )
        samples.append(
            {
                "timestamp": timestamp,
                "center": np.asarray(detection.center, dtype=np.float64),
                "radius": float(detection.radius),
                "association_error_m": association_error,
            }
        )
        fitted = trial.fit_fresh_obstacle_motion(
            samples,
            minimum_frames=runtime_args.post_stop_recheck_min_frames,
            minimum_span_s=runtime_args.post_stop_recheck_min_span_s,
        )
        if fitted.get("accepted", False):
            return {
                **fitted,
                "timestamp": float(fitted["last_timestamp"]),
                "point_count": int(len(np.asarray(selected.points))),
                "frames": frames,
            }
    return {
        "accepted": False,
        "reason": "stable_external_seed_timeout",
        "sample_count": len(samples),
        "frames": frames,
    }


def trigger_reference_time(source: Path) -> float:
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    trigger_frame = int(summary["trigger_frame"])
    import csv

    with (source / "frames.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["frame"]) == trigger_frame:
                return float(row["reference_time_s"])
    raise RuntimeError(f"trigger frame {trigger_frame} is missing")


def virtual_reference_risk(
    runtime_args: argparse.Namespace,
    config: dict[str, Any],
    model: Any,
    reference: Any,
    *,
    reference_start_time_s: float,
    fresh: dict[str, Any],
    geometry: dict[str, Any],
) -> dict[str, Any]:
    evaluator, _, _ = trial.make_risk_stack(config, model, None)
    forecast = trial.constant_multisphere_forecast(
        np.asarray(geometry["component_centers"], dtype=np.float64),
        np.asarray(geometry["component_base_radii"], dtype=np.float64),
        np.asarray(fresh["velocity"], dtype=np.float64),
    )
    rows = []
    for tau in np.arange(
        0.0,
        runtime_args.prediction_horizon_s + 0.5 * runtime_args.prediction_step_s,
        runtime_args.prediction_step_s,
    ):
        risk = evaluator.configuration(
            reference.state_at(reference_start_time_s + float(tau))[0],
            forecast,
            float(tau),
            density="medium",
            with_gradient=False,
        )
        rows.append(
            {"tau_s": float(tau), "distance_m": float(risk.min_distance), "nearest_link": risk.nearest_link}
        )
    best = min(rows, key=lambda row: row["distance_m"])
    return {"min_distance_m": best["distance_m"], "best": best, "preview": rows}


def geometry_quality_audit(
    geometry: dict[str, Any],
    *,
    axial_limit_m: float,
    component_radius_limit_m: float,
) -> dict[str, Any]:
    """Report scene quality without changing planning geometry or authorization."""
    axial_length = float(geometry.get("axial_length_m", math.inf))
    radii = np.asarray(geometry.get("component_base_radii", []), dtype=np.float64)
    max_radius = float(np.max(radii)) if radii.size else math.inf
    warnings = []
    if axial_length > axial_limit_m:
        warnings.append("axial_length_exceeds_nominal_audit_limit")
    if max_radius > component_radius_limit_m:
        warnings.append("component_radius_exceeds_nominal_audit_limit")
    return {
        "audit_only": True,
        "planning_geometry_unchanged": True,
        "axial_length_m": axial_length,
        "max_component_radius_m": max_radius,
        "axial_limit_m": float(axial_limit_m),
        "component_radius_limit_m": float(component_radius_limit_m),
        "compact_scene_quality_ok": not warnings,
        "quality_warnings": warnings,
    }


def obstacle_state_for_mode(fresh: dict[str, Any], motion_mode: str) -> dict[str, Any]:
    """Use one forecast interface; a static obstacle is the v=0 special case."""
    state = copy.deepcopy(fresh)
    if motion_mode == "static" and state.get("accepted", False):
        state["measured_velocity"] = list(state.get("velocity", [0.0, 0.0, 0.0]))
        state["measured_speed_m_s"] = float(state.get("speed_m_s", 0.0))
        state["velocity"] = [0.0, 0.0, 0.0]
        state["speed_m_s"] = 0.0
        state["motion_model"] = "static_zero_velocity"
    else:
        state["motion_model"] = "measured_local_constant_velocity"
    return state


def retry_action(
    segment_gate: dict[str, Any],
    *,
    fresh_accepted: bool,
    has_points: bool,
    has_geometry: bool,
) -> str:
    """Choose the fail-closed virtual transition after one Fast/Fresh attempt."""
    if bool(segment_gate.get("advance", False)):
        return "advance"
    if segment_gate.get("status") == "REFERENCE_SAFE_FOR_REJOIN":
        return "reference_safe"
    if fresh_accepted and has_points and has_geometry:
        return "retry_same_segment"
    return "safe_hold"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_segments < 1 or args.max_wall_s <= 0.0 or args.seed_timeout_s <= 0.0:
        raise ValueError("max-segments, max-wall-s, and seed-timeout-s must be positive")
    source = args.source_trial.resolve()
    obstacle_size = np.asarray(
        [float(value.strip()) for value in args.obstacle_nominal_size_m.split(",")],
        dtype=np.float64,
    )
    if obstacle_size.shape != (3,) or np.any(obstacle_size <= 0.0):
        raise ValueError("obstacle-nominal-size-m must contain three positive dimensions")
    sorted_size = np.sort(obstacle_size)
    geometry_axial_limit_m = float(1.5 * sorted_size[-1] + 0.01)
    geometry_radius_limit_m = float(max(0.12, 1.5 * sorted_size[-2]))
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
        "task_geometry_id": args.task_geometry_id,
        "obstacle_nominal_size_m": obstacle_size.tolist(),
        "obstacle_motion_mode": args.obstacle_motion_mode,
        "forecast_interface": (
            "static_zero_velocity"
            if args.obstacle_motion_mode == "static"
            else "measured_local_constant_velocity"
        ),
        "geometry_quality_audit_limits": {
            "audit_only": True,
            "axial_limit_m": geometry_axial_limit_m,
            "component_radius_limit_m": geometry_radius_limit_m,
        },
        "perception_parameters_frozen": {
            "cluster_eps_m": runtime_args.cluster_eps,
            "cluster_min_samples": runtime_args.cluster_min_samples,
            "cluster_min_points": runtime_args.cluster_min_points,
            "cluster_min_volume_m3": runtime_args.cluster_min_volume,
            "temporal_denoise": runtime_args.temporal_denoise,
        },
        "safety_parameters_frozen": {
            "online_accept_m": runtime_args.online_accept_m,
            "raw_hard_guard_m": runtime_args.guided_hard_stop_m,
            "local_horizon_s": runtime_args.local_horizon_s,
            "fast_budget_ms": runtime_args.fast_budget_ms,
        },
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
        prompt = (
            "VIRTUAL SHADOW ONLY: the robot will not move. Place the static obstacle "
            "in the planned tabletop corridor before pressing Enter, then leave it stationary."
            if args.obstacle_motion_mode == "static"
            else "VIRTUAL SHADOW ONLY: the robot will not move. Press Enter, then introduce "
            f"the moving tabletop obstacle into the D2 corridor within {args.seed_timeout_s:.1f}s "
            "and keep it moving continuously."
        )
        trial.require_confirmation(True, prompt)
        seed = first_external_seed(
            processor, denoiser, runtime_args, timeout_s=args.seed_timeout_s
        )
        seed = obstacle_state_for_mode(seed, args.obstacle_motion_mode)
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
        last_authorized_context = None
        rolling_started = time.perf_counter()
        armed_plan = None
        pre_risk_audit = []
        while time.perf_counter() - rolling_started < args.max_wall_s:
            fresh_wait, wait_frames, wait_points = trial.capture_post_stop_obstacle(
                processor,
                reader,
                denoiser,
                runtime_args,
                trigger_cluster_center=np.asarray(previous["center"], dtype=np.float64),
                trigger_velocity=np.asarray(previous["velocity"], dtype=np.float64),
                trigger_timestamp=float(previous["last_timestamp"]),
                stop_when_ready=True,
            )
            fresh_wait = obstacle_state_for_mode(fresh_wait, args.obstacle_motion_mode)
            wait_row = {
                "fresh": fresh_wait,
                "frame_count": len(wait_frames),
                "reference_risk": None,
            }
            if fresh_wait.get("accepted", False) and wait_points is not None:
                wait_geometry = trial.fit_pca_multisphere(
                    wait_points,
                    fit_margin_m=runtime_args.multisphere_fit_margin_m,
                    max_components=runtime_args.multisphere_max_components,
                )
                wait_risk = virtual_reference_risk(
                    runtime_args,
                    config,
                    model,
                    reference,
                    reference_start_time_s=float(schedule[0]["reference_plan_start_time_s"]),
                    fresh=fresh_wait,
                    geometry=wait_geometry,
                )
                wait_row["reference_risk"] = wait_risk
                previous = fresh_wait
                if wait_risk["min_distance_m"] < runtime_args.moving_shadow_replan_in_m:
                    armed_plan = (fresh_wait, wait_frames, wait_points, wait_geometry)
                    wait_row["risk_armed"] = True
                    pre_risk_audit.append(wait_row)
                    break
            pre_risk_audit.append(wait_row)
        trial.write_json(output / "pre_risk_wait.json", {"attempts": pre_risk_audit})
        if armed_plan is None:
            log["status"] = "ROLLING_LOCAL_VIRTUAL_NO_RISK_BEFORE_TIMEOUT"
            log["accepted_segments"] = 0
            log["pre_risk_attempts"] = len(pre_risk_audit)
            return log
        stop_segments = False
        short_args = copy.copy(runtime_args)
        short_args.post_stop_recheck_duration_s = runtime_args.rolling_observation_duration_s
        short_args.post_stop_recheck_min_frames = runtime_args.rolling_observation_min_frames
        short_args.post_stop_recheck_min_span_s = runtime_args.rolling_observation_min_span_s
        for item in schedule:
            if time.perf_counter() - rolling_started >= args.max_wall_s:
                log["status"] = "ROLLING_LOCAL_VIRTUAL_WALL_LIMIT_HOLD"
                break
            index = int(item["segment"])
            segment_dir = output / f"segment_{index:02d}"
            segment_started = time.perf_counter()
            segment_deadline = min(
                rolling_started + args.max_wall_s,
                segment_started + runtime_args.rolling_fast_max_s,
            )
            if index == 1:
                fresh_plan, plan_frames, plan_points, plan_geometry = armed_plan
            else:
                fresh_plan, plan_frames, plan_points = trial.capture_post_stop_obstacle(
                    processor,
                    reader,
                    denoiser,
                    short_args,
                    trigger_cluster_center=np.asarray(previous["center"], dtype=np.float64),
                    trigger_velocity=np.asarray(previous["velocity"], dtype=np.float64),
                    trigger_timestamp=float(previous["last_timestamp"]),
                    stop_when_ready=True,
                )
                fresh_plan = obstacle_state_for_mode(fresh_plan, args.obstacle_motion_mode)
                plan_geometry = None
            segment: dict[str, Any] = {
                **item,
                "q_virtual_start": q_virtual.tolist(),
                "attempts": [],
                "retry_budget_s": float(runtime_args.rolling_fast_max_s),
            }
            attempt_index = 0
            segment_authorized = False
            while time.perf_counter() < segment_deadline:
                attempt_index += 1
                attempt_dir = segment_dir / f"attempt_{attempt_index:02d}"
                attempt: dict[str, Any] = {
                    "attempt": attempt_index,
                    "q_virtual_start": q_virtual.tolist(),
                    "fresh_plan": fresh_plan,
                    "plan_frame_count": len(plan_frames),
                    "reused_previous_fresh_authorization": attempt_index > 1,
                }
                trial.write_json(
                    attempt_dir / "fresh_plan.json",
                    {"result": fresh_plan, "frames": plan_frames},
                )
                if not fresh_plan.get("accepted", False) or plan_points is None:
                    attempt["status"] = "FRESH_PLAN_NOT_READY_HOLD"
                    segment["attempts"].append(attempt)
                    log["status"] = "ROLLING_LOCAL_VIRTUAL_STOPPED_FAIL_CLOSED"
                    stop_segments = True
                    break
                if plan_geometry is None:
                    plan_geometry = trial.fit_pca_multisphere(
                        plan_points,
                        fit_margin_m=runtime_args.multisphere_fit_margin_m,
                        max_components=runtime_args.multisphere_max_components,
                    )
                attempt["geometry_quality"] = geometry_quality_audit(
                    plan_geometry,
                    axial_limit_m=geometry_axial_limit_m,
                    component_radius_limit_m=geometry_radius_limit_m,
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
                    trial_dir=attempt_dir,
                    reference_goal=reference.state_at(float(item["reference_goal_time_s"])),
                    rejoin_goals=None,
                    obstacle_audit={
                        "rolling_local_virtual_shadow": True,
                        "segment": index,
                        "attempt": attempt_index,
                    },
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
                    short_args,
                    trigger_cluster_center=np.asarray(fresh_plan["center"], dtype=np.float64),
                    trigger_velocity=np.asarray(fresh_plan["velocity"], dtype=np.float64),
                    trigger_timestamp=float(fresh_plan["last_timestamp"]),
                    stop_when_ready=True,
                )
                fresh_auth = obstacle_state_for_mode(fresh_auth, args.obstacle_motion_mode)
                trial.write_json(
                    attempt_dir / "fresh_authorization.json",
                    {"result": fresh_auth, "frames": auth_frames},
                )
                local_auth = {
                    "status": "LOCAL_EXECUTION_RECHECK_FAILED",
                    "local_execution_authorized": False,
                    "reason": fresh_auth.get("reason", "fresh_auth_not_ready"),
                }
                auth_geometry = None
                if fresh_auth.get("accepted", False) and auth_points is not None:
                    auth_geometry = trial.fit_pca_multisphere(
                        auth_points,
                        fit_margin_m=runtime_args.multisphere_fit_margin_m,
                        max_components=runtime_args.multisphere_max_components,
                    )
                if auth_geometry is not None and side["accepted"]:
                    local_auth, _ = trial.authorize_local_repair_execution(
                        runtime_args,
                        config,
                        model,
                        local_repair_ready=bool(result["local_repair_ready"]),
                        local_artifacts=artifacts,
                        fresh_geometry=auth_geometry,
                        fresh_velocity=np.asarray(fresh_auth["velocity"], dtype=np.float64),
                        trial_dir=attempt_dir,
                    )
                segment_gate = trial.rolling_local_segment_gate(
                    reference_min_distance_m=float(result["reference_online_min_distance_m"]),
                    local_repair_ready=bool(result["local_repair_ready"]),
                    side_consistent=bool(side["accepted"]),
                    fresh_authorized=bool(local_auth.get("local_execution_authorized", False)),
                    replan_threshold_m=runtime_args.moving_shadow_replan_in_m,
                )
                workspace = trial.trajectory_workspace_deviation(
                    model,
                    attempt_dir / "candidate/fast_ccro_nubs_candidate.csv",
                    reference,
                    float(item["reference_plan_start_time_s"]),
                )
                attempt.update(
                    {
                        "status": segment_gate["status"],
                        "segment_gate": segment_gate,
                        "fast": result,
                        "fresh_authorization": fresh_auth,
                        "fresh_geometry_quality": (
                            geometry_quality_audit(
                                auth_geometry,
                                axial_limit_m=geometry_axial_limit_m,
                                component_radius_limit_m=geometry_radius_limit_m,
                            )
                            if auth_geometry is not None
                            else None
                        ),
                        "side_continuity": side,
                        "local_authorization": local_auth,
                        "workspace_deviation": workspace,
                        "attempt_elapsed_s": time.perf_counter() - segment_started,
                    }
                )
                segment["attempts"].append(attempt)
                action = retry_action(
                    segment_gate,
                    fresh_accepted=bool(fresh_auth.get("accepted", False)),
                    has_points=auth_points is not None,
                    has_geometry=auth_geometry is not None,
                )
                attempt["next_action"] = action
                if action == "advance":
                    if locked_side is None:
                        locked_side = np.asarray(side["locked_tail_delta_q"], dtype=np.float64)
                    q_virtual = np.asarray(
                        artifacts["candidate_trajectory"].evaluate(
                            artifacts["candidate_trajectory"].total_duration
                        ),
                        dtype=np.float64,
                    )
                    segment["q_virtual_end"] = q_virtual.tolist()
                    segment["status"] = segment_gate["status"]
                    segment["authorized_attempt"] = attempt_index
                    accepted_segments += 1
                    previous = fresh_auth
                    last_authorized_context = {
                        "item": item,
                        "artifacts": artifacts,
                        "fresh": fresh_auth,
                        "fresh_frames": auth_frames,
                        "fresh_geometry": auth_geometry,
                    }
                    segment_authorized = True
                    break
                if action == "reference_safe":
                    segment["status"] = segment_gate["status"]
                    log["status"] = "ROLLING_LOCAL_VIRTUAL_REFERENCE_SAFE_FOR_REJOIN"
                    stop_segments = True
                    break
                if action == "safe_hold":
                    segment["status"] = "FRESH_AUTH_NOT_READY_HOLD"
                    log["status"] = "ROLLING_LOCAL_VIRTUAL_STOPPED_FAIL_CLOSED"
                    stop_segments = True
                    break
                # The virtual state deliberately stays fixed.  Reuse this already captured
                # Fresh observation immediately as the next planning input.
                fresh_plan, plan_frames, plan_points, plan_geometry = (
                    fresh_auth,
                    auth_frames,
                    auth_points,
                    auth_geometry,
                )
                previous = fresh_auth
            if not segment_authorized and not stop_segments:
                segment["status"] = "ROLLING_LOCAL_SEGMENT_RETRY_TIMEOUT_HOLD"
                log["status"] = "ROLLING_LOCAL_VIRTUAL_RETRY_TIMEOUT_HOLD"
                stop_segments = True
            segment["attempt_count"] = len(segment["attempts"])
            segment["segment_elapsed_s"] = time.perf_counter() - segment_started
            log["segments"].append(segment)
            if stop_segments:
                break
        else:
            log["status"] = "ROLLING_LOCAL_VIRTUAL_ALL_SEGMENTS_AUTHORIZED"
        if (
            log["status"] == "ROLLING_LOCAL_VIRTUAL_ALL_SEGMENTS_AUTHORIZED"
            and last_authorized_context is not None
        ):
            closure_started = time.perf_counter()
            context = last_authorized_context
            fresh3, fresh3_frames, fresh3_points = trial.capture_post_stop_obstacle(
                processor,
                reader,
                denoiser,
                short_args,
                trigger_cluster_center=np.asarray(context["fresh"]["center"], dtype=np.float64),
                trigger_velocity=np.asarray(context["fresh"]["velocity"], dtype=np.float64),
                trigger_timestamp=float(context["fresh"]["last_timestamp"]),
                stop_when_ready=True,
            )
            fresh3 = obstacle_state_for_mode(fresh3, args.obstacle_motion_mode)
            fresh3_geometry = None
            if fresh3.get("accepted", False) and fresh3_points is not None:
                fresh3_geometry = trial.fit_pca_multisphere(
                    fresh3_points,
                    fit_margin_m=runtime_args.multisphere_fit_margin_m,
                    max_components=runtime_args.multisphere_max_components,
                )
                if not fresh3_geometry["covered"]:
                    fresh3 = {
                        **fresh3,
                        "accepted": False,
                        "reason": "closure_fresh3_multisphere_coverage_failed",
                    }
            trial.write_json(
                output / "closure_fresh3.json",
                {"result": fresh3, "frames": fresh3_frames},
            )
            if fresh3_geometry is not None:
                trial.write_json(output / "closure_fresh3_multisphere.json", fresh3_geometry)

            virtual_tail_guard_m = -math.inf
            rejoin_goals = []
            if fresh3.get("accepted", False) and fresh3_geometry is not None:
                evaluator, _, _ = trial.make_risk_stack(config, model, None)
                closure_forecast = trial.constant_multisphere_forecast(
                    np.asarray(fresh3_geometry["component_centers"], dtype=np.float64),
                    np.asarray(fresh3_geometry["component_base_radii"], dtype=np.float64),
                    np.asarray(fresh3["velocity"], dtype=np.float64),
                )
                virtual_tail_guard_m = float(
                    evaluator.configuration(
                        q_virtual,
                        closure_forecast,
                        0.0,
                        density="medium",
                        with_gradient=False,
                    ).min_distance
                )
                rejoin_offsets = np.arange(
                    runtime_args.local_horizon_s + runtime_args.rejoin_search_step_s,
                    runtime_args.rejoin_max_offset_s
                    + 0.5 * runtime_args.rejoin_search_step_s,
                    runtime_args.rejoin_search_step_s,
                )
                reference_plan_start = float(context["item"]["reference_plan_start_time_s"])
                rejoin_goals = [
                    (
                        float(offset),
                        reference.state_at(reference_plan_start + float(offset)),
                    )
                    for offset in rejoin_offsets
                    if reference_plan_start + float(offset) <= float(reference.times[-1])
                ]
            virtual_tail_online_safe = bool(
                np.isfinite(virtual_tail_guard_m)
                and virtual_tail_guard_m >= runtime_args.online_accept_m
            )
            if virtual_tail_online_safe:
                # A real raw-cloud hard guard cannot be measured at q_virtual while
                # the physical robot remains at the start.  Do not relabel the
                # medium multisphere distance as a 0.10 m raw guard.  The virtual
                # tail uses the unchanged 0.09 m online gate; the bridge verifier
                # then checks the complete trajectory with the same geometry.
                delayed, bridge = trial.authorize_delayed_rejoin_after_fresh3(
                    runtime_args,
                    config,
                    model,
                    local_artifacts=context["artifacts"],
                    fresh3=fresh3,
                    fresh3_geometry=fresh3_geometry,
                    fresh3_frames=fresh3_frames,
                    rejoin_goals=rejoin_goals,
                    hard_guard_distance_m=math.inf,
                    trial_dir=output / "closure",
                )
            else:
                delayed, bridge = (
                    {
                        "status": "DELAYED_REJOIN_HOLD",
                        "authorized": False,
                        "reason": "virtual_tail_below_online_accept",
                        "virtual_tail_distance_m": virtual_tail_guard_m,
                        "online_accept_m": float(runtime_args.online_accept_m),
                        "hard_guard_distance_m": None,
                        "hard_guard_safe": None,
                        "raw_hard_guard_applicable": False,
                        "rejoin_search_audit": [],
                        "authorized_trajectory_csv": None,
                    },
                    None,
                )
            remainder_audit = {
                "authorized": False,
                "reason": "delayed_rejoin_not_authorized",
            }
            if delayed["authorized"] and bridge is not None and fresh3_geometry is not None:
                reference_plan_start = float(context["item"]["reference_plan_start_time_s"])
                rejoin_absolute_time = reference_plan_start + float(
                    delayed["selected_rejoin_offset_s"]
                )
                remainder_times, remainder_q, _ = reference.remainder_after(rejoin_absolute_time)
                evaluator, _, _ = trial.make_risk_stack(config, model, None)
                closure_forecast = trial.constant_multisphere_forecast(
                    np.asarray(fresh3_geometry["component_centers"], dtype=np.float64),
                    np.asarray(fresh3_geometry["component_base_radii"], dtype=np.float64),
                    np.asarray(fresh3["velocity"], dtype=np.float64),
                )
                bridge_duration = float(bridge.total_duration)
                risk_rows = []
                for tau, q_tau in zip(remainder_times, remainder_q):
                    risk = evaluator.configuration(
                        np.asarray(q_tau, dtype=np.float64),
                        closure_forecast,
                        bridge_duration + float(tau),
                        density="medium",
                        with_gradient=False,
                    )
                    risk_rows.append(
                        {
                            "tau_s": float(tau),
                            "distance_m": float(risk.min_distance),
                            "nearest_link": risk.nearest_link,
                        }
                    )
                minimum = min(risk_rows, key=lambda row: row["distance_m"])
                remainder_audit = {
                    "authorized": bool(
                        minimum["distance_m"] >= runtime_args.online_accept_m
                    ),
                    "reason": (
                        "full_remainder_clear"
                        if minimum["distance_m"] >= runtime_args.online_accept_m
                        else "full_remainder_below_online_accept"
                    ),
                    "rejoin_absolute_time_s": rejoin_absolute_time,
                    "minimum_distance_m": minimum["distance_m"],
                    "minimum_tau_s": minimum["tau_s"],
                    "minimum_link": minimum["nearest_link"],
                    "sample_count": len(risk_rows),
                    "online_accept_m": float(runtime_args.online_accept_m),
                    "risk_profile": risk_rows,
                }
            closure_authorized = bool(delayed["authorized"] and remainder_audit["authorized"])
            log["closure_audit"] = {
                "fresh3": fresh3,
                "virtual_tail_guard_m": virtual_tail_guard_m,
                "virtual_tail_guard_basis": "medium_multisphere_at_virtual_tail",
                "virtual_tail_online_safe": virtual_tail_online_safe,
                "raw_hard_guard_applicable": False,
                "raw_hard_guard_reason": "physical_robot_remains_at_start_in_virtual_shadow",
                "delayed_rejoin": delayed,
                "full_reference_remainder": remainder_audit,
                "authorized": closure_authorized,
                "elapsed_s": time.perf_counter() - closure_started,
            }
            log["status"] = (
                "STATIC_ROLLING_REJOIN_AND_REMAINDER_AUTHORIZED"
                if closure_authorized and args.obstacle_motion_mode == "static"
                else (
                    "ROLLING_LOCAL_VIRTUAL_REJOIN_AND_REMAINDER_AUTHORIZED"
                    if closure_authorized
                    else "ROLLING_LOCAL_VIRTUAL_CLOSURE_HOLD"
                )
            )
        log["accepted_segments"] = accepted_segments
        log["side_lock_initialized"] = locked_side is not None
        log["elapsed_s"] = time.perf_counter() - started
        log["rolling_elapsed_s"] = time.perf_counter() - rolling_started
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
