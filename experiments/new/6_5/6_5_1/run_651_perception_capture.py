#!/usr/bin/env python3
"""Collect revised 6.5.1 perception-only real RGB-D trials.

This program is deliberately read-only with respect to the robot: it connects
RealSense and AUBO state feedback, but never sends motion commands.  It records
per-frame self-filtering, obstacle clustering, tracking, STRO prediction and
CCRO nearest-link evidence for:

    E0          empty scene / self-filtering
    S1/S2/S3    static obstacles near elbow / forearm / wrist
    D1/D2       dynamic hand-held foam obstacle paths

Typical usage:

    python experiments/new/6_5/6_5_1/run_651_perception_capture.py \
      --scenario S1 --repeat 1 --output results/new/6_5/6_5_1/perception_formal
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.geometry_fit import make_occupancy_object  # noqa: E402
from perception.occupancy_tracker import OccupancyTracker  # noqa: E402
from planning.robot_surface_model import RobotSurfaceModel  # noqa: E402
from risk.prediction import RiskSphere, predict_risk_spheres  # noqa: E402
from risk.safety_policy import SafetyPolicy  # noqa: E402
from test_clustering_filtering import FastClusteringFilter, TemporalDenoiser  # noqa: E402
from test_remove_robot_points_fast import SceneProcessor  # noqa: E402
from utils.config import load_config_dir  # noqa: E402


DEFAULT_OUTPUT = ROOT / "results" / "new" / "6_5" / "6_5_1" / "perception_formal"

SCENARIOS = {
    "E0": {
        "name": "empty_self_filter",
        "description": "empty workspace, robot static, no external obstacle",
        "phases": [("empty", 30.0)],
        "expected_region": "none",
    },
    "S1": {
        "name": "static_elbow",
        "description": "static foam obstacle near elbow / middle arm",
        "phases": [("pre_empty", 3.0), ("obstacle", 10.0), ("post_removed", 3.0)],
        "expected_region": "elbow",
    },
    "S2": {
        "name": "static_forearm",
        "description": "static foam obstacle near forearm",
        "phases": [("pre_empty", 3.0), ("obstacle", 10.0), ("post_removed", 3.0)],
        "expected_region": "forearm",
    },
    "S3": {
        "name": "static_wrist",
        "description": "static foam obstacle near wrist / gripper",
        "phases": [("pre_empty", 3.0), ("obstacle", 10.0), ("post_removed", 3.0)],
        "expected_region": "wrist",
    },
    "D1": {
        "name": "dynamic_cross_forearm_elbow",
        "description": "hand-held foam ball crosses forearm/elbow region",
        "phases": [("pre_empty", 3.0), ("dynamic", 15.0), ("post_removed", 3.0)],
        "expected_region": "body",
    },
    "D2": {
        "name": "dynamic_oblique_wrist_approach",
        "description": "hand-held foam ball approaches wrist/forearm obliquely and leaves",
        "phases": [("pre_empty", 3.0), ("dynamic", 15.0), ("post_removed", 3.0)],
        "expected_region": "wrist_forearm",
    },
}

REGION_LINKS = {
    "none": set(),
    "elbow": {"upperArm_Link", "foreArm_Link", "wrist1_Link"},
    "forearm": {"foreArm_Link", "wrist1_Link", "wrist2_Link"},
    "wrist": {"wrist2_Link", "wrist3_Link", "gripper_base_link", "left_link", "right_link"},
    "body": {"upperArm_Link", "foreArm_Link", "wrist1_Link", "wrist2_Link"},
    "wrist_forearm": {"foreArm_Link", "wrist1_Link", "wrist2_Link", "wrist3_Link", "gripper_base_link"},
}

JOINT_NAMES = (
    "shoulder_joint",
    "upperArm_joint",
    "foreArm_joint",
    "wrist1_joint",
    "wrist2_joint",
    "wrist3_joint",
)


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_surface_model(config_dir: Path, urdf_path: Path) -> RobotSurfaceModel:
    config = load_config_dir(config_dir)
    # Keep this independent of ccro_stage2.yaml so the live perception script
    # follows the actual URDF/config directory used by the camera pipeline.
    return RobotSurfaceModel(
        urdf_path,
        JOINT_NAMES,
        {"coarse": 800, "medium": 2400, "dense": 12000},
        seed=20260623,
        min_points_per_link=64,
        cache_dir=ROOT / "data/cache/robot_surface",
        geometry="collision",
    )


def pctl(values: list[float], percentile: float) -> float | None:
    clean = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(clean) == 0:
        return None
    return float(np.percentile(clean, percentile))


def risk_color_level(policy: SafetyPolicy, distance: float) -> str:
    return policy.evaluate(distance).level.value


def q_from_reader(state_reader: Any) -> tuple[dict[str, float], np.ndarray]:
    joints = state_reader.get_joint_positions()
    q = np.asarray([float(joints.get(name, np.nan)) for name in JOINT_NAMES], dtype=np.float64)
    return joints, q


def nearest_cluster_to_links(
    surface_model: RobotSurfaceModel,
    q: np.ndarray,
    clusters: list[Any],
    *,
    density: str,
) -> dict[str, Any]:
    if len(clusters) == 0 or not np.all(np.isfinite(q)):
        return {
            "distance": math.inf,
            "link": None,
            "cluster_index": None,
            "cluster_center": None,
            "cluster_points": 0,
        }
    surfaces = surface_model.surface_by_link(q, density=density)
    best = {
        "distance": math.inf,
        "link": None,
        "cluster_index": None,
        "cluster_center": None,
        "cluster_points": 0,
    }
    for ci, cluster in enumerate(clusters):
        points = np.asarray(cluster.points, dtype=np.float64)
        if len(points) == 0:
            continue
        for link, surface in surfaces.items():
            if len(surface) == 0:
                continue
            tree = cKDTree(surface)
            dists, _ = tree.query(points, k=1)
            d = float(np.min(dists))
            if d < best["distance"]:
                best = {
                    "distance": d,
                    "link": link,
                    "cluster_index": ci,
                    "cluster_center": np.asarray(cluster.center, dtype=np.float64).copy(),
                    "cluster_points": int(len(points)),
                }
    return best


def nearest_sphere_to_links(
    surface_model: RobotSurfaceModel,
    q: np.ndarray,
    spheres: list[RiskSphere],
    *,
    density: str,
) -> dict[str, Any]:
    if len(spheres) == 0 or not np.all(np.isfinite(q)):
        return {"distance": math.inf, "link": None, "object_id": None, "tau": None}
    surfaces = surface_model.surface_by_link(q, density=density)
    best = {"distance": math.inf, "link": None, "object_id": None, "tau": None}
    for sphere in spheres:
        center = np.asarray(sphere.center, dtype=np.float64)
        for link, surface in surfaces.items():
            if len(surface) == 0:
                continue
            tree = cKDTree(surface)
            d, _ = tree.query(center, k=1)
            surface_distance = float(d - sphere.radius)
            if surface_distance < best["distance"]:
                best = {
                    "distance": surface_distance,
                    "link": link,
                    "object_id": int(sphere.object_id),
                    "tau": float(sphere.tau),
                }
    return best


def prompt_phase(scenario: str, phase: str, duration: float, no_prompt: bool) -> None:
    if no_prompt:
        return
    hints = {
        "empty": "确认工作空间内没有临时障碍物，机械臂保持静止",
        "pre_empty": "确认障碍物尚未进入工作空间，机械臂保持静止",
        "obstacle": "将泡沫障碍物放到预设位置并保持静止",
        "dynamic": "准备按标记路径移动长杆泡沫球",
        "post_removed": "移走障碍物，确认工作空间恢复空场景",
    }
    input(f"\n[{scenario}:{phase}] {hints.get(phase, '')}。按 Enter 开始采集 {duration:.1f}s...")


def collect_trial(args: argparse.Namespace) -> dict[str, Any]:
    if args.scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {args.scenario}; choose from {sorted(SCENARIOS)}")
    scenario_cfg = SCENARIOS[args.scenario]
    output_root = args.output.resolve()
    trial_name = f"{args.scenario}_{scenario_cfg['name']}_r{args.repeat:02d}"
    trial_dir = output_root / "trials" / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    config = load_config_dir(args.config_dir)
    safety = config["safety"]
    policy = SafetyPolicy(
        d_safe=float(safety.get("d_safe", 0.15)),
        d_slow=float(safety.get("d_slow", 0.10)),
        d_stop=float(safety.get("d_stop", 0.05)),
    )
    surface_model = load_surface_model(args.config_dir, args.urdf)
    processor = SceneProcessor(
        config_dir=str(args.config_dir),
        urdf_path=str(args.urdf),
        width=args.width,
        height=args.height,
        threshold=args.self_filter_threshold,
        voxel_size=args.voxel_size,
        use_real_robot=True,
        use_mock_camera=False,
    )
    state_reader = getattr(processor, "_state_reader", None)
    if state_reader is None or type(state_reader).__name__ != "RealRobotStateReader":
        processor.stop()
        raise RuntimeError("real AUBO state reader is required; no robot command was sent")

    tracker = OccupancyTracker(
        association_distance=float(safety.get("association_distance", 0.20)),
        alpha=float(safety.get("velocity_alpha", 0.3)),
        pos_alpha=float(safety.get("pos_alpha", 0.3)),
        motion_gate=float(safety.get("motion_gate", 0.005)),
        velocity_dead_zone=float(safety.get("velocity_dead_zone", 0.01)),
        shape_alpha=float(safety.get("shape_alpha", 0.4)),
    )
    denoiser = None
    if args.temporal_denoise:
        denoiser = TemporalDenoiser(
            voxel_size=args.denoise_voxel,
            confidence_threshold=args.denoise_conf,
            decay=args.denoise_decay,
        )

    rows: list[dict[str, Any]] = []
    phase_summaries: list[dict[str, Any]] = []
    frame_index = 0
    start_wall = time.time()

    try:
        for phase, default_duration in scenario_cfg["phases"]:
            duration = args.duration if args.duration is not None else default_duration
            prompt_phase(args.scenario, phase, duration, args.no_prompt)
            phase_start = time.time()
            phase_rows_before = len(rows)
            while time.time() - phase_start < duration:
                t0 = time.perf_counter()
                frame = processor.process_frame()
                timestamp = float(getattr(frame, "timestamp", time.time()))
                scene_points = np.asarray(frame.scene_points, dtype=np.float64)
                robot_points = np.asarray(frame.robot_points, dtype=np.float64)
                if denoiser is not None:
                    scene_points = denoiser.filter(scene_points)
                joints, q = q_from_reader(state_reader)

                plane_removal = None
                if args.remove_planes:
                    plane_removal = {
                        "enabled": True,
                        "distance_threshold": args.plane_dist,
                        "max_planes": args.max_planes,
                    }
                cluster_result = FastClusteringFilter(
                    scene_points,
                    robot_points,
                    workspace=getattr(processor, "_workspace", None),
                    plane_removal=plane_removal,
                    eps=args.cluster_eps,
                    min_samples=args.cluster_min_samples,
                    min_points=args.cluster_min_points,
                    min_volume=args.cluster_min_volume,
                )
                clusters = cluster_result.clusters
                detections = [
                    make_occupancy_object(
                        cluster.points,
                        timestamp=timestamp,
                        margin=float(safety.get("shape_margin", 0.02)),
                    )
                    for cluster in clusters
                ]
                tracked = tracker.update(detections, timestamp=timestamp)
                stable = [obj for obj in tracked if obj.age >= int(safety.get("min_track_age", 3))]
                risk_spheres = predict_risk_spheres(
                    stable,
                    horizon=float(safety.get("prediction_horizon", 0.4)),
                    step=float(safety.get("prediction_step", 0.1)),
                    margin=float(safety.get("risk_margin", 0.035)),
                    uncertainty=float(safety.get("prediction_uncertainty", 0.02)),
                    static_speed_threshold=float(safety.get("prediction_static_speed_threshold", 0.08)),
                    static_margin=float(safety.get("prediction_static_margin", 0.0)),
                    velocity_radius_scale=float(safety.get("prediction_velocity_radius_scale", 0.1)),
                )

                current_best = nearest_cluster_to_links(
                    surface_model,
                    q,
                    clusters,
                    density=args.surface_density,
                )
                predicted_best = nearest_sphere_to_links(
                    surface_model,
                    q,
                    risk_spheres,
                    density=args.surface_density,
                )
                stable_speeds = [float(np.linalg.norm(obj.velocity)) for obj in stable]
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                center = current_best["cluster_center"]
                row = {
                    "frame": frame_index,
                    "phase": phase,
                    "timestamp": f"{timestamp:.6f}",
                    "scene_points": int(len(scene_points)),
                    "robot_points": int(len(robot_points)),
                    "plane_points": int(len(getattr(cluster_result, "plane_points", []))),
                    "cluster_count": int(len(clusters)),
                    "stable_track_count": int(len(stable)),
                    "risk_sphere_count": int(len(risk_spheres)),
                    "nearest_distance_m": "" if math.isinf(current_best["distance"]) else f"{current_best['distance']:.6f}",
                    "nearest_link": current_best["link"] or "",
                    "nearest_cluster_index": "" if current_best["cluster_index"] is None else int(current_best["cluster_index"]),
                    "nearest_cluster_points": int(current_best["cluster_points"]),
                    "nearest_cluster_x": "" if center is None else f"{center[0]:.6f}",
                    "nearest_cluster_y": "" if center is None else f"{center[1]:.6f}",
                    "nearest_cluster_z": "" if center is None else f"{center[2]:.6f}",
                    "predicted_distance_m": "" if math.isinf(predicted_best["distance"]) else f"{predicted_best['distance']:.6f}",
                    "predicted_nearest_link": predicted_best["link"] or "",
                    "predicted_object_id": "" if predicted_best["object_id"] is None else int(predicted_best["object_id"]),
                    "predicted_tau_s": "" if predicted_best["tau"] is None else f"{predicted_best['tau']:.3f}",
                    "risk_state_current": risk_color_level(policy, current_best["distance"]),
                    "risk_state_predicted": risk_color_level(policy, predicted_best["distance"]),
                    "max_track_speed_m_s": "" if not stable_speeds else f"{max(stable_speeds):.6f}",
                    "mean_track_speed_m_s": "" if not stable_speeds else f"{float(np.mean(stable_speeds)):.6f}",
                    "elapsed_ms": f"{elapsed_ms:.4f}",
                    **{f"q{j+1}_rad": "" if not np.isfinite(q[j]) else f"{q[j]:.8f}" for j in range(6)},
                }
                rows.append(row)
                frame_index += 1

            phase_rows = rows[phase_rows_before:]
            phase_summaries.append(summarize_phase(phase, phase_rows, policy))
    finally:
        processor.stop()

    fields = [
        "frame",
        "phase",
        "timestamp",
        "scene_points",
        "robot_points",
        "plane_points",
        "cluster_count",
        "stable_track_count",
        "risk_sphere_count",
        "nearest_distance_m",
        "nearest_link",
        "nearest_cluster_index",
        "nearest_cluster_points",
        "nearest_cluster_x",
        "nearest_cluster_y",
        "nearest_cluster_z",
        "predicted_distance_m",
        "predicted_nearest_link",
        "predicted_object_id",
        "predicted_tau_s",
        "risk_state_current",
        "risk_state_predicted",
        "max_track_speed_m_s",
        "mean_track_speed_m_s",
        "elapsed_ms",
        *[f"q{j+1}_rad" for j in range(6)],
    ]
    write_csv(trial_dir / "frames.csv", rows, fields)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario": args.scenario,
        "scenario_name": scenario_cfg["name"],
        "description": scenario_cfg["description"],
        "repeat": args.repeat,
        "robot_commanded": False,
        "output_dir": str(trial_dir),
        "expected_region": scenario_cfg["expected_region"],
        "expected_links": sorted(REGION_LINKS.get(scenario_cfg["expected_region"], set())),
        "duration_wall_s": time.time() - start_wall,
        "frame_count": len(rows),
        "phase_summaries": phase_summaries,
        "parameters": {
            "remove_planes": args.remove_planes,
            "cluster_eps": args.cluster_eps,
            "cluster_min_points": args.cluster_min_points,
            "surface_density": args.surface_density,
            "temporal_denoise": args.temporal_denoise,
            "d_safe": policy.d_safe,
            "d_slow": policy.d_slow,
            "d_stop": policy.d_stop,
        },
    }
    write_json(trial_dir / "summary.json", summary)
    update_index(output_root)
    return summary


def parse_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return math.inf
    try:
        return float(value)
    except Exception:
        return math.inf


def summarize_phase(phase: str, rows: list[dict[str, Any]], policy: SafetyPolicy) -> dict[str, Any]:
    if not rows:
        return {"phase": phase, "frames": 0}
    cluster_counts = [int(row["cluster_count"]) for row in rows]
    stable_counts = [int(row["stable_track_count"]) for row in rows]
    distances = [parse_float(row, "nearest_distance_m") for row in rows]
    predicted = [parse_float(row, "predicted_distance_m") for row in rows]
    elapsed = [float(row["elapsed_ms"]) for row in rows]
    non_empty = [d for d in distances if np.isfinite(d)]
    pred_non_empty = [d for d in predicted if np.isfinite(d)]
    nearest_links: dict[str, int] = {}
    predicted_links: dict[str, int] = {}
    for row in rows:
        if row["nearest_link"]:
            nearest_links[row["nearest_link"]] = nearest_links.get(row["nearest_link"], 0) + 1
        if row["predicted_nearest_link"]:
            predicted_links[row["predicted_nearest_link"]] = predicted_links.get(row["predicted_nearest_link"], 0) + 1
    risk_frames = sum(
        row["risk_state_current"] != "SAFE" or row["risk_state_predicted"] != "SAFE"
        for row in rows
    )
    detected_frames = sum(count > 0 for count in cluster_counts)
    stable_frames = sum(count > 0 for count in stable_counts)
    return {
        "phase": phase,
        "frames": len(rows),
        "effective_hz": len(rows) / max(float(rows[-1]["timestamp"]) - float(rows[0]["timestamp"]), 1.0e-6),
        "detected_frame_ratio": detected_frames / len(rows),
        "stable_track_frame_ratio": stable_frames / len(rows),
        "risk_frame_ratio": risk_frames / len(rows),
        "nearest_distance_min_m": None if not non_empty else float(np.min(non_empty)),
        "nearest_distance_p50_m": pctl(non_empty, 50),
        "nearest_distance_p95_m": pctl(non_empty, 95),
        "predicted_distance_min_m": None if not pred_non_empty else float(np.min(pred_non_empty)),
        "nearest_links": dict(sorted(nearest_links.items(), key=lambda item: (-item[1], item[0]))),
        "predicted_links": dict(sorted(predicted_links.items(), key=lambda item: (-item[1], item[0]))),
        "elapsed_ms_p50": pctl(elapsed, 50),
        "elapsed_ms_p95": pctl(elapsed, 95),
    }


def update_index(output_root: Path) -> None:
    trial_summaries = []
    for path in sorted((output_root / "trials").glob("*/summary.json")):
        try:
            trial_summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    write_json(output_root / "metrics.json", aggregate_trials(trial_summaries))
    write_markdown_summary(output_root, trial_summaries)


def aggregate_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        by_scenario.setdefault(trial["scenario"], []).append(trial)
    scenarios = {}
    for scenario, items in sorted(by_scenario.items()):
        active_phases = []
        for trial in items:
            for phase in trial.get("phase_summaries", []):
                if phase.get("phase") in {"obstacle", "dynamic", "empty"}:
                    active_phases.append(phase)
        scenarios[scenario] = {
            "trials": len(items),
            "frames": sum(trial.get("frame_count", 0) for trial in items),
            "active_detected_ratio_mean": None if not active_phases else float(np.mean([p.get("detected_frame_ratio", 0.0) for p in active_phases])),
            "active_stable_track_ratio_mean": None if not active_phases else float(np.mean([p.get("stable_track_frame_ratio", 0.0) for p in active_phases])),
            "active_risk_ratio_mean": None if not active_phases else float(np.mean([p.get("risk_frame_ratio", 0.0) for p in active_phases])),
            "nearest_distance_min_m": min(
                [p.get("nearest_distance_min_m") for p in active_phases if p.get("nearest_distance_min_m") is not None],
                default=None,
            ),
        }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "robot_commanded": False,
        "trial_count": len(trials),
        "scenarios": scenarios,
    }


def write_markdown_summary(output_root: Path, trials: list[dict[str, Any]]) -> None:
    metrics = aggregate_trials(trials)
    lines = [
        "# 6.5.1 Perception Capture Summary",
        "",
        "Robot commanded: **false**",
        "",
        "| scenario | trials | frames | active detected | stable tracked | risk ratio | min distance / m |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, row in metrics["scenarios"].items():
        lines.append(
            f"| {scenario} | {row['trials']} | {row['frames']} | "
            f"{fmt(row['active_detected_ratio_mean'])} | {fmt(row['active_stable_track_ratio_mean'])} | "
            f"{fmt(row['active_risk_ratio_mean'])} | {fmt(row['nearest_distance_min_m'])} |"
        )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.4f}"
    return str(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    parser.add_argument("--urdf", type=Path, default=ROOT / "urdf" / "aubo_i16_gripper.urdf")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--duration", type=float, default=None, help="Override every phase duration in seconds.")
    parser.add_argument("--no-prompt", action="store_true", help="Do not wait for Enter before each phase.")
    parser.add_argument("--remove-planes", action="store_true", default=True)
    parser.add_argument("--no-remove-planes", dest="remove_planes", action="store_false")
    parser.add_argument("--plane-dist", type=float, default=0.02)
    parser.add_argument("--max-planes", type=int, default=1)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--self-filter-threshold", type=float, default=0.05)
    parser.add_argument("--cluster-eps", type=float, default=0.06)
    parser.add_argument("--cluster-min-samples", type=int, default=15)
    parser.add_argument("--cluster-min-points", type=int, default=15)
    parser.add_argument("--cluster-min-volume", type=float, default=0.0005)
    parser.add_argument("--surface-density", choices=["coarse", "medium", "dense"], default="coarse")
    parser.add_argument("--temporal-denoise", action="store_true")
    parser.add_argument("--denoise-voxel", type=float, default=0.04)
    parser.add_argument("--denoise-conf", type=int, default=3)
    parser.add_argument("--denoise-decay", type=float, default=0.4)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = collect_trial(args)
    print(
        json.dumps(
            {
                "robot_commanded": False,
                "scenario": summary["scenario"],
                "repeat": summary["repeat"],
                "frames": summary["frame_count"],
                "output_dir": summary["output_dir"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
