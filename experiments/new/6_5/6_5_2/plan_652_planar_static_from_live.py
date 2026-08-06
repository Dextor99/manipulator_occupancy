#!/usr/bin/env python3
"""Capture a live tabletop obstacle and plan a fixed-height planar detour.

This script is the safe planning step after the robot has already been moved to
the planar start pose.  It opens RealSense and AUBO state feedback, but it never
sends robot motion commands.

Output:

* live self-filtered obstacle model;
* straight tabletop reference path;
* planar detour candidate path;
* obstacle/table clearance previews;
* execution waypoint JSON for the later guarded executor.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
EXP651 = ROOT / "experiments" / "new" / "6_5" / "6_5_1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(EXP651) not in sys.path:
    sys.path.insert(0, str(EXP651))

from experiments.exp_ccro_stage2 import _load  # noqa: E402
from run_652_static_avoidance import (  # noqa: E402
    DEFAULT_CONFIG,
    SCENARIOS,
    make_surface_model,
    write_json,
)
from capture_651_visual_snapshots import process_visual_frame, render_pointcloud_snapshot  # noqa: E402
from preview_652_planar_tabletop import (  # noqa: E402
    DEFAULT_CLEARANCE_LINKS,
    bezier,
    build_paths,
    clearance_along_path,
    moving_surface_by_link,
    parse_links,
    plot_clearance,
    plot_pose_sequence,
    plot_top_view,
    stack_surface,
    tcp_position,
)
from risk.safety_policy import SafetyPolicy  # noqa: E402
from run_651_perception_capture import nearest_cluster_to_links, q_from_reader, risk_color_level  # noqa: E402
from test_clustering_filtering import FastClusteringFilter, TemporalDenoiser  # noqa: E402
from test_remove_robot_points_fast import SceneProcessor  # noqa: E402
from utils.config import load_config_dir  # noqa: E402


DEFAULT_OUTPUT = ROOT / "results" / "new" / "6_5" / "6_5_2" / "planar_static_live"


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def scenario_slug(scenario: str) -> str:
    return SCENARIOS[scenario]["slug"]


def choose_static_cluster(surface_model, q: np.ndarray, clusters: list[Any], *, density: str, min_points: int):
    original_indices = {id(cluster): index for index, cluster in enumerate(clusters)}
    filtered = [c for c in clusters if len(c.points) >= min_points]
    if not filtered:
        return None, {"distance": float("inf"), "link": None, "cluster_center": None, "cluster_points": 0}
    best = nearest_cluster_to_links(surface_model, q, filtered, density=density)
    if best["cluster_index"] is None:
        return None, best
    best["cluster_index"] = original_indices[id(filtered[int(best["cluster_index"])])]
    return best["cluster_index"], best


def collect_static_model_light(args: argparse.Namespace, trial_dir: Path, surface_model) -> dict[str, Any]:
    """Live obstacle capture without retaining per-frame RGB/depth objects."""

    config_live = load_config_dir(args.config_dir)
    safety = config_live["safety"]
    policy = SafetyPolicy(
        d_safe=float(safety.get("d_safe", 0.15)),
        d_slow=float(safety.get("d_slow", 0.10)),
        d_stop=float(safety.get("d_stop", 0.05)),
    )
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

    denoiser = (
        TemporalDenoiser(
            voxel_size=args.denoise_voxel,
            confidence_threshold=args.denoise_conf,
            decay=args.denoise_decay,
        )
        if args.temporal_denoise
        else None
    )

    if not args.no_prompt:
        input(f"\n[{args.scenario}] {SCENARIOS[args.scenario]['prompt']}。按 Enter 开始轻量采集...")

    rows: list[dict[str, Any]] = []
    q_samples: list[np.ndarray] = []
    selected_chunks: list[np.ndarray] = []
    representative_scene = np.empty((0, 3))
    representative_robot = np.empty((0, 3))
    representative_best: dict[str, Any] | None = None
    selected_frame_indices: list[int] = []

    started = time.time()
    frame_index = 0
    try:
        while time.time() - started < args.capture_duration_s:
            raw_frame, scene_points, robot_points = process_visual_frame(processor, args)
            timestamp = float(getattr(raw_frame, "timestamp", time.time()))
            if denoiser is not None:
                scene_points = denoiser.filter(scene_points)
            joints, q = q_from_reader(state_reader)
            q_samples.append(q)
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
            clusters = list(cluster_result.clusters)
            cluster_idx, best = choose_static_cluster(
                surface_model,
                q,
                clusters,
                density=args.surface_density,
                min_points=args.obstacle_min_points,
            )
            center = best["cluster_center"]
            rows.append(
                {
                    "frame": frame_index,
                    "timestamp": f"{timestamp:.6f}",
                    "scene_points": int(len(scene_points)),
                    "robot_points": int(len(robot_points)),
                    "cluster_count": int(len(clusters)),
                    "nearest_distance_m": "" if np.isinf(best["distance"]) else f"{best['distance']:.6f}",
                    "nearest_link": best["link"] or "",
                    "nearest_cluster_index": "" if cluster_idx is None else int(cluster_idx),
                    "nearest_cluster_points": int(best["cluster_points"]),
                    "nearest_cluster_x": "" if center is None else f"{center[0]:.6f}",
                    "nearest_cluster_y": "" if center is None else f"{center[1]:.6f}",
                    "nearest_cluster_z": "" if center is None else f"{center[2]:.6f}",
                    "risk_state_current": risk_color_level(policy, best["distance"]),
                    **{f"q{j+1}_rad": "" if not np.isfinite(q[j]) else f"{q[j]:.8f}" for j in range(6)},
                }
            )
            if cluster_idx is not None and len(selected_chunks) < args.obstacle_model_frames:
                selected_chunks.append(np.asarray(clusters[int(cluster_idx)].points, dtype=np.float64).copy())
                selected_frame_indices.append(frame_index)
                if representative_best is None or float(best["distance"]) < float(representative_best["distance"]):
                    representative_best = dict(best)
                    representative_scene = np.asarray(scene_points, dtype=np.float64).copy()
                    representative_robot = np.asarray(robot_points, dtype=np.float64).copy()
            frame_index += 1
    finally:
        processor.stop()

    write_csv(
        trial_dir / "perception_frames.csv",
        rows,
        [
            "frame",
            "timestamp",
            "scene_points",
            "robot_points",
            "cluster_count",
            "nearest_distance_m",
            "nearest_link",
            "nearest_cluster_index",
            "nearest_cluster_points",
            "nearest_cluster_x",
            "nearest_cluster_y",
            "nearest_cluster_z",
            "risk_state_current",
            *[f"q{j+1}_rad" for j in range(6)],
        ],
    )

    if not selected_chunks:
        return {
            "accepted": False,
            "reason": "no stable obstacle cluster selected",
            "frame_count": len(rows),
            "q_mean": np.mean(q_samples, axis=0).tolist() if q_samples else None,
            "obstacle": None,
        }

    obstacle_points = np.vstack(selected_chunks)
    if len(obstacle_points) > args.max_obstacle_points:
        rng = np.random.default_rng(args.seed + args.repeat)
        obstacle_points = obstacle_points[rng.choice(len(obstacle_points), args.max_obstacle_points, replace=False)]
    center = np.mean(obstacle_points, axis=0)
    radius = float(np.max(np.linalg.norm(obstacle_points - center, axis=1))) if len(obstacle_points) else 0.0
    np.savez_compressed(trial_dir / "obstacle_points.npz", points=obstacle_points)
    render_pointcloud_snapshot(
        trial_dir / "figures" / "obstacle_model_pointcloud.png",
        representative_scene,
        representative_robot,
        [obstacle_points],
        0,
    )
    obstacle_payload = {
        "accepted": bool(len(obstacle_points) >= args.obstacle_min_points),
        "reason": "ok" if len(obstacle_points) >= args.obstacle_min_points else "too few obstacle points",
        "frame_count": len(rows),
        "selected_frames": selected_frame_indices,
        "point_count": int(len(obstacle_points)),
        "center": center.tolist(),
        "radius_estimate_m": radius,
        "nearest_link_representative": "" if representative_best is None else representative_best.get("link", ""),
        "nearest_distance_representative_m": None if representative_best is None else float(representative_best["distance"]),
        "q_mean": np.mean(q_samples, axis=0).tolist() if q_samples else None,
        "snapshot_dir": None,
        "capture_mode": "light_no_rgb_depth_snapshots",
    }
    write_json(trial_dir / "detected_obstacle.json", obstacle_payload)
    return {"accepted": obstacle_payload["accepted"], "obstacle": obstacle_payload, "points": obstacle_points}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {args.scenario}")

    slug = scenario_slug(args.scenario)
    trial_dir = args.output.resolve() / slug / "trials" / f"{slug}_r{args.repeat:02d}"
    if trial_dir.exists() and not args.allow_overwrite:
        raise FileExistsError(f"{trial_dir} already exists; use a new --repeat or pass --allow-overwrite")
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "figures").mkdir(exist_ok=True)
    shutil.copy2(args.config, trial_dir / "config_used.yaml")

    config = _load(args.config)
    surface_model = make_surface_model(config)

    capture = collect_static_model_light(args, trial_dir, surface_model)
    if not capture.get("accepted"):
        summary = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "robot_commanded": False,
            "scenario": args.scenario,
            "repeat": args.repeat,
            "status": "OBSTACLE_MODEL_REJECTED",
            "reason": capture.get("reason", "unknown"),
            "trial_dir": str(trial_dir),
            "capture": capture,
        }
        write_json(trial_dir / "summary.json", summary)
        return summary

    obstacle_points = np.asarray(capture["points"], dtype=np.float64)
    q = np.asarray(capture["obstacle"]["q_mean"], dtype=np.float64)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise RuntimeError("live q_mean is invalid; cannot build tabletop preview")

    table_z = float(np.percentile(obstacle_points[:, 2], 2)) if args.table_z == "auto" else float(args.table_z)
    base_tcp = tcp_position(surface_model, q, args.tcp_link)
    p_start = base_tcp.copy()
    if args.y_start is not None:
        p_start[1] = float(args.y_start)
    if args.x_start is not None:
        p_start[0] = float(args.x_start)
    p_goal = p_start.copy()
    if args.y_goal is not None:
        p_goal[1] = float(args.y_goal)
    else:
        p_goal[1] = p_start[1] - abs(float(args.distance_m))
    if args.x_goal is not None:
        p_goal[0] = float(args.x_goal)
    p_goal[2] = p_start[2] = base_tcp[2]

    paths = build_paths(
        p_start,
        p_goal,
        obstacle_points,
        clearance_m=args.clearance_m,
        detour_extra_m=args.detour_extra_m,
        side=args.side,
        samples=args.samples,
    )
    via_clamped = False
    via = paths["via"].copy()
    if args.min_via_x is not None and via[0] < args.min_via_x:
        via[0] = float(args.min_via_x)
        via_clamped = True
    if args.max_via_x is not None and via[0] > args.max_via_x:
        via[0] = float(args.max_via_x)
        via_clamped = True
    if args.min_via_y is not None and via[1] < args.min_via_y:
        via[1] = float(args.min_via_y)
        via_clamped = True
    if args.max_via_y is not None and via[1] > args.max_via_y:
        via[1] = float(args.max_via_y)
        via_clamped = True
    if via_clamped:
        u = np.linspace(0.0, 1.0, args.samples)
        candidate = bezier(paths["p0"], via, paths["p2"], u)
        candidate[:, 2] = paths["p0"][2]
        paths = {**paths, "via_unclamped": paths["via"], "via": via, "candidate": candidate}

    clearance_links = parse_links(args.clearance_links)
    base_by_link = moving_surface_by_link(surface_model, q, args.surface_density)
    clearance_by_link = {link: pts for link, pts in base_by_link.items() if link in set(clearance_links)}
    base_surface = stack_surface(clearance_by_link)
    if len(base_surface) == 0:
        raise RuntimeError(f"none of --clearance-links are available in URDF surface: {clearance_links}")

    ref_clear = clearance_along_path(base_surface, paths["reference"], base_tcp, obstacle_points, table_z)
    cand_clear = clearance_along_path(base_surface, paths["candidate"], base_tcp, obstacle_points, table_z)

    figures = trial_dir / "figures"
    plot_top_view(figures / "top_view_tcp_paths.png", obstacle_points, paths, args.clearance_m)
    plot_clearance(
        figures / "clearance_curves.png",
        ref_clear,
        cand_clear,
        obstacle_threshold=args.clearance_m,
        table_threshold=args.table_clearance_threshold_m,
    )
    plot_pose_sequence(
        figures / "reference_pose_sequence.png",
        obstacle_points,
        base_by_link,
        paths["reference"],
        base_tcp,
        table_z,
        "Straight tabletop reference: fixed posture, constant TCP height",
    )
    plot_pose_sequence(
        figures / "candidate_pose_sequence.png",
        obstacle_points,
        base_by_link,
        paths["candidate"],
        base_tcp,
        table_z,
        "Planar detour candidate: fixed posture, constant TCP height",
    )

    rows = []
    for name, payload in (("reference", ref_clear), ("candidate", cand_clear)):
        for row in payload["rows"]:
            rows.append({"path": name, **row})
    write_csv(
        trial_dir / "planar_path_samples.csv",
        rows,
        ["path", "index", "tcp_x", "tcp_y", "tcp_z", "obstacle_clearance_m", "table_clearance_m"],
    )

    waypoints = {
        "robot_commanded": False,
        "coordinate_frame": "AUBO base frame, TCP xyz only",
        "orientation_policy": "later guarded executor should keep the current TCP orientation and replace xyz with these waypoints",
        "reference_xyz": {
            "P0_start": paths["p0"].tolist(),
            "P2_goal": paths["p2"].tolist(),
        },
        "candidate_xyz": {
            "P0_start": paths["p0"].tolist(),
            "P1_via": paths["via"].tolist(),
            "P2_goal": paths["p2"].tolist(),
        },
        "samples": {
            "reference": paths["reference"].tolist(),
            "candidate": paths["candidate"].tolist(),
        },
    }
    write_json(trial_dir / "planar_execution_waypoints.json", waypoints)

    accepted = bool(
        cand_clear["min_obstacle_clearance_m"] >= args.clearance_m
        and cand_clear["min_table_clearance_m"] >= args.table_clearance_threshold_m
    )
    reference_risky = bool(ref_clear["min_obstacle_clearance_m"] < args.reference_risk_threshold_m)
    status = "PLAN_ACCEPTED" if accepted else "PLAN_REJECTED"
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "robot_commanded": False,
        "scenario": args.scenario,
        "scenario_title": SCENARIOS[args.scenario]["title"],
        "repeat": args.repeat,
        "status": status,
        "trial_dir": str(trial_dir),
        "obstacle_model": capture["obstacle"],
        "q_start_mean_rad": q.tolist(),
        "q_start_mean_deg": np.rad2deg(q).tolist(),
        "tcp_link": args.tcp_link,
        "tcp_base_from_live_q": base_tcp.tolist(),
        "table_z_m": table_z,
        "reference_risky_under_observation": reference_risky,
        "reference_min_obstacle_clearance_m": ref_clear["min_obstacle_clearance_m"],
        "reference_min_table_clearance_m": ref_clear["min_table_clearance_m"],
        "candidate_min_obstacle_clearance_m": cand_clear["min_obstacle_clearance_m"],
        "candidate_min_table_clearance_m": cand_clear["min_table_clearance_m"],
        "clearance_threshold_m": args.clearance_m,
        "reference_risk_threshold_m": args.reference_risk_threshold_m,
        "table_clearance_threshold_m": args.table_clearance_threshold_m,
        "candidate_waypoints": waypoints["candidate_xyz"],
        "via_clamped": via_clamped,
        "via_constraints": {
            "min_via_x": args.min_via_x,
            "max_via_x": args.max_via_x,
            "min_via_y": args.min_via_y,
            "max_via_y": args.max_via_y,
        },
        "figures": [
            "figures/top_view_tcp_paths.png",
            "figures/clearance_curves.png",
            "figures/reference_pose_sequence.png",
            "figures/candidate_pose_sequence.png",
            "figures/obstacle_model_pointcloud.png",
            "snapshots/R-S*_static_obstacle_model_*",
        ],
        "next_step": (
            "Inspect figures and summary. If PLAN_ACCEPTED and the physical path is visibly safe, "
            "use a separate guarded waypoint executor; this script never commands the robot."
        ),
    }
    write_json(trial_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    parser.add_argument("--urdf", type=Path, default=ROOT / "urdf" / "aubo_i16_gripper.urdf")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--capture-duration-s", type=float, default=4.0)
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=20260652)

    parser.add_argument("--self-filter-threshold", type=float, default=0.08)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--cluster-eps", type=float, default=0.06)
    parser.add_argument("--cluster-min-samples", type=int, default=15)
    parser.add_argument("--cluster-min-points", type=int, default=30)
    parser.add_argument("--cluster-min-volume", type=float, default=0.001)
    parser.add_argument("--obstacle-min-points", type=int, default=80)
    parser.add_argument("--obstacle-model-frames", type=int, default=8)
    parser.add_argument("--max-obstacle-points", type=int, default=6000)
    parser.add_argument("--remove-planes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plane-dist", type=float, default=0.02)
    parser.add_argument("--max-planes", type=int, default=1)
    parser.add_argument("--temporal-denoise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--denoise-voxel", type=float, default=0.02)
    parser.add_argument("--denoise-conf", type=int, default=2)
    parser.add_argument("--denoise-decay", type=float, default=0.92)
    parser.add_argument("--surface-density", choices=["coarse", "medium", "dense"], default="coarse")

    parser.add_argument("--tcp-link", default="gripper_base_link")
    parser.add_argument("--x-start", type=float, default=None)
    parser.add_argument("--x-goal", type=float, default=None)
    parser.add_argument("--y-start", type=float, default=None)
    parser.add_argument("--y-goal", type=float, default=-0.4)
    parser.add_argument("--distance-m", type=float, default=0.8)
    parser.add_argument("--clearance-m", type=float, default=0.08)
    parser.add_argument("--reference-risk-threshold-m", type=float, default=0.08)
    parser.add_argument("--detour-extra-m", type=float, default=0.05)
    parser.add_argument("--side", choices=["auto", "positive", "negative"], default="auto")
    parser.add_argument("--min-via-x", type=float, default=None)
    parser.add_argument("--max-via-x", type=float, default=None)
    parser.add_argument("--min-via-y", type=float, default=None)
    parser.add_argument("--max-via-y", type=float, default=None)
    parser.add_argument("--samples", type=int, default=121)
    parser.add_argument("--table-z", default="auto")
    parser.add_argument("--table-clearance-threshold-m", type=float, default=0.06)
    parser.add_argument(
        "--clearance-links",
        default=",".join(DEFAULT_CLEARANCE_LINKS),
        help="comma-separated links used for obstacle/table clearance statistics",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
