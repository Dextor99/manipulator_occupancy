#!/usr/bin/env python3
"""Capture representative visual snapshots for revised 6.5.1 figures.

This script is read-only with respect to the robot.  It connects RealSense and
AUBO state feedback, runs the same self-filtering / clustering / tracking /
risk pipeline as run_651_perception_capture.py, and saves selected event frames:

    rgb.png
    depth_colormap.png
    rgb_overlay.png
    scene_points.npz
    robot_points.npz
    clusters.npz
    snapshot_meta.json

Typical usage:

    python experiments/new/6_5/6_5_1/capture_651_visual_snapshots.py \
      --scenario D1 --repeat 1 --output results/new/6_5/6_5_1/perception_visual_snapshots
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.geometry_fit import make_occupancy_object  # noqa: E402
from perception.occupancy_tracker import OccupancyTracker  # noqa: E402
from risk.prediction import predict_risk_spheres  # noqa: E402
from risk.safety_policy import SafetyPolicy  # noqa: E402
from test_clustering_filtering import FastClusteringFilter, TemporalDenoiser  # noqa: E402
from test_remove_robot_points_fast import (  # noqa: E402
    MAX_RAW_POINTS,
    SceneProcessor,
    crop_workspace,
    voxel_downsample,
)
from utils.config import load_config_dir  # noqa: E402

from run_651_perception_capture import (  # noqa: E402
    JOINT_NAMES,
    SCENARIOS,
    load_surface_model,
    nearest_cluster_to_links,
    nearest_sphere_to_links,
    prompt_phase,
    q_from_reader,
    risk_color_level,
    write_json,
)


DEFAULT_OUTPUT = ROOT / "results" / "new" / "6_5" / "6_5_1" / "perception_visual_snapshots"


@dataclass
class RawSnapshot:
    color: np.ndarray
    depth: np.ndarray
    timestamp: float
    camera_timestamp: float | None = None
    host_receive_timestamp: float | None = None


@dataclass
class FramePacket:
    index: int
    phase: str
    raw_frame: Any
    scene_points: np.ndarray
    robot_points: np.ndarray
    cluster_result: Any
    clusters: list[Any]
    q: np.ndarray
    joints: dict[str, float]
    row: dict[str, Any]


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def process_visual_frame(processor: SceneProcessor, args: argparse.Namespace) -> tuple[Any, np.ndarray, np.ndarray]:
    """Read one RGB-D frame and run the same preprocessing/removal steps."""
    raw_frame = processor._reader.read()
    raw = raw_frame.points_cam
    if len(raw):
        if len(raw) > MAX_RAW_POINTS:
            idx = np.random.default_rng().choice(len(raw), MAX_RAW_POINTS, replace=False)
            raw = raw[idx]
        pts = raw @ processor._cam_xf.R.T + processor._cam_xf.t
        pts = crop_workspace(pts, processor._workspace)
        pts = voxel_downsample(pts, args.voxel_size)
    else:
        pts = np.empty((0, 3))

    robot_pts = np.empty((0, 3))
    if len(pts) > 0:
        angles = processor._state_reader.get_joint_positions()
        fk = processor._urdf.link_transforms(angles)
        pts, robot_pts = processor._remover.remove(pts, fk)
    return raw_frame, pts, robot_pts


def to_float(value: Any, default: float = math.inf) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def packet_score(packet: FramePacket, event: str) -> float:
    row = packet.row
    nearest_points = int(row.get("nearest_cluster_points", 0) or 0)
    # Avoid selecting tiny residual clusters as paper snapshots. The formal
    # numeric logs still keep them; this filter is only for visual key frames.
    if event != "risk_cleared" and nearest_points < 80:
        return -1.0
    if event == "static_detected":
        return float(int(row["stable_track_count"]) > 0) * 100.0 - to_float(row["nearest_distance_m"], 9.0)
    if event == "predicted_risk_onset":
        if row["risk_state_predicted"] == "SAFE":
            return -1.0
        return 1000.0 - packet.index
    if event == "current_risk_onset":
        if row["risk_state_current"] == "SAFE":
            return -1.0
        return 1000.0 - packet.index
    if event == "minimum_clearance":
        return -to_float(row["nearest_distance_m"], 9.0)
    if event == "risk_cleared":
        if packet.phase != "post_removed":
            return -1.0
        if row["risk_state_current"] == "SAFE" and row["risk_state_predicted"] == "SAFE" and int(row["cluster_count"]) == 0:
            return 1000.0 - packet.index
        return -1.0
    return -1.0


def wanted_events(scenario: str) -> list[str]:
    wanted = ["minimum_clearance", "risk_cleared"]
    if scenario.startswith("S"):
        wanted.insert(0, "static_detected")
    else:
        wanted.insert(0, "current_risk_onset")
        wanted.insert(0, "predicted_risk_onset")
    return wanted


def clone_packet(packet: FramePacket, *, max_robot_points: int = 12000) -> FramePacket:
    """Copy only the data needed for snapshots, not the full RealSense point map."""
    raw = packet.raw_frame
    raw_snapshot = RawSnapshot(
        color=np.asarray(raw.color).copy(),
        depth=np.asarray(raw.depth).copy(),
        timestamp=float(getattr(raw, "timestamp", 0.0)),
        camera_timestamp=getattr(raw, "camera_timestamp", None),
        host_receive_timestamp=getattr(raw, "host_receive_timestamp", None),
    )
    robot_points = packet.robot_points
    if len(robot_points) > max_robot_points:
        idx = np.random.default_rng(20260651).choice(len(robot_points), max_robot_points, replace=False)
        robot_points = robot_points[idx]
    copied_clusters = []
    for cluster in packet.clusters:
        copied = type("SnapshotCluster", (), {})()
        copied.points = np.asarray(cluster.points, dtype=np.float64).copy()
        copied.center = np.asarray(cluster.center, dtype=np.float64).copy()
        copied_clusters.append(copied)
    return FramePacket(
        index=packet.index,
        phase=packet.phase,
        raw_frame=raw_snapshot,
        scene_points=packet.scene_points.copy(),
        robot_points=robot_points.copy(),
        cluster_result=None,
        clusters=copied_clusters,
        q=packet.q.copy(),
        joints=dict(packet.joints),
        row=dict(packet.row),
    )


def project_points(points_base: np.ndarray, processor: SceneProcessor) -> np.ndarray:
    if len(points_base) == 0:
        return np.empty((0, 2), dtype=np.int32)
    reader = processor._reader
    intrinsic = getattr(reader, "intrinsic", None)
    if not intrinsic:
        return np.empty((0, 2), dtype=np.int32)
    pts_cam = (points_base - processor._cam_xf.t) @ processor._cam_xf.R
    z = pts_cam[:, 2]
    valid = z > 1.0e-6
    pts_cam = pts_cam[valid]
    if len(pts_cam) == 0:
        return np.empty((0, 2), dtype=np.int32)
    z = pts_cam[:, 2]
    u = intrinsic["fx"] * pts_cam[:, 0] / z + intrinsic["cx"]
    v = intrinsic["fy"] * pts_cam[:, 1] / z + intrinsic["cy"]
    width, height = intrinsic["width"], intrinsic["height"]
    pix = np.column_stack([u, v])
    inside = (pix[:, 0] >= 0) & (pix[:, 0] < width) & (pix[:, 1] >= 0) & (pix[:, 1] < height)
    return pix[inside].astype(np.int32)


def save_snapshot(
    packet: FramePacket,
    event: str,
    scenario: str,
    repeat: int,
    out_dir: Path,
    processor: SceneProcessor,
) -> None:
    import cv2

    snap_dir = out_dir / f"{scenario}_r{repeat:02d}_{event}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    color = np.asarray(packet.raw_frame.color)
    depth = np.asarray(packet.raw_frame.depth)
    cv2.imwrite(str(snap_dir / "rgb.png"), color)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
    cv2.imwrite(str(snap_dir / "depth_colormap.png"), depth_colormap)

    overlay = color.copy()
    overlay_nearest = color.copy()
    cluster_arrays: dict[str, np.ndarray] = {}
    centers = []
    bboxes = []
    colors = [(0, 255, 255), (0, 200, 0), (255, 120, 0), (255, 0, 255)]
    nearest_idx = packet.row.get("nearest_cluster_index", "")
    nearest_idx = None if nearest_idx == "" else int(nearest_idx)
    for i, cluster in enumerate(packet.clusters):
        pts = np.asarray(cluster.points, dtype=np.float64)
        cluster_arrays[f"cluster_{i:02d}"] = pts
        centers.append(np.asarray(cluster.center, dtype=np.float64))
        pixels = project_points(pts, processor)
        if len(pixels) == 0:
            bboxes.append(None)
            continue
        x0, y0 = pixels.min(axis=0)
        x1, y1 = pixels.max(axis=0)
        bboxes.append([int(x0), int(y0), int(x1), int(y1)])
        color_i = colors[i % len(colors)]
        cv2.rectangle(overlay, (int(x0), int(y0)), (int(x1), int(y1)), color_i, 2)
        cv2.putText(overlay, f"C{i}", (int(x0), max(20, int(y0) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_i, 2)
        if nearest_idx == i:
            cv2.rectangle(overlay_nearest, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 3)
            cv2.putText(overlay_nearest, "nearest risk cluster", (int(x0), max(20, int(y0) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    text_lines = [
        f"{scenario} r{repeat:02d} {event}",
        f"phase={packet.phase} frame={packet.index}",
        f"d={packet.row['nearest_distance_m']} link={packet.row['nearest_link']}",
        f"cur={packet.row['risk_state_current']} pred={packet.row['risk_state_predicted']}",
    ]
    y = 28
    for line in text_lines:
        cv2.putText(overlay, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 255), 2)
        cv2.putText(overlay_nearest, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 255), 2)
        y += 28
    cv2.imwrite(str(snap_dir / "rgb_overlay.png"), overlay)
    cv2.imwrite(str(snap_dir / "rgb_overlay_nearest.png"), overlay_nearest)

    np.savez_compressed(snap_dir / "scene_points.npz", points=packet.scene_points)
    np.savez_compressed(snap_dir / "robot_points.npz", points=packet.robot_points)
    np.savez_compressed(
        snap_dir / "clusters.npz",
        centers=np.asarray(centers, dtype=np.float64) if centers else np.empty((0, 3)),
        bboxes_xyxy=np.asarray([b for b in bboxes if b is not None], dtype=np.int32) if any(b is not None for b in bboxes) else np.empty((0, 4), dtype=np.int32),
        **cluster_arrays,
    )
    render_pointcloud_snapshot(
        snap_dir / "pointcloud_view.png",
        packet.scene_points,
        packet.robot_points,
        [np.asarray(cluster.points, dtype=np.float64) for cluster in packet.clusters],
        nearest_idx,
    )

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "repeat": repeat,
        "event": event,
        "phase": packet.phase,
        "frame_index": packet.index,
        "robot_commanded": False,
        "timestamp": packet.row["timestamp"],
        "camera_timestamp": getattr(packet.raw_frame, "camera_timestamp", None),
        "host_receive_timestamp": getattr(packet.raw_frame, "host_receive_timestamp", None),
        "row": packet.row,
        "joint_positions": {name: float(packet.q[i]) for i, name in enumerate(JOINT_NAMES)},
        "cluster_count": len(packet.clusters),
        "cluster_bboxes_xyxy": bboxes,
        "files": [
            "rgb.png",
            "depth_colormap.png",
            "rgb_overlay.png",
            "rgb_overlay_nearest.png",
            "pointcloud_view.png",
            "scene_points.npz",
            "robot_points.npz",
            "clusters.npz",
        ],
    }
    write_json(snap_dir / "snapshot_meta.json", meta)


def render_pointcloud_snapshot(
    path: Path,
    scene_points: np.ndarray,
    robot_points: np.ndarray,
    clusters: list[np.ndarray],
    nearest_idx: int | None,
) -> None:
    """Render a compact 3D point-cloud view for paper figure inspection."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    rng = np.random.default_rng(20260651)

    def sample(points: np.ndarray, n: int) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if len(points) <= n:
            return points
        idx = rng.choice(len(points), n, replace=False)
        return points[idx]

    robot = sample(robot_points, 2500)
    scene = sample(scene_points, 2500)
    fig = plt.figure(figsize=(7.2, 5.2), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    if len(robot):
        ax.scatter(robot[:, 0], robot[:, 1], robot[:, 2], s=0.6, c="#b0b0b0", alpha=0.18, label="robot surface")
    if len(scene):
        ax.scatter(scene[:, 0], scene[:, 1], scene[:, 2], s=0.8, c="#4c78a8", alpha=0.12, label="filtered scene")

    palette = ["#f58518", "#54a24b", "#e45756", "#b279a2"]
    for i, pts in enumerate(clusters):
        pts = sample(pts, 1400)
        if len(pts) == 0:
            continue
        color = "#d62728" if nearest_idx == i else palette[i % len(palette)]
        size = 7.0 if nearest_idx == i else 3.5
        alpha = 0.9 if nearest_idx == i else 0.35
        label = "nearest risk cluster" if nearest_idx == i else f"cluster {i}"
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size, c=color, alpha=alpha, label=label)
        center = pts.mean(axis=0)
        ax.text(center[0], center[1], center[2], f"C{i}", color=color, fontsize=8)

    all_pts = np.vstack([arr for arr in [robot, scene, *clusters] if len(arr)]) if any(len(arr) for arr in [robot, scene, *clusters]) else np.empty((0, 3))
    if len(all_pts):
        lo = np.percentile(all_pts, 2, axis=0)
        hi = np.percentile(all_pts, 98, axis=0)
        center = (lo + hi) * 0.5
        radius = max(float(np.max(hi - lo)) * 0.55, 0.1)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius * 0.65, center[2] + radius * 0.65)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.view_init(elev=22, azim=-55)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def collect_visual_trial(args: argparse.Namespace) -> Path:
    if args.scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {args.scenario}; choose from {sorted(SCENARIOS)}")

    scenario_cfg = SCENARIOS[args.scenario]
    output_root = args.output.resolve()
    trial_dir = output_root / "trials" / f"{args.scenario}_{scenario_cfg['name']}_r{args.repeat:02d}"
    snapshots_dir = trial_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

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

    selected: dict[str, FramePacket] = {}
    selected_scores: dict[str, float] = {event: -math.inf for event in wanted_events(args.scenario)}
    frame_index = 0

    try:
        for phase, default_duration in scenario_cfg["phases"]:
            duration = args.duration if args.duration is not None else default_duration
            prompt_phase(args.scenario, phase, duration, args.no_prompt)
            phase_start = time.time()
            while time.time() - phase_start < duration:
                raw_frame, scene_points, robot_points = process_visual_frame(processor, args)
                timestamp = float(getattr(raw_frame, "timestamp", time.time()))
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
                    make_occupancy_object(cluster.points, timestamp=timestamp, margin=float(safety.get("shape_margin", 0.02)))
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
                current_best = nearest_cluster_to_links(surface_model, q, clusters, density=args.surface_density)
                predicted_best = nearest_sphere_to_links(surface_model, q, risk_spheres, density=args.surface_density)
                stable_speeds = [float(np.linalg.norm(obj.velocity)) for obj in stable]
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
                }
                packet = FramePacket(
                    index=frame_index,
                    phase=phase,
                    raw_frame=raw_frame,
                    scene_points=scene_points,
                    robot_points=robot_points,
                    cluster_result=cluster_result,
                    clusters=list(clusters),
                    q=q,
                    joints=dict(joints),
                    row=row,
                )
                for event in selected_scores:
                    score = packet_score(packet, event)
                    if score > selected_scores[event] and score > -1.0:
                        selected[event] = clone_packet(packet)
                        selected_scores[event] = score
                frame_index += 1
    finally:
        processor.stop()

    for event, packet in selected.items():
        save_snapshot(packet, event, args.scenario, args.repeat, snapshots_dir, processor)

    index = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "robot_commanded": False,
        "scenario": args.scenario,
        "repeat": args.repeat,
        "frames_processed": frame_index,
        "selected_events": {event: packet.index for event, packet in selected.items()},
        "snapshot_root": str(snapshots_dir),
    }
    write_json(trial_dir / "snapshot_index.json", index)
    return trial_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture visual key frames for 6.5.1 perception figures.")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    parser.add_argument("--urdf", type=Path, default=ROOT / "urdf" / "aubo_i16_gripper.urdf")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--self-filter-threshold", type=float, default=0.08)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--cluster-eps", type=float, default=0.06)
    parser.add_argument("--cluster-min-samples", type=int, default=15)
    parser.add_argument("--cluster-min-points", type=int, default=30)
    parser.add_argument("--cluster-min-volume", type=float, default=0.001)
    parser.add_argument("--remove-planes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plane-dist", type=float, default=0.02)
    parser.add_argument("--max-planes", type=int, default=1)
    parser.add_argument("--temporal-denoise", action="store_true", default=True)
    parser.add_argument("--denoise-voxel", type=float, default=0.02)
    parser.add_argument("--denoise-conf", type=int, default=2)
    parser.add_argument("--denoise-decay", type=float, default=0.92)
    parser.add_argument("--surface-density", choices=["coarse", "medium", "dense"], default="coarse")
    args = parser.parse_args()

    trial_dir = collect_visual_trial(args)
    print(json.dumps({"robot_commanded": False, "output_dir": str(trial_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
