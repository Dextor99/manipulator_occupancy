#!/usr/bin/env python3
"""Guarded executor for a 6.5.2 planar static-avoidance candidate.

The script reads ``planar_execution_waypoints.json`` from a planning trial and
executes the sampled candidate curve:

    candidate samples[0] -> samples[1] -> ... -> samples[-1]

``P1_via`` is treated only as a Bezier control point, not as a real mandatory
execution waypoint.  The script keeps the current TCP orientation, replaces
only XYZ, and requires operator confirmation during execution.  It also saves
RealSense RGB/depth/key point-cloud snapshots for later figures.

Default mode is dry-run.  No robot command is sent unless both ``--execute`` and
the exact operator phrase are provided.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from camera.realsense_pipeline_reader import RealSensePipelineReader  # noqa: E402
from execute_652_planar_y_guarded import (  # noqa: E402
    cartesian_distance,
    call_cartesian_motion,
    check_pose_limits,
    fmt_pose,
    json_default,
    load_robot_module,
    require_confirmation,
    wait_for_pose,
    write_json,
)


DEFAULT_TRIAL = (
    ROOT
    / "results"
    / "new"
    / "6_5"
    / "6_5_2"
    / "planar_static_live"
    / "rs1_lateral_table_obstacle"
    / "trials"
    / "rs1_lateral_table_obstacle_r02"
)
REQUIRED_OPERATOR_PHRASE = "CCRO_652_PLANAR_CANDIDATE_APPROVED"


def load_candidate_payload(trial_dir: Path) -> tuple[dict[str, list[float]], np.ndarray]:
    path = trial_dir / "planar_execution_waypoints.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate = payload.get("candidate_xyz", {})
    waypoints = {
        "P0_start": [float(v) for v in candidate["P0_start"]],
        "P1_via": [float(v) for v in candidate["P1_via"]],
        "P2_goal": [float(v) for v in candidate["P2_goal"]],
    }
    samples = payload.get("samples", {}).get("candidate")
    if samples:
        path_xyz = np.asarray(samples, dtype=np.float64)
    else:
        p0 = np.asarray(waypoints["P0_start"], dtype=np.float64)
        p1 = np.asarray(waypoints["P1_via"], dtype=np.float64)
        p2 = np.asarray(waypoints["P2_goal"], dtype=np.float64)
        u = np.linspace(0.0, 1.0, 121)
        path_xyz = (1 - u)[:, None] ** 2 * p0 + 2 * (1 - u)[:, None] * u[:, None] * p1 + u[:, None] ** 2 * p2
    if path_xyz.ndim != 2 or path_xyz.shape[1] != 3 or len(path_xyz) < 2:
        raise RuntimeError(f"invalid candidate samples in {path}")
    return waypoints, path_xyz


def resample_by_spacing(path_xyz: np.ndarray, max_segment_m: float) -> np.ndarray:
    if max_segment_m <= 0:
        return path_xyz
    chosen = [path_xyz[0]]
    last = path_xyz[0]
    for point in path_xyz[1:-1]:
        if float(np.linalg.norm(point - last)) >= max_segment_m:
            chosen.append(point)
            last = point
    chosen.append(path_xyz[-1])
    return np.vstack(chosen)


def make_pose_from_xyz(base_pose: list[float], xyz: list[float]) -> list[float]:
    pose = list(base_pose)
    pose[:3] = [float(v) for v in xyz]
    return pose


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_camera_pointcloud(path: Path, points_cam: np.ndarray, *, max_points: int = 5000) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = np.asarray(points_cam, dtype=np.float64)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    pts = pts[(pts[:, 2] > 0.05) & (pts[:, 2] < 2.0)]
    if len(pts) > max_points:
        rng = np.random.default_rng(20260652)
        pts = pts[rng.choice(len(pts), max_points, replace=False)]

    fig = plt.figure(figsize=(8.0, 6.2), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    if len(pts):
        colors = np.clip(pts[:, 2], np.percentile(pts[:, 2], 5), np.percentile(pts[:, 2], 95))
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.8, c=colors, cmap="viridis", alpha=0.55)
        center = np.mean(pts, axis=0)
        span = max(float(np.max(np.ptp(pts, axis=0))), 0.3)
        ax.set_xlim(center[0] - 0.5 * span, center[0] + 0.5 * span)
        ax.set_ylim(center[1] - 0.5 * span, center[1] + 0.5 * span)
        ax.set_zlim(max(0.0, center[2] - 0.5 * span), center[2] + 0.5 * span)
    ax.set_xlabel("camera X / m")
    ax.set_ylabel("camera Y / m")
    ax.set_zlabel("camera Z / m")
    ax.set_title("RealSense depth point cloud")
    ax.view_init(elev=22, azim=-58)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def capture_rgbd_snapshot(output_dir: Path, label: str, args: argparse.Namespace) -> dict[str, Any]:
    """Save a standalone RealSense snapshot without opening an AUBO state reader."""

    import cv2

    snap_dir = output_dir / "snapshots" / label
    snap_dir.mkdir(parents=True, exist_ok=True)
    reader = RealSensePipelineReader(width=args.width, height=args.height, fps=args.fps)
    try:
        frame = None
        for _ in range(max(1, args.camera_warmup_frames)):
            frame = reader.read()
        assert frame is not None
        color = np.asarray(frame.color)
        depth = np.asarray(frame.depth)
        cv2.imwrite(str(snap_dir / "rgb.png"), color)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imwrite(str(snap_dir / "depth_colormap.png"), depth_colormap)
        np.savez_compressed(snap_dir / "points_cam.npz", points=np.asarray(frame.points_cam, dtype=np.float32))
        render_camera_pointcloud(snap_dir / "pointcloud_cam_view.png", np.asarray(frame.points_cam))
        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "camera_timestamp": frame.camera_timestamp,
            "host_receive_timestamp": frame.host_receive_timestamp,
            "timestamp": frame.timestamp,
            "width": int(color.shape[1]),
            "height": int(color.shape[0]),
            "files": ["rgb.png", "depth_colormap.png", "points_cam.npz", "pointcloud_cam_view.png"],
        }
        write_json(snap_dir / "snapshot_meta.json", meta)
        return meta
    finally:
        reader.stop()


def copy_planning_figures(trial_dir: Path, output_dir: Path) -> None:
    import shutil

    fig_dir = output_dir / "planning_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "top_view_tcp_paths.png",
        "clearance_curves.png",
        "candidate_pose_sequence.png",
        "reference_pose_sequence.png",
        "obstacle_model_pointcloud.png",
    ):
        src = trial_dir / "figures" / name
        if src.exists():
            shutil.copy2(src, fig_dir / name)


def run(args: argparse.Namespace) -> dict[str, Any]:
    trial_dir = args.trial_dir.resolve()
    output_dir = (args.output or (trial_dir / "candidate_execution")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_planning_figures(trial_dir, output_dir)

    waypoints, raw_curve_xyz = load_candidate_payload(trial_dir)
    curve_xyz = resample_by_spacing(raw_curve_xyz, args.max_segment_m)
    log: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "robot_commanded": False,
        "execute_requested": bool(args.execute),
        "operator_phrase_ok": args.operator_phrase == REQUIRED_OPERATOR_PHRASE,
        "required_operator_phrase": REQUIRED_OPERATOR_PHRASE,
        "trial_dir": str(trial_dir),
        "output_dir": str(output_dir),
        "waypoints_xyz": waypoints,
        "curve_sample_count_raw": int(len(raw_curve_xyz)),
        "curve_sample_count_executed": int(len(curve_xyz)),
        "curve_xyz_executed": curve_xyz.tolist(),
        "segments": [],
        "snapshots": [],
        "samples": [],
        "parameters": vars(args),
    }

    if not args.execute:
        log["status"] = "DRY_RUN_NO_ROBOT_COMMAND"
        write_json(output_dir / "execution_plan.json", log)
        print(json.dumps(log, indent=2, ensure_ascii=False, default=json_default))
        return log

    if args.operator_phrase != REQUIRED_OPERATOR_PHRASE:
        log["status"] = "BLOCKED_BAD_OPERATOR_PHRASE"
        write_json(output_dir / "execution_log.json", log)
        raise RuntimeError(f"bad operator phrase; required: {REQUIRED_OPERATOR_PHRASE}")

    if args.capture_snapshots:
        log["snapshots"].append(capture_rgbd_snapshot(output_dir, "before_start", args))

    robot, so_path = load_robot_module(args.sdk_dir)
    log["sdk_so"] = str(so_path)
    print(f"[sdk] loaded: {so_path}")

    try:
        if not robot.init():
            raise RuntimeError("robot.init() failed")
        current_pose = list(robot.get_status())
        current_joint = list(robot.get_joint())
        log["initial_pose"] = current_pose
        log["initial_joint"] = current_joint

        p0_pose = make_pose_from_xyz(current_pose, curve_xyz[0].tolist())
        start_error = cartesian_distance(current_pose, p0_pose)
        log["start_position_error_m"] = start_error
        print(f"[current] {fmt_pose(current_pose)}")
        print(f"[expected P0] {fmt_pose(p0_pose)}")
        print(f"[start error] {start_error:.4f} m")
        if start_error > args.start_tolerance_m:
            raise RuntimeError(
                f"current TCP is not at planned P0; error={start_error:.4f}m > "
                f"{args.start_tolerance_m:.4f}m"
            )

        require_confirmation(
            True,
            (
                f"Execute sampled candidate curve with {len(curve_xyz) - 1} small segments. "
                "P1_via is only a curve control point, not a mandatory waypoint."
            ),
        )
        for index, xyz in enumerate(curve_xyz[1:], start=1):
            label = f"curve_sample_{index:03d}"
            pose = make_pose_from_xyz(current_pose, xyz.tolist())
            snapshot_label = None
            if index == max(1, len(curve_xyz) // 2):
                snapshot_label = "after_mid_curve"
            if index == len(curve_xyz) - 1:
                snapshot_label = "after_goal"
            check_pose_limits(pose, args, label)
            if args.confirm_every > 0 and ((index - 1) % args.confirm_every == 0):
                require_confirmation(
                    True,
                    f"{label}: move TCP to sampled curve point {index}/{len(curve_xyz)-1}: {fmt_pose(pose)}.",
                )
            seg = call_cartesian_motion(robot, pose, args, label)
            log["robot_commanded"] = True
            log["segments"].append(seg)
            print(f"[{label}] reached {fmt_pose(seg['actual_pose_after'])}")
            if args.capture_snapshots and snapshot_label is not None:
                log["snapshots"].append(capture_rgbd_snapshot(output_dir, snapshot_label, args))

        log["final_pose"] = list(robot.get_status())
        log["final_joint"] = list(robot.get_joint())
        log["status"] = "COMPLETED"
    except Exception as exc:
        log["status"] = "ABORTED_OR_FAILED"
        log["error"] = str(exc)
        raise
    finally:
        try:
            robot.log_out()
        except Exception:
            pass
        write_json(output_dir / "execution_log.json", log)
        save_csv(output_dir / "execution_segments.csv", log.get("segments", []))
        print(f"[log] {output_dir / 'execution_log.json'}")
    return log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, default=DEFAULT_TRIAL)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--operator-phrase", default="")
    parser.add_argument("--sdk-dir", default=None)
    parser.add_argument("--line-velocity-m-s", type=float, default=0.018)
    parser.add_argument("--line-acc-m-s2", type=float, default=0.045)
    parser.add_argument("--settle-s", type=float, default=0.5)
    parser.add_argument("--poll-s", type=float, default=0.10)
    parser.add_argument("--motion-timeout-s", type=float, default=80.0)
    parser.add_argument("--pose-tolerance-m", type=float, default=0.015)
    parser.add_argument("--start-tolerance-m", type=float, default=0.030)
    parser.add_argument("--max-segment-m", type=float, default=0.04)
    parser.add_argument("--confirm-every", type=int, default=1, help="ask for Enter every N curve segments; 0 asks only once before the curve")
    parser.add_argument("--allow-movel-fallback", action="store_true")
    parser.add_argument("--min-x", type=float, default=-0.08)
    parser.add_argument("--max-x", type=float, default=0.9)
    parser.add_argument("--min-y", type=float, default=-0.55)
    parser.add_argument("--max-y", type=float, default=0.55)
    parser.add_argument("--min-z", type=float, default=0.25)
    parser.add_argument("--max-z", type=float, default=0.9)
    parser.add_argument("--capture-snapshots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-warmup-frames", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
