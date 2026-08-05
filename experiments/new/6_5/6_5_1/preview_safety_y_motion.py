#!/usr/bin/env python3
"""Preview the Cartesian Y-axis motion used by robot/safety_guided_motion.py.

This program never sends robot motion commands.  It only builds the TCP path
that motion_worker.py would use after homing:

    center pose -> y_max -> y_min -> ...

For the command

    python robot/safety_guided_motion.py --real-robot --range 0.20

the two endpoints are:

    Y_minus = center_y - 0.20
    Y_plus  = center_y + 0.20

The safest way to use this preview is to put the robot in the intended center
pose first, then run with --center-source live-current.  The script reads the
current TCP pose and joint state, draws the Y-axis path, and saves JSON/CSV/PNG.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.robot_state_reader import RealRobotStateReader  # noqa: E402
from robot.urdf_model import URDFModel  # noqa: E402


DEFAULT_OUTPUT = ROOT / "results" / "new" / "6_5" / "6_5_1" / "real_platform" / "safety_y_preview"
HOME_JOINTS = np.deg2rad([0.0, 0.0, 90.0, 0.0, 90.0, 0.0])
JOINT_NAMES = [
    "shoulder_joint",
    "upperArm_joint",
    "foreArm_joint",
    "wrist1_joint",
    "wrist2_joint",
    "wrist3_joint",
]


@dataclass
class CenterState:
    source: str
    pose_xyzrpy: np.ndarray
    joints: np.ndarray | None


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def load_live_current() -> CenterState:
    reader = RealRobotStateReader()
    if not reader.connect():
        raise RuntimeError("failed to connect AUBO state reader; no robot command was sent")
    try:
        pose = reader.get_end_effector_pose()
        joints_dict = reader.get_joint_positions()
        if pose is None:
            raise RuntimeError("AUBO get_status returned no TCP pose")
        xyzrpy = np.concatenate([np.asarray(pose["pos"], dtype=np.float64), np.asarray(pose["rpy"], dtype=np.float64)])
        joints = np.asarray([float(joints_dict[name]) for name in JOINT_NAMES], dtype=np.float64)
        return CenterState("live-current", xyzrpy, joints)
    finally:
        reader.disconnect()


def load_manual_pose(values: list[float]) -> CenterState:
    if len(values) != 6:
        raise ValueError("--center-pose must contain 6 values: x y z rx ry rz")
    return CenterState("manual-pose", np.asarray(values, dtype=np.float64), None)


def load_urdf_home(urdf_path: Path, tcp_link: str) -> CenterState:
    urdf = URDFModel(str(urdf_path))
    joints = {name: float(HOME_JOINTS[index]) for index, name in enumerate(JOINT_NAMES)}
    joints["left_joint"] = -0.02
    joints["right_joint"] = -0.02
    fk = urdf.link_transforms(joints)
    if tcp_link not in fk:
        raise ValueError(f"tcp link `{tcp_link}` not found in URDF FK: {sorted(fk)}")
    transform = fk[tcp_link]
    xyz = transform[:3, 3]
    # URDF FK rotation is not converted here because the SDK path preserves the
    # actual RPY read from get_status.  For preview geometry, position is primary.
    xyzrpy = np.array([xyz[0], xyz[1], xyz[2], 0.0, 0.0, 0.0], dtype=np.float64)
    return CenterState(f"urdf-home:{tcp_link}", xyzrpy, HOME_JOINTS.copy())


def make_path(center: np.ndarray, range_m: float, x_offset: float, samples: int, start_is_plus: bool) -> dict[str, Any]:
    adjusted = center.copy()
    adjusted[0] += x_offset
    y_minus = adjusted.copy()
    y_minus[1] -= range_m
    y_plus = adjusted.copy()
    y_plus[1] += range_m
    # B0 naming: one endpoint, center, the other endpoint.
    start = y_plus if start_is_plus else y_minus
    goal = y_minus if start_is_plus else y_plus
    mid = adjusted
    points = np.linspace(y_minus[:3], y_plus[:3], samples)
    return {
        "center": adjusted,
        "y_minus": y_minus,
        "y_plus": y_plus,
        "start": start,
        "mid": mid,
        "goal": goal,
        "points": points,
        "first_motion_target": y_plus,
    }


def write_path_csv(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "x", "y", "z"])
        for index, point in enumerate(points):
            writer.writerow([index, f"{point[0]:.8f}", f"{point[1]:.8f}", f"{point[2]:.8f}"])


def plot_preview(output: Path, payload: dict[str, Any], workspace: list[float] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = np.asarray(payload["path_points"], dtype=np.float64)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color="tab:blue", linewidth=3, label="TCP Y path")
    markers = {
        "start": ("tab:green", "o"),
        "mid": ("tab:orange", "^"),
        "goal": ("tab:red", "s"),
        "first_motion_target": ("tab:purple", "x"),
    }
    for name, (color, marker) in markers.items():
        pose = np.asarray(payload[name]["pose_xyzrpy"], dtype=np.float64)
        ax.scatter([pose[0]], [pose[1]], [pose[2]], color=color, marker=marker, s=80, label=name)
        ax.text(pose[0], pose[1], pose[2], f" {name}", color=color)

    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.set_title("safety_guided_motion.py Cartesian Y-axis TCP preview")
    ax.legend()
    ax.grid(True, alpha=0.3)

    all_points = np.vstack(
        [
            points,
            np.asarray([payload[name]["pose_xyzrpy"][:3] for name in ("start", "mid", "goal")], dtype=np.float64),
        ]
    )
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(float(np.max(maxs - mins)) * 0.65, 0.15)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center-source", choices=["live-current", "manual-pose", "urdf-home"], default="live-current")
    parser.add_argument("--center-pose", nargs=6, type=float, metavar=("X", "Y", "Z", "RX", "RY", "RZ"))
    parser.add_argument("--urdf", type=Path, default=ROOT / "urdf" / "aubo_i16_gripper.urdf")
    parser.add_argument("--tcp-link", default="gripper_base_link")
    parser.add_argument("--range", dest="range_m", type=float, default=0.20)
    parser.add_argument("--motion-x-offset", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--start-is-plus", action="store_true", help="Map B0 start to +Y endpoint instead of -Y endpoint.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.center_source == "live-current":
        center = load_live_current()
    elif args.center_source == "manual-pose":
        if args.center_pose is None:
            raise SystemExit("--center-source manual-pose requires --center-pose X Y Z RX RY RZ")
        center = load_manual_pose(args.center_pose)
    else:
        center = load_urdf_home(args.urdf, args.tcp_link)

    path = make_path(
        center.pose_xyzrpy,
        range_m=args.range_m,
        x_offset=args.motion_x_offset,
        samples=max(args.samples, 2),
        start_is_plus=args.start_is_plus,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    def pose_payload(pose: np.ndarray) -> dict[str, Any]:
        return {
            "pose_xyzrpy": pose,
            "x": float(pose[0]),
            "y": float(pose[1]),
            "z": float(pose[2]),
            "rx": float(pose[3]),
            "ry": float(pose[4]),
            "rz": float(pose[5]),
        }

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "robot_commanded": False,
        "source": center.source,
        "range_m": args.range_m,
        "motion_x_offset_m": args.motion_x_offset,
        "b0_mapping": "start=Y_plus, mid=center, goal=Y_minus" if args.start_is_plus else "start=Y_minus, mid=center, goal=Y_plus",
        "note": (
            "This preview shows the TCP Cartesian line used by safety_guided_motion.py. "
            "It is not a full-arm collision certificate because the controller's internal IK path is not replayed here."
        ),
        "center_before_x_offset": pose_payload(center.pose_xyzrpy),
        "current_joints_rad": None if center.joints is None else center.joints,
        "home_joints_rad_for_reference": HOME_JOINTS,
        "start": pose_payload(path["start"]),
        "mid": pose_payload(path["mid"]),
        "goal": pose_payload(path["goal"]),
        "y_minus": pose_payload(path["y_minus"]),
        "y_plus": pose_payload(path["y_plus"]),
        "first_motion_target": pose_payload(path["first_motion_target"]),
        "path_points": path["points"],
    }
    write_json(output / "preview_safety_y_motion.json", payload)
    write_path_csv(output / "preview_safety_y_path.csv", path["points"])
    plot_preview(output / "preview_safety_y_motion.png", payload)

    print(json.dumps({
        "robot_commanded": False,
        "output_dir": str(output),
        "source": center.source,
        "mapping": payload["b0_mapping"],
        "start_xyz": payload["start"]["pose_xyzrpy"][:3],
        "mid_xyz": payload["mid"]["pose_xyzrpy"][:3],
        "goal_xyz": payload["goal"]["pose_xyzrpy"][:3],
        "first_motion_target_xyz": payload["first_motion_target"]["pose_xyzrpy"][:3],
    }, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
