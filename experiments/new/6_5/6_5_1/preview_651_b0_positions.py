#!/usr/bin/env python3
"""Preview the B0 start/mid/goal positions used by the 6.5.1 baseline.

This script sends no robot commands.  It reconstructs the exact positions that
the disabled --auto-position-b0 path used:

    start = ccro_stage2.yaml trajectory q_start
    mid   = midpoint of the generated baseline NUBS trajectory
    goal  = ccro_stage2.yaml trajectory q_goal

To expose the dangerous part that caused trouble in practice, it can also read
the current live AUBO joint state and simulate the blocking movej sequence:

    current -> start -> mid -> goal

The simulated movej path is a joint-linear approximation.  It is not a formal
certificate of the controller's internal interpolation, but it is much better
than looking only at the three endpoint configurations.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_ccro_stage2 import _baseline, _load, _states  # noqa: E402
from planning.robot_surface_model import RobotSurfaceModel  # noqa: E402
from robot.robot_state_reader import RealRobotStateReader  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "ccro_stage2.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "new" / "6_5" / "6_5_1" / "real_platform" / "b0_position_preview"
JOINT_NAMES = [
    "shoulder_joint",
    "upperArm_joint",
    "foreArm_joint",
    "wrist1_joint",
    "wrist2_joint",
    "wrist3_joint",
]


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


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_surface_model(config: dict[str, Any]) -> RobotSurfaceModel:
    robot = config["robot"]
    surface = config["surface"]
    return RobotSurfaceModel(
        ROOT / robot["urdf_path"],
        robot["joint_names"],
        surface["density_totals"],
        seed=surface["random_seed"],
        min_points_per_link=surface["min_points_per_link"],
        cache_dir=surface["cache_dir"],
        geometry=surface["geometry"],
    )


def b0_targets(config: dict[str, Any]) -> dict[str, np.ndarray]:
    head, tail, durations = _states(config)
    baseline = _baseline(config, head, tail, durations)
    if not baseline.success:
        raise RuntimeError(f"baseline generation failed: {baseline.message}")
    trajectory = baseline.trajectory
    return {
        "start": head[:, 0].copy(),
        "mid": trajectory.evaluate(0.5 * trajectory.total_duration),
        "goal": tail[:, 0].copy(),
    }


def read_live_current_q() -> np.ndarray:
    reader = RealRobotStateReader()
    if not reader.connect():
        raise RuntimeError("failed to connect AUBO state reader; no robot command was sent")
    try:
        joints = reader.get_joint_positions()
        return np.asarray([float(joints[name]) for name in JOINT_NAMES], dtype=np.float64)
    finally:
        reader.disconnect()


def parse_q(values: list[float] | None) -> np.ndarray | None:
    if values is None:
        return None
    if len(values) != 6:
        raise ValueError("expected exactly 6 joint values")
    return np.asarray(values, dtype=np.float64)


def joint_dict(q: np.ndarray) -> dict[str, float]:
    out = {name: float(q[index]) for index, name in enumerate(JOINT_NAMES)}
    out["left_joint"] = -0.02
    out["right_joint"] = -0.02
    return out


def tcp_position(model: RobotSurfaceModel, q: np.ndarray, tcp_link: str) -> np.ndarray:
    fk = model.urdf.link_transforms(joint_dict(q))
    if tcp_link not in fk:
        raise ValueError(f"tcp link `{tcp_link}` not found; available links: {sorted(fk)}")
    return fk[tcp_link][:3, 3].copy()


def interpolate_segment(q0: np.ndarray, q1: np.ndarray, samples: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, max(samples, 2))
    return (1.0 - t[:, None]) * q0[None, :] + t[:, None] * q1[None, :]


def build_preview_path(
    current_q: np.ndarray | None,
    targets: dict[str, np.ndarray],
    *,
    samples_per_segment: int,
) -> list[tuple[str, np.ndarray]]:
    sequence: list[tuple[str, np.ndarray]] = []
    if current_q is not None:
        named = [("current_to_start", current_q, targets["start"])]
    else:
        named = []
    named.extend(
        [
            ("start_to_mid", targets["start"], targets["mid"]),
            ("mid_to_goal", targets["mid"], targets["goal"]),
        ]
    )
    for segment_name, q0, q1 in named:
        samples = interpolate_segment(q0, q1, samples_per_segment)
        for index, q in enumerate(samples):
            if sequence and index == 0:
                continue
            sequence.append((segment_name, q.copy()))
    return sequence


def analyse_path(
    model: RobotSurfaceModel,
    path: list[tuple[str, np.ndarray]],
    *,
    tcp_link: str,
    density: str,
    links: set[str] | None,
    table_z: float,
    table_clearance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    all_tcp: list[np.ndarray] = []
    all_surface_sparse: list[np.ndarray] = []
    min_z = float("inf")
    min_z_row: dict[str, Any] | None = None
    workspace_min = np.array([float("inf"), float("inf"), float("inf")])
    workspace_max = np.array([-float("inf"), -float("inf"), -float("inf")])
    for global_index, (segment, q) in enumerate(path):
        tcp = tcp_position(model, q, tcp_link)
        surface = model.surface(q, density=density, links=links)
        if len(surface):
            sample_stride = max(len(surface) // 600, 1)
            all_surface_sparse.append(surface[::sample_stride])
            workspace_min = np.minimum(workspace_min, surface.min(axis=0))
            workspace_max = np.maximum(workspace_max, surface.max(axis=0))
            local_min_index = int(np.argmin(surface[:, 2]))
            local_min_z = float(surface[local_min_index, 2])
        else:
            local_min_z = float("nan")
        all_tcp.append(tcp)
        row = {
            "index": global_index,
            "segment": segment,
            **{f"q{j+1}_rad": float(q[j]) for j in range(6)},
            **{f"q{j+1}_deg": float(np.rad2deg(q[j])) for j in range(6)},
            "tcp_x": float(tcp[0]),
            "tcp_y": float(tcp[1]),
            "tcp_z": float(tcp[2]),
            "surface_min_z": local_min_z,
            "table_clearance_estimate": local_min_z - table_z,
            "table_clearance_warning": bool(local_min_z < table_z + table_clearance),
        }
        rows.append(row)
        if np.isfinite(local_min_z) and local_min_z < min_z:
            min_z = local_min_z
            min_z_row = row
    summary = {
        "sample_count": len(path),
        "tcp_min": np.vstack(all_tcp).min(axis=0) if all_tcp else np.full(3, np.nan),
        "tcp_max": np.vstack(all_tcp).max(axis=0) if all_tcp else np.full(3, np.nan),
        "surface_aabb_min": workspace_min,
        "surface_aabb_max": workspace_max,
        "surface_min_z": min_z,
        "surface_min_z_sample": min_z_row,
        "table_z": table_z,
        "table_clearance_threshold": table_clearance,
        "table_clearance_warning": bool(min_z < table_z + table_clearance),
    }
    clouds = {
        "tcp": np.vstack(all_tcp) if all_tcp else np.empty((0, 3)),
        "surface_sparse": np.vstack(all_surface_sparse) if all_surface_sparse else np.empty((0, 3)),
    }
    return rows, summary, clouds


def plot_preview(
    output: Path,
    targets: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    clouds: dict[str, np.ndarray],
    *,
    model: RobotSurfaceModel,
    tcp_link: str,
    table_z: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tcp = clouds["tcp"]
    surface = clouds["surface_sparse"]
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    if len(surface):
        ax.scatter(surface[:, 0], surface[:, 1], surface[:, 2], s=1, alpha=0.08, color="gray", label="swept surface samples")
    if len(tcp):
        ax.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2], color="tab:blue", linewidth=2.5, label="TCP path")
    colors = {"start": "tab:green", "mid": "tab:orange", "goal": "tab:red"}
    markers = {"start": "o", "mid": "^", "goal": "s"}
    for name, q in targets.items():
        p = tcp_position(model, q, tcp_link)
        ax.scatter([p[0]], [p[1]], [p[2]], color=colors[name], marker=markers[name], s=90, label=name)
        ax.text(p[0], p[1], p[2], f" {name}", color=colors[name])
    if rows and rows[0]["segment"] == "current_to_start":
        p0 = np.array([rows[0]["tcp_x"], rows[0]["tcp_y"], rows[0]["tcp_z"]])
        ax.scatter([p0[0]], [p0[1]], [p0[2]], color="black", marker="x", s=80, label="current")
        ax.text(p0[0], p0[1], p0[2], " current", color="black")
    if len(tcp):
        xs = np.linspace(float(np.min(tcp[:, 0])), float(np.max(tcp[:, 0])), 2)
        ys = np.linspace(float(np.min(tcp[:, 1])), float(np.max(tcp[:, 1])), 2)
        xx, yy = np.meshgrid(xs, ys)
        zz = np.full_like(xx, table_z)
        ax.plot_surface(xx, yy, zz, alpha=0.12, color="tab:brown")
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.set_title("6.5.1 B0 auto-position path preview (joint-linear approximation)")
    ax.legend(loc="upper left")
    all_points = tcp
    if len(surface):
        all_points = np.vstack([all_points, surface]) if len(all_points) else surface
    if len(all_points):
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)
        center = 0.5 * (mins + maxs)
        radius = max(float(np.max(maxs - mins)) * 0.58, 0.2)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(max(table_z - 0.05, center[2] - radius), center[2] + radius)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--current-source", choices=["none", "live-current", "manual-q"], default="live-current")
    parser.add_argument("--current-q", nargs=6, type=float, metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"))
    parser.add_argument("--samples-per-segment", type=int, default=80)
    parser.add_argument("--density", choices=["coarse", "medium", "dense"], default="coarse")
    parser.add_argument("--tcp-link", default="gripper_base_link")
    parser.add_argument(
        "--links",
        default="shoulder_Link,upperArm_Link,foreArm_Link,wrist1_Link,wrist2_Link,wrist3_Link,gripper_base_link,left_link,right_link",
        help="Comma-separated links used for swept surface checks; default excludes base_link.",
    )
    parser.add_argument("--table-z", type=float, default=0.0)
    parser.add_argument("--table-clearance", type=float, default=0.03)
    args = parser.parse_args()

    config = _load(args.config)
    model = load_surface_model(config)
    targets = b0_targets(config)
    if args.current_source == "live-current":
        current_q = read_live_current_q()
    elif args.current_source == "manual-q":
        current_q = parse_q(args.current_q)
        if current_q is None:
            raise SystemExit("--current-source manual-q requires --current-q Q1 Q2 Q3 Q4 Q5 Q6")
    else:
        current_q = None

    path = build_preview_path(current_q, targets, samples_per_segment=args.samples_per_segment)
    links = {item.strip() for item in args.links.split(",") if item.strip()}
    if not links:
        links = None
    rows, summary, clouds = analyse_path(
        model,
        path,
        tcp_link=args.tcp_link,
        density=args.density,
        links=links,
        table_z=args.table_z,
        table_clearance=args.table_clearance,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = [
        "index",
        "segment",
        *[f"q{j+1}_rad" for j in range(6)],
        *[f"q{j+1}_deg" for j in range(6)],
        "tcp_x",
        "tcp_y",
        "tcp_z",
        "surface_min_z",
        "table_clearance_estimate",
        "table_clearance_warning",
    ]
    write_csv(output / "b0_position_path_samples.csv", rows, fields)
    plot_preview(output / "b0_position_path_preview.png", targets, rows, clouds, model=model, tcp_link=args.tcp_link, table_z=args.table_z)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "robot_commanded": False,
        "config": str(args.config.resolve()),
        "current_source": args.current_source,
        "current_q": current_q,
        "checked_links": None if links is None else sorted(links),
        "target_definition": {
            "start": "ccro_stage2.yaml trajectory.q_start",
            "mid": "baseline NUBS trajectory midpoint at T/2",
            "goal": "ccro_stage2.yaml trajectory.q_goal",
        },
        "targets": {
            name: {
                "q_rad": q,
                "q_deg": np.rad2deg(q),
                "tcp_xyz": tcp_position(model, q, args.tcp_link),
            }
            for name, q in targets.items()
        },
        "path_summary": summary,
        "warning": (
            "This is a joint-linear approximation of the movej sequence. It is a preview, "
            "not permission to execute. Any collision event means the B0 target set must be redesigned."
        ),
    }
    write_json(output / "b0_position_preview.json", payload)
    np.savez_compressed(output / "b0_position_preview_clouds.npz", tcp=clouds["tcp"], surface_sparse=clouds["surface_sparse"])
    print(
        json.dumps(
            {
                "robot_commanded": False,
                "output_dir": str(output),
                "current_source": args.current_source,
                "targets_deg": {name: np.rad2deg(q).round(3).tolist() for name, q in targets.items()},
                "targets_tcp_xyz": {name: tcp_position(model, q, args.tcp_link).round(4).tolist() for name, q in targets.items()},
                "surface_min_z": summary["surface_min_z"],
                "table_clearance_warning": summary["table_clearance_warning"],
                "preview_png": str(output / "b0_position_path_preview.png"),
            },
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )
    )


if __name__ == "__main__":
    main()
