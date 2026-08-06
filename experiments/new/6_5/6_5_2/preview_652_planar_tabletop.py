#!/usr/bin/env python3
"""Preview a safe tabletop planar path for Chapter 6.5.2.

This is an offline safety preview.  It reads a 6.5.2 perception/planning trial,
uses the recorded robot joint posture as the nominal tabletop posture, then
renders:

* a straight horizontal TCP reference path;
* a horizontal curved TCP detour around the observed obstacle;
* translated URDF moving-link surfaces at several path samples;
* table-clearance and obstacle-clearance estimates.

No RealSense or AUBO connection is opened.  No robot command is sent.

The preview deliberately keeps posture fixed and translates the moving-link
surface by the TCP path displacement.  It is a conservative geometry preview
for tabletop motion design, not an IK/controller execution simulation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_ccro_stage2 import _load  # noqa: E402
from planning.robot_surface_model import RobotSurfaceModel  # noqa: E402


DEFAULT_TRIAL = (
    ROOT
    / "results"
    / "new"
    / "6_5"
    / "6_5_2"
    / "rs1_lateral_table_obstacle"
    / "trials"
    / "rs1_lateral_table_obstacle_r01"
)

MOVING_LINKS = (
    "upperArm_Link",
    "foreArm_Link",
    "wrist1_Link",
    "wrist2_Link",
    "wrist3_Link",
    "gripper_base_link",
    "left_link",
    "right_link",
)

DEFAULT_CLEARANCE_LINKS = (
    "wrist2_Link",
    "wrist3_Link",
    "gripper_base_link",
    "left_link",
    "right_link",
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


def make_surface_model(config: dict[str, Any]) -> RobotSurfaceModel:
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


def load_trial_posture(trial_dir: Path) -> np.ndarray:
    obstacle_json = trial_dir / "detected_obstacle.json"
    if obstacle_json.exists():
        payload = json.loads(obstacle_json.read_text(encoding="utf-8"))
        q = payload.get("q_mean")
        if q is not None:
            values = np.asarray(q, dtype=np.float64)
            if values.shape == (6,) and np.all(np.isfinite(values)):
                return values
    summary_json = trial_dir / "summary.json"
    if summary_json.exists():
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        q = payload.get("obstacle_model", {}).get("q_mean")
        if q is not None:
            values = np.asarray(q, dtype=np.float64)
            if values.shape == (6,) and np.all(np.isfinite(values)):
                return values
    raise RuntimeError(f"cannot find q_mean in {trial_dir}")


def parse_home_degrees(value: str) -> np.ndarray:
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 6:
        raise ValueError("--home-joints-deg must contain six comma-separated values")
    return np.deg2rad(np.asarray(parts, dtype=np.float64))


def joint_dict(surface_model: RobotSurfaceModel, q: np.ndarray) -> dict[str, float]:
    return {name: float(q[i]) for i, name in enumerate(surface_model.joint_names)}


def tcp_position(surface_model: RobotSurfaceModel, q: np.ndarray, tcp_link: str) -> np.ndarray:
    fk = surface_model.urdf.link_transforms(joint_dict(surface_model, q))
    if tcp_link not in fk:
        raise KeyError(f"tcp link `{tcp_link}` not in URDF FK")
    return np.asarray(fk[tcp_link][:3, 3], dtype=np.float64)


def normalize_xy(vec: np.ndarray) -> np.ndarray:
    out = np.asarray([vec[0], vec[1], 0.0], dtype=np.float64)
    n = float(np.linalg.norm(out[:2]))
    if n < 1.0e-9:
        raise ValueError("planar direction must have nonzero XY length")
    return out / n


def bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, u: np.ndarray) -> np.ndarray:
    u = u.reshape(-1, 1)
    return (1.0 - u) ** 2 * p0[None, :] + 2.0 * (1.0 - u) * u * p1[None, :] + u**2 * p2[None, :]


def build_paths(
    p_start: np.ndarray,
    p_goal: np.ndarray,
    obstacle_points: np.ndarray,
    *,
    clearance_m: float,
    detour_extra_m: float,
    side: str,
    samples: int,
) -> dict[str, Any]:
    delta_goal = np.asarray(p_goal, dtype=np.float64) - np.asarray(p_start, dtype=np.float64)
    direction = normalize_xy(delta_goal)
    distance_m = float(np.linalg.norm(delta_goal[:2]))
    normal = np.asarray([-direction[1], direction[0], 0.0], dtype=np.float64)
    obs_center = np.mean(obstacle_points, axis=0)
    obs_radius_xy = float(np.max(np.linalg.norm(obstacle_points[:, :2] - obs_center[None, :2], axis=1)))
    signed = float(np.dot(obs_center - p_start, normal))
    if side == "auto":
        side_sign = -1.0 if signed >= 0.0 else 1.0
    elif side == "positive":
        side_sign = 1.0
    else:
        side_sign = -1.0
    along = float(np.clip(np.dot(obs_center - p_start, direction), 0.15 * distance_m, 0.85 * distance_m))
    lateral = side_sign * (obs_radius_xy + clearance_m + detour_extra_m)
    via = p_start + direction * along + normal * lateral
    via[2] = p_start[2]
    p_goal = np.asarray(p_goal, dtype=np.float64).copy()
    p_goal[2] = p_start[2]

    u = np.linspace(0.0, 1.0, samples)
    reference = p_start[None, :] + u[:, None] * (p_goal - p_start)[None, :]
    candidate = bezier(p_start, via, p_goal, u)
    candidate[:, 2] = p_start[2]
    return {
        "p0": p_start,
        "p2": p_goal,
        "via": via,
        "reference": reference,
        "candidate": candidate,
        "obstacle_center": obs_center,
        "obstacle_radius_xy": obs_radius_xy,
        "detour_side_sign": side_sign,
        "direction": direction,
        "normal": normal,
    }


def moving_surface_by_link(surface_model: RobotSurfaceModel, q: np.ndarray, density: str) -> dict[str, np.ndarray]:
    return surface_model.surface_by_link(q, density=density, links=set(MOVING_LINKS))


def stack_surface(surface_by_link: dict[str, np.ndarray]) -> np.ndarray:
    return np.vstack([pts for pts in surface_by_link.values() if len(pts)]) if surface_by_link else np.empty((0, 3))


def parse_links(value: str) -> tuple[str, ...]:
    links = tuple(item.strip() for item in value.split(",") if item.strip())
    if not links:
        raise ValueError("--clearance-links must contain at least one link")
    return links


def clearance_along_path(
    base_surface: np.ndarray,
    path: np.ndarray,
    base_tcp: np.ndarray,
    obstacle_points: np.ndarray,
    table_z: float,
) -> dict[str, Any]:
    tree = cKDTree(obstacle_points)
    rows: list[dict[str, Any]] = []
    min_obs = math.inf
    min_table = math.inf
    for i, point in enumerate(path):
        delta = point - base_tcp
        moved = base_surface + delta[None, :]
        dists, _ = tree.query(moved, k=1)
        obs_clearance = float(np.min(dists))
        table_clearance = float(np.min(moved[:, 2]) - table_z)
        min_obs = min(min_obs, obs_clearance)
        min_table = min(min_table, table_clearance)
        rows.append(
            {
                "index": i,
                "tcp_x": float(point[0]),
                "tcp_y": float(point[1]),
                "tcp_z": float(point[2]),
                "obstacle_clearance_m": obs_clearance,
                "table_clearance_m": table_clearance,
            }
        )
    return {"rows": rows, "min_obstacle_clearance_m": min_obs, "min_table_clearance_m": min_table}


def sample_points(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= count:
        return points
    return points[rng.choice(len(points), count, replace=False)]


def set_equal_axes(ax, arrays: list[np.ndarray]) -> None:
    pts = [np.asarray(a, dtype=np.float64).reshape(-1, 3) for a in arrays if np.asarray(a).size]
    if not pts:
        return
    all_pts = np.vstack(pts)
    lo = np.percentile(all_pts, 1, axis=0)
    hi = np.percentile(all_pts, 99, axis=0)
    center = 0.5 * (lo + hi)
    radius = max(float(np.max(hi - lo)) * 0.58, 0.18)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - 0.65 * radius, center[2] + 0.65 * radius)


def plot_top_view(path: Path, obstacle_points: np.ndarray, paths: dict[str, Any], clearance_m: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = paths["reference"]
    cand = paths["candidate"]
    obs = obstacle_points
    center = paths["obstacle_center"]
    fig, ax = plt.subplots(figsize=(7.6, 6.6), dpi=180)
    ax.scatter(obs[:, 0], obs[:, 1], s=4, c="#d62728", alpha=0.35, label="observed obstacle")
    circle = plt.Circle(center[:2], paths["obstacle_radius_xy"] + clearance_m, fill=False, color="#d62728", linestyle="--", linewidth=1.4, label="obstacle + clearance")
    ax.add_patch(circle)
    ax.plot(ref[:, 0], ref[:, 1], "--", color="#777777", linewidth=2, label="straight reference")
    ax.plot(cand[:, 0], cand[:, 1], "-", color="#1f77b4", linewidth=2.4, label="planar detour candidate")
    ax.scatter([paths["p0"][0], paths["via"][0], paths["p2"][0]], [paths["p0"][1], paths["via"][1], paths["p2"][1]], c=["#2ca02c", "#1f77b4", "#9467bd"], s=45, zorder=5)
    ax.text(paths["p0"][0], paths["p0"][1], " P0", fontsize=9)
    ax.text(paths["via"][0], paths["via"][1], " via", fontsize=9)
    ax.text(paths["p2"][0], paths["p2"][1], " P2", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_title("Tabletop planar TCP path preview")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_clearance(path: Path, ref: dict[str, Any], cand: dict[str, Any], obstacle_threshold: float, table_threshold: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.0), sharex=True, dpi=180)
    for label, payload, color in (("reference", ref, "#777777"), ("candidate", cand, "#1f77b4")):
        idx = [row["index"] for row in payload["rows"]]
        axes[0].plot(idx, [row["obstacle_clearance_m"] for row in payload["rows"]], label=label, color=color)
        axes[1].plot(idx, [row["table_clearance_m"] for row in payload["rows"]], label=label, color=color)
    axes[0].axhline(obstacle_threshold, color="#d62728", linestyle="--", linewidth=1.2)
    axes[1].axhline(table_threshold, color="#d62728", linestyle="--", linewidth=1.2)
    axes[0].set_ylabel("obstacle clearance / m")
    axes[1].set_ylabel("table clearance / m")
    axes[1].set_xlabel("path sample")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pose_sequence(
    path: Path,
    obstacle_points: np.ndarray,
    base_by_link: dict[str, np.ndarray],
    tcp_path: np.ndarray,
    base_tcp: np.ndarray,
    table_z: float,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(20260652)
    obs = sample_points(obstacle_points, 2600, rng)
    sample_indices = np.linspace(0, len(tcp_path) - 1, 6).round().astype(int)
    fig = plt.figure(figsize=(10.5, 7.4), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2], s=5, c="#d62728", alpha=0.40, label="observed obstacle")
    plotted = [obs]
    colors = ["#d9d9d9", "#bcd7ef", "#8bbfe8", "#5aa2d5", "#2b7bba", "#084594"]
    for ci, idx in enumerate(sample_indices):
        delta = tcp_path[idx] - base_tcp
        pts = stack_surface({link: value + delta[None, :] for link, value in base_by_link.items()})
        pts = sample_points(pts, 1700, rng)
        plotted.append(pts)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.8, c=colors[ci], alpha=0.22 + 0.10 * ci, label=f"s={idx}")
    xlim = [min(np.min(a[:, 0]) for a in plotted), max(np.max(a[:, 0]) for a in plotted)]
    ylim = [min(np.min(a[:, 1]) for a in plotted), max(np.max(a[:, 1]) for a in plotted)]
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 2), np.linspace(ylim[0], ylim[1], 2))
    zz = np.full_like(xx, table_z)
    ax.plot_surface(xx, yy, zz, color="#c7c7c7", alpha=0.18, linewidth=0)
    set_equal_axes(ax, plotted)
    ax.set_title(title)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.view_init(elev=24, azim=-52)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, default=DEFAULT_TRIAL)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--density", choices=["coarse", "medium", "dense"], default="coarse")
    parser.add_argument("--tcp-link", default="gripper_base_link")
    parser.add_argument("--posture-source", choices=["trial", "home"], default="trial")
    parser.add_argument("--home-joints-deg", default="0,0,90,0,90,0")
    parser.add_argument("--axis", choices=["x", "y"], default="y")
    parser.add_argument("--direction-sign", type=float, default=1.0, help="+1 or -1 along the selected axis")
    parser.add_argument("--distance-m", type=float, default=0.20)
    parser.add_argument("--y-start", type=float, default=None, help="Absolute base-frame Y coordinate for planar path start.")
    parser.add_argument("--y-goal", type=float, default=None, help="Absolute base-frame Y coordinate for planar path goal.")
    parser.add_argument("--x-start", type=float, default=None, help="Absolute base-frame X coordinate for planar path start.")
    parser.add_argument("--x-goal", type=float, default=None, help="Absolute base-frame X coordinate for planar path goal.")
    parser.add_argument("--clearance-m", type=float, default=0.08)
    parser.add_argument("--detour-extra-m", type=float, default=0.05)
    parser.add_argument("--side", choices=["auto", "positive", "negative"], default="auto")
    parser.add_argument("--samples", type=int, default=121)
    parser.add_argument("--table-z", default="auto", help="auto uses obstacle z 2nd percentile; otherwise a float in base frame")
    parser.add_argument("--table-clearance-threshold-m", type=float, default=0.06)
    parser.add_argument(
        "--clearance-links",
        default=",".join(DEFAULT_CLEARANCE_LINKS),
        help="Comma-separated links used for obstacle/table clearance statistics. Rendering still shows all moving links.",
    )
    args = parser.parse_args()

    trial_dir = args.trial_dir.resolve()
    output = (args.output or (trial_dir / "planar_tabletop_preview")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = (args.config or (trial_dir / "config_used.yaml")).resolve()
    config = _load(config_path)
    surface_model = make_surface_model(config)
    q = parse_home_degrees(args.home_joints_deg) if args.posture_source == "home" else load_trial_posture(trial_dir)
    obstacle_points = np.asarray(np.load(trial_dir / "obstacle_points.npz")["points"], dtype=np.float64)
    table_z = float(np.percentile(obstacle_points[:, 2], 2)) if args.table_z == "auto" else float(args.table_z)
    base_tcp = tcp_position(surface_model, q, args.tcp_link)
    direction = np.array([1.0, 0.0, 0.0]) if args.axis == "x" else np.array([0.0, 1.0, 0.0])
    direction = direction * (1.0 if args.direction_sign >= 0.0 else -1.0)
    p_start = base_tcp.copy()
    p_goal = base_tcp + direction * args.distance_m
    if args.y_start is not None:
        p_start[1] = float(args.y_start)
    if args.y_goal is not None:
        p_goal[1] = float(args.y_goal)
    if args.x_start is not None:
        p_start[0] = float(args.x_start)
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
    clearance_links = parse_links(args.clearance_links)
    base_by_link = moving_surface_by_link(surface_model, q, args.density)
    clearance_by_link = {
        link: pts for link, pts in base_by_link.items() if link in set(clearance_links)
    }
    base_surface = stack_surface(clearance_by_link)
    if len(base_surface) == 0:
        raise RuntimeError(f"none of --clearance-links are available in URDF surface: {clearance_links}")
    ref_clear = clearance_along_path(base_surface, paths["reference"], base_tcp, obstacle_points, table_z)
    cand_clear = clearance_along_path(base_surface, paths["candidate"], base_tcp, obstacle_points, table_z)

    plot_top_view(output / "top_view_tcp_paths.png", obstacle_points, paths, args.clearance_m)
    plot_clearance(
        output / "clearance_curves.png",
        ref_clear,
        cand_clear,
        obstacle_threshold=args.clearance_m,
        table_threshold=args.table_clearance_threshold_m,
    )
    plot_pose_sequence(
        output / "reference_pose_sequence.png",
        obstacle_points,
        base_by_link,
        paths["reference"],
        base_tcp,
        table_z,
        "Straight tabletop reference: fixed posture, constant TCP height",
    )
    plot_pose_sequence(
        output / "candidate_pose_sequence.png",
        obstacle_points,
        base_by_link,
        paths["candidate"],
        base_tcp,
        table_z,
        "Planar detour candidate: fixed posture, constant TCP height",
    )

    fields = ["index", "path", "tcp_x", "tcp_y", "tcp_z", "obstacle_clearance_m", "table_clearance_m"]
    rows = []
    for name, payload in (("reference", ref_clear), ("candidate", cand_clear)):
        for row in payload["rows"]:
            rows.append({"path": name, **row})
    write_csv(output / "planar_path_samples.csv", rows, fields)

    accepted = bool(
        cand_clear["min_obstacle_clearance_m"] >= args.clearance_m
        and cand_clear["min_table_clearance_m"] >= args.table_clearance_threshold_m
    )
    summary = {
        "robot_commanded": False,
        "preview_type": "fixed-posture tabletop planar translation",
        "accepted_for_next_design_step": accepted,
        "trial_dir": str(trial_dir),
        "config": str(config_path),
        "posture_source": args.posture_source,
        "q_posture": q.tolist(),
        "q_posture_deg": np.rad2deg(q).tolist(),
        "tcp_link": args.tcp_link,
        "tcp_base_posture": base_tcp.tolist(),
        "axis": args.axis,
        "direction_sign": 1.0 if args.direction_sign >= 0.0 else -1.0,
        "distance_m": float(np.linalg.norm(paths["p2"][:2] - paths["p0"][:2])),
        "table_z_m": table_z,
        "clearance_threshold_m": args.clearance_m,
        "table_clearance_threshold_m": args.table_clearance_threshold_m,
        "clearance_links": list(clearance_links),
        "reference_min_obstacle_clearance_m": ref_clear["min_obstacle_clearance_m"],
        "reference_min_table_clearance_m": ref_clear["min_table_clearance_m"],
        "candidate_min_obstacle_clearance_m": cand_clear["min_obstacle_clearance_m"],
        "candidate_min_table_clearance_m": cand_clear["min_table_clearance_m"],
        "candidate_waypoints": {
            "P0": paths["p0"].tolist(),
            "P1_via": paths["via"].tolist(),
            "P2": paths["p2"].tolist(),
        },
        "note": (
            "This is a geometric tabletop preview with fixed robot posture translation. "
            "It confirms planar path intent, but a real execution still needs a guarded Cartesian executor."
        ),
        "figures": [
            "top_view_tcp_paths.png",
            "clearance_curves.png",
            "reference_pose_sequence.png",
            "candidate_pose_sequence.png",
        ],
    }
    write_json(output / "preview_summary.json", summary)
    print(json.dumps({"output_dir": str(output), **summary}, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
