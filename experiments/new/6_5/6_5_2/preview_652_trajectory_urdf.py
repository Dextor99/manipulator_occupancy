#!/usr/bin/env python3
"""Preview 6.5.2 reference/candidate trajectories with the AUBO URDF model.

The script is fully offline: it reads a completed 6.5.2 trial directory, loads
the saved trajectories and obstacle point cloud, then renders URDF surface
snapshots and link-center paths.  It never connects to RealSense or AUBO.
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

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_ccro_stage2 import _load  # noqa: E402
from planning.mesh_risk import MeshRiskEvaluator, StaticObstacleField  # noqa: E402
from planning.nubs_trajectory import NUBSTrajectory6D  # noqa: E402
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

WATCH_LINKS = (
    "upperArm_Link",
    "foreArm_Link",
    "wrist1_Link",
    "wrist2_Link",
    "wrist3_Link",
    "gripper_base_link",
    "left_link",
    "right_link",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


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


def load_trajectory_from_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"empty trajectory csv: {path}")
    times = np.asarray([float(row["t_s"]) for row in rows], dtype=np.float64)
    q = np.asarray([[float(row[f"q{j+1}_rad"]) for j in range(6)] for row in rows], dtype=np.float64)
    return times, q


def load_nubs_from_npz(path: Path, name: str) -> NUBSTrajectory6D:
    data = np.load(path)
    inner_key = "reference_inner" if name == "reference" else "candidate_inner"
    return NUBSTrajectory6D().generate(
        np.asarray(data[inner_key], dtype=np.float64),
        NUBSTrajectory6D.make_boundary_state(np.asarray(data["q_start"], dtype=np.float64)),
        NUBSTrajectory6D.make_boundary_state(np.asarray(data["q_goal"], dtype=np.float64)),
        np.asarray(data["durations"], dtype=np.float64),
    )


def min_clearance_sample(
    trajectory: NUBSTrajectory6D,
    evaluator: MeshRiskEvaluator,
    obstacle: StaticObstacleField,
    *,
    dt: float,
    density: str,
) -> dict[str, Any]:
    times = np.linspace(0.0, trajectory.total_duration, max(2, int(math.ceil(trajectory.total_duration / dt)) + 1))
    distances = []
    links = []
    robot_points = []
    obstacle_points = []
    for t in times:
        risk = evaluator.configuration(trajectory.evaluate(float(t)), obstacle, density=density, with_gradient=False)
        distances.append(float(risk.min_distance))
        links.append(risk.nearest_link)
        robot_points.append(None if risk.robot_point is None else risk.robot_point.copy())
        obstacle_points.append(None if risk.obstacle_point is None else risk.obstacle_point.copy())
    idx = int(np.argmin(distances))
    return {
        "time_s": float(times[idx]),
        "q": trajectory.evaluate(float(times[idx])),
        "distance_m": float(distances[idx]),
        "nearest_link": links[idx],
        "robot_point": robot_points[idx],
        "obstacle_point": obstacle_points[idx],
        "sample_times_s": times,
        "sample_distances_m": np.asarray(distances),
    }


def sample_points(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= n:
        return points
    return points[rng.choice(len(points), n, replace=False)]


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


def link_centers(surface_model: RobotSurfaceModel, q_values: np.ndarray, density: str) -> dict[str, np.ndarray]:
    centers: dict[str, list[np.ndarray]] = {link: [] for link in WATCH_LINKS}
    for q in q_values:
        by_link = surface_model.surface_by_link(q, density=density, links=set(WATCH_LINKS))
        for link in WATCH_LINKS:
            pts = by_link.get(link)
            if pts is None or len(pts) == 0:
                centers[link].append(np.full(3, np.nan))
            else:
                centers[link].append(np.mean(pts, axis=0))
    return {link: np.asarray(values, dtype=np.float64) for link, values in centers.items()}


def plot_link_paths(
    path: Path,
    obstacle_points: np.ndarray,
    reference_centers: dict[str, np.ndarray],
    candidate_centers: dict[str, np.ndarray],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(20260652)
    obs = sample_points(obstacle_points, 2500, rng)
    fig = plt.figure(figsize=(9.5, 7.0), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    if len(obs):
        ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2], s=5, c="#d62728", alpha=0.45, label="observed obstacle")
    palette = {
        "foreArm_Link": "#ff7f0e",
        "wrist2_Link": "#2ca02c",
        "wrist3_Link": "#1f77b4",
        "gripper_base_link": "#9467bd",
        "right_link": "#8c564b",
    }
    plotted = []
    for link in ("foreArm_Link", "wrist2_Link", "wrist3_Link", "gripper_base_link", "right_link"):
        ref = reference_centers.get(link)
        cand = candidate_centers.get(link)
        color = palette[link]
        if ref is not None and np.isfinite(ref).all():
            ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], linestyle="--", color=color, linewidth=1.2, alpha=0.55, label=f"ref {link}")
            plotted.append(ref)
        if cand is not None and np.isfinite(cand).all():
            ax.plot(cand[:, 0], cand[:, 1], cand[:, 2], linestyle="-", color=color, linewidth=2.0, label=f"cand {link}")
            plotted.append(cand)
    set_equal_axes(ax, [obs, *plotted])
    ax.set_title("URDF link-center paths: dashed reference, solid CCRO-NUBS candidate")
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.view_init(elev=24, azim=-52)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pose(
    path: Path,
    surface_model: RobotSurfaceModel,
    q: np.ndarray,
    obstacle_points: np.ndarray,
    title: str,
    *,
    density: str,
    nearest_robot: np.ndarray | None,
    nearest_obstacle: np.ndarray | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(20260653)
    by_link = surface_model.surface_by_link(q, density=density)
    obs = sample_points(obstacle_points, 2800, rng)
    fig = plt.figure(figsize=(9.0, 7.0), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    if len(obs):
        ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2], s=5.0, c="#d62728", alpha=0.45, label="observed obstacle")
    plotted = [obs]
    for link, pts in by_link.items():
        pts = sample_points(pts, 900 if link in WATCH_LINKS else 350, rng)
        plotted.append(pts)
        color = "#4c78a8" if link in WATCH_LINKS else "#b7b7b7"
        alpha = 0.72 if link in WATCH_LINKS else 0.18
        size = 2.6 if link in WATCH_LINKS else 1.0
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size, c=color, alpha=alpha)
    if nearest_robot is not None and nearest_obstacle is not None:
        pts = np.vstack([nearest_robot, nearest_obstacle])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#111111", linewidth=2.2, label="minimum clearance")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=45, c="#111111")
        plotted.append(pts)
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


def plot_pose_sequence(
    path: Path,
    surface_model: RobotSurfaceModel,
    trajectory: NUBSTrajectory6D,
    obstacle_points: np.ndarray,
    *,
    density: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(20260654)
    times = np.linspace(0.0, trajectory.total_duration, 6)
    obs = sample_points(obstacle_points, 2200, rng)
    fig = plt.figure(figsize=(11.0, 7.4), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    if len(obs):
        ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2], s=4.5, c="#d62728", alpha=0.38, label="observed obstacle")
    plotted = [obs]
    colors = ["#cccccc", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]
    for i, t in enumerate(times):
        by_link = surface_model.surface_by_link(trajectory.evaluate(float(t)), density=density, links=set(WATCH_LINKS))
        pts = np.vstack([arr for arr in by_link.values() if len(arr)])
        pts = sample_points(pts, 1400, rng)
        plotted.append(pts)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.8, c=colors[i], alpha=0.22 + 0.10 * i, label=f"t={t:.1f}s")
    set_equal_axes(ax, plotted)
    ax.set_title("CCRO-NUBS candidate URDF pose sequence")
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
    parser.add_argument("--config", type=Path, default=None, help="Defaults to trial config_used.yaml.")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <trial>/urdf_preview.")
    parser.add_argument("--density", choices=["coarse", "medium", "dense"], default="coarse")
    parser.add_argument("--audit-dt", type=float, default=0.04)
    args = parser.parse_args()

    trial_dir = args.trial_dir.resolve()
    output = (args.output or (trial_dir / "urdf_preview")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = (args.config or (trial_dir / "config_used.yaml")).resolve()
    config = _load(config_path)
    surface_model = make_surface_model(config)
    risk_cfg = config["risk"]
    evaluator = MeshRiskEvaluator(
        surface_model,
        d_safe=risk_cfg["d_safe"],
        d_activate=risk_cfg["d_activate"],
        fd_epsilon_q=risk_cfg["fd_epsilon_q"],
        density=risk_cfg["optimizer_density"],
    )

    obstacle_points = np.asarray(np.load(trial_dir / "obstacle_points.npz")["points"], dtype=np.float64)
    obstacle = StaticObstacleField.from_points(obstacle_points)
    reference = load_nubs_from_npz(trial_dir / "trajectories.npz", "reference")
    candidate = load_nubs_from_npz(trial_dir / "trajectories.npz", "candidate")
    _, ref_q = load_trajectory_from_csv(trial_dir / "reference_trajectory.csv")
    _, cand_q = load_trajectory_from_csv(trial_dir / "optimized_trajectory.csv")

    ref_min = min_clearance_sample(reference, evaluator, obstacle, dt=args.audit_dt, density=args.density)
    cand_min = min_clearance_sample(candidate, evaluator, obstacle, dt=args.audit_dt, density=args.density)

    reference_centers = link_centers(surface_model, ref_q, args.density)
    candidate_centers = link_centers(surface_model, cand_q, args.density)
    plot_link_paths(output / "link_center_paths.png", obstacle_points, reference_centers, candidate_centers)
    plot_pose(
        output / "reference_min_clearance_pose.png",
        surface_model,
        ref_min["q"],
        obstacle_points,
        f"Reference min clearance: {ref_min['distance_m']:.4f} m, link={ref_min['nearest_link']}, t={ref_min['time_s']:.2f}s",
        density=args.density,
        nearest_robot=ref_min["robot_point"],
        nearest_obstacle=ref_min["obstacle_point"],
    )
    plot_pose(
        output / "candidate_min_clearance_pose.png",
        surface_model,
        cand_min["q"],
        obstacle_points,
        f"Candidate min clearance: {cand_min['distance_m']:.4f} m, link={cand_min['nearest_link']}, t={cand_min['time_s']:.2f}s",
        density=args.density,
        nearest_robot=cand_min["robot_point"],
        nearest_obstacle=cand_min["obstacle_point"],
    )
    plot_pose_sequence(output / "candidate_pose_sequence.png", surface_model, candidate, obstacle_points, density=args.density)

    summary = {
        "created_at": str(np.datetime64("now")),
        "robot_commanded": False,
        "trial_dir": str(trial_dir),
        "config": str(config_path),
        "density": args.density,
        "reference_min_clearance": {
            "time_s": ref_min["time_s"],
            "distance_m": ref_min["distance_m"],
            "nearest_link": ref_min["nearest_link"],
        },
        "candidate_min_clearance": {
            "time_s": cand_min["time_s"],
            "distance_m": cand_min["distance_m"],
            "nearest_link": cand_min["nearest_link"],
        },
        "figures": [
            "link_center_paths.png",
            "reference_min_clearance_pose.png",
            "candidate_min_clearance_pose.png",
            "candidate_pose_sequence.png",
        ],
    }
    write_json(output / "preview_summary.json", summary)
    print(json.dumps({"robot_commanded": False, "output_dir": str(output), **summary}, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
