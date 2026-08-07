#!/usr/bin/env python3
"""Plot top-view layout for a 6.5.2 joint-space CCRO-NUBS plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_ccro_stage2 import _load  # noqa: E402
from planning.nubs_trajectory import NUBSTrajectory6D  # noqa: E402
from plan_652_static_ccro_nubs_from_trial import DEFAULT_TRIAL  # noqa: E402
from run_652_static_avoidance import make_surface_model  # noqa: E402


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


def joint_dict(surface_model, q: np.ndarray) -> dict[str, float]:
    return {name: float(q[i]) for i, name in enumerate(surface_model.joint_names)}


def tcp_xyz(surface_model, q: np.ndarray, tcp_link: str) -> np.ndarray:
    fk = surface_model.urdf.link_transforms(joint_dict(surface_model, q))
    return np.asarray(fk[tcp_link][:3, 3], dtype=np.float64)


def load_trajectory(npz_path: Path, key: str) -> NUBSTrajectory6D:
    data = np.load(npz_path)
    head = NUBSTrajectory6D.make_boundary_state(data["q_start"])
    tail = NUBSTrajectory6D.make_boundary_state(data["q_goal"])
    return NUBSTrajectory6D().generate(data[key], head, tail, data["durations"])


def find_trial_dir(plan_dir: Path) -> Path:
    for path in [plan_dir.parent, *plan_dir.parents]:
        if (path / "config_used.yaml").exists():
            return path
    raise FileNotFoundError(f"could not find config_used.yaml above {plan_dir}")


def sample_surface(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= n:
        return points
    return points[rng.choice(len(points), n, replace=False)]


def nearest_pair(surface_model, trajectory, obstacle_points, density: str, dt: float):
    tree = cKDTree(obstacle_points)
    times = np.linspace(0.0, trajectory.total_duration, max(2, int(np.ceil(trajectory.total_duration / dt)) + 1))
    best = {
        "distance": float("inf"),
        "time": None,
        "link": None,
        "robot_point": None,
        "obstacle_point": None,
        "q": None,
    }
    for t in times:
        q = trajectory.evaluate(float(t))
        by_link = surface_model.surface_by_link(q, density=density, links=set(WATCH_LINKS))
        for link, points in by_link.items():
            dists, idx = tree.query(points, k=1)
            j = int(np.argmin(dists))
            d = float(dists[j])
            if d < best["distance"]:
                best = {
                    "distance": d,
                    "time": float(t),
                    "link": link,
                    "robot_point": points[j].copy(),
                    "obstacle_point": obstacle_points[int(idx[j])].copy(),
                    "q": q.copy(),
                }
    return best


def plot_top(plan_dir: Path, output: Path, args: argparse.Namespace) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trial_dir = find_trial_dir(plan_dir)
    config = _load(trial_dir / "config_used.yaml")
    surface_model = make_surface_model(config)
    data_path = plan_dir / "ccro_nubs_trajectories.npz"
    reference = load_trajectory(data_path, "reference_inner")
    candidate = load_trajectory(data_path, "candidate_inner")
    obstacle = np.asarray(np.load(data_path)["obstacle_points"], dtype=np.float64)
    times_ref = np.linspace(0.0, reference.total_duration, args.samples)
    times_cand = np.linspace(0.0, candidate.total_duration, args.samples)
    ref_tcp = np.vstack([tcp_xyz(surface_model, reference.evaluate(float(t)), args.tcp_link) for t in times_ref])
    cand_tcp = np.vstack([tcp_xyz(surface_model, candidate.evaluate(float(t)), args.tcp_link) for t in times_cand])
    best = nearest_pair(surface_model, candidate, obstacle, args.density, args.nearest_dt)

    rng = np.random.default_rng(20260652)
    obs_plot = sample_surface(obstacle, args.max_obstacle_points, rng)
    fig, ax = plt.subplots(figsize=(8.2, 7.2), dpi=200)
    ax.scatter(obs_plot[:, 0], obs_plot[:, 1], s=5, c="#d62728", alpha=0.42, label="observed obstacle points")
    ax.plot(ref_tcp[:, 0], ref_tcp[:, 1], "--", color="#7f7f7f", linewidth=2.0, label="reference TCP path")
    ax.plot(cand_tcp[:, 0], cand_tcp[:, 1], "-", color="#1f77b4", linewidth=2.4, label="CCRO-NUBS TCP path")
    ax.scatter(ref_tcp[[0, -1], 0], ref_tcp[[0, -1], 1], c=["#2ca02c", "#9467bd"], s=45, zorder=5)
    ax.text(ref_tcp[0, 0], ref_tcp[0, 1], " start", fontsize=9)
    ax.text(ref_tcp[-1, 0], ref_tcp[-1, 1], " goal", fontsize=9)

    robot_point = best["robot_point"]
    obstacle_point = best["obstacle_point"]
    if robot_point is not None and obstacle_point is not None:
        ax.plot(
            [robot_point[0], obstacle_point[0]],
            [robot_point[1], obstacle_point[1]],
            color="#ff7f0e",
            linewidth=2.2,
            label=f"nearest pair ({best['link']}, {best['distance']:.3f} m)",
        )
        ax.scatter([robot_point[0]], [robot_point[1]], c="#ff7f0e", marker="x", s=80, zorder=6)
        ax.scatter([obstacle_point[0]], [obstacle_point[1]], c="#ff7f0e", marker="o", s=40, zorder=6)

    by_link = surface_model.surface_by_link(best["q"], density=args.density, links=set(WATCH_LINKS)) if best["q"] is not None else {}
    for link, pts in by_link.items():
        pts = sample_surface(pts, 450, rng)
        alpha = 0.28 if link == best["link"] else 0.10
        size = 4 if link == best["link"] else 2
        ax.scatter(pts[:, 0], pts[:, 1], s=size, alpha=alpha, label=f"{link} @ min" if link == best["link"] else None)

    ax.axhline(0.0, color="#dddddd", linewidth=0.8)
    ax.axvline(0.0, color="#dddddd", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_title("Top view: observed obstacle and joint-space CCRO-NUBS path")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    summary = {
        "figure": str(output),
        "candidate_min_pair_distance_m": best["distance"],
        "candidate_min_pair_time_s": best["time"],
        "candidate_min_pair_link": best["link"],
        "candidate_min_robot_point": None if best["robot_point"] is None else best["robot_point"].tolist(),
        "candidate_min_obstacle_point": None if best["obstacle_point"] is None else best["obstacle_point"].tolist(),
        "reference_tcp_start": ref_tcp[0].tolist(),
        "reference_tcp_goal": ref_tcp[-1].tolist(),
        "candidate_tcp_start": cand_tcp[0].tolist(),
        "candidate_tcp_goal": cand_tcp[-1].tolist(),
        "obstacle_xy_min": np.min(obstacle[:, :2], axis=0).tolist(),
        "obstacle_xy_max": np.max(obstacle[:, :2], axis=0).tolist(),
    }
    (output.parent / "top_view_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_TRIAL / "ccro_nubs_jointspace_plan")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tcp-link", default="gripper_base_link")
    parser.add_argument("--density", choices=["coarse", "medium", "dense"], default="medium")
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--nearest-dt", type=float, default=0.025)
    parser.add_argument("--max-obstacle-points", type=int, default=2400)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output or (args.plan_dir / "figures" / "top_view_obstacle_ccro_nubs.png")
    summary = plot_top(args.plan_dir.resolve(), output.resolve(), args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
