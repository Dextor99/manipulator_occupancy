"""Generate qualitative figures for Chapter 4.2 decoupling results."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.decoupling_eval import DecouplingEvaluator, METHODS, parse_omega, parse_omegas, points_in_omega
from experiments.recorder import load_sequence
from test_clustering_filtering import FastClusteringFilter


def _load_one_frame(record_dir: str, frame_index: int) -> dict:
    """Stream to the requested frame without loading the whole recording."""
    selected = None
    for idx, frame in enumerate(load_sequence(record_dir)):
        selected = frame
        if idx >= frame_index:
            return frame
    if selected is None:
        raise SystemExit(f"no frames found in {record_dir}")
    print(f"[exp_42_visualize] requested frame {frame_index}, using last available frame")
    return selected


def _axis_limits(points_list: list[np.ndarray]):
    pts = np.vstack([p for p in points_list if len(p)]) if any(len(p) for p in points_list) else np.zeros((1, 3))
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    pad = np.maximum((hi - lo) * 0.08, 0.05)
    return lo - pad, hi + pad


def _scatter(ax, points: np.ndarray, title: str, color: str = "tab:green", max_points: int = 8000, size: float = 1.0):
    if len(points) == 0:
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        return
    if len(points) > max_points:
        rng = np.random.default_rng(4)
        points = points[rng.choice(len(points), max_points, replace=False)]
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=size, c=color, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Figure 4-2 style point-cloud comparison.")
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--output", default="data/results/ch4_2/fig_4_2.png")
    parser.add_argument("--config", default="config")
    parser.add_argument("--urdf", default="urdf/aubo_i16_gripper.urdf")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--delta-r", type=float, default=0.05)
    parser.add_argument("--delta-eval", type=float, default=0.10)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument(
        "--max-raw-points",
        type=int,
        default=100000,
        help="Maximum raw points used from the selected frame before transform/crop; <=0 disables thinning.",
    )
    parser.add_argument("--mesh-samples", type=int, default=50000)
    parser.add_argument("--remove-planes", action="store_true")
    parser.add_argument("--omega", default=None, help="Optional AABB to highlight obstacle points: x0,x1,y0,y1,z0,z1")
    parser.add_argument(
        "--omegas",
        default=None,
        help='Optional multiple AABBs to highlight, separated by semicolons: "x0,x1,y0,y1,z0,z1;x0,x1,y0,y1,z0,z1"',
    )
    args = parser.parse_args()

    frame = _load_one_frame(args.record_dir, max(args.frame, 0))

    evaluator = DecouplingEvaluator(
        config_dir=args.config,
        urdf_path=args.urdf,
        delta_r=args.delta_r,
        delta_eval=args.delta_eval,
        voxel_size=args.voxel_size,
        max_raw_points=None if args.max_raw_points <= 0 else args.max_raw_points,
        mesh_samples=args.mesh_samples,
        remove_planes=args.remove_planes,
    )

    outputs = {}
    for method in METHODS:
        outputs[method] = evaluator.build_method(method).filter(frame["points_cam"], frame["joint_dict"])
    omegas = parse_omegas(args.omegas) if args.omegas else []
    omega = parse_omega(args.omega) if args.omega else None
    if not omegas and omega is not None:
        omegas = [omega]

    display_points = {}
    obstacle_points = {}
    for method, output in outputs.items():
        source = output.common_points if method == "workspace" else output.external_points
        cluster_result = FastClusteringFilter(source, outputs["ours"].robot_points, **evaluator.cluster_kwargs)
        pts = np.vstack([cluster.points for cluster in cluster_result.clusters]) if cluster_result.clusters else np.empty((0, 3))
        display_points[method] = pts
        if omegas and len(source):
            mask = np.zeros(len(source), dtype=bool)
            for item in omegas:
                mask |= points_in_omega(source, item)
            obstacle_points[method] = source[mask]
        else:
            obstacle_points[method] = np.empty((0, 3))

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib is required for visualization: {exc}") from exc

    fig = plt.figure(figsize=(13, 4.5))
    titles = [
        ("workspace", "Workspace filtering"),
        ("ksi_like", "KSI-like baseline"),
        ("ours", "Ours"),
    ]
    lo, hi = _axis_limits([display_points[name] for name, _ in titles] + [outputs["ours"].robot_points])

    for idx, (name, title) in enumerate(titles, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        pts = display_points[name]
        color = "tab:gray" if name == "workspace" else "tab:green"
        _scatter(ax, pts, title, color=color)
        obs = obstacle_points[name]
        if len(obs):
            _scatter(ax, obs, title, color="tab:orange", max_points=3000, size=3.0)
        robot = outputs["ours"].robot_points
        if len(robot):
            r = robot
            if len(r) > 3000:
                rng = np.random.default_rng(8)
                r = r[rng.choice(len(r), 3000, replace=False)]
            ax.scatter(r[:, 0], r[:, 1], r[:, 2], s=0.5, c="tab:red", alpha=0.25)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.view_init(elev=22, azim=-55)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    print(f"[exp_42_visualize] saved {out}")


if __name__ == "__main__":
    main()
