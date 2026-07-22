"""Generate revised Chapter 6.3 Figure 5 from static benchmark outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def row_feasible(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    return bool(row.get("dense_feasible", row.get("verification", {}).get("accepted")))


def select_representative_instance(input_path: Path, metrics: dict[str, Any]) -> str:
    for scenario in ["B", "C", "A"]:
        instances = metrics.get("scenarios", {}).get(scenario, {}).get("instances", [])
        for item in instances:
            trial = json.loads((input_path / item["trial_path"]).read_text(encoding="utf-8"))
            if row_feasible(trial.get("critical_point_nubs")) and row_feasible(trial.get("ccro_nubs")):
                return item["id"]
    for scenario in ["B", "C", "A"]:
        for item in metrics.get("scenarios", {}).get(scenario, {}).get("instances", []):
            return item["id"]
    raise ValueError("no instances available for figure selection")


def jerk_norm_curve(samples: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(samples["times"], dtype=float)
    jerk = np.asarray(samples["jerk"], dtype=float)
    return times, np.linalg.norm(jerk, axis=1)


def load_surface_model(config: dict[str, Any]):
    from planning.robot_surface_model import RobotSurfaceModel

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


def robot_workspace_snapshot(surface_model: Any, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    by_link = surface_model.surface_by_link(q, density="coarse")
    chunks = [points for points in by_link.values() if len(points)]
    points = np.vstack(chunks) if chunks else np.empty((0, 3), dtype=float)
    centers = np.vstack([points.mean(axis=0) for points in by_link.values() if len(points)])
    return points, centers


def nearest_workspace_pair(surface_model: Any, samples: dict[str, Any], obstacle: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(obstacle) == 0:
        return None
    from scipy.spatial import cKDTree

    tree = cKDTree(obstacle)
    q_values = np.asarray(samples["q"], dtype=float)
    if len(q_values) > 61:
        keep = np.linspace(0, len(q_values) - 1, 61).round().astype(int)
        q_values = q_values[keep]
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for q in q_values:
        robot_points = surface_model.surface(q, density="medium")
        distances, indices = tree.query(robot_points, k=1)
        local = int(np.argmin(distances))
        candidate = (float(distances[local]), robot_points[local].copy(), obstacle[int(indices[local])].copy())
        if best is None or candidate[0] < best[0]:
            best = candidate
    return None if best is None else (best[1], best[2])


def plot(input_dir: str | Path, output_dir: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((input_path / "metrics.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((input_path / "source_ccro_stage2.yaml").read_text(encoding="utf-8"))
    surface_model = load_surface_model(config)
    instance_id = select_representative_instance(input_path, metrics)
    scenario, _ = instance_id.split("_", 1)
    trial = json.loads((input_path / "trials" / f"{instance_id}.json").read_text(encoding="utf-8"))
    frozen = json.loads((input_path / "frozen_instances" / f"{instance_id}.json").read_text(encoding="utf-8"))
    obstacle = np.asarray(frozen["gt_dense_points"], dtype=float)

    fig = plt.figure(figsize=(11.5, 4.5))
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    if len(obstacle):
        keep = np.linspace(0, len(obstacle) - 1, min(len(obstacle), 1200)).round().astype(int)
        shown = obstacle[keep]
        ax0.scatter(shown[:, 0], shown[:, 1], shown[:, 2], s=2, c="black", alpha=0.35, label="GT obstacle")
    colors = {
        "critical_point_nubs": "tab:orange",
        "ccro_nubs": "tab:blue",
    }
    for method, color in colors.items():
        samples = trial.get(method, {}).get("plot_samples")
        if not samples:
            continue
        q_values = np.asarray(samples["q"], dtype=float)
        marker_idx = np.linspace(0, len(q_values) - 1, min(5, len(q_values))).round().astype(int)
        ee_path = []
        for draw_index, index in enumerate(marker_idx):
            robot_points, centers = robot_workspace_snapshot(surface_model, q_values[index])
            if len(robot_points):
                keep = np.linspace(0, len(robot_points) - 1, min(len(robot_points), 260)).round().astype(int)
                shown_robot = robot_points[keep]
                ax0.scatter(
                    shown_robot[:, 0],
                    shown_robot[:, 1],
                    shown_robot[:, 2],
                    s=1.6,
                    color=color,
                    alpha=0.12 + 0.08 * draw_index,
                )
            if len(centers):
                ax0.plot(centers[:, 0], centers[:, 1], centers[:, 2], color=color, alpha=0.55, linewidth=1.0)
                ee_path.append(centers[-1])
        if ee_path:
            ee = np.vstack(ee_path)
            ax0.plot(ee[:, 0], ee[:, 1], ee[:, 2], color=color, linewidth=2.0, label=method)
    pair = nearest_workspace_pair(surface_model, trial.get("ccro_nubs", {}).get("plot_samples", {}), obstacle)
    if pair is not None:
        robot_point, obstacle_point = pair
        segment = np.vstack([robot_point, obstacle_point])
        ax0.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="crimson", linewidth=2.0, label="nearest CCRO pair")
        ax0.scatter([robot_point[0]], [robot_point[1]], [robot_point[2]], color="crimson", s=26)
    ax0.set_title(f"{scenario} representative workspace")
    ax0.set_xlabel("x / m")
    ax0.set_ylabel("y / m")
    ax0.set_zlabel("z / m")
    ax0.legend(fontsize=8)

    ax1 = fig.add_subplot(1, 2, 2)
    for method in ["rrt_connect_smooth", "minco_risk", "critical_point_nubs", "ccro_nubs"]:
        samples = trial.get(method, {}).get("plot_samples")
        if not samples:
            continue
        times, values = jerk_norm_curve(samples)
        ax1.plot(times, values, label=method)
    ax1.set_xlabel("time / s")
    ax1.set_ylabel(r"$\|q^{(3)}(t)\|_2$")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    target = output_path / "figure_5_static_trajectory_and_jerk.png"
    fig.savefig(target, dpi=220)
    plt.close(fig)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/results/6_3")
    parser.add_argument("--output", default="data/results/6_3/paper")
    args = parser.parse_args()
    target = plot(args.input, args.output)
    print(f"[6_3_plot] saved {target}")


if __name__ == "__main__":
    main()
