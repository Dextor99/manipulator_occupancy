"""Critical-point APF risk baseline for E1 whole-body risk evaluation.

The baseline uses the same synthetic scenes as ``exp_63_sim_risk_distance`` but
represents the robot with a sparse set of critical points per link.  The
obstacle input includes the same current and predicted future spheres used by
Ours-CCRO, so the comparison isolates sparse critical points versus dense mesh
surface risk as much as possible.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_63_sim_risk_distance import (  # noqa: E402
    SCENARIOS,
    choose_surface_point,
    count_values,
    distance_to_links,
    load_defaults,
    outward_direction,
    sample_sphere,
)
from planning.mesh_risk import StaticObstacleField  # noqa: E402
from planning.robot_surface_model import RobotSurfaceModel  # noqa: E402


METHODS = ("critical_point_apf", "ours_ccro")
METHOD_NAMES = {
    "critical_point_apf": "Critical-point APF",
    "ours_ccro": "Ours-CCRO Mesh",
}


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    """Pick a deterministic sparse subset spread across one link surface."""
    if len(points) <= count:
        return np.arange(len(points), dtype=np.int64)
    centroid = np.mean(points, axis=0)
    first = int(np.argmax(np.linalg.norm(points - centroid[None, :], axis=1)))
    selected = [first]
    min_dist = np.linalg.norm(points - points[first][None, :], axis=1)
    for _ in range(1, count):
        index = int(np.argmax(min_dist))
        selected.append(index)
        min_dist = np.minimum(min_dist, np.linalg.norm(points - points[index][None, :], axis=1))
    return np.asarray(selected, dtype=np.int64)


def critical_points_by_link(
    surface: RobotSurfaceModel,
    q: np.ndarray,
    links: set[str] | None,
    density: str,
    points_per_link: int,
) -> dict[str, np.ndarray]:
    sparse: dict[str, np.ndarray] = {}
    for link, points in surface.surface_by_link(q, density=density, links=links).items():
        if len(points) == 0:
            continue
        sparse[link] = points[farthest_point_indices(points, points_per_link)]
    return sparse


def evaluate_critical_point_apf(
    critical_points: dict[str, np.ndarray],
    obstacle_points: np.ndarray,
    *,
    d_safe: float,
    d_activate: float,
) -> dict[str, Any]:
    if len(obstacle_points) == 0 or not critical_points:
        return {
            "distance": math.inf,
            "nearest_link": None,
            "risk_detected": False,
            "apf_cost": 0.0,
            "active_points": 0,
        }
    tree = cKDTree(np.asarray(obstacle_points, dtype=np.float64))
    best_distance = math.inf
    nearest_link = None
    active_points = 0
    costs = []
    for link, points in critical_points.items():
        distances, _ = tree.query(points, k=1)
        local_index = int(np.argmin(distances))
        local_distance = float(distances[local_index])
        if local_distance < best_distance:
            best_distance = local_distance
            nearest_link = link
        active = distances < d_activate
        active_points += int(np.count_nonzero(active))
        if np.any(active):
            clipped = np.maximum(distances[active], 1.0e-6)
            cost = np.square((1.0 / clipped) - (1.0 / d_activate))
            costs.extend(cost.tolist())
    return {
        "distance": float(best_distance),
        "nearest_link": nearest_link,
        "risk_detected": bool(best_distance < d_safe),
        "apf_cost": float(np.mean(costs)) if costs else 0.0,
        "active_points": active_points,
    }


def make_trial(args: argparse.Namespace, surface: RobotSurfaceModel, scene: str, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    q_base = np.asarray(args.q_base, dtype=float)
    q_noise = rng.normal(scale=np.asarray(args.q_noise, dtype=float), size=6)
    q = np.clip(q_base + q_noise, -2.7, 2.7)

    if scene == "ee_near":
        target_point, target_link = choose_surface_point(rng, surface, q, args.ee_links, "dense")
        direction = outward_direction(rng, surface, q, target_point)
        clearance = rng.uniform(0.015, 0.055)
        center = target_point + direction * (args.obstacle_radius + clearance)
        velocity = np.zeros(3)
    elif scene == "body_near":
        target_point, target_link = choose_surface_point(rng, surface, q, args.body_links, "dense")
        direction = outward_direction(rng, surface, q, target_point)
        clearance = rng.uniform(0.015, 0.055)
        center = target_point + direction * (args.obstacle_radius + clearance)
        velocity = np.zeros(3)
    elif scene == "dynamic_future":
        target_point, target_link = choose_surface_point(rng, surface, q, args.body_links, "dense")
        direction = outward_direction(rng, surface, q, target_point)
        start_clearance = rng.uniform(0.16, 0.24)
        center = target_point + direction * (args.obstacle_radius + start_clearance)
        speed = rng.uniform(args.dynamic_speed_min, args.dynamic_speed_max)
        velocity = -direction * speed
    else:
        raise ValueError(f"unknown scene: {scene}")

    current_obs = sample_sphere(rng, center, args.obstacle_radius, args.obstacle_points)
    future_chunks = [current_obs]
    future_distances = []
    future_times = np.arange(args.prediction_step, args.prediction_horizon + 1.0e-9, args.prediction_step)
    for tau in future_times:
        shifted_center = center + velocity * tau
        shifted = sample_sphere(
            rng,
            shifted_center,
            args.obstacle_radius + args.risk_margin,
            args.obstacle_points,
        )
        future_chunks.append(shifted)
        d_tau, _ = distance_to_links(surface, q, shifted, links=None, density="dense")
        future_distances.append(d_tau)
    future_obs = np.vstack(future_chunks)

    d_ee, _ = distance_to_links(surface, q, current_obs, set(args.ee_links), "dense")
    d_body_current, link_body_current = distance_to_links(surface, q, current_obs, None, "dense")
    d_future = min([d_body_current, *future_distances]) if future_distances else d_body_current
    true_risk = d_future < args.d_safe
    body_only = d_body_current < args.d_safe and d_ee > args.d_safe
    future_only = d_body_current > args.d_safe and d_future < args.d_safe

    methods: dict[str, Any] = {}
    t0 = time.perf_counter()
    sparse_links = None if args.links == ["all"] else set(args.links)
    cp0 = time.perf_counter()
    points = critical_points_by_link(
        surface,
        q,
        sparse_links,
        args.critical_point_density,
        args.points_per_link,
    )
    cp_result = evaluate_critical_point_apf(
        points,
        future_obs,
        d_safe=args.d_safe,
        d_activate=args.apf_activate,
    )
    cp_result["time_ms"] = (time.perf_counter() - cp0) * 1000.0
    cp_result["critical_point_count"] = int(sum(len(value) for value in points.values()))
    methods["critical_point_apf"] = cp_result

    mesh0 = time.perf_counter()
    obstacle = StaticObstacleField.from_points(future_obs)
    # Use the existing dense mesh distance style from exp_63 for consistency.
    mesh_distance, mesh_link = distance_to_links(surface, q, obstacle.points, links=None, density="dense")
    methods["ours_ccro"] = {
        "distance": mesh_distance,
        "nearest_link": mesh_link,
        "risk_detected": bool(mesh_distance < args.d_safe),
        "apf_cost": None,
        "active_points": None,
        "time_ms": (time.perf_counter() - mesh0) * 1000.0,
        "critical_point_count": None,
    }

    return {
        "scene": scene,
        "seed": seed,
        "q": q.tolist(),
        "target_link": target_link,
        "center": center.tolist(),
        "velocity": velocity.tolist(),
        "D_ee": float(d_ee),
        "D_body_current": float(d_body_current),
        "D_future": float(d_future),
        "nearest_body_link": link_body_current,
        "true_risk": bool(true_risk),
        "body_only_event": bool(body_only),
        "future_only_event": bool(future_only),
        "methods": methods,
        "trial_time_ms": (time.perf_counter() - t0) * 1000.0,
    }


def aggregate(trials: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        grouped.setdefault(trial["scene"], []).append(trial)

    out: dict[str, Any] = {"trial_count": len(trials), "scenes": {}}
    for scene, rows in grouped.items():
        risk_den = [row for row in rows if row["true_risk"]]
        body_den = [row for row in rows if row["body_only_event"]]
        future_den = [row for row in rows if row["future_only_event"]]
        out["scenes"][scene] = {
            "trials": len(rows),
            "body_only_events": int(sum(row["body_only_event"] for row in rows)),
            "future_only_events": int(sum(row["future_only_event"] for row in rows)),
            "nearest_links": count_values([row["nearest_body_link"] for row in rows]),
            "methods": {},
        }
        for method in METHODS:
            distances = [row["methods"][method]["distance"] for row in rows]
            active = [
                row["methods"][method].get("active_points")
                for row in rows
                if row["methods"][method].get("active_points") is not None
            ]
            cp_counts = [
                row["methods"][method].get("critical_point_count")
                for row in rows
                if row["methods"][method].get("critical_point_count") is not None
            ]
            out["scenes"][scene]["methods"][method] = {
                "D_mean": float(np.mean(distances)),
                "D_min": float(np.min(distances)),
                "R_detect": _rate(risk_den, method),
                "R_body": _rate(body_den, method),
                "R_future": _rate(future_den, method),
                "R_false": float(
                    sum(
                        (not row["true_risk"]) and row["methods"][method]["risk_detected"]
                        for row in rows
                    )
                    / max(len(rows), 1)
                ),
                "T_ms_mean": float(np.mean([row["methods"][method]["time_ms"] for row in rows])),
                "active_points_mean": float(np.mean(active)) if active else None,
                "critical_points_mean": float(np.mean(cp_counts)) if cp_counts else None,
                "nearest_links": count_values(
                    [row["methods"][method]["nearest_link"] for row in rows]
                ),
            }
    return out


def _rate(rows: list[dict[str, Any]], method: str) -> float | None:
    if not rows:
        return None
    return float(sum(row["methods"][method]["risk_detected"] for row in rows) / len(rows))


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.4f}"
    return str(value)


def markdown_table(metrics: dict[str, Any]) -> str:
    headers = [
        "scene",
        "method",
        "trials",
        "body-only",
        "future-only",
        "R_detect",
        "R_body",
        "R_future",
        "R_false",
        "D_mean",
        "D_min",
        "T_ms",
        "critical_points",
        "active_points",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for scene, data in metrics["scenes"].items():
        for method in METHODS:
            vals = data["methods"][method]
            lines.append(
                "| "
                + " | ".join(
                    [
                        scene,
                        METHOD_NAMES[method],
                        str(data["trials"]),
                        str(data["body_only_events"]),
                        str(data["future_only_events"]),
                        fmt(vals["R_detect"]),
                        fmt(vals["R_body"]),
                        fmt(vals["R_future"]),
                        fmt(vals["R_false"]),
                        fmt(vals["D_mean"]),
                        fmt(vals["D_min"]),
                        fmt(vals["T_ms_mean"]),
                        fmt(vals["critical_points_mean"]),
                        fmt(vals["active_points_mean"]),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def plot_summary(metrics: dict[str, Any], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[exp_63_apf] skip plot: {exc}")
        return
    scenes = [scene for scene in SCENARIOS if scene in metrics["scenes"]]
    x = np.arange(len(scenes))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for offset, method in [(-width / 2, "critical_point_apf"), (width / 2, "ours_ccro")]:
        body = [
            metrics["scenes"][scene]["methods"][method]["R_body"]
            if metrics["scenes"][scene]["methods"][method]["R_body"] is not None
            else np.nan
            for scene in scenes
        ]
        future = [
            metrics["scenes"][scene]["methods"][method]["R_future"]
            if metrics["scenes"][scene]["methods"][method]["R_future"] is not None
            else np.nan
            for scene in scenes
        ]
        axes[0].bar(x + offset, body, width, label=METHOD_NAMES[method])
        axes[1].bar(x + offset, future, width, label=METHOD_NAMES[method])
    for ax, title in zip(axes, ["Body-only detection", "Future-only detection"]):
        ax.set_xticks(x, scenes, rotation=18)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("rate")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/ccro_stage2.yaml")
    parser.add_argument("--urdf", default=None)
    parser.add_argument("--output", default="data/results/ch6_e1_e5/E1_occupancy_risk_final/critical_point_apf")
    parser.add_argument("--scenes", default="ee_near,body_near,dynamic_future")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=6310)
    parser.add_argument("--q-base", type=parse_csv_floats, default=parse_csv_floats("0.2,-0.55,1.25,0.15,0.85,0.05"))
    parser.add_argument("--q-noise", type=parse_csv_floats, default=parse_csv_floats("0.35,0.25,0.25,0.35,0.25,0.35"))
    parser.add_argument("--ee-links", nargs="*", default=None)
    parser.add_argument("--body-links", nargs="*", default=None)
    parser.add_argument("--links", nargs="*", default=["all"])
    parser.add_argument("--d-safe", type=float, default=None)
    parser.add_argument("--apf-activate", type=float, default=0.18)
    parser.add_argument("--obstacle-radius", type=float, default=None)
    parser.add_argument("--obstacle-points", type=int, default=None)
    parser.add_argument("--prediction-horizon", type=float, default=0.8)
    parser.add_argument("--prediction-step", type=float, default=0.1)
    parser.add_argument("--risk-margin", type=float, default=0.015)
    parser.add_argument("--dynamic-speed-min", type=float, default=0.12)
    parser.add_argument("--dynamic-speed-max", type=float, default=0.20)
    parser.add_argument("--critical-point-density", default="coarse", choices=["coarse", "medium", "dense"])
    parser.add_argument("--points-per-link", type=int, default=3)
    parser.add_argument("--plot", action="store_true")
    return load_defaults(parser.parse_args())


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    surface = RobotSurfaceModel(
        args.urdf,
        args.joint_names,
        args.density_totals,
        seed=args.surface_seed,
        min_points_per_link=args.min_points_per_link,
        cache_dir=args.cache_dir,
    )
    scenes = [item.strip() for item in args.scenes.split(",") if item.strip()]
    trials = []
    for scene_index, scene in enumerate(scenes):
        for trial_index in range(args.trials):
            seed = args.seed + 1009 * scene_index + trial_index
            trial = make_trial(args, surface, scene, seed)
            trials.append(trial)
            with (output / f"trial_{scene}_{trial_index:02d}.json").open("w", encoding="utf-8") as handle:
                json.dump(trial, handle, indent=2, ensure_ascii=False)
    metrics = aggregate(trials)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    table = markdown_table(metrics)
    (output / "table_E1_critical_point_apf.md").write_text(table + "\n", encoding="utf-8")
    if args.plot:
        plot_summary(metrics, output / "fig_E1_critical_point_apf.png")
    print(table)
    print(f"\n[exp_63_apf] saved results to {output}")


if __name__ == "__main__":
    main()
