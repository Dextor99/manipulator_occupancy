"""Chapter 6.3 configuration-coupled whole-body risk distance simulation."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from planning.mesh_risk import StaticObstacleField
from planning.robot_surface_model import RobotSurfaceModel
from utils.config import load_yaml


METHODS = ("eef_only", "body_current", "ours_ccro")
METHOD_NAMES = {
    "eef_only": "EEF-only",
    "body_current": "Body-current",
    "ours_ccro": "Ours-CCRO",
}

SCENARIOS = ("ee_near", "body_near", "dynamic_future")


def sample_sphere(rng: np.random.Generator, center: np.ndarray, radius: float, count: int) -> np.ndarray:
    vec = rng.normal(size=(count, 3))
    vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
    return center + radius * vec + rng.normal(scale=0.002, size=(count, 3))


def unit_random(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    v /= max(float(np.linalg.norm(v)), 1e-12)
    return v


def distance_to_links(surface: RobotSurfaceModel, q: np.ndarray, obstacle_points: np.ndarray, links: set[str] | None, density: str) -> tuple[float, str | None]:
    if len(obstacle_points) == 0:
        return math.inf, None
    obstacle = StaticObstacleField.from_points(obstacle_points)
    best_d = math.inf
    best_link = None
    surfaces = surface.surface_by_link(q, density=density, links=links)
    for link, points in surfaces.items():
        if len(points) == 0:
            continue
        tree = cKDTree(points)
        d, _ = tree.query(obstacle.points, k=1)
        local = float(np.min(d))
        if local < best_d:
            best_d = local
            best_link = link
    return best_d, best_link


def choose_surface_point(
    rng: np.random.Generator,
    surface: RobotSurfaceModel,
    q: np.ndarray,
    links: list[str],
    density: str,
) -> tuple[np.ndarray, str]:
    link = str(rng.choice(links))
    points = surface.surface_by_link(q, density=density, links={link})[link]
    point = points[int(rng.integers(0, len(points)))]
    return point, link


def outward_direction(
    rng: np.random.Generator,
    surface: RobotSurfaceModel,
    q: np.ndarray,
    point: np.ndarray,
) -> np.ndarray:
    robot = surface.surface(q, density="coarse")
    center = robot.mean(axis=0) if len(robot) else np.zeros(3)
    direction = point - center
    if np.linalg.norm(direction) < 1e-9:
        direction = unit_random(rng)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    direction = direction + 0.15 * unit_random(rng)
    return direction / max(float(np.linalg.norm(direction)), 1e-12)


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
    future_times = np.arange(args.prediction_step, args.prediction_horizon + 1e-9, args.prediction_step)
    for tau in future_times:
        shifted_center = center + velocity * tau
        shifted = sample_sphere(rng, shifted_center, args.obstacle_radius + args.risk_margin, args.obstacle_points)
        future_chunks.append(shifted)
        d_tau, _ = distance_to_links(surface, q, shifted, links=None, density="dense")
        future_distances.append(d_tau)
    future_obs = np.vstack(future_chunks)

    d_ee, link_ee = distance_to_links(surface, q, current_obs, set(args.ee_links), "dense")
    d_body_current, link_body_current = distance_to_links(surface, q, current_obs, None, "dense")
    d_future = min([d_body_current, *future_distances]) if future_distances else d_body_current
    true_risk = d_future < args.d_safe
    body_only = d_body_current < args.d_safe and d_ee > args.d_safe
    future_only = d_body_current > args.d_safe and d_future < args.d_safe

    method_inputs = {
        "eef_only": (current_obs, set(args.ee_links)),
        "body_current": (current_obs, None),
        "ours_ccro": (future_obs, None),
    }
    methods: dict[str, Any] = {}
    t0 = time.perf_counter()
    for method, (obs, links) in method_inputs.items():
        m0 = time.perf_counter()
        dist, nearest = distance_to_links(surface, q, obs, links, "dense")
        methods[method] = {
            "distance": dist,
            "nearest_link": nearest,
            "risk_detected": bool(dist < args.d_safe),
            "time_ms": (time.perf_counter() - m0) * 1000.0,
        }
    return {
        "scene": scene,
        "seed": seed,
        "q": q.tolist(),
        "target_link": target_link,
        "center": center.tolist(),
        "velocity": velocity.tolist(),
        "D_ee": d_ee,
        "D_body_current": d_body_current,
        "D_future": d_future,
        "nearest_ee_link": link_ee,
        "nearest_body_link": link_body_current,
        "true_risk": true_risk,
        "body_only_event": body_only,
        "future_only_event": future_only,
        "methods": methods,
        "trial_time_ms": (time.perf_counter() - t0) * 1000.0,
    }


def aggregate(trials: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        grouped[trial["scene"]].append(trial)

    out: dict[str, Any] = {"trial_count": len(trials), "scenes": {}}
    for scene, rows in grouped.items():
        out["scenes"][scene] = {
            "trials": len(rows),
            "body_only_events": int(sum(r["body_only_event"] for r in rows)),
            "future_only_events": int(sum(r["future_only_event"] for r in rows)),
            "nearest_links": count_values([r["nearest_body_link"] for r in rows]),
            "methods": {},
        }
        risk_den = [r for r in rows if r["true_risk"]]
        body_den = [r for r in rows if r["body_only_event"]]
        future_den = [r for r in rows if r["future_only_event"]]
        for method in METHODS:
            distances = [r["methods"][method]["distance"] for r in rows]
            detect_risk = [r for r in risk_den if r["methods"][method]["risk_detected"]]
            detect_body = [r for r in body_den if r["methods"][method]["risk_detected"]]
            detect_future = [r for r in future_den if r["methods"][method]["risk_detected"]]
            false_pos = [
                r for r in rows
                if not r["true_risk"] and r["methods"][method]["risk_detected"]
            ]
            out["scenes"][scene]["methods"][method] = {
                "D_mean": float(np.mean(distances)),
                "D_min": float(np.min(distances)),
                "R_detect": None if not risk_den else float(len(detect_risk) / len(risk_den)),
                "R_body": None if not body_den else float(len(detect_body) / len(body_den)),
                "R_future": None if not future_den else float(len(detect_future) / len(future_den)),
                "R_false": float(len(false_pos) / len(rows)),
                "T_ms_mean": float(np.mean([r["methods"][method]["time_ms"] for r in rows])),
                "nearest_links": count_values([r["methods"][method]["nearest_link"] for r in rows]),
            }
    return out


def count_values(values: list[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = "-" if value is None else str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda item: (-item[1], item[0])))


def markdown_table(metrics: dict[str, Any]) -> str:
    headers = ["scene", "method", "trials", "body-only", "future-only", "R_detect", "R_body", "R_future", "R_false", "D_mean", "D_min", "T_ms"]
    rows = []
    for scene, data in metrics["scenes"].items():
        for method in METHODS:
            vals = data["methods"][method]
            rows.append(
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
                ]
            )
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.4f}"
    return str(value)


def load_defaults(args: argparse.Namespace) -> argparse.Namespace:
    cfg = load_yaml(args.config)
    robot = cfg["robot"]
    surface_cfg = cfg["surface"]
    experiment = cfg.get("experiment", {})
    risk = cfg.get("risk", {})
    args.urdf = args.urdf or robot["urdf_path"]
    args.joint_names = robot["joint_names"]
    args.density_totals = surface_cfg["density_totals"]
    args.surface_seed = surface_cfg.get("random_seed", 20260623)
    args.min_points_per_link = surface_cfg.get("min_points_per_link", 64)
    args.cache_dir = surface_cfg.get("cache_dir", "data/cache/robot_surface")
    args.ee_links = args.ee_links or experiment.get("end_effector_links", ["wrist3_Link", "gripper_base_link", "left_link", "right_link"])
    args.body_links = args.body_links or ["upperArm_Link", "foreArm_Link", "wrist1_Link", "wrist2_Link"]
    args.d_safe = args.d_safe or risk.get("d_safe", 0.12)
    args.obstacle_radius = args.obstacle_radius or risk.get("obstacle_radius", 0.035)
    args.obstacle_points = args.obstacle_points or risk.get("obstacle_points", 1200)
    return args


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chapter 6.3 whole-body risk distance simulation.")
    parser.add_argument("--config", default="config/ccro_stage2.yaml")
    parser.add_argument("--urdf", default=None)
    parser.add_argument("--output", default="data/results/ch6_3_sim")
    parser.add_argument("--scenes", default="ee_near,body_near,dynamic_future")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=6300)
    parser.add_argument("--q-base", type=parse_csv_floats, default=parse_csv_floats("0.2,-0.55,1.25,0.15,0.85,0.05"))
    parser.add_argument("--q-noise", type=parse_csv_floats, default=parse_csv_floats("0.35,0.25,0.25,0.35,0.25,0.35"))
    parser.add_argument("--ee-links", nargs="*", default=None)
    parser.add_argument("--body-links", nargs="*", default=None)
    parser.add_argument("--d-safe", type=float, default=None)
    parser.add_argument("--obstacle-radius", type=float, default=None)
    parser.add_argument("--obstacle-points", type=int, default=None)
    parser.add_argument("--prediction-horizon", type=float, default=0.8)
    parser.add_argument("--prediction-step", type=float, default=0.1)
    parser.add_argument("--risk-margin", type=float, default=0.015)
    parser.add_argument("--dynamic-speed-min", type=float, default=0.12)
    parser.add_argument("--dynamic-speed-max", type=float, default=0.20)
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
    with (output / "table_6_3_sim.md").open("w", encoding="utf-8") as handle:
        handle.write(table + "\n")
    print(table)
    print(f"\n[exp_63_sim] saved results to {output}")


if __name__ == "__main__":
    main()
