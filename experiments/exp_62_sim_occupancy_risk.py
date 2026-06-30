"""Chapter 6.2 synthetic occupancy and object-level risk experiments.

This program is intentionally independent from RealSense and real robot IO.  It
creates repeatable point-cloud scenes with a simple robot surface, dynamic
obstacles, and background noise, then compares current-frame occupancy,
voxel-memory occupancy, OctoMap-like decayed occupancy, and object-level
spatiotemporal risk occupancy.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from perception.geometry_fit import make_occupancy_object
from perception.occupancy_tracker import OccupancyTracker
from perception.clustering import cluster_points
from risk.prediction import RiskSphere, predict_risk_spheres
from risk.safety_policy import RiskLevel, SafetyPolicy


METHODS = ("current_clustering", "voxel_occupancy", "octomap_decay", "ours_stro")
METHOD_NAMES = {
    "current_clustering": "Current Clustering",
    "voxel_occupancy": "Voxel Occupancy",
    "octomap_decay": "OctoMap-like",
    "ours_stro": "Ours-STRO",
}


@dataclasses.dataclass
class SphereObstacle:
    center: np.ndarray
    radius: float
    active: bool = True


@dataclasses.dataclass
class SceneFrame:
    frame_index: int
    timestamp: float
    robot_points: np.ndarray
    obstacle_points: np.ndarray
    background_points: np.ndarray
    obstacles: list[SphereObstacle]

    @property
    def common_points(self) -> np.ndarray:
        parts = [self.robot_points, self.obstacle_points, self.background_points]
        return np.vstack([p for p in parts if len(p)]) if any(len(p) for p in parts) else np.empty((0, 3))


@dataclasses.dataclass
class MethodFrame:
    method: str
    timestamp: float
    occupancy_points: np.ndarray
    risk_spheres: list[RiskSphere]
    distance: float
    state: str
    detected: bool
    keep_ratio: float
    over_ratio: float
    ghost_ratio: float
    process_time_ms: float


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def sample_sphere_surface(
    rng: np.random.Generator,
    center: np.ndarray,
    radius: float,
    count: int,
    noise: float = 0.003,
) -> np.ndarray:
    vec = rng.normal(size=(count, 3))
    vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
    pts = center + radius * vec
    if noise > 0:
        pts += rng.normal(scale=noise, size=pts.shape)
    return pts


def sample_segment_surface(
    rng: np.random.Generator,
    a: np.ndarray,
    b: np.ndarray,
    radius: float,
    count: int,
    noise: float = 0.002,
) -> np.ndarray:
    axis = b - a
    length = float(np.linalg.norm(axis))
    direction = axis / max(length, 1e-12)
    helper = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(direction, helper)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    s = rng.random(count)
    theta = rng.random(count) * 2.0 * np.pi
    pts = a + s[:, None] * axis
    pts += radius * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)
    if noise > 0:
        pts += rng.normal(scale=noise, size=pts.shape)
    return pts


def robot_surface_points(rng: np.random.Generator, t: float, count_per_link: int = 180) -> np.ndarray:
    sway = 0.03 * math.sin(0.55 * t)
    links = [
        (np.array([0.0, 0.0, 0.10]), np.array([0.34, 0.02 + sway, 0.28]), 0.045),
        (np.array([0.34, 0.02 + sway, 0.28]), np.array([0.62, -0.04 + 0.5 * sway, 0.36]), 0.040),
        (np.array([0.62, -0.04 + 0.5 * sway, 0.36]), np.array([0.78, 0.02, 0.30]), 0.035),
    ]
    pts = [sample_segment_surface(rng, a, b, r, count_per_link) for a, b, r in links]
    return np.vstack(pts)


def obstacle_state(scene: str, t: float, trial_seed: int) -> list[SphereObstacle]:
    phase = 0.02 * (trial_seed % 7)
    if scene == "static_safe":
        return [SphereObstacle(np.array([0.54, 0.34, 0.36 + phase]), 0.055, True)]
    if scene == "approach":
        center = np.array([0.55, 0.62 - 0.095 * t, 0.35 + phase])
        return [SphereObstacle(center, 0.055, True)]
    if scene == "crossing":
        center = np.array([0.30 + 0.060 * t, 0.42 - 0.055 * t, 0.31 + phase])
        return [SphereObstacle(center, 0.060, True)]
    if scene == "leave":
        if t < 1.5:
            center = np.array([0.55, 0.22, 0.34 + phase])
        else:
            center = np.array([0.55, 0.22 + 0.16 * (t - 1.5), 0.34 + phase])
        active = t < 4.7
        return [SphereObstacle(center, 0.055, active)]
    raise ValueError(f"unknown scene: {scene}")


def generate_scene(
    scene: str,
    seed: int,
    frames: int = 90,
    dt: float = 0.08,
    obstacle_points: int = 220,
    background_points: int = 120,
) -> list[SceneFrame]:
    rng = _rng(seed)
    out = []
    for i in range(frames):
        t = i * dt
        robot = robot_surface_points(rng, t)
        obstacles = obstacle_state(scene, t, seed)
        obs_pts = []
        for obs in obstacles:
            if obs.active:
                obs_pts.append(sample_sphere_surface(rng, obs.center, obs.radius, obstacle_points))
        obstacle_cloud = np.vstack(obs_pts) if obs_pts else np.empty((0, 3))
        bg = np.column_stack(
            [
                rng.uniform(-0.10, 0.90, background_points),
                rng.uniform(-0.65, 0.75, background_points),
                rng.uniform(0.02, 0.72, background_points),
            ]
        )
        out.append(SceneFrame(i, t, robot, obstacle_cloud, bg, obstacles))
    return out


def self_filter(common_points: np.ndarray, robot_points: np.ndarray, radius: float) -> np.ndarray:
    if len(common_points) == 0:
        return np.empty((0, 3))
    tree = cKDTree(robot_points)
    d, _ = tree.query(common_points, k=1)
    return common_points[d > radius]


def voxelize(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3))
    keys = np.floor(points / voxel_size).astype(np.int64)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((len(uniq), 3), dtype=float)
    np.add.at(sums, inverse, points)
    counts = np.bincount(inverse)
    return sums / counts[:, None]


def min_points_to_robot(points: np.ndarray, robot_points: np.ndarray) -> float:
    if len(points) == 0:
        return math.inf
    tree = cKDTree(robot_points)
    d, _ = tree.query(points, k=1)
    return float(np.min(d))


def min_spheres_to_robot(spheres: list[RiskSphere], robot_points: np.ndarray) -> float:
    if not spheres:
        return math.inf
    tree = cKDTree(robot_points)
    best = math.inf
    for sphere in spheres:
        d, _ = tree.query(sphere.center, k=1)
        best = min(best, float(d - sphere.radius))
    return best


def true_obstacle_distance(frame: SceneFrame) -> float:
    active = [obs for obs in frame.obstacles if obs.active]
    if not active:
        return math.inf
    tree = cKDTree(frame.robot_points)
    best = math.inf
    for obs in active:
        d, _ = tree.query(obs.center, k=1)
        best = min(best, float(d - obs.radius))
    return best


def detection_keep_over(
    occupancy_points: np.ndarray,
    obstacle_points: np.ndarray,
    obstacles: list[SphereObstacle],
    detect_radius: float = 0.12,
    coverage_radius: float = 0.055,
) -> tuple[bool, float, float]:
    active = [obs for obs in obstacles if obs.active]
    if not active:
        return False, math.nan, math.nan
    if len(occupancy_points) == 0:
        return False, 0.0, 0.0
    occ_tree = cKDTree(occupancy_points)
    detected = any(float(occ_tree.query(obs.center, k=1)[0]) <= detect_radius for obs in active)
    if len(obstacle_points) == 0:
        return detected, 0.0, 0.0
    d, _ = occ_tree.query(obstacle_points, k=1)
    kept = int(np.count_nonzero(d <= coverage_radius))
    obs_tree = cKDTree(obstacle_points)
    occ_dist, _ = obs_tree.query(occupancy_points, k=1)
    over = int(np.count_nonzero(occ_dist > coverage_radius))
    return detected, float(kept / max(len(obstacle_points), 1)), float(over / max(len(occupancy_points), 1))


def stable_external_points(points: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, list[np.ndarray]]:
    """Cluster external points and discard sparse background noise.

    The voxel and OctoMap-like baselines should compare occupancy expression,
    not whether a method happens to keep every isolated synthetic noise point.
    This mirrors the real pipeline where workspace cropping is followed by
    clustering or plane/noise rejection before object-level evaluation.
    """
    clusters = cluster_points(points, eps=args.cluster_eps, min_points=args.cluster_min_points)
    stable = np.vstack(clusters) if clusters else np.empty((0, 3))
    return stable, clusters


def ghost_ratio(occupancy_points: np.ndarray, frame: SceneFrame, ghost_region: tuple[np.ndarray, float]) -> float:
    center, radius = ghost_region
    active = any(obs.active and np.linalg.norm(obs.center - center) <= radius for obs in frame.obstacles)
    if active or len(occupancy_points) == 0:
        return 0.0
    d = np.linalg.norm(occupancy_points - center, axis=1)
    return float(np.count_nonzero(d <= radius) / len(occupancy_points))


class VoxelMemory:
    def __init__(self, voxel_size: float, decay_frames: int):
        self.voxel_size = float(voxel_size)
        self.decay_frames = int(decay_frames)
        self.cells: dict[tuple[int, int, int], tuple[np.ndarray, int]] = {}

    def update(self, points: np.ndarray, frame_index: int) -> np.ndarray:
        if len(points):
            keys = np.floor(points / self.voxel_size).astype(np.int64)
            for key, point in zip(map(tuple, keys), points):
                prev = self.cells.get(key)
                if prev is None:
                    self.cells[key] = (point.copy(), frame_index)
                else:
                    self.cells[key] = (0.6 * prev[0] + 0.4 * point, frame_index)
        expired = [key for key, (_, last) in self.cells.items() if frame_index - last > self.decay_frames]
        for key in expired:
            del self.cells[key]
        if not self.cells:
            return np.empty((0, 3))
        return np.vstack([value[0] for value in self.cells.values()])


def run_trial(args: argparse.Namespace, scene: str, seed: int) -> dict[str, Any]:
    frames = generate_scene(scene, seed, frames=args.frames, dt=args.dt)
    policy = SafetyPolicy(d_safe=args.d_safe, d_slow=args.d_slow, d_stop=args.d_stop)
    trackers = {
        method: OccupancyTracker(
            association_distance=0.22,
            alpha=0.45,
            pos_alpha=0.55,
            motion_gate=0.001,
            velocity_dead_zone=0.004,
            shape_alpha=0.35,
        )
        for method in METHODS
    }
    voxel_memory = VoxelMemory(args.voxel_size, decay_frames=0)
    octomap_memory = VoxelMemory(args.voxel_size, decay_frames=args.octomap_decay_frames)
    rows: dict[str, list[MethodFrame]] = {method: [] for method in METHODS}

    ghost_center = np.array([0.55, 0.22, 0.34 + 0.02 * (seed % 7)])
    ghost_region = (ghost_center, 0.18)

    for frame in frames:
        external = self_filter(frame.common_points, frame.robot_points, args.self_filter_radius)
        stable_external, stable_clusters = stable_external_points(external, args)

        for method in METHODS:
            t0 = time.perf_counter()
            risk_spheres: list[RiskSphere] = []
            if method == "current_clustering":
                occupancy = stable_external
                distance = min_points_to_robot(occupancy, frame.robot_points)
            elif method == "voxel_occupancy":
                occupancy = voxel_memory.update(voxelize(stable_external, args.voxel_size), frame.frame_index)
                distance = min_points_to_robot(occupancy, frame.robot_points)
            elif method == "octomap_decay":
                occupancy = octomap_memory.update(voxelize(stable_external, args.voxel_size), frame.frame_index)
                distance = min_points_to_robot(occupancy, frame.robot_points)
            else:
                detections = [
                    make_occupancy_object(cluster, timestamp=frame.timestamp, margin=args.shape_margin)
                    for cluster in stable_clusters
                ]
                tracked = trackers[method].update(detections, timestamp=frame.timestamp)
                stable = [obj for obj in tracked if obj.age >= args.min_track_age]
                risk_spheres = predict_risk_spheres(
                    stable,
                    horizon=args.prediction_horizon,
                    step=args.prediction_step,
                    margin=args.risk_margin,
                    uncertainty=args.prediction_uncertainty,
                    static_speed_threshold=args.static_speed_threshold,
                    static_margin=args.static_margin,
                    velocity_radius_scale=args.velocity_radius_scale,
                )
                centers = [sphere.center for sphere in risk_spheres]
                occupancy = np.vstack([stable_external, np.vstack(centers)]) if centers and len(stable_external) else (
                    np.vstack(centers) if centers else stable_external
                )
                distance = min_spheres_to_robot(risk_spheres, frame.robot_points)

            decision = policy.evaluate(distance)
            detected, keep, over = detection_keep_over(
                occupancy,
                frame.obstacle_points,
                frame.obstacles,
                coverage_radius=max(args.voxel_size * 0.9, 0.045),
            )
            rows[method].append(
                MethodFrame(
                    method=method,
                    timestamp=frame.timestamp,
                    occupancy_points=occupancy,
                    risk_spheres=risk_spheres,
                    distance=distance,
                    state=decision.level.value,
                    detected=detected,
                    keep_ratio=keep,
                    over_ratio=over,
                    ghost_ratio=ghost_ratio(occupancy, frame, ghost_region),
                    process_time_ms=(time.perf_counter() - t0) * 1000.0,
                )
            )

    reference = [
        {
            "frame_index": frame.frame_index,
            "timestamp": frame.timestamp,
            "d_true": true_obstacle_distance(frame),
            "active": any(obs.active for obs in frame.obstacles),
        }
        for frame in frames
    ]
    metrics = {method: aggregate_method(rows[method], reference, args) for method in METHODS}
    return {
        "scene": scene,
        "seed": seed,
        "parameters": vars(args),
        "reference": reference,
        "metrics": metrics,
        "series": {
            method: [
                {
                    "timestamp": row.timestamp,
                    "distance": row.distance,
                    "state": row.state,
                    "detected": row.detected,
                    "keep_ratio": row.keep_ratio,
                    "over_ratio": row.over_ratio,
                    "ghost_ratio": row.ghost_ratio,
                    "process_time_ms": row.process_time_ms,
                }
                for row in method_rows
            ]
            for method, method_rows in rows.items()
        },
    }


def aggregate_method(rows: list[MethodFrame], reference: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    active = [i for i, r in enumerate(reference) if r["active"]]
    det_den = len(active)
    det = sum(rows[i].detected for i in active) if active else 0
    keep_values = [rows[i].keep_ratio for i in active if not math.isnan(rows[i].keep_ratio)]
    over_values = [rows[i].over_ratio for i in active if not math.isnan(rows[i].over_ratio)]
    ghost_candidates = [i for i, r in enumerate(reference) if not r["active"]]
    ghost_values = [rows[i].ghost_ratio for i in ghost_candidates]

    danger = next((i for i, r in enumerate(reference) if r["d_true"] <= args.danger_threshold), None)
    warn = next((i for i, row in enumerate(rows) if row.state != RiskLevel.SAFE.value), None)
    if danger is None:
        t_lead = None
        miss = None
    elif warn is None:
        t_lead = None
        miss = 1.0
    else:
        t_lead = float(reference[danger]["timestamp"] - reference[warn]["timestamp"])
        miss = float(reference[warn]["timestamp"] > reference[danger]["timestamp"] - args.t_req)

    horizon_steps = max(1, int(math.ceil(args.prediction_horizon / max(args.dt, 1e-9))))
    false_den = []
    for i in range(len(reference)):
        future = reference[i : min(len(reference), i + horizon_steps + 1)]
        future_min = min(r["d_true"] for r in future)
        if future_min > args.d_safe:
            false_den.append(i)
    false_num = sum(rows[i].state != RiskLevel.SAFE.value for i in false_den)

    return {
        "R_det": float(det / det_den) if det_den else None,
        "R_keep": float(np.mean(keep_values)) if keep_values else None,
        "R_over": float(np.mean(over_values)) if over_values else None,
        "R_ghost": float(np.mean(ghost_values)) if ghost_values else 0.0,
        "T_lead": t_lead,
        "R_miss": miss,
        "R_false_time": float(false_num / len(false_den)) if false_den else None,
        "T_process_ms_mean": float(np.mean([row.process_time_ms for row in rows])),
        "T_process_ms_p95": float(np.percentile([row.process_time_ms for row in rows], 95)),
    }


def aggregate_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for trial in trials:
        for method, vals in trial["metrics"].items():
            grouped[trial["scene"]][method].append(vals)

    out: dict[str, Any] = {"trial_count": len(trials), "scenes": {}}
    for scene, methods in grouped.items():
        out["scenes"][scene] = {}
        for method, vals in methods.items():
            out["scenes"][scene][method] = {}
            keys = sorted({key for row in vals for key in row})
            for key in keys:
                xs = [row[key] for row in vals if row.get(key) is not None]
                if not xs:
                    out["scenes"][scene][method][key] = None
                    continue
                arr = np.asarray(xs, dtype=float)
                mean = float(np.mean(arr))
                std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                ci95 = float(1.96 * std / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
                out["scenes"][scene][method][key] = mean
                out["scenes"][scene][method][f"{key}_std"] = std
                out["scenes"][scene][method][f"{key}_ci95"] = ci95
    return out


def markdown_table(aggregate: dict[str, Any]) -> str:
    lines = []
    headers = ["scene", "method", "R_det", "R_keep", "R_over", "R_ghost", "T_lead", "R_miss", "R_false_time", "T_ms"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for scene, methods in aggregate["scenes"].items():
        for method in METHODS:
            vals = methods.get(method, {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        scene,
                        METHOD_NAMES[method],
                        fmt(vals.get("R_det")),
                        fmt(vals.get("R_keep")),
                        fmt(vals.get("R_over")),
                        fmt(vals.get("R_ghost")),
                        fmt(vals.get("T_lead")),
                        fmt(vals.get("R_miss")),
                        fmt(vals.get("R_false_time")),
                        fmt(vals.get("T_process_ms_mean")),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def markdown_stats_table(aggregate: dict[str, Any]) -> str:
    lines = []
    headers = ["scene", "method", "metric", "mean", "std", "ci95"]
    metrics = ["R_det", "R_keep", "R_over", "R_ghost", "T_lead", "R_miss", "R_false_time", "T_process_ms_mean"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for scene, methods in aggregate["scenes"].items():
        for method in METHODS:
            vals = methods.get(method, {})
            for metric in metrics:
                if vals.get(metric) is None:
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            scene,
                            METHOD_NAMES[method],
                            metric,
                            fmt(vals.get(metric)),
                            fmt(vals.get(f"{metric}_std")),
                            fmt(vals.get(f"{metric}_ci95")),
                        ]
                    )
                    + " |"
                )
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.4f}"
    return str(value)


def plot_trial(trial: dict[str, Any], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[exp_62] skip plot: {exc}")
        return

    ref = trial["reference"]
    ts = np.array([r["timestamp"] for r in ref])
    d_true = np.array([r["d_true"] for r in ref], dtype=float)
    d_true = np.where(np.isfinite(d_true), d_true, np.nan)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(ts, d_true, label="true obstacle distance", color="black")
    axes[0].axhline(trial["parameters"]["d_safe"], linestyle="--", color="tab:orange", label="d_safe")
    axes[0].axhline(trial["parameters"]["danger_threshold"], linestyle="--", color="tab:red", label="danger")
    for method in METHODS:
        ys = np.array([row["distance"] for row in trial["series"][method]], dtype=float)
        ys = np.where(np.isfinite(ys), ys, np.nan)
        axes[0].plot(ts[: len(ys)], ys, alpha=0.75, label=METHOD_NAMES[method])
    axes[0].set_ylabel("distance (m)")
    axes[0].legend(loc="best", fontsize=8)

    levels = {"SAFE": 0, "WARNING": 1, "SLOW": 2, "STOP": 3}
    for method in METHODS:
        state = [levels.get(row["state"], 0) for row in trial["series"][method]]
        axes[1].step(ts[: len(state)], state, where="post", label=METHOD_NAMES[method])
    axes[1].set_yticks([0, 1, 2, 3], ["SAFE", "WARN", "SLOW", "STOP"])
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("risk state")
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chapter 6.2 synthetic occupancy and STRO comparison.")
    parser.add_argument("--output", default="data/results/ch6_2_sim")
    parser.add_argument("--scenes", default="static_safe,approach,crossing,leave")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=6200)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--self-filter-radius", type=float, default=0.035)
    parser.add_argument("--cluster-eps", type=float, default=0.065)
    parser.add_argument("--cluster-min-points", type=int, default=25)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--octomap-decay-frames", type=int, default=14)
    parser.add_argument("--d-safe", type=float, default=0.16)
    parser.add_argument("--d-slow", type=float, default=0.10)
    parser.add_argument("--d-stop", type=float, default=0.05)
    parser.add_argument("--danger-threshold", type=float, default=0.08)
    parser.add_argument("--t-req", type=float, default=0.25)
    parser.add_argument("--shape-margin", type=float, default=0.015)
    parser.add_argument("--min-track-age", type=int, default=2)
    parser.add_argument("--prediction-horizon", type=float, default=0.65)
    parser.add_argument("--prediction-step", type=float, default=0.10)
    parser.add_argument("--risk-margin", type=float, default=0.04)
    parser.add_argument("--prediction-uncertainty", type=float, default=0.015)
    parser.add_argument("--static-speed-threshold", type=float, default=0.03)
    parser.add_argument("--static-margin", type=float, default=0.01)
    parser.add_argument("--velocity-radius-scale", type=float, default=0.80)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scenes = [item.strip() for item in args.scenes.split(",") if item.strip()]
    trials = []
    for scene in scenes:
        for trial_idx in range(args.trials):
            seed = args.seed + 101 * trial_idx + 17 * scenes.index(scene)
            trial = run_trial(args, scene, seed)
            trials.append(trial)
            trial_path = output / f"trial_{scene}_{trial_idx:02d}.json"
            with trial_path.open("w", encoding="utf-8") as handle:
                json.dump(trial, handle, indent=2, ensure_ascii=False, default=_json_default)
            if args.plot and trial_idx == 0:
                plot_trial(trial, output / f"fig_{scene}.png")

    aggregate = aggregate_trials(trials)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, ensure_ascii=False)
    table = markdown_table(aggregate)
    with (output / "table_6_2_sim.md").open("w", encoding="utf-8") as handle:
        handle.write(table + "\n")
    stats_table = markdown_stats_table(aggregate)
    with (output / "table_6_2_sim_stats.md").open("w", encoding="utf-8") as handle:
        handle.write(stats_table + "\n")
    print(table)
    print(f"\n[exp_62] saved results to {output}")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
