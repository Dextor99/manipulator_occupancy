"""Shared utilities for the revised Chapter 6.2 experiments."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import config_62 as cfg


@dataclass(frozen=True)
class ReferenceTrajectory:
    times: np.ndarray
    q: np.ndarray

    def sample(self, t: float) -> np.ndarray:
        value = float(np.clip(t, self.times[0], self.times[-1]))
        cols = [np.interp(value, self.times, self.q[:, index]) for index in range(self.q.shape[1])]
        return np.asarray(cols, dtype=float)

    def future(self, t: float, horizon: float, step: float) -> tuple[np.ndarray, np.ndarray]:
        times = np.arange(0.0, horizon + 1.0e-9, step)
        global_times = np.clip(float(t) + times, self.times[0], self.times[-1])
        return times, np.vstack([self.sample(item) for item in global_times])


def make_reference_trajectory(dt: float = cfg.DT) -> ReferenceTrajectory:
    times = np.arange(0.0, cfg.TOTAL_DURATION + 1.0e-9, dt)
    u = times / cfg.TOTAL_DURATION
    smooth = 3.0 * u * u - 2.0 * u * u * u
    q0 = np.asarray(cfg.Q_START, dtype=float)
    q1 = np.asarray(cfg.Q_GOAL, dtype=float)
    wave = np.asarray(cfg.Q_WAVE, dtype=float)
    q = q0[None, :] + smooth[:, None] * (q1 - q0)[None, :]
    q += np.sin(np.pi * u)[:, None] * wave[None, :]
    return ReferenceTrajectory(times=times, q=q)


def save_reference_trajectory(path: Path, trajectory: ReferenceTrajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, times=trajectory.times, q=trajectory.q)


def load_reference_trajectory(path: Path) -> ReferenceTrajectory:
    with np.load(path, allow_pickle=False) as data:
        return ReferenceTrajectory(times=np.asarray(data["times"]), q=np.asarray(data["q"]))


def load_surface_model() -> RobotSurfaceModel:
    from planning.robot_surface_model import RobotSurfaceModel

    return RobotSurfaceModel(
        cfg.URDF_PATH,
        list(cfg.JOINT_NAMES),
        cfg.SURFACE_DENSITY_TOTALS,
        seed=20260623,
        min_points_per_link=64,
        cache_dir=cfg.ROOT / "data" / "cache" / "robot_surface",
        geometry="collision",
    )


def min_distance_to_sphere(
    surface: Any,
    q: np.ndarray,
    center: np.ndarray,
    radius: float,
    *,
    density: str,
    links: set[str] | None = None,
) -> tuple[float, str | None, np.ndarray | None]:
    best = math.inf
    best_link = None
    best_point = None
    for link, points in surface.surface_by_link(q, density=density, links=links).items():
        if len(points) == 0:
            continue
        distances = np.linalg.norm(points - center[None, :], axis=1) - float(radius)
        index = int(np.argmin(distances))
        value = float(distances[index])
        if value < best:
            best = value
            best_link = link
            best_point = points[index].copy()
    return best, best_link, best_point


def min_distance_points_to_sphere(points: np.ndarray, center: np.ndarray, radius: float) -> float:
    if len(points) == 0:
        return math.inf
    return float(np.min(np.linalg.norm(points - center[None, :], axis=1) - radius))


def nearest_link_for_sphere(
    surface: Any,
    q: np.ndarray,
    center: np.ndarray,
    radius: float,
    *,
    density: str = "dense",
) -> tuple[float, str | None]:
    distance, link, _ = min_distance_to_sphere(surface, q, center, radius, density=density)
    return distance, link


def outward_direction(points: np.ndarray, point: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    center = np.mean(points, axis=0) if len(points) else np.zeros(3)
    direction = np.asarray(point, dtype=float) - center
    if np.linalg.norm(direction) < 1.0e-9:
        direction = rng.normal(size=3)
    direction = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
    return direction


def perpendicular_direction(direction: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    base = np.asarray(direction, dtype=float)
    helper = rng.normal(size=3)
    perp = np.cross(base, helper)
    if np.linalg.norm(perp) < 1.0e-9:
        helper = np.array([0.0, 0.0, 1.0])
        perp = np.cross(base, helper)
    return perp / max(float(np.linalg.norm(perp)), 1.0e-12)


def sample_sphere_surface(
    rng: np.random.Generator,
    center: np.ndarray,
    radius: float,
    count: int,
    *,
    noise: float = cfg.POINT_NOISE_SIGMA,
    dropout: float = cfg.POINT_DROPOUT,
) -> np.ndarray:
    vectors = rng.normal(size=(count, 3))
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1.0e-12)
    points = center[None, :] + radius * vectors
    if noise > 0:
        points += rng.normal(scale=noise, size=points.shape)
    if dropout > 0:
        keep = rng.random(len(points)) >= dropout
        points = points[keep]
    return points


def lead_time(t_risk: float | None, t_alarm: float | None) -> float | None:
    if t_risk is None or t_alarm is None:
        return None
    if float(t_alarm) > float(t_risk):
        return None
    return float(t_risk) - float(t_alarm)


def summarize_leads(values: list[float | None]) -> dict[str, Any]:
    successes = np.asarray([value for value in values if value is not None], dtype=float)
    total = len(values)
    misses = total - int(len(successes))
    return {
        "mean": float(np.mean(successes)) if len(successes) else math.nan,
        "std": float(np.std(successes, ddof=1)) if len(successes) > 1 else 0.0,
        "misses": misses,
        "total": total,
    }


def scenario_distance_diagnostics(distances: list[float] | np.ndarray, times: list[float] | np.ndarray) -> dict[str, Any]:
    d = np.asarray(distances, dtype=float)
    t = np.asarray(times, dtype=float)
    if d.shape != t.shape or d.ndim != 1 or len(d) == 0:
        raise ValueError("distances and times must be non-empty 1D arrays with the same shape")
    min_index = int(np.argmin(d))
    contact_indices = np.flatnonzero(d < 0.0)
    risk_indices = np.flatnonzero(d <= cfg.DYNAMIC_ALARM_DISTANCE)
    return {
        "min_D_gt": float(d[min_index]),
        "time_of_min_D_gt": float(t[min_index]),
        "first_contact_time": None if len(contact_indices) == 0 else float(t[int(contact_indices[0])]),
        "t_risk": None if len(risk_indices) == 0 else float(t[int(risk_indices[0])]),
        "valid_min_clearance": bool(cfg.MIN_SCENE_CLEARANCE_RANGE[0] <= float(d[min_index]) <= cfg.MIN_SCENE_CLEARANCE_RANGE[1]),
    }


def median_runtime_ms(fn: Callable[[], Any], *, warmup: int = cfg.RUNTIME_WARMUP, repeats: int = cfg.RUNTIME_REPEATS) -> float:
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        values.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(values))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_mean_std(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "miss"
    return f"{mean:.3f} ± {std:.3f}"


def ensure_output_tree(output: Path) -> dict[str, Path]:
    paths = {
        "root": output,
        "calibration": output / "calibration",
        "dynamic": output / "dynamic",
        "dynamic_trials": output / "dynamic" / "trials",
        "body": output / "body",
        "body_trials": output / "body" / "trials",
        "paper": output / "paper",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
