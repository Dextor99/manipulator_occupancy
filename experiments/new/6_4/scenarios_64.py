"""Scenario generation for Chapter 6.4 dynamic virtual-loop experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.exp_ccro_stage3 import _select_sweep_point

from . import config_64 as cfg
from .common_64 import min_distance_to_sphere, write_json


def _tangent_direction(outward: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    axes = np.eye(3)
    axis = axes[int(np.argmin(np.abs(axes @ outward)))]
    direction = np.cross(outward, axis)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
    angle = rng.uniform(-math.radians(5.0), math.radians(5.0))
    direction = (
        direction * math.cos(angle)
        + np.cross(outward, direction) * math.sin(angle)
        + outward * float(np.dot(outward, direction)) * (1.0 - math.cos(angle))
    )
    direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
    return direction if rng.random() < 0.5 else -direction


def obstacle_center_at(
    center0: np.ndarray,
    velocity: np.ndarray,
    timestamp: float,
    motion_start_time: float = 0.0,
    pre_motion_center: np.ndarray | None = None,
) -> np.ndarray:
    if pre_motion_center is not None and float(timestamp) + 1.0e-9 < float(motion_start_time):
        return np.asarray(pre_motion_center, dtype=np.float64)
    return np.asarray(center0, dtype=np.float64) + np.asarray(velocity, dtype=np.float64) * max(0.0, float(timestamp) - float(motion_start_time))


def obstacle_velocity_at(velocity: np.ndarray, timestamp: float, motion_start_time: float = 0.0) -> np.ndarray:
    if float(timestamp) + 1.0e-9 < float(motion_start_time):
        return np.zeros(3, dtype=np.float64)
    return np.asarray(velocity, dtype=np.float64)


def _reference_distance_rows(
    model,
    trajectory,
    center0: np.ndarray,
    velocity: np.ndarray,
    radius: float,
    motion_start_time: float = 0.0,
    pre_motion_center: np.ndarray | None = None,
    *,
    sample_count: int = 21,
    density: str = cfg.SURFACE_DENSITY_LOOP,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for timestamp in np.linspace(0.0, trajectory.total_duration, int(sample_count)):
        q = trajectory.evaluate(float(timestamp))
        center = obstacle_center_at(center0, velocity, float(timestamp), motion_start_time, pre_motion_center)
        distance, link = min_distance_to_sphere(model, q, center, radius, density)
        rows.append({"time": float(timestamp), "distance": float(distance), "nearest_link": link})
    return rows


def _observed_from_gt(rng: np.random.Generator, center0: np.ndarray, velocity: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, float]:
    return (
        center0 + rng.normal(0.0, cfg.OBS_POS_SIGMA, size=3),
        np.asarray(velocity, dtype=np.float64) + rng.normal(0.0, cfg.OBS_VEL_SIGMA, size=3),
        max(0.025, float(radius + rng.normal(0.0, cfg.OBS_RADIUS_SIGMA))),
    )


def _make_crossing_instance(
    model,
    trajectory,
    *,
    scenario_type: str,
    instance_index: int,
    speed: float,
    repeat_index: int,
    seed: int,
) -> dict[str, Any]:
    links = cfg.BODY_LINKS if scenario_type == "D1" else cfg.EE_LINKS
    time_range = (0.82, 0.98) if scenario_type == "D1" else (0.76, 0.96)
    selected = _select_sweep_point(model, trajectory, links, time_range, [])
    best_item: dict[str, Any] | None = None
    for attempt in range(80):
        attempt_seed = seed + attempt * 7919
        rng = np.random.default_rng(attempt_seed)
        radius = float(rng.uniform(0.04, 0.06))
        speed_actual = float(speed * rng.uniform(0.95, 1.05))
        direction = _tangent_direction(selected["outward"], rng)
        velocity = speed_actual * direction
        motion_start_time = float(rng.uniform(0.5, 1.2) if scenario_type in {"D2", "D2M", "D2S"} else rng.uniform(0.8, 1.6))
        if scenario_type == "D1":
            clearance_low, clearance_high = (0.060, 0.092)
        elif scenario_type == "D2M":
            clearance_low, clearance_high = (0.045, 0.074)
        else:
            clearance_low, clearance_high = (0.110, 0.165)
        clearance = float(rng.uniform(clearance_low, clearance_high))
        crossing_center = selected["surface_point"] + (radius + clearance) * selected["outward"]
        travel_time = max(0.4, float(selected["time"]) - motion_start_time)
        center0 = crossing_center - velocity * travel_time + rng.uniform(-0.02, 0.02, size=3)
        pre_motion_center = center0 + selected["outward"] * float(rng.uniform(0.35, 0.50))
        rows = _reference_distance_rows(
            model,
            trajectory,
            center0,
            velocity,
            radius,
            motion_start_time,
            pre_motion_center,
            sample_count=81,
        )
        static_rows = _reference_distance_rows(model, trajectory, pre_motion_center, np.zeros(3), radius, 0.0)
        min_row = min(rows, key=lambda item: item["distance"])
        static_min_row = min(static_rows, key=lambda item: item["distance"])
        initial_distance = rows[0]["distance"]
        if scenario_type == "D1":
            min_low, min_high = 0.065, cfg.D_STOP - 0.002
            min_time_ok = float(min_row["time"]) >= 6.0
        elif scenario_type == "D2M":
            min_low, min_high = 0.060, cfg.D_STOP - 0.002
            min_time_ok = float(min_row["time"]) >= 5.8
        else:
            min_low, min_high = 0.065, 0.120
            min_time_ok = float(min_row["time"]) >= 2.6
        valid = (
            initial_distance > cfg.D_INITIAL_SAFE
            and float(static_min_row["distance"]) > cfg.D_REPLAN_OUT + 0.10
            and min_low <= float(min_row["distance"]) <= min_high
            and min_time_ok
        )
        candidate = {
            "selected": selected,
            "radius": radius,
            "velocity": velocity,
            "motion_start_time": motion_start_time,
            "pre_motion_center": pre_motion_center,
            "center0": center0,
            "min_row": min_row,
            "static_min_distance": float(static_min_row["distance"]),
            "initial_distance": initial_distance,
            "seed": attempt_seed,
        }
        if valid:
            best_item = candidate
            break
        if best_item is None or (
            int(initial_distance > cfg.D_INITIAL_SAFE),
            -abs(float(min_row["distance"]) - (0.055 if scenario_type == "D2M" else 0.06)),
        ) > (
            int(best_item["initial_distance"] > cfg.D_INITIAL_SAFE),
            -abs(float(best_item["min_row"]["distance"]) - (0.055 if scenario_type == "D2M" else 0.06)),
        ):
            best_item = candidate
    if best_item is None:
        raise RuntimeError(f"failed to generate crossing instance {scenario_type}_{instance_index:02d}")
    selected = best_item["selected"]
    center0 = best_item["center0"]
    velocity = best_item["velocity"]
    motion_start_time = best_item["motion_start_time"]
    pre_motion_center = best_item["pre_motion_center"]
    radius = best_item["radius"]
    min_row = best_item["min_row"]
    initial_distance = best_item["initial_distance"]
    frozen_seed = int(best_item["seed"])
    obs_center0, obs_velocity, obs_radius = _observed_from_gt(
        np.random.default_rng(frozen_seed + 100_000), pre_motion_center, np.zeros(3), radius
    )
    return {
        "instance_id": f"{scenario_type}_{instance_index:02d}",
        "scenario_type": {
            "D1": "body_crossing_main",
            "D2M": "ee_crossing_main",
            "D2": "ee_crossing_stress",
            "D2S": "ee_crossing_stress",
        }[scenario_type],
        "speed_group": float(speed),
        "repeat_index": int(repeat_index),
        "seed": frozen_seed,
        "target_region": selected["link"],
        "target_time": float(selected["time"]),
        "gt_center0": center0.tolist(),
        "gt_velocity": velocity.tolist(),
        "motion_start_time": motion_start_time,
        "pre_motion_center": pre_motion_center.tolist(),
        "gt_radius": radius,
        "observed_center0": obs_center0.tolist(),
        "observed_velocity": obs_velocity.tolist(),
        "observed_radius": obs_radius,
        "observation_seed": int(frozen_seed + 100_000),
        "reference_initial_distance": float(initial_distance),
        "reference_min_distance": float(min_row["distance"]),
        "static_wait_min_distance": float(best_item["static_min_distance"]),
        "reference_risk_time": float(min_row["time"]),
        "reference_nearest_link": min_row["nearest_link"],
    }


def _make_far_instance(model, trajectory, index: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    q_mid = trajectory.evaluate(0.5 * trajectory.total_duration)
    center = np.mean(model.surface(q_mid, cfg.SURFACE_DENSITY_TRUTH), axis=0)
    center += np.array([0.80, -0.65, 0.55]) + rng.uniform(-0.03, 0.03, size=3)
    velocity = rng.uniform(-0.01, 0.01, size=3)
    radius = float(rng.uniform(0.04, 0.06))
    rows = _reference_distance_rows(model, trajectory, center, velocity, radius)
    obs_center0, obs_velocity, obs_radius = _observed_from_gt(
        np.random.default_rng(seed + 100_000), center, velocity, radius
    )
    return {
        "instance_id": f"D3_{index:02d}",
        "scenario_type": "far_safe",
        "speed_group": None,
        "repeat_index": int(index),
        "seed": int(seed),
        "target_region": "far",
        "gt_center0": center.tolist(),
        "gt_velocity": velocity.tolist(),
        "motion_start_time": 0.0,
        "gt_radius": radius,
        "observed_center0": obs_center0.tolist(),
        "observed_velocity": obs_velocity.tolist(),
        "observed_radius": obs_radius,
        "observation_seed": int(seed + 100_000),
        "reference_initial_distance": float(rows[0]["distance"]),
        "reference_min_distance": float(min(row["distance"] for row in rows)),
        "reference_risk_time": None,
        "reference_nearest_link": None,
    }


def _make_high_instance(model, trajectory, index: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    link = cfg.BODY_LINKS[index % len(cfg.BODY_LINKS)]
    points = model.surface_by_link(trajectory.evaluate(0.0), cfg.SURFACE_DENSITY_TRUTH, {link})[link]
    point = points[int(rng.integers(0, len(points)))]
    center = point + rng.normal(0.0, 0.004, size=3)
    velocity = rng.uniform(-0.005, 0.005, size=3)
    radius = float(rng.uniform(0.045, 0.055))
    rows = _reference_distance_rows(model, trajectory, center, velocity, radius)
    obs_center0, obs_velocity, obs_radius = _observed_from_gt(
        np.random.default_rng(seed + 100_000), center, velocity, radius
    )
    return {
        "instance_id": f"D4_{index:02d}",
        "scenario_type": "initial_high_risk",
        "speed_group": None,
        "repeat_index": int(index),
        "seed": int(seed),
        "target_region": link,
        "gt_center0": center.tolist(),
        "gt_velocity": velocity.tolist(),
        "motion_start_time": 0.0,
        "gt_radius": radius,
        "observed_center0": obs_center0.tolist(),
        "observed_velocity": obs_velocity.tolist(),
        "observed_radius": obs_radius,
        "observation_seed": int(seed + 100_000),
        "reference_initial_distance": float(rows[0]["distance"]),
        "reference_min_distance": float(min(row["distance"] for row in rows)),
        "reference_risk_time": 0.0,
        "reference_nearest_link": link,
    }


def generate_instances(model, trajectory, output_dir: Path, *, smoke: bool = False) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    instances: list[dict[str, Any]] = []
    d1_per_speed = 1 if smoke else cfg.D1_MAIN_INSTANCES_PER_SPEED
    d2m_per_speed = 1 if smoke else cfg.D2_MAIN_INSTANCES_PER_SPEED
    d2s_per_speed = 1 if smoke else cfg.D2_STRESS_INSTANCES_PER_SPEED
    calibration = 2 if smoke else cfg.CALIBRATION_TRIALS
    for scenario_type, per_speed, offset in [
        ("D1", d1_per_speed, 1000),
        ("D2M", d2m_per_speed, 2000),
        ("D2S", d2s_per_speed, 5000),
    ]:
        index = 0
        for speed in cfg.SPEED_GROUPS:
            for repeat in range(per_speed):
                seed = cfg.RANDOM_SEED + offset + index * 37
                instance = _make_crossing_instance(
                    model,
                    trajectory,
                    scenario_type=scenario_type,
                    instance_index=index,
                    speed=speed,
                    repeat_index=repeat,
                    seed=seed,
                )
                instances.append(instance)
                index += 1
    for index in range(calibration):
        instances.append(_make_far_instance(model, trajectory, index, cfg.RANDOM_SEED + 3000 + index * 37))
    for index in range(calibration):
        instances.append(_make_high_instance(model, trajectory, index, cfg.RANDOM_SEED + 4000 + index * 37))
    for item in instances:
        write_json(output_dir / f"{item['instance_id']}.json", item)
    return instances
