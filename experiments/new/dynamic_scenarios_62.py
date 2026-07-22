"""Dynamic scenario generation for revised Chapter 6.2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config_62 as cfg
from .common_62 import ReferenceTrajectory, outward_direction, perpendicular_direction


@dataclass(frozen=True)
class MotionSchedule:
    start_time: float
    end_time: float
    path_length: float
    speed: float


@dataclass(frozen=True)
class DynamicScenario:
    scene: str
    seed: int
    speed: float
    crossing_time: float
    schedule: MotionSchedule
    target_link: str
    target_point: np.ndarray
    direction: np.ndarray
    crossing_center: np.ndarray
    obstacle_radius: float
    clearance: float

    def center_at(self, t: float) -> np.ndarray:
        if self.scene == "static_safe":
            return self.crossing_center.copy()
        if t <= self.schedule.start_time:
            return self.crossing_center - self.direction * self.schedule.path_length
        if self.scene == "approach" and t > self.schedule.end_time:
            return self.crossing_center - self.direction * self.speed * (float(t) - self.schedule.end_time)
        elapsed = float(t) - self.schedule.start_time
        return self.crossing_center - self.direction * self.schedule.path_length + self.direction * self.speed * max(elapsed, 0.0)

    def velocity_at(self, t: float) -> np.ndarray:
        if self.scene == "static_safe":
            return np.zeros(3)
        if self.scene == "approach" and t > self.schedule.end_time:
            return -self.direction * self.speed
        if self.schedule.start_time <= t <= cfg.TOTAL_DURATION:
            return self.direction * self.speed
        return np.zeros(3)


def compute_motion_schedule(speed: float, crossing_time: float, path_length: float) -> MotionSchedule:
    if speed <= 0.0:
        raise ValueError("speed must be positive")
    if path_length <= 0.0:
        raise ValueError("path_length must be positive")
    start = float(crossing_time) - float(path_length) / float(speed)
    if start < 0.0:
        raise ValueError("motion would start before time zero")
    return MotionSchedule(start_time=start, end_time=float(crossing_time), path_length=float(path_length), speed=float(speed))


def sample_feasible_motion_schedule(rng: np.random.Generator, speed: float) -> MotionSchedule:
    min_length, max_length = cfg.VISIBLE_PATH_LENGTH_RANGE
    tc_min, tc_max = cfg.CROSSING_TIME_RANGE
    for _ in range(100):
        path_length = float(rng.uniform(min_length, max_length))
        lower_tc = max(tc_min, path_length / float(speed))
        if lower_tc <= tc_max:
            crossing_time = float(rng.uniform(lower_tc, tc_max))
            return compute_motion_schedule(speed, crossing_time, path_length)
    path_length = min(max_length, float(speed) * tc_max)
    return compute_motion_schedule(speed, tc_max, path_length)


def choose_target(surface, trajectory: ReferenceTrajectory, rng: np.random.Generator, scene: str, crossing_time: float) -> tuple[np.ndarray, str, np.ndarray]:
    q = trajectory.sample(crossing_time)
    links = ("foreArm_Link", "wrist1_Link") if scene == "crossing" else ("foreArm_Link",)
    link = str(rng.choice(links))
    points = surface.surface_by_link(q, density="dense", links={link})[link]
    point = points[int(rng.integers(0, len(points)))]
    robot_points = surface.surface(q, density="coarse")
    normal = outward_direction(robot_points, point, rng)
    return point, link, normal


def make_dynamic_scenario(surface, trajectory: ReferenceTrajectory, scene: str, speed: float, seed: int) -> DynamicScenario:
    rng = np.random.default_rng(seed)
    schedule = sample_feasible_motion_schedule(rng, speed)
    crossing_time = schedule.end_time
    radius = float(rng.uniform(*cfg.OBSTACLE_RADIUS_RANGE))
    clearance = float(rng.uniform(*cfg.MIN_SCENE_CLEARANCE_RANGE))
    target, link, normal = choose_target(surface, trajectory, rng, scene, crossing_time)
    if scene == "approach":
        direction = -normal
    elif scene == "crossing":
        direction = perpendicular_direction(normal, rng)
    else:
        raise ValueError(f"unsupported dynamic scene: {scene}")
    crossing_center = target + normal * (radius + clearance)
    return DynamicScenario(
        scene=scene,
        seed=seed,
        speed=float(speed),
        crossing_time=crossing_time,
        schedule=schedule,
        target_link=link,
        target_point=target,
        direction=direction,
        crossing_center=crossing_center,
        obstacle_radius=radius,
        clearance=clearance,
    )


def make_static_safe_scenario(seed: int) -> DynamicScenario:
    rng = np.random.default_rng(seed)
    center = np.array([0.35, 0.70, 0.55]) + rng.normal(scale=0.02, size=3)
    return DynamicScenario(
        scene="static_safe",
        seed=seed,
        speed=0.0,
        crossing_time=4.0,
        schedule=MotionSchedule(0.0, cfg.TOTAL_DURATION, 0.0, 1.0),
        target_link="workspace",
        target_point=center,
        direction=np.zeros(3),
        crossing_center=center,
        obstacle_radius=0.05,
        clearance=0.25,
    )


def make_leave_scenario(surface, trajectory: ReferenceTrajectory, seed: int) -> DynamicScenario:
    rng = np.random.default_rng(seed)
    crossing_time = 1.5
    target, link, normal = choose_target(surface, trajectory, rng, "approach", crossing_time)
    radius = float(rng.uniform(*cfg.OBSTACLE_RADIUS_RANGE))
    clearance = 0.16
    crossing_center = target + normal * (radius + clearance)
    return DynamicScenario(
        scene="leave",
        seed=seed,
        speed=0.16,
        crossing_time=crossing_time,
        schedule=MotionSchedule(0.0, cfg.TOTAL_DURATION, cfg.TOTAL_DURATION * 0.16, 0.16),
        target_link=link,
        target_point=target,
        direction=normal,
        crossing_center=crossing_center,
        obstacle_radius=radius,
        clearance=clearance,
    )
