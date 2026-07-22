"""Whole-body coverage helpers for revised Chapter 6.2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import config_62 as cfg
from .common_62 import min_distance_to_sphere, outward_direction


@dataclass(frozen=True)
class CriticalPoint:
    name: str
    link: str
    position: np.ndarray
    radius: float


@dataclass(frozen=True)
class CriticalDistance:
    distance: float
    nearest_point: str | None
    nearest_link: str | None


def critical_point_distance(
    points: list[CriticalPoint],
    obstacle_center: np.ndarray,
    obstacle_radius: float,
) -> CriticalDistance:
    best = math.inf
    best_name = None
    best_link = None
    for point in points:
        distance = float(np.linalg.norm(point.position - obstacle_center) - point.radius - obstacle_radius)
        if distance < best:
            best = distance
            best_name = point.name
            best_link = point.link
    return CriticalDistance(best, best_name, best_link)


def risk_label(distance: float, *, threshold: float = cfg.BODY_RISK_DISTANCE) -> bool:
    return bool(float(distance) <= float(threshold))


def body_clearance_interval(risk: bool) -> tuple[float, float]:
    if risk:
        return (0.4 * cfg.BODY_RISK_DISTANCE, 0.9 * cfg.BODY_RISK_DISTANCE)
    return (1.1 * cfg.BODY_RISK_DISTANCE, 1.5 * cfg.BODY_RISK_DISTANCE)


def body_clearance_is_valid(distance: float, risk: bool) -> bool:
    low, high = body_clearance_interval(risk)
    return bool(low <= float(distance) <= high)


def build_critical_points(surface, q: np.ndarray) -> list[CriticalPoint]:
    out: list[CriticalPoint] = []
    for region, links in cfg.BODY_REGIONS.items():
        radius = float(cfg.CRITICAL_POINT_RADII[region])
        for link in links:
            if link not in surface.link_names:
                continue
            points = surface.surface_by_link(q, density="coarse", links={link})[link]
            if len(points) == 0:
                continue
            centroid = points.mean(axis=0)
            distances = np.linalg.norm(points - centroid[None, :], axis=1)
            first = int(np.argmax(distances))
            selected = [first]
            if cfg.CRITICAL_POINTS_PER_REGION > 1:
                second = int(np.argmax(np.linalg.norm(points - points[first][None, :], axis=1)))
                selected.append(second)
            for local_index, point_index in enumerate(selected[: cfg.CRITICAL_POINTS_PER_REGION]):
                out.append(
                    CriticalPoint(
                        name=f"{region}_{link}_{local_index}",
                        link=link,
                        position=points[point_index].copy(),
                        radius=radius,
                    )
                )
    return out


def choose_region_surface_point(surface, q: np.ndarray, region: str, rng: np.random.Generator) -> tuple[np.ndarray, str, np.ndarray]:
    links = [link for link in cfg.BODY_REGIONS[region] if link in surface.link_names]
    if not links:
        raise KeyError(f"no available links for region {region}")
    link = str(rng.choice(links))
    points = surface.surface_by_link(q, density="dense", links={link})[link]
    point = points[int(rng.integers(0, len(points)))]
    direction = outward_direction(surface.surface(q, density="coarse"), point, rng)
    return point, link, direction


def make_body_sample(surface, q: np.ndarray, config_id: str, region: str, sample_id: int, risk: bool, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    for attempt in range(200):
        point, link, direction = choose_region_surface_point(surface, q, region, rng)
        radius = float(rng.uniform(*cfg.OBSTACLE_RADIUS_RANGE))
        low, high = body_clearance_interval(risk)
        clearance = float(rng.uniform(low, high))
        center = point + direction * (radius + clearance)
        d_gt, nearest_link, nearest_point = min_distance_to_sphere(surface, q, center, radius, density="dense")
        target_links = set(cfg.BODY_REGIONS[region])
        if nearest_link in target_links and body_clearance_is_valid(d_gt, risk):
            return {
                "configuration_id": config_id,
                "region": region,
                "target_link": link,
                "sample_id": sample_id,
                "risk_gt": bool(risk),
                "q": q.tolist(),
                "obstacle_center": center.tolist(),
                "obstacle_radius": radius,
                "D_gt": float(d_gt),
                "nearest_link_gt": nearest_link,
                "nearest_surface_point": None if nearest_point is None else nearest_point.tolist(),
                "seed": seed,
                "attempt": attempt,
            }
    raise RuntimeError(f"failed to generate body sample for {config_id}/{region}/{sample_id}")


def evaluate_body_sample(surface, sample: dict[str, Any]) -> dict[str, Any]:
    q = np.asarray(sample["q"], dtype=float)
    center = np.asarray(sample["obstacle_center"], dtype=float)
    radius = float(sample["obstacle_radius"])
    critical_result = evaluate_critical_sample(surface, q, center, radius)
    ccro_result = evaluate_ccro_sample(surface, q, center, radius)
    return {
        **sample,
        **critical_result,
        **ccro_result,
    }


def evaluate_critical_sample(surface, q: np.ndarray, center: np.ndarray, radius: float) -> dict[str, Any]:
    critical_points = build_critical_points(surface, q)
    critical = critical_point_distance(critical_points, center, radius)
    return {
        "D_critical": float(critical.distance),
        "risk_critical": risk_label(critical.distance),
        "nearest_link_critical": critical.nearest_link,
        "nearest_point_critical": critical.nearest_point,
        "critical_point_count": len(critical_points),
    }


def evaluate_ccro_sample(surface, q: np.ndarray, center: np.ndarray, radius: float) -> dict[str, Any]:
    d_ccro, link_ccro, nearest = min_distance_to_sphere(surface, q, center, radius, density="medium")
    return {
        "D_ccro": float(d_ccro),
        "risk_ccro": risk_label(d_ccro),
        "nearest_link_ccro": link_ccro,
        "nearest_surface_point_ccro": None if nearest is None else nearest.tolist(),
    }
