"""Dynamic occupancy methods for revised Chapter 6.2."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from perception.geometry_fit import make_occupancy_object
from perception.occupancy_tracker import OccupancyTracker

from . import config_62 as cfg
from .common_62 import min_distance_to_sphere


METHODS = ("current_frame", "octomap_like", "stro")
METHOD_NAMES = {
    "current_frame": "Current-frame",
    "octomap_like": "OctoMap-like",
    "stro": "STRO",
}


@dataclass
class ObservationState:
    center: np.ndarray
    radius: float
    velocity: np.ndarray
    age: int
    timestamp: float
    point_count: int


@dataclass
class DynamicMethodState:
    history: list[ObservationState] = field(default_factory=list)

    def append(self, observation: ObservationState) -> None:
        self.history.append(observation)
        self.history = self.history[-20:]


def observe_object(points: np.ndarray, timestamp: float, tracker: OccupancyTracker) -> ObservationState | None:
    if len(points) < 8:
        return None
    obj = make_occupancy_object(points, timestamp, margin=0.0)
    tracked = tracker.update([obj], timestamp)
    if not tracked:
        return None
    item = tracked[0]
    return ObservationState(
        center=item.center.copy(),
        radius=float(item.radius),
        velocity=item.velocity.copy(),
        age=int(item.age),
        timestamp=float(timestamp),
        point_count=int(item.point_count),
    )


def _distance_for_future(
    surface,
    q_future: np.ndarray,
    taus: np.ndarray,
    centers: list[np.ndarray],
    radii: list[float],
) -> tuple[float, str | None, float | None, float | None]:
    best = math.inf
    best_link = None
    best_tau = None
    best_radius = None
    for q_index, q in enumerate(q_future):
        tau = float(taus[min(q_index, len(taus) - 1)]) if len(taus) else 0.0
        for center_index, (center, radius) in enumerate(zip(centers, radii)):
            distance, link, _ = min_distance_to_sphere(surface, q, center, radius, density="medium")
            if distance < best:
                best = distance
                best_link = link
                best_tau = tau if len(centers) == len(taus) else None
                best_radius = float(radius)
    return best, best_link, best_tau, best_radius


def evaluate_method(
    method: str,
    surface,
    q_future: np.ndarray,
    taus: np.ndarray,
    state: DynamicMethodState,
    observation: ObservationState | None,
    *,
    alarm_distance: float = cfg.DYNAMIC_ALARM_DISTANCE,
) -> dict[str, Any]:
    if observation is not None:
        state.append(observation)
    if observation is None and not state.history:
        return {"risk": False, "distance": math.inf, "nearest_link": None, "prediction_center_tau_05": None}
    current = observation or state.history[-1]

    centers: list[np.ndarray] = []
    radii: list[float] = []
    if method == "current_frame":
        centers = [current.center.copy() for _ in taus]
        radii = [current.radius for _ in taus]
    elif method == "octomap_like":
        recent = state.history[-8:]
        centers = [item.center.copy() for item in recent]
        radii = [item.radius for item in recent]
    elif method == "stro":
        for tau in taus:
            tau_value = float(tau)
            centers.append(current.center + current.velocity * tau_value)
            speed = float(np.linalg.norm(current.velocity))
            radii.append(current.radius + 0.015 + 0.03 * tau_value + 0.20 * speed * tau_value)
    else:
        raise ValueError(f"unknown method: {method}")

    distance, link, min_tau, min_radius = _distance_for_future(surface, q_future, taus, centers, radii)
    tau_05 = min(0.5, float(taus[-1])) if len(taus) else 0.0
    if method == "stro":
        prediction = current.center + current.velocity * tau_05
    else:
        prediction = current.center.copy()
    return {
        "risk": bool(distance <= alarm_distance),
        "distance": float(distance),
        "nearest_link": link,
        "alarm_tau": min_tau,
        "inflated_radius_at_min": min_radius,
        "history_count": len(state.history),
        "prediction_center_tau_05": prediction.tolist(),
    }
