"""Static obstacle risk for configuration-coupled robot mesh surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree

from .nubs_trajectory import NUBSTrajectory6D
from .robot_surface_model import RobotSurfaceModel


@dataclass(frozen=True)
class StaticObstacleField:
    points: np.ndarray
    tree: cKDTree | None

    @classmethod
    def from_points(cls, points: np.ndarray) -> "StaticObstacleField":
        values = np.ascontiguousarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("obstacle points must have shape (N, 3)")
        if not np.all(np.isfinite(values)):
            raise ValueError("obstacle points contain NaN or Inf")
        return cls(values, None if len(values) == 0 else cKDTree(values))


@dataclass
class ConfigurationRisk:
    cost: float
    min_distance: float
    nearest_link: str | None
    robot_point: np.ndarray | None
    obstacle_point: np.ndarray | None
    per_link_cost: dict[str, float]
    gradient_q: np.ndarray | None = None


@dataclass
class TrajectoryRisk:
    cost: float
    min_distance: float
    nearest_link: str | None
    active_sample_count: int
    per_link_cost: dict[str, float]
    sample_times: np.ndarray
    sample_costs: np.ndarray
    sample_distances: np.ndarray
    gradient_q: np.ndarray | None = None


def trapezoid_weights(times: np.ndarray) -> np.ndarray:
    values = np.asarray(times, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or np.any(np.diff(values) <= 0.0):
        raise ValueError("times must be a strictly increasing vector with at least two values")
    weights = np.empty_like(values)
    weights[0] = 0.5 * (values[1] - values[0])
    weights[-1] = 0.5 * (values[-1] - values[-2])
    weights[1:-1] = 0.5 * (values[2:] - values[:-2])
    return weights


class MeshRiskEvaluator:
    """Equal-link-weighted squared-hinge risk over mesh surface points."""

    def __init__(
        self,
        surface_model: RobotSurfaceModel,
        *,
        d_safe: float = 0.15,
        d_activate: float = 0.20,
        fd_epsilon_q: float = 1.0e-4,
        density: str = "medium",
        link_weights: dict[str, float] | None = None,
    ) -> None:
        if not (0.0 < d_safe <= d_activate):
            raise ValueError("expected 0 < d_safe <= d_activate")
        if fd_epsilon_q <= 0.0:
            raise ValueError("fd_epsilon_q must be positive")
        self.surface_model = surface_model
        self.d_safe = float(d_safe)
        self.d_activate = float(d_activate)
        self.fd_epsilon_q = float(fd_epsilon_q)
        self.density = density
        self.link_weights = dict(link_weights or {})

    def _evaluate_no_gradient(
        self,
        q: np.ndarray,
        obstacle: StaticObstacleField,
        links: set[str] | None,
        density: str | None,
    ) -> ConfigurationRisk:
        if obstacle.tree is None:
            return ConfigurationRisk(0.0, math.inf, None, None, None, {})
        surfaces = self.surface_model.surface_by_link(
            q, density=density or self.density, links=links
        )
        if not surfaces:
            return ConfigurationRisk(0.0, math.inf, None, None, None, {})
        per_link: dict[str, float] = {}
        weighted_cost = 0.0
        weight_sum = 0.0
        min_distance = math.inf
        nearest_link: str | None = None
        nearest_robot: np.ndarray | None = None
        nearest_obstacle: np.ndarray | None = None
        for link, points in surfaces.items():
            distances, indices = obstacle.tree.query(points, k=1)
            hinge = np.maximum(self.d_safe - distances, 0.0)
            link_cost = float(np.mean(hinge * hinge))
            per_link[link] = link_cost
            weight = float(self.link_weights.get(link, 1.0))
            if weight < 0.0 or not np.isfinite(weight):
                raise ValueError(f"invalid link weight for {link}: {weight}")
            weighted_cost += weight * link_cost
            weight_sum += weight
            local_index = int(np.argmin(distances))
            local_distance = float(distances[local_index])
            if local_distance < min_distance:
                min_distance = local_distance
                nearest_link = link
                nearest_robot = points[local_index].copy()
                nearest_obstacle = obstacle.points[int(indices[local_index])].copy()
        cost = 0.0 if weight_sum <= 0.0 else weighted_cost / weight_sum
        return ConfigurationRisk(
            cost=cost,
            min_distance=min_distance,
            nearest_link=nearest_link,
            robot_point=nearest_robot,
            obstacle_point=nearest_obstacle,
            per_link_cost=per_link,
        )

    def configuration(
        self,
        q: np.ndarray,
        obstacle: StaticObstacleField,
        *,
        links: set[str] | None = None,
        density: str | None = None,
        with_gradient: bool = False,
    ) -> ConfigurationRisk:
        values = np.asarray(q, dtype=np.float64)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("q must be a finite array with shape (6,)")
        result = self._evaluate_no_gradient(values, obstacle, links, density)
        if not with_gradient:
            return result
        gradient = np.zeros(6, dtype=np.float64)
        if result.cost > 0.0 and result.min_distance < self.d_activate:
            for joint in range(6):
                plus = values.copy()
                minus = values.copy()
                plus[joint] += self.fd_epsilon_q
                minus[joint] -= self.fd_epsilon_q
                cost_plus = self._evaluate_no_gradient(
                    plus, obstacle, links, density
                ).cost
                cost_minus = self._evaluate_no_gradient(
                    minus, obstacle, links, density
                ).cost
                gradient[joint] = (cost_plus - cost_minus) / (
                    2.0 * self.fd_epsilon_q
                )
        result.gradient_q = gradient
        return result

    def trajectory(
        self,
        trajectory: NUBSTrajectory6D,
        obstacle: StaticObstacleField,
        sample_times: np.ndarray,
        *,
        links: set[str] | None = None,
        density: str | None = None,
        with_gradient: bool = False,
    ) -> TrajectoryRisk:
        times = np.asarray(sample_times, dtype=np.float64)
        weights = trapezoid_weights(times)
        q_samples = trajectory.sample(times, max_derivative=0).q
        costs = np.zeros(len(times), dtype=np.float64)
        distances = np.full(len(times), math.inf, dtype=np.float64)
        gradients = np.zeros((len(times), 6), dtype=np.float64) if with_gradient else None
        per_link_integral: dict[str, float] = {}
        nearest_link: str | None = None
        min_distance = math.inf
        active = 0
        for index, q in enumerate(q_samples):
            risk = self.configuration(
                q,
                obstacle,
                links=links,
                density=density,
                with_gradient=with_gradient,
            )
            costs[index] = risk.cost
            distances[index] = risk.min_distance
            active += int(risk.cost > 0.0)
            if risk.min_distance < min_distance:
                min_distance = risk.min_distance
                nearest_link = risk.nearest_link
            if gradients is not None and risk.gradient_q is not None:
                gradients[index] = weights[index] * risk.gradient_q
            for link, link_cost in risk.per_link_cost.items():
                per_link_integral[link] = per_link_integral.get(link, 0.0) + (
                    weights[index] * link_cost
                )
        return TrajectoryRisk(
            cost=float(np.dot(weights, costs)),
            min_distance=float(min_distance),
            nearest_link=nearest_link,
            active_sample_count=active,
            per_link_cost=per_link_integral,
            sample_times=times.copy(),
            sample_costs=costs,
            sample_distances=distances,
            gradient_q=gradients,
        )
