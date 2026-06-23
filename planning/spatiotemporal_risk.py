"""Configuration-coupled mesh risk against time-varying sphere occupancy."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .mesh_risk import trapezoid_weights
from .nubs_trajectory import NUBSTrajectory6D
from .obstacle_forecast import ObstacleForecast, ObstacleOccupancy
from .robot_surface_model import RobotSurfaceModel


@dataclass
class DynamicConfigurationRisk:
    cost: float
    min_distance: float
    nearest_link: str | None
    nearest_object_id: int | None
    robot_point: np.ndarray | None
    obstacle_point: np.ndarray | None
    per_link_cost: dict[str, float]
    extrapolated: bool
    gradient_q: np.ndarray | None = None


@dataclass
class SpatioTemporalTrajectoryRisk:
    cost: float
    min_distance: float
    nearest_link: str | None
    nearest_object_id: int | None
    active_sample_count: int
    extrapolated_sample_count: int
    per_link_cost: dict[str, float]
    sample_times: np.ndarray
    sample_costs: np.ndarray
    sample_distances: np.ndarray
    gradient_q: np.ndarray | None = None


class SpatioTemporalRiskEvaluator:
    def __init__(
        self,
        surface_model: RobotSurfaceModel,
        *,
        d_safe: float,
        d_activate: float,
        fd_epsilon_q: float = 2.0e-4,
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

    @staticmethod
    def _surface_to_occupancy_distances(
        points: np.ndarray, occupancy: ObstacleOccupancy
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not occupancy.spheres:
            return (
                np.full(len(points), math.inf),
                np.full(len(points), -1, dtype=np.int64),
                np.full(len(points), -1, dtype=np.int64),
            )
        all_distances = []
        for sphere in occupancy.spheres:
            radial = np.linalg.norm(points - sphere.center[None, :], axis=1)
            all_distances.append(np.maximum(radial - sphere.radius, 0.0))
        matrix = np.column_stack(all_distances)
        sphere_indices = np.argmin(matrix, axis=1)
        distances = matrix[np.arange(len(points)), sphere_indices]
        object_ids = np.asarray(
            [occupancy.spheres[index].object_id for index in sphere_indices],
            dtype=np.int64,
        )
        return distances, sphere_indices, object_ids

    def _evaluate_no_gradient(
        self,
        q: np.ndarray,
        occupancy: ObstacleOccupancy,
        links: set[str] | None,
        density: str | None,
    ) -> DynamicConfigurationRisk:
        surfaces = self.surface_model.surface_by_link(
            q, density=density or self.density, links=links
        )
        if not surfaces or not occupancy.spheres:
            return DynamicConfigurationRisk(
                0.0, math.inf, None, None, None, None, {}, occupancy.extrapolated
            )
        per_link: dict[str, float] = {}
        total_cost = 0.0
        total_weight = 0.0
        min_distance = math.inf
        nearest_link = None
        nearest_object_id = None
        nearest_robot = None
        nearest_obstacle = None
        for link, points in surfaces.items():
            distances, sphere_indices, object_ids = self._surface_to_occupancy_distances(
                points, occupancy
            )
            hinge = np.maximum(self.d_safe - distances, 0.0)
            link_cost = float(np.mean(hinge * hinge))
            per_link[link] = link_cost
            weight = float(self.link_weights.get(link, 1.0))
            if weight < 0.0 or not np.isfinite(weight):
                raise ValueError(f"invalid link weight for {link}")
            total_cost += weight * link_cost
            total_weight += weight
            index = int(np.argmin(distances))
            if float(distances[index]) < min_distance:
                min_distance = float(distances[index])
                nearest_link = link
                nearest_object_id = int(object_ids[index])
                nearest_robot = points[index].copy()
                sphere = occupancy.spheres[int(sphere_indices[index])]
                direction = points[index] - sphere.center
                norm = float(np.linalg.norm(direction))
                if norm < 1.0e-12:
                    direction = np.array([1.0, 0.0, 0.0])
                else:
                    direction = direction / norm
                nearest_obstacle = sphere.center + sphere.radius * direction
        return DynamicConfigurationRisk(
            cost=0.0 if total_weight <= 0.0 else total_cost / total_weight,
            min_distance=min_distance,
            nearest_link=nearest_link,
            nearest_object_id=nearest_object_id,
            robot_point=nearest_robot,
            obstacle_point=nearest_obstacle,
            per_link_cost=per_link,
            extrapolated=occupancy.extrapolated,
        )

    def configuration(
        self,
        q: np.ndarray,
        forecast: ObstacleForecast,
        tau: float,
        *,
        links: set[str] | None = None,
        density: str | None = None,
        with_gradient: bool = False,
    ) -> DynamicConfigurationRisk:
        values = np.asarray(q, dtype=np.float64)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("q must be a finite array with shape (6,)")
        occupancy = forecast.occupancy_at(float(tau))
        result = self._evaluate_no_gradient(values, occupancy, links, density)
        if not with_gradient:
            return result
        gradient = np.zeros(6, dtype=np.float64)
        if result.cost > 0.0 and result.min_distance < self.d_activate:
            for joint in range(6):
                plus = values.copy()
                minus = values.copy()
                plus[joint] += self.fd_epsilon_q
                minus[joint] -= self.fd_epsilon_q
                plus_cost = self._evaluate_no_gradient(
                    plus, occupancy, links, density
                ).cost
                minus_cost = self._evaluate_no_gradient(
                    minus, occupancy, links, density
                ).cost
                gradient[joint] = (plus_cost - minus_cost) / (
                    2.0 * self.fd_epsilon_q
                )
        result.gradient_q = gradient
        return result

    def trajectory(
        self,
        trajectory: NUBSTrajectory6D,
        forecast: ObstacleForecast,
        sample_times: np.ndarray,
        *,
        links: set[str] | None = None,
        density: str | None = None,
        with_gradient: bool = False,
    ) -> SpatioTemporalTrajectoryRisk:
        times = np.asarray(sample_times, dtype=np.float64)
        if times[-1] > forecast.valid_horizon + 1.0e-12:
            # Forecasts with an explicit hold/inflate policy may still answer.
            forecast.occupancy_at(float(times[-1]))
        weights = trapezoid_weights(times)
        q_samples = trajectory.sample(times, max_derivative=0).q
        costs = np.zeros(len(times))
        distances = np.full(len(times), math.inf)
        gradients = np.zeros((len(times), 6)) if with_gradient else None
        per_link: dict[str, float] = {}
        min_distance = math.inf
        nearest_link = None
        nearest_object_id = None
        active = 0
        extrapolated = 0
        for index, (tau, q) in enumerate(zip(times, q_samples)):
            risk = self.configuration(
                q,
                forecast,
                float(tau),
                links=links,
                density=density,
                with_gradient=with_gradient,
            )
            costs[index] = risk.cost
            distances[index] = risk.min_distance
            active += int(risk.cost > 0.0)
            extrapolated += int(risk.extrapolated)
            if risk.min_distance < min_distance:
                min_distance = risk.min_distance
                nearest_link = risk.nearest_link
                nearest_object_id = risk.nearest_object_id
            if gradients is not None and risk.gradient_q is not None:
                gradients[index] = weights[index] * risk.gradient_q
            for link, value in risk.per_link_cost.items():
                per_link[link] = per_link.get(link, 0.0) + weights[index] * value
        return SpatioTemporalTrajectoryRisk(
            cost=float(np.dot(weights, costs)),
            min_distance=float(min_distance),
            nearest_link=nearest_link,
            nearest_object_id=nearest_object_id,
            active_sample_count=active,
            extrapolated_sample_count=extrapolated,
            per_link_cost=per_link,
            sample_times=times.copy(),
            sample_costs=costs,
            sample_distances=distances,
            gradient_q=gradients,
        )
