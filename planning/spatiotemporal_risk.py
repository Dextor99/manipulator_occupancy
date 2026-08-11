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
        topk_clearance_points: int = 0,
        softmin_beta: float = 60.0,
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
        self.topk_clearance_points = int(max(0, topk_clearance_points))
        self.softmin_beta = float(softmin_beta)
        if not np.isfinite(self.softmin_beta) or self.softmin_beta <= 0.0:
            raise ValueError("softmin_beta must be positive")

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
            all_distances.append(radial - sphere.radius)
        matrix = np.column_stack(all_distances)
        sphere_indices = np.argmin(matrix, axis=1)
        distances = matrix[np.arange(len(points)), sphere_indices]
        # Map the nearest-sphere indices in native NumPy code.  This method is
        # called for every link at every verifier sample; a Python loop over
        # every surface point made the otherwise identical medium verification
        # unnecessarily expensive.
        sphere_object_ids = np.fromiter(
            (sphere.object_id for sphere in occupancy.spheres),
            dtype=np.int64,
            count=len(occupancy.spheres),
        )
        object_ids = sphere_object_ids[sphere_indices]
        return distances, sphere_indices, object_ids

    def _link_clearance_cost(self, distances: np.ndarray) -> float:
        if len(distances) == 0:
            return 0.0
        if self.topk_clearance_points <= 0 or len(distances) <= self.topk_clearance_points:
            hinge = np.maximum(self.d_safe - distances, 0.0)
            return float(np.mean(hinge * hinge))
        k = min(self.topk_clearance_points, len(distances))
        active = np.partition(distances, k - 1)[:k]
        scaled = -self.softmin_beta * active
        offset = float(np.max(scaled))
        softmin = -(math.log(float(np.sum(np.exp(scaled - offset)))) + offset) / self.softmin_beta
        hinge = max(self.d_safe - softmin, 0.0)
        return float(hinge * hinge)

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
            link_cost = self._link_clearance_cost(distances)
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

    def configuration_clearance(
        self,
        q: np.ndarray,
        forecast: ObstacleForecast,
        tau: float,
        *,
        links: set[str] | None = None,
        density: str | None = None,
    ) -> DynamicConfigurationRisk:
        """Evaluate the exact minimum clearance without unused risk-cost data.

        Dense trajectory verification needs only minimum distance, nearest link,
        and forecast-horizon status.  Avoiding per-point object-ID construction,
        hinge costs, and nearest-contact reconstruction preserves the geometric
        query and sampling protocol while reducing authorization latency.
        """
        values = np.asarray(q, dtype=np.float64)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("q must be a finite array with shape (6,)")
        occupancy = forecast.occupancy_at(float(tau))
        surfaces = self.surface_model.surface_by_link(
            values, density=density or self.density, links=links
        )
        if not surfaces or not occupancy.spheres:
            return DynamicConfigurationRisk(
                0.0, math.inf, None, None, None, None, {}, occupancy.extrapolated
            )
        min_distance = math.inf
        nearest_link = None
        nearest_object_id = None
        for link, points in surfaces.items():
            for sphere in occupancy.spheres:
                distances = np.linalg.norm(points - sphere.center[None, :], axis=1) - sphere.radius
                index = int(np.argmin(distances))
                distance = float(distances[index])
                if distance < min_distance:
                    min_distance = distance
                    nearest_link = link
                    nearest_object_id = sphere.object_id
        return DynamicConfigurationRisk(
            cost=0.0,
            min_distance=min_distance,
            nearest_link=nearest_link,
            nearest_object_id=nearest_object_id,
            robot_point=None,
            obstacle_point=None,
            per_link_cost={},
            extrapolated=occupancy.extrapolated,
        )

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
