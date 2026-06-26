"""Fast sphere-level risk approximation for online safety control.

This module is intentionally lighter than :mod:`planning.spatiotemporal_risk`.
It represents each robot link by a small set of local bounding spheres and
compares them against object-level forecast spheres.  The dense mesh evaluator
remains the offline/acceptance verifier; this evaluator is for velocity-level
control where p95 latency matters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .obstacle_forecast import ObstacleForecast, ObstacleOccupancy
from .robot_surface_model import RobotSurfaceModel
from .spatiotemporal_risk import DynamicConfigurationRisk


@dataclass(frozen=True)
class LinkSphereSet:
    centers_local: np.ndarray
    radii: np.ndarray


class FastSphereRiskEvaluator:
    """Object-level, low-latency configuration risk evaluator.

    The gradient is a finite difference of the same sphere-level cost.  This
    still uses FK perturbations, but avoids rebuilding full surfaces/KD-trees
    and uses only tens of spheres instead of thousands of mesh points.
    """

    def __init__(
        self,
        surface_model: RobotSurfaceModel,
        *,
        d_safe: float,
        d_activate: float,
        fd_epsilon_q: float = 2.0e-4,
        density: str = "coarse",
        max_spheres_per_link: int = 8,
        radius_padding: float = 0.004,
        link_weights: dict[str, float] | None = None,
    ) -> None:
        if not (0.0 < d_safe <= d_activate):
            raise ValueError("expected 0 < d_safe <= d_activate")
        if fd_epsilon_q <= 0.0:
            raise ValueError("fd_epsilon_q must be positive")
        if max_spheres_per_link <= 0:
            raise ValueError("max_spheres_per_link must be positive")
        self.surface_model = surface_model
        self.d_safe = float(d_safe)
        self.d_activate = float(d_activate)
        self.fd_epsilon_q = float(fd_epsilon_q)
        self.density = density
        self.max_spheres_per_link = int(max_spheres_per_link)
        self.radius_padding = float(radius_padding)
        self.link_weights = dict(link_weights or {})
        self._link_spheres = self._build_link_spheres()

    def _build_link_spheres(self) -> dict[str, LinkSphereSet]:
        out: dict[str, LinkSphereSet] = {}
        for link in self.surface_model.link_names:
            points = self.surface_model.local_samples(link, self.density)
            if len(points) == 0:
                continue
            count = min(self.max_spheres_per_link, len(points))
            order = np.argsort(points[:, 0])
            chunks = np.array_split(points[order], count)
            centers = []
            radii = []
            for chunk in chunks:
                if len(chunk) == 0:
                    continue
                center = np.mean(chunk, axis=0)
                radius = float(np.max(np.linalg.norm(chunk - center[None, :], axis=1)))
                centers.append(center)
                radii.append(radius + self.radius_padding)
            out[link] = LinkSphereSet(
                np.ascontiguousarray(centers, dtype=np.float64),
                np.asarray(radii, dtype=np.float64),
            )
        return out

    def _world_spheres(
        self, q: np.ndarray, links: set[str] | None = None
    ) -> dict[str, LinkSphereSet]:
        fk = self.surface_model.urdf.link_transforms(self.surface_model._joint_dict(q))
        selected = set(self._link_spheres) if links is None else set(links)
        out: dict[str, LinkSphereSet] = {}
        for link, spheres in self._link_spheres.items():
            if link not in selected:
                continue
            transform = fk.get(link)
            if transform is None:
                continue
            centers = spheres.centers_local @ transform[:3, :3].T + transform[:3, 3]
            out[link] = LinkSphereSet(centers, spheres.radii)
        return out

    def _evaluate_no_gradient(
        self,
        q: np.ndarray,
        occupancy: ObstacleOccupancy,
        links: set[str] | None = None,
    ) -> DynamicConfigurationRisk:
        if not occupancy.spheres:
            return DynamicConfigurationRisk(
                0.0, math.inf, None, None, None, None, {}, occupancy.extrapolated
            )
        link_sets = self._world_spheres(q, links)
        if not link_sets:
            return DynamicConfigurationRisk(
                0.0, math.inf, None, None, None, None, {}, occupancy.extrapolated
            )
        total_cost = 0.0
        total_weight = 0.0
        per_link: dict[str, float] = {}
        best_distance = math.inf
        best_link: str | None = None
        best_object_id: int | None = None
        best_robot: np.ndarray | None = None
        best_obstacle: np.ndarray | None = None

        for link, spheres in link_sets.items():
            min_distances = np.full(len(spheres.centers_local), math.inf)
            min_indices = np.zeros(len(spheres.centers_local), dtype=np.int64)
            for index, obstacle in enumerate(occupancy.spheres):
                radial = np.linalg.norm(
                    spheres.centers_local - obstacle.center[None, :], axis=1
                )
                signed = radial - spheres.radii - obstacle.radius
                distances = np.maximum(signed, 0.0)
                update = distances < min_distances
                min_distances[update] = distances[update]
                min_indices[update] = index

            hinge = np.maximum(self.d_safe - min_distances, 0.0)
            link_cost = float(np.mean(hinge * hinge))
            per_link[link] = link_cost
            weight = float(self.link_weights.get(link, 1.0))
            if weight < 0.0 or not np.isfinite(weight):
                raise ValueError(f"invalid link weight for {link}")
            total_cost += weight * link_cost
            total_weight += weight

            local_index = int(np.argmin(min_distances))
            local_distance = float(min_distances[local_index])
            if local_distance < best_distance:
                obstacle = occupancy.spheres[int(min_indices[local_index])]
                center = spheres.centers_local[local_index]
                direction = center - obstacle.center
                norm = float(np.linalg.norm(direction))
                if norm < 1.0e-12:
                    unit = np.array([1.0, 0.0, 0.0])
                else:
                    unit = direction / norm
                best_distance = local_distance
                best_link = link
                best_object_id = int(obstacle.object_id)
                best_robot = center - unit * spheres.radii[local_index]
                best_obstacle = obstacle.center + unit * obstacle.radius

        return DynamicConfigurationRisk(
            cost=0.0 if total_weight <= 0.0 else total_cost / total_weight,
            min_distance=best_distance,
            nearest_link=best_link,
            nearest_object_id=best_object_id,
            robot_point=best_robot,
            obstacle_point=best_obstacle,
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
        result = self._evaluate_no_gradient(values, occupancy, links)
        if not with_gradient:
            return result
        gradient = np.zeros(6, dtype=np.float64)
        if result.cost > 0.0 and result.min_distance < self.d_activate:
            for joint in range(6):
                plus = values.copy()
                minus = values.copy()
                plus[joint] += self.fd_epsilon_q
                minus[joint] -= self.fd_epsilon_q
                plus_cost = self._evaluate_no_gradient(plus, occupancy, links).cost
                minus_cost = self._evaluate_no_gradient(minus, occupancy, links).cost
                gradient[joint] = (plus_cost - minus_cost) / (2.0 * self.fd_epsilon_q)
        result.gradient_q = gradient
        return result
