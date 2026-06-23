"""Time-queryable obstacle occupancy forecasts for CCRO-NUBS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ForecastSphere:
    object_id: int
    center: np.ndarray
    radius: float


@dataclass(frozen=True)
class ObstacleOccupancy:
    tau: float
    spheres: tuple[ForecastSphere, ...]
    extrapolated: bool = False


class ObstacleForecast(Protocol):
    @property
    def valid_horizon(self) -> float: ...

    def occupancy_at(self, tau: float) -> ObstacleOccupancy: ...


class ConstantVelocitySphereForecast:
    """Constant-velocity center prediction with time-growing radius."""

    def __init__(
        self,
        center: np.ndarray,
        velocity: np.ndarray,
        radius: float,
        valid_horizon: float,
        *,
        object_id: int = 1,
        margin: float = 0.0,
        uncertainty: float = 0.0,
        uncertainty_growth: float = 0.0,
        velocity_radius_scale: float = 0.0,
        beyond_horizon: str = "error",
    ) -> None:
        self.center = np.asarray(center, dtype=np.float64)
        self.velocity = np.asarray(velocity, dtype=np.float64)
        if self.center.shape != (3,) or self.velocity.shape != (3,):
            raise ValueError("center and velocity must have shape (3,)")
        if not np.all(np.isfinite(self.center)) or not np.all(np.isfinite(self.velocity)):
            raise ValueError("center and velocity must be finite")
        values = [radius, valid_horizon, margin, uncertainty, uncertainty_growth, velocity_radius_scale]
        if not all(np.isfinite(values)) or radius <= 0.0 or valid_horizon <= 0.0:
            raise ValueError("radius and valid_horizon must be positive and all values finite")
        if any(value < 0.0 for value in values[2:]):
            raise ValueError("margin and uncertainty parameters must be non-negative")
        if beyond_horizon not in {"error", "hold_inflate"}:
            raise ValueError("beyond_horizon must be error or hold_inflate")
        self.radius = float(radius)
        self._valid_horizon = float(valid_horizon)
        self.object_id = int(object_id)
        self.margin = float(margin)
        self.uncertainty = float(uncertainty)
        self.uncertainty_growth = float(uncertainty_growth)
        self.velocity_radius_scale = float(velocity_radius_scale)
        self.beyond_horizon = beyond_horizon

    @property
    def valid_horizon(self) -> float:
        return self._valid_horizon

    def occupancy_at(self, tau: float) -> ObstacleOccupancy:
        if not np.isfinite(tau) or tau < 0.0:
            raise ValueError("tau must be finite and non-negative")
        requested_tau = float(tau)
        extrapolated = requested_tau > self.valid_horizon + 1.0e-12
        if extrapolated and self.beyond_horizon == "error":
            raise ValueError(
                f"tau={requested_tau} exceeds valid_horizon={self.valid_horizon}"
            )
        center_tau = min(requested_tau, self.valid_horizon) if extrapolated else requested_tau
        center = self.center + self.velocity * center_tau
        speed = float(np.linalg.norm(self.velocity))
        sphere_radius = (
            self.radius
            + self.margin
            + self.uncertainty
            + self.uncertainty_growth * requested_tau
            + self.velocity_radius_scale * speed * requested_tau
        )
        return ObstacleOccupancy(
            tau=requested_tau,
            spheres=(ForecastSphere(self.object_id, center, float(sphere_radius)),),
            extrapolated=extrapolated,
        )


class FrozenSphereForecast:
    """Current-frame baseline: keep a forecast's tau=0 occupancy fixed."""

    def __init__(self, source: ObstacleForecast, valid_horizon: float | None = None):
        self.initial = source.occupancy_at(0.0)
        self._valid_horizon = float(valid_horizon or source.valid_horizon)

    @property
    def valid_horizon(self) -> float:
        return self._valid_horizon

    def occupancy_at(self, tau: float) -> ObstacleOccupancy:
        if not np.isfinite(tau) or tau < 0.0 or tau > self.valid_horizon + 1.0e-12:
            raise ValueError("tau is outside the frozen forecast horizon")
        spheres = tuple(
            ForecastSphere(s.object_id, s.center.copy(), s.radius)
            for s in self.initial.spheres
        )
        return ObstacleOccupancy(float(tau), spheres, False)


class CompositeForecast:
    def __init__(self, forecasts: list[ObstacleForecast] | tuple[ObstacleForecast, ...]):
        if not forecasts:
            raise ValueError("CompositeForecast requires at least one forecast")
        self.forecasts = tuple(forecasts)
        self._valid_horizon = min(float(item.valid_horizon) for item in self.forecasts)

    @property
    def valid_horizon(self) -> float:
        return self._valid_horizon

    def occupancy_at(self, tau: float) -> ObstacleOccupancy:
        occupancies = [item.occupancy_at(tau) for item in self.forecasts]
        spheres = tuple(sphere for occupancy in occupancies for sphere in occupancy.spheres)
        return ObstacleOccupancy(
            tau=float(tau),
            spheres=spheres,
            extrapolated=any(item.extrapolated for item in occupancies),
        )


class ShiftedForecast:
    """Expose a global forecast on a candidate trajectory's local clock.

    ``occupancy_at(local_tau)`` queries the source at
    ``time_offset + local_tau``.  This small adapter is what prevents every
    accepted replan from accidentally resetting moving obstacles to time zero.
    """

    def __init__(
        self,
        source: ObstacleForecast,
        time_offset: float,
        local_horizon: float | None = None,
    ) -> None:
        if not np.isfinite(time_offset) or time_offset < 0.0:
            raise ValueError("time_offset must be finite and non-negative")
        remaining = float(source.valid_horizon) - float(time_offset)
        if remaining <= 0.0:
            raise ValueError("time_offset leaves no valid forecast horizon")
        if local_horizon is not None:
            if not np.isfinite(local_horizon) or local_horizon <= 0.0:
                raise ValueError("local_horizon must be finite and positive")
            remaining = min(remaining, float(local_horizon))
        self.source = source
        self.time_offset = float(time_offset)
        self._valid_horizon = remaining

    @property
    def valid_horizon(self) -> float:
        return self._valid_horizon

    def occupancy_at(self, tau: float) -> ObstacleOccupancy:
        if not np.isfinite(tau) or tau < 0.0:
            raise ValueError("tau must be finite and non-negative")
        local_tau = float(tau)
        if local_tau > self.valid_horizon + 1.0e-12:
            raise ValueError(
                f"tau={local_tau} exceeds shifted horizon={self.valid_horizon}"
            )
        source_occupancy = self.source.occupancy_at(self.time_offset + local_tau)
        spheres = tuple(
            ForecastSphere(s.object_id, s.center.copy(), s.radius)
            for s in source_occupancy.spheres
        )
        return ObstacleOccupancy(
            tau=local_tau,
            spheres=spheres,
            extrapolated=source_occupancy.extrapolated,
        )
