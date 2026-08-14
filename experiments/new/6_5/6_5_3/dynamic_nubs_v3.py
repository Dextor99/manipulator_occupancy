"""V3 obstacle geometry, STRO prediction and planning policy helpers.

This module deliberately contains no robot commands.  It unifies the geometry
used by STRO and Fast/Fresh around an adaptive PCA 1--4 sphere cover, separates
prediction uncertainty from decision clearance, and turns the 0.11 m coarse
target and 3 mm improvement into ranking diagnostics rather than execution
gates.  Absolute Fresh clearance, motion limits and raw hard guard remain in
the established verifier/executor.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

import numpy as np

from planning.obstacle_forecast import CompositeForecast, ConstantVelocitySphereForecast
from risk.prediction import RiskSphere

trial = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")
config64 = importlib.import_module("experiments.new.6_4.config_64")
_BASE_ADAPTIVE_FIT = trial.fit_pca_multisphere


V3_PROTOCOL = {
    "protocol_id": "653_DYNAMIC_NUBS_CLOSED_LOOP_V3",
    "geometry": "adaptive_pca_1_to_4_spheres",
    "prediction_uncertainty": "u0_plus_kv_speed_tau",
    "prediction_margin_m": 0.0,
    "uncertainty_base_m": 0.020,
    "uncertainty_velocity_scale": 0.10,
    "trigger_distance_m": 0.14,
    "preferred_seed_clearance_m": 0.11,
    "preferred_seed_clearance_is_hard_gate": False,
    "fresh_execution_clearance_m": 0.09,
    "raw_hard_guard_m": 0.10,
    "clearance_improvement_is_hard_gate": False,
    "accepted_fast_step_is_hard_gate": False,
    "seed_motion_from_fast_reference_is_hard_gate": False,
    "task_progress_is_hard_gate": False,
    "candidate_contract": "verified_bypass_seed_or_fast_repaired_bypass",
    "fixed_x_required": False,
    "execution_forecast": "fresh_geometry_constant_velocity_no_legacy_inflation",
    "execution_forecast_margin_m": 0.0,
    "execution_forecast_uncertainty_m": 0.0,
    "execution_forecast_uncertainty_growth_m_s": 0.0,
    "execution_forecast_velocity_radius_scale_s": 0.0,
}


def adaptive_geometry_adapter(
    points: np.ndarray, *, fit_margin_m: float = 0.005, max_components: int = 4
) -> dict[str, Any]:
    return _BASE_ADAPTIVE_FIT(
        points, fit_margin_m=fit_margin_m, max_components=max_components
    )


def adaptive_multisphere_predictor(
    *,
    stable_objects: list[Any],
    prediction_tracks: list[Any],
    dynamic_audits: dict[int, dict[str, Any]],
    clusters: list[np.ndarray],
    args: Any,
    safety: dict[str, Any],
) -> list[RiskSphere]:
    """Build STRO samples from the same adaptive geometry used by Fresh/Fast.

    All associated stable tracks are represented.  A low-speed track uses a
    zero local velocity (quasi-static); it is not removed from collision
    prediction.  No legacy 35 mm prediction margin is added.  Each component
    receives only ``u0 + kv*|v|*tau`` model uncertainty.
    """
    del prediction_tracks, safety
    stable_by_id = {trial.object_track_id(obj): obj for obj in stable_objects}
    u0 = float(getattr(args, "prediction_uncertainty_m", 0.020))
    kv = 0.10
    step = float(args.prediction_step_s)
    horizon = float(args.prediction_horizon_s)
    taus = np.arange(step, horizon + 1.0e-9, step)
    predictions: list[RiskSphere] = []
    for track_id, audit in dynamic_audits.items():
        obj = stable_by_id.get(track_id)
        index = audit.get("associated_cluster_index")
        if obj is None or index is None or not (0 <= int(index) < len(clusters)):
            continue
        if not audit.get("checks", {}).get("age_ok", False):
            continue
        if not audit.get("checks", {}).get("association_ok", False):
            continue
        points = np.asarray(clusters[int(index)], dtype=np.float64)
        try:
            geometry = adaptive_geometry_adapter(
                points,
                fit_margin_m=float(getattr(args, "multisphere_fit_margin_m", 0.005)),
                max_components=int(getattr(args, "multisphere_max_components", 4)),
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        raw_center = np.mean(points, axis=0)
        tracked_center = np.asarray(audit["center"], dtype=np.float64)
        centers = np.asarray(geometry["component_centers"], dtype=np.float64)
        centers = centers + (tracked_center - raw_center)[None, :]
        radii = np.asarray(geometry["component_base_radii"], dtype=np.float64)
        velocity = np.asarray(audit.get("window_velocity", np.zeros(3)), dtype=np.float64)
        if not audit.get("dynamic_state", False):
            velocity = np.zeros(3, dtype=np.float64)
        speed = float(np.linalg.norm(velocity))
        for tau in taus:
            uncertainty = u0 + kv * speed * float(tau)
            shifted = centers + velocity[None, :] * float(tau)
            predictions.extend(
                RiskSphere(int(track_id), center.copy(), float(radius + uncertainty), float(tau))
                for center, radius in zip(shifted, radii)
            )
    return predictions


def v3_execution_multisphere_forecast(
    centers: np.ndarray,
    radii: np.ndarray,
    velocity: np.ndarray,
    *,
    object_id: int = 1,
) -> CompositeForecast:
    """Move the Fresh geometry without reapplying the legacy 6.4 shell.

    The radii already cover every Fresh point and include the geometry fitter's
    explicit margin.  The final 0.09 m verifier threshold and 0.10 m raw-cloud
    guard remain separate safety distances rather than being folded into the
    obstacle geometry a second time.
    """
    center_values = np.asarray(centers, dtype=np.float64)
    radius_values = np.asarray(radii, dtype=np.float64)
    velocity_value = np.asarray(velocity, dtype=np.float64)
    if center_values.ndim != 2 or center_values.shape[1] != 3:
        raise ValueError("centers must have shape (M, 3)")
    if radius_values.shape != (len(center_values),) or len(center_values) == 0:
        raise ValueError("radii must have shape (M,) with M > 0")
    forecasts = [
        ConstantVelocitySphereForecast(
            center,
            velocity_value,
            float(radius),
            config64.FORECAST_HORIZON,
            object_id=int(object_id),
            margin=0.0,
            uncertainty=0.0,
            uncertainty_growth=0.0,
            velocity_radius_scale=0.0,
            beyond_horizon="hold_inflate",
        )
        for center, radius in zip(center_values, radius_values)
    ]
    return CompositeForecast(forecasts)


@dataclass
class PersistentObstacleState:
    """Small state container for the future continuous perception worker.

    It rejects one-frame radius spikes by storing the latest covered geometry
    instead of the historical maximum radius.  Association continuity remains
    explicit and observable.
    """

    track_id: int
    timestamp: float
    center: np.ndarray
    velocity: np.ndarray
    geometry: dict[str, Any]
    association_error_m: float

    def update(
        self,
        *,
        timestamp: float,
        center: np.ndarray,
        velocity: np.ndarray,
        geometry: dict[str, Any],
        association_error_m: float,
        max_association_error_m: float,
    ) -> bool:
        if association_error_m > max_association_error_m or not geometry.get("covered", False):
            return False
        self.timestamp = float(timestamp)
        self.center = np.asarray(center, dtype=np.float64).copy()
        self.velocity = np.asarray(velocity, dtype=np.float64).copy()
        self.geometry = geometry
        self.association_error_m = float(association_error_m)
        return True


def make_v3_fast_factory(legacy_factory: Any):
    """Return a factory compatible with the validated simple-live wrapper."""

    def factory(original_fast: Any, **kwargs: Any):
        return legacy_factory(
            original_fast,
            **kwargs,
            required_component_count=None,
            coarse_gate_is_hard=False,
            clearance_improvement_is_hard=False,
            verified_seed_is_candidate=True,
        )

    return factory
