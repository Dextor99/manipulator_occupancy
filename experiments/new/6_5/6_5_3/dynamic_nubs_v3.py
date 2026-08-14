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
from pathlib import Path
import threading
import time
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
    "post_plan_authorization": "persistent_latest_state_time_aligned",
    "independent_post_plan_fresh_window": False,
    "maximum_latest_state_replans": 1,
    "precommand_perception": "active_during_0.35s_settle",
    "command_time_authorization": True,
    "candidate_playback_mode": "protected_virtual_remaining_trajectory",
    "single_rgbd_owner": "persistent_perception_worker",
    "real_candidate_execution_enabled": False,
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
    raw_guard_distance_m: float = float("inf")

    def update(
        self,
        *,
        timestamp: float,
        center: np.ndarray,
        velocity: np.ndarray,
        geometry: dict[str, Any],
        association_error_m: float,
        max_association_error_m: float,
        raw_guard_distance_m: float = float("inf"),
    ) -> bool:
        if association_error_m > max_association_error_m or not geometry.get("covered", False):
            return False
        self.timestamp = float(timestamp)
        self.center = np.asarray(center, dtype=np.float64).copy()
        self.velocity = np.asarray(velocity, dtype=np.float64).copy()
        self.geometry = geometry
        self.association_error_m = float(association_error_m)
        self.raw_guard_distance_m = float(raw_guard_distance_m)
        return True

    def snapshot(self, *, now_timestamp: float | None = None) -> dict[str, Any]:
        now = float(time.time() if now_timestamp is None else now_timestamp)
        age = max(0.0, now - float(self.timestamp))
        return {
            "timestamp": float(self.timestamp),
            "snapshot_timestamp": now,
            "state_age_s": age,
            "center": np.asarray(self.center, dtype=np.float64).copy(),
            "velocity": np.asarray(self.velocity, dtype=np.float64).copy(),
            "geometry": {
                **self.geometry,
                "component_centers": np.asarray(
                    self.geometry["component_centers"], dtype=np.float64
                ).copy(),
                "component_base_radii": np.asarray(
                    self.geometry["component_base_radii"], dtype=np.float64
                ).copy(),
            },
            "association_error_m": float(self.association_error_m),
            "raw_guard_distance_m": float(self.raw_guard_distance_m),
        }


def time_aligned_snapshot(snapshot: dict[str, Any], *, execution_timestamp: float) -> dict[str, Any]:
    """Propagate one timestamped state to the expected execution start."""
    dt = max(0.0, float(execution_timestamp) - float(snapshot["timestamp"]))
    center = np.asarray(snapshot["center"], dtype=np.float64)
    velocity = np.asarray(snapshot["velocity"], dtype=np.float64)
    propagated = center + velocity * dt
    geometry = trial.translated_multisphere_geometry(
        snapshot["geometry"], center, propagated
    )
    return {
        **snapshot,
        "execution_timestamp": float(execution_timestamp),
        "propagation_dt_s": dt,
        "propagated_center": propagated,
        "geometry": geometry,
    }


class PersistentPerceptionWorker:
    """Continuously update one associated obstacle while Fast is running."""

    def __init__(
        self,
        *,
        processor: Any,
        denoiser: Any,
        args: Any,
        initial_fresh: dict[str, Any],
        initial_geometry: dict[str, Any],
        initial_frames: list[dict[str, Any]],
        output_dir: Path,
    ) -> None:
        self.processor = processor
        self.denoiser = denoiser
        self.args = args
        self.output_dir = Path(output_dir)
        self._lock = threading.Lock()
        self._updated = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self._audits: list[dict[str, Any]] = []
        self._latest_frame_timestamp = float(initial_fresh["last_timestamp"])
        self._latest_raw_guard_distance_m = float("inf")
        self._samples = [
            {
                "timestamp": float(row["timestamp"]),
                "center": np.asarray(row["center"], dtype=np.float64),
                "radius": float(row["radius"]),
                "association_error_m": float(row.get("association_error_m", 0.0)),
            }
            for row in initial_frames
            if row.get("associated") and row.get("center") is not None
        ][-8:]
        self._state = PersistentObstacleState(
            track_id=int(initial_fresh.get("track_id") or 1),
            timestamp=float(initial_fresh["last_timestamp"]),
            center=np.asarray(initial_fresh["center"], dtype=np.float64),
            velocity=np.asarray(initial_fresh["velocity"], dtype=np.float64),
            geometry=initial_geometry,
            association_error_m=float(initial_fresh.get("max_association_error_m", 0.0)),
            raw_guard_distance_m=float("inf"),
        )
        self._initial_snapshot = self._state.snapshot(
            now_timestamp=float(initial_fresh["last_timestamp"])
        )

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("persistent perception worker already started")
        self._thread = threading.Thread(
            target=self._run, name="v3-persistent-perception", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trial.write_json(
            self.output_dir / "persistent_perception_audit.json",
            {
                "status": "FAILED" if self._error else "STOPPED",
                "error": self._error,
                "updates": self._audits,
                "update_count": len(self._audits),
            },
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result = self._state.snapshot()
            result["latest_frame_timestamp"] = self._latest_frame_timestamp
            result["latest_frame_age_s"] = max(
                0.0, time.time() - self._latest_frame_timestamp
            )
            result["raw_guard_distance_m"] = self._latest_raw_guard_distance_m
            result["worker_error"] = self._error
            result["update_count"] = len(self._audits)
            return result

    def diagnostics(self, *, since: int = 0) -> dict[str, Any]:
        with self._lock:
            rows = [dict(row) for row in self._audits[max(0, int(since)) :]]
            return {
                "start_index": max(0, int(since)),
                "end_index": len(self._audits),
                "updates": rows,
                "association_failures": sum(
                    1 for row in rows if not row.get("associated", False)
                ),
                "geometry_coverage_failures": sum(
                    1
                    for row in rows
                    if row.get("associated", False)
                    and not row.get("geometry_covered", False)
                ),
            }

    def guard_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "distance_m": float(self._latest_raw_guard_distance_m),
                "timestamp": float(self._latest_frame_timestamp),
                "age_s": max(0.0, time.time() - self._latest_frame_timestamp),
            }

    def initial_snapshot(self) -> dict[str, Any]:
        return {
            **self._initial_snapshot,
            "worker_error": None,
            "update_count": 0,
        }

    def wait_for_newer_state(
        self, *, after_timestamp: float, timeout_s: float = 0.20
    ) -> dict[str, Any]:
        """Wait for at most one camera-period-scale update, never a Fresh window."""
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._updated:
            while (
                float(self._state.timestamp) <= float(after_timestamp)
                and self._error is None
                and not self._stop.is_set()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._updated.wait(timeout=remaining)
            result = self._state.snapshot()
            result["latest_frame_timestamp"] = self._latest_frame_timestamp
            result["latest_frame_age_s"] = max(
                0.0, time.time() - self._latest_frame_timestamp
            )
            result["raw_guard_distance_m"] = self._latest_raw_guard_distance_m
            result["worker_error"] = self._error
            result["update_count"] = len(self._audits)
            return result

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._update_once()
        except Exception as exc:  # fail-closed state is reported to authorization
            with self._updated:
                self._error = f"{type(exc).__name__}: {exc}"
                self._updated.notify_all()

    def _update_once(self) -> None:
        frame = self.processor.process_frame()
        timestamp = float(getattr(frame, "timestamp", time.time()))
        scene_points = np.asarray(frame.scene_points, dtype=np.float64)
        robot_points = np.asarray(frame.robot_points, dtype=np.float64)
        if self.denoiser is not None:
            scene_points = self.denoiser.filter(scene_points)
        plane_removal = None
        if self.args.remove_planes:
            plane_removal = {
                "enabled": True,
                "distance_threshold": self.args.plane_dist,
                "max_planes": self.args.max_planes,
            }
        clustered = trial.FastClusteringFilter(
            scene_points,
            robot_points,
            workspace=getattr(self.processor, "_workspace", None),
            plane_removal=plane_removal,
            eps=self.args.cluster_eps,
            min_samples=self.args.cluster_min_samples,
            min_points=self.args.cluster_min_points,
            min_volume=self.args.cluster_min_volume,
        )
        clusters = trial.filter_guard_clusters(list(clustered.clusters), self.args)
        guard, _, _, _, _ = trial._find_nearest_cluster_distance_detail(
            robot_points, clusters, []
        )
        audit: dict[str, Any] = {
            "timestamp": timestamp,
            "cluster_count": len(clusters),
            "raw_guard_distance_m": float(guard),
            "associated": False,
        }
        with self._updated:
            self._latest_frame_timestamp = timestamp
            self._latest_raw_guard_distance_m = float(guard)
            self._updated.notify_all()
        if not clusters:
            with self._lock:
                self._audits.append(audit)
            return
        with self._lock:
            state = self._state.snapshot(now_timestamp=timestamp)
            samples = list(self._samples)
        association = trial.associate_fresh_cluster(
            [np.asarray(cluster.center, dtype=np.float64) for cluster in clusters],
            samples,
            timestamp=timestamp,
            trigger_cluster_center=np.asarray(state["center"], dtype=np.float64),
            trigger_velocity=np.asarray(state["velocity"], dtype=np.float64),
            trigger_timestamp=float(state["timestamp"]),
            bootstrap_threshold_m=self.args.dynamic_tracker_association_distance_m,
            continuity_threshold_m=self.args.max_track_cluster_association_m,
        )
        audit.update(
            {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in association.items()
                if key != "cluster_index"
            }
        )
        if not association.get("associated", False):
            with self._lock:
                self._audits.append(audit)
            return
        index = int(association["cluster_index"])
        points = np.asarray(clusters[index].points, dtype=np.float64)
        detection = trial.make_occupancy_object(points, timestamp=timestamp, margin=0.0)
        sample = {
            "timestamp": timestamp,
            "center": np.asarray(detection.center, dtype=np.float64),
            "radius": float(detection.radius),
            "association_error_m": float(association["association_error_m"]),
        }
        samples = (samples + [sample])[-8:]
        fitted = trial.fit_fresh_obstacle_motion(
            samples,
            minimum_frames=3,
            minimum_span_s=0.10,
        )
        geometry = adaptive_geometry_adapter(
            points,
            fit_margin_m=float(self.args.multisphere_fit_margin_m),
            max_components=int(self.args.multisphere_max_components),
        )
        updated = bool(fitted.get("accepted") and geometry.get("covered"))
        audit.update(
            {
                "associated": True,
                "center": sample["center"].tolist(),
                "radius": sample["radius"],
                "fit_accepted": bool(fitted.get("accepted")),
                "geometry_covered": bool(geometry.get("covered")),
            }
        )
        with self._updated:
            self._samples = samples
            if updated:
                self._state.update(
                    timestamp=float(fitted["last_timestamp"]),
                    center=np.asarray(fitted["center"], dtype=np.float64),
                    velocity=np.asarray(fitted["velocity"], dtype=np.float64),
                    geometry=geometry,
                    association_error_m=float(fitted["max_association_error_m"]),
                    max_association_error_m=float(self.args.max_track_cluster_association_m),
                    raw_guard_distance_m=float(guard),
                )
            self._audits.append(audit)
            self._updated.notify_all()


def make_persistent_perception_worker(**kwargs: Any) -> PersistentPerceptionWorker:
    return PersistentPerceptionWorker(**kwargs)


def latest_state_authorize_with_one_replan(
    *,
    worker: PersistentPerceptionWorker,
    args: Any,
    stage4_config: dict[str, Any],
    stage4_model: Any,
    q_now: np.ndarray,
    qd_now: np.ndarray,
    reference_goal: tuple[np.ndarray, np.ndarray, np.ndarray],
    rejoin_goals: Any,
    risk_links: set[str],
    trial_dir: Path,
    candidate_summary: dict[str, Any],
    local_artifacts: dict[str, Any],
    planning_state: dict[str, Any],
    stop_worker_when_done: bool = False,
) -> dict[str, Any]:
    """Authorize against the latest stream state; replan at most once."""
    attempts: list[dict[str, Any]] = []
    current_candidate = candidate_summary
    current_artifacts = local_artifacts
    latest_snapshot: dict[str, Any] | None = None
    latest_aligned: dict[str, Any] | None = None
    local_authorization: dict[str, Any] = {
        "status": "LOCAL_EXECUTION_RECHECK_FAILED",
        "local_execution_authorized": False,
        "reason": "latest_state_not_checked",
        "robot_executed": False,
    }
    try:
        for attempt_index in (1, 2):
            if attempt_index == 2:
                previous_timestamp = (
                    float(latest_snapshot["timestamp"])
                    if latest_snapshot is not None
                    else float(worker.initial_snapshot()["timestamp"])
                )
                plan_snapshot = worker.wait_for_newer_state(
                    after_timestamp=previous_timestamp, timeout_s=0.20
                )
                if plan_snapshot.get("worker_error"):
                    break
                plan_aligned = time_aligned_snapshot(
                    plan_snapshot, execution_timestamp=time.time()
                )
                attempt_dir = Path(trial_dir) / "latest_state_replan" / "attempt_02"
                attempt_dir.mkdir(parents=True, exist_ok=True)
                current_artifacts = {}
                geometry = plan_aligned["geometry"]
                current_candidate = trial.run_fast_repair(
                    args,
                    stage4_config,
                    stage4_model,
                    q_now=np.asarray(q_now, dtype=np.float64),
                    qd_now=np.asarray(qd_now, dtype=np.float64),
                    center=np.asarray(plan_aligned["propagated_center"], dtype=np.float64),
                    velocity=np.asarray(plan_snapshot["velocity"], dtype=np.float64),
                    radius=float(np.max(geometry["component_base_radii"])),
                    risk_links=risk_links,
                    trial_dir=attempt_dir,
                    reference_goal=reference_goal,
                    rejoin_goals=rejoin_goals,
                    obstacle_audit={
                        "v3_latest_state_replan": True,
                        "planning_state_timestamp": float(plan_snapshot["timestamp"]),
                        "planning_state_age_s": float(plan_aligned["propagation_dt_s"]),
                    },
                    multisphere_geometry=geometry,
                    artifacts_out=current_artifacts,
                )

            latest_snapshot = worker.snapshot()
            latest_aligned = time_aligned_snapshot(
                latest_snapshot, execution_timestamp=time.time()
            )
            state_usable = bool(
                not latest_snapshot.get("worker_error")
                and float(latest_aligned["propagation_dt_s"]) <= 0.25
                and latest_aligned["geometry"].get("covered", False)
                and float(latest_snapshot.get("raw_guard_distance_m", float("inf")))
                > float(args.guided_hard_stop_m)
            )
            if state_usable and current_candidate.get("local_repair_ready", False):
                auth_dir = (
                    Path(trial_dir)
                    if attempt_index == 1
                    else Path(trial_dir) / "latest_state_replan" / "attempt_02"
                )
                local_authorization, _ = trial.authorize_local_repair_execution(
                    args,
                    stage4_config,
                    stage4_model,
                    local_repair_ready=True,
                    local_artifacts=current_artifacts,
                    fresh_geometry=latest_aligned["geometry"],
                    fresh_velocity=np.asarray(latest_snapshot["velocity"], dtype=np.float64),
                    trial_dir=auth_dir,
                )
            else:
                local_authorization = {
                    "status": "LOCAL_EXECUTION_RECHECK_FAILED",
                    "local_execution_authorized": False,
                    "reason": (
                        "persistent_state_stale_or_invalid"
                        if not state_usable
                        else "candidate_not_ready"
                    ),
                    "robot_executed": False,
                }
            attempts.append(
                {
                    "attempt": attempt_index,
                    "candidate_source": current_candidate.get("candidate_source"),
                    "candidate_ready": bool(
                        current_candidate.get("local_repair_ready", False)
                    ),
                    "planning_state_timestamp": float(
                        (
                            planning_state
                            if attempt_index == 1
                            else plan_snapshot
                        )["timestamp"]
                    ),
                    "planning_state_center": np.asarray(
                        (
                            planning_state
                            if attempt_index == 1
                            else plan_snapshot
                        )["center"]
                    ).tolist(),
                    "planning_state_velocity": np.asarray(
                        (
                            planning_state
                            if attempt_index == 1
                            else plan_snapshot
                        )["velocity"]
                    ).tolist(),
                    "authorization_state_timestamp": float(latest_snapshot["timestamp"]),
                    "authorization_state_age_s": float(
                        latest_aligned["propagation_dt_s"]
                    ),
                    "authorization_state_center": np.asarray(
                        latest_snapshot["center"]
                    ).tolist(),
                    "authorization_state_velocity": np.asarray(
                        latest_snapshot["velocity"]
                    ).tolist(),
                    "propagated_execution_center": np.asarray(
                        latest_aligned["propagated_center"]
                    ).tolist(),
                    "worker_update_count": int(latest_snapshot["update_count"]),
                    "local_authorization": local_authorization,
                }
            )
            if local_authorization.get("local_execution_authorized", False):
                break
    finally:
        if stop_worker_when_done:
            worker.stop()

    authorized = bool(local_authorization.get("local_execution_authorized", False))
    result = {
        "status": (
            "V3_LATEST_STATE_AUTHORIZED"
            if authorized
            else "V3_LATEST_STATE_REPLAN_EXHAUSTED_HOLD"
        ),
        "authorized": authorized,
        "attempts": attempts,
        "candidate_summary": current_candidate,
        "local_artifacts": current_artifacts,
        "fresh": None
        if latest_snapshot is None
        else {
            "accepted": True,
            "reason": "persistent_latest_state",
            "center": np.asarray(latest_snapshot["center"]).tolist(),
            "velocity": np.asarray(latest_snapshot["velocity"]).tolist(),
            "last_timestamp": float(latest_snapshot["timestamp"]),
            "state_age_s": float(latest_aligned["propagation_dt_s"]),
        },
        "fresh_geometry": None if latest_aligned is None else latest_aligned["geometry"],
        "local_authorization": local_authorization,
        "execution_authorization": {
            "status": "V3_LOCAL_FIRST_FULL_REJOIN_DEFERRED",
            "execution_authorized": False,
            "robot_executed": False,
        },
    }
    trial.write_json(
        Path(trial_dir) / "v3_latest_state_authorization.json",
        {
            key: value
            for key, value in result.items()
            if key not in {"candidate_summary", "local_artifacts"}
        },
    )
    return result


def _playback_snapshot(
    worker: PersistentPerceptionWorker,
    args: Any,
    *,
    evaluation_timestamp: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any], bool, list[str]]:
    snapshot = worker.snapshot()
    aligned = time_aligned_snapshot(
        snapshot,
        execution_timestamp=(
            time.time() if evaluation_timestamp is None else evaluation_timestamp
        ),
    )
    reasons: list[str] = []
    if snapshot.get("worker_error"):
        reasons.append("persistent_worker_error")
    if float(aligned["propagation_dt_s"]) > 0.25:
        reasons.append("obstacle_state_stale")
    if float(snapshot.get("latest_frame_age_s", float("inf"))) > 0.25:
        reasons.append("camera_frame_stale")
    if not aligned["geometry"].get("covered", False):
        reasons.append("geometry_not_covered")
    if float(snapshot.get("raw_guard_distance_m", float("-inf"))) <= float(
        args.guided_hard_stop_m
    ):
        reasons.append("raw_hard_guard_not_safe")
    return snapshot, aligned, not reasons, reasons


def _remaining_clearance(
    evaluator: Any,
    trajectory: Any,
    forecast: Any,
    *,
    playback_time_s: float,
    sample_step_s: float = 0.05,
) -> dict[str, Any]:
    start = float(np.clip(playback_time_s, 0.0, trajectory.total_duration))
    remaining = float(trajectory.total_duration) - start
    taus = np.arange(0.0, remaining + 0.5 * sample_step_s, sample_step_s)
    if len(taus) == 0 or taus[-1] < remaining - 1.0e-9:
        taus = np.r_[taus, remaining]
    best = {"min_distance_m": float("inf"), "tau_s": None, "nearest_link": None}
    for tau in taus:
        risk = evaluator.configuration(
            np.asarray(trajectory.evaluate(start + float(tau)), dtype=np.float64),
            forecast,
            float(tau),
            density="medium",
            with_gradient=False,
        )
        if float(risk.min_distance) < best["min_distance_m"]:
            best = {
                "min_distance_m": float(risk.min_distance),
                "tau_s": float(tau),
                "nearest_link": risk.nearest_link,
            }
    return best


def _fixed_configuration_clearance(
    evaluator: Any,
    q: np.ndarray,
    forecast: Any,
    *,
    horizon_s: float,
    sample_step_s: float = 0.05,
) -> dict[str, Any]:
    taus = np.arange(0.0, float(horizon_s) + 0.5 * sample_step_s, sample_step_s)
    best = {"min_distance_m": float("inf"), "tau_s": None, "nearest_link": None}
    for tau in taus:
        risk = evaluator.configuration(
            np.asarray(q, dtype=np.float64),
            forecast,
            float(tau),
            density="medium",
            with_gradient=False,
        )
        if float(risk.min_distance) < best["min_distance_m"]:
            best = {
                "min_distance_m": float(risk.min_distance),
                "tau_s": float(tau),
                "nearest_link": risk.nearest_link,
            }
    return best


def _direct_goal_diagnostic(
    evaluator: Any,
    q_tail: np.ndarray,
    q_goal: np.ndarray,
    forecast: Any,
    *,
    duration_s: float = 1.0,
    sample_step_s: float = 0.05,
) -> dict[str, Any]:
    best = {"min_distance_m": float("inf"), "tau_s": None, "nearest_link": None}
    for tau in np.arange(0.0, duration_s + 0.5 * sample_step_s, sample_step_s):
        alpha = float(np.clip(tau / duration_s, 0.0, 1.0))
        q = (1.0 - alpha) * q_tail + alpha * q_goal
        risk = evaluator.configuration(
            q,
            forecast,
            float(tau),
            density="medium",
            with_gradient=False,
        )
        if float(risk.min_distance) < best["min_distance_m"]:
            best = {
                "min_distance_m": float(risk.min_distance),
                "tau_s": float(tau),
                "nearest_link": risk.nearest_link,
            }
    return {
        **best,
        "metric_only": True,
        "trajectory_model": "joint_linear_tail_to_preset_goal_diagnostic",
    }


def run_virtual_candidate_playback_shadow(
    *,
    worker: PersistentPerceptionWorker,
    args: Any,
    stage4_config: dict[str, Any],
    stage4_model: Any,
    local_artifacts: dict[str, Any],
    trial_dir: Path,
    task_goal_q: np.ndarray,
) -> dict[str, Any]:
    """Keep the robot stopped while shadowing pre-command and 1 s playback."""
    output_dir = Path(trial_dir) / "v3_playback_shadow"
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = local_artifacts["candidate_trajectory"]
    evaluator, _, _ = trial.make_risk_stack(stage4_config, stage4_model, None)
    baseline = worker.diagnostics()["end_index"]
    events = ["PERSISTENT_TRACKER_RUNNING"]
    precommand_samples: list[dict[str, Any]] = []
    settle_started = time.monotonic()
    settle_duration = max(0.0, float(args.candidate_pre_execute_settle_s))
    while time.monotonic() - settle_started < settle_duration:
        snapshot = worker.snapshot()
        precommand_samples.append(
            {
                "elapsed_s": time.monotonic() - settle_started,
                "state_timestamp": float(snapshot["timestamp"]),
                "state_age_s": float(snapshot["state_age_s"]),
                "frame_age_s": float(snapshot.get("latest_frame_age_s", float("inf"))),
                "raw_guard_distance_m": float(snapshot["raw_guard_distance_m"]),
                "worker_update_count": int(snapshot["update_count"]),
            }
        )
        time.sleep(0.02)

    command_snapshot, command_aligned, command_state_ok, command_reasons = (
        _playback_snapshot(worker, args)
    )
    command_authorization = {
        "status": "PRECOMMAND_RECHECK_FAILED",
        "local_execution_authorized": False,
        "reason": command_reasons,
    }
    if command_state_ok:
        command_authorization, _ = trial.authorize_local_repair_execution(
            args,
            stage4_config,
            stage4_model,
            local_repair_ready=True,
            local_artifacts=local_artifacts,
            fresh_geometry=command_aligned["geometry"],
            fresh_velocity=np.asarray(command_snapshot["velocity"], dtype=np.float64),
            trial_dir=output_dir / "precommand_authorization",
        )

    final_snapshot, final_aligned, final_state_ok, final_reasons = _playback_snapshot(
        worker, args
    )
    final_forecast = None
    final_clearance = None
    if final_state_ok:
        final_forecast = v3_execution_multisphere_forecast(
            np.asarray(final_aligned["geometry"]["component_centers"], dtype=np.float64),
            np.asarray(final_aligned["geometry"]["component_base_radii"], dtype=np.float64),
            np.asarray(final_snapshot["velocity"], dtype=np.float64),
        )
        final_clearance = _remaining_clearance(
            evaluator, trajectory, final_forecast, playback_time_s=0.0
        )
    diagnostics = worker.diagnostics(since=baseline)
    precommand_min_raw_guard = min(
        [float(row["raw_guard_distance_m"]) for row in precommand_samples]
        + [float(final_snapshot["raw_guard_distance_m"])]
    )
    precommand_authorized = bool(
        command_authorization.get("local_execution_authorized", False)
        and final_state_ok
        and final_clearance is not None
        and final_clearance["min_distance_m"] >= float(args.online_accept_m)
        and precommand_min_raw_guard > float(args.guided_hard_stop_m)
        and diagnostics["association_failures"] == 0
        and diagnostics["geometry_coverage_failures"] == 0
    )
    precommand_failure_reasons = list(final_reasons)
    if not command_authorization.get("local_execution_authorized", False):
        precommand_failure_reasons.append("command_time_full_verifier_rejected")
    if final_clearance is None or final_clearance["min_distance_m"] < float(
        args.online_accept_m
    ):
        precommand_failure_reasons.append("command_time_clearance_below_online_gate")
    if precommand_min_raw_guard <= float(args.guided_hard_stop_m):
        precommand_failure_reasons.append("precommand_raw_hard_guard_not_safe")
    if diagnostics["association_failures"]:
        precommand_failure_reasons.append("precommand_tracker_association_failed")
    if diagnostics["geometry_coverage_failures"]:
        precommand_failure_reasons.append("precommand_geometry_coverage_failed")
    precommand_failure_reasons = list(dict.fromkeys(precommand_failure_reasons))
    if precommand_authorized:
        events.append("PRECOMMAND_RECHECK_AUTHORIZED")
    else:
        events.append("PRECOMMAND_RECHECK_HOLD")

    playback_samples: list[dict[str, Any]] = []
    playback_failure_reasons: list[str] = list(precommand_failure_reasons)
    playback_start_update_count = int(final_snapshot["update_count"])
    minimum_remaining = float("inf")
    minimum_raw_guard = float(final_snapshot["raw_guard_distance_m"])
    if precommand_authorized:
        events.append("VIRTUAL_LOCAL_PLAYBACK_STARTED")
        playback_started = time.monotonic()
        while True:
            playback_t = min(
                float(trajectory.total_duration), time.monotonic() - playback_started
            )
            snapshot, aligned, state_ok, state_reasons = _playback_snapshot(worker, args)
            minimum_raw_guard = min(
                minimum_raw_guard, float(snapshot["raw_guard_distance_m"])
            )
            clearance = None
            if state_ok:
                forecast = v3_execution_multisphere_forecast(
                    np.asarray(aligned["geometry"]["component_centers"], dtype=np.float64),
                    np.asarray(aligned["geometry"]["component_base_radii"], dtype=np.float64),
                    np.asarray(snapshot["velocity"], dtype=np.float64),
                )
                clearance = _remaining_clearance(
                    evaluator,
                    trajectory,
                    forecast,
                    playback_time_s=playback_t,
                )
                minimum_remaining = min(
                    minimum_remaining, clearance["min_distance_m"]
                )
                if clearance["min_distance_m"] < float(args.online_accept_m):
                    playback_failure_reasons.append("remaining_clearance_below_online_gate")
            else:
                playback_failure_reasons.extend(state_reasons)
            playback_samples.append(
                {
                    "playback_time_s": playback_t,
                    "state_timestamp": float(snapshot["timestamp"]),
                    "state_age_s": float(aligned["propagation_dt_s"]),
                    "frame_age_s": float(snapshot.get("latest_frame_age_s", float("inf"))),
                    "raw_guard_distance_m": float(snapshot["raw_guard_distance_m"]),
                    "remaining_clearance": clearance,
                    "worker_update_count": int(snapshot["update_count"]),
                    "state_reasons": state_reasons,
                }
            )
            if playback_failure_reasons or playback_t >= float(trajectory.total_duration):
                break
            time.sleep(0.02)

    playback_diagnostics = worker.diagnostics(since=baseline)
    if playback_diagnostics["association_failures"]:
        playback_failure_reasons.append("tracker_association_failed")
    if playback_diagnostics["geometry_coverage_failures"]:
        playback_failure_reasons.append("geometry_coverage_failed")
    playback_failure_reasons = list(dict.fromkeys(playback_failure_reasons))
    playback_passed = bool(precommand_authorized and not playback_failure_reasons)
    events.append(
        "VIRTUAL_LOCAL_PLAYBACK_COMPLETED"
        if playback_passed
        else "VIRTUAL_LOCAL_PLAYBACK_HOLD"
    )

    tail_snapshot, tail_aligned, tail_state_ok, tail_reasons = _playback_snapshot(
        worker, args
    )
    tail_hold = None
    goal_diagnostic = None
    if playback_passed and tail_state_ok:
        tail_forecast = v3_execution_multisphere_forecast(
            np.asarray(tail_aligned["geometry"]["component_centers"], dtype=np.float64),
            np.asarray(tail_aligned["geometry"]["component_base_radii"], dtype=np.float64),
            np.asarray(tail_snapshot["velocity"], dtype=np.float64),
        )
        q_tail = np.asarray(trajectory.evaluate(trajectory.total_duration), dtype=np.float64)
        tail_hold = _fixed_configuration_clearance(
            evaluator,
            q_tail,
            tail_forecast,
            horizon_s=float(args.prediction_horizon_s),
        )
        goal_diagnostic = _direct_goal_diagnostic(
            evaluator,
            q_tail,
            np.asarray(task_goal_q, dtype=np.float64),
            tail_forecast,
        )
    events.append(
        "TAIL_RISK_EVALUATED"
        if playback_passed and tail_state_ok
        else "TAIL_RISK_NOT_EVALUATED"
    )
    result = {
        "status": (
            "V3_VIRTUAL_PLAYBACK_SHADOW_PASS"
            if playback_passed
            else "V3_VIRTUAL_PLAYBACK_SHADOW_HOLD"
        ),
        "robot_commanded": False,
        "events": events,
        "precommand_wait_s": settle_duration,
        "precommand_samples": precommand_samples,
        "precommand_state_age_s": float(final_aligned["propagation_dt_s"]),
        "precommand_clearance_m": (
            None if final_clearance is None else final_clearance["min_distance_m"]
        ),
        "precommand_min_raw_guard_m": precommand_min_raw_guard,
        "precommand_authorization": command_authorization,
        "precommand_failure_reasons": precommand_failure_reasons,
        "precommand_final_state_reasons": final_reasons,
        "playback_samples": playback_samples,
        "playback_tracker_update_count": max(
            0, int(tail_snapshot["update_count"]) - playback_start_update_count
        ),
        "playback_min_predicted_remaining_clearance_m": (
            None if not np.isfinite(minimum_remaining) else minimum_remaining
        ),
        "playback_min_raw_guard_m": minimum_raw_guard,
        "tracker_association_failures": playback_diagnostics[
            "association_failures"
        ],
        "geometry_coverage_failures": playback_diagnostics[
            "geometry_coverage_failures"
        ],
        "playback_failure_reasons": playback_failure_reasons,
        "tail_state_valid": tail_state_ok,
        "tail_state_reasons": tail_reasons,
        "tail_hold_predicted_clearance_m": (
            None if tail_hold is None else tail_hold["min_distance_m"]
        ),
        "tail_hold_status": (
            None
            if tail_hold is None
            else "TAIL_SHORT_HORIZON_SAFE"
            if tail_hold["min_distance_m"] >= float(args.replan_in_m)
            else "NEXT_LOCAL_REPLAN_REQUIRED"
        ),
        "goal_continuation_diagnostic": goal_diagnostic,
        "goal_continuation_predicted_safe": bool(
            goal_diagnostic is not None
            and goal_diagnostic["min_distance_m"] >= float(args.online_accept_m)
        ),
    }
    trial.write_json(output_dir / "playback_shadow_summary.json", result)
    return result


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
