#!/usr/bin/env python3
"""Run a 6.5.3 dynamic-obstacle Fast CCRO-NUBS trial.

This first real-system implementation is intentionally staged:

* ``--mode shadow`` opens RealSense + AUBO feedback, detects/tracks dynamic
  obstacles, triggers STRO/CCRO risk, generates a 1 s Fast CCRO-NUBS candidate,
  validates it, and saves logs/figures.  It never commands the robot.
* ``--mode moving-shadow-stop`` additionally commands the familiar 6.5.2
  low-speed reference line and stops on trigger/hold.  It still does not switch
  to the candidate trajectory; it is the required pilot before live switching.
* ``--mode live-stop-replan-execute`` prefers a Fresh #2-authorized full
  repair+rejoin trajectory.  If only the local repair is authorized, v2 may
  execute it, hold at its safe tail, and use Fresh #3 to authorize a separate
  C2 bridge to a later reference state.  Every path remains fail-closed before
  the guarded recorded-reference remainder resumes.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[4]
EXP651 = ROOT / "experiments" / "new" / "6_5" / "6_5_1"
EXP652 = ROOT / "experiments" / "new" / "6_5" / "6_5_2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for p in (EXP651, EXP652):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

EXP64 = ROOT / "experiments" / "new" / "6_4"
common64 = importlib.import_module("experiments.new.6_4.common_64")
repair_v3_mod = importlib.import_module("experiments.new.6_4.repair.repair_v3")
constant_forecast = common64.constant_forecast
constant_multisphere_forecast = common64.constant_multisphere_forecast
load_stage4_config = common64.load_stage4_config
load_stage4_surface_model = common64.load_surface_model
make_risk_stack = common64.make_risk_stack
git_commit_hash = common64.git_commit_hash
git_is_dirty = common64.git_is_dirty
run_repair_v3 = repair_v3_mod.run_repair_v3


def v3_execution_multisphere_forecast(*args: Any, **kwargs: Any) -> Any:
    """Lazy-load V3 execution geometry to avoid the trial/V3 import cycle."""
    module = importlib.import_module("experiments.new.6_5.6_5_3.dynamic_nubs_v3")
    return module.v3_execution_multisphere_forecast(*args, **kwargs)
from execute_652_planar_y_guarded import (  # noqa: E402
    call_cartesian_motion,
    check_pose_limits,
    make_pose,
    parse_home_degrees,
    require_confirmation,
    wait_for_joints,
)
from execute_652_ccro_nubs_offline_track_guarded import (  # noqa: E402
    joint_error,
    maybe_downsample,
    resample_for_offline_track,
    trajectory_stats,
)
from perception.geometry_fit import (  # noqa: E402
    create_obb_wireframe,
    create_sphere_wireframe,
    make_occupancy_object,
)
from perception.occupancy_tracker import OccupancyTracker  # noqa: E402
from planning.nubs_trajectory import CompositeTrajectory6D, NUBSTrajectory6D, TrajectorySamples  # noqa: E402
from planning.robot_surface_model import RobotSurfaceModel  # noqa: E402
from risk.prediction import RiskSphere, predict_risk_spheres  # noqa: E402

PRECOMMAND_HOLD_STATUSES = {
    "PERSISTENT_TRACKER_NOT_READY_PRECOMMAND",
    "COMMAND_TIME_REVALIDATION_HOLD_PRECOMMAND",
    "FINAL_PRECOMMAND_HOLD_PRECOMMAND",
}
PRECOMMAND_REPLAN_STATUSES = {
    "COMMAND_TIME_REVALIDATION_REPLAN_REQUIRED",
    "FINAL_PRECOMMAND_REVALIDATION_REPLAN_REQUIRED",
    "ROBOT_NOT_SETTLED_PRECOMMAND",
    "EXECUTION_WAYPOINT_BOUNDARY_REPLAN_REQUIRED",
}
from risk.safety_policy import SafetyPolicy  # noqa: E402
from robot.robot_commander import RobotCommander  # noqa: E402
from robot.safety_guided_motion import (  # noqa: E402
    AdaptiveSafetyController,
    _find_nearest_cluster_distance_detail,
    _is_obstacle_in_motion_direction,
)
from run_651_perception_capture import (  # noqa: E402
    JOINT_NAMES,
    load_surface_model as load_live_surface_model,
    nearest_cluster_to_links,
    nearest_sphere_to_links,
    q_from_reader,
    risk_color_level,
)
from robot.linear_move_debug import fmt_joints  # noqa: E402
from test_clustering_filtering import FastClusteringFilter, TemporalDenoiser  # noqa: E402
from test_remove_robot_points_fast import SceneProcessor  # noqa: E402
from utils.config import load_config_dir  # noqa: E402


DEFAULT_OUTPUT = ROOT / "results" / "new" / "6_5" / "6_5_3" / "dynamic_repair_formal"
REQUIRED_OPERATOR_PHRASE = "CCRO_653_DYNAMIC_SHADOW_APPROVED"
LIVE_CANDIDATE_EXECUTE_PHRASE = "CCRO_653_LIVE_CANDIDATE_EXECUTE_APPROVED"
FORMAL_PROTOCOL_ID = "653_unified_d1_d2_v2"

# Optional, experiment-specific continuation hook.  The formal/default path
# never installs this hook.  It exists so a protected wrapper can consume
# Fresh #3 at the *measured* local tail and perform a bounded event replan
# without changing the already validated first-segment planner or rejoin path.
POST_LOCAL_FRESH3_HANDLER = None
# Optional protocol-level predictor.  The legacy/default path remains the
# original single-sphere STRO model.  V3 installs an adaptive multi-sphere
# predictor without changing archived V2 semantics.
RISK_SPHERE_PREDICTOR = None
# V2 requires the established dynamic-track hysteresis before STRO can
# trigger. V3 sets this false so age+association make a track risk-eligible;
# speed then selects dynamic or quasi-static prediction rather than existence.
RISK_TRIGGER_REQUIRES_DYNAMIC_TRACK = True
# Optional V3-only hooks.  The default/V2 path retains its archived serial
# Fresh #2 behavior.  V3 installs a persistent perception worker before Fast
# and an immediate latest-state authorization policy afterwards.
PERSISTENT_OBSTACLE_WORKER_FACTORY = None
LATEST_STATE_AUTHORIZATION_POLICY = None
POST_AUTHORIZATION_PLAYBACK_SHADOW = None
# Optional final-live consumer for a latest-state-authorized local candidate.
# The default, V2 and V3 shadow paths never install it.  A handler owns the
# persistent worker and all subsequent robot commands, then returns one
# terminal closed-loop result so the legacy single-segment path is skipped.
POST_AUTHORIZATION_CLOSED_LOOP_HANDLER = None
# Optional mid-trajectory predictive monitor.  A rolling wrapper installs this
# only for its live event-replan path; the default pilot remains one-shot.
MID_EXECUTION_MONITOR_FACTORY = None

SCENARIOS = {
    "D1": {
        "name": "crossing_body",
        "description": "lateral obstacle crossing of the future robot swept region",
        "prompt": "准备让障碍从侧面垂直/斜交 reference，横穿机器人未来扫掠区域（不要沿轨迹迎面接近）",
    },
    "D2": {
        "name": "opposing_approach",
        "description": "opposing or oblique obstacle approach to the future robot swept region",
        "prompt": "准备让障碍沿相向或斜向路径接近机器人未来扫掠区域",
    },
}

# One scene-independent protocol for every formal D1/D2 robot-motion trial.
# Scene labels describe obstacle geometry only and have no control authority.
FORMAL_PROTOCOL = {
    "remove_planes": True,
    "plane_dist": 0.02,
    "cluster_eps": 0.05,
    "cluster_min_samples": 15,
    "cluster_min_points": 15,
    "cluster_min_volume": 0.0005,
    "surface_density": "coarse",
    "temporal_denoise": True,
    "denoise_voxel": 0.04,
    "denoise_conf": 2,
    "denoise_decay": 0.4,
    "min_track_age": 3,
    "min_dynamic_trigger_speed_m_s": 0.08,
    "dynamic_exit_speed_m_s": 0.04,
    "dynamic_exit_streak_frames": 3,
    "dynamic_speed_window": 5,
    "dynamic_valid_streak_frames": 2,
    "dynamic_tracker_association_distance_m": 0.12,
    "dynamic_tracker_motion_gate_speed_m_s": 0.03,
    "dynamic_tracker_max_miss": 2,
    "max_track_cluster_association_m": 0.08,
    "prediction_horizon_s": 0.5,
    # Initial STRO early-warning horizon only. Fast/Fresh/rolling execution
    # prediction remains the separate 0.5 s horizon below.
    "stro_trigger_horizon_s": 1.2,
    "prediction_step_s": 0.1,
    "prediction_margin_m": 0.035,
    "prediction_uncertainty_m": 0.02,
    "replan_in_m": 0.14,
    "moving_shadow_replan_in_m": 0.14,
    "moving_shadow_current_stop_m": 0.12,
    "guided_d_safe_m": 0.12,
    "guided_d_slow_m": 0.12,
    "guided_d_stop_m": 0.08,
    "guided_hard_stop_m": 0.10,
    "guided_max_decel": 2.0,
    "guided_max_accel": 0.5,
    "guided_dynamic_lookahead_s": 0.15,
    "local_horizon_s": 1.0,
    "rejoin_search_step_s": 0.25,
    "rejoin_max_offset_s": 2.0,
    "local_segments": 5,
    "online_accept_m": 0.09,
    "min_clearance_improvement_m": 0.003,
    "fast_budget_ms": 150.0,
    "post_stop_recheck_duration_s": 0.6,
    "post_stop_recheck_min_frames": 3,
    "post_stop_recheck_min_span_s": 0.25,
    "multisphere_fit_margin_m": 0.005,
    "multisphere_max_components": 4,
    "gripper_base_min_z_m": 0.46,
    "line_velocity_m_s": 0.020,
    "line_acc_m_s2": 0.05,
    # Frozen after three successful empty-scene executions (r02-r04). This
    # prevents a formal live trial from falling back to a legacy time scale.
    "candidate_playback_duration_s": 1.0,
    "candidate_controller_waypoint_period_s": 0.005,
    "candidate_joint_velc": 0.006,
    "candidate_joint_acc": 0.012,
    # The lateral initializer remains an offline diagnostic only.  Formal
    # robot trials retain the production linear NUBS initialization.
    "fast_warm_start": "linear",
    "rolling_fast_max_s": 3.0,
    "rolling_observation_duration_s": 0.25,
    "rolling_observation_min_frames": 2,
    "rolling_observation_min_span_s": 0.10,
}
ROBOT_MOTION_MODES = {"moving-shadow-stop", "live-stop-replan-execute"}


def formal_protocol_violations(args: argparse.Namespace) -> list[str]:
    violations: list[str] = []
    for name, expected in FORMAL_PROTOCOL.items():
        # The default formal protocol remains 1.0 s.  A separately explicit
        # experimental authorization may test only a bounded shorter playback;
        # all trajectory and latest-state safety gates remain mandatory.
        if name == "candidate_playback_duration_s" and getattr(
            args, "allow_experimental_playback_duration", False
        ):
            duration = float(getattr(args, name))
            if not 0.80 <= duration <= 1.00:
                violations.append(
                    f"{name}={duration!r} outside experimental range [0.80, 1.00]"
                )
            continue
        actual = getattr(args, name)
        if isinstance(expected, bool):
            matches = actual is expected
        elif isinstance(expected, (int, float)):
            matches = math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-12)
        else:
            matches = actual == expected
        if not matches:
            violations.append(f"{name}={actual!r} (formal={expected!r})")
    return violations


def formal_protocol_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {name: getattr(args, name) for name in FORMAL_PROTOCOL}


def select_dynamic_execution_path(
    *,
    local_authorized: bool,
    full_authorized: bool,
    rolling_local_enabled: bool,
) -> str | None:
    """Choose execution semantics without letting a full rejoin preempt rolling-local.

    The full candidate remains useful as a diagnostic, but during one bounded
    rolling-local event only the Fresh-authorized local segment may execute.
    """
    if rolling_local_enabled:
        return "ROLLING_LOCAL_FIRST" if local_authorized else None
    if full_authorized:
        return "FULL_FIRST"
    return "LOCAL_FIRST_DELAYED_REJOIN" if local_authorized else None


def avoidance_side_consistent(
    locked_tail_delta_q: np.ndarray | None,
    candidate_tail_delta_q: np.ndarray,
    *,
    opposite_projection_tolerance_rad: float,
) -> dict[str, Any]:
    """Reject a later repair that substantially reverses the first tail offset.

    This is deliberately a weak continuity lock: orthogonal refinement and a
    smaller same-side offset remain legal.  It does not force clearance growth.
    """
    candidate = np.asarray(candidate_tail_delta_q, dtype=np.float64)
    if candidate.shape != (6,) or not np.all(np.isfinite(candidate)):
        return {"accepted": False, "reason": "invalid_candidate_tail_delta_q"}
    if locked_tail_delta_q is None:
        norm = float(np.linalg.norm(candidate))
        return {
            "accepted": bool(norm > 1.0e-9),
            "reason": "side_lock_initialized" if norm > 1.0e-9 else "zero_initial_avoidance_offset",
            "locked_tail_delta_q": candidate.copy(),
            "projection_rad": norm,
        }
    locked = np.asarray(locked_tail_delta_q, dtype=np.float64)
    norm = float(np.linalg.norm(locked))
    if locked.shape != (6,) or not np.all(np.isfinite(locked)) or norm <= 1.0e-9:
        return {"accepted": False, "reason": "invalid_locked_avoidance_side"}
    direction = locked / norm
    projection = float(candidate @ direction)
    accepted = projection >= -float(opposite_projection_tolerance_rad)
    return {
        "accepted": accepted,
        "reason": "same_side_or_orthogonal" if accepted else "opposite_avoidance_side",
        "locked_tail_delta_q": locked.copy(),
        "projection_rad": projection,
        "opposite_projection_tolerance_rad": float(opposite_projection_tolerance_rad),
    }


def rolling_local_reference_schedule(
    reference_start_time_s: float,
    *,
    local_horizon_s: float,
    max_segments: int,
    reference_end_time_s: float,
) -> list[dict[str, float | int]]:
    """Return monotonically advancing absolute reference anchors per segment."""
    if local_horizon_s <= 0.0 or max_segments < 1:
        raise ValueError("rolling-local horizon and segment count must be positive")
    schedule = []
    for index in range(max_segments):
        plan_start = min(
            float(reference_end_time_s),
            float(reference_start_time_s) + index * float(local_horizon_s),
        )
        goal = min(float(reference_end_time_s), plan_start + float(local_horizon_s))
        if goal <= plan_start + 1.0e-9:
            break
        schedule.append(
            {
                "segment": index + 1,
                "reference_plan_start_time_s": plan_start,
                "reference_goal_time_s": goal,
            }
        )
    return schedule


def rolling_local_segment_gate(
    *,
    reference_min_distance_m: float,
    local_repair_ready: bool,
    side_consistent: bool,
    fresh_authorized: bool,
    replan_threshold_m: float,
) -> dict[str, Any]:
    """Classify one segment before any real or virtual state advance."""
    if not np.isfinite(reference_min_distance_m):
        return {"advance": False, "status": "INVALID_REFERENCE_RISK_HOLD"}
    if reference_min_distance_m >= replan_threshold_m:
        return {"advance": False, "status": "REFERENCE_SAFE_FOR_REJOIN"}
    advance = bool(local_repair_ready and side_consistent and fresh_authorized)
    return {
        "advance": advance,
        "status": "ROLLING_LOCAL_SEGMENT_AUTHORIZED" if advance else "ROLLING_LOCAL_SEGMENT_REJECTED",
    }

FRAME_FIELDS = [
    "frame",
    "t_s",
    "timestamp",
    "scene_points",
    "robot_points",
    "cluster_count",
    "raw_point_count",
    "roi_point_count",
    "safety_roi_point_count",
    "rho_retain",
    "table_z_m",
    "table_plane_valid",
    "stable_track_count",
    "risk_sphere_count",
    "nearest_distance_m",
    "nearest_link",
    "nearest_cluster_index",
    "nearest_cluster_x",
    "nearest_cluster_y",
    "nearest_cluster_z",
    "predicted_distance_m",
    "predicted_nearest_link",
    "predicted_tau_s",
    "trigger_horizon_s",
    "predicted_object_id",
    "trigger_block_reason",
    "reference_state",
    "reference_armed",
    "reference_index",
    "reference_index_step",
    "reference_step_clamped",
    "reference_time_s",
    "reference_tcp_y_m",
    "reference_actual_y_error_m",
    "reference_joint_match_max_rad",
    "reference_future_time_s",
    "reference_future_index",
    "reference_future_delta_q_max_rad",
    "predicted_object_speed_m_s",
    "predicted_object_velocity_x_m_s",
    "predicted_object_velocity_y_m_s",
    "predicted_object_velocity_z_m_s",
    "predicted_object_raw_speed_m_s",
    "predicted_object_filtered_speed_m_s",
    "predicted_object_raw_velocity_x_m_s",
    "predicted_object_raw_velocity_y_m_s",
    "predicted_object_raw_velocity_z_m_s",
    "predicted_object_velocity_ema_alpha",
    "predicted_object_radius_m",
    "predicted_object_age",
    "predicted_object_association_error_m",
    "dynamic_object_prediction_ready",
    "dynamic_object_valid",
    "dynamic_object_block_reason",
    "risk_state_current",
    "risk_state_predicted",
    "max_track_speed_m_s",
    "motion_y_m",
    "guard_distance_m",
    "guard_object_id",
    "guard_in_motion_direction",
    "guard_speed_scale",
    "guard_decision",
    "guard_cluster_count",
    "guard_robot_points_source",
    "guard_robot_points_count",
    "elapsed_ms",
    *[f"q{j+1}_rad" for j in range(6)],
    *[f"qd{j+1}_rad_s" for j in range(6)],
]

CLUSTER_FIELDS = [
    "frame", "t_s", "cluster_index", "dynamic_tracker_input", "radius_in_legacy_band", "point_count",
    "center_x", "center_y", "center_z", "raw_radius_m",
    "bbox_dx_m", "bbox_dy_m", "bbox_dz_m", "raw_centroid_speed_m_s",
]

TRACK_FIELDS = [
    "frame", "t_s", "track_id", "age", "center_x", "center_y", "center_z",
    "instant_speed_m_s", "window_speed_m_s", "median_speed_m_s", "raw_cluster_speed_m_s",
    "window_velocity_x_m_s", "window_velocity_y_m_s", "window_velocity_z_m_s",
    "raw_window_speed_m_s", "filtered_speed_m_s",
    "raw_window_velocity_x_m_s", "raw_window_velocity_y_m_s", "raw_window_velocity_z_m_s",
    "velocity_ema_alpha",
    "cluster_radius_raw_m", "tracked_radius_m", "risk_radius_m", "raw_radius_m",
    "association_error_m", "valid_streak", "dynamic_state", "prediction_ready", "dynamic_valid",
    "block_reason",
]


class RecordedReference:
    """Joint reference recorded by ``prepare_653_reference.py``.

    Online progress is located by the closest recorded joint state in a small
    forward-only window.  This keeps both STRO and local repair tied to the
    same physical reference instead of extrapolating the measured velocity.
    """

    def __init__(self, times: np.ndarray, q: np.ndarray, qd: np.ndarray, y: np.ndarray | None = None):
        if len(times) < 2 or q.shape != qd.shape or q.shape != (len(times), 6):
            raise ValueError("invalid recorded 6.5.3 reference")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("reference timestamps must be strictly increasing")
        self.times = times - times[0]
        self.q = q
        self.qd = qd
        self.qdd = np.gradient(qd, self.times, axis=0, edge_order=1)
        self.y = None if y is None else np.asarray(y, dtype=np.float64)
        if self.y is not None and self.y.shape != (len(times),):
            raise ValueError("reference TCP Y shape mismatch")
        self.index = 0
        self.dt_median = float(np.median(np.diff(self.times)))
        self._increment_p99_cache: dict[float, float] = {}

    @classmethod
    def load(cls, path: Path) -> "RecordedReference":
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) < 2:
            raise ValueError(f"recorded reference has too few rows: {path}")
        if "pose_y" in rows[0]:
            ys = np.asarray([float(row["pose_y"]) for row in rows], dtype=np.float64)
            y_span = float(np.max(ys) - np.min(ys))
            if y_span < 0.75:
                raise ValueError(
                    f"recorded reference covers only {y_span:.4f} m in Y; at least 0.75 m is required"
                )
        times = np.asarray([float(row["t_s"]) for row in rows], dtype=np.float64)
        q = np.asarray([[float(row[f"q{j}_rad"]) for j in range(1, 7)] for row in rows], dtype=np.float64)
        qd = np.asarray([[float(row[f"qd{j}_rad_s"]) for j in range(1, 7)] for row in rows], dtype=np.float64)
        y = None if "pose_y" not in rows[0] else np.asarray([float(row["pose_y"]) for row in rows], dtype=np.float64)
        return cls(times, q, qd, y)

    def reset(self) -> None:
        self.index = 0

    def locate(
        self,
        q_actual: np.ndarray,
        *,
        y_actual: float | None,
        max_forward_step: int,
        joint_refine_window: int = 8,
    ) -> dict[str, Any]:
        """Locate progress by monotonic TCP Y, refine by joints, and clamp jumps."""
        previous = self.index
        if self.y is not None and y_actual is not None and np.isfinite(y_actual):
            y_index = int(np.argmin(np.abs(self.y - float(y_actual))))
            lo = max(previous, y_index - int(joint_refine_window))
            hi = min(len(self.q), y_index + int(joint_refine_window) + 1)
        else:
            lo = previous
            hi = min(len(self.q), previous + max(2, int(max_forward_step)) + 1)
        if hi <= lo:
            candidate = previous
        else:
            candidate = lo + int(np.argmin(np.max(np.abs(self.q[lo:hi] - q_actual[None, :]), axis=1)))
        unclamped_step = max(0, candidate - previous)
        step = min(unclamped_step, max(0, int(max_forward_step)))
        self.index = min(len(self.q) - 1, previous + step)
        return {
            "index": self.index,
            "step": step,
            "candidate_index": candidate,
            "step_was_clamped": unclamped_step > step,
            "time_s": float(self.times[self.index]),
            "tcp_y_m": None if self.y is None else float(self.y[self.index]),
            "actual_y_error_m": None if self.y is None or y_actual is None else abs(float(self.y[self.index]) - float(y_actual)),
            "joint_match_max_rad": float(np.max(np.abs(self.q[self.index] - q_actual))),
        }

    def index_after(self, delta_s: float) -> int:
        target = min(self.times[-1], self.times[self.index] + max(0.0, float(delta_s)))
        return int(np.searchsorted(self.times, target, side="left"))

    def local_increment_p99(self, delta_s: float) -> float:
        cache_key = round(float(delta_s), 6)
        if cache_key in self._increment_p99_cache:
            return self._increment_p99_cache[cache_key]
        increments = []
        for i, t in enumerate(self.times):
            j = min(len(self.times) - 1, int(np.searchsorted(self.times, t + delta_s, side="left")))
            increments.append(float(np.max(np.abs(self.q[j] - self.q[i]))))
        value = float(np.percentile(increments, 99))
        self._increment_p99_cache[cache_key] = value
        return value

    def state_after(self, delta_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target = min(self.times[-1], self.times[self.index] + max(0.0, float(delta_s)))
        return self.state_at(target)

    def state_at(self, absolute_time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate one absolute reference state without mutating progress."""
        target = float(np.clip(absolute_time_s, self.times[0], self.times[-1]))
        return tuple(
            np.asarray([np.interp(target, self.times, values[:, j]) for j in range(6)], dtype=np.float64)
            for values in (self.q, self.qd, self.qdd)
        )

    def remainder_after(self, absolute_time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a zero-based, endpoint-inclusive joint remainder."""
        target = float(np.clip(absolute_time_s, self.times[0], self.times[-1]))
        q0 = np.asarray([np.interp(target, self.times, self.q[:, j]) for j in range(6)])
        later = np.flatnonzero(self.times > target + 1.0e-9)
        times = np.r_[target, self.times[later]] - target
        qs = np.vstack([q0, self.q[later]])
        if len(times) == 1:
            times = np.asarray([0.0, 0.01])
            qs = np.vstack([q0, q0])
        return times, qs, np.asarray([np.interp(target, self.times, self.qd[:, j]) for j in range(6)])


def future_reference_sphere_distance(
    surface_model: RobotSurfaceModel,
    reference: RecordedReference,
    spheres: list[RiskSphere],
    *,
    density: str,
) -> dict[str, Any]:
    """Evaluate each future obstacle sphere against q_ref(t + sphere.tau)."""
    best = {"distance": math.inf, "link": None, "object_id": None, "tau": None}
    surfaces_by_tau: dict[float, dict[str, np.ndarray]] = {}
    for sphere in spheres:
        tau = float(sphere.tau)
        if tau not in surfaces_by_tau:
            q_future, _, _ = reference.state_after(tau)
            surfaces_by_tau[tau] = surface_model.surface_by_link(q_future, density=density)
        center = np.asarray(sphere.center, dtype=np.float64)
        for link, surface in surfaces_by_tau[tau].items():
            if len(surface) == 0:
                continue
            distance = float(cKDTree(surface).query(center, k=1)[0] - sphere.radius)
            if distance < best["distance"]:
                best = {"distance": distance, "link": link, "object_id": int(sphere.object_id), "tau": tau}
    return best


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_trial_dir(args: argparse.Namespace) -> Path:
    cfg = SCENARIOS[args.scene]
    return args.output.resolve() / "trials" / f"{args.scene}_{cfg['name']}_r{args.repeat:02d}"


def estimate_qd(history: list[tuple[float, np.ndarray]]) -> np.ndarray:
    if len(history) < 2:
        return np.zeros(6)
    t1, q1 = history[-1]
    for t0, q0 in reversed(history[:-1]):
        dt = t1 - t0
        if dt > 1.0e-3:
            return (q1 - q0) / dt
    return np.zeros(6)


def track_geometry(obj: Any, clusters: list[Any], fallback: float) -> dict[str, Any]:
    """Return center, velocity and radius from one track plus association audit."""
    center = np.asarray(obj.center, dtype=np.float64)
    velocity = np.asarray(obj.velocity, dtype=np.float64)
    track_radius = float(getattr(obj, "radius", fallback) or fallback)
    distances = [float(np.linalg.norm(np.asarray(c.center, dtype=np.float64) - center)) for c in clusters]
    cluster_index = None if not distances else int(np.argmin(distances))
    cluster_center = None if cluster_index is None else np.asarray(clusters[cluster_index].center, dtype=np.float64)
    raw_radius = track_radius if cluster_index is None else float(raw_cluster_geometry(clusters[cluster_index])["radius"])
    association_error = math.inf if cluster_center is None else float(np.linalg.norm(center - cluster_center))
    return {
        "track_id": object_track_id(obj),
        "center": center,
        "velocity": velocity,
        "speed": float(np.linalg.norm(velocity)),
        "raw_radius": raw_radius,
        "track_radius": track_radius,
        "inflated_radius": max(track_radius, raw_radius, float(fallback)),
        "associated_cluster_index": cluster_index,
        "associated_cluster_center": cluster_center,
        "association_error_m": association_error,
    }


def raw_cluster_geometry(cluster: Any) -> dict[str, Any]:
    points = np.asarray(cluster.points, dtype=np.float64)
    center = np.asarray(cluster.center, dtype=np.float64)
    if len(points) == 0:
        return {"center": center, "point_count": 0, "radius": math.inf, "bbox": np.full(3, math.inf)}
    distances = np.linalg.norm(points - center[None, :], axis=1)
    return {
        "center": center,
        "point_count": int(len(points)),
        "radius": float(np.percentile(distances, 90)),
        "bbox": np.ptp(points, axis=0),
    }


def dynamic_cluster_inputs(clusters: list[Any], args: argparse.Namespace) -> tuple[list[Any], list[dict[str, Any]]]:
    """Return all external clusters; radius is diagnostic geometry, not identity."""
    selected: list[Any] = []
    audits: list[dict[str, Any]] = []
    for index, cluster in enumerate(clusters):
        geometry = raw_cluster_geometry(cluster)
        radius_in_legacy_band = args.dynamic_radius_min_m <= geometry["radius"] <= args.dynamic_radius_max_m
        audits.append({"cluster_index": index, "accepted": True, "radius_in_legacy_band": radius_in_legacy_band, **geometry})
        selected.append(cluster)
    return selected, audits


def update_dynamic_track_validity(
    stable: list[Any],
    clusters: list[Any],
    speed_history: dict[int, list[tuple[float, np.ndarray]]],
    valid_streak: dict[int, int],
    args: argparse.Namespace,
    dynamic_state: dict[int, bool] | None = None,
    low_speed_streak: dict[int, int] | None = None,
    timestamp: float | None = None,
    filtered_velocity_state: dict[int, np.ndarray] | None = None,
) -> tuple[list[Any], dict[int, dict[str, Any]]]:
    """Gate tracks using 5-sample least-squares velocity and asymmetric EMA.

    Position/geometry remains the newest RGB-D measurement.  Only velocity is
    filtered: acceleration follows quickly (alpha=0.65), while deceleration
    decays conservatively (alpha=0.25).  The filtered velocity is used by the
    dynamic gate and copied into prediction-ready objects so STRO, Fresh and
    Fast share one velocity semantic.
    """
    dynamic_state = {} if dynamic_state is None else dynamic_state
    low_speed_streak = {} if low_speed_streak is None else low_speed_streak
    filtered_velocity_state = {} if filtered_velocity_state is None else filtered_velocity_state
    valid: list[Any] = []
    audits: dict[int, dict[str, Any]] = {}
    active_ids: set[int] = set()
    for obj in stable:
        geometry = track_geometry(obj, clusters, args.default_obstacle_radius_m)
        track_id = geometry["track_id"]
        if track_id is None:
            continue
        active_ids.add(track_id)
        sample_time = float(timestamp if timestamp is not None else getattr(obj, "timestamp", 0.0))
        history = speed_history.setdefault(track_id, [])
        history.append((sample_time, np.asarray(geometry["center"], dtype=np.float64).copy()))
        del history[:-args.dynamic_speed_window]
        elapsed = history[-1][0] - history[0][0] if len(history) >= 2 else 0.0
        raw_window_velocity = (
            (history[-1][1] - history[0][1]) / elapsed
            if elapsed > 1.0e-6 else np.zeros(3, dtype=np.float64)
        )
        raw_window_speed = (
            float(np.linalg.norm(history[-1][1] - history[0][1]) / elapsed)
            if elapsed > 1.0e-6 else 0.0
        )
        if len(history) >= 3 and elapsed > 1.0e-6:
            times = np.asarray([item[0] for item in history], dtype=np.float64)
            positions = np.asarray([item[1] for item in history], dtype=np.float64)
            centered = times - float(np.mean(times))
            denominator = float(np.dot(centered, centered))
            ls_velocity = (
                np.sum(centered[:, None] * (positions - np.mean(positions, axis=0)), axis=0)
                / denominator
                if denominator > 1.0e-12
                else raw_window_velocity
            )
        else:
            ls_velocity = raw_window_velocity
        previous_filtered = filtered_velocity_state.get(track_id)
        if previous_filtered is None or not np.all(np.isfinite(previous_filtered)):
            filtered_velocity = np.asarray(ls_velocity, dtype=np.float64)
            ema_alpha = 1.0
        else:
            alpha_up = 0.65
            alpha_down = 0.25
            ema_alpha = alpha_up if np.linalg.norm(ls_velocity) >= np.linalg.norm(previous_filtered) else alpha_down
            filtered_velocity = (
                ema_alpha * np.asarray(ls_velocity, dtype=np.float64)
                + (1.0 - ema_alpha) * np.asarray(previous_filtered, dtype=np.float64)
            )
        filtered_velocity_state[track_id] = filtered_velocity.copy()
        window_velocity = filtered_velocity
        window_speed = float(np.linalg.norm(window_velocity))
        exit_speed = float(getattr(args, "dynamic_exit_speed_m_s", 0.04))
        exit_frames = int(getattr(args, "dynamic_exit_streak_frames", 3))
        if dynamic_state.get(track_id, False):
            low_speed_streak[track_id] = low_speed_streak.get(track_id, 0) + 1 if window_speed < exit_speed else 0
            if low_speed_streak[track_id] >= exit_frames:
                dynamic_state[track_id] = False
        elif len(history) >= args.dynamic_speed_window and window_speed >= args.min_dynamic_trigger_speed_m_s:
            dynamic_state[track_id] = True
            low_speed_streak[track_id] = 0
        checks = {
            "age_ok": int(getattr(obj, "age", 0)) >= args.min_track_age,
            "speed_history_ready": len(history) >= args.dynamic_speed_window,
            "speed_ok": dynamic_state.get(track_id, False),
            "association_ok": geometry["association_error_m"] <= args.max_track_cluster_association_m,
        }
        instant_valid = bool(all(checks.values()))
        valid_streak[track_id] = valid_streak.get(track_id, 0) + 1 if instant_valid else 0
        is_valid = instant_valid and valid_streak[track_id] >= args.dynamic_valid_streak_frames
        audit = {
            **geometry,
            "risk_radius_m": geometry["inflated_radius"]
            + float(getattr(args, "prediction_margin_m", 0.0))
            + float(getattr(args, "prediction_uncertainty_m", 0.0)),
            "age": int(getattr(obj, "age", 0)),
            "window_speed_m_s": window_speed,
            "window_velocity": np.asarray(window_velocity, dtype=np.float64),
            "raw_window_speed_m_s": raw_window_speed,
            "raw_window_velocity": np.asarray(raw_window_velocity, dtype=np.float64),
            "ls_velocity": np.asarray(ls_velocity, dtype=np.float64),
            "velocity_ema_alpha": float(ema_alpha),
            "filtered_velocity": np.asarray(filtered_velocity, dtype=np.float64),
            "filtered_speed_m_s": window_speed,
            # Kept as a compatibility alias for existing result readers.
            "median_speed_m_s": window_speed,
            "speed_samples": len(history),
            "valid_streak": valid_streak[track_id],
            "dynamic_state": dynamic_state.get(track_id, False),
            "prediction_ready": instant_valid,
            "valid": is_valid,
            "checks": checks,
            "block_reasons": [name for name, ok in checks.items() if not ok]
            + ([] if valid_streak[track_id] >= args.dynamic_valid_streak_frames else ["valid_streak_not_ready"]),
        }
        audits[track_id] = audit
        if is_valid:
            valid.append(obj)
    return valid, audits


def make_prediction_ready_objects(objects: list[Any], audits: dict[int, dict[str, Any]]) -> list[Any]:
    """Snapshot prediction-ready tracks with the same window velocity used by gating."""
    ready: list[Any] = []
    for obj in objects:
        track_id = object_track_id(obj)
        audit = audits.get(track_id)
        if audit is None or not audit["prediction_ready"]:
            continue
        snapshot = copy.copy(obj)
        snapshot.center = np.asarray(obj.center, dtype=np.float64).copy()
        snapshot.velocity = np.asarray(audit["window_velocity"], dtype=np.float64).copy()
        ready.append(snapshot)
    return ready


def risk_track_is_eligible(
    audit: dict[str, Any] | None, *, require_dynamic_track: bool
) -> bool:
    """Separate obstacle risk eligibility from V2 dynamic classification."""
    if audit is None:
        return False
    if require_dynamic_track:
        return bool(audit.get("prediction_ready", False))
    checks = audit.get("checks", {})
    return bool(checks.get("age_ok", False) and checks.get("association_ok", False))


def build_runtime_risk_spheres(
    *,
    stable_objects: list[Any],
    prediction_tracks: list[Any],
    dynamic_audits: dict[int, dict[str, Any]],
    clusters: list[Any],
    args: argparse.Namespace,
    safety: dict[str, Any],
) -> list[RiskSphere]:
    """Dispatch STRO prediction while preserving the exact V2 default path."""
    if callable(RISK_SPHERE_PREDICTOR):
        return RISK_SPHERE_PREDICTOR(
            stable_objects=stable_objects,
            prediction_tracks=prediction_tracks,
            dynamic_audits=dynamic_audits,
            clusters=[
                np.asarray(cluster.points, dtype=np.float64) for cluster in clusters
            ],
            args=args,
            safety=safety,
        )
    return predict_risk_spheres(
        prediction_tracks,
        horizon=args.prediction_horizon_s,
        step=args.prediction_step_s,
        margin=args.prediction_margin_m,
        uncertainty=args.prediction_uncertainty_m,
        static_speed_threshold=float(
            safety.get("prediction_static_speed_threshold", 0.08)
        ),
        static_margin=float(safety.get("prediction_static_margin", 0.0)),
        velocity_radius_scale=float(
            safety.get("prediction_velocity_radius_scale", 0.1)
        ),
        already_classified=True,
    )


def object_track_id(obj: Any) -> int | None:
    for name in ("object_id", "id", "track_id"):
        value = getattr(obj, name, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                return None
    return None


def select_stable_object(stable: list[Any], predicted_best: dict[str, Any], risk_spheres: list[RiskSphere]) -> Any:
    if not stable:
        raise RuntimeError("cannot select stable object from an empty track list")
    target_id = predicted_best.get("object_id")
    if target_id is not None:
        for obj in stable:
            if object_track_id(obj) == int(target_id):
                return obj
    target_center = None
    if target_id is not None:
        for sphere in risk_spheres:
            if int(sphere.object_id) == int(target_id):
                target_center = np.asarray(sphere.center, dtype=np.float64)
                break
    if target_center is None:
        return max(stable, key=lambda obj: int(getattr(obj, "age", 0)))
    return min(stable, key=lambda obj: float(np.linalg.norm(np.asarray(obj.center, dtype=np.float64) - target_center)))


def make_local_reference(
    q_now: np.ndarray,
    qd_now: np.ndarray,
    args: argparse.Namespace,
    *,
    reference_goal: tuple[np.ndarray, np.ndarray, np.ndarray],
):
    horizon = float(args.local_horizon_s)
    segments = int(args.local_segments)
    q_goal, qd_goal, qdd_goal = (np.asarray(value, dtype=np.float64) for value in reference_goal)
    if np.linalg.norm(q_goal - q_now) < args.min_local_motion_rad:
        raise RuntimeError("recorded reference has insufficient local motion; refusing a synthetic probe candidate")
    qdd = np.zeros(6)
    head = NUBSTrajectory6D.make_boundary_state(q_now, qd_now, qdd)
    tail = NUBSTrajectory6D.make_boundary_state(q_goal, qd_goal, qdd_goal)
    durations = np.full(segments, horizon / segments, dtype=np.float64)
    p_inner = NUBSTrajectory6D.linear_inner_points(q_now, q_goal, durations)
    return head, tail, durations, p_inner, q_goal


def clearance_guided_lateral_warm_start(
    model: RobotSurfaceModel,
    p_inner: np.ndarray,
    durations: np.ndarray,
    obstacle_center: np.ndarray,
    *,
    offset_m: float,
    tcp_link: str = "gripper_base_link",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Add one smooth tabletop-lateral TCP detour to a linear warm start.

    The task direction is Y, so the lateral direction is signed base X.  A
    damped translational Jacobian maps the requested midpoint displacement to
    joint space; a sine envelope keeps the fixed head/tail unchanged.
    """
    points = np.asarray(p_inner, dtype=np.float64).copy()
    if len(points) == 0 or offset_m <= 0.0:
        return points, {"mode": "linear", "reason": "no_internal_points_or_zero_offset"}
    midpoint_index = int(np.argmin(np.abs(np.cumsum(durations)[:-1] / np.sum(durations) - 0.5)))
    q_mid = points[midpoint_index]
    joint_values = {name: float(q_mid[i]) for i, name in enumerate(model.joint_names)}
    tcp = np.asarray(model.urdf.link_transforms(joint_values)[tcp_link][:3, 3], dtype=np.float64)
    center = np.asarray(obstacle_center, dtype=np.float64)
    sign_x = -1.0 if center[0] >= tcp[0] else 1.0
    desired = np.asarray([sign_x * float(offset_m), 0.0, 0.0], dtype=np.float64)
    jacobian = model.point_jacobian(q_mid, tcp_link, np.zeros(3, dtype=np.float64))
    damping = 1.0e-3
    delta_q = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(3), desired)
    # This is an initializer, not an unconstrained command.  Keep it inside a
    # modest joint-space envelope; the unchanged motion verifier remains final.
    peak = float(np.max(np.abs(delta_q)))
    if peak > 0.10:
        delta_q *= 0.10 / peak
    ratios = np.cumsum(durations)[:-1] / np.sum(durations)
    envelope = np.sin(np.pi * ratios)
    points += envelope[:, None] * delta_q[None, :]
    achieved = jacobian @ delta_q
    return points, {
        "mode": "clearance_guided_lateral",
        "tcp_link": tcp_link,
        "nominal_midpoint_tcp_m": tcp.tolist(),
        "obstacle_center_m": center.tolist(),
        "requested_tcp_offset_m": desired.tolist(),
        "linearized_tcp_offset_m": achieved.tolist(),
        "joint_seed_delta_rad": delta_q.tolist(),
        "joint_seed_delta_max_rad": float(np.max(np.abs(delta_q))),
    }


def make_rejoin_bridge(
    repair_trajectory: NUBSTrajectory6D,
    rejoin_state: tuple[np.ndarray, np.ndarray, np.ndarray],
    duration_s: float,
    *,
    segments: int = 3,
) -> NUBSTrajectory6D:
    """Create a C2 bridge from an elastic repair tail to a later reference state."""
    if duration_s <= 0.0 or segments < 1:
        raise ValueError("rejoin bridge requires positive duration and at least one segment")
    q_goal, qd_goal, qdd_goal = (np.asarray(value, dtype=np.float64) for value in rejoin_state)
    head = repair_trajectory.tail_state
    tail = NUBSTrajectory6D.make_boundary_state(q_goal, qd_goal, qdd_goal)
    durations = np.full(int(segments), float(duration_s) / int(segments), dtype=np.float64)
    inner = NUBSTrajectory6D.linear_inner_points(head[:, 0], q_goal, durations)
    return NUBSTrajectory6D().generate(inner, head, tail, durations)


def save_trajectory_csv(path: Path, trajectory: NUBSTrajectory6D, *, dt: float = 0.01) -> None:
    samples = trajectory.dense_sample(dt)
    rows: list[dict[str, Any]] = []
    for i, t in enumerate(samples.times):
        rows.append(
            {
                "t_s": f"{float(t):.6f}",
                **{f"q{j+1}_rad": f"{samples.q[i, j]:.8f}" for j in range(6)},
                **{f"qd{j+1}_rad_s": f"{samples.qd[i, j]:.8f}" for j in range(6)},
                **{f"qdd{j+1}_rad_s2": f"{samples.qdd[i, j]:.8f}" for j in range(6)},
            }
        )
    write_csv(path, rows, ["t_s", *[f"q{j+1}_rad" for j in range(6)], *[f"qd{j+1}_rad_s" for j in range(6)], *[f"qdd{j+1}_rad_s2" for j in range(6)]])


def save_dynamic_risk_profile(path: Path, trajectory: Any, evaluator: Any, forecast: Any, *, density: str, dt: float) -> None:
    """Persist a named distance profile so STRO/active/verifier distances are not conflated."""
    times = np.linspace(0.0, trajectory.total_duration, max(2, int(np.ceil(trajectory.total_duration / dt)) + 1))
    rows = []
    for tau in times:
        risk = evaluator.configuration(trajectory.evaluate(float(tau)), forecast, float(tau), density=density, with_gradient=False)
        occupancy = forecast.occupancy_at(float(tau))
        radius = math.nan if not occupancy.spheres else max(float(sphere.radius) for sphere in occupancy.spheres)
        rows.append(
            {
                "tau": f"{float(tau):.6f}",
                "distance_m": "" if not np.isfinite(risk.min_distance) else f"{float(risk.min_distance):.9f}",
                "nearest_link": risk.nearest_link or "",
                "surface_density": density,
                "forecast_radius": "" if not np.isfinite(radius) else f"{radius:.9f}",
            }
        )
    write_csv(path, rows, ["tau", "distance_m", "nearest_link", "surface_density", "forecast_radius"])


def new_dynamic_audit_buffers():
    """Create per-run mutable audit buffers (kept out of trajectory helpers)."""
    return [], [], [], None, {}


class AuditVisualizer:
    """Small audit-only Open3D viewer built from the shared geometry helpers."""

    MAX_CLUSTERS = 12

    def __init__(self, *, show_filtered: bool, show_noise: bool) -> None:
        import open3d as o3d

        self.o3d = o3d
        self.show_filtered = bool(show_filtered)
        self.show_noise = bool(show_noise)
        self.view_initialized = False
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="6.5.3 dynamic audit clusters", width=1280, height=800)
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25))
        render = self.vis.get_render_option()
        render.background_color = np.asarray([0.04, 0.04, 0.04])
        render.point_size = 3.0

        def point_cloud(color):
            cloud = o3d.geometry.PointCloud()
            cloud.paint_uniform_color(color)
            self.vis.add_geometry(cloud)
            return cloud

        def line_set():
            lines = o3d.geometry.LineSet()
            # Open3D warns while adding a completely empty LineSet.  Seed it
            # with a zero-length line; the first update replaces this data.
            lines.points = o3d.utility.Vector3dVector(np.zeros((2, 3)))
            lines.lines = o3d.utility.Vector2iVector(np.asarray([[0, 1]], dtype=np.int32))
            self.vis.add_geometry(lines)
            return lines

        self.robot = point_cloud([0.85, 0.15, 0.15])
        self.valid = point_cloud([0.15, 0.75, 0.20])
        self.filtered = point_cloud([0.45, 0.45, 0.45])
        self.noise = point_cloud([0.80, 0.20, 0.80])
        self.plane = point_cloud([0.15, 0.35, 0.90])
        self.centers = point_cloud([1.0, 1.0, 0.0])
        self.obb = [line_set() for _ in range(self.MAX_CLUSTERS)]
        self.spheres = [line_set() for _ in range(self.MAX_CLUSTERS)]

    def update(self, robot_points, cluster_result, clusters, dynamic_audits) -> bool:
        o3d = self.o3d
        self.robot.points = o3d.utility.Vector3dVector(np.asarray(robot_points))
        self.vis.update_geometry(self.robot)

        all_points = []
        all_colors = []
        centers = []
        for index, cluster in enumerate(clusters[: self.MAX_CLUSTERS]):
            points = np.asarray(cluster.points, dtype=np.float64)
            center = np.asarray(cluster.center, dtype=np.float64)
            matching = [
                audit for audit in dynamic_audits.values()
                if audit.get("associated_cluster_center") is not None
                and np.linalg.norm(np.asarray(audit["associated_cluster_center"]) - center) < 1.0e-6
            ]
            prediction_ready = any(bool(audit.get("prediction_ready")) for audit in matching)
            dynamic_valid = any(bool(audit.get("valid")) for audit in matching)
            color = [1.0, 0.15, 0.05] if prediction_ready else ([1.0, 0.65, 0.0] if dynamic_valid else [0.1, 0.8, 0.2])
            all_points.append(points)
            all_colors.append(np.repeat(np.asarray(color)[None, :], len(points), axis=0))
            centers.append(center)
            geometry = raw_cluster_geometry(cluster)
            for target, source in (
                (self.obb[index], create_obb_wireframe(points, color=color)),
                (self.spheres[index], create_sphere_wireframe(center, geometry["radius"], color=color)),
            ):
                target.points = source.points
                target.lines = source.lines
                target.colors = source.colors
                self.vis.update_geometry(target)
        for index in range(len(clusters), self.MAX_CLUSTERS):
            for target in (self.obb[index], self.spheres[index]):
                target.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
                target.lines = o3d.utility.Vector2iVector(np.empty((0, 2), dtype=np.int32))
                self.vis.update_geometry(target)
        self.valid.points = o3d.utility.Vector3dVector(np.vstack(all_points) if all_points else np.empty((0, 3)))
        self.valid.colors = o3d.utility.Vector3dVector(np.vstack(all_colors) if all_colors else np.empty((0, 3)))
        self.centers.points = o3d.utility.Vector3dVector(np.asarray(centers) if centers else np.empty((0, 3)))
        self.vis.update_geometry(self.valid)
        self.vis.update_geometry(self.centers)

        layers = (
            (self.filtered, cluster_result.filtered_out_points if self.show_filtered else np.empty((0, 3))),
            (self.noise, cluster_result.noise_points if self.show_noise else np.empty((0, 3))),
            (self.plane, cluster_result.plane_points),
        )
        for cloud, points in layers:
            cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
            self.vis.update_geometry(cloud)
        # The window is created before the first camera frame, when all point
        # clouds are empty.  Fit the camera once real geometry arrives;
        # otherwise Open3D keeps looking only at the origin and appears black.
        if not self.view_initialized:
            visible_count = len(np.asarray(robot_points)) + sum(len(np.asarray(cluster.points)) for cluster in clusters)
            if visible_count > 0:
                self.vis.reset_view_point(True)
                self.view_initialized = True
        alive = bool(self.vis.poll_events())
        self.vis.update_renderer()
        return alive

    def close(self) -> None:
        self.vis.destroy_window()


def save_anomalous_audit_clusters(
    trial_dir: Path,
    frame_index: int,
    timestamp: float,
    clusters: list[Any],
    dynamic_audits: dict[int, dict[str, Any]],
    *,
    max_bbox_m: float,
    max_radius_m: float,
) -> int:
    """Persist exact points for oversized audit clusters without changing filtering."""
    saved = 0
    output = trial_dir / "anomalous_clusters"
    for cluster_index, cluster in enumerate(clusters):
        geometry = raw_cluster_geometry(cluster)
        if float(np.max(geometry["bbox"])) <= max_bbox_m and geometry["radius"] <= max_radius_m:
            continue
        track_id = -1
        best_error = math.inf
        for candidate_id, audit in dynamic_audits.items():
            associated = audit.get("associated_cluster_center")
            if associated is None:
                continue
            error = float(np.linalg.norm(np.asarray(associated) - geometry["center"]))
            if error < best_error:
                best_error = error
                track_id = int(candidate_id)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output / f"cluster_points_frame{frame_index:04d}_cluster{cluster_index:02d}.npz",
            points=np.asarray(cluster.points, dtype=np.float64),
            center=np.asarray(geometry["center"], dtype=np.float64),
            bbox=np.asarray(geometry["bbox"], dtype=np.float64),
            radius_m=np.asarray(geometry["radius"], dtype=np.float64),
            frame=np.asarray(frame_index, dtype=np.int64),
            cluster_index=np.asarray(cluster_index, dtype=np.int64),
            track_id=np.asarray(track_id, dtype=np.int64),
            association_error_m=np.asarray(best_error, dtype=np.float64),
            timestamp=np.asarray(timestamp, dtype=np.float64),
        )
        saved += 1
    return saved


def load_fast_candidate_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    qs: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            times.append(float(row["t_s"]))
            qs.append([float(row[f"q{i}_rad"]) for i in range(1, 7)])
    if len(qs) < 2:
        raise RuntimeError(f"too few trajectory rows in {path}")
    return np.asarray(times, dtype=np.float64), np.asarray(qs, dtype=np.float64)


def reconstruct_saved_nubs_candidate(path: Path, *, segments: int) -> NUBSTrajectory6D:
    """Reconstruct a saved NUBS candidate from boundary samples for offline replay."""
    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
    if len(rows) < 2 or segments < 1:
        raise ValueError("saved candidate needs at least two rows and one segment")
    times = np.asarray([float(row["t_s"]) for row in rows], dtype=np.float64)
    q = np.asarray([[float(row[f"q{i}_rad"]) for i in range(1, 7)] for row in rows], dtype=np.float64)
    qd = np.asarray([[float(row[f"qd{i}_rad_s"]) for i in range(1, 7)] for row in rows], dtype=np.float64)
    qdd = np.asarray([[float(row[f"qdd{i}_rad_s2"]) for i in range(1, 7)] for row in rows], dtype=np.float64)
    boundary_times = np.linspace(float(times[0]), float(times[-1]), int(segments) + 1)
    boundary_q = np.column_stack([np.interp(boundary_times, times, q[:, j]) for j in range(6)])
    durations = np.diff(boundary_times)
    head = NUBSTrajectory6D.make_boundary_state(q[0], qd[0], qdd[0])
    tail = NUBSTrajectory6D.make_boundary_state(q[-1], qd[-1], qdd[-1])
    return NUBSTrajectory6D().generate(boundary_q[1:-1], head, tail, durations)


def clearance_profile_summary(
    trajectory: Any,
    evaluator: Any,
    forecast: Any,
    *,
    step_s: float = 0.04,
    guide_horizon_s: float | None = None,
) -> dict[str, Any]:
    """Summarize closest approach without imposing a monotonic-distance gate.

    The minimum clearance remains the only geometric acceptance quantity here.
    End-of-segment and longer-horizon values are diagnostics/soft guidance only;
    they describe whether a local segment has started to pass the swept obstacle
    rather than merely postponing the closest approach to its tail.
    """
    duration = float(trajectory.total_duration)
    times = np.unique(
        np.r_[np.arange(0.0, duration + 0.5 * step_s, step_s), duration]
    )
    rows: list[dict[str, Any]] = []
    for tau in times:
        risk = evaluator.configuration(
            trajectory.evaluate(float(tau)),
            forecast,
            float(tau),
            density="medium",
            with_gradient=False,
        )
        rows.append(
            {
                "tau_s": float(tau),
                "distance_m": float(risk.min_distance),
                "nearest_link": risk.nearest_link,
            }
        )
    minimum = min(rows, key=lambda row: row["distance_m"])
    end = rows[-1]
    guide = None
    if guide_horizon_s is not None and float(guide_horizon_s) >= duration:
        q_end = trajectory.evaluate(duration)
        guide_risk = evaluator.configuration(
            q_end,
            forecast,
            float(guide_horizon_s),
            density="medium",
            with_gradient=False,
        )
        guide = {
            "horizon_s": float(guide_horizon_s),
            "clearance_m": float(guide_risk.min_distance),
            "nearest_link": guide_risk.nearest_link,
        }
    return {
        "profile": rows,
        "min_clearance_m": float(minimum["distance_m"]),
        "min_tau_s": float(minimum["tau_s"]),
        "min_nearest_link": minimum["nearest_link"],
        "end_clearance_m": float(end["distance_m"]),
        "end_minus_min_clearance_m": float(end["distance_m"] - minimum["distance_m"]),
        "min_tau_fraction": float(minimum["tau_s"] / max(duration, 1e-9)),
        "closest_approach_before_tail": bool(minimum["tau_s"] < 0.8 * duration),
        "guide": guide,
    }


class TimeScaledTrajectory6D:
    """Expose one geometric trajectory on an explicitly scaled physical clock."""

    def __init__(self, source: Any, execution_duration_s: float) -> None:
        source_duration = float(source.total_duration)
        if source_duration <= 0.0 or execution_duration_s <= 0.0:
            raise ValueError("trajectory durations must be positive")
        self.source = source
        self.source_duration = source_duration
        self.total_duration = float(execution_duration_s)
        self.time_scale = self.total_duration / self.source_duration

    def evaluate(self, time_s: float, derivative_order: int = 0) -> np.ndarray:
        source_time = float(np.clip(time_s, 0.0, self.total_duration)) / self.time_scale
        return np.asarray(self.source.evaluate(source_time, derivative_order), dtype=np.float64) / (
            self.time_scale ** derivative_order
        )

    def sample(self, times: np.ndarray, max_derivative: int = 3) -> TrajectorySamples:
        values = np.asarray(times, dtype=np.float64)
        source_samples = self.source.sample(values / self.time_scale, max_derivative=max_derivative)
        return TrajectorySamples(
            times=values.copy(),
            q=source_samples.q,
            qd=source_samples.qd / self.time_scale,
            qdd=source_samples.qdd / (self.time_scale**2),
            jerk=source_samples.jerk / (self.time_scale**3),
        )

    def dense_sample(self, time_step: float = 0.01) -> TrajectorySamples:
        count = max(2, int(np.ceil(self.total_duration / time_step)) + 1)
        return self.sample(np.linspace(0.0, self.total_duration, count))


def gripper_base_workspace_guard(
    trajectory: Any,
    stage4_model: RobotSurfaceModel,
    *,
    min_z_m: float,
) -> dict[str, Any]:
    """Fail closed if the gripper base descends into the tabletop margin.

    This is an independent workspace guard.  The tabletop is deliberately
    removed from obstacle clustering, so CCRO clearance alone cannot protect
    against a downward arm excursion.
    """
    samples = trajectory.dense_sample(0.02)
    z_values: list[float] = []
    for q in np.asarray(samples.q, dtype=np.float64):
        joints = {name: float(q[i]) for i, name in enumerate(stage4_model.joint_names)}
        transforms = stage4_model.urdf.link_transforms(joints)
        if "gripper_base_link" not in transforms:
            return {
                "passed": False,
                "reason": "gripper_base_link_missing",
                "min_gripper_base_z_m": None,
                "threshold_m": float(min_z_m),
            }
        z_values.append(float(transforms["gripper_base_link"][2, 3]))
    minimum = min(z_values) if z_values else float("-inf")
    return {
        "passed": bool(minimum >= float(min_z_m)),
        "reason": "ok" if minimum >= float(min_z_m) else "gripper_base_below_tabletop_guard",
        "min_gripper_base_z_m": float(minimum),
        "threshold_m": float(min_z_m),
        "sample_count": len(z_values),
    }


def execution_hard_guard_distance(processor: Any, denoiser: Any, args: argparse.Namespace) -> float:
    """Measure the existing all-link raw-cloud guard during candidate playback."""
    frame = processor.process_frame()
    scene_points = np.asarray(frame.scene_points, dtype=np.float64)
    robot_points = np.asarray(frame.robot_points, dtype=np.float64)
    if denoiser is not None:
        scene_points = denoiser.filter(scene_points)
    # The raw hard guard sees the broad safety ROI (not the planning ROI) so
    # an obstacle just outside the task box is still a physical protection.
    rois = apply_two_layer_roi(scene_points, args, need_planning=False)
    plane_removal = None
    if args.remove_planes:
        plane_removal = {"enabled": True, "distance_threshold": args.plane_dist, "max_planes": args.max_planes}
    clustered = FastClusteringFilter(
        rois["safety_points"],
        robot_points,
        workspace=getattr(processor, "_workspace", None),
        plane_removal=plane_removal,
        eps=args.cluster_eps,
        min_samples=args.cluster_min_samples,
        min_points=args.cluster_min_points,
        min_volume=args.cluster_min_volume,
    )
    clusters = filter_guard_clusters(list(clustered.clusters), args)
    distance, _, _, _, _ = _find_nearest_cluster_distance_detail(robot_points, clusters, [])
    return float(distance)


def wait_for_candidate_goal_guarded(
    robot: Any,
    q_goal: np.ndarray,
    *,
    processor: Any,
    denoiser: Any,
    args: argparse.Namespace,
    goal_tolerance_rad: float,
    min_execution_wait_s: float,
    motion_timeout_s: float,
    poll_s: float,
    min_motion_rad: float,
    guard_provider: Any | None = None,
    obstacle_state_provider: Any | None = None,
    motion_monitor_provider: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray | None]:
    started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    initial = np.asarray(robot.get_joint(), dtype=np.float64)
    last = initial.copy()
    max_motion = 0.0
    minimum_guard_distance = math.inf
    startup_freshness_grace_used = False
    while time.perf_counter() - started <= motion_timeout_s:
        now = time.perf_counter()
        last = np.asarray(robot.get_joint(), dtype=np.float64)
        err = joint_error(last, q_goal)
        max_motion = max(max_motion, float(np.max(np.abs(last - initial))))
        guard_snapshot = (
            guard_provider()
            if callable(guard_provider)
            else {
                "distance_m": execution_hard_guard_distance(
                    processor, denoiser, args
                ),
                "timestamp": time.time(),
            }
        )
        guard_distance = float(guard_snapshot["distance_m"])
        obstacle_snapshot = (
            obstacle_state_provider()
            if callable(obstacle_state_provider)
            else None
        )
        motion_monitor = (
            motion_monitor_provider(
                elapsed_s=now - started,
                actual_q=last.copy(),
                obstacle_snapshot=obstacle_snapshot,
            )
            if callable(motion_monitor_provider)
            else None
        )
        minimum_guard_distance = min(minimum_guard_distance, guard_distance)
        samples.append(
            {
                "t_s": now - started,
                "actual_joint_rad": last.tolist(),
                "goal_l2_error_rad": err["l2_rad"],
                "goal_max_abs_error_rad": err["max_abs_rad"],
                "max_motion_from_start_rad": max_motion,
                "hard_guard_distance_m": guard_distance,
                "hard_guard_timestamp": guard_snapshot.get("timestamp"),
                "obstacle_state_timestamp": (
                    None
                    if obstacle_snapshot is None
                    else obstacle_snapshot.get("timestamp")
                ),
                "obstacle_state_age_s": (
                    None
                    if obstacle_snapshot is None
                    else obstacle_snapshot.get("state_age_s")
                ),
                "motion_monitor": motion_monitor,
            }
        )
        # The independent raw-cloud hard guard always has priority over the
        # predictive monitor so an actual distance violation is never
        # mislabeled as a planner-triggered stop.
        if guard_distance <= args.guided_hard_stop_m:
            stop_return = maybe_move_stop(robot)
            return (
                {
                    "reached": False,
                    "guard_stopped": True,
                    "elapsed_s": now - started,
                    "hard_guard_distance_m": guard_distance,
                    "minimum_hard_guard_distance_m": minimum_guard_distance,
                    "stop_return": stop_return,
                    "actual_joint_rad": last.tolist(),
                    "sample_count": len(samples),
                    "max_motion_from_start_rad": max_motion,
                },
                samples,
            )
        if motion_monitor is not None and not bool(motion_monitor.get("motion_safe", False)):
            # AUBO's waypoint/start handoff can make the first feedback sample
            # appear just over the 0.5 s perception watchdog despite a newer
            # persistent sequence and a very safe independent raw guard.  Allow
            # exactly one startup-only grace sample; every later stale sample
            # remains fail-closed.
            reason = motion_monitor.get("reason", "")
            final_seq = motion_monitor.get("final_precommand_state_seq")
            state_seq = motion_monitor.get("state_seq")
            startup_grace = bool(
                not startup_freshness_grace_used
                and (now - started) < 0.10
                and reason == "perception_watchdog_expired"
                and final_seq is not None
                and state_seq is not None
                and int(state_seq) > int(final_seq)
                and guard_distance > float(args.guided_hard_stop_m)
            )
            if startup_grace:
                startup_freshness_grace_used = True
                samples[-1]["motion_monitor"]["startup_freshness_handoff_grace"] = True
                continue
            stop_return = maybe_move_stop(robot)
            return (
                {
                    "reached": False,
                    "guard_stopped": False,
                    "monitor_stopped": True,
                    "monitor_stop_reason": motion_monitor.get(
                        "reason", "motion_monitor_not_safe"
                    ),
                    "replan_requested": bool(
                        motion_monitor.get("replan_requested", False)
                    ),
                    "motion_monitor": motion_monitor,
                    "elapsed_s": now - started,
                    "hard_guard_distance_m": guard_distance,
                    "minimum_hard_guard_distance_m": minimum_guard_distance,
                    "stop_return": stop_return,
                    "actual_joint_rad": last.tolist(),
                    "sample_count": len(samples),
                    "max_motion_from_start_rad": max_motion,
                },
                samples,
            )
        elapsed = now - started
        if (
            err["max_abs_rad"] <= goal_tolerance_rad
            and elapsed >= min_execution_wait_s
            and max_motion >= min_motion_rad
        ):
            return (
                {
                    "reached": True,
                    "elapsed_s": elapsed,
                    "goal_error": err,
                    "actual_joint_rad": last.tolist(),
                    "sample_count": len(samples),
                    "max_motion_from_start_rad": max_motion,
                    "min_motion_required_rad": min_motion_rad,
                    "guard_stopped": False,
                    "minimum_hard_guard_distance_m": minimum_guard_distance,
                },
                samples,
            )
        time.sleep(poll_s)
    err = joint_error(last, q_goal)
    return (
        {
            "reached": False,
            "elapsed_s": time.perf_counter() - started,
            "goal_error": err,
            "actual_joint_rad": last.tolist(),
            "sample_count": len(samples),
            "max_motion_from_start_rad": max_motion,
            "min_motion_required_rad": min_motion_rad,
            "guard_stopped": False,
            "minimum_hard_guard_distance_m": minimum_guard_distance,
        },
        samples,
    )


def candidate_tracking_metrics(
    command_times: np.ndarray,
    command_q: np.ndarray,
    feedback_samples: list[dict[str, Any]],
    *,
    minimum_motion_rad: float,
) -> dict[str, Any]:
    """Compare timestamped feedback with the authorized execution time axis."""
    if not feedback_samples:
        return {
            "requested_duration_s": float(command_times[-1] - command_times[0]),
            "observed_motion_duration_s": None,
            "duration_error_s": None,
            "tracking_rmse_rad": None,
            "tracking_max_error_rad": None,
        }
    feedback_times = np.asarray([float(row["t_s"]) for row in feedback_samples], dtype=np.float64)
    actual_q = np.asarray([row["actual_joint_rad"] for row in feedback_samples], dtype=np.float64)
    relative_command_times = np.asarray(command_times, dtype=np.float64) - float(command_times[0])
    clipped_times = np.clip(feedback_times, 0.0, float(relative_command_times[-1]))
    expected_q = np.column_stack(
        [np.interp(clipped_times, relative_command_times, command_q[:, joint]) for joint in range(command_q.shape[1])]
    )
    errors = actual_q - expected_q
    moving_indices = [
        index for index, row in enumerate(feedback_samples)
        if float(row.get("max_motion_from_start_rad", 0.0)) >= float(minimum_motion_rad)
    ]
    observed_duration = None
    if moving_indices:
        observed_duration = float(feedback_times[-1] - feedback_times[moving_indices[0]])
    requested_duration = float(relative_command_times[-1])
    return {
        "requested_duration_s": requested_duration,
        "command_to_last_feedback_duration_s": float(feedback_times[-1]),
        "observed_motion_duration_s": observed_duration,
        "duration_error_s": float(feedback_times[-1] - requested_duration),
        "tracking_rmse_rad": float(np.sqrt(np.mean(errors**2))),
        "tracking_max_error_rad": float(np.max(np.abs(errors))),
        "tracking_sample_count": int(len(feedback_samples)),
    }


def authorized_execution_timing_check(
    requested_duration_s: float,
    feedback_samples: list[dict[str, Any]],
    *,
    valid_completion_time_s: float | None,
    goal_tolerance_rad: float,
    relative_tolerance: float = 0.20,
) -> dict[str, Any]:
    """Audit the executor's valid completion, not first tolerance entry."""
    goal_hits = [
        float(row["t_s"]) for row in feedback_samples
        if float(row.get("goal_max_abs_error_rad", math.inf)) <= goal_tolerance_rad
    ]
    first_goal_s = None if not goal_hits else min(goal_hits)
    completion_s = None if valid_completion_time_s is None else float(valid_completion_time_s)
    ratio = None if completion_s is None or requested_duration_s <= 0.0 else completion_s / requested_duration_s
    accepted = bool(ratio is not None and (1.0 - relative_tolerance) <= ratio <= (1.0 + relative_tolerance))
    return {
        "accepted": accepted,
        "requested_duration_s": float(requested_duration_s),
        "first_goal_tolerance_time_s": first_goal_s,
        "valid_completion_time_s": completion_s,
        "completion_to_requested_ratio": ratio,
        # Compatibility alias retained for existing result readers.
        "actual_to_requested_ratio": ratio,
        "relative_tolerance": float(relative_tolerance),
        "reason": "timing_consistent" if accepted else "authorized_time_axis_not_followed",
    }


def execute_guarded_cartesian_reference_remainder(
    robot: Any,
    args: argparse.Namespace,
    *,
    processor: Any,
    denoiser: Any,
    target_y_m: float,
) -> dict[str, Any]:
    """Resume the formal straight reference with a timed Cartesian line move."""
    if not hasattr(robot, "movel_line"):
        raise RuntimeError("current robot .so does not expose movel_line")
    start_pose = np.asarray(robot.get_status(), dtype=np.float64)
    target_pose = start_pose.copy()
    target_pose[1] = float(target_y_m)
    distance = float(abs(target_pose[1] - start_pose[1]))
    nominal_duration = distance / float(args.line_velocity_m_s)
    timeout = max(float(args.motion_timeout_s), 1.8 * nominal_duration + 5.0)
    require_confirmation(
        bool(args.candidate_execute_confirm),
        "Cartesian reference remainder is ready. Confirm clear workspace and emergency stop before resuming to goal.",
    )
    ret = robot.movel_line(
        target_pose.tolist(), args.line_velocity_m_s, args.line_acc_m_s2, False, True
    )
    if ret not in (None, 0):
        raise RuntimeError(f"reference remainder movel_line returned {ret}")
    started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    reached_at = None
    minimum_guard = math.inf
    max_tcp_speed = 0.0
    previous_t = None
    previous_xyz = None
    while time.perf_counter() - started <= timeout:
        now = time.perf_counter()
        pose = np.asarray(robot.get_status(), dtype=np.float64)
        guard_distance = execution_hard_guard_distance(processor, denoiser, args)
        minimum_guard = min(minimum_guard, guard_distance)
        speed = 0.0
        if previous_t is not None and now > previous_t:
            speed = float(np.linalg.norm(pose[:3] - previous_xyz) / (now - previous_t))
            max_tcp_speed = max(max_tcp_speed, speed)
        error = float(np.linalg.norm(pose[:3] - target_pose[:3]))
        samples.append(
            {
                "t_s": now - started,
                "actual_pose": pose.tolist(),
                "position_error_m": error,
                "tcp_speed_m_s": speed,
                "hard_guard_distance_m": guard_distance,
            }
        )
        if guard_distance <= args.guided_hard_stop_m:
            return {
                "status": "REFERENCE_REMAINDER_HARD_GUARD_STOPPED",
                "reached": False,
                "stop_return": maybe_move_stop(robot),
                "minimum_hard_guard_distance_m": minimum_guard,
                "samples": samples,
            }
        # A sustained speed far above the commanded Cartesian limit is fail-closed.
        if len(samples) >= 3 and speed > 1.75 * args.line_velocity_m_s:
            return {
                "status": "REFERENCE_REMAINDER_OVERSPEED_STOPPED",
                "reached": False,
                "stop_return": maybe_move_stop(robot),
                "observed_tcp_speed_m_s": speed,
                "commanded_tcp_speed_m_s": args.line_velocity_m_s,
                "samples": samples,
            }
        if error <= args.pose_tolerance_m:
            reached_at = now if reached_at is None else reached_at
            if now - reached_at >= args.settle_s:
                elapsed = now - started
                timing_ratio = elapsed / nominal_duration if nominal_duration > 1.0e-9 else 1.0
                return {
                    "status": "COMPLETED_GUARDED_CARTESIAN_REFERENCE_REMAINDER",
                    "reached": True,
                    "start_pose": start_pose.tolist(),
                    "target_pose": target_pose.tolist(),
                    "final_pose": pose.tolist(),
                    "distance_m": distance,
                    "commanded_velocity_m_s": args.line_velocity_m_s,
                    "commanded_acceleration_m_s2": args.line_acc_m_s2,
                    "nominal_constant_speed_duration_s": nominal_duration,
                    "elapsed_s": elapsed,
                    "elapsed_to_nominal_ratio": timing_ratio,
                    "max_observed_tcp_speed_m_s": max_tcp_speed,
                    "minimum_hard_guard_distance_m": minimum_guard,
                    "samples": samples,
                }
        else:
            reached_at = None
        previous_t = now
        previous_xyz = pose[:3].copy()
        time.sleep(args.poll_s)
    return {
        "status": "REFERENCE_REMAINDER_TIMEOUT",
        "reached": False,
        "stop_return": maybe_move_stop(robot),
        "minimum_hard_guard_distance_m": minimum_guard,
        "samples": samples,
    }


def save_joint_waypoint_csv(path: Path, times: np.ndarray, qs: np.ndarray) -> None:
    rows = []
    for i, t_s in enumerate(times):
        rows.append({"t_s": f"{float(t_s):.8f}", **{f"q{j + 1}_rad": f"{float(qs[i, j]):.8f}" for j in range(6)}})
    write_csv(path, rows, ["t_s", *[f"q{j + 1}_rad" for j in range(6)]])


def locate_authorized_rejoin_on_reference(
    reference: RecordedReference,
    authorized_trajectory_csv: Path,
    *,
    tolerance_rad: float = 0.01,
) -> dict[str, Any]:
    """Locate the authorized trajectory endpoint on the recorded reference."""
    _, qs = load_fast_candidate_csv(authorized_trajectory_csv)
    errors = np.max(np.abs(reference.q - qs[-1][None, :]), axis=1)
    index = int(np.argmin(errors))
    error = float(errors[index])
    if error > tolerance_rad:
        raise RuntimeError(f"authorized rejoin endpoint is not on reference: {error:.6f} rad > {tolerance_rad:.6f} rad")
    return {"index": index, "time_s": float(reference.times[index]), "max_abs_error_rad": error}


def trajectory_workspace_deviation(
    surface_model: RobotSurfaceModel,
    trajectory_csv: Path,
    reference: RecordedReference,
    reference_start_time_s: float,
    *,
    tcp_link: str = "gripper_base_link",
) -> dict[str, Any]:
    """Report candidate/reference workspace deviation; never gate execution."""
    times, qs = load_fast_candidate_csv(trajectory_csv)
    ref_q = np.vstack([
        [np.interp(min(reference.times[-1], reference_start_time_s + float(t)), reference.times, reference.q[:, j]) for j in range(6)]
        for t in times
    ])
    joint_max = float(np.max(np.abs(qs - ref_q)))
    tcp_delta = []
    body_delta = []
    for q_candidate, q_reference in zip(qs, ref_q):
        candidate_fk = surface_model.urdf.link_transforms(
            {name: float(q_candidate[i]) for i, name in enumerate(surface_model.joint_names)}
        )
        reference_fk = surface_model.urdf.link_transforms(
            {name: float(q_reference[i]) for i, name in enumerate(surface_model.joint_names)}
        )
        if tcp_link in candidate_fk and tcp_link in reference_fk:
            tcp_delta.append(float(np.linalg.norm(candidate_fk[tcp_link][:3, 3] - reference_fk[tcp_link][:3, 3])))
        common = set(candidate_fk).intersection(reference_fk)
        if common:
            body_delta.append(max(float(np.linalg.norm(candidate_fk[name][:3, 3] - reference_fk[name][:3, 3])) for name in common))
    return {
        "metric_only": True,
        "max_joint_deviation_rad": joint_max,
        "max_tcp_deviation_m": None if not tcp_delta else max(tcp_delta),
        "max_body_link_origin_deviation_m": None if not body_delta else max(body_delta),
        "tcp_link": tcp_link,
        "sample_count": int(len(times)),
    }


def audit_execution_waypoints(
    times_exec: np.ndarray,
    qs_exec: np.ndarray,
    actual_q: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Audit the exact resampled/downsampled sequence sent to the controller."""
    times_exec = np.asarray(times_exec, dtype=np.float64)
    qs_exec = np.asarray(qs_exec, dtype=np.float64)
    if len(times_exec) < 4 or qs_exec.shape[0] < 4:
        return {"passed": False, "reason": "too_few_execution_waypoints"}
    dt = np.diff(times_exec)
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0.0):
        return {"passed": False, "reason": "invalid_execution_time_axis"}
    dq = np.diff(qs_exec, axis=0)
    qd = dq / dt[:, None]
    qdd = np.diff(qd, axis=0) / (0.5 * (dt[:-1] + dt[1:]))[:, None]
    start_error = float(np.max(np.abs(qs_exec[0] - np.asarray(actual_q, dtype=np.float64))))
    checks = {
        "start_position_ok": start_error <= float(getattr(args, "candidate_start_sync_rad", 0.002)),
        "first_step_ok": float(np.max(np.abs(dq[0]))) <= float(getattr(args, "candidate_start_sync_rad", 0.002)),
        "final_step_ok": float(np.max(np.abs(dq[-1]))) <= float(getattr(args, "candidate_start_sync_rad", 0.002)),
        "start_velocity_ok": float(np.max(np.abs(qd[0]))) <= float(getattr(args, "boundary_qd_tol_rad_s", 0.03)),
        "end_velocity_ok": float(np.max(np.abs(qd[-1]))) <= float(getattr(args, "boundary_qd_tol_rad_s", 0.03)),
        "start_acceleration_ok": float(np.max(np.abs(qdd[0]))) <= float(getattr(args, "boundary_qdd_tol_rad_s2", 0.30)),
        "end_acceleration_ok": float(np.max(np.abs(qdd[-1]))) <= float(getattr(args, "boundary_qdd_tol_rad_s2", 0.30)),
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "failed_checks": [k for k, v in checks.items() if not v],
        "start_error_rad": start_error,
        "first_step_rad": float(np.max(np.abs(dq[0]))),
        "final_step_rad": float(np.max(np.abs(dq[-1]))),
        "start_qd_rad_s": float(np.max(np.abs(qd[0]))),
        "end_qd_rad_s": float(np.max(np.abs(qd[-1]))),
        "start_qdd_rad_s2": float(np.max(np.abs(qdd[0]))),
        "end_qdd_rad_s2": float(np.max(np.abs(qdd[-1]))),
        "global_max_adjacent_step_rad": float(np.max(np.abs(dq))),
        "global_max_qd_rad_s": float(np.max(np.abs(qd))),
        "global_max_qdd_rad_s2": float(np.max(np.abs(qdd))),
        "waypoint_count": int(len(qs_exec)),
    }


def execute_authorized_trajectory_offline_track(
    robot: Any,
    trajectory_csv: Path,
    args: argparse.Namespace,
    *,
    processor: Any,
    denoiser: Any,
    playback_duration_s: float | None = None,
    controller_period_s: float | None = None,
    execution_label: str = "authorized trajectory",
    guard_provider: Any | None = None,
    obstacle_state_provider: Any | None = None,
    motion_monitor_provider: Any | None = None,
) -> dict[str, Any]:
    times, qs = load_fast_candidate_csv(trajectory_csv)
    source_duration = float(times[-1] - times[0])
    requested_duration = source_duration if playback_duration_s is None else float(playback_duration_s)
    expected_duration = requested_duration
    if abs(source_duration - expected_duration) > 0.02:
        raise RuntimeError(
            "authorized trajectory time axis does not match requested playback: "
            f"csv={source_duration:.3f}s requested={expected_duration:.3f}s"
        )
    waypoint_period = float(
        args.candidate_controller_waypoint_period_s if controller_period_s is None else controller_period_s
    )
    times_exec, qs_exec = resample_for_offline_track(
        times,
        qs,
        playback_duration_s=requested_duration,
        controller_period_s=waypoint_period,
    )
    times_exec, qs_exec = maybe_downsample(times_exec, qs_exec, args.candidate_max_waypoints)
    if not hasattr(robot, "offline_track_execute_joints") and not (
        hasattr(robot, "offline_track_prepare_joints") and hasattr(robot, "offline_track_start")
    ):
        raise RuntimeError("current robot .so does not expose offline_track_execute_joints")
    actual_q_for_waypoint_audit = np.asarray(robot.get_joint(), dtype=np.float64)
    min_wait = args.candidate_min_execution_wait_s
    if min_wait <= 0.0:
        min_wait = 0.90 * requested_duration

    log: dict[str, Any] = {
        "trajectory_csv": str(trajectory_csv),
        "robot_commanded": False,
        "source_trajectory_stats": trajectory_stats(times, qs),
        "execution_waypoint_stats": trajectory_stats(times_exec, qs_exec),
        "execution_label": execution_label,
        "playback_duration_s": requested_duration,
        "controller_waypoint_period_s": waypoint_period,
        "joint_velc": args.candidate_joint_velc,
        "joint_acc": args.candidate_joint_acc,
        "min_execution_wait_s": min_wait,
    }
    execution_waypoint_audit = audit_execution_waypoints(
        times_exec, qs_exec, actual_q_for_waypoint_audit, args
    )
    log["execution_waypoint_boundary_audit"] = execution_waypoint_audit
    if not bool(execution_waypoint_audit.get("passed", False)):
        log["status"] = "EXECUTION_WAYPOINT_BOUNDARY_REPLAN_REQUIRED"
        log["robot_commanded"] = False
        return log

    if not hasattr(robot, "offline_track_execute_joints"):
        raise RuntimeError("current robot .so does not expose offline_track_execute_joints")
    actual_start = actual_q_for_waypoint_audit
    start_err = joint_error(actual_start, qs_exec[0])
    log["actual_start_joint_rad"] = actual_start.tolist()
    log["candidate_start_joint_rad"] = qs_exec[0].tolist()
    log["start_error"] = start_err
    if start_err["max_abs_rad"] > args.candidate_start_tolerance_rad:
        raise RuntimeError(
            f"current joints are not near dynamic candidate start: "
            f"{start_err['max_abs_rad']:.5f} rad > {args.candidate_start_tolerance_rad:.5f} rad"
        )

    if args.candidate_execute_confirm:
        require_confirmation(
            True,
            f"{execution_label} has passed software checks. "
            "Confirm obstacle state and emergency stop, then press Enter to execute.",
        )

    # A mid-execution monitor must be live before the command barrier.  Without
    # this check, a missing/stale persistent worker is only discovered by the
    # first feedback callback, after the controller has already moved the arm.
    prearm = getattr(motion_monitor_provider, "prearm", None)
    if callable(prearm):
        prearm_result = prearm()
        log["motion_monitor_prearm"] = prearm_result
        if not bool(prearm_result.get("ready", False)):
            log["status"] = "PERSISTENT_TRACKER_NOT_READY_PRECOMMAND"
            log["motion_monitor_stop_reason"] = prearm_result.get("reason")
            return log

    command_validator = getattr(motion_monitor_provider, "command_time_revalidate", None)
    if callable(command_validator):
        command_q = np.asarray(robot.get_joint(), dtype=np.float64)
        command_validation = command_validator(actual_q=command_q)
        log["command_time_actual_start_joint_rad"] = command_q.tolist()
        log["command_time_revalidation"] = command_validation
        if not bool(command_validation.get("ready", False)):
            action = command_validation.get("action", "hold")
            log["motion_monitor_stop_reason"] = command_validation.get("reason")
            log["replan_requested"] = bool(action == "replan")
            log["status"] = (
                "COMMAND_TIME_REVALIDATION_REPLAN_REQUIRED"
                if action == "replan"
                else "COMMAND_TIME_REVALIDATION_HOLD_PRECOMMAND"
            )
            log["robot_commanded"] = False
            return log

    # The arm must be demonstrably settled before the final perception and
    # trajectory barrier.  This prevents a STOP->START race from becoming a
    # visible twitch; it never clips or alters an already verified trajectory.
    if motion_monitor_provider is not None:
        settle_qs = []
        settle_ts = []
        for _ in range(5):
            settle_ts.append(time.monotonic())
            settle_qs.append(np.asarray(robot.get_joint(), dtype=np.float64))
            time.sleep(0.02)
        settle_stack = np.stack(settle_qs, axis=0)
        settle_span = float(np.max(np.ptp(settle_stack, axis=0)))
        settle_dt = np.diff(np.asarray(settle_ts, dtype=np.float64))
        settle_qd = np.diff(settle_stack, axis=0) / settle_dt[:, None]
        settle_qd_max = float(np.max(np.abs(settle_qd[-2:])))
        settle_limit = min(0.001, 0.5 * float(getattr(args, "candidate_start_sync_rad", 0.002)))
        settle_qd_limit = 0.02
        settle_audit = {
            "passed": settle_span <= settle_limit and settle_qd_max <= settle_qd_limit,
            "sample_count": len(settle_qs),
            "duration_s": settle_ts[-1] - settle_ts[0],
            "max_span_rad": settle_span,
            "span_limit_rad": settle_limit,
            "last_qd_max_rad_s": settle_qd_max,
            "qd_limit_rad_s": settle_qd_limit,
            "q_latest": settle_qs[-1].tolist(),
        }
        log["robot_settle_audit"] = settle_audit
        if not settle_audit["passed"]:
            log["status"] = "ROBOT_NOT_SETTLED_PRECOMMAND"
            log["robot_commanded"] = False
            return log

    # Final barrier: require a state newer than command-time authorization,
    # fresh enough for startup, and a boundary-continuous trajectory.  This is
    # intentionally the last gate before the SDK startup call.
    final_barrier = getattr(motion_monitor_provider, "final_precommand_barrier", None)
    if callable(final_barrier):
        final_result = final_barrier(actual_q=np.asarray(robot.get_joint(), dtype=np.float64))
        log["final_precommand_barrier"] = final_result
        if not bool(final_result.get("ready", False)):
            action = final_result.get("action", "hold")
            log["motion_monitor_stop_reason"] = final_result.get("reason")
            log["replan_requested"] = bool(action == "replan")
            log["status"] = (
                "FINAL_PRECOMMAND_REVALIDATION_REPLAN_REQUIRED"
                if action == "replan"
                else "FINAL_PRECOMMAND_HOLD_PRECOMMAND"
            )
            log["robot_commanded"] = False
            return log

    started = time.perf_counter()
    # Prefer a split prepare/start API when the robot wrapper exposes it.  It
    # places the final barrier after waypoint upload and immediately before
    # MoveStartup.  Older SDKs expose only the combined call; for those the
    # barrier above remains the last Python-level gate available.
    if hasattr(robot, "offline_track_prepare_joints") and hasattr(robot, "offline_track_start"):
        prep_info = robot.offline_track_prepare_joints(
            qs_exec.tolist(), args.candidate_joint_velc, args.candidate_joint_acc
        )
        log["offline_track_prepare_return"] = dict(prep_info)
        if int(prep_info.get("prepare_ret", 0)) != 0:
            log["status"] = "OFFLINE_TRACK_PREPARE_FAILED"
            return log
        final_barrier_after_prepare = getattr(motion_monitor_provider, "final_precommand_barrier", None)
        if callable(final_barrier_after_prepare):
            final_result = final_barrier_after_prepare(actual_q=np.asarray(robot.get_joint(), dtype=np.float64))
            log["final_precommand_barrier_after_prepare"] = final_result
            if not bool(final_result.get("ready", False)):
                action = final_result.get("action", "hold")
                log["status"] = "FINAL_PRECOMMAND_REVALIDATION_REPLAN_REQUIRED" if action == "replan" else "FINAL_PRECOMMAND_HOLD_PRECOMMAND"
                log["robot_commanded"] = False
                return log
        ret_info = robot.offline_track_start(nonblocking=True)
    else:
        ret_info = robot.offline_track_execute_joints(
            qs_exec.tolist(), args.candidate_joint_velc, args.candidate_joint_acc, False, True, True
        )
    log["command_attempted"] = True
    log["offline_track_return"] = dict(ret_info)
    startup_ret = int(ret_info.get("startup_ret", -9999))
    log["startup_ret"] = startup_ret
    log["robot_commanded"] = bool(startup_ret == 0)
    if startup_ret != 0:
        log["status"] = "OFFLINE_TRACK_STARTUP_FAILED"
        return log
    goal_check, feedback_samples = wait_for_candidate_goal_guarded(
        robot,
        qs_exec[-1],
        processor=processor,
        denoiser=denoiser,
        args=args,
        goal_tolerance_rad=args.candidate_goal_tolerance_rad,
        min_execution_wait_s=min_wait,
        motion_timeout_s=args.candidate_motion_timeout_s,
        poll_s=args.poll_s,
        min_motion_rad=args.candidate_min_observed_motion_rad,
        guard_provider=guard_provider,
        obstacle_state_provider=obstacle_state_provider,
        motion_monitor_provider=motion_monitor_provider,
    )
    log["goal_check"] = goal_check
    log["feedback_samples"] = feedback_samples
    log["tracking_metrics"] = candidate_tracking_metrics(
        times_exec,
        qs_exec,
        feedback_samples,
        minimum_motion_rad=args.candidate_min_observed_motion_rad,
    )
    log["timing_check"] = authorized_execution_timing_check(
        requested_duration,
        feedback_samples,
        valid_completion_time_s=(float(goal_check["elapsed_s"]) if goal_check.get("reached", False) else None),
        goal_tolerance_rad=args.candidate_goal_tolerance_rad,
    )
    log["elapsed_s"] = time.perf_counter() - started
    if goal_check.get("monitor_stopped", False):
        log["status"] = "STOPPED_BY_MOTION_MONITOR"
        return log
    if not goal_check["reached"]:
        raise RuntimeError(f"dynamic candidate offline track did not reach goal: {goal_check}")
    log["status"] = (
        "COMPLETED_AUTHORIZED_TRAJECTORY_EXECUTION"
        if log["timing_check"]["accepted"]
        else "FAILED_AUTHORIZED_TRAJECTORY_TIMING"
    )
    return log


def execute_fast_candidate_offline_track(
    robot: Any,
    trajectory_csv: Path,
    args: argparse.Namespace,
    *,
    processor: Any,
    denoiser: Any,
) -> dict[str, Any]:
    """Compatibility wrapper for the calibrated 1 s local-only executor."""
    result = execute_authorized_trajectory_offline_track(
        robot,
        trajectory_csv,
        args,
        processor=processor,
        denoiser=denoiser,
        playback_duration_s=args.candidate_playback_duration_s,
        execution_label="local candidate",
    )
    if result["status"] == "COMPLETED_AUTHORIZED_TRAJECTORY_EXECUTION":
        result["status"] = "COMPLETED_DYNAMIC_CANDIDATE_EXECUTION"
    return result


def forecast_override_authorized(args: argparse.Namespace, obstacle_audit: dict[str, Any]) -> bool:
    """Permit overrides only for offline replay or non-commanding shadow diagnostics."""
    offline_ok = bool(obstacle_audit.get("offline_forecast_override_authorized", False))
    static_shadow_ok = bool(
        args.mode == "shadow"
        and obstacle_audit.get("static20_shadow_forecast_override_authorized", False)
    )
    return offline_ok or static_shadow_ok


def candidate_acceptance_contract(
    *,
    hard_safety_ready: bool,
    repair_step_ok: bool,
    clearance_gain_m: float,
    minimum_clearance_gain_m: float,
    delta_from_fast_seed_rad: float,
    minimum_candidate_delta_rad: float,
    accept_verified_seed_without_fast_step: bool,
) -> dict[str, Any]:
    """Separate execution safety from optimizer-behaviour diagnostics.

    The legacy contract requires Fast to change its input reference.  V3 may
    instead pass an already-safe generated bypass seed through the exact same
    absolute verifier.  Neither an accepted QP step, clearance gain, nor
    motion relative to the Fast seed is an additional V3 safety constraint.
    """
    diagnostic = {
        "fast_accepted_step": bool(repair_step_ok),
        "clearance_gain_meets_preference": bool(
            clearance_gain_m >= minimum_clearance_gain_m
        ),
        "motion_from_fast_seed_meets_preference": bool(
            delta_from_fast_seed_rad >= minimum_candidate_delta_rad
        ),
    }
    if accept_verified_seed_without_fast_step:
        ready = bool(hard_safety_ready)
        source = (
            "FAST_REPAIRED_BYPASS"
            if ready and repair_step_ok
            else "SAFE_BYPASS_SEED"
            if ready
            else "NO_SAFE_CANDIDATE"
        )
    else:
        ready = bool(
            hard_safety_ready
            and repair_step_ok
            and diagnostic["clearance_gain_meets_preference"]
            and diagnostic["motion_from_fast_seed_meets_preference"]
        )
        source = "LEGACY_FAST_REPAIR" if ready else "NO_SAFE_CANDIDATE"
    return {
        "local_repair_ready": ready,
        "candidate_source": source,
        "fast_extra_correction_applied": bool(repair_step_ok),
        "optimizer_diagnostics": diagnostic,
    }


def run_fast_repair(
    args: argparse.Namespace,
    stage4_config: dict[str, Any],
    stage4_model: RobotSurfaceModel,
    *,
    q_now: np.ndarray,
    qd_now: np.ndarray,
    center: np.ndarray,
    velocity: np.ndarray,
    radius: float,
    risk_links: set[str],
    trial_dir: Path,
    reference_goal: tuple[np.ndarray, np.ndarray, np.ndarray],
    rejoin_goals: list[tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]] | None,
    obstacle_audit: dict[str, Any],
    multisphere_geometry: dict[str, Any] | None = None,
    artifacts_out: dict[str, Any] | None = None,
    forecast_override: Any | None = None,
    accept_verified_seed_without_fast_step: bool = False,
    original_task_reference_goal: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    evaluator, verifier, limits = make_risk_stack(stage4_config, stage4_model, None)
    if forecast_override is not None:
        if not forecast_override_authorized(args, obstacle_audit):
            raise ValueError(
                "forecast_override is restricted to explicit offline diagnostics "
                "or non-commanding Static20 shadow mode"
            )
        forecast = forecast_override
        geometry_mode = "offline_forecast_override"
        forecast_builder = type(forecast_override).__name__
    elif multisphere_geometry is None:
        forecast = constant_forecast(center, velocity, radius)
        geometry_mode = "single_sphere"
        forecast_builder = getattr(
            constant_forecast, "__name__", type(constant_forecast).__name__
        )
    else:
        forecast = v3_execution_multisphere_forecast(
            np.asarray(multisphere_geometry["component_centers"], dtype=np.float64),
            np.asarray(multisphere_geometry["component_base_radii"], dtype=np.float64),
            velocity,
            object_id=int(obstacle_audit.get("track_id") or 1),
        )
        geometry_mode = "fresh_pca_multisphere"
        forecast_builder = "v3_execution_multisphere_forecast"
    head, tail, durations, p_inner, q_goal = make_local_reference(
        q_now, qd_now, args, reference_goal=reference_goal
    )
    reference_trajectory = NUBSTrajectory6D().generate(p_inner, head, tail, durations)
    original_task_reference_trajectory: Any = reference_trajectory
    if original_task_reference_goal is not None:
        original_head, original_tail, original_durations, original_inner, _ = make_local_reference(
            q_now,
            qd_now,
            args,
            reference_goal=original_task_reference_goal,
        )
        original_task_reference_trajectory = NUBSTrajectory6D().generate(
            original_inner, original_head, original_tail, original_durations
        )
    warm_start_audit: dict[str, Any] = {"mode": "linear"}
    optimization_inner = p_inner
    if getattr(args, "fast_warm_start", "linear") == "lateral":
        optimization_inner, warm_start_audit = clearance_guided_lateral_warm_start(
            stage4_model,
            p_inner,
            durations,
            center,
            offset_m=float(getattr(args, "lateral_warm_start_m", 0.04)),
        )
    online_started = time.perf_counter()
    fast_target_ms = float(getattr(args, "fast_target_ms", None) or args.fast_budget_ms)
    fast_max_ms = float(getattr(args, "fast_max_ms", None) or args.fast_budget_ms)
    deadline_perf = online_started + fast_max_ms / 1000.0
    result = run_repair_v3(
        evaluator,
        forecast,
        limits,
        optimization_inner,
        head,
        tail,
        durations,
        dense_active=True,
        v4_mode=True,
        deadline_perf=deadline_perf,
        elastic_tail_position=True,
        cheap_scale_screening=True,
        # A small linearization buffer compensates for the finite-distance
        # model; the externally reported/accepted requirement remains 3 mm.
        minimum_distance_improvement=1.10 * args.min_clearance_improvement_m,
    )
    repair_elapsed_ms = (time.perf_counter() - online_started) * 1000.0
    repair_step_ok = int(result.accepted_steps) > 0
    repair_trajectory = result.trajectory
    candidate_trajectory: Any = (
        repair_trajectory
        if repair_step_ok or not accept_verified_seed_without_fast_step
        else reference_trajectory
    )
    reference_full_trajectory: Any = reference_trajectory
    selected_rejoin_offset_s = None
    rejoin_search_audit = []
    if np.max(np.abs(result.tail_delta_q)) > 0.0:
        q_goal = result.tail_state[:, 0].copy()
    diagnostic_reference_verification_ms = 0.0
    paired_verification_wall_ms = 0.0
    if not repair_step_ok and not accept_verified_seed_without_fast_step:
        # The candidate is exactly the reference, so execution is already
        # impossible. Finish the online decision now and run at most one
        # reference verification afterwards for diagnostics only.
        online_elapsed_ms = repair_elapsed_ms
        reference_verify_started = time.perf_counter()
        reference_verification = verifier.verify(
            reference_full_trajectory,
            forecast,
            current_q=q_now,
            current_qd=qd_now,
            current_qdd=np.zeros(6),
            q_goal=q_goal,
            solver_success=True,
        )
        diagnostic_reference_verification_ms = (time.perf_counter() - reference_verify_started) * 1000.0
        verification = reference_verification
        candidate_verification_ms = 0.0
        reference_verification_ms = 0.0
    else:
        verification_pair_started = time.perf_counter()
        candidate_verify_started = time.perf_counter()
        verification = verifier.verify(
            candidate_trajectory,
            forecast,
            current_q=q_now,
            current_qd=qd_now,
            current_qdd=np.zeros(6),
            q_goal=q_goal,
            solver_success=True,
        )
        candidate_verification_ms = (time.perf_counter() - candidate_verify_started) * 1000.0
        if accept_verified_seed_without_fast_step:
            # V3's online authorization needs only the final candidate's full
            # verifier.  Comparing it with the optimization seed is useful
            # diagnostics, but cannot consume the 150 ms execution budget.
            online_elapsed_ms = (time.perf_counter() - online_started) * 1000.0
        if repair_step_ok:
            reference_verify_started = time.perf_counter()
            reference_verification = verifier.verify(
                reference_full_trajectory,
                forecast,
                current_q=q_now,
                current_qd=qd_now,
                current_qdd=np.zeros(6),
                q_goal=q_goal,
                solver_success=True,
            )
            measured_reference_ms = (
                time.perf_counter() - reference_verify_started
            ) * 1000.0
            if accept_verified_seed_without_fast_step:
                diagnostic_reference_verification_ms = measured_reference_ms
                reference_verification_ms = 0.0
            else:
                reference_verification_ms = measured_reference_ms
        else:
            # The bypass seed is the final candidate.  Its one complete
            # verification is part of the online decision; duplicating it as
            # a paired reference verification would add cost but no evidence.
            reference_verification = verification
            reference_verification_ms = 0.0
        paired_verification_wall_ms = (time.perf_counter() - verification_pair_started) * 1000.0
    post_check_started = time.perf_counter()
    clearance_gain = float(verification.min_distance - reference_verification.min_distance)
    candidate_samples = candidate_trajectory.dense_sample(0.02).q
    reference_samples = reference_full_trajectory.dense_sample(0.02).q
    candidate_profile_summary = clearance_profile_summary(
        candidate_trajectory,
        evaluator,
        forecast,
        guide_horizon_s=getattr(args, "guidance_horizon_s", None),
    )
    max_delta_q = float(np.max(np.abs(candidate_samples - reference_samples)))
    original_reference_samples = original_task_reference_trajectory.dense_sample(0.02).q
    max_delta_q_from_original_reference = float(
        np.max(np.abs(candidate_samples - original_reference_samples))
    )
    post_check_ms = (time.perf_counter() - post_check_started) * 1000.0
    if repair_step_ok and not accept_verified_seed_without_fast_step:
        online_elapsed_ms = (time.perf_counter() - online_started) * 1000.0
    hard_safety_ready = bool(
        online_elapsed_ms <= fast_max_ms
        and not result.budget_exhausted
        and verification.min_distance >= args.online_accept_m
        and all({**verification.checks, "solver_ok": True}.values())
    )
    contract = candidate_acceptance_contract(
        hard_safety_ready=hard_safety_ready,
        repair_step_ok=repair_step_ok,
        clearance_gain_m=clearance_gain,
        minimum_clearance_gain_m=args.min_clearance_improvement_m,
        delta_from_fast_seed_rad=max_delta_q,
        minimum_candidate_delta_rad=args.min_candidate_delta_q_rad,
        accept_verified_seed_without_fast_step=accept_verified_seed_without_fast_step,
    )
    local_repair_ready = bool(contract["local_repair_ready"])
    rejection_reasons = []
    if online_elapsed_ms > fast_max_ms or result.budget_exhausted:
        rejection_reasons.append("fast_computation_timeout")
    if verification.min_distance < args.online_accept_m:
        rejection_reasons.append("online_clearance_failed")
    failed_checks = [name for name, ok in verification.checks.items() if not ok]
    if failed_checks:
        rejection_reasons.append("verification_checks_failed:" + ",".join(failed_checks))
    if not repair_step_ok and not accept_verified_seed_without_fast_step:
        rejection_reasons.append("no_accepted_repair_step")
    if (
        clearance_gain < args.min_clearance_improvement_m
        and not accept_verified_seed_without_fast_step
    ):
        rejection_reasons.append("insufficient_clearance_improvement")
    if (
        max_delta_q < args.min_candidate_delta_q_rad
        and not accept_verified_seed_without_fast_step
    ):
        rejection_reasons.append("candidate_motion_indistinguishable_from_reference")
    candidate_source = str(contract["candidate_source"])
    candidate_dir = trial_dir / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    save_trajectory_csv(candidate_dir / "fast_ccro_nubs_candidate.csv", candidate_trajectory, dt=0.01)
    save_dynamic_risk_profile(
        candidate_dir / "fast_candidate_risk_profile.csv",
        candidate_trajectory,
        evaluator,
        forecast,
        density="medium",
        dt=0.04,
    )
    save_dynamic_risk_profile(
        candidate_dir / "fast_reference_risk_profile.csv",
        reference_full_trajectory,
        evaluator,
        forecast,
        density="medium",
        dt=0.04,
    )
    active_profile_rows = []
    for item in result.active_distance_profile:
        tau = float(item["tau"])
        occupancy = forecast.occupancy_at(tau)
        radius_at_tau = math.nan if not occupancy.spheres else max(float(sphere.radius) for sphere in occupancy.spheres)
        active_profile_rows.append(
            {
                "tau": f"{tau:.6f}",
                "distance_m": f"{float(item['distance_m']):.9f}",
                "nearest_link": item.get("nearest_link") or "",
                "surface_density": "dense_active",
                "forecast_radius": "" if not np.isfinite(radius_at_tau) else f"{radius_at_tau:.9f}",
            }
        )
    write_csv(
        candidate_dir / "fast_active_distance_profile.csv",
        active_profile_rows,
        ["tau", "distance_m", "nearest_link", "surface_density", "forecast_radius"],
    )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "LOCAL_REPAIR_READY" if local_repair_ready else "FAST_REPAIR_FAILED",
        "local_repair_status": "LOCAL_REPAIR_READY" if local_repair_ready else "FAST_REPAIR_FAILED",
        "local_repair_ready": local_repair_ready,
        "execution_authorization_status": "PENDING_POST_PLAN_FRESH_RECHECK" if local_repair_ready else "NOT_ELIGIBLE",
        "accepted_for_switch": False,
        "repair_step_ok": repair_step_ok,
        "rejection_reasons": rejection_reasons,
        "candidate_source": candidate_source,
        "fast_extra_correction_applied": bool(
            contract["fast_extra_correction_applied"]
        ),
        "optimizer_diagnostics": contract["optimizer_diagnostics"],
        "verified_seed_is_candidate_contract": bool(
            accept_verified_seed_without_fast_step
        ),
        "candidate_equals_fast_seed": bool(
            max_delta_q < args.min_candidate_delta_q_rad
        ),
        "candidate_is_original_reference": bool(
            max_delta_q_from_original_reference < args.min_candidate_delta_q_rad
        ),
        "candidate_is_reference_continuation": bool(
            max_delta_q_from_original_reference < args.min_candidate_delta_q_rad
        ),
        "warm_start": warm_start_audit,
        "fast_elapsed_ms": online_elapsed_ms,
        "online_pipeline_elapsed_ms": online_elapsed_ms,
        "repair_elapsed_ms": repair_elapsed_ms,
        "candidate_verification_ms": candidate_verification_ms,
        "reference_verification_ms": reference_verification_ms,
        "diagnostic_reference_verification_ms": diagnostic_reference_verification_ms,
        "paired_verification_wall_ms": paired_verification_wall_ms,
        "diagnostic_total_wall_ms": (time.perf_counter() - online_started) * 1000.0,
        "post_check_ms": post_check_ms,
        "budget_exhausted": result.budget_exhausted,
        "tail_delta_q_rad": result.tail_delta_q.tolist(),
        "tail_delta_q_max_rad": float(np.max(np.abs(result.tail_delta_q))),
        "selected_rejoin_offset_s": selected_rejoin_offset_s,
        "rejoin_search_audit": rejoin_search_audit,
        "candidate_total_duration_s": float(candidate_trajectory.total_duration),
        "fast_budget_ms": fast_max_ms,
        "fast_target_ms": fast_target_ms,
        "fast_max_ms": fast_max_ms,
        "realtime_target_met": bool(online_elapsed_ms <= fast_target_ms),
        "realtime_target_diagnostic": (
            "met" if online_elapsed_ms <= fast_target_ms else "missed_soft_target"
        ),
        "online_accept_m": args.online_accept_m,
        "verification_min_distance_m": verification.min_distance,
        "reference_online_min_distance_m": reference_verification.min_distance,
        # Backward-compatible legacy field above is the Fast input seed, not
        # the original task reference.  Keep explicit V3 aliases for audits.
        "fast_seed_online_min_distance_m": reference_verification.min_distance,
        "candidate_online_min_distance_m": verification.min_distance,
        "candidate_min_clearance_m": candidate_profile_summary["min_clearance_m"],
        "candidate_min_tau_s": candidate_profile_summary["min_tau_s"],
        "candidate_min_nearest_link": candidate_profile_summary["min_nearest_link"],
        "candidate_end_clearance_m": candidate_profile_summary["end_clearance_m"],
        "candidate_end_minus_min_clearance_m": candidate_profile_summary[
            "end_minus_min_clearance_m"
        ],
        "candidate_min_tau_fraction": candidate_profile_summary["min_tau_fraction"],
        "candidate_closest_approach_before_tail": candidate_profile_summary[
            "closest_approach_before_tail"
        ],
        "guidance_horizon_s": getattr(args, "guidance_horizon_s", None),
        "guidance_tail_clearance": candidate_profile_summary["guide"],
        "clearance_improvement_m": clearance_gain,
        "clearance_improvement_vs_fast_seed_m": clearance_gain,
        "min_clearance_improvement_m": args.min_clearance_improvement_m,
        "min_clearance_improvement_preference_m": args.min_clearance_improvement_m,
        "clearance_improvement_is_hard_gate": bool(
            not accept_verified_seed_without_fast_step
        ),
        "max_delta_q_from_reference_rad": max_delta_q,
        "delta_candidate_from_bypass_seed_max_abs_rad": max_delta_q,
        "delta_candidate_from_original_reference_max_abs_rad": (
            max_delta_q_from_original_reference
        ),
        "verification_reasons": verification.reasons,
        "verification_checks": verification.checks,
        "repair_iterations": result.iterations,
        "accepted_steps": result.accepted_steps,
        "active_constraints": result.active_constraints,
        "qp_successes": result.qp_successes,
        "risk_scan_ms": result.risk_scan_ms,
        "linearization_ms": result.linearization_ms,
        "qp_ms": result.qp_ms,
        "trajectory_generation_ms": result.trajectory_generation_ms,
        "motion_check_ms": result.motion_check_ms,
        "candidate_distance_check_ms": result.candidate_distance_check_ms,
        "scale_attempts": result.scale_attempts,
        "unaccounted_fast_ms": max(
            0.0,
            repair_elapsed_ms
            - result.risk_scan_ms
            - result.linearization_ms
            - result.qp_ms
            - result.trajectory_generation_ms
            - result.motion_check_ms
            - result.candidate_distance_check_ms,
        ),
        "messages": result.messages,
        "q_now": q_now.tolist(),
        "qd_now": qd_now.tolist(),
        "q_goal": q_goal.tolist(),
        "obstacle_center": center.tolist(),
        "obstacle_velocity": velocity.tolist(),
        "obstacle_radius": radius,
        "obstacle_geometry_mode": geometry_mode,
        "forecast_builder": forecast_builder,
        "forecast_component_radii_at_0s_m": [
            float(sphere.radius) for sphere in forecast.occupancy_at(0.0).spheres
        ],
        "forecast_component_radii_at_candidate_end_m": [
            float(sphere.radius)
            for sphere in forecast.occupancy_at(
                float(candidate_trajectory.total_duration)
            ).spheres
        ],
        "multisphere_geometry": multisphere_geometry,
        "obstacle_association": obstacle_audit,
        "risk_links": sorted(risk_links),
        "candidate_csv": str(candidate_dir / "fast_ccro_nubs_candidate.csv"),
    }
    write_json(candidate_dir / "candidate_summary.json", payload)
    if artifacts_out is not None:
        artifacts_out.update(
            {
                "candidate_trajectory": candidate_trajectory,
                "reference_trajectory": reference_full_trajectory,
                "q_now": np.asarray(q_now, dtype=np.float64).copy(),
                "qd_now": np.asarray(qd_now, dtype=np.float64).copy(),
                "local_tail_state": candidate_trajectory.tail_state,
            }
        )
    return payload


def authorize_local_repair_execution(
    args: argparse.Namespace,
    stage4_config: dict[str, Any],
    stage4_model: RobotSurfaceModel,
    *,
    local_repair_ready: bool,
    local_artifacts: dict[str, Any],
    fresh_geometry: dict[str, Any],
    fresh_velocity: np.ndarray,
    trial_dir: Path,
    execution_duration_s: float | None = None,
) -> tuple[dict[str, Any], Any | None]:
    """Revalidate only the local repair on Fresh #2 at its execution time scale."""
    started = time.perf_counter()
    output_dir = trial_dir / "local_execution_authorization"
    output_dir.mkdir(parents=True, exist_ok=True)
    repair = local_artifacts["candidate_trajectory"]
    native_duration = float(repair.total_duration)
    requested_duration = float(
        execution_duration_s
        if execution_duration_s is not None
        else (args.candidate_playback_duration_s if args.candidate_playback_duration_s > 0.0 else native_duration)
    )
    execution_trajectory = TimeScaledTrajectory6D(repair, requested_duration)
    tabletop_guard = gripper_base_workspace_guard(
        execution_trajectory,
        stage4_model,
        min_z_m=float(args.gripper_base_min_z_m),
    )
    if not tabletop_guard["passed"]:
        payload = {
            "status": "TABLE_CLEARANCE_GUARD_FAILED",
            "authorization_mode": "LOCAL_ONLY",
            "local_execution_authorized": False,
            "authorized_execution_duration_s": requested_duration,
            "tabletop_workspace_guard": tabletop_guard,
            "verification_reasons": [tabletop_guard["reason"]],
            "authorized_trajectory_csv": None,
            "authorization_compute_ms": (time.perf_counter() - started) * 1000.0,
            "robot_executed": False,
        }
        write_json(output_dir / "authorization_summary.json", payload)
        return payload, None
    evaluator, verifier, _ = make_risk_stack(stage4_config, stage4_model, None)
    forecast = v3_execution_multisphere_forecast(
        np.asarray(fresh_geometry["component_centers"], dtype=np.float64),
        np.asarray(fresh_geometry["component_base_radii"], dtype=np.float64),
        np.asarray(fresh_velocity, dtype=np.float64),
    )
    q_now = np.asarray(local_artifacts["q_now"], dtype=np.float64)
    qd_now = np.asarray(local_artifacts["qd_now"], dtype=np.float64)
    q_goal = execution_trajectory.evaluate(execution_trajectory.total_duration)
    verification = verifier.verify(
        execution_trajectory,
        forecast,
        current_q=q_now,
        current_qd=qd_now,
        current_qdd=np.zeros(6),
        q_goal=q_goal,
        solver_success=bool(local_repair_ready),
    )
    authorized = bool(local_repair_ready and verification.accepted)
    trajectory_csv = output_dir / "authorized_local_repair.csv"
    if trajectory_csv.exists():
        trajectory_csv.unlink()
    if authorized:
        save_trajectory_csv(trajectory_csv, execution_trajectory, dt=0.01)
        save_dynamic_risk_profile(
            output_dir / "authorized_local_repair_risk_profile.csv",
            execution_trajectory,
            evaluator,
            forecast,
            density="medium",
            dt=0.04,
        )
    payload = {
        "status": "LOCAL_EXECUTION_AUTHORIZED" if authorized else "LOCAL_EXECUTION_RECHECK_FAILED",
        "authorization_mode": "LOCAL_ONLY",
        "local_execution_authorized": authorized,
        "native_candidate_duration_s": native_duration,
        "authorized_execution_duration_s": requested_duration,
        "time_scale": execution_trajectory.time_scale,
        "verification_min_distance_m": float(verification.min_distance),
        "verification_checks": verification.checks,
        "verification_reasons": verification.reasons,
        "tabletop_workspace_guard": tabletop_guard,
        "verification_ms": float(verification.validation_ms),
        "forecast_builder": "v3_execution_multisphere_forecast",
        "forecast_component_radii_at_0s_m": [
            float(sphere.radius) for sphere in forecast.occupancy_at(0.0).spheres
        ],
        "forecast_component_radii_at_execution_end_m": [
            float(sphere.radius)
            for sphere in forecast.occupancy_at(requested_duration).spheres
        ],
        "authorized_trajectory_csv": str(trajectory_csv) if authorized else None,
        "authorization_compute_ms": (time.perf_counter() - started) * 1000.0,
        "robot_executed": False,
    }
    write_json(output_dir / "authorization_summary.json", payload)
    return payload, execution_trajectory if authorized else None


def translated_multisphere_geometry(
    geometry: dict[str, Any], old_center: np.ndarray, new_center: np.ndarray
) -> dict[str, Any]:
    """Translate a Fresh-initialized rigid multisphere without refitting shape."""
    shift = np.asarray(new_center, dtype=np.float64) - np.asarray(old_center, dtype=np.float64)
    return {
        **geometry,
        "component_centers": np.asarray(geometry["component_centers"], dtype=np.float64) + shift[None, :],
        "component_base_radii": np.asarray(geometry["component_base_radii"], dtype=np.float64).copy(),
        "rolling_rigid_translation_m": shift,
    }


def rolling_fast_until_authorized(
    args: argparse.Namespace,
    stage4_config: dict[str, Any],
    stage4_model: RobotSurfaceModel,
    *,
    processor: Any,
    state_reader: Any,
    denoiser: Any,
    q_now: np.ndarray,
    qd_now: np.ndarray,
    reference_goal: tuple[np.ndarray, np.ndarray, np.ndarray],
    rejoin_goals: list[tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    initial_fresh: dict[str, Any],
    initial_geometry: dict[str, Any],
    risk_links: set[str],
    trial_dir: Path,
) -> dict[str, Any]:
    """Replan while stopped from short fresh updates until one candidate is authorized."""
    started = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    previous = dict(initial_fresh)
    geometry_center = np.asarray(initial_fresh["center"], dtype=np.float64)
    attempt_root = trial_dir / "rolling_fast"
    attempt_root.mkdir(parents=True, exist_ok=True)
    evaluator, _, _ = make_risk_stack(stage4_config, stage4_model, None)

    while time.perf_counter() - started < args.rolling_fast_max_s:
        short_args = copy.copy(args)
        short_args.post_stop_recheck_duration_s = args.rolling_observation_duration_s
        short_args.post_stop_recheck_min_frames = args.rolling_observation_min_frames
        short_args.post_stop_recheck_min_span_s = args.rolling_observation_min_span_s
        fresh, frames, _ = capture_post_stop_obstacle(
            processor,
            state_reader,
            denoiser,
            short_args,
            trigger_cluster_center=np.asarray(previous["center"], dtype=np.float64),
            trigger_velocity=np.asarray(previous["velocity"], dtype=np.float64),
            trigger_timestamp=float(previous["last_timestamp"]),
            stop_when_ready=True,
        )
        index = len(attempts) + 1
        attempt_dir = attempt_root / f"attempt_{index:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        write_json(attempt_dir / "fresh_update.json", {"result": fresh, "frames": frames})
        if not fresh.get("accepted", False):
            attempts.append({"attempt": index, "status": "FRESH_UPDATE_NOT_READY", "fresh": fresh})
            continue
        raw_guards = [float(frame.get("raw_guard_distance_m", math.inf)) for frame in frames]
        raw_guard = min(raw_guards) if raw_guards else math.inf
        geometry = translated_multisphere_geometry(
            initial_geometry, geometry_center, np.asarray(fresh["center"], dtype=np.float64)
        )
        current_forecast = v3_execution_multisphere_forecast(
            np.asarray(geometry["component_centers"]),
            np.asarray(geometry["component_base_radii"]),
            np.asarray(fresh["velocity"]),
        )
        current_risk = evaluator.configuration(q_now, current_forecast, 0.0, density="medium", with_gradient=False)
        if raw_guard <= args.guided_hard_stop_m or current_risk.min_distance <= args.moving_shadow_current_stop_m:
            attempts.append({
                "attempt": index,
                "status": "ROLLING_SAFE_HOLD_DISTANCE",
                "raw_guard_distance_m": raw_guard,
                "current_distance_m": float(current_risk.min_distance),
            })
            break

        artifacts: dict[str, Any] = {}
        plan_started = time.perf_counter()
        candidate = run_fast_repair(
            args, stage4_config, stage4_model,
            q_now=q_now, qd_now=qd_now,
            center=np.asarray(fresh["center"], dtype=np.float64),
            velocity=np.asarray(fresh["velocity"], dtype=np.float64),
            radius=float(fresh["radius"]),
            risk_links=risk_links,
            trial_dir=attempt_dir,
            reference_goal=reference_goal,
            rejoin_goals=rejoin_goals,
            obstacle_audit={"rolling_attempt": index, "fresh_update": fresh},
            multisphere_geometry=geometry,
            artifacts_out=artifacts,
        )
        plan_elapsed = time.perf_counter() - plan_started
        propagated_center = np.asarray(fresh["center"], dtype=np.float64) + np.asarray(fresh["velocity"], dtype=np.float64) * plan_elapsed
        execution_geometry = translated_multisphere_geometry(
            initial_geometry, geometry_center, propagated_center
        )
        local_auth, authorized_local = authorize_local_repair_execution(
            args, stage4_config, stage4_model,
            local_repair_ready=bool(candidate.get("local_repair_ready")),
            local_artifacts=artifacts,
            fresh_geometry=execution_geometry,
            fresh_velocity=np.asarray(fresh["velocity"], dtype=np.float64),
            trial_dir=attempt_dir,
        )
        full_auth = authorize_candidate_execution(
            args, stage4_config, stage4_model,
            local_repair_ready=bool(candidate.get("local_repair_ready")),
            local_artifacts=artifacts,
            fresh_geometry=execution_geometry,
            fresh_velocity=np.asarray(fresh["velocity"], dtype=np.float64),
            rejoin_goals=rejoin_goals,
            trial_dir=attempt_dir,
        )
        attempt = {
            "attempt": index,
            "status": "ROLLING_AUTHORIZED" if (
                local_auth.get("local_execution_authorized") or full_auth.get("execution_authorized")
            ) else "ROLLING_REPLAN_REQUIRED",
            "fresh": fresh,
            "raw_guard_distance_m": raw_guard,
            "current_distance_m": float(current_risk.min_distance),
            "planning_elapsed_s": plan_elapsed,
            "propagated_center": propagated_center,
            "candidate": candidate,
            "local_authorization": local_auth,
            "execution_authorization": full_auth,
        }
        attempts.append(attempt)
        write_json(attempt_dir / "rolling_attempt_summary.json", attempt)
        if attempt["status"] == "ROLLING_AUTHORIZED":
            return {
                "status": "ROLLING_FAST_AUTHORIZED",
                "authorized": True,
                "attempts": attempts,
                "candidate_summary": candidate,
                "local_artifacts": artifacts,
                "fresh": fresh,
                "fresh_geometry": execution_geometry,
                "local_authorization": local_auth,
                "execution_authorization": full_auth,
                "authorized_local_trajectory": authorized_local,
                "elapsed_s": time.perf_counter() - started,
            }
        previous = fresh
    return {
        "status": "ROLLING_FAST_SAFE_HOLD",
        "authorized": False,
        "attempts": attempts,
        "elapsed_s": time.perf_counter() - started,
    }


def authorize_candidate_execution(
    args: argparse.Namespace,
    stage4_config: dict[str, Any],
    stage4_model: RobotSurfaceModel,
    *,
    local_repair_ready: bool,
    local_artifacts: dict[str, Any],
    fresh_geometry: dict[str, Any],
    fresh_velocity: np.ndarray,
    rejoin_goals: list[tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    trial_dir: Path,
) -> dict[str, Any]:
    """Authorize a local repair only after Fresh #2 validates repair plus rejoin."""
    started = time.perf_counter()
    output_dir = trial_dir / "post_plan_authorization"
    output_dir.mkdir(parents=True, exist_ok=True)
    # A safe reference continuation is not a repaired candidate.  In
    # particular, a failed Fast solve may leave candidate_trajectory equal to
    # the nominal reference; Fresh #2 must never turn that fallback into an
    # execution authorization merely because the reference is currently clear.
    if not local_repair_ready:
        payload = {
            "status": "NOT_ELIGIBLE_FAST_REPAIR_FAILED",
            "authorization_mode": "SHADOW",
            "execution_authorized": False,
            "reason": "local_repair_not_ready",
            "selected_rejoin_offset_s": None,
            "full_candidate_min_distance_m": None,
            "authorized_trajectory_csv": None,
            "authorized_duration_s": None,
            "rejoin_search_audit": [],
            "authorization_compute_ms": (time.perf_counter() - started) * 1000.0,
            "robot_executed": False,
        }
        write_json(output_dir / "authorization_summary.json", payload)
        return payload
    evaluator, verifier, limits = make_risk_stack(stage4_config, stage4_model, None)
    forecast = v3_execution_multisphere_forecast(
        np.asarray(fresh_geometry["component_centers"], dtype=np.float64),
        np.asarray(fresh_geometry["component_base_radii"], dtype=np.float64),
        np.asarray(fresh_velocity, dtype=np.float64),
    )
    repair = local_artifacts["candidate_trajectory"]
    q_now = np.asarray(local_artifacts["q_now"], dtype=np.float64)
    qd_now = np.asarray(local_artifacts["qd_now"], dtype=np.float64)
    search_audit = []
    accepted_trajectory = None
    accepted_verification = None
    selected_offset = None
    for offset_s, state in rejoin_goals:
        bridge_duration = float(offset_s) - float(args.local_horizon_s)
        if bridge_duration <= 0.0:
            continue
        endpoint_risk = evaluator.configuration(
            np.asarray(state[0], dtype=np.float64), forecast, float(offset_s), density="medium", with_gradient=False
        )
        audit = {
            "offset_s": float(offset_s),
            "endpoint_distance_m": float(endpoint_risk.min_distance),
            "endpoint_nearest_link": endpoint_risk.nearest_link,
            "endpoint_safe": bool(endpoint_risk.min_distance >= args.online_accept_m),
            "motion_feasible": False,
            "full_candidate_valid": False,
        }
        search_audit.append(audit)
        if not audit["endpoint_safe"]:
            continue
        bridge = make_rejoin_bridge(repair, state, bridge_duration)
        full_candidate = CompositeTrajectory6D([repair, bridge])
        samples = full_candidate.dense_sample(0.04)
        audit["motion_feasible"] = bool(
            np.all(samples.q >= limits.q_min[None, :])
            and np.all(samples.q <= limits.q_max[None, :])
            and np.all(np.abs(samples.qd) <= limits.qd_max[None, :])
            and np.all(np.abs(samples.qdd) <= limits.qdd_max[None, :])
        )
        if not audit["motion_feasible"]:
            continue
        tabletop_guard = gripper_base_workspace_guard(
            full_candidate,
            stage4_model,
            min_z_m=float(args.gripper_base_min_z_m),
        )
        audit["tabletop_workspace_guard"] = tabletop_guard
        if not tabletop_guard["passed"]:
            audit["full_candidate_valid"] = False
            audit["rejection_reason"] = tabletop_guard["reason"]
            continue
        verification = verifier.verify(
            full_candidate,
            forecast,
            current_q=q_now,
            current_qd=qd_now,
            current_qdd=np.zeros(6),
            q_goal=np.asarray(state[0], dtype=np.float64),
            solver_success=True,
        )
        audit.update(
            {
                "full_candidate_valid": bool(verification.accepted),
                "full_candidate_min_distance_m": float(verification.min_distance),
                "verification_checks": verification.checks,
                "verification_ms": float(verification.validation_ms),
            }
        )
        if verification.accepted:
            accepted_trajectory = full_candidate
            accepted_verification = verification
            selected_offset = float(offset_s)
            break
    authorized = accepted_trajectory is not None
    if authorized:
        save_trajectory_csv(output_dir / "authorized_repair_rejoin.csv", accepted_trajectory, dt=0.01)
        save_dynamic_risk_profile(
            output_dir / "authorized_candidate_risk_profile.csv",
            accepted_trajectory,
            evaluator,
            forecast,
            density="medium",
            dt=0.04,
        )
    payload = {
        "status": "EXECUTION_AUTHORIZED" if authorized else "POST_PLAN_RECHECK_FAILED",
        "authorization_mode": "SHADOW",
        "execution_authorized": authorized,
        "selected_rejoin_offset_s": selected_offset,
        "full_candidate_min_distance_m": None if accepted_verification is None else float(accepted_verification.min_distance),
        "authorized_trajectory_csv": str(output_dir / "authorized_repair_rejoin.csv") if authorized else None,
        "authorized_duration_s": None if accepted_trajectory is None else float(accepted_trajectory.total_duration),
        "rejoin_search_audit": search_audit,
        "authorization_compute_ms": (time.perf_counter() - started) * 1000.0,
        "robot_executed": False,
    }
    write_json(output_dir / "authorization_summary.json", payload)
    return payload


def authorize_delayed_rejoin_after_fresh3(
    args: argparse.Namespace,
    stage4_config: dict[str, Any],
    stage4_model: RobotSurfaceModel,
    *,
    local_artifacts: dict[str, Any],
    fresh3: dict[str, Any],
    fresh3_geometry: dict[str, Any] | None,
    fresh3_frames: list[dict[str, Any]],
    rejoin_goals: list[tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    hard_guard_distance_m: float,
    trial_dir: Path,
) -> tuple[dict[str, Any], Any | None]:
    """Authorize a bridge after an already executed local-only repair.

    The bridge is planned from the fixed, Fresh #2-authorized local tail.  It
    does not invoke Fast a second time.  Fresh #3 supplies either the associated
    moving-object geometry or a conservative stationary union of every external
    cluster in the consecutive scene-clear audit frames.
    """
    started = time.perf_counter()
    output_dir = trial_dir / "delayed_rejoin_authorization"
    output_dir.mkdir(parents=True, exist_ok=True)
    hard_guard_safe = bool(
        not np.isfinite(hard_guard_distance_m) or hard_guard_distance_m > args.guided_hard_stop_m
    )
    scene_clear_audit = None
    forecast_basis = None
    forecast = None
    if fresh3.get("accepted", False) and fresh3_geometry is not None:
        forecast = v3_execution_multisphere_forecast(
            np.asarray(fresh3_geometry["component_centers"], dtype=np.float64),
            np.asarray(fresh3_geometry["component_base_radii"], dtype=np.float64),
            np.asarray(fresh3["velocity"], dtype=np.float64),
        )
        forecast_basis = "FRESH3_TRACKED_OBSTACLE"
    else:
        # Reuse the same strict three-frame scene-clear predicate used by the
        # direct resume gate.  Preserve every observed external cluster in the
        # bridge verifier; no missing association is treated as free space.
        repair = local_artifacts["candidate_trajectory"]
        preview_times = np.asarray([0.0, max(args.prediction_horizon_s, repair.total_duration)], dtype=np.float64)
        preview_q = np.vstack([repair.evaluate(repair.total_duration), repair.evaluate(repair.total_duration)])
        scene_clear_audit = authorize_fresh3_scene_clear(
            args,
            stage4_model,
            fresh3_frames=fresh3_frames,
            remainder_times=preview_times,
            remainder_q=preview_q,
        )
        if scene_clear_audit["accepted"]:
            required = int(args.post_stop_recheck_min_frames)
            clusters = [
                cluster
                for frame in fresh3_frames[-required:]
                for cluster in frame.get("all_external_clusters", [])
            ]
            if clusters:
                centers = np.asarray([cluster["center"] for cluster in clusters], dtype=np.float64).reshape(-1, 3)
                radii = np.asarray([float(cluster["radius_m"]) for cluster in clusters], dtype=np.float64)
            else:
                # The camera validity and consecutive clear-scene checks above
                # establish an empty ROI.  A remote sentinel lets the standard
                # verifier still audit all joint/motion constraints.
                centers = np.asarray([[100.0, 100.0, 100.0]], dtype=np.float64)
                radii = np.asarray([1.0e-6], dtype=np.float64)
                forecast_basis = "FRESH3_SCENE_CLEAR_EMPTY_ROI"
            forecast = v3_execution_multisphere_forecast(
                centers, radii, np.zeros(3, dtype=np.float64)
            )
            if forecast_basis is None:
                forecast_basis = "FRESH3_SCENE_CLEAR_ALL_CLUSTERS_STATIC"

    search_audit: list[dict[str, Any]] = []
    accepted_bridge = None
    accepted_verification = None
    selected_offset = None
    if forecast is not None and hard_guard_safe:
        evaluator, verifier, limits = make_risk_stack(stage4_config, stage4_model, None)
        repair = local_artifacts["candidate_trajectory"]
        q_tail = np.asarray(repair.evaluate(repair.total_duration), dtype=np.float64)
        qd_tail = np.asarray(repair.evaluate(repair.total_duration, 1), dtype=np.float64)
        for offset_s, state in rejoin_goals:
            bridge_duration = float(offset_s) - float(args.local_horizon_s)
            if bridge_duration <= 0.0:
                continue
            endpoint_risk = evaluator.configuration(
                np.asarray(state[0], dtype=np.float64),
                forecast,
                bridge_duration,
                density="medium",
                with_gradient=False,
            )
            audit = {
                "offset_s": float(offset_s),
                "bridge_duration_s": bridge_duration,
                "endpoint_distance_m": float(endpoint_risk.min_distance),
                "endpoint_nearest_link": endpoint_risk.nearest_link,
                "endpoint_safe": bool(endpoint_risk.min_distance >= args.online_accept_m),
                "motion_feasible": False,
                "bridge_valid": False,
            }
            search_audit.append(audit)
            if not audit["endpoint_safe"]:
                continue
            bridge = make_rejoin_bridge(repair, state, bridge_duration)
            samples = bridge.dense_sample(0.04)
            audit["motion_feasible"] = bool(
                np.all(samples.q >= limits.q_min[None, :])
                and np.all(samples.q <= limits.q_max[None, :])
                and np.all(np.abs(samples.qd) <= limits.qd_max[None, :])
                and np.all(np.abs(samples.qdd) <= limits.qdd_max[None, :])
            )
            if not audit["motion_feasible"]:
                continue
            verification = verifier.verify(
                bridge,
                forecast,
                current_q=q_tail,
                current_qd=qd_tail,
                current_qdd=np.asarray(repair.evaluate(repair.total_duration, 2), dtype=np.float64),
                q_goal=np.asarray(state[0], dtype=np.float64),
                solver_success=True,
            )
            audit.update(
                {
                    "bridge_valid": bool(verification.accepted),
                    "bridge_min_distance_m": float(verification.min_distance),
                    "verification_checks": verification.checks,
                    "verification_reasons": verification.reasons,
                    "verification_ms": float(verification.validation_ms),
                }
            )
            if verification.accepted:
                accepted_bridge = bridge
                accepted_verification = verification
                selected_offset = float(offset_s)
                save_trajectory_csv(output_dir / "authorized_delayed_rejoin_bridge.csv", bridge, dt=0.01)
                save_dynamic_risk_profile(
                    output_dir / "authorized_delayed_rejoin_risk_profile.csv",
                    bridge,
                    evaluator,
                    forecast,
                    density="medium",
                    dt=0.04,
                )
                break

    authorized = accepted_bridge is not None
    payload = {
        "status": "DELAYED_REJOIN_AUTHORIZED" if authorized else "DELAYED_REJOIN_HOLD",
        "authorized": authorized,
        "forecast_basis": forecast_basis,
        "hard_guard_distance_m": hard_guard_distance_m,
        "hard_guard_safe": hard_guard_safe,
        "scene_clear_audit": scene_clear_audit,
        "selected_rejoin_offset_s": selected_offset,
        "bridge_min_distance_m": None if accepted_verification is None else float(accepted_verification.min_distance),
        "authorized_trajectory_csv": (
            str(output_dir / "authorized_delayed_rejoin_bridge.csv") if authorized else None
        ),
        "authorized_duration_s": None if accepted_bridge is None else float(accepted_bridge.total_duration),
        "rejoin_search_audit": search_audit,
        "authorization_compute_ms": (time.perf_counter() - started) * 1000.0,
        "robot_executed": False,
    }
    write_json(output_dir / "authorization_summary.json", payload)
    return payload, accepted_bridge


def authorize_fresh3_scene_clear(
    args: argparse.Namespace,
    stage4_model: RobotSurfaceModel,
    *,
    fresh3_frames: list[dict[str, Any]],
    remainder_times: np.ndarray,
    remainder_q: np.ndarray,
) -> dict[str, Any]:
    """Authorize a clear scene only from three consecutive full-frame audits."""
    required = int(args.post_stop_recheck_min_frames)
    tail = fresh3_frames[-required:] if len(fresh3_frames) >= required else []
    frame_results = []
    all_ok = len(tail) == required
    preview_t = np.arange(0.0, args.prediction_horizon_s + 0.5 * args.prediction_step_s, args.prediction_step_s)
    for frame in tail:
        valid = bool(frame.get("frame_valid", False))
        unassociated = not bool(frame.get("associated", False))
        guard_distance = float(frame.get("raw_guard_distance_m", -math.inf))
        current_safe = guard_distance > args.moving_shadow_current_stop_m
        future_min = math.inf
        future_nearest_link = None
        clusters = list(frame.get("all_external_clusters", []))
        for tau in preview_t:
            q_tau = np.asarray([
                np.interp(min(float(tau), float(remainder_times[-1])), remainder_times, remainder_q[:, j])
                for j in range(6)
            ])
            surfaces = stage4_model.surface_by_link(q_tau, density="medium")
            for cluster in clusters:
                center = np.asarray(cluster["center"], dtype=np.float64)
                radius = float(cluster["radius_m"]) + args.prediction_margin_m + args.prediction_uncertainty_m
                for link, surface in surfaces.items():
                    if len(surface) == 0:
                        continue
                    distance = float(cKDTree(surface).query(center, k=1)[0] - radius)
                    if distance < future_min:
                        future_min = distance
                        future_nearest_link = link
        future_safe = future_min >= args.moving_shadow_replan_in_m
        checks = {
            "camera_frame_valid": valid,
            "original_target_unassociated": unassociated,
            "raw_hard_guard_safe": guard_distance > args.guided_hard_stop_m,
            "all_cluster_current_distance_safe": current_safe,
            "remaining_reference_0p5s_safe": future_safe,
        }
        frame_ok = bool(all(checks.values()))
        all_ok = bool(all_ok and frame_ok)
        frame_results.append(
            {
                "timestamp": frame.get("timestamp"),
                "cluster_count": len(clusters),
                "raw_guard_distance_m": guard_distance,
                "future_reference_min_distance_m": future_min,
                "future_nearest_link": future_nearest_link,
                "checks": checks,
                "accepted": frame_ok,
            }
        )
    return {
        "status": "FRESH3_SCENE_CLEAR" if all_ok else "FRESH3_SCENE_NOT_CLEAR",
        "accepted": all_ok,
        "required_consecutive_frames": required,
        "available_frames": len(fresh3_frames),
        "evaluated_tail_frames": len(tail),
        "frames": frame_results,
    }


def authorize_reference_resume_after_fresh3(
    args: argparse.Namespace,
    stage4_config: dict[str, Any],
    stage4_model: RobotSurfaceModel,
    *,
    fresh3: dict[str, Any],
    fresh3_geometry: dict[str, Any] | None,
    fresh3_frames: list[dict[str, Any]] | None = None,
    remainder_times: np.ndarray,
    remainder_q: np.ndarray,
    hard_guard_distance_m: float,
) -> dict[str, Any]:
    """Gate one reference resume using current and 0.5 s predicted clearance."""
    if not fresh3.get("accepted", False) or fresh3_geometry is None:
        scene_clear = authorize_fresh3_scene_clear(
            args,
            stage4_model,
            fresh3_frames=[] if fresh3_frames is None else fresh3_frames,
            remainder_times=remainder_times,
            remainder_q=remainder_q,
        )
        hard_guard_safe = bool(
            not np.isfinite(hard_guard_distance_m) or hard_guard_distance_m > args.guided_hard_stop_m
        )
        authorized = bool(scene_clear["accepted"] and hard_guard_safe)
        return {
            "status": "REFERENCE_RESUME_AUTHORIZED" if authorized else "REFERENCE_RESUME_HOLD",
            "resume_basis": "FRESH3_SCENE_CLEAR" if authorized else "FRESH3_NOT_READY_OR_SCENE_UNSAFE",
            "authorized": authorized,
            "reason": "fresh3_scene_clear" if authorized else fresh3.get("reason", "fresh3_not_ready"),
            "hard_guard_distance_m": hard_guard_distance_m,
            "scene_clear_audit": scene_clear,
        }
    evaluator, _, _ = make_risk_stack(stage4_config, stage4_model, None)
    forecast = v3_execution_multisphere_forecast(
        np.asarray(fresh3_geometry["component_centers"], dtype=np.float64),
        np.asarray(fresh3_geometry["component_base_radii"], dtype=np.float64),
        np.asarray(fresh3["velocity"], dtype=np.float64),
    )
    preview_t = np.arange(0.0, args.prediction_horizon_s + 0.5 * args.prediction_step_s, args.prediction_step_s)
    preview_t = np.unique(np.r_[preview_t, min(float(remainder_times[-1]), args.prediction_horizon_s)])
    audits = []
    for tau in preview_t:
        q_tau = np.asarray([
            np.interp(min(float(tau), float(remainder_times[-1])), remainder_times, remainder_q[:, j]) for j in range(6)
        ])
        risk = evaluator.configuration(q_tau, forecast, float(tau), density="medium", with_gradient=False)
        audits.append({"tau_s": float(tau), "distance_m": float(risk.min_distance), "nearest_link": risk.nearest_link})
    current_distance = float(audits[0]["distance_m"])
    predicted_distance = min(float(row["distance_m"]) for row in audits)
    checks = {
        "fresh3_ready": True,
        "hard_guard_safe": bool(not np.isfinite(hard_guard_distance_m) or hard_guard_distance_m > args.guided_hard_stop_m),
        "current_distance_safe": bool(current_distance > args.moving_shadow_current_stop_m),
        "predicted_reference_safe": bool(predicted_distance >= args.moving_shadow_replan_in_m),
    }
    authorized = bool(all(checks.values()))
    return {
        "status": "REFERENCE_RESUME_AUTHORIZED" if authorized else "REFERENCE_RESUME_HOLD",
        "authorized": authorized,
        "resume_basis": "FRESH3_TRACKED_OBSTACLE",
        "checks": checks,
        "current_distance_m": current_distance,
        "predicted_reference_min_distance_m": predicted_distance,
        "hard_guard_distance_m": hard_guard_distance_m,
        "preview": audits,
    }


def maybe_move_stop(robot: Any) -> Any:
    for name in (
        "move_control_stop",
        "offline_track_stop",
        "teach_stop",
        "move_stop",
        "MoveStop",
        "stop",
        "robotServiceRobotMoveStop",
    ):
        if hasattr(robot, name):
            try:
                return {"method": name, "return": getattr(robot, name)(True)}
            except TypeError:
                try:
                    return {"method": name, "return": getattr(robot, name)()}
                except Exception:
                    pass
            except Exception:
                try:
                    return {"method": name, "return": getattr(robot, name)()}
                except Exception:
                    pass
    return {"method": None, "return": None, "error": "no supported stop function found"}


def wait_until_robot_static(
    robot: Any,
    *,
    step_tolerance_rad: float,
    settle_samples: int,
    timeout_s: float,
    poll_s: float,
    label: str = "robot",
) -> dict[str, Any]:
    """Wait until consecutive joint reads stop changing (truly static).

    A STRO stop command returns before the arm physically settles; reading the
    joints while it is still decelerating yields a candidate start up to
    ~1e-2 rad away from the true stopped pose (r01 start jitter).  Poll the
    real joint state and require ``settle_samples`` consecutive reads whose
    per-sample max step stays below ``step_tolerance_rad`` before returning.
    """
    started = time.perf_counter()
    last = np.asarray(robot.get_joint(), dtype=np.float64)
    quiet_samples = 0
    total_samples = 0
    last_step = float("inf")
    while time.perf_counter() - started < max(0.0, float(timeout_s)):
        current = np.asarray(robot.get_joint(), dtype=np.float64)
        total_samples += 1
        last_step = float(np.max(np.abs(current - last)))
        if last_step <= step_tolerance_rad:
            quiet_samples += 1
            if quiet_samples >= int(settle_samples):
                return {
                    "static": True,
                    "label": label,
                    "actual_joint": current.tolist(),
                    "max_step_rad": last_step,
                    "quiet_samples": quiet_samples,
                    "wait_s": time.perf_counter() - started,
                    "sample_count": total_samples,
                }
        else:
            quiet_samples = 0
        last = current
        time.sleep(max(0.0, float(poll_s)))
    return {
        "static": False,
        "label": label,
        "actual_joint": last.tolist(),
        "max_step_rad": last_step,
        "quiet_samples": quiet_samples,
        "wait_s": time.perf_counter() - started,
        "sample_count": total_samples,
    }


def filter_guard_clusters(clusters: list[Any], args: argparse.Namespace) -> list[Any]:
    """Keep only obstacle clusters in the dynamic-test guard workspace.

    The emergency point-cloud guard is intentionally more direct than CCRO/STRO,
    but in the real setup it can see low table/base/self-filter residuals near
    the shoulder. Those are not the handheld D1/D2 obstacles and can otherwise
    cause a false stop before the operator introduces the foam target.
    """
    kept = []
    for cluster in clusters:
        center = np.asarray(cluster.center, dtype=np.float64)
        if not np.all(np.isfinite(center)) or center.shape[0] < 3:
            continue
        if center[0] < args.guard_min_x or center[0] > args.guard_max_x:
            continue
        if center[1] < args.guard_min_y or center[1] > args.guard_max_y:
            continue
        if center[2] < args.guard_min_z or center[2] > args.guard_max_z:
            continue
        kept.append(cluster)
    return kept


def crop_points_roi(points: np.ndarray, roi: dict[str, float]) -> np.ndarray:
    """Keep only points inside an axis-aligned ROI box.

    roi keys: x_min, x_max, y_min, y_max, z_min, z_max.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        return pts
    mask = (
        (pts[:, 0] >= roi["x_min"])
        & (pts[:, 0] <= roi["x_max"])
        & (pts[:, 1] >= roi["y_min"])
        & (pts[:, 1] <= roi["y_max"])
        & (pts[:, 2] >= roi["z_min"])
        & (pts[:, 2] <= roi["z_max"])
    )
    return pts[mask]


def detect_tabletop_z(
    points: np.ndarray,
    *,
    xy_bounds: tuple[float, float, float, float],
    distance_threshold: float,
    min_plane_points: int,
    reference_xy: tuple[float, float] | None = None,
    z_sanity: tuple[float, float] = (0.10, 1.20),
    min_horizontal_normal_z: float = 0.85,
) -> tuple[float | None, dict[str, Any]]:
    """RANSAC-fit the dominant near-horizontal tabletop plane in a broad XY crop.

    Returns (table_z, audit) where table_z is the fitted plane height evaluated
    at the ROI center; None (with table_plane_valid False) when no suitable
    horizontal plane was found, in which case the caller falls back to the
    fixed Z band.  This is a per-frame, stateless estimate -- no caching, no
    gates -- exactly the tabletop-relative Z the two-layer ROI needs.
    """
    x_min, x_max, y_min, y_max = xy_bounds
    pts = np.asarray(points, dtype=np.float64)
    audit: dict[str, Any] = {"table_plane_valid": False}
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None, audit
    mask = (
        (pts[:, 0] >= x_min)
        & (pts[:, 0] <= x_max)
        & (pts[:, 1] >= y_min)
        & (pts[:, 1] <= y_max)
    )
    crop = pts[mask]
    audit["crop_point_count"] = int(len(crop))
    if len(crop) < int(min_plane_points):
        return None, audit
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(crop)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=float(distance_threshold),
        ransac_n=3,
        num_iterations=50,
    )
    if len(inliers) < int(min_plane_points):
        return None, audit
    normal = np.asarray(plane_model[:3], dtype=np.float64)
    norm_len = float(np.linalg.norm(normal))
    if norm_len <= 1.0e-12:
        return None, audit
    normal = normal / norm_len
    audit["plane_inliers"] = int(len(inliers))
    audit["plane_normal"] = normal.tolist()
    # The tabletop is roughly horizontal; a wall that wins RANSAC has a large
    # transverse normal and is rejected before it distorts the Z band.
    if abs(float(normal[2])) < float(min_horizontal_normal_z):
        return None, audit
    # plane model: a*x + b*y + c*z + d = 0  ->  z = -(a*x + b*y + d)/c
    a, b, c, d = (float(value) for value in plane_model)
    if abs(c) <= 1.0e-9:
        return None, audit
    if reference_xy is None:
        reference_xy = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
    z_table = -(a * reference_xy[0] + b * reference_xy[1] + d) / c
    lo, hi = z_sanity
    if not (float(lo) <= z_table <= float(hi)):
        return None, audit
    audit.update(
        {
            "table_plane_valid": True,
            "table_z_m": float(z_table),
            "reference_xy": [float(reference_xy[0]), float(reference_xy[1])],
        }
    )
    return float(z_table), audit


def resolve_planning_roi(
    args: argparse.Namespace,
    table_z: float | None,
    table_valid: bool,
) -> dict[str, Any]:
    """Planning/task ROI applied before clustering (V3 frozen bounds).

    Only the workspace the arm really operates in reaches clustering, the
    persistent tracker, PCA multi-sphere and STRO/Fast.  Z is tabletop-relative
    when the table plane was detected, otherwise a fixed fallback band.
    """
    x_min = float(getattr(args, "planning_roi_x_min", 0.10))
    x_max = float(getattr(args, "planning_roi_x_max", 0.85))
    y_min = float(getattr(args, "planning_roi_y_min", -0.50))
    y_max = float(getattr(args, "planning_roi_y_max", 0.50))
    if table_valid and table_z is not None:
        z_min = table_z + float(
            getattr(args, "planning_roi_z_table_offset_lo", 0.05)
        )
        z_max = table_z + float(
            getattr(args, "planning_roi_z_table_offset_hi", 0.80)
        )
        table_relative = True
    else:
        z_min = float(getattr(args, "planning_roi_z_fallback_lo", 0.40))
        z_max = float(getattr(args, "planning_roi_z_fallback_hi", 0.90))
        table_relative = False
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "z_min": z_min,
        "z_max": z_max,
        "table_relative": table_relative,
        "table_z_m": float(table_z) if table_z is not None else None,
    }


def resolve_safety_roi(
    args: argparse.Namespace,
    table_z: float | None,
    table_valid: bool,
) -> dict[str, Any]:
    """Broad safety ROI for the 0.10 m raw hard guard.

    Deliberately wider than the planning ROI so an obstacle that slips just
    outside the task box does not vanish from the hard-guard world; it only
    removes obviously below-table, far-wall and out-of-workspace points.
    """
    x_min = float(getattr(args, "safety_roi_x_min", 0.00))
    x_max = float(getattr(args, "safety_roi_x_max", 0.85))
    y_min = float(getattr(args, "safety_roi_y_min", -0.65))
    y_max = float(getattr(args, "safety_roi_y_max", 0.65))
    if table_valid and table_z is not None:
        z_min = table_z + float(getattr(args, "safety_roi_z_table_offset_lo", 0.00))
        z_max = table_z + float(getattr(args, "safety_roi_z_table_offset_hi", 0.90))
        table_relative = True
    else:
        z_min = float(getattr(args, "safety_roi_z_fallback_lo", 0.30))
        z_max = float(getattr(args, "safety_roi_z_fallback_hi", 1.10))
        table_relative = False
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "z_min": z_min,
        "z_max": z_max,
        "table_relative": table_relative,
        "table_z_m": float(table_z) if table_z is not None else None,
    }


def apply_two_layer_roi(
    points: np.ndarray,
    args: argparse.Namespace,
    *,
    need_planning: bool = True,
    need_safety: bool = True,
) -> dict[str, Any]:
    """Crop the denoised scene cloud into the planning ROI and the broad
    safety ROI, resolving the tabletop-relative Z band from a RANSAC plane fit.

    Returns both cropped clouds plus an audit payload: raw/roi point counts,
    the retain ratio and the resolved bounds.  No hard gates -- this only
    restricts what clustering (and therefore tracker / STRO / PCA / Fast) and
    the raw hard guard can see.  The table plane is detected from the broad
    safety XY crop so the two Z bands stay consistent with each other.
    """
    pts = np.asarray(points, dtype=np.float64)
    raw_count = int(len(pts))
    if getattr(args, "remove_planes", True):
        safety_xy = (
            float(getattr(args, "safety_roi_x_min", 0.00)),
            float(getattr(args, "safety_roi_x_max", 0.85)),
            float(getattr(args, "safety_roi_y_min", -0.65)),
            float(getattr(args, "safety_roi_y_max", 0.65)),
        )
        table_z, table_audit = detect_tabletop_z(
            pts,
            xy_bounds=safety_xy,
            distance_threshold=float(
                getattr(args, "roi_table_distance_threshold", args.plane_dist)
            ),
            min_plane_points=int(getattr(args, "roi_table_min_plane_points", 150)),
        )
    else:
        table_z, table_audit = None, {"table_plane_valid": False}
    table_valid = bool(table_audit.get("table_plane_valid", False))
    planning_roi = resolve_planning_roi(args, table_z, table_valid)
    safety_roi = resolve_safety_roi(args, table_z, table_valid)
    result: dict[str, Any] = {
        "planning_roi": planning_roi,
        "safety_roi": safety_roi,
        "table_audit": table_audit,
        "raw_point_count": raw_count,
        "planning_roi_point_count": 0,
        "safety_roi_point_count": 0,
        "rho_retain": 0.0,
        "planning_points": (
            crop_points_roi(pts, planning_roi) if need_planning else pts[:0].copy()
        ),
        "safety_points": (
            crop_points_roi(pts, safety_roi) if need_safety else pts[:0].copy()
        ),
    }
    if need_planning:
        result["planning_roi_point_count"] = int(len(result["planning_points"]))
    if need_safety:
        result["safety_roi_point_count"] = int(len(result["safety_points"]))
    if raw_count > 0:
        numerator = (
            result["planning_roi_point_count"]
            if need_planning
            else result["safety_roi_point_count"]
        )
        result["rho_retain"] = float(numerator / raw_count)
    return result


def guided_guard_distance(
    robot_points: np.ndarray,
    clusters: list[Any],
    tracked_objects: list[Any],
    *,
    motion_dir_y: float,
) -> dict[str, Any]:
    distance, obj, obj_id, robot_pt, obs_pt = _find_nearest_cluster_distance_detail(
        robot_points,
        clusters,
        tracked_objects,
    )
    in_motion_dir = _is_obstacle_in_motion_direction(robot_pt, obs_pt, motion_dir_y)
    return {
        "distance": float(distance),
        "object": obj if in_motion_dir or distance <= 0.08 else None,
        "object_id": obj_id if in_motion_dir or distance <= 0.08 else None,
        "raw_object_id": obj_id,
        "in_motion_direction": bool(in_motion_dir),
        "robot_point": None if robot_pt is None else np.asarray(robot_pt, dtype=np.float64).tolist(),
        "obstacle_point": None if obs_pt is None else np.asarray(obs_pt, dtype=np.float64).tolist(),
    }


def fit_fresh_obstacle_motion(samples: list[dict[str, Any]], *, minimum_frames: int, minimum_span_s: float) -> dict[str, Any]:
    """Fit a conservative fresh obstacle state from associated post-stop observations."""
    if len(samples) < int(minimum_frames):
        return {"accepted": False, "reason": "insufficient_fresh_frames", "sample_count": len(samples)}
    times = np.asarray([float(item["timestamp"]) for item in samples], dtype=np.float64)
    centers = np.asarray([item["center"] for item in samples], dtype=np.float64)
    span = float(times[-1] - times[0])
    if span < float(minimum_span_s):
        return {"accepted": False, "reason": "insufficient_fresh_time_span", "sample_count": len(samples), "span_s": span}
    centered_times = times - float(np.mean(times))
    denominator = float(np.dot(centered_times, centered_times))
    if denominator <= 1.0e-12:
        return {"accepted": False, "reason": "degenerate_fresh_timestamps", "sample_count": len(samples), "span_s": span}
    velocity = np.sum(centered_times[:, None] * (centers - np.mean(centers, axis=0)), axis=0) / denominator
    center = centers[-1].copy()
    radius = max(float(item["radius"]) for item in samples)
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(velocity)) or not np.isfinite(radius):
        return {"accepted": False, "reason": "nonfinite_fresh_state", "sample_count": len(samples), "span_s": span}
    return {
        "accepted": True,
        "reason": "fresh_obstacle_ready",
        "sample_count": len(samples),
        "span_s": span,
        "center": center,
        "velocity": velocity,
        "speed_m_s": float(np.linalg.norm(velocity)),
        "radius": radius,
        "max_association_error_m": max(float(item["association_error_m"]) for item in samples),
        "first_timestamp": float(times[0]),
        "last_timestamp": float(times[-1]),
    }


def associate_fresh_cluster(
    cluster_centers: list[np.ndarray],
    samples: list[dict[str, Any]],
    *,
    timestamp: float,
    trigger_cluster_center: np.ndarray,
    trigger_velocity: np.ndarray,
    trigger_timestamp: float,
    bootstrap_threshold_m: float,
    continuity_threshold_m: float,
) -> dict[str, Any]:
    """Associate one fresh frame using CV/hold bootstrap then frame continuity."""
    centers = [np.asarray(center, dtype=np.float64) for center in cluster_centers]
    if not centers:
        return {
            "associated": False,
            "association_mode": "fresh_continuity" if samples else "trigger_bootstrap_v2",
            "association_threshold_m": float(continuity_threshold_m if samples else bootstrap_threshold_m),
            "cluster_index": None,
        }

    if samples:
        expected_center = np.asarray(samples[-1]["center"], dtype=np.float64)
        errors = [float(np.linalg.norm(center - expected_center)) for center in centers]
        index = int(np.argmin(errors))
        selected_error = errors[index]
        return {
            "associated": bool(selected_error <= continuity_threshold_m),
            "association_mode": "fresh_continuity",
            "association_threshold_m": float(continuity_threshold_m),
            "association_error_m": selected_error,
            "selected_error_m": selected_error,
            "expected_center": expected_center,
            "candidate_cluster_center": centers[index],
            "cluster_index": index,
        }

    elapsed = max(0.0, float(timestamp) - float(trigger_timestamp))
    hold_center = np.asarray(trigger_cluster_center, dtype=np.float64)
    cv_center = hold_center + np.asarray(trigger_velocity, dtype=np.float64) * elapsed
    errors_cv = np.asarray([np.linalg.norm(center - cv_center) for center in centers], dtype=np.float64)
    errors_hold = np.asarray([np.linalg.norm(center - hold_center) for center in centers], dtype=np.float64)
    selected_errors = np.minimum(errors_cv, errors_hold)
    index = int(np.argmin(selected_errors))
    use_cv = bool(errors_cv[index] <= errors_hold[index])
    selected_error = float(selected_errors[index])
    return {
        "associated": bool(selected_error <= bootstrap_threshold_m),
        "association_mode": "trigger_bootstrap_v2",
        "association_threshold_m": float(bootstrap_threshold_m),
        "association_error_m": selected_error,
        "selected_error_m": selected_error,
        "bootstrap_model": "constant_velocity" if use_cv else "stopped_or_decelerated",
        "error_cv_m": float(errors_cv[index]),
        "error_hold_m": float(errors_hold[index]),
        "expected_center_cv": cv_center,
        "expected_center_hold": hold_center,
        "expected_center": cv_center if use_cv else hold_center,
        "candidate_cluster_center": centers[index],
        "cluster_index": index,
    }


def fit_pca_multisphere(points: np.ndarray, *, fit_margin_m: float = 0.005, max_components: int = 4) -> dict[str, Any]:
    """Cover one fresh cluster by consecutive PCA-axis spheres with a coverage audit."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 3 or not np.all(np.isfinite(values)):
        raise ValueError("fresh cluster points must be a finite (N,3) array with N >= 3")
    mean = np.mean(values, axis=0)
    centered = values - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = np.asarray(vh[0], dtype=np.float64)
    projection = centered @ axis
    axial_length = float(np.ptp(projection))
    transverse = centered - projection[:, None] * axis[None, :]
    transverse_radius = max(float(np.percentile(np.linalg.norm(transverse, axis=1), 90)), 1.0e-3)
    component_count = int(np.clip(np.ceil(axial_length / max(2.0 * transverse_radius, 1.0e-6)), 1, max_components))
    component_count = min(component_count, len(values))
    ordered_groups = np.array_split(np.argsort(projection), component_count)
    centers = []
    radii = []
    for indices in ordered_groups:
        local = values[indices]
        center = np.mean(local, axis=0)
        radius = float(np.max(np.linalg.norm(local - center[None, :], axis=1)) + fit_margin_m)
        centers.append(center)
        radii.append(radius)
    center_values = np.asarray(centers, dtype=np.float64)
    radius_values = np.asarray(radii, dtype=np.float64)
    signed_union_distance = np.min(
        np.linalg.norm(values[:, None, :] - center_values[None, :, :], axis=2) - radius_values[None, :],
        axis=1,
    )
    coverage_ratio = float(np.mean(signed_union_distance <= 1.0e-9))
    single_center = np.mean(values, axis=0)
    single_radius = float(np.max(np.linalg.norm(values - single_center[None, :], axis=1)))
    return {
        "source_point_count": int(len(values)),
        "component_count": int(component_count),
        "component_centers": center_values,
        "component_base_radii": radius_values,
        "pca_axis": axis,
        "axial_length_m": axial_length,
        "transverse_radius_m": transverse_radius,
        "fit_margin_m": float(fit_margin_m),
        "max_point_to_union_distance": float(np.max(signed_union_distance)),
        "coverage_ratio": coverage_ratio,
        "single_sphere_radius": single_radius,
        "multi_sphere_max_radius": float(np.max(radius_values)),
        "covered": bool(coverage_ratio >= 1.0 and np.max(signed_union_distance) <= 1.0e-9),
    }


def capture_post_stop_obstacle(
    processor: Any,
    state_reader: Any,
    denoiser: Any,
    args: argparse.Namespace,
    *,
    trigger_cluster_center: np.ndarray,
    trigger_velocity: np.ndarray,
    trigger_timestamp: float,
    stop_when_ready: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray | None]:
    """Acquire and associate fresh RGB-D obstacle observations after stopping."""
    started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    frame_audit: list[dict[str, Any]] = []
    latest_points: np.ndarray | None = None
    while time.perf_counter() - started < args.post_stop_recheck_duration_s:
        frame = processor.process_frame()
        timestamp = float(getattr(frame, "timestamp", time.time()))
        raw_scene_points = np.asarray(frame.scene_points, dtype=np.float64)
        scene_points = raw_scene_points
        robot_points = np.asarray(frame.robot_points, dtype=np.float64)
        frame_valid = bool(
            np.isfinite(timestamp)
            and raw_scene_points.ndim == 2
            and raw_scene_points.shape[1:] == (3,)
            and len(raw_scene_points) > 0
            and np.all(np.isfinite(raw_scene_points))
            and robot_points.ndim == 2
            and robot_points.shape[1:] == (3,)
            and len(robot_points) > 0
            and np.all(np.isfinite(robot_points))
        )
        if denoiser is not None:
            scene_points = denoiser.filter(scene_points)
        # Two-layer ROI: the tracker clusters come from the planning ROI; the
        # raw guard distance is measured on the broad safety ROI clusters so a
        # missed task-box point can still stop the robot.
        rois = apply_two_layer_roi(scene_points, args)
        plane_removal = None
        if args.remove_planes:
            plane_removal = {"enabled": True, "distance_threshold": args.plane_dist, "max_planes": args.max_planes}
        clustered = FastClusteringFilter(
            rois["planning_points"],
            robot_points,
            workspace=getattr(processor, "_workspace", None),
            # The planning Z band already excludes the tabletop.  Re-fitting
            # a largest plane here can remove a face-on foam obstacle during
            # the post-stop Fresh capture, leaving zero planning clusters
            # even while the safety ROI still sees the object.  Keep plane
            # removal only on the broad safety/guard ROI below.
            plane_removal=None,
            eps=args.cluster_eps,
            min_samples=args.cluster_min_samples,
            min_points=args.cluster_min_points,
            min_volume=args.cluster_min_volume,
        )
        clusters = filter_guard_clusters(list(clustered.clusters), args)
        guard_clustered = FastClusteringFilter(
            rois["safety_points"],
            robot_points,
            workspace=getattr(processor, "_workspace", None),
            plane_removal=plane_removal,
            eps=args.cluster_eps,
            min_samples=args.cluster_min_samples,
            min_points=args.cluster_min_points,
            min_volume=args.cluster_min_volume,
        )
        guard_clusters = filter_guard_clusters(list(guard_clustered.clusters), args)
        cluster_summaries = []
        for cluster in clusters:
            cluster_points = np.asarray(cluster.points, dtype=np.float64)
            cluster_center = np.asarray(cluster.center, dtype=np.float64)
            cluster_summaries.append(
                {
                    "center": cluster_center.tolist(),
                    "radius_m": float(
                        np.max(np.linalg.norm(cluster_points - cluster_center, axis=1))
                        if len(cluster_points)
                        else 0.0
                    ),
                    "point_count": int(len(cluster_points)),
                }
            )
        frame_guard_distance, _, _, _, _ = _find_nearest_cluster_distance_detail(robot_points, guard_clusters, [])
        common_audit = {
            "timestamp": timestamp,
            "frame_valid": frame_valid,
            "raw_scene_point_count": int(len(raw_scene_points)) if raw_scene_points.ndim == 2 else 0,
            "robot_point_count": int(len(robot_points)) if robot_points.ndim == 2 else 0,
            "raw_point_count": rois["raw_point_count"],
            "roi_point_count": rois["planning_roi_point_count"],
            "safety_roi_point_count": rois["safety_roi_point_count"],
            "rho_retain": rois["rho_retain"],
            "planning_roi": rois["planning_roi"],
            "safety_roi": rois["safety_roi"],
            "table_plane_valid": bool(rois["table_audit"].get("table_plane_valid", False)),
            "cluster_count": len(clusters),
            "guard_cluster_count": len(guard_clusters),
            "all_external_clusters": cluster_summaries,
            "raw_guard_distance_m": float(frame_guard_distance),
        }
        if not clusters:
            frame_audit.append(
                {
                    **common_audit,
                    "associated": False,
                    "association_mode": "fresh_continuity" if samples else "trigger_bootstrap_v2",
                }
            )
            continue
        association = associate_fresh_cluster(
            [np.asarray(cluster.center, dtype=np.float64) for cluster in clusters],
            samples,
            timestamp=timestamp,
            trigger_cluster_center=np.asarray(trigger_cluster_center, dtype=np.float64),
            trigger_velocity=np.asarray(trigger_velocity, dtype=np.float64),
            trigger_timestamp=trigger_timestamp,
            bootstrap_threshold_m=args.dynamic_tracker_association_distance_m,
            continuity_threshold_m=args.max_track_cluster_association_m,
        )
        index = int(association["cluster_index"])
        error = float(association["association_error_m"])
        frame_audit.append(
            {
                **common_audit,
                **{
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in association.items()
                    if key != "cluster_index"
                },
            }
        )
        if not association["associated"]:
            continue
        latest_points = np.asarray(clusters[index].points, dtype=np.float64).copy()
        tracking_center = np.asarray(clusters[index].center, dtype=np.float64)
        tracking_radius = float(
            np.max(np.linalg.norm(latest_points - tracking_center, axis=1))
            if len(latest_points)
            else 0.0
        )
        frame_audit[-1]["center"] = tracking_center.tolist()
        frame_audit[-1]["radius"] = tracking_radius
        samples.append(
            {
                "timestamp": timestamp,
                "center": tracking_center,
                "radius": tracking_radius,
                "association_error_m": error,
            }
        )
        if (
            stop_when_ready
            and len(samples) >= args.post_stop_recheck_min_frames
            and float(samples[-1]["timestamp"] - samples[0]["timestamp"]) >= args.post_stop_recheck_min_span_s
        ):
            break
    result = fit_fresh_obstacle_motion(
        samples,
        minimum_frames=args.post_stop_recheck_min_frames,
        minimum_span_s=args.post_stop_recheck_min_span_s,
    )
    result["capture_elapsed_s"] = time.perf_counter() - started
    result["frame_count"] = len(frame_audit)
    return result, frame_audit, latest_points


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.scene not in SCENARIOS:
        raise ValueError(f"unknown scene {args.scene}")
    if args.visualize_audit and args.mode not in {"dynamic-track-audit", "shadow"}:
        raise ValueError("--visualize-audit is restricted to non-commanding dynamic-track-audit/shadow modes")
    trial_dir = build_trial_dir(args)
    trial_dir.mkdir(parents=True, exist_ok=True)
    protocol_violations = formal_protocol_violations(args) if args.mode in ROBOT_MOTION_MODES else []
    if protocol_violations:
        blocked = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment": "6.5.3 real dynamic Fast CCRO-NUBS",
            "scene": args.scene,
            "scene_name": SCENARIOS[args.scene]["name"],
            "repeat": args.repeat,
            "mode": args.mode,
            "status": "BLOCKED_NONFORMAL_PROTOCOL",
            "robot_commanded": False,
            "formal_protocol": FORMAL_PROTOCOL,
            "formal_protocol_id": FORMAL_PROTOCOL_ID,
            "violations": protocol_violations,
            "parameters": vars(args),
            "git_commit": git_commit_hash(),
            "git_dirty": git_is_dirty(),
        }
        write_json(trial_dir / "summary.json", blocked)
        raise RuntimeError("formal D1/D2 protocol mismatch: " + "; ".join(protocol_violations))
    config_live = load_config_dir(args.config_dir)
    safety = config_live["safety"]
    policy = SafetyPolicy(
        d_safe=float(safety.get("d_safe", 0.15)),
        d_slow=float(safety.get("d_slow", 0.10)),
        d_stop=float(safety.get("d_stop", 0.05)),
    )
    live_model = load_live_surface_model(args.config_dir, args.urdf)
    stage4_config = load_stage4_config(args.stage4_config)
    stage4_model = load_stage4_surface_model(stage4_config)

    log: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "6.5.3 real dynamic Fast CCRO-NUBS",
        "scene": args.scene,
        "scene_name": SCENARIOS[args.scene]["name"],
        "repeat": args.repeat,
        "mode": args.mode,
        "robot_commanded": False,
        "operator_phrase_ok": args.operator_phrase == REQUIRED_OPERATOR_PHRASE,
        "required_operator_phrase": REQUIRED_OPERATOR_PHRASE,
        "trial_dir": str(trial_dir),
        "task_geometry_id": args.task_geometry_id,
        "reference_x_offset_m": float(args.x_offset),
        "parameters": vars(args),
        "stro_trigger_horizon_s": float(args.stro_trigger_horizon_s),
        "execution_prediction_horizon_s": float(args.prediction_horizon_s),
        "formal_protocol": formal_protocol_signature(args),
        "formal_protocol_id": FORMAL_PROTOCOL_ID,
        "protocol_scene_independent": True,
        "risk_sphere_predictor": (
            "legacy_single_sphere_v2"
            if RISK_SPHERE_PREDICTOR is None
            else getattr(RISK_SPHERE_PREDICTOR, "__name__", type(RISK_SPHERE_PREDICTOR).__name__)
        ),
        "risk_trigger_requires_dynamic_track": bool(
            RISK_TRIGGER_REQUIRES_DYNAMIC_TRACK
        ),
        "events": [],
        "git_commit": git_commit_hash(),
        "git_dirty": git_is_dirty(),
    }
    if args.mode not in {"shadow", "dynamic-track-audit"} and args.operator_phrase != REQUIRED_OPERATOR_PHRASE:
        log["status"] = "BLOCKED_BAD_OPERATOR_PHRASE"
        write_json(trial_dir / "summary.json", log)
        raise RuntimeError(f"bad operator phrase; required: {REQUIRED_OPERATOR_PHRASE}")
    if args.mode == "live-execute":
        log["status"] = "BLOCKED_LIVE_SWITCH_NOT_ENABLED"
        log["reason"] = "Use moving-shadow-stop first; online trajectory switch execution is intentionally not enabled in this first implementation."
        write_json(trial_dir / "summary.json", log)
        print(json.dumps(log, indent=2, ensure_ascii=False, default=json_default))
        return log

    processor = SceneProcessor(
        config_dir=str(args.config_dir),
        urdf_path=str(args.urdf),
        width=args.width,
        height=args.height,
        threshold=args.self_filter_threshold,
        voxel_size=args.voxel_size,
        use_real_robot=True,
        use_mock_camera=False,
    )
    state_reader = getattr(processor, "_state_reader", None)
    if state_reader is None or type(state_reader).__name__ != "RealRobotStateReader":
        processor.stop()
        raise RuntimeError("real AUBO state reader is required")
    robot = getattr(state_reader, "sdk_module", None)

    safety_tracker = OccupancyTracker(
        association_distance=float(safety.get("association_distance", 0.20)),
        alpha=float(safety.get("velocity_alpha", 0.3)),
        pos_alpha=float(safety.get("pos_alpha", 0.3)),
        motion_gate=float(safety.get("motion_gate", 0.005)),
        velocity_dead_zone=float(safety.get("velocity_dead_zone", 0.01)),
        shape_alpha=float(safety.get("shape_alpha", 0.4)),
    )
    dynamic_tracker = OccupancyTracker(
        association_distance=args.dynamic_tracker_association_distance_m,
        alpha=float(safety.get("velocity_alpha", 0.3)),
        pos_alpha=float(safety.get("pos_alpha", 0.3)),
        motion_gate_speed=args.dynamic_tracker_motion_gate_speed_m_s,
        velocity_dead_zone=float(safety.get("velocity_dead_zone", 0.01)),
        shape_alpha=float(safety.get("shape_alpha", 0.4)),
        max_miss=args.dynamic_tracker_max_miss,
    )
    denoiser = (
        TemporalDenoiser(args.denoise_voxel, args.denoise_conf, args.denoise_decay)
        if args.temporal_denoise
        else None
    )
    audit_visualizer = (
        AuditVisualizer(show_filtered=args.show_filtered, show_noise=args.show_noise)
        if args.visualize_audit
        else None
    )
    anomalous_cluster_files = 0

    rows: list[dict[str, Any]] = []
    (
        cluster_rows,
        track_rows,
        previous_cluster_centers,
        previous_cluster_timestamp,
        previous_dynamic_cluster_by_track,
    ) = new_dynamic_audit_buffers()
    q_history: list[tuple[float, np.ndarray]] = []
    candidate_summary: dict[str, Any] | None = None
    candidate_execution_summary: dict[str, Any] | None = None
    full_execution_summary: dict[str, Any] | None = None
    execution_path: str | None = None
    reference_resume_summary: dict[str, Any] | None = None
    reference_remainder_execution_summary: dict[str, Any] | None = None
    commander: RobotCommander | None = None
    guided_controller: AdaptiveSafetyController | None = None
    # The persistent perception worker is created only after a risk trigger.
    # Initialize it before entering the acquisition loop so the unconditional
    # cleanup path is also valid when the trial ends (or fails) before any
    # trigger occurs.
    persistent_worker = None
    triggered = False
    trigger_frame = None
    guard_stopped = False
    current_distance_stopped = False
    t0_wall = time.time()
    caught_error: dict[str, Any] | None = None
    ref_robot_points: np.ndarray | None = None
    ref_robot_motion_y: float | None = None
    robot_motion_modes = ROBOT_MOTION_MODES
    reference: RecordedReference | None = None
    if args.reference_feedback_csv is not None:
        reference = RecordedReference.load(args.reference_feedback_csv.resolve())
        log["reference_feedback_csv"] = str(args.reference_feedback_csv.resolve())
        log["reference_samples"] = len(reference.times)
        log["reference_local_increment_1s_p99_rad"] = reference.local_increment_p99(args.local_horizon_s)
    if args.mode in robot_motion_modes and reference is None:
        processor.stop()
        raise RuntimeError(
            "moving 6.5.3 modes require --reference-feedback-csv; velocity extrapolation and synthetic probe references are forbidden"
        )

    try:
        if args.mode in robot_motion_modes:
            if robot is None:
                raise RuntimeError("SceneProcessor state reader did not expose SDK module for guarded motion")
            step_label = "Step 1/2" if args.mode == "live-stop-replan-execute" else "Step 1/1"
            require_confirmation(
                True,
                f"{step_label}: workspace clear, operator at emergency stop. "
                f"Press Enter to start the one-way recorded Y reference and guarded logging for {args.duration_s:.1f}s. "
                "Only introduce the foam obstacle after the robot has started moving.",
            )
            commander = RobotCommander(ip=args.robot_ip, base_speed=args.line_velocity_m_s, robot_mod=robot)
            if not commander.connect(home_joints_deg=[float(v) for v in args.home_joints_deg.split(",")]):
                raise RuntimeError("RobotCommander failed to connect")
            commander.start_y_oscillate(
                range_m=args.guided_range_m,
                base_omega=args.line_velocity_m_s,
                x_offset=args.x_offset,
                one_way=True,
                y_start=args.y_start,
                y_goal=args.y_goal,
            )
            guided_controller = AdaptiveSafetyController(
                d_safe=args.guided_d_safe_m,
                d_slow=args.guided_d_slow_m,
                d_stop=args.guided_d_stop_m,
                max_decel=args.guided_max_decel,
                max_accel=args.guided_max_accel,
                dynamic_lookahead=args.guided_dynamic_lookahead_s,
            )
            log["robot_commanded"] = True
            log["reference_motion_method"] = "RobotCommander.motion_worker_y_one_way_reference"
            log["guided_range_m"] = args.guided_range_m
            log["reference_line_velocity_m_s"] = args.line_velocity_m_s
        elif not args.no_prompt:
            mode_note = "dynamic-track-audit（不发送运动命令）" if args.mode == "dynamic-track-audit" else "shadow"
            input(f"\n[{args.scene}] {mode_note}：{SCENARIOS[args.scene]['prompt']}。按 Enter 开始 {args.duration_s:.1f}s 采集...")

        started = time.perf_counter()
        reference_armed = args.mode not in robot_motion_modes
        reference_state = "RUNNING" if reference_armed else "WAIT_REFERENCE_START"
        reference_arm_perf: float | None = started if reference_armed else None
        reference_start_streak = 0
        reference_audit: dict[str, Any] = {
            "index": 0,
            "step": 0,
            "step_was_clamped": False,
            "time_s": 0.0,
            "tcp_y_m": None if reference is None or reference.y is None else float(reference.y[0]),
            "actual_y_error_m": None,
            "joint_match_max_rad": math.inf,
        }
        previous_frame_perf = started
        dynamic_speed_history: dict[int, list[tuple[float, np.ndarray]]] = {}
        dynamic_valid_streak: dict[int, int] = {}
        dynamic_state: dict[int, bool] = {}
        dynamic_low_speed_streak: dict[int, int] = {}
        dynamic_filtered_velocity: dict[int, np.ndarray] = {}
        frame_index = 0
        while True:
            frame_started = time.perf_counter()
            guard_stop_this_frame = False
            if not reference_armed and frame_started - started > args.reference_preparation_timeout_s:
                raise RuntimeError("reference start was not reached before preparation timeout")
            if reference_armed and reference_arm_perf is not None and frame_started - reference_arm_perf >= args.duration_s:
                break
            frame = processor.process_frame()
            timestamp = float(getattr(frame, "timestamp", time.time()))
            scene_points = np.asarray(frame.scene_points, dtype=np.float64)
            robot_points = np.asarray(frame.robot_points, dtype=np.float64)
            if denoiser is not None:
                scene_points = denoiser.filter(scene_points)
            joints, q = q_from_reader(state_reader)
            now_rel = time.perf_counter() - started
            actual_pose = list(robot.get_status()) if robot is not None else [math.nan] * 6
            actual_y = float(actual_pose[1])
            q_history.append((now_rel, q.copy()))
            q_history = q_history[-20:]
            qd = estimate_qd(q_history)
            if reference is not None and not reference_armed and args.mode in robot_motion_modes:
                start_y_error = math.inf if reference.y is None else abs(actual_y - float(reference.y[0]))
                start_joint_error = float(np.max(np.abs(q - reference.q[0])))
                start_ok = (
                    start_y_error <= args.reference_start_y_tolerance_m
                    and start_joint_error <= args.reference_start_joint_tolerance_rad
                )
                reference_start_streak = reference_start_streak + 1 if start_ok else 0
                reference_audit.update(
                    {
                        "actual_y_error_m": start_y_error,
                        "joint_match_max_rad": start_joint_error,
                    }
                )
                if reference_start_streak >= args.reference_start_consecutive_frames:
                    reference.reset()
                    reference_armed = True
                    reference_state = "REFERENCE_ARMED"
                    reference_arm_perf = time.perf_counter()
                    q_history.clear()
                    dynamic_speed_history.clear()
                    dynamic_valid_streak.clear()
                    dynamic_state.clear()
                    dynamic_low_speed_streak.clear()
                    dynamic_filtered_velocity.clear()
                    previous_dynamic_cluster_by_track.clear()
                    safety_tracker = OccupancyTracker(
                        association_distance=float(safety.get("association_distance", 0.20)),
                        alpha=float(safety.get("velocity_alpha", 0.3)),
                        pos_alpha=float(safety.get("pos_alpha", 0.3)),
                        motion_gate=float(safety.get("motion_gate", 0.005)),
                        velocity_dead_zone=float(safety.get("velocity_dead_zone", 0.01)),
                        shape_alpha=float(safety.get("shape_alpha", 0.4)),
                    )
                    dynamic_tracker = OccupancyTracker(
                        association_distance=args.dynamic_tracker_association_distance_m,
                        alpha=float(safety.get("velocity_alpha", 0.3)),
                        pos_alpha=float(safety.get("pos_alpha", 0.3)),
                        motion_gate_speed=args.dynamic_tracker_motion_gate_speed_m_s,
                        velocity_dead_zone=float(safety.get("velocity_dead_zone", 0.01)),
                        shape_alpha=float(safety.get("shape_alpha", 0.4)),
                        max_miss=args.dynamic_tracker_max_miss,
                    )
                    log["events"].append(
                        {
                            "type": "REFERENCE_ARMED",
                            "frame": frame_index,
                            "t_s": now_rel,
                            "tcp_y": actual_y,
                            "reference_index": 0,
                            "max_joint_error_rad": start_joint_error,
                        }
                    )
                    print(
                        f"[REFERENCE ARMED] introduce {args.scene} foam obstacle now: "
                        f"{SCENARIOS[args.scene]['prompt']}",
                        flush=True,
                    )
            elif reference is not None and reference_armed:
                loop_dt = max(frame_started - previous_frame_perf, 1.0e-3)
                time_based_step = int(math.ceil(args.reference_step_slack * loop_dt / reference.dt_median))
                max_step = min(args.max_reference_step, max(1, time_based_step))
                reference_audit = reference.locate(
                    q,
                    y_actual=actual_y,
                    max_forward_step=max_step,
                    joint_refine_window=args.reference_joint_refine_window,
                )
                reference_state = "RUNNING"
            previous_frame_perf = frame_started
            # Two-layer ROI before clustering: the task trackers / STRO / PCA /
            # Fast only see the planning ROI; the raw hard guard measures the
            # broad safety ROI so a near-miss just outside the task box still
            # stops the robot instead of vanishing from the guard world.
            rois = apply_two_layer_roi(scene_points, args)
            plane_removal = None
            if args.remove_planes:
                plane_removal = {"enabled": True, "distance_threshold": args.plane_dist, "max_planes": args.max_planes}
            # planning filter must NOT re-fit a "largest plane" inside the
            # planning ROI: the planning Z band already excludes the table
            # (z_min = table_z + 0.05), so the flat obstacle panel becomes the
            # dominant plane there and RANSAC removes it entirely (r04: 0
            # clusters from frame 85 while the safety ROI kept it). The safety
            # filter keeps plane removal because the table wins that fit first.
            cluster_result = FastClusteringFilter(
                rois["planning_points"],
                robot_points,
                workspace=getattr(processor, "_workspace", None),
                plane_removal=None,
                eps=args.cluster_eps,
                min_samples=args.cluster_min_samples,
                min_points=args.cluster_min_points,
                min_volume=args.cluster_min_volume,
            )
            clusters = list(cluster_result.clusters)
            eval_clusters = filter_guard_clusters(clusters, args) if args.mode in robot_motion_modes else clusters
            safety_cluster_result = FastClusteringFilter(
                rois["safety_points"],
                robot_points,
                workspace=getattr(processor, "_workspace", None),
                plane_removal=plane_removal,
                eps=args.cluster_eps,
                min_samples=args.cluster_min_samples,
                min_points=args.cluster_min_points,
                min_volume=args.cluster_min_volume,
            )
            guard_clusters = (
                filter_guard_clusters(list(safety_cluster_result.clusters), args)
                if args.mode in robot_motion_modes
                else list(safety_cluster_result.clusters)
            )
            dynamic_clusters, cluster_audits = dynamic_cluster_inputs(eval_clusters, args)
            cluster_dt = None if previous_cluster_timestamp is None else max(timestamp - previous_cluster_timestamp, 1.0e-6)
            for audit in cluster_audits:
                raw_speed = math.nan
                if cluster_dt is not None and previous_cluster_centers:
                    displacement = min(float(np.linalg.norm(audit["center"] - old)) for old in previous_cluster_centers)
                    if displacement <= args.dynamic_tracker_association_distance_m:
                        raw_speed = displacement / cluster_dt
                bbox = audit["bbox"]
                center_audit = audit["center"]
                cluster_rows.append(
                    {
                        "frame": frame_index, "t_s": f"{now_rel:.6f}",
                        "cluster_index": audit["cluster_index"],
                        "dynamic_tracker_input": 1,
                        "radius_in_legacy_band": int(audit["radius_in_legacy_band"]),
                        "point_count": audit["point_count"],
                        "center_x": f"{center_audit[0]:.6f}", "center_y": f"{center_audit[1]:.6f}", "center_z": f"{center_audit[2]:.6f}",
                        "raw_radius_m": f"{audit['radius']:.6f}",
                        "bbox_dx_m": f"{bbox[0]:.6f}", "bbox_dy_m": f"{bbox[1]:.6f}", "bbox_dz_m": f"{bbox[2]:.6f}",
                        "raw_centroid_speed_m_s": "" if not np.isfinite(raw_speed) else f"{raw_speed:.6f}",
                    }
                )
            previous_cluster_centers = [audit["center"].copy() for audit in cluster_audits]
            previous_cluster_timestamp = timestamp
            safety_detections = [
                make_occupancy_object(cluster.points, timestamp=timestamp, margin=float(safety.get("shape_margin", 0.02)))
                for cluster in eval_clusters
            ]
            dynamic_detections = [
                make_occupancy_object(cluster.points, timestamp=timestamp, margin=0.0)
                for cluster in dynamic_clusters
            ]
            tracked = safety_tracker.update(safety_detections, timestamp=timestamp)
            dynamic_tracked = dynamic_tracker.update(dynamic_detections, timestamp=timestamp)
            stable = [obj for obj in dynamic_tracked if obj.age >= args.min_track_age]
            dynamic_valid_tracks, dynamic_audits = update_dynamic_track_validity(
                dynamic_tracked,
                dynamic_clusters,
                dynamic_speed_history,
                dynamic_valid_streak,
                args,
                dynamic_state,
                dynamic_low_speed_streak,
                timestamp=timestamp,
                filtered_velocity_state=dynamic_filtered_velocity,
            )
            prediction_tracks = make_prediction_ready_objects(dynamic_tracked, dynamic_audits)
            for track_id, audit in dynamic_audits.items():
                cluster_center = audit["associated_cluster_center"]
                raw_cluster_speed = math.nan
                previous = previous_dynamic_cluster_by_track.get(track_id)
                if cluster_center is not None and previous is not None:
                    raw_cluster_speed = float(np.linalg.norm(cluster_center - previous[1]) / max(timestamp - previous[0], 1.0e-6))
                if cluster_center is not None:
                    previous_dynamic_cluster_by_track[track_id] = (timestamp, np.asarray(cluster_center, dtype=np.float64).copy())
                track_rows.append(
                    {
                        "frame": frame_index, "t_s": f"{now_rel:.6f}", "track_id": track_id,
                        "age": audit["age"],
                        "center_x": f"{audit['center'][0]:.6f}", "center_y": f"{audit['center'][1]:.6f}", "center_z": f"{audit['center'][2]:.6f}",
                        "instant_speed_m_s": f"{audit['speed']:.6f}",
                        "window_speed_m_s": f"{audit['window_speed_m_s']:.6f}",
                        "median_speed_m_s": f"{audit['median_speed_m_s']:.6f}",
                        "raw_cluster_speed_m_s": "" if not np.isfinite(raw_cluster_speed) else f"{raw_cluster_speed:.6f}",
                        "window_velocity_x_m_s": f"{audit['window_velocity'][0]:.6f}",
                        "window_velocity_y_m_s": f"{audit['window_velocity'][1]:.6f}",
                        "window_velocity_z_m_s": f"{audit['window_velocity'][2]:.6f}",
                        "raw_window_speed_m_s": f"{audit['raw_window_speed_m_s']:.6f}",
                        "filtered_speed_m_s": f"{audit['filtered_speed_m_s']:.6f}",
                        "raw_window_velocity_x_m_s": f"{audit['raw_window_velocity'][0]:.6f}",
                        "raw_window_velocity_y_m_s": f"{audit['raw_window_velocity'][1]:.6f}",
                        "raw_window_velocity_z_m_s": f"{audit['raw_window_velocity'][2]:.6f}",
                        "velocity_ema_alpha": f"{audit['velocity_ema_alpha']:.3f}",
                        "cluster_radius_raw_m": f"{audit['raw_radius']:.6f}",
                        "tracked_radius_m": f"{audit['track_radius']:.6f}",
                        "risk_radius_m": f"{audit['risk_radius_m']:.6f}",
                        "raw_radius_m": f"{audit['raw_radius']:.6f}", "association_error_m": f"{audit['association_error_m']:.6f}",
                        "valid_streak": audit["valid_streak"], "dynamic_valid": int(audit["valid"]),
                        "dynamic_state": int(audit["dynamic_state"]),
                        "prediction_ready": int(audit["prediction_ready"]),
                        "block_reason": ",".join(audit["block_reasons"]),
                    }
                )
            if args.mode in {"dynamic-track-audit", "shadow"} and args.save_anomalous_clusters:
                anomalous_cluster_files += save_anomalous_audit_clusters(
                    trial_dir,
                    frame_index,
                    timestamp,
                    clusters,
                    dynamic_audits,
                    max_bbox_m=args.anomaly_bbox_m,
                    max_radius_m=args.anomaly_radius_m,
                )
            if audit_visualizer is not None and not audit_visualizer.update(
                robot_points, cluster_result, clusters, dynamic_audits
            ):
                log["events"].append({"type": "AUDIT_VISUALIZER_CLOSED", "frame": frame_index, "t_s": now_rel})
                break
            risk_spheres = build_runtime_risk_spheres(
                stable_objects=stable,
                prediction_tracks=prediction_tracks,
                dynamic_audits=dynamic_audits,
                clusters=eval_clusters,
                # Initial STRO only: use the dedicated early-lookahead
                # horizon.  The normal prediction_horizon_s remains 0.5 s
                # for downstream planning, Fresh authorization, and monitors.
                args=argparse.Namespace(
                    **{
                        **vars(args),
                        "prediction_horizon_s": float(args.stro_trigger_horizon_s),
                    }
                ),
                safety=safety,
            )
            current_best = nearest_cluster_to_links(live_model, q, eval_clusters, density=args.surface_density)
            predicted_best = (
                future_reference_sphere_distance(live_model, reference, risk_spheres, density=args.surface_density)
                if reference is not None and reference_armed
                else nearest_sphere_to_links(live_model, q, risk_spheres, density=args.surface_density)
            )
            predicted_audit = dynamic_audits.get(predicted_best.get("object_id"))
            center = current_best["cluster_center"]
            speeds = [float(np.linalg.norm(obj.velocity)) for obj in stable]
            guided_info = {
                "guard_distance_m": math.inf,
                "guard_object_id": None,
                "guard_in_motion_direction": False,
                "guard_speed_scale": 1.0,
                "guard_decision": "",
            }
            if args.mode in robot_motion_modes and guided_controller is not None and commander is not None:
                motion_y = commander.get_y_pos()
                motion_dir_y = 1.0
                if rows:
                    try:
                        prev_y = float(rows[-1].get("motion_y_m", "nan"))
                        if np.isfinite(prev_y) and abs(motion_y - prev_y) > 1.0e-5:
                            motion_dir_y = 1.0 if motion_y > prev_y else -1.0
                    except Exception:
                        pass
                # Use raw current clusters for the emergency guard. This mirrors
                # safety_guided_motion.py and intentionally does not wait for
                # STRO stable-track age. When self-filtering leaves too few robot
                # points during motion, fall back to the first reliable robot cloud
                # translated by the commanded Y displacement, matching the proven
                # guarded-motion script.
                guard_robot_points_source = "current"
                if len(robot_points) > 100:
                    rob_pts_for_guard = robot_points
                    if ref_robot_points is None:
                        ref_robot_points = robot_points.copy()
                        ref_robot_motion_y = motion_y
                elif ref_robot_points is not None and ref_robot_motion_y is not None:
                    dy = motion_y - ref_robot_motion_y
                    rob_pts_for_guard = ref_robot_points + np.array([0.0, dy, 0.0], dtype=np.float64)
                    guard_robot_points_source = "translated_reference"
                else:
                    rob_pts_for_guard = robot_points
                    guard_robot_points_source = "insufficient"

                guard = guided_guard_distance(rob_pts_for_guard, guard_clusters, tracked, motion_dir_y=motion_dir_y)
                guard_distance = guard["distance"]
                guard_obj = guard["object"]
                guard_speed_scale = guided_controller.evaluate(
                    guard_distance,
                    guard_obj,
                    None,
                    max((time.perf_counter() - frame_started), 0.03),
                )
                commander.set_speed_scale(guard_speed_scale)
                guided_info = {
                    "motion_y_m": motion_y,
                    "guard_distance_m": guard_distance,
                    "guard_object_id": guard["object_id"],
                    "guard_in_motion_direction": guard["in_motion_direction"],
                    "guard_speed_scale": guard_speed_scale,
                    "guard_decision": guided_controller.last_decision,
                    "guard_cluster_count": int(len(guard_clusters)),
                    "guard_robot_points_source": guard_robot_points_source,
                    "guard_robot_points_count": int(len(rob_pts_for_guard)),
                    "guard_robot_point": guard["robot_point"],
                    "guard_obstacle_point": guard["obstacle_point"],
                }
                if (
                    np.isfinite(guard_distance)
                    and guard_distance <= args.guided_hard_stop_m
                    and not guard_stopped
                ):
                    commander.set_speed_scale(0.0)
                    guard_stopped = True
                    guard_stop_this_frame = True
                    log["events"].append(
                        {
                            "type": "GUIDED_POINTCLOUD_GUARD_STOP",
                            "frame": frame_index,
                            "t_s": now_rel,
                            "guard_distance_m": guard_distance,
                            "threshold_m": args.guided_hard_stop_m,
                            # The hard stop is decided by the independent
                            # guard_distance <= guided_hard_stop_m check, not by
                            # the guided speed-scale controller, so do not echo
                            # its last (possibly "SAFE") decision as the stop
                            # reason.
                            "guard_decision": "HARD_STOP",
                            "guided_controller_decision": (
                                guided_controller.last_decision
                            ),
                            "guard_cluster_count": len(guard_clusters),
                        }
                    )
            future_index = 0
            future_time = 0.0
            future_delta_q = math.inf
            reference_match_ok = reference is None or not reference_armed
            local_reference_sanity_ok = False
            if reference is not None and reference_armed:
                future_index = reference.index_after(args.local_horizon_s)
                future_time = float(reference.times[future_index])
                future_delta_q = float(np.max(np.abs(reference.q[future_index] - q)))
                reference_match_ok = reference_audit["joint_match_max_rad"] <= args.reference_match_max_rad
                sanity_limit = args.local_reference_sanity_scale * reference.local_increment_p99(args.local_horizon_s)
                local_reference_sanity_ok = future_delta_q <= sanity_limit
            fallback_dynamic_audit = None
            if dynamic_audits:
                fallback_dynamic_audit = max(dynamic_audits.values(), key=lambda item: (item["valid_streak"], item["age"]))
            row_dynamic_audit = predicted_audit or fallback_dynamic_audit
            dynamic_block_reason = ""
            if row_dynamic_audit is not None and not row_dynamic_audit["valid"]:
                dynamic_block_reason = ",".join(row_dynamic_audit["block_reasons"])
            row = {
                "frame": frame_index,
                "t_s": f"{now_rel:.6f}",
                "timestamp": f"{timestamp:.6f}",
                "scene_points": int(len(scene_points)),
                "robot_points": int(len(robot_points)),
                "cluster_count": int(len(clusters)),
                "guard_cluster_count": int(len(guard_clusters)),
                "raw_point_count": int(rois["raw_point_count"]),
                "roi_point_count": int(rois["planning_roi_point_count"]),
                "safety_roi_point_count": int(rois["safety_roi_point_count"]),
                "rho_retain": f"{rois['rho_retain']:.6f}",
                "table_z_m": (
                    "" if rois["planning_roi"]["table_z_m"] is None
                    else f"{rois['planning_roi']['table_z_m']:.6f}"
                ),
                "table_plane_valid": int(bool(rois["table_audit"].get("table_plane_valid", False))),
                "stable_track_count": int(len(stable)),
                "risk_sphere_count": int(len(risk_spheres)),
                "nearest_distance_m": "" if math.isinf(current_best["distance"]) else f"{current_best['distance']:.6f}",
                "nearest_link": current_best["link"] or "",
                "nearest_cluster_index": "" if current_best["cluster_index"] is None else int(current_best["cluster_index"]),
                "nearest_cluster_x": "" if center is None else f"{center[0]:.6f}",
                "nearest_cluster_y": "" if center is None else f"{center[1]:.6f}",
                "nearest_cluster_z": "" if center is None else f"{center[2]:.6f}",
                "predicted_distance_m": "" if math.isinf(predicted_best["distance"]) else f"{predicted_best['distance']:.6f}",
                "predicted_nearest_link": predicted_best["link"] or "",
                "predicted_tau_s": "" if predicted_best["tau"] is None else f"{predicted_best['tau']:.3f}",
                "trigger_horizon_s": f"{float(args.stro_trigger_horizon_s):.3f}",
                "predicted_object_id": "" if predicted_best["object_id"] is None else int(predicted_best["object_id"]),
                "trigger_block_reason": "",
                "reference_state": reference_state,
                "reference_armed": int(reference_armed),
                "reference_index": reference_audit["index"],
                "reference_index_step": reference_audit["step"],
                "reference_step_clamped": int(bool(reference_audit.get("step_was_clamped", False))),
                "reference_time_s": f"{float(reference_audit['time_s']):.6f}",
                "reference_tcp_y_m": "" if reference_audit["tcp_y_m"] is None else f"{float(reference_audit['tcp_y_m']):.6f}",
                "reference_actual_y_error_m": "" if reference_audit["actual_y_error_m"] is None else f"{float(reference_audit['actual_y_error_m']):.6f}",
                "reference_joint_match_max_rad": "" if not np.isfinite(reference_audit["joint_match_max_rad"]) else f"{float(reference_audit['joint_match_max_rad']):.6f}",
                "reference_future_time_s": f"{future_time:.6f}",
                "reference_future_index": future_index,
                "reference_future_delta_q_max_rad": "" if not np.isfinite(future_delta_q) else f"{future_delta_q:.6f}",
                "predicted_object_speed_m_s": "" if row_dynamic_audit is None else f"{float(row_dynamic_audit['median_speed_m_s']):.6f}",
                "predicted_object_velocity_x_m_s": "" if row_dynamic_audit is None else f"{float(row_dynamic_audit['window_velocity'][0]):.6f}",
                "predicted_object_velocity_y_m_s": "" if row_dynamic_audit is None else f"{float(row_dynamic_audit['window_velocity'][1]):.6f}",
                "predicted_object_velocity_z_m_s": "" if row_dynamic_audit is None else f"{float(row_dynamic_audit['window_velocity'][2]):.6f}",
                "predicted_object_raw_speed_m_s": "" if row_dynamic_audit is None else f"{float(row_dynamic_audit['raw_window_speed_m_s']):.6f}",
                "predicted_object_filtered_speed_m_s": "" if row_dynamic_audit is None else f"{float(row_dynamic_audit['filtered_speed_m_s']):.6f}",
                "predicted_object_raw_velocity_x_m_s": "" if row_dynamic_audit is None else f"{float(row_dynamic_audit['raw_window_velocity'][0]):.6f}",
                "predicted_object_raw_velocity_y_m_s": "" if row_dynamic_audit is None else f"{float(row_dynamic_audit['raw_window_velocity'][1]):.6f}",
                "predicted_object_raw_velocity_z_m_s": "" if row_dynamic_audit is None else f"{float(row_dynamic_audit['raw_window_velocity'][2]):.6f}",
                "predicted_object_velocity_ema_alpha": "" if row_dynamic_audit is None else f"{float(row_dynamic_audit['velocity_ema_alpha']):.3f}",
                "predicted_object_radius_m": "" if row_dynamic_audit is None else f"{float(row_dynamic_audit['raw_radius']):.6f}",
                "predicted_object_age": "" if row_dynamic_audit is None else int(row_dynamic_audit["age"]),
                "predicted_object_association_error_m": "" if row_dynamic_audit is None else f"{float(row_dynamic_audit['association_error_m']):.6f}",
                "dynamic_object_prediction_ready": int(bool(predicted_audit and predicted_audit["prediction_ready"])),
                "dynamic_object_valid": int(bool(predicted_audit and predicted_audit["valid"])),
                "dynamic_object_block_reason": dynamic_block_reason,
                "risk_state_current": risk_color_level(policy, current_best["distance"]),
                "risk_state_predicted": risk_color_level(policy, predicted_best["distance"]),
                "max_track_speed_m_s": "" if not speeds else f"{max(speeds):.6f}",
                "motion_y_m": "" if "motion_y_m" not in guided_info else f"{guided_info['motion_y_m']:.6f}",
                "guard_distance_m": "" if math.isinf(guided_info["guard_distance_m"]) else f"{guided_info['guard_distance_m']:.6f}",
                "guard_object_id": "" if guided_info["guard_object_id"] is None else int(guided_info["guard_object_id"]),
                "guard_in_motion_direction": int(bool(guided_info["guard_in_motion_direction"])),
                "guard_speed_scale": f"{guided_info['guard_speed_scale']:.6f}",
                "guard_decision": guided_info["guard_decision"],
                "guard_cluster_count": guided_info.get("guard_cluster_count", len(guard_clusters) if args.mode == "moving-shadow-stop" else ""),
                "guard_robot_points_source": guided_info.get("guard_robot_points_source", ""),
                "guard_robot_points_count": guided_info.get("guard_robot_points_count", ""),
                "elapsed_ms": f"{(time.perf_counter() - frame_started) * 1000.0:.4f}",
                **{f"q{j+1}_rad": f"{q[j]:.8f}" for j in range(6)},
                **{f"qd{j+1}_rad_s": f"{qd[j]:.8f}" for j in range(6)},
            }
            rows.append(row)

            trigger_distance = float(predicted_best["distance"])
            trigger_threshold = (
                float(args.moving_shadow_replan_in_m)
                if args.mode in robot_motion_modes
                else float(args.replan_in_m)
            )
            current_distance = float(current_best["distance"])
            current_link = current_best["link"] or ""
            trigger_block_reason = ""
            if triggered:
                trigger_block_reason = "already_triggered"
            elif not reference_armed:
                trigger_block_reason = "reference_not_armed"
            elif args.reference_audit_only:
                trigger_block_reason = "reference_audit_only"
            elif args.mode == "dynamic-track-audit":
                trigger_block_reason = "dynamic_track_audit_only"
            elif reference_audit.get("step_was_clamped", False):
                trigger_block_reason = "reference_step_jump"
            elif not reference_match_ok:
                trigger_block_reason = "reference_match_error"
            elif not local_reference_sanity_ok:
                trigger_block_reason = "local_reference_sanity_failed"
            elif len(stable) == 0:
                trigger_block_reason = "no_stable_track"
            elif RISK_TRIGGER_REQUIRES_DYNAMIC_TRACK and len(prediction_tracks) == 0:
                trigger_block_reason = "predicted_track_not_dynamic"
            elif not RISK_TRIGGER_REQUIRES_DYNAMIC_TRACK and len(risk_spheres) == 0:
                trigger_block_reason = "no_risk_eligible_track"
            elif not np.isfinite(trigger_distance):
                trigger_block_reason = "no_finite_future_reference_risk"
            elif guard_stop_this_frame:
                trigger_block_reason = "guided_hard_guard_stop"
            elif trigger_distance >= trigger_threshold:
                trigger_block_reason = "future_reference_clearance_above_threshold"
            elif current_distance <= args.moving_shadow_current_stop_m:
                trigger_block_reason = "current_distance_in_stop_zone"
            elif reference_arm_perf is not None and time.perf_counter() - reference_arm_perf < args.arm_delay_s:
                trigger_block_reason = "arm_delay"
            row["trigger_block_reason"] = trigger_block_reason
            if guard_stop_this_frame:
                time.sleep(max(args.post_stop_settle_s, 0.0))
                break
            if (
                args.mode in robot_motion_modes
                and np.isfinite(current_distance)
                and current_distance <= args.moving_shadow_current_stop_m
            ):
                if commander is not None:
                    commander.set_speed_scale(0.0)
                    stop_ret = {"method": "RobotCommander.set_speed_scale", "return": 0}
                else:
                    stop_ret = maybe_move_stop(robot)
                log["events"].append(
                    {
                        "type": "IMMEDIATE_CURRENT_DISTANCE_STOP",
                        "frame": frame_index,
                        "t_s": now_rel,
                        "distance_m": current_distance,
                        "nearest_link": current_link,
                        "threshold_m": args.moving_shadow_current_stop_m,
                        "return": stop_ret,
                    }
                )
                current_distance_stopped = True
                break
            if (
                not triggered
                and trigger_block_reason == ""
            ):
                triggered = True
                trigger_frame = frame_index
                trigger_audit = dynamic_audits.get(predicted_best.get("object_id"))
                log["events"].append(
                    {
                        "type": "TRIGGER",
                        "frame": frame_index,
                        "t_s": now_rel,
                        "track_id": predicted_best.get("object_id"),
                        "predicted_distance_m": trigger_distance,
                        "predicted_tau_s": predicted_best.get("tau"),
                        "trigger_horizon_s": float(args.stro_trigger_horizon_s),
                        "predicted_link": predicted_best.get("link"),
                        "current_distance_m": current_distance,
                        "current_link": current_link,
                        "guard_distance_m": guided_info["guard_distance_m"],
                        "window_velocity_m_s": None if trigger_audit is None else trigger_audit["window_velocity"].tolist(),
                        "window_speed_m_s": None if trigger_audit is None else trigger_audit["window_speed_m_s"],
                        "tracked_radius_m": None if trigger_audit is None else trigger_audit["track_radius"],
                        "tracker_center_m": None if trigger_audit is None else trigger_audit["center"].tolist(),
                        "raw_cluster_center_m": None if trigger_audit is None else trigger_audit["associated_cluster_center"].tolist(),
                        "tracker_raw_center_offset_m": None if trigger_audit is None else float(
                            np.linalg.norm(trigger_audit["center"] - trigger_audit["associated_cluster_center"])
                        ),
                    }
                )
                if args.mode == "moving-shadow-stop":
                    # Pilot mode is for validating sensing/trigger timing during
                    # motion. Stop first; candidate generation is recorded only
                    # after the stop command so optimization latency cannot eat
                    # into the safety margin.
                    if commander is not None:
                        commander.set_speed_scale(0.0)
                        stop_ret = {"method": "RobotCommander.set_speed_scale", "return": 0}
                    else:
                        stop_ret = maybe_move_stop(robot)
                    log["events"].append(
                        {
                            "type": "IMMEDIATE_STOP_AFTER_TRIGGER",
                            "frame": frame_index,
                            "t_s": time.perf_counter() - started,
                            "predicted_distance_m": trigger_distance,
                            "threshold_m": trigger_threshold,
                            "trigger_horizon_s": float(args.stro_trigger_horizon_s),
                            "return": stop_ret,
                        }
                    )
                elif args.mode == "live-stop-replan-execute":
                    if commander is not None:
                        commander.set_speed_scale(0.0)
                        stop_ret = {"method": "RobotCommander.set_speed_scale", "return": 0}
                    else:
                        stop_ret = maybe_move_stop(robot)
                    log["events"].append(
                        {
                            "type": "IMMEDIATE_STOP_BEFORE_LIVE_REPLAN",
                            "frame": frame_index,
                            "t_s": time.perf_counter() - started,
                            "predicted_distance_m": trigger_distance,
                            "threshold_m": trigger_threshold,
                            "return": stop_ret,
                        }
                    )
                risk_selection_objects = (
                    prediction_tracks
                    if RISK_TRIGGER_REQUIRES_DYNAMIC_TRACK
                    else stable
                )
                selected_obj = select_stable_object(
                    risk_selection_objects, predicted_best, risk_spheres
                )
                obstacle = track_geometry(selected_obj, eval_clusters, args.default_obstacle_radius_m)
                trigger_audit = dynamic_audits.get(obstacle["track_id"])
                if not risk_track_is_eligible(
                    trigger_audit,
                    require_dynamic_track=RISK_TRIGGER_REQUIRES_DYNAMIC_TRACK,
                ):
                    raise RuntimeError("selected Fast obstacle is not risk-eligible")
                prediction_velocity = np.asarray(
                    trigger_audit["window_velocity"], dtype=np.float64
                ).copy()
                if not trigger_audit.get("dynamic_state", False):
                    prediction_velocity = np.zeros(3, dtype=np.float64)
                obstacle["window_velocity"] = prediction_velocity
                obstacle["prediction_ready"] = bool(trigger_audit["prediction_ready"])
                obstacle["risk_eligible"] = True
                obstacle["motion_class"] = (
                    "dynamic" if trigger_audit.get("dynamic_state", False) else "quasi_static"
                )
                if obstacle["association_error_m"] > args.max_track_cluster_association_m:
                    raise RuntimeError(
                        f"selected track/cluster association error {obstacle['association_error_m']:.4f} m "
                        f"exceeds {args.max_track_cluster_association_m:.4f} m"
                    )
                fresh_recheck = None
                persistent_worker = None
                if args.mode in robot_motion_modes:
                    if callable(PERSISTENT_OBSTACLE_WORKER_FACTORY):
                        cluster_index = int(obstacle["associated_cluster_index"])
                        fresh_latest_points = np.asarray(
                            eval_clusters[cluster_index].points, dtype=np.float64
                        )
                        multisphere_geometry = fit_pca_multisphere(
                            fresh_latest_points,
                            fit_margin_m=args.multisphere_fit_margin_m,
                            max_components=args.multisphere_max_components,
                        )
                        tracked_center = np.asarray(obstacle["center"], dtype=np.float64)
                        raw_center = np.mean(fresh_latest_points, axis=0)
                        multisphere_geometry = translated_multisphere_geometry(
                            multisphere_geometry, raw_center, tracked_center
                        )
                        history = list(
                            dynamic_speed_history.get(int(obstacle["track_id"]), [])
                        )
                        fresh_frames = [
                            {
                                "timestamp": float(sample_timestamp),
                                "associated": True,
                                "center": np.asarray(sample_center, dtype=np.float64).tolist(),
                                "radius": float(obstacle["raw_radius"]),
                                "association_error_m": float(
                                    obstacle["association_error_m"]
                                ),
                            }
                            for sample_timestamp, sample_center in history
                        ]
                        fresh_recheck = {
                            "accepted": bool(multisphere_geometry.get("covered", False)),
                            "reason": (
                                "persistent_tracker_seed_ready"
                                if multisphere_geometry.get("covered", False)
                                else "persistent_tracker_seed_geometry_failed"
                            ),
                            "track_id": int(obstacle["track_id"]),
                            "center": tracked_center.tolist(),
                            "velocity": np.asarray(
                                obstacle["window_velocity"], dtype=np.float64
                            ).tolist(),
                            "radius": float(obstacle["raw_radius"]),
                            "last_timestamp": float(timestamp),
                            "max_association_error_m": float(
                                obstacle["association_error_m"]
                            ),
                            "source": "pretrigger_tracker_continuation",
                            "history_frame_count": len(fresh_frames),
                        }
                        np.save(
                            trial_dir / "persistent_seed_cluster_points.npy",
                            fresh_latest_points,
                        )
                        write_json(
                            trial_dir / "persistent_seed_multisphere.json",
                            multisphere_geometry,
                        )
                        write_json(
                            trial_dir / "persistent_tracker_seed.json",
                            {"result": fresh_recheck, "frames": fresh_frames},
                        )
                        log["events"].append(
                            {
                                "type": (
                                    "PERSISTENT_TRACKER_SEED_READY"
                                    if fresh_recheck["accepted"]
                                    else "PERSISTENT_TRACKER_SEED_REJECTED"
                                ),
                                "frame": frame_index,
                                "t_s": time.perf_counter() - started,
                                "recheck": fresh_recheck,
                            }
                        )
                    else:
                        time.sleep(max(args.post_stop_settle_s, 0.0))
                        fresh_recheck, fresh_frames, fresh_latest_points = capture_post_stop_obstacle(
                            processor,
                            state_reader,
                            denoiser,
                            args,
                            trigger_cluster_center=np.asarray(obstacle["associated_cluster_center"], dtype=np.float64),
                            trigger_velocity=np.asarray(obstacle["window_velocity"], dtype=np.float64),
                            trigger_timestamp=timestamp,
                        )
                        multisphere_geometry = None
                        if fresh_recheck["accepted"] and fresh_latest_points is not None:
                            multisphere_geometry = fit_pca_multisphere(
                                fresh_latest_points,
                                fit_margin_m=args.multisphere_fit_margin_m,
                                max_components=args.multisphere_max_components,
                            )
                            if not multisphere_geometry["covered"]:
                                fresh_recheck = {
                                    **fresh_recheck,
                                    "accepted": False,
                                    "reason": "fresh_multisphere_coverage_failed",
                                }
                            np.save(trial_dir / "fresh_latest_cluster_points.npy", fresh_latest_points)
                            write_json(trial_dir / "fresh_multisphere.json", multisphere_geometry)
                        write_json(trial_dir / "post_stop_fresh_recheck.json", {"result": fresh_recheck, "frames": fresh_frames})
                        log["events"].append(
                            {
                                "type": "POST_STOP_FRESH_RECHECK_READY" if fresh_recheck["accepted"] else "POST_STOP_FRESH_RECHECK_REJECTED",
                                "frame": frame_index,
                                "t_s": time.perf_counter() - started,
                                "recheck": fresh_recheck,
                            }
                        )
                    if not fresh_recheck["accepted"]:
                        candidate_summary = {
                            "status": "REJECTED_FRESH_RECHECK",
                            "accepted_for_switch": False,
                            "rejection_reasons": [fresh_recheck["reason"]],
                            "fresh_recheck": fresh_recheck,
                        }
                        log["events"].append(
                            {"type": "REJECTED_FRESH_RECHECK", "frame": frame_index, "t_s": time.perf_counter() - started}
                        )
                        break
                    obstacle["trigger_center"] = np.asarray(obstacle["center"], dtype=np.float64).copy()
                    obstacle["trigger_velocity"] = np.asarray(obstacle["window_velocity"], dtype=np.float64).copy()
                    obstacle["trigger_inflated_radius"] = float(obstacle["inflated_radius"])
                    obstacle["center"] = np.asarray(fresh_recheck["center"], dtype=np.float64)
                    obstacle["window_velocity"] = np.asarray(fresh_recheck["velocity"], dtype=np.float64)
                    obstacle["inflated_radius"] = max(float(fresh_recheck["radius"]), args.default_obstacle_radius_m)
                    obstacle["fresh_recheck"] = fresh_recheck
                    obstacle["multisphere_geometry"] = multisphere_geometry
                if reference is None:
                    raise RuntimeError("a recorded reference is required to construct a Fast local repair")
                # Both active modes stop before repair. Anchor the candidate at
                # the measured stopped state, not at the last moving sample.
                if args.mode in robot_motion_modes:
                    # Fix 1: wait until the arm is truly static before reading
                    # q_stop_actual.  The STRO stop returns before the joints
                    # settle; reading q while still decelerating produces a
                    # candidate start up to ~1e-2 rad away from the true stop
                    # (r01 violent start jitter).  Planning must anchor on the
                    # measured stopped pose, never on a drifting read.
                    if robot is not None:
                        static_audit = wait_until_robot_static(
                            robot,
                            step_tolerance_rad=args.candidate_static_tolerance_rad,
                            settle_samples=2,
                            timeout_s=max(
                                3.0, float(args.candidate_pre_execute_settle_s)
                            ),
                            poll_s=args.poll_s,
                            label="post_stop_repair_start",
                        )
                        log["events"].append(
                            {
                                "type": "POST_STOP_WAIT_FOR_STATIC",
                                "frame": frame_index,
                                "t_s": time.perf_counter() - started,
                                "static": static_audit,
                            }
                        )
                        if not static_audit["static"]:
                            raise RuntimeError(
                                "robot did not reach a static state after the "
                                f"STRO stop: {static_audit}"
                            )
                    _, q_repair_start = q_from_reader(state_reader)
                    q_repair_start = np.asarray(q_repair_start, dtype=np.float64)
                    qd_repair_start = np.zeros(6, dtype=np.float64)
                    stopped_pose = list(robot.get_status()) if robot is not None else actual_pose
                    stop_audit = reference.locate(
                        q_repair_start,
                        y_actual=float(stopped_pose[1]),
                        max_forward_step=args.max_reference_step,
                        joint_refine_window=args.reference_joint_refine_window,
                    )
                    if stop_audit["joint_match_max_rad"] > args.reference_match_max_rad:
                        raise RuntimeError(
                            f"stopped state/reference mismatch {stop_audit['joint_match_max_rad']:.4f} rad"
                        )
                else:
                    q_repair_start = q
                    qd_repair_start = qd
                reference_repair_start_time_s = float(reference.times[reference.index])
                reference_goal = reference.state_after(args.local_horizon_s)
                rejoin_offsets = np.arange(
                    args.local_horizon_s + args.rejoin_search_step_s,
                    args.rejoin_max_offset_s + 0.5 * args.rejoin_search_step_s,
                    args.rejoin_search_step_s,
                )
                rejoin_goals = [
                    (float(offset), reference.state_after(float(offset)))
                    for offset in rejoin_offsets
                ]
                # Keep one explicit all-link set for the complete stopped
                # repair event.  The post-local Fresh #3 continuation handler
                # runs later in this same event and must receive the identical
                # safety scope; previously the two planner calls constructed
                # this set inline, leaving the handler with an undefined local
                # name after a successful real local execution.
                risk_links = set(
                    stage4_model.surface_by_link(q_repair_start, density="coarse")
                )
                if (
                    callable(PERSISTENT_OBSTACLE_WORKER_FACTORY)
                    and args.mode in robot_motion_modes
                    and fresh_recheck.get("accepted", False)
                    and multisphere_geometry is not None
                ):
                    persistent_worker = PERSISTENT_OBSTACLE_WORKER_FACTORY(
                        processor=processor,
                        denoiser=denoiser,
                        args=args,
                        initial_fresh=fresh_recheck,
                        initial_geometry=multisphere_geometry,
                        initial_frames=fresh_frames,
                        output_dir=trial_dir / "persistent_perception",
                    )
                    persistent_worker.start()
                    planning_state = persistent_worker.snapshot()
                    planning_timestamp = time.time()
                    planning_dt = max(
                        0.0,
                        planning_timestamp - float(planning_state["timestamp"]),
                    )
                    planning_center = np.asarray(
                        planning_state["center"], dtype=np.float64
                    ) + np.asarray(
                        planning_state["velocity"], dtype=np.float64
                    ) * planning_dt
                    planning_geometry = translated_multisphere_geometry(
                        planning_state["geometry"],
                        np.asarray(planning_state["center"], dtype=np.float64),
                        planning_center,
                    )
                    obstacle["center"] = planning_center
                    obstacle["window_velocity"] = np.asarray(
                        planning_state["velocity"], dtype=np.float64
                    )
                    obstacle["multisphere_geometry"] = planning_geometry
                    obstacle["planning_state_timestamp"] = float(
                        planning_state["timestamp"]
                    )
                    obstacle["planning_state_center"] = np.asarray(
                        planning_state["center"], dtype=np.float64
                    )
                    obstacle["planning_state_velocity"] = np.asarray(
                        planning_state["velocity"], dtype=np.float64
                    )
                    obstacle["planning_state_age_s"] = planning_dt
                local_artifacts: dict[str, Any] = {}
                try:
                    candidate_summary = run_fast_repair(
                        args,
                        stage4_config,
                        stage4_model,
                        q_now=q_repair_start,
                        qd_now=qd_repair_start,
                        center=obstacle["center"],
                        velocity=obstacle["window_velocity"],
                        radius=obstacle["inflated_radius"],
                        risk_links=risk_links,
                        trial_dir=trial_dir,
                        reference_goal=reference_goal,
                        rejoin_goals=rejoin_goals,
                        obstacle_audit=obstacle,
                        multisphere_geometry=obstacle.get("multisphere_geometry"),
                        artifacts_out=local_artifacts,
                    )
                except Exception:
                    if persistent_worker is not None:
                        persistent_worker.stop()
                    raise
                log["events"].append({"type": candidate_summary["status"], "frame": frame_index, "t_s": now_rel, "candidate": candidate_summary})
                # Defaults let both failure modes enter the same rolling loop:
                # (a) Fast itself found no local step, or (b) Fresh #2 later
                # invalidated an initially safe candidate.
                fresh2 = fresh_recheck
                fresh2_geometry = multisphere_geometry
                local_authorization = {
                    "status": "LOCAL_EXECUTION_RECHECK_FAILED",
                    "local_execution_authorized": False,
                    "reason": "initial_fast_not_ready",
                    "robot_executed": False,
                }
                authorization = {
                    "status": "POST_PLAN_RECHECK_FAILED",
                    "execution_authorized": False,
                    "reason": "initial_fast_not_ready",
                    "robot_executed": False,
                }
                if (
                    candidate_summary.get("local_repair_ready")
                    and args.mode in robot_motion_modes
                    and not callable(LATEST_STATE_AUTHORIZATION_POLICY)
                ):
                    fresh2, fresh2_frames, fresh2_points = capture_post_stop_obstacle(
                        processor,
                        state_reader,
                        denoiser,
                        args,
                        trigger_cluster_center=np.asarray(fresh_recheck["center"], dtype=np.float64),
                        trigger_velocity=np.asarray(fresh_recheck["velocity"], dtype=np.float64),
                        trigger_timestamp=float(fresh_recheck["last_timestamp"]),
                        stop_when_ready=True,
                    )
                    fresh2_geometry = None
                    if fresh2["accepted"] and fresh2_points is not None:
                        fresh2_geometry = fit_pca_multisphere(
                            fresh2_points,
                            fit_margin_m=args.multisphere_fit_margin_m,
                            max_components=args.multisphere_max_components,
                        )
                        if not fresh2_geometry["covered"]:
                            fresh2 = {**fresh2, "accepted": False, "reason": "post_plan_multisphere_coverage_failed"}
                        np.save(trial_dir / "post_plan_fresh_cluster_points.npy", fresh2_points)
                        write_json(trial_dir / "post_plan_fresh_multisphere.json", fresh2_geometry)
                    write_json(trial_dir / "post_plan_fresh_recheck.json", {"result": fresh2, "frames": fresh2_frames})
                    log["events"].append(
                        {
                            "type": "POST_PLAN_FRESH_RECHECK_READY" if fresh2["accepted"] else "POST_PLAN_FRESH_RECHECK_FAILED",
                            "frame": frame_index,
                            "t_s": time.perf_counter() - started,
                            "recheck": fresh2,
                        }
                    )
                    local_authorization = {
                        "status": "LOCAL_EXECUTION_RECHECK_FAILED",
                        "local_execution_authorized": False,
                        "reason": fresh2.get("reason", "fresh2_not_ready"),
                        "robot_executed": False,
                    }
                    authorized_local_trajectory = None
                    if fresh2["accepted"] and fresh2_geometry is not None:
                        local_authorization, authorized_local_trajectory = authorize_local_repair_execution(
                            args,
                            stage4_config,
                            stage4_model,
                            local_repair_ready=bool(candidate_summary.get("local_repair_ready")),
                            local_artifacts=local_artifacts,
                            fresh_geometry=fresh2_geometry,
                            fresh_velocity=np.asarray(fresh2["velocity"], dtype=np.float64),
                            trial_dir=trial_dir,
                        )
                        authorization = authorize_candidate_execution(
                            args,
                            stage4_config,
                            stage4_model,
                            local_repair_ready=bool(candidate_summary.get("local_repair_ready")),
                            local_artifacts=local_artifacts,
                            fresh_geometry=fresh2_geometry,
                            fresh_velocity=np.asarray(fresh2["velocity"], dtype=np.float64),
                            rejoin_goals=rejoin_goals,
                            trial_dir=trial_dir,
                        )
                    else:
                        authorization = {
                            "status": "POST_PLAN_RECHECK_FAILED",
                            "execution_authorized": False,
                            "reason": fresh2["reason"],
                            "robot_executed": False,
                        }
                    candidate_summary["execution_authorization_status"] = authorization["status"]
                    candidate_summary["execution_authorized"] = bool(authorization.get("execution_authorized", False))
                    candidate_summary["local_execution_authorization_status"] = local_authorization["status"]
                    candidate_summary["local_execution_authorized"] = bool(
                        local_authorization.get("local_execution_authorized", False)
                    )
                    candidate_summary["accepted_for_switch"] = candidate_summary["local_execution_authorized"]
                    candidate_summary["post_plan_fresh_recheck"] = fresh2
                    candidate_summary["local_execution_authorization"] = local_authorization
                    candidate_summary["execution_authorization"] = authorization
                    write_json(trial_dir / "candidate" / "candidate_summary.json", candidate_summary)
                    authorization_event = (
                        "EXECUTION_AUTHORIZED_SHADOW"
                        if authorization.get("execution_authorized", False)
                        else authorization["status"]
                    )
                    log["events"].append(
                        {"type": authorization_event, "frame": frame_index, "t_s": time.perf_counter() - started, "authorization": authorization}
                    )
                    log["events"].append(
                        {
                            "type": (
                                "LOCAL_EXECUTION_AUTHORIZED_SHADOW"
                                if local_authorization.get("local_execution_authorized", False)
                                else local_authorization["status"]
                            ),
                            "frame": frame_index,
                            "t_s": time.perf_counter() - started,
                            "authorization": local_authorization,
                        }
                    )
                if (
                    callable(LATEST_STATE_AUTHORIZATION_POLICY)
                    and persistent_worker is not None
                    and args.mode in robot_motion_modes
                ):
                    latest_outcome = LATEST_STATE_AUTHORIZATION_POLICY(
                        worker=persistent_worker,
                        args=args,
                        stage4_config=stage4_config,
                        stage4_model=stage4_model,
                        q_now=q_repair_start,
                        qd_now=qd_repair_start,
                        reference_goal=reference_goal,
                        rejoin_goals=rejoin_goals,
                        risk_links=risk_links,
                        trial_dir=trial_dir,
                        candidate_summary=candidate_summary,
                        local_artifacts=local_artifacts,
                        planning_state=planning_state,
                    )
                    candidate_summary = latest_outcome["candidate_summary"]
                    local_artifacts = latest_outcome["local_artifacts"]
                    fresh2 = latest_outcome.get("fresh") or fresh_recheck
                    fresh2_geometry = (
                        latest_outcome.get("fresh_geometry") or multisphere_geometry
                    )
                    local_authorization = latest_outcome["local_authorization"]
                    authorization = latest_outcome["execution_authorization"]
                    candidate_summary["execution_authorization_status"] = authorization[
                        "status"
                    ]
                    candidate_summary["execution_authorized"] = False
                    candidate_summary["local_execution_authorization_status"] = (
                        local_authorization["status"]
                    )
                    candidate_summary["local_execution_authorized"] = bool(
                        local_authorization.get("local_execution_authorized", False)
                    )
                    candidate_summary["accepted_for_switch"] = bool(
                        candidate_summary["local_execution_authorized"]
                    )
                    candidate_summary["v3_latest_state_authorization"] = {
                        key: latest_outcome[key]
                        for key in (
                            "status",
                            "authorized",
                            "attempts",
                            "fresh",
                            "fresh_geometry",
                            "local_authorization",
                            "execution_authorization",
                        )
                    }
                    write_json(
                        trial_dir / "candidate" / "candidate_summary.json",
                        candidate_summary,
                    )
                    log["events"].append(
                        {
                            "type": latest_outcome["status"],
                            "frame": frame_index,
                            "t_s": time.perf_counter() - started,
                            "attempts": latest_outcome["attempts"],
                        }
                    )
                    log["events"].append(
                        {
                            "type": (
                                "LOCAL_EXECUTION_AUTHORIZED_SHADOW"
                                if local_authorization.get(
                                    "local_execution_authorized", False
                                )
                                else local_authorization["status"]
                            ),
                            "frame": frame_index,
                            "t_s": time.perf_counter() - started,
                            "authorization": local_authorization,
                        }
                    )
                closed_loop_result = None
                if (
                    callable(POST_AUTHORIZATION_CLOSED_LOOP_HANDLER)
                    and persistent_worker is not None
                    and args.mode == "live-stop-replan-execute"
                    and local_authorization.get("local_execution_authorized", False)
                ):
                    if commander is not None:
                        commander.stop()
                        commander = None
                    try:
                        closed_loop_result = POST_AUTHORIZATION_CLOSED_LOOP_HANDLER(
                            worker=persistent_worker,
                            args=args,
                            stage4_config=stage4_config,
                            stage4_model=stage4_model,
                            robot=robot,
                            processor=processor,
                            state_reader=state_reader,
                            denoiser=denoiser,
                            local_artifacts=local_artifacts,
                            trial_dir=trial_dir,
                            task_goal_q=np.asarray(reference.q[-1], dtype=np.float64),
                            risk_links=risk_links,
                        )
                    finally:
                        persistent_worker.stop()
                    persistent_worker = None
                    candidate_summary["final_closed_loop_execution"] = (
                        closed_loop_result
                    )
                    write_json(
                        trial_dir / "candidate" / "candidate_summary.json",
                        candidate_summary,
                    )
                    for event_type in closed_loop_result.get("events", []):
                        log["events"].append(
                            {
                                "type": event_type,
                                "frame": frame_index,
                                "t_s": time.perf_counter() - started,
                            }
                        )
                    log["events"].append(
                        {
                            "type": closed_loop_result["status"],
                            "frame": frame_index,
                            "t_s": time.perf_counter() - started,
                            "closed_loop_execution": closed_loop_result,
                        }
                    )
                    break

                playback_shadow = None
                if (
                    callable(POST_AUTHORIZATION_PLAYBACK_SHADOW)
                    and persistent_worker is not None
                    and args.mode == "live-stop-replan-execute"
                    and local_authorization.get("local_execution_authorized", False)
                ):
                    try:
                        playback_shadow = POST_AUTHORIZATION_PLAYBACK_SHADOW(
                            worker=persistent_worker,
                            args=args,
                            stage4_config=stage4_config,
                            stage4_model=stage4_model,
                            local_artifacts=local_artifacts,
                            trial_dir=trial_dir,
                            task_goal_q=np.asarray(reference.q[-1], dtype=np.float64),
                            risk_links=risk_links,
                        )
                    finally:
                        persistent_worker.stop()
                    persistent_worker = None
                    candidate_summary["v3_playback_shadow"] = playback_shadow
                    candidate_summary["command_time_authorized"] = bool(
                        playback_shadow["status"]
                        in {
                            "V3_VIRTUAL_PLAYBACK_SHADOW_PASS",
                            "V3_VIRTUAL_CLOSED_LOOP_GOAL_REACHED",
                        }
                    )
                    for event_type in playback_shadow.get("events", []):
                        log["events"].append(
                            {
                                "type": event_type,
                                "frame": frame_index,
                                "t_s": time.perf_counter() - started,
                            }
                        )
                    log["events"].append(
                        {
                            "type": playback_shadow["status"],
                            "frame": frame_index,
                            "t_s": time.perf_counter() - started,
                            "playback_shadow": playback_shadow,
                        }
                    )
                    if not candidate_summary["command_time_authorized"]:
                        local_authorization = {
                            **local_authorization,
                            "status": "V3_PRECOMMAND_OR_PLAYBACK_SHADOW_HOLD",
                            "local_execution_authorized": False,
                            "reason": playback_shadow.get(
                                "playback_failure_reasons", []
                            ),
                        }
                        candidate_summary["local_execution_authorized"] = False
                        candidate_summary["accepted_for_switch"] = False
                    write_json(
                        trial_dir / "candidate" / "candidate_summary.json",
                        candidate_summary,
                    )
                if (
                    persistent_worker is not None
                    and not callable(MID_EXECUTION_MONITOR_FACTORY)
                ):
                    persistent_worker.stop()
                    persistent_worker = None
                if args.mode == "moving-shadow-stop":
                    break
                if args.mode == "live-stop-replan-execute":
                    full_path_authorized = bool(candidate_summary.get("execution_authorized", False))
                    local_path_authorized = bool(candidate_summary.get("local_execution_authorized", False))
                    if not full_path_authorized and not local_path_authorized:
                        rolling = rolling_fast_until_authorized(
                            args,
                            stage4_config,
                            stage4_model,
                            processor=processor,
                            state_reader=state_reader,
                            denoiser=denoiser,
                            q_now=q_repair_start,
                            qd_now=qd_repair_start,
                            reference_goal=reference_goal,
                            rejoin_goals=rejoin_goals,
                            initial_fresh=(fresh2 if fresh2.get("accepted", False) else fresh_recheck),
                            initial_geometry=(
                                fresh2_geometry
                                if fresh2.get("accepted", False) and fresh2_geometry is not None
                                else multisphere_geometry
                            ),
                            risk_links=risk_links,
                            trial_dir=trial_dir,
                        )
                        write_json(trial_dir / "rolling_fast" / "rolling_summary.json", rolling)
                        log["events"].append(
                            {"type": rolling["status"], "rolling": rolling}
                        )
                        if not rolling["authorized"]:
                            log["events"].append(
                                {
                                    "type": "LIVE_CANDIDATE_NOT_EXECUTED",
                                    "reason": "rolling_fast_timeout_or_safe_hold",
                                    "candidate_status": candidate_summary["status"],
                                    "rejection_reasons": candidate_summary.get("rejection_reasons", []),
                                }
                            )
                            break
                        candidate_summary = rolling["candidate_summary"]
                        local_artifacts = rolling["local_artifacts"]
                        fresh2 = rolling["fresh"]
                        fresh2_geometry = rolling["fresh_geometry"]
                        local_authorization = rolling["local_authorization"]
                        authorization = rolling["execution_authorization"]
                        candidate_summary["execution_authorization"] = authorization
                        candidate_summary["local_execution_authorization"] = local_authorization
                        candidate_summary["execution_authorized"] = bool(authorization.get("execution_authorized", False))
                        candidate_summary["local_execution_authorized"] = bool(
                            local_authorization.get("local_execution_authorized", False)
                        )
                        full_path_authorized = candidate_summary["execution_authorized"]
                        local_path_authorized = candidate_summary["local_execution_authorized"]
                    if not args.allow_live_candidate_execution:
                        observation_s = float(getattr(args, "shadow_hold_observation_s", 0.0))
                        if observation_s > 0.0 and persistent_worker is not None:
                            started = time.monotonic()
                            initial = persistent_worker.snapshot()
                            initial_seq = int(v3._state_seq(initial))
                            while time.monotonic() - started < observation_s:
                                time.sleep(0.05)
                            final = persistent_worker.snapshot()
                            final_seq = int(v3._state_seq(final))
                            log["events"].append({
                                "type": "SHADOW_HOLD_OBSERVATION_COMPLETE",
                                "observation_s": observation_s,
                                "initial_state_seq": initial_seq,
                                "final_state_seq": final_seq,
                                "state_seq_delta": final_seq - initial_seq,
                                "final_center_m": final.get("center"),
                                "final_raw_guard_distance_m": final.get("raw_guard_distance_m"),
                            })
                        log["events"].append(
                            {
                                "type": "LIVE_CANDIDATE_EXECUTION_BLOCKED_BY_DEFAULT",
                                "reason": "rerun with --allow-live-candidate-execution and exact live candidate phrase after a successful dry pilot",
                                "required_live_candidate_phrase": LIVE_CANDIDATE_EXECUTE_PHRASE,
                            }
                        )
                        break
                    if args.live_execute_candidate_phrase != LIVE_CANDIDATE_EXECUTE_PHRASE:
                        log["events"].append(
                            {
                                "type": "LIVE_CANDIDATE_EXECUTION_BLOCKED_BAD_PHRASE",
                                "required_live_candidate_phrase": LIVE_CANDIDATE_EXECUTE_PHRASE,
                            }
                        )
                        break
                    if commander is not None:
                        commander.stop()
                        commander = None
                    time.sleep(max(args.candidate_pre_execute_settle_s, 0.0))
                    candidate_dir = trial_dir / "candidate"
                    try:
                        full_authorization = candidate_summary["execution_authorization"]
                        local_authorization = candidate_summary["local_execution_authorization"]
                        execution_path = select_dynamic_execution_path(
                            local_authorized=local_path_authorized,
                            full_authorized=full_path_authorized,
                            rolling_local_enabled=args.rolling_local_execution,
                        )
                        if execution_path is None:
                            raise RuntimeError("no Fresh-authorized execution path is available")
                        if execution_path == "ROLLING_LOCAL_FIRST":
                            raise RuntimeError(
                                "rolling-local live execution is shadow-gated until the multi-segment state audit passes"
                            )
                        active_authorization = (
                            full_authorization if execution_path == "FULL_FIRST" else local_authorization
                        )
                        authorized_csv = Path(active_authorization["authorized_trajectory_csv"])
                        workspace_deviation = trajectory_workspace_deviation(
                            stage4_model,
                            authorized_csv,
                            reference,
                            reference_repair_start_time_s,
                        )
                        write_json(candidate_dir / "workspace_deviation_metrics.json", workspace_deviation)
                        mid_execution_monitor = None
                        if callable(MID_EXECUTION_MONITOR_FACTORY):
                            mid_execution_monitor = MID_EXECUTION_MONITOR_FACTORY(
                                authorized_csv=authorized_csv,
                                robot=robot,
                                # Pass the live worker explicitly.  The
                                # factory's monitor closes over this object for
                                # both prearm readiness and in-motion checks;
                                # omitting it silently converted every monitor
                                # call into persistent_tracker_unavailable.
                                worker=persistent_worker,
                                processor=processor,
                                state_reader=state_reader,
                                denoiser=denoiser,
                                args=args,
                                stage4_config=stage4_config,
                                stage4_model=stage4_model,
                                trial_dir=trial_dir,
                                reference=reference,
                                risk_links=risk_links,
                                local_artifacts=local_artifacts,
                                event_local_index=1,
                                rolling_continuation=False,
                            )
                        first_execution_summary = execute_authorized_trajectory_offline_track(
                            robot,
                            authorized_csv,
                            args,
                            processor=processor,
                            denoiser=denoiser,
                            playback_duration_s=None,
                            execution_label=(
                                "authorized repair + rejoin"
                                if execution_path == "FULL_FIRST"
                                else "Fresh #2-authorized local repair"
                            ),
                            motion_monitor_provider=mid_execution_monitor,
                        )
                        take_preplan = getattr(mid_execution_monitor, "take_rolling_preplan", None)
                        if callable(take_preplan):
                            warm = take_preplan(wait_s=0.05)
                            if warm is not None:
                                first_execution_summary["rolling_preplan"] = {
                                    "ready": bool(warm.get("ready", False)),
                                    "source_state_seq": warm.get("source_state_seq"),
                                    "trigger_elapsed_s": warm.get("trigger_elapsed_s"),
                                    "planning_wall_ms": warm.get("planning_wall_ms"),
                                    "candidate": warm.get("candidate"),
                                    "artifacts": warm.get("artifacts"),
                                }
                        first_execution_summary["workspace_deviation_metrics"] = workspace_deviation
                        first_execution_summary["execution_path"] = execution_path
                        first_log_name = (
                            "live_full_candidate_execution_log.json"
                            if execution_path == "FULL_FIRST"
                            else "live_local_candidate_execution_log.json"
                        )
                        # Persist the executor result before classifying it;
                        # precommand holds have no timing_check by design.
                        write_json(candidate_dir / first_log_name, first_execution_summary)
                        execution_status = first_execution_summary.get("status")
                        if execution_status in PRECOMMAND_HOLD_STATUSES:
                            log["events"].append(
                                {
                                    "type": "LOCAL_EXECUTION_PRECOMMAND_HOLD",
                                    "execution_path": execution_path,
                                    "execution": first_execution_summary,
                                    "reason": first_execution_summary.get("motion_monitor_stop_reason"),
                                    "prearm": first_execution_summary.get("motion_monitor_prearm"),
                                }
                            )
                            full_execution_summary = first_execution_summary
                            break
                        precommand_replan = execution_status in PRECOMMAND_REPLAN_STATUSES
                        early_monitor_stop = bool(
                            precommand_replan
                            or (
                                first_execution_summary.get("status") == "STOPPED_BY_MOTION_MONITOR"
                                and first_execution_summary.get("goal_check", {}).get("monitor_stopped", False)
                            )
                        )
                        if (
                            first_execution_summary["status"] != "COMPLETED_AUTHORIZED_TRAJECTORY_EXECUTION"
                            and not early_monitor_stop
                        ):
                            raise RuntimeError(
                                "authorized first segment did not follow its time axis: "
                                f"{first_execution_summary.get('timing_check')}"
                            )
                        full_execution_summary = first_execution_summary
                        first_event_type = (
                            "LOCAL_EXECUTION_PRECOMMAND_REPLAN_REQUIRED"
                            if precommand_replan
                            else (
                                "LIVE_REPAIR_REJOIN_EXECUTED"
                                if execution_path == "FULL_FIRST"
                                else "LIVE_LOCAL_REPAIR_EXECUTED_HOLD"
                            )
                        )
                        log["events"].append(
                            {
                                "type": first_event_type,
                                "execution_path": execution_path,
                                "execution": first_execution_summary,
                            }
                        )

                        # One fail-closed Fresh #3 gate before resuming the
                        # original task.  When the persistent worker is alive,
                        # consume its latest associated state immediately;
                        # waiting for a new three-frame capture here can leave
                        # a continuously moving obstacle 30--40 mm closer
                        # before local #2 is even planned.  The worker state is
                        # still revalidated by the local authorization and the
                        # raw hard guard before any command is sent.
                        fresh3_geometry = None
                        fresh3_points = None
                        fresh3_frames = []
                        if persistent_worker is not None:
                            v3_module = importlib.import_module(
                                "experiments.new.6_5.6_5_3.dynamic_nubs_v3"
                            )
                            tail_snapshot = persistent_worker.snapshot()
                            aligned_tail = v3_module.time_aligned_snapshot(
                                tail_snapshot, execution_timestamp=time.time()
                            )
                            fresh3_geometry = aligned_tail["geometry"]
                            fresh3 = {
                                "accepted": bool(fresh3_geometry.get("covered", False)),
                                "reason": "persistent_worker_latest_tail_state",
                                "track_id": int(fresh2.get("track_id") or 1),
                                "center": np.asarray(
                                    aligned_tail["propagated_center"], dtype=np.float64
                                ).tolist(),
                                "velocity": np.asarray(
                                    tail_snapshot["velocity"], dtype=np.float64
                                ).tolist(),
                                "radius": float(
                                    max(
                                        np.asarray(
                                            fresh3_geometry["component_base_radii"],
                                            dtype=np.float64,
                                        )
                                    )
                                ),
                                "last_timestamp": float(time.time()),
                                "max_association_error_m": float(
                                    tail_snapshot.get("association_error_m", 0.0)
                                ),
                                "source": "persistent_worker_latest_tail_state",
                                "state_seq": int(tail_snapshot.get("state_seq", -1)),
                            }
                        else:
                            fresh3, fresh3_frames, fresh3_points = capture_post_stop_obstacle(
                                processor,
                                state_reader,
                                denoiser,
                                args,
                                trigger_cluster_center=np.asarray(fresh2["center"], dtype=np.float64),
                                trigger_velocity=np.asarray(fresh2["velocity"], dtype=np.float64),
                                trigger_timestamp=float(fresh2["last_timestamp"]),
                                stop_when_ready=True,
                            )
                            if fresh3["accepted"] and fresh3_points is not None:
                                fresh3_geometry = fit_pca_multisphere(
                                    fresh3_points,
                                    fit_margin_m=args.multisphere_fit_margin_m,
                                    max_components=args.multisphere_max_components,
                                )
                                if not fresh3_geometry["covered"]:
                                    fresh3 = {**fresh3, "accepted": False, "reason": "fresh3_multisphere_coverage_failed"}
                                np.save(trial_dir / "fresh3_cluster_points.npy", fresh3_points)
                                write_json(trial_dir / "fresh3_multisphere.json", fresh3_geometry)
                        write_json(trial_dir / "fresh3_recheck.json", {"result": fresh3, "frames": fresh3_frames})

                        fresh3_guard_distance = execution_hard_guard_distance(processor, denoiser, args)
                        if execution_path != "FULL_FIRST" and callable(POST_LOCAL_FRESH3_HANDLER):
                            continuation = POST_LOCAL_FRESH3_HANDLER(
                                args=args,
                                stage4_config=stage4_config,
                                stage4_model=stage4_model,
                                robot=robot,
                                processor=processor,
                                state_reader=state_reader,
                                denoiser=denoiser,
                                persistent_worker=persistent_worker,
                                execution_summary=first_execution_summary,
                                local1_interrupted=early_monitor_stop,
                                local_artifacts=local_artifacts,
                                fresh3=fresh3,
                                fresh3_geometry=fresh3_geometry,
                                fresh3_frames=fresh3_frames,
                                fresh3_guard_distance=fresh3_guard_distance,
                                rolling_preplan=first_execution_summary.get("rolling_preplan"),
                                risk_links=risk_links,
                                reference=reference,
                                trial_dir=trial_dir,
                            )
                            log["events"].append(
                                {
                                    "type": continuation.get(
                                        "status", "POST_LOCAL_EVENT_REPLAN_HANDLER_RETURNED"
                                    ),
                                    "continuation": continuation,
                                }
                            )
                            if continuation.get("handled", False):
                                break
                        if execution_path == "FULL_FIRST":
                            rejoin_match = locate_authorized_rejoin_on_reference(reference, authorized_csv)
                            rejoin_absolute_time = float(rejoin_match["time_s"])
                            selected_rejoin_offset_s = float(full_authorization["selected_rejoin_offset_s"])
                            delayed_rejoin_summary = None
                        else:
                            delayed_rejoin_summary, _ = authorize_delayed_rejoin_after_fresh3(
                                args,
                                stage4_config,
                                stage4_model,
                                local_artifacts=local_artifacts,
                                fresh3=fresh3,
                                fresh3_geometry=fresh3_geometry,
                                fresh3_frames=fresh3_frames,
                                rejoin_goals=rejoin_goals,
                                hard_guard_distance_m=fresh3_guard_distance,
                                trial_dir=trial_dir,
                            )
                            log["events"].append(
                                {"type": delayed_rejoin_summary["status"], "authorization": delayed_rejoin_summary}
                            )
                            if not delayed_rejoin_summary["authorized"]:
                                log["events"].append(
                                    {"type": "REMAIN_HOLD_AT_LOCAL_REPAIR_TAIL", "authorization": delayed_rejoin_summary}
                                )
                                break
                            bridge_csv = Path(delayed_rejoin_summary["authorized_trajectory_csv"])
                            bridge_execution = execute_authorized_trajectory_offline_track(
                                robot,
                                bridge_csv,
                                args,
                                processor=processor,
                                denoiser=denoiser,
                                playback_duration_s=None,
                                execution_label="Fresh #3-authorized delayed C2 rejoin bridge",
                            )
                            if bridge_execution["status"] != "COMPLETED_AUTHORIZED_TRAJECTORY_EXECUTION":
                                raise RuntimeError(
                                    "delayed rejoin bridge did not follow its authorized time axis: "
                                    f"{bridge_execution.get('timing_check')}"
                                )
                            delayed_rejoin_summary["robot_executed"] = True
                            write_json(
                                trial_dir / "delayed_rejoin_authorization" / "bridge_execution_log.json",
                                bridge_execution,
                            )
                            log["events"].append({"type": "LIVE_DELAYED_REJOIN_EXECUTED", "execution": bridge_execution})
                            selected_rejoin_offset_s = float(delayed_rejoin_summary["selected_rejoin_offset_s"])
                            rejoin_absolute_time = min(
                                float(reference.times[-1]),
                                reference_repair_start_time_s + selected_rejoin_offset_s,
                            )
                            rejoin_match = {
                                "time_s": rejoin_absolute_time,
                                "source": "delayed_rejoin_selected_reference_state",
                            }
                        remainder_times, remainder_q, _ = reference.remainder_after(rejoin_absolute_time)
                        remainder_csv = trial_dir / "authorized_reference_remainder.csv"
                        save_joint_waypoint_csv(remainder_csv, remainder_times, remainder_q)
                        if execution_path == "FULL_FIRST":
                            reference_resume_summary = authorize_reference_resume_after_fresh3(
                                args,
                                stage4_config,
                                stage4_model,
                                fresh3=fresh3,
                                fresh3_geometry=fresh3_geometry,
                                fresh3_frames=fresh3_frames,
                                remainder_times=remainder_times,
                                remainder_q=remainder_q,
                                hard_guard_distance_m=fresh3_guard_distance,
                            )
                        else:
                            # The delayed bridge and its endpoint were already fully
                            # checked against this Fresh #3 forecast.  The following
                            # Cartesian remainder retains its independent raw guard.
                            reference_resume_summary = {
                                "status": "REFERENCE_RESUME_AUTHORIZED",
                                "authorized": True,
                                "resume_basis": "FRESH3_AUTHORIZED_DELAYED_REJOIN",
                                "delayed_rejoin_authorization": delayed_rejoin_summary,
                                "hard_guard_distance_m": fresh3_guard_distance,
                            }
                        reference_resume_summary.update(
                            {
                                "execution_path": execution_path,
                                "selected_rejoin_offset_s": selected_rejoin_offset_s,
                                "rejoin_reference_match": rejoin_match,
                                "rejoin_absolute_reference_time_s": rejoin_absolute_time,
                                "authorized_reference_remainder_csv": str(remainder_csv),
                                "remainder_duration_s": float(remainder_times[-1]),
                            }
                        )
                        write_json(trial_dir / "reference_resume_authorization.json", reference_resume_summary)
                        log["events"].append({"type": reference_resume_summary["status"], "authorization": reference_resume_summary})
                        if reference_resume_summary["authorized"]:
                            reference_remainder_execution_summary = execute_guarded_cartesian_reference_remainder(
                                robot,
                                args,
                                processor=processor,
                                denoiser=denoiser,
                                target_y_m=args.y_goal,
                            )
                            write_json(trial_dir / "reference_remainder_execution_log.json", reference_remainder_execution_summary)
                            log["events"].append(
                                {"type": "REFERENCE_REMAINDER_EXECUTED", "execution": reference_remainder_execution_summary}
                            )
                        else:
                            log["events"].append({"type": "REMAIN_HOLD_AFTER_REJOIN", "authorization": reference_resume_summary})
                    except Exception as exc:
                        stop_ret = maybe_move_stop(robot)
                        full_execution_summary = {
                            "status": "FAILED_DYNAMIC_EXECUTION",
                            "execution_path": execution_path,
                            "error": str(exc),
                            "traceback": traceback.format_exc(limit=20),
                            "stop_return": stop_ret,
                        }
                        write_json(candidate_dir / "live_dynamic_execution_failure.json", full_execution_summary)
                        log["events"].append({"type": "LIVE_DYNAMIC_EXECUTION_FAILED", "execution": full_execution_summary})
                    break
            if (
                args.mode in robot_motion_modes
                and np.isfinite(float(current_best["distance"]))
                and float(current_best["distance"]) <= args.stop_distance_m
            ):
                stop_ret = maybe_move_stop(robot)
                log["events"].append({"type": "SAFETY_HOLD_DISTANCE", "frame": frame_index, "t_s": now_rel, "distance_m": float(current_best["distance"]), "return": stop_ret})
                break
            frame_index += 1
    except Exception as exc:
        caught_error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=20),
        }
        log["events"].append({"type": "ERROR", "error": caught_error})
    finally:
        if persistent_worker is not None:
            try:
                persistent_worker.stop()
            except Exception:
                pass
        if commander is not None:
            try:
                commander.stop()
            except Exception:
                pass
        if audit_visualizer is not None:
            try:
                audit_visualizer.close()
            except Exception:
                pass
        processor.stop()

    write_csv(trial_dir / "frames.csv", rows, FRAME_FIELDS)
    write_csv(trial_dir / "clusters.csv", cluster_rows, CLUSTER_FIELDS)
    write_csv(trial_dir / "tracks.csv", track_rows, TRACK_FIELDS)
    pred_vals = [float(r["predicted_distance_m"]) for r in rows if r["predicted_distance_m"] != ""]
    cur_vals = [float(r["nearest_distance_m"]) for r in rows if r["nearest_distance_m"] != ""]
    guard_vals = [float(r["guard_distance_m"]) for r in rows if r.get("guard_distance_m", "") != ""]
    final_status = "TRIGGERED" if triggered else "NO_TRIGGER"
    if guard_stopped and not triggered:
        final_status = "GUIDED_GUARD_STOPPED_NO_CCRO_TRIGGER"
    elif guard_stopped and triggered:
        final_status = "TRIGGERED_AND_GUIDED_GUARD_STOPPED"
    elif current_distance_stopped and not triggered:
        final_status = "CURRENT_DISTANCE_STOPPED_NO_CCRO_TRIGGER"
    elif current_distance_stopped and triggered:
        final_status = "TRIGGERED_AND_CURRENT_DISTANCE_STOPPED"
    if candidate_execution_summary is not None:
        if candidate_execution_summary.get("status") == "COMPLETED_DYNAMIC_CANDIDATE_EXECUTION":
            final_status = "TRIGGERED_AND_DYNAMIC_CANDIDATE_EXECUTED"
        else:
            final_status = "TRIGGERED_AND_DYNAMIC_CANDIDATE_EXECUTION_FAILED"
    if full_execution_summary is not None:
        if full_execution_summary.get("status") == "COMPLETED_AUTHORIZED_TRAJECTORY_EXECUTION":
            final_status = (
                "TRIGGERED_AND_REPAIR_REJOIN_EXECUTED_HOLD"
                if execution_path == "FULL_FIRST"
                else "TRIGGERED_AND_LOCAL_REPAIR_EXECUTED_HOLD"
            )
        elif (
            full_execution_summary.get("status") == "STOPPED_BY_MOTION_MONITOR"
            and full_execution_summary.get("goal_check", {}).get("monitor_stopped", False)
            and full_execution_summary.get("goal_check", {}).get("replan_requested", False)
        ):
            final_status = "LOCAL_EXECUTION_INTERRUPTED_FOR_REPLAN"
        else:
            final_status = "TRIGGERED_AND_REPAIR_REJOIN_EXECUTION_FAILED"
    if reference_remainder_execution_summary is not None:
        if reference_remainder_execution_summary.get("status") == "COMPLETED_GUARDED_CARTESIAN_REFERENCE_REMAINDER":
            final_status = "TRIGGERED_REPAIR_REJOIN_RESUMED_AND_GOAL_REACHED"
        else:
            final_status = "TRIGGERED_REPAIR_REJOIN_REFERENCE_RESUME_FAILED"
    if caught_error is not None:
        final_status = "FAILED"
    alignment_rows = [row for row in rows if int(row.get("reference_armed", 0)) == 1]
    reference_match_values = [
        float(row["reference_joint_match_max_rad"])
        for row in alignment_rows
        if row.get("reference_joint_match_max_rad", "") != ""
    ]
    alignment_checks = {
        "reference_armed": any(event.get("type") == "REFERENCE_ARMED" for event in log["events"])
        or args.mode not in robot_motion_modes,
        "armed_rows_present": len(alignment_rows) > 0,
        "index_steps_bounded": bool(alignment_rows)
        and max(int(row["reference_index_step"]) for row in alignment_rows) <= args.max_reference_step,
        "no_clamped_step": bool(alignment_rows)
        and not any(int(row.get("reference_step_clamped", 0)) for row in alignment_rows),
        "reference_match_ok": bool(reference_match_values)
        and max(reference_match_values) <= args.reference_match_max_rad if reference is not None else True,
        "reference_progress_complete": (
            bool(alignment_rows)
            and int(alignment_rows[-1]["reference_index"]) >= int(0.95 * (len(reference.times) - 1))
        ) if reference is not None else True,
        "no_dynamic_trigger": not triggered,
    }
    if args.reference_audit_only and caught_error is None:
        final_status = "REFERENCE_ALIGNMENT_PASS" if all(alignment_checks.values()) else "REFERENCE_ALIGNMENT_FAIL"
    valid_runs: dict[int, int] = {}
    max_valid_runs: dict[int, int] = {}
    for row in track_rows:
        track_id = int(row["track_id"])
        valid_runs[track_id] = valid_runs.get(track_id, 0) + 1 if int(row["dynamic_valid"]) else 0
        max_valid_runs[track_id] = max(max_valid_runs.get(track_id, 0), valid_runs[track_id])
    track_audit_stats: dict[int, dict[str, Any]] = {}
    for track_id in {int(row["track_id"]) for row in track_rows}:
        track = [row for row in track_rows if int(row["track_id"]) == track_id]
        first = np.array([float(track[0][name]) for name in ("center_x", "center_y", "center_z")])
        last = np.array([float(track[-1][name]) for name in ("center_x", "center_y", "center_z")])
        track_audit_stats[track_id] = {
            "rows": len(track),
            "net_displacement_m": float(np.linalg.norm(last - first)),
            "max_window_speed_m_s": max(float(row.get("window_speed_m_s") or 0.0) for row in track),
            "max_valid_run": max_valid_runs.get(track_id, 0),
        }
    qualifying_tracks = [
        track_id for track_id, stats in track_audit_stats.items()
        if stats["rows"] >= args.audit_min_track_frames
        and stats["max_window_speed_m_s"] >= args.min_dynamic_trigger_speed_m_s
        and stats["max_valid_run"] >= args.audit_required_valid_frames
    ]
    dynamic_track_audit = {
        "accepted": bool(qualifying_tracks),
        "required_consecutive_valid_frames": args.audit_required_valid_frames,
        "qualifying_track_ids": qualifying_tracks,
        "max_valid_run_by_track": max_valid_runs,
        "track_stats": track_audit_stats,
        "cluster_rows": len(cluster_rows),
        "track_rows": len(track_rows),
        "dynamic_valid_rows": sum(int(row["dynamic_valid"]) for row in track_rows),
        "robot_motion_commanded": bool(log["robot_commanded"]),
    }
    if args.mode == "dynamic-track-audit" and caught_error is None:
        dynamic_track_audit["accepted"] = bool(dynamic_track_audit["accepted"] and not log["robot_commanded"])
        final_status = "DYNAMIC_TRACK_AUDIT_PASS" if dynamic_track_audit["accepted"] else "DYNAMIC_TRACK_AUDIT_FAIL"
    log.update(
        {
            "status": final_status,
            "error": caught_error,
            "trigger_frame": trigger_frame,
            "candidate_status": None if candidate_summary is None else candidate_summary["status"],
            "candidate_accepted": None if candidate_summary is None else candidate_summary["accepted_for_switch"],
            "local_repair_status": None if candidate_summary is None else candidate_summary.get("local_repair_status"),
            "local_repair_ready": None if candidate_summary is None else candidate_summary.get("local_repair_ready"),
            "execution_authorization_status": None if candidate_summary is None else candidate_summary.get("execution_authorization_status"),
            "execution_authorized": None if candidate_summary is None else candidate_summary.get("execution_authorized", False),
            "local_execution_authorization_status": None if candidate_summary is None else candidate_summary.get("local_execution_authorization_status"),
            "local_execution_authorized": None if candidate_summary is None else candidate_summary.get("local_execution_authorized", False),
            "anomalous_cluster_files": anomalous_cluster_files,
            "candidate_execution_status": None if candidate_execution_summary is None else candidate_execution_summary.get("status"),
            "candidate_execution": candidate_execution_summary,
            "full_candidate_execution_status": None if full_execution_summary is None else full_execution_summary.get("status"),
            "full_candidate_execution": full_execution_summary,
            "execution_path": execution_path,
            "reference_resume_authorization": reference_resume_summary,
            "reference_remainder_execution_status": None if reference_remainder_execution_summary is None else reference_remainder_execution_summary.get("status"),
            "reference_remainder_execution": reference_remainder_execution_summary,
            "frame_count": len(rows),
            "duration_wall_s": time.time() - t0_wall,
            "current_min_distance_m": None if not cur_vals else float(np.min(cur_vals)),
            "predicted_min_distance_m": None if not pred_vals else float(np.min(pred_vals)),
            "guard_min_distance_m": None if not guard_vals else float(np.min(guard_vals)),
            "guard_stopped": guard_stopped,
            "current_distance_stopped": current_distance_stopped,
            "reference_alignment": {
                "accepted": bool(all(alignment_checks.values())),
                "checks": alignment_checks,
                "armed_rows": len(alignment_rows),
                "final_reference_index": None if not alignment_rows else int(alignment_rows[-1]["reference_index"]),
                "max_reference_index_step": None if not alignment_rows else max(int(row["reference_index_step"]) for row in alignment_rows),
                "max_reference_joint_match_rad": None if not reference_match_values else max(reference_match_values),
            },
            "dynamic_track_audit": dynamic_track_audit,
        }
    )
    write_json(trial_dir / "summary.json", log)
    print(json.dumps({"status": log["status"], "candidate_status": log.get("candidate_status"), "trial_dir": str(trial_dir)}, indent=2, ensure_ascii=False))
    return log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=["shadow", "dynamic-track-audit", "moving-shadow-stop", "live-stop-replan-execute", "live-execute"],
        default="shadow",
    )
    parser.add_argument("--operator-phrase", default="")
    parser.add_argument(
        "--task-geometry-id",
        default="UNSPECIFIED",
        help="audit-only label for the frozen physical task geometry; has no control authority",
    )
    parser.add_argument("--robot-ip", default="192.168.123.96")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    parser.add_argument("--stage4-config", type=Path, default=ROOT / "config" / "ccro_stage4.yaml")
    parser.add_argument("--urdf", type=Path, default=ROOT / "urdf" / "aubo_i16_gripper.urdf")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--duration-s", type=float, default=18.0)
    parser.add_argument(
        "--visualize-audit",
        action="store_true",
        help="audit/shadow only: show valid clusters, OBBs, 90%% radius spheres, robot and plane points",
    )
    parser.add_argument("--show-filtered", action="store_true", help="with --visualize-audit, show rejected-cluster points")
    parser.add_argument("--show-noise", action="store_true", help="with --visualize-audit, show DBSCAN noise points")
    parser.add_argument("--save-anomalous-clusters", action="store_true", default=True)
    parser.add_argument("--no-save-anomalous-clusters", dest="save_anomalous_clusters", action="store_false")
    parser.add_argument("--anomaly-bbox-m", type=float, default=0.20)
    parser.add_argument("--anomaly-radius-m", type=float, default=0.12)
    parser.add_argument("--arm-delay-s", type=float, default=1.0)
    parser.add_argument("--reference-audit-only", action="store_true")
    parser.add_argument("--reference-preparation-timeout-s", type=float, default=60.0)
    parser.add_argument("--reference-start-y-tolerance-m", type=float, default=0.015)
    parser.add_argument("--reference-start-joint-tolerance-rad", type=float, default=0.025)
    parser.add_argument("--reference-start-consecutive-frames", type=int, default=3)
    parser.add_argument("--max-reference-step", type=int, default=5)
    parser.add_argument("--reference-step-slack", type=float, default=2.0)
    parser.add_argument("--reference-joint-refine-window", type=int, default=8)
    parser.add_argument("--reference-match-max-rad", type=float, default=0.05)
    parser.add_argument("--local-reference-sanity-scale", type=float, default=1.5)
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--remove-planes", action="store_true", default=True)
    parser.add_argument("--no-remove-planes", dest="remove_planes", action="store_false")
    parser.add_argument("--plane-dist", type=float, default=0.02)
    parser.add_argument("--max-planes", type=int, default=1)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--self-filter-threshold", type=float, default=0.08)
    parser.add_argument("--cluster-eps", type=float, default=0.05)
    parser.add_argument("--cluster-min-samples", type=int, default=15)
    parser.add_argument(
        "--cluster-min-points",
        type=int,
        default=15,
        help="match safety_guided_motion.py default so small hand-held foam obstacles are not filtered out",
    )
    parser.add_argument(
        "--cluster-min-volume",
        type=float,
        default=0.0005,
        help="match safety_guided_motion.py default; larger values can suppress partially visible dynamic obstacles",
    )
    parser.add_argument("--surface-density", choices=["coarse", "medium", "dense"], default="coarse")
    parser.add_argument("--temporal-denoise", action="store_true", default=True)
    parser.add_argument("--no-temporal-denoise", dest="temporal_denoise", action="store_false")
    parser.add_argument("--denoise-voxel", type=float, default=0.04)
    parser.add_argument("--denoise-conf", type=int, default=2)
    parser.add_argument("--denoise-decay", type=float, default=0.4)
    parser.add_argument("--min-track-age", type=int, default=3)
    parser.add_argument("--prediction-horizon-s", type=float, default=0.5)
    parser.add_argument(
        "--stro-trigger-horizon-s",
        type=float,
        default=1.2,
        help=(
            "initial STRO early-warning lookahead only; "
            "Fast/Fresh/rolling prediction remains 0.5 s"
        ),
    )
    parser.add_argument("--prediction-step-s", type=float, default=0.1)
    parser.add_argument("--prediction-margin-m", type=float, default=0.035)
    parser.add_argument("--prediction-uncertainty-m", type=float, default=0.02)
    parser.add_argument("--replan-in-m", type=float, default=0.14)
    parser.add_argument(
        "--moving-shadow-replan-in-m",
        type=float,
        default=0.14,
        help="frozen predicted-distance trigger for moving-shadow-stop",
    )
    parser.add_argument("--stop-distance-m", type=float, default=0.08)
    parser.add_argument(
        "--moving-shadow-current-stop-m",
        type=float,
        default=0.12,
        help="all-link current-distance fallback stop; predicted STRO trigger remains at 0.14 m",
    )
    parser.add_argument("--online-accept-m", type=float, default=0.09)
    parser.add_argument("--min-clearance-improvement-m", type=float, default=0.003)
    parser.add_argument("--min-candidate-delta-q-rad", type=float, default=1.0e-4)
    parser.add_argument("--fast-budget-ms", type=float, default=150.0)
    parser.add_argument(
        "--fast-target-ms",
        type=float,
        default=None,
        help="preferred Fast realtime target; defaults to --fast-budget-ms",
    )
    parser.add_argument(
        "--fast-max-ms",
        type=float,
        default=None,
        help="absolute Fast computation ceiling; defaults to --fast-budget-ms",
    )
    parser.add_argument("--local-horizon-s", type=float, default=1.0)
    parser.add_argument("--rejoin-search-step-s", type=float, default=0.25)
    parser.add_argument("--rejoin-max-offset-s", type=float, default=2.0)
    parser.add_argument(
        "--reference-feedback-csv",
        type=Path,
        default=None,
        help="recorded one-way reference_feedback.csv; mandatory for moving modes",
    )
    parser.add_argument("--local-segments", type=int, default=5)
    parser.add_argument("--fast-warm-start", choices=["linear", "lateral"], default="linear")
    parser.add_argument("--lateral-warm-start-m", type=float, default=0.04)
    parser.add_argument("--rolling-fast-max-s", type=float, default=3.0)
    parser.add_argument("--rolling-observation-duration-s", type=float, default=0.25)
    parser.add_argument("--rolling-observation-min-frames", type=int, default=2)
    parser.add_argument("--rolling-observation-min-span-s", type=float, default=0.10)
    parser.add_argument(
        "--rolling-local-execution",
        action="store_true",
        help="request multi-segment LOCAL_ONLY execution; currently fail-closed behind a shadow audit gate",
    )
    parser.add_argument("--rolling-local-max-segments", type=int, default=3)
    parser.add_argument("--rolling-local-max-total-s", type=float, default=6.0)
    parser.add_argument("--rolling-side-opposite-tolerance-rad", type=float, default=0.002)
    parser.add_argument("--default-obstacle-radius-m", type=float, default=0.055)
    parser.add_argument("--max-track-cluster-association-m", type=float, default=0.08)
    parser.add_argument("--min-dynamic-trigger-speed-m-s", type=float, default=0.08)
    parser.add_argument("--dynamic-exit-speed-m-s", type=float, default=0.04)
    parser.add_argument("--dynamic-exit-streak-frames", type=int, default=3)
    parser.add_argument("--dynamic-speed-window", type=int, default=5)
    parser.add_argument("--dynamic-valid-streak-frames", type=int, default=2)
    parser.add_argument("--dynamic-radius-min-m", type=float, default=0.03, help="legacy radius band for audit logging only; never gates tracking")
    parser.add_argument("--dynamic-radius-max-m", type=float, default=0.10, help="legacy radius band for audit logging only; never gates tracking")
    parser.add_argument("--dynamic-tracker-association-distance-m", type=float, default=0.12)
    parser.add_argument("--dynamic-tracker-motion-gate-speed-m-s", type=float, default=0.03)
    parser.add_argument("--dynamic-tracker-max-miss", type=int, default=2)
    parser.add_argument("--audit-required-valid-frames", type=int, default=2)
    parser.add_argument("--audit-min-track-frames", type=int, default=5)
    parser.add_argument("--min-local-motion-rad", type=float, default=0.002)
    parser.add_argument("--shadow-joint-probe-rad", type=float, default=0.025)
    parser.add_argument("--home-joints-deg", default="0,0,90,0,90,0")
    parser.add_argument("--x-offset", type=float, default=0.0)
    parser.add_argument("--y-start", type=float, default=0.4)
    parser.add_argument("--y-goal", type=float, default=-0.4)
    parser.add_argument("--line-velocity-m-s", type=float, default=0.020)
    parser.add_argument("--line-acc-m-s2", type=float, default=0.05)
    parser.add_argument("--guided-range-m", type=float, default=0.20)
    parser.add_argument("--guided-base-omega", type=float, default=0.15)
    parser.add_argument("--guided-d-safe-m", type=float, default=0.12)
    parser.add_argument("--guided-d-slow-m", type=float, default=0.12)
    parser.add_argument("--guided-d-stop-m", type=float, default=0.08)
    parser.add_argument("--guided-hard-stop-m", type=float, default=0.10)
    parser.add_argument("--guided-max-decel", type=float, default=2.0)
    parser.add_argument("--guided-max-accel", type=float, default=0.5)
    parser.add_argument("--guided-dynamic-lookahead-s", type=float, default=0.15)
    parser.add_argument("--guard-min-x", type=float, default=0.25)
    parser.add_argument("--guard-max-x", type=float, default=0.95)
    parser.add_argument("--guard-min-y", type=float, default=-0.65)
    parser.add_argument("--guard-max-y", type=float, default=0.65)
    parser.add_argument("--guard-min-z", type=float, default=0.30)
    parser.add_argument("--guard-max-z", type=float, default=1.10)
    # Two-layer perception ROI (V3 frozen bounds).  The planning ROI crops the
    # cloud before clustering so the tracker / STRO / PCA / Fast never absorb
    # far walls, table edges or background; the broad safety ROI keeps the
    # 0.10 m raw hard guard wider than the planning box.  Z is tabletop-relative
    # when the table plane is detected, with a fixed-band fallback.  The
    # retained fraction is recorded per frame as an audit, never gated.
    parser.add_argument("--planning-roi-x-min", type=float, default=0.10)
    parser.add_argument("--planning-roi-x-max", type=float, default=0.85)
    parser.add_argument("--planning-roi-y-min", type=float, default=-0.50)
    parser.add_argument("--planning-roi-y-max", type=float, default=0.50)
    parser.add_argument("--planning-roi-z-table-offset-lo", type=float, default=0.05)
    parser.add_argument("--planning-roi-z-table-offset-hi", type=float, default=0.80)
    parser.add_argument("--planning-roi-z-fallback-lo", type=float, default=0.40)
    parser.add_argument("--planning-roi-z-fallback-hi", type=float, default=0.90)
    parser.add_argument("--safety-roi-x-min", type=float, default=0.00)
    parser.add_argument("--safety-roi-x-max", type=float, default=0.85)
    parser.add_argument("--safety-roi-y-min", type=float, default=-0.65)
    parser.add_argument("--safety-roi-y-max", type=float, default=0.65)
    parser.add_argument("--safety-roi-z-table-offset-lo", type=float, default=0.00)
    parser.add_argument("--safety-roi-z-table-offset-hi", type=float, default=0.90)
    parser.add_argument("--safety-roi-z-fallback-lo", type=float, default=0.30)
    parser.add_argument("--safety-roi-z-fallback-hi", type=float, default=1.10)
    parser.add_argument("--roi-table-min-plane-points", type=int, default=150)
    parser.add_argument("--roi-table-distance-threshold", type=float, default=0.02)
    parser.add_argument("--settle-s", type=float, default=0.4)
    parser.add_argument("--poll-s", type=float, default=0.04)
    parser.add_argument("--motion-timeout-s", type=float, default=90.0)
    parser.add_argument("--pose-tolerance-m", type=float, default=0.015)
    parser.add_argument("--joint-tolerance-rad", type=float, default=0.02)
    parser.add_argument("--allow-movel-fallback", action="store_true")
    parser.add_argument("--min-x", type=float, default=-0.2)
    parser.add_argument("--max-x", type=float, default=0.9)
    parser.add_argument("--min-y", type=float, default=-0.55)
    parser.add_argument("--max-y", type=float, default=0.55)
    parser.add_argument("--min-z", type=float, default=0.25)
    parser.add_argument("--max-z", type=float, default=0.9)
    parser.add_argument(
        "--gripper-base-min-z-m",
        type=float,
        default=0.46,
        help="independent tabletop workspace guard for gripper_base_link",
    )
    parser.add_argument("--allow-live-candidate-execution", action="store_true")
    parser.add_argument("--live-execute-candidate-phrase", default="")
    parser.add_argument(
        "--candidate-playback-duration-s",
        type=float,
        default=0.0,
        help="0 uses the authorized candidate's native time axis; formal live pilots still pass the frozen duration explicitly",
    )
    parser.add_argument(
        "--allow-experimental-playback-duration",
        action="store_true",
        help=(
            "explicitly authorize bounded 0.80-1.00 s time-scaled playback "
            "for non-default experiments; all safety verification remains required"
        ),
    )
    parser.add_argument("--candidate-controller-waypoint-period-s", type=float, default=0.005)
    parser.add_argument("--candidate-max-waypoints", type=int, default=0)
    parser.add_argument("--candidate-joint-velc", type=float, default=0.006)
    parser.add_argument("--candidate-joint-acc", type=float, default=0.012)
    parser.add_argument("--candidate-start-tolerance-rad", type=float, default=0.035)
    parser.add_argument(
        "--candidate-start-sync-rad",
        type=float,
        default=0.002,
        help="re-anchor threshold: measured robot start vs candidate first point; a mismatch above this replans once instead of failing the experiment",
    )
    parser.add_argument(
        "--candidate-static-tolerance-rad",
        type=float,
        default=5.0e-4,
        help="per-poll joint step that counts as static while waiting after a STRO stop",
    )
    parser.add_argument("--candidate-goal-tolerance-rad", type=float, default=0.012)
    parser.add_argument("--candidate-min-observed-motion-rad", type=float, default=0.003)
    parser.add_argument("--candidate-min-execution-wait-s", type=float, default=0.0)
    parser.add_argument("--candidate-motion-timeout-s", type=float, default=45.0)
    parser.add_argument("--candidate-pre-execute-settle-s", type=float, default=0.35)
    parser.add_argument("--post-stop-settle-s", type=float, default=0.25)
    parser.add_argument("--post-stop-recheck-duration-s", type=float, default=0.6)
    parser.add_argument("--post-stop-recheck-min-frames", type=int, default=3)
    parser.add_argument("--post-stop-recheck-min-span-s", type=float, default=0.25)
    parser.add_argument("--multisphere-fit-margin-m", type=float, default=0.005)
    parser.add_argument("--multisphere-max-components", type=int, default=4)
    parser.add_argument("--candidate-execute-confirm", action="store_true", default=True)
    parser.add_argument("--no-candidate-execute-confirm", dest="candidate_execute_confirm", action="store_false")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
