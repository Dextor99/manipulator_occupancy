#!/usr/bin/env python3
"""Run a 6.5.3 dynamic-obstacle Fast CCRO-NUBS trial.

This first real-system implementation is intentionally staged:

* ``--mode shadow`` opens RealSense + AUBO feedback, detects/tracks dynamic
  obstacles, triggers STRO/CCRO risk, generates a 1 s Fast CCRO-NUBS candidate,
  validates it, and saves logs/figures.  It never commands the robot.
* ``--mode moving-shadow-stop`` additionally commands the familiar 6.5.2
  low-speed reference line and stops on trigger/hold.  It still does not switch
  to the candidate trajectory; it is the required pilot before live switching.
* ``--mode live-stop-replan-execute`` is currently fail-closed after candidate
  acceptance. Live execution remains disabled until a fresh post-planning
  RGB-D recheck is implemented and validated by three D1 shadow-stop pilots.
"""

from __future__ import annotations

import argparse
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
load_stage4_config = common64.load_stage4_config
load_stage4_surface_model = common64.load_surface_model
make_risk_stack = common64.make_risk_stack
run_repair_v3 = repair_v3_mod.run_repair_v3
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
from perception.geometry_fit import make_occupancy_object  # noqa: E402
from perception.occupancy_tracker import OccupancyTracker  # noqa: E402
from planning.nubs_trajectory import NUBSTrajectory6D  # noqa: E402
from planning.robot_surface_model import RobotSurfaceModel  # noqa: E402
from risk.prediction import RiskSphere, predict_risk_spheres  # noqa: E402
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

SCENARIOS = {
    "D1": {
        "name": "crossing_body",
        "description": "hand-held foam obstacle crosses upper-arm/forearm region",
        "prompt": "准备让泡沫障碍横向经过 upper-arm / forearm 区域",
        "risk_links": {"upperArm_Link", "foreArm_Link", "wrist1_Link", "wrist2_Link"},
    },
    "D2": {
        "name": "approaching_wrist",
        "description": "hand-held foam obstacle obliquely approaches wrist/gripper region",
        "prompt": "准备让泡沫障碍从侧前方斜向接近 wrist / gripper 区域",
        "risk_links": {"foreArm_Link", "wrist1_Link", "wrist2_Link", "wrist3_Link", "gripper_base_link", "left_link", "right_link"},
    },
}

FRAME_FIELDS = [
    "frame",
    "t_s",
    "timestamp",
    "scene_points",
    "robot_points",
    "cluster_count",
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
    "predicted_object_radius_m",
    "predicted_object_age",
    "predicted_object_association_error_m",
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
        return tuple(
            np.asarray([np.interp(target, self.times, values[:, j]) for j in range(6)], dtype=np.float64)
            for values in (self.q, self.qd, self.qdd)
        )


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
    raw_radius = float(getattr(obj, "radius", fallback) or fallback)
    distances = [float(np.linalg.norm(np.asarray(c.center, dtype=np.float64) - center)) for c in clusters]
    cluster_index = None if not distances else int(np.argmin(distances))
    cluster_center = None if cluster_index is None else np.asarray(clusters[cluster_index].center, dtype=np.float64)
    association_error = math.inf if cluster_center is None else float(np.linalg.norm(center - cluster_center))
    return {
        "track_id": object_track_id(obj),
        "center": center,
        "velocity": velocity,
        "speed": float(np.linalg.norm(velocity)),
        "raw_radius": raw_radius,
        "inflated_radius": max(raw_radius, float(fallback)),
        "associated_cluster_index": cluster_index,
        "associated_cluster_center": cluster_center,
        "association_error_m": association_error,
    }


def update_dynamic_track_validity(
    stable: list[Any],
    clusters: list[Any],
    speed_history: dict[int, list[float]],
    valid_streak: dict[int, int],
    args: argparse.Namespace,
) -> tuple[list[Any], dict[int, dict[str, Any]]]:
    """Separate dynamic-trigger tracks from the broader safety-object pool."""
    valid: list[Any] = []
    audits: dict[int, dict[str, Any]] = {}
    active_ids: set[int] = set()
    for obj in stable:
        geometry = track_geometry(obj, clusters, args.default_obstacle_radius_m)
        track_id = geometry["track_id"]
        if track_id is None:
            continue
        active_ids.add(track_id)
        history = speed_history.setdefault(track_id, [])
        history.append(float(geometry["speed"]))
        del history[:-args.dynamic_speed_window]
        median_speed = float(np.median(history))
        checks = {
            "age_ok": int(getattr(obj, "age", 0)) >= args.min_track_age,
            "speed_history_ready": len(history) >= args.dynamic_speed_window,
            "speed_ok": median_speed >= args.min_dynamic_trigger_speed_m_s,
            "radius_ok": args.dynamic_radius_min_m <= geometry["raw_radius"] <= args.dynamic_radius_max_m,
            "association_ok": geometry["association_error_m"] <= args.max_track_cluster_association_m,
        }
        instant_valid = bool(all(checks.values()))
        valid_streak[track_id] = valid_streak.get(track_id, 0) + 1 if instant_valid else 0
        is_valid = instant_valid and valid_streak[track_id] >= args.dynamic_valid_streak_frames
        audit = {
            **geometry,
            "age": int(getattr(obj, "age", 0)),
            "median_speed_m_s": median_speed,
            "speed_samples": len(history),
            "valid_streak": valid_streak[track_id],
            "valid": is_valid,
            "checks": checks,
            "block_reasons": [name for name, ok in checks.items() if not ok]
            + ([] if valid_streak[track_id] >= args.dynamic_valid_streak_frames else ["valid_streak_not_ready"]),
        }
        audits[track_id] = audit
        if is_valid:
            valid.append(obj)
    for mapping in (speed_history, valid_streak):
        for track_id in list(mapping):
            if track_id not in active_ids:
                del mapping[track_id]
    return valid, audits


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


def wait_for_candidate_goal(
    robot: Any,
    q_goal: np.ndarray,
    *,
    goal_tolerance_rad: float,
    min_execution_wait_s: float,
    motion_timeout_s: float,
    poll_s: float,
    min_motion_rad: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    initial = np.asarray(robot.get_joint(), dtype=np.float64)
    last = initial.copy()
    max_motion = 0.0
    while time.perf_counter() - started <= motion_timeout_s:
        now = time.perf_counter()
        last = np.asarray(robot.get_joint(), dtype=np.float64)
        err = joint_error(last, q_goal)
        max_motion = max(max_motion, float(np.max(np.abs(last - initial))))
        samples.append(
            {
                "t_s": now - started,
                "actual_joint_rad": last.tolist(),
                "goal_l2_error_rad": err["l2_rad"],
                "goal_max_abs_error_rad": err["max_abs_rad"],
                "max_motion_from_start_rad": max_motion,
            }
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
        },
        samples,
    )


def execute_fast_candidate_offline_track(
    robot: Any,
    candidate_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidate_csv = candidate_dir / "fast_ccro_nubs_candidate.csv"
    times, qs = load_fast_candidate_csv(candidate_csv)
    times_exec, qs_exec = resample_for_offline_track(
        times,
        qs,
        playback_duration_s=args.candidate_playback_duration_s,
        controller_period_s=args.candidate_controller_waypoint_period_s,
    )
    times_exec, qs_exec = maybe_downsample(times_exec, qs_exec, args.candidate_max_waypoints)
    min_wait = args.candidate_min_execution_wait_s
    if min_wait <= 0.0 and args.candidate_playback_duration_s > 0.0:
        min_wait = 0.90 * args.candidate_playback_duration_s

    log: dict[str, Any] = {
        "candidate_csv": str(candidate_csv),
        "robot_commanded": False,
        "source_trajectory_stats": trajectory_stats(times, qs),
        "execution_waypoint_stats": trajectory_stats(times_exec, qs_exec),
        "playback_duration_s": args.candidate_playback_duration_s,
        "controller_waypoint_period_s": args.candidate_controller_waypoint_period_s,
        "joint_velc": args.candidate_joint_velc,
        "joint_acc": args.candidate_joint_acc,
        "min_execution_wait_s": min_wait,
    }

    if not hasattr(robot, "offline_track_execute_joints"):
        raise RuntimeError("current robot .so does not expose offline_track_execute_joints")
    actual_start = np.asarray(robot.get_joint(), dtype=np.float64)
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
            "Step 2/2: candidate has passed software checks. "
            "Confirm obstacle state and emergency stop, then press Enter to execute the local candidate.",
        )

    started = time.perf_counter()
    ret_info = robot.offline_track_execute_joints(
        qs_exec.tolist(),
        args.candidate_joint_velc,
        args.candidate_joint_acc,
        False,
        True,
        True,
    )
    log["robot_commanded"] = True
    log["offline_track_return"] = dict(ret_info)
    if int(ret_info.get("startup_ret", -9999)) != 0:
        raise RuntimeError(f"offline track startup failed: {ret_info}")
    goal_check, feedback_samples = wait_for_candidate_goal(
        robot,
        qs_exec[-1],
        goal_tolerance_rad=args.candidate_goal_tolerance_rad,
        min_execution_wait_s=min_wait,
        motion_timeout_s=args.candidate_motion_timeout_s,
        poll_s=args.poll_s,
        min_motion_rad=args.candidate_min_observed_motion_rad,
    )
    log["goal_check"] = goal_check
    log["feedback_samples"] = feedback_samples
    log["elapsed_s"] = time.perf_counter() - started
    if not goal_check["reached"]:
        raise RuntimeError(f"dynamic candidate offline track did not reach goal: {goal_check}")
    log["status"] = "COMPLETED_DYNAMIC_CANDIDATE_EXECUTION"
    return log


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
    obstacle_audit: dict[str, Any],
) -> dict[str, Any]:
    evaluator, verifier, limits = make_risk_stack(stage4_config, stage4_model, None)
    forecast = constant_forecast(center, velocity, radius)
    head, tail, durations, p_inner, q_goal = make_local_reference(
        q_now, qd_now, args, reference_goal=reference_goal
    )
    reference_trajectory = NUBSTrajectory6D().generate(p_inner, head, tail, durations)
    started = time.perf_counter()
    result = run_repair_v3(
        evaluator,
        forecast,
        limits,
        p_inner,
        head,
        tail,
        durations,
        dense_active=True,
        v4_mode=True,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    verification = verifier.verify(
        result.trajectory,
        forecast,
        current_q=q_now,
        current_qd=qd_now,
        current_qdd=np.zeros(6),
        q_goal=q_goal,
        solver_success=True,
    )
    reference_verification = verifier.verify(
        reference_trajectory,
        forecast,
        current_q=q_now,
        current_qd=qd_now,
        current_qdd=np.zeros(6),
        q_goal=q_goal,
        solver_success=True,
    )
    clearance_gain = float(verification.min_distance - reference_verification.min_distance)
    candidate_samples = result.trajectory.dense_sample(0.02).q
    reference_samples = reference_trajectory.dense_sample(0.02).q
    max_delta_q = float(np.max(np.abs(candidate_samples - reference_samples)))
    repair_step_ok = int(result.accepted_steps) > 0
    accepted = bool(
        elapsed_ms <= args.fast_budget_ms
        and verification.min_distance >= args.online_accept_m
        and all({**verification.checks, "solver_ok": True}.values())
        and repair_step_ok
        and clearance_gain >= args.min_clearance_improvement_m
        and max_delta_q >= args.min_candidate_delta_q_rad
    )
    rejection_reasons = []
    if elapsed_ms > args.fast_budget_ms:
        rejection_reasons.append("fast_budget_exceeded")
    if verification.min_distance < args.online_accept_m:
        rejection_reasons.append("online_clearance_failed")
    failed_checks = [name for name, ok in verification.checks.items() if not ok]
    if failed_checks:
        rejection_reasons.append("verification_checks_failed:" + ",".join(failed_checks))
    if not repair_step_ok:
        rejection_reasons.append("no_accepted_repair_step")
    if clearance_gain < args.min_clearance_improvement_m:
        rejection_reasons.append("insufficient_clearance_improvement")
    if max_delta_q < args.min_candidate_delta_q_rad:
        rejection_reasons.append("candidate_motion_indistinguishable_from_reference")
    candidate_dir = trial_dir / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    save_trajectory_csv(candidate_dir / "fast_ccro_nubs_candidate.csv", result.trajectory, dt=0.01)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ACCEPTED_CANDIDATE" if accepted else "REJECTED_CANDIDATE",
        "accepted_for_switch": accepted,
        "repair_step_ok": repair_step_ok,
        "rejection_reasons": rejection_reasons,
        "candidate_is_reference_continuation": not repair_step_ok,
        "fast_elapsed_ms": elapsed_ms,
        "fast_budget_ms": args.fast_budget_ms,
        "online_accept_m": args.online_accept_m,
        "verification_min_distance_m": verification.min_distance,
        "reference_online_min_distance_m": reference_verification.min_distance,
        "candidate_online_min_distance_m": verification.min_distance,
        "clearance_improvement_m": clearance_gain,
        "min_clearance_improvement_m": args.min_clearance_improvement_m,
        "max_delta_q_from_reference_rad": max_delta_q,
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
            elapsed_ms
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
        "obstacle_association": obstacle_audit,
        "risk_links": sorted(risk_links),
        "candidate_csv": str(candidate_dir / "fast_ccro_nubs_candidate.csv"),
    }
    write_json(candidate_dir / "candidate_summary.json", payload)
    return payload


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.scene not in SCENARIOS:
        raise ValueError(f"unknown scene {args.scene}")
    trial_dir = build_trial_dir(args)
    trial_dir.mkdir(parents=True, exist_ok=True)
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
        "parameters": vars(args),
        "events": [],
    }
    if args.mode != "shadow" and args.operator_phrase != REQUIRED_OPERATOR_PHRASE:
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

    tracker = OccupancyTracker(
        association_distance=float(safety.get("association_distance", 0.20)),
        alpha=float(safety.get("velocity_alpha", 0.3)),
        pos_alpha=float(safety.get("pos_alpha", 0.3)),
        motion_gate=float(safety.get("motion_gate", 0.005)),
        velocity_dead_zone=float(safety.get("velocity_dead_zone", 0.01)),
        shape_alpha=float(safety.get("shape_alpha", 0.4)),
    )
    denoiser = (
        TemporalDenoiser(args.denoise_voxel, args.denoise_conf, args.denoise_decay)
        if args.temporal_denoise
        else None
    )

    rows: list[dict[str, Any]] = []
    q_history: list[tuple[float, np.ndarray]] = []
    candidate_summary: dict[str, Any] | None = None
    candidate_execution_summary: dict[str, Any] | None = None
    commander: RobotCommander | None = None
    guided_controller: AdaptiveSafetyController | None = None
    triggered = False
    trigger_frame = None
    guard_stopped = False
    current_distance_stopped = False
    t0_wall = time.time()
    caught_error: dict[str, Any] | None = None
    ref_robot_points: np.ndarray | None = None
    ref_robot_motion_y: float | None = None
    robot_motion_modes = {"moving-shadow-stop", "live-stop-replan-execute"}
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
            input(f"\n[{args.scene}] shadow 模式：{SCENARIOS[args.scene]['prompt']}。按 Enter 开始 {args.duration_s:.1f}s 采集...")

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
        dynamic_speed_history: dict[int, list[float]] = {}
        dynamic_valid_streak: dict[int, int] = {}
        frame_index = 0
        while True:
            frame_started = time.perf_counter()
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
                    tracker = OccupancyTracker(
                        association_distance=float(safety.get("association_distance", 0.20)),
                        alpha=float(safety.get("velocity_alpha", 0.3)),
                        pos_alpha=float(safety.get("pos_alpha", 0.3)),
                        motion_gate=float(safety.get("motion_gate", 0.005)),
                        velocity_dead_zone=float(safety.get("velocity_dead_zone", 0.01)),
                        shape_alpha=float(safety.get("shape_alpha", 0.4)),
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
                    print("[REFERENCE ARMED] introduce D1 foam obstacle now", flush=True)
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
            plane_removal = None
            if args.remove_planes:
                plane_removal = {"enabled": True, "distance_threshold": args.plane_dist, "max_planes": args.max_planes}
            cluster_result = FastClusteringFilter(
                scene_points,
                robot_points,
                workspace=getattr(processor, "_workspace", None),
                plane_removal=plane_removal,
                eps=args.cluster_eps,
                min_samples=args.cluster_min_samples,
                min_points=args.cluster_min_points,
                min_volume=args.cluster_min_volume,
            )
            clusters = list(cluster_result.clusters)
            guard_clusters = filter_guard_clusters(clusters, args) if args.mode in robot_motion_modes else clusters
            eval_clusters = guard_clusters if args.mode in robot_motion_modes else clusters
            detections = [
                make_occupancy_object(cluster.points, timestamp=timestamp, margin=float(safety.get("shape_margin", 0.02)))
                for cluster in eval_clusters
            ]
            tracked = tracker.update(detections, timestamp=timestamp)
            stable = [obj for obj in tracked if obj.age >= args.min_track_age]
            dynamic_tracks, dynamic_audits = update_dynamic_track_validity(
                stable,
                eval_clusters,
                dynamic_speed_history,
                dynamic_valid_streak,
                args,
            )
            risk_spheres = predict_risk_spheres(
                dynamic_tracks,
                horizon=args.prediction_horizon_s,
                step=args.prediction_step_s,
                margin=args.prediction_margin_m,
                uncertainty=args.prediction_uncertainty_m,
                static_speed_threshold=float(safety.get("prediction_static_speed_threshold", 0.08)),
                static_margin=float(safety.get("prediction_static_margin", 0.0)),
                velocity_radius_scale=float(safety.get("prediction_velocity_radius_scale", 0.1)),
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

                guard = guided_guard_distance(rob_pts_for_guard, eval_clusters, tracked, motion_dir_y=motion_dir_y)
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
                    log["events"].append(
                        {
                            "type": "GUIDED_POINTCLOUD_GUARD_STOP",
                            "frame": frame_index,
                            "t_s": now_rel,
                            "guard_distance_m": guard_distance,
                            "threshold_m": args.guided_hard_stop_m,
                            "guard_decision": guided_controller.last_decision,
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
                "guard_cluster_count": int(len(eval_clusters)),
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
                "predicted_object_radius_m": "" if row_dynamic_audit is None else f"{float(row_dynamic_audit['raw_radius']):.6f}",
                "predicted_object_age": "" if row_dynamic_audit is None else int(row_dynamic_audit["age"]),
                "predicted_object_association_error_m": "" if row_dynamic_audit is None else f"{float(row_dynamic_audit['association_error_m']):.6f}",
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
                "guard_cluster_count": guided_info.get("guard_cluster_count", len(eval_clusters) if args.mode == "moving-shadow-stop" else ""),
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
            scene_risk_links = set(SCENARIOS[args.scene]["risk_links"])
            trigger_block_reason = ""
            if triggered:
                trigger_block_reason = "already_triggered"
            elif not reference_armed:
                trigger_block_reason = "reference_not_armed"
            elif args.reference_audit_only:
                trigger_block_reason = "reference_audit_only"
            elif reference_audit.get("step_was_clamped", False):
                trigger_block_reason = "reference_step_jump"
            elif not reference_match_ok:
                trigger_block_reason = "reference_match_error"
            elif not local_reference_sanity_ok:
                trigger_block_reason = "local_reference_sanity_failed"
            elif len(stable) == 0:
                trigger_block_reason = "no_stable_track"
            elif len(dynamic_tracks) == 0:
                trigger_block_reason = "predicted_track_not_dynamic"
            elif not np.isfinite(trigger_distance):
                trigger_block_reason = "no_finite_future_reference_risk"
            elif predicted_best["link"] not in scene_risk_links:
                trigger_block_reason = "predicted_non_scene_link"
            elif trigger_distance >= trigger_threshold:
                trigger_block_reason = "future_reference_clearance_above_threshold"
            elif current_distance <= args.moving_shadow_current_stop_m:
                trigger_block_reason = "current_distance_in_stop_zone"
            elif reference_arm_perf is not None and time.perf_counter() - reference_arm_perf < args.arm_delay_s:
                trigger_block_reason = "arm_delay"
            row["trigger_block_reason"] = trigger_block_reason
            if (
                args.mode in robot_motion_modes
                and np.isfinite(current_distance)
                and current_distance <= args.moving_shadow_current_stop_m
            ):
                link_allowed = current_link in scene_risk_links
                hard_any_link = current_distance <= args.moving_shadow_any_link_hard_stop_m
                if not link_allowed and not hard_any_link:
                    log["events"].append(
                        {
                            "type": "CURRENT_DISTANCE_IGNORED_NON_SCENE_LINK",
                            "frame": frame_index,
                            "t_s": now_rel,
                            "distance_m": current_distance,
                            "nearest_link": current_link,
                            "allowed_links": sorted(scene_risk_links),
                            "threshold_m": args.moving_shadow_current_stop_m,
                            "any_link_hard_stop_m": args.moving_shadow_any_link_hard_stop_m,
                        }
                    )
                    frame_index += 1
                    continue
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
                log["events"].append({"type": "TRIGGER", "frame": frame_index, "t_s": now_rel, "predicted_distance_m": trigger_distance})
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
                selected_obj = select_stable_object(stable, predicted_best, risk_spheres)
                obstacle = track_geometry(selected_obj, eval_clusters, args.default_obstacle_radius_m)
                if obstacle["association_error_m"] > args.max_track_cluster_association_m:
                    raise RuntimeError(
                        f"selected track/cluster association error {obstacle['association_error_m']:.4f} m "
                        f"exceeds {args.max_track_cluster_association_m:.4f} m"
                    )
                if reference is None:
                    raise RuntimeError("a recorded reference is required to construct a Fast local repair")
                # Both active modes stop before repair. Anchor the candidate at
                # the measured stopped state, not at the last moving sample.
                if args.mode in robot_motion_modes:
                    time.sleep(max(args.post_stop_settle_s, 0.0))
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
                reference_goal = reference.state_after(args.local_horizon_s)
                candidate_summary = run_fast_repair(
                    args,
                    stage4_config,
                    stage4_model,
                    q_now=q_repair_start,
                    qd_now=qd_repair_start,
                    center=obstacle["center"],
                    velocity=obstacle["velocity"],
                    radius=obstacle["inflated_radius"],
                    risk_links=set(SCENARIOS[args.scene]["risk_links"]),
                    trial_dir=trial_dir,
                    reference_goal=reference_goal,
                    obstacle_audit=obstacle,
                )
                log["events"].append({"type": candidate_summary["status"], "frame": frame_index, "t_s": now_rel, "candidate": candidate_summary})
                if args.mode == "moving-shadow-stop":
                    break
                if args.mode == "live-stop-replan-execute":
                    if not candidate_summary["accepted_for_switch"]:
                        log["events"].append(
                            {
                                "type": "LIVE_CANDIDATE_NOT_EXECUTED",
                                "reason": "candidate_rejected",
                                "candidate_status": candidate_summary["status"],
                                "rejection_reasons": candidate_summary.get("rejection_reasons", []),
                            }
                        )
                        break
                    log["events"].append(
                        {
                            "type": "LIVE_CANDIDATE_EXECUTION_BLOCKED_PENDING_FRESH_RECHECK",
                            "reason": (
                                "trigger-time obstacle data must not authorize execution after planning/operator delay; "
                                "implement and validate a fresh RGB-D candidate recheck first"
                            ),
                        }
                    )
                    break
                    if not args.allow_live_candidate_execution:
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
                        candidate_execution_summary = execute_fast_candidate_offline_track(robot, candidate_dir, args)
                        write_json(candidate_dir / "live_candidate_execution_log.json", candidate_execution_summary)
                        log["events"].append({"type": "LIVE_CANDIDATE_EXECUTED", "execution": candidate_execution_summary})
                    except Exception as exc:
                        stop_ret = maybe_move_stop(robot)
                        candidate_execution_summary = {
                            "status": "FAILED_DYNAMIC_CANDIDATE_EXECUTION",
                            "error": str(exc),
                            "traceback": traceback.format_exc(limit=20),
                            "stop_return": stop_ret,
                        }
                        write_json(candidate_dir / "live_candidate_execution_log.json", candidate_execution_summary)
                        log["events"].append({"type": "LIVE_CANDIDATE_EXECUTION_FAILED", "execution": candidate_execution_summary})
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
        if commander is not None:
            try:
                commander.stop()
            except Exception:
                pass
        processor.stop()

    write_csv(trial_dir / "frames.csv", rows, FRAME_FIELDS)
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
    if caught_error is not None:
        final_status = "FAILED"
    alignment_rows = [row for row in rows if int(row.get("reference_armed", 0)) == 1]
    alignment_checks = {
        "reference_armed": any(event.get("type") == "REFERENCE_ARMED" for event in log["events"])
        or args.mode not in robot_motion_modes,
        "armed_rows_present": len(alignment_rows) > 0,
        "index_steps_bounded": bool(alignment_rows)
        and max(int(row["reference_index_step"]) for row in alignment_rows) <= args.max_reference_step,
        "no_clamped_step": bool(alignment_rows)
        and not any(int(row.get("reference_step_clamped", 0)) for row in alignment_rows),
        "reference_match_ok": bool(alignment_rows)
        and max(float(row["reference_joint_match_max_rad"]) for row in alignment_rows if row["reference_joint_match_max_rad"] != "")
        <= args.reference_match_max_rad,
        "reference_progress_complete": bool(alignment_rows)
        and int(alignment_rows[-1]["reference_index"]) >= int(0.95 * (len(reference.times) - 1)) if reference is not None else True,
        "no_dynamic_trigger": not triggered,
    }
    if args.reference_audit_only and caught_error is None:
        final_status = "REFERENCE_ALIGNMENT_PASS" if all(alignment_checks.values()) else "REFERENCE_ALIGNMENT_FAIL"
    log.update(
        {
            "status": final_status,
            "error": caught_error,
            "trigger_frame": trigger_frame,
            "candidate_status": None if candidate_summary is None else candidate_summary["status"],
            "candidate_accepted": None if candidate_summary is None else candidate_summary["accepted_for_switch"],
            "candidate_execution_status": None if candidate_execution_summary is None else candidate_execution_summary.get("status"),
            "candidate_execution": candidate_execution_summary,
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
                "max_reference_joint_match_rad": None if not alignment_rows else max(
                    float(row["reference_joint_match_max_rad"])
                    for row in alignment_rows
                    if row["reference_joint_match_max_rad"] != ""
                ),
            },
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
        choices=["shadow", "moving-shadow-stop", "live-stop-replan-execute", "live-execute"],
        default="shadow",
    )
    parser.add_argument("--operator-phrase", default="")
    parser.add_argument("--robot-ip", default="192.168.123.96")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    parser.add_argument("--stage4-config", type=Path, default=ROOT / "config" / "ccro_stage4.yaml")
    parser.add_argument("--urdf", type=Path, default=ROOT / "urdf" / "aubo_i16_gripper.urdf")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--duration-s", type=float, default=18.0)
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
    parser.add_argument("--cluster-eps", type=float, default=0.06)
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
        default=0.18,
        help="immediate current-distance stop threshold for moving-shadow-stop pilot",
    )
    parser.add_argument(
        "--moving-shadow-any-link-hard-stop-m",
        type=float,
        default=0.03,
        help="absolute all-link hard stop; above this, current-distance stops are limited to the scene risk links",
    )
    parser.add_argument("--online-accept-m", type=float, default=0.09)
    parser.add_argument("--min-clearance-improvement-m", type=float, default=0.003)
    parser.add_argument("--min-candidate-delta-q-rad", type=float, default=1.0e-4)
    parser.add_argument("--fast-budget-ms", type=float, default=150.0)
    parser.add_argument("--local-horizon-s", type=float, default=1.0)
    parser.add_argument(
        "--reference-feedback-csv",
        type=Path,
        default=None,
        help="recorded one-way reference_feedback.csv; mandatory for moving modes",
    )
    parser.add_argument("--local-segments", type=int, default=5)
    parser.add_argument("--default-obstacle-radius-m", type=float, default=0.055)
    parser.add_argument("--max-track-cluster-association-m", type=float, default=0.08)
    parser.add_argument("--min-dynamic-trigger-speed-m-s", type=float, default=0.08)
    parser.add_argument("--dynamic-speed-window", type=int, default=3)
    parser.add_argument("--dynamic-valid-streak-frames", type=int, default=2)
    parser.add_argument("--dynamic-radius-min-m", type=float, default=0.03)
    parser.add_argument("--dynamic-radius-max-m", type=float, default=0.10)
    parser.add_argument("--min-local-motion-rad", type=float, default=0.002)
    parser.add_argument("--shadow-joint-probe-rad", type=float, default=0.025)
    parser.add_argument("--home-joints-deg", default="0,0,90,0,90,0")
    parser.add_argument("--x-offset", type=float, default=0.10)
    parser.add_argument("--y-start", type=float, default=0.4)
    parser.add_argument("--y-goal", type=float, default=-0.4)
    parser.add_argument("--line-velocity-m-s", type=float, default=0.020)
    parser.add_argument("--line-acc-m-s2", type=float, default=0.05)
    parser.add_argument("--guided-range-m", type=float, default=0.20)
    parser.add_argument("--guided-base-omega", type=float, default=0.15)
    parser.add_argument("--guided-d-safe-m", type=float, default=0.22)
    parser.add_argument("--guided-d-slow-m", type=float, default=0.14)
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
    parser.add_argument("--allow-live-candidate-execution", action="store_true")
    parser.add_argument("--live-execute-candidate-phrase", default="")
    parser.add_argument("--candidate-playback-duration-s", type=float, default=6.0)
    parser.add_argument("--candidate-controller-waypoint-period-s", type=float, default=0.005)
    parser.add_argument("--candidate-max-waypoints", type=int, default=0)
    parser.add_argument("--candidate-joint-velc", type=float, default=0.006)
    parser.add_argument("--candidate-joint-acc", type=float, default=0.012)
    parser.add_argument("--candidate-start-tolerance-rad", type=float, default=0.035)
    parser.add_argument("--candidate-goal-tolerance-rad", type=float, default=0.012)
    parser.add_argument("--candidate-min-observed-motion-rad", type=float, default=0.003)
    parser.add_argument("--candidate-min-execution-wait-s", type=float, default=0.0)
    parser.add_argument("--candidate-motion-timeout-s", type=float, default=45.0)
    parser.add_argument("--candidate-pre-execute-settle-s", type=float, default=0.35)
    parser.add_argument("--post-stop-settle-s", type=float, default=0.25)
    parser.add_argument("--candidate-execute-confirm", action="store_true", default=True)
    parser.add_argument("--no-candidate-execute-confirm", dest="candidate_execute_confirm", action="store_false")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
