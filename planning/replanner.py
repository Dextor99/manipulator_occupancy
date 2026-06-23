"""Non-blocking risk-triggered replanning for CCRO-NUBS stage four."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from enum import Enum
import math
import multiprocessing as mp
import queue
import time

import numpy as np

from .dynamic_optimizer import DynamicRiskNUBSOptimizer, DynamicRiskOptimizationResult
from .nubs_trajectory import DIMENSION, NUBSTrajectory6D
from .obstacle_forecast import ObstacleForecast, ShiftedForecast
from .optimizer import JointLimits
from .spatiotemporal_risk import SpatioTemporalRiskEvaluator
from .trajectory_buffer import TrajectoryBuffer
from .verifier import DynamicTrajectoryVerifier


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class FutureRiskReport:
    level: RiskLevel
    min_distance: float
    risk_cost: float
    time_to_danger: float
    nearest_link: str | None
    nearest_object_id: int | None
    time_to_stop: float = math.inf


@dataclass
class ReplanEvent:
    attempt_id: int
    submitted_timestamp: float
    completed_timestamp: float
    planned_switch_timestamp: float
    outcome: str
    elapsed_ms: float
    solver_success: bool
    candidate_accepted: bool
    candidate_min_distance: float
    required_min_distance: float
    rejection_reasons: list[str]
    verification_checks: dict[str, bool]
    candidate_goal_error: float


@dataclass
class SafetyEvent:
    timestamp: float
    reason: str


@dataclass
class PollResult:
    outcome: str
    result: DynamicRiskOptimizationResult | None = None


OptimizerFactory = Callable[
    [np.ndarray, np.ndarray, np.ndarray, ObstacleForecast],
    DynamicRiskNUBSOptimizer,
]


@dataclass
class _ActiveJob:
    attempt_id: int
    submitted_timestamp: float
    planned_switch_timestamp: float
    deadline_timestamp: float
    process: mp.Process
    result_queue: mp.Queue
    head: np.ndarray
    tail: np.ndarray
    durations: np.ndarray
    shifted_forecast: ShiftedForecast
    required_min_distance: float
    payload: dict | None = None


def _worker(
    optimizer: DynamicRiskNUBSOptimizer,
    warm: np.ndarray,
    verifier: DynamicTrajectoryVerifier,
    forecast: ObstacleForecast,
    q_now: np.ndarray,
    qd_now: np.ndarray,
    qdd_now: np.ndarray,
    q_goal: np.ndarray,
    output: mp.Queue,
    artificial_delay: float,
) -> None:
    """Child-process entry point; only plain arrays/dicts cross the queue."""
    started = time.perf_counter()
    try:
        if artificial_delay > 0.0:
            time.sleep(artificial_delay)
        result = optimizer.optimize(warm)
        verification = verifier.verify(
            result.trajectory,
            forecast,
            current_q=q_now,
            current_qd=qd_now,
            current_qdd=qdd_now,
            q_goal=q_goal,
            solver_success=result.success,
        )
        scalar_result = {
            item.name: getattr(result, item.name)
            for item in fields(result)
            if item.name not in {"trajectory", "p_inner", "durations"}
        }
        output.put(
            {
                "ok": True,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "p_inner": result.p_inner,
                "optimization": scalar_result,
                "verification": asdict(verification),
            }
        )
    except BaseException as exc:  # child must report numerical/native failures
        output.put(
            {
                "ok": False,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


class ReplanManager:
    """Risk monitor plus one-slot asynchronous optimization scheduler.

    A candidate is anchored at a future switch slot.  While it is computed,
    the current reference keeps running; HIGH risk, timeout or rejection
    transfers authority to a conservative position hold.
    """

    def __init__(
        self,
        make_optimizer: OptimizerFactory,
        evaluator: SpatioTemporalRiskEvaluator,
        forecast: ObstacleForecast,
        verifier: DynamicTrajectoryVerifier,
        limits: JointLimits,
        *,
        d_replan: float = 0.12,
        d_stop: float = 0.04,
        d_safe: float = 0.10,
        d_accept: float = 0.08,
        min_improvement: float = 0.005,
        replan_interval: float = 0.5,
        hysteresis_enter: float = 0.015,
        hysteresis_exit: float = 0.01,
        evaluate_steps: int = 20,
        evaluate_horizon: float | None = None,
        max_replan_attempts: int = 2,
        planning_budget: float = 1.0,
        switch_delay: float | None = None,
        emergency_lead_time: float = 0.30,
        artificial_worker_delay: float = 0.0,
    ) -> None:
        if not (0.0 < d_stop < d_accept <= d_safe <= d_replan):
            raise ValueError("expected 0 < d_stop < d_accept <= d_safe <= d_replan")
        if (
            min_improvement < 0.0
            or replan_interval <= 0.0
            or planning_budget <= 0.0
            or emergency_lead_time < 0.0
        ):
            raise ValueError("improvement, interval and budget parameters are invalid")
        if evaluate_steps < 2 or max_replan_attempts < 1:
            raise ValueError("evaluate_steps >= 2 and max_replan_attempts >= 1 required")
        self.make_optimizer = make_optimizer
        self.evaluator = evaluator
        self.forecast = forecast
        self.verifier = verifier
        self.limits = limits
        self.d_replan = float(d_replan)
        self.d_stop = float(d_stop)
        self.d_safe = float(d_safe)
        self.d_accept = float(d_accept)
        self.min_improvement = float(min_improvement)
        self.replan_interval = float(replan_interval)
        self.hysteresis_enter = float(hysteresis_enter)
        self.hysteresis_exit = float(hysteresis_exit)
        self.evaluate_steps = int(evaluate_steps)
        self.evaluate_horizon = None if evaluate_horizon is None else float(evaluate_horizon)
        self.max_replan_attempts = int(max_replan_attempts)
        self.planning_budget = float(planning_budget)
        self.switch_delay = float(planning_budget if switch_delay is None else switch_delay)
        if not (0.0 < self.switch_delay <= self.planning_budget):
            raise ValueError("switch_delay must be in (0, planning_budget]")
        self.emergency_lead_time = float(emergency_lead_time)
        self.artificial_worker_delay = float(artificial_worker_delay)

        self._current_level = RiskLevel.LOW
        self._last_replan_time = -math.inf
        self._events: list[ReplanEvent] = []
        self._safety_events: list[SafetyEvent] = []
        self._job: _ActiveJob | None = None
        self._attempt_count = 0
        self._safety_hold = False

    @property
    def current_level(self) -> RiskLevel:
        return self._current_level

    @property
    def replan_events(self) -> list[ReplanEvent]:
        return list(self._events)

    @property
    def safety_events(self) -> list[SafetyEvent]:
        return list(self._safety_events)

    @property
    def replan_count(self) -> int:
        return self._attempt_count

    @property
    def safety_takeover_count(self) -> int:
        return len(self._safety_events)

    @property
    def replan_in_flight(self) -> bool:
        return self._job is not None

    @property
    def safety_hold_required(self) -> bool:
        return self._safety_hold or self._current_level == RiskLevel.HIGH

    def evaluate_active_trajectory(
        self, timestamp: float, buffer: TrajectoryBuffer
    ) -> FutureRiskReport:
        remaining = buffer.remaining_time(timestamp)
        forecast_remaining = float(self.forecast.valid_horizon) - float(timestamp)
        if remaining <= 1.0e-9 or forecast_remaining <= 1.0e-9 or buffer.trajectory is None:
            self._current_level = RiskLevel.LOW
            return FutureRiskReport(
                RiskLevel.LOW, math.inf, 0.0, math.inf, None, None
            )
        horizon = min(remaining, forecast_remaining)
        if self.evaluate_horizon is not None:
            horizon = min(horizon, self.evaluate_horizon)
        local_start = buffer.elapsed(timestamp)
        deltas = np.linspace(0.0, horizon, self.evaluate_steps)
        min_distance = math.inf
        nearest_link = None
        nearest_oid = None
        time_to_danger = math.inf
        time_to_stop = math.inf
        current_distance = math.inf
        total_cost = 0.0
        for delta in deltas:
            q = buffer.trajectory.evaluate(float(local_start + delta))
            risk = self.evaluator.configuration(
                q, self.forecast, float(timestamp + delta), with_gradient=False
            )
            total_cost += risk.cost
            if delta == 0.0:
                current_distance = risk.min_distance
            if risk.min_distance < min_distance:
                min_distance = risk.min_distance
                nearest_link = risk.nearest_link
                nearest_oid = risk.nearest_object_id
            if risk.min_distance < self.d_safe and not math.isfinite(time_to_danger):
                time_to_danger = float(delta)
            if risk.min_distance <= self.d_stop and not math.isfinite(time_to_stop):
                time_to_stop = float(delta)
        self._current_level = self._resolve_level(
            float(min_distance),
            current_distance=float(current_distance),
            time_to_stop=time_to_stop,
        )
        return FutureRiskReport(
            self._current_level,
            float(min_distance),
            float(total_cost / len(deltas)),
            time_to_danger,
            nearest_link,
            nearest_oid,
            time_to_stop,
        )

    def submit_replan(
        self,
        timestamp: float,
        buffer: TrajectoryBuffer,
        q_goal: np.ndarray,
        duration_template: np.ndarray,
        report: FutureRiskReport,
        *,
        force: bool = False,
    ) -> bool:
        if self._job is not None or buffer.trajectory is None:
            return False
        if not force:
            if self._current_level != RiskLevel.MEDIUM:
                return False
            if timestamp - self._last_replan_time < self.replan_interval:
                return False
            if self._attempt_count >= self.max_replan_attempts:
                return False
        planned_switch = float(timestamp) + self.switch_delay
        remaining_after_budget = buffer.remaining_time(timestamp) - self.switch_delay
        forecast_after_budget = self.forecast.valid_horizon - (
            float(timestamp) + self.switch_delay
        )
        available = min(remaining_after_budget, forecast_after_budget)
        if available <= 0.10:
            self.engage_safety(timestamp, "insufficient remaining prediction/trajectory horizon")
            return False

        q_now, qd_now, qdd_now = buffer.sample_state(planned_switch)
        template = np.asarray(duration_template, dtype=np.float64)
        durations = template * (available / float(np.sum(template)))
        head = NUBSTrajectory6D.make_boundary_state(q_now, qd_now, qdd_now)
        tail = NUBSTrajectory6D.make_boundary_state(
            np.asarray(q_goal, dtype=np.float64), np.zeros(DIMENSION), np.zeros(DIMENSION)
        )
        shifted = ShiftedForecast(self.forecast, planned_switch, available)
        warm = buffer.remaining_waypoints(planned_switch, durations)
        if warm is None or warm.shape != (len(durations) - 1, DIMENSION):
            warm = NUBSTrajectory6D.linear_inner_points(q_now, q_goal, durations)
        optimizer = self.make_optimizer(head, tail, durations, shifted)

        context = mp.get_context("fork")
        output = context.Queue(maxsize=1)
        process = context.Process(
            target=_worker,
            args=(
                optimizer,
                warm,
                self.verifier,
                shifted,
                q_now,
                qd_now,
                qdd_now,
                np.asarray(q_goal, dtype=np.float64),
                output,
                self.artificial_worker_delay,
            ),
            daemon=True,
        )
        self._attempt_count += 1
        process.start()
        required = (
            self.d_accept
            if self.min_improvement <= 0.0
            else max(self.d_accept, report.min_distance + self.min_improvement)
        )
        self._job = _ActiveJob(
            self._attempt_count,
            float(timestamp),
            planned_switch,
            float(timestamp) + self.planning_budget,
            process,
            output,
            head,
            tail,
            durations,
            shifted,
            required,
        )
        return True

    def poll_candidate(
        self, timestamp: float, buffer: TrajectoryBuffer, q_goal: np.ndarray
    ) -> PollResult:
        job = self._job
        if job is None:
            return PollResult("idle")
        if job.payload is None:
            try:
                job.payload = job.result_queue.get_nowait()
            except queue.Empty:
                pass
        if job.payload is None and not job.process.is_alive():
            try:
                job.payload = job.result_queue.get(timeout=0.10)
            except queue.Empty:
                pass
        if job.payload is None and float(timestamp) >= job.deadline_timestamp:
            try:
                job.payload = job.result_queue.get(timeout=0.05)
            except queue.Empty:
                pass
        if job.payload is None and float(timestamp) >= job.deadline_timestamp:
            self._terminate_job(job)
            self._record_failure(timestamp, job, "timeout", ["planning_budget_exceeded"])
            self._job = None
            buffer.pause(timestamp)
            self.engage_safety(timestamp, "planner timeout")
            return PollResult("timeout")
        if job.payload is None:
            return PollResult("running")
        if float(timestamp) + 1.0e-12 < job.planned_switch_timestamp:
            return PollResult("ready")

        job.process.join(timeout=0.2)
        payload = job.payload
        if not payload.get("ok", False):
            reasons = [payload.get("error", "worker failure")]
            self._record_failure(timestamp, job, "failed", reasons, payload)
            self._job = None
            buffer.pause(timestamp)
            self.engage_safety(timestamp, "planner failure")
            return PollResult("failed")

        verification = payload["verification"]
        reasons = list(verification.get("reasons", []))
        if verification["min_distance"] + 1.0e-12 < job.required_min_distance:
            reasons.append("insufficient_distance_improvement")
        accepted = bool(verification["accepted"] and not reasons)
        trajectory = NUBSTrajectory6D().generate(
            np.asarray(payload["p_inner"]), job.head, job.tail, job.durations
        )
        result = self._reconstruct_result(payload, trajectory, job.durations)
        event = ReplanEvent(
            job.attempt_id,
            job.submitted_timestamp,
            float(timestamp),
            job.planned_switch_timestamp,
            "accepted" if accepted else "rejected",
            float(payload["elapsed_ms"]),
            bool(payload["optimization"]["success"]),
            accepted,
            float(verification["min_distance"]),
            job.required_min_distance,
            reasons,
            dict(verification.get("checks", {})),
            float(verification.get("goal_error", math.inf)),
        )
        self._events.append(event)
        self._job = None
        if not accepted:
            buffer.pause(timestamp)
            self.engage_safety(timestamp, "candidate rejected")
            return PollResult("rejected", result)

        buffer.set_active(trajectory, job.planned_switch_timestamp, q_goal)
        self._last_replan_time = float(timestamp)
        self._current_level = RiskLevel.LOW
        self._safety_hold = False
        return PollResult("accepted", result)

    def abort_for_safety(
        self, timestamp: float, buffer: TrajectoryBuffer, reason: str
    ) -> None:
        """Cancel a pending candidate and hold the current executed state."""
        if self._job is not None:
            job = self._job
            self._terminate_job(job)
            self._record_failure(timestamp, job, "aborted", [reason])
            self._job = None
        buffer.pause(timestamp)
        self.engage_safety(timestamp, reason)

    def engage_safety(self, timestamp: float, reason: str) -> None:
        already_engaged = self._safety_hold and self._current_level == RiskLevel.HIGH
        self._current_level = RiskLevel.HIGH
        self._safety_hold = True
        if not already_engaged:
            self._safety_events.append(SafetyEvent(float(timestamp), reason))

    def shutdown(self) -> None:
        if self._job is not None:
            self._terminate_job(self._job)
            self._job = None

    def reset(self) -> None:
        self.shutdown()
        self._current_level = RiskLevel.LOW
        self._last_replan_time = -math.inf
        self._events.clear()
        self._safety_events.clear()
        self._attempt_count = 0
        self._safety_hold = False

    def _resolve_level(
        self,
        min_distance: float,
        *,
        current_distance: float | None = None,
        time_to_stop: float = math.inf,
    ) -> RiskLevel:
        current = min_distance if current_distance is None else current_distance
        if current <= self.d_stop or time_to_stop <= self.emergency_lead_time:
            return RiskLevel.HIGH
        if self._current_level == RiskLevel.HIGH:
            return (
                RiskLevel.MEDIUM
                if min_distance > self.d_stop + self.hysteresis_enter
                else RiskLevel.HIGH
            )
        if self._current_level == RiskLevel.MEDIUM:
            return (
                RiskLevel.LOW
                if min_distance >= self.d_replan + self.hysteresis_enter
                else RiskLevel.MEDIUM
            )
        return (
            RiskLevel.MEDIUM
            if min_distance <= self.d_replan - self.hysteresis_exit
            else RiskLevel.LOW
        )

    @staticmethod
    def _terminate_job(job: _ActiveJob) -> None:
        if job.process.is_alive():
            job.process.terminate()
        job.process.join(timeout=1.0)
        job.result_queue.close()

    def _record_failure(
        self,
        timestamp: float,
        job: _ActiveJob,
        outcome: str,
        reasons: list[str],
        payload: dict | None = None,
    ) -> None:
        data = payload or {}
        verification = data.get("verification", {})
        optimization = data.get("optimization", {})
        self._events.append(
            ReplanEvent(
                job.attempt_id,
                job.submitted_timestamp,
                float(timestamp),
                job.planned_switch_timestamp,
                outcome,
                float(data.get("elapsed_ms", self.planning_budget * 1000.0)),
                bool(optimization.get("success", False)),
                False,
                float(verification.get("min_distance", 0.0)),
                job.required_min_distance,
                reasons,
                dict(verification.get("checks", {})),
                float(verification.get("goal_error", math.inf)),
            )
        )

    @staticmethod
    def _reconstruct_result(
        payload: dict,
        trajectory: NUBSTrajectory6D,
        durations: np.ndarray,
    ) -> DynamicRiskOptimizationResult:
        values = dict(payload["optimization"])
        return DynamicRiskOptimizationResult(
            trajectory=trajectory,
            p_inner=np.asarray(payload["p_inner"], dtype=np.float64),
            durations=np.asarray(durations, dtype=np.float64),
            **values,
        )
