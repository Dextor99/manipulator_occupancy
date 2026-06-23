"""Current and future trajectory state buffer for CCRO-NUBS.

Provides a time-indexed wrapper around ``NUBSTrajectory6D`` that supports
sampling at arbitrary wall-clock timestamps, computing remaining trajectory
horizon, and extracting warm-start waypoints for event-triggered replanning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nubs_trajectory import DIMENSION, NUBSTrajectory6D

MIN_REPLAN_TIME = 0.05  # s — below this, remaining horizon is too short to replan


@dataclass
class TrajectoryBuffer:
    """Active trajectory indexed by an external (simulation) timestamp.

    The buffer holds one active ``NUBSTrajectory6D`` and records the
    wall-clock or simulation time at which it started executing.  All
    sample methods accept an absolute *timestamp* and internally compute
    the trajectory-relative time ``tau = timestamp - start_timestamp``.
    """

    trajectory: NUBSTrajectory6D | None = None
    start_timestamp: float = 0.0
    _q_goal: np.ndarray | None = None
    _paused_at: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_active(
        self,
        trajectory: NUBSTrajectory6D,
        start_timestamp: float,
        q_goal: np.ndarray | None = None,
    ) -> None:
        """Register *trajectory* as the active one starting at *start_timestamp*."""
        self.trajectory = trajectory
        self.start_timestamp = float(start_timestamp)
        self._paused_at = None
        self._q_goal = (
            None if q_goal is None else np.asarray(q_goal, dtype=np.float64)
        )

    @property
    def q_goal(self) -> np.ndarray | None:
        return self._q_goal

    @property
    def total_duration(self) -> float:
        if self.trajectory is None:
            return 0.0
        return float(self.trajectory.total_duration)

    @property
    def is_finished(self) -> bool:
        """True if the trajectory has already been fully elapsed *at an arbitrary far-future time*.

        .. deprecated: Use ``remaining_time(timestamp) <= 0`` instead for
           time-aware checks inside simulation loops.
        """
        return self.trajectory is not None and self.elapsed(999999.0) >= self.total_duration - 1e-9

    def is_finished_at(self, timestamp: float) -> bool:
        """True if the trajectory is finished at *timestamp*."""
        return self.trajectory is None or self.remaining_time(timestamp) <= 1e-9

    def elapsed(self, timestamp: float) -> float:
        """Seconds since the active trajectory started."""
        if self.trajectory is None:
            return 0.0
        effective_timestamp = (
            min(float(timestamp), self._paused_at)
            if self._paused_at is not None
            else float(timestamp)
        )
        return max(0.0, effective_timestamp - self.start_timestamp)

    @property
    def is_paused(self) -> bool:
        return self._paused_at is not None

    def pause(self, timestamp: float) -> None:
        """Freeze trajectory time while an asynchronous candidate is pending."""
        if self.trajectory is None:
            raise RuntimeError("no active trajectory set")
        if self._paused_at is None:
            self._paused_at = max(float(timestamp), self.start_timestamp)

    def resume(self, timestamp: float) -> None:
        """Resume without introducing a jump in trajectory-relative time."""
        if self._paused_at is None:
            return
        resume_time = max(float(timestamp), self._paused_at)
        self.start_timestamp += resume_time - self._paused_at
        self._paused_at = None

    def remaining_time(self, timestamp: float) -> float:
        """Seconds remaining on the active trajectory at *timestamp*."""
        return max(0.0, self.total_duration - self.elapsed(timestamp))

    def sample_now(self, timestamp: float) -> np.ndarray:
        """Return joint position ``q`` at *timestamp*."""
        if self.trajectory is None:
            raise RuntimeError("no active trajectory set")
        tau = min(self.elapsed(timestamp), self.total_duration)
        return self.trajectory.evaluate(float(tau))

    def sample_state(
        self, timestamp: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(q, qd, qdd)`` at *timestamp*."""
        if self.trajectory is None:
            raise RuntimeError("no active trajectory set")
        tau = min(self.elapsed(float(timestamp)), self.total_duration)
        t = float(tau)
        return (
            self.trajectory.evaluate(t, 0),
            self.trajectory.evaluate(t, 1),
            self.trajectory.evaluate(t, 2),
        )

    def remaining_waypoints(
        self, timestamp: float, new_durations: np.ndarray
    ) -> np.ndarray | None:
        """Extract warm-start ``P_inner`` for a replan starting at *timestamp*.

        Samples the active trajectory at the waypoint times defined by
        the cumulative sum of *new_durations*, scaled to the remaining
        horizon of the current trajectory.  Returns ``None`` when there
        is insufficient remaining time.
        """
        if self.trajectory is None:
            return None
        tau = self.elapsed(timestamp)
        if tau >= self.total_duration - 1e-9:
            return None
        remaining = self.remaining_time(timestamp)
        total_new = float(np.sum(new_durations))
        if total_new <= 0.0 or remaining < MIN_REPLAN_TIME:
            return None
        cumulative = np.cumsum(new_durations)
        # waypoint times exclude the first (current) and last (goal) nodes
        waypoint_times = tau + (cumulative[:-1] / total_new) * remaining
        waypoint_times = np.clip(
            waypoint_times, tau, self.total_duration - 1e-10
        )
        points = np.array(
            [self.trajectory.evaluate(float(t)) for t in waypoint_times]
        )
        return points  # shape (M-1, 6)
