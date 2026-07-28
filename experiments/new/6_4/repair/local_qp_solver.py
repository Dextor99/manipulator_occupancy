"""Small linear-constraint QP fallback for 6.4 fast v4 repair.

OSQP is the intended production solver.  The current environment does not ship
with OSQP, so this module implements a deterministic projected half-space
solver with the same linearized-constraint inputs.  It avoids SLSQP and keeps
the online path explicitly convex and small.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .active_distance import ActiveDistance
from .nubs_linearization import LocalSensitivity
from .. import config_64 as cfg


@dataclass(frozen=True)
class LocalQPResult:
    success: bool
    delta: np.ndarray
    objective: float
    status: int
    message: str
    iterations: int
    min_predicted_distance: float


def _project_halfspace(x: np.ndarray, row: np.ndarray, bound: float) -> np.ndarray:
    value = float(np.dot(row, x))
    if value >= bound:
        return x
    denom = float(np.dot(row, row))
    if denom < 1.0e-16:
        return x
    return x + ((bound - value) / denom) * row


def _motion_halfspaces(sensitivity: LocalSensitivity, limits) -> list[tuple[np.ndarray, float]]:
    rows: list[tuple[np.ndarray, float]] = []
    for time_index in range(len(sensitivity.sample_times)):
        for joint in range(6):
            sq = sensitivity.sq[time_index, joint]
            sqd = sensitivity.sqd[time_index, joint]
            sqdd = sensitivity.sqdd[time_index, joint]
            q0 = float(sensitivity.q[time_index, joint])
            qd0 = float(sensitivity.qd[time_index, joint])
            qdd0 = float(sensitivity.qdd[time_index, joint])
            rows.extend(
                [
                    (sq, float(limits.q_min[joint]) - q0),
                    (-sq, q0 - float(limits.q_max[joint])),
                    (sqd, -float(limits.qd_max[joint]) - qd0),
                    (-sqd, qd0 - float(limits.qd_max[joint])),
                    (sqdd, -float(limits.qdd_max[joint]) - qdd0),
                    (-sqdd, qdd0 - float(limits.qdd_max[joint])),
                ]
            )
    return rows


def solve_local_qp(
    active: list[ActiveDistance],
    sensitivity: LocalSensitivity,
    limits,
    *,
    trust_region: float,
    d_safe: float,
) -> LocalQPResult:
    n = sensitivity.variable_count
    if n == 0 or not active:
        return LocalQPResult(False, np.zeros(n), 0.0, -1, "empty local QP", 0, float("inf"))

    distance_rows: list[tuple[float, np.ndarray, float]] = []
    drive = np.zeros(n, dtype=np.float64)
    for item in active:
        index = int(np.argmin(np.abs(sensitivity.sample_times - item.tau)))
        row = np.einsum("j,jv->v", item.gradient_q, sensitivity.sq[index], optimize=True)
        deficit = max(0.0, float(d_safe) - float(item.distance))
        if np.linalg.norm(row) < 1.0e-12:
            continue
        distance_rows.append((float(item.distance), row.copy(), deficit))
        drive += deficit * row / max(float(np.linalg.norm(row)), 1.0e-12)
    if not distance_rows:
        return LocalQPResult(False, np.zeros(n), 0.0, -2, "no usable active rows", 0, float("inf"))

    norm = float(np.linalg.norm(drive))
    if norm > 1.0e-12:
        x = cfg.FAST_V4_DRIVE_SCALE * float(trust_region) * drive / norm
    else:
        x = np.zeros(n, dtype=np.float64)
    x = np.clip(x, -float(trust_region), float(trust_region))

    halfspaces: list[tuple[np.ndarray, float]] = []
    for distance, row, _ in distance_rows:
        halfspaces.append((row, float(d_safe) - distance))
    halfspaces.extend(_motion_halfspaces(sensitivity, limits))

    changed = 0
    for iteration in range(cfg.FAST_V4_PROJECTION_SWEEPS):
        before = x.copy()
        for row, bound in halfspaces:
            x = _project_halfspace(x, row, bound)
            x = np.clip(x, -float(trust_region), float(trust_region))
        if float(np.linalg.norm(x - before)) < 1.0e-7:
            break
        changed += 1

    predicted = [distance + float(np.dot(row, x)) for distance, row, _ in distance_rows]
    min_predicted = float(np.min(predicted)) if predicted else float("inf")
    violations = [max(0.0, bound - float(np.dot(row, x))) for row, bound in halfspaces]
    max_violation = float(np.max(violations)) if violations else 0.0
    objective = 0.5 * float(np.dot(x, x)) + cfg.FAST_V3_SLACK_WEIGHT * max_violation * max_violation
    return LocalQPResult(
        success=bool(np.all(np.isfinite(x))),
        delta=x,
        objective=objective,
        status=0 if max_violation < 1.0e-5 else 1,
        message=f"projected_qp max_violation={max_violation:.3e}",
        iterations=changed,
        min_predicted_distance=min_predicted,
    )
