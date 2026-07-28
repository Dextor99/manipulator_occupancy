"""Small SLSQP-backed convex local repair step for 6.4 fast experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .. import config_64 as cfg
from .active_distance import ActiveDistance
from .nubs_linearization import LocalSensitivity


@dataclass(frozen=True)
class LocalQPResult:
    success: bool
    delta: np.ndarray
    objective: float
    status: int
    message: str
    iterations: int
    min_predicted_distance: float


def solve_local_qp(
    active: list[ActiveDistance],
    sensitivity: LocalSensitivity,
    limits,
    *,
    trust_region: float,
    d_safe: float,
) -> LocalQPResult:
    n_delta = sensitivity.variable_count
    m = len(active)
    n = n_delta + m
    if n_delta == 0 or not active:
        return LocalQPResult(False, np.zeros(n_delta), 0.0, -1, "empty local QP", 0, float("inf"))

    def objective(delta: np.ndarray) -> float:
        dp = np.asarray(delta[:n_delta], dtype=np.float64)
        slack = np.asarray(delta[n_delta:], dtype=np.float64)
        drive = 0.0
        for deficit, row in clearance_drive:
            drive += deficit * float(np.dot(row, dp))
        return 0.5 * float(np.dot(dp, dp)) + cfg.FAST_V3_SLACK_WEIGHT * float(np.dot(slack, slack)) - cfg.FAST_V3_CLEARANCE_GAIN * drive

    def gradient(delta: np.ndarray) -> np.ndarray:
        grad = np.zeros(n, dtype=np.float64)
        grad[:n_delta] = delta[:n_delta]
        for deficit, row in clearance_drive:
            grad[:n_delta] -= cfg.FAST_V3_CLEARANCE_GAIN * deficit * row
        grad[n_delta:] = 2.0 * cfg.FAST_V3_SLACK_WEIGHT * delta[n_delta:]
        return grad

    constraints = []
    predicted_rows: list[tuple[float, np.ndarray]] = []
    clearance_drive: list[tuple[float, np.ndarray]] = []
    for active_index, item in enumerate(active):
        index = int(np.argmin(np.abs(sensitivity.sample_times - item.tau)))
        row = np.einsum("j,jv->v", item.gradient_q, sensitivity.sq[index], optimize=True)
        predicted_rows.append((float(item.distance), row.copy()))
        clearance_drive.append((max(0.0, d_safe - float(item.distance)), row.copy()))
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda delta, distance=float(item.distance), row=row, idx=active_index: distance + float(np.dot(row, delta[:n_delta])) + float(delta[n_delta + idx]) - d_safe,
                "jac": lambda delta, row=row, idx=active_index: np.r_[row, np.eye(m)[idx]],
            }
        )

    for time_index in range(len(sensitivity.sample_times)):
        for joint in range(6):
            sq = sensitivity.sq[time_index, joint]
            sqd = sensitivity.sqd[time_index, joint]
            sqdd = sensitivity.sqdd[time_index, joint]
            q0 = float(sensitivity.q[time_index, joint])
            qd0 = float(sensitivity.qd[time_index, joint])
            qdd0 = float(sensitivity.qdd[time_index, joint])
            constraints.extend(
                [
                    {"type": "ineq", "fun": lambda d, row=sq, base=q0, lo=float(limits.q_min[joint]): base + float(np.dot(row, d[:n_delta])) - lo, "jac": lambda d, row=sq: np.r_[row, np.zeros(m)]},
                    {"type": "ineq", "fun": lambda d, row=sq, base=q0, hi=float(limits.q_max[joint]): hi - base - float(np.dot(row, d[:n_delta])), "jac": lambda d, row=sq: np.r_[-row, np.zeros(m)]},
                    {"type": "ineq", "fun": lambda d, row=sqd, base=qd0, hi=float(limits.qd_max[joint]): hi - base - float(np.dot(row, d[:n_delta])), "jac": lambda d, row=sqd: np.r_[-row, np.zeros(m)]},
                    {"type": "ineq", "fun": lambda d, row=sqd, base=qd0, hi=float(limits.qd_max[joint]): hi + base + float(np.dot(row, d[:n_delta])), "jac": lambda d, row=sqd: np.r_[row, np.zeros(m)]},
                    {"type": "ineq", "fun": lambda d, row=sqdd, base=qdd0, hi=float(limits.qdd_max[joint]): hi - base - float(np.dot(row, d[:n_delta])), "jac": lambda d, row=sqdd: np.r_[-row, np.zeros(m)]},
                    {"type": "ineq", "fun": lambda d, row=sqdd, base=qdd0, hi=float(limits.qdd_max[joint]): hi + base + float(np.dot(row, d[:n_delta])), "jac": lambda d, row=sqdd: np.r_[row, np.zeros(m)]},
                ]
            )

    bounds = [(-float(trust_region), float(trust_region)) for _ in range(n_delta)]
    bounds += [(0.0, float(d_safe)) for _ in range(m)]
    result = minimize(
        objective,
        np.zeros(n, dtype=np.float64),
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 40, "ftol": 1.0e-8, "disp": False},
    )
    full = np.asarray(result.x, dtype=np.float64)
    delta = full[:n_delta]
    predicted = [distance + float(np.dot(row, delta)) for distance, row in predicted_rows]
    return LocalQPResult(
        success=bool(result.success) and np.all(np.isfinite(delta)),
        delta=delta,
        objective=float(result.fun) if np.isfinite(result.fun) else float("inf"),
        status=int(result.status),
        message=str(result.message),
        iterations=int(result.nit),
        min_predicted_distance=float(np.min(predicted)) if predicted else float("inf"),
    )
