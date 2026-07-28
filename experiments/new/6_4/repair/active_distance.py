"""Active distance extraction for fast 6.4 local repair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config_64 as cfg


@dataclass(frozen=True)
class ActiveDistance:
    tau: float
    q: np.ndarray
    distance: float
    gradient_q: np.ndarray
    nearest_link: str | None


def distance_gradient(evaluator, q: np.ndarray, forecast, tau: float, *, density: str) -> np.ndarray:
    eps = cfg.FAST_DISTANCE_GRAD_EPS
    values = np.asarray(q, dtype=np.float64)
    gradient = np.zeros(6, dtype=np.float64)
    for joint in range(6):
        plus = values.copy()
        minus = values.copy()
        plus[joint] += eps
        minus[joint] -= eps
        d_plus = evaluator.configuration(plus, forecast, float(tau), density=density, with_gradient=False).min_distance
        d_minus = evaluator.configuration(minus, forecast, float(tau), density=density, with_gradient=False).min_distance
        gradient[joint] = (float(d_plus) - float(d_minus)) / (2.0 * eps)
    return gradient


def extract_active_distances(
    evaluator,
    trajectory,
    forecast,
    *,
    sample_times: np.ndarray,
    top_k: int,
    density: str,
) -> list[ActiveDistance]:
    rows = []
    for tau in np.asarray(sample_times, dtype=np.float64):
        q = trajectory.evaluate(float(tau))
        risk = evaluator.configuration(q, forecast, float(tau), density=density, with_gradient=False)
        if np.isfinite(risk.min_distance) and risk.min_distance < cfg.D_ONLINE_ACCEPT:
            rows.append((float(risk.min_distance), float(tau), q, risk.nearest_link))
    rows.sort(key=lambda item: item[0])
    active: list[ActiveDistance] = []
    for distance, tau, q, nearest_link in rows[: max(1, int(top_k))]:
        gradient = distance_gradient(evaluator, q, forecast, tau, density=density)
        if np.linalg.norm(gradient) < 1.0e-10 or not np.all(np.isfinite(gradient)):
            continue
        active.append(
            ActiveDistance(
                tau=tau,
                q=q,
                distance=distance,
                gradient_q=gradient,
                nearest_link=nearest_link,
            )
        )
    return active
