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
    local_point: np.ndarray | None = None
    world_point: np.ndarray | None = None


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


def extract_dense_nearest_distances(
    evaluator,
    trajectory,
    forecast,
    *,
    sample_times: np.ndarray,
    top_k: int,
    links: set[str] | None = None,
) -> list[ActiveDistance]:
    """Extract active constraints from dense Mesh nearest surface vertices."""
    model = evaluator.surface_model
    # Collect exact nearest-vertex candidates first.  Jacobians are needed only
    # by the final top-k constraints; computing one for every time/link/sphere
    # candidate dominated the real-system Fast budget and discarded nearly all
    # of that work after sorting.
    rows = []
    for tau in np.asarray(sample_times, dtype=np.float64):
        q = trajectory.evaluate(float(tau))
        occupancy = forecast.occupancy_at(float(tau))
        if not occupancy.spheres:
            continue
        fk = model.urdf.link_transforms(model._joint_dict(q))
        selected = set(model.link_names) if links is None else set(links)
        for link in model.link_names:
            if link not in selected:
                continue
            transform = fk.get(link)
            if transform is None:
                continue
            local_points = model.local_samples(link, density="dense")
            world_points = local_points @ transform[:3, :3].T + transform[:3, 3]
            for sphere in occupancy.spheres:
                vectors = world_points - sphere.center[None, :]
                radial = np.linalg.norm(vectors, axis=1)
                point_index = int(np.argmin(radial - sphere.radius))
                norm = float(radial[point_index])
                distance = float(norm - sphere.radius)
                rows.append(
                    (
                        distance,
                        float(tau),
                        q,
                        link,
                        local_points[point_index].copy(),
                        world_points[point_index].copy(),
                        vectors[point_index].copy(),
                        norm,
                    )
                )
    rows.sort(key=lambda item: item[0])
    selected: list[ActiveDistance] = []
    target = max(1, int(top_k))
    for distance, tau, q, link, local_point, world_point, vector, norm in rows:
        direction = np.array([1.0, 0.0, 0.0]) if norm < 1.0e-12 else vector / norm
        jac_point = model.point_jacobian(q, link, local_point)
        gradient = direction @ jac_point
        if np.linalg.norm(gradient) < 1.0e-12 or not np.all(np.isfinite(gradient)):
            continue
        selected.append(
            ActiveDistance(
                tau=tau,
                q=q,
                distance=distance,
                gradient_q=np.asarray(gradient, dtype=np.float64),
                nearest_link=link,
                local_point=local_point,
                world_point=world_point,
            )
        )
        if len(selected) >= target:
            break
    return selected
