from collections import deque

import numpy as np


def cluster_points(points: np.ndarray, eps: float = 0.05, min_points: int = 30) -> list[np.ndarray]:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return []
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        return _clusters_from_labels(points, labels)
    except Exception:
        return _simple_dbscan(points, eps=eps, min_points=min_points)


def _clusters_from_labels(points: np.ndarray, labels: np.ndarray) -> list[np.ndarray]:
    clusters = []
    for label in sorted(set(labels.tolist())):
        if label == -1:
            continue
        clusters.append(points[labels == label])
    return clusters


def _simple_dbscan(points: np.ndarray, eps: float, min_points: int) -> list[np.ndarray]:
    labels = np.full(points.shape[0], -2, dtype=int)
    cluster_id = 0
    for i in range(points.shape[0]):
        if labels[i] != -2:
            continue
        neighbors = _neighbors(points, i, eps)
        if len(neighbors) < min_points:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        queue = deque(neighbors)
        while queue:
            j = queue.popleft()
            if labels[j] == -1:
                labels[j] = cluster_id
            if labels[j] != -2:
                continue
            labels[j] = cluster_id
            j_neighbors = _neighbors(points, j, eps)
            if len(j_neighbors) >= min_points:
                queue.extend(j_neighbors)
        cluster_id += 1
    return _clusters_from_labels(points, labels)


def _neighbors(points: np.ndarray, index: int, eps: float) -> list[int]:
    distances = np.linalg.norm(points - points[index], axis=1)
    return np.flatnonzero(distances <= eps).tolist()
