from dataclasses import dataclass

import numpy as np


@dataclass
class MockFrame:
    color: np.ndarray
    depth: np.ndarray
    points_cam: np.ndarray
    timestamp: float


class MockRGBDReader:
    def __init__(self, dt: float = 0.1, seed: int = 2):
        self.dt = dt
        self.index = 0
        self.rng = np.random.default_rng(seed)

    def read(self) -> MockFrame:
        timestamp = self.index * self.dt
        self.index += 1
        moving_center = np.array([-0.25 + 0.03 * self.index, 0.25, 0.55])
        static_center = np.array([0.55, -0.2, 0.45])
        points = np.vstack(
            [
                self._sphere_points(moving_center, 0.08, 220),
                self._box_points(static_center, np.array([0.16, 0.12, 0.10]), 180),
                self._table_points(250),
            ]
        )
        color = np.zeros((10, 10, 3), dtype=np.uint8)
        depth = np.ones((10, 10), dtype=np.uint16) * 1000
        return MockFrame(color=color, depth=depth, points_cam=points, timestamp=timestamp)

    def _sphere_points(self, center: np.ndarray, radius: float, count: int) -> np.ndarray:
        directions = self.rng.normal(size=(count, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = radius * np.cbrt(self.rng.random((count, 1)))
        return center + directions * radii

    def _box_points(self, center: np.ndarray, size: np.ndarray, count: int) -> np.ndarray:
        return center + (self.rng.random((count, 3)) - 0.5) * size

    def _table_points(self, count: int) -> np.ndarray:
        xy = self.rng.uniform(-0.8, 0.8, size=(count, 2))
        z = np.zeros((count, 1))
        return np.hstack([xy, z])
