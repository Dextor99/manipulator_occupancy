from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ShapeModel:
    kind: str
    center: np.ndarray
    radius: float | None = None
    extents: dict[str, Any] = field(default_factory=dict)
    rotation: np.ndarray | None = None


@dataclass
class OccupancyObject:
    id: int
    center: np.ndarray
    velocity: np.ndarray
    radius: float
    shape: ShapeModel
    confidence: float
    risk: str
    point_count: int
    age: int
    timestamp: float
