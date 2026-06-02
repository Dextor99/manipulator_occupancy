from dataclasses import dataclass
from enum import Enum
import math


class RiskLevel(Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    SLOW = "SLOW"
    STOP = "STOP"


@dataclass
class SafetyDecision:
    level: RiskLevel
    min_distance: float
    speed_scale: float
    nearest_object_id: int | None = None


class SafetyPolicy:
    def __init__(self, d_safe: float = 0.15, d_slow: float = 0.10, d_stop: float = 0.05):
        self.d_safe = d_safe
        self.d_slow = d_slow
        self.d_stop = d_stop

    def evaluate(self, distance: float, nearest_object_id: int | None = None) -> SafetyDecision:
        if math.isinf(distance):
            return SafetyDecision(RiskLevel.SAFE, distance, 1.0, nearest_object_id)
        if distance <= self.d_stop:
            level = RiskLevel.STOP
        elif distance <= self.d_slow:
            level = RiskLevel.SLOW
        elif distance <= self.d_safe:
            level = RiskLevel.WARNING
        else:
            level = RiskLevel.SAFE
        scale = max(0.0, min(1.0, (distance - self.d_stop) / (self.d_safe - self.d_stop)))
        if level == RiskLevel.STOP:
            scale = 0.0
        return SafetyDecision(level, float(distance), float(scale), nearest_object_id)
