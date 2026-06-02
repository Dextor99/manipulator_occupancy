from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Capsule:
    name: str
    a: np.ndarray
    b: np.ndarray
    radius: float


def mock_capsules() -> list[Capsule]:
    return [
        Capsule("base_link", np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.25]), 0.08),
        Capsule("upper_arm", np.array([0.0, 0.0, 0.25]), np.array([0.25, 0.0, 0.55]), 0.075),
        Capsule("forearm", np.array([0.25, 0.0, 0.55]), np.array([0.55, 0.0, 0.55]), 0.065),
        Capsule("wrist", np.array([0.55, 0.0, 0.55]), np.array([0.70, 0.0, 0.45]), 0.055),
    ]


def capsules_from_config(config: dict, link_points: dict[str, np.ndarray] | None = None) -> list[Capsule]:
    capsules = []
    link_points = link_points or {}
    for item in config.get("robot_capsules", []):
        if item.get("type", "capsule") != "capsule":
            continue
        start = item.get("start")
        end = item.get("end")
        if "a" in item and "b" in item:
            a = np.asarray(item["a"], dtype=float)
            b = np.asarray(item["b"], dtype=float)
        elif start in link_points and end in link_points:
            a = np.asarray(link_points[start], dtype=float)
            b = np.asarray(link_points[end], dtype=float)
        else:
            continue
        capsules.append(Capsule(item["name"], a, b, float(item["radius"])))
    return capsules or mock_capsules()
