from pathlib import Path

import numpy as np

from robot.capsule_model import mock_capsules


class URDFFKModel:
    def __init__(self, urdf_path: str | None):
        self.urdf_path = Path(urdf_path) if urdf_path else None
        self.robot = None
        if self.urdf_path and self.urdf_path.exists():
            try:
                from urdfpy import URDF

                self.robot = URDF.load(str(self.urdf_path))
            except Exception as exc:
                raise RuntimeError(f"failed to load URDF: {self.urdf_path}") from exc

    @property
    def is_mock(self) -> bool:
        return self.robot is None

    def link_transforms(self, joint_positions: dict[str, float] | None = None) -> dict[str, np.ndarray]:
        if self.robot is None:
            transforms = {}
            for capsule in mock_capsules():
                transforms[f"{capsule.name}_start"] = _pose_from_translation(capsule.a)
                transforms[f"{capsule.name}_end"] = _pose_from_translation(capsule.b)
            return transforms
        return self.robot.link_fk(cfg=joint_positions or {})


def _pose_from_translation(translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = translation
    return pose
