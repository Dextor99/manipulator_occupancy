import json
from pathlib import Path
from typing import Any

import yaml


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def load_config_dir(config_dir: str | Path) -> dict[str, Any]:
    root = Path(config_dir)
    return {
        "intrinsic": load_json(root / "camera_intrinsic.json"),
        "extrinsic": load_json(root / "camera_extrinsic.json"),
        "workspace": load_yaml(root / "workspace.yaml"),
        "capsules": load_yaml(root / "robot_capsules.yaml"),
        "safety": load_yaml(root / "safety.yaml"),
        "robot_model": load_yaml(root / "robot_model.yaml"),
    }
