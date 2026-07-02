"""Shared setup utilities for Chapter 6 E1-E5 experiments."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "ch6_e1_e5_unified.yaml"

LOG_FILE_KEYS = {
    "trajectory.csv": "trajectory_csv",
    "obstacle_log.csv": "obstacle_log_csv",
    "risk_log.csv": "risk_log_csv",
    "planner_log.csv": "planner_log_csv",
    "runtime_log.csv": "runtime_log_csv",
}


def load_unified_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the frozen E1-E5 configuration."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(config_path)
    return config


def resolve_output_dir(
    experiment_id: str,
    output: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    """Resolve an experiment output directory under the configured result root."""
    cfg = config or load_unified_config()
    if output is not None:
        out = Path(output)
    else:
        out = Path(cfg["experiment"]["result_root"]) / experiment_id
    if not out.is_absolute():
        out = ROOT / out
    return out


def _write_csv_header(path: Path, columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)


def prepare_run_directory(
    experiment_id: str,
    *,
    output: str | Path | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    scene_config: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Create the standard run directory and seed required log files."""
    cfg = load_unified_config(config_path)
    out = resolve_output_dir(experiment_id, output, cfg)
    if out.exists() and overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    source = Path(cfg["_config_path"])
    shutil.copy2(source, out / "config.yaml")

    scene_payload = scene_config or {
        "experiment_id": experiment_id,
        "description": "Fill with scene, obstacle, start, goal, and trial settings before running.",
        "scenarios": [],
    }
    (out / "scene_config.yaml").write_text(
        yaml.safe_dump(scene_payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    for directory in cfg["experiment"].get("directories", []):
        (out / directory).mkdir(parents=True, exist_ok=True)

    for filename, key in LOG_FILE_KEYS.items():
        columns = list(cfg["logging"][key]["columns"])
        _write_csv_header(out / filename, columns)

    manifest = {
        "experiment_id": experiment_id,
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
        "config_source": str(source.relative_to(ROOT)),
        "required_files": cfg["experiment"]["required_files"],
        "directories": cfg["experiment"].get("directories", []),
        "status": "prepared",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def append_runtime_row(path: str | Path, trial_id: str, frame: int, module: str, elapsed_ms: float) -> None:
    """Append one module timing row to runtime_log.csv."""
    with Path(path).open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([trial_id, int(frame), module, float(elapsed_ms)])
