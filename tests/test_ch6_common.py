import csv
import json
from pathlib import Path

from experiments.ch6_common import load_unified_config, prepare_run_directory


def test_load_unified_config_contains_required_blocks():
    cfg = load_unified_config()
    for key in ("platform", "perception", "risk", "nubs", "acceptance", "logging"):
        assert key in cfg
    assert cfg["platform"]["robot"]["urdf_path"] == "urdf/aubo_i16_gripper.urdf"
    assert cfg["logging"]["planner_log_csv"]["columns"][:3] == ["trial_id", "method", "success"]


def test_prepare_run_directory_creates_standard_files(tmp_path: Path):
    out = prepare_run_directory("E0_test", output=tmp_path / "run")
    expected = {
        "config.yaml",
        "scene_config.yaml",
        "trajectory.csv",
        "obstacle_log.csv",
        "risk_log.csv",
        "planner_log.csv",
        "runtime_log.csv",
        "manifest.json",
    }
    assert expected.issubset({path.name for path in out.iterdir()})
    assert (out / "figures").is_dir()
    assert (out / "videos").is_dir()

    with (out / "planner_log.csv").open(encoding="utf-8", newline="") as handle:
        assert next(csv.reader(handle))[:4] == ["trial_id", "method", "success", "accepted"]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "E0_test"
    assert manifest["status"] == "prepared"
