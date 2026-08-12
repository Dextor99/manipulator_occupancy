#!/usr/bin/env python3
"""Replay one frozen real trigger at Hlocal=1.00/1.25/1.50 s.

This diagnostic never opens the camera or robot connection.  It changes only
the local repair horizon and reuses the saved stopped state, Fresh #1 motion,
multisphere geometry, reference and production Fast repair implementation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

trial = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")
DEFAULT_SOURCE = ROOT / "results/new/6_5/6_5_3/d2_xp00_baseline/trials/D2_opposing_approach_r01"
DEFAULT_REFERENCE = ROOT / "results/new/6_5/6_5_3/reference_xp00_line/reference_feedback.csv"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-trial", type=Path, default=DEFAULT_SOURCE)
    result.add_argument("--reference-feedback-csv", type=Path, default=DEFAULT_REFERENCE)
    result.add_argument("--horizons-s", default="1.0,1.25,1.5")
    result.add_argument("--output", type=Path, default=ROOT / "results/new/6_5/6_5_3/d2_xp00_horizon_check")
    return result


def trigger_reference_index(source: Path) -> int:
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    trigger_frame = int(summary["trigger_frame"])
    with (source / "frames.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["frame"]) == trigger_frame:
                return int(row["reference_index"])
    raise RuntimeError(f"trigger frame {trigger_frame} is absent from frames.csv")


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_trial.resolve()
    candidate_path = source / "candidate/candidate_summary.json"
    geometry_path = source / "fresh_multisphere.json"
    for path in (candidate_path, geometry_path, source / "summary.json", source / "frames.csv"):
        if not path.is_file():
            raise FileNotFoundError(path)
    horizons = [float(value) for value in args.horizons_s.split(",")]
    if horizons != [1.0, 1.25, 1.5]:
        raise ValueError("the frozen diagnostic requires --horizons-s 1.0,1.25,1.5")

    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    reference = trial.RecordedReference.load(args.reference_feedback_csv.resolve())
    reference.index = trigger_reference_index(source)
    runtime_args = trial.build_parser().parse_args(["--scene", "D2", "--mode", "shadow"])
    config = trial.load_stage4_config(runtime_args.stage4_config)
    model = trial.load_stage4_surface_model(config)
    q_now = np.asarray(candidate["q_now"], dtype=np.float64)
    qd_now = np.asarray(candidate["qd_now"], dtype=np.float64)
    center = np.asarray(candidate["obstacle_center"], dtype=np.float64)
    velocity = np.asarray(candidate["obstacle_velocity"], dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        runtime_args.local_horizon_s = horizon
        with tempfile.TemporaryDirectory(prefix="ccro653_horizon_") as temp_dir:
            result = trial.run_fast_repair(
                runtime_args,
                config,
                model,
                q_now=q_now,
                qd_now=qd_now,
                center=center,
                velocity=velocity,
                radius=float(candidate["obstacle_radius"]),
                risk_links=set(model.surface_by_link(q_now, density="coarse")),
                trial_dir=Path(temp_dir),
                reference_goal=reference.state_after(horizon),
                rejoin_goals=[],
                obstacle_audit={"offline_horizon_check": True, "local_horizon_s": horizon},
                multisphere_geometry=geometry,
            )
        checks = result["verification_checks"]
        rows.append(
            {
                "local_horizon_s": horizon,
                "accepted_steps": int(result["accepted_steps"]),
                "local_repair_ready": bool(result["local_repair_ready"]),
                "reference_clearance_m": float(result["reference_online_min_distance_m"]),
                "candidate_clearance_m": float(result["candidate_online_min_distance_m"]),
                "clearance_improvement_m": float(result["clearance_improvement_m"]),
                "position_ok": bool(checks["position_ok"]),
                "velocity_ok": bool(checks["velocity_ok"]),
                "acceleration_ok": bool(checks["acceleration_ok"]),
                "fast_ms": float(result["online_pipeline_elapsed_ms"]),
                "max_delta_q_rad": float(result["max_delta_q_from_reference_rad"]),
                "rejection_reasons": ";".join(result["rejection_reasons"]),
                "messages": " | ".join(result["messages"]),
            }
        )

    def passes(row: dict[str, Any]) -> bool:
        return bool(
            row["accepted_steps"] > 0
            and row["candidate_clearance_m"] >= runtime_args.online_accept_m
            and row["clearance_improvement_m"] >= runtime_args.min_clearance_improvement_m
            and row["velocity_ok"]
            and row["acceleration_ok"]
            and row["fast_ms"] <= runtime_args.fast_budget_ms
        )

    selected = next((row for row in rows if row["local_horizon_s"] >= 1.25 and passes(row)), None)
    status = "FREEZE_1P25" if selected and selected["local_horizon_s"] == 1.25 else (
        "FREEZE_1P50" if selected else "HORIZON_INSUFFICIENT"
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trial.write_csv(output / "horizon_check.csv", rows, list(rows[0]))
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "robot_commanded": False,
        "source_trial": str(source),
        "reference_feedback_csv": str(args.reference_feedback_csv.resolve()),
        "source_git_commit": json.loads((source / "summary.json").read_text(encoding="utf-8")).get("git_commit"),
        "analysis_git_commit": trial.git_commit_hash(),
        "frozen_horizons_s": horizons,
        "selection_rule": "prefer 1.25 s if it passes; otherwise 1.50 s; stop if neither passes",
        "selected_horizon_s": None if selected is None else selected["local_horizon_s"],
        "rows": rows,
    }
    trial.write_json(output / "horizon_check.json", payload)
    return payload


def main() -> None:
    print(json.dumps(run(parser().parse_args()), indent=2, ensure_ascii=False, default=trial.json_default))


if __name__ == "__main__":
    main()
