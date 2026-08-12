#!/usr/bin/env python3
"""One frozen linear-vs-lateral Fast warm-start replay at Hlocal=1.0 s."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
trial = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")


def reference_index(source: Path) -> int:
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    with (source / "frames.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["frame"]) == int(summary["trigger_frame"]):
                return int(row["reference_index"])
    raise RuntimeError("trigger frame is absent from frames.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trial", type=Path, default=ROOT / "results/new/6_5/6_5_3/d2_xp00_baseline/trials/D2_opposing_approach_r01")
    parser.add_argument("--reference-feedback-csv", type=Path, default=ROOT / "results/new/6_5/6_5_3/reference_xp00_line/reference_feedback.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/new/6_5/6_5_3/d2_xp00_lateral_warm_start_ab")
    args = parser.parse_args()
    source = args.source_trial.resolve()
    candidate = json.loads((source / "candidate/candidate_summary.json").read_text(encoding="utf-8"))
    geometry = json.loads((source / "fresh_multisphere.json").read_text(encoding="utf-8"))
    reference = trial.RecordedReference.load(args.reference_feedback_csv.resolve())
    reference.index = reference_index(source)
    runtime = trial.build_parser().parse_args(["--scene", "D2", "--mode", "shadow"])
    runtime.local_horizon_s = 1.0
    config = trial.load_stage4_config(runtime.stage4_config)
    model = trial.load_stage4_surface_model(config)
    rows = []
    for mode in ("linear", "lateral"):
        runtime.fast_warm_start = mode
        with tempfile.TemporaryDirectory(prefix="ccro653_warm_ab_") as temp_dir:
            result = trial.run_fast_repair(
                runtime, config, model,
                q_now=np.asarray(candidate["q_now"], dtype=np.float64),
                qd_now=np.asarray(candidate["qd_now"], dtype=np.float64),
                center=np.asarray(candidate["obstacle_center"], dtype=np.float64),
                velocity=np.asarray(candidate["obstacle_velocity"], dtype=np.float64),
                radius=float(candidate["obstacle_radius"]),
                risk_links=set(model.surface_by_link(np.asarray(candidate["q_now"]), density="coarse")),
                trial_dir=Path(temp_dir),
                reference_goal=reference.state_after(1.0),
                rejoin_goals=[],
                obstacle_audit={"offline_warm_start_ab": True, "mode": mode},
                multisphere_geometry=geometry,
            )
        checks = result["verification_checks"]
        rows.append({
            "warm_start": mode,
            "accepted_steps": int(result["accepted_steps"]),
            "local_repair_ready": bool(result["local_repair_ready"]),
            "reference_clearance_m": float(result["reference_online_min_distance_m"]),
            "candidate_clearance_m": float(result["candidate_online_min_distance_m"]),
            "clearance_improvement_m": float(result["clearance_improvement_m"]),
            "velocity_ok": bool(checks["velocity_ok"]),
            "acceleration_ok": bool(checks["acceleration_ok"]),
            "fast_ms": float(result["online_pipeline_elapsed_ms"]),
            "max_delta_q_rad": float(result["max_delta_q_from_reference_rad"]),
            "warm_start_audit": json.dumps(result["warm_start"], ensure_ascii=False),
            "rejection_reasons": ";".join(result["rejection_reasons"]),
            "messages": " | ".join(result["messages"]),
        })
    lateral = rows[1]
    passed = bool(lateral["accepted_steps"] > 0 and lateral["candidate_clearance_m"] >= 0.09
                  and lateral["clearance_improvement_m"] >= 0.003 and lateral["velocity_ok"]
                  and lateral["acceleration_ok"] and lateral["fast_ms"] <= 150.0)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trial.write_csv(output / "warm_start_ab.csv", rows, list(rows[0]))
    payload = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "LATERAL_WARM_START_PASS" if passed else "LATERAL_WARM_START_FAIL", "robot_commanded": False, "source_trial": str(source), "local_horizon_s": 1.0, "lateral_offset_m": 0.04, "rows": rows}
    trial.write_json(output / "warm_start_ab.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
