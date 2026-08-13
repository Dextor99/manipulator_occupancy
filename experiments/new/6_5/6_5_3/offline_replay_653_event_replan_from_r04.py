#!/usr/bin/env python3
"""Pure offline continuation replay from the archived successful r04 tail.

This tool never opens RealSense, connects to AUBO, or authorizes execution.  It
uses the measured local-#1 endpoint and archived Fresh #3 fixed two-sphere
geometry.  If stationary-tail risk is below 0.14 m, it invokes the exact r06
three-bypass/coarse-gate/Fast planning stack once.  A feasible result explicitly
requires a future live Fresh #4 before it can be called authorized.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

trial = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")
event = importlib.import_module(
    "experiments.new.6_5.6_5_3.run_653_simple_dynamic_nubs_event_replan_live"
)

DEFAULT_SOURCE = ROOT / "results/new/6_5/6_5_3/simple_dynamic_nubs_live/r04"
DEFAULT_OUTPUT = ROOT / "results/new/6_5/6_5_3/offline_event_replan_r04_goal_directed"
DEFAULT_REFERENCE = ROOT / "results/new/6_5/6_5_3/reference_xp10_line/reference_feedback.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-r04", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-feedback-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--forward-m", type=float, default=0.05)
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.12)
    parser.add_argument("--planning-robust-target-m", type=float, default=0.11)
    parser.add_argument("--continuation-side-m", type=float, default=0.04)
    parser.add_argument("--tcp-link", default="gripper_base_link")
    return parser


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive(source: Path) -> dict[str, Any]:
    manifest = read_json(source / "provenance_manifest.json")
    checks = {}
    for relative, expected in manifest["sha256"].items():
        path = source / relative
        actual = sha256(path)
        checks[relative] = {"expected": expected, "actual": actual, "match": actual == expected}
    if not all(item["match"] for item in checks.values()):
        raise RuntimeError("r04 provenance hash mismatch; refusing offline replay")
    return {"accepted": True, "checks": checks, "manifest": manifest}


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_r04.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = validate_archive(source)
    trial_dir = source / "core_live/trials/D2_opposing_approach_r04"
    execution = read_json(trial_dir / "candidate/live_local_candidate_execution_log.json")
    fresh3_payload = read_json(trial_dir / "fresh3_recheck.json")
    fresh3 = fresh3_payload["result"]
    frames3 = fresh3_payload["frames"]
    geometry3 = read_json(trial_dir / "fresh3_multisphere.json")
    q0 = np.asarray(execution["actual_start_joint_rad"], dtype=np.float64)
    q1 = np.asarray(execution["goal_check"]["actual_joint_rad"], dtype=np.float64)

    runtime_args = trial.build_parser().parse_args(["--scene", "D2", "--mode", "shadow"])
    config = trial.load_stage4_config(runtime_args.stage4_config)
    model = trial.load_stage4_surface_model(config)
    reference = trial.RecordedReference.load(args.reference_feedback_csv.resolve())
    forecast, basis = event.forecast_from_fresh(runtime_args, fresh3, geometry3, frames3)
    if forecast is None:
        raise RuntimeError("archived Fresh #3 cannot construct a conservative forecast")
    hold = event.stationary_hold_audit(runtime_args, config, model, q1, forecast)
    result: dict[str, Any] = {
        "status": "INITIALIZED",
        "robot_commanded": False,
        "camera_opened": False,
        "authorization_claimed": False,
        "source_r04": str(source),
        "archive_validation": archive,
        "q1_source": "measured_goal_check_actual_joint_rad",
        "q1_actual_rad": q1.tolist(),
        "fresh3_forecast_basis": basis,
        "fresh3_speed_m_s": float(fresh3["speed_m_s"]),
        "fresh3_scene_clear": event.strict_empty_scene(runtime_args, frames3),
        "stationary_hold_audit": hold,
    }
    if hold["physical_hold_safe"]:
        result["status"] = "PHYSICAL_HOLD_SAFE_WAIT_FOR_SCENE_CLEAR"
        trial.write_json(output / "summary.json", result)
        return result

    if args.planning_robust_target_m < 0.11:
        raise ValueError("planning-robust-target-m must remain at least 0.11 m")
    local2_goal, goal_audit = event.next_recorded_reference_goal(
        reference, q1, runtime_args.local_horizon_s
    )
    result["local2_reference_goal"] = goal_audit
    local2_dir = output / "local2_plan"
    artifacts: dict[str, Any] = {}
    candidate = event.plan_goal_directed_continuation(
        trial.run_fast_repair,
        runtime_args,
        config,
        model,
        q_escape_start=q0,
        q_now=q1,
        q_final=np.asarray(reference.q[-1], dtype=np.float64),
        fresh=fresh3,
        geometry=geometry3,
        risk_links=set(model.surface_by_link(q1, density="coarse")),
        trial_dir=local2_dir,
        nominal_reference_goal=local2_goal,
        artifacts_out=artifacts,
        forward_m=float(args.forward_m),
        side_m=float(args.continuation_side_m),
        robust_target_m=float(args.planning_robust_target_m),
        max_joint_delta_rad=float(args.max_joint_delta_rad),
        tcp_link=args.tcp_link,
    )
    result["local2_candidate"] = candidate
    result["local2_fresh4_available"] = False
    result["authorization_claimed"] = False
    if (
        candidate.get("local_repair_ready", False)
        and int(candidate.get("accepted_steps", 0)) > 0
        and float(candidate.get("verification_min_distance_m", -np.inf))
        >= runtime_args.online_accept_m
        and all(candidate.get("verification_checks", {}).values())
    ):
        result["status"] = "LOCAL2_PLAN_FEASIBLE_REQUIRES_LIVE_FRESH4"
    else:
        result["status"] = "LOCAL2_PLAN_NOT_FEASIBLE_OFFLINE_NO_LIVE_RECOMMENDED"
    trial.write_json(output / "summary.json", result)
    return result


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
