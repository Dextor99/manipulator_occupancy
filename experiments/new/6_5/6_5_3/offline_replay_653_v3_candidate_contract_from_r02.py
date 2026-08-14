#!/usr/bin/env python3
"""Reclassify the archived V3 r02 bypass seed under the revised contract.

This deterministic replay does not optimize, open the camera, connect to the
robot, or alter the archived trajectory.  It runs the saved candidate through
the V3 execution forecast and absolute verifier, then applies the same
candidate acceptance contract used by the live core.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

trial = importlib.import_module(
    "experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial"
)
v3 = importlib.import_module("experiments.new.6_5.6_5_3.dynamic_nubs_v3")

DEFAULT_SOURCE = (
    ROOT
    / "results/new/6_5/6_5_3/dynamic_nubs_closed_loop_v3_shadow/r02"
    / "core_live/trials/D2_opposing_approach_r02"
)
DEFAULT_OUTPUT = ROOT / "results/new/6_5/6_5_3/v3_candidate_contract_r02_replay"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trial-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def run(args: argparse.Namespace) -> dict:
    source = args.source_trial_dir.resolve()
    core = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    archived = json.loads(
        (source / "candidate/candidate_summary.json").read_text(encoding="utf-8")
    )
    geometry = archived["multisphere_geometry"]
    centers = np.asarray(geometry["component_centers"], dtype=np.float64)
    radii = np.asarray(geometry["component_base_radii"], dtype=np.float64)
    velocity = np.asarray(archived["obstacle_velocity"], dtype=np.float64)
    object_id = int(archived.get("obstacle_association", {}).get("track_id") or 1)

    candidate = trial.reconstruct_saved_nubs_candidate(
        source / "candidate/fast_ccro_nubs_candidate.csv",
        segments=int(core["parameters"]["local_segments"]),
    )
    config = trial.load_stage4_config(Path(core["parameters"]["stage4_config"]))
    model = trial.load_stage4_surface_model(config)
    _, verifier, _ = trial.make_risk_stack(config, model, None)
    forecast = v3.v3_execution_multisphere_forecast(
        centers, radii, velocity, object_id=object_id
    )
    verify_started = time.perf_counter()
    verification = verifier.verify(
        candidate,
        forecast,
        current_q=np.asarray(archived["q_now"], dtype=np.float64),
        current_qd=np.asarray(archived["qd_now"], dtype=np.float64),
        current_qdd=np.zeros(6, dtype=np.float64),
        q_goal=np.asarray(archived["q_goal"], dtype=np.float64),
        solver_success=True,
    )
    verifier_ms = (time.perf_counter() - verify_started) * 1000.0
    replay_pipeline_ms = float(archived["repair_elapsed_ms"]) + verifier_ms

    parameters = core["parameters"]
    reference = trial.RecordedReference.load(
        Path(parameters["reference_feedback_csv"])
    )
    reference.index = int(core["reference_alignment"]["final_reference_index"])
    original_goal = reference.state_after(float(parameters["local_horizon_s"]))
    original_args = SimpleNamespace(
        local_horizon_s=float(parameters["local_horizon_s"]),
        local_segments=int(parameters["local_segments"]),
        min_local_motion_rad=float(parameters["min_local_motion_rad"]),
    )
    head, tail, durations, inner, _ = trial.make_local_reference(
        np.asarray(archived["q_now"], dtype=np.float64),
        np.asarray(archived["qd_now"], dtype=np.float64),
        original_args,
        reference_goal=original_goal,
    )
    original_reference = trial.NUBSTrajectory6D().generate(
        inner, head, tail, durations
    )
    candidate_samples = candidate.dense_sample(0.02).q
    original_samples = original_reference.dense_sample(0.02).q
    delta_from_original = float(
        np.max(np.abs(candidate_samples - original_samples))
    )

    hard_safety_ready = bool(
        replay_pipeline_ms <= float(archived["fast_budget_ms"])
        and not bool(archived["budget_exhausted"])
        and verification.min_distance >= float(archived["online_accept_m"])
        and all(verification.checks.values())
    )
    contract = trial.candidate_acceptance_contract(
        hard_safety_ready=hard_safety_ready,
        repair_step_ok=bool(int(archived["accepted_steps"]) > 0),
        clearance_gain_m=float(archived["clearance_improvement_m"]),
        minimum_clearance_gain_m=float(parameters["min_clearance_improvement_m"]),
        delta_from_fast_seed_rad=float(archived["max_delta_q_from_reference_rad"]),
        minimum_candidate_delta_rad=float(parameters["min_candidate_delta_q_rad"]),
        accept_verified_seed_without_fast_step=True,
    )
    result = {
        "status": (
            "V3_R02_CANDIDATE_CONTRACT_REPLAY_PASS"
            if contract["local_repair_ready"]
            else "V3_R02_CANDIDATE_CONTRACT_REPLAY_FAIL"
        ),
        "robot_commanded": False,
        "camera_opened": False,
        "trajectory_reoptimized": False,
        "trajectory_changed": False,
        "source_trial_dir": str(source),
        "seed_verifier_min_distance_m": float(verification.min_distance),
        "seed_verifier_accepted": bool(verification.accepted),
        "seed_verifier_checks": dict(verification.checks),
        "seed_verifier_reasons": list(verification.reasons),
        "replay_verifier_ms": verifier_ms,
        "archived_repair_ms": float(archived["repair_elapsed_ms"]),
        "replay_online_pipeline_ms": replay_pipeline_ms,
        "fast_budget_ms": float(archived["fast_budget_ms"]),
        "fast_accepted_steps": int(archived["accepted_steps"]),
        "fast_extra_correction_applied": bool(
            contract["fast_extra_correction_applied"]
        ),
        "candidate_source": contract["candidate_source"],
        "local_repair_ready": bool(contract["local_repair_ready"]),
        "execution_authorization_status": (
            "PENDING_POST_PLAN_FRESH_RECHECK"
            if contract["local_repair_ready"]
            else "NOT_ELIGIBLE"
        ),
        "delta_candidate_from_bypass_seed_max_abs_rad": float(
            archived["max_delta_q_from_reference_rad"]
        ),
        "delta_candidate_from_original_reference_max_abs_rad": delta_from_original,
        "candidate_equals_fast_seed": bool(
            float(archived["max_delta_q_from_reference_rad"])
            < float(parameters["min_candidate_delta_q_rad"])
        ),
        "candidate_is_original_reference": bool(
            delta_from_original < float(parameters["min_candidate_delta_q_rad"])
        ),
        "optimizer_diagnostics": contract["optimizer_diagnostics"],
        "forecast_builder": "v3_execution_multisphere_forecast",
        "forecast_component_radii_at_0s_m": [
            float(s.radius) for s in forecast.occupancy_at(0.0).spheres
        ],
        "forecast_component_radii_at_1s_m": [
            float(s.radius) for s in forecast.occupancy_at(1.0).spheres
        ],
    }
    trial.write_json(args.output.resolve() / "summary.json", result)
    return result


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
