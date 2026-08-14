#!/usr/bin/env python3
"""A/B replay the archived V3 r01 candidate under legacy and V3 execution geometry.

Pure offline diagnostic: no camera, SDK connection or robot authority.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

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
    / "results/new/6_5/6_5_3/dynamic_nubs_closed_loop_v3_shadow/r01"
    / "core_live/trials/D2_opposing_approach_r01"
)
DEFAULT_OUTPUT = ROOT / "results/new/6_5/6_5_3/v3_execution_forecast_r01_ab"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trial-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def verification_payload(verification: object, forecast: object) -> dict:
    radii_0 = [float(s.radius) for s in forecast.occupancy_at(0.0).spheres]
    radii_1 = [float(s.radius) for s in forecast.occupancy_at(1.0).spheres]
    return {
        "candidate_min_distance_m": float(verification.min_distance),
        "distance_ok": bool(verification.checks["distance_ok"]),
        "accepted": bool(verification.accepted),
        "checks": dict(verification.checks),
        "reasons": list(verification.reasons),
        "component_radii_at_0s_m": radii_0,
        "component_radii_at_1s_m": radii_1,
        "max_radius_at_0s_m": max(radii_0),
        "max_radius_at_1s_m": max(radii_1),
    }


def run(args: argparse.Namespace) -> dict:
    source = args.source_trial_dir.resolve()
    core = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    candidate_summary = json.loads(
        (source / "candidate/candidate_summary.json").read_text(encoding="utf-8")
    )
    geometry = candidate_summary["multisphere_geometry"]
    centers = np.asarray(geometry["component_centers"], dtype=np.float64)
    radii = np.asarray(geometry["component_base_radii"], dtype=np.float64)
    velocity = np.asarray(candidate_summary["obstacle_velocity"], dtype=np.float64)
    object_id = int(
        candidate_summary.get("obstacle_association", {}).get("track_id") or 1
    )

    trajectory = trial.reconstruct_saved_nubs_candidate(
        source / "candidate/fast_ccro_nubs_candidate.csv",
        segments=int(core["parameters"]["local_segments"]),
    )
    config = trial.load_stage4_config(Path(core["parameters"]["stage4_config"]))
    model = trial.load_stage4_surface_model(config)
    _, verifier, _ = trial.make_risk_stack(config, model, None)
    verify_kwargs = {
        "current_q": np.asarray(candidate_summary["q_now"], dtype=np.float64),
        "current_qd": np.asarray(candidate_summary["qd_now"], dtype=np.float64),
        "current_qdd": np.zeros(6, dtype=np.float64),
        "q_goal": np.asarray(candidate_summary["q_goal"], dtype=np.float64),
        "solver_success": True,
    }

    legacy_forecast = trial.common64.constant_multisphere_forecast(
        centers, radii, velocity, object_id=object_id
    )
    v3_forecast = v3.v3_execution_multisphere_forecast(
        centers, radii, velocity, object_id=object_id
    )
    legacy_verification = verifier.verify(trajectory, legacy_forecast, **verify_kwargs)
    v3_verification = verifier.verify(trajectory, v3_forecast, **verify_kwargs)
    legacy = verification_payload(legacy_verification, legacy_forecast)
    revised = verification_payload(v3_verification, v3_forecast)
    result = {
        "status": "V3_EXECUTION_FORECAST_R01_AB_COMPLETE",
        "robot_commanded": False,
        "camera_opened": False,
        "source_trial_dir": str(source),
        "online_accept_m": float(candidate_summary["online_accept_m"]),
        "fit_margin_m": float(geometry["fit_margin_m"]),
        "coverage_ratio": float(geometry["coverage_ratio"]),
        "component_count": int(geometry["component_count"]),
        "component_base_radii_m": radii.tolist(),
        "legacy": legacy,
        "v3_execution_geometry": revised,
        "clearance_delta_v3_minus_legacy_m": (
            revised["candidate_min_distance_m"] - legacy["candidate_min_distance_m"]
        ),
        "archived_legacy_candidate_min_distance_m": float(
            candidate_summary["candidate_online_min_distance_m"]
        ),
        "archived_replay_abs_error_m": abs(
            legacy["candidate_min_distance_m"]
            - float(candidate_summary["candidate_online_min_distance_m"])
        ),
    }
    trial.write_json(args.output.resolve() / "summary.json", result)
    return result


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
