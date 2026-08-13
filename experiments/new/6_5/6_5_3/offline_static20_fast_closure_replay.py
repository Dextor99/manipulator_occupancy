#!/usr/bin/env python3
"""Pure-offline Static20 replay of saved rolling-local candidates and closure.

This diagnostic never commands the robot and never changes the production
dynamic forecast.  It reuses one archived RGB-D cluster, represents it with
the existing coverage-preserving PCA multisphere, and applies a time-invariant
20 mm static-observation inflation (zero velocity and zero uncertainty growth).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
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
from planning.obstacle_forecast import (  # noqa: E402
    CompositeForecast,
    ConstantVelocitySphereForecast,
)


DEFAULT_SOURCE = ROOT / "results/new/6_5/6_5_3/static_online_fast_shadow/r07"
DEFAULT_REFERENCE = ROOT / "results/new/6_5/6_5_3/reference_xp10_line/reference_feedback.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reference-feedback-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--geometry-segment", type=int, default=3)
    parser.add_argument("--geometry-attempt", type=int, default=6)
    parser.add_argument("--static-inflation-m", type=float, default=0.020)
    parser.add_argument("--online-accept-m", type=float, default=0.090)
    parser.add_argument("--reference-step-s", type=float, default=0.025)
    parser.add_argument("--bridge-step-s", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _selected_candidate(summary: dict[str, Any], source: Path, segment: int) -> tuple[int, Path]:
    row = next(item for item in summary["segments"] if int(item["segment"]) == segment)
    attempt = row.get("authorized_attempt")
    if attempt is None:
        attempt = row["attempts"][-1]["attempt"]
    attempt = int(attempt)
    base = source / f"segment_{segment:02d}" / f"attempt_{attempt:02d}"
    authorized = base / "local_execution_authorization/authorized_local_repair.csv"
    candidate = base / "candidate/fast_ccro_nubs_candidate.csv"
    return attempt, authorized if authorized.exists() else candidate


def _static20_forecast(geometry: dict[str, Any], inflation: float, horizon: float) -> CompositeForecast:
    centers = np.asarray(geometry["component_centers"], dtype=np.float64)
    radii = np.asarray(geometry["component_base_radii"], dtype=np.float64)
    forecasts = []
    for index, (center, radius) in enumerate(zip(centers, radii), 1):
        forecasts.append(
            ConstantVelocitySphereForecast(
                center=center,
                velocity=np.zeros(3),
                radius=float(radius),
                valid_horizon=float(horizon),
                object_id=index,
                margin=float(inflation),
                uncertainty=0.0,
                uncertainty_growth=0.0,
                velocity_radius_scale=0.0,
                beyond_horizon="error",
            )
        )
    return CompositeForecast(forecasts)


def _verification_dict(result: Any) -> dict[str, Any]:
    return asdict(result)


def _safe_suffix(rows: list[dict[str, Any]], threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    suffix_min = float("inf")
    for row in reversed(rows):
        suffix_min = min(suffix_min, float(row["distance_m"]))
        row["suffix_min_distance_m"] = suffix_min
        row["safe_suffix"] = bool(suffix_min >= threshold)
    earliest = next((row for row in rows if row["safe_suffix"]), None)
    return rows, earliest


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.static_inflation_m < 0.0 or args.online_accept_m <= 0.0:
        raise ValueError("inflation must be non-negative and acceptance distance positive")
    if args.reference_step_s <= 0.0 or args.bridge_step_s <= 0.0:
        raise ValueError("sampling steps must be positive")

    source = args.source_run.resolve()
    output = (args.output or source / "offline_static20_fast_closure_replay.json").resolve()
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    geometry_dir = source / f"segment_{args.geometry_segment:02d}" / f"attempt_{args.geometry_attempt:02d}"
    points_path = geometry_dir / "fresh_plan_points.npy"
    points = np.asarray(np.load(points_path), dtype=np.float64)

    runtime_args = trial.build_parser().parse_args(["--scene", "D2", "--mode", "shadow"])
    config = trial.load_stage4_config(runtime_args.stage4_config)
    model = trial.load_stage4_surface_model(config)
    reference = trial.RecordedReference.load(args.reference_feedback_csv.resolve())
    geometry = trial.fit_pca_multisphere(
        points,
        fit_margin_m=runtime_args.multisphere_fit_margin_m,
        max_components=runtime_args.multisphere_max_components,
    )
    # Keep every verifier query inside the declared forecast horizon.
    horizon = max(2.0, float(reference.times[-1] - reference.times[0]) + 2.0)
    forecast = _static20_forecast(geometry, args.static_inflation_m, horizon)
    evaluator, verifier, _ = trial.make_risk_stack(config, model, forecast)
    verifier.d_stop = float(args.online_accept_m)

    candidate_rows: list[dict[str, Any]] = []
    trajectories = []
    for segment in (1, 2, 3):
        attempt, path = _selected_candidate(summary, source, segment)
        trajectory = trial.reconstruct_saved_nubs_candidate(path, segments=runtime_args.local_segments)
        samples = trajectory.sample(np.asarray([0.0, trajectory.total_duration]))
        result = verifier.verify(
            trajectory,
            forecast,
            current_q=samples.q[0],
            current_qd=samples.qd[0],
            current_qdd=samples.qdd[0],
            q_goal=samples.q[-1],
            solver_success=True,
        )
        candidate_rows.append(
            {
                "segment": segment,
                "attempt": attempt,
                "candidate_csv": str(path),
                "verification": _verification_dict(result),
            }
        )
        trajectories.append(trajectory)

    third_summary = next(item for item in summary["segments"] if int(item["segment"]) == 3)
    reference_start = float(third_summary["reference_plan_start_time_s"])
    ref_times = np.arange(
        reference_start,
        float(reference.times[-1]) + 0.5 * args.reference_step_s,
        args.reference_step_s,
    )
    ref_times = np.unique(np.r_[ref_times, float(reference.times[-1])])
    reference_rows: list[dict[str, Any]] = []
    for absolute_time in ref_times:
        q, _, _ = reference.state_at(float(absolute_time))
        clearance = evaluator.configuration_clearance(q, forecast, 0.0, density=verifier.density)
        reference_rows.append(
            {
                "absolute_time_s": float(absolute_time),
                "offset_from_segment3_start_s": float(absolute_time - reference_start),
                "distance_m": float(clearance.min_distance),
                "nearest_link": clearance.nearest_link,
            }
        )
    reference_rows, earliest_safe = _safe_suffix(reference_rows, args.online_accept_m)

    repair = trajectories[-1]
    repair_end_absolute = reference_start + float(repair.total_duration)
    bridge_rows: list[dict[str, Any]] = []
    first_bridge = None
    if earliest_safe is not None:
        first_endpoint = max(repair_end_absolute + args.bridge_step_s, float(earliest_safe["absolute_time_s"]))
        endpoints = np.arange(
            first_endpoint,
            float(reference.times[-1]) + 0.5 * args.bridge_step_s,
            args.bridge_step_s,
        )
        endpoints = np.unique(np.r_[endpoints, float(reference.times[-1])])
        endpoints = endpoints[endpoints <= float(reference.times[-1]) + 1.0e-9]
        for endpoint in endpoints:
            duration = float(endpoint - repair_end_absolute)
            if duration <= 0.0:
                continue
            rejoin_state = reference.state_at(float(endpoint))
            bridge = trial.make_rejoin_bridge(repair, rejoin_state, duration)
            head = repair.tail_state
            result = verifier.verify(
                bridge,
                forecast,
                current_q=head[:, 0],
                current_qd=head[:, 1],
                current_qdd=head[:, 2],
                q_goal=rejoin_state[0],
                solver_success=True,
            )
            suffix_row = next(row for row in reference_rows if row["absolute_time_s"] >= endpoint - 1.0e-9)
            row = {
                "rejoin_absolute_time_s": float(endpoint),
                "bridge_duration_s": duration,
                "remainder_suffix_min_distance_m": float(suffix_row["suffix_min_distance_m"]),
                "verification": _verification_dict(result),
            }
            bridge_rows.append(row)
            if result.accepted and first_bridge is None:
                first_bridge = row

    all_candidates_safe = all(row["verification"]["accepted"] for row in candidate_rows)
    safe_suffix_exists = earliest_safe is not None
    if all_candidates_safe and first_bridge is not None:
        decision = "STATIC20_SOLVES_SAVED_CANDIDATES_AND_CLOSURE"
    elif all_candidates_safe and safe_suffix_exists:
        decision = "STATIC20_CANDIDATES_SAFE_BUT_NO_BRIDGE_CONSIDER_OFFSET_PRESERVING_GOAL"
    else:
        decision = "STATIC20_GEOMETRY_OR_CANDIDATE_CHAIN_STILL_FAILS"

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "STATIC20_FAST_CLOSURE_REPLAY_COMPLETE",
        "decision": decision,
        "robot_commanded": False,
        "authoritative_for_execution": False,
        "production_dynamic_forecast_changed": False,
        "source_run": str(source),
        "source_points": str(points_path),
        "reference_feedback_csv": str(args.reference_feedback_csv.resolve()),
        "geometry_frame": {"segment": args.geometry_segment, "attempt": args.geometry_attempt},
        "static_forecast": {
            "model": "coverage_preserving_pca_multisphere",
            "component_count": int(geometry["component_count"]),
            "coverage_ratio": float(geometry["coverage_ratio"]),
            "fit_margin_m": float(runtime_args.multisphere_fit_margin_m),
            "observation_inflation_m": float(args.static_inflation_m),
            "velocity_m_s": [0.0, 0.0, 0.0],
            "uncertainty_growth_m_s": 0.0,
            "velocity_radius_scale": 0.0,
        },
        "online_accept_m": float(args.online_accept_m),
        "candidate_segments": candidate_rows,
        "all_three_candidates_safe": all_candidates_safe,
        "reference_profile": reference_rows,
        "reference_min_distance_m": min(row["distance_m"] for row in reference_rows),
        "earliest_safe_suffix": earliest_safe,
        "bridge_step_s": float(args.bridge_step_s),
        "bridge_search": bridge_rows,
        "first_safe_bridge": first_bridge,
    }
    trial.write_json(output, payload)
    return payload


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    compact = {
        "status": result["status"],
        "decision": result["decision"],
        "robot_commanded": result["robot_commanded"],
        "candidate_clearances_m": [
            row["verification"]["min_distance"] for row in result["candidate_segments"]
        ],
        "earliest_safe_suffix": result["earliest_safe_suffix"],
        "first_safe_bridge": result["first_safe_bridge"],
        "output": str((args.output or args.source_run / "offline_static20_fast_closure_replay.json").resolve()),
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False, default=trial.json_default))


if __name__ == "__main__":
    main()
