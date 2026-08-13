#!/usr/bin/env python3
"""Non-commanding multi-segment rolling-local replay from one real 6.5.3 trial.

This is a structural diagnostic, not a live-safety authorization.  It advances
the saved obstacle with the last Fresh constant-velocity estimate because no
future RGB-D observations exist in an archived one-shot trial.
"""

from __future__ import annotations

import argparse
import csv
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
DEFAULT_SOURCE = (
    ROOT / "results/new/6_5/6_5_3/dynamic_repair_rolling_live_xp10"
    / "trials/D2_opposing_approach_r04"
)
DEFAULT_REFERENCE = ROOT / "results/new/6_5/6_5_3/reference_xp10_line/reference_feedback.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trial", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reference-feedback-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-segments", type=int, default=3)
    return parser


def trigger_reference_time(source: Path) -> float:
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    trigger_frame = int(summary["trigger_frame"])
    with (source / "frames.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["frame"]) == trigger_frame:
                return float(row["reference_time_s"])
    raise RuntimeError(f"trigger frame {trigger_frame} is missing")


def translated_geometry(geometry: dict[str, Any], shift: np.ndarray) -> dict[str, Any]:
    return {
        **geometry,
        "component_centers": np.asarray(geometry["component_centers"], dtype=np.float64)
        + shift[None, :],
        "component_base_radii": np.asarray(
            geometry["component_base_radii"], dtype=np.float64
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_segments < 1:
        raise ValueError("max-segments must be positive")
    source = args.source_trial.resolve()
    reference = trial.RecordedReference.load(args.reference_feedback_csv.resolve())
    candidate = json.loads((source / "candidate/candidate_summary.json").read_text(encoding="utf-8"))
    geometry = json.loads((source / "fresh_multisphere.json").read_text(encoding="utf-8"))
    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    output = (
        source / "offline_rolling_local_replay"
        if args.output is None
        else args.output.resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    runtime_args = trial.build_parser().parse_args(["--scene", "D2", "--mode", "shadow"])
    config = trial.load_stage4_config(runtime_args.stage4_config)
    model = trial.load_stage4_surface_model(config)
    reference_start = trigger_reference_time(source)
    schedule = trial.rolling_local_reference_schedule(
        reference_start,
        local_horizon_s=runtime_args.local_horizon_s,
        max_segments=args.max_segments,
        reference_end_time_s=float(reference.times[-1]),
    )
    q_now = np.asarray(candidate["q_now"], dtype=np.float64)
    center0 = np.asarray(candidate["obstacle_center"], dtype=np.float64)
    velocity = np.asarray(candidate["obstacle_velocity"], dtype=np.float64)
    locked_side = None
    rows: list[dict[str, Any]] = []
    segment_csvs: list[Path] = []
    for item in schedule:
        index = int(item["segment"])
        elapsed = float(item["reference_plan_start_time_s"]) - reference_start
        shift = velocity * elapsed
        segment_dir = output / f"segment_{index:02d}"
        artifacts: dict[str, Any] = {}
        result = trial.run_fast_repair(
            runtime_args,
            config,
            model,
            q_now=q_now,
            qd_now=np.zeros(6),
            center=center0 + shift,
            velocity=velocity,
            radius=float(candidate["obstacle_radius"]),
            risk_links=set(model.surface_by_link(q_now, density="coarse")),
            trial_dir=segment_dir,
            reference_goal=reference.state_at(float(item["reference_goal_time_s"])),
            rejoin_goals=None,
            obstacle_audit={
                "offline_rolling_local_replay": True,
                "segment": index,
                "constant_velocity_propagation_s": elapsed,
            },
            multisphere_geometry=translated_geometry(geometry, shift),
            artifacts_out=artifacts,
        )
        side = trial.avoidance_side_consistent(
            locked_side,
            np.asarray(result["tail_delta_q_rad"], dtype=np.float64),
            opposite_projection_tolerance_rad=runtime_args.rolling_side_opposite_tolerance_rad,
        )
        ready = bool(result["local_repair_ready"] and side["accepted"])
        candidate_csv = segment_dir / "candidate/fast_ccro_nubs_candidate.csv"
        metrics = trial.trajectory_workspace_deviation(
            model,
            candidate_csv,
            reference,
            float(item["reference_plan_start_time_s"]),
        )
        row = {
            **item,
            "q_start": q_now.tolist(),
            "obstacle_center": (center0 + shift).tolist(),
            "obstacle_velocity": velocity.tolist(),
            "local_repair_ready": bool(result["local_repair_ready"]),
            "side_consistent": bool(side["accepted"]),
            "side_reason": side["reason"],
            "side_projection_rad": float(side.get("projection_rad", np.nan)),
            "accepted_for_rolling_execution": ready,
            "accepted_steps": int(result["accepted_steps"]),
            "candidate_clearance_m": float(result["candidate_online_min_distance_m"]),
            "reference_clearance_m": float(result["reference_online_min_distance_m"]),
            "clearance_improvement_m": float(result["clearance_improvement_m"]),
            "fast_ms": float(result["online_pipeline_elapsed_ms"]),
            "tail_delta_q_rad": result["tail_delta_q_rad"],
            "workspace_deviation": metrics,
            "rejection_reasons": result["rejection_reasons"],
        }
        rows.append(row)
        if not ready:
            break
        if locked_side is None:
            locked_side = np.asarray(side["locked_tail_delta_q"], dtype=np.float64)
        q_now = np.asarray(
            artifacts["candidate_trajectory"].evaluate(
                artifacts["candidate_trajectory"].total_duration
            ),
            dtype=np.float64,
        )
        row["q_end"] = q_now.tolist()
        segment_csvs.append(candidate_csv)

    accepted_rows = [row for row in rows if row["accepted_for_rolling_execution"]]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "ROLLING_LOCAL_REPLAY_ALL_SEGMENTS_READY"
            if len(accepted_rows) == len(schedule)
            else "ROLLING_LOCAL_REPLAY_STOPPED_FAIL_CLOSED"
        ),
        "robot_commanded": False,
        "diagnostic_only": True,
        "fresh_rgbd_updated_each_segment": False,
        "obstacle_model": "saved Fresh constant velocity propagated between segments",
        "source_trial": str(source),
        "source_trial_status": source_summary.get("status"),
        "analysis_git_commit": trial.git_commit_hash(),
        "reference_start_time_s": reference_start,
        "requested_segments": len(schedule),
        "accepted_segments": len(accepted_rows),
        "side_lock_initialized": locked_side is not None,
        "max_tcp_deviation_m": max(
            (float(row["workspace_deviation"]["max_tcp_deviation_m"]) for row in accepted_rows),
            default=0.0,
        ),
        "max_body_deviation_m": max(
            (float(row["workspace_deviation"]["max_body_link_origin_deviation_m"]) for row in accepted_rows),
            default=0.0,
        ),
        "segments": rows,
    }
    trial.write_json(output / "rolling_local_replay.json", payload)
    return payload


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False, default=trial.json_default))


if __name__ == "__main__":
    main()
