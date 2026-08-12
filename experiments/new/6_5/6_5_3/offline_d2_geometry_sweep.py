#!/usr/bin/env python3
"""Offline lateral-offset feasibility sweep for one frozen D2 trial.

This analysis never opens the camera or robot connection.  It reuses the
saved trigger state, Fresh #1 multisphere geometry, velocity, recorded
reference, and the unchanged production Fast repair implementation.
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
DEFAULT_TRIAL = ROOT / "results/new/6_5/6_5_3/dynamic_repair_formal_v2/trials/D2_opposing_approach_r02"
DEFAULT_REFERENCE = ROOT / "results/new/6_5/6_5_3/reference_xp10_line/reference_feedback.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trial", type=Path, default=DEFAULT_TRIAL)
    parser.add_argument("--reference-feedback-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--axis", choices=("x", "z"), default="x")
    parser.add_argument("--offset-min-m", type=float, default=-0.20)
    parser.add_argument("--offset-max-m", type=float, default=0.20)
    parser.add_argument("--offset-step-m", type=float, default=0.01)
    return parser


def trigger_reference_index(source: Path) -> int:
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    trigger_frame = int(summary["trigger_frame"])
    with (source / "frames.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["frame"]) == trigger_frame:
                return int(row["reference_index"])
    raise RuntimeError(f"trigger frame {trigger_frame} is absent from frames.csv")


def contiguous_intervals(rows: list[dict[str, Any]], step_m: float) -> list[dict[str, Any]]:
    feasible = [row for row in rows if row["formal_scene_feasible"]]
    if not feasible:
        return []
    intervals: list[list[dict[str, Any]]] = [[feasible[0]]]
    for row in feasible[1:]:
        if abs(float(row["offset_m"]) - float(intervals[-1][-1]["offset_m"]) - step_m) <= 1.0e-8:
            intervals[-1].append(row)
        else:
            intervals.append([row])
    return [
        {
            "start_offset_m": float(group[0]["offset_m"]),
            "end_offset_m": float(group[-1]["offset_m"]),
            "width_m": float(group[-1]["offset_m"] - group[0]["offset_m"]),
            "sample_count": len(group),
            "midpoint_offset_m": 0.5 * float(group[0]["offset_m"] + group[-1]["offset_m"]),
            "minimum_candidate_clearance_m": min(float(row["candidate_clearance_m"]) for row in group),
            "maximum_candidate_clearance_m": max(float(row["candidate_clearance_m"]) for row in group),
        }
        for group in intervals
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.offset_step_m <= 0.0 or args.offset_max_m < args.offset_min_m:
        raise ValueError("invalid offset range")
    source = args.source_trial.resolve()
    candidate_path = source / "candidate/candidate_summary.json"
    geometry_path = source / "fresh_multisphere.json"
    for path in (candidate_path, geometry_path, source / "summary.json", source / "frames.csv"):
        if not path.is_file():
            raise FileNotFoundError(path)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    reference = trial.RecordedReference.load(args.reference_feedback_csv.resolve())
    reference.index = trigger_reference_index(source)
    runtime_args = trial.build_parser().parse_args(["--scene", "D2", "--mode", "shadow"])
    config = trial.load_stage4_config(runtime_args.stage4_config)
    model = trial.load_stage4_surface_model(config)
    evaluator, _, _ = trial.make_risk_stack(config, model, None)
    q_now = np.asarray(candidate["q_now"], dtype=np.float64)
    qd_now = np.asarray(candidate["qd_now"], dtype=np.float64)
    center0 = np.asarray(candidate["obstacle_center"], dtype=np.float64)
    velocity = np.asarray(candidate["obstacle_velocity"], dtype=np.float64)
    centers0 = np.asarray(geometry["component_centers"], dtype=np.float64)
    radii = np.asarray(geometry["component_base_radii"], dtype=np.float64)
    axis_index = 0 if args.axis == "x" else 2
    output = (
        args.output.resolve()
        if args.output is not None
        else source / f"offline_{args.axis}_geometry_feasibility"
    )
    output.mkdir(parents=True, exist_ok=True)
    offsets = np.arange(
        args.offset_min_m,
        args.offset_max_m + 0.5 * args.offset_step_m,
        args.offset_step_m,
    )
    rows: list[dict[str, Any]] = []
    for index, offset in enumerate(offsets):
        shift = np.zeros(3, dtype=np.float64)
        shift[axis_index] = float(offset)
        shifted_geometry = {
            **geometry,
            "component_centers": centers0 + shift[None, :],
            "component_base_radii": radii.copy(),
        }
        forecast = trial.constant_multisphere_forecast(centers0 + shift[None, :], radii, velocity)
        current = evaluator.configuration(q_now, forecast, 0.0, density="medium", with_gradient=False)
        preview = []
        for tau in np.arange(
            0.0,
            runtime_args.prediction_horizon_s + 0.5 * runtime_args.prediction_step_s,
            runtime_args.prediction_step_s,
        ):
            risk = evaluator.configuration(
                reference.state_after(float(tau))[0],
                forecast,
                float(tau),
                density="medium",
                with_gradient=False,
            )
            preview.append((float(tau), float(risk.min_distance), risk.nearest_link))
        predicted_tau, predicted_clearance, predicted_link = min(preview, key=lambda item: item[1])
        # Production Fast writes detailed candidate artifacts.  Keep those in
        # a temporary directory: the sweep's auditable products are the compact
        # CSV/JSON rows, not 5 files for every hypothetical geometry.
        with tempfile.TemporaryDirectory(prefix="ccro653_d2_sweep_") as temp_dir:
            result = trial.run_fast_repair(
                runtime_args,
                config,
                model,
                q_now=q_now,
                qd_now=qd_now,
                center=center0 + shift,
                velocity=velocity,
                radius=float(candidate["obstacle_radius"]),
                risk_links=set(model.surface_by_link(q_now, density="coarse")),
                trial_dir=Path(temp_dir),
                reference_goal=reference.state_after(runtime_args.local_horizon_s),
                rejoin_goals=[],
                obstacle_audit={"offline_geometry_sweep": True, "axis": args.axis, "offset_m": float(offset)},
                multisphere_geometry=shifted_geometry,
            )
        formal_scene_feasible = bool(
            predicted_clearance < runtime_args.moving_shadow_replan_in_m
            and float(result["reference_online_min_distance_m"]) < runtime_args.online_accept_m
            and bool(result["local_repair_ready"])
            and float(result["candidate_online_min_distance_m"]) >= runtime_args.online_accept_m
            and float(result["clearance_improvement_m"]) >= runtime_args.min_clearance_improvement_m
        )
        rows.append(
            {
                "axis": args.axis,
                "offset_m": float(offset),
                "predicted_clearance_m": predicted_clearance,
                "predicted_tau_s": predicted_tau,
                "predicted_nearest_link": predicted_link,
                "current_clearance_m": float(current.min_distance),
                "reference_clearance_m": float(result["reference_online_min_distance_m"]),
                "candidate_clearance_m": float(result["candidate_online_min_distance_m"]),
                "clearance_improvement_m": float(result["clearance_improvement_m"]),
                "fast_ms": float(result["online_pipeline_elapsed_ms"]),
                "accepted_steps": int(result["accepted_steps"]),
                "max_delta_q_rad": float(result["max_delta_q_from_reference_rad"]),
                "local_repair_ready": bool(result["local_repair_ready"]),
                "formal_scene_feasible": formal_scene_feasible,
                "rejection_reasons": ";".join(result["rejection_reasons"]),
            }
        )
    fields = list(rows[0])
    trial.write_csv(output / "d2_geometry_feasibility.csv", rows, fields)
    intervals = contiguous_intervals(rows, args.offset_step_m)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "FEASIBLE_INTERVAL_FOUND" if intervals else "NO_FEASIBLE_INTERVAL",
        "robot_commanded": False,
        "source_trial": str(source),
        "source_git_commit": json.loads((source / "summary.json").read_text(encoding="utf-8")).get("git_commit"),
        "analysis_git_commit": trial.git_commit_hash(),
        "formal_protocol_id": trial.FORMAL_PROTOCOL_ID,
        "axis": args.axis,
        "offset_min_m": float(args.offset_min_m),
        "offset_max_m": float(args.offset_max_m),
        "offset_step_m": float(args.offset_step_m),
        "frozen_thresholds": {
            "predicted_trigger_m": runtime_args.moving_shadow_replan_in_m,
            "online_accept_m": runtime_args.online_accept_m,
            "minimum_clearance_improvement_m": runtime_args.min_clearance_improvement_m,
            "fast_budget_ms": runtime_args.fast_budget_ms,
        },
        "feasibility_definition": {
            "predicted_trigger": "predicted_clearance_m < 0.14",
            "reference_unsafe": "reference_clearance_m < 0.09",
            "candidate_safe": "local_repair_ready and candidate_clearance_m >= 0.09",
            "minimum_gain": "clearance_improvement_m >= 0.003",
        },
        "feasible_intervals": intervals,
        "recommended_midpoint_offset_m": (
            None if not intervals else max(intervals, key=lambda item: (item["width_m"], item["sample_count"]))["midpoint_offset_m"]
        ),
        "rows": rows,
    }
    trial.write_json(output / "d2_geometry_feasibility.json", payload)
    return payload


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False, default=trial.json_default))


if __name__ == "__main__":
    main()
