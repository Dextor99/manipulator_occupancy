#!/usr/bin/env python3
"""Scan fixed-X, pure +Y D2 opposing lanes with frozen online planning.

The obstacle geometry and stopped robot state come from a real protected-live
trial.  Only the obstacle lane X coordinate is translated.  Velocity is pure
base +Y.  STRO, the 0.11 m coarse gate, Fast, 0.09 m online gate, the 1 s
horizon and all robot limits remain unchanged.  This tool never opens the
camera or commands the robot and creates no execution authority.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

trial = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")
live = importlib.import_module("experiments.new.6_5.6_5_3.run_653_simple_dynamic_nubs_live")
event = importlib.import_module("experiments.new.6_5.6_5_3.run_653_simple_dynamic_nubs_event_replan_live")
legacy = importlib.import_module("experiments.new.6_5.6_5_3.offline_search_653_d2_complete_motion_geometry")

DEFAULT_SOURCE = ROOT / "results/new/6_5/6_5_3/simple_dynamic_nubs_complete_live/r02/core_live/trials/D2_opposing_approach_r02"
DEFAULT_REFERENCE = ROOT / "results/new/6_5/6_5_3/reference_xp10_line/reference_feedback.csv"
DEFAULT_OUTPUT = ROOT / "results/new/6_5/6_5_3/d2_fixed_x_opposing_search"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-trial-dir", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--reference-feedback-csv", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--x-offsets-m", default="-0.10,-0.075,-0.05,-0.025,0,0.025,0.05,0.075,0.10")
    p.add_argument("--speeds-m-s", default="0.08,0.10,0.12")
    p.add_argument("--forward-m", type=float, default=0.05)
    p.add_argument("--side-lengths-m", default="0.04,0.06,0.08")
    p.add_argument("--planning-robust-target-m", type=float, default=0.11)
    p.add_argument("--max-joint-delta-rad", type=float, default=0.12)
    return p


def vals(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(","))


def load_source(path: Path) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    summary = json.loads((path / "summary.json").read_text())
    fresh = json.loads((path / "post_stop_fresh_recheck.json").read_text())["result"]
    geometry = json.loads((path / "fresh_multisphere.json").read_text())
    q = None
    for e in summary.get("events", []):
        audit = (e.get("candidate") or {}).get("simple_live_audit") or {}
        if audit.get("q_actual_post_stop_rad") is not None:
            q = np.asarray(audit["q_actual_post_stop_rad"], dtype=np.float64)
            break
    if q is None:
        raise RuntimeError("source trial does not contain q_actual_post_stop_rad")
    return q, fresh, geometry


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_trial_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    q0, fresh, template = load_source(source)
    source_center = np.asarray(fresh["center"], dtype=np.float64)

    runtime = trial.build_parser().parse_args(["--scene", "D2", "--mode", "shadow"])
    config = trial.load_stage4_config(runtime.stage4_config)
    model = trial.load_stage4_surface_model(config)
    reference = trial.RecordedReference.load(args.reference_feedback_csv.resolve())
    goal, goal_audit = event.next_recorded_reference_goal(reference, q0, runtime.local_horizon_s)
    head, tail, durations, inner, _ = trial.make_local_reference(q0, np.zeros(6), runtime, reference_goal=goal)
    nominal = trial.NUBSTrajectory6D().generate(inner, head, tail, durations)
    planner = live.make_r06_fast_wrapper(
        trial.run_fast_repair,
        side_lengths=vals(args.side_lengths_m),
        forward_m=float(args.forward_m),
        max_joint_delta_rad=float(args.max_joint_delta_rad),
        robust_target_m=float(args.planning_robust_target_m),
        tcp_link="gripper_base_link",
    )

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="d2_fixed_x_") as tmp:
        for offset in vals(args.x_offsets_m):
            center = source_center.copy()
            center[0] += offset
            geometry = legacy.translated_geometry(template, source_center, center)
            for speed in vals(args.speeds_m_s):
                velocity = np.asarray([0.0, speed, 0.0], dtype=np.float64)
                forecast = trial.constant_multisphere_forecast(
                    np.asarray(geometry["component_centers"]),
                    np.asarray(geometry["component_base_radii"]), velocity
                )
                evaluator, _, _ = trial.make_risk_stack(config, model, forecast)
                profile = legacy.trajectory_profile(evaluator, forecast, nominal, 0.0, 0.5)
                current = float(profile[0]["distance_m"])
                predicted = min(float(x["distance_m"]) for x in profile)
                trigger_ok = bool(
                    predicted < runtime.moving_shadow_replan_in_m
                    and predicted <= current - 0.003
                    and current > runtime.moving_shadow_current_stop_m
                    and current > runtime.guided_hard_stop_m
                )
                row = {
                    "x_offset_m": offset, "lane_x_m": float(center[0]),
                    "center_y_m": float(center[1]), "center_z_m": float(center[2]),
                    "speed_y_m_s": speed, "trigger_current_m": current,
                    "trigger_predicted_m": predicted, "trigger_ok": trigger_ok,
                    "best_coarse_m": math.nan, "local1_ready": False,
                    "fast_m": math.nan, "fast_ms": math.nan, "feasible": False,
                }
                if trigger_ok:
                    artifacts: dict[str, Any] = {}
                    candidate = planner(
                        runtime, config, model, q_now=q0, qd_now=np.zeros(6),
                        center=center, velocity=velocity, radius=float(fresh["radius"]),
                        risk_links=set(model.surface_by_link(q0, density="coarse")),
                        trial_dir=Path(tmp) / f"x{offset:+.3f}_v{speed:.3f}",
                        reference_goal=goal, rejoin_goals=None,
                        obstacle_audit={"offline_fixed_x_opposing": True},
                        multisphere_geometry=geometry, artifacts_out=artifacts,
                    )
                    audit = candidate.get("simple_live_audit") or {}
                    selected = audit.get("selected_coarse_clearance_m")
                    attempted = [
                        float(c["coarse_min_distance_m"])
                        for c in audit.get("candidates", [])
                        if c.get("coarse_min_distance_m") is not None
                    ]
                    best = selected if selected is not None else (max(attempted) if attempted else math.nan)
                    ready = bool(candidate.get("local_repair_ready", False))
                    row.update({
                        "best_coarse_m": float(best), "local1_ready": ready,
                        "fast_m": float(candidate.get("verification_min_distance_m", math.nan)),
                        "fast_ms": float(candidate.get("online_pipeline_elapsed_ms", math.nan)),
                        "feasible": bool(ready),
                    })
                rows.append(row)

    with (output / "search_rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    feasible = sorted((r for r in rows if r["feasible"]), key=lambda r: (-r["best_coarse_m"], abs(r["x_offset_m"])))
    summary = {
        "status": "D2_FIXED_X_DOMAIN_FOUND" if feasible else "D2_FIXED_X_DOMAIN_NOT_FOUND",
        "robot_commanded": False, "camera_opened": False, "execution_authority": False,
        "algorithms_and_thresholds_frozen": True,
        "source_trial_dir": str(source),
        "source_center_m": source_center.tolist(),
        "source_component_radii_m": template["component_base_radii"],
        "velocity_policy": "pure_base_positive_y",
        "varied_fields": ["fixed_lane_x_m", "speed_y_m_s"],
        "reference_goal_audit": goal_audit,
        "scenario_count": len(rows), "trigger_count": sum(r["trigger_ok"] for r in rows),
        "feasible_count": len(feasible), "recommended_lane": feasible[0] if feasible else None,
        "top_feasible": feasible[:10], "search_rows_csv": str(output / "search_rows.csv"),
    }
    trial.write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
