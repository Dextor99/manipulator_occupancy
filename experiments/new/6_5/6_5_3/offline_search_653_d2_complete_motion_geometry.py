#!/usr/bin/env python3
"""Search only D2 obstacle motion geometry for a complete frozen-method demo.

Robot/reference state, observed two-sphere shape, STRO/Fast thresholds and the
1 s local planner are fixed.  The grid changes only the obstacle center phase,
fixed lateral offset, opposing angle and speed.  A feasible row must trigger
STRO early, pass the unchanged r06 coarse/Fast/local verifier, and leave the
measured local tail physically safe over the next 0.5 s while moving away.

This is a pure offline scene-design tool.  It creates no execution authority.
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
event = importlib.import_module(
    "experiments.new.6_5.6_5_3.run_653_simple_dynamic_nubs_event_replan_live"
)

DEFAULT_R04 = ROOT / "results/new/6_5/6_5_3/simple_dynamic_nubs_live/r04"
DEFAULT_OUTPUT = ROOT / "results/new/6_5/6_5_3/d2_complete_motion_geometry_search"
DEFAULT_REFERENCE = ROOT / "results/new/6_5/6_5_3/reference_xp10_line/reference_feedback.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-r04", type=Path, default=DEFAULT_R04)
    parser.add_argument("--reference-feedback-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--angles-deg", default="-30,-15,0,15,30")
    parser.add_argument("--center-x-m", default="0.56,0.60,0.64,0.68,0.72")
    parser.add_argument("--center-y-m", default="-0.24,-0.16,-0.08,0.00,0.08")
    parser.add_argument("--speeds-m-s", default="0.06,0.08,0.10,0.12")
    parser.add_argument("--max-feasible", type=int, default=20)
    parser.add_argument("--forward-m", type=float, default=0.05)
    parser.add_argument("--side-lengths-m", default="0.04,0.06,0.08")
    parser.add_argument("--planning-robust-target-m", type=float, default=0.11)
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.12)
    return parser


def values(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(","))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def translated_geometry(
    template: dict[str, Any], source_center: np.ndarray, center: np.ndarray
) -> dict[str, Any]:
    shift = np.asarray(center) - np.asarray(source_center)
    return {
        **template,
        "component_centers": (
            np.asarray(template["component_centers"], dtype=np.float64) + shift[None, :]
        ),
        "component_base_radii": np.asarray(
            template["component_base_radii"], dtype=np.float64
        ),
        "search_translation_m": shift,
    }


def trajectory_profile(evaluator: Any, forecast: Any, trajectory: Any, start: float, stop: float):
    profile = []
    for tau in np.arange(start, stop + 0.05, 0.10):
        q = trajectory.evaluate(min(float(tau), float(trajectory.total_duration)))
        risk = evaluator.configuration(q, forecast, float(tau), density="medium", with_gradient=False)
        profile.append(
            {"tau_s": float(tau), "distance_m": float(risk.min_distance), "link": risk.nearest_link}
        )
    return profile


def stationary_profile(evaluator: Any, forecast: Any, q: np.ndarray):
    profile = []
    for tau in np.arange(0.0, 0.51, 0.10):
        risk = evaluator.configuration(q, forecast, float(tau), density="medium", with_gradient=False)
        profile.append(
            {"tau_s": float(tau), "distance_m": float(risk.min_distance), "link": risk.nearest_link}
        )
    return profile


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = args.source_r04.resolve()
    trial_dir = source / "core_live/trials/D2_opposing_approach_r04"
    r04_candidate = read_json(trial_dir / "candidate/candidate_summary.json")
    fresh1 = read_json(trial_dir / "post_stop_fresh_recheck.json")["result"]
    geometry_template = read_json(trial_dir / "fresh_multisphere.json")
    q0 = np.asarray(r04_candidate["q_now"], dtype=np.float64)
    center_template = np.asarray(fresh1["center"], dtype=np.float64)

    runtime_args = trial.build_parser().parse_args(["--scene", "D2", "--mode", "shadow"])
    config = trial.load_stage4_config(runtime_args.stage4_config)
    model = trial.load_stage4_surface_model(config)
    reference = trial.RecordedReference.load(args.reference_feedback_csv.resolve())
    reference_goal, reference_audit = event.next_recorded_reference_goal(
        reference, q0, runtime_args.local_horizon_s
    )
    head, tail, durations, inner, _ = trial.make_local_reference(
        q0, np.zeros(6), runtime_args, reference_goal=reference_goal
    )
    nominal_trajectory = trial.NUBSTrajectory6D().generate(inner, head, tail, durations)
    side_lengths = values(args.side_lengths_m)
    planner = live.make_r06_fast_wrapper(
        trial.run_fast_repair,
        side_lengths=side_lengths,
        forward_m=float(args.forward_m),
        max_joint_delta_rad=float(args.max_joint_delta_rad),
        robust_target_m=float(args.planning_robust_target_m),
        tcp_link="gripper_base_link",
    )

    rows = []
    feasible = []
    scenario_id = 0
    with tempfile.TemporaryDirectory(prefix="d2_geometry_search_") as temp_root:
        for angle_deg in values(args.angles_deg):
            angle = math.radians(angle_deg)
            for center_x in values(args.center_x_m):
                for center_y in values(args.center_y_m):
                    for speed in values(args.speeds_m_s):
                        scenario_id += 1
                        center = np.asarray([center_x, center_y, center_template[2]], dtype=np.float64)
                        # Robot travels primarily -Y; +Y is opposing.  Angle adds
                        # a fixed transverse X component without following robot avoidance.
                        velocity = float(speed) * np.asarray([math.sin(angle), math.cos(angle), 0.0])
                        geometry = translated_geometry(geometry_template, center_template, center)
                        forecast = trial.constant_multisphere_forecast(
                            np.asarray(geometry["component_centers"]),
                            np.asarray(geometry["component_base_radii"]),
                            velocity,
                        )
                        evaluator, _, _ = trial.make_risk_stack(config, model, forecast)
                        trigger_profile = trajectory_profile(
                            evaluator, forecast, nominal_trajectory, 0.0, 0.5
                        )
                        current = trigger_profile[0]["distance_m"]
                        predicted = min(row["distance_m"] for row in trigger_profile)
                        trigger_ok = bool(
                            predicted < runtime_args.moving_shadow_replan_in_m
                            and predicted <= current - 0.003
                            and current > runtime_args.moving_shadow_current_stop_m
                            and current > runtime_args.guided_hard_stop_m
                        )
                        row: dict[str, Any] = {
                            "scenario_id": scenario_id,
                            "angle_deg": angle_deg,
                            "center_x_m": center_x,
                            "center_y_m": center_y,
                            "center_z_m": float(center[2]),
                            "speed_m_s": speed,
                            "velocity_x_m_s": float(velocity[0]),
                            "velocity_y_m_s": float(velocity[1]),
                            "trigger_current_m": current,
                            "trigger_predicted_m": predicted,
                            "trigger_ok": trigger_ok,
                            "predictive_gain_m": current - predicted,
                            "local1_ready": False,
                            "coarse_m": math.nan,
                            "fast_m": math.nan,
                            "fast_ms": math.nan,
                            "hold_start_m": math.nan,
                            "hold_min_m": math.nan,
                            "hold_end_m": math.nan,
                            "hold_min_tau_s": math.nan,
                            "tail_safe": False,
                            "moving_away": False,
                            "feasible": False,
                        }
                        if not trigger_ok:
                            rows.append(row)
                            continue
                        artifacts: dict[str, Any] = {}
                        scenario_dir = Path(temp_root) / f"s{scenario_id:04d}"
                        candidate = planner(
                            runtime_args,
                            config,
                            model,
                            q_now=q0,
                            qd_now=np.zeros(6),
                            center=center,
                            velocity=velocity,
                            radius=float(fresh1["radius"]),
                            risk_links=set(model.surface_by_link(q0, density="coarse")),
                            trial_dir=scenario_dir,
                            reference_goal=reference_goal,
                            rejoin_goals=None,
                            obstacle_audit={"track_id": 1, "offline_scene_search": True},
                            multisphere_geometry=geometry,
                            artifacts_out=artifacts,
                        )
                        audit = candidate.get("simple_live_audit", {})
                        selected = audit.get("selected_coarse_clearance_m")
                        row.update(
                            {
                                "local1_ready": bool(candidate.get("local_repair_ready", False)),
                                "coarse_m": math.nan if selected is None else float(selected),
                                "fast_m": float(candidate.get("verification_min_distance_m", math.nan)),
                                "fast_ms": float(candidate.get("online_pipeline_elapsed_ms", math.nan)),
                            }
                        )
                        if not row["local1_ready"]:
                            rows.append(row)
                            continue
                        local = artifacts["candidate_trajectory"]
                        q_tail = np.asarray(local.evaluate(local.total_duration), dtype=np.float64)
                        tail_center = center + velocity * float(local.total_duration)
                        tail_geometry = translated_geometry(geometry, center, tail_center)
                        tail_forecast = trial.constant_multisphere_forecast(
                            np.asarray(tail_geometry["component_centers"]),
                            np.asarray(tail_geometry["component_base_radii"]),
                            velocity,
                        )
                        tail_evaluator, _, _ = trial.make_risk_stack(config, model, tail_forecast)
                        hold_profile = stationary_profile(tail_evaluator, tail_forecast, q_tail)
                        hold_start = hold_profile[0]["distance_m"]
                        hold_min_index = int(
                            np.argmin([item["distance_m"] for item in hold_profile])
                        )
                        hold_min = hold_profile[hold_min_index]["distance_m"]
                        hold_end = hold_profile[-1]["distance_m"]
                        tail_safe = hold_min >= runtime_args.moving_shadow_replan_in_m
                        moving_away = bool(
                            hold_min_index < len(hold_profile) - 1
                            and hold_end >= hold_min + 0.001
                        )
                        is_feasible = bool(tail_safe and moving_away)
                        row.update(
                            {
                                "hold_start_m": hold_start,
                                "hold_min_m": hold_min,
                                "hold_end_m": hold_end,
                                "hold_min_tau_s": hold_profile[hold_min_index]["tau_s"],
                                "tail_safe": tail_safe,
                                "moving_away": moving_away,
                                "feasible": is_feasible,
                            }
                        )
                        rows.append(row)
                        if is_feasible:
                            feasible.append(row)
                            if len(feasible) >= args.max_feasible:
                                break
                    if len(feasible) >= args.max_feasible:
                        break
                if len(feasible) >= args.max_feasible:
                    break
            if len(feasible) >= args.max_feasible:
                break

    fields = list(rows[0]) if rows else []
    with (output / "search_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    ranked = sorted(
        feasible,
        key=lambda row: (
            -float(row["hold_min_m"]),
            abs(float(row["angle_deg"])),
            -float(row["fast_m"]),
        ),
    )
    summary = {
        "status": "D2_COMPLETE_SCENE_DOMAIN_FOUND" if ranked else "D2_COMPLETE_SCENE_DOMAIN_NOT_FOUND",
        "robot_commanded": False,
        "camera_opened": False,
        "execution_authority": False,
        "algorithms_and_thresholds_frozen": True,
        "varied_fields": ["angle_deg", "center_x_m", "center_y_m", "speed_m_s"],
        "fixed_thresholds_m": {
            "trigger": runtime_args.moving_shadow_replan_in_m,
            "current_stop": runtime_args.moving_shadow_current_stop_m,
            "coarse": args.planning_robust_target_m,
            "online": runtime_args.online_accept_m,
            "hard": runtime_args.guided_hard_stop_m,
        },
        "reference_goal_audit": reference_audit,
        "scenario_count": len(rows),
        "trigger_count": sum(bool(row["trigger_ok"]) for row in rows),
        "local1_ready_count": sum(bool(row["local1_ready"]) for row in rows),
        "feasible_count": len(ranked),
        "recommended_scene": ranked[0] if ranked else None,
        "top_feasible": ranked[:10],
        "search_rows_csv": str(output / "search_rows.csv"),
    }
    trial.write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
