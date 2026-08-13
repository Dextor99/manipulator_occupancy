#!/usr/bin/env python3
"""Pure-offline Static20 rollout with reference-increment transported goals.

No camera or robot interface is opened.  Production Fast and its dynamic
forecast remain unchanged; this diagnostic changes only the nominal local goal
and injects the calibrated, time-invariant Static20 forecast.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

trial = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")
static20 = importlib.import_module(
    "experiments.new.6_5.6_5_3.offline_static20_fast_closure_replay"
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
    parser.add_argument("--reference-step-s", type=float, default=0.025)
    parser.add_argument("--bridge-step-s", type=float, default=0.25)
    parser.add_argument("--tcp-link", default="gripper_base_link")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def transported_reference_goal(
    reference: Any,
    q_now: np.ndarray,
    anchor_s: float,
    horizon_s: float,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    q_anchor, _, _ = reference.state_at(anchor_s)
    goal_time = min(float(reference.times[-1]), anchor_s + horizon_s)
    q_ref_goal, qd_ref_goal, qdd_ref_goal = reference.state_at(goal_time)
    increment = q_ref_goal - q_anchor
    q_goal = np.asarray(q_now, dtype=np.float64) + increment
    return (q_goal, qd_ref_goal, qdd_ref_goal), {
        "reference_anchor_time_s": float(anchor_s),
        "reference_goal_time_s": goal_time,
        "reference_increment_rad": increment.tolist(),
        "transported_goal_rad": q_goal.tolist(),
        "offset_from_reference_at_anchor_rad": (np.asarray(q_now) - q_anchor).tolist(),
        "offset_from_reference_at_goal_rad": (q_goal - q_ref_goal).tolist(),
    }


def tcp_position(model: Any, q: np.ndarray, link: str) -> np.ndarray:
    transforms = model.urdf.link_transforms(
        {name: float(q[index]) for index, name in enumerate(model.joint_names)}
    )
    if link not in transforms:
        raise KeyError(f"TCP link {link!r} is absent from the robot model")
    return np.asarray(transforms[link][:3, 3], dtype=np.float64)


def reference_safe_suffix(
    reference: Any,
    evaluator: Any,
    forecast: Any,
    density: str,
    start_s: float,
    step_s: float,
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    times = np.arange(start_s, float(reference.times[-1]) + 0.5 * step_s, step_s)
    times = np.unique(np.r_[times, float(reference.times[-1])])
    rows = []
    for absolute in times:
        q, _, _ = reference.state_at(float(absolute))
        result = evaluator.configuration_clearance(q, forecast, 0.0, density=density)
        rows.append(
            {
                "absolute_time_s": float(absolute),
                "distance_m": float(result.min_distance),
                "nearest_link": result.nearest_link,
            }
        )
    return static20._safe_suffix(rows, threshold)


def search_closure(
    repair: Any,
    *,
    reference: Any,
    earliest_safe_suffix: dict[str, Any] | None,
    anchor_goal_s: float,
    verifier: Any,
    forecast: Any,
    bridge_step_s: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if earliest_safe_suffix is None:
        return [], None
    repair_tail_time = float(anchor_goal_s)
    first_endpoint = max(
        repair_tail_time + bridge_step_s,
        float(earliest_safe_suffix["absolute_time_s"]),
    )
    endpoints = np.arange(
        first_endpoint,
        float(reference.times[-1]) + 0.5 * bridge_step_s,
        bridge_step_s,
    )
    endpoints = np.unique(np.r_[endpoints, float(reference.times[-1])])
    endpoints = endpoints[endpoints <= float(reference.times[-1]) + 1.0e-9]
    rows = []
    for endpoint in endpoints:
        duration = float(endpoint - repair_tail_time)
        if duration <= 0.0:
            continue
        state = reference.state_at(float(endpoint))
        bridge = trial.make_rejoin_bridge(repair, state, duration)
        head = repair.tail_state
        verification = verifier.verify(
            bridge,
            forecast,
            current_q=head[:, 0],
            current_qd=head[:, 1],
            current_qdd=head[:, 2],
            q_goal=state[0],
            solver_success=True,
        )
        row = {
            "rejoin_absolute_time_s": float(endpoint),
            "bridge_duration_s": duration,
            "verification": asdict(verification),
        }
        rows.append(row)
        if verification.accepted:
            return rows, row
    return rows, None


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.static_inflation_m < 0.0:
        raise ValueError("static inflation must be non-negative")
    if args.reference_step_s <= 0.0 or args.bridge_step_s <= 0.0:
        raise ValueError("sampling steps must be positive")
    source = args.source_run.resolve()
    output_dir = (
        args.output.resolve()
        if args.output is not None
        else source / "offline_static20_offset_preserving_rollout"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    source_segments = source_summary["segments"]
    if not source_segments:
        raise RuntimeError("source run contains no rolling segments")
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
    horizon = max(2.0, float(reference.times[-1] - reference.times[0]) + 2.0)
    forecast = static20._static20_forecast(geometry, args.static_inflation_m, horizon)
    evaluator, verifier, _ = trial.make_risk_stack(config, model, forecast)
    verifier.d_stop = float(runtime_args.online_accept_m)

    anchor = float(source_segments[0]["reference_plan_start_time_s"])
    q_virtual = np.asarray(source_segments[0]["q_virtual_start"], dtype=np.float64)
    initial_tcp = tcp_position(model, q_virtual, args.tcp_link)
    goal_tcp = tcp_position(model, reference.state_at(float(reference.times[-1]))[0], args.tcp_link)
    task_y_direction = float(np.sign(goal_tcp[1] - initial_tcp[1]))
    if task_y_direction == 0.0:
        raise RuntimeError("reference has no TCP-Y task direction")

    reference_rows, earliest_safe = reference_safe_suffix(
        reference,
        evaluator,
        forecast,
        verifier.density,
        anchor,
        args.reference_step_s,
        runtime_args.online_accept_m,
    )
    max_segments = int(math.ceil((float(reference.times[-1]) - anchor) / runtime_args.local_horizon_s))
    locked_side = None
    segments = []
    closure = None
    status = "OFFSET_PRESERVING_ROLLOUT_REACHED_REFERENCE_END_WITHOUT_CLOSURE"

    for index in range(1, max_segments + 1):
        if anchor >= float(reference.times[-1]) - 1.0e-9:
            break
        local_goal, goal_audit = transported_reference_goal(
            reference, q_virtual, anchor, runtime_args.local_horizon_s
        )
        segment_dir = output_dir / f"segment_{index:02d}"
        artifacts: dict[str, Any] = {}
        result = trial.run_fast_repair(
            runtime_args,
            config,
            model,
            q_now=q_virtual,
            qd_now=np.zeros(6),
            center=np.asarray(geometry["component_centers"], dtype=np.float64).mean(axis=0),
            velocity=np.zeros(3),
            radius=float(max(geometry["component_base_radii"])),
            risk_links=set(model.surface_by_link(q_virtual, density="coarse")),
            trial_dir=segment_dir,
            reference_goal=local_goal,
            rejoin_goals=None,
            obstacle_audit={
                "offline_static20_offset_preserving_rollout": True,
                "offline_forecast_override_authorized": True,
                "segment": index,
                "static_observation_inflation_m": float(args.static_inflation_m),
            },
            multisphere_geometry=geometry,
            artifacts_out=artifacts,
            forecast_override=forecast,
        )
        candidate = artifacts.get("candidate_trajectory")
        side = trial.avoidance_side_consistent(
            locked_side,
            np.asarray(result["tail_delta_q_rad"], dtype=np.float64),
            opposite_projection_tolerance_rad=runtime_args.rolling_side_opposite_tolerance_rad,
        )
        accepted = bool(result["local_repair_ready"] and side["accepted"] and candidate is not None)
        row: dict[str, Any] = {
            "segment": index,
            **goal_audit,
            "q_virtual_start": q_virtual.tolist(),
            "fast": result,
            "side_continuity": side,
            "accepted": accepted,
        }
        if not accepted:
            row["status"] = "SAFE_HOLD"
            segments.append(row)
            status = "OFFSET_PRESERVING_ROLLOUT_SAFE_HOLD"
            break

        q_next = np.asarray(candidate.evaluate(candidate.total_duration), dtype=np.float64)
        tcp_start = tcp_position(model, q_virtual, args.tcp_link)
        tcp_end = tcp_position(model, q_next, args.tcp_link)
        progress_y = float(task_y_direction * (tcp_end[1] - tcp_start[1]))
        row.update(
            {
                "status": "SEGMENT_ACCEPTED",
                "q_virtual_end": q_next.tolist(),
                "tcp_start_m": tcp_start.tolist(),
                "tcp_end_m": tcp_end.tolist(),
                "tcp_y_task_progress_m": progress_y,
                "tcp_y_progress_ok": bool(progress_y > 0.0),
            }
        )
        if progress_y <= 0.0:
            row["status"] = "SAFE_HOLD_NO_TCP_Y_PROGRESS"
            segments.append(row)
            status = "OFFSET_PRESERVING_ROLLOUT_SAFE_HOLD_NO_PROGRESS"
            break
        if locked_side is None:
            locked_side = np.asarray(side["locked_tail_delta_q"], dtype=np.float64)

        next_anchor = min(float(reference.times[-1]), anchor + runtime_args.local_horizon_s)
        bridges, safe_bridge = search_closure(
            candidate,
            reference=reference,
            earliest_safe_suffix=earliest_safe,
            anchor_goal_s=next_anchor,
            verifier=verifier,
            forecast=forecast,
            bridge_step_s=args.bridge_step_s,
        )
        row["closure_search"] = bridges
        row["safe_bridge"] = safe_bridge
        segments.append(row)
        q_virtual = q_next
        anchor = next_anchor
        if safe_bridge is not None:
            closure = {"after_segment": index, **safe_bridge}
            status = "OFFSET_PRESERVING_ROLLING_CLOSURE_SUCCESS"
            break

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "robot_commanded": False,
        "authoritative_for_execution": False,
        "production_fast_changed": False,
        "production_dynamic_forecast_changed": False,
        "changed_experimental_variable": "transported_local_goal_only",
        "source_run": str(source),
        "source_points": str(points_path),
        "reference_feedback_csv": str(args.reference_feedback_csv.resolve()),
        "static_observation_inflation_m": float(args.static_inflation_m),
        "online_accept_m": float(runtime_args.online_accept_m),
        "local_horizon_s": float(runtime_args.local_horizon_s),
        "maximum_segments_from_reference_duration": max_segments,
        "task_y_direction": task_y_direction,
        "reference_min_distance_m": min(row["distance_m"] for row in reference_rows),
        "earliest_safe_suffix": earliest_safe,
        "segments_attempted": len(segments),
        "segments_accepted": sum(bool(row["accepted"]) for row in segments),
        "reference_anchor_monotonic": all(
            segments[i]["reference_anchor_time_s"] < segments[i + 1]["reference_anchor_time_s"]
            for i in range(len(segments) - 1)
        ),
        "tcp_y_progress_all_accepted": all(
            row.get("tcp_y_progress_ok", False) for row in segments if row["accepted"]
        ),
        "closure": closure,
        "segments": segments,
    }
    trial.write_json(output_dir / "offset_preserving_rollout.json", payload)
    return payload


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    compact = {
        "status": result["status"],
        "robot_commanded": result["robot_commanded"],
        "segments_attempted": result["segments_attempted"],
        "segments_accepted": result["segments_accepted"],
        "reference_anchor_monotonic": result["reference_anchor_monotonic"],
        "tcp_y_progress_all_accepted": result["tcp_y_progress_all_accepted"],
        "closure": result["closure"],
        "output": str(
            (args.output or args.source_run / "offline_static20_offset_preserving_rollout")
            .resolve()
            / "offset_preserving_rollout.json"
        ),
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False, default=trial.json_default))


if __name__ == "__main__":
    main()
