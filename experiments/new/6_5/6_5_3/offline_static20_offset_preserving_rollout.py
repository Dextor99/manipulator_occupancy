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
    parser.add_argument("--risk-goal-max-delta-rad", type=float, default=0.030)
    parser.add_argument("--terminal-goal-max-step-rad", type=float, default=0.030)
    parser.add_argument("--goal-tolerance-rad", type=float, default=0.010)
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


def bounded_terminal_goal(
    q_now: np.ndarray,
    q_final: np.ndarray,
    *,
    max_step_rad: float,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    delta = np.asarray(q_final, dtype=np.float64) - np.asarray(q_now, dtype=np.float64)
    peak = float(np.max(np.abs(delta)))
    scale = 1.0 if peak <= max_step_rad else float(max_step_rad / peak)
    step = scale * delta
    target = np.asarray(q_now, dtype=np.float64) + step
    zeros = np.zeros(6, dtype=np.float64)
    return (target, zeros, zeros), {
        "terminal_goal_error_max_abs_rad": peak,
        "terminal_step_scale": scale,
        "terminal_step_rad": step.tolist(),
        "transported_goal_rad": target.tolist(),
    }


def tcp_position(model: Any, q: np.ndarray, link: str) -> np.ndarray:
    transforms = model.urdf.link_transforms(
        {name: float(q[index]) for index, name in enumerate(model.joint_names)}
    )
    if link not in transforms:
        raise KeyError(f"TCP link {link!r} is absent from the robot model")
    return np.asarray(transforms[link][:3, 3], dtype=np.float64)


def risk_guided_goal(
    evaluator: Any,
    forecast: Any,
    nominal_goal: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    max_delta_rad: float,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray] | None, dict[str, Any]]:
    """Move a transported local goal a bounded step down the risk-cost gradient."""
    q_goal, qd_goal, qdd_goal = nominal_goal
    risk = evaluator.configuration(
        np.asarray(q_goal, dtype=np.float64),
        forecast,
        0.0,
        density="medium",
        with_gradient=True,
    )
    gradient = np.asarray(risk.gradient_q, dtype=np.float64)
    peak = float(np.max(np.abs(gradient)))
    audit = {
        "nominal_goal_distance_m": float(risk.min_distance),
        "nearest_link": risk.nearest_link,
        "risk_cost": float(risk.cost),
        "gradient_q": gradient.tolist(),
        "gradient_inf_norm": peak,
        "max_delta_rad": float(max_delta_rad),
    }
    if peak <= 1.0e-12 or max_delta_rad <= 0.0:
        audit["available"] = False
        audit["reason"] = "zero_risk_gradient_or_zero_bound"
        return None, audit
    delta = -gradient * (float(max_delta_rad) / peak)
    guided_q = np.asarray(q_goal, dtype=np.float64) + delta
    guided_risk = evaluator.configuration_clearance(
        guided_q, forecast, 0.0, density="medium"
    )
    audit.update(
        {
            "available": True,
            "delta_q_risk_rad": delta.tolist(),
            "delta_q_risk_max_abs_rad": float(np.max(np.abs(delta))),
            "linear_probe_goal_distance_m": float(guided_risk.min_distance),
            "linear_probe_improvement_m": float(guided_risk.min_distance - risk.min_distance),
        }
    )
    if guided_risk.min_distance <= risk.min_distance:
        audit["available"] = False
        audit["reason"] = "bounded_gradient_step_did_not_improve_goal_clearance"
        return None, audit
    return (guided_q, np.asarray(qd_goal), np.asarray(qdd_goal)), audit


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
    if args.risk_goal_max_delta_rad < 0.0:
        raise ValueError("risk goal bound must be non-negative")
    if args.terminal_goal_max_step_rad <= 0.0 or args.goal_tolerance_rad <= 0.0:
        raise ValueError("terminal goal step and tolerance must be positive")
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
    reference_segment_limit = int(
        math.ceil((float(reference.times[-1]) - anchor) / runtime_args.local_horizon_s)
    )
    q_final = reference.state_at(float(reference.times[-1]))[0]
    # A bounded terminal phase is permitted after the recorded task clock ends;
    # its cap is derived from the actual remaining joint displacement.
    initial_terminal_bound = int(
        math.ceil(float(np.max(np.abs(q_final - q_virtual))) / args.terminal_goal_max_step_rad)
    )
    max_segments = reference_segment_limit + initial_terminal_bound + 2
    locked_side = None
    segments = []
    closure = None
    status = "OFFSET_PRESERVING_ROLLOUT_REACHED_REFERENCE_END_WITHOUT_CLOSURE"
    goal_reached = False

    for index in range(1, max_segments + 1):
        final_error = float(np.max(np.abs(q_final - q_virtual)))
        if final_error <= args.goal_tolerance_rad:
            goal_reached = True
            status = "GOAL_DIRECTED_ROLLING_FAST_SUCCESS"
            break
        remaining_reference = float(reference.times[-1]) - anchor
        if remaining_reference > 1.0e-6:
            local_goal, goal_audit = transported_reference_goal(
                reference, q_virtual, anchor, runtime_args.local_horizon_s
            )
            reference_increment = np.asarray(goal_audit["reference_increment_rad"])
            if float(np.max(np.abs(reference_increment))) >= runtime_args.min_local_motion_rad:
                progress_phase = "reference_transport"
            else:
                anchor = float(reference.times[-1])
                local_goal, goal_audit = bounded_terminal_goal(
                    q_virtual, q_final, max_step_rad=args.terminal_goal_max_step_rad
                )
                goal_audit.update(
                    {
                        "reference_anchor_time_s": anchor,
                        "reference_goal_time_s": anchor,
                    }
                )
                progress_phase = "terminal_goal"
        else:
            local_goal, goal_audit = bounded_terminal_goal(
                q_virtual, q_final, max_step_rad=args.terminal_goal_max_step_rad
            )
            goal_audit.update(
                {
                    "reference_anchor_time_s": float(reference.times[-1]),
                    "reference_goal_time_s": float(reference.times[-1]),
                }
            )
            progress_phase = "terminal_goal"
        segment_dir = output_dir / f"segment_{index:02d}"

        def run_goal_attempt(
            label: str,
            goal: tuple[np.ndarray, np.ndarray, np.ndarray],
            *,
            side_delta_override: np.ndarray | None = None,
        ):
            attempt_artifacts: dict[str, Any] = {}
            attempt_result = trial.run_fast_repair(
                runtime_args,
                config,
                model,
                q_now=q_virtual,
                qd_now=np.zeros(6),
                center=np.asarray(geometry["component_centers"], dtype=np.float64).mean(axis=0),
                velocity=np.zeros(3),
                radius=float(max(geometry["component_base_radii"])),
                risk_links=set(model.surface_by_link(q_virtual, density="coarse")),
                trial_dir=segment_dir / label,
                reference_goal=goal,
                rejoin_goals=None,
                obstacle_audit={
                    "offline_static20_offset_preserving_rollout": True,
                    "offline_forecast_override_authorized": True,
                    "goal_mode": label,
                    "segment": index,
                    "static_observation_inflation_m": float(args.static_inflation_m),
                },
                multisphere_geometry=geometry,
                artifacts_out=attempt_artifacts,
                forecast_override=forecast,
            )
            attempt_candidate = attempt_artifacts.get("candidate_trajectory")
            attempt_side = trial.avoidance_side_consistent(
                locked_side,
                (
                    np.asarray(side_delta_override, dtype=np.float64)
                    if side_delta_override is not None
                    else np.asarray(attempt_result["tail_delta_q_rad"], dtype=np.float64)
                ),
                opposite_projection_tolerance_rad=runtime_args.rolling_side_opposite_tolerance_rad,
            )
            side_lock_released = bool(
                progress_phase == "terminal_goal"
                and attempt_result["candidate_online_min_distance_m"] >= runtime_args.online_accept_m
                and all(
                    ok
                    for name, ok in attempt_result["verification_checks"].items()
                    if name != "solver_ok"
                )
            )
            if side_lock_released and not attempt_side["accepted"]:
                attempt_side = {
                    **attempt_side,
                    "accepted": True,
                    "reason": "side_lock_released_for_verified_terminal_goal",
                    "originally_accepted": False,
                }
            fast_repair_accepted = bool(
                attempt_result["local_repair_ready"]
                and attempt_side["accepted"]
                and attempt_candidate is not None
            )
            verified_nominal_accepted = bool(
                attempt_candidate is not None
                and attempt_side["accepted"]
                and attempt_result["online_pipeline_elapsed_ms"] <= runtime_args.fast_budget_ms
                and attempt_result["candidate_online_min_distance_m"] >= runtime_args.online_accept_m
                and all(
                    ok
                    for name, ok in attempt_result["verification_checks"].items()
                    if name != "solver_ok"
                )
            )
            attempt_accepted = fast_repair_accepted or verified_nominal_accepted
            acceptance_mode = (
                "fast_repaired_candidate"
                if fast_repair_accepted
                else f"verified_{label}_nominal_segment"
                if verified_nominal_accepted
                else "rejected"
            )
            return (
                attempt_result,
                attempt_artifacts,
                attempt_candidate,
                attempt_side,
                attempt_accepted,
                acceptance_mode,
            )

        result, artifacts, candidate, side, accepted, acceptance_mode = run_goal_attempt(
            "transported_task_goal", local_goal
        )
        transported_result = result
        selected_trajectory_verification = None
        exact_terminal_target = bool(
            progress_phase == "terminal_goal"
            and float(goal_audit.get("terminal_step_scale", 0.0)) >= 1.0 - 1.0e-12
        )
        if exact_terminal_target:
            nominal = artifacts["reference_trajectory"]
            nominal_samples = nominal.sample(np.asarray([0.0, nominal.total_duration]))
            nominal_verification = verifier.verify(
                nominal,
                forecast,
                current_q=nominal_samples.q[0],
                current_qd=nominal_samples.qd[0],
                current_qdd=nominal_samples.qdd[0],
                q_goal=q_final,
                solver_success=True,
            )
            selected_trajectory_verification = asdict(nominal_verification)
            if nominal_verification.accepted:
                candidate = nominal
                accepted = True
                acceptance_mode = "verified_exact_terminal_goal_nominal_segment"
                side = {
                    **side,
                    "accepted": True,
                    "reason": "side_lock_released_for_verified_exact_terminal_goal",
                }
        goal_mode = "transported_task_goal"
        risk_goal_audit = None
        fallback_result = None
        if not accepted:
            guided_goal, risk_goal_audit = risk_guided_goal(
                evaluator,
                forecast,
                local_goal,
                max_delta_rad=args.risk_goal_max_delta_rad,
            )
            if guided_goal is not None:
                risk_delta = np.asarray(risk_goal_audit["delta_q_risk_rad"], dtype=np.float64)
                (
                    fallback_result,
                    fallback_artifacts,
                    fallback_candidate,
                    fallback_side,
                    fallback_accepted,
                    fallback_acceptance_mode,
                ) = run_goal_attempt(
                    "risk_guided_goal", guided_goal, side_delta_override=risk_delta
                )
                if fallback_accepted:
                    result, artifacts, candidate, side, accepted, acceptance_mode = (
                        fallback_result,
                        fallback_artifacts,
                        fallback_candidate,
                        fallback_side,
                        fallback_accepted,
                        fallback_acceptance_mode,
                    )
                    goal_mode = "risk_guided_goal"
        row: dict[str, Any] = {
            "segment": index,
            "progress_phase": progress_phase,
            **goal_audit,
            "q_virtual_start": q_virtual.tolist(),
            "fast": result,
            "goal_mode": goal_mode,
            "acceptance_mode": acceptance_mode,
            "risk_guided_goal_audit": risk_goal_audit,
            "selected_trajectory_verification": selected_trajectory_verification,
            "transported_goal_failure": (
                None if fallback_result is None else {
                    "candidate_online_min_distance_m": float(
                        transported_result["candidate_online_min_distance_m"]
                    ),
                    "rejection_reasons": transported_result["rejection_reasons"],
                }
            ),
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
        goal_error_before = float(np.max(np.abs(q_final - q_virtual)))
        goal_error_after = float(np.max(np.abs(q_final - q_next)))
        goal_progress = goal_error_before - goal_error_after
        progress_ok = bool(
            progress_y > 0.0
            if progress_phase == "reference_transport"
            else goal_progress > 0.0
        )
        row.update(
            {
                "status": "SEGMENT_ACCEPTED",
                "q_virtual_end": q_next.tolist(),
                "tcp_start_m": tcp_start.tolist(),
                "tcp_end_m": tcp_end.tolist(),
                "tcp_y_task_progress_m": progress_y,
                "tcp_y_progress_ok": bool(progress_y > 0.0),
                "goal_error_before_max_abs_rad": goal_error_before,
                "goal_error_after_max_abs_rad": goal_error_after,
                "terminal_goal_progress_rad": goal_progress,
                "phase_progress_ok": progress_ok,
            }
        )
        if not progress_ok:
            row["status"] = "SAFE_HOLD_NO_PHASE_PROGRESS"
            segments.append(row)
            status = "OFFSET_PRESERVING_ROLLOUT_SAFE_HOLD_NO_PROGRESS"
            break
        if locked_side is None:
            locked_side = np.asarray(side["locked_tail_delta_q"], dtype=np.float64)

        next_anchor = (
            min(float(reference.times[-1]), anchor + runtime_args.local_horizon_s)
            if progress_phase == "reference_transport"
            else float(reference.times[-1])
        )
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
        "changed_experimental_variable": "transported_local_goal_with_bounded_risk_fallback",
        "risk_guided_fallback": {
            "enabled": True,
            "activation": "only_after_transported_task_goal_rejection",
            "maximum_joint_goal_delta_rad": float(args.risk_goal_max_delta_rad),
            "direction": "negative_configuration_risk_cost_gradient",
        },
        "avoidance_side_release_policy": (
            "release only in terminal_goal phase after the complete candidate verifier passes"
        ),
        "source_run": str(source),
        "source_points": str(points_path),
        "reference_feedback_csv": str(args.reference_feedback_csv.resolve()),
        "static_observation_inflation_m": float(args.static_inflation_m),
        "online_accept_m": float(runtime_args.online_accept_m),
        "local_horizon_s": float(runtime_args.local_horizon_s),
        "maximum_segments_from_reference_duration": max_segments,
        "reference_transport_segment_limit": reference_segment_limit,
        "terminal_goal_max_step_rad": float(args.terminal_goal_max_step_rad),
        "goal_tolerance_rad": float(args.goal_tolerance_rad),
        "goal_reached": goal_reached,
        "final_goal_error_max_abs_rad": float(np.max(np.abs(q_final - q_virtual))),
        "task_y_direction": task_y_direction,
        "reference_min_distance_m": min(row["distance_m"] for row in reference_rows),
        "earliest_safe_suffix": earliest_safe,
        "segments_attempted": len(segments),
        "segments_accepted": sum(bool(row["accepted"]) for row in segments),
        "reference_anchor_monotonic": all(
            segments[i]["reference_anchor_time_s"] <= segments[i + 1]["reference_anchor_time_s"]
            for i in range(len(segments) - 1)
        ),
        "tcp_y_progress_all_accepted": all(
            row.get("tcp_y_progress_ok", False)
            for row in segments
            if row["accepted"] and row["progress_phase"] == "reference_transport"
        ),
        "phase_progress_all_accepted": all(
            row.get("phase_progress_ok", False) for row in segments if row["accepted"]
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
