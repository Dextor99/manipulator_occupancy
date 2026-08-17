#!/usr/bin/env python3
"""Protected live pilot for one Fresh-authorized simple dynamic NUBS segment.

This wrapper intentionally reuses the mature 6.5.3 reference-motion, STRO stop,
Fresh authorization, offline-track and raw-cloud guard implementation.  Only
the r06-frozen planning policy is injected: fixed PCA two-sphere geometry,
three away-side risk-link goals, a 0.11 m coarse gate, and the unchanged Fast
CCRO-NUBS verifier.  One authorized 1 s local segment may execute; the robot
then remains at its measured local tail.  No automatic rejoin or goal motion is
permitted in this pilot.
"""

from __future__ import annotations

import argparse
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
simple = importlib.import_module("experiments.new.6_5.6_5_3.run_653_simple_dynamic_nubs_avoidance")
bypass = importlib.import_module("experiments.new.6_5.6_5_3.simple_bypass_planner")

DEFAULT_REFERENCE = ROOT / "results/new/6_5/6_5_3/reference_xp00_line/reference_feedback.csv"
DEFAULT_OUTPUT = ROOT / "results/new/6_5/6_5_3/simple_dynamic_nubs_live"
REFERENCE_OPERATOR_PHRASE = trial.REQUIRED_OPERATOR_PHRASE
LOCAL_EXECUTE_PHRASE = "CCRO_653_SIMPLE_DYNAMIC_LOCAL_EXECUTE_APPROVED"
ACTIVE_BASE_FAST_REPAIR = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-feedback-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--task-geometry-id", default="D2_END_EFFECTOR_OPPOSING_XP00")
    parser.add_argument("--reference-operator-phrase", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--operator-phrase", default="")
    parser.add_argument("--x-offset", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=18.0)
    parser.add_argument("--forward-m", type=float, default=0.05)
    parser.add_argument("--side-lengths-m", default="0.04,0.06,0.08")
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.12)
    parser.add_argument("--planning-robust-target-m", type=float, default=0.11)
    parser.add_argument("--tcp-link", default="gripper_base_link")
    return parser


def validate_request(args: argparse.Namespace) -> tuple[float, ...]:
    side_lengths = tuple(float(value) for value in args.side_lengths_m.split(","))
    if len(side_lengths) != 3 or any(value <= 0.0 for value in side_lengths):
        raise ValueError("side-lengths-m must contain exactly three positive values")
    if args.planning_robust_target_m < 0.11:
        raise ValueError("planning-robust-target-m must remain at least 0.11 m")
    if args.reference_operator_phrase != REFERENCE_OPERATOR_PHRASE:
        raise RuntimeError(
            f"bad reference operator phrase; required: {REFERENCE_OPERATOR_PHRASE}"
        )
    if args.execute and args.operator_phrase != LOCAL_EXECUTE_PHRASE:
        raise RuntimeError(f"bad local execute phrase; required: {LOCAL_EXECUTE_PHRASE}")
    return side_lengths


def fixed_two_sphere_adapter(
    points: np.ndarray, *, fit_margin_m: float = 0.005, max_components: int = 4
) -> dict[str, Any]:
    del max_components
    return simple.fit_fixed_pca_two_sphere(points, fit_margin_m=fit_margin_m)


def nominal_local_risk(
    runtime_args: Any,
    model: Any,
    evaluator: Any,
    forecast: Any,
    q_now: np.ndarray,
    reference_goal: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    head, tail, durations, inner, _ = trial.make_local_reference(
        q_now, np.zeros(6), runtime_args, reference_goal=reference_goal
    )
    trajectory = trial.NUBSTrajectory6D().generate(inner, head, tail, durations)
    best = None
    for tau in np.arange(0.0, trajectory.total_duration + 0.05, 0.10):
        tau = min(float(tau), float(trajectory.total_duration))
        q_tau = trajectory.evaluate(tau)
        risk = evaluator.configuration(
            q_tau, forecast, tau, density="medium", with_gradient=False
        )
        row = {
            "tau_s": tau,
            "distance_m": float(risk.min_distance),
            "nearest_link": risk.nearest_link,
            "risk_object": risk,
            "q_risk": np.asarray(q_tau, dtype=np.float64),
        }
        if best is None or row["distance_m"] < best["distance_m"]:
            best = row
    if best is None:
        raise RuntimeError("nominal local risk profile is empty")
    return best


def make_r06_fast_wrapper(
    original_fast: Any,
    *,
    side_lengths: tuple[float, ...],
    forward_m: float,
    max_joint_delta_rad: float,
    robust_target_m: float,
    tcp_link: str,
    required_component_count: int | None = 2,
    coarse_gate_is_hard: bool = True,
    clearance_improvement_is_hard: bool = True,
    verified_seed_is_candidate: bool = False,
):
    def run_r06_fast(
        runtime_args: Any,
        config: dict[str, Any],
        model: Any,
        *,
        q_now: np.ndarray,
        qd_now: np.ndarray,
        center: np.ndarray,
        velocity: np.ndarray,
        radius: float,
        risk_links: set[str],
        trial_dir: Path,
        reference_goal: tuple[np.ndarray, np.ndarray, np.ndarray],
        rejoin_goals: Any,
        obstacle_audit: dict[str, Any],
        multisphere_geometry: dict[str, Any] | None = None,
        artifacts_out: dict[str, Any] | None = None,
        forecast_override: Any | None = None,
    ) -> dict[str, Any]:
        del rejoin_goals, forecast_override
        component_count = int((multisphere_geometry or {}).get("component_count", 0))
        component_count_ok = multisphere_component_count_allowed(
            component_count, required_component_count
        ) and multisphere_geometry is not None
        if not component_count_ok:
            return {
                "status": "REJECTED_SIMPLE_LIVE_MULTISPHERE_REQUIRED",
                "local_repair_status": "REJECTED_SIMPLE_LIVE_MULTISPHERE_REQUIRED",
                "local_repair_ready": False,
                "accepted_for_switch": False,
                "rejection_reasons": ["current_live_fresh_multisphere_requirement_failed"],
                "simple_live_audit": {"fast_invoked": False},
            }
        forecast = trial.constant_multisphere_forecast(
            np.asarray(multisphere_geometry["component_centers"], dtype=np.float64),
            np.asarray(multisphere_geometry["component_base_radii"], dtype=np.float64),
            np.asarray(velocity, dtype=np.float64),
            object_id=int(obstacle_audit.get("track_id") or 1),
        )
        evaluator, _, _ = trial.make_risk_stack(config, model, forecast)
        best = nominal_local_risk(
            runtime_args, model, evaluator, forecast, np.asarray(q_now), reference_goal
        )
        risk = best["risk_object"]
        if risk.robot_point is None or risk.obstacle_point is None:
            return {
                "status": "REJECTED_SIMPLE_LIVE_MISSING_RISK_POINTS",
                "local_repair_status": "REJECTED_SIMPLE_LIVE_MISSING_RISK_POINTS",
                "local_repair_ready": False,
                "accepted_for_switch": False,
                "rejection_reasons": ["missing_ccro_surface_points"],
                "simple_live_audit": {"fast_invoked": False},
            }
        q_values = np.asarray(q_now, dtype=np.float64)
        tcp_now = simple.tcp_position(model, q_values, tcp_link)
        tcp_goal = simple.tcp_position(model, np.asarray(reference_goal[0]), tcp_link)
        goals, direction = bypass.risk_link_bypass_goal_candidates(
            model,
            q_values,
            tcp_position=tcp_now,
            goal_position=tcp_goal,
            risk_link=str(best["nearest_link"]),
            risk_position=np.asarray(risk.robot_point),
            predicted_obstacle_position=np.asarray(risk.obstacle_point),
            risk_point_q=np.asarray(best["q_risk"]),
            forward_m=forward_m,
            side_lengths_m=side_lengths,
            tcp_link=tcp_link,
            max_joint_delta_rad=max_joint_delta_rad,
        )
        rows = []
        task_direction = np.asarray(direction["task_direction"], dtype=np.float64)
        for index, item in enumerate(goals, 1):
            goal_state = (np.asarray(item["q_goal"]), np.zeros(6), np.zeros(6))
            head, tail, durations, inner, _ = trial.make_local_reference(
                q_values, np.zeros(6), runtime_args, reference_goal=goal_state
            )
            trajectory = trial.NUBSTrajectory6D().generate(inner, head, tail, durations)
            minimum, profile = simple.trajectory_minimum(evaluator, forecast, trajectory)
            tcp_end = simple.tcp_position(
                model, trajectory.evaluate(trajectory.total_duration), tcp_link
            )
            progress = float(np.dot(tcp_end - tcp_now, task_direction))
            rows.append(
                {
                    "candidate": index,
                    "side_sign": 1,
                    "side_m": float(item["side_m"]),
                    "forward_m": float(item["forward_m"]),
                    "mapping": item["mapping"],
                    "coarse_min_distance_m": float(minimum["distance_m"]),
                    "coarse_min_tau_s": float(minimum["tau_s"]),
                    "coarse_nearest_link": minimum["nearest_link"],
                    "task_progress_m": progress,
                    "task_progress_ok": bool(progress > 0.0),
                    "profile": profile,
                }
            )
        selected = select_planning_seed(
            rows,
            robust_target_m=robust_target_m,
            coarse_gate_is_hard=coarse_gate_is_hard,
        )
        audit = {
            "candidate_source": "generated_in_current_live_run_from_post_stop_actual_q",
            "q_actual_post_stop_rad": q_values.tolist(),
            "geometry_policy": (
                "fresh_adaptive_pca_multisphere"
                if required_component_count is None
                else f"fresh_fixed_pca_{required_component_count}_sphere"
            ),
            "direction": direction,
            "candidates": rows,
            "planning_robust_target_m": float(robust_target_m),
            "planning_robust_target_is_hard_gate": bool(coarse_gate_is_hard),
            "fast_invoked": selected is not None,
        }
        trial.write_json(Path(trial_dir) / "simple_live_bypass_audit.json", audit)
        if selected is None:
            return {
                "status": "REJECTED_NO_SIMPLE_LIVE_ROBUST_BYPASS",
                "local_repair_status": "REJECTED_NO_SIMPLE_LIVE_ROBUST_BYPASS",
                "local_repair_ready": False,
                "accepted_for_switch": False,
                "rejection_reasons": [
                    "no_task_progress_candidate"
                    if not coarse_gate_is_hard
                    else "no_coarse_candidate_at_or_above_robust_target"
                ],
                "simple_live_audit": audit,
            }
        selected_goal = goals[int(selected["candidate"]) - 1]
        # Keep the configured improvement as an optimizer preference.  V3
        # changes only the final acceptance contract: failure to improve an
        # already-safe seed is diagnostic, not an execution veto.
        fast_args = runtime_args
        result = original_fast(
            fast_args,
            config,
            model,
            q_now=q_values,
            qd_now=np.asarray(qd_now),
            center=np.asarray(center),
            velocity=np.asarray(velocity),
            radius=float(radius),
            risk_links=risk_links,
            trial_dir=trial_dir,
            reference_goal=(np.asarray(selected_goal["q_goal"]), np.zeros(6), np.zeros(6)),
            rejoin_goals=None,
            obstacle_audit={
                **obstacle_audit,
                "simple_dynamic_nubs_live": True,
                "candidate_source": audit["candidate_source"],
            },
            multisphere_geometry=multisphere_geometry,
            artifacts_out=artifacts_out,
            accept_verified_seed_without_fast_step=verified_seed_is_candidate,
            original_task_reference_goal=reference_goal,
        )
        audit.update(
            {
                "selected_candidate": int(selected["candidate"]),
                "selected_coarse_clearance_m": float(selected["coarse_min_distance_m"]),
                "selected_coarse_meets_preferred_target": bool(
                    selected["coarse_min_distance_m"] >= robust_target_m
                ),
                "clearance_improvement_is_hard_gate": bool(clearance_improvement_is_hard),
                "verified_seed_is_candidate": bool(verified_seed_is_candidate),
                "fast_status": result.get("status"),
                "final_candidate_source": result.get("candidate_source"),
            }
        )
        result["simple_live_audit"] = audit
        trial.write_json(Path(trial_dir) / "simple_live_bypass_audit.json", audit)
        return result

    return run_r06_fast


def multisphere_component_count_allowed(
    component_count: int, required_component_count: int | None
) -> bool:
    """Validate geometry cardinality without changing the V2 two-sphere default."""
    return component_count >= 1 and (
        required_component_count is None or component_count == required_component_count
    )


def select_planning_seed(
    rows: list[dict[str, Any]],
    *,
    robust_target_m: float,
    coarse_gate_is_hard: bool,
) -> dict[str, Any] | None:
    """Select the seed that is allowed to reach Fast.

    V2 retains the established 0.11 m hard gate.  V3 treats both the preferred
    clearance and task progress as ranking diagnostics: every geometrically
    generated seed may reach the unchanged absolute verifier.  Positive task
    progress is preferred, but a temporarily lateral/backward seed is not
    rejected solely for that reason.
    """
    if coarse_gate_is_hard:
        return simple.select_robust_candidate(rows, robust_target_m)
    return (
        max(
            rows,
            key=lambda row: (
                bool(row["task_progress_ok"]),
                row["coarse_min_distance_m"],
                row["task_progress_m"],
            ),
        )
        if rows
        else None
    )


def make_guarded_executor(original_executor: Any, live_trial_dir: Path):
    def guarded_executor(
        robot: Any,
        trajectory_csv: Path,
        runtime_args: argparse.Namespace,
        *,
        processor: Any,
        denoiser: Any,
        playback_duration_s: float | None = None,
        controller_period_s: float | None = None,
        execution_label: str = "authorized trajectory",
        guard_provider: Any | None = None,
        obstacle_state_provider: Any | None = None,
        motion_monitor_provider: Any | None = None,
    ) -> dict[str, Any]:
        candidate_path = Path(trajectory_csv).resolve()
        if not candidate_path.is_relative_to(live_trial_dir.resolve()):
            raise RuntimeError("external or prior-run candidate CSV is forbidden")
        guards = [
            float(guard_provider()["distance_m"])
            if callable(guard_provider)
            else trial.execution_hard_guard_distance(
                processor, denoiser, runtime_args
            )
            for _ in range(3)
        ]
        minimum_guard = float(min(guards))
        if minimum_guard <= runtime_args.guided_hard_stop_m:
            raise RuntimeError(
                f"pre-execution raw hard guard {minimum_guard:.4f} m is not above "
                f"{runtime_args.guided_hard_stop_m:.4f} m"
            )
        result = original_executor(
            robot,
            candidate_path,
            runtime_args,
            processor=processor,
            denoiser=denoiser,
            playback_duration_s=playback_duration_s,
            controller_period_s=controller_period_s,
            execution_label=execution_label,
            guard_provider=guard_provider,
            obstacle_state_provider=obstacle_state_provider,
            motion_monitor_provider=motion_monitor_provider,
        )
        result["pre_execution_hard_guard_samples_m"] = guards
        result["candidate_source"] = "generated_in_current_live_run"
        return result

    return guarded_executor


def copy_wrapper_runtime_parameters(wrapper_args: Any, core_args: Any) -> None:
    """Expose reviewed orchestration parameters to in-core callback hooks."""
    for name, default in (
        ("forward_m", 0.05),
        ("max_joint_delta_rad", 0.12),
        ("planning_robust_target_m", 0.11),
        ("tcp_link", "gripper_base_link"),
        ("continuation_side_m", 0.04),
        ("max_local_replans", 3),
        ("max_closed_loop_segments", 12),
        ("closed_loop_goal_tolerance_rad", 0.01),
    ):
        setattr(core_args, name, getattr(wrapper_args, name, default))


def run(args: argparse.Namespace) -> dict[str, Any]:
    global ACTIVE_BASE_FAST_REPAIR
    side_lengths = validate_request(args)
    output = args.output.resolve() / f"r{args.repeat:02d}"
    output.mkdir(parents=True, exist_ok=True)
    core_output = output / "core_live"
    core_trial_dir = (
        core_output / "trials" / f"D2_{trial.SCENARIOS['D2']['name']}_r{args.repeat:02d}"
    )
    log: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "INITIALIZED",
        "robot_commanded": False,
        "candidate_source_policy": "current_live_run_only",
        "local_only": True,
        "automatic_rejoin": False,
        "execute_requested": bool(args.execute),
        "required_reference_operator_phrase": REFERENCE_OPERATOR_PHRASE,
        "required_local_execute_phrase": LOCAL_EXECUTE_PHRASE,
        "output": str(output),
    }

    original_fit = trial.fit_pca_multisphere
    original_fast = trial.run_fast_repair
    original_select = trial.select_dynamic_execution_path
    original_delayed = trial.authorize_delayed_rejoin_after_fresh3
    original_executor = trial.execute_authorized_trajectory_offline_track
    original_rolling = trial.rolling_fast_until_authorized
    try:
        ACTIVE_BASE_FAST_REPAIR = original_fast
        live_args = trial.build_parser().parse_args(
            [
                "--scene", "D2",
                "--repeat", str(args.repeat),
                "--mode", "live-stop-replan-execute",
                "--reference-feedback-csv", str(args.reference_feedback_csv.resolve()),
                "--task-geometry-id", args.task_geometry_id,
                "--operator-phrase", REFERENCE_OPERATOR_PHRASE,
                "--output", str(core_output),
                "--duration-s", str(args.duration_s),
                "--x-offset", str(args.x_offset),
                "--y-start", "0.40",
                "--y-goal", "-0.40",
                "--line-velocity-m-s", "0.020",
                "--line-acc-m-s2", "0.05",
                "--candidate-playback-duration-s", "1.0",
            ]
            + (
                [
                    "--allow-live-candidate-execution",
                    "--live-execute-candidate-phrase",
                    trial.LIVE_CANDIDATE_EXECUTE_PHRASE,
                ]
                if args.execute
                else []
            )
        )
        # The core trial parser intentionally knows nothing about the simple
        # planner/V3 orchestration surface.  Playback and continuation hooks
        # execute inside the core call, so explicitly carry their reviewed
        # wrapper parameters into that Namespace instead of relying on parser
        # side effects.
        copy_wrapper_runtime_parameters(args, live_args)
        trial.fit_pca_multisphere = fixed_two_sphere_adapter
        trial.run_fast_repair = make_r06_fast_wrapper(
            original_fast,
            side_lengths=side_lengths,
            forward_m=float(args.forward_m),
            max_joint_delta_rad=float(args.max_joint_delta_rad),
            robust_target_m=float(args.planning_robust_target_m),
            tcp_link=args.tcp_link,
        )
        trial.select_dynamic_execution_path = lambda **kwargs: (
            "LOCAL_FIRST_DELAYED_REJOIN" if kwargs.get("local_authorized") else None
        )
        trial.authorize_delayed_rejoin_after_fresh3 = lambda *a, **k: (
            {
                "status": "SIMPLE_LIVE_LOCAL_TAIL_HOLD_REQUIRED",
                "authorized": False,
                "reason": "first_protected_live_pilot_forbids_automatic_rejoin",
            },
            None,
        )
        trial.rolling_fast_until_authorized = lambda *a, **k: {
            "status": "SIMPLE_LIVE_SINGLE_PLAN_FAIL_CLOSED_HOLD",
            "authorized": False,
            "reason": "first_protected_live_pilot_forbids_rolling_replans",
        }
        trial.execute_authorized_trajectory_offline_track = make_guarded_executor(
            original_executor, core_trial_dir
        )
        core = trial.run(live_args)
        log["core_status"] = core.get("status")
        log["core_trial_dir"] = str(core_trial_dir)
        log["robot_commanded"] = bool(core.get("robot_commanded", False))
        log["core_summary"] = str(core_trial_dir / "summary.json")
        event_types = [event.get("type") for event in core.get("events", [])]
        log["event_types"] = event_types
        if "V3_VIRTUAL_CLOSED_LOOP_GOAL_REACHED" in event_types:
            log["status"] = "DYNAMIC_NUBS_CLOSED_LOOP_GOAL_REACHED_SHADOW"
        elif "V3_VIRTUAL_PLAYBACK_SHADOW_PASS" in event_types:
            log["status"] = "SIMPLE_DYNAMIC_NUBS_V3_PLAYBACK_SHADOW_PASS"
        elif "V3_VIRTUAL_PLAYBACK_SHADOW_HOLD" in event_types:
            log["status"] = "SIMPLE_DYNAMIC_NUBS_V3_PLAYBACK_SHADOW_HOLD"
        elif "LIVE_LOCAL_REPAIR_EXECUTED_HOLD" in event_types:
            log["status"] = "SIMPLE_DYNAMIC_NUBS_LIVE_LOCAL_EXECUTED_HOLD"
        elif "LOCAL_EXECUTION_AUTHORIZED_SHADOW" in event_types and not args.execute:
            log["status"] = "SIMPLE_DYNAMIC_NUBS_LIVE_PLAN_AUTHORIZED_HOLD"
        elif not args.execute and "LIVE_CANDIDATE_EXECUTION_BLOCKED_BY_DEFAULT" in event_types:
            log["status"] = "SIMPLE_DYNAMIC_NUBS_LIVE_PLAN_HOLD"
        else:
            log["status"] = "SIMPLE_DYNAMIC_NUBS_LIVE_FAIL_CLOSED_HOLD"
    finally:
        ACTIVE_BASE_FAST_REPAIR = None
        trial.fit_pca_multisphere = original_fit
        trial.run_fast_repair = original_fast
        trial.select_dynamic_execution_path = original_select
        trial.authorize_delayed_rejoin_after_fresh3 = original_delayed
        trial.execute_authorized_trajectory_offline_track = original_executor
        trial.rolling_fast_until_authorized = original_rolling
        trial.write_json(output / "summary.json", log)
    return log


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
