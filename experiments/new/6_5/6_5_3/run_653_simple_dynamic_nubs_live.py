#!/usr/bin/env python3
"""Protected live pilot for one Fresh-authorized simple dynamic NUBS segment.

This wrapper intentionally reuses the mature 6.5.3 reference-motion, STRO stop,
Fresh authorization, offline-track and raw-cloud guard implementation.  Only
the r06-frozen planning policy is injected: fixed PCA two-sphere geometry,
three away-side risk-link goals, a 0.11 m preferred coarse target, and the
unchanged Fast CCRO-NUBS verifier.  One authorized 1 s local segment may
execute; the robot then remains at its measured local tail.  No automatic
rejoin or goal motion is permitted in this pilot.
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
v3 = importlib.import_module("experiments.new.6_5.6_5_3.dynamic_nubs_v3")

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
    parser.add_argument(
        "--candidate-playback-duration-s",
        type=float,
        default=1.0,
        help="physical execution duration; the scaled trajectory is re-verified before authorization",
    )
    parser.add_argument(
        "--allow-experimental-playback-duration",
        action="store_true",
        help="forward an explicitly bounded 0.80-1.00 s time-scale experiment",
    )
    parser.add_argument(
        "--stro-trigger-horizon-s",
        type=float,
        default=1.2,
        help=(
            "initial STRO early-warning horizon only; "
            "Fast/Fresh/rolling execution horizon remains 0.5 s"
        ),
    )
    parser.add_argument(
        "--guidance-horizon-s",
        type=float,
        default=1.5,
        help=(
            "longer diagnostic horizon for swept-obstacle guidance; it does not "
            "extend the 1 s executable candidate or change any safety gate"
        ),
    )
    return parser


def validate_request(args: argparse.Namespace) -> tuple[float, ...]:
    side_lengths = tuple(float(value) for value in args.side_lengths_m.split(","))
    if len(side_lengths) != 3 or any(value <= 0.0 for value in side_lengths):
        raise ValueError("side-lengths-m must contain exactly three positive values")
    if args.planning_robust_target_m < 0.11:
        raise ValueError("planning-robust-target-m must remain at least 0.11 m")
    if args.guidance_horizon_s < args.candidate_playback_duration_s:
        raise ValueError("guidance-horizon-s must not be shorter than candidate playback")
    if args.allow_experimental_playback_duration and not 0.80 <= args.candidate_playback_duration_s <= 1.00:
        raise ValueError("experimental candidate playback must be within 0.80-1.00 s")
    if not args.allow_experimental_playback_duration and not np.isclose(
        args.candidate_playback_duration_s, 1.0, atol=1.0e-12
    ):
        raise ValueError("non-default playback requires --allow-experimental-playback-duration")
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
    tabletop_parallel_bypass: bool = False,
    preserve_tcp_height: bool = False,
    tabletop_seed_gate_is_hard: bool = False,
    max_fast_height_drop_vs_seed_m: float | None = None,
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
        forecast = v3.v3_execution_multisphere_forecast(
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
            tabletop_parallel_side=tabletop_parallel_bypass,
            preserve_tcp_height=preserve_tcp_height,
        )
        rows = []
        task_direction = np.asarray(direction["task_direction"], dtype=np.float64)
        for index, item in enumerate(goals, 1):
            goal_state = (np.asarray(item["q_goal"]), np.zeros(6), np.zeros(6))
            head, tail, durations, inner, _ = trial.make_local_reference(
                q_values, np.zeros(6), runtime_args, reference_goal=goal_state
            )
            trajectory = trial.NUBSTrajectory6D().generate(inner, head, tail, durations)
            tabletop_guard = trial.gripper_base_workspace_guard(
                trajectory, model, min_z_m=float(getattr(runtime_args, "gripper_base_min_z_m", 0.46))
            )
            minimum, profile = simple.trajectory_minimum(evaluator, forecast, trajectory)
            tcp_end = simple.tcp_position(
                model, trajectory.evaluate(trajectory.total_duration), tcp_link
            )
            progress = float(np.dot(tcp_end - tcp_now, task_direction))
            end_risk = evaluator.configuration(
                trajectory.evaluate(trajectory.total_duration),
                forecast,
                float(trajectory.total_duration),
                density="coarse",
                with_gradient=False,
            )
            guide_risk = evaluator.configuration(
                trajectory.evaluate(trajectory.total_duration),
                forecast,
                float(getattr(runtime_args, "guidance_horizon_s", 1.5)),
                density="coarse",
                with_gradient=False,
            )
            end_clearance = float(end_risk.min_distance)
            min_clearance = float(minimum["distance_m"])
            min_tau = float(minimum["tau_s"])
            rows.append(
                {
                    "candidate": index,
                    "side_sign": 1,
                    "side_m": float(item["side_m"]),
                    "forward_m": float(item["forward_m"]),
                    "mapping": item["mapping"],
                    "tabletop_workspace_guard": tabletop_guard,
                    "tabletop_feasible": bool(tabletop_guard["passed"]),
                    "coarse_min_distance_m": min_clearance,
                    "coarse_min_tau_s": min_tau,
                    "coarse_nearest_link": minimum["nearest_link"],
                    "coarse_end_clearance_m": end_clearance,
                    "coarse_end_minus_min_clearance_m": end_clearance - min_clearance,
                    "coarse_min_tau_fraction": min_tau / max(float(trajectory.total_duration), 1e-9),
                    "coarse_closest_approach_before_tail": bool(
                        min_tau < 0.8 * float(trajectory.total_duration)
                    ),
                    "guide_horizon_s": float(getattr(runtime_args, "guidance_horizon_s", 1.5)),
                    "guide_clearance_m": float(guide_risk.min_distance),
                    "guide_nearest_link": guide_risk.nearest_link,
                    "task_progress_m": progress,
                    "task_progress_ok": bool(progress > 0.0),
                    "profile": profile,
                }
            )
        selected = select_planning_seed(
            rows,
            robust_target_m=robust_target_m,
            coarse_gate_is_hard=coarse_gate_is_hard,
            tabletop_gate_is_hard=tabletop_seed_gate_is_hard,
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
            "guidance_horizon_s": float(getattr(runtime_args, "guidance_horizon_s", 1.5)),
            "guidance_policy": "long_horizon_tail_clearance_soft_preference_only",
            "fast_invoked": selected is not None,
            "tabletop_seed_gate_is_hard": bool(tabletop_seed_gate_is_hard),
            "tabletop_safe_candidate_count": int(sum(bool(r.get("tabletop_feasible", False)) for r in rows)),
            "gripper_base_min_z_m": float(getattr(runtime_args, "gripper_base_min_z_m", 0.46)),
        }
        trial.write_json(Path(trial_dir) / "simple_live_bypass_audit.json", audit)
        if selected is None:
            if tabletop_seed_gate_is_hard and audit["tabletop_safe_candidate_count"] == 0:
                status = "REJECTED_NO_TABLETOP_SAFE_BYPASS_SEED"
                reason = "no_tabletop_safe_bypass_seed"
            else:
                status = "REJECTED_NO_SIMPLE_LIVE_ROBUST_BYPASS"
                reason = "no_task_progress_candidate" if not coarse_gate_is_hard else "no_coarse_candidate_at_or_above_robust_target"
            return {
                "status": status,
                "local_repair_status": status,
                "local_repair_ready": False,
                "accepted_for_switch": False,
                "rejection_reasons": [
                    reason
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
        if result.get("local_repair_ready") and artifacts_out is not None:
            candidate_trajectory = artifacts_out.get("candidate_trajectory")
            guard = None if candidate_trajectory is None else trial.gripper_base_workspace_guard(
                candidate_trajectory, model, min_z_m=float(getattr(runtime_args, "gripper_base_min_z_m", 0.46))
            )
            result["planning_tabletop_guard"] = guard
            if guard is None or not guard.get("passed", False):
                result["status"] = "FAST_CANDIDATE_TABLETOP_GUARD_FAILED"
                result["local_repair_status"] = result["status"]
                result["local_repair_ready"] = False
                result["accepted_for_switch"] = False
                result.setdefault("rejection_reasons", []).append("gripper_base_below_tabletop_guard")
        result = apply_tabletop_height_shape_policy(
            result=result,
            artifacts_out=artifacts_out,
            runtime_args=runtime_args,
            config=config,
            model=model,
            forecast=forecast,
            q_now=q_values,
            qd_now=np.asarray(qd_now),
            trial_dir=Path(trial_dir),
            max_drop_m=max_fast_height_drop_vs_seed_m,
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
        result["guidance_horizon_s"] = float(getattr(runtime_args, "guidance_horizon_s", 1.5))
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


def apply_tabletop_height_shape_policy(
    *,
    result: dict[str, Any],
    artifacts_out: dict[str, Any] | None,
    runtime_args: Any,
    config: dict[str, Any],
    model: Any,
    forecast: Any,
    q_now: np.ndarray,
    qd_now: np.ndarray,
    trial_dir: Path,
    max_drop_m: float | None,
) -> dict[str, Any]:
    """Prefer a verified tabletop-safe seed when Fast creates a deep dip."""
    if not result.get("local_repair_ready") or artifacts_out is None or max_drop_m is None:
        return result
    candidate = artifacts_out.get("candidate_trajectory")
    seed = artifacts_out.get("reference_trajectory")
    if candidate is None or seed is None:
        return result
    threshold = float(getattr(runtime_args, "gripper_base_min_z_m", 0.46))
    candidate_guard = trial.gripper_base_workspace_guard(candidate, model, min_z_m=threshold)
    seed_guard = trial.gripper_base_workspace_guard(seed, model, min_z_m=threshold)
    result["planning_tabletop_guard"] = candidate_guard
    result["fast_seed_tabletop_guard"] = seed_guard
    drop = max(0.0, float(seed_guard.get("min_gripper_base_z_m", float("-inf"))) - float(candidate_guard.get("min_gripper_base_z_m", float("inf"))))
    result["fast_height_drop_vs_seed_m"] = drop
    result["max_fast_height_drop_vs_seed_m"] = float(max_drop_m)
    if drop <= float(max_drop_m):
        return result
    _, verifier, _ = trial.make_risk_stack(config, model, forecast)
    seed_goal = np.asarray(seed.evaluate(seed.total_duration), dtype=np.float64)
    verification = verifier.verify(seed, forecast, current_q=np.asarray(q_now), current_qd=np.asarray(qd_now), current_qdd=np.zeros(6), q_goal=seed_goal, solver_success=True)
    result["height_fallback_seed_verification"] = {
        "accepted": bool(verification.accepted),
        "min_distance_m": float(verification.min_distance),
        "checks": verification.checks,
        "reasons": verification.reasons,
    }
    if seed_guard.get("passed") and verification.accepted and float(verification.min_distance) >= float(getattr(runtime_args, "online_accept_m", 0.09)):
        result["fast_optimizer_candidate_source"] = result.get("candidate_source")
        result["fast_optimizer_candidate_online_min_distance_m"] = result.get("candidate_online_min_distance_m")
        result["fast_optimizer_candidate_csv"] = result.get("candidate_csv")
        artifacts_out["candidate_trajectory"] = seed
        artifacts_out["local_tail_state"] = getattr(seed, "tail_state", None)
        fallback_csv = Path(trial_dir) / "candidate" / "height_preserving_seed_fallback.csv"
        fallback_csv.parent.mkdir(parents=True, exist_ok=True)
        trial.save_trajectory_csv(fallback_csv, seed, dt=0.01)
        result["candidate_source"] = "VERIFIED_TABLETOP_BYPASS_SEED"
        result["verification_min_distance_m"] = float(verification.min_distance)
        result["candidate_online_min_distance_m"] = float(verification.min_distance)
        result["verification_checks"] = verification.checks
        result["verification_reasons"] = verification.reasons
        result["candidate_csv"] = str(fallback_csv)
        result["height_preserving_seed_fallback"] = True
        result["selected_execution_candidate_source"] = "VERIFIED_TABLETOP_BYPASS_SEED"
        result["selected_execution_candidate_csv"] = str(fallback_csv)
        result["fast_extra_correction_used_for_execution"] = False
        return result
    result["local_repair_ready"] = False
    result["accepted_for_switch"] = False
    result["status"] = "HEIGHT_CORRIDOR_VIOLATION_WITHOUT_SAFE_SEED_FALLBACK"
    result["local_repair_status"] = result["status"]
    result.setdefault("rejection_reasons", []).append("fast_height_shape_violation")
    return result


def select_planning_seed(
    rows: list[dict[str, Any]],
    *,
    robust_target_m: float,
    coarse_gate_is_hard: bool,
    tabletop_gate_is_hard: bool = False,
) -> dict[str, Any] | None:
    """Select the seed that is allowed to reach Fast.

    V2 retains the established 0.11 m hard gate.  V3 treats both the preferred
    clearance and task progress as ranking diagnostics: every geometrically
    generated seed may reach the unchanged absolute verifier.  Positive task
    progress is preferred, but a temporarily lateral/backward seed is not
    rejected solely for that reason.
    """
    eligible_rows = [r for r in rows if not tabletop_gate_is_hard or bool(r.get("tabletop_feasible", False))]
    if coarse_gate_is_hard:
        return simple.select_robust_candidate(eligible_rows, robust_target_m)
    return (
        max(
            eligible_rows,
            key=lambda row: (
                bool(row["task_progress_ok"]),
                bool(row.get("coarse_closest_approach_before_tail", False)),
                bool(row.get("coarse_end_minus_min_clearance_m", 0.0) > 0.0),
                float(row.get("guide_clearance_m", -np.inf)),
                row["coarse_min_distance_m"],
                row["task_progress_m"],
            ),
        )
        if eligible_rows
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
        ("rolling_replan_m", 0.09),
        ("rolling_preplan_trigger_s", 0.40),
        ("rolling_preplan_clearance_m", 0.12),
        ("rolling_preplan_min_lead_s", 0.25),
        ("final_precommand_fresh_timeout_s", 0.35),
        ("final_precommand_max_state_age_s", 0.35),
        ("boundary_qd_tol_rad_s", 0.03),
        ("boundary_qdd_tol_rad_s2", 0.30),
        # Initial STRO early warning only; execution remains 0.5 s.
        ("stro_trigger_horizon_s", 1.2),
        ("max_local_replans", 3),
        ("max_closed_loop_segments", 12),
        ("closed_loop_goal_tolerance_rad", 0.01),
        ("guidance_horizon_s", 1.5),
        ("stationary_terminal_full_plan", False),
        ("stationary_fast_goal_directed", False),
        ("command_time_fast_retry", False),
        ("stationary_center_span_m", 0.02),
        ("shadow_hold_observation_s", 0.0),
        ("stationary_fast_terminal_bypass", False),
        ("stationary_fast_terminal_duration_s", 6.0),
        ("stationary_fast_terminal_segments", 8),
        ("stationary_fast_terminal_rollout_steps", 4),
        ("stationary_fast_terminal_target_ms", 500.0),
        ("stationary_fast_terminal_max_ms", 1000.0),
        ("stationary_fast_terminal_virtual_max_joint_delta_rad", 0.30),
        ("stationary_fast_terminal_virtual_fast_steps", 6),
        ("stationary_fast_terminal_samples_per_local", 3),
        ("stationary_fast_terminal_route_max_ms", 5000.0),
        ("stationary_virtual_topology_floor_m", 0.08),
        ("stationary_boundary_terminal", True),
        ("stationary_legacy_virtual_fast_fallback", False),
        ("stationary_boundary_direction_count", 8),
        ("stationary_boundary_max_escape_steps", 8),
        ("stationary_boundary_max_pass_steps", 8),
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
                "--candidate-playback-duration-s", str(args.candidate_playback_duration_s),
                "--stro-trigger-horizon-s", str(args.stro_trigger_horizon_s),
                "--fast-target-ms", "150",
                "--fast-max-ms", "250",
            ]
            + (
                ["--allow-experimental-playback-duration"]
                if args.allow_experimental_playback_duration
                else []
            )
            + (
                [
                    "--allow-live-candidate-execution",
                    "--live-execute-candidate-phrase",
                    trial.LIVE_CANDIDATE_EXECUTE_PHRASE,
                    # The experiment-level confirmation has already happened
                    # before reference motion.  Do not pause after Fresh/Fast:
                    # the obstacle must be evaluated and commanded without a
                    # human-induced stale-state delay.  The executor still
                    # enforces all software, raw-guard, and tracker gates.
                    "--no-candidate-execute-confirm",
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
        # Final-live must use the split upload/start contract so the second
        # perception barrier remains immediately before MoveStartup.  Shadow
        # runs retain compatibility with older combined-only SDKs.
        live_args.require_split_offline_track = bool(args.execute)
        trial.fit_pca_multisphere = fixed_two_sphere_adapter
        trial.run_fast_repair = make_r06_fast_wrapper(
            original_fast,
            side_lengths=side_lengths,
            forward_m=float(args.forward_m),
            max_joint_delta_rad=float(args.max_joint_delta_rad),
            robust_target_m=float(args.planning_robust_target_m),
            tcp_link=args.tcp_link,
            # 0.11 m remains the preferred robust seed target, but the
            # unchanged Fast/online verifier must get a chance to repair a
            # lower-clearance seed (r03 best seed was 0.0817 m).
            coarse_gate_is_hard=False,
            # V3: optimizer-quality improvement is diagnostic only; absolute
            # verifier safety governs execution eligibility.
            clearance_improvement_is_hard=False,
            verified_seed_is_candidate=True,
            tabletop_parallel_bypass=True,
            preserve_tcp_height=True,
            tabletop_seed_gate_is_hard=True,
            max_fast_height_drop_vs_seed_m=0.005,
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
