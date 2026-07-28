"""Stage-A candidate-planning replay for the Chapter 6.4 boundary experiment.

This script isolates candidate generation from the closed-loop pending state
machine.  It replays fixed trigger/switch states and compares Critical-NUBS
and CCRO-NUBS under identical forecasts, budgets, continuity checks, and
online/dense validation.

Key design:
  - Independent 4 s local trajectory segments (no global reference dependency)
  - Observed forecast (GT + frozen errors)  → optimizer + online verifier
  - Pure GT forecast (no uncertainty inflation) → dense GT verification
  - Multi-level funnel with independent GT diagnostics
  - Uniform post-switch conflict time (POST_SWITCH_CONFLICT_TIME = 1.5 s)
  - Same frozen observation errors across all lead variants
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from planning.nubs_trajectory import NUBSTrajectory6D
from planning.obstacle_forecast import ShiftedForecast
from planning.verifier import DynamicTrajectoryVerifier

from . import config_64 as cfg
from .common_64 import (
    constant_forecast,
    ground_truth_forecast,
    load_stage4_config,
    load_surface_model,
    make_critical_risk_stack,
    make_reference,
    make_risk_stack,
    manifest_meta,
    optimize_candidate,
    write_json,
)
from .scenarios_64 import generate_instances, obstacle_center_at, obstacle_velocity_at


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def _build_local_reference(instance: dict[str, Any]) -> NUBSTrajectory6D:
    """Reconstruct the 4 s local reference trajectory from instance data."""
    return NUBSTrajectory6D().generate(
        np.asarray(instance["local_reference_inner"], dtype=np.float64),
        np.asarray(instance["local_head_state"], dtype=np.float64),
        np.asarray(instance["local_tail_state"], dtype=np.float64),
        np.asarray(instance["local_durations"], dtype=np.float64),
    )


def _reference_local_min_distance(
    evaluator,
    local_reference: NUBSTrajectory6D,
    forecast,
    duration: float,
    *,
    density: str,
) -> float:
    """Evaluate reference trajectory min distance over the local horizon."""
    count = max(3, int(math.ceil(float(duration) / cfg.DT)) + 1)
    best = math.inf
    for delta in np.linspace(0.0, float(duration), count):
        q = local_reference.evaluate(min(float(delta), local_reference.total_duration))
        risk = evaluator.configuration(q, forecast, float(delta), density=density, with_gradient=False)
        best = min(best, float(risk.min_distance))
    return float(best)


def _observed_and_gt_at_trigger(
    instance: dict[str, Any],
    trigger_time: float,
) -> tuple[
    tuple[np.ndarray, np.ndarray, float],  # observed (center, velocity, radius)
    tuple[np.ndarray, np.ndarray, float],  # GT     (center, velocity, radius)
]:
    """Build observed and GT obstacle state at trigger time.

    Observed = GT + frozen observation errors (one-time, same across leads).
    No double noise injection.
    """
    gt_center0 = np.asarray(instance["gt_center0"], dtype=np.float64)
    gt_velocity = np.asarray(instance["gt_velocity"], dtype=np.float64)
    gt_radius = float(instance["gt_radius"])
    motion_start_time = float(instance.get("motion_start_time", 0.0))
    pre_motion_center = (
        None
        if instance.get("pre_motion_center") is None
        else np.asarray(instance["pre_motion_center"], dtype=np.float64)
    )

    # GT obstacle state at trigger time
    gt_center = obstacle_center_at(gt_center0, gt_velocity, trigger_time, motion_start_time, pre_motion_center)
    gt_vel = obstacle_velocity_at(gt_velocity, trigger_time, motion_start_time)
    gt_obs = (gt_center, gt_vel, gt_radius)

    # Observed = GT + frozen errors (applied once, same for all leads)
    obs_pos_error = np.asarray(instance["obs_pos_error"], dtype=np.float64)
    obs_vel_error = np.asarray(instance["obs_vel_error"], dtype=np.float64)
    obs_radius_error = float(instance.get("obs_radius_error", 0.0))
    obs_center = gt_center + obs_pos_error
    obs_vel = gt_vel + obs_vel_error
    obs_radius = max(0.025, gt_radius + obs_radius_error)
    obs_obs = (obs_center, obs_vel, obs_radius)

    return obs_obs, gt_obs


def _build_funnel_metrics(
    wall_elapsed_ms: float,
    actual_switch_delay: float,
    optimizer_success: bool,
    fallback_used: bool,
    online_accepted: bool,
    beneficial_online: bool,
    dense_gt_accepted: bool,
    dense_gt_candidate_min: float,
    dense_gt_reference_min: float,
) -> dict[str, Any]:
    """Build multi-level funnel metrics.

    Cumulative funnel (strictly cumulative):
      1. ready_before_switch    — planner finished before deadline
      2. online_acceptable      — ready + candidate_available + online distance OK
      3. switch_eligible        — online_acceptable + beneficial (beats reference)

    Independent GT diagnostics (not cumulative):
      4. dense_gt_safe_all      — GT forecast verification passes (any candidate)
      5. candidate_success      — switch_eligible AND dense_gt_safe_all

    Candidate_available = optimizer_success OR feasible-seed fallback
    """
    ready_before_switch = bool(wall_elapsed_ms <= 1000.0 * actual_switch_delay)
    candidate_available = bool(optimizer_success or fallback_used)
    online_acceptable = bool(ready_before_switch and candidate_available and online_accepted)
    switch_eligible = bool(online_acceptable and beneficial_online)

    # Independent GT diagnostics (not cumulative — evaluated for any candidate trajectory)
    dense_gt_safe_all = bool(dense_gt_accepted)
    dense_gt_beneficial_all = bool(
        dense_gt_safe_all
        and dense_gt_candidate_min >= dense_gt_reference_min + cfg.SWITCH_IMPROVEMENT_MARGIN
    )
    # Final success metric
    candidate_success = bool(switch_eligible and dense_gt_safe_all)

    return {
        "ready_before_switch": ready_before_switch,
        "candidate_available": candidate_available,
        "fallback_to_feasible_seed": fallback_used,
        "online_acceptable": online_acceptable,
        "switch_eligible": switch_eligible,
        # Independent GT diagnostics (no cumulative requirement)
        "dense_gt_safe_all": dense_gt_safe_all,
        "dense_gt_beneficial_all": dense_gt_beneficial_all,
        # Final success
        "candidate_success": candidate_success,
    }


def replay_one(
    config: dict[str, Any],
    limits,
    evaluator,
    verifier,
    dense_verifier,
    instance: dict[str, Any],
    method: str,
    lead_label: str,
    lead_time: float,
    *,
    formal: bool = False,
) -> dict[str, Any]:
    """Run one candidate replay trial using independent local instance data.

    Parameters
    ----------
    formal : bool
        If True, strict validation: missing ``first_accept_violation_time`` is an
        error (not a fallback).
    """
    # --- Conflict time ---
    conflict_time = instance.get("first_accept_violation_time") or instance.get("conflict_time_global")
    conflict_time_source = "first_accept_violation_time" if instance.get("first_accept_violation_time") is not None else "conflict_time_global"
    if conflict_time is None:
        if formal:
            return _early_invalid_row(instance, method, lead_label, lead_time, ["missing_conflict_time"])
        conflict_time = instance.get("reference_risk_time")
        conflict_time_source = "reference_risk_time_fallback"
    if conflict_time is None:
        raise ValueError("no conflict time available in instance")
    conflict_time = float(conflict_time)

    # --- Timing ---
    trigger_time = conflict_time - float(lead_time)
    actual_switch_delay = min(cfg.PLANNED_SWITCH_DELAY, float(lead_time) - cfg.POST_SWITCH_CONFLICT_TIME)
    optimization_budget_s = min(cfg.PLANNING_BUDGET, max(0.0, actual_switch_delay - 0.5))

    # --- Early invalidation ---
    invalid_reasons = []
    if trigger_time < 0.0:
        invalid_reasons.append("negative_trigger_time")
    if actual_switch_delay <= 0.0 or optimization_budget_s <= 0.0:
        invalid_reasons.append("nonpositive_switch_or_budget")
    if invalid_reasons:
        return _early_invalid_row(instance, method, lead_label, lead_time, invalid_reasons)

    # --- Local boundary states from independent 4 s segment ---
    head_state = np.asarray(instance["local_head_state"], dtype=np.float64)  # (6, 3)
    tail_state = np.asarray(instance["local_tail_state"], dtype=np.float64)  # (6, 3)
    q_now = head_state[:, 0]
    qd_now = head_state[:, 1]
    qdd_now = head_state[:, 2]
    q_goal = tail_state[:, 0]
    qd_goal = tail_state[:, 1]
    qdd_goal = tail_state[:, 2]
    local_horizon = float(instance.get("local_horizon", cfg.LOCAL_REPLAN_HORIZON))

    # --- Reconstruct local reference trajectory for warm-start and benefit gate ---
    local_reference = _build_local_reference(instance)

    # --- Dual forecasts ---
    (obs_center, obs_velocity, obs_radius), (gt_center, gt_velocity, gt_radius) = \
        _observed_and_gt_at_trigger(instance, trigger_time)

    observed_forecast = constant_forecast(obs_center, obs_velocity, obs_radius)
    observed_local_forecast = ShiftedForecast(observed_forecast, actual_switch_delay, local_horizon)

    # Pure GT forecast — no uncertainty inflation
    gt_forecast = ground_truth_forecast(gt_center, gt_velocity, gt_radius)
    gt_local_forecast = ShiftedForecast(gt_forecast, actual_switch_delay, local_horizon)

    # --- Optimizer (uses OBSERVED forecast, local reference warm-start) ---
    started = time.perf_counter()
    candidate = optimize_candidate(
        config,
        evaluator,
        limits,
        observed_local_forecast,
        q_now=q_now,
        qd_now=qd_now,
        qdd_now=qdd_now,
        q_goal=q_goal,
        qd_goal=qd_goal,
        qdd_goal=qdd_goal,
        remaining_duration=local_horizon,
        verifier=verifier,
        force_inner_initial=np.asarray(instance["local_reference_inner"], dtype=np.float64),
        optimization_budget_s=optimization_budget_s,
    )
    wall_elapsed_ms = float((time.perf_counter() - started) * 1000.0)

    # --- Online verification (OBSERVED forecast) ---
    online_min = float(candidate["verification"]["min_distance"])
    optimizer_success = bool(candidate["optimization"]["success"])
    fallback_used = bool(candidate["optimization"].get("fallback_to_feasible_seed", False))
    online_accepted = bool(candidate["verification"]["accepted"])

    # --- Online reference min (OBSERVED forecast) ---
    online_reference_min = _reference_local_min_distance(
        dense_verifier.risk_evaluator,
        local_reference,
        observed_local_forecast,
        local_horizon,
        density=cfg.SURFACE_DENSITY_VERIFY,
    )

    # --- Dense GT verification (pure GT forecast) ---
    dense_gt = dense_verifier.verify(
        candidate["trajectory"],
        gt_local_forecast,
        current_q=q_now,
        current_qd=qd_now,
        current_qdd=qdd_now,
        q_goal=q_goal,
        solver_success=True,
    )
    dense_gt_accepted = bool(dense_gt.accepted)
    dense_gt_candidate_min = float(dense_gt.min_distance)

    # --- Dense GT reference min (pure GT forecast) ---
    dense_gt_reference_min = _reference_local_min_distance(
        dense_verifier.risk_evaluator,
        local_reference,
        gt_local_forecast,
        local_horizon,
        density="dense",
    )

    # --- Benefit gates ---
    beneficial_online = bool(online_min >= online_reference_min + cfg.SWITCH_IMPROVEMENT_MARGIN)
    beneficial_dense_gt = bool(
        dense_gt_candidate_min >= dense_gt_reference_min + cfg.SWITCH_IMPROVEMENT_MARGIN
    )

    # --- Funnel metrics ---
    funnel = _build_funnel_metrics(
        wall_elapsed_ms=wall_elapsed_ms,
        actual_switch_delay=actual_switch_delay,
        optimizer_success=optimizer_success,
        fallback_used=fallback_used,
        online_accepted=online_accepted,
        beneficial_online=beneficial_online,
        dense_gt_accepted=dense_gt_accepted,
        dense_gt_candidate_min=dense_gt_candidate_min,
        dense_gt_reference_min=dense_gt_reference_min,
    )

    return {
        # --- Identification ---
        "instance_id": instance["instance_id"],
        "scenario_type": instance["scenario_type"],
        "method": method,
        "lead_label": lead_label,
        "lead_time_requested": float(lead_time),
        "lead_time_actual": float(conflict_time - trigger_time),
        "speed_group": instance.get("speed_group"),

        # --- Timing ---
        "trigger_time": trigger_time,
        "conflict_time": conflict_time,
        "conflict_time_source": conflict_time_source,
        "planned_switch_delay": float(actual_switch_delay),
        "optimization_budget_s": float(optimization_budget_s),
        "valid_replay_window": True,
        "invalid_reasons": [],
        "wall_elapsed_ms": wall_elapsed_ms,
        "deadline_slack_ms": float(1000.0 * actual_switch_delay - wall_elapsed_ms),

        # --- Multi-level funnel ---
        **funnel,

        # --- Online verification (observed forecast) ---
        "within_budget": float(candidate["optimization"]["elapsed_ms"]) <= 1000.0 * optimization_budget_s,
        "planner_finished": True,
        "optimizer_converged": optimizer_success,
        "online_feasible": online_accepted,
        "online_min_distance": online_min,
        "online_reference_min_distance": float(online_reference_min),
        "online_beneficial": beneficial_online,
        "delta_min_distance_online": float(online_min - online_reference_min),
        "continuity_pass": bool(candidate["verification"]["checks"].get("continuity_q_ok", False)),
        "online_distance_pass": bool(candidate["verification"]["checks"].get("distance_ok", False)),
        "reasons": candidate["verification"]["reasons"],

        # --- Dense GT verification (pure GT forecast) ---
        "dense_gt_feasible": dense_gt_accepted,
        "dense_gt_safe": dense_gt_accepted,  # alias for backward compat
        "dense_gt_distance_pass": bool(dense_gt.checks.get("distance_ok", False)),
        "dense_gt_candidate_min": dense_gt_candidate_min,
        "dense_gt_reference_min": dense_gt_reference_min,
        "dense_gt_beneficial": beneficial_dense_gt,
        "delta_min_distance_dense_gt": float(dense_gt_candidate_min - dense_gt_reference_min),
        "dense_reasons": dense_gt.reasons,

        # --- Resource ---
        "optimizer_core_ms": float(candidate["optimization"]["elapsed_ms"]),
        "online_verification_ms": float(candidate["verification"].get("validation_ms", 0.0)),
        "dense_gt_verification_ms": float(dense_gt.validation_ms),
        "candidate_ready_wall_ms": wall_elapsed_ms,
        "optimization": candidate["optimization"],
    }


def _early_invalid_row(
    instance: dict[str, Any],
    method: str,
    lead_label: str,
    lead_time: float,
    invalid_reasons: list[str],
) -> dict[str, Any]:
    return {
        "instance_id": instance["instance_id"],
        "scenario_type": instance["scenario_type"],
        "method": method,
        "lead_label": lead_label,
        "lead_time_requested": float(lead_time),
        "lead_time_actual": None,
        "speed_group": instance.get("speed_group"),
        "valid_replay_window": False,
        "invalid_reasons": invalid_reasons,
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        speed = "-" if row.get("speed_group") is None else f"{float(row['speed_group']):.2f}"
        grouped.setdefault((row["scenario_type"], speed, row["lead_label"], row["method"]), []).append(row)
    table = []
    for (scenario_type, speed_group, lead_label, method), items in sorted(grouped.items()):
        valid = [item for item in items if item.get("valid_replay_window", True)]
        invalid = len(items) - len(valid)
        table.append(
            {
                "scenario_type": scenario_type,
                "speed_group": speed_group,
                "lead_label": lead_label,
                "method": method,
                "n": len(items),
                "valid": len(valid),
                "invalid": invalid,
                # Cumulative funnel
                "ready_before_switch": float(np.mean([item["ready_before_switch"] for item in valid])) if valid else None,
                "candidate_available": float(np.mean([item["candidate_available"] for item in valid])) if valid else None,
                "online_acceptable": float(np.mean([item["online_acceptable"] for item in valid])) if valid else None,
                "switch_eligible": float(np.mean([item["switch_eligible"] for item in valid])) if valid else None,
                "candidate_success": float(np.mean([item["candidate_success"] for item in valid])) if valid else None,
                # Independent GT diagnostics
                "dense_gt_safe_all": float(np.mean([item["dense_gt_safe_all"] for item in valid])) if valid else None,
                "dense_gt_beneficial_all": float(np.mean([item["dense_gt_beneficial_all"] for item in valid])) if valid else None,
                # Legacy / convenience
                "within_budget": float(np.mean([item["within_budget"] for item in valid])) if valid else None,
                "optimizer_converged": float(np.mean([item["optimizer_converged"] for item in valid])) if valid else None,
                "online_feasible": float(np.mean([item["online_feasible"] for item in valid])) if valid else None,
                "usable": float(np.mean([item.get("online_acceptable", False) and item.get("dense_gt_safe_all", False) for item in valid])) if valid else None,
                "dense_gt_beneficial_agg": float(np.mean([item["dense_gt_beneficial_all"] for item in valid])) if valid else None,
                "delta_min_distance_dense_gt": float(np.mean([item["delta_min_distance_dense_gt"] for item in valid])) if valid else None,
                "candidate_ready_wall_ms_p95": float(np.percentile([item["candidate_ready_wall_ms"] for item in valid], 95)) if valid else None,
                "optimizer_core_ms_p95": float(np.percentile([item["optimizer_core_ms"] for item in valid], 95)) if valid else None,
            }
        )
    return table


def write_replay_table(summary: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "| scenario | speed | lead | method | n | valid | invalid | ready | candidate_avail | online_accept | switch_elig | dense_GT_safe | success | delta Dmin GT / m | ready wall p95 / ms |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['scenario_type']} | {row['speed_group']} | {row['lead_label']} | "
            f"{cfg.METHOD_NAMES.get(row['method'], row['method'])} | "
            f"{row['n']} | {row['valid']} | {row['invalid']} | "
            f"{_fmt(row['ready_before_switch'], 2)} | {_fmt(row['candidate_available'], 2)} | "
            f"{_fmt(row['online_acceptable'], 2)} | {_fmt(row['switch_eligible'], 2)} | "
            f"{_fmt(row['dense_gt_safe_all'], 2)} | {_fmt(row['candidate_success'], 2)} | "
            f"{_fmt(row['delta_min_distance_dense_gt'])} | "
            f"{_fmt(row['candidate_ready_wall_ms_p95'], 1)} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(Path("results/new/6_4_candidate_replay")))
    parser.add_argument("--config", default=str(cfg.STAGE4_CONFIG))
    parser.add_argument("--scenario", choices=["D1", "D2M"], default="D1")
    parser.add_argument("--method", choices=["critical_point_nubs", "ccro_nubs"], default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--g1", action="store_true", help="G1 deterministic mode (zero observation noise)")
    parser.add_argument("--formal", action="store_true", help="Formal Stage-A mode (strict validation)")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    trials_dir = output / "candidate_trials"
    paper_dir = output / "paper"
    trials_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name("config_64.yaml"), output / "config_64.yaml")

    # G1 deterministic mode: zero observation noise
    if args.g1:
        cfg.OBS_POS_SIGMA = cfg.G1_OBS_POS_SIGMA
        cfg.OBS_VEL_SIGMA = cfg.G1_OBS_VEL_SIGMA
        cfg.OBS_RADIUS_SIGMA = cfg.G1_OBS_RADIUS_SIGMA
        print("[6.4 replay] G1 mode: observation noise set to zero")

    config = load_stage4_config(Path(args.config))
    model = load_surface_model(config)
    # We still need the global reference trajectory to generate instances
    # (the local 4 s segments are extracted from it during generation)
    reference, _, _, _ = make_reference(config)
    evaluator, verifier, limits = make_risk_stack(config, model, None)
    critical_evaluator, critical_verifier = make_critical_risk_stack(config, model)
    dense_verifier = DynamicTrajectoryVerifier(
        evaluator,
        limits,
        d_stop=cfg.D_STOP,
        time_step=cfg.DT,
        density="dense",
        epsilon_goal=1.0e-2,
        epsilon_continuity_q=5.0e-3,
        epsilon_continuity_qd=3.0e-3,
        epsilon_continuity_qdd=3.0e-3,
        limit_tolerance=1.0e-8,
    )
    instances = generate_instances(
        model, reference, output / "instances",
        smoke=args.smoke, gate=args.smoke, formal=args.formal,
    )
    instances = [item for item in instances if item["instance_id"].startswith(args.scenario + "_")]
    methods = (args.method,) if args.method else ("critical_point_nubs", "ccro_nubs")
    rows: list[dict[str, Any]] = []
    for instance in instances:
        for lead_label, lead_time in cfg.LEAD_TIME_GROUPS.items():
            for method in methods:
                path = trials_dir / f"{instance['instance_id']}_{lead_label}_{method}.json"
                if args.resume and path.exists():
                    row = json.loads(path.read_text(encoding="utf-8"))
                else:
                    method_evaluator = critical_evaluator if method == "critical_point_nubs" else evaluator
                    method_verifier = verifier
                    row = replay_one(
                        config,
                        limits,
                        method_evaluator,
                        method_verifier,
                        dense_verifier,
                        instance,
                        method,
                        lead_label,
                        lead_time,
                        formal=args.formal,
                    )
                    write_json(path, row)
                rows.append(row)
                if row.get("valid_replay_window", True):
                    print(
                        f"[6.4 replay] {row['instance_id']} {row['lead_label']} "
                        f"{cfg.METHOD_NAMES.get(method, method)} "
                        f"success={row['candidate_success']} "
                        f"eligible={row['switch_eligible']} "
                        f"gt_safe={row['dense_gt_safe_all']} "
                        f"online={row['online_feasible']} "
                        f"dD_gt={row['delta_min_distance_dense_gt']:.3f}"
                    )
                else:
                    print(
                        f"[6.4 replay] {row['instance_id']} {row['lead_label']} "
                        f"{cfg.METHOD_NAMES.get(method, method)} "
                        f"invalid={'+'.join(row.get('invalid_reasons', []))}"
                    )
    summary = _aggregate(rows)
    meta = manifest_meta(
        extra_source_paths=[
            Path(__file__),
            Path(__file__).with_name("scenarios_64.py"),
            Path(__file__).with_name("common_64.py"),
            Path(__file__).with_name("config_64.py"),
        ],
        stage4_config_path=Path(args.config),
    )
    metrics = {
        "experiment": "6.4 stage-A candidate replay",
        "scope": "candidate generation only; independent 4 s local replay instances",
        "g1_mode": bool(args.g1),
        "formal_mode": bool(args.formal),
        **meta,
        "scenario": args.scenario,
        "methods": list(methods),
        "lead_time_groups": cfg.LEAD_TIME_GROUPS,
        "post_switch_conflict_time_s": cfg.POST_SWITCH_CONFLICT_TIME,
        "trial_count": len(rows),
        "summary": summary,
        "trials": rows,
    }
    write_json(output / "candidate_replay_64.json", metrics)
    write_replay_table(summary, paper_dir / "table_6_4_candidate_replay.md")
    print(f"[6.4 replay] saved {output / 'candidate_replay_64.json'}")
    print(f"[6.4 replay] saved {paper_dir / 'table_6_4_candidate_replay.md'}")


if __name__ == "__main__":
    main()
