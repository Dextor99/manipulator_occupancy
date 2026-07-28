"""Stage-A candidate-planning replay for the Chapter 6.4 boundary experiment.

This script isolates candidate generation from the closed-loop pending state
machine.  It replays fixed trigger/switch states and compares Critical-NUBS
and CCRO-NUBS under identical forecasts, budgets, continuity checks, and
online/dense validation.

Key design:
  - Observed forecast  → optimizer + online verifier
  - GT forecast        → dense GT verification (ground-truth obstacle motion)
  - Multi-level funnel replaces single "usable" flag
  - Uniform post-switch conflict time (POST_SWITCH_CONFLICT_TIME = 1.5 s)
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

from planning.obstacle_forecast import ShiftedForecast
from planning.verifier import DynamicTrajectoryVerifier

from . import config_64 as cfg
from .common_64 import (
    constant_forecast,
    git_commit_hash,
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


def _reference_local_min_distance(
    evaluator,
    reference,
    forecast,
    switch_tau: float,
    duration: float,
    *,
    density: str,
) -> float:
    """Evaluate reference trajectory min distance over the local horizon."""
    count = max(3, int(math.ceil(float(duration) / cfg.DT)) + 1)
    best = math.inf
    for delta in np.linspace(0.0, float(duration), count):
        q = reference.evaluate(min(reference.total_duration, float(switch_tau) + float(delta)))
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
    """Build both observed (noisy) and GT (truth) obstacle state at trigger time."""
    gt_center0 = np.asarray(instance["gt_center0"], dtype=np.float64)
    gt_velocity = np.asarray(instance["gt_velocity"], dtype=np.float64)
    gt_radius = float(instance["gt_radius"])
    motion_start_time = float(instance.get("motion_start_time", 0.0))
    pre_motion_center = (
        None
        if instance.get("pre_motion_center") is None
        else np.asarray(instance["pre_motion_center"], dtype=np.float64)
    )

    # GT obstacle position and velocity at trigger time
    gt_center = obstacle_center_at(gt_center0, gt_velocity, trigger_time, motion_start_time, pre_motion_center)
    gt_vel = obstacle_velocity_at(gt_velocity, trigger_time, motion_start_time)
    gt_obs = (gt_center, gt_vel, gt_radius)

    # Noisy observed version
    obs_center0 = np.asarray(instance["observed_center0"], dtype=np.float64)
    obs_velocity = np.asarray(instance["observed_velocity"], dtype=np.float64)
    obs_radius = float(instance["observed_radius"])
    obs_center = obstacle_center_at(obs_center0, obs_velocity, trigger_time, motion_start_time, pre_motion_center)
    obs_vel = obstacle_velocity_at(obs_velocity, trigger_time, motion_start_time)
    # Inject per-call noise for velocity (only when OBS_VEL_SIGMA > 0)
    rng = np.random.default_rng(int(instance.get("observation_seed", 0)) + int(round(100.0 * trigger_time)))
    if float(np.linalg.norm(obs_vel)) >= 1.0e-9 and cfg.OBS_VEL_SIGMA > 0.0:
        obs_vel = obs_vel + rng.normal(0.0, cfg.OBS_VEL_SIGMA, size=3)
    if cfg.OBS_POS_SIGMA > 0.0:
        obs_center = obs_center + rng.normal(0.0, cfg.OBS_POS_SIGMA, size=3)
    obs_radius = max(0.025, float(obs_radius))
    obs_obs = (obs_center, obs_vel, obs_radius)

    return obs_obs, gt_obs


def _check_motion_norm(
    q_now: np.ndarray,
    q_goal: np.ndarray,
    instance: dict[str, Any],
    method: str,
    lead_label: str,
    lead_time: float,
    trigger_time: float,
    conflict_time: float,
    actual_switch_delay: float,
    switch_tau: float,
    resume_tau: float,
    invalid_reasons: list[str],
) -> dict[str, Any] | None:
    """Return early-exit row if motion is too small, else None."""
    motion_norm = float(np.linalg.norm(q_goal - q_now))
    if motion_norm < cfg.MIN_REPLAY_MOTION_NORM:
        return {
            "instance_id": instance["instance_id"],
            "scenario_type": instance["scenario_type"],
            "method": method,
            "lead_label": lead_label,
            "lead_time_requested": float(lead_time),
            "lead_time_actual": float(conflict_time - trigger_time),
            "speed_group": instance.get("speed_group"),
            "trigger_time": float(trigger_time),
            "conflict_time": float(conflict_time),
            "planned_switch_delay": float(actual_switch_delay),
            "switch_tau": float(switch_tau),
            "resume_tau": float(resume_tau),
            "valid_replay_window": False,
            "invalid_reasons": invalid_reasons + ["zero_motion_window"],
            "motion_norm": motion_norm,
        }
    return None


def _early_invalid_row(
    instance: dict[str, Any],
    method: str,
    lead_label: str,
    lead_time: float,
    trigger_time: float,
    conflict_time: float,
    actual_switch_delay: float,
    switch_tau: float,
    resume_tau: float,
    invalid_reasons: list[str],
) -> dict[str, Any]:
    """Build a row for an invalid replay window."""
    return {
        "instance_id": instance["instance_id"],
        "scenario_type": instance["scenario_type"],
        "method": method,
        "lead_label": lead_label,
        "lead_time_requested": float(lead_time),
        "lead_time_actual": None,
        "speed_group": instance.get("speed_group"),
        "trigger_time": float(trigger_time),
        "conflict_time": float(conflict_time),
        "conflict_time_source": instance.get("conflict_time_source", "unknown"),
        "planned_switch_delay": float(actual_switch_delay),
        "switch_tau": float(switch_tau),
        "resume_tau": float(resume_tau),
        "valid_replay_window": False,
        "invalid_reasons": invalid_reasons,
    }


def _build_funnel_metrics(
    wall_elapsed_ms: float,
    actual_switch_delay: float,
    optimizer_success: bool,
    online_accepted: bool,
    beneficial_online: bool,
    dense_gt_accepted: bool,
    dense_gt_candidate_min: float,
    dense_gt_reference_min: float,
) -> dict[str, Any]:
    """Build multi-level funnel metrics.

    Levels (strictly cumulative):
      1. ready_before_switch    — planner finished before deadline
      2. online_acceptable      — ready + optimizer success + online distance OK
      3. switch_eligible        — online_acceptable + beneficial (beats reference via online forecast)
      4. dense_gt_safe          — GT forecast verification passes
      5. dense_gt_beneficial    — candidate GT min > reference GT min + margin
      6. candidate_success      — switch_eligible AND dense_gt_safe
    """
    ready_before_switch = bool(wall_elapsed_ms <= 1000.0 * actual_switch_delay)
    online_acceptable = bool(ready_before_switch and optimizer_success and online_accepted)
    switch_eligible = bool(online_acceptable and beneficial_online)
    dense_gt_safe = bool(dense_gt_accepted)
    dense_gt_beneficial = bool(
        dense_gt_safe
        and dense_gt_candidate_min >= dense_gt_reference_min + cfg.SWITCH_IMPROVEMENT_MARGIN
    )
    candidate_success = bool(switch_eligible and dense_gt_safe)
    return {
        "ready_before_switch": ready_before_switch,
        "online_acceptable": online_acceptable,
        "switch_eligible": switch_eligible,
        "dense_gt_safe": dense_gt_safe,
        "dense_gt_beneficial": dense_gt_beneficial,
        "candidate_success": candidate_success,
    }


def replay_one(
    config: dict[str, Any],
    reference,
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
    """Run one candidate replay trial.

    Parameters
    ----------
    formal : bool
        If True, strict validation: missing ``first_accept_violation_time`` is an
        error (not a fallback to ``reference_risk_time``).
    """
    # --- Conflict time with optional fallback ---
    conflict_time = instance.get("first_accept_violation_time")
    conflict_time_source = "first_accept_violation_time"
    if conflict_time is None:
        if formal:
            return _early_invalid_row(
                instance, method, lead_label, lead_time,
                trigger_time=-1.0,
                conflict_time=-1.0,
                actual_switch_delay=0.0,
                switch_tau=-1.0,
                resume_tau=-1.0,
                invalid_reasons=["missing_first_accept_violation"],
            )
        conflict_time = instance.get("reference_risk_time")
        conflict_time_source = "reference_risk_time_fallback"
    if conflict_time is None:
        raise ValueError("candidate replay requires first_accept_violation_time or reference_risk_time")

    # --- Timing ---
    trigger_time = float(conflict_time) - float(lead_time)
    # Uniform post-switch conflict guard: all lead groups see same local conflict time
    actual_switch_delay = min(cfg.PLANNED_SWITCH_DELAY, float(lead_time) - cfg.POST_SWITCH_CONFLICT_TIME)
    optimization_budget_s = min(cfg.PLANNING_BUDGET, max(0.0, actual_switch_delay - 0.5))
    switch_tau = trigger_time + actual_switch_delay
    resume_tau = switch_tau + cfg.LOCAL_REPLAN_HORIZON

    # --- Early invalidation checks ---
    invalid_reasons = []
    if trigger_time < 0.0:
        invalid_reasons.append("negative_trigger_time")
    if actual_switch_delay <= 0.0 or optimization_budget_s <= 0.0:
        invalid_reasons.append("nonpositive_switch_or_budget")
    if resume_tau > reference.total_duration + 1.0e-9:
        invalid_reasons.append("clipped_local_horizon")
    if invalid_reasons:
        return _early_invalid_row(
            instance, method, lead_label, lead_time,
            trigger_time, conflict_time,
            actual_switch_delay, switch_tau, resume_tau,
            invalid_reasons,
        )

    # --- Local boundaries ---
    remaining_duration = cfg.LOCAL_REPLAN_HORIZON
    q_now = reference.evaluate(switch_tau)
    qd_now = reference.evaluate(switch_tau, derivative_order=1)
    qdd_now = reference.evaluate(switch_tau, derivative_order=2)
    q_goal = reference.evaluate(resume_tau)
    qd_goal = reference.evaluate(resume_tau, derivative_order=1)
    qdd_goal = reference.evaluate(resume_tau, derivative_order=2)

    early = _check_motion_norm(
        q_now, q_goal,
        instance, method, lead_label, lead_time,
        trigger_time, conflict_time, actual_switch_delay,
        switch_tau, resume_tau, invalid_reasons,
    )
    if early is not None:
        return early

    # --- Dual forecasts: observed (noisy) vs GT (truth) ---
    (obs_center, obs_velocity, obs_radius), (gt_center, gt_velocity, gt_radius) = \
        _observed_and_gt_at_trigger(instance, trigger_time)

    observed_forecast = constant_forecast(obs_center, obs_velocity, obs_radius)
    observed_local_forecast = ShiftedForecast(observed_forecast, actual_switch_delay, remaining_duration)

    gt_forecast = constant_forecast(gt_center, gt_velocity, gt_radius)
    gt_local_forecast = ShiftedForecast(gt_forecast, actual_switch_delay, remaining_duration)

    # --- Optimizer (uses OBSERVED forecast) ---
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
        remaining_duration=remaining_duration,
        verifier=verifier,
        warm_start_trajectory=reference,
        warm_start_tau=switch_tau,
        optimization_budget_s=optimization_budget_s,
    )
    wall_elapsed_ms = float((time.perf_counter() - started) * 1000.0)

    # --- Online verification (uses OBSERVED forecast) ---
    online_min = float(candidate["verification"]["min_distance"])
    optimizer_success = bool(candidate["optimization"]["success"])
    online_accepted = bool(candidate["verification"]["accepted"])

    # --- Online reference min (OBSERVED forecast) ---
    online_reference_min = _reference_local_min_distance(
        dense_verifier.risk_evaluator,
        reference,
        observed_local_forecast,
        switch_tau,
        remaining_duration,
        density=cfg.SURFACE_DENSITY_VERIFY,
    )

    # --- Dense GT verification (uses GT forecast) ---
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

    # --- Dense GT reference min (GT forecast) ---
    dense_gt_reference_min = _reference_local_min_distance(
        dense_verifier.risk_evaluator,
        reference,
        gt_local_forecast,
        switch_tau,
        remaining_duration,
        density="dense",
    )

    # --- Benefit gate (online and GT versions) ---
    beneficial_online = bool(online_min >= online_reference_min + cfg.SWITCH_IMPROVEMENT_MARGIN)
    beneficial_dense_gt = bool(
        dense_gt_candidate_min >= dense_gt_reference_min + cfg.SWITCH_IMPROVEMENT_MARGIN
    )

    # --- Multi-level funnel metrics ---
    funnel = _build_funnel_metrics(
        wall_elapsed_ms=wall_elapsed_ms,
        actual_switch_delay=actual_switch_delay,
        optimizer_success=optimizer_success,
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
        "trigger_time": float(trigger_time),
        "conflict_time": float(conflict_time),
        "conflict_time_source": conflict_time_source,
        "reference_risk_time": float(instance.get("reference_risk_time", conflict_time)),
        "planned_switch_delay": float(actual_switch_delay),
        "optimization_budget_s": float(optimization_budget_s),
        "switch_tau": float(switch_tau),
        "resume_tau": float(resume_tau),
        "valid_replay_window": True,
        "invalid_reasons": [],
        "motion_norm": float(np.linalg.norm(q_goal - q_now)),
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

        # --- Dense GT verification (GT forecast) ---
        "dense_gt_feasible": dense_gt_accepted,
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
                # Funnel metrics (multi-level)
                "ready_before_switch": float(np.mean([item["ready_before_switch"] for item in valid])) if valid else None,
                "online_acceptable": float(np.mean([item["online_acceptable"] for item in valid])) if valid else None,
                "switch_eligible": float(np.mean([item["switch_eligible"] for item in valid])) if valid else None,
                "dense_gt_safe": float(np.mean([item["dense_gt_safe"] for item in valid])) if valid else None,
                "dense_gt_beneficial": float(np.mean([item["dense_gt_beneficial"] for item in valid])) if valid else None,
                "candidate_success": float(np.mean([item["candidate_success"] for item in valid])) if valid else None,
                # Legacy fields for backward compat
                "within_budget": float(np.mean([item["within_budget"] for item in valid])) if valid else None,
                "optimizer_converged": float(np.mean([item["optimizer_converged"] for item in valid])) if valid else None,
                "online_feasible": float(np.mean([item["online_feasible"] for item in valid])) if valid else None,
                "usable": float(np.mean([item["online_acceptable"]
                                          and item["dense_gt_feasible"]
                                          for item in valid])) if valid else None,
                "beneficial": float(np.mean([item["online_beneficial"] for item in valid])) if valid else None,
                "dense_gt_beneficial_agg": float(np.mean([item["dense_gt_beneficial"] for item in valid])) if valid else None,
                "delta_min_distance_dense_gt": float(np.mean([item["delta_min_distance_dense_gt"] for item in valid])) if valid else None,
                "candidate_ready_wall_ms_p95": float(np.percentile([item["candidate_ready_wall_ms"] for item in valid], 95)) if valid else None,
                "optimizer_core_ms_p95": float(np.percentile([item["optimizer_core_ms"] for item in valid], 95)) if valid else None,
            }
        )
    return table


def write_replay_table(summary: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "| scenario | speed | lead | method | n | valid | invalid | ready | online_accept | switch_elig | dense_gt_safe | gt_beneficial | candidate_success | delta Dmin GT / m | ready wall p95 / ms |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['scenario_type']} | {row['speed_group']} | {row['lead_label']} | "
            f"{cfg.METHOD_NAMES.get(row['method'], row['method'])} | "
            f"{row['n']} | {row['valid']} | {row['invalid']} | "
            f"{_fmt(row['ready_before_switch'], 2)} | {_fmt(row['online_acceptable'], 2)} | "
            f"{_fmt(row['switch_eligible'], 2)} | {_fmt(row['dense_gt_safe'], 2)} | "
            f"{_fmt(row['dense_gt_beneficial'], 2)} | {_fmt(row['candidate_success'], 2)} | "
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
    parser.add_argument("--g1", action="store_true", help="G1 deterministic mode (zero noise)")
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

    # --- Override observation noise for G1 deterministic mode ---
    if args.g1:
        import experiments.new  # noqa: F811
        experiments.new._6_4.config_64.OBS_POS_SIGMA = 0.0
        experiments.new._6_4.config_64.OBS_VEL_SIGMA = 0.0
        experiments.new._6_4.config_64.OBS_RADIUS_SIGMA = 0.0

    config = load_stage4_config(Path(args.config))
    model = load_surface_model(config)
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
        model, reference, output / "instances", smoke=args.smoke, gate=args.smoke,
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
                        reference,
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
                        f"dense_gt_safe={row['dense_gt_safe']} "
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
    meta = manifest_meta(extra_source_paths=[
        Path(__file__),
        Path(__file__).with_name("scenarios_64.py"),
        Path(__file__).with_name("common_64.py"),
        Path(__file__).with_name("config_64.py"),
    ])
    metrics = {
        "experiment": "6.4 stage-A candidate replay",
        "scope": "candidate generation only; no closed-loop execution",
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
