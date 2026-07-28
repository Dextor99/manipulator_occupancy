"""G0 consistency audit before launching the Chapter 6.4 formal experiment.

The audit intentionally uses existing pilot/gate outputs.  It does not tune
thresholds or select successful trials; its purpose is to decide whether the
candidate-generation and switching semantics are coherent enough for a formal
boundary experiment.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import config_64 as cfg
from .audit_64 import dense_gt_audit, load_instances, load_trials
from .common_64 import write_json


ASYNC_METHODS = {"critical_point_nubs", "ccro_nubs"}


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


def _stat(values: list[Any]) -> dict[str, float | None]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"n": 0, "mean": None, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "n": len(clean),
        "mean": float(np.mean(clean)),
        "min": float(np.min(clean)),
        "p50": float(np.percentile(clean, 50)),
        "p95": float(np.percentile(clean, 95)),
        "max": float(np.max(clean)),
    }


def _event_check(event: dict[str, Any], check: str, *, validation: str) -> bool:
    payload = event.get(validation) or {}
    checks = payload.get("checks") or {}
    return bool(checks.get(check, False))


def _events(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for trial in trials:
        if trial.get("method") not in ASYNC_METHODS:
            continue
        for event in trial.get("events", []):
            rows.append(
                {
                    **event,
                    "trial_id": trial.get("trial_id"),
                    "instance_id": trial.get("instance_id"),
                    "scenario_type": trial.get("scenario_type"),
                    "method": trial.get("method"),
                    "trial_task_safe": bool(trial.get("task_safe_success", trial.get("success", False))),
                    "trial_gt_violation": float(trial.get("safety_violation_time_s", 0.0)) > 0.0,
                    "trial_first_safety_hold_time": trial.get("first_safety_hold_time"),
                }
            )
    return rows


def _slot_invalidated(event: dict[str, Any]) -> bool:
    hold = event.get("trial_first_safety_hold_time")
    if hold is None:
        return False
    return (
        float(event.get("submitted_timestamp", math.inf)) <= float(hold)
        <= float(event.get("planned_switch_timestamp", -math.inf))
    )


def switch_outcome_attribution(events: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "accepted": 0,
        "deadline_failure": 0,
        "slot_invalidated": 0,
        "state_mismatch": 0,
        "forecast_stale": 0,
        "no_benefit": 0,
        "other_rejection": 0,
    }
    for event in events:
        if bool(event.get("candidate_accepted")):
            counts["accepted"] += 1
            continue
        if float(event.get("completed_timestamp", math.inf)) > float(event.get("deadline_timestamp", -math.inf)):
            counts["deadline_failure"] += 1
            continue
        if _slot_invalidated(event):
            counts["slot_invalidated"] += 1
            continue
        checks = (event.get("switch_validation") or {}).get("checks") or {}
        continuity_ok = bool(checks.get("continuity_q_ok", False)) and bool(checks.get("continuity_qd_ok", False)) and bool(checks.get("continuity_qdd_ok", False))
        distance_ok = bool(checks.get("distance_ok", False))
        reference_gate_ok = bool(checks.get("reference_gate_ok", False))
        if not continuity_ok:
            counts["state_mismatch"] += 1
        elif not distance_ok:
            counts["forecast_stale"] += 1
        elif not reference_gate_ok:
            counts["no_benefit"] += 1
        else:
            counts["other_rejection"] += 1
    return counts


def candidate_funnel(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[(str(event["scenario_type"]), str(event["method"]))].append(event)
    rows = []
    for (scenario_type, method), items in sorted(grouped.items()):
        rows.append(
            {
                "scenario_type": scenario_type,
                "method": method,
                "n_trigger": len(items),
                "planner_finished": int(sum(event.get("elapsed_ms") is not None for event in items)),
                "within_budget": int(
                    sum(float(event.get("elapsed_ms", math.inf)) <= 1000.0 * float(event.get("optimization_budget_s", 0.0)) for event in items)
                ),
                "optimizer_converged": int(sum(bool(event.get("optimizer_converged")) for event in items)),
                "submission_continuity": int(sum(_event_check(event, "continuity_q_ok", validation="submission_validation") for event in items)),
                "submission_online_safe": int(sum(_event_check(event, "distance_ok", validation="submission_validation") for event in items)),
                "switch_continuity": int(sum(_event_check(event, "continuity_q_ok", validation="switch_validation") for event in items)),
                "switch_online_safe": int(sum(_event_check(event, "distance_ok", validation="switch_validation") for event in items)),
                "beneficial": int(sum(bool((event.get("switch_validation") or {}).get("checks", {}).get("reference_gate_ok", False)) for event in items)),
                "switched": int(sum(bool(event.get("candidate_accepted")) for event in items)),
                "switched_and_gt_safe": int(sum(bool(event.get("candidate_accepted")) and bool(event.get("trial_task_safe")) for event in items)),
            }
        )
    return rows


def consistency_audit(trials: list[dict[str, Any]], dense: dict[str, Any]) -> dict[str, Any]:
    events = _events(trials)
    accepted_events = [event for event in events if event.get("candidate_accepted")]
    online_pass_gt_violation = [
        event["trial_id"]
        for event in accepted_events
        if event.get("trial_gt_violation")
    ]
    dense_by_trial = {row["trial_id"]: row for row in dense["per_trial"]}
    trial_medium_dense_delta = []
    for trial in trials:
        dense_row = dense_by_trial.get(trial.get("trial_id"))
        if dense_row is None or trial.get("min_distance_gt") is None:
            continue
        trial_medium_dense_delta.append(
            float(trial["min_distance_gt"]) - float(dense_row["min_distance_gt_recheck"])
        )
    return {
        "scope": "G0 pre-formal consistency audit on existing pilot/gate trials",
        "notes": [
            "candidate trajectories are not serialized in pilot trial files, so candidate dense recheck is not recomputed here",
            "submission_validation audits the candidate at optimization completion; switch_validation audits the same candidate at the planned switch state/time",
        ],
        "distance_definition": {
            "ccro_mesh_distance": "signed robot-surface to inflated obstacle-sphere clearance",
            "online_candidate_acceptance": f"medium mesh verifier, d_stop={cfg.D_ONLINE_ACCEPT}",
            "gt_recheck": f"{dense['density']} mesh replay, d_stop={cfg.D_STOP}",
            "status": "consistent at module level; candidate-level dense artifacts require replay serialization",
        },
        "time_alignment": {
            "tau_prediction_error_s": _stat([event.get("tau_prediction_error_at_switch") for event in events]),
            "deadline_miss": int(
                sum(float(event.get("completed_timestamp", math.inf)) > float(event.get("deadline_timestamp", -math.inf)) for event in events)
            ),
            "candidate_time_ok": int(sum(bool(event.get("candidate_time_ok", False)) for event in events)),
            "events": len(events),
        },
        "bridge_vs_candidate": {
            "bridge_min_distance_obs_predicted": _stat([event.get("bridge_min_distance_obs_predicted") for event in events]),
            "bridge_min_distance_gt_executed": _stat([event.get("bridge_min_distance_gt_executed") for event in events]),
            "candidate_min_distance_at_submission": _stat([
                (event.get("submission_validation") or {}).get("min_distance") for event in events
            ]),
            "candidate_min_distance_at_switch": _stat([
                (event.get("switch_validation") or {}).get("min_distance") for event in events
            ]),
            "reference_min_distance_at_switch": _stat([event.get("reference_min_distance_at_switch") for event in events]),
        },
        "medium_dense_trial_error": {
            "medium_minus_dense_min_distance_m": _stat(trial_medium_dense_delta),
            "online_pass_gt_violation_trial_ids": online_pass_gt_violation,
        },
        "candidate_funnel": candidate_funnel(events),
        "switch_outcome_attribution": switch_outcome_attribution(events),
        "go_no_go": {
            "formal_full_run": False,
            "reason": "Run candidate replay and a simple feasible validation set before any new closed-loop formal test.",
        },
    }


def write_g0_table(audit: dict[str, Any], path: Path) -> None:
    lines = [
        "# 6.4 G0 consistency audit",
        "",
        "This audit is a pre-formal implementation check. It must not be used to tune individual failed trials.",
        "",
        "## Candidate Funnel",
        "",
        "Event counts below are mechanism diagnostics. Formal method success rates must be reported by trial or paired instance, not by candidate event.",
        "",
        "| scenario | method | triggers | finished | within budget | converged | submit continuous | submit online-safe | switch continuous | switch online-safe | beneficial | switched | switched+GT safe |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit["candidate_funnel"]:
        lines.append(
            f"| {row['scenario_type']} | {cfg.METHOD_NAMES.get(row['method'], row['method'])} | "
            f"{row['n_trigger']} | {row['planner_finished']} | {row['within_budget']} | {row['optimizer_converged']} | "
            f"{row['submission_continuity']} | {row['submission_online_safe']} | "
            f"{row['switch_continuity']} | {row['switch_online_safe']} | "
            f"{row['beneficial']} | {row['switched']} | {row['switched_and_gt_safe']} |"
        )
    t = audit["time_alignment"]
    b = audit["bridge_vs_candidate"]
    e = audit["medium_dense_trial_error"]
    outcome = audit["switch_outcome_attribution"]
    lines.extend(
        [
            "",
            "## Switch Outcome Attribution",
            "",
            "| outcome | events | interpretation |",
            "|---|---:|---|",
            f"| Accepted | {outcome['accepted']} | all switch gates passed |",
            f"| Deadline failure | {outcome['deadline_failure']} | candidate completed after deadline |",
            f"| Slot invalidated | {outcome['slot_invalidated']} | safety hold occurred while candidate was pending |",
            f"| State mismatch | {outcome['state_mismatch']} | switch continuity gate failed |",
            f"| Forecast stale | {outcome['forecast_stale']} | continuity passed but distance gate failed |",
            f"| No benefit | {outcome['no_benefit']} | continuity and distance passed but reference-benefit gate failed |",
            f"| Other rejection | {outcome['other_rejection']} | remaining switch-gate failures |",
            "",
            "## G0 Checks",
            "",
            f"- Distance definition: {audit['distance_definition']['status']}.",
            f"- Time alignment: tau prediction error p95 = {_fmt(t['tau_prediction_error_s']['p95'])} s, deadline misses = {t['deadline_miss']}.",
            f"- Bridge/candidate split: bridge GT mean = {_fmt(b['bridge_min_distance_gt_executed']['mean'])} m, candidate submit mean = {_fmt(b['candidate_min_distance_at_submission']['mean'])} m, candidate switch mean = {_fmt(b['candidate_min_distance_at_switch']['mean'])} m.",
            f"- Medium/dense trial recheck: medium-minus-dense p95 = {_fmt(e['medium_minus_dense_min_distance_m']['p95'])} m; accepted-online/GT-violation trials = {len(e['online_pass_gt_violation_trial_ids'])}.",
            "",
            "## Decision",
            "",
            f"- Formal full run: {'yes' if audit['go_no_go']['formal_full_run'] else 'no'}.",
            f"- Reason: {audit['go_no_go']['reason']}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(cfg.DEFAULT_OUTPUT))
    parser.add_argument("--density", default="dense", choices=["coarse", "medium", "dense"])
    parser.add_argument("--time-step", type=float, default=0.04)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    trials = load_trials(output)
    instances = load_instances(output)
    dense = dense_gt_audit(output, trials, instances, density=args.density, time_step=float(args.time_step))
    audit = consistency_audit(trials, dense)
    write_json(output / "g0_audit_64.json", audit)
    write_g0_table(audit, output / "paper" / "table_6_4_g0_consistency_audit.md")
    print(f"[6.4 G0] saved {output / 'g0_audit_64.json'}")
    print(f"[6.4 G0] saved {output / 'paper' / 'table_6_4_g0_consistency_audit.md'}")


if __name__ == "__main__":
    main()
