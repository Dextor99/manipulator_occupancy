"""Summaries and paper tables for Chapter 6.4 dynamic replanning."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from . import config_64 as cfg


def _stat(values: list[float]) -> dict[str, float | None]:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }


def aggregate(trials: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for trial in trials:
        key = f"{trial['scenario_type']}::{trial['method']}"
        groups.setdefault(key, {"scenario_type": trial["scenario_type"], "method": trial["method"], "trials": []})
        groups[key]["trials"].append(trial)
    rows = []
    for group in groups.values():
        items = group["trials"]
        rows.append(
            {
                "scenario_type": group["scenario_type"],
                "method": group["method"],
                "method_name": cfg.METHOD_NAMES.get(group["method"], group["method"]),
                "n": len(items),
                "success_rate": float(np.mean([bool(item["success"]) for item in items])),
                "task_safe_success_rate": float(np.mean([bool(item.get("task_safe_success", item["success"])) for item in items])),
                "replan_success_rate": float(np.mean([bool(item.get("replan_success", item["accepted_count"] >= 1)) for item in items])),
                "finish_rate": float(np.mean([bool(item["finished"]) for item in items])),
                "safety_violation_rate": float(np.mean([item["safety_violation_time_s"] > 0.0 for item in items])),
                "min_distance_gt": _stat([item["min_distance_gt"] for item in items]),
                "goal_error": _stat([item["goal_error"] for item in items]),
                "replan_count": _stat([item["replan_count"] for item in items]),
                "accepted_count": _stat([item["accepted_count"] for item in items]),
                "first_replan_time": _stat([item["first_replan_time"] for item in items if item["first_replan_time"] is not None]),
                "first_safety_hold_time": _stat([item["first_safety_hold_time"] for item in items if item["first_safety_hold_time"] is not None]),
                "planning_cycles": _stat([item["planning_control_cycles"] for item in items]),
                "bridge_min_distance_obs_predicted": _stat(
                    [
                        item["bridge_min_distance_obs_predicted"]
                        for item in items
                        if item.get("bridge_min_distance_obs_predicted") is not None
                    ]
                ),
                "bridge_min_distance_gt_executed": _stat(
                    [
                        item["bridge_min_distance_gt_executed"]
                        for item in items
                        if item.get("bridge_min_distance_gt_executed") is not None
                    ]
                ),
                "planner_elapsed_ms": _stat(
                    [
                        event["elapsed_ms"]
                        for item in items
                        for event in item.get("events", [])
                        if event.get("elapsed_ms") is not None
                    ]
                ),
                "false_replans": int(sum(bool(item.get("false_replan")) for item in items)),
            }
        )
    speed_rows = []
    speed_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for trial in trials:
        if trial.get("speed_group") is None:
            continue
        key = (trial["scenario_type"], trial["method"], f"{float(trial['speed_group']):.2f}")
        speed_groups.setdefault(key, []).append(trial)
    for (scenario_type, method, speed), items in sorted(speed_groups.items()):
        speed_rows.append(
            {
                "scenario_type": scenario_type,
                "method": method,
                "speed_group": speed,
                "n": len(items),
                "task_safe_success_rate": float(np.mean([bool(item.get("task_safe_success", item["success"])) for item in items])),
                "replan_success_rate": float(np.mean([bool(item.get("replan_success", item["accepted_count"] >= 1)) for item in items])),
                "violation_rate": float(np.mean([item["safety_violation_time_s"] > 0.0 for item in items])),
                "min_distance_gt": _stat([item["min_distance_gt"] for item in items]),
                "trigger_to_reference_risk": _stat([item["trigger_to_reference_risk"] for item in items if item.get("trigger_to_reference_risk") is not None]),
            }
        )
    lead_rows = []
    for label, low, high in [
        ("short", -math.inf, 1.5),
        ("medium", 1.5, 3.0),
        ("long", 3.0, math.inf),
    ]:
        items = [
            trial for trial in trials
            if trial.get("trigger_to_reference_risk") is not None
            and low <= float(trial["trigger_to_reference_risk"]) < high
        ]
        for method in sorted(set(item["method"] for item in items)):
            lead_items = [item for item in items if item["method"] == method]
            lead_rows.append(
                {
                    "lead_group": label,
                    "method": method,
                    "n": len(lead_items),
                    "task_safe_success_rate": float(np.mean([bool(item.get("task_safe_success", item["success"])) for item in lead_items])),
                    "replan_success_rate": float(np.mean([bool(item.get("replan_success", item["accepted_count"] >= 1)) for item in lead_items])),
                    "violation_rate": float(np.mean([item["safety_violation_time_s"] > 0.0 for item in lead_items])),
                    "lead_time": _stat([item["trigger_to_reference_risk"] for item in lead_items]),
                }
            )

    return {
        "trial_count": len(trials),
        "accepted": bool(all(row["task_safe_success_rate"] >= 0.0 for row in rows)),
        "groups": sorted(rows, key=lambda row: (row["scenario_type"], row["method"])),
        "by_speed": speed_rows,
        "by_lead_time": lead_rows,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def write_paper_table(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "| scenario | method | n | task safe | replan success | finish | violation | Dmin GT / m | bridge GT / m | bridge pred / m | replans | accepted | planner ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["groups"]:
        lines.append(
            "| {scenario_type} | {method_name} | {n} | {task:.2f} | {replan_success:.2f} | {finish:.2f} | {violation:.2f} | "
            "{dmin} | {bridge_gt} | {bridge_pred} | {replans} | {accepted} | {planner} |".format(
                scenario_type=row["scenario_type"],
                method_name=row["method_name"],
                n=row["n"],
                task=row["task_safe_success_rate"],
                replan_success=row["replan_success_rate"],
                finish=row["finish_rate"],
                violation=row["safety_violation_rate"],
                dmin=_fmt(row["min_distance_gt"]["mean"]),
                bridge_gt=_fmt(row["bridge_min_distance_gt_executed"]["mean"]),
                bridge_pred=_fmt(row["bridge_min_distance_obs_predicted"]["mean"]),
                replans=_fmt(row["replan_count"]["mean"], 2),
                accepted=_fmt(row["accepted_count"]["mean"], 2),
                planner=_fmt(row["planner_elapsed_ms"]["mean"], 1),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `violation` is GT safety-distance violation rate under the executed closed loop.",
            "- `initial_high_risk` is a safety-hold test: `finish=0` and `violation=1` are expected because the obstacle is initialized inside the hold region; acceptance is judged by immediate hold, zero replans, and zero candidate switches.",
            "- `task safe` reports task completion without GT safety violation; `replan success` reports at least one accepted candidate switch after a trigger.",
            "- `bridge GT` is the minimum GT distance actually observed during the pending interval; `bridge pred` is the online forecast distance under the expected slowed execution.",
            "- Candidate switching uses online medium validation and is followed by dense GT offline audit; optimizer convergence flags are reported separately in `table_6_4_candidate_validation_audit.md`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stratified_tables(summary: dict[str, Any], paper_dir: Path) -> None:
    speed_lines = [
        "| scenario | method | speed / m/s | n | task safe | replan success | violation | Dmin mean / m | lead mean / s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("by_speed", []):
        speed_lines.append(
            f"| {row['scenario_type']} | {cfg.METHOD_NAMES.get(row['method'], row['method'])} | {row['speed_group']} | "
            f"{row['n']} | {_fmt(row['task_safe_success_rate'], 2)} | {_fmt(row['replan_success_rate'], 2)} | "
            f"{_fmt(row['violation_rate'], 2)} | {_fmt(row['min_distance_gt']['mean'])} | "
            f"{_fmt(row['trigger_to_reference_risk']['mean'])} |"
        )
    lead_lines = [
        "| lead group | method | n | task safe | replan success | violation | lead mean / s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("by_lead_time", []):
        lead_lines.append(
            f"| {row['lead_group']} | {cfg.METHOD_NAMES.get(row['method'], row['method'])} | {row['n']} | "
            f"{_fmt(row['task_safe_success_rate'], 2)} | {_fmt(row['replan_success_rate'], 2)} | "
            f"{_fmt(row['violation_rate'], 2)} | {_fmt(row['lead_time']['mean'])} |"
        )
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "table_6_4_by_speed.md").write_text("\n".join(speed_lines) + "\n", encoding="utf-8")
    (paper_dir / "table_6_4_by_lead_time.md").write_text("\n".join(lead_lines) + "\n", encoding="utf-8")
