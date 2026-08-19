#!/usr/bin/env python3
"""Batch replay the stationary boundary-terminal planner on archived trials.

This is diagnostic only: it never connects to a robot and never commands
motion.  Each source trial is replayed into its own output directory and the
compact authorization fields are collected into ``benchmark_summary.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
REPLAY = ROOT / "experiments/new/6_5/6_5_3/offline_replay_653_stationary_fast_terminal.py"


def _summary(path: Path) -> dict:
    for candidate in (path / "authorization_summary.json", path / "summary.json"):
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
    return {}


def _source_goal_feasibility(source: Path) -> dict:
    for name in ("event_replan_summary.json", "summary.json"):
        path = source / name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            value = data.get("stationary_goal_feasibility")
            if isinstance(value, dict):
                return value
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--trials", default="r19,r20,r21,r22")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in (item.strip() for item in args.trials.split(",")):
        source = args.source_root / label / "core_live" / "trials" / f"D2_opposing_approach_{label}"
        target = args.output / label
        target.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(REPLAY), "--source-trial", str(source), "--output", str(target)]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        payload = _summary(target / "stationary_fast_terminal_bypass")
        if not payload:
            payload = _summary(target)
        rows.append({
            "trial": label,
            "source_exists": source.exists(),
            "q_goal_feasible": _source_goal_feasibility(source).get("feasible"),
            "q_goal_clearance_m": _source_goal_feasibility(source).get("goal_clearance_m"),
            "replay_returncode": int(completed.returncode),
            "status": payload.get("status"),
            "authorized": payload.get("authorized"),
            "verification_min_distance_m": payload.get("verification_min_distance_m"),
            "stationary_terminal_total_elapsed_ms": payload.get("stationary_terminal_total_elapsed_ms"),
            "connected_route_count": (payload.get("virtual_fast_route") or {}).get("boundary_audit", {}).get("connected_route_count"),
            "selected_direction": (payload.get("virtual_fast_route") or {}).get("selected_direction"),
            "stderr_tail": completed.stderr[-1000:],
        })
    result = {"status": "STATIONARY_BOUNDARY_ROUTE_BENCHMARK_COMPLETE", "rows": rows}
    (args.output / "benchmark_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
