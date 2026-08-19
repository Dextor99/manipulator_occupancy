#!/usr/bin/env python3
"""Replay a captured stationary terminal state without commanding the robot."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-trial", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"REPLAY_OUTPUT_NOT_EMPTY:{output}")
    output.mkdir(parents=True, exist_ok=True)
    event = importlib.import_module("experiments.new.6_5.6_5_3.run_653_simple_dynamic_nubs_event_replan_live")
    core = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")
    source = event.load_stationary_terminal_replay_source(args.source_trial)
    bundle, geometry, snapshot = source["bundle"], source["geometry"], source["snapshot"]
    core_args = core.build_parser().parse_args(["--scene", "D2", "--repeat", "16"])
    config = core.load_stage4_config(Path("config/ccro_stage4.yaml"))
    model = core.load_stage4_surface_model(config)
    q_start = np.asarray(bundle["q_terminal_start_rad"], dtype=float)
    q_goal = np.asarray(bundle["q_goal_rad"], dtype=float)
    forecast = event.v3.v3_confirmed_stationary_multisphere_forecast(
        np.asarray(geometry["component_centers"], dtype=float),
        np.asarray(geometry["component_base_radii"], dtype=float),
        valid_horizon_s=10.0,
    )
    goal_check = event.check_goal_configuration_feasibility(
        config=config, model=model, q_goal=q_goal, forecast=forecast,
        min_clearance_m=float(getattr(core_args, "online_accept_m", 0.09)),
    )
    source_goal = bundle["stationary_goal_feasibility"].get("goal_clearance_m")
    if not goal_check.get("feasible", False):
        payload = {"status": "REPLAY_SOURCE_GOAL_INFEASIBLE", "authorized": False,
                   "robot_commanded": False, "goal_check": goal_check,
                   "source_goal_clearance_m": source_goal}
    elif source_goal is not None and abs(float(goal_check["goal_clearance_m"]) - float(source_goal)) > 0.005:
        payload = {"status": "REPLAY_SOURCE_GEOMETRY_MISMATCH", "authorized": False,
                   "robot_commanded": False, "goal_check": goal_check,
                   "source_goal_clearance_m": source_goal}
    else:
        core_args.stationary_fast_terminal_target_ms = 5000.0
        core_args.stationary_fast_terminal_max_ms = 7000.0
        core_args.stationary_fast_terminal_route_max_ms = 0.0
        core_args.stationary_fast_terminal_virtual_fast_steps = 0
        payload, trajectory = event.plan_stationary_fast_terminal_bypass(
            core.run_fast_repair, core_args, config, model,
            q_start=q_start, q_goal=q_goal, fresh=snapshot["fresh"], geometry=geometry,
            q_escape_start=q_start, trial_dir=output / "virtual_fast_route",
            nominal_reference_goal=(q_goal, np.zeros(6), np.zeros(6)),
            risk_links=set(), artifacts_out={},
        )
        payload["robot_commanded"] = False
        payload["replay_source_audit"] = {"bundle": "PASS", "geometry": "PASS",
                                           "goal_feasible": "PASS", "goal_clearance_consistent": "PASS"}
    (output / "authorization_summary.json").write_text(json.dumps(payload, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)), encoding="utf-8")
    print(json.dumps({k: payload.get(k) for k in ("status", "authorized", "robot_commanded", "reason", "goal_check")}, indent=2))
    raise SystemExit(0 if payload.get("authorized", False) else 2)


if __name__ == "__main__":
    main()
