#!/usr/bin/env python3
"""Offline joint-space path-existence diagnostic for stationary D2 geometry.

This is deliberately not a production planner.  It answers only whether a
bounded, conservative joint-space sampling search can connect the captured
q_start and q_goal using the same stationary multisphere and tabletop guards.
No robot or camera is opened and no trajectory is authorized.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import importlib
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _edge(event, evaluator, forecast, model, q0, q1, *, floor, min_z, samples):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    screen = event.sampled_joint_segment_clearance(evaluator, forecast, q0, q1, samples=samples)
    if float(screen["min_distance_m"]) < float(floor):
        return False, screen
    tcp_link = "gripper_base_link"
    min_tcp_z = min(
        float(event.live.simple.tcp_position(model, (1.0 - a) * q0 + a * q1, tcp_link)[2])
        for a in np.linspace(0.0, 1.0, max(2, samples))
    )
    screen["min_tcp_z_m"] = min_tcp_z
    screen["accepted"] = min_tcp_z >= float(min_z)
    return bool(screen["accepted"]), screen


def _nearest(nodes, q):
    return min(range(len(nodes)), key=lambda i: float(np.linalg.norm(nodes[i]["q"] - q)))


def _extend(event, nodes, target, *, evaluator, forecast, model, limits, floor, min_z, step, edge_samples):
    index = _nearest(nodes, target)
    q0 = nodes[index]["q"]
    delta = np.asarray(target, dtype=np.float64) - q0
    norm = float(np.linalg.norm(delta))
    q1 = np.asarray(target, dtype=np.float64) if norm <= step else q0 + delta * (float(step) / norm)
    q1 = np.clip(q1, limits.q_min, limits.q_max)
    ok, audit = _edge(event, evaluator, forecast, model, q0, q1, floor=floor, min_z=min_z, samples=edge_samples)
    if not ok:
        return None
    nodes.append({"q": q1, "parent": index, "edge": audit})
    return len(nodes) - 1


def _trace(nodes, index):
    path = []
    while index is not None:
        path.append(nodes[index]["q"])
        index = nodes[index]["parent"]
    return np.asarray(path[::-1], dtype=np.float64)


def run_trial(source_trial: Path, *, seed: int, iterations: int, step: float, edge_samples: int,
              floor: float, min_z: float, goal_tolerance: float) -> dict:
    event = importlib.import_module("experiments.new.6_5.6_5_3.run_653_simple_dynamic_nubs_event_replan_live")
    core = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")
    source = event.load_stationary_terminal_replay_source(source_trial)
    bundle, geometry = source["bundle"], source["geometry"]
    q_start = np.asarray(bundle["q_terminal_start_rad"], dtype=np.float64)
    q_goal = np.asarray(bundle["q_goal_rad"], dtype=np.float64)
    config = core.load_stage4_config(Path("config/ccro_stage4.yaml"))
    model = core.load_stage4_surface_model(config)
    forecast = event.v3.v3_confirmed_stationary_multisphere_forecast(
        np.asarray(geometry["component_centers"], dtype=np.float64),
        np.asarray(geometry["component_base_radii"], dtype=np.float64),
        valid_horizon_s=10.0,
    )
    # make_risk_stack returns evaluator, verifier, limits; the oracle needs
    # evaluator and limits, and deliberately never calls the verifier.
    evaluator, _, limits = core.make_risk_stack(config, model, forecast)
    rng = np.random.default_rng(int(seed))
    start_nodes = [{"q": q_start.copy(), "parent": None, "edge": None}]
    goal_nodes = [{"q": q_goal.copy(), "parent": None, "edge": None}]
    best_gap = float(np.linalg.norm(q_start - q_goal))
    connected = False
    start_goal_edge = _edge(event, evaluator, forecast, model, q_start, q_goal,
                            floor=floor, min_z=min_z, samples=edge_samples)
    for iteration in range(int(iterations)):
        # Alternating goal bias makes this diagnostic finite and reproducible.
        if iteration % 7 == 0:
            sample = q_goal.copy()
        else:
            sample = rng.uniform(limits.q_min, limits.q_max)
        ia = _extend(event, start_nodes, sample, evaluator=evaluator, forecast=forecast,
                     model=model, limits=limits, floor=floor, min_z=min_z,
                     step=step, edge_samples=edge_samples)
        if ia is None:
            continue
        q_new = start_nodes[ia]["q"]
        best_gap = min(best_gap, float(np.linalg.norm(q_new - q_goal)))
        ib = _extend(event, goal_nodes, q_new, evaluator=evaluator, forecast=forecast,
                     model=model, limits=limits, floor=floor, min_z=min_z,
                     step=step, edge_samples=edge_samples)
        if ib is not None and float(np.linalg.norm(goal_nodes[ib]["q"] - q_new)) <= float(step) + 1.0e-9:
            ok, join = _edge(event, evaluator, forecast, model, q_new, goal_nodes[ib]["q"],
                             floor=floor, min_z=min_z, samples=edge_samples)
            if ok:
                connected = True
                break
    path = None
    if connected:
        path = np.vstack([_trace(start_nodes, ia), _trace(goal_nodes, ib)[::-1]])
    return {
        "seed": int(seed), "iterations": int(iterations), "step_rad": float(step),
        "edge_samples": int(edge_samples), "topology_floor_m": float(floor),
        "min_tcp_z_m": float(min_z), "goal_tolerance_rad": float(goal_tolerance),
        "q_start": q_start.tolist(), "q_goal": q_goal.tolist(),
        "direct_edge": start_goal_edge[0], "connected": bool(connected),
        "best_joint_goal_gap_rad": float(best_gap),
        "node_count_start": len(start_nodes), "node_count_goal": len(goal_nodes),
        "path_points": int(len(path)) if path is not None else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="11,23,47")
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--step-rad", type=float, default=0.10)
    parser.add_argument("--edge-samples", type=int, default=7)
    parser.add_argument("--topology-floor-m", type=float, default=0.08)
    parser.add_argument("--min-tcp-z-m", type=float, default=0.46)
    parser.add_argument("--goal-tolerance-rad", type=float, default=0.10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [run_trial(args.source_trial, seed=int(seed), iterations=args.iterations,
                      step=args.step_rad, edge_samples=args.edge_samples,
                      floor=args.topology_floor_m, min_z=args.min_tcp_z_m,
                      goal_tolerance=args.goal_tolerance_rad)
            for seed in args.seeds.split(",") if seed.strip()]
    payload = {"status": "OFFLINE_PATH_EXISTENCE_ORACLE_COMPLETE", "rows": rows,
               "any_connected": any(row["connected"] for row in rows),
               "production_planner": False, "robot_commanded": False}
    (args.output / "oracle_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["any_connected"] else 2)


if __name__ == "__main__":
    main()
