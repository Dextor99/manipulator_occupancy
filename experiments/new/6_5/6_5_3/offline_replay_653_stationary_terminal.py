#!/usr/bin/env python3
"""Replay the D2-AH stationary Full CCRO-NUBS plan without robot commands."""
from __future__ import annotations

import argparse
import copy
import importlib
import json
from pathlib import Path
import numpy as np

from experiments.exp_ccro_stage2 import _load


def _translated_geometry(geometry: dict, dy_m: float) -> dict:
    g = copy.deepcopy(geometry)
    centers = np.asarray(g["component_centers"], dtype=float)
    centers[:, 1] += float(dy_m)
    g["component_centers"] = centers.tolist()
    return g


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-trial", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("config/ccro_stage4.yaml"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--geometry-json", type=Path, required=True)
    p.add_argument("--q-start-rad", default="")
    p.add_argument("--q-goal-rad", default="")
    p.add_argument("--use-local1-tail", action="store_true")
    p.add_argument("--scan-hold-dy-m", default="")
    p.add_argument("--geometry-dy-m", type=float, default=0.0)
    args = p.parse_args()
    static = importlib.import_module("experiments.new.6_5.6_5_2.run_652_static_avoidance")
    planner = importlib.import_module("experiments.new.6_5.6_5_3.stationary_terminal_ccro")
    trial = args.source_trial.resolve()
    geometry_path = args.geometry_json if args.geometry_json.is_absolute() else trial / args.geometry_json
    geometry = json.loads(geometry_path.read_text())
    if abs(float(args.geometry_dy_m)) > 1.0e-12:
        geometry = _translated_geometry(geometry, float(args.geometry_dy_m))
    event_summary = json.loads((trial / "event_replan_summary.json").read_text())
    authorization = json.loads((trial / "terminal_goal_authorization" / "authorization_summary.json").read_text())
    def parse_q(value: str) -> np.ndarray:
        q = np.asarray([float(v.strip()) for v in value.split(",") if v.strip()], dtype=float)
        if q.shape != (6,):
            raise ValueError("expected six joints")
        return q
    if args.use_local1_tail:
        q_start = np.asarray(event_summary["q_actual_local1_tail_rad"], dtype=float)
    elif args.q_start_rad:
        q_start = parse_q(args.q_start_rad)
    else:
        raise RuntimeError("provide --use-local1-tail or --q-start-rad")
    q_goal = parse_q(args.q_goal_rad) if args.q_goal_rad else np.asarray(authorization["q_goal_rad"], dtype=float)
    config = _load(args.config)
    model = static.make_surface_model(config)
    if args.scan_hold_dy_m:
        planner_config = planner._make_stationary_full_config(config, min_clearance_m=0.09)
        evaluator, _, _ = static.make_evaluator_and_verifier(planner_config, model)
        rows = []
        for dy in [float(x.strip()) for x in args.scan_hold_dy_m.split(",") if x.strip()]:
            shifted = _translated_geometry(geometry, dy)
            points = planner._sphere_points(shifted)
            obstacle = importlib.import_module("planning.mesh_risk").StaticObstacleField.from_points(points)
            goal_risk = evaluator.configuration(q_goal, obstacle, density="dense", with_gradient=False)
            start_risk = evaluator.configuration(q_start, obstacle, density="dense", with_gradient=False)
            rows.append({
                "dy_m": dy, "candidate_stop_line_y_m": -0.146 + dy,
                "goal_clearance_m": float(goal_risk.min_distance),
                "goal_nearest_link": goal_risk.nearest_link,
                "start_clearance_m": float(start_risk.min_distance),
            })
        payload = {"mode": "hold_stop_line_scan", "rows": rows}
        args.output.resolve().mkdir(parents=True, exist_ok=True)
        (args.output.resolve() / "hold_stop_line_scan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return
    payload, trajectory = planner.plan_stationary_terminal_ccro(
        config=config,
        model=model,
        q_start=q_start,
        q_goal=q_goal,
        geometry=geometry,
        output_dir=args.output.resolve(),
    )
    args.output.resolve().mkdir(parents=True, exist_ok=True)
    (args.output.resolve() / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    raise SystemExit(0 if payload.get("authorized") else 2)


if __name__ == "__main__":
    main()
