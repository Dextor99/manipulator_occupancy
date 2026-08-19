#!/usr/bin/env python3
"""Audit stationary q_goal clearance and auxiliary gripper-joint semantics.

This is diagnostic only.  It never changes the production RobotSurfaceModel.
The production model plans six AUBO joints and leaves URDF auxiliary joints at
zero; this script reproduces that result, then evaluates fixed left/right
prismatic finger positions against the archived stationary multisphere.
"""
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


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def _load(args):
    event = importlib.import_module("experiments.new.6_5.6_5_3.run_653_simple_dynamic_nubs_event_replan_live")
    core = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")
    source = event.load_stationary_terminal_replay_source(args.source_trial)
    config = core.load_stage4_config(Path("config/ccro_stage4.yaml"))
    model = core.load_stage4_surface_model(config)
    q_goal = np.asarray(source["bundle"]["q_goal_rad"], dtype=float)
    geometry = source["geometry"]
    forecast = event.v3.v3_confirmed_stationary_multisphere_forecast(
        np.asarray(geometry["component_centers"], dtype=float),
        np.asarray(geometry["component_base_radii"], dtype=float),
        valid_horizon_s=10.0,
    )
    return event, core, model, q_goal, forecast


def _spheres(forecast):
    return list(forecast.occupancy_at(0.0).spheres)


def _custom_surfaces(model, q, finger):
    """Transform cached surfaces with fixed auxiliary prismatic fingers."""
    joints = model._joint_dict(q)  # six production planning joints
    joints.update({"left_joint": float(finger), "right_joint": float(finger)})
    fk = model.urdf.link_transforms(joints)
    result = {}
    for link in model.link_names:
        transform = fk.get(link)
        if transform is not None:
            local = model._local[link]["dense"]
            result[link] = local @ transform[:3, :3].T + transform[:3, 3]
    return result


def _evaluate(model, q, forecast, finger):
    spheres = _spheres(forecast)
    by_link = _custom_surfaces(model, q, finger)
    best = None
    per_link = {}
    for link, points in by_link.items():
        link_best = None
        for sphere_index, sphere in enumerate(spheres):
            vectors = points - np.asarray(sphere.center)[None, :]
            radial = np.linalg.norm(vectors, axis=1)
            index = int(np.argmin(radial - float(sphere.radius)))
            clearance = float(radial[index] - float(sphere.radius))
            if link_best is None or clearance < link_best["clearance_m"]:
                direction = vectors[index]
                norm = float(np.linalg.norm(direction))
                direction = direction / norm if norm > 1.0e-12 else np.array([1.0, 0.0, 0.0])
                link_best = {
                    "clearance_m": clearance,
                    "sphere_index": sphere_index,
                    "sphere_object_id": int(sphere.object_id),
                    "robot_surface_point_xyz": points[index],
                    "obstacle_surface_point_xyz": np.asarray(sphere.center) + float(sphere.radius) * direction,
                    "center_to_robot_distance_m": float(radial[index]),
                    "sphere_center_xyz": np.asarray(sphere.center),
                    "sphere_radius_m": float(sphere.radius),
                }
        per_link[link] = link_best
        if best is None or link_best["clearance_m"] < best["clearance_m"]:
            best = dict(link_best, nearest_link=link)
    return {"minimum": best, "per_link": per_link}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"AUDIT_OUTPUT_NOT_EMPTY:{args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    event, core, model, q_goal, forecast = _load(args)
    production = event.check_goal_configuration_feasibility(
        config=core.load_stage4_config(Path("config/ccro_stage4.yaml")),
        model=model, q_goal=q_goal, forecast=forecast, min_clearance_m=0.09,
    )
    rows = []
    for finger in (0.0, -0.01, -0.02, -0.03, -0.04):
        result = _evaluate(model, q_goal, forecast, finger)
        rows.append({"left_joint_m": finger, "right_joint_m": finger, **result})
    payload = {
        "status": "GOAL_CLEARANCE_GEOMETRY_AUDIT_COMPLETE",
        "source_trial": str(args.source_trial.resolve()),
        "q_goal_rad": q_goal,
        "production_zero_auxiliary_joint_check": production,
        "auxiliary_joint_sweep": rows,
        "auxiliary_joint_semantics": "fixed left_joint/right_joint applied only for diagnostic FK; production model unchanged",
    }
    (args.output / "goal_clearance_geometry_audit.json").write_text(
        json.dumps(payload, indent=2, default=_json), encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "production": production, "sweep": [{"finger_m": x["left_joint_m"], "clearance_m": x["minimum"]["clearance_m"], "nearest_link": x["minimum"]["nearest_link"]} for x in rows]}, indent=2, default=_json))


if __name__ == "__main__":
    main()
