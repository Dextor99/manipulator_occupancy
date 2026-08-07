#!/usr/bin/env python3
"""Rank 6.5.2 static CCRO-NUBS candidates by tabletop execution cost.

The execution rule remains conservative: only candidates that pass solver,
dense full-body safety, joint limits, continuity, and goal checks are
executable.  Among feasible candidates this script uses obstacle-relative path
classification and a fixed-scale layered selection rule.  It deliberately avoids
candidate-set min/max normalization because adding an unrelated candidate should
not change the score of an existing candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from experiments.exp_ccro_stage2 import _load  # noqa: E402
from planning.mesh_risk import StaticObstacleField  # noqa: E402
from planning.nubs_trajectory import NUBSTrajectory6D  # noqa: E402
from plan_652_static_ccro_nubs_from_trial import trajectory_preference_metrics  # noqa: E402
from run_652_static_avoidance import make_evaluator_and_verifier, make_surface_model  # noqa: E402


DEFAULT_TRIAL_DIR = (
    ROOT
    / "results"
    / "new"
    / "6_5"
    / "6_5_2"
    / "planar_static_live"
    / "rs1_lateral_table_obstacle"
    / "trials"
    / "rs1_lateral_table_obstacle_r08"
)

ACTIVE_LINKS = (
    "upperArm_Link",
    "foreArm_Link",
    "wrist1_Link",
    "wrist2_Link",
    "wrist3_Link",
    "gripper_base_link",
    "left_link",
    "right_link",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_trajectory(npz_path: Path, key: str) -> NUBSTrajectory6D:
    data = np.load(npz_path)
    head = NUBSTrajectory6D.make_boundary_state(data["q_start"])
    tail = NUBSTrajectory6D.make_boundary_state(data["q_goal"])
    return NUBSTrajectory6D().generate(data[key], head, tail, data["durations"])


def get_nested(payload: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    node: Any = payload
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def geometric_dense_ok(checks: dict[str, bool]) -> bool:
    return bool(all(value for key, value in checks.items() if key != "solver_ok"))


def clearance_penalty(distances: np.ndarray, args: argparse.Namespace) -> float:
    denom = max(args.clearance_pref_m - args.clearance_m, 1.0e-9)
    values = np.maximum((args.clearance_pref_m - distances) / denom, 0.0)
    return float(np.mean(values * values))


def path_straight_distance(surface_model: Any, trajectory: NUBSTrajectory6D, tcp_link: str) -> float:
    def tcp(q: np.ndarray) -> np.ndarray:
        fk = surface_model.urdf.link_transforms({name: float(q[i]) for i, name in enumerate(surface_model.joint_names)})
        return np.asarray(fk[tcp_link][:3, 3], dtype=np.float64)

    start = tcp(trajectory.evaluate(0.0))
    goal = tcp(trajectory.evaluate(trajectory.total_duration))
    return float(max(np.linalg.norm(goal - start), 1.0e-9))


def tcp_xyz_samples(surface_model: Any, trajectory: NUBSTrajectory6D, tcp_link: str, samples: int) -> np.ndarray:
    points = []
    times = np.linspace(0.0, trajectory.total_duration, max(2, int(samples)))
    for t in times:
        q = trajectory.evaluate(float(t))
        fk = surface_model.urdf.link_transforms({name: float(q[i]) for i, name in enumerate(surface_model.joint_names)})
        points.append(np.asarray(fk[tcp_link][:3, 3], dtype=np.float64))
    return np.vstack(points)


def robust_obstacle_geometry(
    trial_dir: Path,
    obstacle_points: np.ndarray,
) -> dict[str, Any]:
    bbox_min = np.min(obstacle_points, axis=0)
    bbox_max = np.max(obstacle_points, axis=0)
    try:
        trial_summary = load_json(trial_dir / "summary.json")
    except FileNotFoundError:
        trial_summary = {}
    table_z = float(trial_summary.get("table_z_m", np.percentile(obstacle_points[:, 2], 2)))
    z = obstacle_points[:, 2]
    return {
        "table_z_m": table_z,
        "obstacle_bbox_min": bbox_min.tolist(),
        "obstacle_bbox_max": bbox_max.tolist(),
        "obstacle_top_raw_max_m": float(np.max(z)),
        "obstacle_top_p95_m": float(np.percentile(z, 95)),
        "obstacle_top_p99_m": float(np.percentile(z, 99)),
        "obstacle_bottom_p01_m": float(np.percentile(z, 1)),
        "robust_obstacle_height_p99_m": float(np.percentile(z, 99) - table_z),
    }


def route_geometry_audit(
    trial_dir: Path,
    reference: NUBSTrajectory6D,
    candidate: NUBSTrajectory6D,
    obstacle_points: np.ndarray,
    surface_model: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    geometry = robust_obstacle_geometry(trial_dir, obstacle_points)
    bbox_min = np.asarray(geometry["obstacle_bbox_min"], dtype=np.float64)
    bbox_max = np.asarray(geometry["obstacle_bbox_max"], dtype=np.float64)
    robust_top_z = float(geometry["obstacle_top_p99_m"])
    ref_tcp = tcp_xyz_samples(surface_model, reference, args.tcp_link, args.samples)
    cand_tcp = tcp_xyz_samples(surface_model, candidate, args.tcp_link, args.samples)
    ref_z = float(ref_tcp[0, 2])
    max_z_dev = float(np.max(np.abs(cand_tcp[:, 2] - ref_tcp[:, 2])))
    max_xy_dev = float(np.max(np.linalg.norm(cand_tcp[:, :2] - ref_tcp[:, :2], axis=1)))
    xy_inflation = float(args.overpass_xy_inflation_m)
    tcp_inside_xy = (
        (cand_tcp[:, 0] >= bbox_min[0] - xy_inflation)
        & (cand_tcp[:, 0] <= bbox_max[0] + xy_inflation)
        & (cand_tcp[:, 1] >= bbox_min[1] - xy_inflation)
        & (cand_tcp[:, 1] <= bbox_max[1] + xy_inflation)
    )

    swept_inside_count = 0
    swept_min_z_inside = float("inf")
    swept_max_z_inside = -float("inf")
    route_samples = max(2, int(args.route_samples))
    for t in np.linspace(0.0, candidate.total_duration, route_samples):
        q = candidate.evaluate(float(t))
        by_link = surface_model.surface_by_link(q, density=args.route_density, links=set(ACTIVE_LINKS))
        for points in by_link.values():
            inside = (
                (points[:, 0] >= bbox_min[0] - xy_inflation)
                & (points[:, 0] <= bbox_max[0] + xy_inflation)
                & (points[:, 1] >= bbox_min[1] - xy_inflation)
                & (points[:, 1] <= bbox_max[1] + xy_inflation)
            )
            if np.any(inside):
                z_inside = points[inside, 2]
                swept_inside_count += int(np.count_nonzero(inside))
                swept_min_z_inside = min(swept_min_z_inside, float(np.min(z_inside)))
                swept_max_z_inside = max(swept_max_z_inside, float(np.max(z_inside)))
    swept_crosses = swept_inside_count > 0
    required_top_z = robust_top_z + float(args.clearance_m) + float(args.vertical_uncertainty_m)
    overpass_clearance_ok = bool(swept_crosses and swept_min_z_inside >= required_top_z)
    large_vertical_motion = bool(max_z_dev > args.overpass_z_deviation_m)
    lateral_motion = bool(max_xy_dev > args.lateral_xy_deviation_m)
    if swept_crosses and large_vertical_motion and overpass_clearance_ok:
        route_class = "true_overpass"
    elif swept_crosses and large_vertical_motion:
        route_class = "footprint_crossing_vertical"
    elif lateral_motion and large_vertical_motion:
        route_class = "hybrid_vertical_lateral"
    elif lateral_motion:
        route_class = "lateral"
    elif large_vertical_motion:
        route_class = "vertical_without_footprint_crossing"
    else:
        route_class = "near_reference"

    if swept_crosses and not overpass_clearance_ok:
        feasibility_note = "swept_footprint_crossing_without_required_vertical_margin"
    elif route_class == "true_overpass":
        feasibility_note = "true_overpass_has_required_vertical_margin"
    else:
        feasibility_note = "no_overpass_footprint_crossing"
    return {
        **geometry,
        "reference_tcp_start_z_m": ref_z,
        "obstacle_top_p99_minus_reference_tcp_start_z_m": float(robust_top_z - ref_z),
        "candidate_max_tcp_z_deviation_m": max_z_dev,
        "candidate_max_tcp_xy_deviation_m": max_xy_dev,
        "overpass_z_deviation_threshold_m": args.overpass_z_deviation_m,
        "lateral_xy_deviation_threshold_m": args.lateral_xy_deviation_m,
        "xy_inflation_m": xy_inflation,
        "tcp_samples_inside_inflated_obstacle_xy": int(np.count_nonzero(tcp_inside_xy)),
        "tcp_crosses_inflated_obstacle_xy": bool(np.any(tcp_inside_xy)),
        "robot_surface_points_inside_inflated_obstacle_xy": swept_inside_count,
        "robot_surface_crosses_inflated_obstacle_xy": swept_crosses,
        "robot_surface_min_z_inside_footprint_m": None if not swept_crosses else swept_min_z_inside,
        "robot_surface_max_z_inside_footprint_m": None if not swept_crosses else swept_max_z_inside,
        "required_overpass_surface_z_m": required_top_z,
        "large_vertical_motion": large_vertical_motion,
        "lateral_motion": lateral_motion,
        "overpass_clearance_ok": overpass_clearance_ok,
        "route_class": route_class,
        "route_feasibility_note": feasibility_note,
    }


def time_scaling_audit(candidate: NUBSTrajectory6D, config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    robot = config["robot"]
    qd_max = np.asarray(robot["qd_max"], dtype=np.float64)
    qdd_max = np.asarray(robot["qdd_max"], dtype=np.float64)
    count = max(2, int(np.ceil(candidate.total_duration / args.time_scale_dt)) + 1)
    times = np.linspace(0.0, candidate.total_duration, count)
    samples = candidate.sample(times, max_derivative=2)
    max_qd_ratio = float(np.max(np.abs(samples.qd) / np.maximum(qd_max[None, :], 1.0e-9)))
    max_qdd_ratio = float(np.max(np.abs(samples.qdd) / np.maximum(qdd_max[None, :], 1.0e-9)))
    scale = float(max(1.0, max_qd_ratio, np.sqrt(max_qdd_ratio)))
    return {
        "time_scale_factor": scale,
        "required_execution_time_s": float(candidate.total_duration * scale),
        "max_qd_ratio": max_qd_ratio,
        "max_qdd_ratio": max_qdd_ratio,
        "nominal_duration_s": float(candidate.total_duration),
    }


def fixed_scale_costs(raw: dict[str, float], straight: float, reference_joint_path: float, reference_jerk: float) -> dict[str, float]:
    return {
        "tcp_extra_ratio": float(max(raw["tcp_path_length_m"] / max(straight, 1.0e-9) - 1.0, 0.0)),
        "joint_path_ratio": float(raw["joint_path_length_rad"] / max(reference_joint_path, 1.0e-9)),
        "jerk_ratio": float(raw["jerk_energy"] / max(reference_jerk, 1.0e-9)),
        "clearance_penalty": float(raw["clearance_penalty"]),
        "required_execution_time_s": float(raw["required_execution_time_s"]),
    }


def reference_motion_scales(reference: NUBSTrajectory6D, surface_model: Any, tcp_link: str, samples: int) -> dict[str, float]:
    times = np.linspace(0.0, reference.total_duration, max(2, int(samples)))
    q = reference.sample(times, max_derivative=0).q
    joint_step = np.linalg.norm(np.diff(q, axis=0), axis=1) if len(q) > 1 else np.zeros(1)
    return {
        "reference_joint_path_length_rad": float(max(np.sum(joint_step), 1.0e-9)),
        "reference_jerk_energy": float(max(reference.energy(), 1.0e-9)),
        "reference_duration_s": float(reference.total_duration),
        "reference_tcp_straight_distance_m": path_straight_distance(surface_model, reference, tcp_link),
    }


def dense_clearance_audit(candidate: NUBSTrajectory6D, obstacle_points: np.ndarray, evaluator: Any, config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    obstacle = StaticObstacleField.from_points(obstacle_points)
    dt = float(config["validation"]["dense_time_step"])
    count = max(2, int(np.ceil(candidate.total_duration / dt)) + 1)
    times = np.linspace(0.0, candidate.total_duration, count)
    risk = evaluator.trajectory(
        candidate,
        obstacle,
        times,
        density=config["risk"]["validation_density"],
        with_gradient=False,
    )
    return {
        "clearance_penalty": clearance_penalty(risk.sample_distances, args),
        "dense_sample_count": int(len(times)),
        "dense_active_sample_count": int(risk.active_sample_count),
        "dense_min_distance_recomputed_m": float(risk.min_distance),
        "dense_nearest_link_recomputed": risk.nearest_link,
    }


def normalize_values(items: list[dict[str, Any]], key: str) -> None:
    values = np.asarray([float(item["raw_objectives"][key]) for item in items], dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        for item in items:
            item["normalized_objectives"][key] = 1.0
        return
    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    if max_value - min_value <= 1.0e-12:
        for item in items:
            item["normalized_objectives"][key] = 0.0
        return
    for item in items:
        item["normalized_objectives"][key] = float((float(item["raw_objectives"][key]) - min_value) / (max_value - min_value))


def dominates(a: dict[str, Any], b: dict[str, Any], keys: list[str], eps: float = 1.0e-12) -> bool:
    av = a["raw_objectives"]
    bv = b["raw_objectives"]
    no_worse = all(float(av[key]) <= float(bv[key]) + eps for key in keys)
    strictly_better = any(float(av[key]) < float(bv[key]) - eps for key in keys)
    return bool(no_worse and strictly_better)


def summarize_plan(plan_dir: Path, surface_model: Any, evaluator: Any, config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    summary = load_json(plan_dir / "summary.json")
    data_path = plan_dir / "ccro_nubs_trajectories.npz"
    data = np.load(data_path)
    reference = load_trajectory(data_path, "reference_inner")
    candidate = load_trajectory(data_path, "candidate_inner")
    metrics = trajectory_preference_metrics(
        surface_model,
        reference,
        candidate,
        args.tcp_link,
        samples=args.samples,
    )
    dense = get_nested(summary, ["candidate", "dense_verification"], {})
    checks = dense.get("checks", {})
    z_stats = get_nested(summary, ["candidate", "tcp_z_stats"], {})
    strict_ok = bool(summary.get("accepted_for_real_execution", False))
    geometric_ok = geometric_dense_ok(checks)
    trajectory_energy = float(candidate.energy())
    obstacle_points = np.asarray(data["obstacle_points"], dtype=np.float64)
    clearance = dense_clearance_audit(candidate, obstacle_points, evaluator, config, args)
    route = route_geometry_audit(
        plan_dir.parent,
        reference,
        candidate,
        obstacle_points,
        surface_model,
        args,
    )
    time_scale = time_scaling_audit(candidate, config, args)
    scales = reference_motion_scales(reference, surface_model, args.tcp_link, args.samples)
    straight = path_straight_distance(surface_model, reference, args.tcp_link)
    duration = float(candidate.total_duration)
    raw_objectives = {
        "tcp_path_length_ratio": float(metrics["tcp_path_length_m"] / straight),
        "tcp_path_length_m": float(metrics["tcp_path_length_m"]),
        "joint_path_length_rad": float(metrics["joint_path_length_rad"]),
        "jerk_energy": trajectory_energy,
        "clearance_penalty": float(clearance["clearance_penalty"]),
        "duration_s": duration,
        "required_execution_time_s": float(time_scale["required_execution_time_s"]),
    }
    fixed_costs = fixed_scale_costs(
        raw_objectives,
        scales["reference_tcp_straight_distance_m"],
        scales["reference_joint_path_length_rad"],
        scales["reference_jerk_energy"],
    )
    item = {
        "name": plan_dir.name,
        "plan_dir": str(plan_dir),
        "status": summary.get("status"),
        "optimizer_type": summary.get("optimizer_type"),
        "optimizer_success": bool(get_nested(summary, ["candidate", "optimizer_success"], False)),
        "strict_execution_ok": strict_ok,
        "geometric_dense_ok_without_solver_flag": geometric_ok,
        "dense_reasons": dense.get("reasons", []),
        "dense_min_distance_m": dense.get("min_distance"),
        "nearest_link": dense.get("nearest_link"),
        "tcp_z_range_m": z_stats.get("z_range_m"),
        "trajectory_energy": trajectory_energy,
        "straight_tcp_distance_m": straight,
        "raw_objectives": raw_objectives,
        "fixed_scale_costs": fixed_costs,
        "selection_score": None,
        "near_best_time": None,
        **clearance,
        **time_scale,
        "route_geometry_audit": route,
        **metrics,
    }
    item["route_class"] = route["route_class"]
    item["hard_feasible_for_execution"] = bool(strict_ok)
    return item


def apply_selection(items: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    feasible = [item for item in items if item["hard_feasible_for_execution"]]
    for item in items:
        item["near_best_time"] = None
        item["selection_score"] = None
        item["selection_layer"] = "hard_rejected"
    if feasible:
        best_time = min(float(item["required_execution_time_s"]) for item in feasible)
        cutoff = best_time * float(args.time_near_optimal_ratio)
        for item in feasible:
            item["near_best_time"] = bool(float(item["required_execution_time_s"]) <= cutoff)
            item["selection_layer"] = "near_best_time" if item["near_best_time"] else "feasible_slower"
        near = [item for item in feasible if item["near_best_time"]]
        ordered_near = sorted(
            near,
            key=lambda item: (
                float(item["raw_objectives"]["tcp_path_length_m"]),
                float(item["raw_objectives"]["joint_path_length_rad"]),
                float(item["raw_objectives"]["jerk_energy"]),
                float(item["raw_objectives"]["clearance_penalty"]),
                float(item["required_execution_time_s"]),
            ),
        )
        for rank, item in enumerate(ordered_near):
            item["selection_score"] = float(rank)
    ranked = sorted(
        items,
        key=lambda item: (
            not item["hard_feasible_for_execution"],
            not bool(item["near_best_time"]) if item["near_best_time"] is not None else True,
            float("inf") if item["selection_score"] is None else float(item["selection_score"]),
            float(item["required_execution_time_s"]),
            float(item["raw_objectives"]["tcp_path_length_m"]),
            float(item["raw_objectives"]["joint_path_length_rad"]),
            float(item["raw_objectives"]["jerk_energy"]),
            float(item["raw_objectives"]["clearance_penalty"]),
        ),
    )
    return ranked


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# 6.5.2 Candidate Family Selection",
        "",
        "Selection rule:",
        "",
        "1. Candidates must first pass solver, dense safety, joint limits, continuity, and goal checks.",
        "2. Route class is determined from the robot swept surface relative to the inflated obstacle XY footprint.",
        "3. A route is a true overpass only if the swept surface enters the inflated obstacle footprint and clears the robust obstacle top.",
        "4. Feasible candidates are time-scaled using fixed velocity/acceleration limits.",
        "5. Candidates within the near-best execution-time set are ranked lexicographically by TCP path length, joint path length, jerk energy, and near-boundary clearance penalty.",
        "",
        "Route geometry:",
        "",
        f"- Inflated obstacle XY margin: `{payload['selection']['overpass_xy_inflation_m']:.3f} m`.",
        f"- Required vertical margin above robust obstacle top: `d_accept + {payload['selection']['vertical_uncertainty_m']:.3f} m`.",
        f"- Near-best time ratio: `{payload['selection']['time_near_optimal_ratio']:.3f}`.",
        "",
        "## Ranked Candidates",
        "",
        "| rank | candidate | route | feasible | near T | selected rank | min dist / m | T_req / s | L_TCP / m | L_TCP ratio | L_q / rad | jerk | J_clear | max z dev / m | p99 height / m | footprint pts | reasons |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for i, item in enumerate(payload["ranked_all"], 1):
        score_text = "NA" if item["selection_score"] is None else f"{float(item['selection_score']):.4f}"
        route = item.get("route_geometry_audit", {})
        lines.append(
            f"| {i} | `{item['name']}` | {item['route_class']} | "
            f"{str(item['hard_feasible_for_execution'])} | {str(item['near_best_time'])} | "
            f"{score_text} | "
            f"{float(item['dense_min_distance_m'] or 0.0):.4f} | "
            f"{float(item['required_execution_time_s']):.3f} | "
            f"{float(item['raw_objectives']['tcp_path_length_m']):.4f} | "
            f"{float(item['raw_objectives']['tcp_path_length_ratio']):.4f} | "
            f"{float(item['raw_objectives']['joint_path_length_rad']):.4f} | "
            f"{float(item['raw_objectives']['jerk_energy']):.4f} | "
            f"{float(item['raw_objectives']['clearance_penalty']):.4f} | "
            f"{float(item['max_tcp_z_deviation_m'] or 0.0):.4f} | "
            f"{float(route.get('robust_obstacle_height_p99_m') or 0.0):.4f} | "
            f"{int(route.get('robot_surface_points_inside_inflated_obstacle_xy') or 0)} | "
            f"{','.join(item['dense_reasons']) or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Selected execution candidate: `{payload['selected_execution_candidate'] or 'NONE'}`.",
            f"- Feasible candidate count: `{payload['feasible_candidate_count']}`.",
            f"- Near-best time candidate count: `{payload['near_best_time_candidate_count']}`.",
            "",
            "If no feasible candidate exists, the trial status is `NO_EXECUTABLE_CANDIDATE` and the robot must hold.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    trial_dir = args.trial_dir.resolve()
    config = _load(trial_dir / "config_used.yaml")
    surface_model = make_surface_model(config)
    evaluator, _, _ = make_evaluator_and_verifier(config, surface_model)
    if args.plan_dirs:
        plan_dirs = [p.resolve() for p in args.plan_dirs]
    else:
        plan_dirs = sorted(p.resolve() for p in trial_dir.glob("ccro_nubs_jointspace_plan*") if (p / "summary.json").exists())
    items = [summarize_plan(plan_dir, surface_model, evaluator, config, args) for plan_dir in plan_dirs]
    ranked_all = apply_selection(items, args)
    hard_feasible = [item for item in ranked_all if item["hard_feasible_for_execution"]]
    selected = next((item for item in ranked_all if item["selection_score"] is not None), None)
    payload = {
        "trial_dir": str(trial_dir),
        "clearance_m": args.clearance_m,
        "clearance_pref_m": args.clearance_pref_m,
        "selection": {
            "mode": "layered_fixed_scale_time_then_lexicographic",
            "overpass_z_deviation_m": args.overpass_z_deviation_m,
            "lateral_xy_deviation_m": args.lateral_xy_deviation_m,
            "overpass_xy_inflation_m": args.overpass_xy_inflation_m,
            "vertical_uncertainty_m": args.vertical_uncertainty_m,
            "time_near_optimal_ratio": args.time_near_optimal_ratio,
            "order": [
                "hard_feasibility",
                "route_geometry_audit",
                "required_execution_time_scaling",
                "near_best_time_filter",
                "lexicographic(L_TCP, L_q, J_jerk, J_clear)",
            ],
        },
        "ranked_all": ranked_all,
        "status": "EXECUTABLE_CANDIDATE_SELECTED" if selected else "NO_EXECUTABLE_CANDIDATE",
        "selected_execution_candidate": selected["name"] if selected else None,
        "hard_feasible_execution_candidate": selected["name"] if selected else None,
        "strict_execution_candidate": selected["name"] if selected else None,
        "feasible_candidate_count": len(hard_feasible),
        "near_best_time_candidate_count": len([item for item in hard_feasible if item["near_best_time"]]),
    }
    output_dir = (args.output or (trial_dir / "candidate_selection")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "candidate_selection_summary.json", payload)
    (output_dir / "candidate_selection_report.md").write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, default=DEFAULT_TRIAL_DIR)
    parser.add_argument("--plan-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tcp-link", default="gripper_base_link")
    parser.add_argument("--samples", type=int, default=241)
    parser.add_argument("--clearance-m", type=float, default=0.08)
    parser.add_argument("--clearance-pref-m", type=float, default=0.11)
    parser.add_argument(
        "--overpass-z-deviation-m",
        type=float,
        default=0.09,
        help="Large TCP z deviation threshold used only with footprint crossing to classify a true overpass.",
    )
    parser.add_argument(
        "--lateral-xy-deviation-m",
        type=float,
        default=0.06,
        help="TCP XY deviation threshold used to classify lateral/hybrid routes.",
    )
    parser.add_argument(
        "--overpass-xy-inflation-m",
        type=float,
        default=0.08,
        help="Inflation margin for the obstacle XY footprint used in route classification.",
    )
    parser.add_argument(
        "--vertical-uncertainty-m",
        type=float,
        default=0.02,
        help="Additional margin above d_accept for evaluating true overpass clearance.",
    )
    parser.add_argument("--route-samples", type=int, default=81)
    parser.add_argument("--route-density", choices=["coarse", "medium", "dense"], default="medium")
    parser.add_argument("--time-scale-dt", type=float, default=0.02)
    parser.add_argument(
        "--time-near-optimal-ratio",
        type=float,
        default=1.05,
        help="Candidates with required execution time within this ratio of the best time enter lexicographic ranking.",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
