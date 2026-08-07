#!/usr/bin/env python3
"""Rank 6.5.2 static CCRO-NUBS candidates by general avoidance cost.

The execution rule remains conservative: only candidates that pass solver,
dense full-body safety, joint limits, continuity, and goal checks are
executable.  Among feasible candidates this script performs Pareto filtering and
then selects the lowest normalized cost over 3D TCP path length, joint motion,
NUBS jerk/smooth energy, near-boundary clearance penalty, and duration.
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


def classify_route(z_range_m: float | None, max_z_dev_m: float | None) -> str:
    z = max(float(z_range_m or 0.0), float(max_z_dev_m or 0.0))
    if z <= 0.035:
        return "planar/lateral"
    if z <= 0.090:
        return "hybrid"
    return "overpass"


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


def tabletop_overpass_audit(
    trial_dir: Path,
    reference: NUBSTrajectory6D,
    candidate: NUBSTrajectory6D,
    obstacle_points: np.ndarray,
    surface_model: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    bbox_min = np.min(obstacle_points, axis=0)
    bbox_max = np.max(obstacle_points, axis=0)
    try:
        trial_summary = load_json(trial_dir / "summary.json")
    except FileNotFoundError:
        trial_summary = {}
    table_z = float(trial_summary.get("table_z_m", np.percentile(obstacle_points[:, 2], 2)))
    obstacle_height = float(bbox_max[2] - table_z)
    ref_tcp = tcp_xyz_samples(surface_model, reference, args.tcp_link, args.samples)
    cand_tcp = tcp_xyz_samples(surface_model, candidate, args.tcp_link, args.samples)
    ref_z = float(ref_tcp[0, 2])
    max_z_dev = float(np.max(np.abs(cand_tcp[:, 2] - ref_tcp[:, 2])))
    xy_inflation = float(args.overpass_xy_inflation_m)
    inside_xy = (
        (cand_tcp[:, 0] >= bbox_min[0] - xy_inflation)
        & (cand_tcp[:, 0] <= bbox_max[0] + xy_inflation)
        & (cand_tcp[:, 1] >= bbox_min[1] - xy_inflation)
        & (cand_tcp[:, 1] <= bbox_max[1] + xy_inflation)
    )
    high_obstacle = bool(
        args.tabletop_overpass_policy != "off"
        and obstacle_height >= args.high_obstacle_height_m
    )
    vertical_overpass = bool(max_z_dev > args.overpass_z_deviation_m)
    crosses_inflated_footprint = bool(np.any(inside_xy))
    rejected = bool(
        args.tabletop_overpass_policy == "reject_high_obstacle_overpass"
        and high_obstacle
        and vertical_overpass
    )
    reasons: list[str] = []
    if rejected:
        reasons.append(
            "high_tabletop_obstacle_vertical_overpass_rejected"
        )
    return {
        "policy": args.tabletop_overpass_policy,
        "table_z_m": table_z,
        "obstacle_bbox_min": bbox_min.tolist(),
        "obstacle_bbox_max": bbox_max.tolist(),
        "obstacle_height_from_table_m": obstacle_height,
        "reference_tcp_start_z_m": ref_z,
        "obstacle_top_minus_reference_tcp_start_z_m": float(bbox_max[2] - ref_z),
        "candidate_max_tcp_z_deviation_m": max_z_dev,
        "high_obstacle_height_threshold_m": args.high_obstacle_height_m,
        "overpass_z_deviation_threshold_m": args.overpass_z_deviation_m,
        "xy_inflation_m": xy_inflation,
        "tcp_samples_inside_inflated_obstacle_xy": int(np.count_nonzero(inside_xy)),
        "crosses_inflated_obstacle_xy": crosses_inflated_footprint,
        "high_tabletop_obstacle": high_obstacle,
        "vertical_overpass": vertical_overpass,
        "tabletop_overpass_ok": not rejected,
        "tabletop_overpass_reasons": reasons,
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
    tabletop = tabletop_overpass_audit(
        plan_dir.parent,
        reference,
        candidate,
        obstacle_points,
        surface_model,
        args,
    )
    straight = path_straight_distance(surface_model, reference, args.tcp_link)
    duration = float(candidate.total_duration)
    raw_objectives = {
        "tcp_path_length_ratio": float(metrics["tcp_path_length_m"] / straight),
        "joint_path_length_rad": float(metrics["joint_path_length_rad"]),
        "jerk_energy": trajectory_energy,
        "clearance_penalty": float(clearance["clearance_penalty"]),
        "duration_s": duration,
    }
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
        "normalized_objectives": {},
        "selection_score": None,
        "pareto_dominated": None,
        **clearance,
        "tabletop_overpass_audit": tabletop,
        **metrics,
    }
    item["route_class"] = classify_route(item.get("tcp_z_range_m"), item.get("max_tcp_z_deviation_m"))
    item["hard_feasible_for_execution"] = bool(strict_ok and tabletop["tabletop_overpass_ok"])
    if not tabletop["tabletop_overpass_ok"]:
        item["dense_reasons"] = list(item["dense_reasons"]) + list(tabletop["tabletop_overpass_reasons"])
    return item


def apply_selection(items: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    objective_keys = ["tcp_path_length_ratio", "joint_path_length_rad", "jerk_energy", "clearance_penalty", "duration_s"]
    feasible = [item for item in items if item["hard_feasible_for_execution"]]
    for item in items:
        item["pareto_dominated"] = None
    for item in feasible:
        item["pareto_dominated"] = any(
            dominates(other, item, objective_keys)
            for other in feasible
            if other is not item
        )
    nondominated = [item for item in feasible if not item["pareto_dominated"]]
    for key in objective_keys:
        normalize_values(nondominated, key)
    weights = {
        "tcp_path_length_ratio": args.w_tcp,
        "joint_path_length_rad": args.w_joint,
        "jerk_energy": args.w_jerk,
        "clearance_penalty": args.w_clearance,
        "duration_s": args.w_time,
    }
    for item in items:
        item["selection_score"] = None
    for item in nondominated:
        item["selection_score"] = float(
            sum(weights[key] * item["normalized_objectives"][key] for key in objective_keys)
        )
    ranked = sorted(
        items,
        key=lambda item: (
            not item["hard_feasible_for_execution"],
            bool(item["pareto_dominated"]) if item["pareto_dominated"] is not None else True,
            float("inf") if item["selection_score"] is None else float(item["selection_score"]),
            float(item["raw_objectives"]["tcp_path_length_ratio"]),
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
        "2. For tabletop high obstacles, vertical overpass routes are rejected as task-inappropriate before scoring.",
        "3. Remaining feasible candidates are Pareto-filtered over L_TCP, L_q, jerk energy, near-boundary clearance penalty, and duration.",
        "4. Only Pareto non-dominated candidates receive a normalized weighted score.",
        "5. Rejected or dominated candidates may be used for analysis/figures, not as the selected execution candidate.",
        "",
        "Tabletop overpass policy:",
        "",
        f"- Policy: `{payload['selection']['tabletop_overpass_policy']}`.",
        f"- High obstacle threshold: `{payload['selection']['high_obstacle_height_m']:.3f} m` above the table.",
        f"- Vertical overpass threshold: `{payload['selection']['overpass_z_deviation_m']:.3f} m` TCP z deviation.",
        "",
        "## Ranked Candidates",
        "",
        "| rank | candidate | route | feasible | dominated | score | min dist / m | L_TCP ratio | L_q / rad | jerk | J_clear | T / s | max z dev / m | obs height / m | tabletop ok | reasons |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for i, item in enumerate(payload["ranked_all"], 1):
        score_text = "NA" if item["selection_score"] is None else f"{float(item['selection_score']):.4f}"
        tabletop = item.get("tabletop_overpass_audit", {})
        lines.append(
            f"| {i} | `{item['name']}` | {item['route_class']} | "
            f"{str(item['hard_feasible_for_execution'])} | {str(item['pareto_dominated'])} | "
            f"{score_text} | "
            f"{float(item['dense_min_distance_m'] or 0.0):.4f} | "
            f"{float(item['raw_objectives']['tcp_path_length_ratio']):.4f} | "
            f"{float(item['raw_objectives']['joint_path_length_rad']):.4f} | "
            f"{float(item['raw_objectives']['jerk_energy']):.4f} | "
            f"{float(item['raw_objectives']['clearance_penalty']):.4f} | "
            f"{float(item['raw_objectives']['duration_s']):.2f} | "
            f"{float(item['max_tcp_z_deviation_m'] or 0.0):.4f} | "
            f"{float(tabletop.get('obstacle_height_from_table_m') or 0.0):.4f} | "
            f"{str(tabletop.get('tabletop_overpass_ok', True))} | "
            f"{','.join(item['dense_reasons']) or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Selected execution candidate: `{payload['selected_execution_candidate'] or 'NONE'}`.",
            f"- Feasible candidate count: `{payload['feasible_candidate_count']}`.",
            f"- Pareto non-dominated count: `{payload['pareto_nondominated_count']}`.",
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
    nondominated = [item for item in hard_feasible if not item["pareto_dominated"]]
    selected = next((item for item in ranked_all if item["selection_score"] is not None), None)
    payload = {
        "trial_dir": str(trial_dir),
        "clearance_m": args.clearance_m,
        "clearance_pref_m": args.clearance_pref_m,
        "weights": {
            "w_tcp": args.w_tcp,
            "w_joint": args.w_joint,
            "w_jerk": args.w_jerk,
            "w_clearance": args.w_clearance,
            "w_time": args.w_time,
        },
        "selection": {
            "mode": "pareto_then_normalized_weighted_score",
            "tabletop_overpass_policy": args.tabletop_overpass_policy,
            "high_obstacle_height_m": args.high_obstacle_height_m,
            "overpass_z_deviation_m": args.overpass_z_deviation_m,
            "overpass_xy_inflation_m": args.overpass_xy_inflation_m,
            "order": [
                "hard_feasibility",
                "tabletop_high_obstacle_overpass_gate",
                "pareto_filter",
                "J_select = w_p*L_TCP_bar + w_q*L_q_bar + w_s*J_jerk_bar + w_c*J_clear_bar + w_T*T_bar",
            ],
        },
        "ranked_all": ranked_all,
        "status": "EXECUTABLE_CANDIDATE_SELECTED" if selected else "NO_EXECUTABLE_CANDIDATE",
        "selected_execution_candidate": selected["name"] if selected else None,
        "hard_feasible_execution_candidate": selected["name"] if selected else None,
        "strict_execution_candidate": selected["name"] if selected else None,
        "feasible_candidate_count": len(hard_feasible),
        "pareto_nondominated_count": len(nondominated),
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
    parser.add_argument("--w-tcp", type=float, default=0.35)
    parser.add_argument("--w-joint", type=float, default=0.25)
    parser.add_argument("--w-jerk", type=float, default=0.25)
    parser.add_argument("--w-clearance", type=float, default=0.15)
    parser.add_argument("--w-time", type=float, default=0.0)
    parser.add_argument(
        "--tabletop-overpass-policy",
        choices=["reject_high_obstacle_overpass", "off"],
        default="reject_high_obstacle_overpass",
        help="Reject large vertical overpass routes when the observed tabletop obstacle is tall.",
    )
    parser.add_argument(
        "--high-obstacle-height-m",
        type=float,
        default=0.22,
        help="Obstacle height above the detected table beyond which vertical overpass routes are task-inappropriate.",
    )
    parser.add_argument(
        "--overpass-z-deviation-m",
        type=float,
        default=0.09,
        help="TCP z deviation relative to the reference that classifies a candidate as a vertical overpass.",
    )
    parser.add_argument(
        "--overpass-xy-inflation-m",
        type=float,
        default=0.08,
        help="Inflation used only for reporting whether TCP XY enters the obstacle footprint.",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
