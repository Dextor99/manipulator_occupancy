#!/usr/bin/env python3
"""Rank 6.5.2 static CCRO-NUBS candidate plans by safety and minimal change.

The execution rule remains conservative: only strict dense-accepted candidates
are executable.  Minimal-change metrics are used to rank candidates after the
hard safety gate, and to document rejected-but-informative alternatives.
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
from planning.nubs_trajectory import NUBSTrajectory6D  # noqa: E402
from plan_652_static_ccro_nubs_from_trial import trajectory_preference_metrics  # noqa: E402
from run_652_static_avoidance import make_surface_model  # noqa: E402


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


def score_candidate(item: dict[str, Any], args: argparse.Namespace) -> float:
    min_distance = float(item["dense_min_distance_m"] or 0.0)
    clearance_margin = max(min_distance - args.clearance_m, 0.0)
    return float(
        args.w_xyz * float(item["max_tcp_xyz_deviation_m"] or 0.0)
        + args.w_z * float(item["max_tcp_z_deviation_m"] or 0.0)
        + args.w_joint * float(item["joint_path_length_rad"] or 0.0)
        + args.w_clearance_excess * clearance_margin
    )


def summarize_plan(plan_dir: Path, surface_model: Any, args: argparse.Namespace) -> dict[str, Any]:
    summary = load_json(plan_dir / "summary.json")
    data_path = plan_dir / "ccro_nubs_trajectories.npz"
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
        **metrics,
    }
    item["route_class"] = classify_route(item.get("tcp_z_range_m"), item.get("max_tcp_z_deviation_m"))
    item["minimal_change_score"] = score_candidate(item, args)
    return item


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# 6.5.2 Candidate Family Selection",
        "",
        "Selection rule:",
        "",
        "1. Strict real execution may use only candidates with `strict_execution_ok=true`.",
        "2. Among strict candidates, rank by minimal task-space change after the safety gate.",
        "3. Rejected candidates may be used only for analysis/figures, not for robot execution.",
        "",
        "## Ranked Candidates",
        "",
        "| rank | candidate | route | strict | geom. dense | min dist / m | max TCP z dev / m | max TCP xyz dev / m | joint length / rad | score | reasons |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for i, item in enumerate(payload["ranked_all"], 1):
        lines.append(
            f"| {i} | `{item['name']}` | {item['route_class']} | "
            f"{str(item['strict_execution_ok'])} | {str(item['geometric_dense_ok_without_solver_flag'])} | "
            f"{float(item['dense_min_distance_m'] or 0.0):.4f} | "
            f"{float(item['max_tcp_z_deviation_m'] or 0.0):.4f} | "
            f"{float(item['max_tcp_xyz_deviation_m'] or 0.0):.4f} | "
            f"{float(item['joint_path_length_rad'] or 0.0):.4f} | "
            f"{float(item['minimal_change_score']):.4f} | "
            f"{','.join(item['dense_reasons']) or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Strict execution candidate: `{payload['strict_execution_candidate'] or 'NONE'}`.",
            f"- Minimal-change geometric candidate for analysis: `{payload['minimal_change_geometric_candidate'] or 'NONE'}`.",
            "",
            "If these are different, the result means the current optimizer can find a safer/executable route, "
            "but the lower-spatial-change route still needs either stricter convergence or an explicit hard-constrained planner before real execution.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    trial_dir = args.trial_dir.resolve()
    config = _load(trial_dir / "config_used.yaml")
    surface_model = make_surface_model(config)
    if args.plan_dirs:
        plan_dirs = [p.resolve() for p in args.plan_dirs]
    else:
        plan_dirs = sorted(p.resolve() for p in trial_dir.glob("ccro_nubs_jointspace_plan*") if (p / "summary.json").exists())
    items = [summarize_plan(plan_dir, surface_model, args) for plan_dir in plan_dirs]
    ranked_all = sorted(items, key=lambda item: (not item["geometric_dense_ok_without_solver_flag"], item["minimal_change_score"]))
    strict = [item for item in ranked_all if item["strict_execution_ok"]]
    geometric = [item for item in ranked_all if item["geometric_dense_ok_without_solver_flag"]]
    payload = {
        "trial_dir": str(trial_dir),
        "clearance_m": args.clearance_m,
        "weights": {
            "w_xyz": args.w_xyz,
            "w_z": args.w_z,
            "w_joint": args.w_joint,
            "w_clearance_excess": args.w_clearance_excess,
        },
        "ranked_all": ranked_all,
        "strict_execution_candidate": strict[0]["name"] if strict else None,
        "minimal_change_geometric_candidate": geometric[0]["name"] if geometric else None,
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
    parser.add_argument("--w-xyz", type=float, default=1.0)
    parser.add_argument("--w-z", type=float, default=3.0)
    parser.add_argument("--w-joint", type=float, default=0.05)
    parser.add_argument(
        "--w-clearance-excess",
        type=float,
        default=0.20,
        help="small penalty for excessive clearance after the required safety margin is already met",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
