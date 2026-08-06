#!/usr/bin/env python3
"""Offline URDF + numerical IK preview for a 6.5.2 planar candidate.

This script does not connect to RealSense or AUBO and sends no robot commands.
It reads a planning trial, solves a sequential numerical IK path for the
candidate TCP waypoints while preserving the start TCP orientation, and renders
the actual URDF joint-posture sequence implied by IK.

Use this before any real execution.  The older planar preview translates the
robot surface rigidly; this file instead estimates the joint-space motion that
would be required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from experiments.exp_ccro_stage2 import _load  # noqa: E402
from plan_652_planar_static_from_live import make_surface_model  # noqa: E402
from preview_652_planar_tabletop import (  # noqa: E402
    DEFAULT_CLEARANCE_LINKS,
    parse_links,
    sample_points,
    set_equal_axes,
    stack_surface,
)


DEFAULT_TRIAL = (
    ROOT
    / "results"
    / "new"
    / "6_5"
    / "6_5_2"
    / "planar_static_live"
    / "rs1_lateral_table_obstacle"
    / "trials"
    / "rs1_lateral_table_obstacle_r03"
)


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def joint_dict(surface_model, q: np.ndarray) -> dict[str, float]:
    return {name: float(q[i]) for i, name in enumerate(surface_model.joint_names)}


def load_trial(trial_dir: Path) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    summary = json.loads((trial_dir / "summary.json").read_text(encoding="utf-8"))
    waypoints = json.loads((trial_dir / "planar_execution_waypoints.json").read_text(encoding="utf-8"))
    obstacle_points = np.asarray(np.load(trial_dir / "obstacle_points.npz")["points"], dtype=np.float64)
    return summary, waypoints, obstacle_points


def fk_pose(surface_model, q: np.ndarray, tcp_link: str) -> tuple[np.ndarray, np.ndarray]:
    fk = surface_model.urdf.link_transforms(joint_dict(surface_model, q))
    T = fk[tcp_link]
    return T[:3, 3].copy(), T[:3, :3].copy()


def orientation_error(desired: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(desired @ actual.T).as_rotvec()


def pose_error(surface_model, q: np.ndarray, target_xyz: np.ndarray, target_rot: np.ndarray, tcp_link: str, rot_weight: float) -> np.ndarray:
    xyz, rot = fk_pose(surface_model, q, tcp_link)
    return np.r_[target_xyz - xyz, rot_weight * orientation_error(target_rot, rot)]


def numeric_jacobian(surface_model, q: np.ndarray, target_xyz: np.ndarray, target_rot: np.ndarray, tcp_link: str, rot_weight: float, eps: float) -> np.ndarray:
    base = pose_error(surface_model, q, target_xyz, target_rot, tcp_link, rot_weight)
    J = np.zeros((6, 6), dtype=np.float64)
    for j in range(6):
        qp = q.copy()
        qp[j] += eps
        J[:, j] = (pose_error(surface_model, qp, target_xyz, target_rot, tcp_link, rot_weight) - base) / eps
    return J


def solve_ik(surface_model, q_seed: np.ndarray, target_xyz: np.ndarray, target_rot: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    q = q_seed.astype(np.float64).copy()
    lower = np.deg2rad(np.asarray(args.joint_lower_deg.split(","), dtype=np.float64))
    upper = np.deg2rad(np.asarray(args.joint_upper_deg.split(","), dtype=np.float64))
    if lower.shape != (6,) or upper.shape != (6,):
        raise ValueError("joint limit strings must contain 6 comma-separated values")
    q = np.clip(q, lower, upper)
    best_q = q.copy()
    best_err = math.inf
    for it in range(args.ik_max_iter):
        err_vec = pose_error(surface_model, q, target_xyz, target_rot, args.tcp_link, args.rot_weight)
        pos_err = float(np.linalg.norm(err_vec[:3]))
        rot_err = float(np.linalg.norm(err_vec[3:]) / max(args.rot_weight, 1.0e-9))
        score = pos_err + args.rot_score_weight * rot_err
        if score < best_err:
            best_err = score
            best_q = q.copy()
        if pos_err <= args.ik_pos_tol_m and rot_err <= args.ik_rot_tol_rad:
            return {
                "success": True,
                "q": q,
                "iterations": it,
                "position_error_m": pos_err,
                "rotation_error_rad": rot_err,
            }
        J = numeric_jacobian(surface_model, q, target_xyz, target_rot, args.tcp_link, args.rot_weight, args.fd_eps)
        A = J @ J.T + (args.damping**2) * np.eye(6)
        dq = J.T @ np.linalg.solve(A, err_vec)
        step_norm = float(np.linalg.norm(dq))
        if step_norm > args.max_step_rad:
            dq *= args.max_step_rad / step_norm
        q = np.clip(q + dq, lower, upper)
    err_vec = pose_error(surface_model, best_q, target_xyz, target_rot, args.tcp_link, args.rot_weight)
    return {
        "success": False,
        "q": best_q,
        "iterations": args.ik_max_iter,
        "position_error_m": float(np.linalg.norm(err_vec[:3])),
        "rotation_error_rad": float(np.linalg.norm(err_vec[3:]) / max(args.rot_weight, 1.0e-9)),
    }


def bezier_polyline(points: list[np.ndarray], samples_per_segment: int) -> np.ndarray:
    p0, p1, p2 = points
    u = np.linspace(0.0, 1.0, samples_per_segment)
    return (1 - u)[:, None] ** 2 * p0 + 2 * (1 - u)[:, None] * u[:, None] * p1 + u[:, None] ** 2 * p2


def candidate_tcp_path(waypoints: dict[str, Any], samples: int) -> np.ndarray:
    stored = waypoints.get("samples", {}).get("candidate")
    if stored:
        path = np.asarray(stored, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
            raise RuntimeError("stored candidate samples are invalid")
        if samples > 0 and samples < len(path):
            indices = np.linspace(0, len(path) - 1, samples).round().astype(int)
            return path[indices]
        return path
    candidate = waypoints["candidate_xyz"]
    return bezier_polyline(
        [
            np.asarray(candidate["P0_start"], dtype=np.float64),
            np.asarray(candidate["P1_via"], dtype=np.float64),
            np.asarray(candidate["P2_goal"], dtype=np.float64),
        ],
        samples,
    )


def clearance_stats(surface_model, q_path: np.ndarray, obstacle_points: np.ndarray, table_z: float, links: tuple[str, ...], density: str) -> dict[str, Any]:
    tree = cKDTree(obstacle_points)
    rows: list[dict[str, Any]] = []
    min_obs = math.inf
    min_table = math.inf
    min_obs_idx = -1
    min_table_idx = -1
    link_set = set(links)
    for i, q in enumerate(q_path):
        by_link = surface_model.surface_by_link(q, density=density, links=link_set)
        surf = stack_surface(by_link)
        if len(surf) == 0:
            continue
        dists, _ = tree.query(surf, k=1)
        obs_clear = float(np.min(dists))
        table_clear = float(np.min(surf[:, 2]) - table_z)
        if obs_clear < min_obs:
            min_obs = obs_clear
            min_obs_idx = i
        if table_clear < min_table:
            min_table = table_clear
            min_table_idx = i
        rows.append(
            {
                "index": i,
                "obstacle_clearance_m": obs_clear,
                "table_clearance_m": table_clear,
                **{f"q{j+1}_rad": float(q[j]) for j in range(6)},
            }
        )
    return {
        "rows": rows,
        "min_obstacle_clearance_m": min_obs,
        "min_table_clearance_m": min_table,
        "min_obstacle_index": min_obs_idx,
        "min_table_index": min_table_idx,
    }


def plot_ik_joint_curves(path: Path, rows: list[dict[str, Any]], limits: tuple[np.ndarray, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = [r["index"] for r in rows]
    fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharex=True, dpi=180)
    lower, upper = limits
    for j, ax in enumerate(axes.ravel()):
        ax.plot(idx, [math.degrees(r[f"q{j+1}_rad"]) for r in rows], color="#1f77b4")
        ax.axhline(math.degrees(lower[j]), color="#d62728", linestyle="--", linewidth=0.8)
        ax.axhline(math.degrees(upper[j]), color="#d62728", linestyle="--", linewidth=0.8)
        ax.set_ylabel(f"q{j+1} / deg")
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("path sample")
    axes[-1, 1].set_xlabel("path sample")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_clearance(path: Path, rows: list[dict[str, Any]], obstacle_threshold: float, table_threshold: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = [r["index"] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.0), sharex=True, dpi=180)
    axes[0].plot(idx, [r["obstacle_clearance_m"] for r in rows], color="#1f77b4")
    axes[0].axhline(obstacle_threshold, color="#d62728", linestyle="--", linewidth=1.1)
    axes[0].set_ylabel("obstacle clearance / m")
    axes[1].plot(idx, [r["table_clearance_m"] for r in rows], color="#2ca02c")
    axes[1].axhline(table_threshold, color="#d62728", linestyle="--", linewidth=1.1)
    axes[1].set_ylabel("table clearance / m")
    axes[1].set_xlabel("path sample")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pose_sequence(path: Path, surface_model, q_path: np.ndarray, obstacle_points: np.ndarray, table_z: float, density: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(20260652)
    obs = sample_points(obstacle_points, 2500, rng)
    sample_indices = np.linspace(0, len(q_path) - 1, min(7, len(q_path))).round().astype(int)
    fig = plt.figure(figsize=(10.5, 7.4), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2], s=4, c="#d62728", alpha=0.38, label="observed obstacle")
    plotted = [obs]
    colors = ["#d9d9d9", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]
    for ci, idx in enumerate(sample_indices):
        by_link = surface_model.surface_by_link(q_path[idx], density=density)
        pts = sample_points(stack_surface(by_link), 1800, rng)
        plotted.append(pts)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.6, c=colors[ci], alpha=0.20 + 0.08 * ci, label=f"s={idx}")
    xlim = [min(np.min(a[:, 0]) for a in plotted), max(np.max(a[:, 0]) for a in plotted)]
    ylim = [min(np.min(a[:, 1]) for a in plotted), max(np.max(a[:, 1]) for a in plotted)]
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 2), np.linspace(ylim[0], ylim[1], 2))
    zz = np.full_like(xx, table_z)
    ax.plot_surface(xx, yy, zz, color="#c7c7c7", alpha=0.18, linewidth=0)
    set_equal_axes(ax, plotted)
    ax.set_title("URDF IK preview: actual joint postures along candidate")
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.view_init(elev=24, azim=-52)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    trial_dir = args.trial_dir.resolve()
    output = (args.output or (trial_dir / "ik_urdf_preview")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = trial_dir / "config_used.yaml"
    config = _load(config_path)
    surface_model = make_surface_model(config)
    summary, waypoints, obstacle_points = load_trial(trial_dir)
    q0 = np.asarray(summary["q_start_mean_rad"], dtype=np.float64)
    table_z = float(summary["table_z_m"])
    _, target_rot = fk_pose(surface_model, q0, args.tcp_link)
    candidate = waypoints["candidate_xyz"]
    tcp_path = candidate_tcp_path(waypoints, args.samples)

    ik_results = []
    q_path = []
    q_seed = q0.copy()
    for i, target in enumerate(tcp_path):
        result = solve_ik(surface_model, q_seed, target, target_rot, args)
        result["index"] = i
        result["target_xyz"] = target.tolist()
        ik_results.append(result)
        q_seed = result["q"]
        q_path.append(q_seed.copy())
    q_path_arr = np.vstack(q_path)

    links = parse_links(args.clearance_links)
    clear = clearance_stats(surface_model, q_path_arr, obstacle_points, table_z, links, args.density)
    lower = np.deg2rad(np.asarray(args.joint_lower_deg.split(","), dtype=np.float64))
    upper = np.deg2rad(np.asarray(args.joint_upper_deg.split(","), dtype=np.float64))

    rows = []
    for ik, clear_row in zip(ik_results, clear["rows"]):
        rows.append(
            {
                "index": ik["index"],
                "ik_success": bool(ik["success"]),
                "ik_position_error_m": ik["position_error_m"],
                "ik_rotation_error_rad": ik["rotation_error_rad"],
                "target_x": ik["target_xyz"][0],
                "target_y": ik["target_xyz"][1],
                "target_z": ik["target_xyz"][2],
                "obstacle_clearance_m": clear_row["obstacle_clearance_m"],
                "table_clearance_m": clear_row["table_clearance_m"],
                **{f"q{j+1}_rad": float(ik["q"][j]) for j in range(6)},
                **{f"q{j+1}_deg": float(math.degrees(ik["q"][j])) for j in range(6)},
            }
        )
    fields = list(rows[0].keys()) if rows else []
    write_csv(output / "ik_path_samples.csv", rows, fields)
    plot_ik_joint_curves(output / "ik_joint_curves.png", rows, (lower, upper))
    plot_clearance(output / "ik_clearance_curves.png", rows, args.clearance_m, args.table_clearance_threshold_m)
    plot_pose_sequence(output / "ik_pose_sequence.png", surface_model, q_path_arr, obstacle_points, table_z, args.density)

    pos_errors = [r["ik_position_error_m"] for r in rows]
    rot_errors = [r["ik_rotation_error_rad"] for r in rows]
    dq = np.linalg.norm(np.diff(q_path_arr, axis=0), axis=1) if len(q_path_arr) > 1 else np.asarray([0.0])
    joint_limit_ok = bool(np.all(q_path_arr >= lower[None, :] - args.limit_tolerance_rad) and np.all(q_path_arr <= upper[None, :] + args.limit_tolerance_rad))
    accepted = bool(
        all(r["ik_success"] for r in rows)
        and max(pos_errors) <= args.accept_pos_error_m
        and max(rot_errors) <= args.accept_rot_error_rad
        and float(np.max(dq)) <= args.max_joint_step_accept_rad
        and clear["min_obstacle_clearance_m"] >= args.clearance_m
        and clear["min_table_clearance_m"] >= args.table_clearance_threshold_m
        and joint_limit_ok
    )
    payload = {
        "robot_commanded": False,
        "trial_dir": str(trial_dir),
        "output_dir": str(output),
        "accepted_for_real_execution": accepted,
        "reason": "ok" if accepted else "IK/URDF preview gate failed",
        "samples": len(rows),
        "ik_success_count": int(sum(bool(r["ik_success"]) for r in rows)),
        "max_ik_position_error_m": float(max(pos_errors)),
        "max_ik_rotation_error_rad": float(max(rot_errors)),
        "max_joint_step_rad": float(np.max(dq)),
        "max_joint_step_deg": float(math.degrees(np.max(dq))),
        "joint_limit_ok": joint_limit_ok,
        "min_obstacle_clearance_m": clear["min_obstacle_clearance_m"],
        "min_table_clearance_m": clear["min_table_clearance_m"],
        "min_obstacle_index": clear["min_obstacle_index"],
        "min_table_index": clear["min_table_index"],
        "candidate_waypoints": candidate,
        "tcp_path_source": "stored candidate samples" if waypoints.get("samples", {}).get("candidate") else "reconstructed quadratic Bezier",
        "q_start_rad": q0.tolist(),
        "q_path_start_deg": np.rad2deg(q_path_arr[0]).tolist(),
        "q_path_end_deg": np.rad2deg(q_path_arr[-1]).tolist(),
        "figures": [
            "ik_pose_sequence.png",
            "ik_joint_curves.png",
            "ik_clearance_curves.png",
        ],
    }
    write_json(output / "ik_preview_summary.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, default=DEFAULT_TRIAL)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tcp-link", default="gripper_base_link")
    parser.add_argument("--samples", type=int, default=41)
    parser.add_argument("--density", choices=["coarse", "medium", "dense"], default="coarse")
    parser.add_argument("--clearance-links", default=",".join(DEFAULT_CLEARANCE_LINKS))
    parser.add_argument("--clearance-m", type=float, default=0.08)
    parser.add_argument("--table-clearance-threshold-m", type=float, default=0.06)
    parser.add_argument("--joint-lower-deg", default="-360,-175,-175,-175,-175,-360")
    parser.add_argument("--joint-upper-deg", default="360,175,175,175,175,360")
    parser.add_argument("--limit-tolerance-rad", type=float, default=1.0e-6)
    parser.add_argument("--ik-max-iter", type=int, default=160)
    parser.add_argument("--ik-pos-tol-m", type=float, default=0.003)
    parser.add_argument("--ik-rot-tol-rad", type=float, default=0.03)
    parser.add_argument("--accept-pos-error-m", type=float, default=0.010)
    parser.add_argument("--accept-rot-error-rad", type=float, default=0.08)
    parser.add_argument("--rot-weight", type=float, default=0.18)
    parser.add_argument("--rot-score-weight", type=float, default=0.10)
    parser.add_argument("--damping", type=float, default=0.045)
    parser.add_argument("--fd-eps", type=float, default=1.0e-5)
    parser.add_argument("--max-step-rad", type=float, default=0.035)
    parser.add_argument("--max-joint-step-accept-rad", type=float, default=0.18)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
