#!/usr/bin/env python3
"""Build a four-way static clearance ledger from one saved rolling shadow run."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

trial = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")
from perception.geometry_fit import fit_obb  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument(
        "--reference-feedback-csv",
        type=Path,
        default=ROOT / "results/new/6_5/6_5_3/reference_xp10_line/reference_feedback.csv",
    )
    parser.add_argument("--segment", type=int, default=3)
    parser.add_argument("--attempt", type=int, default=None)
    parser.add_argument("--fast-inflation-m", type=float, default=0.05)
    parser.add_argument("--obb-inflation-m", type=float, default=0.05)
    parser.add_argument("--reference-step-s", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def read_candidate_tail(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty candidate: {path}")
    return np.asarray([float(rows[-1][f"q{i}_rad"]) for i in range(1, 7)])


def point_obb_signed_distance_and_nearest(
    points: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    half_lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    local = (points - center[None, :]) @ rotation
    clipped = np.clip(local, -half_lengths[None, :], half_lengths[None, :])
    delta = np.abs(local) - half_lengths[None, :]
    outside = np.linalg.norm(np.maximum(delta, 0.0), axis=1)
    inside = np.all(delta <= 0.0, axis=1)
    signed = outside.copy()
    if np.any(inside):
        margins = half_lengths[None, :] - np.abs(local[inside])
        axes = np.argmin(margins, axis=1)
        inside_indices = np.flatnonzero(inside)
        for row, axis in zip(inside_indices, axes):
            sign = 1.0 if local[row, axis] >= 0.0 else -1.0
            clipped[row, axis] = sign * half_lengths[axis]
            signed[row] = -margins[np.where(inside_indices == row)[0][0], axis]
    return signed, clipped @ rotation.T + center[None, :]


def state_ledger(
    model: Any,
    q: np.ndarray,
    *,
    raw_points: np.ndarray,
    sphere_centers: np.ndarray,
    sphere_base_radii: np.ndarray,
    fast_inflation_m: float,
    obb_center: np.ndarray,
    obb_rotation: np.ndarray,
    obb_half_lengths: np.ndarray,
) -> dict[str, Any]:
    raw_tree = cKDTree(raw_points)
    best: dict[str, dict[str, Any]] = {}
    for link, robot_points in model.surface_by_link(q, density="medium").items():
        robot_points = np.asarray(robot_points, dtype=np.float64)
        raw_d, raw_i = raw_tree.query(robot_points, k=1)
        raw_row = int(np.argmin(raw_d))
        candidates = {
            "raw_cloud": (
                float(raw_d[raw_row]),
                raw_row,
                np.asarray(raw_points[int(raw_i[raw_row])]),
            )
        }
        vectors = robot_points[:, None, :] - sphere_centers[None, :, :]
        norms = np.linalg.norm(vectors, axis=2)
        for name, radii in (
            ("base_multisphere", sphere_base_radii),
            ("fast_multisphere", sphere_base_radii + fast_inflation_m),
        ):
            distances = norms - radii[None, :]
            flat = int(np.argmin(distances))
            robot_i, sphere_i = np.unravel_index(flat, distances.shape)
            norm = float(norms[robot_i, sphere_i])
            direction = (
                vectors[robot_i, sphere_i] / norm
                if norm > 1.0e-12
                else np.asarray([1.0, 0.0, 0.0])
            )
            obstacle_point = sphere_centers[sphere_i] + radii[sphere_i] * direction
            candidates[name] = (
                float(distances[robot_i, sphere_i]),
                int(robot_i),
                obstacle_point,
            )
        obb_d, obb_nearest = point_obb_signed_distance_and_nearest(
            robot_points,
            obb_center,
            obb_rotation,
            obb_half_lengths,
        )
        obb_i = int(np.argmin(obb_d))
        candidates["inflated_obb"] = (
            float(obb_d[obb_i]),
            obb_i,
            obb_nearest[obb_i],
        )
        for name, (distance, robot_i, obstacle_point) in candidates.items():
            if name not in best or distance < best[name]["distance_m"]:
                best[name] = {
                    "distance_m": distance,
                    "nearest_link": link,
                    "nearest_robot_point_m": robot_points[robot_i].tolist(),
                    "nearest_obstacle_point_m": np.asarray(obstacle_point).tolist(),
                }
    return best


def select_attempt(summary: dict[str, Any], segment: int, requested: int | None) -> int:
    item = next(row for row in summary["segments"] if int(row["segment"]) == segment)
    if requested is not None:
        return requested
    if item.get("authorized_attempt") is not None:
        return int(item["authorized_attempt"])
    return int(item["attempts"][-1]["attempt"])


def make_plot(
    path: Path,
    model: Any,
    states: dict[str, np.ndarray],
    raw_points: np.ndarray,
    ledger: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 11))
    colors = {
        "raw_cloud": "#222222",
        "base_multisphere": "#2ca02c",
        "fast_multisphere": "#d62728",
        "inflated_obb": "#1f77b4",
    }
    for panel, (name, q) in enumerate(states.items(), 1):
        ax = fig.add_subplot(2, 2, panel, projection="3d")
        surfaces = model.surface_by_link(q, density="medium")
        all_robot = np.vstack(list(surfaces.values()))
        ax.scatter(*all_robot.T, s=1, c="#aaaaaa", alpha=0.25)
        ax.scatter(*raw_points.T, s=8, c="#ff9900", alpha=0.7)
        for geometry, row in ledger[name].items():
            a = np.asarray(row["nearest_robot_point_m"])
            b = np.asarray(row["nearest_obstacle_point_m"])
            ax.plot(*np.vstack([a, b]).T, c=colors[geometry], lw=2, label=f"{geometry}: {row['distance_m']:.3f} m")
            ax.scatter(*a, s=24, c=colors[geometry])
            ax.scatter(*b, s=24, c=colors[geometry], marker="x")
        values = np.vstack([all_robot, raw_points])
        center = values.mean(axis=0)
        span = max(float(np.ptp(values, axis=0).max()), 0.4)
        for setter, value in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
            setter(value - span / 2, value + span / 2)
        ax.set_title(name)
        ax.legend(fontsize=7)
    fig.suptitle("Static r07 distance ledger (same RGB-D cluster and robot states)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    source = args.source_run.resolve()
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    attempt = select_attempt(summary, args.segment, args.attempt)
    attempt_dir = source / f"segment_{args.segment:02d}" / f"attempt_{attempt:02d}"
    points_path = attempt_dir / "fresh_plan_points.npy"
    if not points_path.exists():
        raise FileNotFoundError(points_path)
    raw_points = np.asarray(np.load(points_path), dtype=np.float64)
    runtime_args = trial.build_parser().parse_args(["--scene", "D2", "--mode", "shadow"])
    config = trial.load_stage4_config(runtime_args.stage4_config)
    model = trial.load_stage4_surface_model(config)
    reference = trial.RecordedReference.load(args.reference_feedback_csv.resolve())

    multi = trial.fit_pca_multisphere(
        raw_points,
        fit_margin_m=runtime_args.multisphere_fit_margin_m,
        max_components=runtime_args.multisphere_max_components,
    )
    sphere_centers = np.asarray(multi["component_centers"], dtype=np.float64)
    sphere_base_radii = np.asarray(multi["component_base_radii"], dtype=np.float64)
    obb = fit_obb(raw_points)
    obb_center = np.asarray(obb.center, dtype=np.float64)
    obb_rotation = np.asarray(obb.rotation, dtype=np.float64)
    obb_half_lengths = (
        np.asarray(obb.extents["half_lengths"], dtype=np.float64) + args.obb_inflation_m
    )
    geometry = {
        "raw_points": raw_points,
        "sphere_centers": sphere_centers,
        "sphere_base_radii": sphere_base_radii,
        "fast_inflation_m": float(args.fast_inflation_m),
        "obb_center": obb_center,
        "obb_rotation": obb_rotation,
        "obb_half_lengths": obb_half_lengths,
    }

    segments = summary["segments"]
    states = {
        "q0_stop": np.asarray(segments[0]["q_virtual_start"], dtype=np.float64),
        "q1_segment1_tail": np.asarray(segments[0]["q_virtual_end"], dtype=np.float64),
        "q2_segment2_tail": np.asarray(segments[1]["q_virtual_end"], dtype=np.float64),
        "q3_segment3_candidate": read_candidate_tail(
            attempt_dir / "candidate/fast_ccro_nubs_candidate.csv"
        ),
    }
    state_rows = {name: state_ledger(model, q, **geometry) for name, q in states.items()}

    reference_start = float(segments[0]["reference_plan_start_time_s"])
    absolute_times = np.arange(
        reference_start,
        float(reference.times[-1]) + 0.5 * args.reference_step_s,
        args.reference_step_s,
    )
    absolute_times = np.unique(np.r_[absolute_times, float(reference.times[-1])])
    reference_rows = []
    for absolute in absolute_times:
        row = state_ledger(model, reference.state_at(float(absolute))[0], **geometry)
        reference_rows.append(
            {"tau_s": float(absolute - reference_start), "absolute_time_s": float(absolute), **row}
        )
    reference_summary = {}
    for name in ("raw_cloud", "base_multisphere", "fast_multisphere", "inflated_obb"):
        item = min(reference_rows, key=lambda row: row[name]["distance_m"])
        reference_summary[name] = {
            **item[name],
            "tau_s": item["tau_s"],
            "absolute_time_s": item["absolute_time_s"],
            "goal_distance_m": reference_rows[-1][name]["distance_m"],
        }

    output = args.output.resolve() if args.output else source / "static_distance_ledger.json"
    plot_path = output.with_suffix(".png")
    payload = {
        "status": "STATIC_DISTANCE_LEDGER_COMPLETE",
        "source_run": str(source),
        "source_points": str(points_path),
        "segment": int(args.segment),
        "attempt": int(attempt),
        "robot_commanded": False,
        "authoritative_for_execution": False,
        "point_count": int(len(raw_points)),
        "fast_inflation_m": float(args.fast_inflation_m),
        "obb_inflation_m": float(args.obb_inflation_m),
        "multisphere_coverage_ratio": float(multi["coverage_ratio"]),
        "obb_raw_half_lengths_m": np.asarray(obb.extents["half_lengths"]).tolist(),
        "states": state_rows,
        "reference": reference_summary,
        "plot": str(plot_path),
        "elapsed_s": time.perf_counter() - started,
    }
    trial.write_json(output, payload)
    make_plot(plot_path, model, states, raw_points, state_rows)
    return payload


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False, default=trial.json_default))


if __name__ == "__main__":
    main()
