#!/usr/bin/env python3
"""Audit multisphere versus inflated-PCA-OBB clearance on one saved shadow run."""

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
    parser.add_argument("--inflation-m", type=float, default=0.05)
    parser.add_argument("--online-accept-m", type=float, default=0.09)
    parser.add_argument("--reference-step-s", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def point_obb_signed_distance(
    points: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    half_lengths: np.ndarray,
) -> np.ndarray:
    """Exact signed point-to-OBB distance; positive outside, negative inside."""
    local = (np.asarray(points, dtype=np.float64) - center[None, :]) @ rotation
    delta = np.abs(local) - half_lengths[None, :]
    outside = np.linalg.norm(np.maximum(delta, 0.0), axis=1)
    inside = np.minimum(np.max(delta, axis=1), 0.0)
    return outside + inside


def geometry_distance(
    model: Any,
    q: np.ndarray,
    *,
    sphere_centers: np.ndarray,
    sphere_radii: np.ndarray,
    obb_center: np.ndarray,
    obb_rotation: np.ndarray,
    obb_half_lengths: np.ndarray,
) -> dict[str, Any]:
    best_sphere = (float("inf"), None)
    best_obb = (float("inf"), None)
    for link, points in model.surface_by_link(q, density="medium").items():
        values = np.asarray(points, dtype=np.float64)
        sphere_distance = np.min(
            np.linalg.norm(values[:, None, :] - sphere_centers[None, :, :], axis=2)
            - sphere_radii[None, :]
        )
        obb_distance = np.min(
            point_obb_signed_distance(
                values,
                obb_center,
                obb_rotation,
                obb_half_lengths,
            )
        )
        if sphere_distance < best_sphere[0]:
            best_sphere = (float(sphere_distance), link)
        if obb_distance < best_obb[0]:
            best_obb = (float(obb_distance), link)
    return {
        "multisphere_distance_m": best_sphere[0],
        "multisphere_nearest_link": best_sphere[1],
        "obb_distance_m": best_obb[0],
        "obb_nearest_link": best_obb[1],
    }


def load_q_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    times = np.asarray([float(row["t_s"]) for row in rows], dtype=np.float64)
    q = np.asarray(
        [[float(row[f"q{joint}_rad"]) for joint in range(1, 7)] for row in rows],
        dtype=np.float64,
    )
    return times, q


def profile(
    model: Any,
    times: np.ndarray,
    q: np.ndarray,
    **geometry: Any,
) -> list[dict[str, Any]]:
    rows = []
    for tau, q_tau in zip(times, q):
        rows.append({"tau_s": float(tau), **geometry_distance(model, q_tau, **geometry)})
    return rows


def minimum(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    row = min(rows, key=lambda item: item[f"{prefix}_distance_m"])
    return {
        "distance_m": row[f"{prefix}_distance_m"],
        "tau_s": row["tau_s"],
        "nearest_link": row[f"{prefix}_nearest_link"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    source = args.source_run.resolve()
    points_path = source / "closure_fresh3_points.npy"
    if not points_path.exists():
        raise FileNotFoundError(
            f"{points_path} is missing; rerun the no-motion shadow after point archival was added"
        )
    if args.inflation_m < 0.0 or args.reference_step_s <= 0.0:
        raise ValueError("inflation-m must be non-negative and reference-step-s positive")
    points = np.asarray(np.load(points_path), dtype=np.float64)
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    runtime_args = trial.build_parser().parse_args(["--scene", "D2", "--mode", "shadow"])
    config = trial.load_stage4_config(runtime_args.stage4_config)
    model = trial.load_stage4_surface_model(config)
    reference = trial.RecordedReference.load(args.reference_feedback_csv.resolve())

    multisphere = trial.fit_pca_multisphere(
        points,
        fit_margin_m=runtime_args.multisphere_fit_margin_m,
        max_components=runtime_args.multisphere_max_components,
    )
    sphere_centers = np.asarray(multisphere["component_centers"], dtype=np.float64)
    sphere_radii = (
        np.asarray(multisphere["component_base_radii"], dtype=np.float64)
        + float(args.inflation_m)
    )
    obb = fit_obb(points)
    obb_center = np.asarray(obb.center, dtype=np.float64)
    obb_rotation = np.asarray(obb.rotation, dtype=np.float64)
    raw_half_lengths = np.asarray(obb.extents["half_lengths"], dtype=np.float64)
    obb_half_lengths = raw_half_lengths + float(args.inflation_m)
    local = (points - obb_center[None, :]) @ obb_rotation
    raw_coverage = np.all(np.abs(local) <= raw_half_lengths[None, :] + 1.0e-9, axis=1)

    geometry = {
        "sphere_centers": sphere_centers,
        "sphere_radii": sphere_radii,
        "obb_center": obb_center,
        "obb_rotation": obb_rotation,
        "obb_half_lengths": obb_half_lengths,
    }
    segment_profiles = []
    for segment in summary.get("segments", []):
        authorized_attempt = segment.get("authorized_attempt")
        if authorized_attempt is None:
            continue
        path = (
            source
            / f"segment_{int(segment['segment']):02d}"
            / f"attempt_{int(authorized_attempt):02d}"
            / "local_execution_authorization/authorized_local_repair.csv"
        )
        times, q = load_q_csv(path)
        rows = profile(model, times, q, **geometry)
        segment_profiles.append(
            {
                "segment": int(segment["segment"]),
                "trajectory_csv": str(path),
                "multisphere_minimum": minimum(rows, "multisphere"),
                "obb_minimum": minimum(rows, "obb"),
                "tail": rows[-1],
            }
        )

    reference_start = float(summary["segments"][0]["reference_plan_start_time_s"])
    reference_times = np.arange(
        reference_start,
        float(reference.times[-1]) + 0.5 * args.reference_step_s,
        args.reference_step_s,
    )
    reference_times = np.unique(np.r_[reference_times, float(reference.times[-1])])
    reference_q = np.vstack([reference.state_at(float(value))[0] for value in reference_times])
    reference_rows = profile(
        model,
        reference_times - reference_start,
        reference_q,
        **geometry,
    )

    endpoint_rows = []
    segment3_start = float(summary["segments"][-1]["reference_plan_start_time_s"])
    for offset in (1.25, 1.50, 1.75, 2.00):
        q = reference.state_at(segment3_start + offset)[0]
        endpoint_rows.append(
            {"offset_s": offset, **geometry_distance(model, q, **geometry)}
        )

    def safe_suffix_start(prefix: str) -> float | None:
        safe = np.asarray(
            [row[f"{prefix}_distance_m"] >= args.online_accept_m for row in reference_rows]
        )
        suffix = np.logical_and.accumulate(safe[::-1])[::-1]
        indices = np.flatnonzero(suffix)
        return None if not len(indices) else float(reference_rows[int(indices[0])]["tau_s"])

    payload = {
        "status": "STATIC_GEOMETRY_AB_COMPLETE",
        "source_run": str(source),
        "robot_commanded": False,
        "authoritative_for_execution": False,
        "point_count": int(len(points)),
        "inflation_m": float(args.inflation_m),
        "online_accept_m": float(args.online_accept_m),
        "multisphere": {
            "coverage_ratio": float(multisphere["coverage_ratio"]),
            "component_count": int(multisphere["component_count"]),
            "base_radii_m": np.asarray(multisphere["component_base_radii"]).tolist(),
            "inflated_radii_m": sphere_radii.tolist(),
        },
        "obb": {
            "coverage_ratio": float(np.mean(raw_coverage)),
            "center_m": obb_center.tolist(),
            "rotation": obb_rotation.tolist(),
            "raw_half_lengths_m": raw_half_lengths.tolist(),
            "inflated_half_lengths_m": obb_half_lengths.tolist(),
        },
        "segments": segment_profiles,
        "reference": {
            "multisphere_minimum": minimum(reference_rows, "multisphere"),
            "obb_minimum": minimum(reference_rows, "obb"),
            "goal": reference_rows[-1],
            "multisphere_safe_suffix_start_s": safe_suffix_start("multisphere"),
            "obb_safe_suffix_start_s": safe_suffix_start("obb"),
        },
        "rejoin_endpoints": endpoint_rows,
        "elapsed_s": time.perf_counter() - started,
    }
    output = (
        args.output.resolve()
        if args.output is not None
        else source / "offline_static_geometry_ab.json"
    )
    trial.write_json(output, payload)
    return payload


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False, default=trial.json_default))


if __name__ == "__main__":
    main()
