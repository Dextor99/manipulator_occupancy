#!/usr/bin/env python3
"""Compare archived r09 V2 STRO with V3 adaptive multi-sphere STRO.

Pure offline diagnostic: no camera, SDK connection or robot authority.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk.prediction import RiskSphere

trial = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")
v3 = importlib.import_module("experiments.new.6_5.6_5_3.dynamic_nubs_v3")

DEFAULT_SOURCE = ROOT / "results/new/6_5/6_5_3/simple_dynamic_nubs_complete_live/r09/core_live/trials/D2_opposing_approach_r09"
DEFAULT_REFERENCE = ROOT / "results/new/6_5/6_5_3/reference_xp10_line/reference_feedback.csv"
DEFAULT_OUTPUT = ROOT / "results/new/6_5/6_5_3/v3_stro_r09_replay"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-trial-dir", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--reference-feedback-csv", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p


def run(args: argparse.Namespace) -> dict:
    source = args.source_trial_dir.resolve()
    summary = json.loads((source / "summary.json").read_text())
    candidate = json.loads((source / "candidate/candidate_summary.json").read_text())
    trigger = next(e for e in summary["events"] if e["type"] == "TRIGGER")
    points = np.load(source / "fresh_latest_cluster_points.npy")
    geometry = v3.adaptive_geometry_adapter(points, fit_margin_m=0.005, max_components=4)

    q_now = np.asarray(candidate["q_now"], dtype=np.float64)
    reference = trial.RecordedReference.load(args.reference_feedback_csv.resolve())
    reference.index = int(np.argmin(np.max(np.abs(reference.q - q_now[None, :]), axis=1)))
    runtime = trial.build_parser().parse_args(["--scene", "D2", "--mode", "shadow"])
    config = trial.load_stage4_config(runtime.stage4_config)
    model = trial.load_stage4_surface_model(config)

    trigger_center = np.asarray(trigger["tracker_center_m"], dtype=np.float64)
    raw_center = np.mean(points, axis=0)
    centers = np.asarray(geometry["component_centers"], dtype=np.float64)
    centers += (trigger_center - raw_center)[None, :]
    radii = np.asarray(geometry["component_base_radii"], dtype=np.float64)
    velocity = np.asarray(trigger["window_velocity_m_s"], dtype=np.float64)
    speed = float(np.linalg.norm(velocity))
    spheres: list[RiskSphere] = []
    component_rows = []
    v3_best = {
        "distance": float("inf"),
        "tau": None,
        "link": None,
        "object_id": None,
        "component": None,
    }
    for tau in np.arange(runtime.prediction_step_s, runtime.prediction_horizon_s + 1e-9,
                         runtime.prediction_step_s):
        uncertainty = 0.020 + 0.10 * speed * float(tau)
        q_tau, _, _ = reference.state_after(float(tau))
        surfaces = model.surface_by_link(q_tau, density=runtime.surface_density)
        for component, (center, radius) in enumerate(zip(centers, radii), 1):
            sphere = RiskSphere(
                int(trigger["track_id"]), center + velocity * float(tau),
                float(radius + uncertainty), float(tau)
            )
            spheres.append(sphere)
            component_best = {"distance": float("inf"), "link": None}
            for link, surface in surfaces.items():
                if len(surface) == 0:
                    continue
                distance = float(
                    cKDTree(surface).query(np.asarray(sphere.center), k=1)[0]
                    - sphere.radius
                )
                if distance < component_best["distance"]:
                    component_best = {"distance": distance, "link": link}
                if distance < v3_best["distance"]:
                    v3_best = {
                        "distance": distance,
                        "tau": float(tau),
                        "link": link,
                        "object_id": int(trigger["track_id"]),
                        "component": component,
                    }
            component_rows.append({
                "component": component, "tau_s": float(tau),
                "center_m": sphere.center.tolist(), "radius_m": sphere.radius,
                "nearest_link": component_best["link"],
                "clearance_m": component_best["distance"],
            })
    v2_tau = float(trigger["predicted_tau_s"])
    legacy_radius = (
        float(trigger["tracked_radius_m"]) + 0.035 + 0.020
        + 0.10 * speed * v2_tau
    )
    result = {
        "status": "V3_STRO_R09_REPLAY_COMPLETE",
        "robot_commanded": False,
        "camera_opened": False,
        "source_trial_dir": str(source),
        "reference_index": reference.index,
        "v2": {
            "predicted_clearance_m": float(trigger["predicted_distance_m"]),
            "trigger_tau_s": v2_tau,
            "nearest_link": trigger["predicted_link"],
            "single_sphere_radius_at_trigger_tau_m": legacy_radius,
        },
        "v3": {
            "predicted_clearance_m": float(v3_best["distance"]),
            "trigger_tau_s": v3_best["tau"],
            "nearest_link": v3_best["link"],
            "nearest_object_id": v3_best["object_id"],
            "nearest_component": v3_best["component"],
            "component_count": int(geometry["component_count"]),
            "component_base_radii_m": radii.tolist(),
            "max_component_radius_at_0p5s_m": float(np.max(radii) + 0.020 + 0.10 * speed * 0.5),
            "axial_length_m": float(geometry["axial_length_m"]),
            "transverse_radius_m": float(geometry["transverse_radius_m"]),
            "geometry_policy": "adaptive_pca_1_to_4_spheres",
        },
        "predicted_clearance_delta_v3_minus_v2_m": (
            float(v3_best["distance"]) - float(trigger["predicted_distance_m"])
        ),
        "component_profile": component_rows,
    }
    output = args.output.resolve()
    trial.write_json(output / "summary.json", result)
    return result


def main() -> None:
    print(json.dumps(run(parser().parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
