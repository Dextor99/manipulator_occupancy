"""Build E2 perturbation-batch statistics and a P2 D_min(t) figure."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_65_official_tesseract_trajopt import _trajectory_from_waypoints  # noqa: E402
from experiments.exp_ccro_stage2 import _baseline, _limits, _load, _states  # noqa: E402
from planning.mesh_risk import MeshRiskEvaluator, StaticObstacleField  # noqa: E402
from planning.minco_trajectory import MinJerkMINCOTrajectory6D  # noqa: E402
from planning.nubs_trajectory import NUBSTrajectory6D  # noqa: E402
from planning.robot_surface_model import RobotSurfaceModel  # noqa: E402
from planning.verifier import TrajectoryVerifier  # noqa: E402


SCENARIOS = ("A", "B", "C")
SCENE_LABELS = {"A": "P1", "B": "P2", "C": "P3"}
METHODS = [
    ("official_tesseract_trajopt", "Official TrajOpt/Tesseract", "official", "nubs_like"),
    ("chomp_style", "CHOMP-style", "classical", "nubs"),
    ("trajopt_style", "TrajOpt-style", "classical", "nubs"),
    ("gpmp2_style", "GPMP2-style", "classical", "nubs"),
    ("minco_risk", "MINCO-risk", "external", "minco"),
    ("full_body", "Ours CCRO-NUBS", "stage2", "nubs"),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.6g}"
    return str(value)


def markdown(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def make_filled_ball(
    center: np.ndarray, radius: float, point_count: int, rng: np.random.Generator
) -> np.ndarray:
    directions = rng.normal(size=(point_count - 1, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = radius * np.cbrt(rng.random(point_count - 1))
    return np.vstack((center[None, :], center[None, :] + directions * radii[:, None]))


def perturb_obstacle(
    obstacle_info: dict[str, Any],
    *,
    rng: np.random.Generator,
    base_point_count: int,
    radius_jitter: float,
    center_jitter_ratio: float,
) -> tuple[StaticObstacleField, dict[str, Any]]:
    details = []
    chunks = []
    obstacles = obstacle_info["obstacles"]
    per_obstacle_points = max(2, int(base_point_count // max(1, len(obstacles))))
    for obstacle in obstacles:
        center = np.asarray(obstacle["center"], dtype=np.float64)
        radius = float(obstacle.get("radius", 0.035))
        outward = center - np.asarray(obstacle["surface_point"], dtype=np.float64)
        norm = float(np.linalg.norm(outward))
        if norm < 1.0e-9:
            outward = rng.normal(size=3)
            norm = float(np.linalg.norm(outward))
        outward = outward / max(norm, 1.0e-9)
        tangent = rng.normal(size=3)
        tangent -= float(np.dot(tangent, outward)) * outward
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm < 1.0e-9:
            tangent = rng.normal(size=3)
            tangent_norm = float(np.linalg.norm(tangent))
        tangent = tangent / tangent_norm
        shift = radius * center_jitter_ratio * (
            0.65 * rng.normal() * tangent + 0.35 * rng.normal() * outward
        )
        perturbed_radius = radius * float(np.clip(1.0 + radius_jitter * rng.normal(), 0.75, 1.25))
        perturbed_center = center + shift
        chunks.append(make_filled_ball(perturbed_center, perturbed_radius, per_obstacle_points, rng))
        details.append(
            {
                "center": perturbed_center.tolist(),
                "radius": perturbed_radius,
                "source_link": obstacle.get("link"),
            }
        )
    points = np.vstack(chunks)
    return StaticObstacleField.from_points(points), {
        "obstacles": details,
        "point_count": int(len(points)),
    }


def build_context(
    config_path: Path,
    *,
    verifier_density: str | None = None,
    verifier_time_step: float | None = None,
):
    config = _load(config_path)
    head, tail, durations = _states(config)
    limits = _limits(config)
    robot = config["robot"]
    surface_cfg = config["surface"]
    surface_model = RobotSurfaceModel(
        ROOT / robot["urdf_path"],
        robot["joint_names"],
        surface_cfg["density_totals"],
        seed=surface_cfg["random_seed"],
        min_points_per_link=surface_cfg["min_points_per_link"],
        cache_dir=surface_cfg["cache_dir"],
        geometry=surface_cfg["geometry"],
    )
    risk_cfg = config["risk"]
    evaluator = MeshRiskEvaluator(
        surface_model,
        d_safe=risk_cfg["d_safe"],
        d_activate=risk_cfg["d_activate"],
        fd_epsilon_q=risk_cfg["fd_epsilon_q"],
        density=risk_cfg["optimizer_density"],
    )
    verifier = TrajectoryVerifier(
        evaluator,
        limits,
        d_stop=risk_cfg["d_stop"],
        time_step=verifier_time_step or config["validation"]["dense_time_step"],
        density=verifier_density or risk_cfg["validation_density"],
        epsilon_goal=config["validation"]["epsilon_goal"],
        epsilon_continuity_q=config["validation"]["epsilon_continuity_q"],
        epsilon_continuity_qd=config["validation"]["epsilon_continuity_qd"],
        epsilon_continuity_qdd=config["validation"]["epsilon_continuity_qdd"],
        limit_tolerance=config["validation"]["limit_tolerance"],
    )
    baseline = _baseline(config, head, tail, durations).trajectory
    return config, head, tail, durations, evaluator, verifier, baseline


def reconstruct_trajectory(
    row: dict[str, Any],
    kind: str,
    head: np.ndarray,
    tail: np.ndarray,
    durations: np.ndarray,
):
    if kind == "minco":
        return MinJerkMINCOTrajectory6D.from_inner_points(
            np.asarray(row["p_inner"], dtype=np.float64), head, tail, durations
        )
    if kind == "nubs":
        return NUBSTrajectory6D().generate(
            np.asarray(row["p_inner"], dtype=np.float64), head, tail, durations
        )
    if kind == "nubs_like":
        return _trajectory_from_waypoints(
            np.asarray(row["q_waypoints"], dtype=np.float64), head, tail, durations
        )
    raise ValueError(f"unknown trajectory kind: {kind}")


def method_row(
    scenario: str,
    method_key: str,
    source: str,
    stage2: dict[str, Any],
    external: dict[str, Any],
    classical: dict[str, Any],
    official: dict[str, Any],
) -> dict[str, Any]:
    if source == "stage2":
        return stage2["scenarios"][scenario]["methods"][method_key]
    if source == "external":
        return external["scenarios"][scenario]["methods"][method_key]
    if source == "classical":
        return classical["scenarios"][scenario]["methods"][method_key]
    if source == "official":
        return official["scenarios"][scenario]["methods"][method_key]
    raise ValueError(f"unknown source: {source}")


def evaluate_perturbations(
    root: Path,
    config_path: Path,
    *,
    trials: int,
    seed: int,
    radius_jitter: float,
    center_jitter_ratio: float,
    verifier_density: str,
    verifier_time_step: float,
) -> dict[str, Any]:
    config, head, tail, durations, evaluator, _verifier, _ = build_context(
        config_path,
        verifier_density=verifier_density,
        verifier_time_step=verifier_time_step,
    )
    stage2 = load_json(root / "reuse" / "ccro_stage2" / "metrics.json")
    external = load_json(root / "reuse" / "ch6_4_external" / "metrics.json")
    classical = load_json(root / "classical_optimizers" / "metrics.json")
    official = load_json(root / "official_tesseract_trajopt" / "metrics.json")
    rng = np.random.default_rng(seed)
    sample_count = max(2, int(np.ceil(float(np.sum(durations)) / verifier_time_step)) + 1)
    sample_times = np.linspace(0.0, float(np.sum(durations)), sample_count)
    results: dict[str, Any] = {
        "source": "E2 perturbation-batch verifier statistics",
        "trials_per_scenario": trials,
        "seed": seed,
        "jitter": {
            "radius_jitter": radius_jitter,
            "center_jitter_ratio": center_jitter_ratio,
        },
        "verification": {
            "density": verifier_density,
            "time_step": verifier_time_step,
            "role": "supplemental perturbation statistics; final acceptance table remains dense",
        },
        "notes": {
            "scope": "Fixed planned trajectories are re-verified under perturbed obstacle point clouds.",
            "rrt": "RRT-Connect already has multi-seed statistics in external baseline results and is not reconstructed here.",
        },
        "scenarios": {},
    }
    for scenario in SCENARIOS:
        obstacle_info = stage2["scenarios"][scenario]["obstacle"]
        base_point_count = int(obstacle_info["point_count"])
        scenario_payload = {"methods": {}, "perturbations": []}
        perturbations = [
            perturb_obstacle(
                obstacle_info,
                rng=rng,
                base_point_count=base_point_count,
                radius_jitter=radius_jitter,
                center_jitter_ratio=center_jitter_ratio,
            )
            for _ in range(trials)
        ]
        scenario_payload["perturbations"] = [info for _, info in perturbations]
        for method_key, label, source, kind in METHODS:
            row = method_row(scenario, method_key, source, stage2, external, classical, official)
            trajectory = reconstruct_trajectory(row, kind, head, tail, durations)
            accepted = []
            dmins = []
            risks = []
            verify_ms = []
            nearest_links: list[str | None] = []
            for obstacle, _ in perturbations:
                started = time.perf_counter()
                risk = evaluator.trajectory(
                    trajectory,
                    obstacle,
                    sample_times,
                    density=verifier_density,
                    with_gradient=False,
                )
                accepted.append(bool(row["solver_success"]) and risk.min_distance >= config["risk"]["d_stop"])
                dmins.append(float(risk.min_distance))
                risks.append(float(risk.cost))
                verify_ms.append((time.perf_counter() - started) * 1000.0)
                nearest_links.append(risk.nearest_link)
            scenario_payload["methods"][method_key] = {
                "method_label": label,
                "accepted_rate": float(np.mean(accepted)),
                "accepted_count": int(np.sum(accepted)),
                "trial_count": int(trials),
                "d_min_mean": float(np.mean(dmins)),
                "d_min_std": float(np.std(dmins)),
                "d_min_min": float(np.min(dmins)),
                "d_min_max": float(np.max(dmins)),
                "risk_mean": float(np.mean(risks)),
                "risk_std": float(np.std(risks)),
                "verify_ms_mean": float(np.mean(verify_ms)),
                "nearest_links": nearest_links,
            }
        results["scenarios"][scenario] = scenario_payload
    return results


def write_perturbation_table(metrics: dict[str, Any], output: Path) -> None:
    rows = []
    for scenario in SCENARIOS:
        for method_key, label, _, _ in METHODS:
            row = metrics["scenarios"][scenario]["methods"][method_key]
            rows.append(
                [
                    SCENE_LABELS[scenario],
                    scenario,
                    label,
                    f"{row['accepted_count']}/{row['trial_count']}",
                    fmt(row["accepted_rate"]),
                    fmt(row["d_min_mean"]),
                    fmt(row["d_min_std"]),
                    fmt(row["d_min_min"]),
                    fmt(row["risk_mean"]),
                    fmt(row["verify_ms_mean"]),
                ]
            )
    table = markdown(
        [
            "scene",
            "source_scene",
            "method",
            "accepted",
            "accepted_rate",
            "D_min_mean",
            "D_min_std",
            "D_min_min",
            "J_risk_mean",
            "T_eval_ms",
        ],
        rows,
    )
    output.write_text(table + "\n", encoding="utf-8")


def plot_p2_dmin_curve(root: Path, config_path: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    config, head, tail, durations, evaluator, _, _ = build_context(config_path)
    stage2 = load_json(root / "reuse" / "ccro_stage2" / "metrics.json")
    external = load_json(root / "reuse" / "ch6_4_external" / "metrics.json")
    classical = load_json(root / "classical_optimizers" / "metrics.json")
    official = load_json(root / "official_tesseract_trajopt" / "metrics.json")
    obstacle_npz = np.load(root / "reuse" / "ccro_stage2" / "scenario_B_obstacle.npz")
    obstacle = StaticObstacleField.from_points(obstacle_npz["points"])
    times = np.linspace(0.0, float(np.sum(durations)), 161)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for method_key, label, source, kind in METHODS:
        row = method_row("B", method_key, source, stage2, external, classical, official)
        trajectory = reconstruct_trajectory(row, kind, head, tail, durations)
        risk = evaluator.trajectory(trajectory, obstacle, times, with_gradient=False)
        ax.plot(times, risk.sample_distances, linewidth=1.8, label=label)
    ax.axhline(config["risk"]["d_stop"], color="tab:red", linestyle="--", linewidth=1.3, label="d_stop")
    ax.axhline(config["risk"]["d_safe"], color="tab:orange", linestyle=":", linewidth=1.3, label="d_safe")
    ax.set_title("P2 / B middle-link near obstacle")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("D_min(t) (m)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/results/ch6_e1_e5/E2_static_planning_benchmark")
    parser.add_argument("--config", default="config/ccro_stage2.yaml")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--radius-jitter", type=float, default=0.08)
    parser.add_argument("--center-jitter-ratio", type=float, default=0.40)
    parser.add_argument("--verify-density", default="coarse")
    parser.add_argument("--verify-time-step", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output = root / "perturbation_batch"
    final = root / "final"
    figures = final / "figures"
    output.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_perturbations(
        root,
        Path(args.config),
        trials=args.trials,
        seed=args.seed,
        radius_jitter=args.radius_jitter,
        center_jitter_ratio=args.center_jitter_ratio,
        verifier_density=args.verify_density,
        verifier_time_step=args.verify_time_step,
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_perturbation_table(metrics, output / "table_E2_perturbation_batch.md")
    plot_p2_dmin_curve(root, Path(args.config), figures / "fig_E2_P2_Dmin_curve.png")
    print(f"[E2] perturbation batch saved to {output}")
    print(f"[E2] P2 D_min(t) figure saved to {figures / 'fig_E2_P2_Dmin_curve.png'}")


if __name__ == "__main__":
    main()
