"""Run revised Chapter 6.2 whole-body coverage experiment."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import config_62 as cfg
from .body_coverage_62 import critical_point_distance, evaluate_body_sample, risk_label
from .common_62 import ensure_output_tree, load_surface_model, median_runtime_ms, read_json, write_json
from .generate_body_samples import generate_samples


METHOD_NAMES = {
    "critical": "Critical-point APF",
    "ccro": "CCRO",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run revised 6.2 body coverage comparison.")
    parser.add_argument("--output", default=str(cfg.DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED + 1200)
    parser.add_argument("--regenerate-samples", action="store_true")
    return parser.parse_args()


def load_or_generate_samples(output: Path, surface, seed: int, regenerate: bool) -> list[dict[str, Any]]:
    path = output / "body" / "body_samples_62.json"
    if path.exists() and not regenerate:
        return read_json(path)
    from .common_62 import make_reference_trajectory, save_reference_trajectory

    trajectory = make_reference_trajectory()
    save_reference_trajectory(output / "reference_trajectory_62.npz", trajectory)
    samples = generate_samples(surface, trajectory, seed)
    write_json(path, samples)
    return samples


def evaluate_with_runtime(surface, sample: dict[str, Any]) -> dict[str, Any]:
    evaluated = evaluate_body_sample(surface, sample)
    q = np.asarray(sample["q"], dtype=float)
    center = np.asarray(sample["obstacle_center"], dtype=float)
    radius = float(sample["obstacle_radius"])
    from .body_coverage_62 import build_critical_points

    critical_points = build_critical_points(surface, q)
    ccro_points = surface.surface(q, density="medium")

    def critical_query():
        result = critical_point_distance(critical_points, center, radius)
        return risk_label(result.distance)

    def ccro_query():
        distances = np.linalg.norm(ccro_points - center[None, :], axis=1) - radius
        return risk_label(float(np.min(distances)))

    evaluated["runtime_critical_ms"] = median_runtime_ms(critical_query)
    evaluated["runtime_ccro_ms"] = median_runtime_ms(ccro_query)
    return evaluated


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    risk_rows = [row for row in rows if row["risk_gt"]]
    out = {"methods": {}}
    for method, distance_key, risk_key, runtime_key in (
        ("critical", "D_critical", "risk_critical", "runtime_critical_ms"),
        ("ccro", "D_ccro", "risk_ccro", "runtime_ccro_ms"),
    ):
        detected = sum(1 for row in risk_rows if row[risk_key])
        distance_errors = [abs(float(row[distance_key]) - float(row["D_gt"])) * 1000.0 for row in rows]
        runtimes = [float(row[runtime_key]) for row in rows]
        false_positive_rows = [row for row in rows if not row["risk_gt"]]
        false_positives = sum(1 for row in false_positive_rows if row[risk_key])
        out["methods"][method] = {
            "body_detection_rate_percent": 100.0 * detected / max(len(risk_rows), 1),
            "risk_detected": detected,
            "risk_total": len(risk_rows),
            "distance_error_mm_mean": float(np.mean(distance_errors)),
            "distance_error_mm_std": float(np.std(distance_errors, ddof=1)) if len(distance_errors) > 1 else 0.0,
            "runtime_ms_mean": float(np.mean(runtimes)),
            "runtime_ms_std": float(np.std(runtimes, ddof=1)) if len(runtimes) > 1 else 0.0,
            "false_positives": false_positives,
            "safe_total": len(false_positive_rows),
            "false_positive_rate_percent": 100.0 * false_positives / max(len(false_positive_rows), 1),
            "specificity_percent": 100.0 * (len(false_positive_rows) - false_positives) / max(len(false_positive_rows), 1),
            "precision_percent": 100.0 * detected / max(detected + false_positives, 1),
            "balanced_accuracy_percent": 0.5 * (
                100.0 * detected / max(len(risk_rows), 1)
                + 100.0 * (len(false_positive_rows) - false_positives) / max(len(false_positive_rows), 1)
            ),
        }
    return out


def write_table(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "| 方法 | 风险检测率 / % | 误报率 / % | 距离MAE / mm | 查询时间 / ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in ("critical", "ccro"):
        item = metrics["methods"][method]
        lines.append(
            f"| {METHOD_NAMES[method]} | {item['body_detection_rate_percent']:.1f} "
            f"| {item['false_positive_rate_percent']:.1f} "
            f"| {item['distance_error_mm_mean']:.2f} ± {item['distance_error_mm_std']:.2f} "
            f"| {item['runtime_ms_mean']:.3f} ± {item['runtime_ms_std']:.3f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    paths = ensure_output_tree(output)
    surface = load_surface_model()
    samples = load_or_generate_samples(output, surface, int(args.seed), args.regenerate_samples)
    rows = []
    for sample in samples:
        result = evaluate_with_runtime(surface, sample)
        rows.append(result)
        write_json(paths["body_trials"] / f"sample_{int(sample['sample_id']):03d}.json", result)
    metrics = aggregate(rows)
    write_json(paths["body"] / "summary.json", {"samples": rows, "metrics": metrics})
    write_table(paths["paper"] / "table_6_2_body_coverage.md", metrics)


if __name__ == "__main__":
    main()
