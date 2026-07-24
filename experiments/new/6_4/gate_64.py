"""Mechanism gates for the near-final Chapter 6.4 experiment."""

from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

import numpy as np

from .common_64 import (
    constant_forecast,
    load_stage4_config,
    load_surface_model,
    make_critical_risk_stack,
    make_reference,
    make_risk_stack,
    write_json,
)


def benchmark_risk_queries(repeats: int) -> dict[str, object]:
    config = load_stage4_config()
    model = load_surface_model(config)
    reference, *_ = make_reference(config)
    mesh_evaluator, _, _ = make_risk_stack(config, model, None)
    critical_evaluator, _ = make_critical_risk_stack(config, model)
    q = reference.evaluate(0.45 * reference.total_duration)
    forecast = constant_forecast(
        np.array([0.4, 0.0, 0.5], dtype=np.float64),
        np.array([0.08, 0.02, 0.0], dtype=np.float64),
        0.05,
    )
    rows = {}
    for name, evaluator in [("critical", critical_evaluator), ("ccro", mesh_evaluator)]:
        for _ in range(5):
            evaluator.configuration(q, forecast, 0.5, density="coarse", with_gradient=True)
        started = time.perf_counter()
        for _ in range(repeats):
            evaluator.configuration(q, forecast, 0.5, density="coarse", with_gradient=True)
        rows[name] = (time.perf_counter() - started) * 1000.0 / repeats
    return {
        "critical_gradient_query_ms": float(rows["critical"]),
        "ccro_gradient_query_ms": float(rows["ccro"]),
        "critical_point_count": int(critical_evaluator.critical_point_count()),
        "critical_faster_than_ccro": bool(rows["critical"] < rows["ccro"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/new/6_4_final_gate")
    parser.add_argument("--repeats", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    result = {
        "risk_query_benchmark": benchmark_risk_queries(max(1, int(args.repeats))),
        "module": importlib.import_module(__package__ + ".config_64").__name__,
    }
    write_json(output / "gate_64.json", result)
    print(result)


if __name__ == "__main__":
    main()
