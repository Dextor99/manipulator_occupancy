"""Single-instance linearization audit for Fast CCRO-NUBS v4."""

from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from planning.nubs_trajectory import NUBSTrajectory6D

from . import config_64 as cfg
from .common_64 import (
    constant_forecast,
    git_commit_hash,
    load_stage4_config,
    load_surface_model,
    make_reference,
    make_risk_stack,
    write_json,
)
from .fast_local_repair_64 import _make_local_reference, _trajectory_min, make_fast_instances
from .repair.active_distance import extract_dense_nearest_distances
from .repair.local_qp_solver import solve_local_qp
from .repair.nubs_linearization import build_local_sensitivity


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def _world_point(model, q: np.ndarray, link: str, local_point: np.ndarray) -> np.ndarray:
    transform = model.urdf.link_transforms(model._joint_dict(q))[link]
    return np.asarray(local_point) @ transform[:3, :3].T + transform[:3, 3]


def _finite_point_jacobian(model, q: np.ndarray, link: str, local_point: np.ndarray, *, eps: float = 1.0e-6) -> np.ndarray:
    values = np.asarray(q, dtype=np.float64)
    jac = np.zeros((3, 6), dtype=np.float64)
    for joint in range(6):
        plus = values.copy()
        minus = values.copy()
        plus[joint] += eps
        minus[joint] -= eps
        jac[:, joint] = (
            _world_point(model, plus, link, local_point)
            - _world_point(model, minus, link, local_point)
        ) / (2.0 * eps)
    return jac


def _dense_min_row(evaluator, trajectory, forecast) -> dict[str, Any]:
    sample_times = np.arange(0.0, trajectory.total_duration + 0.5 * cfg.FAST_SAMPLE_DT, cfg.FAST_SAMPLE_DT)
    active = extract_dense_nearest_distances(
        evaluator,
        trajectory,
        forecast,
        sample_times=sample_times,
        top_k=1,
    )
    if not active:
        return {"distance": math.inf, "time": None, "link": None}
    item = active[0]
    return {
        "distance": float(item.distance),
        "time": float(item.tau),
        "link": item.nearest_link,
        "world_point": None if item.world_point is None else item.world_point.tolist(),
    }


def run_audit(output: Path, *, scenario: str, instance_index: int) -> dict[str, Any]:
    config = load_stage4_config(cfg.STAGE4_CONFIG)
    model = load_surface_model(config)
    reference, _, _, _ = make_reference(config)
    evaluator, online_verifier, limits = make_risk_stack(config, model, None)
    instances = make_fast_instances(
        model,
        reference,
        evaluator,
        scenario=scenario,
        smoke=False,
        g1=False,
        g1_near=True,
    )
    instance = instances[int(instance_index)]
    local_ref, p_inner, head, tail = _make_local_reference(reference, float(instance["tau_start"]))
    forecast = constant_forecast(
        np.asarray(instance["obstacle_center0"], dtype=np.float64),
        np.asarray(instance["obstacle_velocity"], dtype=np.float64),
        float(instance["obstacle_radius"]),
    )
    sample_times = np.arange(0.0, local_ref.total_duration + 0.5 * cfg.FAST_SAMPLE_DT, cfg.FAST_SAMPLE_DT)
    t0 = time.perf_counter()
    active = extract_dense_nearest_distances(
        evaluator,
        local_ref,
        forecast,
        sample_times=sample_times,
        top_k=cfg.FAST_V3_ACTIVE_CONSTRAINTS,
    )
    sensitivity = build_local_sensitivity(
        p_inner,
        head,
        tail,
        local_ref.durations,
        sample_times,
        epsilon=cfg.FAST_V3_SENSITIVITY_EPS,
    )
    qp = solve_local_qp(
        active,
        sensitivity,
        limits,
        trust_region=cfg.FAST_V3_TRUST_REGION,
        d_safe=cfg.D_ONLINE_ACCEPT,
    )
    delta_applied = cfg.FAST_V3_RELAXATION * qp.delta
    candidate_points = p_inner + delta_applied.reshape(p_inner.shape)
    candidate = NUBSTrajectory6D().generate(candidate_points, head, tail, local_ref.durations)
    online = online_verifier.verify(
        candidate,
        forecast,
        current_q=head[:, 0],
        current_qd=head[:, 1],
        current_qdd=head[:, 2],
        q_goal=tail[:, 0],
        solver_success=True,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    reference_dense_min = _trajectory_min(evaluator, local_ref, forecast, density="dense")
    candidate_dense_min = _trajectory_min(evaluator, candidate, forecast, density="dense")
    rows = []
    jacobian_audit = None
    for index, item in enumerate(active):
        time_index = int(np.argmin(np.abs(sensitivity.sample_times - item.tau)))
        a_row = np.einsum("j,jv->v", item.gradient_q, sensitivity.sq[time_index], optimize=True)
        predicted_gain = float(np.dot(a_row, delta_applied))
        cp_norms = np.linalg.norm(a_row.reshape(p_inner.shape), axis=1)
        row = {
            "index": index,
            "link": item.nearest_link,
            "time": float(item.tau),
            "distance": float(item.distance),
            "gradient_q": item.gradient_q.tolist(),
            "gradient_q_norm": float(np.linalg.norm(item.gradient_q)),
            "a_row_norm": float(np.linalg.norm(a_row)),
            "predicted_gain": predicted_gain,
            "predicted_distance": float(item.distance + predicted_gain),
            "control_point_row_norms": cp_norms.tolist(),
        }
        rows.append(row)
        if index == 0 and item.nearest_link is not None and item.local_point is not None:
            analytic = model.point_jacobian(item.q, item.nearest_link, item.local_point)
            finite = _finite_point_jacobian(model, item.q, item.nearest_link, item.local_point)
            diff = analytic - finite
            jacobian_audit = {
                "link": item.nearest_link,
                "time": float(item.tau),
                "analytic": analytic.tolist(),
                "finite_difference": finite.tolist(),
                "absolute_error_norm": float(np.linalg.norm(diff)),
                "relative_error": float(np.linalg.norm(diff) / max(np.linalg.norm(finite), 1.0e-12)),
            }
    payload = {
        "experiment": "6.4 v4 linearization audit",
        "git_commit": git_commit_hash(),
        "git_dirty_expected": True,
        "instance": instance,
        "reference_dense_min_distance": float(reference_dense_min),
        "candidate_dense_min_distance": float(candidate_dense_min),
        "actual_dense_gain": float(candidate_dense_min - reference_dense_min),
        "global_dense_before": _dense_min_row(evaluator, local_ref, forecast),
        "global_dense_after": _dense_min_row(evaluator, candidate, forecast),
        "qp": {
            "success": qp.success,
            "status": qp.status,
            "message": qp.message,
            "iterations": qp.iterations,
            "objective": qp.objective,
            "min_predicted_distance": qp.min_predicted_distance,
        },
        "delta": {
            "norm": float(np.linalg.norm(delta_applied)),
            "max_abs": float(np.max(np.abs(delta_applied), initial=0.0)),
            "per_control_point_norm": np.linalg.norm(delta_applied.reshape(p_inner.shape), axis=1).tolist(),
        },
        "online_check": {
            "accepted": online.accepted,
            "reasons": online.reasons,
            "min_distance": online.min_distance,
            "max_qd_violation": online.max_qd_violation,
            "max_qdd_violation": online.max_qdd_violation,
        },
        "active_constraints": rows,
        "jacobian_audit": jacobian_audit,
        "elapsed_ms": float(elapsed_ms),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "v4_linearization_audit.json", payload)
    write_audit_markdown(payload, output / "paper" / "table_6_4_v4_linearization_audit.md")
    return payload


def write_audit_markdown(payload: dict[str, Any], path: Path) -> None:
    jac = payload.get("jacobian_audit") or {}
    lines = [
        "# 6.4 v4 linearization audit",
        "",
        "| item | value |",
        "|---|---:|",
        f"| reference dense min / m | {_fmt(payload['reference_dense_min_distance'])} |",
        f"| candidate dense min / m | {_fmt(payload['candidate_dense_min_distance'])} |",
        f"| actual dense gain / m | {_fmt(payload['actual_dense_gain'])} |",
        f"| QP min predicted distance / m | {_fmt(payload['qp']['min_predicted_distance'])} |",
        f"| delta norm / rad | {_fmt(payload['delta']['norm'])} |",
        f"| max abs delta / rad | {_fmt(payload['delta']['max_abs'])} |",
        f"| online accepted | {payload['online_check']['accepted']} |",
        f"| elapsed / ms | {_fmt(payload['elapsed_ms'], 2)} |",
        "",
        "## Jacobian Check",
        "",
        "| link | relative error | absolute error norm |",
        "|---|---:|---:|",
        f"| {jac.get('link', '-')} | {_fmt(jac.get('relative_error'))} | {_fmt(jac.get('absolute_error_norm'))} |",
        "",
        "## Active Constraints",
        "",
        "| i | link | time / s | d / m | grad norm | A norm | predicted gain / m | predicted d / m | cp-row norms |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["active_constraints"]:
        cp = ",".join(_fmt(value, 3) for value in row["control_point_row_norms"])
        lines.append(
            f"| {row['index']} | {row['link']} | {_fmt(row['time'])} | {_fmt(row['distance'])} | "
            f"{_fmt(row['gradient_q_norm'])} | {_fmt(row['a_row_norm'])} | "
            f"{_fmt(row['predicted_gain'])} | {_fmt(row['predicted_distance'])} | {cp} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(Path("results/new/6_4_v4_linearization_audit")))
    parser.add_argument("--scenario", choices=["D1", "D2M"], default="D1")
    parser.add_argument("--instance-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name("config_64.yaml"), output / "config_64.yaml")
    payload = run_audit(output, scenario=args.scenario, instance_index=args.instance_index)
    print(f"[6.4 v4 audit] dense gain={payload['actual_dense_gain']:.6f} m")
    print(f"[6.4 v4 audit] saved {output / 'v4_linearization_audit.json'}")
    print(f"[6.4 v4 audit] saved {output / 'paper' / 'table_6_4_v4_linearization_audit.md'}")


if __name__ == "__main__":
    main()
