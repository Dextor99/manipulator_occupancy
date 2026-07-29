"""Generate paper-ready D1/D2 dynamic tables for 6.4 Fast CCRO-NUBS v4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import config_64 as cfg


METHOD_NAMES = {
    "critical_fast_v4": "Critical-fast-v4",
    "ccro_fast_v4": "CCRO-fast-v4",
}
METHOD_ORDER = {"critical_fast_v4": 0, "ccro_fast_v4": 1}
SCENARIO_ORDER = {"D1": 0, "D2M": 1}


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def _load_payload(input_dir: Path) -> dict[str, Any]:
    return json.loads((input_dir / "fast_v4_formal_dynamic_64.json").read_text(encoding="utf-8"))


def _load_case_trials(case_dir: str) -> list[dict[str, Any]]:
    case_path = Path(case_dir)
    if not case_path.is_absolute():
        case_path = cfg.ROOT / case_path
    metrics_path = case_path / "fast_local_repair_64.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return metrics["trials"]


def _group_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    groups = []
    for case in payload["cases"]:
        trials = _load_case_trials(case["case_dir"])
        groups.append(
            {
                "scenario": case["scenario"],
                "method": case["method"],
                "risk_band": case["risk_band"],
                "case_dir": case["case_dir"],
                "trials": trials,
            }
        )
    return sorted(groups, key=lambda item: (SCENARIO_ORDER.get(item["scenario"], 99), METHOD_ORDER.get(item["method"], 99)))


def _count(trials: list[dict[str, Any]], key: str) -> int:
    return sum(1 for item in trials if bool(item.get(key, False)))


def _verified_text(trials: list[dict[str, Any]]) -> str:
    accepted = [item for item in trials if bool(item.get("online_feasible", False))]
    if not accepted:
        return "-"
    verified = sum(1 for item in accepted if bool(item.get("dense_geometry_only_feasible", False)))
    return f"{verified}/{len(accepted)}"


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def _write_performance(groups: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# 6.4 D1/D2 Admissible Dynamic Performance",
        "",
        "| Scenario | Risk band | Method | N | Repair success | Online acceptance | Verified safety | P95 online / ms | Max online / ms |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        trials = group["trials"]
        online = np.asarray([item["online_ms"] for item in trials], dtype=np.float64)
        lines.append(
            f"| {group['scenario']} | {group['risk_band']} | {METHOD_NAMES.get(group['method'], group['method'])} | {len(trials)} | "
            f"{_count(trials, 'dense_geometry_only_feasible')}/{len(trials)} | "
            f"{_count(trials, 'online_feasible')}/{len(trials)} | {_verified_text(trials)} | "
            f"{_fmt(_percentile(online, 95), 1)} | {_fmt(np.max(online), 1)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_clearance_gain(groups: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# 6.4 D1/D2 Dynamic Dense Clearance Gain",
        "",
        "| Scenario | Risk band | Method | N | Dref mean / m | Dref range / m | Dcand mean / m | Delta D mean / m | Delta D std / m | Delta D P50 / m | Dense-safe repaired | Online accepted |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        trials = group["trials"]
        ref = np.asarray([item["reference_dense_min_distance"] for item in trials], dtype=np.float64)
        cand = np.asarray([item["candidate_dense_min_distance"] for item in trials], dtype=np.float64)
        delta = np.asarray([item["delta_dense_min_distance"] for item in trials], dtype=np.float64)
        lines.append(
            f"| {group['scenario']} | {group['risk_band']} | {METHOD_NAMES.get(group['method'], group['method'])} | {len(trials)} | "
            f"{_fmt(np.mean(ref), 4)} | [{_fmt(np.min(ref), 4)}, {_fmt(np.max(ref), 4)}] | "
            f"{_fmt(np.mean(cand), 4)} | {_fmt(np.mean(delta), 4)} | {_fmt(np.std(delta, ddof=1), 4)} | "
            f"{_fmt(np.median(delta), 4)} | {_count(trials, 'dense_geometry_only_feasible')}/{len(trials)} | "
            f"{_count(trials, 'online_feasible')}/{len(trials)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_runtime(groups: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# 6.4 D1/D2 Dynamic Runtime",
        "",
        "| Scenario | Risk band | Method | Risk scan mean / ms | Sensitivity mean / ms | QP mean / ms | Online gate mean / ms | Online total mean / ms | Online total P95 / ms | Dense recheck mean / ms |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        trials = group["trials"]
        risk = np.asarray([item["risk_scan_ms"] for item in trials], dtype=np.float64)
        linearization = np.asarray([item["linearization_ms"] for item in trials], dtype=np.float64)
        qp = np.asarray([item["qp_ms"] for item in trials], dtype=np.float64)
        gate = np.asarray([item["medium_gate_ms"] for item in trials], dtype=np.float64)
        online = np.asarray([item["online_ms"] for item in trials], dtype=np.float64)
        dense = np.asarray([item["dense_recheck_ms"] for item in trials], dtype=np.float64)
        lines.append(
            f"| {group['scenario']} | {group['risk_band']} | {METHOD_NAMES.get(group['method'], group['method'])} | "
            f"{_fmt(np.mean(risk), 2)} | {_fmt(np.mean(linearization), 2)} | {_fmt(np.mean(qp), 2)} | "
            f"{_fmt(np.mean(gate), 2)} | {_fmt(np.mean(online), 2)} | {_fmt(_percentile(online, 95), 2)} | "
            f"{_fmt(np.mean(dense), 2)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _representative_trial(trials: list[dict[str, Any]], *, accepted: bool) -> dict[str, Any] | None:
    pool = [item for item in trials if bool(item.get("online_feasible", False)) == accepted]
    if not pool:
        return None
    if accepted:
        return max(pool, key=lambda item: (item["delta_dense_min_distance"], item["candidate_dense_min_distance"]))
    return min(pool, key=lambda item: item["candidate_dense_min_distance"])


def _write_representative_cases(groups: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# 6.4 D1/D2 Representative Dynamic Cases",
        "",
        "| Scenario | Risk band | Method | Case type | Trial | Dref / m | Dcand / m | Delta D / m | Online Dmin / m | Online / ms | Dense safe | Online accepted | Path RMS / rad | Path max / rad |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        for label, accepted in (("accepted-high-gain", True), ("rejected-low-margin", False)):
            trial = _representative_trial(group["trials"], accepted=accepted)
            if trial is None:
                continue
            lines.append(
                f"| {group['scenario']} | {group['risk_band']} | {METHOD_NAMES.get(group['method'], group['method'])} | {label} | "
                f"{trial['instance_id']} | {_fmt(trial.get('reference_dense_min_distance'), 4)} | "
                f"{_fmt(trial.get('candidate_dense_min_distance'), 4)} | {_fmt(trial.get('delta_dense_min_distance'), 4)} | "
                f"{_fmt(trial.get('candidate_online_min_distance'), 4)} | {_fmt(trial.get('online_ms'), 1)} | "
                f"{int(bool(trial.get('dense_geometry_only_feasible', False)))} | {int(bool(trial.get('online_feasible', False)))} | "
                f"{_fmt(trial.get('path_deformation_rms'), 4)} | {_fmt(trial.get('path_deformation_max'), 4)} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/new/6_4_fast_v4_formal_dynamic_admissible")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input).resolve()
    output = Path(args.output).resolve() if args.output else input_dir / "paper"
    output.mkdir(parents=True, exist_ok=True)
    groups = _group_cases(_load_payload(input_dir))
    _write_performance(groups, output / "table_6_4_fast_v4_dynamic_performance.md")
    _write_clearance_gain(groups, output / "table_6_4_fast_v4_dynamic_clearance_gain.md")
    _write_runtime(groups, output / "table_6_4_fast_v4_dynamic_runtime.md")
    _write_representative_cases(groups, output / "table_6_4_fast_v4_dynamic_representative_cases.md")
    print(f"[6.4 dynamic paper] saved tables under {output}")


if __name__ == "__main__":
    main()
