"""Generate final paper-ready tables for 6.4 Fast CCRO-NUBS v4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METHOD_NAMES = {
    "critical_fast_v4": "Critical-fast-v4",
    "ccro_fast_v4": "CCRO-fast-v4",
}

BAND_ORDER = {"easy": 0, "main": 1, "hard": 2, "near": 3}
METHOD_ORDER = {"critical_fast_v4": 0, "ccro_fast_v4": 1}


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


def _load_case_trials(root: Path, case_dir: str) -> list[dict[str, Any]]:
    case_path = Path(case_dir)
    candidates = [
        case_path if case_path.is_absolute() else Path.cwd() / case_path,
        root / case_path,
        root / case_path.name,
    ]
    metrics_path = next((candidate / "fast_local_repair_64.json" for candidate in candidates if (candidate / "fast_local_repair_64.json").exists()), None)
    if metrics_path is None:
        raise FileNotFoundError(f"could not locate fast_local_repair_64.json for {case_dir}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return metrics["trials"]


def _load_rows(input_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads((input_dir / "fast_v4_g1_band_study.json").read_text(encoding="utf-8"))
    return payload["rows"]


def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (BAND_ORDER.get(row["band"], 99), METHOD_ORDER.get(row["method"], 99)))


def _write_main_table(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# 6.4 Fast Local Repair Performance",
        "",
        "| Band | Method | N | Repair success | Online acceptance | Verified safety | P95 online / ms | Max online / ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _sorted_rows(rows):
        verified = row["verified_safety"]
        accepted = int(round(row["online_acceptance"] * row["n"]))
        verified_text = f"{int(round(verified * accepted))}/{accepted}" if accepted > 0 and verified is not None else "-"
        lines.append(
            f"| {row['band']} | {METHOD_NAMES.get(row['method'], row['method'])} | {row['n']} | "
            f"{int(round(row['repair_success'] * row['n']))}/{row['n']} | "
            f"{int(round(row['online_acceptance'] * row['n']))}/{row['n']} | "
            f"{verified_text} | {_fmt(row['online_ms_p95'], 1)} | {_fmt(row['online_ms_max'], 1)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gain_table(input_dir: Path, rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# 6.4 Dense Clearance Gain",
        "",
        "| Band | Method | N | Dref mean / m | Dcand mean / m | Delta D mean / m | Delta D std / m | Delta D P50 / m | Delta D min / m |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _sorted_rows(rows):
        trials = _load_case_trials(input_dir, row["case_dir"])
        ref = np.asarray([item["reference_dense_min_distance"] for item in trials], dtype=np.float64)
        cand = np.asarray([item["candidate_dense_min_distance"] for item in trials], dtype=np.float64)
        delta = np.asarray([item["delta_dense_min_distance"] for item in trials], dtype=np.float64)
        lines.append(
            f"| {row['band']} | {METHOD_NAMES.get(row['method'], row['method'])} | {row['n']} | "
            f"{_fmt(np.mean(ref), 4)} | {_fmt(np.mean(cand), 4)} | {_fmt(np.mean(delta), 4)} | "
            f"{_fmt(np.std(delta, ddof=1), 4)} | {_fmt(np.median(delta), 4)} | {_fmt(np.min(delta), 4)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_runtime_table(input_dir: Path, rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# 6.4 Runtime Decomposition",
        "",
        "| Band | Method | Risk extraction mean / ms | Sensitivity mean / ms | QP mean / ms | Online gate mean / ms | Online total mean / ms | Online total P95 / ms | Dense recheck mean / ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _sorted_rows(rows):
        trials = _load_case_trials(input_dir, row["case_dir"])
        risk = np.asarray([item["risk_scan_ms"] for item in trials], dtype=np.float64)
        linearization = np.asarray([item["linearization_ms"] for item in trials], dtype=np.float64)
        qp = np.asarray([item["qp_ms"] for item in trials], dtype=np.float64)
        gate = np.asarray([item["medium_gate_ms"] for item in trials], dtype=np.float64)
        online = np.asarray([item["online_ms"] for item in trials], dtype=np.float64)
        dense = np.asarray([item["dense_recheck_ms"] for item in trials], dtype=np.float64)
        lines.append(
            f"| {row['band']} | {METHOD_NAMES.get(row['method'], row['method'])} | "
            f"{_fmt(np.mean(risk), 2)} | {_fmt(np.mean(linearization), 2)} | {_fmt(np.mean(qp), 2)} | "
            f"{_fmt(np.mean(gate), 2)} | {_fmt(np.mean(online), 2)} | {_fmt(np.percentile(online, 95), 2)} | "
            f"{_fmt(np.mean(dense), 2)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/new/6_4_fast_v4_g1_band_study")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input).resolve()
    output = Path(args.output).resolve() if args.output else input_dir / "paper"
    output.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(input_dir)
    _write_main_table(rows, output / "table_6_4_fast_v4_main_performance.md")
    _write_gain_table(input_dir, rows, output / "table_6_4_fast_v4_clearance_gain.md")
    _write_runtime_table(input_dir, rows, output / "table_6_4_fast_v4_runtime_decomposition.md")
    print(f"[6.4 paper] saved tables under {output}")


if __name__ == "__main__":
    main()
