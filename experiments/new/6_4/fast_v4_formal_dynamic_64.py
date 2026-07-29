"""Frozen-parameter D1/D2 formal dynamic validation for Fast CCRO-NUBS v4."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import config_64 as cfg


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


def _run_case(output: Path, *, scenario: str, method: str, risk_band: str) -> dict[str, Any]:
    case_dir = output / f"{scenario.lower()}_{method}_{risk_band}"
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "experiments.new.6_4.fast_local_repair_64",
        "--output",
        str(case_dir),
        "--scenario",
        scenario,
        "--formal-compact",
        "--formal-risk-band",
        risk_band,
        "--method",
        method,
        "--v4-target-clearance",
        f"{cfg.FAST_V4_TARGET_CLEARANCE:.6f}",
        "--v4-max-iterations",
        str(cfg.FAST_V4_MAX_ITERATIONS),
        "--v4-clearance-reward",
        f"{cfg.FAST_V4_CLEARANCE_REWARD:.6f}",
    ]
    subprocess.run(cmd, cwd=cfg.ROOT, check=True)
    metrics = json.loads((case_dir / "fast_local_repair_64.json").read_text(encoding="utf-8"))
    trials = metrics["trials"]
    summary_rows = metrics["summary"]
    return {
        "case_dir": str(case_dir.relative_to(cfg.ROOT)),
        "scenario": scenario,
        "method": method,
        "risk_band": risk_band,
        "trial_count": len(trials),
        "summary": summary_rows,
    }


def _flatten_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = []
    for case in rows:
        for row in case["summary"]:
            flat.append(
                {
                    "scenario": case["scenario"],
                    "method": case["method"],
                    "risk_band": case["risk_band"],
                    "speed_group": row["speed_group"],
                    "conflict": row["lead_label"],
                    "n": row["n"],
                    "repair_success": row["repair_success"],
                    "online_acceptance": row["online_acceptance"],
                    "verified_safety": row["verified_safety"],
                    "time_pass": row["time_pass"],
                    "acceleration_ok": row["acceleration_ok"],
                    "delta_dense": row["delta_dense"],
                    "online_ms_mean": row["online_ms_mean"],
                    "online_ms_p95": row["online_ms_p95"],
                    "online_ms_max": row["online_ms_max"],
                    "path_deformation_rms": row.get("path_deformation_rms"),
                    "path_deformation_max": row.get("path_deformation_max"),
                    "case_dir": case["case_dir"],
                }
            )
    return flat


def _write_table(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# 6.4 Frozen Fast v4 D1/D2 Formal Dynamic Validation",
        "",
        "| scenario | risk band | method | speed | conflict | n | repair success | online acceptance | verified safety | time pass | accel pass | mean dD dense / m | online mean / ms | online p95 / ms | online max / ms | deformation RMS / rad | case |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['risk_band']} | {row['method']} | {row['speed_group']} | {row['conflict']} | {row['n']} | "
            f"{_fmt(row['repair_success'], 2)} | {_fmt(row['online_acceptance'], 2)} | {_fmt(row['verified_safety'], 2)} | "
            f"{_fmt(row['time_pass'], 2)} | {_fmt(row['acceleration_ok'], 2)} | {_fmt(row['delta_dense'], 4)} | "
            f"{_fmt(row['online_ms_mean'], 1)} | {_fmt(row['online_ms_p95'], 1)} | {_fmt(row['online_ms_max'], 1)} | "
            f"{_fmt(row['path_deformation_rms'], 4)} | `{row['case_dir']}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/new/6_4_fast_v4_formal_dynamic")
    parser.add_argument("--scenarios", nargs="+", choices=["D1", "D2M"], default=["D1", "D2M"])
    parser.add_argument("--methods", nargs="+", choices=["critical_fast_v4", "ccro_fast_v4"], default=["critical_fast_v4", "ccro_fast_v4"])
    parser.add_argument("--risk-band", choices=sorted(cfg.FAST_DYNAMIC_RISK_BANDS), default="admissible")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for scenario in args.scenarios:
            for method in args.methods:
                print(f"[6.4 formal] would run scenario={scenario} method={method} risk_band={args.risk_band}")
        return
    rows = []
    for scenario in args.scenarios:
        for method in args.methods:
            row = _run_case(output, scenario=scenario, method=method, risk_band=args.risk_band)
            rows.append(row)
            print(f"[6.4 formal] {scenario} {method} risk_band={args.risk_band} trials={row['trial_count']}")
    flat = _flatten_summary(rows)
    payload = {
        "experiment": "6.4 frozen Fast v4 formal dynamic validation",
        "risk_band": args.risk_band,
        "risk_band_range": list(cfg.FAST_DYNAMIC_RISK_BANDS[args.risk_band]),
        "cases": rows,
        "summary": flat,
    }
    (output / "fast_v4_formal_dynamic_64.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_table(flat, output / "paper" / "table_6_4_fast_v4_formal_dynamic.md")
    print(f"[6.4 formal] saved {output / 'paper' / 'table_6_4_fast_v4_formal_dynamic.md'}")


if __name__ == "__main__":
    main()
