"""G1-band capability study for frozen Fast CCRO-NUBS v4."""

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


def _run_case(output: Path, *, band: str, method: str, scenario: str) -> dict[str, Any]:
    case_dir = output / f"{band}_{method}"
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "experiments.new.6_4.fast_local_repair_64",
        "--output",
        str(case_dir),
        "--scenario",
        scenario,
        "--g1-band",
        band,
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
    summary = metrics["summary"][0]
    trials = metrics["trials"]
    reference_dense = np.asarray([row["reference_dense_min_distance"] for row in trials], dtype=np.float64)
    online_accepted = [row for row in trials if row["online_feasible"]]
    return {
        "case_dir": str(case_dir.relative_to(cfg.ROOT)),
        "band": band,
        "band_range": list(cfg.FAST_G1_BANDS[band]),
        "method": method,
        "n": summary["n"],
        "reference_dense_mean": float(np.mean(reference_dense)),
        "reference_dense_min": float(np.min(reference_dense)),
        "reference_dense_max": float(np.max(reference_dense)),
        "qp_solved": summary["qp_solved"],
        "repair_success": summary["repair_success"],
        "online_acceptance": summary["online_acceptance"],
        "verified_safety": summary["verified_safety"],
        "time_pass": summary["time_pass"],
        "acceleration_ok": summary["acceleration_ok"],
        "usable": summary["usable"],
        "delta_dense": summary["delta_dense"],
        "online_margin_mean": float(np.mean([row["online_threshold_margin"] for row in trials])),
        "online_margin_min": summary["online_margin_min"],
        "dense_margin_min": summary["dense_margin_min"],
        "online_ms_mean": summary["online_ms_mean"],
        "online_ms_p95": summary["online_ms_p95"],
        "online_ms_max": summary["online_ms_max"],
        "accepted_dense_safe": (
            float(np.mean([row["dense_geometry_only_feasible"] for row in online_accepted]))
            if online_accepted
            else None
        ),
    }


def _write_table(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# 6.4 Fast v4 G1-Band Capability Study",
        "",
        "| band | Dref range / m | method | n | Dref mean / m | QP solved | repair success | online acceptance | verified safety | time pass | accel pass | mean dD dense / m | mean online margin / m | min dense margin / m | online mean / ms | online p95 / ms | online max / ms | case |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    order = {name: index for index, name in enumerate(["easy", "near", "main", "hard"])}
    method_order = {"critical_fast_v4": 0, "ccro_fast_v4": 1}
    for row in sorted(rows, key=lambda item: (order.get(item["band"], 99), method_order.get(item["method"], 99))):
        low, high = row["band_range"]
        lines.append(
            f"| {row['band']} | [{_fmt(low, 3)}, {_fmt(high, 3)}) | {row['method']} | "
            f"{row['n']} | {_fmt(row['reference_dense_mean'], 4)} | {_fmt(row['qp_solved'], 2)} | "
            f"{_fmt(row['repair_success'], 2)} | {_fmt(row['online_acceptance'], 2)} | "
            f"{_fmt(row['verified_safety'], 2)} | {_fmt(row['time_pass'], 2)} | {_fmt(row['acceleration_ok'], 2)} | "
            f"{_fmt(row['delta_dense'], 4)} | {_fmt(row['online_margin_mean'], 4)} | {_fmt(row['dense_margin_min'], 4)} | "
            f"{_fmt(row['online_ms_mean'], 1)} | {_fmt(row['online_ms_p95'], 1)} | {_fmt(row['online_ms_max'], 1)} | "
            f"`{row['case_dir']}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(Path("results/new/6_4_fast_v4_g1_band_study")))
    parser.add_argument("--scenario", choices=["D1", "D2M"], default="D1")
    parser.add_argument("--bands", nargs="+", choices=sorted(cfg.FAST_G1_BANDS), default=["easy", "main", "hard"])
    parser.add_argument("--methods", nargs="+", choices=["critical_fast_v4", "ccro_fast_v4"], default=["critical_fast_v4", "ccro_fast_v4"])
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for band in args.bands:
        for method in args.methods:
            row = _run_case(output, band=band, method=method, scenario=args.scenario)
            rows.append(row)
            print(
                f"[6.4 band] {band} {method} repair={row['repair_success']:.2f} "
                f"online={row['online_acceptance']:.2f} p95={row['online_ms_p95']:.1f} ms"
            )
    payload = {"experiment": "6.4 Fast v4 G1-band capability study", "rows": rows}
    (output / "fast_v4_g1_band_study.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_table(rows, output / "paper" / "table_6_4_fast_v4_g1_band_study.md")
    print(f"[6.4 band] saved {output / 'paper' / 'table_6_4_fast_v4_g1_band_study.md'}")


if __name__ == "__main__":
    main()
