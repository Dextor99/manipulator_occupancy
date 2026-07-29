"""Calibrate online-vs-dense distance margins from fast 6.4 trial outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    metrics_path = path / "fast_local_repair_64.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return metrics.get("trials", [])


def summarize(paths: list[Path]) -> list[dict[str, Any]]:
    summaries = []
    for path in paths:
        rows = _load_rows(path)
        if not rows:
            continue
        gaps = np.asarray([row["candidate_medium_dense_gap"] for row in rows], dtype=np.float64)
        online_margins = np.asarray([row["online_threshold_margin"] for row in rows], dtype=np.float64)
        dense_margins = np.asarray([row["dense_threshold_margin"] for row in rows], dtype=np.float64)
        accepted = [row for row in rows if row.get("online_feasible")]
        summaries.append(
            {
                "source": str(path),
                "method": rows[0].get("method", "-"),
                "n": len(rows),
                "dense_safe": float(np.mean([row.get("dense_geometry_only_feasible", False) for row in rows])),
                "online_acceptance": float(np.mean([row.get("online_feasible", False) for row in rows])),
                "verified_safety": (
                    float(np.mean([row.get("dense_geometry_only_feasible", False) for row in accepted]))
                    if accepted
                    else None
                ),
                "gap_mean": float(np.mean(gaps)),
                "gap_abs_p95": float(np.percentile(np.abs(gaps), 95)),
                "gap_min": float(np.min(gaps)),
                "gap_max": float(np.max(gaps)),
                "online_margin_mean": float(np.mean(online_margins)),
                "online_margin_p05": float(np.percentile(online_margins, 5)),
                "online_margin_min": float(np.min(online_margins)),
                "dense_margin_min": float(np.min(dense_margins)),
            }
        )
    return summaries


def write_table(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# 6.4 Online-vs-Dense Margin Calibration",
        "",
        "| source | method | n | dense safe | online acceptance | verified safety | mean online-dense gap / m | P95 abs gap / m | gap min / m | gap max / m | mean online margin / m | P05 online margin / m | min online margin / m | min dense margin / m |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['source']}` | {row['method']} | {row['n']} | {_fmt(row['dense_safe'], 2)} | "
            f"{_fmt(row['online_acceptance'], 2)} | {_fmt(row['verified_safety'], 2)} | "
            f"{_fmt(row['gap_mean'])} | {_fmt(row['gap_abs_p95'])} | {_fmt(row['gap_min'])} | {_fmt(row['gap_max'])} | "
            f"{_fmt(row['online_margin_mean'])} | {_fmt(row['online_margin_p05'])} | "
            f"{_fmt(row['online_margin_min'])} | {_fmt(row['dense_margin_min'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", default="results/new/6_4_margin_calibration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    rows = summarize([Path(value).resolve() for value in args.inputs])
    output.mkdir(parents=True, exist_ok=True)
    (output / "margin_calibration_64.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    write_table(rows, output / "paper" / "table_6_4_margin_calibration.md")
    print(f"[6.4 margin] saved {output / 'paper' / 'table_6_4_margin_calibration.md'}")


if __name__ == "__main__":
    main()
