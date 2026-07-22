"""Collect revised 6.2 outputs into the paper directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config_62 as cfg
from .common_62 import ensure_output_tree, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize revised 6.2 outputs.")
    parser.add_argument("--output", default=str(cfg.DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    paths = ensure_output_tree(output)
    metrics = {}
    dynamic_path = paths["dynamic"] / "summary.json"
    body_path = paths["body"] / "summary.json"
    if dynamic_path.exists():
        dynamic = read_json(dynamic_path)
        metrics["dynamic"] = dynamic.get("metrics", {})
        metrics["dynamic_calibration"] = dynamic.get("calibration_metrics", {})
    if body_path.exists():
        metrics["body"] = read_json(body_path).get("metrics", {})
    write_json(paths["paper"] / "metrics.json", metrics)


if __name__ == "__main__":
    main()
