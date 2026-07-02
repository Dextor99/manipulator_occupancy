"""Prepare a standard Chapter 6 E1-E5 experiment output directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ch6_common import DEFAULT_CONFIG, prepare_run_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id", help="Experiment id, for example E1_occupancy_risk_final.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = prepare_run_directory(
        args.experiment_id,
        output=args.output,
        config_path=args.config,
        overwrite=args.overwrite,
    )
    print(json.dumps({"prepared": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
