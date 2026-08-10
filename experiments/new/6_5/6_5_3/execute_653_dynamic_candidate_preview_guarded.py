#!/usr/bin/env python3
"""Guarded low-speed playback for a 6.5.3 dynamic shadow candidate.

This is a thin 6.5.3 wrapper around the already validated Offline Track
executor used in 6.5.2.  It is for slow motion-shape preview of an accepted
``candidate_preview_package``.  It is not an online dynamic switch experiment.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
EXP652 = ROOT / "experiments" / "new" / "6_5" / "6_5_2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP652) not in sys.path:
    sys.path.insert(0, str(EXP652))

import execute_652_ccro_nubs_offline_track_guarded as executor  # noqa: E402


DEFAULT_PLAN_DIR = (
    ROOT
    / "results"
    / "new"
    / "6_5"
    / "6_5_3"
    / "dynamic_repair_pilot"
    / "trials"
    / "D1_crossing_body_r14"
    / "candidate_preview_package"
)
DEFAULT_OUTPUT = ROOT / "results" / "new" / "6_5" / "6_5_3" / "candidate_preview_execution"
REQUIRED_OPERATOR_PHRASE = "CCRO_653_DYNAMIC_CANDIDATE_PREVIEW_APPROVED"


def main() -> None:
    executor.REQUIRED_OPERATOR_PHRASE = REQUIRED_OPERATOR_PHRASE
    parser = executor.build_parser()
    parser.description = __doc__
    parser.set_defaults(plan_dir=DEFAULT_PLAN_DIR, output=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.execute and args.min_execution_wait_s <= 0.0 and args.playback_duration_s > 0.0:
        args.min_execution_wait_s = max(0.0, 0.9 * args.playback_duration_s)
    executor.run(args)


if __name__ == "__main__":
    main()
