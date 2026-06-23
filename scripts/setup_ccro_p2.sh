#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/hzy/miniconda3/envs/py310/bin/python}"
cd "$ROOT"
bash scripts/build_ccro_stage1.sh
"$PYTHON" -m pytest -q tests/planning/test_ccro_p2.py
"$PYTHON" -m experiments.exp_ccro_p2
