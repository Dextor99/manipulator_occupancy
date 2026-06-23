#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PYTHON="${PYTHON:-/home/hzy/miniconda3/envs/py310/bin/python}"; cd "$ROOT"
"$PYTHON" -m experiments.exp_ccro_p5
