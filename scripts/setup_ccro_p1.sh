#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/home/hzy/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-py310}"

CONDA_BIN="${CONDA_BIN}" CONDA_ENV="${CONDA_ENV}" \
  bash "${ROOT_DIR}/scripts/setup_ccro_p0.sh"

"${CONDA_BIN}" run -n "${CONDA_ENV}" \
  python -m experiments.exp_ccro_p1

echo "CCRO-NUBS P1 Monte-Carlo robustness experiment passed."
