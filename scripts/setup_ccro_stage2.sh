#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/home/hzy/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-py310}"

"${CONDA_BIN}" run -n "${CONDA_ENV}" python -m pip install \
  -r "${ROOT_DIR}/requirements-stage2.txt"

if ! ls "${ROOT_DIR}"/planning/_nubs_cpp*.so >/dev/null 2>&1; then
  CONDA_BIN="${CONDA_BIN}" CONDA_ENV="${CONDA_ENV}" \
    bash "${ROOT_DIR}/scripts/build_ccro_stage1.sh"
fi

"${CONDA_BIN}" run -n "${CONDA_ENV}" python -m pytest \
  -p no:cacheprovider \
  "${ROOT_DIR}/tests/planning/test_nubs_stage1.py" \
  "${ROOT_DIR}/tests/planning/test_ccro_stage2.py" -q

echo "CCRO-NUBS stage-two environment is ready."
echo "Run: ${CONDA_BIN} run -n ${CONDA_ENV} python -m experiments.exp_ccro_stage2"
