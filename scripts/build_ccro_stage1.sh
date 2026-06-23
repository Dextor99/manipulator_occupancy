#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/home/hzy/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-py310}"
BUILD_DIR="${ROOT_DIR}/planning/cpp/build"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "conda executable not found: ${CONDA_BIN}" >&2
  exit 1
fi

PYTHON_PREFIX="$(${CONDA_BIN} run -n "${CONDA_ENV}" python -c 'import sys; print(sys.prefix)')"
PYBIND11_DIR="$(${CONDA_BIN} run -n "${CONDA_ENV}" python -m pybind11 --cmakedir)"

"${CONDA_BIN}" run -n "${CONDA_ENV}" cmake \
  -S "${ROOT_DIR}/planning/cpp" \
  -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${PYTHON_PREFIX}" \
  -Dpybind11_DIR="${PYBIND11_DIR}" \
  -DNUBS_INCLUDE_DIR="${ROOT_DIR}/NUBSTrajectory-main/include"

"${CONDA_BIN}" run -n "${CONDA_ENV}" cmake \
  --build "${BUILD_DIR}" \
  --config Release \
  --parallel 2

"${CONDA_BIN}" run -n "${CONDA_ENV}" python -c \
  "from planning import _nubs_cpp; print('built', _nubs_cpp.__file__)"
