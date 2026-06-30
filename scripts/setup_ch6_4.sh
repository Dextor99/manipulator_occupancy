#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-data/results/ch6_4}"
EXTERNAL_OUTPUT_DIR="${EXTERNAL_OUTPUT_DIR:-data/results/ch6_4_external}"
RRT_SEEDS="${RRT_SEEDS:-5}"

"${PYTHON_BIN}" -m experiments.exp_64_external_baselines \
  --output "${EXTERNAL_OUTPUT_DIR}" \
  --rrt-seeds "${RRT_SEEDS}"

"${PYTHON_BIN}" -m experiments.exp_64_summary \
  --output "${OUTPUT_DIR}"

echo "[setup_ch6_4] results saved to ${OUTPUT_DIR} and ${EXTERNAL_OUTPUT_DIR}"
