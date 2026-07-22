#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-config/ccro_stage2.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-data/results/6_3}"
INSTANCES_PER_SCENARIO="${INSTANCES_PER_SCENARIO:-10}"

"${PYTHON_BIN}" -m experiments.new.6_3.run_static_benchmark \
  --config "${CONFIG_PATH}" \
  --output "${OUTPUT_DIR}" \
  --instances-per-scenario "${INSTANCES_PER_SCENARIO}" \
  --force-regenerate

"${PYTHON_BIN}" experiments/new/6_3/plot_static_benchmark.py \
  --input "${OUTPUT_DIR}" \
  --output "${OUTPUT_DIR}/paper"

echo "[setup_ch6_3] results saved to ${OUTPUT_DIR}"
