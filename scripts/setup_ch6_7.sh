#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-data/results/ch6_7}"

"${PYTHON_BIN}" -m experiments.exp_67_summary \
  --output "${OUTPUT_DIR}"

echo "[setup_ch6_7] results saved to ${OUTPUT_DIR}"
