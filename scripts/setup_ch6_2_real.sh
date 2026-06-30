#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-data/results/ch6_2_real}"

"${PYTHON_BIN}" -m experiments.exp_62_real_summary \
  --output "${OUTPUT_DIR}" \
  --copy-figures

echo "[setup_ch6_2_real] results saved to ${OUTPUT_DIR}"
