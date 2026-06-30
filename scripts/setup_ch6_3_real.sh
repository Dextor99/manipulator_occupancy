#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-data/results/ch6_3_real}"

"${PYTHON_BIN}" -m experiments.exp_63_real_summary \
  --output "${OUTPUT_DIR}" \
  --copy-figures

echo "[setup_ch6_3_real] results saved to ${OUTPUT_DIR}"
