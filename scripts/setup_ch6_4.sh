#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-results/new/6_4}"

"${PYTHON_BIN}" -m experiments.new.6_4.run_dynamic_replanning \
  --output "${OUTPUT_DIR}"

echo "[setup_ch6_4] results saved to ${OUTPUT_DIR}"
