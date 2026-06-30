#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-data/results/ch6_6}"

"${PYTHON_BIN}" -m experiments.exp_ccro_p7_dry_run
"${PYTHON_BIN}" -m experiments.exp_66_summary \
  --output "${OUTPUT_DIR}"

echo "[setup_ch6_6] results saved to ${OUTPUT_DIR}"
