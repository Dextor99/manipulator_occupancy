#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-data/results/ch6_3_sim}"

"${PYTHON_BIN}" -m experiments.exp_63_sim_risk_distance \
  --output "${OUTPUT_DIR}" \
  --trials 30

echo "[setup_ch6_3] results saved to ${OUTPUT_DIR}"
