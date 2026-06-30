#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-data/results/ch6_2_sim}"

"${PYTHON_BIN}" -m experiments.exp_62_sim_occupancy_risk \
  --output "${OUTPUT_DIR}" \
  --scenes static_safe,approach,crossing,leave \
  --trials 5 \
  --frames 90 \
  --prediction-horizon 0.4 \
  --risk-margin 0.02 \
  --prediction-uncertainty 0.01 \
  --velocity-radius-scale 0.25 \
  --plot

echo "[setup_ch6_2] results saved to ${OUTPUT_DIR}"
