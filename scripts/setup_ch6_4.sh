#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-results/new/6_4}"

"${PYTHON_BIN}" -m experiments.new.6_4.run_dynamic_replanning \
  --output "${OUTPUT_DIR}"

"${PYTHON_BIN}" -m experiments.new.6_4.audit_64 \
  --output "${OUTPUT_DIR}" \
  --density dense \
  --time-step 0.04

echo "[setup_ch6_4] results saved to ${OUTPUT_DIR}"
