#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
for stage in p0 p1 p2 p3 p4 p5 p6 p7; do
  echo "[CCRO-NUBS] running ${stage}"
  bash "scripts/setup_ccro_${stage}.sh"
done
