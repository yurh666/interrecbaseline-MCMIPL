#!/usr/bin/env bash
set -euo pipefail

BASELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${BASELINE_DIR}/MCMIPL"
LOG_DIR="${BASELINE_DIR}/results/raw_logs"
DATA_NAME="${1:-LAST_FM_STAR}"

mkdir -p "${LOG_DIR}"

cd "${REPO_DIR}"

{
  echo "===== graph_init ====="
  echo "data_name=${DATA_NAME}"
  echo "cwd=$(pwd)"
  date
  python graph_init.py --data_name "${DATA_NAME}"
  date
} 2>&1 | tee "${LOG_DIR}/graph_init_${DATA_NAME}.log"
