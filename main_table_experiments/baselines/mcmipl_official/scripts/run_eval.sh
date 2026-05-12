#!/usr/bin/env bash
set -euo pipefail

BASELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${BASELINE_DIR}/MCMIPL"
LOG_DIR="${BASELINE_DIR}/results/raw_logs"
METRIC_DIR="${BASELINE_DIR}/results/metrics"
DATA_NAME="${1:-LAST_FM_STAR}"
CHECKPOINT_EPOCH="${2:-100}"
SEED="${3:-0}"

mkdir -p "${LOG_DIR}" "${METRIC_DIR}"

cd "${REPO_DIR}"

LOG_FILE="${LOG_DIR}/eval_${DATA_NAME}_${SEED}.log"
METRIC_FILE="${METRIC_DIR}/eval_${DATA_NAME}_${SEED}.json"

set +e
{
  echo "===== evaluate MCMIPL ====="
  echo "data_name=${DATA_NAME}"
  echo "checkpoint_epoch=${CHECKPOINT_EPOCH}"
  echo "seed=${SEED}"
  echo "cwd=$(pwd)"
  git rev-parse HEAD || true
  date
  python evaluate.py --data_name "${DATA_NAME}" --load_rl_epoch "${CHECKPOINT_EPOCH}" --seed "${SEED}"
  echo "exit_code=$?"
  date
} 2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

python "${BASELINE_DIR}/scripts/parse_eval_log.py" "${LOG_FILE}" "${METRIC_FILE}" "${DATA_NAME}" "${SEED}" "${CHECKPOINT_EPOCH}" "${status}"

exit "${status}"
