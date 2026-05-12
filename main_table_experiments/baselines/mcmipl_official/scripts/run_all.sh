#!/usr/bin/env bash
set -euo pipefail

BASELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS=("${@:-LAST_FM_STAR}")
SEEDS=(0 1 2)
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-100}"

for data_name in "${DATASETS[@]}"; do
  bash "${BASELINE_DIR}/scripts/prepare_data.sh" "${data_name}"
  bash "${BASELINE_DIR}/scripts/run_graph_init.sh" "${data_name}"
  for seed in "${SEEDS[@]}"; do
    bash "${BASELINE_DIR}/scripts/run_train.sh" "${data_name}" "${seed}"
    bash "${BASELINE_DIR}/scripts/run_eval.sh" "${data_name}" "${CHECKPOINT_EPOCH}" "${seed}"
  done
done
