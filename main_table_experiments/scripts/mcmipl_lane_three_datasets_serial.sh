#!/usr/bin/env bash
# Serial lane: BOOK -> LAST_FM_STAR -> MOVIE; within each dataset run only seeds 0,1
# that are NOT yet archived. Uses seed-isolated RL-agent paths + per-seed archive.
#
# Run inside screen, e.g.:
#   screen -dmS mcmipl_lane_3ds bash -lc \
#     'bash main_table_experiments/scripts/mcmipl_lane_three_datasets_serial.sh 2>&1 | tee -a main_table_experiments/logs/screen_mcmipl_lane_3ds_$(date +%Y%m%d_%H%M%S).log'
set -euo pipefail

MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MCMIPL_FORCE_CPU="${MCMIPL_FORCE_CPU:-1}"
export MCMIPL_CPU_PYTHON="${MCMIPL_CPU_PYTHON:-/home/yrh666/venvs/mcmipl-cpu/bin/python}"
export MCMIPL_SAVE_NUM="${MCMIPL_SAVE_NUM:-10}"

DATASETS=(BOOK LAST_FM_STAR MOVIE)
SEEDS=(0 1)
MAX_STEPS="${MAX_STEPS:-50}"
SAMPLE_TIMES="${SAMPLE_TIMES:-100}"
EVAL_NUM="${EVAL_NUM:-10}"

LANE_ID="3ds_$(date +%Y%m%d_%H%M%S)"
LANE_LOG="${MAIN}/logs/lane_three_datasets_${LANE_ID}.log"
exec > >(tee -a "$LANE_LOG") 2>&1

echo "=== LANE three datasets (serial) | lane_id=${LANE_ID} | $(date) ==="
echo "=== datasets=${DATASETS[*]} seeds=${SEEDS[*]} (skip if already archived) ==="

for DATASET in "${DATASETS[@]}"; do
  echo ""
  echo "=== DATASET ${DATASET} | $(date) ==="
  bash "${MAIN}/scripts/mcmipl_archive_phase_a.sh" "$DATASET" || true

  for SEED in "${SEEDS[@]}"; do
    if bash "${MAIN}/scripts/mcmipl_seed_archive_status.sh" "$DATASET" "$SEED" >/dev/null 2>&1; then
      echo "[skip] ${DATASET} seed=${SEED} already archived (see archives/checkpoints/.../seed_${SEED}/MANIFEST.txt)"
      continue
    fi
    SLUG_FOR_ID="$(bash -c "source ${MAIN}/scripts/mcmipl_data_slug.sh && data_slug ${DATASET}")"
    export MCMIPL_RUN_ID="${SLUG_FOR_ID}_s${SEED}_${LANE_ID}"
    echo "[run] ${DATASET} seed=${SEED} run_id=${MCMIPL_RUN_ID}"
    bash "${MAIN}/scripts/mcmipl_run_seed_with_archive.sh" \
      "$DATASET" "$SEED" "$MAX_STEPS" "$SAMPLE_TIMES" "$EVAL_NUM"
  done
done

echo ""
echo "=== LANE three datasets COMPLETE | lane_id=${LANE_ID} | $(date) ==="
