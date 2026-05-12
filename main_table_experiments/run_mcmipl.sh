#!/bin/bash
DATASET=${1:-LAST_FM_STAR}
SEED=${2:-0}
MAX_STEPS=${3:-100}
SAMPLE_TIMES=${4:-100}
EVAL_NUM=${5:-10}

PYTHON=/home/yurh/.conda/envs/mcmipl-reproduce/bin/python
MCMIPL_DIR=/home/yurh/main_table_experiments/baselines/mcmipl_official/MCMIPL
LOG_DIR=/home/yurh/main_table_experiments/logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${DATASET}_s${SEED}.log"

echo "=== MCMIPL | $DATASET | seed=$SEED | steps=$MAX_STEPS | $(date) ===" | tee "$LOG_FILE"

cd "$MCMIPL_DIR" || exit 1
$PYTHON -u RL_model.py \
  --data_name "$DATASET" \
  --embed transe \
  --seed "$SEED" \
  --gpu 0 \
  --max_steps "$MAX_STEPS" \
  --sample_times "$SAMPLE_TIMES" \
  --attr_num 20 \
  --choice_num 4 \
  --max_turn 15 \
  --eval_num "$EVAL_NUM" \
  --save_num "$MAX_STEPS" \
  2>&1 | tee -a "$LOG_FILE"

echo "=== DONE: $DATASET seed=$SEED at $(date) ===" | tee -a "$LOG_FILE"
