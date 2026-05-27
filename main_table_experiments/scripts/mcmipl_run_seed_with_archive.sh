#!/usr/bin/env bash
# Run one Phase-B seed on CPU, then archive checkpoint immediately (before next seed overwrites).
set -euo pipefail

MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MCMIPL_FORCE_CPU="${MCMIPL_FORCE_CPU:-1}"
export MCMIPL_CPU_PYTHON="${MCMIPL_CPU_PYTHON:-/home/yrh666/venvs/mcmipl-cpu/bin/python}"
export MCMIPL_SAVE_NUM="${MCMIPL_SAVE_NUM:-10}"

DATASET="${1:?DATASET}"
SEED="${2:?SEED}"
MAX_STEPS="${3:-50}"
SAMPLE_TIMES="${4:-100}"
EVAL_NUM="${5:-10}"

RUN_ID="${MCMIPL_RUN_ID:-$(bash -c "source ${MAIN}/scripts/mcmipl_data_slug.sh && data_slug ${DATASET}")_s${SEED}_$(date +%Y%m%d_%H%M%S)}"
export MCMIPL_RUN_ID="$RUN_ID"

echo "=== run+archive | ${DATASET} seed=${SEED} run_id=${RUN_ID} | $(date) ==="

bash "${MAIN}/scripts/mcmipl_record_run_timing.sh" start "$DATASET" "$SEED" "$RUN_ID"
bash "${MAIN}/run_mcmipl.sh" "$DATASET" "$SEED" "$MAX_STEPS" "$SAMPLE_TIMES" "$EVAL_NUM"
bash "${MAIN}/scripts/mcmipl_archive_seed.sh" "$DATASET" "$SEED"
bash "${MAIN}/scripts/mcmipl_record_run_timing.sh" end "$DATASET" "$SEED" "$RUN_ID"

echo "=== run+archive DONE | ${DATASET} seed=${SEED} run_id=${RUN_ID} | $(date) ==="
